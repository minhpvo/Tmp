#include "Geometry1.h"
#include "Geometry2.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::SoftLOneLoss;
using ceres::HuberLoss;
using ceres::Problem;
using ceres::Solver;

using namespace std;
using namespace cv;
using namespace Eigen;

//BA ulti
double PinholeReprojectionErrorSimpleDebug(double *P, Point3d Point, Point2d uv)
{
	double numX = P[0] * Point.x + P[1] * Point.y + P[2] * Point.z + P[3];
	double numY = P[4] * Point.x + P[5] * Point.y + P[6] * Point.z + P[7];
	double denum = P[8] * Point.x + P[9] * Point.y + P[10] * Point.z + P[11];

	double residual = sqrt(pow(numX / denum - uv.x, 2) + pow(numY / denum - uv.y, 2));
	return residual;
}
void FOVReprojectionDistortionDebug(double *intrinsic, double* distortion, double* rt, Point2d observed, Point3d Point, double *residuals)
{
	// camera[0,1,2] are the angle-axis rotation.
	double p[3];
	double point[3] = { Point.x, Point.y, Point.z };
	ceres::AngleAxisRotatePoint(rt, point, p);

	// camera[3,4,5] are the translation.
	p[0] += rt[3], p[1] += rt[4], p[2] += rt[5];

	// Project to image coordinate
	double xcn = p[0] / p[2], ycn = p[1] / p[2];
	double u = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3], v = intrinsic[1] * ycn + intrinsic[4];

	//Apply lens distortion
	double omega = distortion[0], DistCtr[2] = { distortion[1], distortion[2] };
	double x = u - DistCtr[0], y = v - DistCtr[1];
	double ru = sqrt(x*x + y * y), rd = atan(2.0*ru*tan(0.5*omega)) / omega;
	double t = rd / ru;
	double x_u = t * x, y_u = t * y;

	residuals[0] = x_u + DistCtr[0] - observed.x, residuals[1] = y_u + DistCtr[1] - observed.y;
	return;
}
void FOVReprojectionDistortion2Debug(double *intrinsic, double* distortion, double* rt, Point2d observed, Point3d Point, double *residuals)
{
	// camera[0,1,2] are the angle-axis rotation.
	double p[3];
	double point[3] = { Point.x, Point.y, Point.z };
	ceres::AngleAxisRotatePoint(rt, point, p);

	// camera[3,4,5] are the translation.
	p[0] += rt[3], p[1] += rt[4], p[2] += rt[5];

	// Project to image coordinate
	double xcn = p[0] / p[2], ycn = p[1] / p[2];

	//Apply lens distortion
	double omega = distortion[0];
	double ru = sqrt(xcn*xcn + ycn * ycn), rd = atan(2.0*ru*tan(0.5*omega)) / omega;
	double t = rd / ru;
	double x_u = t * xcn, y_u = t * ycn;
	double u_ = intrinsic[0] * x_u + intrinsic[2] * y_u + intrinsic[3], v_ = intrinsic[1] * y_u + intrinsic[4];

	residuals[0] = u_ - observed.x, residuals[1] = v_ - observed.y;
	return;
}
void PinholeReprojectionDebug(double *intrinsic, double* rt, Point2d &observed, Point3d Point, double *residuals)
{
	// camera[0,1,2] are the angle-axis rotation.
	double p[3];
	double point[3] = { Point.x, Point.y, Point.z };
	ceres::AngleAxisRotatePoint(rt, point, p);

	// camera[3,4,5] are the translation.
	p[0] += rt[3], p[1] += rt[4], p[2] += rt[5];

	// Project to image coordinate
	double xcn = p[0] / p[2], ycn = p[1] / p[2];
	double u = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3], v = intrinsic[1] * ycn + intrinsic[4];

	residuals[0] = u - observed.x, residuals[1] = v - observed.y;
	return;
}
void PinholeDistortionReprojectionDebug(double *intrinsic, double* distortion, double* rt, Point2d &observed, Point3d Point, double *residuals)
{
	// camera[0,1,2] are the angle-axis rotation.
	double p[3];
	double point[3] = { Point.x, Point.y, Point.z };
	ceres::AngleAxisRotatePoint(rt, point, p);

	// camera[3,4,5] are the translation.
	p[0] += rt[3], p[1] += rt[4], p[2] += rt[5];

	// Project to image coordinate
	double xcn = p[0] / p[2], ycn = p[1] / p[2];
	Point2d uv(intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3], intrinsic[1] * ycn + intrinsic[4]);

	// Deal with distortion
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };
	double distortionParas[7] = { distortion[0], distortion[1], distortion[2], distortion[3], distortion[4], distortion[5], distortion[6] };

	LensDistortionPoint(&uv, K, distortionParas);

	// The error is the difference between the predicted and observed position.
	residuals[0] = uv.x - observed.x, residuals[1] = uv.y - observed.y;

	return;
}
void PinholeDistortionReprojectionDebug2(double *fxfy, double *skew, double* u0v0, double *Radial12, double *Tangential12, double *Radial3, double *Prism, double* rt, Point2d &observed, double *point, double *residuals)
{
	// camera[0,1,2] are the angle-axis rotation.
	// camera[0,1,2] are the angle-axis rotation.
	double p[3];
	ceres::AngleAxisRotatePoint(rt, point, p);

	// camera[3,4,5] are the translation.
	p[0] += rt[3], p[1] += rt[4], p[2] += rt[5];

	// Project to normalize coordinate
	double  xcn = p[0] / p[2], ycn = p[1] / p[2];

	// Apply second and fourth order radial distortion.
	double xcn2 = xcn * xcn, ycn2 = ycn * ycn, xycn = xcn * ycn, r2 = xcn2 + ycn2, r4 = r2 * r2, r6 = r2 * r4;
	double radial = 1.0 + Radial12[0] * r2 + Radial12[1] * r4 + Radial3[0] * r6;
	double tangentialX = 2.0*Tangential12[1] * xycn + Tangential12[0] * (r2 + 2.0*xcn2);
	double tangentailY = Tangential12[1] * (r2 + 2.0*ycn2) + 2.0*Tangential12[0] * xycn;
	double prismX = Prism[0] * r2;
	double prismY = Prism[1] * r2;
	double xcn_ = radial * xcn + tangentialX + prismX;
	double ycn_ = radial * ycn + tangentailY + prismY;

	// Compute final projected point position.
	double predicted_x = fxfy[0] * xcn_ + skew[0] * ycn_ + u0v0[0];
	double predicted_y = fxfy[1] * ycn_ + u0v0[1];

	// The error is the difference between the predicted and observed position.
	residuals[0] = predicted_x - observed.x;
	residuals[1] = predicted_y - observed.y;

	return;
}

int CayleyProjection(double *intrinsic, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height)
{
	//Solving Eq. (5) of the p6p rolling shutter paper for the row location given all other parameters

	//transformed_X = R(v)*X
	double p[3] = { Point.x, Point.y, Point.z };
	double R_global[9];	getRfromr(rt, R_global);
	double Tx = rt[3], Ty = rt[4], Tz = rt[5];
	double tx = wt[3], ty = wt[4], tz = wt[5];
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };

	/*if (0)//abs(wt[0]) + abs(wt[1]) + abs(wt[2]) > 0.5 && (abs(tx) + abs(ty) + abs(tz) > 30))
	{
	double tp[3]; mat_mul(R_global, p, tp, 3, 3, 1);
	double X = tp[0], Y = tp[1], Z = tp[2];
	double wx = wt[0], wy = wt[1], wz = wt[2], wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;

	//Set up polynomial coefficients (obtained from matlab symbolic)
	double c[5];
	Mat coeffs(1, 5, CV_64F, c);
	c[4] = tz*wz2 + tz*wy2 + tz*wx2;
	c[3] = 2.0 * Y*wyz + 2.0 * X*wxz - ty*wz2 - ty*wy2 - ty*wx2 - Z*wy2 - Z*wx2 + Z*wz2 + Tz*wz2 + Tz*wy2 + Tz*wx2;
	c[2] = -2.0 * Z*wyz - 2.0 * X*wxy - Y*wy2 - Ty*wz2 - Ty*wy2 - Ty*wx2 + 2.0 * Y*wx - 2.0 * X*wy + Y*wz2 + Y*wx2 + tz;
	c[1] = 2.0 * Z*wx - 2.0 * X*wz - ty + Z + Tz;
	c[0] = -Y - Ty;

	std::vector<std::complex<double> > roots;
	solvePoly(coeffs, roots);

	int count = 0;
	for (int ii = 0; ii < roots.size(); ii++)
	{
	if (fabs(roots[ii].imag()) > 1e-10)
	continue;

	double j = roots[ii].real(), j2 = j*j, j3 = j2*j;
	double lamda = (Tz + Z + j*tz + Tz*j2 * wx2 + Tz*j2 * wy2 + Tz*j2 * wz2 - Z*j2 * wx2 - Z*j2 * wy2 + Z*j2 * wz2 + j3 * tz*wx2 + j3 * tz*wy2 + j3 * tz*wz2 - 2.0 * X*j*wy + 2.0 * Y*j*wx + 2.0 * X*j2 * wxz + 2.0 * Y*j2 * wyz) / (j2 * wx2 + j2 * wy2 + j2 * wz2 + 1.0);
	double naiveDepth = Z + Tz;
	if (abs((lamda - naiveDepth) / naiveDepth) > 0.1) //very different from the orginal depth
	continue;
	double i = (Tx + X + j*tx + Tx*j2 * wx2 + Tx*j2 * wy2 + Tx*j2 * wz2 + X*j2 * wx2 - X*j2 * wy2 - X*j2 * wz2 + j3 * tx*wx2 + j3 * tx*wy2 + j3 * tx*wz2 - 2.0 * Y*j*wz + 2.0 * Z*j*wy + 2.0 * Y*j2 * wxy + 2.0 * Z*j2 * wxz) / (Tz + Z + j*tz + Tz*j2 * wx2 + Tz*j2 * wy2 + Tz*j2 * wz2 - Z*j2 * wx2 - Z*j2 * wy2 + Z*j2 * wz2 + j3 * tz*wx2 + j3 * tz*wy2 + j3 * tz*wz2 - 2.0 * X*j*wy + 2.0 * Y*j*wx + 2.0 * X*j2 * wxz + 2.0 * Y*j2 * wyz);

	Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
	if (uv.x < 0 || uv.x > width - 1 || uv.y < 0 || uv.y > height - 1)
	continue;
	else
	{
	predicted = uv;
	count++;
	}
	}
	return count;
	}
	else*/
	{
		double wx, wy, wz, wx2, wy2, wz2, wxy, wxz, wyz, denum, Rw[9], R[9], tp[3];

		mat_mul(R_global, p, tp, 3, 3, 1);
		tp[1] += Ty, tp[2] += Tz;
		double j = tp[1] / tp[2], j_ = j;

		for (int iter = 0; iter < 10; iter++)
		{
			wx = j * wt[0], wy = j * wt[1], wz = j * wt[2];
			wx2 = wx * wx, wy2 = wy * wy, wz2 = wz * wz, wxz = wx * wz, wxy = wx * wy, wyz = wy * wz;

			denum = 1.0 + wx2 + wy2 + wz2;

			Rw[0] = 1.0 + wx2 - wy2 - wz2, Rw[1] = 2.0 * wxy - 2.0 * wz, Rw[2] = 2.0 * wy + 2.0 * wxz,
				Rw[3] = 2.0 * wz + 2.0 * wxy, Rw[4] = 1.0 - wx2 + wy2 - wz2, Rw[5] = 2.0 * wyz - 2.0 * wx,
				Rw[6] = 2.0 * wxz - 2.0 * wy, Rw[7] = 2.0 * wx + 2.0 * wyz, Rw[8] = 1.0 - wx2 - wy2 + wz2;

			for (int ii = 0; ii < 9; ii++)
				Rw[ii] = Rw[ii] / denum;

			mat_mul(Rw, R_global, R, 3, 3, 3);
			mat_mul(R, p, tp, 3, 3, 1);
			tp[0] += Tx, tp[1] += Ty, tp[2] += Tz;

			j = (tp[1] + j * ty) / (tp[2] + j * tz);
			if (abs((j - j_) / j_) < 1.0e-9)
				break;
			j_ = j;
		}
		double i = (tp[0] + j * tx) / (tp[2] + j * tz);

		Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
		predicted = uv;
		if (uv.x < 0 || uv.x > width - 1 || uv.y < 0 || uv.y > height - 1)
			return 0;
		else
			return 1;
	}
}
int CayleyReprojectionDebug(double *intrinsic, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals)
{
	Point2d predicted;
	int count = CayleyProjection(intrinsic, rt, wt, predicted, Point, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return count;
}
int CayleyDistortionProjection(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height)
{
	//Solving Eq. (5) of the p6p rolling shutter paper for the row location given all other parameters
	double p[3] = { Point.x, Point.y, Point.z };
	double R_global[9];	getRfromr(rt, R_global);
	double Tx = rt[3], Ty = rt[4], Tz = rt[5];
	double tx = wt[3], ty = wt[4], tz = wt[5];
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };

	/*if (abs(wt[0]) + abs(wt[1]) + abs(wt[2]) > 0.5 && (abs(tx) + abs(ty) + abs(tz) > 30))
	{
	//Polynomial solving approach. Not very stable if the rolling shutter is small
	double tp[3]; mat_mul(R_global, p, tp, 3, 3, 1);
	double X = tp[0], Y = tp[1], Z = tp[2];
	double wx = wt[0], wy = wt[1], wz = wt[2], wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;

	//Set up polynomial coefficients (obtained from matlab symbolic)
	double c[5];
	Mat coeffs(1, 5, CV_64F, c);
	c[4] = tz*wz2 + tz*wy2 + tz*wx2;
	c[3] = 2.0 * Y*wyz + 2.0 * X*wxz - ty*wz2 - ty*wy2 - ty*wx2 - Z*wy2 - Z*wx2 + Z*wz2 + Tz*wz2 + Tz*wy2 + Tz*wx2;
	c[2] = -2.0 * Z*wyz - 2.0 * X*wxy - Y*wy2 - Ty*wz2 - Ty*wy2 - Ty*wx2 + 2.0 * Y*wx - 2.0 * X*wy + Y*wz2 + Y*wx2 + tz;
	c[1] = 2.0 * Z*wx - 2.0 * X*wz - ty + Z + Tz;
	c[0] = -Y - Ty;

	std::vector<std::complex<double> > roots;
	solvePoly(coeffs, roots);

	int count = 0;
	for (int ii = 0; ii < roots.size(); ii++)
	{
	if (fabs(roots[ii].imag()) > 1e-10)
	continue;

	double j = roots[ii].real(), j2 = j*j, j3 = j2*j;
	double lamda = (Tz + Z + j*tz + Tz*j2 * wx2 + Tz*j2 * wy2 + Tz*j2 * wz2 - Z*j2 * wx2 - Z*j2 * wy2 + Z*j2 * wz2 + j3 * tz*wx2 + j3 * tz*wy2 + j3 * tz*wz2 - 2.0 * X*j*wy + 2.0 * Y*j*wx + 2.0 * X*j2 * wxz + 2.0 * Y*j2 * wyz) / (j2 * wx2 + j2 * wy2 + j2 * wz2 + 1.0);
	double naiveDepth = Z + Tz;
	if (abs((lamda - naiveDepth) / naiveDepth) > 0.1) //very different from the orginal depth
	continue;
	double i = (Tx + X + j*tx + Tx*j2 * wx2 + Tx*j2 * wy2 + Tx*j2 * wz2 + X*j2 * wx2 - X*j2 * wy2 - X*j2 * wz2 + j3 * tx*wx2 + j3 * tx*wy2 + j3 * tx*wz2 - 2.0 * Y*j*wz + 2.0 * Z*j*wy + 2.0 * Y*j2 * wxy + 2.0 * Z*j2 * wxz) / (Tz + Z + j*tz + Tz*j2 * wx2 + Tz*j2 * wy2 + Tz*j2 * wz2 - Z*j2 * wx2 - Z*j2 * wy2 + Z*j2 * wz2 + j3 * tz*wx2 + j3 * tz*wy2 + j3 * tz*wz2 - 2.0 * X*j*wy + 2.0 * Y*j*wx + 2.0 * X*j2 * wxz + 2.0 * Y*j2 * wyz);

	Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
	LensDistortionPoint(&uv, K, distortion);
	if (uv.x < 0 || uv.x > width - 1 || uv.y < 0 || uv.y > height - 1)
	continue;
	else
	{
	predicted = uv;
	count++;
	}
	}
	return count;
	}
	else*/
	{
		//Fix point iteration approach. Very stable
		double wx, wy, wz, wx2, wy2, wz2, wxy, wxz, wyz, denum, Rw[9], R[9], tp[3];

		mat_mul(R_global, p, tp, 3, 3, 1);
		tp[1] += Ty, tp[2] += Tz;
		double j = tp[1] / tp[2], j_ = j;

		for (int iter = 0; iter < 10; iter++)
		{
			wx = j * wt[0], wy = j * wt[1], wz = j * wt[2];
			wx2 = wx * wx, wy2 = wy * wy, wz2 = wz * wz, wxz = wx * wz, wxy = wx * wy, wyz = wy * wz;

			denum = 1.0 + wx2 + wy2 + wz2;

			Rw[0] = 1.0 + wx2 - wy2 - wz2, Rw[1] = 2.0 * wxy - 2.0 * wz, Rw[2] = 2.0 * wy + 2.0 * wxz,
				Rw[3] = 2.0 * wz + 2.0 * wxy, Rw[4] = 1.0 - wx2 + wy2 - wz2, Rw[5] = 2.0 * wyz - 2.0 * wx,
				Rw[6] = 2.0 * wxz - 2.0 * wy, Rw[7] = 2.0 * wx + 2.0 * wyz, Rw[8] = 1.0 - wx2 - wy2 + wz2;

			for (int ii = 0; ii < 9; ii++)
				Rw[ii] = Rw[ii] / denum;

			mat_mul(Rw, R_global, R, 3, 3, 3);
			mat_mul(R, p, tp, 3, 3, 1);
			tp[0] += Tx, tp[1] += Ty, tp[2] += Tz;

			j = (tp[1] + j * ty) / (tp[2] + j * tz);
			if (abs((j - j_) / j_) < 1.0e-9)
				break;
			j_ = j;
		}
		double i = (tp[0] + j * tx) / (tp[2] + j * tz);

		Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
		LensDistortionPoint(&uv, K, distortion);
		predicted = uv;
		if (uv.x < 0 || uv.x > width - 1 || uv.y < 0 || uv.y > height - 1)
			return 0;
		else
			return 1;
	}
}
int CayleyDistortionReprojectionDebug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals)
{
	Point2d predicted;
	int count = CayleyDistortionProjection(intrinsic, distortion, rt, wt, predicted, Point, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return count;
}
int CayleyFOVProjection(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height)
{
	//Solving Eq. (5) of the p6p rolling shutter paper for the row location given all other parameters
	double p[3] = { Point.x, Point.y, Point.z };
	double R_global[9];	getRfromr(rt, R_global);
	double Tx = rt[3], Ty = rt[4], Tz = rt[5];
	double tx = wt[3], ty = wt[4], tz = wt[5];
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };

	//Fix point iteration approach. Very stable
	double wx, wy, wz, wx2, wy2, wz2, wxy, wxz, wyz, denum, Rw[9], R[9], tp[3];

	mat_mul(R_global, p, tp, 3, 3, 1);
	tp[1] += Ty, tp[2] += Tz;
	double j = tp[1] / tp[2], j_ = j;

	for (int iter = 0; iter < 10; iter++)
	{
		wx = j * wt[0], wy = j * wt[1], wz = j * wt[2];
		wx2 = wx * wx, wy2 = wy * wy, wz2 = wz * wz, wxz = wx * wz, wxy = wx * wy, wyz = wy * wz;

		denum = 1.0 + wx2 + wy2 + wz2;

		Rw[0] = 1.0 + wx2 - wy2 - wz2, Rw[1] = 2.0 * wxy - 2.0 * wz, Rw[2] = 2.0 * wy + 2.0 * wxz,
			Rw[3] = 2.0 * wz + 2.0 * wxy, Rw[4] = 1.0 - wx2 + wy2 - wz2, Rw[5] = 2.0 * wyz - 2.0 * wx,
			Rw[6] = 2.0 * wxz - 2.0 * wy, Rw[7] = 2.0 * wx + 2.0 * wyz, Rw[8] = 1.0 - wx2 - wy2 + wz2;

		for (int ii = 0; ii < 9; ii++)
			Rw[ii] = Rw[ii] / denum;

		mat_mul(Rw, R_global, R, 3, 3, 3);
		mat_mul(R, p, tp, 3, 3, 1);
		tp[0] += Tx, tp[1] += Ty, tp[2] += Tz;

		j = (tp[1] + j * ty) / (tp[2] + j * tz);
		if (abs((j - j_) / j_) < 1.0e-9)
			break;
		j_ = j;
	}
	double i = (tp[0] + j * tx) / (tp[2] + j * tz);

	Point2d uv(intrinsic[0] * i + intrinsic[2] * j + intrinsic[3], intrinsic[1] * j + intrinsic[4]);
	FishEyeDistortionPoint(&uv, distortion[0], distortion[1], distortion[2]);
	predicted = uv;
	return 1;
}
int CayleyFOVReprojectionDebug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals)
{
	Point2d predicted;
	int count = CayleyFOVProjection(intrinsic, distortion, rt, wt, predicted, Point, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return count;
}
int CayleyFOVProjection2(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height)
{
	//Solving Eq. (5) of the p6p rolling shutter paper for the row location given all other parameters
	double p[3] = { Point.x, Point.y, Point.z };
	double R_global[9];	getRfromr(rt, R_global);
	double Tx = rt[3], Ty = rt[4], Tz = rt[5];
	double tx = wt[3], ty = wt[4], tz = wt[5];
	double K[9] = { intrinsic[0], intrinsic[2], intrinsic[3], 0.0, intrinsic[1], intrinsic[4], 0.0, 0.0, 1.0 };

	//Fix point iteration approach. Very stable
	double wx, wy, wz, wx2, wy2, wz2, wxy, wxz, wyz, denum, Rw[9], R[9], tp[3];

	mat_mul(R_global, p, tp, 3, 3, 1);
	tp[1] += Ty, tp[2] += Tz;
	double j = tp[1] / tp[2], j_ = j;

	for (int iter = 0; iter < 10; iter++)
	{
		wx = j * wt[0], wy = j * wt[1], wz = j * wt[2];
		wx2 = wx * wx, wy2 = wy * wy, wz2 = wz * wz, wxz = wx * wz, wxy = wx * wy, wyz = wy * wz;

		denum = 1.0 + wx2 + wy2 + wz2;

		Rw[0] = 1.0 + wx2 - wy2 - wz2, Rw[1] = 2.0 * wxy - 2.0 * wz, Rw[2] = 2.0 * wy + 2.0 * wxz,
			Rw[3] = 2.0 * wz + 2.0 * wxy, Rw[4] = 1.0 - wx2 + wy2 - wz2, Rw[5] = 2.0 * wyz - 2.0 * wx,
			Rw[6] = 2.0 * wxz - 2.0 * wy, Rw[7] = 2.0 * wx + 2.0 * wyz, Rw[8] = 1.0 - wx2 - wy2 + wz2;

		for (int ii = 0; ii < 9; ii++)
			Rw[ii] = Rw[ii] / denum;

		mat_mul(Rw, R_global, R, 3, 3, 3);
		mat_mul(R, p, tp, 3, 3, 1);
		tp[0] += Tx, tp[1] += Ty, tp[2] += Tz;

		j = (tp[1] + j * ty) / (tp[2] + j * tz);
		if (abs((j - j_) / j_) < 1.0e-9)
			break;
		j_ = j;
	}
	double i = (tp[0] + j * tx) / (tp[2] + j * tz);

	double omega = distortion[0];
	double ru = sqrt(i*i + j * j), rd = atan(2.0*ru*tan(0.5*omega)) / omega;
	double t = rd / ru;
	double x_u = t * i, y_u = t * j;
	predicted.x = intrinsic[0] * x_u + intrinsic[2] * y_u + intrinsic[3];
	predicted.y = intrinsic[1] * y_u + intrinsic[4];

	return 1;
}
int CayleyFOVReprojection2Debug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals)
{
	Point2d predicted;
	int count = CayleyFOVProjection2(intrinsic, distortion, rt, wt, predicted, Point, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return count;
}
//Triangulation BA
void NviewTriangulationNonLinear(double *P, Point2d *Point2D, Point3d *Point3D, double *ReprojectionError, int nviews, int npts)
{
	ceres::Problem problem;

	//printLOG("Error before: \n");
	double *p3d = new double[3 * npts];
	for (int ii = 0; ii < npts; ii++)
	{
		ReprojectionError[ii] = 0.0;
		p3d[3 * ii] = Point3D[ii].x, p3d[3 * ii + 1] = Point3D[ii].y, p3d[3 * ii + 2] = Point3D[ii].z;
		for (int jj = 0; jj < nviews; jj++)
		{
			//ReprojectionError[ii] += PinholeReprojectionErrorSimpleDebug(P + 12 * jj, Point3d(Point3D[3 * ii], Point3D[3 * ii + 1], Point3D[3 * ii + 2]), Point2d(Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1]));
			ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(P + 12 * jj, Point2D[(ii*nviews + jj)].x, Point2D[(ii*nviews + jj)].y, 1.0);
			problem.AddResidualBlock(cost_function, NULL, &p3d[ii]);
		}
		//ReprojectionError[ii] /= nviews;
		//printLOG("%f ", ReprojectionError[ii]);
	}
	//printLOG("\n");

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.FullReport() << "\n";

	//printLOG("Error after: \n");
	for (int ii = 0; ii < npts; ii++)
	{
		ReprojectionError[ii] = 0.0;
		Point3D[ii].x = p3d[3 * ii], Point3D[ii].y = p3d[3 * ii + 1], Point3D[ii].z = p3d[3 * ii + 2];
		for (int jj = 0; jj < nviews; jj++)
			ReprojectionError[ii] += PinholeReprojectionErrorSimpleDebug(P + 12 * jj, Point3D[ii], Point2D[(ii*nviews + jj)]);
		ReprojectionError[ii] /= nviews;
		//printLOG("%f ", ReprojectionError[ii]);
	}
	//printLOG("\n");

	return;
}
void NviewTriangulationNonLinear(double *P, double *Point2D, double *Point3D, double *ReprojectionError, int nviews, int npts)
{
	ceres::Problem problem;

	//printLOG("Error before: \n");
	for (int ii = 0; ii < npts; ii++)
	{
		ReprojectionError[ii] = 0.0;
		for (int jj = 0; jj < nviews; jj++)
		{
			//ReprojectionError[ii] += PinholeReprojectionErrorSimpleDebug(P + 12 * jj, Point3d(Point3D[3 * ii], Point3D[3 * ii + 1], Point3D[3 * ii + 2]), Point2d(Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1]));
			ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(P + 12 * jj, Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1], 1.0);
			problem.AddResidualBlock(cost_function, NULL, &Point3D[3 * ii]);
		}
		//ReprojectionError[ii] /= nviews;
		//printLOG("%f ", ReprojectionError[ii]);
	}
	//printLOG("\n");

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.FullReport() << "\n";

	//printLOG("Error after: \n");
	for (int ii = 0; ii < npts; ii++)
	{
		ReprojectionError[ii] = 0.0;
		for (int jj = 0; jj < nviews; jj++)
			ReprojectionError[ii] += PinholeReprojectionErrorSimpleDebug(P + 12 * jj, Point3d(Point3D[3 * ii], Point3D[3 * ii + 1], Point3D[3 * ii + 2]), Point2d(Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1]));
		ReprojectionError[ii] /= nviews;
		//printLOG("%f ", ReprojectionError[ii]);
	}
	//printLOG("\n");

	return;
}
void NviewTriangulationNonLinearCayley(CameraData *camInfo, double *Point2D, double *Point3D, double *ReprojectionError, int nviews, int npts)
{
	ceres::Problem problem;

	//printLOG("Error before: \n");

	double residuals[2];
	for (int ii = 0; ii < npts; ii++)
	{
		ReprojectionError[ii] = 0.0;
		for (int jj = 0; jj < nviews; jj++)
		{
			if (!camInfo[jj].valid)
				continue;

			Point2d uv = Point2d(Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1]);
			CayleyReprojectionDebug(camInfo[jj].intrinsic, camInfo[jj].rt, camInfo[jj].wt, uv, Point3d(Point3D[3 * ii], Point3D[3 * ii + 1], Point3D[3 * ii + 2]), camInfo[jj].width, camInfo[jj].height, residuals);
			ReprojectionError[ii] += residuals[0] * residuals[0] + residuals[1] * residuals[1];

			ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camInfo[jj].intrinsic, Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1], 1.0, camInfo[jj].width, camInfo[jj].height);
			problem.AddResidualBlock(cost_function, NULL, camInfo[jj].rt, camInfo[jj].wt, &Point3D[ii]);

			problem.SetParameterBlockConstant(camInfo[jj].rt);
			problem.SetParameterBlockConstant(camInfo[jj].wt);
		}

		ReprojectionError[ii] /= nviews;
		//printLOG("%f ", ReprojectionError[ii]);
	}
	//printLOG("\n");


	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.FullReport() << "\n";

	//printLOG("Error after: \n");
	for (int ii = 0; ii < npts; ii++)
	{
		ReprojectionError[ii] = 0.0;
		for (int jj = 0; jj < nviews; jj++)
		{
			if (!camInfo[jj].valid)
				continue;
			Point2d uv = Point2d(Point2D[2 * (ii*nviews + jj)], Point2D[2 * (ii*nviews + jj) + 1]);
			CayleyReprojectionDebug(camInfo[jj].intrinsic, camInfo[jj].rt, camInfo[jj].wt, uv, Point3d(Point3D[3 * ii], Point3D[3 * ii + 1], Point3D[3 * ii + 2]), camInfo[jj].width, camInfo[jj].height, residuals);
			ReprojectionError[ii] += residuals[0] * residuals[0] + residuals[1] * residuals[1];
		}
		ReprojectionError[ii] /= nviews;
		//printLOG("%f ", ReprojectionError[ii]);
	}
	//printLOG("\n");

	return;
}
double CalibratedTwoViewBA(double *intrinsic1, double *intrinsic2, double *rt1, double *rt2, vector<Point2d> &pts1, vector<Point2d> &pts2, vector<Point3d> &P3D, vector<int> &validPid, double threshold, int LossType, int verbose)
{
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(threshold);

	int nBadCounts = 0, npts = (int)pts1.size();
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);
	double residuals1[2], residuals2[2], maxOutlierX = 0.0, maxOutlierY = 0.0;

	double P1[12], P2[12], R1[9], R2[9];
	double K1[9] = { intrinsic1[0], intrinsic1[2], intrinsic1[3], 0, intrinsic1[1], intrinsic1[4], 0, 0, 1 };
	double K2[9] = { intrinsic2[0], intrinsic2[2], intrinsic2[3], 0, intrinsic2[1], intrinsic2[4], 0, 0, 1 };
	getRfromr(rt1, R1); getRfromr(rt2, R2);
	AssembleP(K1, R1, rt1 + 3, P1), AssembleP(K2, R2, rt2 + 3, P2);
	TwoViewTriangulation(pts1, pts2, P1, P2, P3D);

	double *xyz = new double[3 * npts];
	for (int ii = 0; ii < npts; ii++)
		xyz[3 * ii] = P3D[ii].x, xyz[3 * ii + 1] = P3D[ii].y, xyz[3 * ii + 2] = P3D[ii].z;


	for (int ii = 0; ii < npts; ii++)
	{
		int count = 0;
		PinholeReprojectionDebug(intrinsic1, rt1, pts1[ii], P3D[ii], residuals1);
		if (abs(residuals1[0]) > threshold || abs(residuals1[1]) > threshold)
		{
			if (abs(residuals1[0]) > maxOutlierX)
				maxOutlierX = residuals1[0];
			if (abs(residuals1[1]) > maxOutlierY)
				maxOutlierY = residuals1[1];
		}
		else
			count++;

		PinholeReprojectionDebug(intrinsic2, rt2, pts2[ii], P3D[ii], residuals2);
		if (abs(residuals2[0]) > threshold || abs(residuals2[1]) > threshold)
		{
			if (abs(residuals2[0]) > maxOutlierX)
				maxOutlierX = residuals2[0];
			if (abs(residuals2[1]) > maxOutlierY)
				maxOutlierY = residuals2[1];
		}
		else
			count++;

		if (count < 2)
		{
			P3D[ii].x = 0, P3D[ii].y = 0, P3D[ii].z = 0; //deactivate
			nBadCounts++;
			continue;
		}
		validPid.push_back(ii);

		ceres::CostFunction* cost_function1 = PinholeReprojectionError::Create(pts1[ii].x, pts1[ii].y, 1.0);
		problem.AddResidualBlock(cost_function1, loss_funcion, intrinsic1, rt1, &xyz[3 * ii]);

		ceres::CostFunction* cost_function2 = PinholeReprojectionError::Create(pts2[ii].x, pts2[ii].y, 1.0);
		problem.AddResidualBlock(cost_function2, loss_funcion, intrinsic2, rt2, &xyz[3 * ii]);

		problem.SetParameterBlockConstant(intrinsic1);
		problem.SetParameterBlockConstant(intrinsic2);
		problem.SetParameterBlockConstant(rt1);

		ReProjectionErrorX.push_back(residuals1[0]), ReProjectionErrorX.push_back(residuals2[0]);
		ReProjectionErrorY.push_back(residuals1[1]), ReProjectionErrorY.push_back(residuals2[1]);
	}

	double miniX, maxiX, avgX, stdX, miniY, maxiY, avgY, stdY;
	if (ReProjectionErrorX.size() > 1)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		if (verbose == 1)
		{
			printLOG("(%d/%d) bad points with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, npts, maxOutlierX, maxOutlierY);
			printLOG("Reprojection error before BA: Min: (%.2e, %.2e) Max: (%.2e,%.2e) Mean: (%.2e,%.2e) Std: (%.2e,%.2e)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
		}
	}

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (verbose)
		std::cout << summary.BriefReport() << "\n";


	if (ReProjectionErrorX.size() > 1)
	{
		maxOutlierX = 0, maxOutlierY = 0;
		ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
		for (int jj = 0; jj < (int)validPid.size(); jj++)
		{
			int ii = validPid[jj];

			PinholeReprojectionDebug(intrinsic1, rt1, pts1[ii], P3D[ii], residuals1);
			if (abs(residuals1[0]) > threshold || abs(residuals1[1]) > threshold)
			{
				if (abs(residuals1[0]) > maxOutlierX)
					maxOutlierX = residuals1[0];
				if (abs(residuals1[1]) > maxOutlierY)
					maxOutlierY = residuals1[1];
			}

			PinholeReprojectionDebug(intrinsic2, rt2, pts2[ii], P3D[ii], residuals2);
			if (abs(residuals2[0]) > threshold || abs(residuals2[1]) > threshold)
			{
				if (abs(residuals2[0]) > maxOutlierX)
					maxOutlierX = residuals2[0];
				if (abs(residuals2[1]) > maxOutlierY)
					maxOutlierY = residuals2[1];
			}

			ReProjectionErrorX.push_back(residuals1[0]), ReProjectionErrorX.push_back(residuals2[0]);
			ReProjectionErrorY.push_back(residuals1[1]), ReProjectionErrorY.push_back(residuals2[1]);
		}

		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		if (verbose == 1)
		{
			printLOG("(%d/%d) bad points with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, npts, maxOutlierX, maxOutlierY);
			printLOG("Reprojection error after BA: Min: (%.2e, %.2e) Max: (%.2e,%.2e) Mean: (%.2e,%.2e) Std: (%.2e,%.2e)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
		}
	}

	delete[]xyz;
	double RMSE;
	if (ReProjectionErrorX.size() > 1)
		RMSE = sqrt(avgX*avgX + avgY * avgY);
	else
		RMSE = summary.final_cost;
	return RMSE;
}

//Camera localization
int MatchCameraToCorpus(char *Path, Corpus &CorpusInfo, CameraData &CamInfoI, int cameraID, int timeID, int distortionCorrected, vector<int> &CorpusViewToMatch, const float nndrRatio, const int ninlierThresh)
{
	//Load image and extract features
	const int descriptorSize = SIFTBINS;

	char Fname[512];
	double start = omp_get_wtime();

	sprintf(Fname, "%s/%d/PnP/_PInliers_%.4d.txt", Path, cameraID, timeID);
	if (IsFileExist(Fname) == 1)
	{
		printLOG("%s computed\n", Fname);
		return 0;
	}
	sprintf(Fname, "%s/%d/PnP/PInliers_%.4d.txt", Path, cameraID, timeID);
	if (IsFileExist(Fname) == 1)
	{
		printLOG("%s computed\n", Fname);
		return 0;
	}

	bool readsucces = false;
	/*vector<KeyPoint> keypoints1; keypoints1.reserve(MaxNFeatures);
	if (timeID < 0)
	sprintf(Fname, "%s/%.4d.kpts", Path, timeID);
	else
	sprintf(Fname, "%s/%d/%.4d.kpts", Path, cameraID, timeID);
	readsucces = ReadKPointsBinarySIFT(Fname, keypoints1);
	if (!readsucces)
	{
	printLOG("%s does not have SIFT points. Please precompute it!\n", Fname);
	return 1;
	}

	if (timeID < 0)
	sprintf(Fname, "%s%.4d.desc", Path, cameraID);
	else
	sprintf(Fname, "%s/%d/%.4d.desc", Path, cameraID, timeID);
	Mat descriptors1 = ReadDescriptorBinarySIFT(Fname);
	if (descriptors1.rows == 1)
	{
	printLOG("%s does not have SIFT points. Please precompute it!\n", Fname);
	return 1;
	}*/

	vector<KeyPoint> keypoints1; Mat descriptors1;
	if (timeID < 0)
		sprintf(Fname, "%s%.4d.sift", Path, cameraID);
	else
		sprintf(Fname, "%s/%d/%.4d.sift", Path, cameraID, timeID);
	if (readVisualSFMSiftGPU(Fname, keypoints1, descriptors1) == 1)
		return 0;

	//remove distortion if not removed before in case camera is calibrated
	if (distortionCorrected == 0 && !CamInfoI.notCalibrated)
	{
		Point2d pt;
		if (CamInfoI.LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			for (int ii = 0; ii < keypoints1.size(); ii++)
			{
				pt.x = keypoints1[ii].pt.x, pt.y = keypoints1[ii].pt.y;
				LensCorrectionPoint(&pt, CamInfoI.K, CamInfoI.distortion);
				keypoints1[ii].pt.x = pt.x, keypoints1[ii].pt.y = pt.y;
			}
		}
		else if (CamInfoI.LensModel == FISHEYE)
		{
			for (int ii = 0; ii < keypoints1.size(); ii++)
			{
				pt.x = keypoints1[ii].pt.x, pt.y = keypoints1[ii].pt.y;
				//FishEyeCorrectionPoint(&pt, CamInfoI.distortion[0], CamInfoI.distortion[1], CamInfoI.distortion[2]);
				FishEyeCorrectionPoint(&pt, CamInfoI.K, CamInfoI.distortion[0]);
				keypoints1[ii].pt.x = pt.x, keypoints1[ii].pt.y = pt.y;
			}
		}
	}

	//USAC config
	bool USEPROSAC = false, USESPRT = true, USELOSAC = true;
	ConfigParamsFund cfg;
	cfg.common.confThreshold = 0.99, cfg.common.minSampleSize = 7, cfg.common.inlierThreshold = 3.0;
	cfg.common.maxHypotheses = 850000, cfg.common.maxSolutionsPerSample = 3;
	cfg.common.prevalidateSample = true, cfg.common.prevalidateModel = true, cfg.common.testDegeneracy = true;
	cfg.common.randomSamplingMethod = USACConfig::SAMP_UNIFORM, cfg.common.verifMethod = USACConfig::VERIF_SPRT, cfg.common.localOptMethod = USACConfig::LO_LOSAC;

	if (USEPROSAC)
		cfg.prosac.maxSamples, cfg.prosac.beta, cfg.prosac.nonRandConf, cfg.prosac.minStopLen;
	if (USESPRT)
		cfg.sprt.tM = 200.0, cfg.sprt.mS = 2.38, cfg.sprt.delta = 0.05, cfg.sprt.epsilon = 0.15;
	if (USELOSAC)
		cfg.losac.innerSampleSize = 15, cfg.losac.innerRansacRepetitions = 5, cfg.losac.thresholdMultiplier = 2.0, cfg.losac.numStepsIterative = 4;

	if (distortionCorrected == 0 && CamInfoI.notCalibrated == true) // allow for more error if the image is not corrected and distortion parameters are unknown
		cfg.common.inlierThreshold *= 1.5;

	//Match extracted features with Corpus
	bool useBFMatcher = true;
	const int knn = 2, ntrees = 4, maxLeafCheck = 128;

	vector<float>Scale; Scale.reserve(5000);
	vector<Point2f> twoD; twoD.reserve(5000);
	vector<int> threeDiD; threeDiD.reserve(5000);
	vector<int>viewID; viewID.reserve(5000);
	vector<int>twoDiD; twoDiD.reserve(5000);

	//Finding nearest neighbor
	vector < int>twoDiD1;
	vector<float>Scale1;
	vector<Point2d>key1, key2;
	vector<int>CorrespondencesID;
	double Fmat[9];
	vector<int>cur3Ds, Inliers;
	key1.reserve(5000), key2.reserve(5000);
	Scale1.reserve(5000); twoDiD1.reserve(5000);
	CorrespondencesID.reserve(5000), cur3Ds.reserve(5000), Inliers.reserve(5000);

	bool ShowCorrespondence = false;
	for (int ii = 0; ii < CorpusViewToMatch.size(); ii++)
	{
		printLOG("%d ", CorpusViewToMatch[ii]);
		twoDiD1.clear(), key1.clear(), key2.clear();
		cur3Ds.clear(), Inliers.clear(), CorrespondencesID.clear();

		int camera2ID = CorpusViewToMatch[ii];
		int nsiftInCorpusView = (int)CorpusInfo.DescAllViews[camera2ID].size();
		Mat descriptors2(nsiftInCorpusView, SIFTBINS, CV_8U);
		for (int jj = 0; jj < nsiftInCorpusView; jj++)
		{
			uchar *ptr = descriptors2.ptr<uchar>(jj);
			for (int kk = 0; kk < SIFTBINS; kk++)
				ptr[kk] = CorpusInfo.DescAllViews[camera2ID][jj].desc[kk];
		}

		double start = omp_get_wtime();
		int count = 0;
		Mat indices, dists;
		if (useBFMatcher)
		{
			vector<Point2i> matches;
			matches = MatchTwoViewSIFTBruteForce(descriptors1, descriptors2);

			for (unsigned int i = 0; i < matches.size(); ++i)
			{
				int ind1 = matches[i].x;
				twoDiD1.push_back(ind1);
				key1.push_back(Point2d(keypoints1[ind1].pt.x, keypoints1[ind1].pt.y));
				Scale1.push_back(keypoints1[ind1].size);

				int ind2 = matches[i].y;
				int cur3Did = CorpusInfo.threeDIdAllViews[camera2ID][ind2];
				cur3Ds.push_back(cur3Did);
				key2.push_back(CorpusInfo.uvAllViews[camera2ID][ind2]);
			}
		}
		else //flann does not support uchar descriptor
		{
			vector<vector<DMatch> > matches;
			cv::flann::Index flannIndex(descriptors1, cv::flann::KDTreeIndexParams(ntrees));//, cvflann::FLANN_DIST_EUCLIDEAN);
			flannIndex.knnSearch(descriptors2, indices, dists, knn, cv::flann::SearchParams(maxLeafCheck));//Search in desc1 for every desc in 2

			for (int i = 0; i < descriptors2.rows; ++i)
			{
				int ind1 = indices.at<int>(i, 0);
				if (indices.at<int>(i, 0) >= 0 && indices.at<int>(i, 1) >= 0 && dists.at<float>(i, 0) <= nndrRatio * dists.at<float>(i, 1))
				{
					int cur3Did = CorpusInfo.threeDIdAllViews[camera2ID][i];
					cur3Ds.push_back(cur3Did);

					twoDiD1.push_back(ind1);
					key1.push_back(Point2d(keypoints1[ind1].pt.x, keypoints1[ind1].pt.y));
					Scale1.push_back(keypoints1[ind1].size);
					key2.push_back(CorpusInfo.uvAllViews[camera2ID][i]);
				}
			}
		}

		///****NOTE: 2d points in Corpus are corrected***///
		int ninliers = 0;
		if (key1.size() < ninlierThresh || key2.size() < ninlierThresh)
			continue;

		cfg.common.numDataPoints = (int)key1.size();
		USAC_FindFundamentalMatrix(cfg, key1, key2, Fmat, Inliers, ninliers);

		if (ninliers < ninlierThresh)
		{
#pragma omp critical
			printLOG("(%d, %d) to Corpus %d: failed Fundamental matrix test\n\n", cameraID, timeID, camera2ID);
			continue;
		}

		//Add matches to 2d-3d list
		for (int jj = 0; jj < (int)Inliers.size(); jj++)
		{
			if (Inliers[jj] == 1)
			{
				int cur3Did = cur3Ds[jj];
				bool used = false;
				for (int kk = 0; kk < threeDiD.size(); kk++)
				{
					if (cur3Did == threeDiD[kk])
					{
						used = true;
						break;
					}
				}
				if (used)
					continue;

				twoD.push_back(Point2f(key1[jj].x, key1[jj].y));
				twoDiD.push_back(twoDiD1[jj]);
				Scale.push_back(Scale1[jj]);
				threeDiD.push_back(cur3Did);
				viewID.push_back(camera2ID);
				count++;
			}
		}

		if (ShowCorrespondence)
		{
			int nchannels = 3;
			sprintf(Fname, "%s/%d/%.4d.jpg", Path, cameraID, timeID);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/%d/%.4d.png", Path, cameraID, timeID);
				if (IsFileExist(Fname) == 0)
				{
					printLOG("Cannot load %s\n", Fname);
					return 1;
				}
			}
			Mat Img1 = imread(Fname, nchannels == 3 ? 1 : 0);

			sprintf(Fname, "%s/Corpus/%.4d.jpg", Path, camera2ID);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/Corpus/%.4d.png", Path, camera2ID);
				if (IsFileExist(Fname) == 0)
				{
					printLOG("Cannot load %s\n", Fname);
					return 1;
				}
			}
			Mat Img2 = imread(Fname, nchannels == 3 ? 1 : 0);

			vector<Point2d> _key1, _key2;
			if (distortionCorrected == 0)
			{
				_key1 = key1; _key2 = key2;
				Point2d pt;
				if (CamInfoI.LensModel == RADIAL_TANGENTIAL_PRISM && CamInfoI.notCalibrated == false)
					LensDistortionPoint(_key1, CamInfoI.K, CamInfoI.distortion);
				else  if (CamInfoI.LensModel == FISHEYE && CamInfoI.notCalibrated == false)
				{
					for (int ii = 0; ii < key1.size(); ii++)
						FishEyeDistortionPoint(&_key1[ii], CamInfoI.distortion[0], CamInfoI.distortion[1], CamInfoI.distortion[2]);
				}

				if (CorpusInfo.camera[camera2ID].LensModel == RADIAL_TANGENTIAL_PRISM && CorpusInfo.camera[camera2ID].notCalibrated == false)
					LensDistortionPoint(_key2, CorpusInfo.camera[camera2ID].K, CorpusInfo.camera[camera2ID].distortion);
				else if (CorpusInfo.camera[camera2ID].LensModel == FISHEYE && CorpusInfo.camera[camera2ID].notCalibrated == false)
				{
					for (int ii = 0; ii < key2.size(); ii++)
						FishEyeDistortionPoint(&_key2[ii], CorpusInfo.camera[camera2ID].distortion[0], CorpusInfo.camera[camera2ID].distortion[1], CorpusInfo.camera[camera2ID].distortion[2]);
				}
			}

			CorrespondencesID.clear();
			for (int ii = 0; ii < key1.size(); ii++)
				if (Inliers[ii] == 1)
					CorrespondencesID.push_back(ii), CorrespondencesID.push_back(ii);

			cv::Mat correspond(max(Img1.rows, Img2.rows), Img1.cols + Img2.cols, CV_8UC3);
			cv::Rect rect1(0, 0, Img1.cols, Img1.rows);
			cv::Rect rect2(Img1.cols, 0, Img1.cols, Img1.rows);

			Img1.copyTo(correspond(rect1));
			Img2.copyTo(correspond(rect2));

			if (distortionCorrected == 0)
				DisplayImageCorrespondence(correspond, Img1.cols, 0, _key1, _key2, CorrespondencesID, 1.0);
			else
				DisplayImageCorrespondence(correspond, Img1.cols, 0, key1, key2, CorrespondencesID, 1.0);
		}
#pragma omp critical
		printLOG("(%d, %d) to Corpus %d: %d 3+ points in %.2fs.\n", cameraID, timeID, camera2ID, count, omp_get_wtime() - start);
	}

	//sprintf(Fname, "%s/%d/PnP/PInliers_%.4d.txt", Path, cameraID, timeID); FILE *fp = fopen(Fname, "w+");
	//fprintf(fp, "%d\n", threeDiD.size());
	//for (int jj = 0; jj < threeDiD.size(); jj++)
	//	fprintf(fp, "%d %d %.16f %.16f %.3f\n", threeDiD[jj], twoDiD[jj], twoD[jj].x, twoD[jj].y, Scale[jj]);
	//fclose(fp);

	sprintf(Fname, "%s/%d/PnP/_PInliers_%.4d.txt", Path, cameraID, timeID); FILE *fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", threeDiD.size());
	for (int jj = 0; jj < threeDiD.size(); jj++)
		fprintf(fp, "%d %.6e %.6e  %.6e %d %.6f %.6f %.3f\n", threeDiD[jj], CorpusInfo.xyz[threeDiD[jj]].x, CorpusInfo.xyz[threeDiD[jj]].y, CorpusInfo.xyz[threeDiD[jj]].z, twoDiD[jj], twoD[jj].x, twoD[jj].y, Scale[jj]);
	fclose(fp);

	/*sprintf(Fname, "%s/%d/_3D2D_%.4d.txt", Path, cameraID, timeID); fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", threeDiD.size());
	for (int jj = 0; jj < threeDiD.size(); jj++)
	{
	int pid = threeDiD[jj], vid = viewID[jj];
	Point2d twoDCorpus;
	for (int i = 0; i < CorpusInfo.viewIdAll3D[pid].size(); i++)
	{
	if (CorpusInfo.viewIdAll3D[pid][i] == vid)
	{
	twoDCorpus = CorpusInfo.uvAll3D[pid][i];
	break;
	}
	}
	fprintf(fp, "%d %.2f %.2f %d %.2f %.2f\n", threeDiD[jj], twoD[jj].x, twoD[jj].y, vid, twoDCorpus.x, twoDCorpus.y);
	}
	fclose(fp);*/

	return 0;
}
int LocalizeCameraToCorpus(char *Path, Corpus &CorpusInfo, CameraData  &cameraParas, int cameraID, int fixIntrinsic, int fixDistortion, int distortionCorrected, int timeID)
{
	char Fname[512];
	int threeDid, twoDid, npts, ptsCount = 0;
	double x, y, z, u, v, s = 1.0;
	vector<int> twoDidVec, threeDidVec;
	vector<double> VScale;
	vector<Point2d> Vpts;
	vector<Point3d>Vt3D;

	//sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, cameraID, timeID);
	//if (IsFileExist(Fname) == 1)
	//{
	//	printLOG("%s computed\n", Fname);
	//	return 0;
	//}

	//Note: this input has been undistorted using initIntrinsic
	sprintf(Fname, "%s/%d/PnP/_PInliers_%.4d.txt", Path, cameraID, timeID);
	if (IsFileExist(Fname) == 1)
	{
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%d ", &npts);

		twoDidVec.reserve(npts);
		threeDidVec.reserve(npts);
		Vpts.reserve(npts);
		VScale.reserve(npts);
		Vt3D.reserve(npts);

		while (fscanf(fp, "%d %lf %lf %lf %d %lf %lf %lf ", &threeDid, &x, &y, &z, &twoDid, &u, &v, &s) != EOF)
		{
			if (threeDid< 0 || threeDid > CorpusInfo.n3dPoints)
				continue;

			twoDidVec.push_back(twoDid);
			threeDidVec.push_back(threeDid);
			Vpts.push_back(Point2d(u, v));
			VScale.push_back(s);
			Vt3D.push_back(Point3d(x, y, z));
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/%d/PnP/PInliers_%.4d.txt", Path, cameraID, timeID);
		if (IsFileExist(Fname) == 0)
		{
			printLOG("Cannot load %s\n", Fname);
			return -1;
		}
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%d ", &npts);

		twoDidVec.reserve(npts);
		threeDidVec.reserve(npts);
		Vpts.reserve(npts);
		VScale.reserve(npts);
		Vt3D.reserve(npts);

		while (fscanf(fp, "%d %d %lf %lf %lf ", &threeDid, &twoDid, &u, &v, &s) != EOF)
		{
			if (threeDid< 0 || threeDid > CorpusInfo.n3dPoints)
				continue;

			twoDidVec.push_back(twoDid);
			threeDidVec.push_back(threeDid);
			Vpts.push_back(Point2d(u, v));
			VScale.push_back(s);
			Vt3D.push_back(CorpusInfo.xyz[threeDid]);
		}
		fclose(fp);
	}

	npts = (int)threeDidVec.size();
	if (npts < cameraParas.nInlierThresh)
		return -1;

	vector<int> Inliers;
	GetKFromIntrinsic(cameraParas);
	int ninliers = EstimatePoseAndInliers(cameraParas.K, cameraParas.distortion, cameraParas.LensModel, cameraParas.ShutterModel, cameraParas.R, cameraParas.T, cameraParas.wt, Vpts, Vt3D, Inliers, cameraParas.threshold, fixIntrinsic, distortionCorrected, cameraParas.minFratio, cameraParas.maxFratio, cameraParas.width, cameraParas.height, cameraParas.notCalibrated ? PnP::P4Pf : PnP::EPNP);

	GetrtFromRT(&cameraParas, 1);
	GetIntrinsicFromK(cameraParas); //optimize for focal as well gives better results-->update intrinsic here
	GetCfromT(cameraParas.R, cameraParas.T, cameraParas.camCenter);

	if (ninliers < cameraParas.nInlierThresh)
	{
		sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, cameraID, timeID);	FILE *fp = fopen(Fname, "w+"); fclose(fp);
		printLOG("Estimate pose for View (%d, %d).. fails ... low inliers (%d/%d). Camera center: %.4f %.4f %.4f \n************************************\n", cameraID, timeID, ninliers, npts, cameraParas.camCenter[0], cameraParas.camCenter[1], cameraParas.camCenter[2]);
		return -1;
	}
	else
	{
		sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, cameraID, timeID); FILE *fp = fopen(Fname, "w+");
		for (auto id : Inliers)
			fprintf(fp, "%d %f %f %f %d %.6f %.6f %.2f\n", threeDidVec[id], Vt3D[id].x, Vt3D[id].y, Vt3D[id].z, twoDidVec[id], Vpts[id].x, Vpts[id].y, VScale[id]);
		fclose(fp);

		printLOG("Estimate pose for View (%d, %d).. succeeds ... inliers (%d/%d). Camera center: %.4f %.4f %.4f \n************************************\n", cameraID, timeID, ninliers, npts, cameraParas.camCenter[0], cameraParas.camCenter[1], cameraParas.camCenter[2]);
		return ninliers;
	}
}
int ExhaustiveSearchForBestFocal(char *Path, Corpus &CorpusInfo, CameraData &CamInfoI, int selectedCam, int frameID, bool FhasBeenInit)
{
	int width = CamInfoI.width, height = CamInfoI.height, focal, bestFocal = 0, bestInlier = 0;
	int fixIntrinsic = 0, fixDistortion = 0, distortionCorrected = 0;
	int searchRange = FhasBeenInit ? 5 : 10;

	int orgShutterModel = CamInfoI.ShutterModel;
	double focal0 = FhasBeenInit ? (CamInfoI.intrinsic[0] + CamInfoI.intrinsic[1]) / 2.0 : max(width, height),
		u0 = FhasBeenInit ? CamInfoI.intrinsic[3] : width / 2, v0 = FhasBeenInit ? CamInfoI.intrinsic[3] : height / 2;

	CamInfoI.ShutterModel = 0;
	for (int range = -searchRange; range <= searchRange; range++) // search over focal length which gives highest # inliers
	{
		focal = (1.0 + 0.01*range)*focal0;
		CamInfoI.intrinsic[0] = focal, CamInfoI.intrinsic[1] = focal, CamInfoI.intrinsic[2] = 0, CamInfoI.intrinsic[3] = u0, CamInfoI.intrinsic[4] = v0;
		GetKFromIntrinsic(CamInfoI);

		for (int jj = 0; jj < 7; jj++)
			CamInfoI.distortion[jj] = 0.0;//somehow, distortion avg does not work well as setting them to 0s

		for (int ii = 0; ii < 6; ii++)
			CamInfoI.wt[ii] = 0.0;

		int ninliers = LocalizeCameraToCorpus(Path, CorpusInfo, CamInfoI, selectedCam, 0, 0, distortionCorrected, frameID);
		if (ninliers > bestInlier)
		{
			bestInlier = ninliers;
			bestFocal = CamInfoI.intrinsic[0];
		}
		printLOG("(%d, %d): Current focal: %d, Best focal: %d, best inliers: %d\n\n", selectedCam, frameID, focal, bestFocal, bestInlier);
	}

	CamInfoI.ShutterModel = orgShutterModel;
	CamInfoI.intrinsic[0] = bestFocal, CamInfoI.intrinsic[1] = bestFocal, CamInfoI.intrinsic[2] = 0, CamInfoI.intrinsic[3] = width / 2, CamInfoI.intrinsic[4] = height / 2;
	for (int ii = 0; ii < 6; ii++)
		CamInfoI.wt[ii] = 0.0;

	GetKFromIntrinsic(CamInfoI);
	if (bestInlier < CamInfoI.nInlierThresh)
		return 0;
	else
		return 1;
}
int ForceLocalizeCameraToCorpus(char *Path, CameraData  &cameraParas, int cameraID, int fixIntrinsic, int fixDistortion, int distortionCorrected, int timeID, int fromKeyFrameTracking, double *Extrinsic_Init)
{
	char Fname[512];
	int gpid, lpid;
	double x, y, z, u, v, s = 1.0;

	vector<int> gpidVec, lpidVec; vector<double> Vscale;
	vector<Point2d> Vuv; vector<Point3d> Vxyz;
	if (fromKeyFrameTracking)
	{
		sprintf(Fname, "%s/%d/PnPmTc/Inliers_%.4d.txt", Path, cameraID, timeID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return -1;
		}
		while (fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf ", &gpid, &lpid, &x, &y, &z, &u, &v, &s) != EOF)
		{
			gpidVec.push_back(gpid);
			lpidVec.push_back(lpid);
			Vuv.push_back(Point2d(u, v));
			Vxyz.push_back(Point3d(x, y, z));
			Vscale.push_back(s);
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, cameraID, timeID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return -1;
		}
		while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &gpid, &x, &y, &z, &u, &v, &s) != EOF)
		{
			gpidVec.push_back(gpid);
			Vuv.push_back(Point2d(u, v));
			Vxyz.push_back(Point3d(x, y, z));
			Vscale.push_back(s);
		}
		fclose(fp);
	}

	int npts = (int)gpidVec.size();
	if (npts < cameraParas.nInlierThresh)
		return -1;

	int LossType = 1;
	vector<bool> Good; Good.reserve(npts);
	if (Extrinsic_Init == NULL)
	{
		//undistort for RANSAC
		if (distortionCorrected == 0 && !cameraParas.notCalibrated)
		{
			if (cameraParas.LensModel == 0)
				LensCorrectionPoint(Vuv, cameraParas.K, cameraParas.distortion);
			else
				FishEyeCorrectionPoint(Vuv, cameraParas.K, cameraParas.distortion[0]);
		}

		vector<int> inliers;
		if (PnP_RANSAC(cameraParas, inliers, Vxyz, Vuv, cameraParas.width, cameraParas.height, cameraParas.LensModel, 10, 20, cameraParas.threshold, fixIntrinsic, distortionCorrected, PnP::EPNP) == 0)
			return -1;

		if (cameraParas.ShutterModel == ROLLING_SHUTTER && distortionCorrected == 0 && !cameraParas.notCalibrated)
		{
			//distort for BA
			if (cameraParas.LensModel == 0)
				LensDistortionPoint(Vuv, cameraParas.K, cameraParas.distortion);
			else
				FishEyeDistortionPoint(Vuv, cameraParas.K, cameraParas.distortion[0]);

			if (CameraPose1FrameBA(Path, cameraParas, Vxyz, Vuv, Vscale, gpidVec, Good, fixIntrinsic, fixDistortion, distortionCorrected, fromKeyFrameTracking, LossType, true) == 1)
				return -1;
		}
	}
	else
	{
		for (int ii = 0; ii < 6; ii++)
			cameraParas.rt[ii] = Extrinsic_Init[ii];

		if (CameraPose1FrameBA(Path, cameraParas, Vxyz, Vuv, Vscale, gpidVec, Good, fixIntrinsic, fixDistortion, distortionCorrected, fromKeyFrameTracking, LossType, true) == 1)
			return -1;
	}
	GetCfromT(cameraParas.R, cameraParas.T, cameraParas.camCenter);

	int ninliers = 0;
	for (int ii = 0; ii < (int)Vxyz.size(); ii++)
		if (Good[ii])
			ninliers++;

	if (ninliers < cameraParas.nInlierThresh)
	{

		printLOG("Estimated pose for View (%d, %d).. fails ... low inliers (%d/%d). Camera center: %.4f %.4f %.4f \n\n", cameraID, timeID, ninliers, npts, cameraParas.camCenter[0], cameraParas.camCenter[1], cameraParas.camCenter[2]);
		sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, cameraID, timeID);	FILE *fp = fopen(Fname, "w+"); fclose(fp);
		return -1;
	}
	else
	{
		printLOG("Estimated pose for View (%d, %d).. succeeds ... inliers (%d/%d). Camera center: %.4f %.4f %.4f \n\n", cameraID, timeID, ninliers, npts, cameraParas.camCenter[0], cameraParas.camCenter[1], cameraParas.camCenter[2]);
		sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, cameraID, timeID);	FILE *fp = fopen(Fname, "w+");
		for (int ii = 0; ii < (int)Vxyz.size(); ii++)
		{
			if (Good[ii])
				fprintf(fp, "%d %d %f %f %f %.6f %.6f %.2f\n", gpidVec[ii], lpidVec[ii], Vxyz[ii].x, Vxyz[ii].y, Vxyz[ii].z, Vuv[ii].x, Vuv[ii].y, Vscale[ii]);
		}
		fclose(fp);

		return ninliers;
	}
}
int CameraPose1FrameBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> &uvAll3D, vector<double> &scaleAll3D, vector<int> &GlobalAnchor, vector<bool>& Good, int fixIntrinsic, int fixDistortion, int distortionCorrected, int useGlobalAnchor, int LossType, int debug)
{
	char Fname[512]; FILE *fp = 0;
	int ii, npts = (int)Vxyz.size();
	double residuals[2];

	//printLOG("Set up Pose BA ...");
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(2.0); //new ceres::CauchyLoss(5.0)

	int nBadCounts = 0;
	Good.clear();
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);
	vector<double> ReProjectionErrorXA; ReProjectionErrorXA.reserve(npts);
	vector<double> ReProjectionErrorYA; ReProjectionErrorYA.reserve(npts);
	double maxOutlierX = 0.0, maxOutlierY = 0.0, pointErrX = 0.0, pointErrY = 0.0;
	int fixSkewView = 1, fixPrincipal = 1, fixPrismView = 1;
	double org_focal = (camera.intrinsic[0] + camera.intrinsic[1]) / 2;

	if (debug == 2)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionB.txt"), fp = fopen(Fname, "w+");
#endif
	for (int ii = 0; ii < npts; ii++)
	{
		if (distortionCorrected == 1)
		{
			if (camera.ShutterModel == GLOBAL_SHUTTER)
				PinholeReprojectionDebug(camera.intrinsic, camera.rt, uvAll3D[ii], Vxyz[ii], residuals);
			else
				CayleyReprojectionDebug(camera.intrinsic, camera.rt, camera.wt, uvAll3D[ii], Vxyz[ii], camera.width, camera.height, residuals);
		}
		else if (camera.LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			if (camera.ShutterModel == GLOBAL_SHUTTER)
				PinholeDistortionReprojectionDebug(camera.intrinsic, camera.distortion, camera.rt, uvAll3D[ii], Vxyz[ii], residuals);
			else
				CayleyDistortionReprojectionDebug(camera.intrinsic, camera.distortion, camera.rt, camera.wt, uvAll3D[ii], Vxyz[ii], camera.width, camera.height, residuals);
		}
		else if (camera.LensModel == FISHEYE)
		{
			if (camera.ShutterModel == GLOBAL_SHUTTER)
				FOVReprojectionDistortion2Debug(camera.intrinsic, camera.distortion, camera.rt, uvAll3D[ii], Vxyz[ii], residuals);
			else
			{
				printLOG("PerframePoseBA with FOV model for RS camera is not supported yet!");
				exit(1);
			}
		}

		bool bad = false;
		if (useGlobalAnchor == 1 && GlobalAnchor[ii] != -1 && abs(residuals[0]) > 3.0*camera.threshold || abs(residuals[1]) > 3.0*camera.threshold)
			bad = true;
		else if (abs(residuals[0]) > 1.25*camera.threshold || abs(residuals[1]) > 1.25*camera.threshold)
			bad = true;

		if (bad)
		{
			Good.push_back(false);
			if (abs(residuals[0]) > maxOutlierX)
				maxOutlierX = residuals[0];
			if (abs(residuals[1]) > maxOutlierY)
				maxOutlierY = residuals[1];
			nBadCounts++;
		}
		else
		{
			Good.push_back(true);
			double weight = (useGlobalAnchor == 1 && GlobalAnchor[ii] != -1) ? scaleAll3D[ii] / 10.0 : scaleAll3D[ii]; //smaller scale, higher weighting
			if (distortionCorrected == 1)
			{
				if (camera.ShutterModel == GLOBAL_SHUTTER)
				{
					ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[ii].x, uvAll3D[ii].y, weight);
					problem.AddResidualBlock(cost_function, loss_funcion, camera.intrinsic, camera.rt, &Vxyz[ii].x);
					problem.SetParameterBlockConstant(camera.intrinsic);
				}
				else
				{
					ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera.intrinsic, uvAll3D[ii].x, uvAll3D[ii].y, weight, camera.width, camera.height);
					problem.AddResidualBlock(cost_function, loss_funcion, camera.rt, camera.wt, &Vxyz[ii].x);
				}
			}
			else if (camera.LensModel == RADIAL_TANGENTIAL_PRISM)
			{
				if (camera.ShutterModel == GLOBAL_SHUTTER)
				{
					ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uvAll3D[ii].x, uvAll3D[ii].y, weight);
					problem.AddResidualBlock(cost_function, loss_funcion, camera.intrinsic, camera.distortion, camera.rt, &Vxyz[ii].x);
				}
				else
				{
					ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uvAll3D[ii].x, uvAll3D[ii].y, weight, camera.width, camera.height);
					problem.AddResidualBlock(cost_function, loss_funcion, camera.intrinsic, camera.distortion, camera.rt, camera.wt, &Vxyz[ii].x);
				}
			}
			else if (camera.LensModel == FISHEYE)
			{
				if (camera.ShutterModel == GLOBAL_SHUTTER)
				{
					ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uvAll3D[ii].x, uvAll3D[ii].y, weight);
					problem.AddResidualBlock(cost_function, loss_funcion, camera.intrinsic, camera.distortion, camera.rt, &Vxyz[ii].x);
				}
			}
			problem.SetParameterBlockConstant(&Vxyz[ii].x);

			if (useGlobalAnchor == 1 && GlobalAnchor[ii] != -1)
				ReProjectionErrorXA.push_back(residuals[0]), ReProjectionErrorYA.push_back(residuals[1]);
			else
				ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
		}

		if (debug == 2)
			fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", ii, Vxyz[ii].x, Vxyz[ii].y, Vxyz[ii].z, uvAll3D[ii].x, uvAll3D[ii].y, abs(residuals[0]), abs(residuals[1]));
	}
	if (debug == 2)
		fclose(fp);

	if (debug >= 1)
#pragma omp critical
		printLOG("(%d/%d) bad points detected with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, npts, maxOutlierX, maxOutlierY);

	if (ReProjectionErrorXA.size() > 0 && ReProjectionErrorYA.size() > 0)
	{
		double miniX = *min_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		double maxiX = *max_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		double avgX = MeanArray(ReProjectionErrorXA);
		double stdX = sqrt(VarianceArray(ReProjectionErrorXA, avgX));
		double miniY = *min_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		double maxiY = *max_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		double avgY = MeanArray(ReProjectionErrorYA);
		double stdY = sqrt(VarianceArray(ReProjectionErrorYA, avgY));
		if (debug >= 1)
#pragma omp critical
			printLOG("Anchor before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}
	if (ReProjectionErrorX.size() > 0 && ReProjectionErrorY.size() > 0)
	{
		double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		double avgX = MeanArray(ReProjectionErrorX);
		double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		double avgY = MeanArray(ReProjectionErrorY);
		double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		if (debug >= 1)
#pragma omp critical
			printLOG("Before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	if (nBadCounts > npts * 80 / 100)
		return 1;

	//Set up constant parameters:
	if (fixSkewView == 1 && fixPrincipal == 1)
	{
		std::vector<int> constant_parameters;
		constant_parameters.push_back(2), constant_parameters.push_back(3), constant_parameters.push_back(4);
		problem.SetParameterization(camera.intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
	}
	else if (fixSkewView == 1)
	{
		std::vector<int> constant_parameters;
		constant_parameters.push_back(2);
		problem.SetParameterization(camera.intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
	}
	else if (fixPrincipal == 1)
	{
		std::vector<int> constant_parameters;
		constant_parameters.push_back(3), constant_parameters.push_back(4);
		problem.SetParameterization(camera.intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
	}
	if (fixPrismView == 1)
	{
		std::vector<int> constant_parameters;
		constant_parameters.push_back(5), constant_parameters.push_back(6);
		problem.SetParameterization(camera.distortion, new ceres::SubsetParameterization(7, constant_parameters));
	}
	if (distortionCorrected == 0)
	{
		if (fixIntrinsic)
			problem.SetParameterBlockConstant(camera.intrinsic);
		if (fixDistortion)
			problem.SetParameterBlockConstant(camera.distortion);
	}
	//problem.SetParameterUpperBound(camera.intrinsic, 0, org_focal * 1.5), problem.SetParameterLowerBound(camera.intrinsic, 0, org_focal * 0.5);
	//problem.SetParameterUpperBound(camera.intrinsic, 1, org_focal * 1.5), problem.SetParameterLowerBound(camera.intrinsic, 1, org_focal * 0.5);

	ceres::Solver::Options options;
	if (npts > 2000)
	{
		options.num_threads = omp_get_max_threads();
		options.num_linear_solver_threads = omp_get_max_threads();
	}
	else //too much overhead
	{
		options.num_threads = 4;
		options.num_linear_solver_threads = 4;
	}
	options.max_num_iterations = 300;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	printLOG("%s\n", summary.BriefReport().c_str());

	//Store refined parameters
	GetKFromIntrinsic(&camera, 1);
	GetRTFromrt(&camera, 1);
	AssembleP(camera.K, camera.R, camera.T, camera.P);

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear(), ReProjectionErrorXA.clear(), ReProjectionErrorYA.clear();
	pointErrX = 0.0, pointErrY = 0.0;

	if (debug == 2)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionA.txt"), fp = fopen(Fname, "w+");
#endif

	for (int ii = 0; ii < npts; ii++)
	{
		if (abs(Vxyz[ii].x) + abs(Vxyz[ii].y) + abs(Vxyz[ii].z) > LIMIT3D)
		{
			if (!Good[ii])
				continue;

			if (distortionCorrected == 1)
			{
				if (camera.ShutterModel == GLOBAL_SHUTTER)
					PinholeReprojectionDebug(camera.intrinsic, camera.rt, uvAll3D[ii], Vxyz[ii], residuals);
				else
					CayleyReprojectionDebug(camera.intrinsic, camera.rt, camera.wt, uvAll3D[ii], Vxyz[ii], camera.width, camera.height, residuals);
			}
			else if (camera.LensModel == RADIAL_TANGENTIAL_PRISM)
			{
				if (camera.ShutterModel == GLOBAL_SHUTTER)
					PinholeDistortionReprojectionDebug(camera.intrinsic, camera.distortion, camera.rt, uvAll3D[ii], Vxyz[ii], residuals);
				else
					CayleyDistortionReprojectionDebug(camera.intrinsic, camera.distortion, camera.rt, camera.wt, uvAll3D[ii], Vxyz[ii], camera.width, camera.height, residuals);
			}
			else if (camera.LensModel == FISHEYE)
			{
				if (camera.ShutterModel == GLOBAL_SHUTTER)
					FOVReprojectionDistortion2Debug(camera.intrinsic, camera.distortion, camera.rt, uvAll3D[ii], Vxyz[ii], residuals);
			}

			if (useGlobalAnchor == 1 && GlobalAnchor[ii] != -1)
				ReProjectionErrorXA.push_back((residuals[0])), ReProjectionErrorYA.push_back((residuals[1]));
			else
				ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
			if (debug == 2)
				fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", ii, Vxyz[ii].x, Vxyz[ii].y, Vxyz[ii].z, uvAll3D[ii].x, uvAll3D[ii].y, residuals[0], residuals[1]);
		}
	}
	if (debug == 2)
		fclose(fp);


	if (ReProjectionErrorXA.size() > 0 && ReProjectionErrorYA.size() > 0)
	{
		double miniX = *min_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		double maxiX = *max_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		double avgX = MeanArray(ReProjectionErrorXA);
		double stdX = sqrt(VarianceArray(ReProjectionErrorXA, avgX));
		double miniY = *min_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		double maxiY = *max_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		double avgY = MeanArray(ReProjectionErrorYA);
		double stdY = sqrt(VarianceArray(ReProjectionErrorYA, avgY));
		if (debug >= 1)
#pragma omp critical
			printLOG("Anchor after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}
	if (ReProjectionErrorX.size() > 0 && ReProjectionErrorY.size() > 0)
	{
		double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		double avgX = MeanArray(ReProjectionErrorX);
		double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		double avgY = MeanArray(ReProjectionErrorY);
		double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		if (debug >= 1)
#pragma omp critical
			printLOG("After BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	return 0;
}

//General BA
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2f> > &uvAll3D, vector<vector<float> > &scaleAll3D, vector<int> &sharedIntrinsicCamID, int nimages,
	int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, int fix3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool silent, int maxIter)
{
	//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
	int weightByDistance = 0;

	int *fixSkewView = new int[nimages], *fixPrismView = new int[nimages];
	for (int ii = 0; ii < nimages; ii++)
		fixSkewView[ii] = 0, fixPrismView[ii] = 0;
	if (fixSkew == 1)
	{
		printLOG("Fix skew.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].intrinsic[2] = 0.0;
	}
	if (fixPrism)
	{
		printLOG("Fix prsim.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].distortion[5] = 0.0, camera[ii].distortion[6] = 0.0;
	}

	char Fname[512]; FILE *fp = 0;
	int viewID, npts = (int)Vxyz.size();
	double residuals[3], threshold;

	int nvalidframes = 0, minvalidView = 9e9, maxvalidView = 0;
	bool *VisibleImages = new bool[nimages];
	for (int ii = 0; ii < nimages; ii++)
	{
		VisibleImages[ii] = false;
		if (camera[ii].valid == true)
		{
			if (weightByDistance == 1)
				GetCfromT(camera[ii]);
			nvalidframes++;
		}
	}

	if (camera[0].ShutterModel == GLOBAL_SHUTTER)
		printLOG("set up GS-BA (%d views) ...\n", nvalidframes);
	else
		printLOG("set up Cayley RS-BA (%d views) ...\n", nvalidframes);
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(2.0);

	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionB.txt"), fp = fopen(Fname, "w+");
#endif

	double maxOutlierX = 0.0, maxOutlierY = 0.0;
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);

	bool *discard3Dpoint = new bool[npts];
	vector<bool> *Good = new vector<bool>[npts];
	for (int ii = 0; ii < npts; ii++)
		discard3Dpoint[ii] = false, Good[ii].reserve(viewIdAll3D[ii].size());

	int firstCameraInsharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	bool sharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	for (int ii = 0; ii < MaxSharedIntrinsicGroup; ii++)
		sharedIntrinsicGroup[ii] = false;

	int nBadCounts = 0, goodCount = 0;
	int firstValidViewID = -1, refCam = -1, nPossibleProjections = 0;
	vector<int> validCamID;
	Point2d uv;
	int increPer = 5, Per = 5;
	printLOG("Dumping data to Ceres:");
	for (int jj = 0; jj < npts; jj++)
	{
		if (100 * jj / npts >= Per)
		{
			printLOG("%d ..", Per);
			Per += increPer;
		}
		if (!IsValid3D(Vxyz[jj]))
			continue;

		int nvisibles = (int)viewIdAll3D[jj].size();
		for (int ii = 0; ii < nvisibles; ii++)
		{
			viewID = viewIdAll3D[jj][ii];
			uv = uvAll3D[jj][ii];
			if (!camera[viewID].valid)
				Good[jj].push_back(false);
			else
			{
				bool found = false;
				for (int kk = 0; !found && kk < (int)validCamID.size(); kk++)
					if (viewID == validCamID[kk])
						found = true;
				if (!found)
					validCamID.push_back(viewID);

				if (uv.x < 0 || uv.y < 0)
				{
					Good[jj].push_back(false);
					continue;
				}

				if (firstValidViewID == -1) //just to set the ref pose and determine reprojection threshold
					firstValidViewID = viewID, threshold = camera[firstValidViewID].threshold;

				//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
				if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					else
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				if (IsNumber(residuals[0]) == 0 || IsNumber(residuals[1]) == 0 || abs(residuals[0]) > threshold || abs(residuals[1]) > threshold)//because they are not corrected for rolling shutter yet
				{
					Good[jj].push_back(false);
					//printLOG("\n@P %d (%.3f %.3f %.3f):  %.2f %.2f", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2], residuals[0], residuals[1]);
					if (abs(residuals[0]) > maxOutlierX)
						maxOutlierX = residuals[0];
					if (abs(residuals[1]) > maxOutlierY)
						maxOutlierY = residuals[1];
					nBadCounts++;
				}
				else
				{
					Good[jj].push_back(true);
					if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && !sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					{
						sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = true;
						firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = viewID;
						printLOG("Set group %d master camera to %d\n", sharedIntrinsicCamID[viewID], firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]]);
					}
					goodCount++;
				}
			}
		}

		//Discard point 
		int count = 0;
		discard3Dpoint[jj] = false;
		if (fix3D == 1)
			count = nViewsPlus;
		else
		{
			for (int ii = 0; ii < nvisibles; ii++)
				if (Good[jj][ii] == true)
					count++;
			if (count < nViewsPlus)
			{
				discard3Dpoint[jj] = true;
				continue;
			}
		}

		//add 3D point and its 2D projections to Ceres
		bool once = true;
		int nValidProjectionPerPoint = 0;
		double pointErrX = 0.0, pointErrY = 0.0;
		for (int ii = 0; ii < nvisibles; ii++)
		{
			if (!Good[jj][ii])
				continue;

			nPossibleProjections++;
			uv = uvAll3D[jj][ii];
			viewID = viewIdAll3D[jj][ii];
			minvalidView = min(minvalidView, viewID);
			maxvalidView = max(maxvalidView, viewID);
			VisibleImages[viewID] = true;
			double wScale = scaleAll3D[jj][ii];
			if (weightByDistance == 1)
			{
				double dist = (Distance3D(&Vxyz[jj].x, &camera[viewID].camCenter[0]));
				wScale = sqrt(dist)*wScale / 1000.0;
			}
			double Weight = 1.0;

			if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
			{
				refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[refCam].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[refCam] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[refCam].ShutterModel = GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[jj][ii].x, uvAll3D[jj][ii].y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[refCam].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (fix3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}
			else
			{
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[viewID].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[viewID] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uv.x, uv.y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[viewID].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (fix3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}

			nValidProjectionPerPoint++;
			if ((fix3D == 1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= 0)
				ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
			pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
			if (IsNumber(pointErrX) == 0 || IsNumber(pointErrY) == 0)
				int a = 0;
			if (debug)
			{
				if (once)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.4f %.4f\n", residuals[0], residuals[1]);
			}
		}
		/*if ((fix3D == 1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= nViewsPlus)
		{
			double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
			ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
		}*/

		if (!once)
			fprintf(fp, "\n");
	}
	if (debug)
		fclose(fp);
	printLOG("\n");

	//Add pose smoothness
	vector<double> CamSmoothness;
	if (PoseSmoothness && validCamID.size() > 2)
	{
		sort(validCamID.begin(), validCamID.end());

		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0],
					camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2
					ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, 0.001*nPossibleProjections*nPossibleProjections / vCenterDif.size(), ceres::TAKE_OWNERSHIP); //such that both sum of reprojection2 and sum of smoothness2 are approx 1

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);

					problem.AddResidualBlock(cost_function1, ScaleLoss, camera[validCamID[ii]].rt, camera[validCamID[ii + 1]].rt);
				}
			}
		}
	}

	if (PoseSmoothness && CamSmoothness.size() > 0)
	{
		double minS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		printLOG("Camera center smoothness before BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f\n", minS, maxS, avgS, stdS);
	}
	/*sprintf(Fname, "%s/Good.txt", Path); fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
	fprintf(fp, "%d ", jj);
	for (int ii = 0; ii < Good[jj].size(); ii++)
	{
	if (Good[jj][ii] == false)
	fprintf(fp, "%d ", ii);
	}
	fprintf(fp, "-1\n");
	}
	fclose(fp);*/

	double miniX, maxiX, avgX, stdX, miniY, maxiY, avgY, stdY;
	if (ReProjectionErrorX.size() + ReProjectionErrorX.size() > 0)
		printLOG("(%d/%d) bad projections with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, nBadCounts + goodCount, maxOutlierX, maxOutlierY);
	else
		printLOG("Error. The BA gives 0 inliers!");

	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		printLOG("%d points with reprojection error before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();

	if (validCamID.size() < 1000)
	{
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.preconditioner_type = ceres::JACOBI;
		if (validCamID.size() < 300)
			options.function_tolerance = 1.0e-6;
		else
			options.function_tolerance = 1.0e-5;
		options.max_solver_time_in_seconds = 1000.0;
	}
	else
	{
		options.max_solver_time_in_seconds = 1.0 * (int)validCamID.size();
		if (validCamID.size() < 1500)
			options.max_num_iterations = 30;
		else if (validCamID.size() < 2000)
			options.max_num_iterations = 25;
		else if (validCamID.size() < 4000)
			options.max_num_iterations = 20;
		else if (validCamID.size() < 5000)
			options.max_num_iterations = 10;
		else
			options.max_num_iterations = 7;

		if (validCamID.size() < 1500)
			options.function_tolerance = 1.0e-5;
		if (validCamID.size() < 2000)
			options.function_tolerance = 5.0e-4;
		else if (validCamID.size() < 3000)
			options.function_tolerance = 1.0e-4;
		else
			options.function_tolerance = 5.0e-3;

		options.linear_solver_type = ceres::CGNR;
		options.preconditioner_type = ceres::JACOBI;
	}
	options.max_linear_solver_iterations = 100;
	options.minimizer_progress_to_stdout = silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;
	if (maxIter > -1)
		options.max_num_iterations = maxIter;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if (silent)
	//printLOG("%s\n",summary.BriefReport().c_str());
	printLOG("%s\n", summary.BriefReport().c_str());
	//else
	//	std::cout << summary.FullReport();

	//Store refined parameters
	for (int ii = 0; ii < (int)validCamID.size() && sharedIntrinsicCamID.size() > 0; ii++)
	{
		int vid = validCamID[ii];
		if (sharedIntrinsicCamID[vid] > -1)
		{
			refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[vid]];
			CopyCamereInfo(camera[refCam], camera[vid], false);
		}
	}
	for (int ii = 0; ii < (int)validCamID.size(); ii++)
	{
		GetKFromIntrinsic(camera[validCamID[ii]]);
		GetRTFromrt(camera[validCamID[ii]]);
		AssembleP(camera[validCamID[ii]].K, camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].P);
		GetCfromT(camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].camCenter);
	}

	/*//this may skew thing up if increF was used before
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\nNo points were observed in views: ");
	break;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("%d.. ", camera[ii].frameID);
	camera[ii].valid = 0;
	for (int jj = 0; jj < 5; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 7; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 6; jj++)
	camera[ii].rt[jj] = 0, camera[ii].wt[jj] = 0;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\n\n");
	break;
	}
	}*/

	for (int jj = 0; jj < npts; jj++)
	{
		int count = 0;
		for (size_t ii = 0; ii < Good[jj].size(); ii++)
			if (Good[jj][ii] == true)
				count++;
		if (fix3D != 1 && count < nViewsPlus)
			Vxyz[jj] = Point3d(0, 0, 0);
	}
	if (!silent && threshold < 10) //double iteation, no need to display
		printLOG("\n");

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionA.txt"), fp = fopen(Fname, "w+");
#endif

	for (int jj = 0; jj < npts; jj++)
	{
		if (abs(Vxyz[jj].x) + abs(Vxyz[jj].y) + abs(Vxyz[jj].z) > LIMIT3D && !discard3Dpoint[jj])
		{
			bool once = true;
			int nValidProjectionPerPoint = 0;
			double pointErrX = 0.0, pointErrY = 0.0;
			for (int ii = 0; ii < viewIdAll3D[jj].size(); ii++)
			{
				if (!Good[jj][ii] || uv.x < 0 || uv.y < 0)
					continue;

				viewID = viewIdAll3D[jj][ii];
				uv = uvAll3D[jj][ii];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				nValidProjectionPerPoint++;
				if ((fix3D == 1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= 0)
					ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);

				pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
				if (once && debug)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				if (debug)
					fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.3f %.3f\n", residuals[0], residuals[1]);
			}
			if (!once &&debug)
				fprintf(fp, "\n");

			/*if (nValidProjectionPerPoint >= 1)
			{
				double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
				ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
			}*/
		}
	}
	if (debug)
		fclose(fp);

	//Add pose smoothness
	CamSmoothness.clear();
	if (PoseSmoothness && validCamID.size() > 2)
	{
		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);
				}
			}
		}
	}

	if (PoseSmoothness && CamSmoothness.size() > 0)
	{
		double minS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		printLOG("Camera center smoothness after BA: Min: %.4f Max: %42f Mean: %.4f Std: %.4f\n", minS, maxS, avgS, stdS);
	}

	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		printLOG("%d points with reprojection error after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	delete[]discard3Dpoint;
	delete[]Good, delete[]VisibleImages;

	return 0;
}
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2f> > &uvAll3D, vector<vector<float> > &scaleAll3D, vector<int> &sharedIntrinsicCamID, int nimages,
	int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, vector<int> & VGlobalAnchor, int fixedGlobalAnchor3D, int fixedLocalAnchor3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool silent, int maxIter)
{
	//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
	int weightByDistance = 0;

	int *fixSkewView = new int[nimages], *fixPrismView = new int[nimages];
	for (int ii = 0; ii < nimages; ii++)
		fixSkewView[ii] = 0, fixPrismView[ii] = 0;
	if (fixSkew == 1)
	{
		printLOG("Fix skew.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].intrinsic[2] = 0.0;
	}
	if (fixPrism)
	{
		printLOG("Fix prsim.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].distortion[5] = 0.0, camera[ii].distortion[6] = 0.0;
	}

	char Fname[512]; FILE *fp = 0;
	int viewID, npts = (int)Vxyz.size();
	double residuals[3], threshold;

	int nvalidframes = 0, minvalidView = 9e9, maxvalidView = 0;
	bool *VisibleImages = new bool[nimages];
	for (int ii = 0; ii < nimages; ii++)
	{
		VisibleImages[ii] = false;
		if (camera[ii].valid == true)
		{
			if (weightByDistance == 1)
				GetCfromT(camera[ii]);
			nvalidframes++;
		}
	}

	if (camera[0].ShutterModel == GLOBAL_SHUTTER)
		printLOG("set up GS-BA (%d views) ...\n", nvalidframes);
	else
		printLOG("set up Cayley RS-BA (%d views) ...\n", nvalidframes);
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(2.0);

	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionB.txt"), fp = fopen(Fname, "w+");
#endif

	double maxOutlierX = 0.0, maxOutlierY = 0.0;
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);
	vector<double> ReProjectionErrorXA; ReProjectionErrorXA.reserve(npts);
	vector<double> ReProjectionErrorYA; ReProjectionErrorYA.reserve(npts);

	bool *discard3Dpoint = new bool[npts];
	vector<bool> *Good = new vector<bool>[npts];
	for (int ii = 0; ii < npts; ii++)
		discard3Dpoint[ii] = false, Good[ii].reserve(viewIdAll3D[ii].size());

	int firstCameraInsharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	bool sharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	for (int ii = 0; ii < MaxSharedIntrinsicGroup; ii++)
		sharedIntrinsicGroup[ii] = false;

	int nBadCounts = 0, goodCount = 0;
	int firstValidViewID = -1, refCam = -1, nPossibleProjections = 0;
	vector<int> validCamID;
	Point2d uv;
	int increPer = 5, Per = 5;
	printLOG("Dumping data to Ceres:");
	for (int jj = 0; jj < npts; jj++)
	{
		if (100 * jj / npts >= Per)
		{
			printLOG("%d ..", Per);
			Per += increPer;
		}
		if (!IsValid3D(Vxyz[jj]))
			continue;

		int nvisibles = (int)viewIdAll3D[jj].size(), anchorId = VGlobalAnchor[jj];
		for (int ii = 0; ii < nvisibles; ii++)
		{
			viewID = viewIdAll3D[jj][ii];
			uv = uvAll3D[jj][ii];
			if (!camera[viewID].valid)
				Good[jj].push_back(false);
			else
			{
				bool found = false;
				for (int kk = 0; !found && kk < (int)validCamID.size(); kk++)
					if (viewID == validCamID[kk])
						found = true;
				if (!found)
					validCamID.push_back(viewID);

				if (uv.x < 0 || uv.y < 0)
				{
					Good[jj].push_back(false);
					continue;
				}

				if (firstValidViewID == -1) //just to set the ref pose and determine reprojection threshold
					firstValidViewID = viewID, threshold = camera[firstValidViewID].threshold;

				//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
				if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					else
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				if (IsNumber(residuals[0]) == 0 || IsNumber(residuals[1]) == 0 || abs(residuals[0]) > threshold || abs(residuals[1]) > threshold)//because they are not corrected for rolling shutter yet
				{
					Good[jj].push_back(false);
					//printLOG("\n@P %d (%.3f %.3f %.3f):  %.2f %.2f", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2], residuals[0], residuals[1]);
					if (abs(residuals[0]) > maxOutlierX)
						maxOutlierX = residuals[0];
					if (abs(residuals[1]) > maxOutlierY)
						maxOutlierY = residuals[1];
					nBadCounts++;
				}
				else
				{
					Good[jj].push_back(true);
					if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && !sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					{
						sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = true;
						firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = viewID;
						printLOG("Set group %d master camera to %d\n", sharedIntrinsicCamID[viewID], firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]]);
					}
					goodCount++;
				}
			}
		}

		//Discard point 
		int count = 0;
		discard3Dpoint[jj] = false;
		for (int ii = 0; ii < nvisibles; ii++)
			if (Good[jj][ii] == true)
				count++;

		if (anchorId > -1 && fixedGlobalAnchor3D == 1 && count > 0) //let get it in even if it has only 1 view
			count = nViewsPlus;
		else
		{
			if (count < nViewsPlus)
			{
				discard3Dpoint[jj] = true;
				continue;
			}
		}

		//add 3D point and its 2D projections to Ceres
		bool once = true;
		int nValidProjectionPerPoint = 0;
		double pointErrX = 0.0, pointErrY = 0.0;
		for (int ii = 0; ii < nvisibles; ii++)
		{
			if (!Good[jj][ii])
				continue;

			nPossibleProjections++;
			uv = uvAll3D[jj][ii];
			viewID = viewIdAll3D[jj][ii];
			minvalidView = min(minvalidView, viewID);
			maxvalidView = max(maxvalidView, viewID);
			VisibleImages[viewID] = true;
			double wScale = scaleAll3D[jj][ii];
			if (weightByDistance == 1)
			{
				double dist = (Distance3D(&Vxyz[jj].x, &camera[viewID].camCenter[0]));
				wScale = sqrt(dist)*wScale / 1000.0;
			}
			double Weight = anchorId > -1 ? 10.0 : 1.0;

			if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
			{
				refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[refCam].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[refCam] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[refCam].ShutterModel = GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[jj][ii].x, uvAll3D[jj][ii].y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[refCam].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (anchorId > -1 && fixedGlobalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
				if (anchorId == -1 && fixedLocalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}
			else
			{
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[viewID].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[viewID] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uv.x, uv.y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[viewID].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (anchorId > -1 && fixedGlobalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
				if (anchorId == -1 && fixedLocalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}

			nValidProjectionPerPoint++;
			if (anchorId > -1)
				ReProjectionErrorXA.push_back(residuals[0]), ReProjectionErrorYA.push_back(residuals[1]);
			else
				ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);

			//pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
			//if (IsNumber(pointErrX) == 0 || IsNumber(pointErrY) == 0)
			//int a = 0;
			if (debug)
			{
				if (once)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.4f %.4f\n", residuals[0], residuals[1]);
			}
		}
		/*if ((anchorId > -1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= nViewsPlus)
		{
		double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
		if (anchorId > -1)
		ReProjectionErrorXA.push_back(errX), ReProjectionErrorYA.push_back(errY);
		else
		ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
		}*/

		if (!once)
			fprintf(fp, "\n");
	}
	if (debug)
		fclose(fp);
	printLOG("\n");

	//Add pose smoothness
	vector<double> CamSmoothness;
	if (PoseSmoothness && validCamID.size() > 2)
	{
		sort(validCamID.begin(), validCamID.end());

		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2
					ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, 0.001*nPossibleProjections*nPossibleProjections / vCenterDif.size(), ceres::TAKE_OWNERSHIP); //such that both sum of reprojection2 and sum of smoothness2 are approx 1

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);

					problem.AddResidualBlock(cost_function1, ScaleLoss, camera[validCamID[ii]].rt, camera[validCamID[ii + 1]].rt);
				}
			}
		}
	}
	/*sprintf(Fname, "%s/Good.txt", Path); fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
	fprintf(fp, "%d ", jj);
	for (int ii = 0; ii < Good[jj].size(); ii++)
	{
	if (Good[jj][ii] == false)
	fprintf(fp, "%d ", ii);
	}
	fprintf(fp, "-1\n");
	}
	fclose(fp);*/

	if (PoseSmoothness && CamSmoothness.size() > 0)
	{

		double minS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		printLOG("Camera center smoothness before BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f\n", minS, maxS, avgS, stdS);
	}

	double miniX, maxiX, avgX, stdX, miniY, maxiY, avgY, stdY;
	if (ReProjectionErrorXA.size() + ReProjectionErrorX.size() > 0)
		printLOG("(%d/%d) bad projections with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, nBadCounts + goodCount, maxOutlierX, maxOutlierY);
	else
		printLOG("Error. The BA gives 0 inliers!");
	if (ReProjectionErrorXA.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		maxiX = *max_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		avgX = MeanArray(ReProjectionErrorXA);
		stdX = sqrt(VarianceArray(ReProjectionErrorXA, avgX));
		miniY = *min_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		maxiY = *max_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		avgY = MeanArray(ReProjectionErrorYA);
		stdY = sqrt(VarianceArray(ReProjectionErrorYA, avgY));
		printLOG("%d anchor points with reprojection error before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorYA.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}
	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		printLOG("%d points with reprojection error before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();

	if (validCamID.size() < 1000)
	{
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.preconditioner_type = ceres::JACOBI;
		if (validCamID.size() < 300)
			options.function_tolerance = 1.0e-6;
		else
			options.function_tolerance = 1.0e-5;
		options.max_solver_time_in_seconds = 1000.0;
	}
	else
	{
		options.max_solver_time_in_seconds = 1.0 * (int)validCamID.size();
		if (validCamID.size() < 1500)
			options.max_num_iterations = 30;
		else if (validCamID.size() < 2000)
			options.max_num_iterations = 25;
		else if (validCamID.size() < 4000)
			options.max_num_iterations = 20;
		else if (validCamID.size() < 5000)
			options.max_num_iterations = 10;
		else
			options.max_num_iterations = 7;

		if (validCamID.size() < 1500)
			options.function_tolerance = 1.0e-5;
		if (validCamID.size() < 2000)
			options.function_tolerance = 5.0e-4;
		else if (validCamID.size() < 3000)
			options.function_tolerance = 1.0e-4;
		else
			options.function_tolerance = 5.0e-3;

		options.linear_solver_type = ceres::CGNR;
		options.preconditioner_type = ceres::JACOBI;
	}
	options.max_linear_solver_iterations = 100;
	options.minimizer_progress_to_stdout = silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;
	if (maxIter > -1)
		options.max_num_iterations = maxIter;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if (silent)
	//printLOG("%s\n",summary.BriefReport().c_str());
	printLOG("%s\n", summary.BriefReport().c_str());
	//else
	//	std::cout << summary.FullReport();

	//Store refined parameters
	for (int ii = 0; ii < (int)validCamID.size() && sharedIntrinsicCamID.size() > 0; ii++)
	{
		int vid = validCamID[ii];
		if (sharedIntrinsicCamID[vid] > -1)
		{
			refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[vid]];
			CopyCamereInfo(camera[refCam], camera[vid], false);
		}
	}
	for (int ii = 0; ii < (int)validCamID.size(); ii++)
	{
		GetKFromIntrinsic(camera[validCamID[ii]]);
		GetRTFromrt(camera[validCamID[ii]]);
		AssembleP(camera[validCamID[ii]].K, camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].P);
		GetCfromT(camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].camCenter);
	}

	/*//this may skew thing up if increF was used before
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\nNo points were observed in views: ");
	break;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("%d.. ", camera[ii].frameID);
	camera[ii].valid = 0;
	for (int jj = 0; jj < 5; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 7; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 6; jj++)
	camera[ii].rt[jj] = 0, camera[ii].wt[jj] = 0;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\n\n");
	break;
	}
	}*/

	for (int jj = 0; jj < npts; jj++)
	{
		int count = 0;
		for (size_t ii = 0; ii < Good[jj].size(); ii++)
			if (Good[jj][ii] == true)
				count++;
		if (VGlobalAnchor[jj] == -1 && count < nViewsPlus)
			Vxyz[jj] = Point3d(0, 0, 0);
	}
	if (!silent && threshold < 10) //double iteation, no need to display
		printLOG("\n");

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear(), ReProjectionErrorXA.clear(), ReProjectionErrorYA.clear();
	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionA.txt"), fp = fopen(Fname, "w+");
#endif

	for (int jj = 0; jj < npts; jj++)
	{
		if (abs(Vxyz[jj].x) + abs(Vxyz[jj].y) + abs(Vxyz[jj].z) > LIMIT3D && !discard3Dpoint[jj])
		{
			bool once = true;
			int nValidProjectionPerPoint = 0;
			double pointErrX = 0.0, pointErrY = 0.0;
			for (int ii = 0; ii < viewIdAll3D[jj].size(); ii++)
			{
				if (!Good[jj][ii] || uv.x < 0 || uv.y < 0)
					continue;

				viewID = viewIdAll3D[jj][ii];
				uv = uvAll3D[jj][ii];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				nValidProjectionPerPoint++;
				if (VGlobalAnchor[jj] > -1)
					ReProjectionErrorXA.push_back(residuals[0]), ReProjectionErrorYA.push_back(residuals[1]);
				else
					ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
				//pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
				if (once && debug)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				if (debug)
					fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.3f %.3f\n", residuals[0], residuals[1]);
			}
			if (!once &&debug)
				fprintf(fp, "\n");

			/*if (nValidProjectionPerPoint >= 1)
			{
			double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
			if (VGlobalAnchor[jj] > -1)
			ReProjectionErrorXA.push_back(errX), ReProjectionErrorYA.push_back(errY);
			else
			ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
			}*/
		}
	}
	if (debug)
		fclose(fp);

	//Add pose smoothness
	CamSmoothness.clear();
	if (PoseSmoothness && validCamID.size() > 2)
	{
		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);
				}
			}
		}
	}

	if (PoseSmoothness && CamSmoothness.size() > 0)
	{
		double miniS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxiS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		printLOG("Camera center smoothness after BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f\n", miniS, maxiS, maxiX, avgS, stdS);
	}
	if (ReProjectionErrorXA.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		maxiX = *max_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		avgX = MeanArray(ReProjectionErrorXA);
		stdX = sqrt(VarianceArray(ReProjectionErrorXA, avgX));
		miniY = *min_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		maxiY = *max_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		avgY = MeanArray(ReProjectionErrorYA);
		stdY = sqrt(VarianceArray(ReProjectionErrorYA, avgY));
		printLOG("%d anchor points with reprojection error after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorYA.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}
	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		printLOG("%d points with reprojection error after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	delete[]discard3Dpoint;
	delete[]Good, delete[]VisibleImages;

	return 0;
}
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<vector<double> > &scaleAll3D, vector<int> &sharedIntrinsicCamID, int nimages,
	int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, int fix3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool silent, int maxIter)
{
	//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
	int weightByDistance = 0;

	int *fixSkewView = new int[nimages], *fixPrismView = new int[nimages];
	for (int ii = 0; ii < nimages; ii++)
		fixSkewView[ii] = 0, fixPrismView[ii] = 0;
	if (fixSkew == 1)
	{
		printLOG("Fix skew.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].intrinsic[2] = 0.0;
	}
	if (fixPrism)
	{
		printLOG("Fix prsim.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].distortion[5] = 0.0, camera[ii].distortion[6] = 0.0;
	}

	char Fname[512]; FILE *fp = 0;
	int viewID, npts = (int)Vxyz.size();
	double residuals[3], rho[3], threshold;

	int nvalidframes = 0, minvalidView = 9e9, maxvalidView = 0;
	bool *VisibleImages = new bool[nimages];
	for (int ii = 0; ii < nimages; ii++)
	{
		VisibleImages[ii] = false;
		if (camera[ii].valid == true)
		{
			if (weightByDistance == 1)
				GetCfromT(camera[ii]);
			nvalidframes++;
		}
	}

	if (camera[0].ShutterModel == GLOBAL_SHUTTER)
		printLOG("set up GS-BA (%d views) ...\n", nvalidframes);
	else
		printLOG("set up Cayley RS-BA (%d views) ...\n", nvalidframes);
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(2.0);

	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionB.txt"), fp = fopen(Fname, "w+");
#endif

	double maxOutlierX = 0.0, maxOutlierY = 0.0;
	vector<double> CeresReProjectionError; CeresReProjectionError.reserve(npts * 10);
	vector<double> CeresSmoothnessError; CeresSmoothnessError.reserve(10000);
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);

	bool *discard3Dpoint = new bool[npts];
	vector<bool> *Good = new vector<bool>[npts];
	for (int ii = 0; ii < npts; ii++)
		discard3Dpoint[ii] = false, Good[ii].reserve(viewIdAll3D[ii].size());

	int firstCameraInsharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	bool sharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	for (int ii = 0; ii < MaxSharedIntrinsicGroup; ii++)
		sharedIntrinsicGroup[ii] = false;

	int nBadCounts = 0, goodCount = 0;
	int firstValidViewID = -1, refCam = -1;
	vector<int> validCamID;
	Point2d uv;
	int increPer = 5, Per = 5;
	printLOG("Dumping data to Ceres:");
	for (int jj = 0; jj < npts; jj++)
	{
		if (!IsValid3D(Vxyz[jj]))
			continue;

		int nvisibles = (int)viewIdAll3D[jj].size();
		for (int ii = 0; ii < nvisibles; ii++)
		{
			viewID = viewIdAll3D[jj][ii];
			uv = uvAll3D[jj][ii];
			if (!camera[viewID].valid)
				Good[jj].push_back(false);
			else
			{
				bool found = false;
				for (int kk = 0; !found && kk < (int)validCamID.size(); kk++)
					if (viewID == validCamID[kk])
						found = true;
				if (!found)
					validCamID.push_back(viewID);

				if ((uv.x < 0 || uv.y < 0) && distortionCorrected == 0)
				{
					Good[jj].push_back(false);
					continue;
				}

				if (firstValidViewID == -1) //just to set the ref pose and determine reprojection threshold
					firstValidViewID = viewID, threshold = camera[firstValidViewID].threshold;

				//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
				if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					else
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				if (IsNumber(residuals[0]) == 0 || IsNumber(residuals[1]) == 0 || abs(residuals[0]) > threshold || abs(residuals[1]) > threshold)//because they are not corrected for rolling shutter yet
				{
					Good[jj].push_back(false);
					if (abs(residuals[0]) > maxOutlierX)
						maxOutlierX = residuals[0];
					if (abs(residuals[1]) > maxOutlierY)
						maxOutlierY = residuals[1];
					nBadCounts++;
				}
				else
				{
					Good[jj].push_back(true);
					if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && !sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					{
						sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = true;
						firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = viewID;
						printLOG("Set group %d master camera to %d\n", sharedIntrinsicCamID[viewID], firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]]);
					}
					goodCount++;
				}
			}
		}

		//Discard point 
		int count = 0;
		discard3Dpoint[jj] = false;
		if (fix3D == 1)
			count = nViewsPlus;
		else
		{
			for (int ii = 0; ii < nvisibles; ii++)
				if (Good[jj][ii] == true)
					count++;
			if (count < nViewsPlus)
			{
				discard3Dpoint[jj] = true;
				continue;
			}
		}
	}

	for (int jj = 0; jj < npts; jj++)
	{
		if (100 * jj / npts >= Per)
		{
			printLOG("%d ..", Per);
			Per += increPer;
		}
		if (!IsValid3D(Vxyz[jj]) || discard3Dpoint[jj])
			continue;

		//add 3D point and its 2D projections to Ceres
		bool once = true;
		int nValidProjectionPerPoint = 0, nvisibles = (int)viewIdAll3D[jj].size();
		double pointErrX = 0.0, pointErrY = 0.0;
		for (int ii = 0; ii < nvisibles; ii++)
		{
			if (!Good[jj][ii])
				continue;

			uv = uvAll3D[jj][ii];
			viewID = viewIdAll3D[jj][ii];
			minvalidView = min(minvalidView, viewID);
			maxvalidView = max(maxvalidView, viewID);
			VisibleImages[viewID] = true;
			double wScale = scaleAll3D[jj][ii];
			if (weightByDistance == 1)
			{
				double dist = (Distance3D(&Vxyz[jj].x, &camera[viewID].camCenter[0]));
				wScale = sqrt(dist)*wScale / 1000.0;
			}
			double Weight = 1.0 / goodCount;
			ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);

			if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
			{
				refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];

				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[refCam].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[refCam] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[refCam].ShutterModel = GLOBAL_SHUTTER)
					{
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[jj][ii].x, uvAll3D[jj][ii].y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[refCam].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (fix3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}
			else
			{
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[viewID].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[viewID] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uv.x, uv.y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[viewID].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (fix3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}

			nValidProjectionPerPoint++;
			double res2 = (pow(residuals[0], 2) + pow(residuals[1], 2)) / pow(wScale, 2);
			weightedLoss->Evaluate(res2, rho);
			CeresReProjectionError.push_back(rho[0]);

			if ((fix3D == 1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= 0)
				ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
			//pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);

			if (debug)
			{
				if (once)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.4f %.4f\n", residuals[0], residuals[1]);
			}
		}
		/*if ((fix3D == 1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= nViewsPlus)
		{
			double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
			ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
		}*/

		if (!once)
			fprintf(fp, "\n");
	}
	if (debug)
		fclose(fp);
	printLOG("\n");

	//Add pose smoothness
	vector<double> CamSmoothness;
	if (PoseSmoothness && validCamID.size() > 2)
	{
		sort(validCamID.begin(), validCamID.end());

		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
			{
				double dif = norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2]));
				vCenterDif.push_back(dif);
			}
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, 0.001 / vCenterDif.size(), ceres::TAKE_OWNERSHIP); //such that both sum of reprojection2 and sum of smoothness2 are approx 1

			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);

					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);
					double res2 = pow(residuals[0], 2) + pow(residuals[1], 2) + pow(residuals[2], 2);
					ScaleLoss->Evaluate(res2, rho);
					CeresSmoothnessError.push_back(rho[0]);

					problem.AddResidualBlock(cost_function1, ScaleLoss, camera[validCamID[ii]].rt, camera[validCamID[ii + 1]].rt);
				}
			}
		}
	}

	/*sprintf(Fname, "%s/Good.txt", Path); fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
	fprintf(fp, "%d ", jj);
	for (int ii = 0; ii < Good[jj].size(); ii++)
	{
	if (Good[jj][ii] == false)
	fprintf(fp, "%d ", ii);
	}
	fprintf(fp, "-1\n");
	}
	fclose(fp);*/

	double miniX, maxiX, avgX, stdX, miniY, maxiY, avgY, stdY;
	if (ReProjectionErrorX.size() + ReProjectionErrorX.size() > 0)
		printLOG("(%d/%d) bad projections with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, nBadCounts + goodCount, maxOutlierX, maxOutlierY);
	else
		printLOG("Error. The BA gives 0 inliers!");
	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

		double sos = 0;
		for (int ii = 0; ii < CeresReProjectionError.size(); ii++)
			sos += CeresReProjectionError[ii];
		printLOG("%d points with reprojection error before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f) SoS: %.4f\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY, sos);
	}
	if (PoseSmoothness && CamSmoothness.size() > 0)
	{
		double minS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));

		double sos = 0;
		for (int ii = 0; ii < CeresSmoothnessError.size(); ii++)
			sos += CeresSmoothnessError[ii];

		printLOG("Camera center smoothness before BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f SoS: %.4f\n", minS, maxS, avgS, stdS, sos);
	}

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();

	if (validCamID.size() < 1000)
	{
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.preconditioner_type = ceres::JACOBI;
		if (validCamID.size() < 300)
			options.function_tolerance = 1.0e-6;
		else
			options.function_tolerance = 1.0e-5;
		options.max_solver_time_in_seconds = 1000.0;
	}
	else
	{
		options.max_solver_time_in_seconds = 1.0 * (int)validCamID.size();
		if (validCamID.size() < 1500)
			options.max_num_iterations = 30;
		else if (validCamID.size() < 2000)
			options.max_num_iterations = 25;
		else if (validCamID.size() < 4000)
			options.max_num_iterations = 20;
		else if (validCamID.size() < 5000)
			options.max_num_iterations = 10;
		else
			options.max_num_iterations = 7;

		if (validCamID.size() < 1500)
			options.function_tolerance = 1.0e-5;
		if (validCamID.size() < 2000)
			options.function_tolerance = 5.0e-4;
		else if (validCamID.size() < 3000)
			options.function_tolerance = 1.0e-4;
		else
			options.function_tolerance = 5.0e-3;

		options.linear_solver_type = ceres::CGNR;
		options.preconditioner_type = ceres::JACOBI;
	}
	options.max_linear_solver_iterations = 100;
	options.minimizer_progress_to_stdout = silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;
	if (maxIter > -1)
		options.max_num_iterations = maxIter;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	printLOG("%s\n", summary.BriefReport().c_str());

	//Store refined parameters
	for (int ii = 0; ii < (int)validCamID.size() && sharedIntrinsicCamID.size() > 0; ii++)
	{
		int vid = validCamID[ii];
		if (sharedIntrinsicCamID[vid] > -1)
		{
			refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[vid]];
			CopyCamereInfo(camera[refCam], camera[vid], false);
		}
	}
	for (int ii = 0; ii < (int)validCamID.size(); ii++)
	{
		GetKFromIntrinsic(camera[validCamID[ii]]);
		GetRTFromrt(camera[validCamID[ii]]);
		AssembleP(camera[validCamID[ii]].K, camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].P);
		GetCfromT(camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].camCenter);
	}

	/*//this may skew thing up if increF was used before
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\nNo points were observed in views: ");
	break;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("%d.. ", camera[ii].frameID);
	camera[ii].valid = 0;
	for (int jj = 0; jj < 5; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 7; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 6; jj++)
	camera[ii].rt[jj] = 0, camera[ii].wt[jj] = 0;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\n\n");
	break;
	}
	}*/

	for (int jj = 0; jj < npts; jj++)
	{
		int count = 0;
		for (size_t ii = 0; ii < Good[jj].size(); ii++)
			if (Good[jj][ii] == true)
				count++;
		if (fix3D != 1 && count < nViewsPlus)
			Vxyz[jj] = Point3d(0, 0, 0);
	}
	if (!silent && threshold < 10) //double iteation, no need to display
		printLOG("\n");

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear(), CeresReProjectionError.clear(), CeresSmoothnessError.clear();
	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionA.txt"), fp = fopen(Fname, "w+");
#endif

	for (int jj = 0; jj < npts; jj++)
	{
		if (IsValid3D(Vxyz[jj]) && !discard3Dpoint[jj])
		{
			bool once = true;
			int nValidProjectionPerPoint = 0;
			double pointErrX = 0.0, pointErrY = 0.0;
			for (int ii = 0; ii < viewIdAll3D[jj].size(); ii++)
			{
				if (!Good[jj][ii] || uv.x < 0 || uv.y < 0)
					continue;

				viewID = viewIdAll3D[jj][ii];
				uv = uvAll3D[jj][ii];
				double wScale = scaleAll3D[jj][ii];
				if (weightByDistance == 1)
				{
					double dist = (Distance3D(&Vxyz[jj].x, &camera[viewID].camCenter[0]));
					wScale = sqrt(dist)*wScale / 1000.0;
				}
				double Weight = 1.0 / goodCount;
				ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);

				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				nValidProjectionPerPoint++;

				double res2 = (pow(residuals[0], 2) + pow(residuals[1], 2)) / pow(wScale, 2);
				weightedLoss->Evaluate(res2, rho);
				CeresReProjectionError.push_back(rho[0]);

				if ((fix3D == 1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= 0)
					ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
				//pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);

				if (once && debug)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				if (debug)
					fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.3f %.3f\n", residuals[0], residuals[1]);
			}
			if (!once &&debug)
				fprintf(fp, "\n");

			/*if (nValidProjectionPerPoint >= 1)
			{
				double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
				ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
			}*/
		}
	}
	if (debug)
		fclose(fp);

	//Add pose smoothness
	CamSmoothness.clear();
	if (PoseSmoothness && validCamID.size() > 2)
	{
		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, 0.001 / vCenterDif.size(), ceres::TAKE_OWNERSHIP); //such that both sum of reprojection2 and sum of smoothness2 are approx 1

			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);
					double res2 = pow(residuals[0], 2) + pow(residuals[1], 2) + pow(residuals[2], 2);
					ScaleLoss->Evaluate(res2, rho);
					CeresSmoothnessError.push_back(rho[0]);
				}
			}
		}
	}

	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

		double sos = 0;
		for (int ii = 0; ii < CeresReProjectionError.size(); ii++)
			sos += CeresReProjectionError[ii];
		printLOG("%d points with reprojection error after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f) SoS: %.4f\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY, sos);
		if (CamSmoothness.size() == 0)
			printLOG("\n");
	}
	if (PoseSmoothness && CamSmoothness.size() > 0)
	{
		double minS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		double sos = 0;
		for (int ii = 0; ii < CeresSmoothnessError.size(); ii++)
			sos += CeresSmoothnessError[ii];

		printLOG("Camera center smoothness after BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f SoS: %.4f\n\n", minS, maxS, avgS, stdS, sos);
	}

	delete[]discard3Dpoint;
	delete[]Good, delete[]VisibleImages;

	return 0;
}
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<vector<double> > &scaleAll3D, vector<int> &sharedIntrinsicCamID, int nimages,
	int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, vector<int> & VGlobalAnchor, int fixedGlobalAnchor3D, int fixedLocalAnchor3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool silent, int maxIter)
{
	//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
	int weightByDistance = 0;

	int *fixSkewView = new int[nimages], *fixPrismView = new int[nimages];
	for (int ii = 0; ii < nimages; ii++)
		fixSkewView[ii] = 0, fixPrismView[ii] = 0;
	if (fixSkew == 1)
	{
		printLOG("Fix skew.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].intrinsic[2] = 0.0;
	}
	if (fixPrism)
	{
		printLOG("Fix prsim.\n");
		for (int ii = 0; ii < nimages; ii++)
			camera[ii].distortion[5] = 0.0, camera[ii].distortion[6] = 0.0;
	}

	char Fname[512]; FILE *fp = 0;
	int viewID, npts = (int)Vxyz.size();
	double residuals[3], threshold;

	int nvalidframes = 0, minvalidView = 9e9, maxvalidView = 0;
	bool *VisibleImages = new bool[nimages];
	for (int ii = 0; ii < nimages; ii++)
	{
		VisibleImages[ii] = false;
		if (camera[ii].valid == true)
		{
			if (weightByDistance == 1)
				GetCfromT(camera[ii]);
			nvalidframes++;
		}
	}

	if (camera[0].ShutterModel == GLOBAL_SHUTTER)
		printLOG("set up GS-BA (%d views) ...\n", nvalidframes);
	else
		printLOG("set up Cayley RS-BA (%d views) ...\n", nvalidframes);
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(2.0);

	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionB.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionB.txt"), fp = fopen(Fname, "w+");
#endif

	double maxOutlierX = 0.0, maxOutlierY = 0.0;
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts);
	vector<double> ReProjectionErrorXA; ReProjectionErrorXA.reserve(npts);
	vector<double> ReProjectionErrorYA; ReProjectionErrorYA.reserve(npts);

	bool *discard3Dpoint = new bool[npts];
	vector<bool> *Good = new vector<bool>[npts];
	for (int ii = 0; ii < npts; ii++)
		discard3Dpoint[ii] = false, Good[ii].reserve(viewIdAll3D[ii].size());

	int firstCameraInsharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	bool sharedIntrinsicGroup[MaxSharedIntrinsicGroup];
	for (int ii = 0; ii < MaxSharedIntrinsicGroup; ii++)
		sharedIntrinsicGroup[ii] = false;

	int nBadCounts = 0, goodCount = 0;
	int firstValidViewID = -1, refCam = -1, nPossibleProjections = 0;
	vector<int> validCamID;
	Point2d uv;
	int increPer = 5, Per = 5;
	printLOG("Dumping data to Ceres:");
	for (int jj = 0; jj < npts; jj++)
	{
		if (100 * jj / npts >= Per)
		{
			printLOG("%d ..", Per);
			Per += increPer;
		}
		if (!IsValid3D(Vxyz[jj]))
			continue;

		int nvisibles = (int)viewIdAll3D[jj].size(), anchorId = VGlobalAnchor[jj];
		for (int ii = 0; ii < nvisibles; ii++)
		{
			viewID = viewIdAll3D[jj][ii];
			uv = uvAll3D[jj][ii];
			if (!camera[viewID].valid)
				Good[jj].push_back(false);
			else
			{
				bool found = false;
				for (int kk = 0; !found && kk < (int)validCamID.size(); kk++)
					if (viewID == validCamID[kk])
						found = true;
				if (!found)
					validCamID.push_back(viewID);

				if (uv.x < 0 || uv.y < 0)
				{
					Good[jj].push_back(false);
					continue;
				}

				if (firstValidViewID == -1) //just to set the ref pose and determine reprojection threshold
					firstValidViewID = viewID, threshold = camera[firstValidViewID].threshold;

				//Use the parameters from indi frames to selected inliers. For BA, the para from ref frame is used.
				if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					else
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				if (IsNumber(residuals[0]) == 0 || IsNumber(residuals[1]) == 0 || abs(residuals[0]) > threshold || abs(residuals[1]) > threshold)//because they are not corrected for rolling shutter yet
				{
					Good[jj].push_back(false);
					//printLOG("\n@P %d (%.3f %.3f %.3f):  %.2f %.2f", jj, xyz[3 * jj], xyz[3 * jj + 1], xyz[3 * jj + 2], residuals[0], residuals[1]);
					if (abs(residuals[0]) > maxOutlierX)
						maxOutlierX = residuals[0];
					if (abs(residuals[1]) > maxOutlierY)
						maxOutlierY = residuals[1];
					nBadCounts++;
				}
				else
				{
					Good[jj].push_back(true);
					if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && !sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
					{
						sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = true;
						firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]] = viewID;
						printLOG("Set group %d master camera to %d\n", sharedIntrinsicCamID[viewID], firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]]);
					}
					goodCount++;
				}
			}
		}

		//Discard point 
		int count = 0;
		discard3Dpoint[jj] = false;
		for (int ii = 0; ii < nvisibles; ii++)
			if (Good[jj][ii] == true)
				count++;

		if (anchorId > -1 && fixedGlobalAnchor3D == 1 && count > 0) //let get it in even if it has only 1 view
			count = nViewsPlus;
		else
		{
			if (count < nViewsPlus)
			{
				discard3Dpoint[jj] = true;
				continue;
			}
		}

		//add 3D point and its 2D projections to Ceres
		bool once = true;
		int nValidProjectionPerPoint = 0;
		double pointErrX = 0.0, pointErrY = 0.0;
		for (int ii = 0; ii < nvisibles; ii++)
		{
			if (!Good[jj][ii])
				continue;

			nPossibleProjections++;
			uv = uvAll3D[jj][ii];
			viewID = viewIdAll3D[jj][ii];
			minvalidView = min(minvalidView, viewID);
			maxvalidView = max(maxvalidView, viewID);
			VisibleImages[viewID] = true;
			double wScale = scaleAll3D[jj][ii];
			if (weightByDistance == 1)
			{
				double dist = (Distance3D(&Vxyz[jj].x, &camera[viewID].camCenter[0]));
				wScale = sqrt(dist)*wScale / 1000.0;
			}
			double Weight = anchorId > -1 ? 10.0 : 1.0;

			if (sharedIntrinsicCamID.size() > 0 && sharedIntrinsicCamID[viewID] > -1 && sharedIntrinsicGroup[sharedIntrinsicCamID[viewID]])
			{
				refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[viewID]];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[refCam].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[refCam] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[refCam].intrinsic, camera[refCam].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[refCam].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[refCam].distortion);
						if (fixSkew)
						{
							if (fixSkewView[refCam] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[refCam].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[refCam] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[refCam].ShutterModel = GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uvAll3D[jj][ii].x, uvAll3D[jj][ii].y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[refCam].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[refCam].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[refCam].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (anchorId > -1 && fixedGlobalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
				if (anchorId == -1 && fixedLocalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}
			else
			{
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyDistortionReprojectionError::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
						if (fixPrism)
						{
							if (fixPrismView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(5), constant_parameters.push_back(6);
								problem.SetParameterization(camera[viewID].distortion, new ceres::SubsetParameterization(7, constant_parameters));
								fixPrismView[viewID] = 1;
							}
						}
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = FOVReprojectionError3::Create(uv.x, uv.y, wScale);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, &Vxyz[jj].x);
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						}
						else
						{
							ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = CayleyFOVReprojection2Error::Create(uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
							problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
						}

						if (fixIntrinsic)
							problem.SetParameterBlockConstant(camera[viewID].intrinsic);
						if (fixDistortion)
							problem.SetParameterBlockConstant(camera[viewID].distortion);
						if (fixSkew)
						{
							if (fixSkewView[viewID] == 0)
							{
								std::vector<int> constant_parameters;
								constant_parameters.push_back(2);
								problem.SetParameterization(camera[viewID].intrinsic, new ceres::SubsetParameterization(5, constant_parameters));
								fixSkewView[viewID] = 1;
							}
						}
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = PinholeReprojectionError::Create(uv.x, uv.y, wScale);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].intrinsic, camera[viewID].rt, &Vxyz[jj].x);
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					}
					else
					{
						ceres::LossFunction* weightedLoss = new ceres::ScaledLoss(loss_funcion, Weight, ceres::DO_NOT_TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = CayleyReprojectionError::Create(camera[viewID].intrinsic, uv.x, uv.y, wScale, camera[viewID].width, camera[viewID].height);
						problem.AddResidualBlock(cost_function, weightedLoss, camera[viewID].rt, camera[viewID].wt, &Vxyz[jj].x);
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}

				if (fixPose)
					problem.SetParameterBlockConstant(camera[viewID].rt);
				if (fixLocalPose)
					problem.SetParameterBlockConstant(camera[viewID].wt);
				if (fixFirstCamPose && viewID == firstValidViewID)
					problem.SetParameterBlockConstant(camera[firstValidViewID].rt);
				if (anchorId > -1 && fixedGlobalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
				if (anchorId == -1 && fixedLocalAnchor3D == 1)
					problem.SetParameterBlockConstant(&Vxyz[jj].x);
			}

			nValidProjectionPerPoint++;
			if (anchorId > -1)
				ReProjectionErrorXA.push_back(residuals[0]), ReProjectionErrorYA.push_back(residuals[1]);
			else
				ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);

			//pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
			//if (IsNumber(pointErrX) == 0 || IsNumber(pointErrY) == 0)
				//int a = 0;
			if (debug)
			{
				if (once)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.4f %.4f\n", residuals[0], residuals[1]);
			}
		}
		/*if ((anchorId > -1 && nValidProjectionPerPoint > 0) || nValidProjectionPerPoint >= nViewsPlus)
		{
			double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
			if (anchorId > -1)
				ReProjectionErrorXA.push_back(errX), ReProjectionErrorYA.push_back(errY);
			else
				ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
		}*/

		if (!once)
			fprintf(fp, "\n");
	}
	if (debug)
		fclose(fp);
	printLOG("\n");

	//Add pose smoothness
	vector<double> CamSmoothness;
	if (PoseSmoothness && validCamID.size() > 2)
	{
		sort(validCamID.begin(), validCamID.end());

		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2
					ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, 0.001*nPossibleProjections*nPossibleProjections / vCenterDif.size(), ceres::TAKE_OWNERSHIP); //such that both sum of reprojection2 and sum of smoothness2 are approx 1

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);

					problem.AddResidualBlock(cost_function1, ScaleLoss, camera[validCamID[ii]].rt, camera[validCamID[ii + 1]].rt);
				}
			}
		}
	}
	/*sprintf(Fname, "%s/Good.txt", Path); fp = fopen(Fname, "w+");
	for (int jj = 0; jj < npts; jj++)
	{
	fprintf(fp, "%d ", jj);
	for (int ii = 0; ii < Good[jj].size(); ii++)
	{
	if (Good[jj][ii] == false)
	fprintf(fp, "%d ", ii);
	}
	fprintf(fp, "-1\n");
	}
	fclose(fp);*/

	if (PoseSmoothness && CamSmoothness.size() > 0)
	{

		double minS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		printLOG("Camera center smoothness before BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f\n", minS, maxS, avgS, stdS);
	}

	double miniX, maxiX, avgX, stdX, miniY, maxiY, avgY, stdY;
	if (ReProjectionErrorXA.size() + ReProjectionErrorX.size() > 0)
		printLOG("(%d/%d) bad projections with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, nBadCounts + goodCount, maxOutlierX, maxOutlierY);
	else
		printLOG("Error. The BA gives 0 inliers!");
	if (ReProjectionErrorXA.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		maxiX = *max_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		avgX = MeanArray(ReProjectionErrorXA);
		stdX = sqrt(VarianceArray(ReProjectionErrorXA, avgX));
		miniY = *min_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		maxiY = *max_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		avgY = MeanArray(ReProjectionErrorYA);
		stdY = sqrt(VarianceArray(ReProjectionErrorYA, avgY));
		printLOG("%d anchor points with reprojection error before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorYA.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}
	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		printLOG("%d points with reprojection error before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();

	if (validCamID.size() < 1000)
	{
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.preconditioner_type = ceres::JACOBI;
		if (validCamID.size() < 300)
			options.function_tolerance = 1.0e-6;
		else
			options.function_tolerance = 1.0e-5;
		options.max_solver_time_in_seconds = 1000.0;
	}
	else
	{
		options.max_solver_time_in_seconds = 1.0 * (int)validCamID.size();
		if (validCamID.size() < 1500)
			options.max_num_iterations = 30;
		else if (validCamID.size() < 2000)
			options.max_num_iterations = 25;
		else if (validCamID.size() < 4000)
			options.max_num_iterations = 20;
		else if (validCamID.size() < 5000)
			options.max_num_iterations = 10;
		else
			options.max_num_iterations = 7;

		if (validCamID.size() < 1500)
			options.function_tolerance = 1.0e-5;
		if (validCamID.size() < 2000)
			options.function_tolerance = 5.0e-4;
		else if (validCamID.size() < 3000)
			options.function_tolerance = 1.0e-4;
		else
			options.function_tolerance = 5.0e-3;

		options.linear_solver_type = ceres::CGNR;
		options.preconditioner_type = ceres::JACOBI;
	}
	options.max_linear_solver_iterations = 100;
	options.minimizer_progress_to_stdout = silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;
	if (maxIter > -1)
		options.max_num_iterations = maxIter;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//if (silent)
	//printLOG("%s\n",summary.BriefReport().c_str());
	printLOG("%s\n", summary.BriefReport().c_str());
	//else
	//	std::cout << summary.FullReport();

	//Store refined parameters
	for (int ii = 0; ii < (int)validCamID.size() && sharedIntrinsicCamID.size() > 0; ii++)
	{
		int vid = validCamID[ii];
		if (sharedIntrinsicCamID[vid] > -1)
		{
			refCam = firstCameraInsharedIntrinsicGroup[sharedIntrinsicCamID[vid]];
			CopyCamereInfo(camera[refCam], camera[vid], false);
		}
	}
	for (int ii = 0; ii < (int)validCamID.size(); ii++)
	{
		GetKFromIntrinsic(camera[validCamID[ii]]);
		GetRTFromrt(camera[validCamID[ii]]);
		AssembleP(camera[validCamID[ii]].K, camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].P);
		GetCfromT(camera[validCamID[ii]].R, camera[validCamID[ii]].T, camera[validCamID[ii]].camCenter);
	}

	/*//this may skew thing up if increF was used before
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\nNo points were observed in views: ");
	break;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("%d.. ", camera[ii].frameID);
	camera[ii].valid = 0;
	for (int jj = 0; jj < 5; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 7; jj++)
	camera[ii].intrinsic[jj] = 0;
	for (int jj = 0; jj < 6; jj++)
	camera[ii].rt[jj] = 0, camera[ii].wt[jj] = 0;
	}
	}
	for (int ii = minvalidView; ii < maxvalidView; ii++)
	{
	if (!VisibleImages[ii])
	{
	printLOG("\n\n");
	break;
	}
	}*/

	for (int jj = 0; jj < npts; jj++)
	{
		int count = 0;
		for (size_t ii = 0; ii < Good[jj].size(); ii++)
			if (Good[jj][ii] == true)
				count++;
		if (VGlobalAnchor[jj] == -1 && count < nViewsPlus)
			Vxyz[jj] = Point3d(0, 0, 0);
	}
	if (!silent && threshold < 10) //double iteation, no need to display
		printLOG("\n");

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear(), ReProjectionErrorXA.clear(), ReProjectionErrorYA.clear();
	if (debug)
#ifdef _WINDOWS
		sprintf(Fname, "C:/temp/reprojectionA.txt"), fp = fopen(Fname, "w+");
#else
		sprintf(Fname, "reprojectionA.txt"), fp = fopen(Fname, "w+");
#endif

	for (int jj = 0; jj < npts; jj++)
	{
		if (abs(Vxyz[jj].x) + abs(Vxyz[jj].y) + abs(Vxyz[jj].z) > LIMIT3D && !discard3Dpoint[jj])
		{
			bool once = true;
			int nValidProjectionPerPoint = 0;
			double pointErrX = 0.0, pointErrY = 0.0;
			for (int ii = 0; ii < viewIdAll3D[jj].size(); ii++)
			{
				if (!Good[jj][ii] || uv.x < 0 || uv.y < 0)
					continue;

				viewID = viewIdAll3D[jj][ii];
				uv = uvAll3D[jj][ii];
				if (distortionCorrected == 0)
				{
					if (camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							PinholeDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyDistortionReprojectionDebug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
					else
					{
						if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
							FOVReprojectionDistortion2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, uv, Vxyz[jj], residuals);
						else
							CayleyFOVReprojection2Debug(camera[viewID].intrinsic, camera[viewID].distortion, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
					}
				}
				else
				{
					if (camera[viewID].ShutterModel == GLOBAL_SHUTTER)
						PinholeReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, uv, Vxyz[jj], residuals);
					else
						CayleyReprojectionDebug(camera[viewID].intrinsic, camera[viewID].rt, camera[viewID].wt, uv, Vxyz[jj], camera[viewID].width, camera[viewID].height, residuals);
				}

				nValidProjectionPerPoint++;
				if (VGlobalAnchor[jj] > -1)
					ReProjectionErrorXA.push_back(residuals[0]), ReProjectionErrorYA.push_back(residuals[1]);
				else
					ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
				//pointErrX += pow(residuals[0], 2), pointErrY += pow(residuals[1], 2);
				if (once && debug)
				{
					once = false;
					fprintf(fp, "%d %.4f %.4f %.4f ", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
				}
				if (debug)
					fprintf(fp, "V %d: %.4f %.4f %.4f %.4f ", viewID, uv.x, uv.y, residuals[0], residuals[1]);
				//fprintf(fp, "%.3f %.3f\n", residuals[0], residuals[1]);
			}
			if (!once &&debug)
				fprintf(fp, "\n");

			/*if (nValidProjectionPerPoint >= 1)
			{
				double errX = sqrt(pointErrX / nValidProjectionPerPoint), errY = sqrt(pointErrY / nValidProjectionPerPoint);
				if (VGlobalAnchor[jj] > -1)
					ReProjectionErrorXA.push_back(errX), ReProjectionErrorYA.push_back(errY);
				else
					ReProjectionErrorX.push_back(errX), ReProjectionErrorY.push_back(errY);
			}*/
		}
	}
	if (debug)
		fclose(fp);

	//Add pose smoothness
	CamSmoothness.clear();
	if (PoseSmoothness && validCamID.size() > 2)
	{
		vector<double> vCenterDif;
		for (int ii = 0; ii < validCamID.size() - 1; ii++)
		{
			int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
			int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
			if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				vCenterDif.push_back(norm(Point3d(camera[validCamID[ii]].camCenter[0] - camera[validCamID[ii + 1]].camCenter[0], camera[validCamID[ii]].camCenter[1] - camera[validCamID[ii + 1]].camCenter[1], camera[validCamID[ii]].camCenter[2] - camera[validCamID[ii + 1]].camCenter[2])));
		}
		if (vCenterDif.size() > 2)
		{
			sort(vCenterDif.begin(), vCenterDif.end());

			double sigma_iCenterVel = 1.0 / vCenterDif[vCenterDif.size() / 2];
			for (int ii = 0; ii < validCamID.size() - 1; ii++)
			{
				int viewID0 = camera[validCamID[ii]].viewID, viewID1 = camera[validCamID[ii + 1]].viewID;
				int frameID0 = camera[validCamID[ii]].frameID, frameID1 = camera[validCamID[ii + 1]].frameID;
				if (viewID0 == viewID1 && frameID0 + 1 == frameID1)
				{
					ceres::CostFunction* cost_function1 = LeastMotionPriorCostCameraCeres::CreateAutoDiff(frameID0, frameID1, sigma_iCenterVel); // (v/sigma)^2*dt) =  ((dX/dt)/sigma)^2*dt =  dX^2/dt/sigma^2

					vector<double *> paras; paras.push_back(camera[validCamID[ii]].rt), paras.push_back(camera[validCamID[ii + 1]].rt);
					cost_function1->Evaluate(&paras[0], residuals, NULL);
					for (int ii = 0; ii < 3; ii++)
						CamSmoothness.push_back(residuals[ii]);
				}
			}
		}
	}

	if (PoseSmoothness && CamSmoothness.size() > 0)
	{
		double miniS = *min_element(CamSmoothness.begin(), CamSmoothness.end()),
			maxiS = *max_element(CamSmoothness.begin(), CamSmoothness.end()),
			avgS = MeanArray(CamSmoothness),
			stdS = sqrt(VarianceArray(CamSmoothness, avgS));
		printLOG("Camera center smoothness after BA: Min: %.4f Max: %.4f Mean: %.4f Std: %.4f\n", miniS, maxiS, maxiX, avgS, stdS);
	}
	if (ReProjectionErrorXA.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		maxiX = *max_element(ReProjectionErrorXA.begin(), ReProjectionErrorXA.end());
		avgX = MeanArray(ReProjectionErrorXA);
		stdX = sqrt(VarianceArray(ReProjectionErrorXA, avgX));
		miniY = *min_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		maxiY = *max_element(ReProjectionErrorYA.begin(), ReProjectionErrorYA.end());
		avgY = MeanArray(ReProjectionErrorYA);
		stdY = sqrt(VarianceArray(ReProjectionErrorYA, avgY));
		printLOG("%d anchor points with reprojection error after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", ReProjectionErrorYA.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}
	if (ReProjectionErrorX.size() > 0)
	{
		miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
		avgX = MeanArray(ReProjectionErrorX);
		stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
		miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
		avgY = MeanArray(ReProjectionErrorY);
		stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
		printLOG("%d points with reprojection error after BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n\n", ReProjectionErrorX.size(), miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	delete[]discard3Dpoint;
	delete[]Good, delete[]VisibleImages;

	return 0;
}
//Video BA
int PerVideo_BA(char *Path, int selectedCamID, int startF, int stopF, int increF, int fixIntrinsic, int fixDistortion, int fix3D, int fixLocal3D, int fixPose, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int ShutterModel, int RobustLoss, int doubleRefinement, double threshold)
{
	char Fname[512];
	int nframes = stopF - startF + 1;
	printLOG("Working on camera %d:\n", selectedCamID);

	printLOG("Reading Pose\n");
	VideoData VideoInfoI;
	if (ReadVideoDataI(Path, VideoInfoI, selectedCamID, startF, stopF) == 1)
		return 1;
	for (int ii = 0; ii <= stopF; ii++)
		VideoInfoI.VideoInfo[ii].viewID = selectedCamID, VideoInfoI.VideoInfo[ii].frameID = ii,
		VideoInfoI.VideoInfo[ii].ShutterModel = ShutterModel; //over write the input model (tends to be global in PnP_Ransac) with the desisred one. 

	printLOG("Reading Corpus\n");
	Corpus CorpusInfo;

	bool usingLocalCorpus = true;
	Point3d xyz;
	Point3i rgb;
	int lpid, gpid, dummy;
	vector<int> Vgpid;
	sprintf(Fname, "%s/%d/Video3DCorpus.xyz", Path, selectedCamID); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %d %lf %lf %lf ", &lpid, &gpid, &xyz.x, &xyz.y, &xyz.z) != EOF)
		{
			if (lpid > CorpusInfo.xyz.size())
			{
				dummy = CorpusInfo.xyz.size();
				for (int ii = dummy; ii < lpid; ii++)
					CorpusInfo.xyz.push_back(Point3d(0, 0, 0)), Vgpid.push_back(-1);
			}

			Vgpid.push_back(gpid);
			CorpusInfo.xyz.push_back(xyz);
		}
		fclose(fp);

		CorpusInfo.n3dPoints = CorpusInfo.xyz.size();
	}
	else
	{
		int nPoints, useColor;
		sprintf(Fname, "%s/Corpus/Corpus_3D.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		usingLocalCorpus = false;
		fscanf(fp, "%d %d %d", &dummy, &nPoints, &useColor);
		CorpusInfo.n3dPoints = nPoints;
		CorpusInfo.xyz.reserve(CorpusInfo.n3dPoints);
		if (useColor)
		{
			CorpusInfo.rgb.reserve(CorpusInfo.n3dPoints);
			for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
			{
				fscanf(fp, "%lf %lf %lf %d %d %d", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z);
				Vgpid.push_back(jj);
				CorpusInfo.xyz.push_back(xyz);
				CorpusInfo.rgb.push_back(rgb);
			}
		}
		else
		{
			CorpusInfo.rgb.reserve(CorpusInfo.n3dPoints);
			for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
			{
				fscanf(fp, "%lf %lf %lf ", &xyz.x, &xyz.y, &xyz.z);
				CorpusInfo.xyz.push_back(xyz);
				Vgpid.push_back(jj);
			}
		}
		fclose(fp);
	}

	printLOG("Allocating memory\n");
	//Generate CorpusInfo.viewIdAll3D, CorpusInfo.uvAll3D, CorpusInfo.scaleAll3D
	vector<int> selectedCamID3D;
	vector<Point2d> uv3D;
	vector<double> scale3D;
	for (int ii = 0; ii < CorpusInfo.n3dPoints; ii++)
	{
		CorpusInfo.viewIdAll3D.push_back(selectedCamID3D); CorpusInfo.viewIdAll3D.back().reserve(nframes / increF / 50);
		CorpusInfo.uvAll3D.push_back(uv3D); CorpusInfo.uvAll3D.back().reserve(nframes / increF / 50);
		CorpusInfo.scaleAll3D.push_back(scale3D); CorpusInfo.scaleAll3D.back().reserve(nframes / increF / 50);
	}

	int increPer = 5, cPer = 5;
	Point2d uv; double s;
	printLOG("Reading input PnP: ");
	for (int fid = startF; fid <= stopF; fid += increF)
	{
		if (!VideoInfoI.VideoInfo[fid].valid)
			continue;

		sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, selectedCamID, fid);
		if (IsFileExist(Fname) == 1)
		{
			fp = fopen(Fname, "r");
			if (fp == NULL)
				continue;
			while (fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf ", &gpid, &lpid, &xyz.x, &xyz.y, &xyz.z, &uv.x, &uv.y, &s) != EOF)
			{
				if (lpid < 0 || lpid >CorpusInfo.n3dPoints)
					continue;
				if (s < 1.0)
					continue;
				if (VideoInfoI.VideoInfo[fid].valid)
				{
					CorpusInfo.viewIdAll3D[lpid].push_back(fid);
					CorpusInfo.uvAll3D[lpid].push_back(uv);
					CorpusInfo.scaleAll3D[lpid].push_back(s);
				}
			}
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/%d/PnPc/Inliers_%.4d.txt", Path, selectedCamID, fid);
			if (IsFileExist(Fname) == 1)
			{
				fp = fopen(Fname, "r");
				if (fp == NULL)
					continue;
				while (fscanf(fp, "%d %lf %lf %lf %f %f %lf", &gpid, &xyz.x, &xyz.y, &xyz.z, &uv.x, &uv.y, &s) != EOF)
				{
					if (gpid < 0 || gpid >CorpusInfo.n3dPoints)
						continue;
					if (s < 1.0)
						continue;
					if (VideoInfoI.VideoInfo[fid].valid)
					{
						CorpusInfo.viewIdAll3D[gpid].push_back(fid);
						CorpusInfo.uvAll3D[gpid].push_back(uv);
						CorpusInfo.scaleAll3D[gpid].push_back(s);
					}
				}
				fclose(fp);
			}
		}

		if (100 * (fid - startF) / (stopF - startF + 1) >= cPer)
		{
			printLOG("%d..", cPer);
			cPer += increPer;
		}
	}
	printLOG("100\n");

	//Find 3d points with less than nvisible views
	vector<int> NotOftenVisible;
	for (int pid = 0; pid < CorpusInfo.n3dPoints; pid++)
		if (CorpusInfo.viewIdAll3D[pid].size() < nViewsPlus)
			NotOftenVisible.push_back(pid);
	printLOG("(%d/%d) points not visible by at least %d frames\n", NotOftenVisible.size(), CorpusInfo.n3dPoints, nViewsPlus);

	//Clean from bottom to top
	for (int ii = (int)NotOftenVisible.size() - 1; ii >= 0; ii--)
	{
		int pid = NotOftenVisible[ii];
		CorpusInfo.viewIdAll3D[pid].erase(CorpusInfo.viewIdAll3D[pid].begin(), CorpusInfo.viewIdAll3D[pid].end());
		CorpusInfo.uvAll3D[pid].erase(CorpusInfo.uvAll3D[pid].begin(), CorpusInfo.uvAll3D[pid].end());
		CorpusInfo.scaleAll3D[pid].erase(CorpusInfo.scaleAll3D[pid].begin(), CorpusInfo.scaleAll3D[pid].end());
	}

	if (distortionCorrected == 1) //add distortion to get back the raw points, lets try to re-solve the calibration?
	{
		distortionCorrected = 0;
		for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			for (int ii = 0; ii < (int)CorpusInfo.uvAll3D[jj].size(); ii++)
			{
				int fid = CorpusInfo.viewIdAll3D[jj][ii];
				if (VideoInfoI.VideoInfo[fid].LensModel == RADIAL_TANGENTIAL_PRISM)
					LensDistortionPoint(&CorpusInfo.uvAll3D[jj][ii], VideoInfoI.VideoInfo[fid].K, VideoInfoI.VideoInfo[fid].distortion);
				else
					FishEyeDistortionPoint(&CorpusInfo.uvAll3D[jj][ii], VideoInfoI.VideoInfo[fid].K, VideoInfoI.VideoInfo[fid].distortion[0]);
			}
		}
	}

	vector<int> SharedIntrinsicFrames;
	sprintf(Fname, "%s/CamerasWithFixedIntrinsic2.txt", Path);
	if (IsFileExist(Fname) == 1)
	{
		int camID, count = 0;
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &camID) != EOF)
		{
			if (camID == selectedCamID)
			{
				printLOG("Shared-Intrinsic enforces for camera %d ", selectedCamID);
				for (int ii = 0; ii <= stopF; ii++)
					SharedIntrinsicFrames.push_back(0);
				break;
			}
		}
		fclose(fp);
	}

	int fixLocalPose = 0, fixCamFramePose = 0;
	for (int ii = 0; ii <= stopF; ii++)
		VideoInfoI.VideoInfo[ii].ShutterModel = ShutterModel;

	if (usingLocalCorpus) //fix3D is applied only to global corpus
	{
		for (int ii = 0; ii <= stopF; ii++)
			VideoInfoI.VideoInfo[ii].threshold = doubleRefinement == 0 ? threshold : 3.0 * threshold; //make sure that most points are inliers
		GenericBundleAdjustment(Path, VideoInfoI.VideoInfo, CorpusInfo.xyz, CorpusInfo.viewIdAll3D, CorpusInfo.uvAll3D, CorpusInfo.scaleAll3D,
			SharedIntrinsicFrames, stopF + 1, fixIntrinsic, fixDistortion, fixPose, fixCamFramePose, fixLocalPose, Vgpid, fix3D, fixLocal3D, fixSkew, fixPrism, distortionCorrected, nViewsPlus, RobustLoss, true, false, true);

		if (doubleRefinement > 0)
		{
			for (int ii = 0; ii <= stopF; ii++)
				VideoInfoI.VideoInfo[ii].threshold = threshold;
			GenericBundleAdjustment(Path, VideoInfoI.VideoInfo, CorpusInfo.xyz, CorpusInfo.viewIdAll3D, CorpusInfo.uvAll3D, CorpusInfo.scaleAll3D,
				SharedIntrinsicFrames, stopF + 1, fixIntrinsic, fixDistortion, fixPose, fixCamFramePose, fixLocalPose, Vgpid, fix3D, fixLocal3D, fixSkew, fixPrism, distortionCorrected, nViewsPlus, RobustLoss, true, false, true);
		}
	}
	else //fixed3D is applied globally
	{
		for (int ii = 0; ii <= stopF; ii++)
			VideoInfoI.VideoInfo[ii].threshold = doubleRefinement == 0 ? threshold : 3.0 * threshold; //make sure that most points are inliers
		GenericBundleAdjustment(Path, VideoInfoI.VideoInfo, CorpusInfo.xyz, CorpusInfo.viewIdAll3D, CorpusInfo.uvAll3D, CorpusInfo.scaleAll3D,
			SharedIntrinsicFrames, stopF + 1, fixIntrinsic, fixDistortion, fixPose, fixCamFramePose, fixLocalPose, fix3D, fixSkew, fixPrism, distortionCorrected, nViewsPlus, RobustLoss, true, false, true);

		if (doubleRefinement > 0)
		{
			for (int ii = 0; ii <= stopF; ii++)
				VideoInfoI.VideoInfo[ii].threshold = threshold;
			GenericBundleAdjustment(Path, VideoInfoI.VideoInfo, CorpusInfo.xyz, CorpusInfo.viewIdAll3D, CorpusInfo.uvAll3D, CorpusInfo.scaleAll3D,
				SharedIntrinsicFrames, stopF + 1, fixIntrinsic, fixDistortion, fixPose, fixCamFramePose, fixLocalPose, fix3D, fixSkew, fixPrism, distortionCorrected, nViewsPlus, RobustLoss, true, false, true);
		}
	}
	sprintf(Fname, "%s/Good.txt", Path), remove(Fname);

	//write video data
	printLOG("Writing refined poses ....");
	vector<int> computedTime;
	for (int ii = startF; ii <= stopF; ii++)
		if (VideoInfoI.VideoInfo[ii].valid)
			computedTime.push_back(ii);

	sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, selectedCamID);	SaveVideoCameraIntrinsic(Fname, VideoInfoI.VideoInfo, computedTime, selectedCamID, 0);
	sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, selectedCamID);	SaveVideoCameraPoses(Fname, VideoInfoI.VideoInfo, computedTime, selectedCamID, 0);

	if (usingLocalCorpus)
	{
		sprintf(Fname, "%s/%d/vVideo3DCorpus.xyz", Path, selectedCamID); FILE *fp = fopen(Fname, "w+");
		for (size_t kk = 0; kk < CorpusInfo.xyz.size(); kk++)
			if (IsValid3D(CorpusInfo.xyz[kk]))
				fprintf(fp, "%d %d %f %f %f\n", kk, Vgpid[kk], CorpusInfo.xyz[kk].x, CorpusInfo.xyz[kk].y, CorpusInfo.xyz[kk].z);
		fclose(fp);
	}
	printLOG("Done!\n");

	return 0;
}
int AllVideo_BA(char *Path, int nCams, int startF, int stopF, int increF, int fixIntrinsic, int fixDistortion, int fixPose, int fixfirstCamPose, int fix3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int ShutterModel, int RobustLoss, int doubleRefinement, double threshold)
{
	char Fname[512];
	int nframes = stopF - startF + 1;
	printLOG("Working on all %d cameras\n", nCams);
	printLOG("********NOTE: ONLY SUPPORT GLOBAL CORPUS FOR NOW!********\n");

	VideoData AllVideoInfo;
	if (ReadVideoData(Path, AllVideoInfo, nCams, startF, stopF) == 1)
		return 1;
	for (int cid = 0; cid < nCams; cid++)
	{
		int videoID = AllVideoInfo.nframesI*cid;
		for (int fid = startF; fid <= stopF; fid++)
			AllVideoInfo.VideoInfo[fid + videoID].viewID = cid,
			AllVideoInfo.VideoInfo[fid + videoID].frameID = fid,
			AllVideoInfo.VideoInfo[videoID + fid].ShutterModel = ShutterModel;
	}

	int nCorpusCams, n3DPoints, useColor;
	sprintf(Fname, "%s/Corpus/nCorpus_3D.txt", Path);
	if (IsFileExist(Fname) == 0)
		sprintf(Fname, "%s/Corpus/Corpus_3D.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	printLOG("Loaded %s\n", Fname);
	fscanf(fp, "%d %d %d", &nCorpusCams, &n3DPoints, &useColor);

	Point3d xyz;	Point3i rgb;
	vector<Point3d> Vxyz; 	Vxyz.reserve(n3DPoints);
	vector<Point3i> Vrgb; 	Vrgb.reserve(n3DPoints);
	if (useColor)
	{
		Vrgb.reserve(n3DPoints);
		for (int jj = 0; jj < n3DPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf %d %d %d", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z);
			Vxyz.push_back(xyz);
			Vrgb.push_back(rgb);
		}
	}
	else
	{
		Vrgb.reserve(n3DPoints);
		for (int jj = 0; jj < n3DPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf ", &xyz.x, &xyz.y, &xyz.z);
			Vxyz.push_back(xyz);
		}
	}
	fclose(fp);

	int fixLocalPose = 0, fixLocalAnchor = 0;
	vector<int> VGlobalAnchor;
	for (int ii = 0; ii < Vxyz.size(); ii++)
		VGlobalAnchor.push_back(ii);

	//Generate CorpusInfo.viewIdAll3D, CorpusInfo.uvAll3D, CorpusInfo.scaleAll3D
	printLOG("Prepare data storage...");
	vector < vector<int> > viewIdAll3D; vector<int> selectedCamID3D;
	vector<vector<Point2f> > uvAll3D;	vector<Point2f> uv3D;
	vector<vector<float> > scaleAll3D; vector<float> scale3D;
	for (int ii = 0; ii < n3DPoints; ii++)
	{
		viewIdAll3D.push_back(selectedCamID3D); viewIdAll3D.back().reserve(nframes / increF * nCams / 12);
		uvAll3D.push_back(uv3D); uvAll3D.back().reserve(nframes / increF * nCams / 12);
		scaleAll3D.push_back(scale3D); scaleAll3D.back().reserve(nframes / increF * nCams / 12);
	}

	printLOG("start reading with step %d....", increF);
	int pid; float s;
	Point2f uv;
	for (int camID = 0; camID < nCams; camID++)
	{
		printLOG("%d ..", camID);
		int videoID = AllVideoInfo.nframesI*camID;
		for (int fid = startF; fid <= stopF; fid += increF)
		{
			sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, camID, fid);
			if (IsFileExist(Fname) == 1)
			{
				fp = fopen(Fname, "r");
				while (fscanf(fp, "%d %d %lf %lf %lf %f %f %f ", &pid, &pid, &xyz.x, &xyz.y, &xyz.z, &uv.x, &uv.y, &s) != EOF)
				{
					if (pid < 0 || pid >n3DPoints)
						continue;
					if (s < 1.0)
						continue;

					if (AllVideoInfo.VideoInfo[fid + videoID].valid == 1)
					{
						viewIdAll3D[pid].push_back(fid + videoID);
						uvAll3D[pid].push_back(uv);
						scaleAll3D[pid].push_back(s);
					}
				}
				fclose(fp);
			}
			else
			{
				sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, camID, fid);
				if (IsFileExist(Fname) == 1)
				{
					fp = fopen(Fname, "r");
					while (fscanf(fp, "%d %lf %lf %lf %f %f %f", &pid, &xyz.x, &xyz.y, &xyz.z, &uv.x, &uv.y, &s) != EOF)
					{
						if (pid < 0 || pid > n3DPoints)
							continue;
						if (s < 1.0)
							continue;
						if (AllVideoInfo.VideoInfo[fid + videoID].valid == 1)
						{
							viewIdAll3D[pid].push_back(fid + videoID);
							uvAll3D[pid].push_back(uv);
							scaleAll3D[pid].push_back(s);
						}
					}
					fclose(fp);
				}
			}
		}
	}
	printLOG("done!\n");

	if (distortionCorrected == 1)
	{
		distortionCorrected = 0;
		for (int jj = 0; jj < n3DPoints; jj++)
		{
			for (int ii = 0; ii < (int)uvAll3D[jj].size(); ii++)
			{
				int id = viewIdAll3D[jj][ii];
				Point2d uv = uvAll3D[jj][ii];
				if (AllVideoInfo.VideoInfo[id].LensModel == RADIAL_TANGENTIAL_PRISM)
					LensDistortionPoint(&uv, AllVideoInfo.VideoInfo[id].K, AllVideoInfo.VideoInfo[id].distortion);
				else
					FishEyeDistortionPoint(&uv, AllVideoInfo.VideoInfo[id].K, AllVideoInfo.VideoInfo[id].distortion[0]);
				uvAll3D[jj][ii] = uv;
			}
		}
	}

	vector<int> SharedCameraToBuildCorpus;//size must be equal to the size of AllVideoInfo
	for (int ii = 0; ii < nCams*AllVideoInfo.nframesI; ii++)
		SharedCameraToBuildCorpus.push_back(-1);

	sprintf(Fname, "%s/CamerasWithFixedIntrinsic2.txt", Path);
	if (IsFileExist(Fname) == 1)
	{
		int camID, count = 0;
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &camID) != EOF)
		{
			printLOG("Shared-Intrinsic enforces for camera %d ", camID);
			for (int ii = 0; ii < AllVideoInfo.nframesI; ii++)
				SharedCameraToBuildCorpus[camID*AllVideoInfo.nframesI + ii] = camID;
		}
		fclose(fp);
	}

	for (int camID = 0; camID < nCams; camID++)
	{
		int videoID = AllVideoInfo.nframesI*camID;
		for (int ii = startF; ii < stopF; ii++)
			AllVideoInfo.VideoInfo[ii + videoID].threshold = !doubleRefinement > 0 ? threshold : threshold * 2.0; //make sure that most points are inliers
	}
	GenericBundleAdjustment(Path, AllVideoInfo.VideoInfo, Vxyz, viewIdAll3D, uvAll3D, scaleAll3D,
		SharedCameraToBuildCorpus, nframes*nCams, fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, 0, VGlobalAnchor, fix3D, fixLocalAnchor, fixSkew, fixPrism, distortionCorrected, nViewsPlus, RobustLoss, false, false, false);

	if (doubleRefinement > 0)
	{
		for (int camID = 0; camID < nCams; camID++)
		{
			int videoID = AllVideoInfo.nframesI*camID;
			for (int ii = startF; ii < stopF; ii++)
				AllVideoInfo.VideoInfo[ii + videoID].threshold = threshold;
		}
		GenericBundleAdjustment(Path, AllVideoInfo.VideoInfo, Vxyz, viewIdAll3D, uvAll3D, scaleAll3D,
			SharedCameraToBuildCorpus, nframes*nCams, fixIntrinsic, fixDistortion, fixPose, fixfirstCamPose, 0, VGlobalAnchor, fix3D, fixLocalAnchor, fixSkew, fixPrism, distortionCorrected, nViewsPlus, RobustLoss, false, false, false);
	}
	sprintf(Fname, "%s/Good.txt", Path), remove(Fname);

	//write video data
	printLOG("Writing refined poses ....");
	for (int camID = 0; camID < nCams; camID++)
	{
		int videoID = AllVideoInfo.nframesI*camID;
		vector<int> computedTime;
		for (int ii = startF; ii <= stopF; ii += increF)
			if (AllVideoInfo.VideoInfo[ii + videoID].valid)
				computedTime.push_back(ii);
		sprintf(Fname, "%s/avIntrinsic_%.4d.txt", Path, camID);	SaveVideoCameraIntrinsic(Fname, &AllVideoInfo.VideoInfo[videoID], computedTime, camID, 0);
		sprintf(Fname, "%s/avCamPose_%.4d.txt", Path, camID);	SaveVideoCameraPoses(Fname, &AllVideoInfo.VideoInfo[videoID], computedTime, camID, 0);
	}
	printLOG("done\n");

	if (fix3D == 0)
	{
		printLOG("ReSave corpus 3D points ...");
		sprintf(Fname, "%s/Corpus/nCorpus_3D.txt", Path);	fp = fopen(Fname, "w+");
		fprintf(fp, "%d %d ", nCorpusCams, n3DPoints);
		if (Vrgb.size() == 0)
		{
			fprintf(fp, "0\n");
			for (int jj = 0; jj < (int)Vxyz.size(); jj++)
				fprintf(fp, "%e %e %e\n", Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
		}
		else
		{
			fprintf(fp, "1\n");
			for (int jj = 0; jj < (int)Vxyz.size(); jj++)
				fprintf(fp, "%e %e %e %d %d %d\n", Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z, Vrgb[jj].x, Vrgb[jj].y, Vrgb[jj].z);
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus/n3dGL.xyz", Path);	fp = fopen(Fname, "w+");
		if (Vrgb.size() == 0)
			for (int jj = 0; jj < (int)Vxyz.size(); jj++)
				fprintf(fp, "%d %e %e %e\n", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z);
		else
			for (int jj = 0; jj < (int)Vxyz.size(); jj++)
				fprintf(fp, "%d %e %e %e %d %d %d\n", jj, Vxyz[jj].x, Vxyz[jj].y, Vxyz[jj].z, Vrgb[jj].x, Vrgb[jj].y, Vrgb[jj].z);
		fclose(fp);
		printLOG("done\n");
	}

	return 0;
}

void Virtual3D_RS_ReProjectionEror(double *intrinsic, double *rt1, double * rt2, double *point, double rp, Point2d obser2D, int height, double *residuals)
{
	//transform to camera coord
	double pi[3], p1[3], p2[3];
	ceres::AngleAxisRotatePoint(rt1, point, p1);
	ceres::AngleAxisRotatePoint(rt2, point, p2);

	p1[0] += rt1[3], p1[1] += rt1[4], p1[2] += rt1[5];
	p2[0] += rt2[3], p2[1] += rt2[4], p2[2] += rt2[5];

	//Time of the observation.
	double t = obser2D.y / height - 0.5;
	if (t > 0)
		t = rp * t; //lie on the 1st frame
	else
		t = rp * t + 1.0; //lie on the 2nd frame

						  //Interpolated point
	for (int ii = 0; ii < 3; ii++)
		pi[ii] = (1.0 - t)*p1[ii] + t * p2[ii];

	// Project to normalize coordinate
	double xcn = pi[0] / pi[2], ycn = pi[1] / pi[2];

	// Compute final projected point position.
	double predicted_x = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3];
	double predicted_y = intrinsic[1] * ycn + intrinsic[4];

	// The error is the difference between the predicted and observed position.
	residuals[0] = predicted_x - obser2D.x;
	residuals[1] = predicted_y - obser2D.y;

	return;
}
int Virtual3D_RS_BA(double &rollingshutter_Percent, double *intrinsics, double *rt, bool *valid, double *Vxyz, vector<vector<int> > &fidPer3D, vector<vector<Point2d> > &uvPer3D, vector<vector<float> > &sPer3D, int imgheight, int nframes, int LossType, bool silent)
{
	//input 2D must be distortion corrected
	int npts = (int)fidPer3D.size();

	printLOG("set up Virtual 3D RS-BA ...\n");
	ceres::Problem problem;

	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(2.0);

	double residuals[2], maxOutlierX = 0.0, maxOutlierY = 0.0;
	vector<int> validFrameID;
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve(npts * nframes);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve(npts * nframes);

	for (int jj = 0; jj < npts; jj++)
	{
		for (int ii = 1; ii < (int)uvPer3D[jj].size() - 1; ii++)
		{
			int fid = fidPer3D[jj][ii];
			Point2d p2d = uvPer3D[jj][ii];

			int type;
			if (p2d.y < imgheight / 2)
			{
				type = 0;
				if (valid[fid] != 1 || valid[fid - 1] != 1)
					continue;

				Virtual3D_RS_ReProjectionEror(&intrinsics[5 * fid], &rt[6 * (fid - 1)], &rt[6 * fid], &Vxyz[3 * jj], rollingshutter_Percent, p2d, imgheight, residuals);
			}
			else
			{
				type = 1;
				if (valid[fid] != 1 || valid[fid + 1] != 1)
					continue;

				Virtual3D_RS_ReProjectionEror(&intrinsics[5 * fid], &rt[6 * fid], &rt[6 * (fid + 1)], &Vxyz[3 * jj], rollingshutter_Percent, p2d, imgheight, residuals);
			}

			double wScale = sPer3D[jj][ii];
			ceres::CostFunction* cost_function = Virtual3D_RS_ReProjectionEror::Create(&intrinsics[5 * fid], uvPer3D[jj][ii].x, uvPer3D[jj][ii].y, imgheight, wScale);
			if (type == 0)
				problem.AddResidualBlock(cost_function, loss_funcion, &rt[6 * (fid - 1)], &rt[6 * fid], &Vxyz[3 * jj], &rollingshutter_Percent);
			else
				problem.AddResidualBlock(cost_function, loss_funcion, &rt[6 * fid], &rt[6 * (fid + 1)], &Vxyz[3 * jj], &rollingshutter_Percent);

			ReProjectionErrorX.push_back(residuals[0]);
			ReProjectionErrorY.push_back(residuals[1]);
			validFrameID.push_back(fid);
		}
	}

	double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double avgX = MeanArray(ReProjectionErrorX);
	double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double avgY = MeanArray(ReProjectionErrorY);
	double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
	printLOG("Reprojection error before BA:\nMin: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();

	std::sort(validFrameID.begin(), validFrameID.end());
	validFrameID.erase(std::unique(validFrameID.begin(), validFrameID.end()), validFrameID.end());

	if (validFrameID.size() < 3000)
		options.linear_solver_type = ceres::DENSE_SCHUR;
	else
	{
		options.linear_solver_type = ceres::CGNR;
		options.preconditioner_type = ceres::JACOBI;
	}

	if (validFrameID.size() < 1500)
		options.max_num_iterations = 50, options.function_tolerance = 1.0e-5;
	else if (validFrameID.size() < 2000)
		options.max_num_iterations = 35, options.function_tolerance = 5.0e-4;
	else if (validFrameID.size() < 4000)
		options.max_num_iterations = 25, options.function_tolerance = 1.0e-4;
	else if (validFrameID.size() < 5000)
		options.max_num_iterations = 15, options.function_tolerance = 5.0e-3;
	else
		options.max_num_iterations = 10, options.function_tolerance = 5.0e-3;

	options.minimizer_progress_to_stdout = silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (silent)
		std::cout << summary.BriefReport();
	else
		std::cout << summary.FullReport();

	printLOG("\n\nRS %%: %.4f\n\n", rollingshutter_Percent);

	ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
	for (int jj = 0; jj < npts; jj++)
	{
		for (int ii = 1; ii < (int)uvPer3D[jj].size() - 1; ii++)
		{
			int fid = fidPer3D[jj][ii];
			Point2d p2d = uvPer3D[jj][ii];

			int type;
			if (p2d.y < imgheight / 2)
			{
				type = 0;
				Virtual3D_RS_ReProjectionEror(&intrinsics[5 * fid], &rt[6 * (fid - 1)], &rt[6 * fid], &Vxyz[3 * jj], rollingshutter_Percent, p2d, imgheight, residuals);
			}
			else
			{
				type = 1;
				Virtual3D_RS_ReProjectionEror(&intrinsics[5 * fid], &rt[6 * fid], &rt[6 * (fid + 1)], &Vxyz[3 * jj], rollingshutter_Percent, p2d, imgheight, residuals);
			}

			ReProjectionErrorX.push_back(residuals[0]), ReProjectionErrorY.push_back(residuals[1]);
		}
	}

	if (ReProjectionErrorX.size() == 0 || ReProjectionErrorY.size() == 0)
		printLOG("Error. The BA gives 0 inliers!");
	miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	avgX = MeanArray(ReProjectionErrorX);
	stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	avgY = MeanArray(ReProjectionErrorY);
	stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));
	printLOG("Reprojection error after BA:\nMin: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	return 0;
}

//BA ulti
void RollingShutterSplineProjection(double *intrinsic, int *ActingID, double *ActingControlPose, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, int se3, Point2d &predicted, Point3d &point, int frameID, int width, int height)
{
	double R[9], T[3], twist[6], tr[6], np[3], p[3] = { point.x, point.y, point.z };

	double *Bi = new double[nCtrl];

	//Get initial estimate of the projected location
	double subframeLoc = 0.5 + frameID;
	if (subframeLoc < KnotLoc[SplineOrder - 1])
		subframeLoc = KnotLoc[SplineOrder - 1] + 0.5;
	else if (subframeLoc > KnotLoc[nCtrl])
		subframeLoc = KnotLoc[nCtrl] - 0.5;

	BSplineGetBasis(subframeLoc, Bi, KnotLoc, nBreak, nCtrl, SplineOrder);

	int nlocalControls = SplineOrder + 2;
	for (int jj = 0; jj < 6; jj++)
	{
		/*for (int ii = 0; ii < nCtrl; ii++)
		{
		double bi = Bi[ii];
		if (bi < 1.0e-6)
		continue;
		int found = 0, foundActingID = 0;
		for (int kk = 0; kk < 6; kk++)
		if (ActingID[kk] == ii)
		{
		foundActingID = kk;
		found++;
		break;
		}
		if (found == 0)
		printLOG("CP problem @Frame %d \n", frameID);

		twist[jj] += ActingControlPose[jj + 6 * foundActingID] * bi;
		}*/
		if (se3)
		{
			twist[jj] = 0.0;
			for (int ii = 0; ii < nlocalControls; ii++)
				twist[jj] += ActingControlPose[jj + nlocalControls * ii] * Bi[ActingID[ii]];
		}
		else
		{
			tr[jj] = 0;
			for (int ii = 0; ii < nlocalControls; ii++)
				tr[jj] += ActingControlPose[jj + nlocalControls * ii] * Bi[ActingID[ii]];
		}
	}

	if (se3)
		getRTFromTwist(twist, R, T);
	else
	{
		getRfromr(tr + 3, R);
		for (int ii = 0; ii < 3; ii++)
			T[ii] = tr[ii];
	}

	np[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1];
	np[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2];
	double ycn = np[1] / np[2], ycn_ = ycn;
	double v = intrinsic[1] * ycn + intrinsic[4]; //to get time info

												  //Iteratively solve for ycn = P(ycn)*X
	int iter, iterMax = 20;
	double dif;
	for (iter = 0; iter < iterMax; iter++)
	{
		subframeLoc = v / height + frameID;
		if (subframeLoc < KnotLoc[SplineOrder - 1])
			subframeLoc = KnotLoc[SplineOrder - 1] + 0.5;
		else if (subframeLoc > KnotLoc[nCtrl])
			subframeLoc = KnotLoc[nCtrl] - 0.5;

		BSplineGetBasis(subframeLoc, Bi, KnotLoc, nBreak, nCtrl, SplineOrder);

		for (int jj = 0; jj < 6; jj++)
		{
			if (se3)
			{
				twist[jj] = 0.0;
				for (int ii = 0; ii < nlocalControls; ii++)
					twist[jj] += ActingControlPose[jj + nlocalControls * ii] * Bi[ActingID[ii]];
			}
			else
			{
				tr[jj] = 0;
				for (int ii = 0; ii < nlocalControls; ii++)
					tr[jj] += ActingControlPose[jj + nlocalControls * ii] * Bi[ActingID[ii]];
			}
		}

		if (se3)
			getRTFromTwist(twist, R, T);
		else
		{
			getRfromr(tr + 3, R);
			for (int ii = 0; ii < 3; ii++)
				T[ii] = tr[ii];
		}

		np[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1];
		np[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2];

		ycn = np[1] / np[2];
		v = intrinsic[1] * ycn + intrinsic[4];
		dif = abs((ycn - ycn_) / ycn_);
		if (dif < 1.0e-9)
			break;
		ycn_ = ycn;
	}

	//if (v<-1.0 || v>height)
	//	printLOG("Projection problem @Frame %d (%.2f)\n", frameID, v);

	np[0] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + T[0];
	double xcn = np[0] / np[2], u = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3];
	predicted.x = u, predicted.y = v;

	//if (iter > iterMax - 1 && dif > 1.0e-6)
	//	printLOG("Frame %d: %.2f %.2f %.9e \n", frameID, u, v, dif);

	delete[]Bi;

	return;
}
void RollingShutterSplineReprojectionDebug(double *intrinsic, int *ActingID, double *ActingControlPose, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, int se3, Point2d &observed, Point3d &point, int frameID, int width, int height, double *residuals)
{
	Point2d predicted;
	RollingShutterSplineProjection(intrinsic, ActingID, ActingControlPose, KnotLoc, nBreak, nCtrl, SplineOrder, se3, predicted, point, frameID, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return;
}
void RollingShutterDistortionSplineProjection(double *intrinsic, double *distortion, int *ActingID, double *ActingControlPose, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, Point2d &predicted, Point3d point, int frameID, int width, int height)
{
	double R[9], T[3], twist[6], np[3], p[3] = { point.x, point.y, point.z };
	double *Bi = new double[nCtrl];

	//Get initial estimate of the projected location
	double subframeLoc = 0.5 + frameID;
	if (subframeLoc < KnotLoc[SplineOrder - 1])
		subframeLoc = KnotLoc[SplineOrder - 1] + 0.5;
	else if (subframeLoc > KnotLoc[nCtrl])
		subframeLoc = KnotLoc[nCtrl] - 0.5;

	BSplineGetBasis(subframeLoc, Bi, KnotLoc, nBreak, nCtrl, SplineOrder);

	int nlocalControls = SplineOrder + 2;
	for (int jj = 0; jj < 6; jj++)
	{
		twist[jj] = 0.0;
		for (int ii = 0; ii < nlocalControls; ii++)
			twist[jj] += ActingControlPose[jj + nlocalControls * ii] * Bi[ActingID[ii]];
	}

	getRTFromTwist(twist, R, T);
	np[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1];
	np[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2];
	double ycn = np[1] / np[2], ycn_ = ycn;
	double v = intrinsic[1] * ycn + intrinsic[4]; //to get time info

												  //Iteratively solve for ycn = P(ycn)*X
	for (int iter = 0; iter < 40; iter++)
	{
		subframeLoc = 0.5 + frameID;
		if (subframeLoc < KnotLoc[SplineOrder - 1])
			subframeLoc = KnotLoc[SplineOrder - 1] + 0.5;
		else if (subframeLoc > KnotLoc[nCtrl])
			subframeLoc = KnotLoc[nCtrl] - 0.5;

		BSplineGetBasis(subframeLoc, Bi, KnotLoc, nBreak, nCtrl, SplineOrder);

		for (int jj = 0; jj < 6; jj++)
		{
			twist[jj] = 0.0;
			for (int ii = 0; ii < nlocalControls; ii++)
				twist[jj] += ActingControlPose[jj + nlocalControls * ii] * Bi[ActingID[ii]];
		}

		getRTFromTwist(twist, R, T);
		np[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1];
		np[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2];

		ycn = np[1] / np[2];
		v = intrinsic[1] * ycn + intrinsic[4];
		if (abs((ycn - ycn_) / ycn_) < 1.0e-9)
			break;
		ycn_ = ycn;
	}

	np[0] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + T[0];
	double xcn = np[0] / np[2], u = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3];

	predicted.x = u, predicted.y = v;
	LensDistortionPoint2(&predicted, intrinsic, distortion);

	delete[]Bi;

	return;
}
void RollingShutterDistortionSplineReprojectionDebug(double *intrinsic, double *distortion, int *ActingID, double *ActingControlPose, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, Point2d observed, Point3d point, int frameID, int width, int height, double *residuals)
{
	Point2d predicted;
	RollingShutterDistortionSplineProjection(intrinsic, distortion, ActingID, ActingControlPose, KnotLoc, nBreak, nCtrl, SplineOrder, predicted, point, frameID, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return;
}
void RollingShutterDCTProjection(double *intrinsic, double *Coeffs0, double *Coeffs1, double *Coeffs2, double *Coeffs3, double *Coeffs4, double *Coeffs5, int nCoeffs, Point2d &predicted, Point3d &point, int frameID, int width, int height)
{
	double R[9], T[3], twist[6], np[3], p[3] = { point.x, point.y, point.z };
	double *iB = new double[nCoeffs];

	//Get initial estimate of the projected location: must be in 0->n-1 range
	double subframeLoc = 0.5 + frameID;
	if (subframeLoc > nCoeffs - 1)
		subframeLoc = nCoeffs - 1;

	//Get twist = iB*C;
	GenerateiDCTBasis(iB, nCoeffs, subframeLoc);

	for (int jj = 0; jj < 6; jj++)
		twist[jj] = 0.0;
	for (int ii = 0; ii < nCoeffs; ii++)
	{
		twist[0] += Coeffs0[ii] * iB[ii];
		twist[1] += Coeffs1[ii] * iB[ii];
		twist[2] += Coeffs2[ii] * iB[ii];
		twist[3] += Coeffs3[ii] * iB[ii];
		twist[4] += Coeffs4[ii] * iB[ii];
		twist[5] += Coeffs5[ii] * iB[ii];
	}
	getRTFromTwist(twist, R, T);

	//Initiate projection solver
	np[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1];
	np[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2];
	double ycn = np[1] / np[2], ycn_ = ycn;
	double v = intrinsic[1] * ycn + intrinsic[4]; //to get time info

												  //Iteratively solve for ycn = P(ycn)*X
	int iter, iterMax = 20;
	double dif;
	for (iter = 0; iter < iterMax; iter++)
	{
		subframeLoc = v / height + frameID;
		if (subframeLoc > nCoeffs)
			subframeLoc = nCoeffs;

		GenerateiDCTBasis(iB, nCoeffs, subframeLoc);

		for (int jj = 0; jj < 6; jj++)
			twist[jj] = 0.0;
		for (int ii = 0; ii < nCoeffs; ii++)
		{
			twist[0] += Coeffs0[ii] * iB[ii];
			twist[1] += Coeffs1[ii] * iB[ii];
			twist[2] += Coeffs2[ii] * iB[ii];
			twist[3] += Coeffs3[ii] * iB[ii];
			twist[4] += Coeffs4[ii] * iB[ii];
			twist[5] += Coeffs5[ii] * iB[ii];
		}
		getRTFromTwist(twist, R, T);

		np[1] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1];
		np[2] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2];

		ycn = np[1] / np[2];
		v = intrinsic[1] * ycn + intrinsic[4];
		dif = abs((ycn - ycn_) / ycn_);
		if (dif < 1.0e-9)
			break;
		ycn_ = ycn;
	}

	//if (v<-1.0 || v>height)
	//	printLOG("Projection problem @Frame %d (%.2f)\n", frameID, v);

	np[0] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + T[0];
	double xcn = np[0] / np[2], u = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3];
	predicted.x = u, predicted.y = v;

	//if (iter > iterMax - 1 && dif > 1.0e-6)
	//	printLOG("Frame %d: %.2f %.2f %.9e \n", frameID, u, v, dif);

	delete[]iB;

	return;
}
void RollingShutterDCTReprojectionDebug(double *intrinsic, double *Coeffs0, double *Coeffs1, double *Coeffs2, double *Coeffs3, double *Coeffs4, double *Coeffs5, int nCoeffs, Point2d &observed, Point3d &point, int frameID, int width, int height, double *residuals)
{
	Point2d predicted;
	RollingShutterDCTProjection(intrinsic, Coeffs0, Coeffs1, Coeffs2, Coeffs3, Coeffs4, Coeffs5, nCoeffs, predicted, point, frameID, width, height);
	residuals[0] = predicted.x - observed.x, residuals[1] = predicted.y - observed.y;

	return;
}

int VideoSplineRSBA(char *Path, int startF, int stopF, int selectedCams, int fixIntrinsic, int fixDistortion, int fix3D, int fixSkew, int fixPrism, int distortionCorrected, double threshold, int controlStep, int SplineOrder, int se3, bool debug)
{
	//SplineOrder:  4 (cubic spline)
	if (se3 == 1)
		printLOG("Using se(3) parameterization\n");
	else
		printLOG("Using so(3) parameterization\n");

	char Fname[512];
	VideoData VideoInfoI;
	if (ReadVideoDataI(Path, VideoInfoI, selectedCams, startF, stopF) == 1)
		return 1;

	Point2d uv; Point3d P3d;  double scale;
	vector<int>P3dID;
	vector<double>P3D;
	vector< vector<int> >frameIDPer3D;
	vector< vector<double> >scalePer3D;
	vector<vector<Point2d> >P2dPer3D;

	bool ReadCalibInputData = false, SaveCalibInputData = true;
	sprintf(Fname, "%s/VideoPose_Optim_Input.dat", Path);
	if (IsFileExist(Fname) == 1)
		ReadCalibInputData = true;
	if (ReadCalibInputData)
	{
		ifstream fin; fin.open(Fname, ios::binary);
		if (!fin.is_open())
		{
			cout << "Cannot open: " << Fname << endl;
			return false;
		}

		int npts;  fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
		P3D.reserve(npts * 3);

		vector<int> FrameIDs;
		vector<double>Scales;
		vector<Point2d>P2ds;
		for (int ii = 0; ii < npts; ii++)
		{
			int nvisibles; fin.read(reinterpret_cast<char *>(&nvisibles), sizeof(int));
			float x;  fin.read(reinterpret_cast<char *>(&x), sizeof(float)); P3D.push_back(x);
			float y;  fin.read(reinterpret_cast<char *>(&y), sizeof(float)); P3D.push_back(y);
			float z;  fin.read(reinterpret_cast<char *>(&z), sizeof(float)); P3D.push_back(z);

			frameIDPer3D.push_back(FrameIDs); frameIDPer3D[ii].reserve(nvisibles);
			P2dPer3D.push_back(P2ds), P2dPer3D[ii].reserve(nvisibles);
			scalePer3D.push_back(Scales); scalePer3D[ii].reserve(nvisibles);
			for (int jj = 0; jj < nvisibles; jj++)
			{
				int fid; fin.read(reinterpret_cast<char *>(&fid), sizeof(int));
				float u, v; fin.read(reinterpret_cast<char *>(&u), sizeof(float)); fin.read(reinterpret_cast<char *>(&v), sizeof(float));
				float s; fin.read(reinterpret_cast<char *>(&s), sizeof(float));

				frameIDPer3D[ii].push_back(fid);
				P2dPer3D[ii].push_back(Point2d(u, v));
				scalePer3D[ii].push_back(s);
			}
		}
		fin.close();
	}
	else
	{
		int pid, ReservedSpace = 20000;
		vector<int> FrameIDs;
		vector<Point2d>P2ds;
		vector<double>Scales;
		P3dID.reserve(ReservedSpace);
		P3D.reserve(ReservedSpace * 3);
		for (int ii = 0; ii < ReservedSpace; ii++)
		{
			frameIDPer3D.push_back(FrameIDs), frameIDPer3D[ii].reserve(stopF - startF + 1);
			P2dPer3D.push_back(P2ds), P2dPer3D[ii].reserve(stopF - startF + 1);
			scalePer3D.push_back(Scales); scalePer3D[ii].reserve(stopF - startF + 1);
		}

		for (int frameID = startF; frameID <= stopF; frameID++)
		{
			sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, selectedCams, frameID);	FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("Cannot load %s\n", Fname);
				continue;
			}
			while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &pid, &P3d.x, &P3d.y, &P3d.z, &uv.x, &uv.y, &scale) != EOF)
			{
				int foundLoc = -1, maxLoc = (int)P3dID.size();
				for (foundLoc = 0; foundLoc < maxLoc; foundLoc++)
				{
					if (pid == P3dID[foundLoc])
						break;
				}

				if (foundLoc == maxLoc)
				{
					if (ReservedSpace == maxLoc) //need to add more space
					{
						for (int ii = 0; ii < 5000; ii++)
						{
							frameIDPer3D.push_back(FrameIDs), frameIDPer3D[ii].reserve(stopF - startF + 1);
							P2dPer3D.push_back(P2ds), P2dPer3D[ii].reserve(stopF - startF + 1);
							scalePer3D.push_back(Scales); scalePer3D[ii].reserve(stopF - startF + 1);
						}
						ReservedSpace += 5000;
					}


					frameIDPer3D[maxLoc].push_back(frameID);
					P2dPer3D[maxLoc].push_back(uv);
					scalePer3D[foundLoc].push_back(scale);
					P3D.push_back(P3d.x), P3D.push_back(P3d.y); P3D.push_back(P3d.z);
					P3dID.push_back(pid);
				}
				else
				{
					frameIDPer3D[foundLoc].push_back(frameID);
					P2dPer3D[foundLoc].push_back(uv);
					scalePer3D[foundLoc].push_back(scale);
				}
			}
			fclose(fp);
		}

		//Find 3d points with less than nvisible views
		const int nvisibles = 5;
		vector<int> NotOftenVisible;
		for (int ii = 0; ii < (int)P3dID.size(); ii++)
			if (frameIDPer3D[ii].size() < nvisibles)
				NotOftenVisible.push_back(ii);
		printLOG("(%d/%d) points not visible by at least %d frames\n", NotOftenVisible.size(), P3dID.size(), nvisibles);

		//Clean from bottom to top
		for (int ii = (int)NotOftenVisible.size() - 1; ii >= 0; ii--)
		{
			P3dID.erase(P3dID.begin() + NotOftenVisible[ii]);
			P3D.erase(P3D.begin() + 3 * NotOftenVisible[ii], P3D.begin() + 3 * NotOftenVisible[ii] + 3);
			frameIDPer3D.erase(frameIDPer3D.begin() + NotOftenVisible[ii]);
			P2dPer3D.erase(P2dPer3D.begin() + NotOftenVisible[ii]);
			scalePer3D.erase(scalePer3D.begin() + NotOftenVisible[ii]);
		}

		//Save the Data
		if (SaveCalibInputData)
		{
			sprintf(Fname, "%s/VideoPose_Optim_Input.dat", Path);
			ofstream fout; fout.open(Fname, ios::binary);

			int npts = (int)P3D.size() / 3;
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (int ii = 0; ii < npts; ii++)
			{
				int nvisibles = (int)frameIDPer3D[ii].size();
				float X = (float)P3D[3 * ii], Y = (float)P3D[3 * ii + 1], Z = (float)P3D[3 * ii + 2];

				fout.write(reinterpret_cast<char *>(&nvisibles), sizeof(int));
				fout.write(reinterpret_cast<char *>(&X), sizeof(float));
				fout.write(reinterpret_cast<char *>(&Y), sizeof(float));
				fout.write(reinterpret_cast<char *>(&Z), sizeof(float));
				for (int jj = 0; jj < nvisibles; jj++)
				{
					float u = (float)P2dPer3D[ii][jj].x, v = (float)P2dPer3D[ii][jj].y, s = (float)scalePer3D[ii][jj];
					fout.write(reinterpret_cast<char *>(&frameIDPer3D[ii][jj]), sizeof(int));
					fout.write(reinterpret_cast<char *>(&u), sizeof(float));
					fout.write(reinterpret_cast<char *>(&v), sizeof(float));
					fout.write(reinterpret_cast<char *>(&s), sizeof(float));
				}
			}
			fout.close();
		}
	}

	//Set up Bspline: Control points are placed every controlStep frame
	int nCtrls = (stopF - startF) / controlStep + 1, nbreaks = nCtrls - SplineOrder + 2, extraNControls = 2;
	int ActingID[6];
	double breakStep = 1.0*(stopF - startF) / (nbreaks - 1);

	//Figure out which frame is not available and take value from its neighbor
	int *ControlLoc = new int[nCtrls];
	for (int ii = 0; ii < nCtrls; ii++)
	{
		int fid = controlStep * ii + startF;
		if (!VideoInfoI.VideoInfo[fid].valid)
		{
			int searchRange = 1;
			while (true)
			{
				for (int jj = -searchRange; jj <= searchRange; jj++)
				{
					if (jj == 0 || abs(jj) != searchRange)
						continue;
					if (fid + searchRange<startF || fid + searchRange>stopF || !VideoInfoI.VideoInfo[fid + searchRange].valid)
						continue;
					ControlLoc[ii] = fid + searchRange;
				}
			}
		}
		else
			ControlLoc[ii] = fid;
	}

	//Set open-uniform break points
	double *BreakLoc = new double[nbreaks];
	for (int ii = 0; ii < nbreaks; ii++)
		BreakLoc[ii] = breakStep * ii + startF;

	double *KnotLoc = new double[nCtrls + SplineOrder];
	BSplineGetKnots(KnotLoc, BreakLoc, nbreaks, nCtrls, SplineOrder);

	//Init control pose in se3
	double twist[6], tr[6];
	double *ControlPose = new double[6 * nCtrls];//stack of groups of 6 numbers 
	for (int ii = 0; ii < nCtrls; ii++)
	{
		if (se3 == 1)
		{
			getTwistFromRT(VideoInfoI.VideoInfo[ControlLoc[ii]].R, VideoInfoI.VideoInfo[ControlLoc[ii]].T, twist);
			for (int jj = 0; jj < 6; jj++)
				ControlPose[jj + 6 * ii] = twist[jj];
		}
		else
		{
			getrFromR(VideoInfoI.VideoInfo[ControlLoc[ii]].R, tr + 3);
			for (int jj = 0; jj < 3; jj++)
				tr[jj] = VideoInfoI.VideoInfo[ControlLoc[ii]].T[jj];
			for (int jj = 0; jj < 6; jj++)
				ControlPose[jj + 6 * ii] = tr[jj];
		}
	}

	//Start solver
	bool setReferenceflag = false;
	int frameID, RefFrameID, nBadCounts = 0, validPtsCount = 0;
	vector<bool>Good; Good.reserve((stopF - startF + 1) * 5000);
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve((stopF - startF + 1) * 5000);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve((stopF - startF + 1) * 5000);
	double maxOutlierX = 0.0, maxOutlierY = 0.0, pointErrX = 0.0, pointErrY = 0.0, residuals[2];

	ceres::Problem problem;
	//ceres::LossFunction* loss_function = new HuberLoss(3.0*threshold);

	FILE *fp = 0;
	if (debug)
		sprintf(Fname, "%s/reprojectionB.txt", Path), fp = fopen(Fname, "w+");
	for (int pid = 0; pid < (int)P3D.size() / 3; pid++)
	{
		for (int fid = 0; fid < (int)frameIDPer3D[pid].size(); fid++)
		{
			frameID = frameIDPer3D[pid][fid];
			uv = P2dPer3D[pid][fid];
			scale = scalePer3D[pid][fid];
			P3d.x = P3D[3 * pid], P3d.y = P3D[3 * pid + 1], P3d.z = P3D[3 * pid + 2];

			//Determine its acting controlPts
			double subframeLoc = 0.5 + frameID;
			if (subframeLoc < KnotLoc[SplineOrder - 1])
				subframeLoc = KnotLoc[SplineOrder - 1] + 0.5;
			else if (subframeLoc > KnotLoc[nCtrls])
				subframeLoc = KnotLoc[nCtrls] - 0.5;
			BSplineFindActiveCtrl(ActingID, subframeLoc, KnotLoc, nbreaks, nCtrls, SplineOrder, extraNControls);

			if (!setReferenceflag)
			{
				if (distortionCorrected)
					RollingShutterSplineReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, ActingID, ControlPose + ActingID[0] * (SplineOrder + extraNControls), KnotLoc, nbreaks, nCtrls, SplineOrder, se3,
						uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
				else
					RollingShutterDistortionSplineReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, VideoInfoI.VideoInfo[frameID].distortion, ActingID, ControlPose + ActingID[0] * (SplineOrder + extraNControls), KnotLoc, nbreaks, nCtrls, SplineOrder,
						uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
			}
			else
				if (distortionCorrected)
					RollingShutterSplineReprojectionDebug(VideoInfoI.VideoInfo[RefFrameID].intrinsic, ActingID, ControlPose + ActingID[0] * (SplineOrder + extraNControls), KnotLoc, nbreaks, nCtrls, SplineOrder, se3,
						uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
				else
					RollingShutterDistortionSplineReprojectionDebug(VideoInfoI.VideoInfo[RefFrameID].intrinsic, VideoInfoI.VideoInfo[RefFrameID].distortion, ActingID, ControlPose + ActingID[0] * (SplineOrder + extraNControls), KnotLoc, nbreaks, nCtrls, SplineOrder,
						uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);

			if (abs(residuals[0]) > 10.0*threshold || abs(residuals[1]) > 10.0*threshold)
			{
				Good.push_back(false);
				if (abs(residuals[0]) > maxOutlierX)
					maxOutlierX = residuals[0];
				if (abs(residuals[1]) > maxOutlierY)
					maxOutlierY = residuals[1];
				nBadCounts++;
				continue;
			}
			else
			{
				Good.push_back(true);
				if (!setReferenceflag)
					RefFrameID = frameID, setReferenceflag = true;

				if (distortionCorrected == 1)
				{
					ceres::CostFunction* cost_function = RollingShutterSplineReprojectionError::CreateNumerDiff(VideoInfoI.VideoInfo[RefFrameID].intrinsic, KnotLoc, nbreaks, nCtrls, SplineOrder, se3,
						uv, ActingID, scale, pid, frameID, VideoInfoI.VideoInfo[RefFrameID].width, VideoInfoI.VideoInfo[RefFrameID].height);
					problem.AddResidualBlock(cost_function, NULL, &ControlPose[6 * ActingID[0]], &ControlPose[6 * ActingID[1]], &ControlPose[6 * ActingID[2]],
						&ControlPose[6 * ActingID[3]], &ControlPose[6 * ActingID[4]], &ControlPose[6 * ActingID[5]], &P3D[3 * pid]);
				}
				else
				{
					ceres::CostFunction* cost_function = RollingShutterDistortionSplineReprojectionError::CreateNumerDiff(KnotLoc, nbreaks, nCtrls, SplineOrder, uv, ActingID, scale, frameID, VideoInfoI.VideoInfo[RefFrameID].width, VideoInfoI.VideoInfo[RefFrameID].height);
					problem.AddResidualBlock(cost_function, NULL, VideoInfoI.VideoInfo[RefFrameID].intrinsic, VideoInfoI.VideoInfo[RefFrameID].distortion,
						&ControlPose[6 * ActingID[0]], &ControlPose[6 * ActingID[1]], &ControlPose[6 * ActingID[2]], &ControlPose[6 * ActingID[3]], &ControlPose[6 * ActingID[4]], &ControlPose[6 * ActingID[5]], &P3D[3 * pid]);

					if (fixIntrinsic)
						problem.SetParameterBlockConstant(VideoInfoI.VideoInfo[RefFrameID].intrinsic);
					if (fixDistortion)
						problem.SetParameterBlockConstant(VideoInfoI.VideoInfo[RefFrameID].distortion);
				}

				validPtsCount++;
				ReProjectionErrorX.push_back(residuals[0]);
				ReProjectionErrorY.push_back(residuals[1]);

				if (debug)
					fprintf(fp, "%d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", pid, frameID, P3D[3 * pid], P3D[3 * pid + 1], P3D[3 * pid + 2], uv.x, uv.y, residuals[0], residuals[1]);
			}
		}
	}
	if (debug)
		fclose(fp);

	double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double avgX = MeanArray(ReProjectionErrorX);
	double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double avgY = MeanArray(ReProjectionErrorY);
	double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

#pragma omp critical
	{
		printLOG("\n %d bad points (%d good points) detected with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, validPtsCount, maxOutlierX, maxOutlierY);
		printLOG("Reprojection error before BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	//printLOG("...run \n");
	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 30;

	options.linear_solver_type = ceres::CGNR;
	options.preconditioner_type = ceres::JACOBI;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	//Store refined parameters
	printLOG("Reference cam: %d\n", RefFrameID);
	for (int frameID = startF; frameID <= stopF; frameID++)
	{
		CopyCamereInfo(VideoInfoI.VideoInfo[RefFrameID], VideoInfoI.VideoInfo[frameID], false);
		GetKFromIntrinsic(VideoInfoI.VideoInfo[frameID]);
	}

	int count = -1;
	ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
	pointErrX = 0.0, pointErrY = 0.0, validPtsCount = 0;

	if (debug)
		sprintf(Fname, "%s/reprojectionA.txt", Path), fp = fopen(Fname, "w+");
	for (int pid = 0; pid < (int)P3D.size() / 3; pid++)
	{
		for (int fid = 0; fid < (int)frameIDPer3D[pid].size(); fid++)
		{
			frameID = frameIDPer3D[pid][fid];
			uv = P2dPer3D[pid][fid];
			scale = scalePer3D[pid][fid];
			P3d.x = P3D[3 * pid], P3d.y = P3D[3 * pid + 1], P3d.z = P3D[3 * pid + 2];

			count++;
			if (!Good[count])
				continue;

			//Determine its acting controlPts
			double subframeLoc = 0.5 + frameID;
			if (subframeLoc < KnotLoc[SplineOrder - 1])
				subframeLoc = KnotLoc[SplineOrder - 1] + 0.5;
			else if (subframeLoc > KnotLoc[nCtrls])
				subframeLoc = KnotLoc[nCtrls] - 0.5;
			BSplineFindActiveCtrl(ActingID, subframeLoc, KnotLoc, nbreaks, nCtrls, SplineOrder, extraNControls);

			if (distortionCorrected)
				RollingShutterSplineReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, ActingID, ControlPose + ActingID[0] * (SplineOrder + extraNControls), KnotLoc, nbreaks, nCtrls, SplineOrder, se3,
					uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
			else
				RollingShutterDistortionSplineReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, VideoInfoI.VideoInfo[frameID].distortion, ActingID, ControlPose + ActingID[0] * (SplineOrder + extraNControls), KnotLoc, nbreaks, nCtrls, SplineOrder,
					uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);

			validPtsCount++;
			ReProjectionErrorX.push_back(residuals[0]);
			ReProjectionErrorY.push_back(residuals[1]);

			if (debug)
				fprintf(fp, "%d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", pid, frameID, P3D[3 * pid], P3D[3 * pid + 1], P3D[3 * pid + 2], uv.x, uv.y, residuals[0], residuals[1]);
		}
	}
	if (debug)
		fclose(fp);

	miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	avgX = MeanArray(ReProjectionErrorX);
	stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	avgY = MeanArray(ReProjectionErrorY);
	stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

#pragma omp critical
	printLOG("Reprojection error after BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	//Write the data
	sprintf(Fname, "%s/IntrinsicS_%.4d.txt", Path, selectedCams); fp = fopen(Fname, "w+");
	for (int frameID = startF; frameID <= stopF; frameID++)
	{
		fprintf(fp, "%d %d %d %d ", frameID, VideoInfoI.VideoInfo[frameID].LensModel, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%.4f ", VideoInfoI.VideoInfo[frameID].intrinsic[ii]);
		if (VideoInfoI.VideoInfo[frameID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%.4f ", VideoInfoI.VideoInfo[frameID].distortion[ii]);
		else
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%.4f ", VideoInfoI.VideoInfo[frameID].distortion[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (se3 == 1)
		sprintf(Fname, "%s/CamPoseS_se3_%.4d.txt", Path, selectedCams);
	else
		sprintf(Fname, "%s/CamPoseS_so3_%.4d.txt", Path, selectedCams);
	fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %d %d\n", nCtrls, nbreaks, SplineOrder, se3 == 1 ? 1 : 0);
	for (int ii = 0; ii < nCtrls; ii++)
	{
		fprintf(fp, "%d ", ControlLoc[ii]);
		for (int jj = 0; jj < 6; jj++)
			fprintf(fp, "%.8e ", ControlPose[ii * 6 + jj]);
		fprintf(fp, "\n");
	}
	for (int ii = 0; ii < nbreaks; ii++)
		fprintf(fp, "%.16e\n", BreakLoc[ii]);
	fclose(fp);

	delete[]ControlLoc, delete[]BreakLoc, delete[]KnotLoc, delete[]ControlPose;

	return 0;
}
int VideoDCTRSBA(char *Path, int startF, int stopF, int selectedCams, int distortionCorrected, int fixIntrinsic, int fixDistortion, double threshold, int sampleStep, double lamda, bool debug)
{
	FILE *fp = 0;
	char Fname[512];
	VideoData VideoInfoI;
	if (ReadVideoDataI(Path, VideoInfoI, selectedCams, startF, stopF) == 1)
		return 1;

	Point2d uv; Point3d P3d;  double scale;
	vector<int>P3dID;
	vector<double>P3D;
	vector< vector<int> >frameIDPer3D;
	vector< vector<double> >scalePer3D;
	vector<vector<Point2d> >P2dPer3D;

	bool ReadCalibInputData = true, SaveCalibInputData = false;
	if (ReadCalibInputData)
	{
		sprintf(Fname, "%s/VideoPose_Optim_Input.dat", Path);
		ifstream fin; fin.open(Fname, ios::binary);
		if (!fin.is_open())
		{
			cout << "Cannot open: " << Fname << endl;
			return false;
		}

		int npts;  fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
		P3D.reserve(npts * 3);

		vector<int> FrameIDs;
		vector<double>Scales;
		vector<Point2d>P2ds;
		for (int ii = 0; ii < npts; ii++)
		{
			int nvisibles; fin.read(reinterpret_cast<char *>(&nvisibles), sizeof(int));
			float x;  fin.read(reinterpret_cast<char *>(&x), sizeof(float)); P3D.push_back(x);
			float y;  fin.read(reinterpret_cast<char *>(&y), sizeof(float)); P3D.push_back(y);
			float z;  fin.read(reinterpret_cast<char *>(&z), sizeof(float)); P3D.push_back(z);

			frameIDPer3D.push_back(FrameIDs); frameIDPer3D[ii].reserve(nvisibles);
			P2dPer3D.push_back(P2ds), P2dPer3D[ii].reserve(nvisibles);
			scalePer3D.push_back(Scales); scalePer3D[ii].reserve(nvisibles);
			for (int jj = 0; jj < nvisibles; jj++)
			{
				int fid; fin.read(reinterpret_cast<char *>(&fid), sizeof(int));
				float u, v; fin.read(reinterpret_cast<char *>(&u), sizeof(float)); fin.read(reinterpret_cast<char *>(&v), sizeof(float));
				float s; fin.read(reinterpret_cast<char *>(&s), sizeof(float));

				frameIDPer3D[ii].push_back(fid);
				P2dPer3D[ii].push_back(Point2d(u, v));
				scalePer3D[ii].push_back(s);
			}
		}
		fin.close();
	}
	else
	{
		int pid, ReservedSpace = 20000;
		vector<int> FrameIDs;
		vector<Point2d>P2ds;
		vector<double>Scales;
		P3dID.reserve(ReservedSpace);
		P3D.reserve(ReservedSpace * 3);
		for (int ii = 0; ii < ReservedSpace; ii++)
		{
			frameIDPer3D.push_back(FrameIDs), frameIDPer3D[ii].reserve(stopF - startF + 1);
			P2dPer3D.push_back(P2ds), P2dPer3D[ii].reserve(stopF - startF + 1);
			scalePer3D.push_back(Scales); scalePer3D[ii].reserve(stopF - startF + 1);
		}

		for (int frameID = startF; frameID <= stopF; frameID++)
		{
			sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, selectedCams, frameID);	fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("Cannot load %s\n", Fname);
				continue;
			}
			while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &pid, &P3d.x, &P3d.y, &P3d.z, &uv.x, &uv.y, &scale) != EOF)
			{
				int foundLoc = -1, maxLoc = (int)P3dID.size();
				for (foundLoc = 0; foundLoc < maxLoc; foundLoc++)
				{
					if (pid == P3dID[foundLoc])
						break;
				}

				if (foundLoc == maxLoc)
				{
					if (ReservedSpace == maxLoc) //need to add more space
					{
						for (int ii = 0; ii < 1000; ii++)
						{
							frameIDPer3D.push_back(FrameIDs), frameIDPer3D[ii].reserve(stopF - startF + 1);
							P2dPer3D.push_back(P2ds), P2dPer3D[ii].reserve(stopF - startF + 1);
							scalePer3D.push_back(Scales); scalePer3D[ii].reserve(stopF - startF + 1);
						}
						ReservedSpace += 1000;
					}


					frameIDPer3D[maxLoc].push_back(frameID);
					P2dPer3D[maxLoc].push_back(uv);
					scalePer3D[foundLoc].push_back(scale);
					P3D.push_back(P3d.x), P3D.push_back(P3d.y); P3D.push_back(P3d.z);
					P3dID.push_back(pid);
				}
				else
				{
					frameIDPer3D[foundLoc].push_back(frameID);
					P2dPer3D[foundLoc].push_back(uv);
					scalePer3D[foundLoc].push_back(scale);
				}
			}
			fclose(fp);
		}

		//Find 3d points with less than nvisible views
		const int nvisibles = 5;
		vector<int> NotOftenVisible;
		for (int ii = 0; ii < (int)P3dID.size(); ii++)
			if (frameIDPer3D[ii].size() < nvisibles)
				NotOftenVisible.push_back(ii);
		printLOG("(%d/%d) points not visible by at least %d frames\n", NotOftenVisible.size(), P3dID.size(), nvisibles);

		//Clean from bottom to top
		for (int ii = (int)NotOftenVisible.size() - 1; ii >= 0; ii--)
		{
			P3dID.erase(P3dID.begin() + NotOftenVisible[ii]);
			P3D.erase(P3D.begin() + 3 * NotOftenVisible[ii], P3D.begin() + 3 * NotOftenVisible[ii] + 3);
			frameIDPer3D.erase(frameIDPer3D.begin() + NotOftenVisible[ii]);
			P2dPer3D.erase(P2dPer3D.begin() + NotOftenVisible[ii]);
			scalePer3D.erase(scalePer3D.begin() + NotOftenVisible[ii]);
		}

		//Save the Data
		if (SaveCalibInputData)
		{
			sprintf(Fname, "%s/VideoPose_Optim_Input.dat", Path);
			ofstream fout; fout.open(Fname, ios::binary);

			int npts = (int)P3D.size() / 3;
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (int ii = 0; ii < npts; ii++)
			{
				int nvisibles = (int)frameIDPer3D[ii].size();
				float X = (float)P3D[3 * ii], Y = (float)P3D[3 * ii + 1], Z = (float)P3D[3 * ii + 2];

				fout.write(reinterpret_cast<char *>(&nvisibles), sizeof(int));
				fout.write(reinterpret_cast<char *>(&X), sizeof(float));
				fout.write(reinterpret_cast<char *>(&Y), sizeof(float));
				fout.write(reinterpret_cast<char *>(&Z), sizeof(float));
				for (int jj = 0; jj < nvisibles; jj++)
				{
					float u = (float)P2dPer3D[ii][jj].x, v = (float)P2dPer3D[ii][jj].y, s = (float)scalePer3D[ii][jj];
					fout.write(reinterpret_cast<char *>(&frameIDPer3D[ii][jj]), sizeof(int));
					fout.write(reinterpret_cast<char *>(&u), sizeof(float));
					fout.write(reinterpret_cast<char *>(&v), sizeof(float));
					fout.write(reinterpret_cast<char *>(&s), sizeof(float));
				}
			}
			fout.close();
		}
	}


	//Set up DCT sampled evenly every sampleStep
	int nCoeffs = (int)((stopF - startF) / sampleStep + 1);

	double *sqrtWeight = new double[nCoeffs];
	GenerateDCTBasis(nCoeffs, NULL, sqrtWeight);
	for (int ii = 0; ii < nCoeffs; ii++)
		sqrtWeight[ii] = sqrt(-sqrtWeight[ii]); //(1) using precomputed sqrt is better for ceres' squaring residual square nature; (2) weigths are negative, but that does not matter for ctwc optim.

												//Initialize basis coefficients
	int count = 0, nframes = 0;
	for (int ii = startF; ii <= stopF; ii++)
		if (VideoInfoI.VideoInfo[ii].valid)
			nframes++;

	double twist[6];
	double *FrameTime = new double[nframes], *FramePose = new double[6 * (stopF - startF + 1)];
	for (int ii = startF; ii <= stopF; ii++)
	{
		if (VideoInfoI.VideoInfo[ii].valid)
		{
			getTwistFromRT(VideoInfoI.VideoInfo[ii].R, VideoInfoI.VideoInfo[ii].T, twist);
			for (int jj = 0; jj < 6; jj++)
				FramePose[ii + jj * nframes] = twist[jj];

			FrameTime[count - startF] = 1.0*(ii - startF) / (stopF - startF)*(nCoeffs - 1);//Normalize to [0, n-1] range
			count++;
		}
	}

	double *iBi = new double[nCoeffs], *iBAll = new double[nframes*nCoeffs];
	for (int ii = 0; ii < nframes; ii++)
		GenerateiDCTBasis(iBAll + ii * nCoeffs, nCoeffs, FrameTime[ii]);


	//Trucated basis solver: iPd(:, 1:activeBasis)*C =  X_d
	double err;
	const int nactiveBasis = 20;
	double *C = new double[6 * nCoeffs];
	Map < Matrix < double, Dynamic, Dynamic, RowMajor > > eiBAll(iBAll, nframes, nCoeffs);
	MatrixXd etiBAll = eiBAll.block(0, 0, nframes, nactiveBasis);
	JacobiSVD<MatrixXd> etiP_svd(etiBAll, ComputeThinU | ComputeThinV);
	for (int ii = 0; ii < 6; ii++)
	{
		Map<VectorXd> eX(FramePose + nframes * ii, nframes);
		Map<VectorXd> eC(C + nCoeffs * ii, nactiveBasis);

		if (eX.norm() < 0.1)// happens for rotation sometimes
		{
			for (int jj = 0; jj < nCoeffs; jj++)
				C[jj + nCoeffs * ii] = 0.0;
			err = (etiBAll*eC - eX).norm();
		}
		else
		{
			eC = etiP_svd.solve(eX);

			for (int jj = nactiveBasis; jj < nCoeffs; jj++)
				C[jj + nCoeffs * ii] = 0.0; //set coeffs outside active basis to 0
			err = (etiBAll*eC - eX).norm() / eX.norm();
		}
	}

	//Start solver
	bool setReferenceflag = false;
	int frameID, RefFrameID, nBadCounts = 0, validPtsCount = 0;
	vector<bool>Good; Good.reserve((stopF - startF + 1) * 5000);
	vector<double> ReProjectionErrorX; ReProjectionErrorX.reserve((stopF - startF + 1) * 5000);
	vector<double> ReProjectionErrorY; ReProjectionErrorY.reserve((stopF - startF + 1) * 5000);
	double maxOutlierX = 0.0, maxOutlierY = 0.0, pointErrX = 0.0, pointErrY = 0.0, residuals[2];

	ceres::Problem problem;
	//ceres::LossFunction* loss_function = new HuberLoss(3.0*threshold);

	//Image projection cost
	if (debug)
		sprintf(Fname, "%s/reprojectionB.txt", Path), fp = fopen(Fname, "w+");
	for (int pid = 0; pid < (int)P3D.size() / 3; pid++)
	{
		for (int fid = 0; fid < (int)frameIDPer3D[pid].size(); fid++)
		{
			frameID = frameIDPer3D[pid][fid];
			uv = P2dPer3D[pid][fid];
			scale = scalePer3D[pid][fid];
			P3d.x = P3D[3 * pid], P3d.y = P3D[3 * pid + 1], P3d.z = P3D[3 * pid + 2];

			if (!setReferenceflag)
			{
				if (distortionCorrected)
					RollingShutterDCTReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, C, C + nCoeffs, C + 2 * nCoeffs, C + 3 * nCoeffs, C + 4 * nCoeffs, C + 5 * nCoeffs, nCoeffs,
						uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
				else
					;// RollingShutterDistortionSplineReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, VideoInfoI.VideoInfo[frameID].distortion, ActingID, FramePose + ActingID[0] * 6, KnotLoc, nbreaks, nCtrls, SplineOrder,
					 //uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
			}
			else
				if (distortionCorrected)
					RollingShutterDCTReprojectionDebug(VideoInfoI.VideoInfo[RefFrameID].intrinsic, C, C + nCoeffs, C + 2 * nCoeffs, C + 3 * nCoeffs, C + 4 * nCoeffs, C + 5 * nCoeffs, nCoeffs,
						uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
				else
					;// RollingShutterDistortionSplineReprojectionDebug(VideoInfoI.VideoInfo[RefFrameID].intrinsic, VideoInfoI.VideoInfo[RefFrameID].distortion, ActingID, FramePose + ActingID[0] * 6, KnotLoc, nbreaks, nCtrls, SplineOrder,
					 //uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);

			if (abs(residuals[0]) > 10.0*threshold || abs(residuals[1]) > 10.0*threshold)
			{
				Good.push_back(false);
				if (abs(residuals[0]) > maxOutlierX)
					maxOutlierX = residuals[0];
				if (abs(residuals[1]) > maxOutlierY)
					maxOutlierY = residuals[1];
				nBadCounts++;
				continue;
			}
			else
			{
				Good.push_back(true);
				if (!setReferenceflag)
					RefFrameID = frameID, setReferenceflag = true;

				if (distortionCorrected)
				{
					//ceres::DynamicAutoDiffCostFunction<RollingShutterDCTReprojectionError, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction < RollingShutterDCTReprojectionError, 4 >
					//	(new RollingShutterDCTReprojectionError(VideoInfoI.VideoInfo[RefFrameID].intrinsic, sqrtWeight, nCoeffs, uv, scale, pid, frameID, VideoInfoI.VideoInfo[RefFrameID].width, VideoInfoI.VideoInfo[RefFrameID].height));

					ceres::DynamicNumericDiffCostFunction<RollingShutterDCTReprojectionError, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<RollingShutterDCTReprojectionError, ceres::CENTRAL>
						(new RollingShutterDCTReprojectionError(VideoInfoI.VideoInfo[RefFrameID].intrinsic, sqrtWeight, nCoeffs, uv, scale, pid, frameID, VideoInfoI.VideoInfo[RefFrameID].width, VideoInfoI.VideoInfo[RefFrameID].height));

					vector<double*> parameter_blocks;
					parameter_blocks.push_back(&P3D[3 * pid]);
					cost_function->AddParameterBlock(3);
					for (int ii = 0; ii < 6; ii++)
					{
						parameter_blocks.push_back(C + ii * nCoeffs);
						cost_function->AddParameterBlock(nCoeffs);
					}
					cost_function->SetNumResiduals(2);

					problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

					//problem.SetParameterBlockConstant(&parameter_blocks[npts][0]);
				}
				else
				{
					//ceres::CostFunction* cost_function = RollingShutterDistortionSplineReprojectionError::CreateNumerDiff(KnotLoc, nbreaks, nCtrls, SplineOrder, uv, ActingID, scale, frameID, VideoInfoI.VideoInfo[RefFrameID].width, VideoInfoI.VideoInfo[RefFrameID].height);
					//problem.AddResidualBlock(cost_function, NULL, VideoInfoI.VideoInfo[RefFrameID].intrinsic, VideoInfoI.VideoInfo[RefFrameID].distortion,
					//	&FramePose[6 * ActingID[0]], &FramePose[6 * ActingID[1]], &FramePose[6 * ActingID[2]], &FramePose[6 * ActingID[3]], &FramePose[6 * ActingID[4]], &FramePose[6 * ActingID[5]], &P3D[3 * pid]);

					if (fixIntrinsic)
						problem.SetParameterBlockConstant(VideoInfoI.VideoInfo[RefFrameID].intrinsic);
					if (fixDistortion)
						problem.SetParameterBlockConstant(VideoInfoI.VideoInfo[RefFrameID].distortion);
				}

				validPtsCount++;
				ReProjectionErrorX.push_back(residuals[0]);
				ReProjectionErrorY.push_back(residuals[1]);

				if (debug)
					fprintf(fp, "%d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", pid, frameID, P3D[3 * pid], P3D[3 * pid + 1], P3D[3 * pid + 2], uv.x, uv.y, residuals[0], residuals[1]);
			}
		}
	}
	if (debug)
		fclose(fp);

	//Regularization cost
	ceres::DynamicAutoDiffCostFunction<RollingShutterDCTRegularizationError, 4> *cost_function =
		new ceres::DynamicAutoDiffCostFunction < RollingShutterDCTRegularizationError, 4 >(new RollingShutterDCTRegularizationError(sqrtWeight, nCoeffs, sqrt(lamda)));

	vector<double*> parameter_blocks;
	for (int ii = 0; ii < 6; ii++)
	{
		parameter_blocks.push_back(C + ii * nCoeffs);
		cost_function->AddParameterBlock(nCoeffs);
	}
	cost_function->SetNumResiduals(6 * nCoeffs);
	problem.AddResidualBlock(cost_function, NULL, parameter_blocks);


	double miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	double avgX = MeanArray(ReProjectionErrorX);
	double stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	double miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	double avgY = MeanArray(ReProjectionErrorY);
	double stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

#pragma omp critical
	{
		printLOG("\n %d bad points (%d good points) detected with maximum reprojection error of (%.2f %.2f) \n", nBadCounts, validPtsCount, maxOutlierX, maxOutlierY);
		printLOG("Reprojection error before BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
	}

	//printLOG("...run \n");
	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 30;
	options.linear_solver_type = ceres::SPARSE_SCHUR;

	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	//Store refined parameters
	printLOG("Reference cam: %d\n", RefFrameID);
	for (int frameID = startF; frameID <= stopF; frameID++)
	{
		CopyCamereInfo(VideoInfoI.VideoInfo[RefFrameID], VideoInfoI.VideoInfo[frameID], false);
		GetKFromIntrinsic(VideoInfoI.VideoInfo[frameID]);
	}

	count = -1;
	ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
	pointErrX = 0.0, pointErrY = 0.0, validPtsCount = 0;

	if (debug)
		sprintf(Fname, "%s/reprojectionA.txt", Path), fp = fopen(Fname, "w+");
	for (int pid = 0; pid < (int)P3D.size() / 3; pid++)
	{
		for (int fid = 0; fid < (int)frameIDPer3D[pid].size(); fid++)
		{
			frameID = frameIDPer3D[pid][fid];
			uv = P2dPer3D[pid][fid];
			scale = scalePer3D[pid][fid];
			P3d.x = P3D[3 * pid], P3d.y = P3D[3 * pid + 1], P3d.z = P3D[3 * pid + 2];

			count++;
			if (!Good[count])
				continue;

			if (distortionCorrected)
				RollingShutterDCTReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, C, C + nCoeffs, C + 2 * nCoeffs, C + 3 * nCoeffs, C + 4 * nCoeffs, C + 5 * nCoeffs, nCoeffs,
					uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);
			else
				;// RollingShutterDistortionSplineReprojectionDebug(VideoInfoI.VideoInfo[frameID].intrinsic, VideoInfoI.VideoInfo[frameID].distortion, ActingID, FramePose + 6 * ActingID[0], KnotLoc, nbreaks, nCtrls, SplineOrder,
				 //uv, P3d, frameID, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height, residuals);

			validPtsCount++;
			ReProjectionErrorX.push_back(residuals[0]);
			ReProjectionErrorY.push_back(residuals[1]);

			if (debug)
				fprintf(fp, "%d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", pid, frameID, P3D[3 * pid], P3D[3 * pid + 1], P3D[3 * pid + 2], uv.x, uv.y, residuals[0], residuals[1]);
		}
	}
	if (debug)
		fclose(fp);

	miniX = *min_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	maxiX = *max_element(ReProjectionErrorX.begin(), ReProjectionErrorX.end());
	avgX = MeanArray(ReProjectionErrorX);
	stdX = sqrt(VarianceArray(ReProjectionErrorX, avgX));
	miniY = *min_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	maxiY = *max_element(ReProjectionErrorY.begin(), ReProjectionErrorY.end());
	avgY = MeanArray(ReProjectionErrorY);
	stdY = sqrt(VarianceArray(ReProjectionErrorY, avgY));

#pragma omp critical
	printLOG("Reprojection error after BA \n Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);

	//Write the data
	sprintf(Fname, "%s/IntrinsicDCT_%.4d.txt", Path, selectedCams); fp = fopen(Fname, "w+");
	for (int frameID = startF; frameID <= stopF; frameID++)
	{
		fprintf(fp, "%d %d %d %d ", frameID, VideoInfoI.VideoInfo[frameID].LensModel, VideoInfoI.VideoInfo[frameID].width, VideoInfoI.VideoInfo[frameID].height);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%.4f ", VideoInfoI.VideoInfo[frameID].intrinsic[ii]);
		if (VideoInfoI.VideoInfo[frameID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%.4f ", VideoInfoI.VideoInfo[frameID].distortion[ii]);
		else
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%.4f ", VideoInfoI.VideoInfo[frameID].distortion[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/CamPoseDCT_%.4d.txt", Path, selectedCams); fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %d\n", startF, nCoeffs, sampleStep);
	for (int jj = 0; jj < 6; jj++)
	{
		for (int ii = 0; ii < nCoeffs; ii++)
			fprintf(fp, "%.8e ", C[ii + jj * nCoeffs]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]FrameTime, delete[]FramePose;
	delete[]sqrtWeight, delete[]iBi, delete[]iBAll, delete[]C;

	return 0;
}

//Nonlinear Optimization for Temporal Alignement using BA geometric constraint
int PrepareTrajectoryInfo(char *Path, VideoData *VideoInfo, PerCamNonRigidTrajectory *CamTraj, double *OffsetInfo, int nCams, int npts, int startF, int stopF)
{
	char Fname[512];
	int id, nf, frameID;
	//CamCenter Ccenter;
	//RotMatrix Rmat;
	//Quaternion Qmat;
	//KMatrix Kmat;
	//Pmat P;

	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], camID, startF, stopF) == 1)
			return 1;

	for (int ii = 0; ii < nCams; ii++)
		CamTraj[ii].npts = npts,
		CamTraj[ii].Track2DInfo = new Track2D[npts],
		CamTraj[ii].Track3DInfo = new Track3D[npts];

	Point2d uv;
	vector<Point2d> uvAll;
	vector<int> frameIDAll;
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Track2D/C_%.4d.txt", Path, camID); FILE *fp = fopen(Fname, "r");
		for (int pid = 0; pid < npts; pid++)
		{
			fscanf(fp, "%d %d ", &id, &nf);
			if (id != pid)
				printLOG("Problem at Point %d of Cam %d", id, camID);

			uvAll.clear(), frameIDAll.clear();
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &uv.x, &uv.y);
				if (frameID < startF || frameID>stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (uv.x > 0 && uv.y > 0)
				{
					LensCorrectionPoint(&uv, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					uvAll.push_back(uv);
					frameIDAll.push_back(frameID);
				}
			}

			nf = (int)uvAll.size();
			CamTraj[camID].Track2DInfo[pid].nf = nf;
			CamTraj[camID].Track3DInfo[pid].nf = nf;
			CamTraj[camID].Track2DInfo[pid].uv = new Point2d[nf];
			CamTraj[camID].Track3DInfo[pid].xyz = new double[nf * 3];

			for (int kk = 0; kk < nf; kk++)
				CamTraj[camID].Track2DInfo[pid].uv[kk] = uvAll[kk];
		}
		fclose(fp);
	}

	//Triangulate 3D data
	return 0;
}
int TemporalOptimInterp()
{
	const int nCams = 3, npts = 4;
	PerCamNonRigidTrajectory CamTraj[nCams];

	//PrepareTrajectoryInfo(CamTraj, nCams, npts);

	//Interpolate the trajectory of 2d tracks
	int maxPts = 0;
	for (int ii = 0; ii < npts; ii++)
	{
		for (int jj = 0; jj < nCams; jj++)
		{
			int npts = 0;
			for (int kk = 0; kk < CamTraj[jj].Track3DInfo[ii].nf; kk++)
				npts++;
			if (npts > maxPts)
				maxPts = npts;
		}
	}

	int InterpAlgo = 1;
	double *x = new double[maxPts], *y = new double[maxPts];
	double* AllQuaternion = new double[4 * nCams*maxPts];
	double* AllRotationMat = new double[9 * nCams*maxPts];
	double* AllCamCenter = new double[3 * nCams*maxPts];
	double *AllKMatrix = new double[9 * nCams*maxPts];
	double *AllPMatrix = new double[12 * nCams*maxPts];

	double *z = new double[maxPts];
	double *ParaCamCenterX = new double[nCams*maxPts];
	double *ParaCamCenterY = new double[nCams*maxPts];
	double *ParaCamCenterZ = new double[nCams*maxPts];
	for (int jj = 0; jj < nCams; jj++)
	{
		for (int ii = 0; ii < CamTraj[jj].npts; ii++)
		{
			int nf = CamTraj[jj].Track3DInfo[ii].nf;
			CamTraj[jj].Track2DInfo[ii].ParaX = new double[nf];
			CamTraj[jj].Track2DInfo[ii].ParaY = new double[nf];

			for (int kk = 0; kk < nf; kk++)
				x[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].x, y[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].y;
			Generate_Para_Spline(x, CamTraj[jj].Track2DInfo[ii].ParaX, nf, 1, InterpAlgo);
			Generate_Para_Spline(y, CamTraj[jj].Track2DInfo[ii].ParaY, nf, 1, InterpAlgo);
		}
		for (int ii = 0; ii < maxPts; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			for (int kk = 0; kk < 9; kk++)
				AllKMatrix[9 * jj * maxPts + 9 * ii + kk] = CamTraj[jj].K[ii].K[kk];
			for (int kk = 0; kk < 3; kk++)
				AllCamCenter[3 * jj * maxPts + 3 * ii + kk] = CamTraj[jj].C[ii].C[kk];
			for (int kk = 0; kk < 9; kk++)
				AllRotationMat[9 * jj * maxPts + 9 * ii + kk] = CamTraj[jj].R[ii].R[kk];
			for (int kk = 0; kk < 4; kk++)
				AllQuaternion[4 * jj * maxPts + 4 * ii + kk] = CamTraj[jj].Q[ii].quad[kk];
			for (int kk = 0; kk < 12; kk++)
				AllPMatrix[12 * jj * maxPts + 12 * ii + kk] = CamTraj[jj].P[ii].P[kk];
		}

		for (int ii = 0; ii < maxPts; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			x[ii] = AllCamCenter[3 * jj * maxPts + 3 * ii];
			y[ii] = AllCamCenter[3 * jj * maxPts + 3 * ii + 1];
			z[ii] = AllCamCenter[3 * jj * maxPts + 3 * ii + 2];
		}
		Generate_Para_Spline(x, ParaCamCenterX + jj * maxPts, maxPts, 1, InterpAlgo);
		Generate_Para_Spline(y, ParaCamCenterY + jj * maxPts, maxPts, 1, InterpAlgo);
		Generate_Para_Spline(z, ParaCamCenterZ + jj * maxPts, maxPts, 1, InterpAlgo);
	}
	delete[]x, delete[]y;
	delete[]z;

	//Initialize temporal info
	for (int ii = 0; ii < nCams; ii++)
		CamTraj[ii].F = round(10.0*(1.0*rand() / RAND_MAX - 0.5));
	CamTraj[0].F = 0, CamTraj[1].F = -3.0, CamTraj[2].F = 2.0;

	printLOG("Inital offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printLOG("%f ", CamTraj[ii].F);
	printLOG("\n");


	ceres::Problem problem;
	double Error = 0.0;
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxnf
		int maxnf = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxnf < CamTraj[jj].Track3DInfo[ii].nf)
				maxnf = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		for (int kk = 0; kk < maxnf; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk > CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi > CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*maxPts + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*maxPts + 3 * lFi + ll] + fFi * AllCamCenter[3 * jj*maxPts + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*maxPts + 4 * lFi + ll];

				//QuaternionLinearInterp(&AllQuaternion[4 * jj*maxPts + 4 * lFi], &AllQuaternion[4 * jj*maxPts + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*maxPts, maxPts, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*maxPts, maxPts, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*maxPts, maxPts, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX * residualsX + residualsY * residualsY;
				Error += Residual;

				//ceres::CostFunction* cost_function = TemporalOptimInterpStationaryCameraCeres::Create(CamTraj[jj].P[kk].P, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].npts, InterpAlgo);
				ceres::CostFunction* cost_function = TemporalOptimInterpMovingCameraCeres::Create(&AllPMatrix[12 * jj*maxPts], &AllKMatrix[9 * jj*maxPts], &AllQuaternion[4 * jj*maxPts], &AllRotationMat[9 * jj*maxPts], &AllCamCenter[3 * jj*maxPts],
					ParaCamCenterX + jj * maxPts, ParaCamCenterY + jj * maxPts, ParaCamCenterZ + jj * maxPts, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].nf, InterpAlgo);
				problem.AddResidualBlock(cost_function, NULL, CamTraj[maxCam].Track3DInfo[ii].xyz + 3 * kk, &CamTraj[jj].F);
			}
		}
	}
	printLOG("Initial error: %.6e\n", Error);

	//printLOG("Setting fixed parameters...\n");
	problem.SetParameterBlockConstant(&CamTraj[0].F);

	//printLOG("Running optim..\n");
	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 50;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	Error = 0.0;
	FILE *fp = fopen("C:/temp/Sim/Results.txt", "w+");
	fprintf(fp, "Temporal alignment (ms): ");
	for (int ii = 0; ii < nCams; ii++)
		fprintf(fp, "%f ", CamTraj[ii].F);
	fprintf(fp, "\n");
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxnf
		int maxnf = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxnf < CamTraj[jj].Track3DInfo[ii].nf)
				maxnf = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		fprintf(fp, "3D track %d \n", ii);
		for (int kk = 0; kk < maxnf; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk > CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi > CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*maxPts + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*maxPts + 3 * lFi + ll] + fFi * AllCamCenter[3 * jj*maxPts + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*maxPts + 4 * lFi + ll];
				//QuaternionLinearInterp(&AllQuaternion[4 * jj*maxPts + 4 * lFi], &AllQuaternion[4 * jj*maxPts + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*maxPts, maxPts, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*maxPts, maxPts, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*maxPts, maxPts, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX * residualsX + residualsY * residualsY;
				Error += Residual;

				if (jj == 0)
					fprintf(fp, "%.4f %.4f %.4f ", CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2]);
			}
		}
		fprintf(fp, "\n");
	}
	printLOG("Final error: %.6e\n", Error);
	printLOG("Final offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printLOG("%f ", CamTraj[ii].F);
	printLOG("\n");

	//printLOG("Write results ....\n");
	//WriteTrajectory(CamTraj, nCams, npts, 1.0);
	delete[]AllRotationMat, delete[]AllPMatrix, delete[]AllKMatrix, delete[]AllQuaternion, delete[]AllCamCenter;
	delete[]ParaCamCenterX, delete[]ParaCamCenterY, delete[]ParaCamCenterZ;

	return 0;
}
int TemporalOptimInterpNew(char *Path, double *Offset)
{
	const int nCams = 3, npts = 4, startF = 0, stopF = 150;

	VideoData *VideoInfo = new VideoData[nCams];
	PerCamNonRigidTrajectory CamTraj[nCams];
	PrepareTrajectoryInfo(Path, VideoInfo, CamTraj, Offset, nCams, npts, startF, stopF);

	//Initialize temporal info
	for (int ii = 0; ii < nCams; ii++)
		CamTraj[ii].F = Offset[ii];

	int InterpAlgo = 1;
	double *x = new double[stopF], *y = new double[stopF], *z = new double[stopF];
	double* AllQuaternion = new double[4 * nCams*stopF], *AllRotationMat = new double[9 * nCams*stopF], *AllCamCenter = new double[3 * nCams*stopF];
	double *AllKMatrix = new double[9 * nCams*stopF], *AllPMatrix = new double[12 * nCams*stopF];
	double *ParaCamCenterX = new double[nCams*stopF], *ParaCamCenterY = new double[nCams*stopF], *ParaCamCenterZ = new double[nCams*stopF];

	for (int jj = 0; jj < nCams; jj++)
	{
		//Interpolate the trajectory of 2d tracks
		for (int ii = 0; ii < CamTraj[jj].npts; ii++)
		{
			int nf = CamTraj[jj].Track3DInfo[ii].nf;
			CamTraj[jj].Track2DInfo[ii].ParaX = new double[nf];
			CamTraj[jj].Track2DInfo[ii].ParaY = new double[nf];

			for (int kk = 0; kk < nf; kk++)
				x[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].x, y[kk] = CamTraj[jj].Track2DInfo[ii].uv[kk].y;
			Generate_Para_Spline(x, CamTraj[jj].Track2DInfo[ii].ParaX, nf, 1, InterpAlgo);
			Generate_Para_Spline(y, CamTraj[jj].Track2DInfo[ii].ParaY, nf, 1, InterpAlgo);
		}

		for (int ii = 0; ii < stopF; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			for (int kk = 0; kk < 9; kk++)
				AllKMatrix[9 * jj * stopF + 9 * ii + kk] = CamTraj[jj].K[ii].K[kk];
			for (int kk = 0; kk < 3; kk++)
				AllCamCenter[3 * jj * stopF + 3 * ii + kk] = CamTraj[jj].C[ii].C[kk];
			for (int kk = 0; kk < 9; kk++)
				AllRotationMat[9 * jj * stopF + 9 * ii + kk] = CamTraj[jj].R[ii].R[kk];
			for (int kk = 0; kk < 4; kk++)
				AllQuaternion[4 * jj * stopF + 4 * ii + kk] = CamTraj[jj].Q[ii].quad[kk];
			for (int kk = 0; kk < 12; kk++)
				AllPMatrix[12 * jj * stopF + 12 * ii + kk] = CamTraj[jj].P[ii].P[kk];
		}

		for (int ii = 0; ii < stopF; ii++)
		{
			if (ii >= CamTraj[jj].R.size())
				continue;
			x[ii] = AllCamCenter[3 * jj * stopF + 3 * ii];
			y[ii] = AllCamCenter[3 * jj * stopF + 3 * ii + 1];
			z[ii] = AllCamCenter[3 * jj * stopF + 3 * ii + 2];
		}
		Generate_Para_Spline(x, ParaCamCenterX + jj * stopF, stopF, 1, InterpAlgo);
		Generate_Para_Spline(y, ParaCamCenterY + jj * stopF, stopF, 1, InterpAlgo);
		Generate_Para_Spline(z, ParaCamCenterZ + jj * stopF, stopF, 1, InterpAlgo);
	}
	delete[]x, delete[]y, delete[]z;

	printLOG("Inital offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printLOG("%f ", CamTraj[ii].F);
	printLOG("\n");


	ceres::Problem problem;
	double Error = 0.0;
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxtracks
		int maxTracks = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxTracks < CamTraj[jj].Track3DInfo[ii].nf)
				maxTracks = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		for (int kk = 0; kk < maxTracks; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk > CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi > CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*stopF + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*stopF + 3 * lFi + ll] + fFi * AllCamCenter[3 * jj*stopF + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*stopF + 4 * lFi + ll];

				//QuaternionLinearInterp(&AllQuaternion[4 * jj*stopF + 4 * lFi], &AllQuaternion[4 * jj*stopF + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*stopF, stopF, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*stopF, stopF, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*stopF, stopF, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX * residualsX + residualsY * residualsY;
				Error += Residual;

				//ceres::CostFunction* cost_function = TemporalOptimInterpStationaryCameraCeres::Create(CamTraj[jj].P[kk].P, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].nf, InterpAlgo);
				ceres::CostFunction* cost_function = TemporalOptimInterpMovingCameraCeres::Create(&AllPMatrix[12 * jj*stopF], &AllKMatrix[9 * jj*stopF], &AllQuaternion[4 * jj*stopF], &AllRotationMat[9 * jj*stopF], &AllCamCenter[3 * jj*stopF],
					ParaCamCenterX + jj * stopF, ParaCamCenterY + jj * stopF, ParaCamCenterZ + jj * stopF, CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track2DInfo[ii].ParaY, kk, CamTraj[jj].Track3DInfo[ii].nf, InterpAlgo);
				problem.AddResidualBlock(cost_function, NULL, CamTraj[maxCam].Track3DInfo[ii].xyz + 3 * kk, &CamTraj[jj].F);
			}
		}
	}
	printLOG("Initial error: %.6e\n", Error);

	//printLOG("Setting fixed parameters...\n");
	problem.SetParameterBlockConstant(&CamTraj[0].F);

	//printLOG("Running optim..\n");
	ceres::Solver::Options options;
	options.num_threads = 4;
	options.max_num_iterations = 50;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.use_nonmonotonic_steps = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	Error = 0.0;
	FILE *fp = fopen("C:/temp/Sim/Results.txt", "w+");
	fprintf(fp, "Temporal alignment (ms): ");
	for (int ii = 0; ii < nCams; ii++)
		fprintf(fp, "%f ", CamTraj[ii].F);
	fprintf(fp, "\n");
	for (int ii = 0; ii < npts; ii++)
	{
		//find maxtracks
		int maxTracks = 0, maxCam = 0;
		for (int jj = 0; jj < nCams; jj++)
			if (maxTracks < CamTraj[jj].Track3DInfo[ii].nf)
				maxTracks = CamTraj[jj].Track3DInfo[ii].nf, maxCam = jj;

		fprintf(fp, "3D track %d \n", ii);
		for (int kk = 0; kk < maxTracks; kk++)
		{
			for (int jj = 0; jj < nCams; jj++)
			{
				if (kk > CamTraj[jj].Track3DInfo[ii].nf || kk >= CamTraj[jj].R.size())
					continue;

				double Fi = CamTraj[jj].F + kk;
				if (Fi < 0.0)
					Fi = 0.0;
				if (Fi > CamTraj[jj].Track3DInfo[ii].nf - 2)
					Fi = CamTraj[jj].Track3DInfo[ii].nf - 2;
				int lFi = (int)Fi, uFi = lFi + 1, rFi = (int)(Fi + 0.5);
				double fFi = Fi - lFi;

				double K[9], C[3], Q[4], R[9], RT[12], P[12];
				for (int ll = 0; ll < 9; ll++)
					K[ll] = AllKMatrix[9 * jj*stopF + 9 * lFi + ll];
				for (int ll = 0; ll < 3; ll++)
					C[ll] = (1.0 - fFi)*AllCamCenter[3 * jj*stopF + 3 * lFi + ll] + fFi * AllCamCenter[3 * jj*stopF + 3 * uFi + ll]; //linear interpolation

				for (int ll = 0; ll < 4; ll++)
					Q[ll] = AllQuaternion[4 * jj*stopF + 4 * lFi + ll];
				//QuaternionLinearInterp(&AllQuaternion[4 * jj*stopF + 4 * lFi], &AllQuaternion[4 * jj*stopF + 4 * uFi], Q, fFi);//linear interpolation
				//Get_Value_Spline(ParaCamCenterX + jj*stopF, stopF, 1, Fi, 0, &C[0], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterY + jj*stopF, stopF, 1, Fi, 0, &C[1], -1, InterpAlgo);
				//Get_Value_Spline(ParaCamCenterZ + jj*stopF, stopF, 1, Fi, 0, &C[2], -1, InterpAlgo);

				Quaternion2Rotation(Q, R);
				AssembleRT(R, C, RT, true);
				AssembleP(K, RT, P);

				double XYZ[] = { CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2] };
				double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
				double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
				double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

				double Sx, Sy;
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaX, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sx, -1, InterpAlgo);
				Get_Value_Spline(CamTraj[jj].Track2DInfo[ii].ParaY, CamTraj[jj].Track3DInfo[ii].nf, 1, Fi, 0, &Sy, -1, InterpAlgo);

				double residualsX = numX / denum - Sx;
				double residualsY = numY / denum - Sy;
				double Residual = residualsX * residualsX + residualsY * residualsY;
				Error += Residual;

				if (jj == 0)
					fprintf(fp, "%.4f %.4f %.4f ", CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 1], CamTraj[maxCam].Track3DInfo[ii].xyz[3 * kk + 2]);
			}
		}
		fprintf(fp, "\n");
	}
	printLOG("Final error: %.6e\n", Error);
	printLOG("Final offset: ");
	for (int ii = 0; ii < nCams; ii++)
		printLOG("%f ", CamTraj[ii].F);
	printLOG("\n");

	//printLOG("Write results ....\n");
	//WriteTrajectory(CamTraj, nCams, npts, 1.0);
	delete[]AllRotationMat, delete[]AllPMatrix, delete[]AllKMatrix, delete[]AllQuaternion, delete[]AllCamCenter;
	delete[]ParaCamCenterX, delete[]ParaCamCenterY, delete[]ParaCamCenterZ;

	return 0;
}

int CeresLeastActionNonlinearOptim()
{
	int totalPts = 100, nCams = 2, nPperTracks = 100, npts = 1;

	//double *AllQ, *AllU;
	int *PerTraj_nFrames = new int[npts];


	double lamdaI = 10.0;
	for (int ii = 0; ii < npts; ii++)
		PerTraj_nFrames[ii] = nPperTracks;

	double *parameters = new double[totalPts * 3 + nCams];

	ceres::GradientProblemSolver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 1000;

	ceres::GradientProblemSolver::Summary summary;
	//ceres::GradientProblem problem(new LeastActionProblem(AllQ, AllU, PerTraj_nFrames, totalPts, nCams, nPperTracks, npts, lamdaI));
	//ceres::Solve(options, problem, parameters, &summary);

	std::cout << summary.FullReport() << "\n";

	return 0;
}
double LeastActionError(double *xyz1, double *xyz2, double *timeStamp1, double *timeStamp2, double subframe1, double subframe2, int frameID1, int frameID2, double ialpha1, double ialpha2, double Tscale, double eps, int motionPriorPower)
{
	double difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
	double  t1 = (timeStamp1[0] + subframe1 + frameID1) * ialpha1*Tscale;
	double  t2 = (timeStamp2[0] + subframe2 + frameID2) * ialpha2*Tscale;

	double cost;
	if (motionPriorPower == 4)
		cost = pow(difX*difX + difY * difY + difZ * difZ, 2) / (pow(t2 - t1, 3) + eps); //mv^4*dt
	else if (motionPriorPower == 2)
		cost = (difX*difX + difY * difY + difZ * difZ) / (t2 - t1 + eps); //mv^2*dt

	return cost;
}
double LeastActionError(double *xyz1, double *xyz2, double timeStamp1, double timeStamp2, double eps, int motionPriorPower)
{
	double difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
	double cost;
	if (motionPriorPower == 4)
		cost = pow(difX*difX + difY * difY + difZ * difZ, 2) / (pow(timeStamp2 - timeStamp1, 3) + eps); //mv^4*dt
	else if (motionPriorPower == 2)
		cost = (difX*difX + difY * difY + difZ * difZ) / (timeStamp2 - timeStamp1 + eps); //mv^2*dt

	return cost;
}
double LeastActionError(double *xyz1, double *xyz2, double *xyz3, double *timeStamp1, double *timeStamp2, double *timeStamp3, int frameID1, int frameID2, int frameID3, double ialpha1, double ialpha2, double ialpha3, double Tscale, double eps, int motionPriorPower)
{
	double  t1 = (timeStamp1[0] + frameID1) * ialpha1*Tscale;
	double  t2 = (timeStamp2[0] + frameID2) * ialpha2*Tscale;
	double  t3 = (timeStamp3[0] + frameID3) * ialpha3*Tscale;
	Point3d X1(xyz1[0], xyz1[1], xyz1[2]), X2(xyz2[0], xyz2[1], xyz2[2]), X3(xyz3[0], xyz3[1], xyz3[2]);
	Point3d num1 = X2 - X3, denum1 = ProductPoint3d(X1 - X2, X1 - X3),
		num2 = 2.0*X2 - X1 - X3, denum2 = ProductPoint3d(X2 - X1, X2 - X3),
		num3 = X2 - X1, denum3 = ProductPoint3d(X3 - X1, X3 - X2);
	Point3d dv = ProductPoint3d(X1, DividePoint3d(num1, denum1)) + ProductPoint3d(X2, DividePoint3d(num2, denum2)) + ProductPoint3d(X3, DividePoint3d(num3, denum3));

	double cost;
	if (motionPriorPower == 4)
		cost = pow(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z, 2) / (pow(t2 - t1, 3) + eps); //mv^4*dt
	else if (motionPriorPower == 2)
		cost = (dv.x*dv.x + dv.y*dv.y + dv.z*dv.z) / (t2 - t1 + eps); //mv^2*dt

	return cost;
}
double Compute3DMotionPriorEnergy(vector<Point3d> &traj, vector<double>&Time, double eps, double g, int ApproxOrder)
{
	double Cost = 0.0;
	if (ApproxOrder == 1)
	{
		double T = 0.0, V = 0.0;
		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
			double dT = Time[ii + 1] - Time[ii];
			double dx = traj[ii].x - traj[ii + 1].x, dy = traj[ii].y - traj[ii + 1].y, dz = traj[ii].z - traj[ii + 1].z;
			double vx = dx / (dT + eps), vy = dy / (dT + eps), vz = dz / (dT + eps);
			double v = sqrt(vx*vx + vy * vy + vz * vz);
			double cost = pow(v, 2)*(dT + eps);
			T += cost;
		}
		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
			double dT = Time[ii + 1] - Time[ii];
			double cost = g * (traj[ii].y + traj[ii + 1].y)*dT;
			V += cost;
		}
		Cost = T - V;
	}
	else
	{
		/*//Second order approxiamtaion of speed
		Point3d vel[2000];
		for (int ii = 0; ii < traj.size() - 2; ii++)
		{
		double  t1 = Time[ii], t2 = Time[ii + 1], t3 = Time[ii + 2];

		double num1 = t2 - t3, num2 = 2.0*t2 - t1 - t3, num3 = t2 - t1;
		double denum1 = (t1 - t2)*(t1 - t3) + eps, denum2 = (t2 - t1)*(t2 - t3) + eps, denum3 = (t3 - t1)*(t3 - t2) + eps;
		double a1 = num1 / denum1, a2 = num2 / denum2, a3 = num3 / denum3;

		double velX = a1*traj[ii].x + a2*traj[ii + 1].x + a3*traj[ii + 2].x;
		double velY = a1*traj[ii].y + a2*traj[ii + 1].y + a3*traj[ii + 2].y;
		double velZ = a1*traj[ii].z + a2*traj[ii + 1].z + a3*traj[ii + 2].z;

		vel[ii + 1].x = velX, vel[ii + 1].y = velY, vel[ii + 1].z = velZ;
		}
		vel[0] = vel[1]; vel[traj.size() - 1] = vel[traj.size() - 2];

		for (int ii = 0; ii < traj.size() - 1; ii++)
		{
		Point3d vel1 = vel[ii], vel2 = vel[ii + 1];
		double dT = Time[ii + 1] - Time[ii];
		double cost = 0.5*(pow(vel1.x, 2) + pow(vel2.x, 2))*(dT + eps) + 0.5*(pow(vel1.y, 2) + pow(vel2.y, 2))*(dT + eps) + 0.5*(pow(vel1.z, 2) + pow(vel2.z, 2))*(dT + eps);
		Cost += cost;
		}*/

		//accelatiaon via lagrange interpolation
		double accel[2000];
		double t1, t2, t3, denum1, denum3;
		int npts = (int)traj.size();
		t1 = Time[0], t2 = Time[1];
		denum1 = pow(t2 - t1, 2);
		accel[0] = (-traj[0].x + traj[1].x) / denum1 + (-traj[0].y + traj[1].y) / denum1 + (-traj[0].z + traj[1].z) / denum1;

		for (int ii = 1; ii < npts - 1; ii++)
		{
			double  t1 = Time[ii - 1], t2 = Time[ii], t3 = Time[ii + 1];
			double denum1 = (t1 - t2)*(t1 - t3), denum2 = (t2 - t1)*(t2 - t3), denum3 = (t3 - t1)*(t3 - t2);

			double accelX = traj[ii - 1].x / denum1 + traj[ii].x / denum2 + traj[ii + 1].x / denum3;
			double accelY = traj[ii - 1].y / denum1 + traj[ii].y / denum2 + traj[ii + 1].y / denum3;
			double accelZ = traj[ii - 1].z / denum1 + traj[ii].z / denum2 + traj[ii + 1].z / denum3;

			accel[ii] = accelX * accelX + accelY * accelY + accelZ * accelZ;
		}
		t2 = Time[npts - 2], t3 = Time[npts - 1];
		denum3 = pow(t3 - t2, 2);
		accel[npts - 1] = (traj[npts - 2].x - traj[npts - 1].x) / denum3 + (traj[npts - 2].y - traj[npts - 1].y) / denum3 + (traj[npts - 2].z - traj[npts - 1].z) / denum3;

		Cost = 0.0;
		for (int ii = 0; ii < npts - 1; ii++)
			Cost += 0.5*(accel[ii] + accel[ii + 1])*(Time[ii + 1] - Time[ii]);
	}
	return Cost;
}
void RecursiveUpdateCameraOffset(int *currentOffset, int BruteForceTimeWindow, int currentCam, int nCams)
{
	if (currentOffset[currentCam] > BruteForceTimeWindow)
	{
		currentOffset[currentCam] = -BruteForceTimeWindow;
		if (currentCam < nCams - 1)
			currentOffset[currentCam + 1] ++;
		RecursiveUpdateCameraOffset(currentOffset, BruteForceTimeWindow, currentCam + 1, nCams);
	}

	return;
}

void MotionPrior_ML_Weighting(vector<ImgPtEle> *PerCam_UV, int ntracks, int nCams)
{
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				ImgPtEle ptEle = PerCam_UV[camID*ntracks + trackID][kk];
				ptEle.K[2] = 0, ptEle.K[5] = 0;

				double Scale = PerCam_UV[camID*ntracks + trackID][kk].scale, canonicalScale = PerCam_UV[camID*ntracks + trackID][kk].canonicalScale;
				double std2d = Scale / canonicalScale;

				double sigmaRetina[9], sigma2d[9] = { std2d, 0, 0, 0, std2d, 0, 0, 0, 1 };

				Map < Matrix < double, 3, 3, RowMajor > > eK(ptEle.K);
				Map < Matrix < double, 3, 3, RowMajor > > esigma2d(sigma2d);
				Map < Matrix < double, 3, 3, RowMajor > > esigmaRetina(sigmaRetina);
				Matrix3d eiK = eK.inverse();
				esigmaRetina = eiK * esigma2d*eiK.transpose();


				double depth = sqrt(pow(ptEle.pt3D.x - ptEle.camcenter[0], 2) + pow(ptEle.pt3D.y - ptEle.camcenter[1], 2) + pow(ptEle.pt3D.z - ptEle.camcenter[2], 2));
				double std3d = max(sigmaRetina[0], sigmaRetina[4]) *depth* 1000.0; //1000: um -> mm
				if (std3d < 0.0001)
					printLOG("(%d %d %d) ...", trackID, camID, kk);

				PerCam_UV[camID*ntracks + trackID][kk].std2D = std2d;
				PerCam_UV[camID*ntracks + trackID][kk].std3D = std3d;
			}
		}
	}

	return;
}
double MotionPrior_Optim_Init_SpatialStructure_Triangulation(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int ntracks, int nCams, double Tscale)
{
	vector<double> VectorTime;
	vector<int> VectorCamID, VectorFrameID;
	vector<ImgPtEle> Traj2DAll;
	vector<Point3f> xyz; vector<float>ref_t;

	float x, y, z, t; int cid, fid;
	Point3d P3D;
	ImgPtEle ptEle;
	double earliestTime, currentTime, RollingShutterOffset;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int PerCam_nf[MaxnCams], currentPID_InTrack[MaxnCams];
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int nopoints = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			PerCam_nf[camID] = PerCam_UV[camID*ntracks + trackID].size();

			if (PerCam_nf[camID] == 0)
				nopoints++;
		}
		if (nopoints > nCams - 2)
		{
			PerTraj_nFrames.push_back(0);
			continue;
		}

		xyz.clear(), ref_t.clear();
		VectorTime.clear(), VectorCamID.clear(), VectorFrameID.clear(), Traj2DAll.clear();

		//Assemble trajactory and time from all Cameras
		for (int jj = 0; jj < nCams; jj++)
			currentPID_InTrack[jj] = 0;

		while (true)
		{
			//Determine the next camera
			nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
			for (int camID = 0; camID < nCams; camID++)
			{
				if (currentPID_InTrack[camID] == PerCam_nf[camID])
				{
					nfinishedCams++;
					continue;
				}

				//Time:
				RollingShutterOffset = 0.0;
				if (PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].shutterModel != 0)
				{
					double v = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y,
						h = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight,
						p = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].rollingShutterPercent;
					RollingShutterOffset = v / h * p;
				}
				double ialpha = 1.0 / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].fps;
				frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
				currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

				if (currentTime < earliestTime)
				{
					earliestTime = currentTime;
					earliestCamID = camID;
					earliestCamFrameID = frameID;
				}
			}

			//If all cameras are done
			if (nfinishedCams == nCams)
				break;

			//Add new point to the sequence
			VectorTime.push_back(earliestTime);
			VectorCamID.push_back(earliestCamID);
			VectorFrameID.push_back(earliestCamFrameID);
			Traj2DAll.push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

			currentPID_InTrack[earliestCamID]++;
		}

		char Fname[512]; sprintf(Fname, "%s/Track3D/frameSynced_Track_%.4d.txt", Path, trackID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			Traj2DAll.clear();
			PerTraj_nFrames.push_back(0);
			continue;
		}
		while (fscanf(fp, "%f %f %f %f %d %d", &x, &y, &z, &t, &cid, &fid) != EOF)
			xyz.push_back(Point3f(x, y, z)), ref_t.push_back(t);
		fclose(fp);

		int npts = (int)Traj2DAll.size();
		Allpt3D[trackID] = new double[3 * npts];
		for (int ii = 0; ii < npts; ii++)
		{
			double dtime, mintime = 9e9; int minId = -1;
			for (int jj = 0; jj < (int)ref_t.size(); jj++)
			{
				dtime = abs(VectorTime[ii] - ref_t[jj]);
				if (dtime < mintime)
					minId = jj, mintime = dtime;
			}
			Allpt3D[trackID][3 * ii] = xyz[minId].x, Allpt3D[trackID][3 * ii + 1] = xyz[minId].y, Allpt3D[trackID][3 * ii + 2] = xyz[minId].z;
		}

		for (int ii = 0; ii < Traj2DAll.size(); ii++)
		{
			int camID = VectorCamID[ii], frameID = VectorFrameID[ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					Point3d xyz = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = xyz;

					//depth
					double lamda[3] = { (PerCam_UV[camID*ntracks + trackID][kk].pt3D.x - PerCam_UV[camID*ntracks + trackID][kk].camcenter[0]) / PerCam_UV[camID*ntracks + trackID][kk].ray[0],
						(PerCam_UV[camID*ntracks + trackID][kk].pt3D.y - PerCam_UV[camID*ntracks + trackID][kk].camcenter[1]) / PerCam_UV[camID*ntracks + trackID][kk].ray[1],
						(PerCam_UV[camID*ntracks + trackID][kk].pt3D.z - PerCam_UV[camID*ntracks + trackID][kk].camcenter[2]) / PerCam_UV[camID*ntracks + trackID][kk].ray[2] };

					//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
					int index[3] = { 0, 1, 2 };
					double rayDirect[3] = { abs(PerCam_UV[camID*ntracks + trackID][kk].ray[0]), abs(PerCam_UV[camID*ntracks + trackID][kk].ray[1]), abs(PerCam_UV[camID*ntracks + trackID][kk].ray[2]) };
					Quick_Sort_Double(rayDirect, index, 0, 2);

					PerCam_UV[camID*ntracks + trackID][kk].d = lamda[index[2]];

					found = true;
					break;
				}
			}
			if (!found)
				printLOG("Serious bug in point-camera-frame association\n");
		}
		PerTraj_nFrames.push_back(npts);
	}

	return 0;
}
void MotionPrior_Optim_SpatialStructure_NoSimulatenousPoints(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages, bool silent)
{
	double MotionPriorCost = 0.0, ProjCost = 0.0, costiX, costiY, costi;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<int>triangulatedList;
	vector<double>AllError3D, VectorTime;
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];


	ceres::Problem problem;

	double earliestTime, currentTime;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			for (int camID = 0; camID < nCams; camID++)
				PerCam_nf[camID] = (int)PerCam_UV[camID*ntracks + trackID].size();

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
			for (int jj = 0; jj < nCams; jj++)
				currentPID_InTrack[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentPID_InTrack[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					double ialpha = 1.0 / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].fps;
					currentTime = 1.0*currentOffset[camID] * ialpha*Tscale + 1.0*frameID * ialpha*Tscale;

					if (currentTime < earliestTime)
					{
						earliestTime = currentTime;
						earliestCamID = camID;
						earliestCamFrameID = frameID;
					}
				}

				//If all cameras are done
				if (nfinishedCams == nCams)
					break;

				//Add new point to the sequence
				VectorTime.push_back(earliestTime);
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		int npts = (int)Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		for (int ll = 0; ll < npts; ll++)
			Allpt3D[trackID][3 * ll] = Traj2DAll[trackID][ll].pt3D.x, Allpt3D[trackID][3 * ll + 1] = Traj2DAll[trackID][ll].pt3D.y, Allpt3D[trackID][3 * ll + 2] = Traj2DAll[trackID][ll].pt3D.z;

		//1st order approx of v
		double *Q1, *Q2, *U1, *U2;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];
			double ialpha1 = 1.0 / Traj2DAll[trackID][ll].fps, ialpha2 = 1.0 / Traj2DAll[trackID][ll + 1].fps;

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], 0, 0, VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MotionPriorCost += (1.0 - lamda)*costi;
			Q1 = Traj2DAll[trackID][ll].Q, Q2 = Traj2DAll[trackID][ll + 1].Q, U1 = Traj2DAll[trackID][ll].u, U2 = Traj2DAll[trackID][ll + 1].u;

			costiX = sqrt(lamda)*(Q1[0] * Allpt3D[trackID][3 * ll] + Q1[1] * Allpt3D[trackID][3 * ll + 1] + Q1[2] * Allpt3D[trackID][3 * ll + 2] - U1[0]);
			costiY = sqrt(lamda)*(Q1[3] * Allpt3D[trackID][3 * ll] + Q1[4] * Allpt3D[trackID][3 * ll + 1] + Q1[5] * Allpt3D[trackID][3 * ll + 2] - U1[1]);
			costi = sqrt(costiX*costiX + costiY * costiY);
			ProjCost += costi;

			costiX = sqrt(lamda)*(Q2[0] * Allpt3D[trackID][3 * ll + 3] + Q2[1] * Allpt3D[trackID][3 * ll + 4] + Q2[2] * Allpt3D[trackID][3 * ll + 5] - U2[0]);
			costiY = sqrt(lamda)*(Q2[3] * Allpt3D[trackID][3 * ll + 3] + Q2[4] * Allpt3D[trackID][3 * ll + 4] + Q2[5] * Allpt3D[trackID][3 * ll + 5] - U2[1]);
			costi = sqrt(costiX*costiX + costiY * costiY);
			ProjCost += costi;

			double  t1 = (currentOffset[camID1] + VectorFrameID[trackID][ll]) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + VectorFrameID[trackID][ll + 1]) * ialpha2*Tscale;

			ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateAutoDiff(t1, t2, eps, sqrt(1.0 - lamda)*RealOverSfm, motionPriorPower);
			//ceres::CostFunction* cost_function = LeastMotionPriorCostCeres::CreateNumerDiffSame(VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha, Tscale, rate, eps, motionPriorPower);
			problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + 1)]);

			ceres::CostFunction* cost_function2 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll].Q, Traj2DAll[trackID][ll].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function2, NULL, Allpt3D[trackID] + 3 * ll);

			ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + 1].Q, Traj2DAll[trackID][ll + 1].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function3, NULL, Allpt3D[trackID] + 3 * ll + 3);
		}

		/*//Set fixed parameters
		ceres::Solver::Options options;
		options.num_threads = 4;
		options.max_num_iterations = 1000;
		options.linear_solver_type = ceres::SPARSE_SCHUR; //SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;// silent ? false : true;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (!silent)
		cout << "Point: " << trackID << " (" << npts << ") frames " << summary.BriefReport() << "\n";

		for (int ii = 0; ii < npts; ii++)
		{
		int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

		bool found = false;
		for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
		{
		if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
		{
		//if (summary.final_cost < 1e7)
		PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
		//else
		//	PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(0, 0, 0);
		found = true;
		break;
		}
		}
		if (!found)
		{
		printLOG("Serious bug in point-camera-frame association\n");
		abort();
		}
		}*/
	}
	for (int trackID = 0; trackID < ntracks; trackID++)
		PerTraj_nFrames.push_back((int)Traj2DAll[trackID].size());

	ceres::Solver::Options options;
	options.num_threads = 4;
	options.num_linear_solver_threads = 4;
	options.max_num_iterations = 500;
	options.linear_solver_type = ceres::SPARSE_SCHUR; //SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;// silent ? false : true;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (!silent)
		std::cout << summary.FullReport() << "\n";
	else
		std::cout << summary.BriefReport() << "\n";

	//Save data
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = (int)Traj2DAll[trackID].size();
		for (int ii = 0; ii < npts; ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printLOG("Serious bug in point-camera-frame association\n");
				//abort();
			}
		}
	}

	//Compute cost after optim
	MotionPriorCost = 0.0, ProjCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerTraj_nFrames[trackID];
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + 1];
			double ialpha1 = 1.0 / Traj2DAll[trackID][ll].fps, ialpha2 = 1.0 / Traj2DAll[trackID][ll + 1].fps;

			costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], 0, 0, VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + 1], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MotionPriorCost += costi;

			costiX = sqrt(lamda)*(Traj2DAll[trackID][ll].Q[0] * Allpt3D[trackID][3 * ll] + Traj2DAll[trackID][ll].Q[1] * Allpt3D[trackID][3 * ll + 1] + Traj2DAll[trackID][ll].Q[2] * Allpt3D[trackID][3 * ll + 2] - Traj2DAll[trackID][ll].u[0]);
			costiY = sqrt(lamda)*(Traj2DAll[trackID][ll].Q[3] * Allpt3D[trackID][3 * ll] + Traj2DAll[trackID][ll].Q[4] * Allpt3D[trackID][3 * ll + 1] + Traj2DAll[trackID][ll].Q[5] * Allpt3D[trackID][3 * ll + 2] - Traj2DAll[trackID][ll].u[1]);
			costi = sqrt(costiX*costiX + costiY * costiY);
			ProjCost += costi;

			costiX = sqrt(lamda)*(Traj2DAll[trackID][ll + 1].Q[0] * Allpt3D[trackID][3 * ll + 3] + Traj2DAll[trackID][ll + 1].Q[1] * Allpt3D[trackID][3 * ll + 4] + Traj2DAll[trackID][ll + 1].Q[2] * Allpt3D[trackID][3 * ll + 5] - Traj2DAll[trackID][ll + 1].u[0]);
			costiY = sqrt(lamda)*(Traj2DAll[trackID][ll + 1].Q[3] * Allpt3D[trackID][3 * ll + 3] + Traj2DAll[trackID][ll + 1].Q[4] * Allpt3D[trackID][3 * ll + 4] + Traj2DAll[trackID][ll + 1].Q[5] * Allpt3D[trackID][3 * ll + 5] - Traj2DAll[trackID][ll + 1].u[1]);
			costi = sqrt(costiX*costiX + costiY * costiY);
			ProjCost += costi;
		}
	}

	if (!silent)
		printLOG("Motion cost: %f \nProjection cost: %f ", MotionPriorCost, ProjCost);

	double lengthCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = (int)Traj2DAll[trackID].size();
		double costi;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			costi = sqrt(pow(Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], 2) + pow(Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], 2) + pow(Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5], 2));
			lengthCost += costi;
		}
	}
	if (!silent)
		printLOG("Distance Cost: %e\n", lengthCost);

	double directionCost = 0.0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = (int)Traj2DAll[trackID].size();
		double direct1[3], direct2[3];
		for (int ll = 0; ll < npts - 2; ll++)
		{
			direct1[0] = Allpt3D[trackID][3 * ll] - Allpt3D[trackID][3 * ll + 3], direct1[1] = Allpt3D[trackID][3 * ll + 1] - Allpt3D[trackID][3 * ll + 4], direct1[2] = Allpt3D[trackID][3 * ll + 2] - Allpt3D[trackID][3 * ll + 5];
			direct2[0] = Allpt3D[trackID][3 * ll + 3] - Allpt3D[trackID][3 * ll + 6], direct2[1] = Allpt3D[trackID][3 * ll + 4] - Allpt3D[trackID][3 * ll + 7], direct2[2] = Allpt3D[trackID][3 * ll + 5] - Allpt3D[trackID][3 * ll + 8];
			normalize(direct1), normalize(direct2);
			directionCost += abs(dotProduct(direct1, direct2));
		}
	}
	if (!silent)
		printLOG("Direction Cost: %e\n", directionCost);

	Cost[0] = MotionPriorCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]currentFrame, delete[]PerCam_nf;

	return;
}
double MotionPrior_Optim_SpatialStructure_Algebraic(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages, bool silent)
{
	vector<double> *VectorTime = new vector<double>[ntracks];
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks], *simulatneousPoints = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	double CeresCost = 0.0;
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic,1)
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		bool UsedCam[MaxnCams];
		int PerCam_nf[MaxnCams], currentPID_InTrack[MaxnCams];
		Point3d P3D;
		ImgPtEle ptEle;
		double earliestTime, currentTime, RollingShutterOffset;
		int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

		ceres::Problem problem;
		if (StillImages)
		{
			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime[trackID].push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			int nopoints = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				PerCam_nf[camID] = (int)PerCam_UV[camID*ntracks + trackID].size();

				if (PerCam_nf[camID] == 0)
					nopoints++;
			}
			if (nopoints > nCams - 2)
				continue;

			//Assemble trajactory and time from all Cameras
			for (int jj = 0; jj < nCams; jj++)
				currentPID_InTrack[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentPID_InTrack[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					RollingShutterOffset = 0.0;
					if (PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].shutterModel != 0)
					{
						double v = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y,
							h = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight,
							p = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].rollingShutterPercent;
						RollingShutterOffset = v / h * p;
					}
					double ialpha = 1.0 / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].fps;
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

					if (currentTime < earliestTime)
					{
						earliestTime = currentTime;
						earliestCamID = camID;
						earliestCamFrameID = frameID;
					}
				}

				//If all cameras are done
				if (nfinishedCams == nCams)
					break;

				//Add new point to the sequence
				VectorTime[trackID].push_back(earliestTime);
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		int npts = (int)Traj2DAll[trackID].size();
		Allpt3D[trackID] = new double[3 * npts];
		for (int ll = 0; ll < npts; ll++)
			Allpt3D[trackID][3 * ll] = Traj2DAll[trackID][ll].pt3D.x + gaussian_noise(0.0, 1.0) / RealOverSfm,
			Allpt3D[trackID][3 * ll + 1] = Traj2DAll[trackID][ll].pt3D.y + gaussian_noise(0.0, 1.0) / RealOverSfm,
			Allpt3D[trackID][3 * ll + 2] = Traj2DAll[trackID][ll].pt3D.z + gaussian_noise(0.0, 1.0) / RealOverSfm;

		//Detect points captured simulatenously
		int groupCount = 0;
		for (int ll = 0; ll < npts; ll++)
			simulatneousPoints[trackID].push_back(-1);

		for (int ll = 0; ll < npts - 1; ll++)
		{
			int naddedPoints = 0; bool found = false;
			for (int kk = ll + 1; kk < npts; kk++)
			{
				naddedPoints++;
				if (VectorTime[trackID][kk] - VectorTime[trackID][ll] > FLT_EPSILON)
					break;
				else
				{
					if (kk - 1 == ll)
						simulatneousPoints[trackID][ll] = groupCount;

					simulatneousPoints[trackID][kk] = groupCount;
					found = true;
				}
			}

			if (found)
			{
				ll += naddedPoints - 1;
				groupCount++;
			}
		}

		double MotionPriorCost = 0, ProjCost = 0;

		//1st order approx of v
		int oldtype = simulatneousPoints[trackID][0];
		for (int ll = 0; ll < nCams; ll++)
			UsedCam[ll] = false;

		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				else
				{
					ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + incre].Q, Traj2DAll[trackID][ll + incre].u, sqrt(lamda));
					problem.AddResidualBlock(cost_function3, NULL, Allpt3D[trackID] + 3 * ll);
				}
				incre++;
			}
			if (ll + incre == npts)
				break;

			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[trackID][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[trackID][ll].pt2D.y / Traj2DAll[trackID][ll].imHeight*Traj2DAll[trackID][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[trackID][ll + incre].pt2D.y / Traj2DAll[trackID][ll + incre].imHeight*Traj2DAll[trackID][ll + incre].rollingShutterPercent;
			if (Traj2DAll[trackID][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[trackID][ll].pt2D.y - 0.5*Traj2DAll[trackID][ll].imHeight) *Traj2DAll[trackID][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[trackID][ll + incre].pt2D.y - 0.5* Traj2DAll[trackID][ll + incre].imHeight)*Traj2DAll[trackID][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[trackID][ll].fps;
			double ialpha2 = 1.0 / Traj2DAll[trackID][ll + incre].fps;
			double  t1 = (currentOffset[camID1] + VectorFrameID[trackID][ll] + shutterOffset1) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + VectorFrameID[trackID][ll + incre] + shutterOffset2) * ialpha2*Tscale;

			Point3d P3d_1(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			Point3d P3d_2(Allpt3D[trackID][3 * (ll + incre)], Allpt3D[trackID][3 * (ll + incre) + 1], Allpt3D[trackID][3 * (ll + incre) + 2]);
			double difX = P3d_2.x - P3d_1.x, difY = P3d_2.y - P3d_1.y, difZ = P3d_2.z - P3d_1.z;
			double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
			MotionPriorCost += costi * (1.0 - lamda)*RealOverSfm*RealOverSfm;

			Point2d P2d_1(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			double costx1 = (Traj2DAll[trackID][ll].Q[0] * P3d_1.x + Traj2DAll[trackID][ll].Q[1] * P3d_1.y + Traj2DAll[trackID][ll].Q[2] * P3d_1.z - Traj2DAll[trackID][ll].u[0]);
			double costy1 = (Traj2DAll[trackID][ll].Q[3] * P3d_1.x + Traj2DAll[trackID][ll].Q[4] * P3d_1.y + Traj2DAll[trackID][ll].Q[5] * P3d_1.z - Traj2DAll[trackID][ll].u[1]);
			ProjCost += lamda * (costx1*costx1 + costy1 * costy1);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);

			Point2d P2d_2(Traj2DAll[trackID][(ll + incre)].pt2D.x, Traj2DAll[trackID][(ll + incre)].pt2D.y);
			double costx = (Traj2DAll[trackID][(ll + incre)].Q[0] * P3d_2.x + Traj2DAll[trackID][(ll + incre)].Q[1] * P3d_2.y + Traj2DAll[trackID][(ll + incre)].Q[2] * P3d_2.z - Traj2DAll[trackID][(ll + incre)].u[0]);
			double costy = (Traj2DAll[trackID][(ll + incre)].Q[3] * P3d_2.x + Traj2DAll[trackID][(ll + incre)].Q[4] * P3d_2.y + Traj2DAll[trackID][(ll + incre)].Q[5] * P3d_2.z - Traj2DAll[trackID][(ll + incre)].u[1]);
			ProjCost += lamda * (costx*costx + costy * costy);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);

			ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateAutoDiff(t1, t2, eps, sqrt(1.0 - lamda)*RealOverSfm, motionPriorPower);
			//ceres::CostFunction* cost_function = LeastMotionPriorCostCeres::CreateNumerDiffSame(t1, t2, eps);
			problem.AddResidualBlock(cost_function, NULL, &Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + incre)]);

			ceres::CostFunction* cost_function2 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll].Q, Traj2DAll[trackID][ll].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function2, NULL, &Allpt3D[trackID][3 * ll]);

			ceres::CostFunction* cost_function3 = IdealAlgebraicReprojectionCeres::Create(Traj2DAll[trackID][ll + incre].Q, Traj2DAll[trackID][ll + incre].u, sqrt(lamda));
			problem.AddResidualBlock(cost_function3, NULL, &Allpt3D[trackID][3 * (ll + incre)]);

			ll += incre - 1;
		}

		ceres::Solver::Options options;
		options.num_threads = 2;
		options.num_linear_solver_threads = 2;
		options.max_num_iterations = 2000;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

#pragma omp critical
		{
			CeresCost += summary.final_cost;
			if (!silent)
				cout << "Point: " << trackID << " " << summary.BriefReport() << "\n";// cout << "Point: " << trackID << " " << summary.FullReport() << "\n";
		}

		//copy simultaneous triggered points
		for (int ll = 0; ll < npts - 1; ll++)
		{
			if (simulatneousPoints[trackID][ll] != -1)
			{
				int nPoint = 0;
				for (int kk = ll + 1; kk < npts; kk++)
				{
					nPoint++;
					if (simulatneousPoints[trackID][kk] != simulatneousPoints[trackID][ll])
						break;
					else
					{
						Allpt3D[trackID][3 * kk] = Allpt3D[trackID][3 * ll];
						Allpt3D[trackID][3 * kk + 1] = Allpt3D[trackID][3 * ll + 1];
						Allpt3D[trackID][3 * kk + 2] = Allpt3D[trackID][3 * ll + 2];
					}
				}
				ll += nPoint - 1;
			}
		}

		//cvTime. cvCam;
		for (int ii = 0; ii < Traj2DAll[trackID].size(); ii++)
		{
			int camID = VectorCamID[trackID][ii], frameID = VectorFrameID[trackID][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					Point3d xyz = Point3d(Allpt3D[trackID][3 * ii], Allpt3D[trackID][3 * ii + 1], Allpt3D[trackID][3 * ii + 2]);
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = xyz;
					found = true;
					break;
				}
			}
			if (!found)
			{
				printLOG("Serious bug in point-camera-frame association\n");
				//abort()
			}
		}
	}
	//printLOG("Alge: END\n");
	PerTraj_nFrames.clear();
	for (int trackID = 0; trackID < ntracks; trackID++)
		PerTraj_nFrames.push_back((int)Traj2DAll[trackID].size());

	//Compute cost after optim
	double MotionPriorCost = 0.0, ProjCost = 0.0, lengthCost = 0.0, directionCost = 0;
	int allnpts = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerTraj_nFrames[trackID];
		if (npts == 0)
			continue;

		int oldtype = simulatneousPoints[trackID][0];
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == npts)
				break;

			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[trackID][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[trackID][ll].pt2D.y / Traj2DAll[trackID][ll].imHeight*Traj2DAll[trackID][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[trackID][ll + incre].pt2D.y / Traj2DAll[trackID][ll + incre].imHeight*Traj2DAll[trackID][ll + incre].rollingShutterPercent;
			if (Traj2DAll[trackID][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[trackID][ll].pt2D.y - 0.5*Traj2DAll[trackID][ll].imHeight) *Traj2DAll[trackID][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[trackID][ll + incre].pt2D.y - 0.5* Traj2DAll[trackID][ll + incre].imHeight)*Traj2DAll[trackID][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[trackID][ll].fps;
			double ialpha2 = 1.0 / Traj2DAll[trackID][ll + incre].fps;
			double costi = LeastActionError(&Allpt3D[trackID][3 * ll], &Allpt3D[trackID][3 * (ll + incre)],
				&currentOffset[camID1], &currentOffset[camID2], shutterOffset1, shutterOffset2,
				VectorFrameID[trackID][ll], VectorFrameID[trackID][ll + incre], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MotionPriorCost += costi;
			allnpts++;
			ll += incre - 1;
		}
	}
	if (!silent)
		printLOG("Motion cost: %e ", MotionPriorCost / allnpts);

	allnpts = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerTraj_nFrames[trackID];
		if (npts == 0)
			continue;

		double costi;
		for (int ll = 0; ll < npts; ll++)
		{
			Point2d P2d_1(Traj2DAll[trackID][ll].pt2D.x, Traj2DAll[trackID][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[trackID][3 * ll], Allpt3D[trackID][3 * ll + 1], Allpt3D[trackID][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[trackID][ll].P, P3d_1, P2d_1);
			ProjCost += costi;
		}
		allnpts += npts;
	}
	if (!silent)
		printLOG("Projection cost: %e ", ProjCost / allnpts);

	Cost[0] = MotionPriorCost, Cost[1] = ProjCost, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorTime, delete[]VectorCamID, delete[]VectorFrameID, delete[]simulatneousPoints, delete[]Traj2DAll;

	return CeresCost;
}
double MotionPrior_Optim_SpatialStructure_Rays(char *Path, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int ntracks, bool non_monotonicDescent, int nCams, double Tscale, double eps, double lamdaInterCam, double lamdaIntraCam, double RealOverSfm, double *Cost, bool StillImages, bool silent)
{
	vector<double> *VectorTime = new vector<double>[ntracks];
	vector<int> *VectorCamID = new vector<int>[ntracks], *VectorFrameID = new vector<int>[ntracks], *simulatneousPoints = new vector<int>[ntracks];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[ntracks];

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	double CeresCost = 0.0;
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic,1)
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		if (PerTraj_nFrames[trackID] == 0)
			continue;
		bool UsedCam[MaxnCams];
		int PerCam_nf[MaxnCams], currentPID_InTrack[MaxnCams];
		Point3d P3D;
		ImgPtEle ptEle;
		double earliestTime, currentTime, RollingShutterOffset;
		int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

		ceres::Problem problem;
		if (StillImages)
		{
			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime[trackID].push_back(currentTime);
				VectorCamID[trackID].push_back(camID);
				VectorFrameID[trackID].push_back(0);
				Traj2DAll[trackID].push_back(PerCam_UV[camID*ntracks + trackID][0]);
			}
		}
		else
		{
			int nopoints = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				PerCam_nf[camID] = (int)PerCam_UV[camID*ntracks + trackID].size();
				if (PerCam_nf[camID] == 0)
					nopoints++;
			}
			if (nopoints > nCams - 2)
				continue;

			//Assemble trajactory and time from all Cameras
			for (int jj = 0; jj < nCams; jj++)
				currentPID_InTrack[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentPID_InTrack[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					RollingShutterOffset = 0.0;
					if (PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].shutterModel != 0)
					{
						double v = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].pt2D.y,
							h = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].imHeight,
							p = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].rollingShutterPercent;
						RollingShutterOffset = v / h * p;
					}
					double ialpha = 1.0 / PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].fps;
					frameID = PerCam_UV[camID*ntracks + trackID][currentPID_InTrack[camID]].frameID;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

					if (currentTime < earliestTime)
					{
						earliestTime = currentTime;
						earliestCamID = camID;
						earliestCamFrameID = frameID;
					}
				}

				//If all cameras are done
				if (nfinishedCams == nCams)
					break;

				//Add new point to the sequence
				VectorTime[trackID].push_back(earliestTime);
				VectorCamID[trackID].push_back(earliestCamID);
				VectorFrameID[trackID].push_back(earliestCamFrameID);
				Traj2DAll[trackID].push_back(PerCam_UV[earliestCamID*ntracks + trackID][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}
		int npts = (int)Traj2DAll[trackID].size();

		//Detect points captured simulatenously
		int groupCount = 0;
		for (int ll = 0; ll < npts; ll++)
			simulatneousPoints[trackID].push_back(-1);
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int naddedPoints = 0; bool found = false;
			for (int kk = ll + 1; kk < npts; kk++)
			{
				naddedPoints++;
				if (VectorTime[trackID][kk] - VectorTime[trackID][ll] > FLT_EPSILON)
					break;
				else
				{
					if (kk - 1 == ll)
						simulatneousPoints[trackID][ll] = groupCount;

					simulatneousPoints[trackID][kk] = groupCount;
					found = true;
				}
			}

			if (found)
			{
				ll += naddedPoints - 1;
				groupCount++;
			}
		}

		//1st order approx of v
		double MPCostInterB = 0, MPCostIntraB = 0;
		int oldtype = simulatneousPoints[trackID][0];
		for (int ll = 0; ll < nCams; ll++)
			UsedCam[ll] = false;

		//inter-cam
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				else
				{
					double lamdaIntersection = 1.0; //To enforce triangulation: this is actually a big number compared to non-triangulated points since the cost does not divide the time differnce
					ceres::CostFunction* cost_function = MotionPriorRaysCeres::Create(Traj2DAll[trackID][ll].ray, Traj2DAll[trackID][ll + incre].ray, Traj2DAll[trackID][ll].camcenter, Traj2DAll[trackID][ll + incre].camcenter, 0.0, 0.0, 1.0, lamdaIntersection);
					problem.AddResidualBlock(cost_function, NULL, &Traj2DAll[trackID][ll].d, &Traj2DAll[trackID][ll + incre].d);
				}
				incre++;
			}
			if (ll + incre == npts)
				break;

			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];
			int fid1 = VectorFrameID[trackID][ll], fid2 = VectorFrameID[trackID][ll + incre];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[trackID][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[trackID][ll].pt2D.y / Traj2DAll[trackID][ll].imHeight*Traj2DAll[trackID][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[trackID][ll + incre].pt2D.y / Traj2DAll[trackID][ll + incre].imHeight*Traj2DAll[trackID][ll + incre].rollingShutterPercent;
			if (Traj2DAll[trackID][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[trackID][ll].pt2D.y - 0.5*Traj2DAll[trackID][ll].imHeight) *Traj2DAll[trackID][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[trackID][ll + incre].pt2D.y - 0.5* Traj2DAll[trackID][ll + incre].imHeight)*Traj2DAll[trackID][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[trackID][ll].fps, ialpha2 = 1.0 / Traj2DAll[trackID][ll + incre].fps;
			double  t1 = (currentOffset[camID1] + fid1 + shutterOffset1) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + fid2 + shutterOffset2) * ialpha2*Tscale;

			double d1 = Traj2DAll[trackID][ll].d, d2 = Traj2DAll[trackID][ll + incre].d;
			double *r1 = Traj2DAll[trackID][ll].ray, *r2 = Traj2DAll[trackID][ll + incre].ray;
			double *c1 = Traj2DAll[trackID][ll].camcenter, *c2 = Traj2DAll[trackID][ll + incre].camcenter;
			double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
			double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
			double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

			double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
			MPCostInterB += costi;

			ceres::CostFunction* cost_function = MotionPriorRaysCeres::Create(Traj2DAll[trackID][ll].ray, Traj2DAll[trackID][ll + incre].ray, Traj2DAll[trackID][ll].camcenter, Traj2DAll[trackID][ll + incre].camcenter, t1, t2, eps, sqrt(lamdaInterCam));
			problem.AddResidualBlock(cost_function, NULL, &Traj2DAll[trackID][ll].d, &Traj2DAll[trackID][ll + incre].d);

			ll += incre - 1;
		}

		///intra-camera
		vector<int>IntraID;
		if (lamdaInterCam / (lamdaInterCam + lamdaIntraCam) < 0.99)
		{
			for (int cid = 0; cid < nCams; cid++)
			{
				IntraID.clear();
				for (int ll = 0; ll < npts; ll++)
					if (VectorCamID[trackID][ll] == cid)
						IntraID.push_back(ll);

				for (int ll = 0; ll < (int)IntraID.size() - 1; ll++)
				{
					int id1 = IntraID[ll], id2 = IntraID[ll + 1], camID = VectorCamID[trackID][ll];
					int fid1 = VectorFrameID[trackID][id1], fid2 = VectorFrameID[trackID][id2];

					double shutterOffset1 = 0, shutterOffset2 = 0;
					//if (Traj2DAll[trackID][id1].shutterModel != 0)
					//	shutterOffset1 = Traj2DAll[trackID][id1].pt2D.y / Traj2DAll[trackID][id1].imHeight*Traj2DAll[trackID][id1].rollingShutterPercent,
					//	shutterOffset2 = Traj2DAll[trackID][id2].pt2D.y / Traj2DAll[trackID][id2].imHeight*Traj2DAll[trackID][id2].rollingShutterPercent;
					if (Traj2DAll[trackID][id1].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
						shutterOffset1 = 0.5 + (Traj2DAll[trackID][id1].pt2D.y - 0.5*Traj2DAll[trackID][id1].imHeight) * Traj2DAll[trackID][id1].rollingShutterPercent,
						shutterOffset2 = 0.5 + (Traj2DAll[trackID][id2].pt2D.y - 0.5* Traj2DAll[trackID][id2].imHeight) * Traj2DAll[trackID][id2].rollingShutterPercent;

					double ialpha = 1.0 / Traj2DAll[trackID][id1].fps;
					double  t1 = (currentOffset[camID] + fid1 + shutterOffset1) * ialpha*Tscale;
					double  t2 = (currentOffset[camID] + fid2 + shutterOffset2) * ialpha*Tscale;

					double d1 = Traj2DAll[trackID][id1].d, d2 = Traj2DAll[trackID][id2].d;
					double *r1 = Traj2DAll[trackID][id1].ray, *r2 = Traj2DAll[trackID][id2].ray;
					double *c1 = Traj2DAll[trackID][id1].camcenter, *c2 = Traj2DAll[trackID][id2].camcenter;
					double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
					double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
					double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

					double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
					MPCostIntraB += costi;

					ceres::CostFunction* cost_function = MotionPriorRaysCeres::Create(Traj2DAll[trackID][id1].ray, Traj2DAll[trackID][id2].ray, Traj2DAll[trackID][id1].camcenter, Traj2DAll[trackID][id2].camcenter, t1, t2, eps, sqrt(lamdaIntraCam));
					problem.AddResidualBlock(cost_function, NULL, &Traj2DAll[trackID][id1].d, &Traj2DAll[trackID][id2].d);
				}
			}
		}

		ceres::Solver::Options options;
		options.num_threads = 2;
		options.num_linear_solver_threads = 2;
		options.max_num_iterations = 3000;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;/////ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
		options.preconditioner_type = ceres::JACOBI;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		double MPCostInterA = 0, MPCostIntraA = 0;
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == npts)
				break;

			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];
			int fid1 = VectorFrameID[trackID][ll], fid2 = VectorFrameID[trackID][ll + incre];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[trackID][ll].shutterModel != 0)
			//shutterOffset1 = Traj2DAll[trackID][ll].pt2D.y / Traj2DAll[trackID][ll].imHeight*Traj2DAll[trackID][ll].rollingShutterPercent,
			//shutterOffset2 = Traj2DAll[trackID][ll + incre].pt2D.y / Traj2DAll[trackID][ll + incre].imHeight*Traj2DAll[trackID][ll + incre].rollingShutterPercent;
			if (Traj2DAll[trackID][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[trackID][ll].pt2D.y - 0.5*Traj2DAll[trackID][ll].imHeight) *Traj2DAll[trackID][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[trackID][ll + incre].pt2D.y - 0.5* Traj2DAll[trackID][ll + incre].imHeight)*Traj2DAll[trackID][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[trackID][ll].fps, ialpha2 = 1.0 / Traj2DAll[trackID][ll + incre].fps;
			double  t1 = (currentOffset[camID1] + fid1 + shutterOffset1) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + fid2 + shutterOffset2) * ialpha2*Tscale;

			double d1 = Traj2DAll[trackID][ll].d, d2 = Traj2DAll[trackID][ll + incre].d;
			double *r1 = Traj2DAll[trackID][ll].ray, *r2 = Traj2DAll[trackID][ll + incre].ray;
			double *c1 = Traj2DAll[trackID][ll].camcenter, *c2 = Traj2DAll[trackID][ll + incre].camcenter;
			double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
			double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
			double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

			double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
			MPCostInterA += costi;

			ll += incre - 1;
		}

		if (lamdaInterCam / (lamdaInterCam + lamdaIntraCam) < 0.99)
		{
			for (int cid = 0; cid < nCams; cid++)
			{
				IntraID.clear();
				for (int ll = 0; ll < npts; ll++)
					if (VectorCamID[trackID][ll] == cid)
						IntraID.push_back(ll);

				for (int ll = 0; ll < (int)IntraID.size() - 1; ll++)
				{
					int id1 = IntraID[ll], id2 = IntraID[ll + 1], camID = VectorCamID[trackID][ll];
					int fid1 = VectorFrameID[trackID][id1], fid2 = VectorFrameID[trackID][id2];

					double shutterOffset1 = 0, shutterOffset2 = 0;
					//if (Traj2DAll[trackID][id1].shutterModel != 0)
					//shutterOffset1 = Traj2DAll[trackID][id1].pt2D.y / Traj2DAll[trackID][id1].imHeight*Traj2DAll[trackID][id1].rollingShutterPercent,
					//shutterOffset2 = Traj2DAll[trackID][id2].pt2D.y / Traj2DAll[trackID][id2].imHeight*Traj2DAll[trackID][id2].rollingShutterPercent;
					if (Traj2DAll[trackID][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
						shutterOffset1 = 0.5 + (Traj2DAll[trackID][id1].pt2D.y - 0.5*Traj2DAll[trackID][id1].imHeight) *Traj2DAll[trackID][id1].rollingShutterPercent,
						shutterOffset2 = 0.5 + (Traj2DAll[trackID][id2].pt2D.y - 0.5* Traj2DAll[trackID][id2].imHeight)*Traj2DAll[trackID][id2].rollingShutterPercent;

					double ialpha = 1.0 / Traj2DAll[trackID][id1].fps;
					double  t1 = (currentOffset[camID] + fid1 + shutterOffset1) * ialpha*Tscale;
					double  t2 = (currentOffset[camID] + fid2 + shutterOffset2) * ialpha*Tscale;

					double d1 = Traj2DAll[trackID][id1].d, d2 = Traj2DAll[trackID][id2].d;
					double *r1 = Traj2DAll[trackID][id1].ray, *r2 = Traj2DAll[trackID][id2].ray;
					double *c1 = Traj2DAll[trackID][id1].camcenter, *c2 = Traj2DAll[trackID][id2].camcenter;
					double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
					double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
					double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

					double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
					MPCostIntraA += costi;
				}
			}
		}

#pragma omp critical
		{
			CeresCost += summary.final_cost;
			if (!silent)
				cout << "Point " << trackID << ": " << summary.BriefReport() << "\n";
		}

		//store data
		for (int ll = 0; ll < Traj2DAll[trackID].size(); ll++)
		{
			int camID = VectorCamID[trackID][ll], frameID = VectorFrameID[trackID][ll];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*ntracks + trackID].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*ntracks + trackID][kk].frameID)
				{
					double d = Traj2DAll[trackID][ll].d;
					double *r = Traj2DAll[trackID][ll].ray, *c = Traj2DAll[trackID][ll].camcenter;
					double X = d * r[0] + c[0], Y = d * r[1] + c[1], Z = d * r[2] + c[2];

					PerCam_UV[camID*ntracks + trackID][kk].d = d;
					PerCam_UV[camID*ntracks + trackID][kk].pt3D = Point3d(X, Y, Z);
					found = true;
					break;
				}
			}
			if (!found)
				printLOG("Serious bug in point-camera-frame association\n");
		}
	}
	//printLOG("Alge: END\n");
	PerTraj_nFrames.clear();
	for (int trackID = 0; trackID < ntracks; trackID++)
		PerTraj_nFrames.push_back((int)Traj2DAll[trackID].size());

	//Compute cost after optim
	double MotionPriorCost = 0.0, ProjCost = 0.0, lengthCost = 0.0, directionCost = 0;
	int allnpts = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		int npts = PerTraj_nFrames[trackID];
		if (npts == 0)
			continue;

		int oldtype = simulatneousPoints[trackID][0];
		for (int ll = 0; ll < npts - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < npts)
			{
				if (simulatneousPoints[trackID][ll + incre] == -1 || simulatneousPoints[trackID][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[trackID][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == npts)
				break;

			int camID1 = VectorCamID[trackID][ll], camID2 = VectorCamID[trackID][ll + incre];
			int fid1 = VectorFrameID[trackID][ll], fid2 = VectorFrameID[trackID][ll + incre];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[trackID][ll].shutterModel != 0)
			//shutterOffset1 = Traj2DAll[trackID][ll].pt2D.y / Traj2DAll[trackID][ll].imHeight*Traj2DAll[trackID][ll].rollingShutterPercent,
			//shutterOffset2 = Traj2DAll[trackID][ll + incre].pt2D.y / Traj2DAll[trackID][ll + incre].imHeight*Traj2DAll[trackID][ll + incre].rollingShutterPercent;
			if (Traj2DAll[trackID][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[trackID][ll].pt2D.y - 0.5*Traj2DAll[trackID][ll].imHeight) *Traj2DAll[trackID][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[trackID][ll + incre].pt2D.y - 0.5* Traj2DAll[trackID][ll + incre].imHeight)*Traj2DAll[trackID][ll + incre].rollingShutterPercent;

			double ialpha = 1.0 / Traj2DAll[trackID][ll].fps;
			double  t1 = (currentOffset[camID1] + fid1 + shutterOffset1) * ialpha*Tscale;
			double  t2 = (currentOffset[camID2] + fid2 + shutterOffset2) * ialpha*Tscale;

			double d1 = Traj2DAll[trackID][ll].d, d2 = Traj2DAll[trackID][ll + incre].d;
			double *r1 = Traj2DAll[trackID][ll].ray, *r2 = Traj2DAll[trackID][ll + incre].ray;
			double *c1 = Traj2DAll[trackID][ll].camcenter, *c2 = Traj2DAll[trackID][ll + incre].camcenter;
			double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
			double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
			double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

			double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
			MotionPriorCost += costi;
			ll += incre - 1;
			allnpts++;
		}
	}
	if (!silent)
		printLOG("Motion cost: %e\n", MotionPriorCost / allnpts);

	Cost[0] = MotionPriorCost, Cost[1] = 0.0, Cost[2] = lengthCost, Cost[3] = directionCost;

	delete[]VectorTime, delete[]VectorCamID, delete[]VectorFrameID, delete[]simulatneousPoints, delete[]Traj2DAll;

	return CeresCost;
}
double MotionPrior_Optim_SpatialTemporalStructure_Rays(char *Path, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int npts, bool non_monotonicDescent, int nCams, double Tscale, double eps, double *Cost, bool StillImages, bool silent)
{
	int Allnf = 0;
	double MotionPriorCost = 0.0;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<double> VectorTime;
	vector<int> *VectorCamID = new vector<int>[npts], *VectorFrameID = new vector<int>[npts];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[npts];

	double earliestTime, currentTime, RollingShutterOffset;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	vector<int> usedCameras;
	ceres::Problem problem;
	for (int pid = 0; pid < npts; pid++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[pid].push_back(camID);
				VectorFrameID[pid].push_back(0);
				Traj2DAll[pid].push_back(PerCam_UV[camID*npts + pid][0]);
			}
		}
		else
		{
			int nopoints = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				PerCam_nf[camID] = (int)PerCam_UV[camID*npts + pid].size();

				if (PerCam_nf[camID] == 0)
					nopoints++;
			}
			if (nopoints > nCams - 2)
				continue;

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
			for (int jj = 0; jj < nCams; jj++)
				currentPID_InTrack[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentPID_InTrack[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					RollingShutterOffset = 0;
					if (PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].shutterModel != 0)
					{
						double v = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].pt2D.y,
							h = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].imHeight,
							p = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].rollingShutterPercent;
						RollingShutterOffset = v / h * p;
					}

					double ialpha = 1.0 / PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].fps;
					frameID = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].frameID;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

					if (currentTime < earliestTime)
					{
						earliestTime = currentTime;
						earliestCamID = camID;
						earliestCamFrameID = frameID;
					}
				}

				//If all cameras are done
				if (nfinishedCams == nCams)
					break;

				//Add new point to the sequence
				VectorTime.push_back(earliestTime);
				VectorCamID[pid].push_back(earliestCamID);
				VectorFrameID[pid].push_back(earliestCamFrameID);
				Traj2DAll[pid].push_back(PerCam_UV[earliestCamID*npts + pid][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		int nf = (int)Traj2DAll[pid].size();
		for (int ll = 0; ll < nf - 1; ll++)//1st order approx of v
		{
			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + 1];
			int fid1 = VectorFrameID[pid][ll], fid2 = VectorFrameID[pid][ll + 1];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + 1].pt2D.y / Traj2DAll[pid][ll + 1].imHeight*Traj2DAll[pid][ll + 1].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + 1].pt2D.y - 0.5* Traj2DAll[pid][ll + 1].imHeight)*Traj2DAll[pid][ll + 1].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps, ialpha2 = 1.0 / Traj2DAll[pid][ll + 1].fps;
			double  t1 = (currentOffset[camID1] + fid1 + shutterOffset1) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + fid2 + shutterOffset2) * ialpha2*Tscale;

			double d1 = Traj2DAll[pid][ll].d, d2 = Traj2DAll[pid][ll + 1].d;
			double *r1 = Traj2DAll[pid][ll].ray, *r2 = Traj2DAll[pid][ll + 1].ray;
			double *c1 = Traj2DAll[pid][ll].camcenter, *c2 = Traj2DAll[pid][ll + 1].camcenter;
			double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
			double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
			double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

			double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
			MotionPriorCost += costi;

			bool found = false;
			for (int kk = 0; !found && kk < (int)usedCameras.size(); kk++)
				if (usedCameras[kk] == camID1)
					found = true;
			if (!found)
				usedCameras.push_back(camID1);

			found = false;
			for (int kk = 0; !found && kk < (int)usedCameras.size(); kk++)
				if (usedCameras[kk] == camID2)
					found = true;
			if (!found)
				usedCameras.push_back(camID2);

			if (camID1 == camID2)
			{
				ceres::CostFunction* cost_function = LeastMotionPriorRaysCeres::CreateAutoDiffSame(Traj2DAll[pid][ll].ray, Traj2DAll[pid][ll + 1].ray, Traj2DAll[pid][ll].camcenter, Traj2DAll[pid][ll + 1].camcenter,
					VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], shutterOffset1, shutterOffset2, ialpha1, ialpha2, Tscale, eps);
				problem.AddResidualBlock(cost_function, NULL, &Traj2DAll[pid][ll].d, &Traj2DAll[pid][ll + 1].d, &currentOffset[camID1]);
			}
			else
			{
				ceres::CostFunction* cost_function = LeastMotionPriorRaysCeres::CreateAutoDiff(Traj2DAll[pid][ll].ray, Traj2DAll[pid][ll + 1].ray, Traj2DAll[pid][ll].camcenter, Traj2DAll[pid][ll + 1].camcenter,
					VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], shutterOffset1, shutterOffset2, ialpha1, ialpha2, Tscale, eps);
				problem.AddResidualBlock(cost_function, NULL, &Traj2DAll[pid][ll].d, &Traj2DAll[pid][ll + 1].d, &currentOffset[camID1], &currentOffset[camID2]);
			}
			Allnf++;
		}
	}
	for (int pid = 0; pid < npts; pid++)
		PerTraj_nFrames.push_back((int)Traj2DAll[pid].size());
	//ceres::LossFunction* loss_function = new ceres::HuberLoss(10.0);

	//Set bound on the time 
	double Initoffset[1000];
	Initoffset[0] = 0;
	for (int camID = 1; camID < nCams; camID++)
	{
		//In some real hassle cases, a camera do not see any points --> Ceres cannot otpim variables that are not added to problem.
		bool found = false;
		for (int kk = 0; !found && kk < (int)usedCameras.size(); kk++)
			if (usedCameras[kk] == camID)
				found = true;
		if (!found)
			continue;

		Initoffset[camID] = currentOffset[camID];
		if (abs(currentOffset[camID] - floor(currentOffset[camID])) < 1e-9) //appriximately interger
			problem.SetParameterLowerBound(&currentOffset[camID], 0, Initoffset[camID] - 0.75), problem.SetParameterUpperBound(&currentOffset[camID], 0, Initoffset[camID] + 0.75); //let it range from[a - .75, a + 0.75];
		else
			problem.SetParameterLowerBound(&currentOffset[camID], 0, floor(Initoffset[camID])), problem.SetParameterUpperBound(&currentOffset[camID], 0, ceil(Initoffset[camID])); //lock it in the frame
	}

	//Set fixed parameters
	problem.SetParameterBlockConstant(&currentOffset[0]);

	if (!silent)
		printLOG("Motion cost: %e\n", MotionPriorCost / Allnf);

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 3000;
	options.function_tolerance = 1.0e-6;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;/////ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
	options.preconditioner_type = ceres::JACOBI;
	options.minimizer_progress_to_stdout = !silent;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = non_monotonicDescent;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (!silent)
		std::cout << summary.FullReport() << "\n";
	else
		std::cout << summary.BriefReport() << "\n";
	double CeresCost = summary.final_cost;

	//Save data
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = (int)Traj2DAll[pid].size();
		if (nf == 0)
			continue;

		for (int ii = 0; ii < nf; ii++)
		{
			int camID = VectorCamID[pid][ii], frameID = VectorFrameID[pid][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*npts + pid].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*npts + pid][kk].frameID)
				{
					double d1 = Traj2DAll[pid][ii].d;
					double *r1 = Traj2DAll[pid][ii].ray;
					double *c1 = Traj2DAll[pid][ii].camcenter;
					double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);

					PerCam_UV[camID*npts + pid][kk].pt3D = Point3d(X, Y, Z);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printLOG("Serious bug in point-camera-frame association\n");
				exit(1);
			}
		}
	}


	//Compute cost after optim
	MotionPriorCost = 0.0;
	Allnf = 0;
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerTraj_nFrames[pid];
		if (nf == 0)
			continue;
		for (int ll = 0; ll < nf - 1; ll++)
		{
			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + 1];
			int fid1 = VectorFrameID[pid][ll], fid2 = VectorFrameID[pid][ll + 1];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + 1].pt2D.y / Traj2DAll[pid][ll + 1].imHeight*Traj2DAll[pid][ll + 1].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + 1].pt2D.y - 0.5* Traj2DAll[pid][ll + 1].imHeight)*Traj2DAll[pid][ll + 1].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps, ialpha2 = 1.0 / Traj2DAll[pid][ll + 1].fps;
			double  t1 = (currentOffset[camID1] + fid1 + shutterOffset1) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + fid2 + shutterOffset2) * ialpha2*Tscale;

			double d1 = Traj2DAll[pid][ll].d, d2 = Traj2DAll[pid][ll + 1].d;
			double *r1 = Traj2DAll[pid][ll].ray, *r2 = Traj2DAll[pid][ll + 1].ray;
			double *c1 = Traj2DAll[pid][ll].camcenter, *c2 = Traj2DAll[pid][ll + 1].camcenter;
			double X = (d1 * r1[0] + c1[0]), Y = (d1 * r1[1] + c1[1]), Z = (d1 * r1[2] + c1[2]);
			double X2 = (d2 * r2[0] + c2[0]), Y2 = (d2 * r2[1] + c2[1]), Z2 = (d2 * r2[2] + c2[2]);
			double difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

			double costi = (difX*difX + difY * difY + difZ * difZ) / abs(t2 - t1 + eps);
			MotionPriorCost += costi;
			Allnf++;
		}
	}

	if (!silent)
		printLOG("Motion cost: %e\n ", MotionPriorCost / Allnf);

	Cost[0] = MotionPriorCost, Cost[1] = 0.0, Cost[2] = 0.0, Cost[3] = 0.0;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]currentFrame, delete[]PerCam_nf, delete[]currentPID_InTrack;

	if (StillImages)
		delete[]StillImageTimeOrderID, delete[]StillImageTimeOrder;

	return CeresCost;
}
double MotionPrior_Optim_SpatialStructure_Geometric(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int npts, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages, bool silent)
{
	vector<double> *VectorTime = new vector<double>[npts];
	vector<int> *VectorCamID = new vector<int>[npts], *VectorFrameID = new vector<int>[npts], *simulatneousPoints = new vector<int>[npts];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[npts];

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	double CeresCost = 0.0;

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic,1)
	for (int pid = 0; pid < npts; pid++)
	{
		bool UsedCam[MaxnCams];
		int PerCam_nf[MaxnCams], currentFID_InPoint[MaxnCams];
		Point3d P3D;
		ImgPtEle ptEle;

		double earliestTime, currentTime, RollingShutterOffset;
		int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

		ceres::Problem problem;
		if (StillImages)
		{
			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime[pid].push_back(currentTime);
				VectorCamID[pid].push_back(camID);
				VectorFrameID[pid].push_back(0);
				Traj2DAll[pid].push_back(PerCam_UV[camID*npts + pid][0]);
			}
		}
		else
		{
			int nopoints = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				PerCam_nf[camID] = (int)PerCam_UV[camID*npts + pid].size();

				if (PerCam_nf[camID] == 0)
					nopoints++;
			}
			if (nopoints > nCams - 2)
				continue;

			//Assemble trajactory and time from all Cameras
			for (int jj = 0; jj < nCams; jj++)
				currentFID_InPoint[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentFID_InPoint[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					frameID = PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].frameID;
					RollingShutterOffset = 0;
					if (PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].shutterModel != 0)
						RollingShutterOffset = PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].pt2D.y / PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].imHeight*PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].rollingShutterPercent;

					double ialpha = 1.0 / PerCam_UV[camID*npts + pid][currentFID_InPoint[camID]].fps;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;
					if (currentTime < earliestTime)
					{
						earliestTime = currentTime;
						earliestCamID = camID;
						earliestCamFrameID = frameID;
					}
				}

				//If all cameras are done
				if (nfinishedCams == nCams)
					break;

				//Add new point to the sequence
				VectorTime[pid].push_back(earliestTime);
				VectorCamID[pid].push_back(earliestCamID);
				VectorFrameID[pid].push_back(earliestCamFrameID);
				Traj2DAll[pid].push_back(PerCam_UV[earliestCamID*npts + pid][currentFID_InPoint[earliestCamID]]);
				currentFID_InPoint[earliestCamID]++;
			}
		}

		int nf = (int)Traj2DAll[pid].size();
		Allpt3D[pid] = new double[3 * nf];
		for (int ll = 0; ll < nf; ll++)
			Allpt3D[pid][3 * ll] = Traj2DAll[pid][ll].pt3D.x + gaussian_noise(0.0, 1.0) / RealOverSfm,
			Allpt3D[pid][3 * ll + 1] = Traj2DAll[pid][ll].pt3D.y + gaussian_noise(0.0, 1.0) / RealOverSfm,
			Allpt3D[pid][3 * ll + 2] = Traj2DAll[pid][ll].pt3D.z + gaussian_noise(0.0, 1.0) / RealOverSfm;

		//Detect points captured simulatenously
		int groupCount = 0;
		for (int ll = 0; ll < nf; ll++)
			simulatneousPoints[pid].push_back(-1);

		for (int ll = 0; ll < nf - 1; ll++)
		{
			int naddedPoints = 0; bool found = false;
			for (int kk = ll + 1; kk < nf; kk++)
			{
				naddedPoints++;
				if (VectorTime[pid][kk] - VectorTime[pid][ll] > FLT_EPSILON)
					break;
				else
				{
					if (kk - 1 == ll)
						simulatneousPoints[pid][ll] = groupCount;

					simulatneousPoints[pid][kk] = groupCount;
					found = true;
				}
			}

			if (found)
			{
				ll += naddedPoints - 1;
				groupCount++;
			}
		}

		//1st order approx of v
		double MPCostBefore = 0, ProjCostBefore = 0;
		int oldtype = simulatneousPoints[pid][0];
		for (int ll = 0; ll < nCams; ll++)
			UsedCam[ll] = false;

		for (int ll = 0; ll < nf - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < nf)
			{
				if (simulatneousPoints[pid][ll + incre] == -1 || simulatneousPoints[pid][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[pid][ll + incre];
					break;
				}
				else
				{
					if (Traj2DAll[pid][ll + incre].std3D < 0.001)
					{
						ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, sqrt(lamda));
						problem.AddResidualBlock(cost_function, NULL, Allpt3D[pid] + 3 * ll);
					}
					else
					{
						ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, 1.0 / Traj2DAll[pid][ll + incre].std2D);
						problem.AddResidualBlock(cost_function, NULL, Allpt3D[pid] + 3 * ll);
					}
				}
				incre++;
			}
			if (ll + incre == nf)
				break;

			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + incre];
			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + incre].pt2D.y / Traj2DAll[pid][ll + incre].imHeight*Traj2DAll[pid][ll + incre].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + incre].pt2D.y - 0.5* Traj2DAll[pid][ll + incre].imHeight)*Traj2DAll[pid][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps, ialpha2 = 1.0 / Traj2DAll[pid][ll + incre].fps;
			Point3d P3d1(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]), P3d2(Allpt3D[pid][3 * (ll + incre)], Allpt3D[pid][3 * (ll + incre) + 1], Allpt3D[pid][3 * (ll + incre) + 2]);
			double costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)], &currentOffset[camID1], &currentOffset[camID2],
				shutterOffset1, shutterOffset2, VectorFrameID[pid][ll], VectorFrameID[pid][ll + incre], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MPCostBefore += costi;

			Point2d P2d_1(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][ll].P, P3d_1, P2d_1);
			ProjCostBefore += costi;

			Point2d P2d_2(Traj2DAll[pid][(ll + incre)].pt2D.x, Traj2DAll[pid][(ll + incre)].pt2D.y);
			Point3d P3d_2(Allpt3D[pid][3 * (ll + incre)], Allpt3D[pid][3 * (ll + incre) + 1], Allpt3D[pid][3 * (ll + incre) + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][(ll + incre)].P, P3d_2, P2d_2);
			ProjCostBefore += costi;

			double  t1 = (currentOffset[camID1] + VectorFrameID[pid][ll] + shutterOffset1) * ialpha1*Tscale;
			double  t2 = (currentOffset[camID2] + VectorFrameID[pid][ll + incre] + shutterOffset2) * ialpha2*Tscale;

			if (Traj2DAll[pid][ll + incre].std3D < 0.001)
			{
				ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateAutoDiff(t1, t2, eps, sqrt(1.0 - lamda)*RealOverSfm, motionPriorPower);
				//ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateNumerDiff(t1, t2, eps, sqrt(1.0 - lamda)*RealOverSfm, motionPriorPower);
				problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)]);

				ceres::CostFunction* cost_function2 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll].P, Traj2DAll[pid][ll].pt2D, sqrt(lamda));
				problem.AddResidualBlock(cost_function2, NULL, Allpt3D[pid] + 3 * ll);

				ceres::CostFunction* cost_function3 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, sqrt(lamda));
				problem.AddResidualBlock(cost_function3, NULL, Allpt3D[pid] + 3 * (ll + incre));
			}
			else
			{
				double w = sqrt(abs(t2 - t1) + eps) / sqrt(pow(Traj2DAll[pid][ll].std3D, 2) + pow(Traj2DAll[pid][ll + incre].std3D, 2));
				ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateAutoDiff(t1, t2, eps, w, motionPriorPower);
				//ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateAutoDiff(t1, t2, eps, 1.0 / Traj2DAll[pid][ll].std3D, motionPriorPower);
				//ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres::CreateNumerDiff(t1, t2, eps, 1.0 / Traj2DAll[pid][ll].std3D, motionPriorPower);
				problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)]);

				ceres::CostFunction* cost_function2 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll].P, Traj2DAll[pid][ll].pt2D, 1.0 / Traj2DAll[pid][ll].std2D);
				problem.AddResidualBlock(cost_function2, NULL, Allpt3D[pid] + 3 * ll);

				ceres::CostFunction* cost_function3 = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll + incre].P, Traj2DAll[pid][ll + incre].pt2D, 1.0 / Traj2DAll[pid][(ll + incre)].std2D);
				problem.AddResidualBlock(cost_function3, NULL, Allpt3D[pid] + 3 * (ll + incre));
			}

			ll += incre - 1;
		}

		ceres::Solver::Options options;
		options.num_threads = 2;
		options.num_linear_solver_threads = 2;
		options.max_num_iterations = 3000;
		options.function_tolerance = 1.0e-6;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;/////ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
		options.preconditioner_type = ceres::JACOBI;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = true;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		double MPCostAfter = 0.0, ProjCostAfter = 0.0;
		for (int ll = 0; ll < nf - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < nf)
			{
				if (simulatneousPoints[pid][ll + incre] == -1 || simulatneousPoints[pid][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[pid][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == nf)
				break;

			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + incre];
			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + incre].pt2D.y / Traj2DAll[pid][ll + incre].imHeight*Traj2DAll[pid][ll + incre].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + incre].pt2D.y - 0.5* Traj2DAll[pid][ll + incre].imHeight)*Traj2DAll[pid][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps, ialpha2 = 1.0 / Traj2DAll[pid][ll + incre].fps;
			Point3d P3d1(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]), P3d2(Allpt3D[pid][3 * (ll + incre)], Allpt3D[pid][3 * (ll + incre) + 1], Allpt3D[pid][3 * (ll + incre) + 2]);
			double costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)], &currentOffset[camID1], &currentOffset[camID2],
				shutterOffset1, shutterOffset2, VectorFrameID[pid][ll], VectorFrameID[pid][ll + incre], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MPCostAfter += costi;

			Point2d P2d_1(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d_1(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][ll].P, P3d_1, P2d_1);
			ProjCostAfter += costi;

			Point2d P2d_2(Traj2DAll[pid][(ll + incre)].pt2D.x, Traj2DAll[pid][(ll + incre)].pt2D.y);
			Point3d P3d_2(Allpt3D[pid][3 * (ll + incre)], Allpt3D[pid][3 * (ll + incre) + 1], Allpt3D[pid][3 * (ll + incre) + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][(ll + incre)].P, P3d_2, P2d_2);
			ProjCostAfter += costi;

			ll += incre - 1;
		}

#pragma omp critical
		{
			CeresCost += summary.final_cost;
			if (!silent)
			{
				printLOG("Point %d: (%.3e %.3e) --> (%.3e %.3e): ", pid, MPCostBefore / nf, ProjCostBefore / nf, MPCostAfter / nf, ProjCostAfter / nf);
				cout << summary.BriefReport() << "\n";
			}
		}

		//copy simultaneous triggered points
		for (int ll = 0; ll < nf - 1; ll++)
		{
			if (simulatneousPoints[pid][ll] != -1)
			{
				int nPoint = 0;
				for (int kk = ll + 1; kk < nf; kk++)
				{
					nPoint++;
					if (simulatneousPoints[pid][kk] != simulatneousPoints[pid][ll])
						break;
					else
					{
						Allpt3D[pid][3 * kk] = Allpt3D[pid][3 * ll];
						Allpt3D[pid][3 * kk + 1] = Allpt3D[pid][3 * ll + 1];
						Allpt3D[pid][3 * kk + 2] = Allpt3D[pid][3 * ll + 2];
					}
				}
				ll += nPoint - 1;
			}
		}

		for (int ii = 0; ii < Traj2DAll[pid].size(); ii++)
		{
			int camID = VectorCamID[pid][ii], frameID = VectorFrameID[pid][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*npts + pid].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*npts + pid][kk].frameID)
				{
					PerCam_UV[camID*npts + pid][kk].pt3D = Point3d(Allpt3D[pid][3 * ii], Allpt3D[pid][3 * ii + 1], Allpt3D[pid][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
				printLOG("Serious bug in point-camera-frame association\n");
		}
	}
	PerTraj_nFrames.clear();
	for (int pid = 0; pid < npts; pid++)
		PerTraj_nFrames.push_back((int)Traj2DAll[pid].size());

	//Compute cost after optim
	double MPCost = 0.0, ProjCost = 0.0, lengthCost = 0.0, directionCost = 0;
	int allnpts = 0;
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerTraj_nFrames[pid];
		if (nf == 0)
			continue;

		int oldtype = simulatneousPoints[pid][0];
		for (int ll = 0; ll < nf - 1; ll++)
		{
			int incre = 1;
			while (ll + incre < nf)
			{
				if (simulatneousPoints[pid][ll + incre] == -1 || simulatneousPoints[pid][ll + incre] != oldtype)
				{
					oldtype = simulatneousPoints[pid][ll + incre];
					break;
				}
				incre++;
			}
			if (ll + incre == nf)
				break;

			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + incre];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + incre].pt2D.y / Traj2DAll[pid][ll + incre].imHeight*Traj2DAll[pid][ll + incre].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + incre].pt2D.y - 0.5* Traj2DAll[pid][ll + incre].imHeight)*Traj2DAll[pid][ll + incre].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps;
			double ialpha2 = 1.0 / Traj2DAll[pid][ll + incre].fps;
			double costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + incre)], &currentOffset[camID1], &currentOffset[camID2], shutterOffset1, shutterOffset2,
				VectorFrameID[pid][ll], VectorFrameID[pid][ll + incre], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MPCost += costi;
			allnpts++;
			ll += incre - 1;
		}
	}
	if (!silent)
		printLOG("Motion cost: %e ", MPCost / allnpts);

	allnpts = 0;
	vector<double> Errx, Erry;
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerTraj_nFrames[pid];
		if (nf == 0)
			continue;
		for (int ll = 0; ll < nf; ll++)
		{
			int viewID = Traj2DAll[pid][ll].viewID, frameID = Traj2DAll[pid][ll].frameID;
			Point2d P2d(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);

			double numX = Traj2DAll[pid][ll].P[0] * P3d.x + Traj2DAll[pid][ll].P[1] * P3d.y + Traj2DAll[pid][ll].P[2] * P3d.z + Traj2DAll[pid][ll].P[3];
			double numY = Traj2DAll[pid][ll].P[4] * P3d.x + Traj2DAll[pid][ll].P[5] * P3d.y + Traj2DAll[pid][ll].P[6] * P3d.z + Traj2DAll[pid][ll].P[7];
			double denum = Traj2DAll[pid][ll].P[8] * P3d.x + Traj2DAll[pid][ll].P[9] * P3d.y + Traj2DAll[pid][ll].P[10] * P3d.z + Traj2DAll[pid][ll].P[11];
			double errx = numX / denum - P2d.x, erry = numY / denum - P2d.y;
			Errx.push_back(errx), Erry.push_back(erry);

			ProjCost += sqrt(errx*errx + erry * erry);
		}
		allnpts += nf;
	}

	double mx = MeanArray(Errx), my = MeanArray(Erry);
	double vx = sqrt(VarianceArray(Errx, mx)), vy = sqrt(VarianceArray(Erry, my));
	if (!silent)
		printLOG("Projection cost: (%e, %e, %e) (mean L1, mean L2, std)\n", ProjCost / allnpts, sqrt(mx*mx + my * my), sqrt(vx*vx + vy * vy));


	Cost[0] = MPCost, Cost[1] = ProjCost, Cost[2] = 0.0, Cost[3] = 0.0;

	delete[]VectorTime, delete[]VectorCamID, delete[]VectorFrameID, delete[]simulatneousPoints, delete[]Traj2DAll;

	return CeresCost;
}
double MotionPrior_Optim_SpatialTemporal_Geometric(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerTraj_nFrames, double *currentOffset, int npts, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages, bool silent)
{
	//when ML weighting is used, lamda corresponds to the mass of the point.
	int allnpts = 0;
	double MotionPriorCost = 0.0, ProjCost = 0.0, costi;

	int *currentFrame = new int[nCams], *PerCam_nf = new int[nCams], *currentPID_InTrack = new int[nCams];
	Point3d P3D;
	ImgPtEle ptEle;

	vector<double> VectorTime;
	vector<int> *VectorCamID = new vector<int>[npts], *VectorFrameID = new vector<int>[npts];
	vector<ImgPtEle> *Traj2DAll = new vector<ImgPtEle>[npts];

	double earliestTime, currentTime, RollingShutterOffset;
	int earliestCamFrameID, frameID, nfinishedCams, earliestCamID;

	int *StillImageTimeOrderID = 0;
	double *StillImageTimeOrder = 0;
	if (StillImages)
	{
		StillImageTimeOrderID = new int[nCams];
		StillImageTimeOrder = new double[nCams];
		for (int camID = 0; camID < nCams; camID++)
		{
			StillImageTimeOrder[camID] = currentOffset[camID];
			StillImageTimeOrderID[camID] = camID;
		}
		Quick_Sort_Double(StillImageTimeOrder, StillImageTimeOrderID, 0, nCams - 1);
	}

	vector<int> usedCameras;
	ceres::Problem problem;
	for (int pid = 0; pid < npts; pid++)
	{
		if (StillImages)
		{
			VectorTime.clear();

			for (int ii = 0; ii < nCams; ii++)
			{
				int camID = StillImageTimeOrderID[ii];
				double currentTime = StillImageTimeOrder[ii];
				VectorTime.push_back(currentTime);
				VectorCamID[pid].push_back(camID);
				VectorFrameID[pid].push_back(0);
				Traj2DAll[pid].push_back(PerCam_UV[camID*npts + pid][0]);
			}
		}
		else
		{
			int nopoints = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				PerCam_nf[camID] = (int)PerCam_UV[camID*npts + pid].size();

				if (PerCam_nf[camID] == 0)
					nopoints++;
			}
			if (nopoints > nCams - 2)
				continue;

			//Assemble trajactory and time from all Cameras
			VectorTime.clear();
			for (int jj = 0; jj < nCams; jj++)
				currentPID_InTrack[jj] = 0;

			while (true)
			{
				//Determine the next camera
				nfinishedCams = 0, earliestCamID, earliestTime = 9e9;
				for (int camID = 0; camID < nCams; camID++)
				{
					if (currentPID_InTrack[camID] == PerCam_nf[camID])
					{
						nfinishedCams++;
						continue;
					}

					//Time:
					RollingShutterOffset = 0;
					if (PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].shutterModel != 0)
					{
						double v = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].pt2D.y,
							h = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].imHeight,
							p = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].rollingShutterPercent;
						RollingShutterOffset = v / h * p;
					}

					double ialpha = 1.0 / PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].fps;
					frameID = PerCam_UV[camID*npts + pid][currentPID_InTrack[camID]].frameID;
					currentTime = (currentOffset[camID] + frameID + RollingShutterOffset) * ialpha*Tscale;

					if (currentTime < earliestTime)
					{
						earliestTime = currentTime;
						earliestCamID = camID;
						earliestCamFrameID = frameID;
					}
				}

				//If all cameras are done
				if (nfinishedCams == nCams)
					break;

				//Add new point to the sequence
				VectorTime.push_back(earliestTime);
				VectorCamID[pid].push_back(earliestCamID);
				VectorFrameID[pid].push_back(earliestCamFrameID);
				Traj2DAll[pid].push_back(PerCam_UV[earliestCamID*npts + pid][currentPID_InTrack[earliestCamID]]);

				currentPID_InTrack[earliestCamID]++;
			}
		}

		int nf = (int)Traj2DAll[pid].size();
		Allpt3D[pid] = new double[3 * nf];
		for (int ll = 0; ll < nf; ll++)
			Allpt3D[pid][3 * ll] = Traj2DAll[pid][ll].pt3D.x, Allpt3D[pid][3 * ll + 1] = Traj2DAll[pid][ll].pt3D.y, Allpt3D[pid][3 * ll + 2] = Traj2DAll[pid][ll].pt3D.z;

		for (int ll = 0; ll < nf - 1; ll++)//1st order approx of v
		{
			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + 1];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + 1].pt2D.y / Traj2DAll[pid][ll + 1].imHeight*Traj2DAll[pid][ll + 1].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + 1].pt2D.y - 0.5* Traj2DAll[pid][ll + 1].imHeight)*Traj2DAll[pid][ll + 1].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps, ialpha2 = 1.0 / Traj2DAll[pid][ll + 1].fps;
			costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], shutterOffset1, shutterOffset2,
				VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MotionPriorCost += costi;

			double t1 = (currentOffset[camID1] + shutterOffset1 + VectorFrameID[pid][ll]) * ialpha1*Tscale;
			double t2 = (currentOffset[camID2] + shutterOffset2 + VectorFrameID[pid][ll + 1]) * ialpha2*Tscale;
			double dt = abs(t2 - t1);
			double w = sqrt(dt) / sqrt(pow(Traj2DAll[pid][ll].std3D, 2) + pow(Traj2DAll[pid][ll + 1].std3D, 2)); //1.0 / Traj2DAll[pid][ll].std3D

			bool found = false;
			for (int kk = 0; !found && kk < (int)usedCameras.size(); kk++)
				if (usedCameras[kk] == camID1)
					found = true;
			if (!found)
				usedCameras.push_back(camID1);

			found = false;
			for (int kk = 0; !found && kk < (int)usedCameras.size(); kk++)
				if (usedCameras[kk] == camID2)
					found = true;
			if (!found)
				usedCameras.push_back(camID2);

			if (camID1 == camID2)
			{
				if (Traj2DAll[pid][ll].std3D < 0.0) //not using scale for determine uncertainty
				{
					ceres::CostFunction* cost_function = LeastMotionPriorCostCeres::CreateAutoDiffSame(VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], shutterOffset1, shutterOffset2, ialpha1, ialpha2, Tscale, eps, sqrt(1.0 - lamda)*RealOverSfm, motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + 1)], &currentOffset[camID1]);
				}
				else
				{
					ceres::CostFunction* cost_function = LeastMotionPriorCostCeres::CreateAutoDiffSame(VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], shutterOffset1, shutterOffset2, ialpha1, ialpha2, Tscale, eps, w*sqrt(lamda), motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + 1)], &currentOffset[camID1]);
				}
			}
			else
			{
				if (Traj2DAll[pid][ll].std3D < 0.0)
				{
					ceres::CostFunction* cost_function = LeastMotionPriorCostCeres::CreateAutoDiff(VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], shutterOffset1, shutterOffset2, ialpha1, ialpha2, Tscale, eps, sqrt(1.0 - lamda)*RealOverSfm, motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + 1)], &currentOffset[camID1], &currentOffset[camID2]);
				}
				else
				{
					ceres::CostFunction* cost_function = LeastMotionPriorCostCeres::CreateAutoDiff(VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], shutterOffset1, shutterOffset2, ialpha1, ialpha2, Tscale, eps, w*sqrt(lamda), motionPriorPower);
					problem.AddResidualBlock(cost_function, NULL, &Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * (ll + 1)], &currentOffset[camID1], &currentOffset[camID2]);
				}
			}
		}

		for (int ll = 0; ll < nf; ll++)
		{
			Point2d P2d(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);
			costi = PinholeReprojectionErrorSimpleDebug(Traj2DAll[pid][ll].P, P3d, P2d);
			ProjCost += costi;

			if (Traj2DAll[pid][ll].std3D < 0.0)
			{
				ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll].P, Traj2DAll[pid][ll].pt2D, sqrt(lamda));
				problem.AddResidualBlock(cost_function, NULL, Allpt3D[pid] + 3 * ll);
			}
			else
			{
				ceres::CostFunction* cost_function = IdealGeoProjectionCeres::Create(Traj2DAll[pid][ll].P, Traj2DAll[pid][ll].pt2D, 1.0 / Traj2DAll[pid][ll].std2D);
				problem.AddResidualBlock(cost_function, NULL, Allpt3D[pid] + 3 * ll);
			}
			allnpts++;
		}
	}
	for (int pid = 0; pid < npts; pid++)
		PerTraj_nFrames.push_back((int)Traj2DAll[pid].size());
	//ceres::LossFunction* loss_function = new ceres::HuberLoss(10.0);

	//Set bound on the time 
	double Initoffset[1000];
	Initoffset[0] = 0;
	for (int camID = 1; camID < nCams; camID++)
	{
		//In some real hassle cases, a camera do not see any points --> Ceres cannot otpim variables that are not added to problem.
		bool found = false;
		for (int kk = 0; !found && kk < (int)usedCameras.size(); kk++)
			if (usedCameras[kk] == camID)
				found = true;
		if (!found)
			continue;

		Initoffset[camID] = currentOffset[camID];
		if (abs(currentOffset[camID] - floor(currentOffset[camID])) < 1e-9) //appriximately interger
			problem.SetParameterLowerBound(&currentOffset[camID], 0, Initoffset[camID] - 0.75), problem.SetParameterUpperBound(&currentOffset[camID], 0, Initoffset[camID] + 0.75); //let it range from[a - .75, a + 0.75];
		else
			problem.SetParameterLowerBound(&currentOffset[camID], 0, floor(Initoffset[camID])), problem.SetParameterUpperBound(&currentOffset[camID], 0, ceil(Initoffset[camID])); //lock it in the frame
	}

	//Set fixed parameters
	problem.SetParameterBlockConstant(&currentOffset[0]);

	if (!silent)
		printLOG("Motion cost: %e Projection cost: %e\n", MotionPriorCost / allnpts, ProjCost / allnpts);

	ceres::Solver::Options options;
	options.num_threads = omp_get_max_threads();
	options.num_linear_solver_threads = omp_get_max_threads();
	options.max_num_iterations = 3000;
	options.function_tolerance = 1.0e-7;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;/////ceres::SPARSE_SCHUR;//SPARSE_NORMAL_CHOLESKY
	options.preconditioner_type = ceres::JACOBI;
	options.minimizer_progress_to_stdout = false;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.use_nonmonotonic_steps = non_monotonicDescent;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (!silent)
		std::cout << summary.FullReport() << "\n";
	else
		std::cout << summary.BriefReport() << "\n";
	double CeresCost = summary.final_cost;

	//Save data
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = (int)Traj2DAll[pid].size();
		if (nf == 0)
			continue;

		for (int ii = 0; ii < nf; ii++)
		{
			int camID = VectorCamID[pid][ii], frameID = VectorFrameID[pid][ii];

			bool found = false;
			for (int kk = 0; kk < PerCam_UV[camID*npts + pid].size(); kk++)
			{
				if (frameID == PerCam_UV[camID*npts + pid][kk].frameID)
				{
					PerCam_UV[camID*npts + pid][kk].pt3D = Point3d(Allpt3D[pid][3 * ii], Allpt3D[pid][3 * ii + 1], Allpt3D[pid][3 * ii + 2]);
					found = true;
					break;
				}
			}
			if (!found)
			{
				printLOG("Serious bug in point-camera-frame association\n");
				exit(0);
			}
		}
	}


	//Compute cost after optim
	allnpts = 0, MotionPriorCost = 0.0, ProjCost = 0.0;
	vector<double> Errx, Erry;
	for (int pid = 0; pid < npts; pid++)
	{
		int nf = PerTraj_nFrames[pid];
		if (nf == 0)
			continue;

		double costi;
		for (int ll = 0; ll < nf - 1; ll++)
		{
			int camID1 = VectorCamID[pid][ll], camID2 = VectorCamID[pid][ll + 1];

			double shutterOffset1 = 0, shutterOffset2 = 0;
			//	if (Traj2DAll[pid][ll].shutterModel != 0)
			//	shutterOffset1 = Traj2DAll[pid][ll].pt2D.y / Traj2DAll[pid][ll].imHeight*Traj2DAll[pid][ll].rollingShutterPercent,
			//	shutterOffset2 = Traj2DAll[pid][ll + 1].pt2D.y / Traj2DAll[pid][ll + 1].imHeight*Traj2DAll[pid][ll + 1].rollingShutterPercent;
			if (Traj2DAll[pid][ll].shutterModel != 0) //adjust to the the virutal 3D rs estiatmatioion idea
				shutterOffset1 = 0.5 + (Traj2DAll[pid][ll].pt2D.y - 0.5*Traj2DAll[pid][ll].imHeight) *Traj2DAll[pid][ll].rollingShutterPercent,
				shutterOffset2 = 0.5 + (Traj2DAll[pid][ll + 1].pt2D.y - 0.5* Traj2DAll[pid][ll + 1].imHeight)*Traj2DAll[pid][ll + 1].rollingShutterPercent;

			double ialpha1 = 1.0 / Traj2DAll[pid][ll].fps, ialpha2 = 1.0 / Traj2DAll[pid][ll + 1].fps;
			costi = LeastActionError(&Allpt3D[pid][3 * ll], &Allpt3D[pid][3 * ll + 3], &currentOffset[camID1], &currentOffset[camID2], shutterOffset1, shutterOffset2, VectorFrameID[pid][ll], VectorFrameID[pid][ll + 1], ialpha1, ialpha2, Tscale, eps, motionPriorPower);
			MotionPriorCost += costi;
		}

		for (int ll = 0; ll < nf; ll++)
		{
			Point2d P2d(Traj2DAll[pid][ll].pt2D.x, Traj2DAll[pid][ll].pt2D.y);
			Point3d P3d(Allpt3D[pid][3 * ll], Allpt3D[pid][3 * ll + 1], Allpt3D[pid][3 * ll + 2]);

			double numX = Traj2DAll[pid][ll].P[0] * P3d.x + Traj2DAll[pid][ll].P[1] * P3d.y + Traj2DAll[pid][ll].P[2] * P3d.z + Traj2DAll[pid][ll].P[3];
			double numY = Traj2DAll[pid][ll].P[4] * P3d.x + Traj2DAll[pid][ll].P[5] * P3d.y + Traj2DAll[pid][ll].P[6] * P3d.z + Traj2DAll[pid][ll].P[7];
			double denum = Traj2DAll[pid][ll].P[8] * P3d.x + Traj2DAll[pid][ll].P[9] * P3d.y + Traj2DAll[pid][ll].P[10] * P3d.z + Traj2DAll[pid][ll].P[11];
			double errx = numX / denum - P2d.x, erry = numY / denum - P2d.y;
			Errx.push_back(errx), Erry.push_back(erry);

			ProjCost += sqrt(errx*errx + erry * erry);
			allnpts++;
		}
	}

	double mx = MeanArray(Errx), my = MeanArray(Erry);
	double vx = sqrt(VarianceArray(Errx, mx)), vy = sqrt(VarianceArray(Erry, my));
	//if (!silent)
	printLOG("Motion cost: %e Projection cost: (%e, %e, %e) (mean L1, mean L2, std)\n", MotionPriorCost / allnpts, ProjCost / allnpts, sqrt(mx*mx + my * my), sqrt(vx*vx + vy * vy));

	Cost[0] = MotionPriorCost, Cost[1] = ProjCost, Cost[2] = 0.0, Cost[3] = 0.0;

	delete[]VectorCamID, delete[]VectorFrameID, delete[]Traj2DAll;
	delete[]currentFrame, delete[]PerCam_nf, delete[]currentPID_InTrack;

	if (StillImages)
		delete[]StillImageTimeOrderID, delete[]StillImageTimeOrder;

	return CeresCost;
}

double MotionPriorSyncBruteForce2DStereo(char *Path, vector<int> &SelectedCams, int startF, int stopF, int ntracks, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, double RealOverSfm, int motionPriorPower, int &totalPoints, bool silient)
{
	//Offset is in timestamp format
	const double Tscale = 1000000.0, eps = 1.0e-6;
	char Fname[512]; FILE *fp = 0;
	const int nCams = 2;

	//Read calib info
	VideoData VideoInfo[2];
	if (ReadVideoDataI(Path, VideoInfo[0], SelectedCams[0], startF, stopF) == 1)
		return 9e99;
	if (ReadVideoDataI(Path, VideoInfo[1], SelectedCams[1], startF, stopF) == 1)
		return 9e99;

	int id, frameID, nf;
	int nframes = max(MaxnFrames, stopF);

	double u, v, s, dummy;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks], XYZBK;

	//Get 2D info
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve((stopF - startF + 1) / 50);

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%d ", &ntracks);
		while (fscanf(fp, "%d %d ", &id, &nf) != EOF)
		{
			PerCam_UV[camID*ntracks + id].reserve(nf);
			//if (id != trackID)
			//	printLOG("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < nf; pid++)
			{
				fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf ", &frameID, &u, &v, &s, &dummy, &dummy, &dummy, &dummy, &dummy);
				if (frameID < startF || frameID > stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue;

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.scale = s, ptEle.viewID = camID, ptEle.frameID = frameID;
					ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}


	//Generate Calib Info
	double AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < ntracks; pid++)
	{
		//Get ray direction, Q, U, P
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + pid].size(); frameID++)
			{
				ImgPtEle *ptEle = &PerCam_UV[camID*ntracks + pid][frameID];
				int RealFrameID = ptEle[0].frameID;
				CameraData *camI = VideoInfo[camID].VideoInfo;

				ptEle[0].fps = camI[RealFrameID].fps;
				ptEle[0].rollingShutterPercent = camI[RealFrameID].rollingShutterPercent;
				ptEle[0].shutterModel = camI[RealFrameID].ShutterModel;

				for (int kk = 0; kk < 9; kk++)
					ptEle[0].K[kk] = camI[RealFrameID].K[kk];

				//rayDirection = iR*(lamda*iK*[u,v,1] - T) - C
				double iR[9], tt[3], ttt[3];
				double pcn[3] = { camI[RealFrameID].invK[0] * ptEle[0].pt2D.x + camI[RealFrameID].invK[1] * ptEle[0].pt2D.y + camI[RealFrameID].invK[2],
					camI[RealFrameID].invK[4] * ptEle[0].pt2D.y + camI[RealFrameID].invK[5], 1.0 };

				if (camI[RealFrameID].ShutterModel == 0)
				{
					for (int kk = 0; kk < 12; kk++)
						ptEle[0].P[kk] = camI[RealFrameID].P[kk];
					for (int kk = 0; kk < 9; kk++)
						ptEle[0].R[kk] = camI[RealFrameID].R[kk];
					for (int kk = 0; kk < 3; kk++)
						ptEle[0].camcenter[kk] = camI[RealFrameID].camCenter[kk];

					mat_transpose(camI[RealFrameID].R, iR, 3, 3);
					mat_subtract(pcn, camI[RealFrameID].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].camcenter[ll] = camI[RealFrameID].camCenter[ll], ptEle[0].ray[ll] = camI[RealFrameID].camCenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else if (camI[RealFrameID].ShutterModel == 1)
				{
					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);

					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);
					AssembleRT_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].R, ptEle[0].T);
					GetCfromT(ptEle[0].R, ptEle[0].T, ptEle[0].camcenter);

					mat_transpose(ptEle[0].R, iR, 3, 3);
					mat_subtract(pcn, ptEle[0].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].ray[ll] = ptEle[0].camcenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else
					printLOG("Not supported model for motion prior sync\n");

				//Q, U
				AA[0] = ptEle[0].P[0], AA[1] = ptEle[0].P[1], AA[2] = ptEle[0].P[2], bb[0] = ptEle[0].P[3];
				AA[3] = ptEle[0].P[4], AA[4] = ptEle[0].P[5], AA[5] = ptEle[0].P[6], bb[1] = ptEle[0].P[7];
				ccT[0] = ptEle[0].P[8], ccT[1] = ptEle[0].P[9], ccT[2] = ptEle[0].P[10], dd[0] = ptEle[0].P[11];

				ptEle[0].Q[0] = AA[0] - ptEle[0].pt2D.x*ccT[0], ptEle[0].Q[1] = AA[1] - ptEle[0].pt2D.x*ccT[1], ptEle[0].Q[2] = AA[2] - ptEle[0].pt2D.x*ccT[2];
				ptEle[0].Q[3] = AA[3] - ptEle[0].pt2D.y*ccT[0], ptEle[0].Q[4] = AA[4] - ptEle[0].pt2D.y*ccT[1], ptEle[0].Q[5] = AA[5] - ptEle[0].pt2D.y*ccT[2];
				ptEle[0].u[0] = dd[0] * ptEle[0].pt2D.x - bb[0], ptEle[0].u[1] = dd[0] * ptEle[0].pt2D.y - bb[1];

				ptEle[0].pt3D = Point3d(count, count, count); count++;

				//depth
				double lamda[3] = { (ptEle[0].pt3D.x - ptEle[0].camcenter[0]) / ptEle[0].ray[0],
					(ptEle[0].pt3D.y - ptEle[0].camcenter[1]) / ptEle[0].ray[1],
					(ptEle[0].pt3D.z - ptEle[0].camcenter[2]) / ptEle[0].ray[2] };

				//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
				int index[3] = { 0, 1, 2 };
				double rayDirect[3] = { abs(ptEle[0].ray[0]), abs(ptEle[0].ray[1]), abs(ptEle[0].ray[2]) };
				Quick_Sort_Double(rayDirect, index, 0, 2);

				ptEle[0].d = lamda[index[2]];
			}
		}
	}

	//Initialize data for optim
	totalPoints = 0;
	vector<int> PerTraj_nFrames;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int NTimeInstances, maxNTimeInstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		NTimeInstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			NTimeInstances += (int)PerCam_UV[camID*ntracks + trackID].size();
		totalPoints += NTimeInstances;

		if (maxNTimeInstances < NTimeInstances)
			maxNTimeInstances = NTimeInstances;

		PerTrackFrameID[trackID] = new int[NTimeInstances];
		All3D[trackID] = new double[3 * NTimeInstances];
	}


	//Start sliding
	double currentOffset[2], APLDCost[4], ceresCost;
	vector<double> VTimeStamp; VTimeStamp.reserve(maxNTimeInstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxNTimeInstances);
	int *OffsetID = new int[UpBound - LowBound + 1];
	double*AllCost = new double[UpBound - LowBound + 1],
		*AllMPCost = new double[UpBound - LowBound + 1],
		*AllPCost = new double[UpBound - LowBound + 1],
		*AllLCost = new double[UpBound - LowBound + 1],
		*AllDCost = new double[UpBound - LowBound + 1];

	int count = 0;
	for (int off = LowBound; off <= UpBound; off++)
	{
		OffsetID[off - LowBound] = off;
		currentOffset[0] = OffsetInfo[0], currentOffset[1] = off * frameSize + OffsetInfo[1];

		PerTraj_nFrames.clear();
		MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, nCams, Tscale);
		//ceresCost = MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
		ceresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, true);

		//MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
		//ceresCost = MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

		if (silient)
			printLOG("@off %d (id: %d): C: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\n", off, count, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
		count++;

		//Clean estimated 3D
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			int dummy = 0;
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
				{
					PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(dummy, dummy, dummy);
					dummy++;
				}
			}
		}

		AllCost[off - LowBound] = ceresCost, AllMPCost[off - LowBound] = APLDCost[0], AllPCost[off - LowBound] = APLDCost[1], AllLCost[off - LowBound] = APLDCost[2], AllDCost[off - LowBound] = APLDCost[3];
	}

	//Compute minium cost
	Quick_Sort_Double(AllCost, OffsetID, 0, count - 1);

	printLOG("(%d %d): Min MPCost: %.5e, Offset: %.4f (id: %d)\n", SelectedCams[0], SelectedCams[1], AllCost[0], frameSize*OffsetID[0] + OffsetInfo[1], OffsetID[0] - LowBound);
	OffsetInfo[1] += frameSize * OffsetID[0];

	double finalCost = AllCost[0];

	delete[]PerCam_UV, delete[]PerCam_XYZ, delete[]XYZ;
	delete[]OffsetID, delete[]AllCost, delete[]AllMPCost, delete[]AllPCost, delete[]AllLCost, delete[]AllDCost;
	for (int ii = 0; ii < (int)All3D.size(); ii++)
		delete[]All3D[ii], delete[]PerTrackFrameID[ii];

	return finalCost;
}
int MotionPriorSyncBruteForce2DTriplet(char *Path, vector<int> &SelectedCams, int startF, int stopF, int ntracks, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, double RealOverSfm, int motionPriorPower)
{
	//Offset is in timestamp format
	const double Tscale = 1000000.0, eps = 1.0e-6;

	char Fname[512]; FILE *fp = 0;
	const int nCams = 3;

	sprintf(Fname, "%s/TripletOffsetBruteforce.txt", Path);
	if (IsFileExist(Fname) == 1)
	{
		int camList[3];
		fp = fopen(Fname, "r");
		fscanf(fp, "%d %d %d %lf %lf %lf", &camList[0], &camList[1], &camList[2], &OffsetInfo[0], &OffsetInfo[1], &OffsetInfo[2]);
		fclose(fp);
		for (int ii = 0; ii < 3; ii++)
		{
			if (camList[ii] != SelectedCams[ii])
			{
				printLOG("Precomputed timestamps are not consistent!");
				abort();
			}
		}
		return 0;
	}

	//Read calib info
	VideoData VideoInfo[3];
	if (ReadVideoDataI(Path, VideoInfo[0], SelectedCams[0], startF, stopF) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[1], SelectedCams[1], startF, stopF) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[2], SelectedCams[2], startF, stopF) == 1)
		return 1;

	int id, frameID, nf;
	int nframes = max(MaxnFrames, stopF);

	double u, v, s, dummy;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks];

	//Get 2D info
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve((stopF - startF + 1) / 50);

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot open %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &ntracks);
		while (fscanf(fp, "%d %d ", &id, &nf) != EOF)
		{
			//if (id != trackID)
			//	printLOG("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < nf; pid++)
			{
				fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf ", &frameID, &u, &v, &s, &dummy, &dummy, &dummy, &dummy, &dummy);
				if (frameID < startF || frameID > stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue;

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.scale = s, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}


	//Generate Calib Info
	double  AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < ntracks; pid++)
	{
		//Get ray direction, Q, U, P
		int pcount = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + pid].size(); frameID++)
			{
				ImgPtEle *ptEle = &PerCam_UV[camID*ntracks + pid][frameID];
				int RealFrameID = ptEle[0].frameID;
				CameraData *camI = VideoInfo[camID].VideoInfo;

				ptEle[0].fps = camI[RealFrameID].fps;
				ptEle[0].rollingShutterPercent = camI[RealFrameID].rollingShutterPercent;
				ptEle[0].shutterModel = camI[RealFrameID].ShutterModel;

				for (int kk = 0; kk < 9; kk++)
					ptEle[0].K[kk] = camI[RealFrameID].K[kk];

				//rayDirection = iR*(lamda*iK*[u,v,1] - T) - C
				double iR[9], tt[3], ttt[3];
				double pcn[3] = { camI[RealFrameID].invK[0] * ptEle[0].pt2D.x + camI[RealFrameID].invK[1] * ptEle[0].pt2D.y + camI[RealFrameID].invK[2],
					camI[RealFrameID].invK[4] * ptEle[0].pt2D.y + camI[RealFrameID].invK[5], 1.0 };

				if (camI[RealFrameID].ShutterModel == 0)
				{
					for (int kk = 0; kk < 12; kk++)
						ptEle[0].P[kk] = camI[RealFrameID].P[kk];
					for (int kk = 0; kk < 9; kk++)
						ptEle[0].R[kk] = camI[RealFrameID].R[kk];
					for (int kk = 0; kk < 3; kk++)
						ptEle[0].camcenter[kk] = camI[RealFrameID].camCenter[kk];

					mat_transpose(camI[RealFrameID].R, iR, 3, 3);
					mat_subtract(pcn, camI[RealFrameID].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].camcenter[ll] = camI[RealFrameID].camCenter[ll], ptEle[0].ray[ll] = camI[RealFrameID].camCenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else if (camI[RealFrameID].ShutterModel == 1)
				{
					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);

					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);
					AssembleRT_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].R, ptEle[0].T);
					GetCfromT(ptEle[0].R, ptEle[0].T, ptEle[0].camcenter);

					mat_transpose(ptEle[0].R, iR, 3, 3);
					mat_subtract(pcn, ptEle[0].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].ray[ll] = ptEle[0].camcenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else
					printLOG("Not supported model for motion prior sync\n");

				//Q, U
				AA[0] = ptEle[0].P[0], AA[1] = ptEle[0].P[1], AA[2] = ptEle[0].P[2], bb[0] = ptEle[0].P[3];
				AA[3] = ptEle[0].P[4], AA[4] = ptEle[0].P[5], AA[5] = ptEle[0].P[6], bb[1] = ptEle[0].P[7];
				ccT[0] = ptEle[0].P[8], ccT[1] = ptEle[0].P[9], ccT[2] = ptEle[0].P[10], dd[0] = ptEle[0].P[11];

				ptEle[0].Q[0] = AA[0] - ptEle[0].pt2D.x*ccT[0], ptEle[0].Q[1] = AA[1] - ptEle[0].pt2D.x*ccT[1], ptEle[0].Q[2] = AA[2] - ptEle[0].pt2D.x*ccT[2];
				ptEle[0].Q[3] = AA[3] - ptEle[0].pt2D.y*ccT[0], ptEle[0].Q[4] = AA[4] - ptEle[0].pt2D.y*ccT[1], ptEle[0].Q[5] = AA[5] - ptEle[0].pt2D.y*ccT[2];
				ptEle[0].u[0] = dd[0] * ptEle[0].pt2D.x - bb[0], ptEle[0].u[1] = dd[0] * ptEle[0].pt2D.y - bb[1];

				ptEle[0].pt3D = Point3d(pcount, pcount, pcount); pcount++;

				//depth
				double lamda[3] = { (ptEle[0].pt3D.x - ptEle[0].camcenter[0]) / ptEle[0].ray[0],
					(ptEle[0].pt3D.y - ptEle[0].camcenter[1]) / ptEle[0].ray[1],
					(ptEle[0].pt3D.z - ptEle[0].camcenter[2]) / ptEle[0].ray[2] };

				//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
				int index[3] = { 0, 1, 2 };
				double rayDirect[3] = { abs(ptEle[0].ray[0]), abs(ptEle[0].ray[1]), abs(ptEle[0].ray[2]) };
				Quick_Sort_Double(rayDirect, index, 0, 2);

				ptEle[0].d = lamda[index[2]];
			}
		}
	}

	//Initialize data for optim
	vector<int> PerTraj_nFrames;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += (int)PerCam_UV[camID*ntracks + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	//Start sliding
	double currentOffset[3], APLDCost[4], ceresCost;
	int *OffsetID = new int[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	Point2d *OffsetValue = new Point2d[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllMPCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllPCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllLCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	double *AllDCost = new double[(UpBound - LowBound + 1)*(UpBound - LowBound + 1)];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	int count = 0, bestSum = 0;
#ifdef EC2
	printLOG("EC2 triplet brute-force:\n");
	double nstartTime = omp_get_wtime();
	vector<int> njobs;
	for (int off2 = LowBound; off2 <= UpBound; off2++)
	{
		for (int off1 = LowBound; off1 <= UpBound; off1++)
		{
			OffsetID[count] = count;
			OffsetValue[count] = Point2d(off1, off2);
			currentOffset[0] = OffsetInfo[0], currentOffset[1] = off1 * frameSize + OffsetInfo[1], currentOffset[2] = off2 * frameSize + OffsetInfo[2];

			PerTraj_nFrames.clear();

			sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2);
			if (IsFileExist(Fname) == 0)
			{
#ifdef _WINDOWS
				sprintf(Fname, "EnRecon.exe 7 1 2 %d %d %d %.3f %.3f %.3f %d %d", SelectedCams[0], SelectedCams[1], SelectedCams[2], OffsetInfo[0], OffsetInfo[1], OffsetInfo[2], off1, off2);
#else
				sprintf(Fname, "qsub -b y -cwd -pe orte %d ./EnRecon 7 1 2 %d %d %d %.3f %.3f %.3f %d %d", min(omp_get_max_threads(), 4), SelectedCams[0], SelectedCams[1], SelectedCams[2], OffsetInfo[0], OffsetInfo[1], OffsetInfo[2], off1, off2);
#endif
				printLOG("%s\n", Fname);
				system(Fname);
			}
			njobs.push_back(0);
		}
	}

	//wait until all files are ready
	printLOG("Start waiting ...\n");
	bestSum = 0;
	while (true)
	{
		mySleep(1e3);
		count = 0;
		for (int off2 = LowBound; off2 <= UpBound; off2++)
		{
			for (int off1 = LowBound; off1 <= UpBound; off1++)
			{
				sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2); FILE *fp2 = fopen(Fname, "r");
				if (fp2 != NULL)
				{
					njobs[count] = 1;
					fclose(fp2);
				}
				//else
				//	printLOG("Have not seen %s yet. Found %d\n", Fname, count);
				count++;
			}
		}
		int sumRes = 0;
		for (int ii = 0; ii < (int)njobs.size(); ii++)
			sumRes += njobs[ii];

		if (sumRes == (int)njobs.size())
			break;
		if (bestSum < sumRes)
		{
			bestSum = sumRes;
			printLOG("(%d/%d) .. ", sumRes, (int)njobs.size());
		}
	}
	printLOG("ETime: %.4fs\n", omp_get_wtime() - nstartTime);

	count = 0;
	for (int off2 = LowBound; off2 <= UpBound; off2++)
	{
		for (int off1 = LowBound; off1 <= UpBound; off1++)
		{
			int ii, jj;
			OffsetID[count] = count;
			count++;
			sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2); FILE *fp2 = fopen(Fname, "r");
			fscanf(fp2, "%d %d %lf %lf %lf %lf %lf\n", &ii, &jj, &ceresCost, &APLDCost[0], &APLDCost[1], &APLDCost[2], &APLDCost[3]);
			fclose(fp2);

			AllCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = ceresCost,
				AllMPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[0],
				AllPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[1],
				AllLCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[2],
				AllDCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[3];
		}
	}
#else
	for (int off2 = LowBound; off2 <= UpBound; off2++)
	{
		for (int off1 = LowBound; off1 <= UpBound; off1++)
		{
			sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2);
			if (IsFileExist(Fname) == 0)
			{
				OffsetValue[count] = Point2d(off1, off2);
				currentOffset[0] = OffsetInfo[0], currentOffset[1] = off1 * frameSize + OffsetInfo[1], currentOffset[2] = off2 * frameSize + OffsetInfo[2];

				PerTraj_nFrames.clear();
				MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, nCams, Tscale);
				//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
				ceresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, true);

				//MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
				//ceresCost = MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

				printLOG("@(%d, %d) (id: %d): C: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\n", off1, off2, count, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);

				//Clean estimated 3D
				for (int trackID = 0; trackID < ntracks; trackID++)
				{
					int pcount = 0;
					for (int camID = 0; camID < nCams; camID++)
					{
						for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
						{
							PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(pcount, pcount, pcount);
							pcount++;
						}
					}
				}

				sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2); FILE *fp2 = fopen(Fname, "w+");
				fprintf(fp2, "%d %d %lf %lf %lf %lf %lf\n", off1, off2, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
				fclose(fp2);
			}
			count++;
		}
	}
	count = 0;
	for (int off2 = LowBound; off2 <= UpBound; off2++)
	{
		for (int off1 = LowBound; off1 <= UpBound; off1++)
		{
			int ii, jj;
			OffsetID[count] = count;
			count++;
			sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2); FILE *fp2 = fopen(Fname, "r");
			fscanf(fp2, "%d %d %lf %lf %lf %lf %lf\n", &ii, &jj, &ceresCost, &APLDCost[0], &APLDCost[1], &APLDCost[2], &APLDCost[3]);
			fclose(fp2);

			AllCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = ceresCost,
				AllMPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[0],
				AllPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[1],
				AllLCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[2],
				AllDCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[3];
		}
	}

	/*for (int off2 = LowBound; off2 <= UpBound; off2++)
	{
	for (int off1 = LowBound; off1 <= UpBound; off1++)
	{
	OffsetID[count] = count;
	OffsetValue[count] = Point2d(off1, off2);
	currentOffset[0] = OffsetInfo[0], currentOffset[1] = off1*frameSize + OffsetInfo[1], currentOffset[2] = off2*frameSize + OffsetInfo[2];

	PerTraj_nFrames.clear();
	MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, nCams, Tscale);
	//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
	ceresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, false);

	//MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
	//ceresCost = MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

	printLOG("@(%d, %d) (id: %d): C: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\n", off1, off2, count, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
	count++;

	//Clean estimated 3D
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
	int pcount = 0;
	for (int camID = 0; camID < nCams; camID++)
	{
	for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
	{
	PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(pcount, pcount, pcount);
	pcount++;
	}
	}
	}

	AllCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = ceresCost,
	AllMPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[0],
	AllPCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[1],
	AllLCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[2],
	AllDCost[(off1 - LowBound) + (off2 - LowBound)*(UpBound - LowBound + 1)] = APLDCost[3];
	}
	}*/
#endif

	Quick_Sort_Double(AllCost, OffsetID, 0, count - 1);
	printLOG("Min MPCost of %.5e with Offset of (%.4f, %.4f) (id: %d)\n", AllCost[0], frameSize*OffsetValue[OffsetID[0]].x + OffsetInfo[1], frameSize*OffsetValue[OffsetID[0]].y + OffsetInfo[2], OffsetID[0]);
	OffsetInfo[0] = 0, OffsetInfo[1] += frameSize * OffsetValue[OffsetID[0]].x, OffsetInfo[2] += frameSize * OffsetValue[OffsetID[0]].y; //do not set the ref cam to 0
																																	 //OffsetInfo[1] += frameSize*OffsetValue[OffsetID[0]].x, OffsetInfo[2] += frameSize*OffsetValue[OffsetID[0]].y;

	sprintf(Fname, "%s/TripletOffsetBruteforce.txt", Path); fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %d %f %f %f\n", SelectedCams[0], SelectedCams[1], SelectedCams[2], OffsetInfo[0], OffsetInfo[1], OffsetInfo[2]); fclose(fp);

	delete[]PerCam_UV, delete[]PerCam_XYZ, delete[]XYZ;
	delete[]OffsetID, delete[]AllCost, delete[]AllMPCost, delete[]AllPCost, delete[]AllLCost, delete[]AllDCost;
	for (int ii = 0; ii < (int)All3D.size(); ii++)
		delete[]All3D[ii], delete[]PerTrackFrameID[ii];

	return 0;
}
int MotionPriorSyncBruteForce2DTripletParallel(char *Path, vector<int> &SelectedCams, int startF, int stopF, int ntracks, vector<double> &OffsetInfo, int off1, int off2, double frameSize, double lamda, double RealOverSfm, int motionPriorPower)
{
	//Offset is in timestamp format
	const double Tscale = 1000000.0, eps = 1.0e-6;

	char Fname[512]; FILE *fp = 0;
	const int nCams = 3;

	//Read calib info
	VideoData VideoInfo[3];
	if (ReadVideoDataI(Path, VideoInfo[0], SelectedCams[0], startF, stopF) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[1], SelectedCams[1], startF, stopF) == 1)
		return 1;
	if (ReadVideoDataI(Path, VideoInfo[2], SelectedCams[2], startF, stopF) == 1)
		return 1;

	int id, frameID, nf;
	int nframes = max(MaxnFrames, stopF);

	double u, v, s, dummy;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[ntracks];

	//Get 2D info
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			PerCam_UV[camID*ntracks + trackID].reserve((stopF - startF + 1) / 50);

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot open %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &ntracks);
		while (fscanf(fp, "%d %d ", &id, &nf) != EOF)
		{
			//if (id != trackID)
			//	printLOG("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < nf; pid++)
			{
				fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf ", &frameID, &u, &v, &s, &dummy, &dummy, &dummy, &dummy, &dummy);
				if (frameID < startF || frameID > stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue;

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.scale = s, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}


	//Generate Calib Info
	double  AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < ntracks; pid++)
	{
		//Get ray direction, Q, U, P
		int pcount = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + pid].size(); frameID++)
			{
				ImgPtEle *ptEle = &PerCam_UV[camID*ntracks + pid][frameID];
				int RealFrameID = ptEle[0].frameID;
				CameraData *camI = VideoInfo[camID].VideoInfo;

				ptEle[0].fps = camI[RealFrameID].fps;
				ptEle[0].rollingShutterPercent = camI[RealFrameID].rollingShutterPercent;
				ptEle[0].shutterModel = camI[RealFrameID].ShutterModel;

				for (int kk = 0; kk < 9; kk++)
					ptEle[0].K[kk] = camI[RealFrameID].K[kk];

				//rayDirection = iR*(lamda*iK*[u,v,1] - T) - C
				double iR[9], tt[3], ttt[3];
				double pcn[3] = { camI[RealFrameID].invK[0] * ptEle[0].pt2D.x + camI[RealFrameID].invK[1] * ptEle[0].pt2D.y + camI[RealFrameID].invK[2],
					camI[RealFrameID].invK[4] * ptEle[0].pt2D.y + camI[RealFrameID].invK[5], 1.0 };

				if (camI[RealFrameID].ShutterModel == 0)
				{
					for (int kk = 0; kk < 12; kk++)
						ptEle[0].P[kk] = camI[RealFrameID].P[kk];
					for (int kk = 0; kk < 9; kk++)
						ptEle[0].R[kk] = camI[RealFrameID].R[kk];
					for (int kk = 0; kk < 3; kk++)
						ptEle[0].camcenter[kk] = camI[RealFrameID].camCenter[kk];

					mat_transpose(camI[RealFrameID].R, iR, 3, 3);
					mat_subtract(pcn, camI[RealFrameID].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].camcenter[ll] = camI[RealFrameID].camCenter[ll], ptEle[0].ray[ll] = camI[RealFrameID].camCenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else if (camI[RealFrameID].ShutterModel == 1)
				{
					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);

					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);
					AssembleRT_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].R, ptEle[0].T);
					GetCfromT(ptEle[0].R, ptEle[0].T, ptEle[0].camcenter);

					mat_transpose(ptEle[0].R, iR, 3, 3);
					mat_subtract(pcn, ptEle[0].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].ray[ll] = ptEle[0].camcenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else
					printLOG("Not supported model for motion prior sync\n");

				//Q, U
				AA[0] = ptEle[0].P[0], AA[1] = ptEle[0].P[1], AA[2] = ptEle[0].P[2], bb[0] = ptEle[0].P[3];
				AA[3] = ptEle[0].P[4], AA[4] = ptEle[0].P[5], AA[5] = ptEle[0].P[6], bb[1] = ptEle[0].P[7];
				ccT[0] = ptEle[0].P[8], ccT[1] = ptEle[0].P[9], ccT[2] = ptEle[0].P[10], dd[0] = ptEle[0].P[11];

				ptEle[0].Q[0] = AA[0] - ptEle[0].pt2D.x*ccT[0], ptEle[0].Q[1] = AA[1] - ptEle[0].pt2D.x*ccT[1], ptEle[0].Q[2] = AA[2] - ptEle[0].pt2D.x*ccT[2];
				ptEle[0].Q[3] = AA[3] - ptEle[0].pt2D.y*ccT[0], ptEle[0].Q[4] = AA[4] - ptEle[0].pt2D.y*ccT[1], ptEle[0].Q[5] = AA[5] - ptEle[0].pt2D.y*ccT[2];
				ptEle[0].u[0] = dd[0] * ptEle[0].pt2D.x - bb[0], ptEle[0].u[1] = dd[0] * ptEle[0].pt2D.y - bb[1];

				ptEle[0].pt3D = Point3d(pcount, pcount, pcount); pcount++;

				//depth
				double lamda[3] = { (ptEle[0].pt3D.x - ptEle[0].camcenter[0]) / ptEle[0].ray[0],
					(ptEle[0].pt3D.y - ptEle[0].camcenter[1]) / ptEle[0].ray[1],
					(ptEle[0].pt3D.z - ptEle[0].camcenter[2]) / ptEle[0].ray[2] };

				//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
				int index[3] = { 0, 1, 2 };
				double rayDirect[3] = { abs(ptEle[0].ray[0]), abs(ptEle[0].ray[1]), abs(ptEle[0].ray[2]) };
				Quick_Sort_Double(rayDirect, index, 0, 2);

				ptEle[0].d = lamda[index[2]];
			}
		}
	}

	//Initialize data for optim
	vector<int> PerTraj_nFrames;
	vector<int *> PerTrackFrameID(ntracks);
	vector<double*> All3D(ntracks);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < ntracks; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += (int)PerCam_UV[camID*ntracks + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	//Start sliding
	double APLDCost[4], ceresCost;
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	Point2d OffsetValue(off1, off2);
	double currentOffset[3] = { OffsetInfo[0], off1*frameSize + OffsetInfo[1], off2*frameSize + OffsetInfo[2] };

	PerTraj_nFrames.clear();
	MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, nCams, Tscale);
	//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
	ceresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, false);

	//MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
	//ceresCost = MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

	sprintf(Fname, "%s/TripletOffsetBruteforce_%d_%.4d.txt", Path, off1, off2); fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %.16f %.16f %.16f %.16f %.16f\n", off1, off2, ceresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
	fclose(fp);

	delete[]PerCam_UV, delete[]PerCam_XYZ, delete[]XYZ;
	for (int ii = 0; ii < (int)All3D.size(); ii++)
		delete[]All3D[ii], delete[]PerTrackFrameID[ii];

	return 0;
}
int IncrementalMotionPriorSyncDiscreteContinous2D(char *Path, vector<int> &SelectedCams, int startF, int stopF, int npts, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, double RealOverSfm, double &CeresCost, bool RefineConsiderOrdering)
{
	//Offset is in timestamp format
	char Fname[512]; FILE *fp = 0;
	const double Tscale = 1000000.0, eps = 1.0e-9;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startF, stopF) == 1)
			return 1;

	int frameID, id, nf;
	int nframes = max(MaxnFrames, stopF);

	double u, v, s, dummy;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*npts];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[npts];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int pid = 0; pid < npts; pid++)
			PerCam_UV[camID*npts + pid].reserve((stopF - startF + 1) / 50);

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &npts);
		while (fscanf(fp, "%d %d ", &id, &nf) != EOF)
		{
			PerCam_UV[camID*npts + id].reserve(nf);
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf ", &frameID, &u, &v, &s, &dummy, &dummy, &dummy, &dummy, &dummy);
				if (frameID < startF || frameID > stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue;

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.scale = s, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*npts + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	double  AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < npts; pid++)
	{
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*npts + pid].size(); frameID++)
			{
				ImgPtEle *ptEle = &PerCam_UV[camID*npts + pid][frameID];
				int RealFrameID = ptEle[0].frameID;
				CameraData *camI = VideoInfo[camID].VideoInfo;

				ptEle[0].fps = camI[RealFrameID].fps;
				ptEle[0].rollingShutterPercent = camI[RealFrameID].rollingShutterPercent;
				ptEle[0].shutterModel = camI[RealFrameID].ShutterModel;

				for (int kk = 0; kk < 9; kk++)
					ptEle[0].K[kk] = camI[RealFrameID].K[kk];

				//rayDirection = iR*(lamda*iK*[u,v,1] - T) - C
				double iR[9], tt[3], ttt[3];
				double pcn[3] = { camI[RealFrameID].invK[0] * ptEle[0].pt2D.x + camI[RealFrameID].invK[1] * ptEle[0].pt2D.y + camI[RealFrameID].invK[2],
					camI[RealFrameID].invK[4] * ptEle[0].pt2D.y + camI[RealFrameID].invK[5], 1.0 };

				if (camI[RealFrameID].ShutterModel == 0)
				{
					for (int kk = 0; kk < 12; kk++)
						ptEle[0].P[kk] = camI[RealFrameID].P[kk];
					for (int kk = 0; kk < 9; kk++)
						ptEle[0].R[kk] = camI[RealFrameID].R[kk];
					for (int kk = 0; kk < 3; kk++)
						ptEle[0].camcenter[kk] = camI[RealFrameID].camCenter[kk];

					mat_transpose(camI[RealFrameID].R, iR, 3, 3);
					mat_subtract(pcn, camI[RealFrameID].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].camcenter[ll] = camI[RealFrameID].camCenter[ll], ptEle[0].ray[ll] = camI[RealFrameID].camCenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else if (camI[RealFrameID].ShutterModel == 1)
				{
					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);

					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);
					AssembleRT_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].R, ptEle[0].T);
					GetCfromT(ptEle[0].R, ptEle[0].T, ptEle[0].camcenter);

					mat_transpose(ptEle[0].R, iR, 3, 3);
					mat_subtract(pcn, ptEle[0].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].ray[ll] = ptEle[0].camcenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else
					printLOG("Not supported model for motion prior sync\n");

				//Q, U
				AA[0] = ptEle[0].P[0], AA[1] = ptEle[0].P[1], AA[2] = ptEle[0].P[2], bb[0] = ptEle[0].P[3];
				AA[3] = ptEle[0].P[4], AA[4] = ptEle[0].P[5], AA[5] = ptEle[0].P[6], bb[1] = ptEle[0].P[7];
				ccT[0] = ptEle[0].P[8], ccT[1] = ptEle[0].P[9], ccT[2] = ptEle[0].P[10], dd[0] = ptEle[0].P[11];

				ptEle[0].Q[0] = AA[0] - ptEle[0].pt2D.x*ccT[0], ptEle[0].Q[1] = AA[1] - ptEle[0].pt2D.x*ccT[1], ptEle[0].Q[2] = AA[2] - ptEle[0].pt2D.x*ccT[2];
				ptEle[0].Q[3] = AA[3] - ptEle[0].pt2D.y*ccT[0], ptEle[0].Q[4] = AA[4] - ptEle[0].pt2D.y*ccT[1], ptEle[0].Q[5] = AA[5] - ptEle[0].pt2D.y*ccT[2];
				ptEle[0].u[0] = dd[0] * ptEle[0].pt2D.x - bb[0], ptEle[0].u[1] = dd[0] * ptEle[0].pt2D.y - bb[1];

				ptEle[0].pt3D = Point3d(count, count, count); count++;

				//depth
				double lamda[3] = { (ptEle[0].pt3D.x - ptEle[0].camcenter[0]) / ptEle[0].ray[0],
					(ptEle[0].pt3D.y - ptEle[0].camcenter[1]) / ptEle[0].ray[1],
					(ptEle[0].pt3D.z - ptEle[0].camcenter[2]) / ptEle[0].ray[2] };

				//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
				int index[3] = { 0, 1, 2 };
				double rayDirect[3] = { abs(ptEle[0].ray[0]), abs(ptEle[0].ray[1]), abs(ptEle[0].ray[2]) };
				Quick_Sort_Double(rayDirect, index, 0, 2);

				ptEle[0].d = lamda[index[2]];
			}
		}
	}

	//Initialize data for optim
	vector<int> PerTraj_nFrames;
	vector<int *> PerTrackFrameID(npts);
	vector<double*> All3D(npts);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < npts; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += (int)PerCam_UV[camID*npts + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	//Step 1: Start sliding to make sure that you are at frame level accurate
	double APLDCost[4];
	double currentOffset[MaxnCams];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	printLOG("ST estimation for %d camaras ( ", nCams);
	for (int ii = 0; ii < nCams; ii++)
		printLOG("%d ", SelectedCams[ii]);
	printLOG("): \n");

	int NotSuccess = 0;
	if (RefineConsiderOrdering)
	{
		//Step 2: insert the new camera to the pre-order camears since brute force sliding may not guarantee correct ordering
		int naddedCams = nCams - 1;
		double subframeRemander[MaxnCams], subframeRemander2[MaxnCams];
		int subframeRemanderID[MaxnCams], subframeRemanderID2[MaxnCams];
		for (int ii = 0; ii < naddedCams; ii++)
		{
			subframeRemanderID[ii] = ii;
			subframeRemander[ii] = OffsetInfo[ii] - floor(OffsetInfo[ii]);
		}
		Quick_Sort_Double(subframeRemander, subframeRemanderID, 0, naddedCams - 1);

		double subframeSlots[MaxnCams];
		for (int ii = 0; ii < naddedCams - 1; ii++)
			subframeSlots[ii] = 0.5*(subframeRemander[ii] + subframeRemander[ii + 1]);
		subframeSlots[naddedCams - 1] = 0.5*(subframeRemander[naddedCams - 1] + subframeRemander[0] + 1.0);

		int bestID = 0;
		double CeresCost, bestCeresCost = 9e20, bestMPCost = 9e20;
		double *BestOffset = new double[nCams];
		vector<int> iTimeOrdering, fTimeOrdering;
		int NotFlippedTimes = 0;
#ifdef EC2
		printLOG("EC2 Best Slot Search:\n");
		double nstartTime = omp_get_wtime();
		vector<int> njobs;
		for (int SearchOffset = LowBound; SearchOffset <= UpBound; SearchOffset++)
		{
			for (int cid = 0; cid < naddedCams; cid++)//try different slots
			{
				for (int jj = 0; jj < naddedCams; jj++)
					currentOffset[jj] = OffsetInfo[jj];
				currentOffset[nCams - 1] = floor(OffsetInfo[nCams - 1]) + subframeSlots[cid] + SearchOffset;

				sprintf(Fname, "%s/IncreMotionSync_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset); fp = fopen(Fname, "w+");
				for (int jj = 0; jj < nCams; jj++)
					fprintf(fp, "%d %.16f ", SelectedCams[jj], currentOffset[jj]);
				fclose(fp);

				sprintf(Fname, "%s/IncreST_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset);
				if (IsFileExist(Fname) == 0)
				{
#ifdef _WINDOWS
					sprintf(Fname, "EnRecon.exe 7 1 3 %d %d %d", SearchOffset, cid, nCams);
#else
					sprintf(Fname, "qsub -b y -cwd -pe orte %d ./EnRecon 7 1 3 %d %d %d", omp_get_max_threads(), SearchOffset, cid, nCams);
#endif
					printLOG("%s\n", Fname);
					mySleep(1e3); system(Fname);
				}
				njobs.push_back(0);
			}
		}
		printLOG("Start waiting ...\n");
		int bestSum = 0;
		while (true)
		{
			mySleep(1e3);
			int count = 0;
			for (int SearchOffset = LowBound; SearchOffset <= UpBound; SearchOffset++)
			{
				for (int cid = 0; cid < naddedCams; cid++)//try different slots
				{
					sprintf(Fname, "%s/IncreST_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset); FILE *fp2 = fopen(Fname, "r");
					if (fp2 != NULL)
					{
						njobs[count] = 1;
						fclose(fp2);
					}
					//else
					//	printLOG("Have not seen %s yet. Found %d\n", Fname, count);
					count++;
				}
			}
			int sumRes = 0;
			for (int ii = 0; ii < (int)njobs.size(); ii++)
				sumRes += njobs[ii];

			if (sumRes == (int)njobs.size())
				break;
			if (bestSum < sumRes)
			{
				bestSum = sumRes;
				printLOG("(%d/%d) .. ", sumRes, (int)njobs.size());
			}
		}
		printLOG("ETime: %.4fs\n", omp_get_wtime() - nstartTime);

		int count = 0;
		for (int SearchOffset = LowBound; SearchOffset <= UpBound; SearchOffset++)
		{
			for (int cid = 0; cid < naddedCams; cid++)//try different slots
			{
				for (int jj = 0; jj < naddedCams; jj++)
					currentOffset[jj] = OffsetInfo[jj];
				currentOffset[nCams - 1] = floor(OffsetInfo[nCams - 1]) + subframeSlots[cid] + SearchOffset;
				printLOG("Trial %d/%d (%.4f): ", cid + (SearchOffset - LowBound)*naddedCams, (UpBound - LowBound + 1)*naddedCams, currentOffset[nCams - 1]);

				for (int jj = 0; jj < nCams; jj++)
				{
					subframeRemanderID[jj] = jj;
					subframeRemander[jj] = currentOffset[jj] - floor(currentOffset[jj]);
				}
				Quick_Sort_Double(subframeRemander, subframeRemanderID, 0, nCams - 1);

				sprintf(Fname, "%s/IncreST_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset); fp = fopen(Fname, "r");
				fscanf(fp, "%lf %lf %lf %lf %lf ", &CeresCost, &APLDCost[0], &APLDCost[1], &APLDCost[2], &APLDCost[3]);
				for (int jj = 0; jj < nCams; jj++)
					fscanf(fp, "%lf ", &currentOffset[jj]);
				fclose(fp);

				printLOG("Cost: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\nOffsets:", CeresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
				for (int jj = 0; jj < nCams; jj++)
					printLOG("%.4f ", currentOffset[jj]);
				printLOG("\n");

				//check if time order has been flipped
				for (int jj = 0; jj < nCams; jj++)
				{
					subframeRemanderID2[jj] = jj;
					subframeRemander2[jj] = currentOffset[jj] - floor(currentOffset[jj]);
				}
				Quick_Sort_Double(subframeRemander2, subframeRemanderID2, 0, nCams - 1);

				bool flipped = false;
				for (int jj = 0; jj < nCams; jj++)
				{
					if (subframeRemanderID[jj] != subframeRemanderID2[jj])
					{
						flipped = true;
						break;
					}
				}
				if (flipped)
				{
					printLOG("Local flipping occurs!\n");
					continue;
				}
				else
					NotFlippedTimes++;

				if (bestCeresCost > CeresCost)
				{
					bestID = cid, bestCeresCost = CeresCost;
					for (int jj = 0; jj < nCams; jj++)
						BestOffset[jj] = currentOffset[jj];
				}
			}
		}
#else
		for (int SearchOffset = LowBound; SearchOffset <= UpBound; SearchOffset++)
		{
			for (int cid = 0; cid < naddedCams; cid++)//try different slots
			{
				for (int jj = 0; jj < naddedCams; jj++)
					currentOffset[jj] = OffsetInfo[jj];
				currentOffset[nCams - 1] = floor(OffsetInfo[nCams - 1]) + subframeSlots[cid] + SearchOffset;

				for (int jj = 0; jj < nCams; jj++)
				{
					subframeRemanderID[jj] = jj;
					subframeRemander[jj] = currentOffset[jj] - floor(currentOffset[jj]);
				}
				Quick_Sort_Double(subframeRemander, subframeRemanderID, 0, nCams - 1);

				printLOG("Trial %d/%d (%.4f): ", cid + (SearchOffset - LowBound)*naddedCams, (UpBound - LowBound + 1)*naddedCams, currentOffset[nCams - 1]);
				MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, nCams, Tscale);
				//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
				CeresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, true);
				CeresCost = MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, APLDCost, false, true);

				//MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
				//CeresCost = MotionPrior_Optim_SpatialTemporal_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, true, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

				printLOG("Cost: %.5e Ac: %.5e Pc: %.5e Lc: %.5f Dc: %.5f\nOffsets:", CeresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
				for (int jj = 0; jj < nCams; jj++)
					printLOG("%.4f ", currentOffset[jj]);
				printLOG("\n");

				sprintf(Fname, "%s/_IncreST_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset); fp = fopen(Fname, "w+");
				fprintf(fp, "%.16e %.16e %.16e %.16e %.16e ", CeresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
				for (int jj = 0; jj < nCams; jj++)
					fprintf(fp, "%.8f ", currentOffset[jj]);
				fclose(fp);

				//check if time order has been flipped
				for (int jj = 0; jj < nCams; jj++)
				{
					subframeRemanderID2[jj] = jj;
					subframeRemander2[jj] = currentOffset[jj] - floor(currentOffset[jj]);
				}
				Quick_Sort_Double(subframeRemander2, subframeRemanderID2, 0, nCams - 1);

				bool flipped = false;
				for (int jj = 0; jj < nCams; jj++)
				{
					if (subframeRemanderID[jj] != subframeRemanderID2[jj])
					{
						flipped = true;
						break;
					}
				}
				if (flipped)
				{
					printLOG("Local flipping occurs!\n");
					continue;
				}
				else
					NotFlippedTimes++;

				if (bestCeresCost > CeresCost)
				{
					bestID = cid, bestCeresCost = CeresCost;
					for (int jj = 0; jj < nCams; jj++)
						BestOffset[jj] = currentOffset[jj];
				}

				//Clean estimated 3D
				for (int trackID = 0; trackID < npts; trackID++)
					for (int camID = 0; camID < nCams; camID++)
						for (int frameID = 0; frameID < PerCam_UV[camID*npts + trackID].size(); frameID++)
							PerCam_UV[camID*npts + trackID][frameID].pt3D = Point3d(gaussian_noise(0.0, 1), gaussian_noise(0.0, 1), gaussian_noise(0.0, 1));
			}
		}
#endif

		if (NotFlippedTimes == 0)
			NotSuccess = 1;

		if (bestCeresCost < 9e20)
			for (int jj = 0; jj < nCams; jj++)
				OffsetInfo[jj] = BestOffset[jj];
	}
	else
	{
		for (int ii = 0; ii < nCams; ii++)
			currentOffset[ii] = OffsetInfo[ii];

		MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, nCams, Tscale);
		//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
		MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, true);

		if (nCams == 3)
		{
			double currentOffsetA[3], currentOffsetB[3];
			vector<ImgPtEle> *PerCam_UV_BK = 0;

			//Detect if sup-frame sync happens
			int happen = 0;
			if (abs(currentOffset[0] - currentOffset[1]) < 0.05)
				happen = 1;
			else if (abs(currentOffset[0] - currentOffset[2]) < 0.05)
				happen = 2;
			else if (abs(currentOffset[1] - currentOffset[2]) < 0.05)
				happen = 3;
			else if (abs(currentOffset[0] - currentOffset[1]) < 0.05 && abs(currentOffset[0] - currentOffset[2]) < 0.05)
			{
				happen = 4;
				printLOG("Ops, all 3 initial camera are sub-frame sync. Please manually work it out\n");
				exit(1);
			}

			if (happen == 0) //not overlapping time
				MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, APLDCost, false, true);
			else
			{
				printLOG("Sub-frame sync happens. Pertupating the time and pick the best cost\n");
				PerCam_UV_BK = new vector<ImgPtEle>[nCams*npts];
				for (int trackID = 0; trackID < npts; trackID++)
					for (int camID = 0; camID < nCams; camID++)
						for (int frameID = 0; frameID < PerCam_UV[camID*npts + trackID].size(); frameID++)
							PerCam_UV_BK[camID*npts + trackID].push_back(PerCam_UV[camID*npts + trackID][frameID]);

				if (happen == 1)
				{
					for (int ii = 0; ii < 3; ii++)
						currentOffsetA[ii] = currentOffset[ii];
					currentOffsetA[0] += 0.05, currentOffsetA[1] -= 0.05;
					for (int ii = 0; ii < 3; ii++)
						currentOffsetB[ii] = currentOffset[ii];
					currentOffsetB[0] -= 0.05, currentOffsetB[1] += 0.05;
				}
				else if (happen == 2)
				{
					for (int ii = 0; ii < 3; ii++)
						currentOffsetA[ii] = currentOffset[ii];
					currentOffsetA[0] += 0.05, currentOffsetA[2] -= 0.05;
					for (int ii = 0; ii < 3; ii++)
						currentOffsetB[ii] = currentOffset[ii];
					currentOffsetB[0] -= 0.05, currentOffsetB[2] += 0.05;
				}
				else if (happen == 3)
				{
					for (int ii = 0; ii < 3; ii++)
						currentOffsetA[ii] = currentOffset[ii];
					currentOffsetA[1] += 0.05, currentOffsetA[2] -= 0.05;
					for (int ii = 0; ii < 3; ii++)
						currentOffsetB[ii] = currentOffset[ii];
					currentOffsetB[1] -= 0.05, currentOffsetB[2] += 0.05;
				}

				double cost1 = MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffsetA, npts, false, nCams, Tscale, eps, APLDCost, false, true);
				printLOG("Cost1: %.8e: %.5f %.5f \n", cost1, currentOffsetA[1], currentOffsetA[2]);

				for (int trackID = 0; trackID < npts; trackID++)
					for (int camID = 0; camID < nCams; camID++)
						for (int frameID = 0; frameID < PerCam_UV[camID*npts + trackID].size(); frameID++)
							PerCam_UV[camID*npts + trackID][frameID].pt3D = PerCam_UV_BK[camID*npts + trackID][frameID].pt3D;

				double cost2 = MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffsetB, npts, false, nCams, Tscale, eps, APLDCost, false, true);
				printLOG("Cost2: %.8e: %.5f %.5f \n", cost2, currentOffsetB[1], currentOffsetB[2]);

				if (cost1 < cost2)
					for (int ii = 0; ii < 3; ii++)
						currentOffset[ii] = currentOffsetA[ii];
				else
					for (int ii = 0; ii < 3; ii++)
						currentOffset[ii] = currentOffsetB[ii];

				delete[]PerCam_UV_BK;
			}
		}
		else
		{
			CeresCost = MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, APLDCost, false, true);
			printLOG("Temporal estimation from ray shooting: ");
			for (int ii = 0; ii < nCams; ii++)
				printLOG("%f ", OffsetInfo[ii]);
			printLOG("\n\n");

			MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
			CeresCost = MotionPrior_Optim_SpatialTemporal_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, true, nCams, 2, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
		}

		for (int jj = 0; jj < nCams; jj++)
			OffsetInfo[jj] = currentOffset[jj];
	}

	if (NotSuccess == 0)
	{
		printLOG("Final temporal estimation: ");
		for (int ii = 0; ii < nCams; ii++)
			printLOG("%f ", OffsetInfo[ii]);
		printLOG("\n\n");
	}
	else
		printLOG("Global flipping occurs for all trials. Start to shuffle the ordering stack and retry.\n");

	delete[]VideoInfo, delete[]PerCam_UV, delete[]PerCam_XYZ, delete[]XYZ;
	for (int ii = 0; ii < (int)All3D.size(); ii++)
		delete[]All3D[ii], delete[]PerTrackFrameID[ii];

	return NotSuccess;
}
int IncrementalMotionPriorSyncDiscreteContinous2DParallel(char *Path, vector<int> &SelectedCams, int startF, int stopF, int npts, vector<double> &OffsetInfo, int SearchOffset, int cid, double lamda, double RealOverSfm)
{
	//Offset is in timestamp format
	char Fname[512]; FILE *fp = 0;
	const double Tscale = 1000000.0, eps = 1.0e-9;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startF, stopF) == 1)
			return 1;

	int frameID, id, nf;
	int nframes = max(MaxnFrames, stopF);

	double u, v, s, dummy;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*npts];
	vector<XYZD> *PerCam_XYZ = new vector<XYZD>[nCams], *XYZ = new vector<XYZD>[npts];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int pid = 0; pid < npts; pid++)
			PerCam_UV[camID*npts + pid].reserve((stopF - startF + 1) / 50);

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &npts);
		while (fscanf(fp, "%d %d ", &id, &nf) != EOF)
		{
			//if (id != trackID)
			//	printLOG("Problem at Point %d of Cam %d", id, camID);
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf ", &frameID, &u, &v, &s, &dummy, &dummy, &dummy, &dummy, &dummy);
				if (frameID < startF || frameID > stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue;

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.scale = s, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*npts + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	double  AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < npts; pid++)
	{
		int count = 0;
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < PerCam_UV[camID*npts + pid].size(); frameID++)
			{
				ImgPtEle *ptEle = &PerCam_UV[camID*npts + pid][frameID];
				int RealFrameID = ptEle[0].frameID;
				CameraData *camI = VideoInfo[camID].VideoInfo;

				ptEle[0].fps = camI[RealFrameID].fps;
				ptEle[0].rollingShutterPercent = camI[RealFrameID].rollingShutterPercent;
				ptEle[0].shutterModel = camI[RealFrameID].ShutterModel;

				for (int kk = 0; kk < 9; kk++)
					ptEle[0].K[kk] = camI[RealFrameID].K[kk];

				//rayDirection = iR*(lamda*iK*[u,v,1] - T) - C
				double iR[9], tt[3], ttt[3];
				double pcn[3] = { camI[RealFrameID].invK[0] * ptEle[0].pt2D.x + camI[RealFrameID].invK[1] * ptEle[0].pt2D.y + camI[RealFrameID].invK[2],
					camI[RealFrameID].invK[4] * ptEle[0].pt2D.y + camI[RealFrameID].invK[5], 1.0 };

				if (camI[RealFrameID].ShutterModel == 0)
				{
					for (int kk = 0; kk < 12; kk++)
						ptEle[0].P[kk] = camI[RealFrameID].P[kk];
					for (int kk = 0; kk < 9; kk++)
						ptEle[0].R[kk] = camI[RealFrameID].R[kk];
					for (int kk = 0; kk < 3; kk++)
						ptEle[0].camcenter[kk] = camI[RealFrameID].camCenter[kk];

					mat_transpose(camI[RealFrameID].R, iR, 3, 3);
					mat_subtract(pcn, camI[RealFrameID].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].camcenter[ll] = camI[RealFrameID].camCenter[ll], ptEle[0].ray[ll] = camI[RealFrameID].camCenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else if (camI[RealFrameID].ShutterModel == 1)
				{
					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);

					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);
					AssembleRT_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].R, ptEle[0].T);
					GetCfromT(ptEle[0].R, ptEle[0].T, ptEle[0].camcenter);

					mat_transpose(ptEle[0].R, iR, 3, 3);
					mat_subtract(pcn, ptEle[0].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].ray[ll] = ptEle[0].camcenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else
					printLOG("Not supported model for motion prior sync\n");

				//Q, U
				AA[0] = ptEle[0].P[0], AA[1] = ptEle[0].P[1], AA[2] = ptEle[0].P[2], bb[0] = ptEle[0].P[3];
				AA[3] = ptEle[0].P[4], AA[4] = ptEle[0].P[5], AA[5] = ptEle[0].P[6], bb[1] = ptEle[0].P[7];
				ccT[0] = ptEle[0].P[8], ccT[1] = ptEle[0].P[9], ccT[2] = ptEle[0].P[10], dd[0] = ptEle[0].P[11];

				ptEle[0].Q[0] = AA[0] - ptEle[0].pt2D.x*ccT[0], ptEle[0].Q[1] = AA[1] - ptEle[0].pt2D.x*ccT[1], ptEle[0].Q[2] = AA[2] - ptEle[0].pt2D.x*ccT[2];
				ptEle[0].Q[3] = AA[3] - ptEle[0].pt2D.y*ccT[0], ptEle[0].Q[4] = AA[4] - ptEle[0].pt2D.y*ccT[1], ptEle[0].Q[5] = AA[5] - ptEle[0].pt2D.y*ccT[2];
				ptEle[0].u[0] = dd[0] * ptEle[0].pt2D.x - bb[0], ptEle[0].u[1] = dd[0] * ptEle[0].pt2D.y - bb[1];

				ptEle[0].pt3D = Point3d(count, count, count); count++;

				//depth
				double lamda[3] = { (ptEle[0].pt3D.x - ptEle[0].camcenter[0]) / ptEle[0].ray[0],
					(ptEle[0].pt3D.y - ptEle[0].camcenter[1]) / ptEle[0].ray[1],
					(ptEle[0].pt3D.z - ptEle[0].camcenter[2]) / ptEle[0].ray[2] };

				//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
				int index[3] = { 0, 1, 2 };
				double rayDirect[3] = { abs(ptEle[0].ray[0]), abs(ptEle[0].ray[1]), abs(ptEle[0].ray[2]) };
				Quick_Sort_Double(rayDirect, index, 0, 2);

				ptEle[0].d = lamda[index[2]];
			}
		}
	}

	//Initialize data for optim
	vector<int> PerTraj_nFrames;
	vector<int *> PerTrackFrameID(npts);
	vector<double*> All3D(npts);
	int ntimeinstances, maxntimeinstances = 0;
	for (int trackID = 0; trackID < npts; trackID++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += (int)PerCam_UV[camID*npts + trackID].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[trackID] = new int[ntimeinstances];
		All3D[trackID] = new double[3 * ntimeinstances];
	}

	//insert the new camera to the pre-order camear slots since brute force sliding may be too coase to guarantee correct ordering
	printLOG("ST estimation for %d camaras ( ", nCams);
	for (int ii = 0; ii < nCams; ii++)
		printLOG("%d ", SelectedCams[ii]);
	printLOG("): \n");

	double CeresCost, APLDCost[4];
	double currentOffset[MaxnCams];
	for (int ii = 0; ii < nCams; ii++)
		currentOffset[ii] = OffsetInfo[ii];
	double initOffset = currentOffset[nCams - 1];
	MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, nCams, Tscale);
	//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, 2, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);
	CeresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, true);
	CeresCost = MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, APLDCost, false, true);

	/*printLOG("T1 (%.3f): ", initOffset);
	for (int ii = 0; ii < nCams; ii++)
	printLOG("%f ", currentOffset[ii]);
	printLOG("\n");

	MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
	CeresCost = MotionPrior_Optim_SpatialTemporal_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, true, nCams, 2, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

	printLOG("T2(%.3f): ", initOffset);
	for (int ii = 0; ii < nCams; ii++)
	printLOG("%f ", currentOffset[ii]);
	printLOG("\n");*/

	/*for (int pid = 0; pid < npts; pid++)
	{
	int count = 0;
	for (int camID = 0; camID < nCams; camID++)
	{
	for (int frameID = 0; frameID < PerCam_UV[camID*npts + pid].size(); frameID++)
	{
	PerCam_UV[camID*npts + pid][frameID].pt3D = Point3d(count, count, count);
	count++;

	//depth
	double lamda[3] = { (PerCam_UV[camID*npts + pid][frameID].pt3D.x - PerCam_UV[camID*npts + pid][frameID].camcenter[0]) / PerCam_UV[camID*npts + pid][frameID].ray[0],
	(PerCam_UV[camID*npts + pid][frameID].pt3D.y - PerCam_UV[camID*npts + pid][frameID].camcenter[1]) / PerCam_UV[camID*npts + pid][frameID].ray[1],
	(PerCam_UV[camID*npts + pid][frameID].pt3D.z - PerCam_UV[camID*npts + pid][frameID].camcenter[2]) / PerCam_UV[camID*npts + pid][frameID].ray[2] };

	//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
	int index[3] = { 0, 1, 2 };
	double rayDirect[3] = { abs(PerCam_UV[camID*npts + pid][frameID].ray[0]), abs(PerCam_UV[camID*npts + pid][frameID].ray[1]), abs(PerCam_UV[camID*npts + pid][frameID].ray[2]) };
	Quick_Sort_Double(rayDirect, index, 0, 2);

	PerCam_UV[camID*npts + pid][frameID].d = lamda[index[2]];
	}
	}
	}

	CeresCost = MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, true);
	CeresCost = MotionPrior_Optim_SpatialTemporalStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, currentOffset, npts, false, nCams, Tscale, eps, APLDCost, false, true);

	printLOG("T3 (%.3f): ", initOffset);
	for (int ii = 0; ii < nCams; ii++)
	printLOG("%f ", currentOffset[ii]);
	printLOG("\n");

	MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
	CeresCost = MotionPrior_Optim_SpatialTemporal_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, npts, true, nCams, 2, Tscale, eps, lamda, RealOverSfm, APLDCost, false, true);

	printLOG("T4 (%.3f): ", initOffset);
	for (int ii = 0; ii < nCams; ii++)
	printLOG("%f ", currentOffset[ii]);
	printLOG("\n");*/

	for (int ii = 0; ii < nCams; ii++)
		OffsetInfo[ii] = currentOffset[ii];

	sprintf(Fname, "%s/IncreST_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset); fp = fopen(Fname, "w+");
	fprintf(fp, "%.16e %.16e %.16e %.16e %.16e ", CeresCost, APLDCost[0], APLDCost[1], APLDCost[2], APLDCost[3]);
	for (int jj = 0; jj < nCams; jj++)
		fprintf(fp, "%.8f ", currentOffset[jj]);
	fclose(fp);

	delete[]VideoInfo, delete[]PerCam_UV, delete[]PerCam_XYZ, delete[]XYZ;
	for (int ii = 0; ii < (int)All3D.size(); ii++)
		delete[]All3D[ii], delete[]PerTrackFrameID[ii];

	return 0;
}
int TrajectoryTriangulation(char *Path, vector<int> &SelectedCams, vector<double> &TimeStampInfoVector, int npts, int startF, int stopF, double lamda, double RealOverSfm, int motionPriorPower)
{
	char Fname[512]; FILE *fp = 0;
	const double Tscale = 1000000.0, eps = 1.0e-6;

	//Read calib info
	int notLoaded = 0, nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startF, stopF) == 1)
			notLoaded++;
	if (notLoaded > nCams - 2)
		return 1;

	int frameID, id, nf;
	int nframes = max(MaxnFrames, stopF);

	double u, v, s, dummy;
	vector<int>VectorCamID, VectorFrameID;
	vector<double> AllError2D;
	ImgPtEle ptEle;
	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*npts];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int pid = 0; pid < npts; pid++)
			PerCam_UV[camID*npts + pid].reserve((stopF - startF + 1) / 50);

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &npts);
		while (fscanf(fp, "%d %d ", &id, &nf) != EOF)
		{
			PerCam_UV[camID*npts + id].reserve(nf);
			for (int fid = 0; fid < nf; fid++)
			{
				fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf ", &frameID, &u, &v, &s, &dummy, &dummy, &dummy, &dummy, &dummy);
				if (frameID < startF || frameID > stopF)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue;

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.scale = s, ptEle.viewID = camID, ptEle.frameID = frameID, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					PerCam_UV[camID*npts + id].push_back(ptEle);

				}
			}
		}
		fclose(fp);
	}

	//Generate Calib Info
	double AA[6], bb[2], ccT[3], dd[1];
	for (int pid = 0; pid < npts; pid++)
	{
		//Get ray direction, Q, U, P
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int frameID = 0; frameID < (int)PerCam_UV[camID*npts + pid].size(); frameID++)
			{
				ImgPtEle *ptEle = &PerCam_UV[camID*npts + pid][frameID];
				int RealFrameID = ptEle[0].frameID;
				CameraData *camI = VideoInfo[camID].VideoInfo;

				ptEle[0].fps = camI[RealFrameID].fps;
				ptEle[0].rollingShutterPercent = camI[RealFrameID].rollingShutterPercent;
				ptEle[0].shutterModel = camI[RealFrameID].ShutterModel;

				for (int kk = 0; kk < 9; kk++)
					ptEle[0].K[kk] = camI[RealFrameID].K[kk];

				//rayDirection = iR*(lamda*iK*[u,v,1] - T) - C
				double iR[9], tt[3], ttt[3];
				double pcn[3] = { camI[RealFrameID].invK[0] * ptEle[0].pt2D.x + camI[RealFrameID].invK[1] * ptEle[0].pt2D.y + camI[RealFrameID].invK[2],
					camI[RealFrameID].invK[4] * ptEle[0].pt2D.y + camI[RealFrameID].invK[5], 1.0 };

				if (camI[RealFrameID].ShutterModel == 0)
				{
					for (int kk = 0; kk < 12; kk++)
						ptEle[0].P[kk] = camI[RealFrameID].P[kk];
					for (int kk = 0; kk < 9; kk++)
						ptEle[0].R[kk] = camI[RealFrameID].R[kk];
					for (int kk = 0; kk < 3; kk++)
						ptEle[0].camcenter[kk] = camI[RealFrameID].camCenter[kk];

					mat_transpose(camI[RealFrameID].R, iR, 3, 3);
					mat_subtract(pcn, camI[RealFrameID].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].camcenter[ll] = camI[RealFrameID].camCenter[ll], ptEle[0].ray[ll] = camI[RealFrameID].camCenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else if (camI[RealFrameID].ShutterModel == 1)
				{
					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);

					AssembleP_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].P);
					AssembleRT_RS(ptEle[0].pt2D, camI[RealFrameID], ptEle[0].R, ptEle[0].T);
					GetCfromT(ptEle[0].R, ptEle[0].T, ptEle[0].camcenter);

					mat_transpose(ptEle[0].R, iR, 3, 3);
					mat_subtract(pcn, ptEle[0].T, tt, 3, 1, 1000.0); //Scaling the pcn gives better numerical precision
					mat_mul(iR, tt, ttt, 3, 3, 1);

					for (int ll = 0; ll < 3; ll++)
						ptEle[0].ray[ll] = ptEle[0].camcenter[ll] - ttt[ll];
					normalize(ptEle[0].ray);
				}
				else
					printLOG("Not supported model for motion prior sync\n");

				//Q, U
				AA[0] = ptEle[0].P[0], AA[1] = ptEle[0].P[1], AA[2] = ptEle[0].P[2], bb[0] = ptEle[0].P[3];
				AA[3] = ptEle[0].P[4], AA[4] = ptEle[0].P[5], AA[5] = ptEle[0].P[6], bb[1] = ptEle[0].P[7];
				ccT[0] = ptEle[0].P[8], ccT[1] = ptEle[0].P[9], ccT[2] = ptEle[0].P[10], dd[0] = ptEle[0].P[11];

				ptEle[0].Q[0] = AA[0] - ptEle[0].pt2D.x*ccT[0], ptEle[0].Q[1] = AA[1] - ptEle[0].pt2D.x*ccT[1], ptEle[0].Q[2] = AA[2] - ptEle[0].pt2D.x*ccT[2];
				ptEle[0].Q[3] = AA[3] - ptEle[0].pt2D.y*ccT[0], ptEle[0].Q[4] = AA[4] - ptEle[0].pt2D.y*ccT[1], ptEle[0].Q[5] = AA[5] - ptEle[0].pt2D.y*ccT[2];
				ptEle[0].u[0] = dd[0] * ptEle[0].pt2D.x - bb[0], ptEle[0].u[1] = dd[0] * ptEle[0].pt2D.y - bb[1];

				double stdA = 10000.0;//Interestingly, Ceres does not work if all the input are the same-->need some random perturbation. 
				ptEle[0].pt3D = Point3d(gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA));

				//depth
				double lamda[3] = { (ptEle[0].pt3D.x - ptEle[0].camcenter[0]) / ptEle[0].ray[0],
					(ptEle[0].pt3D.y - ptEle[0].camcenter[1]) / ptEle[0].ray[1],
					(ptEle[0].pt3D.z - ptEle[0].camcenter[2]) / ptEle[0].ray[2] };

				//Find the direction with largest value--> to avoid direction with super small magnitude which leads to inf depth
				int index[3] = { 0, 1, 2 };
				double rayDirect[3] = { abs(ptEle[0].ray[0]), abs(ptEle[0].ray[1]), abs(ptEle[0].ray[2]) };
				Quick_Sort_Double(rayDirect, index, 0, 2);

				ptEle[0].d = lamda[index[2]];
			}
		}
	}

	//Initialize data for optim
	vector<int> PerTraj_nFrames;
	vector<int *> PerTrackFrameID(npts);
	vector<double*> All3D(npts);
	int ntimeinstances, maxntimeinstances = 0;
	for (int pid = 0; pid < npts; pid++)
	{
		ntimeinstances = 0;
		for (int camID = 0; camID < nCams; camID++)
			ntimeinstances += (int)PerCam_UV[camID*npts + pid].size();

		if (maxntimeinstances < ntimeinstances)
			maxntimeinstances = ntimeinstances;

		PerTrackFrameID[pid] = new int[ntimeinstances];
		All3D[pid] = new double[3 * ntimeinstances];
	}

	double APLDCost[4];
	vector<double> VTimeStamp; VTimeStamp.reserve(maxntimeinstances);
	vector<Point3d> VTrajectory3D; VTrajectory3D.reserve(maxntimeinstances);

	double *OffsetInfo = new double[nCams];
	for (int ii = 0; ii < nCams; ii++)
		OffsetInfo[ii] = TimeStampInfoVector[ii];

	printLOG("Naive trajectory triangulation:\n");
	MotionPrior_Optim_Init_SpatialStructure_Triangulation(Path, All3D, PerCam_UV, PerTraj_nFrames, OffsetInfo, npts, nCams, Tscale);
	//MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, OffsetInfo, npts, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, false);
	MotionPrior_Optim_SpatialStructure_Rays(Path, PerCam_UV, PerTraj_nFrames, OffsetInfo, npts, false, nCams, Tscale, eps, lamda, 1.0 - lamda, RealOverSfm, APLDCost, false, false);

	sprintf(Fname, "%s/Track3D", Path); makeDir(Fname);
	for (int pid = 0; pid < npts; pid++)
	{
		if (PerTraj_nFrames[pid] < 5 * nCams)
			continue;
		sprintf(Fname, "%s/Track3D/OptimizedRaw_Track_%.4d.txt", Path, pid);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int fid = 0; fid < PerCam_UV[camID*npts + pid].size(); fid++)
				fprintf(fp, "%.8f %.8f %.8f %.4f %d %d\n", PerCam_UV[camID*npts + pid][fid].pt3D.x, PerCam_UV[camID*npts + pid][fid].pt3D.y, PerCam_UV[camID*npts + pid][fid].pt3D.z,
					1.0*OffsetInfo[camID] / PerCam_UV[camID*npts + pid][fid].fps* Tscale + 1.0*PerCam_UV[camID*npts + pid][fid].frameID / PerCam_UV[camID*npts + pid][fid].fps*Tscale, SelectedCams[camID], PerCam_UV[camID*npts + pid][fid].frameID);
		fclose(fp);
	}

	printLOG("\nMaximum-likelihood trajectory triangulation:\n");
	MotionPrior_ML_Weighting(PerCam_UV, npts, nCams);
	MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, OffsetInfo, npts, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, false);

	sprintf(Fname, "%s/Track3D", Path); makeDir(Fname);
	for (int pid = 0; pid < npts; pid++)
	{
		if (PerTraj_nFrames[pid] < 5 * nCams)
			continue;
		sprintf(Fname, "%s/Track3D/ML_OptimizedRaw_Track_%.4d.txt", Path, pid);  FILE *fp = fopen(Fname, "w+");
		for (int camID = 0; camID < nCams; camID++)
			for (int fid = 0; fid < PerCam_UV[camID*npts + pid].size(); fid++)
				fprintf(fp, "%.8f %.8f %.8f %.4f %d %d\n", PerCam_UV[camID*npts + pid][fid].pt3D.x, PerCam_UV[camID*npts + pid][fid].pt3D.y, PerCam_UV[camID*npts + pid][fid].pt3D.z,
					1.0*OffsetInfo[camID] / PerCam_UV[camID*npts + pid][fid].fps* Tscale + 1.0*PerCam_UV[camID*npts + pid][fid].frameID / PerCam_UV[camID*npts + pid][fid].fps*Tscale, SelectedCams[camID], PerCam_UV[camID*npts + pid][fid].frameID);
		fclose(fp);
	}

	delete[]VideoInfo, delete[]PerCam_UV, delete[]OffsetInfo;
	for (int ii = 0; ii < (int)All3D.size(); ii++)
		delete[]PerTrackFrameID[ii], delete[]All3D[ii];

	return 0;
}

int EvaluateAllPairSTCost(char *Path, int nCams, int nTracks, int startF, int stopF, int SearchRange, double SearchStep, double lamda, double RealOverSfm, int motionPriorPower, double *InitialOffset)
{
	char Fname[512];
	int  totalPts;
	double cost;
	vector<int>Pair(2);
	vector<double> PairOffset(2), baseline;

	//Base on cameras' baseline
	VideoData *VideoIInfo = new VideoData[nCams];
	for (int ii = 0; ii < nCams; ii++)
		if (ReadVideoDataI(Path, VideoIInfo[ii], ii, startF, stopF) == 1)
			abort();

	printLOG("Motion prior sync:\n");
	sprintf(Fname, "%s/PairwiseCost.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < nCams - 1; ii++)
	{
		for (int jj = ii + 1; jj < nCams; jj++)
		{
			Pair[0] = ii, Pair[1] = jj;
			PairOffset[0] = InitialOffset[ii], PairOffset[1] = InitialOffset[jj];

			//Base on Motion prior
			cost = MotionPriorSyncBruteForce2DStereo(Path, Pair, startF, stopF, nTracks, PairOffset, -SearchRange, SearchRange, SearchStep, lamda, RealOverSfm, motionPriorPower, totalPts);

			for (int fid1 = startF; fid1 <= stopF; fid1++)
			{
				int fid2 = fid1 - (int)(InitialOffset[ii] - InitialOffset[jj] + 0.5);//approximately consider synced frame so that baseline estimation is reasonable
				if (fid2 < 0 || fid2>stopF)
					continue;
				if (VideoIInfo[ii].VideoInfo[fid1].valid && VideoIInfo[jj].VideoInfo[fid2].valid)
					baseline.push_back(Distance3D(VideoIInfo[ii].VideoInfo[fid1].camCenter, VideoIInfo[jj].VideoInfo[fid2].camCenter));
			}
			double avgBaseline = MeanArray(baseline);

			fprintf(fp, "%d %d %d %.16e %.16e %.16f\n", Pair[0], Pair[1], totalPts, avgBaseline, cost, PairOffset[1] - PairOffset[0]);
			baseline.clear();
		}
	}
	fclose(fp);

	printLOG("\n");
	return 0;
}
int EvaluateAllPairSTCostParallel(char *Path, int camID1, int camID2, int nCams, int nTracks, int startF, int stopF, int SearchRange, double SearchStep, double lamda, double RealOverSfm, int motionPriorPower)
{
	char Fname[512];
	int  totalPts;

	double *InitialOffset = new double[nCams];
	sprintf(Fname, "%s/FGeoSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		abort();
	}
	int cid; double offset;
	for (int ii = 0; ii < nCams; ii++)
	{
		fscanf(fp, "%d %lf ", &cid, &offset);
		InitialOffset[cid] = offset;
	}
	fclose(fp);

	//Base on cameras' baseline
	VideoData videoCam[2];
	if (ReadVideoDataI(Path, videoCam[0], camID1, startF, stopF) == 1)
		abort();
	if (ReadVideoDataI(Path, videoCam[1], camID2, startF, stopF) == 1)
		abort();

	vector<int>Pair;  Pair.push_back(camID1), Pair.push_back(camID2);
	vector<double> PairOffset;  PairOffset.push_back(InitialOffset[camID1]), PairOffset.push_back(InitialOffset[camID2]);

	//Base on Motion prior
	double cost = MotionPriorSyncBruteForce2DStereo(Path, Pair, startF, stopF, nTracks, PairOffset, -SearchRange, SearchRange, SearchStep, lamda, RealOverSfm, motionPriorPower, totalPts);

	//Weighted distance between 2 cameras
	vector<double> baseline;
	for (int fid1 = startF; fid1 <= stopF; fid1++)
	{
		int fid2 = fid1 - (int)(InitialOffset[camID1] - InitialOffset[camID2] + 0.5);
		if (fid2 < 0 || fid2>stopF)
			continue;
		if (videoCam[0].VideoInfo[fid1].valid && videoCam[1].VideoInfo[fid2].valid)
			baseline.push_back(Distance3D(videoCam[0].VideoInfo[fid1].camCenter, videoCam[1].VideoInfo[fid2].camCenter));
	}
	double avgBaseline = MeanArray(baseline);

	sprintf(Fname, "%s/PairwiseCost_%d_%.4d.txt", Path, camID1, camID2); fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %d %.3e %.8e %.3f\n", Pair[0], Pair[1], totalPts, avgBaseline, cost, PairOffset[1] - PairOffset[0]);
	fclose(fp);

	delete[]InitialOffset;
	return 0;
}
int DetermineCameraOrderingForGreedyDynamicSTBA(char *Path, char *PairwiseSyncFilename, int nCams, vector<int>&CameraOrder, vector<double> &OffsetInfo)
{
	//Offset info is corresponded to the camera order
	typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_distance_t, int>, boost::property < boost::edge_weight_t, double > > Graph;
	typedef boost::graph_traits < Graph >::edge_descriptor Edge;
	typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
	typedef std::pair < int, int >E;

	int v1, v2, nvalidPts;
	double baseline, TrajCost, offset;
	char Fname[512];
	int *nValidPts = new int[nCams*nCams];
	double *TimeOffset = new double[nCams*nCams],
		*BaseLine = new double[nCams*nCams],
		*Traj3dCost = new double[nCams*nCams];
	for (int ii = 0; ii < nCams*nCams; ii++)
		TimeOffset[ii] = 0, BaseLine[ii] = 0, Traj3dCost[ii] = 0;

	sprintf(Fname, "%s/%s.txt", Path, PairwiseSyncFilename);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d %d %lf %lf %lf ", &v1, &v2, &nvalidPts, &baseline, &TrajCost, &offset) != EOF)
		TimeOffset[v1 + v2 * nCams] = offset, TimeOffset[v2 + v1 * nCams] = offset,
		nValidPts[v1 + v2 * nCams] = nvalidPts, nValidPts[v2 + v1 * nCams] = nvalidPts,
		BaseLine[v1 + v2 * nCams] = baseline, BaseLine[v2 + v1 * nCams] = baseline,
		Traj3dCost[v1 + v2 * nCams] = TrajCost, Traj3dCost[v2 + v1 * nCams] = TrajCost;
	fclose(fp);

#ifdef ENABLE_DEBUG_FLAG
	sprintf(Fname, "%s/timeConstrantoffset.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nCams; kk++)
	{
		for (int ll = 0; ll < nCams; ll++)
			fprintf(fp, "%.4f ", TimeOffset[kk + ll * nCams]);
		fprintf(fp, "\n");
	}
	fclose(fp);
#endif

	//Form edges weight based on the consistency of the triplet
	int num_nodes = nCams, nedges = nCams * (nCams - 1) / 2;
	E *edges = new E[nedges];
	double *weightTable = new double[nCams*nCams];
	double *weights = new double[nedges];
	for (int ii = 0; ii < nCams*nCams; ii++)
		weightTable[ii] = 0;

	int count = 0;
	for (int kk = 0; kk < nCams - 1; kk++)
	{
		for (int ll = kk + 1; ll < nCams; ll++)
		{
			edges[count] = E(kk, ll), weights[count] = 0.0;
			//Consistency_score_kl = sum_j(Offset_kj+Offset_jl);
			for (int jj = 0; jj < nCams; jj++)
			{
				if (jj == ll || jj == kk)
					continue;
				if (jj >= ll) //kl = kj-lj
					weights[count] += abs(TimeOffset[kk + jj * nCams] - TimeOffset[ll + jj * nCams] - TimeOffset[kk + ll * nCams]);
				else if (jj <= kk) //kl = -jk + jl
					weights[count] += abs(-TimeOffset[jj + kk * nCams] + TimeOffset[jj + ll * nCams] - TimeOffset[kk + ll * nCams]);
				else //kl = kj+jl
					weights[count] += abs(TimeOffset[kk + jj * nCams] + TimeOffset[jj + ll * nCams] - TimeOffset[kk + ll * nCams]);
			}

			//Weight them by the # visible points along all trajectories and the average baseline between cameras
			weights[count] = weights[count] / (2.0*BaseLine[kk + ll * nCams] * pow(nValidPts[kk + ll * nCams], 2) + DBL_EPSILON);
			weightTable[kk + ll * nCams] = weights[count], weightTable[ll + kk * nCams] = weights[count];
			count++;
		}
	}

#ifdef ENABLE_DEBUG_FLAG
	sprintf(Fname, "%s/weightTable.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nCams; kk++)
	{
		for (int ll = 0; ll < nCams; ll++)
			fprintf(fp, "%.4e ", weightTable[kk + ll * nCams]);
		fprintf(fp, "\n");
	}
	fclose(fp);
#endif

	//Estimate incremental camera order by Kruskal MST 
	Graph g(edges, edges + sizeof(E)*nedges / sizeof(E), weights, num_nodes);
	boost::property_map<Graph, boost::edge_weight_t>::type weightmap = get(boost::edge_weight, g);
	std::vector < Edge > spanning_tree;

	boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

	cout << "Print the edges in the MST:" << endl;
	for (vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ei++)
		cout << source(*ei, g) << " <--> " << target(*ei, g) << " with weight of " << weightmap[*ei] << endl;


	//Store the ordering and subframe offset info. Note that source id is always smaller than target id
	int RootCamera = (int)spanning_tree[0].m_source;
	CameraOrder.push_back(RootCamera); OffsetInfo.push_back(0.0);
	for (int ii = 0; ii < spanning_tree.size(); ii++)
	{
		bool added = false;
		int cam1 = spanning_tree[ii].m_source, cam2 = spanning_tree[ii].m_target;
		for (int jj = 0; jj < (int)CameraOrder.size(); jj++)
		{
			if (CameraOrder[jj] == cam1)
			{
				added = true;
				break;
			}
		}

		if (!added)
		{
			CameraOrder.push_back(cam1);
			if (RootCamera < cam1)
				OffsetInfo.push_back(TimeOffset[RootCamera + cam1 * nCams]);
			else
				OffsetInfo.push_back(-TimeOffset[RootCamera + cam1 * nCams]);
		}

		added = false;
		for (int jj = 0; jj < (int)CameraOrder.size(); jj++)
		{
			if (CameraOrder[jj] == cam2)
			{
				added = true;
				break;
			}
		}

		if (!added)
		{
			CameraOrder.push_back(cam2);
			if (RootCamera < cam2)
				OffsetInfo.push_back(TimeOffset[RootCamera + cam2 * nCams]);
			else
				OffsetInfo.push_back(-TimeOffset[RootCamera + cam2 * nCams]);
		}
	}

	sprintf(Fname, "%s/MotionPriorSync.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nCams; kk++)
		fprintf(fp, "%d %.4e \n", CameraOrder[kk], OffsetInfo[kk]);
	fclose(fp);

	delete[]weights, delete[]nValidPts, delete[]BaseLine, delete[]Traj3dCost, delete[]weightTable, delete[]TimeOffset, delete[]edges;

	return 0;
}

int Combine3DStatisticFromRandomSampling(char *Path, int nCams, int ntracks)
{
	vector<int> availFileID;
	char Fname[512];
	for (int ii = 0; ii < 10000; ii++)
	{
		sprintf(Fname, "%s/ATrack_0_%.4d.txt", Path, ii); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			availFileID.push_back(ii);
			fclose(fp);
		}
	}

	int navailFiles = availFileID.size(), fileCount = 0, nframes;
	double x, y, z, t;
	vector<double> *timeStamp = new vector<double>[ntracks];
	vector<Point3d>*Traject3D_iTrial = new vector<Point3d>[navailFiles*ntracks];
	for (int ii = 0; ii < navailFiles; ii++)
	{
		for (int tid = 0; tid < ntracks; tid++)
		{
			sprintf(Fname, "%s/ATrack_%d_%.4d.txt", Path, tid, availFileID[ii]); FILE *fp = fopen(Fname, "r");
			nframes = 0;
			while (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &z, &t) != EOF)
			{
				if (fileCount == 0)
					timeStamp[tid].push_back(t);
				else	if (abs(timeStamp[tid][nframes] - t) > 0.1)
				{
					printLOG("Something wrong with the time stamp!");
					abort();
				}

				Traject3D_iTrial[fileCount*ntracks + tid].push_back(Point3d(x, y, z));
				nframes++;
			}
			fclose(fp);
		}
		fileCount++;
	}

	vector<Point3d> *P3D_Mean = new vector<Point3d>[ntracks], *P3D_STD = new vector<Point3d>[ntracks];
	for (int tid = 0; tid < ntracks; tid++)
		for (int fid = 0; fid < Traject3D_iTrial[tid].size(); fid++)
			P3D_Mean[tid].push_back(Point3d(0, 0, 0)), P3D_STD[tid].push_back(Point3d(0, 0, 0));

	for (int fileCount = 0; fileCount < navailFiles; fileCount++)
		for (int tid = 0; tid < ntracks; tid++)
			for (int fid = 0; fid < Traject3D_iTrial[fileCount*ntracks + tid].size(); fid++)
				P3D_Mean[tid][fid].x += Traject3D_iTrial[fileCount*ntracks + tid][fid].x / navailFiles,
				P3D_Mean[tid][fid].y += Traject3D_iTrial[fileCount*ntracks + tid][fid].y / navailFiles,
				P3D_Mean[tid][fid].z += Traject3D_iTrial[fileCount*ntracks + tid][fid].z / navailFiles;

	for (int fileCount = 0; fileCount < navailFiles; fileCount++)
		for (int tid = 0; tid < ntracks; tid++)
			for (int fid = 0; fid < Traject3D_iTrial[fileCount*ntracks + tid].size(); fid++)
				P3D_STD[tid][fid].x += pow(Traject3D_iTrial[fileCount*ntracks + tid][fid].x - P3D_Mean[tid][fid].x, 2) / (navailFiles - 1),
				P3D_STD[tid][fid].y += pow(Traject3D_iTrial[fileCount*ntracks + tid][fid].y - P3D_Mean[tid][fid].y, 2) / (navailFiles - 1),
				P3D_STD[tid][fid].z += pow(Traject3D_iTrial[fileCount*ntracks + tid][fid].z - P3D_Mean[tid][fid].z, 2) / (navailFiles - 1);


	for (int tid = 0; tid < ntracks; tid++)
	{
		sprintf(Fname, "%s/ATrackMSTD_%.4d.txt", Path, tid);  FILE *fp = fopen(Fname, "w+");
		for (int fid = 0; fid < P3D_Mean[tid].size(); fid++)
			fprintf(fp, "%.4f %.4f %.4f %.6f %.6f %.6f %.2f\n", P3D_Mean[tid][fid].x, P3D_Mean[tid][fid].y, P3D_Mean[tid][fid].z, sqrt(P3D_STD[tid][fid].x), sqrt(P3D_STD[tid][fid].y), sqrt(P3D_STD[tid][fid].z), timeStamp[tid][fid]);
		fclose(fp);
	}

	return 0;
}
int Generate3DUncertaintyFromRandomSampling(char *Path, vector<int> SelectedCams, vector<double> OffsetInfo, int startF, int stopF, int ntracks, int startSample, int nSamples, int motionPriorPower)
{
	char Fname[512]; FILE *fp = 0;
	const double Tscale = 1000000.0, eps = 1.0e-6, rate = 1.0, lamda = .3, RealOverSfm = 1.0;

	//Read calib info
	int nCams = (int)SelectedCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int camID = 0; camID < nCams; camID++)
		if (ReadVideoDataI(Path, VideoInfo[camID], SelectedCams[camID], startF, stopF) == 1)
			return 1;

	int frameID, id, npts;
	int nframes = max(MaxnFrames, stopF);

	double u, v;
	ImgPtEle ptEle;
	vector<ImgPtEle> *orgPerCam_UV = new vector<ImgPtEle>[nCams*ntracks];

	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < ntracks; trackID++)
			orgPerCam_UV[camID*ntracks + trackID].reserve(stopF - startF + 1);

		sprintf(Fname, "%s/Track2D/C_%.4d.txt", Path, SelectedCams[camID]); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			fscanf(fp, "%d %d ", &id, &npts);
			//if (id != trackID)
			//	printLOG("Problem at Point %d of Cam %d", id, camID);
			for (int pid = 0; pid < npts; pid++)
			{
				fscanf(fp, "%d %lf %lf ", &frameID, &u, &v);
				if (frameID < 0)
					continue;
				if (!VideoInfo[camID].VideoInfo[frameID].valid)
					continue; //camera not localized

				if (u > 0 && v > 0)
				{
					ptEle.pt2D.x = u, ptEle.pt2D.y = v, ptEle.frameID = frameID, ptEle.imHeight = VideoInfo[camID].VideoInfo[frameID].height, ptEle.imWidth = VideoInfo[camID].VideoInfo[frameID].width;
					LensCorrectionPoint(&ptEle.pt2D, VideoInfo[camID].VideoInfo[frameID].K, VideoInfo[camID].VideoInfo[frameID].distortion);
					orgPerCam_UV[camID*ntracks + id].push_back(ptEle);
				}
			}
		}
		fclose(fp);
	}

	//Sample 2d points with gaussian noise
	const double NoiseMag = 0;
	double startTime = omp_get_wtime();

	int numThreads = 1;
	for (int trialID = 0; trialID < nSamples; trialID++)
	{
		vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*ntracks];
		for (int camID = 0; camID < nCams; camID++)
		{
			for (int trackID = 0; trackID < ntracks; trackID++)
				PerCam_UV[camID*ntracks + trackID].reserve(orgPerCam_UV[camID*ntracks + trackID].size());

			for (int trackID = 0; trackID < ntracks; trackID++)
			{
				for (int pid = 0; pid < orgPerCam_UV[camID*ntracks + trackID].size(); pid++)
				{
					ptEle.pt2D.x = orgPerCam_UV[camID*ntracks + trackID][pid].pt2D.x + max(min(gaussian_noise(0, NoiseMag), 4.0*NoiseMag), -4.0*NoiseMag);
					ptEle.pt2D.y = orgPerCam_UV[camID*ntracks + trackID][pid].pt2D.y + max(min(gaussian_noise(0, NoiseMag), 4.0*NoiseMag), -4.0*NoiseMag);
					ptEle.frameID = orgPerCam_UV[camID*ntracks + trackID][pid].frameID;
					PerCam_UV[camID*ntracks + trackID].push_back(ptEle);
				}
			}
		}

		double P[12], AA[6], bb[2], ccT[3], dd[1];
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int frameID = 0; frameID < PerCam_UV[camID*ntracks + trackID].size(); frameID++)
				{
					int RealFrameID = PerCam_UV[camID*ntracks + trackID][frameID].frameID;

					for (int kk = 0; kk < 12; kk++)
					{
						P[kk] = VideoInfo[camID].VideoInfo[RealFrameID].P[kk];
						PerCam_UV[camID*ntracks + trackID][frameID].P[kk] = P[kk];
					}

					for (int kk = 0; kk < 9; kk++)
						PerCam_UV[camID*ntracks + trackID][frameID].K[kk] = VideoInfo[camID].VideoInfo[RealFrameID].K[kk],

						//Q, U
						AA[0] = P[0], AA[1] = P[1], AA[2] = P[2], bb[0] = P[3];
					AA[3] = P[4], AA[4] = P[5], AA[5] = P[6], bb[1] = P[7];
					ccT[0] = P[8], ccT[1] = P[9], ccT[2] = P[10], dd[0] = P[11];

					PerCam_UV[camID*ntracks + trackID][frameID].Q[0] = AA[0] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[0],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[1] = AA[1] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[1],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[2] = AA[2] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x*ccT[2];
					PerCam_UV[camID*ntracks + trackID][frameID].Q[3] = AA[3] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[0],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[4] = AA[4] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[1],
						PerCam_UV[camID*ntracks + trackID][frameID].Q[5] = AA[5] - PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y*ccT[2];
					PerCam_UV[camID*ntracks + trackID][frameID].u[0] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.x - bb[0],
						PerCam_UV[camID*ntracks + trackID][frameID].u[1] = dd[0] * PerCam_UV[camID*ntracks + trackID][frameID].pt2D.y - bb[1];

					double stdA = 100.0;//Interestingly, Ceres does not work if all the input are the same-->need some random perturbation. 
					PerCam_UV[camID*ntracks + trackID][frameID].pt3D = Point3d(gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA), gaussian_noise(0.0, stdA));
				}
			}
		}

		//Initialize data for optim
		vector<int> PerTraj_nFrames;
		vector<int *> PerTrackFrameID(ntracks);
		vector<double*> All3D(ntracks);
		int ntimeinstances, maxntimeinstances = 0;
		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			ntimeinstances = 0;
			for (int camID = 0; camID < nCams; camID++)
				ntimeinstances += (int)PerCam_UV[camID*ntracks + trackID].size();

			if (maxntimeinstances < ntimeinstances)
				maxntimeinstances = ntimeinstances;

			PerTrackFrameID[trackID] = new int[ntimeinstances];
			All3D[trackID] = new double[3 * ntimeinstances];
		}

		double currentOffset[MaxnCams], APLDCost[4];
		for (int ii = 0; ii < nCams; ii++)
			currentOffset[ii] = OffsetInfo[ii];

		MotionPrior_Optim_SpatialStructure_Algebraic(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, motionPriorPower, Tscale, eps, lamda, RealOverSfm, APLDCost, false, false);
		//MotionPrior_ML_Weighting(PerCam_UV, ntracks, nCams);
		//MotionPrior_Optim_SpatialStructure_Geometric(Path, All3D, PerCam_UV, PerTraj_nFrames, currentOffset, ntracks, false, nCams, Tscale, ialpha, eps, lamda, APLDCost, false, false);

		for (int trackID = 0; trackID < ntracks; trackID++)
		{
			sprintf(Fname, "%s/ATrack_%d_%.4d.txt", Path, trackID, trialID + startSample);  FILE *fp = fopen(Fname, "w+");
			for (int camID = 0; camID < nCams; camID++)
				for (int fid = 0; fid < PerCam_UV[camID*ntracks + trackID].size(); fid++)
					fprintf(fp, "%.4f %.4f %.4f %.2f %d %d\n", PerCam_UV[camID*ntracks + trackID][fid].pt3D.x, PerCam_UV[camID*ntracks + trackID][fid].pt3D.y, PerCam_UV[camID*ntracks + trackID][fid].pt3D.z,
						1.0*currentOffset[camID] / PerCam_UV[camID*npts + trackID][fid].fps*Tscale + 1.0*PerCam_UV[camID*ntracks + trackID][fid].frameID / PerCam_UV[camID*npts + trackID][fid].fps*Tscale*rate, camID, PerCam_UV[camID*ntracks + trackID][fid].frameID);
			fclose(fp);
		}

		printLOG("%.1f%% ... TR: %.2fs\n", 100.0*trialID / nSamples, (omp_get_wtime() - startTime) / (trialID + 0.000001)*(nSamples - trialID));
		delete[]PerCam_UV;
	}

	return 0;
}

struct CeresResamplingSpline {
	CeresResamplingSpline(double *XIn, double *PIn, double *PhiDataIn, double *PhiPriorIn, double *TimeStampDatain, double *TimeStampPriorIn, Point2d *pt2DIn, double lamda, int nData, int nResamples, int nCoeffs) :lamda(lamda), nData(nData), nResamples(nResamples), nCoeffs(nCoeffs)
	{
		X = XIn, P = PIn, PhiData = PhiDataIn, PhiPrior = PhiPriorIn, TimeStampData = TimeStampDatain, TimeStampPrior = TimeStampPriorIn, pt2D = pt2DIn;
	}

	template <typename T>    bool operator()(T const* const* C, T* residuals)     const
	{
		//cost = lamda*(PBC-u)^2 + (1-lamda)*Prior
		T x, y, z, numX, numY, denum, sqrtlamda1 = sqrt((T)lamda), lamda2 = (T)1.0 - lamda, lamda3 = (T)2.0;

		//Projection cost 
		for (int ii = 0; ii < nData; ii++)
		{
			x = (T) 0.0, y = (T) 0.0, z = (T)0.0;
			for (int jj = 0; jj < nCoeffs; jj++)
			{
				if (PhiData[jj + ii * nCoeffs] < 0.00001)
					continue;
				x += (T)PhiData[jj + ii * nCoeffs] * C[0][jj],
					y += (T)PhiData[jj + ii * nCoeffs] * C[0][jj + nCoeffs],
					z += (T)PhiData[jj + ii * nCoeffs] * C[0][jj + 2 * nCoeffs];
			}


			numX = (T)P[12 * ii] * x + (T)P[12 * ii + 1] * y + (T)P[12 * ii + 2] * z + (T)P[12 * ii + 3];
			numY = (T)P[12 * ii + 4] * x + (T)P[12 * ii + 5] * y + (T)P[12 * ii + 6] * z + (T)P[12 * ii + 7];
			denum = (T)P[12 * ii + 8] * x + (T)P[12 * ii + 9] * y + (T)P[12 * ii + 10] * z + (T)P[12 * ii + 11];

			residuals[2 * ii] = (T)sqrtlamda1*(numX / denum - (T)pt2D[ii].x);
			residuals[2 * ii + 1] = (T)sqrtlamda1*(numY / denum - (T)pt2D[ii].y);
		}

		//Motion cost 
		/*T xo, yo, zo;
		for (int ii = 0; ii < nResamples; ii++)
		{
		x = (T) 0.0, y = (T) 0.0, z = (T)0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
		if (PhiPrior[jj + ii*nCoeffs] < 0.00001)
		continue;
		x += (T)PhiPrior[jj + ii*nCoeffs] * C[0][jj],
		y += (T)PhiPrior[jj + ii*nCoeffs] * C[0][jj + nCoeffs],
		z += (T)PhiPrior[jj + ii*nCoeffs] * C[0][jj + 2 * nCoeffs];
		}

		if (ii > 0)
		{
		T dx = x - xo, dy = y - yo, dz = z - zo;
		T error = (T)lamda2*(dx*dx + dy*dy + dz*dz) / (T)(TimeStampPrior[ii] - TimeStampPrior[ii - 1]);
		residuals[2 * nData + ii - 1] = sqrt(max((T)(1.0e-16), error));
		}
		xo = x, yo = y, zo = z;
		}*/

		return true;
	}

	int nData, nResamples, nCoeffs;
	double lamda;
	double  *X, *P, *PhiData, *PhiPrior, *TimeStampData, *TimeStampPrior;
	Point2d *pt2D;
};
void ResamplingOf3DTrajectorySpline(vector<ImgPtEle> &Traj3D, bool non_monotonicDescent, double Break_Step, double Resample_Step, double lamda, bool silent)
{
	const int SplineOrder = 4;

	double earliest = Traj3D[0].timeStamp, latest = Traj3D.back().timeStamp;
	int nBreaks = (int)(ceil((ceil(latest) - floor(earliest)) / Break_Step)) + 1, nCoeffs = nBreaks + 2, nData = (int)Traj3D.size(), nptsPrior = (int)((latest - earliest) / Resample_Step);

	double*BreakPts = new double[nBreaks], *X = new double[nData * 3];
	double *PhiData = new double[nData*nCoeffs], *PhiPrior = new double[nptsPrior*nCoeffs], *C = new double[3 * nCoeffs];
	double *PmatData = new double[nData * 12], *TimeStampData = new double[nData], *TimeStampPrior = new double[nptsPrior];
	Point2d *pt2dData = new Point2d[nData];

	for (int ii = 0; ii < nBreaks; ii++)
		BreakPts[ii] = floor(earliest) + Break_Step * ii;

	for (int ii = 0; ii < nData; ii++)
	{
		for (int jj = 0; jj < 12; jj++)
			PmatData[12 * ii + jj] = Traj3D[ii].P[jj];

		TimeStampData[ii] = Traj3D[ii].timeStamp;

		pt2dData[ii] = Traj3D[ii].pt2D;
		X[ii] = Traj3D[ii].pt3D.x, X[ii + nData] = Traj3D[ii].pt3D.y, X[ii + 2 * nData] = Traj3D[ii].pt3D.z;
	}

	for (int ii = 0; ii < nptsPrior; ii++)
		TimeStampPrior[ii] = floor(earliest) + Resample_Step * ii;

	//Find and delete breakpts with no data in between (this makes the basis matrix to be ill-condition)
	vector<int>IdToDel;
	for (int ii = 0; ii < nBreaks - 1; ii++)
	{
		bool found = false;
		for (int jj = 0; jj < nData; jj++)
		{
			if ((BreakPts[ii] < TimeStampData[jj] && TimeStampData[jj] < BreakPts[ii + 1]))
			{
				found = true;
				break;
			}
		}
		if (!found)
			IdToDel.push_back(ii + 1);
	}
	if ((int)IdToDel.size() > 0)
	{
		vector<double> tBreakPts;
		for (int ii = 0; ii < nBreaks; ii++)
			tBreakPts.push_back(BreakPts[ii]);

		for (int ii = (int)IdToDel.size() - 1; ii >= 0; ii--)
			tBreakPts.erase(tBreakPts.begin() + IdToDel[ii]);

		nBreaks = (int)tBreakPts.size(), nCoeffs = nBreaks + 2;
		for (int ii = 0; ii < nBreaks; ii++)
			BreakPts[ii] = tBreakPts[ii];
	}

	//Generate Spline Basis
	//GenerateResamplingSplineBasisWithBreakPts(PhiData, TimeStampData, BreakPts, nData, nBreaks, SplineOrder);
	//GenerateResamplingSplineBasisWithBreakPts(PhiPrior, TimeStampPrior, BreakPts, nptsPrior, nBreaks, SplineOrder);
	BSplineGetAllBasis(PhiData, TimeStampData, BreakPts, nData, nBreaks, SplineOrder);
	BSplineGetAllBasis(PhiPrior, TimeStampPrior, BreakPts, nptsPrior, nBreaks, SplineOrder);


	//Initialize basis coefficients: X_data = Phi_data*C
	for (int jj = 0; jj < 3; jj++)
	{
		LS_Solution_Double(PhiData, X + jj * nData, nData, nCoeffs);
		for (int ii = 0; ii < nCoeffs; ii++)
			C[ii + jj * nCoeffs] = X[ii + jj * nData];
	}

#ifdef ENABLE_DEBUG_FLAG
	double MotionPriorCost = 0, ProjCost = 0;
	for (int ii = 0; ii < nData - 1; ii++)
	{
		double xyz1[] = { Traj3D[ii].pt3D.x, Traj3D[ii].pt3D.y, Traj3D[ii].pt3D.z };
		double xyz2[] = { Traj3D[ii + 1].pt3D.x, Traj3D[ii + 1].pt3D.y, Traj3D[ii + 1].pt3D.z };
		double costi = LeastActionError(xyz1, xyz2, Traj3D[ii].timeStamp, Traj3D[ii + 1].timeStamp, 1.0e-6, 2);
		MotionPriorCost += costi;
	}

	for (int ii = 0; ii < nData; ii++)
	{
		int  camID = Traj3D[ii].viewID, frameID = Traj3D[ii].frameID;
		Point2d p2d = Traj3D[ii].pt2D;
		Point3d p3d = Traj3D[ii].pt3D;
		double	err = PinholeReprojectionErrorSimpleDebug(Traj3D[ii].P, Traj3D[ii].pt3D, Traj3D[ii].pt2D);
		ProjCost += err * err;
	}
	printLOG("(Before Spline: Motion cost, projection cost): %.4e %.4e\n", MotionPriorCost, sqrt(ProjCost / nData));

	ProjCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiData[jj + ii * nCoeffs] < 1e-6)
				continue;
			x += PhiData[jj + ii * nCoeffs] * C[jj],
				y += PhiData[jj + ii * nCoeffs] * C[jj + nCoeffs],
				z += PhiData[jj + ii * nCoeffs] * C[jj + 2 * nCoeffs];
		}

		double numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		double numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		double denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		double errX = (numX / denum - pt2dData[ii].x);
		double errY = (numY / denum - pt2dData[ii].y);
		ProjCost += errX * errX + errY * errY;
	}

	double xo, yo, zo;
	for (int ii = 0; ii < nptsPrior; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiPrior[jj + ii * nCoeffs] < 1e-6)
				continue;
			x += PhiPrior[jj + ii * nCoeffs] * C[jj],
				y += PhiPrior[jj + ii * nCoeffs] * C[jj + nCoeffs],
				z += PhiPrior[jj + ii * nCoeffs] * C[jj + 2 * nCoeffs];
		}

		if (ii > 0)
			MotionPriorCost += (pow(x - xo, 2) + pow(y - yo, 2) + pow(z - zo, 2)) / (TimeStampPrior[ii] - TimeStampPrior[ii - 1]);
		xo = x, yo = y, zo = z;

	}
	printLOG("(Spline: Motion cost, projection cost, Totalcost): %.4e %.4e %.4e\n", MotionPriorCost, sqrt(ProjCost / nData), lamda*ProjCost + (1.0 - lamda)*MotionPriorCost);
#endif

	for (int ii = 0; ii < nData; ii++)
		X[ii] = Traj3D[ii].pt3D.x, X[ii + nData] = Traj3D[ii].pt3D.y, X[ii + 2 * nData] = Traj3D[ii].pt3D.z;

	//Run ceres optimization
	ceres::Problem problem;

	vector<double*> parameter_blocks;
	parameter_blocks.push_back(C);
	ceres::DynamicAutoDiffCostFunction<CeresResamplingSpline, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction<CeresResamplingSpline, 4>(new CeresResamplingSpline(X, PmatData, PhiData, PhiPrior, TimeStampData, TimeStampPrior, pt2dData, lamda, nData, nptsPrior, nCoeffs));
	//ceres::DynamicNumericDiffCostFunction<CeresResamplingSpline, ceres::CENTRAL> *cost_function = new ceres::DynamicNumericDiffCostFunction<CeresResamplingSpline, ceres::CENTRAL>(new CeresResamplingSpline(X, PmatData, PhiData, PhiPrior, TimeStampData, TimeStampPrior, pt2dData, lamda, nData, nptsPrior, nCoeffs));
	cost_function->AddParameterBlock(3 * nCoeffs);
	//cost_function->SetNumResiduals(2 * nData + nptsPrior - 1);
	cost_function->SetNumResiduals(2 * nData);
	problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = silent ? false : true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
#pragma omp critical
	if (silent)
		std::cout << summary.BriefReport() << "\n";
	else
		std::cout << summary.FullReport() << "\n";

	//Final curve
	mat_mul(PhiData, C, X, nData, nCoeffs, 1);
	mat_mul(PhiData, C + nCoeffs, X + nData, nData, nCoeffs, 1);
	mat_mul(PhiData, C + 2 * nCoeffs, X + 2 * nData, nData, nCoeffs, 1);

	for (int ii = 0; ii < nData; ii++)
		Traj3D[ii].pt3D.x = X[ii], Traj3D[ii].pt3D.y = X[ii + nData], Traj3D[ii].pt3D.z = X[ii + 2 * nData];

#ifdef ENABLE_DEBUG_FLAG 
	{double ProjCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiData[jj + ii * nCoeffs] < 1e-6)
				continue;
			x += PhiData[jj + ii * nCoeffs] * C[jj],
				y += PhiData[jj + ii * nCoeffs] * C[jj + nCoeffs],
				z += PhiData[jj + ii * nCoeffs] * C[jj + 2 * nCoeffs];
		}

		double numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		double numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		double denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		double errX = (numX / denum - pt2dData[ii].x);
		double errY = (numY / denum - pt2dData[ii].y);
		ProjCost += errX * errX + errY * errY;
	}

	double xo, yo, zo, MotionPriorCost = 0.0;
	for (int ii = 0; ii < nptsPrior; ii++)
	{
		double x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nCoeffs; jj++)
		{
			if (PhiPrior[jj + ii * nCoeffs] < 1e-6)
				continue;
			x += PhiPrior[jj + ii * nCoeffs] * C[jj],
				y += PhiPrior[jj + ii * nCoeffs] * C[jj + nCoeffs],
				z += PhiPrior[jj + ii * nCoeffs] * C[jj + 2 * nCoeffs];
		}

		if (ii > 0)
			MotionPriorCost += (pow(x - xo, 2) + pow(y - yo, 2) + pow(z - zo, 2)) / (TimeStampPrior[ii] - TimeStampPrior[ii - 1]);
		xo = x, yo = y, zo = z;

	}
	printLOG("(After Spline: Motion cost, projection cost, Totalcost): %.4e %.4e %.4e\n", MotionPriorCost, sqrt(ProjCost / nData), lamda*ProjCost + (1.0 - lamda)*MotionPriorCost);
	}
#endif

	delete[]X, delete[]BreakPts, delete[]PhiData, delete[]PhiPrior, delete[]C;
	delete[]PmatData, delete[]TimeStampData, delete[]pt2dData;

	return;
}
struct CeresResamplingDCT {
	CeresResamplingDCT(double *XIn, double *PIn, double *iBDataIn, double *sqrtWeightIn, Point2d *pt2DIn, double lamda1, double lamda2, int nData, int nResamples) :lamda1(lamda1), lamda2(lamda2), nData(nData), nResamples(nResamples)
	{
		X = XIn, P = PIn, iBData = iBDataIn, sqrtWeight = sqrtWeightIn, pt2D = pt2DIn;
	}

	template <typename T>    bool operator()(T const* const* C, T* residuals)     const
	{
		//cost = lamda1*(PiBC-u)^2 + lamda2*Prior
		T x, y, z, numX, numY, denum, sqrtlamda1 = sqrt((T)lamda1), sqrtlamda2 = sqrt((T)lamda2);

		//Projection cost : lamda1*(PiBC-u)^2
		for (int ii = 0; ii < nData; ii++)
		{
			//X = iB*C
			x = (T) 0.0, y = (T) 0.0, z = (T)0.0;
			for (int jj = 0; jj < nResamples; jj++)
			{
				x += (T)iBData[jj + ii * nResamples] * C[0][jj],
					y += (T)iBData[jj + ii * nResamples] * C[0][jj + nResamples],
					z += (T)iBData[jj + ii * nResamples] * C[0][jj + 2 * nResamples];
			}

			numX = (T)P[12 * ii] * x + (T)P[12 * ii + 1] * y + (T)P[12 * ii + 2] * z + (T)P[12 * ii + 3];
			numY = (T)P[12 * ii + 4] * x + (T)P[12 * ii + 5] * y + (T)P[12 * ii + 6] * z + (T)P[12 * ii + 7];
			denum = (T)P[12 * ii + 8] * x + (T)P[12 * ii + 9] * y + (T)P[12 * ii + 10] * z + (T)P[12 * ii + 11];

			residuals[2 * ii] = (T)sqrtlamda1*(numX / denum - (T)pt2D[ii].x);
			residuals[2 * ii + 1] = (T)sqrtlamda1*(numY / denum - (T)pt2D[ii].y);
		}

		//Motion prior in DCT form: CT*W*C
		for (int ii = 0; ii < nResamples; ii++)
		{
			T lamdaW = sqrtlamda2 * sqrtWeight[ii];
			residuals[2 * nData + 3 * ii] = lamdaW * C[0][ii];
			residuals[2 * nData + 3 * ii + 1] = lamdaW * C[0][ii + nResamples];
			residuals[2 * nData + 3 * ii + 2] = lamdaW * C[0][ii + 2 * nResamples];
		}

		return true;
	}

	int nData, nResamples;
	double  *X, *P, *iBData, *sqrtWeight, lamda1, lamda2;
	Point2d *pt2D;
};
void ResamplingOf3DTrajectoryDCT(vector<ImgPtEle> &Traj3D, int PriorOrder, bool non_monotonicDescent, double Resample_Step, double lamda1, double lamda2, bool silent)
{
	double earliest = Traj3D[0].timeStamp, latest = Traj3D.back().timeStamp;
	int nData = (int)Traj3D.size(), nResamples = (int)((latest - earliest) / Resample_Step);

	double *X = new double[nData * 3], *C = new double[3 * nResamples];
	double *iBData = new double[nData*nResamples], *BResampled = new double[nResamples*nResamples], *sqrtWeight = new double[nResamples];
	double *PmatData = new double[nData * 12], *TimeStampData = new double[nData];
	Point2d *pt2dData = new Point2d[nData];

	for (int ii = 0; ii < nData; ii++)
	{
		for (int jj = 0; jj < 12; jj++)
			PmatData[12 * ii + jj] = Traj3D[ii].P[jj];

		pt2dData[ii] = Traj3D[ii].pt2D;
		X[ii] = Traj3D[ii].pt3D.x, X[ii + nData] = Traj3D[ii].pt3D.y, X[ii + 2 * nData] = Traj3D[ii].pt3D.z;
		TimeStampData[ii] = (Traj3D[ii].timeStamp - earliest) / (latest - earliest)*(nResamples - 1);//Normalize to [0, n-1] range
	}

	//Generate DCT Basis
	GenerateDCTBasis(nResamples, BResampled, sqrtWeight);
	for (int ii = 0; ii < nResamples; ii++)
		if (PriorOrder == 1)
			sqrtWeight[ii] = sqrt(-sqrtWeight[ii]); //(1) using precomputed sqrt is better for ceres' squaring residual square nature; (2) weigths are negative, but that does not matter for ctwc optim.
		else
			sqrtWeight[ii] = -sqrtWeight[ii]; //ctw.^2c-->ceres: residuals = c*W;

	for (int ii = 0; ii < nData; ii++)
		GenerateiDCTBasis(iBData + ii * nResamples, nResamples, TimeStampData[ii]);

	//Initialize basis coefficients: iBd(:, 1:activeBasis)*C =  X_d
	int nactiveBasis = max(nResamples / 20, 10);
	Map < Matrix < double, Dynamic, Dynamic, RowMajor > > eiBData(iBData, nData, nResamples);
	MatrixXd etiBData = eiBData.block(0, 0, nData, nactiveBasis);
	JacobiSVD<MatrixXd> etiP_svd(etiBData, ComputeThinU | ComputeThinV);
	for (int ii = 0; ii < 3; ii++)
	{
		for (int jj = nactiveBasis; jj < nResamples; jj++)
			C[jj + nResamples * ii] = 0.0; //set coeffs outside active basis to 0

		Map<VectorXd> eX(X + nData * ii, nData);
		Map<VectorXd> eC(C + nResamples * ii, nactiveBasis);
		eC = etiP_svd.solve(eX);
	}

	//Before optim
	double x, y, z, numX, numY, denum;

	//Projection cost : (PiBC-u)^2
	double projectionCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		//X = iB*C
		x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nResamples; jj++)
			x += iBData[jj + ii * nResamples] * C[jj], y += iBData[jj + ii * nResamples] * C[jj + nResamples], z += iBData[jj + ii * nResamples] * C[jj + 2 * nResamples];

		numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		projectionCost += (pow(numX / denum - pt2dData[ii].x, 2) + pow(numY / denum - pt2dData[ii].y, 2));
	}

	//Motion prior in DCT form: CT*W*C
	double MotionPrior = 0.0;
	for (int ii = 0; ii < nResamples; ii++)
		MotionPrior += pow(sqrtWeight[ii], 2)*(pow(C[ii], 2) + pow(C[ii + nResamples], 2) + pow(C[ii + 2 * nResamples], 2));

#pragma omp critical
	printLOG("\nDCT before: Motion cost: %.4e, projection cost: %.4e, Totalcost: %.4e", MotionPrior / nData, sqrt(projectionCost / nData), lamda1*projectionCost + lamda2 * MotionPrior);

	//Run ceres optimization
	ceres::Problem problem;

	vector<double*> parameter_blocks;
	parameter_blocks.push_back(C);
	ceres::DynamicAutoDiffCostFunction<CeresResamplingDCT, 4> *cost_function = new ceres::DynamicAutoDiffCostFunction<CeresResamplingDCT, 4>(
		new CeresResamplingDCT(X, PmatData, iBData, sqrtWeight, pt2dData, lamda1, lamda2, nData, nResamples));
	cost_function->AddParameterBlock(3 * nResamples);
	cost_function->SetNumResiduals(2 * nData + 3 * nResamples);
	problem.AddResidualBlock(cost_function, NULL, parameter_blocks);

	ceres::Solver::Options options;
	options.max_num_iterations = 30;
	options.num_threads = omp_get_max_threads();
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = silent ? false : true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	//After optim
	//Projection cost : (PiBC-u)^2
	projectionCost = 0.0;
	for (int ii = 0; ii < nData; ii++)
	{
		//X = iB*C
		x = 0.0, y = 0.0, z = 0.0;
		for (int jj = 0; jj < nResamples; jj++)
			x += iBData[jj + ii * nResamples] * C[jj], y += iBData[jj + ii * nResamples] * C[jj + nResamples], z += iBData[jj + ii * nResamples] * C[jj + 2 * nResamples];

		numX = PmatData[12 * ii] * x + PmatData[12 * ii + 1] * y + PmatData[12 * ii + 2] * z + PmatData[12 * ii + 3];
		numY = PmatData[12 * ii + 4] * x + PmatData[12 * ii + 5] * y + PmatData[12 * ii + 6] * z + PmatData[12 * ii + 7];
		denum = PmatData[12 * ii + 8] * x + PmatData[12 * ii + 9] * y + PmatData[12 * ii + 10] * z + PmatData[12 * ii + 11];

		projectionCost += (pow(numX / denum - pt2dData[ii].x, 2) + pow(numY / denum - pt2dData[ii].y, 2));
	}

	//Motion prior in DCT form: CT*W*C
	MotionPrior = 0.0;
	for (int ii = 0; ii < nResamples; ii++)
		MotionPrior += pow(sqrtWeight[ii], 2)*(pow(C[ii], 2) + pow(C[ii + nResamples], 2) + pow(C[ii + 2 * nResamples], 2));
#pragma omp critical
	printLOG("\nDCT after: Motion cost: %.4e, projection cost: %.4e, Totalcost: %.4e", MotionPrior / nData, sqrt(projectionCost / nData), lamda1*projectionCost + lamda2 * MotionPrior);

#pragma omp critical
	{
		if (silent)
			std::cout << endl << summary.BriefReport();
		else
			std::cout << endl << summary.FullReport();
	}

	//Final curve: iB*C =  X
	for (int ii = 0; ii < 3; ii++)
	{
		Map<VectorXd> eX(X + nData * ii, nData);
		Map<VectorXd> eC(C + nResamples * ii, nResamples);
		eX = eiBData * eC;
	}

	for (int ii = 0; ii < nData; ii++)
		Traj3D[ii].pt3D.x = X[ii], Traj3D[ii].pt3D.y = X[ii + nData], Traj3D[ii].pt3D.z = X[ii + 2 * nData];

	delete[]X, delete[]iBData, delete[]BResampled, delete[]C;
	delete[]PmatData, delete[]TimeStampData, delete[]pt2dData;

	return;
}

