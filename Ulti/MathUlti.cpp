#include "MathUlti.h"
#include "DataIO.h"
#include "MiscAlgo.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void inline RayPlaneIntersection(double *rayDir, double *root, double *plane, double *intersect)
{
	//compute intersection of the ref camera and plane: nX+d = 0, x = at+x0, y = bt+y0, z = ct+z0; --> nx*(at+x0) + ny*(bt+y0)+nz*(ct+z0) +d = 0;

	double num_0 = plane[0] * root[0] + plane[1] * root[1] + plane[2] * root[2] + plane[3];
	double denum_0 = plane[0] * rayDir[0] + plane[1] * rayDir[1] + plane[2] * rayDir[2];
	double t_0 = -num_0 / denum_0;

	intersect[0] = rayDir[0] * t_0 + root[0],
		intersect[1] = rayDir[1] * t_0 + root[1],
		intersect[2] = rayDir[2] * t_0 + root[2];

	return;
}
double inline Area3DTriangle(double *A, double *B, double *C)
{
	double AB[3] = { A[0] - B[0], A[1] - B[1], A[2] - B[2] };
	double AC[3] = { A[0] - C[0], A[1] - C[1], A[2] - C[2] };

	double ABxAC[3];
	cross_product(AB, AC, ABxAC);

	double area = .5*sqrt(ABxAC[0] * ABxAC[0] + ABxAC[1] * ABxAC[1] + ABxAC[2] * ABxAC[2]);

	return area;
}
bool PointInTriangle(Point2f X1, Point2f X2, Point2f X3, Point2f X)
{
	//http://totologic.blogspot.fr/2014/01/accurate-point-in-triangle-test.html
	float x1 = X1.x, y1 = X1.y, x2 = X2.x, y2 = X2.y, x3 = X3.x, y3 = X3.y, x = X.x, y = X.y;
	float EPSILON = 1e-3;

	float xMin = min(x1, min(x2, x3)) - EPSILON;
	float xMax = max(x1, max(x2, x3)) + EPSILON;
	if (x < xMin || xMax < x)
		return false;

	float yMin = min(y1, min(y2, y3)) - EPSILON;
	float yMax = max(y1, max(y2, y3)) + EPSILON;
	if (y < yMin || yMax < y)
		return false;

	float denum = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));

	float a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denum;
	if (a < 0 || a>1)
		return false;

	float b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denum;
	if (b < 0 || b>1)
		return false;

	float c = 1.0f - a - b;
	if (c < 0 || c>1)
		return false;

	return true; //inside
}
void dec2bin(int dec, int*bin, int num_bin)
{
	bool stop = false;
	int ii, digit = 0;
	int temp[32];

	while (!stop)
	{
		temp[digit] = dec % 2;
		dec /= 2;
		digit++;
		if (dec == 0)
			stop = true;
	}

	if (digit > num_bin)
		cout << '\a';

	for (ii = 0; ii < num_bin - digit; ii++)
		bin[ii] = 0;
	for (ii = digit - 1; ii >= 0; ii--)
		bin[num_bin - ii - 1] = temp[ii];

	return;
}

double nChoosek(int n, int k)
{
	if (n < 0 || k < 0)
		return 0.0;
	if (n < k)
		return 0.0;  // special case
	if (n == k)
		return 1.0;

	int iMax;
	double delta;

	if (k < n - k) // eg: Choose(100,3)
	{
		delta = 1.0*(n - k);
		iMax = k;
	}
	else         // eg: Choose(100,97)
	{
		delta = 1.0*k;
		iMax = n - k;
	}

	double res = delta + 1.0;
	for (int i = 2; i <= iMax; i++)
		res = res * (delta + i) / i;

	return res;
}
double nPermutek(int n, int k)
{
	double res = n;
	for (int ii = 0; ii < k; ii++)
		res *= 1.0*(n - ii);
	return res;
}
int MyFtoI(double W)
{
	if (W >= 0.0)
		return (int)(W + 0.5);
	else
		return (int)(W - 0.5);

	return 0;
}
bool IsNumber(double x)
{
	// This looks like it should always be true, but it's false if x is a NaN.
	return (x == x);
}
bool IsFiniteNumber(double x)
{
	return (x <= DBL_MAX && x >= -DBL_MAX);
}

double UniformNoise(double High, double Low)
{
	double noise = 1.0*rand() / RAND_MAX;
	return (High - Low)*noise + Low;
}
double gaussian_noise(double mean, double std)
{
	double u1 = 0.0, u2 = 0.0;
	while (abs(u1) < DBL_EPSILON || abs(u2) < DBL_EPSILON) //avoid 0.0 case since log(0) = inf
	{
		u1 = 1.0 * rand() / RAND_MAX;
		u2 = 1.0 * rand() / RAND_MAX;
	}

	double normal_noise = sqrt(-2.0 * log(u1)) * cos(2.0 * Pi * u2);
	return mean + std * normal_noise;
}


double Distance3D(double *X, double * Y)
{
	Point3d Dif(X[0] - Y[0], X[1] - Y[1], X[2] - Y[2]);
	return sqrt(Dif.x*Dif.x + Dif.y * Dif.y + Dif.z *Dif.z);
}
double MinDistanceTwoLines(double *P0, double *u, double *Q0, double *v, double &s, double &t)
{
	//http://geomalgorithms.com/a07-_distance.html
	double w0[] = { P0[0] - Q0[0], P0[1] - Q0[1], P0[2] - Q0[2] };
	double a = dotProduct(u, u), b = dotProduct(u, v), c = dotProduct(v, v), d = dotProduct(u, w0), e = dotProduct(v, w0);
	double denum = a*c - b*b;

	double distance = 0.0;
	if (denum < 0.00001)//Near parallel line
	{
		s = 0.0, t = d / b;
		double Q[] = { Q0[0] + t*v[0], Q0[1] + t*v[1], Q0[2] + t*v[2] };
		distance = sqrt(pow(P0[0] - Q[0], 2) + pow(P0[1] - Q[1], 2) + pow(P0[2] - Q[2], 2));
	}
	else
	{
		s = (b*e - c*d) / denum, t = (a*e - b*d) / denum;
		double P[] = { P0[0] + s*u[0], P0[1] + s*u[1], P0[2] + s*u[2] };
		double Q[] = { Q0[0] + t*v[0], Q0[1] + t*v[1], Q0[2] + t*v[2] };
		distance = sqrt(pow(P[0] - Q[0], 2) + pow(P[1] - Q[1], 2) + pow(P[2] - Q[2], 2));
	}

	return distance;
}
Point3d ProductPoint3d(Point3d X, Point3d Y)
{
	return Point3d(X.x*Y.x, X.y*Y.y, X.z*Y.z);
}
Point3d DividePoint3d(Point3d X, Point3d Y)
{
	return Point3d(X.x / Y.x, X.y / Y.y, X.z / Y.z);
}
double L1norm(vector<double>A)
{
	double res = 0.0;
	for (int ii = 0; ii < A.size(); ii++)
		res += abs(A[ii]);
	return res;
}
double L2norm(double *A, int dim)
{
	double res = 0.0;
	for (int ii = 0; ii < dim; ii++)
		res += A[ii] * A[ii];
	return sqrt(res);
}
void normalize(double *x, int dim)
{
	double tt = 0;
	for (int ii = 0; ii < dim; ii++)
		tt += x[ii] * x[ii];
	tt = sqrt(tt);
	if (tt < FLT_EPSILON)
		return;
	for (int ii = 0; ii < dim; ii++)
		x[ii] = x[ii] / tt;
	return;
}

float MeanArray(float *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return (float)(mean / length);
}
double MeanArray(double *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return mean / length;
}
Point3d MeanArray(Point3d *data, int length)
{
	Point3d centroid(0,0,0);
	for (int ii = 0; ii < length; ii++)
		centroid.x += data[ii].x, centroid.y += data[ii].y, centroid.z += data[ii].z;
	centroid.x /= length, centroid.y /= length, centroid.z /= length;
	return centroid;
}
double VarianceArray(double *data, int length, double mean)
{
	if (mean == NULL)
		mean = MeanArray(data, length);

	double var = 0.0;
	for (int ii = 0; ii < length; ii++)
		var += pow(data[ii] - mean, 2);
	return var / (length - 1);
}
Point3d VarianceArray(Point3d *data, int length)
{
	Point3d mean = MeanArray(data, length);

	Point3d var(0,0,0);
	for (int ii = 0; ii < length; ii++)
		var.x += pow(data[ii].x - mean.x, 2), var.y += pow(data[ii].y - mean.y, 2), var.z += pow(data[ii].z - mean.z, 2);
	var.x = var.x / (length - 1), var.y = var.y / (length - 1), var.z = var.z / (length - 1);
	return var;
}
double MeanArray(vector<double>&data)
{
	double mean = 0.0;
	for (int ii = 0; ii < data.size(); ii++)
		mean += data[ii];
	return mean / data.size();
}
double VarianceArray(vector<double>&data, double mean)
{
	if (mean == NULL)
		mean = MeanArray(data);

	double var = 0.0;
	for (int ii = 0; ii < data.size(); ii++)
		var += pow(data[ii] - mean, 2);
	return var / (data.size() - 1);
}
double MedianArray(vector<double> &data)
{
	int size = (int)data.size();
	sort(data.begin(), data.end());

	if (size % 2 == 0)
		return (data[size / 2 - 1] + data[size / 2]) / 2;
	else
		return data[size / 2];
}
double dotProduct(double *x, double *y, int dim)
{
	double res = 0.0;
	for (int ii = 0; ii < dim; ii++)
		res += x[ii] * y[ii];
	return res;
}
double dotProduct(float *x, float *y, int dim)
{
	double res = 0.0;
	for (int ii = 0; ii < dim; ii++)
		res += x[ii] * y[ii];
	return res;
}
double norm_dot_product(double *x, double *y, int dim)
{
	double nx = 0.0, ny = 0.0, dxy = 0.0;
	for (int ii = 0; ii < dim; ii++)
	{
		nx += x[ii] * x[ii];
		ny += y[ii] * y[ii];
		dxy += x[ii] * y[ii];
	}
	double radian = dxy / sqrt(nx*ny);

	return radian;
}
void cross_product(double *x, double *y, double *xy)
{
	xy[0] = x[1] * y[2] - x[2] * y[1];
	xy[1] = x[2] * y[0] - x[0] * y[2];
	xy[2] = x[0] * y[1] - x[1] * y[0];

	return;
}
void conv(float *A, int lenA, float *B, int lenB, float *C)
{
	int nconv;
	int i, j, i1;
	double tmp;

	nconv = lenA + lenB - 1;
	for (i = 0; i < nconv; i++)
	{
		i1 = i;
		tmp = 0.0;
		for (j = 0; j < lenB; j++)
		{
			if (i1 >= 0 && i1 < lenA)
				tmp = tmp + (A[i1] * B[j]);

			i1 = i1 - 1;
			C[i] = (float)tmp;
		}
	}

	return;
}
void conv(double *A, int lenA, double *B, int lenB, double *C)
{
	int nconv;
	int i, j, i1;
	double tmp;

	nconv = lenA + lenB - 1;
	for (i = 0; i < nconv; i++)
	{
		i1 = i;
		tmp = 0.0;
		for (j = 0; j < lenB; j++)
		{
			if (i1 >= 0 && i1 < lenA)
				tmp = tmp + (A[i1] * B[j]);

			i1 = i1 - 1;
			C[i] = tmp;
		}
	}

	return;
}
void ZNCC1D(float *A, const int dimA, float *B, const int dimB, float *Result, float *nB)
{
	//Matlab normxcorr2
	const int sdimA = dimA - 1, dimnB = 2 * (dimA - 1) + dimB, dimRes = dimB + dimA - 1;
	bool createMem = false;
	if (nB == NULL)
	{
		createMem = true;
		nB = new float[dimnB];
	}

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int ii = 0; ii < sdimA; ii++)
		nB[ii] = 0;
#pragma omp parallel for
	for (int ii = sdimA; ii < sdimA + dimB; ii++)
		nB[ii] = B[ii - sdimA];
#pragma omp parallel for
	for (int ii = sdimA + dimB; ii < dimnB; ii++)
		nB[ii] = 0;

	Mat ma(1, dimA, CV_32F, A);
	Mat mb(1, dimnB, CV_32F, nB);

	Mat result(1, dimRes, CV_32F);
	matchTemplate(mb, ma, result, 5);

	for (int ii = 0; ii < dimRes; ii++)
		Result[ii] = result.at<float>(ii);

	if (createMem)
		delete[]nB;

	return;
}
void ZNCC1D(double *A, int Asize, double *B, int Bsize, double *Result)
{
	//Matlab normxcorr2
	int ii, jj;

	double A2 = 0.0, meanA = MeanArray(A, Asize);
	double *ZNA = new double[Asize], *ZNB = new double[Asize];
	for (ii = 0; ii < Asize; ii++)
	{
		ZNA[ii] = A[ii] - meanA;
		A2 += pow(ZNA[ii], 2);
	}

	for (ii = 0; ii < Asize; ii++)
	{
		double meanB = 0.0, allZeros = 0;
		for (jj = 0; jj <= ii; jj++)
		{
			meanB += B[jj];
			allZeros += abs(B[jj]);
		}
		if (allZeros < 1e-6)
			Result[ii] = 0.0;
		else
		{
			meanB = meanB / Asize;

			for (jj = 0; jj < Asize - ii - 1; jj++)
				ZNB[jj] = 0.0 - meanB;
			for (jj = 0; jj <= ii; jj++)
				ZNB[Asize - ii - 1 + jj] = B[jj] - meanB;

			double B2 = 0, AB = 0.0;
			for (jj = 0; jj < Asize; jj++)
				AB += ZNA[jj] * ZNB[jj], B2 += pow(ZNB[jj], 2);

			double zncc = AB / sqrt(A2*B2);
			Result[ii] = zncc;
		}
	}

	for (ii = 1; ii < Bsize - Asize + 1; ii++)
	{
		double meanB = 0.0, allZeros = 0;
		for (jj = ii; jj < ii + Asize; jj++)
		{
			meanB += B[jj];
			allZeros += abs(B[jj]);
		}
		if (allZeros < 1.0e-6)
			Result[ii - 1 + Asize] = 0.0;
		else
		{
			meanB = meanB / Asize;

			for (jj = 0; jj < Asize; jj++)
				ZNB[jj] = B[jj + ii] - meanB;

			double B2 = 0, AB = 0.0;
			for (jj = 0; jj < Asize; jj++)
				AB += ZNA[jj] * ZNB[jj], B2 += pow(ZNB[jj], 2);

			double zncc = AB / sqrt(A2*B2);
			Result[ii - 1 + Asize] = zncc;
		}
	}

	for (ii = 1; ii < Asize; ii++)
	{
		double meanB = 0.0, allZeros = 0;
		for (jj = Asize - ii; jj > 0; jj--)
		{
			meanB += B[Bsize - jj];
			allZeros += abs(B[Bsize - jj]);
		}
		if (allZeros < 1e-6)
			Result[ii - 1 + Bsize] = 0.0;
		else
		{
			meanB = meanB / Asize;

			for (jj = Asize - ii; jj > 0; jj--)
				ZNB[Asize - ii - jj] = B[Bsize - jj] - meanB;
			for (jj = Asize - ii; jj < Asize; jj++)
				ZNB[jj] = 0.0 - meanB;

			double B2 = 0, AB = 0.0;
			for (jj = 0; jj < Asize; jj++)
				AB += ZNA[jj] * ZNB[jj], B2 += pow(ZNB[jj], 2);

			double zncc = AB / sqrt(A2*B2);
			Result[ii - 1 + Bsize] = zncc;
		}
	}

	delete[]ZNA, delete[]ZNB;

	return;
}
void XCORR1D(float *s, const int sdim, float *b, const int bdim, float *res)
{
	Mat ms(1, bdim, CV_32F, Scalar(0.0));
	Mat mb(1, bdim * 3 - 2, CV_32F, Scalar(0.0));
	Mat result(1, 2 * bdim - 1, CV_32F);

#pragma omp parallel for
	for (int ii = 0; ii < sdim; ii++)
		ms.at<float>(ii) = s[ii];

#pragma omp parallel for
	for (int ii = 0; ii < bdim; ii++)
		mb.at<float>(ii + bdim - 1) = b[ii];

	matchTemplate(mb, ms, result, CV_TM_CCORR);

#pragma omp parallel for
	for (int ii = 0; ii < 2 * bdim - 1; ii++)
		res[ii] = mb.at<float>(ii);

	return;
}
void mat_invert(double* mat, double* imat, int dims)
{
	if (dims == 2)
	{
		double a0 = mat[0], a1 = mat[1], a2 = mat[2], a3 = mat[3];
		double det = a0*a3 - a1*a2;
		if (abs(det) < 1e-9)
			printLOG("Caution. Matrix is ill-condition\n");

		imat[0] = a3 / det, imat[1] = -a1 / det;
		imat[2] = -a2 / det, imat[3] = a0 / det;
	}
	if (dims == 3)
	{
		// only work for 3x3
		double a = mat[0], b = mat[1], c = mat[2], d = mat[3], e = mat[4], f = mat[5], g = mat[6], h = mat[7], k = mat[8];
		double A = e*k - f*h, B = c*h - b*k, C = b*f - c*e;
		double D = f*g - d*k, E = a*k - c*g, F = c*d - a*f;
		double G = d*h - e*g, H = b*g - a*h, K = a*e - b*d;
		double DET = a*A + b*D + c*G;
		imat[0] = A / DET, imat[1] = B / DET, imat[2] = C / DET;
		imat[3] = D / DET, imat[4] = E / DET, imat[5] = F / DET,
			imat[6] = G / DET, imat[7] = H / DET, imat[8] = K / DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_64FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for (int jj = 0; jj < dims; jj++)
			for (int ii = 0; ii < dims; ii++)
				imat[ii + jj*dims] = outMat.at<double>(jj, ii);
	}

	return;
}
void mat_invert(float* mat, float* imat, int dims)
{
	if (dims == 3)
	{
		// only work for 3x3
		float a = mat[0], b = mat[1], c = mat[2], d = mat[3], e = mat[4], f = mat[5], g = mat[6], h = mat[7], k = mat[8];
		float A = e*k - f*h, B = c*h - b*k, C = b*f - c*e;
		float D = f*g - d*k, E = a*k - c*g, F = c*d - a*f;
		float G = d*h - e*g, H = b*g - a*h, K = a*e - b*d;
		float DET = a*A + b*D + c*G;
		imat[0] = A / DET, imat[1] = B / DET, imat[2] = C / DET;
		imat[3] = D / DET, imat[4] = E / DET, imat[5] = F / DET,
			imat[6] = G / DET, imat[7] = H / DET, imat[8] = K / DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_32FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for (int jj = 0; jj < dims; jj++)
			for (int ii = 0; ii < dims; ii++)
				imat[ii + jj*dims] = outMat.at<float>(jj, ii);
	}

	return;
}
void mat_mul(float *aa, float *bb, float *out, int rowa, int col_row, int colb)
{
	int ii, jj, kk;
	for (ii = 0; ii < rowa*colb; ii++)
		out[ii] = 0;

	for (ii = 0; ii < rowa; ii++)
	{
		for (jj = 0; jj < colb; jj++)
		{
			for (kk = 0; kk < col_row; kk++)
				out[ii*colb + jj] += aa[ii*col_row + kk] * bb[kk*colb + jj];
		}
	}

	return;
}
void mat_mul(double *aa, double *bb, double *out, int rowa, int col_row, int colb)
{
	int ii, jj, kk;
	for (ii = 0; ii < rowa*colb; ii++)
		out[ii] = 0;

	for (ii = 0; ii < rowa; ii++)
	{
		for (jj = 0; jj < colb; jj++)
		{
			for (kk = 0; kk < col_row; kk++)
				out[ii*colb + jj] += aa[ii*col_row + kk] * bb[kk*colb + jj];
		}
	}

	return;
}
void mat_add(double *aa, double *bb, double* cc, int row, int col, double scale_a, double scale_b)
{
	int ii, jj;

	for (ii = 0; ii < row; ii++)
		for (jj = 0; jj < col; jj++)
			cc[ii*col + jj] = scale_a*aa[ii*col + jj] + scale_b*bb[ii*col + jj];

	return;
}
void mat_subtract(double *aa, double *bb, double* cc, int row, int col, double scale_a, double scale_b)
{
	int ii, jj;

	for (ii = 0; ii < row; ii++)
		for (jj = 0; jj < col; jj++)
			cc[ii*col + jj] = scale_a*aa[ii*col + jj] - scale_b*bb[ii*col + jj];

	return;
}
void mat_transpose(double *in, double *out, int row_in, int col_in)
{
	int ii, jj;
	for (jj = 0; jj < row_in; jj++)
		for (ii = 0; ii < col_in; ii++)
			out[ii*row_in + jj] = in[jj*col_in + ii];
	return;
}
void mat_mul_symetric(double *A, double *B, int row, int column)
{
	for (int I = 0; I < row; I++)
	{
		for (int J = 0; J < row; J++)
		{
			if (J < I)
				continue;

			B[I*row + J] = 0.0;
			for (int K = 0; K < column; K++)
				B[I*row + J] += A[I*column + K] * A[J*column + K];
		}
	}

	return;
}
void mat_add_symetric(double *A, double * B, double *C, int row, int column)
{
	for (int I = 0; I < row; I++)
	{
		for (int J = 0; J < column; J++)
		{
			if (J < I)
				continue;

			C[I*column + J] = A[I*column + J] + B[I*column + J];
		}
	}

	return;
}
void mat_completeSym(double *mat, int size, bool upper)
{
	if (upper)
	{
		for (int jj = 0; jj < size; jj++)
			for (int ii = jj; ii < size; ii++)
				mat[jj + ii*size] = mat[ii + jj*size];
	}
	else
	{
		for (int jj = 0; jj < size; jj++)
			for (int ii = jj; ii < size; ii++)
				mat[ii + jj*size] = mat[jj + ii*size];
	}
	return;
}
void RescaleMat(double *mat, double &orgmin, double &orgmax, double nmin, double nmax, int length)
{
	orgmin = 9e9, orgmax = -9e9;
	for (int ii = 0; ii < length; ii++)
	{
		if (mat[ii] > orgmax)
			orgmax = mat[ii];
		if (mat[ii] < orgmin)
			orgmin = mat[ii];
	}

	for (int ii = 0; ii < length; ii++)
		mat[ii] = (mat[ii] - orgmin) / (orgmax - orgmin)*(nmax - nmin) + nmin;
	return;
}
void LS_Solution_Double(double *lpA, double *lpB, int m, int n)
{
	if (m == n)
	{
		QR_Solution_Double(lpA, lpB, n, n);
		return;
	}

	int i, j, k, n2 = n*n;
	double *A = new double[n2];
	double *B = new double[n];

	for (i = 0; i < n2; i++)
		*(A + i) = 0.0;
	for (i = 0; i < n; i++)
		*(B + i) = 0.0;

	for (k = 0; k < m; k++)
	{
		for (j = 0; j < n; j++)
		{
			for (i = 0; i < n; i++)
				*(A + j*n + i) += (*(lpA + k*n + i))*(*(lpA + k*n + j));
			*(B + j) += (*(lpB + k))*(*(lpA + k*n + j));
		}
	}

	QR_Solution_Double(A, B, n, n);

	for (i = 0; i < n; i++)
		*(lpB + i) = *(B + i);

	delete[]B;
	delete[]A;
	return;
}
void QR_Solution_Double(double *lpA, double *lpB, int m, int n)
{
	if (m > 3000)
	{
		LS_Solution_Double(lpA, lpB, m, n);
		return;
	}

	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.QR_Solution(lpA, lpB, m, n);
	return;
}

void Quick_Sort_Int(int * A, int *B, int low, int high)
{
	m_TemplateClass_1<int> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Float(float * A, int *B, int low, int high)
{
	m_TemplateClass_1<float> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Double(double * A, int *B, int low, int high)
{
	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}

double SimpsonThreeEightIntegration(double *y, double step, int npts)
{
	if (npts < 4)
	{
		printLOG("Not enough supprintg points (4) for this integration method. Abort()");
		abort();
	}

	if (npts == 4)
		return  step*(3.0 / 8.0*y[0] + 9.0 / 8.0*y[1] + 9.0 / 8.0*y[2] + 3.0 / 8.0*y[3]);


	double result = 3.0 / 8.0*y[0] + 7.0 / 6.0*y[1] + 23.0 / 24.0*y[2] + 23.0 / 24.0*y[npts - 3] + 7.0 / 6.0*y[npts - 2] + 3.0 / 8.0*y[npts - 1];
	for (int ii = 3; ii < npts - 3; ii++)
		result += y[ii];

	return step*result;
}
bool in_polygon(double u, double v, Point2d *vertex, int num_vertex)
{
	int ii;
	bool position;
	double pi = 3.1415926535897932384626433832795;

	for (ii = 0; ii < num_vertex; ii++)
	{
		if (abs(u - vertex[ii].x) < 0.01 && abs(v - vertex[ii].y) < 0.01)
			return 1;
	}
	double dot = (u - vertex[0].x)*(u - vertex[num_vertex - 1].x) + (v - vertex[0].y)*(v - vertex[num_vertex - 1].y);
	double square1 = (u - vertex[0].x)*(u - vertex[0].x) + (v - vertex[0].y)*(v - vertex[0].y);
	double square2 = (u - vertex[num_vertex - 1].x)*(u - vertex[num_vertex - 1].x) + (v - vertex[num_vertex - 1].y)*(v - vertex[num_vertex - 1].y);
	double angle = acos(dot / sqrt(square1*square2));

	for (ii = 0; ii < num_vertex - 1; ii++)
	{
		dot = (u - vertex[ii].x)*(u - vertex[ii + 1].x) + (v - vertex[ii].y)*(v - vertex[ii + 1].y);
		square1 = (u - vertex[ii].x)*(u - vertex[ii].x) + (v - vertex[ii].y)*(v - vertex[ii].y);
		square2 = (u - vertex[ii + 1].x)*(u - vertex[ii + 1].x) + (v - vertex[ii + 1].y)*(v - vertex[ii + 1].y);

		angle += acos(dot / sqrt(square1*square2));
	}

	angle = angle * 180 / pi;
	if (fabs(angle - 360) <= 2.0)
		position = 1;
	else
		position = 0;

	return position;
}

void SetIntrinisc(CameraData &CamInfo, double *Intrinsic)
{
	for (int ii = 0; ii < 5; ii++)
		CamInfo.intrinsic[ii] = Intrinsic[ii];
	GetKFromIntrinsic(CamInfo);
	return;
}
void GetIntrinsicFromK(CameraData *AllViewsParas, vector<int> AvailViews)
{
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
		AllViewsParas[viewID].intrinsic[0] = AllViewsParas[viewID].K[0];
		AllViewsParas[viewID].intrinsic[1] = AllViewsParas[viewID].K[4];
		AllViewsParas[viewID].intrinsic[2] = AllViewsParas[viewID].K[1];
		AllViewsParas[viewID].intrinsic[3] = AllViewsParas[viewID].K[2];
		AllViewsParas[viewID].intrinsic[4] = AllViewsParas[viewID].K[5];
	}
	return;
}
void GetIntrinsicFromK(CameraData *AllViewsParas, int nviews)
{
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		AllViewsParas[viewID].intrinsic[0] = AllViewsParas[viewID].K[0];
		AllViewsParas[viewID].intrinsic[1] = AllViewsParas[viewID].K[4];
		AllViewsParas[viewID].intrinsic[2] = AllViewsParas[viewID].K[1];
		AllViewsParas[viewID].intrinsic[3] = AllViewsParas[viewID].K[2];
		AllViewsParas[viewID].intrinsic[4] = AllViewsParas[viewID].K[5];
	}
	return;
}
void GetKFromIntrinsic(CameraData *AllViewsParas, vector<int> AvailViews)
{
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
		AllViewsParas[viewID].K[0] = AllViewsParas[viewID].intrinsic[0];
		AllViewsParas[viewID].K[4] = AllViewsParas[viewID].intrinsic[1];
		AllViewsParas[viewID].K[1] = AllViewsParas[viewID].intrinsic[2];
		AllViewsParas[viewID].K[2] = AllViewsParas[viewID].intrinsic[3];
		AllViewsParas[viewID].K[5] = AllViewsParas[viewID].intrinsic[4];
	}
	return;
}
void GetKFromIntrinsic(CameraData *AllViewsParas, int nviews)
{
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		AllViewsParas[viewID].K[0] = AllViewsParas[viewID].intrinsic[0];
		AllViewsParas[viewID].K[4] = AllViewsParas[viewID].intrinsic[1];
		AllViewsParas[viewID].K[1] = AllViewsParas[viewID].intrinsic[2];
		AllViewsParas[viewID].K[2] = AllViewsParas[viewID].intrinsic[3];
		AllViewsParas[viewID].K[5] = AllViewsParas[viewID].intrinsic[4];
	}
	return;
}
void GetIntrinsicFromK(CameraData &camera)
{
	camera.intrinsic[0] = camera.K[0];
	camera.intrinsic[1] = camera.K[4];
	camera.intrinsic[2] = camera.K[1];
	camera.intrinsic[3] = camera.K[2];
	camera.intrinsic[4] = camera.K[5];
	return;
}
void GetKFromIntrinsic(CameraData &camera)
{
	camera.K[0] = camera.intrinsic[0], camera.K[1] = camera.intrinsic[2], camera.K[2] = camera.intrinsic[3];
	camera.K[3] = 0.0, camera.K[4] = camera.intrinsic[1], camera.K[5] = camera.intrinsic[4];
	camera.K[6] = 0.0, camera.K[7] = 0.0, camera.K[8] = 1.0;
	return;
}

void getTwistFromRT(double *R, double *T, double *twist)
{
	//OpenCV code to handle log map for SO(3)
	Map < Matrix < double, 3, 3, RowMajor > > matR(R); //matR is referenced to R;
	JacobiSVD<MatrixXd> svd(matR, ComputeFullU | ComputeFullV);
	//Matrix3d S = svd.singularValues().asDiagonal();
	matR = svd.matrixU()*svd.matrixV().transpose();//Project R to SO(3)

	double rx = R[7] - R[5], ry = R[2] - R[6], rz = R[3] - R[1];
	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R[0] + R[4] + R[8] - 1)*0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;
	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;
		if (c > 0)
			rx = ry = rz = 0.0;
		else
		{
			t = (R[0] + 1)*0.5, rx = sqrt(MAX(t, 0.));
			t = (R[4] + 1)*0.5, ry = sqrt(MAX(t, 0.))*(R[1] < 0 ? -1. : 1.);
			t = (R[8] + 1)*0.5, rz = sqrt(MAX(t, 0.))*(R[2] < 0 ? -1. : 1.);
			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta, ry *= theta, rz *= theta;
		}
	}
	else
	{
		double vth = 1.0 / (2.0 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	twist[3] = rx, twist[4] = ry, twist[5] = rz;

	//Compute V
	double theta2 = theta* theta;
	double wx[9] = { 0.0, -rz, ry, rz, 0.0, -rx, -ry, rx, 0.0 };
	double wx2[9]; mat_mul(wx, wx, wx2, 3, 3, 3);

	double V[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	if (theta < 1.0e-9)
		twist[0] = T[0], twist[1] = T[1], twist[2] = T[2];
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			V[ii] += B*wx[ii] + C*wx2[ii];
	}

	//solve Vt = T;
	Map < Matrix < double, 3, 3, RowMajor > > matV(V);
	Map<Vector3d> matT(T), matt(twist);
	matt = matV.lu().solve(matT);

	return;
}
void getRTFromTwist(double *twist, double *R, double *T)
{
	double t[3] = { twist[0], twist[1], twist[2] };
	double w[3] = { twist[3], twist[4], twist[5] };

	double theta = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), theta2 = theta* theta;
	double wx[9] = { 0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0 };
	double wx2[9]; mat_mul(wx, wx, wx2, 3, 3, 3);

	R[0] = 1.0, R[1] = 0.0, R[2] = 0.0;
	R[3] = 0.0, R[4] = 1.0, R[5] = 0.0;
	R[6] = 0.0, R[7] = 0.0, R[8] = 1.0;

	double V[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	if (theta < 1.0e-9)
		T[0] = t[0], T[1] = t[1], T[2] = t[2]; //Rotation is idenity
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			R[ii] += A*wx[ii] + B*wx2[ii];

		for (int ii = 0; ii < 9; ii++)
			V[ii] += B*wx[ii] + C*wx2[ii];
		mat_mul(V, t, T, 3, 3, 1);
	}

	return;
}
void getrFromR(double *R, double *r)
{
	//Project R to SO(3)
	Map < Matrix < double, 3, 3, RowMajor > > matR(R); //matR is referenced to R;
	JacobiSVD<MatrixXd> svd(matR, ComputeFullU | ComputeFullV);
	matR = svd.matrixU()*svd.matrixV().transpose();

	double rx = R[7] - R[5], ry = R[2] - R[6], rz = R[3] - R[1];
	double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
	double c = (R[0] + R[4] + R[8] - 1)*0.5;
	c = c > 1. ? 1. : c < -1. ? -1. : c;
	double theta = acos(c);

	if (s < 1e-5)
	{
		double t;
		if (c > 0)
			rx = ry = rz = 0.0;
		else
		{
			t = (R[0] + 1)*0.5, rx = sqrt(MAX(t, 0.));
			t = (R[4] + 1)*0.5, ry = sqrt(MAX(t, 0.))*(R[1] < 0 ? -1. : 1.);
			t = (R[8] + 1)*0.5, rz = sqrt(MAX(t, 0.))*(R[2] < 0 ? -1. : 1.);
			if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0))
				rz = -rz;
			theta /= sqrt(rx*rx + ry*ry + rz*rz);
			rx *= theta, ry *= theta, rz *= theta;
		}
	}
	else
	{
		double vth = 1.0 / (2.0 * s);
		vth *= theta;
		rx *= vth; ry *= vth; rz *= vth;
	}
	r[0] = rx, r[1] = ry, r[2] = rz;

	return;
}
void getRfromr(double *r, double *R)
{
	/*Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
	rvec.at<double>(jj) = r[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
	R[jj] = Rmat.at<double>(jj);*/

	double theta = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), theta2 = theta* theta;
	double rx[9] = { 0.0, -r[2], r[1], r[2], 0.0, -r[0], -r[1], r[0], 0.0 };
	double rx2[9]; mat_mul(rx, rx, rx2, 3, 3, 3);

	R[0] = 1.0, R[1] = 0.0, R[2] = 0.0;
	R[3] = 0.0, R[4] = 1.0, R[5] = 0.0;
	R[6] = 0.0, R[7] = 0.0, R[8] = 1.0;

	if (theta < 1.0e-9)
		return;
	else
	{
		double A = sin(theta) / theta, B = (1.0 - cos(theta)) / theta2, C = (1.0 - A) / theta2;
		for (int ii = 0; ii < 9; ii++)
			R[ii] += A*rx[ii] + B*rx2[ii];
	}

	return;
}

void NormalizeQuaternion(double *quat)
{
	Map < Vector4d > equat(quat, 4);
	const double norm = equat.norm();
	if (norm == 0)
	{
		// We do not just use (1, 0, 0, 0) because that is a constant and when used for automatic differentiation that would lead to a zero derivative.
		quat[0] = 1;
		return;
	}
	else
	{
		const double inv_norm = 1.0 / norm;
		equat = inv_norm * equat;
		return;
	}
}
void Rotation2Quaternion(double *R, double *q)
{
	double r11 = R[0], r12 = R[1], r13 = R[2];
	double r21 = R[3], r22 = R[4], r23 = R[5];
	double r31 = R[6], r32 = R[7], r33 = R[8];

	double qw = sqrt(abs(1.0 + r11 + r22 + r33)) / 2;
	double qx, qy, qz;
	if (qw > 1e-6)
	{
		qx = (r32 - r23) / 4 / qw;
		qy = (r13 - r31) / 4 / qw;
		qz = (r21 - r12) / 4 / qw;
	}
	else
	{
		double d = sqrt((r12*r12*r13*r13 + r12*r12*r23*r23 + r13*r13*r23*r23));
		qx = r12*r13 / d;
		qy = r12*r23 / d;
		qz = r13*r23 / d;
	}

	q[0] = qw, q[1] = qx, q[2] = qy, q[3] = qz;

	normalize(q, 4);
}
void Quaternion2Rotation(double *q, double *R)
{
	normalize(q, 4);

	double qw = q[0], qx = q[1], qy = q[2], qz = q[3];

	R[0] = 1.0 - 2 * qy*qy - 2 * qz*qz;
	R[1] = 2 * qx*qy - 2 * qz*qw;
	R[2] = 2 * qx*qz + 2 * qy*qw;

	R[3] = 2 * qx*qy + 2 * qz*qw;
	R[4] = 1.0 - 2 * qx*qx - 2 * qz*qz;
	R[5] = 2 * qz*qy - 2 * qx*qw;

	R[6] = 2 * qx*qz - 2 * qy*qw;
	R[7] = 2 * qy*qz + 2 * qx*qw;
	R[8] = 1.0 - 2 * qx*qx - 2 * qy*qy;
}

void GetrtFromRT(CameraData *AllViewsParas, vector<int> AvailViews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
		for (int jj = 0; jj < 9; jj++)
			R.at<double>(jj) = AllViewsParas[viewID].R[jj];

		Rodrigues(R, r);

		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].rt[jj] = r.at<double>(jj), AllViewsParas[viewID].rt[3 + jj] = AllViewsParas[viewID].T[jj];
	}
}
void GetrtFromRT(CameraData *AllViewsParas, int nviews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int viewID = 0; viewID < nviews; viewID++)
	{
		for (int jj = 0; jj < 9; jj++)
			R.at<double>(jj) = AllViewsParas[viewID].R[jj];

		Rodrigues(R, r);

		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].rt[jj] = r.at<double>(jj), AllViewsParas[viewID].rt[3 + jj] = AllViewsParas[viewID].T[jj];
	}
}
void GetrtFromRT(CameraData &cam)
{
	Mat Rmat(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int jj = 0; jj < 9; jj++)
		Rmat.at<double>(jj) = cam.R[jj];

	Rodrigues(Rmat, r);

	for (int jj = 0; jj < 3; jj++)
		cam.rt[jj] = r.at<double>(jj), cam.rt[3 + jj] = cam.T[jj];

	return;
}
void GetrtFromRT(double *rt, double *R, double *T)
{
	Mat Rmat(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int jj = 0; jj < 9; jj++)
		Rmat.at<double>(jj) = R[jj];

	Rodrigues(Rmat, r);

	for (int jj = 0; jj < 3; jj++)
		rt[jj] = r.at<double>(jj), rt[3 + jj] = T[jj];

	return;
}
void GetRTFromrt(CameraData &camera)
{
	cv::Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = camera.rt[jj];

	cv::Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
		camera.R[jj] = Rmat.at<double>(jj);
	for (int jj = 0; jj < 3; jj++)
		camera.T[jj] = camera.rt[jj + 3];

	return;
}
void GetRTFromrt(double *rt, double *R, double *T)
{
	Mat Rmat(3, 3, CV_64F), rvec(3, 1, CV_64F);
	for (int jj = 0; jj < 3; jj++)
		rvec.at<double>(jj) = rt[jj];

	Rodrigues(rvec, Rmat);

	for (int jj = 0; jj < 9; jj++)
		R[jj] = Rmat.at<double>(jj);
	for (int jj = 0; jj < 3; jj++)
		T[jj] = rt[jj + 3];

	return;
}
void GetRTFromrt(CameraData *AllViewsParas, vector<int> AvailViews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews[ii];
		for (int jj = 0; jj < 3; jj++)
			r.at<double>(jj) = AllViewsParas[viewID].rt[jj];

		Rodrigues(r, R);

		for (int jj = 0; jj < 9; jj++)
			AllViewsParas[viewID].R[jj] = R.at<double>(jj);
		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].T[jj] = AllViewsParas[viewID].rt[jj + 3];
	}

	return;
}
void GetRTFromrt(CameraData *AllViewsParas, int nviews)
{
	Mat R(3, 3, CV_64F), r(3, 1, CV_64F);

	for (int viewID = 0; viewID < nviews; viewID++)
	{
		for (int jj = 0; jj < 3; jj++)
			r.at<double>(jj) = AllViewsParas[viewID].rt[jj];

		Rodrigues(r, R);

		for (int jj = 0; jj < 9; jj++)
			AllViewsParas[viewID].R[jj] = R.at<double>(jj);
		for (int jj = 0; jj < 3; jj++)
			AllViewsParas[viewID].T[jj] = AllViewsParas[viewID].rt[jj + 3];
	}

	return;
}

void GetTfromC(CameraData &camInfo)
{
	double T[3];

	mat_mul(camInfo.R, camInfo.camCenter, T, 3, 3, 1);
	camInfo.T[0] = -T[0], camInfo.T[1] = -T[1], camInfo.T[2] = -T[2];
	camInfo.rt[3] = -T[0], camInfo.rt[4] = -T[1], camInfo.rt[5] = -T[2];

	return;
}
void GetTfromC(double *R, double *C, double *T)
{
	mat_mul(R, C, T, 3, 3, 1);
	T[0] = -T[0], T[1] = -T[1], T[2] = -T[2];
	return;
}
void GetCfromT(CameraData &camInfo)
{
	double iR[9];
	mat_transpose(camInfo.R, iR, 3, 3);

	mat_mul(iR, camInfo.T, camInfo.camCenter, 3, 3, 1);
	for (int ii = 0; ii < 3; ii++)
		camInfo.camCenter[ii] = -camInfo.camCenter[ii];
}
void GetCfromT(double *R, double *T, double *C)
{
	//C = -R't;
	double iR[9];
	mat_transpose(R, iR, 3, 3);

	mat_mul(iR, T, C, 3, 3, 1);
	C[0] = -C[0], C[1] = -C[1], C[2] = -C[2];
	return;
}

void GetRCGL(CameraData &camInfo)
{
	double iR[9], center[3];
	mat_invert(camInfo.R, iR);

	camInfo.Rgl[0] = camInfo.R[0], camInfo.Rgl[1] = camInfo.R[1], camInfo.Rgl[2] = camInfo.R[2], camInfo.Rgl[3] = 0.0;
	camInfo.Rgl[4] = camInfo.R[3], camInfo.Rgl[5] = camInfo.R[4], camInfo.Rgl[6] = camInfo.R[5], camInfo.Rgl[7] = 0.0;
	camInfo.Rgl[8] = camInfo.R[6], camInfo.Rgl[9] = camInfo.R[7], camInfo.Rgl[10] = camInfo.R[8], camInfo.Rgl[11] = 0.0;
	camInfo.Rgl[12] = 0, camInfo.Rgl[13] = 0, camInfo.Rgl[14] = 0, camInfo.Rgl[15] = 1.0;

	mat_mul(iR, camInfo.T, center, 3, 3, 1);
	camInfo.camCenter[0] = -center[0], camInfo.camCenter[1] = -center[1], camInfo.camCenter[2] = -center[2];
	return;
}
void GetRCGL(double *R, double *T, double *Rgl, double *C)
{
	double iR[9], center[3];
	mat_invert(R, iR);

	Rgl[0] = R[0], Rgl[1] = R[1], Rgl[2] = R[2], Rgl[3] = 0.0;
	Rgl[4] = R[3], Rgl[5] = R[4], Rgl[6] = R[5], Rgl[7] = 0.0;
	Rgl[8] = R[6], Rgl[9] = R[7], Rgl[10] = R[8], Rgl[11] = 0.0;
	Rgl[12] = 0, Rgl[13] = 0, Rgl[14] = 0, Rgl[15] = 1.0;

	mat_mul(iR, T, center, 3, 3, 1);
	C[0] = -center[0], C[1] = -center[1], C[2] = -center[2];
	return;
}
void AssembleRT(double *R, double *T, double *RT, bool GivenCenter)
{
	if (!GivenCenter)
	{
		RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = T[0];
		RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = T[1];
		RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = T[2];
	}
	else//RT = [R, -R*C];
	{
		double mT[3];
		mat_mul(R, T, mT, 3, 3, 1);
		RT[0] = R[0], RT[1] = R[1], RT[2] = R[2], RT[3] = -mT[0];
		RT[4] = R[3], RT[5] = R[4], RT[6] = R[5], RT[7] = -mT[1];
		RT[8] = R[6], RT[9] = R[7], RT[10] = R[8], RT[11] = -mT[2];
	}
}
void DesembleRT(double *R, double *T, double *RT)
{
	R[0] = RT[0], R[1] = RT[1], R[2] = RT[2], T[0] = RT[3];
	R[3] = RT[4], R[4] = RT[5], R[5] = RT[6], T[1] = RT[7];
	R[6] = RT[8], R[7] = RT[9], R[8] = RT[10], T[2] = RT[11];
}

void AssembleRT_RS(Point2d &uv, CameraData &cam, double *R, double *T)
{
	double *K = cam.K;
	double *wt = cam.wt;
	double *R_global = cam.R;
	double *T_global = cam.T;

	double ycn = (uv.y - K[5]) / K[4];

	double wx = ycn*wt[0], wy = ycn*wt[1], wz = ycn*wt[2];
	double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;
	double denum = 1.0 + wx2 + wy2 + wz2;

	double Rw[9] = { 1.0 + wx2 - wy2 - wz2, 2.0 * wxy - 2.0 * wz, 2.0 * wy + 2.0 * wxz,
		2.0 * wz + 2.0 * wxy, 1.0 - wx2 + wy2 - wz2, 2.0 * wyz - 2.0 * wx,
		2.0 * wxz - 2.0 * wy, 2.0 * wx + 2.0 * wyz, 1.0 - wx2 - wy2 + wz2 };

	for (int jj = 0; jj < 9; jj++)
		Rw[jj] = Rw[jj] / denum;

	mat_mul(Rw, R_global, R, 3, 3, 3);
	T[0] = T_global[0] + ycn*wt[3], T[1] = T_global[1] + ycn*wt[4], T[2] = T_global[2] + ycn*wt[5];

	return;
}
void AssembleRT_RS(Point2d &uv, double *intrinsic, double *rt, double *wt, double *R, double *T)
{
	double R_global[9], T_global[3] = { rt[3], rt[4], rt[5] };
	getRfromr(rt, R_global);

	double ycn = (uv.y - intrinsic[4]) / intrinsic[1];

	double wx = ycn*wt[0], wy = ycn*wt[1], wz = ycn*wt[2];
	double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;
	double denum = 1.0 + wx2 + wy2 + wz2;

	double Rw[9] = { 1.0 + wx2 - wy2 - wz2, 2.0 * wxy - 2.0 * wz, 2.0 * wy + 2.0 * wxz,
		2.0 * wz + 2.0 * wxy, 1.0 - wx2 + wy2 - wz2, 2.0 * wyz - 2.0 * wx,
		2.0 * wxz - 2.0 * wy, 2.0 * wx + 2.0 * wyz, 1.0 - wx2 - wy2 + wz2 };

	for (int jj = 0; jj < 9; jj++)
		Rw[jj] = Rw[jj] / denum;

	mat_mul(Rw, R_global, R, 3, 3, 3);
	T[0] = T_global[0] + ycn*wt[3], T[1] = T_global[1] + ycn*wt[4], T[2] = T_global[2] + ycn*wt[5];

	return;
}
void AssembleP(CameraData &camera)
{
	double RT[12];
	Set_Sub_Mat(camera.R, RT, 3, 3, 4, 0, 0);
	Set_Sub_Mat(camera.T, RT, 1, 3, 4, 3, 0);
	mat_mul(camera.K, RT, camera.P, 3, 3, 4);
	return;
}
void AssembleP(double *K, double *R, double *T, double *P)
{
	double RT[12];
	Set_Sub_Mat(R, RT, 3, 3, 4, 0, 0);
	Set_Sub_Mat(T, RT, 1, 3, 4, 3, 0);
	mat_mul(K, RT, P, 3, 3, 4);
	return;
}
void AssembleP(double *K, double *RT, double *P)
{
	mat_mul(K, RT, P, 3, 3, 4);
	return;
}

void AssembleP_RS(Point2d uv, double *K, double *R_global, double *T_global, double *wt, double *P)
{
	double ycn = (uv.y - K[5]) / K[4];
	double xcn = (uv.x - K[2] - K[1] * ycn) / K[0];

	double wx = ycn*wt[0], wy = ycn*wt[1], wz = ycn*wt[2];
	double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;
	double denum = 1.0 + wx2 + wy2 + wz2;

	double Rw[9] = { 1.0 + wx2 - wy2 - wz2, 2.0 * wxy - 2.0 * wz, 2.0 * wy + 2.0 * wxz,
		2.0 * wz + 2.0 * wxy, 1.0 - wx2 + wy2 - wz2, 2.0 * wyz - 2.0 * wx,
		2.0 * wxz - 2.0 * wy, 2.0 * wx + 2.0 * wyz, 1.0 - wx2 - wy2 + wz2 };

	for (int jj = 0; jj < 9; jj++)
		Rw[jj] = Rw[jj] / denum;

	double R[9];  mat_mul(Rw, R_global, R, 3, 3, 3);
	double T[3] = { T_global[0] + ycn*wt[3], T_global[1] + ycn*wt[4], T_global[2] + ycn*wt[5] };

	AssembleP(K, R, T, P);

	return;
}
void AssembleP_RS(Point2d uv, CameraData &cam, double *P)
{
	double *K = cam.K;
	double *wt = cam.wt;
	double *R_global = cam.R;
	double *T_global = cam.T;

	double ycn = (uv.y - K[5]) / K[4];
	double xcn = (uv.x - K[2] - K[1] * ycn) / K[0];

	double wx = ycn*wt[0], wy = ycn*wt[1], wz = ycn*wt[2];
	double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz, wxz = wx*wz, wxy = wx*wy, wyz = wy*wz;
	double denum = 1.0 + wx2 + wy2 + wz2;

	double Rw[9] = { 1.0 + wx2 - wy2 - wz2, 2.0 * wxy - 2.0 * wz, 2.0 * wy + 2.0 * wxz,
		2.0 * wz + 2.0 * wxy, 1.0 - wx2 + wy2 - wz2, 2.0 * wyz - 2.0 * wx,
		2.0 * wxz - 2.0 * wy, 2.0 * wx + 2.0 * wyz, 1.0 - wx2 - wy2 + wz2 };

	for (int jj = 0; jj < 9; jj++)
		Rw[jj] = Rw[jj] / denum;

	double R[9];  mat_mul(Rw, R_global, R, 3, 3, 3);
	double T[3] = { T_global[0] + ycn*wt[3], T_global[1] + ycn*wt[4], T_global[2] + ycn*wt[5] };

	AssembleP(K, R, T, P);

	return;
}

void InvertCameraPose(double *R, double *T, double *iR, double *iT)
{
	double RT[16] = { R[0], R[1], R[2], T[0],
		R[3], R[4], R[5], T[1],
		R[6], R[7], R[8], T[2],
		0, 0, 0, 1 };

	double iRT[16];
	mat_invert(RT, iRT, 4);

	iR[0] = iRT[0], iR[1] = iRT[1], iR[2] = iRT[2], iT[0] = iRT[3];
	iR[3] = iRT[4], iR[4] = iRT[5], iR[5] = iRT[6], iT[1] = iRT[7];
	iR[6] = iRT[8], iR[7] = iRT[9], iR[8] = iRT[10], iT[2] = iRT[11];

	return;
}

void CopyCamereInfo(CameraData Src, CameraData &Dst, bool Extrinsic)
{
	int ii;
	Dst.notCalibrated = Src.notCalibrated;
	for (ii = 0; ii < 9; ii++)
		Dst.K[ii] = Src.K[ii];
	for (ii = 0; ii < 7; ii++)
		Dst.distortion[ii] = Src.distortion[ii];
	for (ii = 0; ii < 5; ii++)
		Dst.intrinsic[ii] = Src.intrinsic[ii];

	Dst.LensModel = Src.LensModel;
	Dst.ShutterModel = Src.ShutterModel;
	Dst.nInlierThresh = Src.nInlierThresh;
	Dst.threshold = Src.threshold;
	Dst.width = Src.width, Dst.height = Src.height;
	Dst.valid = Src.valid;
	Dst.viewID = Src.viewID, Dst.frameID = Src.frameID;
	Dst.filename = Src.filename;

	if (Extrinsic)
	{
		for (ii = 0; ii < 9; ii++)
			Dst.R[ii] = Src.R[ii];
		for (ii = 0; ii < 3; ii++)
			Dst.T[ii] = Src.T[ii];
		for (ii = 0; ii < 6; ii++)
			Dst.rt[ii] = Src.rt[ii];
		for (ii = 0; ii < 6; ii++)
			Dst.wt[ii] = Src.wt[ii];
		for (ii = 0; ii < 16; ii++)
			Dst.Rgl[ii] = Src.Rgl[ii];
		for (ii = 0; ii < 3; ii++)
			Dst.camCenter[ii] = Src.camCenter[ii];
	}
	return;
}

void QuaternionLinearInterp(double *quad1, double *quad2, double *quadi, double u)
{
	const double DOT_THRESHOLD = 0.9995;

	double C_phi = dotProduct(quad1, quad2);
	if (C_phi > DOT_THRESHOLD) //do linear interp
		for (int ii = 0; ii < 4; ii++)
			quadi[ii] = (1.0 - u)*quad1[ii] + u*quad2[ii];
	else
	{
		double phi = acos(C_phi);
		double  S_phi = sin(phi), S_1uphi = sin((1.0 - u)*phi) / S_phi, S_uphi = sin(u*phi) / S_phi;
		for (int ii = 0; ii < 4; ii++)
			quadi[ii] = S_1uphi*quad1[ii] + S_uphi*quad2[ii];
		normalize(quadi, 4);
		if (dotProduct(quad1, quadi) < 0.0)
			for (int ii = 0; ii < 4; ii++)
				quadi[ii] = -quadi[ii];
	}
	return;
}
void GetPosesGL(double *R, double *T, double *poseGL)
{
	poseGL[0] = R[0], poseGL[1] = R[1], poseGL[2] = R[2], poseGL[3] = 0.0;
	poseGL[4] = R[3], poseGL[5] = R[4], poseGL[6] = R[5], poseGL[7] = 0.0;
	poseGL[8] = R[6], poseGL[9] = R[7], poseGL[10] = R[8], poseGL[11] = 0.0;
	poseGL[12] = 0, poseGL[13] = 0, poseGL[14] = 0, poseGL[15] = 1.0;

	//Center = -iR*T 
	double iR[9], center[3];
	mat_invert(R, iR);
	mat_mul(iR, T, center, 3, 3, 1);
	poseGL[16] = -center[0], poseGL[17] = -center[1], poseGL[18] = -center[2];

	return;
}
int Pose_se_BSplineInterpolation(char *Fname1, char *Fname2, int nsamples, char *Fname3)
{
	int nControls, nbreaks, SplineOrder, se3;

	//Read data
	FILE *fp = fopen(Fname1, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname1);
		return 1;
	}
	fscanf(fp, "%d %d %d %d\n", &nControls, &nbreaks, &SplineOrder, &se3);

	double *BreakLoc = new double[nbreaks];
	double *ControlLoc = new double[nControls];
	double *ControlPose = new double[nControls * 6];
	for (int ii = 0; ii < nControls; ii++)
	{
		fscanf(fp, "%lf ", &ControlLoc[ii]);
		for (int jj = 0; jj < 6; jj++)
			fscanf(fp, "%lf ", &ControlPose[ii * 6 + jj]);
	}
	for (int ii = 0; ii < nbreaks; ii++)
		fscanf(fp, "%lf", &BreakLoc[ii]);
	fclose(fp);

	//Bspline generator
	double *Bi = new double[nControls];
	double *knots = new double[nControls + SplineOrder];
	BSplineGetKnots(knots, BreakLoc, nbreaks, nControls, SplineOrder);

	//Start interpolation pose
	double *SampleLoc = new double[nsamples];
	double step = (BreakLoc[nbreaks - 1] - BreakLoc[0]) / (nsamples - 1);
	for (int ii = 0; ii < nsamples; ii++)
		SampleLoc[ii] = BreakLoc[0] + step*ii;

	int ActingID[4];
	double twist[6], tr[6], R[9], T[3], poseGL[19];

	fp = fopen(Fname2, "w+");
	for (int ii = 0; ii < nsamples; ii++)
	{
		//FindActingControlPts(SampleLoc[ii], ActingID, nControls, bw, Bi, SplineOrder, 0);
		//gsl_bspline_eval(SampleLoc[ii], Bi, bw);

		BSplineFindActiveCtrl(ActingID, SampleLoc[ii], knots, nbreaks, nControls, SplineOrder, 0);
		BSplineGetBasis(SampleLoc[ii], Bi, knots, nbreaks, nControls, SplineOrder);

		for (int jj = 0; jj < 6; jj++)
		{
			if (se3 == 1)
			{
				twist[jj] = 0.0;
				for (int kk = 0; kk < 4; kk++)
					twist[jj] += ControlPose[jj + 6 * ActingID[kk]] * Bi[ActingID[kk]];// gsl_vector_get(Bi, ActingID[kk]);
			}
			else
			{
				tr[jj] = 0;
				for (int kk = 0; kk < 4; kk++)
					tr[jj] += ControlPose[jj + 6 * ActingID[kk]] * Bi[ActingID[kk]];// gsl_vector_get(Bi, ActingID[kk]);
			}
		}

		if (se3 == 1)
			getRTFromTwist(twist, R, T);
		else
		{
			getRfromr(tr + 3, R);
			for (int jj = 0; jj < 3; jj++)
				T[jj] = tr[jj];
		}

		GetPosesGL(R, T, poseGL);

		fprintf(fp, "%d ", (int)SampleLoc[ii]);
		for (int jj = 0; jj < 19; jj++)
			fprintf(fp, "%.16e ", poseGL[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);


	fp = fopen(Fname3, "w+");
	for (int ii = 0; ii < nControls; ii++)
	{
		if (se3 == 1)
			getRTFromTwist(&ControlPose[6 * ii], R, T);
		else
		{
			getRfromr(&ControlPose[6 * ii], R);
			for (int jj = 0; jj < 3; jj++)
				T[jj] = ControlPose[6 * ii + jj];
		}
		GetPosesGL(R, T, poseGL);

		fprintf(fp, "%d ", (int)(ControlLoc[ii] + 0.5));
		for (int jj = 0; jj < 19; jj++)
			fprintf(fp, "%.16e ", poseGL[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]SampleLoc, delete[]knots, delete[]Bi;
	return 0;
}
int Pose_se_DCTInterpolation(char *FnameIn, char *FnameOut, int nsamples)
{
	int startF, nCoeffs, sampleStep;

	FILE *fp = fopen(FnameIn, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", FnameIn);
		return 1;
	}

	fscanf(fp, "%d %d %d ", &startF, &nCoeffs, &sampleStep);
	double *C = new double[6 * nCoeffs];
	for (int jj = 0; jj < 6; jj++)
		for (int ii = 0; ii < nCoeffs; ii++)
			fscanf(fp, "%lf ", &C[ii + jj*nCoeffs]);
	fclose(fp);

	double *iBi = new double[nCoeffs];
	double twist[6], R[9], T[3], poseGL[19];
	double stopF = sampleStep*(nCoeffs - 1) + startF, resampleStep = 1.0*nCoeffs / (stopF - startF);

	fp = fopen(FnameOut, "w+");
	for (int ii = 0; ii < nsamples; ii++)
	{
		double loc = 1.0*ii * resampleStep; //linspace(0, ncoeffs-1, nsamples)
		GenerateiDCTBasis(iBi, nCoeffs, loc);
		for (int jj = 0; jj < 6; jj++)
		{
			twist[jj] = 0.0;
			for (int ii = 0; ii < nCoeffs; ii++)
				twist[jj] += C[ii + jj*nCoeffs] * iBi[ii];
		}

		getRTFromTwist(twist, R, T);
		GetPosesGL(R, T, poseGL);

		loc = loc / resampleStep + startF; //linspace(startF, stopF, nsamples)
		fprintf(fp, "%d ", (int)(loc + 0.5));
		for (int jj = 0; jj < 19; jj++)
			fprintf(fp, "%.16e ", poseGL[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]iBi;
	return 0;
}

void getRayDir(double *rayDir, double *iK, double *R, double *uv1)
{
	//rayDirection = iR*iK*[u,v,1]
	Map< const MatrixXdr >	eR(R, 3, 3);
	Map< const MatrixXdr >	eiK(iK, 3, 3);
	Map<Vector3d> euv1(uv1, 3);
	Map<Vector3d> eRayDir(rayDir, 3);
	eRayDir = eR.transpose()*eiK*euv1;
	eRayDir = eRayDir / eRayDir.norm();

	return;
}
void GetRelativeTransformation(double *R0, double *T0, double *R1, double *T1, double *R1to0, double *T1to0)
{
	Map < Matrix < double, 3, 3, RowMajor > > eR0(R0, 3, 3), eR1(R1, 3, 3), eR1to0(R1to0, 3, 3);
	Map<Vector3d> eT0(T0, 3), eT1(T1, 3), eT1to0(T1to0, 3);

	eR1to0 = eR1*eR0.transpose();
	eT1to0 = eT1 - eR1*eR0.transpose()*eT0;

	return;
}
void ComputeEpipole(double *F, Point2d &e1, Point2d &e2)
{
	//Fe1 = 0, e2t*F = 0
	Map < Matrix < double, 3, 3, RowMajor > > eF(F, 3, 3);
	JacobiSVD<MatrixXd> eF_svd(eF, ComputeFullU);
	MatrixXd U = eF_svd.matrixU(), V = eF_svd.matrixV();

	e1.x = V(0, 2) / V(2, 2), e1.y = V(1, 2) / V(2, 2); //last column of V + normalize
	e2.x = U(0, 2) / U(2, 2), e2.y = U(1, 2) / U(2, 2); //last column of U + normalize

	return;
}


