#include <vector>
#include <algorithm>
#include "DataStructure.h"
#include "Drivers/Drivers.h"
#include "Ulti/MathUlti.h"
#include "Vis/Visualization.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <ceres/normal_prior.h>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <boost/config.hpp>
#include <iostream>
#include <fstream>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include <time.h>
#ifdef _WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

using namespace std;
using namespace cv;
using namespace boost;
using namespace Eigen;
using namespace smpl;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::SoftLOneLoss;
using ceres::HuberLoss;
using ceres::TukeyLoss;
using ceres::Problem;
using ceres::Solver;

struct DensePose
{
	DensePose()
	{
		cid = -1; fid = -1, height = -1, width = -1;
		valid = 0, nCorresThresh = 30, CorresThresh = 4;
		validParts = new int[14];
		vdfield.resize(15);
		vEdge = new vector<Point2i>[15];
		for (int ii = 0; ii < 14; ii++)
			validParts[ii] = 0;
	}
	DensePose(int cid_, int fid_, int width_, int height_, int nCorresThresh_, double CorresThresh_)
	{
		cid = cid_, fid = fid_, width = width_, height = height_, nCorresThresh = nCorresThresh_, CorresThresh = CorresThresh_;
		validParts = new int[14];
		vEdge = new vector<Point2i>[15];
		vdfield.resize(15);
		for (int ii = 0; ii < 14; ii++)
			validParts[ii] = 0;
	}
	~DensePose()
	{
		for (int ii = 0; ii < 14; ii++)
			if (validParts[ii])
				delete[]vdfield[ii];
		delete[]validParts;
	}

	int cid, fid, refFid, width, height, valid, nCorresThresh;
	int *validParts;
	vector<uint16_t *> vdfield;
	vector<Point2i> *vEdge;
	double CorresThresh;

	vector<int> DP_vid;
	vector<Point2d> DP_uv;
};
class GermanMcClure : public ceres::LossFunction {
public:
	explicit GermanMcClure(double _a, double _mu) :a2(_a*_a), mu(_mu) { }
	virtual void Evaluate(double s, double* rho) const
	{
		double mu2 = mu * mu, tmp = a2 * mu + s, tmp2 = tmp * tmp, tmp3 = tmp * tmp2;
		rho[0] = a2 * mu*s / tmp;
		rho[1] = mu * a2 / tmp2;
		rho[2] = -2.0*mu2 / tmp3;
	}

private:
	const double a2, mu;
};
struct SmplPoseReg {
	SmplPoseReg() {}
	template <typename T>	bool operator()(const T* const pose, T* residuals) const
	{
		for (int ii = 3; ii < 72; ii++)
			residuals[ii - 3] = pose[ii];

		return true;
	}
	static ceres::CostFunction* Create() {
		return (new ceres::AutoDiffCostFunction<SmplPoseReg, 69, 72>(new SmplPoseReg()));
	}
};
vector<int> RasterizeMesh4OccludingContour(Point2f *vuv, int nVertices, int width, int height, vector<Point3i> &vfaces, vector<Point2i> &vConnections, bool *hit)
{
	bool memCreated = false;
	if (hit == NULL)
	{
		memCreated = true;
		hit = new bool[width*height];
	}
	for (int ii = 0; ii < width*height; ii++)
		hit[ii] = 0;

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic, 4)
	for (int fid = 0; fid < vfaces.size(); fid++)
	{
		int vid1 = vfaces[fid].x, vid2 = vfaces[fid].y, vid3 = vfaces[fid].z;
		Point2f uv1 = vuv[vid1], uv2 = vuv[vid2], uv3 = vuv[vid3];
		if (uv1.x < 1 || uv2.x < 1 || uv3.x < 1)
			continue;
		int maxX = min((int)(max(max(max(0, uv1.x), uv2.x), uv3.x)) + 1, width - 1);
		int minX = max((int)(min(min(min(width - 1, uv1.x), uv2.x), uv3.x)) + 1, 0);
		int maxY = min((int)(max(max(max(0, uv1.y), uv2.y), uv3.y)) + 1, height - 1);
		int minY = max((int)(min(min(min(height - 1, uv1.y), uv2.y), uv3.y)) + 1, 0);
		for (int jj = minY; jj <= maxY; jj++)
			for (int ii = minX; ii < maxX; ii++)
				if (PointInTriangle(uv1, uv2, uv3, Point2f(ii, jj)))
					hit[ii + jj * width] = 1;
	}

	vector<int> occludingcontour;
#pragma omp parallel for schedule(dynamic, 4)
	for (int con = 0; con < vConnections.size(); con++)
	{
		int vid1 = vConnections[con].x, vid2 = vConnections[con].y;
		if (vuv[vid1].x < 3 || vuv[vid1].x >width - 3 || vuv[vid1].y<3 || vuv[vid1].y>height - 3 || vuv[vid2].x < 3 || vuv[vid2].x >width - 3 || vuv[vid2].y<3 || vuv[vid2].y>height - 3)
			continue;
		Point2f muv = 0.5*(vuv[vid1] + vuv[vid2]);

		Point2f normal(-(vuv[vid1].y - vuv[vid2].y), vuv[vid1].x - vuv[vid2].x);
		double norm = sqrt(normal.x*normal.x + normal.y*normal.y);
		normal.x = normal.x / norm, normal.y = normal.y / norm;

		Point2i right((int)(muv.x + normal.x + 0.5), (int)(muv.y + normal.y + 0.5));
		Point2i left((int)(muv.x - normal.x + 0.5), (int)(muv.y - normal.y + 0.5));
		if (hit[right.x + right.y*width] && hit[left.x + left.y*width])
			continue;
		else
		{
#pragma omp critical
			occludingcontour.push_back(vid1), occludingcontour.push_back(vid2);
		}
	}

	sort(occludingcontour.begin(), occludingcontour.end());
	std::vector<int>::iterator it = unique(occludingcontour.begin(), occludingcontour.end());
	occludingcontour.resize(std::distance(occludingcontour.begin(), it));

	if (memCreated)
		delete[]hit;
	return occludingcontour;
}
struct SmplFitCallBack : ceres::EvaluationCallback
{
	explicit SmplFitCallBack(SMPLModel &mySMPL_, int nInstances_, int pointFomat_, std::vector<double *> _Vparameters, double* _All_V_ptr, Point2f *_All_uv_ptr, double *_All_dVdp_ptr, double *_All_dVdc_ptr, double *_All_dVds_ptr,
		double* _All_Jsmpl_ptr, double* _All_J_ptr, double *_All_dJdt_ptr, double *_All_dJdp_ptr, double *_All_dJdc_ptr, double *_All_dJds_ptr,
		DensePose*_vDensePose, VideoData *_VideoInfo, char *_vVertexTypeAndVisibility, vector<int>_vCams, int _maxImgLength, bool _hasDensePose) :
		EvaluationCallback(), mySMPL(mySMPL_), nInstances(nInstances_), pointFomat(pointFomat_), Vparameters(_Vparameters),
		All_V_ptr(_All_V_ptr), All_uv_ptr(_All_uv_ptr), All_dVdp_ptr(_All_dVdp_ptr), All_dVdc_ptr(_All_dVdc_ptr), All_dVds_ptr(_All_dVds_ptr),
		All_Jsmpl_ptr(_All_Jsmpl_ptr), All_J_ptr(_All_J_ptr), All_dJdt_ptr(_All_dJdt_ptr), All_dJdp_ptr(_All_dJdp_ptr), All_dJdc_ptr(_All_dJdc_ptr), All_dJds_ptr(_All_dJds_ptr),
		vDensePose(_vDensePose), VideoInfo(_VideoInfo), vVertexTypeAndVisibility(_vVertexTypeAndVisibility), vCams(_vCams), maxImgLength(_maxImgLength), hasDensePose(_hasDensePose)
	{
		const int nVertices = SMPLModel::nVertices;

		int nthreads = omp_get_max_threads();
		//vUV = new vector<Point2d>[nthreads];
		//for (int ii = 0; ii < nthreads; ii++)
		//	vUV[ii].resize(nVertices);
	}

	virtual ~SmplFitCallBack() {
		;// delete[]vUV;
	}

	virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
	{
		// At this point, the incoming parameters are implicitly pushed by Ceres into the user parameter blocks; in contrast to in Evaluate().
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;
		int  nVertices3 = nVertices * 3, nJoints3 = nJoints * 3, naJoints3 = naJoints * 3, nCorrespond3 = 0;
		if (pointFomat == 18)
			nCorrespond3 = 42;
		else// if (pointFomat == 25)
			nCorrespond3 = 72;

		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		//SparseMatrix<double, ColMajor> J_reg = pointFomat == 17 ? mySMPL.J_regl_17_bigl_col_ : (18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_);
		SparseMatrix<double, ColMajor> J_reg = pointFomat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
		SparseMatrix<double, ColMajor>  dVdt_ = kroneckerProduct(VectorXd::Ones(nVertices), eye3);
		int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };

		omp_set_num_threads(omp_get_max_threads());
		for (int idf = 0; idf < nInstances; idf++)
		{
			if (evaluate_jacobians || new_evaluation_point)
			{
				double *s = Vparameters[0], *c = Vparameters[1], *p = Vparameters[2 + 2 * idf], *t = Vparameters[2 + 2 * idf + 1];

				double *V_ptr = All_V_ptr + nVertices3 * idf,
					*dVdp_ptr = All_dVdp_ptr + nVertices3 * nJoints3 * idf,
					*dVdc_ptr = All_dVdc_ptr + nVertices3 * nShapeCoeffs * idf,
					*dVds_ptr = All_dVds_ptr + nVertices3 * idf;
				double *Jsmpl_ptr = All_Jsmpl_ptr + naJoints3 * idf,
					*J_ptr = All_J_ptr + nCorrespond3 * idf,
					*dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf,
					*dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf,
					*dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf,
					*dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;

				Map< VectorXd > V_vec(V_ptr, nVertices3);
				Map< VectorXd > Jsmpl_vec(Jsmpl_ptr, naJoints3);
				Map< VectorXd > J_vec(J_ptr, nCorrespond3);

				if (evaluate_jacobians)
				{
					MatrixXdr dVdp, dVdc;
					smpl::reconstruct(mySMPL, c, p, V_ptr, dVdc, dVdp);

					//V_vec = s[0] * V_vec + dVdt_ * t_vec;  do jacobian here before it is updated to save some operations
					for (int jj = 0; jj < nVertices3; jj++)
						dVds_ptr[jj] = V_vec(jj);

#pragma omp parallel for
					for (int jj = 0; jj < nVertices3; jj++)
					{
						for (int ii = 0; ii < nJoints3; ii++)
							dVdp_ptr[ii + jj * nJoints3] = dVdp(jj, ii);
						for (int ii = 0; ii < nShapeCoeffs; ii++)
							dVdc_ptr[ii + jj * nShapeCoeffs] = dVdp(jj, ii);
					}

					Map< MatrixXdr > dJdt(dJdt_ptr, nCorrespond3, 3), dJdp(dJdp_ptr, nCorrespond3, nJoints * 3), dJdc(dJdc_ptr, nCorrespond3, nShapeCoeffs), dJds(dJds_ptr, nCorrespond3, 1);
					dJdt = J_reg * dVdt_, dJdp = s[0] * J_reg * dVdp, dJdc = s[0] * J_reg * dVdc, dJds = J_reg * V_vec;
				}
				else
					smpl::reconstruct(mySMPL, c, p, V_ptr);

				Map< const Vector3d > t_vec(t);
				V_vec = s[0] * V_vec + dVdt_ * t_vec;
				J_vec = J_reg * V_vec;
				Jsmpl_vec = mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control

				if (hasDensePose) //Precompute for contour. Assume distortion free and global shutter because their effects on contour visibility is mininal
				{
					int nCams = (int)vCams.size();

					char *EdgeVertexTypeAndVisibilityF = vVertexTypeAndVisibility + nVertices * nCams*idf;
					bool *ProjectedMaskF = new bool[maxImgLength];
					DensePose *DensePoseF = vDensePose + idf * nCams;

					for (int jj = 0; jj < nCams; jj++)
					{
						int threadID = 0; //omp_get_thread_num();

						int cid = vCams[jj], offset1 = cid * nVertices;
						int width = DensePoseF[cid].width, height = DensePoseF[cid].height, length = width * height;
						int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
						Point2f *vuv = All_uv_ptr + nVertices * nCams*idf + nVertices * jj;

						for (int vid = 0; vid < nVertices; vid++)
							EdgeVertexTypeAndVisibilityF[offset1 + vid] = -1;

						if (DensePoseF[cid].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
						{
							double *P = VideoInfo[rcid].VideoInfo[rfid].P;
							//int ShutterModel = VideoInfo[rcid].VideoInfo[rfid].ShutterModel;

							//get occluding contour vertices
#pragma omp parallel for
							for (int vid = 0; vid < nVertices; vid++)
							{
								double *X = V_ptr + vid * 3;
								double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
									numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
									denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
								double x = numX / denum, y = numY / denum;

								//if (x<15 || x>width - 15 || y<15 || y>height - 15)
								//	vUV[threadID][vid].x = 0, vUV[threadID][vid].y = 0;
								//else
								//	vUV[threadID][vid].x = x, vUV[threadID][vid].y = y;
								if (x<15 || x>width - 15 || y<15 || y>height - 15)
									vuv[vid].x = 0, vuv[vid].y = 0;
								else
									vuv[vid].x = x, vuv[vid].y = y;
							}

							//vector<int> activeVertices = RasterizeMesh4OccludingContour(&vUV[threadID][0], nVertices, width, height, mySMPL.vFaces, mySMPL.vConnections, ProjectedMaskF + cid*maxImgLength);
							//vector<int> activeVertices = RasterizeMesh4OccludingContour(&vUV[threadID][0], nVertices, width, height, mySMPL.vFaces, mySMPL.vConnections, ProjectedMaskF + maxImgLength);
							//vector<int> activeVertices = RasterizeMesh4OccludingContour(&vUV[threadID][0], nVertices, width, height, mySMPL.vFaces, mySMPL.vConnections, ProjectedMaskF);
							vector<int> activeVertices = RasterizeMesh4OccludingContour(vuv, nVertices, width, height, mySMPL.vFaces, mySMPL.vConnections, ProjectedMaskF);

							for (size_t ii = 0; ii < activeVertices.size(); ii++)
							{
								int vid = activeVertices[ii], id = offset1 + vid;
								int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];

								//if (DensePoseF[cid].validParts[dfId] == 0)
								//	continue;
								EdgeVertexTypeAndVisibilityF[id] = (char)partId;
							}
						}
					}
					delete[]ProjectedMaskF;
				}
			}
		}

		return;
	}

	std::vector<double *> Vparameters;
	double* All_V_ptr, *All_dVdp_ptr, *All_dVdc_ptr, *All_dVds_ptr;
	double*All_Jsmpl_ptr, *All_J_ptr, *All_dJdt_ptr, *All_dJdp_ptr, *All_dJdc_ptr, *All_dJds_ptr;

	//vector<Point2d> *vUV;
	Point2f *All_uv_ptr;
	vector<int> vCams;
	VideoData *VideoInfo;
	DensePose*vDensePose;
	char *vVertexTypeAndVisibility;
	bool hasDensePose;

	int nInstances, pointFomat, maxImgLength;
	SMPLModel &mySMPL;
};
struct SmplFitCallBackUnSync : ceres::EvaluationCallback
{
	explicit SmplFitCallBackUnSync(SMPLModel &mySMPL_, int nInstances_, int pointFomat_, std::vector<double *> _Vparameters, double* _All_V_ptr, Point2f *_All_uv_ptr, double *_All_dVdp_ptr, double *_All_dVdc_ptr, double *_All_dVds_ptr,
		double* _All_Jsmpl_ptr, double* _All_J_ptr, double *_All_dJdt_ptr, double *_All_dJdp_ptr, double *_All_dJdc_ptr, double *_All_dJds_ptr,
		DensePose*_vDensePose, VideoData *_VideoInfo, char *_vVertexTypeAndVisibility, int _maxImgLength, bool _hasDensePose) :
		EvaluationCallback(), mySMPL(mySMPL_), nInstances(nInstances_), pointFomat(pointFomat_), Vparameters(_Vparameters),
		All_V_ptr(_All_V_ptr), All_uv_ptr(_All_uv_ptr), All_dVdp_ptr(_All_dVdp_ptr), All_dVdc_ptr(_All_dVdc_ptr), All_dVds_ptr(_All_dVds_ptr),
		All_Jsmpl_ptr(_All_Jsmpl_ptr), All_J_ptr(_All_J_ptr), All_dJdt_ptr(_All_dJdt_ptr), All_dJdp_ptr(_All_dJdp_ptr), All_dJdc_ptr(_All_dJdc_ptr), All_dJds_ptr(_All_dJds_ptr),
		vDensePose(_vDensePose), VideoInfo(_VideoInfo), vVertexTypeAndVisibility(_vVertexTypeAndVisibility), maxImgLength(_maxImgLength), hasDensePose(_hasDensePose) {}

	virtual ~SmplFitCallBackUnSync() {}

	virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
	{
		// At this point, the incoming parameters are implicitly pushed by Ceres into the user parameter blocks; in contrast to in Evaluate().
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;
		int nVertices3 = nVertices * 3, nCorrespond3 = pointFomat == 18 ? 42 : 72, nJoints3 = nJoints * 3, naJoints3 = naJoints * 3;

		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		SparseMatrix<double, ColMajor> J_reg = pointFomat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
		SparseMatrix<double, ColMajor>  dVdt_ = kroneckerProduct(VectorXd::Ones(nVertices), eye3);
		int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };

		omp_set_num_threads(omp_get_max_threads());
		for (int idf = 0; idf < nInstances; idf++)
		{
			if (evaluate_jacobians || new_evaluation_point)
			{
				double *s = Vparameters[0], *c = Vparameters[1], *p = Vparameters[2 + 2 * idf], *t = Vparameters[2 + 2 * idf + 1];

				double *V_ptr = All_V_ptr + nVertices3 * idf,
					*dVdp_ptr = All_dVdp_ptr + nVertices3 * nJoints3 * idf,
					*dVdc_ptr = All_dVdc_ptr + nVertices3 * nShapeCoeffs * idf,
					*dVds_ptr = All_dVds_ptr + nVertices3 * idf;
				double *Jsmpl_ptr = All_Jsmpl_ptr + naJoints3 * idf,
					*J_ptr = All_J_ptr + nCorrespond3 * idf,
					*dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf,
					*dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf,
					*dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf,
					*dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;

				Map< VectorXd > V_vec(V_ptr, nVertices3);
				Map< VectorXd > Jsmpl_vec(Jsmpl_ptr, naJoints3);
				Map< VectorXd > J_vec(J_ptr, nCorrespond3);

				if (evaluate_jacobians)
				{
					MatrixXdr dVdp, dVdc;
					smpl::reconstruct(mySMPL, c, p, V_ptr, dVdc, dVdp);

					//V_vec = s[0] * V_vec + dVdt_ * t_vec;  do jacobian here before it is updated to save some operations
					for (int jj = 0; jj < nVertices3; jj++)
						dVds_ptr[jj] = V_vec(jj);

#pragma omp parallel for
					for (int jj = 0; jj < nVertices3; jj++)
					{
						for (int ii = 0; ii < nJoints3; ii++)
							dVdp_ptr[ii + jj * nJoints3] = dVdp(jj, ii);
						for (int ii = 0; ii < nShapeCoeffs; ii++)
							dVdc_ptr[ii + jj * nShapeCoeffs] = dVdp(jj, ii);
					}

					Map< MatrixXdr > dJdt(dJdt_ptr, nCorrespond3, 3), dJdp(dJdp_ptr, nCorrespond3, nJoints * 3), dJdc(dJdc_ptr, nCorrespond3, nShapeCoeffs), dJds(dJds_ptr, nCorrespond3, 1);
					dJdt = J_reg * dVdt_, dJdp = s[0] * J_reg * dVdp, dJdc = s[0] * J_reg * dVdc, dJds = J_reg * V_vec;
				}
				else
					smpl::reconstruct(mySMPL, c, p, V_ptr);

				Map< const Vector3d > t_vec(t);
				V_vec = s[0] * V_vec + dVdt_ * t_vec;

				J_vec = J_reg * V_vec;
				Jsmpl_vec = mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control

				if (hasDensePose) //Precompute for contour. Assume distortion free and global shutter because their effects on contour visibility is mininal
				{
					int rcid = vDensePose[idf].cid, rfid = vDensePose[idf].fid;
					int width = vDensePose[idf].width, height = vDensePose[idf].height, length = width * height;

					char *EdgeVertexTypeAndVisibilityF = vVertexTypeAndVisibility + nVertices * idf;
					bool *ProjectedMaskF = new bool[maxImgLength];
					Point2f *vuv = All_uv_ptr + nVertices * idf;
					double *P = VideoInfo[rcid].VideoInfo[rfid].P;

					if (vDensePose[idf].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
					{
						for (int vid = 0; vid < nVertices; vid++)
							EdgeVertexTypeAndVisibilityF[vid] = -1;

						//get occluding contour vertices
#pragma omp parallel for
						for (int vid = 0; vid < nVertices; vid++)
						{
							double *X = V_ptr + vid * 3;
							double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
								numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
								denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
							double x = numX / denum, y = numY / denum;

							if (x<15 || x>width - 15 || y<15 || y>height - 15)
								vuv[vid].x = 0, vuv[vid].y = 0;
							else
								vuv[vid].x = x, vuv[vid].y = y;
						}
						vector<int> activeVertices = RasterizeMesh4OccludingContour(vuv, nVertices, width, height, mySMPL.vFaces, mySMPL.vConnections, ProjectedMaskF);

						for (size_t ii = 0; ii < activeVertices.size(); ii++)
						{
							int vid = activeVertices[ii];
							int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];

							EdgeVertexTypeAndVisibilityF[vid] = (char)partId;
						}
					}
					delete[]ProjectedMaskF;
				}
			}
		}

		return;
	}

	std::vector<double *> Vparameters;
	double* All_V_ptr, *All_dVdp_ptr, *All_dVdc_ptr, *All_dVds_ptr;
	double*All_Jsmpl_ptr, *All_J_ptr, *All_dJdt_ptr, *All_dJdp_ptr, *All_dJdc_ptr, *All_dJds_ptr;

	Point2f *All_uv_ptr;
	VideoData *VideoInfo;
	DensePose*vDensePose;
	char *vVertexTypeAndVisibility;
	bool hasDensePose;

	int nInstances, pointFomat, maxImgLength;
	SMPLModel &mySMPL;
};
struct SmplFitCallBackUnSynced_old : ceres::EvaluationCallback
{
	explicit SmplFitCallBackUnSynced_old(SMPLModel &mySMPL_, int nInstances_, int pointFomat_, std::vector<double *> _Vparameters, double* _All_V_ptr, double *_All_dVdp_ptr, double *_All_dVdc_ptr, double *_All_dVds_ptr,
		double* _All_Jsmpl_ptr, double* _All_J_ptr, double *_All_dJdt_ptr, double *_All_dJdp_ptr, double *_All_dJdc_ptr, double *_All_dJds_ptr,
		vector<ImgPoseEle> &_vPoseEle, DensePose*_vDensePose, VideoData *_VideoInfo, char *_vVertexTypeAndVisibility, bool *_vProjectedMask, vector<int>_vCams, int _maxImgLength) :
		EvaluationCallback(), mySMPL(mySMPL_), nInstances(nInstances_), pointFomat(pointFomat_), Vparameters(_Vparameters),
		All_V_ptr(_All_V_ptr), All_dVdp_ptr(_All_dVdp_ptr), All_dVdc_ptr(_All_dVdc_ptr), All_dVds_ptr(_All_dVds_ptr),
		All_Jsmpl_ptr(_All_Jsmpl_ptr), All_J_ptr(_All_J_ptr), All_dJdt_ptr(_All_dJdt_ptr), All_dJdp_ptr(_All_dJdp_ptr), All_dJdc_ptr(_All_dJdc_ptr), All_dJds_ptr(_All_dJds_ptr),
		vPoseEle(_vPoseEle), vDensePose(_vDensePose), VideoInfo(_VideoInfo), vVertexTypeAndVisibility(_vVertexTypeAndVisibility), vProjectedMask(_vProjectedMask), vCams(_vCams), maxImgLength(_maxImgLength)
	{
		const int nVertices = SMPLModel::nVertices;

		int nthreads = omp_get_max_threads();
		vUV = new vector<Point2d>[nthreads];
		for (int ii = 0; ii < nthreads; ii++)
			vUV[ii].resize(nVertices);
	}

	virtual ~SmplFitCallBackUnSynced_old() {
		delete[]vUV;
	}

	virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point)
	{
		// At this point, the incoming parameters are implicitly pushed by Ceres into the user parameter blocks; in contrast to in Evaluate().
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;
		int nVertices3 = nVertices * 3, nCorrespond3 = 72, nJoints3 = nJoints * 3, naJoints3 = naJoints * 3;

		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		SparseMatrix<double, ColMajor> J_reg = pointFomat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
		SparseMatrix<double, ColMajor>  dVdt_ = kroneckerProduct(VectorXd::Ones(nVertices), eye3);
		int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };

		omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
		for (int idf = 0; idf < nInstances; idf++)
		{
			int threadID = omp_get_thread_num();

			if (evaluate_jacobians || new_evaluation_point)
			{
				double *s = Vparameters[0], *c = Vparameters[1], *p = Vparameters[2 + 2 * idf], *t = Vparameters[2 + 2 * idf + 1];

				double *V_ptr = All_V_ptr + nVertices3 * idf,
					*dVdp_ptr = All_dVdp_ptr + nVertices3 * nJoints3 * idf,
					*dVdc_ptr = All_dVdc_ptr + nVertices3 * nShapeCoeffs * idf,
					*dVds_ptr = All_dVds_ptr + nVertices3 * idf;
				double *Jsmpl_ptr = All_Jsmpl_ptr + naJoints3 * idf,
					*J_ptr = All_J_ptr + nCorrespond3 * idf,
					*dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf,
					*dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf,
					*dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf,
					*dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;

				Map< VectorXd > V_vec(V_ptr, nVertices3);
				Map< VectorXd > Jsmpl_vec(Jsmpl_ptr, naJoints3); //agumented joints for better smothing control
				Map< VectorXd > J_vec(J_ptr, nCorrespond3);

				if (evaluate_jacobians)
				{
					MatrixXdr dVdp, dVdc;
					smpl::reconstruct(mySMPL, c, p, V_ptr, dVdc, dVdp);

					//V_vec = s[0] * V_vec + dVdt_ * t_vec;  do jacobian here before it is updated to save some operations
					for (int jj = 0; jj < nVertices3; jj++)
						dVds_ptr[jj] = V_vec(jj);

					for (int jj = 0; jj < nVertices3; jj++)
					{
						for (int ii = 0; ii < nJoints3; ii++)
							dVdp_ptr[ii + jj * nJoints3] = dVdp(jj, ii);
						for (int ii = 0; ii < nShapeCoeffs; ii++)
							dVdc_ptr[ii + jj * nShapeCoeffs] = dVdp(jj, ii);
					}

					Map< MatrixXdr > dJdt(dJdt_ptr, nCorrespond3, 3), dJdp(dJdp_ptr, nCorrespond3, nJoints * 3), dJdc(dJdc_ptr, nCorrespond3, nShapeCoeffs), dJds(dJds_ptr, nCorrespond3, 1);
					dJdt = J_reg * dVdt_, dJdp = s[0] * J_reg * dVdp, dJdc = s[0] * J_reg * dVdc, dJds = J_reg * V_vec;
				}
				else
					smpl::reconstruct(mySMPL, c, p, V_ptr);

				Map< const Vector3d > t_vec(t);
				V_vec = s[0] * V_vec + dVdt_ * t_vec;
				J_vec = J_reg * V_vec;
				Jsmpl_vec = mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control
														  //cout << "Jsmpl_vec " << idf << ": " << Jsmpl_vec.transpose() << endl << endl;

				//Precompute for contour. Assume distortion free and global shutter because their effects on contour visibility is mininal
				int cid = vPoseEle[idf].viewID, rfid = vPoseEle[idf].frameID, reffid = vPoseEle[idf].refFrameID, nCams = (int)vCams.size();
				int width = VideoInfo[cid].VideoInfo[rfid].width, height = VideoInfo[cid].VideoInfo[rfid].height, length = width * height;

				DensePose *DensePoseI = vDensePose + reffid * nCams + cid;
				//bool *ProjectedMaskI = vProjectedMask + maxImgLength*idf;
				bool *ProjectedMaskI = new bool[maxImgLength];
				char *VertexTypeAndVisibilityI = vVertexTypeAndVisibility + nVertices * idf;
				for (int vid = 0; vid < nVertices; vid++)
					VertexTypeAndVisibilityI[vid] = -1;

				if (DensePoseI[0].valid == 1 && VideoInfo[cid].VideoInfo[rfid].valid == 1)
				{
					double *P = VideoInfo[cid].VideoInfo[rfid].P;

					//get occluding contour vertices
					for (int vid = 0; vid < nVertices; vid++)
					{
						int vid3 = vid * 3;
						double *X = V_ptr + vid3;
						double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
							numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
							denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
						double x = numX / denum, y = numY / denum;

						if (x<15 || x>width - 15 || y<15 || y>height - 15)
							vUV[threadID][vid].x = 0, vUV[threadID][vid].y = 0;
						else
							vUV[threadID][vid].x = x, vUV[threadID][vid].y = y;
					}

					vector<int> activeVertices = RasterizeMesh4OccludingContour(&vUV[threadID][0], nVertices, width, height, mySMPL.vFaces, mySMPL.vConnections, ProjectedMaskI);
					for (auto vid : activeVertices)
						VertexTypeAndVisibilityI[vid] = mySMPL.vDensePosePartId[vid];
				}
				delete[]ProjectedMaskI;
			}
		}

		return;
	}

	std::vector<double *> Vparameters;
	double* All_V_ptr, *All_dVdp_ptr, *All_dVdc_ptr, *All_dVds_ptr;
	double*All_Jsmpl_ptr, *All_J_ptr, *All_dJdt_ptr, *All_dJdp_ptr, *All_dJdc_ptr, *All_dJds_ptr;

	vector<ImgPoseEle> &vPoseEle;
	vector<Point2d> *vUV;
	vector<int> vCams;
	VideoData *VideoInfo;
	DensePose*vDensePose;
	char *vVertexTypeAndVisibility;
	bool *vProjectedMask;

	int nInstances, pointFomat, maxImgLength;
	SMPLModel &mySMPL;
};
class SmplFitSMPL2EdgeCeres_MV :
	public ceres::CostFunction {
public:
	SmplFitSMPL2EdgeCeres_MV(double* V_ptr, SparseMatrix<double, ColMajor>  &dVdt_, double *dVdp_ptr, double *dVdc_ptr, double *dVds_ptr,
		SMPLModel &_mySMPL, DensePose &_DensePoseFC, char *_EdgeVertexTypeAndVisibilityFC, double *_P, int _cid, int _vid, int _pointFomat, double isigma_2D) :
		com_V_ptr(V_ptr), dVdt(dVdt_), com_dVdp_ptr(dVdp_ptr), com_dVdc_ptr(dVdc_ptr), com_dVds_ptr(dVds_ptr),
		mySMPL(_mySMPL), DensePoseFC(_DensePoseFC), EdgeVertexTypeAndVisibilityFC(_EdgeVertexTypeAndVisibilityFC), P(_P), cid(_cid), vid(_vid), pointFomat(_pointFomat), isigma(isigma_2D)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(1);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitSMPL2EdgeCeres_MV() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };
		int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];

		if (EdgeVertexTypeAndVisibilityFC[vid] == -1 || DensePoseFC.validParts[dfId] == 0)
		{
			residuals[0] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 1, 3), drdp(jacobians[1], 1, nJoints * 3), drdc(jacobians[2], 1, nShapeCoeffs), drds(jacobians[3], 1, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		int width = DensePoseFC.width, height = DensePoseFC.height, length = width * height;
		int vid3 = vid * 3, nVertices3 = nVertices * 3, nJoints3 = nJoints * 3;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_V_ptr + vid3;

		//Residuals
		double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
			numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
			denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
		double x = numX / denum, y = numY / denum;
		if (x<15 || x>width - 15 || y<15 || y>height - 15)
		{
			residuals[0] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 1, 3), drdp(jacobians[1], 1, nJoints * 3), drdc(jacobians[2], 1, nShapeCoeffs), drds(jacobians[3], 1, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		uint16_t *df = DensePoseFC.vdfield[dfId];

		int xiD = (int)(x), yiD = (int)(y), xiU = xiD + 1, yiU = yiD + 1, yiDws = yiD * width, yiUws = yiU * width;

		double xxiD = x - xiD, yyiD = y - yiD;
		double f00 = (double)df[xiD + yiDws], f01 = (double)df[xiU + yiDws], f10 = (double)df[xiD + yiUws], f11 = (double)df[xiU + yiUws];
		double a11 = (f11 - f01 - f10 + f00), a10 = (f01 - f00), a01 = (f10 - f00);
		residuals[0] = isigma * (f00 + a10 * xxiD + a01 * yyiD + a11 * xxiD*yyiD);

		if (jacobians)
		{
			double denum2 = denum * denum;
			double dudX = (P[0] * denum - P[8] * numX) / denum2, dudY = (P[1] * denum - P[9] * numX) / denum2, dudZ = (P[2] * denum - P[10] * numX) / denum2;
			double dvdX = (P[4] * denum - P[8] * numY) / denum2, dvdY = (P[5] * denum - P[9] * numY) / denum2, dvdZ = (P[6] * denum - P[10] * numY) / denum2;
			double dfdx = isigma * (a10 + a11 * yyiD), dfdy = isigma * (a01 + a11 * xxiD);

			double drdV[3] = { dfdx*dudX + dfdy * dvdX, dfdx*dudY + dfdy * dvdY, dfdx*dudZ + dfdy * dvdZ };

			Map< MatrixXdr > edrdV(drdV, 1, 3), drdt(jacobians[0], 1, 3), drdp(jacobians[1], 1, nJoints3), drdc(jacobians[2], 1, nShapeCoeffs), drds(jacobians[3], 1, 1);

			if (jacobians[0])
				drdt = edrdV * dVdt.block(vid3, 0, 3, 3);
			if (jacobians[1])
			{
				Map< MatrixXdr > edVdp(com_dVdp_ptr, nVertices3, nJoints3);
				drdp = edrdV * edVdp.block(vid3, 0, 3, nJoints3);
			}
			if (jacobians[2])
			{
				Map< MatrixXdr > edVdc(com_dVdc_ptr, nVertices3, nShapeCoeffs);
				drdc = edrdV * edVdc.block(vid3, 0, 3, nShapeCoeffs);
			}
			if (jacobians[3])
			{
				Map< MatrixXdr > edVds(com_dVds_ptr, nVertices3, 1);
				drds = edrdV * edVds.block(vid3, 0, 3, 1);
			}
		}
		return true;
	}

	SparseMatrix<double, ColMajor>  &dVdt;
	double* com_V_ptr, *com_dVdp_ptr, *com_dVdc_ptr, *com_dVds_ptr;

	double	*P, isigma;
	int cid, vid, pointFomat;
	char*EdgeVertexTypeAndVisibilityFC;
	SMPLModel &mySMPL;
	DensePose &DensePoseFC;
};
class SmplFitSMPL2EdgeCeres_MV2 :
	public ceres::CostFunction {
public:
	SmplFitSMPL2EdgeCeres_MV2(double* V_ptr, SparseMatrix<double, ColMajor>  &dVdt_, double *dVdp_ptr, double *dVdc_ptr, double *dVds_ptr, SMPLModel &_mySMPL, DensePose &_DensePoseFC,
		char *_EdgeVertexTypeAndVisibilityFC, double *_P, int _cid, int _vid, int _pointFomat, double isigma_2D) :
		com_V_ptr(V_ptr), dVdt(dVdt_), com_dVdp_ptr(dVdp_ptr), com_dVdc_ptr(dVdc_ptr), com_dVds_ptr(dVds_ptr), mySMPL(_mySMPL), DensePoseFC(_DensePoseFC),
		EdgeVertexTypeAndVisibilityFC(_EdgeVertexTypeAndVisibilityFC), P(_P), cid(_cid), vid(_vid), pointFomat(_pointFomat), isigma(isigma_2D)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(2);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitSMPL2EdgeCeres_MV2() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };
		int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];

		if (EdgeVertexTypeAndVisibilityFC[vid] == -1 || DensePoseFC.validParts[dfId] == 0)
		{
			residuals[0] = 0, residuals[1] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints * 3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		int width = DensePoseFC.width, height = DensePoseFC.height, length = width * height;
		int vid3 = vid * 3, nVertices3 = nVertices * 3, nJoints3 = nJoints * 3;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_V_ptr + vid3;

		//Residuals
		double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
			numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
			denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
		double x = numX / denum, y = numY / denum;
		if (x<15 || x>width - 15 || y<15 || y>height - 15)
		{
			residuals[0] = 0, residuals[1] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints * 3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		size_t nnId = 0;
		double smallestDist = 9e9;
		for (size_t ii = 0; ii < DensePoseFC.vEdge[dfId].size(); ii++)
		{
			double dist = pow(DensePoseFC.vEdge[dfId][ii].x - x, 2) + pow(DensePoseFC.vEdge[dfId][ii].y - y, 2);
			if (dist < smallestDist)
				nnId = ii, smallestDist = dist;
		}
		residuals[0] = isigma * (x - DensePoseFC.vEdge[dfId][nnId].x);
		residuals[1] = isigma * (y - DensePoseFC.vEdge[dfId][nnId].y);

		//point_t pt = { x, y };
		//point_t res = DensePoseFC.tress[dfId].nearest_point(pt);
		//residuals[0] = isigma * (x - res[0]);
		//residuals[1] = isigma * (y - res[1]);

		//uint16_t *df = DensePoseFC.vdfield[dfId];
		//int xiD = (int)(x), yiD = (int)(y), xiU = xiD + 1, yiU = yiD + 1, yiDws = yiD * width, yiUws = yiU * width;
		//double xxiD = x - xiD, yyiD = y - yiD;
		//double f00 = (double)df[xiD + yiDws], f01 = (double)df[xiU + yiDws], f10 = (double)df[xiD + yiUws], f11 = (double)df[xiU + yiUws];
		//double a11 = (f11 - f01 - f10 + f00), a10 = (f01 - f00), a01 = (f10 - f00);
		//residuals[0] = isigma * (f00 + a10 * xxiD + a01 * yyiD + a11 * xxiD*yyiD);

		if (jacobians)
		{
			double denum2 = denum * denum;
			double dres0dX = isigma * (P[0] * denum - P[8] * numX) / denum2, dres0dY = isigma * (P[1] * denum - P[9] * numX) / denum2, dres0dZ = isigma * (P[2] * denum - P[10] * numX) / denum2;
			double dres1dX = isigma * (P[4] * denum - P[8] * numY) / denum2, dres1dY = isigma * (P[5] * denum - P[9] * numY) / denum2, dres1dZ = isigma * (P[6] * denum - P[10] * numY) / denum2;
			double drdV[6] = { dres0dX, dres0dY, dres0dZ, dres1dX, dres1dY, dres1dZ };
			//double dfdx = isigma * (a10 + a11 * yyiD), dfdy = isigma * (a01 + a11 * xxiD);
			//double drdV[3] = { dfdx*dudX + dfdy * dvdX, dfdx*dudY + dfdy * dvdY, dfdx*dudZ + dfdy * dvdZ };

			Map< MatrixXdr > edrdV(drdV, 2, 3), drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);

			if (jacobians[0])
				drdt = edrdV * dVdt.block(vid3, 0, 3, 3);
			if (jacobians[1])
			{
				Map< MatrixXdr > edVdp(com_dVdp_ptr, nVertices3, nJoints3);
				drdp = edrdV * edVdp.block(vid3, 0, 3, nJoints3);
			}
			if (jacobians[2])
			{
				Map< MatrixXdr > edVdc(com_dVdc_ptr, nVertices3, nShapeCoeffs);
				drdc = edrdV * edVdc.block(vid3, 0, 3, nShapeCoeffs);
			}
			if (jacobians[3])
			{
				Map< MatrixXdr > edVds(com_dVds_ptr, nVertices3, 1);
				drds = edrdV * edVds.block(vid3, 0, 3, 1);
			}
		}
		return true;
	}

	SparseMatrix<double, ColMajor>  &dVdt;
	double* com_V_ptr, *com_dVdp_ptr, *com_dVdc_ptr, *com_dVds_ptr;

	double	*P, isigma;
	int cid, vid, pointFomat;
	char*EdgeVertexTypeAndVisibilityFC;
	SMPLModel &mySMPL;
	DensePose &DensePoseFC;
};
class SmplFitEdge2SMPLCeres_MV :
	public ceres::CostFunction {
public:
	SmplFitEdge2SMPLCeres_MV(double* V_ptr, SparseMatrix<double, ColMajor>  &dVdt_, double *dVdp_ptr, double *dVdc_ptr, double *dVds_ptr, SMPLModel &_mySMPL, DensePose &_DensePoseFC, vector<uchar> *_vMergedPartId2DensePosePartId,
		char *_EdgeVertexTypeAndVisibilityFC, double *_P, int _cid, int _pointFomat, Point2i &_dpPoint, int _partId, double isigma_2D) :
		com_V_ptr(V_ptr), dVdt(dVdt_), com_dVdp_ptr(dVdp_ptr), com_dVdc_ptr(dVdc_ptr), com_dVds_ptr(dVds_ptr), mySMPL(_mySMPL), DensePoseFC(_DensePoseFC), vMergedPartId2DensePosePartId(_vMergedPartId2DensePosePartId),
		EdgeVertexTypeAndVisibilityFC(_EdgeVertexTypeAndVisibilityFC), P(_P), cid(_cid), dpPoint(_dpPoint), partId(_partId), pointFomat(_pointFomat), isigma(isigma_2D)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(2);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitEdge2SMPLCeres_MV() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		int width = DensePoseFC.width, height = DensePoseFC.height, length = width * height;

		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];

		//find nearest neighhor
		int nthreads = omp_get_max_threads();

		int *tnnId = new int[nthreads];
		double *tsmallestDist = new double[nthreads];
		for (int ii = 0; ii < nthreads; ii++)
			tnnId[ii] = -1, tsmallestDist[ii] = 9e9;

		Point2f *vuv = new Point2f[nVertices];
#pragma omp parallel for schedule(dynamic, 4)
		for (int vid = 0; vid < nVertices; vid++)
		{
			if (EdgeVertexTypeAndVisibilityFC[vid] > -1)
			{
				double *X = com_V_ptr + vid * 3;
				double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
					numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
					denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
				vuv[vid].x = numX / denum, vuv[vid].y = numY / denum;
			}
		}

#pragma omp parallel for schedule(dynamic, 4)
		for (int vid = 0; vid < nVertices; vid++)
		{
			for (size_t ii = 0; ii < vMergedPartId2DensePosePartId[partId].size(); ii++)
			{
				if (EdgeVertexTypeAndVisibilityFC[vid] == vMergedPartId2DensePosePartId[partId][ii])
				{
					int threadId = omp_get_thread_num();

					double x = static_cast<double>(vuv[vid].x), y = static_cast<double>(vuv[vid].y);
					if (x<15 || x>width - 15 || y<15 || y>height - 15)
						continue;

					double dist = pow(x - dpPoint.x, 2) + pow(y - dpPoint.y, 2);
					if (dist > 2500)//50*50: too far, probably outliers (happen alot at the intersection between parts)
						continue;
					if (dist < tsmallestDist[threadId])
						tnnId[threadId] = vid, tsmallestDist[threadId] = dist;
				}
			}
		}
		delete[]vuv;

		int nnId = -1;
		double smallestDist = 9e9;
		for (int ii = 0; ii < nthreads; ii++)
			if (tsmallestDist[ii] < smallestDist)
				smallestDist = tsmallestDist[ii], nnId = tnnId[ii];
		delete[]tnnId, delete[]tsmallestDist;

		//compute resisdual and jacobian if needed
		if (nnId == -1)
		{
			residuals[0] = 0, residuals[1] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints * 3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}
		else
		{
			int vid3 = nnId * 3, nVertices3 = nVertices * 3, nJoints3 = nJoints * 3;
			double *X = com_V_ptr + vid3;

			double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
				numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
				denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
			double x = numX / denum, y = numY / denum;

			residuals[0] = isigma * (x - dpPoint.x);
			residuals[1] = isigma * (y - dpPoint.y);

			if (jacobians)
			{
				double denum2 = denum * denum;
				double dres0dX = isigma * (P[0] * denum - P[8] * numX) / denum2, dres0dY = isigma * (P[1] * denum - P[9] * numX) / denum2, dres0dZ = isigma * (P[2] * denum - P[10] * numX) / denum2;
				double dres1dX = isigma * (P[4] * denum - P[8] * numY) / denum2, dres1dY = isigma * (P[5] * denum - P[9] * numY) / denum2, dres1dZ = isigma * (P[6] * denum - P[10] * numY) / denum2;
				double drdV[6] = { dres0dX, dres0dY, dres0dZ, dres1dX, dres1dY, dres1dZ };

				Map< MatrixXdr > edrdV(drdV, 2, 3), drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);

				if (jacobians[0])
					drdt = edrdV * dVdt.block(vid3, 0, 3, 3);
				if (jacobians[1])
				{
					Map< MatrixXdr > edVdp(com_dVdp_ptr, nVertices3, nJoints3);
					drdp = edrdV * edVdp.block(vid3, 0, 3, nJoints3);
				}
				if (jacobians[2])
				{
					Map< MatrixXdr > edVdc(com_dVdc_ptr, nVertices3, nShapeCoeffs);
					drdc = edrdV * edVdc.block(vid3, 0, 3, nShapeCoeffs);
				}
				if (jacobians[3])
				{
					Map< MatrixXdr > edVds(com_dVds_ptr, nVertices3, 1);
					drds = edrdV * edVds.block(vid3, 0, 3, 1);
				}
			}
			return true;
		}
	}

	SparseMatrix<double, ColMajor>  &dVdt;
	double* com_V_ptr, *com_dVdp_ptr, *com_dVdc_ptr, *com_dVds_ptr;

	vector<uchar> *vMergedPartId2DensePosePartId;
	double	*P, isigma;
	int cid, pointFomat, partId;
	Point2i dpPoint;
	char*EdgeVertexTypeAndVisibilityFC;
	SMPLModel &mySMPL;
	DensePose &DensePoseFC;
};
class SmplFitEdge2SMPLCeres_MV2 :
	public ceres::CostFunction {
public:
	SmplFitEdge2SMPLCeres_MV2(double* V_ptr, Point2f *vUV, SparseMatrix<double, ColMajor>  &dVdt_, double *dVdp_ptr, double *dVdc_ptr, double *dVds_ptr, SMPLModel &_mySMPL, DensePose &_DensePoseFC, vector<uchar> *_vMergedPartId2DensePosePartId,
		char *_EdgeVertexTypeAndVisibilityFC, double *_P, int _cid, int _pointFomat, Point2i &_dpPoint, int _partId, double isigma_2D) :
		com_V_ptr(V_ptr), com_vUV_ptr(vUV), dVdt(dVdt_), com_dVdp_ptr(dVdp_ptr), com_dVdc_ptr(dVdc_ptr), com_dVds_ptr(dVds_ptr), mySMPL(_mySMPL), DensePoseFC(_DensePoseFC), vMergedPartId2DensePosePartId(_vMergedPartId2DensePosePartId),
		EdgeVertexTypeAndVisibilityFC(_EdgeVertexTypeAndVisibilityFC), P(_P), cid(_cid), dpPoint(_dpPoint), partId(_partId), pointFomat(_pointFomat), isigma(isigma_2D)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(2);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitEdge2SMPLCeres_MV2() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		int width = DensePoseFC.width, height = DensePoseFC.height, length = width * height;

		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];

		//find nearest neighhor
		int nthreads = omp_get_max_threads();

		int *tnnId = new int[nthreads];
		double *tsmallestDist = new double[nthreads];
		for (int ii = 0; ii < nthreads; ii++)
			tnnId[ii] = -1, tsmallestDist[ii] = 9e9;

#pragma omp parallel for schedule(dynamic, 4)
		for (int vid = 0; vid < nVertices; vid++)
		{
			for (size_t ii = 0; ii < vMergedPartId2DensePosePartId[partId].size(); ii++)
			{
				if (EdgeVertexTypeAndVisibilityFC[vid] == vMergedPartId2DensePosePartId[partId][ii])
				{
					int threadId = omp_get_thread_num();

					double x = static_cast<double>(com_vUV_ptr[vid].x), y = static_cast<double>(com_vUV_ptr[vid].y);
					if (x<15 || x>width - 15 || y<15 || y>height - 15)
						continue;

					double dist = pow(x - dpPoint.x, 2) + pow(y - dpPoint.y, 2);
					if (dist > 2500)//50*50: too far, probably outliers (happen alot at the intersection between parts)
						continue;
					if (dist < tsmallestDist[threadId])
						tnnId[threadId] = vid, tsmallestDist[threadId] = dist;
				}
			}
		}

		int nnId = -1;
		double smallestDist = 9e9;
		for (int ii = 0; ii < nthreads; ii++)
		{
			if (tsmallestDist[ii] < smallestDist)
			{
				smallestDist = tsmallestDist[ii], nnId = tnnId[ii];
			}
		}
		delete[]tnnId, delete[]tsmallestDist;

		//compute resisdual and jacobian if needed
		if (nnId == -1)
		{
			residuals[0] = 0, residuals[1] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints * 3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}
		else
		{
			int vid3 = nnId * 3, nVertices3 = nVertices * 3, nJoints3 = nJoints * 3;
			double *X = com_V_ptr + vid3;

			double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
				numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
				denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
			double x = numX / denum, y = numY / denum;

			residuals[0] = isigma * (x - dpPoint.x);
			residuals[1] = isigma * (y - dpPoint.y);

			if (jacobians)
			{
				double denum2 = denum * denum;
				double dres0dX = isigma * (P[0] * denum - P[8] * numX) / denum2, dres0dY = isigma * (P[1] * denum - P[9] * numX) / denum2, dres0dZ = isigma * (P[2] * denum - P[10] * numX) / denum2;
				double dres1dX = isigma * (P[4] * denum - P[8] * numY) / denum2, dres1dY = isigma * (P[5] * denum - P[9] * numY) / denum2, dres1dZ = isigma * (P[6] * denum - P[10] * numY) / denum2;
				double drdV[6] = { dres0dX, dres0dY, dres0dZ, dres1dX, dres1dY, dres1dZ };

				Map< MatrixXdr > edrdV(drdV, 2, 3), drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);

				if (jacobians[0])
					drdt = edrdV * dVdt.block(vid3, 0, 3, 3);
				if (jacobians[1])
				{
					Map< MatrixXdr > edVdp(com_dVdp_ptr, nVertices3, nJoints3);
					drdp = edrdV * edVdp.block(vid3, 0, 3, nJoints3);
				}
				if (jacobians[2])
				{
					Map< MatrixXdr > edVdc(com_dVdc_ptr, nVertices3, nShapeCoeffs);
					drdc = edrdV * edVdc.block(vid3, 0, 3, nShapeCoeffs);
				}
				if (jacobians[3])
				{
					Map< MatrixXdr > edVds(com_dVds_ptr, nVertices3, 1);
					drds = edrdV * edVds.block(vid3, 0, 3, 1);
				}
			}
			return true;
		}
	}

	SparseMatrix<double, ColMajor>  &dVdt;
	double* com_V_ptr, *com_dVdp_ptr, *com_dVdc_ptr, *com_dVds_ptr;
	Point2f *com_vUV_ptr;

	vector<uchar> *vMergedPartId2DensePosePartId;
	double	*P, isigma;
	int cid, pointFomat, partId;
	Point2i dpPoint;
	char*EdgeVertexTypeAndVisibilityFC;
	SMPLModel &mySMPL;
	DensePose &DensePoseFC;
};
class SmplFitSilCeres_MV :
	public ceres::CostFunction {
public:
	SmplFitSilCeres_MV(double* V_ptr, SparseMatrix<double, ColMajor>  &dVdt_, double *dVdp_ptr, double *dVdc_ptr, double *dVds_ptr,
		SMPLModel &_mySMPL, DensePose &_DensePoseFC, double *_P, int _cid, int _vid, int _pointFomat, double isigma_2D) :
		com_V_ptr(V_ptr), dVdt(dVdt_), com_dVdp_ptr(dVdp_ptr), com_dVdc_ptr(dVdc_ptr), com_dVds_ptr(dVds_ptr),
		mySMPL(_mySMPL), DensePoseFC(_DensePoseFC), P(_P), cid(_cid), vid(_vid), pointFomat(_pointFomat), isigma(isigma_2D)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(1);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitSilCeres_MV() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		int width = DensePoseFC.width, height = DensePoseFC.height, length = width * height;
		int vid3 = vid * 3, nVertices3 = nVertices * 3, nJoints3 = nJoints * 3;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_V_ptr + vid3;

		//Residuals
		double numX = P[0] * X[0] + P[1] * X[1] + P[2] * X[2] + P[3],
			numY = P[4] * X[0] + P[5] * X[1] + P[6] * X[2] + P[7],
			denum = P[8] * X[0] + P[9] * X[1] + P[10] * X[2] + P[11];
		double x = numX / denum, y = numY / denum;
		if (x<15 || x>width - 15 || y<15 || y>height - 15)
		{
			residuals[0] = 0; //so that the loss is not affected when ceres considers if the step is successful. Zeros jacobian already takes care of the rest
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 1, 3), drdp(jacobians[1], 1, nJoints * 3), drdc(jacobians[2], 1, nShapeCoeffs), drds(jacobians[3], 1, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		uint16_t *df = DensePoseFC.vdfield[14]; //free-space term, penalyzing points projected outside the sil

		int xiD = (int)(x), yiD = (int)(y), xiU = xiD + 1, yiU = yiD + 1, yiDws = yiD * width, yiUws = yiU * width;

		double xxiD = x - xiD, yyiD = y - yiD;
		double f00 = (double)df[xiD + yiDws], f01 = (double)df[xiU + yiDws], f10 = (double)df[xiD + yiUws], f11 = (double)df[xiU + yiUws];
		double a11 = (f11 - f01 - f10 + f00), a10 = (f01 - f00), a01 = (f10 - f00);
		residuals[0] = isigma * (f00 + a10 * xxiD + a01 * yyiD + a11 * xxiD*yyiD);

		if (jacobians)
		{
			double denum2 = denum * denum;
			double dudX = (P[0] * denum - P[8] * numX) / denum2, dudY = (P[1] * denum - P[9] * numX) / denum2, dudZ = (P[2] * denum - P[10] * numX) / denum2;
			double dvdX = (P[4] * denum - P[8] * numY) / denum2, dvdY = (P[5] * denum - P[9] * numY) / denum2, dvdZ = (P[6] * denum - P[10] * numY) / denum2;
			double dfdx = isigma * (a10 + a11 * yyiD), dfdy = isigma * (a01 + a11 * xxiD);

			double drdV[3] = { dfdx*dudX + dfdy * dvdX, dfdx*dudY + dfdy * dvdY, dfdx*dudZ + dfdy * dvdZ };

			Map< MatrixXdr > edrdV(drdV, 1, 3), drdt(jacobians[0], 1, 3), drdp(jacobians[1], 1, nJoints3), drdc(jacobians[2], 1, nShapeCoeffs), drds(jacobians[3], 1, 1);

			if (jacobians[0])
				drdt = edrdV * dVdt.block(vid3, 0, 3, 3);
			if (jacobians[1])
			{
				Map< MatrixXdr > edVdp(com_dVdp_ptr, nVertices3, nJoints3);
				drdp = edrdV * edVdp.block(vid3, 0, 3, nJoints3);
			}
			if (jacobians[2])
			{
				Map< MatrixXdr > edVdc(com_dVdc_ptr, nVertices3, nShapeCoeffs);
				drdc = edrdV * edVdc.block(vid3, 0, 3, nShapeCoeffs);
			}
			if (jacobians[3])
			{
				Map< MatrixXdr > edVds(com_dVds_ptr, nVertices3, 1);
				drds = edrdV * edVds.block(vid3, 0, 3, 1);
			}
		}
		return true;
	}

	SparseMatrix<double, ColMajor>  &dVdt;
	double* com_V_ptr, *com_dVdp_ptr, *com_dVdc_ptr, *com_dVds_ptr;

	double	*P, isigma;
	int cid, vid, pointFomat;
	SMPLModel &mySMPL;
	DensePose &DensePoseFC;
};
class SmplFitCOCO2DCeres_MV : public ceres::CostFunction
{
public:
	SmplFitCOCO2DCeres_MV(double* V_ptr, double* J_ptr, double *dJdt_ptr, double *dJdp_ptr, double *dJdc_ptr, double *dJds_ptr,
		SMPLModel &mySMPL, double *P, int Jid, Point2d &detection, int pointFomat, double isigma) :
		com_V_ptr(V_ptr), com_J_ptr(J_ptr), com_dJdt_ptr(dJdt_ptr), com_dJdp_ptr(dJdp_ptr), com_dJdc_ptr(dJdc_ptr), com_dJds_ptr(dJds_ptr),
		mySMPL_(mySMPL), P_(P), Jid_(Jid), detection_(detection), pointFomat_(pointFomat), isigma(isigma)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(2);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitCOCO2DCeres_MV() {	}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;
		//int nCorrespond3 = pointFomat_ == 17 ? 51 : (18 ? 42 : 72), nJoints3 = nJoints * 3, Jid3 = Jid_ * 3;
		int nCorrespond3 = pointFomat_ == 18 ? 42 : 72, nJoints3 = nJoints * 3, Jid3 = Jid_ * 3;

		if (ceres::abs(detection_.x) + ceres::abs(detection_.y) < 1e-16)
		{
			residuals[0] = 0, residuals[1] = 0;
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints * 3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_J_ptr + Jid3, drdJ[6];
		if (jacobians)
		{
			PinholeReprojectionErrorSimple_PointOnly *reproj = new PinholeReprojectionErrorSimple_PointOnly(P_, detection_.x, detection_.y, isigma);
			ceres::AutoDiffCostFunction<PinholeReprojectionErrorSimple_PointOnly, 2, 3> projection_(reproj);
			double *pparameters[1] = { X }, *pjacobians[1] = { drdJ };
			projection_.Evaluate(pparameters, residuals, pjacobians); //this jacobian has implicitly considered  iscale

			Map< MatrixXdr > edrdJ(drdJ, 2, 3), edJdt(com_dJdt_ptr, nCorrespond3, 3), edJdp(com_dJdp_ptr, nCorrespond3, nJoints3), edJdc(com_dJdc_ptr, nCorrespond3, nShapeCoeffs), edJds(com_dJds_ptr, nCorrespond3, 1);
			Map< MatrixXdr > drdt(jacobians[0], 2, 3), drdp(jacobians[1], 2, nJoints * 3), drdc(jacobians[2], 2, nShapeCoeffs), drds(jacobians[3], 2, 1);
			if (jacobians[0])
				drdt = edrdJ * edJdt.block(Jid3, 0, 3, 3);
			if (jacobians[1])
				drdp = edrdJ * edJdp.block(Jid3, 0, 3, nJoints3);
			if (jacobians[2])
				drdc = edrdJ * edJdc.block(Jid3, 0, 3, nShapeCoeffs);
			if (jacobians[3])
				drds = edrdJ * edJds.block(Jid3, 0, 3, 1);
		}
		else
		{
			double numX = P_[0] * X[0] + P_[1] * X[1] + P_[2] * X[2] + P_[3],
				numY = P_[4] * X[0] + P_[5] * X[1] + P_[6] * X[2] + P_[7],
				denum = P_[8] * X[0] + P_[9] * X[1] + P_[10] * X[2] + P_[11];

			residuals[0] = (numX / denum - detection_.x), residuals[1] = (numY / denum - detection_.y);
			residuals[0] *= isigma, residuals[1] *= isigma;
		}

		return true;
	}

	double* com_V_ptr, *com_J_ptr, *com_dJdt_ptr, *com_dJdp_ptr, *com_dJdc_ptr, *com_dJds_ptr;

	double	*P_, isigma;
	int Jid_, pointFomat_;
	Point2d detection_;
	SMPLModel &mySMPL_;
};
class SmplFitDensePoseCeres_MV :
	public ceres::CostFunction {
public:
	SmplFitDensePoseCeres_MV(double* V_ptr, SparseMatrix<double, ColMajor>  &dVdt_, double *dVdp_ptr, double *dVdc_ptr, double *dVds_ptr,
		SMPLModel &_mySMPL, CameraData &_CamI, int _vid, Point2d &_detection, int _pointFomat, double isigma_2D) :
		com_V_ptr(V_ptr), dVdt(dVdt_), com_dVdp_ptr(dVdp_ptr), com_dVdc_ptr(dVdc_ptr), com_dVds_ptr(dVds_ptr),
		mySMPL(_mySMPL), CamI(_CamI), vid(_vid), detection(_detection), pointFomat(_pointFomat), isigma(isigma_2D)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(2);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitDensePoseCeres_MV() {}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;
		int vid3 = vid * 3, nVertices3 = nVertices * 3, nJoints3 = nJoints * 3;

		double *X = com_V_ptr + vid3;
		double numX = CamI.P[0] * X[0] + CamI.P[1] * X[1] + CamI.P[2] * X[2] + CamI.P[3],
			numY = CamI.P[4] * X[0] + CamI.P[5] * X[1] + CamI.P[6] * X[2] + CamI.P[7],
			denum = CamI.P[8] * X[0] + CamI.P[9] * X[1] + CamI.P[10] * X[2] + CamI.P[11];

		residuals[0] = isigma * (numX / denum - detection.x), residuals[1] = isigma * (numY / denum - detection.y);

		if (jacobians)
		{
			double drdV[6], denum2 = denum * denum / isigma;
			drdV[0] = (CamI.P[0] * denum - CamI.P[8] * numX) / denum2, drdV[1] = (CamI.P[1] * denum - CamI.P[9] * numX) / denum2, drdV[2] = (CamI.P[2] * denum - CamI.P[10] * numX) / denum2;
			drdV[3] = (CamI.P[4] * denum - CamI.P[8] * numY) / denum2, drdV[4] = (CamI.P[5] * denum - CamI.P[9] * numY) / denum2, drdV[5] = (CamI.P[6] * denum - CamI.P[10] * numY) / denum2;

			Map< MatrixXdr > edrdV(drdV, 2, 3);
			if (jacobians[0])
			{
				Map< MatrixXdr > drdt(jacobians[0], 2, 3);
				drdt = edrdV * dVdt.block(vid3, 0, 3, 3);
			}
			if (jacobians[1])
			{
				Map< MatrixXdr > drdp(jacobians[1], 2, nJoints3), edVdp(com_dVdp_ptr, nVertices3, nJoints3);
				drdp = edrdV * edVdp.block(vid3, 0, 3, nJoints3);
			}
			if (jacobians[2])
			{
				Map< MatrixXdr > drdc(jacobians[2], 2, nShapeCoeffs), edVdc(com_dVdc_ptr, nVertices3, nShapeCoeffs);
				drdc = edrdV * edVdc.block(vid3, 0, 3, nShapeCoeffs);
			}
			if (jacobians[3])
			{
				Map< MatrixXdr >  drds(jacobians[3], 2, 1), edVds(com_dVds_ptr, nVertices3, 1);
				drds = edrdV * edVds.block(vid3, 0, 3, 1);
			}
		}

		return true;
	}

	SparseMatrix<double, ColMajor>  &dVdt;
	double* com_V_ptr, *com_dVdp_ptr, *com_dVdc_ptr, *com_dVds_ptr;

	double	isigma;
	int vid, pointFomat;
	SMPLModel &mySMPL;
	Point2d detection;
	CameraData &CamI;
};
class Smpl2OPKeypointsTemporalReg :
	public ceres::CostFunction {
public:
	Smpl2OPKeypointsTemporalReg(double* _V0_ptr, double* _J0_ptr, double *_dV0dp_ptr, double *_dV0dc_ptr, double *_dV0ds_ptr,
		double* _V1_ptr, double* _J1_ptr, double *_dV1dp_ptr, double *_dV1dc_ptr, double *_dV1ds_ptr,
		SMPLModel &mySMPL, SparseMatrix<double, ColMajor>  &dVdt_, int _pointFormat, double  _ts0, double  _ts1, double _isigma) :
		V0_ptr(_V0_ptr), J0_ptr(_J0_ptr), dV0dp_ptr(_dV0dp_ptr), dV0dc_ptr(_dV0dc_ptr), dV0ds_ptr(_dV0ds_ptr),
		V1_ptr(_V1_ptr), J1_ptr(_J1_ptr), dV1dp_ptr(_dV1dp_ptr), dV1dc_ptr(_dV1dc_ptr), dV1ds_ptr(_dV1ds_ptr),
		mySMPL_(mySMPL), dVdt(dVdt_), skeletonPointFormat(_pointFormat), ts0(_ts0), ts1(_ts1), isigma(_isigma)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		V_dif_ptr = new double[nVertices * 3];
		J_regl = skeletonPointFormat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;

		nResiduals_ = skeletonPointFormat == 18 ? 14 * 3 : 24 * 3; //use OpenPose joints
		CostFunction::set_num_residuals(nResiduals_);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(1); // scale 1
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(3); // Translation 0
		parameter_block_sizes->push_back(nJoints * 3); // Pose 0
		parameter_block_sizes->push_back(3); // Translation 1
		parameter_block_sizes->push_back(nJoints * 3); // Pose 1		
	}
	virtual ~Smpl2OPKeypointsTemporalReg() {}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		const double *s = parameters[0], *c = parameters[1], *t0 = parameters[2], *p0 = parameters[3], *t1 = parameters[4], *p1 = parameters[5];

		Map< const Vector3d > t0_vec(t0), t1_vec(t1);
		Map< const VectorXd > pose0_vec(p0, nJoints * 3), pose1_vec(p1, nJoints * 3);

		Map< VectorXd > J0_vec(J0_ptr, 72), J1_vec(J1_ptr, 72);
		Map< VectorXd > V0_vec(V0_ptr, nVertices * 3), V1_vec(V1_ptr, nVertices * 3), V_vec_dif(V_dif_ptr, nVertices * 3);

		if (jacobians && jacobians[0])
			V_vec_dif = V0_vec - V1_vec;

		double temp = isigma / sqrt(abs(ts1 - ts0)*nResiduals_); //scale sum square of residuals to 1
		for (int ii = 0; ii < nResiduals_ / 3; ii++)
		{
			int ii3 = ii * 3;
			double tX0 = J0_ptr[ii3], tY0 = J0_ptr[ii3 + 1], tZ0 = J0_ptr[ii3 + 2];
			double tX1 = J1_ptr[ii3], tY1 = J1_ptr[ii3 + 1], tZ1 = J1_ptr[ii3 + 2];
			residuals[ii3] = temp * (tX0 - tX1), residuals[ii3 + 1] = temp * (tY0 - tY1), residuals[ii3 + 2] = temp * (tZ0 - tZ1); //(v/sig_v)^2*dt = (dx/dt/sig_v)^2*dt = (dx/sig_v)^2/dt --> res = dx/sig_v / sqrt(dt)
		}

		if (jacobians)
		{
			Map< MatrixXdr >  dV0dp(dV0dp_ptr, nVertices * 3, nJoints * 3), dV0dc(dV0dc_ptr, nVertices * 3, nShapeCoeffs), dV1dp(dV1dp_ptr, nVertices * 3, nJoints * 3), dV1dc(dV1dc_ptr, nVertices * 3, nShapeCoeffs);

			if (jacobians[0])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drds(jacobians[0], nResiduals_, 1);
				drds = temp * J_regl*V_vec_dif;
			}
			if (jacobians[1])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdc(jacobians[1], nResiduals_, nShapeCoeffs);
				drdc = temp * J_regl*(dV0dc - dV1dc)*s[0];
			}
			if (jacobians[2])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt0(jacobians[2], nResiduals_, 3);
				drdt0 = temp * J_regl*dVdt;
			}
			if (jacobians[3])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp0(jacobians[3], nResiduals_, nJoints * 3);
				drdp0 = temp * J_regl*dV0dp*s[0];
			}
			if (jacobians[4])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt1(jacobians[4], nResiduals_, 3);
				drdt1 = -temp * J_regl*dVdt;
			}
			if (jacobians[5])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp1(jacobians[5], nResiduals_, nJoints * 3);
				drdp1 = -temp * J_regl*dV1dp*s[0];
			}
		}

		return true;
	}

	double *V0_ptr, *J0_ptr, *dV0dp_ptr, *dV0dc_ptr, *dV0ds_ptr;
	double *V1_ptr, *J1_ptr, *dV1dp_ptr, *dV1dc_ptr, *dV1ds_ptr;
	double* V_dif_ptr;

	int skeletonPointFormat;
	double ts0, ts1, isigma;
	SMPLModel &mySMPL_;
	SparseMatrix<double, ColMajor> J_regl, &dVdt;
	int  nResiduals_;
};
class aSmplKeypointsTemporalReg :
	public ceres::CostFunction {
public:
	aSmplKeypointsTemporalReg(double* _V0_ptr, double* _J0_ptr, double *_dV0dp_ptr, double *_dV0dc_ptr, double *_dV0ds_ptr,
		double* _V1_ptr, double* _J1_ptr, double *_dV1dp_ptr, double *_dV1dc_ptr, double *_dV1ds_ptr,
		SMPLModel &mySMPL, SparseMatrix<double, ColMajor>  &dVdt_, double  _ts0, double  _ts1, double _isigma) :
		V0_ptr(_V0_ptr), J0_ptr(_J0_ptr), dV0dp_ptr(_dV0dp_ptr), dV0dc_ptr(_dV0dc_ptr), dV0ds_ptr(_dV0ds_ptr),
		V1_ptr(_V1_ptr), J1_ptr(_J1_ptr), dV1dp_ptr(_dV1dp_ptr), dV1dc_ptr(_dV1dc_ptr), dV1ds_ptr(_dV1ds_ptr),
		mySMPL_(mySMPL), dVdt(dVdt_), ts0(_ts0), ts1(_ts1), isigma(_isigma)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;
		int naJoints3 = naJoints * 3, nJoints3 = nJoints * 3;

		diag = new double[naJoints3 * 3];
		dJ = new double[naJoints3 * 3];
		V_dif_ptr = new double[nVertices * 3];
		J_regl = mySMPL.J_regl_abigl_;

		nResiduals_ = naJoints3;
		CostFunction::set_num_residuals(nResiduals_);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(1); // scale 1
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(3); // Translation 0
		parameter_block_sizes->push_back(nJoints * 3); // Pose 0
		parameter_block_sizes->push_back(3); // Translation 1
		parameter_block_sizes->push_back(nJoints * 3); // Pose 1		
	}
	virtual ~aSmplKeypointsTemporalReg()
	{
		delete[]diag, delete[]dJ, delete[]V_dif_ptr;
	}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;
		int naJoints3 = naJoints * 3, nJoints3 = nJoints * 3;

		const double *s = parameters[0], *c = parameters[1], *t0 = parameters[2], *p0 = parameters[3], *t1 = parameters[4], *p1 = parameters[5];

		Map< const Vector3d > t0_vec(t0), t1_vec(t1);
		Map< const VectorXd > pose0_vec(p0, nJoints * 3), pose1_vec(p1, nJoints * 3);

		Map< VectorXd > J0_vec(J0_ptr, 72), J1_vec(J1_ptr, 72);
		Map< VectorXd > V0_vec(V0_ptr, nVertices * 3), V1_vec(V1_ptr, nVertices * 3), V_vec_dif(V_dif_ptr, nVertices * 3);

		if (jacobians && jacobians[0])
			V_vec_dif = V0_vec - V1_vec;

		for (int ii = 0; ii < naJoints3; ii++)
			dJ[ii] = J1_ptr[ii] - J0_ptr[ii];

		double temp = isigma / sqrt(abs(ts1 - ts0)*naJoints3); //also normalize sum of all res2 to 1
		for (int ii = 0; ii < naJoints3 / 3; ii++)
		{
			int ii3 = ii * 3;
			double tX0 = J0_ptr[ii3], tY0 = J0_ptr[ii3 + 1], tZ0 = J0_ptr[ii3 + 2];
			double tX1 = J1_ptr[ii3], tY1 = J1_ptr[ii3 + 1], tZ1 = J1_ptr[ii3 + 2];

			//(v/sig_v)^2*dt = (dx/dt/sig_v)^2*dt = (dx/sig_v)^2/dt --> res = dx/sig_v / sqrt(dt)
			diag[ii3] = temp * mySMPL_.Mosh_asmpl_J_istd[ii], diag[ii3 + 1] = diag[ii3], diag[ii3 + 2] = diag[ii3];
			residuals[ii3] = diag[ii3] * (tX0 - tX1), residuals[ii3 + 1] = diag[ii3 + 1] * (tY0 - tY1), residuals[ii3 + 2] = diag[ii3 + 2] * (tZ0 - tZ1); //weight less for more mobility joint (larger sigma)
		}

		Map<VectorXd> ediag(diag, naJoints3);
		if (jacobians)
		{
			Map< MatrixXdr >   dV0dp(dV0dp_ptr, nVertices * 3, nJoints * 3), dV0dc(dV0dc_ptr, nVertices * 3, nShapeCoeffs), dV1dp(dV1dp_ptr, nVertices * 3, nJoints * 3), dV1dc(dV1dc_ptr, nVertices * 3, nShapeCoeffs);

			if (jacobians[0])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drds(jacobians[0], nResiduals_, 1);
				drds = ediag.asDiagonal()*J_regl*V_vec_dif;
			}
			if (jacobians[1])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdc(jacobians[1], nResiduals_, nShapeCoeffs);
				drdc = ediag.asDiagonal()*J_regl*(dV0dc - dV1dc)*s[0];
			}
			if (jacobians[2])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt0(jacobians[2], nResiduals_, 3);
				drdt0 = ediag.asDiagonal()* J_regl*dVdt;
			}
			if (jacobians[3])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp0(jacobians[3], nResiduals_, nJoints3);
				drdp0 = ediag.asDiagonal()*J_regl*dV0dp*s[0];
			}
			if (jacobians[4])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt1(jacobians[4], nResiduals_, 3);
				drdt1 = -1.0* ediag.asDiagonal()* J_regl*dVdt;
			}
			if (jacobians[5])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp1(jacobians[5], nResiduals_, nJoints3);
				drdp1 = -1.0* ediag.asDiagonal()* J_regl*dV1dp*s[0];
			}
		}

		return true;
	}

	double *V0_ptr, *J0_ptr, *dV0dp_ptr, *dV0dc_ptr, *dV0ds_ptr;
	double *V1_ptr, *J1_ptr, *dV1dp_ptr, *dV1dc_ptr, *dV1ds_ptr;
	double* V_dif_ptr, *dJ;

	double ts0, ts1, isigma;
	SMPLModel &mySMPL_;
	SparseMatrix<double, ColMajor> J_regl, &dVdt;
	int  nResiduals_;

	double *diag;
};
class SmplVerticesTemporalReg :
	public ceres::CostFunction {
public:
	SmplVerticesTemporalReg(double* _V0_ptr, double* _J0_ptr, double *_dV0dp_ptr, double *_dV0dc_ptr, double *_dV0ds_ptr,
		double* _V1_ptr, double* _J1_ptr, double *_dV1dp_ptr, double *_dV1dc_ptr, double *_dV1ds_ptr,
		SMPLModel &mySMPL, SparseMatrix<double, ColMajor>  &dVdt_, int _pointFormat, double  _ts0, double  _ts1, double _isigma) :
		V0_ptr(_V0_ptr), J0_ptr(_J0_ptr), dV0dp_ptr(_dV0dp_ptr), dV0dc_ptr(_dV0dc_ptr), dV0ds_ptr(_dV0ds_ptr),
		V1_ptr(_V1_ptr), J1_ptr(_J1_ptr), dV1dp_ptr(_dV1dp_ptr), dV1dc_ptr(_dV1dc_ptr), dV1ds_ptr(_dV1ds_ptr),
		mySMPL_(mySMPL), dVdt(dVdt_), skeletonPointFormat(_pointFormat), ts0(_ts0), ts1(_ts1), isigma(_isigma)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		V_dif_ptr = new double[nVertices * 3];

		nResiduals_ = nVertices * 3;
		CostFunction::set_num_residuals(nResiduals_);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(1); // scale 1
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(3); // Translation 0
		parameter_block_sizes->push_back(nJoints * 3); // Pose 0
		parameter_block_sizes->push_back(3); // Translation 1
		parameter_block_sizes->push_back(nJoints * 3); // Pose 1		
	}
	virtual ~SmplVerticesTemporalReg() {}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		const double *s = parameters[0], *c = parameters[1], *t0 = parameters[2], *p0 = parameters[3], *t1 = parameters[4], *p1 = parameters[5];

		Map< const Vector3d > t0_vec(t0), t1_vec(t1);
		Map< const VectorXd > pose0_vec(p0, nJoints * 3), pose1_vec(p1, nJoints * 3);
		Map< VectorXd > V0_vec(V0_ptr, nVertices * 3), V1_vec(V1_ptr, nVertices * 3), V_vec_dif(V_dif_ptr, nVertices * 3);

		if (jacobians && jacobians[0])
			V_vec_dif = V0_vec - V1_vec;

		double temp = isigma / sqrt(abs(ts1 - ts0)*nResiduals_); //scale sum square of residuals to 1
		for (int ii = 0; ii < nResiduals_ / 3; ii++)
		{
			int ii3 = ii * 3;
			double tX0 = V0_ptr[ii3], tY0 = V0_ptr[ii3 + 1], tZ0 = V0_ptr[ii3 + 2];
			double tX1 = V1_ptr[ii3], tY1 = V1_ptr[ii3 + 1], tZ1 = V1_ptr[ii3 + 2];
			residuals[ii3] = temp * (tX0 - tX1), residuals[ii3 + 1] = temp * (tY0 - tY1), residuals[ii3 + 2] = temp * (tZ0 - tZ1); //(v/sig_v)^2*dt = (dx/dt/sig_v)^2*dt = (dx/sig_v)^2/dt --> res = dx/sig_v / sqrt(dt)
		}

		if (jacobians)
		{
			Map< MatrixXdr >  dV0dp(dV0dp_ptr, nVertices * 3, nJoints * 3), dV0dc(dV0dc_ptr, nVertices * 3, nShapeCoeffs), dV1dp(dV1dp_ptr, nVertices * 3, nJoints * 3), dV1dc(dV1dc_ptr, nVertices * 3, nShapeCoeffs);

			if (jacobians[0])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drds(jacobians[0], nResiduals_, 1);
				drds = temp * V_vec_dif;
			}
			if (jacobians[1])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdc(jacobians[1], nResiduals_, nShapeCoeffs);
				drdc = temp * (dV0dc - dV1dc)*s[0];
			}
			if (jacobians[2])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt0(jacobians[2], nResiduals_, 3);
				drdt0 = temp * dVdt;
			}
			if (jacobians[3])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp0(jacobians[3], nResiduals_, nJoints * 3);
				drdp0 = temp * dV0dp*s[0];
			}
			if (jacobians[4])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt1(jacobians[4], nResiduals_, 3);
				drdt1 = -temp * dVdt;
			}
			if (jacobians[5])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp1(jacobians[5], nResiduals_, nJoints * 3);
				drdp1 = -temp * dV1dp*s[0];
			}
		}

		return true;
	}

	double *V0_ptr, *J0_ptr, *dV0dp_ptr, *dV0dc_ptr, *dV0ds_ptr;
	double *V1_ptr, *J1_ptr, *dV1dp_ptr, *dV1dc_ptr, *dV1ds_ptr;
	double* V_dif_ptr;

	int skeletonPointFormat;
	double ts0, ts1, isigma;
	SMPLModel &mySMPL_;
	SparseMatrix<double, ColMajor> &dVdt;
	int  nResiduals_;
};
class SmplFitCOCO3DCeres : public ceres::CostFunction
{
public:
	SmplFitCOCO3DCeres(double* V_ptr, double* J_ptr, double *dJdt_ptr, double *dJdp_ptr, double *dJdc_ptr, double *dJds_ptr,
		SMPLModel &mySMPL, int Jid, Point3d &detection, int pointFomat, double isigma) :
		com_V_ptr(V_ptr), com_J_ptr(J_ptr), com_dJdt_ptr(dJdt_ptr), com_dJdp_ptr(dJdp_ptr), com_dJdc_ptr(dJdc_ptr), com_dJds_ptr(dJds_ptr),
		mySMPL_(mySMPL), Jid_(Jid), detection_(detection), pointFomat_(pointFomat), isigma(isigma) {
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(3);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplFitCOCO3DCeres() {}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;
		//int nCorrespond3 = pointFomat_ == 17 ? 51 : (18 ? 42 : 72), nJoints3 = nJoints * 3, Jid3 = Jid_ * 3;
		int nCorrespond3 = pointFomat_ == 18 ? 42 : 72, nJoints3 = nJoints * 3, Jid3 = Jid_ * 3;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_J_ptr + Jid3;

		if (ceres::abs(detection_.x) + ceres::abs(detection_.y) + ceres::abs(detection_.z) < 1e-16)
		{
			residuals[0] = 0, residuals[1] = 0, residuals[2] = 0;
			if (jacobians)
			{
				Map< MatrixXdr > drdt(jacobians[0], 3, 3), drdp(jacobians[1], 3, nJoints * 3), drdc(jacobians[2], 3, nShapeCoeffs), drds(jacobians[3], 3, 1);
				if (jacobians[0])
					drdt.setZero();
				if (jacobians[1])
					drdp.setZero();
				if (jacobians[2])
					drdc.setZero();
				if (jacobians[3])
					drds.setZero();
			}
			return true;
		}

		residuals[0] = (X[0] - detection_.x), residuals[1] = (X[1] - detection_.y), residuals[2] = X[2] - detection_.z;
		residuals[0] *= isigma, residuals[1] *= isigma, residuals[2] *= isigma;

		if (jacobians)
		{
			Map< MatrixXdr > edJdt(com_dJdt_ptr, nCorrespond3, 3), edJdp(com_dJdp_ptr, nCorrespond3, nJoints3), edJdc(com_dJdc_ptr, nCorrespond3, nShapeCoeffs), edJds(com_dJds_ptr, nCorrespond3, 1);
			Map< MatrixXdr > drdt(jacobians[0], 3, 3), drdp(jacobians[1], 3, nJoints * 3), drdc(jacobians[2], 3, nShapeCoeffs), drds(jacobians[3], 3, 1);
			if (jacobians[0])
				drdt = isigma * edJdt.block(Jid3, 0, 3, 3);
			if (jacobians[1])
				drdp = isigma * edJdp.block(Jid3, 0, 3, nJoints3);
			if (jacobians[2])
				drdc = isigma * edJdc.block(Jid3, 0, 3, nShapeCoeffs);
			if (jacobians[3])
				drds = isigma * edJds.block(Jid3, 0, 3, 1);
		}

		/*const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		Map< const Vector3d > t_vec(t);
		Map< const VectorXd > pose_vec(p, nJoints * 3);

		Map< VectorXd > V_vec(V_ptr, nVertices * 3);
		MatrixXdr dVdp(nVertices * 3, nJoints * 3), dVdc(nVertices * 3, nShapeCoeffs);

		if (jacobians && (jacobians[0] || jacobians[1] || jacobians[2]))
			reconstruct(mySMPL_, c, p, V_ptr, dVdc, dVdp);
		else
			reconstruct(mySMPL_, c, p, V_ptr);

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_J_ptr + Jid3, drdJ[6];

		MatrixXdr  dVds;
		if (jacobians && jacobians[3])
			dVds = V_vec;

		SparseMatrix<double, ColMajor> & J_reg = pointFomat == 18 ? mySMPL_.J_regl_14_bigl_col_ : mySMPL_.J_regl_25_bigl_col_;
		V_vec = V_vec*s[0] + dVdt_ * t_vec;//dVdt_ = kron(one(nV, 1), eye(3))
		VectorXd J = J_reg * V_vec;

		for (int ii = 0; ii < nCorrespond_; ii++)
		{
			mask[ii] = 0;
			int iSMPL = SMPL2Detection[2 * ii], iDetection = SMPL2Detection[2 * ii + 1];
			double tX = J(iSMPL * 3), tY = J(iSMPL * 3 + 1), tZ = J(iSMPL * 3 + 2);
			double oX = detections_.pt3D[iDetection].x, oY = detections_.pt3D[iDetection].y, oZ = detections_.pt3D[iDetection].z;
			if (abs(oX) + abs(oY) + abs(oZ) < 1e-16)
				residuals[ii * 3] = 0, residuals[ii * 3 + 1] = 0, residuals[ii * 3 + 2] = 0, mask[ii] = 1;
			else
				residuals[ii * 3] = (tX - oX)*isigma, residuals[ii * 3 + 1] = (tY - oY)*isigma, residuals[ii * 3 + 2] = (tZ - oZ)*isigma;
		}

		if (jacobians)
		{
			if (jacobians[0])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdt(jacobians[0], nResiduals_, 3);
				drdt = J_reg*dVdt_*isigma;
				for (int ii = 0; ii < nCorrespond_; ii++)
					if (mask[ii])
						drdt.middleRows<3>(3 * ii).setZero(); //every invalid point affects 3 rows (XYZ)
			}
			if (jacobians[1])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdp(jacobians[1], nResiduals_, nJoints * 3);
				drdp = J_reg*dVdp*isigma;
				for (int ii = 0; ii < nCorrespond_; ii++)
					if (mask[ii])
						drdp.middleRows<3>(3 * ii).setZero();
			}
			if (jacobians[2])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drdc(jacobians[2], nResiduals_, nShapeCoeffs);
				drdc = J_reg*dVdc*isigma;
				for (int ii = 0; ii < nCorrespond_; ii++)
					if (mask[ii])
						drdc.middleRows<3>(3 * ii).setZero();
			}
			if (jacobians[3])
			{
				Map< Matrix<double, Dynamic, Dynamic, RowMajor> > drds(jacobians[3], nResiduals_, 1);
				drds = J_reg * dVds*isigma;
				for (int ii = 0; ii < nCorrespond_; ii++)
					if (mask[ii])
						drds.middleRows<3>(3 * ii).setZero();
			}
		}*/

		return true;
	}

	double* com_V_ptr, *com_J_ptr, *com_dJdt_ptr, *com_dJdp_ptr, *com_dJdc_ptr, *com_dJds_ptr;

	double	isigma;
	int Jid_, pointFomat_;
	Point3d detection_;
	SMPLModel &mySMPL_;
};
class SmplClampGroundCeres : public ceres::CostFunction
{
public:
	SmplClampGroundCeres(double* V_ptr, double* J_ptr, double *dJdt_ptr, double *dJdp_ptr, double *dJdc_ptr, double *dJds_ptr, SMPLModel &mySMPL, int Jid, double* plane, int pointFomat, double isigma) :
		com_V_ptr(V_ptr), com_J_ptr(J_ptr), com_dJdt_ptr(dJdt_ptr), com_dJdp_ptr(dJdp_ptr), com_dJdc_ptr(dJdc_ptr), com_dJds_ptr(dJds_ptr), mySMPL_(mySMPL), Jid_(Jid), plane_(plane), pointFomat_(pointFomat), isigma(isigma)
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

		CostFunction::set_num_residuals(1);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(3); // Translation
		parameter_block_sizes->push_back(nJoints * 3); // Pose
		parameter_block_sizes->push_back(nShapeCoeffs); // Shape coefficients
		parameter_block_sizes->push_back(1); // scale
	}
	virtual ~SmplClampGroundCeres() {}

	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;
		//int nCorrespond3 = pointFomat_ == 17 ? 51 : (18 ? 42 : 72), nJoints3 = nJoints * 3, Jid3 = Jid_ * 3;
		int nCorrespond3 = pointFomat_ == 18 ? 42 : 72, nJoints3 = nJoints * 3, Jid3 = Jid_ * 3;

		const double * t = parameters[0], *p = parameters[1], *c = parameters[2], *s = parameters[3];
		double *X = com_J_ptr + Jid3;

		double denum = sqrt(plane_[0] * plane_[0] + plane_[1] * plane_[1] + plane_[2] * plane_[2] + 1e-16);
		residuals[0] = (X[0] * plane_[0] + X[1] * plane_[1] + X[2] * plane_[2] + 1.0) / denum;
		residuals[0] *= isigma;

		if (jacobians)
		{
			double drdJ[3] = { plane_[0] / denum, plane_[1] / denum, plane_[2] / denum };

			Map< MatrixXdr > edrdJ(drdJ, 1, 3), edJdt(com_dJdt_ptr, nCorrespond3, 3), edJdp(com_dJdp_ptr, nCorrespond3, nJoints3), edJdc(com_dJdc_ptr, nCorrespond3, nShapeCoeffs), edJds(com_dJds_ptr, nCorrespond3, 1);
			Map< MatrixXdr > drdt(jacobians[0], 1, 3), drdp(jacobians[1], 1, nJoints * 3), drdc(jacobians[2], 1, nShapeCoeffs), drds(jacobians[3], 1, 1);
			if (jacobians[0])
				drdt = isigma * edrdJ * edJdt.block(Jid3, 0, 3, 3);
			if (jacobians[1])
				drdp = isigma * edrdJ * edJdp.block(Jid3, 0, 3, nJoints3);
			if (jacobians[2])
				drdc = isigma * edrdJ * edJdc.block(Jid3, 0, 3, nShapeCoeffs);
			if (jacobians[3])
				drds = isigma * edrdJ * edJds.block(Jid3, 0, 3, 1);
		}

		return true;
	}

	double* com_V_ptr, *com_J_ptr, *com_dJdt_ptr, *com_dJdp_ptr, *com_dJdc_ptr, *com_dJds_ptr;

	double	isigma;
	int Jid_, pointFomat_;
	double *plane_;
	SMPLModel &mySMPL_;
};

double FitSMPL2Total_Stage(SMPLModel &mySMPL, vector<SMPLParams> &frame_params, vector<HumanSkeleton3D> &vSkeletons, DensePose *vDensePose, int *TimeStamp, VideoData *VideoInfo, vector<int> &vCams, double *ContourPartWeight, double *JointWeight, double *DPPartweight, double *CostWeights, double *isigmas, double Real2SfM, int skeletonPointFormat, Point2i &fixedPoseFrame, int personId)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;

	const double ialpha = 1.0 / 60.0; //assuming 60fps input
	double w0 = CostWeights[0],  //shape prior
		w1 = CostWeights[1] / frame_params.size(),//pose prior
		w2 = CostWeights[2] / (frame_params.size() - 1), //temporal joint
		w3 = CostWeights[3] / frame_params.size(), //Contour fitting
		w4 = CostWeights[4] / frame_params.size(), //Sil fitting
		w5 = CostWeights[5] / frame_params.size(), //2d fitting
		w6 = CostWeights[6] / frame_params.size(); //dense pose points		

	double SilScale = 4.0, GermanMcClure_Scale = 1.0, GermanMcClure_Curve_Influencier = 1.5, GermanMcClure_DP_Influencier = 1.5; //German McClure scale and influential--> look in V. Koltun's Fast Global registration paper
	double isigma_2D = isigmas[1], //2d fitting (pixels)
		isigma_2DSeg = isigmas[2], //2d contour fitting (pixels)
		isigma_Vel = isigmas[3] * Real2SfM;//3d smoothness (mm/s)

	int *SMPL2Detection = new int[24 * 2];
	if (skeletonPointFormat == 18)
	{
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, //dummy
			SMPL2Detection[2] = 1, SMPL2Detection[3] = 1,
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4,
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7,
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 8, SMPL2Detection[18] = 9, SMPL2Detection[19] = 9, SMPL2Detection[20] = 10, SMPL2Detection[21] = 10,
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 11, SMPL2Detection[24] = 12, SMPL2Detection[25] = 12, SMPL2Detection[26] = 13, SMPL2Detection[27] = 13;
	}
	else
	{
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, SMPL2Detection[2] = 1, SMPL2Detection[3] = 1, //noise, neck
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4, //right arm
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7, //left arm
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 9, SMPL2Detection[18] = 9, SMPL2Detection[19] = 10, SMPL2Detection[20] = 10, SMPL2Detection[21] = 11,//right leg
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 12, SMPL2Detection[24] = 12, SMPL2Detection[25] = 13, SMPL2Detection[26] = 13, SMPL2Detection[27] = 14, //left leg
			SMPL2Detection[28] = 14, SMPL2Detection[29] = 15, SMPL2Detection[30] = 15, SMPL2Detection[31] = 16, SMPL2Detection[32] = 16, SMPL2Detection[33] = 17, SMPL2Detection[34] = 17, SMPL2Detection[35] = 18, //face
			SMPL2Detection[36] = 18, SMPL2Detection[37] = 22, SMPL2Detection[38] = 19, SMPL2Detection[39] = 23, SMPL2Detection[40] = 20, SMPL2Detection[41] = 24, //right foot
			SMPL2Detection[42] = 21, SMPL2Detection[43] = 19, SMPL2Detection[44] = 22, SMPL2Detection[45] = 20, SMPL2Detection[46] = 23, SMPL2Detection[47] = 21;//left foot
	}

	int nInstances = (int)frame_params.size(), nCams = (int)vCams.size(), startF = 999999;
	vector<Point2i> *vCidFid = new vector<Point2i>[nInstances];
	vector<Point3i> *ValidSegPixels = new vector<Point3i>[nCams*nInstances];
	vector<ImgPoseEle> *allPoseLandmark = new vector<ImgPoseEle>[nInstances]; //NOTE: could contain different cid than Denspose id due to different source of detections.
	for (int idf = 0; idf < nInstances; idf++)
	{
		for (int cid = 0; cid < nCams; cid++)
		{
			ImgPoseEle temp(skeletonPointFormat);
			for (int jid = 0; jid < skeletonPointFormat; jid++)
				temp.pt2D[jid] = Point2d(0, 0);
			temp.viewID = -1;
			allPoseLandmark[idf].push_back(temp);
		}

		int refFrame = 0;
		for (int jid = 0; jid < skeletonPointFormat; jid++)
		{
			for (int ii = 0; ii < vSkeletons[idf].vViewID_rFid[jid].size(); ii++)
			{
				int rcid = vSkeletons[idf].vViewID_rFid[jid][ii].x, rfid = vSkeletons[idf].vViewID_rFid[jid][ii].y;
				int cid = -1;
				for (int jj = 0; jj < vCams.size(); jj++)
					if (vCams[jj] == rcid)
						cid = jj;
				if (cid == -1)
					continue;

				allPoseLandmark[idf][cid].viewID = rcid, allPoseLandmark[idf][cid].frameID = rfid, allPoseLandmark[idf][cid].ts = ialpha * refFrame;
				refFrame = rfid + TimeStamp[rcid];
				startF = min(startF, rfid + TimeStamp[rcid]);

				CameraData *Cam = VideoInfo[rcid].VideoInfo;
				if (Cam[rfid].valid)
				{
					allPoseLandmark[idf][cid].pt2D[jid] = vSkeletons[idf].vPt2D[jid][ii];
					allPoseLandmark[idf][cid].confidence[jid] = vSkeletons[idf].vConf[jid][ii];
					allPoseLandmark[idf][cid].ts = ialpha * refFrame;

					if (Cam[rfid].ShutterModel == GLOBAL_SHUTTER)
						AssembleP(Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, allPoseLandmark[idf][cid].P + jid * 12);
					else
						AssembleP_RS(allPoseLandmark[idf][cid].pt2D[jid], Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, Cam[rfid].wt, allPoseLandmark[idf][cid].P + jid * 12);

					bool found = false;
					for (int jj = 0; jj < vCidFid[idf].size() && !found; jj++)
						if (vCidFid[idf][jj].x == cid)
							found = true;
					if (!found)
						vCidFid[idf].push_back(Point2i(cid, rfid));
				}
			}
		}
	}

	int maxWidth = 0, maxHeight = 0;
	for (int cid = 0; cid < nCams; cid++)
		maxWidth = max(maxWidth, VideoInfo[vCams[cid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vCams[cid]].VideoInfo[0].height);

	int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };
	int nSMPL2DetectionPairs = skeletonPointFormat == 18 ? 14 : 24, naJoints3 = naJoints * 3;
	double *residuals = new double[nVertices * 3], rho[3];
	vector<double> VshapePriorCeres, VposePriorCeres, VtemporalRes, VtemporalRes_n, Vfitting2DRes1, Vfitting2DRes1_n, Vfitting2DRes2, Vfitting2DRes2_n, Vfitting2DRes3, Vfitting2DRes3_n, Vfitting2DRes4, Vfitting2DRes4_n;

	int nCorrespond3 = nSMPL2DetectionPairs * 3, nJoints3 = nJoints * 3;
	double *All_V_ptr = new double[nVertices * 3 * nInstances],
		*All_dVdp_ptr = new double[nVertices * 3 * nJoints * 3 * nInstances],
		*All_dVdc_ptr = new double[nVertices * 3 * nShapeCoeffs * nInstances],
		*All_dVds_ptr = new double[nVertices * 3 * nInstances],
		*All_aJsmpl_ptr = new double[naJoints3 * nInstances],
		*All_J_ptr = new double[nCorrespond3 * nInstances],
		*All_dJdt_ptr = new double[nCorrespond3 * 3 * nInstances],
		*All_dJdp_ptr = new double[nCorrespond3 * nJoints * 3 * nInstances],
		*All_dJdc_ptr = new double[nCorrespond3 * nShapeCoeffs * nInstances],
		*All_dJds_ptr = new double[nCorrespond3 * nInstances];
	char *All_VertexTypeAndVisibility = 0;// new char[nCams*nVertices*nInstances];
	bool *All_ProjectedMask = 0;// new bool[maxWidth*maxHeight*nInstances];
	Point2f *All_uv_ptr = 0;

	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> J_reg = skeletonPointFormat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
	SparseMatrix<double, ColMajor>  dVdt = kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	double startTime = omp_get_wtime();
	ceres::Problem problem;
	ceres::Solver::Options options;

	//Evaluator callback
	printLOG("Setting up evalution callback\n");
	vector<double *> Vparameters;
	Vparameters.push_back(&mySMPL.scale);
	Vparameters.push_back(mySMPL.coeffs.data());
	for (int idf = 0; idf < nInstances; idf++)
	{
		Vparameters.push_back(frame_params[idf].pose.data());
		Vparameters.push_back(frame_params[idf].t.data());
	}

	bool hasDensePose = true;
	SmplFitCallBack mySmplFitCallBack(mySMPL, nInstances, skeletonPointFormat, Vparameters, All_V_ptr, All_uv_ptr, All_dVdp_ptr, All_dVdc_ptr, All_dVds_ptr, All_aJsmpl_ptr, All_J_ptr, All_dJdt_ptr, All_dJdp_ptr, All_dJdc_ptr, All_dJds_ptr,
		vDensePose, VideoInfo, All_VertexTypeAndVisibility, vCams, maxWidth*maxHeight, hasDensePose);
	mySmplFitCallBack.PrepareForEvaluation(false, true);
	options.evaluation_callback = &mySmplFitCallBack;

	//Iterator Callback function
	printLOG("Setting up iterator callback\n");
	int iter = 0, debug = 0;
	Point2d *uvJ = new Point2d[nSMPL2DetectionPairs], *uvV = new Point2d[nVertices];
	bool *hit = new bool[maxWidth*maxHeight];
	class MyCallBack : public ceres::IterationCallback
	{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &debug) : callback_(callback), iter(iter), debug(debug) {}
		virtual ~MyCallBack() {}

		ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
		{
			iter = summary.iteration;
			if (summary.step_is_successful)
				callback_();
			return ceres::SOLVER_CONTINUE;
		}
		int &iter, &debug;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
#ifdef _WINDOWS
		int idf = 0, cidI = 0;
		if (debug > 0)
		{
			for (int idf = 0; idf < nInstances; idf++)
			{
				int idf1 = idf + 1;
				double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf;
				char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nCams * nVertices*idf;
				DensePose* DensePoseF = vDensePose + nCams * idf;

				//assume distortion free and global shutter camera
				for (int lcid = 0; lcid < nCams; lcid++)
				{
					if (DensePoseF[lcid].valid == 1)
					{
						//get occluding contour vertices
						int rcid = DensePoseF[lcid].cid, rfid = DensePoseF[lcid].fid, width = VideoInfo[lcid].VideoInfo[rfid].width, height = VideoInfo[lcid].VideoInfo[rfid].height;
						int offset = lcid * nVertices;

						CameraData *camI = &VideoInfo[rcid].VideoInfo[rfid];
						for (int vid = 0; vid < nVertices; vid++)
						{
							int vid3 = vid * 3;
							Point3d xyz(V0_ptr[vid3], V0_ptr[vid3 + 1], V0_ptr[vid3 + 2]);

							if (camI[0].ShutterModel == GLOBAL_SHUTTER)
								ProjectandDistort(xyz, &uvV[vid], camI[0].P, camI[0].K, camI[0].distortion);
							else if (camI[0].ShutterModel == ROLLING_SHUTTER)
								CayleyDistortionProjection(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, uvV[vid], xyz, camI[0].width, camI[0].height);
							if (uvV[vid].x<15 || uvV[vid].x>width - 15 || uvV[vid].y<15 || uvV[vid].y>height - 15)
								uvV[vid].x = 0, uvV[vid].y = 0;
						}

						char Fname[512];
						sprintf(Fname, "C:/temp/%.4d_%d_%d.txt", rfid, lcid, personId); FILE *fp = fopen(Fname, "w");
						for (int vid = 0; vid < nVertices; vid++)
							fprintf(fp, "%.4f %.4f\n", uvV[vid].x, uvV[vid].y);
						fclose(fp);

						sprintf(Fname, "C:/temp/%.4d_%d_oc_%d.txt", rfid, lcid, personId);	fp = fopen(Fname, "w");
						for (int vid = 0; vid < nVertices; vid++)
							if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1)
								fprintf(fp, "%d\n", vid);
						fclose(fp);
					}
				}
			}

			/*for (int cid = 0; cid < nCams; cid++)
			{
			if (debug == 1 && cid != cidI)
			continue;

			CameraData *camI = &VideoInfo[cid].VideoInfo[allPoseLandmark[idf][cid].frameID];
			omp_set_num_threads(omp_get_max_threads());
			#pragma omp parallel for schedule(dynamic,1)
			for (int ii = 0; ii < nVertices; ii++)
			{
			Point3d xyz(outV(ii, 0), outV(ii, 1), outV(ii, 2));
			if (camI[0].LensModel == RADIAL_TANGENTIAL_PRISM)
			{
			if (camI[0].ShutterModel == GLOBAL_SHUTTER)
			ProjectandDistort(xyz, &uvV[ii], camI[0].P, camI[0].K, camI[0].distortion);
			else if (camI[0].ShutterModel == ROLLING_SHUTTER)
			CayleyDistortionProjection(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, uvV[ii], xyz, camI[0].width, camI[0].height);
			if (uvV[ii].x < 0 || uvV[ii].x >  camI[0].width - 1 || uvV[ii].y < 0 || uvV[ii].y > camI[0].height - 1)
			uvV[ii].x = 0, uvV[ii].y = 0;
			}
			else
			{
			if (camI[0].ShutterModel == GLOBAL_SHUTTER)
			FisheyeProjectandDistort(xyz, &uvV[ii], camI[0].P, camI[0].K, camI[0].distortion);
			else if (camI[0].ShutterModel == ROLLING_SHUTTER)
			CayleyFOVProjection2(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, uvV[ii], xyz, camI[0].width, camI[0].height);
			if (uvV[ii].x < 0 || uvV[ii].x >  camI[0].width - 1 || uvV[ii].y < 0 || uvV[ii].y > camI[0].height - 1)
			uvV[ii].x = 0, uvV[ii].y = 0;
			}
			}

			char Fname[512];
			sprintf(Fname, "C:/temp/%.4d_%d_%d.txt", allPoseLandmark[idf][0].frameID, cid, iter);  FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < nVertices; ii++)
			fprintf(fp, "%f %f\n", uvV[ii].x, uvV[ii].y);
			fclose(fp);
			}*/
		}
		iter++;
#endif
	};
	options.callbacks.push_back(new MyCallBack(update_Result, iter, debug));

	//debug = 2;
	MyCallBack *myCallback = new MyCallBack(update_Result, iter, debug);
	myCallback->callback_();

	//Ceres residual blocks
	printLOG("Setting up residual blocks\n");
	ceres::LossFunction* coeffs_loss = new ceres::ScaledLoss(NULL, w0, ceres::TAKE_OWNERSHIP);
	ceres::CostFunction *coeffs_reg = new ceres::AutoDiffCostFunction	< SMPLShapeCoeffRegCeres, nShapeCoeffs, nShapeCoeffs >(new SMPLShapeCoeffRegCeres(nShapeCoeffs));
	problem.AddResidualBlock(coeffs_reg, coeffs_loss, mySMPL.coeffs.data());

	const double * parameters[] = { mySMPL.coeffs.data() };
	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0 *0.5*residuals[ii] * residuals[ii]);

	int stageID = nInstances == 1 ? 0 : 2, nStages = nInstances == 1 ? 2 : 3;
	Map< MatrixXdr > Mosh_pose_prior_mu(mySMPL.Mosh_pose_prior_mu, nJoints * 3, 1);
	for (stageID; stageID < nStages; stageID++)
	{
		printf("*****Stage %d*****\n", stageID);
		for (int idf = 0; idf < nInstances; idf++)
		{
			//pose prior
			if (stageID == 0 || stageID == 2)
			{

				ceres::LossFunction* pose_regl_loss = new ceres::ScaledLoss(NULL, w1, ceres::TAKE_OWNERSHIP);
				ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
				//ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.Mosh_pose_prior_A, Mosh_pose_prior_mu);
				problem.AddResidualBlock(pose_reg, pose_regl_loss, frame_params[idf].pose.data());

				const double * parameters1[] = { frame_params[idf].pose.data() };
				pose_reg->Evaluate(parameters1, residuals, NULL);
				for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
					VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

				for (int ii = 3; ii < nJoints * 3; ii++)
				{
					problem.SetParameterLowerBound(frame_params[idf].pose.data(), ii, mySMPL.minPose[ii]);
					problem.SetParameterUpperBound(frame_params[idf].pose.data(), ii, mySMPL.maxPose[ii]);
				}
			}

			//edge fittimg cost and point fitting
			int nValidPoints = 0;
			double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
			double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
			char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * nCams*idf;
			//bool *ProjectedMaskF = All_ProjectedMask + maxWidth*maxHeight*idf;
			DensePose *DensePoseF = vDensePose + idf * nCams;

			//1. Contour fiting
			if (stageID == 1 || stageID == 2)
			{
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1 && DensePoseF[cid].validParts[dfId] > 0)
							nValidPoints++;
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);

						SmplFitSMPL2EdgeCeres_MV *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, DensePoseF[cid], EdgeVertexTypeAndVisibilityF + offset, allPoseLandmark[idf][cid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg);
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						if (EdgeVertexTypeAndVisibilityF[offset + vid] == -1 || DensePoseF[cid].validParts[dfId] == 0)
							continue;

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
					}
				}
			}

			//2. Sil fitting
			if (stageID == 1 || stageID == 2)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					nValidPoints += nVertices;
				}
				double nw4 = w4 / nValidPoints;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						ceres::LossFunction *robust_loss = new ceres::CauchyLoss(1.0); //works better than huber////new GermanMcClure(GermanMcClure_Scale, 10.0*SilScale*GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw4, ceres::TAKE_OWNERSHIP);

						SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, DensePoseF[cid], allPoseLandmark[idf][cid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg);
						//problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes2_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
					}
				}
			}

			//3. Point fitting
			if (stageID == 0 || stageID == 2)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
					{
						int DetectionId = SMPL2Detection[2 * ii + 1];
						if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
							nValidPoints++;
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
					{
						int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
						if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
						{

							SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
								mySMPL, allPoseLandmark[idf][cid].P, SMPLid, allPoseLandmark[idf][cid].pt2D[DetectionId], skeletonPointFormat, isigma_2D);

							double nw5 = w5 * JointWeight[ii] * allPoseLandmark[idf][cid].confidence[DetectionId] / (0.0001 + nValidPoints);
							ceres::LossFunction* robust_loss = new HuberLoss(1);
							ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
							problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double resX = residuals[0], resY = residuals[1];
							robust_loss->Evaluate(resX*resX + resY * resY, rho);
							Vfitting2DRes3_n.push_back(nw5*0.5*rho[0]);

							resX = resX / isigma_2D, resY = resY / isigma_2D;
							Vfitting2DRes3.push_back(resX), Vfitting2DRes3.push_back(resY);

							//printLOG("(%d, %d, %.2f, %.1f %.1f)...", cid, DetectionId, allPoseLandmark[idf][cid].confidence[DetectionId], resX, resY);
						}
					}
					//printLOG("\n");
				}
			}

			//4. Dense pose fitting
			if (stageID == 1 || stageID == 2)
			{
				nValidPoints = 0;
				vector<int> vNvalidPointsPerPart(24);
				for (int cid = 0; cid < nCams; cid++)
				{
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
					{
						nValidPoints += (int)DensePoseF[cid].DP_vid.size();
						for (int ii = 0; ii < DensePoseF[cid].DP_vid.size(); ii++)
						{
							int vid = DensePoseF[cid].DP_vid[ii], partId = mySMPL.vDensePosePartId[vid];
							vNvalidPointsPerPart[partId]++;
						}
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
					{
						for (int ii = 0; ii < DensePoseF[cid].DP_vid.size(); ii++)
						{
							int vid = DensePoseF[cid].DP_vid[ii], partId = mySMPL.vDensePosePartId[vid];

							double nw6 = w6 * DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]);
							ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_DP_Influencier);
							ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw6, ceres::TAKE_OWNERSHIP);
							SmplFitDensePoseCeres_MV *fit_cost_analytic_fr = new SmplFitDensePoseCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
								mySMPL, VideoInfo[rcid].VideoInfo[rfid], vid, DensePoseF[cid].DP_uv[ii], skeletonPointFormat, isigma_2D);
							//problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double resX = residuals[0], resY = residuals[1], ns = resX * resX + resY * resY;
							robust_loss->Evaluate(ns, rho);
							Vfitting2DRes4_n.push_back(nw6*0.5*rho[0]);

							double influence = sqrt(rho[0] / ns);
							resX = resX / isigma_2D, resY = resY / isigma_2D;
							Vfitting2DRes4.push_back(resX*influence), Vfitting2DRes4.push_back(resY*influence);
						}
					}
				}
			}
		}

		for (int idf = 0; idf < nInstances - 1; idf++)
		{
			int idf1 = idf + 1;
			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2, ceres::TAKE_OWNERSHIP);
			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			//SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			problem.AddResidualBlock(temporal_cost_analytic_fr, avgl_loss, &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data());

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = allPoseLandmark[idf + 1][0].ts - allPoseLandmark[idf + 1][0].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
			}
		}
		for (int fid = fixedPoseFrame.x; fid <= fixedPoseFrame.y && fixedPoseFrame.x > -1; fid++)
		{
			problem.SetParameterBlockConstant(&mySMPL.scale);
			problem.SetParameterBlockConstant(mySMPL.coeffs.data());
			problem.SetParameterBlockConstant(frame_params[fid - startF].t.data());
			problem.SetParameterBlockConstant(frame_params[fid - startF].pose.data());
		}

		{
			Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
			Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
			Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
			Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
			Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
			Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
			Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
			Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
			Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());

			double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
				sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
				sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
				sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
				sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size());
			double sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size()),
				mu3 = MeanArray(Vfitting2DRes3), stdev3 = sqrt(VarianceArray(Vfitting2DRes3, mu3));
			double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
				mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
			if (nInstances > 1)
			{
				Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
				double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
				printLOG("Before Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f, PosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
				printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
				printLOG("OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
				printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
				printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes3_n, mu3, stdev3);
				printLOG("Densepoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			}
			else
			{
				printLOG("Before Optim\nInit scale: %.4f\nShapePrior-->sum of square: %.4f, PosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
				printLOG("OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
				printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
				printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes3_n, mu3, stdev3);
				printLOG("Densepoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			}
		}

		options.num_threads = omp_get_max_threads();
		options.eta = 1e-3;
		options.dynamic_sparsity = true;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
#ifndef _DEBUG
		options.minimizer_progress_to_stdout = frame_params.size() > 1 ? true : false;
#else
		options.minimizer_progress_to_stdout = true;
#endif
		options.update_state_every_iteration = true;
		options.max_num_iterations = 100;

#ifndef _DEBUG
		if (frame_params.size() < 5)
			options.max_solver_time_in_seconds = 300;
		else if (frame_params.size() < 50)
			options.max_solver_time_in_seconds = 500;
		else if (frame_params.size() < 500)
			options.max_solver_time_in_seconds = 700;
		else if (frame_params.size() < 1500)
			options.max_solver_time_in_seconds = 1000;
		else
			options.max_solver_time_in_seconds = 1200;
#endif // !_DEBUG

		//options.max_num_iterations = 2;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.BriefReport() << "\n";

		mySmplFitCallBack.PrepareForEvaluation(false, true);

		debug = stageID > 0 ? 2 : 0;
		myCallback->callback_();

		VshapePriorCeres.clear(), VposePriorCeres.clear(), VtemporalRes_n.clear(), VtemporalRes.clear(), Vfitting2DRes1_n.clear(), Vfitting2DRes1.clear(), Vfitting2DRes2_n.clear(), Vfitting2DRes2.clear(), Vfitting2DRes3.clear(), Vfitting2DRes3_n.clear(), Vfitting2DRes4.clear(), Vfitting2DRes4_n.clear();

		coeffs_reg->Evaluate(parameters, residuals, NULL);
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			VshapePriorCeres.push_back(w0  *0.5*residuals[ii] * residuals[ii]);

		for (int idf = 0; idf < nInstances; idf++)
		{
			ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
			//ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.Mosh_pose_prior_A, Mosh_pose_prior_mu);
			const double * parameters1[] = { frame_params[idf].pose.data() };
			pose_reg->Evaluate(parameters1, residuals, NULL);
			for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
				VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

			int nValidPoints = 0;
			double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
			double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
			char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * nCams*idf;
			//bool *ProjectedMaskF = All_ProjectedMask + maxWidth*maxHeight*idf;
			DensePose *DensePoseF = vDensePose + idf * nCams;

			//1. Contour fitting
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
			{
				int offset = cid * nVertices;
				int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
				if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
					continue;
				for (int vid = 0; vid < nVertices; vid++)
				{
					int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
					if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1 && DensePoseF[cid].validParts[dfId] > 0)
						nValidPoints++;
				}
			}
			for (int cid = 0; cid < nCams; cid++)
			{
				int offset = cid * nVertices;
				int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
				if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
					continue;
				for (int vid = 0; vid < nVertices; vid++)
				{
					int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (EdgeVertexTypeAndVisibilityF[offset + vid] == -1 || DensePoseF[cid].validParts[dfId] == 0)
						continue;

					double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
					//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
					SmplFitSMPL2EdgeCeres_MV *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
						mySMPL, DensePoseF[cid], EdgeVertexTypeAndVisibilityF + offset, allPoseLandmark[idf][cid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg);

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

					GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
					Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
				}
			}

			//2. Sil fitting
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
			{
				int offset = cid * nVertices;
				int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
				if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
					continue;
				nValidPoints += nVertices;
			}
			double nw4 = w4 / nValidPoints;
			for (int cid = 0; cid < nCams; cid++)
			{
				int offset = cid * nVertices;
				int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
				if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
					continue;
				for (int vid = 0; vid < nVertices; vid++)
				{
					SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
						mySMPL, DensePoseF[cid], allPoseLandmark[idf][cid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg);

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

					//GermanMcClure(GermanMcClure_Scale, SilScale*GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
					ceres::CauchyLoss(1.0).Evaluate(res2, rho); //works better than huber
					Vfitting2DRes2_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
				}
			}

			//3. KPoint fitting
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
				for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
					if (allPoseLandmark[idf][cid].pt2D[SMPL2Detection[2 * ii + 1]].x != 0.0)
						nValidPoints++;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
				{
					int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
					{
						SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
							mySMPL, allPoseLandmark[idf][cid].P, SMPLid, allPoseLandmark[idf][cid].pt2D[DetectionId], skeletonPointFormat, isigma_2D);

						double nw5 = w5 * JointWeight[ii] * allPoseLandmark[idf][cid].confidence[DetectionId] / nValidPoints;
						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1];
						ceres::HuberLoss(1).Evaluate(resX*resX + resY * resY, rho);
						Vfitting2DRes3_n.push_back(nw5*0.5*rho[0]);

						resX = resX / isigma_2D, resY = resY / isigma_2D;
						Vfitting2DRes3.push_back(resX), Vfitting2DRes3.push_back(resY);

						//printLOG("(%d, %d, %.2f, %.1f %.1f)...", cid, DetectionId, allPoseLandmark[idf][cid].confidence[DetectionId], resX, resY);
					}
				}
				//printLOG("\n");
			}

			//4. DP fitting
			nValidPoints = 0;
			vector<int> vNvalidPointsPerPart(24);
			for (int cid = 0; cid < nCams; cid++)
			{
				int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
				if (DensePoseF[cid].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
				{
					nValidPoints += (int)DensePoseF[cid].DP_vid.size();
					for (int ii = 0; ii < DensePoseF[cid].DP_uv.size(); ii++)
					{
						int vid = DensePoseF[cid].DP_vid[ii], partId = mySMPL.vDensePosePartId[vid];
						vNvalidPointsPerPart[partId]++;
					}
				}
			}
			for (int lcid = 0; lcid < nCams; lcid++)
			{
				int rcid = DensePoseF[lcid].cid, rfid = DensePoseF[lcid].fid;
				if (DensePoseF[lcid].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
				{
					for (int ii = 0; ii < DensePoseF[lcid].DP_vid.size(); ii++)
					{
						int vid = DensePoseF[lcid].DP_vid[ii], partId = mySMPL.vDensePosePartId[vid];

						double nw6 = w6 * DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]);
						SmplFitDensePoseCeres_MV *fit_cost_analytic_fr = new SmplFitDensePoseCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, VideoInfo[rcid].VideoInfo[rfid], vid, DensePoseF[lcid].DP_uv[ii], skeletonPointFormat, isigma_2D);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1], ns = resX * resX + resY * resY;
						GermanMcClure(GermanMcClure_Scale, GermanMcClure_DP_Influencier).Evaluate(ns, rho);
						Vfitting2DRes4_n.push_back(nw6*0.5*rho[0]);

						double influence = sqrt(rho[0] / ns);
						resX = resX / isigma_2D, resY = resY / isigma_2D;
						Vfitting2DRes4.push_back(resX*influence), Vfitting2DRes4.push_back(resY*influence);
					}
				}
			}
		}
		for (int idf = 0; idf < nInstances - 1; idf++)
		{
			int idf1 = idf + 1;
			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			//SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = allPoseLandmark[idf + 1][0].ts - allPoseLandmark[idf + 1][0].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
			}
		}

		{
			Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
			Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
			Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
			Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
			Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
			Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
			Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
			Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
			Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());

			double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
				sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
				sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
				sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
				sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size());
			double sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size()),
				mu3 = MeanArray(Vfitting2DRes3), stdev3 = sqrt(VarianceArray(Vfitting2DRes3, mu3));
			double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
				mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
			if (nInstances > 1)
			{
				Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
				double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
				printLOG("After Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f, PosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
				printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f(mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
				printLOG("OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
				printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
				printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes3_n, mu3, stdev3);
				printLOG("Densepoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
				printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
			}
			else
			{
				printLOG("After Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f, PosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
				printLOG("OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
				printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
				printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes3_n, mu3, stdev3);
				printLOG("Densepoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
				printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
			}
		}
	}

	delete[]hit;
	delete[]SMPL2Detection, delete[]uvJ, delete[]uvV;
	delete[]vCidFid, delete[]ValidSegPixels;
	delete[]allPoseLandmark, delete[]residuals;

	delete[]All_VertexTypeAndVisibility;
	delete[]All_V_ptr, delete[]All_dVdp_ptr, delete[]All_dVdc_ptr, delete[]All_dVds_ptr;
	delete[]All_aJsmpl_ptr, delete[]All_J_ptr, delete[]All_dJdt_ptr, delete[]All_dJdp_ptr, delete[]All_dJdc_ptr, delete[]All_dJds_ptr;

	return 0.0;
}
double FitSMPL2Total(SMPLModel &mySMPL, vector<SMPLParams> &frame_params, vector<HumanSkeleton3D> &vSkeletons, DensePose *vDensePose, Point3d *CamTimeInfo, VideoData *VideoInfo, vector<int> &vCams, double *ContourPartWeight, double *JointWeight, double *DPPartweight, double *CostWeights, double *isigmas, double Real2SfM, int skeletonPointFormat, Point2i &fixedPoseFrame, int personId, bool hasDensePose)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;

	double plane[3] = { 1.3695914362551931e-03, 1.7627470009816534e-03, 5.6420143860900918e-01 };

	double w0 = CostWeights[0],  //shape prior
		w1 = CostWeights[1] / frame_params.size(),//pose prior
		w2 = CostWeights[2] / (frame_params.size() - 1), //temporal joint
		w3 = CostWeights[3] / frame_params.size(), //Contour fitting
		w4 = CostWeights[4] / frame_params.size(), //Sil fitting
		w5 = CostWeights[5] / frame_params.size(), //2d fitting
		w6 = CostWeights[6] / frame_params.size(), //dense pose points		
		w7 = CostWeights[7] / frame_params.size(),// headcamera constraints
		w8 = CostWeights[8] / frame_params.size();//feet clamping

	double SilScale = 4.0, GermanMcClure_Scale = 10.0, GermanMcClure_Curve_Influencier = 1.0, GermanMcClure_DP_Influencier = 1.0; //German McClure scale and influential--> look in V. Koltun's Fast Global registration paper
	double isigma_2D = isigmas[1], //2d fitting (pixels)
		isigma_2DSeg = isigmas[2], //2d contour fitting (pixels)
		isigma_Vel = isigmas[3] * Real2SfM,//3d smoothness (mm/s)
		isigma_3D = isigmas[4] * Real2SfM; //3d std (mm)

	vector<int>SMPL2Detection, SMPL2HeadCameras, SMPLFeet;
	if (skeletonPointFormat == 17)
	{
		SMPL2Detection.push_back(0), SMPL2Detection.push_back(0), //noise
			SMPL2Detection.push_back(14), SMPL2Detection.push_back(2), SMPL2Detection.push_back(16), SMPL2Detection.push_back(4), //right face
			SMPL2Detection.push_back(15), SMPL2Detection.push_back(1), SMPL2Detection.push_back(17), SMPL2Detection.push_back(3),//left face
			SMPL2Detection.push_back(2), SMPL2Detection.push_back(6), SMPL2Detection.push_back(3), SMPL2Detection.push_back(8), SMPL2Detection.push_back(4), SMPL2Detection.push_back(10),//right arm
			SMPL2Detection.push_back(5), SMPL2Detection.push_back(5), SMPL2Detection.push_back(6), SMPL2Detection.push_back(7), SMPL2Detection.push_back(7), SMPL2Detection.push_back(9),//left arm
			SMPL2Detection.push_back(8), SMPL2Detection.push_back(12), SMPL2Detection.push_back(9), SMPL2Detection.push_back(14), SMPL2Detection.push_back(10), SMPL2Detection.push_back(16),//right leg
			SMPL2Detection.push_back(11), SMPL2Detection.push_back(11), SMPL2Detection.push_back(12), SMPL2Detection.push_back(13), SMPL2Detection.push_back(13), SMPL2Detection.push_back(15);//left leg		
		SMPLFeet.push_back(9), SMPLFeet.push_back(12);
		/*if (personId == 0)
			//SMPL2HeadCameras.push_back(0), SMPL2HeadCameras.push_back(3),
			SMPL2HeadCameras.push_back(15), SMPL2HeadCameras.push_back(4),
			SMPL2HeadCameras.push_back(16), SMPL2HeadCameras.push_back(5);
		else if (personId == 1)
			//SMPL2HeadCameras.push_back(0), SMPL2HeadCameras.push_back(6),
			SMPL2HeadCameras.push_back(15), SMPL2HeadCameras.push_back(7),
			SMPL2HeadCameras.push_back(16), SMPL2HeadCameras.push_back(8);
		else if (personId == 5)
			//SMPL2HeadCameras.push_back(0), SMPL2HeadCameras.push_back(0),
			SMPL2HeadCameras.push_back(15), SMPL2HeadCameras.push_back(1),
			SMPL2HeadCameras.push_back(16), SMPL2HeadCameras.push_back(2);*/
	}
	else if (skeletonPointFormat == 18)
	{
		SMPL2Detection.resize(28);
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, //dummy
			SMPL2Detection[2] = 1, SMPL2Detection[3] = 1,
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4,
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7,
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 8, SMPL2Detection[18] = 9, SMPL2Detection[19] = 9, SMPL2Detection[20] = 10, SMPL2Detection[21] = 10,
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 11, SMPL2Detection[24] = 12, SMPL2Detection[25] = 12, SMPL2Detection[26] = 13, SMPL2Detection[27] = 13;
	}
	else if (skeletonPointFormat == 25)
	{
		SMPL2Detection.resize(48);
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, SMPL2Detection[2] = 1, SMPL2Detection[3] = 1, //noise, neck
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4, //right arm
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7, //left arm
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 9, SMPL2Detection[18] = 9, SMPL2Detection[19] = 10, SMPL2Detection[20] = 10, SMPL2Detection[21] = 11,//right leg
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 12, SMPL2Detection[24] = 12, SMPL2Detection[25] = 13, SMPL2Detection[26] = 13, SMPL2Detection[27] = 14, //left leg
			SMPL2Detection[28] = 14, SMPL2Detection[29] = 15, SMPL2Detection[30] = 15, SMPL2Detection[31] = 16, SMPL2Detection[32] = 16, SMPL2Detection[33] = 17, SMPL2Detection[34] = 17, SMPL2Detection[35] = 18, //face
			SMPL2Detection[36] = 18, SMPL2Detection[37] = 22, SMPL2Detection[38] = 19, SMPL2Detection[39] = 23, SMPL2Detection[40] = 20, SMPL2Detection[41] = 24, //right foot
			SMPL2Detection[42] = 21, SMPL2Detection[43] = 19, SMPL2Detection[44] = 22, SMPL2Detection[45] = 20, SMPL2Detection[46] = 23, SMPL2Detection[47] = 21;//left foot
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < *max_element(vCams.begin(), vCams.end()) + 1; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	int nInstances = (int)frame_params.size(), nCams = (int)vCams.size();
	vector<Point2i> *vCidFid = new vector<Point2i>[nInstances];
	vector<Point3i> *ValidSegPixels = new vector<Point3i>[nCams*nInstances];
	vector<ImgPoseEle> *allPoseLandmark = new vector<ImgPoseEle>[nInstances]; //NOTE: could contain different cid than Denspose id due to different source of detections.
	for (int idf = 0; idf < nInstances; idf++)
	{
		for (int cid = 0; cid < nCams; cid++)
		{
			ImgPoseEle temp(skeletonPointFormat);
			for (int jid = 0; jid < skeletonPointFormat; jid++)
				temp.pt2D[jid] = Point2d(0, 0);
			temp.viewID = cid;
			temp.frameID = vSkeletons[idf].refFid - VideoInfo[cid].TimeOffset;
			allPoseLandmark[idf].push_back(temp);
		}

		for (int jid = 0; jid < skeletonPointFormat; jid++)
		{
			if (vSkeletons[idf].vViewID_rFid[jid].size() == 0)
				for (int jj = 0; jj < vCams.size(); jj++)
					allPoseLandmark[idf][jj].ts = CamTimeInfo[0].x * vSkeletons[idf].refFid; //asume same fps

			for (int ii = 0; ii < vSkeletons[idf].vViewID_rFid[jid].size(); ii++)
			{
				int rcid = vSkeletons[idf].vViewID_rFid[jid][ii].x, rfid = vSkeletons[idf].vViewID_rFid[jid][ii].y;
				int cid = -1;
				for (int jj = 0; jj < vCams.size(); jj++)
					if (vCams[jj] == rcid)
						cid = jj;
				if (cid == -1)
					continue;

				allPoseLandmark[idf][cid].viewID = rcid, allPoseLandmark[idf][cid].frameID = rfid;
				for (int jj = 0; jj < vCams.size(); jj++)
					allPoseLandmark[idf][jj].ts = CamTimeInfo[refCid].x * vSkeletons[idf].refFid;

				CameraData *Cam = VideoInfo[rcid].VideoInfo;
				if (Cam[rfid].valid)
				{
					allPoseLandmark[idf][cid].viewID = rcid;
					allPoseLandmark[idf][cid].frameID = rfid;
					allPoseLandmark[idf][cid].pt2D[jid] = vSkeletons[idf].vPt2D[jid][ii];
					allPoseLandmark[idf][cid].confidence[jid] = vSkeletons[idf].vConf[jid][ii];

					if (Cam[rfid].ShutterModel != ROLLING_SHUTTER)
						AssembleP(Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, &allPoseLandmark[idf][cid].P[jid * 12]);
					else
						AssembleP_RS(allPoseLandmark[idf][cid].pt2D[jid], Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, Cam[rfid].wt, &allPoseLandmark[idf][cid].P[jid * 12]);

					bool found = false;
					for (int jj = 0; jj < vCidFid[idf].size() && !found; jj++)
						if (vCidFid[idf][jj].x == cid)
							found = true;
					if (!found)
						vCidFid[idf].push_back(Point2i(cid, rfid));
				}
			}
		}
	}

	int maxWidth = 0, maxHeight = 0;
	for (int cid = 0; cid < nCams; cid++)
		maxWidth = max(maxWidth, VideoInfo[vCams[cid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vCams[cid]].VideoInfo[0].height);

	//remove hand and feet because they usually snap to the boundary between itself the the part above it
	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face
	for (int ii = 0; ii < 14; ii++)
		for (size_t jj = 0; jj < vMergedPartId2DensePosePartId[ii].size(); jj++)
			vMergedPartId2DensePosePartId[ii][jj] = vMergedPartId2DensePosePartId[ii][jj] - 1; //to convert from densepose index to 0 based index

	int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };
	//int nSMPL2DetectionPairs = skeletonPointFormat == 17 ? 17 : (18 ? 14 : 24), naJoints3 = naJoints * 3;
	int nSMPL2DetectionPairs = skeletonPointFormat == 18 ? 14 : 24, naJoints3 = naJoints * 3;
	double *residuals = new double[nVertices * 3], rho[3];
	vector<double> VshapePriorCeres, VposePriorCeres, VtemporalRes, VtemporalRes_n, Vfitting2DRes1, Vfitting2DRes1_n, Vfitting2DRes2, Vfitting2DRes2_n, Vfitting2DRes3, Vfitting2DRes3_n,
		Vfitting2DRes4, Vfitting2DRes4_n, Vfitting2DRes5, Vfitting2DRes5_n, Vfitting2DRes6, Vfitting2DRes6_n, Vfitting2DRes7, Vfitting2DRes7_n;

	int nCorrespond3 = nSMPL2DetectionPairs * 3, nJoints3 = nJoints * 3;
	double *All_V_ptr = new double[nVertices * 3 * nInstances],
		*All_dVdp_ptr = new double[nVertices * 3 * nJoints * 3 * nInstances],
		*All_dVdc_ptr = new double[nVertices * 3 * nShapeCoeffs * nInstances],
		*All_dVds_ptr = new double[nVertices * 3 * nInstances],
		*All_aJsmpl_ptr = new double[naJoints3 * nInstances],
		*All_J_ptr = new double[nCorrespond3 * nInstances],
		*All_dJdt_ptr = new double[nCorrespond3 * 3 * nInstances],
		*All_dJdp_ptr = new double[nCorrespond3 * nJoints * 3 * nInstances],
		*All_dJdc_ptr = new double[nCorrespond3 * nShapeCoeffs * nInstances],
		*All_dJds_ptr = new double[nCorrespond3 * nInstances];
	char *All_VertexTypeAndVisibility = 0;
	Point2f *All_uv_ptr = 0;
	if (hasDensePose)
		All_VertexTypeAndVisibility = new char[nCams*nVertices*nInstances],
		All_uv_ptr = new Point2f[nCams*nVertices*nInstances];

	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> J_reg = skeletonPointFormat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
	SparseMatrix<double, ColMajor>  dVdt = kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	double startTime = omp_get_wtime();
	ceres::Problem problem;
	ceres::Solver::Options options;

	//Evaluator callback
	printLOG("Setting up evalution callback\n");
	vector<double *> Vparameters;
	Vparameters.push_back(&mySMPL.scale);
	Vparameters.push_back(mySMPL.coeffs.data());
	for (int idf = 0; idf < nInstances; idf++)
	{
		Vparameters.push_back(frame_params[idf].pose.data());
		Vparameters.push_back(frame_params[idf].t.data());
	}

	SmplFitCallBack mySmplFitCallBack(mySMPL, nInstances, skeletonPointFormat, Vparameters, All_V_ptr, All_uv_ptr, All_dVdp_ptr, All_dVdc_ptr, All_dVds_ptr, All_aJsmpl_ptr, All_J_ptr, All_dJdt_ptr, All_dJdp_ptr, All_dJdc_ptr, All_dJds_ptr,
		vDensePose, VideoInfo, All_VertexTypeAndVisibility, vCams, maxWidth*maxHeight, hasDensePose);
	mySmplFitCallBack.PrepareForEvaluation(false, true);
	options.evaluation_callback = &mySmplFitCallBack;

	//Iterator Callback function
	printLOG("Setting up iterator callback\n");
	int iter = 0, debug = 0;
	Point2d *uvJ = new Point2d[nSMPL2DetectionPairs], *uvV = new Point2d[nVertices];
	bool *hit = new bool[maxWidth*maxHeight];
	class MyCallBack : public ceres::IterationCallback
	{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &debug) : callback_(callback), iter(iter), debug(debug) {}
		virtual ~MyCallBack() {}

		ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
		{
			iter = summary.iteration;
			if (summary.step_is_successful)
				callback_();
			return ceres::SOLVER_CONTINUE;
		}
		int &iter, &debug;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
#ifdef _WINDOWS
		int idf = 0, cidI = 0;
		if (debug > 0)
		{
			for (int idf = 0; idf < nInstances; idf++)
			{
				int idf1 = idf + 1;
				double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf;
				//char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nCams * nVertices*idf;
				//DensePose* DensePoseF = vDensePose + nCams * idf;

				for (int lcid = 0; lcid < nCams; lcid++)
				{
					//int rcid = DensePoseF[lcid].cid, rfid = DensePoseF[lcid].fid, width = VideoInfo[lcid].VideoInfo[rfid].width, height = VideoInfo[lcid].VideoInfo[rfid].height;
					int rcid = allPoseLandmark[idf][lcid].viewID, rfid = allPoseLandmark[idf][lcid].frameID;
					if (rcid < 0 || rfid < 0)
						continue;
					int width = VideoInfo[rcid].VideoInfo[rfid].width, height = VideoInfo[rcid].VideoInfo[rfid].height;
					CameraData *camI = VideoInfo[rcid].VideoInfo;


					//if (hasDensePose && DensePoseF[lcid].valid == 1)
					{
						char Fname[512]; //sprintf(Fname, "G:/NEA1/Corrected/%d/%.4d.jpg", rcid, rfid);
						char Path[] = { "E:/Dataset" };
						vector<string> SelectedCamNames;
						SelectedCamNames.push_back("T42664764_rjohnston");
						SelectedCamNames.push_back("T42664773_rjohnston");
						SelectedCamNames.push_back("T42664789_rjohnston");
						vector<int> CamIdsPerSeq; CamIdsPerSeq.push_back(2), CamIdsPerSeq.push_back(3), CamIdsPerSeq.push_back(4);
						int SeqId = 0;

						double gamma = lcid % 3 == 0 ? 0.8 : 0.4;
						sprintf(Fname, "%s/%s/general_%d_%d/image_%.10d_0.png", Path, SelectedCamNames[lcid / CamIdsPerSeq.size()].c_str(), SeqId, CamIdsPerSeq[lcid%CamIdsPerSeq.size()], rfid);
						Mat img = correctGamma(imread(Fname), gamma);
						//get joint projection
						Point2d uv;
						for (int jid = 0; jid < nCorrespond3 / 3; jid++)
						{
							int jid3 = jid * 3;
							Point3d xyz(J0_ptr[jid3], J0_ptr[jid3 + 1], J0_ptr[jid3 + 2]);
							ProjectandDistort(xyz, &uv, camI[rfid].P);
							LensDistortionPoint_KB3(uv, camI[rfid].intrinsic, camI[rfid].distortion);
							circle(img, uv, 1, Scalar(0, 0, 255));
							//if (debug == 3)
							//{
							//	cout << xyz << endl;
							//	sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
							//}
						}

						for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
						{
							if (!IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
								continue;
							int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
							ProjectandDistort(vSkeletons[idf].pt3d[DetectionId], &uv, camI[rfid].P);
							LensDistortionPoint_KB3(uv, camI[rfid].intrinsic, camI[rfid].distortion);
							circle(img, uv, 1, Scalar(255, 0, 0));
							//if (debug == 3)
							//	sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
						}


						for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
						{
							int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
							if (allPoseLandmark[idf][lcid].pt2D[DetectionId].x != 0.0)
							{
								uv.x = allPoseLandmark[idf][lcid].pt2D[DetectionId].x, uv.y = allPoseLandmark[idf][lcid].pt2D[DetectionId].y;
								LensDistortionPoint_KB3(uv, camI[rfid].intrinsic, camI[rfid].distortion);
								circle(img, uv, 1, Scalar(0, 255, 0));
								if (debug == 3)
									sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
							}
						}
						/*//get occluding contour vertices
						int offset = lcid * nVertices;
#pragma omp parallel for schedule(dynamic,1)
						for (int vid = 0; vid < nVertices; vid++)
						{
							int vid3 = vid * 3;
							Point3d xyz(V0_ptr[vid3], V0_ptr[vid3 + 1], V0_ptr[vid3 + 2]);
							ProjectandDistort(xyz, &uvV[vid], camI[rfid].P);// , camI[rfid].K, camI[rfid].distortion);

							if (uvV[vid].x<15 || uvV[vid].x>width - 15 || uvV[vid].y<15 || uvV[vid].y>height - 15)
								uvV[vid].x = 0, uvV[vid].y = 0;
						}
						for (int vid = 0; vid < nVertices; vid++)
						{
							if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1)
								circle(img, uvV[vid], 1, Scalar(0, 255, 0));
							else
								circle(img, uvV[vid], 1, Scalar(0, 0, 255));
						}*/
						sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
						int a = 0;
						//sprintf(Fname, "C:/temp/%.4d_%d_oc_%d.txt", lcid, rfid, personId);	FILE *fp = fopen(Fname, "w");
						//for (int vid = 0; vid < nVertices; vid++)
						//	if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1)
						//		fprintf(fp, "%d %d %.1f %.1f \n", vid, EdgeVertexTypeAndVisibilityF[vid + offset], uvV[vid].x, uvV[vid].y);
						//fclose(fp);
					}
				}
			}
		}
		iter++;
#endif
	};
	options.callbacks.push_back(new MyCallBack(update_Result, iter, debug));

	debug = 0;
	MyCallBack *myCallback = new MyCallBack(update_Result, iter, debug);
	//	myCallback->callback_();

		//Ceres residual blocks
	printLOG("Setting up residual blocks\n");
	ceres::LossFunction* coeffs_loss = new ceres::ScaledLoss(NULL, w0, ceres::TAKE_OWNERSHIP);
	ceres::CostFunction *coeffs_reg = new ceres::AutoDiffCostFunction	< SMPLShapeCoeffRegCeres, nShapeCoeffs, nShapeCoeffs >(new SMPLShapeCoeffRegCeres(nShapeCoeffs));
	problem.AddResidualBlock(coeffs_reg, coeffs_loss, mySMPL.coeffs.data());

	const double * parameters[] = { mySMPL.coeffs.data() };
	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0 *0.5*residuals[ii] * residuals[ii]);

	
	Map< MatrixXdr > Mosh_pose_prior_mu(mySMPL.Mosh_pose_prior_mu, nJoints * 3, 1);
	for (int idf = 0; idf < nInstances; idf++)
	{
		//pose prior
		ceres::LossFunction* pose_regl_loss = new ceres::ScaledLoss(NULL, w1, ceres::TAKE_OWNERSHIP);
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
		//ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.Mosh_pose_prior_A, Mosh_pose_prior_mu);
		if (w1 > 0.0)
			problem.AddResidualBlock(pose_reg, pose_regl_loss, frame_params[idf].pose.data());

		const double * parameters1[] = { frame_params[idf].pose.data() };
		pose_reg->Evaluate(parameters1, residuals, NULL);
		for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
			VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

		//for (int ii = 3; ii < nJoints * 3; ii++)
		//{
		//	problem.SetParameterLowerBound(frame_params[idf].pose.data(), ii, mySMPL.minPose[ii]);
		//	problem.SetParameterUpperBound(frame_params[idf].pose.data(), ii, mySMPL.maxPose[ii]);
		//}

		//edge fittimg cost and point fitting
		int nValidPoints = 0;
		double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
		double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
		double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
		char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * nCams*idf;
		DensePose *DensePoseF = vDensePose + idf * nCams;
		Point2f *vUV_ptr = All_uv_ptr + nVertices * idf*nCams;

		if (hasDensePose)
		{
			//1. Contour fiting
			double  isigma2 = isigma_2DSeg * isigma_2DSeg;
			if (w3 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1 && DensePoseF[cid].validParts[dfId] > 0)
							nValidPoints++;
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);
						SmplFitSMPL2EdgeCeres_MV2 *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV2(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid],
							EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						if (EdgeVertexTypeAndVisibilityF[offset + vid] == -1 || DensePoseF[cid].validParts[dfId] == 0)
							continue;

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
					}
				}

				//other direction
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int partId = 0; partId < 14; partId++)
						nValidPoints += (int)DensePoseF[cid].vEdge[partId].size() / 2;
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int partId = 0; partId < 14; partId++)
					{
						for (int eid = 0; eid < DensePoseF[cid].vEdge[partId].size(); eid += 2)
						{
							double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
							//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
							ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
							ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);
							//SmplFitEdge2SMPLCeres_MV *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid],
							//	EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

							SmplFitEdge2SMPLCeres_MV2 *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV2(V_ptr, vUV_ptr + nVertices * cid, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
								EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
							problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
							resX = resX / isigma_2DSeg, resY = resY / isigma_2DSeg;
							robust_loss->Evaluate(res2, rho);
							Vfitting2DRes2_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
						}
					}
				}
			}

			//2. Sil fitting
			if (w4 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					nValidPoints += nVertices;
				}
				double nw4 = w4 / nValidPoints;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						ceres::LossFunction *robust_loss = new ceres::CauchyLoss(1.0); //works better than huber////new GermanMcClure(GermanMcClure_Scale, 10.0*SilScale*GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw4, ceres::TAKE_OWNERSHIP);

						SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, DensePoseF[cid], VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0];
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes3_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes3.push_back(res2 / isigma2);
					}
				}
			}
		}

		//3. Point fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				{
					int DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
						nValidPoints++;
				}
			}
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				{
					int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
					{
						SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
							mySMPL, &allPoseLandmark[idf][cid].P[12 * DetectionId], SMPLid, allPoseLandmark[idf][cid].pt2D[DetectionId], skeletonPointFormat, isigma_2D); //can work with Rolling RS where each point has diff pose

						double nw5 = w5 * JointWeight[SMPLid] * allPoseLandmark[idf][cid].confidence[DetectionId] / (0.0001 + nValidPoints);
						ceres::LossFunction* robust_loss = new HuberLoss(1);
						//ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1];
						robust_loss->Evaluate(resX*resX + resY * resY, rho);
						Vfitting2DRes4_n.push_back(nw5*0.5*rho[0]);

						resX = resX / isigma_2D, resY = resY / isigma_2D;
						Vfitting2DRes4.push_back(resX), Vfitting2DRes4.push_back(resY);
						//if (cid == 0 && (abs(resX) > 0.1 || abs(resY) > 0.1))
						//	printLOG("Ref cam with large error %d %d: %.2f %.2f\n", idf, ii, resX, resY);
					}
				}
			}
		}

		//for (int ii = 0; ii < nCorrespond3 / 3; ii++)
		//	printf("%.5f %.5f %.5f\n", J_ptr[ii * 3], J_ptr[ii * 3 + 1], J_ptr[ii * 3 + 2]);

		//4. smpl2Pts3d fitting
		//if (w5 > 0.0)
		if (0)
		{
			nValidPoints = 0;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				if (IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					nValidPoints++;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
			{
				if (!IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					continue;

				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, vSkeletons[idf].pt3d[DetectionId], skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose
				//printf("%.4f %.4f %.4f  %.4f %.4f %.4f \n", vSkeletons[idf].pt3d[DetectionId].x, vSkeletons[idf].pt3d[DetectionId].y, vSkeletons[idf].pt3d[DetectionId].z, J_ptr[3 * SMPLid], J_ptr[3 * SMPLid + 1], J_ptr[3 * SMPLid + 2]);

				double nw5 = 0.05*w5 * JointWeight[SMPLid] / (0.0001 + nValidPoints); //very small weight of w5
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				//ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				robust_loss->Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				//rho[0] = resX * resX + resY * resY + resZ * resZ;
				Vfitting2DRes5_n.push_back(nw5*0.5*rho[0]);

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes5.push_back(resX), Vfitting2DRes5.push_back(resY), Vfitting2DRes5.push_back(resZ);
			}
		}

		//5. set for camera head pose constraints
		if (w7 > 0.0)
		{
			nValidPoints = 2;
			for (int ii = 0; ii < SMPL2HeadCameras.size() / 2; ii++)
			{
				int SMPLid = SMPL2HeadCameras[2 * ii], cid = SMPL2HeadCameras[2 * ii + 1], rfid = allPoseLandmark[idf][cid].frameID;
				Point3d anchor(VideoInfo[cid].VideoInfo[rfid].camCenter[0], VideoInfo[cid].VideoInfo[rfid].camCenter[1], VideoInfo[cid].VideoInfo[rfid].camCenter[2]);
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, anchor, skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw7 = w7 / nValidPoints;
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(NULL, nw7, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				Vfitting2DRes6_n.push_back(nw7*0.5*(resX*resX + resY * resY + resZ * resZ));

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes6.push_back(resX), Vfitting2DRes6.push_back(resY), Vfitting2DRes6.push_back(resZ);
			}
		}

		//6. set foot clamping
		if (w8 > 0.0)
		{
			nValidPoints = 2;
			for (int ii = 0; ii < 2; ii++)
			{
				int SMPLid = SMPLFeet[ii];
				SmplClampGroundCeres *fit_cost_analytic_fr = new SmplClampGroundCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr, mySMPL, SMPLid, plane, skeletonPointFormat, isigma_3D);

				double nw8 = w8 / nValidPoints;
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw8, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double res = residuals[0];
				robust_loss->Evaluate(res*res, rho);
				Vfitting2DRes7_n.push_back(nw8*0.5*rho[0]);

				res = res / isigma_3D;
				Vfitting2DRes7.push_back(res);
			}
		}
	}
	if (nInstances > 1 && w2 > 0.0)
	{
		for (int idf = 0; idf < nInstances - 1; idf++)
		{
			int idf1 = idf + 1;
			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2, ceres::TAKE_OWNERSHIP);
			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			//ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2 / nVertices, ceres::TAKE_OWNERSHIP);
			//SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			problem.AddResidualBlock(temporal_cost_analytic_fr, avgl_loss, &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data());

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = allPoseLandmark[idf + 1][0].ts - allPoseLandmark[idf][0].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
				//VtemporalRes_n.push_back(w2 / nVertices*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / sqrt(dt)*sqrt(nVertices * 3) / Real2SfM, 2)); //in mm/s
			}
		}
	}

	//fix shape
	problem.SetParameterBlockConstant(mySMPL.coeffs.data());
	problem.SetParameterBlockConstant(&mySMPL.scale);
	//problem.SetParameterLowerBound(&mySMPL.scale, 0, 0.7*mySMPL.scale);
	//problem.SetParameterUpperBound(&mySMPL.scale, 0, 1.3*mySMPL.scale);
	if (w0 > 1000000)
	{
		mySMPL.coeffs.setZero();
		problem.SetParameterBlockConstant(mySMPL.coeffs.data());
	}

	//fix frames from previous window
	if (fixedPoseFrame.x != -1 && fixedPoseFrame.y != -1)
	{
		problem.SetParameterBlockConstant(&mySMPL.scale);
		problem.SetParameterBlockConstant(mySMPL.coeffs.data());
		for (int idf = 0; idf < nInstances; idf++)
		{
			if (frame_params[idf].frame >= fixedPoseFrame.x && frame_params[idf].frame <= fixedPoseFrame.y)
			{
				problem.SetParameterBlockConstant(frame_params[idf].t.data());
				problem.SetParameterBlockConstant(frame_params[idf].pose.data());
			}
		}
	}

	{
		Vfitting2DRes7.push_back(0), Vfitting2DRes7_n.push_back(0), Vfitting2DRes6.push_back(0), Vfitting2DRes6_n.push_back(0), Vfitting2DRes5.push_back(0), Vfitting2DRes5_n.push_back(0),
			Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
		Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
		Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
		Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
		Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
		Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
		Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
		Map< VectorXd > eVfitting2DRes3(&Vfitting2DRes3[0], Vfitting2DRes3.size());
		Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
		Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());
		Map< VectorXd > eVfitting2DRes5_n(&Vfitting2DRes5_n[0], Vfitting2DRes5_n.size());
		Map< VectorXd > eVfitting2DRes6_n(&Vfitting2DRes6_n[0], Vfitting2DRes6_n.size());
		Map< VectorXd > eVfitting2DRes7_n(&Vfitting2DRes7_n[0], Vfitting2DRes7_n.size());

		double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
			sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
			sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
			sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
			sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size()),
			sos_Vfitting2DRes3 = eVfitting2DRes3.sum(), rmse_Vfitting2DRes3 = sqrt(sos_Vfitting2DRes3 / Vfitting2DRes3.size()),
			sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size());
		double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
			mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
		double sos_Vfitting2DRes5_n = eVfitting2DRes5_n.sum(), rmse_Vfitting2DRes5_n = sqrt(sos_Vfitting2DRes5_n / Vfitting2DRes5_n.size()),
			mu5 = MeanArray(Vfitting2DRes5), stdev5 = sqrt(VarianceArray(Vfitting2DRes5, mu5));
		double sos_Vfitting2DRes6_n = eVfitting2DRes6_n.sum(), rmse_Vfitting2DRes6_n = sqrt(sos_Vfitting2DRes6_n / Vfitting2DRes6_n.size()),
			mu6 = MeanArray(Vfitting2DRes6), stdev6 = sqrt(VarianceArray(Vfitting2DRes6, mu6));
		double sos_Vfitting2DRes7_n = eVfitting2DRes7_n.sum(), rmse_Vfitting2DRes7_n = sqrt(sos_Vfitting2DRes7_n / Vfitting2DRes7_n.size()),
			mu7 = MeanArray(Vfitting2DRes7), stdev7 = sqrt(VarianceArray(Vfitting2DRes7, mu7));
		if (nInstances > 1)
		{
			Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
			double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
			printLOG("Before Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("2D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			printLOG("3D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D head-mounted camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
		}
		else
		{
			printLOG("Before Optim\nInit scale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			if (nInstances == 1)
				printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
		}
	}

	options.num_threads = omp_get_max_threads();
	options.eta = 1e-3;
	options.dynamic_sparsity = true;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
#ifndef _DEBUG
	options.minimizer_progress_to_stdout = frame_params.size() > 1 ? true : false;
#else
	options.minimizer_progress_to_stdout = true;
#endif
	options.update_state_every_iteration = true;
	options.max_num_iterations = 100;

#ifndef _DEBUG
	if (frame_params.size() < 5)
		options.max_solver_time_in_seconds = 300;
	else if (frame_params.size() < 50)
		options.max_solver_time_in_seconds = 400;
	else if (frame_params.size() < 500)
		options.max_solver_time_in_seconds = 600;
	else if (frame_params.size() < 1500)
		options.max_solver_time_in_seconds = 800;
	else
		options.max_solver_time_in_seconds = 1000;
#endif // !_DEBUG

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	mySmplFitCallBack.PrepareForEvaluation(false, true);

	//debug = 2;
	myCallback->callback_();

	VshapePriorCeres.clear(), VposePriorCeres.clear(), VtemporalRes_n.clear(), VtemporalRes.clear(), Vfitting2DRes1_n.clear(), Vfitting2DRes1.clear();
	Vfitting2DRes2_n.clear(), Vfitting2DRes2.clear(), Vfitting2DRes3.clear(), Vfitting2DRes3_n.clear(), Vfitting2DRes4.clear(), Vfitting2DRes4_n.clear(), Vfitting2DRes5.clear(), Vfitting2DRes5_n.clear();
	Vfitting2DRes6.clear(), Vfitting2DRes6_n.clear(), Vfitting2DRes7.clear(), Vfitting2DRes7_n.clear();

	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0  *0.5*residuals[ii] * residuals[ii]);

	FILE*fpOut = fopen("ResSync.txt", "a+");
	for (int idf = 0; idf < nInstances; idf++)
	{
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
		//ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.Mosh_pose_prior_A, Mosh_pose_prior_mu);
		const double * parameters1[] = { frame_params[idf].pose.data() };
		pose_reg->Evaluate(parameters1, residuals, NULL);
		for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
			VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

		int nValidPoints = 0;
		double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
		double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
		double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
		char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * nCams*idf;
		DensePose *DensePoseF = vDensePose + idf * nCams;
		Point2f *vUV_ptr = All_uv_ptr + nVertices * idf*nCams;

		if (hasDensePose)
		{
			//1. Contour fitting
			if (w3 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1 && DensePoseF[cid].validParts[dfId] > 0)
							nValidPoints++;
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
						if (EdgeVertexTypeAndVisibilityF[offset + vid] == -1 || DensePoseF[cid].validParts[dfId] == 0)
							continue;

						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						SmplFitSMPL2EdgeCeres_MV2 *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV2(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid],
							EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

						GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
						Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
					}
				}

				//other direction
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int partId = 0; partId < 14; partId++)
						nValidPoints += (int)DensePoseF[cid].vEdge[partId].size() / 2;
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int partId = 0; partId < 14; partId++)
					{
						for (int eid = 0; eid < DensePoseF[cid].vEdge[partId].size(); eid += 2)
						{
							double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
							//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
							//SmplFitEdge2SMPLCeres_MV *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
							//	EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

							SmplFitEdge2SMPLCeres_MV2 *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV2(V_ptr, vUV_ptr + nVertices * cid, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
								EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

							GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
							Vfitting2DRes2_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
						}
					}
				}
			}

			//2. Sil fitting
			if (w4 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					nValidPoints += nVertices;
				}
				double nw4 = w4 / nValidPoints;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, DensePoseF[cid], VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

						//GermanMcClure(GermanMcClure_Scale, SilScale*GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
						ceres::CauchyLoss(1.0).Evaluate(res2, rho); //works better than huber
						Vfitting2DRes3_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes3.push_back(res2 / isigma2);
					}
				}
			}
		}

		//3. KPoint fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
					if (allPoseLandmark[idf][cid].pt2D[SMPL2Detection[2 * ii + 1]].x != 0.0)
						nValidPoints++;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				{
					int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
					{
						allPoseLandmark[idf][cid].viewID;
						allPoseLandmark[idf][cid].frameID;

						SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
							mySMPL, &allPoseLandmark[idf][cid].P[12 * DetectionId], SMPLid, allPoseLandmark[idf][cid].pt2D[DetectionId], skeletonPointFormat, isigma_2D);

						double nw5 = w5 * JointWeight[SMPLid] * allPoseLandmark[idf][cid].confidence[DetectionId] / nValidPoints;
						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1];
						ceres::HuberLoss(1).Evaluate(resX*resX + resY * resY, rho);
						//GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(resX*resX + resY * resY, rho);
						Vfitting2DRes4_n.push_back(nw5*0.5*rho[0]);

						resX = resX / isigma_2D, resY = resY / isigma_2D;
						Vfitting2DRes4.push_back(resX), Vfitting2DRes4.push_back(resY);
						
						fprintf(fpOut, "%d %.2f %.2f\n", SMPLid, resX, resY);
					}
				}
			}
		}

		//4. smpl2Pts3d fitting
		//if (w5 > 0.0)
		if(0)
		{
			nValidPoints = 0;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				if (IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					nValidPoints++;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
			{
				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				if (!IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					continue;

				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, vSkeletons[idf].pt3d[DetectionId], skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw5 = 0.05*w5 * JointWeight[SMPLid] / (0.0001 + nValidPoints);
				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				//rho[0] = resX * resX + resY * resY + resZ * resZ;
				HuberLoss(1).Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				//GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				Vfitting2DRes5_n.push_back(nw5*0.5*rho[0]);

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes5.push_back(resX), Vfitting2DRes5.push_back(resY), Vfitting2DRes5.push_back(resZ);
			}
		}

		//5. set for camera head pose constraints
		if (w7 > 0.0)
		{
			nValidPoints = 3;
			for (int ii = 0; ii < SMPL2HeadCameras.size() / 2; ii++)
			{
				int SMPLid = SMPL2HeadCameras[2 * ii], cid = SMPL2HeadCameras[2 * ii + 1], rfid = allPoseLandmark[idf][cid].frameID;
				Point3d anchor(VideoInfo[cid].VideoInfo[rfid].camCenter[0], VideoInfo[cid].VideoInfo[rfid].camCenter[1], VideoInfo[cid].VideoInfo[rfid].camCenter[2]);
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, anchor, skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw7 = w7 / nValidPoints;
				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				Vfitting2DRes6_n.push_back(nw7*0.5*(resX*resX + resY * resY + resZ * resZ));

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes6.push_back(resX), Vfitting2DRes6.push_back(resY), Vfitting2DRes6.push_back(resZ);
			}
		}

		//6. set foot clamping
		if (w8 > 0.0)
		{
			nValidPoints = 2;
			for (int ii = 0; ii < 2; ii++)
			{
				int SMPLid = SMPLFeet[ii];
				SmplClampGroundCeres *fit_cost_analytic_fr = new SmplClampGroundCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr, mySMPL, SMPLid, plane, skeletonPointFormat, isigma_3D);

				double nw8 = w8 / nValidPoints;
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(NULL, nw8, ceres::TAKE_OWNERSHIP);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double res = residuals[0];
				robust_loss->Evaluate(res*res, rho);
				Vfitting2DRes7_n.push_back(nw8*0.5*rho[0]);

				res = res / isigma_3D;
				Vfitting2DRes7.push_back(res);
			}
		}
	}
	if (nInstances > 1 && w2 > 0.0)
	{
		for (int idf = 0; idf < nInstances - 1; idf++)
		{
			int idf1 = idf + 1;
			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			//SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
				//for (int ii = 0; ii < nVertices * 3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = allPoseLandmark[idf + 1][0].ts - allPoseLandmark[idf][0].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
				//VtemporalRes_n.push_back(w2 / nVertices*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / sqrt(dt)*sqrt(nVertices * 3) / Real2SfM, 2)); //in mm/s
			}
		}
	}
	fclose(fpOut);

	{
		Vfitting2DRes7.push_back(0), Vfitting2DRes7_n.push_back(0), Vfitting2DRes6.push_back(0), Vfitting2DRes6_n.push_back(0), Vfitting2DRes5.push_back(0), Vfitting2DRes5_n.push_back(0),
			Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
		Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
		Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
		Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
		Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
		Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
		Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
		Map< VectorXd > eVfitting2DRes3(&Vfitting2DRes3[0], Vfitting2DRes3.size());
		Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
		Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());
		Map< VectorXd > eVfitting2DRes5_n(&Vfitting2DRes5_n[0], Vfitting2DRes5_n.size());
		Map< VectorXd > eVfitting2DRes6_n(&Vfitting2DRes6_n[0], Vfitting2DRes6_n.size());
		Map< VectorXd > eVfitting2DRes7_n(&Vfitting2DRes7_n[0], Vfitting2DRes7_n.size());

		double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
			sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
			sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
			sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
			sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size()),
			sos_Vfitting2DRes3 = eVfitting2DRes3.sum(), rmse_Vfitting2DRes3 = sqrt(sos_Vfitting2DRes3 / Vfitting2DRes3.size()),
			sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size());
		double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
			mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
		double sos_Vfitting2DRes5_n = eVfitting2DRes5_n.sum(), rmse_Vfitting2DRes5_n = sqrt(sos_Vfitting2DRes5_n / Vfitting2DRes5_n.size()),
			mu5 = MeanArray(Vfitting2DRes5), stdev5 = sqrt(VarianceArray(Vfitting2DRes5, mu5));
		double sos_Vfitting2DRes6_n = eVfitting2DRes6_n.sum(), rmse_Vfitting2DRes6_n = sqrt(sos_Vfitting2DRes6_n / Vfitting2DRes6_n.size()),
			mu6 = MeanArray(Vfitting2DRes6), stdev6 = sqrt(VarianceArray(Vfitting2DRes6, mu6));
		double sos_Vfitting2DRes7_n = eVfitting2DRes7_n.sum(), rmse_Vfitting2DRes7_n = sqrt(sos_Vfitting2DRes7_n / Vfitting2DRes7_n.size()),
			mu7 = MeanArray(Vfitting2DRes7), stdev7 = sqrt(VarianceArray(Vfitting2DRes7, mu7));

		if (nInstances > 1)
		{
			Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
			double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
			printLOG("After Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("2D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			printLOG("3D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
			printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
		}
		else
		{
			printLOG("After Optim\nFinal scale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			if (nInstances == 1)
				printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
			printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
		}
	}

	delete[]hit;
	delete[]uvJ, delete[]uvV;
	delete[]vCidFid, delete[]ValidSegPixels;
	delete[]allPoseLandmark, delete[]residuals;

	delete[]All_VertexTypeAndVisibility, delete[]All_uv_ptr;
	delete[]All_V_ptr, delete[]All_dVdp_ptr, delete[]All_dVdc_ptr, delete[]All_dVds_ptr;
	delete[]All_aJsmpl_ptr, delete[]All_J_ptr, delete[]All_dJdt_ptr, delete[]All_dJdp_ptr, delete[]All_dJdc_ptr, delete[]All_dJds_ptr;

	return 0.0;
}
double FitSMPL2TotalUnSync(SMPLModel &mySMPL, vector<SMPLParams> &frame_params, vector<ImgPoseEle> &frame_skeleton, DensePose *vDensePose, Point3d *CamTimeInfo, VideoData *VideoInfo, vector<int> &vCams, double *ContourPartWeight, double *JointWeight, double *DPPartweight, double *CostWeights, double *isigmas, double Real2SfM, int skeletonPointFormat, Point2i &fixedPoseFrame, int personId, bool hasDensePose)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;

	double w0 = CostWeights[0],  //shape prior
		w1 = CostWeights[1] / frame_params.size(),//pose prior
		w2 = CostWeights[2] / (frame_params.size() - 1), //temporal joint
		w3 = CostWeights[3] / frame_params.size(), //Contour fitting
		w4 = CostWeights[4] / frame_params.size(), //Sil fitting
		w5 = CostWeights[5] / frame_params.size(), //2d fitting
		w6 = CostWeights[6] / frame_params.size(), //dense pose points		
		w7 = CostWeights[7] / frame_params.size(),// headcamera constraints
		w8 = CostWeights[8] / frame_params.size();//feet clamping

	double SilScale = 4.0, GermanMcClure_Scale = 1.0, GermanMcClure_Curve_Influencier = 1.5, GermanMcClure_DP_Influencier = 1.5; //German McClure scale and influential--> look in V. Koltun's Fast Global registration paper
	double isigma_2D = isigmas[1], //2d fitting (pixels)
		isigma_2DSeg = isigmas[2], //2d contour fitting (pixels)
		isigma_Vel = isigmas[3] * Real2SfM,//3d smoothness (mm/s)
		isigma_3D = isigmas[4] * Real2SfM; //3d std (mm)

	int *SMPL2Detection = new int[24 * 2];
	if (skeletonPointFormat == 18)
	{
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, //dummy
			SMPL2Detection[2] = 1, SMPL2Detection[3] = 1,
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4,
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7,
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 8, SMPL2Detection[18] = 9, SMPL2Detection[19] = 9, SMPL2Detection[20] = 10, SMPL2Detection[21] = 10,
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 11, SMPL2Detection[24] = 12, SMPL2Detection[25] = 12, SMPL2Detection[26] = 13, SMPL2Detection[27] = 13;
	}
	else
	{
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, SMPL2Detection[2] = 1, SMPL2Detection[3] = 1, //noise, neck
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4, //right arm
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7, //left arm
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 9, SMPL2Detection[18] = 9, SMPL2Detection[19] = 10, SMPL2Detection[20] = 10, SMPL2Detection[21] = 11,//right leg
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 12, SMPL2Detection[24] = 12, SMPL2Detection[25] = 13, SMPL2Detection[26] = 13, SMPL2Detection[27] = 14, //left leg
			SMPL2Detection[28] = 14, SMPL2Detection[29] = 15, SMPL2Detection[30] = 15, SMPL2Detection[31] = 16, SMPL2Detection[32] = 16, SMPL2Detection[33] = 17, SMPL2Detection[34] = 17, SMPL2Detection[35] = 18, //face
			SMPL2Detection[36] = 18, SMPL2Detection[37] = 22, SMPL2Detection[38] = 19, SMPL2Detection[39] = 23, SMPL2Detection[40] = 20, SMPL2Detection[41] = 24, //right foot
			SMPL2Detection[42] = 21, SMPL2Detection[43] = 19, SMPL2Detection[44] = 22, SMPL2Detection[45] = 20, SMPL2Detection[46] = 23, SMPL2Detection[47] = 21;//left foot
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < *max_element(vCams.begin(), vCams.end()) + 1; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	int nCams = (int)vCams.size(), nInstances = (int)frame_skeleton.size();
	vector<int> SMPL_TimeOrder(nInstances);
	vector<double> ts(nInstances);
	for (int ii = 0; ii < nInstances; ii++)
		SMPL_TimeOrder[ii] = ii, ts[ii] = frame_skeleton[ii].ts;
	Quick_Sort_Double(&ts[0], &SMPL_TimeOrder[0], 0, nInstances - 1);

	int maxWidth = 0, maxHeight = 0;
	for (int cid = 0; cid < nCams; cid++)
		maxWidth = max(maxWidth, VideoInfo[vCams[cid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vCams[cid]].VideoInfo[0].height);

	//remove hand and feet because they usually snap to the boundary between itself the the part above it
	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
																								  //vMergedPartId2DensePosePartId[1].push_back(4), //l hand
																								  //vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
																										//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
																										//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face
	for (int ii = 0; ii < 14; ii++)
		for (size_t jj = 0; jj < vMergedPartId2DensePosePartId[ii].size(); jj++)
			vMergedPartId2DensePosePartId[ii][jj] = vMergedPartId2DensePosePartId[ii][jj] - 1; //to convert from densepose index to 0 based index

	int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };
	int nSMPL2DetectionPairs = skeletonPointFormat == 18 ? 14 : 24, naJoints3 = naJoints * 3;
	double *residuals = new double[nVertices * 3], rho[3];
	vector<double> VshapePriorCeres, VposePriorCeres, VtemporalRes, VtemporalRes_n, Vfitting2DRes1, Vfitting2DRes1_n, Vfitting2DRes2, Vfitting2DRes2_n, Vfitting2DRes3, Vfitting2DRes3_n, Vfitting2DRes4, Vfitting2DRes4_n, Vfitting2DRes5, Vfitting2DRes5_n;

	int nCorrespond3 = nSMPL2DetectionPairs * 3, nJoints3 = nJoints * 3;
	double *All_V_ptr = new double[nVertices * 3 * nInstances],
		*All_dVdp_ptr = new double[nVertices * 3 * nJoints * 3 * nInstances],
		*All_dVdc_ptr = new double[nVertices * 3 * nShapeCoeffs * nInstances],
		*All_dVds_ptr = new double[nVertices * 3 * nInstances],
		*All_aJsmpl_ptr = new double[naJoints3 * nInstances],
		*All_J_ptr = new double[nCorrespond3 * nInstances],
		*All_dJdt_ptr = new double[nCorrespond3 * 3 * nInstances],
		*All_dJdp_ptr = new double[nCorrespond3 * nJoints * 3 * nInstances],
		*All_dJdc_ptr = new double[nCorrespond3 * nShapeCoeffs * nInstances],
		*All_dJds_ptr = new double[nCorrespond3 * nInstances];
	char *All_VertexTypeAndVisibility = 0;
	Point2f *All_uv_ptr = 0;
	if (hasDensePose)
		All_VertexTypeAndVisibility = new char[nVertices*nInstances],
		All_uv_ptr = new Point2f[nVertices*nInstances];

	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> J_reg = skeletonPointFormat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
	SparseMatrix<double, ColMajor>  dVdt = kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	double startTime = omp_get_wtime();
	ceres::Problem problem;
	ceres::Solver::Options options;

	//Evaluator callback
	printLOG("Setting up evalution callback\n");
	vector<double *> Vparameters;
	Vparameters.push_back(&mySMPL.scale);
	Vparameters.push_back(mySMPL.coeffs.data());
	for (int idf = 0; idf < nInstances; idf++)
	{
		Vparameters.push_back(frame_params[idf].pose.data());
		Vparameters.push_back(frame_params[idf].t.data());
	}

	SmplFitCallBackUnSync mySmplFitCallBack(mySMPL, nInstances, skeletonPointFormat, Vparameters, All_V_ptr, All_uv_ptr, All_dVdp_ptr, All_dVdc_ptr, All_dVds_ptr, All_aJsmpl_ptr, All_J_ptr, All_dJdt_ptr, All_dJdp_ptr, All_dJdc_ptr, All_dJds_ptr,
		vDensePose, VideoInfo, All_VertexTypeAndVisibility, maxWidth*maxHeight, hasDensePose);
	mySmplFitCallBack.PrepareForEvaluation(false, true);
	options.evaluation_callback = &mySmplFitCallBack;

	//Iterator Callback function
	printLOG("Setting up iterator callback\n");
	int iter = 0, debug = 0;
	Point2d *uvJ = new Point2d[nSMPL2DetectionPairs], *uvV = new Point2d[nVertices];
	bool *hit = new bool[maxWidth*maxHeight];
	class MyCallBack : public ceres::IterationCallback
	{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &debug) : callback_(callback), iter(iter), debug(debug) {}
		virtual ~MyCallBack() {}

		ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
		{
			iter = summary.iteration;
			if (summary.step_is_successful)
				callback_();
			return ceres::SOLVER_CONTINUE;
		}
		int &iter, &debug;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
#ifdef _WINDOWS
		int idf = 0, cidI = 0;
		if (debug > 0)
		{
			for (int idf = 0; idf < nInstances; idf++)
			{
				int idf1 = idf + 1;
				double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf;
				char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * idf;

				int rcid = vDensePose[idf].cid, rfid = vDensePose[idf].fid, width = VideoInfo[rcid].VideoInfo[rfid].width, height = VideoInfo[rcid].VideoInfo[rfid].height;
				CameraData *camI = VideoInfo[rcid].VideoInfo;
				if (hasDensePose && vDensePose[idf].valid == 1)
				{
					char Fname[512]; sprintf(Fname, "G:/NEA1/Corrected/%d/%.4d.jpg", rcid, rfid);
					Mat img = imread(Fname);
					//get occluding contour vertices
#pragma omp parallel for schedule(dynamic,1)
					for (int vid = 0; vid < nVertices; vid++)
					{
						int vid3 = vid * 3;
						Point3d xyz(V0_ptr[vid3], V0_ptr[vid3 + 1], V0_ptr[vid3 + 2]);
						ProjectandDistort(xyz, &uvV[vid], camI[rfid].P);// , camI[rfid].K, camI[rfid].distortion);

						if (uvV[vid].x<15 || uvV[vid].x>width - 15 || uvV[vid].y<15 || uvV[vid].y>height - 15)
							uvV[vid].x = 0, uvV[vid].y = 0;
					}
					for (int vid = 0; vid < nVertices; vid++)
					{
						if (EdgeVertexTypeAndVisibilityF[vid] > -1)
							circle(img, uvV[vid], 1, Scalar(0, 255, 0));
						else
							circle(img, uvV[vid], 1, Scalar(0, 0, 255));
					}
					sprintf(Fname, "E:/A1/Vis/FitBody/x.jpg"), imwrite(Fname, img);

					sprintf(Fname, "C:/temp/%.4d_%d_oc_%d.txt", rcid, rfid, personId);	FILE *fp = fopen(Fname, "w");
					for (int vid = 0; vid < nVertices; vid++)
						if (EdgeVertexTypeAndVisibilityF[vid] > -1)
							fprintf(fp, "%d %d %.1f %.1f \n", vid, EdgeVertexTypeAndVisibilityF[vid], uvV[vid].x, uvV[vid].y);
					fclose(fp);
				}
			}
		}
		iter++;
#endif
	};
	options.callbacks.push_back(new MyCallBack(update_Result, iter, debug));

	debug = 0;
	MyCallBack *myCallback = new MyCallBack(update_Result, iter, debug);
	myCallback->callback_();

	//Ceres residual blocks
	printLOG("Setting up residual blocks\n");
	ceres::LossFunction* coeffs_loss = new ceres::ScaledLoss(NULL, w0, ceres::TAKE_OWNERSHIP);
	ceres::CostFunction *coeffs_reg = new ceres::AutoDiffCostFunction	< SMPLShapeCoeffRegCeres, nShapeCoeffs, nShapeCoeffs >(new SMPLShapeCoeffRegCeres(nShapeCoeffs));
	if (w0 > 0.0)
		problem.AddResidualBlock(coeffs_reg, coeffs_loss, mySMPL.coeffs.data());

	const double * parameters[] = { mySMPL.coeffs.data() };
	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0 *0.5*residuals[ii] * residuals[ii]);

	//FILE*fpOut = fopen("ResUnSync.txt", "a+");
	Map< MatrixXdr > Mosh_pose_prior_mu(mySMPL.Mosh_pose_prior_mu, nJoints * 3, 1);
	for (int instId = 0; instId < nInstances; instId++)
	{
		int idf = SMPL_TimeOrder[instId];

		//pose prior
		ceres::LossFunction* pose_regl_loss = new ceres::ScaledLoss(NULL, w1, ceres::TAKE_OWNERSHIP);
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
		if (w1 > 0.0)
			problem.AddResidualBlock(pose_reg, pose_regl_loss, frame_params[idf].pose.data());

		const double * parameters1[] = { frame_params[idf].pose.data() };
		pose_reg->Evaluate(parameters1, residuals, NULL);
		for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
			VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

		//for (int ii = 3; ii < nJoints * 3; ii++)
		//{
		//	problem.SetParameterLowerBound(frame_params[idf].pose.data(), ii, mySMPL.minPose[ii]);
		//	problem.SetParameterUpperBound(frame_params[idf].pose.data(), ii, mySMPL.maxPose[ii]);
		//}

		//edge fittimg cost and point fitting
		int nValidPoints = 0;
		double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
		double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
		double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
		char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * idf;
		Point2f *vUV_ptr = All_uv_ptr + nVertices * idf;

		if (hasDensePose)
		{
			double  isigma2 = isigma_2DSeg * isigma_2DSeg;
			int rcid = vDensePose[idf].cid, rfid = vDensePose[idf].fid;

			//1. Contour fiting
			if (w3 > 0.0)
			{
				nValidPoints = 0;
				if (vDensePose[idf].valid == 1 && VideoInfo[rcid].VideoInfo[rfid].valid == 1)
				{
					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						if (EdgeVertexTypeAndVisibilityF[vid] > -1 && vDensePose[idf].validParts[dfId] > 0)
							nValidPoints++;
					}
					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);
						SmplFitSMPL2EdgeCeres_MV2 *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV2(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, vDensePose[idf],
							EdgeVertexTypeAndVisibilityF, VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						if (EdgeVertexTypeAndVisibilityF[vid] == -1 || vDensePose[idf].validParts[dfId] == 0)
							continue;

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
					}

					//other direction
					nValidPoints = 0;
					for (int partId = 0; partId < 14; partId++)
						nValidPoints += (int)vDensePose[idf].vEdge[partId].size() / 2;
					for (int partId = 0; partId < 14; partId++)
					{
						for (int eid = 0; eid < vDensePose[idf].vEdge[partId].size(); eid += 2)
						{
							double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
							//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
							ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
							ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);
							//SmplFitEdge2SMPLCeres_MV *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
							//	EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

							SmplFitEdge2SMPLCeres_MV2 *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV2(V_ptr, vUV_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, vDensePose[idf], vMergedPartId2DensePosePartId,
								EdgeVertexTypeAndVisibilityF, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, vDensePose[idf].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
							problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
							resX = resX / isigma_2DSeg, resY = resY / isigma_2DSeg;
							robust_loss->Evaluate(res2, rho);
							Vfitting2DRes2_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
						}
					}
				}
			}

			//2. Sil fitting
			if (w4 > 0.0 &&vDensePose[idf].valid&& VideoInfo[rcid].VideoInfo[rfid].valid)
			{
				nValidPoints = nVertices;
				double nw4 = w4 / nValidPoints;
				for (int vid = 0; vid < nVertices; vid++)
				{
					ceres::LossFunction *robust_loss = new ceres::CauchyLoss(1.0); //works better than huber////new GermanMcClure(GermanMcClure_Scale, 10.0*SilScale*GermanMcClure_Curve_Influencier);
					ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw4, ceres::TAKE_OWNERSHIP);

					SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
						mySMPL, vDensePose[idf], VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
					problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double res2 = residuals[0] * residuals[0];
					robust_loss->Evaluate(res2, rho);
					Vfitting2DRes3_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes3.push_back(res2 / isigma2);
				}
			}
		}

		//3. Point fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
			{
				int DetectionId = SMPL2Detection[2 * ii + 1];
				if (frame_skeleton[idf].pt2D[DetectionId].x != 0.0)
					nValidPoints++;
			}

			for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
			{
				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				if (frame_skeleton[idf].pt2D[DetectionId].x != 0.0)
				{
					SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
						mySMPL, &frame_skeleton[idf].P[12 * DetectionId], SMPLid, frame_skeleton[idf].pt2D[DetectionId], skeletonPointFormat, isigma_2D); //can work with Rolling RS where each point has diff pose

					double nw5 = w5 * JointWeight[ii] * frame_skeleton[idf].confidence[DetectionId] / (0.0001 + nValidPoints);
					ceres::LossFunction* robust_loss = new HuberLoss(1);
					ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
					problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double resX = residuals[0], resY = residuals[1];
					robust_loss->Evaluate(resX*resX + resY * resY, rho);
					Vfitting2DRes4_n.push_back(nw5*0.5*rho[0]);

					resX = resX / isigma_2D, resY = resY / isigma_2D;
					Vfitting2DRes4.push_back(resX), Vfitting2DRes4.push_back(resY);
					//fprintf(fpOut, "%d %.2f %.2f\n", SMPLid, resX, resY);
				}
			}
		}

		//4. smpl2Pts3d fitting
		if (nInstances == 1)
		{
			nValidPoints = 0;
			for (int pairID = 0; pairID < nSMPL2DetectionPairs; pairID++) //COCO-SMPL pair
				if (IsValid3D(frame_skeleton[idf].pt3D[SMPL2Detection[2 * pairID + 1]]))
					nValidPoints++;
			for (int ii = 0; ii < nSMPL2DetectionPairs; ii++)
			{
				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, frame_skeleton[idf].pt3D[DetectionId], skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw5 = w5 * JointWeight[ii] / (0.0001 + nValidPoints);
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				robust_loss->Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				Vfitting2DRes5_n.push_back(nw5*0.5*rho[0]);

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes5.push_back(resX), Vfitting2DRes5.push_back(resY), Vfitting2DRes5.push_back(resZ);
			}
		}
	}
	if (w2 > 0.0)
	{
		for (int instId = 0; instId < nInstances - 1; instId++)
		{
			//the agumented joints consist of finger tips and toes which lock random joint twist
			int idf = SMPL_TimeOrder[instId], idf1 = SMPL_TimeOrder[instId + 1];

			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			//ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2, ceres::TAKE_OWNERSHIP);
			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, frame_skeleton[idf][0].ts, frame_skeleton[idf1][0].ts, isigma_Vel);
			//aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//	V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, frame_skeleton[idf].ts, frame_skeleton[idf1].ts, isigma_Vel);

			ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2 / nVertices, ceres::TAKE_OWNERSHIP);
			SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, frame_skeleton[idf].ts, frame_skeleton[idf1].ts, isigma_Vel);
			problem.AddResidualBlock(temporal_cost_analytic_fr, avgl_loss, &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf1].t.data(), frame_params[idf1].pose.data());

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf1].t.data(), frame_params[idf1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
				//for (int ii = 0; ii < nVertices * 3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = frame_skeleton[idf1].ts - frame_skeleton[idf].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
				//VtemporalRes_n.push_back(w2 / nVertices * 0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / sqrt(dt)*sqrt(nVertices * 3) / Real2SfM, 2)); //in mm/s
			}
		}
	}

	/*fclose(fpOut);
	delete[]hit;
	delete[]SMPL2Detection, delete[]uvJ, delete[]uvV;
	delete[]residuals;

	delete[]All_VertexTypeAndVisibility, delete[]All_uv_ptr;
	delete[]All_V_ptr, delete[]All_dVdp_ptr, delete[]All_dVdc_ptr, delete[]All_dVds_ptr;
	delete[]All_aJsmpl_ptr, delete[]All_J_ptr, delete[]All_dJdt_ptr, delete[]All_dJdp_ptr, delete[]All_dJdc_ptr, delete[]All_dJds_ptr;
	return 0;*/

	//fix frames from previous window
	if (fixedPoseFrame.x != -1 && fixedPoseFrame.y != -1)
	{
		problem.SetParameterBlockConstant(&mySMPL.scale);
		problem.SetParameterBlockConstant(mySMPL.coeffs.data());
		for (int idf = 0; idf < nInstances; idf++)
		{
			if (frame_skeleton[idf].refFrameID >= fixedPoseFrame.x && frame_skeleton[idf].refFrameID <= fixedPoseFrame.y)
			{
				problem.SetParameterBlockConstant(frame_params[idf].t.data());
				problem.SetParameterBlockConstant(frame_params[idf].pose.data());
			}
		}
	}
	problem.SetParameterBlockConstant(&mySMPL.scale);
	problem.SetParameterBlockConstant(mySMPL.coeffs.data());

	{
		Vfitting2DRes5.push_back(0), Vfitting2DRes5_n.push_back(0), Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
		Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
		Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
		Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
		Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
		Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
		Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
		Map< VectorXd > eVfitting2DRes3(&Vfitting2DRes3[0], Vfitting2DRes3.size());
		Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
		Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());
		Map< VectorXd > eVfitting2DRes5_n(&Vfitting2DRes5_n[0], Vfitting2DRes5_n.size());

		double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
			sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
			sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
			sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
			sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size()),
			sos_Vfitting2DRes3 = eVfitting2DRes3.sum(), rmse_Vfitting2DRes3 = sqrt(sos_Vfitting2DRes3 / Vfitting2DRes3.size()),
			sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size());
		double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
			mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
		double sos_Vfitting2DRes5_n = eVfitting2DRes5_n.sum(), rmse_Vfitting2DRes5_n = sqrt(sos_Vfitting2DRes5_n / Vfitting2DRes5_n.size()),
			mu5 = MeanArray(Vfitting2DRes5), stdev5 = sqrt(VarianceArray(Vfitting2DRes5, mu5));
		if (nInstances > 1)
		{
			Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
			double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
			printLOG("Before Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
		}
		else
		{
			printLOG("Before Optim\nInit scale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			if (nInstances == 1)
				printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
		}
	}

	options.num_threads = omp_get_max_threads();
	options.eta = 1e-3;
	options.dynamic_sparsity = true;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
#ifndef _DEBUG
	options.minimizer_progress_to_stdout = frame_params.size() > 1 ? true : false;
#else
	options.minimizer_progress_to_stdout = true;
#endif
	options.update_state_every_iteration = true;
	options.max_num_iterations = 100;

#ifndef _DEBUG
	if (frame_params.size() < 5)
		options.max_solver_time_in_seconds = 300;
	else if (frame_params.size() < 50)
		options.max_solver_time_in_seconds = 500;
	else if (frame_params.size() < 500)
		options.max_solver_time_in_seconds = 700;
	else if (frame_params.size() < 1500)
		options.max_solver_time_in_seconds = 1000;
	else
		options.max_solver_time_in_seconds = 1200;
#endif // !_DEBUG

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	mySmplFitCallBack.PrepareForEvaluation(false, true);

	//debug = 2;
	myCallback->callback_();

	VshapePriorCeres.clear(), VposePriorCeres.clear(), VtemporalRes_n.clear(), VtemporalRes.clear(), Vfitting2DRes1_n.clear(), Vfitting2DRes1.clear();
	Vfitting2DRes2_n.clear(), Vfitting2DRes2.clear(), Vfitting2DRes3.clear(), Vfitting2DRes3_n.clear(), Vfitting2DRes4.clear(), Vfitting2DRes4_n.clear(), Vfitting2DRes5.clear(), Vfitting2DRes5_n.clear();

	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0  *0.5*residuals[ii] * residuals[ii]);

	for (int instId = 0; instId < nInstances; instId++)
	{
		int idf = SMPL_TimeOrder[instId];

		//pose prior
		ceres::LossFunction* pose_regl_loss = new ceres::ScaledLoss(NULL, w1, ceres::TAKE_OWNERSHIP);
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
		problem.AddResidualBlock(pose_reg, pose_regl_loss, frame_params[idf].pose.data());

		const double * parameters1[] = { frame_params[idf].pose.data() };
		pose_reg->Evaluate(parameters1, residuals, NULL);
		for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
			VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

		for (int ii = 3; ii < nJoints * 3; ii++)
		{
			problem.SetParameterLowerBound(frame_params[idf].pose.data(), ii, mySMPL.minPose[ii]);
			problem.SetParameterUpperBound(frame_params[idf].pose.data(), ii, mySMPL.maxPose[ii]);
		}

		//edge fittimg cost and point fitting
		int nValidPoints = 0;
		double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
		double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
		double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
		char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * idf;
		Point2f *vUV_ptr = All_uv_ptr + nVertices * idf;

		if (hasDensePose)
		{
			//1. Contour fiting
			double  isigma2 = isigma_2DSeg * isigma_2DSeg;
			int rcid = vDensePose[idf].cid, rfid = vDensePose[idf].fid;

			if (w3 > 0.0 &&vDensePose[idf].valid&& VideoInfo[rcid].VideoInfo[rfid].valid)
			{
				nValidPoints = 0;
				for (int vid = 0; vid < nVertices; vid += 2)
				{
					int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
					if (EdgeVertexTypeAndVisibilityF[vid] > -1 && vDensePose[idf].validParts[dfId] > 0)
						nValidPoints++;
				}
				for (int vid = 0; vid < nVertices; vid += 2)
				{
					int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
					double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
					//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
					ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
					SmplFitSMPL2EdgeCeres_MV2 *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV2(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, vDensePose[idf],
						EdgeVertexTypeAndVisibilityF, VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

					if (EdgeVertexTypeAndVisibilityF[vid] == -1 || vDensePose[idf].validParts[dfId] == 0)
						continue;

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
					robust_loss->Evaluate(res2, rho);
					Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
				}

				//other direction
				nValidPoints = 0;
				for (int partId = 0; partId < 14; partId++)
					nValidPoints += (int)vDensePose[idf].vEdge[partId].size() / 2;
				for (int partId = 0; partId < 14; partId++)
				{
					for (int eid = 0; eid < vDensePose[idf].vEdge[partId].size(); eid += 2)
					{
						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);

						SmplFitEdge2SMPLCeres_MV2 *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV2(V_ptr, vUV_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, vDensePose[idf], vMergedPartId2DensePosePartId,
							EdgeVertexTypeAndVisibilityF, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, vDensePose[idf].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
						resX = resX / isigma_2DSeg, resY = resY / isigma_2DSeg;
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes2_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
					}
				}
			}

			//2. Sil fitting
			if (w4 > 0.0 &&vDensePose[idf].valid&& VideoInfo[rcid].VideoInfo[rfid].valid)
			{
				nValidPoints = nVertices;
				double nw4 = w4 / nValidPoints;
				for (int vid = 0; vid < nVertices; vid++)
				{
					ceres::LossFunction *robust_loss = new ceres::CauchyLoss(1.0); //works better than huber////new GermanMcClure(GermanMcClure_Scale, 10.0*SilScale*GermanMcClure_Curve_Influencier);

					SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
						mySMPL, vDensePose[idf], VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double res2 = residuals[0] * residuals[0];
					robust_loss->Evaluate(res2, rho);
					Vfitting2DRes3_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes3.push_back(res2 / isigma2);
				}
			}
		}

		//3. Point fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
			{
				int DetectionId = SMPL2Detection[2 * ii + 1];
				if (frame_skeleton[idf].pt2D[DetectionId].x != 0.0)
					nValidPoints++;
			}

			for (int ii = 0; ii < nSMPL2DetectionPairs; ii++) //COCO-SMPL pair
			{
				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				if (frame_skeleton[idf].pt2D[DetectionId].x != 0.0)
				{
					SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
						mySMPL, &frame_skeleton[idf].P[12 * DetectionId], SMPLid, frame_skeleton[idf].pt2D[DetectionId], skeletonPointFormat, isigma_2D); //can work with Rolling RS where each point has diff pose

					double nw5 = w5 * JointWeight[ii] * frame_skeleton[idf].confidence[DetectionId] / (0.0001 + nValidPoints);
					ceres::LossFunction* robust_loss = new HuberLoss(1);

					fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
					double resX = residuals[0], resY = residuals[1];
					robust_loss->Evaluate(resX*resX + resY * resY, rho);
					Vfitting2DRes4_n.push_back(nw5*0.5*rho[0]);

					resX = resX / isigma_2D, resY = resY / isigma_2D;
					Vfitting2DRes4.push_back(resX), Vfitting2DRes4.push_back(resY);
				}
			}
		}

		//4. smpl2Pts3d fitting
		if (nInstances == 1)
		{
			nValidPoints = 0;
			for (int pairID = 0; pairID < nSMPL2DetectionPairs; pairID++) //COCO-SMPL pair
				if (IsValid3D(frame_skeleton[idf].pt3D[SMPL2Detection[2 * pairID + 1]]))
					nValidPoints++;
			for (int ii = 0; ii < nSMPL2DetectionPairs; ii++)
			{
				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, frame_skeleton[idf].pt3D[DetectionId], skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw5 = w5 * JointWeight[ii] / (0.0001 + nValidPoints);
				ceres::LossFunction* robust_loss = new HuberLoss(1);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				robust_loss->Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				Vfitting2DRes5_n.push_back(nw5*0.5*rho[0]);

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes5.push_back(resX), Vfitting2DRes5.push_back(resY), Vfitting2DRes5.push_back(resZ);
			}
		}
	}
	if (w2 > 0.0)
	{
		for (int instId = 0; instId < nInstances - 1; instId++)
		{
			//the agumented joints consist of finger tips and toes which lock random joint twist
			int idf = SMPL_TimeOrder[instId], idf1 = SMPL_TimeOrder[instId + 1];

			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			//ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2, ceres::TAKE_OWNERSHIP);
			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, frame_skeleton[idf][0].ts, frame_skeleton[idf1][0].ts, isigma_Vel);
			//aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, frame_skeleton[idf].ts, frame_skeleton[idf1].ts, isigma_Vel);

			ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2 / nVertices, ceres::TAKE_OWNERSHIP);
			SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, frame_skeleton[idf].ts, frame_skeleton[idf1].ts, isigma_Vel);

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf1].t.data(), frame_params[idf1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
				//for (int ii = 0; ii < nVertices * 3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = frame_skeleton[idf1].ts - frame_skeleton[idf].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
				//VtemporalRes_n.push_back(w2 / nVertices * 0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / sqrt(dt)*sqrt(nVertices * 3) / Real2SfM, 2)); //in mm/s
			}
		}
	}

	{
		Vfitting2DRes5.push_back(0), Vfitting2DRes5_n.push_back(0), Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
		Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
		Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
		Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
		Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
		Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
		Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
		Map< VectorXd > eVfitting2DRes3(&Vfitting2DRes3[0], Vfitting2DRes3.size());
		Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
		Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());
		Map< VectorXd > eVfitting2DRes5_n(&Vfitting2DRes5_n[0], Vfitting2DRes5_n.size());

		double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
			sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
			sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
			sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
			sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size()),
			sos_Vfitting2DRes3 = eVfitting2DRes3.sum(), rmse_Vfitting2DRes3 = sqrt(sos_Vfitting2DRes3 / Vfitting2DRes3.size()),
			sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size());
		double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
			mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
		double sos_Vfitting2DRes5_n = eVfitting2DRes5_n.sum(), rmse_Vfitting2DRes5_n = sqrt(sos_Vfitting2DRes5_n / Vfitting2DRes5_n.size()),
			mu5 = MeanArray(Vfitting2DRes5), stdev5 = sqrt(VarianceArray(Vfitting2DRes5, mu5));
		if (nInstances > 1)
		{
			Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
			double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
			printLOG("After Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
		}
		else
		{
			printLOG("After Optim\nInit scale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			if (nInstances == 1)
				printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
		}
	}

	delete[]hit;
	delete[]SMPL2Detection, delete[]uvJ, delete[]uvV;
	delete[]residuals;

	delete[]All_VertexTypeAndVisibility, delete[]All_uv_ptr;
	delete[]All_V_ptr, delete[]All_dVdp_ptr, delete[]All_dVdc_ptr, delete[]All_dVds_ptr;
	delete[]All_aJsmpl_ptr, delete[]All_J_ptr, delete[]All_dJdt_ptr, delete[]All_dJdp_ptr, delete[]All_dJdc_ptr, delete[]All_dJds_ptr;

	return 0.0;
}

int SMPLVertices2DensePoseUV(char *Path, int nCams, int startF, int stopF, int maxPeople = 20)
{
	char Fname[512], Fname1[512], Fname2[512];
	const int nVertices = SMPLModel::nVertices, DPnParts = 24;

	float U, V;
	int smplVid, PartId;
	vector<int> vMergedPartId2DensePosePartId(nVertices);
	vector<Point2f> vUV(nVertices);
	FILE *fp = fopen("smpl/SMPL_2_UV_PID.txt", "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return false;
	}
	else
	{
		while (fscanf(fp, "%d %f %f %d ", &smplVid, &U, &V, &PartId) != EOF)
			vUV[smplVid].x = U * 255.0, vUV[smplVid].y = V * 255.0, vMergedPartId2DensePosePartId[smplVid] = PartId;
		fclose(fp);
	}

	Mat PersonIdImage, IUV;
	vector<uint16_t*> Vid;
	vector<int> nVid;
	for (int peopleId = 0; peopleId < maxPeople; peopleId++)
	{
		for (int partId = 0; partId < DPnParts; partId++)
		{
			uint16_t *id = new uint16_t[1920 * 1080]; //already very large number
			Vid.push_back(id);
			nVid.push_back(0);
		}
	}

	vector<int> UniquePeopleId;
	int *smplVidImageXY = new int[nVertices * 3];
	float *RetrievalError = new float[nVertices];

	printLOG("Extracting mapping between SMPL and DensPose\n");
	for (int sCamId = 0; sCamId < nCams; sCamId++)
	{
		sprintf(Fname, "%s/smplVidImageXY", Path); makeDir(Fname);
		sprintf(Fname, "%s/smplVidImageXY/%d", Path, sCamId);  makeDir(Fname);
		printLOG("Working on cam %d: ", sCamId);
		double startTime = omp_get_wtime();

		omp_set_num_threads(omp_get_max_threads());

		for (int fid = startF; fid <= stopF; fid++)
		{
			printf("%d..", fid);
			sprintf(Fname, "%s/smplVidImageXY/%d/%.4d_00.txt", Path, sCamId, fid);
			//if (IsFileExist(Fname) == 1) //1st person exist, others must have been computed as well
			//	continue;

			sprintf(Fname1, "%s/DensePose/Corrected/%d/%.4d_INDS.png", Path, sCamId, fid);
			sprintf(Fname2, "%s/DensePose/Corrected/%d/%.4d_IUV.png", Path, sCamId, fid);
			if (IsFileExist(Fname1) == 0 || IsFileExist(Fname2) == 0)
			{
				sprintf(Fname1, "%s/DensePose/%d/%.4d_INDS.png", Path, sCamId, fid);
				sprintf(Fname2, "%s/DensePose/%d/%.4d_IUV.png", Path, sCamId, fid);
				if (IsFileExist(Fname1) == 0 || IsFileExist(Fname2) == 0)
					continue;
			}

			PersonIdImage = imread(Fname1, 0);
			IUV = imread(Fname2); //reading as IUV. Note that Matlab read as VUI

			int width = IUV.cols, height = IUV.rows, length = width * height;
			for (int ii = 0; ii < maxPeople*DPnParts; ii++)
				nVid[ii] = 0;

			//identify the human region
			UniquePeopleId.clear();
			for (int kk = 0; kk < height; kk++)
			{
				for (int ll = 0; ll < width; ll++)
				{
					int ii = kk * width + ll;
					if (IUV.data[3 * ii + 1] > (uchar)0 && IUV.data[3 * ii + 2] > (uchar)0) //use 2nd channel as the part in the 1st channel could be 0 index
					{
						//PersonIdImage does not produce peopleId with arbitary index
						int peopleId = (int)PersonIdImage.data[ii], upeopleId = -1;
						for (int jj = 0; jj < UniquePeopleId.size() && upeopleId == -1; jj++)
							if (UniquePeopleId[jj] == peopleId)
								upeopleId = jj;
						if (upeopleId == -1)
							UniquePeopleId.push_back(peopleId), upeopleId = (int)UniquePeopleId.size() - 1;

						int partId = (int)IUV.data[3 * ii] - 1, id = DPnParts * upeopleId + partId;
						Vid[id][nVid[id]] = (uint16_t)ii, nVid[id]++;
					}
				}
			}

			//for each person, find the uv corresponding to smpl vertices
			for (int peopleId = 0; peopleId < UniquePeopleId.size(); peopleId++)
			{
				int cnt = 0;
				for (int partId = 0; partId < DPnParts; partId++)
					cnt += nVid[DPnParts*peopleId + partId];
				if (cnt < 8100 * width / 1920)//90x90
					continue;

#pragma omp parallel for schedule(dynamic,4)
				for (int vid = 0; vid < nVertices; vid++)
				{
					int vid3 = 3 * vid;
					smplVidImageXY[vid3] = -1;
					RetrievalError[vid] = -1;

					int partId = vMergedPartId2DensePosePartId[vid];
					float u = vUV[vid].x, v = vUV[vid].y;

					double minDist = 25;
					int BestPixelId = 0;
					for (int ii = 0; ii < nVid[DPnParts*peopleId + partId]; ii++)
					{
						int pixelId = Vid[DPnParts*peopleId + partId][ii], dpu = (int)IUV.data[3 * pixelId + 1], dpv = (int)IUV.data[3 * pixelId + 2];
						double dist = pow(u - dpu, 2) + pow(v - dpv, 2); //any distances should be fine
						if (dist < minDist)
							minDist = dist, BestPixelId = pixelId;
					}

					int x = BestPixelId % width, y = BestPixelId / width;
					if (minDist < 4)
					{
						smplVidImageXY[3 * vid] = vid, smplVidImageXY[3 * vid + 1] = x, smplVidImageXY[3 * vid + 2] = y;
						RetrievalError[vid] = minDist;
					}
				}

				sprintf(Fname, "%s/smplVidImageXY/%d/%.4d_%.2d.txt", Path, sCamId, fid, UniquePeopleId[peopleId]); FILE *fp = fopen(Fname, "w");
				for (int vid = 0; vid < nVertices; vid++)
				{
					int vid3 = 3 * vid;
					if (smplVidImageXY[3 * vid] > -1)
						fprintf(fp, "%d %d %d %.2f\n", smplVidImageXY[3 * vid], smplVidImageXY[3 * vid + 1], smplVidImageXY[3 * vid + 2], RetrievalError[vid]);
				}
				fclose(fp);
			}
		}
		printLOG("..Done. Take: %2fs\n", omp_get_wtime() - startTime);
	}

	delete[]smplVidImageXY, delete[]RetrievalError;
	for (int ii = 0; ii < maxPeople*DPnParts; ii++)
		delete[]Vid[ii];

	return 0;
}
int InitializeBodyPoseParameters1Frame(char *Path, int frameID, int increF, vector<HumanSkeleton3D> &vSkeletons, Point3d *RestJoints, int nPointFormat)
{
	char Fname[512];
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	bool done = true;
	for (int pi = 0; pi < vSkeletons.size() && done; pi++)
	{
		sprintf(Fname, "%s/FitBody/@%d/i/%d/%.4d.txt", Path, increF, pi, frameID);
		if (!IsFileExist(Fname))
			done = false;
	}
	if (done)
		return 1;

	sprintf(Fname, "%s/FitBody/@%d/i", Path, increF); makeDir(Fname);
	for (int pi = 0; pi < vSkeletons.size(); pi++)
		sprintf(Fname, "%s/FitBody/@%d/i/%d", Path, increF, pi), makeDir(Fname);

	int TemplateTorsoJointID[5] = { 2, 1, 5, 8, 11 }, TargetTorsoJointID[5];
	if (nPointFormat == 17)
		TargetTorsoJointID[0] = 6, TargetTorsoJointID[1] = 0, TargetTorsoJointID[2] = 5, TargetTorsoJointID[3] = 12, TargetTorsoJointID[4] = 11;
	else if (nPointFormat == 18)
		TargetTorsoJointID[0] = 2, TargetTorsoJointID[1] = 1, TargetTorsoJointID[2] = 5, TargetTorsoJointID[3] = 8, TargetTorsoJointID[4] = 11;
	else if (nPointFormat == 25)
		TargetTorsoJointID[0] = 3, TargetTorsoJointID[1] = 2, TargetTorsoJointID[2] = 6, TargetTorsoJointID[3] = 9, TargetTorsoJointID[4] = 12;

	vector<Point3d> RestTorso, CurTorso;

	int numPeople = (int)vSkeletons.size();
	vector<int> invalidInit;
	vector<double> vs;
	vector<Point3d> vr, vT;
	for (int pi = 0; pi < numPeople; pi++)
	{
		if (vSkeletons[pi].valid == 0)
			continue;

		//make sure that at least centroid and orientation are aligned.
		double r[3], R[9], T[3], s;
		//get Torso joints
		RestTorso.clear(), CurTorso.clear();
		for (int ii = 0; ii < 5; ii++)
			if (abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].x) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].y) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].z) > 1e-16)
				RestTorso.push_back(RestJoints[TemplateTorsoJointID[ii]]), CurTorso.push_back(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]]);

		if (RestTorso.size() < 3)
			invalidInit.push_back(pi);

		double error3D = computeProcrustesTransform(RestTorso, CurTorso, R, T, s, true);
		getrFromR(R, r);

		vs.push_back(s), vr.push_back(Point3d(r[0], r[1], r[2])), vT.push_back(Point3d(T[0], T[1], T[2]));

		vSkeletons[pi].s = s;
		for (int ii = 0; ii < 3; ii++)
			vSkeletons[pi].r[ii] = r[ii], vSkeletons[pi].t[ii] = T[ii];
	}

	if (invalidInit.size() == numPeople)
		return 1;

	double sMean = MeanArray(vs);
	for (int jj = 0; jj < invalidInit.size(); jj++)
	{
		int pi = invalidInit[jj];
		RestTorso.clear(), CurTorso.clear();
		for (int ii = 0; ii < 5; ii++)
			if (abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].x) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].y) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].z) > 1e-16)
				RestTorso.push_back(RestJoints[TemplateTorsoJointID[ii]]), CurTorso.push_back(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]]);

		if (RestTorso.size() == 0)
			vSkeletons[pi].s = 0.0;
		else
		{
			vSkeletons[pi].s = sMean;
			for (int ii = 0; ii < 3; ii++)
				vSkeletons[pi].r[ii] = 0.0;

			double tX = 0, tY = 0, tZ = 0;
			for (int ii = 0; ii < RestTorso.size(); ii++)
			{
				double orgX = RestTorso[ii].x, orgY = RestTorso[ii].y, orgZ = RestTorso[ii].z, fX = CurTorso[ii].x, fY = CurTorso[ii].y, fZ = CurTorso[ii].z;
				tX += sMean * orgX - fX;
				tY += sMean * orgY - fY;
				tZ += sMean * orgZ - fZ;
			}
			vSkeletons[pi].t[0] = tX / RestTorso.size(), vSkeletons[pi].t[1] = tY / RestTorso.size(), vSkeletons[pi].t[2] = tZ / RestTorso.size();
		}
	}

	for (int pi = 0; pi < (int)vSkeletons.size(); pi++)
	{
		if (!vSkeletons[pi].valid)
			continue;
		sprintf(Fname, "%s/FitBody/@%d/i/%d/%.4d.txt", Path, increF, pi, frameID); FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%e %e %e %e\n%f %f %f ", vSkeletons[pi].s, vSkeletons[pi].t[0], vSkeletons[pi].t[1], vSkeletons[pi].t[2], vSkeletons[pi].r[0], vSkeletons[pi].r[1], vSkeletons[pi].r[2]);
		for (int ii = 1; ii < nJointsSMPL; ii++)
			fprintf(fp, "0.0 0.0 0.0\n");
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fprintf(fp, "0.0 ");
		fclose(fp);
	}

	return 0;
}
int FitSMPL1Frame(char *Path, SMPLModel &mySMPL, int refFid, int increF, vector<HumanSkeleton3D> &vSkeletons, DensePose *vDensePose, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, int Use2DFitting, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose, int nMaxPeople, int selectedPeopleId)
{
	const double TimeScale = 1000000.0;
	char Fname[512];
	sprintf(Fname, "%s/FitBody/@%d/P", Path, increF), makeDir(Fname);
	for (int pi = 0; pi < vSkeletons.size(); pi++)
		sprintf(Fname, "%s/FitBody/@%d/P/%d", Path, increF, pi), makeDir(Fname);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int maxWidth = 0, maxHeight = 0, nsCams = vSCams.size();
	for (int lcid = 0; lcid < nsCams; lcid++)
		maxWidth = max(maxWidth, VideoInfo[vSCams[lcid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vSCams[lcid]].VideoInfo[0].height);

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 17) //specify based on the smpl 24 keypoint Id
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 2, JointWeight[15] = 2, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .1, JointWeight[19] = .1, JointWeight[20] = .1,//rFoot
		JointWeight[21] = .1, JointWeight[22] = .1, JointWeight[23] = .1;//lFoot
	else if (skeletonPointFormat == 18) //openpose 18
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25) //openpose 25
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .1, JointWeight[19] = .1, JointWeight[20] = .1,//rFoot
		JointWeight[21] = .1, JointWeight[22] = .1, JointWeight[23] = .1;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	vector<int> vDP_Vid, vDP_pid;
	vector<Point2d> vDP_uv, vkpts;
	float *Para = new float[maxWidth*maxHeight];
	Mat IUV, INDS;
	for (int pi = 0; pi < vSkeletons.size(); pi++)
	{
		if (!vSkeletons[pi].valid)
			continue;

		printLOG("*****Person %d*******\n", pi);

		double smpl2sfmScale;
		SMPLParams ParaI;

		sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, pi, 0, refFid, TimeScale*refFid);
		if (!IsFileExist(Fname))
		{
			sprintf(Fname, "%s/FitBody/@%d/P/%d/%.2d_%.4d_%.1f.txt", Path, increF, pi, 0, refFid, TimeScale*refFid);
			if (!IsFileExist(Fname))
			{
				sprintf(Fname, "%s/FitBody/@%d/i/%d/%.4d.txt", Path, increF, pi, refFid);
				if (!IsFileExist(Fname))
					continue;
			}
		}
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%lf %lf %lf %lf ", &smpl2sfmScale, &ParaI.t(0), &ParaI.t(1), &ParaI.t(2));
		ParaI.scale = smpl2sfmScale;
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &ParaI.pose(ii, 0), &ParaI.pose(ii, 1), &ParaI.pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fscanf(fp, "%lf ", &ParaI.coeffs(ii));
		fclose(fp);

		ParaI.frame = refFid;
		if (smpl2sfmScale < 1e-16 || !IsNumber(smpl2sfmScale))
			continue; //fail to init

		//since the code is written for multi-frame BA
		vector<SMPLParams> frame_params; frame_params.push_back(ParaI);
		vector<HumanSkeleton3D> frame_skeleton; frame_skeleton.push_back(vSkeletons[pi]);

		//init smpl
		mySMPL.t.setZero(), mySMPL.coeffs.setZero(); // mySMPL.pose.setZero(), 
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);
		mySMPL.scale = smpl2sfmScale;

		Point2i fixedPoseFrames(-1, -1);
		FitSMPL2Total(mySMPL, frame_params, frame_skeleton, vDensePose, CamTimeInfo, VideoInfo, vSCams, ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, Real2SfM, skeletonPointFormat, fixedPoseFrames, pi, hasDensePose);

		sprintf(Fname, "%s/FitBody/@%d/P/%d/%.2d_%.4d_%.1f.txt", Path, increF, pi, 0, refFid, TimeScale*refFid); fp = fopen(Fname, "w+");
		fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[0].t(0), frame_params[0].t(1), frame_params[0].t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fprintf(fp, "%f %f %f\n", frame_params[0].pose(ii, 0), frame_params[0].pose(ii, 1), frame_params[0].pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fprintf(fp, "%f ", mySMPL.coeffs(ii));
		fprintf(fp, "\n");
		fclose(fp);
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;
	delete[]Para;

	return 0;
}
int FitSMPLWindow(char *Path, SMPLModel &mySMPL, HumanSkeleton3D *vSkeleton, DensePose *AllvDensePose, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int startF, int stopF, int increF, Point2i &fixedPoseFrame, int Pid, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose)
{
	const double TimeScale = 1000000.0;

	char Fname[512];
	sprintf(Fname, "%s/FitBody/@%d", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/Wj", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/Wj/%d", Path, increF, Pid); makeDir(Fname);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;
	int maxWidth = 0, maxHeight = 0, nsCams = vSCams.size();
	for (int lcid = 0; lcid < nsCams; lcid++)
		maxWidth = max(maxWidth, VideoInfo[vSCams[lcid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vSCams[lcid]].VideoInfo[0].height);

	int nthreads = omp_get_max_threads();
	omp_set_num_threads(nthreads);

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 17) //org CoCo, specify based on the smpl keypoint Id
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 2, JointWeight[15] = 2, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .1, JointWeight[19] = .1, JointWeight[20] = .1,//rFoot
		JointWeight[21] = .1, JointWeight[22] = .1, JointWeight[23] = .1;//lFoot
	else if (skeletonPointFormat == 18)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .5, JointWeight[19] = .5, JointWeight[20] = .5,//rFoot
		JointWeight[21] = .5, JointWeight[22] = .5, JointWeight[23] = .5;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		///vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	SMPLParams ParaI;
	double smpl2sfmScale;
	vector<int> vsyncFid;
	vector<double> vscale;
	vector<SMPLParams> frame_params;
	vector<HumanSkeleton3D> frame_skeleton;

	Mat IUV, INDS;
	vector<int> vDP_Vid;
	vector<Point2d> vDP_uv;
	float *Para = new float[maxWidth*maxHeight];

	//pre-process the body paras 
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int temp = (refFid - startF) / increF;
		if (refFid >= fixedPoseFrame.x && refFid <= fixedPoseFrame.y)
			sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, 0, refFid, TimeScale*refFid);
		else
		{
			sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, 0, refFid, TimeScale*refFid);
			if (!IsFileExist(Fname))
				sprintf(Fname, "%s/FitBody/@%d/P/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, 0, refFid, TimeScale*refFid);
		}
		if (!IsFileExist(Fname, false))
			continue;
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%lf %lf %lf %lf ", &smpl2sfmScale, &ParaI.t(0), &ParaI.t(1), &ParaI.t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &ParaI.pose(ii, 0), &ParaI.pose(ii, 1), &ParaI.pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fscanf(fp, "%lf ", &ParaI.coeffs(ii));
		fclose(fp);

		ParaI.frame = refFid;
		ParaI.scale = smpl2sfmScale;
		if (smpl2sfmScale < 1e-16 || !IsNumber(smpl2sfmScale) || !IsFiniteNumber(smpl2sfmScale))
		{
			//printLOG("Problem with %s\n", Fname);
			continue; //fail to init
		}

		vscale.push_back(smpl2sfmScale);
		vsyncFid.push_back(refFid);
		frame_params.push_back(ParaI);
		frame_skeleton.push_back(vSkeleton[temp]);
	}

	//detect bad frames and use the nearest neighbor
	vector<bool> invalid(frame_params.size());
	for (int ii = 0; ii < (int)frame_params.size(); ii++)
	{
		invalid[ii] = false;
		if (vscale[ii] > 1.3*Real2SfM / 1000 || vscale[ii] < 0.7*Real2SfM / 1000)
		{
			vscale[ii] = 1.0;
			invalid[ii] = true;
		}
	}
	for (int ii = 0; ii < (int)frame_params.size(); ii++)
	{
		if (!invalid[ii])
			continue;
		for (int jj = 0; jj < (int)frame_params.size(); jj++)
		{
			if (ii + jj<0 || ii + jj>frame_params.size() - 1)
				continue;
			if (!invalid[ii + jj])
			{
				frame_params[ii].t = frame_params[ii + jj].t;
				frame_params[ii].pose = frame_params[ii + jj].pose;
				frame_params[ii].coeffs = frame_params[ii + jj].coeffs;
				break;
			}
		}
	}

	if (vscale.size() > 0)
	{
		smpl2sfmScale = MedianArray(vscale); //robusifier

		 //init smpl
		mySMPL.t.setZero(), mySMPL.pose.setZero(), mySMPL.coeffs.setZero();
		mySMPL.scale = smpl2sfmScale;
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);

		if (fixedPoseFrame.x > -1)
		{
			for (int jj = 0; jj < vsyncFid.size(); jj++)
			{
				if (vsyncFid[jj] >= fixedPoseFrame.x &&vsyncFid[jj] <= fixedPoseFrame.y)
				{
					for (int ii = 0; ii < nShapeCoeffs; ii++)
						mySMPL.coeffs(ii) = frame_params[jj].coeffs(ii);
					mySMPL.scale = frame_params[jj].scale;
					break;
				}
			}
		}

		//Read DensePose data
		if (hasDensePose)
		{
			printLOG("Getting DensePose data: ");
			double startTime = omp_get_wtime();
			for (int inst = 0; inst < vsyncFid.size(); inst++)
			{
				int refFid = vsyncFid[inst], inst2 = (refFid - startF) / increF;
				printLOG("%d..", refFid);

				DensePose *vDensePose = AllvDensePose + nsCams * inst2;
				for (int lcid = 0; lcid < nsCams; lcid++)
				{
					int rcid = -1, rfid = -1;
					for (int jid = 0; jid < skeletonPointFormat && rfid == -1; jid++)
					{
						for (int ii = 0; ii < frame_skeleton[inst].vViewID_rFid[jid].size() && rfid == -1; ii++)
							if (frame_skeleton[inst].vViewID_rFid[jid][ii].x == vSCams[lcid])
								rcid = frame_skeleton[inst].vViewID_rFid[jid][ii].x, rfid = frame_skeleton[inst].vViewID_rFid[jid][ii].y;
					}
					if (rfid == -1)
						continue;

					sprintf(Fname, "%s/DensePose/Corrected/%d/%.4d_IUV.png", Path, rcid, rfid); IUV = imread(Fname);
					sprintf(Fname, "%s/DensePose/Corrected/%d/%.4d_INDS.png", Path, rcid, rfid); INDS = imread(Fname, 0);
					if (IUV.empty() == 1)
					{
						vDensePose[lcid].valid = 0;
						continue;
					}
					else
						vDensePose[lcid].valid = 1, distortionCorrected = 1;
					int width = IUV.cols, height = IUV.rows, length = width * height;

					//Associate skeleton with DP
					vector<Point2d> vkpts;
					for (int jid = 0; jid < skeletonPointFormat; jid++)
					{
						for (int ii = 0; ii < frame_skeleton[inst].vViewID_rFid[jid].size(); ii++)
						{
							if (frame_skeleton[inst].vViewID_rFid[jid][ii].x == rcid)
							{
								vkpts.push_back(frame_skeleton[inst].vPt2D[jid][ii]);
							}
						}
					}

					int DP_pid = 0;
					vector<int> vDP_pid;
					for (int ii = 0; ii < vkpts.size(); ii++)
					{
						if (vkpts[ii].x > 0 && vkpts[ii].x < width - 1 && vkpts[ii].y>0 && vkpts[ii].y < height - 1)
						{
							vDP_pid.push_back((int)INDS.data[(int)(vkpts[ii].x + 0.5) + (int)(vkpts[ii].y + 0.5)*width]);
						}
					}
					if (vDP_pid.size() < 2)
					{
						vDensePose[lcid].valid = 0;
						continue;
					}
					std::sort(vDP_pid.begin(), vDP_pid.end());
					DP_pid = vDP_pid[vDP_pid.size() / 2];

					//precompute distortion map if needed
					vDensePose[lcid].valid = 1, vDensePose[lcid].cid = rcid, vDensePose[lcid].fid = rfid, vDensePose[lcid].width = width, vDensePose[lcid].height = height;

					//compute body mask
					if (CostWeights[4] > 0.0)
					{
#pragma omp parallel for
						for (int jj = 0; jj < height; jj++)
						{
							for (int ii = 0; ii < width; ii++)
							{
								int id = ii + jj * width;
								BinarizeData[id] = ((int)INDS.data[id] == DP_pid && IUV.data[3 * id] > 0) ? 255 : 0, outSideEdge[id] = 0;
							}
						}
						//background mask term: penalizing occluded parts not satisfying the seg (floating parts)
						ComputeDistanceTransform(BinarizeData, float_df, 128, width, height, v, z, DTTps, ADTTps, realADT); //df of edge

#pragma omp parallel for
						for (int ii = 0; ii < length; ii++)
						{
							if (float_df[ii] > 65535)
								vDensePose[lcid].vdfield[14][ii] = 65535;
							else
								vDensePose[lcid].vdfield[14][ii] = uint16_t(float_df[ii]);
						}
					}

					//edge terms
					if (CostWeights[3] > 0.0)
					{
						for (int jj = 15; jj < height - 15; jj++)
						{
							for (int ii = 15; ii < width - 15; ii++)
							{
								int id = ii + jj * width;
								if (abs(BinarizeData[id] - BinarizeData[id + 1]) > 0 || abs(BinarizeData[id] - BinarizeData[id + width]) > 0)
								{
									outSideEdge[id] = 255;
									vDensePose[lcid].vEdge[14].emplace_back(ii, jj);
								}
							}
						}
						for (int partId = 0; partId < 14; partId++)
						{
							//compute part edge
#pragma omp parallel for
							for (int jj = 0; jj < height; jj++)
							{
								for (int ii = 0; ii < width; ii++)
								{
									int id = ii + jj * width;
									BinarizeData[id] = 0, PartEdge[id] = 0;

									bool found = false;
									for (int kk = 0; kk < vMergedPartId2DensePosePartId[partId].size() && !found; kk++)
										if ((int)INDS.data[id] == DP_pid && IUV.data[3 * id] == vMergedPartId2DensePosePartId[partId][kk])
											BinarizeData[id] = 255;
								}
							}

							int nValidEdge = 0;
							for (int jj = 0; jj < height - 1; jj++)
							{
								for (int ii = 0; ii < width - 1; ii++)
								{
									int id = ii + jj * width;
									if ((abs(BinarizeData[id] - BinarizeData[id + 1]) > 0 || abs(BinarizeData[id] - BinarizeData[id + width]) > 0) && outSideEdge[id] == 255)
									{
										vDensePose[lcid].vEdge[partId].emplace_back(ii, jj);
										PartEdge[id] = 255, nValidEdge++;
									}
								}
							}
							if (nValidEdge > 5)
								vDensePose[lcid].validParts[partId] = 1;
						}
					}
				}
			}
			printLOG(" %.4fs ", omp_get_wtime() - startTime);
			printLOG("\n");
		}

		FitSMPL2Total(mySMPL, frame_params, frame_skeleton, AllvDensePose, CamTimeInfo, VideoInfo, vSCams, ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, Real2SfM, skeletonPointFormat, fixedPoseFrame, Pid, hasDensePose);

		for (size_t jj = 0; jj < frame_skeleton.size(); jj++)
		{
			sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, 0, vsyncFid[jj], TimeScale* vsyncFid[jj]);
			FILE *fp = fopen(Fname, "w+");
			fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[jj].t(0), frame_params[jj].t(1), frame_params[jj].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fprintf(fp, "%f %f %f\n", frame_params[jj].pose(ii, 0), frame_params[jj].pose(ii, 1), frame_params[jj].pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fprintf(fp, "%f ", mySMPL.coeffs(ii));
			fclose(fp);
		}
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;
	delete[]Para;

	return 0;
}
int FitSMPLUnSync(char *Path, SMPLModel &mySMPL, HumanSkeleton3D *vSkeleton, DensePose *AllvDensePose, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int startF, int stopF, int increF, Point2i &fixedPoseFrame, int Pid, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose)
{
	const double TimeScale = 1000000.0;

	char Fname[512];
	sprintf(Fname, "%s/FitBody/@%d", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/US_Smoothing1000", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/US_Smoothing1000/%d", Path, increF, Pid); makeDir(Fname);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;
	int maxWidth = 0, maxHeight = 0, nsCams = vSCams.size();
	for (int lcid = 0; lcid < nsCams; lcid++)
		maxWidth = max(maxWidth, VideoInfo[vSCams[lcid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vSCams[lcid]].VideoInfo[0].height);

	int nthreads = omp_get_max_threads();
	omp_set_num_threads(nthreads);

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 18)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .5, JointWeight[19] = .5, JointWeight[20] = .5,//rFoot
		JointWeight[21] = .5, JointWeight[22] = .5, JointWeight[23] = .5;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	int refCid = 0;
	double earliest = DBL_MAX;
	for (auto cid : vSCams)
		if (earliest > CamTimeInfo[cid].y)
			earliest = CamTimeInfo[cid].y, refCid = cid;

	SMPLParams ParaI;
	double smpl2sfmScale;
	vector<double> vscale;
	vector<SMPLParams> frame_params;
	vector<ImgPoseEle> frame_skeleton;

	Mat IUV, INDS;
	vector<int> vDP_Vid;
	vector<Point2d> vDP_uv;
	for (int reffid = startF; reffid <= stopF; reffid += increF)
	{
		for (int ii = 0; ii < vSCams.size(); ii++)
		{
			int cid = vSCams[ii], rcid, temp = (reffid - startF) / increF, found = 0;

			ImgPoseEle skeI(skeletonPointFormat);
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				skeI.pt2D[jid] = Point2d(0, 0), skeI.confidence[jid] = -1.0;
				for (int ii = 0; ii < vSkeleton[temp].vViewID_rFid[jid].size(); ii++)
				{
					int rcid = vSkeleton[temp].vViewID_rFid[jid][ii].x, rfid = vSkeleton[temp].vViewID_rFid[jid][ii].y;
					CameraData *Cam = VideoInfo[rcid].VideoInfo;
					if (rcid != cid || !Cam[rfid].valid)
						continue;

					found = 1;
					skeI.pt2D[jid] = vSkeleton[temp].vPt2D[jid][ii], skeI.confidence[jid] = vSkeleton[temp].vConf[jid][ii];
					skeI.viewID = rcid, skeI.frameID = rfid, skeI.refFrameID = reffid;
					skeI.ts = (CamTimeInfo[rcid].y / CamTimeInfo[refCid].x + 1.0*rfid / CamTimeInfo[rcid].x)*CamTimeInfo[refCid].x;

					AssembleP(Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, skeI.P + jid * 12);
				}
			}
			if (found == 0)
				continue;

			if (reffid >= fixedPoseFrame.x && reffid <= fixedPoseFrame.y)
				sprintf(Fname, "%s/FitBody/@%d/US_Smoothing1000/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, skeI.viewID, skeI.frameID, round(TimeScale*skeI.ts));
			else
				sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, 1, Pid, 0, reffid, TimeScale*reffid);
			if (!IsFileExist(Fname, false))
				continue;
			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%lf %lf %lf %lf ", &smpl2sfmScale, &ParaI.t(0), &ParaI.t(1), &ParaI.t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &ParaI.pose(ii, 0), &ParaI.pose(ii, 1), &ParaI.pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &ParaI.coeffs(ii));
			fclose(fp);

			ParaI.frame = reffid;
			ParaI.scale = smpl2sfmScale;
			if (smpl2sfmScale < 1e-16 || !IsNumber(smpl2sfmScale) || !IsFiniteNumber(smpl2sfmScale))
			{
				printLOG("Problem with %s\n", Fname);
				continue; //fail to init
			}

			vscale.push_back(smpl2sfmScale);
			frame_params.push_back(ParaI);
			frame_skeleton.push_back(skeI);
		}
	}

	if (vscale.size() > 0)
	{
		smpl2sfmScale = MedianArray(vscale); //robusifier

		 //init smpl
		mySMPL.t.setZero(), mySMPL.pose.setZero(), mySMPL.coeffs.setZero();
		mySMPL.scale = smpl2sfmScale;
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);

		if (fixedPoseFrame.x > -1)
		{
			for (int jj = 0; jj < frame_skeleton.size(); jj++)
			{
				if (frame_skeleton[jj].refFrameID >= fixedPoseFrame.x && frame_skeleton[jj].refFrameID <= fixedPoseFrame.y)
				{
					for (int ii = 0; ii < nShapeCoeffs; ii++)
						mySMPL.coeffs(ii) = frame_params[jj].coeffs(ii);
					mySMPL.scale = frame_params[jj].scale;
					break;
				}
			}
		}

		//Read DensePose data
		if (hasDensePose)
		{
			printLOG("Getting DensePose data: ");
			double startTime = omp_get_wtime();
			for (int inst = 0; inst < frame_skeleton.size(); inst++)
			{
				int reffid = frame_skeleton[inst].refFrameID, rfid = frame_skeleton[inst].frameID, rcid = frame_skeleton[inst].viewID;
				printLOG("%d (cid: %d)..", reffid, rcid);

				DensePose *vDensePose = AllvDensePose + inst;

				sprintf(Fname, "%s/DensePose/Corrected/%d/%.4d_IUV.png", Path, rcid, rfid); IUV = imread(Fname);
				sprintf(Fname, "%s/DensePose/Corrected/%d/%.4d_INDS.png", Path, rcid, rfid); INDS = imread(Fname, 0);
				if (IUV.empty() == 1)
				{
					vDensePose[0].valid = 0;
					continue;
				}
				else
					vDensePose[0].valid = 1, distortionCorrected = 1;
				int width = IUV.cols, height = IUV.rows, length = width * height;

				//Associate skeleton with DP
				vector<Point2d> vkpts;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
					vkpts.push_back(frame_skeleton[inst].pt2D[jid]);

				int DP_pid = 0;
				vector<int> vDP_pid;
				for (int ii = 0; ii < vkpts.size(); ii++)
					if (vkpts[ii].x > 0 && vkpts[ii].x < width - 1 && vkpts[ii].y>0 && vkpts[ii].y < height - 1)
						vDP_pid.push_back((int)INDS.data[(int)(vkpts[ii].x + 0.5) + (int)(vkpts[ii].y + 0.5)*width]);
				if (vDP_pid.size() < 2)
				{
					vDensePose[0].valid = 0;
					continue;
				}
				std::sort(vDP_pid.begin(), vDP_pid.end());
				DP_pid = vDP_pid[vDP_pid.size() / 2];

				//precompute distortion map if needed
				vDensePose[0].valid = 1, vDensePose[0].cid = rcid, vDensePose[0].fid = rfid, vDensePose[0].width = width, vDensePose[0].height = height;

				if (CostWeights[4] > 0.0) //compute body mask
				{
#pragma omp parallel for
					for (int jj = 0; jj < height; jj++)
					{
						for (int ii = 0; ii < width; ii++)
						{
							int id = ii + jj * width;
							BinarizeData[id] = ((int)INDS.data[id] == DP_pid && IUV.data[3 * id] > 0) ? 255 : 0, outSideEdge[id] = 0;
						}
					}
					//background mask term: penalizing occluded parts not satisfying the seg (floating parts)
					ComputeDistanceTransform(BinarizeData, float_df, 128, width, height, v, z, DTTps, ADTTps, realADT); //df of edge

#pragma omp parallel for
					for (int ii = 0; ii < length; ii++)
					{
						if (float_df[ii] > 65535)
							vDensePose[0].vdfield[14][ii] = 65535;
						else
							vDensePose[0].vdfield[14][ii] = uint16_t(float_df[ii]);
					}
				}

				//edge terms
				if (CostWeights[3] > 0.0)
				{
					for (int jj = 15; jj < height - 15; jj++)
					{
						for (int ii = 15; ii < width - 15; ii++)
						{
							int id = ii + jj * width;
							if (abs(BinarizeData[id] - BinarizeData[id + 1]) > 0 || abs(BinarizeData[id] - BinarizeData[id + width]) > 0)
							{
								outSideEdge[id] = 255;
								vDensePose[0].vEdge[14].emplace_back(ii, jj);
							}
						}
					}
					for (int partId = 0; partId < 14; partId++)
					{
						//compute part edge
#pragma omp parallel for
						for (int jj = 0; jj < height; jj++)
						{
							for (int ii = 0; ii < width; ii++)
							{
								int id = ii + jj * width;
								BinarizeData[id] = 0, PartEdge[id] = 0;

								bool found = false;
								for (int kk = 0; kk < vMergedPartId2DensePosePartId[partId].size() && !found; kk++)
									if ((int)INDS.data[id] == DP_pid && IUV.data[3 * id] == vMergedPartId2DensePosePartId[partId][kk])
										BinarizeData[id] = 255;
							}
						}

						int nValidEdge = 0;
						for (int jj = 0; jj < height - 1; jj++)
						{
							for (int ii = 0; ii < width - 1; ii++)
							{
								int id = ii + jj * width;
								if ((abs(BinarizeData[id] - BinarizeData[id + 1]) > 0 || abs(BinarizeData[id] - BinarizeData[id + width]) > 0) && outSideEdge[id] == 255)
								{
									vDensePose[0].vEdge[partId].emplace_back(ii, jj);
									PartEdge[id] = 255, nValidEdge++;
								}
							}
						}
						if (nValidEdge > 5)
							vDensePose[0].validParts[partId] = 1;
					}
				}
			}
			printLOG(" %.4fs ", omp_get_wtime() - startTime);
			printLOG("\n");
		}

		FitSMPL2TotalUnSync(mySMPL, frame_params, frame_skeleton, AllvDensePose, CamTimeInfo, VideoInfo, vSCams,
			ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, Real2SfM, skeletonPointFormat, fixedPoseFrame, Pid, hasDensePose);

		for (size_t jj = 0; jj < frame_skeleton.size(); jj++)
		{
			sprintf(Fname, "%s/FitBody/@%d/US_Smoothing1000/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, frame_skeleton[jj].viewID, frame_skeleton[jj].frameID, round(TimeScale* frame_skeleton[jj].ts));
			FILE *fp = fopen(Fname, "w+");
			fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[jj].t(0), frame_params[jj].t(1), frame_params[jj].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fprintf(fp, "%f %f %f\n", frame_params[jj].pose(ii, 0), frame_params[jj].pose(ii, 1), frame_params[jj].pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fprintf(fp, "%f ", mySMPL.coeffs(ii));
			fclose(fp);
		}
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;

	return 0;
}

int FitSMPL1FrameDriver(char *Path, vector<int> &vsCams, int startF, int stopF, int increF, int distortionCorrected, int sharedIntrinsic, int skeletonPointFormat, int Use2DFitting, double *weights, double *isigmas, double Real2SfM, double detectionThresh, bool hasDensePose, int selectedPeopleId = -1)
{
	printLOG("*****************FitSMPL1FrameDriver [%d->%d]*****************\n", startF, stopF);
	char Fname[512];
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int nSCams = vsCams.size(), nGCams = *max_element(vsCams.begin(), vsCams.end()) + 1;

	printLOG("Reading all camera poses\n");
	VideoData *VideoInfo = new VideoData[nGCams];
	for (int cid = 0; cid < nGCams; cid++)
	{
		printLOG("Cam %d ...validating ", cid);
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;
		InvalidateAbruptCameraPose(VideoInfo[cid], -1, -1, 0);
		printLOG("\n");
	}

	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nGCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int temp;
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
					CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nGCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	vector<Point2f*> vMapXY;
	DensePose *vDensePose = new DensePose[nSCams];
	if (hasDensePose)
	{
		for (int lcid = 0; lcid < nSCams; lcid++)
		{
			int width = VideoInfo[vsCams[lcid]].VideoInfo[startF].width, height = VideoInfo[vsCams[lcid]].VideoInfo[startF].height, length = width * height;
			for (int partId = 14; partId < 15; partId++)
				vDensePose[lcid].vdfield[partId] = new uint16_t[length];

			Point2f *MapXY = 0;
			if (distortionCorrected == 0 && sharedIntrinsic == 1)
			{
				MapXY = new Point2f[length];
				for (int jj = 0; jj < height; jj++)
					for (int ii = 0; ii < width; ii++)
						MapXY[ii + jj * width] = Point2d(ii, jj);

				if (VideoInfo[vsCams[lcid]].VideoInfo[startF].LensModel == RADIAL_TANGENTIAL_PRISM)
					LensDistortionPoint(MapXY, VideoInfo[vsCams[lcid]].VideoInfo[startF].K, VideoInfo[vsCams[lcid]].VideoInfo[startF].distortion, length);
				else
					FishEyeDistortionPoint(MapXY, VideoInfo[vsCams[lcid]].VideoInfo[startF].K, VideoInfo[vsCams[lcid]].VideoInfo[startF].distortion[0], length);
			}
			vMapXY.push_back(MapXY);
		}
	}

	Point3d RestJoints[18];
	sprintf(Fname, "smpl/restJointsCOCO.txt"); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	for (int ii = 0; ii < 18; ii++)
		fscanf(fp, "%lf %lf %lf ", &RestJoints[ii].x, &RestJoints[ii].y, &RestJoints[ii].z);
	fclose(fp);


	int nMaxPeople = 6;
	sprintf(Fname, "%s/FitBody", Path), makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d", Path, increF), makeDir(Fname);
	for (int frameID = startF; frameID <= stopF; frameID += increF)
	{
		printLOG("\nFrame %d\n", frameID);

		int rcid;  double u, v, s, avg_error;
		vector<HumanSkeleton3D> vSkeletons;
		if (selectedPeopleId == -1)
		{
			while (true)
			{
				int rfid, inlier, nValidJoints = 0;
				HumanSkeleton3D Body;
				sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, (int)vSkeletons.size(), frameID);
				if (IsFileExist(Fname) == 0)
					sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, (int)vSkeletons.size(), frameID);
				FILE *fp = fopen(Fname, "r");
				if (fp != NULL)
				{
					int  nvis;
					for (int jid = 0; jid < skeletonPointFormat; jid++)
					{
						fscanf(fp, "%lf %lf %lf %lf %d ", &Body.pt3d[jid].x, &Body.pt3d[jid].y, &Body.pt3d[jid].z, &avg_error, &nvis);
						for (int kk = 0; kk < nvis; kk++)
						{
							fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
							if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid)
								continue;

							Point2d uv(u, v);
							if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
								LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
							else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
								FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);

							Body.vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
							Body.vPt2D[jid].push_back(uv);
							Body.vConf[jid].push_back(s);
						}

						if (abs(Body.pt3d[jid].x) + abs(Body.pt3d[jid].y) + abs(Body.pt3d[jid].z) > 1e-16)
							nValidJoints++;
					}
					fclose(fp);

					Body.valid = nValidJoints < skeletonPointFormat / 3 ? 0 : 1;
					vSkeletons.push_back(Body);
				}
				else
					break;
				if (vSkeletons.size() > nMaxPeople)
					break;
			}
		}
		else
		{
			vSkeletons.resize(selectedPeopleId + 1);
			int rfid, inlier, nValidJoints = 0;
			sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, selectedPeopleId, frameID);
			if (IsFileExist(Fname) == 0)
				sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, selectedPeopleId, frameID);
			FILE *fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int  nvis;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
				{
					fscanf(fp, "%lf %lf %lf %lf %d ", &vSkeletons[selectedPeopleId].pt3d[jid].x, &vSkeletons[selectedPeopleId].pt3d[jid].y, &vSkeletons[selectedPeopleId].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
						if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid)
							continue;

						Point2d uv(u, v);
						if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
						else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);

						vSkeletons[selectedPeopleId].vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
						vSkeletons[selectedPeopleId].vPt2D[jid].push_back(uv);
						vSkeletons[selectedPeopleId].vConf[jid].push_back(s);
					}

					if (IsValid3D(vSkeletons[selectedPeopleId].pt3d[jid]))
						nValidJoints++;
				}
				fclose(fp);

				vSkeletons[selectedPeopleId].valid = nValidJoints < skeletonPointFormat / 3 ? 0 : 1;
			}
			else
				break;
		}

		InitializeBodyPoseParameters1Frame(Path, frameID, increF, vSkeletons, RestJoints, skeletonPointFormat);
		FitSMPL1Frame(Path, smplMaster, frameID, increF, vSkeletons, vDensePose, VideoInfo, CamTimeInfo, vsCams, distortionCorrected, sharedIntrinsic, skeletonPointFormat, Use2DFitting, weights, isigmas, Real2SfM, hasDensePose, nMaxPeople, selectedPeopleId);
	}

	delete[]vDensePose;

	return 0;
}
int FitSMPLWindowDriver(char *Path, vector<int> &vsCams, int startF, int stopF, int winSize, int nOverlappingFrames, int increF, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, int syncMode, double *CostWeights, double *isigmas, double Real2SfM, double detectionThresh, bool hasDensePose, int selectedPeopleID = -1, int startChunkdId = 0)
{
	char Fname[512];
	int nChunks = (stopF - startF + 1) / winSize + 1;
	const int  nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int nSCams = vsCams.size(), nGCams = *max_element(vsCams.begin(), vsCams.end()) + 1;

	printLOG("Reading all camera poses\n");
	VideoData *VideoInfo = new VideoData[nGCams];
	for (int lcid = 0; lcid < nSCams; lcid++)
	{
		printLOG("Cam %d ...validating ", lcid);
		if (ReadVideoDataI(Path, VideoInfo[vsCams[lcid]], vsCams[lcid], -1, -1) == 1)
			continue;
		InvalidateAbruptCameraPose(VideoInfo[vsCams[lcid]], -1, -1, 0);
		printLOG("\n");
	}

	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nGCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int temp;
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
					CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	double u, v, s, avg_error;
	int  rcid, nPeople = 0;
	vector<HumanSkeleton3D *> vSkeleton;
	if (selectedPeopleID == -1)
	{
		printLOG("Reading all people 3D skeleton: ");
		while (true)
		{
			printLOG("%d..", nPeople);

			int nvalidFrames = 0;
			HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
			for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				int  nValidJoints = 0, temp = (refFid - startF) / increF;
				sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, 1, nPeople, refFid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, nPeople, refFid);
					if (IsFileExist(Fname) == 0)
					{
						sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, nPeople, refFid);
						if (IsFileExist(Fname) == 0)
							continue;
					}
				}
				fp = fopen(Fname, "r");
				if (fp != NULL)
				{
					int  rfid, nvis, inlier;
					for (int jid = 0; jid < skeletonPointFormat; jid++)
					{
						fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
						for (int kk = 0; kk < nvis; kk++)
						{
							fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
							if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid)
								continue;

							Point2d uv(u, v);
							if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
								LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
							else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
								FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);

							Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
							Skeletons[temp].vPt2D[jid].push_back(uv);
							Skeletons[temp].vConf[jid].push_back(s);
						}

						if (abs(Skeletons[temp].pt3d[jid].x) + abs(Skeletons[temp].pt3d[jid].y) + abs(Skeletons[temp].pt3d[jid].z) > 1e-16)
							Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
						else
							Skeletons[temp].validJoints[jid] = 0;
					}
					fclose(fp);

					if (nValidJoints < skeletonPointFormat / 3)
						Skeletons[temp].valid = 0;
					else
						Skeletons[temp].valid = 1;
					Skeletons[temp].refFid = refFid;

					nvalidFrames++;
				}
			}
			if (nvalidFrames == 0)
			{
				printLOG("\n");
				break;
			}

			vSkeleton.push_back(Skeletons);
			nPeople++;
		}
	}
	else
	{
		printLOG("Reading 3D skeleton for %d:\n", selectedPeopleID);
		for (int pid = 0; pid < selectedPeopleID; pid++)
		{
			HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
			vSkeleton.push_back(Skeletons);
			nPeople++;
		}

		int nvalidFrames = 0;
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int  nValidJoints = 0, temp = (refFid - startF) / increF;
			sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, 1, nPeople, refFid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, nPeople, refFid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, nPeople, refFid);
					if (IsFileExist(Fname) == 0)
						continue;
				}
			}
			fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int  rfid, nvis, inlier;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
				{
					fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
						if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid)
							continue;

						Point2d uv(u, v);
						if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
						else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);

						Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
						Skeletons[temp].vPt2D[jid].push_back(uv);
						Skeletons[temp].vConf[jid].push_back(s);
					}

					if (IsValid3D(Skeletons[temp].pt3d[jid]))
						Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
					else
						Skeletons[temp].validJoints[jid] = 0;
				}
				fclose(fp);

				if (nValidJoints < skeletonPointFormat / 3)
					Skeletons[temp].valid = 0;
				else
					Skeletons[temp].valid = 1;
				Skeletons[temp].refFid = refFid;

				nvalidFrames++;
			}
		}
		if (nvalidFrames == 0)
			printLOG("\n");

		vSkeleton.push_back(Skeletons);
		nPeople++;
	}

	int nSyncInst = ((winSize + nOverlappingFrames) / increF + 1);
	DensePose *vDensePose = new DensePose[nSCams*nSyncInst]; //should have enough mem for unsynced case
	if (hasDensePose)
	{
		for (int lcid = 0; lcid < nSCams; lcid++)
		{
			rcid = vsCams[lcid];
			int width = VideoInfo[rcid].VideoInfo[startF].width, height = VideoInfo[rcid].VideoInfo[startF].height, length = width * height;
			for (int inst = 0; inst < nSyncInst; inst++)
				for (int partId = 14; partId < 15; partId++)
					vDensePose[lcid*nSyncInst + inst].vdfield[partId] = new uint16_t[length]; //for the silhoutte term
		}
	}

	//CostWeights[2] = CostWeights[2] / increF/increF;
	for (int pid = 0; pid < nPeople; pid++)
	{
		if (selectedPeopleID != -1 && pid != selectedPeopleID)
			continue;

		Point2i fixedPoseFrames(-1, -1);
		for (int chunkId = startChunkdId; chunkId < nChunks; chunkId++)
		{
			if (chunkId != startChunkdId)
				hasDensePose = false;
			if (syncMode == 1)
			{
				printLOG("*****************FitSMPLWindowDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
				FitSMPLWindow(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
			}
			else
			{
				printLOG("*****************FitSMPLWindowUnSyncDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
				FitSMPLUnSync(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
			}
			fixedPoseFrames.x = startF + winSize * (chunkId + 1), fixedPoseFrames.y = startF + winSize * (chunkId + 1) + nOverlappingFrames;
		}

		if (startChunkdId > 0)
		{
			for (int chunkId = startChunkdId - 1; chunkId > -1; chunkId--)
			{
				fixedPoseFrames.x = startF + winSize * (chunkId + 1), fixedPoseFrames.y = startF + winSize * (chunkId + 1) + nOverlappingFrames;
				if (syncMode == 1)
				{
					printLOG("*****************FitSMPLWindowDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
					FitSMPLWindow(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
				}
				else
				{
					printLOG("*****************FitSMPLWindowUnSyncDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
					FitSMPLUnSync(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
				}
			}
		}
	}

	delete[]vDensePose, delete[]VideoInfo;
	for (int ii = 0; ii < vSkeleton.size(); ii++)
		delete[]vSkeleton[ii];

	return 0;
}

int VisualizeProjectedSMPLBody_MVS(char *Path, char *SeqName, SMPLModel &smplMaster, vector<HumanSkeleton3D> &vSkeletons, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int skeletonPointFormat, int distortionCorrected, double resizeFactor)
{
	char Fname[512];
	sprintf(Fname, "%s/extracted_frames/%s/Vis_FitBody", Path, SeqName), makeDir(Fname);

	static cv::Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 128, 255), Scalar(0, 255, 255), Scalar(0, 255, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(255, 255, 255) };
	vector<Point3i> vcolors;
	vcolors.push_back(Point3i(0, 0, 255)), vcolors.push_back(Point3i(0, 128, 255)), vcolors.push_back(Point3i(0, 255, 255)), vcolors.push_back(Point3i(0, 255, 0)),
		vcolors.push_back(Point3i(255, 128, 0)), vcolors.push_back(Point3i(255, 255, 0)), vcolors.push_back(Point3i(255, 0, 0)), vcolors.push_back(Point3i(255, 0, 255)), vcolors.push_back(Point3i(255, 255, 255));
	int selected;  double fps;

	omp_set_num_threads(omp_get_max_threads());

	int cid, dummy;

	const int nVertices = smpl::SMPLModel::nVertices, nShapeCoeffs = smpl::SMPLModel::nShapeCoeffs, nJointsSMPL = smpl::SMPLModel::nJoints;
	MatrixXdr outV(nVertices, 3);
	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	int nPeople = (int)vSkeletons.size();
	MatrixXdr *AllV = new MatrixXdr[vSkeletons.size()];
	for (int pi = 0; pi < nPeople; pi++)
		AllV[pi].resize(nVertices, 3);

	Point3i f; vector<Point3i> faces;
	FILE *fp = fopen("smpl/faces.txt", "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %d %d ", &f.x, &f.y, &f.z) != EOF)
			faces.push_back(f);
		fclose(fp);
	}

	Mat img, rimg, mask, blend;
	smpl::SMPLParams params;
	vector<int> vpid(nPeople);
	vector<smpl::SMPLParams>Vparams(nPeople);
	Point2d joints2D[25];
	vector<Point2f> *Vuv = new vector<Point2f>[nPeople];
	Point3f *allVertices = new Point3f[nVertices*nPeople];

	for (int ii = 0; ii < nPeople; ii++)
		vpid[ii] = false;
	for (int pid = 0; pid < nPeople; pid++)
	{
		sprintf(Fname, "%s/extracted_frames/%s/FitBody/f_%d.txt", Path, SeqName, pid);
		if (IsFileExist(Fname) == 0)
			continue;
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%lf %lf %lf %lf ", &Vparams[pid].scale, &Vparams[pid].t(0), &Vparams[pid].t(1), &Vparams[pid].t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &Vparams[pid].pose(ii, 0), &Vparams[pid].pose(ii, 1), &Vparams[pid].pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fscanf(fp, "%lf ", &Vparams[pid].coeffs(ii));
		fclose(fp);

		vpid[pid] = true;
	}

#pragma omp parallel for schedule(dynamic,1)
	for (int pid = 0; pid < nPeople; pid++)
	{
		if (!vpid[pid])
			continue;

		reconstruct(smplMaster, Vparams[pid].coeffs.data(), Vparams[pid].pose.data(), AllV[pid].data());
		Map<VectorXd> V_vec(AllV[pid].data(), AllV[pid].size());
		V_vec = V_vec * Vparams[pid].scale + dVdt * Vparams[pid].t;

		for (int ii = 0; ii < nVertices; ii++)
			allVertices[ii + pid * nVertices] = Point3f(AllV[pid](ii, 0), AllV[pid](ii, 1), AllV[pid](ii, 2));
	}

	int debug = 0;
	int nCams = vSCams.size();
	for (int cid = 0; cid < nCams; cid++)
	{
		int width = VideoInfo[cid].VideoInfo[0].width, height = VideoInfo[cid].VideoInfo[0].height;

		CameraData *camI = VideoInfo[cid].VideoInfo;
		if (camI[0].valid != 1)
			continue;

		sprintf(Fname, "%s/extracted_frames/%s/%d.png", Path, SeqName, VideoInfo[cid].VideoInfo[0].frameID); img = imread(Fname);
		if (img.empty() == 1)
			continue;

		for (int pid = 0; pid < nPeople; pid++)
		{
			//	if (pid != 0 && pid != 2)
				//	continue;
			HumanSkeleton3D *Body0 = &vSkeletons[pid];
			bool visible = 0;
			int bottomID = -1; double bottomY = 0;
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				joints2D[jid] = Point2d(0, 0);
				if (Body0[0].validJoints[jid] > 0)
				{
					Point3d xyz = Body0[0].pt3d[jid];
					if (camI[0].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						ProjectandDistort(xyz, &joints2D[jid], camI[0].P);
						if (joints2D[jid].x < -camI[0].width / 10 || joints2D[jid].x > 11 * camI[0].width / 10 || joints2D[jid].y < -camI[0].height / 10 || joints2D[jid].y > 11 * camI[0].height / 10)
							continue;

						if (distortionCorrected == 0)
						{
							if (camI[0].ShutterModel == GLOBAL_SHUTTER)
								ProjectandDistort(xyz, &joints2D[jid], camI[0].P, camI[0].K, camI[0].distortion);
							else if (camI[0].ShutterModel == ROLLING_SHUTTER)
								CayleyDistortionProjection(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, joints2D[0], xyz, width, height);
						}
					}
					else
					{
						FisheyeProjectandDistort(xyz, &joints2D[jid], camI[0].P, camI[0].K, camI[0].distortion);
						if (joints2D[jid].x < -camI[0].width / 10 || joints2D[jid].x > 11 * camI[0].width / 10 || joints2D[jid].y < -camI[0].height / 10 || joints2D[jid].y > 11 * camI[0].height / 10)
							continue;

						if (distortionCorrected == 0)
						{
							if (camI[0].ShutterModel == GLOBAL_SHUTTER)
								FisheyeProjectandDistort(xyz, &joints2D[jid], camI[0].P, camI[0].K, camI[0].distortion);
							else if (camI[0].ShutterModel == ROLLING_SHUTTER)
								CayleyFOVProjection2(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, joints2D[jid], xyz, width, height);
						}
					}

					if (joints2D[jid].y > bottomY)
						bottomY = joints2D[jid].y, bottomID = jid;
				}
			}
			//if (bottomID > 13 || bottomID < 8)
			//	continue;

			if (debug == 1)
			{
				Draw2DCoCoJoints(img, joints2D, skeletonPointFormat, 2, 1.0, &colors[pid % 8]);
				sprintf(Fname, "%s/extracted_frames/%s/Vis_FitBody/x.jpg", Path, SeqName), imwrite(Fname, img);
			}

			/*for (int ii = 0; ii < nVertices; ii++)
			{
				Point2d uv;
				Point3d xyz(allVertices[nVertices*pid + ii].x, allVertices[nVertices*pid + ii].y, allVertices[nVertices*pid + ii].z);

				if (distortionCorrected == 0)
				{
					if (VideoInfo[cid].VideoInfo[0].LensModel == RADIAL_TANGENTIAL_PRISM)
						CayleyDistortionProjection(VideoInfo[cid].VideoInfo[0].intrinsic, VideoInfo[cid].VideoInfo[0].distortion, VideoInfo[cid].VideoInfo[0].rt, VideoInfo[cid].VideoInfo[0].wt, uv, xyz, width, height);
					else
						CayleyFOVProjection2(VideoInfo[cid].VideoInfo[0].intrinsic, VideoInfo[cid].VideoInfo[0].distortion, VideoInfo[cid].VideoInfo[0].rt, VideoInfo[cid].VideoInfo[0].wt, uv, xyz, width, height);
				}

				int x = (int)(uv.x + 0.5), y = (int)(uv.y + 0.5);
				circle(img, Point2i(x, y), 1, colors[vpid[pid] % 9], 1);
			}*/

			Mat mask = Mat::zeros(img.rows, img.cols, CV_8U);
#pragma omp parallel for schedule(dynamic,1)
			for (int ii = 0; ii < faces.size(); ii++)
			{
				Point2d uv[3];
				int vid[3] = { faces[ii].x,  faces[ii].y,  faces[ii].z };
				for (int jj = 0; jj < 3; jj++)
				{
					Point3d xyz(allVertices[nVertices*pid + vid[jj]].x, allVertices[nVertices*pid + vid[jj]].y, allVertices[nVertices*pid + vid[jj]].z);
					if (camI[0].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						ProjectandDistort(xyz, &uv[jj], camI[0].P);
						if (uv[jj].x < -camI[0].width / 10 || uv[jj].x > 11 * camI[0].width / 10 || uv[jj].y < -camI[0].height / 10 || uv[jj].y > 11 * camI[0].height / 10)
							continue;

						if (distortionCorrected == 0)
						{
							if (camI[0].ShutterModel == GLOBAL_SHUTTER)
								ProjectandDistort(xyz, &uv[jj], camI[0].P, camI[0].K, camI[0].distortion);
							else if (camI[0].ShutterModel == ROLLING_SHUTTER)
								CayleyDistortionProjection(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, uv[jj], xyz, width, height);
						}
					}
					else
					{
						FisheyeProjectandDistort(xyz, &uv[jj], camI[0].P, camI[0].K, camI[0].distortion);
						if (uv[jj].x < -camI[0].width / 10 || uv[jj].x > 11 * camI[0].width / 10 || uv[jj].y < -camI[0].height / 10 || uv[jj].y > 11 * camI[0].height / 10)
							continue;

						if (distortionCorrected == 0)
						{
							if (camI[0].ShutterModel == GLOBAL_SHUTTER)
								FisheyeProjectandDistort(xyz, &uv[jj], camI[0].P, camI[0].K, camI[0].distortion);
							else if (camI[0].ShutterModel == ROLLING_SHUTTER)
								CayleyFOVProjection2(camI[0].intrinsic, camI[0].distortion, camI[0].rt, camI[0].wt, uv[jj], xyz, width, height);
						}
					}
				}
				/*int maxX = min((int)(max(max(max(0, uv[0].x), uv[1].x), uv[2].x)) + 1, width - 1);
				int minX = max((int)(min(min(min(width - 1, uv[0].x), uv[1].x), uv[2].x)) + 1, 0);
				int maxY = min((int)(max(max(max(0, uv[0].y), uv[1].y), uv[2].y)) + 1, height - 1);
				int minY = max((int)(min(min(min(height - 1, uv[0].y), uv[1].y), uv[2].y)) + 1, 0);
				for (int jj = minY; jj <= maxY; jj++)
					for (int ii = minX; ii < maxX; ii++)
						if (PointInTriangle(uv[0], uv[1], uv[2], Point2f(ii, jj)))
							mask.data[ii + jj*width] = 255;*/

				if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[1].x >10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10)
					cv::line(img, uv[0], uv[1], colors[pid], 1, CV_AA);
				if (uv[1].x > 10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
					cv::line(img, uv[0], uv[2], colors[pid], 1, CV_AA);
				if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
					cv::line(img, uv[1], uv[2], colors[pid], 1, CV_AA);
			}
			//sprintf(Fname, "%s/Vis/FitBody/%d/%d/%.4d.png", Path, pid, cid, rfid), imwrite(Fname, mask);

			if (debug == 1)
				sprintf(Fname, "%s/extracted_frames/%s/Vis_FitBody/x.jpg", Path, SeqName), imwrite(Fname, img);
		}

		CvPoint text_origin = { width / 30, height / 30 };
		resize(img, rimg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_AREA);
		sprintf(Fname, "%s/extracted_frames/%s/Vis_FitBody/%.2d.jpg", Path, SeqName, cid), imwrite(Fname, rimg);
	}

	return 0;
}
int ReadMannequinCamPose(char *Path, char *FileName, vector<VideoData> &vVideoI)
{
	char Fname[512];
	sprintf(Fname, "%s/ManneqinChallenge_txt_files/%s.txt", Path, FileName);

	if (IsFileExist(Fname))
	{
		int cnt = 0, w, h;
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%s ", Fname);
		while (true)
		{
			VideoData VideoI;
			VideoI.VideoInfo = new CameraData[1];
			VideoI.startTime = 0, VideoI.stopTime = 0, VideoI.VideoInfo[0].valid = 0, VideoI.VideoInfo[0].LensModel = RADIAL_TANGENTIAL_PRISM, VideoI.VideoInfo[0].ShutterModel = GLOBAL_SHUTTER;

			if (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &VideoI.VideoInfo[0].frameID, &VideoI.VideoInfo[0].intrinsic[0], &VideoI.VideoInfo[0].intrinsic[1], &VideoI.VideoInfo[0].intrinsic[3], &VideoI.VideoInfo[0].intrinsic[4], &VideoI.VideoInfo[0].distortion[0], &VideoI.VideoInfo[0].distortion[1]) == EOF)
				break;
			for (int ii = 2; ii < 7; ii++)
				VideoI.VideoInfo[0].distortion[ii] = 0;

			for (int jj = 0; jj < 3; jj++)
				fscanf(fp, "%lf %lf %lf %lf", &VideoI.VideoInfo[0].R[jj * 3], &VideoI.VideoInfo[0].R[jj * 3 + 1], &VideoI.VideoInfo[0].R[jj * 3 + 2], &VideoI.VideoInfo[0].T[jj]);

			GetrtFromRT(VideoI.VideoInfo[0].rt, VideoI.VideoInfo[0].R, VideoI.VideoInfo[0].T);

			if (cnt == 0)
			{
				sprintf(Fname, "%s/extracted_frames/%s/%d.png", Path, FileName, VideoI.VideoInfo[0].frameID);
				if (!IsFileExist(Fname))
				{
					printf("Cannot load %s\n", Fname);
					return 1;
				}
				cv::Mat cv_img = cv::imread(Fname);
				w = cv_img.cols, h = cv_img.rows;
			}
			VideoI.VideoInfo[0].width = w, VideoI.VideoInfo[0].height = h;
			VideoI.VideoInfo[0].intrinsic[0] *= w, VideoI.VideoInfo[0].intrinsic[1] *= h, VideoI.VideoInfo[0].intrinsic[3] *= w, VideoI.VideoInfo[0].intrinsic[4] *= h;

			VideoI.VideoInfo[0].viewID = cnt;
			VideoI.VideoInfo[0].valid = true;
			cnt++;

			GetKFromIntrinsic(VideoI.VideoInfo[0]);
			mat_invert(VideoI.VideoInfo[0].K, VideoI.VideoInfo[0].invK);
			AssembleP(VideoI.VideoInfo[0].K, VideoI.VideoInfo[0].R, VideoI.VideoInfo[0].T, VideoI.VideoInfo[0].P);

			vVideoI.emplace_back(VideoI);
		}
		fclose(fp);
	}
	else
		return 1;

	return 0;
}
int InitializeBodyPoseParameters_MVS(char *Path, char *SeqName, vector<HumanSkeleton3D> &vSkeletons, int nPointFormat)
{
	char Fname[512];
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	Point3d RestJoints[18];
	sprintf(Fname, "smpl/restJointsCOCO.txt"); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	for (int ii = 0; ii < 18; ii++)
		fscanf(fp, "%lf %lf %lf ", &RestJoints[ii].x, &RestJoints[ii].y, &RestJoints[ii].z);
	fclose(fp);

	int TemplateTorsoJointID[5] = { 2,1,5,8,11 }, TargetTorsoJointID[5];
	if (nPointFormat == 18)
		TargetTorsoJointID[0] = 2, TargetTorsoJointID[1] = 1, TargetTorsoJointID[2] = 5, TargetTorsoJointID[3] = 8, TargetTorsoJointID[4] = 11;
	else if (nPointFormat == 25)
		TargetTorsoJointID[0] = 3, TargetTorsoJointID[1] = 2, TargetTorsoJointID[2] = 6, TargetTorsoJointID[3] = 9, TargetTorsoJointID[4] = 12;

	vector<Point3d> RestTorso, CurTorso;

	int numPeople = (int)vSkeletons.size();
	vector<int> invalidInit;
	vector<double> vs;
	vector<Point3d> vr, vT;
	for (int pi = 0; pi < numPeople; pi++)
	{
		if (vSkeletons[pi].valid == 0)
			continue;

		//check the scale of the 3d points first
		vector<double> medianX, medianY, medianZ;
		for (int ii = 0; ii < nPointFormat; ii++)
		{
			if (abs(vSkeletons[pi].pt3d[ii].x) + abs(vSkeletons[pi].pt3d[ii].y) + abs(vSkeletons[pi].pt3d[ii].z) > 1e-16)
				medianX.push_back(vSkeletons[pi].pt3d[ii].x), medianY.push_back(vSkeletons[pi].pt3d[ii].y), medianZ.push_back(vSkeletons[pi].pt3d[ii].z);
		}
		double mX = MedianArray(medianX), mY = MedianArray(medianY), mZ = MedianArray(medianZ);

		medianX.clear(), medianY.clear(), medianZ.clear();
		for (int ii = 0; ii < nPointFormat; ii++)
		{
			if (abs(vSkeletons[pi].pt3d[ii].x) + abs(vSkeletons[pi].pt3d[ii].y) + abs(vSkeletons[pi].pt3d[ii].z) > 1e-16)
				medianX.push_back(abs(vSkeletons[pi].pt3d[ii].x - mX)), medianY.push_back(abs(vSkeletons[pi].pt3d[ii].y - mY)), medianZ.push_back(abs(vSkeletons[pi].pt3d[ii].z - mZ));
		}
		double mstdX = MedianArray(medianX), mstdY = MedianArray(medianY), mstdZ = MedianArray(medianZ);

		//make sure that at least centroid and orientation are aligned.
		double r[3], R[9], T[3], s;
		//get Torso joints
		RestTorso.clear(), CurTorso.clear();
		for (int ii = 0; ii < 5; ii++)
		{
			if (abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].x) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].y) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].z) > 1e-16)
			{
				if (abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].x - mX) > 3.0*mstdX || abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].y - mY) > 3.0*mstdY || abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].z - mZ) > 3.0*mstdZ)
					continue;
				RestTorso.push_back(RestJoints[TemplateTorsoJointID[ii]]), CurTorso.push_back(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]]);
			}
		}

		if (RestTorso.size() < 3)
			invalidInit.push_back(pi);

		double error3D = computeProcrustesTransform(RestTorso, CurTorso, R, T, s, true);
		getrFromR(R, r);

		vs.push_back(s), vr.push_back(Point3d(r[0], r[1], r[2])), vT.push_back(Point3d(T[0], T[1], T[2]));

		vSkeletons[pi].s = s;
		for (int ii = 0; ii < 3; ii++)
			vSkeletons[pi].r[ii] = r[ii], vSkeletons[pi].t[ii] = T[ii];

		/*#ifdef _DEBUG
		//trying to recon the mesh with estimated rigid trans
		MatrixXdr outV(nVertices, 3); outV.setZero();
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		SparseMatrix<double, ColMajor> dVdt = kroneckerProduct(VectorXd::Ones(nVertices), eye3);

		mySMPL.t.setZero(), mySMPL.pose.setZero(), mySMPL.coeffs.setZero();
		for (int ii = 0; ii < 3; ii++)
		mySMPL.t(ii) = T[ii];
		mySMPL.pose(0, 0) = r[0], mySMPL.pose(0, 1) = r[1], mySMPL.pose(0, 2) = r[2];
		for (int ii = 1; ii < nJointsSMPL; ii++)
		mySMPL.pose(ii, 0) = 0, mySMPL.pose(ii, 1) = 0, mySMPL.pose(ii, 2) = 0;

		reconstruct(mySMPL, mySMPL.coeffs.data(), mySMPL.pose.data(), outV.data());
		Map< VectorXd > V_vec(outV.data(), outV.size());
		V_vec = V_vec*s + dVdt*mySMPL.t;

		// Regress from Vertices to joint
		//VectorXd J = mySMPL.J_regl_14_bigl_col_*V_vec;
		//sprintf(Fname, "%s/JBC/%.4d/pPoseLandmark_%d.txt", Path, frameID, pi);fp = fopen(Fname, "w");
		//for (int ic = 0; ic < 14; ic++)
		//	fprintf(fp, "%f %f %f\n", J(ic * 3), J(ic * 3 + 1), J(ic * 3 + 2));
		//fclose(fp);

		sprintf(Fname, "%s/FitBody/%d", Path, pi); makeDir(Fname);
		sprintf(Fname, "%s/%d/FitBody/%.4d.txt", Path, pi, frameID);  fp = fopen(Fname, "w+");
		for (int ii = 0; ii < nVertices; ii++)
		fprintf(fp, "%e %e %e\n", outV(ii, 0), outV(ii, 1), outV(ii, 2));
		fclose(fp);
		#endif*/
	}

	if (invalidInit.size() == numPeople)
		return 1;

	double sMean = MeanArray(vs);
	for (int jj = 0; jj < invalidInit.size(); jj++)
	{
		int pi = invalidInit[jj];
		RestTorso.clear(), CurTorso.clear();
		for (int ii = 0; ii < 5; ii++)
			if (abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].x) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].y) + abs(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]].z) > 1e-16)
				RestTorso.push_back(RestJoints[TemplateTorsoJointID[ii]]), CurTorso.push_back(vSkeletons[pi].pt3d[TargetTorsoJointID[ii]]);

		if (RestTorso.size() == 0)
			vSkeletons[pi].s = 0.0;
		else
		{
			vSkeletons[pi].s = sMean;
			for (int ii = 0; ii < 3; ii++)
				vSkeletons[pi].r[ii] = 0.0;

			double tX = 0, tY = 0, tZ = 0;
			for (int ii = 0; ii < RestTorso.size(); ii++)
			{
				double orgX = RestTorso[ii].x, orgY = RestTorso[ii].y, orgZ = RestTorso[ii].z, fX = CurTorso[ii].x, fY = CurTorso[ii].y, fZ = CurTorso[ii].z;
				tX += sMean * orgX - fX;
				tY += sMean * orgY - fY;
				tZ += sMean * orgZ - fZ;
			}
			vSkeletons[pi].t[0] = tX / RestTorso.size(), vSkeletons[pi].t[1] = tY / RestTorso.size(), vSkeletons[pi].t[2] = tZ / RestTorso.size();
		}
	}

	sprintf(Fname, "%s/extracted_frames/%s/FitBody", Path, SeqName); makeDir(Fname);
	for (int pi = 0; pi < (int)vSkeletons.size(); pi++)
	{
		if (!vSkeletons[pi].valid)
			continue;
		sprintf(Fname, "%s/extracted_frames/%s/FitBody/i_%d.txt", Path, SeqName, pi); FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%e %e %e %e\n%f %f %f ", vSkeletons[pi].s, vSkeletons[pi].t[0], vSkeletons[pi].t[1], vSkeletons[pi].t[2], vSkeletons[pi].r[0], vSkeletons[pi].r[1], vSkeletons[pi].r[2]);
		for (int ii = 1; ii < nJointsSMPL; ii++)
			fprintf(fp, "0.0 0.0 0.0\n");
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fprintf(fp, "0.0 ");
		fclose(fp);
	}

	return 0;
}
int FitSMPL_MVS(char *Path, char *SeqName, SMPLModel &mySMPL, int refFid, vector<HumanSkeleton3D> &vSkeletons, DensePose *vDensePose, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, int Use2DFitting, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose, int nMaxPeople, int selectedPeopleId)
{
	char Fname[512];
	sprintf(Fname, "%s/extracted_frames/%s/FitBody", Path, SeqName), makeDir(Fname);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int maxWidth = 0, maxHeight = 0, nsCams = vSCams.size();
	for (int lcid = 0; lcid < nsCams; lcid++)
		maxWidth = max(maxWidth, VideoInfo[vSCams[lcid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vSCams[lcid]].VideoInfo[0].height);

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 18)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .1, JointWeight[19] = .1, JointWeight[20] = .1,//rFoot
		JointWeight[21] = .1, JointWeight[22] = .1, JointWeight[23] = .1;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	vector<int> vDP_Vid, vDP_pid;
	vector<Point2d> vDP_uv, vkpts;
	float *Para = new float[maxWidth*maxHeight];
	Mat IUV, INDS;
	for (int pi = 0; pi < vSkeletons.size(); pi++)
	{
		if (!vSkeletons[pi].valid)
			continue;

		printLOG("*****Person %d*******\n", pi);

		double smpl2sfmScale;
		SMPLParams ParaI;

		sprintf(Fname, "%s/extracted_frames/%s/FitBody/i_%d.txt", Path, SeqName, pi);
		if (!IsFileExist(Fname))
			continue;
		else
		{
			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%lf %lf %lf %lf ", &smpl2sfmScale, &ParaI.t(0), &ParaI.t(1), &ParaI.t(2));
			ParaI.scale = smpl2sfmScale;
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &ParaI.pose(ii, 0), &ParaI.pose(ii, 1), &ParaI.pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &ParaI.coeffs(ii));
			fclose(fp);
		}

		ParaI.frame = refFid;
		if (smpl2sfmScale < 1e-16 || !IsNumber(smpl2sfmScale))
			continue; //fail to init

		//since the code is written for multi-frame BA
		vector<SMPLParams> frame_params; frame_params.push_back(ParaI);
		vector<HumanSkeleton3D> frame_skeleton; frame_skeleton.push_back(vSkeletons[pi]);

		//init smpl
		mySMPL.t.setZero(), mySMPL.pose.setZero(), mySMPL.coeffs.setZero();
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);
		mySMPL.scale = smpl2sfmScale;

		//Read DensePose data
		if (hasDensePose)
		{
			printLOG("Getting DensePose data: ");
			double startTime = omp_get_wtime();
			for (int lcid = 0; lcid < nsCams; lcid++)
			{
				printLOG("%d..", vSCams[lcid]);

				int rcid = -1, rfid = -1;
				for (int jid = 0; jid < skeletonPointFormat && rfid == -1; jid++)
				{
					for (int ii = 0; ii < vSkeletons[pi].vViewID_rFid[jid].size() && rfid == -1; ii++)
					{
						if (vSkeletons[pi].vViewID_rFid[jid][ii].x == vSCams[lcid])
							rcid = vSkeletons[pi].vViewID_rFid[jid][ii].x, rfid = vSkeletons[pi].vViewID_rFid[jid][ii].y;
					}
				}
				if (rfid == -1)
					continue;

				distortionCorrected = 0;
				sprintf(Fname, "%s/DensePose/%s/%d_IUV.png", Path, SeqName, VideoInfo[rcid].VideoInfo[0].frameID); IUV = imread(Fname);
				sprintf(Fname, "%s/DensePose/%s/%d_INDS.png", Path, SeqName, VideoInfo[rcid].VideoInfo[0].frameID); INDS = imread(Fname, 0);
				if (IUV.empty() == 1 || INDS.empty() == 1)
				{
					vDensePose[lcid].valid = 0;
					continue;
				}
				else
					vDensePose[lcid].valid = 1, distortionCorrected = 1;
				int width = IUV.cols, height = IUV.rows, length = width * height;

				//Associate with DP
				vkpts.clear();
				for (int jid = 0; jid < skeletonPointFormat; jid++)
					for (int ii = 0; ii < vSkeletons[pi].vViewID_rFid[jid].size(); ii++)
						if (vSkeletons[pi].vViewID_rFid[jid][ii].x == vSCams[lcid])
							vkpts.push_back(vSkeletons[pi].vPt2D[jid][ii]);

				int DP_pid = 0;
				vDP_pid.clear();
				for (int ii = 0; ii < vkpts.size(); ii++)
					if (vkpts[ii].x > 0 && vkpts[ii].x < width - 1 && vkpts[ii].y>0 && vkpts[ii].y < height - 1)
						vDP_pid.push_back((int)INDS.data[(int)(vkpts[ii].x + 0.5) + (int)(vkpts[ii].y + 0.5)*width]);
				if (vDP_pid.size() < 2)
				{
					vDensePose[lcid].valid = 0;
					continue;
				}
				std::sort(vDP_pid.begin(), vDP_pid.end());
				DP_pid = vDP_pid[vDP_pid.size() / 2];

				//precompute undistortion map if needed
				vDensePose[lcid].cid = rcid, vDensePose[lcid].fid = rfid, vDensePose[lcid].width = width, vDensePose[lcid].height = height;

				//compute body  mask and edge
				omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
				for (int jj = 0; jj < height; jj++)
				{
					for (int ii = 0; ii < width; ii++)
					{
						int id = ii + jj * width;
						BinarizeData[id] = ((int)INDS.data[id] == DP_pid && IUV.data[3 * id] > 0) ? 255 : 0, outSideEdge[id] = 0;
					}
				}

				//background mask term: penalizing occluded parts not satisfying the seg (floating parts)
				ComputeDistanceTransform(BinarizeData, float_df, 128, width, height, v, z, DTTps, ADTTps, realADT); //df of edge

#pragma omp parallel for
				for (int ii = 0; ii < length; ii++)
				{
					if (float_df[ii] > 65535)
						vDensePose[lcid].vdfield[14][ii] = 65535;
					else
						vDensePose[lcid].vdfield[14][ii] = uint16_t(float_df[ii]);
				}

				//edge term
//#pragma omp parallel for
				for (int jj = 15; jj < height - 15; jj++)
				{
					for (int ii = 15; ii < width - 15; ii++)
					{
						int id = ii + jj * width;
						if (abs(BinarizeData[id] - BinarizeData[id + 1]) > 0 || abs(BinarizeData[id] - BinarizeData[id + width]) > 0)
						{
							outSideEdge[id] = 255;
							vDensePose[lcid].vEdge[14].emplace_back(ii, jj);
						}
					}
				}
				for (int partId = 0; partId < 14; partId++)
				{
					//compute part edge
#pragma omp parallel for
					for (int jj = 0; jj < height; jj++)
					{
						for (int ii = 0; ii < width; ii++)
						{
							int id = ii + jj * width;
							BinarizeData[id] = 0, PartEdge[id] = 0;

							bool found = false;
							for (int kk = 0; kk < vMergedPartId2DensePosePartId[partId].size() && !found; kk++)
								if ((int)INDS.data[id] == DP_pid && IUV.data[3 * id] == vMergedPartId2DensePosePartId[partId][kk])
									BinarizeData[id] = 255;
						}
					}

					int nValidEdge = 0;
					for (int jj = 0; jj < height - 1; jj++)
					{
						for (int ii = 0; ii < width - 1; ii++)
						{
							int id = ii + jj * width;
							if ((abs(BinarizeData[id] - BinarizeData[id + 1]) > 0 || abs(BinarizeData[id] - BinarizeData[id + width]) > 0) && outSideEdge[id] == 255)
							{
								vDensePose[lcid].vEdge[partId].emplace_back(ii, jj);
								PartEdge[id] = 255, nValidEdge++;
							}
						}
					}
					if (nValidEdge > 5)
					{
						vDensePose[lcid].validParts[partId] = 1;
						/*//has been replaced by brute-froce nn
						ComputeDistanceTransform(PartEdge, float_df, 128, IUV.cols, IUV.rows, v, z, DTTps, ADTTps, realADT); //df of edge
						if (distortionCorrected == 0)
						{
							Generate_Para_Spline(float_df, Para, width, height, -1);

							Point2f *MapXY = vMapXY[lcid];
#pragma omp parallel for
							for (int jj = 0; jj < height; jj++)
							{
								for (int ii = 0; ii < width; ii++)
								{
									int id = ii + jj*width;
									Point2d pt(MapXY[id].x, MapXY[id].y);
									if (pt.x<0 || pt.x > width - 1 || pt.y<0 || pt.y > height - 1)
										float_df[id] = 0.f;
									else
									{
										double value;
										Get_Value_Spline(Para, width, height, pt.x, pt.y, &value, -1, -1);
										float_df[id] = value;
									}
								}
							}
						}

						vDensePose[lcid].validParts[partId] = 1;
#pragma omp parallel for
						for (int ii = 0; ii < length; ii++)
						{
							if (float_df[ii] > 65535)
								vDensePose[lcid].vdfield[partId][ii] = 65535;
							else
								vDensePose[lcid].vdfield[partId][ii] = uint16_t(float_df[ii]);
						}*/
					}
				}
			}
			printLOG(" %.4fs ", omp_get_wtime() - startTime);
			printLOG("\n");
		}

		Point2i fixedPoseFrames(-1, -1);
		FitSMPL2Total(mySMPL, frame_params, frame_skeleton, vDensePose, CamTimeInfo, VideoInfo, vSCams, ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, 100.0*smpl2sfmScale, skeletonPointFormat, fixedPoseFrames, pi, hasDensePose);
		printLOG("\n");

		sprintf(Fname, "%s/extracted_frames/%s/FitBody/f_%d.txt", Path, SeqName, pi);  FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[0].t(0), frame_params[0].t(1), frame_params[0].t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fprintf(fp, "%f %f %f\n", frame_params[0].pose(ii, 0), frame_params[0].pose(ii, 1), frame_params[0].pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fprintf(fp, "%f ", mySMPL.coeffs(ii));
		fprintf(fp, "\n");
		fclose(fp);
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;
	delete[]Para;

	return 0;
}
int FitSMPL_MVS_Driver(char *Path, char *SeqName, int distortionCorrected, int skeletonPointFormat, int Use2DFitting, double *weights, double *isigmas, double Real2SfM, double detectionThresh, bool hasDensePose, int selectedPeopleId = -1)
{
	printLOG("*****************FitSMPL_MVS_Driver [%s]*****************\n", SeqName);
	char Fname[512];
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	printLOG("Reading all camera poses\n");
	vector<VideoData> VideoInfo;
	ReadMannequinCamPose(Path, SeqName, VideoInfo);
	int nCams = VideoInfo.size();

	vector<Point3d> CamTimeInfo(nCams);
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	vector<int> vsCams;
	for (int ii = 0; ii < nCams; ii++)
		vsCams.emplace_back(ii);

	SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	vector<Point2f*> vMapXY;
	DensePose *vDensePose = new DensePose[nCams];
	if (hasDensePose)
	{
		for (int lcid = 0; lcid < nCams; lcid++)
		{
			int width = VideoInfo[lcid].VideoInfo[0].width, height = VideoInfo[lcid].VideoInfo[0].height, length = width * height;
			for (int partId = 14; partId < 15; partId++)
				vDensePose[lcid].vdfield[partId] = new uint16_t[length];

			Point2f *MapXY = 0;
			if (distortionCorrected == 0)
			{
				MapXY = new Point2f[length];
				for (int jj = 0; jj < height; jj++)
					for (int ii = 0; ii < width; ii++)
						MapXY[ii + jj * width] = Point2d(ii, jj);

				if (VideoInfo[lcid].VideoInfo[0].LensModel = RADIAL_TANGENTIAL_PRISM)
					LensDistortionPoint(MapXY, VideoInfo[lcid].VideoInfo[0].K, VideoInfo[lcid].VideoInfo[0].distortion, length);
				else
					FishEyeDistortionPoint(MapXY, VideoInfo[lcid].VideoInfo[0].K, VideoInfo[lcid].VideoInfo[0].distortion[0], length);
			}
			vMapXY.push_back(MapXY);
		}
	}

	int nMaxPeople = 100;
	sprintf(Fname, "%s/%s/FitBody", Path, SeqName), makeDir(Fname);

	int rcid, rfid, did, inlier, nValidJoints = 0, skeI = 0;
	double u, v, s;
	vector<HumanSkeleton3D> vSkeletons;

	while (true)
	{
		HumanSkeleton3D Body;
		sprintf(Fname, "%s/extracted_frames/%s/Ske_%.2d.txt", Path, SeqName, skeI);
		FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			fscanf(fp, "%d ", &rfid); //dummy;
			int  nvis;
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				fscanf(fp, "%lf %lf %lf %d ", &Body.pt3d[jid].x, &Body.pt3d[jid].y, &Body.pt3d[jid].z, &nvis);
				for (int kk = 0; kk < nvis; kk++)
				{
					fscanf(fp, "%d %d %lf %lf ", &rcid, &did, &u, &v); s = 1.0;
					if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[0].width - 1 || v>VideoInfo[rcid].VideoInfo[0].height - 1 || !VideoInfo[rcid].VideoInfo[0].valid)
						continue;

					Point2d uv(u, v);
					if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[0].LensModel == RADIAL_TANGENTIAL_PRISM)
						LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[0].K, VideoInfo[rcid].VideoInfo[0].distortion);
					else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[0].LensModel == FISHEYE)
						FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[0].K, VideoInfo[rcid].VideoInfo[0].distortion[0]);

					Body.vViewID_rFid[jid].push_back(Point2i(rcid, 0));
					Body.vPt2D[jid].push_back(uv);
					Body.vConf[jid].push_back(s);
				}

				Body.validJoints[jid] = 0;
				if (abs(Body.pt3d[jid].x) + abs(Body.pt3d[jid].y) + abs(Body.pt3d[jid].z) > 1e-16)
				{
					nValidJoints++;
					Body.validJoints[jid] = 1;
				}
			}
			fclose(fp);

			Body.valid = nValidJoints < skeletonPointFormat / 3 ? 0 : 1;
			vSkeletons.push_back(Body);
			skeI++;
		}
		else
			break;
	}

	InitializeBodyPoseParameters_MVS(Path, SeqName, vSkeletons, skeletonPointFormat);
	FitSMPL_MVS(Path, SeqName, smplMaster, 0, vSkeletons, vDensePose, &VideoInfo[0], &CamTimeInfo[0], vsCams, distortionCorrected, 1, skeletonPointFormat, Use2DFitting, weights, isigmas, Real2SfM, hasDensePose, nMaxPeople, -1);
	VisualizeProjectedSMPLBody_MVS(Path, SeqName, smplMaster, vSkeletons, &VideoInfo[0], &CamTimeInfo[0], vsCams, skeletonPointFormat, distortionCorrected, 1.0);
	delete[]vDensePose;

	return 0;
}

double FitSMPL_Camera_Total(SMPLModel &mySMPL, vector<SMPLParams> &frame_params, vector<HumanSkeleton3D> &vSkeletons, DensePose *vDensePose, Point3d *CamTimeInfo, VideoData *VideoInfo, vector<int> &vCams, double *ContourPartWeight, double *JointWeight, double *DPPartweight, double *CostWeights, double *isigmas, double Real2SfM, int skeletonPointFormat, Point2i &fixedPoseFrame, int personId, bool hasDensePose)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints, naJoints = SMPLModel::naJoints;

	double plane[3] = { 1.3695914362551931e-03, 1.7627470009816534e-03, 5.6420143860900918e-01 };
	mySMPL.scale = 1.0;

	double w0 = CostWeights[0],  //shape prior
		w1 = CostWeights[1] / frame_params.size(),//pose prior
		w2 = CostWeights[2] / (frame_params.size() - 1), //temporal joint
		w3 = CostWeights[3] / frame_params.size(), //Contour fitting
		w4 = CostWeights[4] / frame_params.size(), //Sil fitting
		w5 = CostWeights[5] / frame_params.size(), //2d fitting
		w6 = CostWeights[6] / frame_params.size(), //dense pose points		
		w7 = CostWeights[7] / frame_params.size(),// headcamera constraints
		w8 = CostWeights[8] / frame_params.size();//feet clamping

	double SilScale = 4.0, GermanMcClure_Scale = 10.0, GermanMcClure_Curve_Influencier = 1.0, GermanMcClure_DP_Influencier = 1.0; //German McClure scale and influential--> look in V. Koltun's Fast Global registration paper
	double isigma_2D = isigmas[1], //2d fitting (pixels)
		isigma_2DSeg = isigmas[2], //2d contour fitting (pixels)
		isigma_Vel = isigmas[3] * Real2SfM,//3d smoothness (mm/s)
		isigma_3D = isigmas[4] * Real2SfM; //3d std (mm)

	vector<int>SMPL2Detection, SMPL2HeadCameras, SMPLFeet;
	if (skeletonPointFormat == 17)
	{
		SMPL2Detection.push_back(0), SMPL2Detection.push_back(0), //noise
			SMPL2Detection.push_back(14), SMPL2Detection.push_back(2), SMPL2Detection.push_back(16), SMPL2Detection.push_back(4), //right face
			SMPL2Detection.push_back(15), SMPL2Detection.push_back(1), SMPL2Detection.push_back(17), SMPL2Detection.push_back(3),//left face
			SMPL2Detection.push_back(2), SMPL2Detection.push_back(6), SMPL2Detection.push_back(3), SMPL2Detection.push_back(8), SMPL2Detection.push_back(4), SMPL2Detection.push_back(10),//right arm
			SMPL2Detection.push_back(5), SMPL2Detection.push_back(5), SMPL2Detection.push_back(6), SMPL2Detection.push_back(7), SMPL2Detection.push_back(7), SMPL2Detection.push_back(9),//left arm
			SMPL2Detection.push_back(8), SMPL2Detection.push_back(12), SMPL2Detection.push_back(9), SMPL2Detection.push_back(14), SMPL2Detection.push_back(10), SMPL2Detection.push_back(16),//right leg
			SMPL2Detection.push_back(11), SMPL2Detection.push_back(11), SMPL2Detection.push_back(12), SMPL2Detection.push_back(13), SMPL2Detection.push_back(13), SMPL2Detection.push_back(15);//left leg		
		SMPLFeet.push_back(9), SMPLFeet.push_back(12);
	}
	else if (skeletonPointFormat == 18)
	{
		//SMPL2Detection.resize(28);
		//SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, //dummy
			//SMPL2Detection[2] = 1, SMPL2Detection[3] = 1,
			//SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4,
			//SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7,
			//SMPL2Detection[16] = 8, SMPL2Detection[17] = 8, SMPL2Detection[18] = 9, SMPL2Detection[19] = 9, SMPL2Detection[20] = 10, SMPL2Detection[21] = 10,
			//SMPL2Detection[22] = 11, SMPL2Detection[23] = 11, SMPL2Detection[24] = 12, SMPL2Detection[25] = 12, SMPL2Detection[26] = 13, SMPL2Detection[27] = 13;

		//the detection order is diff from coco-keypoints
		//SMPL2Detection.push_back(0), SMPL2Detection.push_back(13),
		SMPL2Detection.push_back(1), SMPL2Detection.push_back(12),
			SMPL2Detection.push_back(2), SMPL2Detection.push_back(8), SMPL2Detection.push_back(3), SMPL2Detection.push_back(7), SMPL2Detection.push_back(4), SMPL2Detection.push_back(6),
			SMPL2Detection.push_back(5), SMPL2Detection.push_back(9), SMPL2Detection.push_back(6), SMPL2Detection.push_back(10), SMPL2Detection.push_back(7), SMPL2Detection.push_back(11),
			SMPL2Detection.push_back(8), SMPL2Detection.push_back(2), SMPL2Detection.push_back(9), SMPL2Detection.push_back(1), SMPL2Detection.push_back(10), SMPL2Detection.push_back(0),
			SMPL2Detection.push_back(11), SMPL2Detection.push_back(3), SMPL2Detection.push_back(12), SMPL2Detection.push_back(4), SMPL2Detection.push_back(13), SMPL2Detection.push_back(5);

	}
	else if (skeletonPointFormat == 25)
	{
		SMPL2Detection.resize(48);
		SMPL2Detection[0] = 0, SMPL2Detection[1] = 0, SMPL2Detection[2] = 1, SMPL2Detection[3] = 1, //noise, neck
			SMPL2Detection[4] = 2, SMPL2Detection[5] = 2, SMPL2Detection[6] = 3, SMPL2Detection[7] = 3, SMPL2Detection[8] = 4, SMPL2Detection[9] = 4, //right arm
			SMPL2Detection[10] = 5, SMPL2Detection[11] = 5, SMPL2Detection[12] = 6, SMPL2Detection[13] = 6, SMPL2Detection[14] = 7, SMPL2Detection[15] = 7, //left arm
			SMPL2Detection[16] = 8, SMPL2Detection[17] = 9, SMPL2Detection[18] = 9, SMPL2Detection[19] = 10, SMPL2Detection[20] = 10, SMPL2Detection[21] = 11,//right leg
			SMPL2Detection[22] = 11, SMPL2Detection[23] = 12, SMPL2Detection[24] = 12, SMPL2Detection[25] = 13, SMPL2Detection[26] = 13, SMPL2Detection[27] = 14, //left leg
			SMPL2Detection[28] = 14, SMPL2Detection[29] = 15, SMPL2Detection[30] = 15, SMPL2Detection[31] = 16, SMPL2Detection[32] = 16, SMPL2Detection[33] = 17, SMPL2Detection[34] = 17, SMPL2Detection[35] = 18, //face
			SMPL2Detection[36] = 18, SMPL2Detection[37] = 22, SMPL2Detection[38] = 19, SMPL2Detection[39] = 23, SMPL2Detection[40] = 20, SMPL2Detection[41] = 24, //right foot
			SMPL2Detection[42] = 21, SMPL2Detection[43] = 19, SMPL2Detection[44] = 22, SMPL2Detection[45] = 20, SMPL2Detection[46] = 23, SMPL2Detection[47] = 21;//left foot
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < *max_element(vCams.begin(), vCams.end()) + 1; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	int nInstances = (int)frame_params.size(), nCams = (int)vCams.size();
	vector<Point2i> *vCidFid = new vector<Point2i>[nInstances];
	vector<Point3i> *ValidSegPixels = new vector<Point3i>[nCams*nInstances];
	vector<ImgPoseEle> *allPoseLandmark = new vector<ImgPoseEle>[nInstances]; //NOTE: could contain different cid than Denspose id due to different source of detections.
	for (int idf = 0; idf < nInstances; idf++)
	{
		for (int cid = 0; cid < nCams; cid++)
		{
			ImgPoseEle temp(skeletonPointFormat);
			for (int jid = 0; jid < skeletonPointFormat; jid++)
				temp.pt2D[jid] = Point2d(0, 0);
			temp.viewID = cid;
			temp.frameID = vSkeletons[idf].refFid - VideoInfo[cid].TimeOffset;
			allPoseLandmark[idf].push_back(temp);
		}

		for (int jid = 0; jid < skeletonPointFormat; jid++)
		{
			if (vSkeletons[idf].vViewID_rFid[jid].size() == 0)
				for (int jj = 0; jj < vCams.size(); jj++)
					allPoseLandmark[idf][jj].ts = CamTimeInfo[0].x * vSkeletons[idf].refFid; //asume same fps

			for (int ii = 0; ii < vSkeletons[idf].vViewID_rFid[jid].size(); ii++)
			{
				int rcid = vSkeletons[idf].vViewID_rFid[jid][ii].x, rfid = vSkeletons[idf].vViewID_rFid[jid][ii].y;
				int cid = -1;
				for (int jj = 0; jj < vCams.size(); jj++)
					if (vCams[jj] == rcid)
						cid = jj;
				if (cid == -1)
					continue;

				allPoseLandmark[idf][cid].viewID = rcid, allPoseLandmark[idf][cid].frameID = rfid;
				for (int jj = 0; jj < vCams.size(); jj++)
					allPoseLandmark[idf][jj].ts = CamTimeInfo[refCid].x * vSkeletons[idf].refFid;

				CameraData *Cam = VideoInfo[rcid].VideoInfo;
				if (Cam[rfid].valid)
				{
					allPoseLandmark[idf][cid].viewID = rcid;
					allPoseLandmark[idf][cid].frameID = rfid;
					allPoseLandmark[idf][cid].pt2D[jid] = vSkeletons[idf].vPt2D[jid][ii];
					allPoseLandmark[idf][cid].confidence[jid] = vSkeletons[idf].vConf[jid][ii];

					if (Cam[rfid].ShutterModel != ROLLING_SHUTTER)
						AssembleP(Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, &allPoseLandmark[idf][cid].P[jid * 12]);
					else
						AssembleP_RS(allPoseLandmark[idf][cid].pt2D[jid], Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, Cam[rfid].wt, &allPoseLandmark[idf][cid].P[jid * 12]);

					bool found = false;
					for (int jj = 0; jj < vCidFid[idf].size() && !found; jj++)
						if (vCidFid[idf][jj].x == cid)
							found = true;
					if (!found)
						vCidFid[idf].push_back(Point2i(cid, rfid));
				}
			}
		}
	}

	int maxWidth = 0, maxHeight = 0;
	for (int cid = 0; cid < nCams; cid++)
		maxWidth = max(maxWidth, VideoInfo[vCams[cid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vCams[cid]].VideoInfo[0].height);

	//remove hand and feet because they usually snap to the boundary between itself the the part above it
	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face
	for (int ii = 0; ii < 14; ii++)
		for (size_t jj = 0; jj < vMergedPartId2DensePosePartId[ii].size(); jj++)
			vMergedPartId2DensePosePartId[ii][jj] = vMergedPartId2DensePosePartId[ii][jj] - 1; //to convert from densepose index to 0 based index

	int DensePosePartId2MergedPartId[24] = { 0, 0, 2,1,7,8,12,11,12,11,10,9,10,9,5,6,5,6,3,4,3,4,13,13 };
	//int nSMPL2DetectionPairs = skeletonPointFormat == 17 ? 17 : (18 ? 14 : 24), naJoints3 = naJoints * 3;
	int nSMPL2DetectionPairs = skeletonPointFormat == 18 ? 14 : 24, naJoints3 = naJoints * 3;
	double *residuals = new double[nVertices * 3], rho[3];
	vector<double> VshapePriorCeres, VposePriorCeres, VtemporalRes, VtemporalRes_n, Vfitting2DRes1, Vfitting2DRes1_n, Vfitting2DRes2, Vfitting2DRes2_n, Vfitting2DRes3, Vfitting2DRes3_n,
		Vfitting2DRes4, Vfitting2DRes4_n, Vfitting2DRes5, Vfitting2DRes5_n, Vfitting2DRes6, Vfitting2DRes6_n, Vfitting2DRes7, Vfitting2DRes7_n;

	int nCorrespond3 = nSMPL2DetectionPairs * 3, nJoints3 = nJoints * 3;
	double *All_V_ptr = new double[nVertices * 3 * nInstances],
		*All_dVdp_ptr = new double[nVertices * 3 * nJoints * 3 * nInstances],
		*All_dVdc_ptr = new double[nVertices * 3 * nShapeCoeffs * nInstances],
		*All_dVds_ptr = new double[nVertices * 3 * nInstances],
		*All_aJsmpl_ptr = new double[naJoints3 * nInstances],
		*All_J_ptr = new double[nCorrespond3 * nInstances],
		*All_dJdt_ptr = new double[nCorrespond3 * 3 * nInstances],
		*All_dJdp_ptr = new double[nCorrespond3 * nJoints * 3 * nInstances],
		*All_dJdc_ptr = new double[nCorrespond3 * nShapeCoeffs * nInstances],
		*All_dJds_ptr = new double[nCorrespond3 * nInstances];
	char *All_VertexTypeAndVisibility = 0;
	Point2f *All_uv_ptr = 0;
	if (hasDensePose)
		All_VertexTypeAndVisibility = new char[nCams*nVertices*nInstances],
		All_uv_ptr = new Point2f[nCams*nVertices*nInstances];

	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> J_reg = skeletonPointFormat == 18 ? mySMPL.J_regl_14_bigl_col_ : mySMPL.J_regl_25_bigl_col_;
	SparseMatrix<double, ColMajor>  dVdt = kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	double startTime = omp_get_wtime();
	ceres::Problem problem;
	ceres::Solver::Options options;

	//Evaluator callback
	printLOG("Setting up evalution callback\n");
	vector<double *> Vparameters;
	Vparameters.push_back(&mySMPL.scale);
	Vparameters.push_back(mySMPL.coeffs.data());
	for (int idf = 0; idf < nInstances; idf++)
	{
		Vparameters.push_back(frame_params[idf].pose.data());
		Vparameters.push_back(frame_params[idf].t.data());
	}

	SmplFitCallBack mySmplFitCallBack(mySMPL, nInstances, skeletonPointFormat, Vparameters, All_V_ptr, All_uv_ptr, All_dVdp_ptr, All_dVdc_ptr, All_dVds_ptr, All_aJsmpl_ptr, All_J_ptr, All_dJdt_ptr, All_dJdp_ptr, All_dJdc_ptr, All_dJds_ptr,
		vDensePose, VideoInfo, All_VertexTypeAndVisibility, vCams, maxWidth*maxHeight, hasDensePose);
	mySmplFitCallBack.PrepareForEvaluation(false, true);
	options.evaluation_callback = &mySmplFitCallBack;

	cout << frame_params[0].t << endl;
	cout << frame_params[0].pose << endl;

	//Iterator Callback function
	printLOG("Setting up iterator callback\n");
	int iter = 0, debug = 0;
	Point2d *uvJ = new Point2d[nSMPL2DetectionPairs], *uvV = new Point2d[nVertices];
	bool *hit = new bool[maxWidth*maxHeight];
	class MyCallBack : public ceres::IterationCallback
	{
	public:
		MyCallBack(std::function<void()> callback, int &iter, int &debug) : callback_(callback), iter(iter), debug(debug) {}
		virtual ~MyCallBack() {}

		ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
		{
			iter = summary.iteration;
			if (summary.step_is_successful)
				callback_();
			return ceres::SOLVER_CONTINUE;
		}
		int &iter, &debug;
		std::function<void()> callback_;
	};
	auto update_Result = [&]()
	{
#ifdef _WINDOWS
		int idf = 0, cidI = 0;
		if (debug > 0)
		{
			char Fname[512];
			for (int idf = 0; idf < nInstances; idf++)
			{
				int idf1 = idf + 1;
				double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf;
				//char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nCams * nVertices*idf;
				//DensePose* DensePoseF = vDensePose + nCams * idf;

				vector<int> bestInliers;
				vector<Point3d> Vxyz;
				vector<Point2d> Vuv;

				Mat img = imread("E:/img.png");
				int width = img.cols, height = img.rows;
				VideoInfo[0].VideoInfo[0].width = width, VideoInfo[0].VideoInfo[0].height = height;

				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
				{
					int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
					Vxyz.emplace_back(J0_ptr[SMPLid * 3], J0_ptr[SMPLid * 3 + 1], J0_ptr[SMPLid * 3 + 2]);
					Vuv.emplace_back(vSkeletons[0].vPt2D[DetectionId][0].x, vSkeletons[0].vPt2D[DetectionId][0].y);
				}

				PnP_RANSAC(VideoInfo[0].VideoInfo[0], bestInliers, Vxyz, Vuv, VideoInfo[0].VideoInfo[0].width, VideoInfo[0].VideoInfo[0].height, 0, 100, 7, 10, 0, 0, PnP::P4Pf);

				GetRTFromrt(VideoInfo[0].VideoInfo[0]);
				AssembleP(VideoInfo[0].VideoInfo[0]);
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
				{
					Point2d uv;
					ProjectandDistort(Vxyz[ii], &uv, VideoInfo[0].VideoInfo[0].P);
					circle(img, uv, 1, Scalar(0, 255, 0));
					circle(img, Vuv[ii], 1, Scalar(0, 0, 255));
				}
				sprintf(Fname, "E:/img_%d.png", iter); imwrite(Fname, img);
				img = imread("E:/img.png");


				for (int lcid = 0; lcid < nCams; lcid++)
				{
					//int rcid = DensePoseF[lcid].cid, rfid = DensePoseF[lcid].fid, width = VideoInfo[lcid].VideoInfo[rfid].width, height = VideoInfo[lcid].VideoInfo[rfid].height;
					int rcid = allPoseLandmark[idf][lcid].viewID, rfid = allPoseLandmark[idf][lcid].frameID;
					if (rcid < 0 || rfid < 0)
						continue;

					CameraData *camI = VideoInfo[rcid].VideoInfo;
					//double newfocal = 500;
					//camI[0].K[0] = newfocal; camI[0].K[4] = newfocal;
					//AssembleP(camI[0]);


					int offset = lcid * nVertices;
					//#pragma omp parallel for schedule(dynamic,1)
					for (int vid = 0; vid < nVertices; vid++)
					{
						int vid3 = vid * 3;
						Point3d xyz(V0_ptr[vid3], V0_ptr[vid3 + 1], V0_ptr[vid3 + 2]);
						ProjectandDistort(xyz, &uvV[vid], camI[rfid].P);// , camI[rfid].K, camI[rfid].distortion);

						if (uvV[vid].x<15 || uvV[vid].x>width - 15 || uvV[vid].y<15 || uvV[vid].y>height - 15)
							uvV[vid].x = 0, uvV[vid].y = 0;
					}
					for (int ii = 0; ii < mySMPL.vFaces.size(); ii++)
					{
						Point2d uv[3] = { uvV[mySMPL.vFaces[ii].x],uvV[mySMPL.vFaces[ii].y],uvV[mySMPL.vFaces[ii].z] };
						if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[1].x >10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10)
							cv::line(img, uv[0], uv[1], Scalar(0, 255, 0), 1, CV_AA);
						if (uv[1].x > 10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
							cv::line(img, uv[0], uv[2], Scalar(0, 255, 0), 1, CV_AA);
						if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
							cv::line(img, uv[1], uv[2], Scalar(0, 255, 0), 1, CV_AA);
					}
					//circle(img, uvV[vid], 1, Scalar(0, 0, 255));
					sprintf(Fname, "E:/img__%d.png", iter); imwrite(Fname, img);
					printf("%.3f %.3f\n", camI[rfid].K[0], camI[rfid].K[4]);
					int a = 0;

					//if (hasDensePose && DensePoseF[lcid].valid == 1)
					/*{
						char Fname[512]; //sprintf(Fname, "G:/NEA1/Corrected/%d/%.4d.jpg", rcid, rfid);
						char Path[] = { "E:/Dataset" };
						vector<string> SelectedCamNames;
						SelectedCamNames.push_back("T42664764_rjohnston");
						SelectedCamNames.push_back("T42664773_rjohnston");
						SelectedCamNames.push_back("T42664789_rjohnston");
						vector<int> CamIdsPerSeq; CamIdsPerSeq.push_back(2), CamIdsPerSeq.push_back(3), CamIdsPerSeq.push_back(4);
						int SeqId = 0;

						double gamma = lcid % 3 == 0 ? 0.8 : 0.4;
						sprintf(Fname, "%s/%s/general_%d_%d/image_%.10d_0.png", Path, SelectedCamNames[lcid / CamIdsPerSeq.size()].c_str(), SeqId, CamIdsPerSeq[lcid%CamIdsPerSeq.size()], rfid);
						Mat img = correctGamma(imread(Fname), gamma);
						//get joint projection
						Point2d uv;
						for (int jid = 0; jid < nCorrespond3 / 3; jid++)
						{
							int jid3 = jid * 3;
							Point3d xyz(J0_ptr[jid3], J0_ptr[jid3 + 1], J0_ptr[jid3 + 2]);
							ProjectandDistort(xyz, &uv, camI[rfid].P);
							LensDistortionPoint_KB3(uv, camI[rfid].intrinsic, camI[rfid].distortion);
							circle(img, uv, 1, Scalar(0, 0, 255));
							//if (debug == 3)
							//	sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
						}

						for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
						{
							if (!IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
								continue;
							int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
							ProjectandDistort(vSkeletons[idf].pt3d[DetectionId], &uv, camI[rfid].P);
							LensDistortionPoint_KB3(uv, camI[rfid].intrinsic, camI[rfid].distortion);
							circle(img, uv, 1, Scalar(255, 0, 0));
							//if (debug == 3)
							//	sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
						}


						for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
						{
							int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
							if (allPoseLandmark[idf][lcid].pt2D[DetectionId].x != 0.0)
							{
								uv.x = allPoseLandmark[idf][lcid].pt2D[DetectionId].x, uv.y = allPoseLandmark[idf][lcid].pt2D[DetectionId].y;
								LensDistortionPoint_KB3(uv, camI[rfid].intrinsic, camI[rfid].distortion);
								circle(img, uv, 1, Scalar(0, 255, 0));
								if (debug == 3)
									sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);
							}
						}
						//get occluding contour vertices
						int offset = lcid * nVertices;
#pragma omp parallel for schedule(dynamic,1)
						for (int vid = 0; vid < nVertices; vid++)
						{
							int vid3 = vid * 3;
							Point3d xyz(V0_ptr[vid3], V0_ptr[vid3 + 1], V0_ptr[vid3 + 2]);
							ProjectandDistort(xyz, &uvV[vid], camI[rfid].P);// , camI[rfid].K, camI[rfid].distortion);

							if (uvV[vid].x<15 || uvV[vid].x>width - 15 || uvV[vid].y<15 || uvV[vid].y>height - 15)
								uvV[vid].x = 0, uvV[vid].y = 0;
						}
						for (int vid = 0; vid < nVertices; vid++)
						{
							if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1)
								circle(img, uvV[vid], 1, Scalar(0, 255, 0));
							else
								circle(img, uvV[vid], 1, Scalar(0, 0, 255));
						}
						sprintf(Fname, "%s/Vis/FitBody/%d.jpg", Path, lcid), imwrite(Fname, img);

						//sprintf(Fname, "C:/temp/%.4d_%d_oc_%d.txt", lcid, rfid, personId);	FILE *fp = fopen(Fname, "w");
						//for (int vid = 0; vid < nVertices; vid++)
						//	if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1)
						//		fprintf(fp, "%d %d %.1f %.1f \n", vid, EdgeVertexTypeAndVisibilityF[vid + offset], uvV[vid].x, uvV[vid].y);
						//fclose(fp);
					}*/
				}
			}
		}
		iter++;
#endif
	};
	options.callbacks.push_back(new MyCallBack(update_Result, iter, debug));

	debug = 2;
	MyCallBack *myCallback = new MyCallBack(update_Result, iter, debug);

	bool enough = false;
	while (!enough)
		myCallback->callback_();

	//Ceres residual blocks
	printLOG("Setting up residual blocks\n");
	ceres::LossFunction* coeffs_loss = new ceres::ScaledLoss(NULL, w0, ceres::TAKE_OWNERSHIP);
	ceres::CostFunction *coeffs_reg = new ceres::AutoDiffCostFunction	< SMPLShapeCoeffRegCeres, nShapeCoeffs, nShapeCoeffs >(new SMPLShapeCoeffRegCeres(nShapeCoeffs));
	problem.AddResidualBlock(coeffs_reg, coeffs_loss, mySMPL.coeffs.data());

	const double * parameters[] = { mySMPL.coeffs.data() };
	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0 *0.5*residuals[ii] * residuals[ii]);

	Map< MatrixXdr > Mosh_pose_prior_mu(mySMPL.Mosh_pose_prior_mu, nJoints * 3, 1);
	for (int idf = 0; idf < nInstances; idf++)
	{
		//pose prior
		ceres::LossFunction* pose_regl_loss = new ceres::ScaledLoss(NULL, w1, ceres::TAKE_OWNERSHIP);
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
		//ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.Mosh_pose_prior_A, Mosh_pose_prior_mu);
		problem.AddResidualBlock(pose_reg, pose_regl_loss, frame_params[idf].pose.data());

		const double * parameters1[] = { frame_params[idf].pose.data() };
		pose_reg->Evaluate(parameters1, residuals, NULL);
		for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
			VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

		for (int ii = 3; ii < nJoints * 3; ii++)
		{
			problem.SetParameterLowerBound(frame_params[idf].pose.data(), ii, mySMPL.minPose[ii]);
			problem.SetParameterUpperBound(frame_params[idf].pose.data(), ii, mySMPL.maxPose[ii]);
		}

		//edge fittimg cost and point fitting
		int nValidPoints = 0;
		double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
		double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
		double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
		char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * nCams*idf;
		DensePose *DensePoseF = vDensePose + idf * nCams;
		Point2f *vUV_ptr = All_uv_ptr + nVertices * idf*nCams;

		if (hasDensePose)
		{
			//1. Contour fiting
			double  isigma2 = isigma_2DSeg * isigma_2DSeg;
			if (w3 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1 && DensePoseF[cid].validParts[dfId] > 0)
							nValidPoints++;
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);
						SmplFitSMPL2EdgeCeres_MV2 *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV2(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid],
							EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						if (EdgeVertexTypeAndVisibilityF[offset + vid] == -1 || DensePoseF[cid].validParts[dfId] == 0)
							continue;

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
					}
				}

				//other direction
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int partId = 0; partId < 14; partId++)
						nValidPoints += (int)DensePoseF[cid].vEdge[partId].size() / 2;
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int partId = 0; partId < 14; partId++)
					{
						for (int eid = 0; eid < DensePoseF[cid].vEdge[partId].size(); eid += 2)
						{
							double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
							//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
							ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
							ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw3, ceres::TAKE_OWNERSHIP);
							//SmplFitEdge2SMPLCeres_MV *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], 
							//	EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

							SmplFitEdge2SMPLCeres_MV2 *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV2(V_ptr, vUV_ptr + nVertices * cid, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
								EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
							problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double resX = residuals[0], resY = residuals[1], res2 = resX * resX + resY * resY;
							resX = resX / isigma_2DSeg, resY = resY / isigma_2DSeg;
							robust_loss->Evaluate(res2, rho);
							Vfitting2DRes2_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
						}
					}
				}
			}

			//2. Sil fitting
			if (w4 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					nValidPoints += nVertices;
				}
				double nw4 = w4 / nValidPoints;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						ceres::LossFunction *robust_loss = new ceres::CauchyLoss(1.0); //works better than huber////new GermanMcClure(GermanMcClure_Scale, 10.0*SilScale*GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw4, ceres::TAKE_OWNERSHIP);

						SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, DensePoseF[cid], VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0];
						robust_loss->Evaluate(res2, rho);
						Vfitting2DRes3_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes3.push_back(res2 / isigma2);
					}
				}
			}
		}

		//3. Point fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				{
					int DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
						nValidPoints++;
				}
			}
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				{
					int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
					{
						SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
							mySMPL, &allPoseLandmark[idf][cid].P[12 * DetectionId], SMPLid, allPoseLandmark[idf][cid].pt2D[DetectionId], skeletonPointFormat, isigma_2D); //can work with Rolling RS where each point has diff pose

						double nw5 = w5 * JointWeight[SMPLid] * allPoseLandmark[idf][cid].confidence[DetectionId] / (0.0001 + nValidPoints);
						ceres::LossFunction* robust_loss = new HuberLoss(1);
						//ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
						ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
						problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1];
						robust_loss->Evaluate(resX*resX + resY * resY, rho);
						Vfitting2DRes4_n.push_back(nw5*0.5*rho[0]);

						resX = resX / isigma_2D, resY = resY / isigma_2D;
						Vfitting2DRes4.push_back(resX), Vfitting2DRes4.push_back(resY);
					}
				}
			}
		}

		//for (int ii = 0; ii < nCorrespond3 / 3; ii++)
		//	printf("%.5f %.5f %.5f\n", J_ptr[ii * 3], J_ptr[ii * 3 + 1], J_ptr[ii * 3 + 2]);

		//4. smpl2Pts3d fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				if (IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					nValidPoints++;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
			{
				if (!IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					continue;

				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, vSkeletons[idf].pt3d[DetectionId], skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose
				//printf("%.4f %.4f %.4f  %.4f %.4f %.4f \n", vSkeletons[idf].pt3d[DetectionId].x, vSkeletons[idf].pt3d[DetectionId].y, vSkeletons[idf].pt3d[DetectionId].z, J_ptr[3 * SMPLid], J_ptr[3 * SMPLid + 1], J_ptr[3 * SMPLid + 2]);

				double nw5 = 0.05*w5 * JointWeight[SMPLid] / (0.0001 + nValidPoints); //very small weight of w5
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				//ceres::LossFunction *robust_loss = new GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw5, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				robust_loss->Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				//rho[0] = resX * resX + resY * resY + resZ * resZ;
				Vfitting2DRes5_n.push_back(nw5*0.5*rho[0]);

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes5.push_back(resX), Vfitting2DRes5.push_back(resY), Vfitting2DRes5.push_back(resZ);
			}
		}

		//5. set for camera head pose constraints
		if (w7 > 0.0)
		{
			nValidPoints = 2;
			for (int ii = 0; ii < SMPL2HeadCameras.size() / 2; ii++)
			{
				int SMPLid = SMPL2HeadCameras[2 * ii], cid = SMPL2HeadCameras[2 * ii + 1], rfid = allPoseLandmark[idf][cid].frameID;
				Point3d anchor(VideoInfo[cid].VideoInfo[rfid].camCenter[0], VideoInfo[cid].VideoInfo[rfid].camCenter[1], VideoInfo[cid].VideoInfo[rfid].camCenter[2]);
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, anchor, skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw7 = w7 / nValidPoints;
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(NULL, nw7, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				Vfitting2DRes6_n.push_back(nw7*0.5*(resX*resX + resY * resY + resZ * resZ));

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes6.push_back(resX), Vfitting2DRes6.push_back(resY), Vfitting2DRes6.push_back(resZ);
			}
		}

		//6. set foot clamping
		if (w8 > 0.0)
		{
			nValidPoints = 2;
			for (int ii = 0; ii < 2; ii++)
			{
				int SMPLid = SMPLFeet[ii];
				SmplClampGroundCeres *fit_cost_analytic_fr = new SmplClampGroundCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr, mySMPL, SMPLid, plane, skeletonPointFormat, isigma_3D);

				double nw8 = w8 / nValidPoints;
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, nw8, ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(fit_cost_analytic_fr, weightLoss, frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data(), &mySMPL.scale);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double res = residuals[0];
				robust_loss->Evaluate(res*res, rho);
				Vfitting2DRes7_n.push_back(nw8*0.5*rho[0]);

				res = res / isigma_3D;
				Vfitting2DRes7.push_back(res);
			}
		}
	}
	if (nInstances > 1 && w2 > 0.0)
	{
		for (int idf = 0; idf < nInstances - 1; idf++)
		{
			int idf1 = idf + 1;
			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2, ceres::TAKE_OWNERSHIP);
			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			//ceres::LossFunction* avgl_loss = new ceres::ScaledLoss(NULL, w2 / nVertices, ceres::TAKE_OWNERSHIP);
			//SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			problem.AddResidualBlock(temporal_cost_analytic_fr, avgl_loss, &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data());

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = allPoseLandmark[idf + 1][0].ts - allPoseLandmark[idf][0].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
				//VtemporalRes_n.push_back(w2 / nVertices*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / sqrt(dt)*sqrt(nVertices * 3) / Real2SfM, 2)); //in mm/s
			}
		}
	}

	//fix shape
	//mySMPL.coeffs(0) = 0.0;
	//std::vector<int> constant_parameters;
	//for (int ii = 0; ii < 1; ii++)
	//	constant_parameters.push_back(ii);
	//problem.SetParameterization(mySMPL.coeffs.data(), new ceres::SubsetParameterization(10, constant_parameters));
	//mySMPL.coeffs.setZero(); problem.SetParameterBlockConstant(mySMPL.coeffs.data());
	mySMPL.scale = 1.0;  problem.SetParameterBlockConstant(&mySMPL.scale);
	//problem.SetParameterLowerBound(&mySMPL.scale, 0, 0.7*mySMPL.scale);
	//problem.SetParameterUpperBound(&mySMPL.scale, 0, 1.3*mySMPL.scale);
	if (w0 > 1000000)
	{
		mySMPL.coeffs.setZero();
		problem.SetParameterBlockConstant(mySMPL.coeffs.data());
	}

	//fix frames from previous window
	if (fixedPoseFrame.x != -1 && fixedPoseFrame.y != -1)
	{
		problem.SetParameterBlockConstant(&mySMPL.scale);
		problem.SetParameterBlockConstant(mySMPL.coeffs.data());
		for (int idf = 0; idf < nInstances; idf++)
		{
			if (frame_params[idf].frame >= fixedPoseFrame.x && frame_params[idf].frame <= fixedPoseFrame.y)
			{
				problem.SetParameterBlockConstant(frame_params[idf].t.data());
				problem.SetParameterBlockConstant(frame_params[idf].pose.data());
			}
		}
	}

	{
		Vfitting2DRes7.push_back(0), Vfitting2DRes7_n.push_back(0), Vfitting2DRes6.push_back(0), Vfitting2DRes6_n.push_back(0), Vfitting2DRes5.push_back(0), Vfitting2DRes5_n.push_back(0),
			Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
		Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
		Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
		Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
		Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
		Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
		Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
		Map< VectorXd > eVfitting2DRes3(&Vfitting2DRes3[0], Vfitting2DRes3.size());
		Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
		Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());
		Map< VectorXd > eVfitting2DRes5_n(&Vfitting2DRes5_n[0], Vfitting2DRes5_n.size());
		Map< VectorXd > eVfitting2DRes6_n(&Vfitting2DRes6_n[0], Vfitting2DRes6_n.size());
		Map< VectorXd > eVfitting2DRes7_n(&Vfitting2DRes7_n[0], Vfitting2DRes7_n.size());

		double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
			sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
			sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
			sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
			sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size()),
			sos_Vfitting2DRes3 = eVfitting2DRes3.sum(), rmse_Vfitting2DRes3 = sqrt(sos_Vfitting2DRes3 / Vfitting2DRes3.size()),
			sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size());
		double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
			mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
		double sos_Vfitting2DRes5_n = eVfitting2DRes5_n.sum(), rmse_Vfitting2DRes5_n = sqrt(sos_Vfitting2DRes5_n / Vfitting2DRes5_n.size()),
			mu5 = MeanArray(Vfitting2DRes5), stdev5 = sqrt(VarianceArray(Vfitting2DRes5, mu5));
		double sos_Vfitting2DRes6_n = eVfitting2DRes6_n.sum(), rmse_Vfitting2DRes6_n = sqrt(sos_Vfitting2DRes6_n / Vfitting2DRes6_n.size()),
			mu6 = MeanArray(Vfitting2DRes6), stdev6 = sqrt(VarianceArray(Vfitting2DRes6, mu6));
		double sos_Vfitting2DRes7_n = eVfitting2DRes7_n.sum(), rmse_Vfitting2DRes7_n = sqrt(sos_Vfitting2DRes7_n / Vfitting2DRes7_n.size()),
			mu7 = MeanArray(Vfitting2DRes7), stdev7 = sqrt(VarianceArray(Vfitting2DRes7, mu7));
		if (nInstances > 1)
		{
			Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
			double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
			printLOG("Before Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("2D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			printLOG("3D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D head-mounted camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
		}
		else
		{
			printLOG("Before Optim\nInit scale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			if (nInstances == 1)
				printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
		}
	}

	options.num_threads = omp_get_max_threads();
	options.eta = 1e-3;
	options.dynamic_sparsity = true;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
#ifndef _DEBUG
	options.minimizer_progress_to_stdout = frame_params.size() > 1 ? true : false;
#else
	options.minimizer_progress_to_stdout = true;
#endif
	options.update_state_every_iteration = true;
	options.max_num_iterations = 100;

#ifndef _DEBUG
	if (frame_params.size() < 5)
		options.max_solver_time_in_seconds = 300;
	else if (frame_params.size() < 50)
		options.max_solver_time_in_seconds = 400;
	else if (frame_params.size() < 500)
		options.max_solver_time_in_seconds = 600;
	else if (frame_params.size() < 1500)
		options.max_solver_time_in_seconds = 800;
	else
		options.max_solver_time_in_seconds = 1000;
#endif // !_DEBUG

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	mySmplFitCallBack.PrepareForEvaluation(false, true);

	//debug = 2;
	myCallback->callback_();

	VshapePriorCeres.clear(), VposePriorCeres.clear(), VtemporalRes_n.clear(), VtemporalRes.clear(), Vfitting2DRes1_n.clear(), Vfitting2DRes1.clear();
	Vfitting2DRes2_n.clear(), Vfitting2DRes2.clear(), Vfitting2DRes3.clear(), Vfitting2DRes3_n.clear(), Vfitting2DRes4.clear(), Vfitting2DRes4_n.clear(), Vfitting2DRes5.clear(), Vfitting2DRes5_n.clear();
	Vfitting2DRes6.clear(), Vfitting2DRes6_n.clear(), Vfitting2DRes7.clear(), Vfitting2DRes7_n.clear();

	coeffs_reg->Evaluate(parameters, residuals, NULL);
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		VshapePriorCeres.push_back(w0  *0.5*residuals[ii] * residuals[ii]);

	for (int idf = 0; idf < nInstances; idf++)
	{
		ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.pose_prior_A, mySMPL.pose_prior_mu);
		//ceres::CostFunction *pose_reg = new ceres::NormalPrior(mySMPL.Mosh_pose_prior_A, Mosh_pose_prior_mu);
		const double * parameters1[] = { frame_params[idf].pose.data() };
		pose_reg->Evaluate(parameters1, residuals, NULL);
		for (int ii = 0; ii < (nJoints - 1) * 3; ii++)
			VposePriorCeres.push_back(w1 *0.5*residuals[ii] * residuals[ii]);

		int nValidPoints = 0;
		double * parameters2[4] = { frame_params[idf].t.data(), frame_params[idf].pose.data(), mySMPL.coeffs.data() ,&mySMPL.scale };
		double *V_ptr = All_V_ptr + nVertices * 3 * idf, *dVdp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf, *dVdc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf, *dVds_ptr = All_dVds_ptr + nVertices * 3 * idf;
		double *J_ptr = All_J_ptr + nCorrespond3 * idf, *dJdt_ptr = All_dJdt_ptr + nCorrespond3 * idf, *dJdp_ptr = All_dJdp_ptr + nCorrespond3 * nJoints3*idf, *dJdc_ptr = All_dJdc_ptr + nCorrespond3 * nShapeCoeffs*idf, *dJds_ptr = All_dJds_ptr + nCorrespond3 * idf;
		char *EdgeVertexTypeAndVisibilityF = All_VertexTypeAndVisibility + nVertices * nCams*idf;
		DensePose *DensePoseF = vDensePose + idf * nCams;
		Point2f *vUV_ptr = All_uv_ptr + nVertices * idf*nCams;

		if (hasDensePose)
		{
			//1. Contour fitting
			if (w3 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						if (EdgeVertexTypeAndVisibilityF[vid + offset] > -1 && DensePoseF[cid].validParts[dfId] > 0)
							nValidPoints++;
					}
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int vid = 0; vid < nVertices; vid += 2)
					{
						int partId = mySMPL.vDensePosePartId[vid], dfId = DensePosePartId2MergedPartId[partId];
						int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
						if (EdgeVertexTypeAndVisibilityF[offset + vid] == -1 || DensePoseF[cid].validParts[dfId] == 0)
							continue;

						double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
						//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
						SmplFitSMPL2EdgeCeres_MV2 *fit_cost_analytic_fr = new SmplFitSMPL2EdgeCeres_MV2(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid],
							EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

						GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
						Vfitting2DRes1_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes1.push_back(res2 / isigma2);
					}
				}

				//other direction
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int partId = 0; partId < 14; partId++)
						nValidPoints += (int)DensePoseF[cid].vEdge[partId].size() / 2;
				}
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;

					for (int partId = 0; partId < 14; partId++)
					{
						for (int eid = 0; eid < DensePoseF[cid].vEdge[partId].size(); eid += 2)
						{
							double nw3 = w3 * ContourPartWeight[partId] / (0.0001 + nValidPoints);
							//double nw3 = w3 *DPPartweight[partId] / (0.0001 + vNvalidPointsPerPart[partId]); //not working well
							//SmplFitEdge2SMPLCeres_MV *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
							//	EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs

							SmplFitEdge2SMPLCeres_MV2 *fit_cost_analytic_fr = new SmplFitEdge2SMPLCeres_MV2(V_ptr, vUV_ptr + nVertices * cid, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr, mySMPL, DensePoseF[cid], vMergedPartId2DensePosePartId,
								EdgeVertexTypeAndVisibilityF + offset, VideoInfo[rcid].VideoInfo[rfid].P, rcid, skeletonPointFormat, DensePoseF[cid].vEdge[partId][eid], partId, isigma_2DSeg); //assume global RS is suffient for seg related stuffs
							fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
							double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

							GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
							Vfitting2DRes2_n.push_back(nw3*0.5*rho[0]), Vfitting2DRes2.push_back(res2 / isigma2);
						}
					}
				}
			}

			//2. Sil fitting
			if (w4 > 0.0)
			{
				nValidPoints = 0;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					nValidPoints += nVertices;
				}
				double nw4 = w4 / nValidPoints;
				for (int cid = 0; cid < nCams; cid++)
				{
					int offset = cid * nVertices;
					int rcid = DensePoseF[cid].cid, rfid = DensePoseF[cid].fid;
					if (DensePoseF[cid].valid == 0 || VideoInfo[rcid].VideoInfo[rfid].valid == 0)
						continue;
					for (int vid = 0; vid < nVertices; vid++)
					{
						SmplFitSilCeres_MV *fit_cost_analytic_fr = new SmplFitSilCeres_MV(V_ptr, dVdt, dVdp_ptr, dVdc_ptr, dVds_ptr,
							mySMPL, DensePoseF[cid], VideoInfo[rcid].VideoInfo[rfid].P, rcid, vid, skeletonPointFormat, isigma_2DSeg);

						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double res2 = residuals[0] * residuals[0], isigma2 = isigma_2DSeg * isigma_2DSeg;

						//GermanMcClure(GermanMcClure_Scale, SilScale*GermanMcClure_Curve_Influencier).Evaluate(res2, rho);
						ceres::CauchyLoss(1.0).Evaluate(res2, rho); //works better than huber
						Vfitting2DRes3_n.push_back(nw4*0.5*rho[0]), Vfitting2DRes3.push_back(res2 / isigma2);
					}
				}
			}
		}

		//3. KPoint fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int cid = 0; cid < nCams; cid++)
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
					if (allPoseLandmark[idf][cid].pt2D[SMPL2Detection[2 * ii + 1]].x != 0.0)
						nValidPoints++;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				{
					int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
					if (allPoseLandmark[idf][cid].pt2D[DetectionId].x != 0.0)
					{
						allPoseLandmark[idf][cid].viewID;
						allPoseLandmark[idf][cid].frameID;

						SmplFitCOCO2DCeres_MV *fit_cost_analytic_fr = new SmplFitCOCO2DCeres_MV(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
							mySMPL, &allPoseLandmark[idf][cid].P[12 * DetectionId], SMPLid, allPoseLandmark[idf][cid].pt2D[DetectionId], skeletonPointFormat, isigma_2D);

						double nw5 = w5 * JointWeight[SMPLid] * allPoseLandmark[idf][cid].confidence[DetectionId] / nValidPoints;
						fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
						double resX = residuals[0], resY = residuals[1];
						ceres::HuberLoss(1).Evaluate(resX*resX + resY * resY, rho);
						//GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(resX*resX + resY * resY, rho);
						Vfitting2DRes4_n.push_back(nw5*0.5*rho[0]);

						resX = resX / isigma_2D, resY = resY / isigma_2D;
						Vfitting2DRes4.push_back(resX), Vfitting2DRes4.push_back(resY);
					}
				}
			}
		}

		//4. smpl2Pts3d fitting
		if (w5 > 0.0)
		{
			nValidPoints = 0;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++) //COCO-SMPL pair
				if (IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					nValidPoints++;
			for (int ii = 0; ii < SMPL2Detection.size() / 2; ii++)
			{
				int SMPLid = SMPL2Detection[2 * ii], DetectionId = SMPL2Detection[2 * ii + 1];
				if (!IsValid3D(vSkeletons[idf].pt3d[SMPL2Detection[2 * ii + 1]]))
					continue;

				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, vSkeletons[idf].pt3d[DetectionId], skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw5 = 0.05*w5 * JointWeight[SMPLid] / (0.0001 + nValidPoints);
				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				//rho[0] = resX * resX + resY * resY + resZ * resZ;
				HuberLoss(1).Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				//GermanMcClure(GermanMcClure_Scale, GermanMcClure_Curve_Influencier).Evaluate(resX*resX + resY * resY + resZ * resZ, rho);
				Vfitting2DRes5_n.push_back(nw5*0.5*rho[0]);

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes5.push_back(resX), Vfitting2DRes5.push_back(resY), Vfitting2DRes5.push_back(resZ);
			}
		}

		//5. set for camera head pose constraints
		if (w7 > 0.0)
		{
			nValidPoints = 3;
			for (int ii = 0; ii < SMPL2HeadCameras.size() / 2; ii++)
			{
				int SMPLid = SMPL2HeadCameras[2 * ii], cid = SMPL2HeadCameras[2 * ii + 1], rfid = allPoseLandmark[idf][cid].frameID;
				Point3d anchor(VideoInfo[cid].VideoInfo[rfid].camCenter[0], VideoInfo[cid].VideoInfo[rfid].camCenter[1], VideoInfo[cid].VideoInfo[rfid].camCenter[2]);
				SmplFitCOCO3DCeres *fit_cost_analytic_fr = new SmplFitCOCO3DCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr,
					mySMPL, SMPLid, anchor, skeletonPointFormat, isigma_3D); //can work with Rolling RS where each point has diff pose

				double nw7 = w7 / nValidPoints;
				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double resX = residuals[0], resY = residuals[1], resZ = residuals[2];
				Vfitting2DRes6_n.push_back(nw7*0.5*(resX*resX + resY * resY + resZ * resZ));

				resX = resX / isigma_3D, resY = resY / isigma_3D, resZ = resZ / isigma_3D;
				Vfitting2DRes6.push_back(resX), Vfitting2DRes6.push_back(resY), Vfitting2DRes6.push_back(resZ);
			}
		}

		//6. set foot clamping
		if (w8 > 0.0)
		{
			nValidPoints = 2;
			for (int ii = 0; ii < 2; ii++)
			{
				int SMPLid = SMPLFeet[ii];
				SmplClampGroundCeres *fit_cost_analytic_fr = new SmplClampGroundCeres(V_ptr, J_ptr, dJdt_ptr, dJdp_ptr, dJdc_ptr, dJds_ptr, mySMPL, SMPLid, plane, skeletonPointFormat, isigma_3D);

				double nw8 = w8 / nValidPoints;
				ceres::LossFunction* robust_loss = new HuberLoss(1);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(NULL, nw8, ceres::TAKE_OWNERSHIP);

				fit_cost_analytic_fr->Evaluate(parameters2, residuals, NULL);
				double res = residuals[0];
				robust_loss->Evaluate(res*res, rho);
				Vfitting2DRes7_n.push_back(nw8*0.5*rho[0]);

				res = res / isigma_3D;
				Vfitting2DRes7.push_back(res);
			}
		}
	}
	if (nInstances > 1 && w2 > 0.0)
	{
		for (int idf = 0; idf < nInstances - 1; idf++)
		{
			int idf1 = idf + 1;
			double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_aJsmpl_ptr + naJoints3 * idf,
				//double *V0_ptr = All_V_ptr + nVertices * 3 * idf, *J0_ptr = All_J_ptr + nCorrespond3 * idf,
				*dV0dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf,
				*dV0dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf,
				*dV0ds_ptr = All_dVds_ptr + nVertices * 3 * idf;
			double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_aJsmpl_ptr + naJoints3 * idf1,
				//double *V1_ptr = All_V_ptr + nVertices * 3 * idf1, *J1_ptr = All_J_ptr + nCorrespond3 * idf1,
				*dV1dp_ptr = All_dVdp_ptr + nVertices * 3 * nJoints * 3 * idf1,
				*dV1dc_ptr = All_dVdc_ptr + nVertices * 3 * nShapeCoeffs * idf1,
				*dV1ds_ptr = All_dVds_ptr + nVertices * 3 * idf1;

			//Smpl2OPKeypointsTemporalReg *temporal_cost_analytic_fr = new Smpl2OPKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
			//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			aSmplKeypointsTemporalReg *temporal_cost_analytic_fr = new aSmplKeypointsTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);
			//SmplVerticesTemporalReg *temporal_cost_analytic_fr = new SmplVerticesTemporalReg(V0_ptr, J0_ptr, dV0dp_ptr, dV0dc_ptr, dV0ds_ptr,
				//V1_ptr, J1_ptr, dV1dp_ptr, dV1dc_ptr, dV1ds_ptr, mySMPL, dVdt, skeletonPointFormat, allPoseLandmark[idf][0].ts, allPoseLandmark[idf + 1][0].ts, isigma_Vel);

			const double * parameters[] = { &mySMPL.scale, mySMPL.coeffs.data(), frame_params[idf].t.data(), frame_params[idf].pose.data(), frame_params[idf + 1].t.data(), frame_params[idf + 1].pose.data() };
			temporal_cost_analytic_fr->Evaluate(parameters, residuals, NULL);
			for (int ii = 0; ii < naJoints3; ii++)
				//for (int ii = 0; ii < nVertices * 3; ii++)
			{
				double w = mySMPL.Mosh_asmpl_J_istd[ii / 3], dt = allPoseLandmark[idf + 1][0].ts - allPoseLandmark[idf][0].ts;
				VtemporalRes_n.push_back(w2*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / w / sqrt(dt)*sqrt(naJoints3) / Real2SfM, 2)); //in mm/s
				//VtemporalRes_n.push_back(w2 / nVertices*0.5*residuals[ii] * residuals[ii]), VtemporalRes.push_back(pow(residuals[ii] / sqrt(dt)*sqrt(nVertices * 3) / Real2SfM, 2)); //in mm/s
			}
		}
	}

	{
		Vfitting2DRes7.push_back(0), Vfitting2DRes7_n.push_back(0), Vfitting2DRes6.push_back(0), Vfitting2DRes6_n.push_back(0), Vfitting2DRes5.push_back(0), Vfitting2DRes5_n.push_back(0),
			Vfitting2DRes4.push_back(0), Vfitting2DRes4_n.push_back(0), Vfitting2DRes3.push_back(0), Vfitting2DRes3_n.push_back(0), Vfitting2DRes2.push_back(0), Vfitting2DRes2_n.push_back(0), Vfitting2DRes1.push_back(0), Vfitting2DRes1_n.push_back(0);
		Map< VectorXd > eVshapePriorCeres(&VshapePriorCeres[0], VshapePriorCeres.size());
		Map< VectorXd > eVposePriorCeres(&VposePriorCeres[0], VposePriorCeres.size());
		Map< VectorXd > eVfitting2DRes1(&Vfitting2DRes1[0], Vfitting2DRes1.size());
		Map< VectorXd > eVfitting2DRes1_n(&Vfitting2DRes1_n[0], Vfitting2DRes1_n.size());
		Map< VectorXd > eVfitting2DRes2(&Vfitting2DRes2[0], Vfitting2DRes2.size());
		Map< VectorXd > eVfitting2DRes2_n(&Vfitting2DRes2_n[0], Vfitting2DRes2_n.size());
		Map< VectorXd > eVfitting2DRes3(&Vfitting2DRes3[0], Vfitting2DRes3.size());
		Map< VectorXd > eVfitting2DRes3_n(&Vfitting2DRes3_n[0], Vfitting2DRes3_n.size());
		Map< VectorXd > eVfitting2DRes4_n(&Vfitting2DRes4_n[0], Vfitting2DRes4_n.size());
		Map< VectorXd > eVfitting2DRes5_n(&Vfitting2DRes5_n[0], Vfitting2DRes5_n.size());
		Map< VectorXd > eVfitting2DRes6_n(&Vfitting2DRes6_n[0], Vfitting2DRes6_n.size());
		Map< VectorXd > eVfitting2DRes7_n(&Vfitting2DRes7_n[0], Vfitting2DRes7_n.size());

		double sos_VshapePriorCeres = eVshapePriorCeres.sum(), sos_VposePriorCeres = eVposePriorCeres.sum(),
			sos_Vfitting2DRes1 = eVfitting2DRes1.sum(), rmse_Vfitting2DRes1 = sqrt(sos_Vfitting2DRes1 / Vfitting2DRes1.size()),
			sos_Vfitting2DRes1_n = eVfitting2DRes1_n.sum(), rmse_Vfitting2DRes1_n = sqrt(sos_Vfitting2DRes1_n / Vfitting2DRes1_n.size()),
			sos_Vfitting2DRes2 = eVfitting2DRes2.sum(), rmse_Vfitting2DRes2 = sqrt(sos_Vfitting2DRes2 / Vfitting2DRes2.size()),
			sos_Vfitting2DRes2_n = eVfitting2DRes2_n.sum(), rmse_Vfitting2DRes2_n = sqrt(sos_Vfitting2DRes2_n / Vfitting2DRes2_n.size()),
			sos_Vfitting2DRes3 = eVfitting2DRes3.sum(), rmse_Vfitting2DRes3 = sqrt(sos_Vfitting2DRes3 / Vfitting2DRes3.size()),
			sos_Vfitting2DRes3_n = eVfitting2DRes3_n.sum(), rmse_Vfitting2DRes3_n = sqrt(sos_Vfitting2DRes3_n / Vfitting2DRes3_n.size());
		double sos_Vfitting2DRes4_n = eVfitting2DRes4_n.sum(), rmse_Vfitting2DRes4_n = sqrt(sos_Vfitting2DRes4_n / Vfitting2DRes4_n.size()),
			mu4 = MeanArray(Vfitting2DRes4), stdev4 = sqrt(VarianceArray(Vfitting2DRes4, mu4));
		double sos_Vfitting2DRes5_n = eVfitting2DRes5_n.sum(), rmse_Vfitting2DRes5_n = sqrt(sos_Vfitting2DRes5_n / Vfitting2DRes5_n.size()),
			mu5 = MeanArray(Vfitting2DRes5), stdev5 = sqrt(VarianceArray(Vfitting2DRes5, mu5));
		double sos_Vfitting2DRes6_n = eVfitting2DRes6_n.sum(), rmse_Vfitting2DRes6_n = sqrt(sos_Vfitting2DRes6_n / Vfitting2DRes6_n.size()),
			mu6 = MeanArray(Vfitting2DRes6), stdev6 = sqrt(VarianceArray(Vfitting2DRes6, mu6));
		double sos_Vfitting2DRes7_n = eVfitting2DRes7_n.sum(), rmse_Vfitting2DRes7_n = sqrt(sos_Vfitting2DRes7_n / Vfitting2DRes7_n.size()),
			mu7 = MeanArray(Vfitting2DRes7), stdev7 = sqrt(VarianceArray(Vfitting2DRes7, mu7));

		if (nInstances > 1)
		{
			Map< VectorXd > eVtemporalRes_n(&VtemporalRes_n[0], VtemporalRes_n.size()), eVtemporalRes(&VtemporalRes[0], VtemporalRes.size());
			double sos_VtemporalRes_n = eVtemporalRes_n.sum(), sos_VtemporalRes = eVtemporalRes.sum(), rmse_VtemporalRes = sqrt(sos_VtemporalRes / VtemporalRes.size());
			printLOG("After Optim\nScale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("Temporal smoothess-->sum of square: %.4f, rmse: %.4f (mm/s)\n", sos_VtemporalRes_n, rmse_VtemporalRes);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("2D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			printLOG("3D-3D Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
			printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
		}
		else
		{
			printLOG("After Optim\nFinal scale: %.4f\nShapePrior-->sum of square: %.4f\nPosePrior-->sum of square: %.4f\n", mySMPL.scale, sos_VshapePriorCeres, sos_VposePriorCeres);
			printLOG("SMPL2OContour fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes1_n, rmse_Vfitting2DRes1);
			printLOG("OContour2SMPL fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes2_n, rmse_Vfitting2DRes2);
			printLOG("Sil fitting-->sum of square: %.4f,  unnormalized rmse: %.4f\n", sos_Vfitting2DRes3_n, rmse_Vfitting2DRes3);
			printLOG("Keypoints fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes4_n, mu4, stdev4);
			if (nInstances == 1)
				printLOG("3D-3D fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes5_n, mu5, stdev5);
			if (Vfitting2DRes6_n.size() > 0 && Vfitting2DRes6_n[0] != 0)
				printLOG("3D-3D camera fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes6_n, mu6, stdev6);
			if (Vfitting2DRes7_n.size() > 0 && Vfitting2DRes7_n[0] != 0)
				printLOG("3D-3D ground fitting-->sum of square: %.4f, unnormalized mean %.4f, unnormalized std %.4f\n", sos_Vfitting2DRes7_n, mu7, stdev7);
			printLOG("Taking %2fs\n\n", omp_get_wtime() - startTime);
		}
	}

	delete[]hit;
	delete[]uvJ, delete[]uvV;
	delete[]vCidFid, delete[]ValidSegPixels;
	delete[]allPoseLandmark, delete[]residuals;

	delete[]All_VertexTypeAndVisibility, delete[]All_uv_ptr;
	delete[]All_V_ptr, delete[]All_dVdp_ptr, delete[]All_dVdc_ptr, delete[]All_dVds_ptr;
	delete[]All_aJsmpl_ptr, delete[]All_J_ptr, delete[]All_dJdt_ptr, delete[]All_dJdp_ptr, delete[]All_dJdc_ptr, delete[]All_dJds_ptr;

	return 0.0;
}
int FitSMPL_SingleView(char *Path, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, int Use2DFitting, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose, int nMaxPeople, int selectedPeopleId)
{
	char Fname[512];

	VideoData *VideoInfo = new VideoData[1];
	VideoInfo[0].VideoInfo = new CameraData[1];
	Point3d CamTimeInfo(1, 1, 0);

	vector<HumanSkeleton3D> vSkeletons;
	HumanSkeleton3D Body;
	FILE *fp = fopen("E:/uv.txt", "r");
	if (fp != NULL)
	{
		double dummy;  Point2d uv[25];
		for (int jid = 0; jid < 14; jid++)
			fscanf(fp, "%lf ", &uv[jid].x);

		for (int jid = 0; jid < 14; jid++)
			fscanf(fp, "%lf ", &uv[jid].y);

		for (int jid = 0; jid < 14; jid++)
			fscanf(fp, "%lf ", &dummy);

		for (int jid = 0; jid < 14; jid++)
		{
			Body.vViewID_rFid[jid].push_back(Point2i(0, 0));
			Body.vPt2D[jid].push_back(uv[jid]);
			Body.vConf[jid].push_back(1.0);
			Body.validJoints[jid] = 1;
		}
	}
	fclose(fp);
	Body.valid = true;
	vSkeletons.push_back(Body);

	DensePose *vDensePose = NULL;

	SMPLModel mySMPL;
	if (!ReadSMPLData("smpl", mySMPL))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	VideoInfo[0].VideoInfo[0].LensModel = 0;
	VideoInfo[0].VideoInfo[0].ShutterModel = 0;
	VideoInfo[0].VideoInfo[0].width = 1, VideoInfo[0].VideoInfo[0].height = 1;
	VideoInfo[0].VideoInfo[0].valid = 1;
	VideoInfo[0].VideoInfo[0].intrinsic[0] = 500, VideoInfo[0].VideoInfo[0].intrinsic[1] = 500, VideoInfo[0].VideoInfo[0].intrinsic[2] = 0, VideoInfo[0].VideoInfo[0].intrinsic[3] = 150, VideoInfo[0].VideoInfo[0].intrinsic[4] = 150;
	for (int ii = 0; ii < 6; ii++)
		VideoInfo[0].VideoInfo[0].rt[ii] = 0.0;
	GetRTFromrt(VideoInfo[0].VideoInfo[0]);
	GetKFromIntrinsic(VideoInfo[0].VideoInfo[0]);
	AssembleP(VideoInfo[0].VideoInfo[0]);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 18)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .1, JointWeight[19] = .1, JointWeight[20] = .1,//rFoot
		JointWeight[21] = .1, JointWeight[22] = .1, JointWeight[23] = .1;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int maxWidth = 1000, maxHeight = 1000;
	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	vector<int> vDP_Vid, vDP_pid;
	vector<Point2d> vDP_uv, vkpts;
	float *Para = new float[maxWidth*maxHeight];
	Mat IUV, INDS;
	for (int pi = 0; pi < vSkeletons.size(); pi++)
	{
		if (!vSkeletons[pi].valid)
			continue;

		printLOG("*****Person %d*******\n", pi);

		double smpl2sfmScale;
		SMPLParams ParaI;

		smpl2sfmScale = 1.0;
		ParaI.t.setZero();
		FILE *fp = fopen("E:/shape.txt", "r");
		for (int ii = 0; ii < 10; ii++)
			fscanf(fp, "%lf ", &ParaI.coeffs(ii));
		for (int ii = 0; ii < 72; ii++)
			fscanf(fp, "%lf ", &ParaI.pose(ii));
		fclose(fp);

		//ParaI.pose.setZero();

		ParaI.frame = 0;
		if (smpl2sfmScale < 1e-16 || !IsNumber(smpl2sfmScale))
			continue; //fail to init

		//since the code is written for multi-frame BA
		vector<SMPLParams> frame_params; frame_params.push_back(ParaI);
		vector<HumanSkeleton3D> frame_skeleton; frame_skeleton.push_back(vSkeletons[pi]);

		//init smpl
		mySMPL.t.setZero();
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);
		mySMPL.scale = smpl2sfmScale;

		Point2i fixedPoseFrames(-1, -1);
		vector<int> vSCams; vSCams.emplace_back(0);
		FitSMPL_Camera_Total(mySMPL, frame_params, frame_skeleton, vDensePose, &CamTimeInfo, VideoInfo, vSCams, ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, 100.0*smpl2sfmScale, skeletonPointFormat, fixedPoseFrames, pi, hasDensePose);
		printLOG("\n");

		sprintf(Fname, "E:/f.txt", Path);  fp = fopen(Fname, "w+");
		fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[0].t(0), frame_params[0].t(1), frame_params[0].t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fprintf(fp, "%f %f %f\n", frame_params[0].pose(ii, 0), frame_params[0].pose(ii, 1), frame_params[0].pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fprintf(fp, "%f ", mySMPL.coeffs(ii));
		fprintf(fp, "\n");
		fclose(fp);
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;
	delete[]Para;

	return 0;
}

int FitSMPL1FrameDriver_DensePose(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, vector<int> &vsCams, int startF, int stopF, int increF, int distortionCorrected, int sharedIntrinsic, int skeletonPointFormat, int Use2DFitting, double *weights, double *isigmas, double Real2SfM, double detectionThresh, bool hasDensePose, int selectedPeopleId = -1)
{
	printLOG("*****************FitSMPL1FrameDriver [%d->%d]*****************\n", startF, stopF);
	char Fname[512];
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int nSCams = vsCams.size(), nGCams = *max_element(vsCams.begin(), vsCams.end()) + 1;

	printLOG("Reading all camera poses\n");
	VideoData *VideoInfo = new VideoData[nSCams];
	for (int ii = 0; ii < (int)SelectedCamNames.size(); ii++)
	{
		for (int jj = 0; jj < (int)CamIdsPerSeq.size(); jj++)
		{
			int cid = ii * CamIdsPerSeq.size() + jj;
			printLOG("Reading Cam %d\n ", cid);
			ReadCamCalibInfo(Path, SelectedCamNames[ii], SeqId, CamIdsPerSeq[jj], VideoInfo[ii*CamIdsPerSeq.size() + jj], startF, stopF);
		}
	}

	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nGCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int temp;
		while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
			VideoInfo[selected].TimeOffset = temp;
		}
		fclose(fp);
	}
	else
		printLOG("Cannot load time stamp info. Assume no frame offsets!");

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nGCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	Point3d RestJoints[18];
	sprintf(Fname, "smpl/restJointsCOCO.txt"); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	for (int ii = 0; ii < 18; ii++)
		fscanf(fp, "%lf %lf %lf ", &RestJoints[ii].x, &RestJoints[ii].y, &RestJoints[ii].z);
	fclose(fp);

	vector<Point2f*> vMapXY;
	DensePose *vDensePose = new DensePose[nSCams];

	int nMaxPeople = 14;
	sprintf(Fname, "%s/FitBody", Path), makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d", Path, increF), makeDir(Fname);
	for (int frameID = startF; frameID <= stopF; frameID += increF)
	{
		printLOG("\nFrame %d\n", frameID);

		int rcid;  double u, v, s, avg_error;
		vector<HumanSkeleton3D> vSkeletons;
		if (selectedPeopleId == -1)
		{
			while (true)
			{
				int rfid, inlier, nValidJoints = 0;
				HumanSkeleton3D Body;
				sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, (int)vSkeletons.size(), frameID);
				if (IsFileExist(Fname) == 0)
					sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, (int)vSkeletons.size(), frameID);
				FILE *fp = fopen(Fname, "r");
				if (fp != NULL)
				{
					int  nvis;
					for (int jid = 0; jid < skeletonPointFormat; jid++)
					{
						fscanf(fp, "%lf %lf %lf %lf %d ", &Body.pt3d[jid].x, &Body.pt3d[jid].y, &Body.pt3d[jid].z, &avg_error, &nvis);
						for (int kk = 0; kk < nvis; kk++)
						{
							fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
							if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid || inlier == 0)
								continue;

							Point2d uv(u, v);
							if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
								LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
							else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
								FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);
							else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == 2)
								LensCorrectionPoint_KB3(uv, VideoInfo[rcid].VideoInfo[rfid].intrinsic, VideoInfo[rcid].VideoInfo[rfid].distortion);

							Body.vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
							Body.vPt2D[jid].push_back(uv);
							Body.vConf[jid].push_back(s);
						}

						if (abs(Body.pt3d[jid].x) + abs(Body.pt3d[jid].y) + abs(Body.pt3d[jid].z) > 1e-16)
							nValidJoints++;
					}
					fclose(fp);

					Body.valid = nValidJoints < skeletonPointFormat / 3 ? 0 : 1;
					Body.refFid = frameID;
					vSkeletons.push_back(Body);
				}
				else
					break;
				if (vSkeletons.size() > nMaxPeople)
					break;
			}
		}
		else
		{
			vSkeletons.resize(selectedPeopleId + 1);
			vSkeletons[selectedPeopleId].refFid = frameID;
			int rfid, inlier, nValidJoints = 0;
			sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, selectedPeopleId, frameID);
			if (IsFileExist(Fname) == 0)
				sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, selectedPeopleId, frameID);
			FILE *fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int  nvis;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
				{
					fscanf(fp, "%lf %lf %lf %lf %d ", &vSkeletons[selectedPeopleId].pt3d[jid].x, &vSkeletons[selectedPeopleId].pt3d[jid].y, &vSkeletons[selectedPeopleId].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
						if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid || inlier == 0)
							continue;

						Point2d uv(u, v);
						if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
						else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);
						else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == 2)
							LensCorrectionPoint_KB3(uv, VideoInfo[rcid].VideoInfo[rfid].intrinsic, VideoInfo[rcid].VideoInfo[rfid].distortion);

						vSkeletons[selectedPeopleId].vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
						vSkeletons[selectedPeopleId].vPt2D[jid].push_back(uv);
						vSkeletons[selectedPeopleId].vConf[jid].push_back(s);
					}

					if (IsValid3D(vSkeletons[selectedPeopleId].pt3d[jid]))
						nValidJoints++;
				}
				fclose(fp);

				vSkeletons[selectedPeopleId].valid = nValidJoints < skeletonPointFormat / 3 ? 0 : 1;
			}
			else
				break;
		}

		InitializeBodyPoseParameters1Frame(Path, frameID, increF, vSkeletons, RestJoints, skeletonPointFormat);
		FitSMPL1Frame(Path, smplMaster, frameID, increF, vSkeletons, vDensePose, VideoInfo, CamTimeInfo, vsCams, distortionCorrected, sharedIntrinsic, skeletonPointFormat, Use2DFitting, weights, isigmas, Real2SfM, hasDensePose, nMaxPeople, selectedPeopleId);
	}

	delete[]vDensePose;

	return 0;
}
int FitSMPLWindowDriver_DensePose(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, vector<int> &vsCams, int startF, int stopF, int winSize, int nOverlappingFrames, int increF, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, int syncMode, double *CostWeights, double *isigmas, double Real2SfM, double detectionThresh, bool hasDensePose, int selectedPeopleID = -1, int startChunkdId = 0)
{
	char Fname[512];
	int nChunks = (stopF - startF + 1) / winSize + 1;
	const int  nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int nSCams = vsCams.size(), nGCams = *max_element(vsCams.begin(), vsCams.end()) + 1;

	printLOG("Reading all camera poses\n");
	VideoData *VideoInfo = new VideoData[nSCams];
	for (int ii = 0; ii < (int)SelectedCamNames.size(); ii++)
	{
		for (int jj = 0; jj < (int)CamIdsPerSeq.size(); jj++)
		{
			int cid = ii * CamIdsPerSeq.size() + jj;
			printLOG("Reading Cam %d\n ", cid);
			ReadCamCalibInfo(Path, SelectedCamNames[ii], SeqId, CamIdsPerSeq[jj], VideoInfo[ii*CamIdsPerSeq.size() + jj], startF, stopF);
		}
	}

	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nGCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE * fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int temp;
		while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
			VideoInfo[selected].TimeOffset = temp;
		}
		fclose(fp);
	}
	else
		printLOG("Cannot load time stamp info. Assume no frame offsets!");

	SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	double u, v, s, avg_error;
	int  rcid, nPeople = 0;
	vector<HumanSkeleton3D *> vSkeleton;
	if (selectedPeopleID == -1)
	{
		printLOG("Reading all people 3D skeleton: ");
		while (true)
		{
			printLOG("%d..", nPeople);

			int nvalidFrames = 0;
			HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
			for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				int  nValidJoints = 0, temp = (refFid - startF) / increF;
				Skeletons[temp].refFid = refFid;
				sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, 1, nPeople, refFid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, nPeople, refFid);
					if (IsFileExist(Fname) == 0)
					{
						sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, nPeople, refFid);
						if (IsFileExist(Fname) == 0)
							continue;
					}
				}
				fp = fopen(Fname, "r");
				if (fp != NULL)
				{
					int  rfid, nvis, inlier;
					for (int jid = 0; jid < skeletonPointFormat; jid++)
					{
						fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
						for (int kk = 0; kk < nvis; kk++)
						{
							fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
							if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid || inlier == 0)
								continue;

							Point2d uv(u, v);
							if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
								LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
							else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
								FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);
							else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == 2)
								LensCorrectionPoint_KB3(uv, VideoInfo[rcid].VideoInfo[rfid].intrinsic, VideoInfo[rcid].VideoInfo[rfid].distortion);

							Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
							Skeletons[temp].vPt2D[jid].push_back(uv);
							Skeletons[temp].vConf[jid].push_back(s);
						}

						if (abs(Skeletons[temp].pt3d[jid].x) + abs(Skeletons[temp].pt3d[jid].y) + abs(Skeletons[temp].pt3d[jid].z) > 1e-16)
							Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
						else
							Skeletons[temp].validJoints[jid] = 0;
					}
					fclose(fp);

					if (nValidJoints < skeletonPointFormat / 3)
						Skeletons[temp].valid = 0;
					else
						Skeletons[temp].valid = 1;

					nvalidFrames++;
				}
			}
			if (nvalidFrames == 0)
			{
				printLOG("\n");
				break;
			}

			vSkeleton.push_back(Skeletons);
			nPeople++;
		}
	}
	else
	{
		printLOG("Reading 3D skeleton for %d:\n", selectedPeopleID);
		for (int pid = 0; pid < selectedPeopleID; pid++)
		{
			HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
			vSkeleton.push_back(Skeletons);
			nPeople++;
		}

		int nvalidFrames = 0;
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int  nValidJoints = 0, temp = (refFid - startF) / increF;
			Skeletons[temp].refFid = refFid;

			sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, 1, nPeople, refFid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, nPeople, refFid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, nPeople, refFid);
					if (IsFileExist(Fname) == 0)
						continue;
				}
			}
			fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int  rfid, nvis, inlier;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
				{
					fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
						if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[rfid].width - 1 || v>VideoInfo[rcid].VideoInfo[rfid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[rfid].valid || inlier == 0)
							continue;

						Point2d uv(u, v);
						if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
						else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion[0]);
						else if (distortionCorrected == 0 && VideoInfo[rcid].VideoInfo[rfid].LensModel == 2)
							LensCorrectionPoint_KB3(uv, VideoInfo[rcid].VideoInfo[rfid].intrinsic, VideoInfo[rcid].VideoInfo[rfid].distortion);

						Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(rcid, rfid));
						Skeletons[temp].vPt2D[jid].push_back(uv);
						Skeletons[temp].vConf[jid].push_back(s);
					}

					if (IsValid3D(Skeletons[temp].pt3d[jid]))
						Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
					else
						Skeletons[temp].validJoints[jid] = 0;
				}
				fclose(fp);

				if (nValidJoints < skeletonPointFormat / 3)
					Skeletons[temp].valid = 0;
				else
					Skeletons[temp].valid = 1;
				nvalidFrames++;
			}
		}
		if (nvalidFrames == 0)
			printLOG("\n");

		vSkeleton.push_back(Skeletons);
		nPeople++;
	}

	int nSyncInst = ((winSize + nOverlappingFrames) / increF + 1);
	DensePose *vDensePose = NULL; //should have enough mem for unsynced case

	//CostWeights[2] = CostWeights[2] / increF/increF;
	for (int pid = 0; pid < nPeople; pid++)
	{
		if (selectedPeopleID != -1 && pid != selectedPeopleID)
			continue;

		Point2i fixedPoseFrames(-1, -1);
		for (int chunkId = startChunkdId; chunkId < nChunks; chunkId++)
		{
			if (chunkId != startChunkdId)
				hasDensePose = false;
			if (syncMode == 1)
			{
				printLOG("*****************FitSMPLWindowDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
				FitSMPLWindow(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
			}
			else
			{
				printLOG("*****************FitSMPLWindowUnSyncDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
				FitSMPLUnSync(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
			}
			fixedPoseFrames.x = startF + winSize * (chunkId + 1), fixedPoseFrames.y = startF + winSize * (chunkId + 1) + nOverlappingFrames;
		}

		if (startChunkdId > 0)
		{
			for (int chunkId = startChunkdId - 1; chunkId > -1; chunkId--)
			{
				fixedPoseFrames.x = startF + winSize * (chunkId + 1), fixedPoseFrames.y = startF + winSize * (chunkId + 1) + nOverlappingFrames;
				if (syncMode == 1)
				{
					printLOG("*****************FitSMPLWindowDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
					FitSMPLWindow(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
				}
				else
				{
					printLOG("*****************FitSMPLWindowUnSyncDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
					FitSMPLUnSync(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
				}
			}
		}
	}

	delete[]vDensePose, delete[]VideoInfo;
	for (int ii = 0; ii < vSkeleton.size(); ii++)
		delete[]vSkeleton[ii];

	return 0;
}

int VisualizeProjectedSMPLBody3(char *Path, int nCams, int startF, int stopF, int increF, int maxPeople, double resizeFactor, int WriteVideo, int debug)
{
	const double TimeScale = 1000000.0;

	char Fname[512];
	sprintf(Fname, "%s/Vis/FitBody", Path), makeDir(Fname);
	if (WriteVideo == 0)
	{
		/*for (int pid = 0; pid < maxPeople; pid++)
		{
			sprintf(Fname, "%s/Vis/FitBody/%d", Path, pid), makeDir(Fname);
			for (int cid = 0; cid < nCams; cid++)
				sprintf(Fname, "%s/Vis/FitBody/%d/%d", Path, pid, cid), makeDir(Fname);
		}*/
		for (int cid = 0; cid < nCams; cid++)
			sprintf(Fname, "%s/Vis/FitBody/%d", Path, cid), makeDir(Fname);
	}

	static cv::Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 128, 255), Scalar(0, 255, 255), Scalar(0, 255, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(255, 255, 255) };
	vector<Point3i> vcolors;
	vcolors.push_back(Point3i(0, 0, 255)), vcolors.push_back(Point3i(0, 128, 255)), vcolors.push_back(Point3i(0, 255, 255)), vcolors.push_back(Point3i(0, 255, 0)),
		vcolors.push_back(Point3i(255, 128, 0)), vcolors.push_back(Point3i(255, 255, 0)), vcolors.push_back(Point3i(255, 0, 0)), vcolors.push_back(Point3i(255, 0, 255)), vcolors.push_back(Point3i(255, 255, 255));
	int selected;  double fps;

	omp_set_num_threads(omp_get_max_threads());

	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps;
			CamTimeInfo[selected].y = temp;
			CamTimeInfo[selected].z = 1.0;
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
			{
				CamTimeInfo[selected].x = 1.0 / fps;
				CamTimeInfo[selected].y = temp;
				CamTimeInfo[selected].z = 1.0;
			}
			fclose(fp);
		}
		else
		{
			double fps; int temp;
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				{
					CamTimeInfo[selected].x = 1.0 / fps;
					CamTimeInfo[selected].y = temp;
					CamTimeInfo[selected].z = 1.0;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	printLOG("Reading all people 3D skeleton: ");
	double u, v, s, avg_error;
	int cid, dummy, nPeople = 0, nJoints = 18;
	vector<HumanSkeleton3D *> vSkeletons;
	while (true)
	{
		printLOG("%d..", nPeople);

		int nvalidFrames = 0;
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int nvis, nValidJoints = 0, temp = (refFid - startF) / increF;
			sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, nPeople, refFid); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int rfid, pid, nvis, dummy; float fdummy;
				for (int jid = 0; jid < nJoints; jid++)
				{
					int temp = (refFid - startF) / increF;
					fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d", &cid, &rfid, &u, &v, &s, &dummy);
						//fscanf(fp, "%d %lf %lf %lf", &cid, &u, &v, &s);

						double ts = 1.0*refFid / CamTimeInfo[refCid].x;
						rfid = MyFtoI((ts - CamTimeInfo[cid].y / CamTimeInfo[refCid].x) * CamTimeInfo[cid].x);

						Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(cid, rfid));
						Skeletons[temp].vPt2D[jid].push_back(Point2d(u, v));
						Skeletons[temp].vConf[jid].push_back(s);
					}

					if (abs(Skeletons[temp].pt3d[jid].x) + abs(Skeletons[temp].pt3d[jid].y) + abs(Skeletons[temp].pt3d[jid].z) > 1e-16)
						Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
					else
						Skeletons[temp].validJoints[jid] = 0;
				}
				fclose(fp);

				if (nValidJoints < nJoints / 3)
					Skeletons[temp].valid = 0;
				else
					Skeletons[temp].valid = 1;

				nvalidFrames++;
			}
		}
		if (nvalidFrames == 0)
		{
			printLOG("\n");
			break;
		}

		vSkeletons.push_back(Skeletons);
		nPeople++;
	}

	const int nVertices = smpl::SMPLModel::nVertices, nShapeCoeffs = smpl::SMPLModel::nShapeCoeffs, nJointsSMPL = smpl::SMPLModel::nJoints, nJointsCOCO = 18;
	MatrixXdr outV(nVertices, 3);
	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	MatrixXdr *AllV = new MatrixXdr[maxPeople];
	for (int pi = 0; pi < maxPeople; pi++)
		AllV[pi].resize(nVertices, 3);// .resize(nVertices, 3);

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;

	smpl::SMPLModel smplMaster;
	Point3i f; vector<Point3i> faces;
	fp = fopen("smpl/faces.txt", "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %d %d ", &f.x, &f.y, &f.z) != EOF)
			faces.push_back(f);
		fclose(fp);
	}
	if (!ReadSMPLData("smpl", smplMaster))
		printLOG("Check smpl Path.\n");

	int *firstTime = new int[nCams];
	VideoWriter *writer = new VideoWriter[nCams];
	for (int ii = 0; ii < nCams; ii++)
		firstTime[ii] = 1;

	Mat img, rimg, mask, blend;
	smpl::SMPLParams params;
	vector<int> vpid(nPeople);
	vector<smpl::SMPLParams>Vparams(nPeople);
	Point2d joints2D[18];
	vector<Point2f> *Vuv = new vector<Point2f>[nPeople];
	Point3f *allVertices = new Point3f[nVertices*nPeople];
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		printLOG("%d..", refFid);
		if ((refFid - startF) % 30 == 29)
			printLOG("\n");

		int temp = (refFid - startF) / increF;

		for (int ii = 0; ii < nPeople; ii++)
			vpid[ii] = false;
		for (int pid = 0; pid < nPeople; pid++)
		{
			sprintf(Fname, "%s/FitBody/@%d/P/%d/%.2d_%.4d_%.1f.txt", Path, increF, pid, 0, refFid, TimeScale*refFid); //window based
			//sprintf(Fname, "%s/FitBody/ff_BodyParameters_%d_%.4d.txt", Path, pid, refFid);
			if (IsFileExist(Fname) == 0)
				continue;
			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%lf %lf %lf %lf ", &Vparams[pid].scale, &Vparams[pid].t(0), &Vparams[pid].t(1), &Vparams[pid].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &Vparams[pid].pose(ii, 0), &Vparams[pid].pose(ii, 1), &Vparams[pid].pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &Vparams[pid].coeffs(ii));
			//Vparams[pid].pose(15, 0) = 0.3, Vparams[pid].pose(15, 1) = 0, Vparams[pid].pose(15, 2) = 0;//up straight face
			fclose(fp);

			vpid[pid] = true;
		}
		int cnt = 0;
		for (int pid = 0; pid < nPeople; pid++)
			if (vpid[pid])
				cnt++;
		if (cnt == 0)
			continue;

#pragma omp parallel for schedule(dynamic,1)
		for (int pid = 0; pid < nPeople; pid++)
		{
			if (!vpid[pid])
				continue;

			reconstruct(smplMaster, Vparams[pid].coeffs.data(), Vparams[pid].pose.data(), AllV[pid].data());
			Map<VectorXd> V_vec(AllV[pid].data(), AllV[pid].size());
			V_vec = V_vec * Vparams[pid].scale + dVdt * Vparams[pid].t;

			for (int ii = 0; ii < nVertices; ii++)
				allVertices[ii + pid * nVertices] = Point3f(AllV[pid](ii, 0), AllV[pid](ii, 1), AllV[pid](ii, 2));
		}

		for (int cid = 0; cid < nCams; cid++)
		{
			double ts = 1.0*refFid / CamTimeInfo[refCid].x;
			int rfid = MyFtoI((ts - CamTimeInfo[cid].y / CamTimeInfo[refCid].x) * CamTimeInfo[cid].x);
			int width = VideoInfo[cid].VideoInfo[rfid].width, height = VideoInfo[cid].VideoInfo[rfid].height;

			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[rfid].valid != 1)
				continue;

			sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid, rfid); img = imread(Fname);
			if (img.empty() == 1)
			{
				sprintf(Fname, "%s/%d/%.4d.png", Path, cid, rfid); img = imread(Fname);
				if (img.empty() == 1)
					continue;
			}

			if (firstTime[cid] == 1 && WriteVideo)
			{
				firstTime[cid] = 0;
				CvSize size;
				size.width = (int)(resizeFactor*img.cols), size.height = (int)(resizeFactor*img.rows);
				sprintf(Fname, "%s/Vis/FitBody/%d_%d_%d.avi", Path, cid, startF, stopF), writer[cid].open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 30, size);
			}

			for (int pid = 0; pid < nPeople; pid++)
			{
				HumanSkeleton3D *Body0 = &vSkeletons[pid][temp];

				/*int nvalid = 0;
				for (int jid = 0; jid < nJoints; jid++)
				{
					joints2D[jid] = Point2d(0, 0);
					for (int id = 0; id < Body0[0].vViewID_rFid[jid].size(); id++)
					{
						if (Body0[0].vViewID_rFid[jid][id].x == cid && Body0[0].vViewID_rFid[jid][id].y == rfid)
						{
							joints2D[jid] = Body0[0].vPt2D[jid][id];
							nvalid++;
							break;
						}
					}
				}
				if (nvalid <= 4)
					continue; */
				bool visible = 0;
				int bottomID = -1; double bottomY = 0;
				for (int jid = 0; jid < nJoints; jid++)
				{
					joints2D[jid] = Point2d(0, 0);
					if (Body0[0].validJoints[jid] > 0)
					{
						Point3d xyz = Body0[0].pt3d[jid];
						if (camI[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
						{
							ProjectandDistort(xyz, &joints2D[jid], camI[rfid].P);
							if (joints2D[jid].x < -camI[rfid].width / 10 || joints2D[jid].x > 11 * camI[rfid].width / 10 || joints2D[jid].y < -camI[rfid].height / 10 || joints2D[jid].y > 11 * camI[rfid].height / 10)
								continue;

							if (camI[rfid].ShutterModel == GLOBAL_SHUTTER)
								ProjectandDistort(xyz, &joints2D[jid], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
							else if (camI[rfid].ShutterModel == ROLLING_SHUTTER)
								CayleyDistortionProjection(camI[rfid].intrinsic, camI[rfid].distortion, camI[rfid].rt, camI[rfid].wt, joints2D[jid], xyz, width, height);
						}
						else
						{
							FisheyeProjectandDistort(xyz, &joints2D[jid], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
							if (joints2D[jid].x < -camI[rfid].width / 10 || joints2D[jid].x > 11 * camI[rfid].width / 10 || joints2D[jid].y < -camI[rfid].height / 10 || joints2D[jid].y > 11 * camI[rfid].height / 10)
								continue;

							if (camI[rfid].ShutterModel == GLOBAL_SHUTTER)
								FisheyeProjectandDistort(xyz, &joints2D[jid], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
							else if (camI[rfid].ShutterModel == ROLLING_SHUTTER)
								CayleyFOVProjection2(camI[rfid].intrinsic, camI[rfid].distortion, camI[rfid].rt, camI[rfid].wt, joints2D[jid], xyz, width, height);
						}

						if (joints2D[jid].y > bottomY)
							bottomY = joints2D[jid].y, bottomID = jid;
					}
				}
				if (bottomID > 13 || bottomID < 8)
					continue;

				Draw2DCoCoJoints(img, joints2D, nJoints, 2, 1.0, &colors[pid % 8]);
				if (debug == 1)
				{
					//Draw2DCoCoJoints(img, joints2D, nJoints, 2, 1.0, &colors[pid % 8]);
					sprintf(Fname, "%s/Vis/FitBody/x.jpg", Path), imwrite(Fname, img);
				}

				for (int ii = 0; ii < nVertices; ii++)
				{
					Point2d uv;
					Point3d xyz(allVertices[nVertices*pid + ii].x, allVertices[nVertices*pid + ii].y, allVertices[nVertices*pid + ii].z);

					if (VideoInfo[cid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
						CayleyDistortionProjection(VideoInfo[cid].VideoInfo[rfid].intrinsic, VideoInfo[cid].VideoInfo[rfid].distortion, VideoInfo[cid].VideoInfo[rfid].rt, VideoInfo[cid].VideoInfo[rfid].wt, uv, xyz, width, height);
					else
						CayleyFOVProjection2(VideoInfo[cid].VideoInfo[rfid].intrinsic, VideoInfo[cid].VideoInfo[rfid].distortion, VideoInfo[cid].VideoInfo[rfid].rt, VideoInfo[cid].VideoInfo[rfid].wt, uv, xyz, width, height);

					int x = (int)(uv.x + 0.5), y = (int)(uv.y + 0.5);
					circle(img, Point2i(x, y), 1, colors[vpid[pid] % 9], 1);
				}

				/*Mat mask = Mat::zeros(img.rows, img.cols, CV_8U);
#pragma omp parallel for schedule(dynamic,1)
				for (int ii = 0; ii < faces.size(); ii++)
				{
					Point2d uv[3];
					int vid[3] = { faces[ii].x,  faces[ii].y,  faces[ii].z };
					for (int jj = 0; jj < 3; jj++)
					{
						Point3d xyz(allVertices[nVertices*pid + vid[jj]].x, allVertices[nVertices*pid + vid[jj]].y, allVertices[nVertices*pid + vid[jj]].z);
						if (camI[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
						{
							ProjectandDistort(xyz, &uv[jj], camI[rfid].P);
							if (uv[jj].x < -camI[rfid].width / 10 || uv[jj].x > 11 * camI[rfid].width / 10 || uv[jj].y < -camI[rfid].height / 10 || uv[jj].y > 11 * camI[rfid].height / 10)
								continue;

							if (camI[rfid].ShutterModel == GLOBAL_SHUTTER)
								ProjectandDistort(xyz, &uv[jj], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
							else if (camI[rfid].ShutterModel == ROLLING_SHUTTER)
								CayleyDistortionProjection(camI[rfid].intrinsic, camI[rfid].distortion, camI[rfid].rt, camI[rfid].wt, uv[jj], xyz, width, height);
						}
						else
						{
							FisheyeProjectandDistort(xyz, &uv[jj], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
							if (uv[jj].x < -camI[rfid].width / 10 || uv[jj].x > 11 * camI[rfid].width / 10 || uv[jj].y < -camI[rfid].height / 10 || uv[jj].y > 11 * camI[rfid].height / 10)
								continue;

							if (camI[rfid].ShutterModel == GLOBAL_SHUTTER)
								FisheyeProjectandDistort(xyz, &uv[jj], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
							else if (camI[rfid].ShutterModel == ROLLING_SHUTTER)
								CayleyFOVProjection2(camI[rfid].intrinsic, camI[rfid].distortion, camI[rfid].rt, camI[rfid].wt, uv[jj], xyz, width, height);
						}
					}
					int maxX = min((int)(max(max(max(0, uv[0].x), uv[1].x), uv[2].x)) + 1, width - 1);
					int minX = max((int)(min(min(min(width - 1, uv[0].x), uv[1].x), uv[2].x)) + 1, 0);
					int maxY = min((int)(max(max(max(0, uv[0].y), uv[1].y), uv[2].y)) + 1, height - 1);
					int minY = max((int)(min(min(min(height - 1, uv[0].y), uv[1].y), uv[2].y)) + 1, 0);
					for (int jj = minY; jj <= maxY; jj++)
						for (int ii = minX; ii < maxX; ii++)
							if (PointInTriangle(uv[0], uv[1], uv[2], Point2f(ii, jj)))
								mask.data[ii + jj*width] = 255;

					//if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[1].x >10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10)
						//cv::line(img, uv[0], uv[1], colors[pid], 1, CV_AA);
					//if (uv[1].x > 10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
						//cv::line(img, uv[0], uv[2], colors[pid], 1, CV_AA);
					//if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
						//cv::line(img, uv[1], uv[2], colors[pid], 1, CV_AA);
				}
				//sprintf(Fname, "%s/Vis/FitBody/%d/%d/%.4d.png", Path, pid, cid, rfid), imwrite(Fname, mask);*/

				if (debug == 1)
					sprintf(Fname, "%s/Vis/FitBody/x.jpg", Path), imwrite(Fname, img);
			}

			CvPoint text_origin = { width / 30, height / 30 };
			sprintf(Fname, "Real: %d Ref: %d", rfid, refFid), putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, colors[0], 3);
			resize(img, rimg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_AREA);
			if (WriteVideo)
				writer[cid] << rimg;
			else
				sprintf(Fname, "%s/Vis/FitBody/%d/%.4d_%.4d.jpg", Path, cid, rfid, refFid), imwrite(Fname, rimg);
		}
	}

	for (int ii = 0; ii < nCams; ii++)
		writer[ii].release();

	return 0;
}
int VisualizeProjectedSMPLBodyUnSync(char *Path, int nCams, int startF, int stopF, int increF, int PointFormat, int maxPeople, double resizeFactor, int WriteVideo, int debug)
{
	char Fname[512];
	sprintf(Fname, "%s/Vis/FitBody", Path), makeDir(Fname);
	sprintf(Fname, "%s/Vis/FitBody/US_Smoothing200_%d", Path, increF), makeDir(Fname);

	static cv::Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 128, 255), Scalar(0, 255, 255), Scalar(0, 255, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(255, 255, 255) };
	vector<Point3i> vcolors;
	vcolors.push_back(Point3i(0, 0, 255)), vcolors.push_back(Point3i(0, 128, 255)), vcolors.push_back(Point3i(0, 255, 255)), vcolors.push_back(Point3i(0, 255, 0)),
		vcolors.push_back(Point3i(255, 128, 0)), vcolors.push_back(Point3i(255, 255, 0)), vcolors.push_back(Point3i(255, 0, 0)), vcolors.push_back(Point3i(255, 0, 255)), vcolors.push_back(Point3i(255, 255, 255));
	int selected;  double fps;

	omp_set_num_threads(omp_get_max_threads());

	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps;
			CamTimeInfo[selected].y = temp;
			CamTimeInfo[selected].z = 1.0;
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			double temp;
			while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			{
				CamTimeInfo[selected].x = 1.0 / fps;
				CamTimeInfo[selected].y = temp;
				CamTimeInfo[selected].z = 1.0;
			}
			fclose(fp);
		}
		else
		{
			double fps; int temp;
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				{
					CamTimeInfo[selected].x = 1.0 / fps;
					CamTimeInfo[selected].y = temp;
					CamTimeInfo[selected].z = 1.0;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX, latest = 0;
	for (int ii = 0; ii < nCams; ii++)
	{
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;
		if (latest < CamTimeInfo[ii].y)
			latest = CamTimeInfo[ii].y;
	}

	int nPeople = maxPeople;

	const int nVertices = smpl::SMPLModel::nVertices, nShapeCoeffs = smpl::SMPLModel::nShapeCoeffs, nJointsSMPL = smpl::SMPLModel::nJoints;
	MatrixXdr outV(nVertices, 3);
	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	MatrixXdr *AllV = new MatrixXdr[maxPeople];
	for (int pi = 0; pi < maxPeople; pi++)
		AllV[pi].resize(nVertices, 3);

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;

	smpl::SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
		printLOG("Check smpl Path.\n");

	Mat img, rimg, mask, blend;
	smpl::SMPLParams params;
	vector<int> vpid(maxPeople);
	vector<smpl::SMPLParams>Vparams(maxPeople);
	Point2d joints2D[25];
	Point2d *vuv = new Point2d[nVertices];
	Point3f *allVertices = new Point3f[nVertices*maxPeople];

	bool hasFiles = false;
	vector<string> *vnames = new vector <string>[maxPeople];
	for (int peopleCount = 0; peopleCount < maxPeople; peopleCount++)
	{
		sprintf(Fname, "%s/FitBody/@%d/US_Smoothing200/%d", Path, increF, peopleCount);
#ifdef _WINDOWS
		vnames[peopleCount] = get_all_files_names_within_folder(std::string(Fname));
#endif
		if (vnames[peopleCount].size() == 0)
		{
			printLOG("Cannot load any tracking files for %d.\n", peopleCount);
			continue;
		}
		else
			hasFiles = true;
	}
	if (!hasFiles)
		return 1;

	for (int cid = 0; cid < nCams; cid++)
	{
		printLOG("\n\nCamera %d: ", cid);
		int minF = max(0, startF - (int)max(abs(latest), abs(earliest)) - 1), maxF = stopF + (int)max(abs(latest), abs(earliest)) + 1;
		vector <int> *vpids = new vector<int>[maxF];
		vector<string> *cid_strings = new vector<string>[maxF];
		for (int cnt = 0; cnt < maxPeople; cnt++)
		{
			for (int ii = 0; ii < vnames[cnt].size(); ii++)
			{
				std::string CidString = vnames[cnt][ii].substr(0, 2);
				std::string FidString = vnames[cnt][ii].substr(3, 4);

				if (cid == stoi(CidString))
					vpids[stoi(FidString)].push_back(cnt), cid_strings[stoi(FidString)].push_back(vnames[cnt][ii]);
			}
		}

		int firstTime = 1;
		VideoWriter writer;
		for (int rfid = minF; rfid < maxF; rfid++)
		{
			printLOG("%d..", rfid);

			CameraData *camI = VideoInfo[cid].VideoInfo;
			int width = camI[rfid].width, height = camI[rfid].height;
			if (!camI[rfid].valid)
				continue;

			int cnt = 0;
			for (int ii = 0; ii < nPeople; ii++)
				vpid[ii] = false;
			for (int jj = 0; jj < cid_strings[rfid].size(); jj++)
			{
				int pid = vpids[rfid][jj];
				sprintf(Fname, "%s/FitBody/@%d/US_Smoothing200/%d/%s", Path, increF, pid, cid_strings[rfid][jj].c_str()); //window based
				FILE *fp = fopen(Fname, "r");
				fscanf(fp, "%lf %lf %lf %lf ", &Vparams[pid].scale, &Vparams[pid].t(0), &Vparams[pid].t(1), &Vparams[pid].t(2));
				for (int ii = 0; ii < nJointsSMPL; ii++)
					fscanf(fp, "%lf %lf %lf ", &Vparams[pid].pose(ii, 0), &Vparams[pid].pose(ii, 1), &Vparams[pid].pose(ii, 2));
				for (int ii = 0; ii < nShapeCoeffs; ii++)
					fscanf(fp, "%lf ", &Vparams[pid].coeffs(ii));
				//Vparams[pid].pose(15, 0) = 0.3, Vparams[pid].pose(15, 1) = 0, Vparams[pid].pose(15, 2) = 0;//up straight face
				fclose(fp);

				cnt++;
				vpid[pid] = true;
			}
			if (cnt == 0)
				continue;

#pragma omp parallel for schedule(dynamic,1)
			for (int pid = 0; pid < maxPeople; pid++)
			{
				if (!vpid[pid])
					continue;

				reconstruct(smplMaster, Vparams[pid].coeffs.data(), Vparams[pid].pose.data(), AllV[pid].data());
				Map<VectorXd> V_vec(AllV[pid].data(), AllV[pid].size());
				V_vec = V_vec * Vparams[pid].scale + dVdt * Vparams[pid].t;

				for (int ii = 0; ii < nVertices; ii++)
					allVertices[ii + pid * nVertices] = Point3f(AllV[pid](ii, 0), AllV[pid](ii, 1), AllV[pid](ii, 2));
			}

			sprintf(Fname, "%s/%d/%.4d.png", Path, cid, rfid); img = imread(Fname);
			if (img.empty() == 1)
			{
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid, rfid); img = imread(Fname);
				if (img.empty() == 1)
					continue;
			}

			if (firstTime == 1 && WriteVideo)
			{
				firstTime = 0;
				CvSize size;
				size.width = (int)(resizeFactor*img.cols), size.height = (int)(resizeFactor*img.rows);
				sprintf(Fname, "%s/Vis/FitBody/US_Smoothing200_%d/%d_%d_%d_%d.avi", Path, increF, cid, startF, stopF), writer.open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 30, size);
			}

			for (int pid = 0; pid < nPeople; pid++)
			{
				if (!vpid[pid])
					continue;
#pragma omp parallel for schedule(dynamic,1)
				for (int ii = 0; ii < nVertices; ii++)
				{
					Point3d xyz(allVertices[nVertices*pid + ii].x, allVertices[nVertices*pid + ii].y, allVertices[nVertices*pid + ii].z);
					if (camI[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
					{
						ProjectandDistort(xyz, &vuv[ii], camI[rfid].P);
						if (vuv[ii].x < -camI[rfid].width / 10 || vuv[ii].x > 11 * camI[rfid].width / 10 || vuv[ii].y < -camI[rfid].height / 10 || vuv[ii].y > 11 * camI[rfid].height / 10)
							continue;

						if (camI[rfid].ShutterModel == GLOBAL_SHUTTER)
							ProjectandDistort(xyz, &vuv[ii], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
						else if (camI[rfid].ShutterModel == ROLLING_SHUTTER)
							CayleyDistortionProjection(camI[rfid].intrinsic, camI[rfid].distortion, camI[rfid].rt, camI[rfid].wt, vuv[ii], xyz, width, height);
					}
					else
					{
						FisheyeProjectandDistort(xyz, &vuv[ii], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
						if (vuv[ii].x < -camI[rfid].width / 10 || vuv[ii].x > 11 * camI[rfid].width / 10 || vuv[ii].y < -camI[rfid].height / 10 || vuv[ii].y > 11 * camI[rfid].height / 10)
							continue;

						if (camI[rfid].ShutterModel == GLOBAL_SHUTTER)
							FisheyeProjectandDistort(xyz, &vuv[ii], camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
						else if (camI[rfid].ShutterModel == ROLLING_SHUTTER)
							CayleyFOVProjection2(camI[rfid].intrinsic, camI[rfid].distortion, camI[rfid].rt, camI[rfid].wt, vuv[ii], xyz, width, height);
					}
				}

				for (int ii = 0; ii < smplMaster.vFaces.size(); ii++)
				{
					Point2d uv[3] = { vuv[smplMaster.vFaces[ii].x],vuv[smplMaster.vFaces[ii].y],vuv[smplMaster.vFaces[ii].z] };
					if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[1].x >10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10)
						cv::line(img, uv[0], uv[1], colors[pid], 1, CV_AA);
					if (uv[1].x > 10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
						cv::line(img, uv[0], uv[2], colors[pid], 1, CV_AA);
					if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
						cv::line(img, uv[1], uv[2], colors[pid], 1, CV_AA);
				}
				if (debug == 1)
					sprintf(Fname, "%s/Vis/FitBody/x.jpg", Path), imwrite(Fname, img);
			}

			CvPoint text_origin = { width / 30, height / 30 };
			sprintf(Fname, "Real: %d", rfid), putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, colors[0], 3);
			resize(img, rimg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_AREA);
			if (WriteVideo)
				writer << rimg;
			else
				sprintf(Fname, "%s/Vis/FitBody/US_Smoothing200_%d/%d_%.2d_%.4d.jpg", Path, increF, cid, rfid), imwrite(Fname, rimg);
		}

		writer.release();
	}

	delete[]vuv, delete[]allVertices;

	return 0;
}

int WindowSkeleton3DBundleAdjustment_BF_Sync(char *Path, int nCams, int startF, int stopF, int increF, int distortionCorrected, double detectionThresh, int LossType, double *Weights, double *iSigma, double real2SfM)
{
	printLOG("*****************WindowSkeleton3DBundleAdjustment*****************\n");
	//Weight = [ const limb length, symmetric skeleton, temporal]. It is  helpful for insightful weight setting if metric unit (mm) is used
	double sigma_i2D = iSigma[0], sigma_iL = iSigma[1] * real2SfM, sigma_iVel = iSigma[2] * real2SfM, sigma_iVel2 = iSigma[3] * real2SfM; //also convert physical sigma to sfm scale sigma

	char Fname[512];

	const double ialpha = 1.0 / 60.0; //1/fps

									  //For COCO 18-points 
	const int nJoints = 18, nLimbConnections = 17, nSymLimbConnectionID = 6;
	Point2i LimbConnectionID[nLimbConnections] = { Point2i(0, 1), Point2i(1, 2), Point2i(2, 3), Point2i(3, 4), Point2i(1, 5), Point2i(5, 6), Point2i(6, 7),
		Point2i(1, 8), Point2i(8, 9), Point2i(9, 10), Point2i(1, 11), Point2i(11, 12), Point2i(12, 13), Point2i(0, 14), Point2i(0, 15), Point2i(14, 16), Point2i(15, 17) };
	Vec4i SymLimbConnectionID[nSymLimbConnectionID] = { Vec4i(1, 2, 1, 5), Vec4i(2, 3, 5, 6), Vec4i(3, 4, 6, 7), Vec4i(1, 8, 1, 11), Vec4i(8, 9, 11, 12), Vec4i(9, 10, 12, 13) }; //no eyes, ears since they are unreliable

																																												  //For 25-point format
																																												  /*const int nJoints = 25, nLimbConnections = 32, nSymLimbConnectionID = 18;
																																												  Point2i LimbConnectionID[nLimbConnections] = { Point2i(0, 1), Point2i(1, 2), Point2i(2, 3), Point2i(3, 4), Point2i(1, 5), Point2i(5, 6), Point2i(6, 7),
																																												  Point2i(1, 8), Point2i(8,9),Point2i(9, 10),Point2i(10,11),Point2i(8,12),Point2i(12,13),Point2i(13,14),Point2i(0,15),Point2i(15,17),Point2i(0,16),Point2i(17,18), Point2i(1,9),Point2i(1,12),
																																												  Point2i(11,22),Point2i(11,23),Point2i(11,24),Point2i(22,23),Point2i(22,24),Point2i(23,24),
																																												  Point2i(14,19),Point2i(14,20),Point2i(14,21),Point2i(19,20),Point2i(19,21),Point2i(13,20) };
																																												  Vec4i SymLimbConnectionID[nSymLimbConnectionID] = { Vec4i(1, 2, 1, 5), Vec4i(2, 3, 5, 6), Vec4i(3, 4, 6, 7),
																																												  Vec4i(8,9,8,12), Vec4i(1, 9, 1, 12), Vec4i(9,10, 12, 13), Vec4i(10, 11, 13, 14), Vec4i(1, 9, 1, 12),
																																												  Vec4i(22,23,19,20), Vec4i(23,24,20,14), Vec4i(22,24,19,21), Vec4i(11, 22, 14, 19),Vec4i(11, 23, 14, 20),Vec4i(11, 24, 14, 21),
																																												  Vec4i(0,15, 0,16), Vec4i(15,17,16,18) ,Vec4i(0,17, 0,18), Vec4i(15,18,16,17) };*/

	double TimeStamp[100], vifps[100];
	for (int ii = 0; ii < nCams; ii++)
		TimeStamp[ii] = 0.0, vifps[ii] = 1.0;

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		for (int ii = 0; ii < nCams; ii++)
		{
			fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp);
			TimeStamp[selected] = temp, vifps[selected] = 1.0 / fps;
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			for (int ii = 0; ii < nCams; ii++)
			{
				fscanf(fp, "%d %lf %d ", &selected, &fps, &temp);
				TimeStamp[ii] = (double)temp, vifps[selected] = 1.0 / fps;
			}
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int temp;
				for (int ii = 0; ii < nCams; ii++)
				{
					fscanf(fp, "%d %lf %d ", &selected, &fps, &temp);
					TimeStamp[ii] = (double)temp, vifps[selected] = 1.0 / fps;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int *SkipFrameOffset = new int[nCams];
	for (int ii = 0; ii < nCams; ii++)
		SkipFrameOffset[ii] = 0;
	if (increF != 1)
	{
		int sfo;
		sprintf(Fname, "%s/SkipFrameOffset_%d.txt", Path, increF);	fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			while (fscanf(fp, "%d %d ", &selected, &sfo) != EOF)
				SkipFrameOffset[selected] = sfo;
			fclose(fp);
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > TimeStamp[ii])
			earliest = TimeStamp[ii], refCid = ii;

	printLOG("Reading all camera poses\n");
	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
	{
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;
		//InvalidateAbruptCameraPose(VideoInfo[cid], -1, -1, 0);
	}

	bool debug = false;
	double u, v, s, avg_error, residuals[3], rho[3];
	int cid, nPeople = 0;
	vector<HumanSkeleton3D *> vSkeletons;
	vector<double*> vLimbLength;
	while (true)
	{
		printLOG("reading Person #%d\n", nPeople);

		int nvalidFrames = 0;
		double *LimbLength = new double[nLimbConnections];
		vector<double> *vlimblength = new vector<double>[nLimbConnections];
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int nvis, nValidJoints = 0, tempFid = (refFid - startF) / increF;
			sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, nPeople, refFid); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				nvalidFrames++;
				int pid, nvis, rfid, inlier, dummy; float fdummy;
				for (int jid = 0; jid < nJoints; jid++)
				{
					fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[tempFid].pt3d[jid].x, &Skeletons[tempFid].pt3d[jid].y, &Skeletons[tempFid].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d ", &cid, &rfid, &u, &v, &s, &inlier);
						if (!VideoInfo[cid].VideoInfo[rfid].valid)
							continue;

						int localFid = (int)(vifps[refCid] / vifps[cid] * (refFid - TimeStamp[cid]) - SkipFrameOffset[cid] + 0.5);// == rfid
						Skeletons[tempFid].vViewID_rFid[jid].push_back(Point2i(cid, rfid));
						Skeletons[tempFid].vPt2D_[jid].push_back(Point2d(u, v));
						Skeletons[tempFid].vPt2D[jid].push_back(Point2d(u, v));
						Skeletons[tempFid].vConf[jid].push_back(s);
						Skeletons[tempFid].vInlier[jid].push_back(inlier);
						Skeletons[tempFid].valid = true;
					}

					if (IsValid3D(Skeletons[tempFid].pt3d[jid]))
						Skeletons[tempFid].validJoints[jid] = 1;
					else
						Skeletons[tempFid].validJoints[jid] = 0;
				}
				fclose(fp);

				for (int cid = 0; cid < nLimbConnections; cid++)
				{
					int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
					if (Skeletons[tempFid].validJoints[j0] > 0 && Skeletons[tempFid].validJoints[j1] > 0)
						vlimblength[cid].push_back(norm(Skeletons[tempFid].pt3d[j0] - Skeletons[tempFid].pt3d[j1]));
				}
			}
		}
		if (nvalidFrames == 0)
		{
			printLOG("\n");
			break;
		}

		for (int cid = 0; cid < nLimbConnections; cid++)
		{
			if (vlimblength[cid].size() == 0)
				LimbLength[cid] = 0.0;
			else
				LimbLength[cid] = MedianArray(vlimblength[cid]);
		}

		vLimbLength.push_back(LimbLength);
		vSkeletons.push_back(Skeletons);
		nPeople++;

		delete[]vlimblength;
	}

	vector<double> VreprojectionError, VUnNormedReprojectionErrorX, VUnNormedReprojectionErrorY, VconstLimbError, VsymLimbError, VtemporalError;
	struct offsets
	{
		int of[15];
	};
	vector<offsets> allConfigs;
	{
		offsets of;
		for (int ii = 0; ii < nCams; ii++)
			of.of[ii] = 0;
		allConfigs.push_back(of);
	}
	for (int off0 = -1; off0 <= 1; off0++)
	{
		for (int off1 = -1; off1 <= 1; off1++)
		{
			for (int off2 = -1; off2 <= 1; off2++)
			{
				for (int off3 = -1; off3 <= 1; off3++)
				{
					for (int off4 = -1; off4 <= 1; off4++)
					{
						for (int off5 = -1; off5 <= 1; off5++)
						{
							for (int off6 = -1; off6 <= 1; off6++)
							{
								for (int off7 = -1; off7 <= 1; off7++)
								{
									for (int off8 = -1; off8 <= 1; off8++)
									{
										for (int off9 = 0; off9 <= 0; off9++)
										{
											for (int off10 = -1; off10 <= 1; off10++)
											{
												for (int off11 = -1; off11 <= 1; off11++)
												{
													for (int off12 = -1; off12 <= 1; off12++)
													{
														for (int off13 = -1; off13 <= 1; off13++)
														{
															for (int off14 = -1; off14 <= 1; off14++)
															{
																offsets of;
																of.of[0] = off0;
																of.of[1] = off1;
																of.of[2] = off2;
																of.of[3] = off3;
																of.of[4] = off4;
																of.of[5] = off5;
																of.of[6] = off6;
																of.of[7] = off7;
																of.of[8] = off8;
																of.of[9] = off9;
																of.of[10] = off10;
																of.of[11] = off11;
																of.of[12] = off12;
																of.of[13] = off13;
																of.of[14] = off14;
																allConfigs.push_back(of);
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	int TimeStamp_bk[15];
	for (int ii = 0; ii < 15; ii++)
		TimeStamp_bk[ii] = TimeStamp[ii];

	int bestConfigId = 0;
	double bestError = 9e9;

	for (int configId = 0; configId < allConfigs.size(); configId++)
	{
		double totalError = 0;
		for (int ii = 0; ii < nCams; ii++)
			TimeStamp[ii] = TimeStamp_bk[ii] + allConfigs[configId].of[ii];
		printLOG("Config %d/%d: ", configId, (int)allConfigs.size());
		for (int ii = 0; ii < nCams; ii++)
			printLOG("%d ", allConfigs[configId].of[ii]);
		printLOG("\n");

		ceres::LossFunction *loss_funcion = 0;
		if (LossType == 1) //Huber
			loss_funcion = new ceres::HuberLoss(1.0);

		for (int personId = 0; personId < nPeople; personId++)
		{
			printLOG("recon Person #%d\n", personId);
			ceres::Problem problem;

			int nvalidFrames = 0;
			double *LimbLength = vLimbLength[personId];
			HumanSkeleton3D *Skeletons = vSkeletons[personId];
			for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				int nvis, nValidJoints = 0, tempFid = (refFid - startF) / increF;
				if (vSkeletons[personId][tempFid].valid == 1)
				{
					int pid, nvis, rfid, inlier, dummy; float fdummy;
					for (int jid = 0; jid < nJoints; jid++)
					{
						for (int kk = 0; kk < Skeletons[tempFid].vViewID_rFid[jid].size(); kk++)
						{
							cid = Skeletons[tempFid].vViewID_rFid[jid][kk].x;
							int rfid = (int)(vifps[refCid] / vifps[cid] * (refFid - TimeStamp[cid]) - SkipFrameOffset[cid] + 0.5);// == rfid
							if (!VideoInfo[cid].VideoInfo[rfid].valid)
								continue;

							Skeletons[tempFid].vPt2D[jid][kk] = Skeletons[tempFid].vPt2D_[jid][kk];

							if (distortionCorrected == 0 && VideoInfo[cid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
								LensCorrectionPoint(&Skeletons[tempFid].vPt2D[jid][kk], VideoInfo[cid].VideoInfo[rfid].K, VideoInfo[cid].VideoInfo[rfid].distortion);
							else if (distortionCorrected == 0 && VideoInfo[cid].VideoInfo[rfid].LensModel == FISHEYE)
								FishEyeCorrectionPoint(&Skeletons[tempFid].vPt2D[jid][kk], VideoInfo[cid].VideoInfo[rfid].K, VideoInfo[cid].VideoInfo[rfid].distortion[0]);
						}
					}
					nvalidFrames++;
				}
			}

			int nvalidPoints1 = 0, nvalidPoints2 = 0, nvalidPoints3 = 0, nvalidPoints4 = 0;
			for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body0 = &Skeletons[tempFid];

				//reprojection error
				for (int jid = 0; jid < nJoints; jid++)
				{
					if (Body0[0].validJoints[jid] > 0)
					{
						for (int ii = 0; ii < (int)Body0[0].vConf[jid].size(); ii++)
						{
							int cid = Body0[0].vViewID_rFid[jid][ii].x, rfid = Body0[0].vViewID_rFid[jid][ii].y;
							CameraData *camI = VideoInfo[cid].VideoInfo;
							if (camI[rfid].valid != 1)
								continue;

							Point2d uv = Body0[0].vPt2D[jid][ii]; //has been corrected before
							if (Body0[0].vConf[jid][ii] < detectionThresh || uv.x < 1 || uv.y < 1 || Body0[0].vInlier[jid][ii] == 0)
								continue;
							nvalidPoints1++;
						}
					}
				}

				//constant limb length
				for (int cid = 0; cid < nLimbConnections; cid++)
				{
					int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
					if (Body0[0].validJoints[j0] > 0 && Body0[0].validJoints[j1] > 0)
						nvalidPoints2++;
				}

				//symmetry limb
				for (int cid = 0; cid < nSymLimbConnectionID; cid++)
				{
					int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
					if (Body0[0].validJoints[j0] > 0 && Body0[0].validJoints[j1] > 0 && Body0[0].validJoints[j0_] > 0 && Body0[0].validJoints[j1_] > 0)
						nvalidPoints3++;
				}
			}

			VreprojectionError.clear(), VconstLimbError.clear(), VsymLimbError.clear(), VtemporalError.clear();
			for (int refFid = startF; refFid <= stopF - increF; refFid += increF)
			{
				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body0 = &vSkeletons[personId][tempFid];
				HumanSkeleton3D *Body1 = &vSkeletons[personId][tempFid + 1];

				for (int jid = 0; jid < nJoints; jid++)
					if (Body0[0].validJoints[jid] > 0 && Body1[0].validJoints[jid] > 0) //temporal smoothing
						nvalidPoints4++;
			}

			if (debug)
				fp = fopen("C:/temp/before.txt", "w");
			for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				if (debug)
					fprintf(fp, "%d\n", refFid);

				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body0 = &vSkeletons[personId][tempFid];

				//reprojection error
				for (int jid = 0; jid < nJoints; jid++)
				{
					if (debug)
						fprintf(fp, "%d ", jid);

					if (Body0[0].validJoints[jid] > 0)
					{
						for (int ii = 0; ii < (int)Body0[0].vConf[jid].size(); ii++)
						{
							int cid = Body0[0].vViewID_rFid[jid][ii].x, rfid = Body0[0].vViewID_rFid[jid][ii].y;
							CameraData *camI = VideoInfo[cid].VideoInfo;
							if (camI[rfid].valid != 1)
								continue;

							Point2d uv = Body0[0].vPt2D[jid][ii]; //has been corrected before
							if (Body0[0].vConf[jid][ii] < detectionThresh || uv.x < 1 || uv.y < 1 || Body0[0].vInlier[jid][ii] == 0)
								continue;

							double w = Weights[0] * Body0[0].vConf[jid][ii] / (0.001 + nvalidPoints1);
							ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(camI[rfid].P, uv.x, uv.y, sigma_i2D);
							problem.AddResidualBlock(cost_function, ScaleLoss, &Body0[0].pt3d[jid].x);

							vector<double *> paras; paras.push_back(&Body0[0].pt3d[jid].x);
							cost_function->Evaluate(&paras[0], residuals, NULL);
							double loss = residuals[1] * residuals[1] + residuals[1] * residuals[1];
							ScaleLoss->Evaluate(loss, rho);

							VreprojectionError.push_back(w*0.5*rho[0]);
							VUnNormedReprojectionErrorX.push_back(residuals[0] / sigma_i2D), VUnNormedReprojectionErrorY.push_back(residuals[1] / sigma_i2D);
							if (debug)
								fprintf(fp, "%d %d %.2f %.2f ", cid, rfid, residuals[0] / sigma_i2D, residuals[1] / sigma_i2D);
						}
						if (debug)
							fprintf(fp, "\n");
					}
				}

				//constant limb length
				for (int cid = 0; cid < nLimbConnections; cid++)
				{
					int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
					if (Body0[0].validJoints[j0] > 0 && Body0[0].validJoints[j1] > 0)
					{
						double w = Weights[1] / (0.001 + nvalidPoints2);
						ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = ConstantLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);
						problem.AddResidualBlock(cost_function, ScaleLoss, &LimbLength[cid], &Body0[0].pt3d[j0].x, &Body0[0].pt3d[j1].x);

						vector<double *> paras; paras.push_back(&LimbLength[cid]), paras.push_back(&Body0[0].pt3d[j0].x), paras.push_back(&Body0[0].pt3d[j1].x);
						cost_function->Evaluate(&paras[0], residuals, NULL);
						ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
						VconstLimbError.push_back(rho[0]);
					}
				}

				//symmetry limb
				for (int cid = 0; cid < nSymLimbConnectionID; cid++)
				{
					int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
					if (Body0[0].validJoints[j0] > 0 && Body0[0].validJoints[j1] > 0 && Body0[0].validJoints[j0_] > 0 && Body0[0].validJoints[j1_] > 0)
					{
						double w = Weights[2] / (0001 + nvalidPoints3);
						ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
						if (j0 == j0_)
						{
							ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);
							problem.AddResidualBlock(cost_function, ScaleLoss, &Body0[0].pt3d[j0].x, &Body0[0].pt3d[j1].x, &Body0[0].pt3d[j1_].x);

							vector<double *> paras; paras.push_back(&Body0[0].pt3d[j0].x), paras.push_back(&Body0[0].pt3d[j1].x), paras.push_back(&Body0[0].pt3d[j1_].x);
							cost_function->Evaluate(&paras[0], residuals, NULL);
							ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
							VsymLimbError.push_back(rho[0]);
						}
						else
						{
							ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres::CreateAutoDiff(sigma_iL);
							problem.AddResidualBlock(cost_function, ScaleLoss, &Body0[0].pt3d[j0].x, &Body0[0].pt3d[j1].x, &Body0[0].pt3d[j0_].x, &Body0[0].pt3d[j1_].x);

							vector<double *> paras; paras.push_back(&Body0[0].pt3d[j0].x), paras.push_back(&Body0[0].pt3d[j1].x), paras.push_back(&Body0[0].pt3d[j0_].x), paras.push_back(&Body0[0].pt3d[j1_].x);
							cost_function->Evaluate(&paras[0], residuals, NULL);
							ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
							VsymLimbError.push_back(rho[0]);
						}
					}
				}
			}
			if (debug)
				fclose(fp);

			//temporal
			for (int refFid = startF; refFid <= stopF - increF; refFid += increF)
			{
				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body0 = &vSkeletons[personId][tempFid];
				HumanSkeleton3D *Body1 = &vSkeletons[personId][tempFid + 1];

				for (int jid = 0; jid < nJoints; jid++)
				{
					double actingSigma = sigma_iVel;
					if (nJoints == 18 && (jid == 4 || jid == 7 || jid == 10 || jid == 13)) //18 joint format
						actingSigma = sigma_iVel2;
					else if (nJoints == 18 && (jid == 4 || jid == 7 || jid == 11 || jid == 14 || jid == 19 || jid == 20 || jid == 21 || jid == 22 || jid == 23 || jid == 24))//25 joint format
						actingSigma = sigma_iVel2;

					if (Body0[0].validJoints[jid] > 0 && Body1[0].validJoints[jid] > 0) //temporal smoothing
					{
						int cid0 = Body0[0].vViewID_rFid[jid][0].x, cid1 = Body1[0].vViewID_rFid[jid][0].x;
						double w = Weights[3] / (0.001 + nvalidPoints4);
						ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres2::CreateAutoDiff(vifps[cid0] * refFid, vifps[cid1] * (refFid + 1), actingSigma);
						problem.AddResidualBlock(cost_function, ScaleLoss, &Body0[0].pt3d[jid].x, &Body1[0].pt3d[jid].x);

						vector<double *> paras; paras.push_back(&Body0[0].pt3d[jid].x), paras.push_back(&Body1[0].pt3d[jid].x);
						cost_function->Evaluate(&paras[0], residuals, NULL);
						ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
						VtemporalError.push_back(rho[0]);
					}
				}
			}

			double 	repro = sqrt(MeanArray(VreprojectionError)),
				unNormedReproX = MeanArray(VUnNormedReprojectionErrorX),
				unNormedReproY = MeanArray(VUnNormedReprojectionErrorY),
				stdUnNormedReproX = sqrt(VarianceArray(VUnNormedReprojectionErrorX, unNormedReproX)),
				stdUnNormedReproY = sqrt(VarianceArray(VUnNormedReprojectionErrorY, unNormedReproY)),
				cLimb = sqrt(MeanArray(VconstLimbError)) / sigma_iL * real2SfM,
				sSkele = sqrt(MeanArray(VsymLimbError)) / sigma_iL * real2SfM,
				motionCo = sqrt(MeanArray(VtemporalError)) / sigma_iVel * real2SfM;
			printLOG("Error before: [NormReprojection, MeanUnNormedRprojection, StdUnNormedRprojection, constLimbLength, symSkeleton, temporal coherent] = [%.3e, (%.3f %.3f) (%.3f %.3f) %.3f, %.3f, %.3f]\n", repro, unNormedReproX, unNormedReproY, stdUnNormedReproX, stdUnNormedReproY, cLimb, sSkele, motionCo);

			ceres::Solver::Options options;
			options.num_threads = omp_get_max_threads(); //jacobian eval
			options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
			options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
			//options.preconditioner_type = ceres::JACOBI;
			options.use_nonmonotonic_steps = false;
			options.max_num_iterations = 50;
			options.minimizer_progress_to_stdout = false;

			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << "\n";

			totalError += summary.final_cost;

			if (debug)
				fp = fopen("C:/temp/after.txt", "w");

			VreprojectionError.clear(), VconstLimbError.clear(), VsymLimbError.clear(), VtemporalError.clear();
			for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				if (debug)
					fprintf(fp, "%d\n", refFid);

				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body0 = &Skeletons[tempFid];

				//reprojection error
				for (int jid = 0; jid < nJoints; jid++)
				{
					if (debug)
						fprintf(fp, "%d ", jid);
					if (Body0[0].validJoints[jid] > 0)
					{
						for (int ii = 0; ii < (int)Body0[0].vConf[jid].size(); ii++)
						{
							int cid = Body0[0].vViewID_rFid[jid][ii].x, rfid = Body0[0].vViewID_rFid[jid][ii].y;
							CameraData *camI = VideoInfo[cid].VideoInfo;
							if (camI[rfid].valid != 1)
								continue;

							Point2d uv = Body0[0].vPt2D[jid][ii];
							if (Body0[0].vConf[jid][ii] < detectionThresh || uv.x < 1 || uv.y < 1 || Body0[0].vInlier[jid][ii] == 0)
								continue;

							double w = Weights[0] * Body0[0].vConf[jid][ii] / (0.001 + nvalidPoints1);
							ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
							ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(camI[rfid].P, uv.x, uv.y, sigma_i2D);

							vector<double *> paras; paras.push_back(&Body0[0].pt3d[jid].x);
							cost_function->Evaluate(&paras[0], residuals, NULL);
							double loss = residuals[1] * residuals[1] + residuals[1] * residuals[1];
							ScaleLoss->Evaluate(loss, rho);

							VreprojectionError.push_back(w*0.5*rho[0]);
							VUnNormedReprojectionErrorX.push_back(residuals[0] / sigma_i2D), VUnNormedReprojectionErrorY.push_back(residuals[1] / sigma_i2D);

							if (debug)
								fprintf(fp, "%d %d %.2f %.2f %.2f %.2f ", cid, rfid, uv.x, uv.y, residuals[0] / sigma_i2D, residuals[1] / sigma_i2D);
						}
						if (debug)
							fprintf(fp, "\n");

					}
				}

				//constant limb length
				for (int cid = 0; cid < nLimbConnections; cid++)
				{
					int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
					if (Body0[0].validJoints[j0] > 0 && Body0[0].validJoints[j1] > 0)
					{
						double w = Weights[1] / (0.001 + nvalidPoints2);
						ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = ConstantLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);

						vector<double *> paras; paras.push_back(&LimbLength[cid]), paras.push_back(&Body0[0].pt3d[j0].x), paras.push_back(&Body0[0].pt3d[j1].x);
						cost_function->Evaluate(&paras[0], residuals, NULL);
						ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
						VconstLimbError.push_back(rho[0]);
					}
				}

				//symmetry limb
				for (int cid = 0; cid < nSymLimbConnectionID; cid++)
				{
					int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
					if (Body0[0].validJoints[j0] > 0 && Body0[0].validJoints[j1] > 0 && Body0[0].validJoints[j0_] > 0 && Body0[0].validJoints[j1_] > 0)
					{
						double w = Weights[2] / (0001 + nvalidPoints3);
						ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
						if (j0 == j0_)
						{
							ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);

							vector<double *> paras; paras.push_back(&Body0[0].pt3d[j0].x), paras.push_back(&Body0[0].pt3d[j1].x), paras.push_back(&Body0[0].pt3d[j1_].x);
							cost_function->Evaluate(&paras[0], residuals, NULL);
							ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
							VsymLimbError.push_back(rho[0]);
						}
						else
						{
							ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres::CreateAutoDiff(sigma_iL);

							vector<double *> paras; paras.push_back(&Body0[0].pt3d[j0].x), paras.push_back(&Body0[0].pt3d[j1].x), paras.push_back(&Body0[0].pt3d[j0_].x), paras.push_back(&Body0[0].pt3d[j1_].x);
							cost_function->Evaluate(&paras[0], residuals, NULL);
							ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
							VsymLimbError.push_back(rho[0]);
						}
					}
				}
			}
			if (debug)
				fclose(fp);

			//temporal
			for (int refFid = startF; refFid <= stopF - increF; refFid += increF)
			{
				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body0 = &Skeletons[tempFid];
				HumanSkeleton3D *Body1 = &Skeletons[tempFid + 1];

				for (int jid = 0; jid < nJoints; jid++)
				{
					double actingSigma = sigma_iVel;
					if (nJoints == 18 && (jid == 4 || jid == 7 || jid == 10 || jid == 13)) //18 joint format
						actingSigma = sigma_iVel2;
					else if (nJoints == 18 && (jid == 4 || jid == 7 || jid == 11 || jid == 14 || jid == 19 || jid == 20 || jid == 21 || jid == 22 || jid == 23 || jid == 24)) //25 joint format
						actingSigma = sigma_iVel2;

					if (Body0[0].validJoints[jid] > 0 && Body1[0].validJoints[jid] > 0) //temporal smoothing
					{
						int cid0 = Body0[0].vViewID_rFid[jid][0].x, cid1 = Body1[0].vViewID_rFid[jid][0].x;
						double w = Weights[3] / (0.001 + nvalidPoints4);
						ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP);
						ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres2::CreateAutoDiff(vifps[cid0] * refFid, vifps[cid1] * (refFid + 1), actingSigma);

						vector<double *> paras; paras.push_back(&Body0[0].pt3d[jid].x), paras.push_back(&Body1[0].pt3d[jid].x);
						cost_function->Evaluate(&paras[0], residuals, NULL);
						ScaleLoss->Evaluate(residuals[0] * residuals[0], rho);
						VtemporalError.push_back(rho[0]);
					}
				}
			}

			repro = sqrt(MeanArray(VreprojectionError)),
				unNormedReproX = MeanArray(VUnNormedReprojectionErrorX),
				unNormedReproY = MeanArray(VUnNormedReprojectionErrorY),
				stdUnNormedReproX = sqrt(VarianceArray(VUnNormedReprojectionErrorX, unNormedReproX)),
				stdUnNormedReproY = sqrt(VarianceArray(VUnNormedReprojectionErrorY, unNormedReproY)),
				cLimb = sqrt(MeanArray(VconstLimbError)) / sigma_iL * real2SfM,
				sSkele = sqrt(MeanArray(VsymLimbError)) / sigma_iL * real2SfM,
				motionCo = sqrt(MeanArray(VtemporalError)) / sigma_iVel * real2SfM;
			printLOG("Error after: [NormReprojection, MeanUnNormedRprojection, StdUnNormedRprojection, constLimbLength, symSkeleton, temporal coherent] = [%.3e, (%.3f %.3f) (%.3f %.3f) %.3f, %.3f, %.3f]\n", repro, unNormedReproX, unNormedReproY, stdUnNormedReproX, stdUnNormedReproY, cLimb, sSkele, motionCo);

			/*for (int refFid = startF; refFid <= stopF; refFid += increF)
			{
				int tempFid = (refFid - startF) / increF;
				HumanSkeleton3D *Body = &Skeletons[tempFid];
				sprintf(Fname, "%s/People/@%d/%d/_f_%.4d.txt", Path, increF, personId, refFid); fp = fopen(Fname, "w");
				for (int jid = 0; jid < nJoints; jid++)
				{
					fprintf(fp, "%f %f %f %.2f %d\n", Body[0].pt3d[jid].x, Body[0].pt3d[jid].y, Body[0].pt3d[jid].z, sqrt(MeanArray(VreprojectionError)), (int)Body[0].vViewID_rFid[jid].size());
					for (int ii = 0; ii < Body[0].vViewID_rFid[jid].size(); ii++)
					{
						int cid = Body[0].vViewID_rFid[jid][ii].x, rfid = Body[0].vViewID_rFid[jid][ii].y;
						if (distortionCorrected == 0 && VideoInfo[cid].VideoInfo[rfid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensDistortionPoint(&Body[0].vPt2D[jid][ii], VideoInfo[cid].VideoInfo[rfid].K, VideoInfo[cid].VideoInfo[rfid].distortion);
						else if (distortionCorrected == 0 && VideoInfo[cid].VideoInfo[rfid].LensModel == FISHEYE)
							FishEyeDistortionPoint(&Body[0].vPt2D[jid][ii], VideoInfo[cid].VideoInfo[rfid].K, VideoInfo[cid].VideoInfo[rfid].distortion[0]);

						fprintf(fp, "%d %d %.3f %.3f %.2f %d ", cid, rfid, Body[0].vPt2D[jid][ii].x, Body[0].vPt2D[jid][ii].y, Body[0].vConf[jid][ii], Body[0].vInlier[jid][ii]);
					}
					fprintf(fp, "\n");
				}
				fclose(fp);
			}*/
		}
		printLOG("Config error: %.8e\n", totalError);

		if (bestError > totalError)
		{
			bestError = totalError;
			bestConfigId = configId;
		}
		printLOG("\nCurent bestConfig (%d, %.8e): ", bestConfigId, bestError);
		for (int ii = 0; ii < nCams; ii++)
			printLOG("%d ", allConfigs[bestConfigId].of[ii]);
		printLOG("\n\n");
	}

	for (int pid = 0; pid < nPeople; pid++)
		delete[]vSkeletons[pid], vLimbLength[pid];

	return 0;
}
int STRecalibrateCamerasFromSkeleton(char *Path, vector<int> &SelectedCams, vector<int> & CamsToRecalibrate, int startF, int stopF, int increF, int distortionCorrected, double detectionThresh)
{
	//set iterMax = 0 to strictly use the provided association
	const int nJoints = 18;
	int nCams = *max_element(SelectedCams.begin(), SelectedCams.end()) + 1;

	double TimeStamp[100], vfps[100];
	for (int ii = 0; ii < nCams; ii++)
		TimeStamp[ii] = 0.0, vfps[ii] = 1.0;

	int selected; double fps;
	char Fname[512];  sprintf(Fname, "%s/FMotionPriorSync.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		for (int ii = 0; ii < nCams; ii++)
		{
			fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp);
			TimeStamp[selected] = temp, vfps[selected] = fps;
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			for (int ii = 0; ii < nCams; ii++)
			{
				fscanf(fp, "%d %lf %d ", &selected, &fps, &temp);
				TimeStamp[ii] = (double)temp, vfps[selected] = fps;
			}
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int temp;
				for (int ii = 0; ii < nCams; ii++)
				{
					fscanf(fp, "%d %lf %d ", &selected, &fps, &temp);
					TimeStamp[ii] = (double)temp, vfps[selected] = fps;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int *SkipFrameOffset = new int[nCams];
	for (int ii = 0; ii < nCams; ii++)
		SkipFrameOffset[ii] = 0;
	if (increF != 1)
	{
		int sfo;
		sprintf(Fname, "%s/SkipFrameOffset_%d.txt", Path, increF);	fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			while (fscanf(fp, "%d %d ", &selected, &sfo) != EOF)
				SkipFrameOffset[selected] = sfo;
			fclose(fp);
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > TimeStamp[ii])
			earliest = TimeStamp[ii], refCid = ii;

	int nMaxPeople = 0;
	vector<vector<Point2i> > *MergedTrackletVec = new vector<vector<Point2i> >[nCams];
	for (auto cid : SelectedCams)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_%d_%d.txt", Path, cid, startF, stopF);
		if (!IsFileExist(Fname))
		{
			sprintf(Fname, "%s/%d/MergedTracklets_%d_%d.txt", Path, cid, startF, stopF);
			if (!IsFileExist(Fname))
				return 1;
		}
		std::string line, item;
		std::ifstream file(Fname);
		while (std::getline(file, line))
		{
			StringTrim(&line);//remove white space
			if (line.empty())
				break;
			std::stringstream line_stream(line);
			std::getline(line_stream, item, ' ');  //# pairs

			vector<Point2i> jointTrack;
			int fid, did;
			while (!line_stream.eof())
			{
				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				fid = atoi(item.c_str());
				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				did = atoi(item.c_str());
				jointTrack.push_back(Point2i(fid, did));
			}
			MergedTrackletVec[cid].push_back(jointTrack);
		}
		file.close();

		nMaxPeople = max(nMaxPeople, (int)MergedTrackletVec[cid].size());
	}
	nMaxPeople--;//last one is trash category	

	VideoData *VideoInfo = new VideoData[nCams];
	for (auto cid : SelectedCams)
	{
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;
		//InvalidateAbruptCameraPose(VideoInfo[cid], -1, -1, 0);
	}

	vector<float> *conf = new vector<float>[nCams];
	Point2d *tpts = new Point2d[nCams];
	vector<Point2d> *allPts = new vector<Point2d>[nCams];
	vector<Point2d> *allPts_org = new vector<Point2d>[nCams];

	vector<HumanSkeleton3D *> vSkeletons;
	for (int nPeople = 0; nPeople < nMaxPeople; nPeople++)
	{
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int nvis, nValidJoints = 0, temp = (refFid - startF) / increF;
			sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, nPeople, refFid); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int rfid, pid, nvis, dummy; float fdummy;
				double avg_error;
				for (int jid = 0; jid < nJoints; jid++)
				{
					int temp = (refFid - startF) / increF;
					fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
						fscanf(fp, "%d %d %f %f %f %d", &dummy, &dummy, &fdummy, &fdummy, &fdummy, &dummy);

					if (abs(Skeletons[temp].pt3d[jid].x) + abs(Skeletons[temp].pt3d[jid].y) + abs(Skeletons[temp].pt3d[jid].z) > 1e-16)
						Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
					else
						Skeletons[temp].validJoints[jid] = 0;
				}
				fclose(fp);

				if (nValidJoints < nJoints / 3)
					Skeletons[temp].valid = 0;
				else
					Skeletons[temp].valid = 1;
			}
		}
		vSkeletons.push_back(Skeletons);
	}

	vector<int> allCid;
	vector<float> vConf; vector<Point2f> vUV;
	for (auto cid : CamsToRecalibrate)
	{
		printLOG("Working on camera %d:\n", cid);

		ceres::Problem problem;
		ceres::LossFunction *loss_funcion = 0;
		loss_funcion = new ceres::HuberLoss(1.0);

		double u, v, s, residuals[2];
		vector<double>ReProjectionErrorX, ReProjectionErrorY;
		for (int refFid = startF; refFid <= stopF; refFid++)
		{
			allPts[cid].clear(), allPts_org[cid].clear(), conf[cid].clear();
			int realFid = (int)(vfps[refCid] / vfps[cid] * (refFid - TimeStamp[cid]) - SkipFrameOffset[cid] + 0.5);
			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[realFid].valid != 1)
				continue;

			vConf.clear(), vUV.clear();
			if (readOpenPoseJson(Path, cid, realFid, vUV, vConf) == 1)
			{
				for (int ii = 0; ii < vUV.size(); ii++)
				{
					if (vConf[ii] < detectionThresh)
						vUV.push_back(Point2d(0, 0)), vUV.push_back(Point2d(0, 0));
					else
					{
						Point2d uv(vUV[ii].x, vUV[ii].y);
						allPts_org[cid].push_back(uv);

						if (distortionCorrected == 0 && camI[realFid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion);
						else if (distortionCorrected == 0 && camI[realFid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion[0]);

						vUV.push_back(uv);
					}
				}
			}
			else
			{
				sprintf(Fname, "%s/MP/%d/%d.txt", Path, cid, realFid); FILE *fp = fopen(Fname, "r");
				if (fp == NULL)
					continue;
				while (fscanf(fp, "%lf %lf %lf ", &u, &v, &s) != EOF)
				{
					if (s < detectionThresh)
						vUV.push_back(Point2d(0, 0)), vConf.push_back(s);
					else
					{
						Point2d uv(u, v);
						if (distortionCorrected == 0 && camI[realFid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion);
						else if (distortionCorrected == 0 && camI[realFid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion[0]);
						vUV.push_back(uv), vConf.push_back(s);
					}
				}
				fclose(fp);
			}

			for (int pid = 0; pid < nMaxPeople; pid++)
			{
				HumanSkeleton3D *Skeletons = vSkeletons[pid];
				if (!Skeletons[refFid].valid)
					continue;

				//search for his id in the detection
				int detected = -1;
				for (int jj = 0; jj < (int)MergedTrackletVec[cid][pid].size(); jj++)
					if (MergedTrackletVec[cid][pid][jj].x == realFid) //found
						detected = MergedTrackletVec[cid][pid][jj].y;
				if (detected == -1)
					continue;

				for (int jid = 0; jid < nJoints; jid++)
				{
					if (vUV[detected*nJoints + jid].x < 1 || vUV[detected*nJoints + jid].y < 1 || vConf[detected*nJoints + jid] < 0.1)
						continue;

					ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(vUV[detected*nJoints + jid].x, vUV[detected*nJoints + jid].y, vConf[detected*nJoints + jid]);
					problem.AddResidualBlock(cost_function, loss_funcion, camI[realFid].intrinsic, camI[realFid].distortion, camI[realFid].rt, &Skeletons[refFid].pt3d[jid].x);

					vector<double *> paras; paras.push_back(camI[realFid].intrinsic), paras.push_back(camI[realFid].distortion), paras.push_back(camI[realFid].rt), paras.push_back(&Skeletons[refFid].pt3d[jid].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);

					ReProjectionErrorX.push_back(residuals[0] / vConf[detected*nJoints + jid]), ReProjectionErrorY.push_back(residuals[1] / vConf[detected*nJoints + jid]);

					problem.SetParameterBlockConstant(camI[realFid].intrinsic);
					problem.SetParameterBlockConstant(camI[realFid].distortion);
					problem.SetParameterBlockConstant(&Skeletons[refFid].pt3d[jid].x);
				}
			}
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
#pragma omp critical
			printLOG("Before BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
		}

		ceres::Solver::Options options;
		options.num_threads = 4;
		options.num_linear_solver_threads = 4;
		options.max_num_iterations = 300;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = false;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		printLOG("%s\n", summary.BriefReport().c_str());

		ReProjectionErrorX.clear(), ReProjectionErrorY.clear();
		for (int refFid = startF; refFid <= stopF; refFid++)
		{
			allPts[cid].clear(), allPts_org[cid].clear(), conf[cid].clear();
			int realFid = (int)(vfps[refCid] / vfps[cid] * (refFid - TimeStamp[cid]) - SkipFrameOffset[cid] + 0.5);
			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[realFid].valid != 1)
				continue;

			vConf.clear(), vUV.clear();
			if (readOpenPoseJson(Path, cid, realFid, vUV, vConf) == 1)
			{
				for (int ii = 0; ii < vUV.size(); ii++)
				{
					if (vConf[ii] < detectionThresh)
						vUV.push_back(Point2d(0, 0)), vUV.push_back(Point2d(0, 0));
					else
					{
						Point2d uv(vUV[ii].x, vUV[ii].y);
						allPts_org[cid].push_back(uv);

						if (distortionCorrected == 0 && camI[realFid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion);
						else if (distortionCorrected == 0 && camI[realFid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion[0]);

						vUV.push_back(uv);
					}
				}
			}
			else
			{
				sprintf(Fname, "%s/MP/%d/%d.txt", Path, cid, realFid); FILE *fp = fopen(Fname, "r");
				if (fp == NULL)
					continue;
				while (fscanf(fp, "%lf %lf %lf ", &u, &v, &s) != EOF)
				{
					if (s < detectionThresh)
						vUV.push_back(Point2d(0, 0)), vConf.push_back(s);
					else
					{
						Point2d uv(u, v);
						if (distortionCorrected == 0 && camI[realFid].LensModel == RADIAL_TANGENTIAL_PRISM)
							LensCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion);
						else if (distortionCorrected == 0 && camI[realFid].LensModel == FISHEYE)
							FishEyeCorrectionPoint(&uv, camI[realFid].K, camI[realFid].distortion[0]);
						vUV.push_back(uv), vConf.push_back(s);
					}
				}
				fclose(fp);
			}

			for (int pid = 0; pid < nMaxPeople; pid++)
			{
				HumanSkeleton3D *Skeletons = vSkeletons[pid];
				if (!Skeletons[refFid].valid)
					continue;

				//search for his id in the detection
				int detected = -1;
				for (int jj = 0; jj < (int)MergedTrackletVec[cid][pid].size(); jj++)
					if (MergedTrackletVec[cid][pid][jj].x == realFid) //found
						detected = MergedTrackletVec[cid][pid][jj].y;
				if (detected == -1)
					continue;

				for (int jid = 0; jid < nJoints; jid++)
				{
					if (vUV[detected*nJoints + jid].x < 1 || vUV[detected*nJoints + jid].y < 1 || vConf[detected*nJoints + jid] < 0.1)
						continue;

					ceres::CostFunction* cost_function = PinholeDistortionReprojectionError::Create(vUV[detected*nJoints + jid].x, vUV[detected*nJoints + jid].y, vConf[detected*nJoints + jid]);

					vector<double *> paras; paras.push_back(camI[realFid].intrinsic), paras.push_back(camI[realFid].distortion), paras.push_back(camI[realFid].rt), paras.push_back(&Skeletons[refFid].pt3d[jid].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);

					ReProjectionErrorX.push_back(residuals[0] / vConf[detected*nJoints + jid]), ReProjectionErrorY.push_back(residuals[1] / vConf[detected*nJoints + jid]);
				}
			}
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
#pragma omp critical
			printLOG("After BA: Min: (%.2f, %.2f) Max: (%.2f,%.2f) Mean: (%.2f,%.2f) Std: (%.2f,%.2f)\n", miniX, miniY, maxiX, maxiY, avgX, avgY, stdX, stdY);
		}

		sprintf(Fname, "%s/vCamPoseX_%.4d.txt", Path, cid); fp = fopen(Fname, "w+");
		for (int refFid = startF; refFid <= stopF; refFid++)
		{
			int realFid = (int)(vfps[refCid] / vfps[cid] * (refFid - TimeStamp[cid]) - SkipFrameOffset[cid] + 0.5);
			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[realFid].valid != 1)
				continue;

			fprintf(fp, "%d ", realFid);
			for (int jj = 0; jj < 6; jj++)
				fprintf(fp, "%.16f ", camI[realFid].rt[jj]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	delete[] VideoInfo, SkipFrameOffset, delete[]tpts, delete[]allPts, delete[]allPts_org;
	for (int ii = 0; ii < nMaxPeople; ii++)
		delete[]vSkeletons[ii];

	return 0;
}

float Area2DTriangle(const Point2f &a, const Point2f &b, const Point2f &c)
{
	return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}
vector<int> RasterizeMeshVisibilty(Point3d *vUVZ, int width, int height, int nVertices, vector<Point3i> &vfaces, Point3i *hit, float *zbuffer, bool *validFace = NULL, int *hitFace = NULL)
{
	bool memCreated = false;
	if (hit == NULL)
	{
		memCreated = true;
		hit = new Point3i[width*height];
		zbuffer = new float[width*height];
	}
	for (int ii = 0; ii < width*height; ii++)
		hit[ii] = Point3i(-1, -1, -1);
	for (int ii = 0; ii < width*height; ii++)
		zbuffer[ii] = FLT_MAX;
	if (hitFace != NULL)
		for (int ii = 0; ii < width*height; ii++)
			hitFace[ii] = -1;

	for (int fid = 0; fid < vfaces.size(); fid++)
	{
		int vid1 = vfaces[fid].x, vid2 = vfaces[fid].y, vid3 = vfaces[fid].z;
		Point3f uv1 = (Point3f)vUVZ[vid1], uv2 = (Point3f)vUVZ[vid2], uv3 = (Point3f)vUVZ[vid3];
		if (uv1.x < 1 || uv2.x < 1 || uv3.x < 1)
			continue;
		int maxX = min((int)(max(max(max(0, uv1.x), uv2.x), uv3.x)) + 1, width - 1);
		int minX = max((int)(min(min(min(width - 1, uv1.x), uv2.x), uv3.x)) + 1, 0);
		int maxY = min((int)(max(max(max(0, uv1.y), uv2.y), uv3.y)) + 1, height - 1);
		int minY = max((int)(min(min(min(height - 1, uv1.y), uv2.y), uv3.y)) + 1, 0);
		float area = Area2DTriangle(Point2f(uv1.x, uv1.y), Point2f(uv2.x, uv2.y), Point2f(uv3.x, uv3.y));
		float iz1 = 1.f / uv1.z, iz2 = 1.f / uv2.z, iz3 = 1.f / uv3.z;

		for (int jj = minY; jj <= maxY; jj++)
		{
			for (int ii = minX; ii < maxX; ii++)
			{
				float w0 = Area2DTriangle(Point2f(uv2.x, uv2.y), Point2f(uv3.x, uv3.y), Point2f(ii, jj));
				float w1 = Area2DTriangle(Point2f(uv3.x, uv3.y), Point2f(uv1.x, uv1.y), Point2f(ii, jj));
				float w2 = Area2DTriangle(Point2f(uv1.x, uv1.y), Point2f(uv2.x, uv2.y), Point2f(ii, jj));
				if (w0 >= 0 && w1 >= 0 && w2 >= 0)  //inside triangle
				{
					w0 /= area, w1 /= area, w2 /= area;
					float currentDepth = 1.f / (iz1* w0 + iz2 * w1 + iz3 * w2);
					int id = ii + jj * width;
					if (currentDepth < zbuffer[id])
					{
						hit[id] = Point3i(vid1, vid2, vid3), zbuffer[id] = currentDepth;
						if (hitFace != NULL)
							hitFace[id] = fid;
					}
				}
			}
		}
	}

	vector<int> visbileVertex;
	for (int ii = 0; ii < width*height; ii++)
		if (zbuffer[ii] < FLT_MAX)
			visbileVertex.push_back(hit[ii].x), visbileVertex.push_back(hit[ii].y), visbileVertex.push_back(hit[ii].z);

	sort(visbileVertex.begin(), visbileVertex.end());
	std::vector<int>::iterator it = unique(visbileVertex.begin(), visbileVertex.end());
	visbileVertex.resize(std::distance(visbileVertex.begin(), it));

	if (validFace != NULL)
	{
#pragma omp parallel for schedule(dynamic,1)
		for (int fid = 0; fid < (int)vfaces.size(); fid++)
		{
			validFace[fid] = false;
			bool found = false;
			for (int ii = 0; ii < visbileVertex.size() && !found; ii++)
				if (vfaces[fid].x == visbileVertex[ii])
					found = true;
			if (!found)
				continue;

			found = false;
			for (int ii = 0; ii < visbileVertex.size() && !found; ii++)
				if (vfaces[fid].y == visbileVertex[ii])
					found = true;
			if (!found)
				continue;

			found = false;
			for (int ii = 0; ii < visbileVertex.size() && !found; ii++)
				if (vfaces[fid].z == visbileVertex[ii])
					found = true;
			if (!found)
				continue;

			validFace[fid] = true;
		}
	}

	//WriteGridBinary("C:/temp/z.dat", zbuffer, width, height);

	if (memCreated)
		delete[]hit, delete[]zbuffer;

	return visbileVertex;
}
void ProjectandDistort(Point3d WC, Point3d *pts, double *P, double *K = NULL, double *distortion = NULL, int nviews = 1)
{
	int ii;
	double num1, num2, denum;

	for (ii = 0; ii < nviews; ii++)
	{
		num1 = P[ii * 12 + 0] * WC.x + P[ii * 12 + 1] * WC.y + P[ii * 12 + 2] * WC.z + P[ii * 12 + 3];
		num2 = P[ii * 12 + 4] * WC.x + P[ii * 12 + 5] * WC.y + P[ii * 12 + 6] * WC.z + P[ii * 12 + 7];
		denum = P[ii * 12 + 8] * WC.x + P[ii * 12 + 9] * WC.y + P[ii * 12 + 10] * WC.z + P[ii * 12 + 11];

		pts[ii].x = num1 / denum, pts[ii].y = num2 / denum, pts[ii].z = denum;
		if (K != NULL)
		{
			Point2d uv(pts[ii].x, pts[ii].y);
			LensDistortionPoint(&uv, K + ii * 9, distortion + ii * 7);
			pts[ii].x = uv.x, pts[ii].y = uv.y;
		}
	}

	return;
}
struct Point3u
{
	Point3u(uchar r, uchar g, uchar b) :x(r), y(g), z(b) {}
	uchar x, y, z;
};
Point3u BilinearInterp(Mat &img, int width, int height, double x, double y)
{
	if (x<0 || x>width - 2 || y<0 || y>height - 2)
		return Point3u(0, 0, 0);

	int xiD = (int)(x), yiD = (int)(y);
	int xiU = xiD + 1, yiU = yiD + 1;

	int w3 = width * 3;
	double res[3];
	for (int ii = 0; ii < 3; ii++)
	{
		double f00 = (double)(int)img.data[xiD * 3 + yiD * w3 + ii];
		double f01 = (double)(int)img.data[xiU * 3 + yiD * w3 + ii];
		double f10 = (double)(int)img.data[xiD * 3 + yiU * w3 + ii];
		double  f11 = (double)(int)img.data[xiU * 3 + yiU * w3 + ii];
		res[ii] = (f01 - f00)*(x - xiD) + (f10 - f00)*(y - yiD) + (f11 - f01 - f10 + f00)*(x - xiD)*(y - yiD) + f00;
	}

	return Point3u(res[0], res[1], res[2]);
}
int SMPLBodyTexGen(char *Path, int nCams, int startF, int stopF, int increF, int maxPeople, int debug)
{
	double TimeScale = 1000000.0;

	char Fname[512];

	static cv::Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 128, 255), Scalar(0, 255, 255), Scalar(0, 255, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(255, 255, 255) };
	vector<Point3i> vcolors;
	vcolors.push_back(Point3i(0, 0, 255)), vcolors.push_back(Point3i(0, 128, 255)), vcolors.push_back(Point3i(0, 255, 255)), vcolors.push_back(Point3i(0, 255, 0)),
		vcolors.push_back(Point3i(255, 128, 0)), vcolors.push_back(Point3i(255, 255, 0)), vcolors.push_back(Point3i(255, 0, 0)), vcolors.push_back(Point3i(255, 0, 255)), vcolors.push_back(Point3i(255, 255, 255));
	int selected;  double fps;

	omp_set_num_threads(omp_get_max_threads());

	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps;
			CamTimeInfo[selected].y = temp;
			CamTimeInfo[selected].z = 1.0;
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			double temp;
			while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			{
				CamTimeInfo[selected].x = 1.0 / fps;
				CamTimeInfo[selected].y = temp;
				CamTimeInfo[selected].z = 1.0;
			}
			fclose(fp);
		}
		else
		{
			double fps; int temp;
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				{
					CamTimeInfo[selected].x = 1.0 / fps;
					CamTimeInfo[selected].y = temp;
					CamTimeInfo[selected].z = 1.0;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	printLOG("Reading all people 3D skeleton: ");
	double u, v, s, avg_error;
	int cid, dummy, nPeople = 0, nJoints = 25;
	vector<HumanSkeleton3D *> vSkeletons;
	while (true)
	{
		printLOG("%d..", nPeople);

		int nvalidFrames = 0;
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int nvis, nValidJoints = 0, temp = (refFid - startF) / increF;
			sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, nPeople, refFid); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int rfid, pid, nvis, dummy; float fdummy;
				for (int jid = 0; jid < nJoints; jid++)
				{
					int temp = (refFid - startF) / increF;
					fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
					for (int kk = 0; kk < nvis; kk++)
					{
						fscanf(fp, "%d %d %lf %lf %lf %d", &cid, &rfid, &u, &v, &s, &dummy);
						Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(cid, rfid));
						Skeletons[temp].vPt2D[jid].push_back(Point2d(u, v));
						Skeletons[temp].vConf[jid].push_back(s);
					}

					if (abs(Skeletons[temp].pt3d[jid].x) + abs(Skeletons[temp].pt3d[jid].y) + abs(Skeletons[temp].pt3d[jid].z) > 1e-16)
						Skeletons[temp].validJoints[jid] = 1, nValidJoints++;
					else
						Skeletons[temp].validJoints[jid] = 0;
				}
				fclose(fp);

				if (nValidJoints < nJoints / 3)
					Skeletons[temp].valid = 0;
				else
					Skeletons[temp].valid = 1;

				nvalidFrames++;
			}
		}
		if (nvalidFrames == 0)
		{
			printLOG("\n");
			break;
		}

		vSkeletons.push_back(Skeletons);
		nPeople++;
	}

	const int nVertices = smpl::SMPLModel::nVertices, nShapeCoeffs = smpl::SMPLModel::nShapeCoeffs, nJointsSMPL = smpl::SMPLModel::nJoints, nJointsCOCO = 18;
	MatrixXdr outV(nVertices, 3);
	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	MatrixXdr *AllV = new MatrixXdr[maxPeople];
	for (int pi = 0; pi < maxPeople; pi++)
		AllV[pi].resize(nVertices, 3);// .resize(nVertices, 3);

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;

	smpl::SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
		printLOG("Check smpl Path.\n");


	Mat img, rimg, mask, blend;
	smpl::SMPLParams params;
	vector<int> vpid(nPeople);
	vector<smpl::SMPLParams>Vparams(nPeople);
	Point2d joints2D[25];
	Point2d *vuv = new Point2d[nVertices];
	Point3d* vUVZ = new Point3d[nVertices];
	Point3f *allVertices = new Point3f[nVertices*nPeople];
	Point3i *hit = new Point3i[VideoInfo[0].VideoInfo[(startF + stopF) / 2].width*VideoInfo[0].VideoInfo[(startF + stopF) / 2].height];
	float *zbuffer = new float[VideoInfo[0].VideoInfo[(startF + stopF) / 2].width*VideoInfo[0].VideoInfo[(startF + stopF) / 2].height];

	vector<vector<Point3u> *> AccuVextexColors;
	for (int pid = 0; pid < nPeople; pid++)
	{
		vector<Point3u> *VertexColors = new vector<Point3u>[nVertices];
		AccuVextexColors.push_back(VertexColors);
	}

	for (int refFid = startF; refFid <= stopF; refFid += 10)
	{
		printLOG("%d..", refFid);
		int temp = (refFid - startF) / increF;

		for (int ii = 0; ii < nPeople; ii++)
			vpid[ii] = false;
		for (int pid = 0; pid < nPeople; pid++)
		{
			sprintf(Fname, "%s/FitBody/@%d/Wj4/%d/%.2d_%.4d_%.1f.txt", Path, increF, pid, 0, refFid, TimeScale*refFid); //window based
			if (IsFileExist(Fname) == 0)
				continue;
			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%lf %lf %lf %lf ", &Vparams[pid].scale, &Vparams[pid].t(0), &Vparams[pid].t(1), &Vparams[pid].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &Vparams[pid].pose(ii, 0), &Vparams[pid].pose(ii, 1), &Vparams[pid].pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &Vparams[pid].coeffs(ii));
			//Vparams[pid].pose(15, 0) = 0.3, Vparams[pid].pose(15, 1) = 0, Vparams[pid].pose(15, 2) = 0;//up straight face
			fclose(fp);

			vpid[pid] = true;
		}
		int cnt = 0;
		for (int pid = 0; pid < nPeople; pid++)
		{
			if (vpid[pid])
				cnt++;
		}
		if (cnt == 0)
			continue;

#pragma omp parallel for schedule(dynamic,1)
		for (int pid = 0; pid < nPeople; pid++)
		{
			if (!vpid[pid])
				continue;

			reconstruct(smplMaster, Vparams[pid].coeffs.data(), Vparams[pid].pose.data(), AllV[pid].data());
			Map<VectorXd> V_vec(AllV[pid].data(), AllV[pid].size());
			V_vec = V_vec * Vparams[pid].scale + dVdt * Vparams[pid].t;

			for (int ii = 0; ii < nVertices; ii++)
				allVertices[ii + pid * nVertices] = Point3f(AllV[pid](ii, 0), AllV[pid](ii, 1), AllV[pid](ii, 2));
		}

		for (int cid = 0; cid < nCams; cid++)
		{
			double ts = 1.0*refFid / CamTimeInfo[refCid].x;
			int rfid = MyFtoI((ts - CamTimeInfo[cid].y / CamTimeInfo[refCid].x) * CamTimeInfo[cid].x);
			int width = VideoInfo[cid].VideoInfo[rfid].width, height = VideoInfo[cid].VideoInfo[rfid].height;
			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[rfid].valid != 1)
				continue;

			sprintf(Fname, "%s/%d/Corrected/%.4d.png", "Z:/Users/minh/Adobe1", cid, rfid); img = imread(Fname);
			if (img.empty() == 1)
			{
				sprintf(Fname, "%s/%d/Corrected/%.4d.jpg", "Z:/Users/minh/Adobe1", cid, rfid); img = imread(Fname);
				if (img.empty() == 1)
					continue;
			}

			for (int pid = 0; pid < nPeople; pid++)
			{
				HumanSkeleton3D *Body0 = &vSkeletons[pid][temp];

				bool visible = 0;
				int bottomID = -1; double bottomY = 0;
				for (int jid = 0; jid < nJoints; jid++)
				{
					joints2D[jid] = Point2d(0, 0);
					if (Body0[0].validJoints[jid] > 0)
					{
						Point3d xyz = Body0[0].pt3d[jid];
						ProjectandDistort(xyz, &joints2D[jid], camI[rfid].P);
						if (joints2D[jid].x < -camI[rfid].width / 10 || joints2D[jid].x > 11 * camI[rfid].width / 10 || joints2D[jid].y < -camI[rfid].height / 10 || joints2D[jid].y > 11 * camI[rfid].height / 10)
							continue;

						if (joints2D[jid].y > bottomY)
							bottomY = joints2D[jid].y, bottomID = jid;
					}
				}

				//if (debug == 1)
				//{	
				//Mat img_ = img.clone();
				//Draw2DCoCoJoints(img_, joints2D, nJoints, 2, 1.0, &colors[pid % 8]);
				//sprintf(Fname, "%s/Vis/FitBody/x.jpg", Path), imwrite(Fname, img_);
				//}

#pragma omp parallel for schedule(dynamic,1)
				for (int ii = 0; ii < nVertices; ii++)
				{
					Point3d xyz(allVertices[nVertices*pid + ii].x, allVertices[nVertices*pid + ii].y, allVertices[nVertices*pid + ii].z);
					ProjectandDistort(xyz, &vUVZ[ii], camI[rfid].P);

					if (vUVZ[ii].x < -camI[rfid].width / 10 || vUVZ[ii].x > 11 * camI[rfid].width / 10 || vUVZ[ii].y < -camI[rfid].height / 10 || vUVZ[ii].y > 11 * camI[rfid].height / 10)
						continue;
				}

				if (0)
				{
					for (int ii = 0; ii < smplMaster.vFaces.size(); ii++)
					{
						int vid[3] = { smplMaster.vFaces[ii].x,  smplMaster.vFaces[ii].y,  smplMaster.vFaces[ii].z };
						Point2d uv[3] = { Point2d(vUVZ[vid[0]].x, vUVZ[vid[0]].y), Point2d(vUVZ[vid[1]].x, vUVZ[vid[1]].y), Point2d(vUVZ[vid[2]].x,vUVZ[vid[2]].y) };

						if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[1].x >10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10)
							cv::line(img, uv[0], uv[1], colors[pid], 1, CV_AA);
						if (uv[1].x > 10 && uv[1].y > 10 && uv[1].x < width - 10 && uv[1].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
							cv::line(img, uv[0], uv[2], colors[pid], 1, CV_AA);
						if (uv[0].x > 10 && uv[0].y > 10 && uv[0].x < width - 10 && uv[0].y < height - 10 && uv[2].x >10 && uv[2].y > 10 && uv[2].x < width - 10 && uv[2].y < height - 10)
							cv::line(img, uv[1], uv[2], colors[pid], 1, CV_AA);
					}
					imwrite("C:/temp/tri.png", img);
				}

				vector<int> VisibleVertex = RasterizeMeshVisibilty(vUVZ, camI[rfid].width, camI[rfid].height, nVertices, smplMaster.vFaces, hit, zbuffer);

				if (0)
				{
					FILE *fp = fopen("C:/temp/x.txt", "w");
					for (auto vid : VisibleVertex)
						fprintf(fp, "%d\n", vid);
					fclose(fp);

					fp = fopen("C:/temp/y.txt", "w");
					for (int ii = 0; ii < nVertices; ii++)
					{
						Point3d xyz(allVertices[nVertices*pid + ii].x, allVertices[nVertices*pid + ii].y, allVertices[nVertices*pid + ii].z);
						fprintf(fp, "%.4f %.4f %.4f %.1f %.1f\n", xyz.x, xyz.y, xyz.z, vUVZ[ii].x, vUVZ[ii].y);
					}
					fclose(fp);
				}

				//snap the color
				for (auto vid : VisibleVertex)
				{
					Point3u rgb = BilinearInterp(img, img.cols, img.rows, vUVZ[vid].x, vUVZ[vid].y);
					AccuVextexColors[pid][vid].push_back(rgb);
				}
			}
		}
	}

	//do per-channel median filter
	vector<Point3i> * AllVextexColors = new vector<Point3i>[nPeople];
	for (int pid = 0; pid < nPeople; pid++)
	{
		AllVextexColors[pid].resize(nVertices);
		for (int vid = 0; vid < nVertices; vid++)
		{
			if (AccuVextexColors[pid][vid].size() > 0)
			{
				vector<int> color;
				for (int ii = 0; ii < AccuVextexColors[pid][vid].size(); ii++)
					color.push_back((int)AccuVextexColors[pid][vid][ii].x);
				sort(color.begin(), color.end());
				AllVextexColors[pid][vid].x = color[color.size() / 2];

				color.clear();
				for (int ii = 0; ii < AccuVextexColors[pid][vid].size(); ii++)
					color.push_back((int)AccuVextexColors[pid][vid][ii].y);
				sort(color.begin(), color.end());
				AllVextexColors[pid][vid].y = color[color.size() / 2];

				color.clear();
				for (int ii = 0; ii < AccuVextexColors[pid][vid].size(); ii++)
					color.push_back((int)AccuVextexColors[pid][vid][ii].z);
				sort(color.begin(), color.end());
				AllVextexColors[pid][vid].z = color[color.size() / 2];
			}
			else
				AllVextexColors[pid][vid] = Point3i(-1, -1, -1);
		}
	}

	for (int pid = 0; pid < nPeople; pid++)
	{
		sprintf(Fname, "%s/FitBody/Texture_%d.txt", Path, pid); FILE *fp = fopen(Fname, "w");
		for (int vid = 0; vid < nVertices; vid++)
			fprintf(fp, "%d %d %d\n", AllVextexColors[pid][vid].z, AllVextexColors[pid][vid].y, AllVextexColors[pid][vid].x);
		fclose(fp);
	}

	delete[]vuv, delete[]vUVZ, delete[]allVertices, delete[]AllVextexColors;

	return 0;
}
int interpUlti(char *Path, vector<int> &vsCams, int startF, int stopF, int increF)
{
	const double TimeScale = 1000000.0;

	char Fname[512];

	int nSCams = vsCams.size(), nGCams = *max_element(vsCams.begin(), vsCams.end()) + 1;
	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nGCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int temp;
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
					CamTimeInfo[selected].x = 1.0 / fps, CamTimeInfo[selected].y = temp;
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (auto cid : vsCams)
		if (earliest > CamTimeInfo[cid].y)
			earliest = CamTimeInfo[cid].y, refCid = cid;

	double x, y, z, u, v, s, avg_error;

	FILE *fp1 = fopen("C:/temp/list1.txt", "w");
	FILE *fp2 = fopen("C:/temp/list2.txt", "w");

	int skeletonPointFormat = 25, nvalidFrames = 0;
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int  nValidJoints = 0, temp = (refFid - startF) / increF;
		sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, 1, 0, refFid);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, 0, refFid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, 0, refFid);
				if (IsFileExist(Fname) == 0)
					continue;
			}
		}
		fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			double ts = (CamTimeInfo[refCid].y / CamTimeInfo[refCid].x + 1.0*refFid / CamTimeInfo[refCid].x)*CamTimeInfo[refCid].x;
			fprintf(fp1, "%d %d %.1f %.1f\n", 0, refFid, TimeScale* refFid, round(ts*TimeScale));
			int rcid, rfid, nvis, inlier;
			vector<int> usedCid;
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				fscanf(fp, "%lf %lf %lf %lf %d ", &x, &y, &z, &avg_error, &nvis);
				for (int kk = 0; kk < nvis; kk++)
				{
					fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);

					bool used = false;
					for (auto ii : usedCid)
						if (rcid == ii)
							used = true;
					if (used)
						continue;
					usedCid.push_back(rcid);
					ts = (CamTimeInfo[rcid].y / CamTimeInfo[refCid].x + 1.0*rfid / CamTimeInfo[rcid].x)*CamTimeInfo[refCid].x;
					fprintf(fp2, "%d %d %.1f\n", rcid, rfid, round(ts*TimeScale));
				}
			}
			fclose(fp);
		}
	}

	fclose(fp1), fclose(fp2);

	return 0;
}
int TextureMappingBody(char *Path, int startF, int stopF, int increF, int nCams, int selectedView = -1, int nMaxPeople = 2)
{
	vector<Scalar> colors;
	colors.push_back(Scalar(0, 0, 0));
	colors.push_back(Scalar(128, 128, 0));
	colors.push_back(Scalar(0, 0, 255));
	colors.push_back(Scalar(128, 0, 128));
	colors.push_back(Scalar(0, 128, 255));
	colors.push_back(Scalar(0, 255, 255));
	colors.push_back(Scalar(255, 0, 128));
	colors.push_back(Scalar(0, 0, 128));
	colors.push_back(Scalar(0, 255, 0));
	colors.push_back(Scalar(255, 128, 0));
	colors.push_back(Scalar(255, 255, 0));
	colors.push_back(Scalar(255, 0, 0));
	colors.push_back(Scalar(0, 128, 0));
	colors.push_back(Scalar(255, 128, 0));
	colors.push_back(Scalar(0, 128, 128));
	colors.push_back(Scalar(255, 0, 255));
	colors.push_back(Scalar(255, 255, 255));
	colors.push_back(Scalar(128, 0, 0));

	char Fname[512];
	sprintf(Fname, "%s/ProjectiveTexturemap", Path); makeDir(Fname);
	for (int ii = 0; ii < nCams; ii++)
		sprintf(Fname, "%s/ProjectiveTexturemap/%d", Path, ii), makeDir(Fname);

	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int selected;
		double fps, temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps;
			CamTimeInfo[selected].y = temp;
			CamTimeInfo[selected].z = 1.0;
		}
		fclose(fp);
	}
	else
	{
		int selected;
		double fps, temp;
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			{
				CamTimeInfo[selected].x = 1.0 / fps;
				CamTimeInfo[selected].y = temp;
				CamTimeInfo[selected].z = 1.0;
			}
			fclose(fp);
		}
		else
		{
			int selected;
			double fps, temp;
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				{
					CamTimeInfo[selected].x = 1.0 / fps;
					CamTimeInfo[selected].y = temp;
					CamTimeInfo[selected].z = 1.0;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;

	vector<Point2i> **allNearestKeyFrames = new vector<Point2i>*[nCams];
	for (int cid = 0; cid < nCams; cid++)
	{
		allNearestKeyFrames[cid] = new vector<Point2i>[stopF + 1];
		sprintf(Fname, "%s/%d/kNN4IRB.txt", Path, cid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		int fidi, nn;
		Point2i cid_fid;
		while (fscanf(fp, "%d %d ", &fidi, &nn) != EOF)
		{
			if (fidi > stopF)
				break;
			for (int ii = 0; ii < nn; ii++)
				fscanf(fp, "%d %d ", &cid_fid.x, &cid_fid.y), allNearestKeyFrames[cid][fidi].push_back(cid_fid);
		}
		fclose(fp);
	}

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 18;
	int naJoints = SMPLModel::naJoints;

	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	SMPLModel mySMPL;
	if (!ReadSMPLData("smpl", mySMPL))
		printLOG("Check smpl Path.\n");
	else
	{
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		dVdt = Eigen::kroneckerProduct(VectorXd::Ones(smpl::SMPLModel::nVertices), eye3);
	}
	MatrixXdr *AllV = new MatrixXdr[nMaxPeople];
	for (int pi = 0; pi < nMaxPeople; pi++)
		AllV[pi].resize(SMPLModel::nVertices, 3);

	Mat img, nr_img;
	SMPLParams params;
	double Tscale = 1000000.0;
	vector<int> vpid(nMaxPeople);
	vector<smpl::SMPLParams>Vparams(nMaxPeople);
	bool *validFace = new bool[mySMPL.vFaces.size()],
		*validFaceNR = new bool[mySMPL.vFaces.size()];
	Point3d* vUVZ = new Point3d[nVertices],
		*vUVZ_NR = new Point3d[nVertices];
	int *hitFace = new int[VideoInfo[0].VideoInfo[(startF + stopF) / 2].width*VideoInfo[0].VideoInfo[(startF + stopF) / 2].height];
	Point3i *hit = new Point3i[VideoInfo[0].VideoInfo[(startF + stopF) / 2].width*VideoInfo[0].VideoInfo[(startF + stopF) / 2].height];
	float *zbuffer = new float[VideoInfo[0].VideoInfo[(startF + stopF) / 2].width*VideoInfo[0].VideoInfo[(startF + stopF) / 2].height],
		*zbuffer_NR = new float[VideoInfo[0].VideoInfo[(startF + stopF) / 2].width*VideoInfo[0].VideoInfo[(startF + stopF) / 2].height];

	for (int reffid = startF; reffid <= stopF; reffid++)
	{
		Vparams.clear();
		for (int peopleCount = 0; peopleCount < nMaxPeople; peopleCount++)
		{
			//sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, peopleCount, 0, reffid, round(Tscale*reffid));
			sprintf(Fname, "%s/FitBody/@%d/US_Smoothing50/%d/%.2d_%.4d_%.1f.txt", Path, increF, peopleCount, 0, reffid, round(Tscale*reffid));
			if (IsFileExist(Fname) == 0)
				continue;
			printf("%s\n", Fname);
			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%lf %lf %lf %lf ", &params.scale, &params.t(0), &params.t(1), &params.t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &params.pose(ii, 0), &params.pose(ii, 1), &params.pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &params.coeffs(ii));
			fclose(fp);

			Vparams.push_back(params);
		}
		if (Vparams.size() == 0)
			continue;

#pragma omp parallel for schedule(dynamic,1)
		for (int pi = 0; pi < Vparams.size(); pi++)
		{
			reconstruct(mySMPL, Vparams[pi].coeffs.data(), Vparams[pi].pose.data(), AllV[pi].data());
			Map<VectorXd> V_vec(AllV[pi].data(), AllV[pi].size());
			V_vec = V_vec * Vparams[pi].scale + dVdt * Vparams[pi].t;
		}

		vector<int> sCams;
		if (selectedView == -1)
			for (int ii = 0; ii < nCams; ii++)
				sCams.push_back(ii);
		else
			sCams.push_back(selectedView);
		for (int ll = 0; ll < sCams.size(); ll++)
		{
			int cid = sCams[ll];
			double ts = 1.0*reffid / CamTimeInfo[refCid].x;
			int rfid = MyFtoI((ts - CamTimeInfo[cid].y / CamTimeInfo[refCid].x) * CamTimeInfo[cid].x);

			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[rfid].valid != 1)
				continue;

			if (allNearestKeyFrames[cid][rfid].size() == 0)
				continue;

			/*sprintf(Fname, "%s/%d/Corrected/%.4d.png", Path, cid, rfid); img = imread(Fname);
			if (img.empty() == 1)
			{
				sprintf(Fname, "%s/%d/Corrected/%.4d.jpg", Path, cid, rfid); img = imread(Fname);
				if (img.empty() == 1)
					continue;
			}*/

			//determine visible vertex for the ref view
			for (int pid = 0; pid < Vparams.size(); pid++)
			{
#pragma omp parallel for schedule(dynamic,1)
				for (int ii = 0; ii < nVertices; ii++)
				{
					vUVZ[ii].x = -1, vUVZ[ii].y = -1;
					Point3d xyz(AllV[pid](ii, 0), AllV[pid](ii, 1), AllV[pid](ii, 2));
					ProjectandDistort(xyz, &vUVZ[ii], camI[rfid].P);

					if (vUVZ[ii].x < -camI[rfid].width / 10 || vUVZ[ii].x > 11 * camI[rfid].width / 10 || vUVZ[ii].y < -camI[rfid].height / 10 || vUVZ[ii].y > 11 * camI[rfid].height / 10)
						vUVZ[ii].x = -1, vUVZ[ii].y = -1;
				}

				vector<int> VisibleVertex = RasterizeMeshVisibilty(vUVZ, camI[rfid].width, camI[rfid].height, nVertices, mySMPL.vFaces, hit, zbuffer, validFace, hitFace);
				if (VisibleVertex.size() == 0)
					continue;

				/*Mat timg = img.clone();
				for (auto ii : VisibleVertex)
				{
					int vid1 = mySMPL.vFaces[ii].x, vid2 = mySMPL.vFaces[ii].y, vid3 = mySMPL.vFaces[ii].z;
					Point2f uv1(vUVZ[vid1].x, vUVZ[vid1].y), uv2(vUVZ[vid2].x, vUVZ[vid2].y), uv3(vUVZ[vid3].x, vUVZ[vid3].y);
					line(timg, uv1, uv2, colors[ii % 8], 1);
				}
				imwrite("C:/temp/x.png", timg);*/

				for (int nn = 0; nn < allNearestKeyFrames[cid][rfid].size(); nn++)
				{
					int nr_cid = allNearestKeyFrames[cid][rfid][nn].x, nr_rfid = allNearestKeyFrames[cid][rfid][nn].y;
					CameraData *nr_camI = VideoInfo[nr_cid].VideoInfo;
					if (nr_camI[nr_rfid].valid != 1)
						continue;

					sprintf(Fname, "%s/%d/Corrected/%.4d_.png", Path, nr_cid, nr_rfid); nr_img = imread(Fname);
					if (nr_img.empty() == 1)
					{
						sprintf(Fname, "%s/%d/Corrected/%.4d_.jpg", Path, nr_cid, nr_rfid); nr_img = imread(Fname);
						if (nr_img.empty() == 1)
							continue;
					}

#pragma omp parallel for schedule(dynamic,1)
					for (int ii = 0; ii < nVertices; ii++)
					{
						vUVZ_NR[ii].x = -1, vUVZ_NR[ii].y = -1;
						Point3d xyz(AllV[pid](ii, 0), AllV[pid](ii, 1), AllV[pid](ii, 2));
						ProjectandDistort(xyz, &vUVZ_NR[ii], nr_camI[nr_rfid].P);

						if (vUVZ_NR[ii].x < -camI[rfid].width / 10 || vUVZ_NR[ii].x > 11 * camI[rfid].width / 10 || vUVZ_NR[ii].y < -camI[rfid].height / 10 || vUVZ_NR[ii].y > 11 * camI[rfid].height / 10)
							vUVZ_NR[ii].x = -1, vUVZ_NR[ii].y = -1;
					}
					VisibleVertex.clear();
					VisibleVertex = RasterizeMeshVisibilty(vUVZ_NR, nr_camI[nr_rfid].width, nr_camI[nr_rfid].height, nVertices, mySMPL.vFaces, hit, zbuffer_NR, validFaceNR);

					vector<int> coVisible;
					for (int ii = 0; ii < mySMPL.vFaces.size(); ii++)
						if (validFace[ii] && validFaceNR[ii])
							coVisible.push_back(ii);

					/*Mat timg = img.clone();
					for (auto ii : coVisible)
					{
						int vid1 = mySMPL.vFaces[ii].x, vid2 = mySMPL.vFaces[ii].y, vid3 = mySMPL.vFaces[ii].z;
						Point2f uv1(vUVZ[vid1].x, vUVZ[vid1].y), uv2(vUVZ[vid2].x, vUVZ[vid2].y), uv3(vUVZ[vid3].x, vUVZ[vid3].y);
						line(timg, uv1, uv2, colors[ii % 8], 1);
					}*/

					/*Mat timg_nr = nr_img.clone();
					for (auto ii : coVisible)
					{
						int vid1 = mySMPL.vFaces[ii].x, vid2 = mySMPL.vFaces[ii].y, vid3 = mySMPL.vFaces[ii].z;
						Point2f uv1(vUVZ_NR[vid1].x, vUVZ_NR[vid1].y), uv2(vUVZ_NR[vid2].x, vUVZ_NR[vid2].y), uv3(vUVZ_NR[vid3].x, vUVZ_NR[vid3].y);
						line(timg_nr, uv1, uv2, colors[ii % 8], 1);
					}
					imwrite("C:/temp/x.png", timg_nr);*/

					Mat img_synth = Mat::zeros(camI[rfid].height, camI[rfid].width, CV_8UC3);
#pragma omp parallel for schedule(dynamic,1)
					for (int jj = 0; jj < camI[rfid].height; jj++)
					{
						for (int ii = 0; ii < camI[rfid].width; ii++)
						{
							int id = jj * camI[rfid].width + ii;
							double depth = zbuffer[id];
							if (depth < FLT_MAX)
							{
								bool found = false;
								for (int kk = 0; kk < coVisible.size() && !found; kk++)
									if (hitFace[id] == coVisible[kk])
										found = true;
								if (!found)
									continue;

								double uv1[3] = { ii, jj, 1 }, rayDir[3];
								Map< const MatrixXdr >	eR(camI[rfid].R, 3, 3);
								Map< const MatrixXdr >	eiK(camI[rfid].invK, 3, 3);
								Map<Vector3d> euv1(uv1, 3);
								Map<Vector3d> eRayDir(rayDir, 3);
								eRayDir = eR.transpose()*eiK*euv1;

								Point2d uv;
								Point3d xyz(depth*rayDir[0] + camI[rfid].camCenter[0], depth*rayDir[1] + camI[rfid].camCenter[1], depth*rayDir[2] + camI[rfid].camCenter[2]);
								ProjectandDistort(xyz, &uv, nr_camI[nr_rfid].P);

								Point3u rgb = BilinearInterp(nr_img, nr_img.cols, nr_img.rows, uv.x, uv.y);
								img_synth.data[3 * id] = rgb.x, img_synth.data[3 * id + 1] = rgb.y, img_synth.data[3 * id + 2] = rgb.z;
							}
						}
					}

					sprintf(Fname, "%s/ProjectiveTexturemap/%d/%d_%.4d_%d_%.4d_%d.png", Path, cid, cid, rfid, nr_cid, nr_rfid, pid);
					imwrite(Fname, img_synth);
				}
			}
		}
	}

	return 0;
}

int TrackBody_Landmark_BiDirectLK2(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, int cid, int startF, int stopF, int increF, double bwThresh, int debug)
{
	const int nJoints = 17;
	double confThresh = 0.0;
	char Fname[512]; FILE *fp = 0;
	sprintf(Fname, "%s/Logs/TrackBody_Landmark_BiDirectLK_%d_%.4d_%4d.txt", Path, cid, startF, stopF);
	sprintf(Fname, "%s/Vis", Path); makeDir(Fname);
	sprintf(Fname, "%s/Vis/TrackBody_LM", Path); makeDir(Fname);
	if (debug > 1)
		sprintf(Fname, "%s/Vis/TrackBody_LM/%d", Path, cid), makeDir(Fname);

	printLOG("\n\nWorking on %d LM body track: ", cid);
	if (IsFileExist(Fname) == 1)
	{
		printLOG("%s computed\n", Fname);
		//return 0;
	}
	vector<Scalar> colors;
	colors.push_back(Scalar(0, 0, 255));
	colors.push_back(Scalar(0, 128, 255));
	colors.push_back(Scalar(0, 255, 255));
	colors.push_back(Scalar(0, 255, 0));
	colors.push_back(Scalar(255, 128, 0));
	colors.push_back(Scalar(255, 255, 0));
	colors.push_back(Scalar(255, 0, 0));
	colors.push_back(Scalar(255, 0, 255));
	colors.push_back(Scalar(255, 255, 255));

	double bwThresh2 = bwThresh * bwThresh;
	int winSizeI = 31, cvPyrLevel = 5;
	vector<float> err;
	vector<uchar> status;
	Size winSize(winSizeI, winSizeI);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01);

	Mat Img1, Img2, cImg1, cImg2, bImg, bImg2;
	vector<Point2f> lm1, lm2, lm1_, lm2_; Point2f tl, br, tl2, br2;
	vector<float> vs1, vs2;
	vector<Mat> Img1Pyr, Img2Pyr; vector<Mat> vImg(2);

	sprintf(Fname, "%s/%s/general_%d_%d/image_%.10d_0.png", Path, SelectedCamNames[cid / CamIdsPerSeq.size()], SeqId, CamIdsPerSeq[cid%CamIdsPerSeq.size()], startF);
	if (IsFileExist(Fname) == 0)
		return 1;
	Img1 = imread(Fname, 0);
	buildOpticalFlowPyramid(Img1, Img1Pyr, winSize, cvPyrLevel, false);

	float u, v, s; lm1.clear(), vs1.clear();
	sprintf(Fname, "%s/%s/DensePose/general_%d_%d/image_%.10d_0.png.txt", Path, SelectedCamNames[cid / CamIdsPerSeq.size()], SeqId, CamIdsPerSeq[cid%CamIdsPerSeq.size()], startF);
	if (readKeyPointJson(Fname, lm1, vs1) == 0)
		return 1;

	VideoWriter writer;
	int  firstTime = 1; double resizeFactor = 0.25;
	CvSize size; size.width = (int)(resizeFactor*Img1.cols * 2), size.height = (int)(resizeFactor*Img1.rows);
	if (debug > 0)
		sprintf(Fname, "%s/Vis/TrackBody_LM/%d_%d_%d.avi", Path, cid, startF, stopF), writer.open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 25, size);

	sprintf(Fname, "%s/%d/flowgraph_FromLandmark_%d_%d.txt", Path, cid, startF, stopF); FILE *fpOut = fopen(Fname, "w");
	for (int fid = startF + 1; fid < stopF; fid++)
	{
		printLOG("%d..", fid);
		sprintf(Fname, "%s/%s/general_%d_%d/image_%.10d_0.png", Path, SelectedCamNames[cid / CamIdsPerSeq.size()], SeqId, CamIdsPerSeq[cid%CamIdsPerSeq.size()], fid);
		if (IsFileExist(Fname) == 0)
			continue;
		Img2 = imread(Fname, 0);
		buildOpticalFlowPyramid(Img2, Img2Pyr, winSize, cvPyrLevel, false);

		lm2.clear(), vs2.clear();
		sprintf(Fname, "%s/%s/DensePose/general_%d_%d/image_%.10d_0.png.txt", Path, SelectedCamNames[cid / CamIdsPerSeq.size()], SeqId, CamIdsPerSeq[cid%CamIdsPerSeq.size()], fid);
		if (readKeyPointJson(Fname, lm2, vs2) == 0)
			continue;

		if (lm2.size() == 0)
		{
			lm1.clear();
			continue;
		}
		if (lm1.size() == 0) //no data
		{
			swap(lm1, lm2), swap(vs1, vs2), swap(Img1, Img2), swap(Img1Pyr, Img2Pyr);
			continue;
		}

		lm2_ = lm1, lm1_ = lm2;
		calcOpticalFlowPyrLK(Img1Pyr, Img2Pyr, lm1, lm2_, status, err, winSize, cvPyrLevel, termcrit);
		calcOpticalFlowPyrLK(Img2Pyr, Img1Pyr, lm2, lm1_, status, err, winSize, cvPyrLevel, termcrit);

		int npid1 = lm1.size() / nJoints, npid2 = lm2.size() / nJoints;
		vector<int> association(npid1), used(npid2);
		for (int pid1 = 0; pid1 < npid1; pid1++)
		{
			int nvalidJoints = 0;
			for (int jid = 0; jid < nJoints; jid++)
			{
				if (lm1[pid1*nJoints + jid].x > 0 && vs1[jid + pid1 * nJoints] > confThresh)  //bad detection
					nvalidJoints++;
			}
			if (nvalidJoints < nJoints / 3)  //let's be conservative about occlusion
			{
				association[pid1] = -1;
				continue;
			}

			int bestAssignment = -1, best = 0;
			for (int pid2 = 0; pid2 < npid2; pid2++)
			{
				nvalidJoints = 0;
				for (int jid = 0; jid < nJoints; jid++)
					if (lm2[pid2*nJoints + jid].x > 0 && vs2[jid + pid2 * nJoints] > confThresh)  //bad detection
						nvalidJoints++;
				if (nvalidJoints < nJoints / 3)  //let's be conservative about occlusion
					continue;

				int good = 0;
				double dist1, dist2;
				for (int jid = 0; jid < nJoints; jid++)
				{
					if (lm1[pid1 * nJoints + jid].x > 0 && lm2[pid2 * nJoints + jid].x > 0)
					{
						Point2f p1 = lm1[pid1 * nJoints + jid], p2 = lm2[pid2 * nJoints + jid];
						Point2f p1_ = lm2_[pid1 * nJoints + jid], p2_ = lm2_[pid1 * nJoints + jid];
						dist1 = pow(lm2_[pid1 * nJoints + jid].x - lm2[pid2 * nJoints + jid].x, 2) + pow(lm2_[pid1 * nJoints + jid].y - lm2[pid2 * nJoints + jid].y, 2);
						dist2 = pow(lm1[pid1 * nJoints + jid].x - lm1_[pid2 * nJoints + jid].x, 2) + pow(lm1[pid1 * nJoints + jid].y - lm1_[pid2 * nJoints + jid].y, 2);
						if (dist1 < bwThresh2 && dist2 < bwThresh2)
							good++;
					}
				}
				if (good > best)
					best = good, bestAssignment = pid2;
			}
			if (best >= nJoints / 3 && used[bestAssignment] == 0) //Let's be conservative at this point
				association[pid1] = bestAssignment, used[bestAssignment] = 1;  //establish link.
			else
				association[pid1] = -1;
		}

		if (debug > 0)//Visualization: draw assocation
		{
			vImg[0] = Img1.clone(), vImg[1] = Img2.clone();
			cvtColor(vImg[0], vImg[0], CV_GRAY2BGR), cvtColor(vImg[1], vImg[1], CV_GRAY2BGR);

			for (int pid1 = 0; pid1 < npid1; pid1++)
			{
				int nvalidJoints = 0;
				for (int jid = 0; jid < nJoints; jid++)
					if (lm1[pid1*nJoints + jid].x > 0)
						nvalidJoints++;
				if (nvalidJoints < 5)
					continue;

				tl = Point2f(9e9, 9e9), br = Point2f(0, 0);
				for (size_t jid = 0; jid < nJoints; jid++)
					if (lm1[pid1*nJoints + jid].x > 0)
						tl.x = min(tl.x, lm1[pid1*nJoints + jid].x), tl.y = min(tl.y, lm1[pid1*nJoints + jid].y), br.x = max(br.x, lm1[pid1*nJoints + jid].x), br.y = max(br.y, lm1[pid1*nJoints + jid].y);
				rectangle(vImg[0], tl, br, colors[pid1 % 9], 8, 8, 0);
				sprintf(Fname, "%d", pid1), putText(vImg[0], Fname, Point2i(tl.x, br.y - vImg[0].cols / 60), CV_FONT_HERSHEY_SIMPLEX, 1.0*vImg[0].cols / 640, colors[pid1 % 9], 3);
				Draw2DCoCoJoints(vImg[0], &lm1[pid1*nJoints], nJoints, 1.0*vImg[1].cols / 640);

				int associated = association[pid1];
				if (associated > -1)
				{
					tl = Point2f(9e9, 9e9), br = Point2f(0, 0);
					for (size_t jid = 0; jid < nJoints; jid++)
						if (lm2[associated*nJoints + jid].x > 0)
							tl.x = min(tl.x, lm2[associated*nJoints + jid].x), tl.y = min(tl.y, lm2[associated*nJoints + jid].y), br.x = max(br.x, lm2[associated*nJoints + jid].x), br.y = max(br.y, lm2[associated*nJoints + jid].y);
					rectangle(vImg[1], tl, br, colors[pid1 % 9], 8, 8, 0);
					sprintf(Fname, "%d", pid1), putText(vImg[1], Fname, Point2i(tl.x, br.y - vImg[0].cols / 60), CV_FONT_HERSHEY_SIMPLEX, 1.0*vImg[1].cols / 640, colors[pid1 % 9], 3);
					Draw2DCoCoJoints(vImg[1], &lm2[associated*nJoints], nJoints, 1.0*vImg[1].cols / 640);
				}
			}
			bImg = DrawTitleImages(vImg, 2.0);
			CvPoint text_origin = { bImg.rows / 20, bImg.cols / 20 };
			sprintf(Fname, "%d", fid - 1), putText(bImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*bImg.cols / 640, colors[0], 3);
			//imshow("X", bImg); waitKey(1);
			resize(bImg, bImg2, Size(resizeFactor* bImg.cols, resizeFactor*bImg.rows), 0, 0, INTER_AREA);
			writer << bImg2;
			if (debug > 1)
				sprintf(Fname, "%s/Vis/TrackBody_LM/%d/%.4d.jpg", Path, cid, fid - 1), imwrite(Fname, bImg2);
		}

		//write out data
		fprintf(fpOut, "%d %zd ", fid - 1, association.size());
		for (auto asso : association)
			fprintf(fpOut, "%d ", asso);
		fprintf(fpOut, "\n");

		swap(lm1, lm2), swap(vs1, vs2), swap(Img1, Img2), swap(Img1Pyr, Img2Pyr);
	}
	fclose(fpOut);

	if (debug > 0)
		writer.release();

	sprintf(Fname, "%s/Logs/TrackBody_Landmark_BiDirectLK_%d_%.4d_%4d.txt", Path, cid, startF, stopF);
	fp = fopen(Fname, "w"); fclose(fp);

	return 0;
}
int CleanTrackletBy2DSmoothing_V3(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, int sCamId, int startF, int stopF, int increF, int imWidth, double dispThresh, double  overDispRatioThresh, int debug)
{
	char Fname[512];
	const int nJoints = 17;

	int nf, sf, pid;
	vector<vector<Point2i> >trackletVec;
	sprintf(Fname, "%s/%d/rawTracklet_%d_%d.txt", Path, sCamId, startF, stopF); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d ", &nf, &sf) != EOF)
	{
		vector<Point2i> tracklet;
		for (int f = 0; f < nf; f++)
		{
			fscanf(fp, "%d ", &pid);
			tracklet.push_back(Point2i(sf + f, pid));
		}
		trackletVec.push_back(tracklet);
	}
	fclose(fp);

	float u, v, s;
	vector<float> vConf;
	vector<vector<Point2f> *> allPoses;
	for (int fid = startF; fid <= stopF; fid++)
	{
		vConf.clear();
		vector<Point2f> *PoseI = new vector<Point2f>[1];
		sprintf(Fname, "%s/%s/DensePose/general_%d_%d/image_%.10d_0.png.txt", Path, SelectedCamNames[sCamId / CamIdsPerSeq.size()], SeqId, CamIdsPerSeq[sCamId%CamIdsPerSeq.size()], fid);
		readKeyPointJson(Fname, PoseI[0], vConf);
		allPoses.push_back(PoseI);
	}

	//Enforce smoothing constraint
	vector<double> disp, disp2;
	vector<Point2i> splitingPoint;
	vector<vector<Point2i> > newTrackletVec;
	vector<Point3i> allPair;
	if (nJoints == 18)
	{
		allPair.push_back(Point3i(0, 0, 0));
		allPair.push_back(Point3i(1, 1, 1));
		allPair.push_back(Point3i(2, 2, 5));
		allPair.push_back(Point3i(3, 3, 6));
		allPair.push_back(Point3i(4, 4, 7));
		allPair.push_back(Point3i(5, 5, 2));
		allPair.push_back(Point3i(6, 6, 3));
		allPair.push_back(Point3i(7, 7, 4));
		allPair.push_back(Point3i(8, 8, 11));
		allPair.push_back(Point3i(9, 9, 12));
		allPair.push_back(Point3i(10, 10, 13));
		allPair.push_back(Point3i(11, 11, 8));
		allPair.push_back(Point3i(12, 12, 9));
		allPair.push_back(Point3i(13, 13, 10));
		allPair.push_back(Point3i(14, 14, 15));
		allPair.push_back(Point3i(15, 15, 14));
		allPair.push_back(Point3i(16, 16, 17));
		allPair.push_back(Point3i(17, 17, 16));
	}
	else
		for (int ii = 0; ii < nJoints; ii++)
			allPair.push_back(Point3i(ii, ii, ii));

	for (int id = 0; id < (int)trackletVec.size(); id++)
	{
		vector <Point2i>  newTracklet;
		newTracklet.push_back(Point2i(trackletVec[id][0].x, trackletVec[id][0].y));

		for (int tid = 0; tid < (int)trackletVec[id].size() - 2; tid++)
		{
			int fid0 = trackletVec[id][tid].x, fid1 = trackletVec[id][tid + 1].x, fid2 = trackletVec[id][tid + 2].x;
			int pid0 = trackletVec[id][tid].y, pid1 = trackletVec[id][tid + 1].y, pid2 = trackletVec[id][tid + 2].y;
			Point2f *pts0 = &allPoses[fid0 - startF][0][nJoints*pid0];
			Point2f *pts1 = &allPoses[fid1 - startF][0][nJoints*pid1];
			Point2f *pts2 = &allPoses[fid2 - startF][0][nJoints*pid2];

			disp.clear();
			for (int jid = 0; jid < nJoints; jid++)
			{
				double dist1 = 9e9, dist2 = 9e9, dist;
				if (pts0[allPair[jid].x].x > 0 && pts1[allPair[jid].y].x > 0)
					dist1 = norm(pts0[allPair[jid].x] - pts1[allPair[jid].y]);
				if (pts0[allPair[jid].x].x > 0 && pts1[allPair[jid].z].y > 0)
					dist2 = norm(pts0[allPair[jid].x] - pts1[allPair[jid].z]);
				dist = min(dist1, dist2);
				if (dist < 8e9)
					disp.push_back(dist);
			}
			disp2.clear();
			for (int jid = 0; jid < nJoints; jid++)
			{
				double dist1 = 9e9, dist2 = 9e9, dist;
				if (pts0[allPair[jid].x].x > 0 && pts2[allPair[jid].y].x > 0)
					dist1 = norm(pts0[allPair[jid].x] - pts2[allPair[jid].y]);
				if (pts0[allPair[jid].x].x > 0 && pts2[allPair[jid].z].y > 0)
					dist2 = norm(pts0[allPair[jid].x] - pts2[allPair[jid].z]);
				dist = min(dist1, dist2);
				if (dist < 8e9)
					disp2.push_back(dist);
			}

			if (disp.size() == 0 || disp2.size() == 0)
				newTracklet.push_back(Point2i(fid1, pid1));
			else
			{
				sort(disp.begin(), disp.end());
				sort(disp2.begin(), disp2.end());
				size_t upper = disp.size() * 7 / 10, mid = disp.size() / 2, upper2 = disp2.size() * 7 / 10, mid2 = disp2.size() / 2;
				bool cond1 = (disp[upper] > dispThresh*imWidth / 1920 && disp[upper] - disp[0] > overDispRatioThresh*disp[mid]), cond2 = (disp2[upper2] > dispThresh*imWidth / 1920 && disp2[upper2] - disp2[0] > overDispRatioThresh*disp2[mid2]);
				if (cond1 || cond2)
				{
					splitingPoint.push_back(Point2i(id, tid));
					if (newTracklet.size() > 0) //remove the last element (the next element is also discarded)
						newTracklet.pop_back();
					newTrackletVec.push_back(newTracklet);

					if (cond2 == true)
						tid++;//the next instance is also bad

					newTracklet.clear(); //start new tracklet
										 /*//tracklet of 1 one element
										 newTracklet.push_back(Point2i(fid0, pid0));
										 newTrackletVec.push_back(newTracklet);
										 newTracklet.clear(); //start new tracklet;*/

										 //newTracklet.push_back(Point2i(fid1, pid1));
				}
				else
					newTracklet.push_back(Point2i(fid1, pid1));
			}
		}
		newTrackletVec.push_back(newTracklet);
	}

	int count = 0;
	int *dummy = new int[newTrackletVec.size()];
	int *TrackletStartF = new int[newTrackletVec.size()];
	for (int id = 0; id < (int)newTrackletVec.size(); id++)
	{
		if (newTrackletVec[id].size() == 0)
			continue;
		dummy[count] = id;
		TrackletStartF[count] = newTrackletVec[id][0].x;
		count++;
	}
	Quick_Sort_Int(TrackletStartF, dummy, 0, count - 1);

	vector<vector<Point2i> > newSortedTrackletVec;
	for (int ii = 0; ii < count; ii++)
	{
		int tid = dummy[ii];
		vector<Point2i> sortedTracklet;
		for (int jj = 0; jj < newTrackletVec[tid].size(); jj++)
			sortedTracklet.push_back(newTrackletVec[tid][jj]);
		newSortedTrackletVec.push_back(sortedTracklet);
	}
	delete[]dummy, delete[]TrackletStartF;

	//detect overlapping boxes and discard
	for (int tid1 = 0; tid1 < newSortedTrackletVec.size(); tid1++)
	{
		for (int tid2 = tid1 + 1; tid2 < newSortedTrackletVec.size(); tid2++)
		{
			bool breakFlag = false;
			for (int inst1 = 0; inst1 < newSortedTrackletVec[tid1].size() && !breakFlag; inst1++)
			{
				for (int inst2 = 0; inst2 < newSortedTrackletVec[tid2].size() && !breakFlag; inst2++)
				{
					int fid = newSortedTrackletVec[tid1][inst1].x, pid = newSortedTrackletVec[tid1][inst1].y;
					if (newSortedTrackletVec[tid1][inst1].x == newSortedTrackletVec[tid2][inst2].x && newSortedTrackletVec[tid1][inst1].y == newSortedTrackletVec[tid2][inst2].y)
					{
						breakFlag = true;
						newSortedTrackletVec[tid1].erase(newSortedTrackletVec[tid1].begin() + inst1, newSortedTrackletVec[tid1].end());
						newSortedTrackletVec[tid2].erase(newSortedTrackletVec[tid2].begin() + inst2, newSortedTrackletVec[tid2].end());
					}
				}
			}
		}
	}

	sprintf(Fname, "%s/%d/Tracklet_%d_%d.txt", Path, sCamId, startF, stopF); fp = fopen(Fname, "w");
	for (int id = 0; id < (int)newSortedTrackletVec.size(); id++)
	{
		if (newSortedTrackletVec[id].size() == 0)
			continue;

		fprintf(fp, "%d %d ", (int)newSortedTrackletVec[id].size(), newSortedTrackletVec[id][0].x);
		for (int tid = 0; tid < newSortedTrackletVec[id].size(); tid++)
			fprintf(fp, "%d ", (int)newSortedTrackletVec[id][tid].y);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 0;
}


bool CreateBodyJointFromSeq(char *Path, int nSub, int nActions)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 24;

	char Fname[512];
	SparseMatrix<double, ColMajor> dVdt;
	SMPLModel mySMPL;
	if (!ReadSMPLData("smpl", mySMPL))
		printLOG("Check smpl Path.\n");
	else
	{
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		dVdt = Eigen::kroneckerProduct(VectorXd::Ones(smpl::SMPLModel::nVertices), eye3);
	}
	MatrixXdr AllV(SMPLModel::nVertices, 3);
	SMPLParams params;
	vector< SMPLParams> VParams;
	VectorXd Jsmpl_vec;

	for (int subId = 1; subId <= nSub; subId++)
	{
		for (int actionId = 1; actionId <= nActions; actionId++)
		{
			printLOG("\n%d, %d:\n", subId, actionId);

			VParams.clear();
			sprintf(Fname, "%s/%.2d/%.2d_%.2d.txt", Path, subId, subId, actionId);
			if (IsFileExist(Fname) == 0)
				continue;

			FILE *fp = fopen(Fname, "r");
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &params.coeffs(ii));
			int tid;
			while (fscanf(fp, "%d ", &tid) != EOF)
			{
				params.scale = 1.0;
				fscanf(fp, "%lf %lf %lf ", &params.t(0), &params.t(1), &params.t(2));
				for (int ii = 0; ii < nJointsSMPL; ii++)
					fscanf(fp, "%lf %lf %lf ", &params.pose(ii, 0), &params.pose(ii, 1), &params.pose(ii, 2));
				VParams.push_back(params);
			}
			fclose(fp);

			sprintf(Fname, "%s/%.2d/%.2d_%.2d", Path, subId, subId, actionId), makeDir(Fname);
			sprintf(Fname, "%s/%.2d/%.2d_%.2d/Track3D", Path, subId, subId, actionId), makeDir(Fname);

			for (int tid = 0; tid < VParams.size(); tid++)
			{
				printLOG("%d ", tid);

				reconstruct(mySMPL, VParams[tid].coeffs.data(), VParams[tid].pose.data(), AllV.data());
				Map<VectorXd> V_vec(AllV.data(), AllV.size());
				V_vec = V_vec * VParams[tid].scale + dVdt * VParams[tid].t;

				//Jsmpl_vec = mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control
				Jsmpl_vec = mySMPL.J_regl_25_bigl_col_ * V_vec;

				sprintf(Fname, "%s/%.2d/%.2d_%.2d/Track3D/%.4d.txt", Path, subId, subId, actionId, tid);
				fp = fopen(Fname, "w");
				for (int ii = 0; ii < nJointsCOCO; ii++)
					fprintf(fp, "%.8f %.8f %.8f\n", Jsmpl_vec(3 * ii), Jsmpl_vec(3 * ii + 1), Jsmpl_vec(3 * ii + 2));
				fclose(fp);
			}
		}
	}

	return true;
}
int SimulateCamerasAnd2DPointsForMOSH(char *Path, int nCams, double *Intrinsic, double *distortion, int width, int height, double radius = 5e3, bool show2DImage = false, int Rate = 1, double PMissingData = 0.0, double Noise2D = 2.0, int *UnSyncFrameTimeStamp = NULL, bool backgroundPoints = false)
{
	int n3DTracks = 24;
	vector<Scalar> colors;
	colors.push_back(Scalar(0, 0, 255));
	colors.push_back(Scalar(0, 128, 255));
	colors.push_back(Scalar(0, 255, 255));
	colors.push_back(Scalar(0, 255, 0));
	colors.push_back(Scalar(255, 128, 0));
	colors.push_back(Scalar(255, 255, 0));
	colors.push_back(Scalar(255, 0, 0));
	colors.push_back(Scalar(255, 0, 255));
	colors.push_back(Scalar(255, 255, 255));

	if (UnSyncFrameTimeStamp == NULL)
	{
		UnSyncFrameTimeStamp = new int[nCams];
		for (int ii = 0; ii < nCams; ii++)
			UnSyncFrameTimeStamp[ii] = 0;
	}

	char Fname[512];
	double noise3D_CamShake = 20 / 60;
	double x, y, z;
	Point2d pt;
	Point3d p3d;
	vector<Point3d> XYZ;

	vector<std::string> vnames;
#ifdef _WINDOWS
	sprintf(Fname, "%s/Track3D", Path);
	vnames = get_all_files_names_within_folder(std::string(Fname));
#endif 
	int nframes = (int)vnames.size();

	vector<Point3d> *allXYZ = new vector<Point3d>[n3DTracks];
	if (nframes == 0)
		return 1;
	else
	{
		for (int fid = 0; fid < nframes; fid++)
		{
			Point3d mxyz(0, 0, 0);
			sprintf(Fname, "%s/Track3D/%.4d.txt", Path, fid); FILE *fp = fopen(Fname, "r");
			for (int tid = 0; tid < n3DTracks; tid++)
			{
				fscanf(fp, "%lf %lf %lf", &x, &y, &z);
				allXYZ[tid].emplace_back(1000.0*x, 1000.0* y, 1000.0*z);
				mxyz += allXYZ[tid].back();
			}
			fclose(fp);

			mxyz /= n3DTracks;
			XYZ.emplace_back(mxyz);
		}
	}

	CameraData *Camera = new CameraData[nframes*nCams];
	vector<int> angleList;
	vector<Point3d> Center;
	for (int frameID = 0; frameID < nframes; frameID++)
	{
		angleList.clear(), Center.clear();
		for (int camID = 0; camID < nCams; camID++)
		{
			int count, angleID;
			while (true)
			{
				count = 0, angleID = 5.0*cos(2.0*Pi / 100 * frameID) + 360 * camID / nCams;
				for (int ii = 0; ii < angleList.size(); ii++)
					if (angleID == angleList[ii])
						count++;
				if (count == 0)
					break;
			}
			angleList.push_back(angleID);

			double theta = 1.0*angleID / 180 * Pi;
			Point3d Noise3D(gaussian_noise(0.0, noise3D_CamShake), gaussian_noise(0.0, noise3D_CamShake), gaussian_noise(0.0, noise3D_CamShake));
			if (Noise3D.x > 3.0*noise3D_CamShake)
				Noise3D.x = 3.0*noise3D_CamShake;
			else if (Noise3D.x < -3.0 *noise3D_CamShake)
				Noise3D.x = -3.0*noise3D_CamShake;
			if (Noise3D.y > 3.0*noise3D_CamShake)
				Noise3D.y = 3.0*noise3D_CamShake;
			else if (Noise3D.y < -3.0 *noise3D_CamShake)
				Noise3D.y = -3.0*noise3D_CamShake;
			if (Noise3D.z > 3.0*noise3D_CamShake)
				Noise3D.z = 3.0*noise3D_CamShake;
			else if (Noise3D.z < -3.0 *noise3D_CamShake)
				Noise3D.z = -3.0*noise3D_CamShake;

			Camera[frameID + nframes * camID].valid = true;
			GenerateCamerasExtrinsicOnCircle(Camera[frameID + nframes * camID], theta, radius, XYZ[frameID], XYZ[frameID], Noise3D);
			SetIntrinisc(Camera[frameID + nframes * camID], Intrinsic);
			GetKFromIntrinsic(Camera[frameID + nframes * camID]);
			for (int ii = 0; ii < 7; ii++)
				Camera[frameID + nframes * camID].distortion[ii] = distortion[ii];
			AssembleP(Camera[frameID + nframes * camID]);
			Center.push_back(Point3d(Camera[frameID + nframes * camID].camCenter[0], Camera[frameID + nframes * camID].camCenter[1], Camera[frameID + nframes * camID].camCenter[2]));
		}
	}

	vector<int> UsedFrames;
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, camID); FILE * fp = fopen(Fname, "w+");
		for (int frameID = 0; frameID < nframes; frameID += Rate)
		{
			if (frameID + UnSyncFrameTimeStamp[camID] >= nframes || !Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].valid)
				continue;

			UsedFrames.push_back(frameID + UnSyncFrameTimeStamp[camID]);
			fprintf(fp, "%d 0 0 %d %d ", frameID / Rate, width, height);
			for (int ii = 0; ii < 5; ii++)
				fprintf(fp, "%f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].intrinsic[ii]);
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%f ", distortion[ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/CamPose_%.4d.txt", Path, camID); FILE *fp = fopen(Fname, "w+");
		for (int frameID = 0; frameID < nframes; frameID += Rate)
		{
			if (frameID + UnSyncFrameTimeStamp[camID] >= nframes || !Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].valid)
				continue;

			GetrtFromRT(Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].rt, Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].R, Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].T);

			fprintf(fp, "%d ", frameID / Rate);
			for (int jj = 0; jj < 6; jj++)
				fprintf(fp, "%.16f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].rt[jj]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*n3DTracks];
	for (int camID = 0; camID < nCams; camID++)
	{
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			int nf = 0;
			for (int frameID = 0; frameID < nframes; frameID += Rate)
			{
				if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
					continue;
				if (frameID + UnSyncFrameTimeStamp[camID] >= nframes)
					continue;
				nf++;
			}

			//Simulate random missing data
			vector<int> randomNumber;
			for (int ii = 0; ii < nf; ii++)
				randomNumber.push_back(ii);
			random_shuffle(randomNumber.begin(), randomNumber.end());

			int nMissingData = (int)(PMissingData*nf);
			sort(randomNumber.begin(), randomNumber.begin() + nMissingData);

			for (int frameID = 0; frameID < nframes; frameID += Rate)
			{
				if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
					continue;
				if (frameID + UnSyncFrameTimeStamp[camID] >= nframes)
					continue;

				ImgPtEle ptEle;
				ptEle.pt2D = Point(0, 0), ptEle.viewID = camID, ptEle.frameID = frameID / Rate, ptEle.imWidth = width, ptEle.imHeight = height;
				ptEle.pt3D = allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]];
				ptEle.timeStamp = 1.0 / Rate * (frameID + UnSyncFrameTimeStamp[camID]);

				bool missed = false;
				for (int ii = 0; ii < nMissingData && !missed; ii++)
					if (randomNumber[ii] == frameID / Rate)
						missed = true;
				if (missed)
				{
					PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
					continue;
				}

				Point3d xyz = allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]];
				ProjectandDistort(xyz, &pt, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID * nframes].P, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID * nframes].K, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID * nframes].distortion);
				Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
				if (Noise.x > 3.0*Noise2D)
					Noise.x = 3.0*Noise2D;
				else if (Noise.x < -3.0 *Noise2D)
					Noise.x = -3.0*Noise2D;
				if (Noise.y > 3.0*Noise2D)
					Noise.y = 3.0*Noise2D;
				else if (Noise.y < -3.0 *Noise2D)
					Noise.y = -3.0*Noise2D;
				pt.x += Noise.x, pt.y += Noise.y;

				if (pt.x < 0 || pt.x > width - 1 || pt.y < 0 || pt.y > height - 1)
				{
					PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
					continue;
				}

				ptEle.pt2D = pt;
				PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
			}
		}
	}

	int nStatPts = 3000;
	Corpus CorpusInfo; CorpusInfo.nCameras = nCams, CorpusInfo.n3dPoints = nStatPts;
	if (backgroundPoints)
	{
		CorpusInfo.xyz.reserve(nStatPts);
		for (int ii = 0; ii < nStatPts; ii++)
		{
			double angle = 2.0*Pi*rand() / RAND_MAX;
			Point3d bg(XYZ[0].x + 15.0*radius*cos(angle), XYZ[0].y + 5.0*radius*sin(0.5*Pi*rand() / RAND_MAX - 0.25*Pi), XYZ[0].z + 15.0*radius*sin(angle));
			CorpusInfo.xyz.push_back(bg);
		}

		vector<int> selectedCamID3D;
		vector<Point2d> uv3D;
		vector<double> scale3D;
		CorpusInfo.viewIdAll3D.reserve(nStatPts);
		CorpusInfo.uvAll3D.reserve(nStatPts);
		CorpusInfo.scaleAll3D.reserve(nStatPts);
		for (int ii = 0; ii < nStatPts; ii++)
		{
			CorpusInfo.viewIdAll3D.push_back(selectedCamID3D); CorpusInfo.viewIdAll3D.back().reserve(nframes*nCams);
			CorpusInfo.uvAll3D.push_back(uv3D); CorpusInfo.uvAll3D.back().reserve(nframes*nCams);
			CorpusInfo.scaleAll3D.push_back(scale3D); CorpusInfo.scaleAll3D.back().reserve(nframes*nCams);
		}

		for (int cid = 0; cid < nCams; cid++)
		{
			sprintf(Fname, "%s/%d", Path, cid);
			makeDir(Fname);
		}

		for (int fid = 0; fid < nframes; fid += Rate)
		{
			for (int cid = 0; cid < nCams; cid++)
			{
				sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, cid, fid / Rate); FILE *fp = fopen(Fname, "w+");
				for (int pid = 0; pid < nStatPts; pid++)
				{
					if (fid + UnSyncFrameTimeStamp[cid] > allXYZ[0].size() - 1)
						continue;
					if (fid + UnSyncFrameTimeStamp[cid] >= nframes)
						continue;

					CameraData *cam = &Camera[fid + UnSyncFrameTimeStamp[cid] + cid * nframes];
					ProjectandDistort(CorpusInfo.xyz[pid], &pt, cam[0].P, cam[0].K, cam[0].distortion);
					Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
					if (Noise.x > 3.0*Noise2D)
						Noise.x = 3.0*Noise2D;
					else if (Noise.x < -3.0 *Noise2D)
						Noise.x = -3.0*Noise2D;
					if (Noise.y > 3.0*Noise2D)
						Noise.y = 3.0*Noise2D;
					else if (Noise.y < -3.0 *Noise2D)
						Noise.y = -3.0*Noise2D;
					pt.x += Noise.x, pt.y += Noise.y;

					if (pt.x < 0 || pt.x > width - 1 || pt.y < 0 || pt.y > height - 1)
						continue;

					CorpusInfo.viewIdAll3D[pid].push_back(cid*nframes + fid / Rate);
					CorpusInfo.uvAll3D[pid].push_back(pt);
					CorpusInfo.scaleAll3D[pid].push_back(1.0);

					fprintf(fp, "%d %.4f %.4f %.4f %.8e %.8e 1.0\n", pid, CorpusInfo.xyz[pid].x, CorpusInfo.xyz[pid].y, CorpusInfo.xyz[pid].z, pt.x, pt.y);
				}
				fclose(fp);
			}
		}

		sprintf(Fname, "%s/Corpus", Path); makeDir(Fname);

		//xyz rgb viewid3D pointid3D 3dId2D cumpoint
		sprintf(Fname, "%s/Corpus/n3dGL.xyz", Path);	FILE *fp = fopen(Fname, "w+");
		for (int jj = 0; jj < CorpusInfo.xyz.size(); jj++)
			fprintf(fp, "%d %lf %lf %lf \n", jj, CorpusInfo.xyz[jj].x, CorpusInfo.xyz[jj].y, CorpusInfo.xyz[jj].z);
		fclose(fp);

		sprintf(Fname, "%s/Corpus/Corpus_3D.txt", Path);
		fp = fopen(Fname, "w+");
		CorpusInfo.n3dPoints = (int)CorpusInfo.xyz.size();
		fprintf(fp, "%d %d ", CorpusInfo.nCameras, CorpusInfo.n3dPoints);

		//xyz rgb viewid3D pointid3D 3dId2D cumpoint
		fprintf(fp, "0\n");
		for (int jj = 0; jj < CorpusInfo.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf \n", CorpusInfo.xyz[jj].x, CorpusInfo.xyz[jj].y, CorpusInfo.xyz[jj].z);
		fclose(fp);

		sprintf(Fname, "%s/Corpus/Corpus_viewIdAll3D.txt", Path); fp = fopen(Fname, "w+");
		for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int nviews = (int)CorpusInfo.viewIdAll3D[jj].size();
			fprintf(fp, "%d ", nviews);
			for (int ii = 0; ii < nviews; ii++)
				fprintf(fp, "%d ", CorpusInfo.viewIdAll3D[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus/Corpus_uvAll3D.txt", Path); fp = fopen(Fname, "w+");
		for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.uvAll3D[jj].size();
			fprintf(fp, "%d ", npts);
			for (int ii = 0; ii < npts; ii++)
				fprintf(fp, "%8f %8f %.2f ", CorpusInfo.uvAll3D[jj][ii].x, CorpusInfo.uvAll3D[jj][ii].y, CorpusInfo.scaleAll3D[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	/*sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, camID); fp = fopen(Fname, "w+");
		fprintf(fp, "%d\n", n3DTracks);
		for (int trackID = 0; trackID < n3DTracks; trackID++)
		{
			fprintf(fp, "%d %d ", trackID, (int)PerCam_UV[camID*n3DTracks + trackID].size());
			for (int fid = 0; fid < (int)PerCam_UV[camID*n3DTracks + trackID].size(); fid++)
				fprintf(fp, "%d %.4f %.4f 1.0 ", PerCam_UV[camID*n3DTracks + trackID][fid].frameID, PerCam_UV[camID*n3DTracks + trackID][fid].pt2D.x, PerCam_UV[camID*n3DTracks + trackID][fid].pt2D.y);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}*/
	sprintf(Fname, "%s/MP", Path), makeDir(Fname);
	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname, "%s/MP/%d", Path, camID), makeDir(Fname);
		sprintf(Fname, "%s/MP/_%d", Path, camID), makeDir(Fname);
		for (int id = 0; id < PerCam_UV[camID*n3DTracks].size(); id++)
		{
			Point2d uv[25];
			for (int trackID = 0; trackID < 8; trackID++)
				uv[trackID] = PerCam_UV[camID*n3DTracks + trackID][id].pt2D;
			uv[8] = Point2d(0, -0);
			for (int trackID = 8; trackID < n3DTracks; trackID++)
				uv[trackID + 1] = PerCam_UV[camID*n3DTracks + trackID][id].pt2D;
			swap(uv[22], uv[19]);
			swap(uv[23], uv[20]);
			swap(uv[24], uv[21]);

			sprintf(Fname, "%s/MP/%d/%.4d.txt", Path, camID, PerCam_UV[camID*n3DTracks][id].frameID); FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < 25; ii++)
				fprintf(fp, "%.4f %.4f 1.0\n", uv[ii].x, uv[ii].y);
			fclose(fp);

			if (show2DImage)
			{
				Mat Img(height, width, CV_8UC3, Scalar(0, 0, 0)), displayImg;
				displayImg = Img.clone();
				Draw2DCoCoJoints(displayImg, uv, 25);

				sprintf(Fname, "Cam %d: frame %d", camID, PerCam_UV[camID*n3DTracks][id].frameID);
				CvPoint text_origin = { width / 30, height / 30 };
				putText(displayImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);

				sprintf(Fname, "%s/MP/_%d/%.4d.png", Path, camID, PerCam_UV[camID*n3DTracks][id].frameID);
				imwrite(Fname, displayImg);
			}
		}
	}

	return 0;
}
bool CreateBodyJointFromSeq(char *Path, int nSub, int nActions, int nCams, double *Intrinsic, double *distortion, int width, int height, double radius = 5e3, bool show2DImage = false, int Rate = 1, double PMissingData = 0.0, double Noise2D = 2.0, int *UnSyncFrameTimeStamp = NULL, bool backgroundPoints = false)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 24;
	int n3DTracks = nJointsCOCO;


	vector<Scalar> colors;
	colors.push_back(Scalar(0, 0, 255));
	colors.push_back(Scalar(0, 128, 255));
	colors.push_back(Scalar(0, 255, 255));
	colors.push_back(Scalar(0, 255, 0));
	colors.push_back(Scalar(255, 128, 0));
	colors.push_back(Scalar(255, 255, 0));
	colors.push_back(Scalar(255, 0, 0));
	colors.push_back(Scalar(255, 0, 255));
	colors.push_back(Scalar(255, 255, 255));

	char Fname[512];
	SparseMatrix<double, ColMajor> dVdt;
	SMPLModel mySMPL;
	if (!ReadSMPLData("smpl", mySMPL))
		printLOG("Check smpl Path.\n");
	else
	{
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		dVdt = Eigen::kroneckerProduct(VectorXd::Ones(smpl::SMPLModel::nVertices), eye3);
	}
	MatrixXdr AllV(SMPLModel::nVertices, 3);

	SMPLParams params;
	vector< SMPLParams> VParams;
	VectorXd Jsmpl_vec;

	for (int subId = 1; subId <= nSub; subId++)
	{
		for (int actionId = 1; actionId <= nActions; actionId++)
		{
			printLOG("\n%d, %d:\n", subId, actionId);

			VParams.clear();
			sprintf(Fname, "%s/%.2d/%.2d_%.2d.txt", Path, subId, subId, actionId);
			if (IsFileExist(Fname) == 0)
				continue;

			FILE *fp = fopen(Fname, "r");
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fscanf(fp, "%lf ", &params.coeffs(ii));
			int tid;
			while (fscanf(fp, "%d ", &tid) != EOF)
			{
				params.scale = 1.0;
				fscanf(fp, "%lf %lf %lf ", &params.t(0), &params.t(1), &params.t(2));
				for (int ii = 0; ii < nJointsSMPL; ii++)
					fscanf(fp, "%lf %lf %lf ", &params.pose(ii, 0), &params.pose(ii, 1), &params.pose(ii, 2));
				VParams.push_back(params);
				if (VParams.size() > 30)
					break;
			}
			fclose(fp);

			sprintf(Fname, "%s/%.2d/%.2d_%.2d", Path, subId, subId, actionId), makeDir(Fname);

			vector<Point3d> *allXYZ = new vector<Point3d>[n3DTracks];

			for (int tid = 0; tid < VParams.size(); tid++)
			{
				printLOG("%d..", tid);
				reconstruct(mySMPL, VParams[tid].coeffs.data(), VParams[tid].pose.data(), AllV.data());
				Map<VectorXd> V_vec(AllV.data(), AllV.size());
				V_vec = V_vec * VParams[tid].scale + dVdt * VParams[tid].t;

				Jsmpl_vec = mySMPL.J_regl_25_bigl_col_ * V_vec; //agumented joints for better smothing control

				for (int ii = 0; ii < nJointsCOCO; ii++)
					allXYZ[ii].emplace_back(1000.0*Jsmpl_vec(3 * ii), 1000.0*Jsmpl_vec(3 * ii + 1), 1000.0*Jsmpl_vec(3 * ii + 2));
			}

			if (UnSyncFrameTimeStamp == NULL)
			{
				UnSyncFrameTimeStamp = new int[nCams];
				for (int ii = 0; ii < nCams; ii++)
					UnSyncFrameTimeStamp[ii] = 0;
			}

			double  x, y, z, noise3D_CamShake = 20 / 60;
			int nframes = (int)VParams.size();

			CameraData *Camera = new CameraData[nframes*nCams];
			vector<int> angleList;
			vector<Point3d> Center;
			for (int frameID = 0; frameID < nframes; frameID++)
			{
				angleList.clear(), Center.clear();
				for (int camID = 0; camID < nCams; camID++)
				{
					int count, angleID;
					while (true)
					{
						count = 0, angleID = 5.0*cos(2.0*Pi / 100 * frameID) + 360 * camID / nCams;
						for (int ii = 0; ii < angleList.size(); ii++)
							if (angleID == angleList[ii])
								count++;
						if (count == 0)
							break;
					}
					angleList.push_back(angleID);

					double theta = 1.0*angleID / 180 * Pi;
					Point3d Noise3D(gaussian_noise(0.0, noise3D_CamShake), gaussian_noise(0.0, noise3D_CamShake), gaussian_noise(0.0, noise3D_CamShake));
					if (Noise3D.x > 3.0*noise3D_CamShake)
						Noise3D.x = 3.0*noise3D_CamShake;
					else if (Noise3D.x < -3.0 *noise3D_CamShake)
						Noise3D.x = -3.0*noise3D_CamShake;
					if (Noise3D.y > 3.0*noise3D_CamShake)
						Noise3D.y = 3.0*noise3D_CamShake;
					else if (Noise3D.y < -3.0 *noise3D_CamShake)
						Noise3D.y = -3.0*noise3D_CamShake;
					if (Noise3D.z > 3.0*noise3D_CamShake)
						Noise3D.z = 3.0*noise3D_CamShake;
					else if (Noise3D.z < -3.0 *noise3D_CamShake)
						Noise3D.z = -3.0*noise3D_CamShake;

					Camera[frameID + nframes * camID].valid = true;
					GenerateCamerasExtrinsicOnCircle(Camera[frameID + nframes * camID], theta, radius, allXYZ[frameID][0], allXYZ[frameID][0], Noise3D);
					SetIntrinisc(Camera[frameID + nframes * camID], Intrinsic);
					GetKFromIntrinsic(Camera[frameID + nframes * camID]);
					for (int ii = 0; ii < 7; ii++)
						Camera[frameID + nframes * camID].distortion[ii] = distortion[ii];
					AssembleP(Camera[frameID + nframes * camID]);
					Center.push_back(Point3d(Camera[frameID + nframes * camID].camCenter[0], Camera[frameID + nframes * camID].camCenter[1], Camera[frameID + nframes * camID].camCenter[2]));
				}
			}

			Point2d pt;
			Point3d p3d;
			vector<int> UsedFrames;
			for (int camID = 0; camID < nCams; camID++)
			{
				sprintf(Fname, "%s/%.2d/%.2d_%.2d/Intrinsic_%.4d.txt", Path, subId, subId, actionId, camID); FILE * fp = fopen(Fname, "w+");
				for (int frameID = 0; frameID < nframes; frameID += Rate)
				{
					if (frameID + UnSyncFrameTimeStamp[camID] > nframes || !Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].valid)
						continue;

					UsedFrames.push_back(frameID + UnSyncFrameTimeStamp[camID]);
					fprintf(fp, "%d 0 0 %d %d ", frameID / Rate, width, height);
					for (int ii = 0; ii < 5; ii++)
						fprintf(fp, "%f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].intrinsic[ii]);
					for (int ii = 0; ii < 7; ii++)
						fprintf(fp, "%f ", distortion[ii]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}
			for (int camID = 0; camID < nCams; camID++)
			{
				sprintf(Fname, "%s/%.2d/%.2d_%.2d/CamPose_%.4d.txt", Path, subId, subId, actionId, camID); FILE *fp = fopen(Fname, "w+");
				for (int frameID = 0; frameID < nframes; frameID += Rate)
				{
					if (frameID + UnSyncFrameTimeStamp[camID] > nframes || !Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].valid)
						continue;

					GetrtFromRT(Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].rt, Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].R, Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].T);

					fprintf(fp, "%d ", frameID / Rate);
					for (int jj = 0; jj < 6; jj++)
						fprintf(fp, "%.16f ", Camera[frameID + UnSyncFrameTimeStamp[camID] + nframes * camID].rt[jj]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}

			vector<ImgPtEle> *PerCam_UV = new vector<ImgPtEle>[nCams*n3DTracks];
			for (int camID = 0; camID < nCams; camID++)
			{
				for (int trackID = 0; trackID < n3DTracks; trackID++)
				{
					int nf = 0;
					for (int frameID = 0; frameID < nframes; frameID += Rate)
					{
						if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
							continue;
						if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
							continue;
						nf++;
					}

					//Simulate random missing data
					vector<int> randomNumber;
					for (int ii = 0; ii < nf; ii++)
						randomNumber.push_back(ii);
					random_shuffle(randomNumber.begin(), randomNumber.end());

					int nMissingData = (int)(PMissingData*nf);
					sort(randomNumber.begin(), randomNumber.begin() + nMissingData);

					for (int frameID = 0; frameID < nframes; frameID += Rate)
					{
						if (frameID + UnSyncFrameTimeStamp[camID] > allXYZ[trackID].size() - 1)
							continue;
						if (frameID + UnSyncFrameTimeStamp[camID] > nframes)
							continue;

						ImgPtEle ptEle;
						ptEle.pt2D = Point(0, 0), ptEle.viewID = camID, ptEle.frameID = frameID / Rate, ptEle.imWidth = width, ptEle.imHeight = height;
						ptEle.pt3D = allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]];
						ptEle.timeStamp = frameID + UnSyncFrameTimeStamp[camID];

						bool missed = false;
						for (int ii = 0; ii < nMissingData && !missed; ii++)
							if (randomNumber[ii] == frameID / Rate)
								missed = true;
						if (missed)
						{
							PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
							continue;
						}

						Point3d xyz = allXYZ[trackID][frameID + UnSyncFrameTimeStamp[camID]];
						ProjectandDistort(xyz, &pt, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID * nframes].P, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID * nframes].K, Camera[frameID + UnSyncFrameTimeStamp[camID] + camID * nframes].distortion);
						Point2d Noise(gaussian_noise(0.0, Noise2D), gaussian_noise(0.0, Noise2D));
						if (Noise.x > 3.0*Noise2D)
							Noise.x = 3.0*Noise2D;
						else if (Noise.x < -3.0 *Noise2D)
							Noise.x = -3.0*Noise2D;
						if (Noise.y > 3.0*Noise2D)
							Noise.y = 3.0*Noise2D;
						else if (Noise.y < -3.0 *Noise2D)
							Noise.y = -3.0*Noise2D;
						pt.x += Noise.x, pt.y += Noise.y;

						if (pt.x < 0 || pt.x > width - 1 || pt.y < 0 || pt.y > height - 1)
						{
							PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
							continue;
						}

						ptEle.pt2D = pt;
						PerCam_UV[camID*n3DTracks + trackID].push_back(ptEle);
					}
				}
			}

			sprintf(Fname, "%s/%.2d/%.2d_%.2d/MP", Path, subId, subId, actionId), makeDir(Fname);
			for (int camID = 0; camID < nCams; camID++)
			{
				sprintf(Fname, "%s/%.2d/%.2d_%.2d/MP/%d", Path, subId, subId, actionId, camID), makeDir(Fname);
				sprintf(Fname, "%s/%.2d/%.2d_%.2d/MP/_%d", Path, subId, subId, actionId, camID), makeDir(Fname);
				for (int id = 0; id < PerCam_UV[camID*n3DTracks].size(); id++)
				{
					Point2d uv[25];
					for (int trackID = 0; trackID < 8; trackID++)
						uv[trackID] = PerCam_UV[camID*n3DTracks + trackID][id].pt2D;
					uv[8] = Point2d(0, -0);
					for (int trackID = 8; trackID < n3DTracks; trackID++)
						uv[trackID + 1] = PerCam_UV[camID*n3DTracks + trackID][id].pt2D;
					swap(uv[22], uv[19]);
					swap(uv[23], uv[20]);
					swap(uv[24], uv[21]);

					sprintf(Fname, "%s/%.2d/%.2d_%.2d/MP/%d/%.4d.txt", Path, subId, subId, actionId, camID, PerCam_UV[camID*n3DTracks][id].frameID);  FILE *fp = fopen(Fname, "w+");
					for (int ii = 0; ii < 25; ii++)
						fprintf(fp, "%.4f %.4f 1.0\n", uv[ii].x, uv[ii].y);
					fclose(fp);

					if (show2DImage)
					{
						Mat Img(height, width, CV_8UC3, Scalar(0, 0, 0)), displayImg;
						//namedWindow("Image", CV_WINDOW_NORMAL);

						displayImg = Img.clone();
						Draw2DCoCoJoints(displayImg, uv, 25);

						sprintf(Fname, "Cam %d: frame %d", camID, PerCam_UV[camID*n3DTracks][id].frameID);
						CvPoint text_origin = { width / 30, height / 30 };
						putText(displayImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 3.0 * 640 / Img.cols, CV_RGB(255, 0, 0), 2);

						sprintf(Fname, "%s/%.2d/%.2d_%.2d/MP/_%d/%.4d.png", Path, subId, subId, actionId, camID, PerCam_UV[camID*n3DTracks][id].frameID);
						imwrite(Fname, displayImg);
						//imshow("Image", displayImg);
						//waitKey(1);
					}
				}
			}

			delete[]Camera;
			delete[]allXYZ;
			delete[]PerCam_UV;
		}
	}

	return 0;
}

int FitSMPLWindowMOSH(char *Path, SMPLModel &mySMPL, HumanSkeleton3D *vSkeleton, DensePose *AllvDensePose, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int startF, int stopF, int increF, Point2i &fixedPoseFrame, int Pid, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose)
{
	const double TimeScale = 12.0;

	char Fname[512];
	sprintf(Fname, "%s/FitBody/@%d", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/Wj", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/Wj/%d", Path, increF, Pid); makeDir(Fname);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;
	int maxWidth = 0, maxHeight = 0, nsCams = vSCams.size();
	for (int lcid = 0; lcid < nsCams; lcid++)
		maxWidth = max(maxWidth, VideoInfo[vSCams[lcid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vSCams[lcid]].VideoInfo[0].height);

	int nthreads = omp_get_max_threads();
	omp_set_num_threads(nthreads);

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 17) //org CoCo, specify based on the smpl keypoint Id
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 2, JointWeight[15] = 2, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .1, JointWeight[19] = .1, JointWeight[20] = .1,//rFoot
		JointWeight[21] = .1, JointWeight[22] = .1, JointWeight[23] = .1;//lFoot
	else if (skeletonPointFormat == 18)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .5, JointWeight[19] = .5, JointWeight[20] = .5,//rFoot
		JointWeight[21] = .5, JointWeight[22] = .5, JointWeight[23] = .5;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		///vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	SMPLParams ParaI;
	double smpl2sfmScale = 1000.0;
	vector<int> vsyncFid;
	vector<double> vscale;
	vector<SMPLParams> frame_params;
	vector<HumanSkeleton3D> frame_skeleton;

	Mat IUV, INDS;
	vector<int> vDP_Vid;
	vector<Point2d> vDP_uv;
	float *Para = new float[maxWidth*maxHeight];

	vector<SMPLParams> in_frame_params;
	sprintf(Fname, "%s/BodyParas.txt", Path); FILE *fp = fopen(Fname, "r");
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		fscanf(fp, "%lf ", &ParaI.coeffs(ii));
	ParaI.scale = smpl2sfmScale;
	int tid;
	while (fscanf(fp, "%d ", &tid) != EOF)
	{
		fscanf(fp, "%lf %lf %lf ", &ParaI.t(0), &ParaI.t(1), &ParaI.t(2));
		ParaI.t = ParaI.t * smpl2sfmScale;
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &ParaI.pose(ii, 0), &ParaI.pose(ii, 1), &ParaI.pose(ii, 2));
		in_frame_params.push_back(ParaI);
	}
	fclose(fp);

	//pre-process the body paras 
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int temp = (refFid - startF) / increF;
		int inst = round(1.0*refFid / CamTimeInfo[0].x);
		if (inst > in_frame_params.size() - 1)
			continue;
		for (int ii = 0; ii < nJointsSMPL; ii++)
			ParaI.pose(ii, 0) = in_frame_params[inst].pose(ii, 0), ParaI.pose(ii, 1) = in_frame_params[inst].pose(ii, 1), ParaI.pose(ii, 2) = in_frame_params[inst].pose(ii, 2);
		ParaI.t = in_frame_params[inst].t;

		ParaI.frame = refFid;
		ParaI.scale = smpl2sfmScale;

		vscale.push_back(smpl2sfmScale);
		vsyncFid.push_back(refFid);
		frame_params.push_back(ParaI);
		frame_skeleton.push_back(vSkeleton[temp]);
	}

	if (vscale.size() > 0)
	{
		smpl2sfmScale = MedianArray(vscale); //robusifier

		 //init smpl
		mySMPL.t.setZero(), mySMPL.pose.setZero(), mySMPL.coeffs.setZero();
		mySMPL.scale = smpl2sfmScale;
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);

		if (fixedPoseFrame.x > -1)
		{
			for (int jj = 0; jj < vsyncFid.size(); jj++)
			{
				if (vsyncFid[jj] >= fixedPoseFrame.x &&vsyncFid[jj] <= fixedPoseFrame.y)
				{
					for (int ii = 0; ii < nShapeCoeffs; ii++)
						mySMPL.coeffs(ii) = frame_params[jj].coeffs(ii);
					mySMPL.scale = frame_params[jj].scale;
					break;
				}
			}
		}

		FitSMPL2Total(mySMPL, frame_params, frame_skeleton, AllvDensePose, CamTimeInfo, VideoInfo, vSCams, ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, Real2SfM, skeletonPointFormat, fixedPoseFrame, Pid, hasDensePose);

		for (size_t jj = 0; jj < frame_skeleton.size(); jj++)
		{
			sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, 0, vsyncFid[jj], TimeScale* vsyncFid[jj]);
			FILE *fp = fopen(Fname, "w+");
			fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[jj].t(0), frame_params[jj].t(1), frame_params[jj].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fprintf(fp, "%f %f %f\n", frame_params[jj].pose(ii, 0), frame_params[jj].pose(ii, 1), frame_params[jj].pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fprintf(fp, "%f ", mySMPL.coeffs(ii));
			fclose(fp);
		}
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;
	delete[]Para;

	return 0;
}
int FitSMPLUnSyncMOSH(char *Path, SMPLModel &mySMPL, HumanSkeleton3D *vSkeleton, DensePose *AllvDensePose, VideoData *VideoInfo, Point3d *CamTimeInfo, vector<int> &vSCams, int startF, int stopF, int increF, Point2i &fixedPoseFrame, int Pid, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, double *CostWeights, double *isigmas, double Real2SfM, bool hasDensePose)
{
	const double TimeScale = 12.0;

	char Fname[512];
	sprintf(Fname, "%s/FitBody", Path), makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/US_Smoothing", Path, increF); makeDir(Fname);
	sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d", Path, increF, Pid); makeDir(Fname);

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;
	int maxWidth = 0, maxHeight = 0, nsCams = vSCams.size();
	for (int lcid = 0; lcid < nsCams; lcid++)
		maxWidth = max(maxWidth, VideoInfo[vSCams[lcid]].VideoInfo[0].width), maxHeight = max(maxHeight, VideoInfo[vSCams[lcid]].VideoInfo[0].height);

	int nthreads = omp_get_max_threads();
	omp_set_num_threads(nthreads);

	double JointWeight[24], ContourPartWeight[24], DPPartweight[24];
	if (skeletonPointFormat == 18)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0; //lLeg
	else if (skeletonPointFormat == 25)
		JointWeight[0] = 1, JointWeight[1] = 1, //noise-neck
		JointWeight[2] = 1, JointWeight[3] = 2.75, JointWeight[4] = 10.0, //rArm
		JointWeight[5] = 1, JointWeight[6] = 2.75, JointWeight[7] = 10.0, //lArm
		JointWeight[8] = 1, JointWeight[9] = 2.75, JointWeight[10] = 10.0, //rLeg
		JointWeight[11] = 1, JointWeight[12] = 2.75, JointWeight[13] = 10.0, //lLeg
		JointWeight[14] = 1, JointWeight[15] = 1, JointWeight[16] = 1, JointWeight[17] = 1,//face
		JointWeight[18] = .5, JointWeight[19] = .5, JointWeight[20] = .5,//rFoot
		JointWeight[21] = .5, JointWeight[22] = .5, JointWeight[23] = .5;//lFoot

	ContourPartWeight[0] = 1, ContourPartWeight[1] = 1, //back and front torso
		ContourPartWeight[2] = 3.0, ContourPartWeight[3] = 3.0,//right and left hands
		ContourPartWeight[4] = 3.0, ContourPartWeight[5] = 3.0,//left and right feet
		ContourPartWeight[6] = 1.5, ContourPartWeight[7] = 1.5, ContourPartWeight[8] = 1.5, ContourPartWeight[9] = 1.5,//right and left back and right and left upper leg
		ContourPartWeight[10] = 2.0, ContourPartWeight[11] = 2.0, ContourPartWeight[12] = 2.0, ContourPartWeight[13] = 2.0,//right and left back lower leg and //right and left front lower leg
		ContourPartWeight[14] = 1.5, ContourPartWeight[15] = 1.5, ContourPartWeight[16] = 1.5, ContourPartWeight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		ContourPartWeight[18] = 2.0, ContourPartWeight[19] = 2.0, ContourPartWeight[20] = 2.0, ContourPartWeight[21] = 2.0,//back left and  right lower arm and front left and left lower arm
		ContourPartWeight[22] = 1.5, ContourPartWeight[23] = 1.5;//right and left face

	DPPartweight[0] = 1, DPPartweight[1] = 1, //back and front torso
		DPPartweight[2] = 10.0, DPPartweight[3] = 10.0,//right and left hands
		DPPartweight[4] = 10.0, DPPartweight[5] = 10.0,//left and right feet
		DPPartweight[6] = 1.5, DPPartweight[7] = 1.5, DPPartweight[8] = 1.5, DPPartweight[9] = 1.5,//right and left back and right and left front upper leg
		DPPartweight[10] = 2.75, DPPartweight[11] = 2.75, DPPartweight[12] = 2.75, DPPartweight[13] = 2.75,//right and left back lower leg and //right and left front lower leg
		DPPartweight[14] = 1.5, DPPartweight[15] = 1.5, DPPartweight[16] = 1.5, DPPartweight[17] = 1.5,//front  and back left upper arm and back right and left upper arm
		DPPartweight[18] = 2.75, DPPartweight[19] = 2.75, DPPartweight[20] = 2.75, DPPartweight[21] = 2.75,//back left and  right lower arm and front left and left lower arm
		DPPartweight[22] = 2.0, DPPartweight[23] = 2.0;//right and left face

	vector<uchar> vMergedPartId2DensePosePartId[14];
	vMergedPartId2DensePosePartId[0].push_back(1), vMergedPartId2DensePosePartId[0].push_back(2), //torso
		//vMergedPartId2DensePosePartId[1].push_back(4), //l hand
		//vMergedPartId2DensePosePartId[2].push_back(3), //r hand
		vMergedPartId2DensePosePartId[3].push_back(19), vMergedPartId2DensePosePartId[3].push_back(21), //l lower arm
		vMergedPartId2DensePosePartId[4].push_back(20), vMergedPartId2DensePosePartId[4].push_back(22), // r lower arm
		vMergedPartId2DensePosePartId[5].push_back(15), vMergedPartId2DensePosePartId[5].push_back(17), //l upper arm
		vMergedPartId2DensePosePartId[6].push_back(16), vMergedPartId2DensePosePartId[6].push_back(18), // r upper arm
		//vMergedPartId2DensePosePartId[7].push_back(5),//l foot
		//vMergedPartId2DensePosePartId[8].push_back(6), //r foot
		vMergedPartId2DensePosePartId[9].push_back(12), vMergedPartId2DensePosePartId[9].push_back(14),//l lower foot
		vMergedPartId2DensePosePartId[10].push_back(11), vMergedPartId2DensePosePartId[10].push_back(13),//r lower foot
		vMergedPartId2DensePosePartId[11].push_back(8), vMergedPartId2DensePosePartId[11].push_back(10), // l upper foot
		vMergedPartId2DensePosePartId[12].push_back(7), vMergedPartId2DensePosePartId[12].push_back(9),//r upper foot
		vMergedPartId2DensePosePartId[13].push_back(23), vMergedPartId2DensePosePartId[13].push_back(24); //face

	int *outSideEdge = new int[maxWidth * maxHeight],
		*PartEdge = new int[maxWidth * maxHeight],
		*BinarizeData = new int[maxWidth * maxHeight],
		*ADTTps = new int[maxWidth*maxHeight],
		*realADT = new int[maxWidth*maxHeight];
	double *v = new double[maxWidth*maxHeight],
		*z = new double[maxWidth*maxHeight],
		*DTTps = new double[maxWidth*maxHeight];
	float *float_df = new float[maxWidth*maxHeight];

	int refCid = 0;
	double earliest = DBL_MAX;
	for (auto cid : vSCams)
		if (earliest > CamTimeInfo[cid].y)
			earliest = CamTimeInfo[cid].y, refCid = cid;

	SMPLParams ParaI;
	double smpl2sfmScale = 1000.0;
	vector<double> vscale;
	vector<SMPLParams> frame_params;
	vector<ImgPoseEle> frame_skeleton;

	vector<SMPLParams> in_frame_params;
	sprintf(Fname, "%s/BodyParas.txt", Path); FILE *fp = fopen(Fname, "r");
	for (int ii = 0; ii < nShapeCoeffs; ii++)
		fscanf(fp, "%lf ", &ParaI.coeffs(ii));
	ParaI.scale = smpl2sfmScale;
	int tid;
	while (fscanf(fp, "%d ", &tid) != EOF)
	{
		fscanf(fp, "%lf %lf %lf ", &ParaI.t(0), &ParaI.t(1), &ParaI.t(2));
		ParaI.t = ParaI.t * smpl2sfmScale;
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &ParaI.pose(ii, 0), &ParaI.pose(ii, 1), &ParaI.pose(ii, 2));
		in_frame_params.push_back(ParaI);
	}
	fclose(fp);

	Mat IUV, INDS;
	vector<int> vDP_Vid;
	vector<Point2d> vDP_uv;
	for (int reffid = startF; reffid <= stopF; reffid += increF)
	{
		for (int ii = 0; ii < vSCams.size(); ii++)
		{
			int cid = vSCams[ii], rcid, temp = (reffid - startF) / increF, found = 0;
			if (!vSkeleton[temp].valid)
				continue;

			ImgPoseEle skeI(skeletonPointFormat);
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				skeI.pt2D[jid] = Point2d(0, 0), skeI.confidence[jid] = -1.0;
				for (int ii = 0; ii < vSkeleton[temp].vViewID_rFid[jid].size(); ii++)
				{
					int rcid = vSkeleton[temp].vViewID_rFid[jid][ii].x, rfid = vSkeleton[temp].vViewID_rFid[jid][ii].y;
					CameraData *Cam = VideoInfo[rcid].VideoInfo;
					if (rcid != cid || !Cam[rfid].valid)
						continue;

					found = 1;
					skeI.pt2D[jid] = vSkeleton[temp].vPt2D[jid][ii], skeI.confidence[jid] = vSkeleton[temp].vConf[jid][ii];
					skeI.viewID = rcid, skeI.frameID = rfid, skeI.refFrameID = reffid;
					skeI.ts = (CamTimeInfo[rcid].y / CamTimeInfo[refCid].x + 1.0*rfid / CamTimeInfo[rcid].x)*CamTimeInfo[refCid].x;

					AssembleP(Cam[rfid].K, Cam[rfid].R, Cam[rfid].T, skeI.P + jid * 12);
				}
			}
			if (found == 0)
				continue;

			int inst = round(TimeScale*skeI.ts);
			for (int ii = 0; ii < nJointsSMPL; ii++)
				ParaI.pose(ii, 0) = in_frame_params[inst].pose(ii, 0), ParaI.pose(ii, 1) = in_frame_params[inst].pose(ii, 1), ParaI.pose(ii, 2) = in_frame_params[inst].pose(ii, 2);
			ParaI.t = in_frame_params[inst].t;
			ParaI.frame = reffid;

			vscale.push_back(smpl2sfmScale);
			frame_params.push_back(ParaI);
			frame_skeleton.push_back(skeI);
		}
	}

	if (vscale.size() > 0)
	{
		smpl2sfmScale = MedianArray(vscale); //robusifier

		 //init smpl
		mySMPL.t.setZero(), mySMPL.pose.setZero(), mySMPL.coeffs.setZero();
		mySMPL.scale = smpl2sfmScale;
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			mySMPL.coeffs(ii) = frame_params[0].coeffs(ii);

		if (fixedPoseFrame.x > -1)
		{
			for (int jj = 0; jj < frame_skeleton.size(); jj++)
			{
				if (frame_skeleton[jj].refFrameID >= fixedPoseFrame.x && frame_skeleton[jj].refFrameID <= fixedPoseFrame.y)
				{
					for (int ii = 0; ii < nShapeCoeffs; ii++)
						mySMPL.coeffs(ii) = frame_params[jj].coeffs(ii);
					mySMPL.scale = frame_params[jj].scale;
					break;
				}
			}
		}
		/*bool breakflag = false;
		for (size_t jj = 0; jj < frame_skeleton.size() && !breakflag; jj++)
		{
			sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, frame_skeleton[jj].viewID, frame_skeleton[jj].frameID, round(TimeScale* frame_skeleton[jj].ts));
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				breakflag = true;
				break;
			}
			fscanf(fp, "%lf %lf %lf %lf\n", &mySMPL.scale, &frame_params[jj].t(0), &frame_params[jj].t(1), &frame_params[jj].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &frame_params[jj].pose(ii, 0), &frame_params[jj].pose(ii, 1), &frame_params[jj].pose(ii, 2));
			fclose(fp);
		}
		if (!breakflag)*/
		FitSMPL2TotalUnSync(mySMPL, frame_params, frame_skeleton, AllvDensePose, CamTimeInfo, VideoInfo, vSCams,
			ContourPartWeight, JointWeight, DPPartweight, CostWeights, isigmas, Real2SfM, skeletonPointFormat, fixedPoseFrame, Pid, hasDensePose);

		for (size_t jj = 0; jj < frame_skeleton.size(); jj++)
		{
			sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d/%.2d_%.4d_%.1f.txt", Path, increF, Pid, frame_skeleton[jj].viewID, frame_skeleton[jj].frameID, round(TimeScale* frame_skeleton[jj].ts));
			FILE *fp = fopen(Fname, "w+");
			fprintf(fp, "%e %e %e %e\n", mySMPL.scale, frame_params[jj].t(0), frame_params[jj].t(1), frame_params[jj].t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fprintf(fp, "%f %f %f\n", frame_params[jj].pose(ii, 0), frame_params[jj].pose(ii, 1), frame_params[jj].pose(ii, 2));
			for (int ii = 0; ii < nShapeCoeffs; ii++)
				fprintf(fp, "%f ", mySMPL.coeffs(ii));
			fclose(fp);
		}
	}

	delete[]outSideEdge, delete[]PartEdge, delete[]BinarizeData;
	delete[]v, delete[]z, delete[]DTTps, delete[]ADTTps, delete[]realADT, delete[]float_df;

	return 0;
}
int FitSMPLWindowMOSHDriver(char *Path, std::vector<int> &vsCams, int *TS, int startF, int stopF, int winSize, int nOverlappingFrames, int increF, int distortionCorrected, int sharedIntrinisc, int skeletonPointFormat, int syncMode, double *CostWeights, double *isigmas, double Real2SfM, double detectionThresh, bool hasDensePose, int selectedPeopleID = -1, int startChunkdId = 0)
{
	char Fname[512];
	int nChunks = (stopF - startF + 1) / winSize + 1;
	const int  nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;

	int nSCams = vsCams.size(), nGCams = *max_element(vsCams.begin(), vsCams.end()) + 1;

	printLOG("Reading all camera poses\n");
	int nvalidCams = 0;
	VideoData *VideoInfo = new VideoData[nGCams];
	for (int lcid = 0; lcid < nSCams; lcid++)
	{
		printLOG("Cam %d ...validating ", lcid);
		if (ReadVideoDataI(Path, VideoInfo[vsCams[lcid]], vsCams[lcid], -1, -1) == 1)
			continue;
		nvalidCams++;
		printLOG("\n");
	}
	if (nvalidCams < 2)
		return 1;


	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nGCams; ii++)
		CamTimeInfo[ii].x = 1.0 / 12.0, CamTimeInfo[ii].y = 1.0*TS[ii] / 12.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	SMPLModel smplMaster;
	if (!ReadSMPLData("smpl", smplMaster))
	{
		printLOG("Check smpl Path.\n");
		return 1;
	}

	double u, v, s;
	int  rcid, nPeople = 0;
	vector<HumanSkeleton3D *> vSkeleton;

	printLOG("Reading all people 3D skeleton: ");
	int nvalidFrames = 0;
	HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
	for (int refFid = startF; refFid <= stopF; refFid += increF)
		Skeletons[refFid].valid = 0;

	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int  temp = (refFid - startF) / increF;

		for (int jid = 0; jid < skeletonPointFormat; jid++)
			Skeletons[temp].pt3d[jid] = Point3d(0, 0, 0), Skeletons[temp].validJoints[jid] = 1;

		for (int rcid = 0; rcid < nGCams; rcid++)
		{
			sprintf(Fname, "%s/MP/%d/%.4d.txt", Path, rcid, refFid);
			if (!IsFileExist(Fname))
				continue;
			FILE *fp = fopen(Fname, "r");
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				fscanf(fp, "%lf %lf %lf ", &u, &v, &s);
				if (u < 1 || v < 1 || u>VideoInfo[rcid].VideoInfo[refFid].width - 1 || v>VideoInfo[rcid].VideoInfo[refFid].height - 1 || s < detectionThresh || !VideoInfo[rcid].VideoInfo[refFid].valid)
					continue;

				Skeletons[temp].vViewID_rFid[jid].emplace_back(rcid, refFid);
				Skeletons[temp].vPt2D[jid].emplace_back(u, v);
				Skeletons[temp].vConf[jid].push_back(s);
			}
			fclose(fp);
		}

		Skeletons[temp].valid = 1;
		Skeletons[temp].refFid = refFid;
		nvalidFrames++;
	}
	if (nvalidFrames == 0)
	{
		printLOG("\n");
		return 1;
	}
	vSkeleton.push_back(Skeletons), nPeople = vSkeleton.size();
	stopF = nvalidFrames - 1;

	int nSyncInst = ((winSize + nOverlappingFrames) / increF + 1);
	DensePose *vDensePose = new DensePose[nSCams*nSyncInst]; //should have enough mem for unsynced case

	//CostWeights[2] = CostWeights[2] / increF/increF;
	for (int pid = 0; pid < nPeople; pid++)
	{
		if (selectedPeopleID != -1 && pid != selectedPeopleID)
			continue;

		Point2i fixedPoseFrames(-1, -1);
		for (int chunkId = startChunkdId; chunkId < nChunks; chunkId++)
		{
			if (syncMode == 1)
			{
				if (startF + chunkId * winSize == min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames))
					continue;
				printLOG("*****************FitSMPLWindowDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
				FitSMPLWindowMOSH(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
			}
			else
			{
				printLOG("*****************FitSMPLWindowUnSyncDriver [%d: %d-->%d] *%d Cams  ****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), nGCams);
				FitSMPLUnSyncMOSH(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
			}
			fixedPoseFrames.x = startF + winSize * (chunkId + 1), fixedPoseFrames.y = startF + winSize * (chunkId + 1) + nOverlappingFrames;
		}

		if (startChunkdId > 0)
		{
			for (int chunkId = startChunkdId - 1; chunkId > -1; chunkId--)
			{
				fixedPoseFrames.x = startF + winSize * (chunkId + 1), fixedPoseFrames.y = startF + winSize * (chunkId + 1) + nOverlappingFrames;
				if (syncMode == 1)
				{
					if (startF + chunkId * winSize == min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames))
						continue;
					printLOG("*****************FitSMPLWindowDriver [%d: %d-->%d] *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames));
					FitSMPLWindowMOSH(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
				}
				else
				{
					printLOG("*****************FitSMPLWindowUnSyncDriver [%d: %d-->%d] *%d Cams *****************\n", chunkId, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), nGCams);
					FitSMPLUnSyncMOSH(Path, smplMaster, &vSkeleton[pid][chunkId*winSize / increF], vDensePose, VideoInfo, CamTimeInfo, vsCams, startF + chunkId * winSize, min(stopF, startF + (chunkId + 1)*winSize + nOverlappingFrames), increF, fixedPoseFrames, pid, distortionCorrected, sharedIntrinisc, skeletonPointFormat, CostWeights, isigmas, Real2SfM, hasDensePose);
				}
			}
		}
	}

	delete[]vDensePose, delete[]VideoInfo;
	for (int ii = 0; ii < vSkeleton.size(); ii++)
		delete[]vSkeleton[ii];

	return 0;
}
int EvalUnSyncMOSH(char *Path, int nSub, int nActions)
{
	char Fname[512];

	FILE *fp2 = fopen("MoshSyncDif.txt", "w");
	for (int subId = 1; subId <= nSub; subId++)
	{
		for (int actionId = 1; actionId <= nActions; actionId++)
		{
			sprintf(Fname, "%s/%.2d/%.2d_%.2d", Path, subId, subId, actionId);
			printLOG("%.2d %.2d: ", subId, actionId);

			vector<std::string> vnames;
#ifdef _WINDOWS
			sprintf(Fname, "%s/%.2d/%.2d_%.2d/FitBody/@1/Wj/0", Path, subId, subId, actionId);
			vnames = get_all_files_names_within_folder(std::string(Fname));
#endif 
			if (vnames.size() == 0)
			{
				printLOG("No files in %s\n", Fname);
				continue;
			}

			int increPer = 5, Per = 5;
			const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;
			for (int ii = 0; ii < vnames.size(); ii++)
			{
				if (100 * ii / vnames.size() >= Per)
				{
					printLOG("%d ..", Per);
					Per += increPer;
				}

				std::string CidString = vnames[ii].substr(0, 2);
				std::string FidString = vnames[ii].substr(3, 4);

				std::string str2(".");
				std::size_t pos = vnames[ii].find(str2);
				std::string TsString = vnames[ii].substr(8, pos - 8);

				double scale, params_T[3], params_R[72], params_gt_T[3], params_gt_R[72], R[9], R_gt[9], Rt[9], R_dif[9], r_dif[3], adif;

				sprintf(Fname, "%s/%.2d/%.2d_%.2d/FitBody_gt/@1/US_Smoothing/0/%s", Path, subId, subId, actionId, vnames[ii].c_str()); FILE *fp = fopen(Fname, "r");
				fscanf(fp, "%lf %lf %lf %lf ", &scale, &params_gt_T[0], &params_gt_T[1], &params_gt_T[2]);
				for (int ii = 0; ii < nJointsSMPL; ii++)
					fscanf(fp, "%lf %lf %lf ", &params_gt_R[3 * ii], &params_gt_R[3 * ii + 1], &params_gt_R[3 * ii + 2]);
				fclose(fp);

				sprintf(Fname, "%s/%.2d/%.2d_%.2d/FitBody/@1/Wj/0/%s", Path, subId, subId, actionId, vnames[ii].c_str()); fp = fopen(Fname, "r");
				fscanf(fp, "%lf %lf %lf %lf ", &scale, &params_T[0], &params_T[1], &params_T[2]);
				for (int ii = 0; ii < nJointsSMPL; ii++)
					fscanf(fp, "%lf %lf %lf ", &params_R[3 * ii], &params_R[3 * ii + 1], &params_R[3 * ii + 2]);
				fclose(fp);

				fprintf(fp2, "%d %d %s %.4f %.4f %.4f ", subId, actionId, vnames[ii].c_str(), params_T[0] - params_gt_T[0], params_T[1] - params_gt_T[1], params_T[2] - params_gt_T[2]);
				for (int ii = 0; ii < nJointsSMPL; ii++)
				{
					getRfromr(&params_R[3 * ii], R);
					getRfromr(&params_gt_R[3 * ii], R_gt);
					mat_transpose(R, Rt, 3, 3);
					mat_mul(Rt, R_gt, R_dif, 3, 3, 3);

					adif = std::acos(min(.99999999, max(-0.999999999, 0.5*(R_dif[0] + R_dif[4] + R_dif[8] - 1.0))));
					fprintf(fp2, "%.3f ", adif / Pi*180.0);
				}
				fprintf(fp2, "\n");
			}
			printLOG("\n", Fname);
		}
	}
	fclose(fp2);

	return 0;
}

int main(int argc, char** argv)
{
#ifdef _DEBUG
	srand(2);
#else
	srand(time(NULL));
#endif
	if (argc == 1)
	{
		printf("EnRecon.exe DataPath\n");
		return 0;
	}
	char *Path = argv[1];
	char Fname[512], Fname2[512], buffer[512];
	myGetCurDir(512, buffer);
	sprintf(Fname, "%s/Logs", Path); makeDir(Fname);
	sprintf(Fname, "%s/Vis", Path); makeDir(Fname);
	printLOG("Current input directory: %s\n", Path);

	{
		vector<int> sCams{ 0,1,2,3,4,5,6,7,8,9 };
		int TS[10] = { 0, 3, 6, 9, 1, 4, 7, 2, 5, 8 };
		double intrinsic[5] = { 2000, 2000, 0, 960, 540 };
		double distortion[7] = { 0, 0, 0, 0, 0, 0, 0 };

		double ShapeWeight = 0.00,
			PoseWeight = 0.00,
			TemporalWeight = 2.0,
			ContourWeight = 1.0,// #Trust points passed German - McClure robusifier
			SilWeight = 10.0,// #heavily penalize if not projecting to sil
			KeyPointsWeight = 1.0,// #Main driving force from init
			DensePoseWeight = 0.01;

		double WeightsSMPLFitting[] = { ShapeWeight, PoseWeight, TemporalWeight, ContourWeight, SilWeight,  KeyPointsWeight, DensePoseWeight , 0.0, 0.0 },
			iSigmaSMPLFitting[] = { 1.0 / 10.0, 1.0 / 5.0, 1.0 / 15.0, 1.0 / 100.0, 1.0 / 10.0 };//3D fitting dev (mm) , 2D detection (pixels), 2D seg (pixels), 3D joint smoothing (mm/s), 3D joint std (mm)


		int nSub = 11, nActions = 20, syncMode = 1;
		//CreateBodyJointFromSeq(Path, nSub, nActions);
		//CreateBodyJointFromSeq(Path, 10, intrinsic, distortion, 1920, 1080, 4000, true, 12, 0.0, 2.0, TS, false);
		//return 0;

		EvalUnSyncMOSH(Path, nSub, nActions);

		for (int subId = 1; subId <= nSub; subId++)
		{
			for (int actionId = 1; actionId <= nActions; actionId++)
			{
				if (subId == 1 && actionId == 1)
					continue;
				sprintf(Fname, "%s/%.2d/%.2d_%.2d", Path, subId, subId, actionId);
				//printLOG("\n****************%.2d %.2d*******************\n", subId, actionId);
				//SimulateCamerasAnd2DPointsForMOSH(Fname, 10, intrinsic, distortion, 1920, 1080, 4000, true, 12, 0.0, 0.0, TS, false);
				//FitSMPLWindowMOSHDriver(Fname, sCams, TS, 0, 500, atoi(argv[2]), atoi(argv[3]), 1, 1, 1, 25, syncMode, WeightsSMPLFitting, iSigmaSMPLFitting, 1.0, 0.1, 0, -1, 0);
			}
		}

		printf("Done\n");
		int startF = 0, stopF = 3000;
		//visualizationDriver("Z:/Users/minh/Mosh/01/01_01", sCams, startF, stopF, 1, true, false, false, false, false, 25, startF, 1, 0);
		return 0;
	}
	/*{
		int SkeletonPointFormat = 25;
		double real2sfm = 1.0, detConfThresh = 0.3;

		int hasDensePose = atoi(argv[3]);
		double ShapeWeight = 0.01,
			PoseWeight = 0.02,
			TemporalWeight = 2.0,
			ContourWeight = 1.0,// #Trust points passed German - McClure robusifier
			SilWeight = 10.0,// #heavily penalize if not projecting to sil
			KeyPointsWeight = 1.0,// #Main driving force from init
			DensePoseWeight = 0.01;

		double WeightsSMPLFitting[] = { ShapeWeight, PoseWeight, TemporalWeight, ContourWeight, SilWeight,  KeyPointsWeight, DensePoseWeight },
			iSigmaSMPLFitting[] = { 1.0 / 10.0, 1.0 / 5.0, 1.0 / 15.0, 1.0 / 100.0, 1.0 / 10.0 };//3D fitting dev (mm) , 2D detection (pixels), 2D seg (pixels), 3D joint smoothing (mm/s), 3D joint std (mm)

		FitSMPL_SingleView(Path, 1, 1, 18, 1, WeightsSMPLFitting, iSigmaSMPLFitting, 1.0, 0, 1, -1);

		return 0;
		sprintf(Fname, "%s/%s", Path, argv[2]);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%s ", Fname2) != EOF)
			FitSMPL_MVS_Driver(Path, Fname2, 1, SkeletonPointFormat, 1, WeightsSMPLFitting, iSigmaSMPLFitting, real2sfm, detConfThresh, hasDensePose, -1);
		fclose(fp);

		return 0;
	}*/

	//GroundPlanFitting("E:/Dataset/Corpus/floor.txt", "E:/Dataset/Corpus/plane.txt");
	//return 0;
	SfMPara mySfMPara;
	CommandParse(Path, mySfMPara);

	//GenSparseCorres4IRB_Driver2(Path, 10, 401, 1600);
	//GenRectifiedEpicSparseInterp(Path, 10, 401, 1600, 1, 2);

	//Visualize_VideoKeyFrame2CorpusSfM_Inliers(Path, 0, startF, stopF);
	//AssociatePeopleAndBuild3DFromCalibedSyncedCameras_3DVoting(Path, sCams, 104, 104, 1, 0, 0, 3, 0.4, 20.0, real2SfM, 30);
	//AssociatePeopleAndBuild3DFromCalibedSyncedCameras_RansacTri(Path, sCams, startF, stopF, 1, 0, 0, 3, 20, 10, 0.4, 100);

	int nCams = mySfMPara.nCams, startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF, trackingInstF = 20;
	double biDirectFlowThresh = 20, featFlowConsistencyPercent = 0.7, descSimThesh = 0.9, descRatioThresh = 0.8, overlappingAThresh = 0.1; //interacting/occluding people sometimes confuse the pose

	vector<int>sCams;
	for (int ii = 0; ii < nCams; ii++)
		sCams.push_back(ii);


	/*if (1)
	{
		//"C:\Users\mvo\OneDrive - cs.cmu.edu\Dataset" 3 0 500 5000 2 4 3 T42664764_rjohnston T42664773_rjohnston T42664789_rjohnston 3
		int SeqId = atoi(argv[3]), subCamStart = atoi(argv[6]), subCamStop = atoi(argv[7]);
		int nNames = atoi(argv[8]);
		std::vector<char*>vVideoNames;
		for (int ii = 0; ii < nNames; ii++)
			vVideoNames.push_back(argv[9 + ii]);
		std::vector<int> subCamIds;
		for (int ii = subCamStart; ii <= subCamStop; ii++)
			subCamIds.push_back(ii);

		int nCams = (int)subCamIds.size()*(int)vVideoNames.size();
		vector<int>sCams;
		for (int ii = 0; ii < nCams; ii++)
		{
			sprintf(Fname, "%s/%d", Path, ii); makeDir(Fname);
			sCams.push_back(ii);
		}

		//TrackBody_Landmark_BiDirectLK(Path, cid, startF, stopF, increF, biDirectFlowThresh, 1);
		//TrackBody_Landmark_BiDirectLK2(Path, vVideoNames, SeqId, subCamIds, cid, startF, stopF, increF, biDirectFlowThresh, 1);
		//TrackBody_SmallFeat_BiDirectLK(Path, cid, startF, stopF, increF, biDirectFlowThresh / 4, featFlowConsistencyPercent, overlappingAThresh, 1);
		//TrackBody_Desc_BiDirect(Path, cid, startF, stopF, increF, descSimThesh, descRatioThresh, overlappingAThresh, 1);
		//PerVideoMultiPeopleTracklet(Path, cid, startF, stopF, increF, 0);
		//CleanTrackletBy2DSmoothing_V2(Path, cid, startF, stopF, increF, 1280, 30, 2.0, 0);
		//CleanTrackletBy2DSmoothing_V3(Path, vVideoNames, SeqId, subCamIds, cid, startF, stopF, increF, 1280, 30, 2.0, 0);
		//VisualizeAllTracklets(Path, sCams, startF, stopF, startF, stopF);

		int distortionCorrected = 0, iterMax = 1000, nViewPlus = 3, nMinRanSacPoints = 3;
		double Reprojectionthreshold = 10, detectionThresh = 0.0;

		int LossType = 1, Use2DFitting = 1, selectedPeopleId = argc >= 13 ? atoi(argv[12]) : -1,
			hasDensePose = argc >= 14 ? atoi(argv[13]) : 0,
			startChunkdId = argc == 15 ? atoi(argv[14]) : 0;
		double WeightsSkeleton[] = { 1.0, 1.0, 1.0, 1.0 },  //projecion, const limb, symmetry, temporal
			iSigmaSkeleton[] = { 1.0 / 8.0, 1.0 / 30, 1.0 / 100, 1.0 / 200 }; //2D detection, limb length (mm), velocity variration for slow moving + fast moving joints(mm/s)
		double WeightsSMPLFitting[] = { mySfMPara.ShapeWeight, mySfMPara.PoseWeight, mySfMPara.TemporalWeight, mySfMPara.ContourWeight, mySfMPara.SilWeight,  mySfMPara.KeyPointsWeight, mySfMPara.DensePoseWeight, mySfMPara.HeadMountedCameraWeight, mySfMPara.FootClampingWeight },
			iSigmaSMPLFitting[] = { 1.0 / 10.0, 1.0 / 5.0, 1.0 / 15.0, 1.0 / 100.0, 1.0 / 10.0 };//3D fitting dev (mm) , 2D detection (pixels), 2D seg (pixels), 3D joint smoothing (mm/s), 3D joint std (mm)
		int wFrames = mySfMPara.SMPLWindow, nOverlappingFrames = mySfMPara.SMPLnOverlappingFrames;

		//TriangulateSkeleton3DFromCalibSyncedCameras_DensePose(Path, vVideoNames, SeqId, subCamIds, sCams, startF, stopF, increF, distortionCorrected, mySfMPara.SkeletonPointFormat, nViewPlus, Reprojectionthreshold, detectionThresh, iterMax, nMinRanSacPoints);
		//WindowSkeleton3DBundleAdjustment_DensePose(Path, vVideoNames, SeqId, subCamIds, nCams, startF, stopF, increF, distortionCorrected, mySfMPara.SkeletonPointFormat, 0.0001, 1, WeightsSkeleton, iSigmaSkeleton, mySfMPara.real2SfM, mySfMPara.missingFrameInterp);
		//FitSMPL1FrameDriver_DensePose(Path, vVideoNames, SeqId, subCamIds, sCams, startF, stopF, increF, distortionCorrected, mySfMPara.sharedIntrinsic, mySfMPara.SkeletonPointFormat, Use2DFitting, WeightsSMPLFitting, iSigmaSMPLFitting, mySfMPara.real2SfM, detectionThresh, hasDensePose, selectedPeopleId);
		//FitSMPLWindowDriver_DensePose(Path, vVideoNames, SeqId, subCamIds, sCams, startF, stopF, wFrames, nOverlappingFrames, increF, distortionCorrected, mySfMPara.sharedIntrinsic, mySfMPara.SkeletonPointFormat, mySfMPara.SyncedMode, WeightsSMPLFitting, iSigmaSMPLFitting, mySfMPara.real2SfM, detectionThresh, hasDensePose, selectedPeopleId, startChunkdId);
		//visualizationDriver2(Path, vVideoNames, SeqId, subCamIds, sCams, startF, stopF, 1, true, false, false, false, false, 17, startF, true);

		return 0;
	}*/

	vector<int> TimeStamp(nCams);
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	int id, offset; double fps;
	while (fscanf(fp, "%d %lf %d ", &id, &fps, &offset) != EOF)
	{
		for (size_t ii = 0; ii < sCams.size(); ii++)
		{
			if (sCams[ii] == id)
			{
				TimeStamp[sCams[ii]] = offset;
				break;
			}
		}
	}
	fclose(fp);

	//sprintf(Fname, "%s/Corpus", Path);
	//double SfMdistance = TriangulatePointsFromCorpusCameras(Fname, 0, 2, 2.0);
	//printLOG("SfM measured distance: %.3f\n", SfMdistance);
	double real2SfM = mySfMPara.real2SfM;
	//JointCalibrationAndHumanEstiationDriver(Path, sCams, TimeStamp, startF, stopF, 1);
	//AlignedPerFrameBodySfM(Path, nCams, 30, 300, 1, 10);
	//ComposeAlignedBodySfm(Path, nCams, 30, 50, 1, 10);
	//SimpleBodyPoseSfM_Tracking(Path, 60, 30, 4);
	//SimpleBodyPoseSfM_Tracking(Path, 120, 30, 4);
	//SimpleBodyPoseSfM_Tracking(Path, 90, 60, 4);
	visualizationDriver(Path, sCams, startF, stopF, 1, true, false, false, false, false, 25, startF, 1, 0);
	//visualizePerFrameSfM(Path, 60, 30, 280, 4, 10);

	//int nViewPlus = 4, increFAsso = 100;
	//double SpatialSamplingRate = 0.5, temporalSamplingRate = 0.33, temporalSamplingRateDif = 0.33, IntraDifSamplingChance = 1.0;
	//GenPerCamPeopleGTLabel(Path, sCams, startF, stopF);
	//GenerateGTTracklets(Path, sCams, startF, stopF, startF, stopF, 0);
	//GenerateSpatialPeopleMatchingFromGT(Path, sCams, TimeStamp, startF, stopF, 5);
	//GenerateSTM4SemPeopleDataGT(Path, sCams, TimeStamp, startF, stopF, startF, stopF, increF, SpatialSamplingRate, IntraDifSamplingChance, 1, 0);
	//GenerateSTM4SemPeopleData(Path, sCams, TimeStamp, startF, stopF, startF, stopF, increF, 4, IntraDifSamplingChance, 1, 1);

	//int  knn = nCams, VisIncreF = 5000, nPeople = 14;
	//for (int metricLearning = 0; metricLearning < 0; metricLearning++)
	//{
	//FindPerPersonPerFrameKnnDescTemporalPoolingSpatioMatching(Path, sCams, TimeStamp, knn, startF, stopF, startF, stopF, 1, true, metricLearning, VisIncreF);
	//ValidateSpatialAssociation(Path, sCams, TimeStamp, startF, stopF, startF, stopF, nPeople, 20, 0.5, metricLearning);
	//}

	//SpatialTemporalCalibInTheWildDriver(Path, nCams, startF, stopF, 1, startF, stopF, trackingInstF, 15, 8, 2); //match all pair sifts
	//SpatialTemporalCalibInTheWildDriver(Path, nCams, startF, stopF, 1, startF, stopF, trackingInstF, 15, 8, 3); //track matched sifts
	//SpatialTemporalCalibInTheWildDriver(Path, nCams, startF, stopF, 1, startF, stopF, trackingInstF, 15, 8, 4); //determin static vs. dynamic
	//SpatialTemporalCalibInTheWildDriver(Path, nCams, startF, stopF, 1, startF, stopF, trackingInstF, 15, 8, 5); //FGeosync
	//SpatialTemporalCalibInTheWildDriver(Path, nCams, startF, stopF, 1, startF, stopF, trackingInstF, 15, 8, 6); //Init triangulation
	//SpatialTemporalCalibInTheWildDriver(Path, nCams, startF, stopF, 1, startF, stopF, trackingInstF, 15, 8, 7);

	int LensType = 0, distortionCorrected = 0, iterMax = 1000, nViewPlus = 3, nMinRanSacPoints = 3;
	double Reprojectionthreshold = 300, detectionThresh = 0.3;

	int LossType = 1, Use2DFitting = 1, selectedPeopleId = argc >= 3 ? atoi(argv[2]) : -1,
		hasDensePose = argc >= 4 ? atoi(argv[3]) : 0,
		startChunkdId = argc == 5 ? atoi(argv[4]) : 0;
	double WeightsSkeleton[] = { 1.0, 1.0, 1.0, 1.0 },  //projecion, const limb, symmetry, temporal
		iSigmaSkeleton[] = { 1.0 / 8.0, 1.0 / 30, 1.0 / 100, 1.0 / 200 }; //2D detection, limb length (mm), velocity variration for slow moving + fast moving joints(mm/s)
	double WeightsSMPLFitting[] = { mySfMPara.ShapeWeight, mySfMPara.PoseWeight, mySfMPara.TemporalWeight, mySfMPara.ContourWeight, mySfMPara.SilWeight,  mySfMPara.KeyPointsWeight, mySfMPara.DensePoseWeight },
		iSigmaSMPLFitting[] = { 1.0 / 10.0, 1.0 / 5.0, 1.0 / 15.0, 1.0 / 100.0, 1.0 / 10.0 };//3D fitting dev (mm) , 2D detection (pixels), 2D seg (pixels), 3D joint smoothing (mm/s), 3D joint std (mm)
	int wFrames = mySfMPara.SMPLWindow, nOverlappingFrames = mySfMPara.SMPLnOverlappingFrames;

	//TriangulatePointsFromNonCorpusCameras(Path, sCams, &TimeStamp[0], 0);
	//VisualizeAllViewsEpipolarGeometry(Path, allCams, startF, stopF);
	//VisualizeAllTwoViewsTriangulation(Path, allCams, startF, stopF);
	//AllPairSyncKeyPointsDriver(Path, sCams, startF, stopF, startF, stopF, increF, 60);
	//TriangulateSkeleton3DFromCalibSyncedCameras(Path, sCams, startF, stopF, increF, distortionCorrected, mySfMPara.SkeletonPointFormat, nViewPlus, Reprojectionthreshold, detectionThresh, iterMax, nMinRanSacPoints);
	//WindowSkeleton3DBundleAdjustment(Path, nCams, startF, stopF, increF, distortionCorrected, mySfMPara.SkeletonPointFormat, detectionThresh, LossType, WeightsSkeleton, iSigmaSkeleton, real2SfM);
	FitSMPL1FrameDriver(Path, sCams, startF, stopF, increF, distortionCorrected, mySfMPara.sharedIntrinsic, mySfMPara.SkeletonPointFormat, Use2DFitting, WeightsSMPLFitting, iSigmaSMPLFitting, real2SfM, detectionThresh, hasDensePose, selectedPeopleId);
	//FitSMPLWindowDriver(Path, sCams, startF, stopF, wFrames, nOverlappingFrames, increF, distortionCorrected, mySfMPara.sharedIntrinsic, mySfMPara.SkeletonPointFormat, mySfMPara.SyncedMode, WeightsSMPLFitting, iSigmaSMPLFitting, real2SfM, detectionThresh, hasDensePose, selectedPeopleId, startChunkdId);

	/*sCams.clear();
	for (int ii = 0; ii < 12; ii++)
		sCams.push_back(ii);
	for (int viewID = 0; viewID <= 15; viewID++)
	{
		vector<CameraData> NovelCam;
		ReadVideoDataI2(Path, NovelCam, viewID, 0, 0, 1);
		CollectNearestViewsBaseOnGeometry2(Path, NovelCam, viewID, sCams, mySfMPara.startF, mySfMPara.stopF, 10);
	}*/

	//TextureMappingBody(Path, startF, stopF, increF, nCams, -1, 1);
	//interpUlti(Path, sCams, startF, stopF, increF);
	//STRecalibrateCamerasFromSkeleton(Path, allCams, CamsToCalib, startF, stopF, increF, 0, 0.3);

	//VisualizeProjected3DSkeleton(Path, nCams, startF, stopF, increF, true, 0.5);
	//VisualizeProjectedSMPLBody(Path, mySfMPara.nCams, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, mySfMPara.SkeletonPointFormat, 14, 1.0, 1, 0);
	//VisualizeProjectedSMPLBodyUnSync(Path, mySfMPara.nCams, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, mySfMPara.SkeletonPointFormat, 14, 1.0, 1, 0);
	//#pragma omp parallel for schedule(dynamic,1)
	//for (int cid = 8; cid < mySfMPara.nCams; cid++)5
	//AllBackgroundButHumanBlurring(Path, cid, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, 6, 1920, 1080, 0.5);

	//SMPLBodyTexGen(Path, nCams, startF, stopF, increF, 14, 0);


	return 0;
}
