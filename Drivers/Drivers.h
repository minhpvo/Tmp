#if !defined(DRIVERS_H )
#define DRIVERS_H

#include "../Audio/AudioPro.h"
#include "../Ulti/MathUlti.h"
#include "../Ulti/DataIO.h"
#include "../Geometry/Geometry1.h"
#include "../Geometry/Geometry2.h"
#include "../ImgPro/ImagePro.h"
#include "../TrackMatch/MatchingTracking.h"
#include "../TrackMatch/FeatureEst.h"
#include "../Vis/Visualization.h"
#include "opencv2/video/video.hpp"
#include <unsupported/Eigen/KroneckerProduct>

int CommandParse(char *Path, SfMPara &mySfMPara);

//   (c)coeffs      (p)pose      (t)
//     |             |            |
//  Vs = mu + U*c   /|            |
//     |           / |            |
//    / \         /  |            |
//   |  J=Rj*Vs  /   |            |
//   |       \  /    |            |
//   |        \/     |           /
//   |        /\     |          /
//   | L=p2l(p) T=p2t(p,J)     /
//   |     |     |            /
//   |     |     |          /
//   |     |     |        /
//  Vp=Vs + Rp*L |      /
//   |           /     /
//   |          /     /
//  Vl=lbs(Vp,T)     /
//   |              /
//  Vt = Vl + dVdt*t
// dVtdc = dVldVp*dVsdc + dVldT*dTdJ*dJdVs*dVsdc (+ dVldVp*dVpdL*dLdJ*dJdVs*dVsdc == 0, dLdJ==0)
// dVtdp = dVldT*dTdp + dVldVp*dVpdL*dLdp
// dVtdt = dVdt
// dVtdc = dVldVp*U + dVldT*dTdJ*Rj*U
// dVtdp = dVldT*dTdp + dVldVp*Rp*dLdp
namespace smpl
{
	struct SMPLModel
	{
		static const int nShapeCoeffs = 10;
		static const int nVertices = 6890;
		static const int nJoints = 24, naJoints = 39;
		static const int NUM_POSE_PARAMETERS = nJoints * 3;

		vector<int>vDensePosePartId;
		vector<Point2f> vUV;
		double minPose[72], maxPose[72], Mosh_asmpl_J_istd[39], Mosh_pose_prior_mu[72], Mosh_dv_prior_mu[72];

		// Template vertices (vector) <nVertices*3, 1> xyz xyz xyz ...
		Eigen::Matrix<double, Eigen::Dynamic, 1> mu_;

		// Shape basis, <NUM_FACE_POINTS*3, NUM_COEFFICIENTS>
		Eigen::Matrix<double, Eigen::Dynamic, nShapeCoeffs, Eigen::RowMajor> U_;

		// LBS weights,
		Eigen::Matrix<double, Eigen::Dynamic, nJoints, Eigen::RowMajor> W_;

		// J_mu_ = J_regl_bigl_ * mu_
		Eigen::Matrix<double, nJoints * 3, 1> J_mu_;
		// dJdc = J_regl_bigl_ * U_
		Eigen::Matrix<double, Eigen::Dynamic, nShapeCoeffs, Eigen::RowMajor> dJdc_;

		// Joint regressor, <nJoints, nVertices>
		Eigen::SparseMatrix<double, Eigen::RowMajor> J_regl_;

		// Joint regressor, <nJoints*, nVertices*3>  kron(J_regl_, eye(3))
		Eigen::SparseMatrix<double, Eigen::RowMajor> J_regl_bigl_;
		Eigen::SparseMatrix<double, Eigen::RowMajor> J_regl_abigl_;
		Eigen::SparseMatrix<double, Eigen::ColMajor> J_regl_bigl_col_;

		// Pose regressor, <nVertices*3, (nJoints-1)*9>
		//Eigen::Matrix<double, nVertices * 3, Eigen::Dynamic, Eigen::RowMajor> pose_regl_;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pose_regl_;

		// Shape coefficient weights
		Eigen::Matrix<double, Eigen::Dynamic, 1> d_;

		// Triangle faces
		Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> faces_, connections_;
		vector<Point3i> vFaces;
		vector<Point2i> vConnections;

		// Kinematic tree
		Eigen::Matrix<int, 2, Eigen::Dynamic> kintree_table_;
		int parent_[nJoints];
		int id_to_col_[nJoints];

		// SMPL2OpenPose correspondences
		Eigen::SparseMatrix<double, Eigen::ColMajor> J_regl_14_bigl_col_;
		Eigen::SparseMatrix<double, Eigen::ColMajor> J_regl_17_bigl_col_;
		Eigen::SparseMatrix<double, Eigen::ColMajor> J_regl_25_bigl_col_;
		Eigen::VectorXd J_regl_pm_weights;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mosh_dV_prior_A;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mosh_pose_prior_A;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pose_prior_A;
		Eigen::Matrix<double, Eigen::Dynamic, 1> pose_prior_mu;
		Eigen::Matrix<double, Eigen::Dynamic, 1> pose_prior_b;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GMM_pose_prior_iCov[8];
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GMM_pose_prior_A[8];
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>  GMM_pose_prior_mu;
		Eigen::Matrix<double, Eigen::Dynamic, 1>  GMM_pose_prior_w;

		// A model is fully specified by its coefficients, pose,  a translation, and a global scale
		Eigen::Matrix<double, nShapeCoeffs, 1> coeffs;
		Eigen::Matrix<double, nJoints, 3, Eigen::RowMajor> pose;
		Eigen::Vector3d t;
		double scale;

		SMPLModel()
		{
			pose_regl_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(nVertices * 3, (nJoints - 1) * 9);
			t.setZero();
			coeffs.setZero();
			pose.setZero();
		}

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
	struct SMPLParams
	{
		// A model is fully specified by its coefficients, pose, and a translation
		int frame;
		double scale;
		Eigen::Vector3d t;
		Eigen::Matrix<double, SMPLModel::nJoints, 3, Eigen::RowMajor> pose;
		Eigen::Matrix<double, SMPLModel::nShapeCoeffs, 1> coeffs;
		SMPLParams()
		{
			t.setZero();
			pose.setZero();
			coeffs.setZero();
		}
	};

	struct PoseToTransforms
	{
		PoseToTransforms(const SMPLModel &mod) : smpl(mod) {}
		// nJoints*3,  nJoints*3, (nJoints)*3*4+(nJoints-1)*3*3
		template <typename T>	 bool operator()(const T* const pose, const T* const joints, T* transforms_and_lrotmin) const
		{
			const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

			Map< const Matrix<T, nJoints, 3, RowMajor> > P(pose);
			Map< const Matrix<T, nJoints, 3, RowMajor> > J(joints);
			Map< Matrix<T, 3 * nJoints, 4, RowMajor> > outT(transforms_and_lrotmin);
			Map< Matrix<T, 3 * (nJoints - 1), 3, RowMajor> >	outR(transforms_and_lrotmin + 3 * nJoints * 4);
			Matrix<T, Dynamic, 4, RowMajor> Ms(4 * nJoints, 4);
			Matrix<T, 3, 3, ColMajor> R; // Interface with ceres

			ceres::AngleAxisToRotationMatrix(pose, R.data());
			Ms.setZero();
			Ms.block(0, 0, 3, 3) = R;
			Ms(0, 3) = J(0, 0), Ms(1, 3) = J(0, 1), Ms(2, 3) = J(0, 2), Ms(3, 3) = T(1.0);

			for (int idj = 1; idj < smpl.nJoints; idj++)
			{
				int ipar = smpl.parent_[idj];
				ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());
				outR.block((idj - 1) * 3, 0, 3, 3) = R;
				outR((idj - 1) * 3, 0) -= T(1.0), outR((idj - 1) * 3 + 1, 1) -= T(1.0), outR((idj - 1) * 3 + 2, 2) -= T(1.0);

				Ms.block(idj * 4, 0, 3, 3) = Ms.block(ipar * 4, 0, 3, 3)*R;
				Ms.block(idj * 4, 3, 3, 1) = Ms.block(ipar * 4, 3, 3, 1) + Ms.block(ipar * 4, 0, 3, 3)*(J.row(idj).transpose() - J.row(ipar).transpose());
				Ms(idj * 4 + 3, 3) = T(1.0);
			}
			for (int idj = 0; idj < smpl.nJoints; idj++)
				Ms.block(idj * 4, 3, 3, 1) -= Ms.block(idj * 4, 0, 3, 3)*J.row(idj).transpose();
			for (int idj = 0; idj < smpl.nJoints; idj++)
				outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
			return true;
		}
		const SMPLModel &smpl;
	};
	struct PoseToTransformsNoLR
	{
		PoseToTransformsNoLR(const SMPLModel &mod) : smpl(mod) {}
		// nJoints*3,  nJoints*3,  (nJoints)*3*4
		template <typename T>	bool operator()(const T* const pose, const T* const joints, T* transforms) const
		{
			const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJoints = SMPLModel::nJoints;

			Map< const Matrix<T, nJoints, 3, RowMajor> > P(pose);
			Map< const Matrix<T, nJoints, 3, RowMajor> > J(joints);
			Map< Matrix<T, 3 * nJoints, 4, RowMajor> > outT(transforms);
			Matrix<T, Dynamic, 4, RowMajor> Ms(4 * nJoints, 4);
			Matrix<T, 3, 3, ColMajor> R; // Interface with ceres

			ceres::AngleAxisToRotationMatrix(pose, R.data());
			Ms.setZero();
			Ms.block(0, 0, 3, 3) = R;
			Ms(0, 3) = J(0, 0), Ms(1, 3) = J(0, 1), Ms(2, 3) = J(0, 2), Ms(3, 3) = T(1.0);

			for (int idj = 1; idj < smpl.nJoints; idj++)
			{
				int ipar = smpl.parent_[idj];
				ceres::AngleAxisToRotationMatrix(pose + idj * 3, R.data());

				Ms.block(idj * 4, 0, 3, 3) = Ms.block(ipar * 4, 0, 3, 3)*R;
				Ms.block(idj * 4, 3, 3, 1) = Ms.block(ipar * 4, 3, 3, 1) + Ms.block(ipar * 4, 0, 3, 3)*(J.row(idj).transpose() - J.row(ipar).transpose());
				Ms(idj * 4 + 3, 3) = T(1.0);
			}
			for (int idj = 0; idj < smpl.nJoints; idj++)
				Ms.block(idj * 4, 3, 3, 1) -= Ms.block(idj * 4, 0, 3, 3)*J.row(idj).transpose();
			for (int idj = 0; idj < smpl.nJoints; idj++)
				outT.block(idj * 3, 0, 3, 4) = Ms.block(idj * 4, 0, 3, 4);
			return true;
		}
		const SMPLModel &smpl;
	};

	struct SMPLShapeCoeffRegCeres {
		SMPLShapeCoeffRegCeres(int num_parameters) : num_parameters_(num_parameters) {}
		template <typename T>	inline bool operator()(const T* const p, T* residuals) const
		{
			for (int i = 0; i < num_parameters_; i++)
				residuals[i] = T(1.0)*p[i];
			return true;
		}
		const double num_parameters_;
	};

	// Shape template from coefficients
	void coeffs_to_verts(const SMPLModel &smpl, const double *coeffs, double *outVerts, double *jacobian = 0);
	// LBS warping function with derivatives. No pose regression. (LR)
	void lbs(const SMPLModel &smpl, const double *verts, const MatrixXdr& T, double *outVerts, const MatrixXdr &dVsdc, const MatrixXdr &dTdp, const MatrixXdr &dTdc, MatrixXdr &dVdc, MatrixXdr &dVdp);
	// LBS warping function with derivatives.
	void lbs2(const SMPLModel &smpl, const double *verts, const MatrixXdr& T, double *outVerts, SparseMatrix<double, RowMajor> &dVdVs, SparseMatrix<double, RowMajor> &dVdT);

	// Reconstruct shape with pose & coefficients (no translation)
	void reconstruct(const SMPLModel &smpl, const double *coeffs, const double *pose, double *outVerts);
	void reconstruct(const SMPLModel &smpl, const double *coeffs, const double *pose, double *outVerts, MatrixXdr &dVdc, MatrixXdr &dVdp);
	void reconstruct2(const SMPLModel &smpl, const double *coeffs, const double *pose, double *outVerts, MatrixXdr &dVdc, MatrixXdr &dVdp);
}

bool ReadSMPLData(char *Path, smpl::SMPLModel &smpl);
vector<int> RasterizeMesh4OccludingContour(double *P, int width, int height, double*vXYZ, int nVertices, vector<Point3i> &vfaces, vector<Point2i> &vConnections, bool *hit = NULL);
vector<int> RasterizeMesh4OccludingContour(Point2d *vuv, int nVertices, int width, int height, vector<Point3i> &vfaces, vector<Point2i> &vConnections, bool *hit = NULL);

using namespace cv;
//Main drivers:
int SequenceSaverViewer(char *Path, int cid);
int SpaceTimeDomeViewer(char *Path);
void VisualizeCleanMatches(char *Path, int view1, int view2, int timeID, double fractionMatchesDisplayed = 0.5, int frameTimeStamp1 = 0, int frameTimeStamp2 = 0);
void VisualizeCleanMatches2(char *Path, int view1, int view2, int timeID, double fractionMatchesDisplayed = 0.5);
int VisualizePnPMatches(char *Path, int cameraID, int timeID);

template <class T> void Draw2DCoCoJoints(Mat &img, Point_<T> *joints, int nJointsCOCO, int lineThickness = 1, double resizeFactor = 1.0, Scalar *color = NULL, Point_<T> TopLeft = Point_<T>(0, 0))
{
	if (color == NULL)
	{
		if (nJointsCOCO == 18)
		{
			//head-neck
			if (joints[0].x != 0 && joints[1].x != 0)
				line(img, joints[0] * resizeFactor, joints[1] * resizeFactor, Scalar(0, 0, 255), lineThickness);

			//left arm
			if (joints[1].x != 0 && joints[2].x != 0)
				line(img, joints[1] * resizeFactor, joints[2] * resizeFactor, Scalar(0, 255, 0), lineThickness);
			if (joints[2].x != 0 && joints[3].x != 0)
				line(img, joints[3] * resizeFactor, joints[2] * resizeFactor, Scalar(0, 255, 0), lineThickness);
			if (joints[3].x != 0 && joints[4].x != 0)
				line(img, joints[3] * resizeFactor, joints[4] * resizeFactor, Scalar(0, 255, 0), lineThickness);

			//right arm
			if (joints[1].x != 0 && joints[5].x != 0)
				line(img, joints[1] * resizeFactor, joints[5] * resizeFactor, Scalar(255, 0, 0), lineThickness);
			if (joints[5].x != 0 && joints[6].x != 0)
				line(img, joints[6] * resizeFactor, joints[5] * resizeFactor, Scalar(255, 0, 0), lineThickness);
			if (joints[6].x != 0 && joints[7].x != 0)
				line(img, joints[6] * resizeFactor, joints[7] * resizeFactor, Scalar(255, 0, 0), lineThickness);

			//left leg
			if (joints[1].x != 0 && joints[8].x != 0)
				line(img, joints[1] * resizeFactor, joints[8] * resizeFactor, Scalar(0, 128, 255), lineThickness);
			if (joints[8].x != 0 && joints[9].x != 0)
				line(img, joints[9] * resizeFactor, joints[8] * resizeFactor, Scalar(0, 128, 255), lineThickness);
			if (joints[9].x != 0 && joints[10].x != 0)
				line(img, joints[9] * resizeFactor, joints[10] * resizeFactor, Scalar(0, 128, 255), lineThickness);

			//right leg
			if (joints[1].x != 0 && joints[11].x != 0)
				line(img, joints[1] * resizeFactor, joints[11] * resizeFactor, Scalar(128, 0, 255), lineThickness);
			if (joints[11].x != 0 && joints[12].x != 0)
				line(img, joints[12] * resizeFactor, joints[11] * resizeFactor, Scalar(128, 0, 255), lineThickness);
			if (joints[12].x != 0 && joints[13].x != 0)
				line(img, joints[12] * resizeFactor, joints[13] * resizeFactor, Scalar(128, 0, 255), lineThickness);

			//right eye+ ear
			if (joints[0].x != 0 && joints[14].x != 0)
				line(img, joints[0] * resizeFactor, joints[14] * resizeFactor, Scalar(128, 0, 0), lineThickness);
			if (joints[14].x != 0 && joints[16].x != 0)
				line(img, joints[16] * resizeFactor, joints[14] * resizeFactor, Scalar(128, 0, 0), lineThickness);

			//left eye+ ear
			if (joints[0].x != 0 && joints[15].x != 0)
				line(img, joints[0] * resizeFactor, joints[15] * resizeFactor, Scalar(128, 128, 0), lineThickness);
			if (joints[15].x != 0 && joints[17].x != 0)
				line(img, joints[17] * resizeFactor, joints[15] * resizeFactor, Scalar(128, 128, 0), lineThickness);
		}
		else if (nJointsCOCO == 25)
		{
			//head-neck
			if (joints[0].x != 0 && joints[1].x != 0)
				line(img, joints[0] * resizeFactor, joints[1] * resizeFactor, Scalar(0, 0, 255), lineThickness);

			//left arm
			if (joints[1].x != 0 && joints[2].x != 0)
				line(img, joints[1] * resizeFactor, joints[2] * resizeFactor, Scalar(0, 255, 0), lineThickness);
			if (joints[2].x != 0 && joints[3].x != 0)
				line(img, joints[3] * resizeFactor, joints[2] * resizeFactor, Scalar(0, 255, 0), lineThickness);
			if (joints[3].x != 0 && joints[4].x != 0)
				line(img, joints[3] * resizeFactor, joints[4] * resizeFactor, Scalar(0, 255, 0), lineThickness);

			//right arm
			if (joints[1].x != 0 && joints[5].x != 0)
				line(img, joints[1] * resizeFactor, joints[5] * resizeFactor, Scalar(255, 0, 0), lineThickness);
			if (joints[5].x != 0 && joints[6].x != 0)
				line(img, joints[6] * resizeFactor, joints[5] * resizeFactor, Scalar(255, 0, 0), lineThickness);
			if (joints[6].x != 0 && joints[7].x != 0)
				line(img, joints[6] * resizeFactor, joints[7] * resizeFactor, Scalar(255, 0, 0), lineThickness);

			//spline
			if (joints[1].x != 0 && joints[8].x != 0)
				line(img, joints[1] * resizeFactor, joints[8] * resizeFactor, Scalar(0, 128, 255), lineThickness);

			//left leg
			if (joints[9].x != 0 && joints[8].x != 0)
				line(img, joints[9] * resizeFactor, joints[8] * resizeFactor, Scalar(0, 128, 255), lineThickness);
			if (joints[10].x != 0 && joints[9].x != 0)
				line(img, joints[9] * resizeFactor, joints[10] * resizeFactor, Scalar(0, 128, 255), lineThickness);
			if (joints[11].x != 0 && joints[10].x != 0)
				line(img, joints[11] * resizeFactor, joints[10] * resizeFactor, Scalar(0, 128, 255), lineThickness);

			//right leg
			if (joints[8].x != 0 && joints[12].x != 0)
				line(img, joints[8] * resizeFactor, joints[12] * resizeFactor, Scalar(128, 0, 255), lineThickness);
			if (joints[13].x != 0 && joints[12].x != 0)
				line(img, joints[12] * resizeFactor, joints[13] * resizeFactor, Scalar(128, 0, 255), lineThickness);
			if (joints[14].x != 0 && joints[13].x != 0)
				line(img, joints[14] * resizeFactor, joints[13] * resizeFactor, Scalar(128, 0, 255), lineThickness);

			//right eye+ ear
			if (joints[0].x != 0 && joints[15].x != 0)
				line(img, joints[0] * resizeFactor, joints[15] * resizeFactor, Scalar(128, 0, 0), lineThickness);
			if (joints[15].x != 0 && joints[17].x != 0)
				line(img, joints[15] * resizeFactor, joints[17] * resizeFactor, Scalar(128, 0, 0), lineThickness);

			//left eye+ ear
			if (joints[0].x != 0 && joints[16].x != 0)
				line(img, joints[0] * resizeFactor, joints[16] * resizeFactor, Scalar(128, 128, 0), lineThickness);
			if (joints[16].x != 0 && joints[18].x != 0)
				line(img, joints[16] * resizeFactor, joints[18] * resizeFactor, Scalar(128, 128, 0), lineThickness);

			//left foot	
			if (joints[11].x != 0 && joints[22].x != 0)
				line(img, joints[11] * resizeFactor, joints[22] * resizeFactor, Scalar(128, 128, 0), lineThickness);
			if (joints[11].x != 0 && joints[23].x != 0)
				line(img, joints[11] * resizeFactor, joints[23] * resizeFactor, Scalar(128, 128, 0), lineThickness);
			if (joints[11].x != 0 && joints[24].x != 0)
				line(img, joints[11] * resizeFactor, joints[24] * resizeFactor, Scalar(128, 128, 0), lineThickness);

			//right foot	
			if (joints[14].x != 0 && joints[19].x != 0)
				line(img, joints[14] * resizeFactor, joints[19] * resizeFactor, Scalar(128, 128, 0), lineThickness);
			if (joints[14].x != 0 && joints[20].x != 0)
				line(img, joints[14] * resizeFactor, joints[20] * resizeFactor, Scalar(128, 128, 0), lineThickness);
			if (joints[14].x != 0 && joints[21].x != 0)
				line(img, joints[14] * resizeFactor, joints[21] * resizeFactor, Scalar(128, 128, 0), lineThickness);
		}

		for (int i = 0; i < nJointsCOCO; i++)
			if (joints[i].x != 0.0)
				circle(img, (joints[i] - TopLeft)* resizeFactor, 2, Scalar(0, 0, 255), lineThickness);
	}
	else
	{
		if (nJointsCOCO == 18)
		{
			//head-neck
			if (joints[0].x != 0 && joints[1].x != 0)
				line(img, joints[0] * resizeFactor, joints[1] * resizeFactor, color[0], lineThickness);

			//left arm
			if (joints[1].x != 0 && joints[2].x != 0)
				line(img, joints[1] * resizeFactor, joints[2] * resizeFactor, color[0], lineThickness);
			if (joints[2].x != 0 && joints[3].x != 0)
				line(img, joints[3] * resizeFactor, joints[2] * resizeFactor, color[0], lineThickness);
			if (joints[3].x != 0 && joints[4].x != 0)
				line(img, joints[3] * resizeFactor, joints[4] * resizeFactor, color[0], lineThickness);

			//right arm
			if (joints[1].x != 0 && joints[5].x != 0)
				line(img, joints[1] * resizeFactor, joints[5] * resizeFactor, color[0], lineThickness);
			if (joints[5].x != 0 && joints[6].x != 0)
				line(img, joints[6] * resizeFactor, joints[5] * resizeFactor, color[0], lineThickness);
			if (joints[6].x != 0 && joints[7].x != 0)
				line(img, joints[6] * resizeFactor, joints[7] * resizeFactor, color[0], lineThickness);

			//left leg
			if (joints[1].x != 0 && joints[8].x != 0)
				line(img, joints[1] * resizeFactor, joints[8] * resizeFactor, color[0], lineThickness);
			if (joints[8].x != 0 && joints[9].x != 0)
				line(img, joints[9] * resizeFactor, joints[8] * resizeFactor, color[0], lineThickness);
			if (joints[9].x != 0 && joints[10].x != 0)
				line(img, joints[9] * resizeFactor, joints[10] * resizeFactor, color[0], lineThickness);

			//right leg
			if (joints[1].x != 0 && joints[11].x != 0)
				line(img, joints[1] * resizeFactor, joints[11] * resizeFactor, color[0], lineThickness);
			if (joints[11].x != 0 && joints[12].x != 0)
				line(img, joints[12] * resizeFactor, joints[11] * resizeFactor, color[0], lineThickness);
			if (joints[12].x != 0 && joints[13].x != 0)
				line(img, joints[12] * resizeFactor, joints[13] * resizeFactor, color[0], lineThickness);

			//right eye+ ear
			if (joints[0].x != 0 && joints[14].x != 0)
				line(img, joints[0] * resizeFactor, joints[14] * resizeFactor, color[0], lineThickness);
			if (joints[14].x != 0 && joints[16].x != 0)
				line(img, joints[16] * resizeFactor, joints[14] * resizeFactor, color[0], lineThickness);

			//left eye+ ear
			if (joints[0].x != 0 && joints[15].x != 0)
				line(img, joints[0] * resizeFactor, joints[15] * resizeFactor, color[0], lineThickness);
			if (joints[15].x != 0 && joints[17].x != 0)
				line(img, joints[17] * resizeFactor, joints[15] * resizeFactor, color[0], lineThickness);
		}
		else if (nJointsCOCO == 25)
		{
			//head-neck
			if (joints[0].x != 0 && joints[1].x != 0)
				line(img, joints[0] * resizeFactor, joints[1] * resizeFactor, color[0], lineThickness);

			//left arm
			if (joints[1].x != 0 && joints[2].x != 0)
				line(img, joints[1] * resizeFactor, joints[2] * resizeFactor, color[0], lineThickness);
			if (joints[2].x != 0 && joints[3].x != 0)
				line(img, joints[3] * resizeFactor, joints[2] * resizeFactor, color[0], lineThickness);
			if (joints[3].x != 0 && joints[4].x != 0)
				line(img, joints[3] * resizeFactor, joints[4] * resizeFactor, color[0], lineThickness);

			//right arm
			if (joints[1].x != 0 && joints[5].x != 0)
				line(img, joints[1] * resizeFactor, joints[5] * resizeFactor, color[0], lineThickness);
			if (joints[5].x != 0 && joints[6].x != 0)
				line(img, joints[6] * resizeFactor, joints[5] * resizeFactor, color[0], lineThickness);
			if (joints[6].x != 0 && joints[7].x != 0)
				line(img, joints[6] * resizeFactor, joints[7] * resizeFactor, color[0], lineThickness);

			//spline
			if (joints[1].x != 0 && joints[8].x != 0)
				line(img, joints[1] * resizeFactor, joints[8] * resizeFactor, color[0], lineThickness);

			//left leg
			if (joints[9].x != 0 && joints[8].x != 0)
				line(img, joints[9] * resizeFactor, joints[8] * resizeFactor, color[0], lineThickness);
			if (joints[10].x != 0 && joints[9].x != 0)
				line(img, joints[9] * resizeFactor, joints[10] * resizeFactor, color[0], lineThickness);
			if (joints[11].x != 0 && joints[10].x != 0)
				line(img, joints[11] * resizeFactor, joints[10] * resizeFactor, color[0], lineThickness);

			//right leg
			if (joints[8].x != 0 && joints[12].x != 0)
				line(img, joints[8] * resizeFactor, joints[12] * resizeFactor, color[0], lineThickness);
			if (joints[13].x != 0 && joints[12].x != 0)
				line(img, joints[12] * resizeFactor, joints[13] * resizeFactor, color[0], lineThickness);
			if (joints[14].x != 0 && joints[13].x != 0)
				line(img, joints[14] * resizeFactor, joints[13] * resizeFactor, color[0], lineThickness);

			//right eye+ ear
			if (joints[0].x != 0 && joints[15].x != 0)
				line(img, joints[0] * resizeFactor, joints[15] * resizeFactor, color[0], lineThickness);
			if (joints[15].x != 0 && joints[17].x != 0)
				line(img, joints[15] * resizeFactor, joints[17] * resizeFactor, color[0], lineThickness);

			//left eye+ ear
			if (joints[0].x != 0 && joints[16].x != 0)
				line(img, joints[0] * resizeFactor, joints[16] * resizeFactor, color[0], lineThickness);
			if (joints[16].x != 0 && joints[18].x != 0)
				line(img, joints[16] * resizeFactor, joints[18] * resizeFactor, color[0], lineThickness);

			//left foot	
			if (joints[11].x != 0 && joints[22].x != 0)
				line(img, joints[11] * resizeFactor, joints[22] * resizeFactor, color[0], lineThickness);
			if (joints[11].x != 0 && joints[23].x != 0)
				line(img, joints[11] * resizeFactor, joints[23] * resizeFactor, color[0], lineThickness);
			if (joints[11].x != 0 && joints[24].x != 0)
				line(img, joints[11] * resizeFactor, joints[24] * resizeFactor, color[0], lineThickness);

			//right foot	
			if (joints[14].x != 0 && joints[19].x != 0)
				line(img, joints[14] * resizeFactor, joints[19] * resizeFactor, color[0], lineThickness);
			if (joints[14].x != 0 && joints[20].x != 0)
				line(img, joints[14] * resizeFactor, joints[20] * resizeFactor, color[0], lineThickness);
			if (joints[14].x != 0 && joints[21].x != 0)
				line(img, joints[14] * resizeFactor, joints[21] * resizeFactor, color[0], lineThickness);
		}

		for (int i = 0; i < nJointsCOCO; i++)
			if (joints[i].x != 0.0)
				circle(img, (joints[i] - TopLeft)* resizeFactor, 2, color[0], lineThickness);
	}

	return;
}

static float distancePointLine(const cv::Point2d point, const cv::Vec3d & line);
static void drawEpipolarLines(const std::string& title, Mat F, const cv::Mat& img1, const cv::Mat& img2, const std::vector<Point2d> points1, const std::vector<Point2d> points2, const float inlierDistance = -1);
int VisualizeAllViewsEpipolarGeometry(char *Path, int fid, Corpus &CorpusInfo);
int VisualizeSiftMatchAllPairs(char *Path, int nCams, int fid, int *frameTimeStamp);

int VisualizeSfMMatchTable(char *Path, int nviews);
int VisualizeSfMMatchPair(char *Path);
int VisualizeSfMCorpusFeatures(char *Path);
int Visualize_VideoKeyFrame2CorpusSfM_Inliers(char *Path, int selectedCamId, int startF, int stopF, double resizeFactor = 0.5);

int VisualizeIntraReAssignedPeople(char *Path, int sCamId, int startF, int stopF);
int VisualizeExtraReassignedPeoplePerTriplet(char *Path, vector<int> &sCams, vector<int> &TimeStamp, int startF, int stopF, int increF, int mode = 1);
int VisualizeExtraReassignedPeople(char *Path, int nCams, int startF, int stopF);
int VisualizeReIDPerCam(char *Path, vector<int> &vCams, vector<int> &TimeStamp, int startF, int stopF, int increF);
int Visualize_Spacetime_ReassignedPeople(char *Path, vector<int> &vCams, vector<int> &TimeStamp, int startF, int stopF, int increF);

int VisualizePerPersonPerFrameKnnDescTemporalMatching(char *Path, int cid, int knn, int startF, int stopF, int increF, int VisIncreF, int rangeF);
int VisualizePerPersonAllFramesKnnDescTemporalMatching(char *Path, int cid, int startF, int stopF);

int VisualizeAllViewsDescMatchPerTimeInstance(char *Path, vector<int> &vCams, vector<int> &TimeStamp, int startF, int stopF, int increF, int debug = 0);
int VisualizeOneTrackedPersonAllViewsDescMatch(char *Path, int cid, vector<int> &TimeStamp, int startF, int stopF, int increF);
int VisualizePerPersonAllFramesKnnDescPoolingSpatiolMatching(char *Path, int cid, vector<int> &TimeStamp, int startF, int stopF);
int VisualizeProjected3DSkeleton(char *Path, int nCams, int startF, int stopF, int increF, bool withBA = true, double resizeFactor = 0.5, int debug = 0);

int SparsePointTrackingDriver(char *Path, int viewID, int startF, int rangeF);
int SparsePointMatchingDriver(char *Path, vector<int> &viewID, int startF);

//Feature extraction and matching
int SiftGPUPair(char *Fname1, char *Fname2, const float nndrRatio, const double fractionMatchesDisplayed);
int vfFeatPair(char *Fname1, char *Fname2, const float nndrRatio, const double fractionMatchesDisplayed);
int vlSiftPair(char *Path, char *Fname1, char *Fname2, float nndrRatio, double density, bool useBFMatcher = false);


void ExtractSiftGPU(char *Path, int cid, int fid, Mat &Img, vector<SiftKeypoint> &Feat, vector<uchar>& descriptorsU, SiftGPU* sift);
void ExtractSiftGPU(char *Path, vector<int> &nviews, int startF, int stopF, int increF, int HistogramEqual = 0, bool ROOTL1 = true);
void ExtractSiftCPU(char *Path, vector<int> &nviews, int startF, int stopF, int increF, int HistogramEqual = 0, int ExtractionMethod = 0, bool ROOTL1 = true);
void ExtractSiftGPU(char *Path, int camID, int frameID, Mat &Img, vector<Point2f> &Feat, SiftGPU* sift = 0, bool ROOTL1 = true);
void ExtractSiftCPU(char *Path, int camID, int frameID, Mat &Img, vector<Point2f> &Feat, bool ROOTL1 = true);
void ExtractSiftGPU(char *Path, int camID, int frameID, Mat &Img, vector<Point2f> &Feat, vector<float> &vscale, SiftGPU* sift = 0, bool ROOTL1 = true);
void ExtractSiftCPU(char *Path, int camID, int frameID, Mat &Img, vector<Point2f> &Feat, vector<float> &vscale, bool ROOTL1 = true);
void ExtractSiftGPU_Video(char *Path, vector<int> &nviews, int startF, int stopF, int increF, bool ROOTL1 = true);

int GeneratePointsCorrespondenceMatrix_SiftGPU(char *Path, int nviews, int timeID, int HistogramEqual, float nndrRatio = 0.8, int *frameTimeStamp = NULL, bool ROOTL1 = true, bool visualizeSIFT = false);
int GeneratePointsCorrespondenceMatrix_CPU(char *Path, int nviews, int timeID, int HistogramEqual, float nndrRatio, int *frameTimeStamp, int extractionMethodL, bool ROOTL1 = true);
int GeneratePointsCorrespondenceMatrix(char *Path, int nviews, int timeID);

int GenerateUnduplicatedCorpusMatchesList(char *Path, int knnMax);
int GenerateCameraIMatchList(char *Path, int CameraI, int startF, int stopF, int increF, int knnMax, int nNNKeyFrames = 6);

//Geometric filter
int USAC_FindFundamentalDriver(char *Path, int id1, int id2, int timeID);
int USAC_FindHomographyDriver(char *Path, int id1, int id2, int timeID);

//Lens correction
int LensCorrectionVideoDriver(char *Path, char *VideoName, double *K, double *distortion, int LensType, int nimages, double Imgscale = 1.0, double Contscale = 1.0, int interpAlgo = 5);
int LensCorrectionImageSequenceDriver(char *Path, double *K, double *distortion, int LensType, int StartFrame, int StopFrame, double Imgscale = 1.0, double Contscale = 1.0, int interpAlgo = 5);
void LensCorrectionImageSequenceDriver2(vector<std::string> &vNameIn, vector<std::string> &vNameOut, double *Intrinsic, double *distortion, int startF, int stopF, int nchannels, int LensType, int interpAlgo = 5);
void LensCorrectionImageSequenceDriver3(vector<std::string> &vNameIn, vector<std::string> &vNameOut, VideoData &VideoI, int startF, int stopF, int nchannels, int LensType, int rotated = 0, int interpAlgo = 5);

//Pick static frames for corpus recon
int PickStaticImagesFromVideo(char *PATH, char *VideoName, int SaveFrameDif, int redetectInterval, double percentile, double MovingThresh2, int &nNonBlurImages, bool visual);
int PickStaticImagesFromImages(char *PATH, int SaveFrameDif, int redetectInterval, double percentile, double MovingThresh2, bool visual);

//Matching
int TemplateMatchingECCDriver(char *Path, double *H, int matchingType, int MaxIter = 70, double termination_eps = 1e-6);
int RecomputeNMatches(char *Path, vector<int> TrackInst, int nCams);
int RefineInitialDynamicPointsAppearance(char *Path, int InstFid, int nCams, int NViewPlus = 3, double imgScale = 1920 / 1920);
int CheckPairWiseZNCC(char *Path, vector<int> &validCamID, int nCams, int nframes, int npts, int pid, int f_instance, int nViewsPlus, LKParameters LKArg, int *CamID, int *RealframeID, Point2f *FrameSyncedPointsDistorted, float *FrameSyncedS, Mat *Img, vector<double *> &ImgPara);

//Tracking
int WarpImageFlowDriver(char *Fin, char *Fout, char *FnameX, char *FnameY, int nchannels, int Gsigma, int InterpAlgo, bool removeStatic);
int TVL1OpticalFlowDriver(char *PATH, int selectedCam, int startF, int stopF, int increF, TVL1Parameters argGF, int forward = 1, int backward = 0, int SaveWarpedImage = 0);

int TrackOpenCVLK(char *Path, int startFrame, int stopFrame, int HarrisCornerPatch = 11, int PatchSize = 31, int npryLevels = 4, int nonMaxRadius = 5, int minFeatures = 1000, int maxFeatures = 50000, double successTrackingRatio = 0.5);
//int TrackAllPointsWithRefTemplateDriver(char *Path, int viewID, int startF, int increF, int fps = 60, int trackingTime = 2, int nWins = 2, int WinStep = 3, int cvPyrLevel = 5, double MeanSSGThresh = 400.0, int interpAlgo = 1);
//int TrackAllPointsWithRefTemplate_DenseFlowDriven_Driver(char *Path, int viewID, int startF, int increF, double fps = 60, double trackingTime = 2, int nWins = 2, int WinStep = 3, double MeanSSGThresh = 400.0, int noTemplateUpdate = 0, int interpAlgo = 1);
int TrackAllPointsWithRefTemplateDriver(char *Path, int viewID, int startF, int increF, int fps, int trackingTime, int nWins, int WinStep, int cvPyrLevel, double MeanSSGThresh, int interpAlgo);
int TrackAllPointsWithRefTemplate_DenseFlowDriven_Driver(char *Path, int viewID, int startF, int increF, double fps, double trackingTime, int nWins, int WinStep, double MeanSSGThresh, int noTemplateUpdate, int interpAlgo);

//TrackRange is automaticallly scaled up according to increF. So, if increF = 2, TrackRange = 10, it will track from 0:2:20
//int TrackAllCorpusPointsWithRefTemplateDriver(char *Path, int viewID, int startF, int increF, int TrackRange = 10, int nWins = 3, int WinStep = 3, int cvPyrLevel = 5, double MeanSSGThresh = 400.0, int distortionCorrected = 1, int interpAlgo = 1);
//int TrackHarrisPointsWithRefTemplateDriver(char *Path, int selectedCamID, int startF, int TrackRange, int increF, int HarrisminDistance = 40, int WinSize = 21, int cvPyrLevel = 3, int maxFeatures = 5000, int interpAlgo = 1);
int TrackAllCorpusPointsWithRefTemplateDriver(char *Path, int viewID, int startF, int increF, int TrackRange, int nWins, int WinStep, int cvPyrLevel, double MeanSSGThresh, int CameraNotCalibrated, int distortionCorrected, int interpAlgo);
int TrackCorpusFeatureToNonKeyFrames(char *Path, int viewID, int keyFrameID, int startF, int TrackRange, int winDim = 31, int npryLevels = 3, double bidir_Thresh = 1.0, double successConsecutiveTrackingRatio = 0.75, double successRefTrackingRatio = 0.4, double avgcflowMagThresh = 100.0, int interpAlgo = -1, bool highQualityTracker = true, int nThreads = omp_get_max_threads(), int display = 0);
int TrackGlocalLocalCorpusFeatureToNonKeyFrames(char *Path, int viewID, int keyFrameID, int startF, int TrackRange, int winDim = 31, int npryLevels = 3, double bidir_Thresh = 1.0, double successConsecutiveTrackingRatio = 0.75, double successRefTrackingRatio = 0.4, double avgcflowMagThresh = 100.0, int interpAlgo = -1, bool highQualityTracker = true, int nThreads = omp_get_max_threads(), int display = 0);
int TrackHarrisPointsWithRefTemplateDriver(char *Path, int selectedCamID, int startF, int TrackRange, int increF, int HarrisminDistance, int WinSize, int cvPyrLevel, int nHarrisPartitions, int maxFeatures, int interpAlgo, int debug = 0);
int MergeTrackedCorpusFeaturetoNonKeyframe(char *Path, vector<int> &sCams, int startF, int stopF, int increF);
int MergeTrackedCorpusFeaturetoNonKeyframe2(char *Path, int camID, vector<int> &KeyFrameID2LocalFrameID, int startF, int stopF, int increF);
int MergeTrackedCorpusPoints(char *Path, int camID, int startF, int stopF, int increF);

int VideoBasedBlurDetection(char *Path, int SelectedCamera, int startF, int stopF, vector<int> &goodFid);
int VideoBasedBlurDetection2(char *Path, int SelectedCameraID, int startF, int stopF, vector<int> &goodFid);
int KeyFramesViaOpticalFlow(char *Path, int SelectedCamera, vector<int> &clearFrames, double successTrackingRatio = 0.75, double avgflowMagThresh = 40.0, int minnFeatures = 1000, int maxKFInternal = 60, int startF = 0, int stopF = -1, int rotateImage = 0, double scale = 1.0, int display = 0);
int KeyFramesViaOpticalFlow_HQ(char *Path, int SelectedCamera, int startF, int stopF, int extractedImages, int rotateImage, vector<int> &nonBlurFrames, double successConsecutiveTrackingRatio = 0.75, double successRefTrackingRatio = 0.5, double avgflowMagThresh = 40.0, int minnFeatures = 1000, int minKFInternal = 10, int maxKFInternal = 60, double bidir_Thresh = 1.0, int interpAlgo = -1, bool HighQualityTracking = true, int nThreads = 1, int display = 0, bool binary = true, bool dedicatedCorpus = true);

vector<Point2i> GetKeyFrameID2LocalFrameID(char *Path, int  nCams);
int ExtractSiftGPUFromImageListDriver(char *Path, vector<Point2i> &vCidFid);
int ExtractSiftCPUFromImageListDriver(char *Path, vector<Point2i> &vCidFid);
int GenCorpusImgsFromKeyFrames(char *Path, vector<int> &SelectedCams, vector<Point2i> &rotateImage, vector<CameraData> &InitCamera, int Sample = 1);
int GenCorpusImgsFromCachedText(char *Path, int nCams);
int GenMaskedCorpusFromKeyFrames(char *Path, int nCams);
int DenseMatchingDriver(char *PATH, int *sCams, int *sFid, int nchannels, int semiDense);

int VisualizeKeyFrameCorpusFeatures(char *Path, double resizeFactor = 0.5);
int VisualizeCorpusFeaturesAcrossVideoCamera(char *Path, int nCams, int startF, int stopF, int increF);
int VisualizeTracking(char *Path, int viewID, int startF, int increF, double fps, double trackingTime, int module = 1, int drawScale = 0, int trueStartF = -1, int DenseDriven = 0);
int VisualizeTrackingFull(char *Path, int viewID, int startF, int stopF, int increF, int drawScale = 0, int extractedImages = 0, double resizeFactor = 0.5, int writeImage = 0);
int VisualizeKeyFrameFormationTracking(char *Path, int viewID, int startF, int vis = 0);
int VisualizeProjectedSMPLBody(char *Path, int nCams, int startF, int stopF, int increF, int PointFormat, int maxPeople, double resizeFactor = 0.5, int WriteVideo = 1, int debug = 0);
int VisualizeAllViewsEpipolarGeometry(char *Path, int fid, Corpus &CorpusInfo);

static void drawArrows(Mat& frame, Point2f  prevPts, Point2f nextPts, Scalar line_color = Scalar(0, 0, 255), int line_thickness = 1);
int VisualizeTrackingErr(char *Path, int viewID, int startF, int increF, int fps, int trackingTime);
int VisualizeInliersHarrisTracking(char *Path, int CamID, int startF, int increF, int TrackRange);
Mat DrawTitleImages(vector<Mat>& Img, double desiredRatio = 16.0 / 9);
int VisualizeComparisionSlider(char *Path, char *Path1, char *Path2, char *Path3, vector<int> &sCams, int startF, int stopF);

int GenerateTrackingVisibilityImage(char *Path, int nCams, int startTrackingF, int stopTrackingF, int increTrackingInstance, int*TimeStamp, int TrackRange, int increF, int maxPts = 5000);

int TrajectoryReProjectionError(char *Path, int nCams, int npts, int startFrame, int stopFrame);
int ComputeRectifiedPatch(char *Path, int viewID, int startF, int hsubset = 32, double scaling = 2.0);
int ComputeRectifiedPatch(char *Path, int viewID, int hsubset = 32, double scaling = 2.0);

int SequentialHomoRanSacDriver(char *Path, Point2i refCid_Fid, Point2i nrCid_Fid);

//Triangulation and geometric sync
int VisualizeAllViewsEpipolarGeometry(char *Path, int nViews, int startF, int stopF);
int VisualizeAllViewsEpipolarGeometry(char *Path, vector<int> &sCams, int startF, int stopF);
int VisualizeAllTwoViewsTriangulation(char *Path, vector<int> &sCams, int startF, int stopF);
double TriangulatePointsFromArbitaryCameras(char *Path, int nViews, int distortionCorrected, int maxPts, double threshold);
double TriangulatePointsFromCorpusCameras(char *Path, int distortionCorrected, int maxPts, double threshold = 2.0);
double TriangulatePointsFromNonCorpusCameras(char *Path, vector<int> SelectedCams, int stopF, int distortionCorrected, int maxPts, double threshold = 10);
int TriangulatePointsFromNonCorpusCameras(char *Path, vector<int> SelectedCameras, int *DelayOffset, int refFid = 0);
int GeometricConstraintSyncDriver(char *Path, int nCams, int npts, int realStartFrame, int startFrame, int stopTime, int Range, bool GivenF, double *OffsetInfo, bool HasInitOffset = false);
int GeometricConstraintSyncDriver(char *Path, int nCams, int npts, int realStartFrame, int startFrame, int stopTime, int Range, bool GivenF, double *OffsetInfo, bool HasInitOffset);
int GeometricSyncAllInstancesDriver(char *Path, int nCams, int npts, vector<int> &TrackingInstance, int TrajRange, int startFrame, int stopFrame, int increImgFrames, int SearchRange, double TriangThresh, double *TimeStampOffset, bool HasInitOffset);

//Spatial calibration: checkerboard
int AssembleCheckerboardTrajectory(char *Path, int selectedCam, int startF, int stopF, int npts);
int CheckerBoardDetection(char *Path, int viewID, int startF, int stopF, int bw, int bh);
int SingleCameraCalibration(char *Path, int camID, int startFrame, int stopFrame, int bw, int bh, bool hasPoint, int step, float squareSize, int calibrationPattern, int width = 1920, int height = 1080, bool showUndistorsed = false);

int GlobalShutterBundleAdjustmentDriver(char *Path, int nViews, int distortionCorrected, vector< int> SharedIntrinsicCamID, int nViewsPlus, int LossType = 0);
int CayleyRollingShutterBundleAdjustmentDriver(char *Path, int nViews, int distortionCorrected, vector< int> SharedIntrinsicCamID, int nViewsPlus, int LossType = 0);
int Virtual3D_RS_BA_Driver(char *Path, int selectedCam, int startF, int stopF, int increF, int LossType = 0);

int BuildCorpus(char *Path, int distortionCorrected, int ShutterModel, int fixIntrinsic = 0, int fixDistortion = 0, int fixPose = 0, int fix3D = 0, int fixSkew = 0, int fixPrism = 0, int NDplus = 5, int LossType = 1);
int BuildCorpusVisualSfm(char *Path, int distortionCorrected, int ShutterModel, int fixSkew = 0, int fixPrism = 0, int NDplus = 5, int LossType = 0);
int Build3DFromSyncedImages(char *Path, int nviews, int startTime, int stopTime, int timeStep, int LensType, int distortionCorrected, int NDplus, double Reprojectionthreshold, double DepthThresh, int *frameTimeStamp = NULL, bool SaveToDenseColMap = false, bool Save2DCorres = false, bool Gen3DPatchFile = false, double Patch_World_Unit = 1.0, bool useRANSAC = true);

int BundleAdjustDomeTableCorres(char *Path, int startF_HD, int stopF_HD, int startF_VGA, int stopF_VGA, bool fixIntrinsic, bool fixDistortion, bool fixPose, bool fixIntrinsicVGA, bool fixDistortionVGA, bool fixPoseVGA, bool debug);
int BundleAdjustDomeMultiNVM(char *Path, int nNvm, int maxPtsPerNvM, bool fixIntrinsic, bool fixDistortion, bool fixPose, bool debug);
int ReCalibratedFromGroundTruthCorrespondences(char *Path, int camID, int startFrame, int stopFrame, int Allnpts, int ShutterModel = 0);
int RefineVisualSfMAndCreateCorpus(char *Path, int nimages, int ShutterModel, double threshold, int fixedIntrinsc, int fixedDistortion0, int fixedPose, int fixedfirstCamPose, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, int doubleRefinement);

int LocalizeCameraToCorpusDriver(char *Path, int StartFrame, int StopFrame, int IncreFrame, int module, int selectedCam, int distortionCorrected, int nInlierThresh, int GetIntrinsicFromCorpus);
int ForceLocalizeCameraToCorpusDriver(char *Path, int startFrame, int stopFrame, int IncreFrame, int selectedCam, int distortionCorrected, int nInlierThresh, int fromKeyFrameTracking);

int ClassifyPointsFromTriangulationSingleCam(char *Path, int CamID, int startFInst, int stopFInst, int increFInst, int trackingRange, int increF, int nInlierThresh, double TriangThesh);
int LocalBA_HarrisTracking_And_SiftMatching(char *Path, int selectedCamID, int InstF, int nInst, int increInstF, int rangeF, int increF, int CorpusDistortionCorrected, int fixIntrinsic, int fixDistortion, int fixPose, int fix3D, int fixSkew, int fixPrism, double weightHvsC = 0.5, int nplus = 5, double threshold = 2.0, int sharedInstrinsic = 1);

int convertPnP2KPnP(char *Path, int selectedCam, int startF, int stopF);
int SimpleBodyPoseSfM(char *Path, int fid, Corpus &CorpusInfo, SfMPara mySfMPara, int nMaxPeople = 20, int verbose = 0);
int SimpleBodyPoseSfM_Tracking(char *Path, int refF, int rangeF, int nCams, int nMaxPeople = 20);
int SequenceSfMDriver(char *Path, int sCamId, int startF, int stopF, SfMPara mySfMPara, int debug = 0);
int VideoKeyframe2Corpus_SFM(char *Path, int selectedCamId, SfMPara mySfMPara);

int BuildCorpusAndLocalizeCameraBatch(char *Path, SfMPara mySfMPara, int module);

int TestKSfM(char *Path, int selectedCamId, int distortionCorrected = 0, int nInliersThresh = 30);
int TestPnPf(char *Path, int selectedCamId, int startF, int stopF, int distortionCorrected = 0, int nInliersThresh = 30);

//Spatialtemporal calib
int ConvertTrajectoryToPointCloudTime(char *Path, int npts);
int ResamplingOf3DTrajectorySplineDriver(char *Path, vector<int> &SelectedCams, vector<double> &OffsetInfo, int startFrame, int stopFrame, int ntracks, double lamda_Data);
int ResamplingOf3DTrajectoryDCTDriver(char *Path, vector<int> &SelectedCams, vector<double> &OffsetInfo, int PriorOrder, int startFrame, int stopFrame, int ntracks, double lamda_Data, double lamda_Reg);
int ResamplingOf3DTrajectoryDCTDriverParallel(char *Path, vector<int> &SelectedCams, vector<double> &OffsetInfo, int TargetTrackID, int PriorOrder, int startFrame, int stopFrame, int ntracks, double lamda_Data, double lamda_Reg);

int MotionPriorSTReconstructionDriver(char *Path, int nCams, int startFrame, int stopFrame, int npts, double RealOverSfm, double lamdaData = 0.8, int SearchRange = 10, double SearchStep = 0.1, int module = 1);
int SpatialTemporalCalibInTheWildDriver(char *Path, int nCams, int RefStartF, int RefStopF, int RefIncreF, int TrackInstanceStartF, int TrackInstanceStopF, int TrackInstanceStepF, int fps, int TrackTime, int module);

int StaticDynamicSpatialTemporalBundleAdjustmentDriver(char *Path, int nCams, int startF, int stopF, int StaticIncreF,
	int fixIntrinsic, int  fixDistortion, int fixPose, int fixLocalPose, int fix3D, int sharedIntrinsicIndiCam, int fixSkew, int fixPrism, int distortionCorrected,
	int LossType, double threshold, int nViewsPlus, double reprojectThreshold, double Tscale, double eps, double lamdaStatic, double lamdaDynaData, double RealOverSfm);

struct LeastMotionPriorCost3DCeres2 {
	LeastMotionPriorCost3DCeres2(double timeStamp1, double timeStamp2, double sig_ivel) : timeStamp1(timeStamp1), timeStamp2(timeStamp2), sig_ivel(sig_ivel) {}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, T* residuals) 	const
	{
		T temp = (T)(sig_ivel / ceres::sqrt(ceres::abs(timeStamp2 - timeStamp1)));
		for (int ii = 0; ii < 3; ii++)
			residuals[ii] = (xyz2[ii] - xyz1[ii]) * temp;  //(v/sig_v)^2*dt = (dx/dt/sig_v)^2*dt = (dx/sig_v)^2/dt

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double Stamp1, double Stamp2, double sig_ivel)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCost3DCeres2, 3, 3, 3>(new LeastMotionPriorCost3DCeres2(Stamp1, Stamp2, sig_ivel)));
	}
	static ceres::CostFunction* CreateNumerDiff(double Stamp1, double Stamp2, double sig_ivel)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorCost3DCeres2, ceres::CENTRAL, 3, 3, 3>(new LeastMotionPriorCost3DCeres2(Stamp1, Stamp2, sig_ivel)));
	}
	double timeStamp1, timeStamp2, sig_ivel;
};

struct ConstantLimbLengthCost3DCeres {
	ConstantLimbLengthCost3DCeres(double meanLimbLength, double sig_ivel) : sig_ivel(sig_ivel), meanLimbLength(meanLimbLength) {}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, T* residuals) 	const
	{
		T limbLength = ceres::sqrt(pow(xyz1[0] - xyz2[0], 2) + pow(xyz1[1] - xyz2[1], 2) + pow(xyz1[2] - xyz2[2], 2) + 1e-25);
		residuals[0] = (limbLength - (T)meanLimbLength)* sig_ivel;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double meanLimbLength, double sig_ivel)
	{
		return (new ceres::AutoDiffCostFunction<ConstantLimbLengthCost3DCeres, 1, 3, 3>(new ConstantLimbLengthCost3DCeres(meanLimbLength, sig_ivel)));
	}
	static ceres::CostFunction* CreateNumerDiff(double meanLimbLength, double sig_ivel)
	{
		return (new ceres::NumericDiffCostFunction<ConstantLimbLengthCost3DCeres, ceres::CENTRAL, 1, 3, 3>(new ConstantLimbLengthCost3DCeres(meanLimbLength, sig_ivel)));
	}
	double meanLimbLength, sig_ivel;
};
struct ConstantLimbLengthCost3D_SimilarityTrans_Ceres {
	ConstantLimbLengthCost3D_SimilarityTrans_Ceres(Point3d xyz1, Point3d xyz2, double meanLimbLength, double isig) :xyz1(xyz1), xyz2(xyz2), isig(isig), meanLimbLength(meanLimbLength) {}

	template <typename T>	bool operator()(const T* const srt, T* residuals) 	const
	{
		T limbLength = (T)(ceres::sqrt(pow(xyz1.x - xyz2.x, 2) + pow(xyz1.y - xyz2.y, 2) + pow(xyz1.z - xyz2.z, 2) + 1e-25))* srt[0]; //scale
		residuals[0] = (limbLength - (T)meanLimbLength)* isig;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(Point3d xyz1, Point3d xyz2, double meanLimbLength, double isig)
	{
		return (new ceres::AutoDiffCostFunction<ConstantLimbLengthCost3D_SimilarityTrans_Ceres, 1, 7>(new ConstantLimbLengthCost3D_SimilarityTrans_Ceres(xyz1, xyz2, meanLimbLength, isig)));
	}
	static ceres::CostFunction* CreateNumerDiff(Point3d xyz1, Point3d xyz2, double meanLimbLength, double isig)
	{
		return (new ceres::NumericDiffCostFunction<ConstantLimbLengthCost3D_SimilarityTrans_Ceres, ceres::CENTRAL, 1, 7>(new ConstantLimbLengthCost3D_SimilarityTrans_Ceres(xyz1, xyz2, meanLimbLength, isig)));
	}
	Point3d xyz1, xyz2;
	double meanLimbLength, isig;
};
struct LeastMotionPriorCost3D_SimilarityTrans_Ceres {
	LeastMotionPriorCost3D_SimilarityTrans_Ceres(Point3d xyz1, Point3d xyz2, double timeStamp1, double timeStamp2, double sig_ivel) : xyz1(xyz1), xyz2(xyz2), timeStamp1(timeStamp1), timeStamp2(timeStamp2), sig_ivel(sig_ivel) {}

	template <typename T>	bool operator()(const T* const srt1, const T* const srt2, T* residuals) 	const
	{
		T temp = (T)(sig_ivel / ceres::sqrt(ceres::abs(timeStamp2 - timeStamp1)));

		T point1[3] = { T(xyz1.x), T(xyz1.y), T(xyz1.z) }, tpoint1[3];
		ceres::AngleAxisRotatePoint(srt1 + 1, point1, tpoint1);
		tpoint1[0] = srt1[0] * tpoint1[0] + srt1[4], tpoint1[1] = srt1[0] * tpoint1[1] + srt1[5], tpoint1[2] = srt1[0] * tpoint1[2] + srt1[6];

		T point2[3] = { T(xyz2.x), T(xyz2.y), T(xyz2.z) }, tpoint2[3];
		ceres::AngleAxisRotatePoint(srt2 + 1, point2, tpoint2);
		tpoint2[0] = srt2[0] * tpoint2[0] + srt2[4], tpoint2[1] = srt2[0] * tpoint2[1] + srt2[5], tpoint2[2] = srt2[0] * tpoint2[2] + srt2[6];

		for (int ii = 0; ii < 3; ii++)
			residuals[ii] = (tpoint1[ii] - tpoint2[ii]) * temp;  //(v/sig_v)^2*dt = (dx/dt/sig_v)^2*dt = (dx/sig_v)^2/dt

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(Point3d xyz1, Point3d xyz2, double Stamp1, double Stamp2, double sig_ivel)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCost3D_SimilarityTrans_Ceres, 3, 7, 7>(new LeastMotionPriorCost3D_SimilarityTrans_Ceres(xyz1, xyz2, Stamp1, Stamp2, sig_ivel)));
	}
	Point3d xyz1, xyz2;
	double timeStamp1, timeStamp2, sig_ivel;
};
struct ConstantLimbLengthCost3DCeres2 {
	ConstantLimbLengthCost3DCeres2(double isigma) : isigma(isigma) {}

	template <typename T>	bool operator()(const T* meanLimbLength, const T* const xyz1, const T* const xyz2, T* residuals) 	const
	{
		T limbLength = ceres::sqrt(pow(xyz1[0] - xyz2[0], 2) + pow(xyz1[1] - xyz2[1], 2) + pow(xyz1[2] - xyz2[2], 2) + 1e-25);
		residuals[0] = (limbLength - meanLimbLength[0])* isigma;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double isigma)
	{
		return (new ceres::AutoDiffCostFunction<ConstantLimbLengthCost3DCeres2, 1, 1, 3, 3>(new ConstantLimbLengthCost3DCeres2(isigma)));
	}
	static ceres::CostFunction* CreateNumerDiff(double isigma)
	{
		return (new ceres::NumericDiffCostFunction<ConstantLimbLengthCost3DCeres2, ceres::CENTRAL, 1, 1, 3, 3>(new ConstantLimbLengthCost3DCeres2(isigma)));
	}
	double  isigma;
};
struct SymLimbLengthCost3DCeres {
	SymLimbLengthCost3DCeres(double isigma) : isigma(isigma) {}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const xyz1_, const T* const xyz2_, T* residuals) 	const
	{
		T limbLength = ceres::sqrt(pow(xyz1[0] - xyz2[0], 2) + pow(xyz1[1] - xyz2[1], 2) + pow(xyz1[2] - xyz2[2], 2) + 1e-25);
		T limbLength_ = ceres::sqrt(pow(xyz1_[0] - xyz2_[0], 2) + pow(xyz1_[1] - xyz2_[1], 2) + pow(xyz1_[2] - xyz2_[2], 2) + 1e-25);
		residuals[0] = (limbLength - limbLength_)* isigma;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double isigma)
	{
		return (new ceres::AutoDiffCostFunction<SymLimbLengthCost3DCeres, 1, 3, 3, 3, 3>(new SymLimbLengthCost3DCeres(isigma)));
	}
	static ceres::CostFunction* CreateNumerDiff(double isigma)
	{
		return (new ceres::NumericDiffCostFunction<SymLimbLengthCost3DCeres, ceres::CENTRAL, 1, 3, 3, 3, 3>(new SymLimbLengthCost3DCeres(isigma)));
	}
	double  isigma;
};
struct SymLimbLengthCost3DCeres2 {
	SymLimbLengthCost3DCeres2(double isigma) : isigma(isigma) {}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const xyz2_, T* residuals) 	const
	{
		T limbLength = ceres::sqrt(pow(xyz1[0] - xyz2[0], 2) + pow(xyz1[1] - xyz2[1], 2) + pow(xyz1[2] - xyz2[2], 2) + 1e-25);
		T limbLength_ = ceres::sqrt(pow(xyz1[0] - xyz2_[0], 2) + pow(xyz1[1] - xyz2_[1], 2) + pow(xyz1[2] - xyz2_[2], 2) + 1e-25);
		residuals[0] = (limbLength - limbLength_)* isigma;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double isigma)
	{
		return (new ceres::AutoDiffCostFunction<SymLimbLengthCost3DCeres2, 1, 3, 3, 3>(new SymLimbLengthCost3DCeres2(isigma)));
	}
	static ceres::CostFunction* CreateNumerDiff(double isigma)
	{
		return (new ceres::NumericDiffCostFunction<SymLimbLengthCost3DCeres2, ceres::CENTRAL, 1, 3, 3, 3>(new SymLimbLengthCost3DCeres2(isigma)));
	}
	double  isigma;
};
int AssociatePeopleAndBuild3DFromCalibedSyncedCameras_RansacTri(char *Path, vector<int> &SelectedCameras, int startFrame, int stopFrame, int timeStep, int LensType, int distortionCorrected, int nViewsPlus, double Reprojectionthreshold, double angleThesh, double detectionThresh, int iterMax);
int AssociatePeopleAndBuild3DFromCalibedSyncedCameras_3DVoting(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int increF, int LensType, int distortionCorrected, int nViewPlus = 4, double detectionThresh = 0.2, double reprojectionThesh = 15.0, double real2SfM = 1.0, double cellSize = 20);
int MultiviewPeopleAssociationTrifocal(char *Path, vector<int> &sCams, vector<int> &TimeStamp, int startF, int stopF, int increF);

int ExtractColorForCorpusCloud(char *Path);
int ReBundleFromDifferentShutterModel(char *Path);
int TriangulateSkeleton3DFromCalibSyncedCameras(char *Path, vector<int> &SelectedCams, int startF, int stopF, int increF, int distortionCorrected, int skeletonPointFormat, int nViewsPlus, double Reprojectionthreshold, double detectionThresh, int iterMax = 100, int nMinPointsRanSac = 3);
int WindowSkeleton3DBundleAdjustment(char *Path, int nCams, int startF, int stopF, int increF, int distortionCorrected, int skeletonPointFormat, double detectionThresh, int LossType, double *Weights, double *iSigma, double real2SfM = 1.0);
int TriangulateSkeleton3DFromCalibSyncedCameras_DensePose(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, vector<int> &SelectedCams, int startF, int stopF, int increF, int distortionCorrected, int skeletonPointFormat, int nViewsPlus, double Reprojectionthreshold, double detectionThresh, int iterMax, int nMinPointsRanSac);
int WindowSkeleton3DBundleAdjustment_DensePose(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, int nCams, int startF, int stopF, int increF, int distortionCorrected, int skeletonPointFormat, double detectionThresh, int LossType, double *Weights, double *iSigma, double real2SfM, int missingFrameInterp);

int CollectNearestViewsBaseOnGeometry(char *Path, vector<int> &sCams, int startF, int stopF, int kNN, bool useSyncedFramesOnly);
int SemanticVisualHull(char *Path, vector<int> &sCams, int startFrame, int stopFrame, int increF, int distortionCorrected, int nViewPlus, double real2SfM, double cellSize, double *R_adjust = NULL, double *T_adjust = NULL, int debug = 0);

int TrackBody_Landmark_BiDirectLK(char *Path, int cid, int startF, int stopF, int increF, int nJoints, double bwThresh = 20, int debug = 0);
int TrackBody_SmallFeat_BiDirectLK(char *Path, int cid, int startF, int stopF, int increF, double biDirectThresh = 5, double ConsistencyercentThresh = 0.7, double overlapA_thresh = 0.3, int debug = 0);
int TrackBody_Desc_BiDirect(char *Path, int cid, int startF, int stopF, int increF, double simThresh = 0.9, double ratioThesh = 0.8, double overlapA_thresh = 0.3, int debug = 0);

int CleanTrackletBy2DSmoothing(char *Path, int sCamId, int startF, int stopF, int increF, int imWidth, double dispThresh = 30, double  overDispRatioThresh = 1.5, int debug = 0);
int CleanTrackletBy2DSmoothing_V2(char *Path, int sCamId, int startF, int stopF, int increF, int imWidth, int nJoints, double dispThresh = 30, double  overDispRatioThresh = 1.5, int debug = 0);
int PerVideoMultiPeopleTracklet(char *Path, int sCamId, int startF, int stopF, int increF, int nJoints, int debug = 0);
//Test drivers + demo
int Test();

//Blob detection
int DetectBalls(char *Path, int camID, const int startFrame, const int stopFrame, int search_area = 10, double threshold = 0.75);
int DetectRGBBallCorrelation(char *ImgName, vector<KeyPoint> &kpts, vector<int> &ballType, int nOctaveLayers, int nScalePerOctave, double sigma, int PatternSize, int NMS_BW, double thresh, bool visualize);
int DetectRedLaserCorrelationMultiScale(char *ImgName, int width, int height, unsigned char *MeanImg, vector<Point2d> &kpts, double sigma, int PatternSize, int nscales, int NMS_BW, double thresh, bool visualize, unsigned char *ColorImg, float *colorResponse, double *DImg, double *ImgPara, double *maskSmooth, double *Znssd_reqd);

int CornerDetectorDriver(char *Path, int checkerSize, double ZNCCThreshold, int startF, int stopF, int width, int height);

//ARtag tracker
int ARTag_GenerateVisibilityMatrix(char *Path, int nCams, int npts, int nframes);
int ARTag_TrackMissingMarkersIndiCorner(char *Path, int camID, int npts, int nframes, int subsetSize = 15, int subsetStep = 3, int subsetScale = 3, double Dist2Thesh = 1.0, int PryLevel = 4);
int ARTag_TrackMissingMarkers(char *Path, int camID, int npts, int nframes, int backward = 1, double ZNCCThresh = 0.75, bool Debug = false);


int GroundPlanFittingDriver(char *FnameIn, char *FnameOut);
int AllBackgroundButHumanBlurring(char *Path, int cid, int startF, int stopF, int increF, int nPeople, int w = 1920, int h = 1080, double resizeFactor = 0.5);
#endif