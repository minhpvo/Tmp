#if !defined(GEOMETRY1_H )
#define GEOMETRY1_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <complex>
#include <omp.h>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <ceres/ceres.h>
#include <ceres/types.h>
#include <ceres/rotation.h>
#include "glog/logging.h"

#include "../DataStructure.h"
#include "../TrackMatch/MatchingTracking.h"
#include "../TrackMatch/FeatureEst.h"
#include "../Ulti/GeneralUlti.h"
#include "../ImgPro/ImagePro.h"
#include "../Ulti/MiscAlgo.h"
#include "../Ulti/MathUlti.h"
#include "../Ulti/DataIO.h"

#include "../ThirdParty/USAC/FundamentalMatrixEstimator.h"
#include "../ThirdParty/USAC/HomographyEstimator.h"
#include "../ThirdParty/USAC/USAC.h"
#include "../ThirdParty/SiftGPU/src/SiftGPU/SiftGPU.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif


using namespace cv;


void NormalizePointsForDLT(vector<Point2d> &pts, double *H);
int computeFmat8Point(vector<Point2d> &pts1, vector<Point2d> &pts2, double *Fmat);
int EvaluateFmat(vector<Point2d> &pts1, vector<Point2d> &pts2, double *Fmat, vector<int>  &vinliers, double &meanErr, double thresh);

Eigen::Matrix<double, 3, 3, Eigen::RowMajor> computeHomography4Point(std::vector<Point2d> pts1, std::vector<Point2d> pts2);

//USAC:
int USAC_FindFundamentalMatrix(ConfigParamsFund cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Fmat, vector<int>&InlierIndicator, int &ninliers);
int USAC_FindHomography(ConfigParamsHomog cfg, vector<Point2d> pts1, vector<Point2d>pts2, double *Hmat, vector<int>&InlierIndicator, int &ninliers);

//PnP
//int P4Pf_RANSAC(CameraData &camera, vector<int> &BestInliers, vector<Point3d> &Vxyz, vector<Point2d> &Vuv, int width, int height, int LensModel, int MaxIter, int nInlierThresh, double thresh);
int PnP_RANSAC(CameraData &camera, vector<int> &BestInliers, vector<Point3d> &Vxyz, vector<Point2d> &Vuv, int width, int height, int LensModel, int MaxIter, int nInlierThresh, double thresh, int fixIntrinisc, int disortionCorrectedMode, int method);
int RP6P_RANSAC(double *R, double *T, vector<int> &BestInliers, double *K, double *distortion, int LensModel, Point2d *P2d, Point3d *P3d, int npts, int distortionCorrected, int MaxIter, int nInlierThresh, double thresh);

int ComputeTrifocalTensorDLT(vector<Point2d> pts1, vector<Point2d> pts2, vector<Point2d> pts3, double *T);
void TrifocalTensorPointTransfer(double *T, Point2d &pt1, Point2d &pt2, Point2d &pt3);
int EvaluteTfocal(vector<Point2d> &pts1, vector<Point2d> &pts2, vector<Point2d> &pts3, double *T, vector<int>  &vinliers, double &meanErr, double thresh);
void GetEpipoleFromTrifocalTensor(double *T, double *e2, double *e3);
void GetProjFromTrifocalTensor(double *T, MatrixXd &P1, MatrixXd &P2);
void GetTensorFromProj(MatrixXd& P1, MatrixXd& P2, double* T);

//the returned pts will be modified (undistored)!
int EstimatePoseAndInliers(double *K, double *distortion, int LensModel, int ShutterModel, double *R, double *T, double *wt, vector<Point2d> &pts, vector<Point3d> &ThreeD, vector<int>  &Inliers, double thresh, int fixIntrinsic, int distortionCorrected, double minFRatio, double maxFRatio, int width, int height, int PnPAlgo);
//int EstimateFocalPoseAndInliers(double *K, int ShutterModel, double *R, double *T, double *wt, vector<Point2d> &pts, vector<Point3d> &ThreeD, double thresh, vector<int> &Inliers, int width, int height);

void UndistortAndRectifyPoint(double *K, double *distortion, double *R, double *nK, Point2d &uv);
void UndistortAndRectifyPoint(double *K, double *distortion, double *R, double *nK, Point2f &uv);

Vector3d unproject(const Point2d& uv, const Eigen::VectorXd & params);
void LensCorrectionPoint_KB3(Point2d &uv, double *intrinsic, double *distortion);
void LensDistortionPoint_KB3(Point2d &img_point, double *intrinsic, double *distortion);

/////////////////////////////calib_new.txt
void FishEyeDistortionPoint(Point2d *Points, double omega, double DistCtrX, double DistCtrY, int npts = 1);
void FishEyeCorrectionPoint(Point2d *Points, double omega, double DistCtrX, double DistCtrY, int npts = 1);
void FishEyeCorrectionPoint(Point2f *Points, double omega, double DistCtrX, double DistCtrY, int npts = 1);
void FishEyeDistortionPoint(vector<Point2d> &Points, double omega, double DistCtrX, double DistCtrY);
void FishEyeDistortionPoint(vector<Point2f> &Points, double omega, double DistCtrX, double DistCtrY);
void FishEyeCorrectionPoint(vector<Point2d> &Points, double omega, double DistCtrX, double DistCtrY);
void FishEyeCorrection(unsigned char *Img, int width, int height, int nchannels, double omega, double DistCtrX, double DistCtrY, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

void FishEyeCorrectionPoint(Point2d *Points, double *K, double omega, int npts = 1);
void FishEyeCorrectionPoint(Point2f *Points, double *K, double omega, int npts = 1);
void FishEyeCorrectionPoint(vector<Point2f> &Points, double *K, double omega);
void FishEyeCorrectionPoint(vector<Point2d> &Points, double *K, double omega);
void FishEyeDistortionPoint(Point2f *Points, double *K, double omega, int npts = 1);
void FishEyeDistortionPoint(Point2d *Points, double *K, double omega, int npts = 1);
void FishEyeDistortionPoint(vector<Point2d>&Points, double *K, double omega);
void FishEyeDistortionPoint(vector<Point2f>&Points, double *K, double omega);
void FishEyeCorrection(unsigned char *Img, int width, int height, int nchannels, double *K, double omega, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

/////////////////////////////calib.txt
void FishEyeCorrectionPoint(Point2d *Points, double *K, double* invK, double omega, int npts = 1);
void FishEyeDistortionPoint(Point2d *Points, double *K, double* invK, double omega, int npts = 1);
void FishEyeCorrection(unsigned char *Img, int width, int height, int nchannels, double *K, double* invK, double omega, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

void CC_Calculate_xcn_ycn_from_i_j(double i, double j, double &xcn, double &ycn, double *A, double *distortion, int Method);
void LensDistortionPoint(Point2f *img_point, double *K, double *distortion, int npts = 1);
void LensDistortionPoint(Point2d *img_point, double *K, double *distortion, int npts = 1);
void LensDistortionPoint2(Point2d *img_point, double *Intrinsic, double *distortion, int npts = 1);
void LensCorrectionPoint(vector<Point2f> &uv, double *K, double *distortion);
void LensDistortionPoint(vector<Point2d> &img_point, double *K, double *distortion);
void LensDistortionPoint(vector<Point2f> &img_point, double *K, double *distortion);
void LensCorrectionPoint(Point2d *uv, double *K, double *distortion, int npts = 1);
void LensCorrectionPoint(Point2f *uv, double *K, double *distortion, int npts = 1);
void LensCorrectionPoint(vector<Point2d> &uv, double *K, double *distortion);
void LensUndistortion(unsigned char *Img, int width, int height, int nchannels, double *K, double *distortion, int intepAlgo, double ImgMag, double Contscale, double *Para = NULL);

double FmatPointError(double *Fmat, Point2d p1, Point2d p2);
void computeFmat(double *K1, double *K2, double *R1, double *T1, double *R2, double *T2, double *Fmat);
void computeFmat(CameraData Cam1, CameraData Cam2, double *Fmat);
void computeFmatfromKRT(CameraData *CameraInfo, int nvews, int *selectedIDs, double *Fmat);
void computeFmatfromKRT(CorpusandVideo &CorpusandVideoInfo, int *selectedCams, int *seletectedTime, int ChooseCorpusView1, int ChooseCorpusView2, double *Fmat);

cv::Mat findEssentialMat(InputArray _points1, InputArray _points2, InputArray _cameraMatrix1, InputArray _cameraMatrix2, int method = CV_RANSAC, double prob = 0.99, double threshold = 2.0, OutputArray mask = noArray());
void decomposeEssentialMat(const Mat & E, Mat & R1, Mat & R2, Mat & t);
int recoverPose(const Mat & E, InputArray points1, InputArray points2, Mat & R, Mat & t, Mat K1, Mat K2, InputOutputArray mask = noArray());
int EssentialMatOutliersRemove(char *Path, int timeID, int id1, int id2, int nCams, int cameraToScan = -1, int ninlierThresh = 40, int distortionCorrected = 0, bool needDuplicateRemove = false);
int FundamentalMatOutliersRemove(char *Path, int timeID, int id1, int id2, int ninlierThresh = 40, int LensType = RADIAL_TANGENTIAL_PRISM, int distortionCorrected = 0, bool needDuplicateRemove = false, int nCams = 1, int cameraToScan = -1, int *frameTimeStamp = NULL);

int TwoViewsClean3DReconstructionFmat(CameraData &View1, CameraData &View2, vector<Point2d>imgpts1, vector<Point2d> imgpts2, vector<Point3d> &P3D);

void FisheyeProjectandDistort(Point3d WC, Point2d *pts, double *P, double *distortion, int nviews = 1);
void FisheyeProjectandDistort(Point3d WC, Point2d *pts, double *P, double *K, double *distortion, int nviews = 1);
void ProjectandDistort(Point3d WC, Point2d *pts, double *P, double *K = NULL, double *distortion = NULL, int nviews = 1);
void ProjectandDistort(Point3d WC, vector<Point2d> &pts, double *P, double *K = NULL, double *distortion = NULL);
void ProjectandDistort(vector<Point3d> WC, Point2d *pts, double *P, double *K = NULL, double *distortion = NULL, int nviews = 1);
void TwoViewTriangulation(Point2d *pts1, Point2d *pts2, double *P1, double *P2, Point3d *WC, int npts = 1);
void TwoViewTriangulation(vector<Point2d> pts1, vector<Point2d> pts2, double *P1, double *P2, vector<Point3d> &WC);
void TwoViewTriangulationQualityCheck(Point2d *pts1, Point2d *pts2, Point3d *WC, double *P1, double *P2, bool *GoodPoints, double thresh, int npts = 1, double *K1 = 0, double *K2 = 0, double *distortion1 = 0, double *distortion2 = 0);
void NviewTriangulation(Point2d *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B);
void NviewTriangulation(vector<Point2d> *pts, double *P, Point3d *WC, int nview, int npts, double *Cov, double *A, double *B);
void NviewTriangulation(CameraData *ViewInfo, int AvailViews, vector <vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<Point3d> &AllP3D, bool CayleyRS = false);
double NviewTriangulationRANSAC(Point2d *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int nMinPoints, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL, double *tP = NULL, bool nonlinear = false, bool refineRanSac = false);
double NviewTriangulationRANSAC(vector<Point2d> *pts, double *P, Point3d *WC, bool *PassedTri, vector<int> *Inliers, int nview, int npts, int nMinPoints, int MaxRanSacIter, double inlierPercent, double threshold, double *A = NULL, double *B = NULL, double *tP = NULL, bool nonlinear = false, bool refineRanSac = false);
void MultiViewQualityCheck(Point2d *Pts, double *Pmat, int LensType, double *K, double *distortion, bool *PassedPoints, int nviews, int npts, double thresh, Point3d *aWC, Point2d *apts = 0, Point2d *bkapts = 0, int *DeviceMask = 0, double *tK = 0, double *tdistortion = 0, double *tP = 0, double *A = 0, double *B = 0);

int FmatSyncBruteForce2DStereo(char *Path, int *SelectedCams, int realStartFrame, int startFrame, int stopFrame, int ntracks, int *OffsetInfo, int LowBound, int UpBound, bool GivenF, bool silent = true);
int ClassifyPointsFromTriangulationLite(char *Path, int nCams, int StartFInst, int nViewsPlus, double TriangThesh);
int ClassifyPointsFromTriangulation(char *Path, vector<int> &SelectedCams, int npts, vector<int> &frameTimeStamp, int startFrame, int stopFrame, int refFrame, int nViewsPlus, double TriangThesh, double stationaryThesh);
int FmatSyncBruteForce2DStereoAllInstances(char *Path, int *SelectedCams, vector<int> &TrackingInst, int TrajRange, int startFrame, int stopFrame, int increImgFrames, int allNpts, int *frameTimeStamp, int LowBound, int UpBound, bool GivenF, vector<ImgPtEle> *PerCam_UV, bool silent);
int FmatSyncRobustBruteForce2DStereoAllInstances(char *Path, int *SelectedCams, vector<int> &TrackingInst, int TrajRange, int startFrame, int stopFrame, int increImgFrames, int allNpts, int *frameTimeStamp, int LowBound, int UpBound, vector<ImgPtEle> *PerCam_UV, bool silent);
int TriangulateFrameSync2DTrajectories_Block(char *Path, vector<int> &SelectedCams, vector<int> &frameTimeStamp, int refFrame, int startFrame, int stopFrame, int nViewsPlus, double TriangThesh, double stationaryThesh, int CleanCorrespondencesByTriangulationTest, double *GTFrameTimeStamp = NULL, double *ialpha = NULL, double*Tscale = NULL);
int TriangulateFrameSync2DTrajectories(char *Path, vector<int> &SelectedCams, vector<int> &frameTimeStamp, int refFrame, int startFrame, int stopFrame, int nViewsPlus, double TriangThesh, double stationaryThesh, double MinDistToCamThresh, int Discon3DThresh, int CleanCorrespondencesByTriangulationTest, double *GTFrameTimeStamp = NULL, double *ialpha = NULL, double*Tscale = NULL);
int TriangulateFrameSync2DTrajectories_BK(char *Path, vector<int> &SelectedCams, vector<int> &FrameTimeStamp, int startFrame, int stopFrame, int npts, bool CleanCorrespondencesByTriangulationTest, double *GTFrameTimeStamp, double *ialpha, double*Tscale, int realStartFrame);
int TriangulateFrameSync2DTrajectories2(char *Path, vector<int> &SelectedCams, vector<int> &frameTimeStamp, int refFrame, int startFrame, int stopFrame, int nViewsPlus, double TriangThesh, double stationaryThesh, double MinDistToCamThresh, double Discon3DThresh, int CleanCorrespondencesByTriangulationTest, double *GTFrameTimeStamp = NULL, double *ialpha = NULL, double*Tscale = NULL);
int TriangulationSyncRobustBruteForce2DStereoAllInstances(char *Path, int *SelectedCams, vector<int> &TrackingInst, int TrajRange, int startFrame, int stopFrame, int increImgFrames, int allNpts, int *frameTimeStamp, int LowBound, int UpBound, vector<ImgPtEle> *PerCam_UV, double TriangThresh, bool silent);


//Find depth &scale wrst to camera
void Get3DPtfromDist(double *Kinv, double *Rinv, double *T, Point2d& pt, double depth, Point3d &pt3D);
bool SelectRefCam_InitPatchFixedScale(Point3d *expansionVec, double &scale3D, Point3d p3D, vector<KeyPoint> pts, vector<CameraData> AllViewsInfo, double PATCH_3D_ARROW_SIZE_WORLD_UNIT);

void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, Point2d *pts, Point3d *ThreeD, int npts, int distortionCorrected, double thresh, int &ninliers);
void DetermineDevicePose(double *K, double *distortion, int LensModel, double *R, double *T, vector<Point2d> pts, vector<Point3d> ThreeD, int distortionCorrected, double thresh, int &ninliers, bool directMethod = false);

template<class Vector3> void best_plane_from_points(const std::vector<Vector3> & c, vector<int> &selected, Vector3d &plane_normal, double *A, double *B)
{
	//ax+by+cz+1 = 0;
	for (int ii = 0; ii < c.size(); ii++)
		A[3 * ii] = c[ii](0), A[3 * ii + 1] = c[ii](1), A[3 * ii + 2] = c[ii](2), B[ii] = -1;
	QR_Solution_Double(A, B, c.size(), 3);
	plane_normal(0) = B[0], plane_normal(1) = B[1], plane_normal(2) = B[2];

	return;
}

double computeOverlappingMetric(CameraData &CamI, CameraData &CamJ, double angleDegThresh, double baselineThresh, double nearPlane, double farPlane, int nLayers, int nSamplesPerLayer);
#endif
