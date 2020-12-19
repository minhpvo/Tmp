#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vl/generic.h>
#include <vl/mathop.h>
#include <vl/sift.h>
#include <vl/covdet.h>
#include "../GL/glew.h"
#ifdef _WINDOWS
#include "ThirdParty/GL/freeglut.h"
#else
#include <GL/glut.h>
#endif
using namespace cv;
using namespace std;


#if !defined( DATASTRUCTURE_H )
#define DATASTRUCTURE_H

#ifndef M_PI
#define M_PI 3.1415926535897932
#endif

#define MaxNFeatures 100000
#define SIFTBINS 128
#define FISHEYE  -1
#define RADIAL_TANGENTIAL_PRISM  0
#define VisSFMLens 1
#define GLOBAL_SHUTTER 0
#define ROLLING_SHUTTER 1
#define CONTINOUS_SHUTTER 2
#define LIMIT3D 1e-16
#define Pi 3.1415926535897932
#define MaxnFrames 2000
#define MaxnCams 10000
#define MaxnTrajectories 100000
#define MaxSharedIntrinsicGroup 100 //100 video cameras
#define OPTICALFLOW_BIDIRECT_DIST_THRESH 3.0

#define MOTION_TRANSLATION 0
#define MOTION_EUCLIDEAN 1
#define MOTION_AFFINE 2
#define MOTION_HOMOGRAPHY 3

extern bool newLog;
extern char LOG_FILE_NAME[512];

struct Frustrum
{
	Frustrum(Eigen::Vector3d tln_, Eigen::Vector3d tlf_, Eigen::Vector3d trn_, Eigen::Vector3d trf_, Eigen::Vector3d brn_, Eigen::Vector3d brf_, Eigen::Vector3d bln_, Eigen::Vector3d blf_)
		:tln(tln_), tlf(tlf_), trn(trn_), trf(trf_), brn(brn_), brf(brf_), bln(bln_), blf(blf_) {
		computeNormal();
	}

	void computeNormal()
	{
		normNearPlane = (brn - bln).cross(tln - brn);
		normRightPlane = (brf - brn).cross(trn - brn);
		normFarPlane = (blf - brf).cross(trf - brf);
		normLeftPlane = (bln - blf).cross(tlf - blf);
		normTopPlane = (trf - trn).cross(tln - trn);
		normBottomPlane = (blf - bln).cross(brn - bln);
	}

	Eigen::Vector3d tln, tlf, trn, trf, brn, brf, bln, blf, pt;

	//all normals are pointing outward
	Eigen::Vector3d normNearPlane, normRightPlane, normFarPlane, normLeftPlane, normTopPlane, normBottomPlane;
};


struct SfMPara {
	SfMPara()
	{
		SkeletonPointFormat = 25, SMPLWindow = 80, SMPLnOverlappingFrames = 20, missingFrameInterp = 0;
		ShapeWeight = 1.0, PoseWeight = 0.1, TemporalWeight = 0.2, ContourWeight = 1.0, SilWeight = 10.0, KeyPointsWeight = 10.0, DensePoseWeight = 1.0, HeadMountedCameraWeight = 0, FootClampingWeight = 0;

		real2SfM = 1.0;

		SyncedMode = true;
		extractedFrames = 0, nCams = -1, startF = -1, stopF = -1, increF = 1, UseJpg = 1, imgRescale = 1.0, interpAlgo = -1;
		ExternalCorpus = -1, IncreMatchingFrame = -1, fromKeyFrameTracking = 1, KFSample4Corpus = 1;

		highQualityTracking = true;
		minKFinterval = 100, maxKFinterval = 600, minFeaturesToTrack = 400;
		kfFlowThresh = 200.0, kfSuccessConsecutiveTrackingRatio = 0.7, kfSuccessRefTrackingRatio = 0.4;

		ba_global_images_ratio = 1.2, ba_global_points_ratio = 1.1, ba_global_tri_min_angle = 3.0;
		snapshot_images_freq = 300;

		MatchingMode = 2, //0: exhaustive, 1: sequential with loop closure, 2: 1 with learned vocab, 3: vocab, 4: 2 with learned vocab
			VocMatchingkNN = 400, //# similar images returned by vocabtree matching
			SeqMatchingForcedNearbyImageRange = 80, // # close by images to match
			kClosestKeyFrames = 4; //match nonkeyframes with keyframes in sequentail with loop clousure

		ec2BatchSize = 10, PnPMatchingForcedNearbyKeyFrameRange = 10;

		useRanSac = true, sharedIntrinsic = false;
		maxGlobalPass = 3, maxPassesPerImage = 5, globalBA_freq = 100;
		nInliersThresh = 30, nInliersThresh2 = 100, nViewsPlusBA = 3, LossType = 1,//0: L2, 1: Huber
			reProjectionTrianguatlionThresh = 8.0, reProjectionBAThresh = 8.0, tri_local_min_tri_angle = 2.0, underTriangulationRatio = 0.25, Init_min_tri_angle = 5.0, InitForwadMotionRatioThresh = 2.0;
		BARefinementIter = 2;
		fixIntrinsic = 1, fixDistortion = 1, fixSkew = 1, fixPrism = 1, fix3D = 1, fixLocal3D = 0, distortionCorrected = 0, //0: no correction, 1: correct from accurate intrinsics
			LensModel = RADIAL_TANGENTIAL_PRISM, ShutterModel = GLOBAL_SHUTTER, ShutterModel2 = ROLLING_SHUTTER;
		minFRatio = 0.1, maxFratio = 10.0;
	}

	bool SyncedMode;
	int extractedFrames, nCams, startF, stopF, increF, UseJpg, interpAlgo, missingFrameInterp;
	double imgRescale, real2SfM;

	//SMPLFitting
	int SkeletonPointFormat, SMPLWindow, SMPLnOverlappingFrames;
	double ShapeWeight, PoseWeight, TemporalWeight, ContourWeight, SilWeight, KeyPointsWeight, DensePoseWeight, HeadMountedCameraWeight, FootClampingWeight;

	//Corpus nature
	int ExternalCorpus, //(>-1) have a dedicated video for corpus vs. using frames from all videos for corpus
		IncreMatchingFrame,//legacy mode, try to compute pose to corpus every n-frames. If increF = 2, incrematchingframe = 5, the real distance between 2 sampled video frame is 10 frames
		fromKeyFrameTracking, KFSample4Corpus;

	//Key frames extraction
	bool highQualityTracking;
	int minKFinterval, maxKFinterval, minFeaturesToTrack, cvPyrLevel, trackingWinSize, nWinStep;
	double kfFlowThresh, kfSuccessConsecutiveTrackingRatio, kfSuccessRefTrackingRatio, meanSSGThresh;

	//For Colmap
	double ba_global_images_ratio, ba_global_points_ratio;

	//Vocal matching
	char VocabTreePath[512];
	int MatchingMode, //0: exhaustive, 1: sequential with loop closure, 2: 1 with learned vocab, 3: vocab, 4: 2 with learned vocab
		VocMatchingkNN, //# similar images returned by vocabtree matching
		SeqMatchingForcedNearbyImageRange, // # close by images to match
		kClosestKeyFrames; //match nonkeyframes with keyframes in sequentail with loop clousure

	//For Pnp to Corpus
	int ec2BatchSize, PnPMatchingForcedNearbyKeyFrameRange; //use nn keyframes around the current frame to find corres for PnP

	//For BA and triangulation
	bool useRanSac, sharedIntrinsic;
	int nInliersThresh, nInliersThresh2, maxGlobalPass, maxPassesPerImage, nViewsPlusBA, nViewsPlusBA2, globalBA_freq, snapshot_images_freq;
	int fixIntrinsic, fixDistortion, fixSkew, fixPrism, fixPose, fix3D, fixLocal3D, distortionCorrected, LensModel, ShutterModel, ShutterModel2, BARefinementIter, LossType;
	double reProjectionTrianguatlionThresh, reProjectionBAThresh, ba_global_tri_min_angle, tri_local_min_tri_angle, underTriangulationRatio, globalBA_PointsRatio, minFRatio, maxFratio, Init_min_tri_angle, InitForwadMotionRatioThresh;

};
struct TVL1Parameters
{
	bool useInitialFlow;
	int iterations, nscales, warps;
	double tau, lamda, theta, epsilon;
};
struct LKParameters
{
	LKParameters() {}

	LKParameters(int hsubset, int nscales, int scaleStep, int DIC_Algo, int InterpAlgo, double Gsigma, int Convergence_Criteria, int  IterMax, int Analysis_Speed, double ZNCCThreshold, double PSSDab_thresh, int DisplacementThresh) :
		hsubset(hsubset), nscales(nscales), scaleStep(scaleStep), DIC_Algo(DIC_Algo), InterpAlgo(InterpAlgo), Gsigma(Gsigma), Convergence_Criteria(Convergence_Criteria), IterMax(IterMax), Analysis_Speed(Analysis_Speed), ZNCCThreshold(ZNCCThreshold), PSSDab_thresh(PSSDab_thresh), DisplacementThresh(DisplacementThresh) {}

	//DIC_Algo: 
	//0 epipolar search with translation model
	//1 Affine model with epipolar constraint
	//2 translation model
	//3 Affine model without epipolar constraint
	bool checkZNCC;
	int step, nscales, scaleStep, hsubset, npass, npass2, searchRangeScale, searchRangeScale2, searchRange, Incomplete_Subset_Handling, Convergence_Criteria, Analysis_Speed, IterMax, InterpAlgo, DIC_Algo, EpipEnforce;
	double DisplacementThresh, ZNCCThreshold, PSSDab_thresh, ssigThresh, Gsigma, ProjectorGsigma;
};

typedef Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FeatureDescriptor;

enum PnP
{
	P3P = 0, EPNP = 1, DLS = 2, P4Pf = 3
};

struct BodyImgAtrribute
{
	Point3d pt[25];
	float desc[1]; //not being used for now
	int personLocalId, nPts, nDims;
};

struct HumanSkeleton3D
{
	HumanSkeleton3D()
	{
		valid = 0;
	}

	int nvis, valid, refFid, nPts;
	int validJoints[25];
	Point3d pt3d[25];	
	vector<Point2i> vViewID_rFid[25];
	vector<Point3i> vCidFidDid;
	vector<Point2d> vPt2D[25], vPt2D_[25];
	vector<double> vConf[25];
	vector<int> vInlier[25];

	//smpl rigid alignment
	double s, //from smpl model to sfm scale
		r[3], t[3];
};
struct sHumanSkeleton3D
{
	sHumanSkeleton3D()
	{
		startActiveF = 0, nBAtimes = 0;
		active = 0, lastBAFrame = 0, nframeSinceLastBA = 1;
		for(int ii=0; ii<17; ii++)
			meanBoneLength[ii] = 0.0;
	}
	double meanBoneLength[17];
	vector< HumanSkeleton3D> sSke;
	int active;
	int startActiveF, nBAtimes, lastBAFrame, nframeSinceLastBA;
};
struct SiftFeature
{
	SiftFeature()
	{
		CurrentMaxFeatures = 0, Kpts = 0, Desc = 0;
	}
	int CurrentMaxFeatures;
	double* Kpts;
	uchar* Desc;
};
struct CovFeature
{
	CovFeature()
	{
		Affine = 0, Orientation = 1, CurrentMaxFeatures = 0;

		method = VL_COVDET_METHOD_HESSIAN_LAPLACE;
		doubleImage = 1;
		octaveResolution = 3, patchResolution = 15;
		edgeThreshold = 10, peakThreshold = 2e-6, lapPeakThreshold = 0.01;
		patchRelativeExtent = 7.5, patchRelativeSmoothing = 1.0, boundaryMargin = 2.0;

		Kpts = 0;
		Desc = 0;

	}
	int Affine, Orientation, CurrentMaxFeatures;

	VlCovDetMethod method;
	vl_bool doubleImage;
	vl_index octaveResolution, patchResolution;
	double edgeThreshold, peakThreshold, lapPeakThreshold;
	double patchRelativeExtent, patchRelativeSmoothing, boundaryMargin;

	double* Kpts;
	uchar* Desc;
};

struct FeatureDesc
{
	uchar desc[128];
};
struct ImgPyr
{
	bool rgb;
	int nscales;
	vector<double *> ImgPyrPara;
	vector<unsigned char *> ImgPyrImg;
	vector<Point2i> wh;
	vector<double> factor;
};
struct AffinePara
{
	AffinePara() { for (int ii = 0; ii < 4; ii++) warp[ii] = 0.0; }
	AffinePara(double *w) { for (int ii = 0; ii < 4; ii++) warp[ii] = w[ii]; }
	double warp[4];
};

struct CameraData
{
	CameraData()
	{
		viewID = -1;
		frameID = -1;
		valid = false, processed = false;
		for (int ii = 0; ii < 6; ii++)
			wt[ii] = 0.0;
		for (int ii = 0; ii < 9; ii++)
			R[ii] = 0.0, K[ii] = 0.0;
		for (int ii = 0; ii < 3; ii++)
			T[ii] = 0;
		for (int ii = 0; ii < 6; ii++)
			rt[ii] = 0.0;
		for (int ii = 0; ii < 5; ii++)
			intrinsic[ii] = 0.0;
		for (int ii = 0; ii < 7; ii++)
			distortion[ii] = 0.0;
		fps = 10.0, rollingShutterPercent = 0.0;//global shutter
		hasIntrinsicExtrinisc = 0;
		camCenter[0] = 0, camCenter[1] = 0, camCenter[2] = 0;
		Rgl[0] = 1, Rgl[1] = 0, Rgl[2] = 0, Rgl[3] = 0;
		Rgl[4] = 0, Rgl[5] = 1, Rgl[6] = 0, Rgl[7] = 0;
		Rgl[8] = 0, Rgl[9] = 0, Rgl[10] = 1, Rgl[11] = 0;
		Rgl[12] = 0, Rgl[13] = 0, Rgl[14] = 0, Rgl[15] = 1;
		minFratio = 0.1, maxFratio = 10.0;
		nInlierThresh = 30;
		nTouches = 0, nTouches2Views = 0;
	}

	double K[9], distortion[7], R[9], Quat[4], T[3], rt[6], wt[6], P[12], intrinsic[5], invK[9], invR[9], principleRayDir[3];
	double Rgl[16], camCenter[3];
	int LensModel;
	int ShutterModel; //0: Global, 1: Cayley, 2: Spline
	double threshold, nInlierThresh, minFratio, maxFratio;
	double fps, rollingShutterPercent, TimeOffset;
	std::string filename;
	int nviews, viewID, frameID, width, height, hasIntrinsicExtrinisc, nTouches, nTouches2Views;
	bool notCalibrated, valid, processed;
};
struct Corpus
{
	int nCameras, n3dPoints;
	vector<int> IDCumView;
	vector<int> vTrueFrameId;
	vector<string> filenames;
	CameraData *camera;

	vector<int> GlobalAnchor3DId;
	vector<Point3d>  xyz;
	vector<Point3i >  rgb;

	//needed for BA
	vector <vector<int> > viewIdAll3D; //3D -> visiable views index
	vector <vector<int> > pointIdAll3D; //3D -> 2D index in those visible views
	vector<vector<Point2d> > uvAll3D; //3D -> uv of that point in those visible views
	vector<vector<double> > scaleAll3D; //3D -> scale of that point in those visible views
	vector<Mat>DescAll3D; //desc for all 3d

	//needed for matching and triangulation
	vector<int> *SiftIdAllViews;//all views valid sift feature org id
	vector<Point2d> *uvAllViews; //all views valid 2D points
	vector<double> *scaleAllViews; //all views valid 2D points
	vector<int> *twoDIdAllViews;  //2D point in visible view -> 2D index
	vector<int> *threeDIdAllViews; //3D point in visible view -> 3D index
	vector<FeatureDesc> *DescAllViews;//all views valid desc

	vector<int> n2DPointsPerView, cn3DPointsPerView;
	vector<Point2f*> uvAllViews2;
	vector<float*> scaleAllViews2;
	vector<int*> threeDIdAllViews2;
	vector<bool*> InlierAllViews2;  //2D point in visible view -> inlier (1) or not (0)

	//pairwise matching info
	vector<int> nMatchesCidIPidICidJPidJ;
	vector<Point2i> matchedCidICidJ;
	vector<Point2i*> matchedCidIPidICidJPidJ;
};
struct CorpusandVideo
{
	int nViewsCorpus, nVideos, startTime, stopTime, CorpusSharedIntrinsics;
	CameraData *CorpusInfo;
	CameraData *VideoInfo;
};
struct VideoData
{
	VideoData()
	{
		staticCam = false;
		maxFrameOffset = 0;
	}
	bool staticCam;
	double fps, rollingShutterPercent, TimeOffset;
	int nVideos, startTime, stopTime, nframesI, maxFrameOffset;
	CameraData *VideoInfo;
};

struct MultiViewPtEle
{
	vector<int> viewID, frameID;
	vector<Point2d> pt2D;
	Point3d pt3D;
};
struct ImgFacioEle
{
	ImgFacioEle() {
		assigned = 0, viewID = -1, frameID = -1;
	}

	int assigned, viewID, frameID;
	double P[12];
	Point2d pt2D[49];
};
struct ImgPoseEle
{
	ImgPoseEle(int skeletonPointFormat_) {
		skeletonPointFormat = skeletonPointFormat_;
		assigned = -1, viewID = -1, frameID = -1;
		P = new double[25 * 12];
		for (int ii = 0; ii < 25 * 12; ii++)
			P[ii] = 1.0; //0 would lead to nan during autodif
	}
	ImgPoseEle() {
		skeletonPointFormat = 25;
		assigned = -1, viewID = -1, frameID = -1, refFrameID= -1;
		P = new double[25 * 12];
		for (int ii = 0; ii < 25 * 12; ii++)
			P[ii] = 1.0; //0 would lead to nan during autodif
	}

	int assigned, viewID, frameID, refFrameID, skeletonPointFormat;
	double *P, confidence[25], ts; //ts: time stamp in s
	Point2d pt2D[25];
	Point3d pt3D[25];
};
struct BodyPoseInfo
{
	vector<ImgPoseEle*> PoseInfo;
	vector<int> nbodies;
};
struct ImgPtEle
{
	ImgPtEle() {
		valid = 0, viewID = -1, frameID = -1, pixelSizeToMm = 2.0e-3, std2D = 1.0, std3D = -1, scale = 1.0, canonicalScale = 1.0, rollingShutterPercent = 0.2, fps = 10.0;
	}

	int viewID, frameID, imWidth, imHeight, LensModel, shutterModel, valid;
	Point2d pt2D, pt2DErr;
	Point3d pt3D;
	AffinePara wp;
	double pt2d[2], pt3d[3];
	double ray[3], camcenter[3], d, idepth, timeStamp, angle, scale, canonicalScale, std2D, std3D, pixelSizeToMm, rollingShutterPercent, fps;
	double K[9], iK[9], distortion[7], R[9], Quat[4], T[3], P[12], Q[6], u[2];
};

struct XYZD
{
	Point3d xyz;
	double d;
};
struct Pmat
{
	double P[12];
};
struct KMatrix
{
	double K[9];
};
struct CamCenter
{
	double C[3];
};
struct RotMatrix
{
	double R[9];
};
struct Quaternion
{
	double quad[4];
};
struct Track2D
{
	int *frameID;
	Point2d *uv;
	double *ParaX, *ParaY;

	int nf;
};
struct Track3D
{
	double *xyz;
	int *frameID;
	int nf;
};
struct Track4D
{
	double *xyzt;
	int npts;
};

struct PerCamNonRigidTrajectory
{
	vector<Pmat> P;
	vector<KMatrix> K;
	vector<RotMatrix >R;
	vector<Quaternion> Q;
	vector<CamCenter>C;

	Track3D *Track3DInfo;
	Track2D *Track2DInfo;
	Track3D *camcenter;
	Track4D *quaternion;
	double F;
	int npts;
};
struct Trajectory2D
{
	int timeID, nViews;
	vector<int>viewIDs, frameID;
	vector<Point2d> uv;
	vector<float>angle;
};
struct Trajectory3D
{
	double timeID;
	int frameID, viewID;
	vector<int>viewIDs;
	vector<Point2d> uv;
	Point3d WC, STD;
	Point3f rgb;
};
struct TrajectoryData
{
	vector<Point3d> *cpThreeD;
	vector<Point3d> *fThreeD;
	vector<Point3d> *cpNormal;
	vector<Point3d> *fNormal;
	vector<Point2d> *cpTwoD;
	vector<Point2d> *fTwoD;
	vector<vector<int> > *cpVis;
	vector<vector<int> > *fVis;
	int nTrajectories, nViews;
	vector<Trajectory2D> *trajectoryUnit;
};
struct VisualizationManager
{
	int catPointCurrentTime;
	Point3d g_trajactoryCenter;
	vector<Point3f> CorpusPointPosition, CorpusPointPosition2, PointPosition, PointPosition2, PointPosition3;
	vector<Point3f> CorpusPointColor, CorpusPointColor2, PointColor, PointColor2, PointColor3;
	vector<Point3f>PointNormal, PointNormal2, PointNormal3;
	vector<CameraData> glCorpusCameraInfo, *glCameraPoseInfo, *glCameraPoseInfo2;
	vector<Point3f> *catPointPosition, *catPointPosition2;
	vector<Trajectory3D* > Traject3D;
	vector<int> Track3DLength;
	vector<Trajectory3D* > Traject3D2;
	vector<int> Track3DLength2;
	vector<Trajectory3D* > Traject3D3;
	vector<int> Track3DLength3;
	vector<Trajectory3D* > Traject3D4;
	vector<int> Track3DLength4;
	vector<Trajectory3D* > Traject3D5;
	vector<int> Track3DLength5;
};

struct SurfDesc
{
	float desc[64];
};
struct SiftDesc
{
	float desc[128];
};


#endif 
