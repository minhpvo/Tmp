#if !defined(MATCHINGTRACKING_H )
#define MATCHINGTRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "../Geometry/Geometry1.h"
#include "../DataStructure.h"


using namespace cv;
using namespace std;

vector<Point2i> MatchTwoViewSIFTBruteForce(Mat &descriptors1, Mat &descriptors2, double max_ratio = 0.7, double max_distance = 0.7, bool cross_check = true, int nthreads = 1);
vector<Point2i> MatchTwoViewSIFTBruteForce(vector<uchar> &descriptors1, vector<uchar> &descriptors2, int descDim, double max_ratio = 0.7, double max_distance = 0.7, bool cross_check = true, int nthreads = 1);

void cvFlowtoFloat(Mat_<Point2f> &flow, float *fx, float *fy);
void cvFloattoFlow(Mat_<Point2f> &flow, float *fx, float *fy);
void WarpImageFlow(float *flow, unsigned char *wImg21, unsigned char *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic);
void WarpImageFlowDouble(float *flow, double *wImg21, double *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic);

#define HOMO_VECTOR(H, x, y)\
    H.at<float>(0,0) = (float)(x);\
    H.at<float>(1,0) = (float)(y);\
    H.at<float>(2,0) = 1.;

#define GET_HOMO_VALUES(X, x, y)\
    (x) = static_cast<float> (X.at<float>(0,0)/X.at<float>(2,0));\
    (y) = static_cast<float> (X.at<float>(1,0)/X.at<float>(2,0));

void DetectBlobCorrelation(double *img, int width, int height, Point2d *Checker, int &npts, double sigma, int search_area, int NMS_BW, double thresh);
void DetectBlobCorrelation(char *ImgName, vector<KeyPoint> &kpts, int nOctaveLayers, int nScalePerOctave, double sigma, int templateSize, int NMS_BW, double thresh);

double Compute_AffineHomo(vector<Point2d> &From, vector<Point2d> To, double *Affine, double *A = 0, double *B = 0);
double Compute_AffineHomo(Point2d *From, Point2d *To, int npts, double *Affine, Point2d *sFrom = 0, Point2d *sTo = 0, double *A = 0, double *B = 0);
double findTransformECC(InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix, int motionType, TermCriteria criteria);
double findTransformECC_Optimized(Mat &templateFloat, Mat &imageFloat, Mat &gradientX, Mat &gradientY, Mat &gradientXWarped, Mat &gradientYWarped, Mat &warpMatrix, int motionType, TermCriteria criteria);
double TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, int nchannels, Point2i &POI, int search_area, double thresh, double *T = NULL);
double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd = 0);

//Image Correlation
double ComputeZNCCPatch(double *RefPatch, double *TarPatch, int hsubset, int nchannels = 1, double *T = NULL);
double ComputeZNCCImagePatch(Mat &Ref, Mat &Tar, Point2i RefPt, Point2i TarPt, int hsubset, int nchannels = 1, double *T = 0);
double ComputeSSIG(double *Para, int x, int y, int hsubset, int width, int height, int nchannels, int InterpAlgo);

double TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, int nchannels, Point2i &POI, int search_area, double thresh, double *T);
int TMatchingCoarse(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int search_area, double thresh, double &zncc, int InterpAlgo, double *InitPara = NULL, double *maxZNCC = NULL);
double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd);
double TemplateMatching0(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch = 0, double *ShapePara = 0, double *oPara = 0, double *Timg = 0, double *T = 0, double *ZNCC_reqd = 0);
double TemplateMatching0(float *RefPara, float *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch = 0, double *ShapePara = 0, double *oPara = 0, double *Timg = 0, double *T = 0, double *ZNCC_reqd = 0);
double TemplateMatching(double *RefPara, double *TarPara, int refWidth, int refHeight, int tarWidth, int tarHeight, int nchannels, Point2d From, Point2d &Target, LKParameters LKArg, bool greedySearch, double *Timg = 0, double *CorrelBuf = 0, double *iWp = 0, double *direction = 0);
double TemplateMatching(float *RefPara, float *TarPara, int refWidth, int refHeight, int tarWidth, int tarHeight, int nchannels, Point2d From, Point2d &Target, LKParameters LKArg, bool greedySearch, double *Timg = 0, double *CorrelBuf = 0, double *iWp = 0, double *direction = 0);

int Track_1P_1F_WithRefTemplate(vector<Point2f> &TrackUV, vector<AffinePara> &warp, vector<AffinePara> &iwarp, vector<Mat> *ImgPyr, double *RefPara, double *CPara, int reffid, int fid, int pid, int MaxWinSize, int nWinSize, int WinStep, int cvPyrLevel, LKParameters LKArg, double *T1 = 0, double *T2 = 0);
int TrackAllPointsWithRefTemplate(char *Path, int viewID, int startF, vector<Point2f> uvRef, vector<float> sRef, vector<Point2f> *ForeTrackUV, vector<Point2f> *BackTrackUV, vector<float> *ForeScale, vector<float> *BackScale, vector<AffinePara> *cForeWarp, vector<AffinePara> *cBackWarp, vector<FeatureDesc> *ForeDesc, vector<FeatureDesc> *BackDesc, vector<Mat> *ForePyr, vector<Mat> *BackPyr, int MaxWinSize, int nWinSize, int WinStep, int cvPyrLevel, int fps, int ForeTrackRange, int BackTrackRange, int interpAlgo, bool debug = false);

int Track_1P_1F_WithRefTemplate_DenseFlowDriven(vector<Point2f> &TrackUV, vector<AffinePara> &warp, vector<AffinePara> &iwarp, vector<Mat> *ImgPyr, double *RefPara, double *CPara, float* DFx, float*DFy, int reffid, int fid, int pid, int MaxWinSize, int nWinSize, int WinStep, LKParameters LKArg, double *T1, double *T2);
int TrackAllPointsWithRefTemplate_DenseFlowDriven(char *Path, int viewID, int startF, vector<Point2f> uvRef, vector<float> sRef, vector<Point2f> *ForeTrackUV, vector<Point2f> *BackTrackUV,
	vector<float> *ForeScale, vector<float> *BackScale, vector<AffinePara> *cForeWarp, vector<AffinePara> *cBackWarp, vector<FeatureDesc> *ForeDesc, vector<FeatureDesc> *BackDesc,
	vector<Mat> *ForePyr, vector<Mat> *BackPyr, vector<float*>DFx, vector<float*>DFy, vector<float*>DBx, vector<float*>DBy, int MaxWinSize, int nWinSize, int WinStep, double fps, int ForeTrackRange, int BackTrackRange, int noTemplateUpdate, int interpAlgo);

int DenseGreedyMatching(char *Img1, char *Img2, Point2d *displacement, vector<Point2d> &SSrcPts, vector<Point2d> &SDstPts, bool *lpROI_calculated, bool *tROI, LKParameters LKArg,
	int nchannels, int width1, int height1, int width2, int height2, double Scale, float *WarpingParas, int foundPrecomputedPoints = 0, double *Epipole = NULL, double *Pmat = NULL, double *K = NULL, double *distortion = NULL, double triThresh = 3.0);
int SemiDenseGreedyMatching(char *Img1, char *Img2, vector<Point2d> &SDSrcPts, vector<Point2d> &SDDstPts, vector<Point2d> &SSrcPts, vector<Point2d> &SDstPts, bool *lpROI_calculated, bool *tROI, LKParameters LKArg,
	int nchannels, int width1, int height1, int width2, int height2, double Scale, float *WarpingParas, int foundPrecomputedPoints = 0, double *Epipole = NULL, double *Pmat = NULL, double *K = NULL, double *distortion = NULL, double triThresh = 3.0);
#endif