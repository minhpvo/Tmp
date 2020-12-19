#if !defined(FEATUREEST_H )
#define FEATUREEST_H
#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "../DataStructure.h"


#include "../ThirdParty/SiftGPU/src/SiftGPU/SiftGPU.h"
#include <vl/generic.h>
#include <vl/mathop.h>
#include <vl/sift.h>
#include <vl/covdet.h>

using namespace cv;
using namespace std;

void RootL1DescNorm(float *descf, uchar *descu, int numKeys, int featDim);
void RootL2DescNorm(float *descf, uchar *descu, int numKeys, int featDim);

//npts>0 specifiies intial points whose descriptors are to be computed
int vl_DoG(Mat &Img, vector<Point3f> &DoG, int verbose = 0);
int VLSIFT(Mat &Img, SiftFeature &SF, int &npts, int verbose = 0, bool RootL1 = true);
int VLSIFT(char *Fname, SiftFeature &SF, int &npts, int verbose = 0, bool RootL1 = true);
//npts>0 specifiies intial points whose descriptors are to be computed
int VLCOVDET(char *ImgName, CovFeature &CovF, int &npts, int verbose = 0, bool RootL1 = true);
int ComputeFeatureScaleAndDescriptor(Mat Img, KeyPoint &key, float *desc, int nOctaveLayers = 3, double sigma = 1.6, double contrastThreshold = 0.01, double edgeThreshold = 10);

int BuildImgPyr(char *ImgName, ImgPyr &Pyrad, int nOtaves, int nPerOctaves, bool color, int interpAlgo, double sigma = 1.0);
void BucketGoodFeaturesToTrack(Mat Img, vector<Point2f> &Corners, int nImagePartitions, int maxCorners, double qualityLevel, double minDistance, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

void Gaussian(double *G, int sigma, int PatternSize);
void LaplacianOfGaussian(double *LOG, int sigma);
void synthesize_concentric_circles_mask(double *ring_mask_smooth, int *pattern_bi_graylevel, int pattern_size, double sigma, double scale, double *ring_info, int flag, int num_ring_edge);
void DetectBlobCorrelation(char *ImgName, vector<KeyPoint> &kpts, int nOctaveLayers, int nScalePerOctave, double sigma, int templateSize, int NMS_BW, double thresh);

void DetectCheckerCornersCorrelation(double *img, int width, int height, int nchannels, Point2d *Checker, int &npts, vector<double> PatternAngles, int hsubset, int search_area, double thresh);
void RefineCheckerCorners(double *Para, int width, int height, int nchannels, Point2d *Checker, Point2d *Fcorners, int *FStype, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo);
void RefineCheckerCornersFromInit(double *Para, int width, int height, int nchannels, Point2d *Checker, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo);
void RunCheckerCornersDetector(Point2d *CornerPts, int *CornersType, int &nCpts, double *Img, double *IPara, int width, int height, int nchannels, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCThresh, int InterpAlgo);

int GetPoint2DPairCorrespondence(char *Path, int timeID, vector<int>viewID, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&CorrespondencesID, bool useGPU = true);
int DisplayImageCorrespondence(Mat &correspond, int offsetX, int offsetY, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<int>pair, double density);
int DisplayImageCorrespondence(Mat &correspond, int offsetX, int offsetY, vector<Point2d> keypoints1, vector<Point2d> keypoints2, vector<int>pair, double density);
int DisplayImageCorrespondencesDriver(char *Path, vector<int>viewsID, int timeID, int nchannels, double density = 0.25);

#endif
