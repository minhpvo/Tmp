#if !defined(GENERALULTI_H )
#define GENERALULTI_H
#pragma once

#ifdef _WINDOWS
#include <direct.h>
#else
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>
#include "DataIO.h"
#include "../TrackMatch/FeatureEst.h"

void mySleep(int ms);
void myGetCurDir(int size, char *Path);

//For Minh incremnetal Sfm, which is not frequently used and not fully functional
void BestPairFinder(char *Path, int nviews, int timeID, int *viewPair);
int NextViewFinder(char *Path, int nviews, int timeID, int currentView, int &maxPoints, vector<int> usedPairs);
int GetPoint3D2DPairCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, vector<int> viewID, Point3d *ThreeD, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&TwoDCorrespondencesID, vector<int> &ThreeDCorrespondencesID, vector<int>&SelectedIndex, bool SwapView, bool useGPU = true);
int GetPoint3D2DAllCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, vector<int> AvailViews, vector<int>&Selected3DIndex, vector<Point2d> *selected2D, vector<int>*nselectedviews, int &nselectedPts, bool useGPU = true);

int ComputePnPInlierStats(char *Path, int nCams, int startF, int stopF);

//Read correspondences and build matching (visbiblity) matrix for further sfm
int ReadCumulativePoints(char *Path, int nviews, int timeID, vector<int>&cumulativePts);
void ReadCumulativePointsVisualSfm(char *Path, int nviews, vector<int>&cumulativePts);

void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, vector<int>&mask, int totalPts, bool Merge = false);
void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, int totalPts, bool Merge);
void GenerateMergePointCorrespondences(vector<int> *MergePointCorres, vector<int> *PointCorres, int totalPts);
void GenerateViewandPointCorrespondences(vector<int> *ViewCorres, vector<int> *PointIDCorres, vector<int> *PointCorres, vector<int> CumIDView, int totalPts);

void GenerateMatchingTable(char *Path, int nviews, int timeID, int NviewPlus = 3);
void GenerateMatchingTableVisualSfM(char *Path, int nviews);
void GenerateViewCorrespondenceMatrix(char *Path, int nviews, int timeID);
int GetPutativeMatchesForEachView(char *Path, int nviews, vector<int> TrackingInst, Point2d ScaleThresh, int nViewPlus, int *frameTimeStamp);

void RotY(double *Rmat, double theta);
void GenerateCamerasExtrinsicOnCircle(CameraData &CameraInfo, double theta, double radius, Point3d center, Point3d LookAtTarget, Point3d Noise3D);
double computeProcrustesTransform(vector<Point3d> & src, vector<Point3d>& dst, double *R, double *T, double &scale, bool includeScaling = false);

void InvalidateAbruptCameraPose(VideoData &VideoI, int startF, int stopF, int silent);

//post processing of 2d trajectory
int Track2DConverter(char *Path, int viewID, int startF);
int DelSel2DTrajectory(char *Path, int nCams);
int CleanUp2DTrackingByGradientConsistency(char *Path, int nviews, int refFrame, int increF, int TrackRange, int*frameTimeStamp, int DenseDriven = 0);
int RemoveLargelyDisconnectedPointsInTraj(char *Path, int nCams, int DiscoThresh = 3);
int Clean2DTrajStartEndAppearance(char *Path, int cid, int stopF, int TrackingInst = -1);
int CombineFlow_PLK_B(char *Path, int nviews, int refFrame, int TrackRange, int maxNpts = 10000);
int RemoveDuplicatedMatches(char *Path, int nCams, int startF);
int RemoveMatchesOfDifferentClass(char *Path, int nCams, int InstF, int *TimeStamp, vector<Point3i> &ClassColor, bool inverseColor = false);
int RemoveMatchesOfDifferentClass(vector<Point2f> &uv, Mat &img, vector<Point3i> &ClassColor, bool inverseColor = false);
int Reorder2DTrajectories2(char *Path, int cid, int instF);
int Reorder2DTrajectories(char *Path, int nviews, vector<int> &TrackingInst);
int ReAssembleUltimateTrajectories(char *Path, int nCams, vector<int>&TrackingInstances);
int RemoveShortTrajectory(char *Path, int nCams, int minTrajLength, int maxNpts);
int RemoveWeakOverlappingTrajectories(char *Path, int nCams, int startF, int stopF, int *TimeStamp, int nTplus, int nPplus);
int RemoveWeakOverlappingTrajectories2(char *Path, int nCams, int startF, int stopF, int *TimeStamp, int nTplus, int nPplus);
int RemoveTrajectoryOutsideROI(char *Path, int nCams, int startF, int stopF, int maxNpts = 10000);
int DeletePointsOf2DTracks(char *Path, int nCams, int npts);
int DownSampleTracking(char *Path, vector<int> &viewList, int startF, int stopF, int rate);
int DownSampleVideoCameraPose(char *Path, vector<int>&viewList, int startFrame, int stopFrame, int rate);
int DownSampleImageSequence(char *Path, vector<int> &sCams, int startF, int stopF, int rate);
cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor);

#endif