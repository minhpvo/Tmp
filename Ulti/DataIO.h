#if !defined(DATAIO_H )
#define DATAIO_H
#pragma once

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdarg.h>

#ifdef _WINDOWS
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#endif
#include <opencv2/features2d/features2d.hpp>

#include "../DataStructure.h"
#include "../Ulti/MathUlti.h"
#include "../Drivers/Drivers.h"
#include "../ThirdParty/SiftGPU/src/SiftGPU/SiftGPU.h"

using namespace cv;
using namespace std;


void printLOG( const char* format, ...);

void StringTrim(std::string* str);
void makeDir(char *Fname);
bool MyCopyFile(const char *SRC, const char* DEST);
int IsFileExist(const char *Fname, bool silient = true);

#ifdef _WINDOWS
vector<string> get_all_files_names_within_folder(string folder);
#endif

int readOpenPoseJson(char *Path, int cid, int fid, vector<Point2f> &vUV, vector<float> &vConf);

//Input: sift keys in gpu coord (since the user will be visualsfm, which is default to be gpu coord)
int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, int nsift);
int writeVisualSFMSiftGPU(const char* fn, vector<cv::KeyPoint> &keys, uchar *descriptors);
int writeVisualSFMSiftGPU(const char* fn, vector<SiftGPU::SiftKeypoint> &keys, uchar *descriptors);
int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, int *Order, int nsift);
int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, vector<bool> &mask);
//Output: Sift keys in cpu coord 
int readVisualSFMSiftGPU(const char *fn, vector<KeyPoint>&kpts, bool silent = true);
//Output: Sift keys in cpu coord 
int readVisualSFMSiftGPU(const char *fn, vector<KeyPoint>&kpts, vector<uchar> &descriptors, bool silent = true);
int readVisualSFMSiftGPU(const char *fn, vector<KeyPoint>&kpts, Mat &descriptors, bool silent = true);
//Output: Sift keys in cpu coord
int readVisualSFMSiftGPU(const char *fn, vector<SiftKeypoint>&kpts, Mat &descriptors, bool silent = true);
int readVisualSFMSiftGPU(const char *fn, vector<SiftKeypoint>&kpts, vector<uchar> &descriptors, bool silent = true);
int readVisualSFMSiftGPU(const char *fn, vector<SiftKeypoint>&kpts, bool silent = true);
int convertVisualSFMSiftGPU2KPointsDesc(char *Name_WO_Type);

bool WriteKPointsSIFT(char *fn, vector<SiftKeypoint>kpts, bool verbose);
bool WriteKPointsBinarySIFT(char *fn, vector<SiftGPU::SiftKeypoint>kpts, bool verbose = false);
bool WriteKPointsBinarySIFT(char *fn, vector<SiftKeypoint>kpts, vector<bool> &mask, int npts, bool verbose = false);
bool WriteKPointsBinarySIFT(char *fn, float *kpts, vector<bool> &mask, int npts, bool verbose = false);
bool ReadKPointsBinarySIFT(char *fn, vector<SiftGPU::SiftKeypoint> &kpts, bool verbose = false);
bool WriteKPointsBinarySIFT(char *fn, vector<KeyPoint>kpts, bool verbose = false);
bool ReadKPointsBinarySIFT(char *fn, vector<KeyPoint> &kpts, bool verbose = false);
bool WriteDescriptorBinarySIFT(char *fn, vector<uchar > descriptors, bool verbose = false);
bool WriteDescriptorBinarySIFT(char *fn, vector<uchar > descriptors, vector<bool>&mask, int npts, bool verbose = false);
bool WriteDescriptorBinarySIFT(char *fn, Mat descriptor, bool verbose = false);
bool WriteDescriptorBinarySIFT(char *fn, Mat descriptor, vector<bool>&mask, int npts, bool verbose = false);
bool ReadDescriptorBinarySIFT(char *fn, vector<float > &descriptors, bool verbose = false);
Mat ReadDescriptorBinarySIFT(char *fn, bool verbose = false);

bool WriteRGBBinarySIFT(char *fn, vector<Point3i> rgb, bool verbose = false);
bool ReadRGBBinarySIFT(char *fn, vector<Point3i> &rgb, bool verbose = false);
bool WriteKPointsRGBBinarySIFT(char *fn, vector<SiftKeypoint>kpts, vector<Point3i> rgb, bool verbose = false);
bool ReadKPointsRGBBinarySIFT(char *fn, vector<SiftKeypoint> &kpts, vector<Point3i> &rgb, bool verbose = false);
bool WriteKPointsRGBBinarySIFT(char *fn, vector<KeyPoint>kpts, vector<Point3i> rgb, bool verbose = false);
bool ReadKPointsRGBBinarySIFT(char *fn, vector<KeyPoint> &kpts, vector<Point3i> &rgb, bool verbose = false);

bool GrabImageCVFormat(char *fname, char *Img, int &width, int &height, int nchannels, bool verbose = false);
bool GrabImage(char *fname, char *Img, int &width, int &height, int nchannels, bool verbose = false);
bool GrabImage(char *fname, unsigned char *Img, int &width, int &height, int nchannels, bool verbose = false);
bool GrabImage(char *fname, float *Img, int &width, int &height, int nchannels, bool verbose = false);
bool GrabImage(char *fname, double *Img, int &width, int &height, int nchannels, bool verbose = false);

bool SaveDataToImageCVFormat(char *fname, char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImageCVFormat(char *fname, uchar *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, bool *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, unsigned char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, int *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, float *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels = 1);

int ExtractVideoFrames(char *Path, int camID, int startF, int stopF, int increF, int rotateImage, double resizeFactor = 1.0, int nchannels = 3, int Usejpg = 1, int frameTimeStamp = 99999);
int ExtractVideoFrames(char *Path, char *inName, int startF, int stopF, int increF, int rotateImage, int nchannels = 3, int Usejpg = 1);

bool GrabVideoFrame2Mem(char *fname, char *Data, int &width, int &height, int &nchannels, int &nframes, int frameSample = 1, int fixnframes = 99999999);

void ShowCVDataAsImage(char *Fname, char *Img, int width, int height, int nchannels, IplImage *cvImg = 0, int rotate90 = 0);
void ShowCVDataAsImage(char *Fname, uchar *Img, int width, int height, int nchannels, IplImage *cvImg = 0, int rotate90 = 0);
void ShowDataAsImage(char *fname, unsigned char *Img, int width, int height, int nchannels);
void ShowDataAsImage(char *fname, double *Img, int width, int height, int nchannels);

template <class myType> bool WriteGridBinary(char *fn, myType *data, int width, int height, bool  verbose = false)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	for (int j = 0; j < height; ++j)
		for (int i = 0; i < width; ++i)
			fout.write(reinterpret_cast<char *>(&data[i + j*width]), sizeof(myType));
	fout.close();

	return true;
}
template <class myType> bool ReadGridBinary(char *fn, myType *data, int width, int height, bool  verbose = false)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	for (int j = 0; j < height; ++j)
		for (int i = 0; i < width; ++i)
			fin.read(reinterpret_cast<char *>(&data[i + j*width]), sizeof(myType));
	fin.close();

	return true;
}
template <class myType> bool WriteFlowBinary(char *fnX, char *fnY, myType *fx, myType *fy, int width, int height)
{
	myType u, v;

	ofstream fout1, fout2;
	fout1.open(fnX, ios::binary);
	if (!fout1.is_open())
	{
		cout << "Cannot load: " << fnX << endl;
		return false;
	}
	fout2.open(fnY, ios::binary);
	if (!fout2.is_open())
	{
		cout << "Cannot load: " << fnY << endl;
		return false;
	}

	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			u = fx[i + j*width];
			v = fy[i + j*width];

			fout1.write(reinterpret_cast<char *>(&u), sizeof(myType));
			fout2.write(reinterpret_cast<char *>(&v), sizeof(myType));
		}
	}
	fout1.close();
	fout2.close();

	return true;
}
template <class myType> bool ReadFlowBinary(char *fnX, char *fnY, myType *fx, myType *fy, int width, int height)
{
	myType u, v;

	ifstream fin1, fin2;
	fin1.open(fnX, ios::binary);
	if (!fin1.is_open())
	{
		cout << "Cannot load: " << fnX << endl;
		return false;
	}
	fin2.open(fnY, ios::binary);
	if (!fin2.is_open())
	{
		cout << "Cannot load: " << fnY << endl;
		return false;
	}

	for (int j = 0; j < height; ++j)
		for (int i = 0; i < width; ++i)
		{
			fin1.read(reinterpret_cast<char *>(&u), sizeof(myType));
			fin2.read(reinterpret_cast<char *>(&v), sizeof(myType));

			fx[i + j*width] = u;
			fy[i + j*width] = v;
		}
	fin1.close();
	fin2.close();

	return true;
}

int GenerateVisualSFMinput(char *path, int startFrame, int stopFrame, int npts);
bool readNVMLite(const char *filepath, Corpus &CorpusData, int sharedIntrinsics, int nHDs = 30, int nVGAs = 24, int nPanels = 20);
bool readNVM(const char *filepath, Corpus &CorpusData, vector<Point2i> &ImgSize, int nplus = 0, vector<KeyPoint> *AllKeyPts = 0, Mat *AllDesc = 0); //Also convert sift keys in gpu coord to cpu
bool readColMap(char *path, Corpus &CorpusInfo, int nViewPlus, int nImages = -1);
bool readColMap(char *path, Corpus &CorpusInfo, int nViewPlus, vector<KeyPoint> *AllKeyPts, Mat *AllDesc, int nImages = -1);
bool writeColMap4DenseStereo(char *Path, Corpus &CorpusInfo, int nNeighbors = 20, int frameId = -1);
bool writeDeepMVSInputData(char *Path, vector<int> &sCams, int startF, int stopF, int increF, double nearDepth, double farDepth, int nDepthLayers, double sfm2Real = 1.0);

bool readBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData);
bool saveBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData);
bool ReSaveBundleAdjustedNVMResults(char *BAfileName, double ScaleFactor = 1.0);
bool ReSaveBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusData, double ScaleFactor = 1.0);

int WritePCL_PCD(char *Fname, vector<Point3f> &Vxyz);
int ReadPCL_PLY(char *Fname, vector<Point3f> &Vxyz, vector<Point3i> &Vtri);

void Save3DPoints(char *Path, Point3d *All3D, vector<int>Selected3DIndex);
int SaveCorpusInfo(char *Path, Corpus &CorpusData, bool outputtext = false, bool saveDescriptor = true);
int ReadCorpusInfo(char *Path, Corpus &CorpusData, bool inputtext = false, bool notReadDescriptor = false, vector<int> CorpusImagesToRead = vector<int>());
bool readIndividualNVMforpose(char *Path, CameraData *CameraInfo, vector<int>availViews, int timeIDstart, int timeIDstop, int nviews, bool sharedIntrinsics);
int ReadCorpusAndVideoData(char *Path, CorpusandVideo &CorpusandVideoInfo, int ScannedCopursCam, int nVideoViews, int startTime, int stopTime, int LensModel = RADIAL_TANGENTIAL_PRISM, int distortionCorrected = 1);
int ReadVideoData(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime, double threshold = 5.0, int ninliersThresh = 40);
int ReadVideoDataI(char *Path, VideoData &VideoInfo, int viewID, int startTime = -1, int stopTime = -1, double threshold = 5.0, int ninliersThresh = 40, int silent = 1);
int WriteVideoDataI(char *Path, VideoData &VideoInfo, int viewID, int startTime, int stopTime, int level);


int readKeyPointJson(char *FnameKpts, vector<Point2f> &vUV, vector<float> &vConf, int nKeyPoints=17);
int ReadCamCalibInfo(char *Path, char *VideoName, int SeqId, int camId, VideoData &VideoI, int startF, int stopF);

int MineIntrinsicInfo(char *Path, CameraData &Cam, int viewID, int selectedF);

bool ReadIntrinsicResults(char *path, CameraData *DeviceParas);
bool ReadIntrinsicResultI(char *path, int selectedCamID, CameraData &CamInfoI);
int SaveIntrinsicResults(char *path, CameraData *AllViewsParas, vector<Point2i> camIDs);
int SaveAvgIntrinsicResults(char *path, CameraData *AllViewsParas, vector<int> SharedCameraToBuildCorpus);
void SaveCurrentSfmInfo(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, int npts);
void ReadCurrentSfmInfo(char *path, CameraData *AllViewParas, vector<int>&AvailViews, Point3d *All3D, int npts);

void SaveVideoCameraIntrinsic(char *Fname, CameraData *AllViewParas, vector<int>&AvailTime, int camID, int StartTime);
void SaveVideoCameraPoses(char *Fname, CameraData *AllViewParas, vector<int>&AvailTime, int camID, int StartTime = 0);

void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, Point3i *AllColor, int npts);
void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>&AvailViews, vector<Point3d>&All3D, vector<Point3i>&AllColor);

int ReadDomeCalibFile(char *Path, CameraData *AllCamInfo);
int ImportCalibDatafromHanFormat(char *Path, VideoData &AllVideoInfo, int nVGAPanels, int nVGACamsPerPanel, int nHDs);
void ExportCalibDatatoHanFormat(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime, int selectedCam = -1);

#endif
