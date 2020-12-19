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
using namespace Eigen;
using namespace smpl;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::SoftLOneLoss;
using ceres::HuberLoss;
using ceres::Problem;
using ceres::Solver;

bool autoplay = false, saveFrame = false, snapSync = false;
int ClickX = -1, ClickY = -1;
static void CheckMouse(int event, int x, int y, int, void *) {
	if (event == EVENT_LBUTTONDBLCLK) {
		ClickX = x, ClickY = y;
		cout << "\a";
	}
	else {
#ifdef _WINDOWS
		if (event == EVENT_MBUTTONDBLCLK)
		{
			ClickX = 0, ClickY = 0;
			cout << "\a\a\a";
		}
#else
#ifdef __APPLE__
		if (event == EVENT_MBUTTONDBLCLK)
		{
			ClickX = 0, ClickY = 0;
			cout << "\a\a\a";
		}
#else
		if (event == EVENT_RBUTTONDBLCLK) {
			ClickX = 0, ClickY = 0;
			cout << "\a\a\a";
		}
#endif
#endif
	}

}
void AutomaticPlay(int state, void* userdata)
{
	autoplay = !autoplay;
}
void AutomaticSave(int state, void* userdata)
{
	saveFrame = !saveFrame;
}
void SnapSyncInfo(int state, void* userdata)
{
	snapSync = !snapSync;
}
void RotateImage(Mat &img)
{
	//flip updown
	int width = img.cols, height = img.rows, nchannels = img.channels();
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height / 2; jj++)
			for (int ii = 0; ii < width; ii++)
			{
				char buf = img.data[nchannels*ii + jj * nchannels*width + kk];
				img.data[nchannels*ii + jj * nchannels*width + kk] = img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk];
				img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk] = buf;
			}
	}
	return;
}
void RotateImage(IplImage *img)
{
	//flip updown
	int width = img->width, height = img->height, nchannels = img->nChannels;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height / 2; jj++)
			for (int ii = 0; ii < width; ii++)
			{
				char buf = img->imageData[nchannels*ii + jj * nchannels*width + kk];
				img->imageData[nchannels*ii + jj * nchannels*width + kk] = img->imageData[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk];
				img->imageData[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk] = buf;
			}
	}
	return;
}
int ShowSyncLoadVideos(char *DataPath, char *SynFileName, char *SavePath, vector<int> &SelectedCams, vector<double> fps)
{
	char Fname[2000];
	int WBlock = 1920, HBlock = 1080, nBlockX = 5, nchannels = 3, MaxFrames = 10000, playBackSpeed = 1, id;
	int increF = 1;

	int nCams = (int)SelectedCams.size();
	double offset, *Offset = new double[nCams];
	Sequence *mySeq = new Sequence[nCams];

	vector<CvCapture*> allVideoData(nCams);
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "%s/%d/x.mp4", DataPath, SelectedCams[ii]);
		if (IsFileExist(Fname) == 0)
			sprintf(Fname, "%s/%d/x.mov", DataPath, SelectedCams[ii]);
		allVideoData[ii] = cvCreateFileCapture(Fname);
	}

	printLOG("Please input offset info in the format time-stamp format (f = f_ref - offset)!\n");
	sprintf(Fname, "%s/%s.txt", DataPath, SynFileName);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %lf ", &id, &offset) != EOF)
	{
		for (int ii = 0; ii < nCams; ii++)
			if (SelectedCams[ii] == id)
			{
				Offset[ii] = offset;
				break;
			}
	}
	fclose(fp);

	vector<Point2i> rotateImage;
	sprintf(Fname, "%s/RotationInfo.txt", DataPath); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int cid, code;
		while (fscanf(fp, "%d %d", &cid, &code) != EOF)
			rotateImage.push_back(Point2i(cid, code));
		fclose(fp);
	}

	int refSeq = 0;
	double earliestTime = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
	{
		if (fps.size() > 0)
			mySeq[ii].InitSeq(fps[ii], Offset[ii]);
		else
			mySeq[ii].InitSeq(1, Offset[ii]);
		if (earliestTime > Offset[ii])
			earliestTime = Offset[ii], refSeq = ii;
	}

	//Read video sequences
	int width = 0, height = 0;
	nBlockX = nCams < nBlockX ? nCams : nBlockX;
	for (int ii = 0; ii < nCams; ii++)
		width += WBlock, height += HBlock;

	//Initialize display canvas
	int nBlockY = (1.0*nCams / nBlockX > nCams / nBlockX) ? nCams / nBlockX + 1 : nCams / nBlockX;
	width = WBlock * nBlockX, height = HBlock * nBlockY;
	char *BigImg = new char[width*height*nchannels];
	char *BlackImage = new char[WBlock*HBlock*nchannels], *SubImage = new char[WBlock*HBlock*nchannels];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);

	for (int ii = 0; ii < width*height*nchannels; ii++)
		BigImg[ii] = (char)0;
	for (int ii = 0; ii < WBlock*HBlock*nchannels; ii++)
		BlackImage[ii] = (char)0;

	//Create display window
	int *oFrameID = new int[nCams + 1], *FrameID = new int[nCams + 1];
	for (int ii = 0; ii < nCams + 1; ii++)
		oFrameID[ii] = 0, FrameID[ii] = 0;
	namedWindow("VideoSequences", CV_WINDOW_NORMAL);
	cvSetWindowProperty("VideoSequences", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("Control", CV_WINDOW_NORMAL);
	createTrackbar("Speed", "Control", &playBackSpeed, 10, NULL);
	createTrackbar("Global frame", "Control", &FrameID[0], MaxFrames - 1, NULL);
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "Seq %d", SelectedCams[ii]);
		createTrackbar(Fname, "Control", &FrameID[ii + 1], MaxFrames - 1, NULL);
		cvSetTrackbarPos(Fname, "Control", 0);
	}
	char* nameb1 = "Play/Stop";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 0);
	char* nameb2 = "Not Save/Save";
	createButton(nameb2, AutomaticSave, nameb2, CV_CHECKBOX, 0);

	int BlockXID, BlockYID, setReferenceFrame, same, noUpdate;
	double setSeqFrame;
	bool *GlobalSlider = new bool[nCams]; //True: global slider, false: local slider
	for (int ii = 0; ii < nCams; ii++)
		GlobalSlider[ii] = true;


	int SaveFrameCount = 0;
	while (waitKey(17) != 27)
	{
		noUpdate = 0;
		if (playBackSpeed < 1)
			playBackSpeed = 1;
		for (int ii = 0; ii < nCams; ii++)
		{
			int doRotation = 0;
			for (int jj = 0; jj < (int)rotateImage.size(); jj++)
				if (rotateImage[jj].x == SelectedCams[ii])
					doRotation = 1;

			BlockXID = ii % nBlockX, BlockYID = ii / nBlockX;

			same = 0;
			if (GlobalSlider[ii])
				setReferenceFrame = FrameID[0]; //global frame
			else
				setReferenceFrame = FrameID[ii + 1];

			if (oFrameID[0] != FrameID[0])
				FrameID[ii + 1] = FrameID[0], GlobalSlider[ii] = true;
			else
				same += 1;

			if (oFrameID[ii + 1] != FrameID[ii + 1]) //but if local slider moves
			{
				setReferenceFrame = FrameID[ii + 1];
				if (same == 0 && GlobalSlider[ii])
					GlobalSlider[ii] = true;
				else
					GlobalSlider[ii] = false;
			}
			else
				same += 1;

			sprintf(Fname, "Seq %d", SelectedCams[ii]);
			setSeqFrame = (1.0*setReferenceFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[ii].TimeAlignPara[1]) * mySeq[ii].TimeAlignPara[0]; //(refFrame/fps_ref - offset_i)*fps_i

			if (same == 2)
			{
				noUpdate++;
				if (autoplay)
				{
					sprintf(Fname, "Seq %d", SelectedCams[ii]);
					cvSetTrackbarPos(Fname, "Control", FrameID[ii + 1]);
					FrameID[ii + 1] += playBackSpeed;
				}
				continue;
			}

			if (setSeqFrame <= 0)
			{
				cvSetTrackbarPos(Fname, "Control", (int)(mySeq[ii].TimeAlignPara[1] + 0.5));
				Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else
			{
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "Control", oFrameID[ii + 1]);
				double rem = setSeqFrame - int(setSeqFrame / increF), localFid;
				if (rem > increF / 2)
					localFid = (int(setSeqFrame / increF) + 1)*increF;
				else
					localFid = ((int)(setSeqFrame / increF))*increF;
				printLOG("Sequence %d frame %d\n", SelectedCams[ii], (int)localFid);

				CvCapture *capture = allVideoData[ii];
				if (capture == NULL)
					continue;

				cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, localFid);
				IplImage*frame = cvQueryFrame(capture);
				if (frame->imageData != NULL)
				{
					int swidth = frame->width, sheight = frame->height;
					if (doRotation == 1)
					{
						//cvShowImage("X", frame); waitKey(0);
						RotateImage(frame);
						//cvShowImage("X", frame); waitKey(0);
					}
					Set_Sub_Mat(frame->imageData, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				}
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (saveFrame)
				{
					sprintf(Fname, "%s/Mosaic", DataPath); makeDir(Fname);
					sprintf(Fname, "%s/Mosaic/%d", SavePath, SelectedCams[ii]); makeDir(Fname);

					sprintf(Fname, "%s/Mosaic/%d/%.4d.png", SavePath, SelectedCams[ii], SaveFrameCount / nCams);
					SaveDataToImageCVFormat(Fname, frame->imageData, frame->width, frame->height, nchannels);
					SaveFrameCount++;
				}
				else
					SaveFrameCount = 0;
			}
			if (autoplay)
			{
				sprintf(Fname, "Seq %d", SelectedCams[ii]);
				cvSetTrackbarPos(Fname, "Control", FrameID[ii + 1]);
				FrameID[ii + 1] += playBackSpeed;
			}
		}
		oFrameID[0] = FrameID[0];
		if (noUpdate != nCams)
			ShowCVDataAsImage("VideoSequences", BigImg, width, height, nchannels, cvImg, 0);

		if (autoplay)
		{
			int ii;
			for (ii = 0; ii < nCams; ii++)
				if (!GlobalSlider[ii])
					break;
			if (ii == nCams)
			{
				cvSetTrackbarPos("Global frame", "Control", FrameID[0]);
				FrameID[0] += playBackSpeed;
			}

			if (saveFrame)
			{
				char Fname[512];  sprintf(Fname, "C:/temp/%.4d.png", FrameID[0]);
				SaveDataToImage(Fname, BigImg, width, height, 3);
			}
		}
	}

	cvReleaseImage(&cvImg);
	delete[]Offset, delete[]oFrameID, delete[]FrameID, delete[]GlobalSlider;
	delete[]BigImg, delete[]BlackImage;

	return 0;
}
int ShowSyncLoadImages(char *DataPath, char *SynFileName, char *SavePath, vector<int> &SelectedCams, vector<double> &fps, int startF, int stopF, int nCOCOJoints, double desiredRatio = 16.0 / 9, double resizeFactor = 0.25)
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

	char Fname[2000];
	int WBlock = 0, HBlock = 0, nBlockX = 5, nchannels = 3, playBackSpeed = 1, id;
	int increF = 1;
	Mat img;
	for (auto cid : SelectedCams)
	{
		for (int fid = startF; fid <= stopF; fid++)
		{
			sprintf(Fname, "%s/%d/%.4d.jpg", DataPath, cid, fid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/%d/%.4d.png", DataPath, cid, fid);
				if (IsFileExist(Fname) == 0)
					continue;
			}
			img = imread(Fname, 0);
			WBlock = max(WBlock, img.cols), HBlock = max(HBlock, img.rows);
			break;
		}
	}

	int nCams = (int)SelectedCams.size();
	double offset, *Offset = new double[nCams];
	Sequence *mySeq = new Sequence[nCams];

	double bestRatioDif = 9e9;  nBlockX = 1;
	for (int TitleW = 1; TitleW <= SelectedCams.size(); TitleW++)
	{
		int bWidth = WBlock * TitleW, bHeight = (int)(ceil(1.0* SelectedCams.size() / nBlockX))*HBlock;
		double ratio = 1.0*bWidth / bHeight;
		double ratioDif = abs(ratio - desiredRatio);
		if (ratioDif < bestRatioDif)
		{
			bestRatioDif = ratioDif;
			nBlockX = TitleW;
		}
	}

	printLOG("Please input offset info in the format time-stamp format (f = f_ref - offset)!\n");
	sprintf(Fname, "%s/%s.txt", DataPath, SynFileName);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	double rate;
	while (fscanf(fp, "%d %lf %lf ", &id, &rate, &offset) != EOF)
	{
		for (int ii = 0; ii < nCams; ii++)
			if (SelectedCams[ii] == id)
			{
				Offset[ii] = offset;
				fps[ii] = rate;
				break;
			}
	}
	fclose(fp);

	int refSeq = 0;
	double earliestTime = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
	{
		if (fps.size() > 0)
			mySeq[ii].InitSeq(fps[ii], Offset[ii]);
		else
			mySeq[ii].InitSeq(1, Offset[ii]);
		if (earliestTime > Offset[ii])
			earliestTime = Offset[ii], refSeq = ii;
	}
	printLOG("Pick %d as reference camera\n", refSeq);

	//Read video sequences
	int width = 0, height = 0;
	nBlockX = nCams < nBlockX ? nCams : nBlockX;
	for (int ii = 0; ii < nCams; ii++)
		width += WBlock, height += HBlock;

	//Initialize display canvas
	int nBlockY = (1.0*nCams / nBlockX > nCams / nBlockX) ? nCams / nBlockX + 1 : nCams / nBlockX;
	width = WBlock * nBlockX, height = HBlock * nBlockY;
	uchar *BigImg = new uchar[width*height*nchannels];
	uchar *BlackImage = new uchar[WBlock*HBlock*nchannels], *SubImage = new uchar[WBlock*HBlock*nchannels];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);
	Mat cvBigImg, cvrBigImg;

	for (int ii = 0; ii < width*height*nchannels; ii++)
		BigImg[ii] = (uchar)0;
	for (int ii = 0; ii < WBlock*HBlock*nchannels; ii++)
		BlackImage[ii] = (uchar)0;

	//Create display window
	int *oFrameID = new int[nCams + 1], *FrameID = new int[nCams + 1];
	for (int ii = 0; ii < nCams + 1; ii++)
		oFrameID[ii] = 0, FrameID[ii] = 0;
	namedWindow("VideoSequences", CV_WINDOW_NORMAL);
	//cvSetWindowProperty("VideoSequences", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	namedWindow("Control", CV_WINDOW_NORMAL);
	createTrackbar("Speed", "Control", &playBackSpeed, 10, NULL);
	createTrackbar("Global frame", "Control", &FrameID[0], stopF - 1, NULL);
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "Seq %d", SelectedCams[ii]);
		createTrackbar(Fname, "Control", &FrameID[ii + 1], stopF - 1, NULL);
		cvSetTrackbarPos(Fname, "Control", 0);
	}
	cvSetTrackbarPos("Global frame", "Control", startF);

	char* nameb1 = "Play/Stop Toggle";
	createButton(nameb1, AutomaticPlay, nameb1, CV_CHECKBOX, 0);
	char* nameb2 = "Save Toggle";
	createButton(nameb2, AutomaticSave, nameb2, CV_CHECKBOX, 0);
	char* nameb3 = "Snap Sync Info Toggle";
	createButton(nameb3, SnapSyncInfo, nameb3, CV_CHECKBOX, 0);

	int BlockXID, BlockYID, setReferenceFrame, same, noUpdate;
	double setSeqFrame;
	bool *GlobalSlider = new bool[nCams]; //True: global slider, false: local slider
	for (int ii = 0; ii < nCams; ii++)
		GlobalSlider[ii] = true;

	bool HasMergedTracklet = true;
	Point2f *joints = new Point2f[100 * 18];
	vector<vector<Point2i> >*MultiviewTracketVec = new vector<vector< Point2i> >[nCams * 2];
	for (auto cid : SelectedCams)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_%d_%d.txt", DataPath, cid, startF, stopF);
		if (!IsFileExist(Fname))
		{
			sprintf(Fname, "%s/%d/MergedTracklets_%d_%d.txt", DataPath, cid, startF, stopF);
			if (!IsFileExist(Fname))
			{
				HasMergedTracklet = false;
				continue;
			}
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
			int fid, pid;
			while (!line_stream.eof()) {
				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				fid = atoi(item.c_str());
				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				pid = atoi(item.c_str());
				jointTrack.push_back(Point2i(fid, pid));
			}
			MultiviewTracketVec[cid].push_back(jointTrack);
		}
		file.close();
		MultiviewTracketVec[cid].pop_back(); //last one is trash
	}

	bool drawPose = true, firstTimeSaving = true;
	VideoWriter writer;
	int SaveFrameCount = 0;
	while (true)
	{
		int key = waitKey(17);
		if (key == 27)
			break;
		if (FrameID[0] > stopF)
			break;

		if (key == 13) //enter
			drawPose = !drawPose;

		noUpdate = 0;
		if (playBackSpeed < 1)
			playBackSpeed = 1;

		if (snapSync)
		{
			vector<int> vLocalFid, vOffset;
			for (int ii = 0; ii < nCams; ii++)
			{
				setReferenceFrame = FrameID[ii + 1];
				setSeqFrame = (1.0*setReferenceFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[ii].TimeAlignPara[1]) * mySeq[ii].TimeAlignPara[0]; //f = (refFrame/fps_ref - offset_i)*fps_i
				double rem = setSeqFrame - int(setSeqFrame / increF), localFid;
				if (rem > increF / 2)
					localFid = (int(setSeqFrame / increF) + 1)*increF;
				else
					localFid = ((int)(setSeqFrame / increF))*increF;
				vLocalFid.push_back(localFid);
			}
			for (int ii = 0; ii < nCams; ii++)
			{
				double offset_i = vLocalFid[refSeq] / mySeq[refSeq].TimeAlignPara[0] - vLocalFid[ii] / mySeq[ii].TimeAlignPara[0];
				vOffset.push_back(offset_i);
			}
			sprintf(Fname, "%s/%s_s.txt", DataPath, SynFileName); FILE*fp = fopen(Fname, "w+");
			for (int ii = 0; ii < nCams; ii++)
				fprintf(fp, "%d %.1f %d\n", ii, fps[ii], vOffset[ii]);
			fclose(fp);

			snapSync = false;
		}

		for (int ii = 0; ii < nCams; ii++)
		{
			BlockXID = ii % nBlockX, BlockYID = ii / nBlockX;

			same = 0;
			if (GlobalSlider[ii])
			{
				FrameID[0] = max(startF, FrameID[0]);
				setReferenceFrame = FrameID[0]; //global frame
			}
			else
				setReferenceFrame = FrameID[ii + 1];

			if (oFrameID[0] != FrameID[0])
				FrameID[ii + 1] = FrameID[0], GlobalSlider[ii] = true;
			else
				same += 1;

			if (oFrameID[ii + 1] != FrameID[ii + 1]) //but if local slider moves
			{
				setReferenceFrame = FrameID[ii + 1];
				if (same == 0 && GlobalSlider[ii])
					GlobalSlider[ii] = true;
				else
					GlobalSlider[ii] = false;
			}
			else
				same += 1;

			sprintf(Fname, "Seq %d", SelectedCams[ii]);
			setSeqFrame = (1.0*setReferenceFrame - mySeq[ii].TimeAlignPara[1]) / mySeq[refSeq].TimeAlignPara[0] * mySeq[ii].TimeAlignPara[0];
			if (same == 2)
			{
				noUpdate++;
				if (autoplay)
				{
					sprintf(Fname, "Seq %d", SelectedCams[ii]);
					cvSetTrackbarPos(Fname, "Control", FrameID[ii + 1]);
					FrameID[ii + 1] += playBackSpeed;
				}
				continue;
			}

			if (setSeqFrame <= 0)
			{
				cvSetTrackbarPos(Fname, "Control", (int)(mySeq[ii].TimeAlignPara[1] + 0.5));
				Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
			}
			else
			{
				oFrameID[ii + 1] = FrameID[ii + 1];
				cvSetTrackbarPos(Fname, "Control", oFrameID[ii + 1]);
				//double rem = setSeqFrame - int(setSeqFrame / increF), localFid;
				//if (rem > increF / 2)
				//	localFid = (int(setSeqFrame / increF) + 1)*increF;
				//else
				//	localFid = ((int)(setSeqFrame / increF))*increF;
				int localFid = setSeqFrame;
				printf("Sequence %d frame %d\n", SelectedCams[ii], (int)localFid);

				sprintf(Fname, "%s/%d/%.4d.png", DataPath, SelectedCams[ii], (int)localFid);
				if (IsFileExist(Fname) == 0)
					sprintf(Fname, "%s/%d/%.4d.jpg", DataPath, SelectedCams[ii], (int)localFid);
				Mat frame = imread(Fname);

				int pointCount = 0;
				float u, v, s;
				vector<float> vConf; vector<Point2f> vUV;
				if (readOpenPoseJson(DataPath, SelectedCams[ii], localFid, vUV, vConf) == 0)
				{
					sprintf(Fname, "%s/MP/%d/%.4d.txt", DataPath, SelectedCams[ii], (int)localFid);
					if (IsFileExist(Fname) == 1)
					{
						FILE *fp = fopen(Fname, "r");
						while (fscanf(fp, "%f %f %f ", &u, &v, &s) != EOF)
							joints[pointCount] = Point2f(u, v), pointCount++;
						fclose(fp);
					}
				}
				else
				{
					for (int ii = 0; ii < vUV.size(); ii++)
						joints[ii] = vUV[ii], pointCount++;
				}
				if (pointCount > 0)
				{
					int npeople = pointCount / nCOCOJoints;
					for (int pid = 0; pid < npeople; pid++)
					{
						int realPid = -1;
						for (size_t tid = 0; tid < MultiviewTracketVec[SelectedCams[ii]].size() && realPid == -1; tid++) //people track ID: consistent accross all views
						{
							for (size_t lpid = 0; lpid < MultiviewTracketVec[SelectedCams[ii]][tid].size() && realPid == -1; lpid++) //temporal location of the person in the track
							{
								int lfid = (int)localFid;
								if (lfid == MultiviewTracketVec[SelectedCams[ii]][tid][lpid].x && pid == MultiviewTracketVec[SelectedCams[ii]][tid][lpid].y)
									realPid = (int)tid;
							}
						}
						float minX = 9e9, minY = 9e9, maxX = 0, maxY = 0;
						for (int jid = 0; jid < nCOCOJoints; jid++)
							if (joints[pid*nCOCOJoints + jid].x > 0)
								minX = min(minX, joints[pid*nCOCOJoints + jid].x), maxX = max(maxX, joints[pid*nCOCOJoints + jid].x), minY = min(minY, joints[pid*nCOCOJoints + jid].y), maxY = max(maxY, joints[pid*nCOCOJoints + jid].y);

						if (drawPose)
						{
							Draw2DCoCoJoints(frame, joints + nCOCOJoints * pid, nCOCOJoints, 2, 1);
							CvPoint text_origin = { MyFtoI(minX), MyFtoI(maxY - frame.rows / 20) };

							if (realPid != -1)
							{
								sprintf(Fname, "%d", realPid), putText(frame, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5 * frame.cols / 640, colors[realPid % colors.size()], 3);
								rectangle(frame, Point2i(minX - frame.cols / 50, minY - frame.cols / 50), Point2i(maxX + frame.rows / 50, maxY + frame.rows / 50), colors[realPid % colors.size()], 1, 8, 0);
							}
							else
							{
								if (!HasMergedTracklet)
								{
									sprintf(Fname, "%d", pid), putText(frame, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5 * frame.cols / 640, colors[pid % colors.size()], 3);
									rectangle(frame, Point2i(minX - frame.cols / 50, minY - frame.cols / 50), Point2i(maxX + frame.rows / 50, maxY + frame.rows / 50), colors[pid % colors.size()], 1, 8, 0);
								}
							}
						}
					}
				}

				if (frame.empty() == 0)
				{
					int swidth = frame.cols, sheight = frame.rows;
					Set_Sub_Mat(frame.data, BigImg, nchannels*swidth, sheight, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);
				}
				else
					Set_Sub_Mat(BlackImage, BigImg, nchannels*WBlock, HBlock, nchannels*width, nchannels*BlockXID*WBlock, BlockYID*HBlock);

				if (saveFrame)
				{
					;//sprintf(Fname, "%s/Mosiac", DataPath); makeDir(Fname);
					 //sprintf(Fname, "%s/Mosiac/%d", SavePath, SelectedCams[ii]); makeDir(Fname);
					 //sprintf(Fname, "%s/Mosiac/%d/%.4d.png", SavePath, SelectedCams[ii], SaveFrameCount / nCams);
					 //SaveDataToImageCVFormat(Fname, frame.data, frame.cols, frame.rows, nchannels);
					 //SaveFrameCount++;

					 //sprintf(Fname, "%s/Mosiac/%.4d", SavePath, FrameID[0]); makeDir(Fname);
					 //sprintf(Fname, "%s/Mosiac/%.4d/%d.png", SavePath, FrameID[0], SelectedCams[ii]);
					 //SaveDataToImageCVFormat(Fname, frame.data, frame.cols, frame.rows, nchannels);
				}
				else
					SaveFrameCount = 0;
			}
			if (autoplay)
			{
				sprintf(Fname, "Seq %d", SelectedCams[ii]);
				cvSetTrackbarPos(Fname, "Control", FrameID[ii + 1]);
				FrameID[ii + 1] += playBackSpeed;
			}
		}
		oFrameID[0] = FrameID[0];
		if (noUpdate != nCams)
		{
			ShowCVDataAsImage("VideoSequences", BigImg, width, height, nchannels, cvImg, 0);
			if (saveFrame)
			{
				if (firstTimeSaving)
				{
					firstTimeSaving = false;

					sprintf(Fname, "%s/Mosiac", SavePath); makeDir(Fname);
					CvSize size = cvSize((int)(resizeFactor*width), (int)(resizeFactor*height));
					sprintf(Fname, "%s/Mosiac/mosiac.avi", SavePath), writer.open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 60, size);
					cvBigImg = Mat(height, width, CV_8UC3, BigImg);
				}
				sprintf(Fname, "%s/Mosiac/%.4d.png", SavePath, FrameID[0]);
				SaveDataToImageCVFormat(Fname, BigImg, width, height, nchannels);
				imwrite(Fname, cvBigImg);

				/*if (resizeFactor == 1)
				{
					CvPoint text_origin = { cvBigImg.rows / 20, cvBigImg.cols / 20 };
					sprintf(Fname, "%d ", FrameID[0]);
					putText(cvBigImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*cvBigImg.cols / 640, cv::Scalar(0, 255, 0), 3);
					writer << cvBigImg;
				}
				else
				{
					resize(cvBigImg, cvrBigImg, Size((int)(resizeFactor*width), (int)(resizeFactor*height)), 0, 0, INTER_AREA);
					CvPoint text_origin = { cvrBigImg.rows / 20, cvrBigImg.cols / 20 };
					sprintf(Fname, "%d ", FrameID[0]);
					putText(cvrBigImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*cvrBigImg.cols / 640, cv::Scalar(0, 255, 0), 3);
					writer << cvrBigImg;
				}*/

				//Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
				//for (jj = 0; jj < height; jj++)
				//	for (ii = 0; ii < width; ii++)
				//		for (kk = 0; kk < nchannels; kk++)
				//			M.data[nchannels*ii + kk + nchannels*jj*width] = Img[nchannels*ii + kk + nchannels*jj*width];
			}
		}

		if (autoplay)
		{
			int ii;
			for (ii = 0; ii < nCams; ii++)
				if (!GlobalSlider[ii])
					break;
			if (ii == nCams)
			{
				cvSetTrackbarPos("Global frame", "Control", FrameID[0]);
				FrameID[0] += playBackSpeed;
			}
		}
		if (autoplay && saveFrame)
			if (GlobalSlider[0] == stopF)
				break;
	}
	writer.release();

	cvReleaseImage(&cvImg);
	delete[]Offset, delete[]oFrameID, delete[]FrameID, delete[]GlobalSlider;
	delete[]BigImg, delete[]BlackImage;

	return 0;
}
int ShowSeqAlignLoadImages(char *DataPath, char *SavePath, int startF, int stopF, int timeScale = 1)
{
	char Fname[2000];

	sprintf(Fname, "%s/InitSync.txt", DataPath);	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n. Abort!", Fname);
		return 1;
	}
	int sCid, Ref_lcid = 0, offsetValue, largestValue = -9999;
	double fps; vector<double> vpfs;
	vector<int> vsCams, vOffset;
	while (fscanf(fp, "%d %lf %d ", &sCid, &fps, &offsetValue) != EOF)
	{
		if (largestValue < offsetValue)
			Ref_lcid = vsCams.size(), largestValue = offsetValue; //earliest Cid

		vsCams.push_back(sCid);
		vpfs.push_back(fps);
		vOffset.push_back(offsetValue);
	}
	fclose(fp);
	int nCams = vsCams.size();

	vector<int> oOrder, Order;
	for (int ii = 0; ii < nCams; ii++)
		oOrder.push_back(ii), Order.push_back(ii);

	//now, view them in time
	Mat img;
	int ocurrentRefFrame = startF, currentRefFrame = startF, currentCamOrder = 0, BackStayFore = 1, key;

	cv::namedWindow("SeqAlign", WINDOW_NORMAL);
	namedWindow("Control", CV_WINDOW_NORMAL);
	createTrackbar("RefFrame", "Control", &currentRefFrame, stopF, NULL);
	createTrackbar("Backward/Foreward (0 or 2", "Control", &BackStayFore, 2, NULL);
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "Cam %d order", vsCams[ii]);
		createTrackbar(Fname, "Control", &Order[ii], nCams - 1, NULL);
	}
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "Offset %d", vsCams[ii]);
		createTrackbar(Fname, "Control", &vOffset[ii], largestValue + 20, NULL);
	}

	bool first = true, go = false;
	while (true)
	{
		bool changed = false;
		if (ocurrentRefFrame != currentRefFrame)
			changed = true;

		bool allDone = true;
		for (int ii = 0; ii < nCams &&allDone; ii++)
		{
			for (int jj = 0; jj < nCams &&allDone; jj++)
				if (ii != jj && Order[ii] == Order[jj])
					allDone = false;
		}
		for (int ii = 0; ii < nCams &&allDone && !changed; ii++)
		{
			if (oOrder[ii] != Order[ii])
			{
				changed = true;
				for (int jj = 0; jj < nCams; jj++)
					oOrder[jj] = Order[jj];
			}
		}

		if (go || first || BackStayFore != 1 || changed)
		{
			changed = false;

			if ((BackStayFore == 2 || go || first) && currentCamOrder == nCams - 1)
				currentRefFrame += timeScale;
			if (BackStayFore == 0 && currentCamOrder == 0)
				currentRefFrame -= timeScale;
			ocurrentRefFrame = currentRefFrame;

			if (go)
				currentCamOrder = (currentCamOrder + 1) % nCams;
			else if (BackStayFore != 1)
				currentCamOrder = (currentCamOrder + BackStayFore - 1 + nCams) % nCams;

			int lcid = -1;
			for (int ii = 0; ii < nCams && lcid == -1; ii++)
			{
				if (Order[ii] == currentCamOrder)
					lcid = ii;
			}
			int trueCid = vsCams[lcid], trueFrame = currentRefFrame - vOffset[lcid];

			printf("Loading %d/%d %d\n", trueCid, trueFrame, BackStayFore);
			sprintf(Fname, "%s/%d/%.4d.png", DataPath, trueCid, trueFrame);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/%d/%.4d.jpg", DataPath, trueCid, trueFrame);
				if (IsFileExist(Fname) == 0)
				{
					if (BackStayFore == 1)
						currentCamOrder = (currentCamOrder + 1) % nCams;

					continue;
				}
			}
			img = imread(Fname);
			CvPoint text_origin = { img.rows / 20, img.cols / 20 };
			sprintf(Fname, "%d/%d", trueCid, trueFrame);	putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 255, 0), 3);

			for (int ii = 0; ii < nCams; ii++)
			{
				sprintf(Fname, "Cam %d order", vsCams[ii]);
				setTrackbarPos(Fname, "Control", Order[ii]);
				sprintf(Fname, "Offset %d", vsCams[ii]);
				setTrackbarPos(Fname, "Control", vOffset[ii]);
			}
			first = false;
			BackStayFore = 1;
			setTrackbarPos("RefFrame", "Control", currentRefFrame);
			setTrackbarPos("Backward/Foreward (0 or 2", "Control", BackStayFore);
		}

		imshow("SeqAlign", img);
		key = waitKey(1);
		if (key == 27)
			break;
		if (key == 13) //enter
			go = !go;
	}

	sprintf(Fname, "%s/InitSync2.txt", DataPath); fp = fopen(Fname, "w");
	for (int ii = 0; ii < nCams; ii++)
		fprintf(fp, "%d %.2f %.2f\n", vsCams[ii], vpfs[ii], 1.0*vOffset[ii] - 1.0*Order[ii] / nCams);
	fclose(fp);

	return 0;
}
int ShowSeqAlignLoadImages2(char *DataPath, char *SynFileName, char *SavePath, vector<int> &SelectedCams, int startF, int stopF, int timeScale = 1)
{
	char Fname[2000];

	int nCams = (int)SelectedCams.size();
	double offset, *Offset = new double[nCams];
	Sequence *mySeq = new Sequence[nCams];

	printLOG("Please input offset info in the format time-stamp format (f = f_ref - offset)!\n");
	sprintf(Fname, "%s/%s.txt", DataPath, SynFileName);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	int cid;
	double fps;
	while (fscanf(fp, "%d %lf %lf ", &cid, &fps, &offset) != EOF)
	{
		for (int ii = 0; ii < nCams; ii++)
			if (SelectedCams[ii] == cid)
			{
				mySeq[ii].TimeAlignPara[0] = fps;
				mySeq[ii].TimeAlignPara[1] = offset;
				break;
			}
	}
	fclose(fp);

	int refSeq = 0;
	double earliestTime = DBL_MAX;
	for (auto cid : SelectedCams)
		if (earliestTime > mySeq[cid].TimeAlignPara[1])
			earliestTime = mySeq[cid].TimeAlignPara[1], refSeq = cid;
	printLOG("Pick %d as reference camera\n", refSeq);

	int ocurrentRefFrame = startF, currentRefFrame = startF;
	int * cFrameId = new int[nCams],
		*ocFrameId = new int[nCams];
	for (auto cid : SelectedCams)
	{
		int rfid = (int)((1.0*currentRefFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[cid].TimeAlignPara[1]) * mySeq[cid].TimeAlignPara[0]);
		cFrameId[cid] = rfid, ocFrameId[cid] = rfid;
	}

	//now, view them in time
	int currentCamOrder = 0, BackStayFore = 1, key;

	cv::namedWindow("SeqAlign", WINDOW_NORMAL);
	namedWindow("Control", CV_WINDOW_NORMAL);
	createTrackbar("Refframe", "Control", &currentRefFrame, stopF + 100, NULL);
	createTrackbar("Backward/Foreward (0 or 2", "Control", &BackStayFore, 2, NULL);
	for (int ii = 0; ii < nCams; ii++)
	{
		sprintf(Fname, "Cam %d", SelectedCams[ii]);
		createTrackbar(Fname, "Control", &cFrameId[ii], stopF + 100, NULL);
	}

	int cnt = 0;
	vector<int>order, tFid;
	currentCamOrder = 0;
	Mat img;
	while (currentRefFrame < stopF)
	{
		for (auto cid : SelectedCams)
		{
			int rfid = (int)((1.0*currentRefFrame / mySeq[refSeq].TimeAlignPara[0] - mySeq[cid].TimeAlignPara[1]) * mySeq[cid].TimeAlignPara[0]);
			cFrameId[cid] = rfid, ocFrameId[cid] = rfid;
		}

		order.clear(), tFid.clear();
		for (auto cid_ : SelectedCams)
		{
			order.push_back(cid_);
			tFid.push_back(cFrameId[cid]);
		}
		Quick_Sort_Int(&tFid[0], &order[0], 0, order.size() - 1);

		sprintf(Fname, "%s/%d/%.4d.png", DataPath, SelectedCams[order[0]], tFid[0]);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/%d/%.4d.jpg", DataPath, SelectedCams[order[0]], tFid[0]);
			if (IsFileExist(Fname) == 0)
			{
				currentRefFrame += timeScale;
				ocurrentRefFrame = currentRefFrame;
				continue;
			}
		}
		img = imread(Fname);
		break;
	}
	while (true)
	{
		bool changed = false;
		if (ocurrentRefFrame != currentRefFrame)
		{
			if (abs(currentRefFrame - ocurrentRefFrame) == 1)
			{
				currentRefFrame = ocurrentRefFrame + (currentRefFrame - ocurrentRefFrame)* timeScale;
				for (auto cid : SelectedCams)
					cFrameId[cid] += (currentRefFrame - ocurrentRefFrame)* timeScale;
			}
			else
			{
				cnt = (currentRefFrame - ocurrentRefFrame) / timeScale;
				currentRefFrame = ocurrentRefFrame + cnt * timeScale;
				for (auto cid : SelectedCams)
					cFrameId[cid] += cnt * timeScale;
			}
			ocurrentRefFrame = currentRefFrame, cnt = 0;
			changed = true;
		}
		for (auto cid : SelectedCams)
		{
			if (ocFrameId[cid] != cFrameId[cid])
			{
				for (auto cid_ : SelectedCams)
					ocFrameId[cid] = cFrameId[cid];
				currentCamOrder = 0;

				order.clear(), tFid.clear();
				for (auto cid_ : SelectedCams)
				{
					order.push_back(cid_);
					tFid.push_back(cFrameId[cid]);
				}
				Quick_Sort_Int(&tFid[0], &order[0], 0, order.size() - 1);
				cnt = 0;
				break;
			}
		}

		if (BackStayFore != 1 || changed)
		{
			changed = false;

			int ccid = order[currentCamOrder%SelectedCams.size()];
			int nframeId = cFrameId[ccid];

			if ((BackStayFore == 2))
			{
				nframeId += cnt * timeScale + currentCamOrder * timeScale / nCams;
				currentCamOrder = (currentCamOrder + 1) % SelectedCams.size();
				if (currentCamOrder == 0)
					cnt++;
			}
			if (BackStayFore == 0 && currentCamOrder == 0)
			{
				nframeId -= cnt * timeScale - currentCamOrder * timeScale / nCams;
				currentCamOrder = (currentCamOrder - 1) % SelectedCams.size();
				if (currentCamOrder == SelectedCams.size() - 1)
					cnt--;
			}

			sprintf(Fname, "%s/%d/%.4d.png", DataPath, ccid, nframeId);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/%d/%.4d.jpg", DataPath, ccid, nframeId);
				if (IsFileExist(Fname) == 0)
					continue;
			}
			printf("Loading %d/%d %d\n", ccid, nframeId, BackStayFore);
			img = imread(Fname);
			CvPoint text_origin = { img.rows / 20, img.cols / 20 };
			sprintf(Fname, "%d/%d", ccid, nframeId);	putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 255, 0), 3);
			BackStayFore = 1;
		}

		for (auto cid : SelectedCams)
		{
			sprintf(Fname, "Cam %d", cid);
			setTrackbarPos(Fname, "Control", cFrameId[cid]);
		}
		setTrackbarPos("Refframe", "Control", currentRefFrame);
		setTrackbarPos("Backward/Foreward (0 or 2", "Control", BackStayFore);

		imshow("SeqAlign", img);
		key = waitKey(1);
		if (key == 27)
			break;
		if (key == 13) //enter
			;
	}

	return 0;
}


int main(int argc, char** argv)
{

	if(0)
	{
		char Path[] = "C:/temp/JiuJitsu";

		vector<int> sCams;
		int nthreads = omp_get_max_threads();
		Mat *colorImg = new Mat[nthreads], *grayImg = new Mat[nthreads];
		vector<Point2f> *Feat = new vector<Point2f>[nthreads];

		omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic,1)
		for (int ii = 0; ii < 48; ii++)
		{
			char Fname[512];
			if (ii % 100 == 0)
				printLOG("%d/%d...", ii, 48);

			int threadId = omp_get_thread_num();
			Feat[threadId].clear();
			sprintf(Fname, "%s/%.4d.jpg", Path, ii);
			colorImg[threadId] = imread(Fname);

			cvtColor(colorImg[threadId], grayImg[threadId], CV_BGR2GRAY);
			ExtractSiftCPU(Path, 0, ii, grayImg[threadId], Feat[threadId], true);
		}
		printLOG("\n");
		return 0;
	}
	{
		vector<int> sCams;
		sCams.push_back(0);
		//visualizationDriver(argv[1], sCams, 0, 60000, 1, false, false, false, true, false, 25, 100, true, 0);

		//	VisualizeAllViewsEpipolarGeometry(argv[1], sCams, 500, 1500);
		//return 0;
	}

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

	SfMPara mySfMPara;
	CommandParse(Path, mySfMPara);

	int	mode = atoi(argv[1 + 1]);
	if (0)
	{
		int SeqId = atoi(argv[3]), startF = atoi(argv[4]), stopF = atoi(argv[5]), subCamStart = atoi(argv[6]), subCamStop = atoi(argv[7]);
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
		visualizationDriver2(Path, vVideoNames, SeqId, subCamIds, sCams, startF, stopF, 1, true, false, false, false, false, 17, startF, true);
		return 0;
	}

	if (mode == -2)
	{
		//TestPnPf(Path, 10, 0, 9000);
		//TestKSfM(Path, 10, 0, 30);
		//SequenceSfMDriver(Path, 0, 0, 200, mySfMPara);
		//ExtractColorForCorpusCloud(Path);
		//ReBundleFromDifferentShutterModel(Path);
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 0); //Extract frames
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 1); //Keyframe gen
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 2); //Sift Gen
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 3); //Corpus Gen
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 4); //Find images to match -->match any frames to corpus, usually needed if corpus was generated from keyframe of all videos (no dedicated corpus video)
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 5); //PnP Match -->match any frames to corpus, usually needed if corpus was generated from keyframe of all videos (no dedicated corpus video)
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 6); //PnP Estimate -->match any frames to corpus, usually needed if corpus was generated from keyframe of all videos (no dedicated corpus video)
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 7); //Match kf withtin camera -->apply if a dedicated corpus video was captured
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 8); //Register to corpus via local video BA for kf with anchored global Corpus -->apply if a dedicated corpus video was captured
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 10); //Video BA 
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 11); //All videos BA
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 12);// Per cam rs temporal para estimation
		//BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, 13); //Visualization
		//VisualizeCorpusFeaturesAcrossVideoCamera(Path, nCams, startF, stopF, 1);
		//VisualizeKeyFrameFormationTracking(Path, 5, 179, 1);
		//ComputePnPInlierStats(Path, nCams, startF, stopF);
		//Visualize_VideoKeyFrame2CorpusSfM_Inliers(Path, 0, startF, stopF);
		//AssociatePeopleAndBuild3DFromCalibedSyncedCameras_3DVoting(Path, sCams, 104, 104, 1, 0, 0, 3, 0.4, 20.0, real2SfM, 30);
		//AssociatePeopleAndBuild3DFromCalibedSyncedCameras_RansacTri(Path, sCams, startF, stopF, 1, 0, 0, 3, 20, 10, 0.4, 100);

		return 0;
	}
	else if (mode == -1)
	{
		int startF = 0, stopF = 3400, increF = 1, distortionCorrected = 0, nInlierThresh2 = 30, fromKeyFrameTracking = 1;
		for (int cid = 4; cid < 5; cid++)
		{
			sprintf(Fname, "%s/%d/PnPf", Path, cid), makeDir(Fname);
			sprintf(Fname, "%s/%d/PnPTc", Path, cid), makeDir(Fname);
			sprintf(Fname, "%s/%d/PnPmTc", Path, cid), makeDir(Fname);

			//identify keyframe
			vector<int> KeyFrameID2LocalFrameID;
			for (int ii = 0; ii <= stopF; ii += 30)
				KeyFrameID2LocalFrameID.push_back(ii);

			VideoData VideoI;
			if (ReadVideoDataI(Path, VideoI, cid, mySfMPara.startF, mySfMPara.stopF) == 1)
				return 1;

			convertPnP2KPnP(Path, cid, mySfMPara.startF, mySfMPara.stopF);

			vector<int> ValidKeyFrames;  ValidKeyFrames.reserve(KeyFrameID2LocalFrameID.size());
			for (size_t keyFrameID = 0; keyFrameID < KeyFrameID2LocalFrameID.size(); keyFrameID++)
			{
				sprintf(Fname, "%s/%d/PnP/KF_Inliers_%.4d.txt", Path, cid, KeyFrameID2LocalFrameID[keyFrameID]);
				if (IsFileExist(Fname) == 1 && VideoI.VideoInfo[KeyFrameID2LocalFrameID[keyFrameID]].valid == 1)
					ValidKeyFrames.push_back(1);
				else
					ValidKeyFrames.push_back(0);
			}

			for (size_t keyFrameID = 0; keyFrameID < KeyFrameID2LocalFrameID.size(); keyFrameID++)
			{
				int winDim = mySfMPara.trackingWinSize, cvPyrLevel = mySfMPara.cvPyrLevel, trackRange;
				int reffid = KeyFrameID2LocalFrameID[keyFrameID];

				if (keyFrameID == 0)
				{
					int nextValidFrame = keyFrameID + 2;
					for (int ii = nextValidFrame; ii < KeyFrameID2LocalFrameID.size(); ii++)
					{
						if (ValidKeyFrames[ii] == 1)
						{
							nextValidFrame = ii;
							break;
						}
					}
					trackRange = (KeyFrameID2LocalFrameID[nextValidFrame] - KeyFrameID2LocalFrameID[keyFrameID]);
				}
				else if (keyFrameID == KeyFrameID2LocalFrameID.size() - 1)
				{
					int nextValidFrame = keyFrameID - 2;
					for (int ii = nextValidFrame; ii >= 0; ii--)
					{
						if (ValidKeyFrames[ii] == 1)
						{
							nextValidFrame = ii;
							break;
						}
					}
					trackRange = (KeyFrameID2LocalFrameID[keyFrameID] - KeyFrameID2LocalFrameID[nextValidFrame]);
				}
				else if (keyFrameID > 1 && keyFrameID < KeyFrameID2LocalFrameID.size() - 2)
				{
					int nextValidFrame1 = keyFrameID + 2;
					for (int ii = nextValidFrame1; ii < KeyFrameID2LocalFrameID.size(); ii++)
					{
						if (ValidKeyFrames[ii] == 1)
						{
							nextValidFrame1 = ii;
							break;
						}
					}
					int nextValidFrame2 = keyFrameID - 2;
					for (int ii = nextValidFrame2; ii >= 0; ii--)
					{
						if (ValidKeyFrames[ii] == 1)
						{
							nextValidFrame2 = ii;
							break;
						}
					}
					trackRange = max(KeyFrameID2LocalFrameID[nextValidFrame1] - KeyFrameID2LocalFrameID[keyFrameID], KeyFrameID2LocalFrameID[keyFrameID] - KeyFrameID2LocalFrameID[nextValidFrame2]);
				}
				else if (keyFrameID > 0 && keyFrameID < KeyFrameID2LocalFrameID.size() - 1)
				{
					int nextValidFrame1 = keyFrameID + 1;
					for (int ii = nextValidFrame1; ii < KeyFrameID2LocalFrameID.size(); ii++)
					{
						if (ValidKeyFrames[ii] == 1)
						{
							nextValidFrame1 = ii;
							break;
						}
					}
					int nextValidFrame2 = keyFrameID - 1;
					for (int ii = nextValidFrame2; ii >= 0; ii--)
					{
						if (ValidKeyFrames[ii] == 1)
						{
							nextValidFrame2 = ii;
							break;
						}
					}
					trackRange = 2 * max(KeyFrameID2LocalFrameID[nextValidFrame1] - KeyFrameID2LocalFrameID[keyFrameID], KeyFrameID2LocalFrameID[keyFrameID] - KeyFrameID2LocalFrameID[nextValidFrame2]);
				}
				//this function produces raw, distorted features
				double startTime = omp_get_wtime();
				printLOG("Keyframe #%d-->(%d, %d) [%d] ", keyFrameID, cid, reffid, trackRange);
				TrackGlocalLocalCorpusFeatureToNonKeyFrames(Path, cid, keyFrameID, reffid, trackRange, winDim, cvPyrLevel, 1.0, mySfMPara.kfSuccessConsecutiveTrackingRatio*0.5, mySfMPara.kfSuccessRefTrackingRatio*0.5, 300, -1, 1, omp_get_max_threads(), 0);
				printLOG("%.2fs\n", omp_get_wtime() - startTime);
			}

			MergeTrackedCorpusFeaturetoNonKeyframe2(Path, cid, KeyFrameID2LocalFrameID, startF, stopF, increF);
			ForceLocalizeCameraToCorpusDriver(Path, startF, stopF, increF, cid, distortionCorrected, nInlierThresh2, fromKeyFrameTracking);
		}
	}
	else if (mode == 0) //Step 1: sync all sequences with audio
	{
#ifdef _WINDOWS
		int computeSync = atoi(argv[1 + 2]);
		if (computeSync == 1)
		{
			int allpair = atoi(argv[1 + 3]);
			int minSegLength = INT_MAX;
			if (allpair == 0)
			{
				int srcID1 = atoi(argv[1 + 4]), srcID2 = atoi(argv[1 + 5]);
				double fps1 = atof(argv[1 + 6]), fps2 = atof(argv[1 + 7]), minZNCC = atof(argv[1 + 8]);

				double offset = 0.0, ZNCC = 0.0;
				sprintf(Fname, "%s/%d/audio.wav", Path, srcID1);
				sprintf(Fname2, "%s/%d/audio.wav", Path, srcID2);
				if (SynAudio(Fname, Fname2, fps1, fps2, minSegLength, offset, ZNCC, minZNCC) != 0)
					printLOG("Not succeed\n");
				else
					printLOG("Succeed\n");
			}
			else
			{
				double	minZNCC = atof(argv[1 + 5]);
				int nvideos = atoi(argv[1 + 4]), minSegLength = INT_MAX;

				int *camID = new int[nvideos];
				double *fps = new double[nvideos];
				sprintf(Fname, "%s/CamTimingPara.txt", Path); FILE *fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					printLOG("Fail to load %s\n", Fname);
					return 1;
				}
				int cid, count = 0; double framerate, rs_Percent;
				while (fscanf(fp, "%d %lf %lf ", &cid, &framerate, &rs_Percent) != EOF)
				{
					camID[count] = cid, fps[count] = framerate;
					count++;
					printLOG("Found timing parameters for cam %d\n", cid);
				}
				fclose(fp);
				double offset, ZNCC;

				sprintf(Fname, "%s/audioSync.txt", Path); fp = fopen(Fname, "w+");
				for (int jj = 0; jj < nvideos - 1; jj++)
				{
					for (int ii = jj + 1; ii < nvideos; ii++)
					{
						sprintf(Fname, "%s/%d/audio.wav", Path, camID[jj]);
						sprintf(Fname2, "%s/%d/audio.wav", Path, camID[ii]);
						if (SynAudio(Fname, Fname2, 1.0, 1.0, minSegLength, offset, ZNCC, minZNCC) != 0)
						{
							printLOG("Between %d and %d: not succeed\n\n", camID[jj], camID[ii]);
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], 1000.0*UniformNoise(9e9, 9e9));
						}
						else
							fprintf(fp, "%d %d %.4f\n", camID[jj], camID[ii], -offset);
					}
				}
				fclose(fp);

				PrismMST(Path, "audioSync", nvideos);
				AssignOffsetFromMST(Path, "audioSync", nvideos, NULL, fps);

				delete[]camID, delete[]fps;
			}
		}
		else
		{
			int nCams = atoi(argv[1 + 3]), startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]);
			vector<double> fps;
			vector<int> sCams, TimeStamp;
			for (int ii = 0; ii < nCams; ii++)
				sCams.push_back(ii), fps.push_back(1);

			char *SyncFileName = argv[6 + 1];
			//ShowSeqAlignLoadImages(Path, SyncFileName, startF, stopF, 1);

			//ShowSyncLoadVideos(Path, "InitSync", Path, sCams, fps);
			if (argc == 8)
				ShowSyncLoadImages(Path, SyncFileName, Path, sCams, fps, startF, stopF, 25);
			else if (argc == 9)
			{
				char *SavePath = argv[1 + 7];
				ShowSyncLoadImages(Path, SyncFileName, SavePath, sCams, fps, startF, stopF, 25);
			}
			else if (argc == 10)
			{
				char *SavePath = argv[1 + 7];
				double displayAspectRatio = atof(argv[1 + 8]);
				ShowSyncLoadImages(Path, SyncFileName, SavePath, sCams, fps, startF, stopF, 25, displayAspectRatio);
			}
		}
#else
		printLOG("This mode is not supported in Linux\n");
#endif
		return 0;
	}
	else if (mode == 1) //step 2: calibrate camera individually if needed
	{
		char *Path = argv[1 + 2],
			*VideoName = argv[1 + 3];
		int SaveFrameDif = atoi(argv[1 + 4]);//20;
		int nNonBlurIma = 0;
		PickStaticImagesFromVideo(Path, VideoName, SaveFrameDif, 15, .3, 50, nNonBlurIma, true);
		//BlurDetectionDriver(Path, nNonBlurIma, 1920, 1080, 0.1);
	}
	else if (mode == 2)
	{
		sprintf(Fname, "%s/Corpus", Path);

		double SfMdistance = TriangulatePointsFromCorpusCameras(Fname, 0, 2, 2.0);
		printLOG("SfM measured distance: %.3f\nPlease input the physcial distance (mm): ", SfMdistance);
		double Physicaldistance; cin >> Physicaldistance;
		double ratio = Physicaldistance / SfMdistance;
		printLOG("Real2Sfm ratio: %2.f\n", ratio);
		//sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", Path);
		//ReSaveBundleAdjustedNVMResults(Fname, ratio);
		return 0;
	}
	else if (mode == 3) //step 4: generate corpus
	{
		int nviews = atoi(argv[1 + 2]), nViewsPlus = atoi(argv[1 + 3]), module = atoi(argv[1 + 4]);

		sprintf(Fname, "%s/Corpus", Path);
		if (module == 0)
		{
			int fixIntrinsic = atoi(argv[6]), fixDistortion = atoi(argv[7]), fixPose = atoi(argv[8]), fixSkew = atoi(argv[9]), fixPrism = atoi(argv[10]),
				distortionCorrected = atoi(argv[11]), ShutterModel = atoi(argv[12]), LossType = atoi(argv[13]), doubleRefinement = atoi(argv[14]);
			double threshold = atof(argv[15]);

			printLOG("Refine corpus without retriangulation\n");
			RefineVisualSfMAndCreateCorpus(Fname, nviews, ShutterModel, threshold, fixIntrinsic, fixDistortion, fixPose, 1, fixSkew, fixPrism, distortionCorrected, nViewsPlus, LossType, doubleRefinement);

			double SfMdistance = 1.544;// TriangulatePointsFromCorpusCameras(Fname, 0, 2, 2.0);
			printLOG("SfM measured distance: %.3f\nPlease input the physcial distance (mm): ", SfMdistance);
			double Physicaldistance; //cin >> Physicaldistance;
			Physicaldistance = 900.0;
			double ratio = Physicaldistance / SfMdistance;
			sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", Path);
			ReSaveBundleAdjustedNVMResults(Fname, ratio);

			vector<int> sCams;
			for (int ii = 0; ii < nviews; ii++)
				sCams.push_back(nviews);
			visualizationDriver(Path, sCams, -1, -1, 1, true, false, false, false, false, 25, -1, true, ShutterModel);
		}
		else
		{
			if (module == 1)
			{
				int distortionCorrected = atoi(argv[1 + 5]), HistogramEqual = atoi(argv[1 + 6]), OulierRemoveTestMethod = atoi(argv[1 + 7]), //Fmat is more prefered. USAC is much faster then 5-pts :(
					LensType = atoi(argv[1 + 8]);
				double ratioThresh = atof(argv[1 + 9]);

				int nCams = nviews, cameraToScan = -1;
				if (OulierRemoveTestMethod == 2)
					nCams = atoi(argv[1 + 10]), cameraToScan = atoi(argv[1 + 11]);

				/*int distortionCorrected = 0, HistogramEqual = 0, OulierRemoveTestMethod = 2, LensType = 0;
				double ratioThresh = 0.7;

				int nCams = nviews, cameraToScan = -1;
				if (OulierRemoveTestMethod == 2)
				nCams = 1, cameraToScan = 0;*/

				int timeID = -1, ninlierThesh = 50;
				//GeneratePointsCorrespondenceMatrix_SiftGPU(Fname, nviews, -1, HistogramEqual, ratioThresh);
				//GeneratePointsCorrespondenceMatrix_CPU(Fname, nviews, -1, HistogramEqual, ratioThresh, NULL, 1);

				//omp_set_num_threads(omp_get_max_threads());
				//#pragma omp parallel for schedule(dynamic,1)
				for (int jj = 0; jj < nviews - 1; jj++)
				{
					for (int ii = jj + 1; ii < nviews; ii++)
					{
						if (OulierRemoveTestMethod == 1)
							EssentialMatOutliersRemove(Fname, timeID, jj, ii, nCams, cameraToScan, ninlierThesh, distortionCorrected, false);
						else if (OulierRemoveTestMethod == 2)
							FundamentalMatOutliersRemove(Fname, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nCams, cameraToScan);
					}
				}
				return 0;
				GenerateMatchingTable(Fname, nviews, timeID);
			}
			else
			{
				int distortionCorrected = atoi(argv[1 + 5]), ShutterModel = atoi(argv[1 + 6]), sharedIntrinsic = atoi(argv[1 + 7]), LossType = atoi(argv[1 + 8]); //0: NULL, 1: Huber
																																								  //int distortionCorrected = 0, ShutterModel = 0, sharedIntrinsic = 1, LossType = 0; //0: NULL, 1: Huber

				BuildCorpus(Fname, distortionCorrected, ShutterModel, sharedIntrinsic, nViewsPlus, LossType);
				vector<int> sCams;
				visualizationDriver(Path, sCams, -1, -1, 1, true, false, false, false, false, 25, -1, true, ShutterModel);
			}
		}

		return 0;
	}
	else if (mode == 4) //Step 5: Get features points
	{
		int module = atoi(argv[1 + 2]);
		if (module <= 1) //for corpus
		{
			int startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]), increF = atoi(argv[1 + 5]), HistogramEqual = atoi(argv[1 + 6]);

			vector<int> dummy;
			if (module == 0)
			{
				int Covdet = atoi(argv[1 + 7]);

				int nthreads = omp_get_max_threads();
				int segLength = max((stopF - startF + 1) / nthreads, 1);
				vector<Point2i> seg;
				int segcount = 0;
				while (true)
				{
					seg.push_back(Point2i(startF + segLength * segcount, startF + segLength * (segcount + 1) - 1));
					if (startF + segLength * (segcount + 1) - 1 >= stopF)
						break;
					segcount++;
				}
				sprintf(Fname, "%s/Corpus", Path);

				omp_set_num_threads(nthreads);
#pragma omp parallel for
				for (int ii = 0; ii < (int)seg.size(); ii++)
					ExtractSiftCPU(Path, dummy, seg[ii].x, seg[ii].y, increF, HistogramEqual, Covdet);
			}
			else
				ExtractSiftGPU(Path, dummy, startF, stopF, increF, HistogramEqual);
		}
		else //for test sequence
		{
			int  selectedView = atoi(argv[1 + 3]), nviews = atoi(argv[1 + 4]), startF = atoi(argv[1 + 5]), stopF = atoi(argv[1 + 6]), increF = atoi(argv[1 + 7]), HistogramEqual = atoi(argv[1 + 8]);
			//int nviews = 1, selectedView = 0, startF = 0, stopF = 25, increF = 1, HistogramEqual = 0;

			vector<int> availViews;
			if (selectedView < 0)
				for (int ii = 0; ii < nviews; ii++)
					availViews.push_back(ii);
			else
				availViews.push_back(selectedView);

			if (module == 2) //vlfeat :SIFT || COVDET
			{
				int featureType = atoi(argv[1 + 9]); //1: COVDET, 2: SIFT

				int nthreads = omp_get_max_threads();
				int segLength = max((stopF - startF + 1) / nthreads, 1);
				vector<Point2i> seg;
				int segcount = 0;
				while (true)
				{
					seg.push_back(Point2i(startF + segLength * segcount, startF + segLength * (segcount + 1) - 1));
					seg.back().x = seg.back().x / increF * increF;
					seg.back().y = seg.back().y / increF * increF;
					if (startF + segLength * (segcount + 1) - 1 >= stopF)
						break;
					segcount++;
				}

				omp_set_num_threads(nthreads);
#pragma omp parallel for
				for (int ii = 0; ii < (int)seg.size(); ii++)
					ExtractSiftCPU(Path, availViews, seg[ii].x, min(seg[ii].y, stopF), increF, HistogramEqual, featureType);
			}
			if (module == 3) //SIFTGPU
				ExtractSiftGPU(Path, availViews, startF, stopF, increF, HistogramEqual);
			if (module == 4)
				ExtractSiftGPU_Video(Path, availViews, startF, stopF, increF);
		}
		return 0;
	}
	else if (mode == 5) //Step 6: Localize test sequence wrst to corpus
	{
		int module = atoi(argv[1 + 2]); //0: Matching, 1: Localize, 2+3+4: refine on video, -2: visualize
		if (module == -1)
		{
			module = atoi(argv[1 + 3]);
			BuildCorpusAndLocalizeCameraBatch(Path, mySfMPara, module);
		}
		else if (module == -2)
		{
			printLOG("Displaying\n");
			vector<int> sCams;
			for (int ii = 0; ii < mySfMPara.nCams; ii++)
				sCams.push_back(ii);
			visualizationDriver(Path, sCams, mySfMPara.startF, mySfMPara.stopF, 1, true, false, false, true, false, mySfMPara.SkeletonPointFormat, mySfMPara.startF, true, mySfMPara.ShutterModel2);
		}
		else if (module == 0 || module == 1)
		{
			int selectedCam = atoi(argv[1 + 3]), startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF, distortionCorrected = mySfMPara.distortionCorrected, nInlierThresh = mySfMPara.nInliersThresh;
			if (argc > 5)
				startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]), distortionCorrected = atoi(argv[1 + 7]), nInlierThresh = atoi(argv[1 + 8]);
			LocalizeCameraToCorpusDriver(Path, startF, stopF, increF, module, selectedCam, distortionCorrected, nInlierThresh, 0);

			if (module == 0)
				sprintf(Fname, "%s/Logs/CamLocalizeMatch_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF);
			else
				sprintf(Fname, "%s/Logs/CamLocalizePnP_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF);
			FILE *fp = fopen(Fname, "w+");	fclose(fp);
		}
		else if (module == 2)
		{
			int selectedCam = atoi(argv[1 + 3]), MatchingMode = mySfMPara.MatchingMode;
			if (argc > 5)
				MatchingMode = atoi(argv[1 + 4]);
			sprintf(Fname, "%s/%d/V.db", Path, selectedCam);
			//if (IsFileExist(Fname) == 0)
			{
#ifdef _WINDOWS
				sprintf(Fname, "%s/feature_importer.exe --database_path %s/%d/V.db --image_path %s/%d --import_path %s/%d/ --image_list %s/%d/querry.txt", buffer, Path, selectedCam, Path, selectedCam, Path, selectedCam, Path, selectedCam);
#else
				sprintf(Fname, "./feature_importer --database_path %s/%d/V.db --image_path %s/%d --import_path %s/%d/ --image_list %s/%d/querry.txt", Path, selectedCam, Path, selectedCam, Path, selectedCam, Path, selectedCam);
#endif
				printLOG("%s\n", Fname); system(Fname);
			}

			sprintf(Fname, "%s/%d/instreamMatches.txt", Path, selectedCam);
			//if (IsFileExist(Fname) == 0)
			{
#ifdef _WINDOWS
				if (MatchingMode == 0 || MatchingMode == 1 || MatchingMode == 3)//use pretrained vocabDB
					sprintf(Fname, "%s/vocab_tree_matcher.exe --database_path %s/%d/V.db --SiftMatching.gpu_index=0 --VocabTreeMatching.num_images 40 --VocabTreeMatching.vocab_tree_path %s --VocabTreeMatching.match_list_path %s/%d/querry.txt", buffer, Path, selectedCam, mySfMPara.VocabTreePath, Path, selectedCam);
				else//use scene specific vocabDB
					sprintf(Fname, "%s/vocab_tree_matcher.exe --database_path %s/%d/V.db --SiftMatching.gpu_index=0 --VocabTreeMatching.num_images 40 --VocabTreeMatching.vocab_tree_path %s/Corpus/vocabTree.db --VocabTreeMatching.match_list_path %s/%d/querry.txt", buffer, Path, selectedCam, Path, Path, selectedCam);
				printLOG("%s\n", Fname); system(Fname);
#else
				if (MatchingMode == 0 || MatchingMode == 1 || MatchingMode == 3)//use pretrained vocabDB
					sprintf(Fname, "./vocab_tree_matcher --database_path %s/%d/V.db --SiftMatching.gpu_index=0 --VocabTreeMatching.num_images 40 --VocabTreeMatching.vocab_tree_path %s --VocabTreeMatching.match_list_path %s/%d/querry.txt", Path, selectedCam, mySfMPara.VocabTreePath, Path, selectedCam);
				else//use scene specific vocabDB
					sprintf(Fname, "./vocab_tree_matcher --database_path %s/%d/V.db --SiftMatching.gpu_index=0 --VocabTreeMatching.num_images 40 --VocabTreeMatching.vocab_tree_path %s/Corpus/vocabTree.db --VocabTreeMatching.match_list_path %s/%d/querry.txt", Path, selectedCam, Path, Path, selectedCam);
				printLOG("%s\n", Fname); system(Fname);
#endif

#ifdef _WINDOWS
				sprintf(Fname, "python %s/export_inlier_matches.py --database_path %s/%d/V.db --output_path %s/%d/instreamMatches.txt --min_num_matches %d", buffer, Path, selectedCam, Path, selectedCam, 30);
#else
				sprintf(Fname, "python export_inlier_matches.py --database_path %s/%d/V.db --output_path %s/%d/instreamMatches.txt --min_num_matches %d", Path, selectedCam, Path, selectedCam, 30);
#endif
				printLOG("%s\n", Fname); system(Fname);
			}
		}
		else if (module == 3)
		{
			int selectedCamId = atoi(argv[1 + 3]);
			if (argc > 5)
				mySfMPara.startF = atoi(argv[1 + 4]), mySfMPara.stopF = atoi(argv[1 + 5]), mySfMPara.increF = atoi(argv[1 + 6]);

			VideoKeyframe2Corpus_SFM(Path, selectedCamId, mySfMPara);

			char Fname1[512], Fname2[512];
			sprintf(Fname, "%s/kfPose", Path), makeDir(Fname);
			sprintf(Fname1, "%s/Intrinsic_%.4d.txt", Path, selectedCamId);
			sprintf(Fname2, "%s/kfPose/Intrinsic_%.4d.txt", Path, selectedCamId);
			MyCopyFile(Fname1, Fname2);
			sprintf(Fname1, "%s/CamPose_%.4d.txt", Path, selectedCamId);
			sprintf(Fname2, "%s/kfPose/CamPose_%.4d.txt", Path, selectedCamId);
			MyCopyFile(Fname1, Fname2);

			sprintf(Fname, "%s/Logs/VideoKeyframe2Corpus_SFM_%d_%d_%.4d.txt", Path, selectedCamId, mySfMPara.startF, mySfMPara.stopF);
			FILE*fp = fopen(Fname, "w"); fclose(fp);
		}
		else if (module == 4)
		{
			int selectedCamId = atoi(argv[1 + 3]);
			if (argc > 5)
				mySfMPara.startF = atoi(argv[1 + 4]), mySfMPara.stopF = atoi(argv[1 + 5]), mySfMPara.increF = atoi(argv[1 + 6]);

			sprintf(Fname, "%s/Logs/TrackCorpusPointsFromKeyFrames_%d_%d_%.4d.txt", Path, selectedCamId, mySfMPara.startF, mySfMPara.stopF);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/%d/PnPf", Path, selectedCamId), makeDir(Fname);
				sprintf(Fname, "%s/%d/PnPTc", Path, selectedCamId), makeDir(Fname);
				sprintf(Fname, "%s/%d/PnPmTc", Path, selectedCamId), makeDir(Fname);

				//identify keyframe
				int  dummy, rfid;
				vector<int> KeyFrameID2LocalFrameID;
				sprintf(Fname, "%s/%d/Frame2Corpus.txt", Path, selectedCamId);
				if (!IsFileExist(Fname))
					return 1;
				FILE *fp = fopen(Fname, "r");
				while (fscanf(fp, "%d %d %d %d ", &dummy, &dummy, &rfid, &dummy) != EOF)
					KeyFrameID2LocalFrameID.push_back(rfid);
				fclose(fp);

				VideoData VideoI;
				if (ReadVideoDataI(Path, VideoI, selectedCamId, mySfMPara.startF, mySfMPara.stopF) == 1)
					return 1;

				convertPnP2KPnP(Path, selectedCamId, mySfMPara.startF, mySfMPara.stopF);

				vector<int> ValidKeyFrames;  ValidKeyFrames.reserve(KeyFrameID2LocalFrameID.size());
				for (size_t keyFrameID = 0; keyFrameID < KeyFrameID2LocalFrameID.size(); keyFrameID++)
				{
					sprintf(Fname, "%s/%d/PnP/KF_Inliers_%.4d.txt", Path, selectedCamId, KeyFrameID2LocalFrameID[keyFrameID]);
					if (IsFileExist(Fname) == 1 && VideoI.VideoInfo[KeyFrameID2LocalFrameID[keyFrameID]].valid == 1)
						ValidKeyFrames.push_back(1);
					else
						ValidKeyFrames.push_back(0);
				}

				for (size_t keyFrameID = 0; keyFrameID < KeyFrameID2LocalFrameID.size(); keyFrameID++)
				{
					int winDim = mySfMPara.trackingWinSize, cvPyrLevel = mySfMPara.cvPyrLevel, trackRange;
					int reffid = KeyFrameID2LocalFrameID[keyFrameID];

					if (keyFrameID == 0)
					{
						int nextValidFrame = keyFrameID + 2;
						for (int ii = nextValidFrame; ii < KeyFrameID2LocalFrameID.size(); ii++)
						{
							if (ValidKeyFrames[ii] == 1)
							{
								nextValidFrame = ii;
								break;
							}
						}
						trackRange = (KeyFrameID2LocalFrameID[nextValidFrame] - KeyFrameID2LocalFrameID[keyFrameID]);
					}
					else if (keyFrameID == KeyFrameID2LocalFrameID.size() - 1)
					{
						int nextValidFrame = keyFrameID - 2;
						for (int ii = nextValidFrame; ii >= 0; ii--)
						{
							if (ValidKeyFrames[ii] == 1)
							{
								nextValidFrame = ii;
								break;
							}
						}
						trackRange = (KeyFrameID2LocalFrameID[keyFrameID] - KeyFrameID2LocalFrameID[nextValidFrame]);
					}
					else if (keyFrameID > 1 && keyFrameID < KeyFrameID2LocalFrameID.size() - 2)
					{
						int nextValidFrame1 = keyFrameID + 2;
						for (int ii = nextValidFrame1; ii < KeyFrameID2LocalFrameID.size(); ii++)
						{
							if (ValidKeyFrames[ii] == 1)
							{
								nextValidFrame1 = ii;
								break;
							}
						}
						int nextValidFrame2 = keyFrameID - 2;
						for (int ii = nextValidFrame2; ii >= 0; ii--)
						{
							if (ValidKeyFrames[ii] == 1)
							{
								nextValidFrame2 = ii;
								break;
							}
						}
						trackRange = max(KeyFrameID2LocalFrameID[nextValidFrame1] - KeyFrameID2LocalFrameID[keyFrameID], KeyFrameID2LocalFrameID[keyFrameID] - KeyFrameID2LocalFrameID[nextValidFrame2]);
					}
					else if (keyFrameID > 0 && keyFrameID < KeyFrameID2LocalFrameID.size() - 1)
					{
						int nextValidFrame1 = keyFrameID + 1;
						for (int ii = nextValidFrame1; ii < KeyFrameID2LocalFrameID.size(); ii++)
						{
							if (ValidKeyFrames[ii] == 1)
							{
								nextValidFrame1 = ii;
								break;
							}
						}
						int nextValidFrame2 = keyFrameID - 1;
						for (int ii = nextValidFrame2; ii >= 0; ii--)
						{
							if (ValidKeyFrames[ii] == 1)
							{
								nextValidFrame2 = ii;
								break;
							}
						}
						trackRange = 2 * max(KeyFrameID2LocalFrameID[nextValidFrame1] - KeyFrameID2LocalFrameID[keyFrameID], KeyFrameID2LocalFrameID[keyFrameID] - KeyFrameID2LocalFrameID[nextValidFrame2]);
					}
					//this function produces raw, distorted features
					double startTime = omp_get_wtime();
					printLOG("Keyframe #%d-->(%d, %d) [%d] ", keyFrameID, selectedCamId, reffid, trackRange);
					TrackGlocalLocalCorpusFeatureToNonKeyFrames(Path, selectedCamId, keyFrameID, reffid, trackRange, winDim, cvPyrLevel, 1.0, mySfMPara.kfSuccessConsecutiveTrackingRatio*0.5,
						mySfMPara.kfSuccessRefTrackingRatio*0.5, 300, -1, mySfMPara.highQualityTracking, omp_get_max_threads(), 0);
					printLOG("%.2fs\n", omp_get_wtime() - startTime);
				}

				MergeTrackedCorpusFeaturetoNonKeyframe2(Path, selectedCamId, KeyFrameID2LocalFrameID, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF);
				ForceLocalizeCameraToCorpusDriver(Path, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, selectedCamId, mySfMPara.distortionCorrected, mySfMPara.nInliersThresh2, mySfMPara.fromKeyFrameTracking);

				sprintf(Fname, "%s/Logs/TrackCorpusPointsFromKeyFrames_%d_%d_%.4d.txt", Path, selectedCamId, mySfMPara.startF, mySfMPara.stopF); fp = fopen(Fname, "w"); fclose(fp);
			}
		}
		else if (module == 5)
		{
			int selectedCam = atoi(argv[1 + 3]), startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF, distortionCorrected = mySfMPara.distortionCorrected, fromKeyFrameTracking = mySfMPara.fromKeyFrameTracking;
			if (argc > 5)
				startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]), distortionCorrected = atoi(argv[1 + 7]), fromKeyFrameTracking = atoi(argv[1 + 8]);
			ForceLocalizeCameraToCorpusDriver(Path, startF, stopF, increF, selectedCam, distortionCorrected, mySfMPara.nInliersThresh2, fromKeyFrameTracking);

			sprintf(Fname, "%s/Logs/ForcePnP_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF); FILE *fp = fopen(Fname, "w+");	fclose(fp);
		}
		else if (module == 6)
		{
			int selectedCam = atoi(argv[1 + 3]), startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF,
				fixIntrinsic = mySfMPara.fixIntrinsic, fixDistortion = mySfMPara.fixDistortion, fix3D = mySfMPara.fix3D, fixLocal3D = mySfMPara.fixLocal3D, fixPose = mySfMPara.fixPose,
				fixSkew = mySfMPara.fixSkew, fixPrism = mySfMPara.fixPrism, distortionCorrected = mySfMPara.distortionCorrected,
				nViewsPlus = mySfMPara.nViewsPlusBA2, ShutterModel = mySfMPara.ShutterModel2, LossType = mySfMPara.LossType, doubleRefinement = mySfMPara.BARefinementIter;
			double threshold = mySfMPara.reProjectionBAThresh;

			if (argc > 5)
				startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]),
				fixIntrinsic = atoi(argv[1 + 7]), fixDistortion = atoi(argv[1 + 8]), fix3D = atoi(argv[1 + 9]), fixLocal3D = atoi(argv[1 + 10]), fixPose = atoi(argv[1 + 11]),
				fixSkew = atoi(argv[1 + 12]), fixPrism = atoi(argv[1 + 13]), distortionCorrected = atoi(argv[1 + 14]),
				nViewsPlus = atoi(argv[1 + 15]), ShutterModel = atoi(argv[1 + 16]), LossType = atoi(argv[1 + 17]), doubleRefinement = atoi(argv[1 + 18]),
				threshold = atof(argv[1 + 19]);

			sprintf(Fname, "%s/Logs/PerCamBA_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF);
			if (IsFileExist(Fname) == 0)
			{
				PerVideo_BA(Path, selectedCam, startF, stopF, increF, fixIntrinsic, fixDistortion, fix3D, fixLocal3D, fixPose, fixSkew, fixPrism, distortionCorrected, nViewsPlus, ShutterModel, LossType, doubleRefinement, threshold);

				/*int orgIncreFrame = increF, orgfixed3D = fix3D;
				increF *= 3;
				PerVideo_BA(Path, selectedCam, startF, stopF, increF, fixIntrinsic, fixDistortion, fix3D, fixLocal3D, fixPose, fixSkew, fixPrism, distortionCorrected, nViewsPlus, ShutterModel, LossType, doubleRefinement, threshold);

				fix3D = 1, fixLocal3D = 1;
				PerVideo_BA(Path, selectedCam, startF + 1, stopF, increF, fixIntrinsic, fixDistortion, fix3D, fixLocal3D, fixPose, fixSkew, fixPrism, distortionCorrected, nViewsPlus, ShutterModel, LossType, doubleRefinement, threshold);
				PerVideo_BA(Path, selectedCam, startF + 2, stopF, increF, fixIntrinsic, fixDistortion, fix3D, fixLocal3D, fixPose, fixSkew, fixPrism, distortionCorrected, nViewsPlus, ShutterModel, LossType, doubleRefinement, threshold);

				fix3D = orgfixed3D, increF = orgIncreFrame;*/

				VideoData VideoInfoI;
				ReadVideoDataI(Path, VideoInfoI, selectedCam, startF, stopF);
				sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, selectedCam), remove(Fname);
				sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, selectedCam), remove(Fname);
				WriteVideoDataI(Path, VideoInfoI, selectedCam, startF, stopF, 1);

				sprintf(Fname, "%s/Logs/PerCamBA_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF); FILE *	fp = fopen(Fname, "w+");	fclose(fp);
			}
		}
		else if (module == 7)
		{
			int nCams = mySfMPara.nCams, startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF,
				fixIntrinsic = mySfMPara.fixIntrinsic, fixDistortion = mySfMPara.fixDistortion, fixPose = mySfMPara.fixPose, fix3D = mySfMPara.fix3D, fixSkew = mySfMPara.fixSkew, fixPrism = mySfMPara.fixPrism,
				distortionCorrected = mySfMPara.distortionCorrected, nViewsPlus = mySfMPara.nViewsPlusBA2, ShutterModel = mySfMPara.ShutterModel2, LossType = mySfMPara.LossType, doubleRefinement = mySfMPara.BARefinementIter;
			double threshold = mySfMPara.reProjectionBAThresh;

			if (argc > 4)
				nCams = atoi(argv[1 + 3]), startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]),
				fixIntrinsic = atoi(argv[1 + 7]), fixDistortion = atoi(argv[1 + 8]), fixPose = atoi(argv[1 + 9]), fix3D = atoi(argv[1 + 10]), fixSkew = atoi(argv[1 + 11]), fixPrism = atoi(argv[1 + 12]),
				distortionCorrected = atoi(argv[1 + 13]), nViewsPlus = atoi(argv[1 + 14]), ShutterModel = atoi(argv[1 + 15]), LossType = atoi(argv[1 + 16]), doubleRefinement = atoi(argv[1 + 17]),
				threshold = atof(argv[1 + 18]);

			AllVideo_BA(Path, nCams, startF, stopF, increF, fixIntrinsic, fixDistortion, fixPose, 0, fix3D, fixSkew, fixPrism, distortionCorrected, nViewsPlus, ShutterModel, LossType, doubleRefinement, threshold);
			sprintf(Fname, "%s/Logs/AllCamBA_%d_%.4d.txt", Path, startF, stopF); FILE *	fp = fopen(Fname, "w+");	fclose(fp);
		}
		else if (module == 8)
		{
			int selectedCam = atoi(argv[1 + 3]), startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]), LossType = atoi(argv[1 + 7]);
			Virtual3D_RS_BA_Driver(Path, selectedCam, startF, stopF, increF, LossType);
		}
		else if (module == 9)
		{
			int selectedCam = atoi(argv[1 + 3]), startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]),
				fixIntrinsic = atoi(argv[1 + 7]), fixDistortion = atoi(argv[1 + 8]), fixPose = atoi(argv[1 + 9]), fix3D = atoi(argv[1 + 10]), fixSkew = atoi(argv[1 + 11]), fixPrism = atoi(argv[1 + 12]),
				distortionCorrected = atoi(argv[1 + 13]), controlStep = atoi(argv[1 + 14]);
			double threshold = atof(argv[1 + 15]);

			//double threshold = 5.0;
			//int  fixIntrinsic = 0, fixDistortion = 0, fix3D = 0, fixSkew = 1, fixPrism = 1, distortionCorrected = 0, controlStep = 3; //Recomemded

			VideoSplineRSBA(Path, startF, stopF, selectedCam, fixIntrinsic, fixDistortion, fix3D, fixSkew, fixPrism, distortionCorrected, threshold, controlStep, 4, 1);
			//Pose_se_BSplineInterpolation("E:/RollingShutter/_CamPoseS_0.txt", "E:/RollingShutter/__CamPose_0.txt", 100, "E:/RollingShutter/Pose.txt");
		}
		else if (module == 10)
		{
			int useNewPose = argc > 4 ? atoi(argv[1 + 3]) : 0, nCams = mySfMPara.nCams;

			Corpus CorpusInfo;
			sprintf(Fname, "%s/Corpus", Path);
			ReadCorpusInfo(Fname, CorpusInfo, false, true); //also produce twoDIdAllViews

			if (useNewPose)
			{
				printLOG("Reading all camera poses\n");
				VideoData *VideoInfo = new VideoData[nCams];
				for (int cid = 0; cid < nCams; cid++)
				{
					printLOG("Cam %d ...validating ", cid);
					if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
					{
						continue;
					}

					InvalidateAbruptCameraPose(VideoInfo[cid], -1, -1, 0);
					printLOG("\n");
				}

				Point3i Corpus_cid_Lcid_Lfid;
				vector<Point3i> vCorpus_cid_Lcid_Lfid;
				sprintf(Fname, "%s/Corpus/CameraToBuildCorpus3.txt", Path);
				FILE *fp = fopen(Fname, "r");
				while (fscanf(fp, "%d %d %d ", &Corpus_cid_Lcid_Lfid.y, &Corpus_cid_Lcid_Lfid.x, &Corpus_cid_Lcid_Lfid.z) != EOF)
					vCorpus_cid_Lcid_Lfid.push_back(Corpus_cid_Lcid_Lfid);
				fclose(fp);

				for (int ii = 0; ii < (int)vCorpus_cid_Lcid_Lfid.size(); ii++)
				{
					int cid = vCorpus_cid_Lcid_Lfid[ii].x, ccid = vCorpus_cid_Lcid_Lfid[ii].y, fid = vCorpus_cid_Lcid_Lfid[ii].z;

					CorpusInfo.camera[ccid].valid = false;
					if (VideoInfo[cid].VideoInfo[fid].valid)
					{
						CopyCamereInfo(VideoInfo[cid].VideoInfo[fid], CorpusInfo.camera[ccid], true);
						CorpusInfo.camera[ccid].valid = true;
					}
				}
			}

			sprintf(Fname, "%s/Corpus", Path);
			writeColMap4DenseStereo(Fname, CorpusInfo); //need twoDIdAllViews
		}
		else if (module == 11)
		{
			int nCams = atoi(argv[1 + 3]), startF = atoi(argv[1 + 4]), stopF = atoi(argv[1 + 5]), distortionCorredted = atoi(argv[1 + 6]), nInlierThresh = atoi(argv[1 + 7]);

			for (int selectedCam = 0; selectedCam < nCams; selectedCam++)
				TestPnPf(Path, selectedCam, startF, stopF, distortionCorredted, nInlierThresh);
			//TestKSfM(Path, ii, 0, 30);
		}

		return 0;
	}
	else if (mode == 6) //step 7: generate 3d data from test sequences
	{
		int startF = atoi(argv[1 + 2]), stopF = atoi(argv[1 + 3]), increF = atoi(argv[1 + 4]), module = atoi(argv[1 + 5]);// 0: matching, 2 : triangulation
																														  //int startF = 200, stopF = 2000, increF = 4, module = 2;// 0: matching, 2 : triangulation

		int HistogrameEqual = 0, distortionCorrected = 0, OulierRemoveTestMethod = 2, ninlierThesh = 40, //fmat test
			LensType = RADIAL_TANGENTIAL_PRISM, nViewsPlus = 2, nviews = 12;
		double ratioThresh = 0.7, reprojectionThreshold = 5;

		vector<int> frameTimeStamp(nviews, 0); //input is time delay format
		if (module == 0)
		{
			for (int timeID = startF; timeID <= stopF; timeID += increF)
			{
				GeneratePointsCorrespondenceMatrix_SiftGPU(Path, nviews, timeID, HistogrameEqual, 0.6f, &frameTimeStamp[0]);
#pragma omp parallel
				{
#pragma omp for nowait
					for (int jj = 0; jj < nviews - 1; jj++)
					{
						for (int ii = jj + 1; ii < nviews; ii++)
						{
							if (OulierRemoveTestMethod == 1)
								EssentialMatOutliersRemove(Path, timeID, jj, ii, nviews, 0, ninlierThesh, distortionCorrected, false);
							if (OulierRemoveTestMethod == 2)
								FundamentalMatOutliersRemove(Path, timeID, jj, ii, ninlierThesh, LensType, distortionCorrected, false, nviews, -1, &frameTimeStamp[0]);
						}
					}
				}
				GenerateMatchingTable(Path, nviews, timeID, nViewsPlus);
			}
		}
		else if (module == 1) //Get the matched points
			;/// GetPutativeMatchesForEachView(Path, nviews, startF, stopF, increF, Point2d(2.0, 7.5), nViewsPlus, frameTimeStamp);
		else
		{
			bool SaveToDenseColMap = true, save2DCorres = false;
			double MaxDepthThresh = 99000; //mm
			Build3DFromSyncedImages(Path, nviews, startF, stopF, increF, LensType, distortionCorrected, nViewsPlus, reprojectionThreshold, MaxDepthThresh, &frameTimeStamp[0], SaveToDenseColMap, save2DCorres, false, 10.0, false);
			//vector<int> sCams;
			//for (int ii = 0; ii < nviews; ii++)
			//	sCams.push_back(ii);
			//visualizationDriver(Path, sCams, 0, 600, 1, true, false, false, true, false, 18, startF, true, true);
		}

		return 0;
	}
	else if (mode == 7) //stat-dyna-ST-BA
	{
		int STCalibration, ParallelMode;
		if (argc == 3)
			STCalibration = 1, ParallelMode = 0;
		else
			STCalibration = atoi(argv[1 + 2]), ParallelMode = atoi(argv[1 + 3]);

		int nCams, startF, stopF, SearchRange, nDyna;
		double lamdaData, SearchStep, RealOverSfm;
		char Fname[512]; 	sprintf(Fname, "%s/Parameter.txt", Path); FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%d %d %d %d %lf %d %lf %lf", &nCams, &startF, &stopF, &nDyna, &lamdaData, &SearchRange, &SearchStep, &RealOverSfm);
		fclose(fp);

		int width = 1920, height = 1080, LossType = 0, PriorOrder = 1;
		int fixIntrinsic = 1, fixDistortion = 1, fixPose = 0, fixfirstCamPose = 1, distortionCorrected = 0;

		if (ParallelMode == 0)
		{
			printLOG("Master process\n");
			;// TestLeastActionOnSyntheticData(Path, 1, nDyna);
		}
		else if (ParallelMode == 1)
		{
			int camID1 = atoi(argv[1 + 4]), camID2 = atoi(argv[1 + 5]);
			EvaluateAllPairSTCostParallel(Path, camID1, camID2, nCams, nDyna, startF, stopF, SearchRange, SearchStep, lamdaData, RealOverSfm, 2);
		}
		else if (ParallelMode == 2)
		{
			int off1 = atoi(argv[1 + 10]), off2 = atoi(argv[1 + 11]);
			vector<int> SelectedCamera; SelectedCamera.push_back(atoi(argv[1 + 4])); SelectedCamera.push_back(atoi(argv[1 + 5])); SelectedCamera.push_back(atoi(argv[1 + 6]));
			vector<double>TimeStampInfoVector; TimeStampInfoVector.push_back(atof(argv[1 + 7])); TimeStampInfoVector.push_back(atof(argv[1 + 8])); TimeStampInfoVector.push_back(atof(argv[1 + 9]));

			//int off1 = -5, off2 = -5;
			//vector<int> SelectedCamera; SelectedCamera.push_back(0); SelectedCamera.push_back(3); SelectedCamera.push_back(2);
			//vector<double>TimeStampInfoVector; TimeStampInfoVector.push_back(0); TimeStampInfoVector.push_back(0.5); TimeStampInfoVector.push_back(2.9);
			MotionPriorSyncBruteForce2DTripletParallel(Path, SelectedCamera, startF, stopF, nDyna, TimeStampInfoVector, off1, off2, SearchStep, lamdaData, RealOverSfm, 2);
		}
		else if (ParallelMode == 3)
		{
			int SearchOffset = atoi(argv[1 + 4]), cid = atoi(argv[1 + 5]), nselectedCams = atoi(argv[1 + 6]);
			vector<int> SelectedCamera(nselectedCams);
			vector<double> OffsetInfo(nselectedCams);

			sprintf(Fname, "%s/IncreMotionSync_%d_%d_%.4d.txt", Path, nselectedCams, cid, SearchOffset); FILE *fp = fopen(Fname, "r");
			for (int ii = 0; ii < nselectedCams; ii++)
				fscanf(fp, "%d %lf ", &SelectedCamera[ii], &OffsetInfo[ii]);
			fclose(fp);

			IncrementalMotionPriorSyncDiscreteContinous2DParallel(Path, SelectedCamera, startF, stopF, nDyna, OffsetInfo, SearchOffset, cid, lamdaData, RealOverSfm);

			/*/printLOG("\nInitial offsets: ");
			for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
			printLOG("%.6f ", OffsetInfo[ii]);
			printLOG("\n");

			double CeresCost;
			IncrementalMotionPriorSyncDiscreteContinous2D(Path, SelectedCamera, startF, stopF, nDyna, OffsetInfo, 0, 0, 0, lamdaData, RealOverSfm, CeresCost, false);

			printLOG("T3: ");
			for (int ii = 0; ii < (int)SelectedCamera.size(); ii++)
			printLOG("%.6f ", OffsetInfo[ii]);
			printLOG("\n");

			sprintf(Fname, "%s/IncreST2_%d_%d_%.4d.txt", Path, nCams, cid, SearchOffset); fp = fopen(Fname, "w+");
			fprintf(fp, "%.16e %.16e %.16e %.16e %.16e ", CeresCost, 0.0, 0.0, 0.0, 0.0);
			for (int jj = 0; jj < nCams; jj++)
			fprintf(fp, "%.8f ", OffsetInfo[jj]);
			fclose(fp);*/
		}
		return 0;
	}
	else if (mode == 8)
	{
		int nVideoCams = 5,
			startF = 0, stopF = 199,
			LensModel = RADIAL_TANGENTIAL_PRISM;

		int selectedCam[] = { 1, 2 }, selectedTime[2] = { 0, 0 }, ChooseCorpusView1 = -1, ChooseCorpusView2 = -1;

		VideoData AllVideoInfo;
		ReadVideoData(Path, AllVideoInfo, nVideoCams, startF, stopF);

		double Fmat[9];
		int seletectedIDs[2] = { selectedCam[0] * max(MaxnFrames, stopF) + selectedTime[0], selectedCam[1] * max(MaxnFrames, stopF) + selectedTime[1] };
		computeFmatfromKRT(AllVideoInfo.VideoInfo, 5, seletectedIDs, Fmat);
		return 0;
	}
	else if (mode == 9)
	{
		int selectedCam = atoi(argv[1 + 2]), startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]), increF = atoi(argv[1 + 5]),
			TrackStep = atoi(argv[1 + 6]), TrackRange = atoi(argv[1 + 7]), nWins = atoi(argv[1 + 8]), WinStep = atoi(argv[1 + 9]),
			cvPyrLevel = atoi(argv[1 + 10]), CameraNotCalibrated = atoi(argv[1 + 12]), newDistortionCorrected = atoi(argv[1 + 13]), interpAlgo = atoi(argv[1 + 14]);
		double meanSSGThresh = atof(argv[1 + 11]);

		//double meanSSGThresh = 400.0;
		//int selectedCam = 0, startF = 410, stopF = 410, increF = 1, TrackStep = 1, TrackRange = 15, nWins = 2, WinStep = 3, cvPyrLevel = 3, newDistortionCorrected = 0;

		sprintf(Fname, "%s/Logs/TrackCorpus_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF);
		if (IsFileExist(Fname) == 1)
			return 0;

		for (int fid = startF; fid <= stopF; fid += increF * TrackStep)
			TrackAllCorpusPointsWithRefTemplateDriver(Path, selectedCam, fid, increF, TrackRange, nWins, WinStep, cvPyrLevel, meanSSGThresh, CameraNotCalibrated, newDistortionCorrected, interpAlgo);
		//VisualizeTracking(Path, 0, 0, increF, 1, 30, 1000, 1);
		sprintf(Fname, "%s/Logs/TrackCorpus_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF); FILE *fp = fopen(Fname, "w+"); fclose(fp);

	}
	else if (mode == 10)
	{
		//NOTE: SETTING FPS TO BE SLOWER THAN ITs ACTUALLY VALUE ALLOWS THE TRACKER TO UPDATE IS TEMPLATE MORE OFTEN --> LONGER TRACK
		int selectedCam = atoi(argv[1 + 2]), TrackInstanceStartF = atoi(argv[1 + 3]), TrackInstanceStopF = atoi(argv[1 + 4]), IncreInstanceF = atoi(argv[1 + 5]),
			increF = atoi(argv[1 + 6]), fps = atoi(argv[1 + 7]), TrackTime = atoi(argv[1 + 8]),
			nWins = atoi(argv[1 + 9]), WinStep = atoi(argv[1 + 10]), PyrLevel = atoi(argv[1 + 11]), interpAlgo = atoi(argv[1 + 13]);
		double MeanSSGThresh = atof(argv[1 + 12]);

		//int selectedCam = 9, TrackInstanceStartF = 500, TrackingStopF = 500, IncreInstanceF = 1, increF = 1, fps = 30, TrackTime = 2, nWins = 2, WinStep = 3, PyrLevel = 4;
		//double MeanSSGThresh = 200.0;
		sprintf(Fname, "%s/Logs/PMatchTracking_%d_%.4d_%.4d.txt", Path, selectedCam, TrackInstanceStartF, TrackInstanceStopF);
		if (IsFileExist(Fname) == 1)
			return 0;

		for (int fid = TrackInstanceStartF; fid <= TrackInstanceStopF; fid += IncreInstanceF)
			TrackAllPointsWithRefTemplateDriver(Path, selectedCam, fid, increF, fps, TrackTime, nWins, WinStep, PyrLevel, MeanSSGThresh, interpAlgo);

		sprintf(Fname, "%s/Logs/PMatchTracking_%d_%.4d_%.4d.txt", Path, selectedCam, TrackInstanceStartF, TrackInstanceStopF); FILE *fp = fopen(Fname, "w"); fclose(fp);
	}
	else if (mode == 11)
	{
		int selectedCamId = atoi(argv[1 + 2]), TrackInstanceStartF = atoi(argv[1 + 3]), TrackInstanceStopF = atoi(argv[1 + 4]), IncreInstanceF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]),
			TrackRange = atoi(argv[1 + 7]), HarrisDistance = atoi(argv[1 + 8]), winSize = atoi(argv[1 + 9]), PyrLevel = atoi(argv[1 + 10]), nHarrisPartitions = atoi(argv[1 + 11]), maxHarris = atoi(argv[1 + 12]), interpAlgo = atoi(argv[1 + 13]), debug = 0;
		if (argc == 16)
			debug = atoi(argv[1 + 14]);

		//int increF = 1, trackingInstF = 25, HarrisTrackRange = 15 * 8, HarrisDistance = 10, maxHarris = 30000, HarrisTrackingWinSize = 23, PyrLevel = 4, interpAlgo = -1;
		int c, f, frameTimeStamp = 0;
		double r, rate = 1.0;
		sprintf(Fname, "%s/InitSync.txt", Path);
		if (IsFileExist(Fname) == 0)
			printLOG("Cannot load %s. Please initalize sync\n", Fname);
		else
			printLOG("Load %s. Make sure that it is in timestamp format (f = f_ref - timestamp)\n", Fname);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %lf %d ", &c, &r, &f) != EOF)
		{
			if (c == selectedCamId)
			{
				frameTimeStamp = f;
				rate = r;
				break;
			}
		}
		fclose(fp);

		//f_i = (f_ref/fps_ref - offset_i)*fps_i = f_ref*fps_i/fps_ref - offset*fps_i = f_ref*rate - offset_i*rate, //assuming normal fps = 1
		for (int fid = TrackInstanceStartF; fid <= TrackInstanceStopF; fid += IncreInstanceF)
			TrackHarrisPointsWithRefTemplateDriver(Path, selectedCamId, (fid - frameTimeStamp)*rate, TrackRange*rate, increF, HarrisDistance, winSize, PyrLevel, nHarrisPartitions, maxHarris, interpAlgo, debug);

		sprintf(Fname, "%s/Logs/TrackHarris_%d_%d_%.4d.txt", Path, selectedCamId, TrackInstanceStartF, TrackInstanceStopF); fp = fopen(Fname, "w"); fclose(fp);
	}
	else if (mode == 12)
	{
		int selectedCam = atoi(argv[1 + 2]), startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF,
			rotateImage = 0, useJpg = mySfMPara.UseJpg, multiviewSyncMode = 0;
		double resizeFactor = mySfMPara.imgRescale;

		if (argc > 4)
		{
			selectedCam = atoi(argv[1 + 2]), startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]), increF = atoi(argv[1 + 5]),
				rotateImage = atoi(argv[1 + 6]), useJpg = atoi(argv[1 + 7]), multiviewSyncMode = 0;
			resizeFactor = atof(argv[1 + 8]);
			if (argc == 11)
				multiviewSyncMode = atoi(argv[1 + 9]);
		}

		sprintf(Fname, "%s/RotationInfo.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int cid, code;
			while (fscanf(fp, "%d %d", &cid, &code) != EOF)
			{
				if (cid == selectedCam)
				{
					rotateImage = code;
					break;
				}
			}
			fclose(fp);
		}

		int frameTimeStamp = 0; //deactivated internally
		if (multiviewSyncMode == 1)
		{
			int cid, f; double fps;
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %lf %d ", &cid, &fps, &f) != EOF)
			{
				if (cid == selectedCam)
				{
					frameTimeStamp = f;
					break;
				}
			}
			fclose(fp);
		}

		ExtractVideoFrames(Path, selectedCam, startF, stopF, increF, rotateImage, resizeFactor, 3, useJpg, frameTimeStamp);
		sprintf(Fname, "%s/Logs/ImgExtraction_%d_%d_%.4d.txt", Path, selectedCam, startF, stopF); fp = fopen(Fname, "w"); fclose(fp);

		return 0;
	}
	else if (mode == 13)
	{
		int selectedCam = atoi(argv[1 + 2]), startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]);
		//	int selectedCam = 4, startF = 451, stopF =600; //start from frame 2 instead of 1

		TVL1Parameters tvl1arg;
		tvl1arg.lamda = 0.5, tvl1arg.tau = 0.25, tvl1arg.theta = 0.01, tvl1arg.epsilon = 0.005, tvl1arg.iterations = 30, tvl1arg.nscales = 30, tvl1arg.warps = 20;
		int forward = 0, backward = 1, increF = 1;

		TVL1OpticalFlowDriver(Path, selectedCam, startF, stopF, increF, tvl1arg, forward, backward);

		return 0;
	}
	else if (mode == 14)
	{
		int selectedCam = atoi(argv[1 + 2]), startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]);

		int step = 5;
#pragma omp parallel for
		for (int fid = startF; fid > stopF; fid -= step)
		{
			char Fname[512];  sprintf(Fname, "BroxOF.exe %s %d %d %d", Path, selectedCam, fid, fid - step);
			printLOG("%s\n", Fname);
			system(Fname);
		}
		return 0;
#ifdef _WINDOWS
		sprintf(Fname, "BroxOF.exe %s %d %d %d", Path, selectedCam, startF, stopF);
		printLOG("%s\n", Fname);
		system(Fname);
#else
		step = 5;
		if (startF < stopF)
		{
			for (int fid = startF; fid <= stopF; fid += step)
			{
#ifdef EC2
				sprintf(Fname, "qsub -b y -cwd -pe orte 1 ./BroxOF %s %d %d %d", Path, selectedCam, fid, fid + step);
#else
				sprintf(Fname, "./BroxOF %s %d %d %d", Path, selectedCam, fid, fid + step);
#endif
				printLOG("%s\n", Fname);
				system(Fname);
			}
		}
		else
		{
			for (int fid = startF; fid > stopF; fid -= step)
			{
#ifdef EC2
				sprintf(Fname, "qsub -b y -cwd -pe orte 1 ./BroxOF %s %d %d %d", Path, selectedCam, fid, fid - step);
#else
				sprintf(Fname, "./BroxOF %s %d %d %d", Path, selectedCam, fid, fid - step);
#endif
				printLOG("%s\n", Fname);
				system(Fname);
			}
		}
#endif
	}
	else if (mode == 15)
	{
		//NOTE: SET FPS TO BE SLOWER THAN IT ACTUALLY IS ALLOWS THE TRACKER TO UPDATE IS TEMPLATE MORE OFTEN --> LONGER TRACK
		int selectedCam = atoi(argv[1 + 2]), TrackInstanceStartF = atoi(argv[1 + 3]), TrackInstanceStopF = atoi(argv[1 + 4]), IncreTrackingInstanceF = atoi(argv[1 + 5]), increF = atoi(argv[1 + 6]),
			nWins = atoi(argv[1 + 9]), WinStep = atoi(argv[1 + 10]), noTemplateUpdate = atoi(argv[1 + 12]), interpAlgo = atoi(argv[1 + 13]);
		double fps = atof(argv[1 + 7]), TrackTime = atof(argv[1 + 8]), MeanSSGThresh = atof(argv[1 + 11]);

		/*int selectedCam = 2, TrackInstanceStartF = 497, TrackInstanceStopF = 497, IncreTrackingInstanceF = 1, increF = 1;
		double fps = 40, TrackTime = 0.1; int  nWins = 2, WinStep = 3; double MeanSSGThresh = 200.0; int noTemplateUpdate = 1;

		for (int fid = TrackInstanceStartF; fid <= TrackInstanceStopF; fid += IncreTrackingInstanceF)
		{
		TrackAllPointsWithRefTemplate_DenseFlowDriven_Driver(Path, selectedCam, fid, increF, fps, TrackTime, nWins, WinStep, MeanSSGThresh, noTemplateUpdate);
		VisualizeTracking(Path, selectedCam, fid, increF, fps, TrackTime, 0, 0, -1, 1);
		}
		return 0;*/
		sprintf(Fname, "%s/Logs/PMatchTrackingD_%d_%d_%.4d.txt", Path, selectedCam, TrackInstanceStartF, TrackInstanceStopF);
		if (IsFileExist(Fname) == 1)
			return 0;

		for (int fid = TrackInstanceStartF; fid <= TrackInstanceStopF; fid += IncreTrackingInstanceF)
		{
			TrackAllPointsWithRefTemplate_DenseFlowDriven_Driver(Path, selectedCam, fid, increF, fps, TrackTime, nWins, WinStep, MeanSSGThresh, noTemplateUpdate, interpAlgo);
			VisualizeTracking(Path, selectedCam, fid, increF, fps, TrackTime, 0, 0, -1, 1);
		}
		sprintf(Fname, "%s/Logs/PMatchTrackingD_%d_%d_%.4d.txt", Path, selectedCam, TrackInstanceStartF, TrackInstanceStopF); FILE *fp = fopen(Fname, "w"); fclose(fp);
	}
	else if (mode == 16)
	{
		int selectedCamId = atoi(argv[1 + 2]), startInstF = atoi(argv[1 + 3]), stopInstF = atoi(argv[1 + 4]), instFIncreF = atoi(argv[1 + 5]),
			TrackRange = atoi(argv[1 + 6]), increF = atoi(argv[1 + 7]),
			nInlierThesh = atoi(argv[1 + 8]), reProjectThresh = atof(argv[1 + 9]);
		ClassifyPointsFromTriangulationSingleCam(Path, selectedCamId, startInstF, stopInstF, instFIncreF, TrackRange, increF, nInlierThesh, reProjectThresh);

		sprintf(Fname, "%s/Logs/HarrisClassified_%d_%d_%d_%.4d.txt", Path, selectedCamId, startInstF, stopInstF, instFIncreF); FILE *fp = fopen(Fname, "w"); fclose(fp);

		return 0;
	}
	else if (mode == 17)
	{
		int selectedCamId = atoi(argv[1 + 2]), startInstF = atoi(argv[1 + 3]), nInst = atoi(argv[1 + 4]),
			instFIncreF = atoi(argv[1 + 5]), TrackRange = atoi(argv[1 + 6]), increF = atoi(argv[1 + 7]),
			corpusDistortionCorrected = atoi(argv[1 + 8]), fixDistortion = atoi(argv[1 + 9]),
			fixIntrinsic = atoi(argv[1 + 10]), nPlus = atoi(argv[1 + 11]), sharedInstrinsic = atoi(argv[1 + 13]);
		double reProjectThresh = atof(argv[1 + 12]);

		sprintf(Fname, "%s/Logs/LocalBA_%d_%d_%d_%.4d.txt", Path, selectedCamId, startInstF, nInst, instFIncreF);
		if (IsFileExist(Fname) == 1)
			return 0;

		LocalBA_HarrisTracking_And_SiftMatching(Path, selectedCamId, startInstF, nInst, instFIncreF, TrackRange, increF,
			corpusDistortionCorrected, fixIntrinsic, fixDistortion, 0, 1, 1, 1, 0.5, 5, reProjectThresh, sharedInstrinsic);

		sprintf(Fname, "%s/Logs/LocalBA_%d_%d_%d_%.4d.txt", Path, selectedCamId, startInstF, nInst, instFIncreF); FILE *fp = fopen(Fname, "w"); fclose(fp);
	}
	else if (mode == 18)
	{
		int nCams = atoi(argv[1 + 2]), startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]), increF = atoi(argv[1 + 5]);
		double ratioThresh = atof(argv[1 + 6]);

		int *frameTimeStamp = new int[nCams];
		sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s. Please initalize sync\n", Fname);
			for (int ii = 0; ii < nCams; ii++)
				frameTimeStamp[ii] = 0;
		}
		else
		{
			int cid, f; double fps;
			while (fscanf(fp, "%d %lf %d ", &cid, &fps, &f) != EOF)
				frameTimeStamp[cid] = f;
			fclose(fp);
		}
		int HistogrameEqual = 0;

		for (int fid = startF; fid <= stopF; fid += increF)
		{
			GeneratePointsCorrespondenceMatrix_CPU(Path, nCams, fid, HistogrameEqual, ratioThresh, frameTimeStamp, 2);
			//GenerateMatchingTable(Path, nCams, fid);
		}

		sprintf(Fname, "%s/CorrespondenceMatrix_CPU_%d_%d_%.4d.txt", Path, startF, stopF, increF); FILE *fp2 = fopen(Fname, "w"); fclose(fp2);
		delete[]frameTimeStamp;
	}
	else if (mode == 19)
	{
		int startF = atoi(argv[1 + 2]), stopF = atoi(argv[1 + 3]), increF = atoi(argv[1 + 4]);

		int id, maxID = 0, offset;
		vector<int> sCams, TimeStamp;
		sprintf(Fname, "%s/availableCameras.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot open %s\n", Fname);
			return 1;
		}
		while (fscanf(fp, "%d ", &id) != EOF)
			sCams.push_back(id), maxID = max(maxID, id);
		fclose(fp);

		for (int ii = 0; ii < maxID; ii++)
			TimeStamp.push_back(0);

		sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot open %s. Set Timestamp to 0s\n", Fname);
			for (size_t ii = 0; ii < sCams.size(); ii++)
				TimeStamp.push_back(0);
		}
		double fps;
		while (fscanf(fp, "%d %lf %d ", &id, &fps, &offset) != EOF)
		{
			for (size_t ii = 0; ii < sCams.size(); ii++)
				if (sCams[ii] == id)
				{
					TimeStamp[ii] = offset;
					break;
				}
		}
		fclose(fp);

		MultiviewPeopleAssociationTrifocal(Path, sCams, TimeStamp, startF, stopF, increF);
		sprintf(Fname, "%s/Logs/MultiviewPeopleAssociationTrifocal_%d_%d_%d.txt", Path, startF, stopF, increF); fp = fopen(Fname, "w"); fclose(fp);
		return 0;
	}
	else if (mode == 20)
	{
		int camID = atoi(argv[1 + 2]), sharedIntrinsic = atoi(argv[1 + 3]), InterpAlgo = atoi(argv[1 + 4]), useJpeg = atoi(argv[1 + 5]), startF = mySfMPara.startF, stopF = mySfMPara.stopF;
		if (argc > 7)
			startF = atoi(argv[1 + 6]), stopF = atoi(argv[1 + 7]);

		int rCamID = camID;
		for (camID = 0; camID < mySfMPara.nCams; camID++)
		{
			if (rCamID != -1 && rCamID != camID)
				continue;
			printLOG("Working on %d\n", camID);
			sprintf(Fname, "%s/%d/Corrected", Path, camID); makeDir(Fname);

			VideoData VideoI;
			CameraData *allCamInfo = new CameraData[100];
			if (ReadVideoDataI(Path, VideoI, camID, startF, stopF) == 1)
				return 1;

			int rotated = 0;
			sprintf(Fname, "%s/RotationInfo.txt", Path); FILE *fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int cid, code;
				while (fscanf(fp, "%d %d", &cid, &code) != EOF)
					if (cid == camID)
						rotated = code;
				fclose(fp);
			}

			int nthreads = omp_get_max_threads();
			omp_set_num_threads(nthreads);

			int validframe = -1;
			for (int fid = startF; fid <= stopF && validframe == -1; fid++)
				if (VideoI.VideoInfo[fid].valid == 1)
					validframe = fid;

			if (sharedIntrinsic == 1)
			{
				if (validframe > -1)
				{
					int interval = (stopF - startF) / nthreads;
#pragma omp parallel for schedule(dynamic,1)
					for (int fid = startF; fid <= stopF; fid += interval)
					{
						char Fname[512], Fname2[512];
						vector<std::string> vNamesIn, vNnamesOut;
						for (int fidi = fid; fidi <= fid + interval - 1; fidi++)
						{
							sprintf(Fname, "%s/%d/%.4d.jpg", Path, camID, fidi);
							if (!IsFileExist(Fname))
							{
								sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fidi);
								if (!IsFileExist(Fname))
									continue;
							}

							if (useJpeg)
							{
								sprintf(Fname2, "%s/%d/Corrected/%.4d.jpg", Path, camID, fidi);
								if (IsFileExist(Fname2))
									continue;
							}
							else
							{
								sprintf(Fname2, "%s/%d/Corrected/%.4d.png", Path, camID, fidi);
								if (IsFileExist(Fname2))
									continue;
							}

							vNamesIn.push_back(string(Fname));
							vNnamesOut.push_back(string(Fname2));
						}
						if (vNamesIn.size() > 0)
							LensCorrectionImageSequenceDriver2(vNamesIn, vNnamesOut, VideoI.VideoInfo[validframe].intrinsic, VideoI.VideoInfo[validframe].distortion, fid, fid + interval - 1, 3, VideoI.VideoInfo[validframe].LensModel, InterpAlgo);
					}
				}
			}
			else
			{
				if (validframe > -1)
				{
					int interval = (stopF - startF) / nthreads;
#pragma omp parallel for schedule(dynamic,1)
					for (int fid = startF; fid <= stopF; fid += interval)
					{
						char Fname[512], Fname2[512];
						vector<std::string> vNamesIn, vNnamesOut;
						for (int fidi = fid; fidi <= min(stopF, fid + interval - 1); fidi++)
						{
							sprintf(Fname, "%s/%d/%.4d.jpg", Path, camID, fidi);
							if (!IsFileExist(Fname))
							{
								sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fidi);
								if (!IsFileExist(Fname))
									continue;
							}

							if (useJpeg)
							{
								sprintf(Fname2, "%s/%d/Corrected/%.4d.jpg", Path, camID, fidi);
								if (IsFileExist(Fname2))
									continue;
							}
							else
							{
								sprintf(Fname2, "%s/%d/Corrected/%.4d.png", Path, camID, fidi);
								if (IsFileExist(Fname2))
									continue;
							}
							vNamesIn.push_back(string(Fname));
							vNnamesOut.push_back(string(Fname2));
						}
						if (vNamesIn.size() > 0)
							LensCorrectionImageSequenceDriver3(vNamesIn, vNnamesOut, VideoI, fid, min(stopF, fid + interval - 1), 3, VideoI.VideoInfo[validframe].LensModel, rotated, InterpAlgo);
					}
				}
			}
		}

		return 0;
	}
	else if (mode == 21)
	{
		int nCams = atoi(argv[1 + 2]), keyFrameID = atoi(argv[1 + 3]), trackRange = atoi(argv[1 + 4]), cvPyrLevel = atoi(argv[1 + 5]), nThreads = atoi(argv[1 + 9]);
		double kfSuccessConsecutiveTrackingRatio = atof(argv[1 + 6]), kfSuccessRefTrackingRatio = atof(argv[1 + 7]), flowThresh = atof(argv[1 + 8]);

		int winDim = 31, highQualityTracking = 1, interpAlgo = -1;

		vector<Point2i> KeyFrameID2LocalFrameID = GetKeyFrameID2LocalFrameID(Path, nCams);
		int cid = KeyFrameID2LocalFrameID[keyFrameID].x, reffid = KeyFrameID2LocalFrameID[keyFrameID].y;


		if (keyFrameID > 2 && keyFrameID < KeyFrameID2LocalFrameID.size() - 3)
			trackRange = 2 * max(KeyFrameID2LocalFrameID[keyFrameID + 3].y - KeyFrameID2LocalFrameID[keyFrameID].y, KeyFrameID2LocalFrameID[keyFrameID].y - KeyFrameID2LocalFrameID[keyFrameID - 3].y);
		else if (keyFrameID > 1 && keyFrameID < KeyFrameID2LocalFrameID.size() - 2)
			trackRange = 3 * max(KeyFrameID2LocalFrameID[keyFrameID + 2].y - KeyFrameID2LocalFrameID[keyFrameID].y, KeyFrameID2LocalFrameID[keyFrameID].y - KeyFrameID2LocalFrameID[keyFrameID - 2].y);
		else if (keyFrameID > 0 && keyFrameID < KeyFrameID2LocalFrameID.size() - 1)
			trackRange = 4 * max(KeyFrameID2LocalFrameID[keyFrameID + 1].y - KeyFrameID2LocalFrameID[keyFrameID].y, KeyFrameID2LocalFrameID[keyFrameID].y - KeyFrameID2LocalFrameID[keyFrameID - 1].y);

		printLOG("Keyframe #%d-->(%d, %d) [%d] ", keyFrameID, cid, reffid, trackRange);
		TrackCorpusFeatureToNonKeyFrames(Path, cid, keyFrameID, reffid, trackRange, winDim, cvPyrLevel, 1.0, kfSuccessConsecutiveTrackingRatio*0.5, kfSuccessRefTrackingRatio*0.5, flowThresh, interpAlgo, highQualityTracking, nThreads, 0);
		return 0;
	}
	else if (mode == 22)
	{
		int startF = mySfMPara.startF, stopF = mySfMPara.stopF, extractedFrames = mySfMPara.extractedFrames;
		int cid = atoi(argv[1 + 2]);
		if (argc > 4)
			startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]), extractedFrames = atoi(argv[1 + 5]);

		sprintf(Fname, "%s/Logs/ImgBlurDetection_%d_%d_%d.txt", Path, cid, startF, stopF);
		if (IsFileExist(Fname) == 1)
			return 0;

		vector<int> goodFrames;
		sprintf(Fname, "%s/%d/goodFrames_%d_%d.txt", Path, cid, startF, stopF);
		if (IsFileExist(Fname) == 0)
		{
			if (extractedFrames == 0)
				VideoBasedBlurDetection(Path, cid, startF, stopF, goodFrames);
			else
				VideoBasedBlurDetection2(Path, cid, startF, stopF, goodFrames);
		}

		sprintf(Fname, "%s/Logs/ImgBlurDetection_%d_%d_%d.txt", Path, cid, startF, stopF); FILE *fp = fopen(Fname, "w"); fclose(fp);
		return 0;
	}
	else if (mode == 23)
	{
		int startF = mySfMPara.startF, stopF = mySfMPara.stopF,
			minFeaturesToTrack = mySfMPara.minFeaturesToTrack, minKFinterval = mySfMPara.minKFinterval, maxKFinterval = mySfMPara.maxKFinterval, highQualityTracking = mySfMPara.highQualityTracking,
			extractedFrames = mySfMPara.extractedFrames, dedicatedCorpus = mySfMPara.ExternalCorpus, display = 0;
		double kfSuccessConsecutiveTrackingRatio = mySfMPara.kfSuccessConsecutiveTrackingRatio, kfSuccessRefTrackingRatio = mySfMPara.kfSuccessRefTrackingRatio, kfFlowThresh = mySfMPara.kfFlowThresh;

		int cid = atoi(argv[1 + 2]);
		if (argc > 4)
		{
			startF = atoi(argv[1 + 3]), stopF = atoi(argv[1 + 4]);
			kfSuccessConsecutiveTrackingRatio = atof(argv[1 + 5]), kfSuccessRefTrackingRatio = atof(argv[1 + 6]), kfFlowThresh = atof(argv[1 + 7]);
			minFeaturesToTrack = atoi(argv[1 + 8]), minKFinterval = atoi(argv[1 + 9]), maxKFinterval = atoi(argv[1 + 10]), highQualityTracking = atoi(argv[1 + 11]), extractedFrames = atoi(argv[1 + 12]), dedicatedCorpus = atoi(argv[1 + 13]), display = 0;
			if (argc == 16)
				display = atoi(argv[1 + 14]);
		}

		//int FrameForCorpusStep = 600, minFeaturesToTrack = 1000; double kfFlowThresh = 200.0, kfSuccessConsecutiveTrackingRatio = 0.8, kfSuccessRefTrackingRatio = 0.4;
		printLOG("Keyframe for camera %d\n", cid);
		sprintf(Fname, "%s/Logs/ImKeyFramesExtraction_%d_%d_%d.txt", Path, cid, startF, stopF);
		if (IsFileExist(Fname) == 1)
			return 0;

		int fid;
		vector<int> goodFrames;
		sprintf(Fname, "%s/%d/goodFrames_%d_%d.txt", Path, cid, startF, stopF); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			while (fscanf(fp, "%d ", &fid) != EOF)
				goodFrames.push_back(fid);
			fclose(fp);

			int code = 0;
			vector<Point2i> rotateImage;
			sprintf(Fname, "%s/RotationInfo.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int icid, icode;
				while (fscanf(fp, "%d %d", &icid, &icode) != EOF)
					if (icid == cid)
					{
						code = icode;
						break;
					}
				fclose(fp);
			}

			KeyFramesViaOpticalFlow_HQ(Path, cid, startF, stopF, extractedFrames, code, goodFrames, kfSuccessConsecutiveTrackingRatio, kfSuccessRefTrackingRatio, kfFlowThresh,
				minFeaturesToTrack, minKFinterval, maxKFinterval, 1.0, -1, highQualityTracking, omp_get_max_threads(), display, true, dedicatedCorpus);

			vector<int> vrfid;
			int dummy, rfid;
			sprintf(Fname, "%s/%d/Frame2Corpus.txt", Path, cid);
			FILE *fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %d %d %d ", &dummy, &dummy, &rfid, &dummy) != EOF)
				vrfid.push_back(rfid);
			fclose(fp);

			sprintf(Fname, "%s/%d/querry.txt", Path, cid); fp = fopen(Fname, "w");
			for (auto fid : vrfid)
				fprintf(fp, "%.4d.jpg\n", fid);
			fclose(fp);
		}
		else
			printLOG("Cannot load %s. Skip camera %d\n", Fname, cid);
		sprintf(Fname, "%s/Logs/ImKeyFramesExtraction_%d_%d_%d.txt", Path, cid, startF, stopF); fp = fopen(Fname, "w"); fclose(fp);
		return 0;
	}
	else if (mode == 24)
	{
		int camID = atoi(argv[3]);
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(100);

		printLOG("Rotate image from camera %d: ", camID);
		Mat img;
		for (int fid = atoi(argv[4]); fid <= atoi(argv[5]); fid++)
		{
			printLOG("%d..", fid);
			sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fid);
			if (!IsFileExist(Fname))
			{
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, camID, fid);
				if (!IsFileExist(Fname))
					continue;
			}
			img = imread(Fname);
			int width = img.cols, height = img.rows, nchannels = 3;
			for (int kk = 0; kk < nchannels; kk++)
			{
				for (int jj = 0; jj < height / 2; jj++)
					for (int ii = 0; ii < width; ii++)
					{
						char buf = img.data[nchannels*ii + jj * nchannels*width + kk];
						img.data[nchannels*ii + jj * nchannels*width + kk] = img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk];
						img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk] = buf;
					}
			}

			if (atoi(argv[6]) == 1)
				imwrite(Fname, img, compression_params);
			else
				imwrite(Fname, img);
		}
		return 0;
	}
	else if (mode == 25)
	{
		int camID = atoi(argv[3]);
		double resizeFactor = atof(argv[6]);
		int nCOCOJoints = mySfMPara.SkeletonPointFormat;

		for (int camID = 2; camID < 15; camID++)
		{
			bool firstTime = true;
			VideoWriter writer;

			Mat img, rimg;
			for (int fid = atoi(argv[4]); fid <= atoi(argv[5]); fid++)
			{
				printf("%d..", fid);
				sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fid);
				if (!IsFileExist(Fname))
				{
					sprintf(Fname, "%s/%d/%.4d.jpg", Path, camID, fid);
					if (!IsFileExist(Fname))
						continue;
				}
				img = imread(Fname);

				if (firstTime)
				{
					CvSize size;
					size.width = (int)(resizeFactor*img.cols), size.height = (int)(resizeFactor*img.rows);
					sprintf(Fname, "%s/Vis/%d_%.4d_%.4d.avi", Path, camID, atoi(argv[4]), atoi(argv[5]));
					writer.open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 30, size);
					firstTime = false;
				}

				sprintf(Fname, "%s/MP/%d/%d.txt", Path, camID, fid);
				float u, v, s;
				vector<Point2f> joints;
				FILE *fp = fopen(Fname, "r");
				while (fscanf(fp, "%f %f %f ", &u, &v, &s) != EOF)
					joints.push_back(Point2f(u*resizeFactor, v*resizeFactor));
				fclose(fp);
				int npeople = joints.size() / nCOCOJoints;

				if (resizeFactor == 1)
				{
					CvPoint text_origin = { img.rows / 20, img.cols / 20 };
					sprintf(Fname, "%d ", fid);
					putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 255, 0), 3);

					for (int pid = 0; pid < npeople; pid++)
						Draw2DCoCoJoints(img, &joints[nCOCOJoints*pid], nCOCOJoints, 2, 1);

					writer << img;
				}
				else
				{
					resize(img, rimg, Size((int)(resizeFactor*img.cols), (int)(resizeFactor*img.rows)), 0, 0, INTER_AREA);
					CvPoint text_origin = { img.rows / 20, rimg.cols / 20 };
					sprintf(Fname, "%d ", fid);
					putText(rimg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 255, 0), 3);

					for (int pid = 0; pid < npeople; pid++)
						Draw2DCoCoJoints(rimg, &joints[nCOCOJoints*pid], nCOCOJoints, 2, 1);

					writer << rimg;
				}
			}
			writer.release();
		}
		return 0;
	}

	return 0;
}
