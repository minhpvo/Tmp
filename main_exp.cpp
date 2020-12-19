#include "DataStructure.h"
#include "Drivers/Drivers.h"
#include "Ulti/MathUlti.h"
#include "Vis/Visualization.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <ceres/normal_prior.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <andres/graph/graph.hxx>
#include <andres/graph/complete-graph.hxx>
#include <andres/graph/multicut/kernighan-lin.hxx>
#include <andres/graph/multicut-lifted/kernighan-lin.hxx>
#include <andres/graph/multicut-lifted/greedy-additive.hxx>
#include <andres/graph/digraph.hxx>
#include <andres/graph/bipartite-matching.hxx>
#include <andres/graph/graph.hxx>
#include <andres/graph/shortest-paths.hxx>

#include <time.h>
#ifdef _WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>

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
using cv::Point3d;
using cv::Point2d;
using cv::Mat;
using std::vector;
typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;

int MPosX, MPosY, clicked_;
void onMouseClick(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDBLCLK)
	{
		clicked_ = 1, MPosX = x, MPosY = y;
		printLOG("Selected: %d %d\n", x, y);
		cout << "\a";
	}
}

int ReadBodyKeyPointsAndDesc(char *Path, int cid, int fid, vector<BodyImgAtrribute> &Atrribute, int nBodyKeyPoints = 18)
{
	char Fname[512];

	float u, v, s;
	vector<Point2f> lm1;
	vector<float> vs1;

	lm1.clear(), vs1.clear();
	if (readOpenPoseJson(Path, cid, fid, lm1, vs1) == 0)
	{
		sprintf(Fname, "%s/MP/%d/%.4d.txt", Path, cid, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		while (fscanf(fp, "%f %f %f ", &u, &v, &s) != EOF)
			lm1.push_back(Point2f(u, v)), vs1.push_back(s);
		fclose(fp);
	}

	for (int pid = 0; pid < lm1.size() / nBodyKeyPoints; pid++)
	{
		BodyImgAtrribute attri;
		for (int jid = 0; jid < nBodyKeyPoints; jid++)
			attri.pt[jid].x = lm1[pid*nBodyKeyPoints + jid].x, attri.pt[jid].y = lm1[pid*nBodyKeyPoints + jid].y, attri.pt[jid].z = vs1[pid*nBodyKeyPoints + jid];
		attri.personLocalId = pid;
		attri.nDims = 1, attri.nPts = nBodyKeyPoints;
		Atrribute.emplace_back(attri);
	}

	return 0;
}

void Draw2DCoCo(Mat &img, BodyImgAtrribute lm, int lineThickness = 1, double resizeFactor = 1.0, int personId = -1)
{
	int nKeyPoints = lm.nPts;

	lineThickness = max(1, lineThickness);
	Point2f *joints = new Point2f[nKeyPoints];
	for (int ii = 0; ii < nKeyPoints; ii++)
		joints[ii] = cv::Point2f(lm.pt[ii].x * resizeFactor, lm.pt[ii].y * resizeFactor);

	if (personId > -1)
	{
		for (int ii = 0; ii < nKeyPoints; ii++)
		{
			if (joints[ii].x != 0)
			{
				char Fname[512];  sprintf(Fname, "%d", personId);
				putText(img, Fname, joints[ii], cv::FONT_HERSHEY_SIMPLEX, img.cols / 1080, cv::Scalar(0, 0, 255), 2.0*resizeFactor);
				break;
			}
		}
	}

	//head-neck
	int i = 0, j = 1;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 0, 255), lineThickness);

	//left arm
	i = 1, j = 2;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
	i = 2, j = 3;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
	i = 3, j = 4;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);

	//right arm
	i = 1, j = 5;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
	i = 5, j = 6;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
	i = 6, j = 7;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);

	//left leg
	i = 1, j = 8;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
	i = 8, j = 9;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
	i = 9, j = 10;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);

	//right leg
	i = 1, j = 11;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
	i = 11, j = 12;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
	i = 12, j = 13;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);

	//right eye+ ear
	i = 0, j = 14;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
	i = 14, j = 16;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);

	//left eye+ ear
	i = 0, j = 15;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
	i = 15, j = 17;
	if (joints[i].x != 0 && joints[j].x != 0 && lm.pt[i].z > 0.2 && lm.pt[j].z > 0.2)
		line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);

	for (int i = 0; i < 18; i++)
	{
		if (joints[i].x != 0.0)
		{
			circle(img, joints[i] * resizeFactor, 2, Scalar(0, 0, 255), lineThickness);
			//char Fname[512];  sprintf(Fname, "%d", i);
			//putText(img, Fname, joints[i] * resizeFactor, CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), min(2.5, 1.0));
		}
	}

	delete[]joints;

	return;
}
void Draw2DCoCoJoints(Mat &img, Point2f *joints, int lineThickness = 1, double resizeFactor = 1.0, int NUM_PARTS = 25, float *conf = NULL, int did = -1)
{
	if (NUM_PARTS == 18)
	{
		//head-neck
		int i = 0, j = 1;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 0, 255), lineThickness);

		//left arm
		i = 1, j = 2;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
		i = 2, j = 3;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
		i = 3, j = 4;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);

		//right arm
		i = 1, j = 5;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
		i = 5, j = 6;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
		i = 6, j = 7;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);

		//left leg
		i = 1, j = 8;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
		i = 8, j = 9;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
		i = 9, j = 10;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);

		//right leg
		i = 1, j = 11;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
		i = 11, j = 12;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
		i = 12, j = 13;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);

		//right eye+ ear
		i = 0, j = 14;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);
		i = 14, j = 16;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(255, 0, 0), lineThickness);

		//left eye+ ear
		i = 0, j = 15;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
		i = 15, j = 17;
		if (joints[i].x != 0 && joints[j].x != 0 && conf[i] > 0.2 && conf[j] > 0.2)
			line(img, joints[i] * resizeFactor, joints[j] * resizeFactor, Scalar(0, 255, 0), lineThickness);
	}
	else if (NUM_PARTS == 25)
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

	for (int i = 0; i < NUM_PARTS; i++)
	{
		if (joints[i].x != 0.0)
		{
			//circle(img, joints[i] * resizeFactor, 2, Scalar(0, 0, 255), lineThickness);
			//char Fname[512];  sprintf(Fname, "%d", i);
			// putText(img, Fname, joints[i] * resizeFactor, CV_FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), min(2.5, 1.0));

			if (did != -1)
			{
				circle(img, joints[i] * resizeFactor, 2, Scalar(0, 0, 255), lineThickness);
				char Fname[512];  sprintf(Fname, "%d", did);
				putText(img, Fname, joints[i] * resizeFactor, CV_FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(255, 0, 0), min(2.5, 1.0));
				break;
			}
		}
	}
	return;
}
int VisualizeTwoViewsEpipolarGeometry(Mat &img1, Mat &img2, CameraData &cam1, CameraData &cam2)
{
	char Fname[512];

	vector<Mat> vImg; vImg.push_back(img1), vImg.push_back(img2);

	Mat bImg = DrawTitleImages(vImg, 2.0);

	double Fmat[9];
	computeFmat(cam1, cam2, Fmat);

	Mat cvFmat = Mat(3, 3, CV_64F, &Fmat);
	namedWindow("VideoSequences", CV_WINDOW_NORMAL);
	setMouseCallback("VideoSequences", onMouseClick);
	int oMouseX = -1, oMouseY = -1;
	oMouseX = MPosX, oMouseY = MPosY;

	cv::RNG rng(0);

	while (true)
	{
		imshow("VideoSequences", bImg);
		if (waitKey(1) == 27)
			break;

		if (oMouseX != MPosX || oMouseY != MPosY)
			oMouseX = MPosX, oMouseY = MPosY;
		else
			continue;

		cv::Scalar color(rng(256), rng(256), rng(256));
		cv::Rect rect1(0, 0, vImg[0].cols, vImg[0].rows);
		cv::Rect rect2(vImg[0].cols, 0, vImg[1].cols, vImg[1].rows);

		Point2f pt1(oMouseX, oMouseY);
		double numX = Fmat[0] * pt1.x + Fmat[1] * pt1.y + Fmat[2];
		double numY = Fmat[3] * pt1.x + Fmat[4] * pt1.y + Fmat[5];
		double denum = Fmat[6] * pt1.x + Fmat[7] * pt1.y + Fmat[8];
		double epilines1[3] = { numX / denum, numY / denum, 1 };
		Point2i pt2(0, -(int)(epilines1[2] / epilines1[1])), pt3(vImg[0].cols, -(int)((epilines1[2] + epilines1[0] * vImg[0].cols) / epilines1[1]));
		cv::line(bImg(rect2), pt2, pt3, color);
		cv::circle(bImg(rect1), pt1, 3, color, -1, CV_AA);
		cout << "\a";
	}
	return 0;
}
void Visualize3DTracklet(char *Path, VideoData *VideoInfo, int nCams, int startF, int skeI, vector<sHumanSkeleton3D > &allSkeleton, vector<cv::Mat> *AllCamImages)
{
	char Fname[512];
	sprintf(Fname, "%s/Vis/3DTracklet/%.2d", Path, skeI); makeDir(Fname);

	int nKeyPoints = allSkeleton[0].sSke[0].nPts;

	//sprintf(Fname, "%s/%s/Vis/3DTracklet/%d/%.2d_%.2d.avi", Path, SelectedCamNames[0], SeqId, skeI, allSkeleton[skeI].nBAtimes);
	//cv::Size size; size.width = AllCamImages[0][0].cols, size.height = AllCamImages[0][0].rows;
	//cv::VideoWriter writer; writer.open(Fname,   CV_FOURCC('X', 'V', 'I', 'D'), 25, size);
	for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
	{
		sprintf(Fname, "%s//Vis/3DTracklet/%.2d/%.2d", Path, skeI, allSkeleton[skeI].nBAtimes); makeDir(Fname);
		for (int jj = 0; jj < nCams; jj++)
		{
			for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
			{
				int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, rfid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, refFid = rfid + (int)VideoInfo[cid].TimeOffset;
				if (cid == jj)
				{
					CameraData *camI = VideoInfo[cid].VideoInfo;

					BodyImgAtrribute lm;
					lm.personLocalId = skeI;
					lm.nPts = nKeyPoints;
					lm.nDims = 1;
					for (int jid = 0; jid < nKeyPoints; jid++)
					{
						if (!IsValid3D(allSkeleton[skeI].sSke[inst].pt3d[jid]))
							continue;
						Point2d pt;
						ProjectandDistort(allSkeleton[skeI].sSke[inst].pt3d[jid], &pt, camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
						lm.pt[jid].x = pt.x, lm.pt[jid].y = lm.pt[jid].y;
					}
					cv::Mat img = AllCamImages[cid][refFid - startF].clone();
					Draw2DCoCo(img, lm);
					cv::Point2i text_origin = { img.cols / 20, img.rows / 15 };
					sprintf(Fname, "%d: %d (%d)", cid, rfid, refFid); putText(img, Fname, text_origin, cv::FONT_HERSHEY_SIMPLEX, img.cols / 640, cv::Scalar(0, 255, 0), 2.0);

					sprintf(Fname, "%s/Vis/3DTracklet/%.2d/%.2d/%d_%.4d.jpg", Path, skeI, allSkeleton[skeI].nBAtimes, cid, refFid); cv::imwrite(Fname, img);
					//writer << img;
				}
			}
		}
	}
	//writer.release();

	return;
}

vector<Point2i> MatchTwoFrame_BodyKeyPoints_LK(CameraData &CamI, vector<Mat> &ImgOldPyr, vector<Mat>& ImgNewPyr, vector<BodyImgAtrribute> attri1, vector<BodyImgAtrribute> attri2, double bwThresh, double confThresh = 0.2, bool correctedKeyPoints = true)
{
	static cv::Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 128, 255), Scalar(0, 255, 255), Scalar(0, 255, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(255, 255, 255) };
	int debug = 0;

	int nPts = attri1[0].nPts;
	int npid1 = attri1.size(), npid2 = attri2.size();

	double bwThresh2 = bwThresh * bwThresh;
	int winSizeI = 31, cvPyrLevel = ImgOldPyr.size() - 1;
	vector<float> err;
	vector<uchar> status;
	Size winSize(winSizeI, winSizeI);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01);

	vector<float> vs1, vs2;
	vector<Point2f> lm1, lm2, lm1_, lm2_;
	if (correctedKeyPoints)
	{
		for (int pid = 0; pid < npid1; pid++)
		{
			for (int jid = 0; jid < nPts; jid++)
			{
				Point2f pt(attri1[pid].pt[jid].x, attri1[pid].pt[jid].y);
				LensDistortionPoint(&pt, CamI.K, CamI.distortion);
				lm1.emplace_back(pt), vs1.emplace_back(attri1[pid].pt[jid].z);
			}
		}
		for (int pid = 0; pid < npid2; pid++)
		{
			for (int jid = 0; jid < nPts; jid++)
			{
				Point2f pt(attri2[pid].pt[jid].x, attri2[pid].pt[jid].y);
				LensDistortionPoint(&pt, CamI.K, CamI.distortion);
				lm2.emplace_back(pt), vs2.emplace_back(attri2[pid].pt[jid].z);
			}
		}
	}
	else
	{
		for (int pid = 0; pid < npid1; pid++)
			for (int jid = 0; jid < nPts; jid++)
				lm1.emplace_back(attri1[pid].pt[jid].x, attri1[pid].pt[jid].y), vs1.emplace_back(attri1[pid].pt[jid].z);
		for (int pid = 0; pid < npid2; pid++)
			for (int jid = 0; jid < nPts; jid++)
				lm2.emplace_back(attri2[pid].pt[jid].x, attri2[pid].pt[jid].y), vs2.emplace_back(attri2[pid].pt[jid].z);
	}

	calcOpticalFlowPyrLK(ImgOldPyr, ImgNewPyr, lm1, lm2_, status, err, winSize, cvPyrLevel, termcrit);
	calcOpticalFlowPyrLK(ImgNewPyr, ImgOldPyr, lm2, lm1_, status, err, winSize, cvPyrLevel, termcrit);

	vector<int> association(npid1), used(npid2);
	for (int pid1 = 0; pid1 < npid1; pid1++)
	{
		int nvalidJoints = 0;
		for (int jid = 0; jid < nPts; jid++)
		{
			if (lm1[pid1*nPts + jid].x > 0 && vs1[jid + pid1 * nPts] > confThresh)  //bad detection
				nvalidJoints++;
		}
		if (nvalidJoints < nPts / 3)  //let's be conservative about occlusion
		{
			association[pid1] = -1;
			continue;
		}

		if (debug == 2)
		{
			Point2f tl = Point2f(9e9, 9e9), br = Point2f(0, 0);
			for (size_t jid = 0; jid < nPts; jid++)
				if (lm1[pid1*nPts + jid].x > 0)
					tl.x = min(tl.x, lm1[pid1*nPts + jid].x), tl.y = min(tl.y, lm1[pid1*nPts + jid].y), br.x = max(br.x, lm1[pid1*nPts + jid].x), br.y = max(br.y, lm1[pid1*nPts + jid].y);
			Mat xxx = ImgOldPyr[0].clone(); cvtColor(xxx, xxx, CV_GRAY2BGR);
			rectangle(xxx, tl, br, colors[pid1 % 9], 8, 8, 0);
			namedWindow("1", CV_WINDOW_NORMAL);
			imshow("1", xxx); cvWaitKey(1);
		}

		int bestAssignment = -1, best = 0;
		for (int pid2 = 0; pid2 < npid2; pid2++)
		{
			nvalidJoints = 0;
			for (int jid = 0; jid < nPts; jid++)
				if (lm2[pid2*nPts + jid].x > 0 && vs2[jid + pid2 * nPts] > confThresh)  //bad detection
					nvalidJoints++;
			if (nvalidJoints < nPts / 3)  //let's be conservative about occlusion
				continue;

			int good = 0;
			double dist1, dist2;
			for (int jid = 0; jid < nPts; jid++)
			{
				if (lm1[pid1 * nPts + jid].x > 0 && lm2[pid2 * nPts + jid].x > 0)
				{
					Point2f p1 = lm1[pid1 * nPts + jid], p2 = lm2[pid2 * nPts + jid];
					Point2f p1_ = lm2_[pid1 * nPts + jid], p2_ = lm2_[pid1 * nPts + jid];
					dist1 = pow(lm2_[pid1 * nPts + jid].x - lm2[pid2 * nPts + jid].x, 2) + pow(lm2_[pid1 * nPts + jid].y - lm2[pid2 * nPts + jid].y, 2);
					dist2 = pow(lm1[pid1 * nPts + jid].x - lm1_[pid2 * nPts + jid].x, 2) + pow(lm1[pid1 * nPts + jid].y - lm1_[pid2 * nPts + jid].y, 2);
					if (dist1 < bwThresh2 && dist2 < bwThresh2)
						good++;
				}
			}
			if (good > best)
			{
				best = good, bestAssignment = pid2;
				if (debug == 2)
				{
					Point2f tl = Point2f(9e9, 9e9), br = Point2f(0, 0);
					for (size_t jid = 0; jid < nPts; jid++)
						if (lm2[pid2*nPts + jid].x > 0)
							tl.x = min(tl.x, lm2[pid2*nPts + jid].x), tl.y = min(tl.y, lm2[pid2*nPts + jid].y), br.x = max(br.x, lm2[pid2*nPts + jid].x), br.y = max(br.y, lm2[pid2*nPts + jid].y);
					Mat xxx = ImgNewPyr[0].clone(); cvtColor(xxx, xxx, CV_GRAY2BGR);
					rectangle(xxx, tl, br, colors[pid1 % 9], 8, 8, 0);
					imshow("2", xxx); cvWaitKey(0);
				}
			}
		}
		if (best >= nPts / 3 && used[bestAssignment] == 0) //Let's be conservative at this point
			association[pid1] = bestAssignment, used[bestAssignment] = 1;  //establish link.
		else
			association[pid1] = -1;
	}

	vector<Point2i> matches;
	for (int ii = 0; ii < npid1; ii++)
		if (association[ii] != -1)
			matches.emplace_back(ii, association[ii]);

	if (debug == 1)
	{
		Mat img1 = ImgOldPyr[0].clone(); cvtColor(img1, img1, CV_GRAY2BGR);
		Mat img2 = ImgOldPyr[0].clone(); cvtColor(img2, img2, CV_GRAY2BGR);
		for (int ii = 0; ii < matches.size(); ii++)
		{
			int pid1 = matches[ii].x;

			Point2f tl = Point2f(9e9, 9e9), br = Point2f(0, 0);
			for (size_t jid = 0; jid < nPts; jid++)
				if (lm1[pid1*nPts + jid].x > 0)
					tl.x = min(tl.x, lm1[pid1*nPts + jid].x), tl.y = min(tl.y, lm1[pid1*nPts + jid].y), br.x = max(br.x, lm1[pid1*nPts + jid].x), br.y = max(br.y, lm1[pid1*nPts + jid].y);
			rectangle(img1, tl, br, colors[pid1 % 9], 8, 8, 0);
		}
		for (int ii = 0; ii < matches.size(); ii++)
		{
			int pid1 = matches[ii].x, pid2 = matches[ii].y;

			Point2f tl = Point2f(9e9, 9e9), br = Point2f(0, 0);
			for (size_t jid = 0; jid < nPts; jid++)
				if (lm2[pid2*nPts + jid].x > 0)
					tl.x = min(tl.x, lm2[pid2*nPts + jid].x), tl.y = min(tl.y, lm2[pid2*nPts + jid].y), br.x = max(br.x, lm2[pid2*nPts + jid].x), br.y = max(br.y, lm2[pid2*nPts + jid].y);
			rectangle(img2, tl, br, colors[pid1 % 9], 8, 8, 0);
		}
		namedWindow("1", CV_WINDOW_NORMAL);
		imshow("1", img1); cvWaitKey(1);
		namedWindow("2", CV_WINDOW_NORMAL);
		imshow("2", img2); cvWaitKey(0);
	}

	return matches;
}
Eigen::MatrixXf ComputeFmatDistanceMatrix(std::vector<BodyImgAtrribute> &kpts1, std::vector<BodyImgAtrribute> &kpts2, CameraData &Cam1, CameraData &Cam2)
{
	int nf1 = (int)kpts1.size(), nf2 = (int)kpts2.size(), nKeyPoints = kpts1[0].nPts;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dists(nf1, nf2);

	double Fmat[9];
	computeFmat(Cam1, Cam2, Fmat);

	for (int i1 = 0; i1 < nf1; ++i1)
	{
		for (int i2 = 0; i2 < nf2; ++i2)
		{
			int nvalid = 0;
			double dist = 0.0;
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				if (kpts1[i1].pt[jid].z != 0.0 && kpts1[i1].pt[jid].z != 0.0)
				{
					dist += FmatPointError(Fmat, Point2d(kpts1[i1].pt[jid].x, kpts1[i1].pt[jid].y), Point2d(kpts2[i2].pt[jid].x, kpts2[i2].pt[jid].y)), nvalid++;
				}
			}
			dists(i1, i2) = dist / (1e-6 + nvalid);
		}
	}

	return dists;
}
vector<Point2i> MatchTwoViewDescGeoBipartite(vector<BodyImgAtrribute> &attri1, CameraData &Cam1, vector<BodyImgAtrribute> &attri2, CameraData &Cam2, int descDim, float DescThresh, float GeoThresh)
{
	if (attri1.size() == 0 || attri2.size() == 0)
	{
		vector<Point2i> ematches;
		return ematches;
	}

	int nKeyPoints = attri1[0].nPts;
	BodyImgAtrribute attr;
	vector<BodyImgAtrribute> attri1_, attri2_;
	for (int pid = 0; pid < attri1.size(); pid++)
	{
		double confSum = 0;
		for (int jid = 0; jid < nKeyPoints; jid++)
		{
			attr.pt[jid] = attri1[pid].pt[jid];
			confSum += attri1[pid].pt[jid].z;
		}
		attr.nPts = nKeyPoints;
		attr.personLocalId = pid;
		if (confSum > 0)
			attri1_.push_back(attr);
	}
	if (attri1_.size() == 0)
	{
		vector<Point2i> ematches;
		return ematches;
	}

	for (int pid = 0; pid < attri2.size(); pid++)
	{
		double confSum = 0;
		for (int jid = 0; jid < nKeyPoints; jid++)
		{
			attr.pt[jid] = attri2[pid].pt[jid];
			confSum += attri2[pid].pt[jid].z;
		}
		attr.nPts = nKeyPoints;
		attr.personLocalId = pid;
		if (confSum > 0)
			attri2_.push_back(attr);
	}
	if (attri2_.size() == 0)
	{
		vector<Point2i> ematches;
		return ematches;
	}

	//const Eigen::MatrixXf ApperanceDists = ComputeDescDistanceMatrix(descriptors1, descriptors2, descDim, 1); //1 is best
	const Eigen::MatrixXf GemDists = ComputeFmatDistanceMatrix(attri1_, attri2_, Cam1, Cam2); //0 is best

	bool debug = false;
	if (debug)
	{
		//cout << endl << ApperanceDists << endl << endl;
		cout << endl << GemDists << endl << endl;
	}

	using graph_t = andres::graph::Digraph<>;
	using pair_t = std::pair<std::size_t, std::size_t>;

	//construct graph and costs.
	graph_t graph(attri1_.size() + attri2_.size());
	std::vector<double> costs; costs.reserve(attri1_.size()*attri2_.size());

	auto add_edge = [&](std::size_t v, std::size_t w, typename  std::vector<double>::value_type cost)
	{
		graph.insertEdge(v, w);
		costs.emplace_back(std::move(cost));
	};

	//the graph finds the min cost
	int nValidEdges = 0;
	for (size_t ii1 = 0; ii1 < attri1_.size(); ii1++)
	{
		for (size_t ii2 = 0; ii2 < attri2_.size(); ii2++)
		{
			if (GemDists(ii1, ii2) < GeoThresh)// && ApperanceDists(ii1, ii2) > DescThresh)
			{
				//add_edge(ii1, ii2 + kpts1.size(), (1.0 - ApperanceDists(ii1, ii2)) * GemDists(ii1, ii2));
				add_edge(ii1, ii2 + attri1_.size(), GemDists(ii1, ii2));
				//printLOG( "%d %d %.3f\n", ii1, ii2 + kpts1.size(), (1.0 - ApperanceDists(ii1, ii2)) * GemDists(ii1, ii2));
				nValidEdges++;
			}
			else
			{
				add_edge(ii1, ii2 + attri1_.size(), 9e9);
			}
		}
	}

	if (nValidEdges == 0)
	{
		vector<Point2i> ematches;
		return ematches;
	}
	// find best matching.
	std::vector<unsigned int> edge_labels(costs.size(), 0); //labels: 1 is connected
	if (andres::graph::findMCBM(graph, costs, edge_labels) == 1) //graph fails
	{
		vector<Point2i> ematches;
		return ematches;
	}

	// and construct matches from mask.
	const auto matches = [](graph_t const& graph, std::vector<unsigned int> const& edge_labels)
	{
		std::vector<std::pair<std::size_t, std::size_t>> matches;
		for (std::size_t edge = 0; edge < edge_labels.size(); ++edge)
		{
			if (edge_labels[edge] == 1)
			{
				const auto v0 = graph.vertexOfEdge(edge, 0);
				const auto v1 = graph.vertexOfEdge(edge, 1);
				matches.emplace_back(std::move(v0), std::move(v1));
			}
		}
		return matches;
	}(graph, edge_labels);

	vector<Point2i> ematches_;
	for (auto m : matches)
	{
		if (GemDists(m.first, m.second - attri1_.size()) < GeoThresh)
			ematches_.emplace_back(m.first, m.second - attri1_.size());
	}

	vector<Point2i> ematches;
	for (auto m : matches)
	{
		if (GemDists(m.first, m.second - attri1_.size()) < GeoThresh)
			ematches.emplace_back(attri1_[m.first].personLocalId, attri2_[m.second - attri1_.size()].personLocalId);
	}

	return ematches;
}
int GeneratePeopleMatchingTable(VideoData *VideoInfo, vector<BodyImgAtrribute> *Attri, int nCams, int refFid, vector<int>*ViewMatch, vector<int>*DetectionIdMatch, float DescThresh, float triangulationThresh, vector<vector<Point2i> > & vUsedCidDid)
{
	int reIDDim = Attri[0][0].nDims;
	int debug = 0;

	//Gen allviews matching table
	vector<int> *DetectionBelongTo3D = new vector <int>[nCams];
	for (int cid = 0; cid < nCams; cid++)
	{
		DetectionBelongTo3D[cid].reserve(Attri[cid].size());
		for (size_t ii = 0; ii < Attri[cid].size(); ii++)
			DetectionBelongTo3D[cid].push_back(-1);
	}

	int possiblePeopleCnt = 0;
	for (int cid1 = 0; cid1 < nCams - 1; cid1++)
	{
		for (int cid2 = cid1 + 1; cid2 < nCams; cid2++)
		{
			int lfid1 = refFid - (int)VideoInfo[cid1].TimeOffset, lfid2 = refFid - (int)VideoInfo[cid2].TimeOffset;
			if (debug)
			{
				char Fname[512];
				sprintf(Fname, "D:/WildTrack/%d/Corrected/%.4d.jpg", cid1, lfid1); Mat img1 = imread(Fname);
				for (int pid = 0; pid < Attri[cid1].size(); pid++)
					Draw2DCoCo(img1, Attri[cid1][pid], 1, 1.0, pid);
				sprintf(Fname, "D:/WildTrack/%d/Corrected/%.4d.jpg", cid2, lfid2); Mat img2 = imread(Fname);
				for (int pid = 0; pid < Attri[cid2].size(); pid++)
					Draw2DCoCo(img2, Attri[cid2][pid], 1, 1.0, pid);
				VisualizeTwoViewsEpipolarGeometry(img1, img2, VideoInfo[cid1].VideoInfo[lfid1], VideoInfo[cid2].VideoInfo[lfid2]);
			}
			vector<Point2i> matches = MatchTwoViewDescGeoBipartite(Attri[cid1], VideoInfo[cid1].VideoInfo[lfid1], Attri[cid2], VideoInfo[cid2].VideoInfo[lfid2], reIDDim, DescThresh, triangulationThresh * 2);

			for (int kk = 0; kk < matches.size(); kk++)
			{
				int id1 = matches[kk].x, id2 = matches[kk].y;
				int ID3D1 = DetectionBelongTo3D[cid1][id1], ID3D2 = DetectionBelongTo3D[cid2][id2];

				//detect used hypo
				int usedHypo = -1;
				for (size_t ii = 0; ii < vUsedCidDid.size() && usedHypo == -1; ii++)
				{
					int nfound = 0;
					for (size_t jj = 0; jj < vUsedCidDid[ii].size() && nfound == 0; jj++)
					{
						if (cid1 == vUsedCidDid[ii][jj].x && id1 == vUsedCidDid[ii][jj].y)
							nfound++;
					}
					if (nfound > 0)
						usedHypo = ii;
					nfound = 0;
					for (size_t jj = 0; jj < vUsedCidDid[ii].size() && nfound == 0; jj++)
					{
						if (cid2 == vUsedCidDid[ii][jj].x && id2 == vUsedCidDid[ii][jj].y)
							nfound++;
					}
					if (nfound > 0)
						usedHypo = ii;
				}
				if (usedHypo != -1)
					continue;

				//remove bad detections
				double sumConf = 0.0;
				for (int ii = 0; ii < Attri[cid1][0].nPts; ii++)
					sumConf += Attri[cid1][id1].pt[ii].z;
				if (sumConf == 0)
					continue;
				sumConf = 0.0;
				for (int ii = 0; ii < Attri[cid2][0].nPts; ii++)
					sumConf += Attri[cid2][id2].pt[ii].z;
				if (sumConf == 0)
					continue;

				if (ID3D1 == -1 && ID3D2 == -1) //Both are never seeen before
				{
					ViewMatch[possiblePeopleCnt].push_back(cid1), ViewMatch[possiblePeopleCnt].push_back(cid2);
					DetectionIdMatch[possiblePeopleCnt].push_back(id1), DetectionIdMatch[possiblePeopleCnt].push_back(id2);
					DetectionBelongTo3D[cid1][id1] = possiblePeopleCnt, DetectionBelongTo3D[cid2][id2] = possiblePeopleCnt; //this pair of corres constitutes 3D point #count
					possiblePeopleCnt++;
				}
				else if (ID3D1 == -1 && ID3D2 != -1)
				{
					ViewMatch[ID3D2].push_back(cid1);
					DetectionIdMatch[ID3D2].push_back(id1);
					DetectionBelongTo3D[cid1][id1] = ID3D2; //this point constitutes 3D point #ID3D2
				}
				else if (ID3D1 != -1 && ID3D2 == -1)
				{
					ViewMatch[ID3D1].push_back(cid2);
					DetectionIdMatch[ID3D1].push_back(id2);
					DetectionBelongTo3D[cid2][id2] = ID3D1; //this point constitutes 3D point #ID3D2
				}
				else if (ID3D1 != -1 && ID3D2 != -1 && ID3D1 != ID3D2)//Strange case where 1 point (usually not vey discrimitive or repeating points) is matched to multiple points in the same view pair --> Just concatanate the one with fewer points to largrer one and hope MultiTriangulationRansac can do sth.
				{
					if (ViewMatch[ID3D1].size() >= ViewMatch[ID3D2].size())
					{
						int nmatches = (int)ViewMatch[ID3D2].size();
						for (int ll = 0; ll < nmatches; ll++)
						{
							ViewMatch[ID3D1].push_back(ViewMatch[ID3D2].at(ll));
							DetectionIdMatch[ID3D1].push_back(DetectionIdMatch[ID3D2].at(ll));
						}
						ViewMatch[ID3D2].clear(), DetectionIdMatch[ID3D2].clear();
					}
					else
					{
						int nmatches = (int)ViewMatch[ID3D1].size();
						for (int ll = 0; ll < nmatches; ll++)
						{
							ViewMatch[ID3D2].push_back(ViewMatch[ID3D1].at(ll));
							DetectionIdMatch[ID3D2].push_back(DetectionIdMatch[ID3D1].at(ll));
						}
						ViewMatch[ID3D1].clear(), DetectionIdMatch[ID3D1].clear();
					}
				}
				else//(ID3D1 == ID3D2): cycle in the corres, i.e. a-b, a-c, and b-c
					continue;
			}
		}
	}
	delete[]DetectionBelongTo3D;

	return possiblePeopleCnt;
}

struct LeastMotionPriorCost3DCeres3 {
	LeastMotionPriorCost3DCeres3(double *xyz1_, double timeStamp1, double timeStamp2, double sig_ivel) : timeStamp1(timeStamp1), timeStamp2(timeStamp2), sig_ivel(sig_ivel)
	{
		xyz1 = xyz1_;
	}

	template <typename T>	bool operator()(const T* const xyz2, T* residuals) 	const
	{
		T temp = (T)(sig_ivel / ceres::sqrt(ceres::abs(timeStamp2 - timeStamp1)));
		for (int ii = 0; ii < 3; ii++)
			residuals[ii] = (xyz2[ii] - xyz1[ii]) * temp;  //(v/sig_v)^2*dt = (dx/dt/sig_v)^2*dt = (dx/sig_v)^2/dt

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double *xyz1, double Stamp1, double Stamp2, double sig_ivel)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCost3DCeres3, 3, 3>(new LeastMotionPriorCost3DCeres3(xyz1, Stamp1, Stamp2, sig_ivel)));
	}
	double timeStamp1, timeStamp2, sig_ivel;
	double *xyz1;
};

int PerFrameSkeleton3DBundleAdjustment(sHumanSkeleton3D &sSke, vector<Point3d> &v3DPoints, vector<double> *vP, vector<Point2d> *vPts, int LossType, double *Weights, double *iSigma, int increF, double ifps, double real2SfM, bool silent)
{
	int nKeyPoints = v3DPoints.size();

	//Weight = [ const limb length, symmetric skeleton, temporal]. It is  helpful for insightful weight setting if metric unit (mm) is used
	double sigma_i2D = iSigma[0], sigma_iL = iSigma[1] * real2SfM, sigma_iVel = iSigma[2] * real2SfM, sigma_iVel2 = iSigma[3] * real2SfM; //also convert physical sigma to sfm scale sigma

																																		  //For COCO 18-points 
	const int nLimbConnections = 17, nSymLimbConnectionID = 9;
	Point2i LimbConnectionID[nLimbConnections] = { Point2i(0, 1), Point2i(1, 2), Point2i(2, 3), Point2i(3, 4), Point2i(1, 5), Point2i(5, 6), Point2i(6, 7),
		Point2i(1, 8), Point2i(8, 9), Point2i(9, 10), Point2i(1, 11), Point2i(11, 12), Point2i(12, 13), Point2i(0, 14), Point2i(0, 15), Point2i(14, 16), Point2i(15, 17) };
	Vector4i SymLimbConnectionID[nSymLimbConnectionID] = { Vector4i(1, 2, 1, 5), Vector4i(2, 3, 5, 6), Vector4i(3, 4, 6, 7), Vector4i(1, 8, 1, 11), Vector4i(8, 9, 11, 12), Vector4i(9, 10, 12, 13) , Vector4i(0, 14, 0, 15), Vector4i(14, 16, 15, 17), Vector4i(0, 16, 0, 17) }; //no eyes, ears since they are unreliable

	ceres::Problem problem;
	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(1.0);

	double residuals[3], rho[3];

	vector<double> VreprojectionError, VUnNormedReprojectionError, VconstLimbError, VsymLimbError, VtemporalError;

	//reprojection error
	for (int jid = 0; jid < nKeyPoints; jid++)
	{
		for (int lcid = 0; lcid < vPts[jid].size(); lcid++)
		{
			Point2d uv(vPts[jid][lcid].x, vPts[jid][lcid].y);
			if (uv.x == 0 || uv.y == 0)
				continue;
			ceres::LossFunction* robust_loss = new ceres::HuberLoss(1);
			ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, Weights[0], ceres::TAKE_OWNERSHIP);
			ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(&vP[jid][12 * lcid], uv.x, uv.y, sigma_i2D);
			problem.AddResidualBlock(cost_function, weightLoss, &v3DPoints[jid].x);

			if (!silent)
			{
				vector<double *> paras; paras.push_back(&v3DPoints[jid].x);
				cost_function->Evaluate(&paras[0], residuals, NULL);
				robust_loss->Evaluate(residuals[0] * residuals[0] + residuals[1] * residuals[1], rho);
				VreprojectionError.push_back(Weights[0] * 0.5*rho[0]);
				VUnNormedReprojectionError.push_back(residuals[0] / sigma_i2D), VUnNormedReprojectionError.push_back(residuals[1] / sigma_i2D);
			}
		}
	}

	//constant limb length
	if (sSke.nBAtimes > 0)
	{
		for (int cid = 0; cid < nLimbConnections; cid++)
		{
			int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
			if (IsValid3D(v3DPoints[j0]) && IsValid3D(v3DPoints[j1]))
			{
				ceres::CostFunction* cost_function = ConstantLimbLengthCost3DCeres::CreateAutoDiff(sSke.meanBoneLength[cid], sigma_iL);
				ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[1], ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(cost_function, ScaleLoss, &v3DPoints[j0].x, &v3DPoints[j1].x);

				if (!silent)
				{
					vector<double *> paras;  paras.push_back(&v3DPoints[j0].x), paras.push_back(&v3DPoints[j1].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VconstLimbError.push_back(0.5*Weights[1] * residuals[0] * residuals[0]);
				}
			}
		}
	}

	//symmetry limb
	for (int cid = 0; cid < nSymLimbConnectionID; cid++)
	{
		int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
		if (IsValid3D(v3DPoints[j0]) && IsValid3D(v3DPoints[j1]) && IsValid3D(v3DPoints[j0_]) && IsValid3D(v3DPoints[j1_]))
		{
			if (j0 == j0_)
			{
				ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);
				ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[2], ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(cost_function, ScaleLoss, &v3DPoints[j0].x, &v3DPoints[j1].x, &v3DPoints[j1_].x);

				if (!silent)
				{
					vector<double *> paras; paras.push_back(&v3DPoints[j0].x), paras.push_back(&v3DPoints[j1].x), paras.push_back(&v3DPoints[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
			}
			else
			{
				ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres::CreateAutoDiff(sigma_iL);
				ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[2], ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(cost_function, ScaleLoss, &v3DPoints[j0].x, &v3DPoints[j1].x, &v3DPoints[j0_].x, &v3DPoints[j1_].x);

				if (!silent)
				{
					vector<double *> paras; paras.push_back(&v3DPoints[j0].x), paras.push_back(&v3DPoints[j1].x), paras.push_back(&v3DPoints[j0_].x), paras.push_back(&v3DPoints[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
			}
		}
	}

	if (sSke.sSke[0].vCidFidDid.size() > 0 && Weights[3] > 0.0)
	{
		int tempFid = sSke.sSke.back().refFid;
		for (int jid = 0; jid < nKeyPoints; jid++)
		{
			double actingSigma = sigma_iVel;
			if (jid == 9 || jid == 10 || jid == 15 || jid == 16) //allow hands and feet to move faster
				actingSigma = sigma_iVel2;

			//fit regarless of the point is valid or not
			if (IsValid3D(sSke.sSke.back().pt3d[jid]) && IsValid3D(v3DPoints[jid])) //temporal smoothing
			{
				ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[3], ceres::TAKE_OWNERSHIP);
				ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres3::CreateAutoDiff(&sSke.sSke.back().pt3d[jid].x, ifps* tempFid, ifps * (tempFid + increF), actingSigma);
				problem.AddResidualBlock(cost_function, ScaleLoss, &v3DPoints[jid].x);

				if (!silent)
				{
					vector<double *> paras; paras.push_back(&v3DPoints[jid].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VtemporalError.push_back(0.5*Weights[3] * residuals[0] * residuals[0]);
				}
			}
		}
	}

	if (!silent)
	{
		if (VreprojectionError.size() == 0)
			VreprojectionError.push_back(0);
		if (VUnNormedReprojectionError.size() == 0)
			VUnNormedReprojectionError.push_back(0);
		if (VconstLimbError.size() == 0)
			VconstLimbError.push_back(0);
		if (VsymLimbError.size() == 0)
			VsymLimbError.push_back(0);
		if (VtemporalError.size() == 0)
			VtemporalError.push_back(0);

		double reproSoS = MeanArray(VreprojectionError)*VreprojectionError.size(),
			unNormedRepro = MeanArray(VUnNormedReprojectionError),
			stdUnNormedRepro = sqrt(VarianceArray(VUnNormedReprojectionError, unNormedRepro)),
			maxUnNormedRepro = *std::max_element(VreprojectionError.begin(), VreprojectionError.end()), minUnNormedRePro = *std::min_element(VreprojectionError.begin(), VreprojectionError.end()),
			cLimbSoS = MeanArray(VconstLimbError)*VconstLimbError.size(), cLimb = sqrt(MeanArray(VconstLimbError)*2.0 / Weights[1]) / sigma_iL * real2SfM,
			sSkeleSoS = MeanArray(VsymLimbError)*VsymLimbError.size(), sSkele = sqrt(MeanArray(VsymLimbError)*2.0 / Weights[2]) / sigma_iL * real2SfM,
			motionCoSoS = MeanArray(VtemporalError)*VtemporalError.size(), motionCo = sqrt(MeanArray(VtemporalError)*2.0 / Weights[3]) / sigma_iVel * real2SfM;
		printLOG("Before optim:\n");
		printLOG("Reprojection: normalized: %.3f  unnormalized mean: %.3f unnormalized std: %.3f max: %.3f min: %.3f \n", reproSoS, unNormedRepro, stdUnNormedRepro, maxUnNormedRepro, minUnNormedRePro);
		if (sSke.nBAtimes > 0)
			printLOG("Const Limb: normalized: %.3f  unnormalized: %.3f\n", cLimbSoS, cLimb);
		printLOG("Sym skeleton: normalized: %.3f  unnormalized: %.3f\n", sSkeleSoS, sSkele);
		if (sSke.sSke[0].vCidFidDid.size() > 0 && Weights[3] > 0.0)
			printLOG("Motion coherent: normalized: %.3f  unnormalized: %.3f\n", motionCoSoS, motionCo);
	}

	ceres::Solver::Options options;
	options.num_threads = 1;
	options.num_linear_solver_threads = 1;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = 50;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	if (!silent)
	{
		std::cout << summary.BriefReport() << "\n";

		VreprojectionError.clear(), VUnNormedReprojectionError.clear(), VconstLimbError.clear(), VsymLimbError.clear();
		//reprojection error
		for (int jid = 0; jid < nKeyPoints; jid++)
		{
			for (int lcid = 0; lcid < vPts[jid].size(); lcid++)
			{
				Point2d uv(vPts[jid][lcid].x, vPts[jid][lcid].y);
				if (uv.x == 0 || uv.y == 0)
					continue;

				ceres::LossFunction* robust_loss = new ceres::HuberLoss(1);
				ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(&vP[jid][12 * lcid], uv.x, uv.y, sigma_i2D);

				vector<double *> paras; paras.push_back(&v3DPoints[jid].x);
				cost_function->Evaluate(&paras[0], residuals, NULL);
				robust_loss->Evaluate(residuals[0] * residuals[0] + residuals[1] * residuals[1], rho);
				VreprojectionError.push_back(Weights[0] * 0.5*rho[0]);
				VUnNormedReprojectionError.push_back(residuals[0] / sigma_i2D), VUnNormedReprojectionError.push_back(residuals[1] / sigma_i2D);
			}
		}

		//constant limb length
		if (sSke.nBAtimes > 0)
		{
			for (int cid = 0; cid < nLimbConnections; cid++)
			{
				int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
				if (IsValid3D(v3DPoints[j0]) && IsValid3D(v3DPoints[j1]))
				{
					ceres::CostFunction* cost_function = ConstantLimbLengthCost3DCeres::CreateAutoDiff(sSke.meanBoneLength[cid], sigma_iL);

					vector<double *> paras;  paras.push_back(&v3DPoints[j0].x), paras.push_back(&v3DPoints[j1].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VconstLimbError.push_back(0.5*Weights[1] * residuals[0] * residuals[0]);
				}
			}
		}

		//symmetry limb
		for (int cid = 0; cid < nSymLimbConnectionID; cid++)
		{
			int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
			if (IsValid3D(v3DPoints[j0]) && IsValid3D(v3DPoints[j1]) && IsValid3D(v3DPoints[j0_]) && IsValid3D(v3DPoints[j1_]))
			{
				if (j0 == j0_)
				{
					ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);

					vector<double *> paras; paras.push_back(&v3DPoints[j0].x), paras.push_back(&v3DPoints[j1].x), paras.push_back(&v3DPoints[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
				else
				{
					ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres::CreateAutoDiff(sigma_iL);

					vector<double *> paras; paras.push_back(&v3DPoints[j0].x), paras.push_back(&v3DPoints[j1].x), paras.push_back(&v3DPoints[j0_].x), paras.push_back(&v3DPoints[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
			}
		}

		if (sSke.sSke[0].vCidFidDid.size() > 0 && Weights[3] > 0.0)
		{
			int tempFid = sSke.sSke.back().refFid;
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				double actingSigma = sigma_iVel;
				if (jid == 9 || jid == 10 || jid == 15 || jid == 16) //allow hands and feet to move faster
					actingSigma = sigma_iVel2;

				//fit regarless of the point is valid or not
				if (IsValid3D(sSke.sSke.back().pt3d[jid]) && IsValid3D(v3DPoints[jid])) //temporal smoothing
				{
					ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres3::CreateAutoDiff(&sSke.sSke.back().pt3d[jid].x, ifps* tempFid, ifps * (tempFid + increF), actingSigma);
					vector<double *> paras; paras.push_back(&v3DPoints[jid].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VtemporalError.push_back(0.5*Weights[3] * residuals[0] * residuals[0]);
				}
			}
		}

		if (VreprojectionError.size() == 0)
			VreprojectionError.push_back(0);
		if (VUnNormedReprojectionError.size() == 0)
			VUnNormedReprojectionError.push_back(0);
		if (VconstLimbError.size() == 0)
			VconstLimbError.push_back(0);
		if (VsymLimbError.size() == 0)
			VsymLimbError.push_back(0);
		if (VtemporalError.size() == 0)
			VtemporalError.push_back(0);
		double reproSoS = MeanArray(VreprojectionError)*VreprojectionError.size(),
			unNormedRepro = MeanArray(VUnNormedReprojectionError),
			stdUnNormedRepro = sqrt(VarianceArray(VUnNormedReprojectionError, unNormedRepro)),
			maxUnNormedRepro = *std::max_element(VreprojectionError.begin(), VreprojectionError.end()), minUnNormedRePro = *std::min_element(VreprojectionError.begin(), VreprojectionError.end()),
			cLimbSoS = MeanArray(VconstLimbError)*VconstLimbError.size(), cLimb = sqrt(MeanArray(VconstLimbError)*2.0 / Weights[1]) / sigma_iL * real2SfM,
			sSkeleSoS = MeanArray(VsymLimbError)*VsymLimbError.size(), sSkele = sqrt(MeanArray(VsymLimbError)*2.0 / Weights[2]) / sigma_iL * real2SfM,
			motionCoSoS = MeanArray(VtemporalError)*VtemporalError.size(), motionCo = sqrt(MeanArray(VtemporalError)*2.0 / Weights[3]) / sigma_iVel * real2SfM;
		printLOG("After optim:\n");
		printLOG("Reprojection: normalized: %.3f  unnormalized mean: %.3f unnormalized std: %.3f max: %.3f min: %.3f \n", reproSoS, unNormedRepro, stdUnNormedRepro, maxUnNormedRepro, minUnNormedRePro);
		if (sSke.nBAtimes > 0)
			printLOG("Const Limb: normalized: %.3f  unnormalized: %.3f\n", cLimbSoS, cLimb);
		printLOG("Sym skeleton: normalized: %.3f  unnormalized: %.3f\n", sSkeleSoS, sSkele);
		if (sSke.sSke[0].vCidFidDid.size() > 0 && Weights[3] > 0.0)
			printLOG("Motion coherent: normalized: %.3f  unnormalized: %.3f\n", motionCoSoS, motionCo);
	}

	return 0;
}
bool triangulateNViewAlgebraic(double *vecT_frame_world, const std::vector<Point3d>& rays, Point3d& world_triangulated_point)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	for (int i = 0; i < rays.size(); i++)
	{
		Map< Matrix<double, 3, 4, RowMajor> >	T_frame_world(vecT_frame_world + 12 * i, 3, 4);
		//const Matrix3x4d T_frame_world = vecT_frame_world[i];
		double norm = sqrt(pow(rays[i].x, 2) + pow(rays[i].y, 2) + pow(rays[i].z, 2));
		const Vector3d norm_ray(rays[i].x / norm, rays[i].y / norm, rays[i].z / norm);

		const Matrix3x4d cost_term =
			(Matrix3d::Identity() - norm_ray * norm_ray.transpose()) * T_frame_world;
		design_matrix += cost_term.transpose() * cost_term;
	}

	Eigen::SelfAdjointEigenSolver<Matrix4d> eigen_solver(design_matrix);
	Vector4d world_triangulated = eigen_solver.eigenvectors().leftCols<1>();

	world_triangulated_point.x = world_triangulated(0) / world_triangulated(3);
	world_triangulated_point.y = world_triangulated(1) / world_triangulated(3);
	world_triangulated_point.z = world_triangulated(2) / world_triangulated(3);

	return eigen_solver.info() == Eigen::Success;
}
void TriangulationGuidedPeopleSearch(VideoData *VideoInfo, vector<int> &vCams, vector<int> &FrameMatch, vector<BodyImgAtrribute> *kpts, vector<Point3d> &v3DPoints, vector<int> &FoundViewMatch, vector<int> &FoundFrameMatch, vector<int> &FoundDetectionIdMatch, double triangulationThresh)
{
	int nKeyPoints = kpts[0][0].nPts;

	for (auto cid : vCams)
	{
		int lfid = FrameMatch[cid];
		int BestnValid = 0, bestDid = -1, nValid;
		double bestAvgErrI = 9e9, avgErrI;
		Point2d projected_pt;

		for (int did = 0; did < kpts[cid].size(); did++)
		{
			nValid = 0;
			avgErrI = 0;

			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				if (IsValid3D(v3DPoints[jid]) || kpts[cid][did].pt[jid].z > 0 || VideoInfo[cid].VideoInfo[lfid].valid == 1)
				{
					ProjectandDistort(v3DPoints[jid], &projected_pt, VideoInfo[cid].VideoInfo[lfid].P);
					double reproj_err = 0.5*(abs(kpts[cid][did].pt[jid].x - projected_pt.x) + abs(kpts[cid][did].pt[jid].y - projected_pt.y));
					if (reproj_err < 1.5*triangulationThresh)
						avgErrI += reproj_err, nValid++;
				}
			}
			if (nValid < nKeyPoints / 3)
				continue;

			avgErrI = avgErrI / (0.00001 + nValid);
			if (nValid > BestnValid)
				BestnValid = nValid, bestAvgErrI = avgErrI, bestDid = did;
		}

		//heuristic: relax the thresh if #points is large. --> useful to maintain tracklet (do not apply the same to Ransac since its goal is robustness)
		if (bestAvgErrI < 1.5* triangulationThresh && BestnValid > 2 * nKeyPoints / 3)
			FoundViewMatch.push_back(cid), FoundFrameMatch.push_back(lfid), FoundDetectionIdMatch.push_back(bestDid);
		else if (bestAvgErrI < 1.2* triangulationThresh && BestnValid >  nKeyPoints / 2)
			FoundViewMatch.push_back(cid), FoundFrameMatch.push_back(lfid), FoundDetectionIdMatch.push_back(bestDid);
		else if (bestAvgErrI< triangulationThresh && BestnValid > nKeyPoints / 3)
			FoundViewMatch.push_back(cid), FoundFrameMatch.push_back(lfid), FoundDetectionIdMatch.push_back(bestDid);
	}

	return;
}
int TriangulateSkeletonRANSAC(VideoData *VideoInfo, vector<BodyImgAtrribute> *kpts, vector<int> &ViewMatch, vector<int> &FrameMatch, vector<int> &DetectionIdMatch, vector<int> &bestAsso, vector<Point3d> &v3DPoints, vector<double> &vReProjErr, double triangulationThresh, int RanSac_iterMax)
{
	int nKeyPoints = kpts[0][0].nPts;

	int nmatches = (int)ViewMatch.size();
	if (nmatches < 2)
		return -1;

	Point2d projected_pt;
	vector<int> vcid;
	vector<Point2d> hypo_vPts2d;
	vector<Point3d> hypo_vPts;
	vector<Matrix3x4d> vPs;
	vector<double> vPs_;

	v3DPoints.resize(nKeyPoints); vReProjErr.resize(nKeyPoints);
	vector<Point3d> v3DPoints_(nKeyPoints);

	vector<int> randId;
	for (int ii = 0; ii < nmatches; ii++)
		randId.push_back(ii);

	bool bruteforce2 = false, bruteforce3 = false;
	Combination CamCom2(nmatches, min(2, nmatches));
	int nComs2 = CamCom2.total_com;
	int *allCamCom2, *ComI2 = new int[min(2, nmatches)];
	if (nComs2 < RanSac_iterMax)
	{
		bruteforce2 = true;
		allCamCom2 = new int[min(2, nmatches) * nComs2];
		CamCom2.All_Com(allCamCom2);
	}
	Combination CamCom3(nmatches, min(3, nmatches));
	int nComs3 = CamCom3.total_com;
	int *allCamCom3, *ComI3 = new int[min(3, nmatches)];
	if (nComs3 < RanSac_iterMax)
	{
		bruteforce3 = true;
		allCamCom3 = new int[min(3, nmatches) * nComs3];
		CamCom3.All_Com(allCamCom3);
	}


	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

							//Start doing RANSAC
	int nValid, bestnValid = 0;
	double minAvgErr = 9e9, avgErr, reprojectionError;
	for (int RanSac_iter = 0; RanSac_iter < min(nComs3, RanSac_iterMax); RanSac_iter++)
	{
		std::shuffle(randId.begin(), randId.end(), gen);
		for (int ii = 0; ii < min(2, nmatches); ii++)
			ComI2[ii] = randId[ii];
		for (int ii = 0; ii < min(3, nmatches); ii++)
			ComI3[ii] = randId[ii];

		if (bruteforce2 && RanSac_iter < nComs2)
			for (int ii = 0; ii < min(2, nmatches); ii++)
				ComI2[ii] = allCamCom2[min(2, nmatches)*RanSac_iter + ii];
		if (bruteforce3 && RanSac_iter < nComs3)
			for (int ii = 0; ii < min(3, nmatches); ii++)
				ComI3[ii] = allCamCom3[min(3, nmatches)*RanSac_iter + ii];

		//Try 3views triangulation first then 2 views
		for (int iter = 0; iter < 2; iter++)
		{
			if (RanSac_iter >= nComs2)
				continue; //no more 2 views
			if (RanSac_iter >= nComs3)
				continue; //no more 3 views

			int minNCams = 3 - iter;
			nValid = 0, avgErr = 0;
			v3DPoints_.clear();
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				v3DPoints_.emplace_back(0, 0, 0);
				vcid.clear(), vPs.clear(), vPs_.clear(), hypo_vPts.clear(), hypo_vPts2d.clear();
				for (int ii = 0; ii < min(minNCams, nmatches); ii++)
				{
					int cid, lfid, detectionId;
					if (iter == 0)
						cid = ViewMatch[ComI3[ii]], lfid = FrameMatch[ComI3[ii]], detectionId = DetectionIdMatch[ComI3[ii]];
					else
						cid = ViewMatch[ComI2[ii]], lfid = FrameMatch[ComI2[ii]], detectionId = DetectionIdMatch[ComI2[ii]];

					if (kpts[cid][detectionId].pt[jid].z != 0.0)
					{
						vcid.push_back(cid);
						Matrix3x4d P;
						for (int j = 0; j < 3; j++)
							for (int i = 0; i < 4; i++)
								P(j, i) = VideoInfo[cid].VideoInfo[lfid].P[j * 4 + i], vPs_.emplace_back(P(j, i));
						vPs.push_back(P);
						hypo_vPts.emplace_back(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y, 1);
						hypo_vPts2d.emplace_back(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y);
					}
				}
				sort(vcid.begin(), vcid.end());
				std::vector<int>::iterator it = unique(vcid.begin(), vcid.end());
				if (vcid.size() != std::distance(vcid.begin(), it))
					continue; //using the same cameras in the inital selection
				if (hypo_vPts.size() < min(minNCams, nmatches))
					continue;

				Point3d triangulated_hpoint;
				triangulateNViewAlgebraic(&vPs_[0], hypo_vPts, triangulated_hpoint);
				NviewTriangulationNonLinear(&vPs_[0], &hypo_vPts2d[0], &triangulated_hpoint, &reprojectionError, hypo_vPts2d.size(), 1);
				v3DPoints_[jid] = triangulated_hpoint;

				int nValidI = 0; double avgErrI = 0;
				for (int ii = 0; ii < hypo_vPts2d.size(); ii++)
				{
					ProjectandDistort(triangulated_hpoint, &projected_pt, &vPs_[12 * ii]);
					double reproj_err = 0.5*(abs(hypo_vPts2d[ii].x - projected_pt.x) + abs(hypo_vPts2d[ii].y - projected_pt.y));
					avgErrI += reproj_err, nValidI++;
				}
				avgErrI = avgErrI / (0.00001 + nValidI);
				if (avgErrI < triangulationThresh)
					avgErr += avgErrI, nValid++;
			}
			avgErr = nValid == 0 ? -1 : avgErr / nValid;
			if (nValid >= nKeyPoints / 4 && avgErr < triangulationThresh)
				break;
		}

		//project to other views and test
		vector<int> goodAssso;
		if (nValid >= nKeyPoints / 4 && avgErr < triangulationThresh)
		{
			nValid = 0;
			for (int mid = 0; mid < nmatches; mid++)
			{
				int nValidI = 0; double avgErrI = 0;
				int cid = ViewMatch[mid], lfid = FrameMatch[mid], detectionId = DetectionIdMatch[mid];
				for (int jid = 0; jid < nKeyPoints; jid++)
				{
					if (abs(v3DPoints_[jid].x) + abs(v3DPoints_[jid].y) + abs(v3DPoints_[jid].z) > 0 || VideoInfo[cid].VideoInfo[lfid].valid == 1)
					{
						ProjectandDistort(v3DPoints_[jid], &projected_pt, VideoInfo[cid].VideoInfo[lfid].P);
						double reproj_err = 0.5*(abs(kpts[cid][detectionId].pt[jid].x - projected_pt.x) + abs(kpts[cid][detectionId].pt[jid].y - projected_pt.y));
						if (reproj_err < 1.5*triangulationThresh)
							avgErrI += reproj_err, nValidI++;
					}
				}
				avgErrI = avgErrI / (0.00001 + nValidI);
				if (avgErrI< triangulationThresh && nValidI > nKeyPoints / 3)
					goodAssso.push_back(mid), nValid += nValidI;
			}

			//retriangulate with nViews at once
			if (nValid > bestnValid && goodAssso.size() > 1)
			{
				bestAsso = goodAssso;
				minAvgErr = 0.0, bestnValid = 0;
				v3DPoints.clear();
				for (int jid = 0; jid < nKeyPoints; jid++)
				{
					vPs.clear(), vPs_.clear(), hypo_vPts.clear(), hypo_vPts2d.clear();
					v3DPoints.emplace_back(0, 0, 0);
					vReProjErr[jid] = 1000;

					for (auto ii : bestAsso)
					{
						int cid = ViewMatch[ii], lfid = FrameMatch[ii], detectionId = DetectionIdMatch[ii];
						if (kpts[cid][detectionId].pt[jid].z != 0.0)
						{
							hypo_vPts.push_back(Point3d(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y, 1));
							hypo_vPts2d.push_back(Point2d(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y));

							Matrix3x4d P;
							for (int j = 0; j < 3; j++)
								for (int i = 0; i < 4; i++)
									P(j, i) = VideoInfo[cid].VideoInfo[lfid].P[j * 4 + i], vPs_.emplace_back(P(j, i));
							vPs.push_back(P);
						}
					}
					if (hypo_vPts.size() < 2)
						continue;

					Point3d triangulated_hpoint;
					triangulateNViewAlgebraic(&vPs_[0], hypo_vPts, triangulated_hpoint);
					NviewTriangulationNonLinear(&vPs_[0], &hypo_vPts2d[0], &triangulated_hpoint, &reprojectionError, hypo_vPts2d.size(), 1);
					v3DPoints[jid] = triangulated_hpoint;

					avgErr = 0; nValid = 0;
					for (size_t ii = 0; ii < bestAsso.size(); ii++)
					{
						ProjectandDistort(triangulated_hpoint, &projected_pt, &vPs_[12 * ii]);
						double reproj_err = 0.5*(abs(hypo_vPts[ii].x - projected_pt.x) + abs(hypo_vPts[ii].y - projected_pt.y));
						if (reproj_err < triangulationThresh)
						{
							avgErr += reproj_err, nValid++;
							minAvgErr += reproj_err, bestnValid++;
						}
					}
					vReProjErr[jid] = nValid == 0 ? -1 : avgErr / nValid;
				}
				minAvgErr = minAvgErr / (0.00001 + bestnValid);
			}
		}

		//Ransac found good enough inliers. Terminate
		if (bestnValid > 0.8*nmatches*nKeyPoints)
			break;
	}

	if (bruteforce2)
		delete[]allCamCom2;
	if (bruteforce3)
		delete[]allCamCom3;
	delete[]ComI3, delete[]ComI2;
	return 0;
}
int TriangulateSkeletonRANSAC(VideoData *VideoInfo, sHumanSkeleton3D  &sSke, vector<BodyImgAtrribute> *kpts, vector<int> &ViewMatch, vector<int> &FrameMatch, vector<int> &DetectionIdMatch, vector<int> &bestAsso, vector<Point3d> &v3DPoints, vector<double> &vReProjErr, double triangulationThresh, int RanSac_iterMax, int AtLeastThree4Ransac, int LossType, double *Weights, double *iSigma, int increF, double ifps, double real2SfM)
{
	int nKeyPoints = kpts[0][0].nPts;

	int nmatches = (int)ViewMatch.size();
	if (nmatches < 2)
		return -1;

	Point2d projected_pt;
	vector<int> vcid;
	vector<Point2d> hypo_vPts2d;
	vector<Point3d> hypo_vPts;
	vector<Matrix3x4d> vPs;
	vector<double> vPs_;

	vector<Point2d> *Vhypo_vPts2d = new vector<Point2d>[nKeyPoints];
	vector<double> *VPs = new vector<double>[nKeyPoints];

	v3DPoints.resize(nKeyPoints); vReProjErr.resize(nKeyPoints);
	vector<Point3d> v3DPoints_(nKeyPoints);

	vector<int> randId;
	for (int ii = 0; ii < nmatches; ii++)
		randId.push_back(ii);

	bool bruteforce2 = false, bruteforce3 = false;
	Combination CamCom2(nmatches, min(2, nmatches));
	int nComs2 = CamCom2.total_com;
	int *allCamCom2, *ComI2 = new int[min(2, nmatches)];
	if (nComs2 < RanSac_iterMax)
	{
		bruteforce2 = true;
		allCamCom2 = new int[min(2, nmatches) * nComs2];
		CamCom2.All_Com(allCamCom2);
	}
	Combination CamCom3(nmatches, min(3, nmatches));
	int nComs3 = CamCom3.total_com;
	int *allCamCom3, *ComI3 = new int[min(3, nmatches)];
	if (nComs3 < RanSac_iterMax)
	{
		bruteforce3 = true;
		allCamCom3 = new int[min(3, nmatches) * nComs3];
		CamCom3.All_Com(allCamCom3);
	}


	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

							//Start doing RANSAC
	int nValid, bestnValid = 0;
	double minAvgErr = 9e9, avgErr;
	RanSac_iterMax = max(nComs2, nComs3) < 30 ? max(nComs2, nComs3) : min(nComs3, RanSac_iterMax);
	for (int RanSac_iter = 0; RanSac_iter < RanSac_iterMax; RanSac_iter++)
	{
		std::shuffle(randId.begin(), randId.end(), gen);
		for (int ii = 0; ii < min(2, nmatches); ii++)
			ComI2[ii] = randId[ii];
		for (int ii = 0; ii < min(3, nmatches); ii++)
			ComI3[ii] = randId[ii];

		if (bruteforce2 && RanSac_iter < nComs2)
			for (int ii = 0; ii < min(2, nmatches); ii++)
				ComI2[ii] = allCamCom2[min(2, nmatches)*RanSac_iter + ii];
		if (bruteforce3 && RanSac_iter < nComs3)
			for (int ii = 0; ii < min(3, nmatches); ii++)
				ComI3[ii] = allCamCom3[min(3, nmatches)*RanSac_iter + ii];

		//Try 3views triangulation first then 2 views
		for (int iter = 0; iter < 2; iter++)
		{
			if (iter == 0 && RanSac_iter >= nComs3)
				continue; //no more 3 views
			if (RanSac_iter >= nComs2)
				continue; //no more 2 views			

			int minNCams = 3 - iter;
			if (AtLeastThree4Ransac && minNCams == 2)
				continue;

			nValid = 0, avgErr = 0;
			v3DPoints_.clear();
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				v3DPoints_.emplace_back(0, 0, 0);
				Vhypo_vPts2d[jid].clear(), VPs[jid].clear();
				vcid.clear(), vPs.clear(), vPs_.clear(), hypo_vPts.clear(), hypo_vPts2d.clear();
				for (int ii = 0; ii < min(minNCams, nmatches); ii++)
				{
					int cid, lfid, detectionId;
					if (iter == 0)
						cid = ViewMatch[ComI3[ii]], lfid = FrameMatch[ComI3[ii]], detectionId = DetectionIdMatch[ComI3[ii]];
					else
						cid = ViewMatch[ComI2[ii]], lfid = FrameMatch[ComI2[ii]], detectionId = DetectionIdMatch[ComI2[ii]];

					if (kpts[cid][detectionId].pt[jid].z > 0.0)
					{
						vcid.push_back(cid);
						Matrix3x4d P;
						for (int j = 0; j < 3; j++)
						{
							for (int i = 0; i < 4; i++)
							{
								P(j, i) = VideoInfo[cid].VideoInfo[lfid].P[j * 4 + i], vPs_.emplace_back(P(j, i));
								VPs[jid].emplace_back(P(j, i));
							}
						}
						vPs.push_back(P);
						hypo_vPts.emplace_back(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y, 1);
						hypo_vPts2d.emplace_back(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y);
						Vhypo_vPts2d[jid].emplace_back(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y);
					}
				}
				sort(vcid.begin(), vcid.end());
				std::vector<int>::iterator it = unique(vcid.begin(), vcid.end());
				if (vcid.size() != std::distance(vcid.begin(), it))
					continue; //using the same cameras in the inital selection
				if (hypo_vPts.size() < min(minNCams, nmatches))
					continue;

				Point3d triangulated_hpoint;
				triangulateNViewAlgebraic(&vPs_[0], hypo_vPts, triangulated_hpoint);
				//NviewTriangulationNonLinear(&vPs_[0], &hypo_vPts2d[0], &triangulated_hpoint, &reprojectionError, hypo_vPts2d.size(), 1);
				v3DPoints_[jid] = triangulated_hpoint;
			}

			PerFrameSkeleton3DBundleAdjustment(sSke, v3DPoints_, VPs, Vhypo_vPts2d, LossType, Weights, iSigma, increF, ifps, real2SfM, true);
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				int nValidI = 0; double avgErrI = 0;
				for (int ii = 0; ii < Vhypo_vPts2d[jid].size(); ii++)
				{
					ProjectandDistort(v3DPoints_[jid], &projected_pt, &VPs[jid][12 * ii]);
					double reproj_err = 0.5*(abs(Vhypo_vPts2d[jid][ii].x - projected_pt.x) + abs(Vhypo_vPts2d[jid][ii].y - projected_pt.y));
					avgErrI += reproj_err, nValidI++;
				}
				avgErrI = avgErrI / (0.00001 + nValidI);
				if (avgErrI < triangulationThresh)
					avgErr += avgErrI, nValid++;
			}

			avgErr = nValid == 0 ? -1 : avgErr / nValid;
			if (nValid >= nKeyPoints / 4 && avgErr < triangulationThresh)
				break;
		}

		//project to other views and test
		vector<int> goodAssso;
		if (nValid >= nKeyPoints / 4 && avgErr < triangulationThresh)
		{
			nValid = 0;
			for (int mid = 0; mid < nmatches; mid++)
			{
				int nValidI = 0; double avgErrI = 0;
				int cid = ViewMatch[mid], lfid = FrameMatch[mid], detectionId = DetectionIdMatch[mid];
				for (int jid = 0; jid < nKeyPoints; jid++)
				{
					if (kpts[cid][detectionId].pt[jid].z > 0 && IsValid3D(v3DPoints_[jid]) && VideoInfo[cid].VideoInfo[lfid].valid == 1)
					{
						ProjectandDistort(v3DPoints_[jid], &projected_pt, VideoInfo[cid].VideoInfo[lfid].P);
						double reproj_err = 0.5*(abs(kpts[cid][detectionId].pt[jid].x - projected_pt.x) + abs(kpts[cid][detectionId].pt[jid].y - projected_pt.y));
						if (reproj_err < 1.5*triangulationThresh)
							avgErrI += reproj_err, nValidI++;
					}
				}
				avgErrI = avgErrI / (0.00001 + nValidI);
				if (avgErrI< triangulationThresh && nValidI > nKeyPoints / 3)
					goodAssso.push_back(mid), nValid += nValidI;
			}

			//retriangulate with nViews at once
			if (nValid > bestnValid && goodAssso.size() > 1 + AtLeastThree4Ransac)
			{
				bestAsso = goodAssso;
				v3DPoints.clear();
				for (int jid = 0; jid < nKeyPoints; jid++)
				{
					Vhypo_vPts2d[jid].clear(), VPs[jid].clear();
					vPs.clear(), vPs_.clear(), hypo_vPts.clear(), hypo_vPts2d.clear();
					v3DPoints.emplace_back(0, 0, 0);
					vReProjErr[jid] = 1000;

					for (auto ii : bestAsso)
					{
						int cid = ViewMatch[ii], lfid = FrameMatch[ii], detectionId = DetectionIdMatch[ii];
						if (kpts[cid][detectionId].pt[jid].z > 0.0)
						{
							hypo_vPts.push_back(Point3d(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y, 1));
							hypo_vPts2d.push_back(Point2d(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y));
							Vhypo_vPts2d[jid].emplace_back(kpts[cid][detectionId].pt[jid].x, kpts[cid][detectionId].pt[jid].y);

							Matrix3x4d P;
							for (int j = 0; j < 3; j++)
							{
								for (int i = 0; i < 4; i++)
								{
									P(j, i) = VideoInfo[cid].VideoInfo[lfid].P[j * 4 + i], vPs_.emplace_back(P(j, i));
									VPs[jid].emplace_back(P(j, i));
								}
							}
							vPs.push_back(P);
						}
					}
					if (hypo_vPts.size() < 2 + AtLeastThree4Ransac)
						continue;

					Point3d triangulated_hpoint;
					triangulateNViewAlgebraic(&vPs_[0], hypo_vPts, triangulated_hpoint);
					//NviewTriangulationNonLinear(&vPs_[0], &hypo_vPts2d[0], &triangulated_hpoint, &reprojectionError, hypo_vPts2d.size(), 1);
					v3DPoints[jid] = triangulated_hpoint;
				}

				PerFrameSkeleton3DBundleAdjustment(sSke, v3DPoints, VPs, Vhypo_vPts2d, LossType, Weights, iSigma, increF, ifps, real2SfM, true);

				minAvgErr = 0.0, bestnValid = 0;
				for (int jid = 0; jid < nKeyPoints; jid++)
				{
					avgErr = 0; nValid = 0;
					for (int ii = 0; ii < Vhypo_vPts2d[jid].size(); ii++)
					{
						ProjectandDistort(v3DPoints[jid], &projected_pt, &VPs[jid][12 * ii]);
						double reproj_err = 0.5*(abs(Vhypo_vPts2d[jid][ii].x - projected_pt.x) + abs(Vhypo_vPts2d[jid][ii].y - projected_pt.y));
						if (reproj_err < triangulationThresh)
						{
							avgErr += reproj_err, nValid++;
							minAvgErr += reproj_err, bestnValid++;
						}
					}
					vReProjErr[jid] = nValid == 0 ? -1 : avgErr / nValid;
				}
				minAvgErr = minAvgErr / (0.00001 + bestnValid);
			}
		}

		//Ransac found good enough inliers. Terminate
		if (bestnValid > 0.8*nmatches*nKeyPoints)
			break;
	}

	if (bruteforce2)
		delete[]allCamCom2;
	if (bruteforce3)
		delete[]allCamCom3;
	delete[]ComI3, delete[]ComI2;
	return 0;
}
int WindowSkeleton3DBundleAdjustment(VideoData *VideoInfo, int nCams, sHumanSkeleton3D &sSke, int distortionCorrected, int LossType, double *Weights, double *iSigma, double real2SfM, double ifps, int reliableTrackingRange, bool silent)
{
	int nKeyPoints = sSke.sSke[0].nPts;

	//Weight = [ const limb length, symmetric skeleton, temporal]. It is  helpful for insightful weight setting if metric unit (mm) is used
	double sigma_i2D = iSigma[0], sigma_iL = iSigma[1] * real2SfM, sigma_iVel = iSigma[2] * real2SfM, sigma_iVel2 = iSigma[3] * real2SfM; //also convert physical sigma to sfm scale sigma

	 //For COCO 18-points 
	const int nLimbConnections = 17, nSymLimbConnectionID = 9;
	Point2i LimbConnectionID[nLimbConnections] = { Point2i(0, 1), Point2i(1, 2), Point2i(2, 3), Point2i(3, 4), Point2i(1, 5), Point2i(5, 6), Point2i(6, 7),
		Point2i(1, 8), Point2i(8, 9), Point2i(9, 10), Point2i(1, 11), Point2i(11, 12), Point2i(12, 13), Point2i(0, 14), Point2i(0, 15), Point2i(14, 16), Point2i(15, 17) };
	Vector4i SymLimbConnectionID[nSymLimbConnectionID] = { Vector4i(1, 2, 1, 5), Vector4i(2, 3, 5, 6), Vector4i(3, 4, 6, 7), Vector4i(1, 8, 1, 11), Vector4i(8, 9, 11, 12), Vector4i(9, 10, 12, 13) , Vector4i(0, 14, 0, 15), Vector4i(14, 16, 15, 17), Vector4i(0, 16, 0, 17) }; //no eyes, ears since they are unreliable

	ceres::Problem problem;
	ceres::LossFunction *loss_funcion = 0;
	if (LossType == 1) //Huber
		loss_funcion = new ceres::HuberLoss(1.0);

	double residuals[3], rho[3];

	vector<double> *vlimblength = new vector<double>[nLimbConnections];
	for (size_t inst = 0; inst < sSke.sSke.size(); inst++)
	{
		for (int cid = 0; cid < nLimbConnections; cid++)
		{
			int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
			if (IsValid3D(sSke.sSke[inst].pt3d[j0]) && IsValid3D(sSke.sSke[inst].pt3d[j1]))
				vlimblength[cid].push_back(norm(sSke.sSke[inst].pt3d[j0] - sSke.sSke[inst].pt3d[j1]));
		}
	}
	for (int cid = 0; cid < nLimbConnections; cid++)
	{
		if (vlimblength[cid].size() == 0)
			sSke.meanBoneLength[cid] = 0.0;
		else
			sSke.meanBoneLength[cid] = MedianArray(vlimblength[cid]);
	}
	delete[]vlimblength;

	vector<double> VreprojectionError, VUnNormedReprojectionError, VconstLimbError, VsymLimbError, VtemporalError;
	for (size_t inst = 0; inst < sSke.sSke.size(); inst++)
	{
		//reprojection error
		for (size_t ii = 0; ii < sSke.sSke[inst].vCidFidDid.size(); ii++)
		{
			int cid = sSke.sSke[inst].vCidFidDid[ii].x, rfid = sSke.sSke[inst].vCidFidDid[ii].y;
			CameraData *camI = VideoInfo[cid].VideoInfo;

			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				Point2d uv(sSke.sSke[inst].vPt2D[jid][ii].x, sSke.sSke[inst].vPt2D[jid][ii].y);
				if (uv.x == 0 || uv.y == 0)
					continue;

				ceres::LossFunction* robust_loss = new ceres::HuberLoss(1);
				ceres::LossFunction* weightLoss = new ceres::ScaledLoss(robust_loss, Weights[0], ceres::TAKE_OWNERSHIP);
				ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(camI[rfid].P, uv.x, uv.y, sigma_i2D);
				problem.AddResidualBlock(cost_function, weightLoss, &sSke.sSke[inst].pt3d[jid].x);

				vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[jid].x);
				cost_function->Evaluate(&paras[0], residuals, NULL);
				robust_loss->Evaluate(residuals[0] * residuals[0] + residuals[1] * residuals[1], rho);
				VreprojectionError.push_back(Weights[0] * 0.5*rho[0]);
				VUnNormedReprojectionError.push_back(residuals[0] / sigma_i2D), VUnNormedReprojectionError.push_back(residuals[1] / sigma_i2D);
			}
		}

		//constant limb length
		for (int cid = 0; cid < nLimbConnections; cid++)
		{
			int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
			if (IsValid3D(sSke.sSke[inst].pt3d[j0]) && IsValid3D(sSke.sSke[inst].pt3d[j1]))
			{
				ceres::CostFunction* cost_function = ConstantLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);
				ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[1], ceres::TAKE_OWNERSHIP);
				problem.AddResidualBlock(cost_function, ScaleLoss, &sSke.meanBoneLength[cid], &sSke.sSke[inst].pt3d[j0].x, &sSke.sSke[inst].pt3d[j1].x);

				vector<double *> paras; paras.push_back(&sSke.meanBoneLength[cid]), paras.push_back(&sSke.sSke[inst].pt3d[j0].x), paras.push_back(&sSke.sSke[inst].pt3d[j1].x);
				cost_function->Evaluate(&paras[0], residuals, NULL);
				VconstLimbError.push_back(0.5*Weights[1] * residuals[0] * residuals[0]);
			}
		}

		//symmetry limb
		for (int cid = 0; cid < nSymLimbConnectionID; cid++)
		{
			int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
			if (IsValid3D(sSke.sSke[inst].pt3d[j0]) && IsValid3D(sSke.sSke[inst].pt3d[j1]) && IsValid3D(sSke.sSke[inst].pt3d[j0_]) && IsValid3D(sSke.sSke[inst].pt3d[j1_]))
			{
				if (j0 == j0_)
				{
					ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);
					ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[2], ceres::TAKE_OWNERSHIP);
					problem.AddResidualBlock(cost_function, ScaleLoss, &sSke.sSke[inst].pt3d[j0].x, &sSke.sSke[inst].pt3d[j1].x, &sSke.sSke[inst].pt3d[j1_].x);

					vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[j0].x), paras.push_back(&sSke.sSke[inst].pt3d[j1].x), paras.push_back(&sSke.sSke[inst].pt3d[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
				else
				{
					ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres::CreateAutoDiff(sigma_iL);
					ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[2], ceres::TAKE_OWNERSHIP);
					problem.AddResidualBlock(cost_function, ScaleLoss, &sSke.sSke[inst].pt3d[j0].x, &sSke.sSke[inst].pt3d[j1].x, &sSke.sSke[inst].pt3d[j0_].x, &sSke.sSke[inst].pt3d[j1_].x);

					vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[j0].x), paras.push_back(&sSke.sSke[inst].pt3d[j1].x), paras.push_back(&sSke.sSke[inst].pt3d[j0_].x), paras.push_back(&sSke.sSke[inst].pt3d[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
			}
		}
	}
	if (Weights[3] > 0.0)
	{
		for (size_t inst = 0; inst < sSke.sSke.size() - 1; inst++)
		{
			if (sSke.sSke[inst].valid == 0 || sSke.sSke[inst + 1].valid == 0)
				continue;
			int tempFid1 = sSke.sSke[inst].refFid, tempFid2 = sSke.sSke[inst + 1].refFid;
			if (tempFid2 - tempFid1 > 1)
				continue;
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				double actingSigma = sigma_iVel;
				if (jid == 9 || jid == 10 || jid == 15 || jid == 16) //allow hands and feet to move faster
					actingSigma = sigma_iVel2;

				//fit regarless of the point is valid or not
				if (sSke.sSke.size() > reliableTrackingRange || (IsValid3D(sSke.sSke[inst].pt3d[jid]) && IsValid3D(sSke.sSke[inst + 1].pt3d[jid]))) //temporal smoothing
				{
					ceres::LossFunction *ScaleLoss = new ceres::ScaledLoss(NULL, Weights[3], ceres::TAKE_OWNERSHIP);
					ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres2::CreateAutoDiff(ifps* tempFid1, ifps * tempFid2, actingSigma);
					problem.AddResidualBlock(cost_function, ScaleLoss, &sSke.sSke[inst].pt3d[jid].x, &sSke.sSke[inst + 1].pt3d[jid].x);

					vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[jid].x), paras.push_back(&sSke.sSke[inst + 1].pt3d[jid].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VtemporalError.push_back(0.5*Weights[3] * residuals[0] * residuals[0]);
				}
			}
		}
	}

	if (VreprojectionError.size() == 0)
		VreprojectionError.push_back(0);
	if (VUnNormedReprojectionError.size() == 0)
		VUnNormedReprojectionError.push_back(0);
	if (VconstLimbError.size() == 0)
		VconstLimbError.push_back(0);
	if (VsymLimbError.size() == 0)
		VsymLimbError.push_back(0);
	if (VtemporalError.size() == 0)
		VtemporalError.push_back(0);

	double reproSoS = MeanArray(VreprojectionError)*VreprojectionError.size(),
		unNormedRepro = MeanArray(VUnNormedReprojectionError),
		stdUnNormedRepro = sqrt(VarianceArray(VUnNormedReprojectionError, unNormedRepro)),
		maxUnNormedRepro = *std::max_element(VreprojectionError.begin(), VreprojectionError.end()), minUnNormedRePro = *std::min_element(VreprojectionError.begin(), VreprojectionError.end()),
		cLimbSoS = MeanArray(VconstLimbError)*VconstLimbError.size(), cLimb = sqrt(MeanArray(VconstLimbError)*2.0 / Weights[1]) / sigma_iL * real2SfM,
		sSkeleSoS = MeanArray(VsymLimbError)*VsymLimbError.size(), sSkele = sqrt(MeanArray(VsymLimbError)*2.0 / Weights[2]) / sigma_iL * real2SfM,
		motionCoSoS = MeanArray(VtemporalError)*VtemporalError.size(), motionCo = sqrt(MeanArray(VtemporalError)*2.0 / Weights[3]) / sigma_iVel * real2SfM;
	printLOG("Before optim:\n");
	printLOG("Reprojection: normalized: %.3f  unnormalized mean: %.3f unnormalized std: %.3f max: %.3f min: %.3f \n", reproSoS, unNormedRepro, stdUnNormedRepro, maxUnNormedRepro, minUnNormedRePro);
	printLOG("Const Limb: normalized: %.3f  unnormalized: %.3f\n", cLimbSoS, cLimb);
	printLOG("Sym skeleton: normalized: %.3f  unnormalized: %.3f\n", sSkeleSoS, sSkele);
	printLOG("Motion coherent: normalized: %.3f  unnormalized: %.3f\n", motionCoSoS, motionCo);

	ceres::Solver::Options options;
	if (sSke.sSke.size() > 50)
	{
		options.num_threads = omp_get_max_threads(); //jacobian eval
		options.num_linear_solver_threads = omp_get_max_threads(); //linear solver
	}
	else
	{
		options.num_threads = 2; //omp_get_max_threads(); //jacobian eval
		options.num_linear_solver_threads = 2; //omp_get_max_threads(); //linear solver
	}
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.use_nonmonotonic_steps = false;
	options.max_num_iterations = 50;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	VreprojectionError.clear(), VUnNormedReprojectionError.clear(), VconstLimbError.clear(), VsymLimbError.clear(), VtemporalError.clear();
	for (size_t inst = 0; inst < sSke.sSke.size(); inst++)
	{
		//reprojection error
		for (size_t ii = 0; ii < sSke.sSke[inst].vCidFidDid.size(); ii++)
		{
			int cid = sSke.sSke[inst].vCidFidDid[ii].x, rfid = sSke.sSke[inst].vCidFidDid[ii].y;
			CameraData *camI = VideoInfo[cid].VideoInfo;

			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				Point2d uv(sSke.sSke[inst].vPt2D[jid][ii].x, sSke.sSke[inst].vPt2D[jid][ii].y);
				if (uv.x == 0 || uv.y == 0)
					continue;

				ceres::LossFunction* robust_loss = new ceres::HuberLoss(1);
				ceres::CostFunction* cost_function = PinholeReprojectionErrorSimple_PointOnly::Create(camI[rfid].P, uv.x, uv.y, sigma_i2D);

				vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[jid].x);
				cost_function->Evaluate(&paras[0], residuals, NULL);
				robust_loss->Evaluate(residuals[0] * residuals[0] + residuals[1] * residuals[1], rho);
				VreprojectionError.push_back(Weights[0] * 0.5*rho[0]);
				VUnNormedReprojectionError.push_back(residuals[0] / sigma_i2D), VUnNormedReprojectionError.push_back(residuals[1] / sigma_i2D);
			}
		}

		//constant limb length
		for (int cid = 0; cid < nLimbConnections; cid++)
		{
			int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
			if (IsValid3D(sSke.sSke[inst].pt3d[j0]) && IsValid3D(sSke.sSke[inst].pt3d[j1]))
			{
				ceres::CostFunction* cost_function = ConstantLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);

				vector<double *> paras; paras.push_back(&sSke.meanBoneLength[cid]), paras.push_back(&sSke.sSke[inst].pt3d[j0].x), paras.push_back(&sSke.sSke[inst].pt3d[j1].x);
				cost_function->Evaluate(&paras[0], residuals, NULL);
				VconstLimbError.push_back(0.5*Weights[1] * residuals[0] * residuals[0]);
			}
		}

		//symmetry limb
		for (int cid = 0; cid < nSymLimbConnectionID; cid++)
		{
			int j0 = SymLimbConnectionID[cid](0), j1 = SymLimbConnectionID[cid](1), j0_ = SymLimbConnectionID[cid](2), j1_ = SymLimbConnectionID[cid](3);
			if (IsValid3D(sSke.sSke[inst].pt3d[j0]) && IsValid3D(sSke.sSke[inst].pt3d[j1]) && IsValid3D(sSke.sSke[inst].pt3d[j0_]) && IsValid3D(sSke.sSke[inst].pt3d[j1_]))
			{
				if (j0 == j0_)
				{
					ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres2::CreateAutoDiff(sigma_iL);

					vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[j0].x), paras.push_back(&sSke.sSke[inst].pt3d[j1].x), paras.push_back(&sSke.sSke[inst].pt3d[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
				else
				{
					ceres::CostFunction* cost_function = SymLimbLengthCost3DCeres::CreateAutoDiff(sigma_iL);

					vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[j0].x), paras.push_back(&sSke.sSke[inst].pt3d[j1].x), paras.push_back(&sSke.sSke[inst].pt3d[j0_].x), paras.push_back(&sSke.sSke[inst].pt3d[j1_].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VsymLimbError.push_back(0.5*Weights[2] * residuals[0] * residuals[0]);
				}
			}
		}
	}
	if (Weights[3] > 0.0)
	{
		for (size_t inst = 0; inst < sSke.sSke.size() - 1; inst++)
		{
			if (sSke.sSke[inst].valid == 0 || sSke.sSke[inst + 1].valid == 0)
				continue;
			int tempFid1 = sSke.sSke[inst].refFid, tempFid2 = sSke.sSke[inst + 1].refFid;
			if (tempFid2 - tempFid1 > 1)
				continue;
			for (int jid = 0; jid < nKeyPoints; jid++)
			{
				double actingSigma = sigma_iVel;
				if (jid == 9 || jid == 10 || jid == 15 || jid == 16) //allow hands and feet to move faster
					actingSigma = sigma_iVel2;

				//fit regarless of the point is valid or not
				if (sSke.sSke.size() > reliableTrackingRange || (IsValid3D(sSke.sSke[inst].pt3d[jid]) && IsValid3D(sSke.sSke[inst + 1].pt3d[jid]))) //temporal smoothing
				{
					ceres::CostFunction* cost_function = LeastMotionPriorCost3DCeres2::CreateAutoDiff(ifps* tempFid1, ifps * tempFid2, actingSigma);

					vector<double *> paras; paras.push_back(&sSke.sSke[inst].pt3d[jid].x), paras.push_back(&sSke.sSke[inst + 1].pt3d[jid].x);
					cost_function->Evaluate(&paras[0], residuals, NULL);
					VtemporalError.push_back(0.5*Weights[3] * residuals[0] * residuals[0]);
				}
			}
		}
	}

	if (VreprojectionError.size() == 0)
		VreprojectionError.push_back(0);
	if (VUnNormedReprojectionError.size() == 0)
		VUnNormedReprojectionError.push_back(0);
	if (VconstLimbError.size() == 0)
		VconstLimbError.push_back(0);
	if (VsymLimbError.size() == 0)
		VsymLimbError.push_back(0);
	if (VtemporalError.size() == 0)
		VtemporalError.push_back(0);
	reproSoS = MeanArray(VreprojectionError)*VreprojectionError.size(),
		unNormedRepro = MeanArray(VUnNormedReprojectionError),
		stdUnNormedRepro = sqrt(VarianceArray(VUnNormedReprojectionError, unNormedRepro)),
		maxUnNormedRepro = *std::max_element(VreprojectionError.begin(), VreprojectionError.end()), minUnNormedRePro = *std::min_element(VreprojectionError.begin(), VreprojectionError.end()),
		cLimbSoS = MeanArray(VconstLimbError)*VconstLimbError.size(), cLimb = sqrt(MeanArray(VconstLimbError)*2.0 / Weights[1]) / sigma_iL * real2SfM,
		sSkeleSoS = MeanArray(VsymLimbError)*VsymLimbError.size(), sSkele = sqrt(MeanArray(VsymLimbError)*2.0 / Weights[2]) / sigma_iL * real2SfM,
		motionCoSoS = MeanArray(VtemporalError)*VtemporalError.size(), motionCo = sqrt(MeanArray(VtemporalError)*2.0 / Weights[3]) / sigma_iVel * real2SfM;
	printLOG("After optim:\n");
	printLOG("Reprojection: normalized: %.3f  unnormalized mean: %.3f unnormalized std: %.3f max: %.3f min: %.3f \n", reproSoS, unNormedRepro, stdUnNormedRepro, maxUnNormedRepro, minUnNormedRePro);
	printLOG("Const Limb: normalized: %.3f  unnormalized: %.3f\n", cLimbSoS, cLimb);
	printLOG("Sym skeleton: normalized: %.3f  unnormalized: %.3f\n", sSkeleSoS, sSkele);
	printLOG("Motion coherent: normalized: %.3f  unnormalized: %.3f\n", motionCoSoS, motionCo);


	return 0;
}
int SimultaneousTrackingAndAssocation(char *Path, std::vector<int> &SelectedCamNames, int startF, int stopF, double triangulationThresh, int RanSac_iterMax, int AtLeastThree4Ransac, float max_ratio, float DescThresh, float BAratio, int BAfreq, int reliableTrackRange, double real2SfM, double *WeightsHumanSkeleton3D, double *iSigmaSkeleton, int nBodyKeyPoints = 18, int debug = 0)
{
	int cacheFreq = max(500, (stopF - startF) / 4);
	double ifps = 1.0 / 60;
	int winSize = 31, cvPyrLevel = 5;
	double flowThresh = 8.0;

	char Fname[512];
	sprintf(Fname, "%s/Skeleton_%d_%d/", Path, startF, stopF); makeDir(Fname);
	sprintf(Fname, "%s/Vis/3DTracklet/", Path); makeDir(Fname);

	int nCams = (int)SelectedCamNames.size();

	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	int selected, temp;
	double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
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

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
	{
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;
		VideoInfo[cid].fps = 1.0 / CamTimeInfo[cid].x;
		VideoInfo[cid].TimeOffset = CamTimeInfo[cid].y;
	}

	vector<BodyImgAtrribute> *Attri = new vector<BodyImgAtrribute>[nCams], *AttriOld = new vector<BodyImgAtrribute>[nCams];

	Mat Img1, Img2;
	vector<Mat> *ImgOldPyr = new vector<Mat>[nCams], *ImgNewPyr = new vector<Mat>[nCams];

	//omp_set_num_threads(omp_get_max_threads());

	bool silent = false;
	vector<Mat> *AllCamImages = new vector<Mat>[nCams];
	vector<Mat> maskImgs;
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname, "%s/%d/mask.jpg", Path, SelectedCamNames[cid]);
		if (!IsFileExist(Fname))
		{
			printLOG("Cannot find %s. Exit\n", Fname);
			return 1;
		}
		Img1 = imread(Fname, 0);
		maskImgs.push_back(Img1);
	}

	vector<sHumanSkeleton3D > allSkeleton; //skeleton xyz and state of activenesss

	/*{
		int refFid;
		Point3i cidfiddid; Point2d pt; Point2f cPose[25];
		sHumanSkeleton3D sSke;
		sSke.active = 1;

		sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, 0);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &refFid) != EOF)
		{
			HumanSkeleton3D Ske;
			Ske.refFid = refFid;
			Ske.nPts = nBodyKeyPoints;
			for (int jid = 0; jid < nBodyKeyPoints; jid++)
			{
				fscanf(fp, "%lf %lf %lf ", &Ske.pt3d[jid].x, &Ske.pt3d[jid].y, &Ske.pt3d[jid].z);
				fscanf(fp, "%d ", &Ske.nvis);
				for (size_t ii = 0; ii < Ske.nvis; ii++)
				{
					fscanf(fp, "%d %d %d %lf %lf ", &cidfiddid.x, &cidfiddid.y, &cidfiddid.z, &pt.x, &pt.y);
					if (jid == 0)
						Ske.vCidFidDid.push_back(cidfiddid);
					Ske.vPt2D[jid].push_back(pt);
				}
			}
			sSke.sSke.push_back(Ske);
			if (refFid >= 112)
				break;
		}
		fclose(fp);

		allSkeleton.push_back(sSke);

		for (int cid = 0; cid < nCams; cid++)
		{
			AttriOld[cid].clear();
			int rfid = refFid - (int)CamTimeInfo[cid].y;
			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[rfid].valid != 1)
				continue;

			ReadBodyKeyPointsAndDesc(Path, cid, rfid, AttriOld[cid]);

			for (int pid = 0; pid < AttriOld[cid].size(); pid++)
			{
				bool valid = false;
				for (int jid = 0; jid < nBodyKeyPoints && !valid; jid++)
				{
					if (AttriOld[cid][pid].pt[jid].z > 0.0 && AttriOld[cid][pid].pt[jid].x > 0 && AttriOld[cid][pid].pt[jid].x < camI[rfid].width - 1 && AttriOld[cid][pid].pt[jid].y>0 && AttriOld[cid][pid].pt[jid].y < camI[rfid].height - 1)
					{
						int x = (int)(AttriOld[cid][pid].pt[jid].x), y = (int)(AttriOld[cid][pid].pt[jid].y), w = maskImgs[cid].cols;
						if (maskImgs[cid].data[y*w + x] > 128)
							valid = true;
					}
				}
				if (!valid)
					for (int jid = 0; jid < nBodyKeyPoints; jid++)
						AttriOld[cid][pid].pt[jid] = Point3d(0, 0, 0);
			}


			for (int pid = 0; pid < AttriOld[cid].size(); pid++)
			{
				for (int jid = 0; jid < nBodyKeyPoints; jid++)
				{
					Point2d pt(AttriOld[cid][pid].pt[jid].x, AttriOld[cid][pid].pt[jid].y);
					if (AttriOld[cid][pid].pt[jid].z == 0.0)
						AttriOld[cid][pid].pt[jid] = Point3d(0, 0, 0);
					else
					{
						LensCorrectionPoint(&pt, camI[rfid].K, camI[rfid].distortion);
						AttriOld[cid][pid].pt[jid].x = pt.x, AttriOld[cid][pid].pt[jid].y = pt.y;
					}
				}
			}

			sprintf(Fname, "%s/%d/%.4d.jpg", Path, SelectedCamNames[cid], rfid);
			if (!IsFileExist(Fname))
			{
				printLOG("Cannot find %s. Exit\n", Fname);
				return 1;
			}
			Img1 = imread(Fname, 0);
			buildOpticalFlowPyramid(Img1, ImgOldPyr[cid], Size(winSize, winSize), cvPyrLevel, false);
		}
	}*/

	for (int refFid = startF; refFid <= stopF - 1; refFid++)
	{
		printLOG("\n********************\n%d: %zd total #people\n", refFid, allSkeleton.size());

		//read BodyKeyPoints and desc
		for (int cid = 0; cid < nCams; cid++)
		{
			Attri[cid].clear();
			int rfid = refFid - (int)CamTimeInfo[cid].y;
			CameraData *camI = VideoInfo[cid].VideoInfo;
			if (camI[rfid].valid != 1)
				continue;

			ReadBodyKeyPointsAndDesc(Path, cid, rfid, Attri[cid]);

			for (int pid = 0; pid < Attri[cid].size(); pid++)
			{
				bool valid = false;
				for (int jid = 0; jid < nBodyKeyPoints && !valid; jid++)
				{
					if (Attri[cid][pid].pt[jid].z > 0.0 && Attri[cid][pid].pt[jid].x > 0 && Attri[cid][pid].pt[jid].x < camI[rfid].width - 1 && Attri[cid][pid].pt[jid].y>0 && Attri[cid][pid].pt[jid].y < camI[rfid].height - 1)
					{
						int x = (int)(Attri[cid][pid].pt[jid].x), y = (int)(Attri[cid][pid].pt[jid].y), w = maskImgs[cid].cols;
						if (maskImgs[cid].data[y*w + x] > 128)
							valid = true;
					}
				}
				if (!valid)
					for (int jid = 0; jid < nBodyKeyPoints; jid++)
						Attri[cid][pid].pt[jid] = Point3d(0, 0, 0);
			}


			for (int pid = 0; pid < Attri[cid].size(); pid++)
			{
				for (int jid = 0; jid < nBodyKeyPoints; jid++)
				{
					Point2d pt(Attri[cid][pid].pt[jid].x, Attri[cid][pid].pt[jid].y);
					if (Attri[cid][pid].pt[jid].z == 0.0)
						Attri[cid][pid].pt[jid] = Point3d(0, 0, 0);
					else
					{
						LensCorrectionPoint(&pt, camI[rfid].K, camI[rfid].distortion);
						Attri[cid][pid].pt[jid].x = pt.x, Attri[cid][pid].pt[jid].y = pt.y;
					}
				}
			}

			sprintf(Fname, "%s/%d/%.4d.jpg", Path, SelectedCamNames[cid], rfid);
			if (!IsFileExist(Fname))
			{
				printLOG("Cannot find %s. Exit\n", Fname);
				return 1;
			}
			Img2 = imread(Fname, 0);
			buildOpticalFlowPyramid(Img2, ImgNewPyr[cid], Size(winSize, winSize), cvPyrLevel, false);

			if (debug == 2)
				AllCamImages[cid].push_back(Img2);
		}

		//Stage I: temporal: using perCamera tracklets to predict next assocation and Ransac them.
		vector<vector<Point2i> > vUsedCidDid;
		{
			printLOG("\n********************\nTemporal propagation and recon\n********************\n");
			HumanSkeleton3D *AllSkePerInstance = new HumanSkeleton3D[allSkeleton.size()];
			vector<double> *AllvReProjErrPerInstance = new vector<double>[allSkeleton.size()];

#pragma omp parallel for schedule(dynamic, 1)
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				if (allSkeleton[skeI].active == 1) //active
				{
#pragma omp critical
					printLOG("Continuing %d...%d frames tracked\n", skeI, allSkeleton[skeI].sSke.size());
					vector<Point3d> v3DPoints(nBodyKeyPoints);
					vector<int> bestAsso, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, oPredictedViewMatch, oPredictedFrameMatch, oPredictedDetectionIdMatch;

					//Forward prediction via intra-view tracking -> Ransac
					for (size_t ii = 0; ii < allSkeleton[skeI].sSke.back().vCidFidDid.size(); ii++)
					{
						int cid = allSkeleton[skeI].sSke.back().vCidFidDid[ii].x, rfid = refFid - (int)CamTimeInfo[cid].y, did = allSkeleton[skeI].sSke.back().vCidFidDid[ii].z; //current 2d projection
						vector<Point2i> matches = MatchTwoFrame_BodyKeyPoints_LK(VideoInfo[cid].VideoInfo[rfid], ImgOldPyr[cid], ImgNewPyr[cid], AttriOld[cid], Attri[cid], flowThresh, 0.2, true);

						for (size_t jj = 0; jj < matches.size(); jj++)
						{
							if (matches[jj].x == did)
							{
								PredictedViewMatch.push_back(cid);
								PredictedFrameMatch.push_back(rfid);
								PredictedDetectionIdMatch.push_back(matches[jj].y);
							}
						}
					}
					bestAsso.clear(), AllvReProjErrPerInstance[skeI].clear(), v3DPoints.clear();
					//TriangulateSkeletonRANSAC(VideoInfo, Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI], triangulationThresh, RanSac_iterMax);
					TriangulateSkeletonRANSAC(VideoInfo, allSkeleton[skeI], Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI],
						triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, 1, ifps, real2SfM);

					//finalize the frame with guided exhausive search
					if (v3DPoints.size() == 0)
						allSkeleton[skeI].active = 0;
					else
					{
						//see if  more points can be found using reconstructed 3D
						vector<int> CamToTest, FrameMatchToTest;
						for (int cid = 0; cid < nCams; cid++)
							CamToTest.push_back(cid), FrameMatchToTest.push_back(refFid - (int)CamTimeInfo[cid].y);

						oPredictedViewMatch = PredictedViewMatch, oPredictedFrameMatch = PredictedFrameMatch, oPredictedDetectionIdMatch = PredictedDetectionIdMatch;
						PredictedViewMatch.clear(), PredictedFrameMatch.clear(), PredictedDetectionIdMatch.clear();
						TriangulationGuidedPeopleSearch(VideoInfo, CamToTest, FrameMatchToTest, Attri, v3DPoints, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, triangulationThresh);

						//clear the prediction that comes from the same camera as previous frame but not tracked using optical flow-->likely to have detection failture which will be subsituted be a wrong close-by person
						for (size_t ii = 0; ii < allSkeleton[skeI].sSke.back().vCidFidDid.size(); ii++)
						{
							bool found = false;
							for (size_t jj = 0; jj < oPredictedViewMatch.size() && !found; jj++)
								if (allSkeleton[skeI].sSke.back().vCidFidDid[ii].x == oPredictedViewMatch[jj])
									found = true;

							if (!found) //clear the prediction that comes from the same camera as previous frame but not tracked using optical flo
							{
								found = false;
								size_t id = -1;
								for (size_t jj = 0; jj < PredictedViewMatch.size() && !found; jj++)
									if (allSkeleton[skeI].sSke.back().vCidFidDid[ii].x == PredictedViewMatch[jj])
										found = true, id = jj;

								if (found)
								{
									PredictedViewMatch.erase(PredictedViewMatch.begin() + id);
									PredictedFrameMatch.erase(PredictedFrameMatch.begin() + id);
									PredictedDetectionIdMatch.erase(PredictedDetectionIdMatch.begin() + id);
								}
							}
						}

						//do a final pass through all possible cameras
						if (oPredictedViewMatch != PredictedViewMatch)
						{
							bestAsso.clear(), AllvReProjErrPerInstance[skeI].clear(), v3DPoints.clear();
							//TriangulateSkeletonRANSAC(VideoInfo, Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI], triangulationThresh, RanSac_iterMax);
							TriangulateSkeletonRANSAC(VideoInfo, allSkeleton[skeI], Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI],
								triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, 1, ifps, real2SfM);
						}

						if (bestAsso.size() > 0)
						{
							for (int jid = 0; jid < nBodyKeyPoints; jid++)
							{
								AllSkePerInstance[skeI].pt3d[jid] = v3DPoints[jid];
								for (auto ii : bestAsso)
								{
									int cid = PredictedViewMatch[ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = PredictedDetectionIdMatch[ii];
									AllSkePerInstance[skeI].vPt2D[jid].emplace_back(Attri[cid][did].pt[jid].x, Attri[cid][did].pt[jid].y);
								}
							}
							for (auto ii : bestAsso)
							{
								int cid = PredictedViewMatch[ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = PredictedDetectionIdMatch[ii];
								AllSkePerInstance[skeI].vCidFidDid.emplace_back(cid, lfid, did);
							}
							AllSkePerInstance[skeI].refFid = refFid;
							AllSkePerInstance[skeI].valid = true;
						}
						else
							allSkeleton[skeI].active = 0;
					}

					if (allSkeleton[skeI].active == 0)
					{
#pragma omp critical
						printLOG("\nFinishing person %d [%d-->%d]\n", skeI, allSkeleton[skeI].startActiveF, refFid);
						allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
						allSkeleton[skeI].nframeSinceLastBA = refFid - allSkeleton[skeI].startActiveF;

						WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);

						if (debug == 2)
							Visualize3DTracklet(Path, VideoInfo, nCams, startF, skeI, allSkeleton, AllCamImages);
					}
				}
			}
			//store data afterward
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				if (AllSkePerInstance[skeI].valid)
				{
					allSkeleton[skeI].sSke.push_back(AllSkePerInstance[skeI]);

					vector<Point2i> UsedCidDid;
					for (size_t ii = 0; ii < AllSkePerInstance[skeI].vCidFidDid.size(); ii++)
						UsedCidDid.emplace_back(AllSkePerInstance[skeI].vCidFidDid[ii].x, AllSkePerInstance[skeI].vCidFidDid[ii].z);
					vUsedCidDid.push_back(UsedCidDid);
				}
			}
			delete[]AllSkePerInstance, delete[]AllvReProjErrPerInstance;
		}

		//Stage II: spatial: work on things that are not associated yet.
		{
			printLOG("\n********************\nSpatial matching and recon\n********************\n");
			//step 1: gen matching table from desc and geo constraints
			int totalPeoplePerInstance = 0;
			for (int cid = 0; cid < nCams; cid++)
				totalPeoplePerInstance += (int)Attri[cid].size();
			vector<int>*ViewMatch = new vector<int>[totalPeoplePerInstance];
			vector<int>*DetectionIdMatch = new vector<int>[totalPeoplePerInstance];
			int nHypoPeople = GeneratePeopleMatchingTable(VideoInfo, Attri, nCams, refFid, ViewMatch, DetectionIdMatch, DescThresh, triangulationThresh, vUsedCidDid);

			//step 2: run RANSAC on each matching hypothesis
			int newPersonCount = 0, n3DTracklets = allSkeleton.size();
			vector<double> *HypovReProjErr = new vector<double>[nHypoPeople];
			HumanSkeleton3D *HypoSkePerInstance = new HumanSkeleton3D[nHypoPeople];
#pragma omp parallel for schedule(dynamic, 1)
			for (int hypoId = 0; hypoId < nHypoPeople; hypoId++)
			{
#pragma omp critical
				printLOG("%d..", hypoId);
				//check if hypothesis was used by the prediction
				int usedHypo = -1;
				for (size_t ii = 0; ii < vUsedCidDid.size() && usedHypo == -1; ii++)
				{
					int nfound = 0;
					for (size_t jj = 0; jj < vUsedCidDid[ii].size(); jj++)
					{
						int cid = vUsedCidDid[ii][jj].x, did = vUsedCidDid[ii][jj].y, found = -1;
						for (size_t kk = 0; kk < ViewMatch[hypoId].size() && found == -1; kk++)
							if (cid == ViewMatch[hypoId][kk] && did == DetectionIdMatch[hypoId][kk])
								found = 1;
						if (found == 1)
							nfound++;
					}
					if (nfound >= 3)  //Since the prediction is exhaustive in the second step, 3 nfounds means the hypo is used. The only other posiblity is the hypo has many correct sub-hypo....--> have to improve the hypo quality using multiview consistency graph
						usedHypo = 1;
				}
				if (usedHypo == 1)
				{
#pragma omp critical
					printLOG("\nFound used hypo %d @ frame %d\n", hypoId, refFid);
					continue;
				}

				vector<Point3d> v3DPoints;
				vector<int> FrameMatch, bestAsso;
				for (size_t ii = 0; ii < ViewMatch[hypoId].size(); ii++)
					FrameMatch.push_back(refFid - (int)CamTimeInfo[ViewMatch[hypoId][ii]].y);

				HumanSkeleton3D SkeI;
				sHumanSkeleton3D sSkeI;
				sSkeI.nBAtimes = 0, SkeI.nPts = nBodyKeyPoints;
				sSkeI.sSke.push_back(SkeI);

				//TriangulateSkeletonRANSAC(VideoInfo, Attri, ViewMatch[hypoId], FrameMatch, DetectionIdMatch[hypoId], bestAsso, v3DPoints, HypovReProjErr[hypoId], triangulationThresh, RanSac_iterMax);
				TriangulateSkeletonRANSAC(VideoInfo, sSkeI, Attri, ViewMatch[hypoId], FrameMatch, DetectionIdMatch[hypoId], bestAsso, v3DPoints, HypovReProjErr[hypoId],
					triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, 1, ifps, real2SfM);

				//not doing any geo-guided exhaustive search here since the Ransac hypo already considers all geo matches.
				if (bestAsso.size() > 0)
				{
					//check if hypothesis was used by the prediction
					usedHypo = -1;
					for (size_t ii = 0; ii < vUsedCidDid.size() && usedHypo == -1; ii++)
					{
						int nfound = 0;
						for (size_t jj = 0; jj < vUsedCidDid[ii].size(); jj++)
						{
							int cid = vUsedCidDid[ii][jj].x, did = vUsedCidDid[ii][jj].y, found = -1;
							for (size_t kk = 0; kk < bestAsso.size() && found == -1; kk++)
								if (cid == ViewMatch[hypoId][bestAsso[kk]] && did == DetectionIdMatch[hypoId][bestAsso[kk]])
									found = 1;
							if (found == 1)
								nfound++;
						}
						if (nfound >= 3)  //Since the prediction is exhaustive in the second step, 3 nfounds means the hypo is used. The only other posiblity is the hypo has many correct sub-hypo....--> have to improve the hypo quality using multiview consistency graph
							usedHypo = 1;
					}
					if (usedHypo == 1)
					{
#pragma omp critical
						printLOG("\n\n\n***********\nDetect used hypo %d @ frame %d. After Ransac\n**********\n\n\n", hypoId, refFid);
						continue;
					}

					if (usedHypo == -1)
					{
						HypoSkePerInstance[hypoId].nPts = nBodyKeyPoints;
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							HypoSkePerInstance[hypoId].pt3d[jid] = v3DPoints[jid];
							for (auto ii : bestAsso)
							{
								int cid = ViewMatch[hypoId][ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = DetectionIdMatch[hypoId][ii];
								HypoSkePerInstance[hypoId].vPt2D[jid].emplace_back(Attri[cid][did].pt[jid].x, Attri[cid][did].pt[jid].y);
							}
						}
						for (auto ii : bestAsso)
						{
							int cid = ViewMatch[hypoId][ii], lfid = refFid - (int)VideoInfo[cid].TimeOffset, did = DetectionIdMatch[hypoId][ii];
							HypoSkePerInstance[hypoId].vCidFidDid.emplace_back(cid, lfid, did);
						}
						HypoSkePerInstance[hypoId].refFid = refFid;
						HypoSkePerInstance[hypoId].valid = true;
					}
				}
			}
			delete[]ViewMatch, delete[]DetectionIdMatch;

			/*	for (int hypoId = 0; hypoId < nHypoPeople; hypoId++)
			HypoSkePerInstance[hypoId].valid = 0;
			sprintf(Fname, "%s/cache2.txt", Path); FILE *fp = fopen(Fname, "r");
			int hypoId;
			while (fscanf(fp, "%d", &hypoId) != EOF)
			{
			HypoSkePerInstance[hypoId].valid = 1;
			HypoSkePerInstance[hypoId].nPts = nBodyKeyPoints;
			for (int jid = 0; jid < nBodyKeyPoints; jid++)
			{
			fscanf(fp, "%lf %lf %lf ", &HypoSkePerInstance[hypoId].pt3d[jid].x, &HypoSkePerInstance[hypoId].pt3d[jid].y, &HypoSkePerInstance[hypoId].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
			int nvis;  fscanf(fp, "%d ", &nvis);
			Point3i cidfiddid; Point2d pt;
			for (size_t ii = 0; ii < nvis; ii++)
			{
			fscanf(fp, "%d %d %d %lf %lf ", &cidfiddid.x, &cidfiddid.y, &cidfiddid.z, &pt.x, &pt.y);
			if (jid == 0)
			HypoSkePerInstance[hypoId].vCidFidDid.push_back(cidfiddid);
			HypoSkePerInstance[hypoId].vPt2D[jid].push_back(pt);
			}
			}
			}
			fclose(fp);
			}*/

			//store data afterward
			for (int hypoId = 0; hypoId < nHypoPeople; hypoId++)
			{
				if (HypoSkePerInstance[hypoId].valid)
				{
					printLOG("Adding %d\n", n3DTracklets + newPersonCount);
					newPersonCount++;

					/*if (allSkeleton.size() == 52)
					{
					printf("\n\n%d: ", hypoId);
					for (int ii = 0; ii < HypoSkePerInstance[hypoId].vCidFidDid.size(); ii++)
					printf("%d %d\n", HypoSkePerInstance[hypoId].vCidFidDid[ii].x, HypoSkePerInstance[hypoId].vCidFidDid[ii].z);
					exit(0);
					}*/
					sHumanSkeleton3D sSkeI;
					sSkeI.active = 1, sSkeI.startActiveF = refFid, sSkeI.lastBAFrame = refFid;
					sSkeI.sSke.push_back(HypoSkePerInstance[hypoId]);
					allSkeleton.push_back(sSkeI);
				}
			}

			/*{
			sprintf(Fname, "%s/cache2.txt", Path); FILE *fp = fopen(Fname, "w");
			for (int hypoId = 0; hypoId < nHypoPeople; hypoId++)
			{
			if (HypoSkePerInstance[hypoId].valid)
			{
			fprintf(fp, "%d\n", hypoId);
			for (int jid = 0; jid < nBodyKeyPoints; jid++)
			{
			fprintf(fp, "%.6f %.6f %.6f\n", HypoSkePerInstance[hypoId].pt3d[jid].x, HypoSkePerInstance[hypoId].pt3d[jid].y, HypoSkePerInstance[hypoId].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
			fprintf(fp, "%zd ", HypoSkePerInstance[hypoId].vCidFidDid.size());
			for (size_t ii = 0; ii < HypoSkePerInstance[hypoId].vCidFidDid.size(); ii++)
			fprintf(fp, "%d %d %d %.2f %.2f ", HypoSkePerInstance[hypoId].vCidFidDid[ii].x, HypoSkePerInstance[hypoId].vCidFidDid[ii].y, HypoSkePerInstance[hypoId].vCidFidDid[ii].z, HypoSkePerInstance[hypoId].vPt2D[jid][ii].x, HypoSkePerInstance[hypoId].vPt2D[jid][ii].y);
			fprintf(fp, "\n");
			}
			}
			}
			fclose(fp);
			exit(0);
			}*/

			delete[]HypoSkePerInstance, delete[]HypovReProjErr;
		}

		//STAGE III: do BA if needed
		{
			printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				if (allSkeleton[skeI].active == 1)
				{
					if (refFid - allSkeleton[skeI].lastBAFrame > BAfreq || refFid - allSkeleton[skeI].startActiveF > BAratio*allSkeleton[skeI].nframeSinceLastBA)
					{
						printLOG("\nPerson %d [%d-->%d]\n", skeI, allSkeleton[skeI].startActiveF, refFid);
						allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
						allSkeleton[skeI].nframeSinceLastBA = refFid - allSkeleton[skeI].startActiveF;

						WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);

						if (debug == 2)
							Visualize3DTracklet(Path, VideoInfo, nCams, startF, skeI, allSkeleton, AllCamImages);
					}
				}
			}
		}

		//Done: swap over the kpts and desc for tracking prepare for next frame
		for (int cid = 0; cid < nCams; cid++)
		{
			AttriOld[cid].clear();
			copy(Attri[cid].begin(), Attri[cid].end(), back_inserter(AttriOld[cid]));
			ImgOldPyr[cid].clear();
			copy(ImgNewPyr[cid].begin(), ImgNewPyr[cid].end(), back_inserter(ImgOldPyr[cid]));
		}

		if ((refFid - startF) % cacheFreq == 0)
		{
			//STAGE IV: do BA 
			printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].lastBAFrame != allSkeleton[skeI].sSke.back().refFid)
				{
					printLOG("\nPerson %d [%d-->%d]\n", skeI, allSkeleton[skeI].startActiveF, stopF);
					allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
					allSkeleton[skeI].nframeSinceLastBA = refFid - allSkeleton[skeI].startActiveF;

					WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
				}
			}
		}

		for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
		{
			sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI); FILE *fp = fopen(Fname, "w");
			for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
			{
				fprintf(fp, "%d\n", allSkeleton[skeI].sSke[inst].refFid);
				for (int jid = 0; jid < nBodyKeyPoints; jid++)
				{
					fprintf(fp, "%.6f %.6f %.6f\n", allSkeleton[skeI].sSke[inst].pt3d[jid].x, allSkeleton[skeI].sSke[inst].pt3d[jid].y, allSkeleton[skeI].sSke[inst].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
					fprintf(fp, "%zd ", allSkeleton[skeI].sSke[inst].vCidFidDid.size());
					for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
					{
						int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, did = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].z;

						Point2f pt(allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].x, allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].y);
						LensDistortionPoint(&pt, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
						fprintf(fp, "%d %d %d %.2f %.2f ", cid, fid, did, pt.x, pt.y);
					}
					fprintf(fp, "\n");
				}
			}
			fclose(fp);
		}
	}

	//STAGE IV: do BA 
	{
		printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
		for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
		{
			if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].lastBAFrame != allSkeleton[skeI].sSke.back().refFid)
			{
				printLOG("\nPerson %d [%d-->%d]\n", skeI, allSkeleton[skeI].startActiveF, stopF);
				allSkeleton[skeI].lastBAFrame = stopF, allSkeleton[skeI].nBAtimes++;
				allSkeleton[skeI].nframeSinceLastBA = stopF - allSkeleton[skeI].startActiveF;

				WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
			}
		}
	}

	for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
	{
		sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI); FILE *fp = fopen(Fname, "w");
		for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
		{
			fprintf(fp, "%d\n", allSkeleton[skeI].sSke[inst].refFid);
			for (int jid = 0; jid < nBodyKeyPoints; jid++)
			{
				fprintf(fp, "%.6f %.6f %.6f\n", allSkeleton[skeI].sSke[inst].pt3d[jid].x, allSkeleton[skeI].sSke[inst].pt3d[jid].y, allSkeleton[skeI].sSke[inst].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
				fprintf(fp, "%zd ", allSkeleton[skeI].sSke[inst].vCidFidDid.size());
				for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
				{
					int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, did = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].z;

					Point2f pt(allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].x, allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].y);
					LensDistortionPoint(&pt, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
					fprintf(fp, "%d %d %d %.2f %.2f ", cid, fid, did, pt.x, pt.y);
				}
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}

	if (0)//debug > 0)
	{
		printLOG("Done. Do visualization\n");
		for (int cidM = 0; cidM < nCams; cidM++)
		{
			sprintf(Fname, "%s/Vis/3DTracklet/%.2d.avi", Path, cidM);
			cv::Size size;
			cv::VideoWriter writer;
			bool firstTime = true;
			for (int refFidM = startF; refFidM <= stopF; refFidM++)
			{
				int lfid = refFidM - (int)CamTimeInfo[cidM].y;
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, SelectedCamNames[cidM], lfid);
				Mat img = cv::imread(Fname);
				if (img.empty())
					continue;

				if (firstTime)
				{
					firstTime = false;
					size.width = img.rows, size.height = img.cols;
					writer.open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 25, size);
				}

				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
					{
						for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
						{
							int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, rfid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, refFid = rfid + (int)CamTimeInfo[cid].y;
							if (cid != cidM || refFid != refFidM)
								continue;

							CameraData *camI = VideoInfo[cid].VideoInfo;
							BodyImgAtrribute lm;
							lm.personLocalId = skeI;
							lm.nPts = nBodyKeyPoints;
							for (int jid = 0; jid < nBodyKeyPoints; jid++)
							{
								if (!IsValid3D(allSkeleton[skeI].sSke[inst].pt3d[jid]))
									continue;
								Point2d pt_;
								ProjectandDistort(allSkeleton[skeI].sSke[inst].pt3d[jid], &pt_, camI[rfid].P, camI[rfid].K, camI[rfid].distortion);
								lm.pt[jid].x = pt_.x, lm.pt[jid].y = pt_.y;
							}
							Draw2DCoCo(img, lm, 1, 1.0, skeI);
						}
					}
				}

				int rfid = refFidM - (int)CamTimeInfo[cidM].y;
				cv::Point2i text_origin = { img.cols / 20, img.rows / 15 };
				sprintf(Fname, "%d: %d (%d)", cidM, rfid, refFidM); putText(img, Fname, text_origin, cv::FONT_HERSHEY_SIMPLEX, img.cols / 640, cv::Scalar(0, 255, 0), 2.0);
				writer << img;
			}
			writer.release();
		}
	}

	delete[]AllCamImages;
	return 0;
}
int SimultaneousTrackingAndAssocation_GT_Prop(char *Path, std::vector<int> &SelectedCamNames, int startF, int stopF, double triangulationThresh, int RanSac_iterMax, int AtLeastThree4Ransac, float max_ratio, float DescThresh, float BAratio, int BAfreq, int reliableTrackRange, double real2SfM, double *WeightsHumanSkeleton3D, double *iSigmaSkeleton, int nBodyKeyPoints = 18, int debug = 0)
{
	int cacheFreq = max(500, (stopF - startF) / 4);
	double ifps = 1.0 / 60;
	int winSize = 31, cvPyrLevel = 5;
	double flowThresh = 8.0;

	char Fname[512];
	sprintf(Fname, "%s/Skeleton_%d_%d/", Path, startF, stopF); makeDir(Fname);
	sprintf(Fname, "%s/Vis/3DTracklet/", Path); makeDir(Fname);

	int nCams = (int)SelectedCamNames.size();

	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	int selected, temp;
	double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
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

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
	{
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;
		VideoInfo[cid].fps = 1.0 / CamTimeInfo[cid].x;
		VideoInfo[cid].TimeOffset = CamTimeInfo[cid].y;
	}

	vector<BodyImgAtrribute> *Attri = new vector<BodyImgAtrribute>[nCams], *AttriOld = new vector<BodyImgAtrribute>[nCams];

	Mat Img1, Img2;
	vector<Mat> *ImgOldPyr = new vector<Mat>[nCams], *ImgNewPyr = new vector<Mat>[nCams];

	//omp_set_num_threads(omp_get_max_threads());

	bool silent = false;
	vector<Mat> *AllCamImages = new vector<Mat>[nCams];

	int nMaxPeople = 0;
	vector<bool> hasPerson(10000);
	vector<vector<Point2i> > *MergedTrackletVec = new vector<vector<Point2i> >[nCams];
	for (auto cid : SelectedCamNames)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_18_%d_%d.txt", Path, cid, startF, 3000);
		if (!IsFileExist(Fname))
			return 1;
		std::string line, item;
		std::ifstream file(Fname);
		int pid, fid, did;
		while (std::getline(file, line))
		{
			StringTrim(&line);//remove white space
			if (line.empty())
				break;
			std::stringstream line_stream(line);
			std::getline(line_stream, item, ' ');
			pid = atoi(item.c_str());
			if (pid >= 0)
				hasPerson[pid] = true;

			vector<Point2i> jointTrack;
			if (MergedTrackletVec[cid].size() < pid)
			{
				for (int ii = MergedTrackletVec[cid].size(); ii < pid; ii++)
					MergedTrackletVec[cid].push_back(jointTrack);
			}

			std::getline(file, line);
			std::stringstream line_stream_(line);
			while (!line_stream_.eof())
			{
				std::getline(line_stream_, item, ' ');
				StringTrim(&item);
				fid = atoi(item.c_str());
				std::getline(line_stream_, item, ' ');
				StringTrim(&item);
				did = atoi(item.c_str());
				jointTrack.push_back(Point2i(fid, did));
			}
			jointTrack.pop_back();
			MergedTrackletVec[cid].push_back(jointTrack);
		}
		file.close();

		nMaxPeople = max(nMaxPeople, (int)MergedTrackletVec[cid].size());
	}
	nMaxPeople--;//last one is trash category	

	vector<sHumanSkeleton3D > allSkeleton; //skeleton xyz and state of activenesss
	for (int ii = 0; ii < nMaxPeople; ii++)
	{
		sHumanSkeleton3D sSkeI;
		sSkeI.active = 0;
		if (hasPerson[ii])
		{
			sSkeI.sSke.resize(stopF + 1), sSkeI.active = 1;
			for (int jj = 0; jj < stopF; jj++)
				sSkeI.sSke[jj].refFid = -1, sSkeI.sSke[jj].valid = 0, sSkeI.sSke[jj].nPts = nBodyKeyPoints;
		}
		allSkeleton.emplace_back(sSkeI);
	}

	{
		int refFid;
		Point3i cidfiddid; Point2d pt; Point2f cPose[25];

		const int nLimbConnections = 17;
		Point2i LimbConnectionID[nLimbConnections] = { Point2i(0, 1), Point2i(1, 2), Point2i(2, 3), Point2i(3, 4), Point2i(1, 5), Point2i(5, 6), Point2i(6, 7),
			Point2i(1, 8), Point2i(8, 9), Point2i(9, 10), Point2i(1, 11), Point2i(11, 12), Point2i(12, 13), Point2i(0, 14), Point2i(0, 15), Point2i(14, 16), Point2i(15, 17) };

		for (int skeI = 0; skeI < nMaxPeople; skeI++)
		{
			int cnt = 0;
			sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI);
			FILE *fp = fopen(Fname, "r");
			while (fscanf(fp, "%d ", &refFid) != EOF)
			{
				allSkeleton[skeI].sSke[refFid].refFid = refFid;
				allSkeleton[skeI].sSke[refFid].nPts = nBodyKeyPoints;
				allSkeleton[skeI].sSke[refFid].valid = true;
				for (int jid = 0; jid < nBodyKeyPoints; jid++)
				{
					fscanf(fp, "%lf %lf %lf ", &allSkeleton[skeI].sSke[refFid].pt3d[jid].x, &allSkeleton[skeI].sSke[refFid].pt3d[jid].y, &allSkeleton[skeI].sSke[refFid].pt3d[jid].z);
					fscanf(fp, "%d ", &allSkeleton[skeI].sSke[refFid].nvis);
					for (size_t ii = 0; ii < allSkeleton[skeI].sSke[refFid].nvis; ii++)
					{
						fscanf(fp, "%d %d %d %lf %lf ", &cidfiddid.x, &cidfiddid.y, &cidfiddid.z, &pt.x, &pt.y);
						if (jid == 0)
							allSkeleton[skeI].sSke[refFid].vCidFidDid.push_back(cidfiddid);
						allSkeleton[skeI].sSke[refFid].vPt2D[jid].push_back(pt);
					}
				}
				cnt++;
				allSkeleton[skeI].nBAtimes = 1;
				allSkeleton[skeI].lastBAFrame = refFid;
				allSkeleton[skeI].nframeSinceLastBA = cnt;
			}
			fclose(fp);

			if (cnt > 0)
			{
				vector<double> *vlimblength = new vector<double>[nLimbConnections];
				for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
				{
					if (allSkeleton[skeI].sSke[inst].valid == 0)
						continue;
					for (int cid = 0; cid < nLimbConnections; cid++)
					{
						int j0 = LimbConnectionID[cid].x, j1 = LimbConnectionID[cid].y;
						if (IsValid3D(allSkeleton[skeI].sSke[inst].pt3d[j0]) && IsValid3D(allSkeleton[skeI].sSke[inst].pt3d[j1]))
							vlimblength[cid].push_back(norm(allSkeleton[skeI].sSke[inst].pt3d[j0] - allSkeleton[skeI].sSke[inst].pt3d[j1]));
					}
				}
				for (int cid = 0; cid < nLimbConnections; cid++)
					allSkeleton[skeI].meanBoneLength[cid] = MedianArray(vlimblength[cid]);
				delete[]vlimblength;
			}
		}
	}

	bool forwardTrack = false;
	if (forwardTrack)
	{
		for (int refFid = startF; refFid <= 2970 - 1; refFid++)
		{
			vector<vector<Point2i> > Pid_vCidDid;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int pid = 0; pid < MergedTrackletVec[cid].size(); pid++)
				{
					for (int inst = 0; inst < MergedTrackletVec[cid][pid].size(); inst++)
					{
						int rfid = MergedTrackletVec[cid][pid][inst].x, did = MergedTrackletVec[cid][pid][inst].y;
						if (rfid + (int)CamTimeInfo[cid].y == refFid)
						{
							vector<Point2i> CidrFidDid;
							for (int ii = Pid_vCidDid.size(); ii <= pid; ii++)
								Pid_vCidDid.push_back(CidrFidDid);

							Pid_vCidDid[pid].push_back(Point2i(cid, did));
							break;
						}
					}
				}
			}

			int nHypoPeople = 0;
			vector<int> vrPid;
			vector<vector<int> > ViewMatch, DetectionIdMatch;
			for (int ii = 0; ii < Pid_vCidDid.size(); ii++)
			{
				vector<int> vMatch, dMatch;
				for (int jj = 0; jj < Pid_vCidDid[ii].size(); jj++)
				{
					vMatch.emplace_back(Pid_vCidDid[ii][jj].x);
					dMatch.emplace_back(Pid_vCidDid[ii][jj].y);
				}
				if (vMatch.size() > 1)
					ViewMatch.emplace_back(vMatch), DetectionIdMatch.emplace_back(dMatch), vrPid.emplace_back(ii), nHypoPeople++;
			}
			bool found = false;
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				if (allSkeleton[skeI].active == 1)
					found = true;
			if (nHypoPeople == 0 && !found)
				continue;
			printLOG("\n********************\n%d: %zd total #people\n", refFid, allSkeleton.size());

			//read BodyKeyPoints and desc
			for (int cid = 0; cid < nCams; cid++)
			{
				Attri[cid].clear();
				int rfid = refFid - (int)CamTimeInfo[cid].y;
				CameraData *camI = VideoInfo[cid].VideoInfo;
				if (camI[rfid].valid != 1)
					continue;

				ReadBodyKeyPointsAndDesc(Path, cid, rfid, Attri[cid]);
				for (int pid = 0; pid < Attri[cid].size(); pid++)
				{
					for (int jid = 0; jid < nBodyKeyPoints; jid++)
					{
						Point2d pt(Attri[cid][pid].pt[jid].x, Attri[cid][pid].pt[jid].y);
						if (Attri[cid][pid].pt[jid].z == 0.0)
							Attri[cid][pid].pt[jid] = Point3d(0, 0, 0);
						else
						{
							LensCorrectionPoint(&pt, camI[rfid].K, camI[rfid].distortion);
							Attri[cid][pid].pt[jid].x = pt.x, Attri[cid][pid].pt[jid].y = pt.y;
						}
					}
				}

				AllCamImages[cid].clear();
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, SelectedCamNames[cid], rfid);
				if (!IsFileExist(Fname))
				{
					printLOG("Cannot find %s. Exit\n", Fname);
					continue;
				}
				Img2 = imread(Fname, 0);
				buildOpticalFlowPyramid(Img2, ImgNewPyr[cid], Size(winSize, winSize), cvPyrLevel, false);

				sprintf(Fname, "%s/%d/Corrected/%.4d.jpg", Path, SelectedCamNames[cid], rfid);  Img2 = imread(Fname, 0);
				AllCamImages[cid].push_back(Img2);
			}

			//Stage I: temporal: using perCamera tracklets to predict next assocation and Ransac them.
			{
				printLOG("\n********************\nTemporal propagation and recon\n********************\n");
				vector<double> *AllvReProjErrPerInstance = new vector<double>[allSkeleton.size()];

#pragma omp parallel for schedule(dynamic, 1)
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					if (allSkeleton[skeI].active == 0) //active
						continue;
					if (allSkeleton[skeI].sSke[refFid].valid == 1 || refFid - 1 < startF)
						continue;
					vector<int> bestAsso, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch;
					for (int ii = 0; ii < vrPid.size(); ii++)
					{
						if (vrPid[ii] == skeI)
						{
							PredictedViewMatch = ViewMatch[ii], PredictedDetectionIdMatch = DetectionIdMatch[ii];
							for (size_t jj = 0; jj < PredictedViewMatch.size(); jj++)
								PredictedFrameMatch.push_back(refFid - (int)CamTimeInfo[PredictedViewMatch[jj]].y);
						}
					}

					//if (PredictedViewMatch.size() < 2) //only do tracking if gt is not available
					if (refFid - allSkeleton[skeI].sSke[refFid - 1].refFid == 1) //Forward prediction via intra-view tracking -> Ransac
					{
						for (size_t ii = 0; ii < allSkeleton[skeI].sSke[refFid - 1].vCidFidDid.size(); ii++)
						{
							int cid = allSkeleton[skeI].sSke[refFid - 1].vCidFidDid[ii].x, rfid = refFid - (int)CamTimeInfo[cid].y, did = allSkeleton[skeI].sSke[refFid - 1].vCidFidDid[ii].z; //current 2d projection
							if (AllCamImages[cid].empty() == 1)
								continue;
							vector<Point2i> matches = MatchTwoFrame_BodyKeyPoints_LK(VideoInfo[cid].VideoInfo[rfid], ImgOldPyr[cid], ImgNewPyr[cid], AttriOld[cid], Attri[cid], flowThresh, 0.2, true);

							for (size_t jj = 0; jj < matches.size(); jj++)
							{
								if (matches[jj].x == did)
								{
									bool notfound = true;
									for (int kk = 0; kk < PredictedViewMatch.size() && notfound; kk++)
										if (PredictedViewMatch[kk] == cid && PredictedFrameMatch[kk] == rfid && PredictedDetectionIdMatch[kk] == matches[jj].y)
											notfound = false;
									if (notfound)
									{
										PredictedViewMatch.push_back(cid);
										PredictedFrameMatch.push_back(rfid);
										PredictedDetectionIdMatch.push_back(matches[jj].y);
									}
								}
							}
						}
					}

					if (PredictedViewMatch.size() > 1)
					{
						int cnt = 0;
						for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
							if (allSkeleton[skeI].sSke[inst].valid)
								cnt++;
						printLOG("Continuing %d...%d frames tracked\n", skeI, cnt);
					}
					else
						continue;

					if (debug)
					{
						for (int ii = 0; ii < PredictedViewMatch.size(); ii++)
						{
							int cid = PredictedViewMatch[ii], pid = PredictedDetectionIdMatch[ii];

							Mat img = AllCamImages[cid][0].clone();
							Draw2DCoCo(img, Attri[cid][pid], 1, 1.0, pid);

							CvPoint text_origin = { img.cols / 20,img.rows / 15 }; sprintf(Fname, "%d", cid);
							putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 2);

							namedWindow(Fname, CV_WINDOW_NORMAL); imshow(Fname, img);
						}
						waitKey(0);
					}

					vector<Point3d> v3DPoints(nBodyKeyPoints);
					double WeightsHumanSkeleton3D_bk[4];
					for (int ii = 0; ii < 4; ii++)
						WeightsHumanSkeleton3D_bk[ii] = WeightsHumanSkeleton3D[ii];
					int increF = refFid - allSkeleton[skeI].sSke[refFid - 1].refFid;
					if (increF > 1)
						WeightsHumanSkeleton3D_bk[3] = 0.0;

					TriangulateSkeletonRANSAC(VideoInfo, allSkeleton[skeI], Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI],
						triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D_bk, iSigmaSkeleton, increF, ifps, real2SfM);

					//finalize the frame with guided exhausive search
					if (v3DPoints.size() == 0 || bestAsso.size() == 0)
						continue;

					//see if  more points can be found using reconstructed 3D
					vector<int> CamToTest, FrameMatchToTest;
					for (int cid = 0; cid < nCams; cid++)
						CamToTest.push_back(cid), FrameMatchToTest.push_back(refFid - (int)CamTimeInfo[cid].y);

					vector<int> oPredictedViewMatch, oPredictedFrameMatch, oPredictedDetectionIdMatch;
					oPredictedViewMatch = PredictedViewMatch, oPredictedFrameMatch = PredictedFrameMatch, oPredictedDetectionIdMatch = PredictedDetectionIdMatch;
					PredictedViewMatch.clear(), PredictedFrameMatch.clear(), PredictedDetectionIdMatch.clear();
					TriangulationGuidedPeopleSearch(VideoInfo, CamToTest, FrameMatchToTest, Attri, v3DPoints, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, triangulationThresh);

					//do a final pass through all possible cameras
					if (oPredictedViewMatch != PredictedViewMatch)
					{
						if (debug)
						{
							for (int ii = 0; ii < PredictedViewMatch.size(); ii++)
							{
								int cid = PredictedViewMatch[ii], pid = PredictedDetectionIdMatch[ii];

								Mat img = AllCamImages[cid][0].clone();
								Draw2DCoCo(img, Attri[cid][pid], 1, 1.0, pid);

								CvPoint text_origin = { img.cols / 20,img.rows / 15 }; sprintf(Fname, "%d", cid);
								putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 2);

								namedWindow(Fname, CV_WINDOW_NORMAL); imshow(Fname, img);
							}
							waitKey(0);
						}
						bestAsso.clear(), AllvReProjErrPerInstance[skeI].clear(), v3DPoints.clear();
						TriangulateSkeletonRANSAC(VideoInfo, allSkeleton[skeI], Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI],
							triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D_bk, iSigmaSkeleton, increF, ifps, real2SfM);
					}

					if (bestAsso.size() > 0)
					{
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							allSkeleton[skeI].sSke[refFid].pt3d[jid] = v3DPoints[jid];
							for (auto ii : bestAsso)
							{
								int cid = PredictedViewMatch[ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = PredictedDetectionIdMatch[ii];
								allSkeleton[skeI].sSke[refFid].vPt2D[jid].emplace_back(Attri[cid][did].pt[jid].x, Attri[cid][did].pt[jid].y);
							}
						}
						for (auto ii : bestAsso)
						{
							int cid = PredictedViewMatch[ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = PredictedDetectionIdMatch[ii];
							allSkeleton[skeI].sSke[refFid].vCidFidDid.emplace_back(cid, lfid, did);
						}
						allSkeleton[skeI].sSke[refFid].refFid = refFid, allSkeleton[skeI].sSke[refFid].nPts = nBodyKeyPoints;
						allSkeleton[skeI].sSke[refFid].valid = true;
					}
					else
						int a = 0;
				}
				delete[]AllvReProjErrPerInstance;
			}

			//Stage II: spatial: work on things that are not associated yet.
			{
				printLOG("\n********************\nSpatial matching and recon\n********************\n");

				//step 2: run RANSAC on each matching hypothesis
				int newPersonCount = 0, n3DTracklets = allSkeleton.size();
				vector<double> *HypovReProjErr = new vector<double>[nHypoPeople];

#pragma omp parallel for schedule(dynamic, 1)
				for (int hypoId = 0; hypoId < nHypoPeople; hypoId++)
				{
					if (allSkeleton[vrPid[hypoId]].active == 1) //has been used in stage 1
						continue;
					if (allSkeleton[vrPid[hypoId]].sSke[refFid].valid == 1)
						continue;
					printLOG("Start %d...\n", vrPid[hypoId]);

					if (debug)
					{
						for (int ii = 0; ii < ViewMatch[hypoId].size(); ii++)
						{
							int cid = ViewMatch[hypoId][ii], pid = DetectionIdMatch[hypoId][ii];
							sprintf(Fname, "%d", cid); namedWindow(Fname, CV_WINDOW_NORMAL);
							Mat img = AllCamImages[cid][0].clone();
							Draw2DCoCo(img, Attri[cid][pid], 1, 1.0, pid);
							imshow(Fname, img);
						}
						waitKey(0);
					}
					vector<Point3d> v3DPoints;
					vector<int> FrameMatch, bestAsso;
					for (size_t ii = 0; ii < ViewMatch[hypoId].size(); ii++)
						FrameMatch.push_back(refFid - (int)CamTimeInfo[ViewMatch[hypoId][ii]].y);

					HumanSkeleton3D SkeI;
					sHumanSkeleton3D sSkeI;
					sSkeI.nBAtimes = 0, SkeI.nPts = nBodyKeyPoints;
					sSkeI.sSke.push_back(SkeI);

					//TriangulateSkeletonRANSAC(VideoInfo, Attri, ViewMatch[hypoId], FrameMatch, DetectionIdMatch[hypoId], bestAsso, v3DPoints, HypovReProjErr[hypoId], triangulationThresh, RanSac_iterMax);
					TriangulateSkeletonRANSAC(VideoInfo, sSkeI, Attri, ViewMatch[hypoId], FrameMatch, DetectionIdMatch[hypoId], bestAsso, v3DPoints, HypovReProjErr[hypoId],
						triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, 1, ifps, real2SfM);

					if (bestAsso.size() > 0)
					{
						allSkeleton[vrPid[hypoId]].sSke[refFid].nPts = nBodyKeyPoints;
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							allSkeleton[vrPid[hypoId]].sSke[refFid].pt3d[jid] = v3DPoints[jid];
							for (auto ii : bestAsso)
							{
								int cid = ViewMatch[hypoId][ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = DetectionIdMatch[hypoId][ii];
								allSkeleton[vrPid[hypoId]].sSke[refFid].vPt2D[jid].emplace_back(Attri[cid][did].pt[jid].x, Attri[cid][did].pt[jid].y);
							}
						}
						for (auto ii : bestAsso)
						{
							int cid = ViewMatch[hypoId][ii], lfid = refFid - (int)VideoInfo[cid].TimeOffset, did = DetectionIdMatch[hypoId][ii];
							allSkeleton[vrPid[hypoId]].sSke[refFid].vCidFidDid.emplace_back(cid, lfid, did);
						}
						allSkeleton[vrPid[hypoId]].sSke[refFid].refFid = refFid, allSkeleton[vrPid[hypoId]].sSke[refFid].valid = true;
						allSkeleton[vrPid[hypoId]].active = 1, allSkeleton[vrPid[hypoId]].startActiveF = refFid, allSkeleton[vrPid[hypoId]].lastBAFrame = refFid;
					}
				}
				delete[]HypovReProjErr;
			}

			//STAGE III: do BA if needed
			{
				printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					int cnt = 0;
					for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
						if (allSkeleton[skeI].sSke[inst].valid)
							cnt++;
					if (allSkeleton[skeI].active == 1 && (refFid - allSkeleton[skeI].lastBAFrame > BAfreq || cnt > BAratio*allSkeleton[skeI].nframeSinceLastBA))
					{
						printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, refFid, cnt);
						allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
						allSkeleton[skeI].nframeSinceLastBA = cnt;

						WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);

						if (debug == 2)
							Visualize3DTracklet(Path, VideoInfo, nCams, startF, skeI, allSkeleton, AllCamImages);
					}
				}
			}

			//Done: swap over the kpts and desc for tracking prepare for next frame
			for (int cid = 0; cid < nCams; cid++)
			{
				AttriOld[cid].clear();
				copy(Attri[cid].begin(), Attri[cid].end(), back_inserter(AttriOld[cid]));
				ImgOldPyr[cid].clear();
				copy(ImgNewPyr[cid].begin(), ImgNewPyr[cid].end(), back_inserter(ImgOldPyr[cid]));
			}

			if ((refFid - startF) % cacheFreq == 0)
			{
				//STAGE IV: do BA 
				printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					int cnt = 0;
					for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
						if (allSkeleton[skeI].sSke[inst].valid)
							cnt++;
					if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].nframeSinceLastBA != cnt)
					{
						printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, stopF, cnt);
						allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
						allSkeleton[skeI].nframeSinceLastBA = cnt;

						WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
					}
				}

				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI); FILE *fp = fopen(Fname, "w");
					for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
					{
						if (allSkeleton[skeI].sSke[inst].valid == 0)
							continue;
						fprintf(fp, "%d\n", allSkeleton[skeI].sSke[inst].refFid);
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							fprintf(fp, "%.6f %.6f %.6f\n", allSkeleton[skeI].sSke[inst].pt3d[jid].x, allSkeleton[skeI].sSke[inst].pt3d[jid].y, allSkeleton[skeI].sSke[inst].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
							fprintf(fp, "%zd ", allSkeleton[skeI].sSke[inst].vCidFidDid.size());
							for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
							{
								int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, did = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].z;

								Point2f pt(allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].x, allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].y);
								LensDistortionPoint(&pt, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
								fprintf(fp, "%d %d %d %.2f %.2f ", cid, fid, did, pt.x, pt.y);
							}
							fprintf(fp, "\n");
						}
					}
					fclose(fp);
				}
			}
		}

		//STAGE IV: do BA 
		{
			printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				int cnt = 0, maxValidFid = -1;
				for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
					if (allSkeleton[skeI].sSke[inst].valid)
						maxValidFid = max(maxValidFid, inst), cnt++;
				if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].nframeSinceLastBA != cnt)
				{
					printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, maxValidFid, cnt);
					allSkeleton[skeI].lastBAFrame = stopF, allSkeleton[skeI].nBAtimes++;
					allSkeleton[skeI].nframeSinceLastBA = cnt;

					WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
				}
			}
		}
	}

	bool backwardTrack = true;
	if (backwardTrack) //only track from the seed points
	{
		for (int refFid = 2970; refFid > startF; refFid--)
		{
			vector<vector<Point2i> > Pid_vCidDid;
			for (int cid = 0; cid < nCams; cid++)
			{
				for (int pid = 0; pid < MergedTrackletVec[cid].size(); pid++)
				{
					for (int inst = 0; inst < MergedTrackletVec[cid][pid].size(); inst++)
					{
						int rfid = MergedTrackletVec[cid][pid][inst].x, did = MergedTrackletVec[cid][pid][inst].y;
						if (rfid + (int)CamTimeInfo[cid].y == refFid)
						{
							vector<Point2i> CidrFidDid;
							for (int ii = Pid_vCidDid.size(); ii <= pid; ii++)
								Pid_vCidDid.push_back(CidrFidDid);

							Pid_vCidDid[pid].push_back(Point2i(cid, did));
							break;
						}
					}
				}
			}

			//previous frame already tracked
			int nvalid = 0;
			for (int cid = 0; cid < nCams; cid++)
			{
				if (AttriOld[cid].size() > 0)
					nvalid++;
			}

			//not tracked before
			if (nvalid == 0)
			{
				nvalid = 0;
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].sSke[refFid].valid == 1 && allSkeleton[skeI].sSke[refFid - 1].valid == 0)
						nvalid++;
				}
				if (nvalid == 0)
					continue;

				//read BodyKeyPoints and desc
				for (int cid = 0; cid < nCams; cid++)
				{
					Attri[cid].clear();
					int rfid = refFid - (int)CamTimeInfo[cid].y;
					CameraData *camI = VideoInfo[cid].VideoInfo;
					if (camI[rfid].valid != 1)
						continue;

					ReadBodyKeyPointsAndDesc(Path, cid, rfid, Attri[cid]);
					for (int pid = 0; pid < Attri[cid].size(); pid++)
					{
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							Point2d pt(Attri[cid][pid].pt[jid].x, Attri[cid][pid].pt[jid].y);
							if (Attri[cid][pid].pt[jid].z == 0.0)
								Attri[cid][pid].pt[jid] = Point3d(0, 0, 0);
							else
							{
								LensCorrectionPoint(&pt, camI[rfid].K, camI[rfid].distortion);
								Attri[cid][pid].pt[jid].x = pt.x, Attri[cid][pid].pt[jid].y = pt.y;
							}
						}
					}

					AllCamImages[cid].clear();
					sprintf(Fname, "%s/%d/Corrected/%.4d.jpg", Path, SelectedCamNames[cid], rfid);
					if (!IsFileExist(Fname))
					{
						printLOG("Cannot find %s. Exit\n", Fname);
						continue;
					}
					Img2 = imread(Fname, 0);
					buildOpticalFlowPyramid(Img2, ImgNewPyr[cid], Size(winSize, winSize), cvPyrLevel, false);
					AllCamImages[cid].push_back(Img2);
				}

				//swap over the kpts and desc for tracking prepare for next frame
				for (int cid = 0; cid < nCams; cid++)
				{
					AttriOld[cid].clear();
					copy(Attri[cid].begin(), Attri[cid].end(), back_inserter(AttriOld[cid]));
					ImgOldPyr[cid].clear();
					copy(ImgNewPyr[cid].begin(), ImgNewPyr[cid].end(), back_inserter(ImgOldPyr[cid]));
				}
				continue;
			}

			printLOG("\n********************\n%d: %zd total #people\n", refFid, allSkeleton.size());

			//read BodyKeyPoints and desc
			for (int cid = 0; cid < nCams; cid++)
			{
				Attri[cid].clear(), AllCamImages[cid].clear();
				int rfid = refFid - (int)CamTimeInfo[cid].y;
				CameraData *camI = VideoInfo[cid].VideoInfo;
				if (camI[rfid].valid != 1)
					continue;

				ReadBodyKeyPointsAndDesc(Path, cid, rfid, Attri[cid]);
				for (int pid = 0; pid < Attri[cid].size(); pid++)
				{
					for (int jid = 0; jid < nBodyKeyPoints; jid++)
					{
						Point2d pt(Attri[cid][pid].pt[jid].x, Attri[cid][pid].pt[jid].y);
						if (Attri[cid][pid].pt[jid].z == 0.0)
							Attri[cid][pid].pt[jid] = Point3d(0, 0, 0);
						else
						{
							LensCorrectionPoint(&pt, camI[rfid].K, camI[rfid].distortion);
							Attri[cid][pid].pt[jid].x = pt.x, Attri[cid][pid].pt[jid].y = pt.y;
						}
					}
				}

				sprintf(Fname, "%s/%d/%.4d.jpg", Path, SelectedCamNames[cid], rfid);
				if (!IsFileExist(Fname))
				{
					printLOG("Cannot find %s. Exit\n", Fname);
					continue;
				}
				Img2 = imread(Fname, 0);
				buildOpticalFlowPyramid(Img2, ImgNewPyr[cid], Size(winSize, winSize), cvPyrLevel, false);

				sprintf(Fname, "%s/%d/Corrected/%.4d.jpg", Path, SelectedCamNames[cid], rfid);
				Img2 = imread(Fname, 0);
				AllCamImages[cid].push_back(Img2);
			}

			//Stage I: temporal: using perCamera tracklets to predict next assocation and Ransac them.
			{
				printLOG("\n********************\nTemporal propagation and recon\n********************\n");
				vector<double> *AllvReProjErrPerInstance = new vector<double>[allSkeleton.size()];

				bool success = false;
#pragma omp parallel for schedule(dynamic, 1)
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					if (allSkeleton[skeI].active == 0 || allSkeleton[skeI].sSke[refFid].valid == 1 || allSkeleton[skeI].sSke[refFid + 1].valid == 0)
						continue;

					vector<int> bestAsso, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch;
					for (size_t ii = 0; ii < allSkeleton[skeI].sSke[refFid + 1].vCidFidDid.size(); ii++)
					{
						int cid = allSkeleton[skeI].sSke[refFid + 1].vCidFidDid[ii].x, rfid = refFid + 1 - (int)CamTimeInfo[cid].y, did = allSkeleton[skeI].sSke[refFid + 1].vCidFidDid[ii].z; //current 2d projection
						vector<Point2i> matches = MatchTwoFrame_BodyKeyPoints_LK(VideoInfo[cid].VideoInfo[rfid - 1], ImgOldPyr[cid], ImgNewPyr[cid], AttriOld[cid], Attri[cid], flowThresh, 0.2, true);

						for (size_t jj = 0; jj < matches.size(); jj++)
						{
							if (matches[jj].x == did)
							{
								bool notfound = true;
								for (int kk = 0; kk < PredictedViewMatch.size() && notfound; kk++)
									if (PredictedViewMatch[kk] == cid && PredictedFrameMatch[kk] == rfid - 1 && PredictedDetectionIdMatch[kk] == matches[jj].y)
										notfound = false;
								if (notfound)
								{
									PredictedViewMatch.push_back(cid);
									PredictedFrameMatch.push_back(rfid - 1);
									PredictedDetectionIdMatch.push_back(matches[jj].y);
								}
							}
						}
					}

					if (PredictedViewMatch.size() > 1)
					{
						int cnt = 0;
						for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
							if (allSkeleton[skeI].sSke[inst].valid)
								cnt++;
						printLOG("Continuing %d...%d frames tracked\n", skeI, cnt);
					}
					else
						continue;

					if (debug)
					{
						for (int ii = 0; ii < PredictedViewMatch.size(); ii++)
						{
							int cid = PredictedViewMatch[ii], pid = PredictedDetectionIdMatch[ii];

							Mat img = AllCamImages[cid][0].clone();
							Draw2DCoCo(img, Attri[cid][pid], 1, 1.0, pid);

							CvPoint text_origin = { img.cols / 20,img.rows / 15 }; sprintf(Fname, "%d", cid);
							putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 2);

							namedWindow(Fname, CV_WINDOW_NORMAL); imshow(Fname, img);
						}
						waitKey(0);
					}

					vector<Point3d> v3DPoints(nBodyKeyPoints);
					double WeightsHumanSkeleton3D_bk[4];
					for (int ii = 0; ii < 4; ii++)
						WeightsHumanSkeleton3D_bk[ii] = WeightsHumanSkeleton3D[ii];
					int increF = allSkeleton[skeI].sSke[refFid + 1].refFid - refFid;
					if (increF > 1)
						WeightsHumanSkeleton3D_bk[3] = 0.0;

					TriangulateSkeletonRANSAC(VideoInfo, allSkeleton[skeI], Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI],
						triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D_bk, iSigmaSkeleton, increF, ifps, real2SfM);

					//finalize the frame with guided exhausive search
					if (v3DPoints.size() == 0 || bestAsso.size() == 0)
						continue;

					//see if  more points can be found using reconstructed 3D
					vector<int> CamToTest, FrameMatchToTest;
					for (int cid = 0; cid < nCams; cid++)
						CamToTest.push_back(cid), FrameMatchToTest.push_back(refFid - (int)CamTimeInfo[cid].y);

					vector<int> oPredictedViewMatch, oPredictedFrameMatch, oPredictedDetectionIdMatch;
					oPredictedViewMatch = PredictedViewMatch, oPredictedFrameMatch = PredictedFrameMatch, oPredictedDetectionIdMatch = PredictedDetectionIdMatch;
					PredictedViewMatch.clear(), PredictedFrameMatch.clear(), PredictedDetectionIdMatch.clear();
					TriangulationGuidedPeopleSearch(VideoInfo, CamToTest, FrameMatchToTest, Attri, v3DPoints, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, triangulationThresh);

					//do a final pass through all possible cameras
					if (oPredictedViewMatch != PredictedViewMatch)
					{
						if (debug)
						{
							for (int ii = 0; ii < PredictedViewMatch.size(); ii++)
							{
								int cid = PredictedViewMatch[ii], pid = PredictedDetectionIdMatch[ii];

								Mat img = AllCamImages[cid][0].clone();
								Draw2DCoCo(img, Attri[cid][pid], 1, 1.0, pid);

								CvPoint text_origin = { img.cols / 20,img.rows / 15 }; sprintf(Fname, "%d", cid);
								putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 2);

								namedWindow(Fname, CV_WINDOW_NORMAL); imshow(Fname, img);
							}
							waitKey(0);
						}
						bestAsso.clear(), AllvReProjErrPerInstance[skeI].clear(), v3DPoints.clear();
						TriangulateSkeletonRANSAC(VideoInfo, allSkeleton[skeI], Attri, PredictedViewMatch, PredictedFrameMatch, PredictedDetectionIdMatch, bestAsso, v3DPoints, AllvReProjErrPerInstance[skeI],
							triangulationThresh, RanSac_iterMax, AtLeastThree4Ransac, 1, WeightsHumanSkeleton3D_bk, iSigmaSkeleton, increF, ifps, real2SfM);
					}

					if (bestAsso.size() > 0)
					{
						success = true;
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							allSkeleton[skeI].sSke[refFid].pt3d[jid] = v3DPoints[jid];
							for (auto ii : bestAsso)
							{
								int cid = PredictedViewMatch[ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = PredictedDetectionIdMatch[ii];
								allSkeleton[skeI].sSke[refFid].vPt2D[jid].emplace_back(Attri[cid][did].pt[jid].x, Attri[cid][did].pt[jid].y);
							}
						}
						for (auto ii : bestAsso)
						{
							int cid = PredictedViewMatch[ii], lfid = refFid - (int)CamTimeInfo[cid].y, did = PredictedDetectionIdMatch[ii];
							allSkeleton[skeI].sSke[refFid].vCidFidDid.emplace_back(cid, lfid, did);
						}
						allSkeleton[skeI].sSke[refFid].refFid = refFid, allSkeleton[skeI].sSke[refFid].nPts = nBodyKeyPoints;
						allSkeleton[skeI].sSke[refFid].valid = true;
					}
					else
						int a = 0;
				}
				delete[]AllvReProjErrPerInstance;

				if (!success)
				{
					for (int cid = 0; cid < nCams; cid++)
						AttriOld[cid].clear(), ImgOldPyr[cid].clear();
					printLOG("\n\n\n\n\n**********\nRefresh\n**********\n\n\n\n\n");
					continue;
				}
			}

			//STAGE II: do BA if needed
			{
				printLOG("**********\nWindowBA\n**********\n");

#pragma omp parallel for schedule(dynamic, 1)
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					int cnt = 0;
					for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
						if (allSkeleton[skeI].sSke[inst].valid)
							cnt++;
					if (allSkeleton[skeI].active == 1 && cnt > BAratio*allSkeleton[skeI].nframeSinceLastBA)
					{
						printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, refFid, cnt);
						allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
						allSkeleton[skeI].nframeSinceLastBA = cnt;

						WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
					}
				}
			}

			//Done: swap over the kpts and desc for tracking prepare for next frame
			for (int cid = 0; cid < nCams; cid++)
			{
				AttriOld[cid].clear();
				copy(Attri[cid].begin(), Attri[cid].end(), back_inserter(AttriOld[cid]));
				ImgOldPyr[cid].clear();
				copy(ImgNewPyr[cid].begin(), ImgNewPyr[cid].end(), back_inserter(ImgOldPyr[cid]));
			}

			if ((refFid - startF) % cacheFreq == 0)
			{
				//STAGE IV: do BA 
				printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					int cnt = 0;
					for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
						if (allSkeleton[skeI].sSke[inst].valid)
							cnt++;
					if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].nframeSinceLastBA != cnt)
					{
						printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, refFid, cnt);
						allSkeleton[skeI].lastBAFrame = refFid, allSkeleton[skeI].nBAtimes++;
						allSkeleton[skeI].nframeSinceLastBA = cnt;

						WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
					}
				}

				for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				{
					sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI); FILE *fp = fopen(Fname, "w");
					for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
					{
						if (allSkeleton[skeI].sSke[inst].valid == 0)
							continue;
						fprintf(fp, "%d\n", allSkeleton[skeI].sSke[inst].refFid);
						for (int jid = 0; jid < nBodyKeyPoints; jid++)
						{
							fprintf(fp, "%.6f %.6f %.6f\n", allSkeleton[skeI].sSke[inst].pt3d[jid].x, allSkeleton[skeI].sSke[inst].pt3d[jid].y, allSkeleton[skeI].sSke[inst].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
							fprintf(fp, "%zd ", allSkeleton[skeI].sSke[inst].vCidFidDid.size());
							for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
							{
								int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, did = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].z;

								Point2f pt(allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].x, allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].y);
								LensDistortionPoint(&pt, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
								fprintf(fp, "%d %d %d %.2f %.2f ", cid, fid, did, pt.x, pt.y);
							}
							fprintf(fp, "\n");
						}
					}
					fclose(fp);
				}
			}
		}

		//STAGE IV: do BA 
		{
			printLOG("**********\nWindowBA\n**********\n");
#pragma omp parallel for schedule(dynamic, 1)
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				int cnt = 0, maxValidFid = -1;
				for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
					if (allSkeleton[skeI].sSke[inst].valid)
						maxValidFid = max(maxValidFid, inst), cnt++;
				if (allSkeleton[skeI].active == 1 && allSkeleton[skeI].nframeSinceLastBA != cnt)
				{
					printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, maxValidFid, cnt);
					allSkeleton[skeI].lastBAFrame = stopF, allSkeleton[skeI].nBAtimes++;
					allSkeleton[skeI].nframeSinceLastBA = cnt;

					WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, silent);
				}
			}
		}
	}

	for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
	{
		sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI); FILE *fp = fopen(Fname, "w");
		for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
		{
			if (allSkeleton[skeI].sSke[inst].valid == 0)
				continue;
			fprintf(fp, "%d\n", allSkeleton[skeI].sSke[inst].refFid);
			for (int jid = 0; jid < nBodyKeyPoints; jid++)
			{
				fprintf(fp, "%.6f %.6f %.6f\n", allSkeleton[skeI].sSke[inst].pt3d[jid].x, allSkeleton[skeI].sSke[inst].pt3d[jid].y, allSkeleton[skeI].sSke[inst].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
				fprintf(fp, "%zd ", allSkeleton[skeI].sSke[inst].vCidFidDid.size());
				for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
				{
					int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, did = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].z;

					Point2f pt(allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].x, allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].y);
					//LensDistortionPoint(&pt, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
					fprintf(fp, "%d %d %d %.2f %.2f ", cid, fid, did, pt.x, pt.y);
				}
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}

	delete[]AllCamImages;
	return 0;
}
int SimultaneousTrackingAndAssocation_Vis(char *Path, std::vector<int> &SelectedCamNames, int startF, int stopF, double triangulationThresh, int RanSac_iterMax, int AtLeastThree4Ransac, float max_ratio, float DescThresh, float BAratio, int BAfreq, int reliableTrackRange, double real2SfM, double *WeightsHumanSkeleton3D, double *iSigmaSkeleton, int nBodyKeyPoints = 18)
{
	char Fname[512];
	sprintf(Fname, "%s/Vis/3DTracklet/", Path); makeDir(Fname);

	int nCams = (int)SelectedCamNames.size();
	Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	int selected, temp;
	double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
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

	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	VideoData *VideoInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;

	/*printLOG("Reading keypoints: ");
	vector<BodyImgAtrribute> *Attri = new vector<BodyImgAtrribute>[nCams*stopF];
	for (int cid = 0; cid < nCams; cid++)
	{
		printLOG("%d..", cid);
		for (int fid = startF; fid <= stopF; fid++)
			ReadBodyKeyPointsAndDesc(Path, cid, fid, Attri[cid*stopF + fid]);
	}
	printLOG("...done!\n");*/

	int nMaxPeople = 0;
	vector<bool> hasPerson(10000);
	vector<vector<Point2i> > *MergedTrackletVec = new vector<vector<Point2i> >[nCams];
	for (auto cid : SelectedCamNames)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_18_%d_%d.txt", Path, cid, startF, 3000);
		if (!IsFileExist(Fname))
			return 1;
		std::string line, item;
		std::ifstream file(Fname);
		int pid, fid, did;
		while (std::getline(file, line))
		{
			StringTrim(&line);//remove white space
			if (line.empty())
				break;
			std::stringstream line_stream(line);
			std::getline(line_stream, item, ' ');
			pid = atoi(item.c_str());
			if (pid >= 0)
				hasPerson[pid] = true;

			vector<Point2i> jointTrack;
			if (MergedTrackletVec[cid].size() < pid)
			{
				for (int ii = MergedTrackletVec[cid].size(); ii < pid; ii++)
					MergedTrackletVec[cid].push_back(jointTrack);
			}

			std::getline(file, line);
			std::stringstream line_stream_(line);
			while (!line_stream_.eof())
			{
				std::getline(line_stream_, item, ' ');
				StringTrim(&item);
				fid = atoi(item.c_str());
				std::getline(line_stream_, item, ' ');
				StringTrim(&item);
				did = atoi(item.c_str());
				jointTrack.push_back(Point2i(fid, did));
			}
			jointTrack.pop_back();
			MergedTrackletVec[cid].push_back(jointTrack);
		}
		file.close();

		nMaxPeople = max(nMaxPeople, (int)MergedTrackletVec[cid].size());
	}
	nMaxPeople--;//last one is trash category	
	//nMaxPeople = 5;

	vector<sHumanSkeleton3D > allSkeleton; //skeleton xyz and state of activenesss
	for (int ii = 0; ii < nMaxPeople; ii++)
	{
		sHumanSkeleton3D sSkeI;
		sSkeI.active = 0;
		if (hasPerson[ii])
		{
			sSkeI.sSke.resize(stopF + 1), sSkeI.active = 1;
			for (int jj = 0; jj < stopF; jj++)
				sSkeI.sSke[jj].refFid = -1, sSkeI.sSke[jj].valid = 0, sSkeI.sSke[jj].nPts = nBodyKeyPoints;
		}
		allSkeleton.emplace_back(sSkeI);
	}

	int refFid;
	Point3i cidfiddid; Point2d pt; Point2f cPose[25];
	for (int skeI = 0; skeI < nMaxPeople; skeI++)
	{
		int cnt = 0;
		sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, skeI);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &refFid) != EOF)
		{
			allSkeleton[skeI].sSke[refFid].refFid = refFid;
			allSkeleton[skeI].sSke[refFid].nPts = nBodyKeyPoints;
			allSkeleton[skeI].sSke[refFid].valid = true;
			for (int jid = 0; jid < nBodyKeyPoints; jid++)
			{
				fscanf(fp, "%lf %lf %lf ", &allSkeleton[skeI].sSke[refFid].pt3d[jid].x, &allSkeleton[skeI].sSke[refFid].pt3d[jid].y, &allSkeleton[skeI].sSke[refFid].pt3d[jid].z);
				fscanf(fp, "%d ", &allSkeleton[skeI].sSke[refFid].nvis);
				for (size_t ii = 0; ii < allSkeleton[skeI].sSke[refFid].nvis; ii++)
				{
					fscanf(fp, "%d %d %d %lf %lf ", &cidfiddid.x, &cidfiddid.y, &cidfiddid.z, &pt.x, &pt.y);
					if (jid == 0)
						allSkeleton[skeI].sSke[refFid].vCidFidDid.push_back(cidfiddid);

					//pt.x = Attri[cidfiddid.x*stopF + cidfiddid.y][cidfiddid.z].pt[jid].x, pt.y = Attri[cidfiddid.x*stopF + cidfiddid.y][cidfiddid.z].pt[jid].y;
					LensCorrectionPoint(&pt, VideoInfo[cidfiddid.x].VideoInfo[cidfiddid.y].K, VideoInfo[cidfiddid.x].VideoInfo[cidfiddid.y].distortion);
					allSkeleton[skeI].sSke[refFid].vPt2D[jid].push_back(pt);
				}
				cnt++;
			}
		}
		fclose(fp);
		if (cnt < 10)
		{
			allSkeleton[skeI].active = 0;
			allSkeleton[skeI].sSke.clear();
		}
	}

	bool refine = false;
	if (refine)
	{
		printLOG("**********\nWindowBA\n**********\n");
		//#pragma omp parallel for schedule(dynamic, 1)
		for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
		{
			double ifps = 1.0 / 60;
			int cnt = 0, maxValidFid = -1;
			for (int inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
				if (allSkeleton[skeI].sSke[inst].valid)
					maxValidFid = max(maxValidFid, inst), cnt++;
			if (cnt > 0)
			{
				printLOG("\nPerson %d [%d-->%d] (%d valid frames)\n", skeI, allSkeleton[skeI].startActiveF, maxValidFid, cnt);
				WindowSkeleton3DBundleAdjustment(VideoInfo, nCams, allSkeleton[skeI], 1, 1, WeightsHumanSkeleton3D, iSigmaSkeleton, real2SfM, ifps, reliableTrackRange, true);
			}
		}

		sprintf(Fname, "%s/Skeleton_%d_%d_/", Path, startF, stopF); makeDir(Fname);
		for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
		{
			sprintf(Fname, "%s/Skeleton_%d_%d_/%.4d.txt", Path, startF, stopF, skeI); FILE *fp = fopen(Fname, "w");
			for (size_t inst = 0; inst < allSkeleton[skeI].sSke.size(); inst++)
			{
				if (allSkeleton[skeI].sSke[inst].valid == 0)
					continue;
				fprintf(fp, "%d\n", allSkeleton[skeI].sSke[inst].refFid);
				for (int jid = 0; jid < nBodyKeyPoints; jid++)
				{
					fprintf(fp, "%.6f %.6f %.6f\n", allSkeleton[skeI].sSke[inst].pt3d[jid].x, allSkeleton[skeI].sSke[inst].pt3d[jid].y, allSkeleton[skeI].sSke[inst].pt3d[jid].z);// , HypovReProjErr[hypoId][jid]);
					fprintf(fp, "%zd ", allSkeleton[skeI].sSke[inst].vCidFidDid.size());
					for (size_t ii = 0; ii < allSkeleton[skeI].sSke[inst].vCidFidDid.size(); ii++)
					{
						int cid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].y, did = allSkeleton[skeI].sSke[inst].vCidFidDid[ii].z;

						Point2f pt(allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].x, allSkeleton[skeI].sSke[inst].vPt2D[jid][ii].y);
						LensDistortionPoint(&pt, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
						fprintf(fp, "%d %d %d %.2f %.2f ", cid, fid, did, pt.x, pt.y);
					}
					fprintf(fp, "\n");
				}
			}
			fclose(fp);
		}
	}

	static cv::Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 128, 255), Scalar(0, 255, 255), Scalar(0, 255, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(255, 255, 255) };

	if (1)
	{
		bool debug = false;
		double resizeFactor = 0.5;

		for (int cid = 0; cid < nCams; cid++)
			sprintf(Fname, "%s/Vis/3DTracklet/%d", Path, cid), makeDir(Fname);
		/*VideoWriter *writer = new VideoWriter[nCams];
		for (int ii = 0; ii < nCams; ii++)
		{
			CvSize size;
			size.width = (int)(resizeFactor * 1920), size.height = (int)(resizeFactor * 1080);
			sprintf(Fname, "%s/Vis/3DTracklet/%d_%.4d_%.4d.avi", Path, ii, startF, stopF);
			writer[ii].open(Fname, CV_FOURCC('X', 'V', 'I', 'D'), 30, size);
		}*/
		for (int refFid = startF; refFid <= stopF; refFid++)
		{
			int cnt = 0;
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
				if (allSkeleton[skeI].active  && allSkeleton[skeI].sSke[refFid].valid)
					cnt++;
			if (cnt == 0)
				continue;
			printf("%d..", refFid);
			vector<Mat> vImg;
			for (int ii = 0; ii < nCams; ii++)
			{
				Mat img, rimg;
				int rfid = refFid - (int)CamTimeInfo[ii].y;
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, ii, rfid); img = imread(Fname);
				resize(img, rimg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_AREA);
				CvPoint text_origin = { rimg.cols / 30, rimg.cols / 30 };
				sprintf(Fname, "%d", rfid), putText(rimg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*rimg.cols / 640, colors[0], 2);
				vImg.emplace_back(rimg);
			}

			Point2d joints2D[25];
			for (int skeI = 0; skeI < (int)allSkeleton.size(); skeI++)
			{
				if (allSkeleton[skeI].active == 0 || allSkeleton[skeI].sSke[refFid].valid == 0)
					continue;

				for (int ii = 0; ii < allSkeleton[skeI].sSke[refFid].vCidFidDid.size(); ii++)
				{
					int cid = allSkeleton[skeI].sSke[refFid].vCidFidDid[ii].x, fid = allSkeleton[skeI].sSke[refFid].vCidFidDid[ii].y;

					for (int jid = 0; jid < nBodyKeyPoints; jid++)
						joints2D[jid].x = 0, joints2D[jid].y = 0;
					for (int jid = 0; jid < nBodyKeyPoints; jid++)
					{
						if (IsValid3D(allSkeleton[skeI].sSke[refFid].pt3d[jid]))
						{
							ProjectandDistort(allSkeleton[skeI].sSke[refFid].pt3d[jid], &joints2D[jid], VideoInfo[cid].VideoInfo[fid].P, VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
							joints2D[jid].x *= resizeFactor, joints2D[jid].y *= resizeFactor;
						}
					}

					Draw2DCoCoJoints(vImg[cid], joints2D, nBodyKeyPoints, 1.0, 1.0, &colors[skeI % 8]);
					if (debug)
						sprintf(Fname, "C:/temp/%d.jpg", cid), imwrite(Fname, vImg[cid]);
				}
			}
			for (int ii = 0; ii < nCams; ii++)
			{
				if (debug)
					sprintf(Fname, "C:/temp/%d.jpg", ii), imwrite(Fname, vImg[ii]);
				sprintf(Fname, "%s/Vis/3DTracklet/%d/%.4d.jpg", Path, ii, refFid); imwrite(Fname, vImg[ii]);
				//writer[ii] << vImg[ii];
			}
		}
		//for (int ii = 0; ii < nCams; ii++)
		//	writer[ii].release();
	}
	else
	{
		Mat img;
		int refFid = -1, orefFid = refFid, scid = -1, oscid = -1, skeI = -1, oSkeI = skeI, step = 1;
		Point2d joints2D[25];

		for (int kk = 0; kk < nMaxPeople && (refFid == orefFid || scid == oscid || skeI == oSkeI); kk++)
		{
			for (int jj = 0; jj < nCams && (refFid == orefFid || scid == oscid || skeI == oSkeI); jj++)
			{
				for (int ii = startF; ii <= stopF && (refFid == orefFid || scid == oscid || skeI == oSkeI); ii++)
				{
					if (allSkeleton[kk].active == 0 || allSkeleton[kk].sSke[ii].valid == 0)
						continue;

					int sid = -1;
					for (int ll = 0; ll < allSkeleton[kk].sSke[ii].vCidFidDid.size() && sid == -1; ll++)
						if (allSkeleton[kk].sSke[ii].vCidFidDid[ll].x == jj)
							sid = ll;
					if (sid == -1)
						continue;
					else
						refFid = ii, scid = jj, skeI = kk;
				}
			}
		}

		cvNamedWindow("Tracklet", CV_WINDOW_NORMAL);
		cvCreateTrackbar("cPid", "Tracklet", &skeI, nMaxPeople, NULL);
		cvCreateTrackbar("cid", "Tracklet", &scid, nCams - 1, NULL);
		cvCreateTrackbar("fid", "Tracklet", &refFid, stopF, NULL);

		vector<BodyImgAtrribute> Attri;
		while (true)
		{
			cvSetTrackbarPos("cPid", "Tracklet", skeI);
			cvSetTrackbarPos("cid", "Tracklet", scid);
			cvSetTrackbarPos("fid", "Tracklet", refFid);

			int sid = -1;
			for (int ii = 0; ii < allSkeleton[skeI].sSke[refFid].vCidFidDid.size() && sid == -1; ii++)
			{
				if (allSkeleton[skeI].sSke[refFid].vCidFidDid[ii].x == scid)
					sid = ii;
			}

			if (sid > -1)
			{
				int cid = allSkeleton[skeI].sSke[refFid].vCidFidDid[sid].x, fid = allSkeleton[skeI].sSke[refFid].vCidFidDid[sid].y, did = allSkeleton[skeI].sSke[refFid].vCidFidDid[sid].z;
				if (refFid != orefFid || skeI != oSkeI || scid != oscid)
				{
					orefFid = refFid, oscid = scid, oSkeI = skeI;
					sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid, fid); img = imread(Fname);

					Attri.clear();
					ReadBodyKeyPointsAndDesc(Path, cid, fid, Attri);
					for (int jid = 0; jid < nBodyKeyPoints; jid++)
						joints2D[jid].x = Attri[did].pt[jid].x, joints2D[jid].y = Attri[did].pt[jid].y;

					/*for (int jid = 0; jid < nBodyKeyPoints; jid++)
					{
						joints2D[jid] = allSkeleton[skeI].sSke[refFid].vPt2D[jid][sid];
						if (joints2D[jid].x > 0)
							LensDistortionPoint(&joints2D[jid], VideoInfo[cid].VideoInfo[fid].K, VideoInfo[cid].VideoInfo[fid].distortion);
					}*/

					CvPoint text_origin = { img.cols / 30, img.cols / 30 };
					sprintf(Fname, "%d", fid), putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, colors[0], 2);
					Draw2DCoCoJoints(img, joints2D, nBodyKeyPoints, 1.0, 1.0, &colors[skeI % 8]);
				}
			}

			imshow("Tracklet", img);
			if (waitKey(1) == 27)
				break;
		}
	}

	return 0;
}

int convertGT2Tracklet(char *Path)
{
	char Fname[512];

	int offset[] = { 16, 16, 30, 33, 13, 12, 12 };

	Point3i cidfiddid;
	vector<vector<Point2i> > Cid_pid_fiddid[7];

	VideoData VideoInfo[7];
	for (int cid = 0; cid < 7; cid++)
	{
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;
		//	for (int fid = 1; fid < 10000; fid++)
		//		CopyCamereInfo(VideoInfo[cid].VideoInfo[0], VideoInfo[cid].VideoInfo[fid], true);
		//	WriteVideoDataI(Path, VideoInfo[cid], cid, 0, 3000, 0);
	}


	Point3i fidpidcid;
	vector<vector<Point2i> >vPid_fidcid;
	sprintf(Fname, "%s/del.txt", Path); FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %d %d ", &fidpidcid.x, &fidpidcid.y, &fidpidcid.z) != EOF)
	{
		if (fidpidcid.y > vPid_fidcid.size())
		{
			vector<Point2i> dummy;
			for (int ii = 0; ii <= fidpidcid.y; ii++)
				vPid_fidcid.emplace_back(dummy);
		}
		vPid_fidcid[fidpidcid.y].emplace_back(fidpidcid.x, fidpidcid.z);
	}
	fclose(fp);

	std::string line, item;
	for (int fid = 5; fid <= 500; fid += 5)
	{
		printf("%d..\n", fid);

		vector<Point2f> vUV[7];
		vector<float> vConf[7];
		for (int cid = 0; cid < 7; cid++)
		{
			int rfid = (fid - 5) * 6 + offset[cid];
			readOpenPoseJson(Path, cid, rfid, vUV[cid], vConf[cid]);
			for (int pid = 0; pid < vUV[cid].size() / 18; pid++)
			{
				for (int jid = 0; jid < 18; jid++)
				{
					Point2d pt = vUV[cid][pid * 18 + jid];
					if (vConf[cid][pid * 18 + jid] == 0.0)
						vUV[cid][pid * 18 + jid] = Point2d(0, 0);
					else
					{
						LensCorrectionPoint(&pt, VideoInfo[cid].VideoInfo[rfid].K, VideoInfo[cid].VideoInfo[rfid].distortion);
						vUV[cid][pid * 18 + jid] = pt;
					}
				}
			}
		}

		sprintf(Fname, "%s/Wildtrack_dataset/annotations_positions/%.8d.json", Path, fid);
		std::ifstream file(Fname);
		while (getline(file, line))
		{
			StringTrim(&line);
			if (line.empty() || line[0] == '[' || line[0] == ']' || line[0] == '{' || line[0] == '}')
				continue;

			std::stringstream line_stream(line);
			line_stream >> item >> item; //person Id
			item.pop_back();
			int personId = atoi(item.c_str());
			if (personId == 1)\
				int a = 0;
			getline(file, line);
			getline(file, line); //views
			getline(file, line); //views
			for (int ii = 0; ii < 7; ii++)
			{
				getline(file, line); //viewNumId
				std::stringstream line_stream2(line);
				line_stream2 >> item >> item; //
				item.pop_back();
				cidfiddid.x = atoi(item.c_str());
				cidfiddid.y = (fid - 5) * 6 + offset[cidfiddid.x];

				getline(file, line); //xmax
				std::stringstream line_stream3(line);
				line_stream3 >> item >> item;
				item.pop_back();
				int xmax = atoi(item.c_str());

				getline(file, line); //xmin
				std::stringstream line_stream4(line);
				line_stream4 >> item >> item;
				item.pop_back();
				int xmin = atoi(item.c_str());

				getline(file, line); //ymax
				std::stringstream line_stream5(line);
				line_stream5 >> item >> item;
				item.pop_back();
				int ymax = atoi(item.c_str());

				getline(file, line); //ymin
				std::stringstream line_stream6(line);
				line_stream6 >> item >> item;
				int ymin = atoi(item.c_str());

				getline(file, line); //},
				getline(file, line); //{

				if (xmin == -1 && xmax == -1)
					continue;

				bool found = false;
				for (int ii = 0; ii < vPid_fidcid[personId].size() && !found; ii++)
				{
					if (vPid_fidcid[personId][ii].x == fid && vPid_fidcid[personId][ii].y == cidfiddid.x)
						found = true;
				}
				if (found)
					continue;

				bool debug = false;
				if (debug)
				{
					sprintf(Fname, "D:/WildTrack/Wildtrack_dataset/Image_subsets/C%d/%.8d.png", cidfiddid.x + 1, fid); Mat img = imread(Fname);
					CvPoint text_origin = { img.cols / 20,img.rows / 15 }; sprintf(Fname, "cid: %d fid: %d", cidfiddid.x, fid);
					putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 2);
					rectangle(img, Point2i(xmin, ymin), Point2i(xmax, ymax), cv::Scalar(0, 255, 0), 2, 8, 0);

					for (int pid = 0; pid < vUV[cidfiddid.x].size() / 18; pid++)
						Draw2DCoCoJoints(img, &vUV[cidfiddid.x][18 * pid], 1, 1.0, 18, &vConf[cidfiddid.x][18 * pid], pid);
					cv::namedWindow("X", CV_WINDOW_NORMAL);
					cv::imshow("X", img), cv::waitKey(0);
				}

				int bestId = -1, bestCnt = 0;
				for (int pid = 0; pid < vUV[cidfiddid.x].size() / 18; pid++)
				{
					int cnt = 0;
					for (int jid = 0; jid < 18; jid++)
					{
						if (vUV[cidfiddid.x][pid * 18 + jid].x >= xmin && vUV[cidfiddid.x][pid * 18 + jid].x <= xmax && vUV[cidfiddid.x][pid * 18 + jid].y >= ymin && vUV[cidfiddid.x][pid * 18 + jid].y <= ymax)
							cnt++;
					}
					if (cnt > bestCnt)
					{
						bestCnt = cnt;
						bestId = pid;
					}
				}
				if (bestId > -1)
				{
					if (Cid_pid_fiddid[cidfiddid.x].size() <= personId)
					{
						for (int ii = Cid_pid_fiddid[cidfiddid.x].size(); ii <= personId; ii++)
						{
							vector<Point2i> dummy;
							Cid_pid_fiddid[cidfiddid.x].push_back(dummy);
						}
					}
					Cid_pid_fiddid[cidfiddid.x][personId].emplace_back(cidfiddid.y, bestId);
				}
				else
					printf("Cannot associate person %d view %d\n", personId, cidfiddid.x);
			}
			getline(file, line); //]
			getline(file, line); //}
		}
		file.close();
	}

	for (int cid = 0; cid < 7; cid++)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_18_%d_%d.txt", Path, cid, 0, 3000);
		FILE *fp = fopen(Fname, "w");
		for (int pid = 0; pid < Cid_pid_fiddid[cid].size(); pid++)
		{
			if (Cid_pid_fiddid[cid][pid].size() > 0)
			{
				fprintf(fp, "%d %zd\n", pid, Cid_pid_fiddid[cid][pid].size());
				for (int inst = 0; inst < Cid_pid_fiddid[cid][pid].size(); inst++)
					fprintf(fp, "%d %d ", Cid_pid_fiddid[cid][pid][inst].x, Cid_pid_fiddid[cid][pid][inst].y);
				fprintf(fp, "\n");
			}
		}
		fprintf(fp, "-1 0\n-1 -1\n");//trash catogory
		fclose(fp);
	}
	return 0;
}
int cleanGT2Tracklet(char *Path)
{
	char Fname[512];


	struct Cid_box {
		int cid, xmax, xmin, ymax, ymin;
	};
	struct People
	{
		int pid;
		vector<Cid_box> pdet;
	};
	Cid_box pdet;
	People PeopleI;
	vector<People> allPeople;

	std::string line, item;
	int fid = 1, ofid = 0, step = 1, cPid = 0, ocPid = 0, cVid = 0;
	Mat img;

	cvNamedWindow("Tracklet", CV_WINDOW_NORMAL);
	cvCreateTrackbar("fid", "Tracklet", &fid, 1000, NULL);
	cvCreateTrackbar("cPid", "Tracklet", &cPid, 50, NULL);
	cvCreateTrackbar("Step", "Tracklet", &step, 2, NULL);

	while (true)
	{
		if (fid != ofid)
		{
			ofid = fid;
			cPid = 0, ocPid = -1, cVid = 0;
			allPeople.clear();

			printf("%d..\n", fid * 5);
			sprintf(Fname, "%s/Wildtrack_dataset/annotations_positions/%.8d.json", Path, fid * 5);
			std::ifstream file(Fname);
			while (getline(file, line))
			{
				StringTrim(&line);
				if (line.empty() || line[0] == '[' || line[0] == ']' || line[0] == '{' || line[0] == '}')
					continue;

				std::stringstream line_stream(line);
				line_stream >> item >> item; //person Id
				item.pop_back();
				PeopleI.pid = atoi(item.c_str());
				PeopleI.pdet.clear();
				getline(file, line);
				getline(file, line); //views
				getline(file, line); //views
				for (int ii = 0; ii < 7; ii++)
				{
					getline(file, line); //viewNumId
					std::stringstream line_stream2(line);
					line_stream2 >> item >> item; //
					item.pop_back();
					pdet.cid = atoi(item.c_str());

					getline(file, line); //xmax
					std::stringstream line_stream3(line);
					line_stream3 >> item >> item;
					item.pop_back();
					pdet.xmax = atoi(item.c_str());

					getline(file, line); //xmin
					std::stringstream line_stream4(line);
					line_stream4 >> item >> item;
					item.pop_back();
					pdet.xmin = atoi(item.c_str());

					getline(file, line); //ymax
					std::stringstream line_stream5(line);
					line_stream5 >> item >> item;
					item.pop_back();
					pdet.ymax = atoi(item.c_str());

					getline(file, line); //ymin
					std::stringstream line_stream6(line);
					line_stream6 >> item >> item;
					pdet.ymin = atoi(item.c_str());

					getline(file, line); //},
					getline(file, line); //{

					if (pdet.xmin == -1 && pdet.xmax == -1)
						continue;

					PeopleI.pdet.push_back(pdet);
				}
				getline(file, line); //]
				getline(file, line); //}

				allPeople.push_back(PeopleI);
			}
			file.close();
		}

		if (step != 1 || cPid != ocPid)
		{
			if (step == 2)
			{
				cVid++;
				if (cVid > allPeople[cPid].pdet.size() - 1)
				{
					cPid++, cVid = 0;
					if (cPid > allPeople.size() - 1)
					{
						cPid = 0, fid += 1, step = 1;
						continue;
					}
				}
			}
			if (step == 0)
			{
				cVid--;
				if (cVid == -1)
				{
					cPid--;
					if (cPid == -1)
					{
						cVid = 0, cPid = 0, fid -= 1, step = 1;
						if (fid == -1)
							fid = 0;
						continue;
					}
					cVid = allPeople[cPid].pdet.size() - 1;
				}
			}
			step = 1;

			if (cPid != ocPid)
				ocPid = cPid;
			if (cPid > allPeople.size() - 1)
			{
				cPid = 0, fid += 1, step = 1;
				continue;
			}
			if (cPid == -1)
			{
				cVid = 0, cPid = 0, fid -= 1, step = 1;
				if (fid == -1)
					fid = 0;
				continue;
			}

			sprintf(Fname, "D:/WildTrack/Wildtrack_dataset/Image_subsets/C%d/%.8d.png", allPeople[cPid].pdet[cVid].cid + 1, fid * 5);
			img = imread(Fname);
			CvPoint text_origin = { img.cols / 20,img.rows / 15 }; sprintf(Fname, "Pid: %d (%d/%zd) Cid: %d (%d/%zd). Space: once. Enter:all", allPeople[cPid].pid, cPid, allPeople.size(), allPeople[cPid].pdet[cVid].cid, cVid, allPeople[cPid].pdet.size());
			putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 2);
			rectangle(img, Point2i(allPeople[cPid].pdet[cVid].xmin, allPeople[cPid].pdet[cVid].ymin), Point2i(allPeople[cPid].pdet[cVid].xmax, allPeople[cPid].pdet[cVid].ymax), cv::Scalar(0, 255, 0), 2, 8, 0);
		}

		imshow("Tracklet", img);
		int key = waitKey(30);
		if (key == 27)
			break;
		else if (key == 13) //enter
		{
			sprintf(Fname, "%s/del.txt", Path); FILE *fp = fopen(Fname, "a");
			for (int ii = 0; ii < allPeople[cPid].pdet.size(); ii++)
			{
				fprintf(fp, "%d %d %d\n", 5 * fid, allPeople[cPid].pid, allPeople[cPid].pdet[ii].cid);
				printLOG("%d %d %d\n", 5 * fid, allPeople[cPid].pid, allPeople[cPid].pdet[ii].cid);
			}
			fclose(fp);

			cPid++, cVid = 0;
			if (cPid > allPeople.size() - 1)
			{
				cPid = 0, fid += 1, step = 1;
				continue;
			}
		}
		else if (key == 32) //space
		{
			sprintf(Fname, "%s/del.txt", Path); FILE *fp = fopen(Fname, "a");
			fprintf(fp, "%d %d %d\n", 5 * fid, allPeople[cPid].pid, allPeople[cPid].pdet[cVid].cid);
			fclose(fp);
			printLOG("%d %d %d\n", 5 * fid, allPeople[cPid].pid, allPeople[cPid].pdet[cVid].cid);
		}
		else if (key == 'd')//right arrow
			step = 2;
		else if (key == 'a') //left arrow
			step = 0;
		cvSetTrackbarPos("fid", "Tracklet", fid);
		cvSetTrackbarPos("cPid", "Tracklet", cPid);
		cvSetTrackbarPos("Step", "Tracklet", step);
	}

	return 0;
}

void ComputeNormalFormPlanePointCloud(vector<Point3d> &vpts, double *normal)
{
	int npts = (int)vpts.size();
	double centerX = 0, centerY = 0, centerZ = 0;

	for (int ii = 0; ii < npts; ii++)
		centerX += vpts[ii].x, centerY += vpts[ii].y, centerZ += vpts[ii].z;
	centerX /= npts, centerY /= npts, centerZ /= npts;

	MatrixXdr X_(npts, 3);
	for (int ii = 0; ii < npts; ii++)
		X_(ii, 0) = vpts[ii].x - centerX, X_(ii, 1) = vpts[ii].y - centerY, X_(ii, 2) = vpts[ii].z - centerZ;

	MatrixXdr Cov = X_.transpose()*X_;
	JacobiSVD<MatrixXdr> svd(Cov, ComputeFullV);
	MatrixXdr V = svd.matrixV();

	normal[0] = V(0, 2), normal[1] = V(1, 2), normal[2] = V(2, 2);

	return;
}
int CollectNearestViewsBasedOnSkeleton(char *Path, vector<int> &sCams, VideoData *VideoInfo, int startF, int stopF, int increF, int selectedPersonId, int skeletonPointFormat, int maxIntraLength = 300, double real2SfM = 1.0, double maxCamDistane = 5000)
{
	int nPeople = 14;
	char Fname[512]; FILE *fp = 0;

	//Read calib info
	int nCams = *std::max_element(std::begin(sCams), std::end(sCams)) + 1;

	vector<int> TimeStamp(nCams);
	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); fp = fopen(Fname, "r");
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
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	//Read human skeleton
	printLOG("Reading 3D skeleton....");
	int nvalidFrames = 0;
	vector<HumanSkeleton3D *> allSkeletons;
	for (int personId = 0; personId < nPeople; personId++)
	{
		printLOG("%d..", personId);
		HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
		for (int refFid = startF; refFid <= stopF; refFid += increF)
		{
			int  nValidJoints = 0, temp = (refFid - startF) / increF;
			sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, personId, refFid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, personId, refFid);
				if (IsFileExist(Fname) == 0)
					continue;
			}
			fp = fopen(Fname, "r");
			int  rcid, rfid, nvis, inlier; double u, v, s, avg_error;
			for (int jid = 0; jid < skeletonPointFormat; jid++)
			{
				fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
				for (int kk = 0; kk < nvis; kk++)
				{
					fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
					Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(rcid, -1));
					Skeletons[temp].vPt2D[jid].push_back(Point2d(u, v));
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
		}
		allSkeletons.push_back(Skeletons);
	}
	printLOG("done\n");

	//Compute front facing camera based on skeleton torso's normal
	int TorsoJointId[4];
	if (skeletonPointFormat == 18)
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 8, TorsoJointId[3] = 11;
	else
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 9, TorsoJointId[3] = 12;

	struct CidFidAngleDist
	{
		CidFidAngleDist(int cid_, int fid_, double angle_, double dist_, double cenX_, double cenY_)
		{
			cid = cid_, fid = fid_;
			angle = angle_, dist = dist_, cenX = cenX_, cenY = cenY_;
		}
		int cid, fid;
		double angle, dist, cenX, cenY;
	};
	vector<CidFidAngleDist>  *NearestCamPerSkeleton = new vector<CidFidAngleDist>[(stopF - startF) / increF + 1];
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int lfid = (refFid - startF) / increF;
		if (IsValid3D(allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[0]]) && IsValid3D(allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]]) && IsValid3D(allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[2]]) && IsValid3D(allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[3]]))
		{
			double vec01[3] = { allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[0]].x - allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]].x,
				allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[0]].y - allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]].y,
				allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[0]].z - allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]].z };
			double vec13[3] = { allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]].x - allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[3]].x,
				allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]].y - allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[3]].y,
				allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[1]].z - allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[3]].z };

			double normalVec_[3];
			cross_product(vec01, vec13, normalVec_);
			normalize(normalVec_);

			double meanTorso[3] = { 0, 0,0 };
			for (int ii = 0; ii < 4; ii++)
				meanTorso[0] += allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[ii]].x,
				meanTorso[1] += allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[ii]].y,
				meanTorso[2] += allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[ii]].z;
			meanTorso[0] /= 4.0, meanTorso[1] /= 4.0, meanTorso[2] /= 4.0;

			vector<Point3d> vpts;
			for (int ii = 0; ii < 4; ii++)
				vpts.push_back(allSkeletons[selectedPersonId][lfid].pt3d[TorsoJointId[ii]]);

			double normalVec[3];
			ComputeNormalFormPlanePointCloud(vpts, normalVec);
			normalize(normalVec);

			if (dotProduct(normalVec, normalVec_) < 0)
				for (int ii = 0; ii < 3; ii++)
					normalVec[ii] *= -1.0;

			bool debug = false;
			Mat img;

			vector<CidFidAngleDist> lcid_rfid;
			vector<size_t> indexList;
			vector<double> unsorted, sorted;
			for (int lcid = 0; lcid < sCams.size(); lcid++)
			{
				//let's assume the novel camera is sync to the earliest camera in your list
				double ts = 1.0* refFid / CamTimeInfo[refCid].x; // t = alpha*(f+rs*row) + beta*alpha_ref
				int fid_nr = (ts - CamTimeInfo[sCams[lcid]].y / CamTimeInfo[refCid].x)*CamTimeInfo[sCams[lcid]].x;

				bool visible = false;
				for (int jid = 0; jid < skeletonPointFormat && !visible; jid++)
				{
					for (int i = 0; i < allSkeletons[selectedPersonId][lfid].vViewID_rFid[jid].size() && !visible; i++)
						if (allSkeletons[selectedPersonId][lfid].vViewID_rFid[jid][i].x == sCams[lcid])
							visible = true;
				}
				if (!visible || fid_nr<startF || fid_nr>stopF)
					continue;
				CameraData *Cam_nr = &VideoInfo[sCams[lcid]].VideoInfo[fid_nr];
				if (!Cam_nr[0].valid)
					continue;

				double cenX = 0, cenY = 0; int npts = 0;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
				{
					for (int ii = 0; ii < allSkeletons[selectedPersonId][lfid].vViewID_rFid[jid].size(); ii++)
						if (allSkeletons[selectedPersonId][lfid].vViewID_rFid[jid][ii].x == sCams[lcid])
							cenX += allSkeletons[selectedPersonId][lfid].vPt2D[jid][ii].x, cenY += allSkeletons[selectedPersonId][lfid].vPt2D[jid][ii].y, npts++;
				}
				cenX = cenX / npts, cenY = cenY / npts;

				Vector4i bb;
				int minX = 1920, maxX = 0, minY = 1080, maxY = 0;
				for (int jid = 0; jid < skeletonPointFormat; jid++)
				{
					if (IsValid3D(allSkeletons[selectedPersonId][lfid].pt3d[jid]))
					{
						Point2d pt;
						ProjectandDistort(allSkeletons[selectedPersonId][lfid].pt3d[jid], &pt, Cam_nr[0].P, Cam_nr[0].K, Cam_nr[0].distortion);
						minX = min(minX, pt.x), maxX = max(maxX, pt.x), minY = min(minY, pt.y), maxY = max(maxY, pt.y);
					}
				}
				bb[0] = minX, bb[1] = minY, bb[2] = maxX, bb[3] = maxY;
				if (debug)
				{
					sprintf(Fname, "%s/%d/%.4d.jpg", Path, sCams[lcid], fid_nr);
					img = imread(Fname);
					rectangle(img, Point2i(bb[0], bb[1]), Point2i(bb[2], bb[3]), cv::Scalar(0, 0, 255), 2, 8, 0);
					namedWindow("X", CV_WINDOW_NORMAL);
				}

				bool found = false;
				for (int personId = 0; personId < nPeople && !found; personId++)
				{
					int minX = 1920, maxX = 0, minY = 1080, maxY = 0;
					for (int jid = 0; jid < skeletonPointFormat; jid++)
					{
						for (int ii = 0; ii < allSkeletons[personId][lfid].vViewID_rFid[jid].size(); ii++)
						{
							if (allSkeletons[personId][lfid].vViewID_rFid[jid][ii].x == sCams[lcid])
								minX = min(minX, allSkeletons[personId][lfid].vPt2D[jid][ii].x), maxX = max(maxX, allSkeletons[personId][lfid].vPt2D[jid][ii].x), minY = min(minY, allSkeletons[personId][lfid].vPt2D[jid][ii].y), maxY = max(maxY, allSkeletons[personId][lfid].vPt2D[jid][ii].y);
						}
					}
					if (debug)
					{
						rectangle(img, Point2i(minX, minY), Point2i(maxX, maxY), cv::Scalar(0, 255, 0), 2, 8, 0);
						imshow("X", img); waitKey(0);
					}

					if (minX != 1920 && maxX != 0 && minY != 1080 && maxY != 0 && personId != selectedPersonId && personId < 6)
					{
						//if (cenX > minX && cenX<maxX && cenY>minY && cenY < maxY)
						Point2i p1(bb[0], bb[1]), p2(bb[2], bb[3]), p3(minX, minY), p4(maxX, maxY);
						if (OverlappingArea(p1, p2, p3, p4) > 0)
							found = true;
					}
				}
				if (found)
				{
					unsorted.push_back(1); //ascending order
					lcid_rfid.push_back(CidFidAngleDist(sCams[lcid], fid_nr, -1, 9999, cenX, cenY));
				}
				else
				{
					double uv1_nr[3] = { Cam_nr[0].width / 2, Cam_nr[0].height / 2, 1 }, rayDir_nr[3];
					getRayDir(rayDir_nr, Cam_nr[0].invK, Cam_nr[0].R, uv1_nr);
					normalize(rayDir_nr);

					double angle = dotProduct(normalVec, rayDir_nr);
					double dist = sqrt(pow(Cam_nr[0].camCenter[0] - meanTorso[0], 2) + pow(Cam_nr[0].camCenter[1] - meanTorso[1], 2) + pow(Cam_nr[0].camCenter[2] - meanTorso[2], 2));

					unsorted.push_back(-angle); //ascending order
					lcid_rfid.push_back(CidFidAngleDist(sCams[lcid], fid_nr, angle, dist, cenX, cenY));
				}
			}

			SortWithIndex(unsorted, sorted, indexList);

			for (int lcid = 0; lcid < indexList.size(); lcid++)
				NearestCamPerSkeleton[lfid].push_back(lcid_rfid[indexList[lcid]]);
		}
	}

	sprintf(Fname, "%s/People/@%d/%d/nearestViews.txt", Path, increF, selectedPersonId); fp = fopen(Fname, "w");
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int lfid = (refFid - startF) / increF;
		fprintf(fp, "%d %zd ", refFid, NearestCamPerSkeleton[lfid].size());
		for (int ii = 0; ii < NearestCamPerSkeleton[lfid].size(); ii++)
			fprintf(fp, "%d %d %.3f %.3f %.3f %.3f ", NearestCamPerSkeleton[lfid][ii].cid, NearestCamPerSkeleton[lfid][ii].fid, NearestCamPerSkeleton[lfid][ii].angle, NearestCamPerSkeleton[lfid][ii].dist, NearestCamPerSkeleton[lfid][ii].cenX, NearestCamPerSkeleton[lfid][ii].cenY);
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]NearestCamPerSkeleton;
	return 0;
}
int VisualizeNearestViewsBasedOnSkeleton(char *Path, int nCams, int startF, int stopF, int increF, int selectedPersonId)
{
	char Path2[] = { "E:/A1" };
	struct CidFidAngleDist
	{
		CidFidAngleDist(int cid_, int fid_, double angle_, double dist_)
		{
			cid = cid_, fid = fid_;
			angle = angle_, dist = dist_;
		}
		int cid, fid;
		double angle, dist;
	};

	char Fname[512];

	int nn, refFid, rcid, rfid;
	double angle, dist, cenX, cenY;
	vector<CidFidAngleDist>  *NearestCamPerSkeleton = new vector<CidFidAngleDist>[(stopF - startF) / increF + 1];
	sprintf(Fname, "%s/People/@%d/%d/nearestViews.txt", Path, increF, selectedPersonId); FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %d ", &refFid, &nn) != EOF)
	{
		int lfid = (refFid - startF) / increF;
		for (int ii = 0; ii < nn; ii++)
		{
			fscanf(fp, "%d %d %lf %lf %lf %lf ", &rcid, &rfid, &angle, &dist, &cenX, &cenY);
			NearestCamPerSkeleton[lfid].push_back(CidFidAngleDist(rcid, rfid, angle, dist));
		}
	}
	fclose(fp);

	refFid = startF;
	int  orefFid = startF - 1, rankId = 0, orankId = -1;
	Mat img;
	namedWindow("DirectorView", WINDOW_NORMAL);
	createTrackbar("RefFrame", "DirectorView", &refFid, stopF, NULL);
	createTrackbar("Rank", "DirectorView", &rankId, nCams, NULL);

	bool first = true, go = false;
	while (true)
	{
		bool changed = false, frameChanged = false;
		if (go)
			refFid++;
		if (orefFid != refFid)
			orefFid = refFid, changed = true, frameChanged = true;
		if (orankId != rankId)
			orankId = rankId, changed = true;

		if (changed)
		{
			int lfid = (refFid - startF) / increF;
			if (NearestCamPerSkeleton[lfid].size() == 0)
			{
				refFid = min(refFid + 1, stopF);
				continue;
			}
			if (NearestCamPerSkeleton[lfid].size() == 0)
				refFid++;
			rankId = min(rankId, (int)NearestCamPerSkeleton[lfid].size() - 1);
			if (frameChanged)
				rankId = 0;

			rcid = NearestCamPerSkeleton[lfid][rankId].cid, rfid = NearestCamPerSkeleton[lfid][rankId].fid;
			sprintf(Fname, "%s/%d/%.4d.png", Path2, rcid, rfid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/%d/%.4d.jpg", Path2, rcid, rfid);
				if (IsFileExist(Fname) == 0)
					continue;
			}
			img = imread(Fname);
			CvPoint text_origin = { img.rows / 20, img.cols / 20 };
			sprintf(Fname, "%d/%d", rcid, rfid);	putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 255, 0), 3);

			setTrackbarPos("RefFrame", "DirectorView", refFid);
			setTrackbarPos("Rank", "DirectorView", rankId);
		}

		imshow("DirectorView", img);
		int key = waitKey(1);
		if (key == 27)
			break;
		if (key == 13) //enter
			go = !go;
	}
	return 0;
}

void PoseLinearXYZSlerpInterp(double *c0, double *R0, double *c1, double *R1, double *ct, double *Rt, double t)
{
	double q0[4], q1[4], qt[4];

	t = min(max(t, 0.0), 1.0);

	Rotation2Quaternion(R0, q0);
	Rotation2Quaternion(R1, q1);
	QuaternionLinearInterp(q0, q1, qt, t);
	Quaternion2Rotation(qt, Rt);

	ct[0] = (1.0 - t)*c0[0] + t * c1[0];
	ct[1] = (1.0 - t)*c0[1] + t * c1[1];
	ct[2] = (1.0 - t)*c0[2] + t * c1[2];

	return;
}

struct InterpError {
	InterpError(double *_intrinsic, double *_t, double *_point, double _observed_x, double _observed_y)
	{
		intrinsic = _intrinsic;
		t = _t;
		point = _point;
		observed_x = _observed_x;
		observed_y = _observed_y;
	}

	template <typename T>	bool operator()(const T* const r, T* residuals) 	const
	{
		T org_p[3] = { (T)point[0], (T)point[1], (T)point[2] }, p[3];
		ceres::AngleAxisRotatePoint(r, org_p, p);

		p[0] += (T)t[0], p[1] += (T)t[1], p[2] += (T)t[2];

		T xn = p[0] / p[2], yn = p[1] / p[2];

		residuals[0] = (T)intrinsic[0] * xn + (T)intrinsic[2] * yn + (T)intrinsic[3] - T(observed_x);
		residuals[1] = (T)intrinsic[1] * yn + (T)intrinsic[4] - T(observed_y);

		return true;
	}

	static ceres::CostFunction* Create(double *intrinsic, double *t, double *point, const double observed_x, const double observed_y)
	{
		return (new ceres::AutoDiffCostFunction<InterpError, 2, 3>(new InterpError(intrinsic, t, point, observed_x, observed_y)));
	}

	double observed_x, observed_y, scale;
	double *intrinsic, *t, *point;
};

vector<CameraData> PoseInterpLinearProjectedPath(CameraData &cam1, CameraData &cam2, Point3d target, int nsteps)
{
	Point2d ipt, pt1, pt2;
	ProjectandDistort(target, &pt1, cam1.P);
	ProjectandDistort(target, &pt2, cam2.P);

	vector<CameraData> vInterpCam(nsteps + 1);
	CopyCamereInfo(cam1, vInterpCam[0], true);
	CopyCamereInfo(cam2, vInterpCam[nsteps], true);
	for (int stepi = 1; stepi < nsteps; stepi++)
	{
		double t = 1.0*stepi / nsteps;
		vInterpCam[stepi].camCenter[0] = (1.0 - t)*cam1.camCenter[0] + t * cam2.camCenter[0];
		vInterpCam[stepi].camCenter[1] = (1.0 - t)*cam1.camCenter[1] + t * cam2.camCenter[1];
		vInterpCam[stepi].camCenter[2] = (1.0 - t)*cam1.camCenter[2] + t * cam2.camCenter[2];

		//Look at vector
		double k[3] = { target.x - vInterpCam[stepi].camCenter[0], target.y - vInterpCam[stepi].camCenter[1], target.z - vInterpCam[stepi].camCenter[2] };
		normalize(k, 3);

		//Up vector
		double j[3] = { (1.0 - t)*cam1.R[3] + t * cam2.R[3],  (1.0 - t)*cam1.R[4] + t * cam2.R[4],  (1.0 - t)*cam1.R[5] + t * cam2.R[5] };
		normalize(j, 3);

		//Sideway vector
		double i[3]; cross_product(j, k, i);

		//Camera rotation matrix
		vInterpCam[stepi].R[0] = i[0], vInterpCam[stepi].R[1] = i[1], vInterpCam[stepi].R[2] = i[2];
		vInterpCam[stepi].R[3] = j[0], vInterpCam[stepi].R[4] = j[1], vInterpCam[stepi].R[5] = j[2];
		vInterpCam[stepi].R[6] = k[0], vInterpCam[stepi].R[7] = k[1], vInterpCam[stepi].R[8] = k[2];

		getrFromR(vInterpCam[stepi].R, vInterpCam[stepi].rt);
		GetTfromC(vInterpCam[stepi].R, vInterpCam[stepi].camCenter, vInterpCam[stepi].T);
		GetrtFromRT(vInterpCam[stepi].rt, vInterpCam[stepi].R, vInterpCam[stepi].T);

		for (int i = 0; i < 5; i++)
			vInterpCam[stepi].intrinsic[i] = (1.0 - t)*cam1.intrinsic[i] + t * cam2.intrinsic[i];
		GetKFromIntrinsic(vInterpCam[stepi]);
	}

	for (int stepi = 0; stepi <= nsteps; stepi++)
	{
		double t = 1.0*stepi / nsteps;
		ipt = t * pt1 + (1.0 - t)*pt2;

		ceres::Problem problem;
		ceres::CostFunction* cost_function = InterpError::Create(vInterpCam[stepi].intrinsic, &vInterpCam[stepi].rt[3], &target.x, ipt.x, ipt.y);
		problem.AddResidualBlock(cost_function, NULL, vInterpCam[stepi].rt);

		double residuals[2], *parameters[1] = { vInterpCam[stepi].rt };
		cost_function->Evaluate(parameters, residuals, NULL);

		ceres::Solver::Options options;
		options.max_num_iterations = 300;
		options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = false;

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		cout << summary.BriefReport() << endl;

		cost_function->Evaluate(parameters, residuals, NULL);
	}


	for (int stepi = 0; stepi <= nsteps; stepi++)
	{
		getRfromr(vInterpCam[stepi].rt, vInterpCam[stepi].R);
		AssembleP(vInterpCam[stepi].K, vInterpCam[stepi].R, vInterpCam[stepi].rt + 3, vInterpCam[stepi].P);
	}

	return vInterpCam;
}
void PlanarMorphing(double *Para1, double *Para2, CameraData &cam1, CameraData &cam2, HumanSkeleton3D &Body, vector<Mat> &viImg, int nsteps, int skeletonPointFormat, int interpAlgo)
{
	//double iCamCen[3], iT[3], iR[9], iK[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, iP[12];
	int width = cam1.width, height = cam1.height, length = width * height, nchannels = 3;

	int TorsoJointId[4];
	if (skeletonPointFormat == 18)
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 8, TorsoJointId[3] = 11;
	else
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 9, TorsoJointId[3] = 12;

	Point2d pt;
	vector<Point2d> vptsR1, vptsR2;
	for (int jj = 0; jj < skeletonPointFormat; jj++)
	{
		if (IsValid3D(Body.pt3d[jj]))
		{
			ProjectandDistort(Body.pt3d[jj], &pt, cam1.P), vptsR1.push_back(pt);
			ProjectandDistort(Body.pt3d[jj], &pt, cam2.P), vptsR2.push_back(pt);
		}
	}

	vector<bool> choosenBlobId;
	vector<vector<Point2i> >blobs;
	Mat binary = Mat::zeros(height, width, CV_8U);
	cv::Mat label_image;

	//vector<CameraData> vInterpCam = PoseInterpLinearProjectedPath(cam1, cam2, target, nsteps);
	for (int stepi = 0; stepi <= nsteps; stepi++)
	{
		double t = 1.0*stepi / nsteps;

		//PoseLinearXYZSlerpInterp(cam1.camCenter, cam1.R, cam2.camCenter, cam2.R, vInterpCam[stepi].camCenter, vInterpCam[stepi].R, t);
		//for (int i = 0; i < 5; i++)
		//	vInterpCam[stepi].intrinsic[i] = (1.0 - t)*cam1.intrinsic[i] + t*cam2.intrinsic[i];
		//GetKFromIntrinsic(vInterpCam[stepi]);
		//GetTfromC(vInterpCam[stepi]);
		//AssembleP(vInterpCam[stepi]);

		vector<Point2d> vptsR, vptsNV;
		if (stepi < nsteps / 2)
			vptsR = vptsR1;
		else
			vptsR = vptsR2;
		for (int jj = 0; jj < vptsR1.size(); jj++)
		{
			pt = (1.0 - t)*vptsR1[jj] + t * vptsR2[jj];
			vptsNV.push_back(pt);
		}
		for (int jj = 0; jj < vptsR1.size(); jj++)
			pt += vptsNV[jj];
		pt.x = pt.x / vptsR1.size(), pt.y = pt.y / vptsR1.size();


		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Homo = computeHomography4Point(vptsR, vptsNV);
		double H[9] = { Homo(0), Homo(1), Homo(2),
			Homo(3), Homo(4), Homo(5),
			Homo(6), Homo(7), Homo(8) };

		double reProjecrErr = 0.0;
		for (int ii = 0; ii < vptsNV.size(); ii++)
		{
			// pts1 = H * pts2
			double tx = vptsNV[ii].x, ty = vptsNV[ii].y, dx = vptsR[ii].x, dy = vptsR[ii].y;
			double numx = H[0] * tx + H[1] * ty + H[2] * 1.0,
				numy = H[3] * tx + H[4] * ty + H[5] * 1.0,
				denum = H[6] * tx + H[7] * ty + H[8] * 1.0;
			reProjecrErr += std::sqrt(std::pow(dx - numx / denum, 2) + std::pow(dy - numy / denum, 2));
		}

		t = max(1.0*abs(stepi - nsteps / 2) / nsteps * 2, 0.75); //keep the content, do not blend completely
#pragma omp parallel for
		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				double S, numX = H[0] * ii + H[1] * jj + H[2], numY = H[3] * ii + H[4] * jj + H[5], denum = H[6] * ii + H[7] * jj + H[8];
				Point2d ImgPt(numX / denum, numY / denum);

				if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
					for (int kk = 0; kk < nchannels; kk++)
						viImg[stepi].data[ii*nchannels + jj * width*nchannels + kk] = (unsigned char)0, binary.data[ii + jj * width] = 0;
				else
				{
					binary.data[ii + jj * width] = 255;
					for (int kk = 0; kk < nchannels; kk++)
					{
						if (stepi < nsteps / 2) //1 to novel
							Get_Value_Spline(Para1 + kk * length, width, height, ImgPt.x, ImgPt.y, &S, -1, interpAlgo);
						else//2 to novel
							Get_Value_Spline(Para2 + kk * length, width, height, ImgPt.x, ImgPt.y, &S, -1, interpAlgo);

						//blending
						S = min(max(S, 0.0), 255.0);
						S = t * S + (1.0 - t)*255.0;
						viImg[stepi].data[ii*nchannels + jj * width*nchannels + kk] = (unsigned char)(min(max(S, 0.0), 255.0) + 0.5);
					}
				}
			}
		}

		//remove the warp over image (blobs dont containt the target point)
		blobs.clear();
		choosenBlobId.clear();
		binary.convertTo(label_image, CV_32SC1);

		int label_count = 1;
		for (int y = 0; y < label_image.rows; y++)
		{
			int *row = (int*)label_image.ptr(y);
			for (int x = 0; x < label_image.cols; x++)
			{
				if (row[x] != 255 || row[x] == 0) //255: unlabel, 0: background
					continue;

				cv::Rect rect;
				cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);

				bool found = false;
				std::vector <cv::Point2i> blob;
				for (int j = rect.y; j < (rect.y + rect.height); j++)
				{
					int *row2 = (int*)label_image.ptr(j);
					for (int i = rect.x; i < (rect.x + rect.width); i++)
					{
						if (row2[i] != label_count)
							continue;
						blob.push_back(cv::Point2i(i, j));
						if (abs(i - pt.x) < 2 && abs(j - pt.y) < 2)
							found = true;
					}
				}
				blobs.push_back(blob);
				choosenBlobId.push_back(found);

				label_count++;
			}
		}

		for (int ll = 0; ll < blobs.size(); ll++)
		{
			if (!choosenBlobId[ll])
			{
#pragma omp parallel for
				for (int mm = 0; mm < blobs[ll].size(); mm++)
				{
					int ii = blobs[ll][mm].x, jj = blobs[ll][mm].y;
					for (int kk = 0; kk < nchannels; kk++)
						viImg[stepi].data[ii*nchannels + jj * width*nchannels + kk] = 0;
				}
			}
		}
	}

	return;
}
int PlanarMorphingDriver(char *Path, int selectedPersonId, vector<int> sCams, int startF, int stopF, int increF, int skeletonPointFormat)
{
	char Fname[512]; FILE *fp = 0;

	//Read calib info
	int nCams = 15;// *std::max_element(std::begin(sCams), std::end(sCams)) + 1;
	VideoData *VideoInfo = new VideoData[nCams];

	vector<int> TimeStamp(nCams);
	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); fp = fopen(Fname, "r");
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
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	for (auto camID : sCams)
	{
		printLOG("Cam %d ...validating ", camID);
		if (ReadVideoDataI(Path, VideoInfo[camID], camID, -1, -1) == 1)
			continue;
		else
			InvalidateAbruptCameraPose(VideoInfo[camID], -1, -1, 0);
		printLOG("\n");
	}


	//Read human skeleton
	int nvalidFrames = 0;
	HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int  nValidJoints = 0, temp = (refFid - startF) / increF;
		sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increF, selectedPersonId, refFid);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, selectedPersonId, refFid);
			if (IsFileExist(Fname) == 0)
				continue;
		}
		fp = fopen(Fname, "r");
		int  rcid, rfid, nvis, inlier; double u, v, s, avg_error;
		for (int jid = 0; jid < skeletonPointFormat; jid++)
		{
			fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[temp].pt3d[jid].x, &Skeletons[temp].pt3d[jid].y, &Skeletons[temp].pt3d[jid].z, &avg_error, &nvis);
			for (int kk = 0; kk < nvis; kk++)
			{
				//fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
				fscanf(fp, "%d %lf %lf %lf ", &rcid, &u, &v, &s);
				Skeletons[temp].vViewID_rFid[jid].push_back(Point2i(rcid, -1));
				Skeletons[temp].vPt2D[jid].push_back(Point2d(u, v));
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
	}

	int refFid = 270, cid1 = 3, cid2 = 5, nsteps = 30;
	//int refFid = 500, cid1 = 2, cid2 = 6, nsteps = 30;

	int width = 1920, height = 1080, length = width * height, nchannels = 3, interpAlgo = -1;
	vector<Mat> viImg;
	for (int ii = 0; ii <= nsteps; ii++)
		viImg.push_back(Mat::zeros(height, width, CV_8UC3));

	double *Para1 = new double[length*nchannels], *Para2 = new double[length*nchannels];
	unsigned char *Img = new unsigned char[length*nchannels];
	Mat cvImg1, cvImg2, cvImg;

	double ts = 1.0* refFid / CamTimeInfo[refCid].x; // t = alpha*(f+rs*row) + beta*alpha_ref
	int fid_nr1 = (ts - CamTimeInfo[cid1].y / CamTimeInfo[refCid].x)*CamTimeInfo[cid1].x;
	int fid_nr2 = (ts - CamTimeInfo[cid2].y / CamTimeInfo[refCid].x)*CamTimeInfo[cid2].x;

	sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid1, fid_nr1); cvImg1 = imread(Fname);
	sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid2, fid_nr2); cvImg2 = imread(Fname);
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * width*height] = cvImg1.data[ii*nchannels + jj * width*nchannels + kk];
		Generate_Para_Spline(Img + kk * width*height, Para1 + kk * width*height, width, height, interpAlgo);

		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * width*height] = cvImg2.data[ii*nchannels + jj * width*nchannels + kk];
		Generate_Para_Spline(Img + kk * width*height, Para2 + kk * width*height, width, height, interpAlgo);
	}

	PlanarMorphing(Para1, Para2, VideoInfo[cid1].VideoInfo[fid_nr1], VideoInfo[cid2].VideoInfo[fid_nr2], Skeletons[(refFid - startF) / increF], viImg, nsteps, skeletonPointFormat, interpAlgo);
	for (int stepi = 0; stepi <= nsteps; stepi++)
		sprintf(Fname, "C:/temp/x_%d.jpg", stepi), imwrite(Fname, viImg[stepi]);

	return 0;
}

int CustomizedViewSelectionCost(double maxV, double minV, double optV, double V, double inf = 9e9)
{
	if (V<minV || V>maxV)
		return inf;
	else if (V <= optV)
		return (V - optV) / (minV - optV);
	else
		return (V - optV) / (maxV - optV);
}
int ConstructViewGraphAndCutVideos(char *Path, vector<int> &sCams, int startF, int stopF, int increF, int SkeletonPointFormat, int selectedPersonId, double real2SfM = 1.0, int maxIntraLength = 300)
{
	char Path2[] = { "E:/A1" };
	int width = 1920, height = 1080;
	struct CidFidAngleDist
	{
		CidFidAngleDist(int cid_, int fid_, double angle_, double dist_, double cenX_, double cenY_)
		{
			cid = cid_, fid = fid_;
			angle = angle_, dist = dist_, cenX = cenX_, cenY = cenY_;
		}
		int cid, fid;
		double angle, dist, cenX, cenY;
	};

	char Fname[512];
	int nCams = *std::max_element(std::begin(sCams), std::end(sCams)) + 1;
	vector<int> TimeStamp(nCams);
	vector<Point3d> CamTimeInfo(nCams);
	for (int ii = 0; ii < nCams; ii++)
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
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	VideoData *VideoInfo = new VideoData[nCams];
	for (auto camID : sCams)
	{
		printLOG("Cam %d ...validating ", camID);
		if (ReadVideoDataI(Path, VideoInfo[camID], camID, -1, -1) == 1)
			continue;
		else
			InvalidateAbruptCameraPose(VideoInfo[camID], -1, -1, 0);
		printLOG("\n");
	}

	int nn, refFid, rcid, rfid;
	double angle, dist, cenX, cenY;
	vector<CidFidAngleDist>  *NearestCamPerSkeleton = new vector<CidFidAngleDist>[(stopF - startF) / increF + 1];
	sprintf(Fname, "%s/People/@%d/%d/nearestViews.txt", Path, increF, selectedPersonId);
	if (IsFileExist(Fname) == 0)
	{
		CollectNearestViewsBasedOnSkeleton(Path, sCams, VideoInfo, startF, stopF, increF, selectedPersonId, SkeletonPointFormat);
		//VisualizeNearestViewsBasedOnSkeleton(Path, nCams, startF, stopF, increF, selectedPersonId);
	}
	if (IsFileExist(Fname) == 0)
		return 1;
	else
	{
		fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d ", &refFid, &nn) != EOF)
		{
			if (refFid < startF)
			{
				for (int ii = 0; ii < nn; ii++)
					fscanf(fp, "%d %d %lf %lf %lf %lf", &rcid, &rfid, &angle, &dist, &cenX, &cenY);
				continue;
			}
			if (refFid > stopF)
				break;
			int lfid = (refFid - startF) / increF;
			for (int ii = 0; ii < nn; ii++)
			{
				fscanf(fp, "%d %d %lf %lf %lf %lf", &rcid, &rfid, &angle, &dist, &cenX, &cenY);
				NearestCamPerSkeleton[lfid].push_back(CidFidAngleDist(rcid, rfid, angle, dist, cenX, cenX));
			}
		}
		fclose(fp);
	}

	printLOG("Reading 3D skeleton....");
	HumanSkeleton3D *Skeletons = new HumanSkeleton3D[(stopF - startF) / increF + 1];
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int  nValidJoints = 0, lfid = (refFid - startF) / increF;
		sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, selectedPersonId, refFid);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, selectedPersonId, refFid);
			if (IsFileExist(Fname) == 0)
				continue;
		}
		fp = fopen(Fname, "r");
		int  rcid, rfid, nvis, inlier; double u, v, s, avg_error;
		for (int jid = 0; jid < SkeletonPointFormat; jid++)
		{
			fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[lfid].pt3d[jid].x, &Skeletons[lfid].pt3d[jid].y, &Skeletons[lfid].pt3d[jid].z, &avg_error, &nvis);
			for (int kk = 0; kk < nvis; kk++)
			{
				fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
				Skeletons[lfid].vViewID_rFid[jid].push_back(Point2i(rcid, -1));
				Skeletons[lfid].vPt2D[jid].push_back(Point2d(u, v));
			}

			if (IsValid3D(Skeletons[lfid].pt3d[jid]))
				Skeletons[lfid].validJoints[jid] = 1, nValidJoints++;
			else
				Skeletons[lfid].validJoints[jid] = 0;
		}
		fclose(fp);

		if (nValidJoints < SkeletonPointFormat / 3)
			Skeletons[lfid].valid = 0;
		else
			Skeletons[lfid].valid = 1;
		Skeletons[lfid].refFid = refFid;
	}
	printLOG("done\n");

	double bigValue = 10.0, transitionCost = 0.7, angleWeight = 1.0, centerWeight = 0.25;
	int nframes = (stopF - startF) / increF + 1;
	double *graph = new double[nCams *nframes];
	for (int ii = 0; ii < nCams*nframes; ii++)
		graph[ii] = 2.0 + bigValue;
	for (int ii = 0; ii < nframes; ii++)
	{
		for (int jj = 0; jj < NearestCamPerSkeleton[ii].size(); jj++)
		{
			int cid = NearestCamPerSkeleton[ii][jj].cid, rfid = NearestCamPerSkeleton[ii][jj].fid;
			double angleCost = 1.0 - NearestCamPerSkeleton[ii][jj].angle, distanceCost = NearestCamPerSkeleton[ii][jj].dist;

			double distX = (NearestCamPerSkeleton[ii][jj].cenX - width / 2) / width * 2;
			double distY = (NearestCamPerSkeleton[ii][jj].cenY - height / 2) / height * 2;
			distX = abs(distX) < 0.8 ? 0.0 : distX;
			distY = abs(distY) < 0.8 ? 0.0 : distY;
			double C = sqrt(distX*distX + distY * distY);

			graph[ii*nCams + cid] = angleWeight * angleCost + centerWeight * C;// +CustomizedViewSelectionCost(6000, 2000, 3000, distanceCost*real2SfM, bigValue); //more frontal, center, and closer is better
		}
	}

	fp = fopen("C:/temp/g.txt", "w");
	for (int ii = 0; ii < nframes; ii++)
	{
		for (int jj = 0; jj < nCams; jj++)
			fprintf(fp, "%.2f ", graph[ii*nCams + jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	const int num_nodes = nframes * nCams;
	andres::graph::Digraph<> g(num_nodes);
	std::vector<float> edgeWeights;
	for (int f = 0; f < nframes - 1; f++)
	{
		for (int c = 0; c < nCams; c++)
		{
			for (int c1 = 0; c1 < nCams; c1++)
			{
				int id1 = c + f * nCams, id2 = c1 + (f + 1)*nCams;
				g.insertEdge(id1, id2);
				double weight = (c == c1 ? 0.0 : transitionCost) + graph[id1] * 0.5 + graph[id2] * 0.5;
				edgeWeights.push_back(weight);
			}
		}
	}

	int bestSource, bestTarget;
	float MinPathCostToDate = 9e9;
	vector<Point2i> CurrentPath, ShortestPathToDate;
	for (int c1 = 0; c1 < nCams; c1++)
	{
		for (int c2 = 0; c2 < nCams; c2++)
		{
			//int c1 = 10, c2 = 8;
			double startT = omp_get_wtime();

			float distance = 0;
			std::deque<std::size_t> path;
			andres::graph::spsp(g, c1, c2 + (nframes - 1)*nCams, edgeWeights.begin(), path, distance);

			CurrentPath.clear();
			for (int ii = 0; ii < path.size(); ii++)
				CurrentPath.emplace_back(path[ii] % nCams, path[ii] / nCams);
			if (distance < MinPathCostToDate)
			{
				bestSource = c1, bestTarget = c2;
				MinPathCostToDate = distance;
				ShortestPathToDate = CurrentPath;
			}
			printf("%d-%d: %.4f. Best route: %d-%d  %.4f %.2fs\n", c1, c2, distance, bestSource, bestTarget, MinPathCostToDate, omp_get_wtime() - startT);
		}
	}

	//median filter to reduce jitter
	for (int ii = 0; ii < ShortestPathToDate.size() - 2; ii++)
		if (ShortestPathToDate[ii].x == ShortestPathToDate[ii + 2].x && ShortestPathToDate[ii].x != ShortestPathToDate[ii + 1].x)
			ShortestPathToDate[ii + 1].x = ShortestPathToDate[ii].x, ShortestPathToDate[ii + 1].y = ShortestPathToDate[ii].y + 1;

	fp = fopen("C:/temp/sp2.txt", "w");
	for (auto p : ShortestPathToDate)
		fprintf(fp, "%d %d\n", p.x, p.y);
	fclose(fp);

	Point2i p;
	fp = fopen("C:/temp/sp2.txt", "r");
	while (fscanf(fp, "%d %d\n", &p.x, &p.y) != EOF)
		ShortestPathToDate.push_back(p);
	fclose(fp);


	sprintf(Fname, "%s/People/@%d/%d/SelectedNearestViews.txt", Path, increF, selectedPersonId); fp = fopen(Fname, "w");
	for (auto p : ShortestPathToDate)
	{
		int rcid = p.x, refFid = p.y + startF;
		double ts = 1.0* refFid / CamTimeInfo[refCid].x; // t = alpha*(f+rs*row) + beta*alpha_ref
		int rfid = (ts - CamTimeInfo[rcid].y / CamTimeInfo[refCid].x)*CamTimeInfo[rcid].x;

		fprintf(fp, "%d %d %d %zd ", refFid, rcid, rfid, NearestCamPerSkeleton[p.y].size());
		for (int ii = 0; ii < NearestCamPerSkeleton[p.y].size(); ii++)
		{
			rcid = NearestCamPerSkeleton[p.y][ii].cid, rfid = NearestCamPerSkeleton[p.y][ii].fid;
			fprintf(fp, "%d %d ", rcid, rfid);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	int *PickedCams = new int[nCams *nframes];
	for (int ii = 0; ii < nCams*nframes; ii++)
		PickedCams[ii] = -1;
	for (auto p : ShortestPathToDate)
		PickedCams[p.y*nCams + p.x] = 1;

	refFid = startF;
	int orefFid = refFid - 1, currentCam = 0, ccid = 0, occid = ccid, ttime = 10, cidStart = nCams + 1, reSP = 0;
	namedWindow("DirectorView", WINDOW_NORMAL);
	createTrackbar("RefFrame", "DirectorView", &refFid, stopF, NULL);
	createTrackbar("NearestList", "DirectorView", &ccid, nCams, NULL);
	createTrackbar("TransitionTime", "DirectorView", &ttime, 1000, NULL);
	createTrackbar("RecomputeSP", "DirectorView", &cidStart, nCams + 1, NULL);

	int w = 1920, h = 1080, nchannels = 3, interpAlgo = -1, nsteps = 20;
	double resizeFactor = 0.5;
	vector<Mat> viImg;
	for (int ii = 0; ii <= nsteps; ii++)
		viImg.push_back(Mat::zeros(h, w, CV_8UC3));

	double *Para1 = new double[w*h*nchannels], *Para2 = new double[w*h*nchannels];
	unsigned char *Img = new unsigned char[w*h*nchannels];

	sprintf(Fname, "%s/Vis/SemanticCut", Path); makeDir(Fname);
	sprintf(Fname, "%s/Vis/SemanticCut/%d", Path, selectedPersonId);

	bool writeVideo = true;
	CvSize size;
	size.width = (int)(resizeFactor*w), size.height = (int)(resizeFactor*h);

	Mat img, rimg;
	bool first = true, go = false;
	while (true)
	{
		if (refFid > stopF)
			break;

		bool changed = false, cChanged = false, loaded = false;
		if (go)
			refFid++;
		if (orefFid != refFid)
			changed = true;
		if (occid != ccid)
			cChanged = true;

		if (cidStart != nCams + 1) //re-compute SP
		{
			/*float MinPathCostToDate = 9e9;
			for (int c2 = 0; c2 < nCams; c2++)
			{
				double startT = omp_get_wtime();

				float distance = 0;
				std::deque<std::size_t> path;
				andres::graph::spsp(g, cidStart + (refFid - startF)*nCams, c2 + (nframes - 1)*nCams, edgeWeights.begin(), path, distance);

				CurrentPath.clear();
				for (int ii = 0; ii < path.size(); ii++)
					CurrentPath.emplace_back(path[ii] % nCams, path[ii] / nCams);
				if (distance < MinPathCostToDate)
				{
					bestSource = cidStart, bestTarget = c2;
					MinPathCostToDate = distance;
					for (int lfid = refFid - startF; lfid < nframes; lfid++)
						ShortestPathToDate[lfid] = CurrentPath[lfid - (refFid - startF)];
				}
				printLOG("%d-%d: %.4f. Best route: %d-%d  %.4f %.2fs\n", cidStart, c2, distance, bestSource, bestTarget, MinPathCostToDate, omp_get_wtime() - startT);
			}
			printLOG("\n\n");

			//median filter to reduce jitter
			for (int ii = 0; ii < ShortestPathToDate.size() - 2; ii++)
			{
				if (ShortestPathToDate[ii].x == ShortestPathToDate[ii + 2].x && ShortestPathToDate[ii].x != ShortestPathToDate[ii + 1].x)
					ShortestPathToDate[ii + 1].x = ShortestPathToDate[ii].x, ShortestPathToDate[ii + 1].y = ShortestPathToDate[ii].y + 1;
			}

			for (int ii = 0; ii < nCams*nframes; ii++)
				PickedCams[ii] = -1;
			for (auto p : ShortestPathToDate)
				PickedCams[p.y*nCams + p.x] = 1;

			sprintf(Fname, "C:/temp/sp_%d.txt", reSP);  fp = fopen(Fname, "w");
			for (auto p : ShortestPathToDate)
				fprintf(fp, "%d %d\n", p.x, p.y);
			fclose(fp);*/

			cidStart = nCams + 1;
		}
		if (changed)
		{
			ccid = 0, occid = 0;
			setTrackbarPos("NearestList", "DirectorView", ccid);

			int lfid = (refFid - startF) / increF, rcid = -1;
			for (int cid = 0; cid < nCams; cid++)
				if (PickedCams[cid + lfid * nCams] > -1)
					rcid = cid;

			bool emptyframe = false;
			if (rcid == -1)
				rcid = currentCam, emptyframe = true;
			if (rcid != currentCam && img.empty() == false)
			{
				double ts = 1.0* orefFid / CamTimeInfo[refCid].x; // t = alpha*(f+rs*row) + beta*alpha_ref
				int fid_nr1 = (ts - CamTimeInfo[currentCam].y / CamTimeInfo[refCid].x)*CamTimeInfo[currentCam].x;
				ts = 1.0* refFid / CamTimeInfo[refCid].x;
				int fid_nr2 = (ts - CamTimeInfo[rcid].y / CamTimeInfo[refCid].x)*CamTimeInfo[rcid].x;

				for (int kk = 0; kk < nchannels; kk++)
				{
					for (int jj = 0; jj < h; jj++)
						for (int ii = 0; ii < w; ii++)
							Img[ii + jj * w + kk * w*h] = img.data[ii*nchannels + jj * w*nchannels + kk];
					//Generate_Para_Spline(Img + kk*w*h, Para1 + kk*w*h, w, h, interpAlgo);
				}

				loaded = true;
				sprintf(Fname, "%s/%d/%.4d.png", Path2, rcid, fid_nr2);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/%d/%.4d.jpg", Path2, rcid, fid_nr2);
					if (IsFileExist(Fname) == 0)
						continue;
				}
				img = imread(Fname);
				//sprintf(Fname, "%s/Vis/Feather/%d/%d/%.4d.jpg", Path, selectedPersonId, rcid, fid_nr2); img = imread(Fname);
				if (!img.empty())
				{
					w = img.cols, h = img.rows;
					for (int kk = 0; kk < nchannels; kk++)
					{
						for (int jj = 0; jj < h; jj++)
							for (int ii = 0; ii < w; ii++)
								Img[ii + jj * w + kk * w*h] = img.data[ii*nchannels + jj * w*nchannels + kk];
						//Generate_Para_Spline(Img + kk*w*h, Para2 + kk*w*h, w, h, interpAlgo);
					}

					/*PlanarMorphing(Para1, Para2, VideoInfo[currentCam].VideoInfo[fid_nr1], VideoInfo[rcid].VideoInfo[fid_nr2], Skeletons[lfid], viImg, nsteps, skeletonPointFormat, interpAlgo);
					for (int ii = 0; ii <= nsteps; ii++)
					{
					imshow("DirectorView", viImg[ii]);
					waitKey(ttime);
					if (writeVideo)
					{
					resize(viImg[ii], rimg, Size(resizeFactor * w, resizeFactor * h), 0, 0, INTER_AREA);
					writer << rimg;
					}
					}*/
				}
			}
			currentCam = rcid;

			double ts = 1.0* refFid / CamTimeInfo[refCid].x; // t = alpha*(f+rs*row) + beta*alpha_ref
			int rfid = (ts - CamTimeInfo[rcid].y / CamTimeInfo[refCid].x)*CamTimeInfo[rcid].x;

			if (!loaded)
			{
				sprintf(Fname, "%s/%d/%.4d.png", Path2, rcid, rfid);
				//sprintf(Fname, "%s/Vis/Feather/%d/%d/%.4d.jpg", Path, selectedPersonId, rcid, rfid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/%d/%.4d.jpg", Path2, rcid, rfid);
					if (IsFileExist(Fname) == 0)
						continue;
				}
				img = imread(Fname);
				w = img.cols, h = img.rows;
			}
			CvPoint text_origin = { img.rows / 20, img.cols / 20 };
			sprintf(Fname, "%d/%d", rcid, rfid);	putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, emptyframe ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0), 3);

			int minX = w, maxX = 0, minY = h, maxY = 0;
			for (int jid = 0; jid < SkeletonPointFormat; jid++)
			{
				if (IsValid3D(Skeletons[lfid].pt3d[jid]))
				{
					Point2d pt;
					ProjectandDistort(Skeletons[lfid].pt3d[jid], &pt, VideoInfo[rcid].VideoInfo[rfid].P, VideoInfo[rcid].VideoInfo[rfid].K, VideoInfo[rcid].VideoInfo[rfid].distortion);
					minX = min(minX, pt.x), maxX = max(maxX, pt.x), minY = min(minY, pt.y), maxY = max(maxY, pt.y);
				}
			}
			if (minX != w && minY != h && maxX != 0 && maxY != 0)
				rectangle(img, Point2i(minX, minY), Point2i(maxX, maxY), cv::Scalar(0, 255, 0), 2, 8, 0);

			setTrackbarPos("RefFrame", "DirectorView", refFid);
		}
		if (cChanged)
		{
			int lfid = (refFid - startF) / increF;
			if (ccid < NearestCamPerSkeleton[lfid].size())
			{
				int cid = NearestCamPerSkeleton[lfid][ccid].cid, rfid = NearestCamPerSkeleton[lfid][ccid].fid;
				sprintf(Fname, "%s/%d/%.4d.png", Path2, cid, rfid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/%d/%.4d.jpg", Path2, cid, rfid);
					if (IsFileExist(Fname) == 0)
						continue;
				}
				img = imread(Fname);
				CvPoint text_origin = { img.rows / 20, img.cols / 20 };
				sprintf(Fname, "%d/%d", cid, rfid);	putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 0, 255), 3);

				double angleCost = 1.0 - NearestCamPerSkeleton[lfid][ccid].angle, distanceCost = NearestCamPerSkeleton[lfid][ccid].dist;
				double distX = (NearestCamPerSkeleton[lfid][ccid].cenX - img.cols / 2) / img.cols * 2;
				double distY = (NearestCamPerSkeleton[lfid][ccid].cenY - img.rows / 2) / img.rows * 2;
				double C = sqrt(pow(abs(distX) < 0.8 ? 0.0 : distX, 2) + pow(abs(distY) < 0.8 ? 0.0 : distY, 2));
				double V = angleWeight * angleCost + centerWeight * C;// +CustomizedViewSelectionCost(6000, 2000, 3000, distanceCost*real2SfM, bigValue); //more frontal, center, and closer is better

				text_origin.y = img.rows / 6;
				sprintf(Fname, "A: %.3f, C: (%.3f, %.3f->%.4f), Z: %.3f, V: %.3f ", angleCost, abs(distX) < 0.8 ? 0.0 : distX, abs(distY) < 0.8 ? 0.0 : distY, C, CustomizedViewSelectionCost(6000, 2000, 3000, distanceCost*real2SfM, bigValue), V);
				putText(img, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*img.cols / 640, cv::Scalar(0, 0, 255), 3);

				setTrackbarPos("NearestList", "DirectorView", ccid);
			}
		}

		setTrackbarPos("RecomputeSP", "DirectorView", cidStart);
		imshow("DirectorView", img);
		int key = waitKey(1);
		if (key == 27)
			break;
		if (key == 13) //enter
			go = !go;

		if (changed)
			orefFid = refFid;
		if (cChanged)
			occid = ccid;
	}

	sprintf(Fname, "%s/People/@%d/%d/SelectedNearestViews_2.txt", Path, increF, selectedPersonId);
	fp = fopen(Fname, "w");
	for (auto p : ShortestPathToDate)
	{
		int rcid = p.x, refFid = p.y + startF;
		double ts = 1.0* refFid / CamTimeInfo[refCid].x; // t = alpha*(f+rs*row) + beta*alpha_ref
		int rfid = (ts - CamTimeInfo[rcid].y / CamTimeInfo[refCid].x)*CamTimeInfo[rcid].x;

		fprintf(fp, "%d %d %d %zd ", refFid, rcid, rfid, NearestCamPerSkeleton[p.y].size());
		for (int ii = 0; ii < NearestCamPerSkeleton[p.y].size(); ii++)
		{
			rcid = NearestCamPerSkeleton[p.y][ii].cid, rfid = NearestCamPerSkeleton[p.y][ii].fid;
			fprintf(fp, "%d %d ", rcid, rfid);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	/*//greedy solving of camera path
	int *PickedCams = new int[nCams *nframes];
	for (int ii = 0; ii < nCams*nframes; ii++)
		PickedCams[ii] = -1;
	for (int fid = 0; fid < nframes; fid++)
	{
		int lastBestCam = -1;
		if (fid - 1 >= 0)
		{
			for (int cid = 0; cid < nCams; cid++)
			{
				if (PickedCams[cid*nframes + fid - 1] > -1)
				{
					lastBestCam = cid;
					break;
				}
			}
		}

		double smallestV = 2.0 + bigValue;
		int bestCam = -1;
		for (int cid = 0; cid < nCams; cid++)
		{
			double currentV = graph[cid*nframes + fid];
			if (lastBestCam != -1 && lastBestCam != cid)
				currentV += transitionCost; //avoid switch cam unless it is necessary
			if (currentV < smallestV)
			{
				if (lastBestCam != -1 && PickedCams[lastBestCam*nframes + fid - 1] > maxIntraLength && lastBestCam == cid)
					continue;
				smallestV = currentV, bestCam = cid;
			}
		}

		if (bestCam != -1 && bestCam != lastBestCam && fid + 1 < nframes)
		{
			double nextSmallestV = 2.0 + bigValue;
			int nextBestCam = -1;
			for (int cid = 0; cid < nCams; cid++)
			{
				double currentV = graph[cid*nframes + fid + 1];
				if (bestCam != -1 && bestCam != cid)
					currentV += transitionCost; //avoid switch cam unless it is necessary
				if (currentV < nextSmallestV)
					nextSmallestV = currentV, nextBestCam = cid;
			}
			if (lastBestCam != -1 && bestCam != -1 && nextBestCam == lastBestCam && nextBestCam != bestCam)
				bestCam = lastBestCam;
		}
		PickedCams[bestCam*nframes + fid]++;
	}*/

	return 0;
}
int VisualizeCutVideos(char *Path, vector<int> &sCams, int startF, int stopF, int increF, int SkeletonPointFormat, int selectedPersonId)
{
	char Path2[] = { "E:/A1" };
	char Fname[512];

	int nn, refFid, cid, rcid, rfid;
	vector<vector<Point2i> > VnearestCams(stopF + 1);
	sprintf(Fname, "%s/People/@%d/%d/SelectedNearestViews.txt", Path, increF, selectedPersonId);
	if (!IsFileExist(Fname))
		return 1;
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %d %d %d ", &refFid, &cid, &rfid, &nn) != EOF)
	{
		VnearestCams[refFid].emplace_back(cid, rfid);
		for (int ii = 0; ii < nn; ii++)
			fscanf(fp, "%d %d ", &cid, &rfid);
	}
	fclose(fp);

	int nCams = *std::max_element(std::begin(sCams), std::end(sCams)) + 1;
	VideoData *VideoInfo = new VideoData[nCams];
	for (auto camID : sCams)
	{
		printLOG("Cam %d ...validating ", camID);
		if (ReadVideoDataI(Path, VideoInfo[camID], camID, -1, -1) == 1)
			continue;
		else
			InvalidateAbruptCameraPose(VideoInfo[camID], -1, -1, 0);
		printLOG("\n");
	}

	HumanSkeleton3D *Skeletons = new HumanSkeleton3D[stopF + 1];
	for (int refFid = startF; refFid <= stopF; refFid += increF)
	{
		int  nValidJoints = 0;
		sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increF, selectedPersonId, refFid);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increF, selectedPersonId, refFid);
			if (IsFileExist(Fname) == 0)
				continue;
		}
		fp = fopen(Fname, "r");
		int  rcid, rfid, nvis, inlier; double u, v, s, avg_error;
		for (int jid = 0; jid < SkeletonPointFormat; jid++)
		{
			fscanf(fp, "%lf %lf %lf %lf %d ", &Skeletons[refFid].pt3d[jid].x, &Skeletons[refFid].pt3d[jid].y, &Skeletons[refFid].pt3d[jid].z, &avg_error, &nvis);
			for (int kk = 0; kk < nvis; kk++)
			{
				fscanf(fp, "%d %d %lf %lf %lf %d ", &rcid, &rfid, &u, &v, &s, &inlier);
				Skeletons[refFid].vViewID_rFid[jid].push_back(Point2i(rcid, -1));
				Skeletons[refFid].vPt2D[jid].push_back(Point2d(u, v));
			}

			if (IsValid3D(Skeletons[refFid].pt3d[jid]))
				Skeletons[refFid].validJoints[jid] = 1, nValidJoints++;
			else
				Skeletons[refFid].validJoints[jid] = 0;
		}
		fclose(fp);

		if (nValidJoints < SkeletonPointFormat / 3)
			Skeletons[refFid].valid = 0;
		else
			Skeletons[refFid].valid = 1;
		Skeletons[refFid].refFid = refFid;
	}

	namedWindow("DirectorView", WINDOW_NORMAL);
	createTrackbar("RefFrame", "DirectorView", &refFid, stopF, NULL);

	int w = 1920, h = 1080;
	double resizeFactor = 0.5;

	sprintf(Fname, "%s/Vis/SemanticCut", Path); makeDir(Fname); makeDir(Fname);
	sprintf(Fname, "%s/Vis/SemanticCut/%d", Path, selectedPersonId); makeDir(Fname);

	Mat img, rimg;
	for (int refFid = startF; refFid <= stopF; refFid++)
	{
		cid = VnearestCams[refFid][0].x, rfid = VnearestCams[refFid][0].y;
		sprintf(Fname, "%s/%d/%.4d.png", Path2, cid, rfid);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/%d/%.4d.jpg", Path2, cid, rfid);
			if (IsFileExist(Fname) == 0)
				continue;
		}
		img = imread(Fname);
		resize(img, rimg, Size(resizeFactor * w, resizeFactor * h), 0, 0, INTER_AREA);

		CvPoint text_origin = { rimg.rows / 20, rimg.cols / 20 };
		sprintf(Fname, "%d/%d", cid, rfid);	putText(rimg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*rimg.cols / 640, cv::Scalar(0, 0, 255), 3);
		putText(rimg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*rimg.cols / 640, cv::Scalar(0, 0, 255), 3);

		int minX = w, maxX = 0, minY = h, maxY = 0;
		for (int jid = 0; jid < SkeletonPointFormat; jid++)
		{
			if (IsValid3D(Skeletons[refFid].pt3d[jid]))
			{
				Point2d pt;
				ProjectandDistort(Skeletons[refFid].pt3d[jid], &pt, VideoInfo[cid].VideoInfo[rfid].P, VideoInfo[cid].VideoInfo[rfid].K, VideoInfo[cid].VideoInfo[rfid].distortion);
				minX = min(minX, pt.x), maxX = max(maxX, pt.x), minY = min(minY, pt.y), maxY = max(maxY, pt.y);
			}
		}
		if (minX != w && minY != h && maxX != 0 && maxY != 0)
			rectangle(rimg, Point2i(resizeFactor* minX, resizeFactor*minY), Point2i(resizeFactor*maxX, resizeFactor*maxY), cv::Scalar(0, 255, 0), 2, 8, 0);

		sprintf(Fname, "%s/Vis/SemanticCut/%d/%.4d.jpg", Path, selectedPersonId, refFid);
		imwrite(Fname, rimg);

		imshow("DirectorView", rimg);
		int key = waitKey(1);
		if (key == 27)
			break;
	}
	return 0;
}

int Align2CameraPoseCoord(char *Path1, char *Path2, vector<int> vCid, int startF, int stopF)
{
	VideoData *VideoI1 = new VideoData[vCid.size()], *VideoI2 = new VideoData[vCid.size()];

	for (size_t ii = 0; ii < vCid.size(); ii++)
	{
		ReadVideoDataI(Path1, VideoI1[ii], vCid[ii], startF, stopF);
		ReadVideoDataI(Path2, VideoI2[ii], vCid[ii], startF, stopF);
	}

	double R[9], T[9], scale;
	vector<Point3d> center1, center2;
	for (size_t ii = 0; ii < vCid.size(); ii++)
	{
		for (int fid = startF; fid <= stopF; fid++)
		{
			if (VideoI1[ii].VideoInfo[fid].valid &&VideoI2[ii].VideoInfo[fid].valid)
			{
				center1.emplace_back(VideoI1[ii].VideoInfo[fid].camCenter[0], VideoI1[ii].VideoInfo[fid].camCenter[1], VideoI1[ii].VideoInfo[fid].camCenter[2]);
				center2.emplace_back(VideoI2[ii].VideoInfo[fid].camCenter[0], VideoI2[ii].VideoInfo[fid].camCenter[1], VideoI2[ii].VideoInfo[fid].camCenter[2]);
			}
		}
	}

	double err = computeProcrustesTransform(center1, center2, R, T, scale, true);

	return 0;
}

//stereo IBR
int StereoRectify4IRB_Driver(char *Path, int nCams, int startF, int stopF, int width = 1920, int height = 1080, int debug = 0, int autoplay = 0, double scale = 1.0)
{
	char Fname1[512];
	sprintf(Fname1, "%s/Recitified", Path); makeDir(Fname1);
	sprintf(Fname1, "%s/Recitified/Pairwise", Path); makeDir(Fname1);

	VideoData *AllVideosInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		ReadVideoDataI(Path, AllVideosInfo[cid], cid, startF, stopF);
	Size imageSize(width, height);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	if (debug == 1)
	{
		namedWindow("Rect", CV_WINDOW_NORMAL);
		createTrackbar("AutoPlay", "Rect", &autoplay, 1, NULL);
		createTrackbar("Debug", "Rect", &debug, 1, NULL);
	}

	omp_set_num_threads(omp_get_max_threads());
	//#pragma omp parallel for schedule(dynamic,1)
	for (int Rcid = 0; Rcid < nCams; Rcid++)
	{
		char Fname[512];
		Mat img0, img1, img0r, img1r, map00, map01, map10, map11;
		Mat simg0, simg1, bImg(scale*height, scale*width * 2, CV_8UC3, Scalar(0, 0, 0));

		sprintf(Fname, "%s/Recitified/%d", Path, Rcid); makeDir(Fname);
		sprintf(Fname, "%s/Recitified/Pairwise/%d", Path, Rcid); makeDir(Fname);

		sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, Rcid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		//sprintf(Fname, "%s/Recitified/%d/kNN4IRB.txt", Path, Rcid); FILE *fp1 = fopen(Fname, "w");

		int Rfid, nRcid, nRfid, NN;
		double R10[9], T10[3], nR0[9], nR1[9], nr0[3], nr1[3], nP0[12], nP1[12], Q[16];
		while (fscanf(fp, "%d %d %d ", &Rcid, &Rfid, &NN) != EOF)
		{
#pragma omp critical
			printLOG("(%d,%d)..", Rcid, Rfid);
			vector<double> vQ, vnP1, vnP0, vnr0, vnr1;
			vector<string> vPairNames;
			vector<Point2i> vnRcidfid;
			for (int nn = 0; nn < NN; nn++)
			{
				fscanf(fp, "%d %d ", &nRcid, &nRfid);
				for (int ii = 0; ii < 9; ii++)
					fscanf(fp, "%lf ", &R10[ii]);
				for (int ii = 0; ii < 3; ii++)
					fscanf(fp, "%lf ", &T10[ii]);

				if (AllVideosInfo[Rcid].VideoInfo[Rfid].valid == 0 || AllVideosInfo[nRcid].VideoInfo[nRfid].valid == 0)
					continue;

				double distortion0[5] = { AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[0], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[1], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[3], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[4], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[2] };
				double distortion1[5] = { AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[0], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[1], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[3], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[4], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[2] };

				Mat cvK0(3, 3, CV_64F, AllVideosInfo[Rcid].VideoInfo[Rfid].K),
					cvK1(3, 3, CV_64F, AllVideosInfo[nRcid].VideoInfo[nRfid].K);
				Mat cvDistortion0(1, 5, CV_64F, distortion0), cvDistortion1(1, 5, CV_64F, distortion1);
				Mat cvR10(3, 3, CV_64F, R10), cvT10(3, 1, CV_64F, T10);
				Mat cvnR0(3, 3, CV_64F, nR0), cvnR1(3, 3, CV_64F, nR1), cvnP0(3, 4, CV_64F, nP0), cvnP1(3, 4, CV_64F, nP1), cvQ(4, 4, CV_64F, Q);

				stereoRectify(cvK0, cvDistortion0, cvK1, cvDistortion1, imageSize, cvR10, cvT10, cvnR0, cvnR1, cvnP0, cvnP1, cvQ, CALIB_ZERO_DISPARITY, -1, imageSize);

				/*initUndistortRectifyMap(cvK0, cvDistortion0, cvnR0, cvnP0, imageSize, CV_16SC2, map00, map01);
				initUndistortRectifyMap(cvK1, cvDistortion1, cvnR1, cvnP1, imageSize, CV_16SC2, map10, map11);

				sprintf(Fname, "%s/%d/%.4d.png", Path, Rcid, Rfid);
				if (IsFileExist(Fname) == 0)
				{
				sprintf("%s/%d/%.4d.jpg", Path, Rcid, Rfid);
				if (IsFileExist(Fname) == 0)
				continue;
				}
				img0 = imread(Fname);

				sprintf(Fname, "%s/%d/%.4d.png", Path, nRcid, nRfid);
				if (IsFileExist(Fname) == 0)
				{
				sprintf("%s/%d/%.4d.jpg", Path, nRcid, nRfid);
				if (IsFileExist(Fname) == 0)
				continue;
				}
				img1 = imread(Fname);

				remap(img0, img0r, map00, map01, INTER_CUBIC);
				remap(img1, img1r, map10, map11, INTER_CUBIC);*/

				if (debug == 1)
				{
					Rect rect0(0, 0, scale*width, scale*height);
					Rect rect1(scale*width, 0, scale*width, scale*height);
					resize(img0r, simg0, Size(scale*width, scale*height), 0, 0, INTER_AREA);
					resize(img1r, simg1, Size(scale*width, scale*height), 0, 0, INTER_AREA);
					simg0.copyTo(bImg(rect0));
					simg1.copyTo(bImg(rect1));

					for (int j = 0; j < bImg.rows; j += 16)
					{
						if (j % 3 == 0)
							line(bImg, Point(0, j), Point(bImg.cols, j), Scalar(255, 0, 0), 1, 8);
						else if (j % 3 == 1)
							line(bImg, Point(0, j), Point(bImg.cols, j), Scalar(0, 255, 0), 1, 8);
						else
							line(bImg, Point(0, j), Point(bImg.cols, j), Scalar(0, 0, 255), 1, 8);
					}

					if (autoplay == 1)
					{
						imshow("Rect", bImg);
						waitKey(100);
					}
					else
					{
						int key = 0;
						while (true)
						{
							imshow("Rect", bImg);
							if (waitKey(1) == 27)
								break;
						}
					}
					//setTrackbarPos("AutoPlay", "Rect", debug);
					//setTrackbarPos("Debug", "Rect", autoplay);
				}

				sprintf(Fname, "%s/Recitified/%d/%.4d_%d_%.4d_1.png", Path, Rcid, Rfid, nRcid, nRfid);	string str0(Fname);
				sprintf(Fname, "%s/Recitified/%d/%.4d_%d_%.4d_2.png", Path, Rcid, Rfid, nRcid, nRfid);	string str1(Fname);
				//imwrite(str0, img0r), imwrite(str1, img1r);

				getrFromR(nR0, nr0), getrFromR(nR1, nr1);
				vnRcidfid.push_back(Point2i(nRcid, nRfid));
				for (int ii = 0; ii < 3; ii++)
					vnr0.push_back(nr0[ii]), vnr1.push_back(nr1[ii]);
				for (int ii = 0; ii < 12; ii++)
					vnP0.push_back(nP0[ii]), vnP1.push_back(nP1[ii]);
				for (int ii = 0; ii < 16; ii++)
					vQ.push_back(Q[ii]);
				vPairNames.push_back(str0), vPairNames.push_back(str1);
			}

			/*fprintf(fp1, "%d %d %d\n", Rcid, Rfid, vnRcidfid.size());
			for (int ii = 0; ii < vnRcidfid.size(); ii++)
			{
			fprintf(fp1, "%d %d\n", vnRcidfid[ii].x, vnRcidfid[ii].y);

			fprintf(fp1, "%.16f %.16f %.16f\n", vnr0[3 * ii], vnr0[3 * ii + 1], vnr0[3 * ii + 2]);
			fprintf(fp1, "%f 0.0 %f %f 0.0 %f %f 0.0 0.0 0.0 0.0 1.0\n", vnP0[12 * ii], vnP0[12 * ii + 2], vnP0[12 * ii + 3], vnP0[12 * ii + 5], vnP0[12 * ii + 6]);

			fprintf(fp1, "%.16f %.16f %.16f\n", vnr1[3 * ii], vnr1[3 * ii + 1], vnr1[3 * ii + 2]);
			fprintf(fp1, "%f 0.0 %f %f 0.0 %f %f 0.0 0.0 0.0 0.0 1.0\n", vnP1[12 * ii], vnP1[12 * ii + 2], vnP1[12 * ii + 3], vnP1[12 * ii + 5], vnP1[12 * ii + 6]);

			fprintf(fp1, "1.0 0.0 0.0 %f 0.0 1.0 0.0 %f 0.0 0.0 0.0 %f 0 0 %f %f\n", vQ[ii * 16 + 3], vQ[ii * 16 + 7], vQ[ii * 16 + 11], vQ[ii * 16 + 14], vQ[ii * 16 + 15]);
			}*/

			for (int ii = 0; ii < vnRcidfid.size(); ii++)
			{
				sprintf(Fname, "%s/Recitified/Pairwise/%d/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rcid, Rfid, vnRcidfid[ii].x, vnRcidfid[ii].y);
				FILE *fp2 = fopen(Fname, "w");

				fprintf(fp2, "%.16f %.16f %.16f\n", vnr0[3 * ii], vnr0[3 * ii + 1], vnr0[3 * ii + 2]);
				fprintf(fp2, "%f 0.0 %f %f 0.0 %f %f 0.0 0.0 0.0 1.0 0.0\n", vnP0[12 * ii], vnP0[12 * ii + 2], vnP0[12 * ii + 3], vnP0[12 * ii + 5], vnP0[12 * ii + 6]);

				fprintf(fp2, "%.16f %.16f %.16f\n", vnr1[3 * ii], vnr1[3 * ii + 1], vnr1[3 * ii + 2]);
				fprintf(fp2, "%f 0.0 %f %f 0.0 %f %f 0.0 0.0 0.0 1.0 0.0\n", vnP1[12 * ii], vnP1[12 * ii + 2], vnP1[12 * ii + 3], vnP1[12 * ii + 5], vnP1[12 * ii + 6]);

				fprintf(fp2, "1.0 0.0 0.0 %f 0.0 1.0 0.0 %f 0.0 0.0 0.0 %f 0 0 %f %f\n", vQ[ii * 16 + 3], vQ[ii * 16 + 7], vQ[ii * 16 + 11], vQ[ii * 16 + 14], vQ[ii * 16 + 15]);

				fclose(fp2);
			}
		}
		//fclose(fp1);
		printLOG("\n*********\n");
	}

	return 0;
}
int StereoRectify4IRB_Driver2(char *Path, int nCams, int startF, int stopF, int width = 1920, int height = 1080, int debug = 0, int autoplay = 0, double scale = 1.0)
{
	char Fname[512], Fname1[512];
	sprintf(Fname1, "%s/Recitified2", Path); makeDir(Fname1);
	sprintf(Fname1, "%s/Recitified2/Pairwise", Path); makeDir(Fname1);

	VideoData *AllVideosInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		ReadVideoDataI(Path, AllVideosInfo[cid], cid, startF, stopF);
	Size imageSize(width, height);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	if (debug == 1)
	{
		namedWindow("Org", CV_WINDOW_NORMAL);
		namedWindow("Rect", CV_WINDOW_NORMAL);
		createTrackbar("AutoPlay", "Rect", &autoplay, 1, NULL);
		createTrackbar("Debug", "Rect", &debug, 1, NULL);
	}

	int Rcid, Rfid, nRcid, nRfid;
	double R10[9], T10[3], nR0[9], nR1[9], nr0[3], nr1[3], nP0[12], nP1[12], Q[16];
	Mat img0, img1, img0r, img1r, map00, map01, map10, map11;
	Mat simg0, simg1, bImg(scale*height, scale*width * 2, CV_8UC3, Scalar(0, 0, 0)), bImg_org(height, width * 2, CV_8UC3, Scalar(0, 0, 0));

	std::string line;
	std::string item;
	std::ifstream file("D:/spatial_data_frame_list.txt");
	if (file.fail())
		return 1;

	vector<CameraData> masterCameras;
	while (std::getline(file, line))
	{
		StringTrim(&line);

		std::size_t pos = line.find("_");
		string sub_str = line.substr(0, pos);
		Rcid = atoi(sub_str.c_str());
		line = line.substr(pos + 1);

		pos = line.find("_");
		sub_str = line.substr(0, pos);
		Rfid = atoi(sub_str.c_str());
		line = line.substr(pos + 1);

		pos = line.find("_");
		sub_str = line.substr(0, pos);
		nRcid = atoi(sub_str.c_str());
		line = line.substr(pos + 1);

		pos = line.find("_");
		sub_str = line.substr(0, pos);
		nRfid = atoi(sub_str.c_str());

		if (AllVideosInfo[Rcid].VideoInfo[Rfid].valid == 0 || AllVideosInfo[nRcid].VideoInfo[nRfid].valid == 0)
			continue;

		double distortion0[5] = { AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[0], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[1], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[3], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[4], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[2] };
		double distortion1[5] = { AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[0], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[1], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[3], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[4], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[2] };

		GetRelativeTransformation(AllVideosInfo[Rcid].VideoInfo[Rfid].R, AllVideosInfo[Rcid].VideoInfo[Rfid].T, AllVideosInfo[nRcid].VideoInfo[nRfid].R, AllVideosInfo[nRcid].VideoInfo[nRfid].T, R10, T10);

		Mat cvK0(3, 3, CV_64F, AllVideosInfo[Rcid].VideoInfo[Rfid].K),
			cvK1(3, 3, CV_64F, AllVideosInfo[nRcid].VideoInfo[nRfid].K);
		Mat cvDistortion0(1, 5, CV_64F, distortion0), cvDistortion1(1, 5, CV_64F, distortion1);
		Mat cvR10(3, 3, CV_64F, R10), cvT10(3, 1, CV_64F, T10);
		Mat cvnR0(3, 3, CV_64F, nR0), cvnR1(3, 3, CV_64F, nR1), cvnP0(3, 4, CV_64F, nP0), cvnP1(3, 4, CV_64F, nP1), cvQ(4, 4, CV_64F, Q);

		stereoRectify(cvK0, cvDistortion0, cvK1, cvDistortion1, imageSize, cvR10, cvT10, cvnR0, cvnR1, cvnP0, cvnP1, cvQ, CALIB_ZERO_DISPARITY, -1, imageSize);

		if (debug == 1)
		{
			initUndistortRectifyMap(cvK0, cvDistortion0, cvnR0, cvnP0, imageSize, CV_16SC2, map00, map01);
			initUndistortRectifyMap(cvK1, cvDistortion1, cvnR1, cvnP1, imageSize, CV_16SC2, map10, map11);

			sprintf(Fname, "%s/%d/%.4d.png", Path, Rcid, Rfid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf("%s/%d/%.4d.jpg", Path, Rcid, Rfid);
				if (IsFileExist(Fname) == 0)
					continue;
			}
			img0 = imread(Fname);

			sprintf(Fname, "%s/%d/%.4d.png", Path, nRcid, nRfid);
			if (IsFileExist(Fname) == 0)
			{
				sprintf("%s/%d/%.4d.jpg", Path, nRcid, nRfid);
				if (IsFileExist(Fname) == 0)
					continue;
			}
			img1 = imread(Fname);

			{
				Rect rect0(0, 0, width, height);
				Rect rect1(width, 0, width, height);
				img0.copyTo(bImg_org(rect0));
				img1.copyTo(bImg_org(rect1));
				imshow("Org", bImg_org);
				waitKey(0);
			}

			remap(img0, img0r, map00, map01, INTER_CUBIC);
			remap(img1, img1r, map10, map11, INTER_CUBIC);

			Rect rect0(0, 0, scale*width, scale*height);
			Rect rect1(scale*width, 0, scale*width, scale*height);
			resize(img0r, simg0, Size(scale*width, scale*height), 0, 0, INTER_AREA);
			resize(img1r, simg1, Size(scale*width, scale*height), 0, 0, INTER_AREA);
			simg0.copyTo(bImg(rect0));
			simg1.copyTo(bImg(rect1));

			for (int j = 0; j < bImg.rows; j += 16)
			{
				if (j % 3 == 0)
					cv::line(bImg, Point(0, j), Point(bImg.cols, j), Scalar(255, 0, 0), 1, 8);
				else if (j % 3 == 1)
					cv::line(bImg, Point(0, j), Point(bImg.cols, j), Scalar(0, 255, 0), 1, 8);
				else
					cv::line(bImg, Point(0, j), Point(bImg.cols, j), Scalar(0, 0, 255), 1, 8);
			}

			if (autoplay == 1)
			{
				imshow("Rect", bImg);
				waitKey(100);
			}
			else
			{
				int key = 0;
				while (true)
				{
					imshow("Rect", bImg);
					if (waitKey(1) == 27)
						break;
				}
			}
			//setTrackbarPos("AutoPlay", "Rect", debug);
			//setTrackbarPos("Debug", "Rect", autoplay);
		}

		sprintf(Fname, "%s/Recitified2/%d_%.4d_%d_%.4d_1.png", Path, Rcid, Rfid, nRcid, nRfid);	string str0(Fname);
		sprintf(Fname, "%s/Recitified2/%d_%.4d_%d_%.4d_2.png", Path, Rcid, Rfid, nRcid, nRfid);	string str1(Fname);
		//imwrite(str0, img0r), imwrite(str1, img1r);

		getrFromR(nR0, nr0), getrFromR(nR1, nr1);

		sprintf(Fname, "%s/Recitified2/Pairwise/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rfid, nRcid, nRfid); FILE *fp2 = fopen(Fname, "w");

		fprintf(fp2, "%.16f %.16f %.16f\n", nr0[0], nr0[1], nr0[2]);
		fprintf(fp2, "%f 0.0 %f %f 0.0 %f %f 0.0 0.0 0.0 1.0 0.0\n", nP0[0], nP0[2], nP0[3], nP0[5], nP0[6]);

		fprintf(fp2, "%.16f %.16f %.16f\n", nr1[0], nr1[1], nr1[2]);
		fprintf(fp2, "%f 0.0 %f %f 0.0 %f %f 0.0 0.0 0.0 1.0 0.0\n", nP1[0], nP1[2], nP1[3], nP1[5], nP1[6]);

		fprintf(fp2, "1.0 0.0 0.0 %f 0.0 1.0 0.0 %f 0.0 0.0 0.0 %f 0 0 %f %f\n", Q[3], Q[7], Q[11], Q[14], Q[15]);
		fclose(fp2);

	}
	file.close();
	return 0;
}
int GenSparseCorres4IRB_Driver(char *Path, int nCams, int startF, int stopF)
{
	vector<Scalar> colors;
	colors.push_back(Scalar(0, 0, 255)), colors.push_back(Scalar(0, 128, 255)), colors.push_back(Scalar(0, 255, 255)), colors.push_back(Scalar(0, 255, 0)),
		colors.push_back(Scalar(255, 128, 0)), colors.push_back(Scalar(255, 255, 0)), colors.push_back(Scalar(255, 0, 0)), colors.push_back(Scalar(255, 0, 255)), colors.push_back(Scalar(255, 255, 255));

	char Fname[512];
	bool cross_check = true;
	int  ninlierThresh = 30, nthreads = 7;
	double max_ratio = 0.7, max_distance = 0.7;

	VideoData *AllVideosInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		ReadVideoDataI(Path, AllVideosInfo[cid], cid, startF, stopF);

	//1. Get images ID and extract sift
	double dummy;
	vector<Point2i> vCidFid;
	/*for (int Rcid = 0; Rcid < nCams; Rcid++)
	{
	sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, Rcid);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	continue;

	int Rfid, nRcid, nRfid, NN;
	double R10[9], T10[3], nP0[12], nP1[12], Q[16];
	while (fscanf(fp, "%d %d %d ", &Rcid, &Rfid, &NN) != EOF)
	{
	vCidFid.push_back(Point2i(Rcid, Rfid));
	for (int nn = 0; nn < NN; nn++)
	{
	fscanf(fp, "%d %d ", &nRcid, &nRfid);
	for (int ii = 0; ii < 9; ii++)
	fscanf(fp, "%lf ", &dummy);
	for (int ii = 0; ii < 3; ii++)
	fscanf(fp, "%lf ", &dummy);

	if (AllVideosInfo[Rcid].VideoInfo[Rfid].valid == 0 || AllVideosInfo[nRcid].VideoInfo[nRfid].valid == 0)
	continue;

	vCidFid.push_back(Point2i(Rcid, Rfid));
	}
	}
	fclose(fp);
	}

	//Get sift
	ExtractSiftFromImageListDriver(Path, vCidFid);*/

	//2. Gen corres
	char Path2[] = "D:/Dance";
	sprintf(Fname, "%s/Dynamic", Path2), makeDir(Fname);
	sprintf(Fname, "%s/Dynamic/rawMatches", Path2), makeDir(Fname);
	sprintf(Fname, "%s/Dynamic/geoMatches", Path2), makeDir(Fname);
	sprintf(Fname, "%s/Dynamic/pwMatches", Path2), makeDir(Fname);

	Point3d P3d;
	vector<uchar> *AllSiftDesc = new vector<uchar>[nCams];
	vector<KeyPoint> *AllSiftPoints = new vector<KeyPoint>[nCams];
	vector<int> vInliers[1];
	bool *PassedTri = new bool[nCams * 2];
	Point2d *vPts = new Point2d[2 * nCams];
	double *A = new double[6 * nCams * 2], *B = new double[2 * nCams * 2], *P = new double[12 * nCams * 2], *tP = new double[12 * nCams * 2];


	for (int Rcid = 0; Rcid < nCams; Rcid++)
	{
		printLOG("Working on %d: ", Rcid);
		sprintf(Fname, "%s/Dynamic/geoMatches/%d", Path, Rcid), makeDir(Fname);

		sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, Rcid);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		int Rfid, nRcid, nRfid, NN;
		while (fscanf(fp, "%d %d %d ", &Rcid, &Rfid, &NN) != EOF)
		{
			printLOG("%d: ", Rfid);
			if (Rfid > stopF)
				break;

			vCidFid.clear();
			vCidFid.push_back(Point2i(Rcid, Rfid));
			for (int nn = 0; nn < NN; nn++)
			{
				fscanf(fp, "%d %d ", &nRcid, &nRfid);
				for (int ii = 0; ii < 9; ii++)
					fscanf(fp, "%lf ", &dummy);
				for (int ii = 0; ii < 3; ii++)
					fscanf(fp, "%lf ", &dummy);

				if (AllVideosInfo[Rcid].VideoInfo[Rfid].valid == 0 || AllVideosInfo[nRcid].VideoInfo[nRfid].valid == 0)
					continue;

				vCidFid.push_back(Point2i(nRcid, nRfid));
			}

			/*for (int ii = 0; ii < nCams; ii++)
			AllSiftPoints[ii].clear();
			for (size_t ii = 0; ii < vCidFid.size(); ii++)
			{
			int cid = vCidFid[ii].x, fid = vCidFid[ii].y;
			sprintf(Fname, "%s/%d/%.4d.sift", Path, cid, fid);
			AllSiftPoints[cid].clear(), AllSiftDesc[cid].clear();
			readVisualSFMSiftGPU(Fname, AllSiftPoints[cid], AllSiftDesc[cid]);
			}*/

			//Do raw Sift matching
			/*vector<Point2i> vpair;
			for (int ii = 0; ii < vCidFid.size() - 1; ii++)
			for (int jj = ii + 1; jj < vCidFid.size(); jj++)
			vpair.push_back(Point2i(ii, jj));
			omp_set_num_threads(nthreads);
			#pragma omp parallel for schedule(dynamic,1)
			for (int kk = 0; kk < (int)vpair.size(); kk++)
			{
			int ii = vpair[kk].x, jj = vpair[kk].y;
			int cid1 = vCidFid[ii].x, fid1 = vCidFid[ii].y, cid2 = vCidFid[jj].x, fid2 = vCidFid[jj].y;

			char Fname[512];
			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid1, fid1, cid2, fid2);
			if (IsFileExist(Fname) == 1)
			continue;
			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid2, fid2, cid1, fid1);
			if (IsFileExist(Fname) == 1)
			continue;

			if (AllSiftPoints[cid1].size() == 0 || AllSiftPoints[cid2].size() == 0)
			{
			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid1, fid1, cid2, fid2);
			FILE *fp1 = fopen(Fname, "w+"); fclose(fp1);
			continue;
			}
			vector<Point2i> matches;
			matches = MatchTwoViewSIFTBruteForce(AllSiftDesc[cid1], AllSiftDesc[cid2], 128, max_ratio, max_distance, cross_check, 1);
			if (matches.size() < ninlierThresh)
			{
			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid1, fid1, cid2, fid2);
			FILE *fp1 = fopen(Fname, "w+"); fclose(fp1);
			continue;
			}

			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid1, fid1, cid2, fid2);
			FILE *fp1 = fopen(Fname, "w+");
			for (int ii = 0; ii < matches.size(); ii++)
			fprintf(fp1, "%d %d\n", matches[ii].x, matches[ii].y);
			fclose(fp1);

			if (0)
			{
			sprintf(Fname, "%s/%d/%.4d.png", Path, cid1, fid1);
			Mat img1 = imread(Fname);

			sprintf(Fname, "%s/%d/%.4d.png", Path, cid2, fid2);
			Mat img2 = imread(Fname);

			for (int ii = 0; ii < matches.size(); ii++)
			{
			int i1 = matches[ii].x, i2 = matches[ii].y;
			circle(img1, AllSiftPoints[cid1][i1].pt, 2, colors[ii % 8], 3);
			circle(img2, AllSiftPoints[cid2][i2].pt, 2, colors[ii % 8], 3);
			}

			int width = img1.cols, height = img1.rows;
			Mat bImg(height, width * 2, CV_8UC3, Scalar(0, 0, 0));

			Rect rect1(0, 0, width, height);
			Rect rect2(width, 0, width, height);
			img1.copyTo(bImg(rect1));
			img2.copyTo(bImg(rect2));

			namedWindow("X", CV_WINDOW_NORMAL);
			imshow("X", bImg); waitKey(0);
			destroyAllWindows();
			}
			}*/

			//Gen allviews matching table
			/*int totalPts = 0;
			vector<int> *KeysBelongTo3DPoint = new vector <int>[vCidFid.size()];
			for (size_t jj = 0; jj < vCidFid.size(); jj++)
			{
			totalPts += (int)AllSiftPoints[vCidFid[jj].x].size();
			KeysBelongTo3DPoint[jj].reserve(AllSiftPoints[vCidFid[jj].x].size());
			for (size_t ii = 0; ii < AllSiftPoints[vCidFid[jj].x].size(); ii++)
			KeysBelongTo3DPoint[jj].push_back(-1);
			}

			int count3D = 0;
			vector<int>*ViewMatch = new vector<int>[totalPts];
			vector<int>*PointIDMatch = new vector<int>[totalPts];
			for (int ii = 0; ii < vCidFid.size() - 1; ii++)
			{
			for (int jj = ii + 1; jj < vCidFid.size(); jj++)
			{
			int cid1 = vCidFid[ii].x, fid1 = vCidFid[ii].y, cid2 = vCidFid[jj].x, fid2 = vCidFid[jj].y;
			vector<Point2i> matches; int id1, id2;

			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid1, fid1, cid2, fid2);
			if (IsFileExist(Fname) == 1)
			{
			FILE *fp1 = fopen(Fname, "r");
			while (fscanf(fp1, "%d %d ", &id1, &id2) != EOF)
			matches.push_back(Point2i(id1, id2));
			fclose(fp1);
			}
			else
			{
			sprintf(Fname, "%s/Dynamic/rawMatches/%d_%.4d_%d_%.4d.txt", Path, cid2, fid2, cid1, fid1);
			FILE *fp1 = fopen(Fname, "r");
			while (fscanf(fp1, "%d %d ", &id2, &id1) != EOF)
			matches.push_back(Point2i(id1, id2));
			fclose(fp1);
			}

			for (int kk = 0; kk < matches.size(); kk++)
			{
			int id1 = matches[kk].x, id2 = matches[kk].y;
			int ID3D1 = KeysBelongTo3DPoint[ii][id1], ID3D2 = KeysBelongTo3DPoint[jj][id2];
			if (ID3D1 == -1 && ID3D2 == -1) //Both are never seeen before
			{
			ViewMatch[count3D].push_back(ii), ViewMatch[count3D].push_back(jj);
			PointIDMatch[count3D].push_back(id1), PointIDMatch[count3D].push_back(id2);
			KeysBelongTo3DPoint[ii][id1] = count3D, KeysBelongTo3DPoint[jj][id2] = count3D; //this pair of corres constitutes 3D point #count
			count3D++;
			}
			else if (ID3D1 == -1 && ID3D2 != -1)
			{
			ViewMatch[ID3D2].push_back(ii);
			PointIDMatch[ID3D2].push_back(id1);
			KeysBelongTo3DPoint[ii][id1] = ID3D2; //this point constitutes 3D point #ID3D2
			}
			else if (ID3D1 != -1 && ID3D2 == -1)
			{
			ViewMatch[ID3D1].push_back(jj);
			PointIDMatch[ID3D1].push_back(id2);
			KeysBelongTo3DPoint[jj][id2] = ID3D1; //this point constitutes 3D point #ID3D2
			}
			else if (ID3D1 != -1 && ID3D2 != -1 && ID3D1 != ID3D2)//Strange case where 1 point (usually not vey discrimitive or repeating points) is matched to multiple points in the same view pair --> Just concatanate the one with fewer points to largrer one and hope MultiTriangulationRansac can do sth.
			{
			if (ViewMatch[ID3D1].size() >= ViewMatch[ID3D2].size())
			{
			int nmatches = (int)ViewMatch[ID3D2].size();
			for (int ll = 0; ll < nmatches; ll++)
			{
			ViewMatch[ID3D1].push_back(ViewMatch[ID3D2].at(ll));
			PointIDMatch[ID3D1].push_back(PointIDMatch[ID3D2].at(ll));
			}
			ViewMatch[ID3D2].clear(), PointIDMatch[ID3D2].clear();
			}
			else
			{
			int nmatches = (int)ViewMatch[ID3D1].size();
			for (int ll = 0; ll < nmatches; ll++)
			{
			ViewMatch[ID3D2].push_back(ViewMatch[ID3D1].at(ll));
			PointIDMatch[ID3D2].push_back(PointIDMatch[ID3D1].at(ll));
			}
			ViewMatch[ID3D1].clear(), PointIDMatch[ID3D1].clear();
			}
			}
			else//(ID3D1 == ID3D2): cycle in the corres, i.e. a-b, a-c, and b-c
			continue;
			}
			}
			}
			delete[]KeysBelongTo3DPoint, delete[]ViewMatch, delete[]PointIDMatch;*/

			//RANSAC pruning
			int count = 0, npts = 0;
			Point2f uv;
			int nmatches, lcid, lfid, cnt = 0;
			pair<Point2f, Point2f> matches;
			vector<int> vlcid, vlfid;
			vector<Point2f> vmatches;
			vector<pair<Point2f, Point2f> > *allPairs = new vector<pair<Point2f, Point2f> >[vCidFid.size()*vCidFid.size()];
			/*sprintf(Fname, "%s/Dynamic/geoMatches/%d/View_ID_PM_%.4d.txt", Path, Rcid, Rfid); FILE *fp1 = fopen(Fname, "w+");
			for (int jj = 0; jj < count3D; jj++)
			{
			int nmatches = (int)ViewMatch[jj].size();
			if (nmatches < 2 || nmatches > vCidFid.size() * 2)
			continue;

			for (int ii = 0; ii < nmatches; ii++)
			{
			int cid = vCidFid[ViewMatch[jj][ii]].x, fid = vCidFid[ViewMatch[jj][ii]].y, pid = PointIDMatch[jj][ii];
			vPts[ii] = Point2d(AllSiftPoints[cid][pid].pt.x, AllSiftPoints[cid][pid].pt.y);

			PassedTri[ii] = true;
			for (int kk = 0; kk < 12; kk++)
			P[12 * ii + kk] = AllVideosInfo[cid].VideoInfo[fid].P[kk];
			}
			vInliers[0].clear();
			double reProjError = NviewTriangulationRANSAC(vPts, P, &P3d, PassedTri, vInliers, nmatches, 1, min(nmatches, 3), 100, 0.3, 3.0, A, B, tP);

			if (PassedTri[0])
			{
			npts++;
			count = 0;
			for (int ii = 0; ii < nmatches; ii++)
			if (vInliers[0][ii] == 1)
			count++;

			fprintf(fp1, "%d ", count);
			for (int ii = 0; ii < nmatches; ii++)
			if (vInliers[0][ii] == 1)
			fprintf(fp1, "%d %d %.2f %.2f ", ViewMatch[jj][ii], PointIDMatch[jj][ii], vPts[ii].x, vPts[ii].y);
			fprintf(fp1, "\n");

			//gen pairwise matches
			vlcid.clear(), vlfid.clear(), vmatches.clear();
			for (int ii = 0; ii < nmatches; ii++)
			if (vInliers[0][ii] == 1)
			vlcid.push_back(lcid), vlfid.push_back(lfid), vmatches.push_back(uv);
			for (int ii = 0; ii < vlcid.size(); ii++)
			{
			for (int jj = 0; jj < vlcid.size(); jj++)
			{
			int p1 = vlcid[ii], p2 = vlcid[jj];
			matches.first = vmatches[ii], matches.second = vmatches[jj];
			allPairs[p1 + p2*vCidFid.size()].push_back(matches);
			}
			}
			}
			}
			fclose(fp1);*/

			sprintf(Fname, "%s/Dynamic/geoMatches/%d/View_ID_PM_%.4d.txt", Path, Rcid, Rfid); FILE *fp1 = fopen(Fname, "r");
			while (fscanf(fp1, "%d ", &nmatches) != EOF)
			{
				vlcid.clear(), vlfid.clear(), vmatches.clear();
				for (int ii = 0; ii < nmatches; ii++)
				{
					fscanf(fp1, "%d %d %f %f ", &lcid, &lfid, &uv.x, &uv.y);
					vlcid.push_back(lcid), vlfid.push_back(lfid), vmatches.push_back(uv);
				}
				for (int ii = 0; ii < nmatches; ii++)
				{
					for (int jj = 0; jj < nmatches; jj++)
					{
						int p1 = vlcid[ii], p2 = vlcid[jj];
						matches.first = vmatches[ii], matches.second = vmatches[jj];
						allPairs[p1 + p2 * vCidFid.size()].push_back(matches);
					}
				}
				cnt++;
			}
			fclose(fp1);

			int gpid, lpid;
			double x, y, z, u, v, s = 1.0;
			vector<int> gpidVec1, gpidVec2;
			vector<Point2d> Vuv1, Vuv2, Vpuv1, Vpuv2;

			for (int ii = 0; ii < vCidFid.size() - 1; ii++)
			{
				for (int jj = ii + 1; jj < vCidFid.size(); jj++)
				{
					if (allPairs[ii + jj * vCidFid.size()].size() == 0)
						continue;

					gpidVec1.clear(), Vuv1.clear();
					sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, vCidFid[ii].x, vCidFid[ii].y); fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					while (fscanf(fp1, "%d %d %lf %lf %lf %lf %lf %lf ", &gpid, &lpid, &x, &y, &z, &u, &v, &s) != EOF)
					{
						gpidVec1.push_back(gpid);
						Vuv1.push_back(Point2d(u, v));
					}
					fclose(fp1);

					gpidVec2.clear(), Vuv2.clear();
					sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, vCidFid[jj].x, vCidFid[jj].y); fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					while (fscanf(fp1, "%d %d %lf %lf %lf %lf %lf %lf ", &gpid, &lpid, &x, &y, &z, &u, &v, &s) != EOF)
					{
						gpidVec2.push_back(gpid);
						Vuv2.push_back(Point2d(u, v));
					}
					fclose(fp1);

					Vpuv1.clear(), Vpuv2.clear();
					for (size_t kk = 0; kk < gpidVec1.size(); kk++)
					{
						for (size_t ll = 0; ll < gpidVec2.size(); ll++)
						{
							if (gpidVec1[kk] == gpidVec2[ll])
							{
								Vpuv1.push_back(Vuv1[kk]);
								Vpuv2.push_back(Vuv2[ll]);
								break;
							}
						}
					}

					sprintf(Fname, "%s/Dynamic/pwMatches/%d_%.4d_%d_%.4d.txt", Path2, vCidFid[ii].x, vCidFid[ii].y, vCidFid[jj].x, vCidFid[jj].y);
					fp1 = fopen(Fname, "w");
					for (int kk = 0; kk < Vpuv1.size(); kk++)
						fprintf(fp1, "%.2f %.2f %.2f %.2f\n", Vpuv1[kk].x, Vpuv1[kk].y, Vpuv2[kk].x, Vpuv2[kk].y);
					//fclose(fp1);

					//sprintf(Fname, "%s/Dynamic/pwMatches/%d_%.4d_%d_%.4d.txt", Path, vCidFid[ii].x, vCidFid[ii].y, vCidFid[jj].x, vCidFid[jj].y);
					//fp1 = fopen(Fname, "a");
					for (int kk = 0; kk < allPairs[ii + jj * vCidFid.size()].size(); kk++)
						fprintf(fp1, "%.2f %.2f %.2f %.2f\n", allPairs[ii + jj * vCidFid.size()][kk].first.x, allPairs[ii + jj * vCidFid.size()][kk].first.y, allPairs[ii + jj * vCidFid.size()][kk].second.x, allPairs[ii + jj * vCidFid.size()][kk].second.y);
					fclose(fp1);
				}
			}

			if (0)
			{
				vector<Mat>vImg;
				for (size_t ii = 0; ii < vCidFid.size(); ii++)
				{
					int cid = vCidFid[ii].x, fid = vCidFid[ii].y;
					sprintf(Fname, "%s/%d/%.4d.png", Path, cid, fid);
					vImg.push_back(imread(Fname));
				}

				namedWindow("X", CV_WINDOW_NORMAL);
				for (int ii = 0; ii < vCidFid.size() - 1; ii++)
				{
					for (int jj = ii + 1; jj < vCidFid.size(); jj++)
					{
						Mat img1 = vImg[ii].clone(), img2 = vImg[jj].clone();
						for (int kk = 0; kk < allPairs[ii + jj * vCidFid.size()].size(); kk++)
						{
							circle(img1, allPairs[ii + jj * vCidFid.size()][kk].first, 2, colors[kk % 9], 2);
							circle(img2, allPairs[ii + jj * vCidFid.size()][kk].second, 2, colors[kk % 9], 2);
						}
						int width = img1.cols, height = img1.rows;
						Mat bImg(height, width * 2, CV_8UC3, Scalar(0, 0, 0));

						Rect rect1(0, 0, width, height);
						Rect rect2(width, 0, width, height);
						img1.copyTo(bImg(rect1));
						img2.copyTo(bImg(rect2));

						cv::Point2i text_origin = { bImg.rows / 20, bImg.cols / 20 };
						sprintf(Fname, "%zd", allPairs[ii + jj * vCidFid.size()].size());
						putText(bImg, Fname, text_origin, CV_FONT_HERSHEY_SIMPLEX, 0.5*bImg.cols / 640, Scalar(0, 0, 255), 3);
						imshow("X", bImg); waitKey(0);
					}
				}
			}
		}
		fclose(fp);

		printLOG("\n");
	}
	delete[] AllSiftDesc, delete[]AllSiftPoints;
	delete[]PassedTri, delete[]vPts, delete[]A, delete[]B, delete[]P, delete[]tP;

	return 0;
}
int GenSparseCorres4IRB_Driver2(char *Path, int nCams, int startF, int stopF)
{
	char Fname[512];
	for (int Rcid = 0; Rcid < nCams; Rcid++)
	{
		printLOG("Working on %d: ", Rcid);

		sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, Rcid);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		int Rfid, nRcid, nRfid, NN;
		vector<Point2i> vCidFid; double dummy;
		while (fscanf(fp, "%d %d %d ", &Rcid, &Rfid, &NN) != EOF)
		{
			printLOG("%d: ", Rfid);
			if (Rfid > stopF)
				break;

			vCidFid.clear();
			vCidFid.push_back(Point2i(Rcid, Rfid));
			for (int nn = 0; nn < NN; nn++)
			{
				fscanf(fp, "%d %d ", &nRcid, &nRfid);
				for (int ii = 0; ii < 9; ii++)
					fscanf(fp, "%lf ", &dummy);
				for (int ii = 0; ii < 3; ii++)
					fscanf(fp, "%lf ", &dummy);

				vCidFid.push_back(Point2i(nRcid, nRfid));
			}

			int gpid, lpid;
			double x, y, z, u, v, s = 1.0;
			vector<int> gpidVec1, gpidVec2;
			vector<Point2d> Vuv1, Vuv2, Vpuv1, Vpuv2;

			gpidVec1.clear(), Vuv1.clear();
			sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, vCidFid[0].x, vCidFid[0].y); FILE *fp1 = fopen(Fname, "r");
			if (fp1 == NULL)
				continue;
			while (fscanf(fp1, "%d %d %lf %lf %lf %lf %lf %lf ", &gpid, &lpid, &x, &y, &z, &u, &v, &s) != EOF)
			{
				gpidVec1.push_back(gpid);
				Vuv1.push_back(Point2d(u, v));
			}
			fclose(fp1);

			for (int jj = 1; jj < vCidFid.size(); jj++)
			{
				gpidVec2.clear(), Vuv2.clear();
				sprintf(Fname, "%s/%d/PnPf/Inliers_%.4d.txt", Path, vCidFid[jj].x, vCidFid[jj].y); fp1 = fopen(Fname, "r");
				if (fp1 == NULL)
					continue;
				while (fscanf(fp1, "%d %d %lf %lf %lf %lf %lf %lf ", &gpid, &lpid, &x, &y, &z, &u, &v, &s) != EOF)
				{
					gpidVec2.push_back(gpid);
					Vuv2.push_back(Point2d(u, v));
				}
				fclose(fp1);

				Vpuv1.clear(), Vpuv2.clear();
				for (size_t kk = 0; kk < gpidVec1.size(); kk++)
				{
					for (size_t ll = 0; ll < gpidVec2.size(); ll++)
					{
						if (gpidVec1[kk] == gpidVec2[ll])
						{
							Vpuv1.push_back(Vuv1[kk]);
							Vpuv2.push_back(Vuv2[ll]);
							break;
						}
					}
				}

				sprintf(Fname, "%s/Dynamic/pwMatches/%d_%.4d_%d_%.4d.txt", Path, vCidFid[0].x, vCidFid[0].y, vCidFid[jj].x, vCidFid[jj].y); 	fp1 = fopen(Fname, "a");
				for (int kk = 0; kk < Vpuv1.size(); kk++)
					fprintf(fp1, "%.2f %.2f %.2f %.2f\n", Vpuv1[kk].x, Vpuv1[kk].y, Vpuv2[kk].x, Vpuv2[kk].y);
				fclose(fp1);
			}
		}
		fclose(fp);

		printLOG("\n");
	}

	return 0;
}
int GenEpicSparseInterp(char *Path, int nCams, int startF, int stopF, int increF, int nPeople)
{
	char Fname[512];
	sprintf(Fname, "%s/Dynamic/pwMatches2", Path), makeDir(Fname);

	vector<int> sCams, TimeStamp;
	for (int ii = 0; ii < nCams; ii++)
		sCams.push_back(ii), TimeStamp.push_back(0);

	int nf, fid, pid;
	vector<Point2i> tracklet;
	vector<vector<Point2i> > *trackletVec = new vector<vector<Point2i> >[nCams];
	for (auto cid : sCams)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_%d_%d.txt", Path, cid, startF, stopF); FILE*fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		int np = 0;
		while (fscanf(fp, "%d ", &nf) != EOF)
		{
			tracklet.clear();
			for (int f = 0; f < nf; f++)
			{
				fscanf(fp, "%d %d ", &fid, &pid);
				tracklet.push_back(Point2i(fid, pid));
			}
			trackletVec[cid].push_back(tracklet);
			np++;
			if (np >= nPeople)
				break;
		}
		fclose(fp);
	}

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic,1)
	for (int Rcid = 0; Rcid < nCams; Rcid++)
	{
		printLOG("Working on %d: ", Rcid);
		char Fname[512];
		sprintf(Fname, "%s/Dynamic/pwMatches2/%d", Path, Rcid), makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/DenseGeoMatches/%d", Path, Rcid), makeDir(Fname);

		vector<Point2d> allPts[2];
		sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, Rcid);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		int Rfid, nRcid, nRfid, NN;
		double dummy;
		vector<Point2i> vCidFid;
		while (fscanf(fp, "%d %d %d ", &Rfid, &Rfid, &NN) != EOF) //rcid, rfid, nn
		{
			printLOG("%d: ", Rfid);
			vCidFid.clear();
			vCidFid.push_back(Point2i(Rcid, Rfid));
			for (int nn = 0; nn < NN; nn++)
			{
				fscanf(fp, "%d %d ", &nRcid, &nRfid);
				for (int ii = 0; ii < 9; ii++)
					fscanf(fp, "%lf ", &dummy);
				for (int ii = 0; ii < 3; ii++)
					fscanf(fp, "%lf ", &dummy);

				vCidFid.push_back(Point2i(nRcid, nRfid));
			}

			for (int ii = 1; ii < vCidFid.size(); ii++)
			{
				nRcid = vCidFid[ii].x, nRfid = vCidFid[ii].y;

				sprintf(Fname, "%s/Dynamic/pwMatches2/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rfid, nRcid, nRfid);
				if (IsFileExist(Fname) == 0)
				{
					vector<Point2f> vuv1, vuv2; vector<float> vconf;
					float u1, v1, u2, v2, u, v, s, detectionThresh = 0.1f;
					sprintf(Fname, "%s/Dynamic/pwMatches/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rfid, nRcid, nRfid);
					FILE *fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					while (fscanf(fp1, "%f %f %f %f\n", &u1, &v1, &u2, &v2) != EOF)
						vuv1.push_back(Point2f(u1, v1)), vuv2.push_back(Point2f(u2, v2)), vconf.push_back(1.f);
					fclose(fp1);

					vector<Point2i> pairDetections;
					for (int pid = 0; pid < nPeople; pid++)
					{
						int lid1 = -1, lid2 = -1;
						for (int jj = 0; jj < trackletVec[Rcid][pid].size() && lid1 == -1; jj++)
						{
							if (trackletVec[Rcid][pid][jj].x == Rfid)
								lid1 = trackletVec[Rcid][pid][jj].y;
						}

						for (int jj = 0; jj < trackletVec[nRcid][pid].size() && lid2 == -1; jj++)
						{
							if (trackletVec[nRcid][pid][jj].x == nRfid)
								lid2 = trackletVec[nRcid][pid][jj].y;
						}
						if (lid1 != -1 && lid2 != -1)
							pairDetections.push_back(Point2i(lid1, lid2));
					}
					if (pairDetections.size() == 0)
						continue;

					allPts[0].clear(), allPts[1].clear();
					sprintf(Fname, "%s/MP/%d/%.4d.txt", Path, Rcid, Rfid); fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					while (fscanf(fp1, "%f %f %f ", &u, &v, &s) != EOF)
						if (s < detectionThresh)
							allPts[0].push_back(Point2f(0, 0));
						else
							allPts[0].push_back(Point2f(u, v));
					fclose(fp1);

					sprintf(Fname, "%s/MP/%d/%.4d.txt", Path, nRcid, nRfid); fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					while (fscanf(fp1, "%f %f %f ", &u, &v, &s) != EOF)
						if (s < detectionThresh)
							allPts[1].push_back(Point2f(0, 0));
						else
							allPts[1].push_back(Point2f(u, v));
					fclose(fp1);

					for (int pid = 0; pid < pairDetections.size(); pid++)
						for (int jid = 0; jid < 18; jid++)
							if (allPts[0][pairDetections[pid].x * 18 + jid].x > 0 && allPts[1][pairDetections[pid].y * 18 + jid].x > 0)
								vuv1.push_back(allPts[0][pairDetections[pid].x * 18 + jid]), vuv2.push_back(allPts[1][pairDetections[pid].y * 18 + jid]), vconf.push_back(0.2f);

					sprintf(Fname, "%s/Dynamic/pwMatches2/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rfid, nRcid, nRfid);
					fp1 = fopen(Fname, "w");
					for (int jj = 0; jj < vuv1.size(); jj++)
						fprintf(fp1, "%.3f %.3f %.3f %.3f %.1f\n", vuv1[jj].x, vuv1[jj].y, vuv2[jj].x, vuv2[jj].y, vconf[jj]);
					fclose(fp1);
				}

				/*sprintf(Fname, "./epicflow %s/%d/%d.png /%s/%d/%d.png  %s/Edges/%d/%.4d.edge %s/Dynamic/pwMatches2/%d_%.4d_%d_%.4d.txt 5 %s/Dynamic/DenseGeoMatches/%d/%d_%.4d_%d_%.4d.xyc",
					Path, Rcid, Rfid,
					Path, nRcid, nRfid,
					Path, Rcid, Rfid,
					Path, Rcid, Rfid, nRcid, nRfid,
					Path, Rcid, Rcid, Rfid, nRcid, nRfid);
#pragma omp critical
				printf("%s\n", Fname);
				system(Fname);*/
			}
		}
		fclose(fp);
	}

	return 0;
}
int GenRectifiedEpicSparseInterp(char *Path, int nCams, int startF, int stopF, int increF, int nPeople)
{
	char Fname[512];
	char Path2[] = "D:/Dance";
	sprintf(Fname, "%s/Dynamic/pwMatchesR", Path2), makeDir(Fname);
	sprintf(Fname, "%s/Dynamic/DenseGeoMatchesR", Path2), makeDir(Fname);

	VideoData *AllVideosInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		ReadVideoDataI("X:/User/minh/CVPR16/Dance/60fps/CVPR16", AllVideosInfo[cid], cid, startF, stopF);

	vector<int> sCams, TimeStamp;
	for (int ii = 0; ii < nCams; ii++)
		sCams.push_back(ii), TimeStamp.push_back(0);

	int nf, fid, pid;
	vector<Point2i> tracklet;
	vector<vector<Point2i> > *trackletVec = new vector<vector<Point2i> >[nCams];
	for (auto cid : sCams)
	{
		sprintf(Fname, "%s/%d/CleanedMergedTracklets_%d_%d.txt", Path, cid, startF, stopF); FILE*fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		int np = 0;
		while (fscanf(fp, "%d ", &nf) != EOF)
		{
			tracklet.clear();
			for (int f = 0; f < nf; f++)
			{
				fscanf(fp, "%d %d ", &fid, &pid);
				tracklet.push_back(Point2i(fid, pid));
			}
			trackletVec[cid].push_back(tracklet);
			np++;
			if (np >= nPeople)
				break;
		}
		fclose(fp);
	}

	omp_set_num_threads(omp_get_max_threads());
	//#pragma omp parallel for schedule(dynamic,1)
	for (int Rcid = 0; Rcid < nCams; Rcid++)
	{
		printLOG("Working on %d: ", Rcid);
		char Fname[512];
		sprintf(Fname, "%s/Dynamic/DenseGeoMatchesR/%d", Path, Rcid), makeDir(Fname);

		vector<Point2d> allPts[2];
		sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, Rcid);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		int Rfid, nRcid, nRfid, NN;
		double dummy;
		vector<Point2i> vCidFid;
		while (fscanf(fp, "%d %d %d ", &Rfid, &Rfid, &NN) != EOF) //rcid, rfid, nn
		{
			printLOG("%d: ", Rfid);
			if (Rfid > stopF)
				break;

			vCidFid.clear();
			vCidFid.push_back(Point2i(Rcid, Rfid));
			for (int nn = 0; nn < NN; nn++)
			{
				fscanf(fp, "%d %d ", &nRcid, &nRfid);
				for (int ii = 0; ii < 9; ii++)
					fscanf(fp, "%lf ", &dummy);
				for (int ii = 0; ii < 3; ii++)
					fscanf(fp, "%lf ", &dummy);

				vCidFid.push_back(Point2i(nRcid, nRfid));
			}

			for (int ii = 1; ii < vCidFid.size(); ii++)
			{
				nRcid = vCidFid[ii].x, nRfid = vCidFid[ii].y;

				sprintf(Fname, "%s/Dynamic/pwMatchesR/%d_%.4d_%d_%.4d.txt", Path2, Rcid, Rfid, nRcid, nRfid);
				//if (IsFileExist(Fname) == 0)
				{
					vector<Point2f> vuv1, vuv2; vector<float> vconf;
					float u1, v1, u2, v2, detectionThresh = 0.1f;
					sprintf(Fname, "%s/Dynamic/pwMatches/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rfid, nRcid, nRfid);
					FILE *fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					while (fscanf(fp1, "%f %f %f %f\n", &u1, &v1, &u2, &v2) != EOF)
						vuv1.push_back(Point2f(u1, v1)), vuv2.push_back(Point2f(u2, v2)), vconf.push_back(1.f);
					fclose(fp1);

					vector<Point2i> pairDetections;
					for (int pid = 0; pid < nPeople; pid++)
					{
						int lid1 = -1, lid2 = -1;
						for (int jj = 0; jj < trackletVec[Rcid][pid].size() && lid1 == -1; jj++)
						{
							if (trackletVec[Rcid][pid][jj].x == Rfid)
								lid1 = trackletVec[Rcid][pid][jj].y;
						}

						for (int jj = 0; jj < trackletVec[nRcid][pid].size() && lid2 == -1; jj++)
						{
							if (trackletVec[nRcid][pid][jj].x == nRfid)
								lid2 = trackletVec[nRcid][pid][jj].y;
						}
						if (lid1 != -1 && lid2 != -1)
							pairDetections.push_back(Point2i(lid1, lid2));
					}
					if (pairDetections.size() == 0)
						continue;

					allPts[0].clear(), allPts[1].clear();
					{
						sprintf(Fname, "%s/MP/%d/x_%.12d_keypoints.json", Path, Rcid, Rfid);
						if (IsFileExist(Fname) == 0)
						{
							sprintf(Fname, "%s/MP/%d/%.4d_keypoints.json", Path, Rcid, Rfid);
							if (IsFileExist(Fname) == 0)
								continue;
						}
						FileStorage fs(Fname, 0);
						FileNode root = fs["people"];
						for (int i = 0; i < root.size(); i++)
						{
							FileNode val1 = root[i]["pose_keypoints_2d"];
							for (int j = 0; j < val1.size(); j += 3)
							{
								if (val1[j + 2].real() < detectionThresh)
									allPts[0].push_back(Point2f(0, 0));
								else
									allPts[0].push_back(Point2f(val1[j].real(), val1[j + 1].real()));
							}
						}
					}

					{
						sprintf(Fname, "%s/MP/%d/x_%.12d_keypoints.json", Path, nRcid, nRfid);
						if (IsFileExist(Fname) == 0)
						{
							sprintf(Fname, "%s/MP/%d/%.4d_keypoints.json", Path, nRcid, nRfid);
							if (IsFileExist(Fname) == 0)
								continue;
						}
						FileStorage fs(Fname, 0);
						FileNode root = fs["people"];
						for (int i = 0; i < root.size(); i++)
						{
							FileNode val1 = root[i]["pose_keypoints_2d"];
							for (int j = 0; j < val1.size(); j += 3)
							{
								if (val1[j + 2].real() < detectionThresh)
									allPts[1].push_back(Point2f(0, 0));
								else
									allPts[1].push_back(Point2f(val1[j].real(), val1[j + 1].real()));
							}
						}
					}

					int peoplePoint = 0;
					for (int pid = 0; pid < pairDetections.size(); pid++)
					{
						for (int jid = 0; jid < 25; jid++)
						{
							if (allPts[0][pairDetections[pid].x * 25 + jid].x > 0 && allPts[1][pairDetections[pid].y * 25 + jid].x > 0)
							{
								peoplePoint++;
								vuv1.push_back(allPts[0][pairDetections[pid].x * 25 + jid]), vuv2.push_back(allPts[1][pairDetections[pid].y * 25 + jid]), vconf.push_back(0.2f);
							}
						}
					}

					//Warp matches
					double nR0[9], nR1[9], nr0[3], nr1[3], nP0[12], nP1[12];
					sprintf(Fname, "%s/CVPR16/Recitified/Pairwise/%d/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rcid, Rfid, nRcid, nRfid);
					fp1 = fopen(Fname, "r");
					if (fp1 == NULL)
						continue;
					fscanf(fp1, "%lf %lf %lf\n", &nr0[0], &nr0[1], &nr0[2]);
					for (int ii = 0; ii < 12; ii++)
						fscanf(fp1, "%lf ", &nP0[ii]);
					fscanf(fp1, "%lf %lf %lf\n", &nr1[0], &nr1[1], &nr1[2]);
					for (int ii = 0; ii < 12; ii++)
						fscanf(fp1, "%lf ", &nP1[ii]);
					fclose(fp1);

					getRfromr(nr0, nR0), getRfromr(nr1, nR1);

					/*if(0)
					{
					Mat img0, img1, img0r, img1r, map00, map01, map10, map11;
					cv::Size imageSize; imageSize.width = 1920, imageSize.height = 1080;

					double distortion0[5] = { AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[0], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[1], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[3], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[4], AllVideosInfo[Rcid].VideoInfo[Rfid].distortion[2] };
					double distortion1[5] = { AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[0], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[1], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[3], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[4], AllVideosInfo[nRcid].VideoInfo[nRfid].distortion[2] };

					Mat cvK0(3, 3, CV_64F, AllVideosInfo[Rcid].VideoInfo[Rfid].K), cvK1(3, 3, CV_64F, AllVideosInfo[nRcid].VideoInfo[nRfid].K);
					Mat cvDistortion0(1, 5, CV_64F, distortion0), cvDistortion1(1, 5, CV_64F, distortion1);
					Mat cvnR0(3, 3, CV_64F, nR0), cvnR1(3, 3, CV_64F, nR1), cvnP0(3, 4, CV_64F, nP0), cvnP1(3, 4, CV_64F, nP1);

					initUndistortRectifyMap(cvK0, cvDistortion0, cvnR0, cvnP0, imageSize, CV_16SC2, map00, map01);
					initUndistortRectifyMap(cvK1, cvDistortion1, cvnR1, cvnP1, imageSize, CV_16SC2, map10, map11);

					sprintf(Fname, "%s/%d/%.4d.png", Path, Rcid, Rfid);
					if (IsFileExist(Fname) == 0)
					{
					sprintf("%s/%d/%.4d.jpg", Path, Rcid, Rfid);
					if (IsFileExist(Fname) == 0)
					continue;
					}
					img0 = imread(Fname);

					sprintf(Fname, "%s/%d/%.4d.png", Path, nRcid, nRfid);
					if (IsFileExist(Fname) == 0)
					{
					sprintf("%s/%d/%.4d.jpg", Path, nRcid, nRfid);
					if (IsFileExist(Fname) == 0)
					continue;
					}
					img1 = imread(Fname);

					remap(img0, img0r, map00, map01, INTER_CUBIC);
					remap(img1, img1r, map10, map11, INTER_CUBIC);

					sprintf(Fname, "C:/temp/%d_%.4d.jpg", Rcid, Rfid); imwrite(Fname, img0r);
					sprintf(Fname, "C:/temp/%d_%.4d.jpg", nRcid, nRfid); imwrite(Fname, img1r);
					}*/

					for (int jj = 0; jj < vuv1.size(); jj++)
					{
						UndistortAndRectifyPoint(AllVideosInfo[Rcid].VideoInfo[Rfid].K, AllVideosInfo[Rcid].VideoInfo[Rfid].distortion, nR0, nP0, vuv1[jj]); //Q1*ij
						UndistortAndRectifyPoint(AllVideosInfo[nRcid].VideoInfo[nRfid].K, AllVideosInfo[nRcid].VideoInfo[nRfid].distortion, nR1, nP1, vuv2[jj]); //Q2*ij
					}

					sprintf(Fname, "%s/Dynamic/pwMatchesR/%d_%.4d_%d_%.4d.txt", Path, Rcid, Rfid, nRcid, nRfid);
					fp1 = fopen(Fname, "w");
					for (int jj = 0; jj < vuv1.size(); jj++)
						fprintf(fp1, "%.3f %.3f %.3f %.3f %.1f\n", vuv1[jj].x, vuv1[jj].y, vuv2[jj].x, vuv2[jj].y, vconf[jj]);
					fclose(fp1);
				}

				/*sprintf(Fname, "%s/Dynamic/DenseGeoMatchesR/%d/%.4d_%d_%.4d_1.xyc", Path, Rcid, Rfid, nRcid, nRfid);
				if (IsFileExist(Fname) == 0)
				{
				sprintf(Fname, "./epicflow %s/Recitified/%d/%.4d_%d_%.4d_1.png %s/Recitified/%d/%.4d_%d_%.4d_2.png  %s/EdgesR/%d/%.4d_%d_%.4d_1.edge %s/Dynamic/pwMatchesR/%d_%.4d_%d_%.4d.txt 5 %s/Dynamic/DenseGeoMatchesR/%d/%.4d_%d_%.4d_1.xyc",
				Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid);
				#pragma omp critical
				//printf("%s\n", Fname);
				//system(Fname);
				}

				sprintf(Fname, "%s/Dynamic/DenseGeoMatchesR/%d/%.4d_%d_%.4d_2.xyc", Path, Rcid, Rfid, nRcid, nRfid);
				if (IsFileExist(Fname) == 0)
				{
				sprintf(Fname, "./epicflow %s/Recitified/%d/%.4d_%d_%.4d_2.png %s/Recitified/%d/%.4d_%d_%.4d_1.png  %s/EdgesR/%d/%.4d_%d_%.4d_2.edge %s/Dynamic/pwMatchesR/%d_%.4d_%d_%.4d.txt 5 %s/Dynamic/DenseGeoMatchesR/%d/%.4d_%d_%.4d_2.xyc -backwarp",
				Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid, Path, Rcid, Rfid, nRcid, nRfid);
				#pragma omp critical
				//printf("%s\n", Fname);
				//system(Fname);
				}*/
			}
		}
		fclose(fp);
	}

	return 0;
}

//Corpus mesh rendering
int GenCorpusTriPerVideo(char *Path, int nCams, int startF, int stopF, int LookupRange = 10, int increFLookup = 50)
{
	char Fname[512];
	char Path2[] = "Z:/Users/minh/Snow";

	VideoData *AllVideosInfo = new VideoData[nCams];
	for (int cid = 0; cid < nCams; cid++)
		ReadVideoDataI(Path, AllVideosInfo[cid], cid, startF, stopF);

	vector<Point3f> Vxyz;
	sprintf(Fname, "%s/Corpus/Corpus_3D.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	Point3f xyz; Point3i rgb;
	int nCameras, n3dPoints, useColor;
	fscanf(fp, "%d %d %d", &nCameras, &n3dPoints, &useColor);
	Vxyz.reserve(n3dPoints);
	if (useColor)
	{
		for (int jj = 0; jj < n3dPoints; jj++)
		{
			fscanf(fp, "%f %f %f %d %d %d", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z);
			Vxyz.push_back(xyz);
		}
	}
	else
	{
		for (int jj = 0; jj < n3dPoints; jj++)
		{
			fscanf(fp, "%f %f %f ", &xyz.x, &xyz.y, &xyz.z);
			Vxyz.push_back(xyz);
		}
	}
	fclose(fp);

	//determine visibility for all 3d points (info only available for keyframes before)

	//vector<vector<Point2i>*> VvisibleId; VvisibleId.reserve(n3dPoints);
	//vector<vector<Point2f>*> VprojectedLoc; VprojectedLoc.reserve(n3dPoints);

	//sprintf(Fname1, "%s/CorpusAllFramesVisibilty.dat", Path);
	//sprintf(Fname2, "%s/CorpusAllFramesProjectionLoc.dat", Path);

	/*for (int pid = 0; pid < n3dPoints; pid++)
	{
	vector<Point2i> * VisI = new vector<Point2i>[1];
	vector<Point2f> * projectedLocI = new vector<Point2f>[1];
	VvisibleId.push_back(VisI);
	VprojectedLoc.push_back(projectedLocI);
	}

	Point2i *VisI = new Point2i[n3dPoints];
	Point2f *ProjLocI = new Point2f[n3dPoints];*/

	Mat mask;
	vector<int> VVisPid;
	bool *validPoint = new bool[n3dPoints];
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname, "%s/Logs/VisAllFrames_%d.txt", Path, cid);
		if (IsFileExist(Fname) == 1)
			continue;

		printLOG("Camera %d: ", cid);
		sprintf(Fname, "%s/%d/Vis", Path, cid); makeDir(Fname);

		CameraData *camI = AllVideosInfo[cid].VideoInfo;

		for (int fid = startF; fid <= stopF; fid++)
		{
			if (fid % 100 == 0)
				printLOG("%d..", fid);

			if (camI[fid].valid == 0)
				continue;

			sprintf(Fname, "%s/%d/labelled/%.4d.png", Path2, cid, fid); mask = imread(Fname, 0);
			if (mask.empty() == 1)
				continue;

#pragma omp parallel for schedule(dynamic, 100)
			for (int pid = 0; pid < n3dPoints; pid++)
			{
				//VisI[pid].x = -1, VisI[pid].y = -1, ProjLocI[pid].x = -1, ProjLocI[pid].y = -1;
				validPoint[pid] = false;

				double direction[3] = { Vxyz[pid].x - camI[fid].camCenter[0], Vxyz[pid].y - camI[fid].camCenter[1], Vxyz[pid].z - camI[fid].camCenter[1] };
				double dot = dotProduct(camI[fid].principleRayDir, direction);
				if (dot < 0) //behind the camera
					continue;

				Point2d projectedPt;
				if (camI[fid].LensModel == RADIAL_TANGENTIAL_PRISM)
					ProjectandDistort(Vxyz[pid], &projectedPt, camI[fid].P, camI[fid].K, camI[fid].distortion, 1); //approx is fine for this task. No RS is needed
				else
					FisheyeProjectandDistort(Vxyz[pid], &projectedPt, camI[fid].P, camI[fid].distortion, 1);

				if (projectedPt.x <0 || projectedPt.y <0 || projectedPt.x > camI[fid].width - 1 || projectedPt.y > camI[fid].height - 1)
					continue;

				if (mask.data[(int)(projectedPt.x + 0.5) + (int)(projectedPt.y + 0.5)*camI[fid].width] > (uchar)0)
					continue;

				validPoint[pid] = true;
				//VisI[pid].x = cid, VisI[pid].y = fid, ProjLocI[pid].x = projectedPt.x, ProjLocI[pid].y = projectedPt.y;
			}

			VVisPid.clear();
			for (int pid = 0; pid < n3dPoints; pid++)
				//if (VisI[pid].x > -1)
				if (validPoint[pid])
					VVisPid.push_back(pid); //VprojectedLoc[pid][0].push_back(ProjLocI[pid]), VvisibleId[pid][0].push_back(VisI[pid]);

			sprintf(Fname, "%s/%d/Vis/%.4d.dat", Path, cid, fid);
			ofstream fout; fout.open(Fname, ios::binary);
			if (!fout.is_open())
			{
				printLOG("Cannot write %s\n", Fname);
				return false;
			}
			int nvis = (int)VVisPid.size();
			fout.write(reinterpret_cast<char *>(&nvis), sizeof(int));
			for (int ii = 0; ii < nvis; ii++)
				fout.write(reinterpret_cast<char *>(&VVisPid[ii]), sizeof(int));
			fout.close();
		}
		printLOG("\n");

		sprintf(Fname, "%s/Logs/VisAllFrames_%d.txt", Path, cid); fp = fopen(Fname, "w+"); fclose(fp);
	}

	delete[]validPoint;
	/*ofstream fout;
	sprintf(Fname, "%s/CorpusAllFramesVisibilty.dat", Path); fout.open(Fname, ios::binary);
	if (!fout.is_open())
	{
	cout << "Cannot write: " << Fname << endl;
	return false;
	}
	fout.write(reinterpret_cast<char *>(&n3dPoints), sizeof(int));
	for (int pid = 0; pid < n3dPoints; pid++)
	{
	int nf = (int)VvisibleId[pid][0].size();
	fout.write(reinterpret_cast<char *>(&nf), sizeof(int));
	for (int ii = 0; ii < nf; ii++)
	fout.write(reinterpret_cast<char *>(&VvisibleId[pid][0][ii].x), sizeof(int)), fout.write(reinterpret_cast<char *>(&VvisibleId[pid][0][ii].y), sizeof(int));
	}
	fout.close();

	sprintf(Fname, "%s/CorpusAllFramesProjectionLoc.dat", Path); fout.open(Fname, ios::binary);
	if (!fout.is_open())
	{
	cout << "Cannot write: " << Fname << endl;
	return false;
	}
	fout.write(reinterpret_cast<char *>(&n3dPoints), sizeof(int));
	for (int pid = 0; pid < n3dPoints; pid++)
	{
	int nf = (int)VvisibleId[pid][0].size();
	fout.write(reinterpret_cast<char *>(&nf), sizeof(int));
	for (int ii = 0; ii < nf; ii++)
	{
	int x = (int)(VprojectedLoc[pid][0][ii].x * 1000 + 0.5), y = (int)(VprojectedLoc[pid][0][ii].y * 1000 + 0.5);
	fout.write(reinterpret_cast<char *>(&x), sizeof(int)), fout.write(reinterpret_cast<char *>(&y), sizeof(int));
	}
	}
	fout.close();*/


	return 0;
}

int GenCameraPath(char *Path, int nCams, int startF, int stopF, int increSampling)
{
	char Fname[512];

	int cid;
	vector<int> CamPassingOrder;
	sprintf(Fname, "%s/Vis/CamPassingOrder.txt", Path);
	if (IsFileExist(Fname) == 0)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d ", &cid) != EOF)
		CamPassingOrder.push_back(cid);
	fclose(fp);

	//read time aligment info
	double offset, fps;
	Point3d *CamTimeInfo = new Point3d[nCams + 1];
	for (int ii = 0; ii < nCams + 1; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;

	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %lf %lf ", &cid, &fps, &offset) != EOF)
		{
			CamTimeInfo[cid].x = 1.0 / fps;
			CamTimeInfo[cid].y = offset;
			CamTimeInfo[cid].z = 1.0;
		}
		fclose(fp);
	}

	printLOG("Reading all camera poses\n");
	VideoData *VideoInfo = new VideoData[nCams + 1];
	for (int cid = 0; cid < nCams; cid++)
		if (ReadVideoDataI(Path, VideoInfo[cid], cid, -1, -1) == 1)
			continue;

	//find the ref (earliest) camera
	int refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y * CamTimeInfo[ii].x) //ts = offset_i/fps_ref + f_i/fps_i;
			earliest = CamTimeInfo[ii].y * CamTimeInfo[ii].x, refCid = ii;

	//find where the ref camera is in the camerapassingorder
	int currentCamPosInCamPassingOrder = -1;
	for (int ii = 0; ii < (int)CamPassingOrder.size() && currentCamPosInCamPassingOrder == -1; ii++)
		if (refCid == CamPassingOrder[ii])
			currentCamPosInCamPassingOrder = ii;

	//push camera pose into stack
	int cRefFid = startF;
	vector<Point3i> FKStack;
	while (true)
	{
		int realCid = CamPassingOrder[currentCamPosInCamPassingOrder%CamPassingOrder.size()];
		int realFid = (cRefFid - CamTimeInfo[realCid].y)*CamTimeInfo[refCid].x / CamTimeInfo[realCid].x;

		if (VideoInfo[realCid].VideoInfo[realFid].valid == 1)
			FKStack.push_back(Point3i(cRefFid, realCid, realFid));

		currentCamPosInCamPassingOrder++;
		cRefFid += increSampling;

		if (cRefFid > stopF)
			break;
	}

	sprintf(Fname, "%s/Vis/CamPathKeyframes.txt", Path);  fp = fopen(Fname, "w");
	for (auto kf : FKStack)
	{
		CameraData *camI = &VideoInfo[kf.y].VideoInfo[kf.z];
		fprintf(fp, "%d ", kf.x);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%f ", camI[0].intrinsic[ii]);
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%f ", camI[0].rt[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);


	{
		nCams = nCams + 1;
		int kNN = nCams;
		vector<int> sCams;
		for (int ii = 0; ii < nCams; ii++)
			sCams.push_back(ii);

		//Read calib info
		int nCams = *std::max_element(std::begin(sCams), std::end(sCams)) + 1;
		vector<int>*allVrFid = new vector<int>[nCams];
		vector<Point2i> **allNearestKeyFrames = new vector<Point2i>*[nCams];

		vector<int> TimeStamp(nCams);
		bool useSyncedFramesOnly = true;
		sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot open %s\n", Fname);
			return 1;
		}
		int id; float fps, offset;
		while (fscanf(fp, "%d %f %f ", &id, &fps, &offset) != EOF)
		{
			for (size_t ii = 0; ii < sCams.size(); ii++)
			{
				if (sCams[ii] == id)
				{
					TimeStamp[id] = (int)(offset + 0.5);
					break;
				}
			}
		}
		fclose(fp);

		int cnt = 0;
		for (size_t ii = 0; ii < sCams.size(); ii++)
			for (int fid = startF; fid <= stopF; fid++)
				allVrFid[cid].push_back(fid);

		if (ReadVideoDataI(Path, VideoInfo[nCams - 1], nCams - 1, -1, -1) == 1)
			return 1;
		InvalidateAbruptCameraPose(VideoInfo[nCams - 1], -1, -1, 0);

		const double cos_45 = cos(Pi / 4);

		vector<Point2i> cid_fid;
		vector<size_t> indexList;
		vector<double> unsorted, sorted;
		for (size_t ii = 10; ii < sCams.size(); ii++)
		{
			int cid = sCams[ii];
			printLOG("%d..", cid);
			allNearestKeyFrames[cid] = new vector<Point2i>[stopF + 1];

			for (int fid = startF; fid <= stopF; fid++)
			{
				CameraData *refCam = &VideoInfo[cid].VideoInfo[fid];
				if (!refCam[0].valid)
					continue;

				double uv1[3] = { refCam[0].width / 2, refCam[0].height / 2, 1 }, rayDir[3];
				getRayDir(rayDir, refCam[0].invK, refCam[0].R, uv1);

				cid_fid.clear(), indexList.clear(), unsorted.clear(), sorted.clear();
				for (size_t jj = 0; jj < sCams.size(); jj++)
				{
					if (useSyncedFramesOnly)
					{
						if (ii == jj)
							continue;

						int fid_nr = fid + TimeStamp[ii] - TimeStamp[jj]; //f = f_ref - offset;
						if (fid_nr<startF || fid_nr>stopF)
							continue;
						CameraData *Cam_nr = &VideoInfo[sCams[jj]].VideoInfo[fid_nr];
						if (!Cam_nr[0].valid)
							continue;

						double uv1_nr[3] = { Cam_nr[0].width / 2, Cam_nr[0].height / 2, 1 }, rayDir_nr[3];
						getRayDir(rayDir_nr, Cam_nr[0].invK, Cam_nr[0].R, uv1_nr);

						double angle = dotProduct(rayDir, rayDir_nr);
						if (angle < cos_45)
							continue;

						double baseline = Distance3D(refCam[0].camCenter, Cam_nr[0].camCenter);
						double metric = baseline / angle; //smaller is better

						cid_fid.push_back(Point2i(jj, fid_nr));
						unsorted.push_back(metric);
					}
					else
					{
						for (auto fid_nr : allVrFid[sCams[jj]])//only use keyframes
						{
							if (jj == ii && fid == fid_nr)
								continue;

							CameraData *Cam_nr = &VideoInfo[sCams[jj]].VideoInfo[fid_nr];
							double uv1_nr[3] = { Cam_nr[0].width / 2, Cam_nr[0].height / 2, 1 }, rayDir_nr[3];
							getRayDir(rayDir_nr, Cam_nr[0].invK, Cam_nr[0].R, uv1_nr);

							double angle = dotProduct(rayDir, rayDir_nr);
							if (angle < cos_45)
								continue;

							double baseline = Distance3D(refCam[0].camCenter, Cam_nr[0].camCenter);
							double metric = baseline / angle; //smaller is better

							cid_fid.push_back(Point2i(jj, fid_nr));
							unsorted.push_back(metric);
						}
					}
				}
				SortWithIndex(unsorted, sorted, indexList);

				for (int kk = 0; kk < min(kNN, indexList.size()); kk++)
					allNearestKeyFrames[cid][fid].push_back(cid_fid[indexList[kk]]);
			}
		}

		int cid = nCams - 1;
		sprintf(Fname, "%s/%d/kNN4IRB.txt", Path, cid); fp = fopen(Fname, "w");
		for (int fid = startF; fid <= stopF; fid++)
		{
			int knn = min(kNN, allNearestKeyFrames[cid][fid].size());
			fprintf(fp, "%d %d ", fid, knn);
			for (int ii = 0; ii < knn; ii++)
				fprintf(fp, "%d %d ", allNearestKeyFrames[cid][fid][ii].x, allNearestKeyFrames[cid][fid][ii].y);
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, cid); fp = fopen(Fname, "w");
		for (int fid = startF; fid <= stopF; fid++)
		{
			fprintf(fp, "%d %d %zd\n", cid, fid, min(kNN, allNearestKeyFrames[cid][fid].size()));
			for (int ii = 0; ii < min(kNN, allNearestKeyFrames[cid][fid].size()); ii++)
			{
				fprintf(fp, "%d %d ", allNearestKeyFrames[cid][fid][ii].x, allNearestKeyFrames[cid][fid][ii].y);

				CameraData *Ref = &VideoInfo[cid].VideoInfo[fid];
				CameraData *NRef = &VideoInfo[allNearestKeyFrames[cid][fid][ii].x].VideoInfo[allNearestKeyFrames[cid][fid][ii].y];

				double R1to0[9], T1to0[3];
				GetRelativeTransformation(Ref[0].R, Ref[0].T, NRef[0].R, NRef[0].T, R1to0, T1to0);
				for (int jj = 0; jj < 9; jj++)
					fprintf(fp, "%.16f ", R1to0[jj]);
				for (int jj = 0; jj < 3; jj++)
					fprintf(fp, "%.16f ", T1to0[jj]);
				fprintf(fp, "\n");
			}
		}
		fclose(fp);

		delete[]allNearestKeyFrames, delete[]VideoInfo, delete[]allVrFid;
		return 0;
	}

	return 0;
}
int ReadVideoDataI2(char *Path, vector<CameraData> &vInfo, int viewID, double threshold, int ninliersThresh, int silent)
{
	char Fname[512];
	int frameID, LensType, ShutterModel, width, height;
	CameraData FrameI;

	//READ INTRINSIC: START
	int validFrame = 0;
	sprintf(Fname, "%s/vHIntrinsic_%.4d.txt", Path, viewID);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/avIntrinsic_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			//printLOG("Cannot find %s...", Fname);
			sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printLOG("Cannot find %s...", Fname);
				sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					printLOG("Cannot find %s...\n", Fname);
					return 1;
				}
			}
		}
	}
	FILE *fp = fopen(Fname, "r");
	if (silent == 0)
		printLOG("Loaded %s\n", Fname);
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
	while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &ShutterModel, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		FrameI.K[0] = fx, FrameI.K[1] = skew, FrameI.K[2] = u0,
			FrameI.K[3] = 0.0, FrameI.K[4] = fy, FrameI.K[5] = v0,
			FrameI.K[6] = 0.0, FrameI.K[7] = 0.0, FrameI.K[8] = 1.0;

		FrameI.viewID = viewID;
		FrameI.frameID = frameID;
		FrameI.width = width, FrameI.height = height;
		GetIntrinsicFromK(FrameI);
		mat_invert(FrameI.K, FrameI.invK);

		FrameI.LensModel = LensType, FrameI.ShutterModel = ShutterModel, FrameI.threshold = threshold, FrameI.nInlierThresh = ninliersThresh;
		FrameI.hasIntrinsicExtrinisc++;
		validFrame = frameID;

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			FrameI.distortion[0] = r0, FrameI.distortion[1] = r1, FrameI.distortion[2] = r2;
			FrameI.distortion[3] = t0, FrameI.distortion[4] = t1;
			FrameI.distortion[5] = p0, FrameI.distortion[6] = p1;
		}
		else
		{
			fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			FrameI.distortion[0] = omega, FrameI.distortion[1] = DistCtrX, FrameI.distortion[2] = DistCtrY;
		}
		FrameI.width = width, FrameI.height = height;
		vInfo.push_back(FrameI);
	}
	fclose(fp);
	//END

	//READ POSE FROM VIDEO POSE: START
	sprintf(Fname, "%s/vHCamPose_%.4d.txt", Path, viewID);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/avCamPose_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			//printLOG("Cannot find %s...", Fname);
			sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printLOG("Cannot find %s...", Fname);
				sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					printLOG("Cannot find %s...\n", Fname);
					return 1;
				}
			}
		}
	}
	fp = fopen(Fname, "r");
	if (silent == 0)
		printLOG("Loaded %s\n", Fname);
	double rt[6], wt[6];
	for (int ii = 0; ii < vInfo.size(); ii++)
	{
		fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &frameID, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);
		if (ShutterModel == 1)
			for (int jj = 0; jj < 6; jj++)
				fscanf(fp, "%lf ", &wt[jj]);

		if (FrameI.hasIntrinsicExtrinisc < 1 || abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001)
		{
			FrameI.valid = false;
			continue;
		}

		if (vInfo[ii].hasIntrinsicExtrinisc > 0)
			vInfo[ii].valid = true;

		for (int jj = 0; jj < 6; jj++)
			vInfo[ii].rt[jj] = rt[jj];
		GetRTFromrt(vInfo[ii]);
		GetCfromT(vInfo[ii]);

		if (vInfo[ii].ShutterModel == 1)
			for (int jj = 0; jj < 6; jj++)
				vInfo[ii].wt[jj] = wt[jj];

		Rotation2Quaternion(vInfo[ii].R, vInfo[ii].Quat);

		GetRCGL(vInfo[ii]);
		AssembleP(vInfo[ii].K, vInfo[ii].R, vInfo[ii].T, vInfo[ii].P);

		double principal[] = { vInfo[ii].width / 2, vInfo[ii].height / 2, 1.0 };
		getRayDir(vInfo[ii].principleRayDir, vInfo[ii].invK, vInfo[ii].R, principal);
	}
	fclose(fp);
	//READ FROM VIDEO POSE: END

	return 0;
}
int CollectNearestViewsBaseOnGeometry2(char *Path, vector<CameraData> &NovelCam, int NovelCamId, vector<int> &sCams, int startF, int stopF, int kNN)
{
	char Fname[512]; FILE *fp = 0;

	//Read calib info
	int nCams = *std::max_element(std::begin(sCams), std::end(sCams)) + 1;
	VideoData *VideoInfo = new VideoData[nCams];
	vector<int>*allVrFid = new vector<int>[nCams];

	vector<int> TimeStamp(nCams);
	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0, CamTimeInfo[ii].z = 0.0;//alpha, beta, rs in t = alpha*(f+rs*row) + beta*alpha_ref

	int selected; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); fp = fopen(Fname, "r");
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
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	int cnt = 0;
	for (auto camID : sCams)
	{
		printLOG("%d..", camID);
		for (int fid = startF; fid <= stopF; fid++)
			allVrFid[camID].push_back(fid);

		if (ReadVideoDataI(Path, VideoInfo[camID], camID, -1, -1) == 1)
			continue;
		else
		{
			InvalidateAbruptCameraPose(VideoInfo[camID], -1, -1, 0);
			cnt++;
		}
	}
	if (cnt == 0)
		return 1;

	const double cos_45 = cos(Pi / 4);

	vector<Point2i> cid_fid;
	vector<size_t> indexList;
	vector<double> unsorted, sorted;
	vector<Point2i>  *allNearestKeyFrames = new vector<Point2i>[NovelCam.size()];

	for (int ii = 0; ii < NovelCam.size(); ii++)
	{
		double ts = (CamTimeInfo[refCid].x * CamTimeInfo[NovelCamId].y + CamTimeInfo[NovelCamId].x*  NovelCam[ii].frameID) / CamTimeInfo[refCid].x;
		if (!NovelCam[ii].valid)
			continue;

		double uv1[3] = { NovelCam[0].width / 2, NovelCam[0].height / 2, 1 }, rayDir[3];
		getRayDir(rayDir, NovelCam[0].invK, NovelCam[0].R, uv1);

		cid_fid.clear(), indexList.clear(), unsorted.clear(), sorted.clear();
		for (size_t jj = 0; jj < sCams.size(); jj++)
		{
			//let's assume the novel camera is sync to the earliest camera in your list
			int cid_nr = sCams[jj], fid_nr = (ts * CamTimeInfo[refCid].x - CamTimeInfo[refCid].x * CamTimeInfo[cid_nr].y) / CamTimeInfo[cid_nr].x;
			if (fid_nr<startF || fid_nr>stopF || cid_nr == NovelCamId)
				continue;

			CameraData *Cam_nr = &VideoInfo[cid_nr].VideoInfo[fid_nr];
			if (!Cam_nr[0].valid)
				continue;

			double uv1_nr[3] = { Cam_nr[0].width / 2, Cam_nr[0].height / 2, 1 }, rayDir_nr[3];
			getRayDir(rayDir_nr, Cam_nr[0].invK, Cam_nr[0].R, uv1_nr);

			double angle = dotProduct(rayDir, rayDir_nr);
			if (angle < cos_45)
				continue;

			double baseline = Distance3D(NovelCam[ii].camCenter, Cam_nr[0].camCenter);
			double metric = baseline / angle; //smaller is better

			cid_fid.push_back(Point2i(jj, fid_nr));
			unsorted.push_back(metric);
		}
		SortWithIndex(unsorted, sorted, indexList);

		for (int kk = 0; kk < min(kNN, indexList.size()); kk++)
			allNearestKeyFrames[ii].push_back(cid_fid[indexList[kk]]);
	}

	sprintf(Fname, "%s/%d", Path, NovelCamId); makeDir(Fname);
	sprintf(Fname, "%s/%d/kNN4IRB.txt", Path, NovelCamId);
	fp = fopen(Fname, "w");
	for (int jj = 0; jj < NovelCam.size(); jj++)
	{
		fprintf(fp, "%d %d ", NovelCam[jj].frameID, min(kNN, (int)allNearestKeyFrames[jj].size()));
		for (int ii = 0; ii < min(kNN, allNearestKeyFrames[jj].size()); ii++)
			fprintf(fp, "%d %d ", allNearestKeyFrames[jj][ii].x, allNearestKeyFrames[jj][ii].y);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/%d/kNN4IRB_dif.txt", Path, NovelCamId); fp = fopen(Fname, "w");
	for (int jj = 0; jj < NovelCam.size(); jj++)
	{
		fprintf(fp, "%d %d %zd\n", NovelCamId, NovelCam[jj].frameID, min(kNN, allNearestKeyFrames[jj].size()));
		for (int ii = 0; ii < min(kNN, allNearestKeyFrames[jj].size()); ii++)
		{
			fprintf(fp, "%d %d ", allNearestKeyFrames[jj][ii].x, allNearestKeyFrames[jj][ii].y);

			CameraData *NRef = &VideoInfo[allNearestKeyFrames[jj][ii].x].VideoInfo[allNearestKeyFrames[jj][ii].y];

			double R1to0[9], T1to0[3];
			GetRelativeTransformation(NovelCam[jj].R, NovelCam[jj].T, NRef[0].R, NRef[0].T, R1to0, T1to0);
			for (int jj = 0; jj < 9; jj++)
				fprintf(fp, "%.16f ", R1to0[jj]);
			for (int jj = 0; jj < 3; jj++)
				fprintf(fp, "%.16f ", T1to0[jj]);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	delete[]allNearestKeyFrames, delete[]VideoInfo, delete[]allVrFid;
	return 0;
}
int CorpusViewSelection4MVS(char *Path)
{
	//Corpus CorpusInfo;
	//char Fname[512];  sprintf(Fname, "%s/Corpus", Path);
	//ReadCorpusInfo(Fname, CorpusInfo, false, true);

	char Fname[512];
	/*for (int ll = 0; ll < 1478; ll++)
	{
		sprintf(Fname, "%s/Corpus/%.4d.png", Path, ll);
		Mat img = imread(Fname);

		int width = img.cols, height = img.rows;
		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				if (img.data[jj*width * 3 + ii * 3] == (char)0 && img.data[jj*width * 3 + ii * 3 + 1] == (char)0 && img.data[jj*width * 3 + ii * 3 + 2] == (char)0)
				{
					for (int kk = 0; kk < 3; kk++)
						img.data[jj*width * 3 + ii * 3 + kk] = rand() % 255;
				}
			}
		}
		imwrite(Fname, img);
	}*/

	int nCams = 15;
	vector<int> sCams;
	for (int ii = 0; ii < nCams; ii++)
		sCams.push_back(ii);

	Point3d CamTimeInfo[100];
	for (int ii = 0; ii < nCams; ii++)
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
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	VideoData *VideoInfo = new VideoData[nCams];
	for (auto camID : sCams)
	{
		printLOG("Cam %d ...validating ", camID);
		if (ReadVideoDataI(Path, VideoInfo[camID], camID, -1, -1) == 1)
			continue;
		else
			InvalidateAbruptCameraPose(VideoInfo[camID], -1, -1, 0);
		printLOG("\n");
	}

	double sx = 1600.0 / 1920.0, sy = 1200.0 / 1080.0;
	Mat rimg;
	for (int reffid = 800; reffid <= 800; reffid += 10)
	{
		for (auto cid : sCams)
		{
			int rfid = reffid - (int)CamTimeInfo[cid].y;
			sprintf(Fname, "%s/%d/Corrected/%.4d.png", Path, cid, rfid);
			Mat img = imread(Fname);

			resize(img, rimg, Size(1600, 1200), 0, 0, INTER_AREA);
			char Fname2[512];  sprintf(Fname2, "%s/Sync/images/%.8d.jpg", Path, cid);
			//MyCopyFile(Fname, Fname2);
			imwrite(Fname2, rimg);

			sprintf(Fname, "%s/Sync/cams/%.8d_cam.txt", Path, cid); fp = fopen(Fname, "w");
			fprintf(fp, "extrinsic\n");
			fprintf(fp, "%.8f %.8f %.8f %.4f\n", VideoInfo[cid].VideoInfo[rfid].R[0], VideoInfo[cid].VideoInfo[rfid].R[1], VideoInfo[cid].VideoInfo[rfid].R[2], VideoInfo[cid].VideoInfo[rfid].T[0]);
			fprintf(fp, "%.8f %.8f %.8f %.4f\n", VideoInfo[cid].VideoInfo[rfid].R[3], VideoInfo[cid].VideoInfo[rfid].R[4], VideoInfo[cid].VideoInfo[rfid].R[5], VideoInfo[cid].VideoInfo[rfid].T[1]);
			fprintf(fp, "%.8f %.8f %.8f %.4f\n", VideoInfo[cid].VideoInfo[rfid].R[6], VideoInfo[cid].VideoInfo[rfid].R[7], VideoInfo[cid].VideoInfo[rfid].R[8], VideoInfo[cid].VideoInfo[rfid].T[2]);
			fprintf(fp, "0.0 0.0 0.0 1.0\n");

			fprintf(fp, "intrinsic\n");
			fprintf(fp, "%.4f 0.0 %.4f\n0.0 %.4f %.4f\n0.0 0.0 1.0\n", VideoInfo[cid].VideoInfo[rfid].intrinsic[0] * sx, VideoInfo[cid].VideoInfo[rfid].intrinsic[3] * sx, VideoInfo[cid].VideoInfo[rfid].intrinsic[1] * sy, VideoInfo[cid].VideoInfo[rfid].intrinsic[4] * sy);
			fprintf(fp, "1 0.1 512\n");
			fclose(fp);
		}
	}
	return 0;
}



int main(int argc, char** argv)
{
	{
		char *Path = argv[1];

		//VisualizeAllViewsEpipolarGeometry(Path, 14, 0, 0);
		//TriangulatePointsFromArbitaryCameras(Path, 14, 1, 10, 20);
		{
			vector<int> vCams; vCams.push_back(0);
			TriangulatePointsFromNonCorpusCameras(Path, vCams, 10000, 0, 2);
		return 0;
		}

		char Fname[512], Fname2[512], buffer[512];
		myGetCurDir(512, buffer);
		sprintf(Fname, "%s/Logs", Path); makeDir(Fname);
		sprintf(Fname, "%s/Vis", Path); makeDir(Fname);
		printLOG("Current input directory: %s\n", Path);


		SfMPara mySfMPara;
		CommandParse(Path, mySfMPara);

		int nCams = mySfMPara.nCams, startF = 0, stopF = 999999;

		Corpus CorpusInfo;
		int  nCameras, nPoints, nviews, npts, useColor, viewID = 0, pid;
		cv::Point3d xyz;
		cv::Point3i rgb;

		ifstream fin;
		ofstream fout;
		vector<int> viewIDs;

		//xyz rgb viewid3D pointid3D 3dId2D cumpoint
		sprintf(Fname, "%s/Corpus/Corpus_3D.txt", Path);
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%d %d %d", &nCameras, &nPoints, &useColor);
		CorpusInfo.nCameras = nCameras;
		CorpusInfo.n3dPoints = nPoints;
		CorpusInfo.xyz.reserve(nPoints);
		CorpusInfo.rgb.reserve(nPoints);
		for (int jj = 0; jj < nPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf %d %d %d", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z);
			CorpusInfo.xyz.push_back(xyz);
			CorpusInfo.rgb.push_back(rgb);
		}
		fclose(fp);

		CorpusInfo.viewIdAll3D.reserve(nPoints);
		CorpusInfo.pointIdAll3D.reserve(nPoints);
		CorpusInfo.uvAll3D.reserve(nPoints);
		sprintf(Fname, "%s/Corpus_viewIdAll3D.dat", Path);
		fin.open(Fname, ios::binary);
		for (int jj = 0; jj < nPoints; jj++)
		{
			viewIDs.clear();
			fin.read(reinterpret_cast<char *>(&nviews), sizeof(int));
			for (int ii = 0; ii < nviews; ii++)
			{
				fin.read(reinterpret_cast<char *>(&viewID), sizeof(int));
				viewIDs.push_back(viewID);
			}
			CorpusInfo.viewIdAll3D.push_back(viewIDs);
		}
		fin.close();

		CorpusInfo.camera = new CameraData[nCameras];
		CorpusInfo.threeDIdAllViews = new vector<int>[CorpusInfo.nCameras];

		sprintf(Fname, "%s/Corpus/Corpus_threeDIdAllViews.dat", Path);
		fin.open(Fname, ios::binary);
		for (int jj = 0; jj < CorpusInfo.nCameras; jj++)
		{
			int n3D, id3D;
			fin.read(reinterpret_cast<char *>(&n3D), sizeof(int));
			CorpusInfo.threeDIdAllViews[jj].reserve(n3D);
			for (int ii = 0; ii < n3D; ii++)
			{
				fin.read(reinterpret_cast<char *>(&id3D), sizeof(int));
				CorpusInfo.threeDIdAllViews[jj].push_back(id3D);
			}
			int a = 0;
		}
		fin.close();

		//determin which video frame is used to create the corpus
		cv::Point3i Corpus_cid_Lcid_Lfid;
		vector<cv::Point3i> vCorpus_cid_Lcid_Lfid;
		sprintf(Fname, "%s/Corpus/CameraToBuildCorpus3.txt", Path);
		fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d %d ", &Corpus_cid_Lcid_Lfid.y, &Corpus_cid_Lcid_Lfid.x, &Corpus_cid_Lcid_Lfid.z) != EOF)
			vCorpus_cid_Lcid_Lfid.push_back(Corpus_cid_Lcid_Lfid);
		fclose(fp);
		//done

		//read the video frame pose
		VideoData vInfo;
		vInfo.VideoInfo = new CameraData[stopF + 1];
		int validFrame = 0;
		sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			printLOG("Cannot find %s...\n", Fname);
			return 1;
		}
		fp = fopen(Fname, "r");
		int frameID, LensType, ShutterModel, width, height;
		double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1;
		while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &ShutterModel, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
		{
			if (frameID >= startF && frameID <= stopF)
			{
				vInfo.VideoInfo[frameID].K[0] = fx, vInfo.VideoInfo[frameID].K[1] = skew, vInfo.VideoInfo[frameID].K[2] = u0,
					vInfo.VideoInfo[frameID].K[3] = 0.0, vInfo.VideoInfo[frameID].K[4] = fy, vInfo.VideoInfo[frameID].K[5] = v0,
					vInfo.VideoInfo[frameID].K[6] = 0.0, vInfo.VideoInfo[frameID].K[7] = 0.0, vInfo.VideoInfo[frameID].K[8] = 1.0;

				vInfo.VideoInfo[frameID].viewID = viewID;
				vInfo.VideoInfo[frameID].frameID = frameID;
				vInfo.VideoInfo[frameID].width = width, vInfo.VideoInfo[frameID].height = height;
				GetIntrinsicFromK(vInfo.VideoInfo[frameID]);
				mat_invert(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].invK);

				vInfo.VideoInfo[frameID].LensModel = LensType;
				vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc++;
				validFrame = frameID;
			}

			if (LensType == RADIAL_TANGENTIAL_PRISM)
			{
				fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
				if (frameID >= startF && frameID <= stopF)
				{
					vInfo.VideoInfo[frameID].distortion[0] = r0, vInfo.VideoInfo[frameID].distortion[1] = r1, vInfo.VideoInfo[frameID].distortion[2] = r2;
					vInfo.VideoInfo[frameID].distortion[3] = t0, vInfo.VideoInfo[frameID].distortion[4] = t1;
					vInfo.VideoInfo[frameID].distortion[5] = p0, vInfo.VideoInfo[frameID].distortion[6] = p1;
				}
			}
		}
		fclose(fp);

		sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			printLOG("Cannot find %s...\n", Fname);
			return 1;
		}
		fp = fopen(Fname, "r");
		double rt[6], wt[6];
		while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &frameID, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]) != EOF)
		{
			if (frameID >= startF && frameID <= stopF)
			{
				if (abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001)
				{
					vInfo.VideoInfo[frameID].valid = false;
					continue;
				}
				vInfo.VideoInfo[frameID].valid = true;

				for (int jj = 0; jj < 6; jj++)
					vInfo.VideoInfo[frameID].rt[jj] = rt[jj];

				GetRTFromrt(vInfo.VideoInfo[frameID]);
				GetCfromT(vInfo.VideoInfo[frameID]);
				AssembleP(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].R, vInfo.VideoInfo[frameID].T, vInfo.VideoInfo[frameID].P);

				double principal[] = { vInfo.VideoInfo[frameID].width / 2,vInfo.VideoInfo[frameID].height / 2, 1.0 };
				getRayDir(vInfo.VideoInfo[frameID].principleRayDir, vInfo.VideoInfo[frameID].invK, vInfo.VideoInfo[frameID].R, principal);
			}
		}
		fclose(fp);
		//done

		//for every frame, get the projection of the visble corpus points.
		//These points are taken as the points visible in the 2 nearest corpus (keyframes) frame

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

		for (int testFid = 20; testFid < 999; testFid++)
		{
			if (!vInfo.VideoInfo[testFid].valid)
			{
				printLOG("The selected frame is not localized to the Corpus\n", testFid);
				return -1;
			}

			//find the 2 nearest frame and the corpus points in those 2 nn frames
			vector<int> VisbleCorpus3DPointId;
			{
				int nn2[2] = { 0,1 };
				for (int ii = 0; ii < vCorpus_cid_Lcid_Lfid.size() - 1; ii++)
				{
					if (testFid >= vCorpus_cid_Lcid_Lfid[ii].z && testFid <= vCorpus_cid_Lcid_Lfid[ii + 1].z)
					{
						nn2[0] = vCorpus_cid_Lcid_Lfid[ii].y, nn2[1] = vCorpus_cid_Lcid_Lfid[ii + 1].y;
						break;
					}
				}

				for (int ii = 0; ii < 2; ii++)
					for (auto id3D : CorpusInfo.threeDIdAllViews[nn2[ii]])
						VisbleCorpus3DPointId.push_back(id3D);
			}
			//done

			//project those corpus points to the image
			vector<cv::Point2d> vpt2D;
			vector<double> vdepth;
			cv::Point2d pt2D;
			CameraData &camI = vInfo.VideoInfo[testFid];

			for (int ii = 0; ii < VisbleCorpus3DPointId.size(); ii++)
			{
				int id3D = VisbleCorpus3DPointId[ii];
				cv::Point3d p3d = CorpusInfo.xyz[id3D];

				//get the 2d location
				ProjectandDistort(p3d, &pt2D, camI.P, camI.K, camI.distortion);

				//get the depth
				double cam2point[3] = { p3d.x - camI.camCenter[0], p3d.y - camI.camCenter[1], p3d.z - camI.camCenter[2] };
				double depth = dotProduct(cam2point, camI.principleRayDir);

				vpt2D.emplace_back(pt2D);
				vdepth.emplace_back(depth);
			}
			//done;

			/*sprintf(Fname, "%s/0/%.4d.jpg", Path, testFid);
			cv::Mat img = imread(Fname);

			for (int pid = 0; pid < vpt2D.size(); pid++)
			{
				Point2d uv = vpt2D[pid];
				int id3D = VisbleCorpus3DPointId[pid];
				circle(img, uv, 2, colors[id3D % 8], 2);
			}
			sprintf(Fname, "%s/Vis/%.4d.jpg", Path, testFid);
			//imwrite(Fname, img);*/

			sprintf(Fname, "%s/Vis/%.4d.txt", Path, testFid);
			fp = fopen(Fname, "w");
			for (int ii = 0; ii < vdepth.size(); ii++)
				fprintf(fp, "%d %.2f %.2f %.4f\n", VisbleCorpus3DPointId[ii], vpt2D[ii].x, vpt2D[ii].y, vdepth[ii]);
			fclose(fp);
		}

		return 0;
	}

	{
		char Path[] = { "C:/temp" };
		vector<int> sCams;
		for (int ii = 0; ii < 12; ii++)
			sCams.push_back(ii);
		writeDeepMVSInputData(Path, sCams, 3000, 3000, 1, 0, 66.394845, 192, 1.0);
		return 0;
	}
#ifdef _DEBUG
	srand(0);
#else
	srand(time(NULL));
#endif
	srand(2);
	if (argc == 1)
	{
		printf("EnRecon.exe DataPath\n");
		return 0;
	}

	vector<int> vCid = { 0, 2, 4, 5, 7, 9 };
	char Fname1[] = { "X:/User/minh/CVPR16/Dance/60fps/CVPR16" }, Fname2[] = { "X:/User/minh/CVPR16/Dance/60fps" };
	Align2CameraPoseCoord(Fname1, Fname2, vCid, 600, 1400);

	char *Path = argv[1];
	char Fname[512], buffer[512];
	myGetCurDir(512, buffer);
	sprintf(Fname, "%s/Logs", Path); makeDir(Fname);
	sprintf(Fname, "%s/Vis", Path); makeDir(Fname);
	printLOG("Current input directory: %s\n", Path);

	SfMPara mySfMPara;
	CommandParse(Path, mySfMPara);

	int nCams = mySfMPara.nCams, startF = mySfMPara.startF, stopF = mySfMPara.stopF, increF = mySfMPara.increF, trackingInstF = 20;
	double biDirectFlowThresh = 20, featFlowConsistencyPercent = 0.7, descSimThesh = 0.9, descRatioThresh = 0.8, overlappingAThresh = 0.1; //interacting/occluding people sometimes confuse the pose
	double real2SfM = mySfMPara.real2SfM;

	int LensType = 0, distortionCorrected = 0, iterMax = 1000, nViewPlus = 3, nMinRanSacPoints = 3;
	double Reprojectionthreshold = 10, detectionThresh = 0.3;

	int LossType = 1, Use2DFitting = 1, selectedPeopleId = argc >= 3 ? atoi(argv[2]) : -1,
		hasDensePose = argc >= 4 ? atoi(argv[3]) : 0,
		startChunkdId = argc == 5 ? atoi(argv[4]) : 0;
	double WeightsHumanSkeleton3D[] = { 1.0, 1.0, 1.0, 0.5 },  //projecion, const limb, symmetry, temporal
		iSigmaSkeleton[] = { 1.0 / 8.0, 1.0 / 30, 1.0 / 100, 1.0 / 200 }; //2D detection, limb length (mm), velocity variration for slow moving + fast moving joints(mm/s)

	int wFrames = mySfMPara.SMPLWindow, nOverlappingFrames = mySfMPara.SMPLnOverlappingFrames;

	vector<int>sCams;
	for (int ii = 0; ii < nCams; ii++)
		sCams.push_back(ii);


	int AtLeastThree4Ransac4Ransac = 1;
	//cleanGT2Tracklet(Path);
	//convertGT2Tracklet(Path);
	//VisualizeAllViewsEpipolarGeometry(Path, sCams, startF, stopF);
	//SimultaneousTrackingAndAssocation_GT_Prop(Path, sCams, startF, stopF, Reprojectionthreshold, iterMax, AtLeastThree4Ransac4Ransac, 0.8, descSimThesh, 1.2, 50, 10, real2SfM, WeightsHumanSkeleton3D, iSigmaSkeleton, mySfMPara.SkeletonPointFormat, 0);
	//SimultaneousTrackingAndAssocation_Vis(Path, sCams, startF, stopF, Reprojectionthreshold, iterMax, AtLeastThree4Ransac4Ransac, 0.8, descSimThesh, 1.2, 50, 10, real2SfM, WeightsHumanSkeleton3D, iSigmaSkeleton, mySfMPara.SkeletonPointFormat);
	//WeightsHumanSkeleton3D[3] = 0.0;//remove smoothness for now
	//SimultaneousTrackingAndAssocation_GT_Asso(Path, sCams, startF, stopF, Reprojectionthreshold, iterMax, AtLeastThree4Ransac4Ransac, 0.8, descSimThesh, 1.2, 50, 10, real2SfM, WeightsHumanSkeleton3D, iSigmaSkeleton, mySfMPara.SkeletonPointFormat, 0);
	//SimultaneousTrackingAndAssocation(Path, sCams, startF, stopF, Reprojectionthreshold, iterMax, AtLeastThree4Ransac4Ransac, 0.8, descSimThesh, 1.2, 50, 10, real2SfM, WeightsHumanSkeleton3D, iSigmaSkeleton, mySfMPara.SkeletonPointFormat, 0);
	visualizationDriver(Path, sCams, startF, stopF, increF, true, false, false, true, false, mySfMPara.SkeletonPointFormat, mySfMPara.startF, mySfMPara.SyncedMode, mySfMPara.ShutterModel2);


	//ConstructViewGraphAndCutVideos(Path, sCams, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, mySfMPara.SkeletonPointFormat, 0, real2SfM, 300);
	//VisualizeNearestViewsBasedOnSkeleton(Path, nCams, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, 0);
	//VisualizeCutVideos(Path, sCams, startF, stopF, increF, mySfMPara.SkeletonPointFormat, selectedPeopleId);
	//PlanarMorphingDriver(Path, 0, sCam,  mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, 18);

	/*Point3d *CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	int selected, temp;
	double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
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

	for (int ii = 0; ii < nCams; ii++)
		ExtractVideoFrames(Path, ii, mySfMPara.startF, mySfMPara.stopF, mySfMPara.increF, 0, 1.0, 3, 1, CamTimeInfo[ii].y);*/

	return 0;
}
