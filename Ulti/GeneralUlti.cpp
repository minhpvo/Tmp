#include "GeneralUlti.h"
#include "../Geometry/Geometry1.h"
#include "../Geometry/Geometry2.h"
#include <Eigen/SVD>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace cv;

void mySleep(int ms)
{
#ifdef _WINDOWS
	Sleep(ms);
#else
	usleep(ms);
#endif
	return;
}

void myGetCurDir(int size, char *Path)
{
#ifdef _WINDOWS
	GetCurrentDirectory(size, Path);
#else
	getcwd(Path, size);
#endif
	return;
}
//For Minh incremnetal Sfm, which is not frequently used and not fully functional
static void flannFindPairs(const CvSeq*objectKpts, const CvSeq* objectDescriptors, const CvSeq*imageKpts, const CvSeq* imageDescriptors, vector<int>& ptpairs)
{
	int length = (int)(objectDescriptors->elem_size / sizeof(float));

	cv::Mat m_object(objectDescriptors->total, length, CV_32F);
	cv::Mat m_image(imageDescriptors->total, length, CV_32F);


	// copy descriptors
	CvSeqReader obj_reader;
	float* obj_ptr = m_object.ptr<float>(0);
	cvStartReadSeq(objectDescriptors, &obj_reader);
	for (int i = 0; i < objectDescriptors->total; i++)
	{
		const float* descriptor = (const float*)obj_reader.ptr;
		CV_NEXT_SEQ_ELEM(obj_reader.seq->elem_size, obj_reader);
		memcpy(obj_ptr, descriptor, length * sizeof(float));
		obj_ptr += length;
	}
	CvSeqReader img_reader;
	float* img_ptr = m_image.ptr<float>(0);
	cvStartReadSeq(imageDescriptors, &img_reader);
	for (int i = 0; i < imageDescriptors->total; i++)
	{
		const float* descriptor = (const float*)img_reader.ptr;
		CV_NEXT_SEQ_ELEM(img_reader.seq->elem_size, img_reader);
		memcpy(img_ptr, descriptor, length * sizeof(float));
		img_ptr += length;
	}

	// find nearest neighbors using FLANN
	cv::Mat m_indices(objectDescriptors->total, 2, CV_32S);
	cv::Mat m_dists(objectDescriptors->total, 2, CV_32F);
	cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
	flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64)); // maximum number of leafs checked

	int* indices_ptr = m_indices.ptr<int>(0);
	float* dists_ptr = m_dists.ptr<float>(0);
	for (int i = 0; i < m_indices.rows; ++i) {
		if (dists_ptr[2 * i] < 0.6*dists_ptr[2 * i + 1]) {
			ptpairs.push_back(i);
			ptpairs.push_back(indices_ptr[2 * i]);
		}
	}
}
void BestPairFinder(char *Path, int nviews, int timeID, int *viewPair)
{
	char Fname[512];
	int ii, jj;

	int *viewMatrix = new int[nviews*nviews];

	if (timeID < 0)
		sprintf(Fname, "%s/VM.txt", Path);
	else
		sprintf(Fname, "%s/VM_%.4d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < nviews; jj++)
		for (ii = 0; ii < nviews; ii++)
			fscanf(fp, "%d ", &viewMatrix[ii + jj * nviews]);
	fclose(fp);

	int bestCount = 0;
	for (jj = 0; jj < nviews; jj++)
	{
		for (ii = 0; ii < nviews; ii++)
		{
			if (viewMatrix[ii + jj * nviews] > bestCount)
			{
				bestCount = viewMatrix[ii + jj * nviews];
				viewPair[0] = ii, viewPair[1] = jj;
			}
		}
	}

	delete[]viewMatrix;

	return;
}
int NextViewFinder(char *Path, int nviews, int timeID, int currentView, int &maxPoints, vector<int> usedViews)
{
	char Fname[512];
	int ii, jj, kk;

	int *viewMatrix = new int[nviews*nviews];

	if (timeID < 0)
		sprintf(Fname, "%s/VM.txt", Path);
	else
		sprintf(Fname, "%s/VM_%.4d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < nviews; jj++) {
		for (ii = 0; ii < nviews; ii++) {
			fscanf(fp, "%d ", &viewMatrix[ii + jj * nviews]);
		}
	}
	fclose(fp);

	for (ii = 0; ii < usedViews.size(); ii++) {
		for (jj = 0; jj < usedViews.size(); jj++) {
			if (jj != ii) {
				viewMatrix[usedViews[ii] + usedViews.at(jj)*nviews] = 0, viewMatrix[usedViews.at(jj) + usedViews[ii] * nviews] = 0;
			}
		}
	}

	jj = 0;
	for (ii = 0; ii < nviews; ii++)
	{
		if (viewMatrix[ii + currentView * nviews] > jj)
		{
			jj = viewMatrix[ii + currentView * nviews];
			kk = ii;
		}
	}

	maxPoints = jj;

	delete[]viewMatrix;

	return kk;
}
int GetPoint3D2DPairCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, vector<int> viewID, Point3d *ThreeD, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&TwoDCorrespondencesID, vector<int> &ThreeDCorrespondencesID, vector<int>&SelectedIndex, bool SwapView, bool useGPU)
{
	//SelectedIndex: index of correspondenceID in the total points pool
	keypoints1.clear(), keypoints2.clear(), TwoDCorrespondencesID.clear(), ThreeDCorrespondencesID.clear();

	int ii, jj, kk, ll, id;
	char Fname[512];

	if (timeID < 0)
		sprintf(Fname, "%s/%.4d.kpts", Path, viewID.at(0));
	else
		sprintf(Fname, "%s/%d/%.4d.kpts", Path, viewID.at(0), timeID);
	ReadKPointsBinarySIFT(Fname, keypoints1);

	if (timeID < 0)
		sprintf(Fname, "%s/%.4d.kpts", Path, viewID.at(1));
	else
		sprintf(Fname, "%s/%d/%.4d.kpts", Path, viewID.at(1), timeID);
	ReadKPointsBinarySIFT(Fname, keypoints2, true);


	int totalPts = cumulativePts.at(nviews);
	vector<int>matches; matches.reserve(500);//Cannot be found in more than 500 views!
	//vector<int>CorrespondencesID;CorrespondencesID.reserve((cumulativePts.at(viewID.at(1)+1)-cumulativePts.at(viewID.at(0)+1))*2);

	if (timeID < 0)
		sprintf(Fname, "%s/PM.txt", Path);
	else
		sprintf(Fname, "%s/PM_%.4d.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < totalPts; jj++)
	{
		kk = 0; matches.clear();
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &ll);
			matches.push_back(ll);
		}

		if (jj >= cumulativePts.at(viewID.at(0)) && jj < cumulativePts.at(viewID.at(0) + 1))
		{
			for (ii = 0; ii < matches.size(); ii++)
			{
				int match = matches[ii];
				if (match >= cumulativePts.at(viewID.at(1)) && match < cumulativePts.at(viewID.at(1) + 1))
				{
					TwoDCorrespondencesID.push_back(jj - cumulativePts.at(viewID.at(0)));
					TwoDCorrespondencesID.push_back(match - cumulativePts.at(viewID.at(1)));
					SelectedIndex.push_back(jj);

					if (abs(ThreeD[jj].z) > 0.0 && !SwapView)
					{
						id = match - cumulativePts.at(viewID.at(1));
						ThreeDCorrespondencesID.push_back(id);
						ThreeDCorrespondencesID.push_back(jj);
					}
					else if (abs(ThreeD[match].z) > 0.0 && SwapView)
					{
						id = jj - cumulativePts.at(viewID.at(0));
						ThreeDCorrespondencesID.push_back(id);
						ThreeDCorrespondencesID.push_back(match);
					}
				}
			}
		}
	}
	fclose(fp);

	return 0;
}
int GetPoint3D2DAllCorrespondence(char *Path, int nviews, int timeID, vector<int> cumulativePts, Point3d *ThreeD, vector<int> availViews, vector<int>&Selected3DIndex, vector<Point2d> *selected2D, vector<int>*nSelectedViews, int &nSelectedPts, bool useGPU)
{
	//SelectedIndex: index of correspondenceID in the total points pool
	Selected3DIndex.clear();
	int ii, jj, kk, ll;
	char Fname[512];

	bool PointAdded, PointAdded2, once;
	int viewID1, viewID2, match, totalPts = cumulativePts.at(nviews);

	vector<int>matches; matches.reserve(500);//Cannot be found in more than 500 views!
	//vector<int>CorrespondencesID;CorrespondencesID.reserve((cumulativePts.at(viewsID[1]+1)-cumulativePts.at(viewsID[0]+1))*2);
	vector<int> *selected2Did = new vector<int>[totalPts];
	for (ii = 0; ii < totalPts; ii++)
		selected2Did[ii].reserve(20);

	//fill in selected3D, select3Dindex, index of 2d points in available views
	if (timeID < 0)
		sprintf(Fname, "%s/PM.txt", Path);
	else
		sprintf(Fname, "%s/PM_%.4d.txt", Path, timeID);
	FILE* fp = fopen(Fname, "r");
	nSelectedPts = 0;
	for (jj = 0; jj < totalPts; jj++)
	{
		kk = 0; matches.clear();
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &match);
			matches.push_back(match);
		}

		if (abs(ThreeD[jj].z) > 0.0 && matches.size() > 0)
		{
			once = true, PointAdded = false, PointAdded2 = false;
			for (kk = 0; kk < availViews.size(); kk++)
			{
				viewID1 = availViews.at(kk);
				if (jj >= cumulativePts.at(viewID1) && jj < cumulativePts.at(viewID1 + 1))
				{
					for (ii = 0; ii < matches.size(); ii++)
					{
						PointAdded = false;
						match = matches[ii];
						for (ll = 0; ll < availViews.size(); ll++)
						{
							if (ll == kk)
								continue;

							viewID2 = availViews.at(ll);
							if (match >= cumulativePts.at(viewID2) && match < cumulativePts.at(viewID2 + 1))
							{
								if (once)
								{
									once = false, PointAdded = true, PointAdded2 = true;
									Selected3DIndex.push_back(jj);
									nSelectedViews[nSelectedPts].clear();  nSelectedViews[nSelectedPts].push_back(viewID1);
									selected2Did[nSelectedPts].push_back(jj - cumulativePts.at(viewID1));
								}
								nSelectedViews[nSelectedPts].push_back(viewID2);
								selected2Did[nSelectedPts].push_back(match - cumulativePts.at(viewID2));
							}
							if (PointAdded)
								break;
						}
					}
				}
				if (PointAdded2)
					break;
			}
			if (PointAdded2)
				nSelectedPts++;
		}
	}
	fclose(fp);
	//fill in select2D: points seen in available views
	vector<KeyPoint> keypoints; keypoints.reserve(10000);

	for (ii = 0; ii < nSelectedPts; ii++)
	{
		int nviews = (int)nSelectedViews[ii].size();
		selected2D[ii].clear(); selected2D[ii].reserve(nviews);
		for (jj = 0; jj < nviews; jj++)
			selected2D[ii].push_back(Point2d(0, 0));
	}

	for (kk = 0; kk < availViews.size(); kk++)
	{
		int viewID = availViews.at(kk); keypoints.clear();
		if (timeID < 0)
			sprintf(Fname, "%s/%.4d.kpts", Path, viewID);
		else
			sprintf(Fname, "%s/%d/%.4d.kpts", Path, viewID, timeID);
		ReadKPointsBinarySIFT(Fname, keypoints);

		for (ll = 0; ll < nSelectedPts; ll++)
		{
			for (jj = 0; jj < nSelectedViews[ll].size(); jj++)
			{
				if (nSelectedViews[ll].at(jj) == viewID)
				{
					int poindID = selected2Did[ll].at(jj);
					selected2D[ll].at(jj).x = keypoints.at(poindID).pt.x;
					selected2D[ll].at(jj).y = keypoints.at(poindID).pt.y;
					break;
				}
			}
		}
	}

	delete[]selected2Did;
	return 0;
}

int ComputePnPInlierStats(char *Path, int nCams, int startF, int stopF)
{
	char Fname[512];

	int threeDid;
	double x, y, z, u, v, s;

	for (int cid = 0; cid < nCams; cid++)
	{
		vector<double> per;
		for (int fid = startF; fid <= stopF; fid++)
		{
			int before = 0, after = 0;
			sprintf(Fname, "%s/%d/PnPmTc/Inliers_%.4d.txt", Path, cid, fid);
			if (IsFileExist(Fname) == 0)
				continue;
			FILE *fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &threeDid, &x, &y, &z, &u, &v, &s) != EOF)
				before++;
			fclose(fp);

			sprintf(Fname, "%s/%d/PnP/Inliers_%.4d.txt", Path, cid, fid);
			if (IsFileExist(Fname) == 0)
				continue;
			fp = fopen(Fname, "r");
			while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &threeDid, &x, &y, &z, &u, &v, &s) != EOF)
			{
				after++;
				if (after > 50000)
					printLOG("Problem with %s\n", Fname);
			}
			fclose(fp);

			per.push_back(1.f*after / before);
		}
		printLOG("Cam %d: %.2f\n", cid, MeanArray(per));
	}

	return 0;
}
//Read correspondences and build matching (visbiblity) matrix for further sfm
int ReadCumulativePoints(char *Path, int nviews, int timeID, vector<int>&cumulativePts)
{
	int ii, jj;
	char Fname[512];
	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/CumlativePoints.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s. Abort program!\n", Fname);
		return 1;
	}
	for (ii = 0; ii < nviews + 1; ii++)
	{
		fscanf(fp, "%d\n", &jj);
		cumulativePts.push_back(jj);
	}
	fclose(fp);

	return 0;
}
void ReadCumulativePointsVisualSfm(char *Path, int nviews, vector<int>&cumulativePts)
{
	char Fname[512];
	int dummy, npts, currentNpts = 0;
	cumulativePts.push_back(currentNpts);
	for (int ii = 0; ii < nviews; ii++)
	{
		sprintf(Fname, "%s/%.4d.sift", Path, ii);
		ifstream fin; fin.open(Fname, ios::binary);
		if (!fin.is_open())
		{
			cout << "Cannot open: " << Fname << endl;
			cumulativePts.push_back(currentNpts);
			continue;
		}

		fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
		fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
		fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
		fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5numbers
		fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

		fin.close();

		currentNpts += npts;
		cumulativePts.push_back(currentNpts);
	}

	sprintf(Fname, "%s/CumlativePoints.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < nviews + 1; ii++)
		fprintf(fp, "%d\n", cumulativePts[ii]);
	fclose(fp);

	return;
}

void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, vector<int>&CeresDuplicateAddInMask, int totalPts, bool Merge)
{
	int ii, jj, kk, match;
	char Fname[512];

	for (ii = 0; ii < totalPts; ii++)
		PointCorres[ii].reserve(nviews * 2);

	if (!Merge)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/Corpus/PM.txt", Path);
		else
			sprintf(Fname, "%s/PM_%ds.txt", Path, timeID);
	}
	else
		if (timeID < 0)
			sprintf(Fname, "%s/Corpus/MPM.txt", Path);
		else
			sprintf(Fname, "%s/MPM_%ds.txt", Path, timeID);

	CeresDuplicateAddInMask.reserve(totalPts * 30);
	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < totalPts; jj++)
	{
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &match);
			PointCorres[jj].push_back(match);
			CeresDuplicateAddInMask.push_back(match);
		}
	}
	return;
}
void ReadPointCorrespondences(char *Path, int nviews, int timeID, vector<int> *PointCorres, int totalPts, bool Merge)
{
	int ii, jj, kk, match;
	char Fname[512];

	for (ii = 0; ii < totalPts; ii++)
		PointCorres[ii].reserve(nviews * 2);

	if (!Merge)
	{
		if (timeID < 0)
			sprintf(Fname, "%s/notMergePM.txt", Path);
		else
			sprintf(Fname, "%s/notMergePM_%.4d.txt", Path, timeID);
	}
	else
		if (timeID < 0)
			sprintf(Fname, "%s/MPM.txt", Path);
		else
			sprintf(Fname, "%s/MPM_%.4d.txt", Path, timeID);

	FILE *fp = fopen(Fname, "r");
	for (jj = 0; jj < totalPts; jj++)
	{
		fscanf(fp, "%d ", &kk);
		for (ii = 0; ii < kk; ii++)
		{
			fscanf(fp, "%d ", &match);
			PointCorres[jj].push_back(match);
		}
	}
	return;
}
void GenerateMergePointCorrespondences(vector<int> *MergePointCorres, vector<int> *PointCorres, int totalPts)
{
	//Merging
	for (int kk = 0; kk < totalPts; kk++)
	{
		int nmatches = (int)PointCorres[kk].size();
		if (nmatches > 0) //if that point has matches
		{
			for (int jj = 0; jj < kk; jj++) //look back to previous point
			{
				for (int ii = 0; ii < PointCorres[jj].size(); ii++) //look into all of that previous point matches
				{
					if (PointCorres[jj][ii] == kk) //if it has the same ID as the current point-->merge points
					{
						//printLOG("Merging %d (%d matches) to %d (%d matches)\n", kk, PointCorres[kk].size(), jj, PointCorres[jj].size());
						for (int i = 0; i < PointCorres[kk].size(); i++)
							PointCorres[jj].push_back(PointCorres[kk].at(i));
						PointCorres[kk].clear();//earse matches of the currrent point
						break;
					}
				}
			}
		}
	}

	//Removing duplicated points and sort them
	for (int kk = 0; kk < totalPts; kk++)
	{
		std::sort(PointCorres[kk].begin(), PointCorres[kk].end());
		for (int jj = 0; jj < PointCorres[kk].size(); jj++)
		{
			if (jj == 0)
				MergePointCorres[kk].push_back(PointCorres[kk].at(0));
			else if (jj > 0 && PointCorres[kk][jj] != PointCorres[kk].at(jj - 1))
				MergePointCorres[kk].push_back(PointCorres[kk][jj]);
		}
	}
	return;
}
void GenerateViewandPointCorrespondences(vector<int> *ViewCorres, vector<int> *PointIDCorres, vector<int> *PointCorres, vector<int> CumIDView, int totalPts)
{
	int viewID, PointID, curPID;
	for (int jj = 0; jj < totalPts; jj++)
	{
		for (int ii = 0; ii < PointCorres[jj].size(); ii++)
		{
			curPID = PointCorres[jj][ii];
			for (int j = 0; j < CumIDView.size() - 1; j++)
			{
				if (curPID >= CumIDView.at(j) && curPID < CumIDView.at(j + 1))
				{
					viewID = j;
					PointID = curPID - CumIDView.at(j);
					break;
				}
			}
			ViewCorres[jj].push_back(viewID);
			PointIDCorres[jj].push_back(PointID);
		}
	}

	return;
}

void GenerateMatchingTable(char *Path, int nviews, int timeID, int nViewsPlus)
{
	char Fname[512];

	int totalPts;
	vector<int> cumulativePts;
	ReadCumulativePoints(Path, nviews, timeID, cumulativePts);
	totalPts = cumulativePts.at(nviews);

	vector<Point2i> *AllPairWiseMatchingId = new vector<Point2i>[nviews*(nviews - 1) / 2];
	for (int ii = 0; ii < nviews*(nviews - 1) / 2; ii++)
		AllPairWiseMatchingId[ii].reserve(10000);

	int percent = 10, incre = 10;
	int nfiles = nviews * (nviews - 1) / 2, filesCount = 0;
	double start = omp_get_wtime();
	for (int jj = 0; jj < nviews - 1; jj++)
	{
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			if (100.0*filesCount / nfiles >= percent)
			{
				printLOG("@\r# %.2f%% (%.2fs) Reading pairwise matches....", 100.0*filesCount / nfiles, omp_get_wtime() - start);
				percent += incre;
			}
			filesCount++;
			if (timeID < 0)
				sprintf(Fname, "%s/Dynamic/M_%.2d_%.2d.txt", Path, jj, ii);
			else
				sprintf(Fname, "%s/Dynamic/%.4d/M_%.2d_%.2d.txt", Path, timeID, jj, ii);

			int count = ii - jj - 1;
			for (int i = 0; i <= jj - 1; i++)
				count += nviews - i - 1;

			int id1, id2, npts;
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
				continue;
			fscanf(fp, "%d ", &npts);
			AllPairWiseMatchingId[count].reserve(npts);
			while (fscanf(fp, "%d %d ", &id1, &id2) != EOF)
				AllPairWiseMatchingId[count].push_back(Point2i(id1, id2));
			fclose(fp);
		}
	}

	//Generate Visbible Points Table
	vector<int> *KeysBelongTo3DPoint = new vector <int>[nviews];
	for (int jj = 0; jj < nviews; jj++)
	{
		KeysBelongTo3DPoint[jj].reserve(cumulativePts[jj + 1] - cumulativePts[jj]);
		for (int ii = 0; ii < cumulativePts[jj + 1] - cumulativePts[jj]; ii++)
			KeysBelongTo3DPoint[jj].push_back(-1);
	}

	vector<int>*ViewMatch = new vector<int>[totalPts]; //cotains all visible views of 1 3D point
	vector<int>*PointIDMatch = new vector<int>[totalPts];//cotains all keyID of the visible views of 1 3D point
	int count3D = 0;

	for (int jj = 0; jj < nviews; jj++)
	{
		for (int ii = jj + 1; ii < nviews; ii++)
		{
			int PairWiseID = ii - jj - 1;
			for (int i = 0; i <= jj - 1; i++)
				PairWiseID += nviews - i - 1;
			//printLOG("@(%d, %d) with %d 3+ points ...TE: %.2fs\n ", jj, ii, count3D, omp_get_wtime() - start);
			for (int kk = 0; kk < AllPairWiseMatchingId[PairWiseID].size(); kk++)
			{
				int id1 = AllPairWiseMatchingId[PairWiseID].at(kk).x;
				int id2 = AllPairWiseMatchingId[PairWiseID].at(kk).y;
				int ID3D1 = KeysBelongTo3DPoint[jj][id1], ID3D2 = KeysBelongTo3DPoint[ii][id2];
				if (ID3D1 == -1 && ID3D2 == -1) //Both are never seeen before
				{
					ViewMatch[count3D].push_back(jj), ViewMatch[count3D].push_back(ii);
					PointIDMatch[count3D].push_back(id1), PointIDMatch[count3D].push_back(id2);
					KeysBelongTo3DPoint[jj][id1] = count3D, KeysBelongTo3DPoint[ii][id2] = count3D; //this pair of corres constitutes 3D point #count
					count3D++;
				}
				else if (ID3D1 == -1 && ID3D2 != -1)
				{
					ViewMatch[ID3D2].push_back(jj);
					PointIDMatch[ID3D2].push_back(id1);
					KeysBelongTo3DPoint[jj][id1] = ID3D2; //this point constitutes 3D point #ID3D2
				}
				else if (ID3D1 != -1 && ID3D2 == -1)
				{
					ViewMatch[ID3D1].push_back(ii);
					PointIDMatch[ID3D1].push_back(id2);
					KeysBelongTo3DPoint[ii][id2] = ID3D1; //this point constitutes 3D point #ID3D2
				}
				else if (ID3D1 != -1 && ID3D2 != -1 && ID3D1 != ID3D2)//Strange case where 1 point (usually not vey discrimitive or repeating points) is matched to multiple points in the same view pair 
					//--> Just concatanate the one with fewer points to largrer one and hope MultiTriangulationRansac can do sth.
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
	printLOG("Merged correspondences in %.2fs\n ", omp_get_wtime() - start);

	int count = 0, maxmatches = 0, npts = 0;
	if (timeID < 0)
		sprintf(Fname, "%s/ViewPM.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/ViewPM.txt", Path, timeID);
	FILE *fp = fopen(Fname, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = (int)ViewMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			npts++;
			if (nmatches > nViewsPlus)
				count++;
			if (nmatches > maxmatches)
				maxmatches = nmatches;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", ViewMatch[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printLOG("#%d+ points: (%d/%d). Max #matches views: \n", nViewsPlus, count, npts, maxmatches);


	if (timeID < 0)
		sprintf(Fname, "%s/IDPM.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/IDPM.txt", Path, timeID);
	fp = fopen(Fname, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = (int)PointIDMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", PointIDMatch[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printLOG("Finished generateing point correspondence matrix\n");

	delete[]ViewMatch;
	delete[]PointIDMatch;

	return;
}
void GenerateMatchingTableVisualSfM(char *Path, int nviews)
{
	char buf[512], file1[512], file2[512];

	int id, viewID1, viewID2, nmatch, totalPts;
	vector<int> cumulativePts;
	ReadCumulativePointsVisualSfm(Path, nviews, cumulativePts);
	totalPts = cumulativePts.at(nviews);

	//Generate Visbible Points Table
	vector<int> *KeysBelongTo3DPoint = new vector <int>[nviews];
	for (int jj = 0; jj < nviews; jj++)
	{
		KeysBelongTo3DPoint[jj].reserve(cumulativePts[jj + 1] - cumulativePts[jj]);
		for (int ii = 0; ii < cumulativePts[jj + 1] - cumulativePts[jj]; ii++)
			KeysBelongTo3DPoint[jj].push_back(-1);
	}

	int count3D = 0;
	vector<int>*ViewMatch = new vector<int>[totalPts]; //cotains all visible views of 1 3D point
	vector<int>*PointIDMatch = new vector<int>[totalPts];//cotains all keyID of the visible views of 1 3D point

	//Read visualsfm matches
	double start = omp_get_wtime();
	sprintf(buf, "%s/VSfMmatch.txt", Path); FILE *fp = fopen(buf, "r");
	if (fp == NULL)
	{
		printLOG("Cannot read %s\n", buf);
		abort();
	}
	vector<int>mid1, mid2;
	fscanf(fp, "%s ", buf); 	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);
	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);
	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);
	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);	fscanf(fp, "%s ", buf);
	while (fscanf(fp, "%s %s", file1, file2) != EOF)
	{
		mid1.clear(), mid2.clear();
		string  filename1 = string(file1), filename2 = string(file2);

		std::size_t posDot = filename1.find(".");
		std::size_t posCorpus = filename1.find("Corpus");
		string subs = filename1.substr(posCorpus + 7, posDot - posCorpus - 7);
		const char * str = subs.c_str();
		viewID1 = atoi(str);

		posDot = filename2.find(".");
		posCorpus = filename1.find("Corpus");
		subs = filename2.substr(posCorpus + 7, posDot - posCorpus - 7);
		str = subs.c_str();
		viewID2 = atoi(str);

		fscanf(fp, "%d ", &nmatch);
		for (int ii = 0; ii < nmatch; ii++)
		{
			fscanf(fp, "%d ", &id);
			mid1.push_back(id);
		}
		for (int ii = 0; ii < nmatch; ii++)
		{
			fscanf(fp, "%d ", &id);
			mid2.push_back(id);
		}

		for (int kk = 0; kk < nmatch; kk++)
		{
			int id1 = mid1[kk], id2 = mid2[kk];
			int ID3D1 = KeysBelongTo3DPoint[viewID1][id1], ID3D2 = KeysBelongTo3DPoint[viewID2][id2];
			if (ID3D1 == -1 && ID3D2 == -1) //Both are never seeen before
			{
				ViewMatch[count3D].push_back(viewID1), ViewMatch[count3D].push_back(viewID2);
				PointIDMatch[count3D].push_back(id1), PointIDMatch[count3D].push_back(id2);
				KeysBelongTo3DPoint[viewID1][id1] = count3D, KeysBelongTo3DPoint[viewID2][id2] = count3D; //this pair of corres constitutes 3D point #count
				count3D++;
			}
			else if (ID3D1 == -1 && ID3D2 != -1)
			{
				ViewMatch[ID3D2].push_back(viewID1);
				PointIDMatch[ID3D2].push_back(id1);
				KeysBelongTo3DPoint[viewID1][id1] = ID3D2; //this point constitutes 3D point #ID3D2
			}
			else if (ID3D1 != -1 && ID3D2 == -1)
			{
				ViewMatch[ID3D1].push_back(viewID2);
				PointIDMatch[ID3D1].push_back(id2);
				KeysBelongTo3DPoint[viewID2][id2] = ID3D1; //this point constitutes 3D point #ID3D2
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
	printLOG("Merged correspondences in %.2fs\n ", omp_get_wtime() - start);

	int count = 0, maxmatches = 0, npts = 0;
	sprintf(buf, "%s/ViewPM.txt", Path); fp = fopen(buf, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = (int)ViewMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			npts++;
			if (nmatches > 2)
				count++;
			if (nmatches > maxmatches)
				maxmatches = nmatches;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", ViewMatch[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	printLOG("#3+ points: %.4d. Max #matches views:  %.4d. #matches point: %d\n", count, maxmatches, npts);

	sprintf(buf, "%s/IDPM.txt", Path); fp = fopen(buf, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = (int)PointIDMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d ", PointIDMatch[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	count = 0, maxmatches = 0, npts = 0;
	sprintf(buf, "%s/View_ID_PM.txt", Path); fp = fopen(buf, "w+");
	if (fp != NULL)
	{
		for (int jj = 0; jj < count3D; jj++)
		{
			int nmatches = (int)ViewMatch[jj].size();
			if (nmatches < 2 || nmatches > nviews * 2)
				continue;

			npts++;
			if (nmatches > 2)
				count++;
			if (nmatches > maxmatches)
				maxmatches = nmatches;

			fprintf(fp, "%d ", nmatches);
			for (int ii = 0; ii < nmatches; ii++)
				fprintf(fp, "%d %d ", ViewMatch[jj][ii], PointIDMatch[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	delete[]ViewMatch;
	delete[]PointIDMatch;

	return;
}
void GenerateViewCorrespondenceMatrix(char *Path, int nviews, int timeID)
{
	int ii, jj, kk, ll, mm, nn;
	char Fname[512];

	vector<int> cumulativePts, PtsView;
	if (timeID < 0)
		sprintf(Fname, "%s/CumlativePoints.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/CumlativePoints.txt", Path, timeID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s", Fname);
		return;
	}
	for (ii = 0; ii < nviews + 1; ii++)
	{
		fscanf(fp, "%d\n", &jj);
		cumulativePts.push_back(jj);
	}
	fclose(fp);

	Mat viewMatrix(nviews, nviews, CV_32S);
	viewMatrix = Scalar::all(0);

	vector<int>matches; matches.reserve(nviews * 2);
	for (mm = 0; mm < nviews - 1; mm++)
	{
		for (nn = mm + 1; nn < nviews; nn++)
		{
			int totalPts = cumulativePts.at(nviews);

			int count = 0;
			if (timeID < 0)
				sprintf(Fname, "%s/PM.txt", Path);
			else
				sprintf(Fname, "%s/Dynamic/PM_%.4d.txt", Path, timeID);
			fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("Cannot open %s", Fname);
				return;
			}
			for (jj = 0; jj < totalPts; jj++)
			{
				kk = 0; matches.clear();
				fscanf(fp, "%d ", &kk);
				for (ii = 0; ii < kk; ii++)
				{
					fscanf(fp, "%d ", &ll);
					matches.push_back(ll);
				}

				if (jj >= cumulativePts.at(mm) && jj < cumulativePts.at(mm + 1))
				{
					for (ii = 0; ii < kk; ii++)
					{
						int match = matches[ii];
						if (match >= cumulativePts.at(nn) && match < cumulativePts.at(nn + 1))
							viewMatrix.at<int>(mm + nn * nviews) += 1;
					}
				}
			}
			fclose(fp);

		}
	}
	completeSymm(viewMatrix, true);

	if (timeID < 0)
		sprintf(Fname, "%s/VM.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/VM_%.4d.txt", Path, timeID);
	fp = fopen(Fname, "w+");
	for (jj = 0; jj < nviews; jj++)
	{
		for (ii = 0; ii < nviews; ii++)
			fprintf(fp, "%d ", viewMatrix.at<int>(ii + jj * nviews));
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
int GetPutativeMatchesForEachView(char *Path, int nviews, vector<int> TrackingInst, Point2d ScaleThresh, int nViewPlus, int *frameTimeStamp)
{
	char Fname[512];
	if (frameTimeStamp == NULL)
	{
		frameTimeStamp = new int[nviews];
		for (int ii = 0; ii < nviews; ii++)
			frameTimeStamp[ii] = 0;
	}
	vector<Point2i> matches;

	int totalPts, MAXPTS = 0;
	for (int inst = 0; inst < (int)TrackingInst.size(); inst++)
	{
		int fid = TrackingInst[inst];
		sprintf(Fname, "%s/Dynamic/%.4d/ViewPM.txt", Path, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		int nviewsi, viewi, n3D = 0;
		while (fscanf(fp, "%d ", &nviewsi) != EOF)
		{
			for (int ii = 0; ii < nviewsi; ii++)
				fscanf(fp, "%d ", &viewi);
			n3D++;
		}
		fclose(fp);
		matches.push_back(Point2i(fid, n3D));
		if (n3D > MAXPTS)
			MAXPTS = n3D;
	}

	vector<int> cumulativePts; cumulativePts.reserve(nviews);
	vector<int>*PViewIdAll3D = new vector<int>[MAXPTS];
	vector<int>*PuvIdAll3D = new vector<int>[MAXPTS];
	vector<KeyPoint> AllKeys;

	FILE *fp1 = 0, *fp2 = 0, *fp3 = 0, *fp4 = 0, *fp5 = 0;
	for (int inst = 0; inst < (int)TrackingInst.size(); inst++)
	{
		int fid = TrackingInst[inst];
		printLOG("Get putative matches for frame %d ...\n", fid);
		cumulativePts.clear();
		if (ReadCumulativePoints(Path, nviews, fid, cumulativePts) == 1)
			continue;
		totalPts = cumulativePts.at(nviews);

		sprintf(Fname, "%s/Dynamic/%.4d/ViewPM.txt", Path, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		int nviewsi, viewi, n3D = 0;
		while (fscanf(fp, "%d ", &nviewsi) != EOF)
		{
			PViewIdAll3D[n3D].clear(), PViewIdAll3D[n3D].reserve(nviewsi);
			for (int ii = 0; ii < nviewsi; ii++)
			{
				fscanf(fp, "%d ", &viewi);
				if (nviewsi >= nViewPlus)
					PViewIdAll3D[n3D].push_back(viewi);
			}
			if (nviewsi >= nViewPlus)
				n3D++;
		}
		fclose(fp);

		sprintf(Fname, "%s/Dynamic/%.4d/IDPM.txt", Path, fid); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		int np, pi;
		n3D = 0;
		while (fscanf(fp, "%d ", &np) != EOF)
		{
			PuvIdAll3D[n3D].clear(), PuvIdAll3D[n3D].reserve(np);
			for (int ii = 0; ii < np; ii++)
			{
				fscanf(fp, "%d ", &pi);
				if (np >= nViewPlus)
					PuvIdAll3D[n3D].push_back(pi);
			}
			if (np >= nViewPlus)
				n3D++;
		}
		fclose(fp);

		//Write all matched sift points
		for (int vid = 0; vid < nviews; vid++)
		{
			AllKeys.clear();
			sprintf(Fname, "%s/%d/%.4d.sift", Path, vid, fid - frameTimeStamp[vid]);
			readVisualSFMSiftGPU(Fname, AllKeys);

			sprintf(Fname, "%s/Dynamic/K_%d_%.4d.txt", Path, vid, fid); fp = fopen(Fname, "w+");
			for (int pid = 0; pid < n3D; pid++)
				for (int ii = 0; ii < PViewIdAll3D[pid].size(); ii++)
					if (PViewIdAll3D[pid][ii] == vid && AllKeys[PuvIdAll3D[pid][ii]].size > ScaleThresh.x && AllKeys[PuvIdAll3D[pid][ii]].size < ScaleThresh.y)
						fprintf(fp, "%d %d %.4f %.4f %.3f %.4f\n", pid, fid - frameTimeStamp[vid], AllKeys[PuvIdAll3D[pid][ii]].pt.x, AllKeys[PuvIdAll3D[pid][ii]].pt.y, AllKeys[PuvIdAll3D[pid][ii]].size, AllKeys[PuvIdAll3D[pid][ii]].angle);
			fclose(fp);
		}
	}

	sprintf(Fname, "%s/Dynamic/nMatches.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < (int)matches.size(); ii++)
		fprintf(fp, "%d %d\n", matches[ii].x, matches[ii].y);
	fclose(fp);

	delete[]PViewIdAll3D, delete[]PuvIdAll3D;
	return 0;
}

void RotY(double *Rmat, double theta)
{
	double c = cos(theta), s = sin(theta);
	Rmat[0] = c, Rmat[1] = 0.0, Rmat[2] = s;
	Rmat[3] = 0, Rmat[4] = 1.0, Rmat[5] = 0;
	Rmat[6] = -s, Rmat[7] = 0.0, Rmat[8] = c;
	return;
}
void RotZ(double *Rmat, double theta)
{
	double c = cos(theta), s = sin(theta);
	Rmat[0] = c, Rmat[1] = -s, Rmat[2] = 0;
	Rmat[3] = s, Rmat[4] = c, Rmat[5] = 0;
	Rmat[6] = 0, Rmat[7] = 0.0, Rmat[8] = 1;
	return;
}
void GenerateCamerasExtrinsicOnCircle(CameraData &CameraInfo, double theta, double radius, Point3d center, Point3d LookAtTarget, Point3d Noise3D)
{
	//Adapted from Jack's code
	double Rmat[9], RmatT[9];
	RotY(Rmat, theta); mat_transpose(Rmat, RmatT, 3, 3);
	double CameraPosition[3] = { -(RmatT[0] * 0.0 + RmatT[1] * 0.0 + RmatT[2] * radius) + center.x + Noise3D.x,
		-(RmatT[3] * 0.0 + RmatT[4] * 0.0 + RmatT[5] * radius) + center.y + Noise3D.y,
		-(RmatT[6] * 0.0 + RmatT[7] * 0.0 + RmatT[8] * radius) + center.z + Noise3D.z };

	//Look at vector
	double k[3] = { LookAtTarget.x - CameraPosition[0], LookAtTarget.y - CameraPosition[1], LookAtTarget.z - CameraPosition[2] };
	normalize(k, 3);

	//Up vector
	double j[3] = { 0.0, 1.0, 0.0 };

	//Sideway vector
	double i[3]; cross_product(j, k, i);

	//Camera rotation matrix
	CameraInfo.R[0] = i[0], CameraInfo.R[1] = i[1], CameraInfo.R[2] = i[2];
	CameraInfo.R[3] = j[0], CameraInfo.R[4] = j[1], CameraInfo.R[5] = j[2];
	CameraInfo.R[6] = k[0], CameraInfo.R[7] = k[1], CameraInfo.R[8] = k[2];

	//Translation vector
	mat_mul(CameraInfo.R, CameraPosition, CameraInfo.T, 3, 3, 1);
	CameraInfo.T[0] = -CameraInfo.T[0], CameraInfo.T[1] = -CameraInfo.T[1], CameraInfo.T[2] = -CameraInfo.T[2];
	CameraInfo.camCenter[0] = CameraPosition[0], CameraInfo.camCenter[1] = CameraPosition[1], CameraInfo.camCenter[2] = CameraPosition[2];
	return;
}
double computeProcrustesTransform(vector<Point3d> & src, vector<Point3d>& dst, double *R, double *T, double &scale, bool includeScaling)
{
	int npts = (int)src.size();

	Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);
	for (int i = 0; i < npts; ++i)
	{
		center_src[0] += src[i].x, center_src[1] += src[i].y, center_src[2] += src[i].z;
		center_dst[0] += dst[i].x, center_dst[1] += dst[i].y, center_dst[2] += dst[i].z;
	}
	center_src /= (double)npts, center_dst /= (double)npts;

	MatrixXdr S(npts, 3), D(npts, 3);
	for (int i = 0; i < npts; ++i)
	{
		S(i, 0) = src[i].x - center_src[0], S(i, 1) = src[i].y - center_src[1], S(i, 2) = src[i].z - center_src[2];
		D(i, 0) = dst[i].x - center_dst[0], D(i, 1) = dst[i].y - center_dst[1], D(i, 2) = dst[i].z - center_dst[2];
	}

	//scale
	scale = 1.0;
	if (includeScaling)
	{
		scale = (D.norm() / D.rows()) / (S.norm() / S.rows()); //stdD/stdS
		S = scale * S;
	}

	//rotation
	Eigen::Matrix3d C = D.transpose()*S;
	JacobiSVD<MatrixXd> C_svd;
	C_svd.compute(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
	if (!C_svd.computeU() || !C_svd.computeV())
		return DBL_MAX;

	Map < Matrix < double, 3, 3, RowMajor > > eR(R, 3, 3);
	Map < Vector3d > eT(T, 3);
	eR = C_svd.matrixU()*C_svd.matrixV().transpose();
	if (eR.determinant() < 0) // Check for reflection
	{
		auto W = C_svd.matrixV().eval(); 	// Annoyingly the .eval() is necessary
		W.col(C_svd.matrixV().cols() - 1) *= -1.;
		eR = C_svd.matrixU()*W.transpose();
	}

	//translation
	eT = center_dst - scale * eR*center_src;

	double error = 0.0;
	for (int i = 0; i < npts; ++i)
	{
		double orgX = src[i].x, orgY = src[i].y, orgZ = src[i].z, fX = dst[i].x, fY = dst[i].y, fZ = dst[i].z;
		double tX = scale * (eR(0, 0) * src[i].x + eR(0, 1) * src[i].y + eR(0, 2) * src[i].z) + eT(0);
		double tY = scale * (eR(1, 0) * src[i].x + eR(1, 1) * src[i].y + eR(1, 2) * src[i].z) + eT(1);
		double tZ = scale * (eR(2, 0) * src[i].x + eR(2, 1) * src[i].y + eR(2, 2) * src[i].z) + eT(2);

		double ex = tX - fX;
		double ey = tY - fY;
		double ez = tZ - fZ;
		error += sqrt(ex*ex + ey * ey + ez * ex);
	}

	return error / npts;
}

void InvalidateAbruptCameraPose(VideoData &VideoI, int startF, int stopF, int silent)
{
	if (startF == -1 && stopF == -1)
		startF = 0, stopF = 50000;

	vector<double> vCenterDif;
	for (int fid = startF; fid < stopF; fid++)
	{
		if (VideoI.VideoInfo[fid].valid && VideoI.VideoInfo[fid + 1].valid)
		{
			vCenterDif.push_back(norm(Point3d(VideoI.VideoInfo[fid].camCenter[0] - VideoI.VideoInfo[fid + 1].camCenter[0], VideoI.VideoInfo[fid].camCenter[1] - VideoI.VideoInfo[fid + 1].camCenter[1], VideoI.VideoInfo[fid].camCenter[2] - VideoI.VideoInfo[fid + 1].camCenter[2])));
		}
	}

	if (vCenterDif.size() < 2)
	{
		printLOG("Camera has less than 2 valid frames\n");
		return;
	}

	sort(vCenterDif.begin(), vCenterDif.end());
	double thresh = vCenterDif[vCenterDif.size() * 98 / 100];

	int firstTime = 1, lastValid = startF, ndel = 0;
	for (int fid = startF; fid < stopF; fid++)
	{
		if (firstTime == 1)
		{
			if (!VideoI.VideoInfo[lastValid].valid)
			{
				lastValid++;
				continue;
			}
			else
				firstTime = 0;
		}
		if (VideoI.VideoInfo[lastValid].valid && VideoI.VideoInfo[fid + 1].valid)
		{
			double dif = norm(Point3d(VideoI.VideoInfo[lastValid].camCenter[0] - VideoI.VideoInfo[fid + 1].camCenter[0],
				VideoI.VideoInfo[lastValid].camCenter[1] - VideoI.VideoInfo[fid + 1].camCenter[1],
				VideoI.VideoInfo[lastValid].camCenter[2] - VideoI.VideoInfo[fid + 1].camCenter[2]));
			if (dif <= 2.0*thresh*(fid + 1 - lastValid))
				lastValid = fid + 1;
			else
			{
				VideoI.VideoInfo[fid + 1].valid = 0, ndel++;
				if (silent == 0)
					printLOG("%d..", fid + 1);
			}
		}
	}
	if (ndel > 0 && silent == 0)
		printLOG("\n\n");

	return;
}
//post processing of 2d trajectory
int Track2DConverter(char *Path, int viewID, int startF)
{
	int npts;
	char Fname[512];

	sprintf(Fname, "%s/cTrack2D/FT_%d_%.4d.txt", Path, viewID, startF); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		return 1;
	fscanf(fp, "%d ", &npts);

	int *TruePid = new int[npts];
	vector<int> *FID = new vector<int>[npts];
	vector<Point2f> *TrackUV = new vector<Point2f>[npts];
	vector<float> *Scale = new vector<float>[npts];
	vector<float> *Angle = new vector<float>[npts];
	vector<AffinePara> *cWarp = new vector<AffinePara>[npts];

	for (int ii = 0; ii < npts; ii++)
		TruePid[ii] = -1, FID[ii].reserve(300), TrackUV[ii].reserve(300), Scale[ii].reserve(300), Angle[ii].reserve(300), cWarp[ii].reserve(300);

	int pid, fid, nf;
	float u, v, s, a;
	AffinePara wp;
	int pcount = 0;
	while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
	{
		int spid = -1;
		for (int ii = 0; ii < npts && spid == -1; ii++)
			if (TruePid[ii] == pid)
				spid = ii;
		if (spid == -1)
		{
			spid = pcount;
			TruePid[pcount] = pid;
			pcount++;
		}

		for (int ii = 0; ii < nf; ii++)
		{
			fscanf(fp, "%d %f %f %f %f %lf %lf %lf %lf ", &fid, &u, &v, &s, &a, &wp.warp[0], &wp.warp[1], &wp.warp[2], &wp.warp[3]);
			FID[spid].push_back(fid), TrackUV[spid].push_back(Point2f(u, v)), Scale[spid].push_back(s), Angle[spid].push_back(a), cWarp[spid].push_back(wp);
		}
	}
	fclose(fp);

	sprintf(Fname, "%s/cTrack2D/BT_%d_%.4d.txt", Path, viewID, startF); fp = fopen(Fname, "r");
	fscanf(fp, "%d ", &npts);
	while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
	{
		int spid = -1;
		for (int ii = 0; ii < npts && spid == -1; ii++)
			if (TruePid[ii] == pid)
				spid = ii;
		if (spid == -1)
		{
			spid = pcount;
			TruePid[pcount] = pid;
			pcount++;
		}

		for (int ii = 0; ii < nf; ii++)
		{
			fscanf(fp, "%d %f %f %f %f %lf %lf %lf %lf ", &fid, &u, &v, &s, &a, &wp.warp[0], &wp.warp[1], &wp.warp[2], &wp.warp[3]);
			FID[spid].push_back(fid), TrackUV[spid].push_back(Point2f(u, v)), Scale[spid].push_back(s), Angle[spid].push_back(a), cWarp[spid].push_back(wp);
		}
	}
	fclose(fp);

	sprintf(Fname, "%s/Track2D/%d_%.4d.txt", Path, viewID, startF); fp = fopen(Fname, "w+");
	fprintf(fp, "%d\n", npts);
	for (int jj = 0; jj < npts; jj++)
	{
		if ((int)FID[jj].size() < 1)
			continue;
		else
		{
			fprintf(fp, "%d %d ", TruePid[jj], (int)FID[jj].size());
			for (int ii = 0; ii < (int)TrackUV[jj].size(); ii++)
				fprintf(fp, "%d %.4f %.4f %.3f %.4f %.8f %.8f %.8f %.8f ", FID[jj][ii], TrackUV[jj][ii].x, TrackUV[jj][ii].y, Scale[jj][ii], Angle[jj][ii],
					cWarp[jj][ii].warp[0], cWarp[jj][ii].warp[1], cWarp[jj][ii].warp[2], cWarp[jj][ii].warp[3]);
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
	printLOG("***Done with (cid, fid):  (%d, %d)***\n", viewID, startF);

	delete[]TruePid, delete[]FID, delete[]TrackUV, delete[]Scale, delete[]Angle, delete[]cWarp;
	return 0;
}
int DeletePointsOf2DTracks(char *Path, int nCams, int npts)
{
	char Fname[512];

	int pid;
	vector<int>ChosenPid;
	sprintf(Fname, "%s/chosen.txt", Path);  FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d ", &pid) != EOF)
		ChosenPid.push_back(pid);
	fclose(fp);

	double u, v;
	int fid, nf;
	ImgPtEle ptele;
	for (int camID = 0; camID < nCams; camID++)
	{
		vector<ImgPtEle> *Track2D = new vector<ImgPtEle>[npts];
		sprintf(Fname, "%s/Track2D/C_%.4d.txt", Path, camID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		for (int jj = 0; jj < npts; jj++)
		{
			fscanf(fp, "%d %d ", &pid, &nf);
			if (pid != jj)
				printLOG("Problem at Point %d of Cam %d", jj, camID);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %lf %lf ", &fid, &u, &v);
				ptele.frameID = fid, ptele.pt2D = Point2d(u, v);
				Track2D[jj].push_back(ptele);
			}
		}
		fclose(fp);

		sprintf(Fname, "%s/Track2D/C_%.4d.txt", Path, camID); fp = fopen(Fname, "w+");
		for (int pid = 0; pid < ChosenPid.size(); pid++)
		{
			int trackID = ChosenPid[pid];
			fprintf(fp, "%d %d ", pid, Track2D[trackID].size());
			for (int fid = 0; fid < Track2D[trackID].size(); fid++)
				fprintf(fp, "%d %.3f %.3f ", Track2D[trackID][fid].frameID, Track2D[trackID][fid].pt2D.x, Track2D[trackID][fid].pt2D.y);
			fprintf(fp, "\n");
		}
		fclose(fp);

		delete[]Track2D;
	}

	return 0;
}
int DelSel2DTrajectory(char *Path, int nCams)
{
	char Fname1[200], Fname2[512];

	int npts, pid, nf, fid;
	float u, v, s;

	vector<int> goodPid;
	sprintf(Fname1, "%s/Track2D/good.txt", Path);
	FILE *fp = fopen(Fname1, "r");
	while (fscanf(fp, "%d ", &pid) != EOF)
		goodPid.push_back(pid);
	fclose(fp);

	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		FILE *fp = fopen(Fname1, "r");
		if (IsFileExist(Fname1) == 0)
			return 1;
		fscanf(fp, "%d ", &npts);

		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid); FILE *fp_out = fopen(Fname2, "w+");
		fprintf(fp_out, "%d\n", npts);

		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			bool found = false;
			for (int ii = 0; ii < (int)goodPid.size(); ii++)
			{
				if (pid == goodPid[ii])
				{
					found = true;
					break;
				}
			}
			if (found)
				fprintf(fp_out, "%d %d ", pid, nf);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f ", &fid, &u, &v, &s);
				if (found)
					fprintf(fp_out, "%d %.4f %.4f %.3f ", fid, u, v, s);
			}
			if (found)
				fprintf(fp_out, "\n");
		}
		fclose(fp), fclose(fp_out);

		MyCopyFile(Fname2, Fname1);
		remove(Fname2);
	}
	return 0;
}
int CleanUp2DTrackingByGradientConsistency(char *Path, int nviews, int refFrame, int increF, int TrackRange, int*frameTimeStamp, int DenseDriven)
{
	char Fname[512];
	float s, a, w0, w1, w2, w3;
	Point2f uv;
	int  pid, nf, fid;

	char *InsertC;
	if (DenseDriven == 0)
		InsertC = "";
	else
		InsertC = "D";

	double *disp = new double[2 * TrackRange];
	Point2f *TrackUV = new Point2f[2 * TrackRange];
	float *Scale = new float[2 * TrackRange], *Angle = new float[2 * TrackRange];
	int *realFrameID = new int[2 * TrackRange];
	AffinePara *warpingParas = new AffinePara[2 * TrackRange];
	for (int viewID = 0; viewID < nviews; viewID++)
	{
		int trueStart = refFrame - frameTimeStamp[viewID];
		sprintf(Fname, "%s/Track2D/%s%d_%.4d.txt", Path, InsertC, viewID, refFrame); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		int npts, ndel = 0; fscanf(fp, "%d ", &npts);

		sprintf(Fname, "%s/Track2D/C%s_%d_%.4d.txt", Path, InsertC, viewID, refFrame); FILE *fp2 = fopen(Fname, "w+");
		fprintf(fp2, "%d\n", npts);
		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			for (int ii = 0; ii < 2 * TrackRange; ii++)
				TrackUV[ii] = Point2f(-1, -1);

			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
				realFrameID[(fid - trueStart) / increF + TrackRange] = fid;
				TrackUV[(fid - trueStart) / increF + TrackRange] = uv;
				Scale[(fid - trueStart) / increF + TrackRange] = s;
				Angle[(fid - trueStart) / increF + TrackRange] = a;
				warpingParas[(fid - trueStart) / increF + TrackRange].warp[0] = w0;
				warpingParas[(fid - trueStart) / increF + TrackRange].warp[1] = w1;
				warpingParas[(fid - trueStart) / increF + TrackRange].warp[2] = w2;
				warpingParas[(fid - trueStart) / increF + TrackRange].warp[3] = w3;
			}

			int startF = -1, stopF = -1, validFrames = 0;
			for (int ii = 0; ii < 2 * TrackRange; ii++)
			{
				if (TrackUV[ii].x > 0.0 && TrackUV[ii].y > 0.0)
				{
					if (startF == -1)
						startF = ii;
					stopF = ii, validFrames++;
				}
			}
			if (validFrames < 10)
				continue;

			double meanDisp = 0.0;
			for (int ii = startF; ii < stopF; ii++)
			{
				disp[ii] = sqrt(pow(TrackUV[ii + 1].x - TrackUV[ii].x, 2) + pow(TrackUV[ii + 1].y - TrackUV[ii].y, 2));
				meanDisp += disp[ii];
			}
			meanDisp = meanDisp / (stopF - startF);

			//check for large changes
			bool badtrack = false;
			for (int ii = startF + 1; ii < stopF - 1; ii++)
			{
				if (disp[ii + 1] > 5.0*disp[ii] && disp[ii + 1] > 5.0*disp[ii + 2] && disp[ii + 1] > 5.0*meanDisp)
				{
					ndel++;
					badtrack = true;
					break;
				}
			}

			if (!badtrack)
			{
				fprintf(fp2, "%d %d ", pid, validFrames);
				for (int ii = 0; ii < 2 * TrackRange; ii++)
					if (TrackUV[ii].x > 0.0 && TrackUV[ii].y > 0.0)
						fprintf(fp2, "%d %.4f %.4f %.2f %.3f %.8f %.8f %.8f %.8f ", realFrameID[ii], TrackUV[ii].x, TrackUV[ii].y, Scale[ii], Angle[ii], warpingParas[ii].warp[0], warpingParas[ii].warp[1], warpingParas[ii].warp[2], warpingParas[ii].warp[3]);
				fprintf(fp2, "\n");
			}
		}
		fclose(fp), fclose(fp2);
		printLOG("Cam (%d, %d): %d points deleted\n", viewID, refFrame, ndel);
	}

	delete[]realFrameID, delete[]Scale, delete[]disp;// , delete[]TrackUV;

	return 0;
}
int RemoveLargelyDisconnectedPointsInTraj(char *Path, int nCams, int DiscoThresh)
{
	char Fname1[200], Fname2[512];
	int npts, pid, fid, nf;
	float u, v, s, a;
	AffinePara wp;

	vector<int> vfid, vgood;
	vector<float> vu, vv, vs, va;
	vector<AffinePara> vwp;
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);	FILE *fp = fopen(Fname1, "r");
		if (IsFileExist(Fname1) == 0)
			return 1;
		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid);	FILE *fp2 = fopen(Fname2, "w+");

		fscanf(fp, "%d ", &npts); fprintf(fp2, "%d\n", npts);
		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			vu.clear(), vv.clear(), vs.clear(), va.clear(), vwp.clear(), vfid.clear(), vgood.clear();
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f %f %lf %lf %lf %lf ", &fid, &u, &v, &s, &a, &wp.warp[0], &wp.warp[1], &wp.warp[2], &wp.warp[3]);
				vu.push_back(u), vv.push_back(v), vs.push_back(s), va.push_back(a), vwp.push_back(wp);
				vfid.push_back(fid);
			}

			std::sort(vfid.begin(), vfid.end());

			bool drifted = false;
			for (int ii = 0; ii < (int)vfid.size() - 1; ii++)
			{
				if (vfid[ii + 1] - vfid[ii] < DiscoThresh)
					vgood.push_back(ii);
				else if (vfid[ii + 1] - vfid[ii] > 3 * DiscoThresh)
				{
					drifted = true;
					break;
				}
			}
			if (vfid[nf - 1] - vfid[nf - 2] < DiscoThresh)
				vgood.push_back(nf - 1);

			if (drifted)
				vgood.clear();

			fprintf(fp2, "%d %d ", pid, (int)vgood.size());
			for (int ii = 0; ii < (int)vgood.size(); ii++)
			{
				int gid = vgood[ii];
				fprintf(fp2, "%d %.4f %.4f %.3f %.4f %.8f %.8f %.8f %.8f ", vfid[gid], vu[gid], vv[gid], vs[gid], va[gid], vwp[gid].warp[0], vwp[gid].warp[1], vwp[gid].warp[2], vwp[gid].warp[3]);
			}
			fprintf(fp2, "\n");
		}
		fclose(fp); fclose(fp2);

		MyCopyFile(Fname2, Fname1);
		remove(Fname2);
	}

	return 0;
}
int Clean2DTrajStartEndAppearance(char *Path, int cid, int stopF, int TrackingInst)
{
	char Fname1[200], Fname2[512];
	int npts, pid, fid, nf, ngood = 0;
	float u, v, s, a;
	AffinePara wp;

	vector<int> vfid;
	vector<float> vu, vv, vs, va;
	vector<AffinePara> vwp;

	LKParameters LKArg;
	LKArg.DIC_Algo = 8, LKArg.InterpAlgo = 1, LKArg.ZNCCThreshold = 0.6; LKArg.hsubset = 40;// to be set depending of sift scale
	LKArg.Incomplete_Subset_Handling = 0, LKArg.Convergence_Criteria = 0, LKArg.Analysis_Speed = 0, LKArg.IterMax = 15; //usually, camera are well orientied to see the same point-->many iteration can falsely rotate the patch to match with other wrong points

	double imgScale = 1.0;
	double *Timg = new double[(2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)];
	double *CorrelBuf = new double[6 * (2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)];

	vector<float *>ImgPara(stopF);
	for (int ii = 0; ii < stopF; ii++)
		ImgPara[ii] = NULL;

	Mat Img;
	if (TrackingInst == -1)
	{
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);	FILE *fp = fopen(Fname1, "r");
		if (IsFileExist(Fname1) == 0)
			return 1;
		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid);	FILE *fp2 = fopen(Fname2, "w+");

		int tcount = 1;
		double start = omp_get_wtime();
		fscanf(fp, "%d ", &npts); fprintf(fp2, "%d\n", npts);
		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			if (omp_get_wtime() - start > 5.0e3*tcount)
			{
#pragma omp critical
				printLOG(" @(%d, %d/%d) .. ", cid, pid, npts);
				tcount++;
			}
			vfid.clear(), vu.clear(), vv.clear(), vs.clear(), va.clear(), vwp.clear();
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f %f %lf %lf %lf %lf ", &fid, &u, &v, &s, &a, &wp.warp[0], &wp.warp[1], &wp.warp[2], &wp.warp[3]);
				vu.push_back(u), vv.push_back(v), vs.push_back(s), va.push_back(a), vwp.push_back(wp);
				vfid.push_back(fid);
			}
			if (nf < 6)
				continue;
			std::sort(vfid.begin(), vfid.end());

			if (ImgPara[vfid[2]] == NULL)
			{
				sprintf(Fname1, "%s/%d/%.4d.png", Path, cid, vfid[2]);
				if (IsFileExist(Fname1) == 0)
				{
					sprintf(Fname1, "%s/%d/%.4d.jpg", Path, cid, vfid[2]);
					if (IsFileExist(Fname1) == 0)
						continue;
				}
				Img = imread(Fname1, 0);
				ImgPara[vfid[2]] = new float[Img.cols* Img.rows];
				imgScale = 1.0*Img.cols / 1920.0;
				Generate_Para_Spline(Img.data, ImgPara[vfid[2]], Img.cols, Img.rows, 1);
			}

			if (ImgPara[vfid[nf - 3]] == NULL)
			{
				sprintf(Fname2, "%s/%d/%.4d.png", Path, cid, vfid[nf - 3]);
				if (IsFileExist(Fname2) == 0)
				{
					sprintf(Fname2, "%s/%d/%.4d.jpg", Path, cid, vfid[nf - 3]);
					if (IsFileExist(Fname1) == 0)
						continue;
				}
				Img = imread(Fname2, 0);
				ImgPara[vfid[nf - 3]] = new float[Img.cols* Img.rows];
				imgScale = 1.0*Img.cols / 1920.0;
				Generate_Para_Spline(Img.data, ImgPara[vfid[nf - 3]], Img.cols, Img.rows, 1);
			}

			double maxFscale = fmax(vs[2], vs[nf - 3]);
			LKArg.hsubset = (int)(min(max(3.0*maxFscale, 8.0*imgScale), 20 * imgScale) + 0.5);

			Point2d pt1(vu[2], vv[2]), pt2(vu[nf - 3], vv[nf - 3]);
			double iWp[4]; iWp[0] = vs[2] / vs[nf - 3] - 1.0, iWp[1] = 0, iWp[2] = 0, iWp[3] = iWp[0]; //prescale the features
			double zncc = TemplateMatching(ImgPara[vfid[2]], ImgPara[vfid[nf - 3]], Img.cols, Img.rows, Img.cols, Img.rows, 1, pt1, pt2, LKArg, false, Timg, CorrelBuf, iWp);

			if (zncc < LKArg.ZNCCThreshold) //end & start do not correspond to the same point
				continue;

			ngood++;
			fprintf(fp2, "%d %d ", pid, nf);
			for (int ii = 0; ii < nf; ii++)
				fprintf(fp2, "%d %.4f %.4f %.3f %.4f %.8f %.8f %.8f %.8f ", vfid[ii], vu[ii], vv[ii], vs[ii], va[ii], vwp[ii].warp[0], vwp[ii].warp[1], vwp[ii].warp[2], vwp[ii].warp[3]);
			fprintf(fp2, "\n");
		}
		fclose(fp); fclose(fp2);

		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid);
		MyCopyFile(Fname2, Fname1);
		remove(Fname2);

#pragma omp critical
		printLOG("\nDone with %.4d. %d good trajs\n", cid, ngood);
	}

	delete[]Timg, delete[]CorrelBuf;
	for (int ii = 0; ii < stopF; ii++)
		if (ImgPara[ii] != NULL)
			delete[]ImgPara[ii];

	return 0;
}
int CombineFlow_PLK_B(char *Path, int nviews, int refFrame, int TrackRange, int maxNpts)
{
	char Fname[512];
	float s, a, w0, w1, w2, w3;
	AffinePara wp;
	Point2f uv;
	int  pid, nf, fid, npts, nrep = 0;

	vector<int> TruePid;
	vector<int> *VFid = new vector<int>[maxNpts];
	vector<Point2f> *Vuv = new vector<Point2f>[maxNpts];
	vector<float> *Vs = new vector<float>[maxNpts];
	vector<float> *Va = new vector<float>[maxNpts];
	vector<AffinePara> *Vwp = new vector<AffinePara>[maxNpts];

	int *tID = new int[2 * TrackRange];
	int *sFid = new int[2 * TrackRange];

	for (int cid = 0; cid < nviews; cid++)
	{
		TruePid.clear();
		for (int ii = 0; ii < maxNpts; ii++)
			VFid[ii].clear(), Vuv[ii].clear(), Vs[ii].clear(), Va[ii].clear(), Vwp[ii].clear();

		sprintf(Fname, "%s/Track2D/%d_%.4d.txt", Path, cid, refFrame); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		fscanf(fp, "%d ", &npts);
		while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
		{
			bool failure = false;
			if (TruePid.size() > 0 && pid == TruePid.back())
				failure = true;

			if (!failure)
				TruePid.push_back(pid);

			for (int jj = 0; jj < nf; jj++)
			{
				fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
				wp.warp[0] = w0, wp.warp[1] = w1, wp.warp[2] = w2, wp.warp[3] = w3;
				if (!failure)
					VFid[pid].push_back(fid), Vuv[pid].push_back(uv), Vs[pid].push_back(s), Va[pid].push_back(a), Vwp[pid].push_back(wp);
			}
		}
		fclose(fp);

		sprintf(Fname, "%s/Track2D/D%d_%.4d.txt", Path, cid, refFrame); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			fscanf(fp, "%d ", &npts);
			while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
			{
				bool found = false;
				for (int ii = 0; ii < (int)TruePid.size() && !found; ii++)
					if (pid == TruePid[ii])
						found = true;

				if (!found)
				{
					TruePid.push_back(pid);
					for (int jj = 0; jj < nf; jj++)
					{
						fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
						wp.warp[0] = w0, wp.warp[1] = w1, wp.warp[2] = w2, wp.warp[3] = w3;
						VFid[pid].push_back(fid), Vuv[pid].push_back(uv), Vs[pid].push_back(s), Va[pid].push_back(a), Vwp[pid].push_back(wp);
					}
				}
				else
				{
					for (int jj = 0; jj < nf; jj++)
					{
						fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
						wp.warp[0] = w0, wp.warp[1] = w1, wp.warp[2] = w2, wp.warp[3] = w3;
						int id = -1;
						for (int kk = 0; kk < (int)VFid[pid].size() && id == -1; kk++)
							if (fid == VFid[pid][kk])
								id = kk;

						if (id == -1)
							VFid[pid].push_back(fid), Vuv[pid].push_back(uv), Vs[pid].push_back(s), Va[pid].push_back(a), Vwp[pid].push_back(wp);
						else
							Vuv[pid][id] = uv, Vs[pid][id] = s, nrep++;
					}
				}
			}
			fclose(fp);
		}

		npts = (int)TruePid.size();
		sprintf(Fname, "%s/Track2D/C_%d_%.4d.txt", Path, cid, refFrame); fp = fopen(Fname, "w+");
		fprintf(fp, "%d\n", npts);
		for (int ii = 0; ii < npts; ii++)
		{
			int pid = TruePid[ii], nf = (int)VFid[pid].size();
			fprintf(fp, "%d %d ", pid, nf);

			for (int jj = 0; jj < nf; jj++)
				tID[jj] = jj, sFid[jj] = VFid[pid][jj];
			//Quick_Sort_Int(sFid, tID, 0, nf - 1);

			for (int jj = 0; jj < nf; jj++)
			{
				int id = tID[jj];
				fprintf(fp, "%d %.4f %.4f %.3f %.4f %.8f %.8f %.8f %.8f ", VFid[pid][id], Vuv[pid][id].x, Vuv[pid][id].y, Vs[pid][id], Va[pid][id], Vwp[pid][id].warp[0], Vwp[pid][id].warp[1], Vwp[pid][id].warp[2], Vwp[pid][id].warp[3]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	delete[]VFid, delete[]Vuv, delete[]Vs, delete[]Va, delete[]Vwp, delete[]tID, delete[]sFid;

	return 0;
}
int CombineTrack2D(char *Path, int nviews, vector<int> TrackingInst)
{
	int dummy, pid, fid, nf = 0;
	float u, v, s;
	char Fname[512];

	vector<int> NewID;
	for (int cid = 0; cid < nviews; cid++)
	{
		int npts = 0;
		for (int inst = 0; inst < (int)TrackingInst.size(); inst++)
		{
			int refFid = TrackingInst[inst];
			sprintf(Fname, "%s/Track2D/CC_%d_%.4d.txt", Path, cid, refFid); FILE *fp1 = fopen(Fname, "r");
			fscanf(fp1, "%d ", &dummy);
			npts += dummy;
			fclose(fp1);
		}

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, cid); FILE *fp2 = fopen(Fname, "w+");
		fprintf(fp2, "%d\n", npts);

		npts = 0;
		for (int inst = 0; inst < (int)TrackingInst.size(); inst++)
		{
			int refFid = TrackingInst[inst];
			sprintf(Fname, "%s/Track2D/CC_%d_%.4d.txt", Path, cid, refFid); FILE *fp1 = fopen(Fname, "r");

			fscanf(fp1, "%d ", &dummy);
			while (fscanf(fp1, "%d %d", &pid, &nf) != EOF)
			{
				fprintf(fp2, "%d %d ", npts + pid, nf);
				for (int ii = 0; ii < nf; ii++)
				{
					fscanf(fp1, "%d %f %f %f ", &fid, &u, &v, &s);
					fprintf(fp2, "%d %.4f %.4f %.2f ", fid, u, v, s);
				}
				fprintf(fp2, "\n");
			}
			npts += dummy;
			fclose(fp1);
		}
		fclose(fp2);
	}
	return 0;
}
int RemoveDuplicatedMatches(char *Path, int nCams, int startF)
{
	char Fname[512];

	vector<int> vpid, vfid, vunique, dif;
	vector<float> vu, vv, vs, va;
	int pid, fid; float u, v, s, a;
	for (int cid = 0; cid < nCams; cid++)
	{
		vpid.clear(), vfid.clear(), vu.clear(), vv.clear(), vs.clear(), va.clear(), vunique.clear(), dif.clear();

		sprintf(Fname, "%s/Dynamic/K_%d_%.4d.txt", Path, cid, startF);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%d %d %f %f %f %f ", &pid, &fid, &u, &v, &s, &a) != EOF)
			vpid.push_back(pid), vfid.push_back(fid), vu.push_back(u), vv.push_back(v), vs.push_back(s), va.push_back(a);
		fclose(fp);


		std::sort(vpid.begin(), vpid.end());
		vunique = vpid; vunique.erase(std::unique(vunique.begin(), vunique.end()), vunique.end());
		dif.resize(vpid.size() + vunique.size());

		std::vector<int>::iterator it;
		it = std::set_difference(vpid.begin(), vpid.end(), vunique.begin(), vunique.end(), dif.begin());
		dif.resize(it - dif.begin());

		sprintf(Fname, "%s/Dynamic/K_%d_%.4d.txt", Path, cid, startF); fp = fopen(Fname, "w+");
		for (size_t ll = 0; ll < vpid.size(); ll++)
		{
			pid = vpid[ll], fid = vfid[ll], u = vu[ll], v = vv[ll], s = vs[ll], a = va[ll];
			bool found = false;
			for (int ii = 0; ii < (int)dif.size() && !found; ii++)
				if (pid == dif[ii])
					found = true;
			if (!found)
				fprintf(fp, "%d %d %.4f %.4f %.3f %.4f\n", pid, fid, u, v, s, a);
		}
		fclose(fp);
	}

	return 0;
}
int RemoveMatchesOfDifferentClass(char *Path, int nCams, int InstF, int *TimeStamp, vector<Point3i> &ClassColor, bool inverseColor)
{
	char Fname[512];

	int pid, fid; float u, v, s, a;
	Mat img;
	vector<int> vpid, vfid;
	vector<float> vu, vv, vs, va;
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname, "%s/Seg/%d/%.4d.png", Path, cid, InstF - TimeStamp[cid]); img = imread(Fname);
		if (img.empty() == 1)
			continue;

		sprintf(Fname, "%s/Dynamic/K_%d_%.4d.txt", Path, cid, InstF);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		vpid.clear(), vfid.clear(), vu.clear(), vv.clear(), vs.clear(), va.clear();
		while (fscanf(fp, "%d %d %f %f %f %f ", &pid, &fid, &u, &v, &s, &a) != EOF)
			vpid.push_back(pid), vfid.push_back(fid), vu.push_back(u), vv.push_back(v), vs.push_back(s), va.push_back(a);
		fclose(fp);

		sprintf(Fname, "%s/Dynamic/K_%d_%.4d.txt", Path, cid, InstF); fp = fopen(Fname, "w+");
		for (size_t ll = 0; ll < vpid.size(); ll++)
		{
			pid = vpid[ll], fid = vfid[ll], u = vu[ll], v = vv[ll], s = vs[ll], a = va[ll];
			bool found = false;
			for (size_t kk = 0; kk < ClassColor.size() && !found; kk++)
			{
				int ii = (int)u, jj = (int)v;
				uchar b = img.data[3 * (ii + jj * img.cols)], g = img.data[3 * (ii + jj * img.cols) + 1], r = img.data[3 * (ii + jj * img.cols) + 2];
				if ((r == (uchar)ClassColor[kk].x && g == (uchar)ClassColor[kk].y && b == (uchar)ClassColor[kk].z))
					found = true;
			}
			if (!inverseColor && found)
				fprintf(fp, "%d %d %.4f %.4f %.3f %.4f\n", pid, fid, u, v, s, a);
			else if (inverseColor && !found)
				fprintf(fp, "%d %d %.4f %.4f %.3f %.4f\n", pid, fid, u, v, s, a);
		}
		fclose(fp);
	}
	return 0;
}
int RemoveMatchesOfDifferentClass(vector<Point2f> &uv, Mat &img, vector<Point3i> &ClassColor, bool inverseColor)
{
	vector<int> ToKeep;
	int radius = inverseColor ? 5 : 0;
	for (size_t ll = 0; ll < uv.size(); ll++)
	{
		bool found = false;
		for (size_t kk = 0; kk < ClassColor.size() && !found; kk++)
		{
			for (int y = -radius; y <= radius; y++) //dialation a little bit
			{
				for (int x = -radius; x <= radius; x++)
				{
					int ii = (int)uv[ll].x + x, jj = (int)uv[ll].y + y;
					if (ii<0 || ii>img.cols - 1 || jj<0 || jj>img.rows - 1)
						continue;

					uchar b = img.data[3 * (ii + jj * img.cols)], g = img.data[3 * (ii + jj * img.cols) + 1], r = img.data[3 * (ii + jj * img.cols) + 2];
					if ((r == (uchar)ClassColor[kk].x && g == (uchar)ClassColor[kk].y && b == (uchar)ClassColor[kk].z))
						found = true;
				}
			}
		}
		if (!inverseColor && found)
			ToKeep.push_back(ll);
		else if (inverseColor && !found)
			ToKeep.push_back(ll);
	}

	vector<Point2f> nuv; nuv.reserve(ToKeep.size());
	for (size_t ii = 0; ii < ToKeep.size(); ii++)
		nuv.push_back(uv[ToKeep[ii]]);
	uv = nuv;

	return 0;
}
int Reorder2DTrajectories(char *Path, int nviews, vector<int> &TrackingInst)
{
	char Fname[512];
	int  dummy, fid, pid, nf, npts = 0;
	Point2f uv; float s, a, w0, w1, w2, w3;

	if (TrackingInst.size() == 0)
	{
		vector<int> NewID;
		int maxPid = 0;
		for (int cid = 0; cid < nviews; cid++)
		{
			sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
			if (IsFileExist(Fname) == 0)
				continue;

			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%d ", &dummy);
			while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
			{
				maxPid = max(maxPid, pid);
				bool found = false;
				for (int ii = 0; ii < (int)NewID.size(); ii++)
				{
					if (pid == NewID[ii])
					{
						found = true;
						break;
					}
				}
				if (!found)
					NewID.push_back(pid);

				for (int ii = 0; ii < nf; ii++)
					fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
			}
			fclose(fp);
		}
		std::sort(NewID.begin(), NewID.end());
		int *_NewID = new int[maxPid];
		for (int ii = 0; ii < (int)NewID.size(); ii++)
			_NewID[NewID[ii]] = ii;
		npts = (int)NewID.size();

		for (int cid = 0; cid < nviews; cid++)
		{
			char Fname1[2000], Fname2[2000];
			sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid); FILE *fp1 = fopen(Fname1, "r");
			sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid); FILE *fp2 = fopen(Fname2, "w+");

			fscanf(fp1, "%d ", &dummy);
			fprintf(fp2, "%d\n", npts);

			while (fscanf(fp1, "%d %d", &pid, &nf) != EOF)
			{
				fprintf(fp2, "%d %d ", _NewID[pid], nf);
				for (int ii = 0; ii < nf; ii++)
				{
					fscanf(fp1, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
					fprintf(fp2, "%d %.4f %.4f %.2f %.3f %.8f %.8f %.8f %.8f ", fid, uv.x, uv.y, s, a, w0, w1, w2, w3);
				}
				fprintf(fp2, "\n");
			}
			fclose(fp1), fclose(fp2);

			MyCopyFile(Fname2, Fname1);
			remove(Fname2);
		}
		//delete[]_NewID;
		return npts;
	}
	else
	{
		int cumAllInstNpts = 0;
		for (int inst = 0; inst < (int)TrackingInst.size(); inst++)
		{
			int refFid = TrackingInst[inst];
			vector<int> NewID;
			npts = 0;
			for (int cid = 0; cid < nviews; cid++)
			{
				sprintf(Fname, "%s/Track2D/CC_%d_%.4d.txt", Path, cid, refFid);
				if (IsFileExist(Fname) == 0)
					continue;

				FILE *fp = fopen(Fname, "r");
				fscanf(fp, "%d ", &npts);
				while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
				{
					bool found = false;
					for (int ii = 0; ii < (int)NewID.size(); ii++)
					{
						if (pid == NewID[ii])
						{
							found = true;
							break;
						}
					}
					if (!found)
						NewID.push_back(pid);

					if (npts < pid + 1)
						npts = pid + 1;

					for (int ii = 0; ii < nf; ii++)
						fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
				}
				fclose(fp);
			}
			if (npts == 0)
				continue;

			int *_NewID = new int[npts];
			for (int ii = 0; ii < (int)NewID.size(); ii++)
				_NewID[NewID[ii]] = ii;
			npts = (int)NewID.size();
			cumAllInstNpts += npts;
			for (int cid = 0; cid < nviews; cid++)
			{
				char Fname1[200], Fname2[512];
				sprintf(Fname1, "%s/Track2D/CC_%d_%.4d.txt", Path, cid, refFid); FILE *fp1 = fopen(Fname1, "r");
				sprintf(Fname2, "%s/Track2D/_CC_%d_%.4d.txt", Path, cid, refFid); FILE *fp2 = fopen(Fname2, "w+");

				fscanf(fp1, "%d ", &dummy);
				fprintf(fp2, "%d\n", npts);

				while (fscanf(fp1, "%d %d", &pid, &nf) != EOF)
				{
					fprintf(fp2, "%d %d ", _NewID[pid], nf);
					for (int ii = 0; ii < nf; ii++)
					{
						fscanf(fp1, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
						fprintf(fp2, "%d %.4f %.4f %.2f %.3f %.8f %.8f %.8f %.8f ", fid, uv.x, uv.y, s, a, w0, w1, w2, w3);
					}
					fprintf(fp2, "\n");
				}
				fclose(fp1), fclose(fp2);

				MyCopyFile(Fname2, Fname1);
				remove(Fname2);
			}
			//delete[]_NewID;
		}
		return cumAllInstNpts;
	}
}
int Reorder2DTrajectories2(char *Path, int cid, int instF)
{
	char Fname[512];
	int  dummy, fid, pid, nf, npts = 0;
	Point2f uv; float s, a, w0, w1, w2, w3;

	vector<int> NewID;
	sprintf(Fname, "%s_%d_%.4d.txt", Path, cid, instF);
	if (IsFileExist(Fname, false) == 0)
		return 1;
	FILE *fp = fopen(Fname, "r");
	fscanf(fp, "%d ", &dummy);
	npts = 0;
	while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
	{
		npts++;
		for (int ii = 0; ii < nf; ii++)
			fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
	}
	fclose(fp);

	char Fname1[2000], Fname2[2000];
	sprintf(Fname1, "%s_%d_%.4d.txt", Path, cid, instF); FILE *fp1 = fopen(Fname1, "r");
	sprintf(Fname2, "%sC_%d_%.4d.txt", Path, cid, instF); FILE *fp2 = fopen(Fname2, "w+");

	fscanf(fp1, "%d ", &dummy), fprintf(fp2, "%d\n", npts);
	npts = 0;
	while (fscanf(fp1, "%d %d", &pid, &nf) != EOF)
	{
		fprintf(fp2, "%d %d ", npts, nf); npts++;
		for (int ii = 0; ii < nf; ii++)
		{
			fscanf(fp1, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
			fprintf(fp2, "%d %.4f %.4f %.2f %.1f %.1f %.1f %.1f %.1f ", fid, uv.x, uv.y, s, a, w0, w1, w2, w3);
		}
		fprintf(fp2, "\n");
	}
	fclose(fp1), fclose(fp2);

	MyCopyFile(Fname2, Fname1); remove(Fname2);

	return 0;
}
int ReAssembleUltimateTrajectories(char *Path, int nCams, vector<int>&TrackingInstances)
{
	char Fname[512];
	int ng = 1;

	int nseeds = (int)TrackingInstances.size();
	int pid, fid, nf; vector<int> vpid, vfid;
	Point2f uv;  vector<Point2f> vuv;
	float s, a, w0, w1, w2, w3; vector<float> vs;
	for (int cid = 0; cid < nCams; cid++)
	{
		vpid.clear(), vfid.clear(), vuv.clear();
		for (int gid = 0; gid < ng; gid++)
		{
			sprintf(Fname, "%s/Track2D/UltimateID_%d_%.4d.txt", Path, gid, cid);	FILE *fp = fopen(Fname, "r");
			if (IsFileExist(Fname) == 0)
			{
				printLOG("Cannot load %s\n", Fname);
				continue;
			}
			while (fscanf(fp, "%d ", &pid) != EOF)
				vpid.push_back(pid);
			fclose(fp);
		}

		//remove duplicated elements
		std::sort(vpid.begin(), vpid.end());
		vpid.erase(std::unique(vpid.begin(), vpid.end()), vpid.end());
		int *notfound = new int[(int)vpid.size()];
		for (int ii = 0; ii < (int)vpid.size(); ii++)
			notfound[ii] = true;

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, cid); FILE *fp_out = fopen(Fname, "w+");
		fprintf(fp_out, "%d\n", (int)vpid.size());
		int npts, cumNpts = 0;
		for (int seedID = 0; seedID < nseeds; seedID++)
		{
			sprintf(Fname, "%s/Track2D/CC_%d_%.4d.txt", Path, cid, TrackingInstances[seedID]);
			if (IsFileExist(Fname) == 0)
				continue;

			FILE *fp = fopen(Fname, "r");
			fscanf(fp, "%d ", &npts);
			while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
			{
				bool found = false;
				for (int ii = 0; ii < (int)vpid.size(); ii++)
				{
					if (pid + cumNpts == vpid[ii])
					{
						found = true;
						notfound[ii] = false;
						break;
					}
				}

				if (found)
				{
					fprintf(fp_out, "%d %d ", pid + cumNpts, nf);
					for (int ii = 0; ii < nf; ii++)
					{
						fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
						fprintf(fp_out, "%d %.4f %.4f %.2f %.4f %.8f %.8f %.8f %.8f ", fid, uv.x, uv.y, s, a, w0, w1, w2, w3);
					}
					fprintf(fp_out, "\n");
				}
				else
					for (int ii = 0; ii < nf; ii++)
						fscanf(fp, "%d %f %f %f %f %f %f %f %f ", &fid, &uv.x, &uv.y, &s, &a, &w0, &w1, &w2, &w3);
			}
			cumNpts += npts;
			fclose(fp);
		}
		fclose(fp_out);

		for (int ii = 0; ii < (int)vpid.size(); ii++)
			if (notfound[ii])
				printLOG("Points left un-occupied: %d\n", vpid[ii]);
	}

	return 0;
}
int RemoveWeakOverlappingTrajectories(char *Path, int nCams, int startF, int stopF, int *TimeStamp, int nTplus, int nPplus)
{
	char Fname[512];
	vector<int *> VVisI;

	int npts, pid, nf, fid, nframes = stopF - startF + 1;
	float u, v, s;
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &npts);
		int *VisI = new int[npts*nframes];
		VVisI.push_back(VisI);

		for (int ii = 0; ii < npts*nframes; ii++)
			VisI[ii] = 0;

		while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
		{
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f", &fid, &u, &v, &s);
				int fake_fid = fid + TimeStamp[cid];
				VisI[pid*nframes + fake_fid - startF] = 1;
			}
		}
		fclose(fp);
	}

	float *VisI = new float[npts*nframes];
	for (int ii = 0; ii < npts*nframes; ii++)
		VisI[ii] = 0.0;

	for (int pid = 0; pid < npts; pid++)
		for (int fid = nTplus; fid < nframes - nTplus; fid++)
			for (int cid = 0; cid < nCams; cid++)
				VisI[pid*nframes + fid] += VVisI[cid][pid*nframes + fid];

	bool *goodPid = new bool[npts];
	for (int ii = 0; ii < npts; ii++)
		goodPid[ii] = false;
	int goodCount = 0;
	for (int pid = 0; pid < npts; pid++)
	{
		int nPgood = 0;
		for (int fid = 0; fid < nframes; fid++)
			if (VisI[pid*nframes + fid] >= nPplus)
				nPgood++;

		if (nPgood > nTplus)
			goodPid[pid] = 1, goodCount++;
	}

	sprintf(Fname, "%s/Track2D/StrongTrajectories.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
		if (goodPid[ii] == 1)
			fprintf(fp, "%d\n", ii);
	fclose(fp);

	for (int cid = 0; cid < nCams; cid++)
	{
		char Fname1[2000], Fname2[2000];
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid); FILE *fp1 = fopen(Fname1, "r");
		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid); FILE *fp2 = fopen(Fname2, "w+");

		fscanf(fp1, "%d ", &npts);
		fprintf(fp2, "%d\n", goodCount);

		while (fscanf(fp1, "%d %d", &pid, &nf) != EOF)
		{
			bool good = goodPid[pid];
			if (good)
				fprintf(fp2, "%d %d ", pid, nf);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp1, "%d %f %f %f ", &fid, &u, &v, &s);
				if (good)
					fprintf(fp2, "%d %.4f %.4f %.2f ", fid, u, v, s);
			}
			if (good)
				fprintf(fp2, "\n");
		}
		fclose(fp1), fclose(fp2);

		MyCopyFile(Fname2, Fname1);
		remove(Fname2);
	}

	return 0;
}
int RemoveWeakOverlappingTrajectories2(char *Path, int nCams, int startF, int stopF, int *TimeStamp, int nTplus, int nPplus)
{
	char Fname[512];
	vector<int *> VVisI;

	int npts, pid, nf, fid, nframes = stopF - startF + 1;
	float u, v, s, a;
	AffinePara wp;
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%d ", &npts);
		int *VisI = new int[npts*nframes];
		VVisI.push_back(VisI);

		for (int ii = 0; ii < npts*nframes; ii++)
			VisI[ii] = 0;

		while (fscanf(fp, "%d %d ", &pid, &nf) != EOF)
		{
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f %f %lf %lf %lf %lf ", &fid, &u, &v, &s, &a, &wp.warp[0], &wp.warp[1], &wp.warp[2], &wp.warp[3]);
				int fake_fid = fid + TimeStamp[cid];
				VisI[pid*nframes + fake_fid - startF] = 1;
			}
		}
		fclose(fp);
	}

	float *VisI = new float[npts*nframes];
	for (int ii = 0; ii < npts*nframes; ii++)
		VisI[ii] = 0.0;

	for (int pid = 0; pid < npts; pid++)
		for (int fid = 0; fid < nframes; fid++)
			for (int cid = 0; cid < nCams; cid++)
				VisI[pid*nframes + fid] += VVisI[cid][pid*nframes + fid];

	int goodCount = 0;
	bool *goodPid = new bool[npts*nframes];
	for (int pid = 0; pid < npts; pid++)
	{
		int nTgood = 0;
		for (int fid = 0; fid < nframes; fid++)
		{
			if (VisI[pid*nframes + fid] >= nPplus ||
				(fid > 1 && fid < nframes && (goodPid[pid*nframes + fid - 1] || goodPid[pid*nframes + fid - 1]))) //neighbor points are strong
			{
				goodPid[pid*nframes + fid] = true; //good spatially
				nTgood++; //good temporally
			}
			else
				goodPid[pid*nframes + fid] = false;
		}

		if (nTgood < nTplus)
			for (int fid = 0; fid < nframes; fid++)
				goodPid[pid*nframes + fid] = false;
		else
			goodCount++;
	}
	printLOG("%d/%d good points\n", goodCount, npts);

	for (int cid = 0; cid < nCams; cid++)
	{
		char Fname1[2000], Fname2[2000];
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid); FILE *fp1 = fopen(Fname1, "r");
		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid); FILE *fp2 = fopen(Fname2, "w+");

		fscanf(fp1, "%d ", &npts);
		fprintf(fp2, "%d\n", npts);

		vector<int> vfid; vector<float>vu, vv, vs, va; vector<AffinePara> vwp;
		while (fscanf(fp1, "%d %d", &pid, &nf) != EOF)
		{
			vfid.clear(), vu.clear(), vv.clear(), vs.clear(), va.clear(), vwp.clear();
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp1, "%d %f %f %f %f %lf %lf %lf %lf ", &fid, &u, &v, &s, &a, &wp.warp[0], &wp.warp[1], &wp.warp[2], &wp.warp[3]);
				int fake_fid = fid + TimeStamp[cid];
				bool good = goodPid[pid*nframes + fake_fid - startF];
				if (good)
					vfid.push_back(fid), vu.push_back(u), vv.push_back(v), vs.push_back(s), va.push_back(a), vwp.push_back(wp);
			}

			if (vfid.size() > 20)
			{
				fprintf(fp2, "%d %d ", pid, (int)vfid.size());
				for (int ii = 0; ii < (int)vfid.size(); ii++)
					fprintf(fp2, "%d %.4f %.4f %.2f %.3f %.8f %.8f %.8f %.8f ", vfid[ii], vu[ii], vv[ii], vs[ii], va[ii], vwp[ii].warp[0], vwp[ii].warp[1], vwp[ii].warp[2], vwp[ii].warp[3]);
				fprintf(fp2, "\n");
			}
		}
		fclose(fp1), fclose(fp2);

		MyCopyFile(Fname2, Fname1);
		remove(Fname2);
	}

	return 0;
}
int RemoveShortTrajectory(char *Path, int nCams, int minTrajLength, int maxNpts)
{
	char Fname1[200], Fname2[512];

	int npts, pid, nf, fid;
	float u, v, s;

	Mat Img;
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		FILE *fp = fopen(Fname1, "r");
		if (IsFileExist(Fname1) == 0)
			return 1;
		fscanf(fp, "%d ", &npts);

		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid); FILE *fp_out = fopen(Fname2, "w+");
		fprintf(fp_out, "%d\n", npts);

		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			if (nf > minTrajLength)
				fprintf(fp_out, "%d %d ", pid, nf);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f ", &fid, &u, &v, &s);
				if (nf > minTrajLength)
					fprintf(fp_out, "%d %.4f %.4f %.3f ", fid, u, v, s);
			}
			if (nf > minTrajLength)
				fprintf(fp_out, "\n");
		}
		fclose(fp), fclose(fp_out);

		MyCopyFile(Fname2, Fname1);
		remove(Fname2);
	}
	return 0;
}
int RemoveTrajectoryOutsideROI(char *Path, int nCams, int startF, int stopF, int maxNpts)
{
	char Fname[512], Fname1[200], Fname2[512];

	int npts, pid, nf, fid;
	float u, v, s;

	vector<int> badTrajID;
	vector<int> TruePid;
	vector<int> *TrackFid = new vector<int>[maxNpts];
	vector<Point2f> *TrackUV = new vector<Point2f>[maxNpts];
	for (int ii = 0; ii < maxNpts; ii++)
		TrackFid[ii].reserve(stopF - startF + 1), TrackUV[ii].reserve(stopF - startF + 1);

	Mat Img;
	for (int cid = 0; cid < nCams; cid++)
	{
		TruePid.clear();
		for (int ii = 0; ii < maxNpts; ii++)
			TrackFid[ii].clear(), TrackUV[ii].clear();

		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		FILE *fp = fopen(Fname, "r");
		if (IsFileExist(Fname) == 0)
			return 1;
		fscanf(fp, "%d ", &npts);
		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			TruePid.push_back(pid);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f ", &fid, &u, &v, &s);
				TrackFid[pid].push_back(fid), TrackUV[pid].push_back(Point2f(u, v));
			}
		}
		fclose(fp);

		int count = 0;
		for (int ii = startF; ii < stopF; ii++)
		{
			sprintf(Fname, "%s/%d/TrajVis/%d - Copy.jpg", Path, cid, ii);
			Img = imread(Fname, 0);
			if (Img.empty())
				continue;

			for (int jj = 0; jj < (int)TruePid.size(); jj++)
			{
				pid = TruePid[jj];
				for (int kk = 0; kk < (int)TrackFid[pid].size(); kk++)
				{
					fid = TrackFid[pid][kk];
					if (ii != fid)
						continue;

					int x = (int)TrackUV[pid][kk].x, y = (int)TrackUV[pid][kk].y;
					int imgval = (int)Img.data[x + y * Img.cols];
					if (imgval == 255)
					{
						count++;
						badTrajID.push_back(pid);
						break;
					}
				}
			}
		}
		printLOG("Clean %d points for cam %d\n", count, cid);
	}

	//remove duplicated elements
	std::sort(badTrajID.begin(), badTrajID.end());
	badTrajID.erase(std::unique(badTrajID.begin(), badTrajID.end()), badTrajID.end());
	for (int cid = 0; cid < nCams; cid++)
	{
		sprintf(Fname1, "%s/Track2D/Ultimate_%.4d.txt", Path, cid);
		FILE *fp = fopen(Fname1, "r");
		if (IsFileExist(Fname1) == 0)
			return 1;
		fscanf(fp, "%d ", &npts);

		sprintf(Fname2, "%s/Track2D/CUltimate_%.4d.txt", Path, cid); FILE *fp_out = fopen(Fname2, "w+");
		fprintf(fp_out, "%d\n", npts - (int)badTrajID.size());

		while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
		{
			bool found = false;
			for (int ii = 0; ii < (int)badTrajID.size(); ii++)
			{
				if (badTrajID[ii] == pid)
				{
					found = true;
					break;
				}
			}
			if (!found)
				fprintf(fp_out, "%d %d ", pid, nf);
			for (int ii = 0; ii < nf; ii++)
			{
				fscanf(fp, "%d %f %f %f ", &fid, &u, &v, &s);
				if (!found)
					fprintf(fp_out, "%d %.4f %.4f %.3f ", fid, u, v, s);
			}
			if (!found)
				fprintf(fp_out, "\n");
		}
		fclose(fp), fclose(fp_out);

		MyCopyFile(Fname2, Fname1);
		remove(Fname2);
	}
	return 0;
}
int DownSampleTracking(char *Path, vector<int> &viewList, int startF, int stopF, int rate)
{
	char Fname[512];
	int npts, nframes, pid, cfid;
	Point2d uv; double s;
	vector<int>frameIDList; frameIDList.reserve(1000);
	vector<Point2d> uvList; uvList.reserve(1000);
	vector<double> sList; sList.reserve(1000);

	sprintf(Fname, "%s/@%d", Path, rate); makeDir(Fname);
	sprintf(Fname, "%s/@%d/Track2D", Path, rate); makeDir(Fname);
	for (int ii = 0; ii < (int)viewList.size(); ii++)
	{
		int viewID = viewList[ii];
		sprintf(Fname, "%s/Track2D/Ultimate_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		FILE *fp = fopen(Fname, "r"); fscanf(fp, "%d ", &npts);

		sprintf(Fname, "%s/@%d/Track2D/Ultimate_%.4d.txt", Path, rate, viewID);
		FILE *fp2 = fopen(Fname, "w+"); fprintf(fp2, "%d\n", npts);
		while (fscanf(fp, "%d %d ", &pid, &nframes) != EOF)
		{
			frameIDList.clear(), uvList.clear();
			for (int ii = 0; ii < nframes; ii++)
			{
				fscanf(fp, "%d %lf %lf %lf ", &cfid, &uv.x, &uv.y, &s);
				frameIDList.push_back(cfid), uvList.push_back(uv), sList.push_back(s);
			}

			int nvalidFrames = 0;
			for (int ii = 0; ii < nframes; ii++)
				if ((frameIDList[ii] - startF) % rate == 0)
					nvalidFrames++;

			fprintf(fp2, "%d %d ", pid, nvalidFrames);
			for (int ii = 0; ii < nframes; ii++)
				if ((frameIDList[ii] - startF) % rate == 0)
					fprintf(fp2, "%d %.4f %.4f %.3f ", (frameIDList[ii] - startF) / rate, uvList[ii].x, uvList[ii].y, sList[ii]);
			fprintf(fp2, "\n");
		}
		fclose(fp), fclose(fp2);
	}
	return 0;
}
int DownSampleVideoCameraPose(char *Path, vector<int>&viewList, int startF, int stopF, int rate)
{
	char Fname[512];
	sprintf(Fname, "%s/@%d", Path, rate); makeDir(Fname);

	for (int ii = 0; ii < (int)viewList.size(); ii++)
	{
		int viewID = viewList[ii];
		VideoData VideoDataInfo;
		if (ReadVideoDataI(Path, VideoDataInfo, viewID, startF, stopF) == 1)
			return 1;

		sprintf(Fname, "%s/@%d/savIntrinsic_%.4d.txt", Path, rate, viewID);
		FILE *fp = fopen(Fname, "w+");
		for (int fid = startF; fid < stopF; fid++)
		{
			if ((fid - startF) % rate == 0 && VideoDataInfo.VideoInfo[fid].valid)
			{
				fprintf(fp, "%d %d %d %d %d ", (fid - startF) / rate, VideoDataInfo.VideoInfo[fid].LensModel, VideoDataInfo.VideoInfo[fid].ShutterModel, VideoDataInfo.VideoInfo[fid].width, VideoDataInfo.VideoInfo[fid].height);
				for (int ii = 0; ii < 5; ii++)
					fprintf(fp, "%f ", VideoDataInfo.VideoInfo[fid].intrinsic[ii]);
				if (VideoDataInfo.VideoInfo[fid].LensModel == 0)
					for (int ii = 0; ii < 7; ii++)
						fprintf(fp, "%e ", VideoDataInfo.VideoInfo[fid].distortion[ii]);
				else
					for (int ii = 0; ii < 3; ii++)
						fprintf(fp, "%e ", VideoDataInfo.VideoInfo[fid].distortion[ii]);
				fprintf(fp, "\n");
			}
		}
		fclose(fp);

		if (VideoDataInfo.VideoInfo[startF].ShutterModel == 0)
			sprintf(Fname, "%s/@%d/savCamPose_%.4d.txt", Path, rate, viewID);
		else
			sprintf(Fname, "%s/@%d/savCamPose_RSCayley_%.4d.txt", Path, rate, viewID);
		fp = fopen(Fname, "w+");
		for (int fid = startF; fid < stopF; fid++)
		{
			if ((fid - startF) % rate == 0 && VideoDataInfo.VideoInfo[fid].valid)
			{
				fprintf(fp, "%d ", (fid - startF) / rate);
				for (int ii = 0; ii < 6; ii++)
					fprintf(fp, "%.16f ", VideoDataInfo.VideoInfo[fid].rt[ii]);
				if (VideoDataInfo.VideoInfo[fid].ShutterModel)
					for (int ii = 0; ii < 6; ii++)
						fprintf(fp, "%.16f ", VideoDataInfo.VideoInfo[fid].wt[ii]);
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}

	return 0;
}
int DownSampleImageSequence(char *Path, vector<int> &sCams, int startF, int stopF, int rate)
{
	char Fname[512];
	sprintf(Fname, "%s/@%d", Path, rate); makeDir(Fname);

	Mat  img;
	for (int ii = 0; ii < (int)sCams.size(); ii++)
	{
		int cid = sCams[ii];
		sprintf(Fname, "%s/@%d/%d", Path, rate, cid); makeDir(Fname);

		for (int fid = startF; fid < stopF; fid++)
		{
			if ((fid - startF) % rate == 0)
			{
				sprintf(Fname, "%s/%d/%.4d.png", Path, cid, fid);
				if (IsFileExist(Fname) == 0)
					sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid, fid);
				img = imread(Fname, 1);
				if (img.empty())
					continue;

				sprintf(Fname, "%s/@%d/%d/%.4d.jpg", Path, rate, cid, (fid - startF) / rate);
				imwrite(Fname, img);
			}
		}
	}

	return 0;
}


cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
	cv::Mat output;

	double h1 = dstSize.width * (input.rows / (double)input.cols);
	double w2 = dstSize.height * (input.cols / (double)input.rows);
	if (h1 <= dstSize.height) {
		cv::resize(input, output, cv::Size(dstSize.width, h1));
	}
	else {
		cv::resize(input, output, cv::Size(w2, dstSize.height));
	}

	int top = (dstSize.height - output.rows) / 2;
	int down = (dstSize.height - output.rows + 1) / 2;
	int left = (dstSize.width - output.cols) / 2;
	int right = (dstSize.width - output.cols + 1) / 2;

	cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);

	return output;
}

//experiement specific code: Checkerboard
int GenerateCheckerBoardFreeImage(char *Path, int camID, int npts, int startF, int stopF)
{
	char Fname[512];

	int width = 0, height = 0;
	Mat cvImg;
	for (int fid = startF; fid <= stopF; fid++)
	{
		sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fid); cvImg = imread(Fname, 0);
		if (!cvImg.empty())
		{
			width = cvImg.cols, height = cvImg.rows;
			break;
		}
	}

	if (width == 0 || height == 0)
	{
		printLOG("Found no images\n");
		return 1;
	}

	int *PixelCount = new int[width*height];
	double *AvgImage = new double[width*height];
	for (int ii = 0; ii < width*height; ii++)
		PixelCount[ii] = 0, AvgImage[ii] = 0.0;

	float x, y;
	int maxX, minX, maxY, minY;
	for (int fid = startF; fid <= stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV2_%.4d.txt", Path, camID, fid); 	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;

		maxX = 0, minX = width, maxY = 0, minY = height;
		for (int ii = 0; ii < npts; ii++)
		{
			fscanf(fp, "%f %f ", &x, &y);
			if (x > maxX)
				maxX = min((int)x, width - 1);
			if (x < minX)
				minX = max((int)x, 0);
			if (y > maxY)
				maxY = min((int)y, height - 1);
			if (y < minY)
				minY = max((int)y, 0);
		}

		sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fid); cvImg = imread(Fname, 0);
		if (!cvImg.empty())
			continue;

		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				if (ii > minX && ii <maxX && jj>minY && jj < maxY)
					continue;
				AvgImage[ii + jj * width] += (double)(int)cvImg.data[ii + jj * width];
				PixelCount[ii + jj * width]++;
			}
		}
	}


	for (int ii = 0; ii < width*height; ii++)
		AvgImage[ii] = AvgImage[ii] / PixelCount[ii];

	sprintf(Fname, "%s/Corpus/", Path); makeDir(Fname);
	sprintf(Fname, "%s/Corpus/%.4d.png", Path, camID);
	SaveDataToImage(Fname, AvgImage, width, height, 1);

	delete[]PixelCount, delete[]AvgImage;
	return 0;
}

int CleanUpCheckerboardCorner(char *Path, int startF, int stopF)
{
	char Fname[512];
	vector<int> GoodId;
	vector<double> X, Y;

	//Scattered points signify outliers
	int t;
	double u, v, mx, my, vx, vy, v2, distance;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/Corner/%.4d.txt", Path, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%d %lf %lf ", &t, &u, &v) != EOF)
			X.push_back(u), Y.push_back(v);
		fclose(fp);

		mx = MeanArray(X), my = MeanArray(Y);
		vx = VarianceArray(X, mx), vy = VarianceArray(Y, my);
		v2 = sqrt(vx*vx + vy * vy);

		sprintf(Fname, "%s/Corner/_%.4d.txt", Path, fid); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < X.size(); ii++)
		{
			distance = sqrt(pow(X[ii] - mx, 2) + pow(Y[ii] - my, 2));
			if (distance < 3 * v2)
				fprintf(fp, "%.3f %.3f\n", X[ii], Y[ii]);
		}
		fclose(fp);

		GoodId.clear(), X.clear(), Y.clear();
	}

	return 0;
}
int RefineCheckBoardDetection2(char *Path, int viewID, int startF, int stopF)
{
	char Fname[512];

	//sprintf(Fname, "%s/%d", Path, viewID);
	//int checkerSize = 18; double znccThresh = 0.93;
	//CornerDetectorDriver(Fname, checkerSize, znccThresh, startF, stopF, 1280, 720);
	const int npts = 84;
	int width = 1280, height = 720, length = width * height, nchannels = 1;
	int interpAlgo = 1, hsubset = 10, searchArea = 1;
	double ZNCCThresh = 0.9;

	vector<double> PatternAngles;
	for (int ii = 0; ii < 9; ii++)
		PatternAngles.push_back(10 * ii);


	double *Img = new double[length*nchannels], *Para = new double[length*nchannels];

	int count, nptsI;
	double u, v;
	Point2d cvPts[npts];
	vector<ImgPtEle> *AllPts = NULL;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV_%.4d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		count = 0;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			cvPts[count] = Point2d(u, v), count++;
		fclose(fp);

		if (AllPts == NULL)
			AllPts = new vector<ImgPtEle>[npts];

		//refine those corners with correlation
		sprintf(Fname, "%s/%d/%.4d.png", Path, viewID, fid); GrabImage(Fname, Img, width, height, nchannels);
		for (int kk = 0; kk < nchannels; kk++)
			Generate_Para_Spline(Img + kk * length, Para + kk * length, width, height, interpAlgo);

		nptsI = npts;
		RefineCheckerCornersFromInit(Para, width, height, nchannels, cvPts, nptsI, PatternAngles, hsubset, hsubset, searchArea, ZNCCThresh - 0.3, ZNCCThresh, interpAlgo);

		sprintf(Fname, "%s/%d/Corner/%.4d.txt", Path, viewID, fid); fp = fopen(Fname, "w+");
		for (int ii = 0; ii < npts; ii++)
			fprintf(fp, "%.3f %.3f \n", cvPts[ii].x, cvPts[ii].y);
		fclose(fp);

		for (int ii = 0; ii < nptsI; ii++)
		{
			ImgPtEle impt; impt.frameID = fid;
			if (cvPts[ii].x > 1 && cvPts[ii].y > 1 && cvPts[ii].x < width - 1 && cvPts[ii].y < height - 1)
			{
				impt.pt2D = cvPts[ii];
				AllPts[ii].push_back(impt);
			}
		}
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	sprintf(Fname, "%s/Track2D/%.4d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		fprintf(fp, "%d %d ", ii, AllPts[ii].size());
		for (int jj = 0; jj < AllPts[ii].size(); jj++)
			fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x, AllPts[ii][jj].pt2D.y);
		fprintf(fp, "%\n");
	}
	fclose(fp);

	return 0;
}
int CleanCheckBoardDetection3(char *Path, int viewID, int startF, int stopF)
{
	char Fname[512];

	//sprintf(Fname, "%s/%d", Path, viewID);
	//int checkerSize = 18; double znccThresh = 0.93;
	//CornerDetectorDriver(Fname, checkerSize, znccThresh, startF, stopF, 1280, 720);

	//Merge points 
	int npts = 0;
	double u, v;
	vector<int> goodId;
	vector<Point2d> cvPts, TemplPts;
	vector<ImgPtEle> *AllPts = NULL;
	for (int fid = startF; fid < stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV2_%.4d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			cvPts.push_back(Point2d(u, v));
		fclose(fp);

		if (AllPts == NULL)
			npts = (int)cvPts.size(), AllPts = new vector<ImgPtEle>[cvPts.size()];

		sprintf(Fname, "%s/%d/Corner/CV2_%.4d.txt", Path, viewID, fid); fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			TemplPts.push_back(Point2d(u, v));
		fclose(fp);

		int bestID;
		double distance2, mindistance2;
		for (int ii = 0; ii < cvPts.size(); ii++)
		{
			bestID = -1, mindistance2 = 9e9;
			for (int jj = 0; jj < TemplPts.size(); jj++)
			{
				distance2 = pow(cvPts[ii].x - TemplPts[jj].x, 2) + pow(cvPts[ii].y - TemplPts[jj].y, 2);
				if (distance2 < mindistance2)
				{
					mindistance2 = distance2;
					bestID = jj;
				}
			}

			if (mindistance2 < 9)
			{
				cvPts[ii] = TemplPts[bestID];
				goodId.push_back(ii);
			}
			else
			{
				cvPts[ii] = Point2d(0, 0);
				goodId.push_back(-1);
			}
		}

		for (int ii = 0; ii < cvPts.size(); ii++)
		{
			ImgPtEle impt; impt.frameID = fid;
			if (goodId[ii] > -1)
			{
				impt.pt2D = cvPts[ii];
				AllPts[ii].push_back(impt);
			}
		}
		goodId.clear(), cvPts.clear(), TemplPts.clear();
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	sprintf(Fname, "%s/Track2D/%.4d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		fprintf(fp, "%d %d ", ii, AllPts[ii].size());
		for (int jj = 0; jj < AllPts[ii].size(); jj++)
			fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x - 1, AllPts[ii][jj].pt2D.y - 1);//matlab
		//fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x, AllPts[ii][jj].pt2D.y );//C++
		fprintf(fp, "%\n");
	}
	fclose(fp);

	return 0;
}
int CleanCheckBoardDetection(char *Path, int viewID, int startF, int stopF)
{
	char Fname[512];

	int npts = 0;
	double u, v;
	vector<Point2d> cvPts;
	vector<ImgPtEle> *AllPts = NULL;
	for (int fid = startF; fid <= stopF; fid++)
	{
		sprintf(Fname, "%s/%d/Corner/CV2_%.4d.txt", Path, viewID, fid); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%lf %lf", &u, &v) != EOF)
			cvPts.push_back(Point2d(u, v));
		fclose(fp);

		if (AllPts == NULL)
			npts = (int)cvPts.size(), AllPts = new vector<ImgPtEle>[cvPts.size()];

		for (int ii = 0; ii < cvPts.size(); ii++)
		{
			ImgPtEle impt; impt.frameID = fid;
			impt.pt2D = cvPts[ii];
			AllPts[ii].push_back(impt);
		}
		cvPts.clear();
	}

	sprintf(Fname, "%s/Track2D", Path), makeDir(Fname);
	sprintf(Fname, "%s/Track2D/%.4d.txt", Path, viewID); FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		fprintf(fp, "%d %d ", ii, AllPts[ii].size());
		for (int jj = 0; jj < AllPts[ii].size(); jj++)
			fprintf(fp, "%d %.8f %.8f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x - 1, AllPts[ii][jj].pt2D.y - 1);//matlab
		//fprintf(fp, "%d %.3f %.3f ", AllPts[ii][jj].frameID, AllPts[ii][jj].pt2D.x, AllPts[ii][jj].pt2D.y );//C++
		fprintf(fp, "%\n");
	}
	fclose(fp);

	return 0;
}