#include "MiscAlgo.h"
#include "DataIO.h"

double OverlappingArea(Point2i &tl1, Point2i &br1, Point2i &tl2, Point2i &br2)
{
	double overlapX, overlapY;
	double minV1 = tl1.x, maxV1 = br1.x, minV2 = tl2.x, maxV2 = br2.x;
	if (minV1 <= minV2 && maxV2 <= maxV1)
		overlapX = maxV2 - minV2;
	else if (minV1 <= maxV2 && minV1 >= minV2)
		overlapX = maxV2 - minV1;
	else if (maxV1 >= minV2 && maxV1 <= maxV2)
		overlapX = maxV1 - minV2;
	else if (minV1 >= minV2 && maxV1 <= maxV2)
		overlapX = maxV1 - minV1;
	else
		overlapX = 0;

	minV1 = tl1.y, maxV1 = br1.y, minV2 = tl2.y, maxV2 = br2.y;
	if (minV1 <= minV2 && maxV2 <= maxV1)
		overlapY = maxV2 - minV2;
	else if (minV1 <= maxV2 && minV1 >= minV2)
		overlapY = maxV2 - minV1;
	else if (maxV1 >= minV2 && maxV1 <= maxV2)
		overlapY = maxV1 - minV2;
	else if (minV1 >= minV2 && maxV1 <= maxV2)
		overlapY = maxV1 - minV1;
	else
		overlapY = 0;

	double area = overlapX * overlapY;
	return area;
}
//DCT
void GenerateDCTBasis(int nsamples, double *Basis, double *Weight)
{
	if (Basis != NULL)
	{
		double s = sqrt(1.0 / nsamples);
		for (int ll = 0; ll < nsamples; ll++)
			Basis[ll] = s;

		for (int kk = 1; kk < nsamples; kk++)
		{
			double s = sqrt(2.0 / nsamples);
			for (int ll = 0; ll < nsamples; ll++)
				Basis[kk*nsamples + ll] = s*cos(Pi*kk *(1.0*ll - 0.5) / nsamples);
		}
	}

	if (Weight != NULL)
	{
		for (int ll = 0; ll < nsamples; ll++)
			Weight[ll] = 2.0*(cos(Pi*(ll - 1) / nsamples) - 1.0);
	}

	return;
}
void GenerateiDCTBasis(double *Basis, int nsamples, double t)
{
	Basis[0] = sqrt(1.0 / nsamples);

	double s = sqrt(2.0 / nsamples);
	for (int kk = 1; kk < nsamples; kk++)
		Basis[kk] = s*cos(Pi*kk *(t + 0.5) / nsamples);

	return;
}

//BSpline
static void BSplineLVB(const double * t, const int jhigh, const int index, const double x, const int left, int * j, double * deltal, double * deltar, double * biatx)
{
	int i;
	double saved;
	double term;

	if (index == 1)
	{
		*j = 0;
		biatx[0] = 1.0;
	}

	for (; *j < jhigh - 1; *j += 1)
	{
		deltar[*j] = t[left + *j + 1] - x;
		deltal[*j] = x - t[left - *j];

		saved = 0.0;

		for (i = 0; i <= *j; i++)
		{
			term = biatx[i] / (deltar[i] + deltal[*j - i]);

			biatx[i] = saved + deltar[i] * term;

			saved = deltal[*j - i] * term;
		}

		biatx[*j + 1] = saved;
	}

	return;
}
static void BSplineLVD(const double * knots, const int SplineOrder, const double x, const int left, double * deltal, double * deltar, double * a, double * dbiatx, const int nderiv)
{
	int i, ideriv, il, j, jlow, jp1mid, kmm, ldummy, m, mhigh;
	double factor, fkmm, sum;

	int bsplvb_j;
	double *dbcol = dbiatx;

	mhigh = min(nderiv, SplineOrder - 1);
	BSplineLVB(knots, SplineOrder - mhigh, 1, x, left, &bsplvb_j, deltal, deltar, dbcol);
	if (mhigh > 0)
	{
		ideriv = mhigh;
		for (m = 1; m <= mhigh; m++)
		{
			for (j = ideriv, jp1mid = 0; j < (int)SplineOrder; j++, jp1mid++)
				dbiatx[j + ideriv*SplineOrder] = dbiatx[jp1mid];

			ideriv--;
			BSplineLVB(knots, SplineOrder - ideriv, 2, x, left, &bsplvb_j, deltal, deltar, dbcol);
		}

		jlow = 0;
		for (i = 0; i < (int)SplineOrder; i++)
		{
			for (j = jlow; j < (int)SplineOrder; j++)
				a[i + j*SplineOrder] = 0.0;
			jlow = i;
			a[i + i*SplineOrder] = 1.0;
		}

		for (m = 1; m <= mhigh; m++)
		{
			kmm = SplineOrder - m;
			fkmm = (float)kmm;
			il = left;
			i = SplineOrder - 1;

			for (ldummy = 0; ldummy < kmm; ldummy++)
			{
				factor = fkmm / (knots[il + kmm] - knots[il]);

				for (j = 0; j <= i; j++)
					a[j + i*SplineOrder] = factor*(a[j + i*SplineOrder] - a[j + (i - 1)*SplineOrder]);

				il--;
				i--;
			}

			for (i = 0; i < (int)SplineOrder; i++)
			{
				sum = 0;
				jlow = max(i, m);
				for (j = jlow; j < (int)SplineOrder; j++)
					sum += a[i + j*SplineOrder] * dbiatx[j + m*SplineOrder];

				dbiatx[i + m*SplineOrder] = sum;
			}
		}
	}

	return;

}
void BSplineFindActiveCtrl(int *ActingID, const double x, double *knots, int nbreaks, int nControls, int SplineOrder, int extraNControls)
{
	int i;
	int nknots = nControls + SplineOrder, nPolyPieces = nbreaks - 1;

	// find i such that t_i <= x < t_{i+1} 
	for (i = SplineOrder - 1; i < SplineOrder + nPolyPieces - 1; i++)
	{
		const double ti = knots[i];
		const double tip1 = knots[i + 1];

		if (ti <= x && x < tip1)
			break;
		if (ti < x && x == tip1 && tip1 == knots[SplineOrder + nPolyPieces - 1])
			break;
	}

	int startID = i - SplineOrder + 1;
	while (startID - extraNControls / 2 < 0) //at the begining,  ---> need to decrese starting point
		startID++;
	while (startID + SplineOrder + extraNControls / 2 > nControls) //at the end, #control points is less than SplineOrder ---> need to decrese starting point
		startID--;

	startID = startID - extraNControls / 2;
	for (int ii = 0; ii < SplineOrder + extraNControls; ii++)
		ActingID[ii] = startID + ii;

	return;
}
static inline int BSplineFindInterval(const double x, int *flag, double *knots, int nbreaks, int nControls, int SplineOrder)
{
	int i;
	int nknots = nControls + SplineOrder, nPolyPieces = nbreaks - 1;
	if (x < knots[0])
	{
		*flag = -1;
		return 0;
	}

	// find i such that t_i <= x < t_{i+1} 
	for (i = SplineOrder - 1; i < SplineOrder + nPolyPieces - 1; i++)
	{
		const double ti = knots[i];
		const double tip1 = knots[i + 1];

		if (tip1 < ti)
		{
			printLOG("knots vector is not increasing"); abort();
		}

		if (ti <= x && x < tip1)
			break;

		if (ti < x && x == tip1 && tip1 == knots[SplineOrder + nPolyPieces - 1])//if (ti < x && x == tip1 && tip1 == gsl_vector_get(knots, SplineOrder + nPolyPieces	- 1))
			break;
	}

	if (i == SplineOrder + nPolyPieces - 1)
		*flag = 1;
	else
		*flag = 0;

	return i;
}
static inline int BSplineEvalInterval(const double x, int * i, const int flag, double *knots, int nbreaks, int nControls, int SplineOrder)
{
	if (flag == -1)
	{
		printLOG("x outside of knot interval"); abort();
	}
	else if (flag == 1)
	{
		if (x <= knots[*i] + DBL_EPSILON)
			*i -= 1;
		else
		{
			printLOG("x outside of knot interval"); abort();
		}
	}

	if (knots[*i] == knots[*i + 1])
	{
		printLOG("knot(i) = knot(i+1) will result in division by zero"); abort();
	}

	return 0;
}
int BSplineGetKnots(double *knots, double *BreakLoc, int nbreaks, int nControls, int SplineOrder)
{
	int i;
	for (i = 0; i < SplineOrder; i++)
		knots[i] = BreakLoc[0];

	int nPolyPieces = nbreaks - 1;
	for (i = 1; i < nPolyPieces; i++)
		knots[i + SplineOrder - 1] = BreakLoc[i];

	for (i = nControls; i < nControls + SplineOrder; i++)
		knots[i] = BreakLoc[nPolyPieces];
	return 0;
}
int BSplineGetNonZeroBasis(const double x, double * dB, int * istart, int * iend, double *knots, int nbreaks, int nControls, int SplineOrder, int nderiv)
{
	int flag = 0;

	int i = BSplineFindInterval(x, &flag, knots, nbreaks, nControls, SplineOrder);
	int error = BSplineEvalInterval(x, &i, flag, knots, nbreaks, nControls, SplineOrder);
	if (error)
		return error;

	*istart = i - SplineOrder + 1;
	*iend = i;

	double deltal[4], deltar[4], A[16];//Assuming cubi B spline
	BSplineLVD(knots, SplineOrder, x, *iend, deltal, deltar, A, dB, nderiv);

	return 0;
}
int BSplineGetBasis(const double x, double * B, double *knots, int nbreaks, int nControls, int SplineOrder, int nderiv)
{
	int i, j, istart, iend, error;
	double Bi[4 * 3];//up to 2nd der of cubic spline

	error = BSplineGetNonZeroBasis(x, Bi, &istart, &iend, knots, nbreaks, nControls, SplineOrder, nderiv);
	if (error)
		return error;

	for (j = 0; j <= nderiv; j++)
	{
		for (i = 0; i < istart; i++)
			B[i + j*nControls] = 0.0;
		for (i = istart; i <= iend; i++)
			B[i + j*nControls] = Bi[(i - istart) + j*SplineOrder];
		for (i = iend + 1; i < nControls; i++)
			B[i + j*nControls] = 0.0;
	}

	return 0;
}
int BSplineGetAllBasis(double *AllB, double *samples, double *BreakPts, int nsamples, int nbreaks, int SplineOrder, const int nderiv, double *AlldB, double *Alld2B)
{
	int nCoeffs = nbreaks + SplineOrder - 2;
	double *B = new double[nCoeffs*(nderiv + 1)];
	double *knots = new double[nCoeffs + SplineOrder];

	BSplineGetKnots(knots, BreakPts, nbreaks, nCoeffs, SplineOrder);

	for (int ii = 0; ii < nsamples; ii++)
	{
		int error = BSplineGetBasis(samples[ii], B, knots, nbreaks, nCoeffs, SplineOrder, nderiv);
		if (error != 0)
		{
			printLOG("Abort with error %d\n", error);
			abort();
		}

		for (int jj = 0; jj < nCoeffs; jj++)
		{
			AllB[ii*nCoeffs + jj] = B[jj];
			if (nderiv >= 1)
				AlldB[ii*nCoeffs + jj] = B[jj + nCoeffs];
			if (nderiv == 2)
				Alld2B[ii*nCoeffs + jj] = B[jj + 2 * nCoeffs];
		}
	}

	delete[]B, delete[]knots;
	return 0;
}

int PrismMST(char *Path, char *PairwiseSyncFilename, int nvideos)
{
	typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_distance_t, int>, boost::property < boost::edge_weight_t, double > > Graph;
	typedef std::pair < int, int >E;

	int v1, v2; double offset;
	char Fname[512];
	bool *avail = new bool[nvideos*nvideos];
	double *TimeOffset = new double[nvideos*nvideos];
	for (int ii = 0; ii < nvideos*nvideos; ii++)
		TimeOffset[ii] = 0, avail[ii] = false;;

	vector<int> uniqueCamId;
	vector<Point3d> sync;
	sprintf(Fname, "%s/%s.txt", Path, PairwiseSyncFilename);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d %lf ", &v1, &v2, &offset) != EOF)
		uniqueCamId.push_back(v1), uniqueCamId.push_back(v2), sync.push_back(Point3d(v1, v2, offset));
	fclose(fp);

	sort(uniqueCamId.begin(), uniqueCamId.end());
	std::vector<int>::iterator it = unique(uniqueCamId.begin(), uniqueCamId.end());
	uniqueCamId.resize(std::distance(uniqueCamId.begin(), it));
	nvideos = (int)uniqueCamId.size();



	for (int ii = 0; ii < (int)sync.size(); ii++)
	{
		int id1, id2;
		for (id1 = 0; id1 < (int)uniqueCamId.size(); id1++)
			if (uniqueCamId[id1] == sync[ii].x)
				break;
		for (id2 = 0; id2 < (int)uniqueCamId.size(); id2++)
			if (uniqueCamId[id2] == sync[ii].y)
				break;

		avail[id1 + id2*nvideos] = true, avail[id2 + id1*nvideos] = true;
		TimeOffset[id1 + id2*nvideos] = sync[ii].z, TimeOffset[id2 + id1*nvideos] = sync[ii].z;
	}


	sprintf(Fname, "%s/timeConstrantoffset.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nvideos; kk++)
	{
		for (int ll = 0; ll < nvideos; ll++)
			fprintf(fp, "%.4f ", TimeOffset[kk + ll*nvideos]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	//Form edges weight based on the consistency of the triplet
	int num_nodes = nvideos, nedges = nvideos*(nvideos - 1) / 2;
	E *edges = new E[nedges];
	double *weightTable = new double[nvideos*nvideos];
	double *weights = new double[nedges];
	for (int ii = 0; ii < nvideos*nvideos; ii++)
		weightTable[ii] = 0;

	int count = 0;
	for (int kk = 0; kk < nvideos - 1; kk++)
	{
		for (int ll = kk + 1; ll < nvideos; ll++)
		{
			edges[count] = E(kk, ll);
			weights[count] = 0.0;
			int nvalids = 0;
			for (int jj = 0; jj < nvideos; jj++)
			{
				if (jj == ll || jj == kk)
					continue;
				if (avail[kk + jj*nvideos] == false || avail[ll + jj*nvideos] == false || avail[kk + ll*nvideos] == false)
					continue;
				if (jj >= ll) //kl = kj-lj
					weights[count] += abs(TimeOffset[kk + jj*nvideos] - TimeOffset[ll + jj*nvideos] - TimeOffset[kk + ll*nvideos]);
				else if (jj <= kk) //kl = -jk + jl
					weights[count] += abs(-TimeOffset[jj + kk*nvideos] + TimeOffset[jj + ll*nvideos] - TimeOffset[kk + ll*nvideos]);
				else //kl = kj+jl
					weights[count] += abs(TimeOffset[kk + jj*nvideos] + TimeOffset[jj + ll*nvideos] - TimeOffset[kk + ll*nvideos]);
				nvalids++;
			}
			if (nvalids > 0)
				weights[count] = weights[count] / nvalids;
			else
				weights[count] = 9e9;
			weightTable[kk + ll*nvideos] = weights[count], weightTable[ll + kk*nvideos] = weights[count];
			count++;
		}
	}

	sprintf(Fname, "%s/weightTable.txt", Path);	fp = fopen(Fname, "w+");
	for (int kk = 0; kk < nvideos; kk++)
	{
		for (int ll = 0; ll < nvideos; ll++)
			fprintf(fp, "%.4f ", weightTable[kk + ll*nvideos]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	Graph g(edges, edges + sizeof(E)*nedges / sizeof(E), weights, num_nodes);
	boost::property_map<Graph, boost::edge_weight_t>::type weightmap = get(boost::edge_weight, g);
	std::vector < boost::graph_traits < Graph >::vertex_descriptor >p(boost::num_vertices(g));

	boost::prim_minimum_spanning_tree(g, &p[0]);

	sprintf(Fname, "%s/MST_Sync.txt", Path);	fp = fopen(Fname, "w+");
	for (std::size_t i = 0; i != p.size(); ++i)
	{
		if (p[i] != i)
		{
			std::cout << "parent[" << i << "] = " << p[i] << std::endl;
			fprintf(fp, "%d %d\n", p[i], i);
		}
		else
		{
			std::cout << "parent[" << i << "] = no parent" << std::endl;
			fprintf(fp, "%d %d\n", i, i);
		}
	}
	fclose(fp);

	delete[]weights, delete[]weightTable, delete[]TimeOffset, delete[]edges;

	return 0;
}
int AssignOffsetFromMST(char *Path, char *PairwiseSyncFilename, int nvideos, double *OrderedOffset, double *fps)
{
	bool createdMem = false;
	if (OrderedOffset == NULL)
		OrderedOffset = new double[nvideos], createdMem = true;

	char Fname[512];
	vector<Point2i> ParentChild;

	sprintf(Fname, "%s/MST_Sync.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	int parent, child;
	while (fscanf(fp, "%d %d ", &parent, &child) != EOF)
		ParentChild.push_back(Point2i(parent, child));
	fclose(fp);
	remove(Fname);

	vector<int> uniqueCamId;
	vector<Point3d> sync;
	double *Offset = new double[nvideos*nvideos], t;
	sprintf(Fname, "%s/%s.txt", Path, PairwiseSyncFilename);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot open %s\n", Fname);
		return 1;
	}
	while (fscanf(fp, "%d %d %lf", &parent, &child, &t) != EOF)
		uniqueCamId.push_back(parent), uniqueCamId.push_back(child), sync.push_back(Point3d(parent, child, t));
	fclose(fp);

	sort(uniqueCamId.begin(), uniqueCamId.end());
	std::vector<int>::iterator it = unique(uniqueCamId.begin(), uniqueCamId.end());
	uniqueCamId.resize(std::distance(uniqueCamId.begin(), it));
	nvideos = (int)uniqueCamId.size();

	for (int ii = 0; ii < (int)sync.size(); ii++)
	{
		int id1, id2;
		for (id1 = 0; id1 < (int)uniqueCamId.size(); id1++)
			if (uniqueCamId[id1] == sync[ii].x)
				break;
		for (id2 = 0; id2 < (int)uniqueCamId.size(); id2++)
			if (uniqueCamId[id2] == sync[ii].y)
				break;
		Offset[id1 + id2*nvideos] = sync[ii].z, Offset[id2 + id1*nvideos] = sync[ii].z;
	}

	//OrderedOffset[ParentChild[0].x] = 0; fixing the parent is invalid for multi-group sync
	int ncollected = 1;
	vector<int>currentParent, currentChild, tcurrentChild;
	currentParent.push_back(ParentChild[0].x);
	while (ncollected != nvideos)
	{
		//search for children node
		for (int jj = 0; jj < currentParent.size(); jj++)
		{
			for (int ii = 1; ii < ParentChild.size(); ii++)
			{
				if (ParentChild[ii].x == currentParent[jj])
				{
					tcurrentChild.push_back(ParentChild[ii].y);
					ncollected++;
				}
			}

			//assign offset to children
			for (int ii = 0; ii < tcurrentChild.size(); ii++)
			{
				int c = tcurrentChild[ii], p = currentParent[jj], rc = uniqueCamId[c], rp = uniqueCamId[p];
				if (c > p)
					OrderedOffset[rc] = OrderedOffset[rc] + Offset[p + c * nvideos];
				else if (c < p)
					OrderedOffset[rc] = OrderedOffset[rc] - Offset[p + c * nvideos];
				else
					printLOG("Error: parent is child!\n");
			}

			for (int ii = 0; ii < tcurrentChild.size(); ii++)
				currentChild.push_back(tcurrentChild[ii]);
			tcurrentChild.clear();
		}

		//replace parent with children
		currentParent.clear();
		currentParent = currentChild;
		currentChild.clear();
	}

	//Does not work for multi-group sync
	//Find the video started earliest (has largest offset value) and that it as reference
	double earliest = -999.9;
	for (int ii = 0; ii < nvideos; ii++)
		if (OrderedOffset[ii] > earliest)
			earliest = OrderedOffset[ii];

	//for (int ii = 0; ii < nvideos; ii++)
	//	OrderedOffset[ii] -= earliest;

	//Write results:
	sprintf(Fname, "%s/F%s.txt", Path, PairwiseSyncFilename);	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < (int)uniqueCamId.size(); ii++)
		if (fps == 0)
			fprintf(fp, "%d %.3f\n", uniqueCamId[ii], OrderedOffset[uniqueCamId[ii]]);
		else
			fprintf(fp, "%d %.3f\n", uniqueCamId[ii], OrderedOffset[uniqueCamId[ii]] * fps[uniqueCamId[ii]]);
	fclose(fp);


	delete[]Offset;
	if (createdMem)
		delete[]OrderedOffset;

	return 0;
}

void DynamicTimeWarping3Step(Mat pM, vector<int>&pp, vector<int> &qq)
{
	int ii, jj;
	int nrows = pM.rows, ncols = pM.cols;

	Mat DMatrix(nrows + 1, ncols + 1, CV_64F);
	for (ii = 0; ii < nrows + 1; ii++)
		DMatrix.at<double>(ii, 0) = 10.0e16;
	for (ii = 0; ii < ncols + 1; ii++)
		DMatrix.at<double>(0, ii) = 10.0e16;
	DMatrix.at<double>(0, 0) = 0.0;
	for (jj = 0; jj < nrows; jj++)
		for (ii = 0; ii < ncols; ii++)
			DMatrix.at<double>(jj + 1, ii + 1) = pM.at<double>(jj, ii);

	// traceback
	Mat phi = Mat::zeros(nrows, ncols, CV_32S);

	int id[3]; double val[3];
	for (ii = 0; ii < nrows; ii++)
	{
		for (jj = 0; jj < ncols; jj++)
		{
			double dd = DMatrix.at<double>(ii, jj);
			//find min of sub block
			val[0] = DMatrix.at<double>(ii, jj); id[0] = 0;
			val[1] = DMatrix.at<double>(ii, jj + 1); id[1] = 1;
			val[2] = DMatrix.at<double>(ii + 1, jj); id[2] = 2;

			Quick_Sort_Double(val, id, 0, 2);
			DMatrix.at<double>(ii + 1, jj + 1) += val[0];
			phi.at<int>(ii, jj) = id[0] + 1;
			//cout << phi << endl << endl;
		}
	}

	//Traceback from top left
	{
		int jj = nrows - 1;
		int ii = ncols - 1;
		vector<int>p, q;
		p.reserve(max(nrows, ncols));
		q.reserve(max(nrows, ncols));
		p.push_back(ii);
		q.push_back(jj);
		while (ii > 0 && jj > 0)
		{
			int tb = phi.at<int>(ii, jj);

			if (tb == 1)
				ii = ii - 1, jj = jj - 1;
			else if (tb == 2)
				ii = ii - 1;
			else if (tb == 3)
				jj = jj - 1;
			else
			{
				printLOG("Problem in finding Path of DTW\n");
				abort();
			}
			p.push_back(ii);
			q.push_back(jj);
		}

		// Strip off the edges of the D matrix before returning
		//DMatrix = D(2:(r + 1), 2 : (c + 1));

		//flip the vector, substract 1 and store
		int nele = (int)q.size();
		pp.reserve(nele); qq.reserve(nele);
		for (int ii = 0; ii < nele; ii++)
			pp.push_back(p[nele - 1 - ii]), qq.push_back(q[nele - 1 - ii]);
	}
	return;
}
void DynamicTimeWarping5Step(Mat pM, vector<int>&pp, vector<int> &qq)
{
	int ii, jj;
	int nrows = pM.rows, ncols = pM.cols;

	Mat DMatrix(nrows + 1, ncols + 1, CV_64F);
	for (ii = 0; ii < nrows + 1; ii++)
		DMatrix.at<double>(ii, 0) = 10.0e16;
	for (ii = 0; ii < ncols + 1; ii++)
		DMatrix.at<double>(0, ii) = 10.0e16;
	DMatrix.at<double>(0, 0) = 0.0;
	for (jj = 0; jj < nrows; jj++)
		for (ii = 0; ii < ncols; ii++)
			DMatrix.at<double>(jj + 1, ii + 1) = pM.at<double>(jj, ii);

	// traceback
	Mat phi = Mat::zeros(nrows + 1, ncols + 1, CV_32S);

	int id[5]; double val[5];
	//Scale the 'longer' steps to discourage skipping ahead
	int kk1 = 2, kk2 = 1;
	for (ii = 1; ii < nrows + 1; ii++)
	{
		for (jj = 1; jj < ncols + 1; jj++)
		{
			double dd = DMatrix.at<double>(ii, jj);
			//find min of sub block
			val[0] = DMatrix.at<double>(ii - 1, jj - 1) + dd; id[0] = 0;
			val[1] = DMatrix.at<double>(max(0, ii - 2), jj - 1) + dd*kk1; id[1] = 1;
			val[2] = DMatrix.at<double>(ii - 1, max(0, jj - 2)) + dd*kk1; id[2] = 2;
			val[3] = DMatrix.at<double>(ii - 1, jj) + dd*kk2; id[3] = 3;
			val[4] = DMatrix.at<double>(ii, jj - 1) + dd*kk2; id[4] = 4;

			Quick_Sort_Double(val, id, 0, 4);
			DMatrix.at<double>(ii, jj) = val[0];
			phi.at<int>(ii, jj) = id[0] + 1;
			//cout << phi << endl << endl;
		}
	}
	//cout << phi << endl << endl;
	//Traceback from top left
	jj = nrows;
	ii = ncols;
	vector<int>p, q;
	p.reserve(max(nrows, ncols));
	q.reserve(max(nrows, ncols));
	p.push_back(ii);
	q.push_back(jj);
	while (ii > 1 && jj > 1)
	{
		int tb = phi.at<int>(ii, jj);

		if (tb == 1)
			ii = ii - 1, jj = jj - 1;
		else if (tb == 2)
			ii = ii - 2, jj = jj - 1;
		else if (tb == 3)
			ii = ii - 1, jj = jj - 2;
		else if (tb == 4)
			ii = ii - 1;
		else if (tb == 5)
			jj = jj - 1;
		else
		{
			printLOG("Problem in finding Path of DTW\n");
			abort();
		}
		p.push_back(ii);
		q.push_back(jj);
	}

	// Strip off the edges of the D matrix before returning
	//DMatrix = D(2:(r + 1), 2 : (c + 1));

	//flip the vector, substract 1 and store
	int nele = (int)q.size();
	pp.reserve(nele); qq.reserve(nele);
	for (int ii = 0; ii < nele; ii++)
		pp.push_back(p[nele - 1 - ii] - 1), qq.push_back(q[nele - 1 - ii] - 1);

	return;
}
