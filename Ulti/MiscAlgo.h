#if !defined(MISCALGO_H )
#define MISCALGO_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <complex>
#include <omp.h>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include "MathUlti.h"
using namespace cv;

void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask);
template <class myType>void nonMinimaSuppression1D(myType *src, int nsample, int *MinEle, int &nMinEle, int halfWnd)
{
	int i = 0, minInd = 0, srcCnt = 0, ele;
	nMinEle = 0;
	while (i < nsample)
	{
		if (minInd < i - halfWnd)
			minInd = i - halfWnd;

		ele = min(i + halfWnd, nsample);
		while (minInd <= ele)
		{
			srcCnt++;
			if (src[minInd] < src[i])
				break;
			minInd++;
		}

		if (minInd > ele) // src(i) is a maxima in the search window
		{
			MinEle[nMinEle] = i, nMinEle++; // the loop above suppressed the maximum, so set it back
			minInd = i + 1;
			i += halfWnd;
		}
		i++;
	}

	return;
}
template <class myType>void nonMaximaSuppression1D(myType *src, int nsample, int *MaxEle, int &nMaxEle, int hWind)
{
	myType *src2 = new myType[nsample];
	for (int ii = 0; ii < nsample; ii++)
		src2[ii] = -src[ii];

	nonMinimaSuppression1D(src2, nsample, MaxEle, nMaxEle, hWind);

	return;
}

template<class T> void ComputeDistanceTransform(T* label_image, float *realDT, T mask_th, int width, int height, double *v = NULL, double *z = NULL, double *DTTps = NULL, int *ADTTps = NULL, int *realADT = NULL)
{
	//const double maxLimit = 1.79769e+308, minLimit = 2.22507e-308;
	const float maxLimit = 3.40282e+38, minLimit = 1.17549e-38; 

	bool createMem = false;
	if (v == NULL)
	{
		createMem = true;
		v = new double[width*height];
		z = new double[width*height];
		DTTps = new double[width*height];
		ADTTps = new int[width*height];
		realADT = new int[width*height];
	}

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < width*height; ++i)
	{
		if (label_image[i] < mask_th)
			realDT[i] = maxLimit;
		else
			realDT[i] = float(0);
	}

	/// DT and ADT
	//First PASS (rows)
#pragma omp parallel for
	for (int row = 0; row < height; ++row)
	{
		unsigned int k = 0;
		unsigned int indexpt1 = row*width;
		v[indexpt1] = 0;
		z[indexpt1] = minLimit;
		z[indexpt1 + 1] = maxLimit;
		for (int q = 1; q < width; ++q)
		{
			double sp1 = double(realDT[(indexpt1 + q)] + (q*q));
			unsigned int index2 = indexpt1 + k;
			unsigned int vk = v[index2];
			double s = (sp1 - double(realDT[(indexpt1 + vk)] + (vk*vk))) / double((q - vk) << 1);
			while (s <= z[index2] && k > 0)
			{
				k--;
				index2 = indexpt1 + k;
				vk = v[index2];
				s = (sp1 - double(realDT[(indexpt1 + vk)] + (vk*vk))) / double((q - vk) << 1);
			}
			k++;
			index2 = indexpt1 + k;
			v[index2] = q;
			z[index2] = s;
			z[index2 + 1] = maxLimit;
		}
		k = 0;
		for (int q = 0; q < width; ++q)
		{
			while (z[indexpt1 + k + 1] < q)
				k++;
			unsigned int index2 = indexpt1 + k;
			unsigned int vk = v[index2];
			double tp1 = double(q) - double(vk);
			DTTps[indexpt1 + q] = tp1*tp1 + double(realDT[(indexpt1 + vk)]);
			ADTTps[indexpt1 + q] = indexpt1 + vk;
		}
	}

	//--- Second PASS (columns)
#pragma omp parallel for
	for (int col = 0; col < width; ++col)
	{
		unsigned int k = 0;
		unsigned int indexpt1 = col*height;
		v[indexpt1] = 0;
		z[indexpt1] = minLimit;
		z[indexpt1 + 1] = maxLimit;
		for (int row = 1; row < height; ++row)
		{
			double sp1 = double(DTTps[col + row*width] + (row*row));
			unsigned int index2 = indexpt1 + k;
			unsigned int vk = v[index2];
			double s = (sp1 - double(DTTps[col + vk*width] + (vk*vk))) / double((row - vk) << 1);
			while (s <= z[index2] && k > 0)
			{
				k--;
				index2 = indexpt1 + k;
				vk = v[index2];
				s = (sp1 - double(DTTps[col + vk*width] + (vk*vk))) / double((row - vk) << 1);
			}
			k++;
			index2 = indexpt1 + k;
			v[index2] = row;
			z[index2] = s;
			z[index2 + 1] = maxLimit;
		}
		k = 0;
		for (int row = 0; row < height; ++row)
		{
			while (z[indexpt1 + k + 1] < row)
				k++;
			unsigned int index2 = indexpt1 + k;
			unsigned int vk = v[index2];

			/// Also compute the distance value
			double tp1 = double(row) - double(vk);
			realDT[col + row*width] = float(std::sqrt(tp1*tp1 + DTTps[col + vk*width]));
			realADT[col + row*width] = ADTTps[col + vk*width];
		}
	}

	if (createMem)
	{
		delete[] v;
		delete[] z;
		delete[] DTTps;
		delete[] ADTTps;
		delete[] realADT;
	}

	return;
}

double OverlappingArea(Point2i &tl1, Point2i &br1, Point2i &tl2, Point2i &br2);

void GenerateDCTBasis(int nsamples, double *Basis, double *Weight = NULL);
void GenerateiDCTBasis(double *Basis, int nsamples, double t);

void BSplineFindActiveCtrl(int *ActingID, const double x, double *knots, int nbreaks, int nControls, int SplineOrder, int extraNControls = 2);
int BSplineGetKnots(double *knots, double *BreakLoc, int nbreaks, int nControls, int SplineOrder);
int BSplineGetNonZeroBasis(const double x, double * dB, int * istart, int * iend, double *knots, int nbreaks, int nControls, int SplineOrder, int nderiv);
int BSplineGetBasis(const double x, double * B, double *knots, int nbreaks, int nControls, int SplineOrder, int nderiv = 0);
int BSplineGetAllBasis(double *AllB, double *samples, double *BreakPts, int nsamples, int nbreaks, int SplineOrder, int nderiv = 0, double *AlldB = 0, double *Alld2B = 0);

int PrismMST(char *Path, char *PairwiseSyncFilename, int nvideos);
int AssignOffsetFromMST(char *Path, char *PairwiseSyncFilename, int nvideos, double *OffsetInfo = 0, double *fps = 0);

void DynamicTimeWarping3Step(Mat pM, vector<int>&p, vector<int> &q);
void DynamicTimeWarping5Step(Mat pM, vector<int>&p, vector<int> &q);

class Combination
{
public:
	Combination(int n, int k)
	{
		if (n < 0 || k < 0 || n < k) // normally require n >= k  
		{
			printf("Negative parameter in constructor\n");
			abort();
		}
		this->total_com = Choose(n, k);

		this->n = n;
		this->k = k;
		this->data = new int[k];
		for (int i = 0; i < k; ++i)
			this->data[i] = i;
	} // Combination(n,k)
	~Combination()
	{
		delete[]data;
	}

	Combination(int n, int k, int *a, int la) // Combination from a[]
	{
		if (k != la)
		{
			printf("Array length does not equal k\n");
			abort();
		}
		this->n = n;
		this->k = k;
		this->data = new int[k];
		for (int i = 0; i < la; ++i)
			this->data[i] = a[i];

		if (!this->IsValid())
		{
			printf("Bad value from array\n");
			abort();
		}
	} // Combination(n,k,a)

	void Successor()
	{
		//Minh: let's forget about this case
		//if (this->data.Length == 0 || this->data[0] == this->n - this->k)
		//	return NULL;
		Combination next = Combination(this->n, this->k);

		int i, j;
		for (i = 0; i < this->k; ++i)
			next.data[i] = this->data[i];

		for (i = this->k - 1; i > 0 && next.data[i] == this->n - this->k + i; --i);

		++next.data[i];

		for (j = i; j < this->k - 1; ++j)
			next.data[j + 1] = next.data[j] + 1;

		for (i = 0; i < this->k; ++i)
			this->data[i] = next.data[i];

		return;
	} // Successor()

	Combination First()
	{
		Combination ans = Combination(this->n, this->k);

		for (int i = 0; i < ans.k; ++i)
			ans.data[i] = i;

		return ans;
	} // First()

	static int Choose(int n, int k)
	{
		if (n < 0 || k < 0)
			return 0; //throw new Exception("Invalid negative parameter in Choose()"); 
		if (n < k)
			return 0;  // special case
		if (n == k)
			return 1;

		int delta, iMax, ans;

		if (k < n - k) // ex: Choose(100,3)
		{
			delta = n - k;
			iMax = k;
		}
		else         // ex: Choose(100,97)
		{
			delta = k;
			iMax = n - k;
		}

		ans = delta + 1;

		for (int i = 2; i <= iMax; i++)
			ans = (ans * (delta + i)) / i;

		return ans;
	} // Choose()

	//Return all possible combinations
	void All_Com(int *all_com)
	{
		int n = this->n;
		int k = this->k;

		for (int ii = 0; ii < total_com; ii++)
		{
			for (int jj = 0; jj < k; jj++)
				all_com[ii*k + jj] = this->data[jj];
			this->Successor();
		}

		return;
	}

	// return the mth lexicographic element of combination C(n,k)
	Combination Element(int m)
	{
		int *ans = new int[this->k];

		int a = this->n;
		int b = this->k;
		int x = (Choose(this->n, this->k) - 1) - m; // x is the "dual" of m

		for (int i = 0; i < this->k; ++i)
		{
			ans[i] = LargestV(a, b, x); // largest value v, where v < a and vCb < x    
			x = x - Choose(ans[i], b);
			a = ans[i];
			b = b - 1;
		}

		for (int i = 0; i < this->k; ++i)
			ans[i] = (n - 1) - ans[i];

		return Combination(this->n, this->k, ans, this->k);
	} // Element()

	bool IsValid()
	{
		for (int i = 0; i < this->k; ++i)
		{
			if (this->data[i] < 0 || this->data[i] > this->n - 1)
				return false; // value out of range

			for (int j = i + 1; j < this->k; ++j)
				if (this->data[i] >= this->data[j])
					return false; // duplicate or not lexicographic
		}

		return true;
	} // IsValid()

	// return largest value v where v < a and  Choose(v,b) <= x
	static int LargestV(int a, int b, int x)
	{
		int v = a - 1;

		while (Choose(v, b) > x)
			v--;

		return v;
	} // LargestV()

	int  total_com;
private:
	int n, k;
	int *data;
}; // class Combination
#endif