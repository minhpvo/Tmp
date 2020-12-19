#if !defined(MATHULTI_H )
#define MATHULTI_H
#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdarg.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>

#include "../DataStructure.h"
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace cv;
using namespace std;

using namespace Eigen;
using Eigen::Matrix;
using Eigen::Dynamic;

// Convenience types
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdr;
template<typename T>	Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> make_map_cv2eigen(cv::Mat &mat)
{
	return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mat.ptr<T>(), mat.rows, mat.cols);
}
template<typename DerivedM>	cv::Mat	make_map_eigen2cv(Eigen::PlainObjectBase<DerivedM> &mat)
{
	return cv::Mat_<typename DerivedM::Scalar>(mat.data(), mat.rows(), mat.cols());
}

template <typename T1, typename T2>T2 TruncateCast(const T1 value)
{
	return std::min(static_cast<T1>(std::numeric_limits<T2>::max()), std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value));
}

template<class T> struct index_cmp
{
	// Comparison struct used by sort
	// http://bytes.com/topic/c/answers/132045-sort-get-index

	index_cmp(const T arr) : arr(arr) {}
	bool operator()(const size_t a, const size_t b) const
	{
		return arr[a] < arr[b];
	}
	const T arr;
};
template< class T >void myReorder(std::vector<T> & unordered, std::vector<size_t> const & index_map, std::vector<T> & ordered)
{
	// This implementation is O(n), but also uses O(n) extra memory
	// copy for the reorder according to index_map, because unsorted may also be sorted
	std::vector<T> copy = unordered;
	ordered.resize(index_map.size());
	for (int i = 0; i < index_map.size(); i++)
		ordered[i] = copy[index_map[i]];
}
template <class T>void SortWithIndex(std::vector<T> & unsorted, std::vector<T> & sorted, std::vector<size_t> & index_map)
{
	// Act like matlab's [Y,I] = SORT(X)
	// Input:
	//   unsorted  unsorted vector
	// Output:
	//   sorted     sorted vector, allowed to be same as unsorted
	//   index_map  an index map such that sorted[i] = unsorted[index_map[i]]

	// Original unsorted index map
	index_map.resize(unsorted.size());
	for (size_t i = 0; i < unsorted.size(); i++)
		index_map[i] = i;

	// Sort the index map, using unsorted for comparison
	sort(index_map.begin(), index_map.end(), index_cmp<std::vector<T>& >(unsorted));
	sorted.resize(unsorted.size());
	myReorder(unsorted, index_map, sorted);
}

//Math
void inline RayPlaneIntersection(double *rayDir, double *root, double *plane, double *intersect);
double inline Area3DTriangle(double *A, double *B, double *C);
bool PointInTriangle(Point2f X1, Point2f X2, Point2f X3, Point2f X);
void dec2bin(int dec, int*bin, int num_bin);
double nChoosek(int n, int k);
double nPermutek(int n, int k);
int MyFtoI(double W);
bool IsNumber(double x);
bool IsFiniteNumber(double x);
double UniformNoise(double High, double Low);
double gaussian_noise(double mean, double std);

template <typename Type>double Distance2D(Point_<Type> pt_1, Point_<Type> pt_2)
{
	double dist = pow(double(pt_1.x - pt_2.x), 2) + pow(double(pt_1.y - pt_2.y), 2);;
	return std::sqrt(dist);
}
template <typename Type>double Distance3D(Point3_<Type> pt_1, Point3_<Type> pt_2)
{
	double dist = pow(double(pt_1.x - pt_2.x), 2) + pow(double(pt_1.y - pt_2.y), 2) + pow(double(pt_1.z - pt_2.z), 2);
	return std::sqrt(dist);
}
double Distance3D(double *X, double * Y);
double MinDistanceTwoLines(double *P0, double *u, double *Q0, double *v, double &s, double &t);

void normalize(double *x, int dim = 3);
template <typename Type>void normalize(Point3_<Type>& vect)
{
	double dist = vect.x*vect.x + vect.y*vect.y + vect.z*vect.z;
	dist = sqrt(dist);
	vect.x /= dist;
	vect.y /= dist;
	vect.z /= dist;
}

Point3d ProductPoint3d(Point3d X, Point3d Y);
Point3d DividePoint3d(Point3d X, Point3d Y);

template <typename T>
T Median(const std::vector<T>& elems) {
	CHECK(!elems.empty());

	const size_t mid_idx = elems.size() / 2;

	std::vector<T> ordered_elems = elems;
	std::nth_element(ordered_elems.begin(), ordered_elems.begin() + mid_idx,
		ordered_elems.end());

	if (elems.size() % 2 == 0) {
		const T mid_element1 = ordered_elems[mid_idx];
		const T mid_element2 = *std::max_element(ordered_elems.begin(),
			ordered_elems.begin() + mid_idx);
		return (mid_element1 + mid_element2) / 2.0;
	}
	else {
		return ordered_elems[mid_idx];
	}
}
double L1norm(vector<double>A);
double L2norm(double *A, int dim);

template <typename Type>inline bool IsValid3D(Point3_<Type>& point)
{
	if (std::abs(point.x) + std::abs(point.y) + std::abs(point.z) > LIMIT3D)
		return true;
	else
		return false;
}

float MeanArray(float *data, int length);
double MeanArray(double *data, int length);
Point3d MeanArray(Point3d *data, int length);
double VarianceArray(double *data, int length, double mean = NULL);
Point3d VarianceArray(Point3d *data, int length);
double MeanArray(vector<double>&data);
double VarianceArray(vector<double>&data, double mean = NULL);
double MedianArray(vector<double> &data);

double dotProduct(double *x, double *y, int dim = 3);
double dotProduct(float *x, float *y, int dim = 3);
double norm_dot_product(double *x, double *y, int dim = 3);
void cross_product(double *x, double *y, double *xy);

void conv(float *A, int lenA, float *B, int lenB, float *C);
void conv(double *A, int lenA, double *B, int lenB, double *C);
void ZNCC1D(float *A, const int dimA, float *B, const int dimB, float *Result, float *nB = NULL);
void ZNCC1D(double *A, int Asize, double *B, int Bsize, double *Result);
void XCORR1D(float *s, const int sdim, float *b, const int bdim, float *res);

void mat_invert(double* mat, double* imat, int dims = 3);
void mat_invert(float* mat, float* imat, int dims = 3);
void mat_mul(float *aa, float *bb, float *out, int rowa, int col_row, int colb);
void mat_mul(double *aa, double *bb, double *out, int rowa, int col_row, int colb);
void mat_add(double *aa, double *bb, double* cc, int row, int col, double scale_a = 1.0, double scale_b = 1.0);
void mat_subtract(double *aa, double *bb, double* cc, int row, int col, double scale_a = 1.0, double scale_b = 1.0);
void mat_transpose(double *in, double *out, int row_in, int col_in);
void mat_mul_symetric(double *A, double *B, int row, int column);
void mat_add_symetric(double *A, double * B, double *C, int row, int column);
void mat_completeSym(double *mat, int size, bool upper = true);

template <class myType>void RemoveEleFromArray(myType *Array, int neles, int eleID)
{
	for (int i = eleID; i < neles - 1; i++)
		Array[i] = Array[i + 1];
}

void LS_Solution_Double(double *lpA, double *lpB, int m, int n);
void QR_Solution_Double(double *lpA, double *lpB, int m, int n);
void Quick_Sort_Double(double * A, int *B, int low, int high);
void Quick_Sort_Float(float * A, int *B, int low, int high);
void Quick_Sort_Int(int * A, int *B, int low, int high);

double SimpsonThreeEightIntegration(double *y, double step, int npts);

bool in_polygon(double u, double v, Point2d *vertex, int num_vertex);

void ConvertToHeatMap(double *Map, unsigned char *ColorMap, int width, int height, bool *mask = 0);
void RescaleMat(double *mat, double &orgmin, double &orgmax, double nmin, double nmax, int length);

template <class myType>void Get_Sub_Mat(myType *srcMat, myType *dstMat, int srcWidth, int srcHeight, int dstWidth, int startCol, int startRow)
{
	int ii, jj;

	for (jj = startRow; jj < startRow + srcHeight; jj++)
		for (ii = startCol; ii < startCol + srcWidth; ii++)
			dstMat[ii - startCol + (jj - startRow)*dstWidth] = srcMat[ii + jj*srcWidth];

	return;
}
template <class myType>void Set_Sub_Mat(myType *srcMat, myType *dstMat, int srcWidth, int srcHeight, int dstWidth, int startCol, int startRow)
{
	int ii, jj;

	for (jj = 0; jj < srcHeight; jj++)
	{
		for (ii = 0; ii < srcWidth; ii++)
		{
			dstMat[ii + startCol + (jj + startRow)*dstWidth] = srcMat[ii + jj*srcWidth];
		}
	}

	return;
}
template <class m_Type> class m_TemplateClass_1
{
public:
	void Quick_Sort(m_Type* A, int *B, int low, int high);
	void QR_Solution(m_Type *lpA, m_Type *lpB, int m, int n);
	void QR_Solution_2(m_Type *lpA, m_Type *lpB, int m, int n, int k);
};
template <class m_Type> void m_TemplateClass_1<m_Type>::Quick_Sort(m_Type* A, int *B, int low, int high)
//A: array to be sorted (from min to max); B: index of the original array; low and high: array range
//After sorting, A: sorted array; B: re-sorted index of the original array, e.g., the m-th element of
// new A[] is the original n-th element in old A[]. B[m-1]=n-1;
//B[] is useless for most sorting, it is added here for the special application in this program.  
{
	m_Type A_pivot, A_S;
	int B_pivot, B_S;
	int scanUp, scanDown;
	int mid;
	if (high - low <= 0)
		return;
	else if (high - low == 1)
	{
		if (A[high] < A[low])
		{
			//	Swap(A[low],A[high]);
			//	Swap(B[low],B[high]);
			A_S = A[low];
			A[low] = A[high];
			A[high] = A_S;
			B_S = B[low];
			B[low] = B[high];
			B[high] = B_S;
		}
		return;
	}
	mid = (low + high) / 2;
	A_pivot = A[mid];
	B_pivot = B[mid];

	//	Swap(A[mid],A[low]);
	//	Swap(B[mid],B[low]);
	A_S = A[mid];
	A[mid] = A[low];
	A[low] = A_S;
	B_S = B[mid];
	B[mid] = B[low];
	B[low] = B_S;

	scanUp = low + 1;
	scanDown = high;
	do
	{
		while (scanUp <= scanDown && A[scanUp] <= A_pivot)
			scanUp++;
		while (A_pivot < A[scanDown])
			scanDown--;
		if (scanUp < scanDown)
		{
			//	Swap(A[scanUp],A[scanDown]);
			//	Swap(B[scanUp],B[scanDown]);
			A_S = A[scanUp];
			A[scanUp] = A[scanDown];
			A[scanDown] = A_S;
			B_S = B[scanUp];
			B[scanUp] = B[scanDown];
			B[scanDown] = B_S;
		}
	} while (scanUp < scanDown);

	A[low] = A[scanDown];
	B[low] = B[scanDown];
	A[scanDown] = A_pivot;
	B[scanDown] = B_pivot;
	if (low < scanDown - 1)
		Quick_Sort(A, B, low, scanDown - 1);
	if (scanDown + 1 < high)
		Quick_Sort(A, B, scanDown + 1, high);
}
template <class m_Type> void m_TemplateClass_1<m_Type>::QR_Solution(m_Type *lpA, m_Type *lpB, int m, int n)
{
	int ii, jj, mm, kk;
	m_Type t, d, alpha, u;
	m_Type *lpC = new m_Type[n];
	m_Type *lpQ = new m_Type[m*m];

	for (ii = 0; ii < m; ii++)
	{
		for (jj = 0; jj < m; jj++)
		{
			*(lpQ + ii*m + jj) = (m_Type)0;
			if (ii == jj)
				*(lpQ + ii*m + jj) = (m_Type)1;
		}
	}

	for (kk = 0; kk < n; kk++)
	{
		u = (m_Type)0;
		for (ii = kk; ii<m; ii++)
		{
			if (fabs(*(lpA + ii*n + kk))>u)
				u = (m_Type)(fabs(*(lpA + ii*n + kk)));
		}

		alpha = (m_Type)0;
		for (ii = kk; ii < m; ii++)
		{
			t = *(lpA + ii*n + kk) / u;
			alpha = alpha + t*t;
		}
		if (*(lpA + kk*n + kk) >(m_Type)0)
			u = -u;
		alpha = (m_Type)(u*sqrt(alpha));
		u = (m_Type)(sqrt(2.0*alpha*(alpha - *(lpA + kk*n + kk))));
		if (fabs(u) > 1e-8)
		{
			*(lpA + kk*n + kk) = (*(lpA + kk*n + kk) - alpha) / u;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = *(lpA + ii*n + kk) / u;
			for (jj = 0; jj < m; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpQ + mm*m + jj));
				for (ii = kk; ii < m; ii++)
					*(lpQ + ii*m + jj) = *(lpQ + ii*m + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			for (jj = kk + 1; jj < n; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpA + mm*n + jj));
				for (ii = kk; ii < m; ii++)
					*(lpA + ii*n + jj) = *(lpA + ii*n + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			*(lpA + kk*n + kk) = alpha;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = (m_Type)0;
		}
	}
	for (ii = 0; ii < m - 1; ii++)
	{
		for (jj = ii + 1; jj < m; jj++)
		{
			t = *(lpQ + ii*m + jj);
			*(lpQ + ii*m + jj) = *(lpQ + jj*m + ii);
			*(lpQ + jj*m + ii) = t;
		}
	}
	//Solve the equation
	for (ii = 0; ii < n; ii++)
	{
		d = (m_Type)0;
		for (jj = 0; jj < m; jj++)
			d = d + *(lpQ + jj*m + ii)*(*(lpB + jj));
		*(lpC + ii) = d;
	}
	*(lpB + n - 1) = *(lpC + n - 1) / (*(lpA + (n - 1)*n + n - 1));
	for (ii = n - 2; ii >= 0; ii--)
	{
		d = (m_Type)0;
		for (jj = ii + 1; jj < n; jj++)
			d = d + *(lpA + ii*n + jj)*(*(lpB + jj));
		*(lpB + ii) = (*(lpC + ii) - d) / (*(lpA + ii*n + ii));
	}

	delete[]lpQ;
	delete[]lpC;
	return;
}
template <class m_Type> void m_TemplateClass_1<m_Type>::QR_Solution_2(m_Type *lpA, m_Type *lpB, int m, int n, int k)
{
	int ii, jj, mm, kk;
	m_Type t, d, alpha, u;
	m_Type *lpC = new m_Type[n];
	m_Type *lpQ = new m_Type[m*m];

	for (ii = 0; ii < m; ii++)
	{
		for (jj = 0; jj < m; jj++)
		{
			*(lpQ + ii*m + jj) = (m_Type)0;
			if (ii == jj)
				*(lpQ + ii*m + jj) = (m_Type)1;
		}
	}

	for (kk = 0; kk < n; kk++)
	{
		u = (m_Type)0;
		for (ii = kk; ii<m; ii++)
		{
			if (fabs(*(lpA + ii*n + kk))>u)
				u = (m_Type)(fabs(*(lpA + ii*n + kk)));
		}

		alpha = (m_Type)0;
		for (ii = kk; ii < m; ii++)
		{
			t = *(lpA + ii*n + kk) / u;
			alpha = alpha + t*t;
		}
		if (*(lpA + kk*n + kk) >(m_Type)0)
			u = -u;
		alpha = (m_Type)(u*sqrt(alpha));
		u = (m_Type)(sqrt(2.0*alpha*(alpha - *(lpA + kk*n + kk))));
		if (fabs(u) > 1e-8)
		{
			*(lpA + kk*n + kk) = (*(lpA + kk*n + kk) - alpha) / u;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = *(lpA + ii*n + kk) / u;
			for (jj = 0; jj < m; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpQ + mm*m + jj));
				for (ii = kk; ii < m; ii++)
					*(lpQ + ii*m + jj) = *(lpQ + ii*m + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			for (jj = kk + 1; jj < n; jj++)
			{
				t = (m_Type)0;
				for (mm = kk; mm < m; mm++)
					t = t + *(lpA + mm*n + kk)*(*(lpA + mm*n + jj));
				for (ii = kk; ii < m; ii++)
					*(lpA + ii*n + jj) = *(lpA + ii*n + jj) - (m_Type)(2.0*t*(*(lpA + ii*n + kk)));
			}
			*(lpA + kk*n + kk) = alpha;
			for (ii = kk + 1; ii < m; ii++)
				*(lpA + ii*n + kk) = (m_Type)0;
		}
	}
	for (ii = 0; ii < m - 1; ii++)
	{
		for (jj = ii + 1; jj < m; jj++)
		{
			t = *(lpQ + ii*m + jj);
			*(lpQ + ii*m + jj) = *(lpQ + jj*m + ii);
			*(lpQ + jj*m + ii) = t;
		}
	}
	//Solve the equation

	m_Type *lpBB;
	for (mm = 0; mm < k; mm++)
	{
		lpBB = lpB + mm*m;

		for (ii = 0; ii < n; ii++)
		{
			d = (m_Type)0;
			for (jj = 0; jj < m; jj++)
				d = d + *(lpQ + jj*m + ii)*(*(lpBB + jj));
			*(lpC + ii) = d;
		}
		*(lpBB + n - 1) = *(lpC + n - 1) / (*(lpA + (n - 1)*n + n - 1));
		for (ii = n - 2; ii >= 0; ii--)
		{
			d = (m_Type)0;
			for (jj = ii + 1; jj < n; jj++)
				d = d + *(lpA + ii*n + jj)*(*(lpBB + jj));
			*(lpBB + ii) = (*(lpC + ii) - d) / (*(lpA + ii*n + ii));
		}
	}

	delete[]lpQ;
	delete[]lpC;
	return;
}

void SetIntrinisc(CameraData &CamInfo, double *Intrinsic);
void GetIntrinsicFromK(CameraData *AllViewsParas, vector<int> AvailViews);
void GetIntrinsicFromK(CameraData *AllViewsParas, int nviews);
void GetKFromIntrinsic(CameraData *AllViewsParas, vector<int> AvailViews);
void GetKFromIntrinsic(CameraData *AllViewsParas, int nviews);
void GetIntrinsicFromK(CameraData &camera);
void GetKFromIntrinsic(CameraData &camera);

void getTwistFromRT(double *R, double *T, double *twist);
void getRTFromTwist(double *twist, double *R, double *T);
void getrFromR(double *R, double *r);
void getRfromr(double *r, double *R);

void NormalizeQuaternion(double *quat);
void Rotation2Quaternion(double *R, double *q);
void Quaternion2Rotation(double *q, double *R);

void GetrtFromRT(CameraData *AllViewsParas, vector<int> AvailViews);
void GetrtFromRT(CameraData *AllViewsParas, int nviews);
void GetrtFromRT(CameraData &cam);
void GetrtFromRT(double *rt, double *R, double *T);
void GetRTFromrt(CameraData *AllViewsParas, vector<int> AvailViews);
void GetRTFromrt(CameraData *AllViewsParas, int nviews);
void GetRTFromrt(CameraData &camera);
void GetRTFromrt(double *rt, double *R, double *T);

void GetTfromC(CameraData &camInfo);
void GetTfromC(double *R, double *C, double *T);
void GetCfromT(CameraData &camInfo);
void GetCfromT(double *R, double *T, double *C);
void InvertCameraPose(double *R, double *T, double *iR, double *iT);


void GetRCGL(CameraData &camInfo);
void GetRCGL(double *R, double *T, double *Rgl, double *C);

void AssembleRT(double *R, double *T, double *RT, bool GivenCenter = false);
void DesembleRT(double *R, double *T, double *RT);

void AssembleRT_RS(Point2d &uv, CameraData &cam, double *R, double *T);
void AssembleRT_RS(Point2d &uv, double *intrinsic, double *rt, double *wt, double *R, double *T);

void AssembleP(CameraData &camera);
void AssembleP(double *K, double *RT, double *P);
void AssembleP(double *K, double *R, double *T, double *P);

void AssembleP_RS(Point2d uv, double *K, double *R_global, double *T_global, double *wt, double *P);
void AssembleP_RS(Point2d uv, CameraData &cam, double *P);

void CopyCamereInfo(CameraData Src, CameraData &Dst, bool Extrinsic = true);

double DistanceOfTwoPointsSfM(char *Path, int id1, int id2, int id3);

void ComputeInterCamerasPose(double *R1, double *T1, double *R2, double *T2, double *R21, double *T21);

void QuaternionLinearInterp(double *quad1, double *quad2, double *quadi, double u);
int Pose_se_BSplineInterpolation(char *Fname1, char *Fname2, int nsamples, char *Fname3 = 0);
int Pose_se_DCTInterpolation(char *FnameIn, char *FnameOut, int nsamples);

void getRayDir(double *rayDir, double *iK, double *R, double *uv1);
void GetRelativeTransformation(double *R0, double *T0, double *R1, double *T1, double *R1to0, double *T1to0);
void ComputeEpipole(double *F, Point2d &e1, Point2d &e2);

#endif
