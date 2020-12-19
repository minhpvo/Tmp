#include "../ImgPro/ImagePro.h"
#include "../Ulti/MathUlti.h"
#include "MatchingTracking.h"

using namespace Eigen;
using namespace cv;
using namespace std;

Eigen::MatrixXi ComputeSIFTDistanceMatrix(Mat& descriptors1, Mat& descriptors2, int nthreads)
{
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(descriptors1.rows, descriptors2.rows);

	int dim = descriptors1.cols;

	if (nthreads > 1)
	{
		omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic,1)
		for (int i1 = 0; i1 < descriptors1.rows; ++i1)
		{
			uchar *ptr1 = descriptors1.ptr<uchar>(i1);
			for (int i2 = 0; i2 < descriptors2.rows; ++i2)
			{
				uchar *ptr2 = descriptors2.ptr<uchar>(i2);
				int dot = 0;
				for (int d = 0; d < dim; d++)
					dot += (int)ptr1[d] * (int)ptr2[d];
				dists(i1, i2) = dot;
			}
		}
	}
	else
	{
		for (int i1 = 0; i1 < descriptors1.rows; ++i1)
		{
			uchar *ptr1 = descriptors1.ptr<uchar>(i1);
			for (int i2 = 0; i2 < descriptors2.rows; ++i2)
			{
				uchar *ptr2 = descriptors2.ptr<uchar>(i2);
				int dot = 0;
				for (int d = 0; d < dim; d++)
					dot += (int)ptr1[d] * (int)ptr2[d];
				dists(i1, i2) = dot;
			}
		}
	}

	return dists;
}
Eigen::MatrixXi ComputeSIFTDistanceMatrix(vector<uchar> &descriptors1, vector<uchar> &descriptors2, int descDim, int nthreads)
{
	int nf1 = descriptors1.size() / descDim, nf2 = descriptors2.size() / descDim;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(nf1, nf2);

	if (nthreads > 1)
	{
		omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic,1)
		for (int i1 = 0; i1 < nf1; ++i1)
		{
			uchar *ptr1 = &descriptors1[i1*descDim];
			for (int i2 = 0; i2 < nf2; ++i2)
			{
				uchar *ptr2 = &descriptors2[i2*descDim];
				int dot = 0;
				for (int d = 0; d < descDim; d++)
					dot += (int)ptr1[d] * (int)ptr2[d];
				dists(i1, i2) = dot;
			}
		}
	}
	else
	{
		for (int i1 = 0; i1 < nf1; ++i1)
		{
			uchar *ptr1 = &descriptors1[i1*descDim];
			for (int i2 = 0; i2 < nf2; ++i2)
			{
				uchar *ptr2 = &descriptors2[i2*descDim];
				int dot = 0;
				for (int d = 0; d < descDim; d++)
					dot += (int)ptr1[d] * (int)ptr2[d];
				dists(i1, i2) = dot;
			}
		}
	}

	return dists;
}
size_t FindBestSIFTMatchesOneWay(const Eigen::MatrixXi& dists, const float max_ratio, const float max_distance, std::vector<Point2i>* matches)
{
	// SIFT descriptor vectors are normalized to length 512.
	const float kDistNorm = 1.0f / (512.0f * 512.0f);

	matches->reserve(3000);
	for (Eigen::MatrixXi::Index i1 = 0; i1 < dists.rows(); ++i1)
	{
		int best_i2 = -1, best_dist = 0, second_best_dist = 0;
		for (Eigen::MatrixXi::Index i2 = 0; i2 < dists.cols(); ++i2)
		{
			const int dist = dists(i1, i2);
			if (dist > best_dist)
			{
				best_i2 = i2;
				second_best_dist = best_dist;
				best_dist = dist;
			}
			else if (dist > second_best_dist)
				second_best_dist = dist;
		}

		// Check if any match found.
		if (best_i2 == -1)
			continue;

		// Check if match distance passes threshold.
		const float best_dist_normed = std::acos(min(kDistNorm * best_dist, 1.0f)); //angle is linear, cos is nonlinear
		if (best_dist_normed > max_distance)
			continue;

		// Check if match passes ratio test. Keep this comparison >= in order to ensure that the case of best == second_best is detected.
		const float second_best_dist_normed = std::acos(min(kDistNorm * second_best_dist, 1.0f));
		if (best_dist_normed >= max_ratio * second_best_dist_normed)
			continue;

		(*matches).push_back(Point2i(i1, best_i2));
	}

	return (*matches).size();
}
vector<Point2i> FindBestSIFTMatches(const Eigen::MatrixXi& dists, const float max_ratio, const float max_distance, const bool cross_check)
{
	std::vector<Point2i> matches12;
	const size_t num_matches12 = FindBestSIFTMatchesOneWay(dists, max_ratio, max_distance, &matches12);

	if (cross_check)
	{
		// SIFT descriptor vectors are normalized to length 512.
		const float kDistNorm = 1.0f / (512.0f * 512.0f);

		vector<Point2i> matches;
		for (size_t i1 = 0; i1 < matches12.size(); ++i1)
		{
			int pi1 = matches12[i1].x, pi2 = matches12[i1].y;
			int best_i1 = -1, best_dist = 0, second_best_dist = 0;
			for (Eigen::MatrixXi::Index i1 = 0; i1 < dists.rows(); ++i1)
			{
				const int dist = dists(i1, pi2);
				if (dist > best_dist)
				{
					best_i1 = i1;
					second_best_dist = best_dist;
					best_dist = dist;
				}
				else if (dist > second_best_dist)
					second_best_dist = dist;
			}

			if (best_i1 == -1)
				continue;

			const float best_dist_normed = std::acos(min(kDistNorm * best_dist, 1.0f));
			if (best_dist_normed > max_distance)
				continue;

			const float second_best_dist_normed = std::acos(min(kDistNorm * second_best_dist, 1.0f));
			if (best_dist_normed >= max_ratio * second_best_dist_normed)
				continue;

			if (best_i1 = pi1)
				matches.push_back(Point2i(best_i1, pi2));
			else
				continue;
		}

		return matches;
	}
	else
		return matches12;
}
vector<Point2i> MatchTwoViewSIFTBruteForce(Mat &descriptors1, Mat &descriptors2, double max_ratio, double max_distance, bool cross_check, int nthreads)
{
	const Eigen::MatrixXi dists = ComputeSIFTDistanceMatrix(descriptors1, descriptors2, nthreads);
	return FindBestSIFTMatches(dists, max_ratio, max_distance, cross_check);
}
vector<Point2i> MatchTwoViewSIFTBruteForce(vector<uchar> &descriptors1, vector<uchar> &descriptors2, int descDim, double max_ratio, double max_distance, bool cross_check, int nthreads)
{
	const Eigen::MatrixXi dists = ComputeSIFTDistanceMatrix(descriptors1, descriptors2, descDim, nthreads);
	return FindBestSIFTMatches(dists, max_ratio, max_distance, cross_check);
}

//TVL1
void cvFlowtoFloat(Mat_<Point2f> &flow, float *fx, float *fy)
{
	int  width = flow.cols, height = flow.rows;
	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			const Point2f u = flow(jj, ii);
			fx[ii + jj*width] = u.x;
			fy[ii + jj*width] = u.y;
		}
	}

	return;
}
void cvFloattoFlow(Mat_<Point2f> &flow, float *fx, float *fy)
{
	int  width = flow.cols, height = flow.rows;
	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			const Point2f u = flow(jj, ii);
			flow(jj, ii).x = fx[ii + (height - 1 - jj)*width];
			flow(jj, ii).y = -fy[ii + (height - 1 - jj)*width];
			//fx[ii + (height - 1 - jj)*width] = u.x;
			//fy[ii + (height - 1 - jj)*width] = -u.y;
		}
	}

	return;
}
void WarpImageFlow(float *flow, unsigned char *wImg21, unsigned char *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic)
{
	int ii, jj, kk, length = width*height;
	double u, v, du, dv, S[3];

	double *Para = new double[length*nchannels];
	for (kk = 0; kk < nchannels; kk++)
		Generate_Para_Spline(Img2 + kk*length, Para + kk*length, width, height, InterpAlgo);

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			du = flow[ii + jj*width], dv = flow[ii + jj*width + length];
			if (removeStatic &&abs(du) < 0.005 && abs(dv) < 0.005)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = (unsigned char)(255);
				continue;
			}

			u = du + ii, v = dv + jj;
			if (u< 1.0 || u > width - 1.0 || v<1.0 || v>height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = (unsigned char)(255);
				continue;
			}

			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(Para + kk*length, width, height, u, v, S, -1, InterpAlgo);
				if (S[0] < 0.0)
					S[0] = 0.0;
				else if (S[0] > 255.0)
					S[0] = 255.0;
				wImg21[ii + jj*width + kk*length] = (unsigned char)((int)(S[0] + 0.5));
			}
		}
	}

	delete[]Para;

	return;
}
void WarpImageFlowDouble(float *flow, double *wImg21, double *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic)
{
	int ii, jj, kk, length = width*height;
	double u, v, du, dv, S[3];

	double *Para = new double[length*nchannels];
	for (kk = 0; kk < nchannels; kk++)
		Generate_Para_Spline(Img2 + kk*length, Para + kk*length, width, height, InterpAlgo);

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			du = flow[ii + jj*width], dv = flow[ii + jj*width + length];
			if (removeStatic &&abs(du) < 0.005 && abs(dv) < 0.005)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = 255;
				continue;
			}

			u = du + ii, v = dv + jj;
			if (u< 1.0 || u > width - 1.0 || v<1.0 || v>height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = 255;
				continue;
			}

			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(Para + kk*length, width, height, u, v, S, -1, InterpAlgo);
				if (S[0] < 0.0)
					S[0] = 0.0;
				else if (S[0] > 255.0)
					S[0] = 255.0;
				wImg21[ii + jj*width + kk*length] = S[0];
			}
		}
	}

	return;
}

//ECC image alignment
static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const Mat& src5, Mat& dst)
{
	CV_Assert(src1.size() == src2.size());
	CV_Assert(src1.size() == src3.size());
	CV_Assert(src1.size() == src4.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (src1.cols * 8));
	CV_Assert(dst.type() == CV_32FC1);

	CV_Assert(src5.isContinuous());


	const float* hptr = src5.ptr<float>(0);

	const float h0_ = hptr[0];
	const float h1_ = hptr[3];
	const float h2_ = hptr[6];
	const float h3_ = hptr[1];
	const float h4_ = hptr[4];
	const float h5_ = hptr[7];
	const float h6_ = hptr[2];
	const float h7_ = hptr[5];

	const int w = src1.cols;


	//create denominator for all points as a block
	Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time of this! otherwise use addWeighted

	//create projected points
	Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;
	divide(hatX_, den_, hatX_);
	Mat hatY_ = -src3*h1_ - src4*h4_ - h7_;
	divide(hatY_, den_, hatY_);


	//instead of dividing each block with den,
	//just pre-devide the block of gradients (it's more efficient)

	Mat src1Divided_;
	Mat src2Divided_;

	divide(src1, den_, src1Divided_);
	divide(src2, den_, src2Divided_);


	//compute Jacobian blocks (8 blocks)

	dst.colRange(0, w) = src1Divided_.mul(src3);//1

	dst.colRange(w, 2 * w) = src2Divided_.mul(src3);//2

	Mat temp_ = (hatX_.mul(src1Divided_) + hatY_.mul(src2Divided_));
	dst.colRange(2 * w, 3 * w) = temp_.mul(src3);//3

	hatX_.release();
	hatY_.release();

	dst.colRange(3 * w, 4 * w) = src1Divided_.mul(src4);//4

	dst.colRange(4 * w, 5 * w) = src2Divided_.mul(src4);//5

	dst.colRange(5 * w, 6 * w) = temp_.mul(src4);//6

	src1Divided_.copyTo(dst.colRange(6 * w, 7 * w));//7

	src2Divided_.copyTo(dst.colRange(7 * w, 8 * w));//8
}
static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const Mat& src5, Mat& dst)
{

	CV_Assert(src1.size() == src2.size());
	CV_Assert(src1.size() == src3.size());
	CV_Assert(src1.size() == src4.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (src1.cols * 3));
	CV_Assert(dst.type() == CV_32FC1);

	CV_Assert(src5.isContinuous());

	const float* hptr = src5.ptr<float>(0);

	const float h0 = hptr[0];//cos(theta)
	const float h1 = hptr[3];//sin(theta)

	const int w = src1.cols;

	//create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
	Mat hatX = -(src3*h1) - (src4*h0);

	//create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
	Mat hatY = (src3*h0) - (src4*h1);


	//compute Jacobian blocks (3 blocks)
	dst.colRange(0, w) = (src1.mul(hatX)) + (src2.mul(hatY));//1

	src1.copyTo(dst.colRange(w, 2 * w));//2
	src2.copyTo(dst.colRange(2 * w, 3 * w));//3
}
static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, Mat& dst)
{

	CV_Assert(src1.size() == src2.size());
	CV_Assert(src1.size() == src3.size());
	CV_Assert(src1.size() == src4.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (6 * src1.cols));

	CV_Assert(dst.type() == CV_32FC1);


	const int w = src1.cols;

	//compute Jacobian blocks (6 blocks)

	dst.colRange(0, w) = src1.mul(src3);//1
	dst.colRange(w, 2 * w) = src2.mul(src3);//2
	dst.colRange(2 * w, 3 * w) = src1.mul(src4);//3
	dst.colRange(3 * w, 4 * w) = src2.mul(src4);//4
	src1.copyTo(dst.colRange(4 * w, 5 * w));//5
	src2.copyTo(dst.colRange(5 * w, 6 * w));//6
}
static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{

	CV_Assert(src1.size() == src2.size());

	CV_Assert(src1.rows == dst.rows);
	CV_Assert(dst.cols == (src1.cols * 2));
	CV_Assert(dst.type() == CV_32FC1);

	const int w = src1.cols;

	//compute Jacobian blocks (2 blocks)
	src1.copyTo(dst.colRange(0, w));
	src2.copyTo(dst.colRange(w, 2 * w));
}
static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{
	/* this functions is used for two types of projections. If src1.cols ==src.cols
	it does a blockwise multiplication (like in the outer product of vectors)
	of the blocks in matrices src1 and src2 and dst
	has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
	(number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.
	The number_of_blocks is equal to the number of parameters we are lloking for
	(i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)
	*/
	CV_Assert(src1.rows == src2.rows);
	CV_Assert((src1.cols % src2.cols) == 0);
	int w;

	float* dstPtr = dst.ptr<float>(0);

	if (src1.cols != src2.cols) {//dst.cols==1
		w = src2.cols;
		for (int i = 0; i < dst.rows; i++) {
			dstPtr[i] = (float)src2.dot(src1.colRange(i*w, (i + 1)*w));
		}
	}

	else {
		CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
		w = src2.cols / dst.cols;
		Mat mat;
		for (int i = 0; i < dst.rows; i++) {

			mat = Mat(src1.colRange(i*w, (i + 1)*w));
			dstPtr[i*(dst.rows + 1)] = (float)pow(norm(mat), 2); //diagonal elements

			for (int j = i + 1; j < dst.cols; j++) { //j starts from i+1
				dstPtr[i*dst.cols + j] = (float)mat.dot(src2.colRange(j*w, (j + 1)*w));
				dstPtr[j*dst.cols + i] = dstPtr[i*dst.cols + j]; //due to symmetry
			}
		}
	}
}
static void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType)
{
	CV_Assert(map_matrix.type() == CV_32FC1);
	CV_Assert(update.type() == CV_32FC1);

	CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
		motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

	if (motionType == MOTION_HOMOGRAPHY)
		CV_Assert(map_matrix.rows == 3 && update.rows == 8);
	else if (motionType == MOTION_AFFINE)
		CV_Assert(map_matrix.rows == 2 && update.rows == 6);
	else if (motionType == MOTION_EUCLIDEAN)
		CV_Assert(map_matrix.rows == 2 && update.rows == 3);
	else
		CV_Assert(map_matrix.rows == 2 && update.rows == 2);

	CV_Assert(update.cols == 1);

	CV_Assert(map_matrix.isContinuous());
	CV_Assert(update.isContinuous());


	float* mapPtr = map_matrix.ptr<float>(0);
	const float* updatePtr = update.ptr<float>(0);


	if (motionType == MOTION_TRANSLATION) {
		mapPtr[2] += updatePtr[0];
		mapPtr[5] += updatePtr[1];
	}
	if (motionType == MOTION_AFFINE) {
		mapPtr[0] += updatePtr[0];
		mapPtr[3] += updatePtr[1];
		mapPtr[1] += updatePtr[2];
		mapPtr[4] += updatePtr[3];
		mapPtr[2] += updatePtr[4];
		mapPtr[5] += updatePtr[5];
	}
	if (motionType == MOTION_HOMOGRAPHY) {
		mapPtr[0] += updatePtr[0];
		mapPtr[3] += updatePtr[1];
		mapPtr[6] += updatePtr[2];
		mapPtr[1] += updatePtr[3];
		mapPtr[4] += updatePtr[4];
		mapPtr[7] += updatePtr[5];
		mapPtr[2] += updatePtr[6];
		mapPtr[5] += updatePtr[7];
	}
	if (motionType == MOTION_EUCLIDEAN) {
		double new_theta = updatePtr[0];
		if (mapPtr[3] > 0)
			new_theta += acos(mapPtr[0]);

		if (mapPtr[3] < 0)
			new_theta -= acos(mapPtr[0]);

		mapPtr[2] += updatePtr[1];
		mapPtr[5] += updatePtr[2];
		mapPtr[0] = mapPtr[4] = (float)cos(new_theta);
		mapPtr[3] = (float)sin(new_theta);
		mapPtr[1] = -mapPtr[3];
	}
}
double findTransformECC(InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix, int motionType, TermCriteria criteria)
{
	//Input images: 1-channel images of CV_8U or CV_32F.
	//warpmat: CV_32F

	Mat src = templateImage.getMat();//template iamge
	Mat dst = inputImage.getMat(); //input image (to be warped)
	Mat map = warpMatrix.getMat(); //warp (transformation)

	const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
	const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

	int paramTemp = 6;//default: affine
	switch (motionType) {
	case MOTION_TRANSLATION:
		paramTemp = 2;
		break;
	case MOTION_EUCLIDEAN:
		paramTemp = 3;
		break;
	case MOTION_HOMOGRAPHY:
		paramTemp = 8;
		break;
	}


	const int numberOfParameters = paramTemp;

	const int ws = src.cols;
	const int hs = src.rows;
	const int wd = dst.cols;
	const int hd = dst.rows;

	Mat Xcoord = Mat(1, ws, CV_32F);
	Mat Ycoord = Mat(hs, 1, CV_32F);
	Mat Xgrid = Mat(hs, ws, CV_32F);
	Mat Ygrid = Mat(hs, ws, CV_32F);

	float* XcoPtr = Xcoord.ptr<float>(0);
	float* YcoPtr = Ycoord.ptr<float>(0);
	int j;
	for (j = 0; j < ws; j++)
		XcoPtr[j] = (float)j;
	for (j = 0; j < hs; j++)
		YcoPtr[j] = (float)j;

	repeat(Xcoord, hs, 1, Xgrid);
	repeat(Ycoord, 1, ws, Ygrid);

	Xcoord.release();
	Ycoord.release();

	Mat templateZM = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
	Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
	Mat imageFloat = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
	Mat imageWarped = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
	Mat allOnes = Mat::ones(hd, wd, CV_8U); //to use it for mask warping
	Mat imageMask = Mat(hs, ws, CV_8U); //to store the final mask

	//gaussian filtering is optional
	src.convertTo(templateFloat, templateFloat.type());
	GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);//is in-place filtering slower?

	dst.convertTo(imageFloat, imageFloat.type());
	GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);

	// needed matrices for gradients and warped gradients
	Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
	Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
	Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
	Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


	// calculate first order image derivatives
	Matx13f dx(-0.5f, 0.0f, 0.5f);

	filter2D(imageFloat, gradientX, -1, dx);
	filter2D(imageFloat, gradientY, -1, dx.t());


	// matrices needed for solving linear equation system for maximizing ECC
	Mat jacobian = Mat(hs, ws*numberOfParameters, CV_32F);
	Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
	Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
	Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
	Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
	Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
	Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

	Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
	Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

	const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
	const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;


	// iteratively update map_matrix
	double rho = -1;
	double last_rho = -termination_eps;
	for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++)
	{

		// warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
		if (motionType != MOTION_HOMOGRAPHY)
		{
			warpAffine(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
			warpAffine(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
			warpAffine(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
			warpAffine(allOnes, imageMask, map, imageMask.size(), maskFlags);
		}
		else
		{
			warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
			warpPerspective(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
			warpPerspective(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
			warpPerspective(allOnes, imageMask, map, imageMask.size(), maskFlags);
		}


		Scalar imgMean, imgStd, tmpMean, tmpStd;
		meanStdDev(imageWarped, imgMean, imgStd, imageMask);
		meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

		subtract(imageWarped, imgMean, imageWarped, imageMask);//zero-mean input
		templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
		subtract(templateFloat, tmpMean, templateZM, imageMask);//zero-mean template

		const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
		const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));

		// calculate jacobian of image wrt parameters
		switch (motionType) {
		case MOTION_AFFINE:
			image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
			break;
		case MOTION_HOMOGRAPHY:
			image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
			break;
		case MOTION_TRANSLATION:
			image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
			break;
		case MOTION_EUCLIDEAN:
			image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
			break;
		}

		// calculate Hessian and its inverse
		project_onto_jacobian_ECC(jacobian, jacobian, hessian);

		hessianInv = hessian.inv();

		const double correlation = templateZM.dot(imageWarped);

		// calculate enhanced correlation coefficiont (ECC)->rho
		last_rho = rho;
		rho = correlation / (imgNorm*tmpNorm);

		// project images into jacobian
		project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection);
		project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


		// calculate the parameter lambda to account for illumination variation
		imageProjectionHessian = hessianInv*imageProjection;
		const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
		const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
		if (lambda_d <= 0.0)
		{
			rho = -1;
			printLOG("The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");
		}
		const double lambda = (lambda_n / lambda_d);

		// estimate the update step delta_p
		error = lambda*templateZM - imageWarped;
		project_onto_jacobian_ECC(jacobian, error, errorProjection);
		deltaP = hessianInv * errorProjection;

		// update warping matrix
		update_warping_matrix_ECC(map, deltaP, motionType);
	}

	// return final correlation coefficient
	return rho;
}
double findTransformECC_Optimized(Mat &templateFloat, Mat &imageFloat, Mat &gradientX, Mat &gradientY, Mat &gradientXWarped, Mat &gradientYWarped, Mat &warpMatrix, int motionType, TermCriteria criteria)
{
	//Input images: CV_32F.
	//warpmat: CV_32F

	/*Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
	Mat imageFloat = Mat(hd, wd, CV_32F);// to store the (smoothed) input image

	//gaussian filtering is optional
	templateFloat.convertTo(templateFloat, templateFloat.type());
	GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);//is in-place filtering slower?

	imageFloat.convertTo(imageFloat, imageFloat.type());
	GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);

	// needed matrices for gradients and warped gradients
	Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
	Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
	Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
	Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


	// calculate first order image derivatives
	Matx13f dx(-0.5f, 0.0f, 0.5f);

	filter2D(imageFloat, gradientX, -1, dx);
	filter2D(imageFloat, gradientY, -1, dx.t());*/


	const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
	const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

	int paramTemp = 6;//default: affine
	switch (motionType) {
	case MOTION_TRANSLATION:
		paramTemp = 2;
		break;
	case MOTION_EUCLIDEAN:
		paramTemp = 3;
		break;
	case MOTION_HOMOGRAPHY:
		paramTemp = 8;
		break;
	}
	const int numberOfParameters = paramTemp;

	const int ws = templateFloat.cols, hs = templateFloat.rows, wd = imageFloat.cols, hd = imageFloat.rows;
	Mat templateZM = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
	Mat imageWarped = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
	Mat allOnes = Mat::ones(hd, wd, CV_8U); //to use it for mask warping
	Mat imageMask = Mat(hs, ws, CV_8U); //to store the final mask


	Mat Xcoord = Mat(1, ws, CV_32F), Ycoord = Mat(hs, 1, CV_32F);
	Mat Xgrid = Mat(hs, ws, CV_32F), Ygrid = Mat(hs, ws, CV_32F);

	float* XcoPtr = Xcoord.ptr<float>(0), *YcoPtr = Ycoord.ptr<float>(0);
	for (int j = 0; j < ws; j++)
		XcoPtr[j] = (float)j;
	for (int j = 0; j < hs; j++)
		YcoPtr[j] = (float)j;

	repeat(Xcoord, hs, 1, Xgrid), repeat(Ycoord, 1, ws, Ygrid);
	Xcoord.release(), Ycoord.release();

	// matrices needed for solving linear equation system for maximizing ECC
	Mat jacobian = Mat(hs, ws*numberOfParameters, CV_32F);
	Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
	Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
	Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
	Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
	Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
	Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

	Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
	Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

	const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
	const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;


	// iteratively update map_matrix
	double rho = -1;
	double last_rho = -termination_eps;
	for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++)
	{

		// warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
		if (motionType != MOTION_HOMOGRAPHY)
		{
			warpAffine(imageFloat, imageWarped, warpMatrix, imageWarped.size(), imageFlags);
			warpAffine(gradientX, gradientXWarped, warpMatrix, gradientXWarped.size(), imageFlags);
			warpAffine(gradientY, gradientYWarped, warpMatrix, gradientYWarped.size(), imageFlags);
			warpAffine(allOnes, imageMask, warpMatrix, imageMask.size(), maskFlags);
		}
		else
		{
			warpPerspective(imageFloat, imageWarped, warpMatrix, imageWarped.size(), imageFlags);
			warpPerspective(gradientX, gradientXWarped, warpMatrix, gradientXWarped.size(), imageFlags);
			warpPerspective(gradientY, gradientYWarped, warpMatrix, gradientYWarped.size(), imageFlags);
			warpPerspective(allOnes, imageMask, warpMatrix, imageMask.size(), maskFlags);
		}


		Scalar imgMean, imgStd, tmpMean, tmpStd;
		meanStdDev(imageWarped, imgMean, imgStd, imageMask);
		meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

		subtract(imageWarped, imgMean, imageWarped, imageMask);//zero-mean input
		templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
		subtract(templateFloat, tmpMean, templateZM, imageMask);//zero-mean template

		const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
		const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));

		// calculate jacobian of image wrt parameters
		switch (motionType) {
		case MOTION_AFFINE:
			image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
			break;
		case MOTION_HOMOGRAPHY:
			image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, warpMatrix, jacobian);
			break;
		case MOTION_TRANSLATION:
			image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
			break;
		case MOTION_EUCLIDEAN:
			image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, warpMatrix, jacobian);
			break;
		}

		// calculate Hessian and its inverse
		project_onto_jacobian_ECC(jacobian, jacobian, hessian);

		hessianInv = hessian.inv();

		const double correlation = templateZM.dot(imageWarped);

		// calculate enhanced correlation coefficiont (ECC)->rho
		last_rho = rho;
		rho = correlation / (imgNorm*tmpNorm);

		// project images into jacobian
		project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection);
		project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


		// calculate the parameter lambda to account for illumination variation
		imageProjectionHessian = hessianInv*imageProjection;
		const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
		const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
		if (lambda_d <= 0.0)
		{
			rho = -1;
			//printLOG("The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");
		}
		const double lambda = (lambda_n / lambda_d);

		// estimate the update step delta_p
		error = lambda*templateZM - imageWarped;
		project_onto_jacobian_ECC(jacobian, error, errorProjection);
		deltaP = hessianInv * errorProjection;

		// update warping matrix
		update_warping_matrix_ECC(warpMatrix, deltaP, motionType);
	}

	// return final correlation coefficient
	return rho;
}

double ComputeZNCCPatch(double *RefPatch, double *TarPatch, int hsubset, int nchannels, double *T)
{
	int i, kk, iii, jjj;

	FILE *fp1, *fp2;
	bool printout = false;
	if (printout)
	{
		fp1 = fopen("C:/temp/src.txt", "w+");
		fp2 = fopen("C:/temp/tar.txt", "w+");
	}

	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	bool createMem = false;
	if (T == NULL)
	{
		createMem = true;
		T = new double[2 * Tlength*nchannels];
	}
	double ZNCC = 0.0;

	int m = 0;
	double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
	for (jjj = 0; jjj < TimgS; jjj++)
	{
		for (iii = 0; iii < TimgS; iii++)
		{
			for (kk = 0; kk < nchannels; kk++)
			{
				i = iii + jjj*TimgS + kk*Tlength;
				T[2 * m] = RefPatch[i], T[2 * m + 1] = TarPatch[i];
				t_f += T[2 * m], t_g += T[2 * m + 1];

				if (printout)
					fprintf(fp1, "%.4f ", T[2 * m]), fprintf(fp2, "%.4f ", T[2 * m + 1]);
				m++;
			}
		}
		if (printout)
		{
			fprintf(fp1, "\n"), fprintf(fp2, "\n");
		}
	}
	if (printout)
	{
		fclose(fp1), fclose(fp2);
	}

	t_f = t_f / m;
	t_g = t_g / m;
	t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
	for (i = 0; i < m; i++)
	{
		t_4 = T[2 * i] - t_f, t_5 = T[2 * i + 1] - t_g;
		t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
	}

	t_2 = sqrt(t_2*t_3);
	if (t_2 < 1e-10)
		t_2 = 1e-10;

	ZNCC = t_1 / t_2; //This is the zncc score
	if (abs(ZNCC) > 1.0)
		ZNCC = 0.0;

	if (createMem)
		delete[]T;

	return ZNCC;
}
double ComputeZNCCImagePatch(Mat &Ref, Mat &Tar, Point2i RefPt, Point2i TarPt, int hsubset, int nchannels, double *T)
{
	int MatType = Ref.type();
	if (MatType != CV_8U && MatType != CV_8UC3)
	{
		printLOG("Note: Current code only support uchar type\n");
		return 0.0;
	}

	int i, j, kk, iii, jjj;
	int RefWidth = Ref.cols, RefHeight = Ref.rows, TarWidth = Tar.cols, TarHeight = Tar.rows;
	if (RefPt.x <= hsubset || RefPt.x >= RefWidth - hsubset || RefPt.y <= hsubset || RefPt.y >= RefHeight - hsubset || TarPt.x <= hsubset || TarPt.x >= TarWidth - hsubset || TarPt.y <= hsubset || TarPt.y >= TarHeight - hsubset)
		return 0.0;

	FILE *fp1, *fp2;
	bool printout = false;
	if (printout)
		fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	bool createMem = false;
	if (T == NULL)
	{
		createMem = true;
		T = new double[2 * Tlength*nchannels];
	}
	double ZNCC = 0.0;

	int m = 0;
	double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
	int RefStep = Ref.step[0], RefeSize = (int)Ref.elemSize(), TarStep = Tar.step[0], TareSize = (int)Tar.elemSize();
	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			for (kk = 0; kk < nchannels; kk++)
			{
				i = RefPt.x + iii, j = RefPt.y + jjj;
				T[2 * m] = (double)(int)Ref.data[RefeSize*i + kk + j*RefStep];

				i = TarPt.x + iii, j = TarPt.y + jjj;
				T[2 * m + 1] = (double)(int)Tar.data[TareSize*i + kk + j*TarStep];

				t_f += T[2 * m], t_g += T[2 * m + 1];

				if (printout)
					fprintf(fp1, "%.4f ", T[2 * m]), fprintf(fp2, "%.4f ", T[2 * m + 1]);
				m++;
			}
		}
		if (printout)
			fprintf(fp1, "\n"), fprintf(fp2, "\n");
	}
	if (printout)
		fclose(fp1), fclose(fp2);

	t_f = t_f / m, t_g = t_g / m;
	t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
	for (i = 0; i < m; i++)
	{
		t_4 = T[2 * i] - t_f, t_5 = T[2 * i + 1] - t_g;
		t_1 += t_4*t_5, t_2 += t_4*t_4, t_3 += t_5*t_5;
	}

	t_2 = sqrt(t_2*t_3);
	if (t_2 < 1e-10)
		t_2 = 1e-10;

	ZNCC = t_1 / t_2; //This is the zncc score
	if (abs(ZNCC) > 1.0)
		ZNCC = 0.0;

	if (createMem)
		delete[]T;

	return ZNCC;
}
double ComputeSSIG(double *Para, int x, int y, int hsubset, int width, int height, int nchannels, int InterpAlgo)
{
	int ii, jj, kk, length = width*height;
	double S[3], ssig = 0.0;

	for (kk = 0; kk < nchannels; kk++)
	{
		for (jj = -hsubset; jj <= hsubset; jj++)
		{
			for (ii = -hsubset; ii <= hsubset; ii++)
			{
				Get_Value_Spline(Para + kk*length, width, height, x + ii, y + jj, S, 0, InterpAlgo);
				ssig += S[1] * S[1] + S[2] * S[2];
			}
		}
	}

	return ssig / (2 * hsubset + 1) / (2 * hsubset + 1);
}
double Compute_AffineHomo(Point2d *From, Point2d *To, int npts, double *Affine, Point2d *sFrom, Point2d *sTo, double *A, double *B)
{
	//To = H*From
	int ii;
	bool createMem = false;
	if (A == NULL)
	{
		sFrom = new Point2d[npts], sTo = new Point2d[npts];
		A = new double[npts * 3], B = new double[npts];
		createMem = true;
	}

	//Normalize all pts.
	Point2d meanTo(0, 0), meanFrom(0, 0);
	for (ii = 0; ii < npts; ii++)
	{
		meanTo.x += To[ii].x, meanTo.y += To[ii].y;
		meanFrom.x += From[ii].x, meanFrom.y += From[ii].y;
	}
	meanTo.x /= npts, meanTo.y /= npts;
	meanFrom.x /= npts, meanFrom.y /= npts;

	Point2d scaleF(0, 0), scaleT(0, 0);
	for (ii = 0; ii < npts; ii++)
	{
		scaleF.x += abs(From[ii].x - meanFrom.x) / npts, scaleF.y += abs(From[ii].y - meanFrom.y) / npts;
		scaleT.x += abs(To[ii].x - meanTo.x) / npts, scaleT.y += abs(To[ii].y - meanTo.y) / npts;
	}
	for (ii = 0; ii < npts; ii++)
	{
		sFrom[ii].x = (From[ii].x - meanFrom.x) / scaleF.x, sFrom[ii].y = (From[ii].y - meanFrom.y) / scaleF.y;
		sTo[ii].x = (To[ii].x - meanTo.x) / scaleT.x, sTo[ii].y = (To[ii].y - meanTo.y) / scaleT.y;
	}

	//solve for row 1
	for (ii = 0; ii < npts; ii++)
		A[3 * ii] = sFrom[ii].x, A[3 * ii + 1] = sFrom[ii].y, A[3 * ii + 2] = 1.0, B[ii] = sTo[ii].x;

	Map<VectorXd> eB(B, npts);
	Map<VectorXd> eAffine1(Affine, 3);
	Map<Matrix < double, Dynamic, Dynamic, RowMajor > > eA(A, npts, 3);
	eAffine1 = eA.jacobiSvd(ComputeThinU | ComputeThinV).solve(eB);


	//solve for row 2
	for (ii = 0; ii < npts; ii++)
		A[3 * ii] = sFrom[ii].x, A[3 * ii + 1] = sFrom[ii].y, A[3 * ii + 2] = 1.0, B[ii] = sTo[ii].y;

	Map<VectorXd> eB2(B, npts);
	Map < Matrix < double, Dynamic, Dynamic, RowMajor > > eA2(A, npts, 3);
	Map<VectorXd> eAffine2(Affine + 3, 3);
	eAffine2 = eA2.jacobiSvd(ComputeThinU | ComputeThinV).solve(eB2);


	//denormalize
	double Tfrom[9] = { 1.0 / scaleF.x, 0.0, -meanFrom.x / scaleF.x, 0.0, 1.0 / scaleF.y, -meanFrom.y / scaleF.y, 0, 0, 1 };
	double Tto[9] = { 1.0 / scaleT.x, 0.0, -meanTo.x / scaleT.x, 0.0, 1.0 / scaleT.y, -meanTo.y / scaleT.y, 0, 0, 1 };
	double affine[9] = { Affine[0], Affine[1], Affine[2], Affine[3], Affine[4], Affine[5], 0, 0, 1 };
	Matrix<double, 3, 3, RowMajor> eTfrom(Tfrom), eTto(Tto), eaffine(affine);
	Matrix<double, 3, 3, RowMajor> denormAffine = eTto.inverse()*eaffine*eTfrom;

	Affine[0] = denormAffine(0, 0), Affine[1] = denormAffine(0, 1), Affine[2] = denormAffine(0, 2);
	Affine[3] = denormAffine(1, 0), Affine[4] = denormAffine(1, 1), Affine[5] = denormAffine(1, 2);

	double error = 0.0, errorx, errory;
	for (ii = 0; ii < npts; ii++)
	{
		errorx = (Affine[0] * From[ii].x + Affine[1] * From[ii].y + Affine[2] - To[ii].x);
		errory = (Affine[3] * From[ii].x + Affine[4] * From[ii].y + Affine[5] - To[ii].y);
		error += errorx*errorx + errory*errory;
	}

	if (createMem)
		delete[]A, delete[]B, delete[]sFrom, delete[]sTo;

	return error / npts;
}
double Compute_AffineHomo(vector<Point2d> &From, vector<Point2d> To, double *Affine, double *A, double *B)
{
	//To = H*From
	int ii, npts = (int)From.size();
	bool createMem = false;
	if (A == NULL)
	{
		A = new double[npts * 3], B = new double[npts];
		createMem = true;
	}

	//Normalize all pts.
	Point2d meanTo(0, 0), meanFrom(0, 0);
	for (ii = 0; ii < npts; ii++)
	{
		meanTo.x += To[ii].x, meanTo.y += To[ii].y;
		meanFrom.x += From[ii].x, meanFrom.y += From[ii].y;
	}
	meanTo.x /= npts, meanTo.y /= npts;
	meanFrom.x /= npts, meanFrom.y /= npts;

	Point2d scaleF(0, 0), scaleT(0, 0);
	vector<Point2d> sFrom(npts), sTo(npts);
	for (ii = 0; ii < npts; ii++)
	{
		scaleF.x += abs(From[ii].x - meanFrom.x) / npts, scaleF.y += abs(From[ii].y - meanFrom.y) / npts;
		scaleT.x += abs(To[ii].x - meanTo.x) / npts, scaleT.y += abs(To[ii].y - meanTo.y) / npts;
	}
	for (ii = 0; ii < npts; ii++)
	{
		sFrom[ii].x = (From[ii].x - meanFrom.x) / scaleF.x, sFrom[ii].y = (From[ii].y - meanFrom.y) / scaleF.y;
		sTo[ii].x = (To[ii].x - meanTo.x) / scaleT.x, sTo[ii].y = (To[ii].y - meanTo.y) / scaleT.y;
	}

	//solve for row 1
	for (ii = 0; ii < npts; ii++)
		A[3 * ii] = sFrom[ii].x, A[3 * ii + 1] = sFrom[ii].y, A[3 * ii + 2] = 1.0, B[ii] = sTo[ii].x;

	Map<VectorXd> eB(B, npts);
	Map<VectorXd> eAffine1(Affine, 3);
	Map < Matrix < double, Dynamic, Dynamic, RowMajor > > eA(A, npts, 3);
	eAffine1 = eA.jacobiSvd(ComputeThinU | ComputeThinV).solve(eB);


	//solve for row 2
	for (ii = 0; ii < npts; ii++)
		A[3 * ii] = sFrom[ii].x, A[3 * ii + 1] = sFrom[ii].y, A[3 * ii + 2] = 1.0, B[ii] = sTo[ii].y;

	Map<VectorXd> eB2(B, npts);
	Map<Matrix <double, Dynamic, Dynamic, RowMajor> > eA2(A, npts, 3);
	Map<VectorXd> eAffine2(Affine + 3, 3);
	eAffine2 = eA2.jacobiSvd(ComputeThinU | ComputeThinV).solve(eB2);


	//denormalize
	double Tfrom[9] = { 1.0 / scaleF.x, 0.0, -meanFrom.x / scaleF.x, 0.0, 1.0 / scaleF.y, -meanFrom.y / scaleF.y, 0, 0, 1 };
	double Tto[9] = { 1.0 / scaleT.x, 0.0, -meanTo.x / scaleT.x, 0.0, 1.0 / scaleT.y, -meanTo.y / scaleT.y, 0, 0, 1 };
	double affine[9] = { Affine[0], Affine[1], Affine[2], Affine[3], Affine[4], Affine[5], 0, 0, 1 };
	Matrix<double, 3, 3, RowMajor> eTfrom(Tfrom), eTto(Tto), eaffine(affine);
	Matrix<double, 3, 3, RowMajor> denormAffine = eTto.inverse()*eaffine*eTfrom;

	Affine[0] = denormAffine(0, 0), Affine[1] = denormAffine(0, 1), Affine[2] = denormAffine(0, 2);
	Affine[3] = denormAffine(1, 0), Affine[4] = denormAffine(1, 1), Affine[5] = denormAffine(1, 2);

	double error = 0.0, errorx, errory;
	for (ii = 0; ii < npts; ii++)
	{
		errorx = (Affine[0] * From[ii].x + Affine[1] * From[ii].y + Affine[2] - To[ii].x);
		errory = (Affine[3] * From[ii].x + Affine[4] * From[ii].y + Affine[5] - To[ii].y);
		error += errorx*errorx + errory*errory;
	}

	if (createMem)
		delete[]A, delete[]B;

	return error / npts;
}

double TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, int nchannels, Point2i &POI, int search_area, double thresh, double *T)
{
	//No interpolation at all, just slide the template around to compute the ZNCC
	int m, i, j, ii, jj, iii, jjj, II, JJ, length = width*height, patternLength = pattern_size*pattern_size;
	double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G;

	Point2d w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	FILE *fp1, *fp2;
	bool printout = false;

	Point2i orgPOI = POI;
	bool createdMem = false;
	if (T == NULL)
		T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels], createdMem = true;

	double zncc = 0.0;
	for (j = -search_area; j <= search_area; j++)
	{
		for (i = -search_area; i <= search_area; i++)
		{
			m = -1;
			t_f = 0.0, t_g = 0.0;

			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					for (int kk = 0; kk < nchannels; kk++)
					{
						jj = Pattern_cen_y + jjj, ii = Pattern_cen_x + iii;
						JJ = orgPOI.y + jjj + j, II = orgPOI.x + iii + i;

						m_F = Pattern[ii + jj*pattern_size + kk*patternLength], m_G = Image[II + JJ*width + kk*length];

						if (printout)
							fprintf(fp1, "%.2f ", m_F), fprintf(fp2, "%.2f ", m_G);

						m++;
						T[2 * m] = m_F, T[2 * m + 1] = m_G;
						t_f += m_F, t_g += m_G;
					}
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			t_f = t_f / (m + 1);
			t_g = t_g / (m + 1);
			t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f;
				t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += t_4*t_5, t_2 += t_4*t_4, t_3 += t_5*t_5;
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;
			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3 > thresh && t_3 > zncc)
			{
				zncc = t_3;
				POI.x = orgPOI.x + i, POI.y = orgPOI.y + j;
			}
			else if (t_3 < -thresh && t_3 < zncc)
			{
				zncc = t_3;
				POI.x = orgPOI.x + i, POI.y = orgPOI.y + j;
			}
		}
	}
	zncc = abs(zncc);

	if (createdMem)
		delete[]T;
	return zncc;
}
int TMatchingCoarse(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int search_area, double thresh, double &zncc, int InterpAlgo, double *InitPara, double *maxZNCC)
{
	//Compute the zncc in a local region (5x5). No iteration is used to solve for shape parameters
	//InitPara: 3x3 homography matrix
	int i, j, ii, jj, iii, jjj, kkk, length = width*height, pattern_length = pattern_size*pattern_size, pjump = search_area > 5 ? 2 : 1;
	double II, JJ, t_1, t_2, t_3, t_4, m_F, m_G, S[6];

	Point2d w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;
	int m;
	double t_f, t_g, t_5, xxx = 0.0, yyy = 0.0;
	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];

	zncc = 0.0;
	for (j = -search_area; j <= search_area; j += pjump)
	{
		for (i = -search_area; i <= search_area; i += pjump)
		{
			m = -1;
			t_f = 0.0, t_g = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					for (kkk = 0; kkk < nchannels; kkk++)
					{
						jj = Pattern_cen_y + jjj;
						ii = Pattern_cen_x + iii;

						if (InitPara == NULL)
						{
							II = (int)(POI.x + 0.5) + iii + i;
							JJ = (int)(POI.y + 0.5) + jjj + j;
						}
						else
						{
							II = (InitPara[0] * iii + InitPara[1] * jjj + InitPara[2]) / (InitPara[6] * iii + InitPara[7] * jjj + InitPara[8]);
							JJ = (InitPara[3] * iii + InitPara[4] * jjj + InitPara[5]) / (InitPara[6] * iii + InitPara[7] * jjj + InitPara[8]);
						}

						Get_Value_Spline(Para + kkk*length, width, height, II, JJ, S, -1, InterpAlgo);

						m_F = Pattern[ii + jj*pattern_size + kkk*pattern_length], m_G = S[0];
						m++;
						T[2 * m] = m_F, T[2 * m + 1] = m_G;
						t_f += m_F, t_g += m_G;

						if (printout)
							fprintf(fp1, "%.2f ", m_G), fprintf(fp2, "%.2f ", m_F);
					}
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			t_f = t_f / (m + 1), t_g = t_g / (m + 1);
			t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f, t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += (t_4*t_5), t_2 += (t_4*t_4), t_3 += (t_5*t_5);
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;
			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3 > thresh && t_3 > zncc)
			{
				zncc = t_3;
				xxx = i, yyy = j;
			}
			else if (t_3 < -thresh && abs(t_3) > zncc)
			{
				zncc = t_3;
				xxx = i, yyy = j;
			}
		}
	}
	if (InitPara != NULL)
		maxZNCC[0] = abs(zncc);

	delete[]T;
	if (zncc > thresh)
	{
		POI.x = (int)(POI.x + 0.5) + xxx;
		POI.y = (int)(POI.y + 0.5) + yyy;
		zncc = abs(zncc);
		return 0;
	}
	else if (zncc < -thresh)
	{
		POI.x = (int)(POI.x + 0.5) + xxx;
		POI.y = (int)(POI.y + 0.5) + yyy;
		zncc = abs(zncc);
		return 1;
	}
	else
		return -1;
}
double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, int nchannels, Point2d &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd)
{
	int i, j, k, m, ii, jj, kk, iii, jjj, iii2, jjj2;
	double II, JJ, iii_n, jjj_n, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, numx, numy, denum, denum2, t_7, m_F, m_G, t_f, t_ff, t_g, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.1;
	int NN[] = { 3, 6, 12, 8 }, P_Jump_Incr[] = { 1, 1, 1, 1 };
	int nn = NN[advanced_tech], _iter = 0, Iter_Max = 20;
	int p_jump, p_jump_0 = 1, p_jump_incr = P_Jump_Incr[advanced_tech];
	int length = width*height, pattern_length = pattern_size*pattern_size;

	double AA[144], BB[12], CC[12];

	bool createMem = false;
	if (Znssd_reqd == NULL)
	{
		createMem = true;
		Znssd_reqd = new double[9 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	}

	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	double p[12], p_best[12];
	for (i = 0; i < 12; i++)
		p[i] = 0.0;

	nn = NN[advanced_tech];
	int pixel_increment_in_subset[] = { 1, 2, 2, 3 };

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	/// Iteration: Begin
	bool Break_Flag = false;
	DIC_Coeff_min = 4.0;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1;
			t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < 144; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < 12; iii++)
				BB[iii] = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					ii = Pattern_cen_x + iii, jj = Pattern_cen_y + jjj;
					if (ii<0 || ii>(width - 1) || jj<0 || jj>(height - 1))
						continue;

					iii2 = iii*iii, jjj2 = jjj*jjj;
					if (advanced_tech == 0)
					{
						II = POI.x + iii + p[0] + p[2] * iii;
						JJ = POI.y + jjj + p[1] + p[2] * jjj;
					}
					else if (advanced_tech == 1)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					}
					else if (advanced_tech == 2)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * iii*jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * iii*jjj;
					}
					else
					{
						denum = 1.0 + p[6] * iii + p[7] * jjj;
						numx = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						numy = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
						II = numx / denum;
						JJ = numy / denum;
					}

					if (II<0.0 || II>(double)(width - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Para + kk*length, width, height, II, JJ, S, 0, InterpAlgo);
						m_F = Pattern[ii + jj*pattern_size + kk*pattern_length];
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						Znssd_reqd[9 * m + 0] = m_F, Znssd_reqd[9 * m + 1] = m_G;
						Znssd_reqd[9 * m + 2] = gx, Znssd_reqd[9 * m + 3] = gy;
						Znssd_reqd[9 * m + 4] = (double)iii, Znssd_reqd[9 * m + 5] = (double)jjj;
						if (advanced_tech == 3)
							Znssd_reqd[9 * m + 6] = numx, Znssd_reqd[9 * m + 7] = numy, Znssd_reqd[9 * m + 8] = denum;

						t_1 += m_F, t_2 += m_G;

						if (printout)
							fprintf(fp1, "%e ", m_F), fprintf(fp2, "%e ", m_G);
					}
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			if (k == 0)
			{
				t_f = t_1 / (m + 1);
				t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[9 * iii + 0] - t_f;
					t_1 += t_4*t_4;
				}
				t_ff = sqrt(t_1);
			}

			t_g = t_2 / (m + 1);
			t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_5 = Znssd_reqd[9 * iii + 1] - t_g;
				t_2 += t_5*t_5;
			}
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[9 * iii + 0] - t_f;
				t_5 = Znssd_reqd[9 * iii + 1] - t_g;
				t_6 = t_5 / t_2 - t_4 / t_ff;
				t_3 = t_6 / t_2;
				gx = Znssd_reqd[9 * iii + 2], gy = Znssd_reqd[9 * iii + 3];
				iii_n = Znssd_reqd[9 * iii + 4], jjj_n = Znssd_reqd[9 * iii + 5];
				if (advanced_tech == 3)
				{
					denum = Znssd_reqd[9 * ii + 8];
					denum2 = denum*denum;
					t_7 = (gx*Znssd_reqd[9 * iii + 6] + gy*Znssd_reqd[9 * ii + 7]) / denum2;
					CC[0] = gx / denum, CC[1] = gy / denum;
					CC[2] = gx*iii_n / denum, CC[3] = gx*jjj_n / denum;
					CC[4] = gy*iii_n / denum, CC[5] = gy*jjj_n / denum;
					CC[6] = -t_7*iii_n;
					CC[7] = -t_7*jjj_n;
				}
				else
				{
					CC[0] = gx, CC[1] = gy;
					if (advanced_tech == 0)
						CC[2] = gx*iii_n + gy*jjj_n;
					if (advanced_tech == 1 || advanced_tech == 2)
					{
						CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
						CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					}
					if (advanced_tech == 2)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
				}

				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (!IsNumber(p[0]) || abs(p[0]) > hsubset || abs(p[1]) > hsubset)
			{
				if (createMem)
					delete[]Znssd_reqd;
				return false;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// Iteration: End

	if (createMem)
		delete[]Znssd_reqd;
	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || p[0] != p[0] || p[1] != p[1] || 1.0 - 0.5*DIC_Coeff_min < ZNCCthresh)
		return false;

	POI.x += p[0], POI.y += p[1];

	return 1.0 - 0.5*DIC_Coeff_min;
}
double TemplateMatching0(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T, double *Znssd_reqd)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography
	if (PR.x <0 || PR.y < 0 || PT.x< 0 || PT.y < 0 || PR.x > widthRef - 1 || PR.y > heightRef - 1 || PT.x > widthTar - 1 || PT.y > heightTar - 1)
		return 0.0;

	int i, j, k, m, ii, kk, iii, jjj, iii_n, jjj_n, iii2, jjj2, ij;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14, 6, 12 };
	int nn = NN[advanced_tech - 1], nExtraParas = advanced_tech > 2 ? 0 : 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = 1;
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196], BB[14], CC[14], p[14], ip[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - nExtraParas ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}
	for (i = 0; i < nn; i++)
		ip[i] = p[i];

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		Znssd_reqd = new double[9 * Tlength];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii, JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0], JJ = PT.y + jjj + p[1];
						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1], CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1, t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (advanced_tech % 2 == 1)
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 0)
					{
						iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
						JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
					}

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						if (advanced_tech < 2)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;

							t_5 = t_4*gx, t_6 = t_4*gy;
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj;
							CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;

							for (j = 0; j < nn; j++)
								BB[j] += t_3*CC[j];

							for (j = 0; j < nn; j++)
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
						else
						{
							Znssd_reqd[9 * m + 0] = m_F, Znssd_reqd[9 * m + 1] = m_G;
							Znssd_reqd[9 * m + 2] = gx, Znssd_reqd[9 * m + 3] = gy;
							Znssd_reqd[9 * m + 4] = (double)iii, Znssd_reqd[9 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			if (advanced_tech < 3)
			{
				DIC_Coeff = t_1 / t_2;
				if (t_2 < 10.0e-9)
					break;
			}
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[9 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[9 * iii + 0] - t_f;
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[9 * iii + 2], gy = Znssd_reqd[9 * iii + 3];
					iii_n = Znssd_reqd[9 * iii + 4], jjj_n = Znssd_reqd[9 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
					CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					if (advanced_tech == 4)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
				if (!IsNumber(DIC_Coeff))
					return 9e9;
				if (!IsFiniteNumber(DIC_Coeff))
					return 9e9;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 9e9;
			}

			if (advanced_tech < 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else if (advanced_tech == 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn)
						Break_Flag = true;
				}
			}
			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	if (advanced_tech < 3)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	else
		ZNCC = 1.0 - 0.5*DIC_Coeff_min; //from ZNSSD to ZNCC

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}
double TemplateMatching0(float *RefPara, float *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, Point2d PR, Point2d PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T, double *Znssd_reqd)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography
	if (PR.x <0 || PR.y < 0 || PT.x <0 || PT.y < 0 || PR.x > widthRef - 1 || PR.y > heightRef - 1 || PT.x > widthTar - 1 || PT.y > heightTar - 1)
		return 0.0;

	int i, j, k, m, ii, kk, iii, jjj, iii_n, jjj_n, iii2, jjj2, ij;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14, 6, 12 };
	int nn = NN[advanced_tech - 1], nExtraParas = advanced_tech > 2 ? 0 : 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = 1;
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196 * 196], BB[14], CC[14], p[14], ip[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - nExtraParas ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}
	for (i = 0; i < nn; i++)
		ip[i] = p[i];

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		Znssd_reqd = new double[9 * Tlength];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii, JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0], JJ = PT.y + jjj + p[1];
						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1], CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1, t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (advanced_tech % 2 == 1)
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else if (advanced_tech == 0)
					{
						iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
						II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
						JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
					}

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						if (advanced_tech < 2)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;

							t_5 = t_4*gx, t_6 = t_4*gy;
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj;
							CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;

							for (j = 0; j < nn; j++)
								BB[j] += t_3*CC[j];

							for (j = 0; j < nn; j++)
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
						else
						{
							Znssd_reqd[9 * m + 0] = m_F, Znssd_reqd[9 * m + 1] = m_G;
							Znssd_reqd[9 * m + 2] = gx, Znssd_reqd[9 * m + 3] = gy;
							Znssd_reqd[9 * m + 4] = (double)iii, Znssd_reqd[9 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			if (advanced_tech < 3)
			{
				DIC_Coeff = t_1 / t_2;
				if (t_2 < 10.0e-9)
					break;
			}
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[9 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[9 * iii + 0] - t_f;
					t_5 = Znssd_reqd[9 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[9 * iii + 2], gy = Znssd_reqd[9 * iii + 3];
					iii_n = Znssd_reqd[9 * iii + 4], jjj_n = Znssd_reqd[9 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
					CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					if (advanced_tech == 4)
					{
						CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
						CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
					}
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
				if (!IsNumber(DIC_Coeff))
					return 9e9;
				if (!IsFiniteNumber(DIC_Coeff))
					return 9e9;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 0.0;
			}

			if (advanced_tech < 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else if (advanced_tech == 3)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 0.0;
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn)
						Break_Flag = true;
				}
			}
			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	if (advanced_tech < 3)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	else
		ZNCC = 1.0 - 0.5*DIC_Coeff_min; //from ZNSSD to ZNCC

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}
double TemplateMatching(double *RefPara, double *TarPara, int refWidth, int refHeight, int tarWidth, int tarHeight, int nchannels, Point2d From, Point2d &Target, LKParameters LKArg, bool greedySearch, double *Timg, double *CorrelBuf, double *iWp, double *direction)
{
	//DIC_Algo = -2: similarlity transform
	//DIC_Algo = -1: epip similarlity transform: not yet supported
	//DIC_Algo = 0: epip translation+photometric
	//DIC_Algo = 1: epip affine+photometric
	//DIC_Algo = 2: translation+photometric
	//DIC_Algo = 3: affine+photometric
	//DIC_Algo = 4: epi irreglar + photometric
	//DIC_Algo = 5: epi quadratic + photometric
	//DIC_Algo = 6: irregular + photometric
	//DIC_Algo = 7:  quadratic + photmetric
	//DIC_Algo = 8: ZNCC affine. Only support gray scale image
	//DIC_Algo = 9:  ZNCC quadratic. Only support gray scale image

	if (From.x <0 || From.y < 0 || Target.x < 0 || Target.y < 0 || From.x > refWidth - 1 || From.y > refHeight - 1 || Target.x > tarWidth - 1 || Target.y > tarHeight - 1)
		return 0.0;

	int i, j, k, kk, iii, jjj, ij, i2, j2, m;
	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	int refLength = refWidth*refHeight, tarLength = tarWidth*tarHeight, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;

	double ii, jj, II, JJ, iii_n, jjj_n, a, b, TarIdx, TarIdy, CorrelScore, CorrelScoreMin, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, RefI, TarI, S[9], p_best[14];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 3, 7, 4, 8, 9, 13, 10, 14, 6, 12 }, jumpStep[2] = { 1, 2 };
	int NN2[] = { 5, 6 }; //similarity transform
	int DIC_Algo2 = 0, nn, nExtraParas = 2, iter = 0;
	int p_jump_0 = jumpStep[Speed], p_jump, p_jump_incr = 1;

	if (DIC_Algo >= 0)
	{
		if (DIC_Algo == 4)
			nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
		else if (DIC_Algo == 5)
			nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
		else if (DIC_Algo == 6)
			nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
		else if (DIC_Algo == 7)
			nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
		else if (DIC_Algo == 9)
			nn = 6, DIC_Algo2 = DIC_Algo, DIC_Algo = 8;
		else
			nn = NN[DIC_Algo];
	}
	else if (DIC_Algo == -1)
		nn = NN2[-DIC_Algo];
	else if (DIC_Algo == -2)
		nn = NN2[-DIC_Algo];

	double AA[196], BB[14], CC[14], p[14];
	if (DIC_Algo < 8)
		for (i = 0; i < nn; i++)
			p[i] = (i == nn - 2 ? 1.0 : 0.0);
	else
		for (i = 0; i < nn; i++)
			p[i] = 0.0;

	bool createMem = false;
	if (Timg == NULL)
	{
		createMem = true;
		Timg = new double[Tlength*nchannels], CorrelBuf = new double[6 * Tlength*nchannels];
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			ii = From.x + iii, jj = From.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*refLength, refWidth, refHeight, ii, jj, S, -1, Interpolation_Algorithm);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	//Let's start with translation and initial shape paras (if available)
	if (greedySearch)
	{
		bool Break_Flag = false;
		double initW[4] = { 0, 0, 0, 0 };
		if (iWp != NULL)
			for (int ii = 0; ii < 4; ii++)
				initW[ii] = iWp[ii];

		CorrelScoreMin = 1e10;
		p[2] = 1.0, p[3] = 0.0;
		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			if (DIC_Algo == 0 || DIC_Algo == 1)
				nn = 3;
			else if (DIC_Algo == 2 || DIC_Algo == 3)
				nn = 4;

			a = p[2], b = p[3];
			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					if (DIC_Algo == 0)
						II = Target.x + iii + p[0] * direction[0], JJ = Target.y + jjj + p[0] * direction[1];
					else if (DIC_Algo == 1)
					{
						II = Target.x + iii + p[0] * direction[0] + initW[0] * iii + initW[1] * jjj;
						JJ = Target.y + jjj + p[0] * direction[1] + initW[2] * iii + initW[3] * jjj;
					}
					else if (DIC_Algo == 2)
						II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
					else if (DIC_Algo == 3 || DIC_Algo == 8)
					{
						II = Target.x + iii + p[0] + initW[0] * iii + initW[1] * jjj;
						JJ = Target.y + jjj + p[1] + initW[2] * iii + initW[3] * jjj;
					}

					if (II<0.0 || II>(double)(refWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(refHeight - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

						RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[3 * kk];
						TarIdx = S[3 * kk + 1], TarIdy = S[3 * kk + 2];

						if (printout)
							fprintf(fp, "%.2f ", TarI);

						t_3 = a*TarI + b - RefI;
						t_4 = a, t_5 = t_4*TarIdx, t_6 = t_4*TarIdy;
						if (DIC_Algo == 0 || DIC_Algo == 1)
							CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = TarI, CC[2] = 1.0;
						else if (DIC_Algo == 2 || DIC_Algo == 3)
							CC[0] = t_5, CC[1] = t_6, CC[2] = TarI, CC[3] = 1.0;

						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j * nn + i] += CC[i] * CC[j];
						}
						t_1 += t_3*t_3;
						t_2 += RefI*RefI;
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			CorrelScore = t_1 / t_2;

			mat_completeSym(AA, nn, true);
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (CorrelScore != CorrelScore || CorrelScore > 50)
			{
				if (createMem)
					delete[]CorrelBuf, delete[]Timg;
				return 0.0;
			}
			if (CorrelScore < CorrelScoreMin)	// If the iteration does not converge, this can be helpful
			{
				CorrelScoreMin = CorrelScore;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
				if (p[0] != p[0])
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
			}

			if (DIC_Algo <= 1)
			{
				if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
				if (fabs(BB[0]) < 0.1*conv_crit_1)
					Break_Flag = true;
			}
			else
			{
				if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
				if (fabs(BB[0]) < 0.1*conv_crit_1 && fabs(BB[1]) < 0.1*conv_crit_1)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		//Store results
		if (DIC_Algo == 0 || DIC_Algo == 1)
			p[0] = 0.5*(p[0] / direction[0] + p[1] / direction[1]);
		else if (DIC_Algo == 2 || DIC_Algo == 3)
			p[0] = p_best[0], p[1] = p_best[1];
	}

	if (iWp != NULL)
	{
		if (DIC_Algo == 1)
			p[1] = iWp[0], p[2] = iWp[1], p[3] = iWp[2], p[4] = iWp[3];
		else if (DIC_Algo == 3 || DIC_Algo == 8)
			p[2] = iWp[0], p[3] = iWp[1], p[4] = iWp[2], p[5] = iWp[3];
		else if (DIC_Algo == -2)
			p[2] = iWp[0], p[3] = iWp[1]; //scale+ angle
	}

	if (DIC_Algo != 0 && DIC_Algo != 2)
	{
		//Now, full DIC
		if (DIC_Algo == 4)
			nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
		else if (DIC_Algo == 5)
			nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
		else if (DIC_Algo == 6)
			nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
		else if (DIC_Algo == 7)
			nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
		else
			nn = NN[DIC_Algo];

		for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
		{
			CorrelScoreMin = 1e10;
			bool Break_Flag = false;
			for (k = 0; k < Iter_Max; k++)
			{
				m = -1, t_1 = 0.0, t_2 = 0.0;
				for (i = 0; i < nn*nn; i++)
					AA[i] = 0.0;
				for (i = 0; i < nn; i++)
					BB[i] = 0.0;

				if (printout)
					fp = fopen("C:/temp/tar.txt", "w+");

				a = p[nn - 2], b = p[nn - 1];
				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						if (DIC_Algo == 1) //afine
						{
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
						}
						else if (DIC_Algo == 3 || DIC_Algo == 8)
						{
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
						}
						else if (DIC_Algo == -2) //similarity transform
						{
							II = Target.x + p[0] + p[2] * cos(p[3]) * iii - p[2] * sin(p[3]) * jjj;
							JJ = Target.y + p[1] + p[2] * sin(p[3]) * iii + p[2] * cos(p[3]) * jjj;
						}

						if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

							RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[3 * kk];
							TarIdx = S[3 * kk + 1], TarIdy = S[3 * kk + 2];

							if (printout)
								fprintf(fp, "%.2f ", TarI);

							if (DIC_Algo < 8)
							{
								t_3 = a*TarI + b - RefI;
								t_4 = a, t_5 = t_4*TarIdx, t_6 = t_4*TarIdy;

								//if (DIC_Algo == 0)
								//	CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = TarI, CC[2] = 1.0;
								if (DIC_Algo == 1)
								{
									CC[0] = t_5*direction[0] + t_6*direction[1];
									CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
									CC[5] = TarI, CC[6] = 1.0;
								}
								//else if (DIC_Algo == 2)
								//	CC[0] = t_5, CC[1] = t_6, CC[2] = TarI, CC[3] = 1.0;
								else if (DIC_Algo == 3)
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
									CC[6] = TarI, CC[7] = 1.0;
								}
								else if (DIC_Algo == -2) //similarity transform
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii*cos(p[3]) - t_5*jjj*sin(p[3]) + t_6*iii*sin(p[3]) + t_6*jjj*cos(p[3]);
									CC[3] = -t_5*iii*p[2] * sin(p[3]) - t_5*jjj*p[2] * cos(p[3]) + t_6*iii*p[2] * cos(p[3]) - t_6*jjj*p[2] * sin(p[3]);
									CC[4] = TarI, CC[5] = 1.0;
								}

								for (j = 0; j < nn; j++)
								{
									BB[j] += t_3*CC[j];
									for (i = j; i < nn; i++)
										AA[j*nn + i] += CC[i] * CC[j];
								}

								t_1 += t_3*t_3, t_2 += RefI*RefI;
							}
							else
							{
								m++;
								CorrelBuf[6 * m + 0] = RefI, CorrelBuf[6 * m + 1] = TarI;
								CorrelBuf[6 * m + 2] = TarIdx, CorrelBuf[6 * m + 3] = TarIdy;
								CorrelBuf[6 * m + 4] = (double)iii, CorrelBuf[6 * m + 5] = (double)jjj;
								t_1 += RefI, t_2 += TarI;
							}
						}
					}
					if (printout)
						fprintf(fp, "\n");
				}
				if (printout)
					fclose(fp);

				if (DIC_Algo < 8)
					CorrelScore = t_1 / t_2;
				else
				{
					if (k == 0)
					{
						t_f = t_1 / (m + 1), t_1 = 0.0;
						for (iii = 0; iii <= m; iii++)
							t_4 = CorrelBuf[6 * iii + 0] - t_f, t_1 += t_4*t_4;
						t_ff = sqrt(t_1);
					}

					t_g = t_2 / (m + 1), t_2 = 0.0;
					for (iii = 0; iii <= m; iii++)
						t_5 = CorrelBuf[6 * iii + 1] - t_g, t_2 += t_5*t_5;
					t_2 = sqrt(t_2);

					CorrelScore = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = CorrelBuf[6 * iii + 0] - t_f, t_5 = CorrelBuf[6 * iii + 1] - t_g;
						TarIdx = CorrelBuf[6 * iii + 2], TarIdy = CorrelBuf[6 * iii + 3];
						iii_n = CorrelBuf[6 * iii + 4], jjj_n = CorrelBuf[6 * iii + 5];

						t_6 = t_5 / t_2 - t_4 / t_ff;
						t_3 = t_6 / t_2;
						CC[0] = TarIdx, CC[1] = TarIdy;
						CC[2] = TarIdx*iii_n, CC[3] = TarIdx*jjj_n;
						CC[4] = TarIdy*iii_n, CC[5] = TarIdy*jjj_n;

						t_4 = t_2*t_2;
						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j] / t_4;
						}
						CorrelScore += t_6*t_6;
					}
					if (CorrelScore != CorrelScore)
						return 0.0;
					if (std::isinf(CorrelScore))
						return 0.0;
				}

				mat_completeSym(AA, nn, true);
				QR_Solution_Double(AA, BB, nn, nn);
				for (i = 0; i < nn; i++)
					p[i] -= BB[i];

				if (CorrelScore != CorrelScore || CorrelScore > 50)
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
				if (CorrelScore < CorrelScoreMin)	// If the iteration does not converge, this can be helpful
				{
					CorrelScoreMin = CorrelScore;
					for (i = 0; i < nn; i++)
						p_best[i] = p[i];
					if (p[0] != p[0])
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
				}

				if (DIC_Algo <= 1)
				{
					if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1)
					{
						for (i = 1; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else if (DIC_Algo < 8)
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (i = 2; i < nn - nExtraParas; i++)
						{
							if (fabs(BB[i]) > conv_crit_2)
								break;
						}
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
						return 0;
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (iii = 2; iii < nn; iii++)
							if (fabs(BB[iii]) > conv_crit_2)
								break;
						if (iii == nn)
							Break_Flag = true;
					}
				}

				if (Break_Flag)
					break;
			}
			iter += k;

			// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
			for (i = 0; i < nn; i++)
				p[i] = p_best[i];
		}

		//Quadratic if needed:
		if (DIC_Algo2 > 3)
		{
			DIC_Algo = DIC_Algo2, nn = NN[DIC_Algo];
			if (DIC_Algo == 4)
			{
				p[7] = p[5], p[8] = p[6];
				for (i = 5; i < 7; i++)
					p[i] = 0.0;
			}
			else if (DIC_Algo == 5)
			{
				p[11] = p[5], p[12] = p[6];
				for (i = 5; i < 11; i++)
					p[i] = 0.0;
			}
			else if (DIC_Algo == 6)
			{
				p[8] = p[6], p[9] = p[7];
				for (i = 6; i < 8; i++)
					p[i] = 0.0;
			}
			else if (DIC_Algo == 7)
			{
				p[12] = p[6], p[13] = p[7];
				for (i = 6; i < 12; i++)
					p[i] = 0.0;
			}

			CorrelScoreMin = 1e10;
			bool Break_Flag = false;
			for (k = 0; k < Iter_Max; k++)
			{
				m = -1, t_1 = 0.0, t_2 = 0.0;
				for (i = 0; i < nn*nn; i++)
					AA[i] = 0.0;
				for (i = 0; i < nn; i++)
					BB[i] = 0.0;
				a = p[nn - 2], b = p[nn - 1];

				if (printout)
					fp = fopen("C:/temp/tar.txt", "w+");
				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						if (DIC_Algo == 4) //irregular
						{
							ij = iii*jjj;
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
						}
						else if (DIC_Algo == 5) //Quadratic
						{
							ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
						}
						else if (DIC_Algo == 6)
						{
							ij = iii*jjj;
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
						}
						else if (DIC_Algo == 7 || DIC_Algo == 9)
						{
							ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
						}

						if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

							RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[3 * kk];
							TarIdx = S[3 * kk + 1], TarIdy = S[3 * kk + 2];
							m++;

							if (printout)
								fprintf(fp, "%.2f ", TarI);

							if (DIC_Algo != 9)
							{
								t_3 = a*TarI + b - RefI;
								t_4 = a, t_5 = t_4*TarIdx, t_6 = t_4*TarIdy;
								if (DIC_Algo == 4) //irregular
								{
									CC[0] = t_5*direction[0] + t_6*direction[1];
									CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
									CC[5] = t_5*ij, CC[6] = t_6*ij;
									CC[7] = TarI, CC[8] = 1.0;
								}
								else if (DIC_Algo == 5) //Quadratic
								{
									CC[0] = t_5*direction[0] + t_6*direction[1];
									CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
									CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = t_5*i2, CC[8] = t_5*j2, CC[9] = t_6*i2, CC[10] = t_6*j2;
									CC[11] = TarI, CC[12] = 1.0;
								}
								else if (DIC_Algo == 6)  //irregular
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
									CC[6] = t_5*ij, CC[7] = t_6*ij;
									CC[8] = TarI, CC[9] = 1.0;
								}
								else if (DIC_Algo == 7)
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
									CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = t_5*i2, CC[9] = t_5*j2, CC[10] = t_6*i2, CC[11] = t_6*j2;
									CC[12] = TarI, CC[13] = 1.0;
								}

								for (j = 0; j < nn; j++)
								{
									BB[j] += t_3*CC[j];
									for (i = j; i < nn; i++)
										AA[j*nn + i] += CC[i] * CC[j];
								}

								t_1 += t_3*t_3, t_2 += RefI*RefI;
							}
							else
							{
								CorrelBuf[6 * m + 0] = RefI, CorrelBuf[6 * m + 1] = TarI;
								CorrelBuf[6 * m + 2] = TarIdx, CorrelBuf[6 * m + 3] = TarIdy;
								CorrelBuf[6 * m + 4] = (double)iii, CorrelBuf[6 * m + 5] = (double)jjj;
								t_1 += RefI, t_2 += TarI;
							}
						}
					}
					if (printout)
						fprintf(fp, "\n");
				}
				if (printout)
					fclose(fp);

				if (DIC_Algo != 9)
					CorrelScore = t_1 / t_2;
				else
				{
					if (k == 0)
					{
						t_f = t_1 / (m + 1), t_1 = 0.0;
						for (iii = 0; iii <= m; iii++)
							t_4 = CorrelBuf[6 * iii + 0] - t_f, t_1 += t_4*t_4;
						t_ff = sqrt(t_1);
					}

					t_g = t_2 / (m + 1), t_2 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_5 = CorrelBuf[6 * iii + 1] - t_g;
						t_2 += t_5*t_5;
					}
					t_2 = sqrt(t_2);

					CorrelScore = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = CorrelBuf[6 * iii + 0] - t_f;
						t_5 = CorrelBuf[6 * iii + 1] - t_g;
						t_6 = t_5 / t_2 - t_4 / t_ff;
						t_3 = t_6 / t_2;
						TarIdx = CorrelBuf[6 * iii + 2], TarIdy = CorrelBuf[6 * iii + 3];
						iii_n = CorrelBuf[6 * iii + 4], jjj_n = CorrelBuf[6 * iii + 5];
						CC[0] = TarIdx, CC[1] = TarIdy;
						CC[2] = TarIdx*iii_n, CC[3] = TarIdx*jjj_n;
						CC[4] = TarIdy*iii_n, CC[5] = TarIdy*jjj_n;
						CC[6] = TarIdx*iii_n*iii_n*0.5, CC[7] = TarIdx*jjj_n*jjj_n*0.5, CC[8] = TarIdx*iii_n*jjj_n;
						CC[9] = TarIdy*iii_n*iii_n*0.5, CC[10] = TarIdy*jjj_n*jjj_n*0.5, CC[11] = TarIdy*iii_n*jjj_n;
						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
						}

						CorrelScore += t_6*t_6;
					}
					if (CorrelScore != CorrelScore)
						return 0;
					if (std::isinf(CorrelScore))
						return 0;
				}

				mat_completeSym(AA, nn);
				QR_Solution_Double(AA, BB, nn, nn);
				for (i = 0; i < nn; i++)
					p[i] -= BB[i];

				if (CorrelScore < CorrelScoreMin)	// If the iteration does not converge, this can be helpful
				{
					CorrelScoreMin = CorrelScore;
					for (i = 0; i < nn; i++)
						p_best[i] = p[i];
					if (p[0] != p[0])
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
				}

				if (DIC_Algo <= 5)
				{
					if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1)
					{
						for (i = 1; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else if (DIC_Algo <= 7)
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (i = 2; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
						return 0;
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (iii = 2; iii < nn; iii++)
							if (fabs(BB[iii]) > conv_crit_2)
								break;
						if (iii == nn)
							Break_Flag = true;
					}
				}

				if (Break_Flag)
					break;
			}
			iter += k;

			// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
			for (i = 0; i < nn; i++)
				p[i] = p_best[i];
		}
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close on convergence, but in case of trouble, zncc is more reliable.
	if (DIC_Algo < 8 && CorrelScoreMin < LKArg.PSSDab_thresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				if (DIC_Algo == 0)
					II = Target.x + iii + p[0] * direction[0], JJ = Target.y + jjj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				else if (DIC_Algo == 2)
					II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
				else if (DIC_Algo == 3)
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
				else if (DIC_Algo == 4) //irregular
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
				}
				else if (DIC_Algo == 5) //Quadratic
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
				}
				else if (DIC_Algo == 6)
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
				}
				else if (DIC_Algo == 7)
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
					JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
				}
				else if (DIC_Algo == -2)
					II = Target.x + p[0] + p[2] * cos(p[3]) * iii - p[2] * sin(p[3]) * jjj, JJ = Target.y + p[1] + p[2] * sin(p[3]) * iii + p[2] * cos(p[3]) * jjj;

				if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, -1, Interpolation_Algorithm);
					if (printout)
						fprintf(fp, "%.4f ", S[3 * kk]);

					CorrelBuf[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], CorrelBuf[2 * m + 1] = S[3 * kk];
					t_f += CorrelBuf[2 * m], t_g += CorrelBuf[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m, t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = CorrelBuf[2 * i] - t_f, t_5 = CorrelBuf[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		CorrelScoreMin = t_1 / t_2; //This is the zncc score
		if (abs(CorrelScoreMin) > 1.0)
			CorrelScoreMin = 0.0;
	}
	else if (DIC_Algo >= 8) //convert znssd to zncc
		CorrelScoreMin = 1.0 - 0.5*CorrelScoreMin;

	if (createMem)
		delete[]Timg, delete[]CorrelBuf;
	if (CorrelScoreMin > 1.0)
		return 0.0;

	if (DIC_Algo >= 0 && DIC_Algo <= 1)
	{
		if (CorrelScoreMin< LKArg.ZNCCThreshold || p[0] != p[0] || abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
			return CorrelScoreMin;
	}
	else
	{
		if (CorrelScoreMin< LKArg.ZNCCThreshold || p[0] != p[0] || p[1] != p[1] || abs(p[0]) > 2.0*hsubset || abs(p[1]) > 2.0*hsubset)
			return CorrelScoreMin;
	}
	/*if (iCovariance != NULL)
	{
	a = p[nn - 2], b = p[nn - 1];
	for (i = 0; i < nn*nn; i++)
	AA[i] = 0.0;
	for (i = 0; i < nn; i++)
	BB[i] = 0.0;

	int count = 0;
	int mMinusn = Tlength*nchannels - nn;
	double *B = new double[Tlength];
	double *BtA = new double[nn];
	double *AtA = new double[nn*nn];

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
	for (iii = -hsubset; iii <= hsubset; iii++)
	{
	if (DIC_Algo == 1)
	II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
	else if (DIC_Algo == 2)
	II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
	else if (DIC_Algo == 3)
	II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
	else if (DIC_Algo == 4) //irregular
	{
	ij = iii*jjj;
	II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
	JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
	}
	else if (DIC_Algo == 5) //Quadratic
	{
	ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
	II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
	JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
	}
	else if (DIC_Algo == 6)
	{
	ij = iii*jjj;
	II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij;
	JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
	}
	else if (DIC_Algo == 7)
	{
	ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
	II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
	JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
	}

	if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
	continue;
	for (kk = 0; kk < nchannels; kk++)
	{
	Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S, 0, Interpolation_Algorithm);
	RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[0];

	TarIdx = S[1], TarIdy = S[2];
	t_3 = a*TarI + b - RefI;
	t_5 = a*TarIdx, t_6 = a*TarIdy;

	B[count] = t_3;
	count++;

	if (DIC_Algo == 1)
	{
	CC[0] = t_5*direction[0] + t_6*direction[1];
	CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
	CC[5] = TarI, CC[6] = 1.0;
	}
	else if (DIC_Algo == 2)
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = TarI, CC[3] = 1.0;
	}
	else if (DIC_Algo == 3)
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
	CC[6] = TarI, CC[7] = 1.0;
	}
	else if (DIC_Algo == 4) //irregular
	{
	CC[0] = t_5*direction[0] + t_6*direction[1];
	CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
	CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = TarI, CC[8] = 1.0;
	}
	else if (DIC_Algo == 5) //Quadratic
	{
	CC[0] = t_5*direction[0] + t_6*direction[1];
	CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
	CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = t_5*i2, CC[8] = t_5*j2;
	CC[9] = t_6*i2, CC[10] = t_6*j2, CC[11] = TarI, CC[12] = 1.0;
	}
	else if (DIC_Algo == 6)  //irregular
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
	CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = TarI, CC[9] = 1.0;
	}
	else if (DIC_Algo == 7)
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
	CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = t_5*i2, CC[9] = t_5*j2, CC[10] = t_6*i2, CC[11] = t_6*j2, CC[12] = TarI, CC[13] = 1.0;
	}

	for (j = 0; j < nn; j++)
	{
	BB[j] += t_3*CC[j];
	for (i = j; i < nn; i++)
	AA[j*nn + i] += CC[i] * CC[j];
	}

	t_1 += t_3*t_3;
	t_2 += RefI*RefI;
	}
	}
	}
	CorrelScore = t_1 / t_2;

	mat_completeSym(AA, nn, true);
	for (i = 0; i < nn*nn; i++)
	AtA[i] = AA[i];
	for (i = 0; i < nn; i++)
	BtA[i] = BB[i];

	QR_Solution_Double(AA, BB, nn, nn);

	double BtAx = 0.0, BtB = 0.0;
	for (i = 0; i < count; i++)
	BtB += B[i] * B[i];
	for (i = 0; i < nn; i++)
	BtAx += BtA[i] * BB[i];
	double mse = (BtB - BtAx) / mMinusn;

	Matrix iAtA(nn, nn), Cov(nn, nn);
	iAtA.Matrix_Init(AtA);
	iAtA = iAtA.Inversion(true, true);
	Cov = mse*iAtA;

	double det = Cov[0] * Cov[nn + 1] - Cov[1] * Cov[nn];
	iCovariance[0] = Cov[nn + 1] / det, iCovariance[1] = -Cov[1] / det, iCovariance[2] = iCovariance[1], iCovariance[3] = Cov[0] / det; //actually, this is inverse of the iCovariance

	delete[]B;
	delete[]BtA;
	delete[]AtA;
	}*/

	if (iWp != NULL)
	{
		if (DIC_Algo == 1 || DIC_Algo == 4 || DIC_Algo == 5)
			iWp[0] = p[1], iWp[1] = p[2], iWp[2] = p[3], iWp[3] = p[4];
		else if (DIC_Algo == 3 || DIC_Algo == 6 || DIC_Algo == 7)
			iWp[0] = p[2], iWp[1] = p[3], iWp[2] = p[4], iWp[3] = p[5];
		else if (DIC_Algo == -2)
			iWp[0] = p[2], iWp[1] = p[3];
	}

	if (DIC_Algo == 1 || DIC_Algo == 4 || DIC_Algo == 5)
		Target.x += p[0] * direction[0], Target.y += p[0] * direction[1];
	else
		Target.x += p[0], Target.y += p[1];

	return CorrelScoreMin;
}
double TemplateMatching(float *RefPara, float *TarPara, int refWidth, int refHeight, int tarWidth, int tarHeight, int nchannels, Point2d From, Point2d &Target, LKParameters LKArg, bool greedySearch, double *Timg, double *CorrelBuf, double *iWp, double *direction)
{
	//DIC_Algo = 0: epip translation+photometric
	//DIC_Algo = 1: epip affine+photometric
	//DIC_Algo = 2: translation+photometric
	//DIC_Algo = 3: affine+photometric
	//DIC_Algo = 4: epi irreglar + photometric
	//DIC_Algo = 5: epi quadratic + photometric
	//DIC_Algo = 6: irregular + photometric
	//DIC_Algo = 7:  quadratic + photmetric
	//DIC_Algo = 8: ZNCC affine. Only support gray scale image
	//DIC_Algo = 9:  ZNCC quadratic. Only support gray scale image

	if (From.x <0 || From.y < 0 || Target.x < 0 || Target.y < 0 || From.x > refWidth - 1 || From.y > refHeight - 1 || Target.x > tarWidth - 1 || Target.y > tarHeight - 1)
		return 0.0;

	int i, j, k, kk, iii, jjj, ij, i2, j2, m;
	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	int refLength = refWidth*refHeight, tarLength = tarWidth*tarHeight, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;

	double ii, jj, II, JJ, iii_n, jjj_n, a, b, TarIdx, TarIdy, CorrelScore, CorrelScoreMin, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g, RefI, TarI, S[9], p_best[14];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 3, 7, 4, 8, 9, 13, 10, 14, 6, 12 }, jumpStep[2] = { 1, 2 };
	int DIC_Algo2 = 0, nn, nExtraParas = 2, iter = 0;
	int p_jump_0 = jumpStep[Speed], p_jump, p_jump_incr = 1;

	if (DIC_Algo == 4)
		nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
	else if (DIC_Algo == 5)
		nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
	else if (DIC_Algo == 6)
		nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
	else if (DIC_Algo == 7)
		nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
	else if (DIC_Algo == 9)
		nn = 6, DIC_Algo2 = DIC_Algo, DIC_Algo = 8;
	else
		nn = NN[DIC_Algo];

	double AA[196], BB[14], CC[14], p[14];
	if (DIC_Algo < 8)
		for (i = 0; i < nn; i++)
			p[i] = (i == nn - 2 ? 1.0 : 0.0);
	else
		for (i = 0; i < nn; i++)
			p[i] = 0.0;

	bool createMem = false;
	if (Timg == NULL)
	{
		createMem = true;
		Timg = new double[Tlength*nchannels], CorrelBuf = new double[6 * Tlength*nchannels];
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			ii = From.x + iii, jj = From.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*refLength, refWidth, refHeight, ii, jj, S, -1, Interpolation_Algorithm);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	//Let's start with translation and initial shape paras (if available)
	if (greedySearch)
	{
		bool Break_Flag = false;
		double initW[4] = { 0, 0, 0, 0 };
		if (iWp != NULL)
			for (int ii = 0; ii < 4; ii++)
				initW[ii] = iWp[ii];

		CorrelScoreMin = 1e10;
		p[2] = 1.0, p[3] = 0.0;
		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			if (DIC_Algo == 0 || DIC_Algo == 1)
				nn = 3;
			else if (DIC_Algo == 2 || DIC_Algo == 3)
				nn = 4;

			a = p[2], b = p[3];
			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					if (DIC_Algo == 0)
						II = Target.x + iii + p[0] * direction[0], JJ = Target.y + jjj + p[0] * direction[1];
					else if (DIC_Algo == 1)
					{
						II = Target.x + iii + p[0] * direction[0] + initW[0] * iii + initW[1] * jjj;
						JJ = Target.y + jjj + p[0] * direction[1] + initW[2] * iii + initW[3] * jjj;
					}
					else if (DIC_Algo == 2)
						II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
					else if (DIC_Algo == 3 || DIC_Algo == 8)
					{
						II = Target.x + iii + p[0] + initW[0] * iii + initW[1] * jjj;
						JJ = Target.y + jjj + p[1] + initW[2] * iii + initW[3] * jjj;
					}

					if (II<0.0 || II>(double)(refWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(refHeight - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

						RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[3 * kk];
						TarIdx = S[3 * kk + 1], TarIdy = S[3 * kk + 2];

						if (printout)
							fprintf(fp, "%.2f ", TarI);

						t_3 = a*TarI + b - RefI;
						t_4 = a, t_5 = t_4*TarIdx, t_6 = t_4*TarIdy;
						if (DIC_Algo == 0 || DIC_Algo == 1)
							CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = TarI, CC[2] = 1.0;
						else if (DIC_Algo == 2 || DIC_Algo == 3)
							CC[0] = t_5, CC[1] = t_6, CC[2] = TarI, CC[3] = 1.0;

						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j * nn + i] += CC[i] * CC[j];
						}
						t_1 += t_3*t_3;
						t_2 += RefI*RefI;
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			CorrelScore = t_1 / t_2;

			mat_completeSym(AA, nn, true);
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (CorrelScore != CorrelScore || CorrelScore > 50)
			{
				if (createMem)
					delete[]CorrelBuf, delete[]Timg;
				return 0.0;
			}
			if (CorrelScore < CorrelScoreMin)	// If the iteration does not converge, this can be helpful
			{
				CorrelScoreMin = CorrelScore;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
				if (p[0] != p[0])
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
			}

			if (DIC_Algo <= 1)
			{
				if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
				if (fabs(BB[0]) < 0.1*conv_crit_1)
					Break_Flag = true;
			}
			else
			{
				if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
				if (fabs(BB[0]) < 0.1*conv_crit_1 && fabs(BB[1]) < 0.1*conv_crit_1)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		//Store results
		if (DIC_Algo == 0 || DIC_Algo == 1)
			p[0] = 0.5*(p[0] / direction[0] + p[1] / direction[1]);
		else if (DIC_Algo == 2 || DIC_Algo == 3)
			p[0] = p_best[0], p[1] = p_best[1];
	}

	if (iWp != NULL)
	{
		if (DIC_Algo == 1)
			p[1] = iWp[0], p[2] = iWp[1], p[3] = iWp[2], p[4] = iWp[3];
		else if (DIC_Algo == 3)
			p[2] = iWp[0], p[3] = iWp[1], p[4] = iWp[2], p[5] = iWp[3];
	}

	if (DIC_Algo != 0 && DIC_Algo != 2)
	{
		//Now, full DIC
		if (DIC_Algo == 4)
			nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
		else if (DIC_Algo == 5)
			nn = 7, DIC_Algo2 = DIC_Algo, DIC_Algo = 1;
		else if (DIC_Algo == 6)
			nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
		else if (DIC_Algo == 7)
			nn = 8, DIC_Algo2 = DIC_Algo, DIC_Algo = 3;
		else
			nn = NN[DIC_Algo];

		for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
		{
			CorrelScoreMin = 1e10;
			bool Break_Flag = false;
			for (k = 0; k < Iter_Max; k++)
			{
				m = -1, t_1 = 0.0, t_2 = 0.0;
				for (i = 0; i < nn*nn; i++)
					AA[i] = 0.0;
				for (i = 0; i < nn; i++)
					BB[i] = 0.0;

				if (printout)
					fp = fopen("C:/temp/tar.txt", "w+");

				a = p[nn - 2], b = p[nn - 1];
				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						if (DIC_Algo == 1) //afine
						{
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
						}
						else if (DIC_Algo == 3 || DIC_Algo == 8)
						{
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
						}

						if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

							RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[3 * kk];
							TarIdx = S[3 * kk + 1], TarIdy = S[3 * kk + 2];

							if (printout)
								fprintf(fp, "%.2f ", TarI);

							if (DIC_Algo < 8)
							{
								t_3 = a*TarI + b - RefI;
								t_4 = a, t_5 = t_4*TarIdx, t_6 = t_4*TarIdy;

								//if (DIC_Algo == 0)
								//	CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = TarI, CC[2] = 1.0;
								if (DIC_Algo == 1)
								{
									CC[0] = t_5*direction[0] + t_6*direction[1];
									CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
									CC[5] = TarI, CC[6] = 1.0;
								}
								//else if (DIC_Algo == 2)
								//	CC[0] = t_5, CC[1] = t_6, CC[2] = TarI, CC[3] = 1.0;
								else if (DIC_Algo == 3)
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
									CC[6] = TarI, CC[7] = 1.0;
								}

								for (j = 0; j < nn; j++)
								{
									BB[j] += t_3*CC[j];
									for (i = j; i < nn; i++)
										AA[j*nn + i] += CC[i] * CC[j];
								}

								t_1 += t_3*t_3, t_2 += RefI*RefI;
							}
							else
							{
								m++;
								CorrelBuf[6 * m + 0] = RefI, CorrelBuf[6 * m + 1] = TarI;
								CorrelBuf[6 * m + 2] = TarIdx, CorrelBuf[6 * m + 3] = TarIdy;
								CorrelBuf[6 * m + 4] = (double)iii, CorrelBuf[6 * m + 5] = (double)jjj;
								t_1 += RefI, t_2 += TarI;
							}
						}
					}
					if (printout)
						fprintf(fp, "\n");
				}
				if (printout)
					fclose(fp);

				if (DIC_Algo < 8)
					CorrelScore = t_1 / t_2;
				else
				{
					if (k == 0)
					{
						t_f = t_1 / (m + 1), t_1 = 0.0;
						for (iii = 0; iii <= m; iii++)
							t_4 = CorrelBuf[6 * iii + 0] - t_f, t_1 += t_4*t_4;
						t_ff = sqrt(t_1);
					}

					t_g = t_2 / (m + 1), t_2 = 0.0;
					for (iii = 0; iii <= m; iii++)
						t_5 = CorrelBuf[6 * iii + 1] - t_g, t_2 += t_5*t_5;
					t_2 = sqrt(t_2);

					CorrelScore = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = CorrelBuf[6 * iii + 0] - t_f, t_5 = CorrelBuf[6 * iii + 1] - t_g;
						TarIdx = CorrelBuf[6 * iii + 2], TarIdy = CorrelBuf[6 * iii + 3];
						iii_n = CorrelBuf[6 * iii + 4], jjj_n = CorrelBuf[6 * iii + 5];

						t_6 = t_5 / t_2 - t_4 / t_ff;
						t_3 = t_6 / t_2;
						CC[0] = TarIdx, CC[1] = TarIdy;
						CC[2] = TarIdx*iii_n, CC[3] = TarIdx*jjj_n;
						CC[4] = TarIdy*iii_n, CC[5] = TarIdy*jjj_n;

						t_4 = t_2*t_2;
						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j] / t_4;
						}
						CorrelScore += t_6*t_6;
					}
					if (CorrelScore != CorrelScore)
						return 0.0;
					if (std::isinf(CorrelScore))
						return 0.0;
				}

				mat_completeSym(AA, nn, true);
				QR_Solution_Double(AA, BB, nn, nn);
				for (i = 0; i < nn; i++)
					p[i] -= BB[i];

				if (CorrelScore != CorrelScore || CorrelScore > 50)
				{
					if (createMem)
						delete[]CorrelBuf, delete[]Timg;
					return 0.0;
				}
				if (CorrelScore < CorrelScoreMin)	// If the iteration does not converge, this can be helpful
				{
					CorrelScoreMin = CorrelScore;
					for (i = 0; i < nn; i++)
						p_best[i] = p[i];
					if (p[0] != p[0])
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
				}

				if (DIC_Algo <= 1)
				{
					if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1)
					{
						for (i = 1; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else if (DIC_Algo < 8)
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (i = 2; i < nn - nExtraParas; i++)
						{
							if (fabs(BB[i]) > conv_crit_2)
								break;
						}
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
						return 0;
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (iii = 2; iii < nn; iii++)
							if (fabs(BB[iii]) > conv_crit_2)
								break;
						if (iii == nn)
							Break_Flag = true;
					}
				}

				if (Break_Flag)
					break;
			}
			iter += k;

			// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
			for (i = 0; i < nn; i++)
				p[i] = p_best[i];
		}

		//Quadratic if needed:
		if (DIC_Algo2 > 3)
		{
			DIC_Algo = DIC_Algo2, nn = NN[DIC_Algo];
			if (DIC_Algo == 4)
			{
				p[7] = p[5], p[8] = p[6];
				for (i = 5; i < 7; i++)
					p[i] = 0.0;
			}
			else if (DIC_Algo == 5)
			{
				p[11] = p[5], p[12] = p[6];
				for (i = 5; i < 11; i++)
					p[i] = 0.0;
			}
			else if (DIC_Algo == 6)
			{
				p[8] = p[6], p[9] = p[7];
				for (i = 6; i < 8; i++)
					p[i] = 0.0;
			}
			else if (DIC_Algo == 7)
			{
				p[12] = p[6], p[13] = p[7];
				for (i = 6; i < 12; i++)
					p[i] = 0.0;
			}

			CorrelScoreMin = 1e10;
			bool Break_Flag = false;
			for (k = 0; k < Iter_Max; k++)
			{
				m = -1, t_1 = 0.0, t_2 = 0.0;
				for (i = 0; i < nn*nn; i++)
					AA[i] = 0.0;
				for (i = 0; i < nn; i++)
					BB[i] = 0.0;
				a = p[nn - 2], b = p[nn - 1];

				if (printout)
					fp = fopen("C:/temp/tar.txt", "w+");
				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						if (DIC_Algo == 4) //irregular
						{
							ij = iii*jjj;
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
						}
						else if (DIC_Algo == 5) //Quadratic
						{
							ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
						}
						else if (DIC_Algo == 6)
						{
							ij = iii*jjj;
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
						}
						else if (DIC_Algo == 7 || DIC_Algo == 9)
						{
							ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
						}

						if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

							RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[3 * kk];
							TarIdx = S[3 * kk + 1], TarIdy = S[3 * kk + 2];
							m++;

							if (printout)
								fprintf(fp, "%.2f ", TarI);

							if (DIC_Algo != 9)
							{
								t_3 = a*TarI + b - RefI;
								t_4 = a, t_5 = t_4*TarIdx, t_6 = t_4*TarIdy;
								if (DIC_Algo == 4) //irregular
								{
									CC[0] = t_5*direction[0] + t_6*direction[1];
									CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
									CC[5] = t_5*ij, CC[6] = t_6*ij;
									CC[7] = TarI, CC[8] = 1.0;
								}
								else if (DIC_Algo == 5) //Quadratic
								{
									CC[0] = t_5*direction[0] + t_6*direction[1];
									CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
									CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = t_5*i2, CC[8] = t_5*j2, CC[9] = t_6*i2, CC[10] = t_6*j2;
									CC[11] = TarI, CC[12] = 1.0;
								}
								else if (DIC_Algo == 6)  //irregular
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
									CC[6] = t_5*ij, CC[7] = t_6*ij;
									CC[8] = TarI, CC[9] = 1.0;
								}
								else if (DIC_Algo == 7)
								{
									CC[0] = t_5, CC[1] = t_6;
									CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
									CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = t_5*i2, CC[9] = t_5*j2, CC[10] = t_6*i2, CC[11] = t_6*j2;
									CC[12] = TarI, CC[13] = 1.0;
								}

								for (j = 0; j < nn; j++)
								{
									BB[j] += t_3*CC[j];
									for (i = j; i < nn; i++)
										AA[j*nn + i] += CC[i] * CC[j];
								}

								t_1 += t_3*t_3, t_2 += RefI*RefI;
							}
							else
							{
								CorrelBuf[6 * m + 0] = RefI, CorrelBuf[6 * m + 1] = TarI;
								CorrelBuf[6 * m + 2] = TarIdx, CorrelBuf[6 * m + 3] = TarIdy;
								CorrelBuf[6 * m + 4] = (double)iii, CorrelBuf[6 * m + 5] = (double)jjj;
								t_1 += RefI, t_2 += TarI;
							}
						}
					}
					if (printout)
						fprintf(fp, "\n");
				}
				if (printout)
					fclose(fp);

				if (DIC_Algo != 9)
					CorrelScore = t_1 / t_2;
				else
				{
					if (k == 0)
					{
						t_f = t_1 / (m + 1), t_1 = 0.0;
						for (iii = 0; iii <= m; iii++)
							t_4 = CorrelBuf[6 * iii + 0] - t_f, t_1 += t_4*t_4;
						t_ff = sqrt(t_1);
					}

					t_g = t_2 / (m + 1), t_2 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_5 = CorrelBuf[6 * iii + 1] - t_g;
						t_2 += t_5*t_5;
					}
					t_2 = sqrt(t_2);

					CorrelScore = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = CorrelBuf[6 * iii + 0] - t_f;
						t_5 = CorrelBuf[6 * iii + 1] - t_g;
						t_6 = t_5 / t_2 - t_4 / t_ff;
						t_3 = t_6 / t_2;
						TarIdx = CorrelBuf[6 * iii + 2], TarIdy = CorrelBuf[6 * iii + 3];
						iii_n = CorrelBuf[6 * iii + 4], jjj_n = CorrelBuf[6 * iii + 5];
						CC[0] = TarIdx, CC[1] = TarIdy;
						CC[2] = TarIdx*iii_n, CC[3] = TarIdx*jjj_n;
						CC[4] = TarIdy*iii_n, CC[5] = TarIdy*jjj_n;
						CC[6] = TarIdx*iii_n*iii_n*0.5, CC[7] = TarIdx*jjj_n*jjj_n*0.5, CC[8] = TarIdx*iii_n*jjj_n;
						CC[9] = TarIdy*iii_n*iii_n*0.5, CC[10] = TarIdy*jjj_n*jjj_n*0.5, CC[11] = TarIdy*iii_n*jjj_n;
						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
						}

						CorrelScore += t_6*t_6;
					}
					if (CorrelScore != CorrelScore)
						return 0;
					if (std::isinf(CorrelScore))
						return 0;
				}

				mat_completeSym(AA, nn);
				QR_Solution_Double(AA, BB, nn, nn);
				for (i = 0; i < nn; i++)
					p[i] -= BB[i];

				if (CorrelScore < CorrelScoreMin)	// If the iteration does not converge, this can be helpful
				{
					CorrelScoreMin = CorrelScore;
					for (i = 0; i < nn; i++)
						p_best[i] = p[i];
					if (p[0] != p[0])
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
				}

				if (DIC_Algo <= 5)
				{
					if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1)
					{
						for (i = 1; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else if (DIC_Algo <= 7)
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
					{
						if (createMem)
							delete[]CorrelBuf, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (i = 2; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
						return 0;
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (iii = 2; iii < nn; iii++)
							if (fabs(BB[iii]) > conv_crit_2)
								break;
						if (iii == nn)
							Break_Flag = true;
					}
				}

				if (Break_Flag)
					break;
			}
			iter += k;

			// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
			for (i = 0; i < nn; i++)
				p[i] = p_best[i];
		}
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close on convergence, but in case of trouble, zncc is more reliable.
	if (DIC_Algo < 8 && CorrelScoreMin < LKArg.PSSDab_thresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				if (DIC_Algo == 0)
					II = Target.x + iii + p[0] * direction[0], JJ = Target.y + jjj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				else if (DIC_Algo == 2)
					II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
				else if (DIC_Algo == 3)
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
				else if (DIC_Algo == 4) //irregular
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
				}
				else if (DIC_Algo == 5) //Quadratic
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
				}
				else if (DIC_Algo == 6)
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
				}
				else if (DIC_Algo == 7)
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
					JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
				}

				if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S + 3 * kk, -1, Interpolation_Algorithm);
					if (printout)
						fprintf(fp, "%.4f ", S[3 * kk]);

					CorrelBuf[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], CorrelBuf[2 * m + 1] = S[3 * kk];
					t_f += CorrelBuf[2 * m], t_g += CorrelBuf[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m, t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = CorrelBuf[2 * i] - t_f, t_5 = CorrelBuf[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		CorrelScoreMin = t_1 / t_2; //This is the zncc score
		if (abs(CorrelScoreMin) > 1.0)
			CorrelScoreMin = 0.0;
	}
	else if (DIC_Algo >= 8) //convert znssd to zncc
		CorrelScoreMin = 1.0 - 0.5*CorrelScoreMin;

	if (createMem)
		delete[]Timg, delete[]CorrelBuf;
	if (CorrelScoreMin > 1.0)
		return 0.0;

	if (DIC_Algo <= 1)
	{
		if (CorrelScoreMin< LKArg.ZNCCThreshold || p[0] != p[0] || abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
			return CorrelScoreMin;
	}
	else
	{
		if (CorrelScoreMin< LKArg.ZNCCThreshold || p[0] != p[0] || p[1] != p[1] || abs(p[0]) > 2.0*hsubset || abs(p[1]) > 2.0*hsubset)
			return CorrelScoreMin;
	}
	/*if (iCovariance != NULL)
	{
	a = p[nn - 2], b = p[nn - 1];
	for (i = 0; i < nn*nn; i++)
	AA[i] = 0.0;
	for (i = 0; i < nn; i++)
	BB[i] = 0.0;

	int count = 0;
	int mMinusn = Tlength*nchannels - nn;
	double *B = new double[Tlength];
	double *BtA = new double[nn];
	double *AtA = new double[nn*nn];

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
	for (iii = -hsubset; iii <= hsubset; iii++)
	{
	if (DIC_Algo == 1)
	II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
	else if (DIC_Algo == 2)
	II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
	else if (DIC_Algo == 3)
	II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
	else if (DIC_Algo == 4) //irregular
	{
	ij = iii*jjj;
	II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
	JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
	}
	else if (DIC_Algo == 5) //Quadratic
	{
	ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
	II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
	JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
	}
	else if (DIC_Algo == 6)
	{
	ij = iii*jjj;
	II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij;
	JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
	}
	else if (DIC_Algo == 7)
	{
	ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
	II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
	JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
	}

	if (II<0.0 || II>(double)(tarWidth - 1) - (1e-10) || JJ<0.0 || JJ>(double)(tarHeight - 1) - (1e-10))
	continue;
	for (kk = 0; kk < nchannels; kk++)
	{
	Get_Value_Spline(TarPara + kk*tarLength, tarWidth, tarHeight, II, JJ, S, 0, Interpolation_Algorithm);
	RefI = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], TarI = S[0];

	TarIdx = S[1], TarIdy = S[2];
	t_3 = a*TarI + b - RefI;
	t_5 = a*TarIdx, t_6 = a*TarIdy;

	B[count] = t_3;
	count++;

	if (DIC_Algo == 1)
	{
	CC[0] = t_5*direction[0] + t_6*direction[1];
	CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
	CC[5] = TarI, CC[6] = 1.0;
	}
	else if (DIC_Algo == 2)
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = TarI, CC[3] = 1.0;
	}
	else if (DIC_Algo == 3)
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
	CC[6] = TarI, CC[7] = 1.0;
	}
	else if (DIC_Algo == 4) //irregular
	{
	CC[0] = t_5*direction[0] + t_6*direction[1];
	CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
	CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = TarI, CC[8] = 1.0;
	}
	else if (DIC_Algo == 5) //Quadratic
	{
	CC[0] = t_5*direction[0] + t_6*direction[1];
	CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
	CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = t_5*i2, CC[8] = t_5*j2;
	CC[9] = t_6*i2, CC[10] = t_6*j2, CC[11] = TarI, CC[12] = 1.0;
	}
	else if (DIC_Algo == 6)  //irregular
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
	CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = TarI, CC[9] = 1.0;
	}
	else if (DIC_Algo == 7)
	{
	CC[0] = t_5, CC[1] = t_6;
	CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
	CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = t_5*i2, CC[9] = t_5*j2, CC[10] = t_6*i2, CC[11] = t_6*j2, CC[12] = TarI, CC[13] = 1.0;
	}

	for (j = 0; j < nn; j++)
	{
	BB[j] += t_3*CC[j];
	for (i = j; i < nn; i++)
	AA[j*nn + i] += CC[i] * CC[j];
	}

	t_1 += t_3*t_3;
	t_2 += RefI*RefI;
	}
	}
	}
	CorrelScore = t_1 / t_2;

	mat_completeSym(AA, nn, true);
	for (i = 0; i < nn*nn; i++)
	AtA[i] = AA[i];
	for (i = 0; i < nn; i++)
	BtA[i] = BB[i];

	QR_Solution_Double(AA, BB, nn, nn);

	double BtAx = 0.0, BtB = 0.0;
	for (i = 0; i < count; i++)
	BtB += B[i] * B[i];
	for (i = 0; i < nn; i++)
	BtAx += BtA[i] * BB[i];
	double mse = (BtB - BtAx) / mMinusn;

	Matrix iAtA(nn, nn), Cov(nn, nn);
	iAtA.Matrix_Init(AtA);
	iAtA = iAtA.Inversion(true, true);
	Cov = mse*iAtA;

	double det = Cov[0] * Cov[nn + 1] - Cov[1] * Cov[nn];
	iCovariance[0] = Cov[nn + 1] / det, iCovariance[1] = -Cov[1] / det, iCovariance[2] = iCovariance[1], iCovariance[3] = Cov[0] / det; //actually, this is inverse of the iCovariance

	delete[]B;
	delete[]BtA;
	delete[]AtA;
	}*/

	if (iWp != NULL)
	{
		if (DIC_Algo == 1 || DIC_Algo == 4 || DIC_Algo == 5)
			iWp[0] = p[1], iWp[1] = p[2], iWp[2] = p[3], iWp[3] = p[4];
		else if (DIC_Algo == 3 || DIC_Algo == 6 || DIC_Algo == 7)
			iWp[0] = p[2], iWp[1] = p[3], iWp[2] = p[4], iWp[3] = p[5];
	}

	if (DIC_Algo == 1 || DIC_Algo == 4 || DIC_Algo == 5)
		Target.x += p[0] * direction[0], Target.y += p[0] * direction[1];
	else
		Target.x += p[0], Target.y += p[1];

	return CorrelScoreMin;
}

int Track_1P_1F_WithRefTemplate(vector<Point2f> &TrackUV, vector<AffinePara> &warp, vector<AffinePara> &iwarp, vector<Mat> *ImgPyr, double *RefPara, double *CPara, int reffid, int fid, int pid, int MaxWinSize, int nWinSize, int WinStep, int cvPyrLevel, LKParameters LKArg, double *T1, double *T2)
{
	Size winSize(MaxWinSize, MaxWinSize);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01);

	vector<float> err;
	vector<uchar> status;
	Point2f cvBestNewPt(-1, -1); double minDist = 9e9;
	vector<Point2f> cvRefPt, cvBackRefPt, cvPrePt, cvNewPt;

	int width = ImgPyr[fid][0].cols, height = ImgPyr[fid][0].rows, orghsubset = MaxWinSize / 2 + 1;
	double bestcwarp[4], besticwarp[4], cwarp[4], icwarp[4];
	Point2d refpt, npt, brefpt;
	double dispThresh2 = LKArg.DisplacementThresh*LKArg.DisplacementThresh;

	for (int trial = 0; trial < nWinSize; trial++)
	{
		Size winSizeI(MaxWinSize - trial*WinStep, MaxWinSize - trial*WinStep);
		cvBackRefPt.clear(), cvRefPt.clear(), cvPrePt.clear(), cvNewPt.clear();

		cvBackRefPt.push_back(TrackUV[reffid]), cvRefPt.push_back(TrackUV[reffid]);
		cvPrePt.push_back(TrackUV.back()), cvNewPt.push_back(TrackUV.back());

		if (cvPrePt[0].x <MaxWinSize || cvPrePt[0].y < MaxWinSize || cvPrePt[0].x >ImgPyr[fid - 1][0].cols - MaxWinSize || cvPrePt[0].y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		status.clear(), err.clear();
		calcOpticalFlowPyrLK(ImgPyr[fid - 1], ImgPyr[fid], cvPrePt, cvNewPt, status, err, winSizeI, cvPyrLevel, termcrit);

		if (cvNewPt[0].x <MaxWinSize || cvNewPt[0].y < MaxWinSize || cvNewPt[0].x >ImgPyr[fid - 1][0].cols - MaxWinSize || cvNewPt[0].y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		//Consistent flow wrst the ref template: 
		LKArg.hsubset = orghsubset - trial*WinStep / 2;
		refpt = cvRefPt[0], npt = cvNewPt[0], brefpt = cvRefPt[0];

		for (int ii = 0; ii < 4; ii++)
			cwarp[ii] = warp[fid - 1].warp[ii];
		double score1 = TemplateMatching(RefPara, CPara, width, height, width, height, 1, refpt, npt, LKArg, false, T1, T2, cwarp);
		if (score1 < LKArg.ZNCCThreshold)
			continue;

		if (npt.x <MaxWinSize || npt.y < MaxWinSize || npt.x >ImgPyr[fid - 1][0].cols - MaxWinSize || npt.y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		for (int ii = 0; ii < 4; ii++)
			icwarp[ii] = iwarp[fid - 1].warp[ii];
		double score2 = TemplateMatching(CPara, RefPara, width, height, width, height, 1, npt, brefpt, LKArg, false, T1, T2, icwarp);
		if (score2 < LKArg.ZNCCThreshold)
			continue;

		if (brefpt.x <MaxWinSize || brefpt.y < MaxWinSize || brefpt.x >ImgPyr[fid - 1][0].cols - MaxWinSize || brefpt.y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		double dist = pow(brefpt.x - refpt.x, 2) + pow(brefpt.y - refpt.y, 2);
		if (dist < minDist && dist < dispThresh2)
		{
			minDist = dist;
			cvBestNewPt = npt;
			for (int ii = 0; ii < 4; ii++)
				bestcwarp[ii] = cwarp[ii], besticwarp[ii] = icwarp[ii];
		}
	}

	if (minDist < dispThresh2)
	{
		TrackUV.push_back(cvBestNewPt);
		AffinePara w1(bestcwarp), w2(besticwarp);;
		warp.push_back(w1);
		iwarp.push_back(w2);
		return 1;
	}
	else
		return 0;
}
int TrackAllPointsWithRefTemplate(char *Path, int viewID, int startF, vector<Point2f> uvRef, vector<float> sRef, vector<Point2f> *ForeTrackUV, vector<Point2f> *BackTrackUV,
	vector<float> *ForeScale, vector<float> *BackScale, vector<AffinePara> *cForeWarp, vector<AffinePara> *cBackWarp, vector<FeatureDesc> *ForeDesc, vector<FeatureDesc> *BackDesc, vector<Mat> *ForePyr, vector<Mat> *BackPyr,
	int MaxWinSize, int nWinSize, int WinStep, int cvPyrLevel, int fps, int ForeTrackRange, int BackTrackRange, int interpAlgo, bool debug)
{
	const double  descThresh = 0.7;
	double ratio = 1.0*ForePyr[0][0].cols / 1920;

	LKParameters LKArg; //LKArg.hsubset to be changed according to the point scale
	LKArg.DisplacementThresh = 0.5, LKArg.DIC_Algo = 3, LKArg.InterpAlgo = interpAlgo, LKArg.EpipEnforce = 0;
	LKArg.Incomplete_Subset_Handling = 0, LKArg.Convergence_Criteria = 0, LKArg.Analysis_Speed = 0, LKArg.IterMax = 15;
	LKArg.PSSDab_thresh = 0.1, LKArg.ZNCCThreshold = 0.7;

	int npts = (int)uvRef.size();
	vector<AffinePara> *ForeWarp = new vector<AffinePara>[npts];
	vector<AffinePara> *ForeiWarp = new vector<AffinePara>[npts];
	vector<AffinePara> *BackWarp = new vector<AffinePara>[npts];
	vector<AffinePara> *BackiWarp = new vector<AffinePara>[npts];
	AffinePara w1, w2, w3, w4, currentWarp, cumWarp, baseWarp;
	if (LKArg.DIC_Algo == -2)
		w1.warp[0] = 1, w2.warp[0] = 1, w3.warp[0] = 1, w4.warp[0] = 1;//set scale to be 1 for sim-trans

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic,1)
	for (int ii = 0; ii < npts; ii++)
	{
		ForeTrackUV[ii].reserve(ForeTrackRange), ForeTrackUV[ii].push_back(uvRef[ii]);
		BackTrackUV[ii].reserve(BackTrackRange), BackTrackUV[ii].push_back(uvRef[ii]);

		ForeScale[ii].reserve(ForeTrackRange), ForeScale[ii].push_back(sRef[ii]);
		BackScale[ii].reserve(BackTrackRange), BackScale[ii].push_back(sRef[ii]);

		ForeWarp[ii].reserve(ForeTrackRange), ForeWarp[ii].push_back(w1);
		BackWarp[ii].reserve(BackTrackRange), BackWarp[ii].push_back(w2);
		ForeiWarp[ii].reserve(ForeTrackRange), ForeiWarp[ii].push_back(w3);
		BackiWarp[ii].reserve(BackTrackRange), BackiWarp[ii].push_back(w4);
	}

	int maxThreads = omp_get_max_threads(), Tlength = (MaxWinSize + 2)*(MaxWinSize + 2) * 2;
	int *AllRefFid = new int[npts], *FramesTrackedCount = new int[npts], *JustUpdate = new int[npts], *PermTrackFail = new int[npts], *TempTrackFail = new int[npts];
	double *T1 = new double[Tlength* maxThreads], *T2 = new double[6 * Tlength* maxThreads];

	int width = ForePyr[0][0].cols, height = ForePyr[0][0].rows;
	unsigned char *Img = new unsigned char[width*height];

	for (int jj = 0; jj < height; jj++)
	{
		const unsigned char* RowJJ = ForePyr[0][0].ptr<unsigned char>(jj);
		for (int ii = 0; ii < width; ii++)
			Img[ii + jj*width] = RowJJ[ii];
	}
	double *ImgIParai = new double[height*width];
	Generate_Para_Spline(Img, ImgIParai, width, height, LKArg.InterpAlgo);

	vector<double *>ForeImgIPara;
	ForeImgIPara.push_back(ImgIParai);
	vector<double *>BackImgIPara;
	BackImgIPara.push_back(ImgIParai);

	double start = omp_get_wtime();
	printLOG("Foretrack  @(%d/%d):", startF, viewID);
	for (int ii = 0; ii < npts; ii++)
		AllRefFid[ii] = 0, FramesTrackedCount[ii] = 0, JustUpdate[ii] = 0, PermTrackFail[ii] = 0, TempTrackFail[ii] = 0;

	baseWarp.warp[0] = 1.0, baseWarp.warp[1] = 0.0, baseWarp.warp[2] = 0.0, baseWarp.warp[3] = 1.0;
	for (int ii = 0; ii < npts; ii++)
		cForeWarp[ii].push_back(baseWarp);

	for (int fid = 1; fid < ForeTrackRange; fid++)
	{
		int nvalidPoints = 0;
		for (int pid = 0; pid < npts; pid++)
			if (PermTrackFail[pid] == 0)
				nvalidPoints++;
		if (nvalidPoints > 0)
			printLOG("@f %d ... ", fid + startF);

		for (int jj = 0; jj < height; jj++)
		{
			const unsigned char* RowJJ = ForePyr[fid][0].ptr<unsigned char>(jj);
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width] = RowJJ[ii];
		}
		double *ImgIParai = new double[height*width];
		Generate_Para_Spline(Img, ImgIParai, width, height, LKArg.InterpAlgo);
		ForeImgIPara.push_back(ImgIParai);

#pragma omp parallel for schedule(dynamic,1)
		for (int pid = 0; pid < npts; pid++)
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (fid - AllRefFid[pid] > fps) //force update the template every 1s
				AllRefFid[pid] = (int)ForeTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0;

			//Look for the best window size with minimum drift
			TempTrackFail[pid] = 0;
			int refFid = AllRefFid[pid];
			int winsize = max(min((int)(ForeScale[pid][fid - 1] * 6 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
			int threadID = omp_get_thread_num();
			if (Track_1P_1F_WithRefTemplate(ForeTrackUV[pid], ForeWarp[pid], ForeiWarp[pid], ForePyr, ForeImgIPara[refFid], ForeImgIPara[fid], refFid, fid, pid, winsize, nWinSize, WinStep, cvPyrLevel, LKArg, T1+Tlength*threadID, T2+6*Tlength*threadID))
			{
				FramesTrackedCount[pid]++;
				if (LKArg.DIC_Algo != -2)
				{
					double ns = ForeScale[pid][refFid] * max(ForeWarp[pid][fid].warp[0] + 1.0 + ForeWarp[pid][fid].warp[1], ForeWarp[pid][fid].warp[2] + ForeWarp[pid][fid].warp[3] + 1.0);
					ForeScale[pid].push_back(ns);
				}
				else
				{
					double ns = ForeScale[pid][refFid] * ForeWarp[pid][fid].warp[0];
					ForeScale[pid].push_back(ns);
				}

				//update base template 
				baseWarp.warp[0] = cForeWarp[pid][refFid].warp[0], baseWarp.warp[1] = cForeWarp[pid][refFid].warp[1], baseWarp.warp[2] = cForeWarp[pid][refFid].warp[2], baseWarp.warp[3] = cForeWarp[pid][refFid].warp[3];

				//compute warping wrst 1st template
				if (LKArg.DIC_Algo != -2)
					currentWarp.warp[0] = ForeWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = ForeWarp[pid][fid].warp[1],
					currentWarp.warp[2] = ForeWarp[pid][fid].warp[2], currentWarp.warp[3] = ForeWarp[pid][fid].warp[3] + 1.0;
				else
				{
					double w_s = ForeWarp[pid][fid].warp[0];
					double w_a = ForeWarp[pid][fid].warp[1];
					currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
				}
				mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);
				cForeWarp[pid].push_back(cumWarp);
				/*{
				char Fname[512];
				int hsubset = winsize;
				double sImg[61 * 61],  II, JJ, S, ratio = ForeScale[pid][refFid] / ForeScale[pid][0];
				for (int jjj = -hsubset; jjj <= hsubset; jjj++)
				{
				for (int iii = -hsubset; iii <= hsubset; iii++)
				{
				II = ForeTrackUV[pid].back().x + (ForeWarp[pid][fid].warp[0] + 1.0) *iii * ratio + ForeWarp[pid][fid].warp[1] * jjj * ratio;
				JJ = ForeTrackUV[pid].back().y + ForeWarp[pid][fid].warp[2] * iii *ratio + (ForeWarp[pid][fid].warp[3] + 1.0) * jjj *ratio;
				Get_Value_Spline(ForeImgIPara[fid], 1920, 1080, II, JJ, &S, -1, LKArg.InterpAlgo);
				sImg[iii + hsubset + (jjj + hsubset)*(2 * hsubset + 1)] = S;
				}
				}
				sprintf(Fname, "%s/Patch/%d_%.4d.png", Path, pid, fid);
				SaveDataToImage(Fname, sImg, (2 * hsubset + 1), (2 * hsubset + 1), 1);

				for (int jjj = -hsubset; jjj <= hsubset; jjj++)
				{
				for (int iii = -hsubset; iii <= hsubset; iii++)
				{
				II = ForeTrackUV[pid].back().x + cumWarp.warp[0] * iii + cumWarp.warp[1] * jjj;
				JJ = ForeTrackUV[pid].back().y + cumWarp.warp[2] * iii + cumWarp.warp[3] * jjj;
				Get_Value_Spline(ForeImgIPara[fid], 1920, 1080, II, JJ, &S, -1, LKArg.InterpAlgo);
				sImg[iii + hsubset + (jjj + hsubset)*(2 * hsubset + 1)] = S;
				}
				}
				sprintf(Fname, "%s/Patch/_%d_%.4d.png", Path, pid, fid);
				SaveDataToImage(Fname, sImg, (2 * hsubset + 1), (2 * hsubset + 1), 1);

				double s = 1.0*winsize / 31;
				for (int jjj = -30; jjj <= 30; jjj++)
				{
				for (int iii = -30; iii <= 30; iii++)
				{
				II = ForeTrackUV[pid].back().x + cumWarp.warp[0] * s * iii + cumWarp.warp[1] * s * jjj;
				JJ = ForeTrackUV[pid].back().y + cumWarp.warp[2] * s * iii + cumWarp.warp[3] * s * jjj;
				Get_Value_Spline(ForeImgIPara[fid], 1920, 1080, II, JJ, &S, -1, LKArg.InterpAlgo);
				sImg[iii + 30 + (jjj + 30)*(2 * 30 + 1)] = S;
				}
				}
				sprintf(Fname, "%s/Patch/__%d_%.4d.png", Path, pid, fid);
				SaveDataToImage(Fname, sImg, (2 * 30 + 1), (2 * 30 + 1), 1);
				}*/
			}
			else
				TempTrackFail[pid] = 1;
		}

#pragma omp parallel for schedule(dynamic,1)
		for (int pid = 0; pid < npts; pid++) //Analyze the tracking results
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (TempTrackFail[pid] == 1)//tracking fails
			{
				if (FramesTrackedCount[pid] < 4) //just update the ref template but tracking last only a few frames --> SHOULD STOP THE TRACK (occluded points?)
				{
					ForeTrackUV[pid].erase(ForeTrackUV[pid].end() - FramesTrackedCount[pid], ForeTrackUV[pid].end());
					ForeScale[pid].erase(ForeScale[pid].end() - FramesTrackedCount[pid], ForeScale[pid].end());
					PermTrackFail[pid] = 1;
					//printLOG("(%d: %d) ... ", pid, ForeTrackUV[pid].size());
				}
				else //Lets update the template and re-run the track so that the point is up to the other points' progress
				{
					//if (debug)
					//printLOG("***(%d,%d)*** ", pid, fid + startF);
					AllRefFid[pid] = (int)ForeTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0, JustUpdate[pid] = 1;

					TempTrackFail[pid] = 0;
					int refFid = AllRefFid[pid];
					int winsize = max(min((int)(ForeScale[pid][fid - 1] * 6 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
					int threadID = omp_get_thread_num();
					if (Track_1P_1F_WithRefTemplate(ForeTrackUV[pid], ForeWarp[pid], ForeiWarp[pid], ForePyr, ForeImgIPara[refFid], ForeImgIPara[fid], refFid, fid, pid, winsize, nWinSize, WinStep, cvPyrLevel, LKArg, T1 + Tlength*threadID, T2 + 6 * Tlength*threadID))
					{
						FramesTrackedCount[pid]++;
						if (LKArg.DIC_Algo != -2)
						{
							double ns = ForeScale[pid][refFid] * max(ForeWarp[pid][fid].warp[0] + 1.0 + ForeWarp[pid][fid].warp[1], ForeWarp[pid][fid].warp[2] + ForeWarp[pid][fid].warp[3] + 1.0);
							ForeScale[pid].push_back(ns);
						}
						else
						{
							double ns = ForeScale[pid][refFid] * ForeWarp[pid][fid].warp[0];
							ForeScale[pid].push_back(ns);
						}

						//update base template
						baseWarp.warp[0] = cForeWarp[pid][refFid].warp[0], baseWarp.warp[1] = cForeWarp[pid][refFid].warp[1], baseWarp.warp[2] = cForeWarp[pid][refFid].warp[2], baseWarp.warp[3] = cForeWarp[pid][refFid].warp[3];

						//compute warping wrst 1st template
						if (LKArg.DIC_Algo != -2)
							currentWarp.warp[0] = ForeWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = ForeWarp[pid][fid].warp[1],
							currentWarp.warp[2] = ForeWarp[pid][fid].warp[2], currentWarp.warp[3] = ForeWarp[pid][fid].warp[3] + 1.0;
						else
						{
							double w_s = ForeWarp[pid][fid].warp[0];
							double w_a = ForeWarp[pid][fid].warp[1];
							currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
						}
						mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);

						cForeWarp[pid].push_back(cumWarp);
					}
					else //permanent failure
						PermTrackFail[pid] = 1;
				}
			}
		}

		/*//If Tracking is good, compute sift scale & desc
		#pragma omp parallel for
		for (int pid = 0; pid < npts; pid++)
		{
		if (TempTrackFail[pid] == 0 && PermTrackFail[pid] == 0)
		{
		KeyPoint key; key.pt = ForeTrackUV[pid].back();
		FeatureDesc fd;
		ComputeFeatureScaleAndDescriptor(ForePyr[fid][0], key, fd.desc);
		ForeDesc[pid].push_back(fd);
		ForeScale[pid].push_back(key.size);
		}

		if (JustUpdate[pid] == 1) //Compare its descriptor
		{
		int trackLength = ForeDesc[pid].size();
		double dist = 0.4*dotProduct(ForeDesc[pid].back().desc, ForeDesc[pid][0].desc, 128) + 0.6*dotProduct(ForeDesc[pid].back().desc, ForeDesc[pid][trackLength - 2].desc, 128);
		if (dist < descThresh) //Drift happens
		{
		ForeTrackUV[pid].erase(ForeTrackUV[pid].end() - 1);
		PermTrackFail[pid] = 1;
		}
		JustUpdate[pid] = 0;
		}
		}*/
	}
	if (debug)
		printLOG("\nTime: %.2fs\n", omp_get_wtime() - start);

	for (int ii = 1; ii < (int)ForeImgIPara.size(); ii++) //first one is shared with ForeImgIPara
		delete[]ForeImgIPara[ii];
	delete[]ForeWarp, delete[]ForeiWarp;

	start = omp_get_wtime();
	printLOG("\nBacktrack @(%d, %d):", startF, viewID);
	for (int ii = 0; ii < npts; ii++)
		AllRefFid[ii] = 0, FramesTrackedCount[ii] = 0, JustUpdate[ii] = 0, PermTrackFail[ii] = 0, TempTrackFail[ii] = 0;

	baseWarp.warp[0] = 1.0, baseWarp.warp[1] = 0.0, baseWarp.warp[2] = 0.0, baseWarp.warp[3] = 1.0;
	for (int ii = 0; ii < npts; ii++)
		cBackWarp[ii].push_back(baseWarp);
	for (int fid = 1; fid < BackTrackRange; fid++)
	{
		int nvalidPoints = 0;
		for (int pid = 0; pid < npts; pid++)
			if (PermTrackFail[pid] == 0)
				nvalidPoints++;
		if (nvalidPoints > 0)
			printLOG("@f %d ... ", -fid + startF);

		for (int jj = 0; jj < height; jj++)
		{
			const unsigned char* RowJJ = BackPyr[fid][0].ptr<unsigned char>(jj);
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width] = RowJJ[ii];
		}
		double *ImgIParai = new double[height*width];
		Generate_Para_Spline(Img, ImgIParai, width, height, LKArg.InterpAlgo);
		BackImgIPara.push_back(ImgIParai);

#pragma omp parallel for schedule(dynamic,1)
		for (int pid = 0; pid < npts; pid++)
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (fid - AllRefFid[pid] > fps) //force update the template every 1s
				AllRefFid[pid] = (int)BackTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0;

			//Look for the best window size  with minimum drift
			TempTrackFail[pid] = 0;
			int refFid = AllRefFid[pid];
			int winsize = max(min((int)(BackScale[pid][fid - 1] * 6 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
			int threadID = omp_get_thread_num();
			if (Track_1P_1F_WithRefTemplate(BackTrackUV[pid], BackWarp[pid], BackiWarp[pid], BackPyr, BackImgIPara[refFid], BackImgIPara[fid], refFid, fid, pid, winsize, nWinSize, WinStep, cvPyrLevel, LKArg, T1 + Tlength*threadID, T2 + 6 * Tlength*threadID))
			{
				FramesTrackedCount[pid]++;
				if (LKArg.DIC_Algo != -2)
				{
					double ns = BackScale[pid][refFid] * max(BackWarp[pid][fid].warp[0] + 1.0 + BackWarp[pid][fid].warp[1], BackWarp[pid][fid].warp[2] + BackWarp[pid][fid].warp[3] + 1.0);
					BackScale[pid].push_back(ns);
				}
				else
				{
					double ns = BackScale[pid][refFid] * BackWarp[pid][fid].warp[0];
					BackScale[pid].push_back(ns);
				}

				//update base template
				baseWarp.warp[0] = cBackWarp[pid][refFid].warp[0], baseWarp.warp[1] = cBackWarp[pid][refFid].warp[1], baseWarp.warp[2] = cBackWarp[pid][refFid].warp[2], baseWarp.warp[3] = cBackWarp[pid][refFid].warp[3];

				//compute warping wrst 1st template
				if (LKArg.DIC_Algo != -2)
					currentWarp.warp[0] = BackWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = BackWarp[pid][fid].warp[1],
					currentWarp.warp[2] = BackWarp[pid][fid].warp[2], currentWarp.warp[3] = BackWarp[pid][fid].warp[3] + 1.0;
				else
				{
					double w_s = BackWarp[pid][fid].warp[0];
					double w_a = BackWarp[pid][fid].warp[1];
					currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
				}
				mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);

				cBackWarp[pid].push_back(cumWarp);

				/*{
				char Fname[512];
				const int hsubset = 30;
				double sImg[61 * 61];
				double II, JJ, S, ratio = BackScale[pid][refFid] / BackScale[pid][0];
				for (int jjj = -hsubset; jjj <= hsubset; jjj++)
				{
				for (int iii = -hsubset; iii <= hsubset; iii++)
				{
				II = BackTrackUV[pid].back().x + (BackWarp[pid][fid].warp[0] + 1.0) *iii * ratio + BackWarp[pid][fid].warp[1] * jjj * ratio;
				JJ = BackTrackUV[pid].back().y + BackWarp[pid][fid].warp[2] * iii *ratio + (BackWarp[pid][fid].warp[3] + 1.0) * jjj *ratio;
				Get_Value_Spline(BackImgIPara[fid], 1920, 1080, II, JJ, &S, -1, LKArg.InterpAlgo);
				sImg[iii + hsubset + (jjj + hsubset)*(2 * hsubset + 1)] = S;
				}
				}
				sprintf(Fname, "%s/Patch/%d_%.4d.png", Path, pid, -fid);
				SaveDataToImage(Fname, sImg, (2 * hsubset + 1), (2 * hsubset + 1), 1);

				for (int jjj = -hsubset; jjj <= hsubset; jjj++)
				{
				for (int iii = -hsubset; iii <= hsubset; iii++)
				{
				II = BackTrackUV[pid].back().x + cumWarp.warp[0] * iii + cumWarp.warp[1] * jjj;
				JJ = BackTrackUV[pid].back().y + cumWarp.warp[2] * iii + cumWarp.warp[3] * jjj;
				Get_Value_Spline(BackImgIPara[fid], 1920, 1080, II, JJ, &S, -1, LKArg.InterpAlgo);
				sImg[iii + hsubset + (jjj + hsubset)*(2 * hsubset + 1)] = S;
				}
				}
				sprintf(Fname, "%s/Patch/_%d_%.4d.png", Path, pid, -fid);
				SaveDataToImage(Fname, sImg, (2 * hsubset + 1), (2 * hsubset + 1), 1);
				}*/
			}
			else
				TempTrackFail[pid] = 1;
		}

#pragma omp parallel for schedule(dynamic,1)
		for (int pid = 0; pid < npts; pid++) //Analyze the tracking results
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (TempTrackFail[pid] == 1)//tracking fails
			{
				if (FramesTrackedCount[pid] < 4) //just update the ref template but tracking last only a few frames --> SHOULD STOP THE TRACK (occluded points?)
				{
					BackTrackUV[pid].erase(BackTrackUV[pid].end() - FramesTrackedCount[pid], BackTrackUV[pid].end());
					BackScale[pid].erase(BackScale[pid].end() - FramesTrackedCount[pid], BackScale[pid].end());
					PermTrackFail[pid] = 1;
					//printLOG("(%d: %d) ... ", pid, BackTrackUV[pid].size());
				}
				else //Lets update the template and re-run the track so that the point is up to the other points' progress
				{
					//if (debug)
					//printLOG("***(%d,%d)*** ", pid, fid + startF);

					AllRefFid[pid] = (int)BackTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0, JustUpdate[pid] = 1;

					TempTrackFail[pid] = 0;
					int refFid = AllRefFid[pid];
					int winsize = max(min((int)(BackScale[pid][fid - 1] * 6 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
					int threadID = omp_get_thread_num();
					if (Track_1P_1F_WithRefTemplate(BackTrackUV[pid], BackWarp[pid], BackiWarp[pid], BackPyr, BackImgIPara[refFid], BackImgIPara[fid], refFid, fid, pid, winsize, nWinSize, WinStep, cvPyrLevel, LKArg, T1 + Tlength*threadID, T2 + 6 * Tlength*threadID))
					{
						FramesTrackedCount[pid]++;
						if (LKArg.DIC_Algo != -2)
						{
							double ns = BackScale[pid][refFid] * max(BackWarp[pid][fid].warp[0] + 1.0 + BackWarp[pid][fid].warp[1], BackWarp[pid][fid].warp[2] + BackWarp[pid][fid].warp[3] + 1.0);
							BackScale[pid].push_back(ns);
						}
						else
						{
							double ns = BackScale[pid][refFid] * BackWarp[pid][fid].warp[0];
							BackScale[pid].push_back(ns);
						}

						//update base template
						baseWarp.warp[0] = cBackWarp[pid][refFid].warp[0], baseWarp.warp[1] = cBackWarp[pid][refFid].warp[1], baseWarp.warp[2] = cBackWarp[pid][refFid].warp[2], baseWarp.warp[3] = cBackWarp[pid][refFid].warp[3];

						//compute warping wrst 1st template
						if (LKArg.DIC_Algo != -2)
							currentWarp.warp[0] = BackWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = BackWarp[pid][fid].warp[1],
							currentWarp.warp[2] = BackWarp[pid][fid].warp[2], currentWarp.warp[3] = BackWarp[pid][fid].warp[3] + 1.0;
						else
						{
							double w_s = BackWarp[pid][fid].warp[0];
							double w_a = BackWarp[pid][fid].warp[1];
							currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
						}
						mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);

						cBackWarp[pid].push_back(cumWarp);
					}
					else //permanent failure
						PermTrackFail[pid] = 1;
				}
			}
		}
		/*//If Tracking is good, compute sift scale & desc
		#pragma omp parallel for
		for (int pid = 0; pid < npts; pid++)
		{
		if (TempTrackFail[pid] == 0 && PermTrackFail[pid] == 0)
		{
		KeyPoint key; key.pt = BackTrackUV[pid].back();
		FeatureDesc fd;
		ComputeFeatureScaleAndDescriptor(BackPyr[fid][0], key, fd.desc);
		BackDesc[pid].push_back(fd);
		BackScale[pid].push_back(key.size);
		}

		if (JustUpdate[pid] == 1) //Compare its descriptor
		{
		int trackLength = BackDesc[pid].size();
		double dist = 0.4*dotProduct(ForeDesc[pid].back().desc, BackDesc[pid][0].desc, 128) + 0.6*dotProduct(BackDesc[pid].back().desc, BackDesc[pid][trackLength - 2].desc, 128);
		if (dist < descThresh) //Drift happens
		{
		BackTrackUV[pid].erase(BackTrackUV[pid].end() - 1);
		PermTrackFail[pid] = 1;
		}
		JustUpdate[pid] = 0;
		}
		}*/
	}
	if (debug)
		printLOG("\nTime: %.2fs\n", omp_get_wtime() - start);

	for (int ii = 0; ii < (int)BackImgIPara.size(); ii++)
		delete[]BackImgIPara[ii];
	delete[]BackWarp, delete[]BackiWarp;

	delete[]Img, delete[]T1, delete[]T2;
	delete[]AllRefFid, delete[]FramesTrackedCount, delete[]JustUpdate, delete[]PermTrackFail, delete[]TempTrackFail;

	return 0;
}

int Track_1P_1F_WithRefTemplate_DenseFlowDriven(vector<Point2f> &TrackUV, vector<AffinePara> &warp, vector<AffinePara> &iwarp, vector<Mat> *ImgPyr, double *RefPara, double *CPara, float* DFx, float*DFy, int reffid, int fid, int pid, int MaxWinSize, int nWinSize, int WinStep, LKParameters LKArg, double *T1, double *T2)
{
	int width = ImgPyr[fid][0].cols, height = ImgPyr[fid][0].rows, orghsubset = MaxWinSize / 2;
	double bestcwarp[4], besticwarp[4], cwarp[4], icwarp[4], minDist = 9e9;
	Point2f BestNewPt(-1, -1);  Point2d refpt, npt, brefpt;
	double dispThresh2 = LKArg.DisplacementThresh*LKArg.DisplacementThresh;

	for (int trial = 0; trial < nWinSize; trial++)
	{
		if (TrackUV.back().x <MaxWinSize || TrackUV.back().y < MaxWinSize || TrackUV.back().x >ImgPyr[fid - 1][0].cols - MaxWinSize || TrackUV.back().y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		int u = (int)(TrackUV.back().x + 0.5), v = (int)(TrackUV.back().y + 0.5);
		float dfx = DFx[u + v*width], dfy = DFy[u + v*width];
		Point2d npt(dfx + u, dfy + v);
		if (npt.x <MaxWinSize || npt.y < MaxWinSize || npt.x >ImgPyr[fid - 1][0].cols - MaxWinSize || npt.y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		LKArg.hsubset = orghsubset - trial*WinStep / 2;
		refpt = TrackUV[reffid], brefpt = TrackUV[reffid];

		for (int ii = 0; ii < 4; ii++)
			cwarp[ii] = warp[fid - 1].warp[ii];
		double score1 = TemplateMatching(RefPara, CPara, width, height, width, height, 1, refpt, npt, LKArg, false, T1, T2, cwarp);
		if (score1 < LKArg.ZNCCThreshold)
			continue;

		if (npt.x <MaxWinSize || npt.y < MaxWinSize || npt.x >ImgPyr[fid - 1][0].cols - MaxWinSize || npt.y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		for (int ii = 0; ii < 4; ii++)
			icwarp[ii] = iwarp[fid - 1].warp[ii];
		double score2 = TemplateMatching(CPara, RefPara, width, height, width, height, 1, npt, brefpt, LKArg, false, T1, T2, icwarp);
		if (score2 < LKArg.ZNCCThreshold)
			continue;

		if (brefpt.x <MaxWinSize || brefpt.y < MaxWinSize || brefpt.x >ImgPyr[fid - 1][0].cols - MaxWinSize || brefpt.y > ImgPyr[fid - 1][0].rows - MaxWinSize)
			continue;

		double dist = pow(brefpt.x - refpt.x, 2) + pow(brefpt.y - refpt.y, 2);
		if (dist < minDist && dist < dispThresh2)
		{
			minDist = dist;
			BestNewPt = npt;
			for (int ii = 0; ii < 4; ii++)
				bestcwarp[ii] = cwarp[ii], besticwarp[ii] = icwarp[ii];
		}
	}

	if (minDist < dispThresh2)
	{
		TrackUV.push_back(BestNewPt);
		AffinePara w1(bestcwarp); warp.push_back(w1);
		AffinePara w2(besticwarp); iwarp.push_back(w2);
		return 1;
	}
	else
		return 0;
}
int TrackAllPointsWithRefTemplate_DenseFlowDriven(char *Path, int viewID, int startF, vector<Point2f> uvRef, vector<float> sRef, vector<Point2f> *ForeTrackUV, vector<Point2f> *BackTrackUV,
	vector<float> *ForeScale, vector<float> *BackScale, vector<AffinePara> *cForeWarp, vector<AffinePara> *cBackWarp, vector<FeatureDesc> *ForeDesc, vector<FeatureDesc> *BackDesc,
	vector<Mat> *ForePyr, vector<Mat> *BackPyr, vector<float*>DFx, vector<float*>DFy, vector<float*>DBx, vector<float*>DBy, int MaxWinSize, int nWinSize, int WinStep, double fps, int ForeTrackRange, int BackTrackRange, int noTemplateUpdate, int interpAlgo)
{
	const double  descThresh = 0.7;
	double ratio = 1.0*ForePyr[0][0].cols / 1920;

	LKParameters LKArg; //LKArg.hsubset to be changed according to the point scale
	LKArg.DisplacementThresh = 0.5, LKArg.DIC_Algo = 3, LKArg.InterpAlgo = interpAlgo, LKArg.EpipEnforce = 0;
	LKArg.Incomplete_Subset_Handling = 0, LKArg.Convergence_Criteria = 0, LKArg.Analysis_Speed = 0, LKArg.IterMax = 15;
	LKArg.PSSDab_thresh = 0.1, LKArg.ZNCCThreshold = 0.7;

	int npts = (int)uvRef.size();
	vector<AffinePara> *ForeWarp = new vector<AffinePara>[npts];
	vector<AffinePara> *ForeiWarp = new vector<AffinePara>[npts];
	vector<AffinePara> *BackWarp = new vector<AffinePara>[npts];
	vector<AffinePara> *BackiWarp = new vector<AffinePara>[npts];
	AffinePara w1, w2, w3, w4, currentWarp, cumWarp, baseWarp;

#pragma omp parallel for
	for (int ii = 0; ii < npts; ii++)
	{
		ForeTrackUV[ii].reserve(ForeTrackRange), ForeTrackUV[ii].push_back(uvRef[ii]);
		BackTrackUV[ii].reserve(BackTrackRange), BackTrackUV[ii].push_back(uvRef[ii]);

		ForeScale[ii].reserve(ForeTrackRange), ForeScale[ii].push_back(sRef[ii]);
		BackScale[ii].reserve(BackTrackRange), BackScale[ii].push_back(sRef[ii]);

		ForeWarp[ii].reserve(ForeTrackRange), BackWarp[ii].reserve(ForeTrackRange);
		ForeiWarp[ii].reserve(ForeTrackRange), BackiWarp[ii].reserve(ForeTrackRange);

		AffinePara w1, w2, w3, w4;
		ForeWarp[ii].push_back(w1), BackWarp[ii].push_back(w2);
		ForeiWarp[ii].push_back(w3), BackiWarp[ii].push_back(w4);
	}

	int *AllRefFid = new int[npts], *FramesTrackedCount = new int[npts], *JustUpdate = new int[npts], *PermTrackFail = new int[npts], *TempTrackFail = new int[npts];
	double *T1 = new double[MaxWinSize*MaxWinSize], *T2 = new double[6 * MaxWinSize*MaxWinSize];

	int width = ForePyr[0][0].cols, height = ForePyr[0][0].rows;
	unsigned char *Img = new unsigned char[width*height];

	for (int jj = 0; jj < height; jj++)
	{
		const unsigned char* RowJJ = ForePyr[0][0].ptr<unsigned char>(jj);
		for (int ii = 0; ii < width; ii++)
			Img[ii + jj*width] = RowJJ[ii];
	}
	double *ImgIParai = new double[height*width];
	Generate_Para_Spline(Img, ImgIParai, width, height, LKArg.InterpAlgo);

	vector<double *>ForeImgIPara;
	ForeImgIPara.push_back(ImgIParai);
	vector<double *>BackImgIPara;
	BackImgIPara.push_back(ImgIParai);

	double start = omp_get_wtime();
	printLOG("Foretrack  @%d:", startF);
	for (int ii = 0; ii < npts; ii++)
		AllRefFid[ii] = 0, FramesTrackedCount[ii] = 0, JustUpdate[ii] = 0, PermTrackFail[ii] = 0, TempTrackFail[ii] = 0;

	baseWarp.warp[0] = 1.0, baseWarp.warp[1] = 0.0, baseWarp.warp[2] = 0.0, baseWarp.warp[3] = 1.0;
	for (int ii = 0; ii < npts; ii++)
		cForeWarp[ii].push_back(baseWarp);

	for (int fid = 1; fid < ForeTrackRange; fid++)
	{
		int nvalidPoints = 0;
		for (int pid = 0; pid < npts; pid++)
			if (PermTrackFail[pid] == 0)
				nvalidPoints++;
		if (nvalidPoints > 0)
			printLOG("@f %d ... ", fid + startF);

		for (int jj = 0; jj < height; jj++)
		{
			const unsigned char* RowJJ = ForePyr[fid][0].ptr<unsigned char>(jj);
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width] = RowJJ[ii];
		}
		double *ImgIParai = new double[height*width];
		Generate_Para_Spline(Img, ImgIParai, width, height, LKArg.InterpAlgo);
		ForeImgIPara.push_back(ImgIParai);

		for (int pid = 0; pid < npts; pid++)
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (fid - AllRefFid[pid] > fps) //force update the template every 1s
				AllRefFid[pid] = (int)ForeTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0;

			//Look for the best window size with minimum drift
			TempTrackFail[pid] = 0;
			int refFid = AllRefFid[pid];
			int winsize = max(min((int)(ForeScale[pid][fid - 1] * 6 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
			if (Track_1P_1F_WithRefTemplate_DenseFlowDriven(ForeTrackUV[pid], ForeWarp[pid], ForeiWarp[pid], ForePyr, ForeImgIPara[refFid], ForeImgIPara[fid], DFx[fid - 1], DFy[fid - 1], refFid, fid, pid, winsize, nWinSize, WinStep, LKArg, T1, T2))
			{
				//Newer code check the ZNCC score internally with affine warping
				FramesTrackedCount[pid]++;
				if (LKArg.DIC_Algo != -2)
				{
					double ns = ForeScale[pid][refFid] * max(ForeWarp[pid][fid].warp[0] + 1.0 + ForeWarp[pid][fid].warp[1], ForeWarp[pid][fid].warp[2] + ForeWarp[pid][fid].warp[3] + 1.0);
					ForeScale[pid].push_back(ns);
				}
				else
				{
					double ns = ForeScale[pid][refFid] * ForeWarp[pid][fid].warp[0];
					ForeScale[pid].push_back(ns);
				}

				//update base template 
				baseWarp.warp[0] = cForeWarp[pid][refFid].warp[0], baseWarp.warp[1] = cForeWarp[pid][refFid].warp[1], baseWarp.warp[2] = cForeWarp[pid][refFid].warp[2], baseWarp.warp[3] = cForeWarp[pid][refFid].warp[3];

				//compute warping wrst 1st template
				if (LKArg.DIC_Algo != -2)
					currentWarp.warp[0] = ForeWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = ForeWarp[pid][fid].warp[1],
					currentWarp.warp[2] = ForeWarp[pid][fid].warp[2], currentWarp.warp[3] = ForeWarp[pid][fid].warp[3] + 1.0;
				else
				{
					double w_s = ForeWarp[pid][fid].warp[0];
					double w_a = ForeWarp[pid][fid].warp[1];
					currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
				}
				mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);
				cForeWarp[pid].push_back(cumWarp);
			}
			else
				TempTrackFail[pid] = 1;
		}

		for (int pid = 0; pid < npts; pid++) //Analyze the tracking results
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (TempTrackFail[pid] == 1 && noTemplateUpdate == 1)
			{
				PermTrackFail[pid] = 1;
				continue;
			}

			if (TempTrackFail[pid] == 1)//tracking fails
			{
				if (FramesTrackedCount[pid] < fps / 6) //just update the ref template but tracking last only a few frames --> SHOULD STOP THE TRACK (occluded points?)
				{
					ForeTrackUV[pid].erase(ForeTrackUV[pid].end() - FramesTrackedCount[pid], ForeTrackUV[pid].end());
					ForeScale[pid].erase(ForeScale[pid].end() - FramesTrackedCount[pid], ForeScale[pid].end());
					PermTrackFail[pid] = 1;
					//printLOG("(%d: %d) ... ", pid, ForeTrackUV[pid].size());
				}
				else //Lets update the template and re-run the track so that the point is up to the other points' progress
				{
					//printLOG("***(%d, %d)*** ", pid, fid + startF);
					AllRefFid[pid] = (int)ForeTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0, JustUpdate[pid] = 1;

					TempTrackFail[pid] = 0;
					int refFid = AllRefFid[pid];
					int winsize = max(min((int)(ForeScale[pid][fid - 1] * 6 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
					if (Track_1P_1F_WithRefTemplate_DenseFlowDriven(ForeTrackUV[pid], ForeWarp[pid], ForeiWarp[pid], ForePyr, ForeImgIPara[refFid], ForeImgIPara[fid], DFx[fid - 1], DFy[fid - 1], refFid, fid, pid, winsize, nWinSize, WinStep, LKArg, T1, T2))
					{
						FramesTrackedCount[pid]++;
						if (LKArg.DIC_Algo != -2)
						{
							double ns = ForeScale[pid][refFid] * max(ForeWarp[pid][fid].warp[0] + 1.0 + ForeWarp[pid][fid].warp[1], ForeWarp[pid][fid].warp[2] + ForeWarp[pid][fid].warp[3] + 1.0);
							ForeScale[pid].push_back(ns);
						}
						else
						{
							double ns = ForeScale[pid][refFid] * ForeWarp[pid][fid].warp[0];
							ForeScale[pid].push_back(ns);
						}

						//update base template
						baseWarp.warp[0] = cForeWarp[pid][refFid].warp[0], baseWarp.warp[1] = cForeWarp[pid][refFid].warp[1], baseWarp.warp[2] = cForeWarp[pid][refFid].warp[2], baseWarp.warp[3] = cForeWarp[pid][refFid].warp[3];

						//compute warping wrst 1st template
						if (LKArg.DIC_Algo != -2)
							currentWarp.warp[0] = ForeWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = ForeWarp[pid][fid].warp[1],
							currentWarp.warp[2] = ForeWarp[pid][fid].warp[2], currentWarp.warp[3] = ForeWarp[pid][fid].warp[3] + 1.0;
						else
						{
							double w_s = ForeWarp[pid][fid].warp[0], w_a = ForeWarp[pid][fid].warp[1];
							currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
						}
						mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);

						cForeWarp[pid].push_back(cumWarp);
					}
					else //permanent failure
						PermTrackFail[pid] = 1;
				}
			}
		}
	}
	//printLOG("\nTime: %.2fs\n", omp_get_wtime() - start);

	for (int ii = 1; ii < (int)ForeImgIPara.size(); ii++) //first one is shared with ForeImgIPara
		delete[]ForeImgIPara[ii];
	delete[]ForeWarp, delete[]ForeiWarp;

	start = omp_get_wtime();
	printLOG("\nBacktrack @%d:", startF);
	for (int ii = 0; ii < npts; ii++)
		AllRefFid[ii] = 0, FramesTrackedCount[ii] = 0, JustUpdate[ii] = 0, PermTrackFail[ii] = 0, TempTrackFail[ii] = 0;

	baseWarp.warp[0] = 1.0, baseWarp.warp[1] = 0.0, baseWarp.warp[2] = 0.0, baseWarp.warp[3] = 1.0;
	for (int ii = 0; ii < npts; ii++)
		cBackWarp[ii].push_back(baseWarp);
	for (int fid = 1; fid < BackTrackRange; fid++)
	{
		int nvalidPoints = 0;
		for (int pid = 0; pid < npts; pid++)
			if (PermTrackFail[pid] == 0)
				nvalidPoints++;
		if (nvalidPoints > 0)
			printLOG("@f %d ... ", -fid + startF);

		for (int jj = 0; jj < height; jj++)
		{
			const unsigned char* RowJJ = BackPyr[fid][0].ptr<unsigned char>(jj);
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width] = RowJJ[ii];
		}
		double *ImgIParai = new double[height*width];
		Generate_Para_Spline(Img, ImgIParai, width, height, LKArg.InterpAlgo);
		BackImgIPara.push_back(ImgIParai);

		for (int pid = 0; pid < npts; pid++)
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (fid - AllRefFid[pid] > fps) //force update the template every 1s
				AllRefFid[pid] = (int)BackTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0;

			//Look for the best window size  with minimum drift
			TempTrackFail[pid] = 0;
			int refFid = AllRefFid[pid];
			int winsize = max(min((int)(BackScale[pid][fid - 1] * 6.0 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
			if (Track_1P_1F_WithRefTemplate_DenseFlowDriven(BackTrackUV[pid], BackWarp[pid], BackiWarp[pid], BackPyr, BackImgIPara[refFid], BackImgIPara[fid], DBx[fid - 1], DBy[fid - 1], refFid, fid, pid, winsize, nWinSize, WinStep, LKArg, T1, T2))
			{
				FramesTrackedCount[pid]++;
				if (LKArg.DIC_Algo != -2)
				{
					double ns = BackScale[pid][refFid] * max(BackWarp[pid][fid].warp[0] + 1.0 + BackWarp[pid][fid].warp[1], BackWarp[pid][fid].warp[2] + BackWarp[pid][fid].warp[3] + 1.0);
					BackScale[pid].push_back(ns);
				}
				else
				{
					double ns = BackScale[pid][refFid] * BackWarp[pid][fid].warp[0];
					BackScale[pid].push_back(ns);
				}

				//update base template
				baseWarp.warp[0] = cBackWarp[pid][refFid].warp[0], baseWarp.warp[1] = cBackWarp[pid][refFid].warp[1], baseWarp.warp[2] = cBackWarp[pid][refFid].warp[2], baseWarp.warp[3] = cBackWarp[pid][refFid].warp[3];

				//compute warping wrst 1st template
				if (LKArg.DIC_Algo != -2)
					currentWarp.warp[0] = BackWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = BackWarp[pid][fid].warp[1],
					currentWarp.warp[2] = BackWarp[pid][fid].warp[2], currentWarp.warp[3] = BackWarp[pid][fid].warp[3] + 1.0;
				else
				{
					double w_s = BackWarp[pid][fid].warp[0], w_a = BackWarp[pid][fid].warp[1];
					currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
				}
				mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);

				cBackWarp[pid].push_back(cumWarp);
			}
			else
				TempTrackFail[pid] = 1;
		}

		for (int pid = 0; pid < npts; pid++) //Analyze the tracking results
		{
			if (PermTrackFail[pid] == 1)
				continue;

			if (TempTrackFail[pid] == 1 && noTemplateUpdate == 1)
			{
				PermTrackFail[pid] = 1;
				continue;
			}

			if (TempTrackFail[pid] == 1)//tracking fails
			{
				if (FramesTrackedCount[pid] < fps / 6) //just update the ref template but tracking last only a few frames --> SHOULD STOP THE TRACK (occluded points?)
				{
					BackTrackUV[pid].erase(BackTrackUV[pid].end() - FramesTrackedCount[pid], BackTrackUV[pid].end());
					PermTrackFail[pid] = 1;
					//printLOG("(%d: %d) ... ", pid, BackTrackUV[pid].size());
				}
				else //Lets update the template and re-run the track so that the point is up to the other points' progress
				{
					//printLOG("***(%d, %d)*** ", pid, -fid + startF);
					AllRefFid[pid] = (int)BackTrackUV[pid].size() - 1, FramesTrackedCount[pid] = 0, JustUpdate[pid] = 1;

					TempTrackFail[pid] = 0;
					int refFid = AllRefFid[pid];
					int winsize = max(min((int)(BackScale[pid][fid - 1] * 6.0 + 0.5), MaxWinSize), (int)(ratio * 11 + 0.5) / 2 * 2 + 1);
					if (Track_1P_1F_WithRefTemplate_DenseFlowDriven(BackTrackUV[pid], BackWarp[pid], BackiWarp[pid], BackPyr, BackImgIPara[refFid], BackImgIPara[fid], DBx[fid - 1], DBy[fid - 1], refFid, fid, pid, winsize, nWinSize, WinStep, LKArg, T1, T2))
					{
						FramesTrackedCount[pid]++;
						if (LKArg.DIC_Algo != -2)
						{
							double ns = BackScale[pid][refFid] * max(BackWarp[pid][fid].warp[0] + 1.0 + BackWarp[pid][fid].warp[1], BackWarp[pid][fid].warp[2] + BackWarp[pid][fid].warp[3] + 1.0);
							BackScale[pid].push_back(ns);
						}
						else
						{
							double ns = BackScale[pid][refFid] * BackWarp[pid][fid].warp[0];
							BackScale[pid].push_back(ns);
						}

						//update base template
						baseWarp.warp[0] = cBackWarp[pid][refFid].warp[0], baseWarp.warp[1] = cBackWarp[pid][refFid].warp[1], baseWarp.warp[2] = cBackWarp[pid][refFid].warp[2], baseWarp.warp[3] = cBackWarp[pid][refFid].warp[3];

						//compute warping wrst 1st template
						if (LKArg.DIC_Algo != -2)
							currentWarp.warp[0] = BackWarp[pid][fid].warp[0] + 1.0, currentWarp.warp[1] = BackWarp[pid][fid].warp[1],
							currentWarp.warp[2] = BackWarp[pid][fid].warp[2], currentWarp.warp[3] = BackWarp[pid][fid].warp[3] + 1.0;
						else
						{
							double w_s = BackWarp[pid][fid].warp[0];
							double w_a = BackWarp[pid][fid].warp[1];
							currentWarp.warp[0] = w_s*cos(w_a), currentWarp.warp[1] = -w_s*sin(w_a), currentWarp.warp[2] = w_s*sin(w_a), currentWarp.warp[3] = w_s*cos(w_a);
						}
						mat_mul(baseWarp.warp, currentWarp.warp, cumWarp.warp, 2, 2, 2);

						cBackWarp[pid].push_back(cumWarp);
					}
					else //permanent failure
						PermTrackFail[pid] = 1;
				}
			}
		}
	}
	//printLOG("\nTime: %.2fs\n", omp_get_wtime() - start);

	for (int ii = 0; ii < (int)BackImgIPara.size(); ii++)
		delete[]BackImgIPara[ii];
	delete[]BackWarp, delete[]BackiWarp;

	delete[]Img, delete[]T1, delete[]T2;
	delete[]AllRefFid, delete[]FramesTrackedCount, delete[]JustUpdate, delete[]PermTrackFail, delete[]TempTrackFail;

	return 0;
}


bool IsLocalWarpAvail(float *WarpingParas, double *iWp, int startX, int startY, Point2i &startf, int &xx, int &yy, int &range, int width, int height, int nearestRange = 7)
{
	int id, kk, ll, mm, nn, kk2, length = width*height;
	int sRange = width / 500;

	bool flag = 0;
	for (kk = 0; kk < nearestRange && !flag; kk++)
	{
		kk2 = kk*kk;
		for (mm = -kk; mm <= kk && !flag; mm++)
		{
			for (nn = -kk; nn <= kk; nn++)
			{
				if (mm*mm + nn*nn < kk2)
					continue;
				if (abs(WarpingParas[(startX + nn) + (startY + mm)*width]) + abs(WarpingParas[(startX + nn) + (startY + mm)*width + length]) > 0.001)
				{
					xx = startX + nn + (int)(WarpingParas[(startX + nn) + (startY + mm)*width] + 0.5);
					yy = startY + mm + (int)(WarpingParas[(startX + nn) + (startY + mm)*width + length] + 0.5);
					flag = true, startf.x = startX + nn, startf.y = startY + mm, range = kk; //Adaptively change the search range

					//Get the affine coeffs:
					id = (startX + nn) + (startY + mm)*width;
					for (ll = 2; ll < 6; ll++)
						iWp[ll - 2] = WarpingParas[id + ll*length];
					break;
				}
			}
		}
	}

	if (!flag)
		return false;
	else
		return true;
}
void DIC_FindROI(char *lpROI, Point2i Start_Point, int width, int height, Point2i *bound)
{
	//bound[0]: bottom left, bound[1] = top right
	int m, n, x, y;
	int length = width*height;

	int *Txy = new int[length * 2];
	for (n = 0; n < length; n++)
		*(lpROI + n) = (char)0;

	m = 0;
	x = Start_Point.x;
	y = Start_Point.y;
	*(lpROI + y*width + x) = (char)255;
	*(Txy + 2 * m + 0) = x;
	*(Txy + 2 * m + 1) = y;
	while (m >= 0)
	{
		x = *(Txy + 2 * m + 0);
		y = *(Txy + 2 * m + 1);
		m--;

		if ((y + 1) < bound[1].y && *(lpROI + (y + 1)*width + x) == (char)0)
		{
			m++;
			*(lpROI + (y + 1)*width + x) = (char)255;
			*(Txy + 2 * m + 0) = x;
			*(Txy + 2 * m + 1) = y + 1;
		}
		if (y > bound[0].y && *(lpROI + (y - 1)*width + x) == (char)0)
		{
			m++;
			*(lpROI + (y - 1)*width + x) = (char)255;
			*(Txy + 2 * m + 0) = x;
			*(Txy + 2 * m + 1) = y - 1;
		}
		if (x > bound[0].x && *(lpROI + y*width + x - 1) == (char)0)
		{
			m++;
			*(lpROI + y*width + x - 1) = (char)255;
			*(Txy + 2 * m + 0) = x - 1;
			*(Txy + 2 * m + 1) = y;
		}
		if ((x + 1) < bound[1].x && *(lpROI + y*width + x + 1) == (char)0)
		{
			m++;
			*(lpROI + y*width + x + 1) = (char)255;
			*(Txy + 2 * m + 0) = x + 1;
			*(Txy + 2 * m + 1) = y;
		}
	}

	delete[]Txy;

	return;
}
void DIC_AddtoQueue(double *Coeff, int *Tindex, int M)
{
	int i, j, t;
	double coeff;
	for (i = 0; i <= M - 1; i++)
	{
		if (*(Coeff + M) > *(Coeff + i))
		{
			coeff = *(Coeff + M), t = *(Tindex + M);
			for (j = M - 1; j >= i; j--)
				*(Coeff + j + 1) = *(Coeff + j), *(Tindex + j + 1) = *(Tindex + j);
			*(Coeff + i) = coeff, *(Tindex + i) = t;
			break;
		}
	}

	return;
}
bool DIC_CheckPointValidity(bool *lpROI, int x_n, int y_n, int width, int height, int hsubset, double validity_ratio)
{
	int m = 0, n = 0, ii, jj, iii, jjj;
	for (jjj = -hsubset; jjj <= hsubset; jjj += 2)
	{
		for (iii = -hsubset; iii <= hsubset; iii += 2)
		{
			m++;
			jj = y_n + jjj, ii = x_n + iii;
			if (ii<0 || ii>(width - 1) || jj<0 || jj>(height - 1))
				continue;
			if (*(lpROI + jj*width + ii) == false)
				continue;
			n++;
		}
	}

	if (n < int(m*validity_ratio))
		return false;

	return true;
}
void DIC_Initial_Guess(double *lpImageData, int width, int height, double *UV_Guess, Point2i Start_Point, int *IG_subset, int Initial_Guess_Scheme)
{
	int i;

	for (i = 0; i < 14; i++)
		UV_Guess[i] = 0.0;

	if (Initial_Guess_Scheme == 1) //Epipolar line
	{
		;
	}
	else //Just correlation
	{
		int j, k, m, n, ii, jj, II_0, JJ_0, iii, jjj;
		double ratio = 0.2;
		double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G, C_zncc, C_znssd_min, C_znssd_max;
		int hsubset, m_IG_subset[2];
		int length = width*height;

		m_IG_subset[0] = IG_subset[0] / 2;
		m_IG_subset[1] = IG_subset[1] / 2;

		double *C_znssd = new double[length];
		char *TT = new char[length];
		double *T = new double[2 * (2 * m_IG_subset[1] + 1)*(2 * m_IG_subset[1] + 1)];

		C_znssd_min = 1e12;
		C_znssd_max = -1e12;

		for (n = 0; n < length; n++)
		{
			*(TT + n) = (char)0;
			*(C_znssd + n) = 1e2;
		}

		for (k = 0; k < 2; k++)
		{
			hsubset = m_IG_subset[k];
			for (j = hsubset; j < height - hsubset; j++)
			{
				for (i = hsubset; i < width - hsubset; i++)
				{
					if (*(TT + j*width + i) == (char)1)
						continue;

					m = -1;
					t_f = 0.0;
					t_g = 0.0;
					for (jjj = -hsubset; jjj <= hsubset; jjj++)
					{
						for (iii = -hsubset; iii <= hsubset; iii++)
						{
							jj = Start_Point.y + jjj;
							ii = Start_Point.x + iii;
							JJ_0 = j + jjj;
							II_0 = i + iii;

							m_F = *(lpImageData + jj*width + ii);
							m_G = *(lpImageData + length + JJ_0*width + II_0);

							m++;
							*(T + 2 * m + 0) = m_F;
							*(T + 2 * m + 1) = m_G;
							t_f += m_F;
							t_g += m_G;
						}
					}

					t_f = t_f / (m + 1);
					t_g = t_g / (m + 1);
					t_1 = 0.0;
					t_2 = 0.0;
					t_3 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = *(T + 2 * iii + 0) - t_f;
						t_5 = *(T + 2 * iii + 1) - t_g;
						t_1 += (t_4*t_5);
						t_2 += (t_4*t_4);
						t_3 += (t_5*t_5);
					}
					t_2 = sqrt(t_2*t_3);
					if (t_2 < 1e-10)		// Avoid being divided by 0.
						t_2 = 1e-10;

					C_zncc = t_1 / t_2;

					// Testing shows that C_zncc may not fall into (-1, 1) range, so need the following line.
					if (C_zncc > 1.0 || C_zncc < -1.0)
						C_zncc = 0.0;	// Use 0.0 instead of 1.0 or -1.0

					*(C_znssd + j*width + i) = 2.0*(1.0 - C_zncc);

					if (*(C_znssd + j*width + i) < C_znssd_min)
					{
						C_znssd_min = *(C_znssd + j*width + i);
						UV_Guess[0] = i - Start_Point.x;
						UV_Guess[1] = j - Start_Point.y;
					}

					if (*(C_znssd + j*width + i) > C_znssd_max)
						C_znssd_max = *(C_znssd + j*width + i);	// C_znssd_max should be close to 4.0, C_znssd_min should be close to 0.0
				}
			}

			if (k == 0)
			{
				for (n = 0; n < length; n++)
				{
					if (*(C_znssd + n) > (C_znssd_min + ratio*(C_znssd_max - C_znssd_min)))
						*(TT + n) = (char)1;
				}

				C_znssd_min = 1e12;
				C_znssd_max = -1e12;
			}
		}

		delete[]T;
		delete[]TT;
		delete[]C_znssd;
	}

	return;
}
void DIC_Initial_Guess_Refine(int x_n, int y_n, double *lpImageData, double *Znssd_reqd, bool *lpROI, double *p, int nchannels, int width1, int height1, int width2, int height2, int hsubset, int step, int DIC_Algo, double *direction)
{
	/// Refine initial guess of u and v of the starting point with integral-pixel accuracy using ZNSSD

	int d_u, d_v, u0, v0, U0, V0, alpha, alpha0;
	int m, ii, jj, kk, iii, jjj, II_0, JJ_0;
	int length1 = width1*height1, length2 = width2*height2;
	double t_1, t_2, t_3, t_4, t_5, t_F, t_G, mean_F, mean_G;
	double C_zncc, C_znssd, C_znssd_min;
	C_znssd_min = 1.0E12;

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	if (DIC_Algo <= 1) //Epipoloar constraint on the flow
	{
		alpha = 0;
		for (alpha0 = -3 * step; alpha0 <= 3 * step; alpha0++)
		{
			u0 = (int)(direction[0] * (p[0] + 0.5*alpha0) + 0.5);
			v0 = (int)(direction[1] * (p[0] + 0.5*alpha0) + 0.5);

			m = -1;
			mean_F = 0.0;
			mean_G = 0.0;
			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}
			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					ii = x_n + iii;
					jj = y_n + jjj;

					if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
						continue;

					if (lpROI[jj*width1 + ii] == false)
						continue;

					II_0 = ii + u0 + (int)(p[1] * iii + p[2] * jjj + 0.5);
					JJ_0 = jj + v0 + (int)(p[3] * iii + p[4] * jjj + 0.5);

					if (II_0<0 || II_0>(width2 - 1) || JJ_0<0 || JJ_0>(height2 - 1))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						t_F = lpImageData[jj*width1 + ii + kk*length1];
						t_G = lpImageData[nchannels*length1 + kk*length2 + JJ_0*width2 + II_0];

						if (printout)
						{
							fprintf(fp1, "%.2f ", t_F), fprintf(fp2, "%.2f ", t_G);
						}
						m++;
						Znssd_reqd[2 * m] = t_F;
						Znssd_reqd[2 * m + 1] = t_G;
						mean_F += t_F;
						mean_G += t_G;
					}
				}
				if (printout)
				{
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
				}
			}
			if (printout)
			{
				fclose(fp1); fclose(fp2);
			}
			if (m < 10)
				continue;

			mean_F /= (m + 1);
			mean_G /= (m + 1);
			t_1 = 0.0;
			t_2 = 0.0;
			t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[2 * iii] - mean_F;
				t_5 = Znssd_reqd[2 * iii + 1] - mean_G;
				t_1 += (t_4*t_5);
				t_2 += (t_4*t_4);
				t_3 += (t_5*t_5);
			}

			C_zncc = t_1 / sqrt(t_2*t_3);
			C_znssd = 2.0*(1.0 - C_zncc);

			if (C_znssd < C_znssd_min)
			{
				C_znssd_min = C_znssd;
				alpha = alpha0;
			}
		}

		p[0] = p[0] + 0.5*alpha;
	}
	else //Affine shape 
	{
		U0 = 0, V0 = 0;
		for (d_u = -step; d_u <= step; d_u++)
		{
			for (d_v = -step; d_v <= step; d_v++)
			{
				u0 = d_u + (int)(p[0] + 0.5);
				v0 = d_v + (int)(p[1] + 0.5);

				m = -1;
				mean_F = 0.0;
				mean_G = 0.0;
				if (printout)
				{
					fp1 = fopen("C:/temp/src.txt", "w+");
					fp2 = fopen("C:/temp/tar.txt", "w+");
				}
				for (jjj = -hsubset; jjj <= hsubset; jjj++)
				{
					for (iii = -hsubset; iii <= hsubset; iii++)
					{
						ii = x_n + iii, jj = y_n + jjj;

						if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
							continue;

						if (lpROI[jj*width1 + ii] == false)
							continue;

						II_0 = ii + u0 + (int)(p[2] * iii + p[3] * jjj);
						JJ_0 = jj + v0 + (int)(p[4] * iii + p[5] * jjj);

						if (II_0<0 || II_0>(width2 - 1) || JJ_0<0 || JJ_0>(height2 - 1))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							t_F = lpImageData[jj*width1 + ii + kk*length1];
							t_G = lpImageData[nchannels*length1 + kk*length2 + JJ_0*width2 + II_0];

							if (printout && kk == 0)
							{
								fprintf(fp1, "%.2f ", t_F), fprintf(fp2, "%.2f ", t_G);
							}
							m++;
							*(Znssd_reqd + 2 * m + 0) = t_F;
							*(Znssd_reqd + 2 * m + 1) = t_G;
							mean_F += t_F;
							mean_G += t_G;
						}
					}
					if (printout)
					{
						fprintf(fp1, "\n"), fprintf(fp2, "\n");
					}
				}
				if (printout)
				{
					fclose(fp1); fclose(fp2);
				}
				if (m < 10)
					continue;

				mean_F /= (m + 1);
				mean_G /= (m + 1);
				t_1 = 0.0;
				t_2 = 0.0;
				t_3 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = *(Znssd_reqd + 2 * iii + 0) - mean_F;
					t_5 = *(Znssd_reqd + 2 * iii + 1) - mean_G;
					t_1 += (t_4*t_5);
					t_2 += (t_4*t_4);
					t_3 += (t_5*t_5);
				}

				C_zncc = t_1 / sqrt(t_2*t_3);
				C_znssd = 2.0*(1.0 - C_zncc);

				if (C_znssd < C_znssd_min)
				{
					C_znssd_min = C_znssd;
					U0 = u0;
					V0 = v0;
				}
			}
		}

		p[0] = U0;
		p[1] = V0;
	};

	return;
}
double DIC_Compute(int UV_index_n, int UV_index, double *lpImageData, double *Para, double *lpUV, int *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int DIC_Algo, int hsubset, int step, int Iter_Max, int *iteration_check, double conv_crit_1, double conv_crit_2, int Interpolation_Algorithm, int Analysis_Speed, bool firsttime, double *direction, bool checkZNCC = false, double ZNNCThresh = 0.99)
{
	double DIC_Coeff, a, b;
	int i, j, ii, jj, kk, iii, jjj, iii2, jjj2, ij;
	int k, m, nn, nExtraParas;
	int length1 = width1*height1, length2 = width2*height2;
	int NN[] = { 3, 7, 4, 8, 6, 12 };
	double II, JJ, iii_n, jjj_n;
	double m_F, m_G, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g;
	double S[9];
	double p[8], ip[8], p_best[8];// U, V, Ux, Uy, Vx, Vy, (a) and b.
	double AA[144], BB[12], CC[12], gx, gy;

	nn = NN[DIC_Algo], nExtraParas = 2;
	for (i = 0; i < nn; i++)
		p[i] = lpUV[i], ip[i] = lpUV[i];

	int x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
	int x_n = lpUV_xy[2 * UV_index_n], y_n = lpUV_xy[2 * UV_index_n + 1];

	// The following two lines are needed for large rotation cases.
	if (DIC_Algo == 1)
		p[0] = 0.5*((p[0] * direction[0] + (p[1] * (x_n - x) + p[2] * (y_n - y))) / direction[0] + (p[0] * direction[1] + (p[3] * (x_n - x) + p[4] * (y_n - y))) / direction[1]);
	else if (DIC_Algo == 3)
		p[0] += (p[2] * (x_n - x) + p[3] * (y_n - y)), p[1] += (p[4] * (x_n - x) + p[5] * (y_n - y));

	// Refine initial guess of u and v of the starting point with integral-pixel accuracy using ZNSSD
	if (firsttime)
		DIC_Initial_Guess_Refine(x_n, y_n, lpImageData, Znssd_reqd, lpROI, p, nchannels, width1, height1, width2, height2, hsubset, step, DIC_Algo, direction);

	bool printout = false; FILE *fp1 = 0, *fp2 = 0;
	int piis, pixel_increment_in_subset[] = { 1, 2, 2, 3 };
	double DIC_Coeff_min = 9e9;
	/// Iteration: Begin
	bool Break_Flag = false;
	for (k = 0; k < Iter_Max; k++)
	{
		m = -1;
		t_1 = 0.0, t_2 = 0.0;
		for (iii = 0; iii < nn*nn; iii++)
			AA[iii] = 0.0;
		for (iii = 0; iii < nn; iii++)
			BB[iii] = 0.0;

		a = p[nn - 2], b = p[nn - 1];

		if (printout)
			fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

		piis = pixel_increment_in_subset[Analysis_Speed];	// Depending on algorithms, Analysis_Speed may be changed during the iteration loop.
		for (jjj = -hsubset; jjj <= hsubset; jjj += piis)
		{
			for (iii = -hsubset; iii <= hsubset; iii += piis)
			{
				ii = x_n + iii, jj = y_n + jjj;
				if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1) || lpROI[jj*width1 + ii] == false)
					continue;

				if (DIC_Algo == 0)
					II = ii + p[0] * direction[0], JJ = jj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = ii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = jj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				if (DIC_Algo == 2)
					II = ii + p[0], JJ = jj + p[1];
				else if (DIC_Algo == 3)
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1] + p[4] * iii + p[5] * jjj;
				if (DIC_Algo == 4)
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1] + p[4] * iii + p[5] * jjj;
				else if (DIC_Algo == 5)
				{
					iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
					II = ii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
					JJ = jj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
				}

				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);

					m_F = lpImageData[ii + jj*width1 + kk*length1];
					m_G = S[0], gx = S[1], gy = S[2];
					m++;

					if (DIC_Algo < 4)
					{
						t_3 = a*m_G + b - m_F, t_4 = a;
						t_1 += t_3*t_3, t_2 += m_F*m_F;

						t_5 = t_4*gx, t_6 = t_4*gy;
						if (DIC_Algo == 0)
							CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = m_G, CC[2] = 1.0;
						else if (DIC_Algo == 1)
							CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj, CC[5] = m_G, CC[6] = 1.0;
						else if (DIC_Algo == 2)
							CC[0] = t_5, CC[1] = t_6, CC[2] = m_G, CC[3] = 1.0;
						else if (DIC_Algo == 3)
							CC[0] = t_5, CC[1] = t_6, CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj, CC[6] = m_G, CC[7] = 1.0;

						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = 0; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j];
						}
					}
					else
					{
						Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G, Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
						Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
						t_1 += m_F, t_2 += m_G;
					}

					if (printout)
						fprintf(fp1, "%.2f ", m_F), fprintf(fp2, "%.2f ", m_G);
				}
			}
			if (printout)
				fprintf(fp1, "\n"), fprintf(fp2, "\n");
		}
		if (printout)
			fclose(fp1), fclose(fp2);

		if (DIC_Algo < 4)
			DIC_Coeff = t_1 / t_2;
		else
		{
			if (k == 0)
			{
				t_f = t_1 / (m + 1); t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
					t_4 = Znssd_reqd[6 * iii + 0] - t_f, t_1 += t_4*t_4;
				t_ff = sqrt(t_1);
			}
			t_g = t_2 / (m + 1), t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
				t_5 = Znssd_reqd[6 * iii + 1] - t_g, t_2 += t_5*t_5;
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
				t_4 = Znssd_reqd[6 * iii + 0] - t_f, t_5 = Znssd_reqd[6 * iii + 1] - t_g, t_6 = t_5 / t_2 - t_4 / t_ff, t_3 = t_6 / t_2;
				iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
				CC[0] = gx, CC[1] = gy, CC[2] = gx*iii_n, CC[3] = gx*jjj_n, CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
				if (DIC_Algo == 5)
					CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n, CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}
		}

		if (!IsNumber(DIC_Coeff))
			return 9e9;
		if (!IsFiniteNumber(DIC_Coeff))
			return 9e9;

		QR_Solution_Double(AA, BB, nn, nn);
		for (iii = 0; iii < nn; iii++)
			p[iii] -= BB[iii];

		if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
		{
			DIC_Coeff_min = DIC_Coeff;
			for (iii = 0; iii < nn; iii++)
				p_best[iii] = p[iii];
			if (!IsNumber(p[0]) || !IsNumber(p[1]))
				return 9e9;
		}

		if (DIC_Algo <= 1)
		{
			if (abs((p[0] - ip[0])*direction[0]) > hsubset)
				return 9e9;

			if (fabs(BB[0]) < conv_crit_1)
			{
				for (iii = 1; iii < nn - nExtraParas; iii++)
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				if (iii == nn - nExtraParas)
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
			}
		}
		else if (DIC_Algo <= 3)
		{
			if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
				return 9e9;
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn - nExtraParas; iii++)
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				if (iii == nn - nExtraParas)
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
			}
		}
		else
		{
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}
		}

		if (Break_Flag)
			break;
	}
	if (k < 1)
		k = 1;
	iteration_check[k - 1]++;
	/// Iteration: End

	if (checkZNCC && DIC_Algo < 4)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, ZNCC, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp2 = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				ii = x_n + iii, jj = y_n + jjj;

				if (DIC_Algo == 0)
					II = ii + p[0] * direction[0], JJ = jj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = ii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = jj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				if (DIC_Algo == 2)
					II = ii + p[0], JJ = jj + p[1];
				else if (DIC_Algo == 3)
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);
					if (printout)
						fprintf(fp2, "%.4f ", S[0]);

					Znssd_reqd[2 * m] = lpImageData[ii + jj*width1 + kk*length1], Znssd_reqd[2 * m + 1] = S[0];
					t_f += Znssd_reqd[2 * m], t_g += Znssd_reqd[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp2, "\n");
		}
		if (printout)
			fclose(fp2);

		t_f = t_f / m, t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = Znssd_reqd[2 * i] - t_f, t_5 = Znssd_reqd[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) < ZNNCThresh)
			DIC_Coeff_min = 1.0;
	}

	if (DIC_Algo <= 1)
	{
		if (abs((p[0] - ip[0])*direction[0]) > hsubset)
			return 9e9;
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i] = p_best[i];
			return DIC_Coeff_min;
		}
	}
	else
	{
		if (abs(p_best[0] - ip[0]) > hsubset || abs(p_best[1] - ip[1]) > hsubset || p_best[0] != p_best[0] || p_best[1] != p_best[1])
			return 9e9;
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i] = p_best[i];
			return DIC_Coeff_min;
		}
	}

}
double DIC_Calculation(int UV_index_n, int UV_index, double *lpImageData, double *Para, double *lpUV, int *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int DIC_Algo, int hsubset, int step, double PSSDab_thresh, double ZNCCthresh, double ssigThresh, int Iter_Max, int *iteration_check, double conv_crit_1, double conv_crit_2, int Interpolation_Algorithm, int Analysis_Speed, bool firsttime, double *direction, double *FlowU = 0, double *FlowV = 0, bool InitFlow = 0, bool checkZNCC = false)
{
	int i;
	int NN[] = { 3, 7, 4, 8, 6, 12 }, nn = NN[DIC_Algo];

	double shapepara[8];
	for (i = 0; i < nn; i++)
		shapepara[i] = lpUV[i*UV_length + UV_index];

	double ssig = ComputeSSIG(Para, lpUV_xy[2 * UV_index_n], lpUV_xy[2 * UV_index_n + 1], hsubset, width1, height1, nchannels, Interpolation_Algorithm);
	if (ssig < ssigThresh)
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = 0.0;
		return 9e9;
	}

	double DIC_Coeff = DIC_Compute(UV_index_n, UV_index, lpImageData, Para + width1*height1*nchannels, shapepara, lpUV_xy, Znssd_reqd, lpROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, Interpolation_Algorithm, Analysis_Speed, firsttime, direction, checkZNCC, ZNCCthresh);
	if (DIC_Coeff < PSSDab_thresh)
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = shapepara[i];
		return DIC_Coeff;
	}
	else if (InitFlow)
	{
		for (i = 0; i < nn - 2; i++)
			shapepara[i] = 0.0;
		shapepara[nn - 2] = 1.0, shapepara[nn - 1] = 0.0;

		int x = lpUV_xy[2 * UV_index_n], y = lpUV_xy[2 * UV_index_n + 1];
		if (DIC_Algo == 0)
			shapepara[0] = 0.5*(FlowU[x + y*width1] / direction[0] + FlowV[x + y*width1] / direction[1]);
		else
			shapepara[0] = FlowU[x + y*width1], shapepara[1] = FlowV[x + y*width1];

		DIC_Coeff = DIC_Compute(UV_index_n, UV_index, lpImageData, Para, shapepara, lpUV_xy, Znssd_reqd, lpROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, Interpolation_Algorithm, Analysis_Speed, firsttime, direction, checkZNCC, ZNCCthresh);
		if (DIC_Coeff < PSSDab_thresh)
		{
			for (i = 0; i < nn; i++)
				lpUV[i*UV_length + UV_index_n] = shapepara[i];
			return DIC_Coeff;
		}
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i*UV_length + UV_index_n] = 0.0;
			return 9e9;
		}
	}
	else
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = 0.0;
		return 9e9;
	}
}

int DenseGreedyMatching(char *Img1, char *Img2, Point2d *displacement, vector<Point2d> &SSrcPts, vector<Point2d> &SDstPts, bool *lpROI_calculated, bool *tROI, LKParameters LKArg,
	int nchannels, int width1, int height1, int width2, int height2, double Scale, float *WarpingParas, int foundPrecomputedPoints, double *Epipole, double *Pmat, double *K, double *distortion, double triThresh)
{
	//DIC_Algo = -2: similarlity transform
	//DIC_Algo = -1: epip similarlity transform: not yet supported
	//DIC_Algo = 0: epip translation+photometric
	//DIC_Algo = 1: epip affine+photometric
	//DIC_Algo = 2: translation+photometric
	//DIC_Algo = 3: affine+photometric
	//DIC_Algo = 4: epi irreglar + photometric
	//DIC_Algo = 5: epi quadratic + photometric
	//DIC_Algo = 6: irregular + photometric
	//DIC_Algo = 7:  quadratic + photmetric
	//DIC_Algo = 8: ZNCC affine. Only support gray scale image
	//DIC_Algo = 9:  ZNCC quadratic. Only support gray scale image
	int ii, kk, cp;
	bool debug = false, passed;
	int length1 = width1*height1, length2 = width2*height2;
	double *lpResult_UV = new double[length1 + length2];
	for (ii = 0; ii < length1 + length2; ii++)
		lpResult_UV[ii] = 0.0;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, step = LKArg.step, Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, InterpAlgo = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double Gsigma = LKArg.Gsigma, PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold;

	int GeoVerify = 0;
	if (Pmat != NULL)
		GeoVerify = 1;

	int m, M, x, y;
	double vr_temp[] = { 0.0, 0.65, 0.8, 0.95 };
	double validity_ratio = vr_temp[Incomplete_Subset_Handling];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	double  ImgPt[3], epipline[3], direction[2];

	if (DIC_Algo == 0)
		conv_crit_1 /= 100.0, conv_crit_2 /= 100.0;
	else if (DIC_Algo == 1)
		conv_crit_1 /= 1000.0;

	int total_valid_points = 0, total_calc_points = 0;
	for (ii = 0; ii < length1; ii++)
		if (tROI[ii]) // 1 - Valid, 0 - Other
			total_valid_points++;
	if (total_valid_points == 0)
		return 1;

	int NN[] = { 3, 7, 4, 8, 6, 12 }, nParas = NN[DIC_Algo];
	int UV_length = length1;	// The actual value (i.e., total_calc_points, to be determined later) should be smaller.
	double *lpUV = new double[nParas*UV_length];	// U, V, Ux, Uy, Vx, Vy, Uxx, Uyy, Uxy, Vxx, Vyy, Vxy, (a) and b. or alpha, (a), and b
	int *lpUV_xy = new int[2 * UV_length];	// Coordinates of the points corresponding to lpUV

	int TimgS = (2 * hsubset + 1)*(2 * hsubset + 1);
	double *Znssd_reqd = 0;
	if (DIC_Algo < 4)
		Znssd_reqd = new double[2 * TimgS*nchannels];
	else if (DIC_Algo == 4)
		Znssd_reqd = new double[6 * TimgS*nchannels];
	else if (DIC_Algo == 5)
		Znssd_reqd = new double[12 * TimgS*nchannels];
	double *Coeff = new double[UV_length];
	int *Tindex = new int[UV_length];

	// Prepare image data
	double *Para = new double[(length1 + length2)*nchannels];
	double *lpImageData = new double[nchannels*(length1 + length2)];
	if (Gsigma > 0.0)
		for (kk = 0; kk < nchannels; kk++)
		{
			Gaussian_smooth(Img1 + kk*length1, lpImageData + kk*length1, height1, width1, 255.0, Gsigma);
			Gaussian_smooth(Img2 + kk*length2, lpImageData + kk*length2 + nchannels*length1, height2, width2, 255.0, Gsigma);
		}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			for (ii = 0; ii < length1; ii++)
				lpImageData[ii + kk*length1] = (double)((int)((unsigned char)(Img1[ii + kk*length1])));
			for (ii = 0; ii < length2; ii++)
				lpImageData[ii + kk*length2 + nchannels*length1] = (double)((int)((unsigned char)(Img2[ii + kk*length2])));
		}
	}
	for (kk = 0; kk < nchannels; kk++)
	{
		Generate_Para_Spline(lpImageData + kk*length1, Para + kk*length1, width1, height1, InterpAlgo);
		Generate_Para_Spline(lpImageData + kk*length2 + nchannels*length1, Para + kk*length2 + nchannels*length1, width2, height2, InterpAlgo);
	}

	int *iteration_check = new int[Iter_Max];
	for (m = 0; m < Iter_Max; m++)
		iteration_check[m] = 0;

	int nSeedPoints = (int)SSrcPts.size();
	int *PointPerSeed = new int[nSeedPoints];
	for (ii = 0; ii < nSeedPoints; ii++)
		PointPerSeed[ii] = 0;

	double start = omp_get_wtime();
	int percent = 5, increment = 5;

	int DeviceMask[2];
	Point2d CorresPoints[2], tCorresPoints[2], ttCorresPoints[2];
	double tK[18], tdistortion[14], tP[24], A[12], B[4];

	bool firstpoint = true;
	int tid1, tid2, UV_index = 0, UV_index_n = 0;
	double UV_Guess[8], coeff;
	/*if (foundPrecomputedPoints == 1)
	{
	printLOG("Expanding from precomputed points...\n");
	for (kk = 0; kk < length1; kk++)
	{
	firstpoint = true;
	cp = 0, M = 0;

	x = kk%width1, y = kk / width1;
	if (lpROI_calculated[y*width1 + x] || !tROI[y*width1 + x])
	continue;
	else
	{
	UV_index = UV_index_n;
	lpUV_xy[2 * UV_index] = x;
	lpUV_xy[2 * UV_index + 1] = y;

	if (!IsLocalWarpAvail(WarpingParas, iWp, x, y, startF, xx, yy, rr, width1, height1, 2))
	continue;

	if (DIC_Algo <= 1)
	{
	ImgPt[0] = lpUV_xy[2 * UV_index], ImgPt[1] = lpUV_xy[2 * UV_index + 1], ImgPt[2] = 1.0;
	cross_product(ImgPt, Epipole, epipline);
	direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);

	UV_Guess[0] = 0.5*((xx - x) / direction[0] + (yy - y) / direction[1]);
	if (DIC_Algo == 1)
	UV_Guess[1] = iWp[0], UV_Guess[2] = iWp[1], UV_Guess[3] = iWp[2], UV_Guess[4] = iWp[3], UV_Guess[5] = 1.0, UV_Guess[6] = 0.0;
	}
	else
	{
	UV_Guess[0] = xx - x, UV_Guess[1] = yy - y;
	UV_Guess[2] = iWp[0], UV_Guess[3] = iWp[1], UV_Guess[4] = iWp[2], UV_Guess[5] = iWp[3], UV_Guess[6] = 1.0, UV_Guess[7] = 0.0;
	}

	for (m = 0; m < nParas; m++)
	lpUV[m*UV_length + UV_index] = UV_Guess[m];

	coeff = DIC_Calculation(UV_index_n, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
	if (WarpingParas != NULL)
	for (m = 0; m < 6; m++)
	WarpingParas[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width1 + m*UV_length] = (float)lpUV[UV_index + m*UV_length];

	if (coeff < PSSDab_thresh)
	{
	if (distortion != NULL)
	{
	passed = false;
	CorresPoints[0].x = lpUV[UV_index] + x, CorresPoints[0].y = lpUV[UV_index + UV_length] + y, CorresPoints[1].x = x, CorresPoints[1].y = y;
	MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
	if (passed)
	Coeff[M] = coeff, Tindex[M] = UV_index;
	else
	M--;
	}
	else
	Coeff[M] = coeff, Tindex[M] = UV_index;
	}
	else
	M--;
	x = lpUV_xy[2 * UV_index];
	y = lpUV_xy[2 * UV_index + 1];
	lpROI_calculated[y*width1 + x] = true;
	firstpoint = false;
	}

	while (M >= 0)
	{
	if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) >= 0)
	{
	cout << total_calc_points + cp << " of good points" << endl;
	double elapsed = omp_get_wtime() - start;
	printLOG("%.2f%% . TE: %.2f TR: %.2f\n", 100.0 * (UV_index_n + 1)*step*step / total_valid_points, elapsed, elapsed / (percent + increment)*(100.0 - percent));
	percent += increment;
	}

	UV_index = Tindex[M];
	x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
	M--; // Remove from the queque


	if ((y + step) < height1 && tROI[(y + step)*width1 + x] && !lpROI_calculated[(y + step)*width1 + x])
	{
	if (DIC_CheckPointValidity(tROI, x, y + step, width1, height1, hsubset, validity_ratio))
	{
	lpUV_xy[2 * (UV_index_n + 1)] = x;
	lpUV_xy[2 * (UV_index_n + 1) + 1] = y + step;

	if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
	int a = 0;

	if (DIC_Algo <= 1)
	{
	ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
	cross_product(ImgPt, Epipole, epipline);
	direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	}

	coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
	if (coeff < PSSDab_thresh)
	{
	if (distortion != NULL)
	{
	passed = false;
	CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
	CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
	MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
	}
	else
	passed = true;
	if (passed)
	{
	cp++, M++, UV_index_n++;
	Coeff[M] = coeff;
	Tindex[M] = UV_index_n;
	DIC_AddtoQueue(Coeff, Tindex, M);
	if (WarpingParas != NULL)
	{
	for (m = 0; m < 6; m++)
	WarpingParas[x + (y + step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
	}
	}
	}
	}
	lpROI_calculated[(y + step)*width1 + x] = true;
	}
	if ((y - step) >= 0 && tROI[(y - step)*width1 + x] && !lpROI_calculated[(y - step)*width1 + x])
	{
	if (DIC_CheckPointValidity(tROI, x, y - step, width1, height1, hsubset, validity_ratio))
	{
	lpUV_xy[2 * (UV_index_n + 1)] = x;
	lpUV_xy[2 * (UV_index_n + 1) + 1] = y - step;

	if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
	int a = 0;
	if (DIC_Algo <= 1)
	{
	ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
	cross_product(ImgPt, Epipole, epipline);
	direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	}

	coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
	if (coeff < PSSDab_thresh)
	{
	if (distortion != NULL)
	{
	passed = false;
	CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
	CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
	MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
	}
	else
	passed = true;
	if (passed)
	{
	cp++, M++, UV_index_n++;
	Coeff[M] = coeff;
	Tindex[M] = UV_index_n;
	DIC_AddtoQueue(Coeff, Tindex, M);
	if (WarpingParas != NULL)
	{
	for (m = 0; m < 6; m++)
	WarpingParas[x + (y - step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
	}
	}
	}
	}
	lpROI_calculated[(y - step)*width1 + x] = true;
	}
	if ((x - step) >= 0 && tROI[y*width1 + x - step] && !lpROI_calculated[y*width1 + x - step])
	{
	if (DIC_CheckPointValidity(tROI, x - step, y, width1, height1, hsubset, validity_ratio))
	{
	lpUV_xy[2 * (UV_index_n + 1)] = x - step;
	lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
	if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
	int a = 0;
	if (DIC_Algo <= 1)
	{
	ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
	cross_product(ImgPt, Epipole, epipline);
	direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	}

	coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
	if (coeff < PSSDab_thresh)
	{
	if (distortion != NULL)
	{
	passed = false;
	CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
	CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
	MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
	}
	else
	passed = true;
	if (passed)
	{
	cp++, M++, UV_index_n++;
	Coeff[M] = coeff;
	Tindex[M] = UV_index_n;
	DIC_AddtoQueue(Coeff, Tindex, M);
	if (WarpingParas != NULL)
	{
	for (m = 0; m < 6; m++)
	WarpingParas[x - step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
	}
	}
	}
	}
	lpROI_calculated[y*width1 + x - step] = true;
	}
	if ((x + step) < width1 && tROI[y*width1 + x + step] && !lpROI_calculated[y*width1 + x + step])
	{
	if (DIC_CheckPointValidity(tROI, x + step, y, width1, height1, hsubset, validity_ratio))
	{
	lpUV_xy[2 * (UV_index_n + 1)] = x + step;
	lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
	if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
	int a = 0;
	if (DIC_Algo <= 1)
	{
	ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
	cross_product(ImgPt, Epipole, epipline);
	direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	}

	coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
	if (coeff < PSSDab_thresh)
	{
	if (distortion != NULL)
	{
	passed = false;
	CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
	CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
	MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
	}
	else
	passed = true;
	if (passed)
	{
	cp++, M++, UV_index_n++;
	Coeff[M] = coeff;
	Tindex[M] = UV_index_n;
	DIC_AddtoQueue(Coeff, Tindex, M);
	if (WarpingParas != NULL)
	{
	for (m = 0; m < 6; m++)
	WarpingParas[x + step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
	}
	}
	}
	}
	lpROI_calculated[y*width1 + x + step] = true;
	}
	}

	if (cp > 0)
	UV_index_n++;
	total_calc_points += cp;
	}
	printLOG("...%d points growed\n", total_calc_points);
	}*/

	printLOG("Expanding from seed points...\n");
	int total_calc_points1 = total_calc_points;
	for (kk = 0; kk < nSeedPoints; kk++)
	{
		firstpoint = true;
		cp = 0, M = 0;

		x = (int)round(SSrcPts[kk].x), y = (int)round(SSrcPts[kk].y);
		if (lpROI_calculated[y*width1 + x] || !tROI[y*width1 + x])
			continue;
		else
		{
			UV_index = UV_index_n;
			lpUV_xy[2 * UV_index] = (int)round(SSrcPts[kk].x), lpUV_xy[2 * UV_index + 1] = (int)round(SSrcPts[kk].y);

			if (DIC_Algo <= 1)
			{
				ImgPt[0] = lpUV_xy[2 * UV_index], ImgPt[1] = lpUV_xy[2 * UV_index + 1], ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole, epipline);
				double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
				UV_Guess[0] = 0.5*((SDstPts[kk].x - SSrcPts[kk].x) / direction[0] + (SDstPts[kk].y - SSrcPts[kk].y) / direction[1]);

				if (DIC_Algo == 1)
					UV_Guess[1] = Scale - 1.0, UV_Guess[2] = 0.0, UV_Guess[3] = 0.0, UV_Guess[4] = Scale - 1.0, UV_Guess[5] = 1.0, UV_Guess[6] = 0.0;
			}
			else
			{
				UV_Guess[0] = SDstPts[kk].x - SSrcPts[kk].x, UV_Guess[1] = SDstPts[kk].y - SSrcPts[kk].y;
				UV_Guess[2] = Scale - 1.0, UV_Guess[3] = 0.0, UV_Guess[4] = 0.0, UV_Guess[5] = Scale - 1.0, UV_Guess[6] = 1.0, UV_Guess[7] = 0.0;
			}

			for (m = 0; m < nParas; m++)
				lpUV[m*UV_length + UV_index] = UV_Guess[m];

			coeff = DIC_Calculation(UV_index_n, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold + 0.035, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
			for (m = 0; m < 6; m++)
				WarpingParas[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width1 + m*UV_length] = (float)lpUV[UV_index + m*UV_length];

			if (coeff < PSSDab_thresh)
			{
				if (GeoVerify == 1)
				{
					passed = false;
					CorresPoints[0].x = lpUV[UV_index] + x, CorresPoints[0].y = lpUV[UV_index + UV_length] + y, CorresPoints[1].x = x, CorresPoints[1].y = y;
					MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
					if (passed)
						Coeff[M] = coeff, Tindex[M] = UV_index;
					else
						M--;
				}
				else
					Coeff[M] = coeff, Tindex[M] = UV_index;
			}
			else
				M--;
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			lpROI_calculated[y*width1 + x] = true;
			firstpoint = false;
		}

		while (M >= 0)
		{
			if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) >= 0)
			{
				double elapsed = omp_get_wtime() - start;
				printLOG("%d good points\n", total_calc_points + cp);
				printLOG("%.2f%% . TE: %.2f TR: %.2f\n", 100.0 * (UV_index_n + 1)*step*step / total_valid_points, elapsed, elapsed / (percent + increment)*(100.0 - percent));
				percent += increment;
			}

			UV_index = Tindex[M];
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			M--; // Remove from the queque

			if ((y + step) < height1 && tROI[(y + step)*width1 + x] && !lpROI_calculated[(y + step)*width1 + x])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x, y + step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x, lpUV_xy[tid2 + 1] = y + step;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x + (y + step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[(y + step)*width1 + x] = true;
			}
			if ((y - step) >= 0 && tROI[(y - step)*width1 + x] && !lpROI_calculated[(y - step)*width1 + x])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x, y - step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x, lpUV_xy[tid2 + 1] = y - step;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x + (y - step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[(y - step)*width1 + x] = true;
			}
			if ((x - step) >= 0 && tROI[y*width1 + x - step] && !lpROI_calculated[y*width1 + x - step])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x - step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x - step, lpUV_xy[tid2 + 1] = y;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x - step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[y*width1 + x - step] = true;
			}
			if ((x + step) < width1 && tROI[y*width1 + x + step] && !lpROI_calculated[y*width1 + x + step])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x + step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x + step, lpUV_xy[tid2 + 1] = y;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x + step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[y*width1 + x + step] = true;
			}
		}

		if (cp > 0)
			UV_index_n++;
		PointPerSeed[kk] = cp;
		total_calc_points += cp;
	}
	printLOG("...%d points growed\n", total_calc_points - total_calc_points1);
	//// DIC calculation: End

	for (ii = 0; ii < total_calc_points; ii++)
	{
		if (lpUV[ii] != lpUV[ii])
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = 0.0;
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = 0.0;
			continue;
		}
		if (lpUV[UV_length + ii] != lpUV[UV_length + ii])
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = 0.0;
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = 0.0;
			continue;
		}

		if (DIC_Algo <= 1)
		{
			ImgPt[0] = lpUV_xy[2 * ii], ImgPt[1] = lpUV_xy[2 * ii + 1], ImgPt[2] = 1.0;
			cross_product(ImgPt, Epipole, epipline);
			double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
			direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;

			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = lpUV[ii] * direction[0];
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = lpUV[ii] * direction[1];
		}
		else
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = lpUV[ii];
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = lpUV[UV_length + ii];
		}
	}

	delete[]Tindex, delete[]Coeff, delete[]Znssd_reqd;
	delete[]Para, delete[]lpUV_xy, delete[]lpResult_UV, delete[]lpUV;
	delete[]PointPerSeed, delete[]lpImageData, delete[]iteration_check;

	return 0;
}
int SemiDenseGreedyMatching(char *Img1, char *Img2, vector<Point2d> &SDSrcPts, vector<Point2d> &SDDstPts, vector<Point2d> &SSrcPts, vector<Point2d> &SDstPts, bool *lpROI_calculated, bool *tROI, LKParameters LKArg,
	int nchannels, int width1, int height1, int width2, int height2, double Scale, float *WarpingParas, int foundPrecomputedPoints, double *Epipole, double *Pmat, double *K, double *distortion, double triThresh)
{
	//DIC_Algo = -2: similarlity transform
	//DIC_Algo = -1: epip similarlity transform: not yet supported
	//DIC_Algo = 0: epip translation+photometric
	//DIC_Algo = 1: epip affine+photometric
	//DIC_Algo = 2: translation+photometric
	//DIC_Algo = 3: affine+photometric
	//DIC_Algo = 4: epi irreglar + photometric
	//DIC_Algo = 5: epi quadratic + photometric
	//DIC_Algo = 6: irregular + photometric
	//DIC_Algo = 7:  quadratic + photmetric
	//DIC_Algo = 8: ZNCC affine. Only support gray scale image
	//DIC_Algo = 9:  ZNCC quadratic. Only support gray scale image
	int ii, kk, cp;
	bool debug = false, passed;
	int length1 = width1*height1, length2 = width2*height2;
	double *lpResult_UV = new double[length1 + length2];
	for (ii = 0; ii < length1 + length2; ii++)
		lpResult_UV[ii] = 0.0;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, step = LKArg.step, Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, InterpAlgo = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double Gsigma = LKArg.Gsigma, PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold;

	int GeoVerify = 0;
	if (Pmat != NULL)
		GeoVerify = 1;

	int m, M, x, y;
	double vr_temp[] = { 0.0, 0.65, 0.8, 0.95 };
	double validity_ratio = vr_temp[Incomplete_Subset_Handling];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	double  ImgPt[3], epipline[3], direction[2];

	if (DIC_Algo == 0)
		conv_crit_1 /= 100.0, conv_crit_2 /= 100.0;
	else if (DIC_Algo == 1)
		conv_crit_1 /= 1000.0;

	int total_valid_points = 0, total_calc_points = 0;
	for (ii = 0; ii < length1; ii++)
		if (tROI[ii]) // 1 - Valid, 0 - Other
			total_valid_points++;
	if (total_valid_points == 0)
		return 1;

	int NN[] = { 3, 7, 4, 8, 6, 12 }, nParas = NN[DIC_Algo];
	int UV_length = length1;	// The actual value (i.e., total_calc_points, to be determined later) should be smaller.
	double *lpUV = new double[nParas*UV_length];	// U, V, Ux, Uy, Vx, Vy, Uxx, Uyy, Uxy, Vxx, Vyy, Vxy, (a) and b. or alpha, (a), and b
	int *lpUV_xy = new int[2 * UV_length];	// Coordinates of the points corresponding to lpUV
	Point2d *displacement = new Point2d[UV_length];

	int TimgS = (2 * hsubset + 1)*(2 * hsubset + 1);
	double *Znssd_reqd = 0;
	if (DIC_Algo < 4)
		Znssd_reqd = new double[2 * TimgS*nchannels];
	else if (DIC_Algo == 4)
		Znssd_reqd = new double[6 * TimgS*nchannels];
	else if (DIC_Algo == 5)
		Znssd_reqd = new double[12 * TimgS*nchannels];
	double *Coeff = new double[UV_length];
	int *Tindex = new int[UV_length];

	// Prepare image data
	double *Para = new double[(length1 + length2)*nchannels];
	double *lpImageData = new double[nchannels*(length1 + length2)];
	if (Gsigma > 0.0)
		for (kk = 0; kk < nchannels; kk++)
		{
			Gaussian_smooth(Img1 + kk*length1, lpImageData + kk*length1, height1, width1, 255.0, Gsigma);
			Gaussian_smooth(Img2 + kk*length2, lpImageData + kk*length2 + nchannels*length1, height2, width2, 255.0, Gsigma);
		}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			for (ii = 0; ii < length1; ii++)
				lpImageData[ii + kk*length1] = (double)((int)((unsigned char)(Img1[ii + kk*length1])));
			for (ii = 0; ii < length2; ii++)
				lpImageData[ii + kk*length2 + nchannels*length1] = (double)((int)((unsigned char)(Img2[ii + kk*length2])));
		}
	}
	for (kk = 0; kk < nchannels; kk++)
	{
		Generate_Para_Spline(lpImageData + kk*length1, Para + kk*length1, width1, height1, InterpAlgo);
		Generate_Para_Spline(lpImageData + kk*length2 + nchannels*length1, Para + kk*length2 + nchannels*length1, width2, height2, InterpAlgo);
	}

	int *iteration_check = new int[Iter_Max];
	for (m = 0; m < Iter_Max; m++)
		iteration_check[m] = 0;

	int nSeedPoints = (int)SSrcPts.size();
	int *PointPerSeed = new int[nSeedPoints];
	for (ii = 0; ii < nSeedPoints; ii++)
		PointPerSeed[ii] = 0;

	double start = omp_get_wtime();
	int percent = 5, increment = 5;

	int DeviceMask[2];
	Point2d CorresPoints[2], tCorresPoints[2], ttCorresPoints[2];
	double tK[18], tdistortion[14], tP[24], A[12], B[4];

	bool firstpoint = true;
	int tid1, tid2, UV_index = 0, UV_index_n = 0;
	double UV_Guess[8], coeff;

	printLOG("Expanding from seed points...\n");
	int total_calc_points1 = total_calc_points;
	for (kk = 0; kk < nSeedPoints; kk++)
	{
		firstpoint = true;
		cp = 0, M = 0;

		x = (int)round(SSrcPts[kk].x), y = (int)round(SSrcPts[kk].y);
		if (lpROI_calculated[y*width1 + x] || !tROI[y*width1 + x])
			continue;
		else
		{
			UV_index = UV_index_n;
			lpUV_xy[2 * UV_index] = (int)round(SSrcPts[kk].x), lpUV_xy[2 * UV_index + 1] = (int)round(SSrcPts[kk].y);

			if (DIC_Algo <= 1)
			{
				ImgPt[0] = lpUV_xy[2 * UV_index], ImgPt[1] = lpUV_xy[2 * UV_index + 1], ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole, epipline);
				double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
				UV_Guess[0] = 0.5*((SDstPts[kk].x - SSrcPts[kk].x) / direction[0] + (SDstPts[kk].y - SSrcPts[kk].y) / direction[1]);

				if (DIC_Algo == 1)
					UV_Guess[1] = Scale - 1.0, UV_Guess[2] = 0.0, UV_Guess[3] = 0.0, UV_Guess[4] = Scale - 1.0, UV_Guess[5] = 1.0, UV_Guess[6] = 0.0;
			}
			else
			{
				UV_Guess[0] = SDstPts[kk].x - SSrcPts[kk].x, UV_Guess[1] = SDstPts[kk].y - SSrcPts[kk].y;
				UV_Guess[2] = Scale - 1.0, UV_Guess[3] = 0.0, UV_Guess[4] = 0.0, UV_Guess[5] = Scale - 1.0, UV_Guess[6] = 1.0, UV_Guess[7] = 0.0;
			}

			for (m = 0; m < nParas; m++)
				lpUV[m*UV_length + UV_index] = UV_Guess[m];

			coeff = DIC_Calculation(UV_index_n, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold + 0.035, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
			for (m = 0; m < 6; m++)
				WarpingParas[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width1 + m*UV_length] = (float)lpUV[UV_index + m*UV_length];

			if (coeff < PSSDab_thresh)
			{
				if (GeoVerify == 1)
				{
					passed = false;
					CorresPoints[0].x = lpUV[UV_index] + x, CorresPoints[0].y = lpUV[UV_index + UV_length] + y, CorresPoints[1].x = x, CorresPoints[1].y = y;
					MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
					if (passed)
						Coeff[M] = coeff, Tindex[M] = UV_index;
					else
						M--;
				}
				else
					Coeff[M] = coeff, Tindex[M] = UV_index;
			}
			else
				M--;
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			lpROI_calculated[y*width1 + x] = true;
			firstpoint = false;
		}

		while (M >= 0)
		{
			if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) >= 0)
			{
				double elapsed = omp_get_wtime() - start;
				printLOG("%d good points\n", total_calc_points + cp);
				printLOG("%.2f%% . TE: %.2f TR: %.2f\n", 100.0 * (UV_index_n + 1)*step*step / total_valid_points, elapsed, elapsed / (percent + increment)*(100.0 - percent));
				percent += increment;
			}

			UV_index = Tindex[M];
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			M--; // Remove from the queque

			if ((y + step) < height1 && tROI[(y + step)*width1 + x] && !lpROI_calculated[(y + step)*width1 + x])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x, y + step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x, lpUV_xy[tid2 + 1] = y + step;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x + (y + step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[(y + step)*width1 + x] = true;
			}
			if ((y - step) >= 0 && tROI[(y - step)*width1 + x] && !lpROI_calculated[(y - step)*width1 + x])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x, y - step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x, lpUV_xy[tid2 + 1] = y - step;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x + (y - step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[(y - step)*width1 + x] = true;
			}
			if ((x - step) >= 0 && tROI[y*width1 + x - step] && !lpROI_calculated[y*width1 + x - step])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x - step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x - step, lpUV_xy[tid2 + 1] = y;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x - step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[y*width1 + x - step] = true;
			}
			if ((x + step) < width1 && tROI[y*width1 + x + step] && !lpROI_calculated[y*width1 + x + step])
			{
				tid1 = UV_index_n + 1, tid2 = 2 * tid1;
				if (DIC_CheckPointValidity(tROI, x + step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[tid2] = x + step, lpUV_xy[tid2 + 1] = y;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[tid2], ImgPt[1] = lpUV_xy[tid2 + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;
					}

					coeff = DIC_Calculation(tid1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (GeoVerify == 1)
						{
							passed = false;
							CorresPoints[0].x = lpUV[tid1] + lpUV_xy[tid2], CorresPoints[0].y = lpUV[tid1 + UV_length] + lpUV_xy[tid2 + 1], CorresPoints[1].x = lpUV_xy[tid2], CorresPoints[1].y = lpUV_xy[tid2 + 1];
							MultiViewQualityCheck(CorresPoints, Pmat, 0, K, distortion, &passed, 2, 1, triThresh, NULL, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff, Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							for (m = 0; m < 6; m++)
								WarpingParas[x + step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[y*width1 + x + step] = true;
			}
		}

		if (cp > 0)
			UV_index_n++;
		PointPerSeed[kk] = cp;
		total_calc_points += cp;
	}
	printLOG("...%d points growed\n", total_calc_points - total_calc_points1);
	//// DIC calculation: End

	SDSrcPts.reserve(total_calc_points), SDDstPts.reserve(total_calc_points);
	for (ii = 0; ii < total_calc_points; ii++)
	{
		if (lpUV[ii] != lpUV[ii] || lpUV[UV_length + ii] != lpUV[UV_length + ii])
			continue;

		if (DIC_Algo <= 1)
		{
			ImgPt[0] = lpUV_xy[2 * ii], ImgPt[1] = lpUV_xy[2 * ii + 1], ImgPt[2] = 1.0;
			cross_product(ImgPt, Epipole, epipline);
			double denum = sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
			direction[0] = -epipline[1] / denum, direction[1] = epipline[0] / denum;

			SDSrcPts.push_back(Point2d(lpUV_xy[2 * ii], lpUV_xy[2 * ii + 1]));
			SDDstPts.push_back(Point2d(lpUV[ii] * direction[0], lpUV[ii] * direction[1]));
		}
		else
		{
			SDSrcPts.push_back(Point2d(lpUV_xy[2 * ii], lpUV_xy[2 * ii + 1]));
			SDDstPts.push_back(Point2d(lpUV[ii], lpUV[UV_length + ii]));
		}
	}


	delete[]Tindex, delete[]Coeff, delete[]Znssd_reqd;
	delete[]Para, delete[]lpUV_xy, delete[]lpResult_UV, delete[]lpUV, delete[]displacement;
	delete[]PointPerSeed, delete[]lpImageData, delete[]iteration_check;

	return 0;
}



