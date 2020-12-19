#if !defined(GEOMETRY2_H )
#define GEOMETRY2_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <complex>
#include <omp.h>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>


#include <ceres/ceres.h>
#include <ceres/types.h>
#include <ceres/rotation.h>
#include "glog/logging.h"

#include "../DataStructure.h"
#include "../TrackMatch/MatchingTracking.h"
#include "../TrackMatch/FeatureEst.h"
#include "../Ulti/GeneralUlti.h"
#include "../Ulti/MiscAlgo.h"
#include "../Ulti/MathUlti.h"
#include "../Ulti/DataIO.h"
#include "../ImgPro/ImagePro.h"

#include "../ThirdParty/USAC/FundamentalMatrixEstimator.h"
#include "../ThirdParty/USAC/HomographyEstimator.h"
#include "../ThirdParty/USAC/USAC.h"
#include "../ThirdParty/rP6P/r6p.h"
#include "../ThirdParty/SiftGPU/src/SiftGPU/SiftGPU.h"
#include "Geometry1.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif


using namespace cv;

struct PinholeReprojectionErrorSimple_PointOnly {
	PinholeReprojectionErrorSimple_PointOnly(double *Pmat, double observed_x, double observed_y, double iscale) :observed_x(observed_x), observed_y(observed_y), iscale(iscale)
	{
		P = Pmat;
	}

	template <typename T>	bool operator()(const T* const points, T* residuals) 	const
	{
		T numX = (T)P[0] * points[0] + (T)P[1] * points[1] + (T)P[2] * points[2] + (T)P[3];
		T numY = (T)P[4] * points[0] + (T)P[5] * points[1] + (T)P[6] * points[2] + (T)P[7];
		T denum = (T)P[8] * points[0] + (T)P[9] * points[1] + (T)P[10] * points[2] + (T)P[11];

		//residuals[0] = softl1((numX / denum - T(observed_x)), 1.0) / (T)scale;
		//residuals[1] = softl1((numY / denum - T(observed_y)), 1.0) / (T)scale;

		residuals[0] = (numX / denum - T(observed_x)) * (T)iscale;
		residuals[1] = (numY / denum - T(observed_y)) * (T)iscale;

		return true;
	}

	static ceres::CostFunction* Create(double *Pmat, const double observed_x, const double observed_y, const double iscale)
	{
		return (new ceres::AutoDiffCostFunction<PinholeReprojectionErrorSimple_PointOnly, 2, 3>(new PinholeReprojectionErrorSimple_PointOnly(Pmat, observed_x, observed_y, iscale)));
	}

	double observed_x, observed_y, iscale, *P;
};
struct PinholeDistortionReprojectionError_PointOnly {
	PinholeDistortionReprojectionError_PointOnly(double *IntrinsicIn, double *DistortionIn, double *RTIn, double observed_x, double observed_y, double iscale) :
		Intrinsic(IntrinsicIn), Distortion(DistortionIn), RT(RTIn), observed_x(observed_x), observed_y(observed_y), iscale(iscale) {	}
	template <typename T>	bool operator()(const T* const XYZ, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T point[3] = { T(XYZ[0]), T(XYZ[1]), T(XYZ[2]) }, r[3] = { (T)RT[0], (T)RT[1] , (T)RT[2] }, p[3];
		ceres::AngleAxisRotatePoint(r, point, p);

		// camera[3,4,5] are the translation.
		p[0] += (T)RT[3], p[1] += (T)RT[4], p[2] += (T)RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Apply second and fourth order radial distortion.
		T xcn2 = xcn*xcn, ycn2 = ycn*ycn, xycn = xcn*ycn, r2 = xcn2 + ycn2, r4 = r2*r2, r6 = r2*r4;
		T radial = T(1.0) + (T)Distortion[0] * r2 + (T)Distortion[1] * r4 + (T)Distortion[2] * r6;
		T tangentialX = T(2.0)*(T)Distortion[4] * xycn + (T)Distortion[3] * (r2 + T(2.0)*xcn2);
		T tangentailY = (T)Distortion[4] * (r2 + T(2.0)*ycn2) + T(2.0)*(T)Distortion[3] * xycn;
		T prismX = (T)Distortion[5] * r2;
		T prismY = (T)Distortion[6] * r2;
		T xcn_ = radial*xcn + tangentialX + prismX;
		T ycn_ = radial*ycn + tangentailY + prismY;

		// Compute final projected point position.
		T predicted_x = (T)Intrinsic[0] * xcn_ + (T)Intrinsic[2] * ycn_ + (T)Intrinsic[3];
		T predicted_y = (T)Intrinsic[1] * ycn_ + (T)Intrinsic[4];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) * iscale;
		residuals[1] = (predicted_y - T(observed_y)) * iscale;

		return true;
	}
	static ceres::CostFunction* Create(double *Intrinsic, double *Distortion, double *RT, const double observed_x, const double observed_y, double iscale) {
		return (new ceres::AutoDiffCostFunction<PinholeDistortionReprojectionError_PointOnly, 2, 3>(new PinholeDistortionReprojectionError_PointOnly(Intrinsic, Distortion, RT, observed_x, observed_y, iscale)));
	}
	static ceres::CostFunction* CreateNumDif(double *Intrinsic, double *Distortion, double *RT, const double observed_x, const double observed_y, double iscale) {
		return (new ceres::NumericDiffCostFunction<PinholeDistortionReprojectionError_PointOnly, ceres::CENTRAL, 2, 3>(new PinholeDistortionReprojectionError_PointOnly(Intrinsic, Distortion, RT, observed_x, observed_y, iscale)));
	}
	double *Intrinsic, *Distortion, *RT;
	double observed_x, observed_y, iscale;
};
struct PinholeDistortionReprojectionError_PosePoint {
	PinholeDistortionReprojectionError_PosePoint(double *IntrinsicIn, double observed_x, double observed_y, double iscale) : observed_x(observed_x), observed_y(observed_y), iscale(iscale) { Intrinsic = IntrinsicIn; }
	template <typename T>	bool operator()(const T* const RT, const T* const XYZ, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T point[3] = { T(XYZ[0]), T(XYZ[1]), T(XYZ[2]) }, p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Compute final projected point position.
		T predicted_x = (T)Intrinsic[0] * xcn + (T)Intrinsic[2] * ycn + (T)Intrinsic[3];
		T predicted_y = (T)Intrinsic[1] * ycn + (T)Intrinsic[4];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) * iscale;
		residuals[1] = (predicted_y - T(observed_y)) * iscale;

		return true;
	}
	static ceres::CostFunction* Create(double *Intrinsic, const double observed_x, const double observed_y, double iscale) {
		return (new ceres::AutoDiffCostFunction<PinholeDistortionReprojectionError_PosePoint, 2, 6, 3>(new PinholeDistortionReprojectionError_PosePoint(Intrinsic, observed_x, observed_y, iscale)));
	}
	static ceres::CostFunction* CreateNumDif(double *Intrinsic, const double observed_x, const double observed_y, double iscale) {
		return (new ceres::NumericDiffCostFunction<PinholeDistortionReprojectionError_PosePoint, ceres::CENTRAL, 2, 6, 3>(new PinholeDistortionReprojectionError_PosePoint(Intrinsic, observed_x, observed_y, iscale)));
	}
	double *Intrinsic;
	double observed_x, observed_y, iscale;
};
struct PinholeReprojectionError {
	PinholeReprojectionError(double observed_x, double observed_y, double scale) : observed_x(observed_x), observed_y(observed_y), scale(scale) {}

	template <typename T>	bool operator()(const T* const intrinsic, const T* const RT, const T* const point, T* residuals) 	const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xn = p[0] / p[2];
		T yn = p[1] / p[2];

		residuals[0] = (intrinsic[0] * xn + intrinsic[2] * yn + intrinsic[3] - T(observed_x)) / (T)scale;
		residuals[1] = (intrinsic[1] * yn + intrinsic[4] - T(observed_y)) / (T)scale;

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale)
	{
		return (new ceres::AutoDiffCostFunction<PinholeReprojectionError, 2, 5, 6, 3>(new PinholeReprojectionError(observed_x, observed_y, scale)));
	}

	static ceres::CostFunction* CreateNumerDiff(const double observed_x, const double observed_y, double scale)
	{
		return (new ceres::NumericDiffCostFunction<PinholeReprojectionError, ceres::CENTRAL, 2, 5, 6, 3>(new PinholeReprojectionError(observed_x, observed_y, scale)));
	}

	double observed_x, observed_y, scale;
};
struct PinholeDistortionReprojectionError {
	PinholeDistortionReprojectionError(double observed_x, double observed_y, double scale) : observed_x(observed_x), observed_y(observed_y), scale(scale) {}
	template <typename T>	bool operator()(const T* const intrinsic, const T* const distortion, const T* const RT, const T* const point, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Apply second and fourth order radial distortion.
		T xcn2 = xcn*xcn, ycn2 = ycn*ycn, xycn = xcn*ycn, r2 = xcn2 + ycn2, r4 = r2*r2, r6 = r2*r4;
		T radial = T(1.0) + distortion[0] * r2 + distortion[1] * r4 + distortion[2] * r6;
		T tangentialX = T(2.0)*distortion[4] * xycn + distortion[3] * (r2 + T(2.0)*xcn2);
		T tangentailY = distortion[4] * (r2 + T(2.0)*ycn2) + T(2.0)*distortion[3] * xycn;
		T prismX = distortion[5] * r2;
		T prismY = distortion[6] * r2;
		T xcn_ = radial*xcn + tangentialX + prismX;
		T ycn_ = radial*ycn + tangentailY + prismY;

		// Compute final projected point position.
		T predicted_x = intrinsic[0] * xcn_ + intrinsic[2] * ycn_ + intrinsic[3];
		T predicted_y = intrinsic[1] * ycn_ + intrinsic[4];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) / (T)scale;
		residuals[1] = (predicted_y - T(observed_y)) / (T)scale;

		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale) {
		return (new ceres::AutoDiffCostFunction<PinholeDistortionReprojectionError, 2, 5, 7, 6, 3>(new PinholeDistortionReprojectionError(observed_x, observed_y, scale)));
	}
	static ceres::CostFunction* CreateNumerDiff(const double observed_x, const double observed_y, double scale)
	{
		return (new ceres::NumericDiffCostFunction<PinholeDistortionReprojectionError, ceres::CENTRAL, 2, 5, 7, 6, 3>(new PinholeDistortionReprojectionError(observed_x, observed_y, scale)));
	}

	double observed_x, observed_y, scale;
};
struct FOVReprojectionError {
	FOVReprojectionError(double observed_x, double observed_y, double scale) : observed_x(observed_x), observed_y(observed_y), scale(scale) {}

	template <typename T> bool operator()(const T* const intrinsic, const T* distortion, const T* const RT, const T* const point, T* residuals) 	const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to image coordinate
		T xcn = p[0] / p[2], ycn = p[1] / p[2];
		T u = intrinsic[0] * xcn + intrinsic[2] * ycn + intrinsic[3], v = intrinsic[1] * ycn + intrinsic[4];

		//Apply lens distortion
		T omega = distortion[0], DistCtrX = T(distortion[1]), DistCtrY = T(distortion[2]);
		T x = u - DistCtrX, y = v - DistCtrY;
		T ru = ceres::sqrt(x*x + y*y), rd = ceres::atan(T(2.0)*ru*ceres::tan(T(0.5)*omega)) / omega;
		T t = rd / ru;
		T x_u = t*x, y_u = t*y;

		residuals[0] = (x_u + DistCtrX - T(observed_x)) / (T)scale,
			residuals[1] = (y_u + DistCtrY - T(observed_y)) / (T)scale;
		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double scale)
	{
		return (new ceres::AutoDiffCostFunction<FOVReprojectionError, 2, 5, 3, 3, 3>(new FOVReprojectionError(observed_x, observed_y, scale)));
	}

	double observed_x, observed_y, scale;
};
struct FOVReprojectionError2 {
	FOVReprojectionError2(double observed_x, double observed_y, double scale) : observed_x(observed_x), observed_y(observed_y), scale(scale) {}

	template <typename T> bool operator()(const T* const fxfy, const T* const skew, const T* const uv0, const T* distortion, const T* const RT, const T* const point, T* residuals) 	const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to image coordinate
		T xcn = p[0] / p[2], ycn = p[1] / p[2];
		T u = fxfy[0] * xcn + skew[0] * ycn + uv0[0], v = fxfy[1] * ycn + uv0[1];

		//Apply lens distortion
		T omega = distortion[0], DistCtrX = T(distortion[1]), DistCtrY = T(distortion[2]);
		T x = u - DistCtrX, y = v - DistCtrY;
		T ru = ceres::sqrt(x*x + y*y), rd = ceres::atan(T(2.0)*ru*ceres::tan(T(0.5)*omega)) / omega;
		T t = rd / ru;
		T x_u = t*x, y_u = t*y;

		residuals[0] = (x_u + DistCtrX - T(observed_x)) / (T)scale,
			residuals[1] = (y_u + DistCtrY - T(observed_y)) / (T)scale;
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale)
	{
		return (new ceres::AutoDiffCostFunction<FOVReprojectionError2, 2, 2, 1, 2, 3, 6, 3>(new FOVReprojectionError2(observed_x, observed_y, scale)));
	}

	double observed_x, observed_y, scale;
};
struct FOVReprojectionError3 {
	FOVReprojectionError3(double observed_x, double observed_y, double scale) : observed_x(observed_x), observed_y(observed_y), scale(scale) {}

	template <typename T> bool operator()(const T* const intrinsic, const T* distortion, const T* const RT, const T* const point, T* residuals) 	const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to image coordinate
		T xcn = p[0] / p[2], ycn = p[1] / p[2];

		//Apply lens distortion
		T omega = distortion[0];
		T ru = ceres::sqrt(xcn*xcn + ycn*ycn), rd = ceres::atan(T(2.0)*ru*ceres::tan(T(0.5)*omega)) / omega;
		T t = rd / ru;
		T x_u = t*xcn, y_u = t*ycn;
		T u = intrinsic[0] * x_u + intrinsic[2] * y_u + intrinsic[3], v = intrinsic[1] * y_u + intrinsic[4];

		residuals[0] = (u - T(observed_x)) / (T)scale, residuals[1] = (v - T(observed_y)) / (T)scale;
		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double scale)
	{
		return (new ceres::AutoDiffCostFunction<FOVReprojectionError3, 2, 5, 1, 6, 3>(new FOVReprojectionError3(observed_x, observed_y, scale)));
	}

	double observed_x, observed_y, scale;
};
struct PinholeDistortionReprojectionError2 {
	PinholeDistortionReprojectionError2(double observed_x, double observed_y, double scale) : observed_x(observed_x), observed_y(observed_y), scale(scale) {}
	template <typename T>	bool operator()(const T* const fxfy, const T* const skew, const T* const uv0, const T* const Radial12, const T* const Tangential12, const T*const Radial3, const T*Prism, const T* const RT, const T* const point, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Apply second and fourth order radial distortion.
		T xcn2 = xcn*xcn, ycn2 = ycn*ycn, xycn = xcn*ycn, r2 = xcn2 + ycn2, r4 = r2*r2, r6 = r2*r4;
		T radial = T(1.0) + Radial12[0] * r2 + Radial12[1] * r4 + Radial3[0] * r6;
		T tangentialX = T(2.0)*Tangential12[1] * xycn + Tangential12[0] * (r2 + T(2.0)*xcn2);
		T tangentailY = Tangential12[1] * (r2 + T(2.0)*ycn2) + T(2.0)*Tangential12[0] * xycn;
		T prismX = Prism[0] * r2;
		T prismY = Prism[1] * r2;
		T xcn_ = radial*xcn + tangentialX + prismX;
		T ycn_ = radial*ycn + tangentailY + prismY;

		// Compute final projected point position.
		T predicted_x = fxfy[0] * xcn_ + skew[0] * ycn_ + uv0[0];
		T predicted_y = fxfy[1] * ycn_ + uv0[1];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) / (T)scale;
		residuals[1] = (predicted_y - T(observed_y)) / (T)scale;

		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale) {
		return (new ceres::AutoDiffCostFunction<PinholeDistortionReprojectionError2, 2, 2, 1, 2, 2, 2, 1, 2, 6, 3>(new PinholeDistortionReprojectionError2(observed_x, observed_y, scale)));
	}
	double observed_x, observed_y, scale;
};
struct PinholeDistortionReprojectionError3 {
	PinholeDistortionReprojectionError3(double observed_x, double observed_y, double X, double Y, double Z, double scale) : observed_x(observed_x), observed_y(observed_y), X(X), Y(Y), Z(Z), scale(scale) {}
	template <typename T>	bool operator()(const T* const fxfy, const T* const skew, const T* const uv0, const T* const Radial12, const T* const Tangential12, const T*const Radial3, const T*Prism, const T* const RT, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T point[3] = { T(X), T(Y), T(Z) }, p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Apply second and fourth order radial distortion.
		T xcn2 = xcn*xcn, ycn2 = ycn*ycn, xycn = xcn*ycn, r2 = xcn2 + ycn2, r4 = r2*r2, r6 = r2*r4;
		T radial = T(1.0) + Radial12[0] * r2 + Radial12[1] * r4 + Radial3[0] * r6;
		T tangentialX = T(2.0)*Tangential12[1] * xycn + Tangential12[0] * (r2 + T(2.0)*xcn2);
		T tangentailY = Tangential12[1] * (r2 + T(2.0)*ycn2) + T(2.0)*Tangential12[0] * xycn;
		T prismX = Prism[0] * r2;
		T prismY = Prism[1] * r2;
		T xcn_ = radial*xcn + tangentialX + prismX;
		T ycn_ = radial*ycn + tangentailY + prismY;

		// Compute final projected point position.
		T predicted_x = fxfy[0] * xcn_ + skew[0] * ycn_ + uv0[0];
		T predicted_y = fxfy[1] * ycn_ + uv0[1];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) / (T)scale;
		residuals[1] = (predicted_y - T(observed_y)) / (T)scale;

		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double X, const double Y, const double Z, double scale) {
		return (new ceres::AutoDiffCostFunction<PinholeDistortionReprojectionError3, 2, 2, 1, 2, 2, 2, 1, 2, 6>(new PinholeDistortionReprojectionError3(observed_x, observed_y, X, Y, Z, scale)));
	}
	static ceres::CostFunction* CreateNumDif(const double observed_x, const double observed_y, const double X, const double Y, const double Z, double scale) {
		return (new ceres::NumericDiffCostFunction<PinholeDistortionReprojectionError3, ceres::CENTRAL, 2, 2, 1, 2, 2, 2, 1, 2, 6>(new PinholeDistortionReprojectionError3(observed_x, observed_y, X, Y, Z, scale)));
	}
	double observed_x, observed_y, X, Y, Z, scale;
};
struct PinholeDistortionReprojectionError4 {
	PinholeDistortionReprojectionError4(double *IntrinsicIn, double observed_x, double observed_y, double X, double Y, double Z, double scale) : observed_x(observed_x), observed_y(observed_y), X(X), Y(Y), Z(Z), scale(scale) { Intrinsic = IntrinsicIn; }
	template <typename T>	bool operator()(const T* const RT, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T point[3] = { T(X), T(Y), T(Z) }, p[3];
		ceres::AngleAxisRotatePoint(RT, point, p);

		// camera[3,4,5] are the translation.
		p[0] += RT[3], p[1] += RT[4], p[2] += RT[5];

		// Project to normalize coordinate
		T xcn = p[0] / p[2];
		T ycn = p[1] / p[2];

		// Compute final projected point position.
		T predicted_x = (T)Intrinsic[0] * xcn + (T)Intrinsic[2] * ycn + (T)Intrinsic[3];
		T predicted_y = (T)Intrinsic[1] * ycn + (T)Intrinsic[4];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) / (T)scale;
		residuals[1] = (predicted_y - T(observed_y)) / (T)scale;

		return true;
	}
	static ceres::CostFunction* Create(double *Intrinsic, const double observed_x, const double observed_y, const double X, const double Y, const double Z, double scale) {
		return (new ceres::AutoDiffCostFunction<PinholeDistortionReprojectionError4, 2, 6>(new PinholeDistortionReprojectionError4(Intrinsic, observed_x, observed_y, X, Y, Z, scale)));
	}
	static ceres::CostFunction* CreateNumDif(double *Intrinsic, const double observed_x, const double observed_y, const double X, const double Y, const double Z, double scale) {
		return (new ceres::NumericDiffCostFunction<PinholeDistortionReprojectionError4, ceres::CENTRAL, 2, 6>(new PinholeDistortionReprojectionError4(Intrinsic, observed_x, observed_y, X, Y, Z, scale)));
	}
	double *Intrinsic;
	double observed_x, observed_y, X, Y, Z, scale;
};
struct CayleyReprojectionError {
	CayleyReprojectionError(double *intrinsicIn, double observed_x, double observed_y, double scale, int width, int height) : observed_x(observed_x), observed_y(observed_y), scale(scale), width(width), height(height)
	{
		intrinsic = intrinsicIn;
	}

	template <typename T>	bool operator()(const double* const rt, const double * const wt, const double* const point, T* residuals) const
	{
		Point2d predicted(0.0, 0.0);
		Point3d p3d(point[0], point[1], point[2]);
		double rt_[6] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };
		double wt_[6] = { wt[0], wt[1], wt[2], wt[3], wt[4], wt[5] };

		int count = CayleyProjection(intrinsic, rt_, wt_, predicted, p3d, width, height);
		residuals[0] = (predicted.x - observed_x) / scale, residuals[1] = (predicted.y - observed_y) / scale;

		return true;
	}

	static ceres::CostFunction* Create(double *intrinsic, const double observed_x, const double observed_y, double scale, int width, int height)
	{
		return (new ceres::NumericDiffCostFunction<CayleyReprojectionError, ceres::CENTRAL, 2, 6, 6, 3>(new CayleyReprojectionError(intrinsic, observed_x, observed_y, scale, width, height)));
	}

	int width, height;
	double *intrinsic, observed_x, observed_y, scale;
};
struct CayleyDistortionReprojectionError {
	CayleyDistortionReprojectionError(double observed_x, double observed_y, double scale, int width, int height) : observed_x(observed_x), observed_y(observed_y), scale(scale), width(width), height(height) {}
	template <typename T>	bool operator()(const double* const intrinsic, const double* const distortion, const double* const rt, const double * const wt, const double* const point, T* residuals) const
	{
		Point2d predicted(0.0, 0.0);
		Point3d p3d(point[0], point[1], point[2]);
		double intrinsic_[5] = { intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4] };
		double distortion_[7] = { distortion[0], distortion[1], distortion[2], distortion[3], distortion[4], distortion[5], distortion[6] };
		double rt_[6] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };
		double wt_[6] = { wt[0], wt[1], wt[2], wt[3], wt[4], wt[5] };

		int count = CayleyDistortionProjection(intrinsic_, distortion_, rt_, wt_, predicted, p3d, width, height);
		residuals[0] = (predicted.x - observed_x) / scale, residuals[1] = (predicted.y - observed_y) / scale;

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale, int width, int height)
	{
		return (new ceres::NumericDiffCostFunction<CayleyDistortionReprojectionError, ceres::CENTRAL, 2, 5, 7, 6, 6, 3>(new CayleyDistortionReprojectionError(observed_x, observed_y, scale, width, height)));
	}

	int width, height;
	double observed_x, observed_y, scale;
};
struct CayleyFOVReprojectionError {
	CayleyFOVReprojectionError(double observed_x, double observed_y, double scale, int width, int height) : observed_x(observed_x), observed_y(observed_y), scale(scale), width(width), height(height) {}
	template <typename T>	bool operator()(const double* const intrinsic, const double* const distortion, const double* const rt, const double * const wt, const double* const point, T* residuals) const
	{
		Point2d predicted(0.0, 0.0);
		Point3d p3d(point[0], point[1], point[2]);
		double intrinsic_[5] = { intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4] };
		double distortion_[3] = { distortion[0], distortion[1], distortion[2] };
		double rt_[6] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };
		double wt_[6] = { wt[0], wt[1], wt[2], wt[3], wt[4], wt[5] };

		int count = CayleyFOVProjection(intrinsic_, distortion_, rt_, wt_, predicted, p3d, width, height);
		residuals[0] = (predicted.x - observed_x) / scale, residuals[1] = (predicted.y - observed_y) / scale;

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale, int width, int height)
	{
		return (new ceres::NumericDiffCostFunction<CayleyFOVReprojectionError, ceres::CENTRAL, 2, 5, 3, 6, 6, 3>(new CayleyFOVReprojectionError(observed_x, observed_y, scale, width, height)));
	}

	int width, height;
	double observed_x, observed_y, scale;
};
struct CayleyFOVReprojection2Error {
	CayleyFOVReprojection2Error(double observed_x, double observed_y, double scale, int width, int height) : observed_x(observed_x), observed_y(observed_y), scale(scale), width(width), height(height) {}
	template <typename T>	bool operator()(const double* const intrinsic, const double* const distortion, const double* const rt, const double * const wt, const double* const point, T* residuals) const
	{
		Point2d predicted(0.0, 0.0);
		Point3d p3d(point[0], point[1], point[2]);
		double intrinsic_[5] = { intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4] };
		double distortion_ = { distortion[0] };
		double rt_[6] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };
		double wt_[6] = { wt[0], wt[1], wt[2], wt[3], wt[4], wt[5] };

		int count = CayleyFOVProjection2(intrinsic_, &distortion_, rt_, wt_, predicted, p3d, width, height);
		residuals[0] = (predicted.x - observed_x) / scale, residuals[1] = (predicted.y - observed_y) / scale;

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, double scale, int width, int height)
	{
		return (new ceres::NumericDiffCostFunction<CayleyFOVReprojection2Error, ceres::CENTRAL, 2, 5, 1, 6, 6, 3>(new CayleyFOVReprojection2Error(observed_x, observed_y, scale, width, height)));
	}

	int width, height;
	double observed_x, observed_y, scale;
};
void RollingShutterSplineProjection(double *intrinsic, int *ActingID, double *ActingControlPose, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, int se3, Point2d &predicted, Point3d &point, int frameID, int width, int height);
struct RollingShutterSplineReprojectionError {
	RollingShutterSplineReprojectionError(double *IntrinsicIn, double *KnotLocIn, int nBreak, int nCtrl, int SplineOrder, int se3, int *ActingIDIn, Point2d observed2D, double scale, int pid, int frameID, int width, int height) :
		nBreak(nBreak), nCtrl(nCtrl), SplineOrder(SplineOrder), se3(se3), observed2D(observed2D), scale(scale), pid(pid), frameID(frameID), width(width), height(height)
	{
		KnotLoc = KnotLocIn;
		Intrinsic = IntrinsicIn;
		for (int ii = 0; ii < SplineOrder + 2; ii++)
			ActingID[ii] = ActingIDIn[ii];
	}

	template <typename T>	bool operator()(const double* const ControlPoses0, const double* const ControlPoses1, const double* const ControlPoses2, const double* const ControlPoses3,
		const double* const ControlPoses4, const double* const ControlPoses5, const double* const point, T* residuals) const
	{
		Point3d p3d(point[0], point[1], point[2]);

		double control[36];
		for (int ii = 0; ii < 6; ii++)
			control[ii] = ControlPoses0[ii], control[ii + 6] = ControlPoses1[ii], control[ii + 12] = ControlPoses2[ii],
			control[ii + 18] = ControlPoses3[ii], control[ii + 24] = ControlPoses4[ii], control[ii + 30] = ControlPoses5[ii];

		Point2d predicted2D;
		int *aID = new int[6];
		for (int ii = 0; ii < 6; ii++)
			aID[ii] = ActingID[ii];

		RollingShutterSplineProjection(Intrinsic, aID, control, KnotLoc, nBreak, nCtrl, SplineOrder, se3, predicted2D, p3d, frameID, width, height);

		residuals[0] = (predicted2D.x - observed2D.x) / scale, residuals[1] = (predicted2D.y - observed2D.y) / scale;

		delete[]aID;
		return true;
	}
	static ceres::CostFunction* CreateNumerDiff(double *Intrinsic, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, int se3, Point2d observed2D, int *ActingID, double scale, int pid, int frameID, int width, int height)
	{
		return (new ceres::NumericDiffCostFunction<RollingShutterSplineReprojectionError, ceres::CENTRAL, 2, 6, 6, 6, 6, 6, 6, 3>
			(new RollingShutterSplineReprojectionError(Intrinsic, KnotLoc, nBreak, nCtrl, SplineOrder, se3, ActingID, observed2D, scale, pid, frameID, width, height)));
	}

	int se3;
	int pid, nBreak, nCtrl, SplineOrder;
	double scale, *Intrinsic, *KnotLoc;
	int width, height, frameID, ActingID[6];
	Point2d observed2D;
};
void RollingShutterDistortionSplineProjection(double *intrinsic, double *distortion, int *ActingID, double *ActingControlPose, double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, Point2d &predicted, Point3d point, int frameID, int width, int height);
struct RollingShutterDistortionSplineReprojectionError {
	RollingShutterDistortionSplineReprojectionError(double *KnotLocIn, int nBreak, int nCtrl, int SplineOrder, int *ActingIDIn, Point2d observed2D, double scale, int frameID, int width, int height) :
		nBreak(nBreak), nCtrl(nCtrl), SplineOrder(SplineOrder), observed2D(observed2D), scale(scale), pid(pid), frameID(frameID), width(width), height(height)
	{
		KnotLoc = KnotLocIn;
		for (int ii = 0; ii < SplineOrder + 2; ii++)
			ActingID[ii] = ActingIDIn[ii];
	}

	template <typename T>	bool operator()(const double* const intrinsic, const double* const distortion,
		const double* const ControlPoses0, const double* const ControlPoses1, const double* const ControlPoses2, const double* const ControlPoses3, const double* const ControlPoses4, const double* const ControlPoses5,
		const double* const point, T* residuals) const
	{
		double intrinsic_[5] = { intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4] };
		double distortion_[7] = { distortion[0], distortion[1], distortion[2], distortion[3], distortion[4], distortion[5], distortion[6] };
		Point3d p3d(point[0], point[1], point[2]);

		double control[36];
		for (int ii = 0; ii < 6; ii++)
			control[ii] = ControlPoses0[ii], control[ii + 6] = ControlPoses1[ii], control[ii + 12] = ControlPoses2[ii],
			control[ii + 18] = ControlPoses3[ii], control[ii + 24] = ControlPoses4[ii], control[ii + 30] = ControlPoses5[ii];

		int *aID = new int[6];
		for (int ii = 0; ii < 6; ii++)
			aID[ii] = ActingID[ii];

		Point2d predicted2D;
		RollingShutterDistortionSplineProjection(intrinsic_, distortion_, aID, control, KnotLoc, nBreak, nCtrl, SplineOrder, predicted2D, p3d, frameID, width, height);

		residuals[0] = (predicted2D.x - observed2D.x) / scale, residuals[1] = (predicted2D.y - observed2D.y) / scale;

		return true;
	}
	static ceres::CostFunction* CreateNumerDiff(double *KnotLoc, int nBreak, int nCtrl, int SplineOrder, Point2d observed2D, int *ActingID, double scale, int frameID, int width, int height)
	{
		return (new ceres::NumericDiffCostFunction<RollingShutterDistortionSplineReprojectionError, ceres::CENTRAL, 2, 5, 7, 6, 6, 6, 6, 6, 6, 3>
			(new RollingShutterDistortionSplineReprojectionError(KnotLoc, nBreak, nCtrl, SplineOrder, ActingID, observed2D, scale, frameID, width, height)));
	}

	int pid, nBreak, nCtrl, SplineOrder;
	double scale, *KnotLoc;
	int width, height, frameID, ActingID[6];
	Point2d observed2D;
};
struct RollingShutterDCTReprojectionError {
	RollingShutterDCTReprojectionError(double *IntrinsicIn, double *WeightIn, int nCoeffs, Point2d observed2D, double scale, int pid, int frameID, int width, int height) :
		nCoeffs(nCoeffs), observed2D(observed2D), scale(scale), pid(pid), frameID(frameID), width(width), height(height)
	{
		Intrinsic = IntrinsicIn;
		Weight = WeightIn;
	}

	template <typename T>    bool operator()(T const* const* Parameters, T* residuals)     const
	{
		T R[9], Trans[3], np[3];
		T *iB = new T[nCoeffs];

		//Get initial estimate of the projected location: must be in 0->n-1 range
		T subframeLoc = (T)(0.5 + frameID);
		if (subframeLoc > (T)(nCoeffs - 1))
			subframeLoc = (T)(nCoeffs - 1);

		//GenerateiDCTBasis(iB, nCoeffs, subframeLoc); 	//Get twist = iB*C;
		iB[0] = (T)(ceres::sqrt(1.0 / nCoeffs));
		double s = std::sqrt(2.0 / nCoeffs);
		for (int kk = 1; kk < nCoeffs; kk++)
			iB[kk] = (T)(s)*cos((T)(Pi*kk) *(subframeLoc + (T)0.5) / (T)nCoeffs);

		T twist[6];
		for (int jj = 0; jj < 6; jj++)
		{
			twist[jj] = (T)0.0;
			for (int ii = 0; ii < nCoeffs; ii++)
				twist[jj] += Parameters[jj + 1][ii] * (T)iB[ii];
		}

		//getRTFromTwist(twist, R, Trans);
		T t[3] = { twist[0], twist[1], twist[2] }, w[3] = { twist[3], twist[4], twist[5] };
		T theta = ceres::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), theta2 = theta* theta;
		T wx2[9], wx[9] = { (T)0.0, -w[2], w[1], w[2], (T)0.0, -w[0], -w[1], w[0], (T)0.0 };
		//mat_mul(wx, wx, wx2, 3, 3, 3);
		for (int ii = 0; ii < 3; ii++)
		{
			for (int jj = 0; jj < 3; jj++)
			{
				wx2[ii * 3 + jj] = (T)0.0;
				for (int kk = 0; kk < 3; kk++)
					wx2[ii * 3 + jj] += wx[ii * 3 + kk] * wx[kk * 3 + jj];
			}
		}

		T V[9] = { (T)1.0, (T)0.0, (T)0.0, (T)0.0, (T)1.0, (T)0.0, (T)0.0, (T)0.0, (T)1.0 };
		R[0] = (T)1.0, R[1] = (T)0.0, R[2] = (T)0.0, R[3] = (T)0.0, R[4] = (T)1.0, R[5] = (T)0.0, R[6] = (T)0.0, R[7] = (T)0.0, R[8] = (T)1.0;
		if (theta < (T)1.0e-9)
			Trans[0] = t[0], Trans[1] = t[1], Trans[2] = t[2]; //Rotation is idenity
		else
		{
			T A = sin(theta) / theta, B = ((T)1.0 - cos(theta)) / theta2, C = ((T)1.0 - A) / theta2;
			for (int ii = 0; ii < 9; ii++)
			{
				R[ii] += A*wx[ii] + B*wx2[ii];
				V[ii] += B*wx[ii] + C*wx2[ii];
			}

			//mat_mul(V, t, Trans, 3, 3, 1);
			for (int ii = 0; ii < 3; ii++)
			{
				Trans[ii] = (T)0.0;
				for (int kk = 0; kk < 3; kk++)
					Trans[ii] += V[ii * 3 + kk] * t[kk];
			}
		}

		//Initiate projection solver
		np[1] = R[3] * Parameters[0][0] + R[4] * Parameters[0][1] + R[5] * Parameters[0][2] + Trans[1];
		np[2] = R[6] * Parameters[0][0] + R[7] * Parameters[0][1] + R[8] * Parameters[0][2] + Trans[2];
		T ycn = np[1] / np[2], ycn_ = ycn;
		T v = (T)Intrinsic[1] * ycn + (T)Intrinsic[4]; //to get time info

		//Iteratively solve for ycn = P(ycn)*X
		T dif;
		int iter, iterMax = 20;
		for (iter = 0; iter < iterMax; iter++)
		{
			subframeLoc = (T)(0.5 + frameID);
			if (subframeLoc > (T)(nCoeffs - 1))
				subframeLoc = (T)(nCoeffs - 1);

			//GenerateiDCTBasis(iB, nCoeffs, subframeLoc); 	//Get twist = iB*C;
			iB[0] = (T)(ceres::sqrt(1.0 / nCoeffs));
			double s = std::sqrt(2.0 / nCoeffs);
			for (int kk = 1; kk < nCoeffs; kk++)
				iB[kk] = (T)(s)*cos((T)(Pi*kk) *(subframeLoc + (T)0.5) / (T)nCoeffs);

			T twist[6];
			for (int jj = 0; jj < 6; jj++)
			{
				twist[jj] = (T)0.0;
				for (int ii = 0; ii < nCoeffs; ii++)
					twist[jj] += Parameters[jj + 1][ii] * (T)iB[ii];
			}

			//getRTFromTwist(twist, R, Trans);
			T t[3] = { twist[0], twist[1], twist[2] }, w[3] = { twist[3], twist[4], twist[5] };
			T theta = ceres::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), theta2 = theta* theta;
			T wx2[9], wx[9] = { (T)0.0, -w[2], w[1], w[2], (T)0.0, -w[0], -w[1], w[0], (T)0.0 };
			//mat_mul(wx, wx, wx2, 3, 3, 3);
			for (int ii = 0; ii < 3; ii++)
			{
				for (int jj = 0; jj < 3; jj++)
				{
					wx2[ii * 3 + jj] = (T)0.0;
					for (int kk = 0; kk < 3; kk++)
						wx2[ii * 3 + jj] += wx[ii * 3 + kk] * wx[kk * 3 + jj];
				}
			}

			T V[9] = { (T)1.0, (T)0.0, (T)0.0, (T)0.0, (T)1.0, (T)0.0, (T)0.0, (T)0.0, (T)1.0 };
			R[0] = (T)1.0, R[1] = (T)0.0, R[2] = (T)0.0, R[3] = (T)0.0, R[4] = (T)1.0, R[5] = (T)0.0, R[6] = (T)0.0, R[7] = (T)0.0, R[8] = (T)1.0;
			if (theta < (T)1.0e-9)
				Trans[0] = t[0], Trans[1] = t[1], Trans[2] = t[2]; //Rotation is idenity
			else
			{
				T A = sin(theta) / theta, B = ((T)1.0 - cos(theta)) / theta2, C = ((T)1.0 - A) / theta2;
				for (int ii = 0; ii < 9; ii++)
				{
					R[ii] += A*wx[ii] + B*wx2[ii];
					V[ii] += B*wx[ii] + C*wx2[ii];
				}

				//mat_mul(V, t, Trans, 3, 3, 1);
				for (int ii = 0; ii < 3; ii++)
				{
					Trans[ii] = (T)0.0;
					for (int kk = 0; kk < 3; kk++)
						Trans[ii] += V[ii * 3 + kk] * t[kk];
				}
			}

			np[1] = R[3] * Parameters[0][0] + R[4] * Parameters[0][1] + R[5] * Parameters[0][2] + Trans[1];
			np[2] = R[6] * Parameters[0][0] + R[7] * Parameters[0][1] + R[8] * Parameters[0][2] + Trans[2];

			ycn = np[1] / np[2];
			v = Intrinsic[1] * ycn + Intrinsic[4];
			dif = ceres::abs((ycn - ycn_) / ycn_);
			if (dif < 1.0e-9)
				break;
			ycn_ = ycn;
		}

		//if (v<-1.0 || v>height)
		//	printf("Projection problem @Frame %d (%.2f)\n", frameID, v);

		np[0] = R[0] * Parameters[0][0] + R[1] * Parameters[0][1] + R[2] * Parameters[0][2] + Trans[0];
		T xcn = np[0] / np[2], u = Intrinsic[0] * xcn + Intrinsic[2] * ycn + Intrinsic[3];

		//if (iter > iterMax - 1 && dif > 1.0e-6)
		//	printf("Frame %d: %.2f %.2f %.9e \n", frameID, u, v, dif);

		residuals[0] = (u - (T)observed2D.x) / (T)scale, residuals[1] = (v - (T)observed2D.y) / (T)scale;

		delete[]iB;
		return true;
	}

	int width, height, pid, frameID, nCoeffs;
	double scale, *Intrinsic, *Weight;
	Point2d observed2D;
};
struct RollingShutterDCTRegularizationError {
	RollingShutterDCTRegularizationError(double *WeightIn, int nCoeffs, double sqrtlamda) : nCoeffs(nCoeffs), sqrtlamda(sqrtlamda)
	{
		Weight = WeightIn;
	}

	template <typename T>    bool operator()(T const* const* Parameters, T* residuals)     const
	{
		for (int jj = 0; jj < 6; jj++)
			for (int ii = 0; ii < nCoeffs; ii++)
				residuals[ii + 6 * nCoeffs] = (T)sqrtlamda*Parameters[jj][ii] * (T)Weight[ii];

		return true;
	}

	int nCoeffs;
	double sqrtlamda, *Weight;
};

//BA Ulti
double PinholeReprojectionErrorSimpleDebug(double *P, Point3d Point, Point2d uv);
void FOVReprojectionDistortionDebug(double *intrinsic, double* distortion, double* rt, Point2d observed, Point3d Point, double *residuals);
void FOVReprojectionDistortion2Debug(double *intrinsic, double* distortion, double* rt, Point2d observed, Point3d Point, double *residuals);
void PinholeReprojectionDebug(double *intrinsic, double* rt, Point2d &observed, Point3d Point, double *residuals);
int CayleyProjection(double *intrinsic, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height);
int CayleyReprojectionDebug(double *intrinsic, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals);
int CayleyDistortionProjection(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height);
int CayleyDistortionReprojectionDebug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals);
int CayleyFOVProjection(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height);
int CayleyFOVReprojectionDebug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals);
int CayleyFOVProjection2(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height);
int CayleyFOVReprojection2Debug(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &observed, Point3d Point, int width, int height, double *residuals);
void PinholeDistortionReprojectionDebug(double *intrinsic, double* distortion, double* rt, Point2d &observed, Point3d Point, double *residuals);
int ProjectionCayLeyReProjection(double *intrinsic, double* distortion, double* rt, double *wt, Point2d &predicted, Point3d Point, int width, int height);

//Triangulation BA
void NviewTriangulationNonLinear(double *P, Point2d *Point2D, Point3d *Point3D, double *ReprojectionError, int nviews, int npts);
void NviewTriangulationNonLinear(double *P, double *Point2D, double *Point3D, double *ReprojectionError, int nviews, int npts = 1);
void NviewTriangulationNonLinearCayley(CameraData *camInfo, double *Point2D, double *Point3D, double *ReprojectionError, int nviews, int npts);
//assume distortion-free points 
double CalibratedTwoViewBA(double *intrinsic1, double *intrinsic2, double *rt1, double *rt2, vector<Point2d> &pts1, vector<Point2d> &pts2, vector<Point3d> &P3D, vector<int> &validPid, double threshold, int LossType, int silent = 1);

//Camera localization
int MatchCameraToCorpus(char *Path, Corpus &corpusData, CameraData &CamInfoI, int cameraID, int timeID, int distortionCorrected, vector<int> &CorpusViewToMatch, const float nndrRatio = 0.7, const int ninlierThresh = 40);
int LocalizeCameraToCorpus(char *Path, Corpus &corpusData, CameraData  &cameraParas, int cameraID, int fixedIntrinsc, int fixedDistortion, int distortionCorrected, int timeID);
int ExhaustiveSearchForBestFocal(char *Path, Corpus &corpusData, CameraData &CamInfoI, int selectedCam, int frameID, bool FhasBeenInit = false);
int ForceLocalizeCameraToCorpus(char *Path, CameraData  &cameraParas, int cameraID, int fixIntrinsic, int fixDistortion, int distortionCorrected, int timeID, int fromCorpusGen, double *Extrinsic_Init = NULL);
int CameraPose1FrameBA(char *Path, CameraData &camera, vector<Point3d>  Vxyz, vector<Point2d> &uvAll3D, vector<double> &scaleAll3D, vector<int> &GlobalAnchor, vector<bool> &Good, int fixIntrinsic, int fixDistortion, int distortionCorrected, int useGlobalAnchor, int LossType, int debug);

//General BA
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2f> > &uvAll3D, vector<vector<float> > &scaleAll3D, vector<int> &SharedIntrinsicCamID, int nviews, int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, int fixed3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool verbose, int maxIter = -1);
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2f> > &uvAll3D, vector<vector<float> > &scaleAll3D, vector<int> &SharedIntrinsicCamID, int nviews, int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, vector<int> &VGlobalAnchor, int fixedGlobalAnchor3D, int fixedLocalAnchor3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool verbose, int maxIter = -1);
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<vector<double> > &scaleAll3D, vector<int> &SharedIntrinsicCamID, int nviews, int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, int fixed3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool verbose, int maxIter = -1);
int GenericBundleAdjustment(char *Path, CameraData *camera, vector<Point3d>  &Vxyz, vector < vector<int> > &viewIdAll3D, vector<vector<Point2d> > &uvAll3D, vector<vector<double> > &scaleAll3D, vector<int> &SharedIntrinsicCamID, int nviews, int fixIntrinsic, int fixDistortion, int fixPose, int fixFirstCamPose, int fixLocalPose, vector<int> &VGlobalAnchor, int fixedGlobalAnchor3D, int fixedLocalAnchor3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int LossType, bool PoseSmoothness, bool debug, bool verbose, int maxIter = -1);

//Video BA
int PerVideo_BA(char *Path, int selectedCamID, int startFrame, int stopFrame, int increFrame, int fixIntrinsic, int fixDistortion, int fixed3D, int fixLocal3D, int fixPose, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int ShutterModel, int RobustLoss, int doubleRefinement, double threshold);
int AllVideo_BA(char *Path, int nCams, int startF, int stopF, int increF, int fixIntrinsic, int fixDistortion, int fixPose, int fixfirstCamPose, int fixed3D, int fixSkew, int fixPrism, int distortionCorrected, int nViewsPlus, int ShutterModel, int RobustLoss, int doubleRefinement, double threshold);

struct Virtual3D_RS_ReProjectionEror {
	Virtual3D_RS_ReProjectionEror(double *intrinsicIn, double observed_x, double observed_y, int height, double scale) : observed_x(observed_x), observed_y(observed_y), height(height), scale(scale)
	{
		intrinsic = intrinsicIn;
	}
	template <typename T>	bool operator()(const T* const rt1, const T* const rt2, const T* const point, const T* const rp, T* residuals) const
	{
		//transform to camera coord
		T pi[3], p1[3], p2[3];
		ceres::AngleAxisRotatePoint(rt1, point, p1);
		ceres::AngleAxisRotatePoint(rt2, point, p2);

		p1[0] += rt1[3], p1[1] += rt1[4], p1[2] += rt1[5];
		p2[0] += rt2[3], p2[1] += rt2[4], p2[2] += rt2[5];

		//Time of the observation.
		T t = (T)(observed_y / height - 0.5);
		if (t > (T)0)
			t = rp[0] * t; //lie on the 1st frame
		else
			t = rp[0] * t + (T)1.0; //lie on the 2nd frame

		//Interpolated point
		for (int ii = 0; ii < 3; ii++)
			pi[ii] = (((T)1.0 - t)*p1[ii] + t*p2[ii]);

		// Project to normalize coordinate
		T xcn = pi[0] / pi[2], ycn = pi[1] / pi[2];

		// Compute final projected point position.
		T predicted_x = (T)intrinsic[0] * xcn + (T)intrinsic[2] * ycn + (T)intrinsic[3];
		T predicted_y = (T)intrinsic[1] * ycn + (T)intrinsic[4];

		// The error is the difference between the predicted and observed position.
		residuals[0] = (predicted_x - T(observed_x)) / (T)scale;
		residuals[1] = (predicted_y - T(observed_y)) / (T)scale;

		return true;
	}
	static ceres::CostFunction* Create(double *intrinsic, double observed_x, double observed_y, int height, double scale) {
		return (new ceres::AutoDiffCostFunction<Virtual3D_RS_ReProjectionEror, 2, 6, 6, 3, 1>(new Virtual3D_RS_ReProjectionEror(intrinsic, observed_x, observed_y, height, scale)));
	}
	static ceres::CostFunction* CreateNumDif(double *intrinsic, double observed_x, double observed_y, int height, double scale) {
		return (new ceres::NumericDiffCostFunction<Virtual3D_RS_ReProjectionEror, ceres::CENTRAL, 2, 6, 6, 3, 1>(new Virtual3D_RS_ReProjectionEror(intrinsic, observed_x, observed_y, height, scale)));
	}
	int height;
	double *intrinsic, observed_x, observed_y, scale, Period;
};
void Virtual3D_RS_ReProjectionEror(double *intrinsic, double *rt1, double * rt2, double *point, double rp, Point2d obser2D, int height, double *residuals);
int Virtual3D_RS_BA(double &rollingshutter_Percent, double *intrinsics, double *rt, bool *valid, double *Vxyz, vector<vector<int> > &fidPer3D, vector<vector<Point2d> > &uvPer3D, vector<vector<float> > &sPer3D, int imgheight, int nframes, int LossType, bool silent = true);

int VideoSplineRSBA(char *Path, int startFrame, int stopFrame, int selectedCams, int fixIntrinsic, int fixedDistortion, int fixed3D, int fixSkew, int fixPrism, int distortionCorrected, double threshold, int controlStep = 5, int SplineOrder = 4, int se3 = 0, bool debug = false);
int VideoDCTRSBA(char *Path, int startFrame, int stopFrame, int selectedCams, int distortionCorrected, int fixIntrinsic, int fixedDistortion, double threshold, int sampleStep = 5, double lamda = 0.1, bool debug = false);

struct TemporalOptimInterpStationaryCameraCeres {
	TemporalOptimInterpStationaryCameraCeres(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		P = Pin;
		ParaX = ParaXin, ParaY = ParaYin;
		double x = ParaXin[0], y = ParaX[0];
		frameID = frameIDin, nframes = nframesIn, interpAlgo = interpAlgoIn;
	}

	template <typename T>	bool operator()(const T* const XYZ, const T* const F, T* residuals) 	const
	{
		double Fi = F[0] + frameID;
		double Sx[3], Sy[3];
		if (Fi < 0.0)
			Fi = 0.0;
		if (Fi > nframes - 1)
			Fi = nframes - 1;

		double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];
		double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
		double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];

		Get_Value_Spline(ParaX, nframes, 1, Fi, 0, Sx, -1, interpAlgo);
		Get_Value_Spline(ParaY, nframes, 1, Fi, 0, Sy, -1, interpAlgo);

		residuals[0] = numX / denum - Sx[0];
		residuals[1] = numY / denum - Sy[0];


		return true;
	}

	static ceres::CostFunction* Create(double *Pin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpStationaryCameraCeres, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpStationaryCameraCeres(Pin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaX, *ParaY, *P;
};
struct TemporalOptimInterpMovingCameraCeres {
	TemporalOptimInterpMovingCameraCeres(double *AllPin, double *AllKin, double *AllQin, double *AllRin, double *AllCin, double *ParaCamCenterXIn, double *ParaCamCenterYIn, double *ParaCamCenterZIn, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		AllP = AllPin, AllK = AllKin, AllQ = AllQin, AllR = AllRin, AllC = AllCin;
		ParaCamCenterX = ParaCamCenterXIn, ParaCamCenterY = ParaCamCenterYIn, ParaCamCenterZ = ParaCamCenterZIn, ParaX = ParaXin, ParaY = ParaYin;
		frameID = frameIDin, nframes = nframesIn, interpAlgo = interpAlgoIn;
	}

	template <typename T>	bool operator()(const T* const XYZ, const T* const F, T* residuals) 	const
	{
		double Fi = F[0] + frameID;
		if (Fi < 0.0)
			Fi = 0.0;
		if (Fi > nframes - 2)
			Fi = nframes - 2;
		int lFi = (int)Fi, uFi = lFi + 1;
		double fFi = Fi - lFi;

		if (lFi < 0)
		{
			residuals[0] = 0.0;
			residuals[1] = 0.0;
			return true;
		}
		else if (uFi > nframes - 2)
		{
			residuals[0] = 0.0;
			residuals[1] = 0.0;
			return true;
		}

		double K[9], C[3], R[9], RT[12], P[12], Q[4];
		for (int ll = 0; ll < 9; ll++)
			K[ll] = AllK[9 * lFi + ll];

		for (int ll = 0; ll < 3; ll++)
			C[ll] = (1.0 - fFi)*AllC[3 * lFi + ll] + fFi*AllC[3 * uFi + ll]; //linear interpolation
		//Get_Value_Spline(ParaCamCenterX, nframes, 1, Fi, 0, &C[0], -1, interpAlgo);
		//Get_Value_Spline(ParaCamCenterY, nframes, 1, Fi, 0, &C[1], -1, interpAlgo);
		//Get_Value_Spline(ParaCamCenterZ, nframes, 1, Fi, 0, &C[2], -1, interpAlgo);

		for (int ll = 0; ll < 4; ll++)
			Q[ll] = AllQ[4 * lFi + ll];
		//QuaternionLinearInterp(&AllQ[4 * lFi], &AllQ[4 * uFi], Q, fFi);//linear interpolation

		/*//Give good result given 1frame offset--> strange so I use rigorous interplation instead
		lFi = (int)(Fi + 0.5);
		for (int ll = 0; ll < 3; ll++)
		C[ll] = AllC[3 * lFi + ll];
		for (int ll = 0; ll < 4; ll++)
		Q[ll] = AllQ[4 * lFi + ll];*/

		Quaternion2Rotation(Q, R);
		AssembleRT(R, C, RT, true);
		AssembleP(K, RT, P);

		double numX = P[0] * XYZ[0] + P[1] * XYZ[1] + P[2] * XYZ[2] + P[3];
		double numY = P[4] * XYZ[0] + P[5] * XYZ[1] + P[6] * XYZ[2] + P[7];
		double denum = P[8] * XYZ[0] + P[9] * XYZ[1] + P[10] * XYZ[2] + P[11];

		double Sx[3], Sy[3];
		Get_Value_Spline(ParaX, nframes, 1, Fi, 0, Sx, -1, interpAlgo);
		Get_Value_Spline(ParaY, nframes, 1, Fi, 0, Sy, -1, interpAlgo);

		residuals[0] = numX / denum - Sx[0];
		residuals[1] = numY / denum - Sy[0];
		if (ceres::abs(residuals[0]) > 5 || ceres::abs(residuals[1]) > 5)
			int a = 0;
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(double *AllPin, double *AllKin, double *AllQin, double *AllRin, double *AllCin, double *ParaCamCenterXIn, double *ParaCamCenterYin, double *ParaCamCenterZin, double *ParaXin, double *ParaYin, int frameIDin, int nframesIn, int interpAlgoIn)
	{
		return (new ceres::NumericDiffCostFunction<TemporalOptimInterpMovingCameraCeres, ceres::CENTRAL, 2, 3, 1>(new TemporalOptimInterpMovingCameraCeres(AllPin, AllKin, AllQin, AllRin, AllCin, ParaCamCenterXIn, ParaCamCenterYin, ParaCamCenterZin, ParaXin, ParaYin, frameIDin, nframesIn, interpAlgoIn)));
	}

	int frameID, nframes, interpAlgo;
	double F;
	double *ParaCamCenterX, *ParaCamCenterY, *ParaCamCenterZ, *ParaX, *ParaY;
	double *AllP, *AllK, *AllQ, *AllR, *AllC;
};
struct MotionPriorRaysCeres {
	MotionPriorRaysCeres(double *ray1, double *ray2, double *cc1, double *cc2, double timeStamp1, double timeStamp2, double epsilon, double lamda) : timeStamp1(timeStamp1), timeStamp2(timeStamp2), epsilon(epsilon), lamda(lamda)
	{
		rayDirect1 = ray1, center1 = cc1;
		rayDirect2 = ray2, center2 = cc2;
	}

	template <typename T>	bool operator()(const T* const depth1, const T* const depth2, T* residuals) 	const
	{
		T X = (T)(depth1[0] * rayDirect1[0] + center1[0]), Y = (T)(depth1[0] * rayDirect1[1] + center1[1]), Z = (T)(depth1[0] * rayDirect1[2] + center1[2]);
		T X2 = (T)(depth2[0] * rayDirect2[0] + center2[0]), Y2 = (T)(depth2[0] * rayDirect2[1] + center2[1]), Z2 = (T)(depth2[0] * rayDirect2[2] + center2[2]);
		T difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

		residuals[0] = (T)lamda*ceres::sqrt((difX*difX + difY*difY + difZ*difZ) / ceres::abs(timeStamp2 - timeStamp1 + (T)epsilon) + (T)DBL_EPSILON);

		return true;
	}

	static ceres::CostFunction* Create(double *ray1, double *ray2, double *center1, double *center2, double timeStamp1, double timeStamp2, double epsilon, double lamda)
	{
		return (new ceres::AutoDiffCostFunction< MotionPriorRaysCeres, 1, 1, 1>(new MotionPriorRaysCeres(ray1, ray2, center1, center2, timeStamp1, timeStamp2, epsilon, lamda)));
	}
	static ceres::CostFunction* CreateNumDif(double *ray1, double *ray2, double *center1, double *center2, double timeStamp1, double timeStamp2, double epsilon, double lamda)
	{
		return (new ceres::NumericDiffCostFunction< MotionPriorRaysCeres, ceres::CENTRAL, 1, 1, 1>(new MotionPriorRaysCeres(ray1, ray2, center1, center2, timeStamp1, timeStamp2, epsilon, lamda)));
	}
	double *rayDirect1, *rayDirect2, *center1, *center2, timeStamp1, timeStamp2, epsilon, lamda;
};

struct LeastMotionPriorRaysCeres {
	LeastMotionPriorRaysCeres(double *ray1, double *ray2, double *cc1, double *cc2, int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon) : frameID1(frameID1), frameID2(frameID2), subf1(subf1), subf2(subf2), ialpha1(ialpha1), ialpha2(ialpha2), Tscale(Tscale), epsilon(epsilon)
	{
		rayDirect1 = ray1, center1 = cc1;
		rayDirect2 = ray2, center2 = cc2;
	}
	template <typename T>	bool operator()(const T* const depth1, const T* const depth2, const T* const timeStamp1, const T* const timeStamp2, T* residuals) 	const
	{
		T X = (T)(depth1[0] * rayDirect1[0] + center1[0]), Y = (T)(depth1[0] * rayDirect1[1] + center1[1]), Z = (T)(depth1[0] * rayDirect1[2] + center1[2]);
		T X2 = (T)(depth2[0] * rayDirect2[0] + center2[0]), Y2 = (T)(depth2[0] * rayDirect2[1] + center2[1]), Z2 = (T)(depth2[0] * rayDirect2[2] + center2[2]);
		T difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

		T  t1 = (T)((timeStamp1[0] + (T)(subf1 + 1.0*frameID1)) * ialpha1*Tscale);
		T  t2 = (T)((timeStamp2[0] + (T)(subf2 + 1.0*frameID2)) * ialpha2*Tscale);

		residuals[0] = ceres::sqrt((difX*difX + difY*difY + difZ*difZ) / ceres::abs(t2 - t1 + (T)epsilon));

		return true;
	}
	template <typename T>	bool operator()(const T* const depth1, const T* const depth2, const T* const timeStamp, T* residuals) 	const
	{
		T X = (T)(depth1[0] * rayDirect1[0] + center1[0]), Y = (T)(depth1[0] * rayDirect1[1] + center1[1]), Z = (T)(depth1[0] * rayDirect1[2] + center1[2]);
		T X2 = (T)(depth2[0] * rayDirect2[0] + center2[0]), Y2 = (T)(depth2[0] * rayDirect2[1] + center2[1]), Z2 = (T)(depth2[0] * rayDirect2[2] + center2[2]);
		T difX = X2 - X, difY = Y2 - Y, difZ = Z2 - Z;

		T  t1 = (T)((timeStamp[0] + (T)(subf1 + 1.0*frameID1)) * ialpha1*Tscale);
		T  t2 = (T)((timeStamp[0] + (T)(subf2 + 1.0*frameID2)) * ialpha2*Tscale);

		residuals[0] = ceres::sqrt((difX*difX + difY*difY + difZ*difZ) / ceres::abs(t2 - t1 + (T)epsilon));

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double *ray1, double *ray2, double *center1, double *center2, int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorRaysCeres, 1, 1, 1, 1, 1>(new LeastMotionPriorRaysCeres(ray1, ray2, center1, center2, frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon)));
	}
	static ceres::CostFunction* CreateAutoDiffSame(double *ray1, double *ray2, double *center1, double *center2, int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorRaysCeres, 1, 1, 1, 1>(new LeastMotionPriorRaysCeres(ray1, ray2, center1, center2, frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon)));
	}
	static ceres::CostFunction* CreateNumerDiff(double *ray1, double *ray2, double *center1, double *center2, int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorRaysCeres, ceres::CENTRAL, 1, 1, 1, 1, 1>(new LeastMotionPriorRaysCeres(ray1, ray2, center1, center2, frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon)));
	}
	static ceres::CostFunction* CreateNumerDiffSame(double *ray1, double *ray2, double *center1, double *center2, int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorRaysCeres, ceres::CENTRAL, 1, 1, 1, 1>(new LeastMotionPriorRaysCeres(ray1, ray2, center1, center2, frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon)));
	}

	int frameID1, frameID2;
	double *rayDirect1, *rayDirect2, *center1, *center2;
	double subf1, subf2, ialpha1, ialpha2, Tscale, epsilon;
};
struct LeastMotionPriorCostCeres {
	LeastMotionPriorCostCeres(int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon, double lamda, int motionPriorPower) : frameID1(frameID1), frameID2(frameID2), subf1(subf1), subf2(subf2), ialpha1(ialpha1), ialpha2(ialpha2), Tscale(Tscale), epsilon(epsilon), lamda(lamda), motionPriorPower(motionPriorPower) {	}
	LeastMotionPriorCostCeres(int frameID1, int frameID2, double ialpha1, double ialpha2, double lamda) : frameID1(frameID1), frameID2(frameID2), ialpha1(ialpha1), ialpha2(ialpha2), lamda(lamda) {	}
	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const timeStamp1, const T* const timeStamp2, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)((timeStamp1[0] + (T)(subf1 + 1.0*frameID1)) * ialpha1*Tscale);
		T  t2 = (T)((timeStamp2[0] + (T)(subf2 + 1.0*frameID2)) * ialpha2*Tscale);

		T cost;
		if (motionPriorPower == 4)
			cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / ceres::abs(pow(t2 - t1, 3) + (T)epsilon); //mv4dt
		else if (motionPriorPower == 2)
			cost = (difX*difX + difY*difY + difZ*difZ) / ceres::abs(t2 - t1 + (T)epsilon);
		residuals[0] = ceres::sqrt(cost) * (T)lamda;

		return true;
	}
	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, const T* const timeStamp, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)((timeStamp[0] + (T)(subf1 + 1.0*frameID1)) * ialpha1*Tscale);
		T  t2 = (T)((timeStamp[0] + (T)(subf2 + 1.0*frameID2)) * ialpha2*Tscale);

		T cost;
		if (motionPriorPower == 4)
			cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / ceres::abs(pow(t2 - t1, 3) + (T)epsilon); //mv4dt
		else if (motionPriorPower == 2)
			cost = (difX*difX + difY*difY + difZ*difZ) / ceres::abs(t2 - t1 + (T)epsilon);
		residuals[0] = ceres::sqrt(cost) * (T)lamda;

		return true;
	}
	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];
		T  t1 = (T)(ialpha1 * frameID1);
		T  t2 = (T)(ialpha2 * frameID2);

		T cost = (difX*difX + difY*difY + difZ*difZ) / ceres::abs(t2 - t1);
		residuals[0] = ceres::sqrt(cost) * (T)lamda;

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(int frameID1, int frameID2, double ialpha1, double ialpha2, double lamda)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCostCeres, 1, 3, 3>(new LeastMotionPriorCostCeres(frameID1, frameID2, ialpha1, ialpha2, lamda)));
	}
	static ceres::CostFunction* CreateAutoDiff(int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCostCeres, 1, 3, 3, 1, 1>(new LeastMotionPriorCostCeres(frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateAutoDiffSame(int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCostCeres, 1, 3, 3, 1>(new LeastMotionPriorCostCeres(frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateNumerDiff(int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorCostCeres, ceres::CENTRAL, 1, 3, 3, 1, 1>(new LeastMotionPriorCostCeres(frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateNumerDiffSame(int frameID1, int frameID2, double subf1, double subf2, double ialpha1, double ialpha2, double Tscale, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorCostCeres, ceres::CENTRAL, 1, 3, 3, 1>(new LeastMotionPriorCostCeres(frameID1, frameID2, subf1, subf2, ialpha1, ialpha2, Tscale, epsilon, lamda, motionPriorPower)));
	}

	int frameID1, frameID2, motionPriorPower;
	double subf1, subf2, ialpha1, ialpha2, Tscale, epsilon, lamda;
};
struct LeastMotionPriorCost3DCeres {
	LeastMotionPriorCost3DCeres(double timeStamp1, double timeStamp2, double epsilon, double lamda, int motionPriorPower) : timeStamp1(timeStamp1), timeStamp2(timeStamp2), epsilon(epsilon), lamda(lamda), motionPriorPower(motionPriorPower) {}

	template <typename T>	bool operator()(const T* const xyz1, const T* const xyz2, T* residuals) 	const
	{
		T difX = xyz2[0] - xyz1[0], difY = xyz2[1] - xyz1[1], difZ = xyz2[2] - xyz1[2];

		T cost;
		if (motionPriorPower == 4)
			cost = pow(difX*difX + difY*difY + difZ*difZ, 2) / ceres::abs(pow(timeStamp2 - timeStamp1, 3) + (T)epsilon); //mv4dt
		else if (motionPriorPower == 2)
			cost = (difX*difX + difY*difY + difZ*difZ) / ceres::abs(timeStamp2 - timeStamp1 + (T)epsilon);
		residuals[0] = ceres::sqrt(cost) * (T)lamda;
		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double Stamp1, double Stamp2, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCost3DCeres, 1, 3, 3>(new LeastMotionPriorCost3DCeres(Stamp1, Stamp2, epsilon, lamda, motionPriorPower)));
	}
	static ceres::CostFunction* CreateNumerDiff(double Stamp1, double Stamp2, double epsilon, double lamda, int motionPriorPower)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorCost3DCeres, ceres::CENTRAL, 1, 3, 3>(new LeastMotionPriorCost3DCeres(Stamp1, Stamp2, epsilon, lamda, motionPriorPower)));
	}
	int motionPriorPower;
	double timeStamp1, timeStamp2, epsilon, lamda;
};
struct LeastMotionPriorCostCameraCeres {
	LeastMotionPriorCostCameraCeres(double timeStamp1, double timeStamp2, double sig_ivel) : timeStamp1(timeStamp1), timeStamp2(timeStamp2), sig_ivel(sig_ivel) {}

	template <typename T>	bool operator()(const T* const rt1, const T* const rt2, T* residuals) 	const
	{
		//T temp = (T)(sig_irot / ceres::sqrt(ceres::abs(timeStamp2 - timeStamp1)))*rotWeight;
		//for (int ii = 0; ii < 3; ii++)
		//	residuals[ii] = (rt2[ii] - rt1[ii])*temp;

		T Rt[9];
		ceres::AngleAxisToRotationMatrix(rt1, Rt);//this gives R' due to its column major format

		T C1[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C1[ii] += Rt[ii * 3 + jj] * rt1[jj + 3]; ////-C = R't;

		ceres::AngleAxisToRotationMatrix(rt2, Rt);//this gives R' due to its column major format

		T C2[3] = { (T)0.0, (T) 0.0, (T) 0.0 };
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				C2[ii] += Rt[ii * 3 + jj] * rt2[jj + 3]; ////-C = R't;

		T temp = (T)(sig_ivel / ceres::sqrt(ceres::abs(timeStamp2 - timeStamp1)));
		for (int ii = 0; ii < 3; ii++)
			residuals[ii] = (C2[ii] - C1[ii]) *temp; //(v/sig_v)^2*dt*weight = (dx/dt/sig_v)^2*dt*weight = (dx/sig_v)^2/dt*weight

		return true;
	}
	static ceres::CostFunction* CreateAutoDiff(double timeStamp1, double timeStamp2, double sig_ivel)
	{
		return (new ceres::AutoDiffCostFunction<LeastMotionPriorCostCameraCeres, 3, 6, 6>(new LeastMotionPriorCostCameraCeres(timeStamp1, timeStamp2, sig_ivel)));
	}
	static ceres::CostFunction* CreateNumerDiff(double timeStamp1, double timeStamp2, double sig_ivel)
	{
		return (new ceres::NumericDiffCostFunction<LeastMotionPriorCostCameraCeres, ceres::CENTRAL, 3, 6, 6>(new LeastMotionPriorCostCameraCeres(timeStamp1, timeStamp2, sig_ivel)));
	}
	double timeStamp1, timeStamp2, sig_ivel;
};
struct IdealGeoProjectionCeres {
	IdealGeoProjectionCeres(double *Pmat, Point2d pt2D, double inlamda)
	{
		P = Pmat, observed_x = pt2D.x, observed_y = pt2D.y;
		lamda = inlamda;
	}
	template <typename T>	bool operator()(const T* const pt3D, T* residuals) 	const
	{
		T numX = (T)P[0] * pt3D[0] + (T)P[1] * pt3D[1] + (T)P[2] * pt3D[2] + (T)P[3];
		T numY = (T)P[4] * pt3D[0] + (T)P[5] * pt3D[1] + (T)P[6] * pt3D[2] + (T)P[7];
		T denum = (T)P[8] * pt3D[0] + (T)P[9] * pt3D[1] + (T)P[10] * pt3D[2] + (T)P[11];

		residuals[0] = (T)(lamda)*(numX / denum - T(observed_x));
		residuals[1] = (T)(lamda)*(numY / denum - T(observed_y));

		return true;
	}
	static ceres::CostFunction* Create(double *Pmat, const Point2d pt2D, double lamda)
	{
		return (new ceres::AutoDiffCostFunction<IdealGeoProjectionCeres, 2, 3>(new IdealGeoProjectionCeres(Pmat, pt2D, lamda)));
	}
	double observed_x, observed_y, *P, lamda;
};
struct IdealAlgebraicReprojectionCeres {
	IdealAlgebraicReprojectionCeres(double *iQ, double *iU, double inlamda)
	{
		Q = iQ, U = iU;
		lamda = inlamda;
	}
	template <typename T>	bool operator()(const T* const pt3D, T* residuals) 	const
	{
		residuals[0] = (T)(lamda)*(Q[0] * pt3D[0] + Q[1] * pt3D[1] + Q[2] * pt3D[2] - U[0]);
		residuals[1] = (T)(lamda)*(Q[3] * pt3D[0] + Q[4] * pt3D[1] + Q[5] * pt3D[2] - U[1]);

		return true;
	}
	static ceres::CostFunction* Create(double *Qmat, double* Umat, double lamda)
	{
		return (new ceres::AutoDiffCostFunction<IdealAlgebraicReprojectionCeres, 2, 3>(new IdealAlgebraicReprojectionCeres(Qmat, Umat, lamda)));
	}

	double *Q, *U, lamda;
};
struct KineticEnergy {
	KineticEnergy() {}

	template <typename T>	bool operator()(const T* const xy, T* residuals) 	const
	{
		residuals[0] = ((T)1.0 - xy[0]) * ((T)1.0 - xy[0]) + (T)100.0 * (xy[1] - xy[0] * xy[0]) * (xy[1] - xy[0] * xy[0]);

		return true;
	}

	static ceres::CostFunction* Create()
	{
		return (new ceres::AutoDiffCostFunction<KineticEnergy, 1, 2>(new KineticEnergy()));
	}
};
struct PotentialEnergy {
	PotentialEnergy(double g) :g(g) {}

	template <typename T>	bool operator()(const T* const Y, T* residuals) 	const
	{
		residuals[0] = ((T)-g)*Y[0];

		return true;
	}

	static ceres::CostFunction* Create(double g)
	{
		return (new ceres::AutoDiffCostFunction<PotentialEnergy, 1, 1>(new PotentialEnergy(g)));
	}
	double g;
};
class LeastActionProblem : public ceres::FirstOrderFunction {
public:
	LeastActionProblem(double *AllQ, double *AllU, int *PtsPerTrack, int totalPts, int nCams, int nPperTracks, int npts, double lamdaI) :lamdaImg(lamdaI), totalPts(totalPts), nCams(nCams), nPperTracks(nPperTracks)
	{
		gravity = 9.88;
		lamdaImg = lamdaI;
		PointsPerTrack = PtsPerTrack;
		AllQmat = AllQ, AllUmat = AllU;
	}
	virtual ~LeastActionProblem() {}

	virtual bool MyEvaluate(const double* parameters, double* cost, double* gradient) const
	{
		/*//Kinetic energy depends on velocity computed at multiple points--> get splited
		for (int trackId = 0; trackId < npts; trackId++)
		{
		for (int pid = 0; pid < PointsPerTrack[trackId]; pid++)
		{
		ceres::CostFunction* cost_functionE = KineticEnergy::Create();
		cost_functionE->Evaluate(&parameters, &cost[0], NULL);
		if (gradient != NULL)
		cost_functionE->Evaluate(&parameters, &cost[0], &gradient);
		}
		}

		//Potential energy
		for (int trackId = 0; trackId < npts; trackId++)
		{
		for (int pid = 0; pid < PointsPerTrack[trackId]; pid++)
		{
		ceres::CostFunction* cost_functionP = PotentialEnergy::Create(gravity);
		cost_functionP->Evaluate(&parameters, &cost[0], NULL);

		if (gradient != NULL)
		cost_functionP->Evaluate(&parameters, &cost[0], &gradient);
		}
		}

		//Potential energy + Image constraint
		int currentPts = 0;
		for (int trackId = 0; trackId < npts; trackId++)
		{
		for (int pid = 0; pid < PointsPerTrack[trackId]; pid++)
		{
		ceres::CostFunction* cost_functionI = IdealAlgebraicReprojectionCeres::Create(&AllQmat[6 * pid], &AllUmat[2 * pid], lamdaImg);

		if (gradient != NULL)
		cost_functionI->Evaluate(&parameters, &cost[currentPts], &gradient);
		else
		cost_functionI->Evaluate(&parameters, &cost[currentPts], NULL);
		currentPts++;
		}
		}*/

		return true;
	}

	virtual int NumParameters() const
	{
		return totalPts + nCams;
	}

	int nCams, totalPts, nPperTracks, npts;
	int *PointsPerTrack;
	double *AllQmat, *AllUmat, lamdaImg, gravity;
};
double LeastActionError(double *xyz1, double *xyz2, double *timeStamp1, double *timeStamp2, double subframe1, double subframe2, int frameID1, int frameID2, double ialpha1, double ialpha2, double Tscale, double eps, int motionPriorPower);
double LeastActionError(double *xyz1, double *xyz2, double timeStamp1, double timeStamp2, double eps, int motionPriorPower);
double LeastActionError(double *xyz1, double *xyz2, double *xyz3, double *timeStamp1, double *timeStamp2, double *timeStamp3, int frameID1, int frameID2, int frameID3, double ialpha1, double ialpha2, double ialpha3, double Tscale, double eps, int motionPriorPower);

double Compute3DMotionPriorEnergy(vector<Point3d> &traj, vector<double>&Time, double eps, double g, int ApproxOrder);
void RecursiveUpdateCameraOffset(int *currentOffset, int BruteForceTimeWindow, int currentCam, int nCams);
void MotionPrior_ML_Weighting(vector<ImgPtEle> *PerCam_UV, int ntracks, int nCams);

double MotionPrior_Optim_SpatialStructure_Algebraic(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int npts,
	bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages = false, bool silent = true);
double MotionPrior_Optim_SpatialStructure_Geometric(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int npts,
	bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages = false, bool silent = true);
double MotionPrior_Optim_SpatialTemporal_Geometric(char *Path, vector<double*> &Allpt3D, vector<ImgPtEle> *PerCam_UV, vector<int> &PerPoint_nFrames, double *currentOffset, int npts
	, bool non_monotonicDescent, int nCams, int motionPriorPower, double Tscale, double eps, double lamda, double RealOverSfm, double *Cost, bool StillImages = false, bool silent = true);

double MotionPriorSyncBruteForce2DStereo(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int ntracks, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, double RealOverSfm, int motionPriorPower, int &totalPoints, bool silient = true);
int MotionPriorSyncBruteForce2DTriplet(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int ntracks, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, double RealOverSfm, int motionPriorPower);
int MotionPriorSyncBruteForce2DTripletParallel(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int ntracks, vector<double> &OffsetInfo, int off1, int off2, double frameSize, double lamda, double RealOverSfm, int motionPriorPower);
int IncrementalMotionPriorSyncDiscreteContinous2D(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int npts, vector<double> &OffsetInfo, int LowBound, int UpBound, double frameSize, double lamda, double RealOverSfm, double &CeresCost, bool RefineConsiderOrdering = true);
int IncrementalMotionPriorSyncDiscreteContinous2DParallel(char *Path, vector<int> &SelectedCams, int startFrame, int stopFrame, int npts, vector<double> &OffsetInfo, int SearchOffset, int cid, double lamda, double RealOverSfm);
int TrajectoryTriangulation(char *Path, vector<int> &SelectedCams, vector<double> &TimeStampInfoVector, int npts, int startFrame, int stopFrame, double lamda, double RealOverSfm, int motionPriorPower);

int EvaluateAllPairSTCost(char *Path, int nCams, int nTracks, int startFrame, int stopFrame, int SearchRange, double SearchStep, double lamda, double RealOverSfm, int motionPriorPower, double *InitialOffset);
int EvaluateAllPairSTCostParallel(char *Path, int camID1, int camID2, int nCams, int nTracks, int startFrame, int stopFrame, int SearchRange, double SearchStep, double lamda, double RealOverSfm, int motionPriorPower);
int DetermineCameraOrderingForGreedyDynamicSTBA(char *Path, char *PairwiseSyncFilename, int nCams, vector<int>&CameraOrder, vector<double> &OffsetInfo);

int Generate3DUncertaintyFromRandomSampling(char *Path, vector<int> SelectedCams, vector<double> OffsetInfo, int startFrame, int stopFrame, int ntracks, int startSample = 0, int nSamples = 100, int motionPriorPower = 2);

void ResamplingOf3DTrajectorySpline(vector<ImgPtEle> &Traj3D, bool non_monotonicDescent, double Break_Step, double Resample_Step, double lamda, bool silent = true);
void ResamplingOf3DTrajectoryDCT(vector<ImgPtEle> &Traj3D, int PriorOrder, bool non_monotonicDescent, double Resample_Step, double lamda1, double lamda2, bool silent = true);
#endif
