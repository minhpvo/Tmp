#include "FeatureEst.h"
#include "MatchingTracking.h"
#include "../ImgPro/ImagePro.h"
#include "../Ulti/MathUlti.h"

using namespace std;
using namespace cv;
using namespace Eigen;


void RootL1DescNorm(float *descf, uchar *descu, int numKeys, int featDim)
{
	//do rootL1 norm
	for (int ii = 0; ii < numKeys; ii++)
	{
		double l1 = 0.0;
		for (int jj = 0; jj < featDim; jj++)
			l1 += abs(descf[ii * featDim + jj]);
		for (int jj = 0; jj < featDim; jj++)
			descf[ii * featDim + jj] = sqrt(descf[ii * featDim + jj] / l1);
	}

	//Convert normalized floating point feature descriptor to unsigned byte representation by linear scaling from range [0, 0.5] to [0, 255]. Truncation to a maximum value of 0.5 is used to avoid precision loss and follows the common practice of representing SIFT vectors.
	for (int ii = 0; ii < numKeys; ii++)
	{
		for (int jj = 0; jj < featDim; jj++)
		{
			const float scaled_value = std::round(512.0f * descf[ii * featDim + jj]);
			descu[ii * featDim + jj] = TruncateCast<float, uint8_t>(scaled_value);
		}
	}
	return;
}
void RootL2DescNorm(float *descf, uchar *descu, int numKeys, int featDim)
{
	//do rootL1 norm
	for (int ii = 0; ii < numKeys; ii++)
	{
		double l2 = 0.0;
		for (int jj = 0; jj < featDim; jj++)
			l2 += pow(descf[ii * featDim + jj], 2);
		for (int jj = 0; jj < featDim; jj++)
			descf[ii * featDim + jj] = sqrt(descf[ii * featDim + jj] / l2);
	}

	//Convert normalized floating point feature descriptor to unsigned byte representation by linear scaling from range [0, 0.5] to [0, 255]. Truncation to a maximum value of 0.5 is used to avoid precision loss and follows the common practice of representing SIFT vectors.
	for (int ii = 0; ii < numKeys; ii++)
	{
		for (int jj = 0; jj < featDim; jj++)
		{
			const float scaled_value = std::round(512.0f * descf[ii * featDim + jj]);
			descu[ii * featDim + jj] = TruncateCast<float, uint8_t>(scaled_value);
		}
	}
	return;
}
static inline void transpose_descriptor(float *dst, float const *src)
{
	int const BO = 8;  // number of orientation bins 
	int const BP = 4;  // number of spatial bins     
	int i, j, t;

	for (j = 0; j < BP; ++j) {
		int jp = BP - 1 - j;
		for (i = 0; i < BP; ++i) {
			int o = BO * i + BP * BO * j;
			int op = BO * i + BP * BO * jp;
			dst[op] = src[o];
			for (t = 1; t < BO; ++t)
				dst[BO - t + op] = src[t + o];
		}
	}
}
//feature scale (VLSIFT + VLCOVDET) is the radius of the blob
int vl_DoG(Mat &Img, vector<Point3f> &DoG, int verbose)
{
	int npts = 0;
	if (Img.channels() == 3)
		cvtColor(Img, Img, CV_BGR2GRAY);

	//Take IplImage -> convert to SINGLE (float), (also flip the image?)
	int width = Img.cols, height = Img.rows;
	float* frame = new float[width*height];
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			frame[j*height + i] = (float)Img.data[i*width + j];

	// VL SIFT computation:
	float const *data = (float*)frame;
	int M = Img.rows, N = Img.cols;

	int                O = -1; //Octaves
	int                S = 3; //Levels
	int                o_min = 0;

	double             edge_thresh = -1;
	double             peak_thresh = 1;
	double             norm_thresh = -1;
	double             magnif = -1;
	double             window_size = -1;
	vl_bool            floaSiftDesciptors = 0, force_orientations = 0;


	{
		VlSiftFilt        *filt;
		vl_bool            first;
		double            *frames = 0;
		uchar              *descr = 0;
		int                reserved = 0, i, j, q;

		// create a filter to process the image
		filt = vl_sift_new(M, N, O, S, o_min);

		if (peak_thresh >= 0) vl_sift_set_peak_thresh(filt, peak_thresh);
		if (edge_thresh >= 0) vl_sift_set_edge_thresh(filt, edge_thresh);
		if (norm_thresh >= 0) vl_sift_set_norm_thresh(filt, norm_thresh);
		if (magnif >= 0) vl_sift_set_magnif(filt, magnif);
		if (window_size >= 0) vl_sift_set_window_size(filt, window_size);

		if (verbose)
		{
			printLOG("vl_sift: filter settings:\n");
			printLOG("vl_sift:   octaves      (O)      = %d\n", vl_sift_get_noctaves(filt));
			printLOG("vl_sift:   levels       (S)      = %d\n", vl_sift_get_nlevels(filt));
			printLOG("vl_sift:   first octave (o_min)  = %d\n", vl_sift_get_octave_first(filt));
			printLOG("vl_sift:   edge thresh           = %g\n", vl_sift_get_edge_thresh(filt));
			printLOG("vl_sift:   peak thresh           = %g\n", vl_sift_get_peak_thresh(filt));
			printLOG("vl_sift:   norm thresh           = %g\n", vl_sift_get_norm_thresh(filt));
			printLOG("vl_sift:   window size           = %g\n", vl_sift_get_window_size(filt));
			printLOG("vl_sift:   float descriptor      = %d\n", floaSiftDesciptors);
		}

		//Process each octave
		i = 0;
		first = 1;
		while (1)
		{
			int                   err;
			VlSiftKeypoint const *keys = 0;
			int                   nkeys = 0;

			if (verbose)
				printLOG("vl_sift: processing octave %d\n", vl_sift_get_octave_index(filt));

			// Calculate the GSS for the next octave .................... 
			if (first)
			{
				err = vl_sift_process_first_octave(filt, data);
				first = 0;
			}
			else
				err = vl_sift_process_next_octave(filt);

			if (err) break;

			if (verbose > 1)
				printLOG("vl_sift: GSS octave %d computed\n", vl_sift_get_octave_index(filt));

			//Run detector ............................................. 
			vl_sift_detect(filt);
			keys = vl_sift_get_keypoints(filt);
			nkeys = vl_sift_get_nkeypoints(filt);
			i = 0;

			if (verbose > 1)
				printLOG("vl_sift: detected %d (unoriented) keypoints\n", nkeys);

			for (; i < nkeys; ++i)
				DoG.push_back(Point3f(keys[i].y, keys[i].x, keys[i].sigma));// Save back with MATLAB conventions. Notice tha the input image was the transpose of the actual image.
		}

		if (verbose)
			printLOG("vl_sift: found %d keypoints\n", npts);

		vl_sift_delete(filt);
	}

	delete[]frame;
	return 0;
}
int VLSIFT(char *Fname, SiftFeature &SF, int &npts, int verbose, bool RootL1)
{
	npts = 0;
	//Take IplImage -> convert to SINGLE (float), (also flip the image?)
	if (IsFileExist(Fname) == 0)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}

	IplImage* image = cvLoadImage(Fname, 0);
	float* frame = new float[image->height*image->width];
	unsigned char* Ldata = (unsigned char *)image->imageData;
	for (int i = 0; i < image->height; i++)
		for (int j = 0; j < image->width; j++)
			frame[j*image->height + i] = (float)Ldata[i*image->widthStep + j];

	// VL SIFT computation:
	float const *data = (float*)frame;
	int  M = image->height, N = image->width;


	int                O = -1; //Octaves
	int                S = 3; //Levels
	int                o_min = 0;

	double             edge_thresh = -1;
	double             peak_thresh = 1;
	double             norm_thresh = -1;
	double             magnif = -1;
	double             window_size = -1;
	vl_bool            floaSiftDesciptors = 0, force_orientations = 0;


	{
		VlSiftFilt        *filt;
		vl_bool            first;
		double            *frames = 0;
		uchar              *descr = 0;
		int                reserved = 0, i, j, q;

		// create a filter to process the image
		filt = vl_sift_new(M, N, O, S, o_min);

		if (peak_thresh >= 0) vl_sift_set_peak_thresh(filt, peak_thresh);
		if (edge_thresh >= 0) vl_sift_set_edge_thresh(filt, edge_thresh);
		if (norm_thresh >= 0) vl_sift_set_norm_thresh(filt, norm_thresh);
		if (magnif >= 0) vl_sift_set_magnif(filt, magnif);
		if (window_size >= 0) vl_sift_set_window_size(filt, window_size);

		if (verbose)
		{
			printLOG("vl_sift: filter settings:\n");
			printLOG("vl_sift:   octaves      (O)      = %d\n", vl_sift_get_noctaves(filt));
			printLOG("vl_sift:   levels       (S)      = %d\n", vl_sift_get_nlevels(filt));
			printLOG("vl_sift:   first octave (o_min)  = %d\n", vl_sift_get_octave_first(filt));
			printLOG("vl_sift:   edge thresh           = %g\n", vl_sift_get_edge_thresh(filt));
			printLOG("vl_sift:   peak thresh           = %g\n", vl_sift_get_peak_thresh(filt));
			printLOG("vl_sift:   norm thresh           = %g\n", vl_sift_get_norm_thresh(filt));
			printLOG("vl_sift:   window size           = %g\n", vl_sift_get_window_size(filt));
			printLOG("vl_sift:   float descriptor      = %d\n", floaSiftDesciptors);
		}

		//Process each octave
		i = 0;
		first = 1;
		while (1)
		{
			int                   err;
			VlSiftKeypoint const *keys = 0;
			int                   nkeys = 0;

			if (verbose)
				printLOG("vl_sift: processing octave %d\n", vl_sift_get_octave_index(filt));

			// Calculate the GSS for the next octave .................... 
			if (first)
			{
				err = vl_sift_process_first_octave(filt, data);
				first = 0;
			}
			else
				err = vl_sift_process_next_octave(filt);

			if (err) break;

			if (verbose > 1)
				printLOG("vl_sift: GSS octave %d computed\n", vl_sift_get_octave_index(filt));

			//Run detector ............................................. 
			vl_sift_detect(filt);
			keys = vl_sift_get_keypoints(filt);
			nkeys = vl_sift_get_nkeypoints(filt);
			i = 0;

			if (verbose > 1)
				printLOG("vl_sift: detected %d (unoriented) keypoints\n", nkeys);

			// For each keypoint ........................................ 
			for (; i < nkeys; ++i)
			{
				double                angles[4];
				int                   nangles;
				VlSiftKeypoint const *k;

				k = keys + i;
				nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);

				// For each orientation ...................................
				for (q = 0; q < nangles; ++q)
				{
					float  buf[128], rbuf[128];

					// compute descriptor (if necessary)
					vl_sift_calc_keypoint_descriptor(filt, buf, k, angles[q]);
					transpose_descriptor(rbuf, buf);//The transpose is defined as the descriptor that one obtains from computing the normal descriptor on the transposed image.

					// make enough room for all these keypoints and more 
					if (reserved < npts + 1)
					{
						reserved += 2 * nkeys;
						frames = (double*)realloc(frames, 4 * sizeof(double)* reserved);
						descr = (uchar*)realloc(descr, 128 * sizeof(uchar)* reserved);
					}

					// Save back with MATLAB conventions. Notice tha the input image was the transpose of the actual image.
					frames[4 * npts + 0] = k->y;
					frames[4 * npts + 1] = k->x;
					frames[4 * npts + 2] = k->sigma;
					frames[4 * npts + 3] = VL_PI / 2 - angles[q];

					if (RootL1)
					{
						//Minh: root L1
						double l1 = 0;
						for (j = 0; j < 128; ++j)
							l1 += abs(rbuf[j]);

						for (j = 0; j < 128; ++j)
						{
							float x = 512.0F * sqrt(rbuf[j] / l1);
							x = (x < 255.0F) ? x : 255.0F;
							descr[128 * npts + j] = (uchar)x;
						}
					}
					else
					{
						for (j = 0; j < 128; ++j)
						{
							float x = 512.0F * rbuf[j];
							x = (x < 255.0F) ? x : 255.0F;
							descr[128 * npts + j] = (uchar)x;
						}
					}

					++npts;
				} // next orientation 
			} // next keypoint 
		} // next octave 

		if (verbose)
			printLOG("vl_sift: found %d keypoints\n", npts);

		// save variables:
		if (SF.CurrentMaxFeatures < npts)
		{
			SF.CurrentMaxFeatures = npts;
			SF.Kpts = (double*)realloc(SF.Kpts, npts * 4 * sizeof(double));
			SF.Desc = (uchar*)realloc(SF.Desc, npts * 128 * sizeof(uchar));
		}
		memcpy(SF.Kpts, frames, 4 * npts * sizeof(double));
		memcpy(SF.Desc, descr, 128 * npts * sizeof(uchar));

		vl_sift_delete(filt);
	}

	cvReleaseImage(&image);
	delete[]frame;

	return 0;
}
int VLSIFT(Mat &Img, SiftFeature &SF, int &npts, int verbose, bool RootL1)
{
	npts = 0;
	if (Img.channels() == 3)
		cvtColor(Img, Img, CV_BGR2GRAY);
	//Take IplImage -> convert to SINGLE (float), (also flip the image?)
	int width = Img.cols, height = Img.rows;
	float* frame = new float[width*height];
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			frame[j*height + i] = (float)Img.data[i*width + j];

	// VL SIFT computation:
	float const *data = (float*)frame;
	int M = Img.rows, N = Img.cols;

	int                O = -1; //Octaves
	int                S = 3; //Levels
	int                o_min = 0;

	double             edge_thresh = -1;
	double             peak_thresh = 1;
	double             norm_thresh = -1;
	double             magnif = -1;
	double             window_size = -1;
	vl_bool            floaSiftDesciptors = 0, force_orientations = 0;


	{
		VlSiftFilt        *filt;
		vl_bool            first;
		double            *frames = 0;
		uchar              *descr = 0;
		int                reserved = 0, i, j, q;

		// create a filter to process the image
		filt = vl_sift_new(M, N, O, S, o_min);

		if (peak_thresh >= 0) vl_sift_set_peak_thresh(filt, peak_thresh);
		if (edge_thresh >= 0) vl_sift_set_edge_thresh(filt, edge_thresh);
		if (norm_thresh >= 0) vl_sift_set_norm_thresh(filt, norm_thresh);
		if (magnif >= 0) vl_sift_set_magnif(filt, magnif);
		if (window_size >= 0) vl_sift_set_window_size(filt, window_size);

		if (verbose)
		{
			printLOG("vl_sift: filter settings:\n");
			printLOG("vl_sift:   octaves      (O)      = %d\n", vl_sift_get_noctaves(filt));
			printLOG("vl_sift:   levels       (S)      = %d\n", vl_sift_get_nlevels(filt));
			printLOG("vl_sift:   first octave (o_min)  = %d\n", vl_sift_get_octave_first(filt));
			printLOG("vl_sift:   edge thresh           = %g\n", vl_sift_get_edge_thresh(filt));
			printLOG("vl_sift:   peak thresh           = %g\n", vl_sift_get_peak_thresh(filt));
			printLOG("vl_sift:   norm thresh           = %g\n", vl_sift_get_norm_thresh(filt));
			printLOG("vl_sift:   window size           = %g\n", vl_sift_get_window_size(filt));
			printLOG("vl_sift:   float descriptor      = %d\n", floaSiftDesciptors);
		}

		//Process each octave
		i = 0;
		first = 1;
		while (1)
		{
			int                   err;
			VlSiftKeypoint const *keys = 0;
			int                   nkeys = 0;

			if (verbose)
				printLOG("vl_sift: processing octave %d\n", vl_sift_get_octave_index(filt));

			// Calculate the GSS for the next octave .................... 
			if (first)
			{
				err = vl_sift_process_first_octave(filt, data);
				first = 0;
			}
			else
				err = vl_sift_process_next_octave(filt);

			if (err) break;

			if (verbose > 1)
				printLOG("vl_sift: GSS octave %d computed\n", vl_sift_get_octave_index(filt));

			//Run detector ............................................. 
			vl_sift_detect(filt);
			keys = vl_sift_get_keypoints(filt);
			nkeys = vl_sift_get_nkeypoints(filt);
			i = 0;

			if (verbose > 1)
				printLOG("vl_sift: detected %d (unoriented) keypoints\n", nkeys);

			// For each keypoint ........................................ 
			for (; i < nkeys; ++i)
			{
				double                angles[4];
				int                   nangles;
				VlSiftKeypoint const *k;

				k = keys + i;
				nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);

				// For each orientation ...................................
				for (q = 0; q < nangles; ++q)
				{
					float  buf[128], rbuf[128];

					// compute descriptor (if necessary)
					vl_sift_calc_keypoint_descriptor(filt, buf, k, angles[q]);
					transpose_descriptor(rbuf, buf);//The transpose is defined as the descriptor that one obtains from computing the normal descriptor on the transposed image.

					// make enough room for all these keypoints and more 
					if (reserved < npts + 1)
					{
						reserved += 2 * nkeys;
						frames = (double*)realloc(frames, 4 * sizeof(double)* reserved);
						descr = (uchar*)realloc(descr, 128 * sizeof(uchar)* reserved);
					}

					// Save back with MATLAB conventions. Notice tha the input image was the transpose of the actual image.
					frames[4 * npts + 0] = k->y;
					frames[4 * npts + 1] = k->x;
					frames[4 * npts + 2] = k->sigma;
					frames[4 * npts + 3] = VL_PI / 2 - angles[q];

					if (RootL1)
					{
						//Minh: root L1
						double l1 = 0;
						for (j = 0; j < 128; ++j)
							l1 += abs(rbuf[j]);

						for (j = 0; j < 128; ++j)
						{
							float x = 512.0F * sqrt(rbuf[j] / l1);
							x = (x < 255.0F) ? x : 255.0F;
							descr[128 * npts + j] = (uchar)x;
						}
					}
					else
					{
						for (j = 0; j < 128; ++j)
						{
							float x = 512.0F * rbuf[j];
							x = (x < 255.0F) ? x : 255.0F;
							descr[128 * npts + j] = (uchar)x;
						}
					}

					++npts;
				} // next orientation 
			} // next keypoint 
		} // next octave 

		if (verbose)
			printLOG("vl_sift: found %d keypoints\n", npts);

		// save variables:
		if (SF.CurrentMaxFeatures < npts)
		{
			SF.CurrentMaxFeatures = npts;
			SF.Kpts = (double*)realloc(SF.Kpts, npts * 4 * sizeof(double));
			SF.Desc = (uchar*)realloc(SF.Desc, npts * 128 * sizeof(uchar));
		}
		memcpy(SF.Kpts, frames, 4 * npts * sizeof(double));
		memcpy(SF.Desc, descr, 128 * npts * sizeof(uchar));

		vl_sift_delete(filt);
	}

	delete[]frame;

	return 0;
}
int VLCOVDET(char *ImgName, CovFeature &CovF, int &npts, int verbose, bool RootL1)
{
	//npts: if >0 specifiies intial points whose descriptors are to be computed
	//Take IplImage -> convert to SINGLE (float), (also flip the image?)
	IplImage* cvimage = cvLoadImage(ImgName, 0);
	if (cvimage == NULL)
	{
		printLOG("Cannot load %s\n", ImgName);
		return 1;
	}

	float* image = new float[cvimage->height*cvimage->width];
	unsigned char* Ldata = (unsigned char *)cvimage->imageData;
	for (int i = 0; i < cvimage->height; i++)
		for (int j = 0; j < cvimage->width; j++)
			image[j*cvimage->height + i] = (float)Ldata[i*cvimage->widthStep + j];
	vl_size numRows = cvimage->height, numCols = cvimage->width;

	VlCovDetMethod method = CovF.method;
	vl_bool doubleImage = CovF.doubleImage;
	vl_index octaveResolution = CovF.octaveResolution;
	double edgeThreshold = CovF.edgeThreshold;
	double peakThreshold = CovF.peakThreshold;
	double lapPeakThreshold = CovF.lapPeakThreshold;

	vl_index patchResolution = CovF.patchResolution;
	double patchRelativeExtent = CovF.patchRelativeExtent;
	double patchRelativeSmoothing = CovF.patchRelativeSmoothing;
	double boundaryMargin = CovF.boundaryMargin;

	vl_size w = 2 * patchResolution + 1;
	float*patch = new float[w * w];
	float*patchXY = new float[2 * w * w];

	// Detector
	VlCovDet * covdet = vl_covdet_new(method);

	// set covdet parameters 
	vl_covdet_set_transposed(covdet, VL_TRUE);
	vl_covdet_set_first_octave(covdet, doubleImage ? -1 : 0);
	if (octaveResolution >= 0) vl_covdet_set_octave_resolution(covdet, octaveResolution);
	if (peakThreshold >= 0) vl_covdet_set_peak_threshold(covdet, peakThreshold);
	if (edgeThreshold >= 0) vl_covdet_set_edge_threshold(covdet, edgeThreshold);
	if (lapPeakThreshold >= 0) vl_covdet_set_laplacian_peak_threshold(covdet, lapPeakThreshold);

	if (verbose)
		printLOG("vl_covdet: doubling image: %s\n", VL_YESNO(vl_covdet_get_first_octave(covdet) < 0));

	// process the image 
	vl_covdet_put_image(covdet, image, numRows, numCols);

	//fill with frames : either run the detector of poure them in 
	if (npts > 0)
	{
		vl_index k;

		if (verbose)
			printLOG("vl_covdet: sourcing %d frames\n", npts);


		for (k = 0; k < npts; ++k)
		{
			VlCovDetFeature feature;
			feature.peakScore = VL_INFINITY_F;
			feature.edgeScore = 1.0;
			feature.frame.x = (float)CovF.Kpts[6 * k + 1];
			feature.frame.y = (float)CovF.Kpts[6 * k];

			double a11 = 1.0, a21 = 0.0, a12 = 0.0, a22 = 1.0;

			feature.frame.a11 = (float)a22;
			feature.frame.a21 = (float)a12;
			feature.frame.a12 = (float)a21;
			feature.frame.a22 = (float)a11;
			vl_covdet_append_feature(covdet, &feature);
		}
	}
	else
	{
		if (verbose)
		{
			printLOG("vl_covdet: detector: %s\n", vl_enumeration_get_by_value(vlCovdetMethods, method)->name);
			printLOG("vl_covdet: peak threshold: %g, edge threshold: %g\n", vl_covdet_get_peak_threshold(covdet), vl_covdet_get_edge_threshold(covdet));
		}

		vl_covdet_detect(covdet);

		if (verbose)
		{
			printLOG("vl_covdet: %d features suppressed as duplicate (threshold: %g)\n", vl_covdet_get_num_non_extrema_suppressed(covdet), vl_covdet_get_non_extrema_suppression_threshold(covdet));

			switch (method)
			{
			case VL_COVDET_METHOD_HARRIS_LAPLACE:
			case VL_COVDET_METHOD_HESSIAN_LAPLACE:
			{
				vl_size numScales;
				vl_size const * numFeaturesPerScale = vl_covdet_get_laplacian_scales_statistics(covdet, &numScales);
				printLOG("vl_covdet: Laplacian scales:");
				for (vl_index i = 0; i <= (signed)numScales; ++i)
					printLOG("%d with %d scales;", numFeaturesPerScale[i], i);
				printLOG("\n");
			}
			break;
			default:
				break;
			}
			printLOG("vl_covdet: detected %d features\n", vl_covdet_get_num_features(covdet));
		}

		if (boundaryMargin > 0)
		{
			vl_covdet_drop_features_outside(covdet, boundaryMargin);
			if (verbose)
				printLOG("vl_covdet: kept %d inside the boundary margin (%g)\n", vl_covdet_get_num_features(covdet), boundaryMargin);
		}
	}

	// affine adaptation if needed
	if (CovF.Affine)
	{
		if (verbose)
			printLOG("vl_covdet: estimating affine shape for %d features\n", vl_covdet_get_num_features(covdet));
		vl_covdet_extract_affine_shape(covdet);
		if (verbose)
			printLOG("vl_covdet: %d features passed affine adaptation\n", vl_covdet_get_num_features(covdet));
	}

	// orientation estimation if needed 
	if (CovF.Orientation == 1)
	{
		vl_size numFeaturesBefore = vl_covdet_get_num_features(covdet);
		vl_covdet_extract_orientations(covdet);
		vl_size numFeaturesAfter = vl_covdet_get_num_features(covdet);
		if (verbose && numFeaturesAfter > numFeaturesBefore)
			printLOG("vl_covdet: %d duplicate features were created due to ambiguous orientation detection (%d total)\n", numFeaturesAfter - numFeaturesBefore, numFeaturesAfter);
	}

	//Compute SIFT desc
	vl_size numFeatures = vl_covdet_get_num_features(covdet);
	VlCovDetFeature const * feature = (VlCovDetFeature*)vl_covdet_get_features(covdet);


	VlSiftFilt * sift = vl_sift_new(16, 16, 1, 3, 0);
	vl_size dimension = 128;
	vl_size patchSide = 2 * patchResolution + 1;
	double patchStep = (double)patchRelativeExtent / patchResolution;
	float desc[128], tempDesc[128];
	if (verbose)
		printLOG("vl_covdet: descriptors: type=sift, resolution=%d, extent=%g, smoothing=%g\n", patchResolution, patchRelativeExtent, patchRelativeSmoothing);

	if (numFeatures > CovF.CurrentMaxFeatures)
	{
		CovF.CurrentMaxFeatures = (int)numFeatures;
		CovF.Kpts = (double*)realloc(CovF.Kpts, numFeatures * 6 * sizeof(double));
		CovF.Desc = (uchar*)realloc(CovF.Desc, numFeatures * 128 * sizeof(uchar));
	}

	vl_sift_set_magnif(sift, 3.0);
	for (vl_index i = 0; i < (signed)numFeatures; ++i)
	{
		vl_covdet_extract_patch_for_frame(covdet, patch, patchResolution, patchRelativeExtent, patchRelativeSmoothing, feature[i].frame);
		vl_imgradient_polar_f(patchXY, patchXY + 1, 2, 2 * patchSide, patch, patchSide, patchSide, patchSide);

		//Note: the patch is transposed, so that x and y are swapped.However, if NBO is not divisible by 4, then the configuration of the SIFT orientations is not symmetric by rotations of pi/2.
		//Hence the only option is to rotate the descriptor further by an angle we need to compute the descriptor rotated by an additional pi/2	angle. In this manner, x coincides and y is flipped.
		vl_sift_calc_raw_descriptor(sift, patchXY, tempDesc, (int)patchSide, (int)patchSide, (double)(patchSide - 1) / 2, (double)(patchSide - 1) / 2, (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep, VL_PI / 2);

		transpose_descriptor(desc, tempDesc);

		if (RootL1)
		{
			//Minh: root L1
			double l1 = 0;
			for (int j = 0; j < 128; ++j)
				l1 += abs(desc[j]);

			for (int j = 0; j < 128; ++j)
			{
				float x = 512.0F * sqrt(desc[j] / l1);
				x = (x < 255.0F) ? x : 255.0F;
				CovF.Desc[128 * npts + j] = (uchar)x;
			}
		}
		else
		{
			for (int j = 0; j < 128; ++j)
			{
				float x = 512.0F * desc[j];
				x = (x < 255.0F) ? x : 255.0F;
				CovF.Desc[128 * npts + j] = (uchar)x;
			}
		}
	}
	vl_sift_delete(sift);

	// save the transposed frame
	for (vl_index i = 0; i < (signed)numFeatures; ++i)
	{
		CovF.Kpts[6 * i] = feature[i].frame.y, CovF.Kpts[6 * i + 1] = feature[i].frame.x;
		CovF.Kpts[6 * i + 2] = feature[i].frame.a22, CovF.Kpts[6 * i + 3] = feature[i].frame.a12, CovF.Kpts[6 * i + 4] = feature[i].frame.a21, CovF.Kpts[6 * i + 5] = feature[i].frame.a11;
	}

	npts = (signed)numFeatures;

	vl_covdet_delete(covdet);
	delete[]image, delete[]patch, delete[]patchXY;
	cvReleaseImage(&cvimage);

	return 0;
}

int BuildImgPyr(char *ImgName, ImgPyr &Pyrad, int nOtaves, int nPerOctaves, bool color, int interpAlgo, double sigma)
{
	int width, height, nw, nh, nchannels = color ? 3 : 1;

	Mat view = imread(ImgName, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << ImgName << endl;
		return false;
	}
	width = view.cols, height = view.rows;

	unsigned char*	Img = new unsigned char[width*height*nchannels];
	int length = width * height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * length] = view.data[nchannels*ii + jj * nchannels*width + kk];
	}
	Pyrad.factor.push_back(1.0);
	Pyrad.wh.push_back(Point2i(width, height));
	Pyrad.ImgPyrImg.push_back(Img);

	int nlayers = nOtaves * nPerOctaves, count = 0;
	double scalePerOctave = pow(2.0, 1.0 / nPerOctaves), factor;
	for (int jj = 1; jj <= nOtaves; jj++)
	{
		for (int ii = nPerOctaves - 1; ii >= 0; ii--)
		{
			factor = pow(scalePerOctave, ii) / pow(2.0, jj);
			nw = (int)(factor*width), nh = (int)(factor*height);

			unsigned char *smallImg = new uchar[nw*nh*nchannels];
			ResizeImage(Pyrad.ImgPyrImg[0], smallImg, width, height, nchannels, factor, sigma / factor, interpAlgo);

			Pyrad.factor.push_back(factor);
			Pyrad.wh.push_back(Point2i(nw, nh));
			Pyrad.ImgPyrImg.push_back(smallImg);

			//sprintf(Fname, "C:/temp/_L%.4d.png", count+1);
			//SaveDataToImage(Fname, Pyrad.ImgPyrImg[count+1], nw, nh, nchannels);
			count++;
		}
	}

	Pyrad.nscales = count + 1;
	return 0;
}

void BucketGoodFeaturesToTrack(Mat Img, vector<Point2f> &Corners, int nImagePartitions, int maxCorners, double qualityLevel, double minDistance, int blockSize, bool useHarrisDetector, double harrisK)
{
	int width = Img.cols, height = Img.rows;
	Mat Img2;

	Corners.reserve(maxCorners);

	int partionSize = max(width, height) / nImagePartitions;
	for (int jj = 0; jj < nImagePartitions; jj++)
	{
		for (int ii = 0; ii < nImagePartitions; ii++)
		{
			vector<Point2f> RegionCorners; RegionCorners.reserve(maxCorners);

			Mat mask(height, width, CV_8UC1, Scalar(0, 0, 0));
			for (int kk = jj * partionSize; kk < (jj + 1)*partionSize; kk++)
			{
				for (int ll = ii * partionSize; ll < (ii + 1)*partionSize; ll++)
				{
					if (kk > height - 1 || ll > width - 1)
						continue;
					mask.data[ll + kk * width] = 255;
				}
			}
			goodFeaturesToTrack(Img, RegionCorners, maxCorners, qualityLevel, minDistance* max(1, width / 1920), mask, blockSize* max(1, width / 1920), useHarrisDetector, harrisK);
			/*cvtColor(Img, Img2, CV_GRAY2BGR);
			for (int kk = 0; kk < (int)uvRef.size(); kk++)
			circle(Img2, uvRef[kk], 5, Scalar(0, 255, 0), 2, 8, 0);
			namedWindow("X", WINDOW_NORMAL);
			imshow("X", Img2); waitKey(0);*/

			for (int kk = 0; kk < (int)RegionCorners.size(); kk++)
				Corners.push_back(RegionCorners[kk]);
		}
	}

	return;
}

void LaplacianOfGaussian(double *LOG, int sigma)
{
	int n = ceil(sigma * 3), Size = 2 * n + 1;
	double ii2, jj2, Twosigma2 = 2.0*sigma*sigma, sigma4 = pow(sigma, 4);
	for (int jj = -n; jj <= n; jj++)
	{
		for (int ii = -n; ii <= n; ii++)
		{
			ii2 = ii * ii, jj2 = jj * jj;
			LOG[(ii + n) + (jj + n)*Size] = (ii2 + jj2 - Twosigma2) / sigma4 * exp(-(ii2 + jj2) / Twosigma2);
		}
	}

	return;
}
void LaplacianOfGaussian(double *LOG, int sigma, int PatternSize)
{
	int n = ceil(sigma * 3), Size = 2 * n + 1, hsubset = PatternSize / 2;
	double ii2, jj2, Twosigma2 = 2.0*sigma*sigma, sigma4 = pow(sigma, 4);
	for (int jj = -hsubset; jj <= hsubset; jj++)
	{
		for (int ii = -hsubset; ii <= hsubset; ii++)
		{
			ii2 = ii * ii, jj2 = jj * jj;
			LOG[(ii + hsubset) + (jj + hsubset)*PatternSize] = (ii2 + jj2 - Twosigma2) / sigma4 * exp(-(ii2 + jj2) / Twosigma2);
		}
	}

	return;
}
void Gaussian(double *G, int sigma, int PatternSize)
{
	int ii2, jj2, size = MyFtoI(6.0*sigma + 1) / 2 * 2 + 1;
	double sigma2 = 2.0*sigma*sigma, sqrt2Pi_sigma = sqrt(2.0*Pi)*sigma;
	int hsubset = PatternSize / 2;

	for (int jj = -hsubset; jj <= hsubset; jj++)
	{
		for (int ii = -hsubset; ii <= hsubset; ii++)
		{
			ii2 = ii * ii, jj2 = jj * jj;
			G[(ii + hsubset) + (jj + hsubset)*PatternSize] = exp(-(ii2 + jj2) / sigma2) / sqrt2Pi_sigma;
		}
	}

	return;
}
void synthesize_concentric_circles_mask(double *ring_mask_smooth, int *pattern_bi_graylevel, int pattern_size, double sigma, double scale, double *ring_info, int flag, int num_ring_edge)
{
	//ring_mask's size:[Pattern_size,Pattern_size] 
	int ii, jj, kk;
	int es_cen_x = pattern_size / 2;
	int es_cen_y = pattern_size / 2;
	int dark = pattern_bi_graylevel[0];
	int bright = pattern_bi_graylevel[1];
	int hb = 1;	// half-band

	double t = scale / 2.0;
	double r0[10], r1[9];

	for (ii = 0; ii <= num_ring_edge; ii++)
	{
		r0[ii] = ring_info[ii] * t;
		if (ii >= 1)
			r1[ii - 1] = ring_info[ii] * t;
	}

	char *ring_mask = new char[pattern_size*pattern_size];
	for (ii = 0; ii < pattern_size*pattern_size; ii++)
		ring_mask[ii] = (char)bright;

	int g1, g2;
	if (flag == 0)
	{
		for (jj = 0; jj < pattern_size; jj++)
		{
			for (ii = 0; ii < pattern_size; ii++)
			{
				t = sqrt((double)((ii - es_cen_x)*(ii - es_cen_x) + (jj - es_cen_y)*(jj - es_cen_y)));
				for (kk = 0; kk <= num_ring_edge; kk++)
				{
					if (kk % 2 == 0)
					{
						g1 = dark;
						g2 = bright;
					}
					else
					{
						g1 = bright;
						g2 = dark;
					}

					if (kk <= num_ring_edge - 1 && t <= r0[kk] - hb && t > r0[kk + 1] + hb)
						ring_mask[ii + jj * pattern_size] = (char)g1;
					else if (t <= r0[kk] + hb && t > r0[kk] - hb)
						ring_mask[ii + jj * pattern_size] = (char)(MyFtoI(g2*(t + hb - r0[kk]) / (2.0*hb) + g1 * (r0[kk] + hb - t) / (2.0*hb)));
				}
				if (t <= r0[num_ring_edge] - hb)
					ring_mask[ii + jj * pattern_size] = (char)(num_ring_edge % 2 == 0 ? dark : bright);
			}
		}
	}
	else
	{
		for (jj = 0; jj < pattern_size; jj++)
		{
			for (ii = 0; ii < pattern_size; ii++)
			{
				t = sqrt((double)((ii - es_cen_x)*(ii - es_cen_x) + (jj - es_cen_y)*(jj - es_cen_y)));
				for (kk = 0; kk < num_ring_edge; kk++)
				{
					if (kk % 2 == 0)
					{
						g1 = dark;
						g2 = bright;
					}
					else
					{
						g1 = bright;
						g2 = dark;
					}

					if (kk<num_ring_edge - 1 && t <= r1[kk] - hb && t > r1[kk + 1] + hb)
						ring_mask[ii + jj * pattern_size] = (char)g1;
					else if (t <= r1[kk] + hb && t > r1[kk] - hb)
						ring_mask[ii + jj * pattern_size] = (char)(MyFtoI(g2*(t + hb - r1[kk]) / (2.0*hb) + g1 * (r1[kk] + hb - t) / (2.0*hb)));
				}
				if (t <= r1[num_ring_edge - 1] - hb)
					ring_mask[ii + jj * pattern_size] = (char)(num_ring_edge % 2 == 0 ? bright : dark);
			}
		}
	}

	//SaveDataToGreyImage(ring_mask, pattern_size, pattern_size, TempPathName+(flag==0?_T("syn_ring_1_0.png"):_T("syn_ring_2_0.png")));

	// Gaussian smooth
	Gaussian_smooth(ring_mask, ring_mask_smooth, pattern_size, pattern_size, pattern_bi_graylevel[1], sigma);
	delete[]ring_mask;
	return;
}
void synthesize_square_mask(double *square_mask_smooth, int *pattern_bi_graylevel, int Pattern_size, double sigma, int flag, bool OpenMP)
{
	int ii, jj;
	int es_con_x = Pattern_size / 2;
	int es_con_y = Pattern_size / 2;
	char dark = (char)pattern_bi_graylevel[0];
	char bright = (char)pattern_bi_graylevel[1];
	char mid = (char)((pattern_bi_graylevel[0] + pattern_bi_graylevel[1]) / 2);

	char *square_mask = new char[Pattern_size*Pattern_size];

	for (jj = 0; jj < Pattern_size; jj++)
	{
		for (ii = 0; ii < Pattern_size; ii++)
		{
			if ((ii < es_con_x && jj < es_con_y) || (ii > es_con_x && jj > es_con_y))
				square_mask[ii + jj * Pattern_size] = (flag == 0 ? bright : dark);
			else if (ii == es_con_x || jj == es_con_y)
				square_mask[ii + jj * Pattern_size] = mid;
			else
				square_mask[ii + jj * Pattern_size] = (flag == 0 ? dark : bright);
		}
	}
	//SaveDataToImage("C:/temp/t.png", square_mask, Pattern_size, Pattern_size, 1);

	// Gaussian smooth
	Gaussian_smooth(square_mask, square_mask_smooth, Pattern_size, Pattern_size, pattern_bi_graylevel[1], sigma);

	for (jj = 0; jj < Pattern_size; jj++)
		for (ii = 0; ii < Pattern_size; ii++)
			square_mask[ii + jj * Pattern_size] = (char)MyFtoI((square_mask_smooth[ii + jj * Pattern_size]));

	//SaveDataToImage("C:/temp/st.png", square_mask, Pattern_size, Pattern_size, 1);
	delete[]square_mask;

	return;
}
void synthesize_pattern(int pattern, double *Pattern, int *Pattern_size, int *pattern_bi_graylevel, int *Subset, double *ctrl_pts_info, double scale, int board_width, int board_height, double sigma, int num_ring_edge, int num_target = 0, bool OpenMP = false)
{
	int i, addon = MyFtoI(ctrl_pts_info[0] * scale*0.2) / 4 * 4;
	if (pattern == 0)//CHECKER
	{
		Pattern_size[0] = MyFtoI(ctrl_pts_info[0] * scale) / 4 * 4 + addon;
		Subset[0] = MyFtoI(ctrl_pts_info[0] * scale) / 4 * 2 + addon / 4;
		if (Subset[0] > 20)
			Subset[0] = 20;

		Pattern_size[1] = Pattern_size[0];
		Subset[1] = Subset[0];
	}
	else if (pattern == 1)//RING
	{
		for (i = 0; i <= 1; i++)
		{
			Pattern_size[i] = MyFtoI(ctrl_pts_info[i] * scale) / 4 * 4 + addon;
			Subset[i] = MyFtoI(ctrl_pts_info[i] * scale) / 4 * 2 + addon / 4;
		}
	}

	if (pattern == 0)
		for (i = 0; i <= 1; i++)
			synthesize_square_mask(Pattern + Pattern_size[0] * Pattern_size[0] * i, pattern_bi_graylevel, Pattern_size[0], sigma, i, OpenMP);
	else if (pattern == 1)
		for (i = 0; i <= 1; i++)
			synthesize_concentric_circles_mask(Pattern + Pattern_size[0] * Pattern_size[0] * i, pattern_bi_graylevel, Pattern_size[i], sigma, scale, ctrl_pts_info, i, num_ring_edge);

	return;
}
void DetectBlobCorrelation(char *ImgName, vector<KeyPoint> &kpts, int nOctaveLayers, int nScalePerOctave, double sigma, int PatternSize, int NMS_BW, double thresh)
{
	int jump = 1, numPatterns = 1;
	char Fname[512];

	ImgPyr imgpyrad;
	double starttime = omp_get_wtime();
	BuildImgPyr(ImgName, imgpyrad, nOctaveLayers, nScalePerOctave, false, 1, 1.0);
	for (int ii = 0; ii < imgpyrad.ImgPyrImg.size(); ii++)
	{
		sprintf(Fname, "C:/temp/L%.4d.png", ii);
		SaveDataToImage(Fname, imgpyrad.ImgPyrImg[ii], imgpyrad.wh[ii].x, imgpyrad.wh[ii].y, 1);
	}
	printLOG("Building Image pyramid: %.fs\n", omp_get_wtime() - starttime);

	//build template
	int hsubset = PatternSize / 2, PatternLength = PatternSize * PatternSize;
	int IntensityProfile[] = { 10, 240 };
	double RingInfo[1] = { 0.9 };
	double *maskSmooth = new double[PatternLength];
	synthesize_concentric_circles_mask(maskSmooth, IntensityProfile, PatternSize, 1, PatternSize, RingInfo, 0, 1);
	//SaveDataToImage("C:/temp/mask.png", maskSmooth, PatternSize, PatternSize);

	Mat tpl = Mat::zeros(PatternSize, PatternSize, CV_32F);
	for (int ii = 0; ii < PatternLength; ii++)
		tpl.at<float>(ii) = maskSmooth[ii];

	int width = imgpyrad.wh[0].x, height = imgpyrad.wh[0].y;
	float *response = new float[width*height];

	for (int scaleID = 0; scaleID < imgpyrad.ImgPyrImg.size(); scaleID++)
	{
		starttime = omp_get_wtime();
		printLOG("Layer %d ....", scaleID);
		int width = imgpyrad.wh[scaleID].x, height = imgpyrad.wh[scaleID].y;

		Mat ref = Mat::zeros(height, width, CV_32F);
		for (int ii = 0; ii < width*height; ii++)
			ref.at<float>(ii) = (float)(int)imgpyrad.ImgPyrImg[scaleID][ii];

		Mat dst;
		cv::matchTemplate(ref, tpl, dst, CV_TM_CCORR_NORMED);
		for (int ii = 0; ii < dst.rows*dst.cols; ii++)
			response[ii] = dst.at<float>(ii);

		//sprintf(Fname, "C:/temp/x_%.4d.dat", scaleID);
		//WriteGridBinary(Fname, response, dst.cols, dst.rows);

		//Non-max suppression:
		bool breakflag;
		int ScoreW = dst.cols, ScoreH = dst.rows;
		for (int jj = hsubset; jj < ScoreH - hsubset; jj += jump)
		{
			for (int ii = hsubset; ii < ScoreW - hsubset; ii += jump)
			{
				breakflag = false;
				if (response[ii + jj * ScoreW] < thresh)
					response[ii + jj * ScoreW] = 0.0;
				else
				{
					for (int j = -NMS_BW; j <= NMS_BW && !breakflag; j += jump)
					{
						for (int i = -NMS_BW; i <= NMS_BW && !breakflag; i += jump)
						{
							if (i == 0 && j == 0)
								continue;
							if (ii + i< 0 || ii + i>ScoreW || jj + j < 0 || jj + j>ScoreH)
								continue;
							if (response[ii + jj * ScoreW] < response[(ii + i) + (jj + j)*ScoreW])
							{
								response[ii + jj * ScoreW] = 0.0;
								breakflag = true;
								break;
							}
						}
					}
				}
			}
		}
		//WriteGridBinary("C:/temp/x.dat", response, dst.cols, dst.rows);

		for (int jj = hsubset; jj < ScoreH - hsubset; jj += jump)
		{
			for (int ii = hsubset; ii < ScoreW - hsubset; ii += jump)
			{
				if (response[ii + jj * ScoreW] > thresh)
				{
					KeyPoint kpt;
					kpt.pt.x = (1.0*ii + PatternSize / 2) / imgpyrad.factor[scaleID];
					kpt.pt.y = (1.0*jj + PatternSize / 2) / imgpyrad.factor[scaleID];
					kpt.size = 1.0*PatternSize * imgpyrad.factor[scaleID];
					kpts.push_back(kpt);
				}
			}
		}
		printLOG(" %.2fs\n", omp_get_wtime() - starttime);
	}
	printLOG(" %.2fs\n", omp_get_wtime() - starttime);
	delete[]maskSmooth, delete[]response;

	width = imgpyrad.wh[0].x, height = imgpyrad.wh[0].y;
	Mat ref_gray = Mat::zeros(height, width, CV_8UC1);
	for (int ii = 0; ii < width*height; ii++)
		ref_gray.data[ii] = imgpyrad.ImgPyrImg[0][ii];
	cv::cvtColor(ref_gray, ref_gray, CV_GRAY2BGR);
	for (int ii = 0; ii < kpts.size(); ii++)
	{
		KeyPoint kpt = kpts[ii];
		int startX = kpt.pt.x - kpt.size / 2, startY = kpt.pt.y - kpt.size / 2;
		int stopX = kpt.pt.x + kpt.size / 2, stopY = kpt.pt.y + kpt.size / 2;
		cv::rectangle(ref_gray, Point2i(startX, startY), cv::Point(stopX, stopY), CV_RGB(0, 255, 0), 2);
	}
	namedWindow("result", CV_WINDOW_NORMAL);
	imshow("result", ref_gray); waitKey();


	return;
}

void DetectCheckerCornersCorrelation(double *img, int width, int height, int nchannels, Point2d *Checker, int &npts, vector<double> PatternAngles, int hsubset, int search_area, double thresh)
{
	int i, j, ii, jj, kk, jump = 2, nMaxCorners = npts, numPatterns = (int)PatternAngles.size();

	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize * PatternSize; //Note that the pattern size is deliberately make bigger than the subset because small size give very blurry checkercorner
	double *maskSmooth = new double[PatternLength*numPatterns];

	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (ii = 1; ii < PatternAngles.size(); ii++)
	{
		double c = cos(PatternAngles[ii] * 3.14159265359 / 180), s = sin(PatternAngles[ii] * 3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3);
		mat_mul(H2, H1, temp, 3, 3, 3);
		mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + ii * PatternLength, PatternSize, PatternSize, maskSmooth, PatternSize, PatternSize, trans, 1, 1, NULL);
		//char Fname[512];  sprintf(Fname, "C:/temp/rS_%.4d.png", ii);
		//SaveDataToImage(Fname, maskSmooth + ii*PatternLength, PatternSize, PatternSize, 1);
	}

	double *Cornerness = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness[ii] = 0.0;

	double zncc;
	Point2i POI;
	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	for (jj = hsubset + search_area + 1; jj < height - hsubset - search_area - 1; jj += jump)
	{
		for (ii = hsubset + search_area + 1; ii < width - hsubset - search_area - 1; ii += jump)
		{
			for (kk = 0; kk < numPatterns; kk++)
			{
				POI.x = ii, POI.y = jj;
				zncc = abs(TMatchingSuperCoarse(maskSmooth + kk * PatternLength, PatternSize, hsubset, img, width, height, nchannels, POI, search_area, thresh, T));
				Cornerness[ii + jj * width] = max(zncc, Cornerness[ii + jj * width]);
			}
		}
	}

	double *Cornerness2 = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness2[ii] = Cornerness[ii];
	//WriteGridBinary("C:/temp/cornerness.dat", Cornerness, width, height);

	//Non-max suppression
	bool breakflag;
	for (jj = hsubset + search_area + 1; jj < height - hsubset - search_area - 1; jj += jump)
	{
		for (ii = hsubset + search_area + 1; ii < width - hsubset - search_area - 1; ii += jump)
		{
			breakflag = false;
			if (Cornerness[ii + jj * width] < thresh)
			{
				Cornerness[ii + jj * width] = 0.0;
				Cornerness2[ii + jj * width] = 0.0;
			}
			else
			{
				for (j = -jump; j <= jump; j += jump)
				{
					for (i = -jump; i <= jump; i += jump)
					{
						if (Cornerness[ii + jj * width] < Cornerness[ii + i + (jj + j)*width] - 0.001) //avoid comparing with itself
						{
							Cornerness2[ii + jj * width] = 0.0;
							breakflag = true;
							break;
						}
					}
				}
			}
			if (breakflag == true)
				break;
		}
	}

	npts = 0;
	for (jj = hsubset + search_area + 1; jj < height - hsubset - search_area - 1; jj += jump)
	{
		for (ii = hsubset + search_area + 1; ii < width - hsubset - search_area - 1; ii += jump)
		{
			if (Cornerness2[ii + jj * width] > thresh)
			{
				Checker[npts].x = ii;
				Checker[npts].y = jj;
				npts++;
			}
			if (npts > nMaxCorners)
				break;
		}
	}

	delete[]maskSmooth;
	delete[]Cornerness;
	delete[]Cornerness2;

	return;
}
void RefineCheckerCorners(double *Para, int width, int height, int nchannels, Point2d *Checker, Point2d *Fcorners, int *FStype, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo)
{
	int ii, jj, kk, boundary = hsubset2 + 2;
	int numPatterns = (int)PatternAngles.size();
	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize * PatternSize; //Note that the pattern size is deliberately make bigger than the hsubset because small size give very blurry checkercorner
	double *maskSmooth = new double[PatternLength*numPatterns * 2];

	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	synthesize_square_mask(maskSmooth + PatternLength, bi_graylevel, PatternSize, 1.0, 1, false);

	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (ii = 1; ii < PatternAngles.size(); ii++)
	{
		double c = cos(PatternAngles[ii] * 3.14159265359 / 180), s = sin(PatternAngles[ii] * 3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3), mat_mul(H2, H1, temp, 3, 3, 3), mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + 2 * ii*PatternLength, PatternSize, PatternSize, maskSmooth, PatternSize, PatternSize, trans, 1, 1, NULL);
		TransformImage(maskSmooth + (2 * ii + 1)*PatternLength, PatternSize, PatternSize, maskSmooth + PatternLength, PatternSize, PatternSize, trans, 1, 1, NULL);
	}
	/*	FILE *fp = fopen("C:/temp/coarse.txt", "w+");
	for(ii=0; ii<npts; ii++)
	fprintf(fp, "%.2f %.2f \n", Checker[ii].x, Checker[ii].y);
	fclose(fp);*/

	//Detect coarse corners:
	int *goodCandiates = new int[npts];
	Point2d *goodCorners = new Point2d[npts];
	int count = 0, ngoodCandiates = 0, squaretype;

	int percent = 10, increP = 10;
	double start = omp_get_wtime(), elapsed;
	//#pragma omp critical
	//cout << "Coarse refinement ..." << endl;

	double zncc, bestzncc;
	for (ii = 0; ii < npts; ii++)
	{
		if ((Checker[ii].x < boundary) || (Checker[ii].y < boundary) || (Checker[ii].x > 1.0*width - boundary) || (Checker[ii].y > 1.0*height - boundary))
			continue;

		zncc = 0.0, bestzncc = 0.0;
		for (jj = 0; jj < numPatterns; jj++)
		{
			squaretype = TMatchingCoarse(maskSmooth + 2 * jj*PatternLength, PatternSize, hsubset1, Para, width, height, nchannels, Checker[ii], searchArea, ZNCCCoarseThresh, zncc, InterpAlgo);
			if (squaretype > -1 && zncc > bestzncc)
			{
				goodCorners[count].x = Checker[ii].x;
				goodCorners[count].y = Checker[ii].y;
				goodCandiates[count] = squaretype + 2 * jj;
				bestzncc = zncc;
			}
		}
		if (bestzncc > ZNCCCoarseThresh)
			count++;
	}
	ngoodCandiates = count;
	elapsed = omp_get_wtime() - start;

	/*FILE *fp = fopen("C:/temp/coarseR.txt", "w+");
	for (ii = 0; ii < ngoodCandiates; ii++)
	fprintf(fp, "%.2f %.2f %d\n", goodCorners[ii].x, goodCorners[ii].y, goodCandiates[ii]);
	fclose(fp);*/

	//Merege coarsely detected candidates:
	npts = ngoodCandiates;
	int STACK[30]; //Maximum KNN
	int *squareType = new int[npts];
	Point2d *mergeCorners = new Point2d[npts];
	int *marker = new int[2 * npts];
	for (jj = 0; jj < 2 * npts; jj++)
		marker[jj] = -1;

	int flag, KNN;
	double t1, t2, megre_thresh = 5.0;
	count = 0, ngoodCandiates = 0;
	for (jj = 0; jj < npts; jj++)
	{
		KNN = 0;
		flag = 0;
		for (ii = 0; ii < count; ii++)
		{
			if (marker[ii] == jj)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
			continue;

		for (ii = jj + 1; ii < npts; ii++)
		{
			t1 = goodCorners[ii].x - goodCorners[jj].x;
			t2 = goodCorners[ii].y - goodCorners[jj].y;

			if (t1*t1 + t2 * t2 < megre_thresh*megre_thresh &&goodCandiates[ii] == goodCandiates[jj])
			{
				STACK[KNN] = ii;
				KNN++;
			}
		}
		STACK[KNN] = jj;// include itself

		for (kk = 0; kk < KNN + 1; kk++)
		{
			marker[count] = STACK[kk];
			count++;
		}

		mergeCorners[ngoodCandiates].x = 0.0, mergeCorners[ngoodCandiates].y = 0.0;
		for (kk = 0; kk <= KNN; kk++)
		{
			mergeCorners[ngoodCandiates].x += goodCorners[STACK[kk]].x;
			mergeCorners[ngoodCandiates].y += goodCorners[STACK[kk]].y;
		}
		mergeCorners[ngoodCandiates].x /= (KNN + 1);
		mergeCorners[ngoodCandiates].y /= (KNN + 1);
		squareType[ngoodCandiates] = goodCandiates[jj];
		ngoodCandiates++;
	}

	/*fp = fopen("c:/temp/coarseRM.txt", "w+");
	for (ii = 0; ii < ngoodCandiates; ii++)
	fprintf(fp, "%lf %lf %d\n", mergeCorners[ii].x, mergeCorners[ii].y, squareType[ii]);
	fclose(fp);*/

	//Refine corners:
	int advanced_tech = 3; // affine only
	count = 0;
	double *Znssd_reqd = new double[9 * PatternLength];

	percent = 10;
	start = omp_get_wtime();
	for (ii = 0; ii < ngoodCandiates; ii++)
	{
		if ((mergeCorners[ii].x < boundary) || (mergeCorners[ii].y < boundary) || (mergeCorners[ii].x > 1.0*width - boundary) || (mergeCorners[ii].y > 1.0*height - boundary))
			continue;

		zncc = TMatchingFine_ZNCC(maskSmooth + squareType[ii] * PatternLength, PatternSize, hsubset2, Para, width, height, 1, mergeCorners[ii], advanced_tech, 1, ZNCCthresh, InterpAlgo, Znssd_reqd);
		if (zncc > ZNCCthresh)
		{
			squareType[ii] = squareType[ii];
			count++;
		}
		else
			squareType[ii] = -1;
	}
	delete[]Znssd_reqd;
	elapsed = omp_get_wtime() - start;

	//Final merging:
	count = 0;
	for (ii = 0; ii < ngoodCandiates; ii++)
	{
		if (squareType[ii] != -1)
		{
			goodCorners[count].x = mergeCorners[ii].x;
			goodCorners[count].y = mergeCorners[ii].y;
			goodCandiates[count] = squareType[ii];
			count++;
		}
	}

	npts = count;
	for (jj = 0; jj < npts; jj++)
		marker[jj] = -1;

	megre_thresh = 4.0, count = 0, ngoodCandiates = 0;
	for (jj = 0; jj < npts; jj++)
	{
		KNN = 0, flag = 0;
		for (ii = 0; ii < count; ii++)
		{
			if (marker[ii] == jj)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
			continue;

		for (ii = jj + 1; ii < npts; ii++)
		{
			t1 = goodCorners[ii].x - goodCorners[jj].x;
			t2 = goodCorners[ii].y - goodCorners[jj].y;
			if (t1*t1 + t2 * t2 < megre_thresh*megre_thresh)
			{
				STACK[KNN] = ii;
			}
		}
		STACK[KNN] = jj;// include itself

		for (kk = 0; kk < KNN + 1; kk++)
		{
			marker[count] = STACK[kk];
			count++;
		}

		Fcorners[ngoodCandiates].x = goodCorners[jj].x, Fcorners[ngoodCandiates].y = goodCorners[jj].y;
		for (kk = 0; kk < KNN; kk++)
		{
			Fcorners[ngoodCandiates].x += goodCorners[STACK[kk]].x;
			Fcorners[ngoodCandiates].y += goodCorners[STACK[kk]].y;
		}
		Fcorners[ngoodCandiates].x /= (KNN + 1);
		Fcorners[ngoodCandiates].y /= (KNN + 1);
		FStype[ngoodCandiates] = goodCandiates[jj];
		ngoodCandiates++;
	}
	npts = ngoodCandiates;

	delete[]maskSmooth;
	delete[]goodCorners;
	delete[]goodCandiates;
	delete[]marker;
	delete[]squareType;
	delete[]mergeCorners;

	return;
}
void RefineCheckerCornersFromInit(double *Para, int width, int height, int nchannels, Point2d *Checker, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo)
{
	int numPatterns = (int)PatternAngles.size();
	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize * PatternSize; //Note that the pattern size is deliberately make bigger than the hsubset because small size give very blurry checkercorner

	double *maskSmooth = new double[PatternLength*numPatterns];
	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);

	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (int ii = 1; ii < numPatterns; ii++)
	{
		double c = cos(PatternAngles[ii] * 3.14159265359 / 180), s = sin(PatternAngles[ii] * 3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3), mat_mul(H2, H1, temp, 3, 3, 3), mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + ii * PatternLength, PatternSize, PatternSize, maskSmooth, PatternSize, PatternSize, trans, 1, 1, NULL);
	}

	//Detect coarse corners:
	double zncc, bestzncc;
	Point2d bestPts, bkPt;
	int advanced_tech = 3; // affine only
	double *Znssd_reqd = new double[9 * PatternLength];

	for (int ii = 0; ii < npts; ii++)
	{
		bestzncc = 0.0;
		for (int jj = 0; jj < numPatterns; jj++)
		{
			bkPt = Checker[ii];
			zncc = TMatchingFine_ZNCC(maskSmooth + jj * PatternLength, PatternSize, hsubset2, Para, width, height, nchannels, bkPt, advanced_tech, 1, ZNCCthresh, InterpAlgo, Znssd_reqd);
			if (zncc > bestzncc)
			{
				bestzncc = zncc;
				bestPts = bkPt;
			}
		}

		if (bestzncc < ZNCCthresh)
			Checker[ii] = Point2d(-1, -1);
		else
			Checker[ii] = bestPts;
	}
	delete[]Znssd_reqd;
	delete[]maskSmooth;

	return;
}
void RunCheckerCornersDetector(Point2d *CornerPts, int *CornersType, int &nCpts, double *Img, double *IPara, int width, int height, int nchannels, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCThresh, int InterpAlgo)
{
	int npts = 500000;
	Point2d *Checker = new Point2d[npts];

	//#pragma omp critical
	//cout << "Sliding window for detection..." << endl;

	DetectCheckerCornersCorrelation(Img, width, height, nchannels, Checker, npts, PatternAngles, hsubset1, searchArea, ZNCCCoarseThresh);
	/*FILE *fp = fopen("C:/temp/cornerCorr.txt", "w+");
	for (int ii = 0; ii < npts; ii++)
	fprintf(fp, "%.1f %1f\n", Checker[ii].x, Checker[ii].y);
	fclose(fp);
	FILE *fp = fopen("C:/temp/cornerCorr.txt", "r");
	npts = 0;
	while (fscanf(fp, "%lf %lf ", &Checker[npts].x, &Checker[npts].y) != EOF)
	npts++;
	fclose(fp);*/

	//#pragma omp critical
	//cout << "finished width " << npts << " points. Refine detected corners..." << endl;

	RefineCheckerCorners(IPara, width, height, nchannels, Checker, CornerPts, CornersType, npts, PatternAngles, hsubset1, hsubset2, searchArea, ZNCCCoarseThresh, ZNCCThresh, InterpAlgo);
	nCpts = npts;

	delete[]Checker;
	return;
}

int GetPoint2DPairCorrespondence(char *Path, int timeID, vector<int>viewID, vector<KeyPoint>&keypoints1, vector<KeyPoint>&keypoints2, vector<int>&CorrespondencesID, bool useGPU)
{
	//SelectedIndex: index of correspondenceID in the total points pool
	keypoints1.clear(), keypoints2.clear(), CorrespondencesID.clear();
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
	ReadKPointsBinarySIFT(Fname, keypoints2);


	vector<int>matches; matches.reserve(500);//Cannot be found in more than 500 views!

	if (timeID < 0)
		sprintf(Fname, "%s/M_%.2d_%.2d.dat", Path, viewID.at(0), viewID.at(1));
	else
		sprintf(Fname, "%s/M_%.4d_%.2d_%.2d.dat", Path, timeID, viewID.at(0), viewID.at(1));

	int npts, id1, id2;
	FILE *fp = fopen(Fname, "r");
	fscanf(fp, "%d ", &npts);
	CorrespondencesID.reserve(npts * 2);
	while (fscanf(fp, "%d %d ", &id1, &id2) != EOF)
		CorrespondencesID.push_back(id1), CorrespondencesID.push_back(id2);
	fclose(fp);

	return 0;
}
int DisplayImageCorrespondence(Mat &correspond, int offsetX, int offsetY, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<int>pair, double density)
{
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

	int nmatches = (int)pair.size() / 2, step = (int)((2.0 / density)) / 2 * 2;
	step = step > 0 ? step : 2;

	for (int ii = 0; ii < pair.size(); ii += step)
	{
		int x1 = keypoints1.at(pair[ii]).pt.x, y1 = keypoints1.at(pair[ii]).pt.y;
		int x2 = keypoints2.at(pair[ii + 1]).pt.x + offsetX, y2 = keypoints2.at(pair[ii + 1]).pt.y + offsetY;

		cv::circle(correspond, cvPoint(x1, y1), 2, colors[ii % 9], 2), cv::circle(correspond, cvPoint(x2, y2), 2, colors[ii % 9], 2);
		//cvLine(correspond, cvPoint(x1, y1), cvPoint(x2, y2), colors[ii % 9], 1);
	}

	cv::namedWindow("Correspondence", CV_WINDOW_NORMAL);
	cv::imshow("Correspondence", correspond);
	cv::waitKey(-1);
	printLOG("Images closed\n");
	return 0;
}
int DisplayImageCorrespondence(Mat &correspond, int offsetX, int offsetY, vector<Point2d> keypoints1, vector<Point2d> keypoints2, vector<int>pair, double density)
{
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
	int nmatches = (int)pair.size() / 2, step = (int)((2.0 / density)) / 2 * 2;
	step = step > 0 ? step : 2;

	for (int ii = 0; ii < pair.size(); ii += step)
	{
		int x1 = keypoints1.at(pair[ii]).x, y1 = keypoints1.at(pair[ii]).y;
		int x2 = keypoints2.at(pair[ii + 1]).x + offsetX, y2 = keypoints2.at(pair[ii + 1]).y + offsetY;
		//cvLine(correspond, cvPoint(x1, y1), cvPoint(x2, y2), colors[ii % 9], 1);
		circle(correspond, cvPoint(x1, y1), 1, colors[ii % 9], 2), circle(correspond, cvPoint(x2, y2), 1, colors[ii % 9], 2);
	}

	namedWindow("Correspondence", CV_WINDOW_NORMAL);
	imshow("Correspondence", correspond);
	waitKey(-1);
	printLOG("Images closed\n");
	return 0;
}
int DisplayImageCorrespondencesDriver(char *Path, vector<int>AvailViews, int timeID, int nchannels, double density)
{
	char Fname[512];

	vector<int>CorrespondencesID;
	vector<KeyPoint>keypoints1, keypoints2;
	GetPoint2DPairCorrespondence(Path, timeID, AvailViews, keypoints1, keypoints2, CorrespondencesID);

	if (timeID < 0)
		sprintf(Fname, "%s/%.4d.png", Path, AvailViews.at(0));
	else
		sprintf(Fname, "%s/%d/%.4d.png", Path, AvailViews.at(0), timeID);
	Mat Img1 = imread(Fname, nchannels == 3 ? 1 : 0);
	if (Img1.empty())
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}

	if (timeID < 0)
		sprintf(Fname, "%s/%.4d.png", Path, AvailViews.at(1));
	else
		sprintf(Fname, "%s/%d/%.4d.png", Path, AvailViews.at(1), timeID);
	Mat Img2 = imread(Fname, nchannels == 3 ? 1 : 0);
	if (Img2.empty())
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}

	cv::Mat outImg(max(Img1.rows, Img2.rows), Img1.cols + Img2.cols, CV_8UC3);
	cv::Rect rect1(0, 0, Img1.cols, Img1.rows);
	cv::Rect rect2(Img1.cols, 0, Img1.cols, Img1.rows);

	Img1.copyTo(outImg(rect1));
	Img2.copyTo(outImg(rect2));

	DisplayImageCorrespondence(outImg, Img1.cols, 0, keypoints1, keypoints2, CorrespondencesID, density);

	return 0;
}




