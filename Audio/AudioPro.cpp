#include "AudioPro.h"


using namespace cv;
using namespace std;

#ifdef _WINDOWS
#include "../ThirdParty/libsndfile/include/sndfile.h"
#pragma comment(lib, "../ThirdParty/libsndfile/lib/libsndfile-1.lib")

int ReadAudio(char *Fin, Sequence &mySeq, char *Fout)
{
	SNDFILE      *infile;
	SF_INFO      sinfo;

	int nchannels;
	if (!(infile = sf_open(Fin, SFM_READ, &sinfo)))
	{
		printLOG("Not able to open input file %s.\n", Fin);
		return  1;
	}
	else
	{
		mySeq.nsamples = (int)sinfo.frames, mySeq.sampleRate = (int)sinfo.samplerate, nchannels = (int)sinfo.channels;
		//printLOG("Number of sample per channel=%d, Samplerate=%d, Channels=%d\n", mySeq.nsamples, mySeq.sampleRate, nchannels);
	}

	float *buf = (float *)malloc(mySeq.nsamples*nchannels*sizeof(float));
	int num = (int)sf_read_float(infile, buf, mySeq.nsamples*nchannels);

	//I want only 1 channel
	mySeq.Audio = new float[mySeq.nsamples];
	for (int i = 0; i < mySeq.nsamples; i++)
		mySeq.Audio[i] = buf[nchannels*i];
	delete[]buf;

	if (Fout != NULL)
		WriteGridBinary(Fout, mySeq.Audio, 1, mySeq.nsamples);

	return 0;
}
int SynAudio(char *Fname1, char *Fname2, double fps1, double fps2, int MinSample, double &finalfoffset, double &MaxZNCC, double reliableThreshold)
{
	omp_set_num_threads(omp_get_max_threads());

	int jj;
	Sequence Seq1, Seq2;
	Seq1.InitSeq(fps1, 0.0);
	Seq2.InitSeq(fps2, 0.0);
	if (ReadAudio(Fname1, Seq1) != 0)
		return 1;
	if (ReadAudio(Fname2, Seq2) != 0)
		return 1;

	if (Seq1.sampleRate != Seq2.sampleRate)
	{
		printLOG("Sample rate of %s and %s do not match. Stop!\n", Fname1, Fname2);
		return 1;
	}
	double sampleRate = Seq1.sampleRate;

	MinSample = min(MinSample, min(Seq1.nsamples, Seq2.nsamples));
	int nSpliting = (int)floor(1.0*min(Seq1.nsamples, Seq2.nsamples) / MinSample);
	nSpliting = nSpliting == 0 ? 1 : nSpliting;

	//Take gradient of signals: somehow, this seems to be robust
	int filterSize = 6;
	float GaussianDfilter[] = { -0.0219, -0.0764, -0.0638, 0.0638, 0.0764, 0.0219 };
	float *Grad1 = new float[Seq1.nsamples + filterSize - 1], *Grad2 = new float[Seq2.nsamples + filterSize - 1];
	conv(Seq1.Audio, Seq1.nsamples, GaussianDfilter, filterSize, Grad1);
	conv(Seq2.Audio, Seq2.nsamples, GaussianDfilter, filterSize, Grad2);

	for (int ii = 0; ii < Seq1.nsamples + filterSize - 1; ii++)
		Grad1[ii] = abs(Grad1[ii]);
	for (int ii = 0; ii < Seq2.nsamples + filterSize - 1; ii++)
		Grad2[ii] = abs(Grad2[ii]);

	int ns3, ns4;
	double fps3, fps4;
	float *Seq3, *Seq4;

	bool Switch = false;
	if (Seq1.nsamples <= Seq2.nsamples)
	{
		fps3 = fps1, fps4 = fps2;
		ns3 = Seq1.nsamples + filterSize - 1;
		Seq3 = new float[ns3];
#pragma omp parallel for
		for (int i = 0; i < ns3; i++)
			Seq3[i] = Grad1[i];

		ns4 = Seq2.nsamples + filterSize - 1;
		Seq4 = new float[ns4];
#pragma omp parallel for
		for (int i = 0; i < ns4; i++)
			Seq4[i] = Grad2[i];
	}
	else
	{
		Switch = true;
		fps3 = fps2, fps4 = fps1;
		ns3 = Seq2.nsamples + filterSize - 1;
		Seq3 = new float[ns3];
#pragma omp parallel for
		for (int i = 0; i < ns3; i++)
			Seq3[i] = Grad2[i];

		ns4 = Seq1.nsamples + filterSize - 1;
		Seq4 = new float[ns4];
#pragma omp parallel for
		for (int i = 0; i < ns4; i++)
			Seq4[i] = Grad1[i];
	}

	const int hbandwidth = sampleRate / 30; //usually 30fps, so this give 0.5 frame accuracy
	int nMaxLoc;
	int *MaxLocID = new int[ns3 + ns4 - 1];
	float *res = new float[ns4 + ns3 - 1];
	float *nres = new float[ns4 + ns3 - 1];

	//Correlate the Seq4 with the smaller sequence (i.e. seq3)
	ZNCC1D(Seq3, ns3, Seq4, ns4, res);

	//Quality check: how many peaks, are they close?
	nonMaximaSuppression1D(res, ns3 + ns4 - 1, MaxLocID, nMaxLoc, hbandwidth);
	for (jj = 0; jj < ns3 + ns4 - 1; jj++)
		nres[jj] = 0.0;
	for (jj = 0; jj < nMaxLoc; jj++)
		nres[MaxLocID[jj]] = res[MaxLocID[jj]];

	Mat zncc(1, ns3 + ns4 - 1, CV_32F, nres);

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal, maxVal2; Point minLoc; Point maxLoc, maxLoc2;
	minMaxLoc(zncc, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	double MaxCorr = maxVal;
	double soffset = maxLoc.x - ns3 + 1;
	double foffset = 1.0*(soffset) / sampleRate*fps4;

	zncc.at<float>(maxLoc.x) = 0.0;
	minMaxLoc(zncc, &minVal, &maxVal2, &minLoc, &maxLoc2, Mat());

	double bestscore = maxVal;
	if (ns3 == ns4)
	{
		//sometimes, reversr the order leads to different result. The one with the highest ZNCC score is chosen.
		ZNCC1D(Seq4, ns4, Seq3, ns3, res);

		//Quality check: how many peaks, are they close?
		nonMaximaSuppression1D(res, ns3 + ns4 - 1, MaxLocID, nMaxLoc, hbandwidth);
		for (jj = 0; jj < ns3 + ns4 - 1; jj++)
			nres[jj] = 0.0;
		for (jj = 0; jj < nMaxLoc; jj++)
			nres[MaxLocID[jj]] = res[MaxLocID[jj]];

		for (jj = 0; jj < ns3 + ns4 - 1; jj++)
			zncc.at<float>(jj) = nres[jj];

		/// Localizing the best match with minMaxLoc
		double minVal_, maxVal_; Point minLoc_; Point maxLoc_, maxLoc2_;
		minMaxLoc(zncc, &minVal_, &maxVal_, &minLoc_, &maxLoc_, Mat());

		if (bestscore < maxVal_)
		{
			Switch = !Switch;
			maxVal = maxVal_, minLoc = minLoc_, maxLoc = maxLoc_;

			MaxCorr = maxVal;
			soffset = maxLoc.x - ns4 + 1;
			foffset = 1.0*(soffset) / sampleRate*fps4;

			zncc.at<float>(maxLoc.x) = 0.0;
			minMaxLoc(zncc, &minVal, &maxVal2, &minLoc, &maxLoc2, Mat());
		}
	}

	if (maxVal2 / maxVal > 0.5 && abs(maxLoc2.x - maxLoc.x) < hbandwidth * 2 + 1)
		printLOG("Caution! Distance to the 2nd best peak (%.4f /%.4f): %d or %.2fs\n", maxVal, maxVal2, abs(maxLoc2.x - maxLoc.x), 1.0*abs(maxLoc2.x - maxLoc.x) / sampleRate*fps4);

	if (!Switch && soffset < 0)
		printLOG("%s is behind of %s %d samples or %.4f sec \n", Fname1, Fname2, abs(soffset), foffset);
	if (!Switch && soffset >= 0)
		printLOG("%s is ahead of %s %d samples or %.4f sec \n", Fname1, Fname2, abs(soffset), foffset);
	if (Switch && soffset < 0)
		printLOG("%s is ahead of %s %d samples or %.4f sec \n", Fname1, Fname2, abs(soffset), -foffset);
	if (Switch && soffset >= 0)
		printLOG("%s is behind of %s %d samples or %.4f sec \n", Fname1, Fname2, abs(soffset), -foffset);


	if (bestscore < reliableThreshold)
	{
		printLOG("The result is very unreliable (ZNCC = %.2f)! No offset will be generated.", bestscore);

		delete[]Grad1, delete[]Grad2;
		delete[]Seq3, delete[]Seq4;
		delete[]res, delete[]nres;

		return 1;
	}
	else
	{
		int fsoffset = soffset;
		finalfoffset = Switch ? -1.0*fsoffset / sampleRate*fps3 : 1.0*fsoffset / sampleRate*fps4;
		printLOG("Final offset: %d samples or %.2f frames with ZNCC score %.4f\n\n", fsoffset, finalfoffset, MaxCorr);

		MaxZNCC = MaxCorr;

		delete[]Grad1, delete[]Grad2;
		delete[]Seq3, delete[]Seq4;
		delete[]res, delete[]nres;

		return 0;
	}
}
#endif




