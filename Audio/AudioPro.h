#if !defined(SEQUENCE_H )
#define SEQUENCE_H
#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../Ulti/DataIO.h"
#include "../Ulti/MiscAlgo.h"
using namespace std;
using namespace cv;

class Sequence
{
public:
	int width, height, nchannels, nframes, nsamples, sampleRate;
	char *Img;
	float *Audio;
	double TimeAlignPara[2];

	void InitSeq(double fps, double offset)
	{
		TimeAlignPara[0] = fps,TimeAlignPara[1] = offset;
		Img = 0, Audio = 0;
		return ;
	}

	~Sequence()
	{
		delete []Img;
		delete []Audio;
	}
};

int ReadAudio(char *Fin, Sequence &mySeq, char *Fout = 0);
int SynAudio(char *Fname1, char *Fname2, double fps1, double fps2, int MinSample, double &finalframeOffset, double &MaxZNCC, double reliableThreshold = 0.25);

#endif