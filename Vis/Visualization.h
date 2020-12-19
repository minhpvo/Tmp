#if !defined( VISUALIZATION_H )
#define VISUALIZATION_H 

#include "../GL/glew.h"
#ifdef _WINDOWS
#include "../ThirdParty/GL/freeglut.h"
#else
#include <GL/glut.h>
#endif


#include <cstdlib>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

#include <opencv2/opencv.hpp>
#include "../DataStructure.h"
#include "../Ulti/MathUlti.h"
#include "../Ulti/DataIO.h"
#include "../Drivers/Drivers.h"


#define VIEWING_DISTANCE_MIN  1.0

using namespace cv;
using namespace std;
void SelectionFunction(int x, int y, bool append_rightClick = false);
void Keyboard(unsigned char key, int x, int y);
void SpecialInput(int key, int x, int y);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void ReshapeGL(int width, int height);

void DrawCube(Point3f &length);
void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D);
void RenderGroundPlane();

void Draw_Axes();
void DrawCamera(int highlight = 0);
void RenderObjects();
void display(void);
void Keyboard(unsigned char key, int x, int y);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void ReshapeGL(int width, int height);

void visualization();
int visualizationDriver(char *inPath, vector<int> &sCams, int StartTime, int StopTime, int increTime, bool hasColor, int DoColorCode, bool hasPatchNormal, bool hasTimeEvoling3DPoints, bool CatUnStructured3DPoints, int SkeletonPointFormat, int CurrentTime, bool syncedMode, int ShutterType);
int visualizationDriver2(char *inPath, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, vector<int> &SelectedCams, int startF, int stopF, int increF_, bool hasColor_, int colorCoded_, bool hasPatchNormal_, bool hasTimeEvoling3DPoints_, bool CatUnStructured3DPoints_, int SkeletonPointFormat, int CurrentFrame, bool syncedMode_);

void ReadCurrentSfmGL(char *path, bool hasColor, bool hasNormal);
bool ReadCurrent3DGL(char *path, bool hasColor, bool hasNormal, int timeID, bool setCoordinate);
bool ReadCurrent3DGL2(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate);
bool ReadAllCurrent3DSkeleton(char *path, int timeID, int PointFormat = 25);
bool ReadPerCurrent3DSkeleton(char *path, int timeID, int PointFormat = 25);
bool ReadCurrent3DBody(char *path, int timeID);
bool GetCurrent3DBody(char *Fname, int timeID);

int Read3DTrajectory(char *path, int trialID = 0, int colorCoded = 1);
int ReadCurrentPosesGL(char *path, int StartTime, int StopTime, Point3d *Ts, int ShutterType);
int ReadCurrentPosesGL2(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, int startF, int stopF, Point3d *TimeInfo);

int screenShot(char *Fname, int width, int height, bool color);
//int visualizePerFrameSfM(char *inPath, vector<int> &SelectedCams, int StartTime, int StopTime, bool hasColor, int DoColorCode, bool hasPatchNormal, bool hasTimeEvoling3DPoints, bool CatUnStructured3DPoints, int CurrentTime, int ShutterType);//(char *inPath, int refF, int startF, int stopF, int nCams, int nMaxPeople);
int visualizePerFrameSfM(char *inPath, int refFrame, int startF, int stopF, int nCams, int nMaxPeople);
int LiveSfM(char *Path, int startF, int stopF, int increF, int nCams, int nMaxPeople = 10, int verbose = 0);
#endif


