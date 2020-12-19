#include"Visualization.h"

#ifdef _WINDOWS
#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "glew32.lib")
#endif

using namespace std;
using namespace cv;

#define RADPERDEG 0.0174533

char *Path;

using namespace smpl;
SMPLModel mySMPL;
MatrixXdr *AllV;
vector<SMPLParams> gVparams;

const int nCoCoJoints = 18;
int PointFormat = 25;
int nMaxPeople = 14, maxInstance = 4000;

const GLfloat red[3] = { 1, 0, 0 };
const GLfloat blue[3] = { 0, 0, 1 };
const GLfloat cyan[3] = { 0, 1, 1 };
const GLfloat white[3] = { 1, 1, 1 };
const GLfloat green[3] = { 0, 1, 0 };
const GLfloat black[3] = { 0, 0, 0 };

int StateChanged = 0;
int refCid = 0, refFid = 0, increF = 1, increT = 1;

bool GLProperlyInit = false;
vector<int> TrajRealId;
int renderTime = 0;
bool showImg = false, displayAllMovingCams = true;

GLfloat UnitScale = 1.0f; //1 unit corresponds to 1 mm
GLfloat g_ratio, g_coordAxisLength, g_fViewDistance = 0, g_nearPlane, g_farPlane;
int g_Width = 1600, g_Height = 900, org_Width = g_Width, org_Height = g_Height, g_xClick = 0, g_yClick = 0, g_mouseYRotate = 0, g_mouseXRotate = 0;

GLfloat CameraSize, pointSize, normalSize, arrowThickness;
double Discontinuity3DThresh = 1000.0, //mm
DiscontinuityTimeThresh = 1000.0; //unit: ms
int nCams = 0, nCorpusCams = 0, nNonCorpusCams, timeID = 0, TrajecID = 0, otimeID = 0, oTrajecID = 0, TrialID = 0, oTrialID = 0, startTime = 0, maxTime = 0, maxTrial = 10000, nTraject = 1, snapCount = 0;

enum cam_mode { CAM_DEFAULT, CAM_ROTATE, CAM_ZOOM, CAM_PAN };
static cam_mode g_camMode = CAM_DEFAULT;

VideoData AllVideosInfo[100];

bool rotateInZ = true;
bool showSMPL = false, showSkeleton = false;
bool hasColor = true, drawPatchNormal = false; int colorCoded = 1;
bool g_bButton1Down = false, ReCenterNeeded = false, PickingMode = false, bFullsreen = false, showGroundPlane = false, changeBackgroundColor = false, showAxis = false;
bool SaveScreen = false, ImmediateSnap = false, SaveStaticViewingParameters = false, SetStaticViewingParameters = true, SaveDynamicViewingParameters = false, SetDynamicViewingParameters = false, SaveRendererViewingParameters = false, SetRendererViewingParameters = false, RenderedReady = false;

bool showOCRInstance = false;
bool  OneTimeInstanceOnly = false, IndiviualTrajectory = false, hasTimeEvoling3DPoints = true, EndTime = false;
bool drawCorpusPoints = true, drawCorpusCameras = true, drawNonCorpusCameras = true, drawedNonCorpusCameras = false, drawBriefCameraTraject = true, AutoDisp = false;
bool drawCameraID = true, syncedMode = false;
bool CatUnStructured3DPoints = false, Unorder3DPointsOne = true, Unorder3DPointsTwo = true;
bool FirstPose = true, SecondPose = false;
double DisplayStartTime = 0.0, DisplayTimeStep = 0.01; //60fps

GLfloat PointsCentroid[3], PointsVAR[3];
vector<int> PickedStationaryPoints, PickedDynamicPoints, PickedTraject, PickedCorpusCams, PickedNonCorpusCams;
vector<Point3d> PickPoint3D, SkeletonPoints;
vector<int> selectedCams;
vector<Point3f> *AllPeopleMesh;
vector<Point3f> *AllPeopleVertex;
vector<int>* AllPeopleVertexTimeId;
vector<int>* AllPeopleMeshTimeId;

struct LiteHumanSkeleton3D
{
	int refFid, nPts;
	Point3d pt3d[25];
};
struct sLiteHumanSkeleton3D
{
	vector< LiteHumanSkeleton3D> sSke;
};
vector<sLiteHumanSkeleton3D > allSkeleton;
vector<int> vPeopleId, cPeopleId;

struct TsCidFid {
	TsCidFid(double Ts, int Cid, int Fid) : Ts(Ts), Cid(Cid), Fid(Fid) {}

	double Ts;
	int Cid, Fid;
};
int *SortedTimeInstance = NULL;;
vector<TsCidFid> vAll3D_TidCidFid;

typedef struct { GLfloat  viewDistance, CentroidX, CentroidY, CentroidZ; int timeID, mouseYRotate, mouseXRotate; } ViewingParas;
vector <ViewingParas> DynamicViewingParas, RenderViewingParas;
VisualizationManager g_vis;
Point3d *CamTimeInfo = 0;
GLfloat Red[3] = { 1.f, 0.f, 0.f }, Green[3] = { 0, 1, 0 }, Blue[3] = { 0, 0, 1 }, White[3] = { 1, 1, 1 }, Yellow[3] = { 1.0f, 1.0f, 0 }, Magneta[3] = { 1.f, 0.f, 1.f }, Cycan[3] = { 0.f, 1.f, 1.f }, globalColor[3];

//float fps = 10.0, Tscale = 1000.0, radius = 2.5e3;
//double fps = 1, Tscale = 1000000.0, radius = 2.5e3;
double fps = 1, Tscale = 1000000, radius = 2.5e3;

double VTthresh = 10;
bool hasMedLengthInfo = false;
double medLengh[100];

#include "nlohmann/json.hpp"
using namespace nlohmann;

vector<GLfloat> groupColors;
vector<vector<int> >CamGroups;
vector<Point3f> OCR3D;
vector<string> OCRName;
vector<int> OCR_type;
void ReadGroup(char *Path)
{
	char Fname[512];
	CamGroups.clear();
	int nvis, fid;
	float dummy;
	sprintf(Fname, "%s/split.txt", Path);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d ", &nvis) != EOF)
	{
		vector<int> group;
		for (int ii = 0; ii < nvis; ii++)
		{
			fscanf(fp, "%d ", &fid);
			group.push_back(fid);
		}
		CamGroups.push_back(group);
		groupColors.push_back(0.001f*(rand() % 100));
		groupColors.push_back(0.001f*(rand() % 100));
		groupColors.push_back(0.001f*(rand() % 100));
	}
	fclose(fp);

	return;
}
void ReadOCR3D(char *Path)
{
	char Fname[512];
	int ii;
	float x, y, z;

	sprintf(Fname, "%s/text3D_clean.txt", Path);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d %d %d %f %f %f %s", &ii, &ii, &ii, &x, &y, &z, Fname) != EOF)
	{
		if (abs(x) + abs(y) + abs(z) > 1000)
			continue;
		OCR_type.push_back(ii);
		OCR3D.emplace_back(x, y, z);
		OCRName.push_back(string(Fname));
	}
	fclose(fp);

	return;
}
void ReadCurrentSfmGL2(char *Path, bool hasColor, bool drawPatchNormal)
{
	char Fname[512];
	int viewID, nviews;

	int deviceId = 0;
	sprintf(Fname, "%s/pose_%d.json", Path, deviceId);
	if (!IsFileExist(Fname))
		return;

	std::ifstream ifs(Fname);
	json pose_json;
	ifs >> pose_json;
	for (auto pose_ : pose_json)
	{
		int fid = std::round((double)(pose_["fid"])) - 1; // first frame is indexed 1 while the extracted frames is indexed 0
		CameraData camI;

		double qw = pose_["qwxyz"][0], qx = pose_["qwxyz"][1], qy = pose_["qwxyz"][2], qz = pose_["qwxyz"][3];
		for (int ii = 0; ii < 3; ii++)
			camI.camCenter[ii] = pose_["txyz"][ii];

		camI.Quat[0] = qw, camI.Quat[1] = qx, camI.Quat[2] = qy, camI.Quat[3] = qz;
		Quaternion2Rotation(camI.Quat, camI.invR);
		mat_transpose(camI.invR, camI.R, 3, 3);
		GetTfromC(camI.R, camI.camCenter, camI.T);
		GetrtFromRT(camI.rt, camI.R, camI.T);

		GetRCGL(camI.R, camI.T, camI.Rgl, camI.camCenter);

		camI.viewID = fid;
		g_vis.glCorpusCameraInfo.push_back(camI);
	}

	hasColor = false;
	g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(10e5);

	int pid, dummy;
	Point3i iColor; Point3f fColor; Point3f t3d;
	sprintf(Fname, "%s/points.json", Path, deviceId);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, " %f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.CorpusPointPosition.push_back(t3d);
		g_vis.CorpusPointColor.emplace_back(0.0, 1.0, 0.0);
	}
	fclose(fp);


	if (g_vis.CorpusPointPosition.size() > 0)
	{
		PointsCentroid[0] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[1] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[2] /= g_vis.CorpusPointPosition.size();

		PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
		for (int ii = 0; ii < g_vis.CorpusPointPosition.size(); ii++)
		{
			PointsVAR[0] += pow(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], 2);
			PointsVAR[1] += pow(g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], 2);
			PointsVAR[2] += pow(g_vis.CorpusPointPosition[ii].z - PointsCentroid[2], 2);
		}
		PointsVAR[0] = sqrt(PointsVAR[0] / g_vis.CorpusPointPosition.size());
		PointsVAR[1] = sqrt(PointsVAR[1] / g_vis.CorpusPointPosition.size());
		PointsVAR[2] = sqrt(PointsVAR[2] / g_vis.CorpusPointPosition.size());
	}
	else
		PointsCentroid[0] = PointsCentroid[1] = PointsCentroid[2] = 0;

	return;
}
int num_start = 0;

// Shaders
const char* SimpleVertexTransformShader = "#version 330 core\n"
"// Input vertex data, different for all executions of this shader.\n"
"attribute vec3 vertexPosition_modelspace;\n"
"attribute vec3 vertexColor;\n"
"varying vec3 fragmentColor; \n"
"uniform mat4 proj;\n"
"uniform mat4 V;\n"
"uniform mat4 T;\n"
"in vec3 position; \n"
"void main()\n"
"{\n"
"gl_Position = proj * V * T* vec4(vertexPosition_modelspace, 1.); \n"
"fragmentColor = vertexColor;\n"
"}\n";
const char* SimpleFragmentShader = "#version 330 core\n"
"varying vec3 fragmentColor;\n"
"void main()\n"
"{\n"
"gl_FragColor = vec4(fragmentColor, 1);\n"
"}\n";
const char* SingleVertexTransformShader = "#version 330 core\n"
"uniform mat4 proj;\n"
"uniform mat4 V;\n"
"uniform mat4 T;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = proj * V * T* vec4(position, 1.);\n"
"}\n";
const char* SingleFragmentShader = "#version 330 core\n"
"varying vec3 OutColor;\n"
"uniform vec3 InColor;\n"
"void main()\n"
"{\n"
"OutColor = InColor;\n"
"}\n";

GLuint  pointCloud_VAO, pointCloud_VAO2;
vector<GLuint> smpl_VAO, smpl_VBO, smpl_EBO;
GLuint shaderProgram1, shaderProgram2, shaderProgram3;
vector<Point3i> faces;


int orfid = 0;
struct textureCoord
{
	textureCoord()
	{
		textureIdx = 0;
		lu = cv::Point2d(0, 0);
		ru = cv::Point2d(1, 0);
		ld = cv::Point2d(0, 1);
		rd = cv::Point2d(1, 1);
	}

	GLuint textureIdx;
	cv::Point2d lu;
	cv::Point2d ru;
	cv::Point2d ld;
	cv::Point2d rd;
};
struct VolumeVoting {
	bool avail;
	int nstep;
	double cellSize;
	Point3i dim;
	Point3f centroid;
	vector<Point3f> pts, score;
};
VolumeVoting VT;

GLfloat GlobalRot[16] = {
1, 0, 0, 0,
0, 1,0,0,
0, 0, 1 ,0,
0, 0, 0, 1 };

int zPress = 0;
bool vPress = 0;
int modelAx = 0, modelAy = 0;

vector<vector<Point2i> > VnearestCams;

struct ske18pts
{
	int inst;
	Point3f pts[18];
};
vector<ske18pts > *trajBucket;
bool ReadAll3DSkeleton(char *Path, int startF, int stopF, int PointFormat)
{
	char Fname[512];
	int nSke = 0;
	printf("Reading skeleton tracklets:\n");
	while (true)
	{
		int refFid, nvis;
		Point3i cidfiddid; Point2d pt;

		LiteHumanSkeleton3D Ske;
		sLiteHumanSkeleton3D sSke;
		sprintf(Fname, "%s/Skeleton_%d_%d/%.4d.txt", Path, startF, stopF, nSke);
		if (!IsFileExist(Fname))
			break;
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &refFid) != EOF)
		{
			Ske.refFid = refFid;
			Ske.nPts = PointFormat;
			for (int jid = 0; jid < PointFormat; jid++)
			{
				fscanf(fp, "%lf %lf %lf ", &Ske.pt3d[jid].x, &Ske.pt3d[jid].y, &Ske.pt3d[jid].z);
				fscanf(fp, "%d ", &nvis);
				for (size_t ii = 0; ii < nvis; ii++)
					fscanf(fp, "%d %d %d %lf %lf ", &cidfiddid.x, &cidfiddid.y, &cidfiddid.z, &pt.x, &pt.y);
			}
			sSke.sSke.push_back(Ske);
		}
		printf("%d..", nSke);
		fclose(fp);
		if (sSke.sSke.size() > 10)
		{
			allSkeleton.emplace_back(sSke);
			vPeopleId.push_back(nSke);
		}
		nSke++;
	}
	printf("\n");

	trajBucket = new vector<ske18pts>[nSke];
	if (nSke > 0)
		return true;
	else
		return false;
}
void GetPerCurrentSke(int timeID)
{
	//check if data is available
	bool avail = false;
	for (size_t skeId = 0; skeId < allSkeleton.size() && !avail; skeId++)
	{
		for (int inst = 0; inst < allSkeleton[skeId].sSke.size() && !avail; inst++)
		{
			if (allSkeleton[skeId].sSke[inst].refFid == timeID)
			{
				avail = true;
			}
		}
	}

	if (!avail)
		return;

	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	cPeopleId.clear();
	for (size_t skeId = 0; skeId < allSkeleton.size(); skeId++)
	{
		for (int inst = 0; inst < allSkeleton[skeId].sSke.size(); inst++)
		{
			if (allSkeleton[skeId].sSke[inst].refFid == timeID)
			{
				for (int jid = 0; jid < allSkeleton[skeId].sSke[inst].nPts; jid++)
					g_vis.PointPosition.emplace_back(allSkeleton[skeId].sSke[inst].pt3d[jid].x, allSkeleton[skeId].sSke[inst].pt3d[jid].y, allSkeleton[skeId].sSke[inst].pt3d[jid].z);
				cPeopleId.push_back(vPeopleId[skeId]);
				break;
			}
		}
	}
	return;
}

Eigen::Matrix4f frustrum(float  left, float right, float bottom, float top, float nearVal, float farVal)
{
	Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
	P.setConstant(4, 4, 0.);
	P(0, 0) = (2.0 * nearVal) / (right - left);
	P(1, 1) = (2.0 * nearVal) / (top - bottom);
	P(0, 2) = (right + left) / (right - left);
	P(1, 2) = (top + bottom) / (top - bottom);
	P(2, 2) = -(farVal + nearVal) / (farVal - nearVal);
	P(3, 2) = -1.0;
	P(2, 3) = -(2.0 * farVal * nearVal) / (farVal - nearVal);
	return P;
}
GLuint LoadShaders(const char * vertex_file_Path, const char * fragment_file_Path)
{
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_Path, std::ios::in);
	if (VertexShaderStream.is_open())
	{
		std::string Line = "";
		while (getline(VertexShaderStream, Line))
			VertexShaderCode += "\n" + Line;
		VertexShaderStream.close();
	}
	else
	{
		printLOG("Impossible to open %s\n", vertex_file_Path);
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_Path, std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::string Line = "";
		while (getline(FragmentShaderStream, Line))
			FragmentShaderCode += "\n" + Line;
		FragmentShaderStream.close();
	}


	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printLOG("Compiling shader : %s\n", vertex_file_Path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printLOG("%s\n", &VertexShaderErrorMessage[0]);
	}



	// Compile Fragment Shader
	printLOG("Compiling shader : %s\n", fragment_file_Path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printLOG("%s\n", &FragmentShaderErrorMessage[0]);
	}



	// Link the program
	printLOG("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printLOG("%s\n", &ProgramErrorMessage[0]);
	}

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}
GLuint LoadShaders2(std::string VertexShaderCode, std::string FragmentShaderCode)
{
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printLOG("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printLOG("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printLOG("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printLOG("%s\n", &ProgramErrorMessage[0]);
	}

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}
void DetermineDiscontinuityInTrajectory(vector<Point2i> &segNode, Trajectory3D* Trajec3D, int nframes)
{
	double otime = Trajec3D[0].timeID, ntime = Trajec3D[0].timeID;
	Point3d n3d = Trajec3D[0].WC, o3d = Trajec3D[0].WC;
	Point2i Node; Node.x = 0;
	for (int fid = 0; fid < nframes; fid++)
	{
		ntime = Trajec3D[fid].timeID;
		n3d = Trajec3D[fid].WC;
		double dist = sqrt(pow(n3d.x - o3d.x, 2) + pow(n3d.y - o3d.y, 2) + pow(n3d.z - o3d.z, 2));
		if (dist > Discontinuity3DThresh)
		{
			o3d = Trajec3D[fid].WC;
			otime = Trajec3D[fid].timeID;
			Node.y = fid;
			segNode.push_back(Node);
			if (fid + 1 < nframes)
				Node.x = fid + 1;
			continue;
		}
		if (ntime - otime > DiscontinuityTimeThresh)
		{
			o3d = Trajec3D[fid].WC;
			otime = Trajec3D[fid].timeID;
			Node.y = fid;
			segNode.push_back(Node);
			if (fid + 1 < nframes)
				Node.x = fid + 1;
			continue;
		}

		o3d = Trajec3D[fid].WC;
		otime = Trajec3D[fid].timeID;
	}
	Node.y = nframes - 1;
	segNode.push_back(Node);

	return;
}

int SMPL_mode = 3, showPeople = 1;
int Pick(int x, int y)
{
	GLuint buff[64];
	GLint hits, view[4];

	//selection data
	glSelectBuffer(64, buff);
	glGetIntegerv(GL_VIEWPORT, view);
	glRenderMode(GL_SELECT);

	//Push stack for picking
	glInitNames();
	glPushName(0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPickMatrix(x, view[3] - y, 10.0, 10.0, view);
	gluPerspective(65.0, g_ratio, g_nearPlane, g_farPlane);

	glMatrixMode(GL_MODELVIEW);
	RenderObjects();
	glutSwapBuffers();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	hits = glRenderMode(GL_RENDER);
	glMatrixMode(GL_MODELVIEW);

	if (hits > 0)
		return buff[3];
	else
		return -1;
}
void SelectionFunction(int x, int y, bool append_rightClick)
{
	if (PickingMode)
	{
		int pickedID = Pick(x, y);
		if (pickedID < 0)
			return;
		if (pickedID < nCorpusCams)  //camera
		{
			printLOG("Pick camera #%d\n", pickedID);

			bool already = false;
			for (int ii = 0; ii < PickedCorpusCams.size(); ii++)
			{
				if (pickedID == PickedCorpusCams[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickedCorpusCams.push_back(pickedID);
		}
		else if (pickedID < nCorpusCams + nNonCorpusCams)
		{
			int cumCamID = 0;
			for (int cid = 0; cid < nCams; cid++)
			{
				bool found = false;
				if (g_vis.glCameraPoseInfo != NULL && g_vis.glCameraPoseInfo[cid].size() > 0)
				{
					for (int ii = 0; ii < (int)g_vis.glCameraPoseInfo[cid].size(); ii++)
					{
						if (pickedID == cumCamID)
						{
							printLOG("Pick camera #(%d, %d)\n", cid, g_vis.glCameraPoseInfo[cid][ii].frameID);
							found = true;
							break;
						}
						cumCamID++;
					}
				}
				if (found)
					break;
			}

			bool already = false;
			for (int ii = 0; ii < PickedNonCorpusCams.size(); ii++)
			{
				if (pickedID == PickedNonCorpusCams[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickedNonCorpusCams.push_back(pickedID);
		}
		else if (pickedID >= nCorpusCams + nNonCorpusCams &&
			pickedID < g_vis.CorpusPointPosition.size() + nCorpusCams + nNonCorpusCams)// pick points
		{
			pickedID -= nCorpusCams + nNonCorpusCams;

			if (ReCenterNeeded)
			{
				printLOG("New center picked: %d\n", pickedID);
				PointsCentroid[0] = g_vis.CorpusPointPosition[pickedID].x, PointsCentroid[1] = g_vis.CorpusPointPosition[pickedID].y, PointsCentroid[2] = g_vis.CorpusPointPosition[pickedID].z;
				ReCenterNeeded = false;
			}
			else
			{
				printLOG("Picked %d of (%.3f %.3f %.3f) \n", pickedID, g_vis.CorpusPointPosition[pickedID].x, g_vis.CorpusPointPosition[pickedID].y, g_vis.CorpusPointPosition[pickedID].z);

				bool already = false;
				for (int ii = 0; ii < PickedStationaryPoints.size(); ii++)
				{
					if (pickedID == PickedStationaryPoints[ii])
					{
						already = true; break;
					}
				}
				if (!already)
					PickedStationaryPoints.push_back(pickedID);
			}
		}
		else if (pickedID >= g_vis.CorpusPointPosition.size() + nCorpusCams + nNonCorpusCams &&
			pickedID < g_vis.CorpusPointPosition.size() + nCorpusCams + nNonCorpusCams + MaxnTrajectories) //entire trajectories
		{
			pickedID -= (int)g_vis.CorpusPointPosition.size() + nCorpusCams + nNonCorpusCams;
			if (pickedID<0 || pickedID>g_vis.Track3DLength.size())
				return;

			printLOG("Pick trajectory # %d of length %d\n", pickedID, g_vis.Track3DLength[pickedID]);

			bool already = false;
			for (int ii = 0; ii < PickedTraject.size(); ii++)
			{
				if (pickedID == PickedTraject[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickedTraject.push_back(pickedID);
		}
		else if (pickedID >= g_vis.CorpusPointPosition.size() + nCorpusCams + nNonCorpusCams + MaxnTrajectories) //end point in the trajectory
		{
			int tid = (pickedID - (int)g_vis.CorpusPointPosition.size() - nCorpusCams - nNonCorpusCams - MaxnTrajectories) / MaxnFrames;
			int fid = (pickedID - (int)g_vis.CorpusPointPosition.size() - nCorpusCams - nNonCorpusCams - MaxnTrajectories) % MaxnFrames;
			printLOG("Pick (tid, time): (%d %.2f). 3D: %.4f %.4f %.4f\n", tid, g_vis.Traject3D[tid][fid].timeID, g_vis.Traject3D[tid][fid].WC.x, g_vis.Traject3D[tid][fid].WC.y, g_vis.Traject3D[tid][fid].WC.z);

			bool already = false;
			for (int ii = 0; ii < PickedDynamicPoints.size(); ii++)
			{
				if (pickedID == PickedDynamicPoints[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickedDynamicPoints.push_back(pickedID);
		}
	}

	/*double thresh = 20;
	int count = 0;
	for (int ii = 0; ii < g_vis.Traject3D.size(); ii++)
	{
	double otime = g_vis.Traject3D[ii][0].timeID, ntime = g_vis.Traject3D[ii][0].timeID;
	Point3d n3d = g_vis.Traject3D[ii][0].WC, o3d = g_vis.Traject3D[ii][0].WC;

	for (int fid = 0; fid < g_vis.Track3DLength[ii]; fid++)
	{
	ntime = g_vis.Traject3D[ii][fid].timeID;
	n3d = g_vis.Traject3D[ii][fid].WC;
	double dist = sqrt(pow(n3d.x - o3d.x, 2) + pow(n3d.y - o3d.y, 2) + pow(n3d.z - o3d.z, 2));
	if (dist > thresh)
	{
	o3d = g_vis.Traject3D[ii][fid].WC;
	otime = g_vis.Traject3D[ii][fid].timeID;
	continue;
	}
	if (ntime - otime > 33.3 * 4)
	{
	o3d = g_vis.Traject3D[ii][fid].WC;
	otime = g_vis.Traject3D[ii][fid].timeID;
	continue;
	}

	if (count == pickedID)
	{
	bool already = false;
	for (int ii = 0; ii < PickedStationaryPoints.size(); ii++)
	{
	if (pickedID == PickedStationaryPoints[ii])
	{
	already = true;
	break;
	}
	}

	if (!already)
	{
	PickedStationaryPoints.push_back(pickedID);
	PickPoint3D.push_back(g_vis.Traject3D[ii][fid].WC);
	printLOG("Pick point # %d of trajectory %d\n", fid, ii);
	}
	}
	count++;

	o3d = g_vis.Traject3D[ii][fid].WC;
	otime = g_vis.Traject3D[ii][fid].timeID;
	}
	}*/

	return;
}
void Keyboard(unsigned char key, int x, int y)
{
	char Fname[512];
	switch (key)
	{
	case 27:             // ESCAPE key
		if (SaveDynamicViewingParameters)
		{
			sprintf(Fname, "%s/OpenGLDynamicViewingPara.txt", Path); FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)DynamicViewingParas.size(); ii++)
				fprintf(fp, "%d %.8f %d %d %.8f %.8f %.8f \n", DynamicViewingParas[ii].timeID, DynamicViewingParas[ii].viewDistance, DynamicViewingParas[ii].mouseXRotate, DynamicViewingParas[ii].mouseYRotate,
					DynamicViewingParas[ii].CentroidX, DynamicViewingParas[ii].CentroidY, DynamicViewingParas[ii].CentroidZ);
			fclose(fp);
		}
		exit(0);
		break;
	case 'z':
		zPress = 1;
		break;
	case 'v':
		vPress = !vPress;
		break;
	case 'i':
		printLOG("Please enter commands: ");
		cin >> Fname;
		if (strcmp(Fname, "showPeople") == 0)
		{
			cin >> showPeople;
			printLOG("Showing %d perople\n", showPeople);
		}
		if (strcmp(Fname, "SMPL0") == 0)
			SMPL_mode = 0, printLOG("\nUsing SMPL0\n");
		else if (strcmp(Fname, "SMPL1") == 0)
			SMPL_mode = 1, printLOG("\nUsing SMPL1\n");
		else if (strcmp(Fname, "SMPL2") == 0)
			SMPL_mode = 2, printLOG("\nUsing SMPL2\n");
		else if (strcmp(Fname, "SMPL3") == 0)
			SMPL_mode = 3, printLOG("\nUsing SMPL3\n");
		if (strcmp(Fname, "gotoTime") == 0)
		{
			cin >> timeID;
			printLOG("Currrent time: %d\n", timeID);
		}
		if (strcmp(Fname, "ShowAllMovingCam") == 0)
		{
			displayAllMovingCams = !displayAllMovingCams;
			if (showImg)
				printLOG("ShowAllMovingCam: ON\n");
			else
				printLOG("ShowAllMovingCam: OFF\n");
		}
		if (strcmp(Fname, "showImage") == 0)
		{
			showImg = !showImg;
			if (showImg)
				printLOG("Image: ON\n");
			else
				printLOG("Image: OFF\n");
		}
		if (strcmp(Fname, "showSMPL") == 0)
		{
			showSMPL = !showSMPL;
			if (showSMPL)
				printLOG("SMPL: ON\n");
			else
				printLOG("SMPL: OFF\n");
		}
		if (strcmp(Fname, "showSkeleton") == 0)
		{
			showSkeleton = !showSkeleton;
			if (showSkeleton)
				printLOG("Skeleton: ON\n");
			else
				printLOG("Skeleton: OFF\n");
		}
		if (strcmp(Fname, "EvolvingTrajectory") == 0 || strcmp(Fname, "ETraj") == 0)
		{
			hasTimeEvoling3DPoints = !hasTimeEvoling3DPoints;
			if (hasTimeEvoling3DPoints)
				printLOG("Trajectory Time: ON\n");
			else
				printLOG("Trajectory Time: OFF\n");
		}
		if (strcmp(Fname, "TrajectoryTimeEnd") == 0 || strcmp(Fname, "TrajTimeEnd") == 0)
		{
			EndTime = !EndTime;
			if (EndTime)
				printLOG("Trajectory END Time : ON\n");
			else
				printLOG("Trajectory END Time :OFF\n");
		}
		if (strcmp(Fname, "IndiviualTrajectory") == 0 || strcmp(Fname, "IndiTraj") == 0)
		{
			IndiviualTrajectory = !IndiviualTrajectory;
			if (IndiviualTrajectory)
				printLOG("Indiviual Trajectory: ON\n");
			else
				printLOG("Indiviual Trajectory: OFF\n");
		}
		if (strcmp(Fname, "SCamTraj") == 0)
		{
			drawBriefCameraTraject = !drawBriefCameraTraject;
			if (drawBriefCameraTraject)
				printLOG("Simple camera trajectory: ON\n");
			else
				printLOG("Simple camera trajectory: OFF\n");
		}
		if (strcmp(Fname, "OneTime") == 0)
		{
			OneTimeInstanceOnly = !OneTimeInstanceOnly;
			if (OneTimeInstanceOnly)
				printLOG("Show only 1 time instance: ON\n");
			else
				printLOG("Show only 1 time instance:  OFF\n");
		}
		if (strcmp(Fname, "SwitchPose") == 0)
		{
			if (FirstPose)
				printLOG("Pose 2: OFF\n");
			else
				printLOG("Pose : ON\n");
			FirstPose = !FirstPose; SecondPose = !SecondPose;
		}
		break;
	case 'F':
		bFullsreen = !bFullsreen;
		if (bFullsreen)
			glutFullScreen();
		else
		{
			glutReshapeWindow(org_Width, org_Height);
			glutInitWindowPosition(0, 0);
		}
		break;
	case 'c':
		printLOG("Current cameraSize: %f. Please enter the new size: ", CameraSize);
		cin >> CameraSize;
		printLOG("New cameraSize: %f\n", CameraSize);
		break;
	case 'p':
		printLOG("Current pointSize: %f. Please enter the new size: ", pointSize);
		cin >> pointSize;
		printLOG("New pointSize: %f\n", pointSize);
		break;
	case 'u':
		printLOG("Current UnitScale: %f. Please enter the size: ", UnitScale);
		cin >> UnitScale;
		printLOG("New UnitScale: %f\n", UnitScale);
		break;
	case 'b':
		changeBackgroundColor = !changeBackgroundColor;
		break;

	case 'P':
		PickingMode = !PickingMode;
		if (PickingMode)
			printLOG("Picking Mode: ON\n");
		else
			printLOG("Picking Mode: OFF\n");
		break;
	case 'g':
		printLOG("Toggle ground plane display\n");
		showGroundPlane = !showGroundPlane;
		break;
	case 'r':
		AutoDisp = !AutoDisp;
		if (AutoDisp)
		{
			DisplayStartTime = omp_get_wtime();
			printLOG("Automatic diplay: ON\n");
		}
		else
			printLOG("Automatic diplay: OFF\n");

		break;
	case '1':
		printLOG("Toggle corpus points display\n");
		drawCorpusPoints = !drawCorpusPoints;
		break;
	case '2':
		drawCorpusCameras = !drawCorpusCameras;
		if (drawCorpusCameras)
			printLOG("Corpus cameras display: ON\n");
		else
			printLOG("Corpus cameras display: OFF\n");
		break;
	case '3':
		drawNonCorpusCameras = !drawNonCorpusCameras;
		if (drawNonCorpusCameras)
			printLOG("Corpus moving cameras display: ON\n");
		else
			printLOG("Corpus moving cameras display: OFF\n");
		break;
	case '4':
		printLOG("Save OpenGL viewing parameters\n");
		SaveStaticViewingParameters = true;
		break;
	case '5':
		printLOG("Read OpenGL viewing parameters\n");
		SetStaticViewingParameters = true;
		break;
	case '6':
		SaveDynamicViewingParameters = !SaveDynamicViewingParameters;
		if (SaveDynamicViewingParameters)
		{
			printLOG("Save OpenGL dynamic viewing parameters: ON\nStart pushing into stack");
			timeID = 0;
		}
		else
		{
			printLOG("Save OpenGL dynamic viewing parameters: OFF\n. Flush the stack out\n");
			sprintf(Fname, "%s/OpenGLDynamicViewingPara.txt", Path); FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)DynamicViewingParas.size(); ii++)
				fprintf(fp, "%d %.8f %d %d %.8f %.8f %.8f \n", DynamicViewingParas[ii].timeID, DynamicViewingParas[ii].viewDistance, DynamicViewingParas[ii].mouseXRotate, DynamicViewingParas[ii].mouseYRotate,
					DynamicViewingParas[ii].CentroidX, DynamicViewingParas[ii].CentroidY, DynamicViewingParas[ii].CentroidZ);
			fclose(fp);
		}
		DynamicViewingParas.clear();

		break;
	case '7':
		printLOG("Read OpenGL dynamic viewing parameters\n");
		SetDynamicViewingParameters = true;
		DynamicViewingParas.clear();
		break;
	case '8':
		SaveRendererViewingParameters = !SaveRendererViewingParameters;
		if (SaveRendererViewingParameters)
			printLOG("Save OpenGL Render viewing parameters: ON\nStart pushing into stack");

		break;
	case '9':
		printLOG("Read OpenGL Render viewing parameters\n");
		SetRendererViewingParameters = true;
		RenderViewingParas.clear();
		break;
	case 'n':
		printLOG("Toggle camera name display\n");
		drawCameraID = !drawCameraID;
		break;
	case 'A':
		printLOG("Toggle axis display\n");
		showAxis = !showAxis;
		break;
	case 's':
		SaveScreen = !SaveScreen;
		if (SaveScreen)
			printLOG("Save screen: ON\n");
		else
			printLOG("Save screen: OFF\n");
		break;
	case 'S':
		ImmediateSnap = !ImmediateSnap;
		if (ImmediateSnap)
			printLOG("Snap screen: ON\n");
		else
			printLOG("Snap screen: OFF\n");
		break;
	case 'a':
		g_mouseXRotate += 5;
		break;
	case 'd':
		g_mouseXRotate -= 5;
		break;
	case 'w':
		g_fViewDistance -= 10.0f*UnitScale;
		break;
	case 'x':
		g_fViewDistance += 10.0f*UnitScale;
		break;
	case 'f':
		Unorder3DPointsTwo = !Unorder3DPointsTwo;
		break;
	case 't':
		printLOG("Current VTthresh: %f. Please enter the new size: ", VTthresh);
		cin >> VTthresh;
		VTthresh = max(0.0, VTthresh);
		printLOG("New VTthresh: %f\n", VTthresh);
		break;
	case 'm':
		int cid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
		int refFid = MyFtoI(((CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x)*CamTimeInfo[refCid].x));

		break;
	}
	glutPostRedisplay();
}
void SpecialInput(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_INSERT:
		StateChanged = 1;
		break;

	case GLUT_KEY_PAGE_UP:
		TrialID++;
		if (TrialID > maxTrial)
			TrialID = maxTrial;
		printLOG("Current data trial: %d ...", TrialID);
		break;
	case GLUT_KEY_PAGE_DOWN:
		TrialID--;
		if (TrialID < 0)
			TrialID = 0;
		printLOG("Current data trial: %d ...", TrialID);
		break;
	case GLUT_KEY_UP:
		TrajecID++;
		if (TrajecID > OCRName.size()-1)
			TrajecID = OCRName.size()-1;
		printLOG("Current TrajecID: %d\n", TrajecID);
		break;
	case GLUT_KEY_DOWN:
		TrajecID--;
		if (TrajecID < 0)
			TrajecID = 0;
		printLOG("Current TrajecID: %d\n", TrajecID);
		break;
	case GLUT_KEY_LEFT:
		if (syncedMode == 1)
		{
			refFid -= increF;

			for (int ii = 0; ii < timeID + 1; ii++)
			{
				int cid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Fid;
				double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
				if (MyFtoI(ts*CamTimeInfo[refCid].x) == refFid)
				{
					timeID = ii;
					break;
				}
			}
		}
		else
			timeID -= increT;
		if (timeID < 0)
			timeID = 0;
		printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts, timeID);
		//PickedDynamicPoints.clear();
		break;
	case GLUT_KEY_RIGHT:
		if (syncedMode == 1)
		{
			refFid += increF;

			for (int ii = timeID; ii < (int)vAll3D_TidCidFid.size(); ii++)
			{
				int cid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Fid;
				double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
				if (MyFtoI(ts*CamTimeInfo[refCid].x) == refFid)
				{
					timeID = ii;
					break;
				}
			}
		}
		else
			timeID += increT;
		if (vAll3D_TidCidFid.size() > 0 && timeID > (int)vAll3D_TidCidFid.size() - 1)
			timeID = (int)vAll3D_TidCidFid.size() - 1;
		printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts, timeID);
		//PickedDynamicPoints.clear();
		break;
	case GLUT_KEY_HOME:
		timeID = 0;
		if (vAll3D_TidCidFid.size() > 0)
			printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts, timeID);
		PickedDynamicPoints.clear();
		break;
	case GLUT_KEY_END:
		if (vAll3D_TidCidFid.size() > 0)
		{
			timeID = (int)vAll3D_TidCidFid.size() - 1;
			printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts, timeID);
		}
		PickedDynamicPoints.clear();
		break;
	}

	glutPostRedisplay();
}
void MouseButton(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		g_bButton1Down = (state == GLUT_DOWN) ? true : false;
		g_xClick = x;
		g_yClick = y;

		if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
			g_camMode = CAM_ROTATE;
		else if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
			g_camMode = CAM_PAN;
		else if (glutGetModifiers() == GLUT_ACTIVE_ALT)
			g_camMode = CAM_ZOOM;
		else
		{
			g_camMode = CAM_DEFAULT;

			if (state == GLUT_DOWN) //picking single point
				SelectionFunction(x, y, false);
		}
	}
	else if (button == GLUT_RIGHT_BUTTON)
	{
		if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
		{
			g_camMode = CAM_DEFAULT;

			ReCenterNeeded = true;
			SelectionFunction(x, y, false);
		}
		if (glutGetModifiers() == GLUT_ACTIVE_ALT)//Deselect
		{
			printLOG("Deselect all picked objects\n");
			PickedCorpusCams.clear(), PickedNonCorpusCams.clear(), PickedStationaryPoints.clear(), PickedDynamicPoints.clear(), PickedTraject.clear(), PickPoint3D.clear();
		}
	}
	else if (button == GLUT_MIDDLE_BUTTON)
	{
		g_xClick = x;
		g_yClick = y;
		g_bButton1Down = true;
		g_camMode = CAM_ZOOM;
	}
}
void MouseMotion(int x, int y)
{
	if (g_bButton1Down)
	{
		if (zPress == 1)
		{
			modelAx = (modelAx + y - g_yClick) % 360;
			modelAy = (modelAy + x - g_xClick) % 360;
			zPress = 0;
		}
		if (g_camMode == CAM_ZOOM)
			g_fViewDistance += 5.0f*(y - g_yClick) *UnitScale;
		else if (g_camMode == CAM_ROTATE)
		{
			showAxis = true;
			g_mouseXRotate += (x - g_xClick);
			g_mouseYRotate -= (y - g_yClick);
			g_mouseXRotate = g_mouseXRotate % 360;
			g_mouseYRotate = g_mouseYRotate % 360;
		}
		else if (g_camMode == CAM_PAN)
		{
			showAxis = true;
			float dX = -(x - g_xClick)*UnitScale, dY = (y - g_yClick)*UnitScale;

			float cphi = cos(-Pi * g_mouseYRotate / 180), sphi = sin(-Pi * g_mouseYRotate / 180);
			float Rx[9] = { 1, 0, 0, 0, cphi, -sphi, 0, sphi, cphi };

			cphi = cos(-Pi * g_mouseXRotate / 180), sphi = sin(-Pi * g_mouseXRotate / 180);
			float Ry[9] = { cphi, 0, sphi, 0, 1, 0, -sphi, 0, cphi };

			float R[9];  mat_mul(Rx, Ry, R, 3, 3, 3);
			float incre[3], orgD[3] = { dX, dY, 0 }; mat_mul(R, orgD, incre, 3, 3, 1);

			PointsCentroid[0] += incre[0], PointsCentroid[1] += incre[1], PointsCentroid[2] += incre[2];
		}

		g_xClick = x, g_yClick = y;

		glutPostRedisplay();
	}
}
void ReshapeGL(int width, int height)
{
	g_Width = width;
	g_Height = height;
	glViewport(0, 0, g_Width, g_Height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	g_ratio = (float)g_Width / g_Height;
	gluPerspective(60.0, g_ratio, g_nearPlane, g_farPlane);
	glMatrixMode(GL_MODELVIEW);
}

void DrawStr(Point3f color, char* nameString)
{
	glColor3f(color.x, color.y, color.z);
	glRasterPos3d(0, 0, 0);
	for (int c = 0; c < 10; ++c)
	{
		if (nameString[c] == 0)
			break;
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, nameString[c]);
	}
}
void DrawCube(Point3f &length)
{
	glBegin(GL_QUADS);
	// Top face (z)
	glColor4f(0.0f, 1.0f, 0.0f, 0.3f);     // Green
	glVertex3f(-length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, length.z / 2);

	// Bottom face (-z)
	glColor4f(1.0f, 0.5f, 0.0f, 0.3f);     // Orange
	glVertex3f(-length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, -length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, -length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, -length.z / 2);

	// Front face  (x)
	glColor4f(1.0f, 0.0f, 0.0f, 0.3f);     // Red
	glVertex3f(length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, -length.z / 2);

	// Back face (-x)
	glColor4f(1.0f, 1.0f, 0.0f, 0.3f);     // Yellow
	glVertex3f(-length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, -length.z / 2);

	// Left face (y)
	glColor4f(0.0f, 0.0f, 1.0f, 0.3f);     // Blue
	glVertex3f(-length.x / 2, length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, length.y / 2, -length.z / 2);

	// Right face (-y)
	glColor4f(1.0f, 0.0f, 1.0f, 0.3f);     // Magenta
	glVertex3f(-length.x / 2, -length.y / 2, -length.z / 2);
	glVertex3f(-length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, length.z / 2);
	glVertex3f(length.x / 2, -length.y / 2, -length.z / 2);
	glEnd();  // End of drawing color-cube

}
void DrawSphere(double r, int lats, int longs)
{
	int i, j;
	for (i = 0; i <= lats; i++) {
		double lat0 = M_PI * (-0.5 + (double)(i - 1) / lats);
		double z0 = sin(lat0);
		double zr0 = cos(lat0);

		double lat1 = M_PI * (-0.5 + (double)i / lats);
		double z1 = sin(lat1);
		double zr1 = cos(lat1);

		glBegin(GL_QUAD_STRIP);
		for (j = 0; j <= longs; j++) {
			double lng = 2 * M_PI * (double)(j - 1) / longs;
			double x = cos(lng);
			double y = sin(lng);

			glNormal3f(x * zr0, y * zr0, z0);
			glVertex3f(r * x * zr0, r * y * zr0, r * z0);
			glNormal3f(x * zr1, y * zr1, z1);
			glVertex3f(r * x * zr1, r * y * zr1, r * z1);
		}
		glEnd();
	}
	return;
}

void Draw_Axes(void)
{
	glPushMatrix();

	glBegin(GL_LINES);
	glColor3f(1, 0, 0); // X axis is red.
	glVertex3f(0, 0, 0);
	glVertex3f(g_coordAxisLength * 5, 0, 0);
	glColor3f(0, 1, 0); // Y axis is green.
	glVertex3f(0, 0, 0);
	glVertex3f(0, g_coordAxisLength * 5, 0);
	glColor3f(0, 0, 1); // Z axis is blue.
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, g_coordAxisLength * 5);
	glEnd();

	glPopMatrix();
}
void DrawCamera(int highlight)
{
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	if (highlight == 0)
		glColor3fv(Blue);
	else if (highlight == 1)
		glColor3fv(Red);
	else
		glColor3fv(Green);

	glBegin(GL_LINES);
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize); //
	glEnd();

	if (highlight == 0)
		glColor3fv(Blue);
	else if (highlight == 1)
		glColor3fv(Red);
	else
		glColor3fv(Green);

	// we also has to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
	glBegin(GL_LINE_STRIP);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize);
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
}
void DrawCamera(GLfloat *color)
{
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glColor3fv(color);

	glBegin(GL_LINES);
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize); //
	glEnd();

	glColor3fv(color);

	// we also has to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
	glBegin(GL_LINE_STRIP);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 2.0 * CameraSize);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 2.0 * CameraSize);
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
}
void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D, GLfloat *color)
{
	double x = x2 - x1;
	double y = y2 - y1;
	double z = z2 - z1;
	double L = sqrt(x*x + y * y + z * z);

	GLUquadricObj *quadObj;

	glPushMatrix();
	glColor3fv(color);
	glTranslated(x1, y1, z1);

	if (x != 0.f || y != 0.f)
	{
		glRotated(atan2(y, x) / RADPERDEG, 0., 0., 1.);
		glRotated(atan2(sqrt(x*x + y * y), z) / RADPERDEG, 0., 1., 0.);
	}
	else if (z < 0)
		glRotated(180, 1., 0., 0.);

	glTranslatef(0, 0, L - 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, 2 * D, 0.0, 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, 2 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	glTranslatef(0, 0, -L + 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, D, D, L - 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, D, 32, 1);
	gluDeleteQuadric(quadObj);

	glPopMatrix();

}
void RenderGroundPlane()
{
	int gridNum = 6;
	double width = 1000;
	double halfWidth = width / 2;
	Point3f origin(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]);
	Point3f axis1 = Point3f(1.0f, 0.0f, 0.0f)* width;
	Point3f axis2 = Point3f(0.0f, 0.0f, 1.0f) * width;
	glBegin(GL_QUADS);
	for (int y = -gridNum; y <= gridNum; ++y)
		for (int x = -gridNum; x <= gridNum; ++x)
		{
			if ((x + y) % 2 == 0)
				continue;
			else
				glColor4f(0.7, 0.7, 0.7, 0.9);

			Point3f p1 = origin + axis1 * x + axis2 * y;
			Point3f p2 = p1 + axis1;
			Point3f p3 = p1 + axis2;
			Point3f p4 = p1 + axis1 + axis2;

			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p1.x, p1.y, p1.z);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p2.x, p2.y, p2.z);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p4.x, p4.y, p4.z);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p3.x, p3.y, p3.z);
		}
	glEnd();

}
void RenderSkeleton(vector<Point3d> &pt3D, GLfloat *color)
{
	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	int i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 2; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 4; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 5; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 6; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 3; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 2; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 7; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 8; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 9; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 1; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 12; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 10; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 11; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();
}
void RenderSkeleton2(vector<Point3d> &pt3D, GLfloat *color)
{
	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	int i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 1; i < 6; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 6; i < 11; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 11; i < 17; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 14; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 24; i < 31; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 14; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 17; i < 24; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();
}
void RenderSkeleton3(vector<Point3d> &pt3D, GLfloat *color)
{
	int i = 1;

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 1, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 4, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 7, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 10, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 2, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 5, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 8, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 11, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 3, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 6, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 9, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 12, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 15, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 9, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 13, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 16, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 18, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 20, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 22, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 9, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 14, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 17, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 19, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 21, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	i = 23, glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
	glEnd();
	glPopMatrix();

}
void RenderIntraFace(vector<Point3f> &pt3D, GLfloat *color)
{
	for (int j = 0; j < (int)pt3D.size() / 49; j++)
	{
		int i;
		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j; i < 49 * j + 4; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 5; i < 49 * j + 9; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 10; i < 49 * j + 13; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 14; i < 49 * j + 18; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 19; i < 49 * j + 24; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		i = 49 * j + 19; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 25; i < 49 * j + 30; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		i = 49 * j + 25; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 31; i < 49 * j + 42; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		i = 49 * j + 31; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();

		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (i = 49 * j + 43; i < 49 * j + 48; i++)
			glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		i = 49 * j + 43; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]), glColor3fv(color);
		glEnd();
		glPopMatrix();
	}
}
void Render_MPI_3D_Skeleton(vector<Point3f> &pt3D, GLfloat *color)
{
	glLineWidth(3.0);

	for (int j = 0; j < (int)pt3D.size() / 14; j++)
	{
		int i;
		vector<bool> validJoint(14);
		//head-neck
		if (pt3D[14 * j].x != 0 && pt3D[1 + 14 * j].x != 0)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			glColor3fv(color); i = 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
			glColor3fv(color); i = 1 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
			glEnd();
			glPopMatrix();
			validJoint[0] = 1, validJoint[1] = 1;
		}

		//left arm
		if (pt3D[2 + 14 * j].x != 0 && pt3D[3 + 14 * j].x != 0)
		{
			if (norm(pt3D[2 + 14 * j] - pt3D[3 + 14 * j]) < 0.9)
			{
				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				i = 2 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				i = 3 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				glEnd();
				glPopMatrix();
				validJoint[2] = 1, validJoint[3] = 1;
			}
		}
		if (pt3D[3 + 14 * j].x != 0 && pt3D[4 + 14 * j].x != 0)
		{
			if (norm(pt3D[2 + 14 * j] - pt3D[3 + 14 * j]) < 0.9)
			{
				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				i = 3 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				i = 4 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				glEnd();
				glPopMatrix();
				validJoint[3] = 1, validJoint[4] = 1;
			}
		}

		//right arm
		if (pt3D[5 + 14 * j].x != 0 && pt3D[6 + 14 * j].x != 0)
		{
			if (norm(pt3D[5 + 14 * j] - pt3D[6 + 14 * j]) < 0.9)
			{
				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				i = 5 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				i = 6 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				glEnd();
				glPopMatrix();
				validJoint[5] = 1, validJoint[6] = 1;
			}
		}
		if (pt3D[6 + 14 * j].x != 0 && pt3D[7 + 14 * j].x != 0)
		{
			if (norm(pt3D[6 + 14 * j] - pt3D[7 + 14 * j]) < 0.9)
			{
				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				i = 6 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				i = 7 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
				glEnd();
				glPopMatrix();
				validJoint[7] = 1, validJoint[6] = 1;
			}
		}

		//left leg
		if (pt3D[8 + 14 * j].x != 0 && pt3D[9 + 14 * j].x != 0)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			i = 8 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			i = 9 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			glEnd();
			glPopMatrix();
			validJoint[8] = 1, validJoint[9] = 1;
		}
		if (pt3D[9 + 14 * j].x != 0 && pt3D[10 + 14 * j].x != 0)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			i = 9 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			i = 10 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			glEnd();
			glPopMatrix();
			validJoint[9] = 1, validJoint[10] = 1;
		}

		//right leg
		if (pt3D[11 + 14 * j].x != 0 && pt3D[12 + 14 * j].x != 0)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			i = 11 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			i = 12 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			glEnd();
			glPopMatrix();
			validJoint[11] = 1, validJoint[12] = 1;
		}
		if (pt3D[12 + 14 * j].x != 0 && pt3D[13 + 14 * j].x != 0)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			i = 12 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			i = 13 + 14 * j; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]); glColor3fv(color);
			glEnd();
			glPopMatrix();
			validJoint[13] = 1, validJoint[12] = 1;
		}

		for (int i = 0; i < 14; i++)
		{
			if (pt3D[i + 14 * j].x != 0.0 && validJoint[i])
			{
				glPushMatrix();
				glTranslatef(pt3D[i + 14 * j].x - PointsCentroid[0], pt3D[i + 14 * j].y - PointsCentroid[1], pt3D[i + 14 * j].z - PointsCentroid[2]);
				glColor3fv(color);
				glutSolidSphere(pointSize * 2, 10, 10);
				glPopMatrix();
			}
		}
	}
	glLineWidth(1.0);
}
void Render_COCO_3D_Skeleton(vector<Point3f> &pt3D, GLfloat *color, int PointFormat)
{
	glLineWidth(pointSize / 5.f);

	int TorsoJointId[4];
	if (PointFormat == 17)
		TorsoJointId[0] = 5, TorsoJointId[1] = 6, TorsoJointId[2] = 11, TorsoJointId[3] = 12;
	else if (PointFormat == 18)
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 8, TorsoJointId[3] = 11;
	else
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 9, TorsoJointId[3] = 12;

	vector<double> *MedianLength = new vector<double>[PointFormat - 1];
	if (PointFormat == 17)
	{
		if (!hasMedLengthInfo)
		{
			//determine mean bone length
			for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
			{
				//nose-leye
				int i0 = PointFormat * j, i1 = 1 + PointFormat * j;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[0].push_back(leng);
				}
				//leye-lear
				i0 = PointFormat * j + 1, i1 = PointFormat * j + 3;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[1].push_back(leng);
				}

				//nose-reye
				i0 = PointFormat * j, i1 = PointFormat * j + 2;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[0].push_back(leng);
				}
				//reye-rear
				i0 = PointFormat * j + 2, i1 = PointFormat * j + 4;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[1].push_back(leng);
				}

				//left arm
				i0 = PointFormat * j + 0, i1 = PointFormat * j + 5;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[2].push_back(leng);
				}
				i0 = PointFormat * j + 5, i1 = PointFormat * j + 7;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[3].push_back(leng);
				}
				i0 = PointFormat * j + 7, i1 = PointFormat * j + 9;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[4].push_back(leng);
				}

				//right arm
				i0 = PointFormat * j + 0, i1 = PointFormat * j + 6;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[2].push_back(leng);
				}
				i0 = PointFormat * j + 6, i1 = PointFormat * j + 8;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[3].push_back(leng);
				}
				i0 = PointFormat * j + 8, i1 = PointFormat * j + 10;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[4].push_back(leng);
				}

				//left leg
				i0 = PointFormat * j + 5, i1 = PointFormat * j + 11;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[5].push_back(leng);
				}
				i0 = PointFormat * j + 11, i1 = PointFormat * j + 13;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[6].push_back(leng);
				}
				i0 = PointFormat * j + 13, i1 = PointFormat * j + 15;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[7].push_back(leng);
				}

				//right leg
				i0 = PointFormat * j + 6, i1 = PointFormat * j + 12;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[5].push_back(leng);
				}
				i0 = PointFormat * j + 12, i1 = PointFormat * j + 14;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[6].push_back(leng);
				}
				i0 = PointFormat * j + 14, i1 = PointFormat * j + 16;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[7].push_back(leng);
				}

				//left-right shoulder
				i0 = PointFormat * j + 5, i1 = PointFormat * j + 6;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[8].push_back(leng);
				}

				//left-right hip
				i0 = PointFormat * j + 11, i1 = PointFormat * j + 12;
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[9].push_back(leng);
				}
			}

			double maxLimbLength = 0;
			for (int ii = 0; ii < 10; ii++)
			{
				if (MedianLength[ii].size() > 0)
					medLengh[ii] = Median(MedianLength[ii]), maxLimbLength = std::max(maxLimbLength, medLengh[ii]);
				else
					medLengh[ii] = 500.0 * UnitScale;
			}
			Discontinuity3DThresh = std::min(maxLimbLength, 500.0 * UnitScale);
		}

		for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
		{
			vector<bool> validJoint(PointFormat);
			//nose-leye
			int i0 = PointFormat * j, i1 = PointFormat * j + 1;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[0])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			//leye-lear
			i0 = PointFormat * j + 1, i1 = PointFormat * j + 3;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[1])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//nose-reye
			i0 = PointFormat * j, i1 = PointFormat * j + 2;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[0])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			//reye-rear
			i0 = PointFormat * j + 2, i1 = PointFormat * j + 4;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[1])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//left arm
			i0 = PointFormat * j + 0, i1 = PointFormat * j + 5;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[2])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 5, i1 = PointFormat * j + 7;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[3])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 7, i1 = PointFormat * j + 9;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[4])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//right arm
			i0 = PointFormat * j + 0, i1 = PointFormat * j + 6;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[2])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 6, i1 = PointFormat * j + 8;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[3])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 8, i1 = PointFormat * j + 10;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[4])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//left leg
			i0 = PointFormat * j + 5, i1 = PointFormat * j + 11;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[5])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 11, i1 = PointFormat * j + 13;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[6])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 13, i1 = PointFormat * j + 15;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[7])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//right leg
			i0 = PointFormat * j + 6, i1 = PointFormat * j + 12;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[5])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 12, i1 = PointFormat * j + 14;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[6])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}
			i0 = PointFormat * j + 14, i1 = PointFormat * j + 16;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[7])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//left right shoulder
			i0 = PointFormat * j + 5, i1 = PointFormat * j + 6;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[8])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//left right hip
			i0 = PointFormat * j + 11, i1 = PointFormat * j + 12;
			if (pt3D[i0].x != 0 && pt3D[i1].x != 0)
			{
				if (norm(pt3D[i0] - pt3D[i1]) < 14 * medLengh[9])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					glColor3fv(color); glVertex3f(pt3D[i0].x - PointsCentroid[0], pt3D[i0].y - PointsCentroid[1], pt3D[i0].z - PointsCentroid[2]);
					glColor3fv(color); glVertex3f(pt3D[i1].x - PointsCentroid[0], pt3D[i1].y - PointsCentroid[1], pt3D[i1].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			int once = 1;
			for (int i = 0; i < PointFormat; i++)
			{
				if (pt3D[i + PointFormat * j].x != 0.0)
				{
					glPushMatrix();
					glColor3fv(color);
					glTranslatef(pt3D[i + PointFormat * j].x - PointsCentroid[0], pt3D[i + PointFormat * j].y - PointsCentroid[1], pt3D[i + PointFormat * j].z - PointsCentroid[2]);

					glutSolidSphere(pointSize / 50.f, 10, 10);

					if (once == 1)
					{
						once = 0;
						char name[512];
						sprintf(name, "%d", j);
						DrawStr(Point3f(0, 1, 0), name);
					}
					glPopMatrix();
				}
			}
		}
		glLineWidth(1.0);
	}
	else if (PointFormat == 18)
	{
		if (!hasMedLengthInfo)
		{
			//determine mean bone length
			for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
			{
				//head-neck
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[0].push_back(leng);
				}

				//left arm
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[2 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[2 + PointFormat * j]);
					MedianLength[1].push_back(leng);
				}
				if (pt3D[2 + PointFormat * j].x != 0 && pt3D[3 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[3 + PointFormat * j] - pt3D[2 + PointFormat * j]);
					MedianLength[2].push_back(leng);
				}
				if (pt3D[3 + PointFormat * j].x != 0 && pt3D[4 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[4 + PointFormat * j]);
					MedianLength[3].push_back(leng);
				}

				//right arm
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[5 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[5 + PointFormat * j]);
					MedianLength[4].push_back(leng);
				}
				if (pt3D[5 + PointFormat * j].x != 0 && pt3D[6 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[6 + PointFormat * j] - pt3D[5 + PointFormat * j]);
					MedianLength[5].push_back(leng);
				}
				if (pt3D[6 + PointFormat * j].x != 0 && pt3D[7 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[6 + PointFormat * j] - pt3D[7 + PointFormat * j]);
					MedianLength[6].push_back(leng);
				}

				//left leg
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[8 + PointFormat * j]);
					MedianLength[7].push_back(leng);
				}
				if (pt3D[8 + PointFormat * j].x != 0 && pt3D[9 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[8 + PointFormat * j] - pt3D[9 + PointFormat * j]);
					MedianLength[8].push_back(leng);
				}
				if (pt3D[9 + PointFormat * j].x != 0 && pt3D[10 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[9 + PointFormat * j] - pt3D[10 + PointFormat * j]);
					MedianLength[9].push_back(leng);
				}

				//right leg
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[11 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[11 + PointFormat * j]);
					MedianLength[10].push_back(leng);
				}
				if (pt3D[11 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[11 + PointFormat * j] - pt3D[12 + PointFormat * j]);
					MedianLength[11].push_back(leng);
				}
				if (pt3D[12 + PointFormat * j].x != 0 && pt3D[13 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[12 + PointFormat * j] - pt3D[13 + PointFormat * j]);
					MedianLength[12].push_back(leng);
				}

				//right eye+ ear
				if (pt3D[0 + PointFormat * j].x != 0 && pt3D[14 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[0 + PointFormat * j] - pt3D[14 + PointFormat * j]);
					MedianLength[13].push_back(leng);
				}
				if (pt3D[14 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[16 + PointFormat * j] - pt3D[14 + PointFormat * j]);
					MedianLength[14].push_back(leng);
				}

				//left eye+ ear
				if (pt3D[0 + PointFormat * j].x != 0 && pt3D[15 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[0 + PointFormat * j] - pt3D[15 + PointFormat * j]);
					MedianLength[15].push_back(leng);
				}
				if (pt3D[15 + PointFormat * j].x != 0 && pt3D[17 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[17 + PointFormat * j] - pt3D[15 + PointFormat * j]);
					MedianLength[16].push_back(leng);
				}
			}
			double maxLimbLength = 0;
			for (int ii = 0; ii < PointFormat - 1; ii++)
			{
				if (MedianLength[ii].size() > 0)
					medLengh[ii] = Median(MedianLength[ii]), maxLimbLength = max(maxLimbLength, medLengh[ii]);
				else
					medLengh[ii] = 500.0 * UnitScale;
			}
			Discontinuity3DThresh = min(maxLimbLength, 500.0 * UnitScale);
		}

		for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
		{
			int i;
			if (vPeopleId[j] == 0 && IsValid3D(pt3D[PointFormat * j + TorsoJointId[0]]) && IsValid3D(pt3D[PointFormat * j + TorsoJointId[1]]) && IsValid3D(pt3D[PointFormat * j + TorsoJointId[2]]) && IsValid3D(pt3D[PointFormat * j + TorsoJointId[3]]))
			{
				int pointPair[8] = { 0, 1, 0, 2, 1, 3, 2, 3 };
				for (int k = 0; k < 4; k++)
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = TorsoJointId[pointPair[2 * k]] + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = TorsoJointId[pointPair[2 * k + 1]] + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
				}

				double vec01[3] = { pt3D[PointFormat * j + TorsoJointId[0]].x - pt3D[PointFormat * j + TorsoJointId[1]].x,
				pt3D[PointFormat * j + TorsoJointId[0]].y - pt3D[PointFormat * j + TorsoJointId[1]].y,
				pt3D[PointFormat * j + TorsoJointId[0]].z - pt3D[PointFormat * j + TorsoJointId[1]].z };
				double vec13[3] = { pt3D[PointFormat * j + TorsoJointId[1]].x - pt3D[PointFormat * j + TorsoJointId[3]].x,
				pt3D[PointFormat * j + TorsoJointId[1]].y - pt3D[PointFormat * j + TorsoJointId[3]].y,
				pt3D[PointFormat * j + TorsoJointId[1]].z - pt3D[PointFormat * j + TorsoJointId[3]].z };
				normalize(vec01);
				normalize(vec13);

				double normalVec_[3];
				cross_product(vec01, vec13, normalVec_);
				normalize(normalVec_);

				double meanTorso[3] = { 0, 0,0 };
				for (int ii = 0; ii < 4; ii++)
					meanTorso[0] += pt3D[PointFormat * j + TorsoJointId[ii]].x,
					meanTorso[1] += pt3D[PointFormat * j + TorsoJointId[ii]].y,
					meanTorso[2] += pt3D[PointFormat * j + TorsoJointId[ii]].z;
				meanTorso[0] /= 4.0, meanTorso[1] /= 4.0, meanTorso[2] /= 4.0;

				vector<Point3d> vpts;
				for (int ii = 0; ii < 4; ii++)
					vpts.push_back(pt3D[PointFormat * j + TorsoJointId[ii]]);

				int npts = 4;
				double centerX = 0, centerY = 0, centerZ = 0;
				for (int ii = 0; ii < npts; ii++)
					centerX += vpts[ii].x, centerY += vpts[ii].y, centerZ += vpts[ii].z;
				centerX /= npts, centerY /= npts, centerZ /= npts;

				MatrixXdr X_(npts, 3);
				for (int ii = 0; ii < npts; ii++)
					X_(ii, 0) = vpts[ii].x - centerX, X_(ii, 1) = vpts[ii].y - centerY, X_(ii, 2) = vpts[ii].z - centerZ;

				MatrixXdr Cov = X_.transpose()*X_;
				JacobiSVD<MatrixXdr> svd(Cov, ComputeFullV);
				MatrixXdr V = svd.matrixV();

				double normalVec[3] = { V(0, 2),  V(1, 2),V(2, 2) };
				normalize(normalVec);
				if (dotProduct(normalVec, normalVec_) > 0)
					for (int ii = 0; ii < 3; ii++)
						normalVec[ii] *= -1.0;

				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				glColor3fv(Green), glVertex3f(meanTorso[0] - PointsCentroid[0], meanTorso[1] - PointsCentroid[1], meanTorso[2] - PointsCentroid[2]);
				glColor3fv(Green), glVertex3f(meanTorso[0] + normalVec[0] - PointsCentroid[0], meanTorso[1] + normalVec[1] - PointsCentroid[1], meanTorso[2] + normalVec[2] - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
			}

			vector<bool> validJoint(PointFormat);
			//head-neck
			if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]) < 1.4*medLengh[0])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//left arm
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[2 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[2 + PointFormat * j]) < 1.4*medLengh[1])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 2 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[2] = 1;
				}
			}
			if (pt3D[2 + PointFormat * j].x != 0 && pt3D[3 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[3 + PointFormat * j] - pt3D[2 + PointFormat * j]) < 1.4*medLengh[2])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 2 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 3 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[2] = 1, validJoint[3] = 1;
				}
			}
			if (pt3D[3 + PointFormat * j].x != 0 && pt3D[4 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[4 + PointFormat * j]) < 1.4*medLengh[3])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 3 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 4 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[3] = 1, validJoint[4] = 1;
				}
			}

			//right arm
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[5 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[5 + PointFormat * j]) < 1.4*medLengh[4])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 5 + PointFormat * j; glColor3fv(color); glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[5] = 1;
				}
			}
			if (pt3D[5 + PointFormat * j].x != 0 && pt3D[6 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[6 + PointFormat * j] - pt3D[5 + PointFormat * j]) < 1.4*medLengh[5])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 5 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 6 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[5] = 1, validJoint[6] = 1;
				}
			}
			if (pt3D[6 + PointFormat * j].x != 0 && pt3D[7 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[6 + PointFormat * j] - pt3D[7 + PointFormat * j]) < 1.4*medLengh[6])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 6 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 7 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[6] = 1, validJoint[7] = 1;
				}
			}

			//left leg
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[8 + PointFormat * j]) < 1.4*medLengh[7])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[8] = 1;
				}
			}
			if (pt3D[8 + PointFormat * j].x != 0 && pt3D[9 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[8 + PointFormat * j] - pt3D[9 + PointFormat * j]) < 1.4*medLengh[8])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 9 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[8] = 1, validJoint[9] = 1;
				}
			}
			if (pt3D[9 + PointFormat * j].x != 0 && pt3D[10 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[9 + PointFormat * j] - pt3D[10 + PointFormat * j]) < 1.4*medLengh[9])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 9 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 10 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[9] = 1, validJoint[10] = 1;
				}
			}

			//right leg
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[11 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[11 + PointFormat * j]) < 1.4*medLengh[10])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[11] = 1;
				}
			}
			if (pt3D[11 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[11 + PointFormat * j] - pt3D[12 + PointFormat * j]) < 1.4*medLengh[11])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 12 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[11] = 1, validJoint[12] = 1;
				}
			}
			if (pt3D[12 + PointFormat * j].x != 0 && pt3D[13 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[12 + PointFormat * j] - pt3D[13 + PointFormat * j]) < 1.4*medLengh[12])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 12 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 13 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[13] = 1, validJoint[12] = 1;
				}
			}

			//right eye+ ear
			if (pt3D[0 + PointFormat * j].x != 0 && pt3D[14 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[0 + PointFormat * j] - pt3D[14 + PointFormat * j]) < 1.4*medLengh[13])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 0 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[14] = 1;
				}
			}
			if (pt3D[14 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[16 + PointFormat * j] - pt3D[14 + PointFormat * j]) < 1.4*medLengh[14])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 16 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[14] = 1, validJoint[16] = 1;
				}
			}

			//left eye+ ear
			if (pt3D[0 + PointFormat * j].x != 0 && pt3D[15 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[0 + PointFormat * j] - pt3D[15 + PointFormat * j]) < 1.4*medLengh[15])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 0 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 15 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[15] = 1;
				}
			}
			if (pt3D[15 + PointFormat * j].x != 0 && pt3D[17 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[17 + PointFormat * j] - pt3D[15 + PointFormat * j]) < 1.4*medLengh[16])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 15 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 17 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[15] = 1, validJoint[17] = 1;
				}
			}

			int once = 1;
			for (int i = 0; i < PointFormat; i++)
			{
				if (pt3D[i + PointFormat * j].x != 0.0)
				{
					glPushMatrix();
					glColor3fv(color);
					glTranslatef(pt3D[i + PointFormat * j].x - PointsCentroid[0], pt3D[i + PointFormat * j].y - PointsCentroid[1], pt3D[i + PointFormat * j].z - PointsCentroid[2]);

					glutSolidSphere(pointSize, 10, 10);

					if (once == 1)
					{
						once = 0;
						char name[512];
						sprintf(name, "%d", j);
						DrawStr(Point3f(0, 1, 0), name);
					}
					glPopMatrix();
				}
			}
		}
		glLineWidth(1.0);
	}
	else if (PointFormat == 25)
	{
		if (!hasMedLengthInfo)
		{
			//determine mean bone length
			for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
			{
				//head-neck
				if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
					MedianLength[0].push_back(leng);
				}

				//left arm
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[2 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[2 + PointFormat * j]);
					MedianLength[1].push_back(leng);
				}
				if (pt3D[2 + PointFormat * j].x != 0 && pt3D[3 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[3 + PointFormat * j] - pt3D[2 + PointFormat * j]);
					MedianLength[2].push_back(leng);
				}
				if (pt3D[3 + PointFormat * j].x != 0 && pt3D[4 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[4 + PointFormat * j]);
					MedianLength[3].push_back(leng);
				}

				//right arm
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[5 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[5 + PointFormat * j]);
					MedianLength[4].push_back(leng);
				}
				if (pt3D[5 + PointFormat * j].x != 0 && pt3D[6 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[6 + PointFormat * j] - pt3D[5 + PointFormat * j]);
					MedianLength[5].push_back(leng);
				}
				if (pt3D[6 + PointFormat * j].x != 0 && pt3D[7 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[6 + PointFormat * j] - pt3D[7 + PointFormat * j]);
					MedianLength[6].push_back(leng);
				}

				//neck mid-hip
				if (pt3D[1 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[1 + PointFormat * j] - pt3D[8 + PointFormat * j]);
					MedianLength[7].push_back(leng);
				}

				//left leg
				if (pt3D[9 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[9 + PointFormat * j] - pt3D[8 + PointFormat * j]);
					MedianLength[8].push_back(leng);
				}
				if (pt3D[10 + PointFormat * j].x != 0 && pt3D[9 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[10 + PointFormat * j] - pt3D[9 + PointFormat * j]);
					MedianLength[9].push_back(leng);
				}
				if (pt3D[11 + PointFormat * j].x != 0 && pt3D[10 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[11 + PointFormat * j] - pt3D[10 + PointFormat * j]);
					MedianLength[10].push_back(leng);
				}

				//right leg
				if (pt3D[8 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[8 + PointFormat * j] - pt3D[12 + PointFormat * j]);
					MedianLength[11].push_back(leng);
				}
				if (pt3D[13 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[13 + PointFormat * j] - pt3D[12 + PointFormat * j]);
					MedianLength[12].push_back(leng);
				}
				if (pt3D[14 + PointFormat * j].x != 0 && pt3D[13 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[14 + PointFormat * j] - pt3D[13 + PointFormat * j]);
					MedianLength[13].push_back(leng);
				}

				//right eye+ ear
				if (pt3D[0 + PointFormat * j].x != 0 && pt3D[15 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[0 + PointFormat * j] - pt3D[15 + PointFormat * j]);
					MedianLength[14].push_back(leng);
				}
				if (pt3D[15 + PointFormat * j].x != 0 && pt3D[17 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[17 + PointFormat * j] - pt3D[15 + PointFormat * j]);
					MedianLength[15].push_back(leng);
				}

				//left eye+ ear
				if (pt3D[0 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[0 + PointFormat * j] - pt3D[16 + PointFormat * j]);
					MedianLength[16].push_back(leng);
				}
				if (pt3D[18 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[16 + PointFormat * j] - pt3D[18 + PointFormat * j]);
					MedianLength[17].push_back(leng);
				}

				//left feet
				if (pt3D[11 + PointFormat * j].x != 0 && pt3D[22 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[11 + PointFormat * j] - pt3D[22 + PointFormat * j]);
					MedianLength[18].push_back(leng);
				}
				if (pt3D[11 + PointFormat * j].x != 0 && pt3D[23 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[11 + PointFormat * j] - pt3D[23 + PointFormat * j]);
					MedianLength[19].push_back(leng);
				}
				if (pt3D[11 + PointFormat * j].x != 0 && pt3D[24 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[11 + PointFormat * j] - pt3D[24 + PointFormat * j]);
					MedianLength[20].push_back(leng);
				}

				//right feet
				if (pt3D[14 + PointFormat * j].x != 0 && pt3D[19 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[14 + PointFormat * j] - pt3D[19 + PointFormat * j]);
					MedianLength[21].push_back(leng);
				}
				if (pt3D[14 + PointFormat * j].x != 0 && pt3D[20 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[14 + PointFormat * j] - pt3D[20 + PointFormat * j]);
					MedianLength[22].push_back(leng);
				}
				if (pt3D[14 + PointFormat * j].x != 0 && pt3D[21 + PointFormat * j].x != 0)
				{
					double leng = norm(pt3D[14 + PointFormat * j] - pt3D[21 + PointFormat * j]);
					MedianLength[23].push_back(leng);
				}
			}
			double maxLimbLength = 0;
			for (int ii = 0; ii < PointFormat - 1; ii++)
			{
				if (MedianLength[ii].size() > 0)
					medLengh[ii] = Median(MedianLength[ii]), maxLimbLength = max(maxLimbLength, medLengh[ii]);
				else
					medLengh[ii] = 500.0 * UnitScale;
			}
			Discontinuity3DThresh = min(maxLimbLength, 500.0 * UnitScale);
		}

		for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
		{
			int i;
			if (vPeopleId[j] == 0 && IsValid3D(pt3D[PointFormat * j + TorsoJointId[0]]) && IsValid3D(pt3D[PointFormat * j + TorsoJointId[1]]) && IsValid3D(pt3D[PointFormat * j + TorsoJointId[2]]) && IsValid3D(pt3D[PointFormat * j + TorsoJointId[3]]))
			{
				int pointPair[8] = { 0, 1, 0, 2, 1, 3, 2, 3 };
				for (int k = 0; k < 4; k++)
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = TorsoJointId[pointPair[2 * k]] + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = TorsoJointId[pointPair[2 * k + 1]] + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
				}

				double vec01[3] = { pt3D[PointFormat * j + TorsoJointId[0]].x - pt3D[PointFormat * j + TorsoJointId[1]].x,
					pt3D[PointFormat * j + TorsoJointId[0]].y - pt3D[PointFormat * j + TorsoJointId[1]].y,
					pt3D[PointFormat * j + TorsoJointId[0]].z - pt3D[PointFormat * j + TorsoJointId[1]].z };
				double vec13[3] = { pt3D[PointFormat * j + TorsoJointId[1]].x - pt3D[PointFormat * j + TorsoJointId[3]].x,
					pt3D[PointFormat * j + TorsoJointId[1]].y - pt3D[PointFormat * j + TorsoJointId[3]].y,
					pt3D[PointFormat * j + TorsoJointId[1]].z - pt3D[PointFormat * j + TorsoJointId[3]].z };
				normalize(vec01);
				normalize(vec13);

				double normalVec_[3];
				cross_product(vec01, vec13, normalVec_);
				normalize(normalVec_);

				double meanTorso[3] = { 0, 0,0 };
				for (int ii = 0; ii < 4; ii++)
					meanTorso[0] += pt3D[PointFormat * j + TorsoJointId[ii]].x,
					meanTorso[1] += pt3D[PointFormat * j + TorsoJointId[ii]].y,
					meanTorso[2] += pt3D[PointFormat * j + TorsoJointId[ii]].z;
				meanTorso[0] /= 4.0, meanTorso[1] /= 4.0, meanTorso[2] /= 4.0;

				vector<Point3d> vpts;
				for (int ii = 0; ii < 4; ii++)
					vpts.push_back(pt3D[PointFormat * j + TorsoJointId[ii]]);

				int npts = 4;
				double centerX = 0, centerY = 0, centerZ = 0;
				for (int ii = 0; ii < npts; ii++)
					centerX += vpts[ii].x, centerY += vpts[ii].y, centerZ += vpts[ii].z;
				centerX /= npts, centerY /= npts, centerZ /= npts;

				MatrixXdr X_(npts, 3);
				for (int ii = 0; ii < npts; ii++)
					X_(ii, 0) = vpts[ii].x - centerX, X_(ii, 1) = vpts[ii].y - centerY, X_(ii, 2) = vpts[ii].z - centerZ;

				MatrixXdr Cov = X_.transpose()*X_;
				JacobiSVD<MatrixXdr> svd(Cov, ComputeFullV);
				MatrixXdr V = svd.matrixV();

				double normalVec[3] = { V(0, 2),  V(1, 2),V(2, 2) };
				normalize(normalVec);
				if (dotProduct(normalVec, normalVec_) > 0)
					for (int ii = 0; ii < 3; ii++)
						normalVec[ii] *= -1.0;

				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				glColor3fv(Green), glVertex3f(meanTorso[0] - PointsCentroid[0], meanTorso[1] - PointsCentroid[1], meanTorso[2] - PointsCentroid[2]);
				glColor3fv(Green), glVertex3f(meanTorso[0] + normalVec[0] - PointsCentroid[0], meanTorso[1] + normalVec[1] - PointsCentroid[1], meanTorso[2] + normalVec[2] - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
			}

			vector<bool> validJoint(PointFormat);
			//head-neck
			if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]) < 1.4*medLengh[0])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[1] = 1;
				}
			}

			//left arm
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[2 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[2 + PointFormat * j]) < 1.4*medLengh[1])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 2 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[2] = 1;
				}
			}
			if (pt3D[2 + PointFormat * j].x != 0 && pt3D[3 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[3 + PointFormat * j] - pt3D[2 + PointFormat * j]) < 1.4*medLengh[2])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 2 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 3 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[2] = 1, validJoint[3] = 1;
				}
			}
			if (pt3D[3 + PointFormat * j].x != 0 && pt3D[4 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[4 + PointFormat * j]) < 1.4*medLengh[3])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 3 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 4 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[3] = 1, validJoint[4] = 1;
				}
			}

			//right arm
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[5 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[5 + PointFormat * j]) < 1.4*medLengh[4])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 5 + PointFormat * j; glColor3fv(color); glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[5] = 1;
				}
			}
			if (pt3D[5 + PointFormat * j].x != 0 && pt3D[6 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[6 + PointFormat * j] - pt3D[5 + PointFormat * j]) < 1.4*medLengh[5])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 5 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 6 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[5] = 1, validJoint[6] = 1;
				}
			}
			if (pt3D[6 + PointFormat * j].x != 0 && pt3D[7 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[6 + PointFormat * j] - pt3D[7 + PointFormat * j]) < 1.4*medLengh[6])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 6 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 7 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[6] = 1, validJoint[7] = 1;
				}
			}

			//neck-mid-hip
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[1 + PointFormat * j] - pt3D[8 + PointFormat * j]) < 1.4*medLengh[7])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[1] = 1, validJoint[8] = 1;
				}
			}

			//left leg
			if (pt3D[9 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[9 + PointFormat * j] - pt3D[8 + PointFormat * j]) < 1.4*medLengh[8])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 9 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[9] = 1, validJoint[8] = 1;
				}
			}
			if (pt3D[10 + PointFormat * j].x != 0 && pt3D[9 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[10 + PointFormat * j] - pt3D[9 + PointFormat * j]) < 1.4*medLengh[9])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 10 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 9 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[10] = 1, validJoint[9] = 1;
				}
			}
			if (pt3D[11 + PointFormat * j].x != 0 && pt3D[10 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[11 + PointFormat * j] - pt3D[10 + PointFormat * j]) < 1.4*medLengh[10])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 10 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[11] = 1, validJoint[10] = 1;
				}
			}

			//right leg
			if (pt3D[8 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[8 + PointFormat * j] - pt3D[12 + PointFormat * j]) < 1.4*medLengh[11])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 12 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[8] = 1, validJoint[12] = 1;
				}
			}
			if (pt3D[13 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[13 + PointFormat * j] - pt3D[12 + PointFormat * j]) < 1.4*medLengh[12])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 13 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 12 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[13] = 1, validJoint[12] = 1;
				}
			}
			if (pt3D[14 + PointFormat * j].x != 0 && pt3D[13 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[14 + PointFormat * j] - pt3D[13 + PointFormat * j]) < 1.4*medLengh[13])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 13 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[13] = 1, validJoint[14] = 1;
				}
			}

			//right eye+ ear
			if (pt3D[0 + PointFormat * j].x != 0 && pt3D[15 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[0 + PointFormat * j] - pt3D[15 + PointFormat * j]) < 1.4*medLengh[14])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 0 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 15 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[15] = 1;
				}
			}
			if (pt3D[15 + PointFormat * j].x != 0 && pt3D[17 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[15 + PointFormat * j] - pt3D[17 + PointFormat * j]) < 1.4*medLengh[15])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 15 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 17 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[15] = 1, validJoint[17] = 1;
				}
			}

			//left eye+ ear
			if (pt3D[0 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[0 + PointFormat * j] - pt3D[16 + PointFormat * j]) < 1.4*medLengh[16])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 0 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 16 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[0] = 1, validJoint[17] = 1;
				}
			}
			if (pt3D[18 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[16 + PointFormat * j] - pt3D[18 + PointFormat * j]) < 1.4*medLengh[17])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 18 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 16 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[18] = 1, validJoint[16] = 1;
				}
			}

			//left foot
			if (pt3D[11 + PointFormat * j].x != 0 && pt3D[22 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[11 + PointFormat * j] - pt3D[22 + PointFormat * j]) < 1.4*medLengh[18])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 22 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[11] = 1, validJoint[22] = 1;
				}
			}
			if (pt3D[11 + PointFormat * j].x != 0 && pt3D[23 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[11 + PointFormat * j] - pt3D[23 + PointFormat * j]) < 1.4*medLengh[19])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 23 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[11] = 1, validJoint[23] = 1;
				}
			}
			if (pt3D[11 + PointFormat * j].x != 0 && pt3D[24 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[11 + PointFormat * j] - pt3D[24 + PointFormat * j]) < 1.4*medLengh[20])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 24 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[11] = 1, validJoint[24] = 1;
				}
			}

			//right foot
			if (pt3D[14 + PointFormat * j].x != 0 && pt3D[19 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[14 + PointFormat * j] - pt3D[19 + PointFormat * j]) < 1.4*medLengh[21])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 19 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[14] = 1, validJoint[19] = 1;
				}
			}
			if (pt3D[14 + PointFormat * j].x != 0 && pt3D[20 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[14 + PointFormat * j] - pt3D[20 + PointFormat * j]) < 1.4*medLengh[22])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 20 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[14] = 1, validJoint[20] = 1;
				}
			}
			if (pt3D[14 + PointFormat * j].x != 0 && pt3D[21 + PointFormat * j].x != 0)
			{
				if (norm(pt3D[14 + PointFormat * j] - pt3D[21 + PointFormat * j]) < 1.4*medLengh[23])
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					i = 21 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
					glEnd();
					glPopMatrix();
					validJoint[14] = 1, validJoint[21] = 1;
				}
			}

			int once = 1;
			for (int i = 0; i < PointFormat; i++)
			{
				if (pt3D[i + PointFormat * j].x != 0.0)
				{
					glPushMatrix();
					glColor3fv(color);
					glTranslatef(pt3D[i + PointFormat * j].x - PointsCentroid[0], pt3D[i + PointFormat * j].y - PointsCentroid[1], pt3D[i + PointFormat * j].z - PointsCentroid[2]);

					glutSolidSphere(pointSize / 50.f, 10, 10);

					if (once == 1)
					{
						once = 0;
						char name[512];
						sprintf(name, "%d", j);
						DrawStr(Point3f(0, 1, 0), name);
					}
					glPopMatrix();
				}
			}
		}
		glLineWidth(pointSize);
	}

	delete[]MedianLength;
}
void Render_COCO_3D_SkeletonWithIndex(vector<Point3f> &pt3D, vector<int> &Pid, GLfloat *color, int PointFormat)
{
	glLineWidth(pointSize / 10.f);

	int TorsoJointId[4];
	if (PointFormat == 18)
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 8, TorsoJointId[3] = 11;
	else
		TorsoJointId[0] = 2, TorsoJointId[1] = 5, TorsoJointId[2] = 9, TorsoJointId[3] = 12;

	vector<double> *MedianLength = new vector<double>[PointFormat - 1];
	if (!hasMedLengthInfo)
	{
		//determine mean bone length
		for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
		{
			//head-neck
			if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]);
				MedianLength[0].push_back(leng);
			}

			//left arm
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[2 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[1 + PointFormat * j] - pt3D[2 + PointFormat * j]);
				MedianLength[1].push_back(leng);
			}
			if (pt3D[2 + PointFormat * j].x != 0 && pt3D[3 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[3 + PointFormat * j] - pt3D[2 + PointFormat * j]);
				MedianLength[2].push_back(leng);
			}
			if (pt3D[3 + PointFormat * j].x != 0 && pt3D[4 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[1 + PointFormat * j] - pt3D[4 + PointFormat * j]);
				MedianLength[3].push_back(leng);
			}

			//right arm
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[5 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[1 + PointFormat * j] - pt3D[5 + PointFormat * j]);
				MedianLength[4].push_back(leng);
			}
			if (pt3D[5 + PointFormat * j].x != 0 && pt3D[6 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[6 + PointFormat * j] - pt3D[5 + PointFormat * j]);
				MedianLength[5].push_back(leng);
			}
			if (pt3D[6 + PointFormat * j].x != 0 && pt3D[7 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[6 + PointFormat * j] - pt3D[7 + PointFormat * j]);
				MedianLength[6].push_back(leng);
			}

			//left leg
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[1 + PointFormat * j] - pt3D[8 + PointFormat * j]);
				MedianLength[7].push_back(leng);
			}
			if (pt3D[8 + PointFormat * j].x != 0 && pt3D[9 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[8 + PointFormat * j] - pt3D[9 + PointFormat * j]);
				MedianLength[8].push_back(leng);
			}
			if (pt3D[9 + PointFormat * j].x != 0 && pt3D[10 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[9 + PointFormat * j] - pt3D[10 + PointFormat * j]);
				MedianLength[9].push_back(leng);
			}

			//right leg
			if (pt3D[1 + PointFormat * j].x != 0 && pt3D[11 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[1 + PointFormat * j] - pt3D[11 + PointFormat * j]);
				MedianLength[10].push_back(leng);
			}
			if (pt3D[11 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[11 + PointFormat * j] - pt3D[12 + PointFormat * j]);
				MedianLength[11].push_back(leng);
			}
			if (pt3D[12 + PointFormat * j].x != 0 && pt3D[13 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[12 + PointFormat * j] - pt3D[13 + PointFormat * j]);
				MedianLength[12].push_back(leng);
			}

			//right eye+ ear
			if (pt3D[0 + PointFormat * j].x != 0 && pt3D[14 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[0 + PointFormat * j] - pt3D[14 + PointFormat * j]);
				MedianLength[13].push_back(leng);
			}
			if (pt3D[14 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[16 + PointFormat * j] - pt3D[14 + PointFormat * j]);
				MedianLength[14].push_back(leng);
			}

			//left eye+ ear
			if (pt3D[0 + PointFormat * j].x != 0 && pt3D[15 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[0 + PointFormat * j] - pt3D[15 + PointFormat * j]);
				MedianLength[15].push_back(leng);
			}
			if (pt3D[15 + PointFormat * j].x != 0 && pt3D[17 + PointFormat * j].x != 0)
			{
				double leng = norm(pt3D[17 + PointFormat * j] - pt3D[15 + PointFormat * j]);
				MedianLength[16].push_back(leng);
			}
		}
		double maxLimbLength = 0;
		for (int ii = 0; ii < PointFormat - 1; ii++)
		{
			if (MedianLength[ii].size() > 0)
				medLengh[ii] = Median(MedianLength[ii]), maxLimbLength = max(maxLimbLength, medLengh[ii]);
			else
				medLengh[ii] = 500.0 * UnitScale;
		}
		Discontinuity3DThresh = min(maxLimbLength, 500.0 * UnitScale);
	}

	for (int j = 0; j < (int)pt3D.size() / PointFormat; j++)
	{
		int i;
		vector<bool> validJoint(PointFormat);
		//head-neck
		if (pt3D[PointFormat * j].x != 0 && pt3D[1 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[PointFormat * j] - pt3D[1 + PointFormat * j]) < 1.4*medLengh[0])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[0] = 1, validJoint[1] = 1;
			}
		}

		//left arm
		if (pt3D[1 + PointFormat * j].x != 0 && pt3D[2 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[1 + PointFormat * j] - pt3D[2 + PointFormat * j]) < 1.4*medLengh[1])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 2 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[1] = 1, validJoint[2] = 1;
			}
		}
		if (pt3D[2 + PointFormat * j].x != 0 && pt3D[3 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[3 + PointFormat * j] - pt3D[2 + PointFormat * j]) < 1.4*medLengh[2])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 2 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 3 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[2] = 1, validJoint[3] = 1;
			}
		}
		if (pt3D[3 + PointFormat * j].x != 0 && pt3D[4 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[1 + PointFormat * j] - pt3D[4 + PointFormat * j]) < 1.4*medLengh[3])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 3 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 4 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[3] = 1, validJoint[4] = 1;
			}
		}

		//right arm
		if (pt3D[1 + PointFormat * j].x != 0 && pt3D[5 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[1 + PointFormat * j] - pt3D[5 + PointFormat * j]) < 1.4*medLengh[4])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 1 + PointFormat * j; glColor3fv(color); glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 5 + PointFormat * j; glColor3fv(color); glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[1] = 1, validJoint[5] = 1;
			}
		}
		if (pt3D[5 + PointFormat * j].x != 0 && pt3D[6 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[6 + PointFormat * j] - pt3D[5 + PointFormat * j]) < 1.4*medLengh[5])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 5 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 6 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[5] = 1, validJoint[6] = 1;
			}
		}
		if (pt3D[6 + PointFormat * j].x != 0 && pt3D[7 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[6 + PointFormat * j] - pt3D[7 + PointFormat * j]) < 1.4*medLengh[6])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 6 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 7 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[6] = 1, validJoint[7] = 1;
			}
		}

		//left leg
		if (pt3D[1 + PointFormat * j].x != 0 && pt3D[8 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[1 + PointFormat * j] - pt3D[8 + PointFormat * j]) < 1.4*medLengh[7])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[1] = 1, validJoint[8] = 1;
			}
		}
		if (pt3D[8 + PointFormat * j].x != 0 && pt3D[9 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[8 + PointFormat * j] - pt3D[9 + PointFormat * j]) < 1.4*medLengh[8])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 8 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 9 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[8] = 1, validJoint[9] = 1;
			}
		}
		if (pt3D[9 + PointFormat * j].x != 0 && pt3D[10 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[9 + PointFormat * j] - pt3D[10 + PointFormat * j]) < 1.4*medLengh[9])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 9 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 10 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[9] = 1, validJoint[10] = 1;
			}
		}

		//right leg
		if (pt3D[1 + PointFormat * j].x != 0 && pt3D[11 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[1 + PointFormat * j] - pt3D[11 + PointFormat * j]) < 1.4*medLengh[10])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 1 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[1] = 1, validJoint[11] = 1;
			}
		}
		if (pt3D[11 + PointFormat * j].x != 0 && pt3D[12 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[11 + PointFormat * j] - pt3D[12 + PointFormat * j]) < 1.4*medLengh[11])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 11 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 12 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[11] = 1, validJoint[12] = 1;
			}
		}
		if (pt3D[12 + PointFormat * j].x != 0 && pt3D[13 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[12 + PointFormat * j] - pt3D[13 + PointFormat * j]) < 1.4*medLengh[12])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 12 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 13 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[13] = 1, validJoint[12] = 1;
			}
		}

		//right eye+ ear
		if (pt3D[0 + PointFormat * j].x != 0 && pt3D[14 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[0 + PointFormat * j] - pt3D[14 + PointFormat * j]) < 1.4*medLengh[13])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 0 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[0] = 1, validJoint[14] = 1;
			}
		}
		if (pt3D[14 + PointFormat * j].x != 0 && pt3D[16 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[16 + PointFormat * j] - pt3D[14 + PointFormat * j]) < 1.4*medLengh[14])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 14 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 16 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[14] = 1, validJoint[16] = 1;
			}
		}

		//left eye+ ear
		if (pt3D[0 + PointFormat * j].x != 0 && pt3D[15 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[0 + PointFormat * j] - pt3D[15 + PointFormat * j]) < 1.4*medLengh[15])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 0 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 15 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[0] = 1, validJoint[15] = 1;
			}
		}
		if (pt3D[15 + PointFormat * j].x != 0 && pt3D[17 + PointFormat * j].x != 0)
		{
			if (norm(pt3D[17 + PointFormat * j] - pt3D[15 + PointFormat * j]) < 1.4*medLengh[16])
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glBegin(GL_LINE_STRIP);
				i = 15 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				i = 17 + PointFormat * j; glColor3fv(color); glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);
				glEnd();
				glPopMatrix();
				validJoint[15] = 1, validJoint[17] = 1;
			}
		}

		int once = 1;
		for (int i = 0; i < PointFormat; i++)
		{
			if (pt3D[i + PointFormat * j].x != 0.0)
			{
				glPushMatrix();
				glMultMatrixf(GlobalRot);
				glColor3fv(color);
				glTranslatef(pt3D[i + PointFormat * j].x - PointsCentroid[0], pt3D[i + PointFormat * j].y - PointsCentroid[1], pt3D[i + PointFormat * j].z - PointsCentroid[2]);

				glutSolidSphere(pointSize, 10, 10);

				if (once == 1)
				{
					once = 0;
					char name[512];
					sprintf(name, "%d", Pid[j]);
					DrawStr(Point3f(0, 1, 0), name);
				}
				glPopMatrix();
			}
		}
	}
	glLineWidth(1.0);

	delete[]MedianLength;
}

void RenderSMPLModel()
{
	// Projection and modelview matrices
	float top = tan(60.0 / 360.*M_PI)*g_nearPlane, right = top * g_Width / g_Height;
	Eigen::Matrix4f proj = frustrum(-right, right, -top, top, g_nearPlane, g_farPlane);

	Eigen::Affine3f V = Eigen::Affine3f::Identity();
	V.translate(Eigen::Vector3f(0, 0, -g_fViewDistance));
	V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseYRotate, Eigen::Vector3f(1, 0, 0)));
	if (!rotateInZ)
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 1, 0)));
	else
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 0, 1)));

	//Model transformation
	Eigen::Affine3f T = Eigen::Affine3f::Identity();
	T = AngleAxisf(1.f*modelAx / 180.0*M_PI, Vector3f::UnitX())
		* AngleAxisf(1.f*modelAy / 180.0*M_PI, Vector3f::UnitY());
	T.translate(Eigen::Vector3f(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]));

	// select program and attach uniforms
	glUseProgram(shaderProgram1);
	GLint proj_loc = glGetUniformLocation(shaderProgram1, "proj");
	glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj.data());
	GLint model_View = glGetUniformLocation(shaderProgram1, "V");
	glUniformMatrix4fv(model_View, 1, GL_FALSE, V.matrix().data());
	GLint model_Trans = glGetUniformLocation(shaderProgram1, "T");
	glUniformMatrix4fv(model_Trans, 1, GL_FALSE, T.matrix().data());
	GLint colorID = glGetUniformLocation(shaderProgram1, "InColor");
	GLfloat Cycan[3] = { 0.f, 1.f, 1.f };
	glUniform3fv(colorID, 1, Cycan);

	// Draw mesh as wireframe
	for (int smplID = 0; smplID < g_vis.PointPosition3.size() / 6890; smplID++)
	{
		glBindVertexArray(smpl_VAO[smplID]);

		glBindBuffer(GL_ARRAY_BUFFER, smpl_VBO[smplID]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f) * 6890, &g_vis.PointPosition3[6890 * smplID], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, smpl_EBO[smplID]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Point3i)*faces.size(), &faces[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); //draw wireframe
		glDrawElements(GL_TRIANGLES, 3 * faces.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

	glUseProgram(0);
}
void RenderSMPLModel2()
{
	// Projection and modelview matrices
	float top = tan(60.0 / 360.*M_PI)*g_nearPlane, right = top * g_Width / g_Height;
	Eigen::Matrix4f proj = frustrum(-right, right, -top, top, g_nearPlane, g_farPlane);

	Eigen::Affine3f V = Eigen::Affine3f::Identity();
	V.translate(Eigen::Vector3f(0, 0, -g_fViewDistance));
	V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseYRotate, Eigen::Vector3f(1, 0, 0)));
	if (!rotateInZ)
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 1, 0)));
	else
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 0, 1)));

	//Model transformation
	Eigen::Affine3f T = Eigen::Affine3f::Identity();
	T = AngleAxisf(1.f*modelAx / 180.0*M_PI, Vector3f::UnitX())
		* AngleAxisf(1.f*modelAy / 180.0*M_PI, Vector3f::UnitY());
	T.translate(Eigen::Vector3f(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]));


	for (int smplID = 0; smplID < nMaxPeople; smplID++)
	{
		//if (showPeople != -1 && smplID == showPeople)
		//	continue;
		//if (AllPeopleMesh[smplID].size() == 0)
		//	continue;

		Point3f PointColor;
		Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
		for (unsigned int i = 0; i <= 255; i++)
			colorMapSource.at<uchar>(i, 0) = i;
		Mat colorMap; applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);


		int nframes = AllPeopleMesh[smplID].size() / 6890;
		for (int inst = 0; inst < nframes; inst++)
		{
			// select program and attach uniforms
			glUseProgram(shaderProgram1);
			GLint proj_loc = glGetUniformLocation(shaderProgram1, "proj");
			glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj.data());
			GLint model_View = glGetUniformLocation(shaderProgram1, "V");
			glUniformMatrix4fv(model_View, 1, GL_FALSE, V.matrix().data());
			GLint model_Trans = glGetUniformLocation(shaderProgram1, "T");
			glUniformMatrix4fv(model_Trans, 1, GL_FALSE, T.matrix().data());
			GLint colorID = glGetUniformLocation(shaderProgram1, "InColor");

			double ctime = 1.0*AllPeopleMeshTimeId[smplID][inst];
			double colorIdx = (ctime - startTime) / (maxTime - startTime) * 255.0;
			colorIdx = min(255.0, colorIdx);
			PointColor.z = colorMap.at<Vec3b>(colorIdx, 0)[0] / 255.0; //blue
			PointColor.y = colorMap.at<Vec3b>(colorIdx, 0)[1] / 255.0; //green
			PointColor.x = colorMap.at<Vec3b>(colorIdx, 0)[2] / 255.0; //red

			globalColor[0] = PointColor.x, globalColor[1] = PointColor.y, globalColor[2] = PointColor.z;

			//if (smplID == 0)
			glUniform3fv(colorID, 1, globalColor);
			//else
			//	glUniform3fv(colorID, 1, Red);

			glBindVertexArray(smpl_VAO[smplID*maxInstance + inst]);

			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, smpl_VBO[smplID*maxInstance + inst]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f) * 6890, &AllPeopleMesh[smplID][inst * 6890], GL_DYNAMIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);

			// 2nd attribute buffer : colors
			//glEnableVertexAttribArray(1);
			//glBindBuffer(GL_ARRAY_BUFFER, smpl_VectexColorObject[smplID]);
			//glBufferData(GL_ARRAY_BUFFER, smpl::SMPLModel::nVertices * sizeof(Point3f), &g_vis.SMPL_Colors[smplID][0].x, GL_STATIC_DRAW);
			//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, smpl_EBO[smplID*maxInstance + inst]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Point3i)*faces.size(), &faces[0], GL_STATIC_DRAW);

			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); //draw wireframe
			glDrawElements(GL_TRIANGLES, 3 * faces.size(), GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
		}
	}


	glUseProgram(0);
}
void RenderPointCloud()
{
	// Projection and modelview matrices
	float top = tan(60.0 / 360.*M_PI)*g_nearPlane, right = top * g_Width / g_Height;
	Eigen::Matrix4f proj = frustrum(-right, right, -top, top, g_nearPlane, g_farPlane);

	Eigen::Affine3f V = Eigen::Affine3f::Identity();
	V.translate(Eigen::Vector3f(0, 0, -g_fViewDistance));
	V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseYRotate, Eigen::Vector3f(1, 0, 0)));
	if (!rotateInZ)
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 1, 0)));
	else
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 0, 1)));


	//Model transformation
	//Eigen::Affine3f T = Eigen::Affine3f::Identity();
	//T.translate(Eigen::Vector3f(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]));
	Eigen::Affine3f T = Eigen::Affine3f::Identity();
	T = AngleAxisf(1.f*modelAx / 180.0*M_PI, Vector3f::UnitX())
		* AngleAxisf(1.f*modelAy / 180.0*M_PI, Vector3f::UnitY());
	T.translate(Eigen::Vector3f(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]));

	// select program and attach uniforms
	glUseProgram(shaderProgram2);
	GLint proj_loc = glGetUniformLocation(shaderProgram2, "proj");
	glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj.data());
	GLint model_View = glGetUniformLocation(shaderProgram2, "V");
	glUniformMatrix4fv(model_View, 1, GL_FALSE, V.matrix().data());
	GLint model_Trans = glGetUniformLocation(shaderProgram2, "T");
	glUniformMatrix4fv(model_Trans, 1, GL_FALSE, T.matrix().data());

	glBindVertexArray(pointCloud_VAO);
	glPointSize(pointSize);
	glDrawArrays(GL_POINTS, 0, g_vis.CorpusPointPosition.size() * 3);
	glBindVertexArray(0);

	glUseProgram(0);
}
void RenderVoxel()
{
	// Projection and modelview matrices
	float top = tan(60.0 / 360.*M_PI)*g_nearPlane, right = top * g_Width / g_Height;
	Eigen::Matrix4f proj = frustrum(-right, right, -top, top, g_nearPlane, g_farPlane);

	Eigen::Affine3f V = Eigen::Affine3f::Identity();
	V.translate(Eigen::Vector3f(0, 0, -g_fViewDistance));
	V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseYRotate, Eigen::Vector3f(1, 0, 0)));
	if (!rotateInZ)
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 1, 0)));
	else
		V.rotate(Eigen::AngleAxisf(-1.0f*Pi / 180 * g_mouseXRotate, Eigen::Vector3f(0, 0, 1)));

	//Model transformation
	Eigen::Affine3f T = Eigen::Affine3f::Identity();
	T.translate(Eigen::Vector3f(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]));

	// select program and attach uniforms
	glUseProgram(shaderProgram3);
	GLint proj_loc = glGetUniformLocation(shaderProgram3, "proj");
	glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj.data());
	GLint model_View = glGetUniformLocation(shaderProgram3, "V");
	glUniformMatrix4fv(model_View, 1, GL_FALSE, V.matrix().data());
	GLint model_Trans = glGetUniformLocation(shaderProgram3, "T");
	glUniformMatrix4fv(model_Trans, 1, GL_FALSE, T.matrix().data());

	glBindVertexArray(pointCloud_VAO2);
	glPointSize(VT.cellSize);
	glDrawArrays(GL_POINTS, 0, VT.pts.size() * 3);
	glBindVertexArray(0);

	glUseProgram(0);
}

vector<textureCoord> g_backgroundTexture;
Mat g_backgroundTextureIm;
GLuint matToTexture(cv::Mat &mat)
{
	// Generate a number for our textureID's unique handle
	GLuint textureID = 0;
	glGenTextures(1, &textureID);
	//printLOG("texture ID %d\n",textureID);

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	// select modulate to mix texture with color for shading
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
		GL_LINEAR_MIPMAP_NEAREST);
	// when texture area is large, bilinear filter the first mipmap
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// if wrap is true, the texture wraps over at the edges (repeat)
	//       ... false, the texture ends at the edges (clamp)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
		GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
		GL_CLAMP);

	/*// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
	0,                 // Pyramid level (for mip-mapping) - 0 is the top level
	GL_RGB,            // Internal colour format to convert to
	mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
	mat.rows,          // Image height i.e. 480 for Kinect in standard mode
	0,                 // Border width in pixels (can either be 1 or 0)
	GL_RGB, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
	GL_UNSIGNED_BYTE,  // Image data type
	mat.ptr());        // The actual image data itself
	*/
	// build our texture mipmaps
	gluBuild2DMipmaps(GL_TEXTURE_2D, 3, mat.cols, mat.rows,
		GL_BGR_EXT, GL_UNSIGNED_BYTE, mat.ptr());
	//glGenerateMip(GL_TEXTURE_2D);

	return textureID;
}
GLuint InsertTexture(Mat& image)
{
	GLuint tempTexture = matToTexture(image);
	return tempTexture;
}
void GenerateBackgroundTexture(char *Fname, int camID)
{
	g_backgroundTextureIm = imread(Fname);
	if (g_backgroundTextureIm.rows == 0)
		g_backgroundTextureIm = Mat(540, 960, CV_8UC3, Scalar(0, 0, 0));
	if (g_backgroundTextureIm.rows == 0)
	{
		if (g_backgroundTexture[camID].textureIdx > 0)
		{
			glDeleteTextures(1, &g_backgroundTexture[camID].textureIdx);
			g_backgroundTexture[camID].textureIdx = 0;
		}
		return;
	}
	if (g_backgroundTexture[camID].textureIdx > 0)
	{
		glDeleteTextures(1, &g_backgroundTexture[camID].textureIdx);
		g_backgroundTexture[camID].textureIdx = 0;
	}
	if (g_backgroundTextureIm.rows == 0)
	{
		printLOG("## ERROR: background texture image is not valid\n");
		return;
	}

	//make image to 2^n for texture mapping
	int n = 0;
	if (g_backgroundTextureIm.cols > g_backgroundTextureIm.rows)
		for (n = 1; (g_backgroundTextureIm.cols >> n); n++);
	else
		for (n = 1; (g_backgroundTextureIm.rows >> n); n++);
	n -= 1;
	Mat resizedImage(pow(2.0f, n), pow(2.0f, n), g_backgroundTextureIm.type());
	resize(g_backgroundTextureIm, resizedImage, resizedImage.size());
	g_backgroundTexture[camID].textureIdx = InsertTexture(resizedImage);//g_visData.g_textureId.size()-1;
}
void RenderImage()
{
	int w = g_backgroundTextureIm.cols, h = g_backgroundTextureIm.rows;
	double ratio = 1.0*w / h, s = 1.0 / nCams;

	for (int ii = 0; ii < (int)g_backgroundTexture.size(); ii++)
	{
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0, 1, 1, 0, 1, -1);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glBindTexture(GL_TEXTURE_2D, g_backgroundTexture[ii].textureIdx);
		glColor3f(1, 1, 1);
		glBegin(GL_QUADS);
		glTexCoord2d(0.0, 0.0); glVertex3f(s*ii, 1.f - s, 0);
		glTexCoord2d(1.0, 0.0); glVertex3f(s*(ii + 1), 1.f - s, 0);
		glTexCoord2d(1.0, 1.0); glVertex3f(s*(ii + 1), 1.f, 0);
		glTexCoord2d(0.0, 1.0); glVertex3f(s*ii, 1.f, 0);
		glEnd();
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glDisable(GL_TEXTURE_2D);
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_DEPTH_TEST);
	}
}

bool poseChanged = false;
int pose[72], opose[72];
SparseMatrix<double, ColMajor> dVdt;

void RenderObjects()
{
	if (zPress)
		showOCRInstance = !showOCRInstance, zPress = 0;

	bool drawEverythingAtLast = false;
	if (timeID == (int)vAll3D_TidCidFid.size() - 1)
		drawEverythingAtLast = true;

	/*if (poseChanged)
	{
	SMPLParams myParams;
	myParams.scale = 3.1;
	myParams.t(0) = -2.790808e+00, myParams.t(1) = 3.847903e+00, myParams.t(2) = 1.217875e+01;
	for (int ii = 1; ii < 24; ii++)
	{
	opose[3 * ii] = pose[3 * ii], opose[3 * ii + 1] = pose[3 * ii + 1], opose[3 * ii + 2] = pose[3 * ii + 2];
	myParams.pose(ii, 0) = Pi/180.0*(pose[3 * ii]-180), myParams.pose(ii, 1) = Pi / 180.0*(pose[3 * ii + 1]- 180), myParams.pose(ii, 2) = Pi / 180.0*(pose[3 * ii + 2]- 180);
	}

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 18, nMaxPeople = 30;
	g_vis.PointPosition3.clear();	g_vis.PointPosition3.resize(nVertices);

	reconstruct(mySMPL, myParams.coeffs.data(), myParams.pose.data(), AllV[0].data());
	Map<VectorXd> V_vec(AllV[0].data(), AllV[0].size());
	V_vec = V_vec*myParams.scale + dVdt*myParams.t;

	double J_ptr[72];
	Map< VectorXd > J_vec(J_ptr, 72);
	J_vec = mySMPL.J_regl_bigl_ * V_vec;

	for (int ii = 0; ii < nVertices; ii++)
	g_vis.PointPosition3[ii] = Point3f(AllV[0](ii, 0), AllV[0](ii, 1), AllV[0](ii, 2));

	g_vis.PointPosition.clear();
	for (int ii = 0; ii < 24; ii++)
	g_vis.PointPosition.push_back(Point3f(J_ptr[3 * ii], J_ptr[3 * ii + 1], J_ptr[3 * ii + 2]));
	g_vis.PointPosition.push_back(Point3f(J_ptr[69], J_ptr[70], J_ptr[71]));
	}*/

	GLfloat gcolors[] = { 0, 0, 1,
		0, 0.5,1,
		0, 1, 1,
		0, 1, 0,
		1, 0.5, 0,
		1, 1, 0,
		1, 0, 0,
		1, 0, 1,
		1,1,1 };

	Point3f PointColor;
	Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
	for (unsigned int i = 0; i <= 255; i++)
		colorMapSource.at<uchar>(i, 0) = i;
	Mat colorMap; cv::applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);

	for (int jj = 0; jj < CamGroups.size(); jj++)
	{
		int id = rand() % 255;
		GLfloat myColor[] = { colorMap.at<Vec3b>(id, 0)[0] / 255.0, //blue
		 colorMap.at<Vec3b>(id, 0)[1] / 255.0,//green
		 colorMap.at<Vec3b>(id, 0)[2] / 255.0 };	//red
		bool first = true;
		for (int ii : CamGroups[jj])
		{
			float centerPt[3] = { g_vis.glCorpusCameraInfo[ii].camCenter[0], g_vis.glCorpusCameraInfo[ii].camCenter[1], g_vis.glCorpusCameraInfo[ii].camCenter[2] };
			GLfloat R[16] = { g_vis.glCorpusCameraInfo[ii].Rgl[0], g_vis.glCorpusCameraInfo[ii].Rgl[1], g_vis.glCorpusCameraInfo[ii].Rgl[2], g_vis.glCorpusCameraInfo[ii].Rgl[3],
				g_vis.glCorpusCameraInfo[ii].Rgl[4], g_vis.glCorpusCameraInfo[ii].Rgl[5], g_vis.glCorpusCameraInfo[ii].Rgl[6], g_vis.glCorpusCameraInfo[ii].Rgl[7],
				g_vis.glCorpusCameraInfo[ii].Rgl[8], g_vis.glCorpusCameraInfo[ii].Rgl[9], g_vis.glCorpusCameraInfo[ii].Rgl[10], g_vis.glCorpusCameraInfo[ii].Rgl[11],
				g_vis.glCorpusCameraInfo[ii].Rgl[12], g_vis.glCorpusCameraInfo[ii].Rgl[13], g_vis.glCorpusCameraInfo[ii].Rgl[14], g_vis.glCorpusCameraInfo[ii].Rgl[15] };

			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(R);
			DrawCamera(myColor);
			glPopMatrix();

			if (first)
			{
				glPushMatrix();
				glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1] + 0.5, centerPt[2] - PointsCentroid[2]);
				char Fname[512];  sprintf(Fname, "%d", jj);
				DrawStr(Point3f(myColor[0], myColor[1], myColor[2]), Fname);
				glPopMatrix();
				first = false;
			}
		
		}

		if (drawCorpusPoints)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			float centerPt[3] = { g_vis.glCorpusCameraInfo[CamGroups[jj][0]].camCenter[0], g_vis.glCorpusCameraInfo[CamGroups[jj][0]].camCenter[1], g_vis.glCorpusCameraInfo[CamGroups[jj][0]].camCenter[2] };
			glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]), glColor3fv(myColor);
			for (int ii : CamGroups[jj])
			{
				float centerPt[3] = { g_vis.glCorpusCameraInfo[ii].camCenter[0], g_vis.glCorpusCameraInfo[ii].camCenter[1], g_vis.glCorpusCameraInfo[ii].camCenter[2] };
				glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]), glColor3fv(myColor);
			}
			glEnd();
			glPopMatrix();
		}
	}


	glDisable(GL_CULL_FACE);
	char name[512];
	for (int ii = 0; ii < OCR3D.size(); ii++)
	{
		if (showOCRInstance)
			if (ii != TrajecID)
				continue;

		if (vPress)
		{
			if (OCR_type[ii] == 1)
				continue;
		}
		glPushMatrix();
		glTranslatef(OCR3D[ii].x - PointsCentroid[0], OCR3D[ii].y - PointsCentroid[1], OCR3D[ii].z - PointsCentroid[2]);
		glColor3fv(&gcolors[3 * (ii % 8)]);

		//DrawCube(Point3f(1, 1, 1)*pointSize/2.0);
		//glPointSize(pointSize * 5);
		//glBegin(GL_POINTS);
		//glVertex3f(0.0f, 0.0f, 0.0f);
		//glEnd();

		sprintf(name, "%s", OCRName[ii].c_str());
		DrawStr(Point3f(1, 0, 0), name);
		glPopMatrix();
	}
	glEnable(GL_CULL_FACE);

	if (otimeID != timeID)
	{
		int cid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;

		refFid = MyFtoI(((CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x)*CamTimeInfo[refCid].x));
		//int rfid = fid + (int)(CamTimeInfo[cid].y + 0.5); 	//ts = (1.0*refFid + offset) * ifps or fid~refFid+offset
		if (showSkeleton)
		{
			ReadPerCurrent3DSkeleton(Path, refFid, PointFormat);
			/*GetPerCurrentSke(refFid);
			for (int ii = 0; ii < cPeopleId.size(); ii++)
			{
				int pid = cPeopleId[ii];
				if (pid == 21 && timeID > 1264 && timeID < 1491)
					continue;
				if (pid == 121 && timeID > 7739 && timeID < 7991)
					continue;
				if (pid == 567 && timeID > 13745 && timeID < 13920)
					continue;
				if (pid == 525 && timeID > 19429 && timeID < 19779)
					continue;
				if (pid == 139 && timeID > 9566 && timeID < 9839)
					continue;
				if (pid == 157 || cPeopleId[ii] == 488)
					continue;

				ske18pts ske18;
				ske18.inst = refFid;
				for (int jj = 0; jj < PointFormat; jj++)
					ske18.pts[jj] = g_vis.PointPosition[PointFormat*ii + jj];

				if (trajBucket[pid].size() > 1)
					trajBucket[pid].erase(trajBucket[pid].begin());
				trajBucket[pid].push_back(ske18);
			}*/
		}
		//ReadAllCurrent3DSkeleton(Path, rfid);

		if (showSMPL)
			GetCurrent3DBody("C:/temp/CMU/01/01_01.txt", timeID);
		//ReadCurrent3DBody(Path, timeID);
		if (showImg)
		{
			int cid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
			for (int ii = 0; ii < nCams; ii++)
			{
				int fidj = fid * CamTimeInfo[cid].x / CamTimeInfo[ii].x;
				char Fname[512]; sprintf(Fname, "%s/MP/%d/%d.jpg", Path, ii, fidj);
				GenerateBackgroundTexture(Fname, cid);
			}
		}
		otimeID = timeID;
	}
	if (showSkeleton && g_vis.PointPosition.size() > 0)
	{
		Render_COCO_3D_Skeleton(g_vis.PointPosition, Red, PointFormat);

		/*vector<Point3f> pts;
		for (int ii = 0; ii < cPeopleId.size(); ii++)
		{
			int pid = cPeopleId[ii];
			vector<int> dummy; dummy.push_back(pid);
			pts.clear();
			for (int jj = 0; jj < PointFormat; jj++)
				pts.push_back(g_vis.PointPosition[PointFormat*ii + jj]);
			Render_COCO_3D_SkeletonWithIndex(pts, dummy, &gcolors[3 * (dummy[0] % 8)], PointFormat);

			glLineWidth(2.0f);
			for (int jid = 0; jid < PointFormat; jid++)
			{
				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				for (int jj = trajBucket[pid].size() - 1; jj > 0; jj--)
				{
					if (trajBucket[pid][jj].inst - trajBucket[pid][jj - 1].inst > 1)
						break;
					if (!IsValid3D(trajBucket[pid][jj].pts[jid]))
						continue;
					glColor3fv(&gcolors[3 * (dummy[0] % 8)]);
					glVertex3f(trajBucket[pid][jj].pts[jid].x - PointsCentroid[0], trajBucket[pid][jj].pts[jid].y - PointsCentroid[1], trajBucket[pid][jj].pts[jid].z - PointsCentroid[2]);
				}
				glEnd();
				glPopMatrix();
			}
		}*/
	}
	if (showImg)
		RenderImage();

	if ((oTrialID != TrialID))
	{
		Read3DTrajectory(Path, TrialID, colorCoded);
		oTrialID = TrialID;
	}
	if (EndTime)
		timeID = vAll3D_TidCidFid.size() - 1;

	//draw Corpus camera
	if (drawCorpusCameras)
	{
		int dstep = g_vis.glCorpusCameraInfo.size() > 5000 ? 5 : 1;
		for (int ii = 0; ii < g_vis.glCorpusCameraInfo.size(); ii += dstep)
		{
			float centerPt[3] = { g_vis.glCorpusCameraInfo[ii].camCenter[0], g_vis.glCorpusCameraInfo[ii].camCenter[1], g_vis.glCorpusCameraInfo[ii].camCenter[2] };
			GLfloat R[16] = { g_vis.glCorpusCameraInfo[ii].Rgl[0], g_vis.glCorpusCameraInfo[ii].Rgl[1], g_vis.glCorpusCameraInfo[ii].Rgl[2], g_vis.glCorpusCameraInfo[ii].Rgl[3],
				g_vis.glCorpusCameraInfo[ii].Rgl[4], g_vis.glCorpusCameraInfo[ii].Rgl[5], g_vis.glCorpusCameraInfo[ii].Rgl[6], g_vis.glCorpusCameraInfo[ii].Rgl[7],
				g_vis.glCorpusCameraInfo[ii].Rgl[8], g_vis.glCorpusCameraInfo[ii].Rgl[9], g_vis.glCorpusCameraInfo[ii].Rgl[10], g_vis.glCorpusCameraInfo[ii].Rgl[11],
				g_vis.glCorpusCameraInfo[ii].Rgl[12], g_vis.glCorpusCameraInfo[ii].Rgl[13], g_vis.glCorpusCameraInfo[ii].Rgl[14], g_vis.glCorpusCameraInfo[ii].Rgl[15] };

			glLoadName(ii);//for picking purpose

			bool picked = false;
			for (int jj = 0; jj < PickedCorpusCams.size(); jj++)
				if (ii == PickedCorpusCams[jj])
				{
					picked = true;
					break;
				}

			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(R);
			if (picked)
				DrawCamera(1);
			else
				DrawCamera(0);
			if (drawCameraID)
			{
				char name[512];
				sprintf(name, "%d", g_vis.glCorpusCameraInfo[ii].viewID);
				//DrawStr(Point3f(0, 1, 0), name);
			}
			glPopMatrix();
		}
	}

	//draw 3d points trajectories
	/*glLineWidth(2.0f);
	if (hasTimeEvoling3DPoints && vAll3D_TidCidFid.size() > 0)
	{
	if (OneTimeInstanceOnly &&showSkeleton)
	SkeletonPoints.clear();

	if (g_vis.PointPosition2.size() > 0)
	;//	Render_COCO_3D_Skeleton(g_vis.PointPosition2, Red, PointFormat);

	vector<Point2i> segNode;
	GLfloat TrajPointColorI[4], TrajCamColorI[4];
	vector<int>lastVisibleCam(nCams);
	for (int ii = 0; ii < nCams; ii++)
	lastVisibleCam[ii] = 0;

	float obj_timeID = (TimeInstancesStack[timeID] - TimeInstancesStack[0]) / 100;
	int time_obj = (int)obj_timeID;
	if (showCarSkeleton)
	for (int k = 0; k < 2000; k++)
	RenderCar_skeleton_new(time_obj, k);

	for (int tid = 0; tid < (int)g_vis.Traject3D.size(); tid++)
	{
	if (IndiviualTrajectory)
	if (tid != TrajecID)
	continue;

	segNode.clear();
	DetermineDiscontinuityInTrajectory(segNode, g_vis.Traject3D[tid], g_vis.Track3DLength[tid]);
	if (!OneTimeInstanceOnly)
	{
	for (int segID = 0; segID < (int)segNode.size(); segID++)
	{
	if (segNode[segID].y - segNode[segID].x < 3)
	continue;

	int fid;
	bool found = false;
	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	double t, t2 = TimeInstancesStack[timeID];
	for (fid = segNode[segID].x; fid < segNode[segID].y; fid++)
	{
	t = g_vis.Traject3D[tid][fid].timeID;
	if (t == t2)
	found = true;
	if (t > t2)
	break;

	if (abs(g_vis.Traject3D[tid][fid].WC.x) + abs(g_vis.Traject3D[tid][fid].WC.y) + abs(g_vis.Traject3D[tid][fid].WC.z) < 1.0e-16)
	continue;

	TrajPointColorI[0] = g_vis.Traject3D[tid][fid].rgb.x, TrajPointColorI[1] = g_vis.Traject3D[tid][fid].rgb.y, TrajPointColorI[2] = g_vis.Traject3D[tid][fid].rgb.z, TrajPointColorI[3] = 1.0f;
	if (timeID == TimeInstancesStack.size() - 1)
	TrajPointColorI[3] = 1.0f;//1.0f*t / t2; //display everything clearly
	else
	{
	TrajPointColorI[3] = 1.0f*pow(t / t2, 100);
	if (TrajPointColorI[3] < 5.e-1)
	continue;
	}

	int cidX = g_vis.Traject3D[tid][fid].viewID, rfidX = g_vis.Traject3D[tid][fid].frameID;
	glColor4fv(TrajPointColorI);
	glVertex3f(g_vis.Traject3D[tid][fid].WC.x - PointsCentroid[0], g_vis.Traject3D[tid][fid].WC.y - PointsCentroid[1], g_vis.Traject3D[tid][fid].WC.z - PointsCentroid[2]);

	if (t == t2)
	break;
	}
	glEnd();
	glPopMatrix();

	if (found)
	{
	if (displayAllMovingCams)
	for (int ii = 0; ii < lastVisibleCam.size(); ii++)
	lastVisibleCam[ii] = 1;
	else
	{
	int cid = g_vis.Traject3D[tid][fid].viewID;
	lastVisibleCam[cid] = 1;
	}
	}
	}
	}

	if (OneTimeInstanceOnly)
	{
	for (int segID = 0; segID < (int)segNode.size(); segID++)
	{
	if (segNode[segID].y - segNode[segID].x < 3)
	continue;

	int fid; double t, t1, t2;
	bool found = false;
	for (fid = segNode[segID].x; fid <= segNode[segID].y; fid++)
	{
	if (fid <= segNode[segID].y - 1)
	t1 = g_vis.Traject3D[tid][fid + 1].timeID;
	t = g_vis.Traject3D[tid][fid].timeID, t2 = TimeInstancesStack[timeID];
	if (t == t2)
	found = true;
	if (t > t2)
	break;
	if (t < t2 || (t1 <= t2 && fid <= segNode[segID].y - 1))
	continue;

	TrajPointColorI[0] = g_vis.Traject3D[tid][fid].rgb.x, TrajPointColorI[1] = g_vis.Traject3D[tid][fid].rgb.y, TrajPointColorI[2] = g_vis.Traject3D[tid][fid].rgb.z, TrajPointColorI[3] = 1.0f;
	int pickID = (int)g_vis.CorpusPointPosition.size() + nCorpusCams + nNonCorpusCams + MaxnTrajectories + tid*MaxnFrames + fid;
	glLoadName(pickID);//for picking purpose
	bool picked = false;
	for (unsigned int ii = 0; ii < PickedDynamicPoints.size(); ii++)
	{
	if (pickID == PickedDynamicPoints[ii])
	{
	picked = true;
	break;
	}
	}
	if (picked)
	TrajPointColorI[0] = 0.f, TrajPointColorI[1] = 0.f, TrajPointColorI[2] = 1.f, TrajPointColorI[3] = 1.0f;

	glPushMatrix();
	glColor4fv(TrajPointColorI);
	glTranslatef(g_vis.Traject3D[tid][fid].WC.x - PointsCentroid[0], g_vis.Traject3D[tid][fid].WC.y - PointsCentroid[1], g_vis.Traject3D[tid][fid].WC.z - PointsCentroid[2]);
	glutSolidSphere(pointSize*1.25, 10, 10);
	glPopMatrix();

	if (t == t2)
	break;
	if (showSkeleton)
	SkeletonPoints.push_back(Point3d(g_vis.Traject3D[tid][fid].WC.x, g_vis.Traject3D[tid][fid].WC.y, g_vis.Traject3D[tid][fid].WC.z));
	}

	if (found)
	{
	if (displayAllMovingCams)
	for (int ii = 0; ii < lastVisibleCam.size(); ii++)
	lastVisibleCam[ii] = 1;
	else
	{
	int cid = g_vis.Traject3D[tid][fid].viewID;
	lastVisibleCam[cid] = 1;
	}
	}
	}
	}

	if (drawNonCorpusCameras)
	drawedNonCorpusCameras = true;
	}

	if (drawNonCorpusCameras)
	{
	int cumCamID = 0;
	for (int cid = 0; cid < nCams; cid++)
	{
	if (g_vis.glCameraPoseInfo == NULL)
	continue;
	for (int fid = 0; fid < (int)g_vis.glCameraPoseInfo[cid].size(); fid++)
	{
	cumCamID++;
	if (lastVisibleCam[cid] == 0)
	continue;

	int rfid = g_vis.glCameraPoseInfo[cid][fid].frameID;
	double camTime = (CamTimeInfo[cid] + rfid)* Tscale / fps;
	//double camTime = CamTimeInfo[cid] + rfid; //assume frame sync
	if (camTime < TimeInstancesStack[timeID])
	continue;

	float centerPt[3] = { g_vis.glCameraPoseInfo[cid][fid].camCenter[0], g_vis.glCameraPoseInfo[cid][fid].camCenter[1], g_vis.glCameraPoseInfo[cid][fid].camCenter[2] };
	GLfloat R[16] = { g_vis.glCameraPoseInfo[cid][fid].Rgl[0], g_vis.glCameraPoseInfo[cid][fid].Rgl[1], g_vis.glCameraPoseInfo[cid][fid].Rgl[2], g_vis.glCameraPoseInfo[cid][fid].Rgl[3],
	g_vis.glCameraPoseInfo[cid][fid].Rgl[4], g_vis.glCameraPoseInfo[cid][fid].Rgl[5], g_vis.glCameraPoseInfo[cid][fid].Rgl[6], g_vis.glCameraPoseInfo[cid][fid].Rgl[7],
	g_vis.glCameraPoseInfo[cid][fid].Rgl[8], g_vis.glCameraPoseInfo[cid][fid].Rgl[9], g_vis.glCameraPoseInfo[cid][fid].Rgl[10], g_vis.glCameraPoseInfo[cid][fid].Rgl[11],
	g_vis.glCameraPoseInfo[cid][fid].Rgl[12], g_vis.glCameraPoseInfo[cid][fid].Rgl[13], g_vis.glCameraPoseInfo[cid][fid].Rgl[14], g_vis.glCameraPoseInfo[cid][fid].Rgl[15] };
	if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
	continue;

	glLoadName(cumCamID + nCorpusCams);//for picking purpose
	cumCamID++;

	glPushMatrix();
	glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
	glMultMatrixf(R);
	DrawCamera();
	if (drawCameraID)
	{
	char name[512];
	sprintf(name, "%d", cid);
	//DrawStr(Point3f(0, 1, 0), name);
	}
	glPopMatrix();
	break;
	}

	if (drawBriefCameraTraject)
	{
	if (0)
	{
	if (lastVisibleCam[cid] == 0)
	continue;
	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	for (int fid = 0; fid < (int)g_vis.glCameraPoseInfo[cid].size(); fid++)
	{
	int rfid = g_vis.glCameraPoseInfo[cid][fid].frameID;
	double camTime = (CamTimeInfo[cid] + rfid)* Tscale / fps;
	if (camTime > TimeInstancesStack[timeID])
	continue;

	float centerPt[3] = { g_vis.glCameraPoseInfo[cid][fid].camCenter[0], g_vis.glCameraPoseInfo[cid][fid].camCenter[1], g_vis.glCameraPoseInfo[cid][fid].camCenter[2] };
	if (g_vis.glCameraPoseInfo[cid][fid].frameID < 0 || abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
	continue;

	TrajCamColorI[0] = Red[0], TrajCamColorI[1] = Red[1], TrajCamColorI[2] = Red[2], TrajCamColorI[3] = 1.0f*pow(camTime / TimeInstancesStack[timeID], showSkeleton ? 50 : showSMPL ? 50 : 20);
	if (TrajCamColorI[3] < 2.5e-1)
	continue;

	glColor4fv(TrajCamColorI);
	glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
	}
	glEnd();
	glPopMatrix();
	}
	}
	}
	}

	if (OneTimeInstanceOnly &&showSkeleton)
	if ((int)SkeletonPoints.size() == 31)
	RenderSkeleton(SkeletonPoints, Blue);
	}*/

	if (drawEverythingAtLast)
	{
		int cumCamID = 0;
		for (int j = 0; j < nCams; j++)
		{
			if (g_vis.glCameraPoseInfo != NULL && g_vis.glCameraPoseInfo[j].size() > 0)
			{
				/*glBegin(GL_LINE_STRIP);
				for (int i = 0; i < g_vis.glCameraPoseInfo[j].size(); i++)
				{
				float centerPt[3] = { g_vis.glCameraPoseInfo[j][i].camCenter[0], g_vis.glCameraPoseInfo[j][i].camCenter[1], g_vis.glCameraPoseInfo[j][i].camCenter[2] };
				if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
				continue;

				GLfloat TrajCamColorI[4] = { Red[0], Red[1], Red[2], 1.0f*pow(1.0*i / g_vis.glCameraPoseInfo[j].size(), showSkeleton ? 50 : showSMPL ? 50 : 20) };
				if (TrajCamColorI[3] < 1e-1)
				continue;

				glColor4fv(TrajCamColorI);
				glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
				}
				glEnd();*/

				/*for (size_t i = g_vis.glCameraPoseInfo[j].size() - 1; i < g_vis.glCameraPoseInfo[j].size(); i++)
				{
				float centerPt[3] = { g_vis.glCameraPoseInfo[j][i].camCenter[0], g_vis.glCameraPoseInfo[j][i].camCenter[1], g_vis.glCameraPoseInfo[j][i].camCenter[2] };
				GLfloat R[16] = { g_vis.glCameraPoseInfo[j][i].Rgl[0], g_vis.glCameraPoseInfo[j][i].Rgl[1], g_vis.glCameraPoseInfo[j][i].Rgl[2], g_vis.glCameraPoseInfo[j][i].Rgl[3],
				g_vis.glCameraPoseInfo[j][i].Rgl[4], g_vis.glCameraPoseInfo[j][i].Rgl[5], g_vis.glCameraPoseInfo[j][i].Rgl[6], g_vis.glCameraPoseInfo[j][i].Rgl[7],
				g_vis.glCameraPoseInfo[j][i].Rgl[8], g_vis.glCameraPoseInfo[j][i].Rgl[9], g_vis.glCameraPoseInfo[j][i].Rgl[10], g_vis.glCameraPoseInfo[j][i].Rgl[11],
				g_vis.glCameraPoseInfo[j][i].Rgl[12], g_vis.glCameraPoseInfo[j][i].Rgl[13], g_vis.glCameraPoseInfo[j][i].Rgl[14], g_vis.glCameraPoseInfo[j][i].Rgl[15] };

				if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
				continue;

				glPushMatrix();
				glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
				glMultMatrixf(R);
				DrawCamera();
				if (drawCameraID)
				{
				char name[512];
				sprintf(name, "%d", j);
				DrawStr(Point3f(0, 1, 0), name);
				}
				glPopMatrix();
				}*/
			}
		}
	}

	//draw 3d moving cameras 
	if ((drawNonCorpusCameras && !drawedNonCorpusCameras) && vAll3D_TidCidFid.size() > 0)
	{
		//handle synced cameras
		double currentTime = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts;
		vector<int> vSameTime;
		if (syncedMode)
		{
			int cid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
			double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
			refFid = MyFtoI(ts*CamTimeInfo[refCid].x);
			for (int ii = -nCams; ii <= nCams; ii++)
			{
				if (timeID + ii >= 0 && timeID + ii < vAll3D_TidCidFid.size() - 1)
				{
					int cidi = vAll3D_TidCidFid[SortedTimeInstance[timeID + ii]].Cid, fidi = vAll3D_TidCidFid[SortedTimeInstance[timeID + ii]].Fid;
					double tsi = CamTimeInfo[cidi].y / CamTimeInfo[refCid].x + 1.0*fidi / CamTimeInfo[cidi].x;
					int refFidi = MyFtoI(tsi*CamTimeInfo[refCid].x);
					if (refFid == refFidi)
						vSameTime.push_back(timeID + ii);
				}
			}
			//increT = max(1, (int)vSameTime.size());
		}
		else
			vSameTime.push_back(timeID);

		/*//mean pos
		int cnt = 0;
		double mx = 0, my = 0, mz = 0;
		for (int ii = 0; ii < PointFormat; ii++)
		if (IsValid3D(g_vis.PointPosition[ii]))
		mx += g_vis.PointPosition[ii].x, my += g_vis.PointPosition[ii].y, mz += g_vis.PointPosition[ii].z, cnt++;
		mx = mx / cnt, my = my / cnt, mz = mz / cnt;*/

		int cid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
		refFid = MyFtoI(((CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x)*CamTimeInfo[refCid].x));

		int cumCamID = 0;
		for (int cid = 0; cid < nCams; cid++)
		{
			if (g_vis.glCameraPoseInfo != NULL && g_vis.glCameraPoseInfo[cid].size() > 0)
			{
				int CurrentCidFrame = -1;
				for (int jj = 0; jj < vSameTime.size() && CurrentCidFrame == -1; jj++)
				{
					int sTimeID = vSameTime[jj];
					for (int ii = sTimeID; ii > -1 && CurrentCidFrame == -1; ii--)
					{
						if (vAll3D_TidCidFid[SortedTimeInstance[sTimeID]].Cid == cid)
							CurrentCidFrame = vAll3D_TidCidFid[SortedTimeInstance[sTimeID]].Fid;
					}
				}
				if (CurrentCidFrame == -1)
					continue;

				/*if (cid == 0)
				{
				double n[3] = { g_vis.glCameraPoseInfo[cid][CurrentCidFrame].camCenter[0] - mx,
				g_vis.glCameraPoseInfo[cid][CurrentCidFrame].camCenter[1] - my,
				g_vis.glCameraPoseInfo[cid][CurrentCidFrame].camCenter[2] - mz };
				normalize(normalVec);
				}*/

				//draw camera tail
				glPushMatrix();
				glLineWidth(3);
				glBegin(GL_LINE_STRIP);
				for (int i = 0; i <= CurrentCidFrame; i++)
				{
					CameraData *CamI = &g_vis.glCameraPoseInfo[cid][i];
					if (CamI[0].valid == 0)
						continue;

					float centerPt[3] = { CamI[0].camCenter[0], CamI[0].camCenter[1], CamI[0].camCenter[2] };
					if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
						continue;

					GLfloat TrajCamColorI[4] = { Red[0], Red[1], Red[2], 1.0f*pow(1.0*CamI[0].frameID / CurrentCidFrame, (showSkeleton || showSMPL) ? 4 : 2) };
					if (TrajCamColorI[3] < 1e-1)
						continue;

					glColor4fv(TrajCamColorI);
					glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
				}
				glEnd();
				glLineWidth(1);
				glPopMatrix();

				CameraData *CamI = &g_vis.glCameraPoseInfo[cid][CurrentCidFrame];
				float centerPt[3] = { CamI[0].camCenter[0], CamI[0].camCenter[1], CamI[0].camCenter[2] };
				GLfloat R_[16], R[16] = { CamI[0].Rgl[0], CamI[0].Rgl[1], CamI[0].Rgl[2], CamI[0].Rgl[3], CamI[0].Rgl[4], CamI[0].Rgl[5], CamI[0].Rgl[6], CamI[0].Rgl[7], CamI[0].Rgl[8], CamI[0].Rgl[9], CamI[0].Rgl[10], CamI[0].Rgl[11], CamI[0].Rgl[12], CamI[0].Rgl[13], CamI[0].Rgl[14], CamI[0].Rgl[15] };

				mat_mul(GlobalRot, R, R_, 4, 4, 4);

				glLoadName(cumCamID + nCorpusCams);//for picking purpose
				glPushMatrix();
				glLineWidth(2);
				glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
				glMultMatrixf(R_);

				int hightlight = 0;
				if (VnearestCams.size() > 0)
				{
					for (int i = 0; i < VnearestCams[refFid].size() && hightlight == 0; i++)
					{
						if (VnearestCams[refFid][i].x == cid)
							hightlight = 1;
					}
					if (VnearestCams.size() > 0 && VnearestCams[refFid][0].x == cid)
						hightlight = 2;
				}

				DrawCamera(hightlight);
				if (drawCameraID)
				{
					char name[512];
					sprintf(name, "%d", cid);
					//DrawStr(Point3f(0, 1, 0), name);
				}
				glLineWidth(1);
				glPopMatrix();

			}
		}
	}

	/*if (1)
	{
	GLfloat R[16] = { 1, 0, 0,0,0, 1, 0, 0, 0, 0, 1, 0, 0,0,0, 1 };

	glPushMatrix();
	glTranslatef(PointsCentroid[0] - PointsCentroid[0], PointsCentroid[1] - PointsCentroid[1], PointsCentroid[2] - PointsCentroid[2]);
	glMultMatrixf(R);
	DrawCamera();
	glPopMatrix();

	Point3f xyz(PointsCentroid[0], PointsCentroid[1], PointsCentroid[2]);
	if (pid__ > -1)
	xyz = g_vis.CorpusPointPosition[pid__];

	CameraData *camI = AllVideosInfo[cid__].VideoInfo;
	double rayDir[3], principal[] = { camI[fid__].intrinsic[3], camI[fid__].intrinsic[4], 1.0 };
	getRayDir(rayDir, camI[fid__].invK, camI[fid__].R, principal);

	double direction[3] = { xyz.x - camI[fid__].camCenter[0], xyz.y - camI[fid__].camCenter[1], xyz.z - camI[fid__].camCenter[1] };
	double dot = dotProduct(rayDir, direction);
	if (dot < 0) //behind the camera
	int a = 0;

	Point2d projectedPt;
	if (camI[fid__].LensModel == RADIAL_TANGENTIAL_PRISM)
	ProjectandDistort(xyz, &projectedPt, camI[fid__].P, camI[fid__].K, camI[fid__].distortion, 1); //approx is fine for this task. No RS is needed
	else
	FisheyeProjectandDistort(xyz, &projectedPt, camI[fid__].P, camI[fid__].distortion, 1);

	if (projectedPt.x <0 || projectedPt.y <0 || projectedPt.x > camI[fid__].width - 1 || projectedPt.y > camI[fid__].height - 1)
	int a = 0;

	int a = 0;
	}*/
	return;
}

void display(void)
{
	if (changeBackgroundColor)
		glClearColor(1.0, 1.0, 1.0, 0.0);
	else
		glClearColor(0.0, 0.0, 0.0, 0.0);

	if (SaveStaticViewingParameters)
	{
		char Fname[512]; sprintf(Fname, "%s/OpenGLViewingPara.txt", Path);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%.8f %d %d %.8f %.8f %.8f %d %d ", g_fViewDistance, g_mouseXRotate, g_mouseYRotate, PointsCentroid[0], PointsCentroid[1], PointsCentroid[2], modelAx, modelAy);
		fclose(fp);
		SaveStaticViewingParameters = false;
	}
	if (SetStaticViewingParameters)
	{
		char Fname[512]; sprintf(Fname, "%s/OpenGLViewingPara.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			fscanf(fp, "%f %d %d %f %f %f %d %d", &g_fViewDistance, &g_mouseXRotate, &g_mouseYRotate, &PointsCentroid[0], &PointsCentroid[1], &PointsCentroid[2], &modelAx, &modelAy);
			fclose(fp);
		}
		modelAx = 0, modelAy = 0;
		SetStaticViewingParameters = false;
	}

	if (SaveDynamicViewingParameters)
	{
		ViewingParas vparas;
		vparas.timeID = timeID, vparas.viewDistance = g_fViewDistance, vparas.mouseXRotate = g_mouseXRotate, vparas.mouseYRotate = g_mouseYRotate,
			vparas.CentroidX = PointsCentroid[0], vparas.CentroidY = PointsCentroid[1], vparas.CentroidZ = PointsCentroid[2];
		DynamicViewingParas.push_back(vparas);
	}
	if (SetDynamicViewingParameters)
	{
		char Fname[512]; sprintf(Fname, "%s/OpenGLDynamicViewingPara.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			ViewingParas vparas;
			while (fscanf(fp, "%d %f %d %d %f %f %f", &vparas.timeID, &vparas.viewDistance, &vparas.mouseXRotate, &vparas.mouseYRotate, &vparas.CentroidX, &vparas.CentroidY, &vparas.CentroidZ) != EOF)
				DynamicViewingParas.push_back(vparas);
			fclose(fp);
		}
		SetDynamicViewingParameters = false;

		//Reset time and start rendering;
		timeID = 0;
	}

	if (SaveRendererViewingParameters)
	{
		SaveRendererViewingParameters = false;
		renderTime++;

		char Fname[512]; sprintf(Fname, "%s/OpenGLRenderViewingPara.txt", Path); FILE *fp = fopen(Fname, "a+");
		fprintf(fp, "%d %.8f %d %d %.8f %.8f %.8f \n", renderTime, g_fViewDistance, g_mouseXRotate, g_mouseYRotate,
			PointsCentroid[0], PointsCentroid[1], PointsCentroid[2]);
		fclose(fp);
	}
	if (SetRendererViewingParameters)
	{
		char Fname[512]; sprintf(Fname, "%s/OpenGLRenderViewingPara.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			ViewingParas vparas;
			while (fscanf(fp, "%d %f %d %d %f %f %f", &vparas.timeID, &vparas.viewDistance, &vparas.mouseXRotate, &vparas.mouseYRotate, &vparas.CentroidX, &vparas.CentroidY, &vparas.CentroidZ) != EOF)
				RenderViewingParas.push_back(vparas);
			fclose(fp);
		}
		SetRendererViewingParameters = false;

		//Reset time and start rendering;
		renderTime = 0;
		RenderedReady = true;
	}

	if (SaveDynamicViewingParameters == 0)
	{
		for (int ii = 0; ii < (int)DynamicViewingParas.size(); ii++)
		{
			if (DynamicViewingParas[ii].timeID == timeID)
			{
				g_fViewDistance = DynamicViewingParas[ii].viewDistance, g_mouseXRotate = DynamicViewingParas[ii].mouseXRotate, g_mouseYRotate = DynamicViewingParas[ii].mouseYRotate,
					PointsCentroid[0] = DynamicViewingParas[ii].CentroidX, PointsCentroid[1] = DynamicViewingParas[ii].CentroidY, PointsCentroid[2] = DynamicViewingParas[ii].CentroidZ;
				break;
			}
		}
	}
	if (RenderedReady)
	{
		for (int ii = 0; ii < (int)RenderViewingParas.size(); ii++)
		{
			if (RenderViewingParas[ii].timeID == renderTime)
			{
				g_fViewDistance = RenderViewingParas[ii].viewDistance, g_mouseXRotate = RenderViewingParas[ii].mouseXRotate, g_mouseYRotate = RenderViewingParas[ii].mouseYRotate,
					PointsCentroid[0] = RenderViewingParas[ii].CentroidX, PointsCentroid[1] = RenderViewingParas[ii].CentroidY, PointsCentroid[2] = RenderViewingParas[ii].CentroidZ;
				break;
			}
		}
		renderTime++;
		renderTime = min(renderTime, (int)RenderViewingParas.size() - 1);
	}
	if (timeID == vAll3D_TidCidFid.size() - 1 && !RenderedReady)
		SetRendererViewingParameters = true;

	// Clear frame buffer and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	glTranslatef(0, 0, -g_fViewDistance);
	//glRotated(180,  0, 0, 1), glRotated(180, 0, 1, 0);
	glRotated(-g_mouseYRotate, 1, 0, 0);
	if (!rotateInZ)
		glRotated(-g_mouseXRotate, 0, 1, 0);
	else
		glRotated(-g_mouseXRotate, 0, 0, 1);

	RenderObjects();
	if (drawCorpusPoints && g_vis.CorpusPointPosition.size() > 0)
		RenderPointCloud();
	if (VT.avail)
		RenderVoxel();
	if (showSMPL)
		RenderSMPLModel();
	if (showAxis || zPress)
		Draw_Axes(), showAxis = false;

	if (showGroundPlane)
		RenderGroundPlane();

	glutSwapBuffers();

	if (SaveScreen && oTrajecID != timeID)
	{
		char Fname[512];	sprintf(Fname, "%s/ScreenShot", Path); makeDir(Fname);
		sprintf(Fname, "%s/ScreenShot/%.4d.png", Path, timeID);
		screenShot(Fname, g_Width, g_Height, true);
		oTrajecID = timeID;
	}
	if (ImmediateSnap)
	{
		char Fname[512];	sprintf(Fname, "%s/ScreenShot", Path); makeDir(Fname);
		sprintf(Fname, "%s/ScreenShot/%.4d.png", Path, snapCount);
		screenShot(Fname, g_Width, g_Height, true);
		snapCount++;
	}
}
void IdleFunction(void)
{
	poseChanged = false;
	for (int ii = 3; ii < 72 && !poseChanged; ii++)
		if (pose[ii] - opose[ii] != 0)
			poseChanged = true;
	if (poseChanged)
		display();

	char Fname[512];
	if (AutoDisp)
	{
		double DisplayCurrentFrame = omp_get_wtime();
		if (DisplayCurrentFrame - DisplayStartTime > DisplayTimeStep) //don't really want to do faster than refresh rate
		{
			if (syncedMode == 1)
			{
				refFid += increF;

				for (int ii = timeID; ii < (int)vAll3D_TidCidFid.size(); ii++)
				{
					int cid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Fid;
					double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
					if (MyFtoI(ts*CamTimeInfo[refCid].x) == refFid)
					{
						timeID = ii;
						break;
					}
				}
			}
			else
				timeID += increT;

			if (vAll3D_TidCidFid.size() > 0 && timeID > (int)vAll3D_TidCidFid.size() - 1)
			{
				AutoDisp = false;
				timeID = (int)vAll3D_TidCidFid.size() - 1;
			}

			DisplayStartTime = DisplayCurrentFrame;
			printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts, timeID);
		}
		display();
	}
	return;
}
void Visualization()
{
	char *myargv[1];
	int myargc = 1;
	myargv[0] = "SfM";
	glutInit(&myargc, myargv);

	glutInitWindowSize(g_Width > 2560 ? 2560 : g_Width, g_Height > 1600 ? 1600 : g_Height);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("SfM!");

	glewExperimental = GL_TRUE;
	glewInit();

	shaderProgram1 = LoadShaders2(SingleVertexTransformShader, SingleFragmentShader);

	// Generate and attach buffers to smpl vertex array for largest possible #people
	for (int smplID = 0; smplID < nMaxPeople; smplID++)
	{
		for (int inst = 0; inst < maxInstance; inst++)
		{
			GLuint VBO, EBO;
			smpl_VAO.push_back(smplID*maxInstance + inst);
			smpl_VBO.push_back(VBO);
			smpl_EBO.push_back(EBO);

			glGenVertexArrays(1, &smpl_VAO[smplID*maxInstance + inst]);
			glGenBuffers(1, &smpl_VBO[smplID*maxInstance + inst]), glGenBuffers(1, &smpl_EBO[smplID*maxInstance + inst]);
		}
	}

	if (g_vis.CorpusPointPosition.size() > 0)
	{
		shaderProgram2 = LoadShaders2(SimpleVertexTransformShader, SimpleFragmentShader);

		// Get a handle for our buffers
		GLuint vertexPosition_modelspaceID = glGetAttribLocation(shaderProgram2, "vertexPosition_modelspace");
		GLuint vertexColorID = glGetAttribLocation(shaderProgram2, "vertexColor");

		glGenVertexArrays(1, &pointCloud_VAO);

		GLuint vertexbuffer, colorbuffer;
		glGenBuffers(1, &vertexbuffer);
		glGenBuffers(1, &colorbuffer);

		glBindVertexArray(pointCloud_VAO);

		// 1st attribute buffer : vertices
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f)*g_vis.CorpusPointPosition.size(), &g_vis.CorpusPointPosition[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(vertexPosition_modelspaceID);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(vertexPosition_modelspaceID, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// 2nd  attribute buffer : color
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f)*g_vis.CorpusPointColor.size(), &g_vis.CorpusPointColor[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(vertexColorID);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(vertexColorID, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindVertexArray(0);
	}

	if (VT.pts.size() > 0)
	{
		char buffer[512];  myGetCurDir(512, buffer);
		char Fname1[512];  sprintf(Fname1, "C:/Research/DevSoft/EnRecon/Shader/SimpleTransform.txt", buffer);
		char Fname2[512];  sprintf(Fname2, "C:/Research/DevSoft/EnRecon/Shader/SingleColor.txt", buffer);
		shaderProgram3 = LoadShaders(Fname1, Fname2);

		// Get a handle for our buffers
		GLuint vertexPosition_modelspaceID = glGetAttribLocation(shaderProgram3, "vertexPosition_modelspace");
		GLuint vertexColorID = glGetAttribLocation(shaderProgram3, "vertexColor");

		glGenVertexArrays(1, &pointCloud_VAO2);

		GLuint vertexbuffer, colorbuffer;
		glGenBuffers(1, &vertexbuffer);
		glGenBuffers(1, &colorbuffer);

		glBindVertexArray(pointCloud_VAO2);

		// 1st attribute buffer : vertices
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f)*VT.pts.size(), &VT.pts[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(vertexPosition_modelspaceID);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(vertexPosition_modelspaceID, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// 2nd  attribute buffer : color
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point3f)*VT.score.size(), &VT.score[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(vertexColorID);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(vertexColorID, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindVertexArray(0);
	}

	glShadeModel(GL_SMOOTH);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glClearColor(1.0, 1.0, 1.0, 0.0);

	glutDisplayFunc(display);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialInput);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutIdleFunc(IdleFunction);
	glutMotionFunc(MouseMotion);

	glutMainLoop();
}

int visualizationDriver(char *inPath, vector<int> &SelectedCams, int startF, int stopF, int increF_, bool hasColor_, int colorCoded_, bool hasPatchNormal_, bool hasTimeEvoling3DPoints_, bool CatUnStructured3DPoints_, int SkeletonPointFormat, int CurrentFrame, bool syncedMode_, int ShutterType)
{
	Path = inPath;
	selectedCams = SelectedCams;
	nCams = (int)SelectedCams.size(), startTime = startF, maxTime = stopF, increF = increF_;
	hasColor = hasColor_, drawPatchNormal = hasPatchNormal_, colorCoded = colorCoded_, syncedMode = syncedMode_;
	hasTimeEvoling3DPoints = hasTimeEvoling3DPoints_, CatUnStructured3DPoints = CatUnStructured3DPoints_;
	PointFormat = SkeletonPointFormat;
	syncedMode = 0;

	int selected;  double fps;
	CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	char Fname[512];
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps;
			CamTimeInfo[selected].y = temp;
			CamTimeInfo[selected].z = 1.0;
		}
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
			{
				CamTimeInfo[selected].x = 1.0 / fps;
				CamTimeInfo[selected].y = temp;
				CamTimeInfo[selected].z = 1.0;
			}
			fclose(fp);
		}
		else
		{
			double fps; int temp;
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				{
					CamTimeInfo[selected].x = 1.0 / fps;
					CamTimeInfo[selected].y = temp;
					CamTimeInfo[selected].z = 1.0;
				}
				fclose(fp);
			}
			else
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
		}
	}

	sprintf(Fname, "%s/CamTimingPara.txt", Path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int cid; double framerate, rs_Percent;
		while (fscanf(fp, "%d %lf %lf ", &cid, &framerate, &rs_Percent) != EOF)
			CamTimeInfo[cid].x = framerate, CamTimeInfo[cid].z = rs_Percent;
		fclose(fp);
	}

	//if (syncedMode == 1)
	//	for (int ii = 0; ii < nCams; ii++)
	//		CamTimeInfo[ii].x = 1.0;

	refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	/*sprintf(Fname, "%s/People/@1/0/SelectedNearestViews.txt", inPath);
	if (IsFileExist(Fname))
	{
		int nn, refFid, rcid, rfid;
		double angle, dist, cenX, cenY;
		struct CidFidAngleDist
		{
			CidFidAngleDist(int cid_, int fid_, double angle_, double dist_, double cenX_, double cenY_)
			{
				cid = cid_, fid = fid_;
				angle = angle_, dist = dist_, cenX = cenX_, cenY = cenY_;
			}
			int cid, fid;
			double angle, dist, cenX, cenY;
		};
		vector<CidFidAngleDist>  *NearestCamPerSkeleton = new vector<CidFidAngleDist>[stopF + 1];
		sprintf(Fname, "%s/People/@%d/%d/nearestViews.txt", Path, increF, 0); FILE * fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d ", &refFid, &nn) != EOF)
		{
			if (refFid < startF)
			{
				for (int ii = 0; ii < nn; ii++)
					fscanf(fp, "%d %d %lf %lf %lf %lf", &rcid, &rfid, &angle, &dist, &cenX, &cenY);
				continue;
			}
			if (refFid > stopF)
				break;
			for (int ii = 0; ii < nn; ii++)
			{
				fscanf(fp, "%d %d %lf %lf %lf %lf", &rcid, &rfid, &angle, &dist, &cenX, &cenY);
				NearestCamPerSkeleton[refFid].push_back(CidFidAngleDist(rcid, rfid, angle, dist, cenX, cenX));
			}
		}
		fclose(fp);

		int cid;
		VnearestCams.resize(stopF + 1);

		sprintf(Fname, "%s/People/@1/0/SelectedNearestViews.txt", inPath);	fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %d %d %d ", &refFid, &cid, &rfid, &nn) != EOF)
		{
			VnearestCams[refFid].emplace_back(cid, rfid);
			for (int ii = 0; ii < nn; ii++)
			{
				fscanf(fp, "%d %d ", &cid, &rfid);
				bool found = false;
				for (int jj = 0; jj < NearestCamPerSkeleton[refFid].size() && !found; jj++)
				{
					if (NearestCamPerSkeleton[refFid][jj].cid == cid && NearestCamPerSkeleton[refFid][jj].angle < 0)
						found = true;
				}
				if (!found)
					VnearestCams[refFid].emplace_back(cid, rfid);
			}
		}
		fclose(fp);
	}*/

	ReadCurrentPosesGL(Path, startF, stopF, CamTimeInfo, ShutterType);
	ReadCurrentSfmGL(Path, hasColor, drawPatchNormal);
	//ReadAll3DSkeleton(Path, startF, stopF, PointFormat);

	//ReadCurrentPosesGL3(Path, startF, stopF, CamTimeInfo, ShutterType);
	//ReadOCR3D(Path);
	//ReadCurrentSfmGL2(Path, hasColor, drawPatchNormal);
	//ReadGroup(Path);

	UnitScale = sqrt(pow(PointsVAR[0], 2) + pow(PointsVAR[1], 2) + pow(PointsVAR[2], 2)) / 2000.0;
	g_coordAxisLength = 20.f*UnitScale, g_fViewDistance = 1000 * UnitScale* VIEWING_DISTANCE_MIN;
	g_nearPlane = 1.0*UnitScale, g_farPlane = 30000.f * UnitScale;
	CameraSize = 5.0f*UnitScale, pointSize = 1.0f, normalSize = 5.f*UnitScale, arrowThickness = .1f*UnitScale;
	Discontinuity3DThresh = 20.0*UnitScale;

	Point3i f;
	fp = fopen("smpl/faces.txt", "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %d %d ", &f.x, &f.y, &f.z) != EOF)
			faces.push_back(f);
		fclose(fp);
	}
	if (!ReadSMPLData("smpl", mySMPL))
		printLOG("Check smpl Path.\n");
	else
	{
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		dVdt = Eigen::kroneckerProduct(VectorXd::Ones(smpl::SMPLModel::nVertices), eye3);
	}
	AllV = new MatrixXdr[nMaxPeople];
	for (int pi = 0; pi < nMaxPeople; pi++)
		AllV[pi].resize(SMPLModel::nVertices, 3);

	//ReadPerCurrent3DSkeleton(Path, CurrentFrame, PointFormat);
	//ReadAllCurrent3DSkeleton(Path, CurrentFrame);
	//Read3DTrajectory(Path, 0, DoColorCode);
	//Read3DTrajectoryPeople(Path, startF, stopF, 0); //for cvpr18

	if (SortedTimeInstance != NULL)
		delete[]SortedTimeInstance;
	SortedTimeInstance = new int[vAll3D_TidCidFid.size()];
	{
		double *ts = new double[vAll3D_TidCidFid.size()];
		for (int ii = 0; ii < vAll3D_TidCidFid.size(); ii++)
			SortedTimeInstance[ii] = ii, ts[ii] = vAll3D_TidCidFid[ii].Ts;
		Quick_Sort_Double(ts, SortedTimeInstance, 0, vAll3D_TidCidFid.size() - 1);
		delete[]ts;
	}

	if (showSMPL)
	{
		AllPeopleMeshTimeId = new vector<int>[nMaxPeople];
		AllPeopleVertexTimeId = new vector<int>[nMaxPeople];
		AllPeopleMesh = new vector<Point3f>[nMaxPeople];
		AllPeopleVertex = new vector<Point3f>[nMaxPeople];
		if (SMPL_mode == 3)
		{
			for (int ii = 0; ii < 3000; ii++)
				vAll3D_TidCidFid.push_back(TsCidFid(ii, ii%nCams, ii));
		}
		else
		{
			for (int peopleCount = 0; peopleCount < nMaxPeople; peopleCount++)
			{
				if (SMPL_mode == 0)
					sprintf(Fname, "%s/FitBody/@%d/P/%d", Path, increF, peopleCount);
				else if (SMPL_mode == 1)
					sprintf(Fname, "%s/FitBody/@%d/Wj/%d", Path, increF, peopleCount);
				else if (SMPL_mode == 2)
					sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d", Path, increF, peopleCount);

				vector<std::string> vnames;
#ifdef _WINDOWS
				vnames = get_all_files_names_within_folder(std::string(Fname));
#endif 
				if (vnames.size() == 0)
					break;

				for (int ii = 0; ii < vnames.size(); ii++)
				{
					//sprintf(Fname, "%s/FitBody/%d/sffj_us_BodyParameters_%.2d_%.4d_%.16d.txt", Path, pi, vPoseLandmark[jj].viewID, vPoseLandmark[jj].frameID, (int)(TScale*vPoseLandmark[jj].ts + 0.5));
					std::string CidString = vnames[ii].substr(0, 2);
					std::string FidString = vnames[ii].substr(3, 4);

					std::string str2(".");
					std::size_t pos = vnames[ii].find(str2);
					std::string TsString = vnames[ii].substr(8, pos - 8);

					bool found = false;
					if (syncedMode)
					{
						int cid_ = stoi(CidString), fid_ = stoi(FidString);
						int refFid_ = MyFtoI(CamTimeInfo[cid_].y + 1.0*fid_ / CamTimeInfo[cid_].x*CamTimeInfo[refCid].x); //(CamTimeInfo[cid_].y / CamTimeInfo[refCid].x + 1.0*fid_ / CamTimeInfo[cid_].x)*CamTimeInfo[refCid].x;
																														  //refFid_ = fid_ + (int)(CamTimeInfo[cid_].y + 0.5);
						for (int jj = 0; jj < vAll3D_TidCidFid.size() && !found; jj++)
						{
							int cid__ = vAll3D_TidCidFid[jj].Cid, fid__ = vAll3D_TidCidFid[jj].Fid, refFid__ = MyFtoI(CamTimeInfo[cid__].y + 1.0*fid__ / CamTimeInfo[cid__].x*CamTimeInfo[refCid].x);
							if (refFid_ == refFid__)
								found = true;
						}
					}
					else
					{
						for (int jj = 0; jj < vAll3D_TidCidFid.size() && !found; jj++)
							if (abs(stod(TsString) - vAll3D_TidCidFid[jj].Ts) < 2)
								found = true;
					}
					if (!found)
						vAll3D_TidCidFid.push_back(TsCidFid(stod(TsString), stoi(CidString), stoi(FidString)));
				}
			}
		}
	}

	vector< SMPLParams> _gVparams;
	vector<std::string> vnames;
#ifdef _WINDOWS
	//sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d", Path, increF, 0);
	sprintf(Fname, "%s/FitBody/@%d/Wj/%d", Path, increF, 0);
	vnames = get_all_files_names_within_folder(std::string(Fname));
#endif 

	vector<int> vTS;
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints;
	for (int ii = 0; ii < vnames.size(); ii++)
	{
		std::string CidString = vnames[ii].substr(0, 2);
		std::string FidString = vnames[ii].substr(3, 4);

		std::string str2(".");
		std::size_t pos = vnames[ii].find(str2);
		std::string TsString = vnames[ii].substr(8, pos - 8);

		vTS.push_back(stoi(TsString));
		bool found = false;
		int cid_ = stoi(CidString), fid_ = stoi(FidString);
		SMPLParams params;

		sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%s", Path, increF, 0, vnames[ii].c_str());
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%lf %lf %lf %lf ", &params.scale, &params.t(0), &params.t(1), &params.t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &params.pose(ii, 0), &params.pose(ii, 1), &params.pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fscanf(fp, "%lf ", &params.coeffs(ii));
		fclose(fp);

		_gVparams.push_back(params);
	}
	double *ts = new double[vAll3D_TidCidFid.size()];
	int *t_id = new int[_gVparams.size()];
	for (int ii = 0; ii < _gVparams.size(); ii++)
		t_id[ii] = ii;
	if (vTS.size() > 0)
		Quick_Sort_Int(&vTS[0], t_id, 0, _gVparams.size() - 1);
	for (int ii = 0; ii < _gVparams.size(); ii++)
		gVparams.push_back(_gVparams[t_id[ii]]);

	//GetCurrent3DBody("C:/temp/CMU/01/01_01.txt", 0);// ReadCurrent3DBody(Path, timeID);

	if (SortedTimeInstance != NULL)
		delete[]SortedTimeInstance;
	SortedTimeInstance = new int[vAll3D_TidCidFid.size()];
	{
		double *ts = new double[vAll3D_TidCidFid.size()];
		for (int ii = 0; ii < vAll3D_TidCidFid.size(); ii++)
			SortedTimeInstance[ii] = ii, ts[ii] = vAll3D_TidCidFid[ii].Ts;
		Quick_Sort_Double(ts, SortedTimeInstance, 0, vAll3D_TidCidFid.size() - 1);
		delete[]ts;
	}

	if (syncedMode == 1)
	{
		for (int ii = 0; ii < (int)vAll3D_TidCidFid.size(); ii++)
		{
			int cid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Fid;
			double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
			refFid = MyFtoI(ts*CamTimeInfo[refCid].x);
			if (cid == refCid && refFid == CurrentFrame)
			{
				timeID = ii;
				break;
			}
		}
	}
	else
		timeID = 0;

	if (vAll3D_TidCidFid.size() > 0)
		printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts / Tscale, timeID);

	if (showImg)
	{
		g_backgroundTexture.resize(nCams);
		int cid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
		double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
		for (int ii = 0; ii < nCams; ii++)
		{
			//int fidj = fid* CamTimeInfo[cid].x / CamTimeInfo[ii].x;
			int fidj = (int)((ts - CamTimeInfo[ii].y / CamTimeInfo[refCid].x) / CamTimeInfo[ii].x + 0.5);
			char Fname[512]; sprintf(Fname, "%s/MP/%d/%d.jpg", Path, ii, fidj);
			GenerateBackgroundTexture(Fname, cid);
		}
	}

	Visualization();
	destroyAllWindows();

	return 0;
}
int visualizationDriver2(char *inPath, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, vector<int> &SelectedCams, int startF, int stopF, int increF_, bool hasColor_, int colorCoded_, bool hasPatchNormal_, bool hasTimeEvoling3DPoints_, bool CatUnStructured3DPoints_, int SkeletonPointFormat, int CurrentFrame, bool syncedMode_)
{

	g_vis.PointPosition3.resize(nMaxPeople * SMPLModel::nVertices);
	Path = inPath;
	selectedCams = SelectedCams;
	nCams = (int)SelectedCams.size(), startTime = startF, maxTime = stopF, increF = increF_;
	hasColor = hasColor_, drawPatchNormal = hasPatchNormal_, colorCoded = colorCoded_, syncedMode = syncedMode_;
	hasTimeEvoling3DPoints = hasTimeEvoling3DPoints_, CatUnStructured3DPoints = CatUnStructured3DPoints_;
	PointFormat = SkeletonPointFormat;

	int selected, temp;  double fps;
	CamTimeInfo = new Point3d[nCams];
	for (int ii = 0; ii < nCams; ii++)
		CamTimeInfo[ii].x = 1.0, CamTimeInfo[ii].y = 0.0;
	char Fname[512];

	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
		{
			CamTimeInfo[selected].x = 1.0 / fps;
			CamTimeInfo[selected].y = temp;
			CamTimeInfo[selected].z = 1.0;
		}
		fclose(fp);
	}
	else
		printLOG("Cannot load time stamp info. Assume no frame offsets!");

	sprintf(Fname, "%s/CamTimingPara.txt", Path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int cid; double framerate, rs_Percent;
		while (fscanf(fp, "%d %lf %lf ", &cid, &framerate, &rs_Percent) != EOF)
			CamTimeInfo[cid].x = framerate, CamTimeInfo[cid].z = rs_Percent;
		fclose(fp);
	}

	refCid = 0;
	double earliest = DBL_MAX;
	for (int ii = 0; ii < nCams; ii++)
		if (earliest > CamTimeInfo[ii].y)
			earliest = CamTimeInfo[ii].y, refCid = ii;

	ReadCurrentPosesGL2(Path, SelectedCamNames, SeqId, CamIdsPerSeq, startF, stopF, CamTimeInfo);
	ReadCurrentSfmGL(Path, hasColor, drawPatchNormal);
	//ReadAll3DSkeleton(Path, startF, stopF, PointFormat);

	UnitScale = sqrt(pow(PointsVAR[0], 2) + pow(PointsVAR[1], 2) + pow(PointsVAR[2], 2)) / 2000.0;
	g_coordAxisLength = 20.f*UnitScale, g_fViewDistance = 1000 * UnitScale* VIEWING_DISTANCE_MIN;
	g_nearPlane = 1.0*UnitScale, g_farPlane = 30000.f * UnitScale;
	CameraSize = 30.0f*UnitScale, pointSize = 1.0f, normalSize = 5.f*UnitScale, arrowThickness = .1f*UnitScale;
	Discontinuity3DThresh = 20.0*UnitScale;

	Point3i f;
	fp = fopen("smpl/faces.txt", "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d %d %d ", &f.x, &f.y, &f.z) != EOF)
			faces.push_back(f);
		fclose(fp);
	}
	if (!ReadSMPLData("smpl", mySMPL))
		printLOG("Check smpl Path.\n");
	else
	{
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		dVdt = Eigen::kroneckerProduct(VectorXd::Ones(smpl::SMPLModel::nVertices), eye3);
	}
	AllV = new MatrixXdr[nMaxPeople];
	for (int pi = 0; pi < nMaxPeople; pi++)
		AllV[pi].resize(SMPLModel::nVertices, 3);

	/*{
		const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 18;
		int naJoints = SMPLModel::naJoints;

		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

		SMPLParams params;
		vector<SMPLParams>Vparams;
		params.scale = 1; params.t.setZero();
		params.pose.setZero(), params.coeffs.setZero();

		FILE *fp = fopen("E:/shape.txt", "r");
		for (int ii = 0; ii < 10; ii++)
			fscanf(fp, "%lf ", &params.coeffs(ii));
		for (int ii = 0; ii < 72; ii++)
			fscanf(fp, "%lf ", &params.pose(ii));
		fclose(fp);

		Vparams.push_back(params);

		if (Vparams.size() > 0)//keep the last one to avoid interuptingly model disappearing
		{
			g_vis.PointPosition3.clear(), g_vis.PointPosition3.resize(Vparams.size()*nVertices);
			omp_set_num_threads(omp_get_max_threads());
			for (int pi = 0; pi < Vparams.size(); pi++)
			{
				reconstruct(mySMPL, Vparams[pi].coeffs.data(), Vparams[pi].pose.data(), AllV[pi].data());
				Map<VectorXd> V_vec(AllV[pi].data(), AllV[pi].size());
				V_vec = V_vec * Vparams[pi].scale + dVdt * Vparams[pi].t;

				//VectorXd Jsmpl_vec= mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control
				VectorXd Jsmpl_vec = mySMPL.J_regl_25_bigl_col_ * V_vec;
				//cout << Jsmpl_vec << endl;

				FILE *fp = fopen("E:/xyz.txt", "w");
				for (int ii = 0; ii < nVertices; ii++)
					fprintf(fp, "%.4f %.4f %.4f\n", V_vec[3 * ii], V_vec[3 * ii + 1], V_vec[3 * ii + 2]);
				fclose(fp);
			}
		}
	}*/

	//ReadPerCurrent3DSkeleton(Path, CurrentFrame, PointFormat);
	//ReadAllCurrent3DSkeleton(Path, CurrentFrame);
	//Read3DTrajectory(Path, 0, DoColorCode);
	//Read3DTrajectoryPeople(Path, startF, stopF, 0); //for cvpr18

	if (SortedTimeInstance != NULL)
		delete[]SortedTimeInstance;
	SortedTimeInstance = new int[vAll3D_TidCidFid.size()];
	{
		double *ts = new double[vAll3D_TidCidFid.size()];
		for (int ii = 0; ii < vAll3D_TidCidFid.size(); ii++)
			SortedTimeInstance[ii] = ii, ts[ii] = vAll3D_TidCidFid[ii].Ts;
		Quick_Sort_Double(ts, SortedTimeInstance, 0, vAll3D_TidCidFid.size() - 1);
		delete[]ts;
	}

	if (showSMPL)
	{
		AllPeopleMeshTimeId = new vector<int>[nMaxPeople];
		AllPeopleVertexTimeId = new vector<int>[nMaxPeople];
		AllPeopleMesh = new vector<Point3f>[nMaxPeople];
		AllPeopleVertex = new vector<Point3f>[nMaxPeople];
		for (int peopleCount = 0; peopleCount < nMaxPeople; peopleCount++)
		{
			if (SMPL_mode == 0)
				sprintf(Fname, "%s/FitBody/@%d/P/%d", Path, increF, peopleCount);
			else if (SMPL_mode == 1)
				sprintf(Fname, "%s/FitBody/@%d/Wj/%d", Path, increF, peopleCount);
			else if (SMPL_mode == 2)
				sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d", Path, increF, peopleCount);

			vector<std::string> vnames;
#ifdef _WINDOWS
			vnames = get_all_files_names_within_folder(std::string(Fname));
#endif 
			if (vnames.size() == 0)
				break;

			for (int ii = 0; ii < vnames.size(); ii++)
			{
				//sprintf(Fname, "%s/FitBody/%d/sffj_us_BodyParameters_%.2d_%.4d_%.16d.txt", Path, pi, vPoseLandmark[jj].viewID, vPoseLandmark[jj].frameID, (int)(TScale*vPoseLandmark[jj].ts + 0.5));
				std::string CidString = vnames[ii].substr(0, 2);
				std::string FidString = vnames[ii].substr(3, 4);

				std::string str2(".");
				std::size_t pos = vnames[ii].find(str2);
				std::string TsString = vnames[ii].substr(8, pos - 8);

				bool found = false;
				if (syncedMode)
				{
					int cid_ = stoi(CidString), fid_ = stoi(FidString);
					int refFid_ = MyFtoI(CamTimeInfo[cid_].y + 1.0*fid_ / CamTimeInfo[cid_].x*CamTimeInfo[refCid].x); //(CamTimeInfo[cid_].y / CamTimeInfo[refCid].x + 1.0*fid_ / CamTimeInfo[cid_].x)*CamTimeInfo[refCid].x;
																													  //refFid_ = fid_ + (int)(CamTimeInfo[cid_].y + 0.5);
					for (int jj = 0; jj < vAll3D_TidCidFid.size() && !found; jj++)
					{
						int cid__ = vAll3D_TidCidFid[jj].Cid, fid__ = vAll3D_TidCidFid[jj].Fid, refFid__ = MyFtoI(CamTimeInfo[cid__].y + 1.0*fid__ / CamTimeInfo[cid__].x*CamTimeInfo[refCid].x);
						if (refFid_ == refFid__)
							found = true;
					}
				}
				else
				{
					for (int jj = 0; jj < vAll3D_TidCidFid.size() && !found; jj++)
						if (abs(stod(TsString) - vAll3D_TidCidFid[jj].Ts) < 2)
							found = true;
				}
				if (!found)
					vAll3D_TidCidFid.push_back(TsCidFid(stod(TsString), stoi(CidString), stoi(FidString)));
			}
		}
	}

	if (SortedTimeInstance != NULL)
		delete[]SortedTimeInstance;
	SortedTimeInstance = new int[vAll3D_TidCidFid.size()];
	{
		double *ts = new double[vAll3D_TidCidFid.size()];
		for (int ii = 0; ii < vAll3D_TidCidFid.size(); ii++)
			SortedTimeInstance[ii] = ii, ts[ii] = vAll3D_TidCidFid[ii].Ts;
		Quick_Sort_Double(ts, SortedTimeInstance, 0, vAll3D_TidCidFid.size() - 1);
		delete[]ts;
	}

	if (syncedMode == 1)
	{
		for (int ii = 0; ii < (int)vAll3D_TidCidFid.size(); ii++)
		{
			int cid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Cid, fid = vAll3D_TidCidFid[SortedTimeInstance[ii]].Fid;
			double ts = CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*fid / CamTimeInfo[cid].x;
			refFid = MyFtoI(ts*CamTimeInfo[refCid].x);
			if (refFid == CurrentFrame)
			{
				timeID = ii;
				break;
			}
		}
	}
	else
		timeID = 0;

	if (vAll3D_TidCidFid.size() > 0)
		printLOG("Current time: %.2f (id: %d)\n", vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts / Tscale, timeID);

	Visualization();
	destroyAllWindows();

	return 0;
}

int selectedPeople = 0;
void ReadCurrentSfmGL(char *Path, bool hasColor, bool drawPatchNormal)
{
	char Fname[512];
	int viewID, nviews;

	CameraData temp;
	sprintf(Fname, "%s/Corpus/DinfoGL.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d ", &viewID) != EOF)
		{
			for (int jj = 0; jj < 16; jj++)
				fscanf(fp, "%lf ", &temp.Rgl[jj]);
			for (int jj = 0; jj < 3; jj++)
				fscanf(fp, "%lf ", &temp.camCenter[jj]);
			temp.viewID = viewID;
			g_vis.glCorpusCameraInfo.push_back(temp);
		}
		fclose(fp);
	}
	else
	{
		printLOG("Cannot load %s. Try with %s/BA_Camera_AllParams_after.txt ...", Fname, Path);

		Corpus CorpusInfo;
		sprintf(Fname, "%s/Corpus/BA_Camera_AllParams_after.txt", Path);
		if (readBundleAdjustedNVMResults(Fname, CorpusInfo))
		{
			for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
			{
				if (abs(CorpusInfo.camera[ii].camCenter[0]) + abs(CorpusInfo.camera[ii].camCenter[1]) + abs(CorpusInfo.camera[ii].camCenter[2]) < 1e-16)
					continue;
				GetRCGL(CorpusInfo.camera[ii].R, CorpusInfo.camera[ii].T, CorpusInfo.camera[ii].Rgl, CorpusInfo.camera[ii].camCenter);
				for (int jj = 0; jj < 16; jj++)
					temp.Rgl[jj] = CorpusInfo.camera[ii].Rgl[jj];
				for (int jj = 0; jj < 3; jj++)
					temp.camCenter[jj] = CorpusInfo.camera[ii].camCenter[jj];
				temp.viewID = ii;
				g_vis.glCorpusCameraInfo.push_back(temp);
			}
			printLOG("succeeded.\n");
		}
		else
			printLOG("Cannot load %s\n", Fname);
	}

	g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(10e5);
	if (hasColor)
		g_vis.CorpusPointColor.clear(), g_vis.CorpusPointColor.reserve(10e5);

	int pid, dummy;
	Point3i iColor; Point3f fColor; Point3f t3d;
	bool filenotvalid = false;
	sprintf(Fname, "%s/Corpus/MVS.ply", Path);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/Corpus/3dGL.xyz", Path);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/Corpus/n3dGL.xyz", Path);
			if (IsFileExist(Fname) == 0)
			{
				printLOG("Cannot load %s\n", Fname);
				filenotvalid = true;
			}
		}
		if (!filenotvalid)
		{
			fp = fopen(Fname, "r");
			printLOG("Loaded %s\n", Fname);
			while (fscanf(fp, "%d %f %f %f ", &pid, &t3d.x, &t3d.y, &t3d.z) != EOF)
				//while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
			{
				if (hasColor)
				{
					fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
					fColor.x = 1.0f*iColor.x / 255;
					fColor.y = 1.0f*iColor.y / 255;
					fColor.z = 1.0f*iColor.z / 255;
					g_vis.CorpusPointColor.push_back(fColor);
				}
				PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
				g_vis.CorpusPointPosition.push_back(t3d);
			}
			fclose(fp);
		}
	}
	else
	{
		unsigned int npts, color = 0;
		ifstream fin;
		fin.open(Fname, ios::binary);
		if (!fin.is_open())
			return;
		fin.read(reinterpret_cast<char *>(&npts), sizeof(unsigned int)), fin.read(reinterpret_cast<char *>(&color), sizeof(unsigned int));
		g_vis.CorpusPointPosition.reserve(npts);
		if (color > 0)
			g_vis.CorpusPointColor.reserve(npts);
		for (unsigned int j = 0; j < npts; ++j)
		{
			fin.read(reinterpret_cast<char *>(&t3d.x), sizeof(float));
			fin.read(reinterpret_cast<char *>(&t3d.y), sizeof(float));
			fin.read(reinterpret_cast<char *>(&t3d.z), sizeof(float));

			if (color > 0)
			{
				fin.read(reinterpret_cast<char *>(&fColor.x), sizeof(float));
				fin.read(reinterpret_cast<char *>(&fColor.y), sizeof(float));
				fin.read(reinterpret_cast<char *>(&fColor.z), sizeof(float));
				g_vis.CorpusPointColor.push_back(fColor);
			}
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
			g_vis.CorpusPointPosition.push_back(t3d);
		}
		fin.close();

		/*fp = fopen(Fname, "r");
		printLOG("Loaded %s\n", Fname);
		while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
		{
		if (hasColor)
		{
		fscanf(fp, "%d %d %d %d ", &iColor.x, &iColor.y, &iColor.z, &dummy);
		fColor.x = 1.0f*iColor.x / 255;
		fColor.y = 1.0f*iColor.y / 255;
		fColor.z = 1.0f*iColor.z / 255;
		g_vis.CorpusPointColor.push_back(fColor);
		}
		PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.CorpusPointPosition.push_back(t3d);
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus/MVS_.ply", Path);
		ofstream fout; fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
		cout << "Cannot write: " << Fname << endl;
		return;
		}
		npts = g_vis.CorpusPointPosition.size(), color = 1;
		fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
		if (hasColor)
		fout.write(reinterpret_cast<char *>(&color), sizeof(int));
		else
		{
		color = 0;
		fout.write(reinterpret_cast<char *>(&color), sizeof(int));
		}
		for (int j = 0; j < npts; ++j)
		{
		fout.write(reinterpret_cast<char *>(&g_vis.CorpusPointPosition[j].x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&g_vis.CorpusPointPosition[j].y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&g_vis.CorpusPointPosition[j].z), sizeof(float));
		if (hasColor)
		{
		fout.write(reinterpret_cast<char *>(&g_vis.CorpusPointColor[j].x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&g_vis.CorpusPointColor[j].y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&g_vis.CorpusPointColor[j].z), sizeof(float));
		}
		}
		fout.close();*/
	}

	if (g_vis.CorpusPointPosition.size() > 0)
	{
		PointsCentroid[0] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[1] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[2] /= g_vis.CorpusPointPosition.size();

		PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
		for (int ii = 0; ii < g_vis.CorpusPointPosition.size(); ii++)
		{
			PointsVAR[0] += pow(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], 2);
			PointsVAR[1] += pow(g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], 2);
			PointsVAR[2] += pow(g_vis.CorpusPointPosition[ii].z - PointsCentroid[2], 2);
		}
		PointsVAR[0] = sqrt(PointsVAR[0] / g_vis.CorpusPointPosition.size());
		PointsVAR[1] = sqrt(PointsVAR[1] / g_vis.CorpusPointPosition.size());
		PointsVAR[2] = sqrt(PointsVAR[2] / g_vis.CorpusPointPosition.size());
	}
	else
		PointsCentroid[0] = PointsCentroid[1] = PointsCentroid[2] = 0;

	for (int cid = 0; cid < 0; cid++)
	{
		sprintf(Fname, "%s/%d/Video3DCorpus.xyz", Path, cid);
		if (IsFileExist(Fname) == 1)
		{
			fColor = Point3f(1.0*(rand() % 255) / 255.0, 1.0*(rand() % 255) / 255.0, 1.0*(rand() % 255) / 255.0);
			FILE *fp = fopen(Fname, "r");
			printLOG("Loaded %s\n", Fname);
			while (fscanf(fp, "%d %d %f %f %f ", &pid, &pid, &t3d.x, &t3d.y, &t3d.z) != EOF)
			{
				if (abs(t3d.x) + abs(t3d.y) + abs(t3d.z) > 1e-12 &&abs(t3d.x) + abs(t3d.y) + abs(t3d.z) < 1e6)
				{
					g_vis.CorpusPointColor.push_back(fColor);
					g_vis.CorpusPointPosition.push_back(t3d);
				}
			}
			fclose(fp);
		}
	}

	return;
}
bool ReadCurrent3DGL(char *Path, bool hasColor, bool drawPatchNormal, int timeID, bool setCoordinate)
{
	char Fname[512];
	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (hasColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	if (setCoordinate)
		PointsCentroid[0] = 0.0f, PointsCentroid[1] = 0.0f, PointsCentroid[2] = 0.f;

	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	/*sprintf(Fname, "%s/Dynamic/3dg_%.4d.xyz", Path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
	printLOG("Cannot load %s\n", Fname);
	return false;
	}
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
	if (drawPatchNormal)
	{
	fscanf(fp, "%f %f %f ", &n3d.x, &n3d.y, &n3d.z);
	g_vis.PointNormal.push_back(n3d);
	}
	if (hasColor)
	{
	fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
	fColor.x = 1.0*iColor.x / 255;
	fColor.y = 1.0*iColor.y / 255;
	fColor.z = 1.0*iColor.z / 255;
	g_vis.PointColor.push_back(fColor);
	}
	else
	g_vis.PointColor.push_back(Point3f(255, 0, 0));

	if (setCoordinate)
	PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
	g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);*/

	sprintf(Fname, "%s/Dynamic/ClusteredFaces_%.4d.txt", Path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return false;
	}
	int ii, jj;
	while (fscanf(fp, "%d %d %f %f %f ", &jj, &ii, &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		//Point3f PointColor(0.2f, 0.7f, 1.0f); //light blue
		//Point3f PointColor(1.0f, 0.5f, 0.2f); //light ora
		if (jj == 1)
			g_vis.PointColor.push_back(Point3f(0.2f, 0.7f, 1.0f));
		else
			g_vis.PointColor.push_back(Point3f(1.0f, 0.5f, 0.2));

		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	if (setCoordinate)
		PointsCentroid[0] /= g_vis.PointPosition.size(), PointsCentroid[1] /= g_vis.PointPosition.size(), PointsCentroid[2] /= g_vis.PointPosition.size();

	//Concatenate points in case the trajectory mode is used
	if (CatUnStructured3DPoints) //in efficient of there are many points
	{
		if (g_vis.catPointPosition == NULL)
			g_vis.catPointPosition = new vector<Point3f>[g_vis.PointPosition.size()];
		if (timeID == g_vis.catPointPosition[0].size())
			for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
				g_vis.catPointPosition[ii].push_back(g_vis.PointPosition[ii]);
	}

	return true;
}
bool ReadCurrent3DGL2(char *Path, bool hasColor, bool drawPatchNormal, int timeID, bool setCoordinate)
{
	char Fname[512];
	g_vis.PointPosition2.clear(); g_vis.PointPosition2.reserve(10e5);
	if (hasColor)
		g_vis.PointColor2.clear(), g_vis.PointColor2.reserve(10e5);

	if (setCoordinate)
		PointsCentroid[0] = 0.0f, PointsCentroid[1] = 0.0f, PointsCentroid[2] = 0.f;
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;

	sprintf(Fname, "%s/Dynamic/ClusteredPoseLandmark_%d_%.4d.txt", Path, timeID, 0); FILE *fp = fopen(Fname, "r");
	//sprintf(Fname, "%s/3DPoints/%.4d.txt", Path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return false;
	}
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (drawPatchNormal)
		{
			fscanf(fp, "%f %f %f ", &n3d.x, &n3d.y, &n3d.z);
			g_vis.PointNormal2.push_back(n3d);
		}
		if (hasColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x;
			fColor.y = 1.0*iColor.y;
			fColor.z = 1.0*iColor.z;
			g_vis.PointColor2.push_back(fColor);
		}

		if (setCoordinate)
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition2.push_back(t3d);
	}
	fclose(fp);

	if (setCoordinate)
		PointsCentroid[0] /= g_vis.PointPosition2.size(), PointsCentroid[1] /= g_vis.PointPosition2.size(), PointsCentroid[2] /= g_vis.PointPosition2.size();

	//Concatenate points in case the trajectory mode is used
	if (CatUnStructured3DPoints) //in efficient when there are many points
	{
		if (g_vis.catPointPosition2 == NULL)
			g_vis.catPointPosition2 = new vector<Point3f>[g_vis.PointPosition.size()];
		for (int ii = 0; ii < g_vis.PointPosition2.size(); ii++)
			g_vis.catPointPosition2[ii].push_back(g_vis.PointPosition2[ii]);
	}

	return true;
}
bool ReadAllCurrent3DSkeleton(char *Path, int timeID, int PointFormat)
{
	char Fname[512];
	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);

	vector<int> ValidPersonID;
	sprintf(Fname, "%s/VoxelVoting/A/%.4d.txt", Path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int pid = 0, ninliers, dummy;
		while (fscanf(fp, "%d ", &ninliers) != EOF)
		{
			for (int ii = 0; ii < ninliers; ii++)
				fscanf(fp, "%d %d ", &dummy, &dummy);
			if (ninliers > 4)
			{
				ValidPersonID.push_back(pid);
				pid++;
			}
		}
		fclose(fp);
	}

	Point3f t3d;
	sprintf(Fname, "%s/VoxelVoting/%.4d.txt", Path, timeID); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int pid = 0;
		vector<Point3f> v3d;
		while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
		{
			v3d.clear();  v3d.push_back(t3d);
			for (int ii = 1; ii < PointFormat; ii++)
			{
				fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z);
				v3d.push_back(t3d);
			}

			for (auto p : ValidPersonID)
			{
				if (p == pid)
				{
					for (int ii = 0; ii < PointFormat; ii++)
						g_vis.PointPosition.push_back(v3d[ii]);
					break;
				}
			}
			pid++;
		}
		fclose(fp);
	}
	return true;
}
bool ReadPerCurrent3DSkeleton(char *Path, int timeID, int PointFormat)
{
	char Fname[512];

	int pid = selectedPeople, nvis = 0; float avgErr;
	cPeopleId.clear();
	bool first = true;
	Point3f t3d;
	while (true)
	{
		if (pid > 10)
			break;

		//sprintf(Fname, "%s/JBC/%.4d/PoseLandmark_%d.txt", Path, timeID, pid); 
		sprintf(Fname, "%s/People/@%d/%d/m_%.4d.txt", Path, increT, pid, timeID);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/People/@%d/%d/f_%.4d.txt", Path, increT, pid, timeID);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/People/@%d/%d/%.4d.txt", Path, increT, pid, timeID);
				if (IsFileExist(Fname) == 0)
					break;
			}
		}
		if (first)
		{
			first = false;
			g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
			printLOG("Load %s\n", Fname);
		}

		vPeopleId.push_back(pid);
		std::string line, item;
		std::ifstream file(Fname);
		while (std::getline(file, line))
		{
			if (line.empty())
				break;
			StringTrim(&line);//remove white space
			std::stringstream line_stream(line);
			while (!line_stream.eof()) {
				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				t3d.x = atof(item.c_str());

				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				t3d.y = atof(item.c_str());

				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				t3d.z = atof(item.c_str());

				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				avgErr = atof(item.c_str());

				std::getline(line_stream, item, ' ');
				StringTrim(&item);
				nvis = atoi(item.c_str());
			}
			g_vis.PointPosition.push_back(t3d);
			std::getline(file, line);
		}
		file.close();
		cPeopleId.push_back(pid);
		pid++;
	}
	printLOG("%d..", timeID);

	return true;
}
bool ReadCurrent3DBody(char *Path, int timeID)
{
	char Fname[512];

	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 18;
	int naJoints = SMPLModel::naJoints;

	SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
	SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

	SMPLParams params;
	vector<SMPLParams>Vparams;
	vector<int> validPid;
	int peopleCount = selectedPeople;
	while (true)
	{
		if (peopleCount > 10)
			break;

		int rcid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, rfid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
		int reffid = MyFtoI((CamTimeInfo[rcid].y / CamTimeInfo[refCid].x + 1.0*rfid / CamTimeInfo[rcid].x)*CamTimeInfo[refCid].x);	//int reffid = rfid + (int)(CamTimeInfo[rcid].y + 0.5); 	//ts = (1.0*refFid + offset) * ifps or fid~refFid+offset
		//double ts = round(Tscale* (CamTimeInfo[rcid].y / CamTimeInfo[refCid].x + 1.0*rfid / CamTimeInfo[rcid].x)*CamTimeInfo[refCid].x);
		double ts = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Ts;

		if (SMPL_mode == 0)
			sprintf(Fname, "%s/FitBody/@%d/P/%d/%.2d_%.4d_%.1f.txt", Path, increF, peopleCount, 0, reffid, round(Tscale*reffid));
		//sprintf(Fname, "%s/FitBody/@%d/i/%d/%.4d.txt", Path, increF, peopleCount, reffid);
		else if (SMPL_mode == 1)
			sprintf(Fname, "%s/FitBody/@%d/Wj/%d/%.2d_%.4d_%.1f.txt", Path, increF, peopleCount, 0, reffid, round(Tscale*reffid));
		else if (SMPL_mode == 2)
			sprintf(Fname, "%s/FitBody/@%d/US_Smoothing/%d/%.2d_%.4d_%.1f.txt", Path, increF, peopleCount, rcid, rfid, ts);
		if (IsFileExist(Fname) == 0)
		{
			peopleCount++;
			continue;
		}
		validPid.push_back(peopleCount);
		printf("%s\n", Fname);
		FILE *fp = fopen(Fname, "r");
		fscanf(fp, "%lf %lf %lf %lf ", &params.scale, &params.t(0), &params.t(1), &params.t(2));
		for (int ii = 0; ii < nJointsSMPL; ii++)
			fscanf(fp, "%lf %lf %lf ", &params.pose(ii, 0), &params.pose(ii, 1), &params.pose(ii, 2));
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fscanf(fp, "%lf ", &params.coeffs(ii));
		//params.pose(15, 0) = 0.3, params.pose(15, 1) = 0, params.pose(15, 2) = 0;//up straight face
		fclose(fp);

		Vparams.push_back(params);
		peopleCount++;
	}

	if (Vparams.size() > 0)//keep the last one to avoid interuptingly model disappearing
	{
		int rcid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Cid, rfid = vAll3D_TidCidFid[SortedTimeInstance[timeID]].Fid;
		int reffid = MyFtoI((CamTimeInfo[rcid].y / CamTimeInfo[refCid].x + 1.0*rfid / CamTimeInfo[rcid].x)*CamTimeInfo[refCid].x);	//int reffid = rfid + (int)(CamTimeInfo[rcid].y + 0.5); 	//ts = (1.0*refFid + offset) * ifps or fid~refFid+offset


		g_vis.PointPosition3.resize(Vparams.size() * SMPLModel::nVertices);
		omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic,1)
		for (int pi = 0; pi < Vparams.size(); pi++)
		{
			reconstruct(mySMPL, Vparams[pi].coeffs.data(), Vparams[pi].pose.data(), AllV[pi].data());
			Map<VectorXd> V_vec(AllV[pi].data(), AllV[pi].size());
			V_vec = V_vec * Vparams[pi].scale + dVdt * Vparams[pi].t;

			VectorXd Jsmpl_vec;
			Jsmpl_vec = mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control

			for (int ii = 0; ii < nVertices; ii++)
				g_vis.PointPosition3[ii + validPid[pi] * nVertices] = Point3f(AllV[pi](ii, 0), AllV[pi](ii, 1), AllV[pi](ii, 2));

			//AllPeopleVertexTimeId[pi].emplace_back(reffid);
			//for (int ii = 0; ii < naJoints; ii++)
				//AllPeopleVertex[pi].emplace_back(Jsmpl_vec(3 * ii), Jsmpl_vec(3 * ii + 1), Jsmpl_vec(3 * ii + 2));
		}
	}

	return true;
}
bool GetCurrent3DBody(char *Fname, int timeID)
{
	const int nVertices = SMPLModel::nVertices, nShapeCoeffs = SMPLModel::nShapeCoeffs, nJointsSMPL = SMPLModel::nJoints, nJointsCOCO = 18;
	int naJoints = SMPLModel::naJoints;

	if (gVparams.size() == 0)
	{
		SMPLParams params;

		if (IsFileExist(Fname) == 0)
			return false;

		FILE *fp = fopen(Fname, "r");
		for (int ii = 0; ii < nShapeCoeffs; ii++)
			fscanf(fp, "%lf ", &params.coeffs(ii));
		int tid;
		while (fscanf(fp, "%d ", &tid) != EOF)
		{
			params.scale = 1.0;
			fscanf(fp, "%lf %lf %lf ", &params.t(0), &params.t(1), &params.t(2));
			for (int ii = 0; ii < nJointsSMPL; ii++)
				fscanf(fp, "%lf %lf %lf ", &params.pose(ii, 0), &params.pose(ii, 1), &params.pose(ii, 2));
			gVparams.push_back(params);
		}
		fclose(fp);
	}

	if (gVparams.size() > 0 && timeID < gVparams.size())//keep the last one to avoid interuptingly model disappearing
	{
		SparseMatrix<double, ColMajor> eye3(3, 3); eye3.setIdentity();
		SparseMatrix<double, ColMajor> dVdt = Eigen::kroneckerProduct(VectorXd::Ones(nVertices), eye3);

		reconstruct(mySMPL, gVparams[timeID].coeffs.data(), gVparams[timeID].pose.data(), AllV[0].data());
		Map<VectorXd> V_vec(AllV[0].data(), AllV[0].size());
		V_vec = (V_vec * gVparams[timeID].scale + dVdt * gVparams[timeID].t);

		VectorXd Jsmpl_vec;
		Jsmpl_vec = mySMPL.J_regl_abigl_ * V_vec; //agumented joints for better smothing control

		g_vis.PointPosition3.clear(); g_vis.PointPosition3.reserve(nVertices);
		for (int ii = 0; ii < nVertices; ii++)
			g_vis.PointPosition3.emplace_back(AllV[0](ii, 0), AllV[0](ii, 1), AllV[0](ii, 2));

		if (!IsNumber(UnitScale))
		{
			PointsCentroid[0] = 0.0, PointsCentroid[1] = 0.0, PointsCentroid[2] = 0.0;
			for (int ii = 0; ii < nVertices; ii++)
				PointsCentroid[0] += g_vis.PointPosition3[ii].x, PointsCentroid[1] += g_vis.PointPosition3[ii].y, PointsCentroid[2] += g_vis.PointPosition3[ii].z;
			PointsCentroid[0] /= nVertices, PointsCentroid[1] /= nVertices, PointsCentroid[2] /= nVertices;

			PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
			for (int ii = 0; ii < nVertices; ii++)
			{
				PointsVAR[0] += pow(g_vis.PointPosition3[ii].x - PointsCentroid[0], 2);
				PointsVAR[1] += pow(g_vis.PointPosition3[ii].y - PointsCentroid[1], 2);
				PointsVAR[2] += pow(g_vis.PointPosition3[ii].z - PointsCentroid[2], 2);
			}
			PointsVAR[0] = sqrt(PointsVAR[0] / nVertices), PointsVAR[1] = sqrt(PointsVAR[1] / nVertices), PointsVAR[2] = sqrt(PointsVAR[2] / nVertices);

			UnitScale = sqrt(pow(PointsVAR[0], 2) + pow(PointsVAR[1], 2) + pow(PointsVAR[2], 2)) / 100.0;
			g_coordAxisLength = 20.f*UnitScale, g_fViewDistance = 600 * UnitScale* VIEWING_DISTANCE_MIN;
			g_nearPlane = 1.0*UnitScale, g_farPlane = 30000.f * UnitScale;
			CameraSize = 20.0f*UnitScale, pointSize = 1.0f*UnitScale, normalSize = 5.f*UnitScale, arrowThickness = .1f*UnitScale;
			Discontinuity3DThresh = 20.0*UnitScale;
		}
	}

	return true;
}

int Read3DTrajectory(char *Path, int trialID, int colorCoded)
{
	char Fname[512];
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	g_vis.Track3DLength.clear(), g_vis.Traject3D.clear();

	const int nframes = 10000;
	Point3d P3dTemp[nframes];
	double x, y, z, t, timeStamp[nframes];
	int camID[nframes], sortedCamID[nframes], frameID[nframes], dummy[nframes];
	vector<int>AvailCamID, nVisibles, AddedCamID;
	vector<double> cTrajTime; cTrajTime.reserve(500);

	Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
	for (unsigned int i = 0; i <= 255; i++)
		colorMapSource.at<uchar>(i, 0) = i;
	Mat colorMap; applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);

	if (trialID == 0)
		sprintf(Fname, "Loading OptimizedRaw_Track");
	else if (trialID == 1)
		sprintf(Fname, "Loading SplineResampled_Track");
	else if (trialID == 2)
		sprintf(Fname, "Loading DCTResampled_Track");
	else if (trialID == 3)
		sprintf(Fname, "Loading frameSynced_Track");
	else if (trialID == 4)
		sprintf(Fname, "Loading GTTrack");
	printLOG("%s\n", Fname);

	vector<int> validList;
	sprintf(Fname, "%s/toKeep.txt", Path);  FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int id;  while (fscanf(fp, "%d ", &id) != EOF)
			validList.push_back(id);
		fclose(fp);
	}
	else
	{
		FILE *fp = fopen(Fname, "w+");
		fclose(fp);
	}

	sort(validList.begin(), validList.end());
	std::vector<int>::iterator it = unique(validList.begin(), validList.end());
	validList.resize(std::distance(validList.begin(), it));

	maxTime = 0;
	int npts = 0;
	while (npts < 200000)
	{
		if (trialID == 0)
			sprintf(Fname, "%s/Track3D/OptimizedRaw_Track_%d.txt", Path, npts);
		else if (trialID == 1)
			sprintf(Fname, "%s/Track3D/SplineResampled_Track_%.4d.txt", Path, npts);
		else if (trialID == 2)
			sprintf(Fname, "%s/Track3D/DCTResampled_Track_%.4d.txt", Path, npts);
		else if (trialID == 3)
			sprintf(Fname, "%s/Track3D/frameSynced_Track_%.4d.txt", Path, npts);
		else if (trialID == 4)
			sprintf(Fname, "%s/Track3D/GTTrack_%.4d.txt", Path, npts);
		npts++;
		if (IsFileExist(Fname) == 0)
			continue;

		bool notvalid = false;
		for (auto pid : validList)
		{
			if (pid == npts - 1)
			{
				notvalid = true;
				break;
			}
		}
		if (notvalid)
			continue;
		if (npts % 500 == 0)
			printLOG("%d ..", npts);

		AvailCamID.clear(), cTrajTime.clear();
		int count = 0, alreadyAddedCount, cID, fID;
		bool NotOutSideROI = true;
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%lf %lf %lf %lf %d %d", &x, &y, &z, &t, &cID, &fID) != EOF)
		{
			//Read all the time possible for the current trajectory
			bool found = false;
			for (int ii = 0; ii < cTrajTime.size(); ii++)
			{
				if (abs(cTrajTime[ii] - t) < 0.01)
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				cTrajTime.push_back(t);
				timeStamp[count] = t, camID[count] = cID, frameID[count] = fID;
				P3dTemp[count].x = x, P3dTemp[count].y = y, P3dTemp[count].z = z;
				count++;

				/*if (abs(x - 5.3) > 5 || abs(y - 1.7) > 5 || abs(z - 14.7) > 5)
				{
				NotOutSideROI = false;
				fclose(fp);

				//FILE *fp2 = fopen("C:/temp/toKeep.txt", "a"); fprintf(fp2, "%d\n", npts-1); fclose(fp2);
				//break;
				}*/

				alreadyAddedCount = 0;
				for (int ii = 0; ii < (int)AvailCamID.size(); ii++)
					if (AvailCamID[ii] == cID)
						alreadyAddedCount++;

				if (alreadyAddedCount == 0)
					AvailCamID.push_back(cID);
			}

			//Read all the time possible for all trajectories
			found = false;
			for (int ii = 0; ii < vAll3D_TidCidFid.size() && !found; ii++)
				if (abs(vAll3D_TidCidFid[ii].Ts / Tscale - t) < 0.01)
					found = true;
			if (!found)
				vAll3D_TidCidFid.push_back(TsCidFid(t, cID, fID));
		}
		fclose(fp);

		if (NotOutSideROI)
		{
			for (int ii = 0; ii < count; ii++)
				dummy[ii] = ii;
			Quick_Sort_Double(timeStamp, dummy, 0, count - 1);

			Trajectory3D *track3D = new Trajectory3D[count];
			for (int ii = 0; ii < count; ii++)
			{
				int id = dummy[ii];
				sortedCamID[ii] = camID[id];
				track3D[ii].timeID = timeStamp[ii];
				track3D[ii].WC = P3dTemp[id], track3D[ii].viewID = camID[id], track3D[ii].frameID = frameID[id];
			}

			//look in a window to determine #camera sees the point at that time instance in the window
			if (colorCoded == 1)
			{
				int nCams = (int)AvailCamID.size(), minVis = nCams;
				nVisibles.clear();
				for (int jj = 0; jj < nCams / 2; jj++)
				{
					AddedCamID.clear();
					for (int ii = 0; ii < nCams; ii++)
					{
						for (int kk = 0; kk < nCams; kk++)
						{
							if (sortedCamID[jj + ii] == AvailCamID[kk])
							{
								alreadyAddedCount = 0;
								for (int ll = 0; ll < (int)AddedCamID.size(); ll++)
									if (AvailCamID[kk] == AddedCamID[ll])
										alreadyAddedCount++;
								if (alreadyAddedCount == 0)
									AddedCamID.push_back(AvailCamID[kk]);
								break;
							}
						}
					}
					if ((int)AddedCamID.size() < minVis)
						minVis = (int)AddedCamID.size();
					nVisibles.push_back((int)AddedCamID.size());
				}
				for (int jj = nCams / 2; jj < count - nCams / 2; jj++)
				{
					AddedCamID.clear();
					for (int ii = 0; ii < nCams; ii++)
					{
						for (int kk = 0; kk < nCams; kk++)
						{
							if (sortedCamID[jj - nCams / 2 + ii] == AvailCamID[kk])
							{
								alreadyAddedCount = 0;
								for (int ll = 0; ll < (int)AddedCamID.size(); ll++)
									if (AvailCamID[kk] == AddedCamID[ll])
										alreadyAddedCount++;
								if (alreadyAddedCount == 0)
									AddedCamID.push_back(AvailCamID[kk]);
								break;
							}
						}
					}
					if ((int)AddedCamID.size() < minVis)
						minVis = (int)AddedCamID.size();
					nVisibles.push_back((int)AddedCamID.size());
				}
				for (int jj = count - nCams / 2; jj < count; jj++)
				{
					AddedCamID.clear();
					for (int ii = 0; ii < nCams; ii++)
					{
						for (int kk = 0; kk < nCams; kk++)
						{
							if (sortedCamID[jj - nCams + ii] == AvailCamID[kk])
							{
								alreadyAddedCount = 0;
								for (int ll = 0; ll < (int)AddedCamID.size(); ll++)
									if (AvailCamID[kk] == AddedCamID[ll])
										alreadyAddedCount++;
								if (alreadyAddedCount == 0)
									AddedCamID.push_back(AvailCamID[kk]);
								break;
							}
						}
					}
					if ((int)AddedCamID.size() < minVis)
						minVis = (int)AddedCamID.size();
					nVisibles.push_back((int)AddedCamID.size());
				}

				int range = nCams - minVis, nvis;
				for (int ii = 0; ii < count; ii++)
				{
					nvis = nVisibles[ii];
					int colorIdx = (int)(1.0*(nvis - minVis) / (0.01 + range)* 255.0 + 0.5);
					Point3f PointColor;
					PointColor.z = colorMap.at<Vec3b>(colorIdx, 0)[0] / 255.0; //blue
					PointColor.y = colorMap.at<Vec3b>(colorIdx, 0)[1] / 255.0; //green
					PointColor.x = colorMap.at<Vec3b>(colorIdx, 0)[2] / 255.0;	//red
					track3D[ii].rgb = PointColor;
				}
			}
			else
			{
				Point3f PointColor(0.2f, 0.7f, 1.0f); //light blue
													  //Point3f PointColor(1.0f, 0.5f, 0.2f); //light ora
													  //Point3f PointColor(0.0f, 0.f, 1.f); //blue
				for (int ii = 0; ii < count; ii++)
					track3D[ii].rgb = PointColor;
			}

			TrajRealId.push_back(npts - 1);
			g_vis.Track3DLength.push_back(count);
			g_vis.Traject3D.push_back(track3D);

			maxTime = max(maxTime, count);
		}
	}

	SortedTimeInstance = new int[vAll3D_TidCidFid.size()];
	double *ts = new double[vAll3D_TidCidFid.size()];
	for (int ii = 0; ii < vAll3D_TidCidFid.size(); ii++)
		SortedTimeInstance[ii] = ii, ts[ii] = vAll3D_TidCidFid[ii].Ts;
	Quick_Sort_Double(ts, SortedTimeInstance, 0, vAll3D_TidCidFid.size() - 1);
	delete[]ts;

	/*vector<int>validList;
	FILE *fp = fopen("C:/temp/toDel.txt", "r");
	while (fscanf(fp, "%d ", &id) != EOF)
	validList.push_back(id);
	fclose(fp);*/
	/*fp = fopen("C:/temp/realID.txt", "w");
	for (int ii = 0; ii < validList.size(); ii++)
	fprintf(fp, "%d\n", TrajRealId[validList[ii]]);
	fclose(fp);*/

	if (colorCoded == 0)
	{
		Point3f PointColor;
		Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
		for (unsigned int i = 0; i <= 255; i++)
			colorMapSource.at<uchar>(i, 0) = i;
		Mat colorMap; applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);

		for (int jj = 0; jj < (int)g_vis.Traject3D.size(); jj++)
		{
			for (int ii = 0; ii < g_vis.Track3DLength[jj]; ii++)
			{
				double ctime = 1.0*g_vis.Traject3D[jj][ii].timeID;
				double colorIdx = (ctime - vAll3D_TidCidFid[SortedTimeInstance[0]].Ts) / vAll3D_TidCidFid[SortedTimeInstance[vAll3D_TidCidFid.size()]].Ts * 255.0;
				colorIdx = min(255.0, colorIdx);
				PointColor.z = colorMap.at<Vec3b>(colorIdx, 0)[0] / 255.0; //blue
				PointColor.y = colorMap.at<Vec3b>(colorIdx, 0)[1] / 255.0; //green
				PointColor.x = colorMap.at<Vec3b>(colorIdx, 0)[2] / 255.0;	//red
				g_vis.Traject3D[jj][ii].rgb = PointColor;
			}
		}
	}

	nTraject = max(nTraject, npts);

	return 0;
}

int ReadCurrentPosesGL(char *Path, int startF, int stopF, Point3d *TimeInfo, int ShutterType)
{
	char Fname[512];

	CameraData temp;
	g_vis.glCameraPoseInfo = new vector<CameraData>[nCams];
	for (int cid = 0; cid < nCams; cid++)
		for (int fid = 0; fid <= stopF + (int)(abs(TimeInfo[cid].y)) + 1; fid++)
			g_vis.glCameraPoseInfo[cid].push_back(temp);

	int frameID;
	double rt[6], R[9], T[3], Rgl[16], Cgl[3], dummy[6];
	for (int cid = 0; cid < nCams; cid++)
	{
		int lstartF = -1, lstopF = -1;
		if (ShutterType == GLOBAL_SHUTTER || ShutterType == ROLLING_SHUTTER)
		{
			sprintf(Fname, "%s/avHCamPose_%.4d.txt", Path, cid);
			if (IsFileExist(Fname) == 0)
			{
				//printLOG("Cannot find %s...\n", Fname);
				sprintf(Fname, "%s/avCamPose_%.4d.txt", Path, cid);
				if (IsFileExist(Fname) == 0)
				{
					sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, cid);
					if (IsFileExist(Fname) == 0)
					{
						//printLOG("Cannot find %s...\n", Fname);
						sprintf(Fname, "%s/CamPose_%.4d.txt", Path, cid);
						if (IsFileExist(Fname) == 0)
						{
							//printLOG("Cannot find %s...\n", Fname);
							continue;
						}
					}
				}
			}
		}
		else
		{
			sprintf(Fname, "%s/CamPose_Spline_%.4d.txt", Path, cid);
			if (IsFileExist(Fname) == 0)
			{
				printLOG("Cannot find %s...\n", Fname);
				continue;
			}
		}
		int lastFrameID = -1;
		printLOG("Loaded %s\n", Fname);
		FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &frameID) != EOF)
		{
			for (int jj = 0; jj < 6; jj++)
				fscanf(fp, "%lf ", &rt[jj]);
			if (ShutterType == ROLLING_SHUTTER)
				for (int jj = 0; jj < 6; jj++)
					fscanf(fp, "%lf ", &dummy[jj]);

			if (frameID + (int)(TimeInfo[cid].y + 0.5) < startF || frameID + (int)(TimeInfo[cid].y + 0.5) > stopF)
				continue;

			if (lstartF == -1)
				lstartF = frameID;
			lstopF = max(lstopF, frameID);

			g_vis.glCameraPoseInfo[cid][frameID].valid = 1;
			g_vis.glCameraPoseInfo[cid][frameID].frameID = frameID;
			GetRTFromrt(rt, R, T);
			GetRCGL(R, T, Rgl, Cgl);

			for (int jj = 0; jj < 16; jj++)
				g_vis.glCameraPoseInfo[cid][frameID].Rgl[jj] = Rgl[jj];
			for (int jj = 0; jj < 3; jj++)
				g_vis.glCameraPoseInfo[cid][frameID].camCenter[jj] = Cgl[jj];
		}
		fclose(fp);

		vector<double> vCenterDif;
		for (int fid = lstartF; fid <= lstopF; fid++)
			if (g_vis.glCameraPoseInfo[cid][fid].valid)
				vCenterDif.push_back(norm(Point3d(g_vis.glCameraPoseInfo[cid][fid].camCenter[0] - g_vis.glCameraPoseInfo[cid][fid + 1].camCenter[0], g_vis.glCameraPoseInfo[cid][fid].camCenter[1] - g_vis.glCameraPoseInfo[cid][fid + 1].camCenter[1], g_vis.glCameraPoseInfo[cid][fid].camCenter[2] - g_vis.glCameraPoseInfo[cid][fid + 1].camCenter[2])));
		if (vCenterDif.size() < 2)
			continue;

		sort(vCenterDif.begin(), vCenterDif.end());
		double thresh = vCenterDif[vCenterDif.size() * 98 / 100];

		int firstTime = 1, lastValid = lstartF, ndel = 0;
		for (int fid = lstartF; fid <= lstopF; fid++)
		{
			if (firstTime == 1)
			{
				if (!g_vis.glCameraPoseInfo[cid][lastValid].valid)
				{
					lastValid++;
					continue;
				}
				else
					firstTime = 0;
			}
			if (g_vis.glCameraPoseInfo[cid][lastValid].valid &&g_vis.glCameraPoseInfo[cid][fid + 1].valid)
			{
				double dif1 = norm(Point3d(g_vis.glCameraPoseInfo[cid][lastValid].camCenter[0] - g_vis.glCameraPoseInfo[cid][fid + 1].camCenter[0], g_vis.glCameraPoseInfo[cid][lastValid].camCenter[1] - g_vis.glCameraPoseInfo[cid][fid + 1].camCenter[1], g_vis.glCameraPoseInfo[cid][lastValid].camCenter[2] - g_vis.glCameraPoseInfo[cid][fid + 1].camCenter[2]));
				if (dif1 <= 2.0*thresh*(fid + 1 - lastValid))
					lastValid = fid + 1;
				else
				{
					g_vis.glCameraPoseInfo[cid][fid + 1].valid = 0, ndel++;
					printLOG("%d..", fid + 1);
				}
			}
		}
		for (int fid = max(startF - (int)(TimeInfo[cid].y + 0.5) - 1, 0); fid < stopF - (int)(TimeInfo[cid].y + 0.5) + 1; fid++)
		{
			if (!g_vis.glCameraPoseInfo[cid][fid].valid)
				continue;

			frameID = g_vis.glCameraPoseInfo[cid][fid].frameID;
			//double ts = round(Tscale*(1.0*frameID + TimeInfo[cid].y) / TimeInfo[cid].x);

			//double ts = round(Tscale*(TimeInfo[cid].y / TimeInfo[refCid].x + 1.0*frameID / TimeInfo[cid].x));
			double ts = round(Tscale*(CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*frameID / CamTimeInfo[cid].x)*CamTimeInfo[refCid].x);
			vAll3D_TidCidFid.push_back(TsCidFid(ts, cid, frameID));
		}
		if (ndel > 0)
			printLOG("\n");
	}

	nNonCorpusCams = 0;
	for (int cid = 0; cid < nCams; cid++)
		nNonCorpusCams += (int)g_vis.glCameraPoseInfo[cid].size();

	if (g_vis.CorpusPointPosition.size() == 0)
	{
		int npts = 0;
		PointsCentroid[0] = 0.0, PointsCentroid[1] = 0.0, PointsCentroid[2] = 0.0;
		for (int cid = 0; cid < nCams; cid++)
		{
			for (int fid = 0; fid <= stopF + (int)(abs(TimeInfo[cid].y)) + 1; fid++)
			{
				if (g_vis.glCameraPoseInfo[cid][fid].valid)
				{
					for (int ii = 0; ii < 3; ii++)
						PointsCentroid[ii] += g_vis.glCameraPoseInfo[cid][fid].camCenter[ii];
					npts++;
				}
			}
		}
		for (int ii = 0; ii < 3; ii++)
			PointsCentroid[ii] /= npts;

		PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
		for (int cid = 0; cid < nCams; cid++)
		{
			for (int fid = 0; fid <= stopF + (int)(abs(TimeInfo[cid].y)) + 1; fid++)
			{
				if (g_vis.glCameraPoseInfo[cid][fid].valid)
				{
					for (int ii = 0; ii < 3; ii++)
						PointsVAR[ii] += pow(g_vis.glCameraPoseInfo[cid][fid].camCenter[ii] - PointsCentroid[ii], 2);
				}
			}
		}
		for (int ii = 0; ii < 3; ii++)
			PointsVAR[ii] = sqrt(PointsVAR[ii] / npts);
	}

	return 0;
}
int ReadCurrentPosesGL2(char *Path, std::vector<char*> SelectedCamNames, int SeqId, std::vector<int> &CamIdsPerSeq, int startF, int stopF, Point3d *TimeInfo)
{
	char Fname[512];
	int nCams = SelectedCamNames.size()*CamIdsPerSeq.size();

	int selected, maxFrameOffset = -9999; double fps;
	sprintf(Fname, "%s/InitSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int temp;
		while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
			maxFrameOffset = max(maxFrameOffset, abs((int)temp));
		fclose(fp);
	}
	else
		printLOG("Cannot load time stamp info. Assume no frame offsets!");

	CameraData temp;
	g_vis.glCameraPoseInfo = new vector<CameraData>[nCams];
	for (int cid = 0; cid < nCams; cid++)
	{
		for (int fid = 0; fid <= stopF + maxFrameOffset + 1; fid++)
			g_vis.glCameraPoseInfo[cid].push_back(temp);
	}

	int frameID, idummy;
	char sdummy[512];
	double q[4], C[3], rt[6], R[9], invR[9], T[3], Rgl[16], Cgl[3], dummy[6];
	for (int ii = 0; ii < (int)SelectedCamNames.size(); ii++)
	{
		for (int jj = 0; jj < (int)CamIdsPerSeq.size(); jj++)
		{
			int cid = ii * CamIdsPerSeq.size() + jj;
			int lstartF = -1, lstopF = -1;

			int lastFrameID = -1;
			sprintf(Fname, "%s/%s/%s_general_%d_%d.csv", Path, SelectedCamNames[ii], SelectedCamNames[ii], SeqId, CamIdsPerSeq[jj]);
			if (IsFileExist)
				printLOG("Loaded %s\n", Fname);
			fp = fopen(Fname, "r");
			fscanf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s ", sdummy, sdummy, sdummy, sdummy, sdummy, sdummy, sdummy, sdummy, sdummy);
			while (fscanf(fp, "%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf ", &frameID, &idummy, &q[1], &q[2], &q[3], &q[0], &C[0], &C[1], &C[2]) != EOF)
			{
				frameID = frameID - 1;//Kiran first frame is indexed 1 while the extracted frames is indexed 0
				if (frameID < startF - maxFrameOffset || frameID > stopF + maxFrameOffset)
					continue;

				if (lstartF == -1)
					lstartF = frameID;
				lstopF = max(lstopF, frameID);

				g_vis.glCameraPoseInfo[cid][frameID].valid = 1;
				g_vis.glCameraPoseInfo[cid][frameID].frameID = frameID;

				Quaternion2Rotation(q, invR);
				mat_transpose(invR, R, 3, 3);
				GetTfromC(R, C, T);
				GetRCGL(R, T, Rgl, Cgl);

				for (int jj = 0; jj < 16; jj++)
					g_vis.glCameraPoseInfo[cid][frameID].Rgl[jj] = Rgl[jj];
				for (int jj = 0; jj < 3; jj++)
					g_vis.glCameraPoseInfo[cid][frameID].camCenter[jj] = Cgl[jj];
			}
			fclose(fp);

			for (int fid = max(startF - (int)(TimeInfo[cid].y + 0.5) - 1, 0); fid < stopF - (int)(TimeInfo[cid].y + 0.5) + 1; fid++)
			{
				if (!g_vis.glCameraPoseInfo[cid][fid].valid)
					continue;

				frameID = g_vis.glCameraPoseInfo[cid][fid].frameID;
				//double ts = round(Tscale*(1.0*frameID + TimeInfo[cid].y) / TimeInfo[cid].x);

				//double ts = round(Tscale*(TimeInfo[cid].y / TimeInfo[refCid].x + 1.0*frameID / TimeInfo[cid].x));
				double ts = round(Tscale*(CamTimeInfo[cid].y / CamTimeInfo[refCid].x + 1.0*frameID / CamTimeInfo[cid].x)*CamTimeInfo[refCid].x);
				vAll3D_TidCidFid.push_back(TsCidFid(ts, cid, frameID));
			}
		}
	}

	nNonCorpusCams = 0;
	for (int cid = 0; cid < nCams; cid++)
		nNonCorpusCams += (int)g_vis.glCameraPoseInfo[cid].size();

	if (g_vis.CorpusPointPosition.size() == 0)
	{
		int npts = 0;
		PointsCentroid[0] = 0.0, PointsCentroid[1] = 0.0, PointsCentroid[2] = 0.0;
		for (int cid = 0; cid < nCams; cid++)
		{
			for (int fid = 0; fid <= stopF + (int)(abs(TimeInfo[cid].y)) + 1; fid++)
			{
				if (g_vis.glCameraPoseInfo[cid][fid].valid)
				{
					for (int ii = 0; ii < 3; ii++)
						PointsCentroid[ii] += g_vis.glCameraPoseInfo[cid][fid].camCenter[ii];
					npts++;
				}
			}
		}
		for (int ii = 0; ii < 3; ii++)
			PointsCentroid[ii] /= npts;

		PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
		for (int cid = 0; cid < nCams; cid++)
		{
			for (int fid = 0; fid <= stopF + (int)(abs(TimeInfo[cid].y)) + 1; fid++)
			{
				if (g_vis.glCameraPoseInfo[cid][fid].valid)
				{
					for (int ii = 0; ii < 3; ii++)
						PointsVAR[ii] += pow(g_vis.glCameraPoseInfo[cid][fid].camCenter[ii] - PointsCentroid[ii], 2);
				}
			}
		}
		for (int ii = 0; ii < 3; ii++)
			PointsVAR[ii] = sqrt(PointsVAR[ii] / npts);
	}

	return 0;
}
int screenShot(char *Fname, int width, int height, bool color)
{
	int ii, jj;

	unsigned char *data = new unsigned char[width*height * 4];
	//IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	Mat cvImg(height, width, CV_8UC3, Scalar(0, 0, 0));

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

	for (jj = 0; jj < height; jj++)
	{
		uchar *ptr = cvImg.ptr<uchar>(jj);
		for (ii = 0; ii < width; ii++)
		{
			ptr[3 * ii] = data[3 * ii + 3 * (height - 1 - jj)*width + 2],
				ptr[3 * ii + 1] = data[3 * ii + 3 * (height - 1 - jj)*width + 1],
				ptr[3 * ii + 2] = data[3 * ii + 3 * (height - 1 - jj)*width];
			//cvImg->imageData[3 * ii + 3 * jj*width] = data[3 * ii + 3 * (height - 1 - jj)*width + 2],
			//	cvImg->imageData[3 * ii + 3 * jj*width + 1] = data[3 * ii + 3 * (height - 1 - jj)*width + 1],
			//	cvImg->imageData[3 * ii + 3 * jj*width + 2] = data[3 * ii + 3 * (height - 1 - jj)*width];
		}
	}
	//cvSaveImage(Fname, cvImg);
	imwrite(Fname, cvImg);

	delete[]data;
	return 0;
}

Corpus CorpusInfo;
int oProcessedCam, ProcessedCam;
int ReadPerFrameSfMResults(char *Path, int fid)
{
	char Fname[512];
	double fdummy, rt[6], R[9], T[3];  int nCams, LensModel, dummy;

	CameraData temp;
	sprintf(Fname, "%s/JBC/%.4d/BA.txt", Path, fid, fid); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		fscanf(fp, "%d %d ", &nCams, &dummy);
		//printLOG("# good projections: %d\n", dummy);
		g_vis.glCorpusCameraInfo.clear();
		for (int cid = 0; cid < nCams; cid++)
		{
			fscanf(fp, "%d %d %d %d %d ", &temp.viewID, &LensModel, &dummy, &dummy, &dummy);
			for (int ii = 0; ii < 5; ii++)
				fscanf(fp, "%lf ", &fdummy);
			if (LensModel == RADIAL_TANGENTIAL_PRISM)
				for (int ii = 0; ii < 7; ii++)
					fscanf(fp, "%lf ", &fdummy);
			else
				for (int ii = 0; ii < 3; ii++)
					fscanf(fp, "%lf ", &fdummy);
			for (int ii = 0; ii < 6; ii++)
				fscanf(fp, "%lf ", &rt[ii]);

			GetRTFromrt(rt, R, T);
			GetRCGL(R, T, temp.Rgl, temp.camCenter);
			g_vis.glCorpusCameraInfo.push_back(temp);
		}
		fclose(fp);
	}
	else
		return 1;

	g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(360);

	int npts = 0; Point3f t3d;
	sprintf(Fname, "%s/JBC/%.4d/3dGLt.txt", Path, fid, fid); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		//printLOG("Loaded %s\n", Fname);
		g_vis.CorpusPointPosition.clear();
		if (UnitScale == 1)
			PointsCentroid[0] = 0, PointsCentroid[1] = 0, PointsCentroid[2] = 0;
		while (fscanf(fp, "%d %f %f %f ", &dummy, &t3d.x, &t3d.y, &t3d.z) != EOF)
		{
			if (UnitScale == 1 && abs(t3d.x) >= 0.0000001)
				PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z, npts++;

			g_vis.CorpusPointPosition.push_back(t3d);
		}
		fclose(fp);
	}
	else
		return 1;
	printLOG("%d..", fid);

	if (UnitScale == 1)
	{
		PointsCentroid[0] /= npts, PointsCentroid[1] /= npts, PointsCentroid[2] /= npts;
		PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
		for (int ii = 0; ii < g_vis.CorpusPointPosition.size(); ii++)
		{
			if (abs(g_vis.CorpusPointPosition[ii].x) >= LIMIT3D)
			{
				PointsVAR[0] += pow(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], 2);
				PointsVAR[1] += pow(g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], 2);
				PointsVAR[2] += pow(g_vis.CorpusPointPosition[ii].z - PointsCentroid[2], 2);
			}
		}
		PointsVAR[0] = sqrt(PointsVAR[0] / npts), PointsVAR[1] = sqrt(PointsVAR[1] / npts), PointsVAR[2] = sqrt(PointsVAR[2] / npts);

		UnitScale = sqrt(pow(PointsVAR[0], 2) + pow(PointsVAR[1], 2) + pow(PointsVAR[2], 2)) / 100.0;
		g_coordAxisLength = 20.f*UnitScale, g_fViewDistance = 600 * UnitScale* VIEWING_DISTANCE_MIN;
		g_nearPlane = 1.0*UnitScale, g_farPlane = 30000.f * UnitScale;
		CameraSize = 20.0f*UnitScale, pointSize = 1.0f*UnitScale, normalSize = 5.f*UnitScale, arrowThickness = .1f*UnitScale;
	}

	//printLOG("%.4e %.4e %.4e\n", PointsCentroid[0], PointsCentroid[1], PointsCentroid[2]);

	return 0;
}
bool CheckForUpdate()
{
	ProcessedCam = 0;
	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
		if (CorpusInfo.camera[ii].processed)
			ProcessedCam++;

	if (timeID == 0)
		PointsCentroid[0] = 0, PointsCentroid[1] = 0, PointsCentroid[2] = 0;

	if (ProcessedCam > oProcessedCam)
	{
		oProcessedCam = ProcessedCam;

		g_vis.glCorpusCameraInfo.clear(), g_vis.glCorpusCameraInfo.reserve(100);
		g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(10e5);
		g_vis.CorpusPointColor.clear(); g_vis.CorpusPointColor.reserve(10e5);

		CameraData temp;
		for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
		{
			if (CorpusInfo.camera[ii].processed)
			{
				getRfromr(CorpusInfo.camera[ii].rt, CorpusInfo.camera[ii].R);
				GetCfromT(CorpusInfo.camera[ii].R, CorpusInfo.camera[ii].rt + 3, CorpusInfo.camera[ii].camCenter);
				for (int jj = 0; jj < 3; jj++)
					for (int kk = 0; kk < 3; kk++)
						temp.Rgl[kk + jj * 4] = CorpusInfo.camera[ii].R[kk + jj * 3];

				for (int jj = 0; jj < 3; jj++)
					temp.camCenter[jj] = CorpusInfo.camera[ii].camCenter[jj];

				temp.viewID = ii;
				g_vis.glCorpusCameraInfo.push_back(temp);
			}
		}

		int nValidPoints = 0;
		for (int ii = 0; ii < CorpusInfo.n3dPoints; ii++)
		{
			Point3f t3d(CorpusInfo.xyz[ii].x, CorpusInfo.xyz[ii].y, CorpusInfo.xyz[ii].z);
			if (abs(t3d.x) + abs(t3d.y) + abs(t3d.z) == 0)
			{
				g_vis.CorpusPointPosition.push_back(t3d);
				if (hasColor)
				{
					Point3f fColor(0, 0, 0);
					g_vis.CorpusPointColor.push_back(fColor);
				}
				continue;
			}

			nValidPoints++;
			if (timeID == 0)
				PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
			g_vis.CorpusPointPosition.push_back(t3d);
			if (hasColor)
			{
				Point3f fColor(1.0*CorpusInfo.rgb[ii].x / 255, 1.0*CorpusInfo.rgb[ii].y / 255, 1.0*CorpusInfo.rgb[ii].z / 255);
				g_vis.CorpusPointColor.push_back(fColor);
			}
		}

		if (nValidPoints > 0)
		{
			PointsCentroid[0] /= nValidPoints;
			PointsCentroid[1] /= nValidPoints;
			PointsCentroid[2] /= nValidPoints;

			PointsVAR[0] = 0.0, PointsVAR[1] = 0.0, PointsVAR[2] = 0.0;
			for (int ii = 0; ii < nValidPoints; ii++)
			{
				PointsVAR[0] += pow(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], 2);
				PointsVAR[1] += pow(g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], 2);
				PointsVAR[2] += pow(g_vis.CorpusPointPosition[ii].z - PointsCentroid[2], 2);
			}
			PointsVAR[0] = sqrt(PointsVAR[0] / nValidPoints);
			PointsVAR[1] = sqrt(PointsVAR[1] / nValidPoints);
			PointsVAR[2] = sqrt(PointsVAR[2] / nValidPoints);
		}
		else
			PointsCentroid[0] = PointsCentroid[1] = PointsCentroid[2] = 0;

		UnitScale = sqrt(pow(PointsVAR[0], 2) + pow(PointsVAR[1], 2) + pow(PointsVAR[2], 2)) / 250.0;
		g_coordAxisLength = 20.f*UnitScale;
		g_fViewDistance = 1000 * UnitScale* VIEWING_DISTANCE_MIN;
		g_nearPlane = 0.01*UnitScale, g_farPlane = 3000000.f * UnitScale;
		CameraSize = 20.0f*UnitScale, pointSize = 2.0f*UnitScale;

		return true;
	}

	return false;
}
void SimpleRender()
{
	GLfloat red[3] = { 1, 0, 0 };
	GLfloat white[3] = { 1, 1, 1 };
	GLfloat green[3] = { 0, 1, 0 };
	GLfloat black[3] = { 0, 0, 0 };

	//RenderImage();
	Render_COCO_3D_Skeleton(g_vis.CorpusPointPosition, green, PointFormat);

	//draw Corpus camera 
	if (drawCorpusCameras)
	{
		for (int ii = 0; ii < g_vis.glCorpusCameraInfo.size(); ii++)
		{
			float centerPt[3] = { g_vis.glCorpusCameraInfo[ii].camCenter[0], g_vis.glCorpusCameraInfo[ii].camCenter[1], g_vis.glCorpusCameraInfo[ii].camCenter[2] };
			GLfloat R[16] = { g_vis.glCorpusCameraInfo[ii].Rgl[0], g_vis.glCorpusCameraInfo[ii].Rgl[1], g_vis.glCorpusCameraInfo[ii].Rgl[2], g_vis.glCorpusCameraInfo[ii].Rgl[3],
				g_vis.glCorpusCameraInfo[ii].Rgl[4], g_vis.glCorpusCameraInfo[ii].Rgl[5], g_vis.glCorpusCameraInfo[ii].Rgl[6], g_vis.glCorpusCameraInfo[ii].Rgl[7],
				g_vis.glCorpusCameraInfo[ii].Rgl[8], g_vis.glCorpusCameraInfo[ii].Rgl[9], g_vis.glCorpusCameraInfo[ii].Rgl[10], g_vis.glCorpusCameraInfo[ii].Rgl[11],
				g_vis.glCorpusCameraInfo[ii].Rgl[12], g_vis.glCorpusCameraInfo[ii].Rgl[13], g_vis.glCorpusCameraInfo[ii].Rgl[14], g_vis.glCorpusCameraInfo[ii].Rgl[15] };

			glLoadName(ii);//for picking purpose


			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(R);

			DrawCamera();
			char name[512];
			sprintf(name, "%d", g_vis.glCorpusCameraInfo[ii].viewID);
			//DrawStr(Point3f(0, 1, 0), name);

			glPopMatrix();
		}
	}


}
void SimpleDisplay(void)
{
	CheckForUpdate();
	if (timeID != otimeID)
	{
		ReadPerFrameSfMResults(Path, timeID);
		for (int cid = 0; cid < nCams; cid++)
		{
			char Fname[512]; sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid, timeID);
			GenerateBackgroundTexture(Fname, cid);
		}
		otimeID = timeID;
	}

	if (changeBackgroundColor)
		glClearColor(1.0, 1.0, 1.0, 0.0);
	else
		glClearColor(0.0, 0.0, 0.0, 0.0);

	// Clear frame buffer and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	glTranslatef(0, 0, -g_fViewDistance);
	glRotated(-g_mouseYRotate, 1, 0, 0);
	glRotated(-g_mouseXRotate, 0, 1, 0);

	SimpleRender();
	if (showAxis)
		Draw_Axes(), showAxis = false;

	glutSwapBuffers();
}
void SimpleIdleFunction(void)
{
	CheckForUpdate();
	if (timeID != otimeID)
	{
		ReadPerFrameSfMResults(Path, timeID);
		otimeID = timeID;
	}

	return;
}
int visualizePerFrameSfM(char *inPath, int refFrame, int startF, int stopF, int nCameras, int nMaxPeople)
{
	increT = 10;
	Path = inPath;
	nCams = nCameras;

	char *myargv[1];
	int myargc = 1;
	myargv[0] = "SfM";
	glutInit(&myargc, myargv);

	glutInitWindowSize(1600, 900);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("SfM");

	glShadeModel(GL_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);

	CorpusInfo.nCameras = nCams;
	CorpusInfo.camera = new CameraData[nCams];
	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
		CorpusInfo.camera[ii].processed = false;
	CorpusInfo.n3dPoints = nCoCoJoints * 4;

	refFid = refFrame;
	timeID = startF, otimeID = timeID, UnitScale = 1.0;
	ReadPerFrameSfMResults(Path, timeID);

	g_backgroundTexture.resize(nCams);

	glutDisplayFunc(SimpleDisplay);
	glutKeyboardFunc(Keyboard);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutSpecialFunc(SpecialInput);
	glutIdleFunc(SimpleIdleFunction);
	glutMotionFunc(MouseMotion);

	glutMainLoop();

	return 0;
}
int LiveSfM(char *Path, int startF, int stopF, int increT, int nCams, int nMaxPeople, int verbose)
{
	newLog = true;

	char *myargv[1];
	int myargc = 1;
	myargv[0] = "SfM";
	glutInit(&myargc, myargv);

	glutInitWindowSize(1600, 900);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("SfM");

	glShadeModel(GL_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);

	CorpusInfo.nCameras = nCams;
	CorpusInfo.camera = new CameraData[nCams];
	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
		CorpusInfo.camera[ii].processed = false;
	CorpusInfo.n3dPoints = nCoCoJoints * 4;

	UnitScale = 0.02;
	g_coordAxisLength = 20.f*UnitScale;
	g_fViewDistance = 1000 * UnitScale* VIEWING_DISTANCE_MIN;
	g_nearPlane = 0.01*UnitScale, g_farPlane = 3000000.f * UnitScale;
	CameraSize = 20.0f*UnitScale, pointSize = 5.0f*UnitScale;

	glutDisplayFunc(SimpleDisplay);
	glutKeyboardFunc(Keyboard);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutSpecialFunc(SpecialInput);
	glutIdleFunc(SimpleIdleFunction);
	glutMotionFunc(MouseMotion);

#pragma omp parallel
	{
#pragma omp sections nowait
		{
#pragma omp section
			{
				mySleep(1000);
				printLOG("Run GLUT\n");
				glutMainLoop();
			}
#pragma omp section
			{
				for (int fid = startF; fid <= stopF; fid += increT)
				{
					mySleep(1000);
					printLOG("\n\nWorking on %d:\n\n", fid);
					oProcessedCam = 2;
					for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
						CorpusInfo.camera[ii].processed = false;
					CorpusInfo.xyz.clear();
					for (int pid = 0; pid < CorpusInfo.n3dPoints; pid++)
						CorpusInfo.viewIdAll3D.clear(), CorpusInfo.uvAll3D.clear(), CorpusInfo.scaleAll3D.clear();

					//SimpleBodyPoseSfM(Path, fid, CorpusInfo, nMaxPeople, verbose);

					double startWait = omp_get_wtime(), elapseTime = 0;
					while (elapseTime < 3)
						elapseTime = omp_get_wtime() - startWait;
				}
			}
		}
	}

	return 0;
}