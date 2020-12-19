#include "DataIO.h"
#include "GeneralUlti.h"
#include <ctime>

using namespace std;
using namespace cv;

void StringAppendV(std::string* dst, const char* format, va_list ap) {
	// First try with a small fixed size buffer.
	static const int kFixedBufferSize = 1024;
	char fixed_buffer[kFixedBufferSize];

	// It is possible for methods that use a va_list to invalidate the data in it upon use.  The fix is to make a copy of the structure before using it and use that copy instead.
	va_list backup_ap;
	va_copy(backup_ap, ap);
	int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
	va_end(backup_ap);

	if (result < kFixedBufferSize)
	{
		if (result >= 0)  // Normal case - everything fits.
		{
			dst->append(fixed_buffer, result);
			return;
		}

#ifdef _MSC_VER
		// Error or MSVC running out of space.  MSVC 8.0 and higher can be asked about space needed with the special idiom below:
		va_copy(backup_ap, ap);
		result = vsnprintf(nullptr, 0, format, backup_ap);
		va_end(backup_ap);
#endif

		if (result < 0) // Just an error.
			return;
	}

	// Increase the buffer size to the size requested by vsnprintf,
	// plus one for the closing \0.
	const int variable_buffer_size = result + 1;
	std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

	// Restore the va_list before we use it again.
	va_copy(backup_ap, ap);
	result = vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
	va_end(backup_ap);

	if (result >= 0 && result < variable_buffer_size)
		dst->append(variable_buffer.get(), result);
}
void printLOG(const char* format, ...)
{
	va_list ap;
	va_start(ap, format);
	std::string str;
	StringAppendV(&str, format, ap);
	va_end(ap);

	std::cout << str;

	if (newLog)
	{
		newLog = false;
		time_t t = time(NULL);
		tm* timePtr = localtime(&t);
		makeDir("Log");
		sprintf(LOG_FILE_NAME, "Log/[%.4d_%.2d_%.2d][%.2d_%.2d_%.2d]_%.6d.txt",
			timePtr->tm_year + 1900, timePtr->tm_mon + 1, timePtr->tm_mday,
			timePtr->tm_hour, timePtr->tm_min, timePtr->tm_sec,
			rand()); //in case you run many command at once.
	}

	std::ofstream log_file(LOG_FILE_NAME, std::ios_base::out | std::ios_base::app);
	log_file << str;
	log_file.close();
}

bool IsNotWhiteSpace(const int character)
{
	return character != ' ' && character != '\n' && character != '\r' && character != '\t';
}
void StringLeftTrim(std::string* str)
{
	str->erase(str->begin(), std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
}
void StringRightTrim(std::string* str)
{
	str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(), str->end());
}
void StringTrim(std::string* str)
{
	StringLeftTrim(str);
	StringRightTrim(str);
}
void makeDir(char *Fname)
{
#ifdef _WINDOWS
	_mkdir(Fname);
#else
	mkdir(Fname, 0755);
#endif
	return;
}
bool MyCopyFile(const char *SRC, const char* DEST)
{
	std::ifstream src(SRC, std::ios::binary);
	if (!src)
	{
		printLOG("Cannot find %s\n", SRC);
		return false;
	}
	std::ofstream dest(DEST, std::ios::binary);
	if (!dest)
		return false;

	dest << src.rdbuf();
	return src && dest;
}
int IsFileExist(const char *Fname, bool silient)
{
	std::ifstream test(Fname);
	if (test.is_open())
	{
		test.close();
		return 1;
	}
	else
	{
		if (!silient)
			printLOG("Cannot load %s\n", Fname);
		return 0;
	}
}

#ifdef _WINDOWS
vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do // read all (real) files in current folder, delete '!' read other 2 default folder . and ..
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}
#endif

enum
{
	SIFT_NAME = ('S' + ('I' << 8) + ('F' << 16) + ('T' << 24)),
	MSER_NAME = ('M' + ('S' << 8) + ('E' << 16) + ('R' << 24)),
	RECT_NAME = ('R' + ('E' << 8) + ('C' << 16) + ('T' << 24)),
	SIFT_VERSION_4 = ('V' + ('4' << 8) + ('.' << 16) + ('0' << 24)),
	SIFT_EOF = (0xff + ('E' << 8) + ('O' << 16) + ('F' << 24)),
};

int readOpenPoseJson(char *Path, int cid, int fid, vector<Point2f> &vUV, vector<float> &vConf)
{
	char Fname[512], text[512];

	vUV.clear(), vConf.clear();

	bool nofile = true;
	sprintf(Fname, "%s/%d/sep-json/%d.json", Path, cid, fid);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/MP/%d/%.4d.json", Path, cid, fid);
		if (IsFileExist(Fname) == 0)
			nofile = true;
		else
			nofile = false;
	}
	else
		nofile = false;
	if (!nofile)
	{
		FileStorage fs(Fname, 0);
		FileNode root = fs["bodies"];
		for (int i = 0; i < root.size(); i++)
		{
			FileNode val1 = root[i]["joints"];
			for (int j = 0; j < val1.size(); j += 3)
			{
				vUV.push_back(Point2f(val1[j].real(), val1[j + 1].real()));
				vConf.push_back(val1[j + 2].real());
			}
		}
		return 1;
	}

	sprintf(Fname, "%s/MP/%d/x_%.12d_keypoints.json", Path, cid, fid);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/MP/%d/%.4d_keypoints.json", Path, cid, fid);
		if (IsFileExist(Fname) == 0)
			nofile = true;
		else
			nofile = false;
	}
	else
		nofile = false;
	if (!nofile)
	{
		FileStorage fs(Fname, 0);
		FileNode root = fs["people"];
		for (int i = 0; i < root.size(); i++)
		{
			FileNode val1 = root[i]["pose_keypoints_2d"];
			for (int j = 0; j < val1.size(); j += 3)
			{
				vUV.push_back(Point2f(val1[j].real(), val1[j + 1].real()));
				vConf.push_back(val1[j + 2].real());
			}
		}
		return 1;
	}
	else
		return 0;
}

int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, int nsift)
{
	int sift_name = SIFT_NAME, sift_version = SIFT_VERSION_4, KeyDim = 5, DescDim = 128, sift_eof = SIFT_EOF;
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		cout << "Cannot write: " << fn << endl;
		return 1;
	}
	fout.write(reinterpret_cast<char *>(&sift_name), sizeof(int));
	fout.write(reinterpret_cast<char *>(&sift_version), sizeof(int));
	fout.write(reinterpret_cast<char *>(&nsift), sizeof(int));
	fout.write(reinterpret_cast<char *>(&KeyDim), sizeof(int));
	fout.write(reinterpret_cast<char *>(&DescDim), sizeof(int));
	for (int j = 0; j < nsift; ++j)
	{
		float x = KeyPts[4 * j], y = KeyPts[4 * j + 1], dummy = 0.f;
		fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&dummy), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 2]), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 3]), sizeof(float));
	}

	for (int j = 0; j < nsift; ++j)
		for (int i = 0; i < DescDim; i++)
			fout.write(reinterpret_cast<char *>(&desc[j * 128 + i]), sizeof(unsigned char));

	fout.write(reinterpret_cast<char *>(&sift_eof), sizeof(int));
	fout.close();

	return 0;
}
int writeVisualSFMSiftGPU(const char* fn, vector<SiftGPU::SiftKeypoint> &keys, uchar *descriptors)
{
	int numKeys = (int)keys.size();
	float dummy = 0.f;
	int sift_name = SIFT_NAME, sift_version = SIFT_VERSION_4, KeyDim = 5, DescDim = 128, sift_eof = SIFT_EOF;
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		cout << "Cannot write: " << fn << endl;
		return 1;
	}
	fout.write(reinterpret_cast<char *>(&sift_name), sizeof(int));
	fout.write(reinterpret_cast<char *>(&sift_version), sizeof(int));
	fout.write(reinterpret_cast<char *>(&numKeys), sizeof(int));
	fout.write(reinterpret_cast<char *>(&KeyDim), sizeof(int));
	fout.write(reinterpret_cast<char *>(&DescDim), sizeof(int));
	for (int j = 0; j < numKeys; ++j)
	{
		fout.write(reinterpret_cast<char *>(&keys[j].x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&keys[j].y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&dummy), sizeof(float));
		fout.write(reinterpret_cast<char *>(&keys[j].s), sizeof(float));
		fout.write(reinterpret_cast<char *>(&keys[j].o), sizeof(float));
	}

	for (int j = 0; j < numKeys; ++j)
		for (int i = 0; i < DescDim; i++)
			fout.write(reinterpret_cast<char *>(&descriptors[j * 128 + i]), sizeof(unsigned char));

	fout.write(reinterpret_cast<char *>(&sift_eof), sizeof(int));
	fout.close();

	return 0;
}
int writeVisualSFMSiftGPU(const char* fn, vector<cv::KeyPoint> &keys, uchar *descriptors)
{
	int numKeys = (int)keys.size();
	float dummy = 0.f;
	int sift_name = SIFT_NAME, sift_version = SIFT_VERSION_4, KeyDim = 5, DescDim = 128, sift_eof = SIFT_EOF;
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		cout << "Cannot write: " << fn << endl;
		return 1;
	}
	fout.write(reinterpret_cast<char *>(&sift_name), sizeof(int));
	fout.write(reinterpret_cast<char *>(&sift_version), sizeof(int));
	fout.write(reinterpret_cast<char *>(&numKeys), sizeof(int));
	fout.write(reinterpret_cast<char *>(&KeyDim), sizeof(int));
	fout.write(reinterpret_cast<char *>(&DescDim), sizeof(int));
	for (int j = 0; j < numKeys; ++j)
	{
		fout.write(reinterpret_cast<char *>(&keys[j].pt.x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&keys[j].pt.y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&dummy), sizeof(float));
		fout.write(reinterpret_cast<char *>(&keys[j].size), sizeof(float));
		fout.write(reinterpret_cast<char *>(&keys[j].angle), sizeof(float));
	}

	for (int j = 0; j < numKeys; ++j)
		for (int i = 0; i < DescDim; i++)
			fout.write(reinterpret_cast<char *>(&descriptors[j * 128 + i]), sizeof(unsigned char));
	fout.write(reinterpret_cast<char *>(&sift_eof), sizeof(int));
	fout.close();

	return 0;
}
int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, int *Order, int nsift)
{
	int sift_name = SIFT_NAME, sift_version = SIFT_VERSION_4, KeyDim = 5, DescDim = 128, sift_eof = SIFT_EOF;
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		cout << "Cannot write: " << fn << endl;
		return 1;
	}
	fout.write(reinterpret_cast<char *>(&sift_name), sizeof(int));
	fout.write(reinterpret_cast<char *>(&sift_version), sizeof(int));
	fout.write(reinterpret_cast<char *>(&nsift), sizeof(int));
	fout.write(reinterpret_cast<char *>(&KeyDim), sizeof(int));
	fout.write(reinterpret_cast<char *>(&DescDim), sizeof(int));
	for (int i = 0; i < nsift; i++)
	{
		int j = Order[i];
		float x = KeyPts[4 * j], y = KeyPts[4 * j + 1], dummy = 0.f;
		fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&dummy), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 2]), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 3]), sizeof(float));
	}

	for (int k = 0; k < nsift; k++)
	{
		int j = Order[k];
		for (int i = 0; i < DescDim; i++)
			fout.write(reinterpret_cast<char *>(&desc[j * 128 + i]), sizeof(unsigned char));
	}
	fout.write(reinterpret_cast<char *>(&sift_eof), sizeof(int));
	fout.close();

	return 0;
}
int writeVisualSFMSiftGPU(const char* fn, float *KeyPts, unsigned char *desc, vector<bool> &mask, int nsift)
{
	int sift_name = SIFT_NAME, sift_version = SIFT_VERSION_4, KeyDim = 5, DescDim = 128, sift_eof = SIFT_EOF;
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		cout << "Cannot write: " << fn << endl;
		return 1;
	}
	fout.write(reinterpret_cast<char *>(&sift_name), sizeof(int));
	fout.write(reinterpret_cast<char *>(&sift_version), sizeof(int));
	fout.write(reinterpret_cast<char *>(&nsift), sizeof(int));
	fout.write(reinterpret_cast<char *>(&KeyDim), sizeof(int));
	fout.write(reinterpret_cast<char *>(&DescDim), sizeof(int));
	for (int j = 0; j < nsift; ++j)
	{
		if (!mask[j])
			continue;
		float x = KeyPts[4 * j], y = KeyPts[4 * j + 1], dummy = 0.f;
		fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&dummy), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 2]), sizeof(float));
		fout.write(reinterpret_cast<char *>(&KeyPts[4 * j + 3]), sizeof(float));
	}

	for (int j = 0; j < nsift; ++j)
	{
		if (!mask[j])
			continue;
		for (int i = 0; i < DescDim; i++)
			fout.write(reinterpret_cast<char *>(&desc[j * 128 + i]), sizeof(unsigned char));
	}
	fout.write(reinterpret_cast<char *>(&sift_eof), sizeof(int));
	fout.close();

	return 0;
}
int readVisualSFMSiftGPU(const char *fn, vector<KeyPoint>&kpts, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

															//kpts.reserve(npts);
	KeyPoint kpt;
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.pt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.pt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.size), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.angle), sizeof(float));
		kpt.pt.x -= 0.5, kpt.pt.y -= 0.5;
		kpts.push_back(kpt);
	}
	fin.close();

	return 0;
}
int readVisualSFMSiftGPU(const char *fn, vector<KeyPoint>&kpts, vector<uchar> &descriptors, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	KeyPoint kpt;
	kpts.clear(); kpts.reserve(npts);
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.pt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.pt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.size), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.angle), sizeof(float));
		kpt.pt.x -= 0.5, kpt.pt.y -= 0.5;
		kpts.push_back(kpt);
	}


	uchar d;
	descriptors.clear();
	descriptors.reserve(npts * 128);
	for (int j = 0; j < npts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uchar));
			descriptors.push_back(d);
		}
	}
	fin.close();

	return 0;
}
int readVisualSFMSiftGPU(const char *fn, vector<KeyPoint>&kpts, Mat &descriptors, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	KeyPoint kpt;
	kpts.clear(); kpts.reserve(npts);
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.pt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.pt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.size), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.angle), sizeof(float));
		kpt.pt.x -= 0.5, kpt.pt.y -= 0.5;
		kpts.push_back(kpt);
	}

	uint8_t d;
	float desci[SIFTBINS];
	/*descriptors.create(npts, SIFTBINS, CV_32F);
	for (int j = 0; j < npts; j++)
	{
	val = 0.0;
	for (int i = 0; i < descriptorSize; i++)
	{
	fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
	dummy = (int)d;
	desci[i] = (float)(int)d;
	val += desci[i] * desci[i];
	}
	val = sqrt(val);

	for (int i = 0; i < descriptorSize; i++)
	descriptors.at<float>(j, i) = desci[i] / val;
	}*/
	descriptors.create(npts, SIFTBINS, CV_8U);
	for (int j = 0; j < npts; j++)
	{
		uint8_t* Mi = descriptors.ptr<uint8_t>(j);
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
			//descriptors.at<uint8_t>(j, i) = d;
			Mi[i] = d;
		}
	}
	fin.close();

	return 0;
}
int readVisualSFMSiftGPU(const char *fn, vector<SiftKeypoint>&kpts, Mat &descriptors, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	kpts.reserve(npts);
	SiftKeypoint kpt;
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.s), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.o), sizeof(float));
		kpt.x -= 0.5, kpt.y -= 0.5;
		kpts.push_back(kpt);
	}


	uint8_t d;
	float desci[SIFTBINS];
	/*descriptors.create(npts, SIFTBINS, CV_32F);
	for (int j = 0; j < npts; j++)
	{
	val = 0.0;
	for (int i = 0; i < descriptorSize; i++)
	{
	fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
	dummy = (int)d;
	desci[i] = (float)(int)d;
	val += desci[i] * desci[i];
	}
	val = sqrt(val);
	for (int i = 0; i < descriptorSize; i++)
	descriptors.at<float>(j, i) = desci[i] / val;
	}*/
	descriptors.create(npts, SIFTBINS, CV_8U);
	for (int j = 0; j < npts; j++)
	{
		uint8_t* Mi = descriptors.ptr<uint8_t>(j);
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
			//descriptors.at<uint8_t>(j, i) = d;
			Mi[i] = d;
		}
	}
	fin.close();

	return 0;
}
int readVisualSFMSiftGPU(const char *fn, vector<SiftKeypoint>&kpts, vector<uchar> &descriptors, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	SiftKeypoint kpt;
	kpts.clear(); kpts.reserve(npts);
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.s), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.o), sizeof(float));
		kpt.x -= 0.5, kpt.y -= 0.5;
		kpts.push_back(kpt);
	}


	uchar d;
	descriptors.clear();
	descriptors.reserve(npts * 128);
	for (int j = 0; j < npts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&d), sizeof(uchar));
			descriptors.push_back(d);
		}
	}
	fin.close();

	return 0;
}
int readVisualSFMSiftGPU(const char *fn, vector<SiftKeypoint>&kpts, bool silent)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return 1;
	}
	if (!silent)
		cout << "Load " << fn << endl;

	int dummy, npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//SIFT
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));///V4.0
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));//npts
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//5 numbers
	fin.read(reinterpret_cast<char *>(&dummy), sizeof(int));//descriptorSize

	kpts.reserve(npts);
	SiftKeypoint kpt;
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&kpt.x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&val), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.s), sizeof(float));
		fin.read(reinterpret_cast<char *>(&kpt.o), sizeof(float));
		kpt.x -= 0.5, kpt.y -= 0.5;
		kpts.push_back(kpt);
	}
	fin.close();

	return 0;
}
int convertVisualSFMSiftGPU2KPointsDesc(char *Name_WO_Type)
{
	/*char Fname[512];
	Mat desc;
	vector<SiftKeypoint>kpts; kpts.reserve(15000);

	sprintf(Fname, "%s.kpts", Name_WO_Type);
	if (IsFileExist(Fname) == 1)
	return 0;

	sprintf(Fname, "%s.sift", Name_WO_Type);
	if (readVisualSFMSiftGPU(Fname, kpts, desc) == 1)
	return 1;

	sprintf(Fname, "%s.kpts", Name_WO_Type);
	WriteKPointsBinarySIFT(Fname, kpts, false);

	sprintf(Fname, "%s.desc", Name_WO_Type);
	WriteDescriptorBinarySIFT(Fname, desc);

	sprintf(Fname, "%s.sift", Name_WO_Type);
	remove(Fname);*/

	return 0;
}

//Input/Output SIFT with Minh's format (CPU coord)
bool WriteKPointsSIFT(char *fn, vector<SiftKeypoint>kpts, bool verbose)
{
	FILE *fp = fopen(fn, "w+");
	if (fp == NULL)
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = (int)kpts.size();
	fprintf(fp, "%d\n", npts);
	for (int j = 0; j < npts; ++j)
		fprintf(fp, "%.4f %.4f %.4f %.4f\n", kpts.at(j).x, kpts.at(j).y, kpts.at(j).o, kpts.at(j).s);
	fclose(fp);

	return true;
}
bool WriteKPointsBinarySIFT(char *fn, vector<SiftKeypoint>kpts, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = (int)kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).o), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).s), sizeof(float));
	}
	fout.close();

	return true;
}
bool WriteKPointsBinarySIFT(char *fn, vector<SiftKeypoint>kpts, vector<bool> &mask, int npts, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int realNpts = 0;
	for (int j = 0; j < npts; ++j)
		if (mask[j])
			realNpts++;
	fout.write(reinterpret_cast<char *>(&realNpts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		if (!mask[j])
			continue;

		fout.write(reinterpret_cast<char *>(&kpts.at(j).x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).o), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).s), sizeof(float));
	}
	fout.close();

	return true;
}
bool WriteKPointsBinarySIFT(char *fn, float *kpts, vector<bool> &mask, int npts, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		if (!mask[j])
			continue;

		fout.write(reinterpret_cast<char *>(&kpts[4 * j]), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts[4 * j + 1]), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts[4 * j + 3]), sizeof(float));//orientation
		fout.write(reinterpret_cast<char *>(&kpts[4 * j + 2]), sizeof(float));//scale
	}
	fout.close();

	return true;
}
bool ReadKPointsBinarySIFT(char *fn, vector<SiftKeypoint> &kpts, bool verbose)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		if (verbose)
			cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	float x, y, orirent, scale;
	SiftKeypoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); kpts.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		kpt.x = x, kpt.y = y, kpt.o = orirent, kpt.s = scale;
		kpts.push_back(kpt);
	}

	return true;
}
bool WriteKPointsBinarySIFT(char *fn, vector<KeyPoint>kpts, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = (int)kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).angle), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).size), sizeof(float));
	}
	fout.close();

	return true;
}
bool ReadKPointsBinarySIFT(char *fn, vector<KeyPoint> &kpts, bool verbose)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		//if (verbose)
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	float x, y, orirent, scale;
	KeyPoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	if (npts < 10 || npts > 1e6) //dummy code in case sift file is corrupted
		return false;
	kpts.reserve(npts); kpts.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		kpt.pt.x = x, kpt.pt.y = y, kpt.angle = orirent, kpt.size = scale;
		kpts.push_back(kpt);
	}

	return true;
}
bool WriteDescriptorBinarySIFT(char *fn, vector<uchar > descriptors, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int descriptorSize = SIFTBINS, npts = (int)descriptors.size() / descriptorSize;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
		for (int i = 0; i < descriptorSize; i++)
			fout.write(reinterpret_cast<char *>(&descriptors.at(i + j * descriptorSize)), sizeof(uchar));
	fout.close();

	return true;
}
bool WriteDescriptorBinarySIFT(char *fn, vector<uchar > descriptors, vector<bool>&mask, int npts, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int descriptorSize = SIFTBINS;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < (int)descriptors.size() / descriptorSize; ++j)
	{
		if (!mask[j])
			continue;
		for (int i = 0; i < descriptorSize; i++)
			fout.write(reinterpret_cast<char *>(&descriptors.at(i + j * descriptorSize)), sizeof(uchar));
	}
	fout.close();

	return true;
}
bool WriteDescriptorBinarySIFT(char *fn, Mat descriptor, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = descriptor.rows, descriptorSize = descriptor.cols;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
		for (int i = 0; i < descriptorSize; i++)
		{
			float x = descriptor.at<float>(j, i);
			fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		}
	fout.close();

	return true;
}
bool WriteDescriptorBinarySIFT(char *fn, Mat descriptor, vector<bool>&mask, int npts, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int descriptorSize = descriptor.cols;
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < descriptor.rows; ++j)
	{
		if (!mask[j])
			continue;
		for (int i = 0; i < descriptorSize; i++)
		{
			float x = descriptor.at<float>(j, i);
			fout.write(reinterpret_cast<char *>(&x), sizeof(float));
		}
	}
	fout.close();

	return true;
}
bool ReadDescriptorBinarySIFT(char *fn, vector<float > &descriptors, bool verbose)
{
	descriptors.clear();
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		if (verbose)
			cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	int npts, descriptorSize = SIFTBINS;
	float val;

	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	descriptors.reserve(descriptorSize * npts);
	for (int j = 0; j < npts; j++)
	{
		for (int i = 0; i < descriptorSize; i++)
		{
			fin.read(reinterpret_cast<char *>(&val), sizeof(float));
			descriptors.push_back(val);
		}
	}
	fin.close();

	return true;
}
Mat ReadDescriptorBinarySIFT(char *fn, bool verbose)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		if (verbose)
			cout << "Cannot open: " << fn << endl;
		Mat descriptors(1, SIFTBINS, CV_32F);
		return descriptors;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	int npts, descriptorSize = SIFTBINS;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	Mat descriptors(npts, SIFTBINS, CV_32F);
	for (int j = 0; j < npts; j++)
		for (int i = 0; i < descriptorSize; i++)
			fin.read(reinterpret_cast<char *>(&descriptors.at<float>(j, i)), sizeof(float));
	fin.close();

	return descriptors;
}

//Input/Output SIFT+rgb with Minh's format
bool WriteRGBBinarySIFT(char *fn, vector<Point3i> rgb, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = (int)rgb.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&rgb.at(j).x), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).y), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).z), sizeof(int));
	}
	fout.close();

	return true;
}
bool ReadRGBBinarySIFT(char *fn, vector<Point3i> &rgb, bool verbose)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	int r, g, b, npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	rgb.reserve(npts); rgb.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&r), sizeof(int));
		fin.read(reinterpret_cast<char *>(&g), sizeof(int));
		fin.read(reinterpret_cast<char *>(&b), sizeof(int));
		rgb.push_back(Point3i(r, g, b));
	}

	return true;
}
bool WriteKPointsRGBBinarySIFT(char *fn, vector<SiftKeypoint>kpts, vector<Point3i> rgb, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = (int)kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).o), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).s), sizeof(float));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).x), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).y), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).z), sizeof(int));
	}
	fout.close();

	return true;
}
bool ReadKPointsRGBBinarySIFT(char *fn, vector<SiftKeypoint> &kpts, vector<Point3i> &rgb, bool verbose)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	int r, g, b;
	float x, y, orirent, scale;
	SiftKeypoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); rgb.reserve(npts);  kpts.clear(); rgb.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&r), sizeof(int));
		fin.read(reinterpret_cast<char *>(&g), sizeof(int));
		fin.read(reinterpret_cast<char *>(&b), sizeof(int));
		kpt.x = x, kpt.y = y, kpt.o = orirent, kpt.s = scale;
		kpts.push_back(kpt);
		rgb.push_back(Point3i(r, g, b));
	}

	return true;
}
bool WriteKPointsRGBBinarySIFT(char *fn, vector<KeyPoint>kpts, vector<Point3i> rgb, bool verbose)
{
	ofstream fout;
	fout.open(fn, ios::binary);
	if (!fout.is_open())
	{
		if (verbose)
			cout << "Cannot write: " << fn << endl;
		return false;
	}

	int npts = (int)kpts.size();
	fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
	for (int j = 0; j < npts; ++j)
	{
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.x), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).pt.y), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).angle), sizeof(float));
		fout.write(reinterpret_cast<char *>(&kpts.at(j).size), sizeof(float));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).x), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).y), sizeof(int));
		fout.write(reinterpret_cast<char *>(&rgb.at(j).z), sizeof(int));
	}
	fout.close();

	return true;
}
bool ReadKPointsRGBBinarySIFT(char *fn, vector<KeyPoint> &kpts, vector<Point3i> &rgb, bool verbose)
{
	ifstream fin;
	fin.open(fn, ios::binary);
	if (!fin.is_open())
	{
		cout << "Cannot open: " << fn << endl;
		return false;
	}
	if (verbose)
		cout << "Load " << fn << endl;

	int r, g, b;
	float x, y, orirent, scale;
	KeyPoint kpt;

	int npts;
	fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
	kpts.reserve(npts); rgb.reserve(npts);  kpts.clear(); rgb.clear();
	for (int ii = 0; ii < npts; ii++)
	{
		fin.read(reinterpret_cast<char *>(&x), sizeof(float));
		fin.read(reinterpret_cast<char *>(&y), sizeof(float));
		fin.read(reinterpret_cast<char *>(&orirent), sizeof(float));
		fin.read(reinterpret_cast<char *>(&scale), sizeof(float));
		fin.read(reinterpret_cast<char *>(&r), sizeof(int));
		fin.read(reinterpret_cast<char *>(&g), sizeof(int));
		fin.read(reinterpret_cast<char *>(&b), sizeof(int));
		kpt.pt.x = x, kpt.pt.y = y, kpt.angle = orirent, kpt.size = scale;
		kpts.push_back(kpt);
		rgb.push_back(Point3i(r, g, b));
	}

	return true;
}

bool GrabImageCVFormat(char *fname, char *Img, int &width, int &height, int nchannels, bool verbose)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (verbose)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	width = view.cols, height = view.rows;
	if (Img == NULL)
		Img = new char[width*height*nchannels];
	int length = width * height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[nchannels*ii + jj * nchannels*width + kk] = (char)view.data[nchannels*ii + jj * nchannels*width + kk];
	}

	return true;
}
bool GrabImage(char *fname, char *Img, int &width, int &height, int nchannels, bool verbose)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (verbose)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new char[width*height*nchannels];
	}
	int length = width * height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * length] = (char)view.data[nchannels*ii + jj * nchannels*width + kk];
	}

	return true;
}
bool GrabImage(char *fname, unsigned char *Img, int &width, int &height, int nchannels, bool verbose)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (verbose)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new unsigned char[width*height*nchannels];
	}
	int length = width * height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * length] = view.data[nchannels*ii + jj * nchannels*width + kk];
	}

	return true;
}
bool GrabImage(char *fname, float *Img, int &width, int &height, int nchannels, bool verbose)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (verbose)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new float[width*height*nchannels];
	}
	int length = width * height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * length] = (float)(int)view.data[nchannels*ii + jj * nchannels*width + kk];
	}

	return true;
}
bool GrabImage(char *fname, double *Img, int &width, int &height, int nchannels, bool verbose)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (verbose)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	width = view.cols, height = view.rows;
	int length = width * height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj * width + kk * length] = (double)(int)view.data[nchannels*ii + jj * nchannels*width + kk];
	}

	return true;
}

bool SaveDataToImageCVFormat(char *fname, char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = (unsigned char)Img[nchannels*ii + kk + nchannels * jj*width];

	return imwrite(fname, M);
}
bool SaveDataToImageCVFormat(char *fname, uchar *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = Img[nchannels*ii + kk + nchannels * jj*width];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, bool *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = !Img[ii + jj * width + kk * length] ? 0 : 255;

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = (unsigned char)Img[ii + jj * width + kk * length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = Img[ii + jj * width + kk * length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, int *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = (unsigned char)Img[ii + jj * width + kk * length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, float *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = (unsigned char)(int)(Img[ii + jj * width + kk * length] + 0.5);

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = (unsigned char)(int)(Img[ii + jj * width + kk * length] + 0.5);

	return imwrite(fname, M);
}
int ExtractVideoFrames(char *Path, int camID, int startF, int stopF, int increF, int rotateImage, double resizeFactor, int nchannels, int Usejpg, int frameTimeStamp)
{
	char Fname[512];

	sprintf(Fname, "%s/%d/x.mp4", Path, camID);
	cv::VideoCapture cap = VideoCapture(Fname);
	if (!cap.isOpened())
	{
		sprintf(Fname, "%s/%d/x.mov", Path, camID);
		cap.open(Fname);
		if (!cap.isOpened())
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	bool pass = false;
	if (frameTimeStamp != 0)
	{
		startF -= frameTimeStamp, stopF -= frameTimeStamp;
		while (startF < 0)
			startF += increF; //avoid negative frame by skipping some increF cycles
		printLOG("Working on %d on sync mode with rotation %d\n", camID, rotateImage);
	}
	else
		printLOG("Working on %d with rotation %d\n", camID, rotateImage);

	Mat img, grayImg, rImg;
	int fid = 0;
	while (true)
	{
		cap >> img;
		if (img.empty())
			break;
		if (fid < startF)
		{
			fid++;
			continue;
		}

		if (fid >= startF && fid <= stopF && (fid - startF) % increF == 0)
		{
			if (rotateImage == 1) //flip updown
			{
				int width = img.cols, height = img.rows;
				for (int kk = 0; kk < nchannels; kk++)
				{
					for (int jj = 0; jj < height / 2; jj++)
						for (int ii = 0; ii < width; ii++)
						{
							char buf = img.data[nchannels*ii + jj * nchannels*width + kk];
							img.data[nchannels*ii + jj * nchannels*width + kk] = img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk];
							img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk] = buf;
						}
				}
			}

			if (Usejpg == 1)
			{
				sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fid + frameTimeStamp);
				if (IsFileExist(Fname) == 1)
				{
					fid++;
					continue;
				}
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, camID, fid + frameTimeStamp);
				if (IsFileExist(Fname) == 1)
				{
					fid++;
					continue;
				}
				if (resizeFactor == 1)
					imwrite(Fname, img, compression_params);
				else
				{
					resize(img, rImg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_CUBIC);
					imwrite(Fname, rImg, compression_params);
				}
			}
			else
			{
				sprintf(Fname, "%s/%d/%.4d.jpg", Path, camID, fid + frameTimeStamp);
				if (IsFileExist(Fname) == 1)
				{
					fid++;
					continue;
				}
				sprintf(Fname, "%s/%d/%.4d.png", Path, camID, fid + frameTimeStamp);
				if (IsFileExist(Fname) == 1)
				{
					fid++;
					continue;
				}
				if (nchannels == 1)
				{
					cvtColor(img, grayImg, CV_BGR2GRAY);
					if (resizeFactor == 1)
						imwrite(Fname, img);
					else
					{
						resize(img, rImg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_CUBIC);
						imwrite(Fname, rImg);
					}
				}
				else
				{
					if (resizeFactor == 1)
						imwrite(Fname, img);
					else
					{
						resize(img, rImg, Size(resizeFactor* img.cols, resizeFactor*img.rows), 0, 0, INTER_CUBIC);
						imwrite(Fname, rImg);
					}
				}
			}
		}
		if (fid > stopF)
			break;
		fid++;
	}
	cap.release();

	return 0;
}
int ExtractVideoFrames(char *Path, char *inName, int startF, int stopF, int increF, int rotateImage, int nchannels, int Usejpg)
{
	char Fname[512];

	sprintf(Fname, "%s/%s.mp4", Path, inName);
	cv::VideoCapture cap = VideoCapture(Fname);
	if (!cap.isOpened())
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	Mat img;
	int fid = 0;
	while (true)
	{
		cap >> img;
		if (img.empty())
			break;

		if (rotateImage == 1)  //flip updown
		{
			int width = img.cols, height = img.rows;
			for (int kk = 0; kk < nchannels; kk++)
			{
				for (int jj = 0; jj < height / 2; jj++)
					for (int ii = 0; ii < width; ii++)
					{
						char buf = img.data[nchannels*ii + jj * nchannels*width + kk];
						img.data[nchannels*ii + jj * nchannels*width + kk] = img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk];
						img.data[nchannels*(width - 1 - ii) + (height - 1 - jj)*nchannels*width + kk] = buf;
					}
			}
		}
		else if (rotateImage == 2) //rotate right
		{
			;
		}
		else if (rotateImage == 3)//roate left
		{
			;
		}
		if (fid >= startF && fid <= stopF && (fid - startF) % increF == 0)
		{
			if (Usejpg == 1)
				sprintf(Fname, "%s/%.4d.jpg", Path, fid), imwrite(Fname, img, compression_params);
			else
				sprintf(Fname, "%s/%.4d.png", Path, fid), imwrite(Fname, img);

			//if (IsFileExist(Fname) == 1)
			//	continue;

		}
		if (fid >= stopF)
			break;
		fid++;
	}
	cap.release();

	return 0;
}
bool GrabVideoFrame2Mem(char *fname, char *Data, int &width, int &height, int &nchannels, int &nframes, int frameSample, int fixnframes)
{
	IplImage  *frame = 0;
	CvCapture *capture = cvCaptureFromFile(fname);
	if (!capture)
		return false;

	bool flag = false;
	int length, frameID = 0, frameID2 = 0;
	while (true && fixnframes > frameID)
	{
		IplImage  *frame = cvQueryFrame(capture);
		if (!frame)
		{
			cvReleaseImage(&frame);
			return true;
		}

		if (frameID == 0)
			width = frame->width, height = frame->height, nchannels = frame->nChannels, length = width * height*nchannels;

		for (int ii = 0; ii < length; ii++)
			Data[ii + length * frameID] = frame->imageData[ii];

		frameID2++;
		if (frameID2 == frameSample)
			frameID++, frameID2 = 0;
		nframes = frameID;
	}

	//cvReleaseImage(&frame);
	cvReleaseCapture(&capture);

	return true;
}

void ShowCVDataAsImage(char *Fname, char *Img, int width, int height, int nchannels, IplImage *cvImg, int rotate90)
{
	//Need to call waitkey
	int ii, jj, kk, length = width * height;

	bool createMem = false;
	if (cvImg == 0)
	{
		createMem = true;
		if (rotate90 == 0)
			cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);
		else
			cvImg = cvCreateImage(cvSize(height, width), IPL_DEPTH_8U, nchannels);
	}

	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				cvImg->imageData[nchannels*ii + kk + nchannels * jj*width] = Img[nchannels*ii + kk + nchannels * jj*width];//Img[ii+jj*width+kk*length];

	Mat img = cv::cvarrToMat(cvImg);
	if (rotate90 == 0)
		imshow(Fname, img);
	else
		imshow(Fname, img.t());
	//cvShowImage(Fname, cvImg);

	return;
}
void ShowCVDataAsImage(char *Fname, uchar *Img, int width, int height, int nchannels, IplImage *cvImg, int rotate90)
{
	//Need to call waitkey
	int ii, jj, kk, length = width * height;

	bool createMem = false;
	if (cvImg == 0)
	{
		createMem = true;
		if (rotate90 == 0)
			cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, nchannels);
		else
			cvImg = cvCreateImage(cvSize(height, width), IPL_DEPTH_8U, nchannels);
	}

	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				cvImg->imageData[nchannels*ii + kk + nchannels * jj*width] = Img[nchannels*ii + kk + nchannels * jj*width];//Img[ii+jj*width+kk*length];

	Mat img = cv::cvarrToMat(cvImg);
	if (rotate90 == 0)
		imshow(Fname, img);
	else
		imshow(Fname, img.t());
	//cvShowImage(Fname, cvImg);

	return;
}
void ShowDataAsImage(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = Img[ii + jj * width + kk * length];

	if (nchannels == 3)
	{
		imshow(fname, M);
		waitKey(-1);
		destroyWindow(fname);
	}
	else
	{
		Mat cM;
		cvtColor(M, cM, CV_GRAY2RGB);
		imshow(fname, cM);
		waitKey(-1);
		destroyWindow(fname);
	}

	return;
}
void ShowDataAsImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width * height;

	Mat M = Mat::zeros(height, width, nchannels == 1 ? CV_8UC1 : CV_8UC3);
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				M.data[nchannels*ii + kk + nchannels * jj*width] = (unsigned char)(int)(Img[ii + jj * width + kk * length] + 0.5);

	if (nchannels == 3)
	{
		imshow(fname, M);
		waitKey(-1);
		destroyWindow(fname);
	}
	else
	{
		Mat cM;
		cvtColor(M, cM, CV_GRAY2RGB);
		imshow(fname, cM);
		waitKey(-1);
		destroyWindow(fname);
	}

	return;
}

int WirteVisualSFMinput(char *Path, int startF, int stopF, int npts)
{
	//This function will write down points with upper left coor.
	int ii, jj, mm, kk;
	char Fname[512], Fname1[200], Fname2[512];

	//nptsxnviews
	int  nframes = stopF - startF + 1;
	Point2d *Correspondences = new Point2d[npts*nframes];

	for (ii = startF; ii <= stopF; ii++)
	{
		sprintf(Fname, "%s/CV_%.4d.txt", Path, ii); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot open %s", Fname);
			return 0;
		}
		for (jj = 0; jj < npts; jj++)
			fscanf(fp, "%lf %lf ", &Correspondences[ii + jj * nframes].x, &Correspondences[ii + jj * nframes].y);
		fclose(fp);
	}

	//Write out sift format
	for (jj = startF; jj <= stopF; jj++)
	{
		sprintf(Fname, "%s/%.4d.sift", Path, jj);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%d 128\n", npts);
		for (ii = 0; ii < npts; ii++)
		{
			fprintf(fp, "%.6f %.6f 0.0 0.0\n", Correspondences[jj + ii * nframes].x, Correspondences[jj + ii * nframes].y);
			for (kk = 0; kk < 128; kk++)
				fprintf(fp, "%d ", 0);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	//Write out pair of correspondences
	vector<int> matchID;
	sprintf(Fname, "%s/FeatureMatches.txt", Path); FILE *fp = fopen(Fname, "w+");
	for (jj = startF; jj < stopF; jj++)
	{
		for (ii = jj + 1; ii <= stopF; ii++)
		{
			sprintf(Fname1, "%s/%.4d.jpg", Path, jj);
			sprintf(Fname2, "%s/%.4d.jpg", Path, ii);

			fprintf(fp, "%s\n", Fname1);
			fprintf(fp, "%s\n", Fname2);
			matchID.clear();
			for (mm = 0; mm < npts; mm++)
				if (Correspondences[mm*nframes + jj].x > 0.0 && Correspondences[mm*nframes + jj].y > 0.0 && Correspondences[mm*nframes + ii].x > 0.0 && Correspondences[mm*nframes + ii].y > 0.0)
					matchID.push_back(mm);

			fprintf(fp, "%d\n", matchID.size());
			for (mm = 0; mm < matchID.size(); mm++)
				fprintf(fp, "%d ", matchID.at(mm));
			fprintf(fp, "\n");
			for (mm = 0; mm < matchID.size(); mm++)
				fprintf(fp, "%d ", matchID.at(mm));
			fprintf(fp, "\n\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	delete[]Correspondences;
	return 0;
}
bool readNVMLite(const char *filePath, Corpus &CorpusInfo, int sharedIntrinsics, int nHDs, int nVGAs, int nPanels)
{
	ifstream ifs(filePath);
	if (ifs.fail())
	{
		cerr << "Cannot load " << filePath << endl;
		return false;
	}

	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		cerr << "Can only load NVM_V3" << endl;
		return false;
	}

	if (sharedIntrinsics == 1)
	{
		double fx, fy, u0, v0, radial1;
		ifs >> token >> fx >> u0 >> fy >> v0 >> radial1;
	}

	//loading camera parameters
	int nviews;
	ifs >> nviews;
	if (nviews <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	CorpusInfo.nCameras = nviews;
	CorpusInfo.camera = new CameraData[nviews];
	double Quaterunion[4], CamCenter[3];
	for (int ii = 0; ii < nviews; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		int viewID, panelID, camID, width, height;
		std::size_t posDot = filename.find("."), slength = filename.length();
		if (slength - posDot == 4)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos != string::npos)
			{
				filename.erase(pos, 4);
				const char * str = filename.c_str();
				viewID = atoi(str);
			}
			else
			{
				pos = filename.find(".jpg");
				if (pos != string::npos)
				{
					filename.erase(pos, 4);
					const char * str = filename.c_str();
					viewID = atoi(str);
				}
				else
				{
					printLOG("cannot find %s\n", filename.c_str());
					abort();
				}
			}
		}
		else
		{
			std::size_t pos1 = filename.find("_");
			string PanelName; PanelName = filename.substr(0, 2);
			const char * str = PanelName.c_str();
			panelID = atoi(str);

			string CamName; CamName = filename.substr(pos1 + 1, 2);
			str = CamName.c_str();
			camID = atoi(str);

			viewID = panelID == 0 ? camID : nHDs + nVGAs * (panelID - 1) + camID - 1;
			width = viewID > nHDs ? 640 : 1920, height = viewID > nHDs ? 480 : 1080;
		}

		CorpusInfo.camera[viewID].intrinsic[0] = f, CorpusInfo.camera[viewID].intrinsic[1] = f,
			CorpusInfo.camera[viewID].intrinsic[2] = 0, CorpusInfo.camera[viewID].intrinsic[3] = width / 2, CorpusInfo.camera[viewID].intrinsic[4] = height / 2;

		for (int jj = 0; jj < 4; jj++)
			CorpusInfo.camera[viewID].camCenter[jj] = CamCenter[jj];

		ceres::QuaternionToRotation(Quaterunion, CorpusInfo.camera[viewID].R);
		GetTfromC(CorpusInfo.camera[viewID]);
		//mat_mul(CorpusInfo.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		//CorpusInfo.camera[viewID].T[0] = -T[0], CorpusInfo.camera[viewID].T[1] = -T[1], CorpusInfo.camera[viewID].T[2] = -T[2];

		CorpusInfo.camera[viewID].LensModel = 0;
		CorpusInfo.camera[viewID].distortion[0] = -d[0];
		for (int jj = 1; jj < 7; jj++)
			CorpusInfo.camera[viewID].distortion[jj] = 0.0;

		GetrtFromRT(CorpusInfo.camera[viewID]);
		GetKFromIntrinsic(CorpusInfo.camera[viewID]);
		AssembleP(CorpusInfo.camera[viewID]);
	}

	return true;
}
bool readNVM(const char *Fname, Corpus &CorpusInfo, vector<Point2i> &ImgSize, int nViewPlus, vector<KeyPoint> *AllKeyPts, Mat *AllDesc)
{
	ifstream ifs(Fname);
	if (ifs.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	int orgnImages = (int)ImgSize.size();
	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		cerr << "Can only load NVM_V3" << endl;
		return false;
	}

	//loading camera parameters
	cout << "Loading nvm cameras" << endl;
	int nviews;
	ifs >> nviews;
	if (nviews <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	CorpusInfo.nCameras = orgnImages;
	CorpusInfo.camera = new CameraData[orgnImages + 1];//1 is just in case camera start at 1
	double Quaterunion[4], CamCenter[3];
	vector<int> CameraOrder;

	for (int ii = 0; ii < orgnImages; ii++)
		CorpusInfo.camera[ii].valid = false;

	for (int ii = 0; ii < nviews; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		int viewID;
		std::size_t posDot = filename.find("."), slength = filename.length();
		if (slength - posDot == 4)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos != string::npos)
			{
				filename.erase(pos, 4);
				const char * str = filename.c_str();
				viewID = atoi(str);
			}
			else
			{
				pos = filename.find(".jpg");
				if (pos != string::npos)
				{
					filename.erase(pos, 4);
					const char * str = filename.c_str();
					viewID = atoi(str);
				}
				else
				{
					pos = filename.find(".png");
					if (pos != string::npos)
					{
						filename.erase(pos, 4);
						const char * str = filename.c_str();
						viewID = atoi(str);
					}
					else
					{
						printLOG("cannot find %s\n", filename.c_str());
						abort();
					}
				}
			}
		}
		CameraOrder.push_back(viewID);

		CorpusInfo.camera[viewID].valid = true;
		CorpusInfo.camera[viewID].intrinsic[0] = f, CorpusInfo.camera[viewID].intrinsic[1] = f,
			CorpusInfo.camera[viewID].intrinsic[2] = 0, CorpusInfo.camera[viewID].intrinsic[3] = 1.0*ImgSize[viewID].x / 2, CorpusInfo.camera[viewID].intrinsic[4] = 1.0*ImgSize[viewID].y / 2;
		CorpusInfo.camera[viewID].width = ImgSize[viewID].x, CorpusInfo.camera[viewID].height = ImgSize[viewID].y;

		for (int jj = 0; jj < 4; jj++)
			CorpusInfo.camera[viewID].camCenter[jj] = CamCenter[jj];

		ceres::QuaternionToRotation(Quaterunion, CorpusInfo.camera[viewID].R);
		GetTfromC(CorpusInfo.camera[viewID]);
		//mat_mul(CorpusInfo.camera[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		//CorpusInfo.camera[viewID].T[0] = -T[0], CorpusInfo.camera[viewID].T[1] = -T[1], CorpusInfo.camera[viewID].T[2] = -T[2];

		CorpusInfo.camera[viewID].LensModel = 0;
		CorpusInfo.camera[viewID].distortion[0] = -d[0];
		for (int jj = 1; jj < 7; jj++)
			CorpusInfo.camera[viewID].distortion[jj] = 0.0;

		GetrtFromRT(CorpusInfo.camera[viewID]);
		GetKFromIntrinsic(CorpusInfo.camera[viewID]);
		AssembleP(CorpusInfo.camera[viewID]);
	}

	cout << "Loading nvm points" << endl;
	int nPoints, viewID, pid;
	ifs >> nPoints;
	CorpusInfo.n3dPoints = nPoints;
	CorpusInfo.xyz.reserve(nPoints);
	CorpusInfo.viewIdAll3D.reserve(nPoints);
	CorpusInfo.pointIdAll3D.reserve(nPoints);
	CorpusInfo.uvAll3D.reserve(nPoints);
	CorpusInfo.scaleAll3D.reserve(nPoints);

	Point2d uv;
	Point3d xyz;
	Point3i rgb;
	vector<int>viewID3D, pid3D;
	vector<Point2d> uv3D;
	vector<double> scale3D;

	FeatureDesc desci;
	CorpusInfo.scaleAllViews = new vector<double>[CorpusInfo.nCameras];
	CorpusInfo.uvAllViews = new vector<Point2d>[CorpusInfo.nCameras];
	CorpusInfo.threeDIdAllViews = new vector<int>[CorpusInfo.nCameras];
	CorpusInfo.DescAllViews = new vector<FeatureDesc>[CorpusInfo.nCameras];
	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		CorpusInfo.scaleAllViews[ii].reserve(5000);
		CorpusInfo.uvAllViews[ii].reserve(5000);
		CorpusInfo.threeDIdAllViews[ii].reserve(5000);
		CorpusInfo.DescAllViews[ii].reserve(5000);
	}

	for (int i = 0; i < nPoints; i++)
	{
		viewID3D.clear(), pid3D.clear(), uv3D.clear(), scale3D.clear();
		ifs >> xyz.x >> xyz.y >> xyz.z >> rgb.x >> rgb.y >> rgb.z;
		ifs >> nviews;

		int cur3DID = (int)CorpusInfo.viewIdAll3D.size();
		for (int ii = 0; ii < nviews; ii++)
		{
			ifs >> viewID >> pid >> uv.x >> uv.y;
			uv.x += 0.5*(CorpusInfo.camera[CameraOrder[viewID]].width) - 0.5;//siftgu (0,0) is at top left of pixel, not pixel center as in cpu
			uv.y += 0.5*(CorpusInfo.camera[CameraOrder[viewID]].height) - 0.5;

			if (nviews >= nViewPlus)
			{
				viewID3D.push_back(CameraOrder[viewID]);
				pid3D.push_back((pid));
				uv3D.push_back(uv);
				scale3D.push_back(1.0);


				if (AllDesc != NULL && AllDesc[CameraOrder[viewID]].rows > 0)
					for (int jj = 0; jj < 128; jj++)
						desci.desc[jj] = AllDesc[CameraOrder[viewID]].at<uchar>(pid, jj);

				CorpusInfo.uvAllViews[CameraOrder[viewID]].push_back(uv);//CorpusInfo.uvAllViews[CameraOrder[viewID]].push_back(AllKeyPts[CameraOrder[viewID]][pid].pt); (same thing)
				if (AllKeyPts != NULL && AllKeyPts[CameraOrder[viewID]].size() > 0)
					CorpusInfo.scaleAllViews[CameraOrder[viewID]].push_back(AllKeyPts[CameraOrder[viewID]][pid].size);
				else
					CorpusInfo.scaleAllViews[CameraOrder[viewID]].push_back(1.0);
				CorpusInfo.threeDIdAllViews[CameraOrder[viewID]].push_back(cur3DID);
				if (AllDesc != NULL && AllDesc[CameraOrder[viewID]].rows > 0)
					CorpusInfo.DescAllViews[CameraOrder[viewID]].push_back(desci);
			}
		}

		if (nviews >= nViewPlus)
		{
			CorpusInfo.xyz.push_back(xyz);
			CorpusInfo.rgb.push_back(rgb);
			CorpusInfo.viewIdAll3D.push_back(viewID3D);
			CorpusInfo.uvAll3D.push_back(uv3D);
			CorpusInfo.scaleAll3D.push_back(scale3D);
			CorpusInfo.pointIdAll3D.push_back(pid3D);
		}
	}
	CorpusInfo.n3dPoints = (int)CorpusInfo.xyz.size();


	printLOG("Done with nvm\n");

	return true;
}

bool readColMap(char *Path, Corpus &CorpusInfo, int nViewPlus, int nImages)
{
	char Fname[512];

	std::string line;
	std::string item;

	printLOG("Reading camera info\n");
	sprintf(Fname, "%s/0/cameras.txt", Path);

	std::ifstream file(Fname);
	if (file.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	vector<CameraData> masterCameras;
	while (std::getline(file, line))
	{
		StringTrim(&line);

		if (line.empty() || line[0] == '#')
			continue;

		CameraData dummyCam;
		std::stringstream line_stream(line);

		// ID
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		dummyCam.viewID = atoi(item.c_str());

		if (dummyCam.viewID > masterCameras.size())
			for (int ii = 0; ii < dummyCam.viewID; ii++)
				masterCameras.push_back(dummyCam);

		// MODEL
		int lensmodel;
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		if (strcmp(item.c_str(), "SIMPLE_RADIAL") == 0)
			dummyCam.LensModel = RADIAL_TANGENTIAL_PRISM, lensmodel = 0;
		else if (strcmp(item.c_str(), "OPENCV") == 0)
			dummyCam.LensModel = RADIAL_TANGENTIAL_PRISM, lensmodel = 1;
		else
			dummyCam.LensModel = FISHEYE, lensmodel = -1;

		// WIDTH
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		dummyCam.width = atoi(item.c_str());

		// HEIGHT
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		dummyCam.height = atoi(item.c_str());

		// PARAMS
		vector<double> params;
		while (!line_stream.eof())
		{
			std::getline(line_stream, item, ' ');
			StringTrim(&item);
			params.push_back(atof(item.c_str()));
		}
		if (lensmodel == 0) //only radial:  f, cx, cy, k
		{
			dummyCam.intrinsic[0] = params[0], dummyCam.intrinsic[1] = params[0],
				dummyCam.intrinsic[2] = 0, dummyCam.intrinsic[3] = params[1], dummyCam.intrinsic[4] = params[2],
				dummyCam.distortion[0] = -params[3]; //good enough approx of radial term
		}
		else if (lensmodel == 1) //opencv: fx, fy, cx, cy, k1, k2, p1, p2
		{
			dummyCam.intrinsic[0] = params[0], dummyCam.intrinsic[1] = params[1],
				dummyCam.intrinsic[2] = 0, dummyCam.intrinsic[3] = params[2], dummyCam.intrinsic[4] = params[3],
				dummyCam.distortion[0] = params[4], dummyCam.distortion[1] = params[5],
				dummyCam.distortion[2] = params[6], dummyCam.distortion[3] = params[7];
		}
		else //FOV:  fx, fy, cx, cy, omega
		{
			dummyCam.intrinsic[0] = params[0], dummyCam.intrinsic[1] = params[1],
				dummyCam.intrinsic[2] = 0, dummyCam.intrinsic[3] = params[2], dummyCam.intrinsic[4] = params[3],
				dummyCam.distortion[0] = params[4], dummyCam.distortion[1] = dummyCam.width / 2, dummyCam.distortion[2] = dummyCam.height / 2;
		}
		dummyCam.ShutterModel = 0;
		masterCameras.push_back(dummyCam);
	}
	file.close();

	printLOG("Reading 3D points cloud info\n");
	sprintf(Fname, "%s/0/points3D.txt", Path);
	file.open(Fname);
	if (file.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	int npts;
	Point3i rgb;
	Point3f pts;

	CorpusInfo.n3dPoints = 0;
	while (std::getline(file, line))
	{
		StringTrim(&line);
		std::stringstream line_stream(line);

		if (line.empty() || line[0] == '#')
		{
			std::getline(line_stream, item, ' ');
			std::getline(line_stream, item, ' ');
			if (strcmp(item.c_str(), "Number") == 0)
			{
				std::getline(line_stream, item, ' '); //of
				std::getline(line_stream, item, ' '); //points
				std::getline(line_stream, item, ' ');
				CorpusInfo.n3dPoints = atoi(item.c_str());

				CorpusInfo.xyz.resize(CorpusInfo.n3dPoints);
				CorpusInfo.rgb.resize(CorpusInfo.n3dPoints);
				continue;
			}
			else
				continue;
		}

		// ID
		std::getline(line_stream, item, ' ');
		int ThreeDid = atoi(item.c_str()) - 1;
		if (ThreeDid >= CorpusInfo.n3dPoints)
		{
			CorpusInfo.xyz.resize(ThreeDid + 1);
			CorpusInfo.rgb.resize(ThreeDid + 1);
			CorpusInfo.n3dPoints = ThreeDid + 1;
		}

		std::getline(line_stream, item, ' '), CorpusInfo.xyz[ThreeDid].x = atof(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.xyz[ThreeDid].y = atof(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.xyz[ThreeDid].z = atof(item.c_str());

		// Color
		std::getline(line_stream, item, ' '), CorpusInfo.rgb[ThreeDid].x = atoi(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.rgb[ThreeDid].y = atoi(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.rgb[ThreeDid].z = atoi(item.c_str());

		// ERROR
		/*std::getline(line_stream, item, ' ');

		// TRACK
		while (!line_stream.eof())
		{
		std::getline(line_stream, item, ' ');
		StringTrim(&item);

		int frameID, pointID;
		std::getline(line_stream, item, ' '), frameID = atoi(item.c_str()) - 1; //colmap with multiple shared cameras is based 1.
		std::getline(line_stream, item, ' '), pointID = atoi(item.c_str());
		CorpusInfo.pointIdAll3D[ThreeDid].push_back(pointID);
		CorpusInfo.viewIdAll3D[ThreeDid].push_back(frameID);
		CorpusInfo.scaleAll3D[ThreeDid].push_back(0.1); //dummy data at first
		CorpusInfo.uvAll3D[ThreeDid].push_back(Point2d(-1, -1)); //dummy data at first
		}*/
	}
	file.close();

	CorpusInfo.scaleAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.viewIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.pointIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.uvAll3D.resize(CorpusInfo.n3dPoints);

	printLOG("Loading images info\n");
	sprintf(Fname, "%s/0/images.txt", Path);
	file.open(Fname);
	if (file.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	int nframes; CorpusInfo.nCameras = 0;
	vector<FeatureDesc> descPerView;
	vector<Point2d> uvPerView;
	vector<double> scalePerView;
	vector<int> ThreeDIDPerView;
	while (std::getline(file, line))
	{
		StringTrim(&line);
		std::stringstream line_stream(line);

		if (line.empty() || line[0] == '#')
		{
			std::getline(line_stream, item, ' ');
			std::getline(line_stream, item, ' ');
			if (strcmp(item.c_str(), "Number") == 0)
			{
				std::getline(line_stream, item, ' '); //of
				std::getline(line_stream, item, ' '); //points
				std::getline(line_stream, item, ' ');
				nframes = atoi(item.c_str());

				if (nImages != -1)
				{
					CorpusInfo.camera = new CameraData[max(nframes * 5, nframes + 50)]; //should be large enough to hold even unregistered frames
					CorpusInfo.uvAllViews = new vector<Point2d>[max(nframes * 5, nframes + 50)];
					CorpusInfo.scaleAllViews = new vector<double>[max(nframes * 5, nframes + 50)];
					CorpusInfo.DescAllViews = new 	vector<FeatureDesc>[max(nframes * 5, nframes + 50)];
					CorpusInfo.threeDIdAllViews = new vector<int>[max(nframes * 5, nframes + 50)];
					for (int ii = 0; ii < max(nframes * 5, nframes + 50); ii++)
						CorpusInfo.camera[ii].frameID = -1, CorpusInfo.camera[ii].valid = false;
				}
				else
				{
					CorpusInfo.nCameras = nImages;
					CorpusInfo.camera = new CameraData[nImages];
					CorpusInfo.uvAllViews = new vector<Point2d>[nImages];
					CorpusInfo.scaleAllViews = new vector<double>[nImages];
					CorpusInfo.DescAllViews = new 	vector<FeatureDesc>[nImages];
					CorpusInfo.threeDIdAllViews = new vector<int>[nImages];
					for (int ii = 0; ii < nImages; ii++)
						CorpusInfo.camera[ii].frameID = -1, CorpusInfo.camera[ii].valid = false;
				}
				continue;
			}
			else
				continue;
		}

		// ID
		std::getline(line_stream, item, ' ');
		if (nImages == -1)
			CorpusInfo.nCameras = max(CorpusInfo.nCameras, atoi(item.c_str()));

		CameraData dummyCam;

		// QVEC (qw, qx, qy, qz)
		std::getline(line_stream, item, ' '), dummyCam.Quat[0] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.Quat[1] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.Quat[2] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.Quat[3] = atof(item.c_str());

		NormalizeQuaternion(dummyCam.Quat);

		// TVEC
		std::getline(line_stream, item, ' '), dummyCam.T[0] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.T[1] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.T[2] = atof(item.c_str());

		// CAMERA_ID
		std::getline(line_stream, item, ' ');
		int viewID = atoi(item.c_str());
		dummyCam.viewID = viewID;

		ceres::QuaternionToRotation(dummyCam.Quat, dummyCam.R);
		GetrtFromRT(dummyCam.rt, dummyCam.R, dummyCam.T);
		CopyCamereInfo(masterCameras[viewID], dummyCam, false);
		dummyCam.valid = true;

		// NAME
		std::getline(line_stream, dummyCam.filename, ' ');

		std::size_t found = dummyCam.filename.find(".");
		std::string NameOnly = dummyCam.filename.substr(0, found);
		dummyCam.frameID = atoi(NameOnly.c_str());

		if (dummyCam.frameID >= CorpusInfo.nCameras - 1)
		{
			vector<CameraData> vTemp;
			for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
			{
				CameraData camI;
				CopyCamereInfo(CorpusInfo.camera[ii], camI);
				vTemp.push_back(camI);
			}

			delete[]CorpusInfo.camera;
			CorpusInfo.camera = new CameraData[max(dummyCam.frameID * 5, dummyCam.frameID + 50)]; //should be large enough to hold even unregistered frames
			for (int ii = 0; ii < max(dummyCam.frameID * 5, dummyCam.frameID + 50); ii++)
				CorpusInfo.camera[ii].frameID = -1, CorpusInfo.camera[ii].valid = false;

			for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
				CopyCamereInfo(vTemp[ii], CorpusInfo.camera[ii]);

			CorpusInfo.nCameras = dummyCam.frameID + 10;
		}

		CopyCamereInfo(dummyCam, CorpusInfo.camera[dummyCam.frameID]);

		// POINTS2D
		if (!std::getline(file, line))
			break;

		StringTrim(&line);
		std::stringstream line_stream2(line);
		if (!line.empty())
		{
			//read sift points
			cv::Mat descriptors;
			std::vector<KeyPoint> kpts;
			sprintf(Fname, "%s/%s.sift", Path, NameOnly.c_str());
			//if (readVisualSFMSiftGPU(Fname, kpts, descriptors) == 1)
			readVisualSFMSiftGPU(Fname, kpts);

			uvPerView.clear(), ThreeDIDPerView.clear(), descPerView.clear(), scalePerView.clear();

			int twoDid = -1;
			while (!line_stream2.eof())
			{
				twoDid++;

				int threeDid;
				Point2d pt;
				FeatureDesc descI;

				std::getline(line_stream2, item, ' '), pt.x = atof(item.c_str());
				std::getline(line_stream2, item, ' '), pt.y = atof(item.c_str());

				std::getline(line_stream2, item, ' '), threeDid = atoi(item.c_str()) - 1;
				if (item == "-1")
					continue;
				else
				{
					//For PnP
					//uvPerView.push_back(pt);
					//scalePerView.push_back(kpts[twoDid].size);
					//ThreeDIDPerView.push_back(atoi(item.c_str()));

					//descPerView.push_back(descI);
					//uchar* descPtr = descriptors.ptr<uchar>(twoDid);
					//for (int ii = 0; ii < 128; ii++)
					//	descI.desc[ii] = descPtr[ii];

					//For BA
					CorpusInfo.uvAll3D[threeDid].push_back(pt);
					if (kpts.size() > 0)
						CorpusInfo.scaleAll3D[threeDid].push_back(kpts[twoDid].size);
					else
						CorpusInfo.scaleAll3D[threeDid].push_back(1.0);
					CorpusInfo.viewIdAll3D[threeDid].push_back(dummyCam.frameID);
					CorpusInfo.pointIdAll3D[threeDid].push_back(twoDid);
				}
			}

			//CorpusInfo.uvAllViews[dummyCam.frameID] = uvPerView;
			//CorpusInfo.scaleAllViews[dummyCam.frameID] = scalePerView;
			//CorpusInfo.threeDIdAllViews[dummyCam.frameID] = ThreeDIDPerView;
			//CorpusInfo.DescAllViews[dummyCam.frameID] = descPerView;
		}
	}
	file.close();

	//Remove not nViewPlus point
	int nRemoves = 0;
	vector<int> *notGood = new vector<int>[CorpusInfo.n3dPoints];
	for (int pid = 0; pid < CorpusInfo.n3dPoints; pid++)
	{
		if (CorpusInfo.viewIdAll3D[pid].size() < nViewPlus)
		{
			nRemoves++;
			for (int fid = 0; fid < CorpusInfo.viewIdAll3D[pid].size(); fid++)
				notGood[pid].push_back(fid);
		}
	}

	int realNcorpusPoints = 0;
	for (int pid = 0; pid < CorpusInfo.n3dPoints; pid++)
		if (CorpusInfo.viewIdAll3D[pid].size() > 0)
			realNcorpusPoints++;

	printLOG("Remove (%d/%d)  %d+ points...", nRemoves, realNcorpusPoints, nViewPlus);
	for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
	{
		for (int ii = (int)notGood[jj].size() - 1; ii >= 0; ii--)//start from last to first when deleting vector stack of data
		{
			int fid = notGood[jj][ii];
			if (fid > CorpusInfo.viewIdAll3D[jj].size() - 1)
				printLOG("%d\n", jj);
			else
			{
				CorpusInfo.viewIdAll3D[jj].erase(CorpusInfo.viewIdAll3D[jj].begin() + fid);
				CorpusInfo.pointIdAll3D[jj].erase(CorpusInfo.pointIdAll3D[jj].begin() + fid);
				CorpusInfo.uvAll3D[jj].erase(CorpusInfo.uvAll3D[jj].begin() + fid);
				CorpusInfo.scaleAll3D[jj].erase(CorpusInfo.scaleAll3D[jj].begin() + fid);
			}
		}
	}
	delete[]notGood;

	//deleting points with not views
	int nValidPts = 0;
	for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
	{
		if (CorpusInfo.viewIdAll3D[jj].size() > 0)
		{
			CorpusInfo.xyz[nValidPts] = CorpusInfo.xyz[jj];
			CorpusInfo.rgb[nValidPts] = CorpusInfo.rgb[jj];
			CorpusInfo.viewIdAll3D[nValidPts] = CorpusInfo.viewIdAll3D[jj];
			CorpusInfo.pointIdAll3D[nValidPts] = CorpusInfo.pointIdAll3D[jj];
			CorpusInfo.uvAll3D[nValidPts] = CorpusInfo.uvAll3D[jj];
			CorpusInfo.scaleAll3D[nValidPts] = CorpusInfo.scaleAll3D[jj];
			nValidPts++;
		}
	}
	CorpusInfo.n3dPoints = nValidPts;
	CorpusInfo.xyz.resize(CorpusInfo.n3dPoints);
	CorpusInfo.rgb.resize(CorpusInfo.n3dPoints);
	CorpusInfo.viewIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.pointIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.uvAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.scaleAll3D.resize(CorpusInfo.n3dPoints);

	printLOG("Done!\n");

	return true;
}
bool readColMap(char *Path, Corpus &CorpusInfo, int nViewPlus, vector<KeyPoint> *AllKeyPts, Mat *AllDesc, int nImages)
{
	char Fname[512];

	std::string line;
	std::string item;

	printLOG("Reading camera info\n");
	sprintf(Fname, "%s/0/cameras.txt", Path);

	std::ifstream file(Fname);
	if (file.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	vector<CameraData> masterCameras;
	while (std::getline(file, line))
	{
		StringTrim(&line);

		if (line.empty() || line[0] == '#')
			continue;

		CameraData dummyCam;
		std::stringstream line_stream(line);

		// ID
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		dummyCam.viewID = atoi(item.c_str());

		if (dummyCam.viewID > masterCameras.size())
			for (int ii = masterCameras.size() + 1; ii <= dummyCam.viewID; ii++)
				masterCameras.push_back(dummyCam);

		// MODEL
		int lensmodel;
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		if (strcmp(item.c_str(), "SIMPLE_RADIAL") == 0)
			dummyCam.LensModel = RADIAL_TANGENTIAL_PRISM, lensmodel = 0;
		else if (strcmp(item.c_str(), "OPENCV") == 0)
			dummyCam.LensModel = RADIAL_TANGENTIAL_PRISM, lensmodel = 1;
		else
			dummyCam.LensModel = FISHEYE, lensmodel = -1;

		// WIDTH
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		dummyCam.width = atoi(item.c_str());

		// HEIGHT
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		dummyCam.height = atoi(item.c_str());

		// PARAMS
		vector<double> params;
		while (!line_stream.eof())
		{
			std::getline(line_stream, item, ' ');
			StringTrim(&item);
			params.push_back(atof(item.c_str()));
		}
		if (lensmodel == 0) //only radial:  f, cx, cy, k
		{
			dummyCam.intrinsic[0] = params[0], dummyCam.intrinsic[1] = params[0],
				dummyCam.intrinsic[2] = 0, dummyCam.intrinsic[3] = params[1], dummyCam.intrinsic[4] = params[2],
				dummyCam.distortion[0] = -params[3]; //good enough approx of radial term
		}
		else if (lensmodel == 1) //opencv: fx, fy, cx, cy, k1, k2, p1, p2
		{
			dummyCam.intrinsic[0] = params[0], dummyCam.intrinsic[1] = params[1],
				dummyCam.intrinsic[2] = 0, dummyCam.intrinsic[3] = params[2], dummyCam.intrinsic[4] = params[3],
				dummyCam.distortion[0] = params[4], dummyCam.distortion[1] = params[5],
				dummyCam.distortion[3] = params[6], dummyCam.distortion[4] = params[7];
		}
		else //FOV:  fx, fy, cx, cy, omega
		{
			dummyCam.intrinsic[0] = params[0], dummyCam.intrinsic[1] = params[1],
				dummyCam.intrinsic[2] = 0, dummyCam.intrinsic[3] = params[2], dummyCam.intrinsic[4] = params[3],
				dummyCam.distortion[0] = params[4], dummyCam.distortion[1] = dummyCam.width / 2, dummyCam.distortion[2] = dummyCam.height / 2;
		}
		dummyCam.ShutterModel = 0;
		if (dummyCam.viewID < masterCameras.size())
			CopyCamereInfo(dummyCam, masterCameras[dummyCam.viewID], true);
		else
			masterCameras.push_back(dummyCam);
	}
	file.close();

	printLOG("Reading 3D points cloud info\n");
	sprintf(Fname, "%s/0/points3D.txt", Path);
	file.open(Fname);
	if (file.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	int npts;
	Point3i rgb;
	Point3f pts;

	CorpusInfo.n3dPoints = 0;
	while (std::getline(file, line))
	{
		StringTrim(&line);
		std::stringstream line_stream(line);

		if (line.empty() || line[0] == '#')
		{
			std::getline(line_stream, item, ' ');
			std::getline(line_stream, item, ' ');
			if (strcmp(item.c_str(), "Number") == 0)
			{
				std::getline(line_stream, item, ' '); //of
				std::getline(line_stream, item, ' '); //points
				std::getline(line_stream, item, ' ');
				CorpusInfo.n3dPoints = atoi(item.c_str());

				CorpusInfo.xyz.resize(CorpusInfo.n3dPoints);
				CorpusInfo.rgb.resize(CorpusInfo.n3dPoints);
				continue;
			}
			else
				continue;
		}

		// ID
		std::getline(line_stream, item, ' ');
		int ThreeDid = atoi(item.c_str()) - 1;
		if (ThreeDid >= CorpusInfo.n3dPoints)
		{
			CorpusInfo.xyz.resize(ThreeDid + 1);
			CorpusInfo.rgb.resize(ThreeDid + 1);
			CorpusInfo.n3dPoints = ThreeDid + 1;
		}

		std::getline(line_stream, item, ' '), CorpusInfo.xyz[ThreeDid].x = atof(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.xyz[ThreeDid].y = atof(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.xyz[ThreeDid].z = atof(item.c_str());

		// Color
		std::getline(line_stream, item, ' '), CorpusInfo.rgb[ThreeDid].x = atoi(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.rgb[ThreeDid].y = atoi(item.c_str());
		std::getline(line_stream, item, ' '), CorpusInfo.rgb[ThreeDid].z = atoi(item.c_str());

		/*// ERROR
		std::getline(line_stream, item, ' ');

		// TRACK
		while (!line_stream.eof())
		{
		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		if (item.empty())
		break;

		std::getline(line_stream, item, ' '), frameID = atoi(item.c_str()) - 1; //colmap with multiple shared cameras is based 1.
		std::getline(line_stream, item, ' '), pointID = atoi(item.c_str());
		CorpusInfo.pointIdAll3D[ThreeDid].push_back(pointID);
		CorpusInfo.viewIdAll3D[ThreeDid].push_back(frameID);
		CorpusInfo.scaleAll3D[ThreeDid].push_back(0.1); //dummy data at first
		CorpusInfo.uvAll3D[ThreeDid].push_back(Point2d(-1, -1)); //dummy data at first
		}*/
	}
	file.close();

	CorpusInfo.scaleAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.viewIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.pointIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.uvAll3D.resize(CorpusInfo.n3dPoints);

	printLOG("Loading images info\n");
	sprintf(Fname, "%s/0/images.txt", Path);
	file.open(Fname);
	if (file.fail())
	{
		cerr << "Cannot load " << Fname << endl;
		return false;
	}

	int nframes; CorpusInfo.nCameras = 0;
	vector<FeatureDesc> descPerView;
	vector<Point2d> uvPerView;
	vector<double> scalePerView;
	vector<int> ThreeDIDPerView;
	while (std::getline(file, line))
	{
		StringTrim(&line);
		std::stringstream line_stream(line);

		if (line.empty() || line[0] == '#')
		{
			std::getline(line_stream, item, ' ');
			std::getline(line_stream, item, ' ');
			if (strcmp(item.c_str(), "Number") == 0)
			{
				std::getline(line_stream, item, ' '); //of
				std::getline(line_stream, item, ' '); //points
				std::getline(line_stream, item, ' ');
				nframes = atoi(item.c_str());

				if (nImages != -1)
				{
					CorpusInfo.nCameras = nImages;
					CorpusInfo.camera = new CameraData[nImages]; //should be large enough to hold even unregistered frames
					for (int ii = 0; ii < nImages; ii++)
						CorpusInfo.camera[ii].frameID = -1, CorpusInfo.camera[ii].valid = false;
				}
				else
				{
					CorpusInfo.camera = new CameraData[max(nframes * 5, nframes + 50)]; //should be large enough to hold even unregistered frames
					for (int ii = 0; ii < max(nframes * 5, nframes + 50); ii++)
						CorpusInfo.camera[ii].frameID = -1, CorpusInfo.camera[ii].valid = false;
				}

				continue;
			}
			else
				continue;
		}

		// ID
		std::getline(line_stream, item, ' ');
		if (nImages == -1)
			CorpusInfo.nCameras = max(CorpusInfo.nCameras, atoi(item.c_str()));

		CameraData dummyCam;

		// QVEC (qw, qx, qy, qz)
		std::getline(line_stream, item, ' '), dummyCam.Quat[0] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.Quat[1] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.Quat[2] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.Quat[3] = atof(item.c_str());

		NormalizeQuaternion(dummyCam.Quat);

		// TVEC
		std::getline(line_stream, item, ' '), dummyCam.T[0] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.T[1] = atof(item.c_str());
		std::getline(line_stream, item, ' '), dummyCam.T[2] = atof(item.c_str());

		// CAMERA_ID
		std::getline(line_stream, item, ' ');
		int viewID = atoi(item.c_str());
		dummyCam.viewID = viewID;

		ceres::QuaternionToRotation(dummyCam.Quat, dummyCam.R);
		GetrtFromRT(dummyCam.rt, dummyCam.R, dummyCam.T);
		CopyCamereInfo(masterCameras[viewID], dummyCam, false);
		dummyCam.valid = true;

		// NAME
		std::getline(line_stream, dummyCam.filename, ' ');

		std::size_t found = dummyCam.filename.find(".");
		std::string NameOnly = dummyCam.filename.substr(0, found);
		dummyCam.frameID = atoi(NameOnly.c_str());

		if (dummyCam.frameID >= CorpusInfo.nCameras - 1)
		{
			vector<CameraData> vTemp;
			for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
			{
				CameraData camI;
				CopyCamereInfo(CorpusInfo.camera[ii], camI);
				vTemp.push_back(camI);
			}

			delete[]CorpusInfo.camera;
			CorpusInfo.camera = new CameraData[max(dummyCam.frameID * 5, dummyCam.frameID + 50)]; //should be large enough to hold even unregistered frames
			for (int ii = 0; ii < max(dummyCam.frameID * 5, dummyCam.frameID + 50); ii++)
				CorpusInfo.camera[ii].frameID = -1, CorpusInfo.camera[ii].valid = false;

			for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
				CopyCamereInfo(vTemp[ii], CorpusInfo.camera[ii]);

			CorpusInfo.nCameras = dummyCam.frameID + 10;
		}

		CopyCamereInfo(dummyCam, CorpusInfo.camera[dummyCam.frameID]);

		// POINTS2D
		if (!std::getline(file, line))
			break;

		StringTrim(&line);
		std::stringstream line_stream2(line);
		if (!line.empty())
		{
			if (AllKeyPts[dummyCam.frameID].size() == 0)
				continue;

			uvPerView.clear(), ThreeDIDPerView.clear(), descPerView.clear(), scalePerView.clear();

			int twoDid = -1;
			while (!line_stream2.eof())
			{
				twoDid++;

				int threeDid;
				Point2d pt;
				FeatureDesc descI;

				std::getline(line_stream2, item, ' '), pt.x = atof(item.c_str());
				std::getline(line_stream2, item, ' '), pt.y = atof(item.c_str());

				std::getline(line_stream2, item, ' '), threeDid = atoi(item.c_str()) - 1;
				if (item == "-1")
					continue;
				else
				{
					//For PnP
					//uvPerView.push_back(pt);
					//scalePerView.push_back(kpts[twoDid].size);
					//ThreeDIDPerView.push_back(atoi(item.c_str()));

					//descPerView.push_back(descI);
					//uchar* descPtr = descriptors.ptr<uchar>(twoDid);
					//for (int ii = 0; ii < 128; ii++)
					//	descI.desc[ii] = descPtr[ii];

					//For BA
					int fid = dummyCam.frameID;
					float fsize = AllKeyPts[fid][twoDid].size;
					CorpusInfo.uvAll3D[threeDid].push_back(pt);
					CorpusInfo.scaleAll3D[threeDid].push_back(fsize);
					CorpusInfo.viewIdAll3D[threeDid].push_back(fid);
					CorpusInfo.pointIdAll3D[threeDid].push_back(twoDid);
				}
			}

			//CorpusInfo.uvAllViews[dummyCam.frameID] = uvPerView;
			//CorpusInfo.scaleAllViews[dummyCam.frameID] = scalePerView;
			//CorpusInfo.threeDIdAllViews[dummyCam.frameID] = ThreeDIDPerView;
			//CorpusInfo.DescAllViews[dummyCam.frameID] = descPerView;
		}
	}
	file.close();

	//Remove not nViewPlus point
	int nRemoves = 0, realNcorpusPoints = 0;
	vector<int> *notGood = new vector<int>[CorpusInfo.n3dPoints];
	for (int pid = 0; pid < CorpusInfo.n3dPoints; pid++)
	{
		if (CorpusInfo.viewIdAll3D[pid].size() == 0)
		{
			for (int fid = 0; fid < CorpusInfo.viewIdAll3D[pid].size(); fid++)
				notGood[pid].push_back(fid);
		}
		else
		{
			if (CorpusInfo.viewIdAll3D[pid].size() < nViewPlus)
			{
				nRemoves++;
				for (int fid = 0; fid < CorpusInfo.viewIdAll3D[pid].size(); fid++)
					notGood[pid].push_back(fid);
			}
			realNcorpusPoints++;
		}
	}

	printLOG("Remove (%d/%d)  non %d+ points...", nRemoves, realNcorpusPoints, nViewPlus);
	for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
	{
		for (int ii = (int)notGood[jj].size() - 1; ii >= 0; ii--)//start from last to first when deleting vector stack of data
		{
			int fid = notGood[jj][ii];
			if (fid > CorpusInfo.viewIdAll3D[jj].size() - 1)
				printLOG("%d\n", jj);
			else
			{
				CorpusInfo.viewIdAll3D[jj].erase(CorpusInfo.viewIdAll3D[jj].begin() + fid);
				CorpusInfo.pointIdAll3D[jj].erase(CorpusInfo.pointIdAll3D[jj].begin() + fid);
				CorpusInfo.uvAll3D[jj].erase(CorpusInfo.uvAll3D[jj].begin() + fid);
				CorpusInfo.scaleAll3D[jj].erase(CorpusInfo.scaleAll3D[jj].begin() + fid);
			}
		}
	}
	delete[]notGood;

	//deleting points with no views
	int nValidPts = 0;
	for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
	{
		if (CorpusInfo.viewIdAll3D[jj].size() > 0)
		{
			CorpusInfo.xyz[nValidPts] = CorpusInfo.xyz[jj];
			CorpusInfo.rgb[nValidPts] = CorpusInfo.rgb[jj];
			CorpusInfo.viewIdAll3D[nValidPts] = CorpusInfo.viewIdAll3D[jj];
			CorpusInfo.pointIdAll3D[nValidPts] = CorpusInfo.pointIdAll3D[jj];
			CorpusInfo.uvAll3D[nValidPts] = CorpusInfo.uvAll3D[jj];
			CorpusInfo.scaleAll3D[nValidPts] = CorpusInfo.scaleAll3D[jj];
			nValidPts++;
		}
	}
	CorpusInfo.n3dPoints = nValidPts;
	CorpusInfo.xyz.resize(CorpusInfo.n3dPoints);
	CorpusInfo.viewIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.pointIdAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.uvAll3D.resize(CorpusInfo.n3dPoints);
	CorpusInfo.scaleAll3D.resize(CorpusInfo.n3dPoints);

	printLOG("Generating per view 2D-3D-Desc info\n");
	vector<int> twoDiDAllViews(CorpusInfo.nCameras);
	CorpusInfo.SiftIdAllViews = new vector<int>[CorpusInfo.nCameras];
	CorpusInfo.scaleAllViews = new vector<double>[CorpusInfo.nCameras];
	CorpusInfo.uvAllViews = new vector<Point2d>[CorpusInfo.nCameras];
	CorpusInfo.threeDIdAllViews = new vector<int>[CorpusInfo.nCameras];
	CorpusInfo.DescAllViews = new vector<FeatureDesc>[CorpusInfo.nCameras];
	CorpusInfo.n3dPoints = (int)CorpusInfo.xyz.size();

	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		CorpusInfo.SiftIdAllViews[ii].reserve(10000);
		CorpusInfo.threeDIdAllViews[ii].reserve(10000);
		CorpusInfo.uvAllViews[ii].reserve(10000);
		CorpusInfo.scaleAllViews[ii].reserve(10000);
		CorpusInfo.DescAllViews[ii].reserve(10000);
	}

	int fid, pid;
	Point2d uv;
	double scale;
	FeatureDesc desci;
	for (int jj = 0; jj < CorpusInfo.n3dPoints; jj++)
	{
		for (int ii = 0; ii < (int)CorpusInfo.viewIdAll3D[jj].size(); ii++)
		{
			fid = CorpusInfo.viewIdAll3D[jj][ii], pid = CorpusInfo.pointIdAll3D[jj][ii], uv = CorpusInfo.uvAll3D[jj][ii], scale = CorpusInfo.scaleAll3D[jj][ii];

			twoDiDAllViews[fid]++;
			CorpusInfo.SiftIdAllViews[fid].push_back(pid);
			CorpusInfo.threeDIdAllViews[fid].push_back(jj);
			CorpusInfo.uvAllViews[fid].push_back(uv);
			CorpusInfo.scaleAllViews[fid].push_back(scale);

			uchar* descPtr = AllDesc[fid].ptr<uchar>(pid);
			for (int kk = 0; kk < 128; kk++)
				desci.desc[kk] = descPtr[kk];
			CorpusInfo.DescAllViews[fid].push_back(desci);
		}
	}

	//Get sift desc for all views
	int nSift, totalSift = 0, maxSift = 0;
	CorpusInfo.IDCumView.reserve(CorpusInfo.nCameras + 1);
	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		CorpusInfo.IDCumView.push_back(totalSift);
		nSift = twoDiDAllViews[ii];
		if (nSift > maxSift)
			maxSift = nSift;
		totalSift += nSift;
	}
	CorpusInfo.IDCumView.push_back(totalSift);

	printLOG("Done!\n");

	return true;
}
bool writeColMap4DenseStereo(char *Path, Corpus &CorpusInfo, int nNeighbors, int frameId)
{
	char Fname[512];
	FILE *fp = 0;

	if (nNeighbors == -1)
		nNeighbors = CorpusInfo.nCameras - 1;
	if (frameId == -1)
	{
		sprintf(Fname, "%s/dense", Path); makeDir(Fname);
		sprintf(Fname, "%s/dense/images", Path); makeDir(Fname);
		sprintf(Fname, "%s/dense/sparse", Path); makeDir(Fname);
		sprintf(Fname, "%s/dense/stereo", Path); makeDir(Fname);
		sprintf(Fname, "%s/dense/stereo/consistency_graphs", Path); makeDir(Fname);
		sprintf(Fname, "%s/dense/stereo/depth_maps", Path); makeDir(Fname);
		sprintf(Fname, "%s/dense/stereo/normal_maps", Path); makeDir(Fname);

		sprintf(Fname, "%s/dense/stereo/fusion.cfg", Path);
		fp = fopen(Fname, "w");
		for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
			fprintf(fp, "%.4d.jpg\n", cid);
		fclose(fp);

		sprintf(Fname, "%s/dense/stereo/patch-match.cfg", Path);
		fp = fopen(Fname, "w");
		for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
			fprintf(fp, "%.4d.jpg\n__auto__, %d\n", cid, nNeighbors);
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/Dynamic/%.4d", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense/images", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense/sparse", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense/stereo", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense/stereo/consistency_graphs", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense/stereo/depth_maps", Path, frameId); makeDir(Fname);
		sprintf(Fname, "%s/Dynamic/%.4d/dense/stereo/normal_maps", Path, frameId); makeDir(Fname);

		sprintf(Fname, "%s/Dynamic/%.4d/dense/stereo/fusion.cfg", Path, frameId);
		fp = fopen(Fname, "w");
		for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
			fprintf(fp, "%.4d.jpg\n", cid);
		fclose(fp);

		sprintf(Fname, "%s/Dynamic/%.4d/dense/stereo/patch-match.cfg", Path, frameId);
		fp = fopen(Fname, "w");
		for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
			fprintf(fp, "%.4d.jpg\n__auto__, %d\n", cid, nNeighbors);
		fclose(fp);
	}

	printLOG("write undistored images\n");
	int percent = 0, increment = 5, interpAlgo = -1;
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	Mat cvImg;
	uchar *Img;
	double *Para = 0;

	omp_set_num_threads(omp_get_max_threads());
	for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
	{
		int per = 100 * cid / CorpusInfo.nCameras;
		if (per >= percent)
		{
			percent += increment;
			printLOG("%d%% ...", per);
		}

		if (frameId == -1)
			sprintf(Fname, "%s/dense/images/%.4d.jpg", Path, cid);
		else
			sprintf(Fname, "%s/Dynamic/%.4d/dense/images/%.4d.jpg", Path, frameId, cid);
		if (IsFileExist(Fname))
			continue;

		if (frameId == -1)
		{
			sprintf(Fname, "%s/%.4d.png", Path, cid);
			if (!IsFileExist(Fname))
				sprintf(Fname, "%s/%.4d.jpg", Path, cid);
		}
		else
		{
			sprintf(Fname, "%s/%d/%.4d.jpg", Path, cid, CorpusInfo.vTrueFrameId[cid]);
			if (!IsFileExist(Fname))
				sprintf(Fname, "%s/%d/%.4d.png", Path, cid, CorpusInfo.vTrueFrameId[cid]);
		}
		cvImg = imread(Fname, 1);
		if (!CorpusInfo.camera[cid].valid)
		{
			if (frameId == -1)
				sprintf(Fname, "%s/dense/images/%.4d.jpg", Path, cid);
			else
				sprintf(Fname, "%s/Dynamic/%.4d/dense/images/%.4d.jpg", Path, frameId, cid);
			imwrite(Fname, cvImg.setTo(0));
			continue;
		}

		int width = cvImg.cols, height = cvImg.rows, nchannels = cvImg.channels(), length = width * height;
		if (Para == NULL)
			Para = new double[length*nchannels], Img = new uchar[length*nchannels];

#pragma omp_parallel for
		for (int kk = 0; kk < nchannels; kk++)
		{
			for (int jj = 0; jj < height; jj++)
				for (int ii = 0; ii < width; ii++)
					Img[ii + jj * width + kk * length] = cvImg.data[ii*nchannels + jj * width*nchannels + kk];
			if (Para != NULL)
				Generate_Para_Spline(Img + kk * length, Para + kk * length, width, height, interpAlgo);
		}

#pragma omp_parallel for
		for (int jj = 0; jj < height; jj++)
		{
			double S;
			for (int ii = 0; ii < width; ii++)
			{
				Point2d ImgPt(ii, jj);
				if (CorpusInfo.camera[cid].LensModel == RADIAL_TANGENTIAL_PRISM)
					LensDistortionPoint(&ImgPt, CorpusInfo.camera[cid].K, CorpusInfo.camera[cid].distortion);
				else
					FishEyeDistortionPoint(&ImgPt, CorpusInfo.camera[cid].K, CorpusInfo.camera[cid].distortion[0]);

				if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
					for (int kk = 0; kk < nchannels; kk++)
						cvImg.data[ii*nchannels + jj * width*nchannels + kk] = (unsigned char)0;
				else
				{
					for (int kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Para + kk * length, width, height, ImgPt.x, ImgPt.y, &S, -1, interpAlgo);
						cvImg.data[ii*nchannels + jj * width*nchannels + kk] = (unsigned char)(min(max(S, 0.0), 255.0) + 0.5);
					}
				}
			}
		}

		if (frameId == -1)
			sprintf(Fname, "%s/dense/images/%.4d.jpg", Path, cid);
		else
			sprintf(Fname, "%s/Dynamic/%.4d/dense/images/%.4d.jpg", Path, frameId, cid);
		imwrite(Fname, cvImg, compression_params);
	}
	delete[]Para, delete[]Img;

	//write colmap txt
	std::string line;
	std::string item;

	printLOG("\nWrite camera info\n");
	if (frameId == -1)
		sprintf(Fname, "%s/dense/sparse/cameras.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/dense/sparse/cameras.txt", Path, frameId);
	fp = fopen(Fname, "w");
	fprintf(fp, "# Camera list with one line of data per camera:\n");
	fprintf(fp, "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
	fprintf(fp, "# Number of cameras: %d\n", CorpusInfo.nCameras);
	for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
	{
		CameraData *camI = CorpusInfo.camera;
		fprintf(fp, "%d PINHOLE %d %d %.4f %.4f %.4f %.4f\n", cid + 1, camI[cid].width, camI[cid].height, camI[cid].intrinsic[0], camI[cid].intrinsic[1], camI[cid].intrinsic[3], camI[cid].intrinsic[4]);
	}
	fclose(fp);

	printLOG("Writing 3D points cloud info\n");
	if (frameId == -1)
		sprintf(Fname, "%s/dense/sparse/points3D.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/dense/sparse/points3D.txt", Path, frameId);
	fp = fopen(Fname, "w");
	fprintf(fp, "# 3D point list with one line of data per point:\n");
	fprintf(fp, "	#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as(IMAGE_ID, POINT2D_IDX)\n");
	fprintf(fp, "	# Number of points : %d, mean track length : 10.0\n", CorpusInfo.n3dPoints);
	for (int pid = 0; pid < CorpusInfo.n3dPoints; pid++)
	{
		fprintf(fp, "%d %.6f %.6f %.6f %d %d %d 0.0 ", pid, CorpusInfo.xyz[pid].x, CorpusInfo.xyz[pid].y, CorpusInfo.xyz[pid].z,
			CorpusInfo.rgb[pid].x, CorpusInfo.rgb[pid].y, CorpusInfo.rgb[pid].z);
		for (size_t ii = 0; ii < CorpusInfo.pointIdAll3D[pid].size(); ii++)
			fprintf(fp, "%d %d ", CorpusInfo.viewIdAll3D[pid][ii] + 1, CorpusInfo.pointIdAll3D[pid][ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	printLOG("Write images info\n");
	if (frameId == -1)
		sprintf(Fname, "%s/dense/sparse/images.txt", Path);
	else
		sprintf(Fname, "%s/Dynamic/%.4d/dense/sparse/images.txt", Path, frameId);
	fp = fopen(Fname, "w");
	fprintf(fp, "# Image list with two lines of data per image:\n");
	fprintf(fp, "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
	fprintf(fp, "#   POINTS2D[] as(X, Y, POINT3D_ID)\n");
	fprintf(fp, "# Number of images : %d, mean observations per image : 1000\n", CorpusInfo.nCameras);
	for (int cid = 0; cid < CorpusInfo.nCameras; cid++)
	{
		if (CorpusInfo.twoDIdAllViews[cid].size() == 0)
			continue;

		CameraData *camI = CorpusInfo.camera;
		GetRTFromrt(camI[cid]);
		ceres::AngleAxisToQuaternion(camI[cid].rt, camI[cid].Quat);

		fprintf(fp, "%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %d %.4d.jpg\n", cid + 1, camI[cid].Quat[0], camI[cid].Quat[1], camI[cid].Quat[2], camI[cid].Quat[3],
			camI[cid].T[0], camI[cid].T[1], camI[cid].T[2], cid + 1, cid);

		int maxSiftId = *max_element(CorpusInfo.twoDIdAllViews[cid].begin(), CorpusInfo.twoDIdAllViews[cid].end()) + 1;
		for (int pid = 0; pid < maxSiftId; pid++)
		{
			std::vector<int>::iterator iter = find(CorpusInfo.twoDIdAllViews[cid].begin(), CorpusInfo.twoDIdAllViews[cid].end(), pid);
			if (iter != CorpusInfo.twoDIdAllViews[cid].end())
			{
				int lpid = std::distance(CorpusInfo.twoDIdAllViews[cid].begin(), iter);
				Point2d ImgPt = CorpusInfo.uvAllViews[cid][lpid]; //coprus should be correctd
																  //if (CorpusInfo.camera[cid].LensModel == RADIAL_TANGENTIAL_PRISM)
																  //	LensDistortionPoint(&ImgPt, CorpusInfo.camera[cid].K, CorpusInfo.camera[cid].distortion);
																  //else
																  //	FishEyeDistortionPoint(&ImgPt, CorpusInfo.camera[cid].K, CorpusInfo.camera[cid].distortion[0]);

				fprintf(fp, "%.3f %.3f %d ", ImgPt.x, ImgPt.y, CorpusInfo.threeDIdAllViews[cid][lpid]);
			}
			else
				fprintf(fp, "1.0 1.0 -1 ");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	printLOG("Done!\n");

	return true;
}
bool writeDeepMVSInputData(char *Path, vector<int> &sCams, int startF, int stopF, int increF, double nearDepth, double farDepth, int nDepthLayers, double sfm2Real)
{
	//assume all images of the same camera share the same intrinsic
	//assume all cameras have the same image resolution

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	char Fname[512];
	sprintf(Fname, "%s/DenseMVS", Path);
	makeDir(Fname);

	double angleThresh = 120,
		baselineThresh = 1000; //set the threshold to ignore this option because it is hard to tune
	int nLayers = 10, nSamplesPerLayer = 64;

	double depthNum = 512, depthInterval = (farDepth - nearDepth) / depthNum;

	int nCams = (int)sCams.size();
	VideoData *VideoInfo = new VideoData[nCams];
	for (int ii = 0; ii < (int)sCams.size(); ii++)
		ReadVideoDataI(Path, VideoInfo[ii], sCams[ii], -1, -1);

	vector<double> cameraDistMatrix(nCams*nCams);
	vector<double> vDist(nCams);
	vector<int> vIds(nCams);

	int width = 0, height = 0, nchannels = 3, interpAlgo = 1;

	//Generate mapping data, assuming they share the same intrinsic
	vector<Point2d *>vMapXY(nCams);
	for (int lcid = 0; lcid < nCams; lcid++)
	{
		int validFid = 0;
		for (validFid = startF; validFid <= stopF; validFid++)
			if (VideoInfo[lcid].VideoInfo[validFid].valid)
				break;

		width = VideoInfo[lcid].VideoInfo[validFid].width, height = VideoInfo[lcid].VideoInfo[validFid].height;
		vMapXY[lcid] = new Point2d[width*height];

		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				vMapXY[lcid][ii + jj * width] = Point2d(ii, jj);


		if (VideoInfo[lcid].VideoInfo[validFid].LensModel == RADIAL_TANGENTIAL_PRISM)
			LensDistortionPoint(vMapXY[lcid], VideoInfo[lcid].VideoInfo[validFid].K, VideoInfo[lcid].VideoInfo[validFid].distortion, width*height);
		else
			FishEyeDistortionPoint(vMapXY[lcid], VideoInfo[lcid].VideoInfo[validFid].K, VideoInfo[lcid].VideoInfo[validFid].distortion[0], width*height);
	}

	Mat cvImg;
	int length = width * height;
	double *Para = new double[length*nchannels];
	unsigned char *Img = new unsigned char[length*nchannels];

	for (int fid = startF; fid <= stopF; fid += increF)
	{
		//create cameraDistMatrix
		for (int lcid1 = 0; lcid1 < sCams.size(); lcid1++)
		{
			for (int lcid2 = 0; lcid2 < sCams.size(); lcid2++)
			{
				if (lcid1 == lcid2)
				{
					cameraDistMatrix[lcid1*nCams + lcid2] = 1.0;
					continue;
				}

				int fid1 = fid, fid2 = fid;
				double score = computeOverlappingMetric(VideoInfo[lcid1].VideoInfo[fid1], VideoInfo[lcid2].VideoInfo[fid2], angleThresh, baselineThresh, nearDepth, farDepth, nLayers, nSamplesPerLayer);
				cameraDistMatrix[lcid1*nCams + lcid2] = score;
				cameraDistMatrix[lcid2*nCams + lcid1] = score;
			}
		}

		//write down the pair.txt
		sprintf(Fname, "%s/DenseMVS/%.4d", Path, fid); makeDir(Fname);
		sprintf(Fname, "%s/DenseMVS/%.4d/pair.txt", Path, fid);
		FILE *fp = fopen(Fname, "w");
		fprintf(fp, "%d\n", nCams);
		for (int lcid = 0; lcid < nCams; lcid++)
		{
			//sort the distance
			for (int ii = 0; ii < nCams; ii++)
				vDist[ii] = -cameraDistMatrix[lcid*nCams + ii], vIds[ii] = ii;
			Quick_Sort_Double(&vDist[0], &vIds[0], 0, nCams - 1);

			fprintf(fp, "%d\n%d ", sCams[lcid], nCams - 1);
			for (int ii = 1; ii < nCams; ii++)
				fprintf(fp, "%d %.2f ", sCams[vIds[ii]], -vDist[ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);

		//write down the camera
		sprintf(Fname, "%s/DenseMVS/%.4d/cams", Path, fid); makeDir(Fname);
		for (int lcid = 0; lcid < nCams; lcid++)
		{
			CameraData *camI = VideoInfo[lcid].VideoInfo;
			sprintf(Fname, "%s/DenseMVS/%.4d/cams/%.8d_cam.txt", Path, fid, sCams[lcid]);
			fp = fopen(Fname, "w");
			fprintf(fp, "extrinsic\n");
			fprintf(fp, "%.8f %.8f %.8f %.8f\n", camI[fid].R[0], camI[fid].R[1], camI[fid].R[2], camI[fid].T[0]);
			fprintf(fp, "%.8f %.8f %.8f %.8f\n", camI[fid].R[3], camI[fid].R[4], camI[fid].R[5], camI[fid].T[1]);
			fprintf(fp, "%.8f %.8f %.8f %.8f\n", camI[fid].R[6], camI[fid].R[7], camI[fid].R[8], camI[fid].T[2]);
			fprintf(fp, "0.0 0.0 0.0 1.0\n\n");

			fprintf(fp, "intrinsic\n");
			fprintf(fp, "%.8f %.8f %.8f\n", camI[fid].intrinsic[0], camI[fid].intrinsic[2], camI[fid].intrinsic[3]);
			fprintf(fp, "0.0 %.8f %.8f\n", camI[fid].intrinsic[1], camI[fid].intrinsic[4]);
			fprintf(fp, "0.0 0.0 1.0\n\n");

			fprintf(fp, "%f %f %f %f\n", nearDepth, depthInterval, depthNum, farDepth);
			fclose(fp);
		}

		//write images
		sprintf(Fname, "%s/DenseMVS/%.4d/images_post", Path, fid); makeDir(Fname);
		for (int lcid = 0; lcid < nCams; lcid++)
		{
			sprintf(Fname, "%s/%d/%.4d.jpg", Path, sCams[lcid], fid);
			cvImg = imread(Fname);
			if (cvImg.empty())
				continue;

			for (int kk = 0; kk < nchannels; kk++)
			{
				for (int jj = 0; jj < height; jj++)
					for (int ii = 0; ii < width; ii++)
						Img[ii + jj * width + kk * width*height] = cvImg.data[ii*nchannels + jj * width*nchannels + kk];
				Generate_Para_Spline(Img + kk * width*height, Para + kk * width*height, width, height, interpAlgo);
			}

			double S[3];
			for (int jj = 0; jj < height; jj++)
			{
				for (int ii = 0; ii < width; ii++)
				{
					Point2d ImgPt = vMapXY[lcid][ii + jj * width];
					if (ImgPt.x < 0 || ImgPt.x > width - 1 || ImgPt.y<0.0 || ImgPt.y > height - 1)
						for (int kk = 0; kk < nchannels; kk++)
							cvImg.data[ii*nchannels + jj * width*nchannels + kk] = (unsigned char)0;
					else
					{
						for (int kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(Para + kk * length, width, height, ImgPt.x, ImgPt.y, S, -1, interpAlgo);
							S[0] = min(max(S[0], 0.0), 255.0);
							cvImg.data[ii*nchannels + jj * width*nchannels + kk] = (unsigned char)(S[0] + 0.5);
						}
					}
				}
			}

			sprintf(Fname, "%s/DenseMVS/%.4d/images_post/%.8d.jpg", Path, fid, sCams[lcid]);
			imwrite(Fname, cvImg, compression_params);
		}
		printLOG("%d\n", fid);
	}

	return true;
}

int WritePCL_PCD(char *Fname, vector<Point3f> &Vxyz)
{
	FILE *fp = fopen(Fname, "w");
	if (fp == NULL)
	{
		printLOG("Cannot write %s\n", Fname);
		return 1;
	}

	fprintf(fp, "# .PCD v.5 - Point Cloud Data file format\n");
	fprintf(fp, "VERSION .5\n");
	fprintf(fp, "FIELDS x y z\n");
	fprintf(fp, "SIZE 4 4 4\n");
	fprintf(fp, "TYPE F F F\n");
	fprintf(fp, "COUNT 1 1 1\n");
	fprintf(fp, "WIDTH %d\n", Vxyz.size());
	fprintf(fp, "HEIGHT 1\n");
	fprintf(fp, "POINTS %d\n", Vxyz.size());
	fprintf(fp, "DATA ascii\n");
	for (auto pt : Vxyz)
		fprintf(fp, "%f %f %f\n", pt.x, pt.y, pt.z);
	fclose(fp);

	return 0;
}
int ReadPCL_PLY(char *Fname, vector<Point3f> &Vxyz, vector<Point3i> &Vtri)
{
	if (IsFileExist(Fname) == 0)
	{
		printLOG("Cannot read %s\n", Fname);
		return 1;
	}

	std::string line, item;
	std::ifstream file(Fname);
	std::getline(file, line); //ply
	std::getline(file, line);//format ascii 1.0
	std::getline(file, line);//comment VTK generated PLY File
	std::getline(file, line);//obj_info vtkPolyData points and polygons: vtk4.0

	std::getline(file, line); //element vertex XXX
	StringTrim(&line);
	std::stringstream line_stream(line);
	std::getline(line_stream, item, ' ');
	std::getline(line_stream, item, ' ');
	std::getline(line_stream, item, ' ');
	StringTrim(&item);
	int nvertices = atoi(item.c_str());

	std::getline(file, line); //property float x
	std::getline(file, line); //property float y
	std::getline(file, line); //property float z

	std::getline(file, line); //element face XXX
	StringTrim(&line);
	std::stringstream line_stream2(line);
	std::getline(line_stream2, item, ' ');
	std::getline(line_stream2, item, ' ');
	std::getline(line_stream2, item, ' ');
	StringTrim(&item);
	int nfaces = atoi(item.c_str());

	std::getline(file, line); //property list uchar int vertex_indices
	std::getline(file, line); //end_header

	Point3d xyz;
	Vxyz.reserve(nvertices);
	for (int vid = 0; vid < nvertices; vid++)
	{
		std::getline(file, line);
		StringTrim(&line);
		std::stringstream line_stream(line);

		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		xyz.x = atof(item.c_str());

		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		xyz.y = atof(item.c_str());

		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		xyz.z = atof(item.c_str());

		Vxyz.push_back(xyz);
	}

	Point3i tri;
	Vtri.reserve(nfaces);
	for (int fid = 0; fid < nfaces; fid++)
	{
		std::getline(file, line);
		StringTrim(&line);
		std::stringstream line_stream(line);

		std::getline(line_stream, item, ' ');
		StringTrim(&item);

		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		tri.x = atoi(item.c_str());

		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		tri.y = atoi(item.c_str());

		std::getline(line_stream, item, ' ');
		StringTrim(&item);
		tri.z = atoi(item.c_str());

		Vtri.push_back(tri);
	}
	file.close();
	return 0;
}

void Save3DPoints(char *Path, Point3d *All3D, vector<int>Selected3DIndex)
{
	char Fname[512];
	sprintf(Fname, "%s/3D.xyz", Path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < Selected3DIndex.size(); ii++)
	{
		int pID = Selected3DIndex[ii];
		fprintf(fp, "%.3f %.3f %.3f\n", All3D[pID].x, All3D[pID].y, All3D[pID].z);
	}
	fclose(fp);
}
int SaveCorpusInfo(char *Path, Corpus &CorpusInfo, bool notbinary, bool saveDescriptor)
{
	int ii, jj, kk;
	char Fname[512];
	sprintf(Fname, "%s/Corpus_3D.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	CorpusInfo.n3dPoints = (int)CorpusInfo.xyz.size();
	fprintf(fp, "%d %d ", CorpusInfo.nCameras, CorpusInfo.n3dPoints);

	//xyz rgb viewid3D pointid3D 3dId2D cumpoint
	if (CorpusInfo.rgb.size() == 0)
	{
		fprintf(fp, "0\n");
		for (jj = 0; jj < CorpusInfo.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf \n", CorpusInfo.xyz[jj].x, CorpusInfo.xyz[jj].y, CorpusInfo.xyz[jj].z);
	}
	else
	{
		fprintf(fp, "1\n");
		for (jj = 0; jj < CorpusInfo.xyz.size(); jj++)
			fprintf(fp, "%lf %lf %lf %d %d %d\n", CorpusInfo.xyz[jj].x, CorpusInfo.xyz[jj].y, CorpusInfo.xyz[jj].z, CorpusInfo.rgb[jj].x, CorpusInfo.rgb[jj].y, CorpusInfo.rgb[jj].z);
	}
	fclose(fp);


	if (notbinary)
	{
		sprintf(Fname, "%s/Corpus_viewIdAll3D.txt", Path); fp = fopen(Fname, "w+");
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int nviews = (int)CorpusInfo.viewIdAll3D[jj].size();
			fprintf(fp, "%d ", nviews);
			for (ii = 0; ii < nviews; ii++)
				fprintf(fp, "%d ", CorpusInfo.viewIdAll3D[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_pointIdAll3D.txt", Path); fp = fopen(Fname, "w+");
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.pointIdAll3D[jj].size();
			fprintf(fp, "%d ", npts);
			for (ii = 0; ii < npts; ii++)
				fprintf(fp, "%d ", CorpusInfo.pointIdAll3D[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_uvAll3D.txt", Path); fp = fopen(Fname, "w+");
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.uvAll3D[jj].size();
			fprintf(fp, "%d ", npts);
			for (ii = 0; ii < npts; ii++)
				fprintf(fp, "%8f %8f %.2f ", CorpusInfo.uvAll3D[jj][ii].x, CorpusInfo.uvAll3D[jj][ii].y, CorpusInfo.scaleAll3D[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "w+");
		for (jj = 0; jj < CorpusInfo.nCameras; jj++)
		{
			int n3D = (int)CorpusInfo.threeDIdAllViews[jj].size();
			fprintf(fp, "%d\n", n3D);
			for (ii = 0; ii < n3D; ii++)
				fprintf(fp, "%d ", CorpusInfo.threeDIdAllViews[jj][ii]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	else
	{
		ofstream fout;
		sprintf(Fname, "%s/Corpus_viewIdAll3D.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int nviews = (int)CorpusInfo.viewIdAll3D[jj].size();
			fout.write(reinterpret_cast<char *>(&nviews), sizeof(int));
			for (ii = 0; ii < nviews; ii++)
				fout.write(reinterpret_cast<char *>(&CorpusInfo.viewIdAll3D[jj][ii]), sizeof(int));
		}
		fout.close();

		sprintf(Fname, "%s/Corpus_pointIdAll3D.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.pointIdAll3D[jj].size();
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (ii = 0; ii < npts; ii++)
				fout.write(reinterpret_cast<char *>(&CorpusInfo.pointIdAll3D[jj][ii]), sizeof(int));
		}
		fout.close();

		sprintf(Fname, "%s/Corpus_uvAll3D.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.uvAll3D[jj].size();
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (ii = 0; ii < npts; ii++)
			{
				float u = CorpusInfo.uvAll3D[jj][ii].x, v = CorpusInfo.uvAll3D[jj][ii].y, s = CorpusInfo.scaleAll3D[jj][ii];
				fout.write(reinterpret_cast<char *>(&u), sizeof(float));
				fout.write(reinterpret_cast<char *>(&v), sizeof(float));
				fout.write(reinterpret_cast<char *>(&s), sizeof(float));
			}
		}
		fout.close();

		sprintf(Fname, "%s/Corpus_threeDIdAllViews.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.nCameras; jj++)
		{
			int n3D = (int)CorpusInfo.threeDIdAllViews[jj].size();
			fout.write(reinterpret_cast<char *>(&n3D), sizeof(int));
			for (ii = 0; ii < n3D; ii++)
				fout.write(reinterpret_cast<char *>(&CorpusInfo.threeDIdAllViews[jj][ii]), sizeof(int));
		}
		fout.close();
	}

	sprintf(Fname, "%s/Corpus_cum.txt", Path); fp = fopen(Fname, "w+");
	for (int ii = 0; ii < CorpusInfo.IDCumView.size(); ii++)
		fprintf(fp, "%d ", CorpusInfo.IDCumView[ii]);
	fclose(fp);

	for (ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		sprintf(Fname, "%s/CorpusK_%.4d.txt", Path, ii); FILE *fp = fopen(Fname, "w+");
		int npts = (int)CorpusInfo.uvAllViews[ii].size();
		for (int jj = 0; jj < npts; jj++)
			fprintf(fp, "%d %.4f %.4f %.2f %d\n", CorpusInfo.threeDIdAllViews[ii][jj], CorpusInfo.uvAllViews[ii][jj].x, CorpusInfo.uvAllViews[ii][jj].y, CorpusInfo.scaleAllViews[ii][jj], CorpusInfo.SiftIdAllViews[ii][jj]);
		fclose(fp);
	}

	sprintf(Fname, "%s/Corpus_Intrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusInfo.nCameras; viewID++)
	{
		fprintf(fp, "%d %d %d %d %d ", CorpusInfo.camera[viewID].frameID, CorpusInfo.camera[viewID].LensModel, CorpusInfo.camera[viewID].ShutterModel, CorpusInfo.camera[viewID].width, CorpusInfo.camera[viewID].height);
		for (int ii = 0; ii < 5; ii++)
			fprintf(fp, "%.6f ", CorpusInfo.camera[viewID].intrinsic[ii]);

		if (CorpusInfo.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fprintf(fp, "%.6f ", CorpusInfo.camera[viewID].distortion[ii]);
		else
		{
			for (int ii = 0; ii < 3; ii++)
				fprintf(fp, "%.6f ", CorpusInfo.camera[viewID].distortion[ii]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_Extrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusInfo.nCameras; viewID++)
	{
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%.16e ", CorpusInfo.camera[viewID].rt[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_extendedExtrinsics.txt", Path); fp = fopen(Fname, "w+");
	for (int viewID = 0; viewID < CorpusInfo.nCameras; viewID++)
	{
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%.16e ", CorpusInfo.camera[viewID].wt[ii]);
		//CorpusInfo.camera[viewID].ShutterModel = 1;
		fprintf(fp, "\n");
	}
	fclose(fp);

	if (!saveDescriptor)
		return 0;

	if (notbinary)
	{
		for (kk = 0; kk < CorpusInfo.nCameras; kk++)
		{
			sprintf(Fname, "%s/CorpusD_%.4d.txt", Path, kk);	fp = fopen(Fname, "w+");
			int npts = (int)CorpusInfo.threeDIdAllViews[kk].size();
			fprintf(fp, "%d\n", npts);
			for (jj = 0; jj < npts; jj++)
			{
				fprintf(fp, "%d ", CorpusInfo.threeDIdAllViews[kk][jj]);
				for (ii = 0; ii < SIFTBINS; ii++)
					fprintf(fp, "%d ", CorpusInfo.DescAllViews[kk][jj].desc[ii]);
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
	}
	else
	{
		for (kk = 0; kk < CorpusInfo.nCameras; kk++)
		{
			if (CorpusInfo.DescAllViews[kk].size() == 0)
				continue;
			sprintf(Fname, "%s/CorpusD_%.4d.txt", Path, kk); ofstream fout; fout.open(Fname, ios::binary);
			int npts = (int)CorpusInfo.threeDIdAllViews[kk].size();
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (jj = 0; jj < npts; jj++)
			{
				fout.write(reinterpret_cast<char *>(&CorpusInfo.threeDIdAllViews[kk][jj]), sizeof(int));
				for (ii = 0; ii < SIFTBINS; ii++)
					fout.write(reinterpret_cast<char *>(&CorpusInfo.DescAllViews[kk][jj].desc[ii]), sizeof(uchar));
			}
			fout.close();
		}
	}

	return 0;
}
int ReadCorpusInfo(char *Path, Corpus &CorpusInfo, bool notbinary, bool notReadDescriptor, vector<int> CorpusImagesToRead)
{
	int ii, jj, kk, nCameras, nPoints, useColor;
	Point3d xyz;
	Point3i rgb;

	//xyz rgb viewid3D pointid3D 3dId2D cumpoint

	char Fname[512];
	sprintf(Fname, "%s/nHCorpus_3D.txt", Path);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/nCorpus_3D.txt", Path);
		if (IsFileExist(Fname) == 0)
			sprintf(Fname, "%s/Corpus_3D.txt", Path);
	}
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	fscanf(fp, "%d %d %d", &nCameras, &nPoints, &useColor);
	CorpusInfo.nCameras = nCameras;
	CorpusInfo.n3dPoints = nPoints;
	CorpusInfo.xyz.reserve(nPoints);
	if (useColor)
	{
		CorpusInfo.rgb.reserve(nPoints);
		for (jj = 0; jj < nPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf %d %d %d", &xyz.x, &xyz.y, &xyz.z, &rgb.x, &rgb.y, &rgb.z);
			CorpusInfo.xyz.push_back(xyz);
			CorpusInfo.rgb.push_back(rgb);
		}
	}
	else
	{
		CorpusInfo.rgb.reserve(nPoints);
		for (jj = 0; jj < nPoints; jj++)
		{
			fscanf(fp, "%lf %lf %lf ", &xyz.x, &xyz.y, &xyz.z);
			CorpusInfo.xyz.push_back(xyz);
		}
	}
	fclose(fp);

	CorpusInfo.viewIdAll3D.reserve(nPoints);
	CorpusInfo.pointIdAll3D.reserve(nPoints);
	CorpusInfo.uvAll3D.reserve(nPoints);
	CorpusInfo.scaleAll3D.reserve(nPoints);

	int nviews, viewID, npts, pid, totalPts = 0, n3D, id3D;
	vector<int> pointIDs, viewIDs; viewIDs.reserve(nCameras / 10);
	Point2d uv;	vector<Point2d> uvVector; uvVector.reserve(50);
	double scale = 1.0;  vector<double> scaleVector; scaleVector.reserve(2000);

	CorpusInfo.threeDIdAllViews = new vector<int>[CorpusInfo.nCameras];
	CorpusInfo.twoDIdAllViews = new vector<int>[CorpusInfo.nCameras];
	CorpusInfo.uvAllViews = new vector<Point2d>[CorpusInfo.nCameras];
	CorpusInfo.scaleAllViews = new vector<double>[CorpusInfo.nCameras];
	CorpusInfo.DescAllViews = new vector<FeatureDesc>[CorpusInfo.nCameras];

	ifstream fin;
	ofstream fout;
	sprintf(Fname, "%s/Corpus_viewIdAll3D.dat", Path);
	if (IsFileExist(Fname))
	{
		fin.open(Fname, ios::binary);
		for (jj = 0; jj < nPoints; jj++)
		{
			viewIDs.clear();
			fin.read(reinterpret_cast<char *>(&nviews), sizeof(int));
			for (ii = 0; ii < nviews; ii++)
			{
				fin.read(reinterpret_cast<char *>(&viewID), sizeof(int));
				viewIDs.push_back(viewID);
			}
			CorpusInfo.viewIdAll3D.push_back(viewIDs);
		}
		fin.close();
	}
	else
	{
		sprintf(Fname, "%s/Corpus_viewIdAll3D.txt", Path);
		fp = fopen(Fname, "r");
		for (jj = 0; jj < nPoints; jj++)
		{
			viewIDs.clear();
			fscanf(fp, "%d ", &nviews);
			for (ii = 0; ii < nviews; ii++)
			{
				fscanf(fp, "%d ", &viewID);
				viewIDs.push_back(viewID);
			}
			CorpusInfo.viewIdAll3D.push_back(viewIDs);
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_viewIdAll3D.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int nviews = (int)CorpusInfo.viewIdAll3D[jj].size();
			fout.write(reinterpret_cast<char *>(&nviews), sizeof(int));
			for (ii = 0; ii < nviews; ii++)
				fout.write(reinterpret_cast<char *>(&CorpusInfo.viewIdAll3D[jj][ii]), sizeof(int));
		}
		fout.close();
	}

	sprintf(Fname, "%s/Corpus_pointIdAll3D.dat", Path);
	if (IsFileExist(Fname))
	{
		fin.open(Fname, ios::binary);
		for (jj = 0; jj < nPoints; jj++)
		{
			pointIDs.clear();
			fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
			for (ii = 0; ii < npts; ii++)
			{
				fin.read(reinterpret_cast<char *>(&pid), sizeof(int));
				pointIDs.push_back(pid);
			}
			CorpusInfo.pointIdAll3D.push_back(pointIDs);
		}
		fin.close();
	}
	else
	{
		sprintf(Fname, "%s/Corpus_pointIdAll3D.txt", Path); fp = fopen(Fname, "r");
		for (jj = 0; jj < nPoints; jj++)
		{
			pointIDs.clear();
			fscanf(fp, "%d ", &npts);
			for (ii = 0; ii < npts; ii++)
			{
				fscanf(fp, "%d ", &pid);
				pointIDs.push_back(pid);
			}
			CorpusInfo.pointIdAll3D.push_back(pointIDs);
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_pointIdAll3D.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.pointIdAll3D[jj].size();
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (ii = 0; ii < npts; ii++)
				fout.write(reinterpret_cast<char *>(&CorpusInfo.pointIdAll3D[jj][ii]), sizeof(int));
		}
		fout.close();
	}

	sprintf(Fname, "%s/Corpus_uvAll3D.dat", Path);
	if (IsFileExist(Fname))
	{
		fin.open(Fname, ios::binary);
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			uvVector.clear(), scaleVector.clear();
			fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
			for (ii = 0; ii < npts; ii++)
			{
				float u, v, s;
				fin.read(reinterpret_cast<char *>(&u), sizeof(int));
				fin.read(reinterpret_cast<char *>(&v), sizeof(int));
				fin.read(reinterpret_cast<char *>(&s), sizeof(int));
				uvVector.push_back(Point2d(u, v)), scaleVector.push_back(s);
			}
			CorpusInfo.uvAll3D.push_back(uvVector);
			CorpusInfo.scaleAll3D.push_back(scaleVector);
		}
		fin.close();
	}
	else
	{
		sprintf(Fname, "%s/Corpus_uvAll3D.txt", Path); fp = fopen(Fname, "r");
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			uvVector.clear(), scaleVector.clear();
			fscanf(fp, "%d ", &npts);
			for (ii = 0; ii < npts; ii++)
			{
				fscanf(fp, "%lf %lf %lf", &uv.x, &uv.y, &scale);
				uvVector.push_back(uv), scaleVector.push_back(scale);
			}
			CorpusInfo.uvAll3D.push_back(uvVector);
			CorpusInfo.scaleAll3D.push_back(scaleVector);
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_uvAll3D.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.n3dPoints; jj++)
		{
			int npts = (int)CorpusInfo.uvAll3D[jj].size();
			fout.write(reinterpret_cast<char *>(&npts), sizeof(int));
			for (ii = 0; ii < npts; ii++)
			{
				float u = CorpusInfo.uvAll3D[jj][ii].x, v = CorpusInfo.uvAll3D[jj][ii].y, s = CorpusInfo.scaleAll3D[jj][ii];
				fout.write(reinterpret_cast<char *>(&u), sizeof(float));
				fout.write(reinterpret_cast<char *>(&v), sizeof(float));
				fout.write(reinterpret_cast<char *>(&s), sizeof(float));
			}
		}
		fout.close();
	}

	sprintf(Fname, "%s/Corpus_threeDIdAllViews.dat", Path);
	if (IsFileExist(Fname))
	{
		fin.open(Fname, ios::binary);
		for (jj = 0; jj < CorpusInfo.nCameras; jj++)
		{
			fin.read(reinterpret_cast<char *>(&n3D), sizeof(int));
			CorpusInfo.threeDIdAllViews[jj].reserve(n3D);
			for (ii = 0; ii < n3D; ii++)
			{
				fin.read(reinterpret_cast<char *>(&id3D), sizeof(int));
				CorpusInfo.threeDIdAllViews[jj].push_back(id3D);
			}
		}
		fin.close();
	}
	else
	{
		sprintf(Fname, "%s/Corpus_threeDIdAllViews.txt", Path); fp = fopen(Fname, "r");
		for (jj = 0; jj < CorpusInfo.nCameras; jj++)
		{
			fscanf(fp, "%d ", &n3D);
			CorpusInfo.threeDIdAllViews[jj].reserve(n3D);
			for (ii = 0; ii < n3D; ii++)
			{
				fscanf(fp, "%d ", &id3D);
				CorpusInfo.threeDIdAllViews[jj].push_back(id3D);
			}
		}
		fclose(fp);

		sprintf(Fname, "%s/Corpus_threeDIdAllViews.dat", Path);
		fout.open(Fname, ios::binary);
		if (!fout.is_open())
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < CorpusInfo.nCameras; jj++)
		{
			int n3D = (int)CorpusInfo.threeDIdAllViews[jj].size();
			fout.write(reinterpret_cast<char *>(&n3D), sizeof(int));
			for (ii = 0; ii < n3D; ii++)
				fout.write(reinterpret_cast<char *>(&CorpusInfo.threeDIdAllViews[jj][ii]), sizeof(int));
		}
		fout.close();
	}

	sprintf(Fname, "%s/Corpus_cum.txt", Path); fp = fopen(Fname, "r");
	CorpusInfo.IDCumView.reserve(nCameras + 1);
	for (kk = 0; kk < nCameras + 1; kk++)
	{
		fscanf(fp, "%d ", &totalPts);
		CorpusInfo.IDCumView.push_back(totalPts);
	}
	fclose(fp);

	for (ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		int threeDID, TwoDID;
		CorpusInfo.uvAllViews[ii].reserve(3000);
		CorpusInfo.scaleAllViews[ii].reserve(3000);
		sprintf(Fname, "%s/CorpusK_%.4d.txt", Path, ii); FILE *fp = fopen(Fname, "r");
		while (fscanf(fp, "%d %lf %lf %lf %d", &threeDID, &uv.x, &uv.y, &scale, &TwoDID) != EOF)
		{
			uvVector.push_back(uv), scaleVector.push_back(scale);
			CorpusInfo.uvAllViews[ii].push_back(uv);
			CorpusInfo.twoDIdAllViews[ii].push_back(TwoDID);
			CorpusInfo.scaleAllViews[ii].push_back(scale);
		}
		fclose(fp);
	}

	sprintf(Fname, "%s/Corpus_Intrinsics.txt", Path); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	CorpusInfo.camera = new CameraData[nCameras];
	for (viewID = 0; viewID < nCameras; viewID++)
	{
		fscanf(fp, "%d %d %d %d %d ", &CorpusInfo.camera[viewID].frameID, &CorpusInfo.camera[viewID].LensModel, &CorpusInfo.camera[viewID].ShutterModel, &CorpusInfo.camera[viewID].width, &CorpusInfo.camera[viewID].height);
		for (int ii = 0; ii < 5; ii++)
			fscanf(fp, "%lf ", &CorpusInfo.camera[viewID].intrinsic[ii]);

		if (CorpusInfo.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int ii = 0; ii < 7; ii++)
				fscanf(fp, "%lf ", &CorpusInfo.camera[viewID].distortion[ii]);
		else
		{
			for (int ii = 0; ii < 3; ii++)
				fscanf(fp, "%lf ", &CorpusInfo.camera[viewID].distortion[ii]);
		}
		GetKFromIntrinsic(CorpusInfo.camera[viewID]);
		CorpusInfo.camera[viewID].notCalibrated = false;
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_Extrinsics.txt", Path); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}
	for (viewID = 0; viewID < nCameras; viewID++)
	{
		for (int ii = 0; ii < 6; ii++)
			fscanf(fp, "%lf ", &CorpusInfo.camera[viewID].rt[ii]);
		if (abs(CorpusInfo.camera[viewID].rt[0]) > 1e-16)
			CorpusInfo.camera[viewID].valid = true;
		GetRTFromrt(CorpusInfo.camera[viewID]);
	}
	fclose(fp);

	sprintf(Fname, "%s/Corpus_extendedExtrinsics.txt", Path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		for (viewID = 0; viewID < nCameras; viewID++)
			for (int ii = 0; ii < 6; ii++)
				fscanf(fp, "%lf ", &CorpusInfo.camera[viewID].wt[ii]);
		fclose(fp);
	}
	else
	{
		for (viewID = 0; viewID < nCameras; viewID++)
			for (int ii = 0; ii < 6; ii++)
				CorpusInfo.camera[viewID].wt[ii] = 0.0;
	}

	if (notReadDescriptor)
		return 0;

	FeatureDesc desci;
	totalPts = 0;
	if (notbinary)
	{
		for (kk = 0; kk < nCameras; kk++)
		{
			if (CorpusImagesToRead.size() > 0)
			{
				bool flag = true;
				for (int ll = 0; ll < (int)CorpusImagesToRead.size(); ll++)
				{
					if (CorpusImagesToRead[ll] == kk)
					{
						flag = false;
						break;
					}
				}
				if (flag)
					continue;
			}

			sprintf(Fname, "%s/CorpusD_%.4d.txt", Path, kk);	fp = fopen(Fname, "r");
			int npts; fscanf(fp, "%d ", &npts);
			CorpusInfo.DescAllViews[kk].reserve(npts);
			for (jj = 0; jj < npts; jj++)
			{
				fscanf(fp, "%d ", &id3D);
				for (ii = 0; ii < SIFTBINS; ii++)
					fscanf(fp, "%d ", &desci.desc[ii]);
				CorpusInfo.DescAllViews[kk].push_back(desci);
				totalPts++;
			}
			fclose(fp);
		}
	}
	else
	{
		for (kk = 0; kk < nCameras; kk++)
		{
			if (CorpusImagesToRead.size() > 0)
			{
				bool flag = true;
				for (int ll = 0; ll < (int)CorpusImagesToRead.size(); ll++)
				{
					if (CorpusImagesToRead[ll] == kk)
					{
						flag = false;
						break;
					}
				}
				if (flag)
					continue;
			}
			sprintf(Fname, "%s/CorpusD_%.4d.txt", Path, kk);
			ifstream fin; fin.open(Fname, ios::binary);
			if (!fin.is_open())
			{
				cout << "Cannot open: " << Fname << endl;
				continue;
			}

			int npts; fin.read(reinterpret_cast<char *>(&npts), sizeof(int));
			CorpusInfo.DescAllViews[kk].reserve(npts);
			for (jj = 0; jj < npts; jj++)
			{
				fin.read(reinterpret_cast<char *>(&id3D), sizeof(int));
				for (ii = 0; ii < SIFTBINS; ii++)
					fin.read(reinterpret_cast<char *>(&desci.desc[ii]), sizeof(uchar));
				CorpusInfo.DescAllViews[kk].push_back(desci);
				totalPts++;
			}
			fin.close();
		}
	}

	return 0;
}
int ReadCorpusAndVideoData(char *Path, CorpusandVideo &CorpusandVideoInfo, int ScannedCopursCam, int nVideoViews, int startTime, int stopTime, int LensModel, int distortionCorrected)
{
	char Fname[512];

	//READ INTRINSIC: START
	CameraData *IntrinsicInfo = new CameraData[nVideoViews];
	if (ReadIntrinsicResults(Path, IntrinsicInfo) != 0)
		return 1;
	for (int ii = 0; ii < nVideoViews; ii++)
	{
		IntrinsicInfo[ii].LensModel = LensModel, IntrinsicInfo[ii].threshold = 3.0, IntrinsicInfo[ii].nInlierThresh = 40;
		if (distortionCorrected == 1)
			for (int jj = 0; jj < 7; jj++)
				IntrinsicInfo[ii].distortion[jj] = 0.0;
	}
	//END

	//READ POSE FROM CORPUS: START
	sprintf(Fname, "%s/Corpus.nvm", Path);
	ifstream ifs(Fname);
	if (ifs.fail())
	{
		printLOG("Cannot load %s\n", Fname);
		return 1;
	}

	string token;
	ifs >> token; //NVM_V3
	if (token != "NVM_V3")
	{
		printLOG("Can only load NVM_V3\n");
		return 1;
	}
	double fx, fy, u0, v0, radial1;
	ifs >> token >> fx >> u0 >> fy >> v0 >> radial1;

	//loading camera parameters
	ifs >> CorpusandVideoInfo.nViewsCorpus;
	if (CorpusandVideoInfo.nViewsCorpus <= 1)
	{
		cerr << "# of cameras must be more than 1." << endl;
		return false;
	}
	CorpusandVideoInfo.CorpusInfo = new CameraData[CorpusandVideoInfo.nViewsCorpus];

	double Quaterunion[4], CamCenter[3], T[3];
	for (int ii = 0; ii < CorpusandVideoInfo.nViewsCorpus; ii++)
	{
		string filename;
		double f;
		vector<double> q(4), c(3), d(2);
		ifs >> filename >> f >> Quaterunion[0] >> Quaterunion[1] >> Quaterunion[2] >> Quaterunion[3] >> CamCenter[0] >> CamCenter[1] >> CamCenter[2] >> d[0] >> d[1];

		std::size_t pos = filename.find(".ppm");
		filename.erase(pos, 4);
		const char * str = filename.c_str();
		int viewID = atoi(str);

		ceres::QuaternionToRotation(Quaterunion, CorpusandVideoInfo.CorpusInfo[viewID].R);
		mat_mul(CorpusandVideoInfo.CorpusInfo[viewID].R, CamCenter, T, 3, 3, 1); //t = -RC
		CorpusandVideoInfo.CorpusInfo[viewID].T[0] = -T[0], CorpusandVideoInfo.CorpusInfo[viewID].T[1] = -T[1], CorpusandVideoInfo.CorpusInfo[viewID].T[2] = -T[2];

		for (int jj = 0; jj < 5; jj++)
			CorpusandVideoInfo.CorpusInfo[viewID].intrinsic[jj] = IntrinsicInfo[ScannedCopursCam].intrinsic[jj];
		for (int jj = 0; jj < 7; jj++)
			CorpusandVideoInfo.CorpusInfo[viewID].distortion[jj] = IntrinsicInfo[ScannedCopursCam].distortion[jj];

		GetKFromIntrinsic(CorpusandVideoInfo.CorpusInfo[viewID]);
		GetrtFromRT(CorpusandVideoInfo.CorpusInfo[viewID].rt, CorpusandVideoInfo.CorpusInfo[viewID].R, CorpusandVideoInfo.CorpusInfo[viewID].T);
		AssembleP(CorpusandVideoInfo.CorpusInfo[viewID].K, CorpusandVideoInfo.CorpusInfo[viewID].R, CorpusandVideoInfo.CorpusInfo[viewID].T, CorpusandVideoInfo.CorpusInfo[viewID].P);
	}
	//READ POSE FROM CORPUS: END

	//READ POSE FROM VIDEO POSE: START
	CorpusandVideoInfo.nVideos = nVideoViews;
	CorpusandVideoInfo.VideoInfo = new CameraData[nVideoViews*MaxnFrames];
	int id;
	double rt[6];
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			continue;
		}
		while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &id, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]) != EOF)
		{
			for (int jj = 0; jj < 6; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].rt[jj] = rt[jj];
			GetRTFromrt(CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames]);

			for (int jj = 0; jj < 5; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].intrinsic[jj] = IntrinsicInfo[viewID].intrinsic[jj];
			for (int jj = 0; jj < 7; jj++)
				CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].distortion[jj] = IntrinsicInfo[viewID].distortion[jj];

			GetKFromIntrinsic(CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames]);
			AssembleP(CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].K, CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].R,
				CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].T, CorpusandVideoInfo.VideoInfo[id + viewID * MaxnFrames].P);
		}
	}
	//READ FROM VIDEO POSE: END

	return 0;
}

bool readBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusInfo)
{
	const int nHDs = 30, nVGAs = 24, nPanels = 20;
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", BAfileName);
		return false;
	}

	char Fname[512];
	int lensType, shutterModel, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rt[6];

	fscanf(fp, "%d ", &CorpusInfo.nCameras);
	CorpusInfo.camera = new CameraData[CorpusInfo.nCameras + 20];

	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
		CorpusInfo.camera[ii].valid = false;

	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		if (fscanf(fp, "%s %d %d %d %d", &Fname, &lensType, &shutterModel, &width, &height) == EOF)
			break;
		string filename = Fname;
		size_t dotpos = filename.find("."), slength = filename.length();
		int viewID, camID, panelID;
		if (slength - dotpos == 4)
		{
			std::size_t pos = filename.find(".ppm");
			if (pos > 1000)
			{
				pos = filename.find(".png");
				if (pos > 1000)
				{
					pos = filename.find(".jpg");
					if (pos > 1000)
					{
						printLOG("Something wrong with the image name in the BA file!\n");
						abort();
					}
				}
			}
			filename.erase(pos, 4);
			const char * str = filename.c_str();
			viewID = atoi(str);
		}
		else
		{
			std::size_t pos1 = filename.find("_");
			string PanelName; PanelName = filename.substr(0, 2);
			const char * str = PanelName.c_str();
			panelID = atoi(str);

			string CamName; CamName = filename.substr(pos1 + 1, 2);
			str = CamName.c_str();
			camID = atoi(str);

			viewID = panelID == 0 ? camID : nHDs + nVGAs * (panelID - 1) + camID - 1;
		}

		if (width != 0 && height != 0)
			CorpusInfo.camera[viewID].valid = true;
		CorpusInfo.camera[viewID].LensModel = lensType, CorpusInfo.camera[viewID].ShutterModel = shutterModel;
		CorpusInfo.camera[viewID].width = width, CorpusInfo.camera[viewID].height = height, CorpusInfo.camera[viewID].frameID = viewID;
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&r1, &r2, &r3, &t1, &t2, &p1, &p2,
				&rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);

			CorpusInfo.camera[viewID].distortion[0] = r1,
				CorpusInfo.camera[viewID].distortion[1] = r2,
				CorpusInfo.camera[viewID].distortion[2] = r3,
				CorpusInfo.camera[viewID].distortion[3] = t1,
				CorpusInfo.camera[viewID].distortion[4] = t2,
				CorpusInfo.camera[viewID].distortion[5] = p1,
				CorpusInfo.camera[viewID].distortion[6] = p2;
		}
		else
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&omega, &DistCtrX, &DistCtrY,
				&rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);

			CorpusInfo.camera[viewID].distortion[0] = omega,
				CorpusInfo.camera[viewID].distortion[1] = DistCtrX,
				CorpusInfo.camera[viewID].distortion[2] = DistCtrY;
			for (int jj = 3; jj < 7; jj++)
				CorpusInfo.camera[viewID].distortion[jj] = 0;
		}
		if (CorpusInfo.camera[viewID].ShutterModel == 1)
			fscanf(fp, "%lf %lf %lf %lf %lf %lf ", &CorpusInfo.camera[viewID].wt[0], &CorpusInfo.camera[viewID].wt[1], &CorpusInfo.camera[viewID].wt[2], &CorpusInfo.camera[viewID].wt[3], &CorpusInfo.camera[viewID].wt[4], &CorpusInfo.camera[viewID].wt[5]);
		else
			for (int jj = 0; jj < 6; jj++)
				CorpusInfo.camera[viewID].wt[jj] = 0.0;

		CorpusInfo.camera[viewID].intrinsic[0] = fx,
			CorpusInfo.camera[viewID].intrinsic[1] = fy,
			CorpusInfo.camera[viewID].intrinsic[2] = skew,
			CorpusInfo.camera[viewID].intrinsic[3] = u0,
			CorpusInfo.camera[viewID].intrinsic[4] = v0;
		GetKFromIntrinsic(CorpusInfo.camera[viewID]);

		for (int jj = 0; jj < 6; jj++)
			CorpusInfo.camera[viewID].rt[jj] = rt[jj];

		GetRTFromrt(CorpusInfo.camera[viewID].rt, CorpusInfo.camera[viewID].R, CorpusInfo.camera[viewID].T);
		GetCfromT(CorpusInfo.camera[viewID].R, CorpusInfo.camera[viewID].rt + 3, CorpusInfo.camera[viewID].camCenter);

		GetRCGL(CorpusInfo.camera[viewID]);
		AssembleP(CorpusInfo.camera[viewID]);
	}
	fclose(fp);

	return true;
}
bool saveBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusInfo)
{
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rt[6];

	FILE *fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusInfo.nCameras);

	for (int viewID = 0; viewID < CorpusInfo.nCameras; viewID++)
	{
		CameraData *camI = &CorpusInfo.camera[viewID];
		fprintf(fp, "%.4d.png %d %d %d %d ", viewID, camI->LensModel, camI->ShutterModel, camI->width, camI->height);

		fx = camI->intrinsic[0], fy = camI->intrinsic[1],
			skew = camI->intrinsic[2],
			u0 = camI->intrinsic[3], v0 = camI->intrinsic[4];

		if (camI->LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			r1 = camI->distortion[0], r2 = camI->distortion[1], r3 = camI->distortion[2],
				t1 = camI->distortion[3], t2 = camI->distortion[4],
				p1 = camI->distortion[5], p2 = camI->distortion[6];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				r1, r2, r3, t1, t2, p1, p2,
				camI->rt[0], camI->rt[1], camI->rt[2], camI->rt[3], camI->rt[4], camI->rt[5]);
			if (camI->ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", camI->wt[0], camI->wt[1], camI->wt[2],
					camI->wt[3], camI->wt[4], camI->wt[5]);
			else
				fprintf(fp, "\n");
		}
		else
		{
			omega = camI->distortion[0], DistCtrX = camI->distortion[1], DistCtrY = camI->distortion[2];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				omega, DistCtrX, DistCtrY,
				rt[0], rt[1], rt[2], rt[3], rt[4], rt[5]);
			if (camI->ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", camI->wt[0], camI->wt[1], camI->wt[2],
					camI->wt[3], camI->wt[4], camI->wt[5]);
			else
				fprintf(fp, "\n");
		}
	}
	fclose(fp);
	return true;
}
bool ReSaveBundleAdjustedNVMResults(char *BAfileName, double ScaleFactor)
{
	FILE *fp = fopen(BAfileName, "r");
	if (fp == NULL)
	{
		printLOG("Cannot load %s\n", BAfileName);
		return false;
	}

	char Fname[512];
	int lensType, shutterModel, width, height;
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rt[6], wt[6];

	Corpus CorpusInfo;
	fscanf(fp, "%d ", &CorpusInfo.nCameras);
	CorpusInfo.camera = new CameraData[CorpusInfo.nCameras];

	for (int ii = 0; ii < CorpusInfo.nCameras; ii++)
	{
		if (fscanf(fp, "%s %d %d %d %d", &Fname, &lensType, &shutterModel, &width, &height) == EOF)
			break;
		string filename = Fname;
		std::size_t pos = filename.find(".ppm");
		if (pos > 1000)
		{
			pos = filename.find(".png");
			if (pos > 1000)
			{
				pos = filename.find(".jpg");
				if (pos > 100)
				{
					printLOG("Something wrong with the image name in the BA file!\n");
					abort();
				}
			}
		}
		filename.erase(pos, 4);
		const char * str = filename.c_str();
		int viewID = atoi(str);

		CorpusInfo.camera[viewID].LensModel = lensType, CorpusInfo.camera[viewID].ShutterModel = shutterModel;
		CorpusInfo.camera[viewID].width = width, CorpusInfo.camera[viewID].height = height;
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&r1, &r2, &r3, &t1, &t2, &p1, &p2,
				&rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);

			CorpusInfo.camera[viewID].distortion[0] = r1,
				CorpusInfo.camera[viewID].distortion[1] = r2,
				CorpusInfo.camera[viewID].distortion[2] = r3,
				CorpusInfo.camera[viewID].distortion[3] = t1,
				CorpusInfo.camera[viewID].distortion[4] = t2,
				CorpusInfo.camera[viewID].distortion[5] = p1,
				CorpusInfo.camera[viewID].distortion[6] = p2;
		}
		else
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ", &fx, &fy, &skew, &u0, &v0,
				&omega, &DistCtrX, &DistCtrY,
				&rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]);

			CorpusInfo.camera[viewID].distortion[0] = omega,
				CorpusInfo.camera[viewID].distortion[1] = DistCtrX,
				CorpusInfo.camera[viewID].distortion[2] = DistCtrY;
			for (int jj = 3; jj < 7; jj++)
				CorpusInfo.camera[viewID].distortion[jj] = 0;
		}

		CorpusInfo.camera[viewID].intrinsic[0] = fx,
			CorpusInfo.camera[viewID].intrinsic[1] = fy,
			CorpusInfo.camera[viewID].intrinsic[2] = skew,
			CorpusInfo.camera[viewID].intrinsic[3] = u0,
			CorpusInfo.camera[viewID].intrinsic[4] = v0;

		for (int jj = 0; jj < 6; jj++)
			CorpusInfo.camera[viewID].rt[jj] = rt[jj];

		if (CorpusInfo.camera[viewID].ShutterModel == 1)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf ", &wt[0], &wt[1], &wt[2], &wt[3], &wt[4], &wt[5]);
			for (int jj = 0; jj < 6; jj++)
				CorpusInfo.camera[viewID].wt[jj] = wt[jj];
		}
	}
	fclose(fp);

	fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusInfo.nCameras);
	for (int viewID = 0; viewID < CorpusInfo.nCameras; viewID++)
	{
		fprintf(fp, "%.4d.png %d %d %d %d ", viewID, CorpusInfo.camera[viewID].LensModel, CorpusInfo.camera[viewID].ShutterModel, CorpusInfo.camera[viewID].width, CorpusInfo.camera[viewID].height);

		fx = CorpusInfo.camera[viewID].intrinsic[0], fy = CorpusInfo.camera[viewID].intrinsic[1], skew = CorpusInfo.camera[viewID].intrinsic[2], u0 = CorpusInfo.camera[viewID].intrinsic[3], v0 = CorpusInfo.camera[viewID].intrinsic[4];

		//Scale data
		for (int jj = 0; jj < 3; jj++)
			CorpusInfo.camera[viewID].rt[jj + 3] *= ScaleFactor;
		for (int jj = 0; jj < 6; jj++)
			rt[jj] = CorpusInfo.camera[viewID].rt[jj];

		if (CorpusInfo.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
		{
			r1 = CorpusInfo.camera[viewID].distortion[0], r2 = CorpusInfo.camera[viewID].distortion[1], r3 = CorpusInfo.camera[viewID].distortion[2],
				t1 = CorpusInfo.camera[viewID].distortion[3], t2 = CorpusInfo.camera[viewID].distortion[4], p1 = CorpusInfo.camera[viewID].distortion[5], p2 = CorpusInfo.camera[viewID].distortion[6];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				r1, r2, r3, t1, t2, p1, p2,
				rt[0], rt[1], rt[2], rt[3], rt[4], rt[5]);
			if (CorpusInfo.camera[viewID].ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusInfo.camera[viewID].wt[0], CorpusInfo.camera[viewID].wt[1], CorpusInfo.camera[viewID].wt[2],
					CorpusInfo.camera[viewID].wt[3], CorpusInfo.camera[viewID].wt[4], CorpusInfo.camera[viewID].wt[5]);
			else
				fprintf(fp, "\n");
		}
		else
		{
			omega = CorpusInfo.camera[viewID].distortion[0], DistCtrX = CorpusInfo.camera[viewID].distortion[1], DistCtrY = CorpusInfo.camera[viewID].distortion[2];
			fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
				omega, DistCtrX, DistCtrY,
				rt[0], rt[1], rt[2], rt[3], rt[4], rt[5]);
			if (CorpusInfo.camera[viewID].ShutterModel == 1)
				fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusInfo.camera[viewID].wt[0], CorpusInfo.camera[viewID].wt[1], CorpusInfo.camera[viewID].wt[2],
					CorpusInfo.camera[viewID].wt[3], CorpusInfo.camera[viewID].wt[4], CorpusInfo.camera[viewID].wt[5]);
			else
				fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return true;
}
bool ReSaveBundleAdjustedNVMResults(char *BAfileName, Corpus &CorpusInfo, double ScaleFactor)
{
	double fx, fy, skew, u0, v0, r1, r2, r3, t1, t2, p1, p2, omega, DistCtrX, DistCtrY, rt[6];

	FILE *fp = fopen(BAfileName, "w+");
	fprintf(fp, "%d \n", CorpusInfo.nCameras);
	for (int viewID = 0; viewID < CorpusInfo.nCameras; viewID++)
	{
		if (CorpusInfo.camera[viewID].valid)
		{
			fprintf(fp, "%.4d.png %d %d %d %d ", viewID, CorpusInfo.camera[viewID].LensModel, CorpusInfo.camera[viewID].ShutterModel, CorpusInfo.camera[viewID].width, CorpusInfo.camera[viewID].height);

			fx = CorpusInfo.camera[viewID].intrinsic[0], fy = CorpusInfo.camera[viewID].intrinsic[1],
				skew = CorpusInfo.camera[viewID].intrinsic[2],
				u0 = CorpusInfo.camera[viewID].intrinsic[3], v0 = CorpusInfo.camera[viewID].intrinsic[4];

			getRfromr(CorpusInfo.camera[viewID].rt, CorpusInfo.camera[viewID].R);
			GetCfromT(CorpusInfo.camera[viewID].R, CorpusInfo.camera[viewID].T, CorpusInfo.camera[viewID].camCenter);

			//Scale data
			for (int jj = 0; jj < 3; jj++)
				CorpusInfo.camera[viewID].rt[jj + 3] *= ScaleFactor;
			for (int jj = 0; jj < 6; jj++)
				rt[jj] = CorpusInfo.camera[viewID].rt[jj];

			if (CorpusInfo.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			{
				r1 = CorpusInfo.camera[viewID].distortion[0], r2 = CorpusInfo.camera[viewID].distortion[1], r3 = CorpusInfo.camera[viewID].distortion[2],
					t1 = CorpusInfo.camera[viewID].distortion[3], t2 = CorpusInfo.camera[viewID].distortion[4],
					p1 = CorpusInfo.camera[viewID].distortion[5], p2 = CorpusInfo.camera[viewID].distortion[6];
				fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
					r1, r2, r3, t1, t2, p1, p2,
					rt[0], rt[1], rt[2], rt[3], rt[4], rt[5]);
				if (CorpusInfo.camera[viewID].ShutterModel == 1)
					fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusInfo.camera[viewID].wt[0], CorpusInfo.camera[viewID].wt[1], CorpusInfo.camera[viewID].wt[2],
						CorpusInfo.camera[viewID].wt[3], CorpusInfo.camera[viewID].wt[4], CorpusInfo.camera[viewID].wt[5]);
				else
					fprintf(fp, "\n");
			}
			else
			{
				omega = CorpusInfo.camera[viewID].distortion[0], DistCtrX = CorpusInfo.camera[viewID].distortion[1], DistCtrY = CorpusInfo.camera[viewID].distortion[2];
				fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", fx, fy, skew, u0, v0,
					omega, DistCtrX, DistCtrY,
					rt[0], rt[1], rt[2], rt[3], rt[4], rt[5]);
				if (CorpusInfo.camera[viewID].ShutterModel == 1)
					fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", CorpusInfo.camera[viewID].wt[0], CorpusInfo.camera[viewID].wt[1], CorpusInfo.camera[viewID].wt[2],
						CorpusInfo.camera[viewID].wt[3], CorpusInfo.camera[viewID].wt[4], CorpusInfo.camera[viewID].wt[5]);
				else
					fprintf(fp, "\n");
			}
		}
		else
		{
			fprintf(fp, "%.4d.png %d %d %d %d ", viewID, CorpusInfo.camera[viewID].LensModel, CorpusInfo.camera[viewID].ShutterModel, 0, 0);
			if (CorpusInfo.camera[viewID].LensModel == RADIAL_TANGENTIAL_PRISM)
			{
				fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				if (CorpusInfo.camera[viewID].ShutterModel == 1)
					fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", 0, 0, 0, 0, 0, 0);
				else
					fprintf(fp, "\n");
			}
			else
			{
				fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %.16f %.16f %.16f %.16f %.16f %.16f ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				if (CorpusInfo.camera[viewID].ShutterModel == 1)
					fprintf(fp, "%.16f %.16f  %.16f %.16f %.16f %.16f \n", 0, 0, 0, 0, 0, 0);
				else
					fprintf(fp, "\n");
			}
		}
	}
	fclose(fp);

	return true;
}

bool ReadIntrinsicResults(char *Path, CameraData *AllViewsParas)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[512];
	int id = 0, lensType, shutterType, width, height;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;


	sprintf(Fname, "%s/AvgDevicesIntrinsics.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return false;
	}
	while (fscanf(fp, "%s %d %d %d %d %lf %lf %lf %lf %lf ", Fname, &lensType, &shutterType, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		string  filename = string(Fname);

		std::size_t posDot = filename.find(".");
		filename.erase(posDot, 4);
		const char * str = filename.c_str();
		id = atoi(str);

		AllViewsParas[id].LensModel = lensType, AllViewsParas[id].ShutterModel = shutterType, AllViewsParas[id].width = width, AllViewsParas[id].height = height;
		AllViewsParas[id].K[0] = fx, AllViewsParas[id].K[1] = skew, AllViewsParas[id].K[2] = u0,
			AllViewsParas[id].K[3] = 0.0, AllViewsParas[id].K[4] = fy, AllViewsParas[id].K[5] = v0,
			AllViewsParas[id].K[6] = 0.0, AllViewsParas[id].K[7] = 0.0, AllViewsParas[id].K[8] = 1.0;

		GetIntrinsicFromK(AllViewsParas[id]);
		mat_invert(AllViewsParas[id].K, AllViewsParas[id].invK);
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, " %lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			AllViewsParas[id].distortion[0] = r0, AllViewsParas[id].distortion[1] = r1, AllViewsParas[id].distortion[2] = r2;
			AllViewsParas[id].distortion[3] = t0, AllViewsParas[id].distortion[4] = t1;
			AllViewsParas[id].distortion[5] = p0, AllViewsParas[id].distortion[6] = p1;
		}
		else
		{
			fscanf(fp, " %lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			AllViewsParas[id].distortion[0] = omega, AllViewsParas[id].distortion[1] = DistCtrX, AllViewsParas[id].distortion[2] = DistCtrY;
			AllViewsParas[id].distortion[3] = 0, AllViewsParas[id].distortion[4] = 0;
			AllViewsParas[id].distortion[5] = 0, AllViewsParas[id].distortion[6] = 0;
		}
	}
	fclose(fp);

	return true;
}
bool ReadIntrinsicResultI(char *Path, int selectedCamID, CameraData &CamInfoI)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[512];
	bool found = false;
	int id = 0, lensType, shutterType, width, height;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;

	CamInfoI.valid = false;

	sprintf(Fname, "%s/AvgDevicesIntrinsics.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return false;
	}
	while (fscanf(fp, "%s %d %d %d %d %lf %lf %lf %lf %lf ", Fname, &lensType, &shutterType, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		string  filename = string(Fname);

		std::size_t posDot = filename.find(".");
		filename.erase(posDot, 4);
		const char * str = filename.c_str();
		id = atoi(str);

		if (id == selectedCamID)
		{
			CamInfoI.LensModel = lensType, CamInfoI.ShutterModel = shutterType, CamInfoI.width = width, CamInfoI.height = height,
				CamInfoI.K[0] = fx, CamInfoI.K[1] = skew, CamInfoI.K[2] = u0,
				CamInfoI.K[3] = 0.0, CamInfoI.K[4] = fy, CamInfoI.K[5] = v0,
				CamInfoI.K[6] = 0.0, CamInfoI.K[7] = 0.0, CamInfoI.K[8] = 1.0;
		}
		GetIntrinsicFromK(CamInfoI);
		mat_invert(CamInfoI.K, CamInfoI.invK);
		if (lensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, " %lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			if (id == selectedCamID)
				CamInfoI.distortion[0] = r0, CamInfoI.distortion[1] = r1, CamInfoI.distortion[2] = r2, CamInfoI.distortion[3] = t0, CamInfoI.distortion[4] = t1, CamInfoI.distortion[5] = p0, CamInfoI.distortion[6] = p1,
				found = true;
		}
		else
		{
			fscanf(fp, " %lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			if (id == selectedCamID)
				CamInfoI.distortion[0] = omega, CamInfoI.distortion[1] = DistCtrX, CamInfoI.distortion[2] = DistCtrY, CamInfoI.distortion[3] = 0, CamInfoI.distortion[4] = 0, CamInfoI.distortion[5] = 0, CamInfoI.distortion[6] = 0,
				found = true;
		}
	}
	fclose(fp);

	return found;
}
int SaveIntrinsicResults(char *Path, CameraData *AllViewsParas, vector<Point2i> camIDs)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[512];
	int LensType;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;

	sprintf(Fname, "%s/DevicesIntrinsics.txt", Path); FILE *fp = fopen(Fname, "w+");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	for (int ii = 0; ii < (int)camIDs.size(); ii++)
	{
		int rcid = camIDs[ii].x, fid = camIDs[ii].y;
		LensType = AllViewsParas[fid].LensModel;
		fx = AllViewsParas[fid].K[0], fy = AllViewsParas[fid].K[4], skew = AllViewsParas[fid].K[1], u0 = AllViewsParas[fid].K[2], v0 = AllViewsParas[fid].K[5];

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			r0 = AllViewsParas[fid].distortion[0], r1 = AllViewsParas[fid].distortion[1], r2 = AllViewsParas[fid].distortion[2];
			t0 = AllViewsParas[fid].distortion[3], t1 = AllViewsParas[fid].distortion[4];
			p0 = AllViewsParas[fid].distortion[5], p1 = AllViewsParas[fid].distortion[6];
			fprintf(fp, "%.4d.png %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", rcid, LensType, AllViewsParas[fid].ShutterModel, AllViewsParas[fid].width, AllViewsParas[fid].height, fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1);
		}
		else
		{
			omega = AllViewsParas[fid].distortion[0], DistCtrX = AllViewsParas[fid].distortion[1], DistCtrY = AllViewsParas[fid].distortion[2];
			fprintf(fp, "%.4d.png %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf \n", rcid, LensType, AllViewsParas[fid].ShutterModel, AllViewsParas[fid].width, AllViewsParas[fid].height, fx, fy, skew, u0, v0, omega, DistCtrX, DistCtrY);
		}
	}
	fclose(fp);

	return 0;
}
int SaveAvgIntrinsicResults(char *Path, CameraData *AllViewsParas, vector<int> SharedCameraToBuildCorpus)
{
	//Note that visCamualSfm use different lens model than openCV or matlab or yours (inverse model)
	char Fname[512];
	int LensType;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;

	int nVideoCams = -1;
	vector<int>CamID;
	for (int ii = 0; ii < (int)SharedCameraToBuildCorpus.size(); ii++)
		if (nVideoCams < SharedCameraToBuildCorpus[ii])
			nVideoCams = SharedCameraToBuildCorpus[ii], CamID.push_back(nVideoCams);

	int *nImgesToAvg = new int[(int)CamID.size()];
	CameraData *avgCams = new CameraData[(int)CamID.size()];
	for (int cid = 0; cid < (int)CamID.size(); cid++)
	{
		nImgesToAvg[cid] = 0;
		for (int jj = 0; jj < (int)SharedCameraToBuildCorpus.size(); jj++)
		{
			if (SharedCameraToBuildCorpus[jj] == cid)
			{
				if (AllViewsParas[jj].intrinsic[0] > 0) //do not average unlocalized images
				{
					for (int kk = 0; kk < 5; kk++)
						avgCams[cid].intrinsic[kk] += AllViewsParas[jj].intrinsic[kk];
					for (int kk = 0; kk < 7; kk++)
						avgCams[cid].distortion[kk] += AllViewsParas[jj].distortion[kk];
					avgCams[cid].width = AllViewsParas[jj].width, avgCams[cid].height = AllViewsParas[jj].height;
					avgCams[cid].ShutterModel = AllViewsParas[jj].ShutterModel, avgCams[cid].LensModel = AllViewsParas[jj].LensModel;
					nImgesToAvg[cid]++;
				}
			}
		}
		if (nImgesToAvg[cid] > 0)
		{
			for (int kk = 0; kk < 5; kk++)
				avgCams[cid].intrinsic[kk] = avgCams[cid].intrinsic[kk] / nImgesToAvg[cid];
			for (int kk = 0; kk < 7; kk++)
				avgCams[cid].distortion[kk] = avgCams[cid].distortion[kk] / nImgesToAvg[cid];
		}
	}

	sprintf(Fname, "%s/AvgDevicesIntrinsics.txt", Path); FILE *fp = fopen(Fname, "w+");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 1;
	}
	for (int cid = 0; cid < (int)CamID.size(); cid++)
	{
		LensType = avgCams[cid].LensModel;
		fx = avgCams[cid].intrinsic[0], fy = avgCams[cid].intrinsic[1], skew = avgCams[cid].intrinsic[2], u0 = avgCams[cid].intrinsic[3], v0 = avgCams[cid].intrinsic[4];

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			r0 = avgCams[cid].distortion[0], r1 = avgCams[cid].distortion[1], r2 = avgCams[cid].distortion[2];
			t0 = avgCams[cid].distortion[3], t1 = avgCams[cid].distortion[4];
			p0 = avgCams[cid].distortion[5], p1 = avgCams[cid].distortion[6];
			fprintf(fp, "%.4d.png %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", cid, LensType, avgCams[cid].ShutterModel, avgCams[cid].width, avgCams[cid].height, fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1);
		}
		else
		{
			omega = avgCams[cid].distortion[0], DistCtrX = avgCams[cid].distortion[1], DistCtrY = avgCams[cid].distortion[2];
			fprintf(fp, "%.4d.png %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf \n", cid, LensType, avgCams[cid].ShutterModel, avgCams[cid].width, avgCams[cid].height, fx, fy, skew, u0, v0, omega, DistCtrX, DistCtrY);
		}
	}
	fclose(fp);

	return 0;
}
void ReadCurrentSfmInfo(char *Path, CameraData *AllViewParas, vector<int>&AvailViews, Point3d *All3D, int npts)
{
	char Fname[512];
	int viewID;

	AvailViews.clear();
	sprintf(Fname, "%s/Dinfo.txt", Path);
	FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%d: ", &viewID) != EOF)
	{
		AvailViews.push_back(viewID);
		for (int jj = 0; jj < 5; jj++)
			fscanf(fp, "%lf ", &AllViewParas[viewID].intrinsic[jj]);
		for (int jj = 0; jj < 7; jj++)
			fscanf(fp, "%lf ", &AllViewParas[viewID].distortion[jj]);
		for (int jj = 0; jj < 6; jj++)
			fscanf(fp, "%lf ", &AllViewParas[viewID].rt[jj]);
	}
	fclose(fp);
	sort(AvailViews.begin(), AvailViews.end());

	GetKFromIntrinsic(AllViewParas, AvailViews);
	GetRTFromrt(AllViewParas, AvailViews);

	if (All3D != NULL)
	{
		sprintf(Fname, "%s/3d.xyz", Path);
		fp = fopen(Fname, "r");
		for (int ii = 0; ii < npts; ii++)
			fscanf(fp, "%lf %lf %lf ", &All3D[ii].x, &All3D[ii].y, &All3D[ii].z);
		fclose(fp);
	}

	return;
}
void SaveVideoCameraIntrinsic(char *Fname, CameraData *AllViewParas, vector<int>&AvailTime, int camID, int StartTime)
{
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < AvailTime.size(); ii++)
	{
		int timeID = AvailTime[ii];

		fprintf(fp, "%d %d %d %d %d ", timeID + StartTime, AllViewParas[timeID].LensModel, AllViewParas[timeID].ShutterModel, AllViewParas[timeID].width, AllViewParas[timeID].height);
		for (int jj = 0; jj < 5; jj++)
			fprintf(fp, "%f ", AllViewParas[timeID].intrinsic[jj]);
		if (AllViewParas[timeID].LensModel == RADIAL_TANGENTIAL_PRISM)
			for (int jj = 0; jj < 7; jj++)
				fprintf(fp, "%e ", AllViewParas[timeID].distortion[jj]);
		else
			for (int jj = 0; jj < 3; jj++)
				fprintf(fp, "%e ", AllViewParas[timeID].distortion[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void SaveVideoCameraPoses(char *Fname, CameraData *AllViewParas, vector<int>&AvailTime, int camID, int StartTime)
{
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < AvailTime.size(); ii++)
	{
		int timeID = AvailTime[ii];
		fprintf(fp, "%d ", timeID + StartTime);

		for (int jj = 0; jj < 6; jj++)
			fprintf(fp, "%.16f ", AllViewParas[timeID].rt[jj]);
		if (AllViewParas[timeID].ShutterModel == ROLLING_SHUTTER)
			for (int jj = 0; jj < 6; jj++)
				fprintf(fp, "%.16f ", AllViewParas[timeID].wt[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}

int ReadVideoData(char *Path, VideoData &AllVideoInfo, int nVideoViews, int startTime, int stopTime, double threshold, int ninliersThresh)
{
	char Fname[512];
	int videoID, frameID, LensType, ShutterModel, width, height;
	int maxFrameOffset = AllVideoInfo.maxFrameOffset, nframes = stopTime + maxFrameOffset + 1;

	AllVideoInfo.nframesI = nframes;
	AllVideoInfo.nVideos = nVideoViews;
	AllVideoInfo.VideoInfo = new CameraData[nVideoViews*nframes];

	for (int ii = 0; ii < nVideoViews*nframes; ii++)
		AllVideoInfo.VideoInfo[ii].valid = false;

	sprintf(Fname, "%s/CamTimingPara.txt", Path);
	FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int cid; double fps, rs_Percent;
		while (fscanf(fp, "%d %lf %lf ", &cid, &fps, &rs_Percent) != EOF)
		{
			if (cid < nVideoViews)
			{
				printLOG("Found timing parameters for cam %d\n", cid);
				videoID = nframes * cid;
				for (int frameID = 0; frameID < nframes; frameID++)
					AllVideoInfo.VideoInfo[frameID + videoID].fps = fps, AllVideoInfo.VideoInfo[frameID + videoID].rollingShutterPercent = rs_Percent;
			}
		}
		fclose(fp);
	}

	int count = 0, validFrame = -1;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		videoID = nframes * viewID;
		sprintf(Fname, "%s/vHIntrinsic_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/avIntrinsic_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printLOG("Cannot find %s...", Fname);
				sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					//printLOG("Cannot find %s...", Fname);
					sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, viewID);
					if (IsFileExist(Fname) == 0)
					{
						printLOG("Cannot find %s...\n", Fname);
						count++;
						continue;
					}
				}
			}
		}
		FILE *fp = fopen(Fname, "r");
		printLOG("Loaded %s\n", Fname);
		double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
		while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &ShutterModel, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
		{
			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
			{
				AllVideoInfo.VideoInfo[frameID + videoID].intrinsic[0] = fx, AllVideoInfo.VideoInfo[frameID + videoID].intrinsic[1] = fy,
					AllVideoInfo.VideoInfo[frameID + videoID].intrinsic[2] = skew, AllVideoInfo.VideoInfo[frameID + videoID].intrinsic[3] = u0, AllVideoInfo.VideoInfo[frameID + videoID].intrinsic[4] = v0;

				GetKFromIntrinsic(AllVideoInfo.VideoInfo[frameID + videoID]);
				mat_invert(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].invK);

				AllVideoInfo.VideoInfo[frameID + videoID].LensModel = LensType, AllVideoInfo.VideoInfo[frameID + videoID].ShutterModel = ShutterModel,
					AllVideoInfo.VideoInfo[frameID + videoID].width = width, AllVideoInfo.VideoInfo[frameID + videoID].height = height,
					AllVideoInfo.VideoInfo[frameID + videoID].threshold = threshold, AllVideoInfo.VideoInfo[frameID + videoID].nInlierThresh = ninliersThresh;
				AllVideoInfo.VideoInfo[frameID + videoID].hasIntrinsicExtrinisc++;
				validFrame = frameID + videoID;
			}

			if (LensType == RADIAL_TANGENTIAL_PRISM)
			{
				fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
				if (frameID >= startTime && frameID <= stopTime)
				{
					AllVideoInfo.VideoInfo[frameID + videoID].distortion[0] = r0, AllVideoInfo.VideoInfo[frameID + videoID].distortion[1] = r1, AllVideoInfo.VideoInfo[frameID + videoID].distortion[2] = r2;
					AllVideoInfo.VideoInfo[frameID + videoID].distortion[3] = t0, AllVideoInfo.VideoInfo[frameID + videoID].distortion[4] = t1;
					AllVideoInfo.VideoInfo[frameID + videoID].distortion[5] = p0, AllVideoInfo.VideoInfo[frameID + videoID].distortion[6] = p1;
				}
			}
			else
			{
				fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
				if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
					AllVideoInfo.VideoInfo[frameID + videoID].distortion[0] = omega, AllVideoInfo.VideoInfo[frameID + videoID].distortion[1] = DistCtrX, AllVideoInfo.VideoInfo[frameID + videoID].distortion[2] = DistCtrY;
			}
			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
				AllVideoInfo.VideoInfo[frameID + videoID].width = width, AllVideoInfo.VideoInfo[frameID + videoID].height = height;
		}
		fclose(fp);
		for (int ii = 0; ii < nframes; ii++)
			AllVideoInfo.VideoInfo[frameID + videoID].width = width, AllVideoInfo.VideoInfo[frameID + videoID].height = height, AllVideoInfo.VideoInfo[frameID + videoID].viewID = viewID;
	}
	if (count == nVideoViews)
		return 1;

	count = 0;
	for (int viewID = 0; viewID < nVideoViews; viewID++)
	{
		videoID = nframes * viewID;
		if (AllVideoInfo.VideoInfo[validFrame].ShutterModel == 0)
		{
			sprintf(Fname, "%s/vHCamPose_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/avCamPose_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					//printLOG("Cannot find %s...", Fname);
					sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, viewID);
					if (IsFileExist(Fname) == 0)
					{
						//printLOG("Cannot find %s...", Fname);
						sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
						if (IsFileExist(Fname) == 0)
						{
							printLOG("Cannot find %s...\n", Fname);
							continue;
						}
					}
				}
			}
		}
		else if (AllVideoInfo.VideoInfo[validFrame].ShutterModel == 1)
		{
			sprintf(Fname, "%s/vHCamPose_RSCayley_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				sprintf(Fname, "%s/avCamPose_RSCayley_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					//printLOG("Cannot find %s...", Fname);
					sprintf(Fname, "%s/vCamPose_RSCayley_%.4d.txt", Path, viewID);
					if (IsFileExist(Fname) == 0)
					{
						//printLOG("Cannot find %s...", Fname);
						sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
						if (IsFileExist(Fname) == 0)
						{
							printLOG("Cannot find %s...\n", Fname);
							continue;
						}
					}
				}
			}
		}
		else
		{
			sprintf(Fname, "%s/CamPose_Spline_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				printLOG("Cannot find %s...\n", Fname);
				continue;
			}
		}
		FILE *fp = fopen(Fname, "r");
		printLOG("Loaded %s\n", Fname);
		double rt[6], wt[6];
		while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &frameID, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]) != EOF)
		{
			if (AllVideoInfo.VideoInfo[validFrame].ShutterModel == 1)
				for (int jj = 0; jj < 6; jj++)
					fscanf(fp, "%lf ", &wt[jj]);

			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
			{
				if (AllVideoInfo.VideoInfo[frameID + videoID].hasIntrinsicExtrinisc < 1 || abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001)
				{
					AllVideoInfo.VideoInfo[frameID + videoID].valid = false;
					continue;
				}

				if (AllVideoInfo.VideoInfo[frameID + videoID].hasIntrinsicExtrinisc > 0)
					AllVideoInfo.VideoInfo[frameID + videoID].valid = true;

				for (int jj = 0; jj < 6; jj++)
					AllVideoInfo.VideoInfo[frameID + videoID].rt[jj] = rt[jj];
				GetRTFromrt(AllVideoInfo.VideoInfo[frameID + videoID]);
				GetCfromT(AllVideoInfo.VideoInfo[frameID + videoID]);

				if (AllVideoInfo.VideoInfo[validFrame].ShutterModel == 1)
					for (int jj = 0; jj < 6; jj++)
						AllVideoInfo.VideoInfo[frameID + videoID].wt[jj] = wt[jj];

				Rotation2Quaternion(AllVideoInfo.VideoInfo[frameID + videoID].R, AllVideoInfo.VideoInfo[frameID + videoID].Quat);

				GetRCGL(AllVideoInfo.VideoInfo[frameID + videoID]);
				AssembleP(AllVideoInfo.VideoInfo[frameID + videoID].K, AllVideoInfo.VideoInfo[frameID + videoID].R, AllVideoInfo.VideoInfo[frameID + videoID].T, AllVideoInfo.VideoInfo[frameID + videoID].P);

				double principal[] = { AllVideoInfo.VideoInfo[frameID + videoID].width / 2, AllVideoInfo.VideoInfo[frameID + videoID].height / 2, 1.0 };
				getRayDir(AllVideoInfo.VideoInfo[frameID + videoID].principleRayDir, AllVideoInfo.VideoInfo[frameID + videoID].invK, AllVideoInfo.VideoInfo[frameID + videoID].R, principal);
			}
		}
		fclose(fp);
	}

	if (count == nVideoViews)
		return 1;

	return 0;
}
int ReadVideoDataI(char *Path, VideoData &vInfo, int viewID, int startTime, int stopTime, double threshold, int ninliersThresh, int silent)
{
	char Fname[512];

	int selected, maxFrameOffset = -9999; double fps;
	sprintf(Fname, "%s/FMotionPriorSync.txt", Path); FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		double temp;
		while (fscanf(fp, "%d %lf %lf ", &selected, &fps, &temp) != EOF)
			maxFrameOffset = max(maxFrameOffset, abs((int)temp));
		fclose(fp);
	}
	else
	{
		sprintf(Fname, "%s/FGeoSync.txt", Path); fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			int temp;
			while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
				maxFrameOffset = max(maxFrameOffset, abs((int)temp));
			fclose(fp);
		}
		else
		{
			sprintf(Fname, "%s/InitSync.txt", Path); fp = fopen(Fname, "r");
			if (fp != NULL)
			{
				int temp;
				while (fscanf(fp, "%d %lf %d ", &selected, &fps, &temp) != EOF)
					maxFrameOffset = max(maxFrameOffset, abs((int)temp));
				fclose(fp);
			}
			else
			{
				maxFrameOffset = 0;
				printLOG("Cannot load time stamp info. Assume no frame offsets!");
			}
		}
	}

	int frameID, LensType, ShutterModel, width, height, nframes;
	if (startTime == -1 && stopTime == -1)
		startTime = 0, stopTime = 50000, maxFrameOffset = 0, nframes = stopTime + 1;
	else
		nframes = stopTime + maxFrameOffset + 1;

	vInfo.nframesI = nframes;
	vInfo.startTime = startTime;
	vInfo.stopTime = stopTime;
	vInfo.VideoInfo = new CameraData[nframes];
	for (int ii = 0; ii < nframes; ii++)
	{
		vInfo.VideoInfo[ii].valid = false;
		vInfo.VideoInfo[ii].threshold = threshold;
	}


	sprintf(Fname, "%s/staticCamList.txt", Path);
	if (IsFileExist(Fname))
	{
		int cid;
		fp = fopen(Fname, "r");
		while (fscanf(fp, "%d ", &cid) != EOF)
			if (cid == viewID)
				vInfo.staticCam = true;
		fclose(fp);
	}

	sprintf(Fname, "%s/CamTimingPara.txt", Path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int cid; double fps, rs_Percent;
		while (fscanf(fp, "%d %lf %lf ", &cid, &fps, &rs_Percent) != EOF)
		{
			if (cid == viewID)
			{
				printLOG("Found timing parameters for cam %d\n", cid);
				for (int ii = 0; ii < nframes; ii++)
					vInfo.VideoInfo[ii].fps = fps, vInfo.VideoInfo[ii].rollingShutterPercent = rs_Percent;
				break;
			}
		}
		fclose(fp);
	}

	//READ INTRINSIC: START
	int validFrame = 0;
	sprintf(Fname, "%s/vHIntrinsic_%.4d.txt", Path, viewID);
	if (IsFileExist(Fname) == 0)
	{
		sprintf(Fname, "%s/avIntrinsic_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			//printLOG("Cannot find %s...", Fname);
			sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printLOG("Cannot find %s...", Fname);
				sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					printLOG("Cannot find %s...\n", Fname);
					return 1;
				}
			}
		}
	}
	fp = fopen(Fname, "r");
	if (silent == 0)
		printLOG("Loaded %s\n", Fname);
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
	while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &ShutterModel, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
		{
			vInfo.VideoInfo[frameID].K[0] = fx, vInfo.VideoInfo[frameID].K[1] = skew, vInfo.VideoInfo[frameID].K[2] = u0,
				vInfo.VideoInfo[frameID].K[3] = 0.0, vInfo.VideoInfo[frameID].K[4] = fy, vInfo.VideoInfo[frameID].K[5] = v0,
				vInfo.VideoInfo[frameID].K[6] = 0.0, vInfo.VideoInfo[frameID].K[7] = 0.0, vInfo.VideoInfo[frameID].K[8] = 1.0;

			vInfo.VideoInfo[frameID].viewID = viewID;
			vInfo.VideoInfo[frameID].frameID = frameID;
			vInfo.VideoInfo[frameID].width = width, vInfo.VideoInfo[frameID].height = height;
			GetIntrinsicFromK(vInfo.VideoInfo[frameID]);
			mat_invert(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].invK);

			vInfo.VideoInfo[frameID].LensModel = LensType, vInfo.VideoInfo[frameID].ShutterModel = ShutterModel, vInfo.VideoInfo[frameID].threshold = threshold, vInfo.VideoInfo[frameID].nInlierThresh = ninliersThresh;
			vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc++;
			validFrame = frameID;
		}

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
			{
				vInfo.VideoInfo[frameID].distortion[0] = r0, vInfo.VideoInfo[frameID].distortion[1] = r1, vInfo.VideoInfo[frameID].distortion[2] = r2;
				vInfo.VideoInfo[frameID].distortion[3] = t0, vInfo.VideoInfo[frameID].distortion[4] = t1;
				vInfo.VideoInfo[frameID].distortion[5] = p0, vInfo.VideoInfo[frameID].distortion[6] = p1;
			}
		}
		else
		{
			fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
				vInfo.VideoInfo[frameID].distortion[0] = omega, vInfo.VideoInfo[frameID].distortion[1] = DistCtrX, vInfo.VideoInfo[frameID].distortion[2] = DistCtrY;
		}
		if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
			vInfo.VideoInfo[frameID].width = width, vInfo.VideoInfo[frameID].height = height;

	}
	fclose(fp);
	for (int ii = 0; ii < nframes; ii++)
		vInfo.VideoInfo[ii].width = width, vInfo.VideoInfo[ii].height = height, vInfo.VideoInfo[ii].viewID = viewID;
	//END

	//READ POSE FROM VIDEO POSE: START
	if (vInfo.VideoInfo[validFrame].ShutterModel <= 1)
	{
		sprintf(Fname, "%s/vHCamPose_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			sprintf(Fname, "%s/avCamPose_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				//printLOG("Cannot find %s...", Fname);
				sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, viewID);
				if (IsFileExist(Fname) == 0)
				{
					//printLOG("Cannot find %s...", Fname);
					sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
					if (IsFileExist(Fname) == 0)
					{
						printLOG("Cannot find %s...\n", Fname);
						return 1;
					}
				}
			}
		}
	}
	else
	{
		sprintf(Fname, "%s/CamPose_Spline_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			printLOG("Cannot find %s...", Fname);
			return 1;
		}
	}
	fp = fopen(Fname, "r");
	if (silent == 0)
		printLOG("Loaded %s\n", Fname);
	double rt[6], wt[6];
	while (fscanf(fp, "%d %lf %lf %lf %lf %lf %lf ", &frameID, &rt[0], &rt[1], &rt[2], &rt[3], &rt[4], &rt[5]) != EOF)
	{
		if (vInfo.VideoInfo[validFrame].ShutterModel == 1)
			for (int jj = 0; jj < 6; jj++)
				fscanf(fp, "%lf ", &wt[jj]);

		if (frameID >= startTime - maxFrameOffset && frameID <= stopTime + maxFrameOffset)
		{
			if (vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc < 1 || abs(rt[3]) + abs(rt[4]) + abs(rt[5]) < 0.001)
			{
				vInfo.VideoInfo[frameID].valid = false;
				continue;
			}

			if (vInfo.VideoInfo[frameID].hasIntrinsicExtrinisc > 0)
				vInfo.VideoInfo[frameID].valid = true;

			for (int jj = 0; jj < 6; jj++)
				vInfo.VideoInfo[frameID].rt[jj] = rt[jj];
			GetRTFromrt(vInfo.VideoInfo[frameID]);
			GetCfromT(vInfo.VideoInfo[frameID]);

			if (vInfo.VideoInfo[frameID].ShutterModel == 1)
				for (int jj = 0; jj < 6; jj++)
					vInfo.VideoInfo[frameID].wt[jj] = wt[jj];

			Rotation2Quaternion(vInfo.VideoInfo[frameID].R, vInfo.VideoInfo[frameID].Quat);

			GetRCGL(vInfo.VideoInfo[frameID]);
			AssembleP(vInfo.VideoInfo[frameID].K, vInfo.VideoInfo[frameID].R, vInfo.VideoInfo[frameID].T, vInfo.VideoInfo[frameID].P);

			double principal[] = { vInfo.VideoInfo[frameID].width / 2, vInfo.VideoInfo[frameID].height / 2, 1.0 };
			getRayDir(vInfo.VideoInfo[frameID].principleRayDir, vInfo.VideoInfo[frameID].invK, vInfo.VideoInfo[frameID].R, principal);
		}
	}
	fclose(fp);
	//READ FROM VIDEO POSE: END

	if (frameID <= stopTime + maxFrameOffset)
	{
		nframes = frameID + 1;
		vInfo.nframesI = nframes;
	}

	if (vInfo.staticCam)
	{
		vector<Point2i> NearestValidFrame;
		for (int frameId = 0; frameId <= stopTime; frameId++)
		{
			for (int validFrameId = 0; validFrameId <= stopTime; validFrameId++)
			{
				if (vInfo.VideoInfo[validFrameId].valid)
				{
					NearestValidFrame.emplace_back(frameId, validFrameId);
					break;
				}
			}
		}

		for (auto p : NearestValidFrame)
		{
			int  targetFrameId = p.x, sourceFrameId = p.y;

			CopyCamereInfo(vInfo.VideoInfo[sourceFrameId], vInfo.VideoInfo[targetFrameId], true);

			GetKFromIntrinsic(vInfo.VideoInfo[targetFrameId]);
			AssembleP(vInfo.VideoInfo[targetFrameId].K, vInfo.VideoInfo[targetFrameId].R, vInfo.VideoInfo[targetFrameId].T, vInfo.VideoInfo[targetFrameId].P);

			mat_invert(vInfo.VideoInfo[targetFrameId].K, vInfo.VideoInfo[targetFrameId].invK);

			double principal[] = { vInfo.VideoInfo[targetFrameId].width / 2, vInfo.VideoInfo[targetFrameId].height / 2, 1.0 };
			getRayDir(vInfo.VideoInfo[targetFrameId].principleRayDir, vInfo.VideoInfo[targetFrameId].invK, vInfo.VideoInfo[targetFrameId].R, principal);
		}
	}
	return 0;
}
int WriteVideoDataI(char *Path, VideoData &VideoInfo, int viewID, int startTime, int stopTime, int level)
{
	//WRITE  INTRINSIC
	int validFrame = -1;
	char Fname[512];
	if (level == 0)
		sprintf(Fname, "%s/avIntrinsic_%.4d.txt", Path, viewID);
	else if (level == 1)
		sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, viewID);
	else if (level == 2)
		sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, viewID);
	FILE *fp = fopen(Fname, "w+");
	for (int fid = startTime; fid <= stopTime; fid++)
	{
		if (!VideoInfo.VideoInfo[fid].valid)
			continue;
		fprintf(fp, "%d %d %d %d %d %.8f %.8f %.8f %.8f %.8f  ", fid, VideoInfo.VideoInfo[fid].LensModel, VideoInfo.VideoInfo[fid].ShutterModel, VideoInfo.VideoInfo[fid].width, VideoInfo.VideoInfo[fid].height,
			VideoInfo.VideoInfo[fid].K[0], VideoInfo.VideoInfo[fid].K[4], VideoInfo.VideoInfo[fid].K[1], VideoInfo.VideoInfo[fid].K[2], VideoInfo.VideoInfo[fid].K[5]);

		if (VideoInfo.VideoInfo[fid].LensModel == RADIAL_TANGENTIAL_PRISM)
			fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f \n", VideoInfo.VideoInfo[fid].distortion[0], VideoInfo.VideoInfo[fid].distortion[1], VideoInfo.VideoInfo[fid].distortion[2],
				VideoInfo.VideoInfo[fid].distortion[3], VideoInfo.VideoInfo[fid].distortion[4], VideoInfo.VideoInfo[fid].distortion[5], VideoInfo.VideoInfo[fid].distortion[6]);
		else
			fprintf(fp, "%.8f %.8f %.8f \n", VideoInfo.VideoInfo[fid].distortion[0], VideoInfo.VideoInfo[fid].distortion[1], VideoInfo.VideoInfo[fid].distortion[2]);
		validFrame = fid;
	}
	fclose(fp);

	//WRITE VIDEO POSE
	if (VideoInfo.VideoInfo[validFrame].ShutterModel == 0 || VideoInfo.VideoInfo[validFrame].ShutterModel == 1)
	{
		if (level == 0)
			sprintf(Fname, "%s/avCamPose_%.4d.txt", Path, viewID);
		if (level == 1)
			sprintf(Fname, "%s/vCamPose_%.4d.txt", Path, viewID);
		if (level == 2)
			sprintf(Fname, "%s/CamPose_%.4d.txt", Path, viewID);
	}
	else
		sprintf(Fname, "%s/CamPose_Spline_%.4d.txt", Path, viewID);
	fp = fopen(Fname, "w+");
	for (int fid = startTime; fid <= stopTime; fid++)
	{
		if (!VideoInfo.VideoInfo[fid].valid)
			continue;

		fprintf(fp, "%d ", fid);
		for (int ii = 0; ii < 6; ii++)
			fprintf(fp, "%.16f ", VideoInfo.VideoInfo[fid].rt[ii]);
		if (VideoInfo.VideoInfo[fid].ShutterModel)
			for (int ii = 0; ii < 6; ii++)
				fprintf(fp, "%.16f ", VideoInfo.VideoInfo[fid].wt[ii]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 0;
}

int readKeyPointJson(char *FnameKpts, vector<Point2f> &vUV, vector<float> &vConf, int nKeyPoints)
{
	vUV.clear(), vConf.clear();
	if (!IsFileExist(FnameKpts))
		return -1;

	int personLocalId;
	vector<int> vpersonLocalId;

	vector<vector<Point2f> >vuv_;
	vector <vector<float> > vconf_;
	FILE *fp = fopen(FnameKpts, "r");
	while (fscanf(fp, "%d ", &personLocalId) != EOF)
	{
		vector<Point2f> vuv(nKeyPoints); vector<float> conf(nKeyPoints);

		for (int ii = 0; ii < nKeyPoints; ii++)
			fscanf(fp, "%f ", &vuv[ii].x);
		for (int ii = 0; ii < nKeyPoints; ii++)
			fscanf(fp, "%f ", &vuv[ii].y);
		for (int ii = 0; ii < nKeyPoints; ii++)
			fscanf(fp, "%f ", &conf[ii]);
		for (int ii = 0; ii < nKeyPoints; ii++)
			fscanf(fp, "%f ", &conf[ii]);

		vuv_.push_back(vuv);
		vconf_.push_back(conf);
		vpersonLocalId.push_back(personLocalId);
	}
	fclose(fp);

	const double overlapKeyPointsthresh = 10;
	std::vector<bool> ToDiscard(vpersonLocalId.size());
	for (size_t ii = 0; ii < vpersonLocalId.size(); ii++)
		ToDiscard[ii] = false;
	for (size_t jj = 0; jj < vpersonLocalId.size(); jj++)
	{
		if (ToDiscard[jj])
			continue;
		for (size_t ii = 0; ii < vpersonLocalId.size(); ii++)
		{
			if (ii == jj)
				continue;
			int cnt = 0;
			for (int kk = 0; kk < nKeyPoints; kk++)
				if (vconf_[jj][kk] != 0.0 && vconf_[ii][kk] != 0.0 && norm(vuv_[jj][kk] - vuv_[ii][kk]) < overlapKeyPointsthresh)
					cnt++;

			if (cnt > nKeyPoints / 3)
			{
				ToDiscard[ii] = true;
				break;
			}
		}
	}
	int nValidPeople = 0;
	for (size_t ii = 0; ii < vpersonLocalId.size(); ii++)
	{
		if (!ToDiscard[ii])
		{
			for (int jj = 0; jj < nKeyPoints; jj++)
			{
				vUV.push_back(vuv_[ii][jj]);
				vConf.push_back(vconf_[ii][jj]);
			}
			nValidPeople++;
		}
	}

	return nValidPeople;
}
int ReadCamCalibInfo(char *Path, char *VideoName, int SeqId, int camId, VideoData &VideoI, int startF, int stopF)
{
	char Fname[512], dummy[512];
	int fid, idummy;
	double q[4], C[3], r[3], R[9];

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

	VideoI.startTime = startF;
	VideoI.stopTime = stopF;
	VideoI.VideoInfo = new CameraData[stopF + maxFrameOffset + 1];

	for (int fid = 0; fid < stopF + maxFrameOffset + 1; fid++)
		VideoI.VideoInfo[fid].valid = 0, VideoI.VideoInfo[fid].frameID = fid;

	int width, height;
	double intrinsic[4], distortion[4];
	sprintf(Fname, "%s/%s/intrinsic_%d.txt", Path, VideoName, camId); fp = fopen(Fname, "r");
	fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %d %d", &intrinsic[0], &intrinsic[1], &intrinsic[2], &intrinsic[3], &distortion[0], &distortion[1], &distortion[2], &distortion[3], &width, &height);
	fclose(fp);

	for (int fid = 0; fid <= stopF + maxFrameOffset; fid++)
		VideoI.VideoInfo[fid].width = width, VideoI.VideoInfo[fid].height = height;

	for (int fid = max(0, startF - maxFrameOffset); fid <= stopF + maxFrameOffset; fid++)
	{
		VideoI.VideoInfo[fid].intrinsic[0] = intrinsic[0], VideoI.VideoInfo[fid].intrinsic[1] = intrinsic[1], VideoI.VideoInfo[fid].intrinsic[2] = 0, VideoI.VideoInfo[fid].intrinsic[3] = intrinsic[2], VideoI.VideoInfo[fid].intrinsic[4] = intrinsic[3];
		for (int ii = 0; ii < 4; ii++)
			VideoI.VideoInfo[fid].distortion[ii] = distortion[ii];

		VideoI.VideoInfo[fid].LensModel = 2; //kb3
		GetKFromIntrinsic(VideoI.VideoInfo[fid]);
		mat_invert(VideoI.VideoInfo[fid].K, VideoI.VideoInfo[fid].invK);
	}

	//sprintf(Fname, "%s/%s/%s_general_%d_slam_cam_poses.csv", Path, VideoName, VideoName, SeqId); fp = fopen(Fname, "r");
	sprintf(Fname, "%s/%s/%s_general_%d_%d.csv", Path, VideoName, VideoName, SeqId, camId); fp = fopen(Fname, "r");
	fscanf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s ", dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
	while (fscanf(fp, "%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf ", &fid, &idummy, &q[1], &q[2], &q[3], &q[0], &C[0], &C[1], &C[2]) != EOF)
	{
		fid = fid - 1;//Kiran first frame is indexed 1 while the extracted frames is indexed 0
		if (fid >= startF - maxFrameOffset && fid <= stopF + maxFrameOffset)
		{
			VideoI.VideoInfo[fid].frameID = fid;
			VideoI.VideoInfo[fid].Quat[0] = q[0], VideoI.VideoInfo[fid].Quat[1] = q[1], VideoI.VideoInfo[fid].Quat[2] = q[2], VideoI.VideoInfo[fid].Quat[3] = q[3];
			VideoI.VideoInfo[fid].camCenter[0] = C[0], VideoI.VideoInfo[fid].camCenter[1] = C[1], VideoI.VideoInfo[fid].camCenter[2] = C[2];

			Quaternion2Rotation(q, VideoI.VideoInfo[fid].invR);
			mat_transpose(VideoI.VideoInfo[fid].invR, VideoI.VideoInfo[fid].R, 3, 3);
			GetTfromC(VideoI.VideoInfo[fid].R, VideoI.VideoInfo[fid].camCenter, VideoI.VideoInfo[fid].T);
			GetrtFromRT(VideoI.VideoInfo[fid].rt, VideoI.VideoInfo[fid].R, VideoI.VideoInfo[fid].T);

			AssembleP(VideoI.VideoInfo[fid].K, VideoI.VideoInfo[fid].R, VideoI.VideoInfo[fid].T, VideoI.VideoInfo[fid].P);
			VideoI.VideoInfo[fid].valid = 1;
		}
		if (fid > stopF + maxFrameOffset)
			break;
	}
	fclose(fp);

	return 0;
}

int MineIntrinsicInfo(char *Path, CameraData &Cam, int viewID, int selectedF)
{
	//mine intrinsic info if available
	char Fname[512];
	sprintf(Fname, "%s/avIntrinsic_%.4d.txt", Path, viewID);
	if (IsFileExist(Fname) == 0)
	{
		//printLOG("Cannot find %s...", Fname);
		sprintf(Fname, "%s/vIntrinsic_%.4d.txt", Path, viewID);
		if (IsFileExist(Fname) == 0)
		{
			//printLOG("Cannot find %s...", Fname);
			sprintf(Fname, "%s/Intrinsic_%.4d.txt", Path, viewID);
			if (IsFileExist(Fname) == 0)
			{
				printLOG("Cannot find %s...", Fname);
				return 0;
			}
		}
	}
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		return 0;

	int found = 0;
	int frameID, LensType, ShutterModel, width, height;
	double fx, fy, skew, u0, v0, r0, r1, r2, t0, t1, p0, p1, omega, DistCtrX, DistCtrY;
	while (fscanf(fp, "%d %d %d %d %d %lf %lf %lf %lf %lf ", &frameID, &LensType, &ShutterModel, &width, &height, &fx, &fy, &skew, &u0, &v0) != EOF)
	{
		if (frameID == selectedF)
		{
			Cam.intrinsic[0] = fx, Cam.intrinsic[1] = fy, Cam.intrinsic[2] = skew, Cam.intrinsic[3] = u0, Cam.intrinsic[4] = v0;

			GetKFromIntrinsic(Cam);
			mat_invert(Cam.K, Cam.invK);

			Cam.LensModel = LensType, Cam.ShutterModel = ShutterModel;
			Cam.valid = true;
			found = 1;
		}

		if (LensType == RADIAL_TANGENTIAL_PRISM)
		{
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &r0, &r1, &r2, &t0, &t1, &p0, &p1);
			if (frameID == selectedF)
			{
				Cam.distortion[0] = r0, Cam.distortion[1] = r1, Cam.distortion[2] = r2;
				Cam.distortion[3] = t0, Cam.distortion[4] = t1;
				Cam.distortion[5] = p0, Cam.distortion[6] = p1;
			}
		}
		else
		{
			fscanf(fp, "%lf %lf %lf ", &omega, &DistCtrX, &DistCtrY);
			if (frameID == selectedF)
				Cam.distortion[0] = omega, Cam.distortion[1] = DistCtrX, Cam.distortion[2] = DistCtrY;
		}
		if (frameID == selectedF)
			Cam.width = width, Cam.height = height;
		if (found == 1)
			break;
	}
	fclose(fp);

	return found;
}

void SaveCurrentSfmGL(char *Path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, Point3i *AllColor, int npts)
{
	char Fname[512];
	for (int ii = 0; ii < AvailViews.size(); ii++)
		GetRCGL(AllViewParas[AvailViews.at(ii)]);

	sprintf(Fname, "%s/DinfoGL.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d ", viewID);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	bool hasColor = false;
	if (AllColor != NULL)
		hasColor = true;

	sprintf(Fname, "%s/nH3dGL.xyz", Path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%d %.16f %.16f %.16f ", ii, All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (hasColor)
			fprintf(fp, "%d %d %d\n", AllColor[ii].x, AllColor[ii].y, AllColor[ii].z);
		else
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void SaveCurrentSfmGL(char *Path, CameraData *AllViewParas, vector<int>&AvailViews, vector<Point3d>&All3D, vector<Point3i>&AllColor)
{
	char Fname[512];
	for (int ii = 0; ii < AvailViews.size(); ii++)
		GetRCGL(AllViewParas[AvailViews.at(ii)]);

	sprintf(Fname, "%s/DinfoGL.txt", Path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d ", viewID);
		if (AllViewParas[viewID].valid)
		{
			for (int jj = 0; jj < 16; jj++)
				fprintf(fp, "%.16f ", AllViewParas[viewID].Rgl[jj]);
			for (int jj = 0; jj < 3; jj++)
				fprintf(fp, "%.16f ", AllViewParas[viewID].camCenter[jj]);
		}
		else
			fprintf(fp, "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0");
		fprintf(fp, "\n");
	}
	fclose(fp);

	bool hasColor = false;
	if (AllColor.size() != 0)
		hasColor = true;

	sprintf(Fname, "%s/3dGL.xyz", Path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < All3D.size(); ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%d %.16f %.16f %.16f ", ii, All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (hasColor)
			fprintf(fp, "%d %d %d\n", AllColor[ii].x, AllColor[ii].y, AllColor[ii].z);
		else
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}

int ReadDomeCalibFile(char *Path, CameraData *AllCamInfo)
{
	const int nHDs = 30, nVGAs = 480, nPanels = 20, nCamsPanel = 24;
	char Fname[512];

	double Quaterunion[4], CamCenter[3], T[3];
	for (int camID = 0; camID < nHDs; camID++)
	{
		sprintf(Fname, "%s/In/Calib/%.2d_%.2d.txt", Path, 00, camID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		for (int kk = 0; kk < 9; kk++)
			fscanf(fp, "%lf ", &AllCamInfo[camID].K[kk]);
		fclose(fp);
		for (int kk = 0; kk < 7; kk++)
			AllCamInfo[camID].distortion[kk] = 0.0;
		AllCamInfo[camID].LensModel = RADIAL_TANGENTIAL_PRISM;

		sprintf(Fname, "%s/In/Calib/%.2d_%.2d_ext.txt", Path, 00, camID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("Cannot load %s\n", Fname);
			return 1;
		}
		fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &Quaterunion[0], &Quaterunion[1], &Quaterunion[2], &Quaterunion[3], &CamCenter[0], &CamCenter[1], &CamCenter[2]);
		fclose(fp);
		ceres::QuaternionToRotation(Quaterunion, AllCamInfo[camID].R);
		mat_mul(AllCamInfo[camID].R, CamCenter, T, 3, 3, 1); //t = -RC
		AllCamInfo[camID].T[0] = -T[0], AllCamInfo[camID].T[1] = -T[1], AllCamInfo[camID].T[2] = -T[2];

		GetIntrinsicFromK(AllCamInfo[camID]);
		GetrtFromRT(AllCamInfo[camID].rt, AllCamInfo[camID].R, AllCamInfo[camID].T);
		AssembleP(AllCamInfo[camID].K, AllCamInfo[camID].R, AllCamInfo[camID].T, AllCamInfo[camID].P);
		GetRCGL(AllCamInfo[camID]);
	}

	for (int jj = 0; jj < nPanels; jj++)
	{
		for (int ii = 0; ii < nCamsPanel; ii++)
		{
			int camID = jj * nCamsPanel + ii + nHDs;

			sprintf(Fname, "%s/In/Calib/%.2d_%.2d.txt", Path, jj + 1, ii + 1); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("Cannot load %s\n", Fname);
				return 1;
			}
			for (int kk = 0; kk < 9; kk++)
				fscanf(fp, "%lf ", &AllCamInfo[camID].K[kk]);
			fclose(fp);
			for (int kk = 0; kk < 7; kk++)
				AllCamInfo[camID].distortion[kk] = 0.0;
			AllCamInfo[camID].LensModel = RADIAL_TANGENTIAL_PRISM;

			sprintf(Fname, "%s/In/Calib/%.2d_%.2d_ext.txt", Path, jj + 1, ii + 1); fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("Cannot load %s\n", Fname);
				return 1;
			}
			fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf ", &Quaterunion[0], &Quaterunion[1], &Quaterunion[2], &Quaterunion[3], &CamCenter[0], &CamCenter[1], &CamCenter[2]);
			fclose(fp);
			ceres::QuaternionToRotation(Quaterunion, AllCamInfo[camID].R);
			mat_mul(AllCamInfo[camID].R, CamCenter, T, 3, 3, 1); //t = -RC
			AllCamInfo[camID].T[0] = -T[0], AllCamInfo[camID].T[1] = -T[1], AllCamInfo[camID].T[2] = -T[2];

			GetIntrinsicFromK(AllCamInfo[camID]);
			GetrtFromRT(AllCamInfo[camID].rt, AllCamInfo[camID].R, AllCamInfo[camID].T);
			AssembleP(AllCamInfo[camID].K, AllCamInfo[camID].R, AllCamInfo[camID].T, AllCamInfo[camID].P);
			GetRCGL(AllCamInfo[camID]);
		}
	}


	sprintf(Fname, "%s/CamPose_%.4d.txt", Path, 0);
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < nHDs + nVGAs; ii++)
	{
		fprintf(fp, "%d ", ii);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllCamInfo[ii].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllCamInfo[ii].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	return 0;
}
int ImportCalibDatafromHanFormat(char *Path, VideoData &AllVideoInfo, int nVGAPanels, int nVGACamsPerPanel, int nHDs)
{
	char Fname[512];
	int offset = 0;

	for (int viewID = 0; viewID < nHDs; viewID++)
	{
		sprintf(Fname, "%s/In/Calib/00_%02d.txt", Path, viewID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("cannot load %s\n", Fname);
			continue;
		}
		//KMatrix load
		for (int j = 0; j < 9; j++)
			fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].K[j]);
		fscanf(fp, "%lf %lf ", &AllVideoInfo.VideoInfo[viewID].distortion[0], &AllVideoInfo.VideoInfo[viewID].distortion[1]);//lens distortion parameter

																															 //RT load
		double Quaterunion[4];
		sprintf(Fname, "%s/In/Calib/00_%02d_ext.txt", Path, viewID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printLOG("cannot load %s\n", Fname);
			return 1;
		}
		for (int j = 0; j < 4; j++)
			fscanf(fp, "%lf ", &Quaterunion[j]);
		for (int j = 0; j < 3; j++)
			fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].camCenter[j]);
		fclose(fp);

		ceres::QuaternionToAngleAxis(Quaterunion, AllVideoInfo.VideoInfo[viewID].rt);
		ceres::QuaternionRotatePoint(Quaterunion, AllVideoInfo.VideoInfo[viewID].camCenter, AllVideoInfo.VideoInfo[viewID].rt + 3);
		for (int j = 0; j < 3; j++) //position to translation t=-R*c
			AllVideoInfo.VideoInfo[viewID].rt[j + 3] = -AllVideoInfo.VideoInfo[viewID].rt[j + 3];

		AllVideoInfo.VideoInfo[viewID].LensModel = VisSFMLens;
		GetIntrinsicFromK(AllVideoInfo.VideoInfo[viewID]);
		GetRTFromrt(AllVideoInfo.VideoInfo[viewID]);
		AssembleP(AllVideoInfo.VideoInfo[viewID]);
	}

	for (int panelID = 0; panelID < nVGAPanels; panelID++)
	{
		for (int camID = 0; camID < nVGACamsPerPanel; camID++)
		{
			int viewID = panelID * nVGACamsPerPanel + camID + nHDs;
			sprintf(Fname, "%s/In/Calib/%02d_%02d.txt", Path, panelID + 1, camID + 1); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("cannot load %s\n", Fname);
				continue;
			}

			//KMatrix load
			for (int j = 0; j < 9; j++)
				fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].K[j]);
			fscanf(fp, "%lf %lf ", &AllVideoInfo.VideoInfo[viewID].distortion[0], &AllVideoInfo.VideoInfo[viewID].distortion[1]);//lens distortion parameter

																																 //RT load
			double Quaterunion[4];
			sprintf(Fname, "%s/In/Calib/%02d_%02d_ext.txt", Path, panelID + 1, camID + 1); fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				printLOG("cannot load %s\n", Fname);
				return 1;
			}
			for (int j = 0; j < 4; j++)
				fscanf(fp, "%lf ", &Quaterunion[j]);
			for (int j = 0; j < 3; j++)
				fscanf(fp, "%lf ", &AllVideoInfo.VideoInfo[viewID].camCenter[j]);
			fclose(fp);

			ceres::QuaternionToAngleAxis(Quaterunion, AllVideoInfo.VideoInfo[viewID].rt);
			ceres::QuaternionRotatePoint(Quaterunion, AllVideoInfo.VideoInfo[viewID].camCenter, AllVideoInfo.VideoInfo[viewID].rt + 3);
			for (int j = 0; j < 3; j++)//position to translation t=-R*c
				AllVideoInfo.VideoInfo[viewID].rt[j + 3] = -AllVideoInfo.VideoInfo[viewID].rt[j + 3];

			AllVideoInfo.VideoInfo[viewID].LensModel = VisSFMLens;
			GetIntrinsicFromK(AllVideoInfo.VideoInfo[viewID]);
			GetRTFromrt(AllVideoInfo.VideoInfo[viewID]);
			AssembleP(AllVideoInfo.VideoInfo[viewID]);
		}
	}

	return 0;
}