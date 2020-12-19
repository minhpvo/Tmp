#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <random>

#include "perspective_three_point.h"
#include "four_point_focal_length.h"
#include "epnp.h"
#include "dls_pnp.h"

#include "util.h"

using namespace std;
using namespace Eigen;
using Eigen::AngleAxisd;
using Eigen::Map;
using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;

typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
double RandDouble(const double lower, const double upper)
{
	std::uniform_real_distribution<double> distribution(lower, upper);
	return distribution(gen);
}
void AddNoiseToProjection(const double noise_factor, Vector2d* ray)
{
	*ray += Eigen::Vector2d(RandDouble(-noise_factor, noise_factor), RandDouble(-noise_factor, noise_factor));
}
int P4Pf_Test(const double noise, const double reproj_tolerance)
{
	const double focal = 800;

	const double x = -0.10;  // rotation of the view around x axis
	const double y = -0.20;  // rotation of the view around y axis
	const double z = 0.30;   // rotation of the view around z axis

	// Create a ground truth pose.
	Matrix3d Rz, Ry, Rx;
	Rz << cos(z), sin(z), 0,
		-sin(z), cos(z), 0,
		0, 0, 1;
	Ry << cos(y), 0, -sin(y),
		0, 1, 0,
		sin(y), 0, cos(y);
	Rx << 1, 0, 0,
		0, cos(x), sin(x),
		0, -sin(x), cos(x);
	const Matrix3d gt_rotation = Rz * Ry * Rx;
	const Vector3d gt_translation = Vector3d(-0.00950692, 0.0171496, 0.0508743);

	cout << gt_rotation << endl;
	// Create 3D world points that are viable based on the camera intrinsics and extrinsics.
	std::vector<Vector3d> world_points_vector = { Vector3d(-1.0, 0.5, 1.2),
		Vector3d(-0.79, -0.68, 1.9),
		Vector3d(1.42, 1.01, 2.19),
		Vector3d(0.87, -0.49, 0.89),
		Vector3d(0.0, -0.69, -1.09) };

	Map<const Matrix<double, 3, 4> > world_points(world_points_vector[0].data());

	// Camera intrinsics matrix.
	const Matrix3d camera_matrix = Eigen::DiagonalMatrix<double, 3>(focal, focal, 1.0);
	// Create the projection matrix P = K * [R t].
	Matrix<double, 3, 4> gt_projection;
	gt_projection << gt_rotation, gt_translation;
	gt_projection = camera_matrix * gt_projection;

	// Reproject 3D points to get undistorted image points.
	std::vector<Eigen::Vector2d> image_points_vector(4);
	Map<Matrix<double, 2, 4> > image_point(image_points_vector[0].data());
	image_point = (gt_projection * world_points.colwise().homogeneous()).colwise().hnormalized();

	/*FILE *fp = fopen("C:/temp/X.txt", "w+");
	for (int ii = 0; ii < 4; ii++)
		fprintf(fp, "%.16f %.16f %.16f\n", world_points(0, ii), world_points(1, ii), world_points(2, ii));
	fclose(fp);

	fp = fopen("C:/temp/y.txt", "w+");
	for (int ii = 0; ii < 4; ii++)
		fprintf(fp, "%.16f %.16f\n", image_point(0, ii), image_point(1, ii));
	fclose(fp);*/

	// Run P4pf algorithm.
	std::vector<double> focal_length;
	std::vector<Matrix<double, 3, 4> > soln_projection;
	std::vector<Matrix<double, 3, 3> > soln_rotation;
	std::vector<Vector3d > soln_translation;
	int num_solns = FourPointPoseAndFocalLength(image_points_vector, world_points_vector, focal_length, &soln_rotation, &soln_translation);

	double K[9], R[9], T[3], P[12];
	int goodSol;
	bool matched_transform;
	for (int jj = 0; jj < num_solns; jj++)
	{
		matched_transform = true;

		Matrix<double, 3, 4> transformation_matrix;
		transformation_matrix.block<3, 3>(0, 0) = soln_rotation[jj];
		transformation_matrix.col(3) = soln_translation[jj];
		Matrix3d camera_matrix = Eigen::DiagonalMatrix<double, 3>(focal_length[jj], focal_length[jj], 1.0);
		Matrix<double, 3, 4> projection_matrix = camera_matrix * transformation_matrix;
		cout << projection_matrix << endl;

		K[0] = focal_length[jj], K[1] = 0, K[2] = 0, K[3] = 0, K[4] = focal_length[jj], K[5] = 0, K[6] = 0, K[7] = 0, K[8] = 1;
		R[0] = soln_rotation[jj].coeff(0, 0), R[1] = soln_rotation[jj].coeff(0, 1), R[2] = soln_rotation[jj].coeff(0, 2),
			R[3] = soln_rotation[jj].coeff(1, 0), R[4] = soln_rotation[jj].coeff(1, 1), R[5] = soln_rotation[jj].coeff(1, 2),
			R[6] = soln_rotation[jj].coeff(2, 0), R[7] = soln_rotation[jj].coeff(2, 1), R[8] = soln_rotation[jj].coeff(2, 2);
		T[0] = soln_translation[jj].coeff(0), T[1] = soln_translation[jj].coeff(1), T[2] = soln_translation[jj].coeff(2);


		Map < Vector3d > eT(T, 3);
		Map < Matrix < double, 3, 3, RowMajor > > eK(K, 3, 3), eR(R, 3, 3);
		Map < Matrix < double, 3, 4, RowMajor > > eP(P, 3, 4);
		Matrix<double, 3, 4> eRT;
		eRT.block<3, 3>(0, 0) = eR;
		eRT.col(3) = eT;
		eP = eK*eRT;

		// Check that the reprojection error is very small.
		for (int n = 0; n < 4; n++)
		{
			Vector3d reproj_point = projection_matrix * world_points.col(n).homogeneous();
			cout << reproj_point.hnormalized() << endl << endl;

			double x = world_points(0, n), y = world_points(1, n), z = world_points(2, n);
			double numx = P[0] * x + P[1] * y + P[2] * z + P[3];
			double numy = P[4] * x + P[5] * y + P[6] * z + P[7];
			double denum = P[8] * x + P[9] * y + P[10] * z + P[11];
			double errx = numx / denum - image_point(0, n), erry = numy / denum - image_point(1, n);
			if (errx*errx + erry *erry > reproj_tolerance)
			{
				matched_transform = false;
				;// break;
			}
			else
				goodSol = jj;
		}
		if (matched_transform)
			;// break;
	}

	std::cout << focal_length[goodSol] << std::endl;
	std::cout << soln_rotation[goodSol] << std::endl;
	std::cout << soln_translation[goodSol] << std::endl;
	// One of the solutions must have been a valid solution.
	if (matched_transform)
		return 1;
	else
		return 0;
}


void DSPNP_Test()
{
	const std::vector<Vector3d> world_points = { Vector3d(-1.0, 3.0, 3.0),
		Vector3d(1.0, -1.0, 2.0),
		Vector3d(-1.0, 1.0, 2.0),
		Vector3d(2.0, 1.0, 3.0) };
	const Quaterniond expected_rotation = Quaterniond(AngleAxisd(M_PI / 6.0, Vector3d(0.0, 0.0, 1.0)));
	const Vector3d expected_translation(1.0, 1.0, 1.0);
	const double projection_noise_std_dev = 0.01;
	const double max_reprojection_error = 1e-4;
	const double max_rotation_difference = 1e-5;
	const double max_translation_difference = 1e-8;

	const int num_points = world_points.size();

	Matrix3x4d expected_transform;
	expected_transform << expected_rotation.toRotationMatrix(), expected_translation;

	std::vector<Vector2d> feature_points;
	feature_points.reserve(num_points);
	for (int i = 0; i < num_points; i++)
		feature_points.push_back((expected_transform * world_points[i].homogeneous()).eval().hnormalized());

	if (projection_noise_std_dev)
		for (int i = 0; i < num_points; i++)
			AddNoiseToProjection(projection_noise_std_dev, &feature_points[i]);

	// Run DLS PnP.
	std::vector<Quaterniond> soln_rotation;
	std::vector<Vector3d> soln_translation;
	DlsPnp(feature_points, world_points, &soln_rotation, &soln_translation);

	// Check solutions and verify at least one is close to the actual solution.
	const int num_solutions = soln_rotation.size();
	if (num_solutions == 0)
		abort();

	bool matched_transform = false;
	for (int i = 0; i < num_solutions; i++)
	{
		// Check that reprojection errors are small.
		Matrix3x4d soln_transform;
		soln_transform << soln_rotation[i].toRotationMatrix(), soln_translation[i];
		cout << soln_transform;

		for (int j = 0; j < num_points; j++)
		{
			const Vector2d reprojected_point = (soln_transform * world_points[j].homogeneous()).eval().hnormalized();
			const double reprojection_error = (feature_points[j] - reprojected_point).squaredNorm();
			if (reprojection_error > max_reprojection_error)
				abort();
		}

		// Check that the solution is accurate.
		const double rotation_difference = expected_rotation.angularDistance(soln_rotation[i]);
		const bool matched_rotation = (rotation_difference < max_rotation_difference);
		const double translation_difference = (expected_translation - soln_translation[i]).squaredNorm();
		const bool matched_translation = (translation_difference < max_translation_difference);

		if (matched_translation && matched_rotation)
			matched_transform = true;
	}
	//EXPECT_TRUE(matched_transform);
	return;
}
void P3P_Test(const double noise)
{

	// Projection matrix.
	const Matrix3d gt_rotation = (Eigen::AngleAxisd(DegToRad(15.0), Vector3d(1.0, 0.0, 0.0)) *Eigen::AngleAxisd(DegToRad(-10.0), Vector3d(0.0, 1.0, 0.0))).toRotationMatrix();
	const Vector3d gt_translation(0.3, -1.7, 1.15);
	Matrix3x4d projection_mat;
	projection_mat << gt_rotation, gt_translation;

	// Points in the 3D scene.
	const Vector3d kPoints3d[3] = { Vector3d(-0.3001, -0.5840, 1.2271),
		Vector3d(-1.4487, 0.6965, 0.3889),
		Vector3d(-0.7815, 0.7642, 0.1257) };

	// Points in the camera view.
	Vector2d kPoints2d[3];
	for (int i = 0; i < 3; i++)
	{
		kPoints2d[i] = (projection_mat * kPoints3d[i].homogeneous()).eval().hnormalized();
		if (noise)
			AddNoiseToProjection(noise, &kPoints2d[i]);
	}

	std::vector<Matrix3d> rotations;
	std::vector<Vector3d> translations;
	PoseFromThreePoints(kPoints2d, kPoints3d, &rotations, &translations);

	bool matched_transform = false;
	for (int i = 0; i < rotations.size(); ++i)
	{
		// Check that the rotation and translation are close.
		double angular_diff = RadToDeg(Eigen::Quaterniond(rotations[i]).angularDistance(Eigen::Quaterniond(gt_rotation)));
		double trans_diff = ((-gt_rotation * gt_translation) - (-rotations[i] * translations[i])).norm();
		bool rot_match = angular_diff < 1.0;
		bool trans_match = trans_diff < 0.1;
		if (rot_match && trans_match)
		{
			matched_transform = true;

			Matrix3x4d soln_proj;
			soln_proj << rotations[i], translations[i];
			// Check the reprojection error.
			for (int j = 0; j < 3; j++)
			{
				const Vector3d projected_pt = soln_proj * kPoints3d[j].homogeneous();
				//EXPECT_LT((kPoints2d[j] - projected_pt.hnormalized()).norm() * 800.0, 2.0);
				if ((kPoints2d[j] - projected_pt.hnormalized()).norm() * 800.0 > 2.0)
					abort();
			}
		}
	}
	if (matched_transform)
		printf("1\n");
	return;
	//EXPECT_TRUE(matched_transform);
}

const int n =5;
const double noise = 0;
const double uc = 320;
const double vc = 240;
const double fu = 800;
const double fv = 800;

double rand(double min, double max)
{
	return min + (max - min) * double(rand()) / RAND_MAX;
}
void random_pose(double R[3][3], double t[3])
{
	const double range = 1;

	double phi = rand(0, range * 3.14159 * 2);
	double theta = rand(0, range * 3.14159);
	double psi = rand(0, range * 3.14159 * 2);

	R[0][0] = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
	R[0][1] = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
	R[0][2] = sin(psi) * sin(theta);

	R[1][0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
	R[1][1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
	R[1][2] = cos(psi) * sin(theta);

	R[2][0] = sin(theta) * sin(phi);
	R[2][1] = -sin(theta) * cos(phi);
	R[2][2] = cos(theta);

	t[0] = 0.0f;
	t[1] = 0.0f;
	t[2] = 6.0f;
}
void random_point(double & Xw, double & Yw, double & Zw)
{
	double theta = rand(0, 3.14159), phi = rand(0, 2 * 3.14159), R = rand(0, +2);

	Xw = sin(theta) * sin(phi) * R;
	Yw = -sin(theta) * cos(phi) * R;
	Zw = cos(theta) * R;
}
void project_with_noise(double R[3][3], double t[3], double Xw, double Yw, double Zw, double & u, double & v)
{
	double Xc = R[0][0] * Xw + R[0][1] * Yw + R[0][2] * Zw + t[0];
	double Yc = R[1][0] * Xw + R[1][1] * Yw + R[1][2] * Zw + t[1];
	double Zc = R[2][0] * Xw + R[2][1] * Yw + R[2][2] * Zw + t[2];

	double nu = rand(-noise, +noise);
	double nv = rand(-noise, +noise);
	u = uc + fu * Xc / Zc + nu;
	v = vc + fv * Yc / Zc + nv;
}
void EPNP_Test()
{
	epnp PnP;

	srand(time(0));

	PnP.set_internal_parameters(uc, vc, fu, fv);
	PnP.set_maximum_number_of_correspondences(n);

	double R_true[3][3], t_true[3];
	random_pose(R_true, t_true);

	PnP.reset_correspondences();
	for (int i = 0; i < n; i++) {
		double Xw, Yw, Zw, u, v;

		random_point(Xw, Yw, Zw);

		project_with_noise(R_true, t_true, Xw, Yw, Zw, u, v);
		PnP.add_correspondence(Xw, Yw, Zw, u, v);
	}

	double R_est[3][3], t_est[3];
	double err2 = PnP.compute_pose(R_est, t_est);
	double rot_err, transl_err;

	PnP.relative_error(rot_err, transl_err, R_true, t_true, R_est, t_est);
	cout << ">>> Reprojection error: " << err2 << endl;
	cout << ">>> rot_err: " << rot_err << ", transl_err: " << transl_err << endl;
	cout << endl;
	cout << "'True reprojection error':"
		<< PnP.reprojection_error(R_true, t_true) << endl;
	cout << endl;
	cout << "True pose:" << endl;
	PnP.print_pose(R_true, t_true);
	cout << endl;
	cout << "Found pose:" << endl;
	PnP.print_pose(R_est, t_est);

	return;
}

int main()
{
	//P4Pf_Test(0, 0.1);
	//P3P_Test(1.0 / 800.0);
	EPNP_Test();
	//DSPNP_Test();

	return 0;
}



