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

	FILE *fp = fopen("C:/temp/X.txt", "w+");
	for (int ii = 0; ii < 4; ii++)
		fprintf(fp, "%.16f %.16f %.16f\n", world_points(0, ii), world_points(1, ii), world_points(2, ii));
	fclose(fp);

	fp = fopen("C:/temp/y.txt", "w+");
	for (int ii = 0; ii < 4; ii++)
		fprintf(fp, "%.16f %.16f\n", image_point(0, ii), image_point(1, ii));
	fclose(fp);

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

		AssembleP(K, R, T, P);

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