/*
Rolling shutter absolute camera pose solver R6P

int r6p(double X[3 * 6], double u[2 * 6], double r0, double(&C)[60], double(&t)[60], double(&v)[60], double(&w)[60])

input	X - 6x3 matrix of 3D points (X == [x y z]')
		u - 6x2 matrix of corresponding 2D image points (u == [r c]')
		r0 - rolling shutter setpoint, linearization point of the model, default 0 - center of the image

output	parameters of the camera projecting the 3D point as 

[r c 1]' = (I+X_((r-r0)*w))*(I+X_(v))*X+C+(r-r0)*t);

where X_(a) produces a shew symmetric matrix such that X_(a)*a= [0 0 0]'

Please, refer to the following paper:

Cenek Albl, Zuzana Kukelova, Tomas Pajdla:
R6P - Rolling Shutter Absolute Camera Pose,
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
June, 2015, Boston, USA.

Copyright (c) 2015, Cenek Albl <alblcene@cmp.felk.cvut.cz> and Zuzana Kukelova <zukuke@microsoft.com> 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the names of the Czech Technical University and Microsoft nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _R6P_H_
#define _R6P_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <list>
#include <iostream>

#include "cref.h"
#include "constants.h"

template <class Scalar>
int solveVW(Scalar g[60], std::vector<Scalar> & v1, std::vector<Scalar> & v2, std::vector<Scalar> & v3, std::vector<Scalar> & w1, std::vector<Scalar> & w2, std::vector<Scalar> & w3){
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> M(196, 216);
	M.setZero();
	int idx;

	for (idx = 0; idx < 33; idx++) {
		M(uv0[idx]) = 1.0;
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv1[idx]) = g[0];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv2[idx]) = g[1];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv3[idx]) = g[2];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv4[idx]) = g[3];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv5[idx]) = g[4];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv6[idx]) = g[5];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv7[idx]) = g[6];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv8[idx]) = g[7];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv9[idx]) = g[8];
	}

	for (idx = 0; idx < 33; idx++) {
		M(uv10[idx]) = g[9];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv11[idx]) = g[10];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv12[idx]) = 1.0;
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv13[idx]) = g[11];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv14[idx]) = g[12];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv15[idx]) = g[13];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv16[idx]) = g[14];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv17[idx]) = g[15];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv18[idx]) = g[16];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv19[idx]) = g[17];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv20[idx]) = g[18];
	}

	for (idx = 0; idx < 36; idx++) {
		M(uv21[idx]) = g[19];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv22[idx]) = g[20];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv23[idx]) = g[21];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv24[idx]) = 1.0;
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv25[idx]) = g[22];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv26[idx]) = g[23];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv27[idx]) = g[24];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv28[idx]) = g[25];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv29[idx]) = g[26];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv30[idx]) = g[27];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv31[idx]) = g[28];
	}

	for (idx = 0; idx < 38; idx++) {
		M(uv32[idx]) = g[29];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv33[idx]) = 1.0;
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv34[idx]) = g[30];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv35[idx]) = g[31];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv36[idx]) = g[32];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv37[idx]) = g[33];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv38[idx]) = g[34];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv39[idx]) = g[35];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv40[idx]) = g[36];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv41[idx]) = g[37];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv42[idx]) = g[38];
	}

	for (idx = 0; idx < 31; idx++) {
		M(uv43[idx]) = g[39];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv44[idx]) = g[40];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv45[idx]) = 1.0;
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv46[idx]) = g[41];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv47[idx]) = g[42];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv48[idx]) = g[43];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv49[idx]) = g[44];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv50[idx]) = g[45];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv51[idx]) = g[46];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv52[idx]) = g[47];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv53[idx]) = g[48];
	}

	for (idx = 0; idx < 30; idx++) {
		M(uv54[idx]) = g[49];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv55[idx]) = g[50];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv56[idx]) = g[51];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv57[idx]) = 1.0;
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv58[idx]) = g[52];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv59[idx]) = g[53];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv60[idx]) = g[54];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv61[idx]) = g[55];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv62[idx]) = g[56];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv63[idx]) = g[57];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv64[idx]) = g[58];
	}

	for (idx = 0; idx < 28; idx++) {
		M(uv65[idx]) = g[59];
	}

	unsigned char basis[20] = { 187U, 191U, 192U, 193U, 197U, 201U,
		202U, 203U, 204U, 205U, 206U, 207U, 208U, 209U, 210U, 211U, 212U, 213U, 214U,
		215U };

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mbasis(196,20);
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mnotbasis(196,196);
	int id = 0;
	for (int i = 0; i < 216; i++)
	{
		if (i == basis[id]){
			Mbasis.col(id) = M.col(i);
			id++;
		}
		else{
			Mnotbasis.col(i - id) = M.col(i);
		}
	}

	Eigen::SparseMatrix<Scalar> Mnotbasis_s = Mnotbasis.sparseView();


	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mr;


	Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int> > sparseLU;
	sparseLU.compute(Mnotbasis_s);
	if (sparseLU.info() == Eigen::Success){
		Mr = sparseLU.solve(Mbasis);
	}
	else{
		return 0;
	}

	Eigen::Matrix<Scalar,20,20> A;
	A.setZero();
	A(60) = 1.0;
	A(181) = 1.0;
	A(282) = 1.0;
	A(303) = 1.0;
	int notbasis_size = 196;
	for (int idx = 0; idx < 20; idx++) {
		A(4 + 20 * idx) = -Mr(192 + notbasis_size * (19 - idx));
		A(5 + 20 * idx) = -Mr(191 + notbasis_size * (19 - idx));
		A(6 + 20 * idx) = -Mr(190 + notbasis_size * (19 - idx));
		A(7 + 20 * idx) = -Mr(185 + notbasis_size * (19 - idx));
		A(8 + 20 * idx) = -Mr(180 + notbasis_size * (19 - idx));
		A(9 + 20 * idx) = -Mr(176 + notbasis_size * (19 - idx));
		A(10 + 20 * idx) = -Mr(175 + notbasis_size * (19 - idx));
		A(11 + 20 * idx) = -Mr(174 + notbasis_size * (19 - idx));
		A(12 + 20 * idx) = -Mr(173 + notbasis_size * (19 - idx));
		A(13 + 20 * idx) = -Mr(165 + notbasis_size * (19 - idx));
		A(14 + 20 * idx) = -Mr(161 + notbasis_size * (19 - idx));
		A(15 + 20 * idx) = -Mr(151 + notbasis_size * (19 - idx));
		A(16 + 20 * idx) = -Mr(147 + notbasis_size * (19 - idx));
		A(17 + 20 * idx) = -Mr(146 + notbasis_size * (19 - idx));
		A(18 + 20 * idx) = -Mr(145 + notbasis_size * (19 - idx));
		A(19 + 20 * idx) = -Mr(140 + notbasis_size * (19 - idx));
	}
	//out << A;
	Eigen::EigenSolver<Eigen::Matrix<Scalar,20,20> > es;
	es.compute(A, true);
	Eigen::Matrix<std::complex<Scalar>,6,20>  V;
	V.row(0) = es.eigenvectors().row(6);
	V.row(1) = es.eigenvectors().row(5);
	V.row(2) = es.eigenvectors().row(4);
	V.row(3) = es.eigenvectors().row(3);
	V.row(4) = es.eigenvectors().row(2);
	V.row(5) = es.eigenvectors().row(1);
	//out << es.eigenvectors();
	Eigen::Matrix<std::complex<Scalar>,6,20> sol;
	sol = V.array() / (Eigen::Matrix<std::complex<Scalar>,6,1 >::Ones()*es.eigenvectors().row(0)).array();
	//out << sol;
	for (int i = 0; i < 20; i++)
	{
		if (sol(0, i).imag() == 0){
			v1.push_back(sol(0, i).real());
			v2.push_back(sol(1, i).real());
			v3.push_back(sol(2, i).real());
			w1.push_back(sol(3, i).real());
			w2.push_back(sol(4, i).real());
			w3.push_back(sol(5, i).real());
		}
	}



	//out.close();
	return 1;
}

template <class Scalar>
int solveVWPlanar(Scalar g[60], std::vector<Scalar> & v1, std::vector<Scalar> & v2, std::vector<Scalar> & v3, std::vector<Scalar> & w1, std::vector<Scalar> & w2, std::vector<Scalar> & w3){
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> M(152, 168);
	M.setZero();
	int ii;

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv0[ii]] = 1.0;
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv1[ii]] = g[0];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv2[ii]] = g[1];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv3[ii]] = g[2];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv4[ii]] = g[3];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv5[ii]] = g[4];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv6[ii]] = g[5];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv7[ii]] = g[6];
	}

	for (ii = 0; ii < 25; ii++) {
		M.data()[iv8[ii]] = g[7];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv9[ii]] = 1.0;
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv10[ii]] = g[8];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv11[ii]] = g[9];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv12[ii]] = g[10];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv13[ii]] = g[11];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv14[ii]] = g[12];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv15[ii]] = g[13];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv16[ii]] = g[14];
	}

	for (ii = 0; ii < 26; ii++) {
		M.data()[iv17[ii]] = g[15];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv18[ii]] = 1.0;
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv19[ii]] = g[16];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv20[ii]] = g[17];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv21[ii]] = g[18];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv22[ii]] = g[19];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv23[ii]] = g[20];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv24[ii]] = g[21];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv25[ii]] = g[22];
	}

	for (ii = 0; ii < 27; ii++) {
		M.data()[iv26[ii]] = g[23];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv27[ii]] = 1.0;
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv28[ii]] = g[24];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv29[ii]] = g[25];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv30[ii]] = g[26];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv31[ii]] = g[27];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv32[ii]] = g[28];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv33[ii]] = g[29];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv34[ii]] = g[30];
	}

	for (ii = 0; ii < 28; ii++) {
		M.data()[iv35[ii]] = g[31];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv36[ii]] = 1.0;
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv37[ii]] = g[32];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv38[ii]] = g[33];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv39[ii]] = g[34];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv40[ii]] = g[35];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv41[ii]] = g[36];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv42[ii]] = g[37];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv43[ii]] = g[38];
	}

	for (ii = 0; ii < 24; ii++) {
		M.data()[iv44[ii]] = g[39];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv45[ii]] = 1.0;
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv46[ii]] = g[40];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv47[ii]] = g[41];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv48[ii]] = g[42];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv49[ii]] = g[43];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv50[ii]] = g[44];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv51[ii]] = g[45];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv52[ii]] = g[46];
	}

	for (ii = 0; ii < 22; ii++) {
		M.data()[iv53[ii]] = g[47];
	}


	unsigned char basis[16] = { 139U, 145U, 154U, 155U, 156U, 157U,
		158U, 159U, 160U, 161U, 162U, 163U, 164U, 165U, 166U, 167U };

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mbasis(152, 16);
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mnotbasis(152, 152);

	int id = 0;
	for (int i = 0; i < 168; i++)
	{
		if (i == basis[id]){
			Mbasis.col(id) = M.col(i);
			id++;
		}
		else{
			Mnotbasis.col(i - id) = M.col(i);
		}
	}

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Mr;

	Eigen::SparseMatrix<Scalar> Mnotbasis_s = Mnotbasis.sparseView();
	Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int> > sparseLU;
	sparseLU.compute(Mnotbasis_s);
	if (sparseLU.info() == Eigen::Success){
		Mr = sparseLU.solve(Mbasis);
	}
	else{
		return 0;
	}

	//Mr = Mnotbasis.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Mbasis);

	//Mr = Mnotbasis.colPivHouseholderQr().solve(Mbasis);

	Eigen::Matrix<Scalar, 16, 16> A;
	A.setZero();
	A(0) = 0;
	A(64) = 1.0;
	A(161) = 1.0;

	for (ii = 0; ii < 16; ii++) {
		A(2 + (ii << 4)) = -Mr(150 + 152 * (15 - ii));
		A(3 + (ii << 4)) = -Mr(146 + 152 * (15 - ii));
	}

	A(228) = 1.0;
	for (ii = 0; ii < 16; ii++) {
		A(5 + (ii << 4)) = -Mr(143 + 152 * (15 - ii));
		A(6 + (ii << 4)) = -Mr(142 + 152 * (15 - ii));
		A(7 + (ii << 4)) = -Mr(136 + 152 * (15 - ii));
		A(8 + (ii << 4)) = -Mr(131 + 152 * (15 - ii));
		A(9 + (ii << 4)) = -Mr(127 + 152 * (15 - ii));
		A(10 + (ii << 4)) = -Mr(124 + 152 * (15 - ii));
		A(11 + (ii << 4)) = -Mr(123 + 152 * (15 - ii));
		A(12 + (ii << 4)) = -Mr(122 + 152 * (15 - ii));
		A(13 + (ii << 4)) = -Mr(116 + 152 * (15 - ii));
		A(14 + (ii << 4)) = -Mr(93 + 152 * (15 - ii));
		A(15 + (ii << 4)) = -Mr(83 + 152 * (15 - ii));
	}

	Eigen::EigenSolver<Eigen::Matrix<Scalar, 16, 16> > es;
	es.compute(A, true);
	Eigen::Matrix<std::complex<Scalar>,6,16>  V;
	V.row(0) = es.eigenvectors().row(6);
	V.row(1) = es.eigenvectors().row(5);
	V.row(2) = es.eigenvectors().row(4);
	V.row(3) = es.eigenvectors().row(3);
	V.row(4) = es.eigenvectors().row(2);
	V.row(5) = es.eigenvectors().row(1);

	Eigen::Matrix<std::complex<Scalar>, 6, 16> sol;
	sol = V.array() / (Eigen::Matrix<std::complex<Scalar>,6,1>::Ones()*es.eigenvectors().row(0)).array();
	for (int i = 0; i < 16; i++)
	{
		if (sol(0, i).imag() == 0){
			v1.push_back(sol(0, i).real());
			v2.push_back(sol(1, i).real());
			v3.push_back(sol(2, i).real());
			w1.push_back(sol(3, i).real());
			w2.push_back(sol(4, i).real());
			w3.push_back(sol(5, i).real());
		}
	}
	return 0;
}

template <class Scalar>
int r6p(Scalar X[3 * 6], Scalar u[2 * 6], Scalar r0, Scalar(&C)[60], Scalar(&t)[60], Scalar(&v)[60], Scalar(&w)[60]){

	Scalar X_1, X_2, X_3, r, c;
	bool planar = false;
	//Check if planar scene
	Eigen::Matrix<Scalar,4,6>  XX;
	XX.topLeftCorner(3, 6) = Eigen::Map<Eigen::Matrix<Scalar, 3, 6> >(X);
	XX.row(3) = Eigen::Matrix<Scalar,1,6>::Ones();
	Eigen::JacobiSVD<Eigen::Matrix<Scalar,4,6> > svd(XX, Eigen::ComputeFullU | Eigen::ComputeFullV);
	if (svd.singularValues()(3) < 1e-8*svd.singularValues()(0)){
		planar = true;
	}

	bool r_zero = false;
	//check if u1 not zero
	for (int i = 0; i < 6; i++)
	{
		if (std::abs(XX(0, i)) <= 1e-10){
			r_zero = true;
		}
	}

	Eigen::Matrix<Scalar,12,22> H;

	for (int i = 0; i < 6; i++)
	{
		X_1 = X[i * 3];
		X_2 = X[i * 3 + 1];
		X_3 = X[i * 3 + 2];
		r = u[i * 2];
		c = u[i * 2 + 1];

		if (r_zero)
			//we use the equation premultiplied by a row of the skew symmetric matrix which contained c
			H.row(i * 2) << 0, -1, c, 0, r0 - r, c*r - c*r0, X_2*r - X_2*r0 - X_3*c*r + X_3*c*r0, 0, 0, X_1*r0 - X_1*r, X_3*c*r0 - X_3*c*r, X_3*r0 - X_3*r, X_1*c*r - X_1*c*r0, X_2*c*r - X_2*c*r0, X_2*r - X_2*r0, X_3 + X_2*c, -X_1*c, -X_1, X_3*r - X_3*r0 + X_2*c*r - X_2*c*r0, X_1*c*r0 - X_1*c*r, X_1*r0 - X_1*r, X_3*c - X_2;
		else{
			//we can use the row premultiplied by r
			H.row(i * 2) << 1, 0, -r, r - r0, 0, -r*r + r0*r, X_3*r*r - X_3*r0*r, X_2*r - X_2*r0, X_3*r - X_3*r0, 0, X_1*r0 - X_1*r + X_3*r*r - X_3*r*r0, 0, -X_1*r*r + X_1*r0*r, -X_2*r*r + X_2*r0*r, X_1*r0 - X_1*r, -X_2*r, X_3 + X_1*r, -X_2, -X_2*r*r + X_2*r0*r, X_3*r - X_3*r0 + X_1*r*r - X_1*r*r0, X_2*r0 - X_2*r, X_1 - X_3*r;
		}
		H.row(i * 2 + 1) << -c, r, 0, c*r0 - c*r, r*r - r0*r, 0, -X_2*r*r + X_2*r0*r, X_2*c*r0 - X_2*c*r, X_3*c*r0 - X_3*c*r, X_1*r*r - X_1*r0*r, X_1*c*r - X_1*c*r0, X_3*r*r - X_3*r0*r, 0, 0, X_1*c*r - X_2*r*r - X_1*c*r0 + X_2*r*r0, -X_3*r, -X_3*c, X_2*c + X_1*r, -X_3*r*r + X_3*r0*r, X_3*c*r0 - X_3*c*r, X_1*r*r + X_2*c*r - X_2*c*r0 - X_1*r*r0, X_2*r - X_1*c;
	}

	Eigen::Matrix<Scalar,22,12> Helim = H.transpose();
	std::list<int> b;
	colEchelonForm(Helim, b);


	Eigen::MatrixXd Helimtr = Helim.transpose();

	
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> g;
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> p;
	std::vector<Scalar> v1, v2, v3, w1, w2, w3;
	if (planar){
		g = Helimtr.block(6, 14, 6, 8).transpose();
		p = Eigen::Map<Eigen::Matrix<Scalar,1,48> >(g.data(), 6 * 8);
		solveVWPlanar(p.data(), v1, v2, v3, w1, w2, w3);
	}
	else{
		g = Helimtr.block(6, 12, 6, 10).transpose();
		p = Eigen::Map<Eigen::Matrix<Scalar, 1, 60> >(g.data(), 6 * 10);
		solveVW(p.data(), v1, v2, v3, w1, w2, w3);
	}

	Eigen::Matrix<Scalar,16,1> x;
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A = Helimtr.block(0, 6, 6, 16);
	Eigen::Matrix<Scalar,6,1> Ct;
	Eigen::Matrix<Scalar,3,1> Cres;

	for (int i = 0; i <(int) v1.size(); i++)
	{
		x << v1[i] * w1[i], v1[i] * w2[i], v1[i] * w3[i], v2[i] * w1[i], v2[i] * w2[i], v2[i] * w3[i], v3[i] * w1[i], v3[i] * w2[i], v3[i] * w3[i], v1[i], v2[i], v3[i], w1[i], w2[i], w3[i], 1;
		Ct = -A*x;
		memcpy(C + i * 3, Ct.segment(0, 3).data(), 3 * sizeof(Scalar));
		memcpy(t + i * 3, Ct.segment(3, 3).data(), 3 * sizeof(Scalar));
		Eigen::Matrix<Scalar,3,1> vr;
		vr << v1[i], v2[i], v3[i];
		memcpy(v + i * 3, vr.data(), 3 * sizeof(Scalar));
		w[i * 3] = w1[i];
		w[i * 3 + 1] = w2[i];
		w[i * 3 + 2] = w3[i];
	}

	return v1.size();
}

#endif