/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>

#define DIRECTLAYER 2

namespace StokesPVel1D3D {

using EVec3 = Eigen::Vector3d;
using EVec4 = Eigen::Vector4d;
using EMat3 = Eigen::Matrix3d;
using EMat4 = Eigen::Matrix4d;

/**
 * \brief Returns the coordinates of points on the surface of a cube.
 * \param[in] p Number of points on an edge of the cube is (n+1)
 * \param[in] c Coordinates to the centre of the cube (3D array).
 * \param[in] alpha Scaling factor for the size of the cube.
 * \param[in] depth Depth of the cube in the octree.
 * \return Vector with coordinates of points on the surface of the cube in the
 * format [x0 y0 z0 x1 y1 z1 .... ].
 */
template <class Real_t>
std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
    size_t n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

    std::vector<Real_t> coord(n_ * 3);
    coord[0] = coord[1] = coord[2] = -1.0;
    size_t cnt = 1;
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = -1.0;
            coord[cnt * 3 + 1] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = (2.0 * j - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * i - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = -1.0;
            coord[cnt * 3 + 2] = (2.0 * (j + 1) - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = (2.0 * j - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = -1.0;
            cnt++;
        }
    for (size_t i = 0; i < (n_ / 2) * 3; i++)
        coord[cnt * 3 + i] = -coord[i];

    Real_t r = 0.5 * pow(0.5, depth);
    Real_t b = alpha * r;
    for (size_t i = 0; i < n_; i++) {
        coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
        coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
        coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
    }
    return coord;
}

// fx,fy,fz,trD -> p, vx,vy,vz
inline void Wkernel(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
    auto rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < 1e-13) {
        answer.setZero();
        return;
    }
    double rnorm3 = rnorm * rnorm * rnorm;

    answer.block<3, 3>(1, 0) = EMat3::Identity() / rnorm;
    answer.block<3, 3>(1, 0) += rst * rst.transpose() / rnorm3;

    answer(0, 0) = rst[0] / rnorm3;
    answer(0, 1) = rst[1] / rnorm3;
    answer(0, 2) = rst[2] / rnorm3;
    answer(0, 3) = 0;
    answer(1, 3) = -rst[0] / rnorm3;
    answer(2, 3) = -rst[1] / rnorm3;
    answer(3, 3) = -rst[2] / rnorm3;
    answer.row(0) *= (1 / (4 * M_PI));
    answer.block<3, 4>(1, 0) *= (1 / (8 * M_PI));
}

inline void WkernelFF(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
    answer.setZero();
    const int imageN = 1000000; // images to sum
    for (int per = DIRECTLAYER + 1; per < imageN; per++) {
        EVec3 perVec(1.0 * per, 0, 0);
        EMat4 W1 = EMat4::Zero();
        EMat4 W2 = EMat4::Zero();
        Wkernel(target, source + perVec, W1);
        Wkernel(target, source - perVec, W2);
        answer += (W1 + W2);
    }
}

// calculate the M2L matrix of images from 2 to 1000
int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {-(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {-(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};
    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    auto pointLEquiv = surface(pEquiv, (double *)&(pCenterCheck[0]), scaleCheck,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointLCheck = surface(pCheck, (double *)&(pCenterEquiv[0]), scaleEquiv,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd A(4 * checkN, 4 * equivN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        EMat4 W;
        W.setZero();
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l], pointLEquiv[3 * l + 1], pointLEquiv[3 * l + 2]);
            Wkernel(Cpoint, Lpoint, W);
            A.block<4, 4>(4 * k, 4 * l) = W;
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

    Eigen::MatrixXd M2L(4 * equivN, 4 * equivN);
#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);
        Eigen::MatrixXd f(4 * checkN, 4);
        for (int k = 0; k < checkN; k++) {
            EMat4 temp = EMat4::Zero();
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
            WkernelFF(Cpoint, Mpoint, temp);
            f.block<4, 4>(4 * k, 0) = temp;
        }
        M2L.block(0, 4 * i, 4 * equivN, 4) = (ApinvU.transpose() * (ApinvVT.transpose() * f));
    }

    // dump M2L
    for (int i = 0; i < 4 * equivN; i++) {
        for (int j = 0; j < 4 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    // Test
    // Sum of force and trD must be zero
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(3);
    std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(3);
    forcePoint[0] = EVec3(0.5, 0.55, 0.2);
    forcePoint[1] = EVec3(0.5, 0.5, 0.5);
    forcePoint[2] = EVec3(0.7, 0.7, 0.7);
    forceValue[0] = EVec4(0.1, 0.2, 0.3, 0.4);
    forceValue[1] = EVec4(-0.1, -0.1, -0.3, -0.4);
    forceValue[2] = EVec4(0, -0.1, 0, 0);

    // solve M
    A.resize(4 * checkN, 4 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(4 * checkN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        EVec4 temp = EVec4::Zero();
        for (int p = 0; p < forceValue.size(); p++) {
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, forcePoint[p], W);
            temp += W * (forceValue[p]);
        }
        f.block<4, 1>(4 * k, 0) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, Mpoint, W);
            A.block<4, 4>(4 * k, 4 * l) = W;
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    std::cout << "Msource: " << Msource << std::endl;
    std::cout << "Msource Sum: " << Msource.sum() << std::endl;
    Eigen::VectorXd M2Lsource = M2L * (Msource);

    {
        EVec3 samplePoint(0.5, 0.2, 0.8);
        // Compute: WFF from L, WFF from WkernelFF
        EVec4 WFFL = EVec4::Zero();
        EVec4 WFFK = EVec4::Zero();

        for (int k = 0; k < equivN; k++) {
            EVec3 Lpoint(pointLEquiv[3 * k], pointLEquiv[3 * k + 1], pointLEquiv[3 * k + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(samplePoint, Lpoint, W);
            WFFL += W * M2Lsource.block<4, 1>(4 * k, 0);
        }

        for (int k = 0; k < forceValue.size(); k++) {
            EMat4 W;
            WkernelFF(samplePoint, forcePoint[k], W);
            WFFK += W * forceValue[k];
        }
        std::cout << "WFF from Lequiv: " << WFFL << std::endl;
        std::cout << "WFF from Kernel: " << WFFK << std::endl;
        std::cout << "FF Error: " << WFFL - WFFK << std::endl;
    }

    return 0;
}

} // namespace StokesPVel1D3D

#undef DIRECTLAYER
