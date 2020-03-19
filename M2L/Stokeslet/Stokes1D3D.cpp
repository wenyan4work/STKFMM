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

namespace Stokes1D3D {

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

inline void Gkernel(const Eigen::Vector3d &target,
                    const Eigen::Vector3d &source, Eigen::Matrix3d &answer) {
    auto rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < 1e-13) {
        answer.setZero();
        return;
    }

    auto part2 = rst * rst.transpose() / (rnorm * rnorm * rnorm);
    auto part1 = Eigen::Matrix3d::Identity() / rnorm;
    answer = part1 + part2;
}

// calculate the M2L matrix of images from 2 to 1000
int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {
        -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {
        -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};
    auto pointMEquiv = surface(
        pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(
        pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    auto pointLEquiv = surface(
        pEquiv, (double *)&(pCenterCheck[0]), scaleCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointLCheck = surface(
        pCheck, (double *)&(pCenterEquiv[0]), scaleEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    //	for (int i = 0; i < pointLEquiv.size() / 3; i++) {
    //		std::cout << pointLEquiv[3 * i] << " " << pointLEquiv[3 * i + 1]
    //<< " " << pointLEquiv[3 * i + 2] << " "
    //				<< std::endl;
    //	}
    //
    //	for (int i = 0; i < pointLCheck.size() / 3; i++) {
    //		std::cout << pointLCheck[3 * i] << " " << pointLCheck[3 * i + 1]
    //<< " " << pointLCheck[3 * i + 2] << " "
    //				<< std::endl;
    //	}

    const int imageN = 500000; // images to sum
    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd A(3 * checkN, 3 * equivN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            Gkernel(Cpoint, Lpoint, G);
            A.block<3, 3>(3 * k, 3 * l) = G;
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

    Eigen::MatrixXd M2L(3 * equivN, 3 * equivN);
#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        Eigen::MatrixXd f(3 * checkN, 3);
        //		std::cout<<"debug:"<<Mpoint<<std::endl;
        for (int k = 0; k < checkN; k++) {
            Eigen::Matrix3d temp = Eigen::Matrix3d::Zero();
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            for (int per = DIRECTLAYER + 1; per < imageN; per++) {
                Eigen::Vector3d perVec(1.0 * per, 0, 0);
                Eigen::Matrix3d G1 = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d G2 = Eigen::Matrix3d::Zero();
                Gkernel(Cpoint, Mpoint + perVec, G1);
                Gkernel(Cpoint, Mpoint - perVec, G2);
                temp = temp + (G1 + G2);
            }
            //			std::cout << temp << std::endl;
            f.block<3, 3>(3 * k, 0) = temp;
        }
        M2L.block(0, 3 * i, 3 * equivN, 3) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f));
    }

    // dump M2L
    for (int i = 0; i < 3 * equivN; i++) {
        for (int j = 0; j < 3 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific
                      << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forcePoint(3);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forceValue(3);
    forcePoint[0] = Eigen::Vector3d(0.5, 0.55, 0.2);
    forceValue[0] = Eigen::Vector3d(-0.2, 0, 0);
    forcePoint[1] = Eigen::Vector3d(0.5, 0.5, 0.5);
    forceValue[1] = Eigen::Vector3d(1, 0, 0);
    forcePoint[2] = Eigen::Vector3d(0.7, 0.7, 0.7);
    forceValue[2] = Eigen::Vector3d(-0.8, 0, 0);

    // solve M
    A.resize(3 * checkN, 3 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(3 * checkN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d temp = Eigen::Vector3d::Zero();
        Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (int p = 0; p < 3; p++) {
            Gkernel(Cpoint, forcePoint[p], G);
            temp = temp + G * (forceValue[p]);
        }
        f.block<3, 1>(3 * k, 0) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            Gkernel(Cpoint, Mpoint, G);
            A.block<3, 3>(3 * k, 3 * l) = G;
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));
    // impose zero sum
    double fx = 0, fy = 0, fz = 0;
    for (int i = 0; i < equivN; i++) {
        fx += Msource[3 * i];
        fy += Msource[3 * i + 1];
        fz += Msource[3 * i + 2];
    }
    std::cout << "fx svd before correction: " << fx << std::endl;
    std::cout << "fy svd before correction: " << fy << std::endl;
    std::cout << "fz svd before correction: " << fz << std::endl;

    fx /= equivN;
    fy /= equivN;
    fz /= equivN;
    for (int i = 0; i < equivN; i++) {
        Msource[3 * i] -= fx;
        Msource[3 * i + 1] -= fy;
        Msource[3 * i + 2] -= fz;
    }

    std::cout << "Msource: " << Msource << std::endl;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forcePointExt(0);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forceValueExt(0);
    for (size_t p = 0; p < forcePoint.size(); p++) {
        for (int k = -DIRECTLAYER; k < DIRECTLAYER + 1; k++) {
            forcePointExt.push_back(Eigen::Vector3d(k, 0, 0) + forcePoint[p]);
            forceValueExt.push_back(forceValue[p]);
        }
    }

    Eigen::VectorXd M2Lsource = M2L * (Msource);
    Eigen::Vector3d samplePoint(0.6, 0.5, 0.5);
    Eigen::Vector3d Usample(0, 0, 0);
    Eigen::Vector3d UsampleSP(0, 0, 0);
    Eigen::Matrix3d G;
    for (size_t p = 0; p < forcePointExt.size(); p++) {
        Gkernel(samplePoint, forcePointExt[p], G);
        Usample += G * (forceValueExt[p]);
    }
    std::cout << "Usample Direct:" << Usample << std::endl;

    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        Eigen::Vector3d Fpoint(M2Lsource[3 * p], M2Lsource[3 * p + 1],
                               M2Lsource[3 * p + 2]);
        Gkernel(samplePoint, Lpoint, G);
        UsampleSP = UsampleSP + G * (Fpoint);
    }

    std::cout << "Usample M2L:" << UsampleSP << std::endl;
    std::cout << "Usample M2L total:" << UsampleSP + Usample << std::endl;

    Eigen::Vector3d UsampleN(0, 0, 0);
    for (size_t p = 0; p < forcePoint.size(); p++) {
        Eigen::Matrix3d G1 = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d G2 = Eigen::Matrix3d::Zero();
        for (int per = DIRECTLAYER + 1; per < imageN; per++) {
            Eigen::Vector3d perVec(1.0 * per, 0, 0);
            Gkernel(samplePoint, forcePoint[p] + perVec, G1);
            Gkernel(samplePoint, forcePoint[p] - perVec, G2);
            UsampleN += (G1 + G2) * (forceValue[p]);
        }
        Gkernel(samplePoint, forcePoint[p], G2);
    }

    std::cout << "Usample N Images:" << UsampleN + Usample << std::endl;
    std::cout << "error" << UsampleSP - UsampleN << std::endl;

    return 0;
}

} // namespace Stokes1D3D

#undef DIRECTLAYER
