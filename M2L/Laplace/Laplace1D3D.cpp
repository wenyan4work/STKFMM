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
#define PI314 (static_cast<double>(3.1415926535897932384626433))

namespace Laplace1D3D {

using EVec3 = Eigen::Vector3d;

inline double gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < 1e-14 ? 0 : 1 / rnorm;
}

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

double directSum(const EVec3 &target, const EVec3 &source,
                 const int directTerm = 500000) {
    // use asymptotic
    const double L3 = 1.0;
    double potentialDirect = 0;
    for (int t = DIRECTLAYER + 1; t < directTerm; t++) {
        potentialDirect += gKernel(target, source + EVec3(t * L3, 0, 0)) +
                           gKernel(target, source - EVec3(t * L3, 0, 0));
    }

    return potentialDirect;
}

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

    const double scaleLEquiv = 1.05;
    const double scaleLCheck = 2.95;
    const double pCenterLEquiv[3] = {
        -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2};
    const double pCenterLCheck[3] = {
        -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2};

    auto pointMEquiv = surface(
        pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(
        pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    auto pointLEquiv = surface(
        pEquiv, (double *)&(pCenterLCheck[0]), scaleLCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointLCheck = surface(
        pCheck, (double *)&(pCenterLEquiv[0]), scaleLEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd M2L(equivN, equivN); // Laplace, 1->1

    Eigen::MatrixXd A(1 * checkN, 1 * equivN);
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            A(k, l) = gKernel(Cpoint, Lpoint);
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        //		std::cout << "debug:" << Mpoint << std::endl;

        // assemble linear system
        Eigen::VectorXd f(checkN);
        for (int k = 0; k < checkN; k++) {
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            f(k) = directSum(Cpoint, Mpoint); // gKernelFF(Cpoint, Mpoint);
        }
        //		std::cout << "debug:" << f << std::endl;

        M2L.col(i) = (ApinvU.transpose() * (ApinvVT.transpose() * f));
    }

    // dump M2L
    for (int i = 0; i < equivN; i++) {
        for (int j = 0; j < equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific
                      << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        chargePoint(4);
    std::vector<double> chargeValue(4);
    chargePoint[0] = Eigen::Vector3d(0.125, 0.5, 0.5);
    chargeValue[0] = 1;
    chargePoint[1] = Eigen::Vector3d(0.375, 0.5, 0.5);
    chargeValue[1] = -1;
    chargePoint[2] = Eigen::Vector3d(0.625, 0.5, 0.5);
    chargeValue[2] = 1;
    chargePoint[3] = Eigen::Vector3d(0.875, 0.5, 0.5);
    chargeValue[3] = -1;

    // solve M
    A.resize(checkN, equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(checkN);
    for (int k = 0; k < checkN; k++) {
        double temp = 0;
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < chargePoint.size(); p++) {
            temp = temp + gKernel(Cpoint, chargePoint[p]) * (chargeValue[p]);
        }
        f(k) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            A(k, l) = gKernel(Mpoint, Cpoint);
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    // impose zero sum
    double fx = 0;
    for (int i = 0; i < equivN; i++) {
        fx += Msource[i];
    }
    fx /= equivN;
    double test = 0;
    for (int i = 0; i < equivN; i++) {
        Msource[i] -= fx;
        test += Msource[i];
    }

    std::cout << "Msource net: " << test << std::endl;
    std::cout << "Msource: " << Msource << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    Eigen::Vector3d samplePoint(0.125, 0.5, 0.5);
    double Usample = 0;
    double UsampleSP = 0;

    for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
        for (size_t p = 0; p < chargePoint.size(); p++) {
            Usample += gKernel(samplePoint, chargePoint[p] + EVec3(k, 0, 0)) *
                       chargeValue[p];
        }
    }

    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint) * M2Lsource[p];
    }

    std::cout << "samplePoint:" << samplePoint << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;
    std::cout << "error:" << UsampleSP + Usample + 8 * log(2) << std::endl;

    samplePoint = EVec3(0.625, 0.5, 0.5);
    Usample = 0;
    UsampleSP = 0;

    for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
        for (size_t p = 0; p < chargePoint.size(); p++) {
            Usample += gKernel(samplePoint, chargePoint[p] + EVec3(k, 0, 0)) *
                       chargeValue[p];
        }
    }

    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint) * M2Lsource[p];
    }

    std::cout << "samplePoint:" << samplePoint << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;
    std::cout << "error:" << UsampleSP + Usample + 8 * log(2) << std::endl;

    return 0;
}

} // namespace Laplace1D3D

#undef DIRECTLAYER
#undef PI314
