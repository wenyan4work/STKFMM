/*
 * LaplaceM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#define DIRECTLAYER 2
#define PI314 (static_cast<double>(3.1415926535897932384626433))

namespace Laplace3D3D {

using EVec3 = Eigen::Vector3d;
using EVec4 = Eigen::Vector4d;
constexpr double eps = 1e-10;

// real and wave sum of 2D Laplace kernel Ewald

inline double freal(double xi, double r) { return std::erfc(xi * r) / r; }

inline double frealp(double xi, double r) {
    return -(2. * exp(-r * r * (xi * xi)) * xi) / (sqrt(M_PI) * r) -
           std::erfc(r * xi) / (r * r);
}

// xm: target, xn: source
inline double realSum(const double xi, const EVec3 &xn, const EVec3 &xm) {
    EVec3 rmn = xm - xn;
    double rnorm = rmn.norm();
    if (rnorm < eps) {
        return 0;
    }
    return freal(xi, rnorm);
}

inline double potEwald(const EVec3 &xm, const EVec3 &xn) {
    const double xi = 2; // recommend for box=1 to get machine precision
    EVec3 target = xm;
    EVec3 source = xn;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    target[2] = target[2] - floor(target[2]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);
    source[2] = source[2] - floor(source[2]);

    // real sum
    int rLim = 4;
    double Kreal = 0;
    for (int i = -rLim; i <= rLim; i++) {
        for (int j = -rLim; j <= rLim; j++) {
            for (int k = -rLim; k <= rLim; k++) {
                Kreal += realSum(xi, target, source - EVec3(i, j, k));
            }
        }
    }

    // wave sum
    int wLim = 4;
    double Kwave = 0;
    EVec3 rmn = target - source;
    const double xi2 = xi * xi;
    const double rmnnorm = rmn.norm();
    for (int i = -wLim; i <= wLim; i++) {
        for (int j = -wLim; j <= wLim; j++) {
            for (int k = -wLim; k <= wLim; k++) {
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                EVec3 kvec = EVec3(i, j, k) * (2 * PI314);
                double k2 = kvec.dot(kvec);
                Kwave +=
                    4 * PI314 * cos(kvec.dot(rmn)) * exp(-k2 / (4 * xi2)) / k2;
            }
        }
    }

    double Kself = rmnnorm < 1e-10 ? -2 * xi / sqrt(PI314) : 0;

    return Kreal + Kwave + Kself - PI314 / xi2;
}

inline void realGradSum(double xi, const EVec3 &target, const EVec3 &source,
                        EVec3 &v) {
    EVec3 rvec = target - source;
    double rnorm = rvec.norm();
    if (rnorm < eps) {
        v.setZero();
    } else {
        v = (frealp(xi, rnorm) / rnorm) * rvec;
    }
}

inline void gradkernel(const EVec3 &target, const EVec3 &source,
                       EVec3 &answer) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < eps) {
        answer.setZero();
        return;
    }
    double rnorm3 = rnorm * rnorm * rnorm;
    answer = -rst / rnorm3;
}

// grad of Laplace potential, without 1/4pi prefactor, periodic of -r_k/r^3
inline void gradEwald(const EVec3 &target_, const EVec3 &source_,
                      EVec3 &answer) {
    EVec3 target = target_;
    EVec3 source = source_;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    target[2] = target[2] - floor(target[2]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);
    source[2] = source[2] - floor(source[2]);

    double xi = 0.54;

    // real sum
    int rLim = 10;
    EVec3 Kreal = EVec3::Zero();
    for (int i = -rLim; i < rLim + 1; i++) {
        for (int j = -rLim; j < rLim + 1; j++) {
            for (int k = -rLim; k < rLim + 1; k++) {
                EVec3 v = EVec3::Zero();
                realGradSum(xi, target, source + EVec3(i, j, k), v);
                Kreal += v;
            }
        }
    }

    // wave sum
    int wLim = 10;
    EVec3 rmn = target - source;
    double xi2 = xi * xi;
    EVec3 Kwave(0., 0., 0.);
    for (int i = -wLim; i < wLim + 1; i++) {
        for (int j = -wLim; j < wLim + 1; j++) {
            for (int k = -wLim; k < wLim + 1; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                EVec3 kvec = EVec3(i, j, k) * (2 * M_PI);
                double k2 = kvec.dot(kvec);
                double knorm = kvec.norm();
                Kwave +=
                    -kvec * (sin(kvec.dot(rmn)) * exp(-k2 / (4 * xi2)) / k2);
            }
        }
    }

    answer = Kreal + Kwave;
}

inline double pot(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < eps ? 0 : 1 / rnorm;
}

inline EVec4 gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    EVec4 pgrad = EVec4::Zero();
    double rnorm = rst.norm();
    if (rnorm < eps) {
        pgrad.setZero();
    } else {
        pgrad[0] = 1 / rnorm;
        double rnorm3 = rnorm * rnorm * rnorm;
        pgrad.block<3, 1>(1, 0) = -rst / rnorm3;
    }
    return pgrad;
}

inline EVec4 gKernelEwald(const EVec3 &target, const EVec3 &source) {
    EVec4 pgrad = EVec4::Zero();
    pgrad[0] = potEwald(target, source);
    EVec3 grad;
    gradEwald(target, source, grad);
    pgrad[1] = grad[0];
    pgrad[2] = grad[1];
    pgrad[3] = grad[2];
    return pgrad;
}

inline EVec4 gKernelNF(const EVec3 &target, const EVec3 &source,
                       int N = DIRECTLAYER) {
    EVec4 gNF = EVec4::Zero();
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                EVec4 gFree = gKernel(target, source + EVec3(i, j, k));
                gNF += gFree;
            }
        }
    }
    return gNF;
}

// Out of Direct Sum Layer, far field part
inline EVec4 gKernelFF(const EVec3 &target, const EVec3 &source) {
    EVec4 fEwald = gKernelEwald(target, source);
    fEwald -= gKernelNF(target, source);
    return fEwald;
}

inline double potNF(const EVec3 &target, const EVec3 &source,
                    int N = DIRECTLAYER) {
    double gNF = 0;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                gNF += pot(target, source + EVec3(i, j, k));
            }
        }
    }
    return gNF;
}

// Out of Direct Sum Layer, far field part
inline double potFF(const EVec3 &target, const EVec3 &source) {
    double fEwald = potEwald(target, source);
    fEwald -= potNF(target, source);
    return fEwald;
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

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    std::cout << std::scientific << std::setprecision(18);

    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
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

    Eigen::MatrixXd A(checkN, 1 * equivN);
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            // A.block<4, 1>(4 * k, l) = gKernel(Cpoint, Lpoint);
            A(k, l) = pot(Cpoint, Lpoint);
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        const EVec3 Npoint(0.5, 0.5, 0.5);
        // assemble linear system
        Eigen::VectorXd f(checkN * 4);
        for (int k = 0; k < checkN; k++) {
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            // sum the images
            // f.block<4, 1>(4 * k, 0) =
            //     gKernelFF(Cpoint, Mpoint) - gKernelFF(Cpoint, Npoint);
            f[k] = potFF(Cpoint, Mpoint) - potFF(Cpoint, Npoint);
        }

        M2L.col(i) = (ApinvU.transpose() * (ApinvVT.transpose() * f));

        // std::cout << "debug f: \n" << f << std::endl;
        // std::cout << "debug M: \n" << M2L.col(i) << std::endl;
    }
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // dump M2L
    for (int i = 0; i < equivN; i++) {
        for (int j = 0; j < equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific
                      << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    // operator A
    A.resize(checkN, equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            A(k, l) = pot(Cpoint, Mpoint);
        }
    }
    pinv(A, ApinvU, ApinvVT);

    // Test
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        chargePoint(2);
    std::vector<double> chargeValue(2);
    chargePoint[0] = Eigen::Vector3d(0.6, 0.6, 0.6);
    chargeValue[0] = -1;
    chargePoint[1] = Eigen::Vector3d(0.1, 0.1, 0.1);
    chargeValue[1] = 1;

    // solve M
    Eigen::VectorXd f(checkN);
    for (int k = 0; k < checkN; k++) {
        double temp = 0;
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < chargePoint.size(); p++) {
            temp = temp + pot(Cpoint, chargePoint[p]) * (chargeValue[p]);
        }
        f[k] = temp;
    }
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    std::cout << "Msource: " << Msource << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    for (int is = 0; is < 5; is++) {

        Eigen::Vector3d samplePoint =
            EVec3::Random() * 0.2 + EVec3(0.5, 0.5, 0.5);
        EVec4 Usample = EVec4::Zero();
        EVec4 UsampleSP = EVec4::Zero();
        EVec4 UFF = EVec4::Zero();

        for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
            for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
                for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
                    for (size_t p = 0; p < chargePoint.size(); p++) {
                        Usample += gKernel(samplePoint,
                                           chargePoint[p] + EVec3(i, j, k)) *
                                   chargeValue[p];
                    }
                }
            }
        }

        for (int p = 0; p < chargePoint.size(); p++) {
            UFF += gKernelFF(samplePoint, chargePoint[p]) * chargeValue[p];
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
        std::cout << "Error : " << UsampleSP - UFF << std::endl;
    }

    return 0;
}

} // namespace Laplace3D3D

#undef DIRECTLAYER
#undef PI314
