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

namespace StokesPVel3D3D {

using EVec3 = Eigen::Vector3d;
using EVec4 = Eigen::Vector4d;
using EMat3 = Eigen::Matrix3d;
using EMat4 = Eigen::Matrix4d;
constexpr double eps = 1e-10;

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

inline Eigen::Matrix3d AEW(const double xi, const Eigen::Vector3d &rvec) {
    const double r = rvec.norm();
    Eigen::Matrix3d A =
        2 *
            (xi * exp(-(xi * xi) * (r * r)) / (sqrt(M_PI) * r * r) +
             erfc(xi * r) / (2 * r * r * r)) *
            (r * r * Eigen::Matrix3d::Identity() + (rvec * rvec.transpose())) -
        4 * xi / sqrt(M_PI) * exp(-(xi * xi) * (r * r)) *
            Eigen::Matrix3d::Identity();
    return A;
}

inline Eigen::Matrix3d BEW(const double xi, const Eigen::Vector3d &kvec) {
    const double k = kvec.norm();
    Eigen::Matrix3d B =
        8 * M_PI * (1 + k * k / (4 * (xi * xi))) *
        ((k * k) * Eigen::Matrix3d::Identity() - (kvec * kvec.transpose())) /
        (k * k * k * k);
    B *= exp(-k * k / (4 * xi * xi));
    return B;
}

inline void GkernelEwald(const Eigen::Vector3d &rvec, Eigen::Matrix3d &Gsum) {
    const double xi = 2;
    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 5;
    if (r < 1e-14) {
        auto Gself =
            -4 * xi / sqrt(M_PI) * Eigen::Matrix3d::Identity(); // the self term
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                for (int k = -N; k < N + 1; k++) {
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }
                    real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, k));
                }
            }
        }
        real += Gself;
    } else {
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                for (int k = -N; k < N + 1; k++) {
                    real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, k));
                }
            }
        }
    }
    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();

    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                Eigen::Vector3d kvec(2 * M_PI * i, 2 * M_PI * j, 2 * M_PI * k);
                if (i == 0 and j == 0 and k == 0) {
                    continue;
                } else {
                    wave = wave + BEW(xi, kvec) * cos(kvec.dot(rvec));
                }
            }
        }
    }
    Gsum = real + wave;
}

inline double freal(double xi, double r) { return std::erfc(xi * r) / r; }

inline double frealp(double xi, double r) {
    return -(2. * exp(-r * r * (xi * xi)) * xi) / (sqrt(M_PI) * r) -
           std::erfc(r * xi) / (r * r);
}

inline void realSum(double xi, const EVec3 &target, const EVec3 &source,
                    EVec3 &v) {

    EVec3 rvec = target - source;
    double rnorm = rvec.norm();
    if (rnorm < eps) {
        v.setZero();
    } else {
        v = (frealp(xi, rnorm) / rnorm) * rvec;
    }
}

inline void Lkernel(const EVec3 &target, const EVec3 &source, EVec3 &answer) {
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
inline void LkernelEwald(const EVec3 &target_, const EVec3 &source_,
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
                realSum(xi, target, source + EVec3(i, j, k), v);
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

// fx,fy,fz,trD -> p, vx,vy,vz
inline void Wkernel(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
    auto rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < eps) {
        answer.setZero();
        return;
    }
    double rnorm3 = rnorm * rnorm * rnorm;

    answer.setZero();
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

inline void WkernelEwald(const EVec3 &target, const EVec3 &source,
                         EMat4 &answer) {
    answer.setZero();
    EMat3 G = EMat3::Zero();
    GkernelEwald(target - source, G);
    EVec3 L = EVec3::Zero();
    LkernelEwald(target, source, L);
    answer.block<3, 3>(1, 0) = G;
    answer(0, 0) = -L[0];
    answer(0, 1) = -L[1];
    answer(0, 2) = -L[2];
    answer(0, 3) = 0;
    answer(1, 3) = L[0];
    answer(2, 3) = L[1];
    answer(3, 3) = L[2];
    answer.row(0) *= (1 / (4 * M_PI));
    answer.block<3, 4>(1, 0) *= (1 / (8 * M_PI));
}

inline void WkernelFF(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
    EMat4 WEwald = EMat4::Zero();
    WkernelEwald(target, source, WEwald);
    // for (int i = -2 * DIRECTLAYER; i < 2 * DIRECTLAYER + 1; i++) {
    //     for (int j = -2 * DIRECTLAYER; j < 2 * DIRECTLAYER + 1; j++) {
    //         for (int k = -2 * DIRECTLAYER; k < 2 * DIRECTLAYER + 1; k++) {
    //             EMat4 W = EMat4::Zero();
    //             Wkernel(target, source + EVec3(i, j, k), W);
    //             WEwald += W;
    //         }
    //     }
    // }

    for (int i = -DIRECTLAYER; i < DIRECTLAYER + 1; i++) {
        for (int j = -DIRECTLAYER; j < DIRECTLAYER + 1; j++) {
            for (int k = -DIRECTLAYER; k < DIRECTLAYER + 1; k++) {
                EMat4 W = EMat4::Zero();
                Wkernel(target, source + EVec3(i, j, k), W);
                WEwald -= W;
            }
        }
    }
    answer = WEwald;
}

// calculate the M2L matrix of images from 2 to 1000
int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    // {
    //     EMat4 G1 = EMat4::Zero(), G2 = EMat4::Zero();
    //     std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(4);
    //     std::vector<double> forceValue(4);
    //     forcePoint[0] = EVec3(0.5, 0.5, 0.5);
    //     forcePoint[1] = EVec3(0.6, 0.6, 0.6);
    //     forcePoint[2] = EVec3(0.7, 0.7, 0.7);
    //     forcePoint[3] = EVec3(0.8, 0.8, 0.8);
    //     forceValue[0] = 1;
    //     forceValue[1] = -1;
    //     forceValue[2] = -1;
    //     forceValue[3] = 1;

    //     EVec3 spoint(0.000001, 0.000001, 0.000001);
    //     EVec3 vE, vD;
    //     vE.setZero();
    //     vD.setZero();
    //     for (int i = 0; i < forceValue.size(); i++) {
    //         EVec3 temp = EVec3::Zero();
    //         LkernelEwald(spoint, forcePoint[i], temp);
    //         vE += temp * forceValue[i];
    //     }
    //     // for (int i = 0; i < forceValue.size(); i++) {
    //     //     EVec3 temp = EVec3::Zero();
    //     //     LkernelEwald(EVec3(1, 1, 1) - spoint, forcePoint[i], temp);
    //     //     vD += temp * forceValue[i];
    //     // }
    //     const int N = 300;
    //     for (int i = -N; i <= N; i++) {
    //         for (int j = -N; j <= N; j++) {
    //             for (int k = -N; k <= N; k++) {
    //                 for (int p = 0; p < forceValue.size(); p++) {
    //                     EVec3 temp = EVec3::Zero();
    //                     Lkernel(spoint, forcePoint[p] + EVec3(i, j, k),
    //                     temp); vD += temp * forceValue[p];
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << std::scientific << std::setprecision(16);
    //     std::cout << vE.transpose() << std::endl;
    //     std::cout << vD.transpose() << std::endl;
    //     std::exit(0);
    // }

    // {
    //     EMat4 G1 = EMat4::Zero(), G2 = EMat4::Zero();
    //     std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(3);
    //     std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(3);
    //     forcePoint[0] = EVec3(0.5, 0.55, 0.2);
    //     forcePoint[1] = EVec3(0.5, 0.5, 0.5);
    //     forcePoint[2] = EVec3(0.7, 0.7, 0.7);
    //     forceValue[0] = EVec4(0.1, 0.2, 0.3, 0.4);
    //     forceValue[1] = EVec4(-0.1, -0.1, -0.3, -0.4);
    //     forceValue[2] = EVec4(0, -0.1, 0, 0);

    //     EVec3 spoint(0.2, 0.3, 0.4);
    //     EVec4 vE, vD;
    //     vE.setZero();
    //     vD.setZero();
    //     for (int i = 0; i < forceValue.size(); i++) {
    //         EMat4 temp = EMat4::Zero();
    //         WkernelEwald(spoint, forcePoint[i], temp);
    //         vE += temp * forceValue[i];
    //     }

    //     const int N = 50;
    //     for (int i = -N; i <= N; i++) {
    //         for (int j = -N; j <= N; j++) {
    //             for (int k = -N; k <= N; k++) {
    //                 for (int p = 0; p < forceValue.size(); p++) {
    //                     EMat4 temp = EMat4::Zero();
    //                     Wkernel(spoint, forcePoint[p] + EVec3(i, j, k),
    //                     temp); vD += temp * forceValue[p];
    //                 }
    //             }
    //         }
    //     }

    //     std::cout << vE.transpose() << std::endl;
    //     std::cout << (vE - vD).transpose() << std::endl;
    //     std::exit(0);
    // }

    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {
        -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {
        -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};
    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1,
                                   // scale 1.05, depth = 0
    auto pointMCheck = surface(pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1,
                                   // scale 1.05, depth = 0

    auto pointLEquiv = surface(pEquiv, (double *)&(pCenterCheck[0]), scaleCheck,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1,
                                   // scale 1.05, depth = 0
    auto pointLCheck = surface(pCheck, (double *)&(pCenterEquiv[0]), scaleEquiv,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1,
                                   // scale 1.05, depth = 0

    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd A(4 * checkN, 4 * equivN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, Lpoint, W);
            A.block<4, 4>(4 * k, 4 * l) = W;
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);
    // std::cout << A << std::endl;
    // std::cout << ApinvU << std::endl;
    // std::cout << ApinvVT << std::endl;

    Eigen::MatrixXd M2L(4 * equivN, 4 * equivN);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        Eigen::MatrixXd f(4 * checkN, 4);
        const EVec3 npoint(0.5, 0.5, 0.5);
        for (int k = 0; k < checkN; k++) {
            EMat4 val = EMat4::Zero();
            EMat4 neu = EMat4::Zero();
            EMat4 temp = EMat4::Zero();
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            WkernelFF(Cpoint, Mpoint, val);
            WkernelFF(Cpoint, npoint, neu);
            temp = val - neu;
            // temp.col(3) = val.col(3) - neu.col(3);
            // temp.row(0) = val.row(0) - neu.row(0);
            // std::cout << temp << std::endl;

            f.block<4, 4>(4 * k, 0) = temp;
        }
        M2L.block(0, 4 * i, 4 * equivN, 4) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f));
        // std::cout << M2L.block(0, 4 * i, 4 * equivN, 4) << std::endl;
    }

    // dump M2L
    for (int i = 0; i < 4 * equivN; i++) {
        for (int j = 0; j < 4 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific
                      << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    // Test
    // Sum of force and trD must be zero
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(3);
    std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(3);
    forcePoint[0] = EVec3(0.5, 0.55, 0.2);
    forcePoint[1] = EVec3(0.5, 0.5, 0.5);
    forcePoint[2] = EVec3(0.7, 0.7, 0.7);
    forceValue[0] = EVec4(0.1, 0.2, -0.3, 0.4);
    forceValue[1] = EVec4(0.1, -0.1, 0.2, -0.2);
    forceValue[2] = EVec4(-0.2, -0.1, 0.1, -0.2);

    // solve M
    A.resize(4 * checkN, 4 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(4 * checkN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        EVec4 temp = EVec4::Zero();
        for (int p = 0; p < forceValue.size(); p++) {
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, forcePoint[p], W);
            temp += W * (forceValue[p]);
        }
        f.block<4, 1>(4 * k, 0) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, Mpoint, W);
            A.block<4, 4>(4 * k, 4 * l) = W;
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    std::cout << "Msource: " << Msource << std::endl;
    std::cout << "Msource Sum: " << Msource.sum() << std::endl;

    double fx = 0, fy = 0, fz = 0, trd = 0;
    for (int i = 0; i < equivN; i++) {
        fx += Msource[4 * i + 0];
        fy += Msource[4 * i + 1];
        fz += Msource[4 * i + 2];
        trd += Msource[4 * i + 3];
    }
    std::cout << fx << std::endl;
    std::cout << fy << std::endl;
    std::cout << fz << std::endl;
    std::cout << trd << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    {
        EVec3 samplePoint(0.5, 0.2, 0.8);
        // Compute: WFF from L, WFF from WkernelFF
        EVec4 WFFL = EVec4::Zero();
        EVec4 WFFK = EVec4::Zero();

        for (int k = 0; k < equivN; k++) {
            EVec3 Lpoint(pointLEquiv[3 * k], pointLEquiv[3 * k + 1],
                         pointLEquiv[3 * k + 2]);
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

    {
        EVec3 samplePoint(0.8, 0.7, 0.6);
        // Compute: WFF from L, WFF from WkernelFF
        EVec4 WFFL = EVec4::Zero();
        EVec4 WFFK = EVec4::Zero();

        for (int k = 0; k < equivN; k++) {
            EVec3 Lpoint(pointLEquiv[3 * k], pointLEquiv[3 * k + 1],
                         pointLEquiv[3 * k + 2]);
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

} // namespace StokesPVel3D3D

#undef DIRECTLAYER
