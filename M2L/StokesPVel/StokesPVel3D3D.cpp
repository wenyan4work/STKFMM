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
using EMat64 = Eigen::Matrix<double, 6, 4>;
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
    Eigen::Matrix3d A = 2 * (xi * exp(-(xi * xi) * (r * r)) / (sqrt(M_PI) * r * r) + erfc(xi * r) / (2 * r * r * r)) *
                            (r * r * Eigen::Matrix3d::Identity() + (rvec * rvec.transpose())) -
                        4 * xi / sqrt(M_PI) * exp(-(xi * xi) * (r * r)) * Eigen::Matrix3d::Identity();
    return A;
}

inline Eigen::Matrix3d BEW(const double xi, const Eigen::Vector3d &kvec) {
    const double k = kvec.norm();
    Eigen::Matrix3d B = 8 * M_PI * (1 + k * k / (4 * (xi * xi))) *
                        ((k * k) * Eigen::Matrix3d::Identity() - (kvec * kvec.transpose())) / (k * k * k * k);
    B *= exp(-k * k / (4 * xi * xi));
    return B;
}

inline void GkernelEwald(const Eigen::Vector3d &rvec_, Eigen::Matrix3d &Gsum) {
    const double xi = 2;
    EVec3 rvec = rvec_;
    rvec[0] = rvec[0] - floor(rvec[0]);
    rvec[1] = rvec[1] - floor(rvec[1]);
    rvec[2] = rvec[2] - floor(rvec[2]);
    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 5;
    if (r < eps) {
        auto Gself = -4 * xi / sqrt(M_PI) * Eigen::Matrix3d::Identity(); // the self term
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
    return -(2. * exp(-r * r * (xi * xi)) * xi) / (sqrt(M_PI) * r) - std::erfc(r * xi) / (r * r);
}

inline void realSum(double xi, const EVec3 &target, const EVec3 &source, EVec3 &v) {

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
inline void LkernelEwald(const EVec3 &target_, const EVec3 &source_, EVec3 &answer) {
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
                Kwave += -kvec * (sin(kvec.dot(rmn)) * exp(-k2 / (4 * xi2)) / k2);
            }
        }
    }

    answer = Kreal + Kwave;
}

// inline double f(double r, double eta) {
//     return std::erfc(sqrt(M_PI / eta) * r) / r;
// }

// inline double fp(double r, double eta) {
//     return -std::erfc(sqrt(M_PI / eta) * r) / (r * r) -
//            2 * exp(-M_PI * r * r / eta) / (r * sqrt(eta));
// }

// inline EVec3 realgradPSum(const double eta, const EVec3 &xn, const EVec3 &xm)
// {
//     EVec3 rmn = xm - xn;
//     double rnorm = rmn.norm();
//     if (rnorm < eps) {
//         return EVec3(0, 0, 0);
//     }
//     return -fp(rnorm, eta) / rnorm * rmn;
// }

// inline EVec3 LGradEwald(const EVec3 &xm, const EVec3 &xn) {
//     const double eta = 1.0; // recommend for box=1 to get machine precision
//     EVec3 target = xm;
//     EVec3 source = xn;
//     target[0] = target[0] - floor(target[0]); // periodic BC
//     target[1] = target[1] - floor(target[1]);
//     target[2] = target[2] - floor(target[2]);
//     source[0] = source[0] - floor(source[0]);
//     source[1] = source[1] - floor(source[1]);
//     source[2] = source[2] - floor(source[2]);

//     // real sum
//     int rLim = 6;
//     EVec3 Kreal(0, 0, 0);
//     for (int i = -rLim; i <= rLim; i++) {
//         for (int j = -rLim; j <= rLim; j++) {
//             for (int k = -rLim; k <= rLim; k++) {
//                 EVec3 rmn = target - source + EVec3(i, j, k);
//                 if (rmn.norm() < eps) {
//                     continue;
//                 }
//                 Kreal += realgradPSum(eta, EVec3(0, 0, 0), rmn);
//             }
//         }
//     }

//     // wave sum
//     int wLim = 6;
//     EVec3 Kwave(0, 0, 0);
//     EVec3 rmn = target - source;

//     for (int i = -wLim; i <= wLim; i++) {
//         for (int j = -wLim; j <= wLim; j++) {
//             for (int k = -wLim; k <= wLim; k++) {
//                 if (i == 0 && j == 0 && k == 0) {
//                     continue;
//                 }
//                 EVec3 kvec = EVec3(i, j, k);
//                 double knorm = kvec.norm();
//                 Kwave += 2 * M_PI * sin(2 * M_PI * kvec.dot(rmn)) *
//                          exp(-eta * M_PI * knorm * knorm) /
//                          (M_PI * knorm * knorm) * kvec;
//             }
//         }
//     }

//     return -(Kreal + Kwave);
// }

// inline EMat3 LKernelGrad(const EVec3 &target, const EVec3 &source) {
//     EVec3 rst = target - source;
//     double rnorm = rst.norm();
//     if (rnorm < eps) {
//         return Eigen::Matrix3d::Zero();
//     } else {
//         return -Eigen::Matrix3d::Identity() / pow(rnorm, 3) +
//                3 * rst * rst.transpose() / pow(rnorm, 5);
//     }
// }

// inline void WGkernel(const EVec3 &target, const EVec3 &source,
//                      Eigen::Matrix<double, 6, 4> &answer) {
//     auto rst = target - source;
//     double rnorm = rst.norm();
//     answer.setZero();
//     if (rnorm < eps) {
//         return;
//     }
//     double rnorm3 = rnorm * rnorm * rnorm;
//     EMat3 grad = LKernelGrad(target, source);
//     EMat3 Gstk = EMat3::Identity() / rnorm;
//     Gstk += rst * rst.transpose() / rnorm3;
//     answer.block<3, 3>(0, 0) = grad;
//     answer.block<3, 3>(3, 0) = Gstk;
//     answer(3, 3) = -rst[0] / rnorm3;
//     answer(4, 3) = -rst[1] / rnorm3;
//     answer(5, 3) = -rst[2] / rnorm3;
//     answer.block<3, 4>(0, 0) *= (1 / (4 * M_PI));
//     answer.block<3, 4>(3, 0) *= (1 / (8 * M_PI));
// }

// inline void WGkernelEwald(const EVec3 &target, const EVec3 &source,
//                           Eigen::Matrix<double, 6, 4> &answer) {
//     answer.setZero();
//     EMat3 Gstk = EMat3::Zero();
//     GkernelEwald(target - source, Gstk);
//     EVec3 L = EVec3::Zero();
//     LkernelEwald(target, source, L);
//     EMat3 LGrad = LGradEwald(target, source);
//     answer.block<3, 3>(0, 0) = LGrad;
//     answer.block<3, 3>(3, 0) = Gstk;
//     answer(3, 3) = L[0];
//     answer(3, 4) = L[1];
//     answer(3, 5) = L[2];
//     answer.block<3, 4>(0, 0) *= (1 / (4 * M_PI));
//     answer.block<3, 4>(3, 0) *= (1 / (8 * M_PI));
// }

// inline EMat64 WGkernelNF(const EVec3 &target, const EVec3 &source,
//                          const int N = DIRECTLAYER) {
//     EMat64 WNF = EMat64::Zero();

//     for (int i = -N; i < N + 1; i++) {
//         for (int j = -N; j < N + 1; j++) {
//             for (int k = -N; k < N + 1; k++) {
//                 EMat64 W = EMat64::Zero();
//                 WGkernel(target, source + EVec3(i, j, k), W);
//                 WNF += W;
//             }
//         }
//     }
//     return WNF;
// }

// inline void WGkernelFF(const EVec3 &target, const EVec3 &source,
//                        EMat64 &answer) {
//     // EMat4 WEwald = EMat4::Zero();
//     // WkernelEwald(target, source, WEwald);
//     // answer = WEwald - WkernelNF(target, source);
//     answer = WGkernelNF(target, source, 5) - WGkernelNF(target, source);
// }

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

inline void WkernelEwald(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
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
    // pressure gradient balancing net force
    EVec3 x = target;
    x[0] = x[0] - floor(x[0]);
    x[1] = x[1] - floor(x[1]);
    x[2] = x[2] - floor(x[2]);
    answer(0, 0) += x[0];
    answer(0, 1) += x[1];
    answer(0, 2) += x[2];
    // dipole balancing net flux
    EVec3 y = source;
    y[0] = y[0] - floor(y[0]);
    y[1] = y[1] - floor(y[1]);
    y[2] = y[2] - floor(y[2]);
    answer(1, 3) += 0.5 * y[0];
    answer(2, 3) += 0.5 * y[1];
    answer(3, 3) += 0.5 * y[2];
}

inline EMat4 WkernelNF(const EVec3 &target, const EVec3 &source, const int N = DIRECTLAYER) {
    EMat4 WNF = EMat4::Zero();

    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                EMat4 W = EMat4::Zero();
                Wkernel(target, source + EVec3(i, j, k), W);
                WNF += W;
            }
        }
    }
    return WNF;
}

inline void WkernelFF(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
    EMat4 WEwald = EMat4::Zero();
    WkernelEwald(target, source, WEwald);
    answer = WEwald - WkernelNF(target, source);
    // answer = WkernelNF(target, source, 5) - WkernelNF(target, source);
}

void readM2L(const std::string &name, Eigen::MatrixXd &mat) {
    FILE *fin = fopen(name.c_str(), "r");
    int size = mat.rows();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int iread, jread;
            double fread;
            fscanf(fin, "%d %d %lf\n", &iread, &jread, &fread);
            if (i != iread || j != jread) {
                printf("read ij error %d %d\n", i, j);
                exit(1);
            }
            mat(i, j) = fread;
        }
    }

    fclose(fin);
}

// calculate the M2L matrix of images from 2 to 1000
int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    std::cout << std::scientific << std::setprecision(18);

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
    const double pCenterEquiv[3] = {-(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {-(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};
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
    Eigen::MatrixXd A;
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    Eigen::MatrixXd M2L(4 * equivN, 4 * equivN);
    Eigen::MatrixXd ML(equivN, equivN);
    Eigen::MatrixXd MS(3 * equivN, 3 * equivN);
    readM2L("M2L_laplace_3D3Dp" + std::to_string(pEquiv), ML);
    readM2L("M2L_stokes_vel_3D3Dp" + std::to_string(pEquiv), MS);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        for (int j = 0; j < equivN; j++) {
            // 4x4 block of M2L
            EMat3 G = MS.block<3, 3>(3 * i, 3 * j);
            double L = ML(i, j);
            M2L.block<4, 4>(4 * i, 4 * j).setZero();
            M2L.block<3, 3>(4 * i, 4 * j) = G;
            M2L(4 * i + 3, 4 * j + 3) = L;
        }
    }

    //     Eigen::MatrixXd A(4 * checkN, 4 * equivN);
    // #pragma omp parallel for
    //     for (int k = 0; k < checkN; k++) {
    //         Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k +
    //         1],
    //                                pointLCheck[3 * k + 2]);
    //         for (int l = 0; l < equivN; l++) {
    //             const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
    //                                          pointLEquiv[3 * l + 1],
    //                                          pointLEquiv[3 * l + 2]);
    //             EMat4 W = EMat4::Zero();
    //             Wkernel(Cpoint, Lpoint, W);
    //             W.row(0).setZero();
    //             A.block<4, 4>(4 * k, 4 * l) = W;
    //         }
    //     }
    //     Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    //     Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    //     pinv(A, ApinvU, ApinvVT);
    //     // std::cout << A << std::endl;
    //     // std::cout << ApinvU << std::endl;
    //     // std::cout << ApinvVT << std::endl;

    //     Eigen::MatrixXd M2L(4 * equivN, 4 * equivN);

    //     // #pragma omp parallel for
    //     for (int i = 0; i < equivN; i++) {
    //         const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 *
    //         i + 1],
    //                                      pointMEquiv[3 * i + 2]);
    //         const EVec3 npoint(0.5, 0.5, 0.5);
    //         Eigen::MatrixXd f(4 * checkN, 4);
    //         for (int k = 0; k < checkN; k++) {
    //             Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k
    //             + 1],
    //                                    pointLCheck[3 * k + 2]);
    //             EMat4 val = EMat4::Zero();
    //             EMat4 neu = EMat4::Zero();
    //             EMat4 temp = EMat4::Zero();
    //             WkernelFF(Cpoint, Mpoint, val);
    //             // WkernelFF(Cpoint, npoint, neu);
    //             // temp = val - neu;
    //             temp = val;
    //             temp.row(0).setZero();
    //             // temp.col(3) = val.col(3) - neu.col(3);
    //             // temp.row(0) = val.row(0) - neu.row(0);
    //             // std::cout << temp << std::endl;
    //             // f.block<3, 4>(3 * k, 0) = temp.block<3, 4>(1, 0);
    //             f.block<4, 4>(4 * k, 0) = temp;
    //         }
    //         M2L.block(0, 4 * i, 4 * equivN, 4) =
    //             (ApinvU.transpose() * (ApinvVT.transpose() * f));
    //         std::cout << "M2L\n" << M2L.block(0, 4 * i, 4 * equivN, 4) <<
    //         std::endl;
    //     }

    // dump M2L
    for (int i = 0; i < 4 * equivN; i++) {
        for (int j = 0; j < 4 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    // A operator
    A.resize(4 * checkN, 4 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, Mpoint, W);
            A.block<4, 4>(4 * k, 4 * l) = W;
        }
    }
    pinv(A, ApinvU, ApinvVT);

    // Test
    // Sum of force and trD must be zero
    // std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(3);
    // std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(3);
    // forcePoint[0] = EVec3(0.2, 0.9, 0.2);
    // forcePoint[1] = EVec3(0.5, 0.5, 0.5);
    // forcePoint[2] = EVec3(0.7, 0.7, 0.7);
    // forceValue[0] = EVec4(0.1, 0.2, 0.3, 0.);
    // forceValue[1] = EVec4(0.1, -0.1, 0.2, 0.);
    // forceValue[2] = EVec4(0.2, 0.1, 0.1, -0.);

    const int nPts = 5;
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(nPts);
    std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(nPts);
    double dsum = 0;
    for (int i = 0; i < nPts; i++) {
        forcePoint[i] = EVec3::Random() * 0.2 + EVec3(0.5, 0.5, 0.5);
        forceValue[i] = EVec4::Random();
        // forceValue[i][3] = 0;
        dsum += forceValue[i][3];
    }
    for (auto &v : forceValue) {
        v[3] -= dsum / nPts;
    }

    // solve M
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
    }
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));

    std::cout << "Msource: " << Msource << std::endl;

    double fx = 0, fy = 0, fz = 0, trd = 0;
    EVec3 dipole = EVec3::Zero();
    for (int i = 0; i < equivN; i++) {
        fx += Msource[4 * i + 0];
        fy += Msource[4 * i + 1];
        fz += Msource[4 * i + 2];
        trd += Msource[4 * i + 3];
        dipole[0] += Msource[4 * i + 3] * pointLEquiv[3 * i];
        dipole[1] += Msource[4 * i + 3] * pointLEquiv[3 * i + 1];
        dipole[2] += Msource[4 * i + 3] * pointLEquiv[3 * i + 2];
    }

    std::cout << "Fx sum: " << fx << std::endl;
    std::cout << "Fy sum: " << fy << std::endl;
    std::cout << "Fz sum: " << fz << std::endl;
    std::cout << "trd sum: " << trd << std::endl;
    std::cout << "dipole sum: " << dipole.transpose() << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    for (int i = 0; i < 5; i++) {
        EVec3 samplePoint = EVec3(0.1, 0, 0) * i + EVec3(0.0, 0.5, 0.5);
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
            EMat4 W = EMat4::Zero();
            WkernelFF(samplePoint, forcePoint[k], W);
            WFFK += W * forceValue[k];
        }
        // WFFL[1] += dipole[0];
        // WFFL[2] += dipole[1];
        // WFFL[3] += dipole[2];
        EVec4 Error = WFFL - WFFK;
        std::cout << "WFF from Lequiv: " << WFFL.transpose() << std::endl;
        std::cout << "WFF from Kernel: " << WFFK.transpose() << std::endl;
        std::cout << "FF Error: " << Error.transpose() << std::endl;
        std::cout << "ratio: " << Error[1] / dipole[0] << " " << Error[2] / dipole[1] << " " << Error[3] / dipole[2]
                  << " " << std::endl;
    }

    return 0;
}

} // namespace StokesPVel3D3D

#undef DIRECTLAYER
