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

namespace StokesPVel2D3D {

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

inline double lbda(double k, double xi, double z) {
    return exp(-k * k / (4 * xi * xi) - (xi * xi) * (z * z));
}

inline double thetaplus(double k, double xi, double z) {
    return exp(k * z) * std::erfc(k / (2 * xi) + xi * z);
}

inline double thetaminus(double k, double xi, double z) {
    return exp(-k * z) * std::erfc(k / (2 * xi) - xi * z);
}

inline double J00(double k, double xi, double z) {
    return sqrt(M_PI) * lbda(k, xi, z) * xi;
}

inline double J10(double k, double xi, double z) {
    return M_PI * (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (4 * k);
}

inline double J20(double k, double xi, double z) {
    return sqrt(M_PI) * lbda(k, xi, z) / (4 * k * k * xi) +
           M_PI *
               ((thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (8 * k * k * k) +
                (thetaminus(k, xi, z) - thetaplus(k, xi, z)) * z / (8 * k * k) -
                (thetaplus(k, xi, z) + thetaminus(k, xi, z)) /
                    (16 * k * (xi * xi)));
}

inline double J12(double k, double xi, double z) {
    return M_PI * (-thetaplus(k, xi, z) - thetaminus(k, xi, z)) * k / 4 +
           sqrt(M_PI) * lbda(k, xi, z) * xi;
}

inline double J22(double k, double xi, double z) {
    return M_PI * ((thetaplus(k, xi, z) + thetaminus(k, xi, z)) * k /
                       (16 * xi * xi) +
                   (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (8 * k) +
                   (thetaplus(k, xi, z) - thetaminus(k, xi, z)) * z / 8) -
           sqrt(M_PI) * lbda(k, xi, z) / (4 * xi);
}

inline double K11(double k, double xi, double z) {
    return M_PI * ((thetaminus(k, xi, z) - thetaplus(k, xi, z))) / 4;
}

inline double K12(double k, double xi, double z) {
    return M_PI *
           ((thetaplus(k, xi, z) - thetaminus(k, xi, z)) / (16 * xi * xi) +
            (thetaminus(k, xi, z) + thetaplus(k, xi, z)) * z / (8 * k));
}

inline void QI(const Eigen::Vector3d &kvec, double xi, double z,
               Eigen::Matrix3d &QI) {
    // 3*3 tensor
    // kvec: np.array([k1,k2,0])
    double knorm = sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1]);
    QI = 2 * (J00(knorm, xi, z) / (4 * xi * xi) + J10(knorm, xi, z)) *
         Eigen::Matrix3d::Identity();
}

inline void Qkk(const Eigen::Vector3d &kvec, double xi, double z,
                Eigen::Matrix3d &Qreal, Eigen::Matrix3d &Qimg) {
    double k1 = kvec[0];
    double k2 = kvec[1];
    double knorm = sqrt(k1 * k1 + k2 * k2);
    auto j10 = J10(knorm, xi, z);
    auto j20 = J20(knorm, xi, z);
    auto j12 = J12(knorm, xi, z);
    auto j22 = J22(knorm, xi, z);

    auto k11 = K11(knorm, xi, z);
    auto k12 = K12(knorm, xi, z);
    Qreal.setZero();
    Qreal(0, 0) = k1 * k1;
    Qreal(1, 1) = k2 * k2;
    Qreal(0, 1) = k1 * k2;
    Qreal(1, 0) = k1 * k2;

    Qreal *= (j10 / (4 * (xi * xi)) + j20);
    Qreal(2, 2) = (j12 / (4 * xi * xi) + j22);
    Qreal *= -2;

    Qimg.setZero();
    Qimg(0, 2) = k1;
    Qimg(1, 2) = k2;
    Qimg(2, 0) = k1;
    Qimg(2, 1) = k2;
    // Qimg=np.array([[0,0,k1],[0,0,k2],[k1,k2,0]])*( k11/(4*xi**2) + k12 )
    Qimg *= (k11 / (4 * xi * xi) + k12);
    Qimg *= -2;
}

// without 1/8pi prefactor
inline void GkernelEwald(const Eigen::Vector3d &rvecIn, Eigen::Matrix3d &Gsum) {
    const double xi = 2;
    Eigen::Vector3d rvec = rvecIn;
    rvec[0] = rvec[0] - floor(rvec[0]);
    rvec[1] = rvec[1] - floor(rvec[1]); // reset to a periodic cell

    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 5;
    if (r < eps) {
        auto Gself =
            -4 * xi / sqrt(M_PI) * Eigen::Matrix3d::Identity(); // the self term
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, 0));
            }
        }
        real += Gself;
    } else {
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, 0));
            }
        }
    }

    // k
    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();

    double zmn = rvec[2];
    Eigen::Vector3d rhomn = rvec;
    rhomn[2] = 0;
    Eigen::Matrix3d Qreal;
    Eigen::Matrix3d Qimg;
    Eigen::Matrix3d QImat;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            Eigen::Vector3d kvec(2 * M_PI * i, 2 * M_PI * j, 0);
            if (i == 0 and j == 0) {
                continue;
            }
            Qkk(kvec, xi, zmn, Qreal, Qimg);
            QI(kvec, xi, zmn, QImat);
            wave = wave + (QImat + Qreal) * cos(kvec.dot(rhomn)) -
                   (Qimg)*sin(kvec.dot(rhomn));
        }
    }
    wave *= 4;

    // k=0
    Eigen::Matrix3d waveK0;
    waveK0.setZero();
    /*
     *   I2fn=force
     I2fn[2]=0
     wavek0=-(4/1)*(np.pi*(zmn)*ss.erf(zmn*xi)+np.sqrt(np.pi)/(2*xi)*np.exp(-zmn**2*xi**2))*I2fn
     *
     * */
    waveK0 = -(4 / 1.0) *
             (M_PI * (zmn)*std::erf(zmn * xi) +
              sqrt(M_PI) / (2 * xi) * exp(-zmn * zmn * xi * xi)) *
             Eigen::Matrix3d::Identity();
    waveK0(2, 2) = 0;

    Gsum = real + wave + waveK0;
}

inline double freal(double xi, double r) { return std::erfc(xi * r) / r; }

inline double frealp(double xi, double r) {
    return -(2. * exp(-r * r * (xi * xi)) * xi) / (sqrt(M_PI) * r) -
           std::erfc(r * xi) / (r * r);
}

inline double gxkz(double xi, double k, double z) {
    return exp(k * z) * std::erfc(xi * z + k / (2 * xi)) +
           exp(-k * z) * std::erfc(-xi * z + k / (2 * xi));
}

inline double gxkzp(double xi, double k, double z) {
    double pisqrt = sqrt(M_PI);
    return k * exp(k * z) * std::erfc(xi * z + k / (2 * xi)) -
           k * exp(-k * z) * std::erfc(-xi * z + k / (2 * xi)) +
           (2 * xi / pisqrt) * (exp(-pow((-xi * z + k / (2 * xi)), 2) - k * z) -
                                exp(-pow((xi * z + k / (2 * xi)), 2) + k * z));
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
    answer = rst / rnorm3;
}

// grad of Laplace potential, without 1/4pi prefactor, periodic of -r_k/r^3
inline void LkernelEwald(const EVec3 &target_, const EVec3 &source_,
                         EVec3 &answer) {
    EVec3 target = target_;
    EVec3 source = source_;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);

    double xi = 2;
    //  real sum
    int rLim = 4;
    EVec3 Kreal = EVec3::Zero();
    for (int i = -rLim; i < rLim + 1; i++) {
        for (int j = -rLim; j < rLim + 1; j++) {
            EVec3 v = EVec3::Zero();
            realSum(xi, target, source - EVec3(i, j, 0), v);
            Kreal += v;
        }
    }

    //  wave sum
    using EVec2 = Eigen::Vector2d;
    int wLim = 4;
    EVec3 rmn = target - source;
    // double xi2 = xi * xi;
    // double rmnnorm = rmn.norm();
    EVec2 rxy(rmn[0], rmn[1]);
    double rz = rmn[2];
    EVec3 Kwave(0, 0, -2 * M_PI * std::erf(xi * rz));
    for (int i = -wLim; i < wLim + 1; i++) {
        for (int j = -wLim; j < wLim + 1; j++) {
            if (i == 0 && j == 0)
                continue;
            EVec2 kvec = EVec2(i, j) * (2 * M_PI);

            double k2 = kvec.dot(kvec);
            double knorm = sqrt(k2);
            double xyfac =
                -M_PI * sin(kvec.dot(rxy)) * gxkz(xi, knorm, rz) / knorm;
            Kwave[0] += xyfac * kvec[0];
            Kwave[1] += xyfac * kvec[1];
            Kwave[2] +=
                M_PI * cos(kvec.dot(rxy)) * gxkzp(xi, knorm, rz) / knorm;
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
    //         EMat4 W = EMat4::Zero();
    //         Wkernel(target, source + EVec3(i, j, 0), W);
    //         WEwald += W;
    //     }
    // }

    for (int i = -DIRECTLAYER; i < DIRECTLAYER + 1; i++) {
        for (int j = -DIRECTLAYER; j < DIRECTLAYER + 1; j++) {
            EMat4 W = EMat4::Zero();
            Wkernel(target, source + EVec3(i, j, 0), W);
            WEwald -= W;
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
    //     const int N = 200;
    //     for (int i = -N; i <= N; i++) {
    //         for (int j = -N; j <= N; j++) {
    //             for (int k = 0; k < forceValue.size(); k++) {
    //                 EMat4 temp = EMat4::Zero();
    //                 Wkernel(spoint, forcePoint[k] + EVec3(i, j, 0), temp);
    //                 vD += temp * forceValue[k];
    //             }
    //         }
    //     }
    //     std::cout << vE.transpose() << std::endl;
    //     std::cout << (vE - vD).transpose() << std::endl;
    //     // std::exit(0);
    // }

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

    Eigen::MatrixXd M2L(4 * equivN, 4 * equivN);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        Eigen::MatrixXd f(4 * checkN, 4);
        for (int k = 0; k < checkN; k++) {
            EMat4 temp = EMat4::Zero();
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            WkernelFF(Cpoint, Mpoint, temp);
            f.block<4, 4>(4 * k, 0) = temp;
        }
        M2L.block(0, 4 * i, 4 * equivN, 4) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f));
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

    return 0;
}

} // namespace StokesPVel2D3D

#undef DIRECTLAYER
