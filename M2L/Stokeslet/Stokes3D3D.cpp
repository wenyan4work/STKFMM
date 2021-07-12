/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

namespace Stokes3D3D {

/**************************************
 *
 *
 *   Stokeslet triply periodic
 *
 * **************************************/

inline EMat3 AEW(const double xi, const EVec3 &rvec) {
    const double r = rvec.norm();
    EMat3 A = 2 * (xi * exp(-(xi * xi) * (r * r)) / (sqrt(M_PI) * r * r) + erfc(xi * r) / (2 * r * r * r)) *
                  (r * r * EMat3::Identity() + (rvec * rvec.transpose())) -
              4 * xi / sqrt(M_PI) * exp(-(xi * xi) * (r * r)) * EMat3::Identity();
    return A;
}

inline EMat3 BEW(const double xi, const EVec3 &kvec) {
    const double k = kvec.norm();
    EMat3 B = 8 * M_PI * (1 + k * k / (4 * (xi * xi))) * ((k * k) * EMat3::Identity() - (kvec * kvec.transpose())) /
              (k * k * k * k);
    B *= exp(-k * k / (4 * xi * xi));
    return B;
}

inline void GkernelEwald(const EVec3 &rvec, EMat3 &Gsum) {
    const double xi = 2;
    const double r = rvec.norm();
    EMat3 real = EMat3::Zero();
    const int N = 5;
    EMat3 Gself = -4 * xi / sqrt(M_PI) * EMat3::Identity(); // the self term
    if (r < eps) {
        real += Gself;
    }
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                if (i == 0 && j == 0 && k == 0 && r < eps) {
                    continue;
                }
                real = real + AEW(xi, rvec + EVec3(i, j, k));
            }
        }
    }

    EMat3 wave = EMat3::Zero();
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                EVec3 kvec(2 * M_PI * i, 2 * M_PI * j, 2 * M_PI * k);
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                } else {
                    wave = wave + BEW(xi, kvec) * cos(kvec.dot(rvec));
                }
            }
        }
    }
    Gsum = (real + wave) / (8 * M_PI);
}

inline void Gkernel(const EVec3 &target, const EVec3 &source, EMat3 &answer) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < eps) {
        answer = EMat3::Zero();
        return;
    }
    EMat3 part2 = rst * rst.transpose() / (rnorm * rnorm * rnorm);
    EMat3 part1 = EMat3::Identity() / rnorm;
    answer = (part1 + part2) / (8 * M_PI);
}

inline void GkernelNF(const EVec3 &rvec, EMat3 &GNF) {
    GNF.setZero();
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                EMat3 G = EMat3::Zero();
                Gkernel(rvec, EVec3(i, j, k), G);
                GNF += G;
            }
        }
    }
}

inline void GkernelFF(const EVec3 &rvec, EMat3 &GFF) {
    EMat3 GNF = EMat3::Zero();
    GkernelNF(rvec, GNF);
    GFF = EMat3::Zero();
    GkernelEwald(rvec, GFF);
    GFF -= GNF;
}

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    constexpr int kdim[2] = {3, 3}; // target, source dimension

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const int pEquiv = atoi(argv[1]);
    const int pCheck = atoi(argv[1]);

    const double pCenterMEquiv[3] = {-(scaleIn - 1) / 2, -(scaleIn - 1) / 2, -(scaleIn - 1) / 2};
    const double pCenterMCheck[3] = {-(scaleOut - 1) / 2, -(scaleOut - 1) / 2, -(scaleOut - 1) / 2};

    const double pCenterLEquiv[3] = {-(scaleOut - 1) / 2, -(scaleOut - 1) / 2, -(scaleOut - 1) / 2};
    const double pCenterLCheck[3] = {-(scaleIn - 1) / 2, -(scaleIn - 1) / 2, -(scaleIn - 1) / 2};

    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterMEquiv[0]), scaleIn, 0);
    auto pointMCheck = surface(pCheck, (double *)&(pCenterMCheck[0]), scaleOut, 0);

    auto pointLCheck = surface(pCheck, (double *)&(pCenterLCheck[0]), scaleIn, 0);
    auto pointLEquiv = surface(pEquiv, (double *)&(pCenterLEquiv[0]), scaleOut, 0);

    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointMCheck.size() / 3;
    EMat M2L(kdim[1] * equivN, kdim[1] * equivN); // M2L density
    EMat M2C(kdim[0] * checkN, kdim[1] * equivN); // M2C check surface

    EMat AL(kdim[0] * checkN, kdim[1] * equivN); // L den to L check
    EMat ALpinvU(AL.cols(), AL.rows());
    EMat ALpinvVT(AL.cols(), AL.rows());
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Lpoint(pointLEquiv[3 * l], pointLEquiv[3 * l + 1], pointLEquiv[3 * l + 2]);
            EMat3 G = EMat3::Zero();
            Gkernel(Cpoint, Lpoint, G);
            AL.block<kdim[0], kdim[1]>(kdim[0] * k, kdim[1] * l) = G;
        }
    }
    pinv(AL, ALpinvU, ALpinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const EVec3 Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);

        EMat f(3 * checkN, 3);
        for (int k = 0; k < checkN; k++) {
            EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
            EMat3 G = EMat3::Zero();
            GkernelFF(Cpoint - Mpoint, G);
            f.block<kdim[0], kdim[1]>(3 * k, 0) = G;
        }
        M2C.block(0, 3 * i, 3 * checkN, 3) = f;
        M2L.block(0, 3 * i, 3 * checkN, 3) = (ALpinvU.transpose() * (ALpinvVT.transpose() * f));
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    saveEMat(M2L, "M2L_stokes_vel_3D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_stokes_vel_3D3D_p" + std::to_string(pEquiv));

    EMat AM(kdim[0] * checkN, kdim[1] * equivN); // M den to M check
    EMat AMpinvU(AM.cols(), AM.rows());
    EMat AMpinvVT(AM.cols(), AM.rows());
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            EMat3 G = EMat3::Zero();
            Gkernel(Cpoint, Mpoint, G);
            AM.block<kdim[0], kdim[1]>(kdim[0] * k, kdim[1] * l) = G;
        }
    }
    pinv(AM, AMpinvU, AMpinvVT);

    // test
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(3);
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forceValue(3);
    forcePoint[0] = EVec3(0.1, 0.55, 0.2);
    forceValue[0] = EVec3(1, 0, 0);
    forcePoint[1] = EVec3(0.5, 0.1, 0.3);
    forceValue[1] = EVec3(-1, 1, 1);
    forcePoint[2] = EVec3(0.8, 0.5, 0.7);
    forceValue[2] = EVec3(0, 0, -1);

    // solve M
    EVec f(3 * checkN);
    for (int k = 0; k < checkN; k++) {
        EVec3 temp = EVec3::Zero();
        EMat3 G = EMat3::Zero();
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < forcePoint.size(); p++) {
            Gkernel(Cpoint, forcePoint[p], G);
            temp = temp + G * (forceValue[p]);
        }
        f.block<3, 1>(3 * k, 0) = temp;
    }
    EVec Msource = AMpinvU.transpose() * (AMpinvVT.transpose() * f);
    EVec M2Lsource = M2L * Msource;

    std::cout << "Msource: " << Msource.transpose() << std::endl;
    std::cout << "M2Lsource: " << M2Lsource.transpose() << std::endl;

    EVec3 samplePoint(0.5, 0.5, 0.5);
    EVec3 UNF(0, 0, 0);
    EVec3 UFFL2T(0, 0, 0);
    EVec3 UEwald(0, 0, 0);
    EMat3 G = EMat3::Zero();

    for (size_t p = 0; p < forcePoint.size(); p++) {
        G.setZero();
        GkernelNF(samplePoint - forcePoint[p], G);
        UNF += G * forceValue[p];
    }

    for (int p = 0; p < equivN; p++) {
        G.setZero();
        EVec3 Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
        EVec3 Fpoint(M2Lsource[3 * p], M2Lsource[3 * p + 1], M2Lsource[3 * p + 2]);
        Gkernel(samplePoint, Lpoint, G);
        UFFL2T += G * Fpoint;
    }

    for (size_t p = 0; p < forcePoint.size(); p++) {
        G.setZero();
        GkernelEwald(samplePoint - forcePoint[p], G);
        UEwald += G * forceValue[p];
    }

    std::cout << "UNF:" << UNF.transpose() << std::endl;
    std::cout << "UFFL2T:" << UFFL2T.transpose() << std::endl;
    std::cout << "USum:" << (UNF + UFFL2T).transpose() << std::endl;
    std::cout << "UEwald:" << UEwald.transpose() << std::endl;

    std::cout << "error: " << (UNF + UFFL2T - UEwald).transpose() << std::endl;

    return 0;
}

} // namespace Stokes3D3D
