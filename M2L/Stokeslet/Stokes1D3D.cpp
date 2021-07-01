/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

namespace Stokes1D3D {

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
        EMat3 G = EMat3::Zero();
        Gkernel(rvec, EVec3(i, 0, 0), G);
        GNF += G;
    }
}

inline void GkernelFF(const EVec3 &rvec, EMat3 &GFF) {
    GFF.setZero();
    const int N = DIRECTLAYER;
    for (int i = N + 1; i < SUM1D; i++) {
        EMat3 G1 = EMat3::Zero();
        EMat3 G2 = EMat3::Zero();
        Gkernel(rvec, EVec3(i, 0, 0), G1);
        Gkernel(rvec, EVec3(-i, 0, 0), G2);
        GFF += G1 + G2;
    }
}

// calculate the M2L matrix of images from 2 to 1000
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

    saveEMat(M2L, "M2L_stokes_vel_1D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_stokes_vel_1D3D_p" + std::to_string(pEquiv));

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
    forcePoint[0] = EVec3(0.1, 0.5, 0.5);
    forceValue[0] = EVec3(1, 1, -1);
    forcePoint[1] = EVec3(0.9, 0.5, 0.5);
    forceValue[1] = EVec3(-2, 2, 2);
    forcePoint[2] = EVec3(0.2, 0.2, 0.2);
    forceValue[2] = EVec3(1, -3, -1);

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
        EVec3 Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
        EVec3 Fpoint(M2Lsource[3 * p], M2Lsource[3 * p + 1], M2Lsource[3 * p + 2]);
        Gkernel(samplePoint, Lpoint, G);
        UFFL2T += G * Fpoint;
    }

    for (size_t p = 0; p < forcePoint.size(); p++) {
        EMat3 GNF, GFF;
        GkernelNF(samplePoint - forcePoint[p], GNF);
        GkernelFF(samplePoint - forcePoint[p], GFF);
        UEwald += (GNF + GFF) * forceValue[p];
    }

    std::cout << "UNF:" << UNF.transpose() << std::endl;
    std::cout << "UFFL2T:" << UFFL2T.transpose() << std::endl;
    std::cout << "USum:" << (UNF + UFFL2T).transpose() << std::endl;
    std::cout << "UEwald:" << UEwald.transpose() << std::endl;

    std::cout << "error: " << (UNF + UFFL2T - UEwald).transpose() << std::endl;

    return 0;
}

} // namespace Stokes1D3D
