/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

namespace StokesPVel1D3D {

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
    constexpr int kdim[2] = {4, 4}; // target, source dimension

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
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, Lpoint, W);
            AL.block<kdim[0], kdim[1]>(kdim[0] * k, kdim[1] * l) = W;
        }
    }
    pinv(AL, ALpinvU, ALpinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const EVec3 Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);

        EMat f(kdim[0] * checkN, kdim[1]);
        for (int k = 0; k < checkN; k++) {
            EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
            EMat4 W = EMat4::Zero();
            WkernelFF(Cpoint, Mpoint, W);
            f.block<kdim[0], kdim[1]>(kdim[0] * k, 0) = W;
        }
        M2C.block(0, kdim[1] * i, kdim[0] * checkN, kdim[1]) = f;
        M2L.block(0, kdim[1] * i, kdim[0] * checkN, kdim[1]) = (ALpinvU.transpose() * (ALpinvVT.transpose() * f));
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    saveEMat(M2L, "M2L_stokesPVel_1D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_stokesPVel_1D3D_p" + std::to_string(pEquiv));

    EMat AM(kdim[0] * checkN, kdim[1] * equivN); // M den to M check
    EMat AMpinvU(AM.cols(), AM.rows());
    EMat AMpinvVT(AM.cols(), AM.rows());

#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, Mpoint, W);
            AM.block<kdim[0], kdim[1]>(kdim[0] * k, kdim[1] * l) = W;
        }
    }
    pinv(AM, AMpinvU, AMpinvVT);

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
    Eigen::VectorXd f(kdim[0] * checkN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        EVec4 temp = EVec4::Zero();
        for (int p = 0; p < forceValue.size(); p++) {
            EMat4 W = EMat4::Zero();
            Wkernel(Cpoint, forcePoint[p], W);
            temp += W * (forceValue[p]);
        }
        f.block<kdim[0], 1>(kdim[0] * k, 0) = temp;
    }
    Eigen::VectorXd Msource = AMpinvU.transpose() * (AMpinvVT.transpose() * f);
    Eigen::VectorXd M2Lsource = M2L * (Msource);
    std::cout << "Msource: " << Msource.transpose() << std::endl;
    std::cout << "M2Lsource: " << M2Lsource.transpose() << std::endl;

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
        std::cout << "WFF from Lequiv: " << WFFL.transpose() << std::endl;
        std::cout << "WFF from Kernel: " << WFFK.transpose() << std::endl;
        std::cout << "FF Error: " << (WFFL - WFFK).transpose() << std::endl;
    }

    return 0;
}

} // namespace StokesPVel1D3D

#undef DIRECTLAYER
