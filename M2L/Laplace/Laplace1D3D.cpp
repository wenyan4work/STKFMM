/*
 * LaplaceM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

namespace Laplace1D3D {

inline double gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < eps ? 0 : 1 / (4 * M_PI * rnorm);
}

double gKernelFF(const EVec3 &target, const EVec3 &source, const int directTerm = SUM1D) {
    // use asymptotic
    const double L3 = 1.0;
    double potentialDirect = 0;
    for (int t = DIRECTLAYER + 1; t < directTerm; t++) {
        potentialDirect +=
            gKernel(target, source + EVec3(t * L3, 0, 0)) + gKernel(target, source - EVec3(t * L3, 0, 0));
    }

    return potentialDirect;
}

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    constexpr int kdim[2] = {1, 1}; // target, source dimension

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
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Lpoint(pointLEquiv[3 * l], pointLEquiv[3 * l + 1], pointLEquiv[3 * l + 2]);
            AL(k, l) = gKernel(Cpoint, Lpoint);
        }
    }
    pinv(AL, ALpinvU, ALpinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const EVec3 Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);

        EVec f(checkN);
        for (int k = 0; k < checkN; k++) {
            EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
            f(k) = gKernelFF(Cpoint, Mpoint); // sum the images
        }
        M2C.col(i) = f;
        M2L.col(i) = (ALpinvU.transpose() * (ALpinvVT.transpose() * f));
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    saveEMat(M2L, "M2L_laplace_1D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_laplace_1D3D_p" + std::to_string(pEquiv));

    EMat AM(kdim[0] * checkN, kdim[1] * equivN); // M den to M check
    EMat AMpinvU(AM.cols(), AM.rows());
    EMat AMpinvVT(AM.cols(), AM.rows());
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            AM(k, l) = gKernel(Cpoint, Mpoint);
        }
    }
    pinv(AM, AMpinvU, AMpinvVT);

    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> chargePoint(4);
    std::vector<double> chargeValue(4);
    chargePoint[0] = EVec3(0.125, 0.5, 0.5);
    chargeValue[0] = 1;
    chargePoint[1] = EVec3(0.375, 0.5, 0.5);
    chargeValue[1] = -1;
    chargePoint[2] = EVec3(0.625, 0.5, 0.5);
    chargeValue[2] = 1;
    chargePoint[3] = EVec3(0.875, 0.5, 0.5);
    chargeValue[3] = -1;

    // solve M
    EVec f(checkN);
    for (int k = 0; k < checkN; k++) {
        double temp = 0;
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < chargePoint.size(); p++) {
            temp = temp + gKernel(Cpoint, chargePoint[p]) * (chargeValue[p]);
        }
        f(k) = temp;
    }
    EVec Msource = (AMpinvU.transpose() * (AMpinvVT.transpose() * f));

    EVec M2Lsource = M2L * Msource;

    std::cout << "Msource: " << Msource.transpose() << std::endl;
    std::cout << "M2Lsource: " << M2Lsource.transpose() << std::endl;

    EVec3 samplePoint(0.125, 0.5, 0.5);
    double Usample = 0;
    double UsampleSP = 0;

    for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
        for (size_t p = 0; p < chargePoint.size(); p++) {
            Usample += gKernel(samplePoint, chargePoint[p] + EVec3(k, 0, 0)) * chargeValue[p];
        }
    }

    for (int p = 0; p < equivN; p++) {
        EVec3 Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint) * M2Lsource[p];
    }

    std::cout << "samplePoint:" << samplePoint << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;
    std::cout << "error:" << UsampleSP + Usample + 8 * log(2) / (4 * M_PI) << std::endl;

    samplePoint = EVec3(0.625, 0.5, 0.5);
    Usample = 0;
    UsampleSP = 0;

    for (int k = -DIRECTLAYER; k < 1 + DIRECTLAYER; k++) {
        for (size_t p = 0; p < chargePoint.size(); p++) {
            Usample += gKernel(samplePoint, chargePoint[p] + EVec3(k, 0, 0)) * chargeValue[p];
        }
    }

    for (int p = 0; p < equivN; p++) {
        EVec3 Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
        UsampleSP += gKernel(samplePoint, Lpoint) * M2Lsource[p];
    }

    std::cout << "samplePoint:" << samplePoint << std::endl;
    std::cout << "Usample NF:" << Usample << std::endl;
    std::cout << "Usample FF:" << UsampleSP << std::endl;
    std::cout << "Usample FF+NF total:" << UsampleSP + Usample << std::endl;
    std::cout << "error:" << UsampleSP + Usample + 8 * log(2) / (4 * M_PI) << std::endl;

    return 0;
}

} // namespace Laplace1D3D
