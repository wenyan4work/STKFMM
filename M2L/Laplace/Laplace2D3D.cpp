/*
 * LaplaceM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

namespace Laplace2D3D {

inline double realSum(const double xi, const EVec3 &xn, const EVec3 &xm) {
    // xm: target, xn: source
    EVec3 rmn = xm - xn;
    double rnorm = rmn.norm();
    if (rnorm < eps) {
        return 0;
    }
    return std::erfc(rnorm * xi) / rnorm;
}

inline double realSum2(const double xi, const EVec3 &xn, const EVec3 &xm) {
    // xm: target, xn: source
    double zmn = xm[2] - xn[2];
    double answer = exp(-xi * xi * zmn * zmn) / xi + sqrt(M_PI) * zmn * std::erf(xi * zmn);
    return answer;
}

inline double gkzxi(const double k, double zmn, double xi) {
    double answer =
        exp(k * zmn) * std::erfc(k / (2 * xi) + xi * zmn) + exp(-k * zmn) * std::erfc(k / (2 * xi) - xi * zmn);
    return answer;
}

inline double selfTerm(double xi) { return -2 * xi / sqrt(M_PI); }

inline double gKernelEwald(const EVec3 &xm, const EVec3 &xn) {
    // xm: target, xn: source
    const double xi = 1.8; // recommend for box=1 to get machine precision
    EVec3 target = xm;
    EVec3 source = xn;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);

    // real sum
    int rLim = 4;
    double Kreal = 0;
    for (int i = -rLim; i <= rLim; i++) {
        for (int j = -rLim; j <= rLim; j++) {
            EVec3 rmn = target - source + EVec3(i, j, 0);
            if (rmn.norm() < 1e-13) {
                continue;
            }
            Kreal += realSum(xi, EVec3(0, 0, 0), rmn);
        }
    }

    // wave sum
    int wLim = 4;
    double Kwave = 0;
    EVec3 rmn = target - source;
    const double rmnnorm = rmn.norm();
    double zmn = rmn[2];
    rmn[2] = 0;
    for (int i = -wLim; i <= wLim; i++) {
        for (int j = -wLim; j <= wLim; j++) {
            if (i == 0 && j == 0) {
                continue;
            }
            EVec3 kvec = EVec3(i, j, 0) * (2 * M_PI);
            double knorm = kvec.norm();
            Kwave += cos(kvec[0] * rmn[0] + kvec[1] * rmn[1]) * (1 / knorm) * gkzxi(knorm, zmn, xi);
        }
    }
    Kwave *= M_PI;

    double Kreal2 = 2 * sqrt(M_PI) * realSum2(xi, source, target);
    double Kself = rmnnorm < eps ? -2 * xi / sqrt(M_PI) : 0;

    return (Kreal + Kwave - Kreal2 + Kself) / (4 * M_PI);
}

inline double gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < eps ? 0 : 1 / (4 * M_PI * rnorm);
}

// Out of Direct Sum Layer, far field part
inline double gKernelFF(const EVec3 &target, const EVec3 &source) {
    double fEwald = gKernelEwald(target, source);
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            double gFree = gKernel(target, source - EVec3(i, j, 0));
            fEwald -= gFree;
        }
    }

    return fEwald;
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

        Eigen::VectorXd f(checkN);
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

    saveEMat(M2L, "M2L_laplace_2D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_laplace_2D3D_p" + std::to_string(pEquiv));

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

    // testing Ewald routine
    double Madelung2D =
        gKernelEwald(EVec3(0, 0, 0), EVec3(0.5, 0.5, 0)) * (-1) + gKernelEwald(EVec3(0, 0, 0), EVec3(0, 0, 0)) * 1;
    std::cout << std::setprecision(16) << "Madelung2D: " << Madelung2D << " Error: " << Madelung2D + 2.2847222932891311
              << std::endl;

    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> chargePoint(2);
    std::vector<double> chargeValue(2);
    chargePoint[0] = EVec3(0.5, 0.5, 0);
    chargeValue[0] = -1;
    chargePoint[1] = EVec3(0, 0, 0);
    chargeValue[1] = 1;

    // solve M
    Eigen::VectorXd f(checkN);
    for (int k = 0; k < checkN; k++) {
        double temp = 0;
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < chargePoint.size(); p++) {
            temp = temp + gKernel(Cpoint, chargePoint[p]) * (chargeValue[p]);
        }
        f(k) = temp;
    }
    Eigen::VectorXd Msource = (AMpinvU.transpose() * (AMpinvVT.transpose() * f));

    Eigen::VectorXd M2Lsource = M2L * Msource;

    std::cout << "Msource: " << Msource.transpose() << std::endl;
    std::cout << "M2Lsource: " << M2Lsource.transpose() << std::endl;

    EVec3 samplePoint(0, 0, 0);
    double Usample = 0;
    double UsampleSP = 0;

    for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
        for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
            for (size_t p = 0; p < chargePoint.size(); p++) {
                Usample += gKernel(samplePoint, chargePoint[p] + EVec3(i, j, 0)) * chargeValue[p];
            }
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
    std::cout << "Error : " << UsampleSP + Usample + 2.284722293289131159 / (4 * M_PI) << std::endl;

    samplePoint = EVec3(0.5, 0.5, 0);
    Usample = 0;
    UsampleSP = 0;

    for (int i = -DIRECTLAYER; i < 1 + DIRECTLAYER; i++) {
        for (int j = -DIRECTLAYER; j < 1 + DIRECTLAYER; j++) {
            for (size_t p = 0; p < chargePoint.size(); p++) {
                Usample += gKernel(samplePoint, chargePoint[p] + EVec3(i, j, 0)) * chargeValue[p];
            }
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
    std::cout << "Error : " << UsampleSP + Usample - 2.284722293289131159 / (4 * M_PI) << std::endl;

    return 0;
}

} // namespace Laplace2D3D
