/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

namespace StokesPVel3D3D {

;
using EMat64 = Eigen::Matrix<double, 6, 4>;

/**********************************************
 *   Ewald for Stokeslet
 * ********************************************/
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
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                if (r < eps && i == 0 && j == 0 && k == 0)
                    continue;
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, k));
            }
        }
    }

    EMat3 Gself = -4 * xi / sqrt(M_PI) * Eigen::Matrix3d::Identity(); // the self term
    if (r < eps) {
        real += Gself;
    }

    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                Eigen::Vector3d kvec(i, j, k);
                kvec *= (2 * M_PI);
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                wave += BEW(xi, kvec) * cos(kvec.dot(rvec));
            }
        }
    }
    Gsum = real + wave;
}

/*********************************************************
 * Ewald for Laplace potential grad
 * *******************************************************/
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

/***************************************************************
 * Combination of Laplace grad and Stokeslet
 * *************************************************************/
// fx,fy,fz,trD -> p, vx,vy,vz
inline void Wkernel(const EVec3 &target, const EVec3 &source, EMat4 &answer) {
    EVec3 rst = target - source;
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

    /************************************
     * Verification for this part is done in testEwald()
     * **********************************/
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

    /************************************
     * Verification needed for this part
     * **********************************/
    // pressure gradient balancing net force
    answer(0, 0) += (target[0] - 0.5);
    answer(0, 1) += (target[1] - 0.5);
    answer(0, 2) += (target[2] - 0.5);

    /************************************
     * Verification for this part is done in calcFlux()
     * **********************************/
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
}

/**
 * @brief calculate the flux across xy, xz, yz surface
 *
 * @param val
 * @param pos
 */
void calcFlux(const std::vector<EVec4, Eigen::aligned_allocator<EVec4>> &val,
              const std::vector<EVec3, Eigen::aligned_allocator<EVec3>> &pos) {
    constexpr int NX = 32;
    std::vector<double> meshPos(NX * NX * 2, 0.0);
    std::vector<double> meshVal(NX * NX, 0.0);
    const double dx = 1.0 / NX;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NX; j++) {
            meshPos[2 * (i * NX + j) + 0] = dx * i;
            meshPos[2 * (i * NX + j) + 1] = dx * j;
        }
    }

    const int nsrc = pos.size();
    const int ni = NX * NX;
    for (int ax = 0; ax < 3; ax++) {
        meshVal.clear();
        meshVal.resize(ni, 0);
#pragma omp parallel for
        for (int i = 0; i < ni; i++) {
            double a = meshPos[2 * i];
            double b = meshPos[2 * i + 1];
            EVec3 trg;
            trg[ax] = 0;
            trg[(ax + 1) % 3] = a;
            trg[(ax + 2) % 3] = b;
            double f = 0;
            for (int j = 0; j < nsrc; j++) {
                EMat4 WE = EMat4::Zero();
                WkernelEwald(trg, pos[j], WE);
                EVec4 pvel = WE * val[j];
                f += pvel[ax + 1];
            }
            meshVal[i] = f;
        }
        double flux = std::accumulate(meshVal.begin(), meshVal.end(), 0.0);
        std::cout << "flux along axis " << ax << " = " << flux * dx * dx << std::endl;
    }
}

void testEwald() {
    // test Ewald kernel with absolute convergent source configuration
    {
        const EVec3 center(0.5, 0.5, 0.5);
        const double rad = 0.2;
        const int nPts = 8;

        std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(nPts);
        std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(nPts);

        for (int i = 0; i < nPts; i++) {
            const double alpha = (2 * M_PI * i) / nPts;
            forcePoint[i] = rad * EVec3(cos(alpha), sin(alpha), 0) + center;
            forceValue[i] = EVec4(cos(alpha), sin(alpha), 0, 1) * cos(i * M_PI);
        }

        {
            // check src settings
            EVec4 fnetS;
            EVec3 dnetS;
            fnetS.setZero();
            dnetS.setZero();
            for (int i = 0; i < nPts; i++) {
                fnetS += forceValue[i];
                dnetS += forceValue[i][3] * forcePoint[i];
            }
            std::cout << "From S: " << std::endl;
            std::cout << "Source Sum: " << fnetS.transpose() << std::endl;
            std::cout << "TrD Dipole Sum: " << dnetS.transpose() << std::endl;
        }

        calcFlux(forceValue, forcePoint);

#pragma omp parallel for
        for (int s = 0; s < 200; s++) {
            const EVec3 samplePoint(0.2, 0.3, 0.01 * s - 0.5);
            EVec4 pvelDirect = EVec4::Zero();
            EVec4 pvelEwald = EVec4::Zero();

            for (int p = 0; p < nPts; p++) {
                EMat4 W = EMat4::Zero();
                WkernelEwald(samplePoint, forcePoint[p], W);
                pvelEwald += W * forceValue[p];
            }

            for (int p = 0; p < nPts; p++) {
                EMat4 WNF = EMat4::Zero();
                const int N = 40;
                for (int i = -N; i < N + 1; i++) {
                    for (int j = -N; j < N + 1; j++) {
                        for (int k = -N; k < N + 1; k++) {
                            EMat4 W = EMat4::Zero();
                            Wkernel(samplePoint, forcePoint[p] + EVec3(i, j, k), W);
                            WNF += W;
                        }
                    }
                }
                pvelDirect += WNF * forceValue[p];
            }

#pragma omp critical
            {

                std::cout << samplePoint.transpose() << std::endl;
                std::cout << "x,p " << samplePoint[2] << " " << pvelEwald[0] << std::endl;
                std::cout << "Ewald " << pvelEwald.transpose() << std::endl;
                std::cout << "Direct " << pvelDirect.transpose() << std::endl;
                std::cout << "Error " << (pvelDirect - pvelEwald).transpose() << std::endl;
            }
        }
    }

    std::exit(0);
}

/***
 * Explain: this kernel WEwald does not satisfy translational invariance.
 *
 * **/
void testTranslation() {
    const int nPts = 2;
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(nPts);
    std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(nPts);
    forcePoint[0] = EVec3(0.3, 0.1, 0.1);
    forcePoint[1] = EVec3(0.7, 0.1, 0.1);
    forceValue[0] = EVec4(0, 0, 0, 1);
    forceValue[1] = EVec4(0, 0, 0, -1);

    for (int i = 0; i < nPts; i++) {
        std::cout << "src loc " << forcePoint[i].transpose() << " val " << forceValue[i].transpose() << std::endl;
    }

    EVec3 samplePoint(0.1, 0.2, 0.3);

    for (int i = 0; i < 5; i++) {
        EVec3 shift(i * 0.2, 0, 0);
        auto shiftPoint = forcePoint;
        shiftPoint[0] += shift;
        shiftPoint[1] += shift;
        EVec3 shiftSample = samplePoint + shift;
        // reset to [0,1]
        for (int j = 0; j < 3; j++) {
            shiftPoint[0][j] = shiftPoint[0][j] - floor(shiftPoint[0][j]);
            shiftPoint[1][j] = shiftPoint[1][j] - floor(shiftPoint[1][j]);
            shiftSample[j] = shiftSample[j] - floor(shiftSample[j]);
        }
        EVec4 pvel = EVec4::Zero();
        for (int j = 0; j < nPts; j++) {
            EMat4 W;
            WkernelEwald(shiftSample, shiftPoint[j], W);
            pvel += W * forceValue[j];
        }
        std::cout << "pvel " << pvel.transpose() << std::endl;
        calcFlux(forceValue, shiftPoint);
    }
}

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

    saveEMat(M2L, "M2L_stokesPVel_3D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_stokesPVel_3D3D_p" + std::to_string(pEquiv));

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
    const int nPts = 5;
    std::vector<EVec3, Eigen::aligned_allocator<EVec3>> forcePoint(nPts);
    std::vector<EVec4, Eigen::aligned_allocator<EVec4>> forceValue(nPts);
    double dsum = 0;
    for (int i = 0; i < nPts; i++) {
        forcePoint[i] = EVec3::Random() * 0.1 + EVec3(0.5, 0.5, 0.5);
        forceValue[i] = EVec4::Random();
        dsum += forceValue[i][3];
    }
    for (auto &v : forceValue) {
        v[3] -= dsum / nPts;
    }

    for (int i = 0; i < nPts; i++) {
        std::cout << "src loc " << forcePoint[i].transpose() << " val " << forceValue[i].transpose() << std::endl;
    }

    // check src settings
    EVec4 fnetS;
    EVec3 dnetS;
    fnetS.setZero();
    dnetS.setZero();
    for (int i = 0; i < nPts; i++) {
        fnetS += forceValue[i];
        dnetS += forceValue[i][3] * forcePoint[i];
    }
    std::cout << "From S: " << std::endl;
    std::cout << "Source Sum: " << fnetS.transpose() << std::endl;
    std::cout << "TrD Dipole Sum: " << dnetS.transpose() << std::endl;

    calcFlux(forceValue, forcePoint);

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
    Eigen::VectorXd Msource = (AMpinvU.transpose() * (AMpinvVT.transpose() * f));

    EVec4 fnetM;
    EVec3 dnetM;
    fnetM.setZero();
    dnetM.setZero();
    for (int i = 0; i < equivN; i++) {
        fnetM += EVec4(Msource[4 * i + 0], Msource[4 * i + 1], Msource[4 * i + 2], Msource[4 * i + 3]);
        dnetM[0] += Msource[4 * i + 3] * pointMEquiv[3 * i];
        dnetM[1] += Msource[4 * i + 3] * pointMEquiv[3 * i + 1];
        dnetM[2] += Msource[4 * i + 3] * pointMEquiv[3 * i + 2];
    }

    std::cout << "From M: " << std::endl;
    std::cout << "Source Sum: " << fnetM.transpose() << std::endl;
    std::cout << "TrD Dipole Sum: " << dnetM.transpose() << std::endl;
    EVec4 fnetError = fnetM - fnetS;
    std::cout << "Source MS Error: " << fnetError.transpose() << std::endl;
    EVec3 dnetError = dnetM - dnetS;
    std::cout << "Dipole MS Error: " << dnetError.transpose() << std::endl;

    EVec3 dnetMP;
    dnetMP.setZero();
    auto pointMP = pointMEquiv;
    for (int i = 0; i < 3 * equivN; i++) {
        pointMP[i] = pointMP[i] - floor(pointMP[i]);
    }
    for (int i = 0; i < equivN; i++) {
        dnetMP[0] += Msource[4 * i + 3] * pointMP[3 * i];
        dnetMP[1] += Msource[4 * i + 3] * pointMP[3 * i + 1];
        dnetMP[2] += Msource[4 * i + 3] * pointMP[3 * i + 2];
    }

    std::cout << "From MP: " << std::endl;
    std::cout << "TrD Dipole Sum: " << dnetMP.transpose() << std::endl;
    std::cout << "Dipole MPS Error: " << (dnetMP - dnetS).transpose() << std::endl;

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    for (int i = 0; i < 6; i++) {
        // EVec3 samplePoint = EVec3(0.2, 0.2, 0.2) * i + EVec3(0., 0., 0.);
        EVec3 samplePoint = EVec3::Random() * 0.5 + EVec3(0.5, 0.5, 0.5);

        // Compute: WFF from L, WFF from WkernelFF
        EVec4 WFFL = EVec4::Zero();
        EVec4 WFFM = EVec4::Zero();
        EVec4 WFFS = EVec4::Zero();
        EVec4 WS = EVec4::Zero();
        std::cout << "----------------------------" << std::endl;
        std::cout << "sample point: " << samplePoint.transpose() << std::endl;

        // S2T
        for (int k = 0; k < forceValue.size(); k++) {
            EMat4 W = EMat4::Zero();
            WkernelFF(samplePoint, forcePoint[k], W);
            WFFS += W * forceValue[k];
            W.setZero();
            WkernelEwald(samplePoint, forcePoint[k], W);
            WS += W * forceValue[k];
        }
        std::cout << "WS: " << WS.transpose() << std::endl;
        std::cout << "WFF S2T: " << WFFS.transpose() << std::endl;

        // L2T
        for (int k = 0; k < equivN; k++) {
            EVec3 Lpoint(pointLEquiv[3 * k], pointLEquiv[3 * k + 1], pointLEquiv[3 * k + 2]);
            EMat4 W = EMat4::Zero();
            Wkernel(samplePoint, Lpoint, W);
            WFFL += W * M2Lsource.block<4, 1>(4 * k, 0);
        }
        WFFL[1] += (dnetM[0] - dnetMP[0]) * 0.5;
        WFFL[2] += (dnetM[1] - dnetMP[1]) * 0.5;
        WFFL[3] += (dnetM[2] - dnetMP[2]) * 0.5;
        std::cout << "WFF L2T: " << WFFL.transpose() << std::endl;
        std::cout << "WFF Error LS: " << (WFFL - WFFS).transpose() << std::endl;

        // M2T
        for (int k = 0; k < equivN; k++) {
            EVec3 Mpoint(pointMEquiv[3 * k], pointMEquiv[3 * k + 1], pointMEquiv[3 * k + 2]);
            EMat4 W = EMat4::Zero();
            WkernelFF(samplePoint, Mpoint, W);
            WFFM += W * Msource.block<4, 1>(4 * k, 0);
        }
        WFFM[1] += (dnetM[0] - dnetMP[0]) * 0.5;
        WFFM[2] += (dnetM[1] - dnetMP[1]) * 0.5;
        WFFM[3] += (dnetM[2] - dnetMP[2]) * 0.5;
        std::cout << "WFF M2T: " << WFFM.transpose() << std::endl;
        std::cout << "WFF Error MS: " << (WFFM - WFFS).transpose() << std::endl;
    }

    return 0;
}

} // namespace StokesPVel3D3D

#undef DIRECTLAYER
