#include "STKFMM/STKFMM.hpp"

namespace stkfmm {

StkWallFMM::StkWallFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_, bool enableFF_)
    : STKFMM(multOrder_, maxPts_, pbc_, kernelComb_) {
    using namespace impl;
    poolFMM.clear();

    if (kernelComb & asInteger(KERNEL::Stokes)) {
        // Stokes image, activate Stokes & Laplace kernels
        poolFMM[KERNEL::Stokes] = new FMMData(KERNEL::Stokes, pbc, multOrder, maxPts, enableFF_);             // uS
        poolFMM[KERNEL::LapPGrad] = new FMMData(KERNEL::LapPGrad, pbc, multOrder, maxPts, enableFF_);         // uL1+uD
        poolFMM[KERNEL::LapPGradGrad] = new FMMData(KERNEL::LapPGradGrad, pbc, multOrder, maxPts, enableFF_); // uL2
        if (!rank)
            std::cout << "enable Stokes image kernel " << std::endl;
    }

    if (kernelComb & asInteger(KERNEL::RPY)) {
        // RPY image, activate RPY, Laplace, & LapQuad kernels
        poolFMM[KERNEL::RPY] = new FMMData(KERNEL::RPY, pbc, multOrder, maxPts, enableFF_);           // uS
        poolFMM[KERNEL::LapPGrad] = new FMMData(KERNEL::LapPGrad, pbc, multOrder, maxPts, enableFF_); // phiSZ+phiDZ
        poolFMM[KERNEL::LapPGradGrad] =
            new FMMData(KERNEL::LapPGradGrad, pbc, multOrder, maxPts, enableFF_); // phiS+phiD
        poolFMM[KERNEL::LapQPGradGrad] = new FMMData(KERNEL::LapQPGradGrad, pbc, multOrder, maxPts, enableFF_); // phibQ
        std::cout << "enable RPY image kernel " << std::endl;
    }

    if (poolFMM.empty()) {
        std::cout << "Error: no kernel activated\n";
    }
}

StkWallFMM::~StkWallFMM() {
    // delete all FMMData
    for (auto &fmm : poolFMM) {
        safeDeletePtr(fmm.second);
    }
}

void StkWallFMM::setPoints(const int nSL, const double *srcSLCoordPtr, const int nTrg, const double *trgCoordPtr,
                           const int nDL, const double *srcDLCoordPtr) {
    if (!poolFMM.empty()) {
        for (auto &fmm : poolFMM) {
            // if (rank == 0)
            //     printf("kernel %u \n", asInteger(fmm.second->kernelChoice));
            fmm.second->deleteTree();
        }
        if (verbose && rank == 0)
            std::cout << "ALL FMM Tree Cleared\n";
    }

    // setup point coordinates
    auto setCoord = [&](const int nPts, const double *coordPtr, std::vector<double> &coord) {
        coord.resize(nPts * 3);
        std::copy(coordPtr, coordPtr + 3 * nPts, coord.begin());
        scaleCoord(nPts, coord.data()); // scale to [0,1)^2x[0,0.5)
        wrapCoord(nPts, coord.data());  // pbc wrap
#pragma omp parallel for
        for (int i = 0; i < nPts; i++) {
            coord[3 * i + 2] += 0.5; // shift z from [0,0.5) to [0.5,1)
        }
    };

    // trg origin -> trgInternal
    setCoord(nTrg, trgCoordPtr, trgCoordInternal);

    // src origin+image -> srcSLInternal
    srcSLCoordInternal.reserve(6 * nSL);
    setCoord(nSL, srcSLCoordPtr, srcSLCoordInternal);
    srcSLCoordInternal.resize(6 * nSL);
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        srcSLCoordInternal[3 * (i + nSL)] = srcSLCoordInternal[3 * i];
        srcSLCoordInternal[3 * (i + nSL) + 1] = srcSLCoordInternal[3 * i + 1];
        srcSLCoordInternal[3 * (i + nSL) + 2] = 1 - srcSLCoordInternal[3 * i + 2];
    }

    // src origin/image
    srcSLImageCoordInternal.resize(3 * nSL);
    std::copy(srcSLCoordInternal.begin() + 3 * nSL, srcSLCoordInternal.begin() + 6 * nSL,
              srcSLImageCoordInternal.begin());
    srcSLOriginCoordInternal.resize(3 * nSL);
    std::copy(srcSLCoordInternal.begin(), srcSLCoordInternal.begin() + 3 * nSL, srcSLOriginCoordInternal.begin());

    if (verbose && rank == 0)
        std::cout << "points set\n";
}

void StkWallFMM::setupTree(KERNEL kernel) {
    std::vector<double> treeCoord(srcSLCoordInternal.size() + trgCoordInternal.size());
    std::copy(srcSLCoordInternal.begin(), srcSLCoordInternal.end(), treeCoord.begin());
    std::copy(trgCoordInternal.begin(), trgCoordInternal.end(), treeCoord.begin() + srcSLCoordInternal.size());
    if (kernel == KERNEL::Stokes) {
        poolFMM[KERNEL::Stokes]->setupTree(srcSLCoordInternal, std::vector<double>(), trgCoordInternal);
        poolFMM[KERNEL::LapPGrad]->setupTree(srcSLCoordInternal, srcSLImageCoordInternal, trgCoordInternal);
        poolFMM[KERNEL::LapPGradGrad]->setupTree(srcSLCoordInternal, std::vector<double>(), trgCoordInternal);
    } else if (kernel == KERNEL::RPY) {
        poolFMM[KERNEL::RPY]->setupTree(srcSLCoordInternal, std::vector<double>(), trgCoordInternal);
        poolFMM[KERNEL::LapPGrad]->setupTree(srcSLCoordInternal, srcSLCoordInternal, trgCoordInternal);
        poolFMM[KERNEL::LapPGradGrad]->setupTree(srcSLCoordInternal, srcSLImageCoordInternal, trgCoordInternal);
        poolFMM[KERNEL::LapQPGradGrad]->setupTree(srcSLImageCoordInternal, std::vector<double>(), trgCoordInternal,
                                                  srcSLCoordInternal.size() / 3, srcSLCoordInternal.data());
    } else {
        std::cout << "Kernel not supported\n";
        std::exit(1);
    }
}

void StkWallFMM::evaluateFMM(const KERNEL kernel, const int nSL, const double *srcSLValuePtr, const int nTrg,
                             double *trgValuePtr, const int nDL, const double *srcDLValuePtr) {

    if (kernel == KERNEL::Stokes) {
        // 3->3
        srcSLValueInternal.resize(nSL * 3);
        trgValueInternal.resize(nTrg * 3);
        std::copy(srcSLValuePtr, srcSLValuePtr + 3 * nSL, srcSLValueInternal.begin());
        evalStokes();
        const int nloop = 3 * nTrg;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValuePtr[i] += trgValueInternal[i];
        }
    } else if (kernel == KERNEL::RPY) {
        // 4->6
        srcSLValueInternal.resize(nSL * 4);
        trgValueInternal.resize(nTrg * 6);
        std::copy(srcSLValuePtr, srcSLValuePtr + 4 * nSL, srcSLValueInternal.begin());
        evalRPY();
        const int nloop = 6 * nTrg;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValuePtr[i] += trgValueInternal[i];
        }
    } else {
        std::cout << "Kernel not supported\n";
        std::exit(1);
    }
}

void StkWallFMM::clearFMM(KERNEL kernel) {
    if (kernel == KERNEL::Stokes) {
        poolFMM[KERNEL::Stokes]->clear();
        poolFMM[KERNEL::LapPGrad]->clear();
        poolFMM[KERNEL::LapPGradGrad]->clear();
    } else if (kernel == KERNEL::RPY) {
        poolFMM[KERNEL::RPY]->clear();
        poolFMM[KERNEL::LapPGrad]->clear();
        poolFMM[KERNEL::LapPGradGrad]->clear();
        poolFMM[KERNEL::LapQPGradGrad]->clear();
    } else {
        std::cout << "Kernel not supported\n";
        std::exit(1);
    }
}

void StkWallFMM::evalStokes() {
    const int nSL = srcSLOriginCoordInternal.size() / 3;
    const int nTrg = trgCoordInternal.size() / 3;
    std::vector<double> srcValStk(nSL * 3 * 2, 0), trgValStk(nTrg * 3, 0);                 // StokesFMM, 3->3
    std::vector<double> srcValL1(nSL * 2, 0), srcValD(nSL * 3, 0), trgValL1D(nTrg * 4, 0); // LapPGrad, 1/3->4
    std::vector<double> srcValL2(nSL * 2, 0), trgValL2(nTrg * 10, 0);                      // LapPGradGrad, 1->10
    std::vector<double> empty;
    const double sF = scaleFactor;
    // step1 Stokes FMM
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        srcValStk[3 * i] = srcSLValueInternal[3 * i];
        srcValStk[3 * i + 1] = srcSLValueInternal[3 * i + 1];
        srcValStk[3 * (i + nSL)] = -srcSLValueInternal[3 * i];
        srcValStk[3 * (i + nSL) + 1] = -srcSLValueInternal[3 * i + 1];
    }
    empty.clear();
    poolFMM[KERNEL::Stokes]->evaluateFMM(srcValStk, empty, trgValStk, scaleFactor);

    // step2 LapPGrad L1D
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        srcValL1[i] = -0.5 * srcSLValueInternal[3 * i + 2];
        srcValL1[i + nSL] = 0.5 * srcSLValueInternal[3 * i + 2];
    }
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        const double y3 = (srcSLOriginCoordInternal[3 * i + 2] - 0.5) / sF;
        srcValD[3 * i + 0] = -y3 * srcSLValueInternal[3 * i + 0];
        srcValD[3 * i + 1] = -y3 * srcSLValueInternal[3 * i + 1];
        srcValD[3 * i + 2] = y3 * srcSLValueInternal[3 * i + 2];
    }
    poolFMM[KERNEL::LapPGrad]->evaluateFMM(srcValL1, srcValD, trgValL1D, scaleFactor);

    // step3 LapPGradGrad L2
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        const double y3 = (srcSLOriginCoordInternal[3 * i + 2] - 0.5) / sF;
        srcValL2[i] = srcSLValueInternal[3 * i + 2] * y3;
        srcValL2[i + nSL] = -srcValL2[i];
    }
    empty.clear();
    poolFMM[KERNEL::LapPGradGrad]->evaluateFMM(srcValL2, empty, trgValL2, scaleFactor);

    // step 4 Assemble together
#pragma omp parallel for
    for (int i = 0; i < nTrg; i++) {
        const double x3 = (trgCoordInternal[3 * i + 2] - 0.5) / sF;
        for (int j = 0; j < 3; j++) {
            trgValueInternal[3 * i + j] =
                trgValStk[3 * i + j] + 0.5 * trgValL2[10 * i + j + 1] + x3 * trgValL1D[4 * i + j + 1];
        }
        trgValueInternal[3 * i + 2] -= trgValL1D[4 * i];
    }
}

void StkWallFMM::evalRPY() {
    const int nSL = srcSLOriginCoordInternal.size() / 3;
    const int nTrg = trgCoordInternal.size() / 3;
    std::vector<double> srcValRPY(nSL * 4 * 2, 0), trgValRPY(nTrg * 6, 0);                    // RPYFMM, 4->6
    std::vector<double> srcValLS(nSL * 2, 0), srcValLD(nSL * 3, 0), trgValSD(nTrg * 10, 0);   // LapPGradGrad S, 1/3->10
    std::vector<double> srcValLSZ(nSL * 2, 0), srcValLDZ(nSL * 6, 0), trgValSDZ(nTrg * 4, 0); // LapPGrad, 1/3->4
    std::vector<double> srcValQ(nSL * 9, 0), trgValQ(nTrg * 10, 0);                           // LapQPGradGrad, 9->10
    std::vector<double> empty;
    const double sF = scaleFactor;

// step1 RPYFMM
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        srcValRPY[4 * i] = srcSLValueInternal[4 * i];         // fx
        srcValRPY[4 * i + 1] = srcSLValueInternal[4 * i + 1]; // fy
        srcValRPY[4 * i + 3] = srcSLValueInternal[4 * i + 3]; // b
        srcValRPY[4 * (i + nSL)] = -srcSLValueInternal[4 * i];
        srcValRPY[4 * (i + nSL) + 1] = -srcSLValueInternal[4 * i + 1];
        srcValRPY[4 * (i + nSL) + 3] = srcSLValueInternal[4 * i + 3]; // b
    }
    empty.clear();
    poolFMM[KERNEL::RPY]->evaluateFMM(srcValRPY, empty, trgValRPY, sF);

// step2 Laplace SD
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        srcValLS[i] = srcSLValueInternal[4 * i + 2] * (-0.5);
        srcValLS[i + nSL] = -srcSLValueInternal[4 * i + 2] * (-0.5);
        const double y3 = (srcSLOriginCoordInternal[3 * i + 2] - 0.5) / sF;
        srcValLD[3 * i] = -y3 * srcSLValueInternal[4 * i];
        srcValLD[3 * i + 1] = -y3 * srcSLValueInternal[4 * i + 1];
        srcValLD[3 * i + 2] = y3 * srcSLValueInternal[4 * i + 2];
    }
    poolFMM[KERNEL::LapPGradGrad]->evaluateFMM(srcValLS, srcValLD, trgValSD, sF);

// step3 Laplace SDZ
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        const double y3 = (srcSLOriginCoordInternal[3 * i + 2] - 0.5) / sF;
        const double b = srcSLValueInternal[4 * i + 3];
        const double b2 = b * b;
        const double f3 = srcSLValueInternal[4 * i + 2];
        srcValLSZ[i] = y3 * f3 * 0.5;
        srcValLSZ[i + nSL] = -y3 * f3 * 0.5;
        srcValLDZ[3 * i + 2] = b2 * f3 * (1. / 6.);
        srcValLDZ[3 * (i + nSL) + 2] = b2 * f3 * (1. / 6.);
    }
    poolFMM[KERNEL::LapPGrad]->evaluateFMM(srcValLSZ, srcValLDZ, trgValSDZ, sF);

// step4 Laplace QPGradGrad
#pragma omp parallel for
    for (int i = 0; i < nSL; i++) {
        const double f1 = srcSLValueInternal[4 * i];
        const double f2 = srcSLValueInternal[4 * i + 1];
        const double f3 = srcSLValueInternal[4 * i + 2];
        const double b = srcSLValueInternal[4 * i + 3];
        const double twob2 = 2 * b * b;
        srcValQ[9 * i] = twob2 * f3 * (1. / 6.);
        srcValQ[9 * i + 4] = twob2 * f3 * (1. / 6.);
        srcValQ[9 * i + 6] = twob2 * f1 * (1. / 6.);
        srcValQ[9 * i + 7] = twob2 * f2 * (1. / 6.);
    }
    empty.clear();
    poolFMM[KERNEL::LapQPGradGrad]->evaluateFMM(srcValQ, empty, trgValQ, sF);

// assemble
#pragma omp parallel for
    for (int i = 0; i < nTrg; i++) {
        // 6 dimensional array per target [vx,vy,vz,gx,gy,gz]
        // u = [vx,vy,vz]+a^2/6*[gx,gy,gz]
        const double x3 = (trgCoordInternal[3 * i + 2] - 0.5) / sF;
        trgValueInternal[6 * i + 0] = //
            trgValRPY[6 * i + 0] + trgValSDZ[4 * i + 1] + x3 * trgValSD[10 * i + 1] + x3 * trgValQ[10 * i + 1];
        trgValueInternal[6 * i + 1] = //
            trgValRPY[6 * i + 1] + trgValSDZ[4 * i + 2] + x3 * trgValSD[10 * i + 2] + x3 * trgValQ[10 * i + 2];
        trgValueInternal[6 * i + 2] =                                                                          //
            trgValRPY[6 * i + 2] + trgValSDZ[4 * i + 3] + x3 * trgValSD[10 * i + 3] + x3 * trgValQ[10 * i + 3] //
            - trgValSD[10 * i] - trgValQ[10 * i];
        trgValueInternal[6 * i + 3] = trgValRPY[6 * i + 3] + 2 * trgValSD[10 * i + 6] + 2 * trgValQ[10 * i + 6];
        trgValueInternal[6 * i + 4] = trgValRPY[6 * i + 4] + 2 * trgValSD[10 * i + 8] + 2 * trgValQ[10 * i + 8];
        trgValueInternal[6 * i + 5] = trgValRPY[6 * i + 5] + 2 * trgValSD[10 * i + 9] + 2 * trgValQ[10 * i + 9];
    }
}

} // namespace stkfmm
