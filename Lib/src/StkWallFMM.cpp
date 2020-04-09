#include "STKFMM/STKFMM.hpp"

namespace stkfmm {
StkWallFMM::StkWallFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_)
    : STKFMM(multOrder_, maxPts_, pbc_, kernelComb_) {
    using namespace impl;
    poolFMM.clear();

    if (kernelComb & asInteger(KERNEL::Stokes)) {
        // Stokes image, activate Stokes & Laplace kernels
        poolFMM[KERNEL::Stokes] = new FMMData(KERNEL::Stokes, pbc, multOrder, maxPts);             // uS
        poolFMM[KERNEL::LapPGrad] = new FMMData(KERNEL::LapPGrad, pbc, multOrder, maxPts);         // uL1+uD
        poolFMM[KERNEL::LapPGradGrad] = new FMMData(KERNEL::LapPGradGrad, pbc, multOrder, maxPts); // uL2
        if (!rank)
            std::cout << "enable Stokes image kernel " << std::endl;
    }

    if (kernelComb & asInteger(KERNEL::RPY)) {
        // RPY image, activate RPY, Laplace, & LapQuad kernels
        poolFMM[KERNEL::RPY] = new FMMData(KERNEL::RPY, pbc, multOrder, maxPts);                     // uS
        poolFMM[KERNEL::LapPGrad] = new FMMData(KERNEL::LapPGrad, pbc, multOrder, maxPts);           // phiSZ+phiDZ
        poolFMM[KERNEL::LapPGradGrad] = new FMMData(KERNEL::LapPGradGrad, pbc, multOrder, maxPts);   // phiS+phiD
        poolFMM[KERNEL::LapQPGradGrad] = new FMMData(KERNEL::LapQPGradGrad, pbc, multOrder, maxPts); // phibQ
        std::cout << "enable RPY image kernel " << std::endl;
    }

    if (poolFMM.empty()) {
        printf("Error: no kernel activated\n");
    }
}

StkWallFMM::~StkWallFMM() {
    // delete all FMMData
    for (auto &fmm : poolFMM) {
        safeDeletePtr(fmm.second);
    }
}

void StkWallFMM::setPoints(const int nSL, const double *srcSLCoordPtr, const int nTrg, const double *trgCoordPtr,
                           const int nDL, const double *srcDLCoordPtr) {}

void StkWallFMM::setupTree(KERNEL kernel) {}

void StkWallFMM::evaluateFMM(const KERNEL kernel, const int nSL, const double *srcSLValuePtr, const int nTrg,
                             double *trgValuePtr, const int nDL, const double *srcDLValuePtr) {}

void StkWallFMM::clearFMM(KERNEL kernel) {}

} // namespace stkfmm