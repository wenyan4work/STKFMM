#include "STKFMM/STKFMM.hpp"

namespace stkfmm {
StkWallFMM::StkWallFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_)
    : STKFMM(multOrder_, maxPts_, pbc_, kernelComb_) {
    using namespace impl;
    poolFMM.clear();

    if (kernelComb & asInteger(KERNEL::Stokes)) {
        // Stokes image, activate Stokes & Laplace kernels
        poolFMM[KERNEL::Stokes] = new FMMData(KERNEL::Stokes, pbc, multOrder, maxPts);
        poolFMM[KERNEL::LapPGrad] = new FMMData(KERNEL::LapPGrad, pbc, multOrder, maxPts);
        if (!rank)
            std::cout << "enable Stokes image kernel " << std::endl;
    }

    if (kernelComb & asInteger(KERNEL::PVel)) {
        // RPY image, activate RPY, Laplace, & LapQuad kernels
        poolFMM[KERNEL::Stokes] = new FMMData(KERNEL::Stokes, pbc, multOrder, maxPts);
        poolFMM[KERNEL::LapPGrad] = new FMMData(KERNEL::LapPGrad, pbc, multOrder, maxPts);
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
}