#include "STKFMM/STKFMM.hpp"
#include <stdlib.h>

namespace stkfmm {

Stk3DFMM::Stk3DFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_, bool enableFF_)
    : STKFMM(multOrder_, maxPts_, pbc_, kernelComb_) {
    using namespace impl;
    poolFMM.clear();

    for (const auto &it : kernelMap) {
        const auto kernel = it.first;
        if (kernelComb & asInteger(kernel)) {
            poolFMM[kernel] = new FMMData(kernel, pbc, multOrder, maxPts, enableFF_);
            if (!rank)
                std::cout << "enable kernel " << it.second->ker_name << std::endl;
        }
    }

    if (poolFMM.empty()) {
        std::cout << "Error: no kernel activated\n";
    }
}

Stk3DFMM::~Stk3DFMM() {
    // delete all FMMData
    for (auto &fmm : poolFMM) {
        safeDeletePtr(fmm.second);
    }
}

void Stk3DFMM::setPoints(const int nSL, const double *srcSLCoordPtr, const int nTrg, const double *trgCoordPtr,
                         const int nDL, const double *srcDLCoordPtr) {

    if (!poolFMM.empty()) {
        for (auto &fmm : poolFMM) {
            // if (rank == 0)
            //     printf("kernel %u \n", asInteger(fmm.second->kernelChoice));
            fmm.second->deleteTree();
        }
        if (stkfmm::verbose && rank == 0)
            std::cout << "ALL FMM Tree Cleared\n";
    }

    // setup point coordinates
    auto setCoord = [&](const int nPts, const double *coordPtr, std::vector<double> &coord) {
        coord.resize(nPts * 3);
        std::copy(coordPtr, coordPtr + 3 * nPts, coord.begin());
        scaleCoord(nPts, coord.data());
        wrapCoord(nPts, coord.data());
    };

#pragma omp parallel sections
    {
#pragma omp section
        { setCoord(nSL, srcSLCoordPtr, srcSLCoordInternal); }
#pragma omp section
        {
            if (nDL > 0 && srcDLCoordPtr != nullptr)
                setCoord(nDL, srcDLCoordPtr, srcDLCoordInternal);
        }
#pragma omp section
        { setCoord(nTrg, trgCoordPtr, trgCoordInternal); }
    }

    if (stkfmm::verbose && rank == 0)
        std::cout << "points set\n";
}

void Stk3DFMM::setupTree(KERNEL kernel) {
    auto &fmmPtr = poolFMM[kernel];
    if (fmmPtr->hasDL()) {
        poolFMM[kernel]->setupTree(srcSLCoordInternal, srcDLCoordInternal, trgCoordInternal);
    } else {
        std::vector<double> empty;
        poolFMM[kernel]->setupTree(srcSLCoordInternal, empty, trgCoordInternal);
    }
}

void Stk3DFMM::evaluateFMM(const KERNEL kernel, const int nSL, const double *srcSLValuePtr, const int nTrg,
                           double *trgValuePtr, const int nDL, const double *srcDLValuePtr) {

    using namespace impl;
    if (poolFMM.find(kernel) == poolFMM.end()) {
        std::cout << "Error: no such FMMData exists for kernel " << getKernelName(kernel) << std::endl;
        exit(1);
    }
    FMMData &fmm = *((*poolFMM.find(kernel)).second);

    srcSLValueInternal.resize(nSL * fmm.kdimSL);
    trgValueInternal.resize(nTrg * fmm.kdimTrg);
    std::copy(srcSLValuePtr, srcSLValuePtr + nSL * fmm.kdimSL, srcSLValueInternal.begin());

    // run FMM with proper scaling
    if (fmm.hasDL()) {
        srcDLValueInternal.resize(nDL * fmm.kdimDL);
        std::copy(srcDLValuePtr, srcDLValuePtr + nDL * fmm.kdimDL, srcDLValueInternal.begin());
        fmm.evaluateFMM(srcSLValueInternal, srcDLValueInternal, trgValueInternal, scaleFactor);
    } else {
        std::vector<double> empty(0);
        fmm.evaluateFMM(srcSLValueInternal, empty, trgValueInternal, scaleFactor);
    }

    const int nloop = nTrg * fmm.kdimTrg;
#pragma omp parallel for
    for (int i = 0; i < nloop; i++) {
        trgValuePtr[i] += trgValueInternal[i];
    }

    return;
}

void Stk3DFMM::clearFMM(KERNEL kernel) {
    trgValueInternal.clear();
    auto it = poolFMM.find(kernel);
    if (it != poolFMM.end())
        it->second->clear();
    else {
        std::cout << "kernel not found\n";
        std::exit(1);
    }
}

} // namespace stkfmm
