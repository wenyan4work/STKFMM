#include "STKFMM/STKFMM.hpp"

// extern pvfmm::PeriodicType pvfmm::periodicType;

namespace stkfmm {

bool get_verbosity() {
    char *verbose_env;
    verbose_env = getenv("STKFMM_VERBOSE");
    if (verbose_env == nullptr || verbose_env[0] == '0')
        return false;

    return true;
}

const bool verbose = get_verbosity();

const std::unordered_map<KERNEL, const pvfmm::Kernel<double> *> kernelMap = {
    {KERNEL::LapPGrad, &pvfmm::LaplaceLayerKernel<double>::PGrad()},
    {KERNEL::LapPGradGrad, &pvfmm::LaplaceLayerKernel<double>::PGradGrad()},
    {KERNEL::LapQPGradGrad, &pvfmm::LaplaceLayerKernel<double>::QPGradGrad()},
    {KERNEL::Stokes, &pvfmm::StokesLayerKernel<double>::Vel()},
    {KERNEL::RPY, &pvfmm::RPYKernel<double>::ulapu()},
    {KERNEL::StokesRegVel, &pvfmm::StokesRegKernel<double>::Vel()},
    {KERNEL::StokesRegVelOmega, &pvfmm::StokesRegKernel<double>::FTVelOmega()},
    {KERNEL::PVel, &pvfmm::StokesLayerKernel<double>::PVel()},
    {KERNEL::PVelGrad, &pvfmm::StokesLayerKernel<double>::PVelGrad()},
    {KERNEL::PVelLaplacian, &pvfmm::StokesLayerKernel<double>::PVelLaplacian()},
    {KERNEL::Traction, &pvfmm::StokesLayerKernel<double>::Traction()},
    // {KERNEL::LapGrad, &pvfmm::LaplaceLayerKernel<double>::Grad()}, // for internal test only
};

std::tuple<int, int, int> getKernelDimension(KERNEL kernel_) {
    using namespace impl;
    const pvfmm::Kernel<double> *kernelFunctionPtr = getKernelFunction(kernel_);
    int kdimSL = kernelFunctionPtr->ker_dim[0];
    int kdimTrg = kernelFunctionPtr->ker_dim[1];
    int kdimDL = kernelFunctionPtr->surf_dim;
    return std::tuple<int, int, int>(kdimSL, kdimDL, kdimTrg);
}

std::string getKernelName(KERNEL kernel_) {
    using namespace impl;
    const pvfmm::Kernel<double> *kernelFunctionPtr = getKernelFunction(kernel_);
    return kernelFunctionPtr->ker_name;
}

const pvfmm::Kernel<double> *getKernelFunction(KERNEL kernelChoice_) {
    auto it = kernelMap.find(kernelChoice_);
    if (it != kernelMap.end()) {
        return it->second;
    } else {
        printf("Error: Kernel not found.\n");
        std::exit(1);
        return nullptr;
    }
}

// base class STKFMM

STKFMM::STKFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_, bool enableFF_)
    : multOrder(multOrder_), maxPts(maxPts_), pbc(pbc_), kernelComb(kernelComb_) {
    using namespace impl;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef FMMDEBUG
    pvfmm::Profile::Enable(true);
    if (myRank == 0)
        printf("FMM Initialized\n");
#endif
}

void STKFMM::setBox(double origin_[3], double len_) {
    origin[0] = origin_[0];
    origin[1] = origin_[1];
    origin[2] = origin_[2];
    len = len_;
    // find and calculate scale & shift factor to map the box to [0,1)
    scaleFactor = 1.0 / len;
    // new coordinate = (pos-origin)*scaleFactor, in [0,1)

    if (stkfmm::verbose && rank == 0) {
        std::cout << "scale factor " << scaleFactor << std::endl;
    }
};

void STKFMM::evaluateKernel(const KERNEL kernel, const int nThreads, const PPKERNEL p2p, const int nSrc,
                            double *srcCoordPtr, double *srcValuePtr, const int nTrg, double *trgCoordPtr,
                            double *trgValuePtr) {
    using namespace impl;
    if (poolFMM.find(kernel) == poolFMM.end()) {
        std::cout << "Error: no such FMMData exists for kernel " << getKernelName(kernel) << std::endl;
        exit(1);
    }
    FMMData &fmm = *((*poolFMM.find(kernel)).second);

    fmm.evaluateKernel(nThreads, p2p, nSrc, srcCoordPtr, srcValuePtr, nTrg, trgCoordPtr, trgValuePtr);
}

void STKFMM::showActiveKernels() const {
    if (!rank) {
        std::cout << "active kernels: ";
        for (auto it : kernelMap) {
            if (kernelComb & asInteger(it.first)) {
                std::cout << "\t" << it.second->ker_name;
            }
        }
        std::cout << std::endl;
    }
}

void STKFMM::scaleCoord(const int npts, double *coordPtr) const {
    // scale and shift points to [0,1)
    const double sF = this->scaleFactor;

#pragma omp parallel for
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < 3; j++) {
            coordPtr[3 * i + j] = (coordPtr[3 * i + j] - origin[j]) * sF;
        }
    }
}

void STKFMM::wrapCoord(const int npts, double *coordPtr) const {
    // wrap periodic images
    if (pbc == PAXIS::PX) {
#pragma omp parallel for
        for (int i = 0; i < npts; i++) {
            fracwrap(coordPtr[3 * i]);
        }
    } else if (pbc == PAXIS::PXY) {
#pragma omp parallel for
        for (int i = 0; i < npts; i++) {
            fracwrap(coordPtr[3 * i]);
            fracwrap(coordPtr[3 * i + 1]);
        }
    } else if (pbc == PAXIS::PXYZ) {
#pragma omp parallel for
        for (int i = 0; i < npts; i++) {
            fracwrap(coordPtr[3 * i]);
            fracwrap(coordPtr[3 * i + 1]);
            fracwrap(coordPtr[3 * i + 2]);
        }
    } else {
        assert(pbc == PAXIS::NONE);
    }

    return;
}

} // namespace stkfmm
