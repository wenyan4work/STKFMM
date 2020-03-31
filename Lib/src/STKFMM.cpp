/*
 * STKFMM.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: wyan
 */

#include <bitset>
#include <cassert>

#include <mpi.h>
#include <omp.h>

#include "STKFMM/STKFMM.hpp"

#include "STKFMM/LaplaceLayerKernel.hpp"
#include "STKFMM/RPYKernel.hpp"
#include "STKFMM/StokesLayerKernel.hpp"
#include "STKFMM/StokesRegSingleLayerKernel.hpp"

extern pvfmm::PeriodicType pvfmm::periodicType;

namespace stkfmm {

const std::unordered_map<KERNEL, const pvfmm::Kernel<double> *> kernelMap = {
    {KERNEL::LAPPGrad, &pvfmm::LaplaceLayerKernel<double>::PGrad()},
    {KERNEL::Stokes, &pvfmm::StokesKernel<double>::velocity()},
    {KERNEL::RPY, &pvfmm::RPYKernel<double>::ulapu()},
    {KERNEL::StokesRegVel, &pvfmm::StokesRegKernel<double>::Vel()},
    {KERNEL::StokesRegVelOmega, &pvfmm::StokesRegKernel<double>::FTVelOmega()},
    {KERNEL::PVel, &pvfmm::StokesLayerKernel<double>::PVel()},
    {KERNEL::PVelGrad, &pvfmm::StokesLayerKernel<double>::PVelGrad()},
    {KERNEL::PVelLaplacian, &pvfmm::StokesLayerKernel<double>::PVelLaplacian()},
    {KERNEL::Traction, &pvfmm::StokesLayerKernel<double>::Traction()},
};

namespace impl {

void FMMData::setKernel() {
    matrixPtr->Initialize(multOrder, comm, kernelFunctionPtr);
    kdimSL = kernelFunctionPtr->k_s2t->ker_dim[0];
    kdimTrg = kernelFunctionPtr->k_s2t->ker_dim[1];
    kdimDL = kernelFunctionPtr->surf_dim;
}

const pvfmm::Kernel<double> *FMMData::getKernelFunction(KERNEL kernelChoice_) {
    auto it = kernelMap.find(kernelChoice_);
    if (it != kernelMap.end()) {
        return it->second;
    } else {
        printf("Error: Kernel not found.\n");
        std::exit(1);
        return nullptr;
    }
}

void FMMData::readM2LMat(const int kDim, const std::string &dataName,
                         std::vector<double> &data) {
    // int size = kDim * (6 * (multOrder - 1) * (multOrder - 1) + 2);
    int size = kDim * equivCoord.size() / 3;
    data.resize(size * size);

    char *pvfmm_dir = getenv("PVFMM_DIR");
    std::string file =
        std::string(pvfmm_dir) + std::string("/pdata/") + dataName;

    std::cout << dataName << " " << size << std::endl;
    FILE *fin = fopen(file.c_str(), "r");
    if (fin == nullptr) {
        std::cout << "M2L data " << dataName << " not found" << std::endl;
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int iread, jread;
            double fread;
            fscanf(fin, "%d %d %lf\n", &iread, &jread, &fread);
            if (i != iread || j != jread) {
                printf("read ij error %d %d\n", i, j);
                exit(1);
            }
            data[i * size + j] = fread;
        }
    }

    fclose(fin);
}

void FMMData::setupM2Ldata() {
    int pbc = static_cast<int>(periodicity);
    std::string M2Lname = kernelFunctionPtr->k_m2l->ker_name;
    if (!M2Lname.compare(std::string("stokes_PVel"))) {
        // compose M2L data from Laplace and stokes
        std::vector<double> M2L_laplace;
        std::vector<double> M2L_stokes;
        {
            std::string dataName = "M2L_laplace_" + std::to_string(pbc) +
                                   "D3Dp" + std::to_string(multOrder);
            readM2LMat(1, dataName, M2L_laplace);
        }
        {

            std::string dataName = "M2L_stokes_vel_" + std::to_string(pbc) +
                                   "D3Dp" + std::to_string(multOrder);
            readM2LMat(3, dataName, M2L_stokes);
        }
        int nequiv = this->equivCoord.size() / 3;
        M2Ldata.resize(4 * nequiv * 4 * nequiv);
        for (int i = 0; i < nequiv; i++) {
            for (int j = 0; j < nequiv; j++) {
                // each 4x4 block consists of 3x3 of stokes and 1x1 of laplace
                // top-left 4*i, 4*j, size 4x4
                M2Ldata[(4 * i + 3) * (4 * nequiv) + 4 * j + 3] =
                    M2L_laplace[i * nequiv + j];
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        M2Ldata[(4 * i + k) * (4 * nequiv) + 4 * j + l] =
                            M2L_stokes[(3 * i + k) * (3 * nequiv) + 3 * j + l];
                    }
                }
            }
        }

    } else {
        // read M2L data directly
        std::string dataName = "M2L_" + M2Lname + "_" + std::to_string(pbc) +
                               "D3Dp" + std::to_string(multOrder);
        int kdim = kernelFunctionPtr->k_m2l->ker_dim[0];
        readM2LMat(kdim, dataName, this->M2Ldata);
    }
}

// constructor
FMMData::FMMData(KERNEL kernelChoice_, PAXIS periodicity_, int multOrder_,
                 int maxPts_)
    : kernelChoice(kernelChoice_), periodicity(periodicity_),
      multOrder(multOrder_), maxPts(maxPts_), treePtr(nullptr),
      matrixPtr(nullptr), treeDataPtr(nullptr) {

    comm = MPI_COMM_WORLD;
    matrixPtr = new pvfmm::PtFMM<double>();

    // choose a kernel
    kernelFunctionPtr = getKernelFunction(kernelChoice);
    setKernel();
    treeDataPtr = new pvfmm::PtFMM_Data<double>;
    // treeDataPtr remain nullptr after constructor

    // load periodicity M2L data
    if (periodicity != PAXIS::NONE) {
        // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
        // RAD1 = 2.95 defined in pvfmm_common.h
        double scaleLEquiv = PVFMM_RAD1;
        double pCenterLEquiv[3];
        pCenterLEquiv[0] = -(scaleLEquiv - 1) / 2;
        pCenterLEquiv[1] = -(scaleLEquiv - 1) / 2;
        pCenterLEquiv[2] = -(scaleLEquiv - 1) / 2;

        equivCoord =
            surface(multOrder, (double *)&(pCenterLEquiv[0]), scaleLEquiv, 0);

        setupM2Ldata();
    }
}

FMMData::~FMMData() {
    clear();
    safeDeletePtr(treePtr);
    safeDeletePtr(treeDataPtr);
    safeDeletePtr(matrixPtr);
}

void FMMData::clear() {
    //    treeDataPtr->Clear();
    if (treePtr != nullptr)
        treePtr->ClearFMMData();
    return;
}

void FMMData::setupTree(const std::vector<double> &srcSLCoord,
                        const std::vector<double> &srcDLCoord,
                        const std::vector<double> &trgCoord) {
    // trgCoord and srcCoord have been scaled to [0,1)^3

    // setup treeData
    treeDataPtr->dim = 3;
    treeDataPtr->max_depth =
        PVFMM_MAX_DEPTH; // must <= MAX_DEPTH in pvfmm_common.hpp
    treeDataPtr->max_pts = maxPts;

    treeDataPtr->src_coord = srcSLCoord;
    treeDataPtr->surf_coord = srcDLCoord;
    treeDataPtr->trg_coord = trgCoord;

    // this is used to setup FMM octree
    treeDataPtr->pt_coord =
        srcSLCoord.size() > trgCoord.size() ? srcSLCoord : trgCoord;
    const int nSL = srcSLCoord.size() / 3;
    const int nDL = srcDLCoord.size() / 3;
    const int nTrg = trgCoord.size() / 3;

    printf("nSL %d, nDL %d, nTrg %d\n", nSL, nDL, nTrg);

    // space allocate
    treeDataPtr->src_value.Resize(nSL * kdimSL);
    treeDataPtr->surf_value.Resize(nDL * kdimDL);
    treeDataPtr->trg_value.Resize(nTrg * kdimTrg);

    // construct tree
    treePtr = new pvfmm::PtFMM_Tree<double>(comm);
    // printf("tree alloc\n");
    treePtr->Initialize(treeDataPtr);
    // printf("tree init\n");
    treePtr->InitFMM_Tree(true, pvfmm::periodicType == pvfmm::PeriodicType::NONE
                                    ? pvfmm::FreeSpace
                                    : pvfmm::Periodic);
    // printf("tree build\n");
    treePtr->SetupFMM(matrixPtr);
    // printf("tree fmm matrix setup\n");
    return;
}

void FMMData::deleteTree() {
    clear();
    safeDeletePtr(treePtr);
    return;
}

void FMMData::evaluateFMM(std::vector<double> &srcSLValue,
                          std::vector<double> &srcDLValue,
                          std::vector<double> &trgValue) {
    const int nSrc = treeDataPtr->src_coord.Dim() / 3;
    const int nSurf = treeDataPtr->surf_coord.Dim() / 3;
    const int nTrg = treeDataPtr->trg_coord.Dim() / 3;

    int rank;
    MPI_Comm_rank(comm, &rank);

    if (nTrg * kdimTrg != trgValue.size()) {
        printf("trg value size error from rank %d\n", rank);
        exit(1);
    }
    if (nSrc * kdimSL != srcSLValue.size()) {
        printf("src SL value size error from rank %d\n", rank);
        exit(1);
    }
    if (nSurf * kdimDL != srcDLValue.size()) {
        printf("src DL value size error from rank %d\n", rank);
        exit(1);
    }
    PtFMM_Evaluate(treePtr, trgValue, nTrg, &srcSLValue, &srcDLValue);
    periodizeFMM(trgValue);
}

void FMMData::periodizeFMM(std::vector<double> &trgValue) {
    if (periodicity == PAXIS::NONE) {
        return;
    }

    // the value calculated by pvfmm
    pvfmm::Vector<double> v = treePtr->RootNode()->FMMData()->upward_equiv;

    // add to trg_value
    auto &trgCoord = treeDataPtr->trg_coord;
    const int nTrg = trgCoord.Dim() / 3;
    const int equivN = equivCoord.size() / 3;

    int kDim = kernelFunctionPtr->k_m2l->ker_dim[0];
    int M = kDim * equivN;
    int N = kDim * equivN; // checkN = equivN in this code.
    std::vector<double> M2Lsource(v.Dim());

#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        double temp = 0;
        for (int j = 0; j < N; j++) {
            temp += M2Ldata[i * N + j] * v[j];
        }
        M2Lsource[i] = temp;
    }

    // L2T evaluation with openmp
    evaluateKernel(-1, PPKERNEL::L2T, equivN, equivCoord.data(),
                   M2Lsource.data(), nTrg, trgCoord.Begin(), trgValue.data());
}

void FMMData::evaluateKernel(int nThreads, PPKERNEL p2p, const int nSrc,
                             double *srcCoordPtr, double *srcValuePtr,
                             const int nTrg, double *trgCoordPtr,
                             double *trgValuePtr) {
    if (nThreads < 1 || nThreads > omp_get_max_threads()) {
        nThreads = omp_get_max_threads();
    }

    const int chunkSize = 4000; // each chunk has some target points.
    const int chunkNumber = floor(1.0 * (nTrg) / chunkSize) + 1;

    pvfmm::Kernel<double>::Ker_t kerPtr = nullptr; // a function pointer
    if (p2p == PPKERNEL::SLS2T) {
        kerPtr = kernelFunctionPtr->k_s2t->ker_poten;
    } else if (p2p == PPKERNEL::DLS2T) {
        kerPtr = kernelFunctionPtr->k_s2t->dbl_layer_poten;
    } else if (p2p == PPKERNEL::L2T) {
        kerPtr = kernelFunctionPtr->k_l2t->ker_poten;
    }

    if (kerPtr == nullptr) {
        std::cout << "PPKernel " << (uint)p2p
                  << " not found for direct evaluation" << std::endl;
        return;
    }

#pragma omp parallel for schedule(static, 1) num_threads(nThreads)
    for (int i = 0; i < chunkNumber; i++) {
        // each thread process one chunk
        const int idTrgLow = i * chunkSize;
        const int idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize
                                                    : nTrg; // not inclusive
        kerPtr(srcCoordPtr, nSrc, srcValuePtr, 1, trgCoordPtr + 3 * idTrgLow,
               idTrgHigh - idTrgLow, trgValuePtr + kdimTrg * idTrgLow, NULL);
    }
}
} // namespace impl

STKFMM::STKFMM(int multOrder_, int maxPts_, PAXIS pbc_,
               unsigned int kernelComb_)
    : multOrder(multOrder_), maxPts(maxPts_), pbc(pbc_),
      kernelComb(kernelComb_), xlow(0), xhigh(1), ylow(0), yhigh(1), zlow(0),
      zhigh(1), scaleFactor(1), xshift(0), yshift(0), zshift(0) {
    using namespace impl;
    // set periodic boundary condition
    switch (pbc) {
    case PAXIS::NONE:
        pvfmm::periodicType = pvfmm::PeriodicType::NONE;
        break;
    case PAXIS::PX:
        pvfmm::periodicType = pvfmm::PeriodicType::PX;
        break;
    case PAXIS::PXY:
        pvfmm::periodicType = pvfmm::PeriodicType::PXY;
        break;
    case PAXIS::PXYZ:
        pvfmm::periodicType = pvfmm::PeriodicType::PXYZ;
        break;
    }

    comm = MPI_COMM_WORLD;
    int myRank;
    MPI_Comm_rank(comm, &myRank);

    poolFMM.clear();

    for (const auto &it : kernelMap) {
        const auto kernel = it.first;
        if (kernelComb & asInteger(kernel)) {
            poolFMM[kernel] = new FMMData(kernel, pbc, multOrder, maxPts);
            if (!myRank)
                std::cout << "enable kernel " << it.second->ker_name
                          << std::endl;
        }
    }

    if (poolFMM.empty()) {
        printf("Error: no kernel activated\n");
        exit(1);
    }

#ifdef FMMDEBUG
    pvfmm::Profile::Enable(true);
    if (myRank == 0)
        printf("FMM Initialized\n");
#endif
}

STKFMM::~STKFMM() {
    // delete all FMMData
    for (auto &fmm : poolFMM) {
        safeDeletePtr(fmm.second);
    }
}

void STKFMM::setBox(double xlow_, double xhigh_, double ylow_, double yhigh_,
                    double zlow_, double zhigh_) {
    xlow = xlow_;
    xhigh = xhigh_;
    ylow = ylow_;
    yhigh = yhigh_;
    zlow = zlow_;
    zhigh = zhigh_;

    // find and calculate scale & shift factor to map the box to [0,1)
    xshift = -xlow;
    yshift = -ylow;
    zshift = -zlow;
    double xlen = xhigh - xlow;
    double ylen = yhigh - ylow;
    double zlen = zhigh - zlow;
    scaleFactor = 1 / std::max(zlen, std::max(xlen, ylen));
    // new coordinate = (x+xshift)*scaleFactor, in [0,1)

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "box x " << xlen << " box y " << ylen << " box z " << zlen
                  << std::endl;
        std::cout << "scale factor " << scaleFactor << std::endl;
    }

    // sanity check of box setting, ensure fitting in a cubic box [0,1)^3
    const double eps = pow(10, -12) / scaleFactor;
    if (abs(xlen - ylen) > eps || abs(xlen - zlen) > eps ||
        abs(ylen - zlen) > eps) {
        printf("Error: box must be a cube\n");
        std::exit(1);
    }
}

void STKFMM::setupCoord(const int npts, const double *coordInPtr,
                        std::vector<double> &coord) const {
    // scale points into internal data array, without rotation
    // Set points
    if (npts == 0) {
        coord.clear();
        return;
    }
    coord.resize(npts * 3);

    const double xs = this->xshift;
    const double ys = this->yshift;
    const double zs = this->zshift;
    const double sF = this->scaleFactor;

// scale
#pragma omp parallel for
    for (int i = 0; i < npts; i++) {
        coord[3 * i + 0] = (coordInPtr[3 * i + 0] + xs) * sF;
        coord[3 * i + 1] = (coordInPtr[3 * i + 1] + ys) * sF;
        coord[3 * i + 2] = (coordInPtr[3 * i + 2] + zs) * sF;
    }

    // wrap periodic images
    if (pbc == PAXIS::PX) {
#pragma omp parallel for
        for (int i = 0; i < npts; i++) {
            coord[3 * i] = fracwrap(coord[3 * i]);
        }
    } else if (pbc == PAXIS::PXY) {
#pragma omp parallel for
        for (int i = 0; i < npts; i++) {
            coord[3 * i] = fracwrap(coord[3 * i]);
            coord[3 * i + 1] = fracwrap(coord[3 * i + 1]);
        }
    } else if (pbc == PAXIS::PXYZ) {
#pragma omp parallel for
        for (int i = 0; i < npts; i++) {
            coord[3 * i] = fracwrap(coord[3 * i]);
            coord[3 * i + 1] = fracwrap(coord[3 * i + 1]);
            coord[3 * i + 2] = fracwrap(coord[3 * i + 2]);
        }
    } else {
        assert(pbc == PAXIS::NONE);
        // no fracwrap
    }

    return;
}

void STKFMM::setPoints(const int nSL, const double *srcSLCoordPtr,
                       const int nDL, const double *srcDLCoordPtr,
                       const int nTrg, const double *trgCoordPtr) {
    int np, myRank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &myRank);

    if (!poolFMM.empty()) {
        for (auto &fmm : poolFMM) {
            if (myRank == 0)
                printf("kernel %u \n", asInteger(fmm.second->kernelChoice));
            fmm.second->deleteTree();
        }
        if (myRank == 0)
            printf("ALL FMM Tree Cleared\n");
    }

    // setup point coordinates
    setupCoord(nSL, srcSLCoordPtr, srcSLCoordInternal);
    setupCoord(nDL, srcDLCoordPtr, srcDLCoordInternal);
    setupCoord(nTrg, trgCoordPtr, trgCoordInternal);
    if (myRank == 0)
        printf("points set\n");
}

void STKFMM::setupTree(KERNEL kernel_) {
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    poolFMM[kernel_]->setupTree(srcSLCoordInternal, srcDLCoordInternal,
                                trgCoordInternal);
    if (myRank == 0)
        printf("Coord setup for kernel %d\n", static_cast<int>(kernel_));
}

void STKFMM::evaluateFMM(const int nSL, const double *srcSLValuePtr,
                         const int nDL, const double *srcDLValuePtr,
                         const int nTrg, double *trgValuePtr,
                         const KERNEL kernel) {

    using namespace impl;
    if (poolFMM.find(kernel) == poolFMM.end()) {
        printf("Error: no such FMMData exists for kernel %d\n",
               static_cast<int>(kernel));
        exit(1);
    }
    FMMData &fmm = *((*poolFMM.find(kernel)).second);

    srcSLValueInternal.resize(nSL * fmm.kdimSL);
    srcDLValueInternal.resize(nDL * fmm.kdimDL);

    // scale the source strength, SL as 1/r, DL as 1/r^2
    // SL no extra scaling
    // DL scale as scaleFactor
    std::copy(srcSLValuePtr, srcSLValuePtr + nSL * fmm.kdimSL,
              srcSLValueInternal.begin());
    int nloop = nDL * fmm.kdimDL;
#pragma omp parallel for
    for (int i = 0; i < nloop; i++) {
        srcDLValueInternal[i] = srcDLValuePtr[i] * scaleFactor;
    }

    if (fmm.kdimSL == 4) {
        // Stokes, RPY, StokesRegVel
#pragma omp parallel for
        for (int i = 0; i < nSL; i++) {
            // the Trace term scales as double layer, epsilon terms of
            // RPY/StokesRegVel length scale as well
            srcSLValueInternal[4 * i + 3] *= scaleFactor;
        }
    }
    if (kernel == KERNEL::StokesRegVelOmega) {
#pragma omp parallel for
        for (int i = 0; i < nSL; i++) {
            // Scale torque / epsilon
            for (int j = 3; j < 7; ++j)
                srcSLValueInternal[7 * i + j] *= scaleFactor;
        }
    }

    // run FMM
    // evaluate on internal sources with proper scaling
    trgValueInternal.resize(nTrg * fmm.kdimTrg);
    fmm.evaluateFMM(srcSLValueInternal, srcDLValueInternal, trgValueInternal);

    // scale back according to kernel
    switch (kernel) {
    case KERNEL::PVel: {
        // 1+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // pressure 1/r^2
            trgValuePtr[4 * i] +=
                trgValueInternal[4 * i] * scaleFactor * scaleFactor;
            // vel 1/r
            trgValuePtr[4 * i + 1] += trgValueInternal[4 * i + 1] * scaleFactor;
            trgValuePtr[4 * i + 2] += trgValueInternal[4 * i + 2] * scaleFactor;
            trgValuePtr[4 * i + 3] += trgValueInternal[4 * i + 3] * scaleFactor;
        }
    } break;
    case KERNEL::PVelGrad: {
        // 1+3+3+9
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p
            trgValuePtr[16 * i] +=
                trgValueInternal[16 * i] * scaleFactor * scaleFactor;
            // vel
            for (int j = 1; j < 4; j++) {
                trgValuePtr[16 * i + j] +=
                    trgValueInternal[16 * i + j] * scaleFactor;
            }
            // grad p
            for (int j = 4; j < 7; j++) {
                trgValuePtr[16 * i + j] += trgValueInternal[16 * i + j] *
                                           scaleFactor * scaleFactor *
                                           scaleFactor;
            }
            // grad vel
            for (int j = 7; j < 16; j++) {
                trgValuePtr[16 * i + j] +=
                    trgValueInternal[16 * i + j] * scaleFactor * scaleFactor;
            }
        }
    } break;
    case KERNEL::Traction: {
        // 9
        int nloop = 9 * nTrg;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValuePtr[i] += trgValueInternal[i] * scaleFactor *
                              scaleFactor; // traction 1/r^2
        }
    } break;
    case KERNEL::PVelLaplacian: {
        // 1+3+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p
            trgValuePtr[7 * i] +=
                trgValueInternal[7 * i] * scaleFactor * scaleFactor;
            // vel
            for (int j = 1; j < 4; j++) {
                trgValuePtr[7 * i + j] +=
                    trgValueInternal[7 * i + j] * scaleFactor;
            }
            // laplacian vel
            for (int j = 4; j < 7; j++) {
                trgValuePtr[7 * i + j] += trgValueInternal[7 * i + j] *
                                          scaleFactor * scaleFactor *
                                          scaleFactor;
            }
        }
    } break;
    case KERNEL::LAPPGrad: {
        // 1+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p, 1/r
            trgValuePtr[4 * i] += trgValueInternal[4 * i] * scaleFactor;
            // grad p, 1/r^2
            for (int j = 1; j < 4; j++) {
                trgValuePtr[4 * i + j] +=
                    trgValueInternal[4 * i + j] * scaleFactor * scaleFactor;
            }
        }
    } break;
    case KERNEL::Stokes: {
        // 3
        const int nloop = nTrg * 3;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValuePtr[i] += trgValueInternal[i] * scaleFactor; // vel 1/r
        }
    } break;
    case KERNEL::StokesRegVel: {
        // 3
        const int nloop = nTrg * 3;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValuePtr[i] += trgValueInternal[i] * scaleFactor; // vel 1/r
        }
    } break;
    case KERNEL::StokesRegVelOmega: {
        // 3 + 3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // vel 1/r
            for (int j = 0; j < 3; ++j)
                trgValuePtr[i * 6 + j] +=
                    trgValueInternal[i * 6 + j] * scaleFactor;
            // omega 1/r^2
            for (int j = 3; j < 6; ++j)
                trgValuePtr[i * 6 + j] +=
                    trgValueInternal[i * 6 + j] * scaleFactor * scaleFactor;
        }
    } break;
    case KERNEL::RPY: {
        // 3 + 3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // vel 1/r
            for (int j = 0; j < 3; ++j)
                trgValuePtr[i * 6 + j] +=
                    trgValueInternal[i * 6 + j] * scaleFactor;
            // laplacian vel
            for (int j = 3; j < 6; ++j)
                trgValuePtr[i * 6 + j] += trgValueInternal[i * 6 + j] *
                                          scaleFactor * scaleFactor *
                                          scaleFactor;
        }
    } break;
    }

    return;
}

void STKFMM::evaluateKernel(const int nThreads, const PPKERNEL p2p,
                            const int nSrc, double *srcCoordPtr,
                            double *srcValuePtr, const int nTrg,
                            double *trgCoordPtr, double *trgValuePtr,
                            const KERNEL kernel) {
    using namespace impl;
    if (poolFMM.find(kernel) == poolFMM.end()) {
        printf("Error: no such FMMData exists for kernel %d\n",
               static_cast<int>(kernel));
        exit(1);
    }
    FMMData &fmm = *((*poolFMM.find(kernel)).second);

    fmm.evaluateKernel(nThreads, p2p, nSrc, srcCoordPtr, srcValuePtr, nTrg,
                       trgCoordPtr, trgValuePtr);
}

void STKFMM::showActiveKernels() {
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    for (auto it : kernelMap) {
        if (kernelComb & asInteger(it.first)) {
            std::cout << it.second->ker_name;
        }
    }
}

std::tuple<int, int, int> STKFMM::getKernelDimension(KERNEL kernel_) {
    using namespace impl;
    const pvfmm::Kernel<double> *kernelFunctionPtr =
        FMMData::getKernelFunction(kernel_);
    int kdimSL = kernelFunctionPtr->ker_dim[0];
    int kdimTrg = kernelFunctionPtr->ker_dim[1];
    int kdimDL = kernelFunctionPtr->surf_dim;
    return std::tuple<int, int, int>(kdimSL, kdimDL, kdimTrg);
}

void STKFMM::clearFMM(KERNEL kernelChoice) {
    trgValueInternal.clear();
    poolFMM[kernelChoice]->clear();
}
} // namespace stkfmm
