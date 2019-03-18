/*
 * FMMWrapper.cpp
 *
 *  Created on: Oct 20, 2016
 *      Author: wyan
 */

#include <bitset>
#include <cassert>

#include <mpi.h>
#include <omp.h>

#include "Kernel/LaplaceLayerKernel.hpp"
#include "Kernel/StokesLayerKernel.hpp"

#include "STKFMM.h"

extern pvfmm::PeriodicType pvfmm::periodicType;

namespace stkfmm {

template <class T>
void safeDeletePtr(T *ptr) {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

// return fraction part between [0,1)
/*
 * This function is only applied in the PERIODIC DIRECTION
 * The user of the library must ensure that all points are located within [0,1)
 * */
inline double fracwrap(double x) { return x - floor(x); }

template <class Real_t>
std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
    int n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

    std::vector<Real_t> coord(n_ * 3);
    coord[0] = coord[1] = coord[2] = -1.0;
    int cnt = 1;
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = -1.0;
            coord[cnt * 3 + 1] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = (2.0 * j - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * i - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = -1.0;
            coord[cnt * 3 + 2] = (2.0 * (j + 1) - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = (2.0 * j - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = -1.0;
            cnt++;
        }
    for (int i = 0; i < (n_ / 2) * 3; i++)
        coord[cnt * 3 + i] = -coord[i];

    Real_t r = 0.5 * pow(0.5, depth);
    Real_t b = alpha * r;
    for (int i = 0; i < n_; i++) {
        coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
        coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
        coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
    }
    return coord;
}

void FMMData::setKernel() {
    matrixPtr->Initialize(multOrder, comm, kernelFunctionPtr);
    kdimSL = kernelFunctionPtr->k_s2t->ker_dim[0];
    kdimTrg = kernelFunctionPtr->k_s2t->ker_dim[1];
    kdimDL = kernelFunctionPtr->surf_dim;
}

void FMMData::readM2LMat(const std::string dataName) {
    const int size = 3 * (6 * (multOrder - 1) * (multOrder - 1) + 2);
    double *fdata = new double[size * size];
    M2Ldata.resize(size * size);

    char *pvfmm_dir = getenv("PVFMM_DIR");
    std::stringstream st;
    st << pvfmm_dir;
    st << "/pdata/";
    st << dataName.c_str();

    FILE *fin = fopen(st.str().c_str(), "r");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int iread, jread;
            double fread;
            fscanf(fin, "%d %d %lf\n", &iread, &jread, &fread);
            if (i != iread || j != jread) {
                printf("read ij error \n");
                exit(1);
            }
            fdata[i * size + j] = fread;
        }
    }

    fclose(fin);
}

// constructor
FMMData::FMMData(KERNEL kernelChoice_, PAXIS periodicity_, int multOrder_, int maxPts_)
    : kernelChoice(kernelChoice_), periodicity(periodicity_), multOrder(multOrder_), maxPts(maxPts_), treePtr(nullptr),
      matrixPtr(nullptr), treeDataPtr(nullptr) {
    comm = MPI_COMM_WORLD;
    matrixPtr = new pvfmm::PtFMM<double>();
    // choose a kernel
    switch (kernelChoice) {
    case KERNEL::PVel:
        kernelFunctionPtr = &pvfmm::StokesLayerKernel<double>::PVel();
        break;
    case KERNEL::PVelGrad:
        kernelFunctionPtr = &pvfmm::StokesLayerKernel<double>::PVelGrad();
        break;
    case KERNEL::PVelLaplacian:
        kernelFunctionPtr = &pvfmm::StokesLayerKernel<double>::PVelLaplacian();
        break;
    case KERNEL::Traction:
        kernelFunctionPtr = &pvfmm::StokesLayerKernel<double>::Traction();
        break;
    case KERNEL::LAPPGrad:
        kernelFunctionPtr = &pvfmm::LaplaceLayerKernel<double>::PGrad();
        break;
    }
    setKernel();
    treeDataPtr = new pvfmm::PtFMM_Data<double>;
    // treeDataPtr remain nullptr after constructor

    // load periodicity M2L data
    if (periodicity != PAXIS::NONE) {

        // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
        double scaleLEquiv = PVFMM_RAD1; // RAD1 = 2.95 defined in pvfmm_common.h
        double pCenterLEquiv[3];
        pCenterLEquiv[0] = -(scaleLEquiv - 1) / 2;
        pCenterLEquiv[1] = -(scaleLEquiv - 1) / 2;
        pCenterLEquiv[2] = -(scaleLEquiv - 1) / 2;

        equivCoord = surface(multOrder, (double *)&(pCenterLEquiv[0]), scaleLEquiv, 0);

        if (kernelChoice == KERNEL::LAPPGrad) {
            // load Laplace 1D, 2D, 3D data
            std::string dataName;
            if (periodicity == PAXIS::PX) {
                dataName = "M2LLaplace1D3DpX";
            } else if (periodicity == PAXIS::PXY) {
                dataName = "M2LLaplace2D3DpX";
            } else if (periodicity == PAXIS::PXYZ) {
                dataName = "M2LLaplace3D3DpX";
            }
            dataName.replace(dataName.length() - 1, 1, std::to_string(multOrder));
            std::cout << "reading M2L data: " << dataName << std::endl;
            readM2LMat(dataName);
        } else {
            // load Stokes 1D, 2D, 3D data
            std::string dataName;
            if (periodicity == PAXIS::PX) {
                // TODO: generate Stokes PVel periodicity data
                dataName = "M2LStokesPVel1D3DpX";
            } else if (periodicity == PAXIS::PXY) {
                dataName = "M2LStokesPVel2D3DpX";
            } else if (periodicity == PAXIS::PXYZ) {
                dataName = "M2LStokesPVel3D3DpX";
            }
            dataName.replace(dataName.length() - 1, 1, std::to_string(multOrder));
            std::cout << "reading M2L data: " << dataName << std::endl;
            readM2LMat(dataName);
        }
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

void FMMData::setupTree(const std::vector<double> &srcSLCoord, const std::vector<double> &srcDLCoord,
                        const std::vector<double> &trgCoord) {
    // trgCoord and srcCoord have been scaled to [0,1)^3

    // setup treeData
    treeDataPtr->dim = 3;
    treeDataPtr->max_depth = PVFMM_MAX_DEPTH; // must <= MAX_DEPTH in pvfmm_common.hpp
    treeDataPtr->max_pts = maxPts;

    treeDataPtr->src_coord = srcSLCoord;
    treeDataPtr->surf_coord = srcDLCoord;
    treeDataPtr->trg_coord = trgCoord;

    // this is used to setup FMM octree
    treeDataPtr->pt_coord = srcSLCoord.size() > trgCoord.size() ? srcSLCoord : trgCoord;
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
    treePtr->InitFMM_Tree(true, pvfmm::periodicType == pvfmm::PeriodicType::NONE ? pvfmm::FreeSpace : pvfmm::Periodic);
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

void FMMData::evaluateFMM(std::vector<double> &srcSLValue, std::vector<double> &srcDLValue,
                          std::vector<double> &trgValue) {
    const int nSrc = treeDataPtr->src_coord.Dim() / 3;
    const int nSurf = treeDataPtr->surf_coord.Dim() / 3;
    const int nTrg = treeDataPtr->trg_coord.Dim() / 3;

    int rank;
    MPI_Comm_rank(comm, &rank);

    if (nTrg * kdimTrg != trgValue.size()) {
        printf("trg value size error for kernel %u from rank %d\n", kernelChoice, rank);
        exit(1);
    }
    if (nSrc * kdimSL != srcSLValue.size()) {
        printf("src SL value size error for kernel %u\n from rank %d", kernelChoice, rank);
        exit(1);
    }
    if (nSurf * kdimDL != srcDLValue.size()) {
        printf("src DL value size error for kernel %u\n from rank %d", kernelChoice, rank);
        exit(1);
    }
    PtFMM_Evaluate(treePtr, trgValue, nTrg, &srcSLValue, &srcDLValue);
    periodizeFMM(trgValue);
}

void FMMData::periodizeFMM(std::vector<double> &trgValue) {
    if (periodicity == PAXIS::NONE) {
        return;
    }

    pvfmm::Vector<double> v = treePtr->RootNode()->FMMData()->upward_equiv; // the value calculated by pvfmm
    // assert(v.Dim() == 3 * this->equivN);

    // add to trg_value
    auto &trgCoord = treeDataPtr->trg_coord;
    const int nTrg = trgCoord.Dim() / 3;
    const int equivN = equivCoord.size() / 3;

    int M = 3 * equivN;
    int N = 3 * equivN; // checkN = equivN in this code.
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
    evaluateKernel(-1, PPKERNEL::L2T, equivN, equivCoord.data(), M2Lsource.data(), nTrg, trgCoord.Begin(),
                   trgValue.data());
}

void FMMData::evaluateKernel(int nThreads, PPKERNEL p2p, const int nSrc, double *srcCoordPtr, double *srcValuePtr,
                             const int nTrg, double *trgCoordPtr, double *trgValuePtr) {
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

#pragma omp parallel for schedule(static, 1) num_threads(nThreads)
    for (int i = 0; i < chunkNumber; i++) {
        // each thread process one chunk
        const int idTrgLow = i * chunkSize;
        const int idTrgHigh = (i + 1 < chunkNumber) ? idTrgLow + chunkSize : nTrg; // not inclusive
        kerPtr(srcCoordPtr, nSrc, srcValuePtr, 1, trgCoordPtr + 3 * idTrgLow, idTrgHigh - idTrgLow,
               trgValuePtr + kdimTrg * idTrgLow, NULL);
    }
}

STKFMM::STKFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_)
    : multOrder(multOrder_), maxPts(maxPts_), pbc(pbc_), kernelComb(kernelComb_), xlow(0), xhigh(1), ylow(0), yhigh(1),
      zlow(0), zhigh(1), scaleFactor(1), xshift(0), yshift(0), zshift(0) {
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
    if (pbc != PAXIS::NONE) {
        printf("to be implemented\n");
        exit(1);
    }
    comm = MPI_COMM_WORLD;
    int myRank;
    MPI_Comm_rank(comm, &myRank);

    poolFMM.clear();

    // parse the choice of kernels, use bitwise and
    if (kernelComb & asInteger(KERNEL::PVel)) {
        if (myRank == 0)
            printf("enable PVel %u\n", kernelComb & asInteger(KERNEL::PVel));
        poolFMM[KERNEL::PVel] = new FMMData(KERNEL::PVel, pbc, multOrder, maxPts);
    }
    if (kernelComb & asInteger(KERNEL::PVelGrad)) {
        if (myRank == 0)
            printf("enable PVelGrad %u\n", kernelComb & asInteger(KERNEL::PVelGrad));
        poolFMM[KERNEL::PVelGrad] = new FMMData(KERNEL::PVelGrad, pbc, multOrder, maxPts);
    }
    if (kernelComb & asInteger(KERNEL::PVelLaplacian)) {
        if (myRank == 0)
            printf("enable PVelLaplacian %u\n", kernelComb & asInteger(KERNEL::PVelLaplacian));
        poolFMM[KERNEL::PVelLaplacian] = new FMMData(KERNEL::PVelLaplacian, pbc, multOrder, maxPts);
    }
    if (kernelComb & asInteger(KERNEL::Traction)) {
        if (myRank == 0)
            printf("enable Traction %u\n", kernelComb & asInteger(KERNEL::Traction));
        poolFMM[KERNEL::Traction] = new FMMData(KERNEL::Traction, pbc, multOrder, maxPts);
    }
    if (kernelComb & asInteger(KERNEL::LAPPGrad)) {
        if (myRank == 0)
            printf("enable LAPPGrad %u\n", kernelComb & asInteger(KERNEL::LAPPGrad));
        poolFMM[KERNEL::LAPPGrad] = new FMMData(KERNEL::LAPPGrad, pbc, multOrder, maxPts);
    }

#ifdef FMMDEBUG
    pvfmm::Profile::Enable(true);
#endif

    if (poolFMM.empty()) {
        printf("Error: no kernel choosed");
        exit(1);
    }

    if (myRank == 0)
        printf("FMM Initialized\n");
}

STKFMM::~STKFMM() {
    // delete all FMMData
    for (auto &fmm : poolFMM) {
        safeDeletePtr(fmm.second);
    }
}

void STKFMM::setBox(double xlow_, double xhigh_, double ylow_, double yhigh_, double zlow_, double zhigh_) {
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
        std::cout << "box x " << xlen << " box y " << ylen << " box z " << zlen << std::endl;
        std::cout << "scale factor " << scaleFactor << std::endl;
    }

    // sanity check of box setting, ensure fitting in a cubic box [0,1)^3
    const double eps = pow(10, -12) / scaleFactor;
    switch (pbc) {
    case PAXIS::NONE:
        // for PNONE, scale max length to [0,1), all choices are valid
        break;
    case PAXIS::PX:
        if (zlen < xlen || zlen < ylen) {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PXY:
        // for PXY,PXZ,PYZ, periodic direcitons must have equal size, and larger than the third direction
        if (fabs(xlen - ylen) < eps && xlen >= zlen) {
            // correct
        } else {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    case PAXIS::PXYZ:
        // for PXYZ, must be cubic
        if (fabs(xlen - ylen) < eps && fabs(xlen - zlen) < eps && fabs(ylen - zlen) < eps) {
            // correct
        } else {
            std::cout << "periodic box size error" << std::endl;
            exit(1);
        }
        break;
    }
}

void STKFMM::setupCoord(const int npts, const double *coordInPtr, std::vector<double> &coord) const {
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

void STKFMM::setPoints(const int nSL, const double *srcSLCoordPtr, const int nDL, const double *srcDLCoordPtr,
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
    poolFMM[kernel_]->setupTree(srcSLCoordInternal, srcDLCoordInternal, trgCoordInternal);
    if (myRank == 0)
        printf("Coord setup for kernel %d\n", static_cast<int>(kernel_));
}

void STKFMM::evaluateFMM(const int nSL, const double *srcSLValuePtr, const int nDL, const double *srcDLValuePtr,
                         const int nTrg, double *trgValuePtr, const KERNEL kernel) {

    if (poolFMM.find(kernel) == poolFMM.end()) {
        printf("Error: no such FMMData exists for kernel %d\n", static_cast<int>(kernel));
        exit(1);
    }
    FMMData &fmm = *((*poolFMM.find(kernel)).second);

    srcSLValueInternal.resize(nSL * fmm.kdimSL);
    srcDLValueInternal.resize(nDL * fmm.kdimDL);

    // scale the source strength, SL as 1/r, DL as 1/r^2
    // SL no extra scaling
    // DL scale as scaleFactor
    std::copy(srcSLValuePtr, srcSLValuePtr + nSL * fmm.kdimSL, srcSLValueInternal.begin());
    int nloop = nDL * fmm.kdimDL;
#pragma omp parallel for
    for (int i = 0; i < nloop; i++) {
        srcDLValueInternal[i] = srcDLValuePtr[i] * scaleFactor;
    }
    if (fmm.kdimSL == 4) {
        // stokes kernel
#pragma omp parallel for
        for (int i = 0; i < nSL; i++) {
            // the Trace term scales as double layer
            srcSLValueInternal[4 * i + 3] *= scaleFactor;
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
            trgValuePtr[4 * i] += trgValueInternal[4 * i] * scaleFactor * scaleFactor; // pressure 1/r^2
            trgValuePtr[4 * i + 1] += trgValueInternal[4 * i + 1] * scaleFactor;       // vel 1/r
            trgValuePtr[4 * i + 2] += trgValueInternal[4 * i + 2] * scaleFactor;
            trgValuePtr[4 * i + 3] += trgValueInternal[4 * i + 3] * scaleFactor;
        }
    } break;
    case KERNEL::PVelGrad: {
        // 1+3+3+9
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            trgValuePtr[16 * i] += trgValueInternal[16 * i] * scaleFactor * scaleFactor; // p
            for (int j = 1; j < 4; j++) {
                trgValuePtr[16 * i + j] += trgValueInternal[16 * i + j] * scaleFactor; // vel
            }
            for (int j = 4; j < 7; j++) {
                trgValuePtr[16 * i + j] +=
                    trgValueInternal[16 * i + j] * scaleFactor * scaleFactor * scaleFactor; // grad p
            }
            for (int j = 7; j < 16; j++) {
                trgValuePtr[16 * i + j] += trgValueInternal[16 * i + j] * scaleFactor * scaleFactor; // grad vel
            }
        }
    } break;
    case KERNEL::Traction: {
        // 9
        int nloop = 9 * nTrg;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValuePtr[i] += trgValueInternal[i] * scaleFactor * scaleFactor; // traction 1/r^2
        }
    } break;
    case KERNEL::PVelLaplacian: {
        // 1+3+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            trgValuePtr[7 * i] += trgValueInternal[7 * i] * scaleFactor * scaleFactor; // p
            for (int j = 1; j < 4; j++) {
                trgValuePtr[7 * i + j] += trgValueInternal[7 * i + j] * scaleFactor; // vel
            }
            for (int j = 4; j < 7; j++) {
                trgValuePtr[7 * i + j] +=
                    trgValueInternal[7 * i + j] * scaleFactor * scaleFactor * scaleFactor; // laplacian vel
            }
        }
    } break;
    case KERNEL::LAPPGrad: {
        // 1+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            trgValuePtr[4 * i] += trgValueInternal[4 * i] * scaleFactor; // p, 1/r
            for (int j = 1; j < 4; j++) {
                trgValuePtr[4 * i + j] += trgValueInternal[4 * i + j] * scaleFactor * scaleFactor; // grad p, 1/r^2
            }
        }

    } break;
    }

    return;
}

void STKFMM::evaluateKernel(const int nThreads, const PPKERNEL p2p, const int nSrc, double *srcCoordPtr,
                            double *srcValuePtr, const int nTrg, double *trgCoordPtr, double *trgValuePtr,
                            const KERNEL kernel) {
    if (poolFMM.find(kernel) == poolFMM.end()) {
        printf("Error: no such FMMData exists for kernel %d\n", static_cast<int>(kernel));
        exit(1);
    }
    FMMData &fmm = *((*poolFMM.find(kernel)).second);

    fmm.evaluateKernel(nThreads, p2p, nSrc, srcCoordPtr, srcValuePtr, nTrg, trgCoordPtr, trgValuePtr);
}

void STKFMM::showActiveKernels() {
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    if (myRank == 0) {
        printf("active kernels:\n");
        if (kernelComb & asInteger(KERNEL::PVel)) {
            printf("PVel\n");
        }
        if (kernelComb & asInteger(KERNEL::PVelGrad)) {
            printf("PVelGrad\n");
        }
        if (kernelComb & asInteger(KERNEL::Traction)) {
            printf("Traction\n");
        }
        if (kernelComb & asInteger(KERNEL::PVelLaplacian)) {
            printf("PVelLaplacian\n");
        }
        if (kernelComb & asInteger(KERNEL::LAPPGrad)) {
            printf("LAPPGrad\n");
        }
    }
}

void STKFMM::clearFMM(KERNEL kernelChoice) {
    trgValueInternal.clear();
    poolFMM[kernelChoice]->clear();
}
} // namespace stkfmm
