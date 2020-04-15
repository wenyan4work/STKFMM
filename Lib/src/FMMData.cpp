#include "STKFMM/STKFMM_impl.hpp"

namespace stkfmm {
namespace impl {

void FMMData::setKernel() {
    matrixPtr->Initialize(multOrder, comm, kernelFunctionPtr);
    kdimSL = kernelFunctionPtr->k_s2t->ker_dim[0];
    kdimTrg = kernelFunctionPtr->k_s2t->ker_dim[1];
    kdimDL = kernelFunctionPtr->surf_dim;
}

// const pvfmm::Kernel<double> *FMMData::getKernelFunction(KERNEL kernelChoice_) {
//     auto it = kernelMap.find(kernelChoice_);
//     if (it != kernelMap.end()) {
//         return it->second;
//     } else {
//         printf("Error: Kernel not found.\n");
//         std::exit(1);
//         return nullptr;
//     }
// }

void FMMData::readM2LMat(const int kDim, const std::string &dataName, std::vector<double> &data) {
    // int size = kDim * (6 * (multOrder - 1) * (multOrder - 1) + 2);
    int size = kDim * equivCoord.size() / 3;
    data.resize(size * size);

    char *pvfmm_dir = getenv("PVFMM_DIR");
    std::string file = std::string(pvfmm_dir) + std::string("/pdata/") + dataName;

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
            std::string dataName = "M2L_laplace_" + std::to_string(pbc) + "D3Dp" + std::to_string(multOrder);
            readM2LMat(1, dataName, M2L_laplace);
        }
        {

            std::string dataName = "M2L_stokes_vel_" + std::to_string(pbc) + "D3Dp" + std::to_string(multOrder);
            readM2LMat(3, dataName, M2L_stokes);
        }
        int nequiv = this->equivCoord.size() / 3;
        M2Ldata.resize(4 * nequiv * 4 * nequiv);
        for (int i = 0; i < nequiv; i++) {
            for (int j = 0; j < nequiv; j++) {
                // each 4x4 block consists of 3x3 of stokes and 1x1 of laplace
                // top-left 4*i, 4*j, size 4x4
                M2Ldata[(4 * i + 3) * (4 * nequiv) + 4 * j + 3] = M2L_laplace[i * nequiv + j];
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
        std::string dataName = "M2L_" + M2Lname + "_" + std::to_string(pbc) + "D3Dp" + std::to_string(multOrder);
        int kdim = kernelFunctionPtr->k_m2l->ker_dim[0];
        readM2LMat(kdim, dataName, this->M2Ldata);
    }
}

// constructor
FMMData::FMMData(KERNEL kernelChoice_, PAXIS periodicity_, int multOrder_, int maxPts_)
    : kernelChoice(kernelChoice_), periodicity(periodicity_), multOrder(multOrder_), maxPts(maxPts_), treePtr(nullptr),
      matrixPtr(nullptr), treeDataPtr(nullptr) {

    comm = MPI_COMM_WORLD;
    matrixPtr = new pvfmm::PtFMM<double>();

    // choose a kernel
    kernelFunctionPtr = getKernelFunction(kernelChoice);
    setKernel();
    treeDataPtr = new pvfmm::PtFMM_Data<double>;

    // load periodicity M2L data
    if (periodicity != PAXIS::NONE) {
        // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
        // RAD1 = 2.95 defined in pvfmm_common.h
        double scaleLEquiv = PVFMM_RAD1;
        double pCenterLEquiv[3];
        pCenterLEquiv[0] = -(scaleLEquiv - 1) / 2;
        pCenterLEquiv[1] = -(scaleLEquiv - 1) / 2;
        pCenterLEquiv[2] = -(scaleLEquiv - 1) / 2;

        equivCoord = surface(multOrder, (double *)&(pCenterLEquiv[0]), scaleLEquiv, 0);

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

void FMMData::setupTree(const std::vector<double> &srcSLCoord, const std::vector<double> &srcDLCoord,
                        const std::vector<double> &trgCoord, const int ntreePts, const double *treePtsPtr) {
    // trgCoord and srcCoord have been scaled to [0,1)^3
    // setup treeData
    treeDataPtr->dim = 3;
    treeDataPtr->max_depth = PVFMM_MAX_DEPTH;
    treeDataPtr->max_pts = maxPts;

    treeDataPtr->src_coord = srcSLCoord;
    treeDataPtr->surf_coord = srcDLCoord;
    treeDataPtr->trg_coord = trgCoord;
    const int nSL = srcSLCoord.size() / 3;
    const int nDL = srcDLCoord.size() / 3;
    const int nTrg = trgCoord.size() / 3;

    // pt_coord is used to setup FMM octree
    if (treePtsPtr == nullptr || ntreePts == 0) {
        // default case, use the largest set of SL,DL,trg
        if (nSL > nDL && nSL > nTrg)
            treeDataPtr->pt_coord = srcSLCoord;
        if (nDL > nSL && nDL > nTrg)
            treeDataPtr->pt_coord = srcDLCoord;
        if (nTrg > nSL && nTrg > nDL)
            treeDataPtr->pt_coord = trgCoord;
    } else {
        // custom case, use custom set of points
        treeDataPtr->pt_coord.Resize(3 * ntreePts);
        std::copy(treePtsPtr, treePtsPtr + 3 * ntreePts, treeDataPtr->pt_coord.Begin());
    }

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
                          std::vector<double> &trgValue, const double scale) {
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
    scaleSrc(srcSLValue, srcDLValue, scale);
    PtFMM_Evaluate(treePtr, trgValue, nTrg, &srcSLValue, &srcDLValue);
    periodizeFMM(trgValue);
    scaleTrg(trgValue, scale);
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

    if (kerPtr == nullptr) {
        std::cout << "PPKernel " << (uint)p2p << " not found for direct evaluation" << std::endl;
        return;
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

void FMMData::scaleSrc(std::vector<double> &srcSLValue, std::vector<double> &srcDLValue, const double scaleFactor) {
    // scale the source strength, SL as 1/r, DL as 1/r^2
    // SL no extra scaling
    // DL scale as scaleFactor

    const int nloop = srcDLValue.size();
#pragma omp parallel for
    for (int i = 0; i < nloop; i++) {
        srcDLValue[i] *= scaleFactor;
    }

    const int nSL = srcSLValue.size() / kdimSL;
    if (kernelChoice == KERNEL::PVel || kernelChoice == KERNEL::PVelGrad || kernelChoice == KERNEL::PVelLaplacian ||
        kernelChoice == KERNEL::Traction || kernelChoice == KERNEL::RPY || kernelChoice == KERNEL::StokesRegVel) {
        // Stokes, RPY, StokesRegVel
#pragma omp parallel for
        for (int i = 0; i < nSL; i++) {
            // the Trace term of PVel
            // the epsilon terms of RPY/StokesRegVel
            // scale as double layer
            srcSLValue[4 * i + 3] *= scaleFactor;
        }
    }

    if (kernelChoice == KERNEL::StokesRegVelOmega) {
#pragma omp parallel for
        for (int i = 0; i < nSL; i++) {
            // Scale torque / epsilon
            for (int j = 3; j < 7; ++j)
                srcSLValue[7 * i + j] *= scaleFactor;
        }
    }
}

void FMMData::scaleTrg(std::vector<double> &trgValue, const double scaleFactor) {

    const int nTrg = trgValue.size() / kdimTrg;
    // scale back according to kernel
    switch (kernelChoice) {
    case KERNEL::PVel: {
        // 1+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // pressure 1/r^2
            trgValue[4 * i] *= scaleFactor * scaleFactor;
            // vel 1/r
            trgValue[4 * i + 1] *= scaleFactor;
            trgValue[4 * i + 2] *= scaleFactor;
            trgValue[4 * i + 3] *= scaleFactor;
        }
    } break;
    case KERNEL::PVelGrad: {
        // 1+3+3+9
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p

            trgValue[16 * i] *= scaleFactor * scaleFactor;
            // vel
            for (int j = 1; j < 4; j++) {
                trgValue[16 * i + j] *= scaleFactor;
            }
            // grad p
            for (int j = 4; j < 7; j++) {
                trgValue[16 * i + j] *= scaleFactor * scaleFactor * scaleFactor;
            }
            // grad vel
            for (int j = 7; j < 16; j++) {
                trgValue[16 * i + j] *= scaleFactor * scaleFactor;
            }
        }
    } break;
    case KERNEL::Traction: {
        // 9
        int nloop = 9 * nTrg;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            // traction 1/r^2
            trgValue[i] *= scaleFactor * scaleFactor;
        }
    } break;
    case KERNEL::PVelLaplacian: {
        // 1+3+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p
            trgValue[7 * i] *= scaleFactor * scaleFactor;
            // vel
            for (int j = 1; j < 4; j++) {
                trgValue[7 * i + j] *= scaleFactor;
            }
            // laplacian vel
            for (int j = 4; j < 7; j++) {
                trgValue[7 * i + j] *= scaleFactor * scaleFactor * scaleFactor;
            }
        }
    } break;
    case KERNEL::LapPGrad: {
        // 1+3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p, 1/r
            trgValue[4 * i] *= scaleFactor;
            // grad p, 1/r^2
            for (int j = 1; j < 4; j++) {
                trgValue[4 * i + j] *= scaleFactor * scaleFactor;
            }
        }
    } break;
    case KERNEL::LapPGradGrad: {
        // 1+3+6
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p, 1/r
            trgValue[10 * i] *= scaleFactor;
            // grad p, 1/r^2
            for (int j = 1; j < 4; j++) {
                trgValue[10 * i + j] *= scaleFactor * scaleFactor;
            }
            // grad grad p, 1/r^3
            for (int j = 4; j < 10; j++) {
                trgValue[10 * i + j] *= scaleFactor * scaleFactor * scaleFactor;
            }
        }
    } break;
    case KERNEL::LapQPGradGrad: {
        // 1+3+6
        const double sf3 = scaleFactor * scaleFactor * scaleFactor;
        const double sf4 = scaleFactor * sf3;
        const double sf5 = scaleFactor * sf4;
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // p, 1/r^3
            trgValue[10 * i] *= sf3;
            // grad p, 1/r^4
            for (int j = 1; j < 4; j++) {
                trgValue[10 * i + j] *= sf4;
            }
            // grad grad p, 1/r^5
            for (int j = 4; j < 10; j++) {
                trgValue[10 * i + j] *= sf5;
            }
        }
    } break;
    case KERNEL::Stokes: {
        // 3
        const int nloop = nTrg * 3;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValue[i] *= scaleFactor; // vel 1/r
        }
    } break;
    case KERNEL::StokesRegVel: {
        // 3
        const int nloop = nTrg * 3;
#pragma omp parallel for
        for (int i = 0; i < nloop; i++) {
            trgValue[i] *= scaleFactor; // vel 1/r
        }
    } break;
    case KERNEL::StokesRegVelOmega: {
        // 3 + 3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // vel 1/r
            for (int j = 0; j < 3; ++j)
                trgValue[i * 6 + j] *= scaleFactor;
            // omega 1/r^2
            for (int j = 3; j < 6; ++j)
                trgValue[i * 6 + j] *= scaleFactor * scaleFactor;
        }
    } break;
    case KERNEL::RPY: {
        // 3 + 3
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            // vel 1/r
            for (int j = 0; j < 3; ++j)
                trgValue[i * 6 + j] *= scaleFactor;
            // laplacian vel
            for (int j = 3; j < 6; ++j)
                trgValue[i * 6 + j] *= scaleFactor * scaleFactor * scaleFactor;
        }
    } break;
    }
}

} // namespace impl
} // namespace stkfmm