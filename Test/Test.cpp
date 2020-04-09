#include "Test.hpp"

#include <memory>

std::unordered_map<KERNEL, std::pair<kernel_func, kernel_func>>
    SL_kernels({{KERNEL::LapPGrad, std::make_pair(LaplaceSLPGrad, LaplaceDLPGrad)},
                {KERNEL::LapPGradGrad, std::make_pair(LaplaceSLPGradGrad, LaplaceDLPGradGrad)},
                {KERNEL::LapQPGradGrad, std::make_pair(LaplaceQPGradGrad, nullptr)},
                {KERNEL::Stokes, std::make_pair(StokesSL, StokesDL)},
                {KERNEL::RPY, std::make_pair(StokesSLRPY, nullptr)},
                {KERNEL::StokesRegVel, std::make_pair(StokesRegSLVel, nullptr)},
                {KERNEL::StokesRegVelOmega, std::make_pair(StokesRegSLVelOmega, nullptr)},
                {KERNEL::PVel, std::make_pair(StokesSLPVel, StokesDLPVel)},
                {KERNEL::PVelGrad, std::make_pair(StokesSLPVelGrad, StokesDLPVelGrad)},
                {KERNEL::Traction, std::make_pair(StokesSLTraction, StokesDLTraction)},
                {KERNEL::PVelLaplacian, std::make_pair(StokesSLPVelLaplacian, StokesDLPVelLaplacian)}});

void configure_parser(cli::Parser &parser) {
    parser.set_optional<int>("S", "nSLSource", 1, "1/2/4 for 1/2/4 point forces, other for same as target, default=1");
    parser.set_optional<int>("D", "nDLSource", 1, "1/2/4 for 1/2/4 point forces, other for same as target, default=1");
    parser.set_optional<int>("T", "nTarget", 2, "total number of targets = (T+1)^3, default T=2");
    parser.set_optional<int>("s", "Seed", 1, "RNG Seed");
    parser.set_optional<double>("B", "box", 1.0, "box edge length, default B=1.0");
    parser.set_optional<double>("M", "move", 0.0, "box origin shift move, default M=0");
    parser.set_optional<int>("K", "Kernel Combination", 0, "activated kernels, default=0 means all kernels");
    parser.set_optional<int>("R", "Random", 1, "0 for random, 1 for regular mesh, default 1");
    parser.set_optional<int>("F", "FMM", 1, "0 to test S2T kernel, 1 to test FMM, default 1");
    parser.set_optional<int>("V", "Verify", 1, "1 for O(N^2) and 0 for p=16 verification, default 1");
    parser.set_optional<int>("P", "Periodic", 0, "0 for NONE, 1 for PX, 2 for PXY, 3 for PXYZ, default 0");
    parser.set_optional<int>("m", "maxPoints", 50, "Max number of points in adaptive Octree, default 50");
    parser.set_optional<double>("e", "epsilon", 0.01,
                                "Maximum size of particle for RPY and StokesReg kernels, default 0.01");
}

void showOption(const cli::Parser &parser) {
    std::cout << "Running setting: " << std::endl;
    std::cout << "nSL Source: " << parser.get<int>("S") << std::endl;
    std::cout << "nDL Source: " << parser.get<int>("D") << std::endl;
    std::cout << "nTarget: " << parser.get<int>("T") << std::endl;
    std::cout << "RNG Seed: " << parser.get<int>("s") << std::endl;
    std::cout << "Box: " << parser.get<double>("B") << std::endl;
    std::cout << "Shift: " << parser.get<double>("M") << std::endl;
    std::cout << "KERNEL: " << parser.get<int>("K") << std::endl;
    std::cout << "Random: " << parser.get<int>("R") << std::endl;
    std::cout << "Using FMM: " << parser.get<int>("F") << std::endl;
    std::cout << "Verification: " << parser.get<int>("V") << std::endl;
    std::cout << "Periodic BC: " << parser.get<int>("P") << std::endl;
    std::cout << "maxPoints: " << parser.get<int>("m") << std::endl;
    std::cout << "eps: " << parser.get<double>("e") << std::endl;
}

// generate (distributed) FMM points
void genPoint(const cli::Parser &parser, FMMpoint &point, bool wall) {
    const double shift = parser.get<double>("M");
    const double box = parser.get<double>("B");
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    auto &srcLocalSL = point.srcLocalSL;
    auto &srcLocalDL = point.srcLocalDL;
    auto &trgLocal = point.trgLocal;
    srcLocalSL.clear();
    srcLocalDL.clear();
    trgLocal.clear();

    PointDistribution pd(parser.get<int>("s"));

    if (myRank == 0) {
        // set trg coord
        const int nPts = parser.get<int>("T");
        if (parser.get<int>("R") > 0) {
            pd.randomPoints(nPts, box, shift, trgLocal);
        } else {
            PointDistribution::meshPoints(nPts, box, shift, trgLocal);
        }

        // set src SL coord
        const int nSL = parser.get<int>("S");
        if (nSL == 0) {
            srcLocalSL.clear();
        } else if (nSL == 1 || nSL == 2 || nSL == 4) {
            PointDistribution::fixedPoints(nSL, box, shift, srcLocalSL);
        } else {
            srcLocalSL = trgLocal;
        }

        const int nDL = parser.get<int>("D");
        if (nDL == 0) {
            srcLocalDL.clear();
        } else if (nDL == 1 || nDL == 2 || nDL == 4) {
            PointDistribution::fixedPoints(nDL, box, shift, srcLocalDL);
        } else {
            srcLocalDL = trgLocal;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // distribute points
    PointDistribution::distributePts(srcLocalSL, 3);
    PointDistribution::distributePts(srcLocalDL, 3);
    PointDistribution::distributePts(trgLocal, 3);

    MPI_Barrier(MPI_COMM_WORLD);

    if (wall) {
        auto scaleZ = [&](std::vector<double> &pts) {
            const int npts = pts.size() / 3;
            // from [shift,shift+box) to [shift,shift+box/2)
            for (int i = 0; i < npts; i++) {
                pts[3 * i + 2] = shift + 0.5 * (pts[3 * i + 2] - shift);
            }
        };
        scaleZ(point.srcLocalSL);
        scaleZ(point.srcLocalDL);
        scaleZ(point.trgLocal);
    }
}

// generate SrcValue, distributed with given points
void genSrcValue(const cli::Parser &parser, const FMMpoint &point, FMMinput &inputs, bool neutral) {
    using namespace stkfmm;
    inputs.clear();
    const int nSL = point.srcLocalSL.size() / 3;
    const int nDL = point.srcLocalDL.size() / 3;
    const int nTrg = point.trgLocal.size() / 3;
    const int pbc = parser.get<int>("P");
    const int kernelComb = parser.get<int>("K");

    PointDistribution pd(parser.get<int>("s"));

    // loop over each activated kernel
    for (const auto &it : kernelMap) {
        auto kernel = it.first;
        if (kernelComb != 0 && !(asInteger(kernel) & kernelComb)) {
            continue;
        }
        FMMsrcval value;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
        value.srcLocalSL.resize(kdimSL * nSL);
        value.srcLocalDL.resize(kdimDL * nDL);

        // generate random values
        pd.randomUniformFill(value.srcLocalSL, -1, 1);
        pd.randomUniformFill(value.srcLocalDL, -1, 1);

        if (neutral) {
            // special requirements
            if ((kernel == KERNEL::LapPGrad || kernel == KERNEL::LapPGradGrad) && pbc) { // must be neutral for periodic
                int nSLGlobal = nSL;
                MPI_Allreduce(MPI_IN_PLACE, &nSLGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                double netCharge = std::accumulate(value.srcLocalSL.begin(), value.srcLocalSL.end(), 0.0);
                MPI_Allreduce(MPI_IN_PLACE, &netCharge, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                netCharge /= nSLGlobal;
                for (auto &v : value.srcLocalSL) {
                    v -= netCharge;
                }
            }

            if ((kernel == KERNEL::PVel || kernel == KERNEL::PVelGrad || kernel == KERNEL::PVelLaplacian ||
                 kernel == KERNEL::Traction) &&
                pbc) {
                // must be force-neutral for x-y-z-trD
                int nSLGlobal = nSL;
                MPI_Allreduce(MPI_IN_PLACE, &nSLGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                assert(kdimSL == 4);
                double fnet[4] = {0, 0, 0, 0};
                for (int i = 0; i < nSL; i++) {
                    for (int j = 0; j < 4; j++) {
                        fnet[j] += value.srcLocalSL[4 * i + j];
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, fnet, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                fnet[0] /= nSLGlobal;
                fnet[1] /= nSLGlobal;
                fnet[2] /= nSLGlobal;
                fnet[3] /= nSLGlobal;
                for (int i = 0; i < nSL; i++) {
                    for (int j = 0; j < 4; j++) {
                        value.srcLocalSL[4 * i + j] -= fnet[j];
                    }
                }
                // must be trace-free for double layer
                int nDLGlobal = nDL;
                MPI_Allreduce(MPI_IN_PLACE, &nDLGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                assert(kdimDL == 9);
                double trD = 0;
                for (int i = 0; i < nDL; i++) {
                    trD += value.srcLocalDL[9 * i + 0];
                    trD += value.srcLocalDL[9 * i + 4];
                    trD += value.srcLocalDL[9 * i + 8];
                }
                MPI_Allreduce(MPI_IN_PLACE, &trD, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                trD /= (3 * nDL);
                for (int i = 0; i < nDL; i++) {
                    value.srcLocalDL[9 * i + 0] -= trD;
                    value.srcLocalDL[9 * i + 4] -= trD;
                    value.srcLocalDL[9 * i + 8] -= trD;
                }
            }

            if (kernel == KERNEL::StokesRegVel || kernel == KERNEL::StokesRegVelOmega || kernel == KERNEL::RPY) {
                // sphere radius/regularization must be small
                const double reg = parser.get<double>("e");
                auto setreg = [&](double &v) { v = std::abs(v) * reg; };
                for (int i = 0; i < nSL; i++) {
                    setreg(value.srcLocalSL[kdimSL * i + kdimSL - 1]);
                }
            }
        }
        inputs[kernel] = value;
    }
}

void translatePoints(const cli::Parser &parser, FMMpoint &point) {
    const double box = parser.get<double>("B");
    const int paxis = parser.get<int>("P");
    const double shift = parser.get<double>("M");

    const int nSL = point.srcLocalSL.size() / 3;
    const int nDL = point.srcLocalDL.size() / 3;
    const int nTrg = point.trgLocal.size() / 3;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double trans[3] = {0, 0, 0};
    if (!rank) {
        // Standard mersenne_twister_engine seeded
        std::mt19937 gen(parser.get<int>("s"));
        std::uniform_real_distribution<double> dis(-1, 1);
        if (paxis == 1)
            trans[0] = dis(gen);
        else if (paxis == 2) {
            trans[0] = dis(gen);
            trans[1] = dis(gen);
        } else if (paxis == 3) {
            trans[0] = dis(gen);
            trans[1] = dis(gen);
            trans[2] = dis(gen);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(trans, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto translate = [&](std::vector<double> &coord, const int np) {
        for (int i = 0; i < np; i++) {
            for (int j = 0; j < 3; j++) {
                auto &pos = coord[3 * i + j];
                pos += trans[j] * box;
                while (pos < shift)
                    pos += box;
                while (pos >= shift + box)
                    pos -= box;
            }
        }
    };
    translate(point.srcLocalSL, nSL);
    translate(point.srcLocalDL, nDL);
    translate(point.trgLocal, nTrg);
}

void dumpValue(const std::string &tag, const FMMpoint &point, const FMMinput &inputs, const FMMresult &results) {
    auto writedata = [&](std::string name, const std::vector<double> &coord_, const std::vector<double> &value_,
                         const int kdim) {
        auto coord = coord_;
        auto value = value_;
        PointDistribution::dumpPoints(name + ".txt", coord, value, kdim);
    };

    for (auto &data : inputs) {
        auto &kernel = data.first;
        auto &value = data.second;
        std::vector<double> trgLocal;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
        auto it = results.find(kernel);
        if (it != results.end()) {
            trgLocal = it->second;
        } else {
            printf("result not found for kernel %d\n", asInteger(kernel));
        }
        writedata(tag + "_srcSL_K" + std::to_string(asInteger(kernel)), point.srcLocalSL, value.srcLocalSL, kdimSL);
        writedata(tag + "_srcDL_K" + std::to_string(asInteger(kernel)), point.srcLocalDL, value.srcLocalDL, kdimDL);
        writedata(tag + "_trg_K" + std::to_string(asInteger(kernel)), point.trgLocal, trgLocal, kdimTrg);
    }
}

void checkError(const FMMresult &A, const FMMresult &B, bool component) {
    for (auto &data : A) {
        auto kernel = data.first;
        auto it = B.find(kernel);
        if (it == B.end()) {
            printf("check result error, reference not found\n");
            exit(1);
        }
        std::cout << "Error for kernel: " << asInteger(kernel) << std::endl;
        if (component) {
            int kdimSL, kdimDL, kdimTrg;
            std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
            PointDistribution::checkError(data.second, it->second, kdimTrg);
        } else
            PointDistribution::checkError(data.second, it->second);
    }
}

void runSimpleKernel(const FMMpoint &point, FMMinput &inputs, FMMresult &results) {
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // create a copy for MPI
    std::vector<double> srcSLCoordGlobal = point.srcLocalSL;
    std::vector<double> srcDLCoordGlobal = point.srcLocalDL;
    std::vector<double> trgCoordLocal = point.trgLocal;

    // src is fully replicated on every node
    // trg remains distributed
    PointDistribution::collectPtsAll(srcSLCoordGlobal);
    PointDistribution::collectPtsAll(srcDLCoordGlobal);

    // loop over all activated kernels
    for (auto &data : inputs) {
        KERNEL kernel = data.first;
        auto &value = data.second;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);

        std::vector<double> srcSLValueGlobal = value.srcLocalSL;
        std::vector<double> srcDLValueGlobal = value.srcLocalDL;
        PointDistribution::collectPtsAll(srcSLValueGlobal);
        PointDistribution::collectPtsAll(srcDLValueGlobal);

        // on every node, from global src to local trg
        const int nSL = srcSLCoordGlobal.size() / 3;
        const int nDL = srcDLCoordGlobal.size() / 3;
        const int nTrg = trgCoordLocal.size() / 3;

        // Create mapping of kernels to 'true value' functions
        kernel_func kernelTestSL, kernelTestDL;
        std::tie(kernelTestSL, kernelTestDL) = SL_kernels[kernel];

        std::vector<double> trgLocal(nTrg * kdimTrg, 0);
        // check results
#pragma omp parallel for
        for (int i = 0; i < nTrg; i++) {
            const double *trg = trgCoordLocal.data() + 3 * i;
            double t[3] = {trg[0], trg[1], trg[2]};

            // add SL values
            for (int j = 0; j < nSL; j++) {
                double result[20] = {0.0};
                double *s = srcSLCoordGlobal.data() + 3 * j;
                double *sval = srcSLValueGlobal.data() + kdimSL * j;

                kernelTestSL(s, t, sval, result);

                for (int k = 0; k < kdimTrg; k++) {
                    trgLocal[kdimTrg * i + k] += result[k];
                }
            }

            // add DL values
            for (int j = 0; j < nDL; j++) {
                double result[20] = {0.0};
                double *s = srcDLCoordGlobal.data() + 3 * j;
                double *sval = srcDLValueGlobal.data() + kdimDL * j;

                kernelTestDL(s, t, sval, result);

                for (int k = 0; k < kdimTrg; k++) {
                    trgLocal[kdimTrg * i + k] += result[k];
                }
            }
        }
        results[kernel] = trgLocal;
    }
}

void runFMM(const cli::Parser &parser, const int p, const FMMpoint &point, FMMinput &inputs, FMMresult &results,
            bool wall) {
    using namespace stkfmm;
    results.clear();
    const double shift = parser.get<double>("M");
    const double box = parser.get<double>("B");
    const int temp = parser.get<int>("K");
    const int k = (temp == 0) ? ~((int)0) : temp;
    const PAXIS paxis = (PAXIS)parser.get<int>("P");
    const int maxPoints = parser.get<int>("m");

    std::shared_ptr<STKFMM> fmmPtr;
    if (wall) {
        if (k != static_cast<int>(KERNEL::Stokes) & static_cast<int>(KERNEL::RPY)) {
            printf("WallFMM supports only Stokes and RPY Single Layer kernels\n");
            std::exit(1);
        }
        fmmPtr = std::make_shared<StkWallFMM>(p, maxPoints, paxis, k);
    } else {
        fmmPtr = std::make_shared<Stk3DFMM>(p, maxPoints, paxis, k);
    }

    double origin[3] = {shift, shift, shift};
    fmmPtr->setBox(origin, box);
    fmmPtr->showActiveKernels();
    const int nSL = point.srcLocalSL.size() / 3;
    const int nDL = point.srcLocalDL.size() / 3;
    const int nTrg = point.trgLocal.size() / 3;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fmmPtr->setPoints(nSL, point.srcLocalSL.data(), nTrg, point.trgLocal.data(), nDL, point.srcLocalDL.data());

    for (auto &data : inputs) {
        auto &kernel = data.first;
        auto &value = data.second;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
        std::vector<double> trgLocal(nTrg * kdimTrg, 0);
        if (parser.get<int>("F")) {

            fmmPtr->clearFMM(kernel);
            Timer timer;
            timer.tick();
            fmmPtr->setupTree(kernel);
            timer.tock("setupTree");
            timer.tick();
            fmmPtr->evaluateFMM(kernel, nSL, value.srcLocalSL.data(), nTrg, trgLocal.data(), nDL,
                                value.srcLocalDL.data());
            timer.tock("evaluateFMM");
            if (!rank)
                timer.dump();
        } else {
            auto srcLocalCoord = point.srcLocalSL;
            auto trgLocalCoord = point.trgLocal;
            fmmPtr->evaluateKernel(kernel, 0, PPKERNEL::SLS2T, nSL, srcLocalCoord.data(), value.srcLocalSL.data(), nTrg,
                                   trgLocalCoord.data(), trgLocal.data());
        }
        results[kernel] = trgLocal;
    }
}
