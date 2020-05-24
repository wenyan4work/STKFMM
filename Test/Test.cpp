#include "Test.hpp"
#include "Util/CLI11.hpp"
#include "Util/json.hpp"

#include <iostream>
#include <memory>

#include <mpi.h>

typedef void (*kernel_func)(double *, double *, double *, double *);

std::unordered_map<KERNEL, std::pair<kernel_func, kernel_func>>
    SL_kernels({{KERNEL::LapPGrad, std::make_pair(LaplaceSLPGrad, LaplaceDLPGrad)},
                {KERNEL::LapPGradGrad, std::make_pair(LaplaceSLPGradGrad, LaplaceDLPGradGrad)},
                {KERNEL::LapQPGradGrad, std::make_pair(LaplaceQPGradGrad, nullptr)},
                {KERNEL::Stokes, std::make_pair(StokesSL, nullptr)},
                {KERNEL::RPY, std::make_pair(StokesSLRPY, nullptr)},
                {KERNEL::StokesRegVel, std::make_pair(StokesRegSLVel, nullptr)},
                {KERNEL::StokesRegVelOmega, std::make_pair(StokesRegSLVelOmega, nullptr)},
                {KERNEL::PVel, std::make_pair(StokesSLPVel, StokesDLPVel)},
                {KERNEL::PVelGrad, std::make_pair(StokesSLPVelGrad, StokesDLPVelGrad)},
                {KERNEL::Traction, std::make_pair(StokesSLTraction, StokesDLTraction)},
                {KERNEL::PVelLaplacian, std::make_pair(StokesSLPVelLaplacian, StokesDLPVelLaplacian)}});

void Config::parse(int argc, char **argv) {
    CLI::App app("Test Driver for Stk3DFMM and StkWallFMM\n");
    app.set_config("--config", "", "config file name");
    // basic settings
    app.add_option("-S,--nsl", nSL, "number of source SL points");
    app.add_option("-D,--ndl", nDL, "number of source DL points");
    app.add_option("-T,--ntrg", nTrg, "number of source TRG points");
    app.add_option("-B,--box", box, "testing cubic box edge length");
    app.add_option("-O,--origin", origin, "testing cubic box origin point");
    app.add_option("-K,--kernel", K, "test which kernels");
    app.add_option("-P,--pbc", pbc, "periodic boundary condition. 0=none, 1=PX, 2=PXY, 3=PXYZ");
    app.add_option("-M,--maxOrder", maxOrder, "max KIFMM order, must be even number. Default 16.");

    // tunnings
    app.add_option("--seed", rngseed, "seed for random number generator");
    app.add_option("--eps", epsilon, "epsilon or a for Regularized and RPY kernels");
    app.add_option("--max", maxPoints, "max number of points in an octree leaf box");

    // flags
    app.add_flag("--direct,!--no-direct", direct, "run O(N^2) direct summation with S2T kernels");
    app.add_flag("--verify,!--no-verify", verify, "verify results with O(N^2) direct summation");
    app.add_flag("--convergence,!--no-convergence", convergence, "calculate convergence error relative to FMM at p=16");
    app.add_flag("--random,!--no-random", random, "use random points, otherwise regular mesh");

    // wall settings
    app.add_flag("--wall,!--no-wall", wall, "test StkWallFMM, otherwise Stk3DFMM");

    // parse
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        app.exit(e);
        exit(1);
    }

    // sanity check
    if (wall) {
        if (pbc == 3) {
            printf_rank0("PXYZ doesn't work for wall fmm\n");
            exit(1);
        }
        if (verify) {
            printf_rank0("Verify + wall checks no-slip condition only\n");
        }
        if (direct) {
            printf_rank0("option direct doesn't work for wall fmm\n");
            exit(1);
        }
    }

    if (pbc && verify) {
        printf_rank0("option verify doesn't work for periodic boundary conditions\n");
        exit(1);
    }

    if (pbc && direct) {
        printf_rank0("option direct doesn't work for periodic boundary conditions\n");
        exit(1);
    }
}

void Config::print() const {
    printf_rank0("Testing settings:\n");
    printf_rank0("nSL %d, nDL %d, nTrg %d\n", nSL, nDL, nTrg);
    printf_rank0("box %g\n", box);
    printf_rank0("origin %g,%g,%g\n", origin[0], origin[1], origin[2]);
    printf_rank0("Kernel %d\n", K);
    printf_rank0("PBC %d\n", pbc);

    printf_rank0("rngseed %d\n", rngseed);
    printf_rank0("maxPoints %d\n", maxPoints);
    printf_rank0("epsilon RPY/REG %g\n", epsilon);

    printf_rank0(direct ? "Run S2T N2 direct summation\n" : "Run FMM\n");
    printf_rank0(verify ? "Show true error\n" : "");
    printf_rank0(convergence ? "Show convergence error\n" : "");
    printf_rank0(random ? "Random points\n" : "Regular mesh\n");

    printf_rank0(wall ? "Testing StkWallFMM\n" : "Testing Stk3DFMM\n");
}

ComponentError::ComponentError(const std::vector<double> &A, const std::vector<double> &B) {
    if (A.size() != B.size()) {
        printf("size error calc drift\n");
        exit(1);
    }
    const int N = A.size();
    drift = 0;
    for (int i = 0; i < N; i++) {
        drift += A[i] - B[i];
    }
    drift /= N;

    std::vector<double> value = A;
    const auto &valueTrue = B;
    for (auto &v : value) {
        v -= drift;
    }

    double L2 = 0;
    for (int i = 0; i < N; i++) {
        double e2 = pow(valueTrue[i] - value[i], 2);
        errorL2 += e2;
        L2 += pow(valueTrue[i], 2);
        errorMaxRel = std::max(errorMaxRel, fabs(sqrt(e2) / valueTrue[i]));
    }
    errorRMS = sqrt(errorL2 / N);
    errorL2 = sqrt(errorL2 / L2);
    driftL2 = drift * N / sqrt(L2);
}

// generate (distributed) FMM points
void genPoint(const Config &config, Point &point, int dim) {
    const auto &origin = config.origin;
    const auto &box = config.box;

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    auto &srcLocalSL = point.srcLocalSL;
    auto &srcLocalDL = point.srcLocalDL;
    auto &trgLocal = point.trgLocal;
    srcLocalSL.clear();
    srcLocalDL.clear();
    trgLocal.clear();

    PointDistribution pd(config.rngseed);

    if (myRank == 0) {
        if (config.random) {
            pd.randomPoints(dim, config.nTrg, box, 0, trgLocal);
        } else {
            PointDistribution::meshPoints(dim, config.nTrg, box, 0, trgLocal);
        }

        if (config.nSL == 0) {
            srcLocalSL.clear();
        } else if (config.nSL == 1 || config.nSL == 2 || config.nSL == 4) {
            PointDistribution::fixedPoints(config.nSL, box, 0, srcLocalSL);
        } else {
            srcLocalSL = trgLocal;
        }

        if (config.nDL == 0) {
            srcLocalDL.clear();
        } else if (config.nDL == 1 || config.nDL == 2 || config.nDL == 4) {
            PointDistribution::fixedPoints(config.nDL, box, 0, srcLocalDL);
        } else {
            srcLocalDL = trgLocal;
        }

        // config.nSL/nDL/nTrg are not the actual nSL/nDL/nTrg
        const int nSL = srcLocalSL.size() / 3;
        const int nDL = srcLocalDL.size() / 3;
        const int nTrg = trgLocal.size() / 3;

        // shift points
        auto shift = [&](std::vector<double> &pts, int npts) {
            for (int i = 0; i < npts; i++) {
                pts[3 * i + 0] += origin[0];
                pts[3 * i + 1] += origin[1];
                pts[3 * i + 2] += origin[2];
            }
        };

        shift(trgLocal, nTrg);
        shift(srcLocalSL, nSL);
        shift(srcLocalDL, nDL);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // distribute points
    PointDistribution::distributePts(srcLocalSL, 3);
    PointDistribution::distributePts(srcLocalDL, 3);
    PointDistribution::distributePts(trgLocal, 3);

    MPI_Barrier(MPI_COMM_WORLD);

    if (config.wall) {
        auto scaleZ = [&](std::vector<double> &pts) {
            const int npts = pts.size() / 3;
            // from [shift,shift+box) to [shift,shift+box/2)
            for (int i = 0; i < npts; i++) {
                pts[3 * i + 2] = origin[2] + 0.5 * (pts[3 * i + 2] - origin[2]);
            }
        };
        scaleZ(point.srcLocalSL);
        scaleZ(point.srcLocalDL);
        scaleZ(point.trgLocal);

        if (config.verify) {
            // verify wall vel = 0.
            const int ntrg = point.trgLocal.size() / 3;
            for (int i = 0; i < ntrg; i++) {
                point.trgLocal[3 * i + 2] = origin[2];
            }
        }
    }
}

// translate distributed points
void translatePoint(const Config &config, Point &point) {
    const double box = config.box;
    const int paxis = config.pbc;
    const auto &origin = config.origin;

    const int nSL = point.srcLocalSL.size() / 3;
    const int nDL = point.srcLocalDL.size() / 3;
    const int nTrg = point.trgLocal.size() / 3;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double trans[3] = {0, 0, 0};
    if (!rank) {
        // generate random number on rank0
        // Standard mersenne_twister_engine seeded
        std::mt19937 gen(config.rngseed);
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
                while (pos < origin[j])
                    pos += box;
                while (pos >= origin[j] + box)
                    pos -= box;
            }
        }
    };
    translate(point.srcLocalSL, nSL);
    translate(point.srcLocalDL, nDL);
    translate(point.trgLocal, nTrg);
}

// generate SrcValue, distributed with given points
void genSrcValue(const Config &config, const Point &point, Input &input) {
    using namespace stkfmm;
    input.clear();
    const int nSL = point.srcLocalSL.size() / 3;
    const int nDL = point.srcLocalDL.size() / 3;
    const int nTrg = point.trgLocal.size() / 3;
    const int pbc = config.pbc;
    const int kernelComb = config.K;

    PointDistribution pd(config.rngseed);

    // loop over each activated kernel
    for (const auto &it : kernelMap) {
        auto kernel = it.first;
        if (kernelComb != 0 && !(asInteger(kernel) & kernelComb)) {
            continue;
        }
        Source value;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
        value.srcLocalSL.resize(kdimSL * nSL);
        value.srcLocalDL.resize(kdimDL * nDL);

        // generate random values
        pd.randomUniformFill(value.srcLocalSL, -1, 1);
        pd.randomUniformFill(value.srcLocalDL, -1, 1);

        if (kernel == KERNEL::StokesRegVel || kernel == KERNEL::StokesRegVelOmega || kernel == KERNEL::RPY) {
            // sphere radius/regularization must be small
            const double reg = config.epsilon;
            auto setreg = [&](double &v) { v = std::abs(v) * reg; };
            for (int i = 0; i < nSL; i++) {
                setreg(value.srcLocalSL[kdimSL * i + kdimSL - 1]);
            }
        }

        if (config.pbc != 0 && config.wall == false) {
            // must be neutral for some periodic without wall
            if (kernel == KERNEL::StokesRegVel || kernel == KERNEL::StokesRegVelOmega || kernel == KERNEL::RPY) {
                int nSLGlobal = nSL;
                MPI_Allreduce(MPI_IN_PLACE, &nSLGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                assert(kdimSL == 4);
                double fnet[4] = {0, 0, 0};
                for (int i = 0; i < nSL; i++) {
                    for (int j = 0; j < 3; j++) {
                        fnet[j] += value.srcLocalSL[4 * i + j];
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, fnet, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                fnet[0] /= nSLGlobal;
                fnet[1] /= nSLGlobal;
                fnet[2] /= nSLGlobal;
                for (int i = 0; i < nSL; i++) {
                    for (int j = 0; j < 3; j++) {
                        value.srcLocalSL[4 * i + j] -= fnet[j];
                    }
                }
            }

            if (kernel == KERNEL::LapPGrad || kernel == KERNEL::LapPGradGrad) {
                int nSLGlobal = nSL;
                MPI_Allreduce(MPI_IN_PLACE, &nSLGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                double netCharge = std::accumulate(value.srcLocalSL.begin(), value.srcLocalSL.end(), 0.0);
                MPI_Allreduce(MPI_IN_PLACE, &netCharge, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                netCharge /= nSLGlobal;
                for (auto &v : value.srcLocalSL) {
                    v -= netCharge;
                }
            }

            if (kernel == KERNEL::PVel || kernel == KERNEL::PVelGrad || kernel == KERNEL::PVelLaplacian ||
                kernel == KERNEL::Traction) {
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
        }
        input[kernel] = value;
    }
}

void dumpValue(const std::string &tag, const Point &point, const Input &input, const Result &result) {
    auto writedata = [&](std::string name, const std::vector<double> &coord_, const std::vector<double> &value_,
                         const int kdim) {
        auto coord = coord_;
        auto value = value_;
        PointDistribution::dumpPoints(name + ".txt", coord, value, kdim);
    };

    for (auto &data : input) {
        auto &kernel = data.first;
        auto &value = data.second;
        std::vector<double> trgLocal;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
        auto it = result.find(kernel);
        if (it != result.end()) {
            trgLocal = it->second;
        } else {
            std::cout << "result not found for kernel " << getKernelName(kernel) << std::endl;
            exit(1);
        }
        writedata(tag + "_srcSL_K" + std::to_string(asInteger(kernel)), point.srcLocalSL, value.srcLocalSL, kdimSL);
        writedata(tag + "_srcDL_K" + std::to_string(asInteger(kernel)), point.srcLocalDL, value.srcLocalDL, kdimDL);
        writedata(tag + "_trg_K" + std::to_string(asInteger(kernel)), point.trgLocal, trgLocal, kdimTrg);
    }
}

void runSimpleKernel(const Point &point, Input &input, Result &result) {
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
    for (auto &data : input) {
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
            if (kernelTestSL)
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
            if (kernelTestDL)
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
        result[kernel] = trgLocal;
    }
}

void runFMM(const Config &config, const int p, const Point &point, Input &input, Result &result, Timing &timing) {
    using namespace stkfmm;
    result.clear();

    const int k = (config.K == 0) ? ~((int)0) : config.K;
    const PAXIS paxis = static_cast<PAXIS>(config.pbc);
    const int maxPoints = config.maxPoints;

    std::shared_ptr<STKFMM> fmmPtr;
    if (config.wall) {
        fmmPtr = std::make_shared<StkWallFMM>(p, maxPoints, paxis, k);
    } else {
        fmmPtr = std::make_shared<Stk3DFMM>(p, maxPoints, paxis, k);
    }
    fmmPtr->showActiveKernels();

    for (auto &data : input) {
        auto &kernel = data.first;
        auto &value = data.second;
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = getKernelDimension(kernel);
        std::vector<double> trgLocalValue;

        Timer timer;
        double treeTime, runTime;
        Record record;

        if (config.direct) {
            auto srcSLCoord = point.srcLocalSL; // a copy
            auto srcSLValue = value.srcLocalSL; // a copy
            auto srcDLCoord = point.srcLocalDL; // a copy
            auto srcDLValue = value.srcLocalDL; // a copy
            auto trgLocalCoord = point.trgLocal;
            PointDistribution::collectPtsAll(srcSLCoord); // src from all ranks
            PointDistribution::collectPtsAll(srcSLValue); // src from all ranks
            PointDistribution::collectPtsAll(srcDLCoord); // src from all ranks
            PointDistribution::collectPtsAll(srcDLValue); // src from all ranks

            const int nSL = srcSLCoord.size() / 3;
            const int nDL = srcDLCoord.size() / 3;
            const int nTrg = trgLocalCoord.size() / 3;
            trgLocalValue.clear();
            trgLocalValue.resize(kdimTrg * nTrg, 0);

            timer.tick();
            fmmPtr->evaluateKernel(kernel, 0, PPKERNEL::SLS2T,                //
                                   nSL, srcSLCoord.data(), srcSLValue.data(), //
                                   nTrg, trgLocalCoord.data(), trgLocalValue.data());
            fmmPtr->evaluateKernel(kernel, 0, PPKERNEL::DLS2T,                //
                                   nDL, srcDLCoord.data(), srcDLValue.data(), //
                                   nTrg, trgLocalCoord.data(), trgLocalValue.data());
            timer.tock("evaluateKernel");

            const auto &time = timer.getTime();
            treeTime = 0;
            runTime = time[0];
        } else {
            double origin[3] = {config.origin[0], config.origin[1], config.origin[2]};
            const double box = config.box;
            const int nSL = point.srcLocalSL.size() / 3;
            const int nDL = point.srcLocalDL.size() / 3;
            const int nTrg = point.trgLocal.size() / 3;
            trgLocalValue.clear();
            trgLocalValue.resize(kdimTrg * nTrg, 0);

            fmmPtr->clearFMM(kernel);
            fmmPtr->setBox(origin, box);
            fmmPtr->setPoints(nSL, point.srcLocalSL.data(), nTrg, point.trgLocal.data(), nDL, point.srcLocalDL.data());

            timer.tick();
            fmmPtr->setupTree(kernel);
            timer.tock("setupTree");

            timer.tick();
            fmmPtr->evaluateFMM(kernel, nSL, value.srcLocalSL.data(), //
                                nTrg, trgLocalValue.data(),           //
                                nDL, value.srcLocalDL.data());
            timer.tock("evaluateFMM");
            const auto &time = timer.getTime();
            treeTime = time[0];
            runTime = time[1];
        }
        result[kernel] = trgLocalValue;
        timing[kernel] = std::make_pair(treeTime, runTime);
    }
}

void checkError(const int dim, const std::vector<double> &A, const std::vector<double> &B,
                std::vector<ComponentError> &error) {
    error.clear();
    // collect to rank0
    std::vector<double> value = A;
    std::vector<double> valueTrue = B;
    PointDistribution::collectPts(value);
    PointDistribution::collectPts(valueTrue);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank) {
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        // check error for each component on rank 0
        const int nPts = value.size() / dim;
        if (nPts * dim != valueTrue.size()) {
            printf("error check size error\n");
            exit(1);
        }
        std::vector<double> comp(nPts), compTrue(nPts);
        auto getComp = [&](int k) {
            comp.clear();
            compTrue.clear();
            comp.resize(nPts);
            compTrue.resize(nPts);
            for (int i = 0; i < nPts; i++) {
                comp[i] = value[i * dim + k];
                compTrue[i] = valueTrue[i * dim + k];
            }
        };
        for (int i = 0; i < dim; i++) {
            getComp(i);
            error.emplace_back(comp, compTrue);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
    return;
}

void appendHistory(std::vector<Record> &history, const int p, const Timing &timing, const Result &result,
                   const Result &verifyResult, const Result &convergeResult, const Result &translateResult) {
    for (auto &it : result) {
        auto &kernel = it.first;
        auto &trgValue = it.second;
        Record record;
        record.kernel = kernel;
        record.multOrder = p;

        // get time
        {
            auto it = timing.find(kernel);
            if (it != timing.end()) {
                record.treeTime = it->second.first;
                record.runTime = it->second.second;
            }
        }
        auto getError = [&](const Result &compare, std::vector<ComponentError> &error) {
            auto it = compare.find(kernel);
            if (it != compare.end()) {
                auto &compareValue = it->second;
                checkError(std::get<2>(stkfmm::getKernelDimension(kernel)), trgValue, compareValue, error);
            }
        };
        getError(verifyResult, record.errorVerify);
        getError(convergeResult, record.errorConvergence);
        getError(translateResult, record.errorTranslate);
        history.push_back(record);
    }
}

auto errorJson(const ComponentError &error) {
    using json = nlohmann::json;
    json output;
    output["drift"] = error.drift;
    output["driftL2"] = error.driftL2;
    output["errorL2"] = error.errorL2;
    output["errorRMS"] = error.errorRMS;
    output["errorMaxRel"] = error.errorMaxRel;
    return output;
};

void recordJson(const Config &config, const std::vector<Record> &history) {
    std::string filename("TestLog.json");
    // write settings and record to a json file
    using json = nlohmann::json;
    json output;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank) {
        return;
    }

    auto getJsonFromRecord = [&](const Record &record) -> json {
        json output;
        output["kernel"] = stkfmm::getKernelName(record.kernel);
        output["multOrder"] = record.multOrder;
        output["treeTime"] = record.treeTime;
        output["runTime"] = record.runTime;
        // Convert vec of struct to json
        auto eC = json::array(), eV = json::array(), eT = json::array();
        for (auto &e : record.errorConvergence) {
            eC.push_back(errorJson(e));
        }
        for (auto &e : record.errorVerify) {
            eV.push_back(errorJson(e));
        }
        for (auto &e : record.errorTranslate) {
            eT.push_back(errorJson(e));
        }
        output["errorConvergence"] = eC;
        output["errorVerify"] = eV;
        output["errorTranslate"] = eT;
        return output;
    };

    auto jsonObjects = json::array();
    for (auto &r : history) {
        jsonObjects.push_back(getJsonFromRecord(r));
    }

    std::ofstream o(filename);
    o << std::setw(4) << jsonObjects << std::endl;
}