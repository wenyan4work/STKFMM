#include "STKFMM/STKFMM.hpp"

#include "SimpleKernel.hpp"

#include "Util/PointDistribution.hpp"
#include "Util/Timer.hpp"
#include "Util/cmdparser.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

using namespace stkfmm;

void configure_parser(cli::Parser &parser) {
    parser.set_optional<int>(
        "S", "nSLSource", 1,
        "1 for point force, 2 for force dipole, 4 for 4 point forces, other "
        "for same as target, default=1");
    parser.set_optional<int>(
        "D", "nDLSource", 1,
        "1 for point force, 2 for force dipole, 4 for 4 point forces, other "
        "for same as target, default=1");
    parser.set_optional<int>("s", "Seed", 1, "RNG Seed");
    parser.set_optional<int>("T", "nTarget", 2,
                             "total target number = (T+1)^3, default T=2");
    parser.set_optional<double>("B", "box", 1.0,
                                "box edge length, default B=1.0");
    parser.set_optional<double>("M", "move", 0.0,
                                "box origin shift move, default M=0");
    parser.set_optional<int>("K", "Kernel Combination", 0,
                             "any positive number for arbitrary combination of "
                             "kernels, default=0 means all kernels");
    parser.set_optional<int>("R", "Random", 1,
                             "0 for random, 1 for Chebyshev, default 1");
    parser.set_optional<int>(
        "F", "FMM", 1, "0 for test S2T kernel, 1 for test FMM, default 1");
    parser.set_optional<int>("V", "Verify", 1,
                             "2 for translational invariance verification, 1 "
                             "for O(N^2) verification, 0 for false, default 1");
    parser.set_optional<int>(
        "P", "Periodic", 0,
        "0 for NONE, 1 for PX, 2 for PXY, 3 for PXYZ, default 0");
}

void showOption(const cli::Parser &parser) {
    std::cout << "Running setting: " << std::endl;
    std::cout << "nSLSource: " << parser.get<int>("S") << std::endl;
    std::cout << "nDLSource: " << parser.get<int>("D") << std::endl;
    std::cout << "nTarget: " << parser.get<int>("T") << std::endl;
    std::cout << "Box: " << parser.get<double>("B") << std::endl;
    std::cout << "Shift: " << parser.get<double>("M") << std::endl;
    std::cout << "KERNEL: " << parser.get<int>("K") << std::endl;
    std::cout << "Random: " << parser.get<int>("R") << std::endl;
    std::cout << "Using FMM: " << parser.get<int>("F") << std::endl;
    std::cout << "Verification: " << parser.get<int>("V") << std::endl;
    std::cout << "Periodic BC: " << parser.get<int>("P") << std::endl;
}

void calcFMMShifted(STKFMM &myFMM, KERNEL testKernel,
                    std::vector<double> &srcSLCoordLocal,
                    std::vector<double> &srcDLCoordLocal,
                    std::vector<double> &trgCoordLocal,
                    const std::vector<double> &srcSLValueLocal,
                    const std::vector<double> &srcDLValueLocal,
                    std::vector<double> &trgValueShifted, int kdimTrg) {
    double rlow[3], rhigh[3];
    myFMM.getBox(rlow[0], rhigh[0], rlow[1], rhigh[1], rlow[2], rhigh[2]);

    std::vector<double> shift(3, 0.5);
    for (int i = 0; i < 3; ++i)
        shift[i] = (rhigh[i] - rlow[i]) * shift[i] + rlow[i];

    int n_periodic = pvfmm::periodicType;

    auto shiftCoords = [n_periodic, shift, rlow, rhigh](std::vector<double> &r,
                                                        int sign) {
        for (int i = 0; i < r.size() / 3; ++i) {
            for (int j = 0; j < n_periodic; ++j) {
                r[i * 3 + j] += sign * shift[j];
                double dr = rhigh[j] - rlow[j];
                r[i * 3 + j] -= (r[i * 3 + j] >= rhigh[j]) ? dr : 0.0;
                r[i * 3 + j] += (r[i * 3 + j] <= rlow[j]) ? dr : 0.0;
            }
        }
    };
    shiftCoords(srcSLCoordLocal, 1);
    shiftCoords(srcDLCoordLocal, 1);
    shiftCoords(trgCoordLocal, 1);

    myFMM.setPoints(srcSLCoordLocal.size() / 3, srcSLCoordLocal.data(),
                    srcDLCoordLocal.size() / 3, srcDLCoordLocal.data(),
                    trgCoordLocal.size() / 3, trgCoordLocal.data());

    myFMM.clearFMM(testKernel);
    myFMM.setupTree(testKernel);
    myFMM.evaluateFMM(srcSLCoordLocal.size() / 3, srcSLValueLocal.data(),
                      srcDLCoordLocal.size() / 3, srcDLValueLocal.data(),
                      trgCoordLocal.size() / 3, trgValueShifted.data(),
                      testKernel);

    PointDistribution::dumpPoints("trgPointsShifted" +
                                      std::to_string((uint)testKernel) + ".txt",
                                  trgCoordLocal, trgValueShifted, kdimTrg);

    shiftCoords(srcSLCoordLocal, -1);
    shiftCoords(srcDLCoordLocal, -1);
    shiftCoords(trgCoordLocal, -1);

    myFMM.setPoints(srcSLCoordLocal.size() / 3, srcSLCoordLocal.data(),
                    srcDLCoordLocal.size() / 3, srcDLCoordLocal.data(),
                    trgCoordLocal.size() / 3, trgCoordLocal.data());
}

void calcTrueValue(KERNEL kernel, const int kdimSL, const int kdimDL,
                   const int kdimTrg,
                   const std::vector<double> &srcSLCoordLocal,
                   const std::vector<double> &srcDLCoordLocal,
                   const std::vector<double> &trgCoordLocal,
                   const std::vector<double> &srcSLValueLocal,
                   const std::vector<double> &srcDLValueLocal,
                   std::vector<double> &trgValueTrueLocal) {
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // create a copy for MPI
    std::vector<double> srcSLCoordGlobal = srcSLCoordLocal;
    std::vector<double> srcDLCoordGlobal = srcDLCoordLocal;
    std::vector<double> srcSLValueGlobal = srcSLValueLocal;
    std::vector<double> srcDLValueGlobal = srcDLValueLocal;
    std::vector<double> trgCoordGlobal = trgCoordLocal;

    // src is fully repeated on every node
    // trg remains distributed
    PointDistribution::collectPtsAll(srcSLCoordGlobal);
    PointDistribution::collectPtsAll(srcDLCoordGlobal);
    PointDistribution::collectPtsAll(srcSLValueGlobal);
    PointDistribution::collectPtsAll(srcDLValueGlobal);

    // on every node, from global src to local trg
    const int nSL = srcSLCoordGlobal.size() / 3;
    const int nDL = srcDLCoordGlobal.size() / 3;
    const int nTrg = trgCoordLocal.size() / 3;

    // Create mapping of kernels to 'true value' functions
    using std::make_pair;
    typedef void (*kernel_func)(double *, double *, double *, double *);
    // clang-format off
    std::unordered_map<KERNEL, std::pair<kernel_func, kernel_func>> SL_kernels(
        {{KERNEL::PVel, make_pair(StokesSLPVel, StokesDLPVel)},
         {KERNEL::PVelGrad, make_pair(StokesSLPVelGrad, StokesDLPVelGrad)},
         {KERNEL::Traction, make_pair(StokesSLTraction, StokesSLTraction)},
         {KERNEL::PVelLaplacian, make_pair(StokesSLPVelLaplacian, StokesSLPVelLaplacian)},
         {KERNEL::LAPPGrad, make_pair(LaplaceSLPGrad, LaplaceDLPGrad)},
         {KERNEL::StokesRegVel, make_pair(StokesRegSLVel, StokesRegDLVel)},
         {KERNEL::StokesRegVelOmega, make_pair(StokesRegSLVelOmega, StokesRegDLVelOmega)},
         {KERNEL::RPY, make_pair(StokesSLRPY, StokesDLRPY)}});
    // clang-format on

    kernel_func kernelTestSL, kernelTestDL;
    std::tie(kernelTestSL, kernelTestDL) = SL_kernels[kernel];

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
                trgValueTrueLocal[kdimTrg * i + k] += result[k];
            }
        }

        // add DL values
        for (int j = 0; j < nDL; j++) {
            double result[20] = {0.0};
            double *s = srcDLCoordGlobal.data() + 3 * j;
            double *sval = srcDLValueGlobal.data() + kdimDL * j;

            kernelTestDL(s, t, sval, result);

            for (int k = 0; k < kdimTrg; k++) {
                trgValueTrueLocal[kdimTrg * i + k] += result[k];
            }
        }
    }

    return;
}

void testOneKernelS2T(STKFMM &myFMM, KERNEL testKernel,
                      std::vector<double> &srcSLCoordLocal,
                      std::vector<double> &srcDLCoordLocal,
                      std::vector<double> &trgCoordLocal,
                      std::vector<double> &srcSLValueLocal,
                      std::vector<double> &srcDLValueLocal, uint verify = 1) {
    // test S2T kernel, on rank 0 only
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    int kdimSL, kdimDL, kdimTrg;
    myFMM.getKernelDimension(kdimSL, kdimDL, kdimTrg, testKernel);
    if (myRank == 0)
        printf("kdim: SL %d, DL %d, TRG %d\n", kdimSL, kdimDL, kdimTrg);
    int nSrcSLLocal = srcSLCoordLocal.size() / 3;
    int nSrcDLLocal = srcDLCoordLocal.size() / 3;
    int nTrgLocal = trgCoordLocal.size() / 3;

    std::vector<double> trgValueLocal(nTrgLocal * kdimTrg);
    std::vector<double> trgValueTrueLocal(nTrgLocal * kdimTrg);

    myFMM.evaluateKernel(-1, PPKERNEL::SLS2T, nSrcSLLocal,
                         srcSLCoordLocal.data(), srcSLValueLocal.data(),
                         nTrgLocal, trgCoordLocal.data(), trgValueLocal.data(),
                         testKernel); // SL
    if (myRank == 0)
        printf("SLS2T kernel evaluated\n");

    myFMM.evaluateKernel(-1, PPKERNEL::DLS2T, nSrcDLLocal,
                         srcDLCoordLocal.data(), srcDLValueLocal.data(),
                         nTrgLocal, trgCoordLocal.data(), trgValueLocal.data(),
                         testKernel); // DL

    if (verify == 1) {
        if (myRank == 0)
            printf("DLS2T kernel evaluated\n");
        calcTrueValue(testKernel, kdimSL, kdimDL, kdimTrg, srcSLCoordLocal,
                      srcDLCoordLocal, trgCoordLocal, srcSLValueLocal,
                      srcDLValueLocal, trgValueTrueLocal);
        PointDistribution::checkError(trgValueLocal, trgValueTrueLocal);

        // output for debug
        PointDistribution::dumpPoints(
            "srcSLPoints" + std::to_string(((uint)testKernel)) + ".txt",
            srcSLCoordLocal, srcSLValueLocal, kdimSL);
        PointDistribution::dumpPoints(
            "srcDLPoints" + std::to_string(((uint)testKernel)) + ".txt",
            srcDLCoordLocal, srcDLValueLocal, kdimDL);
        PointDistribution::dumpPoints(
            "trgPoints" + std::to_string(((uint)testKernel)) + ".txt",
            trgCoordLocal, trgValueLocal, kdimTrg);
        PointDistribution::dumpPoints(
            "trgPointsTrue" + std::to_string(((uint)testKernel)) + ".txt",
            trgCoordLocal, trgValueTrueLocal, kdimTrg);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void testOneKernelFMM(STKFMM &myFMM, KERNEL testKernel,
                      std::vector<double> &srcSLCoordLocal,
                      std::vector<double> &srcDLCoordLocal,
                      std::vector<double> &trgCoordLocal,
                      std::vector<double> &srcSLValueLocal,
                      std::vector<double> &srcDLValueLocal, uint verify = 1) {
    // srcSLCoord, srcDLCoord, trgCoord are distributed

    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // srcCoord and trgCoord are distributed
    int kdimSL, kdimDL, kdimTrg;
    myFMM.getKernelDimension(kdimSL, kdimDL, kdimTrg, testKernel);
    if (myRank == 0) {
        printf("kdim: SL %d, DL %d, TRG %d\n", kdimSL, kdimDL, kdimTrg);
    }

    int nSrcSLLocal = srcSLCoordLocal.size() / 3;
    int nSrcDLLocal = srcDLCoordLocal.size() / 3;
    int nTrgLocal = trgCoordLocal.size() / 3;

    std::vector<double> trgValueLocal(nTrgLocal * kdimTrg, 0);

    // FMM1
    Timer timer;
    timer.tick();
    myFMM.setupTree(testKernel);
    timer.tock("Tree setup ");
    timer.tick();
    myFMM.evaluateFMM(nSrcSLLocal, srcSLValueLocal.data(), nSrcDLLocal,
                      srcDLValueLocal.data(), nTrgLocal, trgValueLocal.data(),
                      testKernel);
    timer.tock("FMM Evaluation ");
    if (myRank == 0)
        timer.dump();

    if (verify == 1) {
        if (myRank == 0)
            printf("fmm evaluated, computing true results with simple O(N^2) "
                   "sum\n");
        std::vector<double> trgValueTrueLocal(nTrgLocal * kdimTrg, 0);
        calcTrueValue(testKernel, kdimSL, kdimDL, kdimTrg, srcSLCoordLocal,
                      srcDLCoordLocal, trgCoordLocal, srcSLValueLocal,
                      srcDLValueLocal, trgValueTrueLocal);
        PointDistribution::checkError(trgValueLocal, trgValueTrueLocal);

        // output for debug
        PointDistribution::dumpPoints(
            "srcSLPoints" + std::to_string(((uint)testKernel)) + ".txt",
            srcSLCoordLocal, srcSLValueLocal, kdimSL);
        PointDistribution::dumpPoints(
            "srcDLPoints" + std::to_string(((uint)testKernel)) + ".txt",
            srcDLCoordLocal, srcDLValueLocal, kdimDL);
        PointDistribution::dumpPoints(
            "trgPoints" + std::to_string(((uint)testKernel)) + ".txt",
            trgCoordLocal, trgValueLocal, kdimTrg);
        PointDistribution::dumpPoints(
            "trgPointsTrue" + std::to_string(((uint)testKernel)) + ".txt",
            trgCoordLocal, trgValueTrueLocal, kdimTrg);
    } else if (verify == 2) {
        if (myRank == 0)
            printf("fmm evaluated, computing result with periodic dimensions "
                   "shifted\n");
        std::vector<double> trgValueLocalShifted(nTrgLocal * kdimTrg, 0);

        calcFMMShifted(myFMM, testKernel, srcSLCoordLocal, srcDLCoordLocal,
                       trgCoordLocal, srcSLValueLocal, srcDLValueLocal,
                       trgValueLocalShifted, kdimTrg);

        PointDistribution::dumpPoints(
            "srcSLPoints" + std::to_string(((uint)testKernel)) + ".txt",
            srcSLCoordLocal, srcSLValueLocal, kdimSL);
        PointDistribution::dumpPoints(
            "srcDLPoints" + std::to_string(((uint)testKernel)) + ".txt",
            srcDLCoordLocal, srcDLValueLocal, kdimDL);
        PointDistribution::dumpPoints(
            "trgPoints" + std::to_string(((uint)testKernel)) + ".txt",
            trgCoordLocal, trgValueLocal, kdimTrg);

        PointDistribution::checkError(trgValueLocal, trgValueLocalShifted);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void testFMM(const cli::Parser &parser, int order) {
    const double shift = parser.get<double>("M");
    const double box = parser.get<double>("B");
    const int temp = parser.get<int>("K");
    const int k = (temp == 0) ? ~((int)0) : temp;
    const int paxis = parser.get<int>("P");
    STKFMM myFMM(order, 2000, (PAXIS)paxis, k);
    myFMM.setBox(shift, shift + box, shift, shift + box, shift, shift + box);
    myFMM.showActiveKernels();

    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    std::vector<double> srcSLCoord;
    std::vector<double> srcDLCoord;
    std::vector<double> trgCoord;

    PointDistribution pd(parser.get<int>("s"));

    if (myRank == 0) {
        // set trg coord
        const int nPts = parser.get<int>("T");
        if (parser.get<int>("R") > 0) {
            pd.randomPoints(nPts, box, shift, trgCoord);
        } else {
            PointDistribution::chebPoints(nPts, box, shift, trgCoord);
        }
        // set src SL coord
        const int nSL = parser.get<int>("S");
        if (nSL == 0) {
            srcSLCoord.clear();
        } else if (nSL == 1 || nSL == 2 || nSL == 4) {
            PointDistribution::fixedPoints(nSL, box, shift, srcSLCoord);
        } else {
            srcSLCoord = trgCoord;
        }

        const int nDL = parser.get<int>("D");
        if (nDL == 0) {
            srcDLCoord.clear();
        } else if (nDL == 1 || nDL == 2 || nDL == 4) {
            PointDistribution::fixedPoints(nDL, box, shift, srcDLCoord);
        } else {
            srcDLCoord = trgCoord;
        }

    } else {
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const uint verify = parser.get<int>("V");
    const uint nPeriodic = parser.get<int>("P");
    if (verify == 1 && nPeriodic > 0) {
        std::cout << "Periodic boundary conditions currently incompatible with "
                     "N^2 check\n";
        exit(1);
    }

    if (myRank == 0) {
        std::cout << "nSL: " << srcSLCoord.size() / 3 << "\n";
        std::cout << "nDL: " << srcDLCoord.size() / 3 << "\n";
        std::cout << "nTrg: " << trgCoord.size() / 3 << std::endl;
    }

    std::vector<KERNEL> kernels = {
        KERNEL::PVel,
        KERNEL::PVelGrad,
        KERNEL::PVelLaplacian,
        KERNEL::Traction,
        KERNEL::LAPPGrad,
        KERNEL::StokesRegVel,
        KERNEL::StokesRegVelOmega,
        KERNEL::RPY,
    };

    // test each active kernel
    int nSrcSL, nSrcDL;

    if (myRank == 0) {
        nSrcSL = srcSLCoord.size() / 3;
        nSrcDL = srcDLCoord.size() / 3;
        MPI_Bcast(&nSrcSL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nSrcDL, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        nSrcSL = nSrcDL = 0;
        MPI_Bcast(&nSrcSL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nSrcDL, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    PointDistribution::distributePts(srcSLCoord, 3);
    PointDistribution::distributePts(srcDLCoord, 3);
    PointDistribution::distributePts(trgCoord, 3);
    myFMM.setPoints(srcSLCoord.size() / 3, srcSLCoord.data(),
                    srcDLCoord.size() / 3, srcDLCoord.data(),
                    trgCoord.size() / 3, trgCoord.data());

    for (auto testKernel : kernels) {
        if (!myFMM.isKernelActive(testKernel))
            continue;

        int kdimSL, kdimDL, kdimTrg;
        myFMM.getKernelDimension(kdimSL, kdimDL, kdimTrg, testKernel);

        std::vector<double> srcSLValue;
        std::vector<double> srcDLValue;

        if (myRank == 0) {
            srcSLValue.resize(nSrcSL * kdimSL);
            srcDLValue.resize(nSrcDL * kdimDL);
            pd.randomUniformFill(srcSLValue, -1, 1);
            pd.randomUniformFill(srcDLValue, -1, 1);

            if (testKernel == KERNEL::LAPPGrad) {
                std::cout << "Zeroing Laplace Kernel charge\n";
                double charge =
                    std::accumulate(srcSLValue.begin(), srcSLValue.end(), 0.0);
                for (auto &el : srcSLValue)
                    el -= charge / nSrcSL;
            }

            if (testKernel == KERNEL::RPY) {
                const double eps = 0.01;
                for (int i = 3; i < srcSLValue.size(); i += 4) {
                    srcSLValue[i] = eps * (srcSLValue[i] + 1);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        PointDistribution::distributePts(srcSLValue, kdimSL);
        PointDistribution::distributePts(srcDLValue, kdimDL);

        if (parser.get<int>("F") == 1) {
            testOneKernelFMM(myFMM, testKernel, srcSLCoord, srcDLCoord,
                             trgCoord, srcSLValue, srcDLValue, verify);
        } else {
            testOneKernelS2T(myFMM, testKernel, srcSLCoord, srcDLCoord,
                             trgCoord, srcSLValue, srcDLValue, verify);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    if (myRank == 0)
        showOption(parser);

    for (int p = 6; p <= 14; p += 2) {
        if (myRank == 0) {
            printf("------------------------------------\n");
            printf("Testing order p = %d\n", p);
        }
        testFMM(parser, p);
        if (myRank == 0)
            printf("------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}
