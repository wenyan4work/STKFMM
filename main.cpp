#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

#include "PointDistribution.h"
#include "STKFMM.h"
#include "SimpleKernel.h"
#include "Util/cmdparser.hpp"

using namespace stkfmm;

void configure_parser(cli::Parser &parser) {
    parser.set_optional<int>(
        "S", "nSLSource", 1,
        "1 for point force, 2 for force dipole, 4 for 4 point forces, other for same as target, default=1");
    parser.set_optional<int>(
        "D", "nDLSource", 1,
        "1 for point force, 2 for force dipole, 4 for 4 point forces, other for same as target, default=1");
    parser.set_optional<int>("T", "nTarget", 2, "total target number = (T+1)^3, default T=2");
    parser.set_optional<double>("B", "box", 1.0, "box edge length, default B=1.0");
    parser.set_optional<double>("M", "move", 0.0, "box origin shift move, default M=0");
    parser.set_optional<int>("K", "Kernel Combination", 0,
                             "any positive number for arbitrary combination of kernels, default=0 means all kernels");
    parser.set_optional<int>("R", "Random", 1, "0 for random, 1 for Chebyshev, default 1");
    parser.set_optional<int>("F", "FMM", 1, "0 for test S2T kernel, 1 for test FMM, default 1");
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
}

void calcTrueValue(KERNEL kernel, const int kdimSL, const int kdimDL, const int kdimTrg,
                   const std::vector<double> &srcSLCoordLocal, const std::vector<double> &srcDLCoordLocal,
                   const std::vector<double> &trgCoordLocal, const std::vector<double> &srcSLValueLocal,
                   const std::vector<double> &srcDLValueLocal, std::vector<double> &trgValueTrueLocal) {
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
    collectPtsAll(srcSLCoordGlobal);
    collectPtsAll(srcDLCoordGlobal);
    collectPtsAll(srcSLValueGlobal);
    collectPtsAll(srcDLValueGlobal);

    // on every node, from global src to local trg
    const int nSL = srcSLCoordGlobal.size() / 3;
    const int nDL = srcDLCoordGlobal.size() / 3;
    const int nTrg = trgCoordLocal.size() / 3;

    // check results
#pragma omp parallel for
    for (int i = 0; i < nTrg; i++) {
        double t[3];
        const double *trg = trgCoordLocal.data() + 3 * i;
        t[0] = trg[0];
        t[1] = trg[1];
        t[2] = trg[2];

        // add SL values
        for (int j = 0; j < nSL; j++) {
            double result[20];
            for (int ii = 0; ii < 20; ii++) {
                result[ii] = 0;
            }
            double *s = srcSLCoordGlobal.data() + 3 * j;
            double *sval = srcSLValueGlobal.data() + kdimSL * j;

            switch (kernel) {
            case KERNEL::PVel:
                StokesSLPVel(s, t, sval, result);
                break;
            case KERNEL::PVelGrad:
                StokesSLPVelGrad(s, t, sval, result);
                break;
            case KERNEL::Traction:
                StokesSLTraction(s, t, sval, result);
                break;
            case KERNEL::PVelLaplacian:
                StokesSLPVelLaplacian(s, t, sval, result);
                break;
            case KERNEL::LAPPGrad:
                LaplaceSLPGrad(s, t, sval, result);
                break;
            }

            for (int k = 0; k < kdimTrg; k++) {
                trgValueTrueLocal[kdimTrg * i + k] += result[k];
            }
        }

        // add DL values
        for (int j = 0; j < nDL; j++) {
            double result[20];
            for (int ii = 0; ii < 20; ii++) {
                result[ii] = 0;
            }
            double *s = srcDLCoordGlobal.data() + 3 * j;
            double *sval = srcDLValueGlobal.data() + kdimDL * j;

            switch (kernel) {
            case KERNEL::PVel:
                StokesDLPVel(s, t, sval, result);
                break;
            case KERNEL::PVelGrad:
                StokesDLPVelGrad(s, t, sval, result);
                break;
            case KERNEL::Traction:
                StokesDLTraction(s, t, sval, result);
                break;
            case KERNEL::PVelLaplacian:
                StokesDLPVelLaplacian(s, t, sval, result);
                break;
            case KERNEL::LAPPGrad:
                LaplaceDLPGrad(s, t, sval, result);
                break;
            }

            for (int k = 0; k < kdimTrg; k++) {
                trgValueTrueLocal[kdimTrg * i + k] += result[k];
            }
        }
    }

    return;
}

void testOneKernelS2T(STKFMM &myFMM, KERNEL testKernel, std::vector<double> &srcSLCoordLocal,
                      std::vector<double> &srcDLCoordLocal, std::vector<double> &trgCoordLocal) {
    // test one S2T kernel. on rank 0 only.
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

    std::vector<double> srcSLValueLocal;
    std::vector<double> srcDLValueLocal;
    std::vector<double> trgValueLocal;
    std::vector<double> trgValueTrueLocal;

    int nSrcSLLocal = srcSLCoordLocal.size() / 3;
    int nSrcDLLocal = srcDLCoordLocal.size() / 3;
    int nTrgLocal = trgCoordLocal.size() / 3;

    srcSLValueLocal.resize(nSrcSLLocal * kdimSL, 0);
    // randomUniformFill(srcSLValueLocal, -1, 1);

    srcDLValueLocal.resize(nSrcDLLocal * kdimDL);
    randomUniformFill(srcDLValueLocal, -1, 1);

    trgValueLocal.resize(nTrgLocal * kdimTrg);
    trgValueTrueLocal.resize(nTrgLocal * kdimTrg);

    printf("S2T kernel evaluated\n");
    myFMM.evaluateKernel(true, false, nSrcSLLocal, srcSLCoordLocal.data(), srcSLValueLocal.data(), nTrgLocal,
                         trgCoordLocal.data(), trgValueLocal.data(), testKernel); // SL
    myFMM.evaluateKernel(true, false, nSrcDLLocal, srcDLCoordLocal.data(), srcDLValueLocal.data(), nTrgLocal,
                         trgCoordLocal.data(), trgValueLocal.data(), testKernel); // DL

    calcTrueValue(testKernel, kdimSL, kdimDL, kdimTrg, srcSLCoordLocal, srcDLCoordLocal, trgCoordLocal, srcSLValueLocal,
                  srcDLValueLocal, trgValueTrueLocal);
    checkError(trgValueLocal, trgValueTrueLocal);

    // output for debug
    dumpPoints("srcSLPoints.txt", srcSLCoordLocal, srcSLValueLocal, kdimSL);
    dumpPoints("srcDLPoints.txt", srcDLCoordLocal, srcDLValueLocal, kdimDL);
    dumpPoints("trgPoints.txt", trgCoordLocal, trgValueLocal, kdimTrg);
    dumpPoints("trgPointsTrue.txt", trgCoordLocal, trgValueTrueLocal, kdimTrg);

    MPI_Barrier(MPI_COMM_WORLD);
}

void testOneKernelFMM(STKFMM &myFMM, KERNEL testKernel, std::vector<double> &srcSLCoordLocal,
                      std::vector<double> &srcDLCoordLocal, std::vector<double> &trgCoordLocal) {
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

    std::vector<double> srcSLValueLocal;
    std::vector<double> srcDLValueLocal;
    std::vector<double> trgValueLocal;
    std::vector<double> trgValueTrueLocal;

    int nSrcSLLocal = srcSLCoordLocal.size() / 3;
    int nSrcDLLocal = srcDLCoordLocal.size() / 3;
    int nTrgLocal = trgCoordLocal.size() / 3;

    srcSLValueLocal.resize(nSrcSLLocal * kdimSL, 1);
    // randomUniformFill(srcSLValueLocal, -1, 1);

    srcDLValueLocal.resize(nSrcDLLocal * kdimDL, 1);
    // trace-free debugging
    // randomUniformFill(srcDLValueLocal, -1, 1);

    trgValueLocal.resize(nTrgLocal * kdimTrg, 0);
    trgValueTrueLocal.resize(nTrgLocal * kdimTrg, 0);

    // FMM1
    myFMM.setupTree(testKernel);
    myFMM.evaluateFMM(srcSLValueLocal, srcDLValueLocal, trgValueLocal, testKernel);

    // FMM2
    // randomUniformFill(srcValue, -1, 1);

    // myFMM.clearFMM(testKernel);
    // myFMM.setupTree(testKernel);
    // myFMM.evaluateFMM(srcValue, trgValue, testKernel);
    printf("fmm evaluated\n");

    calcTrueValue(testKernel, kdimSL, kdimDL, kdimTrg, srcSLCoordLocal, srcDLCoordLocal, trgCoordLocal, srcSLValueLocal,
                  srcDLValueLocal, trgValueTrueLocal);
    checkError(trgValueLocal, trgValueTrueLocal);

    // output for debug
    dumpPoints("srcSLPoints.txt", srcSLCoordLocal, srcSLValueLocal, kdimSL);
    dumpPoints("srcDLPoints.txt", srcDLCoordLocal, srcDLValueLocal, kdimDL);
    dumpPoints("trgPoints.txt", trgCoordLocal, trgValueLocal, kdimTrg);
    dumpPoints("trgPointsTrue.txt", trgCoordLocal, trgValueTrueLocal, kdimTrg);

    MPI_Barrier(MPI_COMM_WORLD);
}

void testFMM(const cli::Parser &parser, int order) {
    const double shift = parser.get<double>("M");
    const double box = parser.get<double>("B");
    const int temp = parser.get<int>("K");
    const size_t k = (temp == 0) ? ~((size_t)0) : temp;
    STKFMM myFMM(order, 100, PAXIS::NONE, k);
    myFMM.setBox(shift, shift + box, shift, shift + box, shift, shift + box);
    myFMM.showActiveKernels();

    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    std::vector<double> srcSLCoord;
    std::vector<double> srcDLCoord;
    std::vector<double> trgCoord;

    if (myRank == 0) {
        // set trg coord
        const int nPts = parser.get<int>("T");
        if (parser.get<int>("R") > 0) {
            randomPoints(nPts, box, shift, trgCoord);
        } else {
            chebPoints(nPts, box, shift, trgCoord);
        }
        // set src SL coord
        const int nSL = parser.get<int>("S");
        if (nSL == 1 || nSL == 2 || nSL == 4) {
            fixedPoints(nSL, box, shift, srcSLCoord);
        } else {
            srcSLCoord = trgCoord;
        }

        const int nDL = parser.get<int>("D");
        if (nDL == 1 || nDL == 2 || nDL == 4) {
            fixedPoints(nDL, box, shift, srcDLCoord);
        } else {
            srcDLCoord = trgCoord;
        }

    } else {
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // test each active kernel
    if (parser.get<int>("F") == 1) {
        distributePts(srcSLCoord, 3);
        distributePts(srcDLCoord, 3);
        distributePts(trgCoord, 3);
        myFMM.setPoints(srcSLCoord, srcDLCoord, trgCoord);

        if (myFMM.isKernelActive(KERNEL::PVel)) {
            testOneKernelFMM(myFMM, KERNEL::PVel, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::PVelGrad)) {
            testOneKernelFMM(myFMM, KERNEL::PVelGrad, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::PVelLaplacian)) {
            testOneKernelFMM(myFMM, KERNEL::PVelLaplacian, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::Traction)) {
            testOneKernelFMM(myFMM, KERNEL::Traction, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::LAPPGrad)) {
            testOneKernelFMM(myFMM, KERNEL::LAPPGrad, srcSLCoord, srcDLCoord, trgCoord);
        }
    } else {
        // test S2T kernel, on rank 0 only
        // TODO: this is not fully implemented yet
        if (myFMM.isKernelActive(KERNEL::PVel)) {
            testOneKernelS2T(myFMM, KERNEL::PVel, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::PVelGrad)) {
            testOneKernelS2T(myFMM, KERNEL::PVelGrad, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::PVelLaplacian)) {
            testOneKernelS2T(myFMM, KERNEL::PVelLaplacian, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::Traction)) {
            testOneKernelS2T(myFMM, KERNEL::Traction, srcSLCoord, srcDLCoord, trgCoord);
        }
        if (myFMM.isKernelActive(KERNEL::LAPPGrad)) {
            testOneKernelS2T(myFMM, KERNEL::LAPPGrad, srcSLCoord, srcDLCoord, trgCoord);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    showOption(parser);
    for (int p = 10; p <= 10; p += 2) {
        testFMM(parser, p);
    }

    MPI_Finalize();
    return 0;
}
