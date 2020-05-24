#include "Test.hpp"

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    Config config;
    config.parse(argc, argv);
    config.print();

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Point point, trans_point;
    Input input;
    Result pResult, verifyResult, convResult, transResult;
    Timing timing;
    std::vector<Record> history;

    genPoint(config, point, 3);

    genSrcValue(config, point, input);
    if (config.pbc) {
        trans_point = point;
        translatePoint(config, trans_point);
    }

    printf_rank0("src value generated\n");

    if (config.verify) {
        if (config.wall) {
            // verify with zero on wall
            for (auto &k : input) {
                auto kernel = k.first;
                int kdimTrg = std::get<2>(stkfmm::getKernelDimension(kernel));
                int nTrg = point.trgLocal.size() / 3;
                verifyResult[kernel].resize(nTrg * kdimTrg, 0);
            }
        } else {
            runSimpleKernel(point, input, verifyResult);
        }
        dumpValue("verify", point, input, verifyResult);
    }

    if (config.convergence) {
        runFMM(config, config.maxOrder, point, input, convResult, timing);
        dumpValue("maxp" + std::to_string(config.maxOrder), point, input, convResult);
    }

    if (config.direct) {
        printf_rank0("*********Testing direct sum***********\n");
        pResult.clear();
        transResult.clear();
        timing.clear();
        int order = 2;
        runFMM(config, 2, point, input, pResult, timing);
        dumpValue("direct", point, input, pResult);
        appendHistory(history, 2, timing, pResult, verifyResult, convResult, transResult);
    } else {
        for (int p = 6; p < config.maxOrder; p += 2) {
            printf_rank0("*********Testing order p = %d*********\n", p);
            pResult.clear();
            transResult.clear();
            timing.clear();

            runFMM(config, p, point, input, pResult, timing);
            dumpValue("p" + std::to_string(p), point, input, pResult);

            if (config.pbc) {
                Timing transTiming;
                runFMM(config, p, trans_point, input, transResult, transTiming);
                dumpValue("trans_p" + std::to_string(p), trans_point, input, transResult);
            }

            appendHistory(history, p, timing, pResult, verifyResult, convResult, transResult);
        }
    }

    recordJson(config, history);

    MPI_Finalize();
    return 0;
}
