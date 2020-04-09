#include "Test.hpp"

#include <mpi.h>

void checkWallError(const cli::Parser &parser, const FMMpoint &point, const FMMresult &results) {
    const double Z = parser.get<double>("M");
    const double B = parser.get<double>("B");
    const int nTrg = point.trgLocal.size() / 3;
    for (auto &data : results) {
        auto kernel = data.first;
        std::vector<double> v;
        for (int i = 0; i < nTrg; i++) {
            if ((point.trgLocal[3 * i + 2] - Z) < 1e-10 * B) {
                v.push_back(data.second[3 * i]);
                v.push_back(data.second[3 * i + 1]);
                v.push_back(data.second[3 * i + 2]);
            }
        }

        std::vector<double> zero(v.size(), 0);
        PointDistribution::checkError(v, zero, 3);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    if (myRank == 0) {
        printf("The options V and F have no effect.\n");
        printf("Check wall velocity error only when R=0.\n");
        showOption(parser);
    }

    FMMpoint point;
    FMMinput inputs;
    FMMresult true_results;

    genPoint(parser, point, true);
    genSrcValue(parser, point, inputs, false);
    printf("src value generated\n");

    runFMM(parser, maxP, point, inputs, true_results, true);
    dumpValue("true", point, inputs, true_results);

    for (int p = 6; p < maxP; p += 2) {
        FMMresult results;
        // check error vs trueValues
        runFMM(parser, p, point, inputs, results, true);
        if (myRank == 0) {
            printf("*********Testing order p = %d*********\n", p);
            printf("---------Error vs \"True\" Value------\n");
        }
        checkError(results, true_results, true);
        dumpValue("p" + std::to_string(p), point, inputs, results);

        // check error on wall
        if (parser.get<int>("R") == 0) {
            if (myRank == 0)
                printf("---------Error on the Wall------\n");
            checkWallError(parser, point, results);
        }

        // check error vs translational shift
        const int pbc = parser.get<int>("P");
        if (pbc) {
            FMMpoint trans_point = point;
            FMMresult trans_results;
            translatePoints(parser, trans_point);
            runFMM(parser, p, trans_point, inputs, trans_results, true);
            if (myRank == 0) {
                printf("---------Error vs Translation------\n");
            }
            checkError(results, trans_results, true);
            dumpValue("trans_p" + std::to_string(p), trans_point, inputs, trans_results);
        }

        if (myRank == 0)
            printf("------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}
