#include "Test.hpp"

#include <mpi.h>

void checkWallError(const cli::Parser &parser, const FMMpoint &point, const FMMresult &results) {
    const double Z = parser.get<double>("M");
    const double B = parser.get<double>("B");
    const int nTrg = point.trgLocal.size() / 3;
    for (auto &data : results) {
        auto kernel = data.first;
        std::vector<double> zero(data.second.size(), 0);
        int kdimSL, kdimDL, kdimTrg;
        std::tie(kdimSL, kdimDL, kdimTrg) = stkfmm::getKernelDimension(kernel);
        PointDistribution::checkError(data.second, zero, kdimTrg);
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
        printf("The option D must be zero.\n");
        printf("The option F has no effect.\n");
        printf("For this Wall test, V=1 means verify (T+1)^2 trg points on the wall.\n");
        showOption(parser);
    }

    FMMpoint point;
    FMMinput inputs;
    FMMresult true_results;

    genPoint(3, parser, point, true);
    const int V = parser.get<int>("V");
    if (V) {
        FMMpoint wall_point;
        genPoint(2, parser, wall_point, true);
        point.trgLocal = wall_point.trgLocal;
    }

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

        const int pbc = parser.get<int>("P");
        if (V) {
            // check error on wall
            if (myRank == 0)
                printf("---------Error on the Wall------\n");
            checkWallError(parser, point, results);
        } else if (pbc) {
            // check error vs translational shift
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
