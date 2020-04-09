#include "Test.hpp"

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    if (myRank == 0)
        showOption(parser);

    FMMpoint point;
    FMMinput inputs;
    FMMresult true_results;

    genPoint(parser, point);
    genSrcValue(parser, point, inputs, true);
    printf("src value generated\n");

    if (parser.get<int>("V")) {
        runSimpleKernel(point, inputs, true_results);
    } else {
        runFMM(parser, maxP, point, inputs, true_results);
    }

    dumpValue("true", point, inputs, true_results);

    for (int p = 6; p < maxP; p += 2) {
        FMMresult results;
        // check error vs trueValues
        runFMM(parser, p, point, inputs, results);
        if (myRank == 0) {
            printf("*********Testing order p = %d*********\n", p);
            printf("---------Error vs \"True\" Value------\n");
        }
        checkError(results, true_results, true);
        dumpValue("p" + std::to_string(p), point, inputs, results);

        // check error vs translational shift
        const int pbc = parser.get<int>("P");
        if (pbc) {
            FMMpoint trans_point = point;
            FMMresult trans_results;
            translatePoints(parser, trans_point);
            runFMM(parser, p, trans_point, inputs, trans_results);
            if (myRank == 0) {
                printf("---------Error vs Translation------\n");
            }
            checkError(results, trans_results, true);
            dumpValue("trans_p" + std::to_string(p), trans_point, inputs,
                      trans_results);
        }

        if (myRank == 0)
            printf("------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}
