#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../Test/SimpleKernel.hpp"

#include "mpi.h"
#include "omp.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    double reps = 1e-4;
    std::vector<double> srcCoord = {0.6 + reps, 0.6 + 2 * reps, 0.6 + 3 * reps, 0.6, 0.6, 0.6};
    std::vector<double> trgCoord = srcCoord;
    std::vector<double> srcValue = {0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4};
    std::vector<double> trgValue(2 * 9, 0.0);
    std::vector<double> trgValue2(2 * 9, 0.0);

    double origin[3] = {0, 0, 0};
    double box = 1;

    {
        StokesSLTraction(srcCoord.data() + 3, trgCoord.data(), srcValue.data() + 4, trgValue.data());
        // shift points
        double shift[3] = {0.5, 0.5, 0};
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                srcCoord[3 * i + j] += shift[j];
                trgCoord[3 * i + j] += shift[j];
            }
        }

        StokesSLTraction(srcCoord.data() + 3, trgCoord.data(), srcValue.data() + 4, trgValue2.data());
        for (int i = 0; i < 18; i++) {
            printf("%18.16g,%18.16g,%g,%g\n", trgValue[i], trgValue2[i], trgValue[i] - trgValue2[i],
                   (trgValue[i] - trgValue2[i]) / trgValue2[i]);
        }
    }

    MPI_Finalize();
    return 0;
}