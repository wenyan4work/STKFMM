#include <cstdio>
#include <cstdlib>
#include <vector>

#include "STKFMM/STKFMM.hpp"

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

    auto kernel = stkfmm::KERNEL::Traction;
    unsigned int kernelComb = stkfmm::asInteger(kernel);
    {
        auto fmm = stkfmm::Stk3DFMM(16, 2000, stkfmm::PAXIS::PXY, kernelComb, false);
        fmm.showActiveKernels();
        fmm.setBox(origin, box);
        // first evaluation
        {
            fmm.clearFMM(kernel);
            fmm.setPoints(2, srcCoord.data(), 2, trgCoord.data());
            fmm.setupTree(kernel);
            fmm.evaluateFMM(kernel, 2, srcValue.data(), 2, trgValue.data());
        }

        // shift points
        double shift[3] = {0.5, 0.5, 0};
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                srcCoord[3 * i + j] += shift[j];
                trgCoord[3 * i + j] += shift[j];
            }
        }

        // second evaluation
        {
            fmm.clearFMM(kernel);
            fmm.setPoints(2, srcCoord.data(), 2, trgCoord.data());
            fmm.setupTree(kernel);
            fmm.evaluateFMM(kernel, 2, srcValue.data(), 2, trgValue2.data());
        }

        for (int i = 0; i < 18; i++) {
            printf("%18.16g,%18.16g,%g,%g\n", trgValue[i], trgValue2[i], trgValue[i] - trgValue2[i],
                   (trgValue[i] - trgValue2[i]) / trgValue2[i]);
        }
    }

    MPI_Finalize();
    return 0;
}