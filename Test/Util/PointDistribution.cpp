#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "ChebNodal.hpp"
#include "PointDistribution.hpp"

void fixedPoints(int nPts, double box, double shift, std::vector<double> &srcCoord) {
    switch (nPts) {
    case 1: {
        srcCoord.push_back(0.7 * box + shift);
        srcCoord.push_back(0.6 * box + shift);
        srcCoord.push_back(0.4 * box + shift);
    } break;
    case 2: {
        srcCoord.push_back(0.7 * box + shift); // 1
        srcCoord.push_back(0.6 * box + shift);
        srcCoord.push_back(0.5 * box + shift);
        srcCoord.push_back(0.2 * box + shift); // 2
        srcCoord.push_back(0.8 * box + shift);
        srcCoord.push_back(0.7 * box + shift);
    } break;
    case 4: {                                  // quadrupole, no dipole
        srcCoord.push_back(0.1 * box + shift); // 1
        srcCoord.push_back(0.1 * box + shift);
        srcCoord.push_back(0.1 * box + shift);
        srcCoord.push_back(0.2 * box + shift); // 2
        srcCoord.push_back(0.2 * box + shift);
        srcCoord.push_back(0.2 * box + shift);
        srcCoord.push_back(0.3 * box + shift); // 3
        srcCoord.push_back(0.3 * box + shift);
        srcCoord.push_back(0.3 * box + shift);
        srcCoord.push_back(0.4 * box + shift); // 4
        srcCoord.push_back(0.4 * box + shift);
        srcCoord.push_back(0.4 * box + shift);
    } break;
    default:
        srcCoord.clear();
        break;
    }
}

void chebPoints(int nPts, double box, double shift, std::vector<double> &ptsCoord) {

    ChebNodal chebData(nPts);
    chebData.points[0] += 0;
    chebData.points.back() -= 1e-14; // prevent PVFMM crash with point located at the edge

    std::vector<double> &chebMesh = ptsCoord;
    const int dimension = chebData.points.size();
    chebMesh.resize(pow(dimension, 3) * 3);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            for (int k = 0; k < dimension; k++) {
                chebMesh[3 * (i * dimension * dimension + j * dimension + k)] =
                    (chebData.points[i] + 1) * box / 2 + shift;
                chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 1] =
                    (chebData.points[j] + 1) * box / 2 + shift;
                chebMesh[3 * (i * dimension * dimension + j * dimension + k) + 2] =
                    (chebData.points[k] + 1) * box / 2 + shift;
            }
        }
    }
}

void randomPoints(int nPts, double box, double shift, std::vector<double> &ptsCoord) {
    ptsCoord.resize(pow(nPts + 1, 3) * 3);
    randomLogNormalFill(ptsCoord, 1.0, 1.0);
    for (auto &v : ptsCoord) {
        v = fmod(v, 1.0); // put to [0,1)
        v = v * box + shift;
    }
}

void shiftAndScalePoints(std::vector<double> &ptsCoord, double shift[3], double scale) {
    // user's job to guarantee pts stays in the unit cube after shift
    const int nPts = ptsCoord.size() / 3;
    for (int i = 0; i < nPts; i++) {
        ptsCoord[3 * i] = ptsCoord[3 * i] * scale + shift[0];
        ptsCoord[3 * i + 1] = ptsCoord[3 * i + 1] * scale + shift[1];
        ptsCoord[3 * i + 2] = ptsCoord[3 * i + 2] * scale + shift[2];
    }
}

void randomUniformFill(std::vector<double> &vec, double low, double high) {
    // random fill every entry between [-1,1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(low, high);
    for (auto &v : vec) {
        v = dist(gen);
    }
}

void randomLogNormalFill(std::vector<double> &vec, double a, double b) {
    // random fill according to log normal
    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<double> dist(log(a), b);
    for (auto &v : vec) {
        v = dist(gen);
    }
}

void dumpPoints(const std::string &filename, std::vector<double> &coordLocal, std::vector<double> &valueLocal,
                const int valueDimension) {
    FILE *fp = fopen(filename.c_str(), "w");

    auto coord = coordLocal;
    auto value = valueLocal;
    collectPts(coord);
    collectPts(value);

    const int npts = coord.size() / 3;
    if (value.size() != valueDimension * npts) {
        printf("size errir in dump points, %s\n", filename.c_str());
        exit(1);
    }
    for (int i = 0; i < npts; i++) {
        fprintf(fp, "%6f %6f %6f;", coord[3 * i], coord[3 * i + 1], coord[3 * i + 2]);
        for (int j = 0; j < valueDimension; j++) {
            fprintf(fp, " %6f", value[valueDimension * i + j]);
        }
        fprintf(fp, " \n");
    }

    fclose(fp);
}

void checkError(const std::vector<double> &valueLocal, const std::vector<double> &valueTrueLocal) {
    // value and valueTrue are distributed
    // collect to rank 0 first
    std::vector<double> value = valueLocal;
    std::vector<double> valueTrue = valueTrueLocal;
    collectPts(value);
    collectPts(valueTrue);

    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    if (myRank == 0) {
        if (value.size() != valueTrue.size()) {
            printf("size errir in checking error\n");
            exit(1);
        }
        printf("checking error\n");
        // calc error and max error
        double errorL2 = 0, errorAbs = 0, L2 = 0, errorMaxL2 = 0, maxU = 0;
        double errorMaxRel = 0;

        for (int i = 0; i < valueTrue.size(); i++) {
            double temp = pow(valueTrue[i] - value[i], 2);
            errorL2 += temp;
            errorAbs += sqrt(temp);
            L2 += pow(valueTrue[i], 2);
            errorMaxL2 = std::max(sqrt(temp), errorMaxL2);
            maxU = std::max(maxU, fabs(valueTrue[i]));
            errorMaxRel = std::max(sqrt(temp) / std::abs(valueTrue[i]), errorMaxRel);
        }

        printf("Max Abs Error L2: %18.16g \n", errorMaxL2);
        printf("Max Rel Error L2: %18.16g \n", errorMaxRel);
        printf("Ave Abs Error L2: %18.16g \n", errorAbs / value.size());
        printf("RMS Error L2: %18.16g \n", sqrt(errorL2 / value.size()));
        printf("Relative Error L2: %18.16g \n", sqrt(errorL2 / L2));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void distributePts(std::vector<double> &pts, int dimension) {
    // from rank 0 to all ranks
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs == 1) {
        return;
    }
    int ptsGlobalSize;
    if (myRank == 0) {
        ptsGlobalSize = pts.size();
        MPI_Bcast(&ptsGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // std::cout << "rank " << myRank << " global size" << ptsGlobalSize << std::endl;
    } else {
        ptsGlobalSize = 0;
        MPI_Bcast(&ptsGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // std::cout << "rank " << myRank << " global size" << ptsGlobalSize << std::endl;
    }

    // bcast to all
    pts.resize(ptsGlobalSize);
    MPI_Bcast(pts.data(), ptsGlobalSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // each take a portion
    const int nPts = ptsGlobalSize / dimension;
    // inclusive low
    int indexlow = dimension * floor(myRank * nPts / static_cast<double>(nProcs));
    // non-inclusive high
    int indexhigh = dimension * floor((myRank + 1) * nPts / static_cast<double>(nProcs));
    if (myRank == nProcs - 1) {
        indexhigh = ptsGlobalSize;
    }
    std::vector<double>::const_iterator first = pts.begin() + indexlow;
    std::vector<double>::const_iterator last = pts.begin() + indexhigh;
    std::vector<double> newVec(first, last);
    pts = std::move(newVec);
}

void collectPts(std::vector<double> &pts) {
    // from all ranks to rank 0
    int myRank;
    int nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    if (nProcs == 1) {
        return;
    }
    int ptsLocalSize = pts.size();
    int ptsGlobalSize = 0;

    std::vector<int> recvSize(0);
    std::vector<int> displs(0);
    if (myRank == 0) {
        recvSize.resize(nProcs);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&ptsLocalSize, 1, MPI_INT, recvSize.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    // void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    // MPI_Comm comm)
    for (auto &p : recvSize) {
        ptsGlobalSize += p;
    }
    // std::cout << "rank " << myRank << " globalSize " << ptsGlobalSize << std::endl;
    displs.resize(recvSize.size());
    if (displs.size() > 0) {
        displs[0] = 0;
        for (int i = 1; i < displs.size(); i++) {
            displs[i] = recvSize[i - 1] + displs[i - 1];
        }
    }

    // int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    // void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
    // int root, MPI_Comm comm)
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> ptsRecv(ptsGlobalSize); // size=0 on rank !=0
    // std::cout << "globalSize " << ptsGlobalSize << std::endl;
    if (myRank == 0) {
        MPI_Gatherv(pts.data(), pts.size(), MPI_DOUBLE, ptsRecv.data(), recvSize.data(), displs.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(pts.data(), pts.size(), MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    pts = std::move(ptsRecv);
}

void collectPtsAll(std::vector<double> &pts) {
    // first collect to rank 0
    collectPts(pts);
    // broadcast to all rank
    int numGlobal = pts.size();
    MPI_Bcast(&numGlobal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    pts.resize(numGlobal);
    MPI_Bcast(pts.data(), numGlobal, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
}