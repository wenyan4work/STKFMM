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

void PointDistribution::fixedPoints(int nPts, double box, double shift, std::vector<double> &srcCoord) {
    switch (nPts) {
    case 1: {
        srcCoord.push_back(0.3 * box + shift);
        srcCoord.push_back(0.2 * box + shift);
        srcCoord.push_back(0.1 * box + shift);
    } break;
    case 2: {
        srcCoord.push_back(0.3 * box + shift); // 1
        srcCoord.push_back(0.2 * box + shift);
        srcCoord.push_back(0.1 * box + shift);
        srcCoord.push_back(0.2 * box + shift); // 2
        srcCoord.push_back(0.4 * box + shift);
        srcCoord.push_back(0.3 * box + shift);
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

void PointDistribution::meshPoints(int dim, int nPts, double box, double shift, std::vector<double> &ptsCoord,
                                   bool cheb) {
    std::vector<double> pts; // nPts+1 points on [-1,1]
    if (cheb) {
        ChebNodal chebData(nPts, true);
        pts = chebData.points;
    } else {
        pts.resize(nPts + 1);
        for (int i = 0; i < nPts + 1; i++) {
            pts[i] = -1 + i * 2.0 / nPts;
        }
    }
    // prevent PVFMM crash when point located at the edge
    pts.back() -= 1e-12;

    const int dimension = pts.size();
    const int n = pow(dimension, dim);
    ptsCoord.resize(n * 3);
    if (dim == 3) {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                for (int k = 0; k < dimension; k++) {
                    const int index = 3 * (i * dimension * dimension + j * dimension + k);
                    ptsCoord[index] = (pts[i] + 1) * box / 2 + shift;
                    ptsCoord[index + 1] = (pts[j] + 1) * box / 2 + shift;
                    ptsCoord[index + 2] = (pts[k] + 1) * box / 2 + shift;
                }
            }
        }
    }
    if (dim == 2) {
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                const int index = 3 * (i * dimension + j);
                ptsCoord[index] = (pts[i] + 1) * box / 2 + shift;
                ptsCoord[index + 1] = (pts[j] + 1) * box / 2 + shift;
                ptsCoord[index + 2] = shift;
            }
        }
    }
    if (dim == 1) {
        for (int i = 0; i < dimension; i++) {
            const int index = 3 * i;
            ptsCoord[index] = (pts[i] + 1) * box / 2 + shift;
            ptsCoord[index + 1] = shift;
            ptsCoord[index + 2] = shift;
        }
    }
}

void PointDistribution::randomPoints(int dim, int nPts, double box, double shift, std::vector<double> &ptsCoord) {
    const int n = pow(nPts + 1, dim);
    ptsCoord.resize(n * 3);
    randomLogNormalFill(ptsCoord, 1.0, 1.0);
    for (auto &v : ptsCoord) {
        v = fmod(v, 1.0); // put to [0,1)
        v = v * box + shift;
    }
    for (int i = 0; i < n; i++) {
        if (dim < 3)
            ptsCoord[3 * i + 2] = shift;
        if (dim < 2)
            ptsCoord[3 * i + 1] = shift;
    }
}

void PointDistribution::shiftAndScalePoints(std::vector<double> &ptsCoord, double shift[3], double scale) {
    // user's job to guarantee pts stays in the unit cube after shift
    const int nPts = ptsCoord.size() / 3;
    for (int i = 0; i < nPts; i++) {
        ptsCoord[3 * i] = ptsCoord[3 * i] * scale + shift[0];
        ptsCoord[3 * i + 1] = ptsCoord[3 * i + 1] * scale + shift[1];
        ptsCoord[3 * i + 2] = ptsCoord[3 * i + 2] * scale + shift[2];
    }
}

void PointDistribution::randomUniformFill(std::vector<double> &vec, double low, double high) {
    // random fill every entry between [-1,1)
    std::uniform_real_distribution<double> dist(low, high);
    for (auto &v : vec) {
        v = dist(gen_);
    }
}

void PointDistribution::randomLogNormalFill(std::vector<double> &vec, double a, double b) {
    // random fill according to log normal
    std::lognormal_distribution<double> dist(log(a), b);
    for (auto &v : vec) {
        v = dist(gen_);
    }
}

void PointDistribution::dumpPoints(const std::string &filename, std::vector<double> &coordLocal,
                                   std::vector<double> &valueLocal, const int valueDimension) {
    FILE *fp = fopen(filename.c_str(), "w");

    auto coord = coordLocal;
    auto value = valueLocal;
    collectPts(coord);
    collectPts(value);

    const int npts = coord.size() / 3;
    if (value.size() != valueDimension * npts) {
        printf("size error in dump points, %s\n", filename.c_str());
        exit(1);
    }
    for (int i = 0; i < npts; i++) {
        fprintf(fp, "%.10e, %.10e, %.10e", coord[3 * i], coord[3 * i + 1], coord[3 * i + 2]);
        for (int j = 0; j < valueDimension; j++) {
            fprintf(fp, ", %.10e", value[valueDimension * i + j]);
        }
        fprintf(fp, " \n");
    }

    fclose(fp);
}


void PointDistribution::distributePts(std::vector<double> &pts, int dimension) {
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
        // std::cout << "rank " << myRank << " global size" << ptsGlobalSize <<
        // std::endl;
    } else {
        ptsGlobalSize = 0;
        MPI_Bcast(&ptsGlobalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // std::cout << "rank " << myRank << " global size" << ptsGlobalSize <<
        // std::endl;
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

void PointDistribution::collectPts(std::vector<double> &pts) {
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
    // std::cout << "rank " << myRank << " globalSize " << ptsGlobalSize <<
    // std::endl;
    displs.resize(recvSize.size());
    if (displs.size() > 0) {
        displs[0] = 0;
        for (int i = 1; i < displs.size(); i++) {
            displs[i] = recvSize[i - 1] + displs[i - 1];
        }
    }

    // int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype
    // sendtype, void *recvbuf, const int recvcounts[], const int displs[],
    // MPI_Datatype recvtype, int root, MPI_Comm comm)
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

void PointDistribution::collectPtsAll(std::vector<double> &pts) {
    // first collect to rank 0
    collectPts(pts);
    // broadcast to all rank
    int numGlobal = pts.size();
    MPI_Bcast(&numGlobal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    pts.resize(numGlobal);
    MPI_Bcast(pts.data(), numGlobal, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
}
