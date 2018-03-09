/*
 * FMMWrapper.h
 *
 *  Created on: Oct 6, 2016
 *      Author: wyan
 */

#ifndef INCLUDE_STKFMM_H
#define INCLUDE_STKFMM_H

#include <string>
#include <unordered_map>

// a wrapper for pvfmm
// choose kernel at compile time
#include <mpi.h>
#include <pvfmm.hpp>

namespace stkfmm {

enum class PAXIS : size_t {
    NONE,
    // TODO: add periodic BC
    PZ,
    PXY,
    PXYZ
};

enum class KERNEL : size_t {
    PVel = 1, // single layer kernel
    PVelGrad = 2,
    PVelLaplacian = 4,
    Traction = 8,
    LAPPGrad = 16, // laplace single layer
};

enum class PPKERNEL : size_t {
    SLS2T = 1,
    DLS2T = 2,
    L2T = 4,
};

struct EnumClassHash {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

class FMMData {
    // computes FMM with assigned kernel
    // all source and target points must be in [0,1) box
    // handles periodicity
  public:
    const KERNEL kernelChoice;
    const PAXIS periodicity;

    int kdimSL;
    int kdimDL;
    int kdimTrg;

    int multOrder;
    size_t maxPts;

    const pvfmm::Kernel<double> *kernelFunctionPtr;

    std::vector<double> equivCoord;
    std::vector<double> M2Ldata;

    // forbid default constructor
    FMMData() = delete;
    // copy constructor
    FMMData(const FMMData &) = delete;
    FMMData &operator=(const FMMData &) = delete;
    FMMData(FMMData &&) = delete;
    FMMData &operator=(FMMData &&) = delete;

    // constructor with choice of kernel
    FMMData(KERNEL kernelChoice_, PAXIS periodicity_, int multOrder_, int maxPts_);

    // destructor
    ~FMMData();

    // helper
    // void setKernel(const pvfmm::Kernel<double> &kernelFunction);
    void setKernel();

    // computation routines
    void setupTree(const std::vector<double> &srcSLCoord, const std::vector<double> &srcDLCoord,
                   const std::vector<double> &trgCoord);
    void evaluateFMM(std::vector<double> &srcSLValue, std::vector<double> &srcDLValue, std::vector<double> &trgValue);

    void periodizeFMM(std::vector<double> &trgValue);

    void evaluateKernel(int nThreads, PPKERNEL chooseSD, const int nSrc, double *srcCoordPtr, double *srcValuePtr,
                        const int nTrg, double *trgCoordPtr, double *trgValuePtr);

    void deleteTree();
    void clear();

  private:
    pvfmm::PtFMM *matrixPtr;
    pvfmm::PtFMM_Tree *treePtr;
    pvfmm::PtFMM_Data *treeDataPtr;
    MPI_Comm comm;

    void readM2LMat(const std::string dataName);

    template <class Real_t>
    std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
        size_t n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

        std::vector<Real_t> coord(n_ * 3);
        coord[0] = coord[1] = coord[2] = -1.0;
        size_t cnt = 1;
        for (int i = 0; i < p - 1; i++)
            for (int j = 0; j < p - 1; j++) {
                coord[cnt * 3] = -1.0;
                coord[cnt * 3 + 1] = (2.0 * (i + 1) - p + 1) / (p - 1);
                coord[cnt * 3 + 2] = (2.0 * j - p + 1) / (p - 1);
                cnt++;
            }
        for (int i = 0; i < p - 1; i++)
            for (int j = 0; j < p - 1; j++) {
                coord[cnt * 3] = (2.0 * i - p + 1) / (p - 1);
                coord[cnt * 3 + 1] = -1.0;
                coord[cnt * 3 + 2] = (2.0 * (j + 1) - p + 1) / (p - 1);
                cnt++;
            }
        for (int i = 0; i < p - 1; i++)
            for (int j = 0; j < p - 1; j++) {
                coord[cnt * 3] = (2.0 * (i + 1) - p + 1) / (p - 1);
                coord[cnt * 3 + 1] = (2.0 * j - p + 1) / (p - 1);
                coord[cnt * 3 + 2] = -1.0;
                cnt++;
            }
        for (size_t i = 0; i < (n_ / 2) * 3; i++)
            coord[cnt * 3 + i] = -coord[i];

        Real_t r = 0.5 * pow(0.5, depth);
        Real_t b = alpha * r;
        for (size_t i = 0; i < n_; i++) {
            coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
            coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
            coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
        }
        return coord;
    }
};

class STKFMM {
  public:
    template <typename Enumeration>
    auto asInteger(Enumeration const value) -> typename std::underlying_type<Enumeration>::type {
        return static_cast<typename std::underlying_type<Enumeration>::type>(value);
    }

    STKFMM(int multOrder = 10, int maxPts = 1000, PAXIS pbc_ = PAXIS::NONE, unsigned int kernelComb_ = 1);

    ~STKFMM();

    // results already in trgValue are cleaned
    void evaluateFMM(std::vector<double> &srcSLValue, std::vector<double> &srcDLValue, std::vector<double> &trgValue,
                     KERNEL kernelChoice);

    // results are added to values already in trgValuePtr.
    void evaluateKernel(const int nThreads, const PPKERNEL p2p, const int nSrc, double *srcCoordPtr,
                        double *srcValuePtr, const int nTrg, double *trgCoordPtr, double *trgValuePtr,
                        const KERNEL kernel);

    void clearFMM(KERNEL kernelChoice);

    void setPoints(const std::vector<double> &srcSLCoord_, const std::vector<double> &srcDLCoord_,
                   const std::vector<double> &trgCoord_);

    void setupTree(KERNEL kernel_);

    void setBox(double, double, double, double, double, double);

    void showActiveKernels();

    bool isKernelActive(KERNEL kernel_) { return asInteger(kernel_) & kernelComb; }

    void getKernelDimension(int &kdimSL_, int &kdimDL_, int &kdimTrg_, KERNEL kernel_) {
        kdimSL_ = poolFMM[kernel_]->kdimSL;
        kdimDL_ = poolFMM[kernel_]->kdimDL;
        kdimTrg_ = poolFMM[kernel_]->kdimTrg;
    }

  private:
    const int multOrder;
    const int maxPts;
    PAXIS pbc;
    const unsigned int kernelComb;

    double xlow, xhigh; // box
    double ylow, yhigh;
    double zlow, zhigh;
    double scaleFactor;
    double xshift, yshift, zshift;

    MPI_Comm comm;

    std::vector<double> srcSLCoordInternal; // scaled coordinate Single Layer
    std::vector<double> srcDLCoordInternal; // scaled coordinate Double Layer
    std::vector<double> trgCoordInternal;
    std::vector<double> srcSLValueInternal; // scaled SL value
    std::vector<double> srcDLValueInternal; // scaled SL value
    std::vector<double> trgValueInternal;   // scaled trg value

    void setupCoord(const std::vector<double> &, std::vector<double> &); // setup the internal srcCoord and
                                                                         // trgCoord, with proper rotation and BC

    std::unordered_map<KERNEL, FMMData *, EnumClassHash> poolFMM;
    // return fraction part between [0,1)
    /*
     * This function is only applied in the PERIODIC DIRECTION
     * The user of the library must ensure that all points are located within [0,1)
     * */
    inline double fracwrap(double x) { return x - floor(x); }
};
} // namespace stkfmm

#endif /* INCLUDE_FMMWRAPPER_H_ */
