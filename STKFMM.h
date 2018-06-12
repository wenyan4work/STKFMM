/*
 * FMMWrapper.h
 *
 *  Created on: Oct 6, 2016
 *      Author: wyan
 * 
 * use int as index type, easier compatibility with MPI functions.
 * for larger data, compile with ILP64 model, no code changes.
 * 
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

enum class PAXIS : unsigned {
    NONE,
    // TODO: add periodic BC
    PX,
    PXY,
    PXYZ
};

enum class KERNEL : unsigned {
    PVel = 1, // single layer kernel
    PVelGrad = 2,
    PVelLaplacian = 4,
    Traction = 8,
    LAPPGrad = 16, // laplace single layer
};

enum class PPKERNEL : unsigned {
    SLS2T = 1,
    DLS2T = 2,
    L2T = 4,
};

struct EnumClassHash {
    template <typename T>
    size_t operator()(T t) const {
        return static_cast<size_t>(t);
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
    int maxPts;

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

};

class STKFMM {
  public:
    template <typename Enumeration>
    auto asInteger(Enumeration const value) -> typename std::underlying_type<Enumeration>::type {
        return static_cast<typename std::underlying_type<Enumeration>::type>(value);
    }

    STKFMM(int multOrder = 10, int maxPts = 2000, PAXIS pbc_ = PAXIS::NONE, unsigned int kernelComb_ = 1);

    ~STKFMM();

    void setPoints(const int nSL, const double *srcSLCoordPtr, const int nDL, const double *srcDLCoordPtr,
                   const int nTrg, const double *trgCoordPtr);

    // results are added to values already in trgValuePtr
    void evaluateFMM(const int nSL, const double *srcSLValuePtr, const int nDL, const double *srcDLValuePtr,
                     const int nTrg, double *trgValuePtr, const KERNEL kernel);

    // results are added to values already in trgValuePtr.
    void evaluateKernel(const int nThreads, const PPKERNEL p2p, const int nSrc, double *srcCoordPtr,
                        double *srcValuePtr, const int nTrg, double *trgCoordPtr, double *trgValuePtr,
                        const KERNEL kernel);

    void clearFMM(KERNEL kernelChoice);

    void setupTree(KERNEL kernel_);

    void setBox(double xlow_, double xhigh_, double ylow_, double yhigh_, double zlow_, double zhigh_);

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
    const unsigned kernelComb;

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

    void setupCoord(const int npts, const double *coordInPtr,
                    std::vector<double> &coord); // setup the internal srcCoord and
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
