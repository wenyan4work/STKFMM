/*
 * STKFMM.hpp
 *
 *  Created on: Oct 6, 2016
 *      Author: wyan
 *
 * use int as index type, easier compatibility with MPI functions.
 * for larger data, compile with ILP64 model, no code changes.
 *
 */

#ifndef STKFMM_HPP_
#define STKFMM_HPP_

#include <string>
#include <unordered_map>

#include <mpi.h>
#include <pvfmm.hpp>

/**
 * @brief namespace for stkfmm
 *
 */
namespace stkfmm {

/**
 * @brief choose the periodic boundary condition type
 *
 */
enum class PAXIS : unsigned {
    NONE = 0, ///< non-periodic, free-space
    PX = 1,   ///< periodic along x axis
    PXY = 2,  ///< periodic along XY axis
    PXYZ = 3  ///< periodic along XYZ axis
};

/**
 * @brief directly run point-to-point kernels without buildling FMM tree
 *
 */
enum class PPKERNEL : unsigned {
    SLS2T = 1, ///< Single Layer S -> T kernel
    DLS2T = 2, ///< Double Layer S -> T kernel
    L2T = 4,   ///< L -> T kernel
};

/**
 * @brief choose a kernel
 * each kernel has booth single and double layer options
 * except RPY and StokesReg kernels
 */
enum class KERNEL : unsigned {
    LAPPGrad = 1,           ///< Laplace
    Stokes = 2,             ///< Stokeslet
    RPY = 4,                ///< RPY
    StokesRegVel = 8,       ///< Regularized Stokes Velocity
    StokesRegVelOmega = 16, ///< Regularized Stokes Velocity/Rotation
    PVel = 32,              ///< Stokes
    PVelGrad = 64,          ///< Stokes
    PVelLaplacian = 128,    ///< Stokes
    Traction = 256,         ///< Stokes
};

extern const std::unordered_map<KERNEL, const pvfmm::Kernel<double> *>
    kernelMap;

// /**
//  * @brief return an integer (size_t) from a enum class member
//  *
//  */
// struct EnumClassHash {
//     template <typename T>
//     size_t operator()(T t) const {
//         return static_cast<size_t>(t);
//     }
// };

/**
 * @brief Enum to integer
 *
 * @tparam Enumeration
 * @param value
 * @return std::underlying_type<Enumeration>::type
 */
template <typename Enumeration>
auto asInteger(Enumeration const value) ->
    typename std::underlying_type<Enumeration>::type {
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

namespace impl {

/**
 * @brief Run FMM for a chosen kernel
 * (1) accept only coordinates within [0,1) box
 * (2) handles periodicity
 * Remark: this class is not supposed to be used by the user of this library
 */
class FMMData {
  public:
    const KERNEL kernelChoice; ///< chosen kernel
    const PAXIS periodicity;   ///< chosen periodicity

    int kdimSL;  ///< Single Layer kernel dimension
    int kdimDL;  ///< Double Layer kernel dimension
    int kdimTrg; ///< Target kernel dimension

    int multOrder; ///< multipole order
    int maxPts;    ///< max number of points per octant

    const pvfmm::Kernel<double>
        *kernelFunctionPtr; ///< pointer to kernel function

    std::vector<double> equivCoord; ///< periodicity L2T equivalent point coord
    std::vector<double> M2Ldata;    ///< periodicity M2L operator data

    FMMData() = delete; ///< forbid default constructor

    // forbid copy constructor
    FMMData(const FMMData &) = delete;
    FMMData &operator=(const FMMData &) = delete;
    FMMData(FMMData &&) = delete;
    FMMData &operator=(FMMData &&) = delete;

    /**
     * @brief constructor
     *
     */
    FMMData(KERNEL kernelChoice_, PAXIS periodicity_, int multOrder_,
            int maxPts_);

    /**
     * @brief Destroy the FMMData object
     *
     */
    ~FMMData();

    static const pvfmm::Kernel<double> *getKernelFunction(KERNEL kernelChoice_);

    /**
     * @brief Set kernel function in pvfmm data structure
     *
     */
    void setKernel();

    // computation routines

    /**
     * @brief setup tree
     *
     * @param srcSLCoord single layer source coordinate
     * @param srcDLCoord double layer source coordinate
     * @param trgCoord target coordinate
     */
    void setupTree(const std::vector<double> &srcSLCoord,
                   const std::vector<double> &srcDLCoord,
                   const std::vector<double> &trgCoord);

    /**
     * @brief run FMM
     *
     * @param srcSLValue [in] single layer source value
     * @param srcDLValue [in] double layer source value
     * @param trgValue [out] target value
     */
    void evaluateFMM(std::vector<double> &srcSLValue,
                     std::vector<double> &srcDLValue,
                     std::vector<double> &trgValue);

    /**
     * @brief periodize the target values
     *
     *
     * @param trgValue
     */
    void periodizeFMM(std::vector<double> &trgValue);

    /**
     * @brief directly evaluate kernel functions without FMM tree
     *
     * @param nThreads number of threads to use
     * @param chooseSD choose which kernel function to use
     * @param nSrc source number of points
     * @param srcCoordPtr source coordinate
     * @param srcValuePtr source value
     * @param nTrg target number of points
     * @param trgCoordPtr target coordinate
     * @param trgValuePtr target value
     */
    void evaluateKernel(int nThreads, PPKERNEL chooseSD, //
                        const int nSrc, double *srcCoordPtr,
                        double *srcValuePtr, //
                        const int nTrg, double *trgCoordPtr,
                        double *trgValuePtr);

    /**
     * @brief delete the fmm tree
     *
     */
    void deleteTree();

    /**
     * @brief clear the FMM data
     *
     */
    void clear();

  private:
    pvfmm::PtFMM<double> *matrixPtr;        ///< pvfmm PtFMM pointer
    pvfmm::PtFMM_Tree<double> *treePtr;     ///< pvfmm PtFMM_Tree pointer
    pvfmm::PtFMM_Data<double> *treeDataPtr; ///< pvfmm PtFMM_Data pointer
    MPI_Comm comm;                          ///< MPI_comm communicator

    // work out the Stokes PVel kernel M2L data
    /**
     * @brief read the M2L Matrix from file
     *
     */
    void readM2LMat(const int kDim, const std::string &dataName,
                    std::vector<double> &data);

    /**
     * @brief setup this->M2Ldata
     *
     */
    void setupM2Ldata();
};

} // namespace impl

/**
 * @brief a virtual interface for STKFMM cases
 *
 */
class STKFMM_INTERFACE {};

/**
 * @brief STKFMM class, exposed to user
 *
 */
class STKFMM : public STKFMM_INTERFACE {
  public:
    /**
     * @brief Construct a new STKFMM object
     *
     * @param multOrder FMM multipole order
     * @param maxPts max number of points per octant
     * @param pbc_ periodic boundary condition
     * @param kernelComb_ combination of kernels to use
     */
    STKFMM(int multOrder = 10, int maxPts = 2000, PAXIS pbc_ = PAXIS::NONE,
           unsigned int kernelComb_ = 1);

    /**
     * @brief Destroy the STKFMM object
     *
     */
    ~STKFMM();

    /**
     * @brief Set point coordinates
     * coordinates are read from the pointers with (3nSL,3nDL,3nTrg) contiguous
     * double numbers
     *
     * @param nSL single layer source point number
     * @param srcSLCoordPtr single layer source point coordinate
     * @param nDL double layer source point number
     * @param srcDLCoordPtr double layer source point coordinate
     * @param nTrg target point number
     * @param trgCoordPtr target point coordinate
     */
    void setPoints(const int nSL, const double *srcSLCoordPtr, const int nDL,
                   const double *srcDLCoordPtr, const int nTrg,
                   const double *trgCoordPtr);

    /**
     * @brief evaluate FMM
     * results are added to values already in trgValuePtr
     * setPoints() and setupTree() must be called first
     * nSL, nDL, nTrg must be the same as used by setPoints()
     * length of arrays must match (kdimSL,kdimDL,kdimTrg) in the chosen
     * kernel
     * @param nSL single layer source point number
     * @param srcSLValuePtr pointer to single layer source value
     * @param nDL double layer source point number
     * @param srcDLValuePtr pointer to double layer source value
     * @param nTrg target point number
     * @param trgValuePtr pointer to target value
     * @param kernel one of the activated kernels to evaluate
     */
    void evaluateFMM(const int nSL, const double *srcSLValuePtr, const int nDL,
                     const double *srcDLValuePtr, const int nTrg,
                     double *trgValuePtr, const KERNEL kernel);

    /**
     * @brief evaluate kernel functions by direct O(N^2) summation without FMM
     * results are added to values already in trgValuePtr
     * setPoints() does not have to be called
     * length of arrays must match (kdimSL,kdimDL,kdimTrg) in the chosen kernel
     * @param nThreads number of threads
     * @param p2p choose which sub-kernel in the kernel to evaluate
     * @param nSrc number of source point
     * @param srcCoordPtr pointer to source point coordinate
     * @param srcValuePtr pointer to source point value
     * @param nTrg number of target point
     * @param trgCoordPtr pointer to target point coordinate
     * @param trgValuePtr pointer to target point value
     * @param kernel one of the activated kernels to evaluate
     */
    void evaluateKernel(const int nThreads, const PPKERNEL p2p, const int nSrc,
                        double *srcCoordPtr, double *srcValuePtr,
                        const int nTrg, double *trgCoordPtr,
                        double *trgValuePtr, const KERNEL kernel);

    /**
     * @brief clear the data and prepare for another FMM evaluation
     *
     * @param kernelChoice
     */
    void clearFMM(KERNEL kernelChoice);

    /**
     * @brief setup the tree for the chosen kernel
     * setPoints() must have been called
     *
     * @param kernel_ one of the activated kernels to use
     */
    void setupTree(KERNEL kernel_);

    /**
     * @brief Set the FMM box
     * a cubic box as [xlow,xhigh)x[ylow,yhigh)x[zlow,zhigh)
     *
     * @param xlow_
     * @param xhigh_
     * @param ylow_
     * @param yhigh_
     * @param zlow_
     * @param zhigh_
     */
    void setBox(double xlow_, double xhigh_, double ylow_, double yhigh_,
                double zlow_, double zhigh_);

    /**
     * @brief Get the FMM box
     * a cubic box as [xlow,xhigh)x[ylow,yhigh)x[zlow,zhigh)
     *
     * @return [xlow, xhigh, ylow, yhigh, zlow, zhigh]
     */
    std::tuple<double, double, double, double, double, double> getBox() {
        return std::tuple<double, double, double, double, double, double>(
            xlow, xhigh, ylow, yhigh, zlow, zhigh);
    };

    /**
     * @brief show activated kernels
     *
     */
    void showActiveKernels();

    /**
     * @brief show if a kernel is activated
     *
     * @param kernel_
     * @return true
     * @return false
     */
    bool isKernelActive(KERNEL kernel_) {
        return asInteger(kernel_) & kernelComb;
    }

    /**
     * @brief Get kernel dimension
     *
     * @param kernel_ one of the activated kernels
     * @return [single layer kernel dimension, double layer kernel dimension,
     *          target kernel dimension]
     */
    static std::tuple<int, int, int> getKernelDimension(KERNEL kernel_);

    /**
     * @brief Get multipole order
     *
     * @return multipole order
     */
    int getMultOrder() { return multOrder; }

  private:
    const int multOrder;       ///< multipole order
    const int maxPts;          ///< max number of points to use
    PAXIS pbc;                 ///< periodic boundary condition
    const unsigned kernelComb; ///< combination of activated kernels

    double xlow, xhigh;            ///< box x
    double ylow, yhigh;            ///< box y
    double zlow, zhigh;            ///< box z
    double scaleFactor;            ///< scale factor to fit in box of [0,1)^3
    double xshift, yshift, zshift; ///< shift to fit in box of [0,1)^3

    MPI_Comm comm; ///< MPI_Comm object

    std::vector<double> srcSLCoordInternal; ///< scaled Single Layer coordinate
    std::vector<double> srcDLCoordInternal; ///< scaled Double Layer coordinate
    std::vector<double> trgCoordInternal;   ///< scaled target coordinate
    std::vector<double> srcSLValueInternal; ///< scaled SL value
    std::vector<double> srcDLValueInternal; ///< scaled DL value
    std::vector<double> trgValueInternal;   ///< scaled trg value

    /**
     * @brief setup the internal coord, with proper scaling and BC
     *
     * @param npts number of points
     * @param coordInPtr pointer to unscaled coordinate
     * @param coord scaled coordinate
     */
    void setupCoord(const int npts, const double *coordInPtr,
                    std::vector<double> &coord) const;

    std::unordered_map<KERNEL, impl::FMMData *> poolFMM; ///< bookkeeping of all FMMData object
};
} // namespace stkfmm

#endif
