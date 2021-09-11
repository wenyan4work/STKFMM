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

#include "STKFMM_common.hpp"
#include "STKFMM_impl.hpp"

/**
 * @brief namespace for stkfmm
 *
 */
namespace stkfmm {
/**
 * @brief get STKFMM_VERBOSE environment variable
 */
bool get_verbosity();

/**
 * @brief a virtual interface for STKFMM cases
 *
 */
class STKFMM {
  public:
    /**
     * @brief Construct a new STKFMM object
     *
     * @param multOrder_
     * @param maxPts_
     * @param pbc_
     * @param kernelComb_
     */
    STKFMM(int multOrder_, int maxPts_, PAXIS pbc_, unsigned int kernelComb_, bool enableFF_ = true);

    /**
     * @brief Set FMM Box
     * a cubic or half-cubic box
     *
     * @param origin_
     * @param len_
     */
    void setBox(double origin_[3], double len_);

    /**
     * @brief Set point coordinates
     * coordinates are read from the pointers with (3nSL,3nDL,3nTrg) contiguous double numbers
     *
     * @param nSL single layer source point number
     * @param srcSLCoordPtr single layer source point coordinate
     * @param nTrg target point number
     * @param trgCoordPtr target point coordinate
     * @param nDL double layer source point number
     * @param srcDLCoordPtr double layer source point coordinate
     */
    virtual void setPoints(const int nSL, const double *srcSLCoordPtr, const int nTrg, const double *trgCoordPtr,
                           const int nDL = 0, const double *srcDLCoordPtr = nullptr) = 0;

    /**
     * @brief setup the tree for the chosen kernel
     * setPoints() must have been called
     *
     * @param kernel one of the activated kernels to use
     */
    virtual void setupTree(KERNEL kernel) = 0;

    /**
     * @brief evaluate FMM
     * results are added to values already in trgValuePtr
     * setPoints() and setupTree() must be called first
     * nSL, nDL, nTrg must be the same as used by setPoints()
     * length of arrays must match (kdimSL,kdimDL,kdimTrg) in the chosen kernel
     *
     * @param kernel one of the activated kernels to evaluate
     * @param nSL single layer source point number
     * @param srcSLValuePtr pointer to single layer source value
     * @param nDL double layer source point number
     * @param srcDLValuePtr pointer to double layer source value
     * @param nTrg target point number
     * @param trgValuePtr pointer to target value
     */
    virtual void evaluateFMM(const KERNEL kernel, const int nSL, const double *srcSLValuePtr, const int nTrg,
                             double *trgValuePtr, const int nDL = 0, const double *srcDLValuePtr = nullptr) = 0;

    /**
     * @brief evaluate kernel functions by direct O(N^2) summation without FMM
     * results are added to values already in trgValuePtr
     * setPoints() does not have to be called
     * length of arrays must match (kdimSL,kdimDL,kdimTrg) in the chosen kernel
     * @param kernel one of the activated kernels to evaluate
     * @param nThreads number of threads
     * @param p2p choose which sub-kernel in the kernel to evaluate
     * @param nSrc number of source point
     * @param srcCoordPtr pointer to source point coordinate
     * @param srcValuePtr pointer to source point value
     * @param nTrg number of target point
     * @param trgCoordPtr pointer to target point coordinate
     * @param trgValuePtr pointer to target point value
     */
    void evaluateKernel(const KERNEL kernel, const int nThreads, const PPKERNEL p2p, const int nSrc,
                        double *srcCoordPtr, double *srcValuePtr, const int nTrg, double *trgCoordPtr,
                        double *trgValuePtr);

    /**
     * @brief clear the data and prepare for another FMM evaluation
     *
     * @param kernel
     */
    virtual void clearFMM(KERNEL kernel) = 0;

    /**
     * @brief Get the FMM box
     *
     * @return [xlow, xhigh, ylow, yhigh, zlow, zhigh]
     */
    virtual std::tuple<double, double, double, double, double, double> getBox() const = 0;

    /**
     * @brief show activated kernels
     *
     */
    void showActiveKernels() const;

    /**
     * @brief show if a kernel is activated
     *
     * @param kernel_
     * @return true
     * @return false
     */
    bool isKernelActive(KERNEL kernel_) const { return asInteger(kernel_) & kernelComb; }

    /**
     * @brief Get multipole order
     *
     * @return multipole order
     */
    int getMultOrder() const { return multOrder; }

  protected:
    int rank;                  ///< MPI rank
    const int multOrder;       ///< multipole order
    const int maxPts;          ///< max number of points to use
    PAXIS pbc;                 ///< periodic boundary condition
    const unsigned kernelComb; ///< combination of activated kernels

    double origin[3];   ///< coordinate of box origin
    double len;         ///< cubic box size
    double scaleFactor; ///< scale factor to fit in box of [0,1)^3

    std::vector<double> srcSLCoordInternal; ///< scaled Single Layer coordinate
    std::vector<double> srcDLCoordInternal; ///< scaled Double Layer coordinate
    std::vector<double> trgCoordInternal;   ///< scaled target coordinate
    std::vector<double> srcSLValueInternal; ///< scaled SL value
    std::vector<double> srcDLValueInternal; ///< scaled DL value
    std::vector<double> trgValueInternal;   ///< scaled trg value

    std::unordered_map<KERNEL, impl::FMMData *> poolFMM; ///< all FMMData objects

    /**
     * @brief scale and shift coordPtr
     *
     * @param npts number of points
     * @param coordPtr pointer to unscaled coordinate
     */
    void scaleCoord(const int npts, double *coordPtr) const;

    /**
     * @brief handle pbc [0,1) of coordPtr
     *
     * @param npts
     * @param coordPtr
     */
    void wrapCoord(const int npts, double *coordPtr) const;
};

/**
 * @brief FMM in 3D space
 *
 */
class Stk3DFMM : public STKFMM {
  public:
    /**
     * @brief Construct a new Stk3DFMM object
     *
     * @param multOrder
     * @param maxPts
     * @param pbc_
     * @param kernelComb_
     */
    Stk3DFMM(int multOrder = 10, int maxPts = 2000, PAXIS pbc_ = PAXIS::NONE,
             unsigned int kernelComb_ = asInteger(KERNEL::Stokes) | asInteger(KERNEL::RPY), bool enableFF_ = true);

    virtual void setPoints(const int nSL, const double *srcSLCoordPtr, const int nTrg, const double *trgCoordPtr,
                           const int nDL = 0, const double *srcDLCoordPtr = nullptr);

    virtual void setupTree(KERNEL kernel);

    virtual void evaluateFMM(const KERNEL kernel, const int nSL, const double *srcSLValuePtr, const int nTrg,
                             double *trgValuePtr, const int nDL = 0, const double *srcDLValuePtr = nullptr);

    virtual void clearFMM(KERNEL kernel);

    virtual std::tuple<double, double, double, double, double, double> getBox() const {
        return std::make_tuple(origin[0], origin[0] + len, origin[1], origin[1] + len, origin[2], origin[2] + len);
    };

    ~Stk3DFMM();
};

/**
 * @brief FMM in 3D space above a no-slip wall. Supports only Stokeslet and RPY kernels
 *
 */
class StkWallFMM : public STKFMM {
  public:
    /**
     * @brief Construct a new StkWallFMM object
     *
     * @param multOrder
     * @param maxPts
     * @param pbc_
     * @param kernelComb_
     */
    StkWallFMM(int multOrder = 10, int maxPts = 2000, PAXIS pbc_ = PAXIS::NONE,
               unsigned int kernelComb_ = asInteger(KERNEL::Stokes) | asInteger(KERNEL::RPY), bool enableFF_ = true);

    virtual void setPoints(const int nSL, const double *srcSLCoordPtr, const int nTrg, const double *trgCoordPtr,
                           const int nDL = 0, const double *srcDLCoordPtr = nullptr);

    virtual void setupTree(KERNEL kernel);

    virtual void evaluateFMM(const KERNEL kernel, const int nSL, const double *srcSLValuePtr, const int nTrg,
                             double *trgValuePtr, const int nDL = 0, const double *srcDLValuePtr = nullptr);

    virtual void clearFMM(KERNEL kernel);

    virtual std::tuple<double, double, double, double, double, double> getBox() const {
        return std::make_tuple(origin[0], origin[0] + len, origin[1], origin[1] + len, origin[2], origin[2] + len);
    };

    ~StkWallFMM();

  protected:
    std::vector<double> srcSLImageCoordInternal;  ///< scaled SL point coord
    std::vector<double> srcSLOriginCoordInternal; ///< scaled SL image point coord

    /**
     * @brief evaluate Stokes image system
     *
     */
    void evalStokes();

    /**
     * @brief evaluate RPY image system
     *
     */
    void evalRPY();
};

} // namespace stkfmm

#endif
