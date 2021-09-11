#ifndef STKFMM_IMPL_
#define STKFMM_IMPL_

#include "STKFMM_common.hpp"

#include <string>
#include <unordered_map>

namespace stkfmm {

namespace impl {

/**
 * @brief Run FMM for a chosen kernel
 * (1) accept only coordinates within [0,1) box
 * (2) handles periodicity
 * Remark: this class is not supposed to be used by the user of this library
 */
class FMMData {
  public:
    const stkfmm::KERNEL kernelChoice; ///< chosen kernel
    const stkfmm::PAXIS periodicity;   ///< chosen periodicity
    bool enableFF;                     ///< enable periodic Far-Field fix

    int kdimSL;  ///< Single Layer kernel dimension
    int kdimDL;  ///< Double Layer kernel dimension
    int kdimTrg; ///< Target kernel dimension

    int multOrder; ///< multipole order
    int maxPts;    ///< max number of points per octant

    const pvfmm::Kernel<double> *kernelFunctionPtr; ///< pointer to kernel function

    std::vector<double> equivCoord; ///< periodicity L2T equivalent point coord
    std::vector<double> M2Ldata;    ///< periodicity M2L operator data
    std::vector<double> M2Cdata;    ///< periodicity M2C operator data

    FMMData() = delete; ///< forbid default constructor

    // forbid copy constructor
    FMMData(const FMMData &) = delete;
    FMMData &operator=(const FMMData &) = delete;
    FMMData(FMMData &&) = delete;
    FMMData &operator=(FMMData &&) = delete;

    /**
     * @brief Construct a new FMMData object
     *
     * @param kernelChoice_
     * @param periodicity_
     * @param multOrder_
     * @param maxPts_
     */
    FMMData(KERNEL kernelChoice_, PAXIS periodicity_, int multOrder_, int maxPts_, bool enableFF_ = true);

    /**
     * @brief Destroy the FMMData object
     *
     */
    ~FMMData();

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
     * @param ntreePts
     * @param treePtsPtr
     */
    void setupTree(const std::vector<double> &srcSLCoord, const std::vector<double> &srcDLCoord,
                   const std::vector<double> &trgCoord, const int ntreePts = 0, const double *treePtsPtr = nullptr);

    /**
     * @brief runFMM
     *
     * @param srcSLValue [in] single layer source value
     * @param srcDLValue [in] double layer source value
     * @param trgValue [out] target value
     * @param scale
     */
    void evaluateFMM(std::vector<double> &srcSLValue, std::vector<double> &srcDLValue, std::vector<double> &trgValue,
                     const double scale);

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
                        const int nTrg, double *trgCoordPtr, double *trgValuePtr);

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

    /**
     * @brief if this kernel has DL
     *
     * @return true
     * @return false
     */
    bool hasDL() const { return kernelFunctionPtr->dbl_layer_poten; }

  private:
    pvfmm::PtFMM<double> *matrixPtr;        ///< pvfmm PtFMM pointer
    pvfmm::PtFMM_Tree<double> *treePtr;     ///< pvfmm PtFMM_Tree pointer
    pvfmm::PtFMM_Data<double> *treeDataPtr; ///< pvfmm PtFMM_Data pointer
    MPI_Comm comm;                          ///< MPI_comm communicator

    /**
     * @brief scale SrcSl and SrcDL Values before FMM call
     *  operate on srcSLValue and srcDLValue
     *
     * @param srcSLValue
     * @param srcDLValue
     * @param scaleFactor
     */
    void scaleSrc(std::vector<double> &srcSLValue, std::vector<double> &srcDLValue, const double scaleFactor);

    /**
     * @brief scale Trg Values after FMM call
     *  operate on trgSLValue
     *
     * @param trgDLValue
     * @param scaleFactor
     */
    void scaleTrg(std::vector<double> &trgDLValue, const double scaleFactor);

    /**
     * @brief read the M2L Matrix from file
     *
     */
    void readMat(const int kDim, const std::string &dataName, std::vector<double> &data);

    /**
     * @brief setup this->M2Ldata, this->M2Cdata
     *
     */
    void setupPeriodicData();

    /**
     * @brief periodize the target values
     *
     *
     * @param trgValue
     */
    void periodizeFMM(std::vector<double> &trgValue);
};

} // namespace impl
} // namespace stkfmm
#endif