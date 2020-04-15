#ifndef STKFMM_COMMON_
#define STKFMM_COMMON_

#include "LaplaceLayerKernel.hpp"
#include "RPYKernel.hpp"
#include "StokesLayerKernel.hpp"
#include "StokesRegSingleLayerKernel.hpp"

#include <unordered_map>

#include <pvfmm.hpp>

namespace stkfmm{
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
    LapPGrad = 1,         ///< Laplace
    LapPGradGrad = 2,     ///< Laplace
    LapQPGradGrad = 4, ///< Laplace

    Stokes = 8,             ///< Stokeslet 3x3
    RPY = 16,               ///< RPY
    StokesRegVel = 32,      ///< Regularized Stokes Velocity
    StokesRegVelOmega = 64, ///< Regularized Stokes Velocity/Rotation

    PVel = 128,          ///< Stokes 4x4
    PVelGrad = 256,      ///< Stokes
    PVelLaplacian = 512, ///< Stokes
    Traction = 1024,     ///< Stokes
};

extern const std::unordered_map<KERNEL, const pvfmm::Kernel<double> *>
    kernelMap;

/**
 * @brief Get kernel dimension
 *
 * @param kernel_ one of the activated kernels
 * @return [single layer kernel dimension, double layer kernel dimension,
 *          target kernel dimension]
 */
std::tuple<int, int, int> getKernelDimension(KERNEL kernel_);

/**
 * @brief Get the name of the kernel
 * 
 * @param kernel_ 
 * @return std::string 
 */
std::string getKernelName(KERNEL kernel_);

/**
 * @brief Get the Kernel Function Pointer
 * 
 * @param kernelChoice_ 
 * @return const pvfmm::Kernel<double>* 
 */
const pvfmm::Kernel<double> *getKernelFunction(KERNEL kernelChoice_) ;

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

}
#endif