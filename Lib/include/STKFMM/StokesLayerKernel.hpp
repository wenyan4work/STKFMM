/**
 * @file StokesLayerKernel.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief Stokes kernels
 * @version 0.1
 * @date 2019-12-23
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef STOKESLAYERKERNEL_HPP
#define STOKESLAYERKERNEL_HPP

#include "StokesDoubleLayerKernel.hpp"
#include "StokesSingleLayerKernel.hpp"

/**
 * @brief inject into pvfmm namespace
 *
 */
namespace pvfmm {

/**
 * @brief Stokes Layer Kernels
 *
 * @tparam T float or double
 */
template <class T>
struct StokesLayerKernel {
    // S2U - single-layer density — to — potential kernel
    // D2U - double-layer density — to — potential kernel
    // S2UdU - single-layer density — to — potential & gradient
    // D2UdU - double-layer density — to — potential & gradient

    inline static const Kernel<T> &PVel();          ///< SL+DL -> PVel
    inline static const Kernel<T> &PVelGrad();      ///< SL+DL -> PVelGrad
    inline static const Kernel<T> &PVelLaplacian(); ///< SL+DL -> PVelLaplacian
    inline static const Kernel<T> &Traction();      ///< SL+DL -> Traction

  private:
    /**
     * @brief number of Newton iteration times
     *  generate NEWTON_ITE at compile time.
     * 1 for float and 2 for double
     *
     */
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
};

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::PVel() {
    static Kernel<T> stokes_pker =
        BuildKernel<T, stokes_pvel<T, NEWTON_ITE>,
                    stokes_doublepvel<T, NEWTON_ITE>>(
            "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    return stokes_pker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::PVelGrad() {
    static Kernel<T> stokes_pker =
        BuildKernel<T, stokes_pvel<T, NEWTON_ITE>,
                    stokes_doublepvel<T, NEWTON_ITE>>(
            "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    static Kernel<T> stokes_pgker =
        BuildKernel<T, stokes_pvelgrad<T, NEWTON_ITE>,
                    stokes_doublepvelgrad<T, NEWTON_ITE>>(
            "stokes_PVelGrad", 3, std::pair<int, int>(4, 16), &stokes_pker,
            &stokes_pker, NULL, &stokes_pker, &stokes_pker, NULL, &stokes_pker,
            NULL);
    stokes_pgker.surf_dim = 9;
    return stokes_pgker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::PVelLaplacian() {
    static Kernel<T> stokes_pker =
        BuildKernel<T, stokes_pvel<T, NEWTON_ITE>,
                    stokes_doublepvel<T, NEWTON_ITE>>(
            "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    static Kernel<T> stokes_pgker =
        BuildKernel<T, stokes_pvellaplacian<T, NEWTON_ITE>,
                    stokes_doublelaplacian<T, NEWTON_ITE>>(
            "stokes_PVelLaplacian", 3, std::pair<int, int>(4, 7), &stokes_pker,
            &stokes_pker, NULL, &stokes_pker, &stokes_pker, NULL, &stokes_pker,
            NULL);
    stokes_pgker.surf_dim = 9;
    return stokes_pgker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::Traction() {
    static Kernel<T> stokes_pker =
        BuildKernel<T, stokes_pvel<T, NEWTON_ITE>,
                    stokes_doublepvel<T, NEWTON_ITE>>(
            "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    static Kernel<T> stokes_pgker =
        BuildKernel<T, stokes_traction<T, NEWTON_ITE>,
                    stokes_doubletraction<T, NEWTON_ITE>>(
            "stokes_Traction", 3, std::pair<int, int>(4, 9), &stokes_pker,
            &stokes_pker, NULL, &stokes_pker, &stokes_pker, NULL, &stokes_pker,
            NULL);
    stokes_pgker.surf_dim = 9;
    return stokes_pgker;
}
} // namespace pvfmm

#endif