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
    inline static const Kernel<T> &Vel();           ///< Stokeslet 3x3, SL->Vel
    inline static const Kernel<T> &PVel();          ///< SL+DL -> PVel
    inline static const Kernel<T> &PVelGrad();      ///< SL+DL -> PVelGrad
    inline static const Kernel<T> &PVelLaplacian(); ///< SL+DL -> PVelLaplacian
    inline static const Kernel<T> &Traction();      ///< SL+DL -> Traction
};


template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::Vel() {
    static Kernel<T> ker = BuildKernel<T, stokes_vel::Eval<T>>("stokes_vel", 3, std::pair<int, int>(3, 3));
    return ker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::PVel() {
    static Kernel<T> stokes_pker = BuildKernel<T, stokes_pvel::Eval<T>, stokes_doublepvel::Eval<T>>(
        "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    return stokes_pker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::PVelGrad() {
    static Kernel<T> stokes_pker = BuildKernel<T, stokes_pvel::Eval<T>, stokes_doublepvel::Eval<T>>(
        "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    static Kernel<T> stokes_pgker = BuildKernel<T, stokes_pvelgrad::Eval<T>, stokes_doublepvelgrad::Eval<T>>(
        "stokes_PVelGrad", 3, std::pair<int, int>(4, 16), &stokes_pker, &stokes_pker, NULL, &stokes_pker, &stokes_pker,
        NULL, &stokes_pker, NULL);
    stokes_pgker.surf_dim = 9;
    return stokes_pgker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::PVelLaplacian() {
    static Kernel<T> stokes_pker = BuildKernel<T, stokes_pvel::Eval<T>, stokes_doublepvel::Eval<T>>(
        "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    static Kernel<T> stokes_pgker =
        BuildKernel<T, stokes_pvellaplacian::Eval<T>, stokes_doublelaplacian::Eval<T>>(
            "stokes_PVelLaplacian", 3, std::pair<int, int>(4, 7), &stokes_pker, &stokes_pker, NULL, &stokes_pker,
            &stokes_pker, NULL, &stokes_pker, NULL);
    stokes_pgker.surf_dim = 9;
    return stokes_pgker;
}

template <class T>
inline const Kernel<T> &StokesLayerKernel<T>::Traction() {
    static Kernel<T> stokes_pker = BuildKernel<T, stokes_pvel::Eval<T>, stokes_doublepvel::Eval<T>>(
        "stokes_PVel", 3, std::pair<int, int>(4, 4));
    stokes_pker.surf_dim = 9;
    static Kernel<T> stokes_pgker =
        BuildKernel<T, stokes_traction::Eval<T>, stokes_doubletraction::Eval<T>>(
            "stokes_Traction", 3, std::pair<int, int>(4, 9), &stokes_pker, &stokes_pker, NULL, &stokes_pker,
            &stokes_pker, NULL, &stokes_pker, NULL);
    stokes_pgker.surf_dim = 9;
    return stokes_pgker;
}
} // namespace pvfmm

#endif
