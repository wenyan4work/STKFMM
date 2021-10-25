/**
 * @file LaplaceLayerKernel.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief Laplace kernels
 * @version 0.1
 * @date 2019-12-23
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef LAPLACELAYERKERNEL_HPP_
#define LAPLACELAYERKERNEL_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

// Kernel building utilities
#include "stkfmm_helpers.hpp"

/**
 * @brief insert kernel functions to pvfmm namespace
 *
 */
namespace pvfmm {

template <class T>

/**
 * @brief LaplaceLayerkernel class
 *
 * @tparam T float or double
 */
struct LaplaceLayerKernel {
    // inline static const Kernel<T> &Grad();    ///< Laplace Grad Kernel, for test only
    inline static const Kernel<T> &PGrad();      ///< Laplace PGrad Kernel
    inline static const Kernel<T> &PGradGrad();  ///< Laplace PGradGrad
    inline static const Kernel<T> &QPGradGrad(); ///< Laplace Quadruple PGradGrad, no double layer
};

struct laplace_p : public GenericKernel<laplace_p> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }

    /**
     * @brief micro kernel for Laplace single layer potential + gradient
     *
     * @tparam VecType
     * @tparam digits
     * @param trg_value
     * @param src_coord
     * @param trg_value
     * @param ctx_ptr
     */
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[1], const VecType (&r)[3], const VecType (&f)[1], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());

        u[0] += f[0] * rinv;
    }
};

struct laplace_pgrad : public GenericKernel<laplace_pgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }

    /**
     * @brief micro kernel for Laplace single layer potential + gradient
     *
     * @tparam VecType
     * @tparam digits
     * @param trg_value
     * @param src_coord
     * @param trg_value
     * @param ctx_ptr
     */
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[4], const VecType (&r)[3], const VecType (&f)[1], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());

        VecType sv = f[0] * rinv * rinv * rinv;

        u[0] += sv * r2;
        u[1] -= sv * r[0];
        u[2] -= sv * r[1];
        u[3] -= sv * r[2];
    }
};

struct laplace_dipolep : public GenericKernel<laplace_dipolep> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[1], const VecType (&r)[3], const VecType (&f)[3], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rdotn = f[0] * r[0] + f[1] * r[1] + f[2] * r[2];

        u[0] += rinv3 * rdotn;
    }
};

struct laplace_dipolepgrad : public GenericKernel<laplace_dipolepgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[4], const VecType (&r)[3], const VecType (&f)[3], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv2 * rinv;
        VecType rinv5 = rinv3 * rinv2;
        VecType rdotn = f[0] * r[0] + f[1] * r[1] + f[2] * r[2];
        VecType three = (typename VecType::ScalarType)(3.0);

        u[0] += rdotn * rinv3;
        u[1] += (f[0] * r2 - three * rdotn * r[0]) * rinv5;
        u[2] += (f[1] * r2 - three * rdotn * r[1]) * rinv5;
        u[3] += (f[2] * r2 - three * rdotn * r[2]) * rinv5;
    }
};

struct laplace_pgradgrad : public GenericKernel<laplace_pgradgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[10], const VecType (&r)[3], const VecType (&f)[1], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv2 * rinv;
        VecType three = (typename VecType::ScalarType)(3.0);
        VecType sv = f[0] * rinv3;

        u[0] += sv * r2;
        u[1] -= sv * r[0];
        u[2] -= sv * r[1];
        u[3] -= sv * r[2];

        sv *= rinv2;
        u[4] += sv * (three * r[0] * r[0] - r2);
        u[5] += sv * three * r[0] * r[1];
        u[6] += sv * three * r[0] * r[2];
        u[7] += sv * (three * r[1] * r[1] - r2);
        u[8] += sv * three * r[1] * r[2];
        u[9] += sv * (three * r[2] * r[2] - r2);
    }
};

struct laplace_dipolepgradgrad : public GenericKernel<laplace_dipolepgradgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[10], const VecType (&r)[3], const VecType (&f)[3], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType threer2 = three * r2;
        const VecType six = (typename VecType::ScalarType)(6.0);
        const VecType fifteen = (typename VecType::ScalarType)(15.0);
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv2 * rinv;
        VecType rinv5 = rinv2 * rinv3;
        VecType rinv7 = rinv2 * rinv5;
        VecType rdotn = r[0] * f[0] + r[1] * f[1] + r[2] * f[2];

        u[0] += rdotn * rinv3;
        u[1] += (f[0] * r2 - rdotn * three * r[0]) * rinv5;
        u[2] += (f[1] * r2 - rdotn * three * r[1]) * rinv5;
        u[3] += (f[2] * r2 - rdotn * three * r[2]) * rinv5;

        u[4] += (fifteen * r[0] * r[0] * rdotn - r2 * (three * rdotn + six * r[0] * f[0])) * rinv7;
        u[5] += (fifteen * r[0] * r[1] * rdotn - threer2 * (r[0] * f[1] + r[1] * f[0])) * rinv7;
        u[6] += (fifteen * r[0] * r[2] * rdotn - threer2 * (r[0] * f[2] + r[2] * f[0])) * rinv7;
        u[7] += (fifteen * r[1] * r[1] * rdotn - r2 * (three * rdotn + six * r[1] * f[1])) * rinv7;
        u[8] += (fifteen * r[1] * r[2] * rdotn - threer2 * (r[1] * f[2] + r[2] * f[1])) * rinv7;
        u[9] += (fifteen * r[2] * r[2] * rdotn - r2 * (three * rdotn + six * r[2] * f[2])) * rinv7;
    }
};

struct laplace_quadp : public GenericKernel<laplace_quadp> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[1], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv5 = rinv * rinv;
        rinv5 = rinv5 * rinv5 * rinv;

        // clang-format off
        const VecType &sxx = f[0], &sxy = f[1], &sxz = f[2];
        const VecType &syx = f[3], &syy = f[4], &syz = f[5];
        const VecType &szx = f[6], &szy = f[7], &szz = f[8];
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        // clang-format on

        VecType commonCoeff = f[0] * dx * dx;
        commonCoeff += (sxy + syx) * dx * dy;
        commonCoeff += (sxz + szx) * dx * dz;
        commonCoeff += syy * dy * dy;
        commonCoeff += (syz + szy) * dy * dz;
        commonCoeff += szz * dz * dz;
        commonCoeff *= (typename VecType::ScalarType)(-3.0);

        u[0] += (commonCoeff + r2 * (sxx + syy + szz)) * rinv5;
    }
};

struct laplace_quadpgradgrad : public GenericKernel<laplace_quadpgradgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (4.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[10], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv2 * rinv;
        VecType rinv5 = rinv3 * rinv2;
        VecType rinv7 = rinv5 * rinv2;
        VecType rinv9 = rinv7 * rinv2;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType five = (typename VecType::ScalarType)(5.0);
        const VecType fifteen = (typename VecType::ScalarType)(15.0);
        const VecType onezerofive = (typename VecType::ScalarType)(105.0);

        // clang-format off
        const VecType &sxx = f[0], &sxy = f[1], &sxz = f[2];
        const VecType &syx = f[3], &syy = f[4], &syz = f[5];
        const VecType &szx = f[6], &szy = f[7], &szz = f[8];
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        // clang-format on

        VecType rrQ = dx * dx * sxx + dx * dy * (sxy + syx) + dx * dz * (sxz + szx) + dy * dy * syy +
                      dy * dz * (syz + szy) + dz * dz * szz;

        VecType commonCoeff3 = rrQ * three;
        VecType commonCoeff5 = rrQ * five;

        VecType trace = sxx + syy + szz;

        VecType rksxk = dx * sxx + dy * sxy + dz * sxz;
        VecType rksyk = dx * syx + dy * syy + dz * syz;
        VecType rkszk = dx * szx + dy * szy + dz * szz;

        VecType rkskx = dx * sxx + dy * syx + dz * szx;
        VecType rksky = dx * sxy + dy * syy + dz * szy;
        VecType rkskz = dx * sxz + dy * syz + dz * szz;

        u[0] += (commonCoeff3 - r2 * trace) * rinv5; // p
        VecType px = dx * commonCoeff5 - r2 * (rksxk + rkskx + dx * trace);
        VecType py = dy * commonCoeff5 - r2 * (rksyk + rksky + dy * trace);
        VecType pz = dz * commonCoeff5 - r2 * (rkszk + rkskz + dz * trace);

        u[1] -= three * px * rinv7; // gx
        u[2] -= three * py * rinv7; // gy
        u[3] -= three * pz * rinv7; // gz

        // gxx
        u[4] +=
            rinv9 * (onezerofive * dx * dx * rrQ - r2 * fifteen * (rrQ + dx * two * (rksxk + rkskx) + dx * dx * trace) +
                     three * r2 * r2 * (trace + two * sxx));
        // gxy
        u[5] += rinv9 * (onezerofive * (dx * dy * rrQ) -
                         r2 * fifteen * ((dy * rksxk + dy * rkskx + dx * rksyk + dx * rksky) + dx * dy * trace) +
                         three * r2 * r2 * (sxy + syx));
        // gxz
        u[6] += rinv9 * (onezerofive * (dx * dz * rrQ) -
                         r2 * fifteen * ((dz * rksxk + dz * rkskx + dx * rkszk + dx * rkskz) + dx * dz * trace) +
                         three * r2 * r2 * (sxz + szx));
        // gyy
        u[7] +=
            rinv9 * (onezerofive * dy * dy * rrQ - r2 * fifteen * (rrQ + dy * two * (rksyk + rksky) + dy * dy * trace) +
                     three * r2 * r2 * (trace + two * syy));
        // gyz
        u[8] += rinv9 * (onezerofive * (dy * dz * rrQ) -
                         r2 * fifteen * ((dy * rkszk + dy * rkskz + dz * rksyk + dz * rksky) + dy * dz * trace) +
                         three * r2 * r2 * (syz + szy));
        // gzz
        u[9] +=
            rinv9 * (onezerofive * dz * dz * rrQ - r2 * fifteen * (rrQ + dz * two * (rkszk + rkskz) + dz * dz * trace) +
                     three * r2 * r2 * (trace + two * szz));
    }
};

template <class T>
const Kernel<T> &LaplaceLayerKernel<T>::PGrad() {
    static Kernel<T> lap_pker =
        BuildKernel<T, laplace_p::Eval<T>, laplace_dipolep::Eval<T>>("laplace", 3, std::pair<int, int>(1, 1));
    lap_pker.surf_dim = 3;

    static Kernel<T> lap_pgker = BuildKernel<T, laplace_pgrad::Eval<T>, laplace_dipolepgrad::Eval<T>>(
        "laplace_PGrad", 3, std::pair<int, int>(1, 4), &lap_pker, &lap_pker, NULL, &lap_pker, &lap_pker, NULL,
        &lap_pker, NULL);
    lap_pgker.surf_dim = 3;

    return lap_pgker;
}

template <class T>
inline const Kernel<T> &LaplaceLayerKernel<T>::PGradGrad() {

    static Kernel<T> lap_pker =
        BuildKernel<T, laplace_p::Eval<T>, laplace_dipolep::Eval<T>>("laplace", 3, std::pair<int, int>(1, 1));
    lap_pker.surf_dim = 3;

    static Kernel<T> lap_pgker = BuildKernel<T, laplace_pgradgrad::Eval<T>, laplace_dipolepgradgrad::Eval<T>>(
        "laplace_PGradGrad", 3, std::pair<int, int>(1, 10), &lap_pker, &lap_pker, NULL, &lap_pker, &lap_pker, NULL,
        &lap_pker, NULL);
    lap_pgker.surf_dim = 3;

    return lap_pgker;
}

template <class T>
inline const Kernel<T> &LaplaceLayerKernel<T>::QPGradGrad() {
    static Kernel<T> lap_pker = BuildKernel<T, laplace_p::Eval<T>>("laplace", 3, std::pair<int, int>(1, 1));
    static Kernel<T> lap_pggker = BuildKernel<T, laplace_pgradgrad::Eval<T>>("laplace", 3, std::pair<int, int>(1, 10));
    static Kernel<T> lap_qpker = BuildKernel<T, laplace_quadp::Eval<T>>("laplace", 3, std::pair<int, int>(9, 1));

    static Kernel<T> lap_pgker = BuildKernel<T, laplace_quadpgradgrad::Eval<T>>(
        "laplace_QPGradGrad", 3, std::pair<int, int>(9, 10), &lap_qpker, &lap_qpker, NULL, //
        &lap_pker, &lap_pker, &lap_pggker,                                                 //
        &lap_pker, &lap_pggker);

    return lap_pgker;
}

} // namespace pvfmm

#endif
