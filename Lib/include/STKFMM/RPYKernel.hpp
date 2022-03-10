#ifndef INCLUDE_RPYCUSTOMKERNEL_H_
#define INCLUDE_RPYCUSTOMKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {

/**********************************************************
 *                                                        *
 *     RPY velocity kernel, source: 4, target: 3          *
 *           fx,fy,fz,a -> ux,uy,uz                       *
 **********************************************************/
struct rpy_u : public GenericKernel<rpy_u> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType one_over_three = (typename VecType::ScalarType)(0.3333333333333);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &a = f[3];
        // clang-format on

        VecType fdotr = fx * dx + fy * dy + fz * dz;
        VecType three_fdotr_rinv5 = three * fdotr * rinv5;
        VecType a2_over_three = one_over_three * a * a;
        u[0] += (r2 * fx + dx * fdotr) * rinv3 + a2_over_three * (fx * rinv3 - three_fdotr_rinv5 * dx);
        u[1] += (r2 * fy + dy * fdotr) * rinv3 + a2_over_three * (fy * rinv3 - three_fdotr_rinv5 * dy);
        u[2] += (r2 * fz + dz * fdotr) * rinv3 + a2_over_three * (fz * rinv3 - three_fdotr_rinv5 * dz);
    }
};

/**********************************************************
 *                                                        *
 * RPY Force,a Vel,lapVel kernel, source: 4, target: 6    *
 *       fx,fy,fz,a -> ux,uy,uz,lapux,lapuy,lapuz         *
 **********************************************************/
struct rpy_ulapu : public GenericKernel<rpy_ulapu> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[6], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType one_over_three = (typename VecType::ScalarType)(0.3333333333333);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &a = f[3];
        // clang-format on

        VecType fdotr = fx * dx + fy * dy + fz * dz;
        VecType fdotr_rinv3 = fdotr * rinv3;
        VecType three_fdotr_rinv5 = three * fdotr * rinv5;
        VecType a2_over_three = one_over_three * a * a;

        VecType cx = fx * rinv3 - three_fdotr_rinv5 * dx;
        VecType cy = fy * rinv3 - three_fdotr_rinv5 * dy;
        VecType cz = fz * rinv3 - three_fdotr_rinv5 * dz;

        u[0] += fx * rinv + dx * fdotr_rinv3 + a2_over_three * cx;
        u[1] += fy * rinv + dy * fdotr_rinv3 + a2_over_three * cy;
        u[2] += fz * rinv + dz * fdotr_rinv3 + a2_over_three * cz;
        u[3] += two * cx;
        u[4] += two * cy;
        u[5] += two * cz;
    }
};

/**********************************************************
 *                                                        *
 * Stokes Force Vel,lapVel kernel,source: 3, target: 6    *
 *       fx,fy,fz -> ux,uy,uz,lapux,lapuy,lapuz           *
 **********************************************************/
struct stk_ulapu : public GenericKernel<stk_ulapu> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[6], const VecType (&r)[3], const VecType (&f)[3], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        // clang-format on

        VecType fdotr = fx * dx + fy * dy + fz * dz;
        VecType fdotr_rinv3 = fdotr * rinv3;
        VecType three_fdotr_rinv5 = three * fdotr * rinv5;

        u[0] += fx * rinv + dx * fdotr_rinv3;
        u[1] += fy * rinv + dy * fdotr_rinv3;
        u[2] += fz * rinv + dz * fdotr_rinv3;

        u[3] += two * (fx * rinv3 - three_fdotr_rinv5 * dx);
        u[4] += two * (fy * rinv3 - three_fdotr_rinv5 * dy);
        u[5] += two * (fz * rinv3 - three_fdotr_rinv5 * dz);
    }
};

template <class T>
struct RPYKernel {
    inline static const Kernel<T> &ulapu(); //   3+1->6
};

template <class T>
inline const Kernel<T> &RPYKernel<T>::ulapu() {
    static Kernel<T> g_ker = StokesKernel<T>::velocity();
    static Kernel<T> gr_ker = BuildKernel<T, rpy_u::Eval<T>>("rpy_u", 3, std::pair<int, int>(4, 3));

    static Kernel<T> glapg_ker = BuildKernel<T, stk_ulapu::Eval<T>>("stk_ulapu", 3, std::pair<int, int>(3, 6));

    static Kernel<T> grlapgr_ker = BuildKernel<T, rpy_ulapu::Eval<T>>("rpy_ulapu", 3, std::pair<int, int>(4, 6),
                                                                      &gr_ker,    // k_s2m
                                                                      &gr_ker,    // k_s2l
                                                                      NULL,       // k_s2t
                                                                      &g_ker,     // k_m2m
                                                                      &g_ker,     // k_m2l
                                                                      &glapg_ker, // k_m2t
                                                                      &g_ker,     // k_l2l
                                                                      &glapg_ker, // k_l2t
                                                                      NULL);
    return grlapgr_ker;
}

} // namespace pvfmm

#endif
