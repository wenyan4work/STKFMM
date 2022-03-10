#ifndef STOKESREGSINGLELAYER_HPP_
#define STOKESREGSINGLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {

/*********************************************************
 *                                                        *
 *     Stokes Reg Vel kernel, source: 4, target: 3        *
 *              fx,fy,fz,eps -> ux,uy,uz                  *
 **********************************************************/
struct stokes_regvel : public GenericKernel<stokes_regvel> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &reg = f[3];
        // clang-format on
        VecType r2 = dx * dx + dy * dy + dz * dz + reg * reg;

        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType r2reg2 = r2 + reg * reg;

        VecType commonCoeff = fx * dx + fy * dy + fz * dz;
        u[0] += rinv3 * (r2reg2 * fx + dx * commonCoeff);
        u[1] += rinv3 * (r2reg2 * fy + dy * commonCoeff);
        u[2] += rinv3 * (r2reg2 * fz + dz * commonCoeff);
    }
};

/**********************************************************
 *                                                        *
 * Stokes Reg Force Torque Vel kernel,source: 7, target: 3*
 *       fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz                *
 **********************************************************/
struct stokes_regftvel : public GenericKernel<stokes_regftvel> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[7], const void *ctx_ptr) {
        const VecType half = (typename VecType::ScalarType)(0.5);
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType five = (typename VecType::ScalarType)(5.0);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &tx = f[3], &ty = f[4], &tz = f[5];
        const VecType &eps = f[6];
        // clang-format on
        VecType eps2 = eps * eps;

        VecType r2 = dx * dx + dy * dy + dz * dz;
        VecType denom_arg = r2 + eps2;
        VecType rinv = sctl::approx_rsqrt<digits>(denom_arg);

        VecType stokeslet_denom_inv = rinv * rinv * rinv;
        VecType rotlet_denom_inv = half * stokeslet_denom_inv * rinv * rinv;
        VecType rotlet_coef = (two * r2 + five * eps2) * rotlet_denom_inv;
        VecType H2 = stokeslet_denom_inv;
        VecType H1 = (r2 + two * eps2) * H2;

        VecType tcurlrx = ty * dz - tz * dy;
        VecType tcurlry = tz * dx - tx * dz;
        VecType tcurlrz = tx * dy - ty * dx;

        VecType fdotr = fx * dx + fy * dy + fz * dz;
        VecType tdotr = tx * dx + ty * dy + tz * dz;

        u[0] += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
        u[1] += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
        u[2] += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;
    }
};

/****************************************************************
 *                                                              *
 *Stokes Reg Force Torque Vel Omega kernel, source: 7, target: 6*
 *    fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz,wx,wy,wz                *
 ****************************************************************/
struct stokes_regftvelomega : public GenericKernel<stokes_regftvelomega> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[6], const VecType (&r)[3], const VecType (&f)[7], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        const VecType half = (typename VecType::ScalarType)(0.5);
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType five = (typename VecType::ScalarType)(5.0);
        const VecType seven = (typename VecType::ScalarType)(7.0);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &tx = f[3], &ty = f[4], &tz = f[5];
        const VecType &eps = f[6];
        // clang-format on
        VecType r4 = r2 * r2;
        VecType eps2 = eps * eps;
        VecType eps4 = eps2 * eps2;

        VecType rinv = sctl::approx_rsqrt<digits>(eps2 + r2);

        const VecType stokeslet_denom_inv = rinv * rinv * rinv;
        VecType rotlet_denom_inv = half * stokeslet_denom_inv * rinv * rinv;
        VecType dipole_denom_inv = half * rotlet_denom_inv * rinv * rinv;
        VecType rotlet_coef = (two * r2 + five * eps2) * rotlet_denom_inv;
        VecType D1 = (two * five * eps4 - seven * eps2 * r2 - two * r4) * dipole_denom_inv;
        VecType D2 = (seven * three * eps2 + two * three * r2) * dipole_denom_inv;
        const VecType &H2 = stokeslet_denom_inv;
        VecType H1 = (r2 + two * eps2) * H2;

        VecType fcurlrx = fy * dz - fz * dy;
        VecType fcurlry = fz * dx - fx * dz;
        VecType fcurlrz = fx * dy - fy * dx;

        VecType tcurlrx = ty * dz - tz * dy;
        VecType tcurlry = tz * dx - tx * dz;
        VecType tcurlrz = tx * dy - ty * dx;

        VecType fdotr = fx * dx + fy * dy + fz * dz;
        VecType tdotr = tx * dx + ty * dy + tz * dz;

        u[0] += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
        u[1] += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
        u[2] += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;

        u[3] += D1 * tx + D2 * tdotr * dx + rotlet_coef * fcurlrx;
        u[4] += D1 * ty + D2 * tdotr * dy + rotlet_coef * fcurlry;
        u[5] += D1 * tz + D2 * tdotr * dz + rotlet_coef * fcurlrz;
    }
};

/**********************************************************
 *                                                         *
 *   Stokes Force Vel Omega kernel, source: 3, target: 6   *
 *           fx,fy,fz -> ux,uy,uz, wx,wy,wz                *
 **********************************************************/
struct stokes_velomega : public GenericKernel<stokes_velomega> {
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
        const VecType half = (typename VecType::ScalarType)(0.5);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType one_over_three = (typename VecType::ScalarType)(0.3333333333333);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &reg = f[3];
        // clang-format on

        VecType r4 = r2 * r2;
        const VecType &denom_arg = r2;

        const VecType &stokeslet_denom_inv = rinv3;
        VecType rotlet_denom_inv = (typename VecType::ScalarType)(0.5) * stokeslet_denom_inv * rinv * rinv;
        VecType rotlet_coef = (typename VecType::ScalarType)(2.0) * r2 * rotlet_denom_inv;
        VecType H2 = stokeslet_denom_inv;
        VecType H1 = r2 * H2;

        VecType fcurlrx = fy * dz - fz * dy;
        VecType fcurlry = fz * dx - fx * dz;
        VecType fcurlrz = fx * dy - fy * dx;

        VecType fdotr = fx * dx + fy * dy + fz * dz;

        u[0] += H1 * fx + H2 * fdotr * dx;
        u[1] += H1 * fy + H2 * fdotr * dy;
        u[2] += H1 * fz + H2 * fdotr * dz;

        u[3] += rotlet_coef * fcurlrx;
        u[4] += rotlet_coef * fcurlry;
        u[5] += rotlet_coef * fcurlrz;
    }
};

template <class T>
struct StokesRegKernel {
    inline static const Kernel<T> &Vel();        //   3+1->3
    inline static const Kernel<T> &FTVelOmega(); //   3+3+1->3+3
};

template <class T>
inline const Kernel<T> &StokesRegKernel<T>::Vel() {
    static Kernel<T> stk_ker = StokesKernel<T>::velocity();
    static Kernel<T> s2t_ker =
        BuildKernel<T, stokes_regvel::Eval<T>>("stokes_regvel", 3, std::pair<int, int>(4, 3), NULL, NULL, NULL,
                                               &stk_ker, &stk_ker, &stk_ker, &stk_ker, &stk_ker, NULL, true);

    return s2t_ker;
}

template <class T>
inline const Kernel<T> &StokesRegKernel<T>::FTVelOmega() {
    static Kernel<T> stk_ker = StokesKernel<T>::velocity();
    static Kernel<T> stk_velomega =
        BuildKernel<T, stokes_velomega::Eval<T>>("stokes_velomega", 3, std::pair<int, int>(3, 6));
    static Kernel<T> stk_regftvel =
        BuildKernel<T, stokes_regftvel::Eval<T>>("stokes_regftvel", 3, std::pair<int, int>(7, 3));
    static Kernel<T> s2t_ker = BuildKernel<T, stokes_regftvelomega::Eval<T>>(
        "stokes_regftvelomega", 3, std::pair<int, int>(7, 6), &stk_regftvel, &stk_regftvel, NULL, &stk_ker, &stk_ker,
        &stk_velomega, &stk_ker, &stk_velomega);

    return s2t_ker;
}

} // namespace pvfmm
#endif // STOKESSINGLELAYERKERNEL_HPP
