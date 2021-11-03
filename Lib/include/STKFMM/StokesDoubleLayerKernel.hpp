/**
 * @file StokesDoubleLayerKernel.hpp
 * @author Wen Yan (wenyan4work@gmail.com), Robert Blackwell (rblackwell@flatironinstitute.org)
 * @brief Stokes double layer kernels
 * @version 0.2
 * @date 2019-12-23, 2021-10-27
 *
 * @copyright Copyright (c) 2019, 2021
 *
 */
#ifndef STOKESDOUBLELAYER_HPP_
#define STOKESDOUBLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

namespace pvfmm {

/*********************************************************
 *                                                        *
 *   Stokes Double P Vel kernel, source: 9, target: 4     *
 *                                                        *
 **********************************************************/
struct stokes_doublepvel : public GenericKernel<stokes_doublepvel> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[4], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType two = (typename VecType::ScalarType)(2.0);
        // clang-format off
        const VecType sxx = f[0], sxy = f[1], sxz = f[2];
        const VecType syx = f[3], syy = f[4], syz = f[5];
        const VecType szx = f[6], szy = f[7], szz = f[8];
        const VecType dx  = r[0], dy  = r[1], dz  = r[2];
        // clang-format on

        VecType commonCoeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
        commonCoeff += (sxy + syx) * dx * dy;
        commonCoeff += (sxz + szx) * dx * dz;
        commonCoeff += (syz + szy) * dy * dz;
        commonCoeff *= (typename VecType::ScalarType)(-3.0) * rinv5;

        const VecType trace = sxx + syy + szz;
        u[0] += two * (commonCoeff + r2 * rinv5 * trace);
        u[1] += dx * commonCoeff;
        u[2] += dy * commonCoeff;
        u[3] += dz * commonCoeff;
    }
};

/*********************************************************
 *                                                        *
 * Stokes Double P Vel Grad kernel, source: 9, target: 16 *
 *                                                        *
 **********************************************************/
struct stokes_doublepvelgrad : public GenericKernel<stokes_doublepvelgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[16], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv * rinv2;
        VecType rinv5 = rinv3 * rinv2;
        VecType rinv7 = rinv5 * rinv2;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType five = (typename VecType::ScalarType)(5.0);
        // clang-format off
        const VecType sxx = f[0], sxy = f[1], sxz = f[2];
        const VecType syx = f[3], syy = f[4], syz = f[5];
        const VecType szx = f[6], szy = f[7], szz = f[8];
        const VecType dx  = r[0], dy  = r[1], dz  = r[2];
        // clang-format on

        VecType commonCoeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
        commonCoeff += (sxy + syx) * dx * dy;
        commonCoeff += (sxz + szx) * dx * dz;
        commonCoeff += (syz + szy) * dy * dz;
        VecType commonCoeffn3 = (typename VecType::ScalarType)(-3.0) * commonCoeff;
        VecType commonCoeff5 = (typename VecType::ScalarType)(5.0) * commonCoeff;

        const VecType trace = sxx + syy + szz;

        VecType rksxk = dx * sxx + dy * sxy + dz * sxz;
        VecType rksyk = dx * syx + dy * syy + dz * syz;
        VecType rkszk = dx * szx + dy * szy + dz * szz;

        VecType rkskx = dx * sxx + dy * syx + dz * szx;
        VecType rksky = dx * sxy + dy * syy + dz * szy;
        VecType rkskz = dx * sxz + dy * syz + dz * szz;

        // pressure terms pick up an extra factor of two
        u[0] += two * (commonCoeffn3 + r2 * trace) * rinv5; // p

        // velocity
        u[1] += rinv5 * dx * commonCoeffn3;
        u[2] += rinv5 * dy * commonCoeffn3;
        u[3] += rinv5 * dz * commonCoeffn3;

        // All r7 terms pick up a factor of -3.0
        // pressure terms an extra two
        rinv7 *= -three;
        u[4] -= two * rinv7 * (dx * commonCoeff5 - r2 * ((rksxk + rkskx) + dx * trace));
        u[5] -= two * rinv7 * (dy * commonCoeff5 - r2 * ((rksyk + rksky) + dy * trace));
        u[6] -= two * rinv7 * (dz * commonCoeff5 - r2 * ((rkszk + rkskz) + dz * trace));

        // vgrad
        VecType commonCoeffn1 = -commonCoeff;
        // (-2*(t0 - s0)*sv0 - (t1 - s1)*(sv1 + sv3) - (t2 - s2)*(sv2 + sv6))
        VecType dcFd0 = -two * dx * sxx - dy * (sxy + syx) - dz * (sxz + szx);

        // (-2*(t1 - s1)*sv4 - (t0 - s0)*(sv1 + sv3) - (t2 - s2)*(sv5 + sv7))
        VecType dcFd1 = -two * dy * syy - dx * (sxy + syx) - dz * (syz + szy);

        // (-2*(t2 - s2)*sv8 - (t0 - s0)*(sv2 + sv6) - (t1 - s1)*(sv5 + sv7))
        VecType dcFd2 = -two * dz * szz - dx * (sxz + szx) - dy * (syz + szy);

        u[7] += (five * commonCoeffn1 * dx * dx - r2 * dx * dcFd0 - r2 * commonCoeffn1) * rinv7;
        u[8] += (five * commonCoeffn1 * dx * dy - r2 * dx * dcFd1) * rinv7;
        u[9] += (five * commonCoeffn1 * dx * dz - r2 * dx * dcFd2) * rinv7;

        u[10] += (five * commonCoeffn1 * dy * dx - r2 * dy * dcFd0) * rinv7;
        u[11] += (five * commonCoeffn1 * dy * dy - r2 * dy * dcFd1 - r2 * commonCoeffn1) * rinv7;
        u[12] += (five * commonCoeffn1 * dy * dz - r2 * dy * dcFd2) * rinv7;

        u[13] += (five * commonCoeffn1 * dz * dx - r2 * dz * dcFd0) * rinv7;
        u[14] += (five * commonCoeffn1 * dz * dy - r2 * dz * dcFd1) * rinv7;
        u[15] += (five * commonCoeffn1 * dz * dz - r2 * dz * dcFd2 - r2 * commonCoeffn1) * rinv7;
    }
};

/*********************************************************
 *                                                        *
 *  Stokes Double P Vel Lap kernel, source: 9, target: 7 *
 *                                                        *
 **********************************************************/
struct stokes_doublelaplacian : public GenericKernel<stokes_doublelaplacian> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[7], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv * rinv2;
        VecType rinv5 = rinv3 * rinv2;
        VecType rinv7 = rinv5 * rinv2;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType five = (typename VecType::ScalarType)(5.0);
        // clang-format off
        const VecType sxx = f[0], sxy = f[1], sxz = f[2];
        const VecType syx = f[3], syy = f[4], syz = f[5];
        const VecType szx = f[6], szy = f[7], szz = f[8];
        const VecType dx  = r[0], dy  = r[1], dz  = r[2];
        // clang-format on

        VecType commonCoeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
        commonCoeff += (sxy + syx) * dx * dy;
        commonCoeff += (sxz + szx) * dx * dz;
        commonCoeff += (syz + szy) * dy * dz;
        VecType commonCoeffn3 = (typename VecType::ScalarType)(-3.0) * commonCoeff;
        VecType commonCoeff5 = (typename VecType::ScalarType)(5.0) * commonCoeff;

        const VecType trace = sxx + syy + szz;

        VecType rksxk = dx * sxx + dy * sxy + dz * sxz;
        VecType rksyk = dx * syx + dy * syy + dz * syz;
        VecType rkszk = dx * szx + dy * szy + dz * szz;

        VecType rkskx = dx * sxx + dy * syx + dz * szx;
        VecType rksky = dx * sxy + dy * syy + dz * szy;
        VecType rkskz = dx * sxz + dy * syz + dz * szz;

        // pressure terms pick up an extra factor of two
        u[0] += two * (commonCoeffn3 + r2 * trace) * rinv5; // p

        // velocity
        u[1] += rinv5 * dx * commonCoeffn3;
        u[2] += rinv5 * dy * commonCoeffn3;
        u[3] += rinv5 * dz * commonCoeffn3;

        // All r7 terms pick up a factor of -3.0
        // pressure terms an extra two
        rinv7 *= -three;
        u[4] -= two * rinv7 * (dx * commonCoeff5 - r2 * ((rksxk + rkskx) + dx * trace));
        u[5] -= two * rinv7 * (dy * commonCoeff5 - r2 * ((rksyk + rksky) + dy * trace));
        u[6] -= two * rinv7 * (dz * commonCoeff5 - r2 * ((rkszk + rkskz) + dz * trace));
    }
};

/*********************************************************
 *                                                        *
 *   Stokes Double Traction kernel, source: 9, target: 9  *
 *                                                        *
 **********************************************************/
struct stokes_doubletraction : public GenericKernel<stokes_doubletraction> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return -3.0 / (8.0 * const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[9], const VecType (&r)[3], const VecType (&f)[9], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv2 = rinv * rinv;
        VecType rinv3 = rinv * rinv2;
        VecType rinv5 = rinv3 * rinv2;
        VecType rinv7 = rinv5 * rinv2;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType three = (typename VecType::ScalarType)(3.0);
        const VecType five = (typename VecType::ScalarType)(5.0);
        const VecType facp = (typename VecType::ScalarType)(0.66666666666666);

        // clang-format off
        const VecType sxx = f[0], sxy = f[1], sxz = f[2];
        const VecType syx = f[3], syy = f[4], syz = f[5];
        const VecType szx = f[6], szy = f[7], szz = f[8];
        const VecType dx  = r[0], dy  = r[1], dz  = r[2];
        // clang-format on

        VecType commonCoeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
        commonCoeff += (sxy + syx) * dx * dy;
        commonCoeff += (sxz + szx) * dx * dz;
        commonCoeff += (syz + szy) * dy * dz;
        VecType commonCoeffn3 = (typename VecType::ScalarType)(-3.0) * commonCoeff;
        VecType commonCoeff5 = (typename VecType::ScalarType)(5.0) * commonCoeff;

        VecType trace = sxx + syy + szz;
        VecType dcFd0 = -two * dx * sxx - dy * (sxy + syx) - dz * (sxz + szx);
        VecType dcFd1 = -two * dy * syy - dx * (sxy + syx) - dz * (syz + szy);
        VecType dcFd2 = -two * dz * szz - dx * (sxz + szx) - dy * (syz + szy);

        VecType np = r2 * facp * rinv7 * (commonCoeffn3 + r2 * trace);

        VecType tv0 = (-five * commonCoeff * dx * dx - r2 * dx * dcFd0 + r2 * commonCoeff) * rinv7;
        VecType tv1 = (-five * commonCoeff * dx * dy - r2 * dx * dcFd1) * rinv7;
        VecType tv2 = (-five * commonCoeff * dx * dz - r2 * dx * dcFd2) * rinv7;
        VecType tv3 = (-five * commonCoeff * dy * dx - r2 * dy * dcFd0) * rinv7;
        VecType tv4 = (-five * commonCoeff * dy * dy - r2 * dy * dcFd1 + r2 * commonCoeff) * rinv7;
        VecType tv5 = (-five * commonCoeff * dy * dz - r2 * dy * dcFd2) * rinv7;
        VecType tv6 = (-five * commonCoeff * dz * dx - r2 * dz * dcFd0) * rinv7;
        VecType tv7 = (-five * commonCoeff * dz * dy - r2 * dz * dcFd1) * rinv7;
        VecType tv8 = (-five * commonCoeff * dz * dz - r2 * dz * dcFd2 + r2 * commonCoeff) * rinv7;

        u[0] += np + tv0 + tv0;
        u[1] += tv1 + tv3;
        u[2] += tv2 + tv6;
        u[3] += tv1 + tv3;
        u[4] += np + tv4 + tv4;
        u[5] += tv5 + tv7;
        u[6] += tv2 + tv6;
        u[7] += tv5 + tv7;
        u[8] += np + tv8 + tv8;
    }
};
} // namespace pvfmm
#endif
