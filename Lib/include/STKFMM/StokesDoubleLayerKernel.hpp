/**
 * @file StokesDoubleLayerKernel.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief Stokes double layer kernels
 * @version 0.1
 * @date 2019-12-23
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef STOKESDOUBLELAYER_HPP_
#define STOKESDOUBLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {


struct stokes_doublepvel_new : public GenericKernel<stokes_doublepvel_new> {
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


struct stokes_doublepvelgrad_new : public GenericKernel<stokes_doublepvelgrad_new> {
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



struct stokes_doublelaplacian_new : public GenericKernel<stokes_doublelaplacian_new> {
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


struct stokes_doubletraction_new : public GenericKernel<stokes_doubletraction_new> {
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

/*********************************************************
 *                                                        *
 *   Stokes Double P Vel kernel, source: 9, target: 4     *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_doublepvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                               Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal);
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);     // vi = 1/8pi (-3rirjrk/r^5) Djk
    const Vec_t facp = set_intrin<Vec_t, Real_t>(FACV * 2); // p = 1/4pi (-3 rjrk/r^5 + delta_jk/r^3) Djk
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t p = zero_intrin<Vec_t>();
            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                // sxx,sxy,sxz,...,szz
                Vec_t sxx = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sxy = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sxz = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t syx = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t syy = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t syz = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t szx = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t szy = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t szz = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv2 = mul_intrin(rinv, rinv);
                Vec_t rinv4 = mul_intrin(rinv2, rinv2);

                Vec_t rinv5 = mul_intrin(rinv, rinv4);

                // commonCoeff = -3 rj rk Djk
                Vec_t commonCoeff = mul_intrin(sxx, mul_intrin(dx, dx));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxy, mul_intrin(dx, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxz, mul_intrin(dx, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syx, mul_intrin(dy, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syy, mul_intrin(dy, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syz, mul_intrin(dy, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szx, mul_intrin(dz, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szy, mul_intrin(dz, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szz, mul_intrin(dz, dz)));
                commonCoeff = mul_intrin(commonCoeff, nthree);

                const Vec_t trace = add_intrin(add_intrin(sxx, syy), szz);
                // p = add_intrin(p, mul_intrin(add_intrin(commonCoeff, mul_intrin(r2, trace)), rinv5));
                p = add_intrin(p, mul_intrin(commonCoeff, rinv5));
                p = add_intrin(p, mul_intrin(mul_intrin(r2, rinv5), trace));
                vx = add_intrin(vx, mul_intrin(rinv5, mul_intrin(dx, commonCoeff)));
                vy = add_intrin(vy, mul_intrin(rinv5, mul_intrin(dy, commonCoeff)));
                vz = add_intrin(vz, mul_intrin(rinv5, mul_intrin(dz, commonCoeff)));
            }

            p = add_intrin(mul_intrin(p, facp), load_intrin<Vec_t>(&trg_value[0][t]));
            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[2][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[3][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], vx);
            store_intrin(&trg_value[2][t], vy);
            store_intrin(&trg_value[3][t], vz);
        }
    }
}

GEN_KERNEL(stokes_doublepvel, stokes_doublepvel_uKernel, 9, 4)

/*********************************************************
 *                                                        *
 * Stokes Double P Vel Grad kernel, source: 9, target: 16 *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_doublepvelgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                   Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV5 = 1 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal);
    const Vec_t facv5 = set_intrin<Vec_t, Real_t>(FACV5);     // vi = 1/8pi (-3rirjrk/r^5) Djk
    const Vec_t facp5 = set_intrin<Vec_t, Real_t>(FACV5 * 2); // p = 1/4pi (-4 rjrk/r^5 + delta_jk) Djk

    const Real_t FACV7 = 3 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal *
                              nwtn_scal * nwtn_scal);
    const Vec_t facv7 = set_intrin<Vec_t, Real_t>(-FACV7);    // -3/8pi
    const Vec_t facp7 = set_intrin<Vec_t, Real_t>(FACV7 * 2); // 3/4pi(5 ... - ...)

    const Vec_t none = set_intrin<Vec_t, Real_t>(-1.0);
    const Vec_t ntwo = set_intrin<Vec_t, Real_t>(-2.0);
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);
    const Vec_t five = set_intrin<Vec_t, Real_t>(5.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t p = zero_intrin<Vec_t>();
            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

            // grad p
            Vec_t pxSum = zero_intrin<Vec_t>();
            Vec_t pySum = zero_intrin<Vec_t>();
            Vec_t pzSum = zero_intrin<Vec_t>();

            // grad v
            Vec_t vxxSum = zero_intrin<Vec_t>();
            Vec_t vxySum = zero_intrin<Vec_t>();
            Vec_t vxzSum = zero_intrin<Vec_t>();

            Vec_t vyxSum = zero_intrin<Vec_t>();
            Vec_t vyySum = zero_intrin<Vec_t>();
            Vec_t vyzSum = zero_intrin<Vec_t>();

            Vec_t vzxSum = zero_intrin<Vec_t>();
            Vec_t vzySum = zero_intrin<Vec_t>();
            Vec_t vzzSum = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                // sxx,sxy,sxz,...,szz
                Vec_t sxx = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sxy = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sxz = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t syx = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t syy = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t syz = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t szx = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t szy = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t szz = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                const Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                const Vec_t rinv2 = mul_intrin(rinv, rinv);
                const Vec_t rinv3 = mul_intrin(rinv, rinv2);
                const Vec_t rinv5 = mul_intrin(rinv3, rinv2);
                const Vec_t rinv7 = mul_intrin(rinv5, rinv2);

                // commonCoeffn3 = -3 rj rk Djk
                // commonCoeff5 = rj rk Djk
                Vec_t commonCoeff = mul_intrin(sxx, mul_intrin(dx, dx));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxy, mul_intrin(dx, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxz, mul_intrin(dx, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syx, mul_intrin(dy, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syy, mul_intrin(dy, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syz, mul_intrin(dy, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szx, mul_intrin(dz, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szy, mul_intrin(dz, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szz, mul_intrin(dz, dz)));
                Vec_t commonCoeff5 = mul_intrin(commonCoeff, five);
                Vec_t commonCoeffn3 = mul_intrin(commonCoeff, nthree);

                Vec_t trace = add_intrin(add_intrin(sxx, syy), szz);
                Vec_t rksxk = add_intrin(mul_intrin(dx, sxx), add_intrin(mul_intrin(dy, sxy), mul_intrin(dz, sxz)));
                Vec_t rksyk = add_intrin(mul_intrin(dx, syx), add_intrin(mul_intrin(dy, syy), mul_intrin(dz, syz)));
                Vec_t rkszk = add_intrin(mul_intrin(dx, szx), add_intrin(mul_intrin(dy, szy), mul_intrin(dz, szz)));

                Vec_t rkskx = add_intrin(mul_intrin(dx, sxx), add_intrin(mul_intrin(dy, syx), mul_intrin(dz, szx)));
                Vec_t rksky = add_intrin(mul_intrin(dx, sxy), add_intrin(mul_intrin(dy, syy), mul_intrin(dz, szy)));
                Vec_t rkskz = add_intrin(mul_intrin(dx, sxz), add_intrin(mul_intrin(dy, syz), mul_intrin(dz, szz)));

                p = add_intrin(p, mul_intrin(add_intrin(commonCoeffn3, mul_intrin(r2, trace)), rinv5));
                vx = add_intrin(vx, mul_intrin(rinv5, mul_intrin(dx, commonCoeffn3)));
                vy = add_intrin(vy, mul_intrin(rinv5, mul_intrin(dy, commonCoeffn3)));
                vz = add_intrin(vz, mul_intrin(rinv5, mul_intrin(dz, commonCoeffn3)));

                // pgrad
                Vec_t px = sub_intrin(mul_intrin(dx, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rksxk, rkskx), mul_intrin(dx, trace))));
                Vec_t py = sub_intrin(mul_intrin(dy, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rksyk, rksky), mul_intrin(dy, trace))));
                Vec_t pz = sub_intrin(mul_intrin(dz, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rkszk, rkskz), mul_intrin(dz, trace))));

                pxSum = add_intrin(pxSum, mul_intrin(px, rinv7));
                pySum = add_intrin(pySum, mul_intrin(py, rinv7));
                pzSum = add_intrin(pzSum, mul_intrin(pz, rinv7));

                // vgrad
                Vec_t commonCoeffn1 = mul_intrin(none, commonCoeff);
                // (-2*(t0 - s0)*sv0 - (t1 - s1)*(sv1 + sv3) - (t2 - s2)*(sv2 + sv6))
                Vec_t dcFd0 = mul_intrin(ntwo, mul_intrin(dx, sxx));
                dcFd0 = add_intrin(dcFd0, mul_intrin(none, mul_intrin(dy, add_intrin(sxy, syx))));
                dcFd0 = add_intrin(dcFd0, mul_intrin(none, mul_intrin(dz, add_intrin(sxz, szx))));

                // (-2*(t1 - s1)*sv4 - (t0 - s0)*(sv1 + sv3) - (t2 - s2)*(sv5 + sv7))
                Vec_t dcFd1 = mul_intrin(ntwo, mul_intrin(dy, syy));
                dcFd1 = add_intrin(dcFd1, mul_intrin(none, mul_intrin(dx, add_intrin(sxy, syx))));
                dcFd1 = add_intrin(dcFd1, mul_intrin(none, mul_intrin(dz, add_intrin(syz, szy))));

                // (-2*(t2 - s2)*sv8 - (t0 - s0)*(sv2 + sv6) - (t1 - s1)*(sv5 + sv7))
                Vec_t dcFd2 = mul_intrin(ntwo, mul_intrin(dz, szz));
                dcFd2 = add_intrin(dcFd2, mul_intrin(none, mul_intrin(dx, add_intrin(sxz, szx))));
                dcFd2 = add_intrin(dcFd2, mul_intrin(none, mul_intrin(dy, add_intrin(syz, szy))));

                Vec_t tv0 = zero_intrin<Vec_t>();
                Vec_t tv1 = zero_intrin<Vec_t>();
                Vec_t tv2 = zero_intrin<Vec_t>();
                Vec_t tv3 = zero_intrin<Vec_t>();
                Vec_t tv4 = zero_intrin<Vec_t>();
                Vec_t tv5 = zero_intrin<Vec_t>();
                Vec_t tv6 = zero_intrin<Vec_t>();
                Vec_t tv7 = zero_intrin<Vec_t>();
                Vec_t tv8 = zero_intrin<Vec_t>();

                // (5 * rrtensor * commonCoeff / rnorm ^ 7
                //  - r^2 Outer[Times, rvec, {dcFd0, dcFd1, dcFd2}] / rnorm ^7
                //  - r^2 IdentityMatrix[3] * commonCoeff / rnorm ^ 7);
                tv0 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dx, dx)));
                tv0 = add_intrin(tv0, mul_intrin(none, mul_intrin(r2, mul_intrin(dx, dcFd0))));
                tv0 = add_intrin(tv0, mul_intrin(none, mul_intrin(r2, commonCoeffn1)));
                tv0 = mul_intrin(tv0, rinv7);
                tv1 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dx, dy)));
                tv1 = add_intrin(tv1, mul_intrin(none, mul_intrin(r2, mul_intrin(dx, dcFd1))));
                tv1 = mul_intrin(tv1, rinv7);
                tv2 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dx, dz)));
                tv2 = add_intrin(tv2, mul_intrin(none, mul_intrin(r2, mul_intrin(dx, dcFd2))));
                tv2 = mul_intrin(tv2, rinv7);

                tv3 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dy, dx)));
                tv3 = add_intrin(tv3, mul_intrin(none, mul_intrin(r2, mul_intrin(dy, dcFd0))));
                tv3 = mul_intrin(tv3, rinv7);
                tv4 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dy, dy)));
                tv4 = add_intrin(tv4, mul_intrin(none, mul_intrin(r2, mul_intrin(dy, dcFd1))));
                tv4 = add_intrin(tv4, mul_intrin(none, mul_intrin(r2, commonCoeffn1)));
                tv4 = mul_intrin(tv4, rinv7);
                tv5 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dy, dz)));
                tv5 = add_intrin(tv5, mul_intrin(none, mul_intrin(r2, mul_intrin(dy, dcFd2))));
                tv5 = mul_intrin(tv5, rinv7);

                tv6 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dz, dx)));
                tv6 = add_intrin(tv6, mul_intrin(none, mul_intrin(r2, mul_intrin(dz, dcFd0))));
                tv6 = mul_intrin(tv6, rinv7);
                tv7 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dz, dy)));
                tv7 = add_intrin(tv7, mul_intrin(none, mul_intrin(r2, mul_intrin(dz, dcFd1))));
                tv7 = mul_intrin(tv7, rinv7);
                tv8 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dz, dz)));
                tv8 = add_intrin(tv8, mul_intrin(none, mul_intrin(r2, mul_intrin(dz, dcFd2))));
                tv8 = add_intrin(tv8, mul_intrin(none, mul_intrin(r2, commonCoeffn1)));
                tv8 = mul_intrin(tv8, rinv7);

                vxxSum = add_intrin(vxxSum, tv0);
                vxySum = add_intrin(vxySum, tv1);
                vxzSum = add_intrin(vxzSum, tv2);

                vyxSum = add_intrin(vyxSum, tv3);
                vyySum = add_intrin(vyySum, tv4);
                vyzSum = add_intrin(vyzSum, tv5);

                vzxSum = add_intrin(vzxSum, tv6);
                vzySum = add_intrin(vzySum, tv7);
                vzzSum = add_intrin(vzzSum, tv8);
            }

            p = add_intrin(mul_intrin(p, facp5), load_intrin<Vec_t>(&trg_value[0][t]));
            vx = add_intrin(mul_intrin(vx, facv5), load_intrin<Vec_t>(&trg_value[1][t]));
            vy = add_intrin(mul_intrin(vy, facv5), load_intrin<Vec_t>(&trg_value[2][t]));
            vz = add_intrin(mul_intrin(vz, facv5), load_intrin<Vec_t>(&trg_value[3][t]));

            pxSum = add_intrin(mul_intrin(pxSum, facp7), load_intrin<Vec_t>(&trg_value[4][t]));
            pySum = add_intrin(mul_intrin(pySum, facp7), load_intrin<Vec_t>(&trg_value[5][t]));
            pzSum = add_intrin(mul_intrin(pzSum, facp7), load_intrin<Vec_t>(&trg_value[6][t]));

            vxxSum = add_intrin(mul_intrin(vxxSum, facv7), load_intrin<Vec_t>(&trg_value[7][t]));
            vxySum = add_intrin(mul_intrin(vxySum, facv7), load_intrin<Vec_t>(&trg_value[8][t]));
            vxzSum = add_intrin(mul_intrin(vxzSum, facv7), load_intrin<Vec_t>(&trg_value[9][t]));

            vyxSum = add_intrin(mul_intrin(vyxSum, facv7), load_intrin<Vec_t>(&trg_value[10][t]));
            vyySum = add_intrin(mul_intrin(vyySum, facv7), load_intrin<Vec_t>(&trg_value[11][t]));
            vyzSum = add_intrin(mul_intrin(vyzSum, facv7), load_intrin<Vec_t>(&trg_value[12][t]));

            vzxSum = add_intrin(mul_intrin(vzxSum, facv7), load_intrin<Vec_t>(&trg_value[13][t]));
            vzySum = add_intrin(mul_intrin(vzySum, facv7), load_intrin<Vec_t>(&trg_value[14][t]));
            vzzSum = add_intrin(mul_intrin(vzzSum, facv7), load_intrin<Vec_t>(&trg_value[15][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], vx);
            store_intrin(&trg_value[2][t], vy);
            store_intrin(&trg_value[3][t], vz);

            store_intrin(&trg_value[4][t], pxSum);
            store_intrin(&trg_value[5][t], pySum);
            store_intrin(&trg_value[6][t], pzSum);

            store_intrin(&trg_value[7][t], vxxSum);
            store_intrin(&trg_value[8][t], vxySum);
            store_intrin(&trg_value[9][t], vxzSum);

            store_intrin(&trg_value[10][t], vyxSum);
            store_intrin(&trg_value[11][t], vyySum);
            store_intrin(&trg_value[12][t], vyzSum);

            store_intrin(&trg_value[13][t], vzxSum);
            store_intrin(&trg_value[14][t], vzySum);
            store_intrin(&trg_value[15][t], vzzSum);
        }
    }
}

GEN_KERNEL(stokes_doublepvelgrad, stokes_doublepvelgrad_uKernel, 9, 16)

/*********************************************************
 *                                                        *
 *  Stokes Double P Vel Lap kernel, source: 9, target: 7 *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_doublelaplacian_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                    Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV5 = 1 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal);
    const Vec_t facv5 = set_intrin<Vec_t, Real_t>(FACV5);     // vi = 1/8pi (-3rirjrk/r^5) Djk
    const Vec_t facp5 = set_intrin<Vec_t, Real_t>(FACV5 * 2); // p = 1/4pi (-4 rjrk/r^5 + delta_jk) Djk

    const Real_t FACV7 = 3 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal *
                              nwtn_scal * nwtn_scal);
    const Vec_t facv7 = set_intrin<Vec_t, Real_t>(-FACV7);    // -3/8pi
    const Vec_t facp7 = set_intrin<Vec_t, Real_t>(FACV7 * 2); // 3/4pi(5 ... - ...)

    const Vec_t none = set_intrin<Vec_t, Real_t>(-1.0);
    const Vec_t ntwo = set_intrin<Vec_t, Real_t>(-2.0);
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);
    const Vec_t five = set_intrin<Vec_t, Real_t>(5.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t p = zero_intrin<Vec_t>();
            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

            // grad p
            Vec_t pxSum = zero_intrin<Vec_t>();
            Vec_t pySum = zero_intrin<Vec_t>();
            Vec_t pzSum = zero_intrin<Vec_t>();

            // grad v
            Vec_t vxxSum = zero_intrin<Vec_t>();
            Vec_t vxySum = zero_intrin<Vec_t>();
            Vec_t vxzSum = zero_intrin<Vec_t>();

            Vec_t vyxSum = zero_intrin<Vec_t>();
            Vec_t vyySum = zero_intrin<Vec_t>();
            Vec_t vyzSum = zero_intrin<Vec_t>();

            Vec_t vzxSum = zero_intrin<Vec_t>();
            Vec_t vzySum = zero_intrin<Vec_t>();
            Vec_t vzzSum = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                // sxx,sxy,sxz,...,szz
                Vec_t sxx = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sxy = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sxz = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t syx = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t syy = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t syz = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t szx = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t szy = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t szz = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                const Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                const Vec_t rinv2 = mul_intrin(rinv, rinv);
                const Vec_t rinv3 = mul_intrin(rinv, rinv2);
                const Vec_t rinv5 = mul_intrin(rinv3, rinv2);
                const Vec_t rinv7 = mul_intrin(rinv5, rinv2);

                // commonCoeffn3 = -3 rj rk Djk
                // commonCoeff5 = rj rk Djk
                Vec_t commonCoeff = mul_intrin(sxx, mul_intrin(dx, dx));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxy, mul_intrin(dx, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxz, mul_intrin(dx, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syx, mul_intrin(dy, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syy, mul_intrin(dy, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syz, mul_intrin(dy, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szx, mul_intrin(dz, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szy, mul_intrin(dz, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szz, mul_intrin(dz, dz)));
                Vec_t commonCoeff5 = mul_intrin(commonCoeff, five);
                Vec_t commonCoeffn3 = mul_intrin(commonCoeff, nthree);

                Vec_t trace = add_intrin(add_intrin(sxx, syy), szz);
                Vec_t rksxk = add_intrin(mul_intrin(dx, sxx), add_intrin(mul_intrin(dy, sxy), mul_intrin(dz, sxz)));
                Vec_t rksyk = add_intrin(mul_intrin(dx, syx), add_intrin(mul_intrin(dy, syy), mul_intrin(dz, syz)));
                Vec_t rkszk = add_intrin(mul_intrin(dx, szx), add_intrin(mul_intrin(dy, szy), mul_intrin(dz, szz)));

                Vec_t rkskx = add_intrin(mul_intrin(dx, sxx), add_intrin(mul_intrin(dy, syx), mul_intrin(dz, szx)));
                Vec_t rksky = add_intrin(mul_intrin(dx, sxy), add_intrin(mul_intrin(dy, syy), mul_intrin(dz, szy)));
                Vec_t rkskz = add_intrin(mul_intrin(dx, sxz), add_intrin(mul_intrin(dy, syz), mul_intrin(dz, szz)));

                p = add_intrin(p, mul_intrin(add_intrin(commonCoeffn3, mul_intrin(r2, trace)), rinv5));
                vx = add_intrin(vx, mul_intrin(rinv5, mul_intrin(dx, commonCoeffn3)));
                vy = add_intrin(vy, mul_intrin(rinv5, mul_intrin(dy, commonCoeffn3)));
                vz = add_intrin(vz, mul_intrin(rinv5, mul_intrin(dz, commonCoeffn3)));

                // lap v = pgrad
                Vec_t px = sub_intrin(mul_intrin(dx, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rksxk, rkskx), mul_intrin(dx, trace))));
                Vec_t py = sub_intrin(mul_intrin(dy, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rksyk, rksky), mul_intrin(dy, trace))));
                Vec_t pz = sub_intrin(mul_intrin(dz, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rkszk, rkskz), mul_intrin(dz, trace))));

                pxSum = add_intrin(pxSum, mul_intrin(px, rinv7));
                pySum = add_intrin(pySum, mul_intrin(py, rinv7));
                pzSum = add_intrin(pzSum, mul_intrin(pz, rinv7));
            }

            p = add_intrin(mul_intrin(p, facp5), load_intrin<Vec_t>(&trg_value[0][t]));
            vx = add_intrin(mul_intrin(vx, facv5), load_intrin<Vec_t>(&trg_value[1][t]));
            vy = add_intrin(mul_intrin(vy, facv5), load_intrin<Vec_t>(&trg_value[2][t]));
            vz = add_intrin(mul_intrin(vz, facv5), load_intrin<Vec_t>(&trg_value[3][t]));

            pxSum = add_intrin(mul_intrin(pxSum, facp7), load_intrin<Vec_t>(&trg_value[4][t]));
            pySum = add_intrin(mul_intrin(pySum, facp7), load_intrin<Vec_t>(&trg_value[5][t]));
            pzSum = add_intrin(mul_intrin(pzSum, facp7), load_intrin<Vec_t>(&trg_value[6][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], vx);
            store_intrin(&trg_value[2][t], vy);
            store_intrin(&trg_value[3][t], vz);

            store_intrin(&trg_value[4][t], pxSum);
            store_intrin(&trg_value[5][t], pySum);
            store_intrin(&trg_value[6][t], pzSum);
        }
    }
}

GEN_KERNEL(stokes_doublelaplacian, stokes_doublelaplacian_uKernel, 9, 7)

/*********************************************************
 *                                                        *
 *   Stokes Double Traction kernel, source: 9, target: 9  *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_doubletraction_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                   Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }

    const Real_t FACV7 = 3 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal *
                              nwtn_scal * nwtn_scal);
    const Vec_t facv7 = set_intrin<Vec_t, Real_t>(-FACV7); // -3/8pi

    const Vec_t facp = set_intrin<Vec_t, Real_t>((Real_t)(2.0 / 3.0));
    const Vec_t none = set_intrin<Vec_t, Real_t>(-1.0);
    const Vec_t ntwo = set_intrin<Vec_t, Real_t>(-2.0);
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);
    const Vec_t five = set_intrin<Vec_t, Real_t>(5.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            // traction
            Vec_t txxSum = zero_intrin<Vec_t>();
            Vec_t txySum = zero_intrin<Vec_t>();
            Vec_t txzSum = zero_intrin<Vec_t>();

            Vec_t tyxSum = zero_intrin<Vec_t>();
            Vec_t tyySum = zero_intrin<Vec_t>();
            Vec_t tyzSum = zero_intrin<Vec_t>();

            Vec_t tzxSum = zero_intrin<Vec_t>();
            Vec_t tzySum = zero_intrin<Vec_t>();
            Vec_t tzzSum = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                // sxx,sxy,sxz,...,szz
                Vec_t sxx = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sxy = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sxz = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t syx = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t syy = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t syz = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t szx = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t szy = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t szz = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                const Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                const Vec_t rinv2 = mul_intrin(rinv, rinv);
                const Vec_t rinv3 = mul_intrin(rinv, rinv2);
                const Vec_t rinv5 = mul_intrin(rinv3, rinv2);
                const Vec_t rinv7 = mul_intrin(rinv5, rinv2);

                Vec_t np = zero_intrin<Vec_t>();

                // commonCoeffn3 = -3 rj rk Djk
                // commonCoeff5 = 5 rj rk Djk
                Vec_t commonCoeff = mul_intrin(sxx, mul_intrin(dx, dx));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxy, mul_intrin(dx, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sxz, mul_intrin(dx, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syx, mul_intrin(dy, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syy, mul_intrin(dy, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(syz, mul_intrin(dy, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szx, mul_intrin(dz, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szy, mul_intrin(dz, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(szz, mul_intrin(dz, dz)));
                Vec_t commonCoeff5 = mul_intrin(commonCoeff, five);
                Vec_t commonCoeffn3 = mul_intrin(commonCoeff, nthree);

                Vec_t trace = add_intrin(add_intrin(sxx, syy), szz);

                // np =
                // mul_intrin(mul_intrin(mul_intrin(add_intrin(commonCoeffn3,
                // mul_intrin(r2, trace)), rinv7), r2), none);
                np = mul_intrin(mul_intrin(mul_intrin(add_intrin(commonCoeffn3, mul_intrin(r2, trace)), rinv7), r2),
                                facp);

                // vgrad
                Vec_t commonCoeffn1 = mul_intrin(none, commonCoeff);
                // (-2*(t0 - s0)*sv0 - (t1 - s1)*(sv1 + sv3) - (t2 - s2)*(sv2 + sv6))
                Vec_t dcFd0 = mul_intrin(ntwo, mul_intrin(dx, sxx));
                dcFd0 = add_intrin(dcFd0, mul_intrin(none, mul_intrin(dy, add_intrin(sxy, syx))));
                dcFd0 = add_intrin(dcFd0, mul_intrin(none, mul_intrin(dz, add_intrin(sxz, szx))));

                // (-2*(t1 - s1)*sv4 - (t0 - s0)*(sv1 + sv3) - (t2 - s2)*(sv5 + sv7))
                Vec_t dcFd1 = mul_intrin(ntwo, mul_intrin(dy, syy));
                dcFd1 = add_intrin(dcFd1, mul_intrin(none, mul_intrin(dx, add_intrin(sxy, syx))));
                dcFd1 = add_intrin(dcFd1, mul_intrin(none, mul_intrin(dz, add_intrin(syz, szy))));

                // (-2*(t2 - s2)*sv8 - (t0 - s0)*(sv2 + sv6) - (t1 - s1)*(sv5 + sv7))
                Vec_t dcFd2 = mul_intrin(ntwo, mul_intrin(dz, szz));
                dcFd2 = add_intrin(dcFd2, mul_intrin(none, mul_intrin(dx, add_intrin(sxz, szx))));
                dcFd2 = add_intrin(dcFd2, mul_intrin(none, mul_intrin(dy, add_intrin(syz, szy))));

                // grad u
                Vec_t tv0 = zero_intrin<Vec_t>();
                Vec_t tv1 = zero_intrin<Vec_t>();
                Vec_t tv2 = zero_intrin<Vec_t>();
                Vec_t tv3 = zero_intrin<Vec_t>();
                Vec_t tv4 = zero_intrin<Vec_t>();
                Vec_t tv5 = zero_intrin<Vec_t>();
                Vec_t tv6 = zero_intrin<Vec_t>();
                Vec_t tv7 = zero_intrin<Vec_t>();
                Vec_t tv8 = zero_intrin<Vec_t>();

                // (5 * rrtensor * commonCoeff / rnorm ^ 7
                //  - r^2 Outer[Times, rvec, {dcFd0, dcFd1, dcFd2}] / rnorm ^7
                //  - r^2 IdentityMatrix[3] * commonCoeff / rnorm ^ 7);
                tv0 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dx, dx)));
                tv0 = add_intrin(tv0, mul_intrin(none, mul_intrin(r2, mul_intrin(dx, dcFd0))));
                tv0 = add_intrin(tv0, mul_intrin(none, mul_intrin(r2, commonCoeffn1)));
                tv0 = mul_intrin(tv0, rinv7);
                tv1 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dx, dy)));
                tv1 = add_intrin(tv1, mul_intrin(none, mul_intrin(r2, mul_intrin(dx, dcFd1))));
                tv1 = mul_intrin(tv1, rinv7);
                tv2 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dx, dz)));
                tv2 = add_intrin(tv2, mul_intrin(none, mul_intrin(r2, mul_intrin(dx, dcFd2))));
                tv2 = mul_intrin(tv2, rinv7);

                tv3 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dy, dx)));
                tv3 = add_intrin(tv3, mul_intrin(none, mul_intrin(r2, mul_intrin(dy, dcFd0))));
                tv3 = mul_intrin(tv3, rinv7);
                tv4 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dy, dy)));
                tv4 = add_intrin(tv4, mul_intrin(none, mul_intrin(r2, mul_intrin(dy, dcFd1))));
                tv4 = add_intrin(tv4, mul_intrin(none, mul_intrin(r2, commonCoeffn1)));
                tv4 = mul_intrin(tv4, rinv7);
                tv5 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dy, dz)));
                tv5 = add_intrin(tv5, mul_intrin(none, mul_intrin(r2, mul_intrin(dy, dcFd2))));
                tv5 = mul_intrin(tv5, rinv7);

                tv6 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dz, dx)));
                tv6 = add_intrin(tv6, mul_intrin(none, mul_intrin(r2, mul_intrin(dz, dcFd0))));
                tv6 = mul_intrin(tv6, rinv7);
                tv7 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dz, dy)));
                tv7 = add_intrin(tv7, mul_intrin(none, mul_intrin(r2, mul_intrin(dz, dcFd1))));
                tv7 = mul_intrin(tv7, rinv7);
                tv8 = mul_intrin(five, mul_intrin(commonCoeffn1, mul_intrin(dz, dz)));
                tv8 = add_intrin(tv8, mul_intrin(none, mul_intrin(r2, mul_intrin(dz, dcFd2))));
                tv8 = add_intrin(tv8, mul_intrin(none, mul_intrin(r2, commonCoeffn1)));
                tv8 = mul_intrin(tv8, rinv7);

                // traction = -p \delta_{ij} + d u_i / d x_j + d u_j / d x_i
                txxSum = add_intrin(txxSum, add_intrin(np, add_intrin(tv0, tv0)));
                txySum = add_intrin(txySum, add_intrin(tv1, tv3));
                txzSum = add_intrin(txzSum, add_intrin(tv2, tv6));

                tyxSum = add_intrin(tyxSum, add_intrin(tv1, tv3));
                tyySum = add_intrin(tyySum, add_intrin(np, add_intrin(tv4, tv4)));
                tyzSum = add_intrin(tyzSum, add_intrin(tv5, tv7));

                tzxSum = add_intrin(tzxSum, add_intrin(tv2, tv6));
                tzySum = add_intrin(tzySum, add_intrin(tv5, tv7));
                tzzSum = add_intrin(tzzSum, add_intrin(np, add_intrin(tv8, tv8)));
            }

            txxSum = add_intrin(mul_intrin(txxSum, facv7), load_intrin<Vec_t>(&trg_value[0][t]));
            txySum = add_intrin(mul_intrin(txySum, facv7), load_intrin<Vec_t>(&trg_value[1][t]));
            txzSum = add_intrin(mul_intrin(txzSum, facv7), load_intrin<Vec_t>(&trg_value[2][t]));

            tyxSum = add_intrin(mul_intrin(tyxSum, facv7), load_intrin<Vec_t>(&trg_value[3][t]));
            tyySum = add_intrin(mul_intrin(tyySum, facv7), load_intrin<Vec_t>(&trg_value[4][t]));
            tyzSum = add_intrin(mul_intrin(tyzSum, facv7), load_intrin<Vec_t>(&trg_value[5][t]));

            tzxSum = add_intrin(mul_intrin(tzxSum, facv7), load_intrin<Vec_t>(&trg_value[6][t]));
            tzySum = add_intrin(mul_intrin(tzySum, facv7), load_intrin<Vec_t>(&trg_value[7][t]));
            tzzSum = add_intrin(mul_intrin(tzzSum, facv7), load_intrin<Vec_t>(&trg_value[8][t]));

            store_intrin(&trg_value[0][t], txxSum);
            store_intrin(&trg_value[1][t], txySum);
            store_intrin(&trg_value[2][t], txzSum);

            store_intrin(&trg_value[3][t], tyxSum);
            store_intrin(&trg_value[4][t], tyySum);
            store_intrin(&trg_value[5][t], tyzSum);

            store_intrin(&trg_value[6][t], tzxSum);
            store_intrin(&trg_value[7][t], tzySum);
            store_intrin(&trg_value[8][t], tzzSum);
        }
    }
}

GEN_KERNEL(stokes_doubletraction, stokes_doubletraction_uKernel, 9, 9)

} // namespace pvfmm
#endif
