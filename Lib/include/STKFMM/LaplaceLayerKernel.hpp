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
#include <STKFMM/stkfmm_helpers.hpp>

/**
 * @brief insert kernel functions to pvfmm namespace
 *
 */
namespace pvfmm {

/**
 * @brief micro kernel for Laplace single layer potential + gradient
 *
 * @tparam Real_t
 * @tparam Real_t
 * @tparam rsqrt_intrin0<Vec_t>
 * @param src_coord
 * @param src_value
 * @param trg_coord
 * @param trg_value
 */
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER = 0>
void laplace_pgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                           Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOFP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    Vec_t oofp = set_intrin<Vec_t, Real_t>(OOFP);
    Vec_t noofp = set_intrin<Vec_t, Real_t>(-OOFP);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t tp = zero_intrin<Vec_t>();
            Vec_t tv0 = zero_intrin<Vec_t>();
            Vec_t tv1 = zero_intrin<Vec_t>();
            Vec_t tv2 = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));
                Vec_t sv = bcast_intrin<Vec_t>(&src_value[0][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t r3inv = mul_intrin(mul_intrin(rinv, rinv), rinv);

                sv = mul_intrin(sv, r3inv);
                tp = add_intrin(tp, mul_intrin(sv, r2));
                tv0 = add_intrin(tv0, mul_intrin(sv, dx));
                tv1 = add_intrin(tv1, mul_intrin(sv, dy));
                tv2 = add_intrin(tv2, mul_intrin(sv, dz));
            }
            tp = add_intrin(mul_intrin(tp, oofp),
                            load_intrin<Vec_t>(&trg_value[0][t])); // potential
            tv0 = add_intrin(mul_intrin(tv0, noofp),
                             load_intrin<Vec_t>(&trg_value[1][t])); // gradient
            tv1 = add_intrin(mul_intrin(tv1, noofp), load_intrin<Vec_t>(&trg_value[2][t]));
            tv2 = add_intrin(mul_intrin(tv2, noofp), load_intrin<Vec_t>(&trg_value[3][t]));
            store_intrin(&trg_value[0][t], tp);
            store_intrin(&trg_value[1][t], tv0);
            store_intrin(&trg_value[2][t], tv1);
            store_intrin(&trg_value[3][t], tv2);
        }
    }
#undef SRC_BLK
}

GEN_KERNEL(laplace_pgrad, laplace_pgrad_uKernel, 1, 4)

/**
 * @brief micro kernel for Laplace double layer (dipole) potential
 *
 * @tparam Real_t
 * @tparam Real_t
 * @tparam rsqrt_intrin0<Vec_t>
 * @param src_coord
 * @param src_value
 * @param trg_coord
 * @param trg_value
 */
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER = 0>
void laplace_dipolep_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                             Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOFP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    Vec_t oofp = set_intrin<Vec_t, Real_t>(OOFP);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);
            Vec_t tv = zero_intrin<Vec_t>();
            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));
                Vec_t sn0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sn1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sn2 = bcast_intrin<Vec_t>(&src_value[2][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t r3inv = mul_intrin(mul_intrin(rinv, rinv), rinv);

                Vec_t rdotn = mul_intrin(sn0, dx);
                rdotn = add_intrin(rdotn, mul_intrin(sn1, dy));
                rdotn = add_intrin(rdotn, mul_intrin(sn2, dz));

                tv = add_intrin(tv, mul_intrin(r3inv, rdotn));
            }
            tv = add_intrin(mul_intrin(tv, oofp), load_intrin<Vec_t>(&trg_value[0][t]));
            store_intrin(&trg_value[0][t], tv);
        }
    }
#undef SRC_BLK
}

GEN_KERNEL(laplace_dipolep, laplace_dipolep_uKernel, 3, 1)

/**
 * @brief micro kernel for Laplace double layer (dipole) potential + gradient
 * from dim 3 to dim 1 + 3
 *
 * @tparam Real_t
 * @tparam Real_t
 * @tparam rsqrt_intrin0<Vec_t>
 * @param src_coord
 * @param src_value
 * @param trg_coord
 * @param trg_value
 */
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void laplace_dipolepgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                 Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOFP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Real_t OOFP2 = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    Vec_t oofp = set_intrin<Vec_t, Real_t>(OOFP);
    Vec_t oofp2 = set_intrin<Vec_t, Real_t>(OOFP2);
    Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);
            Vec_t tg0 = zero_intrin<Vec_t>();
            Vec_t tg1 = zero_intrin<Vec_t>();
            Vec_t tg2 = zero_intrin<Vec_t>();
            Vec_t tv = zero_intrin<Vec_t>();
            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));
                Vec_t s0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t s1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t s2 = bcast_intrin<Vec_t>(&src_value[2][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t r3inv = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t r5inv = mul_intrin(mul_intrin(rinv, rinv), r3inv);

                Vec_t rdotn = mul_intrin(s0, dx);
                rdotn = add_intrin(rdotn, mul_intrin(s1, dy));
                rdotn = add_intrin(rdotn, mul_intrin(s2, dz));

                tv = add_intrin(tv, mul_intrin(rdotn, r3inv));
                tg0 = add_intrin(tg0, mul_intrin(mul_intrin(s0, r2), r5inv));
                tg0 = add_intrin(tg0, mul_intrin(mul_intrin(mul_intrin(rdotn, nthree), r5inv), dx));
                tg1 = add_intrin(tg1, mul_intrin(mul_intrin(s1, r2), r5inv));
                tg1 = add_intrin(tg1, mul_intrin(mul_intrin(mul_intrin(rdotn, nthree), r5inv), dy));
                tg2 = add_intrin(tg2, mul_intrin(mul_intrin(s2, r2), r5inv));
                tg2 = add_intrin(tg2, mul_intrin(mul_intrin(mul_intrin(rdotn, nthree), r5inv), dz));
            }
            tv = add_intrin(mul_intrin(tv, oofp), load_intrin<Vec_t>(&trg_value[0][t]));
            tg0 = add_intrin(mul_intrin(tg0, oofp2), load_intrin<Vec_t>(&trg_value[1][t]));
            tg1 = add_intrin(mul_intrin(tg1, oofp2), load_intrin<Vec_t>(&trg_value[2][t]));
            tg2 = add_intrin(mul_intrin(tg2, oofp2), load_intrin<Vec_t>(&trg_value[3][t]));
            store_intrin(&trg_value[0][t], tv);
            store_intrin(&trg_value[1][t], tg0);
            store_intrin(&trg_value[2][t], tg1);
            store_intrin(&trg_value[3][t], tg2);
        }
    }
#undef SRC_BLK
}

GEN_KERNEL(laplace_dipolepgrad, laplace_dipolepgrad_uKernel, 3, 4)

template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER = 0>
void laplace_pgradgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                               Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOFP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Real_t OOFP5 = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    Vec_t oofp = set_intrin<Vec_t, Real_t>(OOFP);
    Vec_t noofp = set_intrin<Vec_t, Real_t>(-OOFP);
    Vec_t oofp5 = set_intrin<Vec_t, Real_t>(OOFP5);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t tp = zero_intrin<Vec_t>();
            Vec_t tv0 = zero_intrin<Vec_t>();
            Vec_t tv1 = zero_intrin<Vec_t>();
            Vec_t tv2 = zero_intrin<Vec_t>();
            Vec_t gxx = zero_intrin<Vec_t>();
            Vec_t gxy = zero_intrin<Vec_t>();
            Vec_t gxz = zero_intrin<Vec_t>();
            Vec_t gyy = zero_intrin<Vec_t>();
            Vec_t gyz = zero_intrin<Vec_t>();
            Vec_t gzz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));
                Vec_t sv = bcast_intrin<Vec_t>(&src_value[0][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t r3inv = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t r2inv = mul_intrin(rinv, rinv);

                sv = mul_intrin(sv, r3inv);
                tp = add_intrin(tp, mul_intrin(sv, r2));
                tv0 = add_intrin(tv0, mul_intrin(sv, dx));
                tv1 = add_intrin(tv1, mul_intrin(sv, dy));
                tv2 = add_intrin(tv2, mul_intrin(sv, dz));
                gxx = add_intrin(
                    gxx, mul_intrin(sv, mul_intrin(r2inv, sub_intrin(mul_intrin(three, mul_intrin(dx, dx)), r2))));
                gyy = add_intrin(
                    gyy, mul_intrin(sv, mul_intrin(r2inv, sub_intrin(mul_intrin(three, mul_intrin(dy, dy)), r2))));
                gzz = add_intrin(
                    gzz, mul_intrin(sv, mul_intrin(r2inv, sub_intrin(mul_intrin(three, mul_intrin(dz, dz)), r2))));
                gxy = add_intrin(gxy, mul_intrin(sv, mul_intrin(r2inv, mul_intrin(three, mul_intrin(dx, dy)))));
                gxz = add_intrin(gxz, mul_intrin(sv, mul_intrin(r2inv, mul_intrin(three, mul_intrin(dx, dz)))));
                gyz = add_intrin(gyz, mul_intrin(sv, mul_intrin(r2inv, mul_intrin(three, mul_intrin(dy, dz)))));
            }
            // potential
            tp = add_intrin(mul_intrin(tp, oofp), load_intrin<Vec_t>(&trg_value[0][t]));
            // gradient
            tv0 = add_intrin(mul_intrin(tv0, noofp), load_intrin<Vec_t>(&trg_value[1][t]));
            tv1 = add_intrin(mul_intrin(tv1, noofp), load_intrin<Vec_t>(&trg_value[2][t]));
            tv2 = add_intrin(mul_intrin(tv2, noofp), load_intrin<Vec_t>(&trg_value[3][t]));
            // grad grad, symmetric
            gxx = add_intrin(mul_intrin(gxx, oofp5), load_intrin<Vec_t>(&trg_value[4][t]));
            gxy = add_intrin(mul_intrin(gxy, oofp5), load_intrin<Vec_t>(&trg_value[5][t]));
            gxz = add_intrin(mul_intrin(gxz, oofp5), load_intrin<Vec_t>(&trg_value[6][t]));
            gyy = add_intrin(mul_intrin(gyy, oofp5), load_intrin<Vec_t>(&trg_value[7][t]));
            gyz = add_intrin(mul_intrin(gyz, oofp5), load_intrin<Vec_t>(&trg_value[8][t]));
            gzz = add_intrin(mul_intrin(gzz, oofp5), load_intrin<Vec_t>(&trg_value[9][t]));
            store_intrin(&trg_value[0][t], tp);
            store_intrin(&trg_value[1][t], tv0);
            store_intrin(&trg_value[2][t], tv1);
            store_intrin(&trg_value[3][t], tv2);
            store_intrin(&trg_value[4][t], gxx);
            store_intrin(&trg_value[5][t], gxy);
            store_intrin(&trg_value[6][t], gxz);
            store_intrin(&trg_value[7][t], gyy);
            store_intrin(&trg_value[8][t], gyz);
            store_intrin(&trg_value[9][t], gzz);
        }
    }
#undef SRC_BLK
}

GEN_KERNEL(laplace_pgradgrad, laplace_pgradgrad_uKernel, 1, 10)

template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void laplace_dipolepgradgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                     Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOFP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Real_t OOFP5 = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Real_t OOFP7 = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal *
                                const_pi<Real_t>());
    Vec_t oofp = set_intrin<Vec_t, Real_t>(OOFP);
    Vec_t oofp5 = set_intrin<Vec_t, Real_t>(OOFP5);
    Vec_t oofp7 = set_intrin<Vec_t, Real_t>(OOFP7);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    Vec_t six = set_intrin<Vec_t, Real_t>(6.0);
    Vec_t nine = set_intrin<Vec_t, Real_t>(9.0);
    Vec_t fifteen = set_intrin<Vec_t, Real_t>(15.0);
    Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);
            Vec_t tv = zero_intrin<Vec_t>();
            Vec_t tg0 = zero_intrin<Vec_t>();
            Vec_t tg1 = zero_intrin<Vec_t>();
            Vec_t tg2 = zero_intrin<Vec_t>();
            Vec_t gxx = zero_intrin<Vec_t>();
            Vec_t gxy = zero_intrin<Vec_t>();
            Vec_t gxz = zero_intrin<Vec_t>();
            Vec_t gyy = zero_intrin<Vec_t>();
            Vec_t gyz = zero_intrin<Vec_t>();
            Vec_t gzz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));
                Vec_t s0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t s1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t s2 = bcast_intrin<Vec_t>(&src_value[2][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t r3inv = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t r5inv = mul_intrin(mul_intrin(rinv, rinv), r3inv);
                Vec_t r7inv = mul_intrin(mul_intrin(rinv, rinv), r5inv);

                Vec_t rdotn = mul_intrin(s0, dx);
                rdotn = add_intrin(rdotn, mul_intrin(s1, dy));
                rdotn = add_intrin(rdotn, mul_intrin(s2, dz));

                tv = add_intrin(tv, mul_intrin(rdotn, r3inv));
                tg0 = add_intrin(tg0, mul_intrin(mul_intrin(s0, r2), r5inv));
                tg0 = add_intrin(tg0, mul_intrin(mul_intrin(mul_intrin(rdotn, nthree), r5inv), dx));
                tg1 = add_intrin(tg1, mul_intrin(mul_intrin(s1, r2), r5inv));
                tg1 = add_intrin(tg1, mul_intrin(mul_intrin(mul_intrin(rdotn, nthree), r5inv), dy));
                tg2 = add_intrin(tg2, mul_intrin(mul_intrin(s2, r2), r5inv));
                tg2 = add_intrin(tg2, mul_intrin(mul_intrin(mul_intrin(rdotn, nthree), r5inv), dz));
                gxx = add_intrin(gxx, mul_intrin(mul_intrin(mul_intrin(fifteen, mul_intrin(dx, dx)), rdotn), r7inv));
                gxx = sub_intrin(gxx, mul_intrin(r7inv, mul_intrin(add_intrin(mul_intrin(three, rdotn),
                                                                              mul_intrin(six, mul_intrin(dx, s0))),
                                                                   r2)));
                gyy = add_intrin(gyy, mul_intrin(mul_intrin(mul_intrin(fifteen, mul_intrin(dy, dy)), rdotn), r7inv));
                gyy = sub_intrin(gyy, mul_intrin(r7inv, mul_intrin(add_intrin(mul_intrin(three, rdotn),
                                                                              mul_intrin(six, mul_intrin(dy, s1))),
                                                                   r2)));
                gzz = add_intrin(gzz, mul_intrin(mul_intrin(mul_intrin(fifteen, mul_intrin(dz, dz)), rdotn), r7inv));
                gzz = sub_intrin(gzz, mul_intrin(r7inv, mul_intrin(add_intrin(mul_intrin(three, rdotn),
                                                                              mul_intrin(six, mul_intrin(dz, s2))),
                                                                   r2)));
                const Vec_t threer2 = mul_intrin(r2, three);
                gxy = add_intrin(gxy, mul_intrin(r7inv, mul_intrin(fifteen, mul_intrin(dx, mul_intrin(dy, rdotn)))));
                gxy = sub_intrin(
                    gxy, mul_intrin(r7inv, mul_intrin(threer2, add_intrin(mul_intrin(dx, s1), mul_intrin(dy, s0)))));
                gxz = add_intrin(gxz, mul_intrin(r7inv, mul_intrin(fifteen, mul_intrin(dx, mul_intrin(dz, rdotn)))));
                gxz = sub_intrin(
                    gxz, mul_intrin(r7inv, mul_intrin(threer2, add_intrin(mul_intrin(dx, s2), mul_intrin(dz, s0)))));
                gyz = add_intrin(gyz, mul_intrin(r7inv, mul_intrin(fifteen, mul_intrin(dy, mul_intrin(dz, rdotn)))));
                gyz = sub_intrin(
                    gyz, mul_intrin(r7inv, mul_intrin(threer2, add_intrin(mul_intrin(dy, s2), mul_intrin(dz, s1)))));
            }
            tv = add_intrin(mul_intrin(tv, oofp), load_intrin<Vec_t>(&trg_value[0][t]));
            tg0 = add_intrin(mul_intrin(tg0, oofp5), load_intrin<Vec_t>(&trg_value[1][t]));
            tg1 = add_intrin(mul_intrin(tg1, oofp5), load_intrin<Vec_t>(&trg_value[2][t]));
            tg2 = add_intrin(mul_intrin(tg2, oofp5), load_intrin<Vec_t>(&trg_value[3][t]));
            gxx = add_intrin(mul_intrin(gxx, oofp7), load_intrin<Vec_t>(&trg_value[4][t]));
            gxy = add_intrin(mul_intrin(gxy, oofp7), load_intrin<Vec_t>(&trg_value[5][t]));
            gxz = add_intrin(mul_intrin(gxz, oofp7), load_intrin<Vec_t>(&trg_value[6][t]));
            gyy = add_intrin(mul_intrin(gyy, oofp7), load_intrin<Vec_t>(&trg_value[7][t]));
            gyz = add_intrin(mul_intrin(gyz, oofp7), load_intrin<Vec_t>(&trg_value[8][t]));
            gzz = add_intrin(mul_intrin(gzz, oofp7), load_intrin<Vec_t>(&trg_value[8][t]));
            store_intrin(&trg_value[0][t], tv);
            store_intrin(&trg_value[1][t], tg0);
            store_intrin(&trg_value[2][t], tg1);
            store_intrin(&trg_value[3][t], tg2);
            store_intrin(&trg_value[4][t], gxx);
            store_intrin(&trg_value[5][t], gxy);
            store_intrin(&trg_value[6][t], gxz);
            store_intrin(&trg_value[7][t], gyy);
            store_intrin(&trg_value[8][t], gyz);
            store_intrin(&trg_value[9][t], gzz);
        }
    }
#undef SRC_BLK
}

GEN_KERNEL(laplace_dipolepgradgrad, laplace_dipolepgradgrad_uKernel, 3, 10)

/**
 * @brief LaplaceLayerkernel class
 *
 * @tparam T float or double
 */
template <class T>
struct LaplaceLayerKernel {
    inline static const Kernel<T> &PGrad();     ///< Laplace PGrad Kernel
    inline static const Kernel<T> &PGradGrad(); ///< Laplace PGradGrad

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
inline const Kernel<T> &LaplaceLayerKernel<T>::PGrad() {
    // S2U - single-layer density — to — potential kernel (1 x 1)
    // D2U - double-layer density — to — potential kernel (1+3 x 1)
    // S2UdU - single-layer density — to — potential & gradient (1 x 1)
    // D2UdU - double-layer density — to — potential & gradient (1+3 x 1)

    static Kernel<T> lap_pker = BuildKernel<T, laplace_poten<T, NEWTON_ITE>, laplace_dipolep<T, NEWTON_ITE>>(
        "laplace", 3, std::pair<int, int>(1, 1));
    lap_pker.surf_dim = 3;

    static Kernel<T> lap_pgker = BuildKernel<T, laplace_pgrad<T, NEWTON_ITE>, laplace_dipolepgrad<T, NEWTON_ITE>>(
        "laplace_PGrad", 3, std::pair<int, int>(1, 4), &lap_pker, &lap_pker, NULL, &lap_pker, &lap_pker, NULL,
        &lap_pker, NULL);
    lap_pgker.surf_dim = 3;

    return lap_pgker;
}

template <class T>
inline const Kernel<T> &LaplaceLayerKernel<T>::PGradGrad() {

    static Kernel<T> lap_pker = BuildKernel<T, laplace_poten<T, NEWTON_ITE>, laplace_dipolep<T, NEWTON_ITE>>(
        "laplace", 3, std::pair<int, int>(1, 1));
    lap_pker.surf_dim = 3;

    static Kernel<T> lap_pgker =
        BuildKernel<T, laplace_pgradgrad<T, NEWTON_ITE>, laplace_dipolepgradgrad<T, NEWTON_ITE>>(
            "laplace_PGradGrad", 3, std::pair<int, int>(1, 10), &lap_pker, &lap_pker, NULL, &lap_pker, &lap_pker, NULL,
            &lap_pker, NULL);
    lap_pgker.surf_dim = 3;

    return lap_pgker;
}

} // namespace pvfmm

#endif
