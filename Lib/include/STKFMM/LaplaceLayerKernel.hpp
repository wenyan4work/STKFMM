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
            gzz = add_intrin(mul_intrin(gzz, oofp7), load_intrin<Vec_t>(&trg_value[9][t]));
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

template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void laplace_quadp_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
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
    Vec_t oofp5 = set_intrin<Vec_t, Real_t>(-OOFP5);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
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
            Vec_t p = zero_intrin<Vec_t>();

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
                Vec_t commonCoeffn3 = mul_intrin(commonCoeff, nthree);

                Vec_t trace = add_intrin(add_intrin(sxx, syy), szz);
                p = add_intrin(p, mul_intrin(add_intrin(commonCoeffn3, mul_intrin(r2, trace)), rinv5));
            }
            p = add_intrin(mul_intrin(p, oofp5), load_intrin<Vec_t>(&trg_value[0][t]));

            store_intrin(&trg_value[0][t], p);
        }
    }
#undef SRC_BLK
}

template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void laplace_quadpgradgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
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
    const Real_t OOFP9 = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal *
                                nwtn_scal * nwtn_scal * const_pi<Real_t>());
    Vec_t oofp5 = set_intrin<Vec_t, Real_t>(-OOFP5);
    Vec_t oofp7 = set_intrin<Vec_t, Real_t>(OOFP7);
    Vec_t oofp9 = set_intrin<Vec_t, Real_t>(OOFP9);

    Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    Vec_t onezerofive = set_intrin<Vec_t, Real_t>(105.0);
    Vec_t six = set_intrin<Vec_t, Real_t>(6.0);
    Vec_t nine = set_intrin<Vec_t, Real_t>(9.0);
    Vec_t fifteen = set_intrin<Vec_t, Real_t>(15.0);
    Vec_t five = set_intrin<Vec_t, Real_t>(5.0);
    Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);
    Vec_t onethree = set_intrin<Vec_t, Real_t>(1. / 3.0);

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
            Vec_t p = zero_intrin<Vec_t>();
            // g
            Vec_t gx = zero_intrin<Vec_t>();
            Vec_t gy = zero_intrin<Vec_t>();
            Vec_t gz = zero_intrin<Vec_t>();
            // gg
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
                const Vec_t rinv9 = mul_intrin(rinv7, rinv2);

                // commonCoeffn3 = -3 rk rl Qkl
                // commonCoeff5 = 5 rk rl Qkl
                Vec_t rrQ = dx * dx * sxx + dx * dy * sxy + dx * dz * sxz + dy * dx * syx + dy * dy * syy +
                            dy * dz * syz + dz * dx * szx + dz * dy * szy + dz * dz * szz;
                Vec_t commonCoeff5 = rrQ * five;
                Vec_t commonCoeffn3 = rrQ * nthree;

                Vec_t trace = sxx + syy + szz;

                Vec_t rksxk = add_intrin(mul_intrin(dx, sxx), add_intrin(mul_intrin(dy, sxy), mul_intrin(dz, sxz)));
                Vec_t rksyk = add_intrin(mul_intrin(dx, syx), add_intrin(mul_intrin(dy, syy), mul_intrin(dz, syz)));
                Vec_t rkszk = add_intrin(mul_intrin(dx, szx), add_intrin(mul_intrin(dy, szy), mul_intrin(dz, szz)));

                Vec_t rkskx = add_intrin(mul_intrin(dx, sxx), add_intrin(mul_intrin(dy, syx), mul_intrin(dz, szx)));
                Vec_t rksky = add_intrin(mul_intrin(dx, sxy), add_intrin(mul_intrin(dy, syy), mul_intrin(dz, szy)));
                Vec_t rkskz = add_intrin(mul_intrin(dx, sxz), add_intrin(mul_intrin(dy, syz), mul_intrin(dz, szz)));

                p = add_intrin(p, mul_intrin(add_intrin(commonCoeffn3, mul_intrin(r2, trace)), rinv5));
                Vec_t px = sub_intrin(mul_intrin(dx, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rksxk, rkskx), mul_intrin(dx, trace))));
                Vec_t py = sub_intrin(mul_intrin(dy, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rksyk, rksky), mul_intrin(dy, trace))));
                Vec_t pz = sub_intrin(mul_intrin(dz, commonCoeff5),
                                      mul_intrin(r2, add_intrin(add_intrin(rkszk, rkskz), mul_intrin(dz, trace))));
                gx = add_intrin(gx, mul_intrin(mul_intrin(px, nthree), rinv7));
                gy = add_intrin(gy, mul_intrin(mul_intrin(py, nthree), rinv7));
                gz = add_intrin(gz, mul_intrin(mul_intrin(pz, nthree), rinv7));
                gxx = gxx + rinv9 * (onezerofive * dx * dx * rrQ -
                                     r2 * fifteen * (rrQ + dx * two * (rksxk + rkskx) + dx * dx * trace) +
                                     three * r2 * r2 * (trace + two * sxx));
                gyy = gyy + rinv9 * (onezerofive * dy * dy * rrQ -
                                     r2 * fifteen * (rrQ + dy * two * (rksyk + rksky) + dy * dy * trace) +
                                     three * r2 * r2 * (trace + two * syy));
                gzz = gzz + rinv9 * (onezerofive * dz * dz * rrQ -
                                     r2 * fifteen * (rrQ + dz * two * (rkszk + rkskz) + dz * dz * trace) +
                                     three * r2 * r2 * (trace + two * szz));
                gxy = gxy +
                      rinv9 * (onezerofive * (dx * dy * rrQ) -
                               r2 * fifteen * ((dy * rksxk + dy * rkskx + dx * rksyk + dx * rksky) + dx * dy * trace) +
                               three * r2 * r2 * (sxy + syx));
                gxz = gxz +
                      rinv9 * (onezerofive * (dx * dz * rrQ) -
                               r2 * fifteen * ((dz * rksxk + dz * rkskx + dx * rkszk + dx * rkskz) + dx * dz * trace) +
                               three * r2 * r2 * (sxz + szx));
                gyz = gyz +
                      rinv9 * (onezerofive * (dy * dz * rrQ) -
                               r2 * fifteen * ((dy * rkszk + dy * rkskz + dz * rksyk + dz * rksky) + dy * dz * trace) +
                               three * r2 * r2 * (syz + szy));
            }
            p = add_intrin(mul_intrin(p, oofp5), load_intrin<Vec_t>(&trg_value[0][t]));
            gx = add_intrin(mul_intrin(gx, oofp7), load_intrin<Vec_t>(&trg_value[1][t]));
            gy = add_intrin(mul_intrin(gy, oofp7), load_intrin<Vec_t>(&trg_value[2][t]));
            gz = add_intrin(mul_intrin(gz, oofp7), load_intrin<Vec_t>(&trg_value[3][t]));
            gxx = add_intrin(mul_intrin(gxx, oofp9), load_intrin<Vec_t>(&trg_value[4][t]));
            gxy = add_intrin(mul_intrin(gxy, oofp9), load_intrin<Vec_t>(&trg_value[5][t]));
            gxz = add_intrin(mul_intrin(gxz, oofp9), load_intrin<Vec_t>(&trg_value[6][t]));
            gyy = add_intrin(mul_intrin(gyy, oofp9), load_intrin<Vec_t>(&trg_value[7][t]));
            gyz = add_intrin(mul_intrin(gyz, oofp9), load_intrin<Vec_t>(&trg_value[8][t]));
            gzz = add_intrin(mul_intrin(gzz, oofp9), load_intrin<Vec_t>(&trg_value[9][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], gx);
            store_intrin(&trg_value[2][t], gy);
            store_intrin(&trg_value[3][t], gz);
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

GEN_KERNEL(laplace_pgrad, laplace_pgrad_uKernel, 1, 4)
GEN_KERNEL(laplace_pgradgrad, laplace_pgradgrad_uKernel, 1, 10)
GEN_KERNEL(laplace_dipolep, laplace_dipolep_uKernel, 3, 1)
GEN_KERNEL(laplace_dipolepgrad, laplace_dipolepgrad_uKernel, 3, 4)
GEN_KERNEL(laplace_dipolepgradgrad, laplace_dipolepgradgrad_uKernel, 3, 10)
GEN_KERNEL(laplace_quadp, laplace_quadp_uKernel, 9, 1)
GEN_KERNEL(laplace_quadpgradgrad, laplace_quadpgradgrad_uKernel, 9, 10)

/**
 * @brief LaplaceLayerkernel class
 *
 * @tparam T float or double
 */
template <class T>
struct LaplaceLayerKernel {
    inline static const Kernel<T> &PGrad();      ///< Laplace PGrad Kernel
    inline static const Kernel<T> &PGradGrad();  ///< Laplace PGradGrad
    inline static const Kernel<T> &QPGradGrad(); ///< Laplace Quadruple PGradGrad, no double layer

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

template <class T>
inline const Kernel<T> &LaplaceLayerKernel<T>::QPGradGrad() {

    static Kernel<T> lap_pker = BuildKernel<T, laplace_poten<T, NEWTON_ITE>>("laplace", 3, std::pair<int, int>(1, 1));
    static Kernel<T> lap_pggker =
        BuildKernel<T, laplace_pgradgrad<T, NEWTON_ITE>>("laplace", 3, std::pair<int, int>(1, 10));
    static Kernel<T> lap_qpker = BuildKernel<T, laplace_quadp<T, NEWTON_ITE>>("laplace", 3, std::pair<int, int>(9, 1));

    static Kernel<T> lap_pgker = BuildKernel<T, laplace_quadpgradgrad<T, NEWTON_ITE>>(
        "laplace_QPGradGrad", 3, std::pair<int, int>(9, 10), &lap_qpker, &lap_qpker, NULL, //
        &lap_pker, &lap_pker, &lap_pggker,                                                 //
        &lap_pker, &lap_pggker);

    return lap_pgker;
}

} // namespace pvfmm

#endif
