#ifndef STOKESDOUBLELAYER_HPP
#define STOKESDOUBLELAYER_HPP

#include <cmath>
#include <cstdlib>
#include <vector>

// pvfmm headers
#include <pvfmm.hpp>

namespace pvfmm {

/*********************************************************
 *                                                        *
 *   Stokes Double P Vel kernel, source: 9, target: 4     *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_doublepvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                               Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1 / (8 * const_pi<Real_t>() * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal);
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);     // vi = 1/8pi (-3rirjrk/r^5) Djk
    const Vec_t facp = set_intrin<Vec_t, Real_t>(FACV * 2); // p = 1/4pi (-3 rjrk/r^5 + delta_jk) Djk
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

                Vec_t rinv = RSQRT_INTRIN(r2);
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

                Vec_t trace = add_intrin(add_intrin(sxx, syy), szz);
                p = add_intrin(p, mul_intrin(add_intrin(commonCoeff, mul_intrin(r2, trace)), rinv5));
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
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_doublepvel(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                       mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 4, stokes_doublepvel_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(         \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

/*********************************************************
 *                                                        *
 * Stokes Double P Vel Grad kernel, source: 9, target: 16 *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_doublepvelgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                   Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

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

                const Vec_t rinv = RSQRT_INTRIN(r2);
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
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_doublepvelgrad(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                           mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 16, stokes_doublepvelgrad_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(    \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

/*********************************************************
 *                                                        *
 *  Stokes Double P Vel Lap kernel, source: 9, target: 7 *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_doublelaplacian_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                    Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

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

                const Vec_t rinv = RSQRT_INTRIN(r2);
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
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_doublelaplacian(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                            mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 7, stokes_doublelaplacian_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(    \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

/*********************************************************
 *                                                        *
 *   Stokes Double Traction kernel, source: 9, target: 9  *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_doubletraction_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                   Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

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

                const Vec_t rinv = RSQRT_INTRIN(r2);
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

                // np = mul_intrin(mul_intrin(mul_intrin(add_intrin(commonCoeffn3, mul_intrin(r2, trace)), rinv7), r2),
                // none);
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
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_doubletraction(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                           mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 9, stokes_doubletraction_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(     \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

/*********************************************************
 *                                                        *
 *   Stokes Double Vel kernel, source: 9, target: 3       *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_double_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                           Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOEP = -3.0 / (4 * const_pi<Real_t>());
    Vec_t inv_nwtn_scal5 = set_intrin<Vec_t, Real_t>(1.0 / (nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal));

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

            Vec_t tvx = zero_intrin<Vec_t>();
            Vec_t tvy = zero_intrin<Vec_t>();
            Vec_t tvz = zero_intrin<Vec_t>();
            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                Vec_t sv0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sv1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sv2 = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t sv3 = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t sv4 = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t sv5 = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t sv6 = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t sv7 = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t sv8 = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t rinv2 = mul_intrin(rinv, rinv);
                Vec_t rinv4 = mul_intrin(rinv2, rinv2);

                Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv4), inv_nwtn_scal5);

                Vec_t commonCoeff = mul_intrin(sv0, mul_intrin(dx, dx));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv1, mul_intrin(dx, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv2, mul_intrin(dx, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv3, mul_intrin(dy, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv4, mul_intrin(dy, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv5, mul_intrin(dy, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv6, mul_intrin(dz, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv7, mul_intrin(dz, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv8, mul_intrin(dz, dz)));

                tvx = add_intrin(tvx, mul_intrin(rinv5, mul_intrin(dx, commonCoeff)));
                tvy = add_intrin(tvy, mul_intrin(rinv5, mul_intrin(dy, commonCoeff)));
                tvz = add_intrin(tvz, mul_intrin(rinv5, mul_intrin(dz, commonCoeff)));
            }
            Vec_t ooep = set_intrin<Vec_t, Real_t>(OOEP);

            tvx = add_intrin(mul_intrin(tvx, ooep), load_intrin<Vec_t>(&trg_value[0][t]));
            tvy = add_intrin(mul_intrin(tvy, ooep), load_intrin<Vec_t>(&trg_value[1][t]));
            tvz = add_intrin(mul_intrin(tvz, ooep), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], tvx);
            store_intrin(&trg_value[1][t], tvy);
            store_intrin(&trg_value[2][t], tvz);
        }
    }
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_double(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                   mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 3, stokes_double_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(             \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

/*********************************************************
 *                                                        *
 *  Stokes Double Pressure kernel, source: 9, target: 1   *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_double_pressure_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                    Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOEP = 1.0 / (2 * const_pi<Real_t>());
    Vec_t inv_nwtn_scal5 = set_intrin<Vec_t, Real_t>(1.0 / (nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal));
    Vec_t negthree = set_intrin<Vec_t, Real_t>(-3.0);

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

                Vec_t sv0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sv1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sv2 = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t sv3 = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t sv4 = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t sv5 = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t sv6 = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t sv7 = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t sv8 = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t rinv2 = mul_intrin(rinv, rinv);
                Vec_t rinv4 = mul_intrin(rinv2, rinv2);

                Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv4), inv_nwtn_scal5);

                Vec_t pressure = mul_intrin(sv0, mul_intrin(dx, dx));
                pressure = add_intrin(pressure, mul_intrin(sv1, mul_intrin(dx, dy)));
                pressure = add_intrin(pressure, mul_intrin(sv2, mul_intrin(dx, dz)));
                pressure = add_intrin(pressure, mul_intrin(sv3, mul_intrin(dy, dx)));
                pressure = add_intrin(pressure, mul_intrin(sv4, mul_intrin(dy, dy)));
                pressure = add_intrin(pressure, mul_intrin(sv5, mul_intrin(dy, dz)));
                pressure = add_intrin(pressure, mul_intrin(sv6, mul_intrin(dz, dx)));
                pressure = add_intrin(pressure, mul_intrin(sv7, mul_intrin(dz, dy)));
                pressure = add_intrin(pressure, mul_intrin(sv8, mul_intrin(dz, dz)));
                pressure = mul_intrin(pressure, negthree);
                pressure = add_intrin(pressure, mul_intrin(sv0, r2));
                pressure = add_intrin(pressure, mul_intrin(sv4, r2));
                pressure = add_intrin(pressure, mul_intrin(sv8, r2));

                tv = add_intrin(tv, mul_intrin(rinv5, pressure));
            }
            Vec_t ooep = set_intrin<Vec_t, Real_t>(OOEP);

            tv = add_intrin(mul_intrin(tv, ooep), load_intrin<Vec_t>(&trg_value[0][t]));

            store_intrin(&trg_value[0][t], tv);
        }
    }
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_double_pressure(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                            mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 1, stokes_double_pressure_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(    \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

// Stokes double layer gradient of velocity, source: 9, target: 9
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_dgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                          Matrix<Real_t> &trg_value) {
#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOEP = -3.0 / (4 * const_pi<Real_t>());
    Vec_t inv_nwtn_scal7 = set_intrin<Vec_t, Real_t>(
        1.0 / (nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal));
    Vec_t inv_nwtn_scal5 = set_intrin<Vec_t, Real_t>(1.0 / (nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal));
    Vec_t nwtn_scal2 = set_intrin<Vec_t, Real_t>((nwtn_scal * nwtn_scal));
    Vec_t negone = set_intrin<Vec_t, Real_t>(-1.0);
    Vec_t negtwo = set_intrin<Vec_t, Real_t>(-2.0);
    Vec_t five = set_intrin<Vec_t, Real_t>(5.0);

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

            Vec_t tv0ac = zero_intrin<Vec_t>();
            Vec_t tv1ac = zero_intrin<Vec_t>();
            Vec_t tv2ac = zero_intrin<Vec_t>();
            Vec_t tv3ac = zero_intrin<Vec_t>();
            Vec_t tv4ac = zero_intrin<Vec_t>();
            Vec_t tv5ac = zero_intrin<Vec_t>();
            Vec_t tv6ac = zero_intrin<Vec_t>();
            Vec_t tv7ac = zero_intrin<Vec_t>();
            Vec_t tv8ac = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                Vec_t sv0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sv1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sv2 = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t sv3 = bcast_intrin<Vec_t>(&src_value[3][s]);
                Vec_t sv4 = bcast_intrin<Vec_t>(&src_value[4][s]);
                Vec_t sv5 = bcast_intrin<Vec_t>(&src_value[5][s]);
                Vec_t sv6 = bcast_intrin<Vec_t>(&src_value[6][s]);
                Vec_t sv7 = bcast_intrin<Vec_t>(&src_value[7][s]);
                Vec_t sv8 = bcast_intrin<Vec_t>(&src_value[8][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t rinv2 = mul_intrin(rinv, rinv);
                Vec_t rinv4 = mul_intrin(rinv2, rinv2);

                Vec_t rinv5 = mul_intrin(rinv, rinv4);
                Vec_t rinv7 = mul_intrin(mul_intrin(rinv2, rinv5), inv_nwtn_scal7);
                rinv5 = mul_intrin(rinv5, inv_nwtn_scal5);

                Vec_t commonCoeff = mul_intrin(sv0, mul_intrin(dx, dx));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv1, mul_intrin(dx, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv2, mul_intrin(dx, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv3, mul_intrin(dy, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv4, mul_intrin(dy, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv5, mul_intrin(dy, dz)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv6, mul_intrin(dz, dx)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv7, mul_intrin(dz, dy)));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv8, mul_intrin(dz, dz)));
                commonCoeff = mul_intrin(negone, commonCoeff);

                // (-2*(t0 - s0)*sv0 - (t1 - s1)*(sv1 + sv3) - (t2 - s2)*(sv2 + sv6))
                Vec_t dcFd0 = mul_intrin(negtwo, mul_intrin(dx, sv0));
                dcFd0 = add_intrin(dcFd0, mul_intrin(negone, mul_intrin(dy, add_intrin(sv1, sv3))));
                dcFd0 = add_intrin(dcFd0, mul_intrin(negone, mul_intrin(dz, add_intrin(sv2, sv6))));

                // (-2*(t1 - s1)*sv4 - (t0 - s0)*(sv1 + sv3) - (t2 - s2)*(sv5 + sv7))
                Vec_t dcFd1 = mul_intrin(negtwo, mul_intrin(dy, sv4));
                dcFd1 = add_intrin(dcFd1, mul_intrin(negone, mul_intrin(dx, add_intrin(sv1, sv3))));
                dcFd1 = add_intrin(dcFd1, mul_intrin(negone, mul_intrin(dz, add_intrin(sv5, sv7))));

                // (-2*(t2 - s2)*sv8 - (t0 - s0)*(sv2 + sv6) - (t1 - s1)*(sv5 + sv7))
                Vec_t dcFd2 = mul_intrin(negtwo, mul_intrin(dz, sv8));
                dcFd2 = add_intrin(dcFd2, mul_intrin(negone, mul_intrin(dx, add_intrin(sv2, sv6))));
                dcFd2 = add_intrin(dcFd2, mul_intrin(negone, mul_intrin(dy, add_intrin(sv5, sv7))));

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
                //  - Outer[Times, rvec, {dcFd0, dcFd1, dcFd2}] / rnorm ^5
                //  - IdentityMatrix[3] * commonCoeff / rnorm ^ 5);
                tv0 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dx, dx)));
                tv0 = mul_intrin(tv0, rinv7);
                tv0 = add_intrin(tv0, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dx, dcFd0))));
                tv0 = add_intrin(tv0, mul_intrin(negone, mul_intrin(rinv5, commonCoeff)));
                tv1 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dx, dy)));
                tv1 = mul_intrin(tv1, rinv7);
                tv1 = add_intrin(tv1, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dx, dcFd1))));
                tv2 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dx, dz)));
                tv2 = mul_intrin(tv2, rinv7);
                tv2 = add_intrin(tv2, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dx, dcFd2))));

                tv3 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dy, dx)));
                tv3 = mul_intrin(tv3, rinv7);
                tv3 = add_intrin(tv3, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dy, dcFd0))));
                tv4 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dy, dy)));
                tv4 = mul_intrin(tv4, rinv7);
                tv4 = add_intrin(tv4, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dy, dcFd1))));
                tv4 = add_intrin(tv4, mul_intrin(negone, mul_intrin(rinv5, commonCoeff)));
                tv5 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dy, dz)));
                tv5 = mul_intrin(tv5, rinv7);
                tv5 = add_intrin(tv5, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dy, dcFd2))));

                tv6 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dz, dx)));
                tv6 = mul_intrin(tv6, rinv7);
                tv6 = add_intrin(tv6, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dz, dcFd0))));
                tv7 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dz, dy)));
                tv7 = mul_intrin(tv7, rinv7);
                tv7 = add_intrin(tv7, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dz, dcFd1))));
                tv8 = mul_intrin(five, mul_intrin(commonCoeff, mul_intrin(dz, dz)));
                tv8 = mul_intrin(tv8, rinv7);
                tv8 = add_intrin(tv8, mul_intrin(negone, mul_intrin(rinv5, mul_intrin(dz, dcFd2))));
                tv8 = add_intrin(tv8, mul_intrin(negone, mul_intrin(rinv5, commonCoeff)));

                tv0ac = add_intrin(tv0ac, tv0);
                tv1ac = add_intrin(tv1ac, tv1);
                tv2ac = add_intrin(tv2ac, tv2);
                tv3ac = add_intrin(tv3ac, tv3);
                tv4ac = add_intrin(tv4ac, tv4);
                tv5ac = add_intrin(tv5ac, tv5);
                tv6ac = add_intrin(tv6ac, tv6);
                tv7ac = add_intrin(tv7ac, tv7);
                tv8ac = add_intrin(tv8ac, tv8);
            }
            Vec_t ooep = set_intrin<Vec_t, Real_t>(OOEP);

            tv0ac = add_intrin(mul_intrin(tv0ac, ooep), load_intrin<Vec_t>(&trg_value[0][t]));
            tv1ac = add_intrin(mul_intrin(tv1ac, ooep), load_intrin<Vec_t>(&trg_value[1][t]));
            tv2ac = add_intrin(mul_intrin(tv2ac, ooep), load_intrin<Vec_t>(&trg_value[2][t]));
            tv3ac = add_intrin(mul_intrin(tv3ac, ooep), load_intrin<Vec_t>(&trg_value[3][t]));
            tv4ac = add_intrin(mul_intrin(tv4ac, ooep), load_intrin<Vec_t>(&trg_value[4][t]));
            tv5ac = add_intrin(mul_intrin(tv5ac, ooep), load_intrin<Vec_t>(&trg_value[5][t]));
            tv6ac = add_intrin(mul_intrin(tv6ac, ooep), load_intrin<Vec_t>(&trg_value[6][t]));
            tv7ac = add_intrin(mul_intrin(tv7ac, ooep), load_intrin<Vec_t>(&trg_value[7][t]));
            tv8ac = add_intrin(mul_intrin(tv8ac, ooep), load_intrin<Vec_t>(&trg_value[8][t]));

            store_intrin(&trg_value[0][t], tv0ac);
            store_intrin(&trg_value[1][t], tv1ac);
            store_intrin(&trg_value[2][t], tv2ac);
            store_intrin(&trg_value[3][t], tv3ac);
            store_intrin(&trg_value[4][t], tv4ac);
            store_intrin(&trg_value[5][t], tv5ac);
            store_intrin(&trg_value[6][t], tv6ac);
            store_intrin(&trg_value[7][t], tv7ac);
            store_intrin(&trg_value[8][t], tv8ac);
        }
    }
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void stokes_dgrad(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                  mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 9, 9, stokes_dgrad_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(              \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

template <class T>
struct StokesDoubleLayerKernel {
    inline static const Kernel<T> &PVel();     // tested
    inline static const Kernel<T> &PVelGrad(); // tested

  private:
    static constexpr int NEWTON_ITE =
        sizeof(T) / 4; // generate NEWTON_ITE at compile time. 1 for float and 2 for double
};

template <class T>
inline const Kernel<T> &StokesDoubleLayerKernel<T>::PVel() {
    static Kernel<T> s2t_ker =
        BuildKernel<T, stokes_doublepvel<T, NEWTON_ITE>>("stokes_double_pvel", 3, std::pair<int, int>(9, 4));
    return s2t_ker;
}

template <class T>
inline const Kernel<T> &StokesDoubleLayerKernel<T>::PVelGrad() {
    static Kernel<T> double_ker =
        BuildKernel<T, stokes_doublepvel<T, NEWTON_ITE>>("stokes_double_pvel", 3, std::pair<int, int>(9, 4));
    static Kernel<T> s2t_ker = BuildKernel<T, stokes_doublepvelgrad<T, NEWTON_ITE>>(
        "stokes_double_pvelgrad", 3, std::pair<int, int>(9, 16), &double_ker, &double_ker, NULL, &double_ker,
        &double_ker, NULL, &double_ker, NULL);
    return s2t_ker;
}

} // namespace pvfmm
#endif
