#ifndef INCLUDE_RPYCUSTOMKERNEL_H_
#define INCLUDE_RPYCUSTOMKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

namespace pvfmm {

/**********************************************************
 *                                                        *
 *     Stokes P Vel kernel, source: 4, target: 4          *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t,
          Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void rpy_u_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                   Matrix<Real_t> &trg_coord, Matrix<Real_t> &trg_value) {

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
    Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    const Vec_t one_over_three =
        set_intrin<Vec_t, Real_t>(static_cast<Real_t>(1.0 / 3.0));

    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t a = bcast_intrin<Vec_t>(&src_value[3][s]);

                const Vec_t a2_over_three =
                    mul_intrin(mul_intrin(a, a), one_over_three);
                const Vec_t r2 = add_intrin(
                    add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)),
                    mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(RSQRT_INTRIN(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(
                    add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)),
                    mul_intrin(fz, dz));

                vx =
                    add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx),
                                                         mul_intrin(dx, fdotr)),
                                              rinv3));
                vy =
                    add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy),
                                                         mul_intrin(dy, fdotr)),
                                              rinv3));
                vz =
                    add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz),
                                                         mul_intrin(dz, fdotr)),
                                              rinv3));
                const Vec_t three_fdotr_rinv5 =
                    mul_intrin(mul_intrin(three, fdotr), rinv5);

                vx = add_intrin(
                    vx,
                    mul_intrin(a2_over_three,
                               sub_intrin(mul_intrin(fx, rinv3),
                                          mul_intrin(three_fdotr_rinv5, dx))));
                vy = add_intrin(
                    vy,
                    mul_intrin(a2_over_three,
                               sub_intrin(mul_intrin(fy, rinv3),
                                          mul_intrin(three_fdotr_rinv5, dy))));
                vz = add_intrin(
                    vz,
                    mul_intrin(a2_over_three,
                               sub_intrin(mul_intrin(fz, rinv3),
                                          mul_intrin(three_fdotr_rinv5, dz))));
            }

            vx = add_intrin(mul_intrin(vx, FACV),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV),
                            load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
#undef SRC_BLK
}

// '##' is the token parsing operator
template <class T, int newton_iter = 0>
void rpy_u(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt,
           T *v_trg, mem::MemoryManager *mem_mgr) {
#define RPY_KER_NWTN(nwtn)                                                     \
    if (newton_iter == nwtn)                                                   \
    generic_kernel<                                                            \
        Real_t, 4, 3,                                                          \
        rpy_u_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(      \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg,       \
        trg_cnt, (Real_t *)v_trg, mem_mgr)
#define RPY_KERNEL                                                             \
    RPY_KER_NWTN(0);                                                           \
    RPY_KER_NWTN(1);                                                           \
    RPY_KER_NWTN(2);                                                           \
    RPY_KER_NWTN(3);

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
        RPY_KERNEL;
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
        RPY_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        RPY_KERNEL;
#undef Vec_t
    }

#undef RPY_KER_NWTN
#undef RPY_KERNEL
}

/**********************************************************
 *                                                        *
 * RPY Force,a Vel kernel,source: 4, target: 6            *
 *       fx,fy,fz,a -> ux,uy,uz,lapux,lapuy,lapuz         *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t,
          Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void rpy_ulapu_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                       Matrix<Real_t> &trg_coord, Matrix<Real_t> &trg_value) {

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
    const Vec_t FACV =
        set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    const Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    const Vec_t one_over_three = set_intrin<Vec_t, Real_t>(1.0 / 3.0);
    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();
            Vec_t lapvx = zero_intrin<Vec_t>();
            Vec_t lapvy = zero_intrin<Vec_t>();
            Vec_t lapvz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t a = bcast_intrin<Vec_t>(&src_value[3][s]);

                const Vec_t a2_over_three =
                    mul_intrin(mul_intrin(a, a), one_over_three);
                const Vec_t r2 = add_intrin(
                    add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)),
                    mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(RSQRT_INTRIN(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(
                    add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)),
                    mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 =
                    mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(
                    vx, add_intrin(mul_intrin(fx, rinv),
                                   add_intrin(mul_intrin(dx, fdotr_rinv3),
                                              mul_intrin(a2_over_three, cx))));
                vy = add_intrin(
                    vy, add_intrin(mul_intrin(fy, rinv),
                                   add_intrin(mul_intrin(dy, fdotr_rinv3),
                                              mul_intrin(a2_over_three, cy))));
                vz = add_intrin(
                    vz, add_intrin(mul_intrin(fz, rinv),
                                   add_intrin(mul_intrin(dz, fdotr_rinv3),
                                              mul_intrin(a2_over_three, cz))));

                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV),
                            load_intrin<Vec_t>(&trg_value[2][t]));
            lapvx = add_intrin(mul_intrin(lapvx, FACV),
                               load_intrin<Vec_t>(&trg_value[3][t]));
            lapvy = add_intrin(mul_intrin(lapvy, FACV),
                               load_intrin<Vec_t>(&trg_value[4][t]));
            lapvz = add_intrin(mul_intrin(lapvz, FACV),
                               load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], lapvx);
            store_intrin(&trg_value[4][t], lapvy);
            store_intrin(&trg_value[5][t], lapvz);
        }
    }
#undef SRC_BLK
}

// '##' is the token parsing operator
template <class T, int newton_iter = 0>
void rpy_ulapu(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt,
               T *v_trg, mem::MemoryManager *mem_mgr) {
#define RPY_KER_NWTN(nwtn)                                                     \
    if (newton_iter == nwtn)                                                   \
    generic_kernel<                                                            \
        Real_t, 4, 6,                                                          \
        rpy_ulapu_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(  \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg,       \
        trg_cnt, (Real_t *)v_trg, mem_mgr)
#define RPY_KERNEL                                                             \
    RPY_KER_NWTN(0);                                                           \
    RPY_KER_NWTN(1);                                                           \
    RPY_KER_NWTN(2);                                                           \
    RPY_KER_NWTN(3);

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
        RPY_KERNEL;
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
        RPY_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        RPY_KERNEL;
#undef Vec_t
    }

#undef RPY_KER_NWTN
#undef RPY_KERNEL
}

/**********************************************************
 *                                                        *
 * Stokes Force Vel,lapVel kernel,source: 3, target: 6    *
 *       fx,fy,fz -> ux,uy,uz,lapux,lapuy,lapuz           *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t,
          Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stk_ulapu_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                       Matrix<Real_t> &trg_coord, Matrix<Real_t> &trg_value) {

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
    const Vec_t FACV =
        set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();
            Vec_t lapvx = zero_intrin<Vec_t>();
            Vec_t lapvy = zero_intrin<Vec_t>();
            Vec_t lapvz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);

                const Vec_t r2 = add_intrin(
                    add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)),
                    mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(RSQRT_INTRIN(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(
                    add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)),
                    mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 =
                    mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(vx, add_intrin(mul_intrin(fx, rinv),
                                               mul_intrin(dx, fdotr_rinv3)));
                vy = add_intrin(vy, add_intrin(mul_intrin(fy, rinv),
                                               mul_intrin(dy, fdotr_rinv3)));
                vz = add_intrin(vz, add_intrin(mul_intrin(fz, rinv),
                                               mul_intrin(dz, fdotr_rinv3)));

                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV),
                            load_intrin<Vec_t>(&trg_value[2][t]));
            lapvx = add_intrin(mul_intrin(lapvx, FACV),
                               load_intrin<Vec_t>(&trg_value[3][t]));
            lapvy = add_intrin(mul_intrin(lapvy, FACV),
                               load_intrin<Vec_t>(&trg_value[4][t]));
            lapvz = add_intrin(mul_intrin(lapvz, FACV),
                               load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], lapvx);
            store_intrin(&trg_value[4][t], lapvy);
            store_intrin(&trg_value[5][t], lapvz);
        }
    }
#undef SRC_BLK
}

// '##' is the token parsing operator
template <class T, int newton_iter = 0>
void stk_ulapu(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt,
               T *v_trg, mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                     \
    if (newton_iter == nwtn)                                                   \
    generic_kernel<                                                            \
        Real_t, 3, 6,                                                          \
        stk_ulapu_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(  \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg,       \
        trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STK_KERNEL                                                             \
    STK_KER_NWTN(0);                                                           \
    STK_KER_NWTN(1);                                                           \
    STK_KER_NWTN(2);                                                           \
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
        STK_KERNEL;
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
        STK_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STK_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STK_KERNEL
}

template <class T>
struct RPYTestKernel {
    inline static const Kernel<T> &ulapu(); //   3+1->6
  private:
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
};

// 1 newton for float, 2 newton for double
// the string for stk_ker must be exactly the same as in kernel.txx of pvfmm
template <class T>
inline const Kernel<T> &RPYTestKernel<T>::ulapu() {

    static Kernel<T> g_ker = StokesKernel<T>::velocity();
    static Kernel<T> gr_ker = BuildKernel<T, rpy_u<T, NEWTON_ITE>>(
        "rpy_u", 3, std::pair<int, int>(4, 3));

    static Kernel<T> glapg_ker = BuildKernel<T, stk_ulapu<T, NEWTON_ITE>>(
        "stk_ulapu", 3, std::pair<int, int>(3, 6));
    // glapg_ker.surf_dim = 3;

    static Kernel<T> grlapgr_ker = BuildKernel<T, rpy_ulapu<T, NEWTON_ITE>>(
        "rpy_ulapu", 3, std::pair<int, int>(4, 6),
        &gr_ker,    // k_s2m
        &gr_ker,    // k_s2l
        NULL,       // k_s2t
        &g_ker,     // k_m2m
        &g_ker,     // k_m2l
        &glapg_ker, // k_m2t
        &g_ker,     // k_l2l
        &glapg_ker, // k_l2t
        NULL);
    // grlapgr_ker.surf_dim = 4;
    return grlapgr_ker;
}

} // namespace pvfmm

#endif
