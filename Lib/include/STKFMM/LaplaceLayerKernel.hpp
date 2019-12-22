#ifndef LAPLACELAYERKERNEL_H_
#define LAPLACELAYERKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

// pvfmm headers
#include <pvfmm.hpp>

namespace pvfmm {

// Laplace potential + grdient kernel.
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void laplace_pgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
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

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t r3inv = mul_intrin(mul_intrin(rinv, rinv), rinv);

                sv = mul_intrin(sv, r3inv);
                tp = add_intrin(tp, mul_intrin(sv, r2));
                tv0 = add_intrin(tv0, mul_intrin(sv, dx));
                tv1 = add_intrin(tv1, mul_intrin(sv, dy));
                tv2 = add_intrin(tv2, mul_intrin(sv, dz));
            }
            tp = add_intrin(mul_intrin(tp, oofp), load_intrin<Vec_t>(&trg_value[0][t]));    // potential
            tv0 = add_intrin(mul_intrin(tv0, noofp), load_intrin<Vec_t>(&trg_value[1][t])); // gradient
            tv1 = add_intrin(mul_intrin(tv1, noofp), load_intrin<Vec_t>(&trg_value[2][t]));
            tv2 = add_intrin(mul_intrin(tv2, noofp), load_intrin<Vec_t>(&trg_value[3][t]));
            store_intrin(&trg_value[0][t], tp);
            store_intrin(&trg_value[1][t], tv0);
            store_intrin(&trg_value[2][t], tv1);
            store_intrin(&trg_value[3][t], tv2);
        }
    }

    { // Add FLOPS
#ifndef __MIC__
        Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (21 + 4 * (NWTN_ITER)));
#endif
    }
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void laplace_pgrad(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                   mem::MemoryManager *mem_mgr) {
#define LAP_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 1, 4, laplace_pgrad_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(             \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define LAPLACE_KERNEL                                                                                                 \
    LAP_KER_NWTN(0);                                                                                                   \
    LAP_KER_NWTN(1);                                                                                                   \
    LAP_KER_NWTN(2);                                                                                                   \
    LAP_KER_NWTN(3);

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
        LAPLACE_KERNEL;
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
        LAPLACE_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        LAPLACE_KERNEL;
#undef Vec_t
    }

#undef LAP_KER_NWTN
#undef LAPLACE_KERNEL
}

// Laplace dipole potential
// Laplace double layer potential.
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void laplace_dipolepotential_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
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

                Vec_t rinv = RSQRT_INTRIN(r2);
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

    { // Add FLOPS
#ifndef __MIC__
        Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (20 + 4 * (NWTN_ITER)));
#endif
    }
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void laplace_dipolepotential_poten(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                                   mem::MemoryManager *mem_mgr) {
#define LAP_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 3, 1, laplace_dipolepotential_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(   \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define LAPLACE_KERNEL                                                                                                 \
    LAP_KER_NWTN(0);                                                                                                   \
    LAP_KER_NWTN(1);                                                                                                   \
    LAP_KER_NWTN(2);                                                                                                   \
    LAP_KER_NWTN(3);

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
        LAPLACE_KERNEL;
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
        LAPLACE_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        LAPLACE_KERNEL;
#undef Vec_t
    }

#undef LAP_KER_NWTN
#undef LAPLACE_KERNEL
}

// Laplace dipole grad potential
// from dim 3 to dim 1 + 3
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void laplace_dipolepgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
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

                Vec_t rinv = RSQRT_INTRIN(r2);
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

    { // Add FLOPS
#ifndef __MIC__
        Profile::Add_FLOP((long long)trg_cnt_ * (long long)src_cnt_ * (26 + 4 * (NWTN_ITER)));
#endif
    }
#undef SRC_BLK
}

template <class T, int newton_iter = 0>
void laplace_dipolepgrad_poten(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                               mem::MemoryManager *mem_mgr) {
#define LAP_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 3, 4, laplace_dipolepgrad_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(       \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define LAPLACE_KERNEL                                                                                                 \
    LAP_KER_NWTN(0);                                                                                                   \
    LAP_KER_NWTN(1);                                                                                                   \
    LAP_KER_NWTN(2);                                                                                                   \
    LAP_KER_NWTN(3);

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
        LAPLACE_KERNEL;
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
        LAPLACE_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        LAPLACE_KERNEL;
#undef Vec_t
    }

#undef LAP_KER_NWTN
#undef LAPLACE_KERNEL
}

template <class T>
struct LaplaceLayerKernel {
    inline static const Kernel<T> &PGrad();

  private:
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
    // generate NEWTON_ITE at compile time. 1 for float and 2 for double
};

template <class T>
inline const Kernel<T> &LaplaceLayerKernel<T>::PGrad() {
    // S2U - single-layer density — to — potential kernel (1 x 1)
    // D2U - double-layer density — to — potential kernel (1+3 x 1)
    // S2UdU - single-layer density — to — potential & gradient (1 x 1)
    // D2UdU - double-layer density — to — potential & gradient (1+3 x 1)

    static Kernel<T> lap_pker =
        BuildKernel<T, laplace_poten<T, NEWTON_ITE>, laplace_dipolepotential_poten<T, NEWTON_ITE>>(
            "laplace", 3, std::pair<int, int>(1, 1));
    lap_pker.surf_dim = 3;

    static Kernel<T> lap_pgker = BuildKernel<T, laplace_pgrad<T, NEWTON_ITE>, laplace_dipolepgrad_poten<T, NEWTON_ITE>>(
        "laplace_PGrad", 3, std::pair<int, int>(1, 4), &lap_pker, &lap_pker, NULL, &lap_pker, &lap_pker, NULL,
        &lap_pker, NULL);
    lap_pgker.surf_dim = 3;

    return lap_pgker;
}

} // namespace pvfmm

#endif
