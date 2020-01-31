#ifndef INCLUDE_RPYCUSTOMKERNEL_H_
#define INCLUDE_RPYCUSTOMKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "LaplaceLayerKernel.hpp"

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

#define SRC_BLK 1
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
    const Real_t FACV = 1.0 / (8 * const_pi<Real_t>());

    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);

    const Vec_t one_over_three =
        set_intrin<Vec_t, Real_t>(static_cast<Real_t>(1.0 / 3.0));
    const Vec_t three = set_intrin<Vec_t, Real_t>(static_cast<Real_t>(3.0));

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

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx =
                    sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy =
                    sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz =
                    sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t a = bcast_intrin<Vec_t>(&src_value[3][s]);

                Vec_t a2_over_3 = mul_intrin(mul_intrin(a, a), one_over_three);
                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = RSQRT_INTRIN(r2);
                rinv = mul_intrin(rinv, nwtn_factor);

                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t rinv5 = mul_intrin(mul_intrin(rinv3, rinv), rinv);

                Vec_t fdotr = mul_intrin(fx, dx);
                fdotr = add_intrin(fdotr, mul_intrin(fy, dy));
                fdotr = add_intrin(fdotr, mul_intrin(fz, dz));

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

                Vec_t t1, t2;
                t1 = mul_intrin(fx, rinv3);
                t2 =
                    mul_intrin(three, mul_intrin(mul_intrin(fdotr, dx), rinv5));
                vx = add_intrin(vx, mul_intrin(a2_over_3, sub_intrin(t1, t2)));

                t1 = mul_intrin(fy, rinv3);
                t2 =
                    mul_intrin(three, mul_intrin(mul_intrin(fdotr, dy), rinv5));
                vy = add_intrin(vy, mul_intrin(a2_over_3, sub_intrin(t1, t2)));

                t1 = mul_intrin(fz, rinv3);
                t2 =
                    mul_intrin(three, mul_intrin(mul_intrin(fdotr, dz), rinv5));
                vz = add_intrin(vz, mul_intrin(a2_over_3, sub_intrin(t1, t2)));
            }

            vx = add_intrin(mul_intrin(vx, facv),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv),
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
template <class T, int newton_iter = 0>
void rpy_ulapu(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt,
               T *v_trg, mem::MemoryManager *mem_mgr) {
    constexpr T pi8 = 1 / (8 * 3.14159265358979323846);
    for (int i = 0; i < trg_cnt; i++) {
        T vx = 0, vy = 0, vz = 0;
        T lapvx = 0, lapvy = 0, lapvz = 0;
        const T trgx = r_trg[3 * i];
        const T trgy = r_trg[3 * i + 1];
        const T trgz = r_trg[3 * i + 2];

        for (int j = 0; j < src_cnt; j++) {
            const T fx = v_src[4 * j + 0];
            const T fy = v_src[4 * j + 1];
            const T fz = v_src[4 * j + 2];
            const T a = v_src[4 * j + 3];

            const T sx = r_src[3 * j + 0];
            const T sy = r_src[3 * j + 1];
            const T sz = r_src[3 * j + 2];
            const T dx = trgx - sx;
            const T dy = trgy - sy;
            const T dz = trgz - sz;

            T r2 = dx * dx + dy * dy + dz * dz;
            T a2 = a * a;

            if (r2 == 0.0)
                continue;

            T invr = 1.0 / sqrt(r2);
            T invr3 = invr / r2;
            T invr5 = invr3 / r2;
            T fdotr = fx * dx + fy * dy + fz * dz;
            vx += fx * invr + dx * fdotr * invr3;
            vy += fy * invr + dy * fdotr * invr3;
            vz += fz * invr + dz * fdotr * invr3;

            vx += a2 * (2 * fx * invr3 - 6 * fdotr * dx * invr5) / 6.0;
            vy += a2 * (2 * fy * invr3 - 6 * fdotr * dy * invr5) / 6.0;
            vz += a2 * (2 * fz * invr3 - 6 * fdotr * dz * invr5) / 6.0;

            lapvx += 2 * fx * invr3 - 6 * fdotr * dx * invr5;
            lapvy += 2 * fy * invr3 - 6 * fdotr * dy * invr5;
            lapvz += 2 * fz * invr3 - 6 * fdotr * dz * invr5;
        }
        v_trg[6 * i + 0] += pi8 * vx;
        v_trg[6 * i + 1] += pi8 * vy;
        v_trg[6 * i + 2] += pi8 * vz;
        v_trg[6 * i + 3] += pi8 * lapvx;
        v_trg[6 * i + 4] += pi8 * lapvy;
        v_trg[6 * i + 5] += pi8 * lapvz;
    }
}

/**********************************************************
 *                                                        *
 * Stokes Force-Vel,Lap(Vel) kernel,source: 3, target: 6  *
 *       fx,fy,fz -> ux,uy,uz,lapux,lapuy,lapuz           *
 **********************************************************/
template <class T, int newton_iter = 0>
void stk_ulapu(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt,
               T *v_trg, mem::MemoryManager *mem_mgr) {
    constexpr T pi8 = 1 / (8 * 3.14159265358979323846);
    for (int i = 0; i < trg_cnt; i++) {
        T vx = 0, vy = 0, vz = 0;
        T lapvx = 0, lapvy = 0, lapvz = 0;
        const T trgx = r_trg[3 * i];
        const T trgy = r_trg[3 * i + 1];
        const T trgz = r_trg[3 * i + 2];

        for (int j = 0; j < src_cnt; j++) {
            const T fx = v_src[3 * j + 0];
            const T fy = v_src[3 * j + 1];
            const T fz = v_src[3 * j + 2];

            const T sx = r_src[3 * j + 0];
            const T sy = r_src[3 * j + 1];
            const T sz = r_src[3 * j + 2];
            const T dx = trgx - sx;
            const T dy = trgy - sy;
            const T dz = trgz - sz;

            T r2 = dx * dx + dy * dy + dz * dz;

            T invr = 1.0 / sqrt(r2);
            T invr3 = invr / r2;
            T invr5 = invr3 / r2;
            T fdotr = fx * dx + fy * dy + fz * dz;
            vx += fx * invr + dx * fdotr * invr3;
            vy += fy * invr + dy * fdotr * invr3;
            vz += fz * invr + dz * fdotr * invr3;

            lapvx += 2 * fx * invr3 - 6 * fdotr * dx * invr5;
            lapvy += 2 * fy * invr3 - 6 * fdotr * dy * invr5;
            lapvz += 2 * fz * invr3 - 6 * fdotr * dz * invr5;
        }
        v_trg[6 * i + 0] += pi8 * vx;
        v_trg[6 * i + 1] += pi8 * vy;
        v_trg[6 * i + 2] += pi8 * vz;
        v_trg[6 * i + 3] += pi8 * lapvx;
        v_trg[6 * i + 4] += pi8 * lapvy;
        v_trg[6 * i + 5] += pi8 * lapvz;
    }
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
