#ifndef STOKESREGSINGLELAYER_HPP_
#define STOKESREGSINGLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

// pvfmm headers
#include <pvfmm.hpp>

namespace pvfmm {
// TODO: vectorize these kernels
// Stokes Reg Force Torque Vel kernel, 7 -> 3
// Stokes Reg Force Torque Vel Omega kernel, 7 -> 6
// Stokes Force Vel Omega kernel, 3 -> 6

/*********************************************************
 *                                                        *
 *     Stokes Reg Vel kernel, source: 4, target: 3        *
 *              fx,fy,fz,eps -> ux,uy,uz                  *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t,
          Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_regvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                           Matrix<Real_t> &trg_coord,
                           Matrix<Real_t> &trg_value) {

#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    // Number of newton iterations
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
    const Real_t FACV =
        1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);

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
                const Vec_t reg =
                    bcast_intrin<Vec_t>(&src_value[3][s]); // reg parameter

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));
                r2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2+eps^2

                Vec_t r2reg2 =
                    add_intrin(r2, mul_intrin(reg, reg)); // r^2 + 2 eps^2

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                vx = add_intrin(
                    vx, mul_intrin(add_intrin(mul_intrin(r2reg2, fx),
                                              mul_intrin(dx, commonCoeff)),
                                   rinv3));
                vy = add_intrin(
                    vy, mul_intrin(add_intrin(mul_intrin(r2reg2, fy),
                                              mul_intrin(dy, commonCoeff)),
                                   rinv3));
                vz = add_intrin(
                    vz, mul_intrin(add_intrin(mul_intrin(r2reg2, fz),
                                              mul_intrin(dz, commonCoeff)),
                                   rinv3));
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
void stokes_regvel(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg,
                   int trg_cnt, T *v_trg, mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                     \
    if (newton_iter == nwtn)                                                   \
    generic_kernel<Real_t, 4, 3,                                               \
                   stokes_regvel_uKernel<Real_t, Vec_t,                        \
                                         rsqrt_intrin##nwtn<Vec_t, Real_t>>>(  \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg,       \
        trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                          \
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

/**********************************************************
 *                                                        *
 * Stokes Reg Force Torque Vel kernel,source: 7, target: 3*
 *       fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz                *
 **********************************************************/
template <class T, int newton_iter = 0>
void stokes_regftvel(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg,
                     int trg_cnt, T *v_trg, mem::MemoryManager *mem_mgr) {
    constexpr T pi8 = (8 * 3.14159265358979323846);
    for (int i = 0; i < trg_cnt; i++) {
        T vx = 0, vy = 0, vz = 0;
        const T trgx = r_trg[3 * i];
        const T trgy = r_trg[3 * i + 1];
        const T trgz = r_trg[3 * i + 2];
        for (int j = 0; j < src_cnt; j++) {
            const T fx = v_src[7 * j + 0];
            const T fy = v_src[7 * j + 1];
            const T fz = v_src[7 * j + 2];
            const T tx = v_src[7 * j + 3];
            const T ty = v_src[7 * j + 4];
            const T tz = v_src[7 * j + 5];
            const T eps = v_src[7 * j + 6];

            const T sx = r_src[3 * j];
            const T sy = r_src[3 * j + 1];
            const T sz = r_src[3 * j + 2];
            const T dx = trgx - sx;
            const T dy = trgy - sy;
            const T dz = trgz - sz;
            // length squared of r
            T r2 = dx * dx + dy * dy + dz * dz;

            // regularization parameter squared
            T eps2 = eps * eps;

            T denom_arg = eps2 + r2;
            T stokeslet_denom = pi8 * denom_arg * std::sqrt(denom_arg);
            T rotlet_denom = 2 * stokeslet_denom * denom_arg;
            T rotlet_coef = (2 * r2 + 5.0 * eps2) / rotlet_denom;
            // T D1 = (10 * eps4 - 7 * eps2 * r2 - 2 * r4) / dipole_denom;
            // T D2 = (21 * eps2 + 6 * r2) / dipole_denom;
            T H2 = 1.0 / stokeslet_denom;
            T H1 = (r2 + 2.0 * eps2) * H2;

            T tcurlrx = ty * dz - tz * dy;
            T tcurlry = tz * dx - tx * dz;
            T tcurlrz = tx * dy - ty * dx;

            T fdotr = fx * dx + fy * dy + fz * dz;
            // T tdotr = tx * dx + ty * dy + tz * dz;

            vx += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
            vy += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
            vz += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;
        }
        v_trg[3 * i] += vx;
        v_trg[3 * i + 1] += vy;
        v_trg[3 * i + 2] += vz;
    }
}

/**********************************************************
 *                                                        *
 *Stokes Reg Force Torque Vel Omega kernel, source: 7, target: 6*
 *    fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz, wx,wy,wz         *
 **********************************************************/
template <class T, int newton_iter = 0>
void stokes_regftvelomega(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg,
                          int trg_cnt, T *v_trg, mem::MemoryManager *mem_mgr) {
    constexpr T pi8 = (8 * 3.14159265358979323846);
    for (int i = 0; i < trg_cnt; i++) {
        T vx = 0, vy = 0, vz = 0, wx = 0, wy = 0, wz = 0;
        const T trgx = r_trg[3 * i];
        const T trgy = r_trg[3 * i + 1];
        const T trgz = r_trg[3 * i + 2];
        for (int j = 0; j < src_cnt; j++) {
            const T fx = v_src[7 * j + 0];
            const T fy = v_src[7 * j + 1];
            const T fz = v_src[7 * j + 2];
            const T tx = v_src[7 * j + 3];
            const T ty = v_src[7 * j + 4];
            const T tz = v_src[7 * j + 5];
            const T eps = v_src[7 * j + 6];

            const T sx = r_src[3 * j];
            const T sy = r_src[3 * j + 1];
            const T sz = r_src[3 * j + 2];
            const T dx = trgx - sx;
            const T dy = trgy - sy;
            const T dz = trgz - sz;
            // length squared of r
            T r2 = dx * dx + dy * dy + dz * dz;
            T r4 = r2 * r2;

            // regularization parameter squared
            T eps2 = eps * eps;
            T eps4 = eps2 * eps2;

            T denom_arg = eps2 + r2;
            T stokeslet_denom = pi8 * denom_arg * std::sqrt(denom_arg);
            T rotlet_denom = 2 * stokeslet_denom * denom_arg;
            T dipole_denom = 2 * rotlet_denom * denom_arg;
            T rotlet_coef = (2 * r2 + 5.0 * eps2) / rotlet_denom;
            T D1 = (10 * eps4 - 7 * eps2 * r2 - 2 * r4) / dipole_denom;
            T D2 = (21 * eps2 + 6 * r2) / dipole_denom;
            T H2 = 1.0 / stokeslet_denom;
            T H1 = (r2 + 2.0 * eps2) * H2;

            T fcurlrx = fy * dz - fz * dy;
            T fcurlry = fz * dx - fx * dz;
            T fcurlrz = fx * dy - fy * dx;

            T tcurlrx = ty * dz - tz * dy;
            T tcurlry = tz * dx - tx * dz;
            T tcurlrz = tx * dy - ty * dx;

            T fdotr = fx * dx + fy * dy + fz * dz;
            T tdotr = tx * dx + ty * dy + tz * dz;

            vx += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
            vy += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
            vz += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;

            wx += D1 * tx + D2 * tdotr * dx + rotlet_coef * fcurlrx;
            wy += D1 * ty + D2 * tdotr * dy + rotlet_coef * fcurlry;
            wz += D1 * tz + D2 * tdotr * dz + rotlet_coef * fcurlrz;
        }
        v_trg[6 * i] += vx;
        v_trg[6 * i + 1] += vy;
        v_trg[6 * i + 2] += vz;
        v_trg[6 * i + 3] += wx;
        v_trg[6 * i + 4] += wy;
        v_trg[6 * i + 5] += wz;
    }
}

/**********************************************************
 *                                                         *
 *   Stokes Force Vel Omega kernel, source: 3, target: 6   *
 *           fx,fy,fz -> ux,uy,uz, wx,wy,wz                *
 **********************************************************/
template <class T, int newton_iter = 0>
void stokes_velomega(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg,
                     int trg_cnt, T *v_trg, mem::MemoryManager *mem_mgr) {
    constexpr T pi8 = (8 * 3.14159265358979323846);
    for (int i = 0; i < trg_cnt; i++) {
        T vx = 0, vy = 0, vz = 0, wx = 0, wy = 0, wz = 0;
        const T trgx = r_trg[3 * i];
        const T trgy = r_trg[3 * i + 1];
        const T trgz = r_trg[3 * i + 2];
        for (int j = 0; j < src_cnt; j++) {
            const T fx = v_src[3 * j + 0];
            const T fy = v_src[3 * j + 1];
            const T fz = v_src[3 * j + 2];

            const T sx = r_src[3 * j];
            const T sy = r_src[3 * j + 1];
            const T sz = r_src[3 * j + 2];
            const T dx = trgx - sx;
            const T dy = trgy - sy;
            const T dz = trgz - sz;
            // length squared of r
            T r2 = dx * dx + dy * dy + dz * dz;
            T r4 = r2 * r2;

            constexpr T eps2 = 0; // will be optimized out
            constexpr T tx = 0;
            constexpr T ty = 0;
            constexpr T tz = 0;
            T eps4 = eps2 * eps2;

            T denom_arg = eps2 + r2;
            T stokeslet_denom = pi8 * denom_arg * std::sqrt(denom_arg);
            T rotlet_denom = 2 * stokeslet_denom * denom_arg;
            T dipole_denom = 2 * rotlet_denom * denom_arg;
            T rotlet_coef = (2 * r2 + 5.0 * eps2) / rotlet_denom;
            T D1 = (10 * eps4 - 7 * eps2 * r2 - 2 * r4) / dipole_denom;
            T D2 = (21 * eps2 + 6 * r2) / dipole_denom;
            T H2 = 1.0 / stokeslet_denom;
            T H1 = (r2 + 2.0 * eps2) * H2;

            T fcurlrx = fy * dz - fz * dy;
            T fcurlry = fz * dx - fx * dz;
            T fcurlrz = fx * dy - fy * dx;

            T tcurlrx = ty * dz - tz * dy;
            T tcurlry = tz * dx - tx * dz;
            T tcurlrz = tx * dy - ty * dx;

            T fdotr = fx * dx + fy * dy + fz * dz;
            T tdotr = tx * dx + ty * dy + tz * dz;

            vx += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
            vy += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
            vz += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;

            wx += D1 * tx + D2 * tdotr * dx + rotlet_coef * fcurlrx;
            wy += D1 * ty + D2 * tdotr * dy + rotlet_coef * fcurlry;
            wz += D1 * tz + D2 * tdotr * dz + rotlet_coef * fcurlrz;
        }
        v_trg[6 * i + 0] += vx;
        v_trg[6 * i + 1] += vy;
        v_trg[6 * i + 2] += vz;
        v_trg[6 * i + 3] += wx;
        v_trg[6 * i + 4] += wy;
        v_trg[6 * i + 5] += wz;
    }
}

template <class T>
struct StokesRegKernel {
    inline static const Kernel<T> &Vel();        //   3+1->3
    inline static const Kernel<T> &FTVelOmega(); //   3+3+1->3+3
  private:
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
};

// 1 newton for float, 2 newton for double
// the string for stk_ker must be exactly the same as in kernel.txx of pvfmm
template <class T>
inline const Kernel<T> &StokesRegKernel<T>::Vel() {
    static Kernel<T> stk_ker = StokesKernel<T>::velocity();
    static Kernel<T> s2t_ker = BuildKernel<T, stokes_regvel<T, NEWTON_ITE>>(
        "stokes_regvel", 3, std::pair<int, int>(4, 3), NULL, NULL, NULL,
        &stk_ker, &stk_ker, &stk_ker, &stk_ker, &stk_ker, NULL, true);

    return s2t_ker;
}

template <class T>
inline const Kernel<T> &StokesRegKernel<T>::FTVelOmega() {
    static Kernel<T> stk_ker = StokesKernel<T>::velocity();
    static Kernel<T> stk_velomega =
        BuildKernel<T, stokes_velomega<T, NEWTON_ITE>>(
            "stokes_velomega", 3, std::pair<int, int>(3, 6));
    static Kernel<T> stk_regftvel =
        BuildKernel<T, stokes_regftvel<T, NEWTON_ITE>>(
            "stokes_regvelomega", 3, std::pair<int, int>(7, 3));
    static Kernel<T> s2t_ker =
        BuildKernel<T, stokes_regftvelomega<T, NEWTON_ITE>>(
            "stokes_regvel", 3, std::pair<int, int>(7, 6), &stk_regftvel,
            &stk_regftvel, NULL, &stk_ker, &stk_ker, &stk_velomega, &stk_ker,
            &stk_velomega);

    return s2t_ker;
}

} // namespace pvfmm
#endif // STOKESSINGLELAYERKERNEL_HPP
