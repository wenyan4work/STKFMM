#ifndef INCLUDE_RPYCUSTOMKERNEL_H_
#define INCLUDE_RPYCUSTOMKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "LaplaceLayerKernel.hpp"

namespace pvfmm {

/**********************************************************
 *                                                        *
 * RPY Force,a Vel kernel,source: 4, target: 3            *
 *       fx,fy,fz,a -> ux,uy,uz,                          *
 **********************************************************/
template <class T, int newton_iter = 0>
void rpy_u(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt,
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
            // TODO: s = t will obviously explode.
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
        }
        v_trg[3 * i + 0] += pi8 * vx;
        v_trg[3 * i + 1] += pi8 * vy;
        v_trg[3 * i + 2] += pi8 * vz;
    }
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
            // TODO: s = t will obviously explode.
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
