#ifndef STOKESREGSINGLELAYER_HPP_
#define STOKESREGSINGLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {

/*********************************************************
 *                                                        *
 *     Stokes Reg Vel kernel, source: 4, target: 3        *
 *              fx,fy,fz,eps -> ux,uy,uz                  *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_regvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                           Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t trgx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t trgy = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t trgz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(trgx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(trgy, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(trgz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t reg = bcast_intrin<Vec_t>(&src_value[3][s]); // reg parameter

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));
                r2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2+eps^2

                Vec_t r2reg2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2 + 2 eps^2

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2reg2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2reg2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2reg2, fz), mul_intrin(dz, commonCoeff)), rinv3));
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
}

/**********************************************************
 *                                                        *
 * Stokes Reg Force Torque Vel kernel,source: 7, target: 3*
 *       fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz                *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_regftvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                             Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facnwtn = set_intrin<Vec_t, Real_t>(1 / (nwtn_scal));

    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const Vec_t five = set_intrin<Vec_t, Real_t>(5.0);
    const Vec_t seven = set_intrin<Vec_t, Real_t>(7.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t trgx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t trgy = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t trgz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(trgx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(trgy, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(trgz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t tx = bcast_intrin<Vec_t>(&src_value[3][s]);
                const Vec_t ty = bcast_intrin<Vec_t>(&src_value[4][s]);
                const Vec_t tz = bcast_intrin<Vec_t>(&src_value[5][s]);
                const Vec_t eps = bcast_intrin<Vec_t>(&src_value[6][s]); // reg parameter

                // length squared of r
                Vec_t r2 = dx * dx + dy * dy + dz * dz;
                Vec_t r4 = r2 * r2;

                // regularization parameter squared
                Vec_t eps2 = eps * eps;
                Vec_t eps4 = eps2 * eps2;

                Vec_t denom_arg = eps2 + r2;
                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(denom_arg);
                rinv = rinv * facnwtn;

                Vec_t stokeslet_denom_inv = rinv * rinv * rinv;
                Vec_t rotlet_denom_inv = set_intrin<Vec_t, Real_t>(0.5) * stokeslet_denom_inv * rinv * rinv;
                Vec_t rotlet_coef = (two * r2 + five * eps2) * rotlet_denom_inv;
                Vec_t H2 = stokeslet_denom_inv;
                Vec_t H1 = (r2 + two * eps2) * H2;

                Vec_t tcurlrx = ty * dz - tz * dy;
                Vec_t tcurlry = tz * dx - tx * dz;
                Vec_t tcurlrz = tx * dy - ty * dx;

                Vec_t fdotr = fx * dx + fy * dy + fz * dz;
                Vec_t tdotr = tx * dx + ty * dy + tz * dz;

                vx += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
                vy += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
                vz += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
}

/**********************************************************
 *                                                        *
 *Stokes Reg Force Torque Vel Omega kernel, source: 7, target: 6*
 *    fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz,wx,wy,wz         *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_regftvelomega_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                  Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facnwtn = set_intrin<Vec_t, Real_t>(1 / (nwtn_scal));

    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const Vec_t five = set_intrin<Vec_t, Real_t>(5.0);
    const Vec_t seven = set_intrin<Vec_t, Real_t>(7.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t trgx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t trgy = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t trgz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz
            Vec_t wx = zero_intrin<Vec_t>(); // wx
            Vec_t wy = zero_intrin<Vec_t>(); // wy
            Vec_t wz = zero_intrin<Vec_t>(); // wz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(trgx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(trgy, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(trgz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t tx = bcast_intrin<Vec_t>(&src_value[3][s]);
                const Vec_t ty = bcast_intrin<Vec_t>(&src_value[4][s]);
                const Vec_t tz = bcast_intrin<Vec_t>(&src_value[5][s]);
                const Vec_t eps = bcast_intrin<Vec_t>(&src_value[6][s]); // reg parameter

                // length squared of r
                Vec_t r2 = dx * dx + dy * dy + dz * dz;
                Vec_t r4 = r2 * r2;

                // regularization parameter squared
                Vec_t eps2 = eps * eps;
                Vec_t eps4 = eps2 * eps2;

                Vec_t denom_arg = eps2 + r2;
                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(denom_arg);
                rinv = rinv * facnwtn;

                // Vec_t stokeslet_denom = pi8 * denom_arg * std::sqrt(denom_arg);
                // Vec_t rotlet_denom = 2 * stokeslet_denom * denom_arg;
                // Vec_t dipole_denom = 2 * rotlet_denom * denom_arg;
                // Vec_t rotlet_coef = (2 * r2 + 5.0 * eps2) / rotlet_denom;
                // Vec_t D1 = (10 * eps4 - 7 * eps2 * r2 - 2 * r4) / dipole_denom;
                // Vec_t D2 = (21 * eps2 + 6 * r2) / dipole_denom;
                // Vec_t H2 = 1.0 / stokeslet_denom;
                // Vec_t H1 = (r2 + 2.0 * eps2) * H2;
                Vec_t stokeslet_denom_inv = rinv * rinv * rinv;
                Vec_t rotlet_denom_inv = set_intrin<Vec_t, Real_t>(0.5) * stokeslet_denom_inv * rinv * rinv;
                Vec_t dipole_denom_inv = set_intrin<Vec_t, Real_t>(0.5) * rotlet_denom_inv * rinv * rinv;
                Vec_t rotlet_coef = (two * r2 + five * eps2) * rotlet_denom_inv;
                Vec_t D1 = (two * five * eps4 - seven * eps2 * r2 - two * r4) * dipole_denom_inv;
                Vec_t D2 = (seven * three * eps2 + two * three * r2) * dipole_denom_inv;
                Vec_t H2 = stokeslet_denom_inv;
                Vec_t H1 = (r2 + two * eps2) * H2;

                Vec_t fcurlrx = fy * dz - fz * dy;
                Vec_t fcurlry = fz * dx - fx * dz;
                Vec_t fcurlrz = fx * dy - fy * dx;

                Vec_t tcurlrx = ty * dz - tz * dy;
                Vec_t tcurlry = tz * dx - tx * dz;
                Vec_t tcurlrz = tx * dy - ty * dx;

                Vec_t fdotr = fx * dx + fy * dy + fz * dz;
                Vec_t tdotr = tx * dx + ty * dy + tz * dz;

                vx += H1 * fx + H2 * fdotr * dx + rotlet_coef * tcurlrx;
                vy += H1 * fy + H2 * fdotr * dy + rotlet_coef * tcurlry;
                vz += H1 * fz + H2 * fdotr * dz + rotlet_coef * tcurlrz;

                wx += D1 * tx + D2 * tdotr * dx + rotlet_coef * fcurlrx;
                wy += D1 * ty + D2 * tdotr * dy + rotlet_coef * fcurlry;
                wz += D1 * tz + D2 * tdotr * dz + rotlet_coef * fcurlrz;
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));
            wx = add_intrin(mul_intrin(wx, facv), load_intrin<Vec_t>(&trg_value[3][t]));
            wy = add_intrin(mul_intrin(wy, facv), load_intrin<Vec_t>(&trg_value[4][t]));
            wz = add_intrin(mul_intrin(wz, facv), load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], wx);
            store_intrin(&trg_value[4][t], wy);
            store_intrin(&trg_value[5][t], wz);
        }
    }
}

/**********************************************************
 *                                                         *
 *   Stokes Force Vel Omega kernel, source: 3, target: 6   *
 *           fx,fy,fz -> ux,uy,uz, wx,wy,wz                *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_velomega_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                             Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facnwtn = set_intrin<Vec_t, Real_t>(1 / (nwtn_scal));

    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const Vec_t five = set_intrin<Vec_t, Real_t>(5.0);
    const Vec_t seven = set_intrin<Vec_t, Real_t>(7.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t trgx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t trgy = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t trgz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz
            Vec_t wx = zero_intrin<Vec_t>(); // wx
            Vec_t wy = zero_intrin<Vec_t>(); // wy
            Vec_t wz = zero_intrin<Vec_t>(); // wz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(trgx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(trgy, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(trgz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);

                // length squared of r
                Vec_t r2 = dx * dx + dy * dy + dz * dz;
                Vec_t r4 = r2 * r2;

                Vec_t denom_arg = r2;
                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(denom_arg);
                rinv = rinv * facnwtn;

                Vec_t stokeslet_denom_inv = rinv * rinv * rinv;
                Vec_t rotlet_denom_inv = set_intrin<Vec_t, Real_t>(0.5) * stokeslet_denom_inv * rinv * rinv;
                Vec_t rotlet_coef = (two * r2) * rotlet_denom_inv;
                Vec_t H2 = stokeslet_denom_inv;
                Vec_t H1 = (r2)*H2;

                Vec_t fcurlrx = fy * dz - fz * dy;
                Vec_t fcurlry = fz * dx - fx * dz;
                Vec_t fcurlrz = fx * dy - fy * dx;

                Vec_t fdotr = fx * dx + fy * dy + fz * dz;

                vx += H1 * fx + H2 * fdotr * dx;
                vy += H1 * fy + H2 * fdotr * dy;
                vz += H1 * fz + H2 * fdotr * dz;

                wx += rotlet_coef * fcurlrx;
                wy += rotlet_coef * fcurlry;
                wz += rotlet_coef * fcurlrz;
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));
            wx = add_intrin(mul_intrin(wx, facv), load_intrin<Vec_t>(&trg_value[3][t]));
            wy = add_intrin(mul_intrin(wy, facv), load_intrin<Vec_t>(&trg_value[4][t]));
            wz = add_intrin(mul_intrin(wz, facv), load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], wx);
            store_intrin(&trg_value[4][t], wy);
            store_intrin(&trg_value[5][t], wz);
        }
    }
}

GEN_KERNEL(stokes_regvel, stokes_regvel_uKernel, 4, 3)
GEN_KERNEL(stokes_velomega, stokes_velomega_uKernel, 3, 6)
GEN_KERNEL(stokes_regftvel, stokes_regftvel_uKernel, 7, 3)
GEN_KERNEL(stokes_regftvelomega, stokes_regftvelomega_uKernel, 7, 6)

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
    static Kernel<T> s2t_ker =
        BuildKernel<T, stokes_regvel<T, NEWTON_ITE>>("stokes_regvel", 3, std::pair<int, int>(4, 3), NULL, NULL, NULL,
                                                     &stk_ker, &stk_ker, &stk_ker, &stk_ker, &stk_ker, NULL, true);

    return s2t_ker;
}

template <class T>
inline const Kernel<T> &StokesRegKernel<T>::FTVelOmega() {
    static Kernel<T> stk_ker = StokesKernel<T>::velocity();
    static Kernel<T> stk_velomega =
        BuildKernel<T, stokes_velomega<T, NEWTON_ITE>>("stokes_velomega", 3, std::pair<int, int>(3, 6));
    static Kernel<T> stk_regftvel =
        BuildKernel<T, stokes_regftvel<T, NEWTON_ITE>>("stokes_regftvel", 3, std::pair<int, int>(7, 3));
    static Kernel<T> s2t_ker = BuildKernel<T, stokes_regftvelomega<T, NEWTON_ITE>>(
        "stokes_regftvelomega", 3, std::pair<int, int>(7, 6), &stk_regftvel, &stk_regftvel, NULL, &stk_ker, &stk_ker,
        &stk_velomega, &stk_ker, &stk_velomega);

    return s2t_ker;
}

} // namespace pvfmm
#endif // STOKESSINGLELAYERKERNEL_HPP
