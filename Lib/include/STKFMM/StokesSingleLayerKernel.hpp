/**
 * @file StokesSingleLayerKernel.hpp
 * @author Wen Yan (wenyan4work@gmail.com), Robert Blackwell (rblackwell@flatironinstitute.org)
 * @brief Stokes single layer kernels
 * @version 0.2
 * @date 2019-12-23, 2021-10-27
 *
 * @copyright Copyright (c) 2019, 2021
 *
 */
#ifndef STOKESSINGLELAYER_HPP_
#define STOKESSINGLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {

/*********************************************************
 *                                                        *
 *     Stokes P Vel kernel, source: 4, target: 4          *
 *                                                        *
 **********************************************************/
struct stokes_pvel : public GenericKernel<stokes_pvel> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[4], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType commonCoeff = f[0] * r[0] + f[1] * r[1] + f[2] * r[2];
        const VecType two = (typename VecType::ScalarType)(2.0);

        u[0] += two * commonCoeff * rinv3;
        commonCoeff -= f[3];
        u[1] += rinv3 * (r2 * f[0] + r[0] * commonCoeff);
        u[2] += rinv3 * (r2 * f[1] + r[1] * commonCoeff);
        u[3] += rinv3 * (r2 * f[2] + r[2] * commonCoeff);
    }
};

/*********************************************************
 *                                                        *
 *   Stokes P Vel Grad kernel, source: 4, target: 1+3+3+9 *
 *                                                        *
 **********************************************************/
struct stokes_pvelgrad : public GenericKernel<stokes_pvelgrad> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[16], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType nthree = (typename VecType::ScalarType)(-3.0);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &tr = f[3];
        // clang-format on

        VecType commonCoeff = fx * dx + fy * dy + fz * dz;
        u[0] += two * rinv3 * commonCoeff;

        commonCoeff -= tr;
        u[1] += rinv3 * (r2 * fx + commonCoeff * dx);
        u[2] += rinv3 * (r2 * fy + commonCoeff * dy);
        u[3] += rinv3 * (r2 * fz + commonCoeff * dz);

        // px dp/dx, etc
        commonCoeff += tr;
        u[4] += two * (r2 * fx + nthree * dx * commonCoeff) * rinv5;
        u[5] += two * (r2 * fy + nthree * dy * commonCoeff) * rinv5;
        u[6] += two * (r2 * fz + nthree * dz * commonCoeff) * rinv5;

        // qij = r^2 \delta_{ij} - 3 ri rj, symmetric
        VecType qxx = r2 + nthree * dx * dx;
        VecType qxy = nthree * dx * dy;
        VecType qxz = nthree * dx * dz;
        VecType qyy = r2 + nthree * dy * dy;
        VecType qyz = nthree * dy * dz;
        VecType qzz = r2 + nthree * dz * dz;

        // vxx = dvx/dx , etc
        commonCoeff -= tr;
        u[7] += qxx * commonCoeff * rinv5;
        u[8] += (qxy * commonCoeff + r2 * (dx * fy - dy * fx)) * rinv5;
        u[9] += (qxz * commonCoeff + r2 * (dx * fz - dz * fx)) * rinv5;

        u[10] += (qxy * commonCoeff + r2 * (dy * fx - dx * fy)) * rinv5;
        u[11] += qyy * commonCoeff * rinv5;
        u[12] += (qyz * commonCoeff + r2 * (dy * fz - dz * fy)) * rinv5;

        u[13] += (qxz * commonCoeff + r2 * (dz * fx - dx * fz)) * rinv5;
        u[14] += (qyz * commonCoeff + r2 * (dz * fy - dy * fz)) * rinv5;
        u[15] += qzz * commonCoeff * rinv5;
    }
};

/*********************************************************
 *                                                        *
 * Stokes P Vel Laplacian kernel, source: 4, target: 7      *
 *                                                        *
 **********************************************************/
struct stokes_pvellaplacian : public GenericKernel<stokes_pvellaplacian> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return 1.0 / (8.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[7], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType two = (typename VecType::ScalarType)(2.0);
        const VecType nthree = (typename VecType::ScalarType)(-3.0);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &tr = f[3];
        // clang-format on

        VecType commonCoeff = fx * dx + fy * dy + fz * dz;
        // pressure
        u[0] += two * rinv3 * commonCoeff;

        // velocity
        commonCoeff -= tr;
        u[1] += rinv3 * (r2 * fx + commonCoeff * dx);
        u[2] += rinv3 * (r2 * fy + commonCoeff * dy);
        u[3] += rinv3 * (r2 * fz + commonCoeff * dz);

        // laplacian
        commonCoeff = nthree * (commonCoeff + tr);
        u[4] += two * (fx * r2 + commonCoeff * dx) * rinv5;
        u[5] += two * (fy * r2 + commonCoeff * dy) * rinv5;
        u[6] += two * (fz * r2 + commonCoeff * dz) * rinv5;
    }
};

/*********************************************************
 *                                                        *
 *      Stokes traction kernel, source: 4, target: 9      *
 *                                                        *
 **********************************************************/
struct stokes_traction : public GenericKernel<stokes_traction> {
    static const int FLOPS = 20;
    template <class Real>
    static Real ScaleFactor() {
        return -3.0 / (4.0 * sctl::const_pi<Real>());
    }
    template <class VecType, int digits>
    static void uKerEval(VecType (&u)[9], const VecType (&r)[3], const VecType (&f)[4], const void *ctx_ptr) {
        VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
        VecType rinv3 = rinv * rinv * rinv;
        VecType rinv5 = rinv3 * rinv * rinv;
        const VecType invthree = (typename VecType::ScalarType)(0.3333333333333333333);

        // clang-format off
        const VecType &dx = r[0], &dy = r[1], &dz = r[2];
        const VecType &fx = f[0], &fy = f[1], &fz = f[2];
        const VecType &tr = f[3];
        // clang-format on

        VecType commonCoeff = (fx * dx + fy * dy + fz * dz - tr) * rinv5;
        VecType diag = tr * r2 * rinv5 * invthree;

        u[0] += dx * dx * commonCoeff + diag;
        u[1] += dx * dy * commonCoeff;
        u[2] += dx * dz * commonCoeff;
        u[3] += dy * dx * commonCoeff;
        u[4] += dy * dy * commonCoeff + diag;
        u[5] += dy * dz * commonCoeff;
        u[6] += dz * dx * commonCoeff;
        u[7] += dz * dy * commonCoeff;
        u[8] += dz * dz * commonCoeff + diag;
    }
};

} // namespace pvfmm
#endif // STOKESSINGLELAYERKERNEL_HPP
