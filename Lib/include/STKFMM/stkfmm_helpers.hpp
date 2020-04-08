#ifndef STKFMM_HELPERS_HPP
#define STKFMM_HELPERS_HPP

#include <pvfmm.hpp>

/**
 * @brief delete the pointer ptr if not null
 *
 * @tparam T
 * @param ptr
 */
template <class T>
void safeDeletePtr(T *&ptr) {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

/**
 * @brief return fraction part between [0,1)
 * This function is only applied in the PERIODIC DIRECTION
 * The user of the library must ensure that all points are located within [0,1)
 *
 * @param x
 * @return double
 */
inline void fracwrap(double &x) { x = x - floor(x); }

/**
 * @brief generate equivalent point coordinate
 *
 * @tparam Real_t
 * @param p
 * @param c
 * @param alpha
 * @param depth
 * @return std::vector<Real_t>
 */
template <class Real_t>
std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
    int n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

    std::vector<Real_t> coord(n_ * 3);
    coord[0] = coord[1] = coord[2] = -1.0;
    int cnt = 1;
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = -1.0;
            coord[cnt * 3 + 1] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = (2.0 * j - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * i - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = -1.0;
            coord[cnt * 3 + 2] = (2.0 * (j + 1) - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = (2.0 * j - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = -1.0;
            cnt++;
        }
    for (int i = 0; i < (n_ / 2) * 3; i++)
        coord[cnt * 3 + i] = -coord[i];

    Real_t r = 0.5 * pow(0.5, depth);
    Real_t b = alpha * r;
    for (int i = 0; i < n_; i++) {
        coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
        coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
        coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
    }
    return coord;
}

template <typename Vec_t, typename Real_t, int nwtn>
inline Vec_t rsqrt_wrapper(Vec_t r2) {
    switch (nwtn) {
    case 0:
        return pvfmm::rsqrt_intrin0<Vec_t, Real_t>(r2);
    case 1:
        return pvfmm::rsqrt_intrin1<Vec_t, Real_t>(r2);
    case 2:
        return pvfmm::rsqrt_intrin2<Vec_t, Real_t>(r2);
    case 3:
        return pvfmm::rsqrt_intrin3<Vec_t, Real_t>(r2);
    default:
        break;
    }
};

#if defined __MIC__
#define Vec_ts Real_t
#define Vec_td Real_t
#elif defined __AVX__
#define Vec_ts __m256
#define Vec_td __m256d
#elif defined __SSE3__
#define Vec_ts __m128
#define Vec_td __m128d
#else
#define Vec_ts Real_t
#define Vec_td Real_t
#endif

#define GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, VEC_T, REAL_T)          \
    generic_kernel<REAL_T, SRCDIM, TARDIM,                                     \
                   MICROKERNEL<REAL_T, VEC_T, newton_iter>>(                   \
        (REAL_T *)r_src, src_cnt, (REAL_T *)v_src, dof, (REAL_T *)r_trg,       \
        trg_cnt, (REAL_T *)v_trg, mem_mgr)

#define GEN_KERNEL(KERNEL, MICROKERNEL, SRCDIM, TARDIM)                        \
    template <class T, int newton_iter = 0>                                    \
    void KERNEL(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg,            \
                int trg_cnt, T *v_trg, mem::MemoryManager *mem_mgr) {          \
                                                                               \
        if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {        \
            typedef float Real_t;                                              \
            GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, Vec_ts, Real_t);    \
        } else if (mem::TypeTraits<T>::ID() ==                                 \
                   mem::TypeTraits<double>::ID()) {                            \
            typedef double Real_t;                                             \
            GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, Vec_td, Real_t);    \
        } else {                                                               \
            typedef T Real_t;                                                  \
            GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, Real_t, Real_t);    \
        }                                                                      \
    }

#endif
