#ifndef STKFMM_HELPERS_HPP
#define STKFMM_HELPERS_HPP

#include <pvfmm.hpp>
#include <intrin_wrapper.hpp>

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
