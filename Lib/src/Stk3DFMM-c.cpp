#include "STKFMM/STKFMM.hpp"

extern "C" {
    using namespace stkfmm;
    Stk3DFMM *Stk3DFMM_create(int mult_order, int max_pts, int pbc, unsigned kernelComb) {
        return new Stk3DFMM(mult_order, max_pts, static_cast<PAXIS>(pbc), kernelComb);
    }

    void Stk3DFMM_destroy(Stk3DFMM *fmm) {
        delete fmm;
    }

    void Stk3DFMM_set_points(Stk3DFMM *fmm, const int nSL, double *src_SL_coord, const int nTrg, double *trg_coord,
                             const int nDL, double *src_DL_coord) {
        fmm->setPoints(nSL, src_SL_coord, nTrg, trg_coord, nDL, src_DL_coord);
    }

    void Stk3DFMM_get_kernel_dimension(unsigned kernel, int *dims) {
        std::tie(dims[0], dims[1], dims[2]) = getKernelDimension(static_cast<KERNEL>(kernel));
    }

    void Stk3DFMM_set_box(Stk3DFMM *fmm, double *origin, double len) { fmm->setBox(origin, len); }

    void Stk3DFMM_setup_tree(Stk3DFMM *fmm, unsigned kernel) { fmm->setupTree(static_cast<KERNEL>(kernel)); }

    void Stk3DFMM_clear_fmm(Stk3DFMM *fmm, unsigned kernel) { fmm->clearFMM(static_cast<KERNEL>(kernel)); }

    void Stk3DFMM_evaluate_fmm(Stk3DFMM *fmm, unsigned kernel, const int nSL, double *src_SL_value, const int nTrg,
                               double *trg_value, const int nDL, double *src_DL_value) {
        fmm->evaluateFMM(static_cast<KERNEL>(kernel), nSL, src_SL_value, nTrg, trg_value, nDL, src_DL_value);
    }

    void Stk3DFMM_show_active_kernels(Stk3DFMM *fmm) {
        fmm->showActiveKernels();
    }
}
