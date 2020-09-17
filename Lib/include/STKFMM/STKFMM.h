
typedef struct Stk3DFMM Stk3DFMM;
typedef struct StkWallFMM StkWallFMM;

Stk3DFMM *Stk3DFMM_create(int mult_order, int max_pts, int pbc, unsigned kernelComb);

void Stk3DFMM_destroy(Stk3DFMM *fmm);

void Stk3DFMM_set_points(Stk3DFMM *fmm, const int nSL, double *src_SL_coord, const int nTrg, double *trg_coord,
                         const int nDL, double *src_DL_coord);

void Stk3DFMM_set_box(Stk3DFMM *fmm, double *origin, double len);

void Stk3DFMM_setup_tree(Stk3DFMM *fmm, unsigned kernel);

void Stk3DFMM_evaluate_fmm(Stk3DFMM *fmm, unsigned kernel, const int nSL, double *src_SL_value, const int nTrg,
                           double *trg_value, const int nDL, double *src_DL_value);

void Stk3DFMM_show_active_kernels(Stk3DFMM *fmm);


StkWallFMM *StkWallFMM_create(int mult_order, int max_pts, int pbc, unsigned kernelComb);

void StkWallFMM_destroy(StkWallFMM *fmm);

void StkWallFMM_set_points(StkWallFMM *fmm, const int nSL, double *src_SL_coord, const int nTrg, double *trg_coord,
                         const int nDL, double *src_DL_coord);

void StkWallFMM_set_box(StkWallFMM *fmm, double *origin, double len);

void StkWallFMM_setup_tree(StkWallFMM *fmm, unsigned kernel);

void StkWallFMM_evaluate_fmm(StkWallFMM *fmm, unsigned kernel, const int nSL, double *src_SL_value, const int nTrg,
                           double *trg_value, const int nDL, double *src_DL_value);

void StkWallFMM_show_active_kernels(StkWallFMM *fmm);
