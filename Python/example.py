from __future__ import division, print_function
import numpy as np
import sys
try:
    from mpi4py import MPI
except ImportError:
    print('It didn\'t find mpi4py!')
import stkfmm 
import kernels as kr


def calc_true_value(kernel_index, src_SL_coord, trg_coord, src_SL_value):
    if kernel_index == 1:
        trg_value = kr.oseen_kernel_source_target_numba(src_SL_coord, trg_coord, src_SL_value)
    else:
        trg_value = None
    return trg_value


if __name__ == '__main__':
    print('# Start')

    # FMM parameters
    mult_order = 10
    max_pts = 1024
    pbc = stkfmm.PAXIS.NONE
    # kernels = [stkfmm.KERNEL.PVel, stkfmm.KERNEL.PVelGrad, stkfmm.KERNEL.PVelLaplacian, stkfmm.KERNEL.Traction, stkfmm.KERNEL.LAPPGrad]
    kernels = [stkfmm.KERNEL.PVel]
    kernels_index = [stkfmm.KERNEL(k) for k in kernels]
    verify = True

    # Get MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create sources and targets coordinates
    nsrc_SL = 1
    nsrc_DL = 1
    ntrg = 1
    src_SL_coord = np.random.rand(nsrc_SL, 3)
    src_DL_coord = np.random.rand(nsrc_DL, 3)
    trg_coord = np.random.rand(ntrg, 3)
    sys.stdout.flush()
    comm.Barrier()

    print('src_SL_coord = ', src_SL_coord)
    print('src_DL_coord = ', src_DL_coord)
    print('trg_coord = ', trg_coord)
    print(' ')
    
    # Try STKFMM
    for k, kernel in enumerate(kernels):
        # Create FMM
        print('\n\n==============================')
        myFMM = stkfmm.STKFMM(mult_order, max_pts, pbc, kernels_index[k])
        a = stkfmm.showActiveKernels(myFMM)
        kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(myFMM, kernel)

        # Create sources and target values
        src_SL_value = np.random.randn(nsrc_SL, kdimSL)
        src_DL_value = np.random.randn(nsrc_DL, kdimDL)
        trg_value = np.zeros((ntrg, kdimTrg))
        print('kdimSL = ', kdimSL)
        print('kdimDL = ', kdimDL)
        print('kdimTrg = ', kdimTrg)

        # Set tree
        stkfmm.setBox(myFMM, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        stkfmm.setPoints(myFMM, nsrc_SL, src_SL_coord, nsrc_DL, src_DL_coord, ntrg, trg_coord)
        stkfmm.setupTree(myFMM, kernel)

        # Evaluate FMM
        stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, kernel)
        print('trg_value = \n', trg_value)

        # Clear FMM and evaluate again
        trg_value[:,:] = 0
        stkfmm.clearFMM(myFMM, kernel)
        stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, kernel)
        print('trg_value = \n', trg_value)

        if verify:
            trg_value_true = calc_true_value(kernels_index[k], src_SL_coord, trg_coord, src_SL_value)
            diff = trg_value - trg_value_true
            print('relative L2 error = ', np.linalg.norm(diff) / np.linalg.norm(trg_value_true))
            print('Linf error        = ', np.linalg.norm(diff.flatten(), norm=np.inf))

    comm.Barrier()
    print('# End')
