from __future__ import division, print_function
import numpy as np
import sys
try:
    from mpi4py import MPI
except ImportError:
    print('It didn\'t find mpi4py!')
import stkfmm 
import kernels as kr
import timer


def calc_true_value(kernel_index, src_SL_coord, trg_coord, src_SL_value):
    if kernel_index == 1:
        trg_value  = kr.StokesSLPVel(src_SL_coord, trg_coord, src_SL_value)
        trg_value += kr.StokesDLPVel(src_DL_coord, trg_coord, src_DL_value)
    elif kernel_index == 2:
        trg_value  = kr.StokesSLPVelGrad(src_SL_coord, trg_coord, src_SL_value)
        trg_value += kr.StokesDLPVelGrad(src_DL_coord, trg_coord, src_DL_value)
    elif kernel_index == 4:
        trg_value  = kr.StokesSLPVelLaplacian(src_SL_coord, trg_coord, src_SL_value)
        trg_value += kr.StokesDLPVelLaplacian(src_DL_coord, trg_coord, src_DL_value)
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
    kernels = [stkfmm.KERNEL.PVel, stkfmm.KERNEL.PVelGrad, stkfmm.KERNEL.PVelLaplacian]
    kernels_index = [stkfmm.KERNEL(k) for k in kernels]
    verify = True

    # Get MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create sources and targets coordinates
    nsrc_SL = 10
    nsrc_DL = 10
    ntrg = 10
    src_SL_coord = np.random.rand(nsrc_SL, 3)
    src_DL_coord = np.random.rand(nsrc_DL, 3)
    trg_coord = np.random.rand(ntrg, 3)
    sys.stdout.flush()
    comm.Barrier()
    
    # Try STKFMM
    for k, kernel in enumerate(kernels):
        # Create FMM
        print('\n\n==============================')
        timer.timer('create_fmm')
        myFMM = stkfmm.STKFMM(mult_order, max_pts, pbc, kernels_index[k])
        timer.timer('create_fmm')
        timer.timer('show_active_kernels')
        stkfmm.showActiveKernels(myFMM)
        timer.timer('show_active_kernels')
        timer.timer('get_kernel_dimension')
        kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(myFMM, kernel)
        timer.timer('get_kernel_dimension')

        # Create sources and target values
        src_SL_value = np.random.randn(nsrc_SL, kdimSL) * 1
        src_DL_value = np.random.randn(nsrc_DL, kdimDL) * 0
        trg_value = np.zeros((ntrg, kdimTrg))
        print('kdimSL = ', kdimSL)
        print('kdimDL = ', kdimDL)
        print('kdimTrg = ', kdimTrg)

        # Set tree
        timer.timer('set_box')
        stkfmm.setBox(myFMM, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        timer.timer('set_box')
        timer.timer('set_points')
        stkfmm.setPoints(myFMM, nsrc_SL, src_SL_coord, nsrc_DL, src_DL_coord, ntrg, trg_coord)
        timer.timer('set_points')
        timer.timer('setup_tree')
        stkfmm.setupTree(myFMM, kernel)
        timer.timer('setup_tree')

        # Evaluate FMM
        timer.timer('evaluate_fmm')
        stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, kernel)
        timer.timer('evaluate_fmm')

        # Clear FMM and evaluate again
        trg_value[:,:] = 0
        timer.timer('clear_fmm')
        stkfmm.clearFMM(myFMM, kernel)
        timer.timer('clear_fmm')
        timer.timer('evaluate_fmm')
        stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, kernel)
        timer.timer('evaluate_fmm')

        if verify:
            timer.timer('true_value')
            trg_value_true  = calc_true_value(kernels_index[k], src_SL_coord, trg_coord, src_SL_value)
            timer.timer('true_value')
            if trg_value_true is not None:
                diff = trg_value - trg_value_true
                print('diff = \n', diff)
                print('relative L2 error = ', np.linalg.norm(diff) / np.linalg.norm(trg_value_true))
                print('Linf error        = ', np.linalg.norm(diff.flatten(), ord=np.inf))

    comm.Barrier()
    timer.timer(' ', print_all=True)
    print('# End')
