from __future__ import division, print_function
import numpy as np
import sys
try:
    from mpi4py import MPI
except ImportError:
    print('It didn\'t find mpi4py!')
import stkfmm 

if __name__ == '__main__':
    print('# Start')

    # FMM parameters
    mult_order = 10
    max_pts = 1024
    kernelComb = 1

    # Get MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create sources and targets
    nsrc_SL = 1024
    src_SL_coord = np.random.rand(nsrc_SL, 3)
    src_SL_value = np.random.rand(nsrc_SL, 3)
    nsrc_DL = 1024
    src_DL_coord = np.random.rand(nsrc_DL, 3)
    src_DL_value = np.random.rand(nsrc_DL, 3)
    ntrg = 1024
    trg_coord = np.random.rand(ntrg, 3)
    sys.stdout.flush()
    comm.Barrier()

    # Try STKFMM
    pbc = stkfmm.FMM_PAXIS.NONE
    myFMM = stkfmm.STKFMM(mult_order, max_pts, pbc, kernelComb)
    stkfmm.setBox(myFMM, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    stkfmm.showActiveKernels(myFMM)
    stkfmm.setPoints(myFMM, nsrc_SL, src_SL_coord, nsrc_DL, src_DL_coord, ntrg, trg_coord)
    kdimSL = 0
    kdimDL = -1
    kdimTrg = -1
    test_kernel = stkfmm.FMM_KERNEL.PVel
    kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(myFMM, test_kernel)  
    print('kdimSL = ', kdimSL)
    print('kdimDL = ', kdimDL)
    print('kdimTrg = ', kdimTrg)
    trg_value = np.zeros((ntrg, kdimTrg))

    stkfmm.setupTree(myFMM, test_kernel)
    print('test_kernel = ', test_kernel)
    print('isKernelActive = ', stkfmm.isKernelActive(myFMM, test_kernel))

    stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, test_kernel)
    print('trg_value = \n', trg_value)
    trg_value[:,:] = 0.
    print('trg_value = \n', trg_value)

    stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, test_kernel)
    print('trg_value = \n', trg_value)
    trg_value[:,:] = -1.0

    stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, trg_value, test_kernel)
    print('trg_value = \n', trg_value)


    new_trg_value = np.zeros_like(trg_value)
    stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, new_trg_value, test_kernel)
    print('trg_value = \n', new_trg_value)


    stkfmm.clearFMM(myFMM, test_kernel)
    a = np.zeros_like(trg_value)
    stkfmm.evaluateFMM(myFMM, nsrc_SL, src_SL_value, nsrc_DL, src_DL_value, ntrg, a, test_kernel)
    print('trg_value = \n', a)



    p2p = stkfmm.FMM_PPKERNEL.SLS2T
    test_kernel = stkfmm.FMM_KERNEL.PVel
    if stkfmm.isKernelActive(myFMM, test_kernel):
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        stkfmm.evaluateKernel(myFMM, -1, stkfmm.FMM_PPKERNEL.SLS2T, nsrc_SL, src_SL_coord, src_SL_value, 
                              ntrg, trg_coord, trg_value, test_kernel)
        print(trg_value)


    comm.Barrier()

    print('# End')
