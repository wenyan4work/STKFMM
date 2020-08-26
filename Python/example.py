from mpi4py import MPI
import numpy as np
import timer
# FIXME: Rename to PySTKFMM when no collision with .so
from STKFMM import Stk3DFMM, DArray, KERNEL
import kernels as kr

# Get MPI parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# Convenience wrapper to print only on MPI rank 0
def printer(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


# Convenience wrapper to remove timer.timer() spam
def time_func(name, func, *args, **kwargs):
    timer.timer(name)
    res = func(*args, **kwargs)
    timer.timer(name)
    return res


def calc_true_value(kernel, src_SL_coord, trg_coord, src_SL_value, src_DL_coord, src_DL_value):
    epsilon_distance = 2e-4
    if kernel == KERNEL.PVel:
        trg_value = kr.StokesSLPVel(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLPVel(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel == KERNEL.PVelGrad:
        trg_value = kr.StokesSLPVelGrad(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLPVelGrad(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel == KERNEL.PVelLaplacian:
        trg_value = kr.StokesSLPVelLaplacian(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLPVelLaplacian(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel == KERNEL.Traction:
        trg_value = kr.StokesSLTraction(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLTraction(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel == KERNEL.LapPGrad:
        trg_value = kr.LaplaceSLPGrad(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.LaplaceDLPGrad(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    else:
        trg_value = None
    return trg_value


# FMM parameters
mult_order = 10  # Multipole order (higher is slower, but more accurate)
max_pts = 128  # Max points per OctTree leaf
pbc = 0  # Number of dimensions to periodize (0, 1, 2, 3)
kernels = [KERNEL.PVel, KERNEL.PVelGrad, KERNEL.PVelLaplacian, KERNEL.Traction, KERNEL.LapPGrad]
verify = True

# Create sources and targets coordinates
nsrc_SL = 1000
nsrc_DL = 1000
ntrg = 1000

# Create points on rank = 0, handled by DArray wrapper
# DArray only necessary when using MPI, otherwise a plain numpy object is fine
src_SL_coord = DArray(None if rank else np.random.rand(nsrc_SL, 3))
src_DL_coord = DArray(None if rank else np.random.rand(nsrc_DL, 3))
trg_coord = DArray(None if rank else np.random.rand(ntrg, 3))

# Distribute points among MPI ranks
src_SL_coord.scatter()
src_DL_coord.scatter()
trg_coord.scatter()

for kernel in kernels:
    # Create FMM
    printer("\n\n==============================")
    fmm = time_func('create_fmm', Stk3DFMM, mult_order, max_pts, pbc, kernel)
    kdimSL, kdimDL, kdimTrg = time_func('get_kernel_dimension', fmm.get_kernel_dimension, kernel)

    printer("kdimSL = {}\nkdimDL = {}\nkdimTrg = {}".format(kdimSL, kdimDL, kdimTrg))

    # Create source and target values
    src_SL_value = DArray(None if rank else np.random.randn(nsrc_SL, kdimSL))
    src_DL_value = DArray(None if rank else np.random.randn(nsrc_DL, kdimDL))
    trg_value = DArray(None if rank else np.zeros((ntrg, kdimTrg)))

    src_SL_value.scatter()
    src_DL_value.scatter()
    trg_value.scatter()

    # Create box, add points, and build tree
    # Tree only needs to be built when points move - their values can change
    time_func('set_box', fmm.set_box, np.zeros(3), 1.0)
    time_func('set_points', fmm.set_points, src_SL_coord.chunk, trg_coord.chunk, src_DL_coord.chunk)
    time_func('setup_tree', fmm.setup_tree, kernel)

    # Evaluate FMM
    time_func('evaluate_fmm', fmm.evaluate_fmm, kernel, src_SL_value.chunk, trg_value.chunk, src_DL_value.chunk)

    # Clear FMM and evaluate again
    trg_value.chunk[:, :] = 0
    time_func('clear_fmm', fmm.clear_fmm, kernel)
    time_func('evaluate_fmm', fmm.evaluate_fmm, kernel, src_SL_value.chunk, trg_value.chunk, src_DL_value.chunk)

    # Collect target points into rank=0 array (for MPI)
    src_SL_value.gather()
    src_DL_value.gather()
    trg_value.gather()
    if verify and rank == 0:
        trg_value_true = time_func('true_value', calc_true_value,
                                   kernel, src_SL_coord.data,
                                   trg_coord.data, src_SL_value.data,
                                   src_DL_coord.data, src_DL_value.data)
        if trg_value_true is not None:
            diff = trg_value.data - trg_value_true
            printer('relative L2 error = ', np.linalg.norm(diff) / np.linalg.norm(trg_value_true))
            printer('Linf error        = ', np.linalg.norm(diff.flatten(), ord=np.inf))

    comm.Barrier()
    if rank == 0:
        timer.timer(None, print_all=True)
    printer('# End')
