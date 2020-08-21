import numpy as np
import sys
try:
    from mpi4py import MPI
except ImportError:
    print('It didn\'t find mpi4py!')
import PySTKFMM
import kernels as kr
import timer


def printer(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)


def calc_true_value(kernel_index, src_SL_coord, trg_coord, src_SL_value, src_DL_coord, src_DL_value):
    epsilon_distance = 2e-4
    if kernel_index == PySTKFMM.KERNEL.PVel:
        trg_value = kr.StokesSLPVel(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLPVel(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel_index == PySTKFMM.KERNEL.PVelGrad:
        trg_value = kr.StokesSLPVelGrad(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLPVelGrad(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel_index == PySTKFMM.KERNEL.PVelLaplacian:
        trg_value = kr.StokesSLPVelLaplacian(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLPVelLaplacian(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel_index == PySTKFMM.KERNEL.Traction:
        trg_value = kr.StokesSLTraction(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.StokesDLTraction(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    elif kernel_index == PySTKFMM.KERNEL.LapPGrad:
        trg_value = kr.LaplaceSLPGrad(src_SL_coord, trg_coord, src_SL_value, epsilon_distance=epsilon_distance)
        trg_value += kr.LaplaceDLPGrad(src_DL_coord, trg_coord, src_DL_value, epsilon_distance=epsilon_distance)
    else:
        trg_value = None
    return trg_value


class DArray():
    def __init__(self, array):
        self.data = array

        self.comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        if self.rank == 0:
            self.num, self.dim = array.shape
            self.update_split_sizes()
        else:
            self.split_sizes = None
            self.displacements = None
            self.num, self.dim = None, None

        self.split_sizes = comm.bcast(self.split_sizes, root=0)
        self.displacements = comm.bcast(self.displacements, root=0)
        self.num = comm.bcast(self.num, root=0)
        self.dim = comm.bcast(self.dim, root=0)

        self.chunk = np.empty((self.split_sizes[self.rank] // self.dim, self.dim), dtype='float64')

    def scatter(self):
        self.comm.Scatterv([self.data, self.split_sizes, self.displacements, MPI.DOUBLE], self.chunk, root=0)

    def gather(self):
        self.comm.Gatherv(self.chunk, [self.data, self.split_sizes, self.displacements, MPI.DOUBLE], root=0)

    def update_split_sizes(self):
        bigc = self.num % self.size

        self.split_sizes = np.empty(self.size, dtype='int64')
        self.split_sizes[:bigc] = (self.num // self.size + 1) * self.dim
        self.split_sizes[bigc:] = (self.num // self.size) * self.dim

        self.displacements = np.zeros(self.size, dtype='int64')
        for i in range(1, self.size):
            self.displacements[i] = self.displacements[i-1] + self.split_sizes[i-1]


if __name__ == '__main__':
    printer('# Start')

    # FMM parameters
    mult_order = 10
    max_pts = 128
    pbc = PySTKFMM.PAXIS.NONE
    kernels = [PySTKFMM.KERNEL.PVel, PySTKFMM.KERNEL.PVelGrad, PySTKFMM.KERNEL.PVelLaplacian,
               PySTKFMM.KERNEL.Traction, PySTKFMM.KERNEL.LapPGrad]
    kernels_index = [PySTKFMM.KERNEL(k) for k in kernels]
    verify = True

    # Get MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create sources and targets coordinates
    nsrc_SL = 1000
    nsrc_DL = 1000
    ntrg = 1000
    src_SL_coord = DArray(None if rank else np.random.rand(nsrc_SL, 3))
    src_DL_coord = DArray(None if rank else np.random.rand(nsrc_DL, 3))
    trg_coord = DArray(None if rank else np.random.rand(ntrg, 3))

    src_SL_coord.scatter()
    src_DL_coord.scatter()
    trg_coord.scatter()

    sys.stdout.flush()
    comm.Barrier()

    # Try STKFMM
    for k, kernel in enumerate(kernels):
        # Create FMM
        printer('\n\n==============================')
        timer.timer('create_fmm')
        myFMM = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernels_index[k])
        timer.timer('create_fmm')
        timer.timer('show_active_kernels')
        myFMM.showActiveKernels()
        timer.timer('show_active_kernels')
        timer.timer('get_kernel_dimension')
        kdimSL, kdimDL, kdimTrg = myFMM.getKernelDimension(kernel)
        timer.timer('get_kernel_dimension')

        # Create sources and target values
        src_SL_value = DArray(None if rank else np.random.randn(nsrc_SL, kdimSL))
        src_DL_value = DArray(None if rank else np.random.randn(nsrc_DL, kdimDL))
        trg_value = DArray(None if rank else np.zeros((ntrg, kdimTrg)))

        src_SL_value.scatter()
        src_DL_value.scatter()
        trg_value.scatter()

        printer('kdimSL = ', kdimSL)
        printer('kdimDL = ', kdimDL)
        printer('kdimTrg = ', kdimTrg)

        # Set tree
        timer.timer('set_box')
        myFMM.setBox([0.0, 0.0, 0.0], 2)
        timer.timer('set_box')
        timer.timer('set_points')
        myFMM.setPoints(src_SL_coord.chunk.shape[0], src_SL_coord.chunk,
                        trg_coord.chunk.shape[0], trg_coord.chunk,
                        src_DL_coord.chunk.shape[0], src_DL_coord.chunk)
        timer.timer('set_points')
        timer.timer('setup_tree')
        myFMM.setupTree(kernel)
        timer.timer('setup_tree')

        # Evaluate FMM
        timer.timer('evaluate_fmm')
        myFMM.evaluateFMM(kernel, src_SL_value.chunk.shape[0], src_SL_value.chunk,
                          trg_value.chunk.shape[0], trg_value.chunk,
                          src_DL_value.chunk.shape[0], src_DL_value.chunk)
        timer.timer('evaluate_fmm')

        # Clear FMM and evaluate again
        trg_value.chunk[:, :] = 0
        timer.timer('clear_fmm')
        myFMM.clearFMM(kernel)
        timer.timer('clear_fmm')
        timer.timer('evaluate_fmm')
        myFMM.evaluateFMM(kernel, src_SL_value.chunk.shape[0], src_SL_value.chunk,
                          trg_value.chunk.shape[0], trg_value.chunk,
                          src_DL_value.chunk.shape[0], src_DL_value.chunk)
        timer.timer('evaluate_fmm')

        src_SL_value.gather()
        src_DL_value.gather()
        trg_value.gather()
        if verify and rank == 0:
            timer.timer('true_value')
            trg_value_true = calc_true_value(kernels_index[k], src_SL_coord.data,
                                             trg_coord.data, src_SL_value.data,
                                             src_DL_coord.data, src_DL_value.data)
            timer.timer('true_value')
            if trg_value_true is not None:
                diff = trg_value.data - trg_value_true
                printer('relative L2 error = ', np.linalg.norm(diff) / np.linalg.norm(trg_value_true))
                printer('Linf error        = ', np.linalg.norm(diff.flatten(), ord=np.inf))

    comm.Barrier()
    if rank == 0:
        timer.timer(' ', print_all=True)
    printer('# End')
