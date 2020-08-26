import numpy as np
from ctypes import cdll, c_void_p, c_int, POINTER, c_double
from mpi4py import MPI
import enum

lib = cdll.LoadLibrary("libSTKFMM_SHARED.so")
lib.Stk3DFMM_create.restype = c_void_p


class PAXIS(enum.IntEnum):
    NONE = 0  # non-periodic, free-space
    PX = 1  # periodic along x axis
    PXY = 2  # periodic along XY axis
    PXYZ = 3  # periodic along XYZ axis


class KERNEL(enum.IntFlag):
    LapPGrad = 1  # Laplace
    LapPGradGrad = 2  # Laplace
    LapQPGradGrad = 4  # Laplace quadrupole

    Stokes = 8  # Stokeslet 3x3
    RPY = 16  # RPY
    StokesRegVel = 32  # Regularized Stokes Velocity
    StokesRegVelOmega = 64  # Regularized Stokes Velocity/Rotation

    PVel = 128  # Stokes
    PVelGrad = 256  # Stokes
    PVelLaplacian = 512  # Stokes
    Traction = 1024  # Stokes


class Stk3DFMM():
    def __init__(self, mult_order, max_pts, pbc, kernels):
        self.mult_order = c_int(mult_order)
        self.max_pts = c_int(max_pts)
        self.pbc = c_int(pbc)
        self.kernels = c_int(kernels)

        self.fmm = c_void_p(lib.Stk3DFMM_create(self.mult_order, self.max_pts, self.pbc, self.kernels))

    def __del__(self):
        lib.Stk3DFMM_destroy(self.fmm)

    def set_box(self, origin, length):
        lib.Stk3DFMM_set_box(self.fmm, origin.ctypes.data_as(POINTER(c_double)), c_double(length))

    def get_kernel_dimension(self, kernel):
        dims = np.zeros(3, dtype='int32')
        lib.Stk3DFMM_get_kernel_dimension(c_int(kernel), dims.ctypes.data_as(POINTER(c_int)))
        return dims

    def set_points(self, src_SL_coord, trg_coord, src_DL_coord):
        lib.Stk3DFMM_set_points(self.fmm,
                                c_int(src_SL_coord.shape[0]),
                                src_SL_coord.ctypes.data_as(POINTER(c_double)),
                                c_int(trg_coord.shape[0]),
                                trg_coord.ctypes.data_as(POINTER(c_double)),
                                c_int(src_DL_coord.shape[0]),
                                src_DL_coord.ctypes.data_as(POINTER(c_double)))

    def evaluate_fmm(self, kernel, src_SL_value, trg_value, src_DL_value):
        lib.Stk3DFMM_evaluate_fmm(self.fmm, c_int(kernel),
                                  c_int(src_SL_value.shape[0]),
                                  src_SL_value.ctypes.data_as(POINTER(c_double)),
                                  c_int(trg_value.shape[0]),
                                  trg_value.ctypes.data_as(POINTER(c_double)),
                                  c_int(src_DL_value.shape[0]),
                                  src_DL_value.ctypes.data_as(POINTER(c_double)))

    def setup_tree(self, kernel):
        lib.Stk3DFMM_setup_tree(self.fmm, c_int(kernel))

    def clear_fmm(self, kernel):
        lib.Stk3DFMM_clear_fmm(self.fmm, c_int(kernel))

    def show_active_kernels(self):
        lib.Stk3DFMM_show_active_kernels(self.fmm)


class DArray():
    def __init__(self, array):
        self.data = array

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        if self.rank == 0:
            self.num, self.dim = array.shape
            self.update_split_sizes()
        else:
            self.split_sizes = None
            self.displacements = None
            self.num, self.dim = None, None

        self.split_sizes = self.comm.bcast(self.split_sizes, root=0)
        self.displacements = self.comm.bcast(self.displacements, root=0)
        self.num, self.dim = self.comm.bcast((self.num, self.dim), root=0)

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
