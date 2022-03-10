import numpy as np
import sys
try:
    from mpi4py import MPI
except ImportError:
    print('It didn\'t find mpi4py!')
import PySTKFMM


def rotne_prager_tensor(r_vectors, eta, a):
  '''
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors of radius a.
  '''
  # Extract variables
  r_vectors = r_vectors.reshape((r_vectors.size // 3, 3))
  x = r_vectors[:,0]
  y = r_vectors[:,1]
  z = r_vectors[:,2]

  # Compute distances between blobs
  dx = x - x[:, None]
  dy = y - y[:, None]
  dz = z - z[:, None]
  dr = np.sqrt(dx**2 + dy**2 + dz**2)

  # Compute scalar functions f(r) and g(r)
  factor = 1.0 / (6.0 * np.pi * eta)
  fr = np.zeros_like(dr)
  gr = np.zeros_like(dr)
  sel = dr > 2.0 * a
  nsel = np.logical_not(sel)
  sel_zero = dr == 0.
  nsel[sel_zero] = False

  fr[sel] = factor * (0.75 / dr[sel] + a**2 / (2.0 * dr[sel]**3))
  gr[sel] = factor * (0.75 / dr[sel]**3 - 1.5 * a**2 / dr[sel]**5)

  fr[sel_zero] = (factor / a)
  fr[nsel] = factor * (1.0 / a - 0.28125 * dr[nsel] / a**2)
  gr[nsel] = factor * (3.0 / (32.0 * a**2 * dr[nsel]))

  # Build mobility matrix of size 3N \times 3N
  M = np.zeros((r_vectors.size, r_vectors.size))
  M[0::3, 0::3] = fr + gr * dx * dx
  M[0::3, 1::3] =      gr * dx * dy
  M[0::3, 2::3] =      gr * dx * dz

  M[1::3, 0::3] =      gr * dy * dx
  M[1::3, 1::3] = fr + gr * dy * dy
  M[1::3, 2::3] =      gr * dy * dz

  M[2::3, 0::3] =      gr * dz * dx
  M[2::3, 1::3] =      gr * dz * dy
  M[2::3, 2::3] = fr + gr * dz * dz
  return M


if __name__ == '__main__':
    # FMM parameters
    # expansion order
    mult_order = 10
    # max points for fmm cell
    max_pts = 128
    # set pbc to PX, PXY, or PXYZ for periodicity
    pbc = PySTKFMM.PAXIS.NONE
    # u, lapu kernel (4->6)
    kernel = PySTKFMM.KERNEL.RPY
    # RPY diameter of particle. Each particle can have separate diameter, but we'll set it to a const here
    a = 0.01

    # Get MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create source and target coordinates
    nsrc = 100
    ntrg = nsrc
    src_coord = np.random.rand(nsrc, 3)
    trg_coord = src_coord
    sys.stdout.flush()
    comm.Barrier()

    # Setup FMM
    myFMM = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel)
    kdim, _, kdimTrg = myFMM.get_kernel_dimension(kernel)

    # Create sources and target values
    src_value = np.random.randn(nsrc, kdim)
    src_value[:, 3] = a
    trg_value = np.zeros((ntrg, kdimTrg))
    if rank == 0:
        myFMM.show_active_kernels()
        print('kdimSL = ', kdim)
        print('kdimTrg = ', kdimTrg)

    # Set tree
    myFMM.set_box(np.array([0.0, 0.0, 0.0]), 2.0)
    myFMM.set_points(src_coord, trg_coord, np.empty(shape=(0,0)))
    myFMM.setup_tree(kernel)

    # Evaluate FMM
    myFMM.evaluate_fmm(kernel, src_value, trg_value, np.empty(shape=(0,0)))

    # Clear FMM and evaluate again
    trg_value[:,:] = 0
    myFMM.clear_fmm(kernel)
    myFMM.evaluate_fmm(kernel, src_value, trg_value, np.empty(shape=(0,0)))

    comm.Barrier()

    # calculate rpy tensor minus self-diffusion matrix
    rpy = rotne_prager_tensor(src_coord, 1, a) - np.identity(src_coord.size) / (6.*np.pi*a)

    # unpack FMM result
    u_trg = trg_value[:,0:3]
    lapu_trg = trg_value[:,3:]

    # Calculate M*F using calculated matrix directly
    MFdirect = (np.dot(rpy, src_value[:,0:3].flatten())).reshape(src_value[:,0:3].shape)

    # Calculate M*F using FMM results
    # WARNING: THIS DOES NOT FILTER OUT OVERLAPS, WILL SPONTANEOUSLY GENERATE LARGE ERROR
    MFstkfmm = u_trg + a * a * lapu_trg / 6.0

    # Print RMS difference of two techniques
    L2 = np.sqrt(np.multiply(MFdirect - MFstkfmm, MFdirect - MFstkfmm))

    print("RMS error: {}".format(np.mean(L2, axis=(0,1))))
    print("Max L2 error: {}".format(np.max(L2, axis=(0,1))))
