'''
Kernels to evaluate Green's functions.
'''
from __future__ import division, print_function
import numpy as np
import imp
import sys
try:
  from numba import njit, prange
except ImportError:
  print('Numba not found')


@njit(parallel=True)
def StokesSLPVel(r_source, r_target, density, epsilon_distance = 1e-10):
  '''
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.
  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  r_source = r_source.reshape(Nsource, 3)
  r_target = r_target.reshape(Ntarget, 3)
  density = density.reshape(Nsource, 4)
  pvel = np.zeros((Ntarget, 4))

  # Loop over targets
  for xn in prange(Ntarget):
    tx = r_target[xn, 0] 
    ty = r_target[xn, 1] 
    tz = r_target[xn, 2] 
    for yn in range(Nsource):
      sx = r_source[yn, 0]
      sy = r_source[yn, 1]
      sz = r_source[yn, 2]
      x = sx - tx
      y = sy - ty
      z = sz - tz
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     

      fx = density[yn, 0]
      fy = density[yn, 1]
      fz = density[yn, 2]
      TrD = density[yn, 3]

      pvel[xn, 0] += (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz)) / (4.0 * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))
      pvel[xn, 1] += ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) + fx * (2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) - 4 * sx * tx + 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (8.0 * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))
      pvel[xn, 2] += ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) + fy * (np.power(sx, 2) + 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 4 * sy * ty + 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (8.0 * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))
      pvel[xn, 3] += ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) + 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 4 * sz * tz + 2 * np.power(tz, 2))) / (8.0 * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

  return pvel


@njit(parallel=True)
def StokesDLPVel(r_source, r_target, density, epsilon_distance = 1e-10):
  '''
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.
  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  r_source = r_source.reshape(Nsource, 3)
  r_target = r_target.reshape(Ntarget, 3)
  density = density.reshape(Nsource, 9)
  pvel = np.zeros((Ntarget, 4))

  # Loop over targets
  for xn in prange(Ntarget):
    tx = r_target[xn, 0] 
    ty = r_target[xn, 1] 
    tz = r_target[xn, 2] 
    for yn in range(Nsource):
      sx = r_source[yn, 0]
      sy = r_source[yn, 1]
      sz = r_source[yn, 2]
      x = sx - tx
      y = sy - ty
      z = sz - tz
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     

      dbxx = density[yn, 0]
      dbxy = density[yn, 1]
      dbxz = density[yn, 2]
      dbyx = density[yn, 3]
      dbyy = density[yn, 4]
      dbyz = density[yn, 5]
      dbzx = density[yn, 6]
      dbzy = density[yn, 7]
      dbzz = density[yn, 8]

      pvel[xn,0] += (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz + 3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz - 3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * np.power(tz, 2) + dbyy * (np.power(sx, 2) - 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) + 4 * sy * ty - 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)) + dbxx * (-2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) + 4 * sx * tx - 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvel[xn,1] += ((-sx + tx) * (-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvel[xn,2] += ((-sy + ty) * (-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvel[xn,3] += ((-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2)) * (-sz + tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

  return pvel


@njit(parallel=True)
def StokesSLPVelGrad(r_source, r_target, density, epsilon_distance = 1e-10):
  '''
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.
  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  r_source = r_source.reshape(Nsource, 3)
  r_target = r_target.reshape(Ntarget, 3)
  density = density.reshape(Nsource, 4)
  pvelGrad = np.zeros((Ntarget, 16))

  # Loop over targets
  for xn in prange(Ntarget):
    tx = r_target[xn, 0] 
    ty = r_target[xn, 1] 
    tz = r_target[xn, 2] 
    for yn in range(Nsource):
      sx = r_source[yn, 0]
      sy = r_source[yn, 1]
      sz = r_source[yn, 2]
      x = sx - tx
      y = sy - ty
      z = sz - tz
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     

      fx = density[yn, 0]
      fy = density[yn, 1]
      fz = density[yn, 2]
      TrD = density[yn, 3]

      pvelGrad[xn,0] += (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelGrad[xn,1] += ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) + fx * (2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) - 4 * sx * tx + 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelGrad[xn,2] += ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) + fy * (np.power(sx, 2) + 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 4 * sy * ty + 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelGrad[xn,3] += ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) + 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 4 * sz * tz + 2 * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelGrad[xn,4] += fx / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sx - tx) * (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,5] += fy / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sy - ty) * (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,6] += fz / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sz - tz) * (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      # grad vx
      pvelGrad[xn,7] += (-(fz * sz) - TrD + fx * (-4 * sx + 4 * tx) - fy * (sy - ty) + fz * tz) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sx - tx) * ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) + fx * (2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) - 4 * sx * tx + 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,8] += (-(fy * (sx - tx)) + fx * (-2 * sy + 2 * ty)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sy - ty) * ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) + fx * (2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) - 4 * sx * tx + 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,9] += (-(fz * (sx - tx)) + fx * (-2 * sz + 2 * tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sz - tz) * ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) + fx * (2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) - 4 * sx * tx + 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      # grad vy
      pvelGrad[xn,10] += (fy * (-2 * sx + 2 * tx) - fx * (sy - ty)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sx - tx) * ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) + fy * (np.power(sx, 2) + 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 4 * sy * ty + 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
      
      pvelGrad[xn,11] += (-(fz * sz) - TrD - fx * (sx - tx) + fy * (-4 * sy + 4 * ty) + fz * tz) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sy - ty) * ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) + fy * (np.power(sx, 2) + 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 4 * sy * ty + 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,12] += (-(fz * (sy - ty)) + fy * (-2 * sz + 2 * tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sz - tz) * ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) + fy * (np.power(sx, 2) + 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 4 * sy * ty + 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)))) /    (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      # grad vz
      pvelGrad[xn,13] += (fz * (-2 * sx + 2 * tx) - fx * (sz - tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sx - tx) * ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) + 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 4 * sz * tz + 2 * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
            
      pvelGrad[xn,14] += (fz * (-2 * sy + 2 * ty) - fy * (sz - tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sy - ty) * ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) + 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 4 * sz * tz + 2 * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
      
      pvelGrad[xn,15] += (-(fy * sy) - TrD - fx * (sx - tx) + fy * ty + fz * (-4 * sz + 4 * tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5)) + (3 * (sz - tz) * ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) + 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 4 * sz * tz + 2 * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

  return pvelGrad


@njit(parallel=True)
def StokesDLPVelGrad(r_source, r_target, density, epsilon_distance = 1e-10):
  '''
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.
  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  r_source = r_source.reshape(Nsource, 3)
  r_target = r_target.reshape(Ntarget, 3)
  density = density.reshape(Nsource, 9)
  pvelGrad = np.zeros((Ntarget, 16))

  # Loop over targets
  for xn in prange(Ntarget):
    tx = r_target[xn, 0] 
    ty = r_target[xn, 1] 
    tz = r_target[xn, 2] 
    for yn in range(Nsource):
      sx = r_source[yn, 0]
      sy = r_source[yn, 1]
      sz = r_source[yn, 2]
      x = sx - tx
      y = sy - ty
      z = sz - tz
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     

      dbxx = density[yn, 0]
      dbxy = density[yn, 1]
      dbxz = density[yn, 2]
      dbyx = density[yn, 3]
      dbyy = density[yn, 4]
      dbyz = density[yn, 5]
      dbzx = density[yn, 6]
      dbzy = density[yn, 7]
      dbzz = density[yn, 8]

      pvelGrad[xn,0] += (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) +         3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz + 3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz - 3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * np.power(tz, 2) + dbyy * (np.power(sx, 2) - 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) + 4 * sy * ty - 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)) + dbxx * (-2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) + 4 * sx * tx - 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,1] += ((-sx + tx) * (-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,2] += ((-sy + ty) * (-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
      
      pvelGrad[xn,3] += ((-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2)) * (-sz + tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelGrad[xn,4] += ((4 * dbxx * (sx - tx) - 2 * (dbyy + dbzz) * (sx - tx) + 3 * (dbxy + dbyx) * (sy - ty) + 3 * (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + dbyy * (np.power(sx - tx, 2) - 2 * np.power(sy - ty, 2) + np.power(sz - tz, 2)) + dbxx * (-2 * np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + (4 * dbzz * sz + 3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty)) * tz - 2 * dbzz * np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,5] += ((3 * (dbxy + dbyx) * (sx - tx) - 2 * (dbxx - 2 * dbyy + dbzz) * (sy - ty) + 3 * (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + dbyy * (np.power(sx - tx, 2) - 2 * np.power(sy - ty, 2) + np.power(sz - tz, 2)) + dbxx * (-2 * np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + (4 * dbzz * sz + 3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty)) * tz - 2 * dbzz * np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,6] += ((3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty) - 2 * (dbxx + dbyy - 2 * dbzz) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sz - tz) * (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + dbyy * (np.power(sx - tx, 2) - 2 * np.power(sy - ty, 2) + np.power(sz - tz, 2)) + dbxx * (-2 * np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + (4 * dbzz * sz + 3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty)) * tz - 2 * dbzz * np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      # vx grad
      pvelGrad[xn,7] += (3 * (-sx + tx) * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (-sx + tx) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)) + (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,8] += ((-sx + tx) * (3 * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,9] += ((-sx + tx) * (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sz - tz) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      # vy grad
      pvelGrad[xn,10] += ((-sy + ty) * (3 * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,11] += (3 * (-sy + ty) * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (-sy + ty) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)) + (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,12] += ((-sy + ty) * (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sz - tz) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      # vz grad
      pvelGrad[xn,13] += ((-sz + tz) *(3 * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,14] += ((-sz + tz) * (3 * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelGrad[xn,15] += (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-sz + tz) + (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)) + 5 * (sz - tz) * (-sz + tz) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

  return pvelGrad
