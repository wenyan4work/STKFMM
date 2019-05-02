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

      # p grad
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


@njit(parallel=True)
def StokesSLPVelLaplacian(r_source, r_target, density, epsilon_distance = 1e-10):
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
  pvelLaplacian = np.zeros((Ntarget, 7))

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

      pvelLaplacian[xn,0] += (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelLaplacian[xn,1] += ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) + fx * (2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) - 4 * sx * tx + 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelLaplacian[xn,2] += ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) + fy * (np.power(sx, 2) + 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 4 * sy * ty + 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))

      pvelLaplacian[xn,3] += ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) + 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 4 * sz * tz + 2 * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 1.5))
      
      pvelLaplacian[xn,4] += (-3 * (sx - tx) * (fy * (sy - ty) + fz * (sz - tz)) + fx * (-2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) + 4 * sx * tx - 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
      pvelLaplacian[xn,5] += (-3 * (sy - ty) * (fx * (sx - tx) + fz * (sz - tz)) + fy * (np.power(sx, 2) - 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) + 4 * sy * ty - 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
      
      pvelLaplacian[xn,6] += (-3 * (fx * (sx - tx) + fy * (sy - ty)) * (sz - tz) + fz * (np.power(sx, 2) + np.power(sy, 2) - 2 * np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) + 4 * sz * tz - 2 * np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

  return pvelLaplacian


@njit(parallel=True)
def StokesDLPVelLaplacian(r_source, r_target, density, epsilon_distance = 1e-10):
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
  pvelLaplacian = np.zeros((Ntarget, 7))

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

      # p,vx,vy,vz,vxlap,vylap,vzlap
      pvelLaplacian[xn,0] += (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz + 3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz - 3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * np.power(tz, 2) + dbyy * (np.power(sx, 2) - 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) + 4 * sy * ty - 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)) + dbxx * (-2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) + 4 * sx * tx - 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelLaplacian[xn,1] += ((-sx + tx) * (-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelLaplacian[xn,2] += ((-sy + ty) * (-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelLaplacian[xn,3] += ((-3 * dbxx * np.power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) - 3 * dbyy * np.power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) - 3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * np.power(sz - tz, 2)) * (-sz + tz)) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      pvelLaplacian[xn,4] += (3 * (dbzz * np.power(sx, 3) - 4 * dbxy * np.power(sx, 2) * sy - 4 * dbyx * np.power(sx, 2) * sy + dbzz * sx * np.power(sy, 2) + dbxy * np.power(sy, 3) + dbyx * np.power(sy, 3) - 4 * dbxz * np.power(sx, 2) * sz - 4 * dbzx * np.power(sx, 2) * sz - 5 * dbyz * sx * sy * sz - 5 * dbzy * sx * sy * sz + dbxz * np.power(sy, 2) * sz + dbzx * np.power(sy, 2) * sz - 4 * dbzz * sx * np.power(sz, 2) + dbxy * sy * np.power(sz, 2) + dbyx * sy * np.power(sz, 2) + dbxz * np.power(sz, 3) + dbzx * np.power(sz, 3) - 3 * dbzz * np.power(sx, 2) * tx + 8 * dbxy * sx * sy * tx + 8 * dbyx * sx * sy * tx - dbzz * np.power(sy, 2) * tx + 8 * dbxz * sx * sz * tx + 8 * dbzx * sx * sz * tx + 5 * dbyz * sy * sz * tx + 5 * dbzy * sy * sz * tx + 4 * dbzz * np.power(sz, 2) * tx + 3 * dbzz * sx * np.power(tx, 2) - 4 * dbxy * sy * np.power(tx, 2) - 4 * dbyx * sy * np.power(tx, 2) - 4 * dbxz * sz * np.power(tx, 2) - 4 * dbzx * sz * np.power(tx, 2) - dbzz * np.power(tx, 3) + 4 * dbxy * np.power(sx, 2) * ty + 4 * dbyx * np.power(sx, 2) * ty - 2 * dbzz * sx * sy * ty - 3 * dbxy * np.power(sy, 2) * ty - 3 * dbyx * np.power(sy, 2) * ty + 5 * dbyz * sx * sz * ty + 5 * dbzy * sx * sz * ty - 2 * dbxz * sy * sz * ty - 2 * dbzx * sy * sz * ty - dbxy * np.power(sz, 2) * ty - dbyx * np.power(sz, 2) * ty - 8 * dbxy * sx * tx * ty - 8 * dbyx * sx * tx * ty + 2 * dbzz * sy * tx * ty - 5 * dbyz * sz * tx * ty - 5 * dbzy * sz * tx * ty + 4 * dbxy * np.power(tx, 2) * ty + 4 * dbyx * np.power(tx, 2) * ty + dbzz * sx * np.power(ty, 2) + 3 * dbxy * sy * np.power(ty, 2) + 3 * dbyx * sy * np.power(ty, 2) + dbxz * sz * np.power(ty, 2) + dbzx * sz * np.power(ty, 2) - dbzz * tx * np.power(ty, 2) - dbxy * np.power(ty, 3) - dbyx * np.power(ty, 3) - dbxx * (sx - tx) * (2 * np.power(sx - tx, 2) - 3 * (np.power(sy - ty, 2) + np.power(sz - tz, 2))) + dbyy * (sx - tx) * (np.power(sx - tx, 2) - 4 * np.power(sy - ty, 2) + np.power(sz - tz, 2)) + (5 * dbyz * sx * sy + 5 * dbzy * sx * sy + 8 * dbzz * sx * sz - 2 * dbxy * sy * sz - 2 * dbyx * sy * sz - 5 * dbyz * sy * tx - 5 * dbzy * sy * tx - 8 * dbzz * sz * tx + dbxz * (-3 * np.power(sz, 2) + 4 * np.power(sx - tx, 2) - np.power(sy - ty, 2)) + dbzx * (-3 * np.power(sz, 2) + 4 * np.power(sx - tx, 2) - np.power(sy - ty, 2)) - 5 * dbyz * sx * ty - 5 * dbzy * sx * ty + 2 * dbxy * sz * ty + 2 * dbyx * sz * ty + 5 * dbyz * tx * ty + 5 * dbzy * tx * ty) * tz + (3 * (dbxz + dbzx) * sz + 4 * dbzz * (-sx + tx) + (dbxy + dbyx) * (sy - ty)) * np.power(tz, 2) - (dbxz + dbzx) * np.power(tz, 3))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelLaplacian[xn,5] += (3 * (-4 * dbxx * np.power(sx, 2) * sy + 3 * dbyy * np.power(sx, 2) * sy + dbzz * np.power(sx, 2) * sy + dbxx * np.power(sy, 3) - 2 * dbyy * np.power(sy, 3) + dbzz * np.power(sy, 3) + dbyz * np.power(sx, 2) * sz + dbzy * np.power(sx, 2) * sz - 5 * dbxz * sx * sy * sz - 5 * dbzx * sx * sy * sz - 4 * dbyz * np.power(sy, 2) * sz - 4 * dbzy * np.power(sy, 2) * sz + dbxx * sy * np.power(sz, 2) + 3 * dbyy * sy * np.power(sz, 2) - 4 * dbzz * sy * np.power(sz, 2) + dbyz * np.power(sz, 3) + dbzy * np.power(sz, 3) + 8 * dbxx * sx * sy * tx - 6 * dbyy * sx * sy * tx - 2 * dbzz * sx * sy * tx - 2 * dbyz * sx * sz * tx - 2 * dbzy * sx * sz * tx + 5 * dbxz * sy * sz * tx + 5 * dbzx * sy * sz * tx - 4 * dbxx * sy * np.power(tx, 2) + 3 * dbyy * sy * np.power(tx, 2) + dbzz * sy * np.power(tx, 2) + dbyz * sz * np.power(tx, 2) + dbzy * sz * np.power(tx, 2) + 4 * dbxx * np.power(sx, 2) * ty - 3 * dbyy * np.power(sx, 2) * ty - dbzz * np.power(sx, 2) * ty - 3 * dbxx * np.power(sy, 2) * ty + 6 * dbyy * np.power(sy, 2) * ty - 3 * dbzz * np.power(sy, 2) * ty + 5 * dbxz * sx * sz * ty + 5 * dbzx * sx * sz * ty + 8 * dbyz * sy * sz * ty + 8 * dbzy * sy * sz * ty - dbxx * np.power(sz, 2) * ty - 3 * dbyy * np.power(sz, 2) * ty + 4 * dbzz * np.power(sz, 2) * ty - 8 * dbxx * sx * tx * ty + 6 * dbyy * sx * tx * ty + 2 * dbzz * sx * tx * ty - 5 * dbxz * sz * tx * ty - 5 * dbzx * sz * tx * ty + 4 * dbxx * np.power(tx, 2) * ty - 3 * dbyy * np.power(tx, 2) * ty - dbzz * np.power(tx, 2) * ty + 3 * dbxx * sy * np.power(ty, 2) - 6 * dbyy * sy * np.power(ty, 2) + 3 * dbzz * sy * np.power(ty, 2) - 4 * dbyz * sz * np.power(ty, 2) - 4 * dbzy * sz * np.power(ty, 2) - dbxx * np.power(ty, 3) + 2 * dbyy * np.power(ty, 3) - dbzz * np.power(ty, 3) + dbxy * (sx - tx) * (np.power(sx - tx, 2) - 4 * np.power(sy - ty, 2) + np.power(sz - tz, 2)) + dbyx * (sx - tx) * (np.power(sx - tx, 2) - 4 * np.power(sy - ty, 2) + np.power(sz - tz, 2)) - (dbyz * (3 * np.power(sz, 2) + np.power(sx - tx, 2) - 4 * np.power(sy - ty, 2)) + dbzy * (3 * np.power(sz, 2) + np.power(sx - tx, 2) - 4 * np.power(sy - ty, 2)) -(-2 * (dbxx + 3 * dbyy - 4 * dbzz) * sz + 5 * (dbxz + dbzx) * (sx - tx)) * (sy - ty)) * tz + (3 * (dbyz + dbzy) * sz + (dbxx + 3 * dbyy - 4 * dbzz) * (sy - ty)) * np.power(tz, 2) - (dbyz + dbzy) * np.power(tz, 3))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      pvelLaplacian[xn,6] += (3 * (dbyz * np.power(sx, 2) * sy + dbzy * np.power(sx, 2) * sy + dbyz * np.power(sy, 3) + dbzy * np.power(sy, 3) - 4 * dbxx * np.power(sx, 2) * sz + dbyy * np.power(sx, 2) * sz + 3 * dbzz * np.power(sx, 2) * sz - 5 * dbxy * sx * sy * sz - 5 * dbyx * sx * sy * sz + dbxx * np.power(sy, 2) * sz - 4 * dbyy * np.power(sy, 2) * sz + 3 * dbzz * np.power(sy, 2) * sz - 4 * dbyz * sy * np.power(sz, 2) - 4 * dbzy * sy * np.power(sz, 2) + dbxx * np.power(sz, 3) + dbyy * np.power(sz, 3) - 2 * dbzz * np.power(sz, 3) - 2 * dbyz * sx * sy * tx - 2 * dbzy * sx * sy * tx + 8 * dbxx * sx * sz * tx - 2 * dbyy * sx * sz * tx - 6 * dbzz * sx * sz * tx + 5 * dbxy * sy * sz * tx + 5 * dbyx * sy * sz * tx + dbyz * sy * np.power(tx, 2) + dbzy * sy * np.power(tx, 2) - 4 * dbxx * sz * np.power(tx, 2) + dbyy * sz * np.power(tx, 2) + 3 * dbzz * sz * np.power(tx, 2) - dbyz * np.power(sx, 2) * ty - dbzy * np.power(sx, 2) * ty - 3 * dbyz * np.power(sy, 2) * ty - 3 * dbzy * np.power(sy, 2) * ty + 5 * dbxy * sx * sz * ty + 5 * dbyx * sx * sz * ty - 2 * dbxx * sy * sz * ty + 8 * dbyy * sy * sz * ty - 6 * dbzz * sy * sz * ty + 4 * dbyz * np.power(sz, 2) * ty + 4 * dbzy * np.power(sz, 2) * ty + 2 * dbyz * sx * tx * ty + 2 * dbzy * sx * tx * ty - 5 * dbxy * sz * tx * ty - 5 * dbyx * sz * tx * ty - dbyz * np.power(tx, 2) * ty - dbzy * np.power(tx, 2) * ty + 3 * dbyz * sy * np.power(ty, 2) + 3 * dbzy * sy * np.power(ty, 2) + dbxx * sz * np.power(ty, 2) - 4 * dbyy * sz * np.power(ty, 2) + 3 * dbzz * sz * np.power(ty, 2) - dbyz * np.power(ty, 3) - dbzy * np.power(ty, 3) + dbxz * (sx - tx) * (np.power(sx, 2) - 2 * sx * tx + np.power(tx, 2) + np.power(sy - ty, 2) - 4 * np.power(sz - tz, 2)) + dbzx * (sx - tx) * (np.power(sx, 2) - 2 * sx * tx + np.power(tx, 2) + np.power(sy - ty, 2) - 4 * np.power(sz - tz, 2)) + (-3 * dbzz * np.power(sx, 2) + 5 * dbxy * sx * sy + 5 * dbyx * sx * sy - 3 * dbzz * np.power(sy, 2) + 8 * dbyz * sy * sz + 8 * dbzy * sy * sz + 6 * dbzz * np.power(sz, 2) + 6 * dbzz * sx * tx - 5 * dbxy * sy * tx - 5 * dbyx * sy * tx - 3 * dbzz * np.power(tx, 2) - dbyy * (3 * np.power(sz, 2) + np.power(sx - tx, 2) - 4 * np.power(sy - ty, 2)) +  dbxx * (-3 * np.power(sz, 2) + 4 * np.power(sx - tx, 2) - np.power(sy - ty, 2)) + (6 * dbzz * sy - 8 * (dbyz + dbzy) * sz - 5 * (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbzz * np.power(ty, 2)) * tz + (3 * (dbxx + dbyy - 2 * dbzz) * sz - 4 * (dbyz + dbzy) * (sy - ty)) * np.power(tz, 2) - (dbxx + dbyy - 2 * dbzz) * np.power(tz, 3))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

  return pvelLaplacian


@njit(parallel=True)
def StokesSLTraction(r_source, r_target, density, epsilon_distance = 1e-10):
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
  traction = np.zeros((Ntarget, 9))

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

      traction[xn,0] += (3 * fz * np.power(sx, 2) * sz + 2 * np.power(sx, 2) * TrD - np.power(sy, 2) * TrD - np.power(sz, 2) * TrD + 3 * fx * np.power(sx - tx, 3) - 6 * fz * sx * sz * tx - 4 * sx * TrD * tx + 3 * fz * sz * np.power(tx, 2) + 2 * TrD * np.power(tx, 2) + 3 * fy * np.power(sx - tx, 2) * (sy - ty) + 2 * sy * TrD * ty - TrD * np.power(ty, 2) - 3 * fz * np.power(sx, 2) * tz + 2 * sz * TrD * tz + 6 * fz * sx * tx * tz - 3 * fz * np.power(tx, 2) * tz - TrD * np.power(tz, 2)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,1] += (3 * (sx - tx) * (sy - ty) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,2] += (3 * (sx - tx) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))
          
      traction[xn,3] += (3 * (sx - tx) * (sy - ty) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,4] += (3 * fz * np.power(sy, 2) * sz - np.power(sx, 2) * TrD + 2 * np.power(sy, 2) * TrD - np.power(sz, 2) * TrD + 2 * sx * TrD * tx - TrD * np.power(tx, 2) + 3 * fx * (sx - tx) * np.power(sy - ty, 2) + 3 * fy * np.power(sy - ty, 3) - 6 * fz * sy * sz * ty - 4 * sy * TrD * ty + 3 * fz * sz * np.power(ty, 2) + 2 * TrD * np.power(ty, 2) - 3 * fz * np.power(sy, 2) * tz + 2 * sz * TrD * tz + 6 * fz * sy * ty * tz - 3 * fz * np.power(ty, 2) * tz - TrD * np.power(tz, 2)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,5] += (3 * (sy - ty) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,6] += (3 * (sx - tx) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,7] += (3 * (sy - ty) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      traction[xn,8] += (3 * fz * np.power(sz, 3) - (np.power(sx, 2) + np.power(sy, 2)) * TrD + 2 * np.power(sz, 2) * TrD + 2 * sx * TrD * tx - TrD * np.power(tx, 2) + 2 * sy * TrD * ty - TrD * np.power(ty, 2) + 3 * fx * (sx - tx) * np.power(sz - tz, 2) + 3 * fy * (sy - ty) * np.power(sz - tz, 2) - sz * (9 * fz * sz + 4 * TrD) * tz + (9 * fz * sz + 2 * TrD) * np.power(tz, 2) - 3 * fz * np.power(tz, 3)) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))


  return traction


@njit(parallel=True)
def StokesDLTraction(r_source, r_target, density, epsilon_distance = 1e-10):
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
  traction = np.zeros((Ntarget, 9))

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

      # p
      p = (dbzz * np.power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * np.power(sy, 2) - 3 * dbxz * sx * sz - 3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * np.power(sz, 2) - 2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * np.power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * np.power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz + 3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz - 3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * np.power(tz, 2) + dbyy * (np.power(sx, 2) - 2 * np.power(sy, 2) + np.power(sz, 2) - 2 * sx * tx + np.power(tx, 2) + 4 * sy * ty - 2 * np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2)) + dbxx * (-2 * np.power(sx, 2) + np.power(sy, 2) + np.power(sz, 2) + 4 * sx * tx - 2 * np.power(tx, 2) - 2 * sy * ty + np.power(ty, 2) - 2 * sz * tz + np.power(tz, 2))) / (4. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 2.5))

      # vx grad
      vxx = (3 * (-sx + tx) * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (-sx + tx) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)) + (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      vxy = ((-sx + tx) * (3 * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      vxz = ((-sx + tx) * (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sz - tz) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      # vy grad
      vyx = ((-sy + ty) * (3 * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      vyy = (3 * (-sy + ty) * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (-sy + ty) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)) + (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      vyz = ((-sy + ty) * (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sz - tz) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      # vz grad
      vzx = ((-sz + tz) *(3 * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sx - tx) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      vzy = ((-sz + tz) * (3 * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) + 5 * (sy - ty) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      vzz = (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) * (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-sz + tz) + (np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2)) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2)) + 5 * (sz - tz) * (-sz + tz) * (-3 * ((dbxy + dbyx) * sx * sy + dbyy * np.power(sy, 2) + (dbxz + dbzx) * sx * sz + (dbyz + dbzy) * sy * sz + dbzz * np.power(sz, 2)) - 3 * dbxx * np.power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx + 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * np.power(ty, 2) + 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz - 3 * dbzz * np.power(tz, 2))) / (8. * np.pi * np.power(np.power(sx - tx, 2) + np.power(sy - ty, 2) + np.power(sz - tz, 2), 3.5))

      traction[xn,0] += vxx + vxx - p
      traction[xn,1] += vxy + vyx
      traction[xn,2] += vxz + vzx
      traction[xn,3] += vxy + vyx
      traction[xn,4] += vyy + vyy - p
      traction[xn,5] += vyz + vzy
      traction[xn,6] += vzx + vxz
      traction[xn,7] += vzy + vyz
      traction[xn,8] += vzz + vzz - p

  return traction
