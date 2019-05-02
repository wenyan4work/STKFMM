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
def oseen_kernel_source_target_numba(r_source, r_target, density, eta = 1.0, epsilon_distance = 1e-10):
  '''
  Oseen tensor product with a density force for N points (sources and targets).
  Set to zero diagonal terms.

  u_i = sum_j G_ij * density_j 

  Input:
  r_vectors = coordinates.
  normal = vector used to contract the Stresslet (in general
        this will be the normal vector of a surface).
  density = vector used to contract the Stresslet (in general
        this will be a double layer potential).
  eta = (default 1.0) viscosity
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.

  Output:
  u = velocity, dimension (3*num_points)
  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  factor = 1.0 / (8.0 * np.pi * eta)
  print('aaa')
  r_source = r_source.reshape(Nsource, 3)
  print('bbb')
  r_target = r_target.reshape(Ntarget, 3)
  print('ccc', density.shape, density.size, Nsource)
  density = density.reshape(Nsource, 3)
  print('ddd')
  u = np.zeros((Ntarget, 3))
  print('eee')

  # Loop over targets
  for xn in prange(Ntarget):
    for yn in range(Nsource):
      x = r_target[xn, 0] - r_source[yn, 0]
      y = r_target[xn, 1] - r_source[yn, 1]
      z = r_target[xn, 2] - r_source[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     
      fr = factor / r_norm
      gr = factor / r_norm**3     
      Mxx = fr + gr * x*x
      Mxy =      gr * x*y
      Mxz =      gr * x*z
      Myy = fr + gr * y*y
      Myz =      gr * y*z
      Mzz = fr + gr * z*z

      u[xn,0] += Mxx * density[yn,0] + Mxy * density[yn,1] + Mxz * density[yn,2]
      u[xn,1] += Mxy * density[yn,0] + Myy * density[yn,1] + Myz * density[yn,2]
      u[xn,2] += Mxz * density[yn,0] + Myz * density[yn,1] + Mzz * density[yn,2]
   
  return u.flatten()


@njit(parallel=True)
def rotlet_kernel_source_target_numba(r_source, r_target, density, eta = 1.0, epsilon_distance = 1e-10):
  '''
  Oseen tensor product with a density force for N points (sources and targets).
  Set to zero diagonal terms.

  u_i = sum_j G_ij * density_j 

  Input:
  r_vectors = coordinates.
  normal = vector used to contract the Stresslet (in general
        this will be the normal vector of a surface).
  density = vector used to contract the Stresslet (in general
        this will be a double layer potential).
  eta = (default 1.0) viscosity
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.

  Output:
  u = velocity, dimension (3*num_points)
  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  factor = 1.0 / (8.0 * np.pi * eta)
  r_source = r_source.reshape(Nsource, 3)
  r_target = r_target.reshape(Ntarget, 3)
  density = density.reshape(Nsource, 3)
  u = np.zeros((Ntarget, 3))

  # Loop over targets
  for xn in prange(Ntarget):
    for yn in range(Nsource):
      x = r_target[xn, 0] - r_source[yn, 0]
      y = r_target[xn, 1] - r_source[yn, 1]
      z = r_target[xn, 2] - r_source[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     
      fr = factor / r_norm**3
      Mxy =  fr * z
      Mxz = -fr * y
      Myx = -fr * z
      Myz =  fr * x
      Mzx =  fr * y
      Mzy = -fr * x

      u[xn,0] += Mxy * density[yn,1] + Mxz * density[yn,2]
      u[xn,1] += Myx * density[yn,0] + Myz * density[yn,2]
      u[xn,2] += Mzx * density[yn,0] + Mzy * density[yn,1] 
   
  return u.flatten()


def stresslet_kernel(r_vectors, eta = 1.0, epsilon_distance = 1e-10, input_format = 'r'):
  '''
  Build the Stresslet tensor for N points (sources and targets).
  Set to zero diagonal terms.

  S_ijk = -(3/(4*pi)) * r_i * r_j * r_k / r**5

  Input:
  r_vectors = coordinates.
  eta = (default 1.0) viscosity
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.
  input_format = 'r' or 'xyz'. Only 'r' is implemented.
                 If 'r' r_vectors = (r_1, r_2, ..., r_N) with r_1 = (x_1, y_1, z_1).
                 If 'xyz' r_vector = (x_all_points, y_all_points, z_all_points).

  Output:
  S = Stresslet tensor with dimensions (3*num_points) x (3*num_points) x (3)
      Output with format
    
       | S_11 S_12 ...|  
   S = | S_12 S_12 ...|  
       | ...            |  

  with S_12 the stresslet between points (3x3x3) tensor between points 1 and 2.
  '''

  N = r_vectors.size / 3
  if input_format == 'r':
    rx = r_vectors[:,0]
    ry = r_vectors[:,1]
    rz = r_vectors[:,2]
  else:
    rx = r_vectors[0*N : 1*N]
    ry = r_vectors[1*N : 2*N]
    rz = r_vectors[2*N : 3*N]
  
  # Compute vectors between points and distance
  drx = rx - rx[:,None]
  dry = ry - ry[:,None]
  drz = rz - rz[:,None]
  dr = np.sqrt(drx**2 + dry**2 + drz**2)
  
  # Compute scalar functions f(r) and g(r)
  sel = dr > epsilon_distance
  fr = np.zeros_like(dr)
  fr[sel] = -3.0 / (4.0 * np.pi * eta * dr[sel]**5)
  
  # Compute stresslet
  S = np.zeros((r_vectors.size, r_vectors.size, 3))
  S[0::3, 0::3, 0] = fr * drx * drx * drx
  S[0::3, 0::3, 1] = fr * drx * drx * dry
  S[0::3, 0::3, 2] = fr * drx * drx * drz
  S[0::3, 1::3, 0] = fr * drx * dry * drx
  S[0::3, 1::3, 1] = fr * drx * dry * dry
  S[0::3, 1::3, 2] = fr * drx * dry * drz
  S[0::3, 2::3, 0] = fr * drx * drz * drx
  S[0::3, 2::3, 1] = fr * drx * drz * dry
  S[0::3, 2::3, 2] = fr * drx * drz * drz
  S[1::3, 0::3, 0] = fr * dry * drx * drx
  S[1::3, 0::3, 1] = fr * dry * drx * dry
  S[1::3, 0::3, 2] = fr * dry * drx * drz
  S[1::3, 1::3, 0] = fr * dry * dry * drx
  S[1::3, 1::3, 1] = fr * dry * dry * dry
  S[1::3, 1::3, 2] = fr * dry * dry * drz
  S[1::3, 2::3, 0] = fr * dry * drz * drx
  S[1::3, 2::3, 1] = fr * dry * drz * dry
  S[1::3, 2::3, 2] = fr * dry * drz * drz
  S[2::3, 0::3, 0] = fr * drz * drx * drx
  S[2::3, 0::3, 1] = fr * drz * drx * dry
  S[2::3, 0::3, 2] = fr * drz * drx * drz
  S[2::3, 1::3, 0] = fr * drz * dry * drx
  S[2::3, 1::3, 1] = fr * drz * dry * dry
  S[2::3, 1::3, 2] = fr * drz * dry * drz
  S[2::3, 2::3, 0] = fr * drz * drz * drx
  S[2::3, 2::3, 1] = fr * drz * drz * dry
  S[2::3, 2::3, 2] = fr * drz * drz * drz

  return S


def stresslet_kernel_times_normal(r_vectors, normal, eta = 1.0, epsilon_distance = 1e-10):
  '''
  Build the Stresslet tensor contracted with a vector for N points (sources and targets).
  Set to zero diagonal terms.

  S_ij = sum_k -(3/(4*pi)) * r_i * r_j * r_k * normal_k / r**5

  Input:
  r_vectors = coordinates.
  normal = vector used to contract the Stresslet (in general
           this will be the normal vector of a surface).
  eta = (default 1.0) viscosity
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.

  Output:
  S_normal = Stresslet tensor contracted with a vector with dimensions (3*num_points) x (3*num_points).

  Output with format

              | S_normal11 S_normal12 ...|  
   S_normal = | S_normal21 S_normal22 ...|  
              | ...                      |  

  with S_normal12 the stresslet between points r_1 and r_2.
  
  S_normal12 has dimensions 3 x 3.
  '''
  N = r_vectors.size / 3
  rx = r_vectors[:,0]
  ry = r_vectors[:,1]
  rz = r_vectors[:,2]
  
  # Compute vectors between points and distance
  drx = rx - rx[:,None]
  dry = ry - ry[:,None]
  drz = rz - rz[:,None]
  dr = np.sqrt(drx**2 + dry**2 + drz**2)
  
  # Compute scalar functions f(r) and g(r)
  sel = dr > epsilon_distance
  fr = np.zeros_like(dr)
  fr[sel] = 3.0 / (4.0 * np.pi * eta * dr[sel]**5)

  # Contract r_k with vector
  normal = np.reshape(normal, (normal.size // 3, 3))
  contraction = drx * normal[:,0] + dry * normal[:,1] + drz * normal[:,2]

  # Compute contracted stresslet
  Snormal = np.zeros((r_vectors.size, r_vectors.size))   
  Snormal[0::3, 0::3] = fr * drx * drx * contraction
  Snormal[0::3, 1::3] = fr * drx * dry * contraction
  Snormal[0::3, 2::3] = fr * drx * drz * contraction

  Snormal[1::3, 0::3] = fr * dry * drx * contraction
  Snormal[1::3, 1::3] = fr * dry * dry * contraction
  Snormal[1::3, 2::3] = fr * dry * drz * contraction

  Snormal[2::3, 0::3] = fr * drz * drx * contraction 
  Snormal[2::3, 1::3] = fr * drz * dry * contraction
  Snormal[2::3, 2::3] = fr * drz * drz * contraction
  
  return Snormal


@njit(parallel=False)
def stresslet_kernel_times_normal_numba(r_vectors, normal, eta = 1.0, epsilon_distance = 1e-10):
  '''
  Build the Stresslet tensor contracted with a vector for N points (sources and targets).
  Set to zero diagonal terms.

  S_ij = sum_k -(3/(4*pi)) * r_i * r_j * r_k

  Input:
  r_vectors = coordinates.
  normal = vector used to contract the Stresslet (in general
        this will be the normal vector of a surface).
  eta = (default 1.0) viscosity
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.

  Output:
  S_normal = Stresslet tensor contracted with a vector with dimensions (3*num_points) x (3*num_points).

  Output with format

              | S_normal11 S_normal12 ...|  
   S_normal = | S_normal21 S_normal22 ...|  
              | ...                      |  

  with S_normal12 the stresslet between points r_1 and r_2.
  
  S_normal12 has dimensions 3 x 3.
  '''
  # Variables
  N = r_vectors.size // 3
  factor = -3.0 / (4.0 * np.pi * eta)
  r_vectors = r_vectors.reshape(N, 3)
  normal = normal.reshape(N, 3)
  Snormal = np.zeros((3*N, 3*N))

  # Loop over targets
  for xn in prange(N):
    for yn in range(xn):
      r = r_vectors[xn] - r_vectors[yn]
      r_norm = np.linalg.norm(r)
      if r_norm < epsilon_distance:
        continue
      S = (factor * np.dot(r, normal[yn]) / r_norm**5) * np.outer(r, r)
      Snormal[3*xn:3*(xn+1), 3*yn:3*(yn+1)] = S

    for yn in range(xn+1, N):
      r = r_vectors[xn] - r_vectors[yn]
      r_norm = np.linalg.norm(r)
      if r_norm < epsilon_distance:
        continue
      S = (factor * np.dot(r, normal[yn]) / r_norm**5) * np.outer(r, r) 
      Snormal[3*xn:3*(xn+1), 3*yn:3*(yn+1)] = S
    
  return Snormal


@njit(parallel=True)
def stresslet_kernel_times_normal_times_density_numba(r_vectors, normal, density, eta = 1.0, epsilon_distance = 1e-10):
  '''
  Build the Stresslet tensor contracted with two vectors for N points (sources and targets).
  Set to zero diagonal terms.

  S_i = sum_jk -(3/(4*pi)) * r_i * r_j * r_k * density_j * normal_k / r**5

  Input:
  r_vectors = coordinates.
  normal = vector used to contract the Stresslet (in general
        this will be the normal vector of a surface).
  density = vector used to contract the Stresslet (in general
        this will be a double layer potential).
  eta = (default 1.0) viscosity
  epsilon_distance = (default 1e-10) set elements to zero for 
                     distances < epsilon_distance.

  Output:
  S_normal = Stresslet tensor contracted with two vectors with dimensions (3*num_points).
  '''
  # Variables
  N = r_vectors.size // 3
  factor = -3.0 / (4.0 * np.pi * eta)
  r_vectors = r_vectors.reshape(N, 3)
  normal = normal.reshape(N, 3)
  density = density.reshape(N, 3)
  Sdn = np.zeros((N, 3))

  # Loop over targets
  for xn in prange(N):
    for yn in range(xn):
      x = r_vectors[xn, 0] - r_vectors[yn, 0]
      y = r_vectors[xn, 1] - r_vectors[yn, 1]
      z = r_vectors[xn, 2] - r_vectors[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     
      r_inv5 = 1.0 / r_norm**5
      f0 = factor * (x*density[yn,0] + y*density[yn,1] + z*density[yn,2]) * (x*normal[yn,0] + y*normal[yn,1] + z*normal[yn,2]) * r_inv5
      Sdn[xn, 0] += f0 * x
      Sdn[xn, 1] += f0 * y
      Sdn[xn, 2] += f0 * z

    for yn in range(xn+1, N):
      x = r_vectors[xn, 0] - r_vectors[yn, 0]
      y = r_vectors[xn, 1] - r_vectors[yn, 1]
      z = r_vectors[xn, 2] - r_vectors[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue
      r_inv5 = 1.0 / r_norm**5
      f0 = factor * (x*density[yn,0] + y*density[yn,1] + z*density[yn,2]) * (x*normal[yn,0] + y*normal[yn,1] + z*normal[yn,2]) * r_inv5
      Sdn[xn, 0] += f0 * x
      Sdn[xn, 1] += f0 * y
      Sdn[xn, 2] += f0 * z
    
  return Sdn.flatten()


@njit(parallel=True)
def stresslet_kernel_source_target_numba(r_source, r_target, normal, density, eta = 1.0, epsilon_distance = 1e-10):
  '''

  '''
  # Variables
  Nsource = r_source.size // 3
  Ntarget = r_target.size // 3
  factor = -3.0 / (4.0 * np.pi * eta)
  r_source = r_source.reshape((Nsource, 3))
  r_target = r_target.reshape(Ntarget, 3)
  normal = normal.reshape(Nsource, 3)
  density = density.reshape(Nsource, 3)
  Sdn = np.zeros((Ntarget, 3))

  # Loop over targets
  for xn in prange(Ntarget):
    for yn in range(Nsource):
      x = r_target[xn, 0] - r_source[yn, 0]
      y = r_target[xn, 1] - r_source[yn, 1]
      z = r_target[xn, 2] - r_source[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     
      r_inv5 = 1.0 / r_norm**5
      f0 = factor * (x*density[yn,0] + y*density[yn,1] + z*density[yn,2]) * (x*normal[yn,0] + y*normal[yn,1] + z*normal[yn,2]) * r_inv5
      Sdn[xn, 0] += f0 * x
      Sdn[xn, 1] += f0 * y
      Sdn[xn, 2] += f0 * z
    
  return Sdn.flatten()


@njit(parallel=True)
def traction_kernel_times_normal_times_density_numba(r_vectors, normal, density, eta = 1.0, epsilon_distance = 1e-10):
  '''

  '''
  # Variables
  N = r_vectors.size // 3
  factor = -3.0 / (4.0 * np.pi * eta)
  r_vectors = r_vectors.reshape(N, 3)
  normal = normal.reshape(N, 3)
  density = density.reshape(N, 3)
  Sdn = np.zeros((N, 3))
  eta = 1.0

  # Loop over targets
  for xn in prange(N):
    for yn in range(xn):
      x = r_vectors[xn, 0] - r_vectors[yn, 0]
      y = r_vectors[xn, 1] - r_vectors[yn, 1]
      z = r_vectors[xn, 2] - r_vectors[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue     
      r_inv5 = 1.0 / r_norm**5
      f0 = factor * (x*density[yn,0] + y*density[yn,1] + z*density[yn,2]) * (x*normal[xn,0] + y*normal[xn,1] + z*normal[xn,2]) * r_inv5
      Sdn[xn, 0] += f0 * x
      Sdn[xn, 1] += f0 * y
      Sdn[xn, 2] += f0 * z

    for yn in range(xn+1, N):
      x = r_vectors[xn, 0] - r_vectors[yn, 0]
      y = r_vectors[xn, 1] - r_vectors[yn, 1]
      z = r_vectors[xn, 2] - r_vectors[yn, 2]
      r_norm = np.sqrt(x**2 + y**2 + z**2)
      if r_norm < epsilon_distance:
        continue
      r_inv5 = 1.0 / r_norm**5
      f0 = factor * (x*density[yn,0] + y*density[yn,1] + z*density[yn,2]) * (x*normal[xn,0] + y*normal[xn,1] + z*normal[xn,2]) * r_inv5
      Sdn[xn, 0] += f0 * x
      Sdn[xn, 1] += f0 * y
      Sdn[xn, 2] += f0 * z
    
  return Sdn.flatten()


@njit(parallel=True)
def complementary_kernel_times_density_numba(r_vectors, normal, density):
  '''

  '''
  # Variables
  N = normal.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  density = density.reshape((N, 3))
  u = np.zeros((N, 3))

  # Loop over targets
  for xn in prange(N):
    for yn in range(N):
      u[xn] += normal[xn] * (normal[yn,0] * density[yn,0] + normal[yn,1] * density[yn,1] + normal[yn,2] * density[yn,2])
   
  return u.flatten()

  
def complementary_kernel(r_vectors, normal):
  '''

  '''
  # Variables
  N = normal.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  Nk = np.outer(normal.flatten(), normal.flatten())
  
  return Nk


