# STKFMM
A C++ wrapper for PVFMM. No extra dependence except PVFMM.

# Features
1. Convenient. All PVFMM data structures are wrapped in a single class, with scaling functions to fit source/target points into the unit cubic box required by PVFMM 
2. Flexible. Multiple kernels can be activated.
3. Efficient. Single Layer and Double Layer potentials are simultaneously calculated through a single octree. M2M, M2L, L2L operations are combined into single layer operations only.
4. Optimized. All kernels are hand-written with AVX intrinsic instructions.
5. (To be implemented). Singly, doubly and triply periodicity in a unified interface.

# Usage
Construct an STKFMM object, with chosen BC and kernels. Multiple kernels can be activated through bitwise combination of each kernel:
```cpp
STKFMM myFMM(order, maxPts, PAXIS::NONE, KERNEL::PVel | KERNEL::LAPPGrad);
//order: numer of equivalent points on each cubic octree box edge of KIFMM
//maxPts: max number of points in an octree box 
//PAXIS::NONE: The axis of periodic BC. (To be implemented)
//KERNEL::PVel | KERNEL::LAPPGrad: A combination of supported kernels, using the bitwise | operator.
```
### Run FMM for one kernel:
```cpp
myFMM.setBox(xlow, xhigh, ylow, yhigh, zlow, zhigh); 
myFMM.setPoints(srcSLCoord, srcDLCoord, trgCoord);
auto testKernel1 = KERNEL::PVel;
myFMM.setupTree(testKernel1);
myFMM.evaluateFMM(srcSLValue, srcDLValue, trgValue, testKernel1);
```
If for another kernel the points (srcSLCoord, srcDLCoord, trgCoord) do not change, then no need to call setPoints() again.
```cpp
auto testKernel2 = KERNEL::Traction;
myFMM.setupTree(testKernel2);
myFMM.evaluateFMM(srcSLValue, srcDLValue, trgValue, testKernel2);
```

# Supported Kernel
| Kernel | SrcSL(dim) | SrcDL(dim) | Trg(dim) |
| ------ | --- |---	|---	|---	|---
|PVel |force+TrD(4)| double layer (9)| pressure,velocity (1+3)|  
|PVelGrad |force+TrD(4)| double layer (9)| pressure,velocity,gradP,gradVel (1+3+3+9)|  
|PVelLaplacian |force+TrD(4)| double layer (9)| pressure,velocity,laplacian velocity (1+3+3)|  
|Traction |force+TrD(4)| force.direction (9)| traction(9)|  
|LAPPGrad |charge(1)| double layer (3)| potential,gradpotential (1+3)|  

Here TrD means an arbitrary number performing as the trace of the double layer 3x3 matrix. The reason for including this extra dimension is to use the single layer kernel in the M2M, M2L, L2L operations for both single layer and double layer. Explaination is available in the document.

**For normal computations, set the input to single layer as (fx,fy,fz,0). The extra dimension of TrD is only for some tricky cases and internal uses.**

The `PVelLaplacian` and `Traction` kernels are provided only for convenience. The same results can be achieved by a proper combination and scaling of the kernel `PVelGrad`, since the Laplacian of velocity is just the gradient of pressure, and the traction is just a combination of gradients of pressure and velocity.

# Compile and Run tests:
If PVFMM is properly installed, you should be able to compile it by simply `make`. For documentation, simply `make doc` if you have pdflatex already installed.

To run the test driver, type:
```bash
./TestSTKFMM.X --help
```
Available test options will be displayed. MPI is supported. For example:
```bash
mpiexec -n 3 ./TestSTKFMM.X -S 1 -D 2 -T 32 -B 10 -R 0
``` 
means test the FMM on 3 mpi ranks. In total, there are 1 single layer source point, 2 double layer source point, (32+1)^3 target points distributed on a chebyshev 3D grid (-R 0), in a cubic box with edge length 10.

OpenMP parallelism is controlled by the environment variable, for example:
```bash
export OMP_NUM_THREADS=10
```

# Note
1. For double layer source strength, this interface requires the input to be a full 3x3 matrix (9 entries per point) instead of a direction vector plus a strength vector (3+3=6 entries per point). This is for compatibility with situations where the double layer kernel is applied to a general stresslet tensor, which is not always contructible from two vectors. The 3x3 matrix is flattened as (Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz) in the object `srcDLValue` as an input ot the `evaluateFMM()` function. `srcDLValue` is a `std::vector<double>` object. (Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz) can be arbitrary numbers, not limited to being trace-free or the outer product of two 3D vectors. 

2. The mathematical formulas for each kernel (Stokes kernel only) can be found in the documentation. Note that the prefactor for Stokes double layer potential is 3/8pi in this code instead of 3/4pi in the usual representation of Stokes double layer. The codes computes eq. 35 for double layer in the document, and some clarifications can be found in the document also.

3. If in question about what eactly is computed, refer to the mathematica scripts and the last section of the document in the folder `Scripts/`. They are used to symbolically generate the code in SimpleKernel.cpp, which is used to validate the FMM computation. 