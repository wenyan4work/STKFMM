# STKFMM
A C++ library implementing mix-tree KIFMM based on pvfmm.

# What does it compute
It computes the classic kernel sum problem: for a given set of single layer sources $s^j$ at points $y_s^j$, double layer sources $d^j$ points $y_d^j$, target points $x_t$, and single layer potential $K_s$, double layer potential $K_d$:
$$p(x_t^i)=\sum_j K_s(x_t^i,y_s^j) s^j +\sum_j K_d (x_t^i,y_d^j)d^j $$

**Note** For some problems the kernels $K_s$ and $K_d$ may not be linear operators applied onto $s^j, d^j$. 

This package computes Laplace kernel $L=\dfrac{1}{4\pi r}$ and Stokeslet kernel $G_{ij}=\dfrac{1}{8\pi}\left(\dfrac{1}{r}\delta_{ij}+\dfrac{1}{r^3}r_ir_j\right)$ and their derivatives.


Here is a detailed table, in which the summation $\sum$ is dropped for clarity and the subscript indices $i,j,k,l$ denote the tensor indices.
Einstein summation and comma notation are used to simplify the expressions, for example, $G_{ij,i}f_j=\nabla\cdot (\bm{G}\cdot\bm{f})$. 

In the table:
1. **NA** means input ignored
2. **Q_{ij}, D_{ij}** are 3x3 tensors written as 9-dimension vectors in row-major format
3. $\nabla\nabla p$ is symmetric so it is written as $p_{,xx},p_{,xy},p_{,xz},p_{,yy},p_{,yz},p_{,zz}$.
4. For `RPY`, `StokesRegVel` and `StokesRegVelOmega` kernels, the parameter $b$ and $\epsilon$ can be different for each source point, and the summations are nonlinear functions of $b$ and $\epsilon$. Also $b$ and $\epsilon$ must be much smaller than the lower level leaf box of the adaptive octree, otherwise the convergence property of KIFMM is invalidated.
5. For all kernels, the electrostatic conductivity and fluid viscosity are ignored (set to 1).
6. The regularized Stokeslet is $G_{ij}^\epsilon = \dfrac{1}{8\pi}\dfrac{r^{2}+2 \epsilon^{2}}{\left(r^{2}+\epsilon^{2}\right)^{3 / 2}} \delta_{i j} f_j+\dfrac{1}{\left(r^{2}+\epsilon^{2}\right)^{3 / 2}} r_ir_jf_j$.
7. For Stokes `PVel`, `PVelGrad`, `PVelLaplacian`, and `Traction` kernels, the pressure and velocity fields are:
    $$ p=\frac{1}{4 \pi} \frac{r_{j}}{r^{3}} f_{j} + \frac{1}{4 \pi}\left(-3 \frac{r_{j} r_{k}}{r^{5}}+\frac{\delta_{j k}}{r^{3}}\right) D_{j k}, \quad u_{i}=G_{ij}f_j + \frac{1}{8 \pi \mu}\left(-\frac{r_{i}}{r^{3}} trD\right) + \frac{1}{8 \pi \mu}\left[-\frac{3 r_{i} r_{j} r_{k}}{r^{5}}\right] D_{j k} $$


| Kernel              | Single Layer Source (dim)  | Double Layer Source (dim) | Summation                                       | Target Value (dim)                                                  |
| ------------------- | -------------------------- | ------------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| `LapPGrad`          | $q$ (1)                    | $d$ (3)                   | $p=Lq-L_{,j}d_j$                                | $p,\nabla p$ (1+3)                                                  |
| `LapPGradGrad`      | $q$ (1)                    | $d$ (3)                   | $p=Lq-L_{,j}d_j$                                | $p,\nabla p, \nabla\nabla p$ (1+3+6).                               |
| `LapQPGradGrad`     | $Q_{ij}$ (9)               | NA                        | $p=L_{,ij}Q_{ij}$                               | $p,\nabla p, \nabla\nabla p$ (1+3+6).                               |
| `Stokes`            | $f_j$ (3)                  | NA                        | $u_i = G_{ij} f_j$                              | $u_i$ (3)                                                           |
| `RPY`               | $f_j,b$ (3+1)              | NA                        | $u_i = (1+\frac{1}{6}b^2\nabla^2) G_{ij} f_j$   | $u_i,\nabla^2 u_i$ (3+3)                                            |
| `StokesRegVel`      | $f_j,\epsilon$ (3+1)       | NA                        | $u_i = G_{ij}^\epsilon f_j$                     | $u_i$                                                               |
| `StokesRegVelOmega` | $f_k,n_l,\epsilon$ (3+3+1) | NA                        | See Appendix A of doi 10.1016/j.jcp.2012.12.026 | $u_i,w_j$ (3+3)                                                     |
| `PVel`              | $f_j,trD$ (3+1)            | $D_{jk}$ (9)              | see above                                       | $p,u_i$ (1+3)                                                       |
| `PVelGrad`          | $f_j,trD$ (3+1)            | $D_{jk}$ (9)              | see above                                       | $p,u_i,p_{,i},u_{i,j}$ (1+3+3+9)                                    |
| `PVelLapLacian`     | $f_j,trD$ (3+1)            | $D_{jk}$ (9)              | see above                                       | $p,u_i,u_{i,jj}$ (1+3+3)                                            |
| `Traction`          | $f_j,trD$ (3+1)            | $D_{jk}$ (9)              | see above                                       | $\sigma_{ij}=-p \delta_{i j}+\mu\left(u_{i, j}+u_{j, i}\right)$ (9) |

# Features
* All kernels are hand-written with optimized SIMD intrinsic instructions.
* Singly, doubly and triply periodicity in a unified interface.
* Support no-slip boundary condition imposed on a flat wall through image method.
* Single Layer and Double Layer potentials are simultaneously calculated through a single octree. 
* M2M, M2L, L2L operations are combined into single layer operations only.
* All PVFMM data structures are wrapped in a single class.
* Multiple kernels can be activated simultaneously.
* Complete MPI and OpenMP support.

# Usage
This library defines an abstract base class `STKFMM` for the common interface and utility functions. Two concrete derived classes `Stk3DFMM` and `StkWallFMM` are defined for two separate cases: 3D spatial FMM and Stokes FMM with no-slip boundary condition imposed on a flat wall.

For details of usage, look at the function `runFMM()` in `Test/Test.cpp`.

Instructions here.
### Step 0 Decide BC and Kernels to use
```cpp
PAXIS paxis = PAXIS::NONE; // or other bc
int k = KERNEL::Stokes | KERNEL::RPY; // bitwise | operator, other combinations also work
```

### Step 1 Construct an object
Construct an STKFMM object, with chosen BC and kernels, depending on if you need the no-slip wall.
```cpp
std::shared_ptr<STKFMM> fmmPtr;
if (wall) {
    fmmPtr = std::make_shared<StkWallFMM>(p, maxPoints, paxis, k);
} else {
    fmmPtr = std::make_shared<Stk3DFMM>(p, maxPoints, paxis, k);
}
```
* `order`: number of equivalent points on each cubic octree box edge of KIFMM, usually chosen from $8,10,12$. This affects the trade of between accuracy and computation time.
* `maxPts`: max number of points in an octree leaf box, usually $500\sim2000$. This affects the depth of adaptive octree, thus the computation time.
* `PAXIS::NONE`: the axis of periodic BC. For periodic boundary conditions, replace `NONE` with `PX`, `PXY`, or `PXYZ`.
* `KERNEL::PVel | KERNEL::LAPPGrad`: A combination of supported kernels, using the | `bitwise or` operator.

### Step 2 Specify the box and source/target points
```cpp
double origin[3] = {x0, y0, z0};
fmmPtr->setBox(origin, box);
```
* if both SL and DL points exist:
```cpp
fmmPtr->setPoints(nSL, point.srcLocalSL.data(), nTrg, point.trgLocal.data(), nDL, point.srcLocalDL.data());
```
* if no DL points:
```cpp
fmmPtr->setPoints(nSL, point.srcLocalSL.data(), nTrg, point.trgLocal.data());
```
* For `Stk3DFMM`, all points must in the cube defined by [x0,x0+box)$\times$[y0,y0+box)$\times$[z0,z0+box)
* For `StkWallFMM`, all points must in the half cube defined by [x0,x0+box)$\times$[y0,y0+box)$\times$[z0,z0+box/2) 

### Step 3 Run FMM for one kernel:
```cpp
fmmPtr->setupTree(KERNEL::Stokes);
fmmPtr->evaluateFMM(kernel, nSL, value.srcLocalSL.data(), nTrg, trgLocal.data(), nDL, value.srcLocalDL.data());
```
* `nDL` and the values for DL sources will be ignored if the chosen kernel does not support DL. 

# Supported kernels and boundary conditions
In these tables
* `SL Neutral` means the summation of each component of SL sources within the box must be zero
* $trD$ Neutral means the summation of $trD$ within the box must be zero
* $D_{jj}$ Neutral means the summation of trace of DL sources $D_{jk}$ within the box must be zero
* `Yes` means no requirements
### `Stk3DFMM`
| Kernel              | `PNONE` | `PX`                 | `PXY`                | `PXYZ`                    |
| ------------------- | ------- | -------------------- | -------------------- | ------------------------- |
| `LapPGrad`          | Yes     | SL Neutral           | SL Neutral           | SL Neutral                |
| `LapPGradGrad`      | Yes     | SL Neutral           | SL Neutral           | SL Neutral                |
| `LapQPGradGrad`     | Yes     | SL Neutral           | SL Neutral           | SL Neutral                |
| `Stokes`            | Yes     | SL Neutral           | SL Neutral           | Yes                       |
| `RPY`               | Yes     | SL Neutral           | SL Neutral           | Yes                       |
| `StokesRegVel`      | Yes     | SL Neutral           | SL Neutral           | Yes                       |
| `StokesRegVelOmega` | Yes     | SL Neutral           | SL Neutral           | Yes Neutral               |
| `PVel`              | Yes     | SL, $D_{jj}$ Neutral | SL, $D_{jj}$ Neutral | $trD$,  $D_{jj}$  Neutral |
| `PVelGrad`          | Yes     | SL, $D_{jj}$ Neutral | SL, $D_{jj}$ Neutral | $trD$, $D_{jj}$  Neutral  |
| `PVelLapLacian`     | Yes     | SL, $D_{jj}$ Neutral | SL, $D_{jj}$ Neutral | $trD$, $D_{jj}$  Neutral  |
| `Traction`          | Yes     | SL, $D_{jj}$ Neutral | SL, $D_{jj}$ Neutral | $trD$, $D_{jj}$  Neutral  |

### `StkWallFMM`
| Kernel   | `PNONE` | `PX` | `PXY` | `PXYZ` |
| -------- | ------- | ---- | ----- | ------ |
| `Stokes` | Yes     | Yes  | Yes   | No     |
| `RPY`    | Yes     | Yes  | Yes   | No     |

# Compile and Run tests:
Install the following:
* `new_BC` branch of `pvfmm`
* `Eigen` if you want to generate periodicity precomputed data yourself


If PVFMM is properly installed, you should be able to compile this project using the `CMakeLists.txt`. The script `do-cmake.sh` is an example of how to invoke `cmake` command with optional features (python interface and doxygen documentation). 

To run the test driver, go to the build folder and type:
```bash
./Test/Test3DFMM.X --help
./Test/TestWallFMM.X --help
```
For possible test options.

For large scale convergence tests of all possible BCs (roughly ~100GB of memory will be used and a lot of precomputed data will be generated for the first run):
```bash
./Test/Test3DFMM.X -S 96 -D 96 -T 96 -B 50 -K 0 -m 2000 -V 0 -P 0
./Test/Test3DFMM.X -S 96 -D 96 -T 96 -B 50 -K 0 -m 2000 -V 0 -P 1
./Test/Test3DFMM.X -S 96 -D 96 -T 96 -B 50 -K 0 -m 2000 -V 0 -P 2
./Test/Test3DFMM.X -S 96 -D 96 -T 96 -B 50 -K 0 -m 2000 -V 0 -P 3
```
For `TestWallFMM.X`, since only two kernels are supported so you cannot specify `-K 0`. To test Stokes image kernel, use `-K 8` and to test RPY image kernel, use `-K 16`.
```bash
./Test/TestWallFMM.X -S 96 -D 0 -T 96 -B 50 -K 0 -m 2000 -V 1 -P 0
./Test/TestWallFMM.X -S 96 -D 0 -T 96 -B 50 -K 0 -m 2000 -V 1 -P 1
./Test/TestWallFMM.X -S 96 -D 0 -T 96 -B 50 -K 0 -m 2000 -V 1 -P 2
```
**Note** If your machine's memory is limited (<24GB), use smaller number of points and test one kernel at a time. 

# Acknowledgement
Dhairya Malhotra and Alex Barnett for useful coding instructions and discussions.
