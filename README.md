<img src="./STKFMM_Logo_RGB.svg" width="200">

###

A C++ library implements the [Kernel Aggregated Fast Multipole Method](https://link.springer.com/article/10.1007/s10444-021-09896-1) based on the library PVFMM.

# What does it compute

It computes the classic kernel sum problem: for a given set of single layer sources <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1069b8abbb5837aa1e07cd46c48ff62d.svg?invert_in_darkmode" align=middle width=13.80998849999999pt height=27.15900329999998pt/> at points <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/12dd280bcf2f8b88954a119a6fe0cc82.svg?invert_in_darkmode" align=middle width=14.75371589999999pt height=27.15900329999998pt/>, double layer sources <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/0ec0e7629c233c51a807937c9c2e0008.svg?invert_in_darkmode" align=middle width=14.66047274999999pt height=27.15900329999998pt/> points <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/4ed9389b413af04e9786fa6e147ddbdb.svg?invert_in_darkmode" align=middle width=14.902509599999991pt height=31.02729300000001pt/>, target points <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/23776aad854f2d33e83e4f4cad44e1b9.svg?invert_in_darkmode" align=middle width=14.360779949999989pt height=14.15524440000002pt/>, and single layer potential <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1fc018edd54a76a01783d1cf35676916.svg?invert_in_darkmode" align=middle width=20.16558224999999pt height=22.465723500000017pt/>, double layer potential <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c4dd4df1478960c5f0d78f517ad773e5.svg?invert_in_darkmode" align=middle width=20.804288999999986pt height=22.465723500000017pt/>:
<p align="center"><img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/aab7f6d0d35c1902f2a8b8ac1cc3061a.svg?invert_in_darkmode" align=middle width=304.96957425pt height=38.89287435pt/></p>

**Note** For some problems the kernels <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1fc018edd54a76a01783d1cf35676916.svg?invert_in_darkmode" align=middle width=20.16558224999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c4dd4df1478960c5f0d78f517ad773e5.svg?invert_in_darkmode" align=middle width=20.804288999999986pt height=22.465723500000017pt/> may not be linear operators applied onto <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/df06f340b9915e0682b914a0b1de03b9.svg?invert_in_darkmode" align=middle width=36.59823914999999pt height=27.15900329999998pt/>.

This package computes Laplace kernel <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/b1f2675e5b0e8444482d1bdfac266e90.svg?invert_in_darkmode" align=middle width=61.129723049999996pt height=43.42856099999997pt/> and Stokeslet kernel <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1492e52db7f896468254a8034fcbf840.svg?invert_in_darkmode" align=middle width=193.04796389999998pt height=47.6716218pt/> and their derivatives.

Here is a detailed table, in which the summation <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/a5a3c89b53bed887e7e194b0670abc9a.svg?invert_in_darkmode" align=middle width=17.35165739999999pt height=24.657735299999988pt/> is dropped for clarity and the subscript indices <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/d4b5cf8f522b37d2d4a1d1ee619261ec.svg?invert_in_darkmode" align=middle width=48.68176169999998pt height=22.831056599999986pt/> denote the tensor indices.
Einstein summation and comma notation are used to simplify the expressions, for example, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/b6d71cb440aa28c10ccfa14da6d8700e.svg?invert_in_darkmode" align=middle width=141.90060884999997pt height=24.65753399999998pt/>.

In the table:

1. **NA** means input ignored
2. **Q*{ij}, D*{ij}** are 3x3 tensors written as 9-dimension vectors in row-major format
3. <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1abfc937b0f1b385c8c69b2730a6cda6.svg?invert_in_darkmode" align=middle width=35.667911399999994pt height=22.465723500000017pt/> is symmetric so it is written as <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/a04fd15dcb86e6e1eadfc08d64cb38d6.svg?invert_in_darkmode" align=middle width=198.83223689999997pt height=14.15524440000002pt/>.
4. For `RPY`, `StokesRegVel` and `StokesRegVelOmega` kernels, the parameter <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054796099999991pt height=22.831056599999986pt/> and <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/> can be different for each source point, and the summations are nonlinear functions of <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054796099999991pt height=22.831056599999986pt/> and <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/>. Also <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054796099999991pt height=22.831056599999986pt/> and <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/> must be much smaller than the lower level leaf box of the adaptive octree, otherwise the convergence property of KIFMM is invalidated.
5. For all kernels, the electrostatic conductivity and fluid viscosity are ignored (set to 1).
6. The regularized Stokeslet is <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1a0ddfa5e81ec1e06ac5c4b3e4530e4e.svg?invert_in_darkmode" align=middle width=337.4336229pt height=49.00309590000003pt/>.
7. For Stokes `PVel`, `PVelGrad`, `PVelLaplacian`, and `Traction` kernels, the pressure and velocity fields are:
   $$
      p=\frac{1}{4 \pi} \frac{r_{j}}{r^{3}} f_{j} + \frac{1}{4 \pi}\left(-3 \frac{r_{j} r_{k}}{r^{5}}+\frac{\delta_{j k}}{r^{3}}\right) D_{j k}, \quad u_{i}=G_{ij}f_j + \frac{1}{8 \pi \mu}\left(-\frac{r_{i}}{r^{3}} trD\right) + \frac{1}{8 \pi \mu}\left[-\frac{3 r_{i} r_{j} r_{k}}{r^{5}}\right] D_{j k}
   $$

| Kernel              | Single Layer Source (dim)  | Double Layer Source (dim) | Summation                                       | Target Value (dim)                                                  |
| ------------------- | -------------------------- | ------------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| `LapPGrad`          | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode" align=middle width=7.928106449999989pt height=14.15524440000002pt/> (1)                    | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> (3)                   | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/080e940370a8293ef0ea9c02e8836013.svg?invert_in_darkmode" align=middle width=106.07294609999998pt height=22.831056599999986pt/>                                | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/23bbbdfa14b7ee7a030d0c04fd38250a.svg?invert_in_darkmode" align=middle width=37.545688949999985pt height=22.465723500000017pt/> (1+3)                                                  |
| `LapPGradGrad`      | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode" align=middle width=7.928106449999989pt height=14.15524440000002pt/> (1)                    | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> (3)                   | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/080e940370a8293ef0ea9c02e8836013.svg?invert_in_darkmode" align=middle width=106.07294609999998pt height=22.831056599999986pt/>                                | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/e714d66356b6c29eeee3f7985e73c67f.svg?invert_in_darkmode" align=middle width=80.51948189999999pt height=22.465723500000017pt/> (1+3+6).                               |
| `LapQPGradGrad`     | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/0b5c36a960bf1e20da870975949caf38.svg?invert_in_darkmode" align=middle width=23.75083259999999pt height=22.465723500000017pt/> (9)               | NA                        | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/a4256cdebe78f0dbefcbeef82a7adb35.svg?invert_in_darkmode" align=middle width=80.60766449999998pt height=22.465723500000017pt/>                               | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/e714d66356b6c29eeee3f7985e73c67f.svg?invert_in_darkmode" align=middle width=80.51948189999999pt height=22.465723500000017pt/> (1+3+6).                               |
| `Stokes`            | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/ac9424c220341fa74016e5769014f456.svg?invert_in_darkmode" align=middle width=14.152495499999992pt height=22.831056599999986pt/> (3)                  | NA                        | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/00ab3fb1d4498352d564322b3d8281ab.svg?invert_in_darkmode" align=middle width=75.45512204999999pt height=22.831056599999986pt/>                              | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/194516c014804d683d1ab5a74f8c5647.svg?invert_in_darkmode" align=middle width=14.061172949999989pt height=14.15524440000002pt/> (3)                                                           |
| `RPY`               | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/07e476cc0252962199ca482cc8788e94.svg?invert_in_darkmode" align=middle width=29.335069499999992pt height=22.831056599999986pt/> (3+1)              | NA                        | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/2f5303a6c997e9f9a4ebdb12d00348c1.svg?invert_in_darkmode" align=middle width=162.55108155pt height=27.77565449999998pt/>   | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/79a624f595dfa02aaede80594ce7a077.svg?invert_in_darkmode" align=middle width=57.323257199999986pt height=26.76175259999998pt/> (3+3)                                            |
| `StokesRegVel`      | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0e8acfba65f15b77a9457b9e727c409.svg?invert_in_darkmode" align=middle width=28.952665499999988pt height=22.831056599999986pt/> (3+1)       | NA                        | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/199f58fd308548442348e4a586184098.svg?invert_in_darkmode" align=middle width=75.45512204999999pt height=22.831056599999986pt/>                     | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/194516c014804d683d1ab5a74f8c5647.svg?invert_in_darkmode" align=middle width=14.061172949999989pt height=14.15524440000002pt/>                                                               |
| `StokesRegVelOmega` | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/a499cec8684687006cb7f8267c392af8.svg?invert_in_darkmode" align=middle width=52.332649049999986pt height=22.831056599999986pt/> (3+3+1) | NA                        | See Appendix A of doi 10.1016/j.jcp.2012.12.026 | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/3d9f8b131aee6306786da85243ef8109.svg?invert_in_darkmode" align=middle width=40.061970299999984pt height=14.15524440000002pt/> (3+3)                                                     |
| `PVel`              | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/635bf1b5a0d6b2f1f190d90b7ceb4060.svg?invert_in_darkmode" align=middle width=50.15555654999999pt height=22.831056599999986pt/> (3+1)            | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/34be5e6cbc28b74e8f561c1d527644ce.svg?invert_in_darkmode" align=middle width=26.98011194999999pt height=22.465723500000017pt/> (9)              | see above                                       | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/912607c89eae037134fdf3e74d602929.svg?invert_in_darkmode" align=middle width=29.63762174999999pt height=14.15524440000002pt/> (1+3)                                                       |
| `PVelGrad`          | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/635bf1b5a0d6b2f1f190d90b7ceb4060.svg?invert_in_darkmode" align=middle width=50.15555654999999pt height=22.831056599999986pt/> (3+1)            | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/34be5e6cbc28b74e8f561c1d527644ce.svg?invert_in_darkmode" align=middle width=26.98011194999999pt height=22.465723500000017pt/> (9)              | see above                                       | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/75d6f59e125d5e0ccc1984cd60ecbaca.svg?invert_in_darkmode" align=middle width=86.78857274999999pt height=14.15524440000002pt/> (1+3+3+9)                                    |
| `PVelLapLacian`     | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/635bf1b5a0d6b2f1f190d90b7ceb4060.svg?invert_in_darkmode" align=middle width=50.15555654999999pt height=22.831056599999986pt/> (3+1)            | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/34be5e6cbc28b74e8f561c1d527644ce.svg?invert_in_darkmode" align=middle width=26.98011194999999pt height=22.465723500000017pt/> (9)              | see above                                       | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/9a709c64f59f11dbeaecdd88b3339783.svg?invert_in_darkmode" align=middle width=67.93970369999998pt height=14.15524440000002pt/> (1+3+3)                                            |
| `Traction`          | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/635bf1b5a0d6b2f1f190d90b7ceb4060.svg?invert_in_darkmode" align=middle width=50.15555654999999pt height=22.831056599999986pt/> (3+1)            | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/34be5e6cbc28b74e8f561c1d527644ce.svg?invert_in_darkmode" align=middle width=26.98011194999999pt height=22.465723500000017pt/> (9)              | see above                                       | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/1519c3ecab000d56ee33670b9426c1ed.svg?invert_in_darkmode" align=middle width=197.51520194999998pt height=24.65753399999998pt/> (9) |

# Features

- All kernels are hand-written with optimized SIMD intrinsic instructions.
- Singly, doubly and triply periodicity in a unified interface.
- Support no-slip boundary condition imposed on a flat wall through image method.
- Single Layer and Double Layer potentials are simultaneously calculated through a single octree.
- M2M, M2L, L2L operations are combined into single layer operations only.
- All PVFMM data structures are wrapped in a single class.
- Multiple kernels can be activated simultaneously.
- Complete MPI and OpenMP support.

# Usage

This library defines an abstract base class `STKFMM` for the common interface and utility functions. Two concrete derived classes `Stk3DFMM` and `StkWallFMM` are defined for two separate cases: 3D spatial FMM and Stokes FMM with no-slip boundary condition imposed on a flat wall.

For details of usage, look at the function `runFMM()` in `Test/Test.cpp`.

Instructions here.

### Step 0 Decide BC and Kernels to use

```cpp
PAXIS paxis = PAXIS::NONE; // or other bc
int k = asInteger(KERNEL::Stokes) | asInteger(KERNEL::RPY); // bitwise | operator, other combinations also work
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

- `order`: number of equivalent points on each cubic octree box edge of KIFMM, usually chosen from <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c37ded03564c90141c5f1e058edc4ab8.svg?invert_in_darkmode" align=middle width=55.70781314999999pt height=21.18721440000001pt/>. This affects the trade of between accuracy and computation time.
- `maxPts`: max number of points in an octree leaf box, usually <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/3ce145d17b292a694572c25966e7805f.svg?invert_in_darkmode" align=middle width=79.45209689999999pt height=21.18721440000001pt/>. This affects the depth of adaptive octree, thus the computation time.
- `PAXIS::NONE`: the axis of periodic BC. For periodic boundary conditions, replace `NONE` with `PX`, `PXY`, or `PXYZ`.
- `KERNEL::PVel | KERNEL::LAPPGrad`: A combination of supported kernels, using the | `bitwise or` operator.

### Step 2 Specify the box and source/target points

```cpp
double origin[3] = {x0, y0, z0};
fmmPtr->setBox(origin, box);
```

- if both SL and DL points exist:

```cpp
fmmPtr->setPoints(nSL, point.srcLocalSL.data(), nTrg, point.trgLocal.data(), nDL, point.srcLocalDL.data());
```

- if no DL points:

```cpp
fmmPtr->setPoints(nSL, point.srcLocalSL.data(), nTrg, point.trgLocal.data());
```

- For `Stk3DFMM`, all points must in the cube defined by [x0,x0+box)<img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bdbf342b57819773421273d508dba586.svg?invert_in_darkmode" align=middle width=12.785434199999989pt height=19.1781018pt/>[y0,y0+box)<img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bdbf342b57819773421273d508dba586.svg?invert_in_darkmode" align=middle width=12.785434199999989pt height=19.1781018pt/>[z0,z0+box)
- For `StkWallFMM`, all points must in the half cube defined by [x0,x0+box)<img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bdbf342b57819773421273d508dba586.svg?invert_in_darkmode" align=middle width=12.785434199999989pt height=19.1781018pt/>[y0,y0+box)<img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bdbf342b57819773421273d508dba586.svg?invert_in_darkmode" align=middle width=12.785434199999989pt height=19.1781018pt/>[z0,z0+box/2), and the no-slip boundary condition is always imposed at the z0 plane.

### Step 3 Run FMM for one kernel:

```cpp
fmmPtr->setupTree(KERNEL::Stokes);
fmmPtr->evaluateFMM(kernel, nSL, value.srcLocalSL.data(), nTrg, trgLocal.data(), nDL, value.srcLocalDL.data());
```

- `nDL` and the values for DL sources will be ignored if the chosen kernel does not support DL.

# Supported kernels and boundary conditions

In these tables

- `SL Neutral` means the summation of each component of SL sources within the box must be zero
- <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bee1683ac5a86212efac5d2804145b0f.svg?invert_in_darkmode" align=middle width=27.87528314999999pt height=22.465723500000017pt/> Neutral means the summation of <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bee1683ac5a86212efac5d2804145b0f.svg?invert_in_darkmode" align=middle width=27.87528314999999pt height=22.465723500000017pt/> within the box must be zero
- <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral means the summation of trace of DL sources <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/34be5e6cbc28b74e8f561c1d527644ce.svg?invert_in_darkmode" align=middle width=26.98011194999999pt height=22.465723500000017pt/> within the box must be zero
- `Yes` means no requirements

### `Stk3DFMM`

| Kernel              | `PNONE` | `PX`                 | `PXY`                | `PXYZ`                  |
| ------------------- | ------- | -------------------- | -------------------- | ----------------------- |
| `LapPGrad`          | Yes     | SL Neutral           | SL Neutral           | SL Neutral              |
| `LapPGradGrad`      | Yes     | SL Neutral           | SL Neutral           | SL Neutral              |
| `LapQPGradGrad`     | Yes     | SL Neutral           | SL Neutral           | SL Neutral              |
| `Stokes`            | Yes     | SL Neutral           | SL Neutral           | Yes                     |
| `RPY`               | Yes     | SL Neutral           | SL Neutral           | Yes                     |
| `StokesRegVel`      | Yes     | SL Neutral           | SL Neutral           | Yes                     |
| `StokesRegVelOmega` | Yes     | SL Neutral           | SL Neutral           | Yes Neutral             |
| `PVel`              | Yes     | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bee1683ac5a86212efac5d2804145b0f.svg?invert_in_darkmode" align=middle width=27.87528314999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral |
| `PVelGrad`          | Yes     | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bee1683ac5a86212efac5d2804145b0f.svg?invert_in_darkmode" align=middle width=27.87528314999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral |
| `PVelLapLacian`     | Yes     | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bee1683ac5a86212efac5d2804145b0f.svg?invert_in_darkmode" align=middle width=27.87528314999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral |
| `Traction`          | Yes     | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | SL, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral | <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/bee1683ac5a86212efac5d2804145b0f.svg?invert_in_darkmode" align=middle width=27.87528314999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/wenyan4work/STKFMM/None/svgs/c0b7bdafbb8aef85d4275c543c04eeb7.svg?invert_in_darkmode" align=middle width=25.81859279999999pt height=22.465723500000017pt/> Neutral |

### `StkWallFMM`

| Kernel   | `PNONE` | `PX` | `PXY` | `PXYZ` |
| -------- | ------- | ---- | ----- | ------ |
| `Stokes` | Yes     | Yes  | Yes   | No     |
| `RPY`    | Yes     | Yes  | Yes   | No     |

# Compile and Run tests:

## Prerequisite:

- Install the `develop` branch (b9de1a) of `pvfmm` by cmake. If you install `pvfmm` by gnu automake you will have to manually help `STKFMM` discover `pvfmm`.

If PVFMM is properly installed, you should be able to compile this project using the `CMakeLists.txt`. The script `do-cmake.sh` is an example of how to invoke `cmake` command with optional features (python interface and doxygen documentation).

To run the test driver, go to the build folder and type:

```bash
./Test/TestFMM.X --help
Test Driver for Stk3DFMM and StkWallFMM

Usage: ./Test/TestFMM.X [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --config                    config file name
  -S,--nsl INT                number of source SL points
  -D,--ndl INT                number of source DL points
  -T,--ntrg INT               number of source TRG points
  -B,--box FLOAT              testing cubic box edge length
  -O,--origin [FLOAT,FLOAT,FLOAT]
                              testing cubic box origin point
  -K,--kernel INT             test which kernels
  -P,--pbc INT                periodic boundary condition. 0=none, 1=PX, 2=PXY, 3=PXYZ
  -M,--maxOrder INT           max KIFMM order, must be even number. Default 16.
  --eps FLOAT                 epsilon or a for Regularized and RPY kernels
  --max INT                   max number of points in an octree leaf box
  --seed INT                  seed for random number generator
  --dist [FLOAT,FLOAT]        parameters for the random distribution
  --type INT                  type of random distribution, Uniform = 1, LogNormal = 2, Gaussian = 3, Ellipse = 4
  --direct,--no-direct{false} run O(N^2) direct summation with S2T kernels
  --verify,--no-verify{false} verify results with O(N^2) direct summation
  --convergence,--no-convergence{false}
                              calculate convergence error relative to FMM at maxOrder
  --random,--no-random{false} use random points, otherwise regular mesh
  --dump,--no-dump{false}     write src/trg coord and values to files
  --wall,--no-wall{false}     test StkWallFMM, otherwise Stk3DFMM
```

For possible test options. Several test configuration files are included in the folder `Config`, and can be loaded by `TestFMM.X` as this:

```
./Test/TestFMM.X --config ../Config/Verify.toml
```

For large scale convergence tests of all possible BCs (roughly ~100GB of memory will be used and a lot of precomputed data will be generated for the first run):

```bash
./Test/TestFMM.X --config ../Config/BenchP0.toml
```

The options in the config toml file can be overridden by extra flags, for example, use other boundary conditions:

```bash
./Test/TestFMM.X --config ../Config/BenchP0.toml -P 1
./Test/TestFMM.X --config ../Config/BenchP0.toml -P 2
./Test/TestFMM.X --config ../Config/BenchP0.toml -P 3
```

`TestFMM.X` will write a `TestLog.json` file, which can be loaded into python for convenient performance/accuracy analysis and plotting.

**Note** If your machine's memory is limited (<24GB), use smaller number of points and test one kernel at a time.

## Optional:

`STKFMM` has a few optional features that can be turned on or off during the cmake configuration stage with the following switches:

```bash
  -D BUILD_TEST=ON \
  -D BUILD_DOC=OFF \
  -D BUILD_M2L=OFF \
  -D PyInterface=OFF \
```

By default, only the `BUILD_TEST` is turned on.

- If you need doxygen document, set `BUILD_DOC=ON`.
- If you want to generate periodicity precomputed `M2L` data yourself, set `BUILD_M2L=ON`. In this case you will have to install the linear algebra library `Eigen`. If you do not want to generate periodicity precomputed data yourself, you can download the `M2C.7z` file from `https://zenodo.org/record/6338525#.YijCaXrMJD8` and unzip all data files to folder `$PVFMM_DIR/pdata`.
- If you want to call this library from python, set `PyInterface=ON`. In this case you need some basic python facilities. Here is a basic example for `requirements.txt` used for python virtualenv:

```
argh==0.26.2
h5py==2.9.0
llvmlite==0.32.1
mpi4py==3.0.2
numba==0.49.1
numpy==1.18.4
scipy==1.4.1
six==1.15.0
```

# Acknowledgement

Dhairya Malhotra and Alex Barnett for useful coding instructions and discussions.
