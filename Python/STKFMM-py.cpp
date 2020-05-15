#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <vector>

#include "../Lib/include/STKFMM/STKFMM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(PySTKFMM, m) {
    // Enums
    py::enum_<stkfmm::PAXIS>(m, "PAXIS")
        .value("NONE", stkfmm::PAXIS::NONE)
        .value("PX", stkfmm::PAXIS::PX)
        .value("PXY", stkfmm::PAXIS::PXY)
        .value("PXYZ", stkfmm::PAXIS::PXYZ);

    py::enum_<stkfmm::KERNEL>(m, "KERNEL")
        .value("LapPGrad", stkfmm::KERNEL::LapPGrad)
        .value("LapPGradGrad", stkfmm::KERNEL::LapPGradGrad)
        .value("LapQPGradGrad", stkfmm::KERNEL::LapQPGradGrad)
        .value("Stokes", stkfmm::KERNEL::Stokes)
        .value("RPY", stkfmm::KERNEL::RPY)
        .value("StokesRegVel", stkfmm::KERNEL::StokesRegVel)
        .value("StokesRegVelOmega", stkfmm::KERNEL::StokesRegVelOmega)
        .value("PVel", stkfmm::KERNEL::PVel)
        .value("PVelGrad", stkfmm::KERNEL::PVelGrad)
        .value("PVelLaplacian", stkfmm::KERNEL::PVelLaplacian)
        .value("Traction", stkfmm::KERNEL::Traction);

    py::enum_<stkfmm::PPKERNEL>(m, "PPKERNEL")
        .value("SLS2T", stkfmm::PPKERNEL::SLS2T)
        .value("DLS2T", stkfmm::PPKERNEL::DLS2T)
        .value("L2T", stkfmm::PPKERNEL::L2T);

    py::class_<stkfmm::Stk3DFMM>(m, "Stk3DFMM")
        .def(py::init<int, int, stkfmm::PAXIS, unsigned>())
        // .def("setBox", &stkfmm::STKFMM::setBox)
        .def("setBox",
             [](stkfmm::Stk3DFMM &fmm, py::array_t<double> origin, double len) {
                 double origina[3];
                 auto x = origin.unchecked<1>();
                 for (int i = 0; i < 3; ++i)
                     origina[i] = x(i);
                 fmm.setBox(origina, len);
             })
        .def("showActiveKernels", &stkfmm::STKFMM::showActiveKernels)
        .def("getBox", &stkfmm::STKFMM::getBox)
        .def("setPoints",
             [](stkfmm::Stk3DFMM &fmm, const int nSL, py::array_t<double> src_SL_coord, const int nTrg,
                py::array_t<double> trg_coord, const int nDL, py::array_t<double> src_DL_coord) {
                 fmm.setPoints(nSL, src_SL_coord.data(), nTrg, trg_coord.data(), nDL, src_DL_coord.data());
             })
        .def_static("getKernelDimension", &stkfmm::getKernelDimension)
        .def("setupTree", &stkfmm::STKFMM::setupTree)
        .def("evaluateFMM",
             [](stkfmm::Stk3DFMM &fmm, const stkfmm::KERNEL kernel, const int nSL, py::array_t<double> src_SL_value,
                const int nTrg, py::array_t<double> trg_value, const int nDL, py::array_t<double> src_DL_value) {
                 // Call method
                 fmm.evaluateFMM(kernel, nSL, src_SL_value.data(), nTrg, (double *)trg_value.data(), nDL,
                                 src_DL_value.data());
             })
        .def("isKernelActive", &stkfmm::STKFMM::isKernelActive)
        .def("clearFMM", &stkfmm::STKFMM::clearFMM)
        .def("evaluateKernel",
             [](stkfmm::Stk3DFMM &fmm, const stkfmm::KERNEL kernel, const int nThreads, const stkfmm::PPKERNEL p2p,
                const int nSrc, py::array_t<double> src_coord, py::array_t<double> src_value, const int nTrg,
                py::array_t<double> trg_coord, py::array_t<double> trg_value) {
                 // Call method
                 fmm.evaluateKernel(kernel, nThreads, p2p, nSrc, (double *)src_coord.data(), (double *)src_value.data(),
                                    nTrg, (double *)trg_coord.data(), (double *)trg_value.data());
             });

    py::class_<stkfmm::StkWallFMM>(m, "StkWallFMM")
        .def(py::init<int, int, stkfmm::PAXIS, unsigned>())
        .def("setBox",
             [](stkfmm::StkWallFMM &fmm, py::array_t<double> origin, double len) {
                 double origina[3];
                 auto x = origin.unchecked<1>();
                 for (int i = 0; i < 3; ++i)
                     origina[i] = x(i);
                 fmm.setBox(origina, len);
             })
        .def("showActiveKernels", &stkfmm::STKFMM::showActiveKernels)
        .def("getBox", &stkfmm::STKFMM::getBox)
        .def("setPoints",
             [](stkfmm::StkWallFMM &fmm, const int nSL, py::array_t<double> src_SL_coord, const int nTrg,
                py::array_t<double> trg_coord, const int nDL, py::array_t<double> src_DL_coord) {
                 fmm.setPoints(nSL, src_SL_coord.data(), nTrg, trg_coord.data(), nDL, src_DL_coord.data());
             })
        .def_static("getKernelDimension", &stkfmm::getKernelDimension)
        .def("setupTree", &stkfmm::STKFMM::setupTree)
        .def("evaluateFMM",
             [](stkfmm::StkWallFMM &fmm, const stkfmm::KERNEL kernel, const int nSL, py::array_t<double> src_SL_value,
                const int nTrg, py::array_t<double> trg_value, const int nDL, py::array_t<double> src_DL_value) {
                 // Call method
                 fmm.evaluateFMM(kernel, nSL, src_SL_value.data(), nTrg, (double *)trg_value.data(), nDL,
                                 src_DL_value.data());
             })
        .def("isKernelActive", &stkfmm::STKFMM::isKernelActive)
        .def("clearFMM", &stkfmm::STKFMM::clearFMM)
        .def("evaluateKernel",
             [](stkfmm::StkWallFMM &fmm, const stkfmm::KERNEL kernel, const int nThreads, const stkfmm::PPKERNEL p2p,
                const int nSrc, py::array_t<double> src_coord, py::array_t<double> src_value, const int nTrg,
                py::array_t<double> trg_coord, py::array_t<double> trg_value) {
                 // Call method
                 fmm.evaluateKernel(kernel, nThreads, p2p, nSrc, (double *)src_coord.data(), (double *)src_value.data(),
                                    nTrg, (double *)trg_coord.data(), (double *)trg_value.data());
             });
}
