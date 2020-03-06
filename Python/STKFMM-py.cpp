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
        .value("PVel", stkfmm::KERNEL::PVel) // single layer kernel
        .value("PVelGrad", stkfmm::KERNEL::PVelGrad)
        .value("PVelLaplacian", stkfmm::KERNEL::PVelLaplacian)
        .value("Traction", stkfmm::KERNEL::Traction)
        .value("LAPPGrad", stkfmm::KERNEL::LAPPGrad) // laplace single layer
        .value("StokesRegVel", stkfmm::KERNEL::StokesRegVel)
        .value("StokesRegVelOmega", stkfmm::KERNEL::StokesRegVelOmega)
        .value("RPY", stkfmm::KERNEL::RPY);

    py::enum_<stkfmm::PPKERNEL>(m, "PPKERNEL")
        .value("SLS2T", stkfmm::PPKERNEL::SLS2T)
        .value("DLS2T", stkfmm::PPKERNEL::DLS2T)
        .value("L2T", stkfmm::PPKERNEL::L2T);

    py::class_<stkfmm::STKFMM>(m, "STKFMM")
        .def(py::init<int, int, stkfmm::PAXIS, unsigned>())
        .def("setBox", &stkfmm::STKFMM::setBox)
        .def("showActiveKernels", &stkfmm::STKFMM::showActiveKernels)
        .def("setPoints",
             [](stkfmm::STKFMM &fmm, const int nSL,
                py::array_t<double> src_SL_coord, const int nDL,
                py::array_t<double> src_DL_coord, const int nTrg,
                py::array_t<double> trg_coord) {
                 fmm.setPoints(nSL, src_SL_coord.data(), nDL,
                               src_DL_coord.data(), nTrg, trg_coord.data());
             })
        .def("getKernelDimension",
             [](stkfmm::STKFMM &fmm, stkfmm::KERNEL kernel_) {
                 int kdimSL, kdimDL, kdimTrg;
                 fmm.getKernelDimension(kdimSL, kdimDL, kdimTrg, kernel_);
                 return std::tuple<int, int, int>(kdimSL, kdimDL, kdimTrg);
             })
        .def("setupTree", &stkfmm::STKFMM::setupTree)
        .def("evaluateFMM",
             [](stkfmm::STKFMM &fmm, const int nSL,
                py::array_t<double> src_SL_value, const int nDL,
                py::array_t<double> src_DL_value, const int nTrg,
                py::array_t<double> trg_value, const stkfmm::KERNEL kernel) {
                 // Call method
                 fmm.evaluateFMM(nSL, src_SL_value.data(), nDL,
                                 src_DL_value.data(), nTrg,
                                 (double *)trg_value.data(), kernel);
             })
        .def("isKernelActive", &stkfmm::STKFMM::isKernelActive)
        .def("clearFMM", &stkfmm::STKFMM::clearFMM)
        .def("evaluateKernel",
             [](stkfmm::STKFMM &fmm, const int nThreads,
                const stkfmm::PPKERNEL p2p, const int nSrc,
                py::array_t<double> src_coord, py::array_t<double> src_value,
                const int nTrg, py::array_t<double> trg_coord,
                py::array_t<double> trg_value, const stkfmm::KERNEL kernel) {
                 // Call method
                 fmm.evaluateKernel(nThreads, p2p, nSrc,
                                    (double *)src_coord.data(),
                                    (double *)src_value.data(), nTrg,
                                    (double *)trg_coord.data(),
                                    (double *)trg_value.data(), kernel);
             });
}
