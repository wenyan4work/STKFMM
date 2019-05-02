#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include "../STKFMM.h"

namespace p = boost::python;
namespace np = boost::python::numpy;


void setBox(stkfmm::STKFMM *fmm, double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh){
    fmm->setBox(xlow, xhigh, ylow, yhigh, zlow, zhigh);
}

void showActiveKernels(stkfmm::STKFMM *fmm){
    fmm->showActiveKernels();
}

void setPoints(stkfmm::STKFMM *fmm, const int nSL, np::ndarray src_SL_coord, const int nDL, np::ndarray src_DL_coord, 
               const int nTrg, np::ndarray trg_coord){

    // Transform ndarray to pointers
    double *src_SL_iter = reinterpret_cast<double *>(src_SL_coord.get_data());
    double *src_DL_iter = reinterpret_cast<double *>(src_DL_coord.get_data());
    double *trg_iter = reinterpret_cast<double *>(trg_coord.get_data());

    // Call method
    fmm->setPoints(nSL, src_SL_iter, nDL, src_DL_iter, nTrg, trg_iter);
}

p::tuple getKernelDimension(stkfmm::STKFMM *fmm, stkfmm::KERNEL kernel){
    int kdimSL, kdimDL, kdimTrg;
    fmm->getKernelDimension(kdimSL, kdimDL, kdimTrg, kernel);
    return p::make_tuple(kdimSL, kdimDL, kdimTrg);
}

void setupTree(stkfmm::STKFMM *fmm, stkfmm::KERNEL kernel){
    fmm->setupTree(kernel);
}

bool isKernelActive(stkfmm::STKFMM *fmm, stkfmm::KERNEL kernel) { 
    return fmm->isKernelActive(kernel);
}

void clearFMM(stkfmm::STKFMM *fmm, stkfmm::KERNEL kernelChoice){
    fmm->clearFMM(kernelChoice);
}

// results are added to values already in trgValuePtr
void evaluateFMM(stkfmm::STKFMM *fmm, const int nSL, np::ndarray src_SL_value, const int nDL, np::ndarray src_DL_value,
                 const int nTrg, np::ndarray trg_value, const stkfmm::KERNEL kernel){

    // Transform ndarray to pointers
    double *src_SL_iter = reinterpret_cast<double *>(src_SL_value.get_data());
    double *src_DL_iter = reinterpret_cast<double *>(src_DL_value.get_data());
    double *trg_iter = reinterpret_cast<double *>(trg_value.get_data());

    // Call method
    fmm->evaluateFMM(nSL, src_SL_iter, nDL, src_DL_iter, nTrg, trg_iter, kernel);
}

// results are added to values already in trgValuePtr.
void evaluateKernel(stkfmm::STKFMM *fmm, const int nThreads, const stkfmm::PPKERNEL p2p, const int nSrc, np::ndarray src_coord,
                    np::ndarray src_value, const int nTrg, np::ndarray trg_coord, np::ndarray trg_value, const stkfmm::KERNEL kernel){

    // Transform ndarray to pointers
    double *src_coord_iter = reinterpret_cast<double *>(src_coord.get_data());
    double *src_value_iter = reinterpret_cast<double *>(src_value.get_data());
    double *trg_coord_iter = reinterpret_cast<double *>(trg_coord.get_data());
    double *trg_value_iter = reinterpret_cast<double *>(trg_value.get_data());

    // Call method
    fmm->evaluateKernel(nThreads, p2p, nSrc, src_coord_iter, src_value_iter, nTrg, trg_coord_iter, trg_value_iter, kernel);
}




BOOST_PYTHON_MODULE(stkfmm) {
    using namespace boost::python;

    // Initialize numpy
    Py_Initialize();
    np::initialize();

    // Enums
    enum_<stkfmm::PAXIS>("PAXIS")
      .value("NONE", stkfmm::PAXIS::NONE)
      .value("PX", stkfmm::PAXIS::PX)
      .value("PXY", stkfmm::PAXIS::PXY)
      .value("PXYZ", stkfmm::PAXIS::PXYZ);

    enum_<stkfmm::KERNEL>("KERNEL")
      .value("PVel", stkfmm::KERNEL::PVel) // single layer kernel
      .value("PVelGrad", stkfmm::KERNEL::PVelGrad)
      .value("PVelLaplacian", stkfmm::KERNEL::PVelLaplacian)
      .value("Traction", stkfmm::KERNEL::Traction)
      .value("LAPPGrad", stkfmm::KERNEL::LAPPGrad); // laplace single layer

    enum_<stkfmm::PPKERNEL>("PPKERNEL")
      .value("SLS2T", stkfmm::PPKERNEL::SLS2T)
      .value("DLS2T", stkfmm::PPKERNEL::DLS2T)
      .value("L2T", stkfmm::PPKERNEL::L2T);

    // Class STKFMM
    class_<stkfmm::STKFMM>("STKFMM", init<int, int, stkfmm::PAXIS, unsigned int>());

    // Define functions for stkfmm
    def("setBox", setBox);
    def("showActiveKernels", showActiveKernels);
    def("setPoints", setPoints);
    def("getKernelDimension", getKernelDimension);
    def("setupTree", setupTree);
    def("evaluateFMM", evaluateFMM);
    def("isKernelActive", isKernelActive);
    def("clearFMM", clearFMM);
    def("evaluateKernel", evaluateKernel);    
}
