add_executable(STKFMM.X main.cpp SimpleKernel.cpp Util/PointDistribution.cpp)
target_compile_definitions(STKFMM.X PRIVATE "PVFMM_FFTW3_MKL")
target_include_directories(STKFMM.X PRIVATE ${CMAKE_SOURCE_DIR}/Util)
target_link_libraries(
  STKFMM.X PRIVATE STKFMM ${PVFMM_STATIC_LIBRARY} ${MKL_LIBRARIES}
                   Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX)
