add_executable(TestFMM.X main.cpp SimpleKernel.cpp Test.cpp
                         Util/PointDistribution.cpp)
target_include_directories(TestFMM.X PRIVATE ${CMAKE_SOURCE_DIR}/Util)
target_link_libraries(TestFMM.X PRIVATE STKFMM_STATIC Eigen3::Eigen
                                        OpenMP::OpenMP_CXX MPI::MPI_CXX)
