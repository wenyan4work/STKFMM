add_executable(Test3DFMM.X main3D.cpp SimpleKernel.cpp Test.cpp
                           Util/PointDistribution.cpp)
target_include_directories(Test3DFMM.X PRIVATE ${CMAKE_SOURCE_DIR}/Util)
target_link_libraries(Test3DFMM.X PRIVATE STKFMM Eigen3::Eigen
                                          OpenMP::OpenMP_CXX MPI::MPI_CXX)
