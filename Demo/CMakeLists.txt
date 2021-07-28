add_executable(Demo.X Demo.cpp)
# target_include_directories(Demo.X PRIVATE ${CMAKE_SOURCE_DIR}/Util)
target_link_libraries(Demo.X PRIVATE STKFMM_STATIC OpenMP::OpenMP_CXX
                                     MPI::MPI_CXX)
