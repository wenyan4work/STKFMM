add_executable(M2LLaplace src/main.cpp src/Laplace1D3D.cpp src/Laplace2D3D.cpp
                          src/Laplace3D3D.cpp)

target_link_libraries(M2LLaplace Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX)
target_include_directories(M2LLaplace PRIVATE ${CMAKE_SOURCE_DIR}/Util)
