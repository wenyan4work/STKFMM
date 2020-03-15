add_executable(M2LLaplace Laplace/main.cpp Laplace/Laplace1D3D.cpp
                          Laplace/Laplace2D3D.cpp Laplace/Laplace3D3D.cpp)
target_link_libraries(M2LLaplace Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX)
target_include_directories(M2LLaplace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(M2LStokeslet Stokeslet/main.cpp Stokeslet/Stokes1D3D.cpp
                            Stokeslet/Stokes2D3D.cpp Stokeslet/Stokes3D3D.cpp)
target_link_libraries(M2LStokeslet Eigen3::Eigen OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
target_include_directories(M2LStokeslet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(M2LStokesPVel StokesPVel/main.cpp StokesPvel/StokesPVel1D3D.cpp)
target_link_libraries(M2LStokesPVel Eigen3::Eigen OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
target_include_directories(M2LStokesPVel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
