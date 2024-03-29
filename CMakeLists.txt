cmake_minimum_required(VERSION 3.10)
project(
  STKFMM
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)

# compiler flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel, disable offloading
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

# Add all the SCTL relevant flags
add_compile_options(-DSCTL_QUAD_T=__float128 -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK -DSCTL_HAVE_FFTW -I${PROJECT_SOURCE_DIR}/SCTL/include)

set(MPI_CXX_SKIP_MPICXX
    true
    CACHE BOOL "The MPI-2 C++ bindings are disabled.")
# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# library
find_package(pvfmm REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()

set(BUILD_M2L
    OFF
    CACHE BOOL "compile M2L data generator")
set(BUILD_TEST
    ON
    CACHE BOOL "compile c++ test driver")
set(BUILD_DOC
    OFF
    CACHE BOOL "build doxygen doc")
set(PyInterface
    OFF
    CACHE BOOL "build python interface")

add_subdirectory(Lib)
add_subdirectory(Demo)

if(BUILD_M2L)
  add_subdirectory(M2L)
endif()

if(BUILD_TEST)
  add_subdirectory(Test)
endif()

if(PyInterface)
  add_subdirectory(Python)
endif()

# doxygen documentation
if(BUILD_DOC)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_GENERATE_LATEX NO)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_GENERATE_MAN NO)

  # set input and output files
  set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/doc/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message("Doxygen build started")

  # note the option ALL which allows to build the docs together with the application
  add_custom_target(
    doc_doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif()
