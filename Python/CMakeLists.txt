# python interface, not working yet
# PVFMM should be compiled with -fPIC
if(PyInterface)

  # requried by python interface
  # find_package(Boost 1.67 REQUIRED COMPONENTS python numpy)
  find_package(PythonInterp 3 REQUIRED)
  find_package(PythonLibs 3 REQUIRED)
  find_package(pybind11 REQUIRED)

  message("Python Includes: " ${PYTHON_INCLUDE_DIRS})
  message("Python Libs: " ${PYTHON_LIBRARIES})

  # compile python module .so library
  add_library(PyPeriodicFMM SHARED Interface-py.cpp)
  set_target_properties(PyPeriodicFMM PROPERTIES PREFIX "" SUFFIX ".so")
  target_include_directories(
    PyPeriodicFMM
    PUBLIC $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
           ${PVFMM_INCLUDE_DIR} ${PVFMM_DEP_INCLUDE_DIR}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
  target_compile_definitions(PyPeriodicFMM PUBLIC "PVFMM_FFTW3_MKL")
  target_compile_options(PyPeriodicFMM PUBLIC ${OpenMP_CXX_FLAGS}
                                              ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(
    PyPeriodicFMM PUBLIC PeriodicFMM ${pybind11_LIBRARIES} ${PVFMM_LIBRARIES}
                         OpenMP::OpenMP_CXX ${MKL_LIBRARIES})

  # add example.py as
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/example.py
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/example.py
            ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example.py)
  # Custom target for activate the custom command above
  add_custom_target(copy_example_py
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/example.py)
  add_dependencies(PyPeriodicFMM copy_example_py)

  add_test(NAME PyInterface COMMAND python example.py)

endif()
