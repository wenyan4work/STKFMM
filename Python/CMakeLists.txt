if(PyInterface)
  find_package(pybind11 REQUIRED)

  message("Python Includes: " ${PYTHON_INCLUDE_DIRS})
  message("Python Libs: " ${PYTHON_LIBRARIES})

  # compile python module .so library
  add_library(PySTKFMM SHARED STKFMM-py.cpp)
  set_target_properties(PySTKFMM PROPERTIES PREFIX "" SUFFIX ".so")
  target_include_directories(
    PySTKFMM
    PUBLIC $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
           ${PVFMM_INCLUDE_DIR} ${PVFMM_DEP_INCLUDE_DIR}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
  target_compile_options(PySTKFMM PUBLIC ${OpenMP_CXX_FLAGS}
                                         ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(
    PySTKFMM PUBLIC STKFMM_SHARED ${pybind11_LIBRARIES} ${PVFMM_SHARED_LIBRARY}
                    OpenMP::OpenMP_CXX ${PVFMM_DEP_LIB} MPI::MPI_CXX)

  # add example.py as
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/example.py
    COMMAND
      ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/example.py
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels.py
      ${CMAKE_CURRENT_SOURCE_DIR}/timer.py ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example.py)
  # Custom target for activate the custom command above
  add_custom_target(copy_example_py
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/example.py)
  add_dependencies(PySTKFMM copy_example_py)

  add_test(NAME PyInterface COMMAND python example.py)

  install(TARGETS PySTKFMM LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python)

endif()
