if(PyInterface)
  add_custom_target(PySTKFMM)

  # add example.py as
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/example.py
    COMMAND
      ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/example.py
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels.py
      ${CMAKE_CURRENT_SOURCE_DIR}/STKFMM.py
      ${CMAKE_CURRENT_SOURCE_DIR}/timer.py ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example.py)
  # Custom target for activate the custom command above
  add_custom_target(copy_example_py
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/example.py)
  add_dependencies(PySTKFMM copy_example_py)

  add_test(NAME PyInterface COMMAND python3 example.py)

  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/STKFMM.py DESTINATION ${CMAKE_INSTALL_LIBDIR}/python)

endif()
