if(PyInterface)
  # FIXME: We should just move all python stuff to pip
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/STKFMM.py DESTINATION ${CMAKE_INSTALL_LIBDIR}/python)

endif()
