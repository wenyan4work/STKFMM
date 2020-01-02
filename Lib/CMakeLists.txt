# part 1, core library
add_library(STKFMM SHARED src/STKFMM.cpp)
target_include_directories(
  STKFMM
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${PVFMM_INCLUDE_DIR} ${MKL_INCLUDE_DIRS})
target_compile_definitions(STKFMM PUBLIC "PVFMM_FFTW3_MKL")
target_compile_options(STKFMM PUBLIC ${OpenMP_CXX_FLAGS}
                                     ${MPI_CXX_INCLUDE_DIRS})

# install core library and headers
include(GNUInstallDirs)
install(
  TARGETS STKFMM
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
