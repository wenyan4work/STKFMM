# part 1, core library
add_library(STKFMM SHARED src/FMMData.cpp src/STKFMM.cpp src/Stk3DFMM.cpp
                          src/StkWallFMM.cpp)
target_include_directories(
  STKFMM
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
target_link_libraries(STKFMM PUBLIC ${PVFMM_LIB_DIR}/${PVFMM_SHARED_LIB}
                                    ${PVFMM_DEP_LIB})

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
