# part 1, core library

# shared lib
add_library(STKFMM_SHARED SHARED src/FMMData.cpp src/STKFMM.cpp
                                 src/Stk3DFMM.cpp src/StkWallFMM.cpp)
target_include_directories(
  STKFMM_SHARED
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
target_link_libraries(STKFMM_SHARED PUBLIC ${PVFMM_LIB_DIR}/${PVFMM_SHARED_LIB}
                                           ${PVFMM_DEP_LIB})

target_compile_options(STKFMM_SHARED PUBLIC ${OpenMP_CXX_FLAGS})
# static lib
add_library(STKFMM_STATIC STATIC src/FMMData.cpp src/STKFMM.cpp
                                 src/Stk3DFMM.cpp src/StkWallFMM.cpp)
target_include_directories(
  STKFMM_STATIC
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR})
target_link_libraries(STKFMM_STATIC PUBLIC ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB}
                                           ${PVFMM_DEP_LIB})

target_compile_options(STKFMM_STATIC PUBLIC ${OpenMP_CXX_FLAGS})

# install core library and headers
include(GNUInstallDirs)
install(
  TARGETS STKFMM_SHARED STKFMM_STATIC
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
