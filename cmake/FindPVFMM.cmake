#  Copyright Olivier Parcollet 2010.
#  Copyright Simons Foundation 2019
#    Author: Nils Wentzell
#    Customized for PVFMM by: Robert Blackwell, Wen Yan

#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

#
# This module looks for PVFMM.
# It sets up : PVFMM_INCLUDE_DIR, PVFMM_LIBRARIES

find_path(PVFMM_INCLUDE_DIR
  NAMES pvfmm.hpp
  PATHS
    $ENV{HOME}/local/include/pvfmm
    $ENV{CONDA_PREFIX}/include/pvfmm
    $ENV{PVFMM_PREFIX}/include/pvfmm
    $ENV{CPATH}/pvfmm
    $ENV{C_INCLUDE_PATH}/pvfmm
    $ENV{CPLUS_INCLUDE_PATH}/pvfmm
    $ENV{OBJC_INCLUDE_PATH}/pvfmm
    $ENV{OBJCPLUS_INCLUDE_PATH}/pvfmm
    /usr/include/pvfmm
    /usr/local/include/pvfmm
    /opt/local/include/pvfmm
    /sw/include/pvfmm
  DOC "Include Directory for PVFMM"
)

find_library(PVFMM_LIBRARIES
  NAMES pvfmm
  PATHS
    $ENV{HOME}/local/lib/pvfmm
    $ENV{PVFMM_PREFIX}/lib/pvfmm
    $ENV{CONDA_PREFIX}/lib/pvfmm
    $ENV{LIBRARY_PATH}/pvfmm
    $ENV{LD_LIBRARY_PATH}/pvfmm
    /usr/lib/pvfmm
    /usr/local/lib/pvfmm
    /opt/local/lib/pvfmm
    /sw/lib/pvfmm
  DOC "PVFMM library"
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PVFMM DEFAULT_MSG PVFMM_LIBRARIES PVFMM_INCLUDE_DIR)

# mark_as_advanced(PVFMM_INCLUDE_DIR PVFMM_LIBRARIES)

# Interface target
# We refrain from creating an imported target since those cannot be exported

# this may cause wrong order of linked libraries in link line
# add_library(pvfmm INTERFACE)
# target_link_libraries(pvfmm INTERFACE ${PVFMM_LIBRARIES})
# target_include_directories(pvfmm INTERFACE ${PVFMM_INCLUDE_DIR})
