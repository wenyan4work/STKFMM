#! /bin/bash

export SFTPATH=$HOME/local
export SYSTEM_LOCAL=/usr/local

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D CMAKE_BUILD_TYPE=Release \
  -D Eigen3_DIR="${SFTPATH}/share/eigen3/cmake" \
  -D CMAKE_INSTALL_PREFIX=${SFTPATH} \
  -D BUILD_TEST=ON \
  -D BUILD_DOC=OFF \
  -D BUILD_M2L=OFF \
  -D PyInterface=OFF \
../

