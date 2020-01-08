#! /bin/bash

export USER_LOCAL=$HOME/local
export SYSTEM_LOCAL=/usr/local

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D Eigen3_DIR="${USER_LOCAL}/share/eigen3/cmake" \
  -D CMAKE_INSTALL_PREFIX=${USER_LOCAL} \
  -D PyInterface=ON \
  -D BUILD_DOC=OFF \
../

