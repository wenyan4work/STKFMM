#! /bin/bash

# change this to find Eigen3Config.cmake in this folder
export EIGENPATH=$HOME/env_intel/share/eigen3/cmake/  

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_BUILD_TYPE=Release \
  -D Eigen3_DIR=${EIGENPATH} \
  -D BUILD_TEST=ON \
  -D BUILD_DOC=OFF \
  -D BUILD_M2L=OFF \
  -D PyInterface=OFF \
../

