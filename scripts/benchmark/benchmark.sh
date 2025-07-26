#!/bin/bash

# build the project for the different configurations
# 1. CPU only
# 2. Eigen with CPU
# 3. OpenACC

build_dir="build"
code_dir="../../poc-gpu-likelihood-calculation"
if [ ! -d "$build_dir" ]; then
    mkdir "$build_dir"
fi

cd "$build_dir"

mkdir -p cpu
mkdir -p eigen
mkdir -p openacc

# Build CPU only
cd cpu
cmake $code_dir
make -j
cd ..

# Build Eigen with CPU
cd eigen
cmake -DUSE_EIGEN=ON -DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3 $code_dir
make -j
cd ..

# Build OpenACC
cd openacc
module load nvhpc-compilers/24.7

export OMPI_CC=nvc
export OMPI_CXX=nvc++

export CC=nvc
export CXX=nvc++
export CUDACXX=nvcc

export LDFLAGS="-L/apps/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/compilers/lib"
export CPPFLAGS="-I/apps/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/compilers/include"

cmake -DCMAKE_CXX_FLAGS="$LDFLAGS $CPPFLAGS" -DUSE_OPENACC=ON $code_dir
make -j
cd ..

###############################################