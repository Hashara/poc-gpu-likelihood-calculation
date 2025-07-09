module load nvhpc-compilers/24.7


export OMPI_CC=nvc
export OMPI_CXX=nvc++

export CC=nvc
export CXX=nvc++
export CUDACXX=nvcc

export LDFLAGS="-L/apps/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/compilers/lib"
export CPPFLAGS="-I/apps/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/compilers/include"

cmake -DCMAKE_CXX_FLAGS="$LDFLAGS $CPPFLAGS" -DUSE_OPENACC=ON  ..

make -j
