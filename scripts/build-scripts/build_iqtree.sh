module load openmpi/4.1.5 boost/1.84.0 eigen/3.3.7 llvm/17.0.1

export OMPI_CC=clang
export OMPI_CXX=clang++


export CC=clang
export CXX=clang++


export LDFLAGS="-L/apps/llvm/17.0.1/lib"
export CPPFLAGS="-I/apps/llvm/17.0.1/lib/clang/17/include"


############


cmake -DCMAKE_CXX_FLAGS="$LDFLAGS $CPPFLAGS" -DEIGEN3_INCLUDE_DIR=/apps/eigen/3.3.7/include/eigen3 -DUSE_CMAPLE=OFF $code_dir
make -j