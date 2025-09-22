
cmake -DCMAKE_CXX_FLAGS="$LDFLAGS $CPPFLAGS" -DUSE_EIGEN=ON  -DEIGEN3_INCLUDE_DIR=/scratch/dx61/sa0557/iqtree2/eigen   ..

make -j
