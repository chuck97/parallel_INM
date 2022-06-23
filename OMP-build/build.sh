module load intel
cmake -DUSE_OMP=ON -DCMAKE_C_COMPILER=icc ..
make all