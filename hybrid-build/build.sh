module load intel
cmake -DUSE_MPI=ON -DUSE_OMP=ON -DCMAKE_C_COMPILER=mpiicc ..
make all