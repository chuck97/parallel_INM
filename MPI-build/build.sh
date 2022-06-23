module load intel
cmake -DUSE_MPI=ON -DCMAKE_C_COMPILER=mpiicc ..
make all