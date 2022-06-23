cd build
find . -type f ! -name '*.sh' -delete
bash build.sh
cd ..

cd MPI-build 
find . -type f ! -name '*.sh' -delete
bash build.sh
cd ..

cd OMP-build 
find . -type f ! -name '*.sh' -delete
bash build.sh
cd ..

cd hybrid-build
find . -type f ! -name '*.sh' -delete
bash build.sh
cd ..