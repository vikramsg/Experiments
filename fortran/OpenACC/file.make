./rungcc6.sh gcc -Wall -g -I/usr/local/cuda/include -I/usr/local/cuda/src -DCUBLAS_GFORTRAN -c fortran.c

./rungcc6.sh gfortran -Wall -g test.f90 fortran.o -fopenacc -foffload=nvptx-none -foffload=-O3 -O3 -o gpu.x -L/usr/local/cuda/lib64 -lcublas -lcudart
