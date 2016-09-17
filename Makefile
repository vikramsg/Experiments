EXE = acc.x

FC = pgfortran

LDFLAGS = -Mcudalib:cublas -Mcuda 
FFFLAGS = -acc -Minfo=all -ta=tesla:cc30 

MCLDFLAGS = -L/home/vikram/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm   -ldl 
MCFFFLAGS = -acc -Minfo=accel -ta=multicore

BLASLDFLAGS = -lblas
BLASFFFLAGS = 

INTELFC = ifort
INTELFFLAGS = -coarray
INTELMKLFLAGS = -L/home/vikram/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm   -ldl

all: $(EXE)

clean: 
	rm *.x *.o *.mod

omp: omp.f90 
	$(INTELFC) -o $@  $^  $(INTELMKLFLAGS)

coarray: coarray.F90
	$(INTELFC) $(INTELFFLAGS) $^ -o $@  

mc_speedup.x: speedup.F90 
	$(FC) -o $@ $(MCFFFLAGS) $^  $(MCLDFLAGS)

intel_speedup.x: speedup.F90 
	$(FC) -o $@  $^  $(INTELMKLFLAGS)

blas_speedup.x: speedup.F90 
	$(FC) -o $@ $(BLASFFFLAGS) $^  $(BLASLDFLAGS)
	
speed.x: speedup.F90 
	$(FC) -o $@ $(FFFLAGS) $^  $(LDFLAGS)
	
rout.x: demo.o routine.F90 
	$(FC) -o $@ $(FFFLAGS) $^  $(LDFLAGS)

demo.o: demo.F90
	$(FC) -c -o $@ $(FFFLAGS) $^  $(LDFLAGS)

