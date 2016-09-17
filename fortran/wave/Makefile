EXE = wave.x

FC = gfortran

LDFLAGS =

all = $(EXE)

wave.x: wave.o main.o
	$(FC) -o $@ $^  $(LDFLAGS)
	
wave.o: wave.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
main.o: main.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
clean: 
	rm *.x *.o

