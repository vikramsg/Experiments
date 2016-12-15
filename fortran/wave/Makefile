EXE = test.x

FC = gfortran

LDFLAGS =

all = $(EXE)

test.x: polynomial.o modules.o main.o
	$(FC) -o $@ $^  $(LDFLAGS)
	
polynomial.o: polynomial.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)

modules.o: modules.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
main.o: main.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
clean: 
	rm *.x *.o

