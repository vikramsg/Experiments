EXE = test.x

FC = gfortran

LDFLAGS =

all = $(EXE)

test.x: types.o polynomial.o fr.o modules.o main.o
	$(FC) -o $@ $^  $(LDFLAGS)
	
fr.o: fr.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)

polynomial.o: polynomial.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)

modules.o: modules.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)

types.o: types.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)

main.o: main.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
clean: 
	rm *.x *.o

