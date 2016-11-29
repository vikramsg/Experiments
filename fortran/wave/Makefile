EXE = test.x

FC = gfortran

LDFLAGS =

all = $(EXE)

test.x: modules.o main.o
	$(FC) -o $@ $^  $(LDFLAGS)
	
modules.o: modules.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
main.o: main.f90
	$(FC) -o $@ -c $^  $(LDFLAGS)
	
clean: 
	rm *.x *.o

