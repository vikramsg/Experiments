EXE = shock.x 

CPP = g++ -O0 -g 

LDFLAGS =

all = $(EXE)

shock.x: weno.o
	$(CPP) -o $@ $^  $(LDFLAGS)
	
weno.o: weno.cpp
	$(CPP) -o $@ -c $^  $(LDFLAGS)
	
clean: 
	rm -rf *.x *.o *.dSYM

