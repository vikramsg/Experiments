EXE = acc.x

FC = pgfortran

LDFLAGS = -L/usr/local/cuda/lib64 -lcublas -Mcuda

FFFLAGS = -acc -Minfo=accel

all: $(EXE)

acc.x: pgi_cublas.f90
	$(FC) -o $@ $(FFFLAGS) $^  $(LDFLAGS)
