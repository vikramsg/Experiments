program cuBLAS_example

  use iso_c_binding, only: c_double, c_int
  use omp_lib

#ifdef _OPENACC
  use cublas
  use openacc
#endif

  implicit none

  integer, parameter :: nelems = 500000
  integer, parameter :: Np = 16, Nvar = 4, Nflux = 16
  integer, parameter :: chunkSize = 50

  type mesh2d
      real(c_double) :: u(Np, Nvar, nelems)
      real(c_double) :: ux(Np, Nvar, nelems)
      real(c_double) :: uflux(Nflux, Nvar, nelems)
      real(c_double) :: ucommon(Nflux, Nvar, nelems)
  end type mesh2d
  
  real(c_double)   :: Lgrad(Np, Np), phi(Np, Nflux) 
  real(c_double)   :: uelem_tmp(Np, Nvar*chunkSize), ucorr_tmp(Nflux, Nvar*chunkSize) 
  real(c_double)   :: ugrad_tmp(Np, Nvar*chunkSize)

  type(mesh2d)     :: mesh

  integer(c_int)   :: i, j, k, iter, rep
  integer(c_int)   :: nthreads 

  real(c_double)   :: t1, t2

  call CPU_TIME(t1)

  !$OMP PARALLEL
  nthreads = omp_get_num_threads()
  !$OMP END PARALLEL

  !$OMP  PARALLEL DO num_threads(nthreads - 1) 
  do rep = 1, 500
   do i = 1, nelems
      do j = 1, Nvar
          do k = 1, Np
              mesh%u(k, j, i) = i+j+k
          end do
          do k = 1, Nflux
              mesh%uflux(k, j, i) = i-j-k
              mesh%ucommon(k, j, i) = i-j+k
          end do
      end do
   end do

   do j = 1, Np
    do k = 1, Np
        Lgrad(j, k) = k*j                  
    end do
    do k = 1, Nflux
        phi(j, k) = k*j + k + j
    end do
   end do
  end do
  !$OMP END PARALLEL DO  

  write(*, *) phi(2, 4)
!
!  call acc_test()

end program cuBLAS_example


subroutine acc_test
  use iso_c_binding, only: c_double, c_int

#ifdef _OPENACC
  use cublas
  use openacc
#endif

  implicit none

  integer, parameter :: nelems = 500000
  integer, parameter :: Np = 16, Nvar = 4, Nflux = 16
  integer, parameter :: chunkSize = 50

  type mesh2d
      real(c_double) :: u(Np, Nvar, nelems)
      real(c_double) :: ux(Np, Nvar, nelems)
      real(c_double) :: uflux(Nflux, Nvar, nelems)
      real(c_double) :: ucommon(Nflux, Nvar, nelems)
  end type mesh2d
  
  real(c_double)   :: Lgrad(Np, Np), phi(Np, Nflux) 
  real(c_double)   :: uelem_tmp(Np, Nvar*chunkSize), ucorr_tmp(Nflux, Nvar*chunkSize) 
  real(c_double)   :: ugrad_tmp(Np, Nvar*chunkSize)

  type(mesh2d)     :: mesh

  integer(c_int)   :: i, j, k, iter, rep

  real(c_double)   :: t1, t2


  !$acc parallel loop
  do rep = 1, 500
  do i = 1, nelems
      do j = 1, Nvar
          do k = 1, Np
              mesh%u(k, j, i) = i+j+k
          end do
          do k = 1, Nflux
              mesh%uflux(k, j, i) = i-j-k
              mesh%ucommon(k, j, i) = i-j+k
          end do
      end do
  end do

  do j = 1, Np
    do k = 1, Np
        Lgrad(j, k) = k*j                  
    end do
    do k = 1, Nflux
        phi(j, k) = k*j + k + j
    end do
  end do
  end do
  !$acc end parallel

  write(*, *) phi(2, 4)

end subroutine acc_test
