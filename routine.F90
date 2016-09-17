program example

  use iso_c_binding, only: c_double, c_int
  use demo_rout

#ifdef _OPENACC
  use openacc
  use cublas
#endif

  implicit none

  integer, parameter :: nelems = 500000
  integer, parameter :: Np = 16, Nvar = 4, Nflux = 16
  integer, parameter :: chunkSize = 50

  type mesh2d
      real(c_double) :: uflux(Nflux, Nvar, nelems)
      real(c_double) :: ucommon(Nflux, Nvar, nelems)
  end type mesh2d
  
  real(c_double)   :: ucorr_tmp(Nflux, Nvar*chunkSize) 

  type(mesh2d)     :: mesh

  integer(c_int)   :: i, j, k, iter, rep

  real(c_double)   :: t1, t2

  call CPU_TIME(t1)

  do i = 1, nelems
      do j = 1, Nvar
          do k = 1, Nflux
              mesh%uflux(k, j, i) = i-j-k
              mesh%ucommon(k, j, i) = i-j+k
          end do
      end do
  end do

  !$acc enter data copyin(mesh)
  !$acc enter data create(ucorr_tmp)

  do iter = 1, nelems/chunkSize
  
       !$acc parallel loop independent collapse(3)      &
       !$acc present(mesh, mesh%uflux, mesh%ucommon, ucorr_tmp)
       do i = 1, chunkSize 
           do j = 1, Nvar
               do k = 1, Nflux
                   call demo(mesh%uflux(k, j, (iter-1)*chunkSize + i), mesh%ucommon(k, j, (iter-1)*chunkSize + i) &
                                                    , ucorr_tmp(k, Nvar*(i-1) + j) ) 
               end do
           end do
       end do
       !$acc end parallel

   end do


   !$acc exit data delete(ucorr_tmp)
   !$acc exit data delete(mesh)

   call CPU_TIME(t2)

   print*, 'Time taken', t2-t1


end program example

