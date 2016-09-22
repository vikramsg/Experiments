program derived_type

  use openacc

  use iso_c_binding, only: c_double, c_int

  implicit none

  integer, parameter :: nelems = 500000
  integer, parameter :: Np = 16, Nvar = 4, Nflux = 16

  type mesh2d
      real(c_double) :: u(Np, Nvar, nelems)
  end type mesh2d
  
  type(mesh2d)     :: mesh

  integer(c_int)   :: i, j, k

  !$acc enter data create(mesh)

  !$acc parallel present(mesh)               &
  !$acc num_gangs(40) num_workers(4)         &
  !$acc vector_length(32) 
  !$acc loop gang worker vector collapse(3) 
  do i = 1, nelems
      do j = 1, Nvar
          do k = 1, Np
              mesh%u(k, j, i) = i+j+k
          end do
      end do
  end do
  !$acc end parallel

  !$acc exit data delete(mesh)

  !$acc enter data create(mesh)

  !$acc parallel present(mesh)               
  !$acc loop gang worker vector collapse(3) 
  do i = 1, nelems
      do j = 1, Nvar
          do k = 1, Np
              mesh%u(k, j, i) = i+j+k
          end do
      end do
  end do
  !$acc end parallel

  !$acc update self(mesh)

  !$acc exit data delete(mesh)

  write(*, *) mesh%u(2, 4, 12)


end program derived_type 
