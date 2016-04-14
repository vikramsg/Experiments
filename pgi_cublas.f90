program cuBLAS_example
  use iso_c_binding, only: c_double, c_int

  use cudafor
  use cublas

  implicit none

  integer, parameter :: NCOLUMNS = 3 
  integer, parameter :: NROW = 3, NCOL=3
  
  real(c_double), allocatable :: A(:,:), B(:, :), C(:, :)
  real(c_double) :: Ctmp(NROW, NCOL)
  integer           :: size_of_real, i, j, k


  real(c_double) :: t1, t2

!  integer*8 :: devPtrA, devPtrB, devPtrC

!  size_of_real = 16

  allocate(A(NROW, NCOL))
  allocate(B(NROW, NCOL*NCOLUMNS))
  allocate(C(NROW, NCOL*NCOLUMNS))

  !$acc data create(A, B, C)
  !$acc kernels
  A(:, :) = 0.0 
  B(:, :) = 2.0_c_double
  A(1, 1) = 3.0
  A(2, 2) = 2.0
  A(3, 3) = 1.0
!
  C(:, :)      = -0.5_c_double
  !$acc end kernels

  !$acc host_data use_device(A, B, C)
  call cublasDGEMM('N', 'N', NROW, NCOL*NCOLUMNS, NROW, &
       1.0_c_double, A, NROW, B, NROW, 0.0_c_double, C, NROW)
  !$acc end host_data

  !$acc update self(A, B, C)
  !$acc end data

!  print*, A, B, C

  deallocate(A)
  deallocate(B)
  deallocate(C)

!  print*, 'Time = ', NCOLUMNS, t2-t1

end program cuBLAS_example
