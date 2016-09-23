    program test
    
      use iso_c_binding
    
      implicit none
    
      integer(c_int), parameter :: N = 1000
      integer(c_int) :: i, j
      real(c_double) :: x(N, N), y(N, N), z(N, N)
      character(kind=c_char)     :: flag 
      
      interface
         subroutine cublasdgemm(transa, transb, m, n, k, alpha, A, lda, B, &
                 ldb, beta, C, ldc) bind(c, name="cublasDgemm")
           use iso_c_binding
           character(kind=c_char), value     :: transa, transb 
           integer(kind=c_int), value :: m, n, k 
           real(c_double), value      :: alpha
           type(*), dimension(*)      :: A
           integer(kind=c_int), value :: lda 
           type(*), dimension(*)      :: B
           integer(kind=c_int), value :: ldb 
           real(c_double), value      :: beta
           type(*), dimension(*)      :: C
           integer(kind=c_int), value :: ldc 
    
         end subroutine cublasdgemm
    
      end interface

      flag = 'n'

      !$acc enter data create (x, y, z) 

      !$acc parallel 
      !$acc loop gang vector collapse(2)
      do i = 1, N
         do j = 1, N
           x(i, j) = 4.0 * i
           y(i, j) = 3.0 + j
           z(i, j) = 0.0 
         end do
      end do
      !$acc end parallel 
    
    
      !$acc host_data use_device (x, y, z)
      call cublasdgemm('n', 'n', n, n, n, 1.0_c_double, x, n, y, n, 0.0_c_double, z, n)
      !$acc end host_data

      !$acc update self(z)
    
      !$acc exit data delete(x, y, z)

      write(*, *) z(1:2, 1:2)
    
!     call dgemm(flag, flag, n, n, n, 1.0_c_double, x, n, y, n, 0.0_c_double, z, n)
!    
!      write(*, *) z(1:2, 1:2)
    
    end program test


