    program test
    
      use iso_c_binding
    
      implicit none
    
      integer(c_int), parameter :: N = 1000
      integer(c_int) :: i, j
      real(c_double) :: x(N, N), y(N, N), z(N, N)

      !$acc enter data create (x, y, z) 

      !$acc parallel 
      !$acc loop gang vector collapse(2)
      do i = 1, N
         do j = 1, N
           x(i, j) = 4.0 * i
           y(i, j) = 3.0 + j
           z(i, j) = x(i, j) + y(i, j) 
         end do
      end do
      !$acc end parallel 
    
      !$acc update self(z)
    
      !$acc exit data delete(x, y, z)

      write(*, *) z(1:2, 1:2)
    
    end program test


