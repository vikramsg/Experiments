module test

 use iso_c_binding, only: sp => C_FLOAT, dp => C_DOUBLE, i8 => C_INT
 
 implicit none
 
 contains
 
  subroutine add (a, b, n, c)
      integer(kind=i8), intent(in)  :: n
      real(kind=dp), intent(in)  :: a(n)
      real(kind=dp), intent(in)  :: b(n)
      real(kind=dp), intent(out) :: c(n) 

      integer(kind=i8)  :: i

      do i = 1, n  
          c(i) = a(i) + b(i)
      end do

  end subroutine add

  subroutine mult (a, b, c)
      real(kind=dp), intent(in)  :: a
      real(kind=dp), intent(in)  :: b 
      real(kind=dp), intent(out) :: c 
  
      c = a * b
   
  end subroutine mult 
 
end module test
