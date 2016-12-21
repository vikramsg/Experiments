  !> Main program to create grid and calculate error for derivative
  program template
  
    use subroutines
  
    implicit none  

    integer(c_int) :: nele_x ! Number of elements in x
    integer(c_int) :: order  ! Polynomial order 
    real(c_double) :: startX, stopX 

    real(c_double) :: nu !CFL 

    real(c_double) :: stopT 

    ! 1D domain 
    startX = 0; stopX = one 

    ! CFL value
    nu  = two/ten 

    !Stop Time
    stopT = half 

    ! Echo print your input to make sure it is correct
    write(*,*) 'Your 1D domain is from ', startX, ' to ', stopX

    order = 2
    do nele_x = 10, 10
!        call validate_derivative(nele_x, startX, stopX, order)
!        call wave_solver(nele_x, startX, stopX, stopT, order, nu)
        call diff_solver(nele_x, startX, stopX, stopT, order, nu)
    end do

  end program template
