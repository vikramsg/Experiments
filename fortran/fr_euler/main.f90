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
    startX = -one; stopX = one 

    ! CFL value
    nu  = two/ten !+ one 

    !Stop Time
    stopT = one 

    ! Echo print your input to make sure it is correct
    write(*,*) 'Your 1D domain is from ', startX, ' to ', stopX

    order = 2
    do nele_x = 4, 4
        call wave_solver(nele_x, startX, stopX, stopT, order, nu)
    end do

  end program template
