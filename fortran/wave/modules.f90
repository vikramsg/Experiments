  
  !> Module to defined data types and some constants
  MODULE types_vars
      use iso_c_binding
    ! Symbolic names for kind types of single- and double-precison reals
    INTEGER, PARAMETER :: SP = KIND(1.0_c_float)
    INTEGER, PARAMETER :: DP = KIND(1.0_c_double)

    ! Frequently used mathematical constants (with precision to spare)
    REAL(DP), PARAMETER :: zero  = 0.0_c_double
    REAL(DP), PARAMETER :: half  = 0.5_c_double  
    REAL(DP), PARAMETER :: one   = 1.0_c_double
    REAL(DP), PARAMETER :: two   = 2.0_c_double
    REAL(DP), PARAMETER :: three = 3.0_c_double
    REAL(DP), PARAMETER :: four  = 4.0_c_double
    REAL(DP), PARAMETER :: pi    = 3.141592653589793238462643383279502884197_c_double
    REAL(DP), PARAMETER :: pio2  = 1.57079632679489661923132169163975144209858_c_double
    REAL(DP), PARAMETER :: twopi = 6.283185307179586476925286766559005768394_c_double
  END MODULE types_vars

  module polynomial

      use types_vars

      contains

          !> Subroutine that returns Gauss Nodes
          !! @param order: order of polynomial
          !! @param x: nodes in [-1,1] cell
          subroutine gauss_nodes(order, x)

              integer(c_int), intent(in)  :: order
              real(c_double), intent(out) :: x(:) 

              integer(c_int) :: Np  ! Number of points in 1D

              Np = order + 1

              if (Np .eq. 2) then
                  x(1) = -0.577350
                  x(2) =  0.577350
              else if (Np .eq. 3) then
                  x(1) = -0.774597
                  x(2) =  0
                  x(3) =  0.774597
              else if (Np .eq. 4) then
                  x(1) = -0.861136
                  x(2) = -0.339981
                  x(3) =  0.339981
                  x(4) =  0.861136
              end if

          end subroutine gauss_nodes

          !> Subroutine that wraps node generation
          !! @param order: order of polynomial
          !! @param x: nodes in [-1,1] cell
          subroutine cell_coordi(order, x)
              integer(c_int), intent(in)     :: order
              real(c_double), intent(inout)  :: x(:) 

              call gauss_nodes(order, x)

          end subroutine cell_coordi

  end module polynomial


  MODULE subroutines
    USE types_vars

    CONTAINS

    !> Subroutine to make simple 1D grid
    !! @param nptsx num. of points
    !! @param startX, stopX starting and ending location
    !! @param x array containing grid points
    subroutine make_grid(nele_x, startX, stopX, Np, x_nodes, x)
        implicit none

        integer(c_int), intent(in)  :: nele_x, Np
        real(c_double), intent(in)  :: startX, stopX 
        real(c_double), intent(in)  :: x_nodes(Np) 

        real(c_double), intent(out) :: x(:, :)

        integer(c_int) :: i 
        real(c_double) :: dx, temp_x(nele_x + 1) 

        ! Grid spacing
        dx = (stopX - startX)/FLOAT(nele_x)

        ! Generate grid
        do i = 1, nele_x + 1 
            temp_x(i) = startX + (i-1)*dx
        end do

        write(*, *) temp_x

    end subroutine make_grid


    !> Subroutine to validate 2nd and 4th order derivatives
    !! @param nptsx number of points
    !! @param startX, stopX starting and ending location of grid
    !! @param max2nd max4th maximum error for 2nd and 4th order derivative
    subroutine validate_derivative(nele_x, startX, stopX, order)
        
        use polynomial

        implicit none

        integer(c_int), intent(in)  :: nele_x, order
        real(c_double), intent(in)  :: startX, stopX 

        real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
        real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 

        integer(c_int) :: Np 

        Np = order + 1

        call cell_coordi(order, x_nodes)

        call make_grid(nele_x, startX, stopX, Np, x_nodes, x)

    end subroutine validate_derivative


  END MODULE subroutines


