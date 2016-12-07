  
  !> Module to defined data types and some constants
  module types_vars

    use iso_c_binding
    implicit none

    ! Symbolic names for kind types of single- and double-precison reals
    INTEGER, PARAMETER :: SP = KIND(1.0_c_float)
    INTEGER, PARAMETER :: DP = KIND(1.0_c_double)

    ! Frequently used mathematical constants (with precision to spare)
    REAL(DP), PARAMETER :: zero    = 0.0_c_double
    REAL(DP), PARAMETER :: one     = 1.0_c_double
    REAL(DP), PARAMETER :: two     = 2.0_c_double
    REAL(DP), PARAMETER :: three   = 3.0_c_double
    REAL(DP), PARAMETER :: four    = 4.0_c_double
    REAL(DP), PARAMETER :: five    = 5.0_c_double
    REAL(DP), PARAMETER :: six     = 6.0_c_double
    REAL(DP), PARAMETER :: seven   = 7.0_c_double
    REAL(DP), PARAMETER :: eight   = 8.0_c_double
    REAL(DP), PARAMETER :: ten     = 10.0_c_double
    REAL(DP), PARAMETER :: hundred = 100.0_c_double

    REAL(DP), PARAMETER :: half    = 0.5_c_double  
    REAL(DP), PARAMETER :: fourth  = 0.25_c_double  
    REAL(DP), PARAMETER :: five3rd = 1.66666666666666666666667_c_double  

    REAL(DP), PARAMETER :: pi    = 3.141592653589793238462643383279502884197_c_double
    REAL(DP), PARAMETER :: pio2  = 1.57079632679489661923132169163975144209858_c_double
    REAL(DP), PARAMETER :: twopi = 6.283185307179586476925286766559005768394_c_double

  end module types_vars

  module polynomial

      use types_vars

      contains

          !> Suborutine to evaluate legendre basis derivative at a point 
          !! @param mode: Particular mode whose interpolation is required
          !! @param r: location where evaluation is required
          !! @param f: interpolated value
          recursive subroutine get_legendre_d(r, mode, f)
              integer(c_int), intent(in)  :: mode 
              real(c_double), intent(in)  :: r

              real(c_double), intent(out) :: f 

              real(c_double) :: p1, p2
              integer(c_int) :: j 

              if (mode .eq. 1) then
                  f = zero
              else
                  if ((r .gt. -one) .and. (r .lt. one)) then
                      call get_legendre(r, mode    , p1)
                      call get_legendre(r, mode - 1, p2)
                      f = ((mode - 1)*(r*p1 - p2))/(r**2 - one)
                  else if (abs(r - (-one)) .lt. epsilon(one)) then
                      f = (-one)**(mode - 2)*half*(mode - 1)*(mode)
                  else if (abs(r - one) .lt. epsilon(one)) then
                      f = half*(mode - 1)*mode
                  end if
              end if

          end subroutine get_legendre_d


          !> Suborutine to evaluate legendre basis at a point 
          !! @param mode: Order or mode whose interpolation is required
          !! @param r: location where evaluation is required
          !! @param f: interpolated value
          subroutine get_legendre(r, mode, f)
              integer(c_int), intent(in)  :: mode 
              real(c_double), intent(in)  :: r

              real(c_double), intent(out) :: f 

              real(c_double) :: p1, p2
              integer(c_int) :: j 

              p1 = one 
              p2 = r

              if (mode .eq. 1) then
                  f = p1
              else if (mode .eq. 2) then
                  f = p2
              else
                  do j = 3, mode 
                      f = ((two*(j - two) + one)*r*p2 - (j - two)*p1)/((j-two) + one)
                      p1 = p2
                      p2 = f
                  end do
              end if

          end subroutine get_legendre


          !> Suborutine to evaluate lagrange interpolant
          !! @param n_pts: number of points of input array(nodes)
          !! @param mode: Particular mode whose interpolation is required
          !! @param r: location where interpolation is required
          !! @param nodes: vector of nodes 
          !! @param f: interpolation co-efficient at r 
          subroutine get_lagrange(r, mode, nodes, n_pts, f)
              integer(c_int), intent(in)  :: n_pts, mode 
              real(c_double), intent(in)  :: r, nodes(:) 

              real(c_double), intent(out) :: f 

              integer(c_int) :: j 

              f = one

              do j = 1, n_pts
                  if (j .ne. mode) then
                      f = f*(r - nodes(j))/(nodes(mode) - nodes(j))
                  end if
              end do

          end subroutine get_lagrange


          !> Suborutine to evaluate lagrange interpolant derivative
          !! @param n_pts: number of points of input array(nodes)
          !! @param mode: Particular mode whose interpolation is required
          !! @param r: location where interpolation is required
          !! @param nodes: vector of nodes 
          !! @param f: interpolated co-efficient at r for derivative 
          subroutine get_lagrange_d(r, mode, nodes, n_pts, f)
              integer(c_int), intent(in)  :: n_pts, mode 
              real(c_double), intent(in)  :: r, nodes(:) 

              real(c_double), intent(out) :: f 

              real(c_double) :: temp1, temp2 
              integer(c_int) :: i, j 

              f = zero 

              do i = 1, n_pts
                  outer: if (i .ne. mode) then
                      temp1 = one
                      temp2 = one

                      do j = 1, n_pts
                          if ( (j .ne. mode) .and. (j .ne. i)) then
                              temp1 = temp1*(r - nodes(j))
                          end if

                          if ( j .ne. mode) then
                              temp2 = temp2*(nodes(mode) - nodes(j))
                          end if
                      end do

                      f = f + temp1/temp2 

                  end if outer
              end do

          end subroutine get_lagrange_d


          !> Subroutine that tests polynomials 
          subroutine test_poly()
              real(c_double)    :: x(4) 
              integer(c_int)    :: order, i

              real(c_double)    :: f, r, temp 

              order = 3
              call gauss_nodes(order, x)

              do i = 1, order + 1
                  call get_lagrange(x(i), i, x, order + 1, f)
                  if (abs(f - one) .gt. epsilon(one)) then
                      write(*, *) "Error: Lagrange basis is incorrect"
                      stop
                  end if
                  call get_lagrange_d(x(i), i, x, order + 1, f)
              end do

              call get_legendre(fourth, 3, f)
              if (abs(f - (-0.40625_c_double)) .gt. epsilon(one)) then
                  write(*, *) "Error: Legendre basis is incorrect"
                  stop
              end if

              do i = 1, 7
                  r = -one + i*fourth
                  call get_legendre_d(r, 4, f)
                  temp = half*(five*three*r*r - three)
                  if (abs(f - temp) .gt. epsilon(one)) then
                      write(*, *) "Error: Legendre derivative is incorrect"
                      stop
                  end if
              end do
 
          end subroutine test_poly


          !> Subroutine that returns Gauss Nodes
          !! @param order: order of polynomial
          !! @param x: nodes in [-1,1] cell
          subroutine gauss_nodes(order, x)

              integer(c_int), intent(in)  :: order
              real(c_double), intent(out) :: x(:) 

              integer(c_int) :: Np  ! Number of points in 1D

              Np = order + 1

              if (Np .eq. 2) then
                  x(1) = -one/sqrt(three)
                  x(2) =  one/sqrt(three) 
              else if (Np .eq. 3) then
                  x(1) = -one/sqrt(five3rd)
                  x(2) =  zero 
                  x(3) =  one/sqrt(five3rd)
              else if (Np .eq. 4) then
                  x(1) = -sqrt(three/seven + two/seven*sqrt(six/five))
                  x(2) = -sqrt(three/seven - two/seven*sqrt(six/five))
                  x(3) =  sqrt(three/seven - two/seven*sqrt(six/five))
                  x(4) =  sqrt(three/seven + two/seven*sqrt(six/five))
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


  module subroutines
    use types_vars
    implicit none

    contains

        !> Subroutine to initialize solution on the FR grid
        !! @param nele_x: Number of elements in grid
        !! @param Np: Number of points in each cell
        !! @param x, u: location of FR grid points and soln vec
        subroutine init_sol(nele_x, Np, x, u)
            integer(c_int), intent(in)  :: nele_x, Np
            real(c_double), intent(out) :: x(:, :), u(:, :)

            integer(c_int) :: i, j 

            do i = 1, nele_x
                do j = 1, Np
                    u(j, i) = sin(two*pi*x(j, i)/40*one)
                end do
            end do
 
        end subroutine init_sol

        !> Subroutine to make simple 1D grid
        !! @param nptsx num. of points
        !! @param startX, stopX starting and ending location
        !! @param x array containing grid points
        subroutine make_grid(nele_x, startX, stopX, Np, x_nodes, x)
            integer(c_int), intent(in)  :: nele_x, Np
            real(c_double), intent(in)  :: startX, stopX 
            real(c_double), intent(in)  :: x_nodes(Np) 
    
            real(c_double), intent(out) :: x(:, :)
    
            integer(c_int) :: i, j 
            real(c_double) :: dx, temp_x(nele_x + 1) 
    
            ! Grid spacing
            dx = (stopX - startX)/FLOAT(nele_x)
    
            ! Generate grid
            do i = 1, nele_x + 1 
                temp_x(i) = startX + (i-1)*dx
            end do
    
            ! Generate grid
            do i = 1, nele_x
                do j = 1, Np
                    x(j, i) = half*(1 - x_nodes(j))*temp_x(i)    &
                              + half*(1 + x_nodes(j))*temp_x(i+1) 
                end do
            end do

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
    
            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: u(order + 1, nele_x) ! x co-ordinates
    
            integer(c_int) :: Np 
    
            call test_poly()

            Np = order + 1

            call cell_coordi(order, x_nodes)
    
            call make_grid(nele_x, startX, stopX, Np, x_nodes, x)
    
            call init_sol(nele_x, Np, x, u)
    
!            write(*, *) x
    
        end subroutine validate_derivative



  end module subroutines


