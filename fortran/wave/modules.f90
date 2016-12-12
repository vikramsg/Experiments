  
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
      implicit none

      contains

          !> Suborutine to evaluate legendre basis derivative at a point 
          !! @param mode: Order or mode whose derivative is required
          !! @param r: location where evaluation is required
          !! @param f: interpolated value
          recursive subroutine get_legendre_d(r, mode, f)
              integer(c_int), intent(in)  :: mode 
              real(c_double), intent(in)  :: r

              real(c_double), intent(out) :: f 

              real(c_double) :: p1, p2
              integer(c_int) :: j 

              if ((r .gt. -one) .and. (r .lt. one)) then
                  call get_legendre(r, mode      , p1)
                  call get_legendre(r, mode - 1  , p2)
                  f = ((mode)*(r*p1 - p2))/(r**two - one)
              else if (abs(r - (-one)) .lt. epsilon(one)) then
                  f = (-one)**(mode - one)*half*(mode)*(mode + one)
              else if (abs(r - one) .lt. epsilon(one)) then
                  f = half*(mode)*(mode + one)
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

              real(c_double) :: p0, p1
              integer(c_int) :: j 

              p0 = one 
              p1 = r

              if      (mode .eq. 0) then
                  f = p0
              else if (mode .eq. 1) then
                  f = p1
              else
                  do j = 2, mode 
                      f = ((two*(j - one) + one)*r*p1 - (j - one)*p0)/((j-one) + one)
                      p0 = p1
                      p1 = f
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
          !! @param mode: Particular mode whose derivative is required
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

              call get_legendre(fourth, 2, f)
              if (abs(f - (-0.40625_c_double)) .gt. epsilon(one)) then
                  write(*, *) "Error: Legendre basis is incorrect"
                  stop
              end if

              do i = 1, 7
                  r = -one + i*fourth
                  call get_legendre_d(r, 3, f)
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

          !> Subroutine to create Lagrange vector 
          !! to interpolate to flux points
          !! @param npts: Number of Lagrange points 
          !! @param nodes: location of points
          !! @param lag_flux_l: interpolation vector at left flux point
          !! @param lag_flux_r: interpolation vector at right flux point
          subroutine lagr_flux_matrix(npts, nodes, lag_flux_l, lag_flux_r)
              integer(c_int), intent(in)     :: npts
              real(c_double), intent(in)     :: nodes(npts) 

              real(c_double), intent(out)    :: lag_flux_l(npts) 
              real(c_double), intent(out)    :: lag_flux_r(npts) 

              integer(c_int)   :: i, j 

              do i = 1, npts
                  call get_lagrange( one, i, nodes, npts, lag_flux_l(i))
                  call get_lagrange(-one, i, nodes, npts, lag_flux_r(i))
              end do

          end subroutine lagr_flux_matrix


          !> Subroutine to create Lagrange differentiation matrix 
          !! @param npts: Number of Lagrange points 
          !! @param nodes: location of points
          !! @param lad_d: differentiation matrix
          subroutine lagr_d_matrix(npts, nodes, lag_d)
              integer(c_int), intent(in)     :: npts
              real(c_double), intent(in)     :: nodes(npts) 

              real(c_double), intent(out)    :: lag_d(npts, npts) 

              integer(c_int)   :: i, j 

              do i = 1, npts
                  do j = 1, npts
                      call get_lagrange_d(nodes(j), i, nodes, npts, lag_d(j, i))
                  end do
              end do

          end subroutine lagr_d_matrix

          !> Subroutine to create Legendre differentiation matrix 
          !! @param order: order of Legendre polynomial
          !! @param npts: Number of points where derivative is required
          !! @param nodes: location of points where deri. is required
          !! @param lege_d: differentiation matrix
          subroutine lege_d_matrix(order, npts, nodes, lege_d)
              integer(c_int), intent(in)     :: order, npts
              real(c_double), intent(in)     :: nodes(npts) 

              real(c_double), intent(out)    :: lege_d(npts) 

              integer(c_int)   :: i, j 

              do i = 1, npts
                  call get_legendre_d(nodes(i), order, lege_d(i))
              end do

          end subroutine lege_d_matrix

          !> Subroutine to create Left Radau differentiation matrix 
          !! @param order: order of Legendre polynomial
          !! @param npts: Number of points where derivative is required
          !! @param nodes: location of points where deri. is required
          !! @param g_l: differentiation matrix
          subroutine left_radau_d(order, npts, nodes, g_l)
              integer(c_int), intent(in)     :: order, npts
              real(c_double), intent(in)     :: nodes(npts) 

              real(c_double), intent(out)    :: g_l(npts) 

              real(c_double)  :: temp1(npts), temp2(npts) 

              call lege_d_matrix(order    , npts, nodes, temp1)
              call lege_d_matrix(order + 1, npts, nodes, temp2)

              g_l = (-one)**(order) * (temp1 - temp2)/(two)

          end subroutine left_radau_d

          !> Subroutine to create Right Radau differentiation matrix 
          !! @param order: order of Legendre polynomial
          !! @param npts: Number of points where derivative is required
          !! @param nodes: location of points where deri. is required
          !! @param g_l: differentiation matrix
          subroutine right_radau_d(order, npts, nodes, g_r)
              integer(c_int), intent(in)     :: order, npts
              real(c_double), intent(in)     :: nodes(npts) 

              real(c_double), intent(out)    :: g_r(npts) 

              real(c_double)  :: temp1(npts), temp2(npts) 

              call lege_d_matrix(order    , npts, nodes, temp1)
              call lege_d_matrix(order + 1, npts, nodes, temp2)

              g_r = half*(temp1 + temp2)

          end subroutine right_radau_d

          subroutine test_matrix
              real(c_double), allocatable :: x(:), deri(:, :) 
              real(c_double), allocatable :: deri1(:) 

              real(c_double), allocatable :: temp1(:) 
              real(c_double), allocatable :: temp2(:) 

              integer(c_int)    :: order, npts, i

              real(c_double)    :: f, r, temp 

              order = 3
              npts  = order + 1 

              allocate(x(npts))
              allocate(deri(npts, npts))
              allocate(deri1(npts))
              allocate(temp1(npts))
              allocate(temp2(npts))

              call gauss_nodes(order, x)

              call lagr_d_matrix(npts, x, deri)

              call lege_d_matrix(order, npts, x, deri1)

              call left_radau_d(order, npts, x, deri1)
!              call right_radau_d(order, npts, x, deri)

              do i = 1, npts 
!                  write(*, *) deri(i, :) 
              end do
!              write(*, *) deri1
!              write(*, *) x
!              write(*, *) matmul(deri, x)
          
              call lagr_flux_matrix(npts, x, temp1, temp2)

!              write(*, *) temp1
!              write(*, *) temp2

              deallocate(x)
              deallocate(deri)
              deallocate(deri1)
              deallocate(temp1)
              deallocate(temp2)


          end subroutine test_matrix

  end module polynomial

  module operators 

      use types_vars
      use polynomial
      implicit none

      contains

          !> Get Jacobian of 1D mesh
          !! We calculate Jacobian assuming a simple 1D mesh
          !! @param nele_x: Number of elements in grid
          !! @param npts: Number of points in each cell
          !! @param order: Order of polynomial 
          !! @param x, x_r: location of FR grid points and Jacobian vec
          subroutine get_jacob(nele_x, npts, order, x, x_r)
              integer(c_int), intent(in)     :: order, nele_x, npts
              real(c_double), intent(in)     :: x(npts, nele_x) 

              real(c_double), intent(out)    :: x_r(npts, nele_x) 

              integer(c_int)  :: i, j 
              real(c_double)  :: nodes(order + 1) 

              call gauss_nodes(order, nodes)
              do i = 1, nele_x
                  do j = 1, npts
                      x_r(j, i) = (x(npts, i) - x(1, i))/(nodes(order + 1) - nodes(1))
                  end do
              end do

          end subroutine get_jacob

          !> Get Roe flux 
          !! @param f_l, f_r: Left and right flux
          !! @param f_I: Interaction or common flux
          subroutine get_roe_flux(u_l, u_r, f_l, f_r, f_I)
              real(c_double), intent(in)     :: u_l, u_r 
              real(c_double), intent(in)     :: f_l, f_r 

              real(c_double), intent(out)    :: f_I

              real(c_double) :: a, temp

              if (abs(u_l - u_r) .gt. epsilon(one)) then
                  a = (f_l - f_r)/(u_l - u_r)
                  f_I = half*(f_l + f_r - abs(a)*(u_r - u_l) ) 
              else
                  if(u_l .gt. zero) then
                      f_I = f_l
                  else
                      f_I = f_r
                  end if
              end if

          end subroutine get_roe_flux


          subroutine get_derivative(nele_x, npts, x, u, du)
              integer(c_int), intent(in)     :: nele_x, npts
              real(c_double), intent(in)     :: x(npts, nele_x) 
              real(c_double), intent(in)     :: u(npts, nele_x) 

              real(c_double), intent(out)    :: du(npts, nele_x) 

              real(c_double)  :: x_r(npts, nele_x) !Jacobian

              real(c_double)  :: lagr_l(npts) !Interpolation vector left flux point 
              real(c_double)  :: flux_l !Left flux
              real(c_double)  :: lagr_r(npts) !Interpolation vector right flux point 
              real(c_double)  :: flux_r !Right flux

              real(c_double)  :: f_I_l, f_I_r !Left and right interation flux

              integer(c_int)  :: order 
              real(c_double)  :: nodes(npts)
              real(c_double)  :: deri(npts, npts)
              integer(c_int)  :: i, j 

              order = npts - 1

              call gauss_nodes(order, nodes)
              call get_jacob(nele_x, npts, order, x, x_r)

              call lagr_d_matrix(npts, nodes, deri)
              do i = 1, nele_x
                  du(:, i) = matmul(deri, u(:, i))/x_r(:, i)
              end do

              call lagr_flux_matrix(npts, x, lagr_l, lagr_r)

              !Get left and right flux at each face
              do i = 2, nele_x
                  flux_l = dot_product(lagr_l, u(:, i - 1)) 
                  flux_r = dot_product(lagr_r, u(:, i    )) 
              end do

          end subroutine get_derivative


  end module operators 


  module plot_data

      use types_vars

      contains

          subroutine plot_sol(nele, np, x, u, du)
              integer(c_int), intent(in)  :: nele, Np
              real(c_double), intent(in)  :: x(:, :), u(:, :)
              real(c_double), intent(in)  :: du(:, :)
            
              integer(c_int)       :: i, j 
              character(len=256)   :: filename
    
              filename='out.dat'
    
              OPEN(10,file=filename,status='replace')
    
              do i = 1, nele
                  do j = 1, np
                      WRITE(10,*) x(j, i), u(j, i), du(j, i)
                  end do
              end do
    
              CLOSE(10)
    
    
          end subroutine plot_sol

  end module plot_data




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
                    u(j, i) = sin(two*pi*x(j, i)/(four*ten))
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
                    x(j, i) = half*(one - x_nodes(j))*temp_x(i)    &
                              + half*(one + x_nodes(j))*temp_x(i+1) 
                end do
            end do

        end subroutine make_grid


        !> Subroutine to validate 2nd and 4th order derivatives
        !! @param nptsx number of points
        !! @param startX, stopX starting and ending location of grid
        !! @param max2nd max4th maximum error for 2nd and 4th order derivative
        subroutine validate_derivative(nele_x, startX, stopX, order)
            
            use polynomial
            use plot_data
            use operators
    
            implicit none
    
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
    
            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
    
            integer(c_int) :: Np 
    
            call test_poly()
            call test_matrix()

            Np = order + 1

            call cell_coordi(order, x_nodes)
    
            call make_grid(nele_x, startX, stopX, Np, x_nodes, x)
    
            call init_sol(nele_x, Np, x, u)

            call get_derivative(nele_x, np, x, u, du)

            call plot_sol(nele_x, Np, x, u, du)
    
        end subroutine validate_derivative



  end module subroutines


