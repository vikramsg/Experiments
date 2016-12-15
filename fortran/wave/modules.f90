  
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

          !> Get centered flux 
          !! @param f_l, f_r: Left and right flux
          !! @param f_I: Interaction or common flux
          subroutine get_centered_flux(u_l, u_r, f_l, f_r, f_I)
              real(c_double), intent(in)     :: u_l, u_r 
              real(c_double), intent(in)     :: f_l, f_r 

              real(c_double), intent(out)    :: f_I

              f_I = half*(f_l + f_r)

          end subroutine get_centered_flux

          !> Subroutine to get discont. flux at flux points 
          !! It assumes periodic boundary conditions for now
          !! @param nele_x: number of elements
          !! @param extrapol_mat: extraopolation matrix to flux points 
          !! @param f: f at solution points 
          !! @param f_d: extrapolated discontinuous flux 
          subroutine get_flux_disc_f(nele_x, extrapol_mat, f, f_d)
              integer(c_int), intent(in)     :: nele_x
              real(c_double), intent(in)     :: extrapol_mat(:, :) 
              real(c_double), intent(in)     :: f(:, :) 

              real(c_double), intent(out)    :: f_d(:, :) 

              integer(c_int) :: i 

              do i = 1, nele_x
                  f_d(:, i) = matmul(extrapol_mat, f(:, i))
              end do


          end subroutine get_flux_disc_f


          !> Subroutine to get second derivative
          !! It assumes periodic boundary conditions for now
          !! @param nele_x: number of elements
          !! @param order: order of interpolating polynomial 
          !! @param npts: number of points in each element
          !! @param x: solution points vector
          !! @param u: solution vector
          !! @param du: derivative vector 
          subroutine get_derivative_2(nele_x, order, npts, x, u, du)
              integer(c_int), intent(in)     :: nele_x, npts, order
              real(c_double), intent(in)     :: x(npts, nele_x) 
              real(c_double), intent(in)     :: u(npts, nele_x) 

              real(c_double), intent(out)    :: du(npts, nele_x) 

              real(c_double)  :: x_r(npts, nele_x) !Jacobian

              real(c_double)  :: g_l(npts) !Left radau derivative
              real(c_double)  :: g_r(npts) !Right radau derivative

              real(c_double)  :: flux_f(2, nele_x) !Flux at cell edge

              real(c_double)  :: lagr_l(npts) !Interpolation vector left flux point 
              real(c_double)  :: flux_l !Left flux
              real(c_double)  :: lagr_r(npts) !Interpolation vector right flux point 
              real(c_double)  :: flux_r !Right flux

              real(c_double)  :: extrap_flux(2, npts) !Extrapolation matrix for flux points

              real(c_double)  :: f_I_l, f_I_r !Left and right interation flux

              real(c_double)  :: nodes(npts)
              real(c_double)  :: deri(npts, npts)
              integer(c_int)  :: i, j 

              call gauss_nodes(order, nodes)
              call get_jacob(nele_x, npts, order, x, x_r)

              call lagr_d_matrix(npts, nodes, deri)
              
              call left_radau_d(order,  npts, nodes, g_l)
              call right_radau_d(order, npts, nodes, g_r)

              call lagr_flux_matrix(npts, nodes, lagr_l, lagr_r)
              extrap_flux(1, :) = lagr_l 
              extrap_flux(2, :) = lagr_r 

              do i = 1, nele_x
                  du(:, i) = matmul(deri, u(:, i)) !Get discontinuous derivative
              end do

              call get_flux_disc_f(nele_x, extrap_flux, u, flux_f) !Get discontinuous flux at flux point

              do i = 2, nele_x - 1
                  call get_centered_flux(flux_f(2, i-1), flux_f(1, i), flux_f(2, i-1), flux_f(1, i), f_I_l)

                  call get_centered_flux(flux_f(2, i), flux_f(1, i+1), flux_f(2, i), flux_f(1, i+1), f_I_r)
              
                  du(:, i) = du(:, i) + (f_I_l - flux_f(1, i))*g_l +  (f_I_r - flux_f(2, i))*g_r
              end do

              !At left boundary
              call get_centered_flux(flux_f(2, nele_x), flux_f(1, 1), flux_f(2, nele_x), flux_f(1, 1), f_I_l)
              call get_centered_flux(flux_f(2, 1), flux_f(1, 1+1), flux_f(2, 1), flux_f(1, 1+1), f_I_r)
                  
              du(:, 1) = du(:, 1) + (f_I_l - flux_f(1, 1))*g_l +  (f_I_r - flux_f(2, 1))*g_r
              
              !At right boundary
              call get_centered_flux(flux_f(2, nele_x-1), flux_f(1, nele_x), flux_f(2, nele_x -1 ), flux_f(1, nele_x), f_I_l)
              call get_centered_flux(flux_f(2, nele_x), flux_f(1, 1), flux_f(2, nele_x), flux_f(1, 1), f_I_r)
                  
              du(:, nele_x) = du(:, nele_x) + (f_I_l - flux_f(1, nele_x))*g_l +  (f_I_r - flux_f(2, nele_x))*g_r

              !Transform derivative to physical space                  
              du(:, 1:nele_x) = du(:, 1:nele_x)/x_r(:, 1:nele_x)
          end subroutine get_derivative_2



          !> Subroutine to get first derivative
          !! It assumes periodic boundary conditions for now
          !! @param nele_x: number of elements
          !! @param npts: number of points in each element
          !! @param x: solution points vector
          !! @param u: solution vector
          !! @param du: derivative vector 
          subroutine get_derivative(nele_x, npts, x, u, du)
              integer(c_int), intent(in)     :: nele_x, npts
              real(c_double), intent(in)     :: x(npts, nele_x) 
              real(c_double), intent(in)     :: u(npts, nele_x) 

              real(c_double), intent(out)    :: du(npts, nele_x) 

              integer(c_int)  :: order 

              real(c_double)  :: x_r(npts, nele_x) !Jacobian

              real(c_double)  :: g_l(npts) !Left radau derivative
              real(c_double)  :: g_r(npts) !Right radau derivative

              real(c_double)  :: lagr_l(npts) !Interpolation vector left flux point 
              real(c_double)  :: flux_l !Left flux
              real(c_double)  :: lagr_r(npts) !Interpolation vector right flux point 
              real(c_double)  :: flux_r !Right flux

              real(c_double)  :: f_I_l, f_I_r !Left and right interation flux

              real(c_double)  :: nodes(npts)
              real(c_double)  :: deri(npts, npts)
              integer(c_int)  :: i, j 

              order = npts - 1

              call gauss_nodes(order, nodes)
              call get_jacob(nele_x, npts, order, x, x_r)

              call lagr_d_matrix(npts, nodes, deri)
              
              do i = 1, nele_x
                  du(:, i) = matmul(deri, u(:, i))
              end do

              call left_radau_d(order,  npts, nodes, g_l)
              call right_radau_d(order, npts, nodes, g_r)

              call lagr_flux_matrix(npts, nodes, lagr_l, lagr_r)

              !Get left and right flux at each face
              do i = 2, nele_x - 1
                  !Get interaction flux at left face
                  flux_l = dot_product(lagr_r, u(:, i - 1)) 
                  flux_r = dot_product(lagr_l, u(:, i    )) 

                  call get_roe_flux(flux_l, flux_r, flux_l, flux_r, f_I_l)

                  !Get interaction flux at right face
                  flux_l = dot_product(lagr_r, u(:, i    )) 
                  flux_r = dot_product(lagr_l, u(:, i + 1)) 

                  call get_roe_flux(flux_l, flux_r, flux_l, flux_r, f_I_r)

                  !Get interaction flux at cell boundaries 
                  flux_r = dot_product(lagr_r, u(:, i)) 
                  flux_l = dot_product(lagr_l, u(:, i)) 

                  du(:, i) = du(:, i) + (f_I_l - flux_l)*g_l + (f_I_r - flux_r)*g_r 
              end do

              !Periodic boundary Left
              !Get interaction flux at left face
              flux_l = dot_product(lagr_r, u(:, nele_x)) 
              flux_r = dot_product(lagr_l, u(:, 1     )) 

              call get_roe_flux(flux_l, flux_r, flux_l, flux_r, f_I_l)

              !Get interaction flux at right face
              flux_l = dot_product(lagr_r, u(:, 1    )) 
              flux_r = dot_product(lagr_l, u(:, 1 + 1)) 

              call get_roe_flux(flux_l, flux_r, flux_l, flux_r, f_I_r)

              !Get interaction flux at cell boundaries 
              flux_r = dot_product(lagr_r, u(:, 1)) 
              flux_l = dot_product(lagr_l, u(:, 1)) 

              du(:, 1) = du(:, 1) + (f_I_l - flux_l)*g_l + (f_I_r - flux_r)*g_r 

              !Periodic boundary Right 
              !Get interaction flux at left face
              flux_l = dot_product(lagr_r, u(:, nele_x - 1)) 
              flux_r = dot_product(lagr_l, u(:, nele_x    )) 

              call get_roe_flux(flux_l, flux_r, flux_l, flux_r, f_I_l)
              
              !Get interaction flux at right face
              flux_l = dot_product(lagr_r, u(:, nele_x)) 
              flux_r = dot_product(lagr_l, u(:, 1     )) 

              call get_roe_flux(flux_l, flux_r, flux_l, flux_r, f_I_r)

              !Get interaction flux at cell boundaries 
              flux_r = dot_product(lagr_r, u(:, nele_x)) 
              flux_l = dot_product(lagr_l, u(:, nele_x)) 

              du(:, nele_x) = du(:, nele_x) + (f_I_l - flux_l)*g_l + (f_I_r - flux_r)*g_r 

              !Transform derivative to physical space                  
              du(:, 1:nele_x) = du(:, 1:nele_x)/x_r(:, 1:nele_x)
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
            real(c_double) :: x_l, x_r 

            !!!!!!!!!!!!!!!!!!!!!!!
            !sine wave
            do i = 1, nele_x
                do j = 1, Np
                    u(j, i) = sin(two*pi*x(j, i)/(one))
                end do
            end do

!            x_l = four/ten 
!            x_r = six/ten
!            !Delta function
!            do i = 1, nele_x
!                do j = 1, Np
!                    u(j, i) = 0 
!                    if ((x(j, i) .le. x_r) .and. (x(j, i) .ge. x_l)) then
!                        u(j, i) = hundred*(x(j, i)-x_l)*(x_r-x(j, i))
!                    end if
!                end do
!            end do

        end subroutine init_sol

        !> Subroutine to make simple 1D grid
        !! @param nptsx num. of points
        !! @param startX, stopX starting and ending location
        !! @param x array containing grid points
        subroutine make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx)
            integer(c_int), intent(in)  :: nele_x, Np
            real(c_double), intent(in)  :: startX, stopX 
            real(c_double), intent(in)  :: x_nodes(Np) 
    
            real(c_double), intent(out) :: x(:, :), dx
    
            integer(c_int) :: i, j 
            real(c_double) :: temp_x(nele_x + 1) 
    
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


        !> Subroutine to validate FR derivative
        !! @param nptsx number of points
        !! @param startX, stopX starting and ending location of grid
        !! @param order: order of polynomial
        subroutine validate_derivative(nele_x, startX, stopX, order)
            
            use polynomial
            use plot_data
            use operators
    
            implicit none
    
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
    
            real(c_double) :: dx ! Characteristic length of cell 

            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
    
            integer(c_int) :: Np, i 
    
            call test_poly()
            call test_matrix()
            call fun_matrix()

            Np = order + 1

            call cell_coordi(order, x_nodes)
    
            call make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx)
    
            call init_sol(nele_x, Np, x, u)

            call get_derivative(nele_x, np, x, u, du)

            write(*, *) du(:, 1)

            call get_derivative_2(nele_x, order, np, x, u, du)

            write(*, *) du(:, 1)

            call plot_sol(nele_x, Np, x, u, du)
    
        end subroutine validate_derivative



        !> Subroutine to solve wave equation 
        !! @param nptsx number of points
        !! @param startX, stopX starting and ending location of grid
        !! @param stopT: stop time 
        !! @param order: order of polynomial
        !! @param nu: CFL value
        subroutine wave_solver(nele_x, startX, stopX, stopT, order, nu)
            use polynomial
            use plot_data
            use operators
    
            implicit none
    
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
            real(c_double), intent(in)  :: stopT 
            real(c_double), intent(in)  :: nu 
    
            real(c_double) :: dx ! Characteristic length of cell 
            real(c_double) :: dt ! time spacing 

            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: u_new(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
    
            integer(c_int) :: Np !Number of points in a cell 

            integer(c_int) :: steps, num_steps  !Iterator for time, number of steps
 
            Np = order + 1

            call cell_coordi(order, x_nodes) !Get intra-cell co-ordis

            call make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx) !Get grid and char. len

            dt = nu*dx/(2*order + 1) !Assuming wave speed = 1
            num_steps = stopT/dt

            call init_sol(nele_x, Np, x, u)

            do steps = 1, num_steps 
                call get_derivative(nele_x, np, x, u, du)

                u = u - dt*du
            end do

            call plot_sol(nele_x, Np, x, u, du)

        end subroutine wave_solver


  end module subroutines


