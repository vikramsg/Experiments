  
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

        subroutine exact_sol_sin_diff(nptsx, x, c, L, alpha, t, u_ex)

            integer(c_int), intent(in)  :: nptsx
            real(c_double), intent(in)  :: c, L, t, alpha
            real(c_double), intent(in)  :: x(0:nptsx)
            real(c_double), intent(out) :: u_ex(0:nptsx)
    
            integer(c_int) :: i 
    
            do i = 0, nptsx
                u_ex(i) = c*exp((-alpha*pi**2*t)/L**2)*sin(pi*(x(i)/L))
            end do
    
    
        end subroutine exact_sol_sin_diff 


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
            use class_fr
    
            implicit none
    
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
            real(c_double), intent(in)  :: stopT 
            real(c_double), intent(in)  :: nu 
    
            real(c_double) :: dx ! Characteristic length of cell 
            real(c_double) :: dt ! time spacing 

            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: x_r(order + 1, nele_x) ! Jacobian 
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: u_new(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
    
            integer(c_int) :: Np !Number of points in a cell 

            integer(c_int) :: steps, num_steps  !Iterator for time, number of steps

            type(fr)       :: fr_run

            Np = order + 1

            call cell_coordi(order, x_nodes) !Get intra-cell co-ordis

            call make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx) !Get grid and char. len
          
            call get_jacob(nele_x, Np, order, x, x_r)

            dt = nu*dx/((2*order + 1)) !Assuming wave speed = 1
            num_steps = stopT/dt

            call init_sol(nele_x, Np, x, u)

            !Initializing FR class
            call fr_run%init_operators(order, Np, nele_x, x_r)

            do steps = 1, num_steps 

                call fr_run%get_first_deri(x, u, du)

                u = u - dt*du
            end do

            call plot_sol(nele_x, Np, x, u, du)

        end subroutine wave_solver



        !> Subroutine to solve diffusion equation 
        !! @param nptsx number of points
        !! @param startX, stopX starting and ending location of grid
        !! @param stopT: stop time 
        !! @param order: order of polynomial
        !! @param nu: CFL value
        subroutine diff_solver(nele_x, startX, stopX, stopT, order, nu)
            use polynomial
            use plot_data
            use operators
            use class_fr
    
            implicit none
    
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
            real(c_double), intent(in)  :: stopT 
            real(c_double), intent(in)  :: nu 
    
            real(c_double) :: dx ! Characteristic length of cell 
            real(c_double) :: dt ! time spacing 

            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: x_r(order + 1, nele_x) ! Jacobian 
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: u_new(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
    
            integer(c_int) :: Np !Number of points in a cell 

            integer(c_int) :: steps, num_steps  !Iterator for time, number of steps

            type(fr)       :: fr_run

            Np = order + 1

            call cell_coordi(order, x_nodes) !Get intra-cell co-ordis

            call make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx) !Get grid and char. len
          
            call get_jacob(nele_x, Np, order, x, x_r)

            dt = nu*dx*dx/((two*order + one)**two) 
            num_steps = stopT/dt

            call init_sol(nele_x, Np, x, u)

            !Initializing FR class
            call fr_run%init_operators(order, Np, nele_x, x_r)

!            do steps = 1, num_steps 
            do steps = 1, 10

                call fr_run%get_sec_deri(x, u, du)

                u = u + dt*du
            end do

            !Finalize FR class
            call fr_run%kill_all()

            write(*, *) u(:, 10:15)

            call plot_sol(nele_x, Np, x, u, du)

        end subroutine diff_solver


        !> Subroutine to validate FR derivative
        !! @param nptsx number of points
        !! @param startX, stopX starting and ending location of grid
        !! @param order: order of polynomial
        subroutine validate_derivative(nele_x, startX, stopX, order)
            
            use polynomial
            use plot_data
            use operators
            use class_fr
      
            implicit none
      
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
      
            real(c_double) :: dx ! Characteristic length of cell 
  
            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: x_r(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
      
            integer(c_int) :: Np, i 

            type(fr) :: fr_run
      
            call test_poly()
            call test_matrix()
            call fun_matrix()

            Np = order + 1

            call cell_coordi(order, x_nodes) !Get intra-cell co-ordis

            call make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx) !Get grid and char. len
          
            call get_jacob(nele_x, Np, order, x, x_r)

            call init_sol(nele_x, Np, x, u)

            !Initializing FR class
            call fr_run%init_operators(order, Np, nele_x, x_r)

            call fr_run%get_first_deri(x, u, du)

            call fr_run%get_sec_deri(x, u, du)

            call plot_sol(nele_x, Np, x, u, du)
      
        end subroutine validate_derivative



  end module subroutines


