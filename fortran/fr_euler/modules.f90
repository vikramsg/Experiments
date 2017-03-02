  module util
      implicit none

      contains

          subroutine assert(cond, err_msg)
              logical, intent(in) :: cond

              character(256), intent(in), optional :: err_msg

              if ( .not. cond) then
                  if (present(err_msg)) then
                      write(*, *) 'Error: Assertion failed with error '//trim(err_msg)
                  else
                      write(*, *) 'Error: Assertion failed with error '
                  end if
                  call abort() 
              end if
          end subroutine assert

  end module util


  !> Module to define constants
  MODULE const_data 

      use types_vars
      use iso_c_binding

      implicit none

      REAL(DP), PARAMETER :: gamma     = 1.4_c_double
      REAL(DP), PARAMETER :: R_gas     = 287_c_double

      integer(c_int), PARAMETER :: dim = 2  !2 dimensions

  END MODULE const_data 



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

          subroutine plot_sol(np, nele, x, u, du)
              integer(c_int), intent(in)  :: nele, Np
              real(c_double), intent(in)  :: x(:, :), u(:, :, :)
              real(c_double), intent(in)  :: du(:, :, :)
            
              integer(c_int)       :: i, j 
              character(len=256)   :: filename
    
              filename='out.dat'
    
              OPEN(10,file=filename,status='replace')
    
              do i = 1, nele
                  do j = 1, np
                      WRITE(10,*) x(j, i), u(1, j, i), du(1, j, i)
                  end do
              end do
    
              CLOSE(10)
    
    
          end subroutine plot_sol

  end module plot_data


  module solution 
    use types_vars
    use const_data
    use class_fr
    use operators
    use util

      type soln2d

           integer(c_int) :: nele_x, npts, order, Nvar

           real(c_double) :: global_s_max

           real(c_double), allocatable :: x(:, :), u(:, :, :)

           real(c_double), allocatable :: x_r(:, :) !! Jacobian

           real(c_double), allocatable :: deri(:, :)
           real(c_double), allocatable :: extrap(:, :) !! extrapolation matrix to get values at edge

           real(c_double), allocatable :: left_c(:) !! Left correction 
           real(c_double), allocatable :: rght_c(:) !! 

           real(c_double), allocatable :: inv_rhs(:, :, :)

           contains

               procedure, pass :: setup_sol_vec

               procedure, pass :: init_sol 

               procedure, pass :: get_euler_flux
               procedure, pass :: get_rhs 

               procedure, pass :: destructor

      end type soln2d


      contains

       !> Setup solution vector. Cell centerd and solution array
       !! @param nx, ny: number of cells in x and y
       !! @param x, y : grid x and y co-ordinates
       subroutine setup_sol_vec(this, nele_x, np, order, x) 
           class(soln2d), intent(inout) :: this

           integer(c_int), intent(in) :: nele_x, np, order

           real(c_double), intent(in) :: x(:, :)
       
           integer(c_int) :: allo_stat 
           character(256) :: err_msg

           call assert(nele_x .gt. 1)
           call assert(np     .ge. 1)

           this%nele_x = nele_x
           this%npts   = np 
           this%order  = order 

           allocate(this%x(np, nele_x), &
                    stat=allo_stat, errmsg=err_msg)  !grid
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%x_r(np, nele_x), &
                    stat=allo_stat, errmsg=err_msg)  !grid
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%u(this%Nvar, np, nele_x), &
                    stat=allo_stat, errmsg=err_msg)  !soln vec
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%inv_rhs(this%Nvar, np, nele_x), &
                    stat=allo_stat, errmsg=err_msg)  !inviscid rhs
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%deri(np, np), &
                    stat=allo_stat, errmsg=err_msg)  !derivative matrix
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%left_c(np), &
                    stat=allo_stat, errmsg=err_msg)  !derivative matrix
           call assert(allo_stat .eq. 0, err_msg)
           allocate(this%rght_c(np), &
                    stat=allo_stat, errmsg=err_msg)  !derivative matrix
           call assert(allo_stat .eq. 0, err_msg)


           allocate(this%extrap(2, np), &
                    stat=allo_stat, errmsg=err_msg)  !derivative matrix
           call assert(allo_stat .eq. 0, err_msg)

           this%x = x
                   
           call get_jacob(nele_x, np, order, x, this%x_r)

       end subroutine setup_sol_vec


       !> Setup solution vector. Cell centerd and solution array
       !! @param nx, ny: number of cells in x and y
       !! @param x, y : grid x and y co-ordinates
       subroutine destructor(this) 
           class(soln2d), intent(inout) :: this

           if (allocated(this%x)) deallocate(this%x)
           if (allocated(this%x_r)) deallocate(this%x_r)
           if (allocated(this%left_c)) deallocate(this%left_c)
           if (allocated(this%rght_c)) deallocate(this%rght_c)
           if (allocated(this%u)) deallocate(this%u)
           if (allocated(this%deri)) deallocate(this%deri)
           if (allocated(this%extrap)) deallocate(this%extrap)
           if (allocated(this%inv_rhs)) deallocate(this%inv_rhs)
       end subroutine destructor 



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


        !> Subroutine to initialize solution on the FR grid
        !! @param nele_x: Number of elements in grid
        !! @param Np: Number of points in each cell
        !! @param x, u: location of FR grid points and soln vec
        subroutine init_sol(this)
           class(soln2d), intent(inout) :: this

           real(c_double) :: a, k 

           integer(c_int) :: i, j 

           a = three 
           k = pi

           do i = 1, this%nele_x
               do j = 1, this%npts 
                   this%u(1, j, i) =  sin(k*this%x(j, i)) + a
                   this%u(2, j, i) =  sin(k*this%x(j, i)) + a
                   this%u(3, j, i) = (sin(k*this%x(j, i)) + a)**2
               end do
           end do

        end subroutine init_sol


        !> Get Lax Friedrichs type flux
        !! @param f_l, f_r: Left and right flux
        !! @param f_I: Interaction or common flux
        subroutine get_lax_flux(u_l, u_r, f_l, f_r, s_max, f_In)

            real(c_double), intent(in)     :: u_l, u_r 
            real(c_double), intent(in)     :: f_l, f_r
            real(c_double), intent(in)     :: s_max ! max characteristic speed 

            real(c_double), intent(out)    :: f_In

            f_In = half*(f_l + f_r) 
            f_In = f_In - s_max*half*(u_r - u_l)

        end subroutine get_lax_flux



        subroutine get_euler_flux(this, u_vec, f_vec, u, a)
           class(soln2d), intent(inout) :: this

           real(c_double), intent(in ) :: u_vec(:) 
           real(c_double), intent(out) :: f_vec(:) 
           real(c_double), intent(out) :: u, a 

           real(c_double) :: rho, p, v_sq

           integer(c_int) :: Nvar 

           Nvar = this%Nvar

           rho  = u_vec(1)
           u    = u_vec(2)/rho

           v_sq = u**2 

           p    = (gamma - 1)*(u_vec(Nvar) - half*rho*v_sq )

!           write(*, 100) u_vec(1), u_vec(2), u_vec(3), p
!100        format(4e15.5)

           a    = sqrt(gamma * p/rho)

           f_vec(1)    =  u_vec(2)           ! f(1) = rho*u 
           f_vec(2)    =  u_vec(2)*u + p     ! f(2) = rho*u*u + p 
           f_vec(Nvar) = (u_vec(Nvar) + p)*u ! f(4) = (E_t + p)*u 

        end subroutine get_euler_flux 



        subroutine get_rhs(this)
           class(soln2d), intent(inout) :: this

           real(c_double) :: f_vec(this%Nvar, this%npts) 

           real(c_double) ::   u_d(this%Nvar, 2, this%nele_x) !! Discontinuous flux at edge 

           real(c_double) ::   f_d(this%Nvar, 2, this%nele_x) !! Discontinuous flux at edge 
           real(c_double) ::   f_I(this%Nvar, 2, this%nele_x) !! Interaction flux at edge 
           real(c_double) ::  df_x(this%Nvar, this%npts, this%nele_x) 

           real(c_double) ::  u, a, s_max 

           integer(c_int) :: i, j, var, Nvar

           Nvar = this%Nvar

           s_max = zero

           do i = 1, this%nele_x
               do j = 1, this%npts
                   call this%get_euler_flux(this%u(:, j, i), f_vec(:, j), u, a)
                   s_max = max(s_max, abs(u) + a)
               end do

               do var = 1, Nvar
                   df_x(var, :, i) = matmul(this%deri  ,  f_vec(var, :)    ) !Get discontinuous derivative
                   f_d( var, :, i) = matmul(this%extrap,  f_vec(var, :)    ) !Get extrapolation 
                   u_d( var, :, i) = matmul(this%extrap, this%u(var, :, i) ) !Get extrapolation 
               end do
           end do

           this%global_s_max = s_max

           f_I = zero

           do i = 2, this%nele_x - 1
               do j = 1, Nvar 
                   call get_lax_flux(u_d(j, 2, i - 1), u_d(j, 1, i    ), f_d(j, 2, i - 1), &
                                 f_d(j, 1, i    ), s_max, f_I(j, 1, i))
                   call get_lax_flux(u_d(j, 2, i    ), u_d(j, 1, i + 1), f_d(j, 2, i    ), &
                                 f_d(j, 1, i + 1), s_max, f_I(j, 2, i))
               end do
           end do

           !! Periodic boundary conditions
           do j = 1, Nvar
               call get_lax_flux(u_d(j, 2, this%nele_x), u_d(j, 1, 1), f_d(j, 2, this%nele_x), &
                             f_d(j, 1, 1), s_max, f_I(j, 1, 1))
               call get_lax_flux(u_d(j, 2, 1          ), u_d(j, 1, 2), f_d(j, 2, 1          ), &
                             f_d(j, 1, 2), s_max, f_I(j, 2, 1))


               call get_lax_flux(u_d(j, 2, this%nele_x    ), u_d(j, 1, 1          ), &
                            f_d(j, 2, this%nele_x    ),  f_d(j, 1, 1          ), s_max, f_I(j, 2, this%nele_x))
               call get_lax_flux(u_d(j, 2, this%nele_x - 1), u_d(j, 1, this%nele_x), &
                             f_d(j, 2, this%nele_x - 1), f_d(j, 1, this%nele_x), s_max, f_I(j, 1, this%nele_x))
           end do

           this%inv_rhs = zero

           do i = 1, this%nele_x 
               do j = 1, Nvar 
                   this%inv_rhs(j, :, i) = df_x(j, :, i) + (f_I(j, 1, i) - f_d(j, 1, i))*this%left_c + &
                                                           (f_I(j, 2, i) - f_d(j, 2, i))*this%rght_c 
               
                   !           !Transform derivative to physical space                  
                   this%inv_rhs(j, :, i) = this%inv_rhs(j, :, i)/this%x_r(:, i)
               end do
           end do

        end subroutine get_rhs 


  end module solution 





  module subroutines
    use types_vars
    use class_fr

    implicit none
    
    contains

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
            use solution
    
            implicit none
    
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 
            real(c_double), intent(in)  :: stopT 
            real(c_double), intent(in)  :: nu 
    
            real(c_double) :: dx ! Characteristic length of cell 
            real(c_double) :: dt ! time spacing 
            real(c_double) :: time ! time 

            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: x_r(order + 1, nele_x) ! Jacobian 
    
            integer(c_int) :: Np !Number of points in a cell 

            integer(c_int) :: steps, num_steps  !Iterator for time, number of steps

            type(fr)       :: fr_run
        
            type(soln2d)   :: soln

            Np = order + 1

            call cell_coordi(order, x_nodes) !Get intra-cell co-ordis

            call make_grid(nele_x, startX, stopX, Np, x_nodes, x, dx) !Get grid and char. len
          
            !Initializing FR class
            call fr_run%init_operators(order, Np, nele_x, x_r)

            dt = (nu*dx/((2*order + 1)))/four !Assuming wave speed = 4
            num_steps = int(stopT/dt)

            soln%Nvar = 1 + 2

            call soln%setup_sol_vec(nele_x, Np, order, x)
            soln%deri         = fr_run%lagr_deri !! Get derivative matrix
            soln%extrap(1, :) = fr_run%lagr_l    !! Get extrapolation matrix
            soln%extrap(2, :) = fr_run%lagr_r    !! Get extrapolation matrix
            soln%left_c(   :) = fr_run%g_l       !! Get left correction 
            soln%rght_c(   :) = fr_run%g_r       !! Get right correction 

            call soln%init_sol

            time  = zero
            steps = zero
            do 
                call soln%get_rhs
                soln%u = soln%u - dt*soln%inv_rhs
               
                time = time + dt
                write(*, *) steps, time, soln%global_s_max
            
                dt = (nu*dx/((2*order + 1)))/(soln%global_s_max)

                if (time .gt. stopT) stop

                steps = steps + 1
            end do

!            call plot_sol(Np, nele_x, x, soln%u, soln%u)

            !Finalize FR class
            call fr_run%kill_all()

            call soln%destructor

        end subroutine wave_solver

  end module subroutines


