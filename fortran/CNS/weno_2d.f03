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


  !> Module to defined data types and some constants
  MODULE types_vars

      use iso_c_binding
      use util

      implicit none

    ! Symbolic names for kind types of single- and double-precison reals
    INTEGER, PARAMETER :: SP = KIND(1.0_c_float)
    INTEGER, PARAMETER :: DP = KIND(1.0_c_double)

    ! Frequently used mathematical constants (with precision to spare)
    REAL(DP), PARAMETER :: zero    = 0.0_c_double
    REAL(DP), PARAMETER :: half    = 0.5_c_double  
    REAL(DP), PARAMETER :: tenth   = 0.1_c_double  
    REAL(DP), PARAMETER :: third   = 0.3333333333333333333333333333333333333_c_double  
    REAL(DP), PARAMETER :: fourth  = 0.25_c_double  
    REAL(DP), PARAMETER :: fifth   = 0.20_c_double  
    REAL(DP), PARAMETER :: one     = 1.0_c_double
    REAL(DP), PARAMETER :: two     = 2.0_c_double
    REAL(DP), PARAMETER :: three   = 3.0_c_double
    REAL(DP), PARAMETER :: four    = 4.0_c_double
    REAL(DP), PARAMETER :: five    = 5.0_c_double
    REAL(DP), PARAMETER :: six     = 6.0_c_double
    REAL(DP), PARAMETER :: seven   = 7.0_c_double
    REAL(DP), PARAMETER :: eight   = 8.0_c_double
    REAL(DP), PARAMETER :: nine    = 9.0_c_double
    REAL(DP), PARAMETER :: ten     = 10.0_c_double
    REAL(DP), PARAMETER :: hundred = 100.0_c_double
    REAL(DP), PARAMETER :: pi    = 3.141592653589793238462643383279502884197_c_double
    REAL(DP), PARAMETER :: pio2  = 1.57079632679489661923132169163975144209858_c_double
    REAL(DP), PARAMETER :: twopi = 6.283185307179586476925286766559005768394_c_double

    REAL(DP), PARAMETER :: tol   = epsilon(one)  !machine tolerance
  END MODULE types_vars

!> Module to define input data 
  module input_data 

      use types_vars
      use iso_c_binding

      implicit none

      REAL(DP) :: Re_inf        = 400_c_double
      REAL(DP) :: mu_inf        = 0.0025_c_double 
      REAL(DP) :: Pr_inf        = 0.72_c_double
      REAL(DP) :: M_inf         = 0.2_c_double
      REAL(DP) :: u_inf         = 350_c_double
      REAL(DP) :: T_inf         = 300_c_double
      REAL(DP) :: rho_inf       = 1_c_double

  end module input_data 



!> Module to define constants
  MODULE const_data 

      use types_vars
      use iso_c_binding

      implicit none

      REAL(DP), PARAMETER :: gamma     = 1.4_c_double
      REAL(DP), PARAMETER :: R_gas     = 287_c_double

      integer(c_int), PARAMETER :: dim = 2  !2 dimensions

  END MODULE const_data 


  module grid 

      use types_vars
      implicit none

      type mesh2d
           real(c_double) :: x_l, x_r 
           real(c_double) :: y_l, y_r 

           integer(c_int) :: ncells_x
           integer(c_int) :: ncells_y

           real(c_double), allocatable :: x(:, :), y(:, :) 

           real(c_double), allocatable :: dx(:, :), dy(:, :) 

           contains

               procedure, pass :: get_grid2d

               final :: destructor

      end type mesh2d

      contains

       subroutine destructor(this) 
           type(mesh2d) :: this

           if (allocated(this%x)) deallocate(this%x)
           if (allocated(this%y)) deallocate(this%y)
           
           if (allocated(this%dx)) deallocate(this%dx)
           if (allocated(this%dy)) deallocate(this%dy)

       end subroutine destructor 

       !> Create a 2D grid. Spacing is uniform in x and y respectively
       !! @param nx, ny: number of cells in x and y direction
       !! @param start*, stop* : Starting and ending co-ordinates in each direction
       subroutine get_grid2d(this, nx, ny, startX, stopX, startY, stopY)
           class(mesh2d), intent(inout) :: this

           integer(c_int) :: nx, ny

           real(c_double) :: startX, stopX 
           real(c_double) :: startY, stopY 

           integer(c_int) :: i, j 

           integer(c_int) :: allo_stat 
           character(256) :: err_msg

           call assert(nx .gt. 1)

           allocate(this%x(nx+1, ny+1), stat=allo_stat, errmsg=err_msg)
           call assert(allo_stat .eq. 0, err_msg)
           allocate(this%y(nx+1, ny+1), stat=allo_stat, errmsg=err_msg)
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%dx(nx+1, ny+1), stat=allo_stat, errmsg=err_msg)
           call assert(allo_stat .eq. 0, err_msg)
           allocate(this%dy(nx+1, ny+1), stat=allo_stat, errmsg=err_msg)
           call assert(allo_stat .eq. 0, err_msg)

           this%ncells_x = nx
           this%ncells_y = ny

           this%x_l = startX 
           this%x_r = stopX 

           this%y_l = startY 
           this%y_r = stopY 

           this%dx = (this%x_r - this%x_l)/this%ncells_x
           this%dy = (this%y_r - this%y_l)/this%ncells_y

           do i = 1, this%ncells_x + 1
               do j = 1, this%ncells_y + 1
                   this%x(i, j) = this%x_l + (i - 1)*this%dx(i, j)
                   this%y(i, j) = this%y_l + (j - 1)*this%dy(i, j)
               end do
           end do

       end subroutine get_grid2d

  end module grid



  module solution 
      
      use types_vars
      use const_data
      implicit none

      type soln2d
           integer(c_int) :: ncells_x, ncells_y  

           integer(c_int) :: n_ghost_cells !Now we will actually include ghost cells as well

!           real(c_double), allocatable :: global_s_max
           real(c_double) :: global_s_max
           !! Maximum signal velocity on the grid

           real(c_double) :: pres_time 

           real(c_double), allocatable :: u(:, :, :)   !Soln vector
           real(c_double), allocatable :: rhs(:, :, :) !RHS for the finite volume

           real(c_double), allocatable :: inv_rhs(:, :, :) !Inviscid RHS for the finite volume
           real(c_double), allocatable :: vis_rhs(:, :, :) !Viscous RHS for the finite volume

           real(c_double), allocatable :: x_c(:, :) !Cell center
           real(c_double), allocatable :: y_c(:, :) !Cell center

           real(c_double), allocatable :: dx(:, :)  !x direction spacing
           real(c_double), allocatable :: dy(:, :)  !x direction spacing

           contains

               procedure, pass :: setup_sol_vec

               procedure, pass :: init_sol_case3 
               procedure, pass :: init_sol
               procedure, pass :: extrapolate 
               procedure, pass :: set_periodic 

               procedure, pass :: init_vst
               procedure, pass :: set_vst 

               procedure, pass :: init_ldc
               procedure, pass :: set_ldc 

               procedure, pass :: get_right_face_derivative_1
               procedure, pass :: get_top_face_derivative_1
               procedure, pass :: get_face_derivative_1

               procedure, pass :: get_vis_rhs
 
               procedure, pass :: get_lax_f_flux 

               procedure, pass :: get_inv_rhs 

               procedure, pass :: cns_solve 

               final :: destructor

      end type soln2d

      interface
       subroutine solve(this)
           import soln2d
           class(soln2d), intent(inout) :: this
       end subroutine solve 
      end interface
 
      contains


       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !! Initial and boundary conditions
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


          !> Setup case 3 solution from sec 4.3
       subroutine init_sol_case3(this) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny

           real(c_double) :: rho11, u11, v11, p11  
           real(c_double) :: rho12, u12, v12, p12

           real(c_double) :: rho21, u21, v21, p21
           real(c_double) :: rho22, u22, v22, p22

           real(c_double) :: rho, u, v, p

           real(c_double) :: x, y
           real(c_double) :: v_sq 

           integer(c_int) :: i, j

           p21 = three*tenth;  rho21 = 0.5323*one;  u21 = 1.206*one; v21 = zero
           p22 = one + half;   rho22 = one + half;  u22 = zero;      v22 = zero

           p11 = 0.029*one;    rho11 = 0.138*one;   u11 = 1.206*one; v11 = 1.206*one;
           p12 = three*tenth;  rho12 = 0.5323*one;  u12 = zero;      v12 = 1.206*one;

!           p21 = one;          rho21 = one;         u21 = 0.7276*one; v21 = zero
!           p22 = four*tenth;   rho22 = 0.5313*one;  u22 = zero;       v22 = zero
!
!           p11 = one;          rho11 = eight*tenth; u11 = zero;       v11 = zero;
!           p12 = one;          rho12 = one;         u12 = zero;       v12 = 0.7276*one;

           nx = this%ncells_x
           ny = this%ncells_y

           do i = 0, this%ncells_x + 1
               do j = 0, this%ncells_y + 1
                   x   = this%x_c(i, j)
                   y   = this%y_c(i, j)

                   if ((x .le. half) .and. (y .le. half)) then
                       p   = p11 
                       rho = rho11
                       u   = u11
                       v   = v11

                       v_sq = u**2 + v**2 
                   else if ((x .gt. half) .and. (y .le. half)) then
                       p   = p12 
                       rho = rho12
                       u   = u12
                       v   = v12

                       v_sq = u**2 + v**2 
                   else if ((x .le. half) .and. (y .gt. half)) then
                       p   = p21 
                       rho = rho21
                       u   = u21
                       v   = v21

                       v_sq = u**2 + v**2 
                   else if ((x .gt. half) .and. (y .gt. half)) then
                       p   = p22 
                       rho = rho22
                       u   = u22
                       v   = v22

                       v_sq = u**2 + v**2 
                   end if

                   this%u(1, i, j) = rho
                   this%u(2, i, j) = rho*u
                   this%u(3, i, j) = rho*v
                   this%u(dim + 2, i, j) = p/(gamma - 1) + half*rho*v_sq  !E_t
               end do
           end do

       end subroutine init_sol_case3



       !> Setup exact smooth solution from sec 4.1
       subroutine init_sol(this) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny

           real(c_double) :: rho, u, v, p, x, y
           real(c_double) :: v_sq 

           integer(c_int) :: i, j 

           nx = this%ncells_x
           ny = this%ncells_y

           u = one
           v = -half 
           p = one

           do i = 0, this%ncells_x + 1
               do j = 0, this%ncells_y + 1
                   x   = this%x_c(i, j)
                   y   = this%y_c(i, j)
                   rho = one + two*tenth*sin(pi*(x + y)) 

                   v_sq = u**2 + v**2 

                   this%u(1, i, j) = rho
                   this%u(2, i, j) = rho*u
                   this%u(3, i, j) = rho*v
                   this%u(dim + 2, i, j) = p/(gamma - 1) + half*rho*v_sq  !E_t
               end do
           end do

       end subroutine init_sol


       !> Set bc for viscous shock tube 
       subroutine init_vst(this) 
           use input_data
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny, n_g_cell

           integer(c_int) :: i, j

           real(c_double) :: rho, u, v, p, x, y
           real(c_double) :: T, M, a
           real(c_double) :: v_sq 

           nx = this%ncells_x
           ny = this%ncells_y

           T   = T_inf
           rho = rho_inf

           u   = 0 
           v   = 0
           v_sq= u**2 + v**2

           p = rho*R_gas*T

           do i = 1, this%ncells_x
               do j = 1, this%ncells_y
                   x   = this%x_c(i, j)

                   if (x .lt. half) then
                       rho = 120*one
                       p   = rho/gamma 
                       u   = zero
                       v   = zero
                   else
                       rho = 1.2*one 
                       p   = rho/gamma 
                       u   = zero 
                       v   = zero 
                   end if

                   v_sq = u**2 + v**2

                   this%u(      1, i, j) = rho                            !rho
                   this%u(      2, i, j) = rho*u                          !rho*u
                   this%u(      3, i, j) = rho*v                          !rho*v
                   this%u(dim + 2, i, j) = p/(gamma - 1) + half*rho*v_sq  !E_t
               end do
           end do

       end subroutine init_vst



       !> set bc for viscous shock tube 
       subroutine set_vst(this) 
           use input_data
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny, n_g_cell

           integer(c_int) :: i, j

           real(c_double) :: rho, u, v, p, x, y, vel
           real(c_double) :: v_sq, m, t
           real(c_double) :: dy, u2, u1, h, a, b

           nx = this%ncells_x
           ny = this%ncells_y

           n_g_cell = this%n_ghost_cells

           do i = 1, this%n_ghost_cells        
               !! rho is simply set equal
               this%u(1, -(i - 1), 1:ny) =  this%u(1, 1  + (i - 1), 1:ny) 
               !! velocities are set as opposite
               this%u(2, -(i - 1), 1:ny) = -this%u(2, 1  + (i - 1), 1:ny) 
               this%u(3, -(i - 1), 1:ny) = -this%u(3, 1  + (i - 1), 1:ny) 
               !! energy is set equal for adiabatic wall
               this%u(dim + 2, -(i - 1), 1:ny) = this%u(dim + 2, 1 + (i - 1), 1:ny) 

               !! rho is simply set equal
               this%u(1,   nx + i, 1:ny) =  this%u(1, nx - (i - 1), 1:ny) 
               !! velocities are set as opposite
               this%u(2,   nx + i, 1:ny) = -this%u(2, nx - (i - 1), 1:ny) 
               this%u(3,   nx + i, 1:ny) = -this%u(3, nx - (i - 1), 1:ny) 
               !! energy is set equal for adiabatic wall
               this%u(dim + 2,   nx + i, 1:ny) = this%u(dim + 2, nx - (i - 1), 1:ny) 
           end do

           do j = 1, this%n_ghost_cells           
               this%u(1, 1:nx, -(j - 1)) =  this%u(1, 1:nx,  1 + (j - 1)) 
               this%u(2, 1:nx, -(j - 1)) = -this%u(2, 1:nx,  1 + (j - 1)) 
               this%u(3, 1:nx, -(j - 1)) = -this%u(3, 1:nx,  1 + (j - 1)) 
               this%u(dim + 2, 1:nx, -(j - 1)) = this%u(dim + 2, 1:nx, 1  + (j - 1)) 

               !! top wall
               this%u(1, 1:nx,   ny + j) =  this%u(1, 1:nx, ny - (j - 1)) 
               this%u(2, 1:nx,   ny + j) = -this%u(2, 1:nx, ny - (j - 1)) 
               this%u(3, 1:nx,   ny + j) = -this%u(3, 1:nx, ny - (j - 1)) 
               this%u(dim + 2, 1:nx,   ny + j) = this%u(dim + 2, 1:nx, ny - (j - 1)) 
           end do

           !! Corner points(required for cross derivatives)           
           do i = 1, n_g_cell
               do j = 1, n_g_cell
                   this%u(:, -(i - 1), -(j - 1)) = this%u(:, 1 + (i - 1), -(j - 1))   
                   this%u(:, -(i - 1),   ny + j) = this%u(:, 1 + (i - 1),   ny + j)   
                   
                   this%u(:,   nx + i, -(j - 1)) = this%u(:, nx - (i - 1), -(j - 1))   
                   this%u(:,   nx + i,   ny + j) = this%u(:, nx - (i - 1),   ny + j)   

               end do
           end do
           !!!!!!!!!!!!!!!


       end subroutine set_vst


       !> Set bc for Lid driven cavity 
       subroutine init_ldc(this) 
           use input_data
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny, n_g_cell

           integer(c_int) :: i, j

           real(c_double) :: rho, u, v, p, x, y
           real(c_double) :: T, M, a
           real(c_double) :: v_sq 

           nx = this%ncells_x
           ny = this%ncells_y

           T   = T_inf
           rho = rho_inf

           u   = zero 
           v   = zero
           v_sq= u**2 + v**2

           p = rho*R_gas*T

                   
           this%u(      1, :, :) = rho                            !rho
           this%u(      2, :, :) = rho*u                          !rho*u

           this%u(      3, :, :) = rho*v                          !rho*v
           this%u(dim + 2, :, :) = p/(gamma - 1) + half*rho*v_sq  !E_t
           

       end subroutine init_ldc




       !> Set bc for Lid driven cavity 
       subroutine set_ldc(this) 
           use input_data
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny, n_g_cell

           integer(c_int) :: i, j

           real(c_double) :: rho, u, v, p, x, y, vel
           real(c_double) :: v_sq, a_inf, v_fac, T, cv_gas

           nx = this%ncells_x
           ny = this%ncells_y

           n_g_cell = this%n_ghost_cells

           do i = 1, this%n_ghost_cells        
               !! Rho is simply set equal
               this%u(1, -(i - 1), 1:ny) =  this%u(1, 1  + (i - 1), 1:ny) 
               !! Velocities are set as opposite
               this%u(2, -(i - 1), 1:ny) = -this%u(2, 1  + (i - 1), 1:ny) 
               this%u(3, -(i - 1), 1:ny) = -this%u(3, 1  + (i - 1), 1:ny) 
               !! Energy is set equal for adiabatic wall
               this%u(dim + 2, -(i - 1), 1:ny) = this%u(dim + 2,  1 + (i - 1), 1:ny) 

               !! Rho is simply set equal
               this%u(1,   nx + i, 1:ny) =  this%u(1, nx - (i - 1), 1:ny) 
               !! Velocities are set as opposite
               this%u(2,   nx + i, 1:ny) = -this%u(2, nx - (i - 1), 1:ny) 
               this%u(3,   nx + i, 1:ny) = -this%u(3, nx - (i - 1), 1:ny) 
               !! Energy is set equal for adiabatic wall
               this%u(dim + 2,   nx + i, 1:ny) = this%u(dim + 2, nx - (i - 1), 1:ny) 
           end do

           a_inf = sqrt(gamma*R_gas*T_inf)
           u_inf = M_inf*a_inf

           cv_gas = R_gas/(gamma - 1)

           do j = 1, this%n_ghost_cells           
               this%u(1, 1:nx, -(j - 1)) =  this%u(1, 1:nx,  1 + (j - 1)) 
               this%u(2, 1:nx, -(j - 1)) = -this%u(2, 1:nx,  1 + (j - 1)) 
               this%u(3, 1:nx, -(j - 1)) = -this%u(3, 1:nx,  1 + (j - 1)) 
               this%u(dim + 2, 1:nx, -(j - 1)) = this%u(dim + 2, 1:nx, 1  + (j - 1)) 

               !! Top wall
               this%u(1, 1:nx,   ny + j) =  this%u(1, 1:nx, ny - (j - 1)) 
               do i = 1, nx
                   x                     =  this%x_c(i, j)
                   rho                   =  this%u(1, i, ny - (j - 1)) 
                   u                     =  this%u(2, i, ny - (j - 1))/rho
                   this%u(2, i,   ny + j)=  rho*(-u + two*u_inf) 
               end do

               this%u(3, 1:nx,   ny + j)    =  -this%u(3, 1:nx, ny - (j - 1))

               !! T taken as same 
               do i = 1, nx
                   rho                   =  this%u(1, i, ny - (j - 1)) 
                   u                     =  this%u(2, i, ny - (j - 1))/rho
                   v                     =  this%u(3, i, ny - (j - 1))/rho
                   v_sq                  =  u*u + v*v
                   T                     =  (this%u(dim + 2, i, ny - (j - 1))/rho - half*v_sq)/cv_gas
                   u                     = -u + two*u_inf 
                   v_sq                  =  u*u + v*v
                   this%u(dim + 2, i, ny + j) = rho*(cv_gas*T + half*v_sq)
               end do

           end do
                   

           !! Corner points(required for cross derivatives)           
           do i = 1, n_g_cell
               do j = 1, n_g_cell
                   this%u(:, -(i - 1), -(j - 1)) = half*(this%u(:, 1 + (i - 1), -(j - 1))   + &
                                                   this%u(:, -(i - 1), 1 + (j - 1))   )
                   this%u(:, -(i - 1),   ny + j) = half*(this%u(:, 1 + (i - 1),   ny + j)  + & 
                                                   this%u(:, -(i - 1), ny - (j - 1))   )
                   this%u(:,   nx + i, -(j - 1)) = half*(this%u(:, nx - (i - 1), -(j - 1))   + &
                                                   this%u(:, nx + i , 1 + (j - 1))   )
                   this%u(:,   nx + i,   ny + j) = half*(this%u(:, nx - (i - 1),   ny + j)   + &
                                                   this%u(:, nx + i , ny - (j - 1))   )
               end do
           end do
           !!!!!!!!!!!!!!!

       end subroutine set_ldc


       !> Extrapolate boundary conditions
       !! It approximates zero gradient
       subroutine extrapolate(this) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny, i, j 
           integer(c_int) :: n_g_cell 

           real(c_double) :: rho11, u11, v11, p11
           real(c_double) :: rho12, u12, v12, p12

           real(c_double) :: rho21, u21, v21, p21
           real(c_double) :: rho22, u22, v22, p22

           real(c_double) :: rho, u, v, p, x, y
           real(c_double) :: v_sq 

           nx = this%ncells_x
           ny = this%ncells_y

           n_g_cell = this%n_ghost_cells

           do i = 1, n_g_cell
               do j = 1, this%ncells_y 
                   this%u(1, -(i - 1), j)        = this%u(1, 1 + (i - 1), j)
                   this%u(2, -(i - 1), j)        = this%u(2, 1 + (i - 1), j)
                   this%u(3, -(i - 1), j)        = this%u(3, 1 + (i - 1), j)
                   this%u(dim + 2, -(i - 1), j)  = this%u(dim + 2, i + (i - 1), j)
               end do
           end do

           do i = 1,  n_g_cell
               do j = 1, this%ncells_y 
                   this%u(1, nx + i, j)       = this%u(1, nx - (i - 1), j)
                   this%u(2, nx + i, j)       = this%u(2, nx - (i - 1), j)
                   this%u(3, nx + i, j)       = this%u(3, nx - (i - 1), j)
                   this%u(dim + 2, nx + i, j) = this%u(dim + 2, nx - (i - 1), j)
               end do
           end do

           do j = 1,  n_g_cell
               do i = 1, this%ncells_x 
                   this%u(1,       i, -(j - 1))  =  this%u(1,      i, 1 + (j - 1))
                   this%u(2,       i, -(j - 1))  =  this%u(2,      i, 1 + (j - 1))      
                   this%u(3,       i, -(j - 1))  =  this%u(3,      i, 1 + (j - 1))      
                   this%u(dim + 2, i, -(j - 1)) =  this%u(dim + 2, i, 1 + (j - 1)) 
               end do
           end do

           do j = 1,  n_g_cell
               do i = 1, this%ncells_x 
                   this%u(1,       i, ny + j) = this%u(1,       i, ny - (j - 1))
                   this%u(2,       i, ny + j) = this%u(2,       i, ny - (j - 1))      
                   this%u(3,       i, ny + j) = this%u(3,       i, ny - (j - 1))      
                   this%u(dim + 2, i, ny + j) = this%u(dim + 2, i, ny - (j - 1)) 
               end do
           end do

           !! Corner points(required for cross derivatives)           
           do i = 1, n_g_cell
               do j = 1, n_g_cell
                   this%u(:, -(i - 1), -(j - 1)) = this%u(:, 1 + (i - 1), -(j - 1))   
                   this%u(:, -(i - 1),   ny + j) = this%u(:, 1 + (i - 1),   ny + j)   
                   
                   this%u(:,   nx + i, -(j - 1)) = this%u(:, nx - (i - 1), -(j - 1))   
                   this%u(:,   nx + i,   ny + j) = this%u(:, nx - (i - 1),   ny + j)   

               end do
           end do
           !!!!!!!!!!!!!!!


       end subroutine extrapolate 


       !> Set periodic bcs
       subroutine set_periodic(this) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny 

           integer(c_int) :: i, j

           real(c_double) :: rho, u, v, p, x, y
           real(c_double) :: v_sq 

           nx = this%ncells_x
           ny = this%ncells_y

           do i = 1, this%n_ghost_cells           
               this%u(:, -(i - 1), :) = this%u(:, nx - (i - 1), :) 
               this%u(:,   nx + i, :) = this%u(:, i, :) 
           end do

           do j = 1, this%n_ghost_cells           
               this%u(:, :, -(j - 1)) = this%u(:, :, ny - (j - 1)) 
               this%u(:, :,   ny + j) = this%u(:, :, j) 
           end do

           !! Corner points(required for cross derivatives)           
           do i = 1, this%n_ghost_cells 
               do j = 1, this%n_ghost_cells
                   this%u(:, -(i - 1), -(j - 1)) = this%u(:, 1 + (i - 1), -(j - 1))   
                   this%u(:, -(i - 1),   ny + j) = this%u(:, 1 + (i - 1),   ny + j)   
                   
                   this%u(:,   nx + i, -(j - 1)) = this%u(:, nx - (i - 1), -(j - 1))   
                   this%u(:,   nx + i,   ny + j) = this%u(:, nx - (i - 1),   ny + j)   

               end do
           end do
           !!!!!!!!!!!!!!!


       end subroutine set_periodic 


       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !! End of  Initial and boundary conditions
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !! Calculation of viscous part 
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


       !! Get derivatives on right face using the control volume approach of Blazek
       subroutine get_right_face_derivative_1(this, i, j, cv_n, u, du) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int), intent(in ) :: i, j
           real(c_double), intent(in ) :: cv_n(dim) 
           !! Normal for control volume face. n=(1,0) right face deri, n=(0,1) top face deri 
           real(c_double), intent(in ) :: u(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 

           real(c_double), intent(out) :: du
           real(c_double) :: dx, dy 

           real(c_double) :: cv(2**dim)  !Control volume integral on each face
           real(c_double) :: ds  !Area normal 

           integer(c_int) :: iter 

           dx = this%x_c(i + 1, j) - this%x_c(i, j)
           dy = this%y_c(i, j + 1) - this%y_c(i, j)

           !! Calculate derivatives based on Blazek cell centered scheme
           ds  = (cv_n(1)*dy + cv_n(2)*zero)
           cv(1) =  u(i + 1, j)*ds

           ds  = (cv_n(1)*zero + cv_n(2)*dx)
           cv(2) =  fourth*(u(i + 1, j  + 1) + u(i, j + 1) &
                 +  u(i + 1, j) + u(i, j))*ds 
             
           ds  = (cv_n(1)*(-dy) + cv_n(2)*zero)
           cv(3) =  u(i    , j)*ds

           ds  = (cv_n(1)*zero + cv_n(2)*(-dx))
           cv(4) =  fourth*(u(i + 1, j) + u(i, j) &
                 +  u(i + 1, j - 1) + u(i, j - 1))*ds

           du = sum(cv(:))/(dx*dy)

       end subroutine get_right_face_derivative_1

       !! Get derivatives on top face using the control volume approach of Blazek
       subroutine get_top_face_derivative_1(this, i, j, cv_n, u, du) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int), intent(in ) :: i, j
           real(c_double), intent(in ) :: cv_n(dim) 
           !! Normal for control volume face. n=(1,0) right face deri, n=(0,1) top face deri 
           real(c_double), intent(in ) :: u(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 

           real(c_double), intent(out) :: du
           real(c_double) :: dx, dy 

           real(c_double) :: cv(2**dim)  !Control volume integral on each face
           real(c_double) :: ds  !Area normal 

           integer(c_int) :: iter 

           dx = this%x_c(i + 1, j) - this%x_c(i, j)
           dy = this%y_c(i, j + 1) - this%y_c(i, j)
           !! FIXME
           dx = dy


           ds = (cv_n(1)*zero + cv_n(2)*dy)
           cv(1) =  u(i, j + 1)*ds

           ds = (cv_n(1)*(-dy) + cv_n(2)*zero)
           cv(2) =  fourth*(u(i, j) + u(i, j + 1) &
                 +  u(i - 1, j + 1) + u(i - 1, j))*ds

           ds = (cv_n(1)*zero + cv_n(2)*(-dx))
           cv(3) =  u(i, j    )*ds

           ds = (cv_n(1)*dy + cv_n(2)*zero)
           cv(4) =  fourth*(u(i + 1, j) + u(i + 1, j + 1) &
                 +  u(i, j + 1) + u(i, j))*ds

           du = sum(cv(:))/(dx*dy)

       end subroutine get_top_face_derivative_1


       !! Call top face and right face derivatives
       subroutine get_face_derivative_1(this, i, j, u, x_du_dx, x_du_dy, y_du_dx, y_du_dy) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int), intent(in ) :: i, j
           real(c_double), intent(in ) :: u(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 

           real(c_double), intent(out) :: x_du_dx, x_du_dy, y_du_dx, y_du_dy
           real(c_double) :: dx, dy 

           real(c_double) :: cv(2**dim)  !Control volume integral on each face
           real(c_double) :: normal(dim) !Normal on each face
           real(c_double) :: cv_n(dim) 
           !! Normal for control volume face. n=(1,0) du/dx, n=(0,1) du/dy 

           dx = this%x_c(i + 1, j) - this%x_c(i, j)
           dy = this%y_c(i, j + 1) - this%y_c(i, j)
           !! FIXME
           dx = dy


           !! Calculate derivatives based on Blazek cell centered scheme
           cv_n(1)   = one; cv_n(2)   = zero !get du/dx on right face
           call this%get_right_face_derivative_1(i, j, cv_n, u, x_du_dx)

           cv_n(1)   = zero;cv_n(2)   = one  !get du/dy on right face
           call this%get_right_face_derivative_1(i, j, cv_n, u, x_du_dy)

           cv_n(1)   = one; cv_n(2)   = zero !get du/dx on top face
           call this%get_top_face_derivative_1(i, j, cv_n, u, y_du_dx)

           cv_n(1) = zero;cv_n(2) = one !get du/dy on top face
           call this%get_top_face_derivative_1(i, j, cv_n, u, y_du_dy)

       end subroutine get_face_derivative_1 



       subroutine get_vis_rhs(this) 
           use input_data
           class(soln2d), intent(inout) :: this
           
           integer(c_int) :: nx, ny, i, j, n_g_cell

           real(c_double) :: dx, dy, x, y
           real(c_double) :: x_du_dx, x_du_dy, y_du_dx, y_du_dy
           !! du/dx, du/dy on right face and top face
           real(c_double) :: rho_x, rhoU_x, rhoV_x, E_x
           real(c_double) :: u_x, v_x, T_x 
           real(c_double) :: rho_y, rhoU_y, rhoV_y, E_y
           real(c_double) :: u_y, v_y, T_y 
           real(c_double) :: div 
           real(c_double) :: rho, u, v, p, T, E
           real(c_double) :: v_sq, mu, cv_gas, Pr 

           real(c_double) :: rho_cell(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 
           real(c_double) :: u_cell(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 
           real(c_double) :: v_cell(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 
           real(c_double) :: T_cell(1-this%n_ghost_cells:this%ncells_x+this%n_ghost_cells,&
               1-this%n_ghost_cells:this%ncells_y+this%n_ghost_cells) 

           real(c_double) :: tau11, tau12, tau21, tau22 

           real(c_double) :: cv(2**dim)  !Control volume integral on each face
           real(c_double) :: normal(dim) !Normal on each face
           real(c_double) :: ds_x, ds_y  !Area normal in each direction
           real(c_double) :: f_v_p_half(dim + 2, 0:this%ncells_x, 0:this%ncells_y) !x flux at i+1/2, j+1/2
           real(c_double) :: g_v_p_half(dim + 2, 0:this%ncells_x, 0:this%ncells_y) !y flux at i+1/2, j+1/2
           real(c_double) :: f_v_r(dim + 2), f_v_l(dim + 2), g_v_r(dim + 2), g_v_l(dim + 2)

           real(c_double) :: temp1, temp2 

           nx = this%ncells_x
           ny = this%ncells_x

           cv_gas = r_gas/(gamma - 1)
           mu     = mu_inf
           Pr     = Pr_inf

           !! Store u, v, T to get derivatives
           do i = 1 - this%n_ghost_cells, nx + this%n_ghost_cells
               do j = 1- this%n_ghost_cells, ny + this%n_ghost_cells
                   u_cell(i, j) = this%u(2, i, j)/this%u(1, i, j)
                   v_cell(i, j) = this%u(3, i, j)/this%u(1, i, j)
                   v_sq         = u_cell(i, j)**2 + v_cell(i, j)**2
                   T_cell(i, j) = ((this%u(4, i, j)/this%u(1, i, j)) - half*v_sq)/cv_gas
               end do
           end do

           do i = 0, nx
               do j = 0, ny
                   dx = this%x_c(i + 1, j) - this%x_c(i, j)
                   dy = this%y_c(i, j + 1) - this%y_c(i, j)

                   x = half*(this%x_c(i + 1, j) + this%x_c(i, j))
                   y = half*(this%y_c(i, j + 1) + this%y_c(i, j))

                   !! x flux
                   !! Average to get values at face
                   rho = half*(this%u(1, i, j) + this%u(1, i + 1, j))
                   u   = half*(this%u(2, i, j) + this%u(2, i + 1, j))/rho
                   v   = half*(this%u(3, i, j) + this%u(3, i + 1, j))/rho
                   E   = half*(this%u(dim + 2, i, j) + this%u(dim + 2, i + 1, j))/rho

                   v_sq= u**2 + v**2
                   T   = (E/rho - half*v_sq)/cv_gas

                   call this%get_face_derivative_1(i, j, u_cell, x_du_dx, x_du_dy, y_du_dx, y_du_dy)
                   u_x    = x_du_dx
                   u_y    = x_du_dy
                   call this%get_face_derivative_1(i, j, v_cell, x_du_dx, x_du_dy, y_du_dx, y_du_dy)
                   v_x    = x_du_dx
                   v_y    = x_du_dy

                   call this%get_face_derivative_1(i, j, T_cell, x_du_dx, x_du_dy, y_du_dx, y_du_dy)
                   T_x    = x_du_dx

                   div    = u_x + v_y
                   tau11  = mu*(two*u_x - two*third*div)
                   tau12  = mu*(u_y + v_x)

                   f_v_p_half(1,       i, j) = zero 
                   f_v_p_half(2,       i, j) = tau11 
                   f_v_p_half(3,       i, j) = tau12 
                   f_v_p_half(dim + 2, i, j) = u*tau11 + v*tau12 + mu*Pr*T_x 

                   !! y flux
                   !! Average to get values at face
                   rho = half*(this%u(1,       i, j) + this%u(1,       i, j + 1))
                   u   = half*(this%u(2,       i, j) + this%u(2,       i, j + 1))/rho
                   v   = half*(this%u(3,       i, j) + this%u(3,       i, j + 1))/rho
                   E   = half*(this%u(dim + 2, i, j) + this%u(dim + 2, i, j + 1))/rho

                   v_sq= u**2 + v**2
                   T   = (E/rho - half*v_sq)/cv_gas


                   call this%get_face_derivative_1(i, j, u_cell, x_du_dx, x_du_dy, y_du_dx, y_du_dy)
                   u_x    = y_du_dx
                   u_y    = y_du_dy
                   call this%get_face_derivative_1(i, j, v_cell, x_du_dx, x_du_dy, y_du_dx, y_du_dy)
                   v_x    = y_du_dx
                   v_y    = y_du_dy

                   call this%get_face_derivative_1(i, j, T_cell, x_du_dx, x_du_dy, y_du_dx, y_du_dy)
                   T_y    = y_du_dy

                   div    = u_x + v_y
                   tau21  = mu*(u_y + v_x)
                   tau22  = mu*(two*v_y - two*third*div)

                   g_v_p_half(1,       i, j) = zero 
                   g_v_p_half(2,       i, j) = tau21 
                   g_v_p_half(3,       i, j) = tau22 
                   g_v_p_half(dim + 2, i, j) = u*tau21 + v*tau22 + mu*Pr*T_y 


               end do
           end do


           do i = 1, nx
               do j = 1, ny
                   dx = this%dx(i, j)
                   dy = this%dy(i, j)
                   f_v_r = f_v_p_half(:, i,     j) 
                   f_v_l = f_v_p_half(:, i - 1, j) 
                   g_v_r = g_v_p_half(:, i,     j) 
                   g_v_l = g_v_p_half(:, i, j - 1) 

                   this%vis_rhs(:, i, j) = ((f_v_r - f_v_l)/dx &
                                     + (g_v_r - g_v_l)/dy )
               end do
           end do

       end subroutine get_vis_rhs

       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !! End of calculation of viscous part 
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !! Inviscid part starts here
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       !> Create 3rd order WENO flux
       subroutine get_weno_reconstructed_flux(this, i, j, n, f_mark, u_vec) 

           class(soln2d), intent(inout) :: this

           integer(c_int), intent(in)  :: i, j 
           !! Subscript for cell where reconstruction is required
           real(c_double), intent(out) :: n(dim) 
           !! Normal vector
           character, intent(in)       :: f_mark 
           !! Specify left or right face 

           real(c_double), intent(out) :: u_vec(dim+2) !x Flux for Euler 

           integer(c_int) :: sten_mark !To use subscript Q = half(q_{i} + q_{i+sten_mark})
           integer(c_int) :: n1, n2 
           !! To select i, or j direction using normals

           !!!!WENO variables
           real(c_double) :: u_weno_1(dim+2), u_weno_2(dim+2)
           real(c_double) :: d(2), beta(dim + 2, 2), alpha(dim + 2, 2)
           real(c_double) :: summ, eps!tolerance 

           real(c_double) :: temp_u_1(dim+2), temp_u_2(dim+2), temp_u_3(dim+2)

           integer(c_int) :: iter 

           call assert( ((f_mark .eq. 'l') .or. (f_mark .eq. 'r')))

           if (f_mark .eq. 'l') then
               sten_mark = -one
           else
               sten_mark = one
           end if
           n1 = n(1); n2 = n(2)

           temp_u_1 = this%u(:, i            , j)
           temp_u_2 = this%u(:, i + sten_mark*n1, j + sten_mark*n2)

           u_weno_1 = half*(temp_u_1 + temp_u_2)

           temp_u_3 = this%u(:, i - sten_mark*n1, j - sten_mark*n2)

           u_weno_2 = three*half*temp_u_1 - half*temp_u_3

           d(1) = two*third; d(2) = third

           eps  = 10e-6*one

           do iter = 1, dim + 2
               beta(iter, 1)  =  (temp_u_2(iter) - temp_u_1(iter))**2
               alpha(iter, 1) =  d(1)/(eps + beta(iter, 1))**2

               beta(iter, 2)  =  (temp_u_3(iter) - temp_u_1(iter))**2
               alpha(iter, 2) =  d(2)/(eps + beta(iter, 2))**2
           end do
           do iter = 1, dim + 2
               summ        = sum(alpha(iter, :))
               alpha(iter, :) = alpha(iter, :)/summ
           end do

           do iter = 1, dim + 2
               u_vec(iter) = alpha(iter, 1)*u_weno_1(iter) + alpha(iter, 2)*u_weno_2(iter)
           end do
       end subroutine get_weno_reconstructed_flux 



       !! Get Euler fluxes
       subroutine get_inv_flux(u_vec, f_vec, g_vec, u, v, a) 
           real(c_double), intent(in)  :: u_vec(dim+2) !Euler Conserved variables
           real(c_double), intent(out) :: f_vec(dim+2) !x Flux for Euler 
           real(c_double), intent(out) :: g_vec(dim+2) !x Flux for Euler 
           real(c_double), intent(out) :: u, v, a  !Velocities and speed of sound 

           real(c_double) :: rho, p, v_sq

           rho = u_vec(1)
           u   = u_vec(2)/rho
           v   = u_vec(3)/rho

           v_sq = u**2 + v**2 

           p   = (gamma - 1)*(u_vec(dim + 2) - half*rho*v_sq )

           a = sqrt(gamma * p/rho)

           f_vec(1) = u_vec(2)         ! f(1) = rho*u 
           f_vec(2) = u_vec(2)*u + p   ! f(2) = rho*u*u + p 
           f_vec(3) = u_vec(2)*v       ! f(2) = rho*u*v
           f_vec(dim + 2) = (u_vec(dim + 2) + p)*u ! f(4) = (E_t + p)*u 

           g_vec(1) = u_vec(3)         ! f(1) = rho*u 
           g_vec(2) = u_vec(3)*u       ! f(2) = rho*u*u + p 
           g_vec(3) = u_vec(3)*v + p   ! f(2) = rho*u*v
           g_vec(dim + 2) = (u_vec(dim + 2) + p)*v ! f(4) = (E_t + p)*u 

       end subroutine get_inv_flux 

       !! Get Lax Friedrichs flux
       subroutine get_lax_f_flux(this, u_vec, up1_vec, f_vec, fp1_vec, s_max, flux_vec) 
           class(soln2d), intent(inout) :: this
           real(c_double), intent(in)  :: u_vec(dim+2) !Euler Conserved variables
           real(c_double), intent(in)  :: up1_vec(dim+2) !Euler Conserved variables

           real(c_double), intent(in) :: f_vec(dim+2) ! Flux for Euler 
           real(c_double), intent(in) :: fp1_vec(dim+2) ! Flux for Euler 

           real(c_double), intent(in) :: s_max 

           real(c_double), intent(out) :: flux_vec(dim+2) 

           flux_vec = half*(f_vec + fp1_vec - s_max*(up1_vec - u_vec))
!           flux_vec = half*(f_vec + fp1_vec - this%global_s_max*(up1_vec - u_vec))

       end subroutine get_lax_f_flux




       !> Get flux at face (i + 1/2) and (j + 1/2)
       subroutine get_flux_at_face(this, i, j, s_max_inv, f_inv, g_inv) 
           class(soln2d), intent(inout) :: this
           
           integer(c_int), intent(in)  :: i, j
           real(c_double), intent(out) :: f_inv(:), g_inv(:) 
           real(c_double), intent(out) :: s_max_inv 

           integer(c_int) :: nx, ny

           real(c_double) :: rho, p, x, y
           real(c_double) :: v_sq 

           real(c_double) :: normal(dim) 
           !! Normal 

           real(c_double) :: u, up1 !x velocity
           real(c_double) :: v, vp1 !y velocity
           real(c_double) :: a, ap1 !speed of sound

           real(c_double) :: u_vec(dim + 2)  
           real(c_double) :: up1_vec(dim + 2)
           real(c_double) :: um1_vec(dim + 2)

           real(c_double) :: v_vec(dim + 2)  
           real(c_double) :: vp1_vec(dim + 2)

           real(c_double) :: s_max_f, s_max_g !Max characteristic

           real(c_double) :: f_vec(dim+2) !x Flux for Euler 
           real(c_double) :: fp1_vec(dim+2) !x Flux for Euler for cell plus 1 

           real(c_double) :: g_vec(dim+2) !x Flux for Euler 
           real(c_double) :: gp1_vec(dim+2) !x Flux for Euler for cell plus 1 

           character :: f_mark 

           nx = this%ncells_x
           ny = this%ncells_y

           !! x flux at right
           normal(1) = one; normal(2) = zero !x - direction
           
           f_mark = 'r' !On the present cell I want flux at right edge
           call get_weno_reconstructed_flux(this, i, j, normal, f_mark, u_vec) 
           call get_inv_flux(u_vec, f_vec, g_vec, u, v, a)
     
           f_mark = 'l'
           call get_weno_reconstructed_flux(this, i + 1, j, normal, f_mark, up1_vec) 
           call get_inv_flux(up1_vec, fp1_vec, gp1_vec, up1, vp1, ap1)
           s_max_f   = max(abs(u) + a, abs(up1) + ap1)

           call this%get_lax_f_flux(u_vec, up1_vec, f_vec, fp1_vec, s_max_f, f_inv)

           !! y flux at top 
           normal(1) = zero; normal(2) = one !y - direction

           f_mark = 'r' !On the present cell I want flux at right edge(top)
           call get_weno_reconstructed_flux(this, i, j, normal, f_mark, v_vec) 
           call get_inv_flux(v_vec, f_vec, g_vec, u, v, a)

           f_mark = 'l' 
           call get_weno_reconstructed_flux(this, i, j + 1, normal, f_mark, vp1_vec) 
           call get_inv_flux(vp1_vec, fp1_vec, gp1_vec, up1, vp1, ap1)
           s_max_g   = max(abs(v) + a, abs(vp1) + ap1)
     
           call this%get_lax_f_flux(v_vec, vp1_vec, g_vec, gp1_vec, s_max_g, g_inv)

           s_max_inv = max(s_max_f, s_max_g)

           rho = v_vec(1)
           u   = v_vec(2)/rho
           v   = v_vec(3)/rho

           v_sq = u**2 + v**2 

           p   = (gamma - 1)*(v_vec(dim + 2) - half*rho*v_sq )

           a = sqrt(gamma * p/rho)

       end subroutine get_flux_at_face 




       !> Solve for Euler using Lax Friedrichs 
       !! @param global_s_max: gives maximum characteristic vel over all elements
       subroutine get_inv_rhs(this) 
           class(soln2d), intent(inout) :: this
           
           real(c_double) :: x, y, dx, dy
           real(c_double) :: s_max !Max characteristic

           real(c_double) :: f_vec_l(dim+2) !x Flux at left edge 
           real(c_double) :: f_vec_r(dim+2) !x Flux at right edge 

           real(c_double) :: g_vec_l(dim+2) !x Flux at left edge 
           real(c_double) :: g_vec_r(dim+2) !x Flux at right edge 

           real(c_double) :: f_v_p_half(dim + 2, 0:this%ncells_x, 0:this%ncells_y) !x flux at i+1/2, j+1/2
           real(c_double) :: g_v_p_half(dim + 2, 0:this%ncells_x, 0:this%ncells_y) !y flux at i+1/2, j+1/2

           integer(c_int) :: i, j, nx, ny

           nx = this%ncells_x
           ny = this%ncells_y

           this%global_s_max = zero
           do j = 0, this%ncells_y 
               do i = 0, this%ncells_x 
                   x   = this%x_c(i, j)
                   y   = this%y_c(i, j)

                   call get_flux_at_face(this, i, j, s_max, f_v_p_half(:, i, j), g_v_p_half(:, i, j)) 
                   this%global_s_max = max(s_max, this%global_s_max) 
               end do
           end do

           do j = 1, this%ncells_y
               do i = 1, this%ncells_x 

                   f_vec_r = f_v_p_half(:, i,     j    )
                   f_vec_l = f_v_p_half(:, i - 1, j    )

                   g_vec_r = g_v_p_half(:, i    , j    )
                   g_vec_l = g_v_p_half(:, i    , j - 1)

                   this%inv_rhs(:, i, j) = -((f_vec_r - f_vec_l)/this%dx(i, j) &
                                     + (g_vec_r - g_vec_l)/this%dy(i, j))

               end do
           end do


       end subroutine get_inv_rhs

       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !! End of inviscid part
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       !> Do euler time stepping 
       subroutine cns_solve(this) 
           class(soln2d), intent(inout) :: this

           call this%get_inv_rhs
           call this%get_vis_rhs

           this%rhs = this%inv_rhs + this%vis_rhs
   
       end subroutine cns_solve 




       !> Setup solution vector. Cell centerd and solution array
       !! @param nx, ny: number of cells in x and y
       !! @param x, y : grid x and y co-ordinates
       subroutine setup_sol_vec(this, nx, ny, x, y) 
           class(soln2d), intent(inout) :: this

           integer(c_int), intent(in) :: nx, ny

           real(c_double), intent(in) :: x(:, :)
           real(c_double), intent(in) :: y(:, :)
       
           integer(c_int) :: i, j 

           integer(c_int) :: n_g_cell !Number of ghost cells 

           integer(c_int) :: allo_stat 
           character(256) :: err_msg

           call assert(nx .gt. 1)

           this%ncells_x = nx  
           this%ncells_y = ny 

           n_g_cell = this%n_ghost_cells

           call assert(this%n_ghost_cells .gt. zero)

           allocate(this%x_c(1-n_g_cell:nx+n_g_cell, 1-n_g_cell:ny+n_g_cell), &
                    stat=allo_stat, errmsg=err_msg)  !Two additional ghost cells in each direction
           call assert(allo_stat .eq. 0, err_msg)
           allocate(this%y_c(1-n_g_cell:nx+n_g_cell, 1-n_g_cell:ny+n_g_cell), &
                    stat=allo_stat, errmsg=err_msg)  !Two additional ghost cells in each direction
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%dx(nx, ny), stat=allo_stat, errmsg=err_msg)  
           call assert(allo_stat .eq. 0, err_msg)
           allocate(this%dy(nx, ny), stat=allo_stat, errmsg=err_msg)  
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%u(dim + 2, 1-n_g_cell:nx+n_g_cell, 1-n_g_cell:ny+n_g_cell), &
                    stat=allo_stat, errmsg=err_msg) !For Euler 2D
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%inv_rhs(dim + 2, nx, ny), stat=allo_stat, errmsg=err_msg) !For Euler 2D
           call assert(allo_stat .eq. 0, err_msg)
           allocate(this%vis_rhs(dim + 2, nx, ny), stat=allo_stat, errmsg=err_msg) !For Euler 2D
           call assert(allo_stat .eq. 0, err_msg)

           allocate(this%rhs(dim + 2, nx, ny), stat=allo_stat, errmsg=err_msg) !For Euler 2D
           call assert(allo_stat .eq. 0, err_msg)

           this%x_c = zero
           this%y_c = zero

           do i = 1, nx
               do j = 1, ny
                   this%x_c(i, j) = half*(x(i, j) + x(i + 1, j))
                   this%y_c(i, j) = half*(y(i, j) + y(i, j + 1))

                   this%dx(i, j) = (x(i + 1, j) - x(i, j))
                   this%dy(i, j) = (y(i, j + 1) - y(i, j))
               end do
           end do

           do i = 0, 1 - n_g_cell, -1
               this%x_c(i, :)    = x(1, 1) - (-one*i + half)*(x(1 + 1, 1) - x(1, 1))
               this%y_c(i, 1:ny) = half*(y(1, 1:ny) + y(1, 2:ny+1))
           end do

           do i = nx + 1, nx + n_g_cell
               this%x_c(i, :)    = x(nx + 1, 1) + (i - nx - 1 + half)*(x(nx + 1, 1) - x(nx, 1))
               this%y_c(i, 1:ny) = half*(y(nx, 1:ny) + y(nx, 2:ny+1))
           end do

           do j = 0, 1 - n_g_cell, -1
               this%x_c(1:nx, j) = half*(x(1:nx, 1) + x(2:nx+1, 1))
               this%y_c(:, j)    = y(1, 1) - (-one*j + half)*(y(1, 1 + 1) - y(1, 1))
           end do

           do j = ny + 1, ny + n_g_cell 
               this%x_c(1:nx, j) = half*(x(1:nx, ny) + x(2:nx+1, ny))
               this%y_c(:, j)    = y(1, ny + 1) + (j - ny - 1 + half)*(y(1, ny + 1) - y(1, ny))
           end do

           !! Corner points
           this%x_c(0,      0) = this%x_c(1,  0) - (this%x_c(2,  0) - this%x_c(1,      0))
           this%y_c(0,      0) = this%y_c(0,  1) - (this%y_c(0,  2) - this%y_c(0,      1))
           this%x_c(0, ny + 1) = this%x_c(1, ny + 1) - (this%x_c(2, ny + 1) - this%x_c(1, ny + 1))
           this%y_c(0, ny + 1) = this%y_c(0,     ny) + (this%y_c(0,     ny) - this%y_c(0, ny - 1))

           this%x_c(nx + 1, 0) = this%x_c(nx,  0) + (this%x_c(nx,  0) - this%x_c(nx - 1,      0))
           this%y_c(nx + 1, 0) = this%y_c(nx + 1, 1) - (this%y_c(nx + 1, 2) - this%y_c(nx + 1, 1))
           this%x_c(nx + 1, ny + 1) = this%x_c(nx, ny + 1) + (this%x_c(nx, ny + 1) - this%x_c(nx - 1, ny + 1))
           this%y_c(nx + 1, ny + 1) = this%y_c(nx + 1, ny) + (this%y_c(nx + 1, ny) - this%y_c(nx + 1, ny - 1))


       end subroutine setup_sol_vec


       subroutine destructor(this) 
           type(soln2d) :: this

           if (allocated(this%u)) deallocate(this%u)
           if (allocated(this%rhs)) deallocate(this%rhs)
           if (allocated(this%inv_rhs)) deallocate(this%inv_rhs)
           if (allocated(this%vis_rhs)) deallocate(this%vis_rhs)

           if (allocated(this%x_c)) deallocate(this%x_c)
           if (allocated(this%y_c)) deallocate(this%y_c)

           if (allocated(this%dx)) deallocate(this%dx)
           if (allocated(this%dy)) deallocate(this%dy)

       end subroutine destructor 

  end module solution





  MODULE subroutines
    use types_vars
    use grid
    use solution

    implicit none

    CONTAINS


    !> Subroutine to plot data 
    SUBROUTINE plot_data(sol, filename, steps)!, rho_ex)
      use input_data
      class(soln2d), intent(in)   :: sol

      character(len=256), intent(in)   :: filename
      integer(c_int),     intent(in)   :: steps

      real(c_double) :: x, y
      real(c_double) :: rho, p, u, v, v_sq
      integer(c_int) :: nx, ny
      integer(c_int) :: i, j

      character(len=256) :: soln_file, plt_indx 
      character(len=256) :: base_file, res1

      write (res1, '(i0.2)') int(Re_inf) 
      base_file = trim(filename)//"_Re_"//trim(res1)

      write (plt_indx, '(i9.9)') steps 

      soln_file = trim(base_file)//"_"//trim(plt_indx)//".tec"

      nx = sol%ncells_x
      ny = sol%ncells_y

      open(10,file=soln_file,status='replace')
      write(10,*) 'VARIABLES = "x", "y", "u", "v", "rho", "p", "rhs1", "rhs2", "rhs3", "rhs4"'
      write(10,*) 'ZONE I=',nx,', J=',ny, ', F=POINT'

      do i = 1, nx
          do j = 1, ny
              x   = sol%x_c(i, j)
              y   = sol%y_c(i, j)

              rho = sol%u(1, i, j)

              u   = sol%u(2, i, j)/rho
              v   = sol%u(3, i, j)/rho

              v_sq = u**2 + v**2

              p   = (gamma - 1)*(sol%u(dim + 2, i, j) - half*rho*v_sq )

              if (abs(u) .lt. epsilon(one)) u = zero
              if (abs(v) .lt. epsilon(one)) v = zero
              if (abs(v_sq) .lt. epsilon(one)) v_sq = zero
              if (abs(rho) .lt. epsilon(one)) rho = zero
              if (abs(p) .lt. epsilon(one)) p = zero
              write(10, 100) x, y, u, v, rho, p, sol%rhs(1, i, j), sol%rhs(2, i, j), sol%rhs(3, i, j), sol%rhs(4, i, j) 
100           FORMAT (e17.10, 2x, e17.10, 2x, e17.10, 2x, e17.10, 2x, e17.10, 2x, e17.10)              
          end do
      end do

      close(10)


      soln_file = trim(base_file)//"_uv_"//trim(plt_indx)//".dat"

      nx = sol%ncells_x
      ny = sol%ncells_y

      open(20,file=soln_file,status='replace')

      j = ny/2 + 1
      do i = 1, nx
         !! Write velocities at the center line
         !! First we write v at y = 0.5
         x   = sol%x_c(i, j)

         rho = sol%u(1, i, j)

         v   = sol%u(3, i, j)/rho

         if (mod(ny, 2) .eq. 0) then         
             rho = sol%u(1, i, j - 1)

             v   = half*(v + sol%u(3, i, j - 1)/rho)
         end if

         !! Now we write u at x = 0.5
         
         y   = sol%y_c(j, i)

         rho = sol%u(1, j, i)

         u   = sol%u(2, j, i)/rho

         if (mod(ny, 2) .eq. 0) then         
             rho = sol%u(1, j - 1, i)

             u   = half*(u + sol%u(2, j - 1, i)/rho)
         end if

         write(20, *) x, y, u, v

      end do

      close(20)


    END SUBROUTINE plot_data 


    !> Do euler time stepping 
    !! @param dt: time step 
    subroutine euler_time(soln, dt) 
        class(soln2d), intent(inout)   :: soln 

        real(c_double), intent(in) :: dt 

        real(c_double) :: u(size(soln%u, 1), size(soln%u, 2), size(soln%u, 3))
        real(c_double) :: u_temp(size(soln%u, 1), size(soln%u, 2), size(soln%u, 3))

        integer(c_int) :: nx, ny 

        nx = soln%ncells_x
        ny = soln%ncells_y

        call soln%cns_solve
        soln%u(:, 1:nx, 1:ny) = soln%u(:, 1:nx, 1:ny) + dt*(soln%rhs)
    end subroutine euler_time 




    !> Do strong stability preserving time stepping    
    !! @param dt: time step 
    subroutine ssrk(soln, dt) 
        class(soln2d), intent(inout)   :: soln 

        real(c_double), intent(in) :: dt 

        real(c_double) :: u(size(soln%u, 1), size(soln%u, 2), size(soln%u, 3))
        real(c_double) :: u_temp(size(soln%u, 1), size(soln%u, 2), size(soln%u, 3))

        integer(c_int) :: nx, ny 

        nx = soln%ncells_x
        ny = soln%ncells_y

        u(:, 1:nx, 1:ny) = soln%u(:, 1:nx, 1:ny)

        call soln%cns_solve
        u_temp(:, 1:nx, 1:ny) = u(:, 1:nx, 1:ny) + dt*(soln%rhs)

        soln%u(:, 1:nx, 1:ny) = u_temp(:, 1:nx, 1:ny)
        call soln%cns_solve
        u_temp(:, 1:nx, 1:ny) = three*fourth*u(:, 1:nx, 1:ny) + fourth*u_temp(:, 1:nx, 1:ny) &
                          + fourth*dt*(soln%rhs)

        soln%u(:, 1:nx, 1:ny) = u_temp(:, 1:nx, 1:ny)
        call soln%cns_solve
        soln%u(:, 1:nx, 1:ny) = third*u(:, 1:nx, 1:ny) + two*third*u_temp(:, 1:nx, 1:ny) &
                          + two*third*dt*(soln%rhs)
    end subroutine ssrk



    !> Subroutine to solve euler equation 
    !! @param nptsx : number of points
    !! @param stopT : stopping time 
    subroutine euler_solver(mesh, cfl, stopT)
        use input_data
        class(mesh2d), intent(in)   :: mesh

        real(c_double), intent(in)  :: cfl 
        real(c_double), intent(in)  :: stopT 

        type(soln2d)  :: soln

        integer(c_int) :: i, nx, ny, steps, plot_n_step
        real(c_double) :: dx, dy, dt, t_coun
        real(c_double) :: plot_t, plot_dt 

        real(c_double), allocatable :: rho_ex(:, :) 

        character(len=256)   :: res1, res2 
        character(len=256)   :: filename

        soln%n_ghost_cells = two !Select number of ghost cells to be added at the end
        call soln%setup_sol_vec(mesh%ncells_x, mesh%ncells_y, mesh%x, mesh%y)

!        call soln%init_vst
        call soln%init_ldc
!        call soln%init_sol
!        call soln%init_sol_case3

        dx = minval(mesh%dx)
        dy = minval(mesh%dy)
        dt = cfl*min(dx, dy)/(4*u_inf)

        nx = soln%ncells_x
        ny = soln%ncells_y

        write (res1, '(i0.2)') soln%ncells_x  ! converting number to string using a 'internal file'
        write (res2, '(i0.2)') soln%ncells_y  ! converting number to string using a 'internal file'

        filename='ldc_weno_nx_'//trim(res1)//'_ny_'//trim(res2)

        plot_n_step = 750
!        plot_dt = 0.1

        steps = zero
        plot_t = plot_dt 
        t_coun = zero         
        do !i = 1,  2 
!            call soln%set_vst
            call soln%set_ldc
!            call soln%extrapolate
!            call soln%set_periodic

!            call euler_time(soln, dt)
            call ssrk(soln, dt)

            dt = cfl*min(dx, dy)/soln%global_s_max

            t_coun = t_coun + dt

            write(*, *) steps, t_coun, soln%global_s_max

!            if (((t_coun .le. plot_t) .and. (t_coun + dt .gt. plot_t)) .or. (t_coun > plot_t)) then
            if (mod(steps, plot_n_step) .eq. 0) then
!                call plot_data(soln, filename, int((t_coun + dt)/plot_dt))
                call plot_data(soln, filename, steps/plot_n_step)
!                plot_t = plot_t + plot_dt
            end if

            if (t_coun > stopT) exit 
            steps = steps + 1

            soln%pres_time = t_coun

        end do

    end subroutine euler_solver 

  END MODULE subroutines



  !> Main program to create grid and calculate error for derivative
  program template

    use types_vars
  
    use subroutines
    use grid

    use input_data
    use const_data
  
    implicit none  

    integer(c_int) :: nx, ny !Number of CELLS in each direction
    real(c_double) :: startX, stopX 
    real(c_double) :: startY, stopY 

    real(c_double) :: cfl, stopT 

    real(c_double) :: a_inf 

    type(mesh2d)   :: mesh

    ! 2D domain 
    startX = zero; stopX = one 
    startY = zero; stopY = one 

    ! Stop time
    stopT = ten + one 

    cfl = three*tenth 

    nx  = three*ten*ten + one
    ny  = three*ten*ten + one

    ! Echo print your input to make sure it is correct
    write(*,*) 'Your 2D domain is '
    write(*, *) 'X', startX, ' to ', stopX
    write(*, *) 'Y', startY, ' to ', stopY

    Re_inf =   400*one     !Reynolds number
    T_inf  =   one        !Temperature
!    T_inf  =   one/(gamma*R_gas)        !Temperature
    M_inf  = two*tenth    !Mach no.
!    M_inf  = one    !Mach no.

    a_inf  = sqrt(gamma*R_gas*T_inf)
    u_inf  = a_inf*M_inf

    mu_inf = u_inf/Re_inf  !Dynamic viscosity

    call mesh%get_grid2d(nx, ny, startX, stopX, startY, stopY)

    call assert(cfl .gt. tol)
    call euler_solver(mesh, cfl, stopT)

  end program template
