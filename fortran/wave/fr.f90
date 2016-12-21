  
  module class_fr 

      use types_vars
      use polynomial
      implicit none

      type :: fr
          integer(c_int) :: order
          integer(c_int) :: npts 
          integer(c_int) :: nele_x 

          real(c_double), allocatable :: lagr_deri(:, :)

          real(c_double), allocatable :: g_l(:)
          real(c_double), allocatable :: g_r(:)

          real(c_double), allocatable :: lagr_l(:)
          real(c_double), allocatable :: lagr_r(:)

          real(c_double), allocatable :: x_r(:, :)

          contains
              procedure, pass   :: print_glob_var 
              procedure, pass   :: init_operators 

              procedure, nopass :: get_interaction_flux
              procedure, pass   :: get_disc_flux_f 
              procedure, pass   :: get_discont_deriv

              procedure, pass   :: get_inter_flux
              procedure, pass   :: get_boundary_flux

              procedure, pass   :: get_derivative

              procedure, pass   :: get_first_deri
              procedure, pass   :: get_sec_deri

              !FIXME destructors implemented from gcc 4.9
!              final :: kill_all
              procedure, pass   :: kill_all

      end type fr

      contains

          subroutine kill_all(this) 
              class(fr), intent(inout) :: this

              deallocate(this%lagr_deri)
              
              deallocate(this%g_l)
              deallocate(this%g_r)

              deallocate(this%lagr_l)
              deallocate(this%lagr_r)

              deallocate(this%x_r)


          end subroutine kill_all 



          !> Allocate and get all operators for FR
          subroutine init_operators(this, order, npts, nele_x, x_r) 
              class(fr), intent(inout)   :: this

              integer(c_int), intent(in) :: order, npts, nele_x

              real(c_double), intent(in) :: x_r(:, :) 

              real(c_double) :: nodes(this%order + 1) 

              this%order  = order
              this%npts   = npts
              this%nele_x = nele_x 

              allocate(this%lagr_deri(this%npts, this%npts))

              allocate(this%g_l(this%npts))
              allocate(this%g_r(this%npts))

              allocate(this%lagr_l(this%npts))
              allocate(this%lagr_r(this%npts))

              allocate(this%x_r(this%npts, this%nele_x))

              this%x_r = x_r

              call gauss_nodes(this%order, nodes)

              call lagr_d_matrix(this%npts, nodes, this%lagr_deri)
              
              call left_radau_d(this%order,  this%npts, nodes, this%g_l)
              call right_radau_d(this%order, this%npts, nodes, this%g_r)

              call lagr_flux_matrix(this%npts, nodes, this%lagr_l, this%lagr_r)

          end subroutine init_operators 




          subroutine print_glob_var(this)
              class(fr), intent(inout)   :: this

              write(*, *) this%order, this%npts

          end subroutine print_glob_var


          !> Get interaction flux
          !! k = 0 gives fully upwind, k = 1 center
          !! @param f_l, f_r: Left and right flux
          !! @param f_I: Interaction or common flux
          subroutine get_interaction_flux(u_l, u_r, f_l, f_r, k, f_I)

              real(c_double), intent(in)     :: u_l, u_r 
              real(c_double), intent(in)     :: f_l, f_r, k 

              real(c_double), intent(out)    :: f_I

              real(c_double) :: a, temp

              if (abs(u_l - u_r) .gt. epsilon(one)) then
                  a = (f_l - f_r)/(u_l - u_r)
                  f_I = a*half*(u_l + u_r) 
                  f_I = f_I - abs(a)*(1-k)*half*(u_r - u_l)
              else
                  f_I = half*(f_l + f_r) - (1-k)*(f_r - f_l) 
              end if

          end subroutine get_interaction_flux



          !> Subroutine to get discont. flux at flux points 
          !! It assumes periodic boundary conditions for now
          !! @param nele_x: number of elements
          !! @param extrapol_mat: extraopolation matrix to flux points 
          !! @param f: f at solution points 
          !! @param f_d: extrapolated discontinuous flux 
          subroutine get_disc_flux_f(this, nele_x, f, f_d)
              class(fr), intent(inout) :: this

              integer(c_int), intent(in)     :: nele_x
              real(c_double), intent(in)     :: f(:, :) 

              real(c_double), intent(out)    :: f_d(:, :) 

              real(c_double)  :: extrap_mat(2, this%npts) !Extrapolation matrix for flux points

              integer(c_int) :: i 

              extrap_mat(1, :) = this%lagr_l 
              extrap_mat(2, :) = this%lagr_r 

              do i = 1, nele_x
                  f_d(:, i) = matmul(extrap_mat, f(:, i))
              end do

          end subroutine get_disc_flux_f


          !> Subroutine to get discontinuous derivative
          !! @param nele_x: number of elements
          !! @param u: solution vector
          !! @param du: derivative vector 
          subroutine get_discont_deriv(this, nele_x, u, du)
              class(fr), intent(inout)   :: this

              integer(c_int), intent(in)     :: nele_x
              real(c_double), intent(in)     :: u(this%npts, nele_x) 

              real(c_double), intent(out)    :: du(this%npts, nele_x) 

              integer(c_int)  :: i

              do i = 1, nele_x
                  du(:, i) = matmul(this%lagr_deri, u(:, i)) !Get discontinuous derivative
              end do

          end subroutine get_discont_deriv

          !> Subroutine to get interaction flux at boundary points 
          !! right now only implemented periodic condition
          !! @param nele_x: number of elements
          !! @param k: parameter for selecting flux type 0 for upwind, 1 for centered
          !! @param bnd: Parameter for bound. condn. 1 for imposing boundary and 0 for extrapolating 
          !! @param flux_f: flux at face 
          !! @param f_I: interaction flux at face 
          subroutine get_boundary_flux(this, nele_x, k, bnd, flux_f, f_I)
              class(fr), intent(inout)   :: this

              integer(c_int), intent(in)     :: nele_x
              real(c_double), intent(in)     :: k 
              integer(c_int), intent(in)     :: bnd 
              real(c_double), intent(in)     :: flux_f(2, nele_x) 

              real(c_double), intent(out)    :: f_I(2, nele_x) 

              real(c_double)  :: extrap_flux(2, this%npts) !Extrapolation matrix for flux points

              if (bnd .eq. one) then
!                  !!!!!!!!!!!!!!!!!!!!!!!
!                  !Periodic boundary conditions
!                  !At left boundary
!                  call get_interaction_flux(flux_f(2, nele_x), flux_f(1, 1), &
!                                            flux_f(2, nele_x), flux_f(1, 1), k, f_I(1, 1))
!                  call get_interaction_flux(flux_f(2, 1), flux_f(1, 1+1), &
!                                            flux_f(2, 1), flux_f(1, 1+1), k, f_I(2, 1))
!                      
!                  !At right boundary
!                  call get_interaction_flux(flux_f(2, nele_x-1), flux_f(1, nele_x), &
!                                        flux_f(2, nele_x -1 ), flux_f(1, nele_x), k, f_I(1, nele_x))
!                  call get_interaction_flux(flux_f(2, nele_x), flux_f(1, 1), &
!                                        flux_f(2, nele_x), flux_f(1, 1), k, f_I(2, nele_x))
!                  !!!!!!!!!!!!!!!!!!!!!!!
                  !Homogeneous boundary conditions
                  !At left boundary
                  f_I(1, 1) = zero
                  call get_interaction_flux(flux_f(2, 1), flux_f(1, 1+1), &
                                            flux_f(2, 1), flux_f(1, 1+1), k, f_I(2, 1))
                      
                  !At right boundary
                  call get_interaction_flux(flux_f(2, nele_x-1), flux_f(1, nele_x), &
                                        flux_f(2, nele_x -1 ), flux_f(1, nele_x), k, f_I(1, nele_x))
                  f_I(2, nele_x) = zero
               else if (bnd .eq. zero) then
                   f_I(:, 1)      = flux_f(:, 1)
                   f_I(:, nele_x) = flux_f(:, nele_x)
               else
                   write(*, *) "Error: Incorrect boundary option"
                   stop
               end if
                  
          end subroutine get_boundary_flux


          !> Subroutine to get interaction flux at interior points 
          !! @param nele_x: number of elements
          !! @param k: parameter for selecting flux type 0 for upwind, 1 for centered
          !! @param flux_f: flux at face 
          !! @param f_I: interaction flux at face 
          subroutine get_inter_flux(this, nele_x, k, flux_f, f_I)
              class(fr), intent(inout)   :: this

              integer(c_int), intent(in)     :: nele_x
              real(c_double), intent(in)     :: k 
              real(c_double), intent(in)     :: flux_f(2, nele_x) 

              real(c_double), intent(out)    :: f_I(2, nele_x) 

              real(c_double)  :: extrap_flux(2, this%npts) !Extrapolation matrix for flux points

              integer(c_int)  :: i

              do i = 2, nele_x - 1
                  call get_interaction_flux(flux_f(2, i-1), flux_f(1, i), &
                                        flux_f(2, i-1), flux_f(1, i), k, f_I(1, i))

                  call get_interaction_flux(flux_f(2, i), flux_f(1, i+1), &
                                        flux_f(2, i), flux_f(1, i+1), k, f_I(2, i))
              
              end do
          end subroutine get_inter_flux


          !> Subroutine to get derivative
          !! It assumes periodic boundary conditions for now
          !! @param nele_x: number of elements
          !! @param order: order of interpolating polynomial 
          !! @param npts: number of points in each element
          !! @param x: solution points vector
          !! @param k: parameter for selecting flux type 0 for upwind, 1 for centered
          !! @param bnd: Parameter for bound. condn. 1 for imposing boundary and 0 for extrapolating 
          !! @param u: solution vector
          !! @param du: derivative vector 
          subroutine get_derivative(this, nele_x, x, k, bnd, u, du)
              class(fr), intent(inout)   :: this

              integer(c_int), intent(in)     :: nele_x
              real(c_double), intent(in)     :: x(this%npts, nele_x) 
              real(c_double), intent(in)     :: u(this%npts, nele_x) 
              real(c_double), intent(in)     :: k !parameter for selecting flux type 0 for upwind, 1 for centered
              integer(c_int), intent(in)     :: bnd 

              real(c_double), intent(out)    :: du(this%npts, nele_x) 

              real(c_double)  :: du_temp(this%npts, nele_x) 
              real(c_double)  :: flux_f(2, nele_x) !Flux at cell edge

              real(c_double)  :: flux_l !Left flux
              real(c_double)  :: flux_r !Right flux

              real(c_double)  :: extrap_flux(2, this%npts) !Extrapolation matrix for flux points

              real(c_double)  :: f_I_l, f_I_r !Left and right interation flux
              real(c_double)  :: f_I(2, nele_x) !Left and right interation flux

              integer(c_int)  :: i, j 

              call this%get_discont_deriv(nele_x, u, du)

              call this%get_disc_flux_f(nele_x, u, flux_f) !Get discontinuous flux at flux point

              call this%get_boundary_flux(nele_x, k, bnd, flux_f, f_I)

              call this%get_inter_flux(nele_x, k, flux_f, f_I)

              do i = 1, nele_x 
                  du(:, i) = du(:, i) + (f_I(1, i) - flux_f(1, i))*this%g_l +  &
                                        (f_I(2, i) - flux_f(2, i))*this%g_r
              
                  !Transform derivative to physical space                  
                  du(:, i) = du(:,i )/this%x_r(:, i)
              end do

          end subroutine get_derivative




          !> Subroutine to get second derivative
          !! It assumes periodic boundary conditions for now
          !! @param x: solution points vector
          !! @param u: solution vector
          !! @param du: second derivative vector 
          subroutine get_sec_deri(this, x, u, du)
              class(fr), intent(inout)   :: this

              real(c_double), intent(in)     :: x(this%npts, this%nele_x) 
              real(c_double), intent(in)     :: u(this%npts, this%nele_x) 

              real(c_double), intent(out)    :: du(this%npts, this%nele_x) 

              real(c_double) :: du_temp(this%npts, this%nele_x) 
              real(c_double) :: k !Parameter for correction, 0 for upwind, 1 for central 
              integer(c_int) :: bnd !Parameter for bound. condn. 1 for imposing boundary and 0 for extrapolating 

              bnd = one 
              k   = one 
              call this%get_derivative(this%nele_x, x, k, bnd, u, du_temp)
              bnd = zero 
              k   = one
              call this%get_derivative(this%nele_x, x, k, bnd, du_temp, du)

          end subroutine get_sec_deri


          !> Subroutine to get first derivative
          !! It assumes periodic boundary conditions for now
          !! @param x: solution points vector
          !! @param u: solution vector
          !! @param du: second derivative vector 
          subroutine get_first_deri(this, x, u, du)
              class(fr), intent(inout)   :: this

              real(c_double), intent(in)     :: x(this%npts, this%nele_x) 
              real(c_double), intent(in)     :: u(this%npts, this%nele_x) 

              real(c_double), intent(out)    :: du(this%npts, this%nele_x) 

              real(c_double) :: k !Parameter for correction, 0 for upwind, 1 for central 
              integer(c_int) :: bnd !Parameter for bound. condn. 1 for imposing boundary and 0 for extrapolating 

              bnd = one
              k = zero
              call this%get_derivative(this%nele_x, x, k, bnd, u, du)

          end subroutine get_first_deri



  end module class_fr 



