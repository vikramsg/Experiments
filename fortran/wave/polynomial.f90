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
                  call get_lagrange( one, i, nodes, npts, lag_flux_r(i))
                  call get_lagrange(-one, i, nodes, npts, lag_flux_l(i))
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

              g_l = (-one)**(order) * half*(temp1 - temp2)

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




          subroutine fun_matrix
              real(c_double), allocatable :: x(:), deri(:, :) 
              real(c_double), allocatable :: deri1(:) 

              real(c_double), allocatable :: temp1(:) 
              real(c_double), allocatable :: temp2(:) 

              real(c_double), allocatable :: m1(:, :) 
              real(c_double), allocatable :: m2_l(:, :) 
              real(c_double), allocatable :: m2_r(:, :) 
              real(c_double), allocatable :: l_b(:) 
              real(c_double), allocatable :: x_b(:) 

              integer(c_int)    :: order, npts, i, j

              real(c_double)    :: f, r, temp 

              order = 1
              npts  = order + 1 

              allocate(x(npts))
              allocate(deri(npts, npts))
              allocate(m1(npts, npts))
              allocate(m2_l(npts, npts))
              allocate(m2_r(npts, npts))
              allocate(deri1(npts))
              allocate(temp1(npts))
              allocate(temp2(npts))
              allocate(l_b(npts))
              allocate(x_b(npts + 2))

              call gauss_nodes(order, x)

              call lagr_d_matrix(npts, x, deri)

              call lege_d_matrix(order, npts, x, deri1)

              call left_radau_d(order, npts, x, deri1)
              call right_radau_d(order, npts, x, deri)

              call lagr_flux_matrix(npts, x, temp1, temp2)

              do i = 1, npts
                  do j = 1, npts
                      m1(i, j) = deri1(i)*temp1(j)
                  end do
!                  write(*, *) m1(i, :)
              end do

              x_b(2:npts + 1) =  x
              x_b(1)          = -one
              x_b(npts + 2)   =  one
              do i = 1, npts + 2
                  call get_lagrange(x_b(i), 1, x_b, npts + 2, f)
!                  write(*, *) x_b(i), f
              end do
              
              do i = 1, npts          
                  call get_lagrange_d(x(i), 1, x_b, npts + 2, l_b(i))
              end do

              do i = 1, npts
                  do j = 1, npts
                      m2_l(i, j) = l_b(i)*temp1(j)
                      m2_r(i, j) = l_b(i)*temp2(j)
                  end do
!                  write(*, *) i
!                  write(*, *) m2_l(i, :)
!                  write(*, *) m2_r(i, :)
              end do
         
              deallocate(x)
              deallocate(deri)
              deallocate(deri1)
              deallocate(temp1)
              deallocate(temp2)
              deallocate(m1)
              deallocate(m2_l)
              deallocate(m2_r)
              deallocate(l_b)
              deallocate(x_b)

          end subroutine fun_matrix


  end module polynomial

