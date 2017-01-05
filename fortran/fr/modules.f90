  
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


  module face_mod
    use types_vars

    implicit none

    integer(c_int) :: num_faces  !Number of unique faces
            
    integer(c_int), allocatable :: face_glob_i(:, :) !The global index for each face using index of vertex
    integer(c_int), allocatable :: face_ele(:, :)    !The elements connected to each face 

    contains

        subroutine kill_face_mod_arrays 
!            deallocate(face_glob_i)
            deallocate(face_ele)
        end subroutine kill_face_mod_arrays 


        subroutine sort_indices(in_array, out_ind)
            integer(c_int), intent(in ) :: in_array(:) !Input array

            integer(c_int), intent(out) :: out_ind(:) !output array

            integer(c_int) :: temp_arr(size(in_array, 1)) 

            integer(c_int) :: i, temp, ind_temp

            temp_arr = in_array

            do i = 1, size(in_array, 1)
                ind_temp = minloc(temp_arr, dim = 1) 
                temp_arr(ind_temp) = maxval(temp_arr) + one

                out_ind(i) = ind_temp
            end do

        end subroutine sort_indices



        subroutine sort(ind)
            integer(c_int), intent(in ) :: ind(:) !Input array

            integer(c_int) :: temp_arr(size(ind, 1)) 

            integer(c_int) :: i, temp, ind_temp

            temp_arr = ind

            do i = 1, size(ind, 1)
                ind_temp = maxloc(temp_arr(i:size(ind, 1)), dim = 1) + (i - 1) 
                temp               = temp_arr(ind_temp)
                temp_arr(ind_temp) = temp_arr(i)
                temp_arr(i)        = temp
            end do

            write(*, *) temp_arr 

        end subroutine sort



        subroutine get_connect(x)
            real(c_double), intent(in ) :: x(:, :)

            integer(c_int), allocatable :: face(:)
            integer(c_int), allocatable :: face2elem(:)

            integer(c_int), allocatable :: sorted_indices(:) 
            integer(c_int), allocatable :: temp(:) 

            integer(c_int) :: i, j, e, n_uni_edge, n_assoc_ele

            allocate(face(2_c_int*size(x, 2) + 1)) !Contains the index of each vertex/face
            allocate(face2elem(2_c_int*size(x, 2) + 1)) !Contains the index of each vertex/face

            face      = -one !Initialize to detect padded array members
            face2elem = -one

            e = 1_c_int
            do i = 1, size(x, 2) 
                do j = 1, 2 !Hard coded. Each cell has 2 faces
                    face(e)      = i + (j - 1) !Index of each vertex
                    face2elem(e) = i !Associate that face with the cell
                    e            = e + 1
                end do
            end do

            allocate(sorted_indices(e-1))
            allocate(temp(e-1))

            !Get the indices of array in ascending order
            call sort_indices(face(1:e-1), sorted_indices) !Don't send padded elements

            temp = face(1:e-1)
            face(:) = temp(sorted_indices(:)) !So the assumption is that edges are now placed contiguously
            !which means that the same edge will be next to each other

            temp = face2elem(1:e-1)
            face2elem(:) = temp(sorted_indices(:))

            n_uni_edge = one 
            do i = 2, e - 1
                if (face(i) .ne. face(i - 1)) then
                    n_uni_edge = n_uni_edge + 1
                end if
            end do

            allocate(face_ele(2, n_uni_edge))
            face_ele = -one
            face_ele(1, 1) = face2elem(1)
            n_uni_edge  = one 
            n_assoc_ele = one
            do i = 2, e - 1
                if (face(i) .ne. face(i - 1)) then
                    n_uni_edge = n_uni_edge + 1
                    n_assoc_ele = one
                    face_ele(1, n_uni_edge) = face2elem(i)
                else
                    n_assoc_ele = n_assoc_ele + 1
                    face_ele(n_assoc_ele, n_uni_edge) = face2elem(i)
                end if

                if (n_assoc_ele .gt. 2) then
                    write(*, *) "Error: each face must have only two elements"
                    exit
                end if
            end do

            write(*, *) face_ele

            deallocate(temp)
            deallocate(sorted_indices)

            deallocate(face)
            deallocate(face2elem)
        

        end subroutine get_connect


  end module face_mod 



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
                    u(j, i) = hundred*sin(pi*x(j, i)/(one))
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

        subroutine exact_sol_sin_diff(nele, npts, x, c, L, alpha, t, u_ex, du_ex)

            integer(c_int), intent(in)  :: nele, npts
            real(c_double), intent(in)  :: c, L, t, alpha
            real(c_double), intent(in)  :: x(:, :)
            real(c_double), intent(out) :: u_ex(:, :)
            real(c_double), intent(out) :: du_ex(:, :)
    
            integer(c_int) :: i, j 
    
            do i = 1, nele
                do j = 1, npts
                    u_ex(j, i) = c*exp((-alpha*pi**2*t)/L**2)*sin(pi*(x(j, i)/L))
                    du_ex(j, i) = -c*(pi/L)*(pi/L)*exp((-alpha*pi**2*t)/L**2)*sin(pi*(x(j, i)/L))
                end do
            end do


    
    
        end subroutine exact_sol_sin_diff 

            
        subroutine get_grid_nodes(nele_x, startX, stopX, cart_x) !Get cartesian grid 
            integer(c_int), intent(in)  :: nele_x
            real(c_double), intent(in)  :: startX, stopX 
    
            real(c_double), intent(out) :: cart_x(:, :)
    
            integer(c_int) :: i, j 
            real(c_double) :: dx, temp 
    
            ! Grid spacing
            dx = (stopX - startX)/FLOAT(nele_x)
    
            ! Generate grid
            do i = 1, nele_x 
                temp         = startX + (i-1)*dx

                cart_x(1, i) = temp
                cart_x(2, i) = temp + dx
            end do

        end subroutine get_grid_nodes 

        !> Subroutine to make simple 1D grid
        !! @param nptsx num. of points
        !! @param startX, stopX starting and ending location
        !! @param x array containing grid points
        subroutine make_fr_grid(nele_x, startX, stopX, Np, x_nodes, x, dx)
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

        end subroutine make_fr_grid



        !> Subroutine to validate FR derivative
        !! @param nptsx number of points
        !! @param startX, stopX starting and ending location of grid
        !! @param order: order of polynomial
        subroutine validate_derivative(nele_x, startX, stopX, order)
            
            use polynomial
            use plot_data
            use operators
            use face_mod
      
            implicit none
      
            integer(c_int), intent(in)  :: nele_x, order
            real(c_double), intent(in)  :: startX, stopX 

            real(c_double) :: cart_x(2, nele_x) ! co-ordinates of each cell in the cartesian grid
      
            real(c_double) :: dx ! Characteristic length of cell 
  
            real(c_double) :: x_nodes(order + 1)   ! intra cell nodes 
            real(c_double) :: x(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: x_r(order + 1, nele_x) ! x co-ordinates
            real(c_double) :: u(order + 1, nele_x) ! soln vector 
            real(c_double) :: du(order + 1, nele_x) ! derivative vector 
      
            integer(c_int) :: Np, i 

            call test_poly()
            call test_matrix()
            call fun_matrix()

            call get_grid_nodes(nele_x, startX, stopX, cart_x) !Get cartesian grid 

            Np = order + 1

            call cell_coordi(order, x_nodes) !Get intra-cell co-ordis

            call make_fr_grid(nele_x, startX, stopX, Np, x_nodes, x, dx) !Get fr grid and charac. length
          
            call get_jacob(nele_x, Np, order, x, x_r)

            call init_sol(nele_x, Np, x, u)

            call get_connect(cart_x)

            call kill_face_mod_arrays 

        end subroutine validate_derivative



  end module subroutines


