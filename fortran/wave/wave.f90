
module wave

    use iso_c_binding

    implicit none

    integer(c_int) :: nCells

    real(c_double), allocatable :: u(:)
    real(c_double), allocatable :: x(:)


    contains


        !Create simple 1D uniform mesh with number of points
        !and domain bounds
        subroutine mesh(n, startX, endX)
            integer(c_int), intent(in) :: n!number of points
            real(c_double), intent(in) :: startX, endX!Domain bounds

            integer(c_int) :: i
            real(c_double) :: h

            allocate(x(n))

            nCells = n

            h = (endX - startX)/n

            do i = 1, n 
                x(i) = startX + (i - 1)*h + h/2.0
            end do

        end subroutine mesh

        !Create solution container with number of points
        subroutine solution(n)
            integer(c_int), intent(in) :: n!number of points

            allocate(u(n))

        end subroutine solution

        subroutine init

            integer(c_int) :: i

            do i = 1, nCells
            end do

        end subroutine init


        subroutine shutdown
            deallocate(x)
            deallocate(u)
        end subroutine shutdown


end module wave


