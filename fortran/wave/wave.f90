
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

        !Initialize solution
        subroutine init

            integer(c_int) :: i

            do i = 1, nCells
                if (x(i) .ge. 0) then
                    u(i) = 1.0
                else
                    u(i) = 0.0
                end if
            end do

        end subroutine init


        subroutine solver

            call step()

        end subroutine solver

        subroutine step

            real(c_double) :: fRight, fLeft
            real(c_double) :: uRight, uLeft
            real(c_double) :: duRight, duLeft
            real(c_double) :: musclSigma 

            integer(c_int) :: i

            do i = 1+1, nCells-1
                call uGrad(x(i-1), x(i), u(i-1), u(i), duLeft )
                call uGrad(x(i), x(i+1), u(i), u(i+1), duRight)

                call minMod(duLeft, duRight, musclSigma)

                uLeft = u(i) - musclSigma*(x(i+1) - x(i))/2.0
                uRight= u(i) + musclSigma*(x(i) - x(i-1))/2.0

                call getFlux(uLeft, fLeft)
                call getFlux(uRight, fRight)

!                call RusanovFlux(fLeft, fRight)
            end do

        end subroutine step

        subroutine 

        subroutine getFlux(u, f)
            real(c_double), intent(in)  :: u
            real(c_double), intent(out) :: f

            f = u*u/2.0

        end subroutine getFlux


        subroutine minMod(x, y, mnMod)

            real(c_double), intent(in)  :: x, y
            real(c_double), intent(out) :: mnMod

            if ((x .gt. 0) .and. (y .gt. 0)) then
                mnMod = min(x, y)
            else if ((x .lt. 0) .and. (y .lt. 0)) then
                mnMod = max(x, y)
            else
                mnMod = 0
            end if

        end subroutine minMod



        !Get simple gradient
        subroutine uGrad(xL, xR, uL, uR, grad)
            real(c_double), intent(in)  :: xL, xR, uL, uR
            real(c_double), intent(out) :: grad 

            grad = (uR - uL)/(xR-xL)

        end subroutine uGrad


        subroutine shutdown
            deallocate(x)
            deallocate(u)
        end subroutine shutdown


end module wave


