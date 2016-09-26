
module wave

    use iso_c_binding

    implicit none

    integer(c_int) :: nCells
    integer(c_int) :: nFaces

    !u is at cell centers
    real(c_double), allocatable :: u(:)
    !x is at boundaries
    real(c_double), allocatable :: x(:)


    contains


        !Create simple 1D uniform mesh with number of points
        !and domain bounds
        subroutine mesh(n, startX, endX)
            integer(c_int), intent(in) :: n!number of edge points
            real(c_double), intent(in) :: startX, endX!Domain bounds

            integer(c_int) :: i
            real(c_double) :: h


            nFaces = n
            nCells = nFaces - 1

            allocate(x(nFaces))

            nCells = n
            nFaces = n + 1

            h = (endX - startX)/n

            do i = 1, n 
                x(i) = startX + (i - 1)*h + h/2.0
            end do

        end subroutine mesh

        !Initialize solution
        subroutine init

            integer(c_int) :: i

            allocate(u(nCells))

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

            integer(c_int) :: i, left, rght

            do i = 1 + 1 + 1, nFaces - 1 - 1 
                left = i - 1 !Cell index on the left
                rght = i + 1

                !Get soln at left and right cell of this face
                call getReconstructedSoln(left,  1.0_c_double, uLeft)
                call getReconstructedSoln(rght, -1.0_c_double, uRght)

                call getFlux(uLeft, fLeft)
                call getFlux(uRght, fRght)

!                call RusanovFlux(fLeft, fRight, uLeft, uRght)
            end do

        end subroutine step

        subroutine RusanovFlux(fLeft, fRght, uLeft, uRght, fCommon)
            real(c_double), intent(in)  :: fLeft, fRight, uLeft, uRght 
            real(c_double), intent(out) :: fCommon 


        end subroutine RusanovFlux



        !Get reconstructed soln at specified location inside cell
        !Location is normalized to [-1, 1]
        subroutine getReconstructedSoln(cellNo, location, soln)
            real(c_double), intent(in)  :: location 
            integer(c_int), intent(in)  :: cellNo 
            real(c_double), intent(out) :: soln

            real(c_double) :: h 
            real(c_double) :: duLeft, duRght 
            real(c_double) :: xLeft_1,  xLeft_2
            real(c_double) :: xRght_1,  xRght_2
            real(c_double) :: musclSigma 

            if ((location .gt. 1) .or. (location .lt. -1)) then
                print*, "Location has to be in [-1, 1]"
                stop
            end if

            !Get cell center location
            xLeft_1 = 0.5*(x(cellNo - 1) + x(cellNo))
            xLeft_2 = 0.5*(x(cellNo) + x(cellNo + 1))

            xRght_1 = 0.5*(x(cellNo)     + x(cellNo + 1))
            xRght_2 = 0.5*(x(cellNo + 1) + x(cellNo + 2))

            !Get cell dimension
            h = x(cellNo + 1) - x(cellNo)

            call uGrad(xLeft_1, xLeft_2, u(cellNo - 1), u(cellNo), duLeft )
            call uGrad(xRght_1, xRght_2, u(cellNo), u(cellNo + 1), duRght)

            call minMod(duLeft, duRght, musclSigma)

            soln = u(cellNo) + musclSigma*(location - 0) * h/(2.0_c_double)

        end subroutine getReconstructedFlux

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


