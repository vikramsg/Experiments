

program main

    use wave
    use iso_c_binding 


    integer(c_int) :: numPoints
    real(c_double) :: startX, stopX


    numPoints = 50
    startX    = -1.0
    stopX     =  1.0

    call mesh(numPoints, startX, stopX)

    call solution(numPoints)


    call shutdown()


end program main



