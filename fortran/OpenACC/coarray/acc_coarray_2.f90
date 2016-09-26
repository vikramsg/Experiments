program acc_coarray 

    use iso_c_binding
    use openacc

    implicit none

    integer(c_int) :: ngpus

    integer(c_int), parameter :: N = 1000

    real(c_double) :: x(N)[*], y(N)[*], z(N)[*]
    integer(c_int) :: i, b, nprocs

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Ensuring processes and devices are matched
    ngpus  = acc_get_num_devices(acc_device_nvidia)
    nprocs = num_images()

    write(*, *) 'Number of Nvidia devices on this node are ', ngpus
    
    write(*,*) "Hello from image ", this_image(), &
              "out of ", nprocs," total images"

    if (ngpus .lt. num_images()) then
        print*, "Number of processes greater than number of NVIDIA devices"
        stop
    end if
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    b = this_image()
    call acc_set_device_num(b, acc_device_nvidia)

    !$acc enter data create(x, y, z)

    !$acc kernels
    do i = 1, N
        x(i) = i*i
        y(i) = i + y(i-1)

        z(i) = b*x(i) + y(i)
    end do
    !$acc end kernels

    sync all()

    !$acc host_data use_device(z)
    z(1)[1] = z(1)[2]
    !$acc end host_data

    sync all()

    !$acc update self(z)

    !$acc exit data delete(x, y, z)

    if (this_image() .eq. 1) then
        do i = 1, num_images()
            write(*, *) "Values on process ", i
            write(*, *) z(1:4)[i]
        end do
    end if


end program acc_coarray

