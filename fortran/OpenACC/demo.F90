
module demo_rout

    use iso_c_binding, only: c_double, c_int

#ifdef _OPENACC
    use openacc
    use cublas
#endif

    implicit none

    contains

    subroutine demo(uflux, ucommon, ucorr)

        !$acc routine seq

        use iso_c_binding, only: c_double, c_int

        real(c_double)    :: ucorr, uflux, ucommon

        ucorr = uflux - ucommon
    end subroutine demo

end module demo_rout
