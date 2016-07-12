program cuBLAS_example

  use iso_c_binding, only: c_double, c_int

  implicit none

  integer, parameter :: nelems = 500000
  integer, parameter :: Np = 16, Nvar = 4, Nflux = 16
  integer, parameter :: chunkSize = 50

  type mesh2d
      real(c_double) :: u(Np, Nvar, nelems)
      real(c_double) :: ux(Np, Nvar, nelems)
      real(c_double) :: uflux(Nflux, Nvar, nelems)
      real(c_double) :: ucommon(Nflux, Nvar, nelems)
  end type mesh2d
  
  real(c_double)   :: Lgrad(Np, Np), phi(Np, Nflux) 
  real(c_double)   :: uelem_tmp(Np, Nvar*chunkSize), ucorr_tmp(Nflux, Nvar*chunkSize) 
  real(c_double)   :: ugrad_tmp(Np, Nvar*chunkSize)

  type(mesh2d)     :: mesh

  integer(c_int)   :: i, j, k, iter, rep

  real(c_double)   :: t1, t2

  call CPU_TIME(t1)

  !$OMP PARALLEL DO PRIVATE(i, j, k) 
  do rep = 1, 10
  do i = 1, nelems
      do j = 1, Nvar
          do k = 1, Np
              mesh%u(k, j, i) = i+j+k
          end do
          do k = 1, Nflux
              mesh%uflux(k, j, i) = i-j-k
              mesh%ucommon(k, j, i) = i-j+k
          end do
      end do
  end do

  do j = 1, Np
    do k = 1, Np
        Lgrad(j, k) = k*j                  
    end do
    do k = 1, Nflux
        phi(j, k) = k*j + k + j
    end do
  end do
  end do
  !$END PARALLEL DO 

  !$OMP PARALLEL DO PRIVATE(i, j, k, uelem_tmp, ucorr_tmp)
  do rep = 1, 10 
      do iter = 1, nelems/chunkSize

          do i = 1, chunkSize 
              do j = 1, Nvar
                  do k = 1, Np
                      uelem_tmp(k, Nvar*(i-1) + j) = mesh%u(k, j, (iter-1)*chunkSize + i)
                  end do
              end do
          end do
    
          do i = 1, chunkSize 
              do j = 1, Nvar
                  do k = 1, Nflux
                      ucorr_tmp(k, Nvar*(i-1) + j) = mesh%uflux(k, j, (iter-1)*chunkSize + i) &
                                                - mesh%ucommon(k, j, (iter-1)*chunkSize + i) 
    
                  end do
              end do
          end do

    
         call DGEMM('N', 'N', Np, Nvar*chunkSize, Np, &
             1.0_c_double, Lgrad, Np, uelem_tmp, Np, &
             0.0_c_double, ugrad_tmp, Np)
    
          call DGEMM('N', 'N', Np, Nvar*chunkSize, Np, &
             1.0_c_double, phi, Np, uelem_tmp, Np, &
             1.0_c_double, ugrad_tmp, Np)
    
          do i = 1, chunkSize 
              do j = 1, Nvar
                  do k = 1, Nflux
                      mesh%ux(k, j, (iter-1)*chunkSize + i) = ugrad_tmp(k, Nvar*(i-1) + j) 
                  end do
              end do
          end do
    
       end do
   end do
   !$OMP END PARALLEL

   call CPU_TIME(t2)

   print*, 'Time taken', t2-t1


end program cuBLAS_example
