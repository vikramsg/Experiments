program cuBLAS_example

  use iso_c_binding, only: c_double, c_int

#ifdef _OPENACC
  use cublas
  use openacc
#endif

  implicit none

  integer, parameter :: nelems = 500000
  integer, parameter :: Np = 16, Nvar = 4, Nflux = 16
  integer, parameter :: chunkSize = 100

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

  integer(c_int)      :: i, j, k, iter, rep


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


  !$acc enter data copyin(mesh)
  !$acc enter data copyin(Lgrad, phi)
  !$acc enter data create(uelem_tmp, ucorr_tmp, ugrad_tmp)

  do rep = 1, 100 
      do iter = 1, nelems/chunkSize
    
          !$acc parallel loop present(mesh, mesh%u, uelem_tmp)
          do i = 1, chunkSize 
              do j = 1, Nvar
                  do k = 1, Np
                      uelem_tmp(k, Nvar*(i-1) + j) = mesh%u(k, j, (iter-1)*chunkSize + i)
                  end do
              end do
          end do
          !$acc end parallel
    
          !$acc parallel loop present(mesh, mesh%uflux, mesh%ucommon, ucorr_tmp)
          do i = 1, chunkSize 
              do j = 1, Nvar
                  do k = 1, Nflux
                      ucorr_tmp(k, Nvar*(i-1) + j) = mesh%uflux(k, j, (iter-1)*chunkSize + i) &
                                                - mesh%ucommon(k, j, (iter-1)*chunkSize + i) 
    
                  end do
              end do
          end do
          !$acc end parallel
    
         !$acc host_data use_device(Lgrad, uelem_tmp, ugrad_tmp)
         call DGEMM('N', 'N', Np, Nvar*chunkSize, Np, &
             1.0_c_double, Lgrad, Np, uelem_tmp, Np, &
             0.0_c_double, ugrad_tmp, Np)
         !$acc end host_data
    
          !$acc host_data use_device(phi, uelem_tmp, ugrad_tmp)
          call DGEMM('N', 'N', Np, Nvar*chunkSize, Np, &
             1.0_c_double, phi, Np, uelem_tmp, Np, &
             1.0_c_double, ugrad_tmp, Np)
          !$acc end host_data
    
          !$acc parallel loop present(mesh, mesh%ux, ugrad_tmp)  
          do i = 1, chunkSize 
              do j = 1, Nvar
                  do k = 1, Nflux
                      mesh%ux(k, j, (iter-1)*chunkSize + i) = ugrad_tmp(k, Nvar*(i-1) + j) 
                  end do
              end do
          end do
          !$acc end parallel
    
       end do
    
       !$acc update self(mesh%ux)

   end do

   do i = 1, 10
       print*, mesh%ux(2, 3, i)
   end do

   !$acc exit data delete(uelem_tmp, ugrad_tmp, ucorr_tmp)
   !$acc exit data delete(Lgrad, phi)
   !$acc exit data delete(mesh)


end program cuBLAS_example
