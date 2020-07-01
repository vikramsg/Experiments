using PyPlot

gamm = 1.4

##############################################
## Get initial condition 
##############################################
function fn(nx, x)

  p   = 1.0
  vel = 1.0

  for i = 1:nx
      u[       i] = 1.0 + 0.2*sin(pi*(x[i]))
      u[  nx + i] = u[i]
      u[2*nx + i] = p/(gamm - 1) + 0.5*u[i]*vel*vel
  end

  return u 

end

##############################################
## Get Euler flux from conservative variables 
##############################################
function euler_flux(u)
  f    = zero(u)
  nsh  = length(u)

  nx   = Int(nsh/3)

  for i = 1:nx
    rho    = u[i]
    rho_V  = u[nx + i]
    rho_E  = u[2*nx + i]

    vel    = rho_V/rho
    v_sq   = vel*vel
    p      = (gamm - 1.0)*(rho_E - 0.5*rho*v_sq)

    f[       i] = rho_V 
    f[  nx + i] = rho*v_sq 
    f[2*nx + i] = (rho_E + p)*vel 
  end

  return f

end

####################################
## Use Rusanov flux 
####################################
function getLFFlux(u_L, u_R, f_L, f_R)
  rho_L    = u_L[1]
  rho_V_L  = u_L[2]
  rho_E_L  = u_L[3]

  vel_L    = rho_V_L/rho_L
  v_sq_L   = vel_L*vel_L
  p_L      = (gamm - 1.0)*(rho_E_L - 0.5*rho_L*v_sq_L)
  a_L      = sqrt(gamm*p_L/rho_L)


  rho_R    = u_R[1]
  rho_V_R  = u_R[2]
  rho_E_R  = u_R[3]

  vel_R    = rho_V_R/rho_R
  v_sq_R   = vel_R*vel_R
  p_R      = (gamm - 1.0)*(rho_E_R - 0.5*rho_R*v_sq_R)
  a_R      = sqrt(gamm*p_R/rho_R)

  lambda   = max(a_L + abs(vel_L), a_R + abs(vel_R))

  return 0.5*( (f_L + f_R) - lambda*(u_R - u_L) )
end 

####################################
## Get fluxes at face 
## Assume periodic boundary
####################################
function edge_flux(u, f)
  nsh    = length(u)
  nx     = Int(nsh/3)

  # Edge flux
  f_edge = zeros(3* (nx + 1) )

    u_L    = zeros(3)
    f_L    = zeros(3)
    u_R    = zeros(3)
    f_R    = zeros(3)
  for i=2:nx  
    u_L[1] = u[       i - 1]
    u_L[2] = u[  nx + i - 1]
    u_L[3] = u[2*nx + i - 1]
    f_L    = zeros(3)
    f_L[1] = f[       i - 1]
    f_L[2] = f[  nx + i - 1]
    f_L[3] = f[2*nx + i - 1]

    u_R[1] = u[       i]
    u_R[2] = u[  nx + i]
    u_R[3] = u[2*nx + i]
    f_R[1] = f[       i]
    f_R[2] = f[  nx + i]
    f_R[3] = f[2*nx + i]

    riem_flux = getLFFlux(u_L, u_R, f_L, f_R)
    f_edge[       i] = riem_flux[1]
    f_edge[  nx + i] = riem_flux[2] 
    f_edge[2*nx + i] = riem_flux[3]
  end

  ####################################
  #Periodic boundary

  i = 1
  u_L[1] = u[       nx   ]
  u_L[2] = u[  nx + nx   ]
  u_L[3] = u[2*nx + nx   ]
  f_L[1] = f[       nx   ]
  f_L[2] = f[  nx + nx   ]
  f_L[3] = f[2*nx + nx   ]

  u_R[1] = u[       i]
  u_R[2] = u[  nx + i]
  u_R[3] = u[2*nx + i]
  f_R[1] = f[       i]
  f_R[2] = f[  nx + i]
  f_R[3] = f[2*nx + i]

  riem_flux = getLFFlux(u_L, u_R, f_L, f_R)
  f_edge[       i] = riem_flux[1]
  f_edge[  nx + i] = riem_flux[2] 
  f_edge[2*nx + i] = riem_flux[3]


  i = nx + 1
  u_L[1] = u[       i - 1]
  u_L[2] = u[  nx + i - 1]
  u_L[3] = u[2*nx + i - 1]
  f_L[1] = f[       i - 1]
  f_L[2] = f[  nx + i - 1]
  f_L[3] = f[2*nx + i - 1]

  u_R[1] = u[       1]
  u_R[2] = u[  nx + 1]
  u_R[3] = u[2*nx + 1]
  f_R[1] = f[       1]
  f_R[2] = f[  nx + 1]
  f_R[3] = f[2*nx + 1]

  riem_flux = getLFFlux(u_L, u_R, f_L, f_R)
  f_edge[       i] = riem_flux[1]
  f_edge[  nx + i] = riem_flux[2] 
  f_edge[2*nx + i] = riem_flux[3]

  ####################################

  return f_edge

end 

  
####################################
## Get right hand side
####################################
function getRHS(nx, dx, u)
  global f     = euler_flux(u)

  global f_edg = edge_flux(u, f)  

  rhs = zeros(3*nx)
  for i=1:nx  
    rhs[       i] = (f_edg[       i + 1] - f_edg[       i] )/dx
    rhs[  nx + i] = (f_edg[  nx + i + 1] - f_edg[  nx + i] )/dx
    rhs[2*nx + i] = (f_edg[2*nx + i + 1] - f_edg[2*nx + i] )/dx
  end

#  println(rhs[1:nx])
#  println(rhs[nx])
#  println(f_edg[nx - 1])
#  println(f_edg[nx])
#  println(f_edg[nx + 1])
#  println(f_edg[     1])
#  println(f_edg[     2])

  return rhs
end


function ssp_rk2(dt, nx, dx, fn, u)
 
  k = -1*fn(nx, dx, u)
  y = u + dt*k

  k = -1*fn(nx, dx, y)
  u = 0.5*u + 0.5*y + 0.5*dt*k

  return u

end


startX = -1.0
stopX  =  1.0
nx = 410;
dx = (stopX - startX)/(nx);

edg_grd = range(startX, length=nx+1, stop = stopX)
x_grd   = zeros(nx)
for i=1:nx  
  x_grd[i] = 0.5*( edg_grd[i] + edg_grd[i + 1] )
end

## Euler conserved variables in 1D
u     = zeros(3*nx)

global u     = fn(nx, x_grd)
global rhs   = getRHS(nx, dx, u) 

dt = 0.2*dx
for j = 1:400

  # This is stupid. If I don't put the subscript
  # Julia thinks this is a new variable and so
  # changes size, so I have to put global
  global u = ssp_rk2(dt, nx, dx, getRHS, u)

end

plt.plot(x_grd, u[1:nx])
plt.show()


