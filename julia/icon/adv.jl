####################################
## SSP RK2 time stepping 
function ssp_rk2(dt, nx, dx, fn, vn_e, phi)
 
  k   = -1*fn(nx, dx, vn_e, phi)
  y   = phi + dt*k

  k   = -1*fn(nx, dx, vn_e, y)
  phi = 0.5*phi + 0.5*y + 0.5*dt*k

  return phi

end

function ssp_rk33(dt, nx, dx, fn, vn_e, phi)
  # x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
  # x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
  # x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt

  k   = -1*fn(nx, dx, vn_e, phi)
  y   =  phi + dt*k

  k   = -1*fn(nx, dx, vn_e, y)
  y   = (3.0/4)*phi + (1.0/4)*y + (1.0/4)*dt*k
  
  k   =  -1*fn(nx, dx, vn_e, y)
  phi = (1.0/3)*phi + (2.0/3)*y + (2.0/3)*dt*k

  return phi
end



####################################
## Get RHS 
function getRHS(nx, dx, vn_e, phi)

  edg_flx = get_edg_flux(nx, vn_e, phi)
  du      = get_du(nx, dx, edg_flx)

  return du

end



####################################
## Get derivative 
function get_du(nx, dx, flx)

  du = zeros(nx)

  for i = 1:nx
    du[i] = ( flx[i + 1] - flx[i] )/dx 
  end 

  return du
end


####################################
## We assume for now a uniform grid
function get_edg_flux(nx, vn_e, phi)

  cen_vel = zeros(nx)
  cen_flx = zeros(nx)
  for i = 1:nx
    cen_vel[i] = 0.5*( vn_e[i] + vn_e[i + 1] )
    cen_flx[i] = cen_vel[i]*phi[i] 
  end 
  edg_flx = zeros(nx + 1)
  for i = 2:nx    
    edg_flx[i] = 0.5*( cen_flx[i - 1] + cen_flx[i] )
  end

  ####################################
  ## Periodic boundary
  ####################################
  
  i = 1
  edg_flx[i] = 0.5*( cen_flx[nx] + cen_flx[i] )

  i = nx + 1
  edg_flx[i] = 0.5*( cen_flx[i - 1] + cen_flx[1] )


  return edg_flx

end

####################################
## Create 1D grid 
####################################

nx     = 400
startX = 0.0  
stopX  = 2.0*pi
dx     = (stopX - startX)/(nx);

edg_grd = range(startX, length=nx+1, stop = stopX)
cen_grd = zeros(nx)
for i=1:nx  
  cen_grd[i] = 0.5*( edg_grd[i] + edg_grd[i + 1] )
end

vel  = zeros(nx + 1) 
phi  = zeros(nx    )

dt   = 0.8*dx

global phi = exp.(-40*( cen_grd .- 1.0).^2) 
global vel =  ( sin.(( edg_grd .- 1.0)) ).^2 .+ 0.2

using Plots
anim = Animation()
for j = 1:525  

  # This is stupid. If I don't put the subscript
  # Julia thinks this is a new variable and so
  # changes size, so I have to put global
  global phi = ssp_rk33(dt, nx, dx, getRHS, vel, phi)

  if (j%10 == 0)
    plot(cen_grd, phi, xlim=( startX, stopX ), lw=4, legend=false)
    frame(anim)
  end

end

gif(anim, "var_adv.gif", fps=10)
