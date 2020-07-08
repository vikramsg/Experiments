function fn(nx, ny, x, y)
  u = zeros(nx, ny)

  for i=1:nx
    for j = 1:ny
      dist     = (x[i] - 0.5).^2 + (y[j] - 0.5).^2

      u[i, j]  = exp.(-40*dist) 

#      st = string(x[i]) * " " * string(y[j]) * " " * string(u[i, j]) 
    end
  end

  return u
end 


####################################
## Get fluxes at face 
## Assume periodic boundary
####################################
function edge_flux(nx, ny, u)
  # Edge flux
  f_edge = zeros(nx+1, ny)

  for i=2:nx  
    f_edge[i] = 0.5*( u[i - 1] + u[i] )
  end

  ####################################
  #Periodic boundary
  
  i = 1
  f_edge[i] = 0.5*( u[nx] + u[i] )

  i = nx + 1
  f_edge[i] = 0.5*( u[i - 1] + u[1] )

  return f_edge

end 


function upwd(nx, ny, dx, dy, u)
    
  u_der = zeros(size(u)) 

  f_edg = edge_flux(nx, ny, u)

  for j = 2:ny-1
    for i = 2:nx-1
      u_der[i, j] = ( u[i, j] - u[i - 1, j] )/(dx)
    end
  end

  return u_der

end


function ssp_rk2(dt, nx, ny, dx, dy, fn, u)
 
  k = -1*fn(nx, ny, dx, dy, u)
  y = u + dt*k

  k = -1*fn(nx, ny, dx, dy, y)
  u = 0.5*u + 0.5*y + 0.5*dt*k

  return u

end


nx = 50; ny = 50;
startX = 0.0; stopX = 1.0;
startY = 0.0; stopY = 1.0;

x_grd  = zeros(nx)
y_grd  = zeros(ny)

edg_grd_x = range(startX, length=nx+1, stop = stopX)
edg_grd_y = range(startY, length=ny+1, stop = stopY)
for i=1:nx
  x_grd[i] = 0.5*( edg_grd_x[i] + edg_grd_x[i + 1] )
end
for j = 1:ny
  y_grd[j] = 0.5*( edg_grd_y[j] + edg_grd_y[j + 1] )
end
dx = (stopX - startX)/(nx);
dy = (stopY - startY)/(ny);

## Initialize
global u = fn(nx, ny, x_grd, y_grd)
  
dt = 0.8*dx 
for j = 1:20 

  global u = ssp_rk2(dt, nx, ny, dx, dy, upwd, u)

end


using Plots
Plots.pyplot()
for i = 1:1
  p1 = contour(x_grd, y_grd, u)
  plot(p1)
end
savefig("see.png")

