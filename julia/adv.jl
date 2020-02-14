using PyPlot

function upwd(nx, dx, u)
    
  u_der = zeros(size(u)) 

  for i = 2:nx-1
    u_der[i] = ( u[i ] - u[i - 1] )/(dx)
  end

  return u_der

end

function cd2(nx, dx, u)
    
  u_der = zeros(size(u)) 

  for i = 2:nx-1
    u_der[i] = ( u[i + 1] - u[i - 1] )/(2*dx)
  end

  return u_der

end

function ssp_rk2(dt, nx, dx, fn, u)
 
  k = -1*fn(nx, dx, u)
  y = u + dt*k

  k = -1*fn(nx, dx, y)
  u = 0.5*u + 0.5*y + 0.5*dt*k

  return u

end


X  = 2.0
nx = 410;
dx = X/(nx - 1);

x_grd = range(0, length=nx, stop = X)

dt = 0.8*dx 

u    = zeros(nx)
init = 0

if init==0
  global u  = ones(nx)
  
  s=Int(floor(0.5/dx)); e=Int(floor(1/dx));
  
  u[s:e] = 2*ones(e - s + 1)
elseif init==1

  global u  = exp.(-40*(x_grd .- 1.0).^2) 

end 

y  = zeros(nx)
k  = zeros(nx)

for j = 1:100

  # This is stupid. If I don't put the subscript
  # Julia thinks this is a new variable and so
  # changes size, so I have to put global
  global u = ssp_rk2(dt, nx, dx, upwd, u)

end

plot(x_grd, u)
plt.show()
