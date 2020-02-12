using PyPlot

function deri(dx, u)
    
  u_der = zeros(size(u)) 

  v     = copy(u)
  for i = 2:nx-1
    u_der[i] = ( v[i    ] - v[i - 1] )/(dx)
  end

  return u_der

end

function ssp_rk2(dt, dx, u)
 
  k = -1*deri(dx, u)

  y = u + dt*k

  k = -1*deri(dx, y)
  
  u = 0.5*u + 0.5*y + 0.5*dt*k

  return u


end


nx = 41;
dx = 2/(nx - 1);
dt = 0.04
u  = ones(nx)

s=Int(0.5/dx); e=Int(1/dx);

u[s:e] = 2*ones(e - s + 1)

y  = zeros(nx)
k  = zeros(nx)

for j = 1:10

  # This is stupid. If I don't put the subscript
  # Julia thinks this is a new variable and so
  # changes size, so I have to put global
  global u = ssp_rk2(dt, dx, u)

end

plot(range(0, length=nx, stop = 2), u)
plt.show()
