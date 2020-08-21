include("mesh.jl")
include("poly.jl")

struct Solver 
  nx   ::  Int
  p    ::  Int
  dx   ::  Array{Float64  } 
  mesh ::  Array{Float64,2} 
  M    ::  Array{Float64,2} 
  invM ::  Array{Float64,2} 
  D    ::  Array{Float64,2} 
  R    ::  Array{Float64,2} 
  u    ::  Array{Float64,2} 
end

function init(grd)
#  return exp.(-40*(grd ).^2) 
  return sin.(pi*grd) 
end

function getOperators(p)
  Np = p + 1

  M, D  = getDerivativeOp(Np)

  R     = restrictionOp(Np) 

  return M, D, R

end

function createSolver(startX, stopX, p, nx)

  dx, mesh = uni_mesh(startX, stopX, p, nx)

  u  = zeros(nx, p + 1)
  u  = init(mesh) 

  M, D, R = getOperators(p)
  invM    = inv(M)
  
  sol     = Solver(nx, p, dx, mesh, M, invM, D, R, u)

  return sol 

end 


function getCommonFlux(nx, u_f)
  f_edge = zeros(nx + 1)

  for i = 2:nx
#    f_edge[i] = 0.5*( u_f[i - 1, 2] + u_f[i, 1] )
    f_edge[i] = u_f[i - 1, 2] 
  end

  ## Periodic boundary condition 
  f_edge[nx + 1] = u_f[nx, 2] 
  f_edge[1     ] = u_f[nx, 2] 

  return f_edge

end

function getRHS(sol)
  Np   = sol.p + 1

  du   = zeros(sol.nx, Np)
  u_f  = zeros(sol.nx, 2)

  ## Get discontinuous derivative
  for i = 1:nx
    for j = 1:Np
      for k = 1:Np
        ## Recall our ref. domain is [-1, 1] so multiply with (2/h)
        du[i, j]  = du[i, j] + (2.0/sol.dx[i])*sol.D[j, k]*sol.u[i, k]
      end
    end
 end

 ## Project to face
 for i = 1:nx
    for j = 1:Np
      u_f[i, 1] = u_f[i, 1] + sol.R[1, j]*sol.u[i, j]
      u_f[i, 2] = u_f[i, 2] + sol.R[2, j]*sol.u[i, j]
    end
 end

 ## Get edge flux
 f_edg = getCommonFlux(sol.nx, u_f)
  
 ## Get SAT correction 
 for i = 1:nx
   for j = 1:Np
     ## Recall our ref. domain is [-1, 1] so multiply with (2/h)
     du[i, j]  = du[i, j] + (2.0/sol.dx[i])*sol.invM[j, j]*
                     sol.R[1, j]*(-1.0)*(f_edg[i    ] - u_f[i, 1])
     du[i, j]  = du[i, j] + (2.0/sol.dx[i])*sol.invM[j, j]*
                     sol.R[2, j]*( 1.0)*(f_edg[i + 1] - u_f[i, 2])
   end
#   println(i, " ", f_edg[i], " ", f_edg[i + 1])
#   println(i, " ", u_f[i, 1], " ", u_f[i, 2])
 end

# display(sol.M)
# display(sol.invM*sol.R[1, :])
# display(du)
 du = -1 .*du

# error("stop")

 return du

end

function ssp_rk33(dt, sol, rhs_fn) 
  # x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
  # x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
  # x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt
  
  u       = zero( sol.u )
  u[:, :] = sol.u[:, :]
  
  rhs =  rhs_fn(sol)
  y   =  u + dt*rhs

  sol.u[:, :] = y[:, :]
  rhs =  rhs_fn(sol)
  y   = (3.0/4)*u + (1.0/4)*y + (1.0/4)*dt*rhs
 
  sol.u[:, :] = y[:, :]
  rhs =  rhs_fn(sol)
  u   = (1.0/3)*u + (2.0/3)*y + (2.0/3)*dt*rhs

  sol.u[:, :] = u[:, :]

end


  
using Plots


function flatten(arr)
  sz = size(arr)

  length = sz[1]*sz[2]

  ret_arr = zeros(length)
  for i = 1:sz[1]
    for j = 1:sz[2]
      ret_arr[( i - 1 )*sz[2] + j] =  arr[i, j]
    end
  end

  return ret_arr

end

function runSolver(startX, stopX, p, nx, CFL)
  sol = createSolver(startX, stopX, p, nx);
  
  dt = 0.03 

  T_final = 0.03
  T       = 0
  dt_real = min(dt, T_final - T)

  it_coun = 0
  while (T < T_final)  
    u   = ssp_rk33(dt, sol, getRHS) 

    T       = T + dt_real
    dt_real = min(dt, T_final - T)

#    if (it_coun % 10 == 0):
    println("Time: ", T, " Max u: ", maximum(u))

    it_coun  = it_coun + 1
  end

  flat_x = flatten(sol.mesh)
  flat_u = flatten(sol.u)

  plot(flat_x, flat_u, lw=4, legend=true)

end


########################################################################
## Run
########################################################################


startX = -1.0
stopX  =  1.0
 
p  = 2 
nx = 10

CFL = 0.9

runSolver(startX, stopX, p, nx, CFL)
