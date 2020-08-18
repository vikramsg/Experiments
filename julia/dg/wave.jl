include("mesh.jl")
include("poly.jl")

function init(grd)
  return exp.(-40*(grd ).^2) 
end

function getOperators(p)
  Np = p + 1

  M, D  = getDerivativeOp(Np)

  R     = restrictionOp(Np) 

  return M, D, R

end

function createSolver(startX, stopX, p, nx)
  mesh = uni_mesh(startX, stopX, p, nx)

  u  = zeros(nx, p + 1)
  u  = init(mesh) 

  M, D, R = getOperators(p)

  return u, M, D, R

end 

function getRHS(u, M, D, R)
end

function runSolver(startX, stopX, p, nx, CFL)
  u, M, D, R = createSolver(startX, stopX, p, nx)

  getRHS(u, M, D, R)
end


########################################################################
## Run
########################################################################


startX = -1.0
stopX  =  1.0
 
p  = 3 
nx = 10

CFL = 0.9

runSolver(startX, stopX, p, nx, CFL)
