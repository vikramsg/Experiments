using  FastGaussQuadrature

function dg_mesh(nx, P, grd, typ)
  Np       = P + 1

  dg_msh  = zeros(nx, Np)

  if (typ == 1)
    nodes   = gausslobatto(Np)[1] 
  elseif (typ == 1)
    nodes   = gausslegendre(Np)[1] 
  end

  for i = 1:nx 
    for j = 1:Np 
      dg_msh[i, j] = 0.5*(1 - nodes[j])*grd[i] + 0.5*(1 + nodes[j])*grd[i + 1] 
    end
#    println(dg_msh[i, :])
  end

  return dg_msh

end

function uni_mesh(startX, stopX, P, nx)
  dx = (stopX - startX)/(nx);

  edg_grd = range(startX, length=nx+1, stop = stopX)
  return dg_mesh(nx, P, edg_grd, 1)

end


########################################################################
## Run
## Create 1D grid 
########################################################################

startX = -1.0
stopX  =  1.0
 
P  = 3 
nx = 10

dx = (stopX - startX)/(nx);

edg_grd = range(startX, length=nx+1, stop = stopX)
dg_mesh(nx, P, edg_grd, 1)

