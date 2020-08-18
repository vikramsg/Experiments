using  FastGaussQuadrature
import SymPy

function massMatrix(length, lag) 

  M     = zeros(length, length)

#  quad  = gausslegendre(length)
  quad  = gausslobatto(length)
  nodes = quad[1]
  wts   = quad[2]

  r     = SymPy.symbols("r", real=true)

  for k = 1:length
    for l = 1:length
      for j = 1:length
        lk = lag[k].subs(r, nodes[j])
        ll = lag[l].subs(r, nodes[j])
        M[k, l] = M[k, l] + wts[j]*lk*ll 
      end
    end
  end

  return M

end



function stiffMatrix(length, lag) 

  S     = zeros(length, length)

  quad  = gausslegendre(length)
  quad  = gausslobatto(length)
  nodes = quad[1]
  wts   = quad[2]

  r     = SymPy.symbols("r", real=true)

  for k = 1:length
    for l = 1:length
      for j = 1:length
        lk  =  lag[k].subs(r, nodes[j])
        dll = diff(lag[l], r).subs(r, nodes[j])
        S[k, l] = S[k, l] + wts[j]*lk*dll 
      end
    end
  end

  return S

end



function lagrange(length, nodes) 
  """
  Lagrange polynomial 
  """
  r   = SymPy.symbols("r", real=true)
  phi = SymPy.sympy.ones(1, length)
  for k = 1:length
    for l = 1:length 
      if (k != l)
        phi[k] = phi[k]*(r - nodes[l])/(nodes[k] - nodes[l])
      end
    end
  end

  return phi 

end


function lagrangeDeri(length, nodes) 
  """
  Lagrange matrix at the nodes is just an Identity
  We'll come back to interpolation at points other than nodes
  at a later time
  Here we create derivative operator at the nodes
  Lagrange polynomial is
  phi = Product(l, l.neq.k) (r - r_l)/(r_k - r_l)
  r_i are the nodes
  """
  r    = SymPy.symbols("r", real=true)
  phi  = SymPy.sympy.ones(1, length)
  dPhi = SymPy.sympy.zeros(length, length)
  for k = 1:length 
    for l = 1:length 
      if (k != l)
        phi[k] *= (r - nodes[l])/(nodes[k] - nodes[l])
      end
    end
  end
  for k = 1:length 
    for l = 1:length
      dPhi[k, l] = diff(phi[l], r).subs(r, nodes[k])
    end 
  end

  return dPhi

end

function getDerivativeOp(Np)

  typ = 1
  if (typ == 1)
    nodes   = gausslobatto(Np)[1] 
  elseif (typ == 1)
    nodes   = gausslegendre(Np)[1] 
  end

  lag   = lagrange(p + 1, nodes)

  M = massMatrix(p + 1, lag)

  dPhi = lagrangeDeri(p + 1, nodes)
  dPhi = convert(Array{Float64,2}, dPhi)

  return M, dPhi 
end


function restrictionOp(Np) 
  """
  We need the restriction matrix that gives
  the values at the edges
  """

  typ = 1
  if (typ == 1)
    nodes   = gausslobatto(Np)[1] 
  elseif (typ == 1)
    nodes   = gausslegendre(Np)[1] 
  end

  r    = SymPy.symbols("r", real=true)
  phi  = SymPy.sympy.ones(1, Np)
  for k = 1:Np
    for l = 1:Np 
      if (k != l)
        phi[k] *= (r - nodes[l])/(nodes[k] - nodes[l])
      end
    end
  end

  R = zeros(2, Np)
  for k = 1:Np
    R[1, k] = phi[k].subs(r, -1.0)
    R[2, k] = phi[k].subs(r,  1.0)
  end

  return R

end


########################################################################
## Run
########################################################################

p     = 3
quad  = gausslobatto(p + 1)
nodes = quad[1]
wts   = quad[2]

lag   = lagrange(p + 1, nodes)

#dPhi  = lagrangeDeri(p + 1, nodes)
#dPhi = convert(Array{Float64,2}, dPhi)
#println(dPhi)

#M = massMatrix(p + 1, lag)
#S = stiffMatrix(p + 1, lag) 
#
#println(M)
##println(S)
#println(inv(M)*S)

#R = restriction(p+1, nodes) 
