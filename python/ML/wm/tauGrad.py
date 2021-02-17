import sympy as smp

"""
Below is the code to get the derivative
The derivative and fnc needs to be evaluated only once
and then the functions will only substitute
"""
h     = smp.Symbol('h')
u     = smp.Symbol('u')
nu    = smp.Symbol('nu')
tau   = smp.Symbol('tau')

u_tau = smp.sqrt(tau)

kappa = 0.38
C     = 4.1

yplus = h*u_tau/nu
uplus = u/u_tau 

fnc   = uplus - ( (1./kappa)*smp.log(1 + kappa*yplus) + 
        ( C - (1./kappa)*smp.log(kappa) )*
          (1. - smp.exp(-yplus/11.) - (yplus/11.)*smp.exp(-yplus/3.)) )

der = smp.diff(fnc, tau).simplify()


"""
Get gradient of wall function to do 
Newton iterations
"""
def tauGrad(nu_s, h_s, u_s, tau_s):
  der_eval = der.evalf(subs = {tau:tau_s, nu:nu_s, h:h_s, u:u_s})
  fnc_eval = fnc.evalf(subs = {tau:tau_s, nu:nu_s, h:h_s, u:u_s})

  return [fnc_eval, der_eval]

"""
Return tau using Newton iterations
"""
def getTau(nu, h, u):
  tau0  = 0.001 
  tau   = tau0
 
  for i in range(0, 20):
    out  = tauGrad(nu, h, u, tau0) 
    tau  = tau0 - out[0]/out[1] 
 
    tau0 = tau
 
  return tau 




if __name__=="__main__":
    tauGrad(0.1, 1., 1., 0.1)
 
