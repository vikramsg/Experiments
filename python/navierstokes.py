import sympy as smp

#def get_navier_stokes_3D(u, p):
def get_navier_stokes_3D():
    '''
    Create a manufactured solution for 3D compressbile NS
    '''
#    u = smp.DeferredVector('u')
    x = smp.DeferredVector('x')
    t, mu, rho = smp.symbols('t, mu, rho')
    T, Cv, k = smp.symbols('T, Cv, k')
#    p = smp.symbols('p')
    p = x[1]**2 

    u = (x[0] ,
         x[1] ,
          x[2] 
         )

#    u = smp.MatrixSymbol('u', 3, 1)
#    x = smp.MatrixSymbol('x', 3, 1)

    div = smp.diff(u[0], x[0]) + smp.diff(u[1], x[1]) + smp.diff(u[2], x[2])

    tau = smp.zeros(3, 3) - (2/3.0)*smp.eye(3)*div
    for i in range(0,3):
         for j in range(0,3):
             tau[i, j] = tau[i, j] + mu*(smp.diff(u[i], x[j]) + smp.diff(u[j], x[i]))

    f_inv = smp.zeros(3, 3) + p*smp.eye(3)
    for i in range(0,3):
         for j in range(0,3):
             f_inv[i, j] = f_inv[i, j] + rho*(u[i]*u[j])  

    f_vis = -tau 

    E = Cv*T
    H = E + p/rho

    E_inv = 0
    for i in range(0,3):
        E_inv = E_inv + smp.diff(rho*u[i]*H, x[i])

    E_vis = 0
    for i in range(0,3):
        E_vis = E_vis - k*smp.diff(T, x[i])
        for j in range(0, 3):
            E_vis = E_vis - smp.diff(tau[i, j]*u[j], x[i])

    print(E_vis)
    

    #The residue vector
    resi = smp.zeros(5, 1) 

    #Density equation
    resi[0] = 0
    for i in range(0, 3):
        resi[0] = resi[0] + smp.diff(rho*u[i], x[i])

    #Momentum equation
    for i in range(0, 3):
        resi[i+1] = smp.diff(u[i], t)
        for j in range(0, 3):
            resi[i+1] = resi[i+1] + smp.diff(f_inv[i, j], x[j])
            resi[i+1] = resi[i+1] + smp.diff(f_vis[i, j], x[j])

    #Density equation
    resi[4] = E_inv + E_vis 

    print(resi)


#    f = (smp.simplify(f0),
#         smp.simplify(f1)
#         )
#    return f

def problem_mms():
    get_navier_stokes_3D()

    return f



if __name__=="__main__":
    problem_taylor()
