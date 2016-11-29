import sympy as smp

def get_euler_2D():
    '''
    Create a manufactured solution for 3D compressbile NS
    '''
    x = smp.DeferredVector('x')
    t, mu, rho = smp.symbols('t, mu, rho')
    T, Cv, k = smp.symbols('T, Cv, k')

    p = x[1]**2 

    u = (x[0] ,
         x[1] 
         )

    #Divergence
    div = 0
    for i in range(0, 2):
        div = div + smp.diff(u[i], x[i]) 

    #Momentum
    f_inv = smp.zeros(2, 2) + p*smp.eye(2)
    for i in range(0,2):
         for j in range(0,2):
             f_inv[i, j] = f_inv[i, j] + rho*(u[i]*u[j])  


    #Energy
    E = Cv*T 
    for i in range(0,2):
        E = E + 0.5*u[i]+u[i]
    H = E + p/rho

    E_inv =  smp.zeros(2, 1) 
    for i in range(0,2):
        E_inv[i] = rho*u[i]*H

    #The residue vector
    resi = smp.zeros(4, 1) 

    #Density equation
    resi[0] = 0
    resi[0] = smp.diff(u[0], t)
    for i in range(0, 2):
        resi[0] = resi[0] + smp.diff(rho*u[i], x[i])

    #Momentum equation
    for i in range(0, 2):
        resi[i+1] = smp.diff(u[i], t)
        for j in range(0, 2):
            resi[i+1] = resi[i+1] + smp.diff(f_inv[i, j], x[j])

    #Momentum equation
    resi[3] = smp.diff(rho*E, t)
    for j in range(0, 2):
        resi[3] = resi[3] + smp.diff(E_inv[j], x[j])

    print(resi)



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

    #Energy equation
    resi[4] = E_inv + E_vis 

    print(resi)


#    f = (smp.simplify(f0),
#         smp.simplify(f1)
#         )
#    return f

def get_divergence(u, x, r):
    '''
    Calculate divergence
    @param u vector whose divergence needs to be calculated
    @param x co-ordinates for divergence
    '''
    temp = 0
    for i in range(0, r):
        temp = temp + smp.diff(u[i], x[i])

    return temp



def get_incomp_ns_2D():
    '''
    Create a manufactured solution for 3D compressbile NS
    '''
    x = smp.DeferredVector('x')
    t, mu, rho = smp.symbols('t, mu, rho')
    T, gamma, Cv, k = smp.symbols('T, gamma, Cv, k')

    dimension = 2

    u = (rho*smp.sin(x[0])*smp.cos(x[1])*smp.exp(-2*(mu/rho)*t),
         -rho*smp.cos(x[0])*smp.sin(x[1])*smp.exp(-2*(mu/rho)*t)
         )

    p = (rho/4.0)*(smp.cos(2*x[0]) + smp.cos(2*x[1]))*smp.exp(-4.0*(mu/rho)*t)

    #Divergence of rho*velocity
    v = ()
    for i in range(0, dimension):
        v = v + (u[i]/rho, )
    div = get_divergence(v, x, dimension)

    #Momentum
    f_inv = smp.zeros(dimension, dimension) + p*smp.eye(dimension)
    for i in range(0,dimension):
         for j in range(0,dimension):
             f_inv[i, j] = f_inv[i, j] + rho*((u[i]/rho)*(u[j]/rho))  

    tau = smp.zeros(dimension, dimension) - (2/3.0)*smp.eye(dimension)*div
    for i in range(0,2):
         for j in range(0,2):
             tau[i, j] = tau[i, j] + mu*(smp.diff((u[i]/rho), x[j]) + smp.diff((u[j]/rho), x[i]))
    f_vis = -tau 

    resi = smp.zeros(dimension + 1, 1)

    for i in range(0, dimension):
        resi[i + 1] = smp.diff(u[i], t)
        resi[i + 1] = resi[i + 1] + get_divergence(f_inv[i, :], x, dimension)
        resi[i + 1] = resi[i + 1] + get_divergence(f_vis[i, :], x, dimension)

#    print(smp.simplify(resi))

    R = (gamma - 1)*Cv
    T = p/(rho*R)

    E = Cv*T     
    H = E + p/rho

    E_inv = smp.zeros(dimension, 1) 
    for i in range(0,dimension):
        E_inv[i] = u[i]*H

    E_vis = smp.zeros(dimension, 1) 
    for i in range(0, dimension):
#        E_vis[i] = k*smp.diff(T, x[i])
        for j in range(0, dimension):
            E_vis[i] = E_vis[i] + tau[i, j]*v[j]

    print(tau[0, 0]*v[0])
    print(smp.diff(tau[0, 0]*v[0], x[0]))
    print(tau[1, 1]*v[1])
    print(smp.diff(tau[1, 1]*v[1], x[1]))

#    print(smp.simplify(get_divergence(E_inv - E_vis, x, dimension)))



def problem_mms():
#    get_navier_stokes_3D()
#    get_euler_2D()
    get_incomp_ns_2D()



if __name__=="__main__":
    problem_mms()
