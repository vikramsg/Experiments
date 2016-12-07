import sympy as smp

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

def get_ns_3d():
    '''
    Create a manufactured solution for compressbile NS
    '''
    x = smp.DeferredVector('x')
    t, mu, rho = smp.symbols('t, mu, rho')
    T, gamma, Cv, k = smp.symbols('T, gamma, Cv, k')

    dimension = 2

    v = (rho*smp.sin(x[0])*smp.cos(x[1])*smp.exp(-2*(mu/rho)*t),
          -rho*smp.cos(x[0])*smp.sin(x[1])*smp.exp(-2*(mu/rho)*t),
         )
    p = (rho/4.0)*(smp.cos(2*x[0]) + smp.cos(2*x[1]))*smp.exp(-4.0*(mu/rho)*t)

    ke = 0
    for i in v:
        ke = ke + 0.5*i*i

    u = (rho,
          rho*smp.sin(x[0])*smp.cos(x[1])*smp.exp(-2*(mu/rho)*t),
           -rho*smp.cos(x[0])*smp.sin(x[1])*smp.exp(-2*(mu/rho)*t),
            p/(gamma - 1) + rho*ke
         )

    rho = u[0]
    #velocity
    v = ()
    for i in range(1, dimension + 1):
        v = v + (u[i]/rho, )

    p = (rho/4.0)*(smp.cos(2*x[0]) + smp.cos(2*x[1]))*smp.exp(-4.0*(mu/rho)*t)

    #Divergence of velocity
    div = get_divergence(v, x, dimension)

    ke = 0
    for i in v:
        ke = ke + 0.5*i*i

    #Momentum
    f_inv = smp.zeros(dimension, dimension) + p*smp.eye(dimension)
    for i in range(0,dimension):
         for j in range(0,dimension):
             f_inv[i, j] = f_inv[i, j] + rho*(v[i]*v[j])  

    tau = smp.zeros(dimension, dimension) - (2/3.0)*smp.eye(dimension)*div
    for i in range(0,2):
         for j in range(0,2):
             tau[i, j] = tau[i, j] + mu*(smp.diff(v[i], x[j]) + smp.diff(v[j], x[i]))
    f_vis = -tau 

    resi = smp.zeros(dimension + 1, 1)

    for i in range(0, dimension):
        resi[i + 1] = smp.diff(u[i], t)
        resi[i + 1] = resi[i + 1] + get_divergence(f_inv[i, :], x, dimension)
        resi[i + 1] = resi[i + 1] + get_divergence(f_vis[i, :], x, dimension)

    R = (gamma - 1)*Cv
    T = p/(rho*R)

    E = Cv*T
    for i in range(0, dimension):
        E = E + 0.5*v[i]*v[i]
    H = E + p/rho

    E_inv = smp.zeros(dimension, 1) 
    for i in range(0,dimension):
        E_inv[i] = rho*v[i]*H

    E_vis = smp.zeros(dimension, 1) 
    for i in range(0, dimension):
        E_vis[i] = k*smp.diff(T, x[i])
        for j in range(0, dimension):
            E_vis[i] = E_vis[i] + tau[i, j]*v[j]

    print(smp.simplify(get_divergence(E_inv - E_vis, x, dimension)))




def get_ns():
    '''
    Create a manufactured solution for compressbile NS
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
             f_inv[i, j] = f_inv[i, j] + rho*(v[i]*v[j])  

    tau = smp.zeros(dimension, dimension) - (2/3.0)*smp.eye(dimension)*div
    for i in range(0,2):
         for j in range(0,2):
             tau[i, j] = tau[i, j] + mu*(smp.diff(v[i], x[j]) + smp.diff(v[j], x[i]))
    f_vis = -tau 

    resi = smp.zeros(dimension + 1, 1)

    for i in range(0, dimension):
        resi[i + 1] = smp.diff(u[i], t)
        resi[i + 1] = resi[i + 1] + get_divergence(f_inv[i, :], x, dimension)
        resi[i + 1] = resi[i + 1] + get_divergence(f_vis[i, :], x, dimension)

    R = (gamma - 1)*Cv
    T = p/(rho*R)

    E = Cv*T     
    H = E + p/rho

    E_inv = smp.zeros(dimension, 1) 
    for i in range(0,dimension):
        E_inv[i] = rho*v[i]*H

    E_vis = smp.zeros(dimension, 1) 
    for i in range(0, dimension):
        E_vis[i] = k*smp.diff(T, x[i])
        for j in range(0, dimension):
            E_vis[i] = E_vis[i] + tau[i, j]*v[j]

    print(smp.simplify(get_divergence(E_inv - E_vis, x, dimension)))



def problem_mms():
#    get_navier_stokes_3D()
#    get_euler_2D()
    get_ns_3d()
    get_ns()



if __name__=="__main__":
    problem_mms()
