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



def get_cns_2d(): 
    '''
    Create a manufactured solution for compressbile NS
    '''
    x = smp.DeferredVector('x')
    t = smp.symbols('t')

    dimension = 2

    v     = (1, -0.5         )#Velocity
    p     = 1                 #Pressure
    gamma = 1.4               
    mu    = 0.100             #Dynamic viscosity  
    Pr    = 0.72              #Prandtl number
    R     = 287               #Gas constant
    Cv    = R/(gamma - 1)     #Gas constant

    rho = 1 + 0.2*smp.sin(smp.pi*(x[0] + x[1] - t*(v[0] + v[1])))

    #Conserved variables
    u =  [0, 0, 0, 0]
    u[0] = rho
    u[1] = u[0]*v[0]
    u[2] = u[0]*v[1]
    u[3] = p/(gamma - 1) + 0.5*u[0]*(v[0]*v[0] + v[1]*v[1]) 

    T    = (u[3]/rho - 0.5*(v[0]*v[0] + v[1]*v[1]))/Cv


    ##########################
    #Inviscid
    ##########################
    #x Inviscid flux
    f_inv =  [0, 0, 0, 0]
    f_inv[0] = rho*v[0]
    f_inv[1] = rho*v[0]*v[0] + p
    f_inv[2] = rho*v[0]*v[1]
    f_inv[3] = (u[3] + p)*v[0] 

    #y Inviscid flux
    g_inv =  [0, 0, 0, 0]
    g_inv[0] = rho*v[1]
    g_inv[1] = rho*v[0]*v[1]
    g_inv[2] = rho*v[1]*v[1] + p
    g_inv[3] = (u[3] + p)*v[1] 

    #Time derivative
    u_t =  [0, 0, 0, 0]
    for i, u_i in enumerate(u):
        u_t[i] = smp.diff(u[i], t)

    #x derivative of inviscid flux
    f_inv_x =  [0, 0, 0, 0]
    for i, f_i in enumerate(f_inv):
        f_inv_x[i] = smp.diff(f_inv[i], x[0])

    #y derivative of inviscid flux
    g_inv_y =  [0, 0, 0, 0]
    for i, g_i in enumerate(g_inv):
        g_inv_y[i] = smp.diff(g_inv[i], x[1])


    ##########################
    #Viscous
    ##########################

    u_x = smp.diff(u[1]/u[0], x[0])
    u_y = smp.diff(u[1]/u[0], x[1])
    v_x = smp.diff(u[2]/u[0], x[0])
    v_y = smp.diff(u[2]/u[0], x[1])

    div = u_x + v_y

    T_x = smp.diff(T, x[0])
    T_y = smp.diff(T, x[1])

    #x Viscous flux
    tau11 = mu*(2*u_x - (2.0/3.0)*div)
    tau12 = mu*(u_y + v_x)
    f_vis =  [0, 0, 0, 0]
    f_vis[0] = 0 
    f_vis[1] = -(tau11) 
    f_vis[2] = -(tau12) 
    f_vis[3] = -(v[0]*tau11 + v[1]*tau12 + Pr*T_x)

    #y Viscous flux
    tau21 = mu*(u_y + v_x)
    tau22 = mu*(2*v_y - (2.0/3.0)*div)
    g_vis    =  [0, 0, 0, 0]
    g_vis[0] = 0 
    g_vis[1] = -(tau21) 
    g_vis[2] = -(tau22) 
    g_vis[3] = -(v[0]*tau21 + v[1]*tau22 + Pr*T_y)

    #Residual
    resi =  [0, 0, 0, 0]
    for i, resi_i in enumerate(resi):
        resi[i] = smp.simplify(u_t[i] + f_inv_x[i] + g_inv_y[i]
                               +f_vis[i] + g_vis[i]) 

    print(resi)



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
#    get_ns_3d()
    get_euler_2d()
#    get_ns()



if __name__=="__main__":
    problem_mms()
