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
    t = smp.Symbol('t')

    dimension = 2

#    v     = (1, -0.5         )#Velocity
    v     = (smp.sin(x[0]), -0.5         )#Velocity
    p     = 1                 #Pressure
    gamma = 1.4               
    mu    = 0.001             #Dynamic viscosity  
    Pr    = 0.72              #Prandtl number
    R     = 287               #Gas constant
    Cv    = R/(gamma - 1)     #Gas constant

    rho = 1 + 0.2*smp.sin(smp.pi*(x[0] + x[1] - t*(v[0] + v[1])))
    p   = smp.exp(smp.pi*(x[0] - t*(v[0] + v[1])))

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

    rho_x = smp.diff(u[0], x[0])
    rho_y = smp.diff(u[0], x[1])

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
        resi[i] = (u_t[i] + f_inv_x[i] + g_inv_y[i]
                               +f_vis[i] + g_vis[i] )

#    for i, resi_i in enumerate(resi):
#        print(resi_i.evalf(subs = {t:0, x[0]:0, x[1]:0}))

    print(T_x.evalf(subs = {t:0, x[0]:0.8, x[1]:0.45}))


def problem_mms():
    get_cns_2d()



if __name__=="__main__":
    problem_mms()
