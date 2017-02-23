import sympy as smp

def get_cns_3d(x, t, gamma, R, mu, Pr, u): 
    assert(len(u) == 5)
    Cv     = R/(gamma - 1)
    v      = [0, 0, 0] #Velocity
    rho    = u[0]
    v[0]   = u[1]/rho
    v[1]   = u[2]/rho
    v[2]   = u[3]/rho

    v_sq   = (v[0]*v[0] + v[1]*v[1])

    T      = (u[4]/rho - 0.5*v_sq)/Cv
    p      = rho*R*T

    ##########################
    #Inviscid
    ##########################
    #x Inviscid flux
    f_inv =  [0, 0, 0, 0, 0]
    f_inv[0] = rho*v[0]
    f_inv[1] = rho*v[0]*v[0] + p
    f_inv[2] = rho*v[0]*v[1]
    f_inv[3] = rho*v[0]*v[2]
    f_inv[4] = (u[4] + p)*v[0] 

    #y Inviscid flux
    g_inv =  [0, 0, 0, 0, 0]
    g_inv[0] = rho*v[1]
    g_inv[1] = rho*v[1]*v[0]
    g_inv[2] = rho*v[1]*v[1] + p
    g_inv[3] = rho*v[1]*v[2]
    g_inv[4] = (u[4] + p)*v[1] 

    #z Inviscid flux
    h_inv =  [0, 0, 0, 0, 0]
    h_inv[0] = rho*v[2]
    h_inv[1] = rho*v[2]*v[0]
    h_inv[2] = rho*v[2]*v[1] 
    h_inv[3] = rho*v[2]*v[2] + p
    h_inv[4] = (u[4] + p)*v[2] 


    #Time derivative
    u_t =  [0, 0, 0, 0, 0]
    for i, u_i in enumerate(u):
        u_t[i] = smp.diff(u[i], t)

    #x derivative of inviscid flux
    f_inv_x =  [0, 0, 0, 0, 0]
    for i, f_i in enumerate(f_inv):
        f_inv_x[i] = smp.diff(f_inv[i], x[0])

    #y derivative of inviscid flux
    g_inv_y =  [0, 0, 0, 0, 0]
    for i, g_i in enumerate(g_inv):
        g_inv_y[i] = smp.diff(g_inv[i], x[1])

    #y derivative of inviscid flux
    h_inv_z =  [0, 0, 0, 0, 0]
    for i, h_i in enumerate(h_inv):
        h_inv_z[i] = smp.diff(h_inv[i], x[2])


    ##########################
    #Viscous
    ##########################

    rho_x = smp.diff(u[0], x[0])
    rho_y = smp.diff(u[0], x[1])
    rho_z = smp.diff(u[0], x[2])

    u_x = smp.diff(u[1]/u[0], x[0])
    u_y = smp.diff(u[1]/u[0], x[1])
    u_z = smp.diff(u[1]/u[0], x[2])

    v_x = smp.diff(u[2]/u[0], x[0])
    v_y = smp.diff(u[2]/u[0], x[1])
    v_z = smp.diff(u[2]/u[0], x[2])

    w_x = smp.diff(u[3]/u[0], x[0])
    w_y = smp.diff(u[3]/u[0], x[1])
    w_z = smp.diff(u[3]/u[0], x[2])


    div = u_x + v_y + w_z

    T_x = smp.diff(T, x[0])
    T_y = smp.diff(T, x[1])
    T_z = smp.diff(T, x[2])

    #x Viscous flux
    tau11 = mu*(2*u_x - (2.0/3.0)*div)
    tau12 = mu*(u_y + v_x)
    tau13 = mu*(u_z + w_x)
    f_vis =  [0, 0, 0, 0, 0]
    f_vis[0] = 0 
    f_vis[1] = tau11 
    f_vis[2] = tau12 
    f_vis[3] = tau13 
    #FIXME
    f_vis[4] = v[0]*tau11 + v[1]*tau12 + v[2]*tau13 + Pr*mu*T_x

    #y Viscous flux
    tau21 = mu*(u_y + v_x)
    tau22 = mu*(2*v_y - (2.0/3.0)*div)
    tau23 = mu*(v_z + w_y)
    g_vis    =  [0, 0, 0, 0, 0]
    g_vis[0] = 0 
    g_vis[1] = tau21 
    g_vis[2] = tau22 
    g_vis[3] = tau23 
    #FIXME
    g_vis[4] = v[0]*tau21 + v[1]*tau22 + v[2]*tau23 + Pr*mu*T_y

    #z Viscous flux
    tau31 = mu*(w_x + u_z)
    tau32 = mu*(w_y + v_z)
    tau33 = mu*(2*w_z - (2.0/3.0)*div)
    h_vis    =  [0, 0, 0, 0, 0]
    h_vis[0] = 0 
    h_vis[1] = tau31 
    h_vis[2] = tau32 
    h_vis[3] = tau33 
    #FIXME
    h_vis[4] = v[0]*tau31 + v[1]*tau32 + v[2]*tau33 + Pr*mu*T_z

    #x derivative of viscous flux
    f_vis_x =  [0, 0, 0, 0, 0]
    for i, f_i in enumerate(f_vis):
        f_vis_x[i] = smp.diff(f_vis[i], x[0])

    #y derivative of viscous flux
    g_vis_y =  [0, 0, 0, 0, 0]
    for i, g_i in enumerate(g_vis):
        g_vis_y[i] = smp.diff(g_vis[i], x[1])

    #z derivative of viscous flux
    h_vis_z =  [0, 0, 0, 0, 0]
    for i, h_i in enumerate(h_vis):
        h_vis_z[i] = smp.diff(h_vis[i], x[2])

    #Residual
    resi =  [0, 0, 0, 0, 0]
    for i, resi_i in enumerate(resi):
        resi[i] = (u_t[i] + f_inv_x[i] + g_inv_y[i] + h_inv_z[i]
                          - f_vis_x[i] - g_vis_y[i] - h_vis_z[i] )

    for i, resi_i in enumerate(resi):
#        print(resi_i.evalf(subs = {t:0, x[0]:0, x[1]:0}))
        print(resi_i.simplify())


def get_cns_2d(x, t, gamma, R, mu, Pr, u): 
    Cv     = R/(gamma - 1)
    v      = [0, 0] #Velocity
    rho    = u[0]
    v[0]   = u[1]/rho
    v[1]   = u[2]/rho

    T      = (u[3]/rho - 0.5*(v[0]*v[0] + v[1]*v[1]))/Cv
    p      = rho*R*T

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
    f_vis[1] = tau11 
    f_vis[2] = tau12 
    #FIXME
    f_vis[3] = v[0]*tau11 + v[1]*tau12 + Pr*mu*T_x

    #y Viscous flux
    tau21 = mu*(u_y + v_x)
    tau22 = mu*(2*v_y - (2.0/3.0)*div)
    g_vis    =  [0, 0, 0, 0]
    g_vis[0] = 0 
    g_vis[1] = tau21 
    g_vis[2] = tau22 
    #FIXME
    g_vis[3] = v[0]*tau21 + v[1]*tau22 + Pr*mu*T_y

    #x derivative of viscous flux
    f_vis_x =  [0, 0, 0, 0]
    for i, f_i in enumerate(f_vis):
        f_vis_x[i] = smp.diff(f_vis[i], x[0])

    #y derivative of viscous flux
    g_vis_y =  [0, 0, 0, 0]
    for i, g_i in enumerate(g_vis):
        g_vis_y[i] = smp.diff(g_vis[i], x[1])


    #Residual
    resi =  [0, 0, 0, 0]
    for i, resi_i in enumerate(resi):
        resi[i] = (u_t[i] + f_inv_x[i] + g_inv_y[i]
                          - f_vis_x[i] - g_vis_y[i] )

    for i, resi_i in enumerate(resi):
#        print(resi_i.evalf(subs = {t:0, x[0]:0, x[1]:0}))
        print(resi_i.simplify())



def smooth_density():
    '''
    Create a manufactured solution for compressbile NS
    '''
    x = smp.DeferredVector('x')
    t = smp.Symbol('t')

    dimension = 2

    v     = (1, -0.5         )#Velocity
    p     = 1                 #Pressure
    gamma = 1.4               
    mu    = 0.001             #Dynamic viscosity  
    Pr    = 0.72              #Prandtl number
    R     = 287               #Gas constant
    Cv    = R/(gamma - 1)     #Gas constant

    rho   = 1 + 0.2*smp.sin(smp.pi*(x[0] + x[1] - t*(v[0] + v[1])))

    #Conserved variables
    u =  [0, 0, 0, 0]
    u[0] = rho
    u[1] = u[0]*v[0]
    u[2] = u[0]*v[1]
    u[3] = p/(gamma - 1) + 0.5*u[0]*(v[0]*v[0] + v[1]*v[1]) 

    get_cns_2d(x, t, gamma, R, mu, Pr, u)


def hifiles_cns_2d():
    '''
    Create a manufactured solution for compressbile NS
    '''
    x = smp.DeferredVector('x')
    t = smp.Symbol('t')

    dimension = 2

    gamma = 1.4               
    mu    = 0.001             #Dynamic viscosity  
    Pr    = 0.72              #Prandtl number
    R     = 287               #Gas constant
    Cv    = R/(gamma - 1)     #Gas constant

    a     = 3.0
    k     = smp.pi
    om    = smp.pi

    #Conserved variables
    u    =  [0, 0, 0, 0]
    u[0] =  smp.sin(k*(x[0] + x[1]) - om*t) + a
    u[1] =  smp.sin(k*(x[0] + x[1]) - om*t) + a
    u[2] =  smp.sin(k*(x[0] + x[1]) - om*t) + a
    u[3] = (smp.sin(k*(x[0] + x[1]) - om*t) + a)**2

    get_cns_2d(x, t, gamma, R, mu, Pr, u)

def hifiles_cns_3d():
    '''
    Create a manufactured solution for compressbile NS
    '''
    x = smp.DeferredVector('x')
    t = smp.Symbol('t')

    dimension = 2

    gamma = 1.4               
    mu    = 0.001             #Dynamic viscosity  
    Pr    = 0.72              #Prandtl number
    R     = 287               #Gas constant
    Cv    = R/(gamma - 1)     #Gas constant

    a     = 3.0
    k     = smp.pi
    om    = smp.pi

    #Conserved variables
    u    =  [0, 0, 0, 0, 0]
#    u[0] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
#    u[1] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
#    u[2] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
#    u[3] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
#    u[4] = (smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a)**2
    u[0] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
    u[1] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
    u[2] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
    u[3] =  smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a
    u[4] = (smp.sin(k*(x[0] + x[1] + x[2]) - om*t) + a)**2


    get_cns_3d(x, t, gamma, R, mu, Pr, u)


def problem_mms():
#    smooth_density()
    hifiles_cns_3d()



if __name__=="__main__":
    problem_mms()
