import sympy as smp

def get_euler_1d(x, t, gamma, R, mu, Pr, u): 
    Cv     = R/(gamma - 1)
    Cp     = Cv + R 
    v      = [0] #Velocity
    rho    = u[0]
    v[0]   = u[1]/rho

    T      = (u[2]/rho - 0.5*(v[0]*v[0]))/Cv
    p      = rho*R*T

    ##########################
    #Inviscid
    ##########################
    #x Inviscid flux
    f_inv =  [0, 0, 0]
    f_inv[0] = rho*v[0]
    f_inv[1] = rho*v[0]*v[0] + p
    f_inv[2] = (u[2] + p)*v[0] 

    #Time derivative
    u_t =  [0, 0, 0]
    for i, u_i in enumerate(u):
        u_t[i] = smp.diff(u[i], t)

    #x derivative of inviscid flux
    f_inv_x =  [0, 0, 0]
    for i, f_i in enumerate(f_inv):
        f_inv_x[i] = smp.diff(f_inv[i], x[0])

    #Residual
    resi =  [0, 0, 0]
    for i, resi_i in enumerate(resi):
        resi[i] = (u_t[i] + f_inv_x[i]) 

    for i, resi_i in enumerate(resi):
#        print(resi_i.evalf(subs = {t:0, x[0]:0.55635083268962915}))
        print(resi_i.simplify())


def hifiles_cns_1d():
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

    a     = 9.0
    k     = smp.pi
    om    = smp.pi

    #Conserved variables
    u    =  [0, 0, 0]
#    u[0] =  smp.sin(k*(x[0]) - om*t) + a
#    u[1] =  smp.sin(k*(x[0]) - om*t) + a
#    u[2] = (smp.sin(k*(x[0]) - om*t) + a)**2
    u[0] =  smp.sin(k*(x[0]) - om*t) + a
    u[1] =  smp.sin(k*(x[0]) - om*t) + a
    u[2] = (smp.sin(k*(x[0]) - om*t) + a)**2


    get_euler_1d(x, t, gamma, R, mu, Pr, u)


def problem_mms():
    hifiles_cns_1d()



if __name__=="__main__":
    problem_mms()
