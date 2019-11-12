import numpy as np

def TDMA(a,b,c,d):
    '''
    Copied verbatim from
    https://stackoverflow.com/a/43214907
    '''
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)

    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p


def diff(n, dt_inv):
    '''
    Create a prototype heat equation matrix with
    zero Neumann boundary conditions
    '''
    z     = 200.; mu = 0.001
    h     = z/n
    a     = -1*mu*np.ones(n)/(h*h)
    c     = -1*mu*np.ones(n)/(h*h)
    b     = - a - c + dt_inv; b[0] = dt_inv - a[0] 
    b[-1] = dt_inv - c[-1] 

    d = np.zeros(n); d[11] = 1.0
    d = d*dt_inv

    print(TDMA(a, b, c, d))
   



if __name__=="__main__":
    n = 3
    a = -1*np.ones(n)
    b =  2*np.ones(n)
    c = -1*np.ones(n)

    d = np.zeros(n); d[1] = 1.0

#    print(TDMA(a, b, c, d))

    diff(20, 0.001)
