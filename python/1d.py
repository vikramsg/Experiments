import numpy as np
import scipy as sp

def ptp():
    a    = np.zeros( [5, 5] )
    a[0] = np.array( [0.5, 0.5, 0. , 0. , 0. ] )
    a[1] = np.array( [0. , 0.5, 0.5, 0. , 0. ] )
    a[2] = np.array( [0. , 0. , 0.5, 0.5, 0. ] )
    a[3] = np.array( [0. , 0. , 0. , 0.5, 0.5] )
    a[4] = np.array( [0.5, 0. , 0. , 0. , 0.5] )
    
    b    = np.zeros( [5, 5] )
    b[0] = np.array( [0.5, 0. , 0. , 0. , 0.5 ] )
    b[1] = np.array( [0.5, 0.5, 0. , 0. , 0.  ] )
    b[2] = np.array( [0. , 0.5, 0.5, 0. , 0.  ] )
    b[3] = np.array( [0. , 0. , 0.5, 0.5, 0.  ] )
    b[4] = np.array( [0. , 0. , 0. , 0.5, 0.5 ] )

    ptp = np.dot(b, a) 

    minv = np.linalg.inv(ptp)
 
    d    = np.zeros( [5, 5] )
    d[0] = np.array( [ 1. , 0.  , 0.  , 0.  ,-1.  ] )
    d[1] = np.array( [-1. , 1.  , 0.  , 0.  , 0.  ] )
    d[2] = np.array( [ 0. ,-1.  , 1.  , 0.  , 0.  ] )
    d[3] = np.array( [ 0. , 0.  ,-1.  , 1.  , 0.  ] )
    d[4] = np.array( [ 0. , 0.  , 0.  ,-1.  , 1.  ] )

    return ( np.dot(d, a) )   


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def ker():
    m  = ptp()

    d    = np.zeros( [5, 5] )
    d[0] = np.array( [ 2. , 2.  , 0.  , 0.  , 0.  ] )
    d[1] = np.array( [-1. , 0.  , 1.  , 0.  , 0.  ] )
    d[2] = np.array( [ 0. ,-1.  , 0.  , 1.  , 0.  ] )
    d[3] = np.array( [ 0. , 0.  ,-1.  , 0.  , 1.  ] )
    d[4] = np.array( [ 0. , 0.  , 0.  ,-2.  ,-2.  ] )

    ns = nullspace(d)

    print(ns)



if __name__=="__main__":
    ker()



