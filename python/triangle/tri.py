import math
import numpy as np

def lengthSquare(X, Y): 
   xDiff = X[0] - Y[0]
   yDiff = X[1] - Y[1]
   return xDiff*xDiff + yDiff*yDiff; 
 
  
def normals(verts):
    nor = np.zeros( (3, 2) )

    nor[0] = [ verts[1, 1] - verts[0, 1], -(verts[1, 0] - verts[0, 0]) ]
    nor[0] = nor[0]/np.linalg.norm( nor[0] )
    nor[1] = [ verts[2, 1] - verts[1, 1], -(verts[2, 0] - verts[1, 0]) ]
    nor[1] = nor[1]/np.linalg.norm( nor[1] )
    nor[2] = [ verts[0, 1] - verts[2, 1], -(verts[0, 0] - verts[2, 0]) ]
    nor[2] = nor[2]/np.linalg.norm( nor[2] )

    return nor

def getLengths(verts):
    a2 = lengthSquare(verts[1], verts[2]); 
    b2 = lengthSquare(verts[0], verts[2]); 
    c2 = lengthSquare(verts[0], verts[1]); 
  
    a = math.sqrt(a2); 
    b = math.sqrt(b2); 
    c = math.sqrt(c2); 

    return [a, b, c]
 
def getAngles(verts):
    [a, b, c] = getLengths(verts)

    a2 = a*a; b2 = b*b; c2 = c*c

    alpha = math.acos((b2 + c2 - a2)/(2*b*c)); 
    betta = math.acos((a2 + c2 - b2)/(2*a*c)); 
    gamma = math.acos((a2 + b2 - c2)/(2*a*b)); 
  
    alpha = alpha * 180 / math.pi; 
    betta = betta * 180 / math.pi; 
    gamma = gamma * 180 / math.pi; 

    return [ alpha, betta, gamma ]
  
def getArea(verts):
    [a, b, c] = getLengths(verts)
    p = 0.5*(a + b + c)

    return math.sqrt(p*(p - a)*(p - b)*(p - c))

# Cosine of angle ABC
def cosine(A,B,C):
    verts = np.zeros( (3, 2) )
    verts[0] = A; verts[1] = B; verts[2] = C
    [a, b, c] = getLengths(verts) 
    return (a*a+c*c-b*b)/(2*a*c)

# Cartesian coordinates of the point whose barycentric coordinates
# with respect to the triangle ABC are [p,q,r]
def barycentric(A,B,C,p,q,r):
    n = len(A)
    assert len(B) == len(C) == n
    s = p+q+r
    p, q, r = p/s, q/s, r/s
    return tuple([p*A[i]+q*B[i]+r*C[i] for i in xrange(n)])

# Cartesian coordinates of the point whose trilinear coordinates
# with respect to the triangle ABC are [alpha,beta,gamma]
def trilinear(A,B,C,alpha,beta,gamma):
    verts = np.zeros( (3, 2) )
    verts[0] = A; verts[1] = B; verts[2] = C
    [a, b, c] = getLengths(verts) 
 
    return barycentric(A,B,C,a*alpha,b*beta,c*gamma)
               
# Cartesian coordinates of the circumcenter of triangle ABC
def circumcenter(verts):
    A = verts[0]; B = verts[1]; C = verts[2]

    cosA = cosine(C,A,B)
    cosB = cosine(A,B,C)
    cosC = cosine(B,C,A)

    return trilinear(A,B,C,cosA,cosB,cosC)

#Get value at location r, given vals at dofs
def getValue(verts, vals, r):
    [b, c, a] = getLengths(verts) 
    ar        = getArea(verts)
    
    A  = verts[0]; B = verts[1]; C = verts[2]

    # RT basis function
    # https://en.wikipedia.org/wiki/Raviart%E2%80%93Thomas_basis_functions
    b1 = 0.5*(a/ar)*( r - C ) 
    b2 = 0.5*(b/ar)*( r - A ) 
    b3 = 0.5*(c/ar)*( r - B ) 

    val = vals[0]*b1 + vals[1]*b2 + vals[2]*b3

    print(val)



if __name__=="__main__":
    verts = np.zeros( (3, 2) )
    verts[0] = [0., 0.] 
    verts[1] = [1., 0.]
#    verts[2] = [0.5, np.sqrt(3)*0.5]
    verts[2] = [0.7, 1.]
    
    ns = normals(verts)
    an = getAngles(verts)
    cn = circumcenter(verts)

#    print(ns)
#    print(an)
#    print(cn)

    const = [0.0, 1.0]
    vals  = [ np.dot(const, ns[0]), np.dot(const, ns[1]), np.dot(const, ns[2]) ]

    r    = [0.5, 0.3]  
    getValue(verts, vals, cn)























