#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


// Cosine of angle ABC
double cosine(Vector &A, Vector &B, Vector &C);

// Cartesian coordinates of the point whose barycentric coordinates
// with respect to the triangle ABC are [p,q,r]
void barycentric(Vector &A, Vector &B, Vector &C, double &p, double &q, double &r, Vector &bcn);

// Cartesian coordinates of the point whose trilinear coordinates
// with respect to the triangle ABC are [alpha,beta,gamma]
void trilinear(Vector &A, Vector &B, Vector &C, double &alpha, double &beta, double &gamma, Vector &bcn);

// Cartesian coordinates of the circumcenter of triangle ABC
void circumcenter(Vector &A, Vector &B, Vector &C);


int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "t1_per.mesh";
   int order = 0;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   DG_FECollection fec(order, dim);

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   FiniteElementSpace *dg_fes  = new FiniteElementSpace(mesh, &fec);
   
   GridFunction *coord = new GridFunction(dg_fes);

   DenseMatrix pointmat;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
     mesh->GetPointMatrix(i, pointmat);
   }

   FaceElementTransformations *T;
   int nfaces = mesh->GetNumFaces();
   for (int i = 0; i < nfaces; i++)
   {
     T = mesh->GetInteriorFaceTransformations(i);
   }


   delete coord;
   delete R_space;
   delete dg_fes;
   delete hdiv_coll;
   delete mesh;

   return 0;
}


// Cosine of angle ABC
double cosine(Vector &A, Vector &B, Vector &C)
{
    double a = A.Norml2();
    double b = B.Norml2();
    double c = C.Norml2();
    return (a*a+c*c-b*b)/(2.*a*c);
}

// Cartesian coordinates of the point whose barycentric coordinates
// with respect to the triangle ABC are [p,q,r]
void barycentric(Vector &A, Vector &B, Vector &C, double &p, double &q, double &r, Vector &bcn)
{
    int n = A.Size();
    MFEM_ASSERT( B.Size() == C.Size() , "");
    MFEM_ASSERT( B.Size() == n , "");
    double s = p+q+r;
    p = p/s;
    q = q/s;
    r = r/s;

    bcn.SetSize(n);
    add(p, A, q, B, bcn);
    bcn.Add(r, C);
}


// Cartesian coordinates of the point whose trilinear coordinates
// with respect to the triangle ABC are [alpha,beta,gamma]
void trilinear(Vector &A, Vector &B, Vector &C, double &alpha, double &beta, double &gamma, Vector &bcn)
{
    double a = A.Norml2()*alpha;
    double b = B.Norml2()*beta;
    double c = C.Norml2()*gamma;

    barycentric(A, B, C, a, b, c, bcn);
}

               
// Cartesian coordinates of the circumcenter of triangle ABC
void circumcenter(Vector &A, Vector &B, Vector &C)
{
    double cosA = cosine(C,A,B);
    double cosB = cosine(A,B,C);
    double cosC = cosine(B,C,A);

    Vector ccn;
    trilinear(A,B,C,cosA,cosB,cosC, ccn);
}









