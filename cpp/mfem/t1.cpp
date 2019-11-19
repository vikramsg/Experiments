#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v);

void getLengths(Vector &A, Vector &B, Vector &C, double &a, double &b, double &c);

// Cosine of angle ABC
double cosine(Vector &A, Vector &B, Vector &C);

// Cartesian coordinates of the point whose barycentric coordinates
// with respect to the triangle ABC are [p,q,r]
void barycentric(Vector &A, Vector &B, Vector &C, double &p, double &q, double &r, Vector &bcn);

// Cartesian coordinates of the point whose trilinear coordinates
// with respect to the triangle ABC are [alpha,beta,gamma]
void trilinear(Vector &A, Vector &B, Vector &C, double &alpha, double &beta, double &gamma, Vector &bcn);

// Cartesian coordinates of the circumcenter of triangle ABC
void circumcenter(Vector &A, Vector &B, Vector &C, Vector &ccn);


int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "para.mesh";
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
   FiniteElementSpace *dg_fes  = new FiniteElementSpace(mesh, &fec, dim);
   
   GridFunction *coord = mesh->GetNodes();
   const FiniteElementSpace *cspace = mesh->GetNodalFESpace();
   
   GridFunction ccn_gf(dg_fes);

   Array<int> vdofs;
   Vector el_vert; 
   Vector A(dim), B(dim), C(dim);
   Vector ccn; 
   for (int i = 0; i < mesh->GetNE(); i++)
   {
     cspace->GetElementVDofs(i, vdofs);
     coord->GetSubVector(vdofs, el_vert);
     int ndof = el_vert.Size()/dim;

     for (int j = 0; j < dim; j++)
     {
       A[j] = el_vert(j*ndof + 0); 
       B[j] = el_vert(j*ndof + 1); 
       C[j] = el_vert(j*ndof + 2); 
     }

     circumcenter(A, B, C, ccn);
     
     dg_fes->GetElementVDofs(i, vdofs);
     ccn_gf.SetSubVector(vdofs, ccn);
   }

   Vector loc(dim), nor(dim);
   const IntegrationRule *ir ;
   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;
   IntegrationPoint eip;
   
   int nfaces = mesh->GetNumFaces();
   for (int i = 0; i < nfaces; i++)
   {
     T = mesh->GetInteriorFaceTransformations(i);

     el1 = R_space->GetFE(T -> Elem1No);
     el2 = R_space->GetFE(T -> Elem2No);

     ir = &IntRules.Get(T->FaceGeom, 2*order);

     for (int j = 0; j < ir->GetNPoints(); j++)
     {
       const IntegrationPoint &ip = ir->IntPoint(j);
       
       T->Loc1.Transform(ip, eip);
       T->Elem2->Transform(eip, loc);

       T->Face->SetIntPoint(&ip); 
       CalcOrtho(T->Face->Jacobian(), nor);
       
     }

   }
   
   VectorFunctionCoefficient u0(dim, init_function);
   GridFunction u_sol(R_space);
   u_sol.ProjectCoefficient(u0);

   ElementTransformation *eltrans;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
     eltrans = mesh->GetElementTransformation(i);
     double div = u_sol.GetDivergence(*eltrans);
   }

   delete R_space;
   delete dg_fes;
   delete hdiv_coll;
   delete mesh;

   return 0;
}



//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       v(0) = 0.0; 
       v(1) = 0.0; 
   }
}




void getLengths(Vector &A, Vector &B, Vector &C, double &a, double &b, double &c)
{
    Vector temp(A.Size());
    subtract(B, C, temp);
    a = temp.Norml2();
    
    subtract(A, C, temp);
    b = temp.Norml2();
    
    subtract(A, B, temp);
    c = temp.Norml2();
}
 

// Cosine of angle ABC
double cosine(Vector &A, Vector &B, Vector &C)
{
    double a, b, c;
    getLengths(A, B, C, a, b, c);
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
    double a, b, c;
    getLengths(A, B, C, a, b, c);
    a = a*alpha;
    b = b*beta ;
    c = c*gamma;

    barycentric(A, B, C, a, b, c, bcn);
}

               
// Cartesian coordinates of the circumcenter of triangle ABC
void circumcenter(Vector &A, Vector &B, Vector &C, Vector &ccn)
{
    double cosA = cosine(C,A,B);
    double cosB = cosine(A,B,C);
    double cosC = cosine(B,C,A);

    trilinear(A,B,C,cosA,cosB,cosC, ccn);
}









