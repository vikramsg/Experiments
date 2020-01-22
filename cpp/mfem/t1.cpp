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
//   const char *mesh_file = "../data/inline-tri.mesh";
   int order = 0;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim  = mesh->Dimension();
   int nele = mesh->GetNE();

   // 4. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   DG_FECollection fec(order, dim);

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   FiniteElementSpace *dg_fes  = new FiniteElementSpace(mesh, &fec, dim);
   
   bool node_pres = true;
   GridFunction *coord = mesh->GetNodes();
   const FiniteElementSpace *cspace = mesh->GetNodalFESpace();
   if (!cspace)
       node_pres = false;

   GridFunction ccn_gf(dg_fes);

   Array<int> vdofs;
   Vector el_vert; 
   Vector A(dim), B(dim), C(dim);
   Vector ccn; 
   for (int i = 0; i < mesh->GetNE(); i++)
   {
     if (node_pres)
     {
       cspace->GetElementVDofs(i, vdofs);
       coord->GetSubVector(vdofs, el_vert);
     }
     else
     {
       mesh->GetElementVertices(i, vdofs); 
       el_vert.SetSize(2*vdofs.Size());
       for(int j = 0; j < vdofs.Size(); j++)
       {
           double *coord = mesh->GetVertex(vdofs[j]);
           el_vert[               j] = coord[0];
           el_vert[vdofs.Size() + j] = coord[1];

       }
     }

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

   // RT dofs include size of normal
   VectorFunctionCoefficient u0(dim, init_function);
   GridFunction u_sol(R_space);
   u_sol.ProjectCoefficient(u0);

   const FiniteElement *el;
   double vol = 0.;
   ElementTransformation *eltrans;
   Vector area(dg_fes->GetVSize()/dim);
   for (int i = 0; i < mesh->GetNE(); i++)
   {
       ElementTransformation *T = dg_fes->GetElementTransformation(i);
       el = dg_fes->GetFE(i);

       const IntegrationRule *ir ;
       int   elorder;

       elorder = 2*el->GetOrder() + 1;
       ir    = &IntRules.Get(el->GetGeomType(), elorder);

       double ar = 0.0;
       for (int p = 0; p < ir->GetNPoints(); p++)
       {
           const IntegrationPoint &ip = ir->IntPoint(p);
           T->SetIntPoint(&ip);

           ar += ip.weight*T->Weight();
       }
       area[i] = ar;
       vol    += ar;

   }
 
   Vector loc(dim), nor(dim), nor_dim(dim);
   const IntegrationRule *ir ;
   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;
   IntegrationPoint eip;
   
   int nfaces = mesh->GetNumFaces();
     
   Vector vn;
   Vector cc_vec(dim);
   GridFunction sc_gf(dg_fes);
   sc_gf = 0.0; 
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
       T->Elem1->Transform(eip, loc);

       T->Face->SetIntPoint(&ip); 
       CalcOrtho(T->Face->Jacobian(), nor);
       double nor_l2 = nor.Norml2(); 
       nor_dim.Set(1/nor_l2, nor);

       R_space->GetFaceDofs(i, vdofs);
       u_sol.GetSubVector(vdofs, vn);
//       cout << i << "\t" << vn[0] << "\t" << vdofs[0] << endl;

       dg_fes->GetElementVDofs(T->Elem1No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);

       double dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       double nx = nor_dim[0]*dist;
       double ny = nor_dim[1]*dist;
       
//       cout << sc_gf[0] << endl;
       sc_gf[vdofs[0]] += vn[0]*nx/area[vdofs[0]];
       sc_gf[vdofs[1]] += vn[0]*ny/area[vdofs[0]];
//       cout << sc_gf[0] << endl;

//       cout << vdofs[0] << "\t" << loc[0] - ccn[0] << "\t" << vn[0] << "\t" << area[vdofs[0]]<< endl;
//       cout << vdofs[0] << "\t" << loc[0]  << "\t" << ccn[0] << endl;

       T->Loc2.Transform(ip, eip);
       T->Elem2->Transform(eip, loc);

       dg_fes->GetElementVDofs(T->Elem2No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);

       dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       nx = nor_dim[0]*dist;
       ny = nor_dim[1]*dist;
 
       sc_gf[vdofs[0]] += vn[0]*nx/area[vdofs[0]];
       sc_gf[vdofs[1]] += vn[0]*ny/area[vdofs[0]];
       
//       cout << vdofs[0] << "\t" << loc[0]  << "\t" << ccn[0] << endl;

       if ( (T->Elem1No == 0) || (T->Elem2No == 0) )
       {
//           cout << i << "\t" << vn[0] << "\t" << ccn[0] << "\t" << loc[0] << endl;
//           cout << i << "\t" << vn[0] << "\t" << loc[0] << "\t" << loc[1] << endl;
//           cout << i << "\t" <<  nor[0] << "\t" << nor[1] << endl;
       }
     }
//     cout << sc_gf[0] << endl;

   }

   // FIXME: Reconstruction works now, need to vary angles and check values
   // Need to implement the flux reconstruction to get divergence
   // Then we do the other way of doing P^T and then do the same steps
   // There are aleady errors with the approximately equilateral triangles
   // Is that fixed with the other reconstruction
//   for(int j = 0; j < nfaces; j++)
//       cout << j << "\t" << u_sol[j] << endl; 
   for(int j = 0; j < nele; j++)
   {
//       cout << sc_gf[j] << "\t" << sc_gf[nele + j] << endl;
   }
//   sc_gf.Print();

   
   GridFunction flux(R_space);
   Array<int> rdofs;
   vn.SetSize(0);
   double dist_tot;

   for (int i = 0; i < nfaces; i++)
   {
     T = mesh->GetInteriorFaceTransformations(i);

     el1 = R_space->GetFE(T -> Elem1No);
     el2 = R_space->GetFE(T -> Elem2No);

     ir = &IntRules.Get(T->FaceGeom, 2*order);
     dist_tot = 0.;

     for (int j = 0; j < ir->GetNPoints(); j++)
     {
       const IntegrationPoint &ip = ir->IntPoint(j);
       
       T->Loc1.Transform(ip, eip);
       T->Elem1->Transform(eip, loc);

       T->Face->SetIntPoint(&ip); 
       CalcOrtho(T->Face->Jacobian(), nor);
       double nor_l2 = nor.Norml2(); 
       nor_dim.Set(1/nor_l2, nor);

       R_space->GetFaceDofs(i, rdofs);

       dg_fes->GetElementVDofs(T->Elem1No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);
       sc_gf. GetSubVector(vdofs, cc_vec);

       double dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       double nx = nor[0]*dist;
       double ny = nor[1]*dist;
       dist_tot += dist;

       vn[0] = cc_vec[0]*nx + cc_vec[1]*ny;
       
       T->Loc2.Transform(ip, eip);
       T->Elem2->Transform(eip, loc);

       dg_fes->GetElementVDofs(T->Elem2No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);
       sc_gf. GetSubVector(vdofs, cc_vec);

       dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       nx = nor[0]*dist;
       ny = nor[1]*dist;
       dist_tot += dist;
 
       vn[0] = vn[0] + cc_vec[0]*nx + cc_vec[1]*ny;
       vn[0] = vn[0]/dist_tot;
       flux.SetSubVector(rdofs, vn);
     }
 
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
       v(0) = 1.0; 
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









