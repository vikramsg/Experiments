#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


//  Initialize velocity 
void init_function(const Vector &x, Vector &v);

// Initialize scalar
void init_scalar(const int &nele, const Vector &x_ccn, Vector &sc);

// Get velocity reconstruction
void getPv(Mesh &mesh, FiniteElementSpace &R_space, FiniteElementSpace &dg_fes, int &order, 
    Vector &u_sol, const Vector &ccn_gf, const Vector &area, Vector &vel_gf);

// Get P^T V 
void getPTransposev(Mesh &mesh, FiniteElementSpace &R_space, FiniteElementSpace &dg_fes, int &order, 
    Vector &vel_gf, const Vector &ccn_gf, Vector &flux);


// Get circumcenters of the mesh
void getCircumCenter(Mesh &mesh, FiniteElementSpace &dg_fes, Vector &ccn_gf);

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

void writeData(const int &cycle, const double &time, FiniteElementSpace &dg_fes, GridFunction &u);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
  Vector area;
  GridFunction ccn_gf;
  Mesh *mesh; 
  FiniteElementSpace &R_space;
  FiniteElementSpace &dg_fes;
  GridFunction &vel_n;
public:
  FE_Evolution(FiniteElementSpace &R_space_, FiniteElementSpace &dg_fes_, GridFunction &vel_n_);

  virtual void Mult(const Vector &x, Vector &y) const;

  virtual ~FE_Evolution() { }
};

void Step(const int &ode_solver_type, const FE_Evolution &adv, 
            const double &dt, double &t, 
            GridFunction &u);

int order             = 0; int np = order + 1;

int main(int argc, char *argv[])
{
   const char *mesh_file = "para_30.mesh";
   int ref_levels        = 2;
   int ode_solver_type   = 2;
   double t_final        = 10. ;
   double dt             = 0.01 ;
   bool visualization    = true;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   DG_FECollection fec(order, dim);

   // Velocity field
   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   // Reconstructed velocity filed at circumcenter
   FiniteElementSpace *dg_fes  = new FiniteElementSpace(mesh, &fec, dim);
   // Scalar field for advection
   FiniteElementSpace *sc_fes  = new FiniteElementSpace(mesh, &fec);
 
   GridFunction ccn_gf(dg_fes);
   getCircumCenter(*mesh, *dg_fes, ccn_gf);

   int nele = mesh->GetNE();

   // RT dofs include size of normal
   VectorFunctionCoefficient u0(dim, init_function);
   GridFunction vel_n(R_space);
   vel_n.ProjectCoefficient(u0);

   GridFunction scalar(sc_fes), rhs(sc_fes);
   init_scalar(nele, ccn_gf, scalar);

   Vector area(nele); double tot_area = 0.;
   for (int i = 0; i < nele; i++)
   {
       area[i]  = mesh->GetElementVolume(i);
       tot_area = tot_area + area[i]; 
   }
 
   Vector vel_gf(ccn_gf.Size());
   getPv(*mesh, *R_space, *dg_fes, order, vel_n, ccn_gf, area, vel_gf);
   
   for(int j = 0; j < nele; j++)
   {
     vel_gf[j]        *= scalar[j];
     vel_gf[nele + j] *= scalar[j];
   }

   GridFunction flux(R_space);
   getPTransposev(*mesh, *R_space, *dg_fes, order, vel_gf, ccn_gf, flux);

   double error = 0.;
   Vector div(scalar.Size());
   for (int i = 0; i < nele; i++)
   {
       ElementTransformation *T = dg_fes->GetElementTransformation(i);
       div[i] = flux.GetDivergence(*T);
       
       double x_ccn = ccn_gf[i]; 
       double y_ccn = ccn_gf[nele + i];

       double ex = 0.; double fn = 0.;
       {
//         fn = exp( -50.*(x_ccn - 0.8)*(x_ccn - 0.8));
//         ex = (-100.*x_ccn + 100*0.8)*exp( -50.*(x_ccn - 0.8)*(x_ccn - 0.8));
         fn = sin(4.*M_PI* (x_ccn - 0.25) );
         ex = 4.*M_PI*cos(4.*M_PI* (x_ccn - 0.25) );
//         fn = 2.*sin(2.*M_PI* (x_ccn - 0.25) )*cos(2.*M_PI* ( x_ccn ) );
//         ex = 2.*M_PI*cos(2.*M_PI* (x_ccn - 0.25) )*2.*cos(2.*M_PI* ( x_ccn ) ) - 
//              2.*M_PI*sin(2.*M_PI* (x_ccn - 0.25) )*2.*sin(2.*M_PI* ( x_ccn ) ) ;


         error += pow( div[i] - ex, 2 )*area[i];

//          cout << i << "\t" << x_ccn  << "\t" << y_ccn << "\t" << div[i]  << "\t" << ex << "\t" << endl;
//          cout << i << "\t" << div[i]  << "\t" << ex << "\t" << endl;
//          cout << i  << "\t" << x_ccn << "\t" << y_ccn << "\t" << scalar[i]  << "\t" << fn << "\t" << endl;
       }
   }

   cout << "h is " << sqrt(area[0]) << endl;
   cout << "Error is " << error << endl;
 
   return 0;
}


void Step(const int &ode_solver_type, const FE_Evolution &adv, 
            const double &dt, double &t, 
            GridFunction &u)
{
    Vector u_t(u.Size()), k_t(u.Size()), y_t(u.Size());
    GridFunction eta(u);
    Vector v(u.Size());
    Vector u_init(u.Size());

    u_init = u;
    u_t    = u;

    double eps = 0., error = 0., denom = 0.;
    double adpt_err = 0.;

    double dt_init = dt;

    if (ode_solver_type == 1)
    {
        adv.Mult(u_t, k_t);
        add(u_t, dt, k_t, u);

    }
    else if (ode_solver_type == 2)
    {
        adv.Mult(u_t, k_t);

        add(u_t, dt, k_t, y_t);
        
        adv.Mult(y_t, k_t);

        add(1./2, u_t, 1./2, y_t, u);
        add(u, dt/2., k_t, u_t);

        u = u_t;
    }

    t += dt_init;
}




void writeData(const int &cycle, const double &time, FiniteElementSpace &dg_fes, GridFunction &u)
{
    FiniteElementSpace *fes = u.FESpace();
    Mesh *mesh              = fes->GetMesh();
  
    int dim  = mesh->Dimension();
    int nele = mesh->GetNE();

    GridFunction nodes(&dg_fes);
    mesh->GetNodes(nodes);

    std::ostringstream oss;
    oss << std::setw(8) << std::setfill('0') << cycle;

    std::string root_name = "Wave_"  + oss.str() + ".dat";
    std::ofstream root_file(root_name.c_str());
 
    for (int i = 0; i < nele; i++)
    {
      if (abs(nodes[nele + i] - 0.28) < .03)
        root_file << nodes[i] << '\t' << u[i] << endl;
    }

    root_file.close();
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace &R_space_, FiniteElementSpace &dg_fes_, GridFunction &vel_n_)
   : TimeDependentOperator(), R_space(R_space_), dg_fes(dg_fes_), vel_n(vel_n_)
{
  mesh    = R_space.GetMesh();
  
  int dim  = mesh->Dimension();
  int nele = mesh->GetNE();

  double vol = 0.;
  area.SetSize(nele);

  for (int i = 0; i < nele; i++)
  {
      area[i] = mesh->GetElementVolume(i);
      vol    += area[i];
  }
  
  ccn_gf.SetSpace(&dg_fes);
  getCircumCenter(*mesh, dg_fes, ccn_gf);
}


void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   int nele = mesh->GetNE();

   Vector vel_gf(ccn_gf.Size());
   getPv(*mesh, R_space, dg_fes, order, vel_n, ccn_gf, area, vel_gf);
   
   for(int j = 0; j < nele; j++)
   {
     vel_gf[j]        *= x[j];
     vel_gf[nele + j] *= x[j];
   }

   GridFunction flux(&R_space);
   getPTransposev(*mesh, R_space, dg_fes, order, vel_gf, ccn_gf, flux);

   for (int i = 0; i < nele; i++)
   {
       ElementTransformation *T = dg_fes.GetElementTransformation(i);
       y[i] = -1.*flux.GetDivergence(*T);
   }
 
}



//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       v[0] = 1.;//2.*cos(2.*M_PI* (x[0] )); 
       v[1] = 0.0; 
   }
}


//  Initialize variables coefficient
void init_scalar(const int &nele, const Vector &x_ccn, Vector &sc)
{
   Vector x(2);
   for(int i = 0; i < nele; i++)
   {
     x[0] = x_ccn[i];
     x[1] = x_ccn[nele + i];

     double scalar = 0.;

//     scalar = exp( -50.*(x[0] - 0.8)*(x[0] - 0.8));

     scalar = sin(4.*M_PI* (x[0] - 0.25) );
 
     sc[i] = scalar;
   }

}




// Get reconstructed velocity
void getPv(Mesh &mesh, FiniteElementSpace &R_space, FiniteElementSpace &dg_fes, int &order, 
    Vector &u_sol, const Vector &ccn_gf, const Vector &area, Vector &vel_gf)
{
   int dim  = mesh.Dimension();

   Vector vn;
   Vector cc_vec(dim);
   Array<int> vdofs, rdofs;
   Vector loc(dim), nor(dim), nor_dim(dim), ccn(dim);
   
   const IntegrationRule *ir ;
   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;
   IntegrationPoint eip;
 
   int nfaces = mesh.GetNumFaces();

   vel_gf   = 0.0; 
   for (int i = 0; i < nfaces; i++)
   {
     T = mesh.GetInteriorFaceTransformations(i);

     el1 = R_space.GetFE(T -> Elem1No);
     el2 = R_space.GetFE(T -> Elem2No);

     ir = &IntRules.Get(T->FaceGeom, 2*order);
       
     R_space.GetFaceDofs(i, rdofs);
     u_sol.GetSubVector(rdofs, vn);

     for (int j = 0; j < ir->GetNPoints(); j++)
     {
       const IntegrationPoint &ip = ir->IntPoint(j);
       
       T->Loc1.Transform(ip, eip);
       T->Elem1->Transform(eip, loc);

       T->Face->SetIntPoint(&ip); 
       CalcOrtho(T->Face->Jacobian(), nor);
       double nor_l2 = nor.Norml2(); 
       nor_dim.Set(1/nor_l2, nor);

       dg_fes.GetElementVDofs(T->Elem1No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);

       double dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       double nx = nor_dim[0]*dist;
       double ny = nor_dim[1]*dist;
       
       vel_gf[vdofs[0]] += vn[0]*nx/area[vdofs[0]];
       vel_gf[vdofs[1]] += vn[0]*ny/area[vdofs[0]];

       T->Loc2.Transform(ip, eip);
       T->Elem2->Transform(eip, loc);

       dg_fes.GetElementVDofs(T->Elem2No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);

       dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       nx = nor_dim[0]*dist;
       ny = nor_dim[1]*dist;
 
       vel_gf[vdofs[0]] += vn[0]*nx/area[vdofs[0]];
       vel_gf[vdofs[1]] += vn[0]*ny/area[vdofs[0]];
     }
    
   }

}



// Get P^T V 
void getPTransposev(Mesh &mesh, FiniteElementSpace &R_space, FiniteElementSpace &dg_fes, int &order, 
    Vector &vel_gf, const Vector &ccn_gf, Vector &flux)
{
   int dim  = mesh.Dimension();

   Vector vn;
   Vector cc_vec(dim);
   Array<int> vdofs, rdofs;
   Vector loc(dim), nor(dim), nor_dim(dim), ccn(dim);
   
   const IntegrationRule *ir ;
   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;
   IntegrationPoint eip;
 
   int nfaces = mesh.GetNumFaces();

   vn.SetSize(1);
   double dist_tot;

   for (int i = 0; i < nfaces; i++)
   {
     T = mesh.GetInteriorFaceTransformations(i);

     el1 = R_space.GetFE(T -> Elem1No);
     el2 = R_space.GetFE(T -> Elem2No);

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

       R_space.GetFaceDofs(i, rdofs);

       dg_fes.GetElementVDofs(T->Elem1No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);
       vel_gf. GetSubVector(vdofs, cc_vec);

       double dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       double nx = nor[0]*dist;
       double ny = nor[1]*dist;
       dist_tot += dist;

       vn[0] = cc_vec[0]*nx + cc_vec[1]*ny;
       
       T->Loc2.Transform(ip, eip);
       T->Elem2->Transform(eip, loc);

       dg_fes.GetElementVDofs(T->Elem2No, vdofs);
       ccn_gf.GetSubVector(vdofs, ccn);
       vel_gf. GetSubVector(vdofs, cc_vec);

       dist = sqrt( pow(ccn[0] - loc[0], 2) + pow(ccn[1] - loc[1], 2) );
       nx = nor[0]*dist;
       ny = nor[1]*dist;
       dist_tot += dist;
 
       vn[0] = vn[0] + cc_vec[0]*nx + cc_vec[1]*ny;
       vn[0] = vn[0]/dist_tot;
       flux.SetSubVector(rdofs, vn);

     }
 
   }

}


void getCircumCenter(Mesh &mesh, FiniteElementSpace &dg_fes, Vector &ccn_gf)
{
   int dim  = mesh.Dimension();

   bool node_pres = true;
   GridFunction *coord = mesh.GetNodes();
   const FiniteElementSpace *cspace = mesh.GetNodalFESpace();
   if (!cspace)
       node_pres = false;

   Array<int> vdofs;
   Vector el_vert; 
   Vector A(dim), B(dim), C(dim);
   Vector ccn; 

   for (int i = 0; i < mesh.GetNE(); i++)
   {
     if (node_pres)
     {
       cspace->GetElementVDofs(i, vdofs);
       coord->GetSubVector(vdofs, el_vert);
     }
     else
     {
       mesh.GetElementVertices(i, vdofs); 
       el_vert.SetSize(2*vdofs.Size());
       for(int j = 0; j < vdofs.Size(); j++)
       {
           double *coord = mesh.GetVertex(vdofs[j]);
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
     
     dg_fes.GetElementVDofs(i, vdofs);
     ccn_gf.SetSubVector(vdofs, ccn);

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









