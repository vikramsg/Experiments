#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double R_gas = 287;
const double   Pr  = 0.72;

// Velocity coefficient
void init_function(const Vector &x, Vector &v);

// Characteristic boundary condition specification
void char_bnd_cnd(const Vector &x, Vector &v);
// Wall boundary condition specification
void wall_bnd_cnd(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);

void getFields(const GridFunction &u_sol, GridFunction &rho, GridFunction &u1, GridFunction &u2, GridFunction &E);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form is M du/dt = K u + b, where M and K are the mass
    and operator matrices, and b describes the face correction terms. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K_inv_x, &K_inv_y;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K_inv_x, SparseMatrix &_K_inv_y, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};



int main(int argc, char *argv[])
{
//   const char *mesh_file = "char_wall.msh";
   const char *mesh_file = "n0012.msh";
   int    order      = 2;
   double t_final    = 0.0005;
   double dt         = 0.0005;
   int    vis_steps  = 50;
   int    ref_levels = 0;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int     dim = mesh->Dimension();
   int var_dim = dim + 2;

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec, var_dim);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient u0(var_dim, init_function);
   GridFunction u_sol(&fes);
   u_sol.ProjectCoefficient(u0);

   FiniteElementSpace fes_vec(mesh, &fec, dim*var_dim);
   GridFunction f_inv(&fes_vec);
   getInvFlux(dim, u_sol, f_inv);

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector dir(dim);
   dir(0) = 1.0; dir(1) = 0.0;
   VectorConstantCoefficient x_dir(dir);
   dir(0) = 0.0; dir(1) = 1.0;
   VectorConstantCoefficient y_dir(dir);

   FiniteElementSpace fes_op(mesh, &fec);
   BilinearForm m(&fes_op);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k_inv_x(&fes_op);
   k_inv_x.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));
   BilinearForm k_inv_y(&fes_op);
   k_inv_y.AddDomainIntegrator(new ConvectionIntegrator(y_dir, -1.0));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 1;
   k_inv_x.Assemble(skip_zeros);
   k_inv_x.Finalize(skip_zeros);
   k_inv_y.Assemble(skip_zeros);
   k_inv_y.Finalize(skip_zeros);
   SparseMatrix &M   = m.SpMat();
   /////////////////////////////////////////////////////////////
   
   VectorGridFunctionCoefficient u_vec(&u_sol);
   VectorGridFunctionCoefficient f_vec(&f_inv);

   /////////////////////////////////////////////////////////////
   // Linear form
   LinearForm b(&fes);
   b.AddFaceIntegrator(
      new DGEulerIntegrator(u_vec, f_vec, var_dim, -1.0));
   ///////////////////////////////////////////////////////////
   VectorFunctionCoefficient u_char_bnd(var_dim, char_bnd_cnd); // Defines characterstic boundary condition
   // Linear form for boundary condition
   Array<int> dir_bdr(mesh->bdr_attributes.Max());
   dir_bdr     = 0; // Deactivate all boundaries

   dir_bdr[0]  = 1; // Activate boundary 0

   LinearForm b1(&fes);
   b1.AddBdrFaceIntegrator(
      new DG_CNS_Characteristic_Integrator(
      u_vec, u_char_bnd, var_dim, -1.0), dir_bdr); 
   b1.Assemble();
   ///////////////////////////////////////////////////////////
   VectorFunctionCoefficient u_wall_bnd(dim, wall_bnd_cnd); // Defines characterstic boundary condition
   // Linear form for boundary condition
   dir_bdr     = 0; // Deactivate all boundaries

   dir_bdr[1]  = 1; // Activate boundary 1

   LinearForm b2(&fes);
   b2.AddBdrFaceIntegrator(
      new DG_CNS_NoSlipWall_Integrator(
      u_vec, u_wall_bnd, var_dim, -1.0), dir_bdr); 
   b2.Assemble();
   ///////////////////////////////////////////////////////////


   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   
   dc = new VisItDataCollection("Euler", mesh);
   dc->SetPrecision(precision);
   
   FiniteElementSpace fes_fields(mesh, &fec);
   GridFunction rho(&fes_fields);
   GridFunction u1(&fes_fields);
   GridFunction u2(&fes_fields);
   GridFunction E(&fes_fields);

   dc->RegisterField("rho", &rho);
   dc->RegisterField("u1", &u1);
   dc->RegisterField("u2", &u2);
   dc->RegisterField("E", &E);

   getFields(u_sol, rho, u1, u2, E);

   dc->SetCycle(0);
   dc->SetTime(0.0);
   dc->Save();


   FE_Evolution adv(m.SpMat(), k_inv_x.SpMat(), k_inv_y.SpMat(), b);
//   ODESolver *ode_solver = new ForwardEulerSolver; 
   ODESolver *ode_solver = new RK3SSPSolver; 

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      b.Assemble();
      b1.Assemble();
      b2.Assemble();

      add(b, b1, b);
      add(b, b2, b);

      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u_sol, t, dt_real);
      ti++;

      cout << "time step: " << ti << ", time: " << t << ", max_sol: " << u_sol.Max() << endl;

      getInvFlux(dim, u_sol, f_inv); // To update f_vec

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
          getFields(u_sol, rho, u1, u2, E);

          dc->SetCycle(ti);
          dc->SetTime(t);
          dc->Save();
      }
   }
  
   // Print all nodes in the finite element space 
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int offset = nodes.Size()/dim;
       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub3) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << b[sub4] << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << b[sub1] << "\t" << b1[sub1] << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << inv_flux(sub1) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << u_sol(sub2) << '\t' << u_sol(sub3) << '\t' << u_sol(sub4) << endl;      
   }


   return 0;
}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K_inv_x, SparseMatrix &_K_inv_y, const Vector &_b)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), b(_b), z(_b.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
    int dim = x.Size()/K_inv_x.Size() - 2;
    int var_dim = dim + 2;

    y.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector y_temp;
    y_temp.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector f(dim*x.Size());
    getInvFlux(dim, x, f);

    int offset  = K_inv_x.Size();
    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }

    Vector f_sol(offset), f_x(offset), f_x_m(offset);
    Vector b_sub(offset);
    y = 0.0;
    for(int i = 0; i < var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_x.Mult(f_sol, f_x);
        b.GetSubVector(offsets[i], b_sub);
        f_x += b_sub; // Needs to be added only once
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i], f_x_m);
    }
    y += y_temp;
    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_y.Mult(f_sol, f_x);
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i - var_dim], f_x_m);
    }
    y += y_temp;

//    for (int j = 0; j < offset; j++) cout << x(j) << '\t'<< f(offset + j) << endl;
//    for (int j = 0; j < offset; j++) cout << j << '\t'<< b(j) << endl;
//    for (int j = 0; j < offset; j++) cout << x(offset + j) << '\t'<< f(var_dim*offset + 3*offset + j) << endl;
//    for (int j = 0; j < offset; j++) cout << x(offset + j) << '\t'<< y(0*offset + j) << endl;

}



// Inviscid flux 
void getInvFlux(int dim, const Vector &u, Vector &f)
{
    int var_dim = dim + 2;
    int offset  = u.Size()/var_dim;

    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }
    Vector rho, rho_u1, rho_u2, E;
    u.GetSubVector(offsets[0], rho   );
    u.GetSubVector(offsets[1], rho_u1);
    u.GetSubVector(offsets[2], rho_u2);
    u.GetSubVector(offsets[3],      E);

    Vector f1(offset), f2(offset), f3(offset);
    Vector g1(offset), g2(offset), g3(offset);
    for(int i = 0; i < offset; i++)
    {
        double u1   = rho_u1(i)/rho(i);
        double u2   = rho_u2(i)/rho(i);
        
        double v_sq = pow(u1, 2) + pow(u2, 2);
        double p    = (E(i) - 0.5*rho(i)*v_sq)*(gamm - 1);

        f1(i) = rho_u1(i)*u1 + p; //rho*u*u + p    
        f2(i) = rho_u1(i)*u2    ; //rho*u*v
        f3(i) = (E(i) + p)*u1   ; //(E+p)*u
        
        g1(i) = rho_u2(i)*u1    ; //rho*u*v 
        g2(i) = rho_u2(i)*u2 + p; //rho*v*v + p
        g3(i) = (E(i) + p)*u2   ; //(E+p)*v
    }

    f.SetSubVector(offsets[0], rho_u1);
    f.SetSubVector(offsets[1],     f1);
    f.SetSubVector(offsets[2],     f2);
    f.SetSubVector(offsets[3],     f3);

    f.SetSubVector(offsets[4], rho_u2);
    f.SetSubVector(offsets[5],     g1);
    f.SetSubVector(offsets[6],     g2);
    f.SetSubVector(offsets[7],     g3);

}

// Characteristic boundary condition
void char_bnd_cnd(const Vector &x, Vector &v)
{
    double rho, u1, u2, p;

    rho = 1.0; 
    u1  = 0.2; u2 = 0.0;
    p   = 1;
 
    double v_sq = pow(u1, 2) + pow(u2, 2);

    v(0) = rho;                     //rho
    v(1) = rho * u1;                //rho * u
    v(2) = rho * u2;                //rho * v
    v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
}


void wall_bnd_cnd(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
   }
}




//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double rho, u1, u2, p;
       rho = 1.0; 
       u1  = 0.2; u2 = 0.0;
       p   = 1;

       double v_sq = pow(u1, 2) + pow(u2, 2);

       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
}


void getFields(const GridFunction &u_sol, GridFunction &rho, GridFunction &u1, GridFunction &u2, GridFunction &E)
{

    int vDim  = u_sol.VectorDim();
    int dofs  = u_sol.Size()/vDim;

    for (int i = 0; i < dofs; i++)
    {
        rho[i] = u_sol[         i];        
        u1 [i] = u_sol[  dofs + i]/rho[i];        
        u2 [i] = u_sol[2*dofs + i]/rho[i];        
        E  [i] = u_sol[3*dofs + i];        
    }
}
