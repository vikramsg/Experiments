#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Initial condition
double u0_function(const Vector &x);

// Per coefficient
void per_function(const Vector &x, Vector &v);

// Mesh bounding box
Vector bb_min, bb_max;



int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-cube.mesh";
   int ser_ref_levels = 3;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 4;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool adios2 = false;
   bool binary = false;
   int vis_steps = 5;

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 7. Define the parallel discontinuous DG finite element space on the
   //    parallel refined mesh of the given polynomial order.
//   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   DG_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (mpi.Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   FunctionCoefficient u0(u0_function);

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   if (pa)
   {
      m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m->SetAssemblyLevel(AssemblyLevel::FULL);
      k->SetAssemblyLevel(AssemblyLevel::FULL);
   }

   Vector xdir(dim); 
   xdir = 0.0; xdir(0) = 1.0; 
   VectorConstantCoefficient x_dir(xdir);  

   m->AddDomainIntegrator(new MassIntegrator);
   constexpr double alpha = -1.0;
   k->AddDomainIntegrator(new ConvectionIntegrator(x_dir, alpha));
   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   m->Finalize();
   k->Finalize(skip_zeros);

   DenseMatrix elmat;
   k->ComputeElementMatrix(18, elmat);
//   elmat.Print();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   HypreParMatrix *M_mat = m->ParallelAssemble();
   HypreParMatrix *K_mat = k->ParallelAssemble();
      
   HypreSmoother *hypre_prec = new HypreSmoother(*M_mat, HypreSmoother::Jacobi);
      
   CGSolver M_solver;
   M_solver.SetPreconditioner(*hypre_prec);
   M_solver.SetOperator(*M_mat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);

   ParGridFunction z(fes);
   ParGridFunction y(fes);

   z = *u;

   ParGridFunction per(fes);
   VectorFunctionCoefficient per0(dim, per_function);

   StopWatch chrono; 
   chrono.Clear();
   chrono.Start();
   for(int i = 0; i < 100; i++)
   {
     K_mat->Mult(z, y);

     z += 10;

     cout << "Step:" << i << "\t" << y.Min() << "\t" << y.Max() << endl;

   }

   chrono.Stop();
   cout << "100 Steps took "<< chrono.RealTime() << " s "<< endl;

//   dgemv_(trans, m, n, DONE, A.data(), lda, &x[0], ONE, DZERO, &y[0], ONE);


   int pp1  = order + 1;
   int npts = pp1*pp1*pp1;
   Vector loc_u(npts), ac(npts);
   Array<int> dof_idx;

   z = *u;

   chrono.Clear();
   chrono.Start();
   for(int i = 0; i < 100; i++)
   {
     for(int ele = 0; ele < fes->GetNE(); ele++)
     {
       fes->GetElementVDofs(ele, dof_idx);
       z.GetSubVector(dof_idx, loc_u);
       elmat.Mult(loc_u, ac);
       y.SetSubVector(dof_idx, ac);
     }
     
     z += 10;
     
     cout << "Step:" << i << "\t" << y.Min() << "\t" << y.Max() << endl;

   }

   chrono.Stop();
   cout << "100 Steps took "<< chrono.RealTime() << " s "<< endl;


   // 13. Free the used memory.
   delete U;
   delete u;
   delete k;
   delete m;
   delete fes;
   delete pmesh;

   return 0;
}



// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}


// Perturbation coefficient
void per_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   double seed = 123;
   srand(seed);

   double rnd = rand()/RAND_MAX;

   v[0] = 0.01*sin(x[0]);
   v[1] =-0.01*cos(x[0]);

}

