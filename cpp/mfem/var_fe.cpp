#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Problem to solve
int problem;

// Velocity coefficient
double init_function(const Vector &x);



int main(int argc, char *argv[])
{
   const char *mesh_file = "periodic-square.mesh";
   int    order      = 1;
   double t_final    = 0.0100;
   double dt         = 0.0100;
   int    vis_steps  = 100;
   int    ref_levels = 0;

          problem    = 0;

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

   int ne = mesh->GetNE();
   Array<int> eleOrder(ne);

   for (int i = 0; i < ne; i++) eleOrder[i] = order; 
   for (int i = 2; i < 4 ; i++) eleOrder[i] = 3; 

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   VarL2_FiniteElementCollection *vfec = new VarL2_FiniteElementCollection(mesh, eleOrder);
   FiniteElementSpace fes(mesh, vfec);

//   {
//       for (int i = 2; i < 4 ; i++) eleOrder[i] = 4; 
//       vfec->Update(eleOrder); 
//       fes.Update();
//   }

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   FunctionCoefficient u0(init_function);
   GridFunction u_sol(&fes);
   u_sol.ProjectCoefficient(u0);

   Vector dir(dim);
   dir(0) = 1.0; dir(1) = 0.0;
   VectorConstantCoefficient x_dir(dir);

   BilinearForm k_inv_x(&fes);
   k_inv_x.AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));

   int skip_zeros = 1;
   k_inv_x.Assemble(skip_zeros);
   k_inv_x.Finalize(skip_zeros);

   SparseMatrix &K_inv_x = k_inv_x.SpMat();

   GridFunction u_out(&fes);
   K_inv_x.Mult(u_sol, u_out);

   // Print all nodes in the finite element space 
   FiniteElementSpace fes_nodes(mesh, vfec, dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int offset = nodes.Size()/dim;
       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << endl;      
       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << u_out(sub1) << endl;      
   }

   delete mesh;
   delete vfec;

   return 0;
}



//  Initialize variables coefficient
double init_function(const Vector &x)
{
   //Space dimensions 
   int dim = x.Size();

   return 1 + 0.2*sin(M_PI*(x(0) + x(1)));
}

