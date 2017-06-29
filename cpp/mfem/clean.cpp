#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;



int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "periodic-square.mesh";
   int order = 1;
   double t_final = 0.02;
   double dt = 0.01;
   bool visit = true;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   FunctionCoefficient u0(u0_function);

   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   //Get derivatives
   GridFunction u_grad(&fes);
   u.GetDerivative(1, 0, u_grad);

   u.Print();

   // Print all nodes in the finite element space 
   GridFunction nodes(&fes);
   mesh->GetNodes(nodes);

//   nodes.Print();
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); 
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

   double fn = cos(M_PI*X(0))*sin(M_PI*X(1));

//   cout << x(0) << '\t' << x(1) << '\t' << fn << endl;

   return fn;
}

