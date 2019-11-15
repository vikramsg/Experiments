#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};

// Initial condition
double u0_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "periodic-hexagon.mesh";
   int ref_levels = 0;
   int order = 1;
   double t_final = 0.03;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 1;

   int precision = 8;
   cout.precision(precision);


   ///////////////First simple test
   //
//   // 2. Read the mesh from the given mesh file. We can handle triangular,
//   //    quadrilateral, tetrahedral, hexahedral, surface and volume, as well as
//   //    periodic meshes with the same code.
//   Mesh *mesh = new Mesh(mesh_file, 1, 1);
//   int dim    = mesh->Dimension();
//   int sdim   = mesh->SpaceDimension();
//
//   int nfaces = mesh->GetNFaces();
//   int nele   = mesh->GetNE();
//
//   int nv     = mesh->GetNV();
//
//   int nbe    = mesh->GetNBE();
//
//   cout << dim    << '\t' << sdim << endl;
//   cout << nfaces << '\t' << nele << '\t' << nbe << endl;

//   // Get global vertex counter
//   Array<int> vert;
//   for (int i = 0; i < nele; i ++ )
//   {
//       mesh->GetElementVertices(i, vert);
//       for (int j = 0; j < vert.Size(); j++)
//       {
//           cout << i << '\t' << j << '\t' << vert[j] << endl;
//       }
//   }

//   // Get vertex coordinates of all mesh elements
//   for (int i = 0; i < nele; i ++ )
//   {
//      double coord[dim];
//
//      for (int j = 0; j < nv ; j++)
//      {
//          mesh->GetNode(j, coord);
//          cout << coord[0] << '\t' << coord[1] << '\t' << coord[2] << endl;
//      }
//   }
//
//   // Get vertex coordinates of all mesh elements
//   Array<int> vert;
//   for (int i = 0; i < nele; i ++ )
//   {
//      double coord[dim];
//
//      cout << "Element " << i << endl;
//      mesh->GetElementVertices(i, vert);
//  
//      for (int j = 0; j < vert.Size() ; j++)
//      {
//          mesh->GetNode(vert[j], coord);
//          cout << coord[0] << '\t' << coord[1] << endl;
//      }
//   }
//


//   //  Define the discontinuous DG finite element space of the given
//   //    polynomial order on the refined mesh.
//   FiniteElementCollection *fec = new DG_FECollection(order, dim);
//   FiniteElementSpace      *fes = new FiniteElementSpace(mesh, fec);
//
//   Array<int> dofs;
//   for (int i = 0; i < nele; i ++ )
//   {
//      fes->GetElementDofs(i, dofs);
//      cout << "Element " << i << " DOF "<< dofs.Size() << endl;
//   }
//
//
//   GridFunction nodes(fes);
//
//   // Print all nodes in the finite element space 
//   mesh->GetNodes(nodes);
//
//   const int nNodes = nodes.Size() / dim;
//   double coord[dim]; // coordinates of a node
//   for (int i = 0; i < nNodes; ++i) {
//     for (int j = 0; j < dim; ++j) {
//       coord[j] = nodes(j * nNodes + i); 
//       cout << coord[j] << " ";
//     }   
//     cout << endl;
//   }
//   ////////////////////////////


   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

//   mesh->UniformRefinement();

//   // 5. Define the discontinuous DG finite element space of the given
//   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace      fes(mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   int nele   = fes.GetNE();
   int nv     = fes.GetNV();
   int nbe    = fes.GetNBE();

   FunctionCoefficient u0(u0_function);

   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   GridFunction u_grad(&fes);
   u.GetDerivative(1, 0, u_grad);

   // We need a vector finite element space for nodes
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   // Print all nodes in the finite element space 
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int sub1 = i, sub2 = nodes.Size()/dim + i;
           cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u(sub1) << '\t'<< u_grad(sub1) << endl;   
   }



//   cout << "Dimensions " << dim    << endl;
//   cout << "Elements " << nele  << " Vertices " << nv   << endl;

   // Get vertex coordinates of all mesh elements
//   Array<int> dofs;
//   for (int i = 0; i < nele; i ++ )
//   {
//      cout << "Element " << i << endl;
//      fes->GetElementDofs(i, dofs);
//
//      const FiniteElement *fe = fes->GetFE(i);
//
//      ElementTransformation *eltrans = fes->GetElementTransformation(i);
//
//      IntegrationRule *ir = new IntegrationRule(fe->GetDof());
//
//      DenseMatrix dshape(fe->GetDof(), fe->GetDim());
//      DenseMatrix Jinv(fe->GetDim());
//      Vector lval, gh(fe->GetDim()), gcol;

//      dshape.Print();

//   Array<int> dofs;
//   fes->GetElementDofs(elem, dofs);
 
//      cout << fes->GetOrder(i) << endl;
//      cout << fes->GetElementType(i) << endl;
//    
//      for (int j = 0; j < dofs.Size(); j++)
//      {
//          int dof = dofs[j];
//
//          cout << dof << endl;
//      }
//   }
 
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

   double fn = cos(M_PI*X(0))*sin(M_PI*X(1));

//   cout << x(0) << '\t' << x(1) << '\t' << fn << endl;

   return fn;
}

