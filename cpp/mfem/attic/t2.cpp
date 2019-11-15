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


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/periodic-hexagon.mesh";
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

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace      *fes = new FiniteElementSpace(mesh, fec);

   cout << "Number of unknowns: " << fes->GetVSize() << endl;

   int nele   = fes->GetNE();
   int nv     = fes->GetNV();
   int nbe    = fes->GetNBE();

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
 
   delete fes;
   delete fec;

   return 0;
}


