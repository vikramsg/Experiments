#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


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

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace      *fes = new FiniteElementSpace(mesh, fec);

   cout << "Number of unknowns: " << fes->GetVSize() << endl;

   int nele   = fes->GetNE();
   int nv     = fes->GetNV();
   int nbe    = fes->GetNBE();

//   DenseMatrix elmat, shape;
//   ElementTransformation *eltrans;
//
////   elmat.SetSize(nd);
////   elmat = 0.0;
//   Array<int> vdofs;
//   for (int i = 0; i < fes -> GetNE(); i++)
//   {
//      fes->GetElementVDofs(i, vdofs);
//      const FiniteElement &fe = *fes->GetFE(i);
//   
//      int nd = fe.GetDof();
//
//      elmat.SetSize(nd);
//      elmat = 0.0;
//
//      elmat.SetSize(nd);
//
//      eltrans = fes->GetElementTransformation(i);
//
//      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), fe.GetOrder());
//
//      int nir = ir->GetNPoints();
//
//      cout << nir << endl;
//
//
//   }
    
//     const FiniteElement *fe;
//     ElementTransformation *transf;
//     Vector shape;
//     Array<int> vdofs;
//     int fdof, d, i, intorder, j, k;
//  
//     for (i = 0; i < fes->GetNE(); i++)
//     {
//        fe = fes->GetFE(i);
//        fdof = fe->GetDof();
//        transf = fes->GetElementTransformation(i);
//        shape.SetSize(fdof);
//        intorder = 2*fe->GetOrder() + 1; // <----------
//        const IntegrationRule *ir;
//        
//        ir = &(IntRules.Get(fe->GetGeomType(), intorder));
//
//        fes->GetElementVDofs(i, vdofs);
//        for (j = 0; j < ir->GetNPoints(); j++)
//        {
//           const IntegrationPoint &ip = ir->IntPoint(j);
//           fe->CalcShape(ip, shape);
//           cout << *shape << endl;
//        }
//     }


   
//     GridFunction nodes(fes);
//     for (int i = 0; i < fes->GetNE(); i++)
//     {
//        int elNo = i; 
//        const FiniteElement *fe = fes->GetFE(elNo);
//        int dim = fe->GetDim(), dof = fe->GetDof();
//        DenseMatrix dshape(dof, dim), Jinv(dim);
//        Vector lval, gh(dim), grad, shape(dof);
//        Array<int> dofs;
//
//        grad.SetSize(dim);
//
//        fes->GetElementDofs(elNo, dofs);
//
//        ElementTransformation *tr = fes->GetElementTransformation(elNo);
//
////        fe->CalcShape(tr->GetIntPoint(), shape);
////        for (int j = 0; j < dof; j ++)
////            cout << shape[j] << endl;
//
//        fe->CalcDShape(tr->GetIntPoint(), dshape);
//        CalcInverse(tr->Jacobian(), Jinv);
//        nodes.GetSubVector(dofs, lval);
//        dshape.MultTranspose(lval, gh);
//        Jinv.MultTranspose(gh, grad);
//        dshape.Print();
//        Jinv.Print();
//     }



//     // For the reference element it will print the location of all points inside
//     // All x co-ordinates first, then y, then z 
//     for (int i = 0; i < 1 ; i++)
//     {
//        int elNo = i; 
//        const FiniteElement *fe = fes->GetFE(elNo);
//
//        const IntegrationRule &nodes = fe->GetNodes();
//        DenseMatrix pm;
//
//        pm.SetSize(fe->GetDim(), fe->GetDof());
//
//        for (int j = 0; j < fe -> GetDof(); j ++)
//        {
//            nodes.IntPoint(j).Get(&pm(0, j), fe -> GetDim());
//        }
//
//        pm.Print();
//
//      }
 
//     // For the reference element it will print the location of all points inside
//     // All x co-ordinates first, then y, then z 
//     for (int i = 0; i < 1 ; i++)
//     {
//        int elNo = i; 
//        const FiniteElement *fe = fes->GetFE(elNo);
//
//        const IntegrationRule &nodes = fe->GetNodes();
//        DenseMatrix pm;
//        Vector shape;
//
//        pm.SetSize(fe->GetDim(), fe->GetDof());
//
//        for (int j = 0; j < fe -> GetDof(); j ++)
//        {
//            nodes.IntPoint(j).Get(&pm(0, j), fe -> GetDim());
//            const IntegrationPoint &ip = nodes.IntPoint(j);
//        }
//
//
//        pm.Print();
//
//      }
 

   delete fes;
   delete fec;

   return 0;
}


