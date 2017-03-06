
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "cube.msh";

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume, as well as
   //    periodic meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim    = mesh->Dimension();
   int sdim   = mesh->SpaceDimension();

   int nfaces = mesh->GetNFaces();
   int nele   = mesh->GetNE();

   int nv     = mesh->GetNV();

   int nbe    = mesh->GetNBE();

   cout << dim    << '\t' << sdim << endl;
   cout << nfaces << '\t' << nele << '\t' << nbe << endl;

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

   int order = 1;

   //  Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace      *fes = new FiniteElementSpace(mesh, fec);

   Array<int> dofs;
   for (int i = 0; i < nele; i ++ )
   {
      fes->GetElementDofs(i, dofs);
      cout << "Element " << i << " DOF "<< dofs.Size() << endl;
   }


   GridFunction nodes(fes);

   // Print all nodes in the finite element space 
   mesh->GetNodes(nodes);

   const int nNodes = nodes.Size() / dim;
   double coord[dim]; // coordinates of a node
   for (int i = 0; i < nNodes; ++i) {
     for (int j = 0; j < dim; ++j) {
       coord[j] = nodes(j * nNodes + i); 
       cout << coord[j] << " ";
     }   
     cout << endl;
   }


   return 0;
}




