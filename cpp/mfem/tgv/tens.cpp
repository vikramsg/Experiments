#include "mfem.hpp"
#include<iostream>

using namespace mfem;
using namespace std;

// Initial condition
double u0_function(const Vector &x);
template <int rows, int cols>
void printMatrix( double (&array)[rows][cols] );

const int order = 2;
const int np    = order + 1;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv); 

  const char *mesh_file = "periodic-square.mesh";
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh); 
  delete mesh;

  DG_FECollection fec(order, dim, BasisType::GaussLobatto);
  ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);
   
  FunctionCoefficient u0(u0_function);

  int irank;
  MPI_Comm_rank(MPI_COMM_WORLD, &irank);

  HYPRE_Int global_vSize = fes->GlobalTrueVSize();
  if (irank == 0)
  {
    std::cout << "Number of unknowns: " << global_vSize << std::endl;
  }

  Vector xdir(dim), ydir(dim);
  xdir[0] = 1.0; xdir[1] = 0.0;
  ydir[0] = 0.0; ydir[1] = 1.0;
  VectorConstantCoefficient x_dir(xdir);
  VectorConstantCoefficient y_dir(ydir);

  IntegrationRules GLIntRules(0, Quadrature1D::GaussLobatto); 
  const IntegrationRule *ir = &GLIntRules.Get(Geometry::CUBE  , 2*np-3);

  ParBilinearForm k(fes);
  ParBilinearForm m(fes);

  ConvectionIntegrator cix(x_dir, 1.0);
  cix.SetIntRule(ir);
  ConvectionIntegrator ciy(y_dir, 1.0);
  ciy.SetIntRule(ir);
  k.AddDomainIntegrator(&ciy);
  MassIntegrator mix;
  mix.SetIntRule(ir);
  m.AddDomainIntegrator(&mix);

  int skip_zeros = 0;
  k.Assemble(skip_zeros);
  k.Finalize(skip_zeros);
  m.Assemble();
  m.Finalize();

  CGSolver M_solver( m.ParFESpace()->GetComm() );
  OperatorHandle M;
  M.Reset(m.ParallelAssemble(), true);
  M_solver.SetOperator(*M);
  HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
  HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
  M_solver.SetPreconditioner(*hypre_prec);
  M_solver.iterative_mode = false;
  M_solver.SetRelTol(1e-9);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(100);
  M_solver.SetPrintLevel(0);

  ParGridFunction *u = new ParGridFunction(fes);
  u->ProjectCoefficient(u0);
  
  const Vector x(u->GetData(), u->Size());
  Vector y(u->Size());
  Vector y_fn(u->Size());

  // This is what we want to reploicate
  // The values printed here should be the same ones
  // that we get with tensor product multiplication
  {
    k.Mult(x, y);
    M_solver.Mult(y, y_fn);
//    y.Print();
  }

  // FIXME: ignoring Jacobians for now
  // will need to get to it eventually
  const int ne = fes->GetNE();

  Poly_1D poly;
  const double *pts = poly.GetPoints(order, 1); 
  Vector lag(np), lag_dx(np);

  Poly_1D::Basis cbasis1d = poly.GetBasis(order, 1);

  double der[np][np];

  for(int i = 0; i < np; i++)
  {
    cbasis1d.Eval(pts[i], lag, lag_dx);
    for(int j = 0; j < np; j++)
    {
      der[i][j] = lag_dx[j];
    }
  }
  printMatrix<np, np>(der);

  k.SpMat().Print();

// The action of der will be equivalent to 
// M^-1 Q, so we need to mult by k, then invert
// with m to compare with der

  double rhs_x[ne*np*np];
  double rhs_y[ne*np*np];
  for(int i = 0; i < ne; i++)
//  for(int i = 0; i < 1; i++)
  {

    for(int j = 0; j < np; j++)
    {
      for(int k1 = 0; k1 < np; k1++)
      {
        rhs_x[i*np*np + j*np  + k1] = 0.0;
        rhs_y[i*np*np + k1*np + j ] = 0.0;
        for(int k2 = 0; k2 < np; k2++)
        {
          int sub1          = i*np*np + j*np + k2;
          rhs_x[i*np*np + j*np + k1] += der[k1][k2] * x[sub1];

          int sub2          = i*np*np + k2*np + j;
          rhs_y[i*np*np + k1*np + j] += der[k1][k2] * x[sub2];

        }
      }
    }
 
  }
  

  for(int i = 0; i < x.Size(); i++)
  {
//    std::cout << setprecision(8)<< x[i] << " " << y_fn[i] << " " << 1.5*rhs_y[i] << std::endl; 
  }

//  delete u;
//  delete fes;
//  delete pmesh;

  MPI_Finalize();

  return 0;

}


// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   double value = exp(-10.*pow(x(1)-0.5,2));

   return value;
}

template <int rows, int cols>
void printMatrix( double (&array)[rows][cols] )
{
   for(int i = 0; i < rows; i++)
   {
    for(int j = 0; j < cols; j++)
    {
      std::cout << array[i][j] << " ";
    }
    std::cout << std::endl;
   }
}
