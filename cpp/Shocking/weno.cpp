#include <iostream>
#include <vector>
#include <cmath>
#include "console.h"

using namespace std;

void createMesh(const int &N, const double &startX, const double &stopX, 
    vector<double> &mesh);
void initSol(const int &N, const vector<double> &mesh, vector<double> &u);
void getEulerFlux(const int &N, const int &dim, const vector<double> &u, 
    vector<double> &f);

const double gamm = 1.4;

int main(int argc, char **argv)
{
  double startX = -1., stopX = 1;

  int N = 10;

  vector<double> mesh;
  createMesh(N, startX, stopX, mesh);

  vector<double> u;
  initSol(N, mesh, u);

  int dim = 1;
  vector<double> f;
  getEulerFlux(N, dim, u, f);

  for(int i = 0; i < N; i++)
  {
    cons_out(i, mesh[i], u[i], f[2*N + i]);
  }

  return 0;
}


void createMesh(const int &N, const double &startX, const double &stopX, 
    std::vector<double> &mesh)
{
  mesh.resize(N);

  auto dx     = (stopX - startX)/(N );
  for(int i = 0; i < N; i++)
  {
    mesh[i] = startX + i*dx + 0.5*dx;
  }
}


void initSol(const int &N, const vector<double> &mesh, vector<double> &u) 
{
  int var_dim = 3; // Hardcode for 1D
  u.resize(var_dim*N);

  double rho, p, v;
  for(int i = 0; i < N; i++)
  {
    if (mesh[i] < 0.)
    {
      rho   = 1.;
      p     = 1.;
    }
    else
    {
      rho   = 0.125;
      p     = 0.1;
    }
    v = 0.;

    u[0*N + i] = rho;
    u[1*N + i] = rho*v;
    u[2*N + i] = p/(gamm - 1.) + 0.5*rho*v*v;

  }

}

void getEulerFlux(const int &N, const int &dim, const vector<double> &u, 
    vector<double> &f) 
{

  int var_dim = dim + 2; // Hardcode for 1D
  f.resize(var_dim*N);

  double rho, rhoU, rhoE, p, v_sq;
  for(int i = 0; i < N; i++)
  {
    rho  = u[0*N + i];
    rhoU = u[1*N + i];
    rhoE = u[2*N + i];

    v_sq = pow( rhoU/rho, 2. );
    p    = ( gamm - 1. )*( rhoE - 0.5*rho*v_sq );

    f[0*N + i] = rhoU;
    f[1*N + i] = rho*v_sq + p;
    f[2*N + i] = ( rhoE + p )*rhoU/rho;
  }

}
