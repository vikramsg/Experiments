#include <iostream>
#include <vector>
#include <cmath>
#include "console.h"
#include "mesh.h"
#include "util.h"

using namespace std;

class Solver 
{
  private:
    Mesh &msh;
    int N;
    vector<double> f_e;
    vector<double> rhs;
    
    vector<double> y_t;

  public:
    Solver(Mesh &msh_): msh(msh_) 
    {
      int var_dim = 3;
      N = msh.getElements();
      f_e.resize(var_dim*( N + 1 ) );
      rhs.resize(var_dim*( N  ) );

      y_t.resize(var_dim*( N  ) );
    }; 

    void initSol(vector<double> &u); 
    void getEdgeFlux(const vector<double> &u); 
    void getBndFlux(const vector<double> &u); 
    void getRHS(vector<double> &rhs); 

    void timeStep(const double &dt_real, const vector<double> &u, vector<double> &u_new);
    void driver(const vector<double> &u, vector<double> &rhs);
};

void getEulerFlux(const int &N, const int &dim, const vector<double> &u, 
    vector<double> &f);
void getRusanovFlux(int &var_dim, const vector<double> &u_L, const vector<double> &u_R, 
    vector<double> &f_com); 

const double gamm = 1.4;

int main(int argc, char **argv)
{
  double startX = -1., stopX = 1;

  int var_dim = 3;
  int N       = 10;

  Mesh msh(N, startX, stopX);
  Solver sol(msh);

  vector<double> u(var_dim*N), u_new(var_dim*N);
  sol.initSol(u);

  double dt = 0.1;
  sol.timeStep(dt, u, u_new);

  for(int i = 0; i < N; i++)
  {
    cons_out(i, u[i], u_new[i]);
  }

  return 0;
}


void Solver::timeStep(const double &dt_real, const vector<double> &u, 
    vector<double> &u_new)
{
  driver(u, this->rhs);

  // x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
  add(u, dt_real, this->rhs, y_t);
  driver(y_t, this->rhs);
  
  // x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
  add(y_t, dt_real, this->rhs, y_t);
  add(3./4, u, 1./4, y_t, y_t);
  driver(y_t, this->rhs);
  
  // x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt
  add(y_t, dt_real, this->rhs, y_t);
  add(1./3, u, 2./3, y_t, u_new);
}

void Solver::driver(const vector<double> &u, vector<double> &rhs_)
{
  getEdgeFlux(u);
  getBndFlux(u);
  getRHS(rhs_);
}


void Solver::getRHS(vector<double> &rhs_)
{
  double dx = msh.getDx();
  for(int i = 0; i < N; i++)
  {
    rhs_[0*N + i] = -(f_e[0*(N + 1) + i+1] - f_e[0*(N + 1) + i])/dx; 
    rhs_[1*N + i] = -(f_e[1*(N + 1) + i+1] - f_e[1*(N + 1) + i])/dx; 
    rhs_[2*N + i] = -(f_e[2*(N + 1) + i+1] - f_e[2*(N + 1) + i])/dx; 
  }
}


void Solver::initSol(vector<double> &u) 
{
  int var_dim = 3; // Hardcode for 1D

  double rho, p, v;
  for(int i = 0; i < N; i++)
  {
    if (msh.mesh[i] < 0.)
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


void Solver::getEdgeFlux(const vector<double> &u) 
{
  int var_dim = 3; // Hardcode for 1D

  vector<double> u_L(var_dim), u_R(var_dim);
  vector<double> f_com(var_dim);
  for(int i = 1; i < N; i++)
  {
    u_L[0] = u[0*N + i - 1];
    u_L[1] = u[1*N + i - 1];
    u_L[2] = u[2*N + i - 1];

    u_R[0] = u[0*N + i];
    u_R[1] = u[1*N + i];
    u_R[2] = u[2*N + i];

    getRusanovFlux(var_dim, u_L, u_R, f_com);

    f_e[0*( N + 1 ) + i] = f_com[0];
    f_e[1*( N + 1 ) + i] = f_com[1];
    f_e[2*( N + 1 ) + i] = f_com[2];
    
  }
}


void Solver::getBndFlux(const vector<double> &u) 
{
  int var_dim = 3; // Hardcode for 1D

  vector<double> u_L(var_dim), u_R(var_dim), f_com(var_dim);
  {
    int i = 0;
    u_L[0] = u[0*N + i];
    u_L[1] = u[1*N + i];
    u_L[2] = u[2*N + i];
  
    u_R[0] = u[0*N + i];
    u_R[1] = u[1*N + i];
    u_R[2] = u[2*N + i];
  
    getRusanovFlux(var_dim, u_L, u_R, f_com);
  
    f_e[0*( N + 1 ) + i] = f_com[0];
    f_e[1*( N + 1 ) + i] = f_com[1];
    f_e[2*( N + 1 ) + i] = f_com[2];
  }
  {
    int i = N;
    u_L[0] = u[0*N + i - 1];
    u_L[1] = u[1*N + i - 1];
    u_L[2] = u[2*N + i - 1];
  
    u_R[0] = u[0*N + i - 1];
    u_R[1] = u[1*N + i - 1];
    u_R[2] = u[2*N + i - 1];
  
    getRusanovFlux(var_dim, u_L, u_R, f_com);
  
    f_e[0*( N + 1 ) + i] = f_com[0];
    f_e[1*( N + 1 ) + i] = f_com[1];
    f_e[2*( N + 1 ) + i] = f_com[2];
  }
}



void getEulerFlux(const int &N, const int &dim, const vector<double> &u, 
    vector<double> &f) 
{

  int var_dim = dim + 2; // Hardcode for 1D

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



void getRusanovFlux(int &var_dim, const vector<double> &u_L, const vector<double> &u_R, 
    vector<double> &f_com) 
{
  double v_sq_L, v_sq_R, p_L, p_R, a_L, a_R, lam;
  double f_L[var_dim], f_R[var_dim];

  v_sq_L = pow( u_L[1]/u_L[0], 2. );
  p_L    = ( gamm - 1. )*( u_L[2] - 0.5*u_L[0]*v_sq_L );
  a_L    = sqrt(gamm*p_L/u_L[0]);

  v_sq_R = pow( u_R[1]/u_R[0], 2. );
  p_R    = ( gamm - 1. )*( u_R[2] - 0.5*u_R[0]*v_sq_R );
  a_R    = sqrt(gamm*p_R/u_R[0]);

  lam = max( a_L + sqrt(v_sq_L), a_R + sqrt(v_sq_R));

  f_L[0] = u_L[1];
  f_L[1] = u_L[0]*v_sq_L + p_L;
  f_L[2] = ( u_L[2] + p_L )*u_L[1]/u_L[0];

  f_R[0] = u_R[1];
  f_R[1] = u_R[0]*v_sq_R + p_R;
  f_R[2] = ( u_R[2] + p_R )*u_R[1]/u_R[0];

  f_com[0] = 0.5*(f_L[0] + f_R[0]) - 0.5*lam*(u_R[0] - u_L[0]); 
  f_com[1] = 0.5*(f_L[1] + f_R[1]) - 0.5*lam*(u_R[1] - u_L[1]); 
  f_com[2] = 0.5*(f_L[2] + f_R[2]) - 0.5*lam*(u_R[2] - u_L[2]); 
}



