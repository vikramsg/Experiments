#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "console.h"
#include "mesh.h"
#include "util.h"
#include "euler.h"

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


    double getTimeStep(const double &lam, const double &CFL); 
    void timeStep(const double &dt_real, const vector<double> &u, vector<double> &u_new);
    void driver(const vector<double> &u, vector<double> &rhs);
};


int problem = 1;

int main(int argc, char **argv)
{
  int var_dim   = 3;

  int N;
  double startX, stopX;
  double T_fin;
  if (problem == 0)
  {
    N      = 500;
    T_fin  = 0.4;
    startX = -1.; stopX = 1;
  }
  else if (problem == 1)
  {
    N      = 20000;
    T_fin  = 1.8;
    startX = -5.; stopX = 5.;
  }

  Mesh msh(N, startX, stopX);
  Solver sol(msh);

  vector<double> u(var_dim*N), u_new(var_dim*N);
  sol.initSol(u);

  double CFL = 0.6 ;
  double lam = getMaxCharVel(N, u) ;
  double dt  = sol.getTimeStep(lam, CFL);
  
  double dt_real  = dt;

  double time = 0.;
  bool done   = false;
  for (int ti = 0; !done; )
  {
    lam = getMaxCharVel(N, u) ;
    dt  = sol.getTimeStep(lam, CFL);

    dt_real = min(dt, T_fin - time);
    
    sol.timeStep(dt_real, u, u_new);
    
    done = (time >= T_fin - 1e-8*dt);

    time = time + dt_real;
    u    = u_new;

    cons_out("Time:", time, "Char vel:", lam);
  }

  ofstream out_file;
  out_file.open("postp/shock.dat");
  for(int i = 0; i < N; i++)
  {
    out_file << msh.mesh[i] << " " << u_new[i] << " " << u_new[N + i] 
      << " " << u_new[2*N + i] << endl;
  }
  out_file.close();

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

double Solver::getTimeStep(const double &lam, const double &CFL) 
{
  double dx = msh.getDx();

  return CFL*dx/lam;

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
  double rho, p, v;
  for(int i = 0; i < N; i++)
  {
    if (problem == 0)
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
    }
    else if (problem == 1)
    {
       double rho_L  = 3.857143;
       double p_L    = 10.33333;
       double u_L    = 2.629369;

       double x = msh.mesh[i];

       double rho_R  = 1 + 0.2*sin(5.0*x) ;
       double p_R    = 1;
       double u_R    = 0;

       if (x >= -4.0)
       {
         rho = rho_R;
         p   = p_R;
         v   = u_R;
       }
       else
       {
         rho = rho_L;
         p   = p_L  ;
         v   = u_L  ;
       }

    }

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




