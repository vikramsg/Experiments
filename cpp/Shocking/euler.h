#include <vector>
#include <cmath>

const double gamm = 1.4;

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


double getMaxCharVel(const int &N, const vector<double> &u) 
{
  double lam = 0;

  double rho, rhoU, rhoE, p, v_sq, a;
  for(int i = 0; i < N; i++)
  {
    rho  = u[0*N + i];
    rhoU = u[1*N + i];
    rhoE = u[2*N + i];

    v_sq = pow( rhoU/rho, 2. );
    p    = ( gamm - 1. )*( rhoE - 0.5*rho*v_sq );
    a    = sqrt(gamm*p/rho);

    lam  = max( lam, a + sqrt(v_sq));
  }

  return lam;

}



