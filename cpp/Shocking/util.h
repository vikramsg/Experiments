#include<vector>

using namespace std;

void add(const vector<double> &a, const double &beta, const vector<double> &b, 
    vector<double> &c); 
void add(const double &alphs, const vector<double> &a, 
    const double &beta, const vector<double> &b, 
    vector<double> &c); 

void add(const vector<double> &a, const double &beta, const vector<double> &b, 
    vector<double> &c) 
{
  for(int i=0; i < a.size(); i++)
  {
    c[i] = a[i] + beta*b[i]; 
  }
}

void add(const double &alpha, const vector<double> &a, 
    const double &beta, const vector<double> &b, 
    vector<double> &c) 
{
  for(int i=0; i < a.size(); i++)
  {
    c[i] = alpha*a[i] + beta*b[i]; 
  }
}
