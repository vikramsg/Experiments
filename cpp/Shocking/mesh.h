#include <vector>

using namespace std;

class Mesh
{
  private:
    int N;
    double startX, stopX;
    double dx;

  public:
    vector<double> mesh;

    Mesh(const int &N_, double &startX_, double &stopX_): 
      N(N_), startX(startX_), stopX(stopX_) 
    {
      createMesh(N, startX, stopX);
    }; 

    void createMesh(const int &N, const double &startX, const double &stopX)
    {
      mesh.resize(N);

      dx     = (stopX - startX)/(N );
      for(int i = 0; i < N; i++)
      {
        mesh[i] = startX + i*dx + 0.5*dx;
      }
    };

    double getDx(){return dx;};
    double getElements(){return N;};
    
    ~Mesh() {}; 
};



