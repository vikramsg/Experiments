#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;


int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */  

  uint32_t dact, mact, yact;
  uint32_t dexp, mexp, yexp;

  cin >> dact >> mact >> yact ;
  cin >> dexp >> mexp >> yexp ;

  uint32_t fine = 0;
  if ( (mact == mexp) && (yact == yexp) && (dact > dexp))
  {
    fine = 15*( dact - dexp );
  }
  else if ( (yact == yexp) && (mact > mexp) )
  {
    fine = 500*( mact - mexp );
  }
  else if ( (yact > yexp) )
  {
    fine = 10000;
  }

  cout << fine << endl;
     
  return 0;

}
