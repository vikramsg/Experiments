#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;


bool isPrime(int n)
{
  bool pr = true;

  if ( (n == 1) || (n == 2) )
    return pr;

  int lim = sqrt(n) + 1;
  for(int i=2; i < lim; i++)
  {
    if (n%i == 0)
    {
      pr = false;

      break;
    }
  }

  return pr;  
}

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */   
    int n;
    cin >> n;

    vector<string> str;

    int num;
    for(int i=0; i < n; i++)
    {
      cin >> num;
                  
      bool p = isPrime(num);

      if (p)
        str.push_back("Prime");
      else
        str.push_back("Not prime");
      
    }
    for(string s: str)
      cout << s << endl;
    
    return 0;
}
