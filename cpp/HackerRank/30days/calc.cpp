#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>
using namespace std;

//Write your code here
class Calculator
{
  public:
     int power(int n, int p)
     {
       if (  (n < 0) || (p < 0) )
         throw std::invalid_argument( "n and p should be non-negative" );
       else
         return std::pow(n, p);
     }
    
};

int main()
{
    Calculator myCalculator=Calculator();
    int T,n,p;
    cin>>T;
    // Note the following implies, T-- > 0
    while(T-->0){
      if(scanf("%d %d",&n,&p)==2){
         try{
               int ans=myCalculator.power(n,p);
               cout<<ans<<endl; 
         }
         catch(exception& e){
             cout<<e.what()<<endl;
         }
      }
    }
    
}
