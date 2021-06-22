#include <bits/stdc++.h>

using namespace std;

// Get max number of consecutive 1's in binary representation
int main()
{
    int n;
    cin >> n;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    
    int quotient, remainder;
    
    vector<string> bin;
    
    string number;
    
    int i = 0;
    while(n > 0)
    {
        quotient  = n/2;
        remainder = n%2;
        
        number.append(to_string(remainder));
        
        n = quotient;
        i++;
    }
    // Reverse to get binary number as string
    std::reverse(number.begin(), number.end());
    
    std::size_t found = number.find("1");
    std::size_t found0;
    
    size_t ln = 0;
    while (found!=std::string::npos)
    {
        found0 = number.find("0", found);
        
        size_t sz = found0 == std::string::npos ? number.length() - found: found0 - found;
        
        ln = std::max(ln, sz);
        
        found = number.find("1", found0);
        
    }
    
    cout << ln << endl;

    return 0;
}

