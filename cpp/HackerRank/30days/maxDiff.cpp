#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

class Difference {
    private:
    vector<int> elements;
  
  	public:
  	int maximumDifference;

	// Add your code here
    
    Difference(vector<int> &elements_): elements(elements_){};
    
    void computeDifference()
    {
        int maxDiff = std::numeric_limits<int>::min();
        for(int i = 0; i < elements.size() - 1; i++)
        {
            for(int j = i + 1; j < elements.size(); j++)
            {
                maxDiff = max(maxDiff, abs(elements[i] - elements[j]));
                
            }
        }
        maximumDifference = maxDiff;
    }
}; // End of Difference class

int main() {
    int N;
    cin >> N;
    
    vector<int> a;
    
    for (int i = 0; i < N; i++) {
        int e;
        cin >> e;
        
        a.push_back(e);
    }
    
    Difference d(a);
    
    d.computeDifference();
    
    cout << d.maximumDifference;
    
    return 0;
}
