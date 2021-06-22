#include <bits/stdc++.h>

using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for(int a_i = 0; a_i < n; a_i++){
    	cin >> a[a_i];
    }
    // Write Your Code Here
    
    int totSwaps = 0;
    for (int i = 0; i < n; i++) {
    // Track number of elements swapped during a single array traversal
    int numberOfSwaps = 0;
    
    for (int j = 0; j < n - 1; j++) {
        // Swap adjacent elements if they are in decreasing order
        if (a[j] > a[j + 1]) {
            
            int temp = a[j];
            a[j]     = a[j + 1];
            a[j + 1] = temp;
            
            numberOfSwaps++;
        }
    }
    
    totSwaps += numberOfSwaps;
    
      // If no elements were swapped during a traversal, array is sorted
      if (numberOfSwaps == 0) {
          break;
      }
    }
    
    cout << "Array is sorted in "<< totSwaps << " swaps."<< endl;
    cout << "First Element: " << a[0]   << endl;
    cout << "Last Element: "  << a[n-1] <<endl;
    return 0;
}