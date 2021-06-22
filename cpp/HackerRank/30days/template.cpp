#include <iostream>
#include <vector>
#include <string>

using namespace std;

/**
*    Name: printArray
*    Print each element of the generic vector on a new line. Do not return anything.
*    @param A generic vector
**/

// Write your code here
template<class T>
void printArray(vector<T> array)
{
  for(T it: array)
    cout << it << endl;
}

template<class T>
void printArray1(vector<T> array)
{
  typename vector<T>::iterator it;
  for(it=array.begin();it!=array.end();it++)
  {
    cout << *it << endl;
  }
}


int main() {
	int n;
	
	cin >> n;
	vector<int> int_vector(n);
	for (int i = 0; i < n; i++) {
		int value;
		cin >> value;
		int_vector[i] = value;
	}
	
	cin >> n;
	vector<string> string_vector(n);
	for (int i = 0; i < n; i++) {
		string value;
		cin >> value;
		string_vector[i] = value;
	}

	printArray1<int>(int_vector);
	printArray1<string>(string_vector);

	return 0;
}
