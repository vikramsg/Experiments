#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>
using namespace std;

vector<string> split_string(string);

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */  
    
    int n;
    cin >> n >> ws;
    
    string s_in;
    vector<string> splits;
    std::map<string,int> phonebook;
    for(int i=0; i < n; i++)
    {
        getline(cin, s_in);
        
        splits = split_string(s_in);
        
        phonebook[splits[0]] = stoi(splits[1]);
    }
    while(getline(cin, s_in))
    {
      if ( phonebook.find(s_in) == phonebook.end() ) 
       {
          cout << "Not found" << endl;
       } 
       else 
       {
          cout << s_in << "=" << phonebook[s_in] << endl;
       }
    }

     
    return 0;
}

vector<string> split_string(string input_string) {
    string::iterator new_end = unique(input_string.begin(), input_string.end(), [] (const char &x, const char &y) {
        return x == y and x == ' ';
    });

    input_string.erase(new_end, input_string.end());

    while (input_string[input_string.length() - 1] == ' ') {
        input_string.pop_back();
    }

    vector<string> splits;
    char delimiter = ' ';

    size_t i = 0;
    size_t pos = input_string.find(delimiter);

    while (pos != string::npos) {
        splits.push_back(input_string.substr(i, pos - i));

        i = pos + 1;
        pos = input_string.find(delimiter, i);
    }

    splits.push_back(input_string.substr(i, min(pos, input_string.length()) - i + 1));

    return splits;
}

