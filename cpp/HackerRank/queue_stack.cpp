#include <iostream>

using namespace std;

class Solution {
    //Write your code here

   string stack;
   string queue;

  public:
    void pushCharacter(char s)
    {
      stack.push_back(s);
    }
    void enqueueCharacter(const char s)
    {
      queue.push_back(s);
    }
    char popCharacter()
    {
      char s = stack.back();
      stack.pop_back();
      return s;
    }
    char dequeueCharacter()
    {
      char s = queue.front();
      queue.erase(0, 1);
      return s;
    }

};

int main() {
    // read the string s.
    string s;
    getline(cin, s);
    
  	// create the Solution class object p.
    Solution obj;
    
    // push/enqueue all the characters of string s to stack.
    for (int i = 0; i < s.length(); i++) {
        obj.pushCharacter(s[i]);
        obj.enqueueCharacter(s[i]);
    }
    
    bool isPalindrome = true;
    
    // pop the top character from stack.
    // dequeue the first character from queue.
    // compare both the characters.
    for (int i = 0; i < s.length() / 2; i++) {
        char s, q;
        s = obj.popCharacter();
        q = obj.dequeueCharacter();
//        cout << s << "\t" << q << endl;
        if (s != q) {
            isPalindrome = false;
            
            break;
        }
    }
    
    // finally print whether string s is palindrome or not.
    if (isPalindrome) {
        cout << "The word, " << s << ", is a palindrome.";
    } else {
        cout << "The word, " << s << ", is not a palindrome.";
    }

    cout << endl;
    
    return 0;
}
