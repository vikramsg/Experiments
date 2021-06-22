#include<iostream>

int main(int argc, char **argv)
{
  std::string s;

  std::cin >> s;

  int si;

  si = stoi(s);

  try
  {
    si = stoi(s);
  }
  catch (std::invalid_argument)
  {
    std::cout << "Bad String" << std::endl;
  }


  return 0;
}
