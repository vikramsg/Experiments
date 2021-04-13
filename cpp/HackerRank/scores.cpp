#include<iostream>
#include<vector>

int main(int argc, char **argv)
{

  std::vector<int> scores {10, 20, 30, 40, 50, 60, 70};

  int n, acc;
  n = 0; acc = 0;
  for (int sc: scores)
  {
    acc = acc + sc;
    n++;
  }

  std::cout << n << "\t" << acc << "\t" << acc/n << std::endl;

  return 0;
}
