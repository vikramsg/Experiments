#include <iostream>
#include <string>

// Inspired by 
// https://stackoverflow.com/questions/1657883/variable-number-of-arguments-in-c
template <typename T>
void cons_func(T t) 
{
  std::cout << t << " ";
}

template<typename T, typename... Args>
void cons_func(T t, Args... args) // recursive variadic function
{
  std::cout << t <<" ";

  cons_func(args...) ;
}

template<typename... Args>
void cons_out(Args... args) 
{
  cons_func(args...);

  std::cout << std::endl;
}
