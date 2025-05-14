#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

inline double pdf(double x) {
    return 3*x*x/8;;
}

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
  // [min,max) の実数乱数を返す
  return min + (max-min)*random_double();
}

int main() {
  int N = 1;
  auto sum = 0.0;
  for (int i = 0; i < N; i++) {
    auto x = pow(random_double(0,8), 1./3.);
    sum += x*x / pdf(x);
  }
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "I = " << sum/N << '\n';
}
