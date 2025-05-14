#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

const double pi = 3.1415926535897932385;

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

int main() {
  int N = 1000000;
  auto sum = 0.0;
  for (int i = 0; i < N; i++) {
    // auto r1 = random_double();
    auto r2 = random_double();
    // auto x = cos(2*pi*r1)*2*sqrt(r2*(1-r2));
    // auto y = sin(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto z = 1 - r2;
    sum += z*z*z / (1.0/(2.0*pi));
  }
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "Pi/2   = " << pi/2 << '\n';
  std::cout << "Estimate = " << sum/N << '\n';
}
