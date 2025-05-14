#include <iostream>
#include <math.h>
#include <stdlib.h>

const double pi = 3.1415926535897932385;

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

int main() {
  for (int i = 0; i < 200; i++) {
    auto r1 = random_double();
    auto r2 = random_double();
    auto x = cos(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto y = sin(2*pi*r1)*2*sqrt(r2*(1-r2));
    auto z = 1 - 2*r2;
    std::cout << x << " " << y << " " << z << '\n';
  }
}
