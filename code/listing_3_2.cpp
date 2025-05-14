#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
  // [min,max) の実数乱数を返す
  return min + (max-min)*random_double();
}

int main() {
  int inside_circle = 0;
  int runs = 0;
  std::cout << std::fixed << std::setprecision(12);
  while (true) {
    runs++;
    auto x = random_double(-1,1);
    auto y = random_double(-1,1);
    if (x*x + y*y < 1)
      inside_circle++;

    if (runs % 100000 == 0)
      std::cout << "Estimate of Pi = "
                << 4*double(inside_circle) / runs
                << '\n';
  }
}
