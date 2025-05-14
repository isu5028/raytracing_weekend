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
  int N = 1000;
  int inside_circle = 0;
  for (int i = 0; i < N; i++) {
    auto x = random_double(-1,1);
    auto y = random_double(-1,1);
    if (x*x + y*y < 1)
      inside_circle++;
  }
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "Estimate of Pi = " << 4*double(inside_circle) / N << '\n';
}
