#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>
#include <time.h>

inline double pdf(double x) {
  return 0.5*x;
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
  srand((unsigned) time(NULL));
  int N = 1000000;
  auto sum = 0.0;
  for (int i = 0; i < N; i++) {
    auto x = sqrt(random_double(0,4));
    while (x == 0.0) {
      // PDF が 0.5*x の乱数で x が 0 になる確率は 0
      x = sqrt(random_double(0,4));
    }
    sum += x*x / pdf(x);
  }
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "I = " << sum/N << '\n';
}
