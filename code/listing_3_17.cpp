#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

const double pi = 3.1415926535897932385;

struct vec3 {
  vec3() : e{0,0,0} {}
  vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

  double e[3];
};

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

inline vec3 random_cosine_direction() {
  auto r1 = random_double();
  auto r2 = random_double();
  auto z = sqrt(1-r2);

  auto phi = 2*pi*r1;
  auto x = cos(phi)*sqrt(r2);
  auto y = sin(phi)*sqrt(r2);

  return vec3(x, y, z);
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
