#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

const double pi = 3.1415926535897932385;

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
  // [min,max) の実数乱数を返す
  return min + (max-min)*random_double();
}

class vec3 {
public:
  vec3() : e{0,0,0} {}
  vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

  double x() const { return e[0]; }
  double y() const { return e[1]; }
  double z() const { return e[2]; }

public:
  double e[3];
};

// vec3 ユーティリティ関数

inline vec3 operator*(double t, const vec3 &v) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3 &v, double t) {
  return t * v;
}

vec3 random_unit_vector() {
  auto a = random_double(0, 2*pi);
  auto z = random_double(-1, 1);
  auto r = sqrt(1 - z*z);
  return vec3(r*cos(a), r*sin(a), z);
}

inline double pdf(const vec3& p) {
  return 1 / (4*pi);
}

int main() {
  int N = 1000000;
  auto sum = 0.0;
  for (int i = 0; i < N; i++) {
    vec3 d = random_unit_vector();
    auto cosine_squared = d.z()*d.z();
    sum += cosine_squared / pdf(d);
  }
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "I = " << sum/N << '\n';
}
