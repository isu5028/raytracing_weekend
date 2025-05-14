#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) { return degrees * pi / 180; }

inline double random_double() {
    // [0,1)の実数乱数を返す
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    //[min, max)の実数乱数を返す
    return min + (max - min) * random_double();
}

/*
inline double random_double(){
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt 19937 generator;
    return generator(distribution);
}
*/

inline double clamp(double x, double min, double max) {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

#include "ray.h"
#include "vec3.h"

#endif