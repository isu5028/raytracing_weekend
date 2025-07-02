#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <string>

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "my_code/stb_image_write.h"

using std::sqrt;
using std::shared_ptr;
using std::make_shared;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

class material;
class texture;
class hittable;
class pdf;
class ray;  // 前方宣言を追加

int id_counter = 0;


// ユーティリティ関数

inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180;
}

inline int random_int(int min, int max) {
  // {min, min+1, ..., max} から整数をランダムに返す
  return min + rand() % (max - min + 1);
}

inline double random_double() {
  // [0,1) の実数乱数を返す
  return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
  // [min,max) の実数乱数を返す
  return min + (max-min)*random_double();
}

inline double random_double_cpp() {
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
  static std::mt19937 generator;
  return distribution(generator);
}

inline double clamp(double x, double min, double max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

// ベクトルクラス

class vec3 {
public:
  vec3() : e{0,0,0} {}
  vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

  double x() const { return e[0]; }
  double y() const { return e[1]; }
  double z() const { return e[2]; }

  vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  double operator[](int i) const { return e[i]; }
  double& operator[](int i) { return e[i]; }

  vec3& operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  vec3& operator*=(const double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }

  vec3& operator/=(const double t) {
    return *this *= 1/t;
  }

  double length() const {
    return sqrt(length_squared());
  }

  double length_squared() const {
    return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
  }

  inline static vec3 random() {
    return vec3(random_double(), random_double(), random_double());
  }

  inline static vec3 random(double min, double max) {
    return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
  }

public:
  double e[3];
};

// vec3 ユーティリティ関数

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3 &v) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3 &v, double t) {
  return t * v;
}

inline vec3 operator/(vec3 v, double t) {
  return (1/t) * v;
}

inline double dot(const vec3 &u, const vec3 &v) {
  return u.e[0] * v.e[0]
    + u.e[1] * v.e[1]
    + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(vec3 v) {
  return v / v.length();
}

vec3 random_in_unit_sphere() {
  while (true) {
    auto p = vec3::random(-1,1);
    if (p.length_squared() >= 1) continue;
    return p;
  }
}

vec3 random_in_hemisphere(const vec3& normal) {
  vec3 in_unit_sphere = random_in_unit_sphere();
  if (dot(in_unit_sphere, normal) > 0.0)
    return in_unit_sphere; // in_unit_sphere は normal と同じ半球にある
  else
    return -in_unit_sphere;
}

vec3 random_unit_vector() {
  auto a = random_double(0, 2*pi);
  auto z = random_double(-1, 1);
  auto r = sqrt(1 - z*z);
  return vec3(r*cos(a), r*sin(a), z);
}

vec3 random_in_unit_disk() {
  while (true) {
    auto p = vec3(random_double(-1,1), random_double(-1,1), 0);
    if (p.length_squared() >= 1) continue;
    return p;
  }
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

vec3 reflect(const vec3& v, const vec3& n) {
  return v - 2*dot(v,n)*n;
}

vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
  auto cos_theta = dot(-uv, n);
  vec3 r_out_parallel =  etai_over_etat * (uv + cos_theta*n);
  vec3 r_out_perp = -sqrt(1.0 - r_out_parallel.length_squared()) * n;
  return r_out_parallel + r_out_perp;
}

// Overload for bool-returning refract (for dielectric::scatter)
bool refract(const vec3& v, const vec3& n, double ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    double dt = dot(uv, n);
    double discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else {
        return false;
    }
}

// vec3 の型エイリアス
using point3 = vec3;   // 3D 点
using color = vec3;    // RGB 色

class onb {
  public:
    onb() {}

    inline vec3 operator[](int i) const { return axis[i]; }

    vec3 u() const { return axis[0]; }
    vec3 v() const { return axis[1]; }
    vec3 w() const { return axis[2]; }

    vec3 local(double a, double b, double c) const {
      return a*u() + b*v() + c*w();
    }

    vec3 local(const vec3& a) const {
      return a.x()*u() + a.y()*v() + a.z()*w();
    }

    void build_from_w(const vec3&);

  public:
    vec3 axis[3];
};

void onb::build_from_w(const vec3& n) {
  axis[2] = unit_vector(n);
  vec3 a = (fabs(w().x()) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);
  axis[1] = unit_vector(cross(w(), a));
  axis[0] = cross(w(), v());
}

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

  // NaN の要素を 0 に置き換える (詳細は『週末レイトレーシング: 余生』を参照)
  if (r != r) r = 0.0;
  if (g != g) g = 0.0;
  if (b != b) b = 0.0;

  // 色の合計をサンプルの数で割り、gamma = 2.0 のガンマ補正を行う
  auto scale = 1.0 / samples_per_pixel;
  r = sqrt(scale * r);
  g = sqrt(scale * g);
  b = sqrt(scale * b);

  // 各成分を [0,255] に変換して出力する
  out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
      << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
      << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

// PNG出力用の色変換関数
void write_color_to_buffer(unsigned char* buffer, int index, color pixel_color, int samples_per_pixel) {
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

  // NaN の要素を 0 に置き換える
  if (r != r) r = 0.0;
  if (g != g) g = 0.0;
  if (b != b) b = 0.0;

  // 色の合計をサンプルの数で割り、gamma = 2.0 のガンマ補正を行う
  auto scale = 1.0 / samples_per_pixel;
  r = sqrt(scale * r);
  g = sqrt(scale * g);
  b = sqrt(scale * b);

  // 各成分を [0,255] に変換してバッファに保存
  buffer[index * 3 + 0] = static_cast<unsigned char>(256 * clamp(r, 0.0, 0.999));
  buffer[index * 3 + 1] = static_cast<unsigned char>(256 * clamp(g, 0.0, 0.999));
  buffer[index * 3 + 2] = static_cast<unsigned char>(256 * clamp(b, 0.0, 0.999));
}

class ray {
public:
  ray() {}
  ray(const point3& origin, const vec3& direction, double time = 0.0)
    : orig(origin), dir(direction), tm(time)
  {}

  point3 origin() const  { return orig; }
  vec3 direction() const { return dir; }
  double time() const    { return tm; }

  point3 at(double t) const {
    return orig + t*dir;
  }

public:
  point3 orig;
  vec3 dir;
  double tm;
};

// rayクラス定義後にグローバル変数を宣言
std::vector<std::shared_ptr<pdf>> path_pdfs;
std::vector<ray>                 path_rays;
int                               current_path_length = 0;
const int depth_max = 50; // 最大深度を定義

double hit_sphere(const point3& center, double radius, const ray& r) {
  vec3 oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius*radius;
  auto discriminant = half_b*half_b - a*c;

  if (discriminant < 0) {
    return -1.0;
  } else {
    return (-half_b - sqrt(discriminant) ) / a;
  }
}

class aabb {
public:
  aabb() {}
  aabb(const point3& a, const point3& b) { _min = a; _max = b;}

  point3 min() const {return _min; }
  point3 max() const {return _max; }
  bool hit(const ray& r, double tmin, double tmax) const;

  point3 _min;
  point3 _max;
};

inline bool aabb::hit(const ray& r, double tmin, double tmax) const {
  for (int a = 0; a < 3; a++) {
    auto invD = 1.0f / r.direction()[a];
    auto t0 = (min()[a] - r.origin()[a]) * invD;
    auto t1 = (max()[a] - r.origin()[a]) * invD;
    if (invD < 0.0f)
      std::swap(t0, t1);
    tmin = t0 > tmin ? t0 : tmin;
    tmax = t1 < tmax ? t1 : tmax;
    if (tmax <= tmin)
      return false;
  }
  return true;
}

aabb surrounding_box(aabb box0, aabb box1) {
  point3 small(fmin(box0.min().x(), box1.min().x()),
               fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z()));

  point3 big(fmax(box0.max().x(), box1.max().x()),
             fmax(box0.max().y(), box1.max().y()),
             fmax(box0.max().z(), box1.max().z()));

  return aabb(small,big);
}

struct hit_record {
  point3 p;
  vec3 normal;
  shared_ptr<material> mat_ptr;
  double t;
  double u;
  double v;
  bool front_face;
  int object_id = 0;

  inline void set_face_normal(const ray& r, const vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal :-outward_normal;
  }
};

class hittable {
public:
  virtual ~hittable() {}
  virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const = 0;

  virtual double pdf_value(const point3& o, const vec3& v) const {
    return 0.0;
  }

  virtual vec3 random(const vec3& o) const {
    return vec3(1, 0, 0);
  }

  int object_id = 0; 
};

class sphere: public hittable {
public:
  sphere() {}
  sphere(point3 cen, double r, shared_ptr<material> m, int id = 0)
    : center(cen), radius(r), mat_ptr(m) { object_id = id; };

  virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;
  double pdf_value(const point3& o, const vec3& v) const;
  vec3 random(const point3& o) const;

public:
  point3 center;
  double radius;
  shared_ptr<material> mat_ptr;
};

void get_sphere_uv(const vec3& p, double& u, double& v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1-(phi + pi) / (2*pi);
  v = (theta + pi/2) / pi;
}

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  vec3 oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius*radius;
  auto discriminant = half_b*half_b - a*c;

  if (discriminant > 0) {
    auto root = sqrt(discriminant);
    auto temp = (-half_b - root)/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
      rec.object_id = object_id;
      return true;
    }
    temp = (-half_b + root) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
      rec.object_id = object_id; 
      return true;
    }
  }
  return false;
}

bool sphere::bounding_box(double t0, double t1, aabb& output_box) const {
  output_box = aabb(center - vec3(radius, radius, radius),
                    center + vec3(radius, radius, radius));
  return true;
}

double sphere::pdf_value(const point3& o, const vec3& v) const {
  hit_record rec;
  if (!this->hit(ray(o, v), 0.001, infinity, rec))
    return 0;

  auto cos_theta_max = sqrt(1 - radius*radius/(center-o).length_squared());
  auto solid_angle = 2*pi*(1-cos_theta_max);

  return  1 / solid_angle;
}

inline vec3 random_to_sphere(double radius, double distance_squared) {
  auto r1 = random_double();
  auto r2 = random_double();
  auto z = 1 + r2*(sqrt(1-radius*radius/distance_squared) - 1);

  auto phi = 2*pi*r1;
  auto x = cos(phi)*sqrt(1-z*z);
  auto y = sin(phi)*sqrt(1-z*z);

  return vec3(x, y, z);
}

vec3 sphere::random(const point3& o) const {
  vec3 direction = center - o;
  auto distance_squared = direction.length_squared();
  onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(random_to_sphere(radius, distance_squared));
}

class hittable_list: public hittable {
public:
  hittable_list() {}
  hittable_list(shared_ptr<hittable> object) { add(object); }

  void clear() { objects.clear(); }
  void add(shared_ptr<hittable> object) { objects.push_back(object); }

  virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;
  double pdf_value(const point3& o, const vec3& v) const;
  vec3 random(const vec3& o) const;

public:
  std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  hit_record temp_rec;
  bool hit_anything = false;
  auto closest_so_far = t_max;

  for (const auto& object : objects) {
    if (object->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

bool hittable_list::bounding_box(double t0, double t1, aabb& output_box) const {
  if (objects.empty()) return false;

  aabb temp_box;
  bool first_box = true;

  for (const auto& object : objects) {
    if (!object->bounding_box(t0, t1, temp_box)) return false;
    output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
    first_box = false;
  }

  return true;
}

double hittable_list::pdf_value(const point3& o, const vec3& v) const {
  auto weight = 1.0/objects.size();
  auto sum = 0.0;

  for (const auto& object : objects)
    sum += weight * object->pdf_value(o, v);

  return sum;
}

vec3 hittable_list::random(const vec3& o) const {
  auto int_size = static_cast<int>(objects.size());
  return objects[random_int(0, int_size-1)]->random(o);
}

class moving_sphere : public hittable {
public:
  moving_sphere() {}
  moving_sphere(point3 cen0, point3 cen1, double t0, double t1, double r, shared_ptr<material> m, int id = 0)
    : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m)
  { object_id = id; };

  virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;

  point3 center(double time) const;

public:
  point3 center0, center1;
  double time0, time1;
  double radius;
  shared_ptr<material> mat_ptr;
};

point3 moving_sphere::center(double time) const{
  return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

bool moving_sphere::hit(
  const ray& r,
  double t_min,
  double t_max,
  hit_record& rec
) const {
  vec3 oc = r.origin() - center(r.time());
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius*radius;

  auto discriminant = half_b*half_b - a*c;

  if (discriminant > 0) {
    auto root = sqrt(discriminant);

    auto temp = (-half_b - root)/a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      auto outward_normal = (rec.p - center(r.time())) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      return true;
    }

    temp = (-half_b + root) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      auto outward_normal = (rec.p - center(r.time())) / radius;
      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

bool moving_sphere::bounding_box(double t0, double t1, aabb& output_box) const {
  aabb box0(center(t0) - vec3(radius, radius, radius),
            center(t0) + vec3(radius, radius, radius));
  aabb box1(center(t1) - vec3(radius, radius, radius),
            center(t1) + vec3(radius, radius, radius));
  output_box = surrounding_box(box0, box1);
  return true;
}

class bvh_node : public hittable {
public:
  bvh_node();

  bvh_node(hittable_list& list, double time0, double time1)
    : bvh_node(list.objects, 0, list.objects.size(), time0, time1)
  {}

  bvh_node(std::vector<shared_ptr<hittable>>& objects,
           size_t start, size_t end, double time0, double time1);

  virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;

public:
  shared_ptr<hittable> left;
  shared_ptr<hittable> right;
  aabb box;
};

bool bvh_node::bounding_box(double t0, double t1, aabb& output_box) const {
  output_box = box;
  return true;
}

bool bvh_node::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  if (!box.hit(r, t_min, t_max))
    return false;

  bool hit_left = left->hit(r, t_min, t_max, rec);
  bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

  return hit_left || hit_right;
}

inline bool box_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis) {
  aabb box_a;
  aabb box_b;

  if (!a->bounding_box(0,0, box_a) || !b->bounding_box(0,0, box_b))
    std::cerr << "No bounding box in bvh_node constructor.\n";

  return box_a.min().e[axis] < box_b.min().e[axis];
}

bool box_x_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
  return box_compare(a, b, 0);
}

bool box_y_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
  return box_compare(a, b, 1);
}

bool box_z_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
  return box_compare(a, b, 2);
}

bvh_node::bvh_node(
  std::vector<shared_ptr<hittable>>& objects,
  size_t start, size_t end, double time0, double time1
) {
  int axis = random_int(0,2);
  auto comparator = (axis == 0) ? box_x_compare
                  : (axis == 1) ? box_y_compare
                  : box_z_compare;

  size_t object_span = end - start;

  if (object_span == 1) {
    left = right = objects[start];
  } else if (object_span == 2) {
    if (comparator(objects[start], objects[start+1])) {
      left = objects[start];
      right = objects[start+1];
    } else {
      left = objects[start+1];
      right = objects[start];
    }
  } else {
    std::sort(objects.begin() + start, objects.begin() + end, comparator);

    auto mid = start + object_span/2;
    left = make_shared<bvh_node>(objects, start, mid, time0, time1);
    right = make_shared<bvh_node>(objects, mid, end, time0, time1);
  }

  aabb box_left, box_right;

  if (  !left->bounding_box (time0, time1, box_left)
        || !right->bounding_box(time0, time1, box_right))
    std::cerr << "No bounding box in bvh_node constructor.\n";

  box = surrounding_box(box_left, box_right);
}

class xy_rect: public hittable {
public:
  xy_rect() {}

  xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, shared_ptr<material> mat, int id = 0)
    : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) { object_id = id; };

  virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const;

  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    // AABB の辺の長さはゼロであってはならないので、
    // z 方向に少しだけ厚みを持たせる
    output_box = aabb(point3(x0,y0, k-0.0001), point3(x1, y1, k+0.0001));
    return true;
  }

public:
  double x0, x1, y0, y1, k;
  shared_ptr<material> mp;
};

class xz_rect: public hittable {
public:
  xz_rect() {}

  xz_rect(double _x0, double _x1, double _z0, double _z1, double _k, shared_ptr<material> mat, int id = 0)
    : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) { object_id = id; };

  virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const;

  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    // AABB の辺の長さはゼロであってはならないので、
    // y 方向に少しだけ厚みを持たせる
    output_box = aabb(point3(x0,k-0.0001,z0), point3(x1, k+0.0001, z1));
    return true;
  }

  virtual double pdf_value(const point3& origin, const vec3& v) const {
    hit_record rec;
    if (!this->hit(ray(origin, v), 0.001, infinity, rec))
      return 0;

    auto area = (x1-x0)*(z1-z0);
    auto distance_squared = rec.t * rec.t * v.length_squared();
    auto cosine = fabs(dot(v, rec.normal) / v.length());

    return distance_squared / (cosine * area);
  }

  virtual vec3 random(const point3& origin) const {
    auto random_point = point3(random_double(x0,x1), k, random_double(z0,z1));
    return random_point - origin;
  }

public:
  double x0, x1, z0, z1, k;
  shared_ptr<material> mp;
};

class yz_rect: public hittable {
public:
  yz_rect() {}

  yz_rect(double _y0, double _y1, double _z0, double _z1, double _k, shared_ptr<material> mat, int id = 0)
    : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) { object_id = id; };

  virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const;

  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    // AABB の辺の長さはゼロであってはならないので、
    // x 方向に少しだけ厚みを持たせる
    output_box = aabb(point3(k-0.0001, y0, z0), point3(k+0.0001, y1, z1));
    return true;
  }

public:
  double y0, y1, z0, z1, k;
  shared_ptr<material> mp;
};

bool xy_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const {
  auto t = (k-r.origin().z()) / r.direction().z();
  if (t < t0 || t > t1)
    return false;
  auto x = r.origin().x() + t*r.direction().x();
  auto y = r.origin().y() + t*r.direction().y();
  if (x < x0 || x > x1 || y < y0 || y > y1)
    return false;
  rec.u = (x-x0)/(x1-x0);
  rec.v = (y-y0)/(y1-y0);
  rec.t = t;
  auto outward_normal = vec3(0, 0, 1);
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);
  rec.object_id = object_id; 
  return true;
}

bool xz_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const {
  auto t = (k-r.origin().y()) / r.direction().y();
  if (t < t0 || t > t1)
    return false;
  auto x = r.origin().x() + t*r.direction().x();
  auto z = r.origin().z() + t*r.direction().z();
  if (x < x0 || x > x1 || z < z0 || z > z1)
    return false;
  rec.u = (x-x0)/(x1-x0);
  rec.v = (z-z0)/(z1-z0);
  rec.t = t;
  auto outward_normal = vec3(0, 1, 0);
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);
  rec.object_id = object_id; 
  return true;
}

bool yz_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const {
  auto t = (k-r.origin().x()) / r.direction().x();
  if (t < t0 || t > t1)
    return false;
  auto y = r.origin().y() + t*r.direction().y();
  auto z = r.origin().z() + t*r.direction().z();
  if (y < y0 || y > y1 || z < z0 || z > z1)
    return false;
  rec.u = (y-y0)/(y1-y0);
  rec.v = (z-z0)/(z1-z0);
  rec.t = t;
  auto outward_normal = vec3(1, 0, 0);
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);
  rec.object_id = object_id; 
  return true;
}

class box: public hittable  {
public:
  box() {}
  box(const point3& p0, const point3& p1, shared_ptr<material> ptr, int id = 0);

  virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const;

  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    output_box = aabb(box_min, box_max);
    return true;
  }

public:
  point3 box_min;
  point3 box_max;
  hittable_list sides;
};

box::box(const point3& p0, const point3& p1, shared_ptr<material> ptr, int id) {
  box_min = p0;
  box_max = p1;
  object_id = id;

  sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr, id));
  sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr, id));

  sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr, id));
  sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr, id));

  sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr, id));
  sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr, id));
}

bool box::hit(const ray& r, double t0, double t1, hit_record& rec) const {
  rec.object_id = object_id; 
  return sides.hit(r, t0, t1, rec);
}

class translate : public hittable {
public:
  translate(shared_ptr<hittable> p, const vec3& displacement)
    : ptr(p), offset(displacement) {}

  virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;

public:
  shared_ptr<hittable> ptr;
  vec3 offset;
};

bool translate::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  ray moved_r(r.origin() - offset, r.direction(), r.time());
  if (!ptr->hit(moved_r, t_min, t_max, rec))
    return false;

  rec.p += offset;
  rec.set_face_normal(moved_r, rec.normal);

  return true;
}

bool translate::bounding_box(double t0, double t1, aabb& output_box) const {
  if (!ptr->bounding_box(t0, t1, output_box))
    return false;

  output_box = aabb(
                    output_box.min() + offset,
                    output_box.max() + offset);

  return true;
}

class rotate_y : public hittable {
public:
  rotate_y(shared_ptr<hittable> p, double angle);

  virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    output_box = bbox;
    return hasbox;
  }

public:
  shared_ptr<hittable> ptr;
  double sin_theta;
  double cos_theta;
  bool hasbox;
  aabb bbox;
};

rotate_y::rotate_y(shared_ptr<hittable> p, double angle) : ptr(p) {
  auto radians = degrees_to_radians(angle);
  sin_theta = sin(radians);
  cos_theta = cos(radians);
  hasbox = ptr->bounding_box(0, 1, bbox);

  point3 min( infinity,  infinity,  infinity);
  point3 max(-infinity, -infinity, -infinity);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        auto x = i*bbox.max().x() + (1-i)*bbox.min().x();
        auto y = j*bbox.max().y() + (1-j)*bbox.min().y();
        auto z = k*bbox.max().z() + (1-k)*bbox.min().z();

        auto newx =  cos_theta*x + sin_theta*z;
        auto newz = -sin_theta*x + cos_theta*z;

        vec3 tester(newx, y, newz);

        for (int c = 0; c < 3; c++) {
          min[c] = fmin(min[c], tester[c]);
          max[c] = fmax(max[c], tester[c]);
        }
      }
    }
  }

  bbox = aabb(min, max);
}

bool rotate_y::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  auto origin = r.origin();
  auto direction = r.direction();

  origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
  origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];

  direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
  direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];

  ray rotated_r(origin, direction, r.time());

  if (!ptr->hit(rotated_r, t_min, t_max, rec))
    return false;

  auto p = rec.p;
  auto normal = rec.normal;

  p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
  p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];

  normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
  normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];

  rec.p = p;
  rec.set_face_normal(rotated_r, normal);

  return true;
}

inline double perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
  auto uu = u*u*(3-2*u);
  auto vv = v*v*(3-2*v);
  auto ww = w*w*(3-2*w);
  auto accum = 0.0;

  for (int i=0; i < 2; i++)
    for (int j=0; j < 2; j++)
      for (int k=0; k < 2; k++) {
        vec3 weight_v(u-i, v-j, w-k);
        accum += (i*uu + (1-i)*(1-uu)) *
                 (j*vv + (1-j)*(1-vv)) *
                 (k*ww + (1-k)*(1-ww)) * dot(c[i][j][k], weight_v);
      }

  return accum;
}

class perlin {
public:
  perlin() {
    ranvec = new vec3[point_count];

    for (int i = 0; i < point_count; ++i) {
      ranvec[i] = unit_vector(vec3::random(-1,1));
    }

    perm_x = perlin_generate_perm();
    perm_y = perlin_generate_perm();
    perm_z = perlin_generate_perm();
  }

  ~perlin() {
    delete[] ranvec;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
  }

  double noise(const point3& p) const {
    auto u = p.x() - floor(p.x());
    auto v = p.y() - floor(p.y());
    auto w = p.z() - floor(p.z());
    int i = floor(p.x());
    int j = floor(p.y());
    int k = floor(p.z());
    vec3 c[2][2][2];

    for (int di=0; di < 2; di++)
      for (int dj=0; dj < 2; dj++)
        for (int dk=0; dk < 2; dk++)
          c[di][dj][dk] = ranvec[
            perm_x[(i+di) & 255] ^
            perm_y[(j+dj) & 255] ^
            perm_z[(k+dk) & 255]
          ];

    return perlin_interp(c, u, v, w);
  }

  double turb(const point3& p, int depth=7) const {
    auto accum = 0.0;
    auto temp_p = p;
    auto weight = 1.0;

    for (int i = 0; i < depth; i++) {
      accum += weight*noise(temp_p);
      weight *= 0.5;
      temp_p *= 2;
    }

    return fabs(accum);
  }

private:
  static const int point_count = 256;
  vec3* ranvec;
  int* perm_x;
  int* perm_y;
  int* perm_z;

  static int* perlin_generate_perm() {
    auto p = new int[point_count];

    for (int i = 0; i < perlin::point_count; i++)
      p[i] = i;

    permute(p, point_count);

    return p;
  }

  static void permute(int* p, int n) {
    for (int i = n-1; i > 0; i--) {
      int target = random_int(0, i);
      int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }
};

class texture {
public:
  virtual ~texture() {};
  virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture {
public:
  solid_color() {}
  solid_color(color c) : color_value(c) {}

  solid_color(double red, double green, double blue)
    : solid_color(color(red,green,blue)) {}

  virtual color value(double u, double v, const vec3& p) const {
    return color_value;
  }

private:
  color color_value;
};

class checker_texture : public texture {
public:
  checker_texture() {}
  checker_texture(shared_ptr<texture> t0, shared_ptr<texture> t1): even(t0), odd(t1) {}

  virtual color value(double u, double v, const point3& p) const {
    auto sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
    if (sines < 0)
      return odd->value(u, v, p);
    else
      return even->value(u, v, p);
  }

public:
  shared_ptr<texture> even;
  shared_ptr<texture> odd;
};

class noise_texture : public texture {
public:
  noise_texture() {}
  noise_texture(double sc) : scale(sc) {}

  virtual color value(double u, double v, const point3& p) const {
    return color(1,1,1) * 0.5 * (1 + sin(scale*p.z() + 10*noise.turb(p)));
  }

public:
  perlin noise;
  double scale;
};

class image_texture : public texture {
  public:
    const static int bytes_per_pixel = 3;

    image_texture()
      : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

    image_texture(const char* filename) {
      auto components_per_pixel = bytes_per_pixel;

      data = stbi_load(filename, &width, &height, &components_per_pixel, components_per_pixel);

      if (!data) {
        std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
        width = height = 0;
      }

      bytes_per_scanline = bytes_per_pixel * width;
    }

    ~image_texture() {
      delete data;
    }

    virtual color value(double u, double v, const vec3& p) const {
      // テクスチャのデータがない場合には、そのことが分かるようにシアン色を返す。
      if (data == nullptr)
        return color(0,1,1);

      // 入力されたテクスチャ座標を [0,1] で切り捨てる。
      u = clamp(u, 0.0, 1.0);
      v = 1.0 - clamp(v, 0.0, 1.0);  // v を反転させて画像の座標系に合わせる。

      auto i = static_cast<int>(u * width);
      auto j = static_cast<int>(v * height);

      // 整数座標をさらに切り捨てる (テクスチャ座標は 1.0 になってはいけない)。
      if (i >= width)  i = width-1;
      if (j >= height) j = height-1;

      const auto color_scale = 1.0 / 255.0;
      auto pixel = data + j*bytes_per_scanline + i*bytes_per_pixel;

      return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

  private:
    unsigned char *data;
    int width, height;
    int bytes_per_scanline;
};

struct scatter_record {
  ray specular_ray;
  bool is_specular;
  color attenuation;
  shared_ptr<pdf> pdf_ptr;
};

class pdf {
public:
  virtual ~pdf() {}

  virtual double value(const vec3& direction) const = 0;
  virtual vec3 generate() const = 0;
};

class cosine_pdf : public pdf {
public:
  cosine_pdf(const vec3& w) { uvw.build_from_w(w); }

  virtual double value(const vec3& direction) const {
    auto cosine = dot(unit_vector(direction), uvw.w());
    return (cosine <= 0) ? 0 : cosine/pi;
  }

  virtual vec3 generate() const {
    return uvw.local(random_cosine_direction());
  }

public:
  onb uvw;
};

class hittable_pdf : public pdf {
public:
  hittable_pdf(shared_ptr<hittable> p, const point3& origin)
    : ptr(p), o(origin) {}

  virtual double value(const vec3& direction) const {
    return ptr->pdf_value(o, direction);
  }

  virtual vec3 generate() const {
    return ptr->random(o);
  }

public:
  shared_ptr<hittable> ptr;
  point3 o;
};

class mixture_pdf : public pdf {
public:
  mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
    p[0] = p0;
    p[1] = p1;
  }

  virtual double value(const vec3& direction) const {
    return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
  }

  virtual vec3 generate() const {
    if (random_double() < 0.5)
      return p[0]->generate();
    else
      return p[1]->generate();
  }

public:
  shared_ptr<pdf> p[2];
};

class material {
public:
  virtual ~material() {}
  virtual color emitted(
    const ray& r_in, const hit_record& rec, double u, double v, const point3& p
  ) const {
    return color(0,0,0);
  }

  virtual bool scatter(
    const ray& r_in, const hit_record& rec, scatter_record& srec
  ) const {
    return false;
  }

  virtual double scattering_pdf(
    const ray& r_in, const hit_record& rec, const ray& scattered
  ) const {
    return 0;
  }
};;

class diffuse_light : public material  {
public:
  diffuse_light(shared_ptr<texture> a) : emit(a) {}

  virtual bool scatter(
    const ray& r_in,
    const hit_record& rec,
    color& attenuation,
    ray& scattered
  ) const {
    return false;
  }

  virtual color emitted(
    const ray &r_in,
    const hit_record& rec,
    double u,
    double v,
    const point3& p) const {
    if (rec.front_face)
      return emit->value(u, v, p);
    else
      return color(0,0,0);
  }

public:
  shared_ptr<texture> emit;
};

class lambertian : public material {
public:
  lambertian(shared_ptr<texture> a) : albedo(a) {}

  virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override;

  double scattering_pdf(
    const ray& r_in, const hit_record& rec, const ray& scattered
  ) const {
    auto cosine = dot(rec.normal, unit_vector(scattered.direction()));
    return cosine < 0 ? 0 : cosine/pi;
  }

  shared_ptr<texture> albedo;
};

bool lambertian::scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const {
    srec.is_specular   = false;
    srec.attenuation   = albedo->value(rec.u, rec.v, rec.p);
    // サンプリング PDF を生成して記録
    auto pdf_ptr = std::make_shared<cosine_pdf>(rec.normal);
    srec.pdf_ptr      = pdf_ptr;
    path_pdfs.push_back(pdf_ptr);          // ★ここで記録
    return true;
}

class metal : public material {
public:
  metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

  virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const;

public:
  color albedo;
  double fuzz;
};

// Dirac delta PDF: ある方向 direction のみ確率 1、他は 0 とする
class specular_pdf : public pdf {
public:
    vec3 dir;
    specular_pdf(const vec3& direction) : dir(unit_vector(direction)) {}
    virtual double value(const vec3& w) const override {
        // 完全に一致すれば 1、そうでなければ 0
        return (dot(unit_vector(w), dir) > 1.0 - 1e-6) ? 1.0 : 0.0;
    }
    virtual vec3 generate() const override {
        return dir;
    }
};

bool metal::scatter(
    const ray& r_in,
    const hit_record& rec,
    scatter_record& srec
) const {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    vec3 scattered_dir = reflected + fuzz * random_in_unit_sphere();

    // スペキュラーレイをセット
    srec.specular_ray = ray(rec.p, scattered_dir, r_in.time());
    srec.attenuation = albedo;
    srec.is_specular = true;

    // ★Dirac δ PDF を作って記録
    auto pdf_ptr = std::make_shared<specular_pdf>(scattered_dir);
    srec.pdf_ptr = pdf_ptr;
    path_pdfs.push_back(pdf_ptr);

    return true;
}


double schlick(double cosine, double ref_idx) {
  auto r0 = (1-ref_idx) / (1+ref_idx);
  r0 = r0*r0;
  return r0 + (1-r0)*pow((1 - cosine),5);
}

class dielectric : public material {
public:
  dielectric(double ri) : ref_idx(ri) {}

  virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const;

  double ref_idx;
};

bool dielectric::scatter(
    const ray& r_in,
    const hit_record& rec,
    scatter_record& srec
) const {
    srec.is_specular = true;        // 全てスペキュラー扱い
    srec.attenuation = color(1.0, 1.0, 1.0);  // 屈折媒質は損失なし

    vec3 outward_normal;
    double ni_over_nt;
    double cosine;
    if (dot(r_in.direction(), rec.normal) > 0) {
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
    } else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0 / ref_idx;
        cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }

    vec3 refracted;
    double reflect_prob;
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
        reflect_prob = schlick(cosine, ref_idx);
    } else {
        reflect_prob = 1.0;
    }

    vec3 scattered_dir;
    if (random_double() < reflect_prob) {
        scattered_dir = reflect(r_in.direction(), rec.normal);
    } else {
        scattered_dir = refracted;
    }

    // スペキュラーレイをセット
    srec.specular_ray = ray(rec.p, scattered_dir, r_in.time());

    // ★Dirac δ PDF を作って記録
    auto pdf_ptr = std::make_shared<specular_pdf>(scattered_dir);
    srec.pdf_ptr = pdf_ptr;
    path_pdfs.push_back(pdf_ptr);

    return true;
}


class isotropic : public material {
public:
  isotropic(shared_ptr<texture> a) : albedo(a) {}

  virtual bool scatter(
    const ray& r_in,
    const hit_record& rec,
    color& attenuation,
    ray& scattered
  ) const {
    scattered = ray(rec.p, random_in_unit_sphere(), r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

public:
  shared_ptr<texture> albedo;
};

class constant_medium : public hittable {
public:
  constant_medium(shared_ptr<hittable> b, double d, shared_ptr<texture> a)
    : boundary(b), neg_inv_density(-1/d) {
    phase_function = make_shared<isotropic>(a);
  }

  virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;

  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    return boundary->bounding_box(t0, t1, output_box);
  }

public:
  shared_ptr<hittable> boundary;
  shared_ptr<material> phase_function;
  double neg_inv_density;
};

bool constant_medium::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  // デバッグ中には低い確率でサンプルの様子を出力する。
  // enableDebug を true にすると有効になる。
  const bool enableDebug = false;
  const bool debugging = enableDebug && random_double() < 0.00001;

  hit_record rec1, rec2;

  if (!boundary->hit(r, -infinity, infinity, rec1))
    return false;

  if (!boundary->hit(r, rec1.t+0.0001, infinity, rec2))
    return false;

  if (debugging) std::cerr << "\nt0=" << rec1.t << ", t1=" << rec2.t << '\n';

  if (rec1.t < t_min) rec1.t = t_min;
  if (rec2.t > t_max) rec2.t = t_max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  const auto ray_length = r.direction().length();
  const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
  const auto hit_distance = neg_inv_density * log(random_double());

  if (hit_distance > distance_inside_boundary)
    return false;

  rec.t = rec1.t + hit_distance / ray_length;
  rec.p = r.at(rec.t);

  if (debugging) {
    std::cerr << "hit_distance = " <<  hit_distance << '\n'
              << "rec.t = " <<  rec.t << '\n'
              << "rec.p = " <<  rec.p << '\n';
  }

  rec.normal = vec3(1,0,0); // どんな値でもよい
  rec.front_face = true;    // 同じくどんな値でもよい
  rec.mat_ptr = phase_function;

  return true;
}

class flip_face : public hittable {
public:
  flip_face(shared_ptr<hittable> p) : ptr(p) {}

  virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    if (!ptr->hit(r, t_min, t_max, rec))
      return false;

    rec.front_face = !rec.front_face;
    return true;
  }

  virtual bool bounding_box(double t0, double t1, aabb& output_box) const {
    return ptr->bounding_box(t0, t1, output_box);
  }

public:
  shared_ptr<hittable> ptr;
};

class camera {
public:
  camera() {}

  camera(
   point3 lookfrom,
   point3 lookat,
   vec3   vup,
   double vfov, // 垂直方向の視野角 (弧度法)
   double aspect_ratio,
   double aperture,
   double focus_dist,
   double t0 = 0,
   double t1 = 0
 ) {
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta/2);
    auto viewport_height = 2.0 * h;
    auto viewport_width = aspect_ratio * viewport_height;

    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    origin = lookfrom;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

    lens_radius = aperture / 2;
    time0 = t0;
    time1 = t1;
  }

  ray get_ray(double s, double t) const {
    vec3 rd = lens_radius * random_in_unit_disk();
    vec3 offset = u * rd.x() + v * rd.y();

    return ray(
      origin + offset,
      lower_left_corner + s*horizontal + t*vertical - origin - offset,
      random_double(time0, time1)
    );
  }

public:
  // Getter methods for u and v
  vec3 get_u() const { return u; }
  vec3 get_v() const { return v; }

private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  double lens_radius;
  double time0, time1;  // シャッターの開閉時間
};

// 特徴線の設定
namespace FeatureLine {
    const color line_color = color(1.0, 0.0, 0.0);
    const int samples_per_query = 16; // 論文の図9参照 
    const double line_width_screen = 0.0003; // スクリーン空間での線の太さ（正規化）

    // 特徴線のメトリクス（しきい値） 
    const double normal_threshold = 0.9; // (1.0 - n1・n2) > threshold
    const double depth_threshold_beta = 0.01;
}

// メトリクス評価関数
bool evaluate_line_metrics(const hit_record& query_hit, const hit_record& sample_hit, const ray& query_ray, const ray& sample_ray) {
  
    // 1. オブジェクトIDが異なる場合 (シルエット) 
    if (query_hit.object_id != sample_hit.object_id) {
        return true;
    }
        
    /*
    // 2. 法線の差 (クリース) 
    if (dot(query_hit.normal, sample_hit.normal) < (1.0 - FeatureLine::normal_threshold)) {
        return true;
    }
    */

    // 3. 深度差の判定（修正版）
    /*
    // それぞれのレイの方向ベクトルの長さを考慮して、正確な深度を計算
    double depth_q = query_hit.t * query_ray.direction().length();
    // sample_rayを正しく使用して深度を計算
    double depth_s = sample_hit.t * sample_ray.direction().length();

    // 論文の式(14)に基づくしきい値の計算
    // 分母のdot積では、必ず正規化した単位ベクトルを使用する
    vec3 query_dir_unit = unit_vector(query_ray.direction());
    double denominator = fabs(dot(query_dir_unit, query_hit.normal));

    // ゼロ除算を避けるための安全対策
    if (denominator < 1e-6) {
        // 視線とサーフェスがほぼ平行な場合（グレイジングアングル）、
        // 不安定になるため特徴線とはみなさない
        return false;
    }
        

    double dist_3d = (sample_hit.p - query_hit.p).length();
    double depth_diff_threshold = FeatureLine::depth_threshold_beta
                                  * fmin(depth_q, depth_s)
                                  * dist_3d / denominator;

    if (fabs(depth_q - depth_s) > depth_diff_threshold) {
        return true;
    }
        */

    return false;
}

// 特徴線検出のメイン関数
bool check_for_feature_line(
    const ray& r,
    const hit_record& query_hit,
    const hittable& world,
    const camera& cam // line_widthを計算するためにカメラ情報が必要
) {
    // 論文の Algorithm 3 に相当 
        vec3 offset_uv = cam.get_u() * (random_double() - 0.5) + cam.get_v() * (random_double() - 0.5);

    for (int i = 0; i < FeatureLine::samples_per_query; ++i) {
        // 論文の図3c, 図4のように、クエリレイの周りにサンプルを生成する 
        // ここではカメラのu,vベクトルを使ってスクリーンに平行な円盤状にサンプルを飛ばす簡易的な方法を実装
        vec3 offset_uv = cam.get_u() * (random_double() - 0.5) + cam.get_v() * (random_double() - 0.5);
        double scaled_line_width = FeatureLine::line_width_screen * query_hit.t; // 簡易的な錐台の計算 
        
        // サンプルレイを作成
        ray sample_ray(r.origin(), unit_vector(r.direction() + offset_uv * scaled_line_width));
        
        hit_record sample_hit;
        if (world.hit(sample_ray, 0.001, infinity, sample_hit)) {
            if (evaluate_line_metrics(query_hit, sample_hit, r, sample_ray)) {
                return true; // 特徴線を発見
            }
        }
    }
    return false; // 特徴線は見つからなかった
}

// modified_path_length は特徴線検出で打ち切られたあとの頂点数
double recomputeModifiedPathPDF(int modified_path_length) {
    double p = 1.0;
    // path_pdfs[0] はセンサー→第1頂点、path_pdfs[1] は第1→第2頂点、… のはず
    for (int i = 0; i < modified_path_length - 1; ++i) {
        // path_pdfs[i]->value はその頂点間を生成した PDF
        p *= path_pdfs[i]->value(path_rays[i+1].direction());
    }
    return p;
}


color ray_color(
  const ray& r,
  const color& background,
  const hittable& world,
  shared_ptr<hittable> lights,
  int depth,
  const camera& cam
) {
   hit_record rec;

  // --- 1) 深度リミット到達チェック ---
  if (depth <= 0)
    return color(0,0,0);

  // --- 2) パス情報の初期化（最初の呼び出し時にだけ） ---
  //    depth_max はレンダラー側で最初に指定する最大深度
  if (depth == depth_max) {
    path_pdfs.clear();
    path_rays.clear();
  }

  // --- 3) レイ交差判定 ---
  if (!world.hit(r, 0.001, infinity, rec))
    return background;

  // --- 4) パス追跡情報を記録 ---
  path_rays.push_back(r);
  // （PDF は scatter 後に記録します）

  // --- 5) 特徴線検出 ---
  if (check_for_feature_line(r, rec, world, cam)) {
    // 打ち切れた頂点数 L は path_rays.size() と同じ
    int L = static_cast<int>(path_rays.size());

    // PDF を再計算する関数（論文式(8)）
    double p = 1.0;
    for (int i = 0; i < L-1; ++i) {
      // path_pdfs[i] は i→i+1 のサンプリング PDF
      p *= path_pdfs[i]->value(path_rays[i+1].direction());
    }

    // f(x')=線の色, p(x')=p として重み付けを返す
    return FeatureLine::line_color / p;
  }

  // --- 6) マテリアルからの光（Emission）と散乱処理 ---
  scatter_record srec;
  color emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
  if (!rec.mat_ptr->scatter(r, rec, srec))
    return emitted;

  // specular（鏡面反射）なら再帰
  if (srec.is_specular) {
    // PDF は内部で Dirac 的に扱うので記録不要
    return srec.attenuation
       * ray_color(srec.specular_ray, background, world, lights, depth-1, cam);
  }

  // --- 7) PDF を記録 ---
  path_pdfs.push_back(srec.pdf_ptr);

  // 次のサンプリング方向を mixture_pdf で生成
  auto light_ptr = std::make_shared<hittable_pdf>(lights, rec.p);
  mixture_pdf p(light_ptr, srec.pdf_ptr);

  ray scattered = ray(rec.p, p.generate(), r.time());
  double pdf_val = p.value(scattered.direction());

  // --- 8) 再帰的に次のバウンスを計算 ---
  return emitted
       + srec.attenuation
         * rec.mat_ptr->scattering_pdf(r, rec, scattered)
         * ray_color(scattered, background, world, lights, depth-1, cam)
         / pdf_val;
}



hittable_list final_scene() {
  hittable_list boxes1;
  auto ground = make_shared<lambertian>(make_shared<solid_color>(0.48, 0.83, 0.53));

  const int boxes_per_side = 20;
  for (int i = 0; i < boxes_per_side; i++) {
    for (int j = 0; j < boxes_per_side; j++) {
      auto w = 100.0;
      auto x0 = -1000.0 + i*w;
      auto z0 = -1000.0 + j*w;
      auto y0 = 0.0;
      auto x1 = x0 + w;
      auto y1 = random_double(1,101);
      auto z1 = z0 + w;

      boxes1.add(make_shared<box>(point3(x0,y0,z0), point3(x1,y1,z1), ground));
    }
  }

  hittable_list objects;

  objects.add(make_shared<bvh_node>(boxes1, 0, 1));

  auto light = make_shared<diffuse_light>(make_shared<solid_color>(7, 7, 7));
  objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));

  auto center1 = point3(400, 400, 200);
  auto center2 = center1 + vec3(30,0,0);
  auto moving_sphere_material =
    make_shared<lambertian>(make_shared<solid_color>(0.7, 0.3, 0.1));
  objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50, moving_sphere_material));

  objects.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
  objects.add(make_shared<sphere>(point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 10.0)));

  auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
  objects.add(boundary);
  objects.add(make_shared<constant_medium>(boundary, 0.2, make_shared<solid_color>(0.2, 0.4, 0.9)));
  boundary = make_shared<sphere>(point3(0, 0, 0), 5000, make_shared<dielectric>(1.5));
  objects.add(make_shared<constant_medium>(boundary, .0001, make_shared<solid_color>(1,1,1)));

  auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
  objects.add(make_shared<sphere>(point3(400,200,400), 100, emat));
  auto pertext = make_shared<noise_texture>(0.1);
  objects.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

  hittable_list boxes2;
  auto white = make_shared<lambertian>(make_shared<solid_color>(.73, .73, .73));
  int ns = 1000;
  for (int j = 0; j < ns; j++) {
    boxes2.add(make_shared<sphere>(point3::random(0,165), 10, white));
  }

  objects.add(make_shared<translate>(make_shared<rotate_y>(make_shared<bvh_node>(boxes2, 0.0, 1.0), 15), vec3(-100,270,395)));

  return objects;
}

hittable_list cornell_smoke() {
  hittable_list objects;

  auto red   = make_shared<lambertian>(make_shared<solid_color>(.65, .05, .05));
  auto white = make_shared<lambertian>(make_shared<solid_color>(.73, .73, .73));
  auto green = make_shared<lambertian>(make_shared<solid_color>(.12, .45, .15));
  auto light = make_shared<diffuse_light>(make_shared<solid_color>(7, 7, 7));

  objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
  objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
  objects.add(make_shared<xz_rect>(113, 443, 127, 432, 554, light));
  objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
  objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
  objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

  shared_ptr<hittable> box1 = make_shared<box>(point3(0,0,0), point3(165,330,165), white);
  box1 = make_shared<rotate_y>(box1,  15);
  box1 = make_shared<translate>(box1, vec3(265,0,295));

  shared_ptr<hittable> box2 = make_shared<box>(point3(0,0,0), point3(165,165,165), white);
  box2 = make_shared<rotate_y>(box2, -18);
  box2 = make_shared<translate>(box2, vec3(130,0,65));

  objects.add(make_shared<constant_medium>(box1, 0.01, make_shared<solid_color>(0,0,0)));
  objects.add(make_shared<constant_medium>(box2, 0.01, make_shared<solid_color>(1,1,1)));

  return objects;
}

hittable_list cornell_box(camera& cam, double aspect) {
  hittable_list world;

  auto red   = make_shared<lambertian>(make_shared<solid_color>(.65, .05, .05));
  auto white = make_shared<lambertian>(make_shared<solid_color>(.73, .73, .73));
  auto green = make_shared<lambertian>(make_shared<solid_color>(.12, .45, .15));
  auto light = make_shared<diffuse_light>(make_shared<solid_color>(15, 15, 15));

  world.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green,id_counter++));
  world.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red,id_counter++));
  world.add(make_shared<flip_face>(make_shared<xz_rect>(213, 343, 227, 332, 554, light, id_counter++)));
  world.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white,id_counter++));
  world.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white,id_counter++)); // 下
  world.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white,id_counter++)); // 奥

  shared_ptr<hittable> box1 = make_shared<box>(point3(0,0,0), point3(165,330,165), white, id_counter++);
  box1 = make_shared<rotate_y>(box1, 15);
  box1 = make_shared<translate>(box1, vec3(265,0,295));
  world.add(box1);

  auto glass = make_shared<dielectric>(1.5);
  world.add(make_shared<sphere>(point3(190, 90, 190), 90, glass, id_counter++));

  point3 lookfrom(278, 278, -800);
  point3 lookat(278, 278, 0);
  vec3 vup(0, 1, 0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.0;
  auto vfov = 40.0;
  auto t0 = 0.0;
  auto t1 = 1.0;

  cam = camera(lookfrom, lookat, vup, vfov, aspect, aperture, dist_to_focus, t0, t1);

  return world;
}

hittable_list simple_light() {
  hittable_list objects;

  auto pertext = make_shared<noise_texture>(4);
  objects.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
  objects.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

  auto difflight = make_shared<diffuse_light>(make_shared<solid_color>(4,4,4));
  objects.add(make_shared<sphere>(point3(0,7,0), 2, difflight));
  objects.add(make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));

  return objects;
}

hittable_list earth() {
  auto earth_texture = make_shared<image_texture>("earthmap.jpg");
  auto earth_surface = make_shared<lambertian>(earth_texture);
  auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface);

  return hittable_list(globe);
}

hittable_list two_perlin_spheres() {
  hittable_list objects;

  auto pertext = make_shared<noise_texture>(5);
  objects.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
  objects.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));

  return objects;
}

hittable_list two_spheres() {
  hittable_list objects;

  auto checker = make_shared<checker_texture>(
    make_shared<solid_color>(0.2, 0.3, 0.1),
    make_shared<solid_color>(0.9, 0.9, 0.9)
  );

  objects.add(make_shared<sphere>(point3(0,-10, 0), 10, make_shared<lambertian>(checker)));
  objects.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker)));

  return objects;
}

hittable_list random_scene() {
  hittable_list world;

  auto checker = make_shared<checker_texture>(
    make_shared<solid_color>(0.2, 0.3, 0.1),
    make_shared<solid_color>(0.9, 0.9, 0.9)
  );
  world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_double();
      point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

      if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
        shared_ptr<material> sphere_material;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = color::random() * color::random();
          sphere_material = make_shared<lambertian>(make_shared<solid_color>(albedo));
          auto center2 = center + vec3(0, random_double(0,.5), 0);
          world.add(make_shared<moving_sphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = color::random(0.5, 1);
          auto fuzz = random_double(0, 0.5);
          sphere_material = make_shared<metal>(albedo, fuzz);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = make_shared<dielectric>(1.5);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        }
      }
    }
  }

  auto material1 = make_shared<dielectric>(1.5);
  world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

  auto material2 = make_shared<lambertian>(make_shared<solid_color>(0.4, 0.2, 0.1));
  world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

  auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
  world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

  return hittable_list(make_shared<bvh_node>(world, 0.0, 1.0));
}

int main() {
  const auto aspect_ratio = 1.0;
  const int image_width = 300;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 30;
  const int max_depth = 10;
  const color background(0,0,0);

  // PNG出力用のバッファを作成
  unsigned char* image_buffer = new unsigned char[image_width * image_height * 3];

  camera camera;
  hittable_list world = cornell_box(camera, aspect_ratio);
  auto lights = make_shared<hittable_list>();
  lights->add(make_shared<xz_rect>(213, 343, 227, 332, 554, nullptr));
  lights->add(make_shared<sphere>(point3(190, 90, 190), 90, nullptr));

  std::cerr << "Rendering " << image_width << "x" << image_height << " image with " 
            << samples_per_pixel << " samples per pixel.\n";

  for (int j = image_height-1; j >= 0; --j) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
      color pixel_color(0, 0, 0);
      for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (i + random_double()) / (image_width-1);
        auto v = (j + random_double()) / (image_height-1);
        ray r = camera.get_ray(u, v);
        pixel_color += ray_color(r, background, world, lights, max_depth, camera);
      }
      
      // ピクセルインデックスを計算 (上下反転)
      int pixel_index = (image_height - 1 - j) * image_width + i;
      write_color_to_buffer(image_buffer, pixel_index, pixel_color, samples_per_pixel);
    }
  }

  // PNG形式で保存
  std::string filename = "output.png";
  if (stbi_write_png(filename.c_str(), image_width, image_height, 3, image_buffer, image_width * 3)) {
    std::cerr << "\nImage saved as " << filename << "\n";
  } else {
    std::cerr << "\nFailed to save image!\n";
  }

  delete[] image_buffer;
  std::cerr << "\nDone.\n";
}
