#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

using std::sqrt;
using std::shared_ptr;
using std::make_shared;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

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

vec3 reflect(const vec3& v, const vec3& n) {
  return v - 2*dot(v,n)*n;
}

vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
  auto cos_theta = dot(-uv, n);
  vec3 r_out_parallel =  etai_over_etat * (uv + cos_theta*n);
  vec3 r_out_perp = -sqrt(1.0 - r_out_parallel.length_squared()) * n;
  return r_out_parallel + r_out_perp;
}

// vec3 の型エイリアス
using point3 = vec3;   // 3D 点
using color = vec3;    // RGB 色

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

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

class material;

struct hit_record {
  point3 p;
  vec3 normal;
  shared_ptr<material> mat_ptr;
  double t;
  double u;
  double v;
  bool front_face;

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
};

class sphere: public hittable {
public:
  sphere() {}
  sphere(point3 cen, double r, shared_ptr<material> m)
    : center(cen), radius(r), mat_ptr(m) {};

  virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;

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

class hittable_list: public hittable {
public:
  hittable_list() {}
  hittable_list(shared_ptr<hittable> object) { add(object); }

  void clear() { objects.clear(); }
  void add(shared_ptr<hittable> object) { objects.push_back(object); }

  virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
  virtual bool bounding_box(double t0, double t1, aabb& output_box) const;

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

class moving_sphere : public hittable {
public:
  moving_sphere() {}
  moving_sphere(point3 cen0, point3 cen1, double t0, double t1, double r, shared_ptr<material> m)
    : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m)
  {};

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

class material {
public:
  virtual ~material() {};
  virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
  public:
    lambertian(shared_ptr<texture> a) : albedo(a) {}

    virtual bool scatter(
      const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
    ) const {
      vec3 scatter_direction = rec.normal + random_unit_vector();
      scattered = ray(rec.p, scatter_direction, r_in.time());
      attenuation = albedo->value(rec.u, rec.v, rec.p);
      return true;
    }

    shared_ptr<texture> albedo;
};

class metal : public material {
public:
  metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

  virtual bool scatter(
                       const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
                       ) const {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

public:
  color albedo;
  double fuzz;
};

double schlick(double cosine, double ref_idx) {
  auto r0 = (1-ref_idx) / (1+ref_idx);
  r0 = r0*r0;
  return r0 + (1-r0)*pow((1 - cosine),5);
}

class dielectric : public material {
public:
  dielectric(double ri) : ref_idx(ri) {}

  virtual bool scatter(
                       const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
                       ) const {
    attenuation = color(1.0, 1.0, 1.0);
    double etai_over_etat;
    if (rec.front_face) {
      etai_over_etat = 1.0 / ref_idx;
    } else {
      etai_over_etat = ref_idx;
    }

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
    if (etai_over_etat * sin_theta > 1.0 ) {
      vec3 reflected = reflect(unit_direction, rec.normal);
      scattered = ray(rec.p, reflected);
      return true;
    }
    double reflect_prob = schlick(cos_theta, etai_over_etat);
    if (random_double() < reflect_prob) {
      vec3 reflected = reflect(unit_direction, rec.normal);
      scattered = ray(rec.p, reflected);
      return true;
    }

    vec3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
    scattered = ray(rec.p, refracted);
    return true;
  }

  double ref_idx;
};

class camera {
public:
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

private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  double lens_radius;
  double time0, time1;  // シャッターの開閉時間
};

color ray_color(const ray& r, const hittable& world, int depth) {
  hit_record rec;

  // 反射回数が一定よりも多くなったら、その時点で追跡をやめる
  if (depth <= 0)
    return color(0,0,0);

  if (world.hit(r, 0.001, infinity, rec)) {
    ray scattered;
    color attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
      return attenuation * ray_color(scattered, world, depth-1);
    return color(0,0,0);
  }

  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5*(unit_direction.y() + 1.0);
  return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
  hittable_list world;

  auto ground_material = make_shared<lambertian>(make_shared<solid_color>(0.5, 0.5, 0.5));
  world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

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
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 384;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 100;
  const int max_depth = 50;

  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  hittable_list world = random_scene();

  point3 lookfrom(13,2,3);
  point3 lookat(0,0,0);
  vec3 vup(0,1,0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.0;

  camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

  for (int j = image_height-1; j >= 0; --j) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
      color pixel_color(0, 0, 0);
      for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (i + random_double()) / (image_width-1);
        auto v = (j + random_double()) / (image_height-1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, max_depth);
      }
      write_color(std::cout, pixel_color, samples_per_pixel);
    }
  }

  std::cerr << "\nDone.\n";
}
