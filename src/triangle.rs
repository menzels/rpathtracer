use crate::base::{Hit, Material, Primitive, Ray};
use crate::material::Lambertian;
use glam::Vec3;
use std::cmp::Ordering;

pub struct Plane {
    pub origin: Vec3,
    pub normal: Vec3,
    pub material: Box<dyn Material + Send + Sync>,
}

impl Primitive for Plane {
    fn intersect(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<Hit> {
        let denom = self.normal.dot(ray.direction);
        if denom.abs() < 1e-6 {
            return None;
        }

        let p0l0 = self.origin - ray.origin;
        let t = p0l0.dot(self.normal) / denom;

        if t < tmin || t > tmax {
            return None;
        }
        Some(Hit {
            t,
            point: ray.point_at_t(t),
            normal: self.normal,
            direction: ray.direction,
            material: &*self.material,
        })
    }
}

pub struct Mesh {
    triangles: Vec<Triangle>,
    material: Box<dyn Material + Send + Sync>,
}

impl Mesh {
    pub fn new_tetraedron(
        origin: Vec3,
        length: f32,
        material: Box<dyn Material + Send + Sync>,
    ) -> Mesh {
        let l2 = length / 2.0;
        let height = (3.0f32).sqrt() / 2.0 * length;
        let h2 = height / 2.0;
        let a = origin + Vec3::new(h2, 0.0, l2);
        let b = origin + Vec3::new(h2, 0.0, -l2);
        let c = origin + Vec3::new(-h2, 0.0, 0.0);
        let d = origin + Vec3::new(0.0, height, 0.0);
        Mesh {
            triangles: vec![
                Triangle::new(a, b, c, Box::new(Lambertian::new(Vec3::one()))),
                Triangle::new(a, d, b, Box::new(Lambertian::new(Vec3::one()))),
                Triangle::new(b, d, c, Box::new(Lambertian::new(Vec3::one()))),
                Triangle::new(c, d, a, Box::new(Lambertian::new(Vec3::one()))),
            ],
            material,
        }
    }
}

impl Primitive for Mesh {
    fn intersect(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<Hit> {
        let res = self
            .triangles
            .iter()
            .filter_map(|p| p.intersect(ray, tmin, tmax))
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal));
        if let Some(hit) = res {
            Some(Hit {
                t: hit.t,
                point: hit.point,
                normal: hit.normal,
                direction: hit.direction,
                material: &*self.material,
            })
        } else {
            res
        }
    }
}

pub struct Triangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    normal: Vec3,
    pub material: Box<dyn Material + Send + Sync>,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3, material: Box<dyn Material + Send + Sync>) -> Triangle {
        let v0v1 = b - a;
        let v0v2 = c - a;
        let normal = v0v1.cross(v0v2).normalize();
        Triangle {
            a,
            b,
            c,
            normal,
            material,
        }
    }
}

impl Primitive for Triangle {
    fn intersect(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<Hit> {
        //         bool rayTriangleIntersect(
        //     const Vec3f &orig, const Vec3f &dir,
        //     const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
        //     float &t, float &u, float &v)
        // {
        let v0v1 = self.b - self.a;
        let v0v2 = self.c - self.a;
        let pvec = ray.direction.cross(v0v2);
        let det = v0v1.dot(pvec);

        // ray and triangle are parallel if det is close to 0
        if det.abs() < 0.0001 {
            return None;
        }

        let invDet = 1.0 / det;

        let tvec = ray.origin - self.a;
        let u = tvec.dot(pvec) * invDet;

        if u < 0.0 || u > 1.0 {
            return None;
        };

        let qvec = tvec.cross(v0v1);
        let v = ray.direction.dot(qvec) * invDet;

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = v0v2.dot(qvec) * invDet;

        if t < tmin || t > tmax {
            return None;
        }

        let point = ray.point_at_t(t);

        Some(Hit {
            t,
            normal: self.normal,
            point,
            direction: ray.direction,
            material: &*self.material,
        })
    }
}
