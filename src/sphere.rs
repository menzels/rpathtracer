use crate::base::{Hit, Material, Primitive, Ray};
use glam::Vec3;

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Box<dyn Material + Send + Sync>,
}

impl Primitive for Sphere {
    fn intersect(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<Hit> {
        let l = self.center - ray.origin;
        let tca = l.dot(ray.direction);
        let d2 = l.dot(l) - tca * tca;
        if d2 > self.radius * self.radius {
            return None;
        }
        let thc = (self.radius * self.radius - d2).sqrt();
        let mut t0 = tca - thc;
        let t1 = tca + thc;

        if t0 < tmin {
            t0 = t1;
        }
        if t0 < tmin {
            return None;
        }
        if t0 > tmax {
            return None;
        }
        let point = ray.point_at_t(t0);
        Some(Hit {
            t: t0,
            normal: (point - self.center) / self.radius,
            point,
            direction: ray.direction,
            material: &*self.material,
        })
    }
}
