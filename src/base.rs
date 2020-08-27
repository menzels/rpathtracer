use glam::Vec3;
use rand_xoshiro::Xoshiro512StarStar;
use std::cmp::Ordering;

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }
    pub fn point_at_t(&self, t: f32) -> Vec3 {
        self.origin + (t * self.direction)
    }
}

pub struct Hit<'a> {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub direction: Vec3,
    pub material: &'a dyn Material,
}

impl PartialOrd for Hit<'_> {
    fn partial_cmp(&self, other: &Hit) -> Option<Ordering> {
        self.t.partial_cmp(&other.t)
    }
}

impl PartialEq for Hit<'_> {
    fn eq(&self, other: &Hit) -> bool {
        self.t == other.t
    }
}

pub trait Primitive {
    fn intersect(&self, _ray: &Ray, _tmin: f32, _tmax: f32) -> Option<Hit> {
        None
    }
}

pub trait Material {
    fn scatter(&self, _hit: &Hit, _rng: &mut Xoshiro512StarStar) -> Option<(Vec3, Vec3, Ray)> {
        None
    }
}
