use crate::base::Primitive;
use crate::material::Lambertian;
use crate::sphere::Sphere;
use glam::Vec3;

pub struct Scene {
    primitives: Vec<Box<dyn Primitive + Send + Sync>>,
}

impl Scene {
    pub fn new_testscene() -> Scene {
        Scene {
            primitives: vec![
                Box::new(Sphere {
                    center: Vec3::new(2.0, -0.5, 2.0),
                    radius: 1.0,
                    material: Box::new(Lambertian::new_transparent(
                        Vec3::new(1.0, 1.0, 1.0),
                        1.0,
                        0.0,
                        1.0,
                    )),
                }),
                Box::new(Sphere {
                    center: Vec3::new(0.0, 5.0, 0.0),
                    radius: 3.0,
                    material: Box::new(Lambertian::new(Vec3::new(0.8, 0.8, 0.8))),
                }),
                Box::new(Sphere {
                    center: Vec3::new(5.0, 3.0, 6.0),
                    radius: 2.0,
                    material: Box::new(Lambertian::new_glow(Vec3::new(1.0, 0.7, 0.1), 10.0)),
                }),
                Box::new(Sphere {
                    center: Vec3::new(0.0, -1002.0, -5.0),
                    radius: 1000.0,
                    material: Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))),
                }),
                Box::new(Sphere {
                    center: Vec3::new(-3.0, 1.5, 3.0),
                    radius: 2.0,
                    material: Box::new(Lambertian::new_specular(
                        Vec3::new(1.0, 0.1, 0.8),
                        0.6,
                        0.2,
                    )),
                }),
                Box::new(Sphere {
                    center: Vec3::new(7.0, 2.0, 0.0),
                    radius: 2.5,
                    material: Box::new(Lambertian::new_transparent(
                        Vec3::new(0.2, 1.0, 0.1),
                        1.0,
                        0.0,
                        1.0,
                    )),
                }),
            ],
        }
    }
    pub fn primitives(&self) -> &Vec<Box<dyn Primitive + Send + Sync>> {
        &self.primitives
    }
}
