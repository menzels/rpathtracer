use crate::base::{Hit, Material, Ray, Scatter};
use crate::helpers::{random_in_unit_sphere, rnd};
use glam::Vec3;
use rand_xoshiro::Xoshiro512StarStar;

pub struct Lambertian {
    albedo: Vec3,
    glow: f32,
    reflect: f32,
    roughness: f32,
    transparency: f32,
    ref_index: f32,
}

impl Lambertian {
    pub fn new(albedo: Vec3) -> Lambertian {
        Lambertian {
            albedo,
            glow: 0.0,
            reflect: 0.0,
            roughness: 0.0,
            transparency: 0.0,
            ref_index: 1.0,
        }
    }
    pub fn new_specular(albedo: Vec3, reflect: f32, roughness: f32) -> Lambertian {
        Lambertian {
            albedo,
            reflect,
            glow: 0.0,
            roughness,
            transparency: 0.0,
            ref_index: 1.0,
        }
    }
    pub fn new_glow(albedo: Vec3, glow: f32) -> Lambertian {
        Lambertian {
            albedo,
            reflect: 0.0,
            glow,
            roughness: 0.0,
            transparency: 0.0,
            ref_index: 1.0,
        }
    }
    pub fn new_transparent(
        albedo: Vec3,
        reflect: f32,
        roughness: f32,
        transparency: f32,
    ) -> Lambertian {
        Lambertian {
            albedo,
            reflect,
            glow: 0.0,
            roughness,
            transparency,
            ref_index: 2.0,
        }
    }
}
fn Hue(H: f32) -> Vec3 {
    let R = ((H * 6.0 - 3.0).abs() - 1.0).max(0.0).min(1.0);
    let G = (2.0 - (H * 6.0 - 2.0).abs()).max(0.0).min(1.0);
    let B = (2.0 - (H * 6.0 - 4.0).abs()).max(0.0).min(1.0);
    Vec3::new(R, G, B)
}

// fn HSVtoRGB(HSV: Vec3) -> Vec3 {
//     ((Hue(HSV.x()) - 1.0) * HSV.y() + 1.0) * HSV.z()
// }

impl Material for Lambertian {
    fn scatter(&self, hit: &Hit, rng: &mut Xoshiro512StarStar) -> Option<Scatter> {
        let direction = hit.direction.normalize();
        let normal = hit.normal.normalize();

        // if self.transparency > 0.0 {}

        let (albedo, target) = if self.transparency > 0.0 {
            let hue = rnd(rng);
            let (reflectance, refl_dir, refr_dir) =
                fresnel(direction, hit.normal, self.ref_index + hue / 3.0);

            match rnd(rng) {
                n if n < reflectance => (
                    self.albedo * self.reflect,
                    refl_dir
                        + if self.roughness > 0.0 {
                            random_in_unit_sphere(rng) * self.roughness
                        } else {
                            Vec3::zero()
                        },
                ),
                _ => (Hue(hue) * 1.9 * self.transparency, refr_dir),
            }
        } else {
            match rnd(rng) {
                n if n < self.reflect => (
                    self.albedo * self.reflect,
                    ((direction - 2.0 * direction.dot(normal) * normal).normalize()
                        + random_in_unit_sphere(rng) * self.roughness)
                        .normalize(),
                ),
                _ => (self.albedo, (hit.normal + random_in_unit_sphere(rng)) / 2.0),
            }
        };

        Some(Scatter {
            attenuation: albedo,
            additive: self.albedo * self.glow,
            ray: Ray::new(hit.point, target.normalize()),
        })
    }
}

// returns (reflectance, reflexion direction, refraction direction), transmittans = 1 - reflectance
fn fresnel(direction: Vec3, normal: Vec3, n: f32) -> (f32, Vec3, Vec3) {
    let cosi = direction.dot(normal);
    let etai = if cosi > 0.0 { n } else { 1.0 };
    let etat = if cosi > 0.0 { 1.0 } else { n };
    let normal = if cosi > 0.0 { -normal } else { normal };
    let cosi = cosi.abs();
    let eta = etai / etat;

    // Compute sini using Snell's law
    let sint2 = eta * eta * (1.0 - cosi * cosi);
    let cost = (1.0 - sint2).sqrt();

    let reflectance = if sint2 >= 1.0 {
        // Total internal reflection
        1.0
    } else {
        let rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        let rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        (rs * rs + rp * rp) / 2.0
    };

    let refl_dir = (direction + 2.0 * cosi.abs() * normal).normalize();

    let refr_dir = (eta * direction + (eta * cosi.abs() - cost) * normal).normalize();

    (reflectance, refl_dir, refr_dir)
}
