use crate::base::Ray;
use crate::helpers;
use glam::Vec3;
use rand_xoshiro::Xoshiro512StarStar;
use std::f32;

pub struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new(
        lookfrom: Vec3,
        lookat: Vec3,
        vup: Vec3,
        vfov: f32,
        aspect: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Camera {
        let theta = vfov * f32::consts::PI / 180.0;
        let half_height = (theta * 0.5).tan();
        let half_width = aspect * half_height;
        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);
        Camera {
            origin: lookfrom,
            lower_left_corner: lookfrom
                - half_width * focus_dist * u
                - half_height * focus_dist * v
                - focus_dist * w,
            horizontal: 2.0 * half_width * focus_dist * u,
            vertical: 2.0 * half_height * focus_dist * v,
            u,
            v,
            lens_radius: aperture * 0.5,
        }
    }

    pub fn new_testcamera(w: u32, h: u32) -> Camera {
        let lookfrom = Vec3::new(40.0, 4.0, 0.0);
        let lookat = Vec3::new(0.0, 2.0, 2.5);
        let dist_to_focus = 40.0;
        let aperture = 0.2;
        Camera::new(
            lookfrom,
            lookat,
            Vec3::new(0.0, 1.0, 0.0),
            20.0,
            w as f32 / h as f32,
            aperture,
            dist_to_focus,
        )
    }

    pub fn get_ray(&self, s: f32, t: f32, rng: &mut Xoshiro512StarStar) -> Ray {
        let rd = self.lens_radius * helpers::random_in_unit_disk(rng);
        let offset = self.u * rd.x() + self.v * rd.y();
        Ray::new(
            self.origin + offset,
            (self.lower_left_corner + s * self.horizontal + t * self.vertical
                - self.origin
                - offset)
                .normalize(),
        )
    }
}
