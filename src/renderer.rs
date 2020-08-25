use glam::Vec3;
use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro512StarStar;
use scoped_threadpool::Pool;
use std::cmp::Ordering;
use std::f32;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

fn mean(old: f32, new: f32, cnt: f32) -> f32 {
    let diff = (new - old) / cnt;
    old + diff
}

pub struct Renderer {
    width: usize,
    height: usize,
    scene: Scene,
    camera: Camera,
}

struct Chunk {
    i: usize,
    len: usize,
}

impl Renderer {
    pub fn new(width: usize, height: usize, scene: Scene, camera: Camera) -> Renderer {
        Renderer {
            width,
            height,
            scene,
            camera,
        }
    }
    fn render_chunk(&self, buf: &mut [u8], chunk: Chunk, rng: &mut Xoshiro512StarStar) {
        let start = chunk.i;
        let end = chunk.i + chunk.len;
        for (i, (y, x)) in (start..end)
            .map(|i| (i / self.width, i % self.width))
            .enumerate()
        {
            const N: usize = 1;
            let mut col = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..N {
                let u: f32 = (x as f32 + rnd(rng)) / self.width as f32;
                let v: f32 = ((self.height - y - 1) as f32 + rnd(rng)) / self.height as f32;
                let ray = self.camera.get_ray(u, v, rng);
                col += self.ray_trace(&ray, 0, rng);
            }
            col /= N as f32;
            buf[i * 4] = (col.x().sqrt() * 255.99) as u8;
            buf[i * 4 + 1] = (col.y().sqrt() * 255.99) as u8;
            buf[i * 4 + 2] = (col.z().sqrt() * 255.99) as u8;
        }
    }
    pub fn render(&self, buf: Arc<Mutex<Vec<u8>>>, rng: &mut Xoshiro512StarStar, frame: usize) {
        const CHUNK_SIZE: usize = 5000;
        let chunks = self.width * self.height / CHUNK_SIZE;
        let n_threads = 4;
        let (tx, rx) = mpsc::channel();

        let mut pool = Pool::new(n_threads);

        pool.scoped(|scope| {
            for i in 0..chunks {
                let tx = tx.clone();
                let mut buf = [0u8; CHUNK_SIZE * 4];
                let mut rng = Xoshiro512StarStar::seed_from_u64(rng.next_u64());
                scope.execute(move || {
                    self.render_chunk(
                        &mut buf,
                        Chunk {
                            i: i * CHUNK_SIZE,
                            len: CHUNK_SIZE,
                        },
                        &mut rng,
                    );
                    tx.send((i, buf)).expect("send failed");
                });
            }
        });
        let mut buf = buf.lock().unwrap();
        rx.iter().take(chunks).for_each(|(i, data)| {
            let start = i * CHUNK_SIZE * 4;
            let end = (i * CHUNK_SIZE + CHUNK_SIZE) * 4;
            buf[start..end]
                .iter_mut()
                .zip(data.iter())
                .for_each(|(b, d)| *b = mean(*b as f32, *d as f32, frame as f32).round() as u8)
        });
    }
    fn ray_hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<Hit> {
        self.scene
            .primitives
            .iter()
            .filter_map(|p| p.intersect(ray, tmin, tmax))
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal))
    }
    fn ray_trace(&self, ray_in: &Ray, depth: u32, rng: &mut Xoshiro512StarStar) -> Vec3 {
        const MAX_DEPTH: u32 = 8;
        const MAX_T: f32 = f32::MAX;
        const MIN_T: f32 = 0.001;
        const NS: usize = 3;
        if let Some(ray_hit) = self.ray_hit(ray_in, MIN_T, MAX_T) {
            if depth < MAX_DEPTH {
                let mut rng2 = Xoshiro512StarStar::seed_from_u64(rng.next_u64());
                return (0..NS)
                    .filter_map(|_| ray_hit.material.scatter(&ray_hit, rng))
                    .fold(Vec3::zero(), |acc, (attenuation, additive, scattered)| {
                        acc + additive
                            + attenuation * self.ray_trace(&scattered, depth + 1, &mut rng2)
                    })
                    / NS as f32;
            }
            Vec3::zero()
        } else {
            let unit_direction = ray_in.direction.normalize();
            let t = 0.5 * (unit_direction.y() + 1.0);
            (1.0 - t) * Vec3::new(0.1, 0.3, 0.2) + t * Vec3::new(0.2, 0.4, 1.0)
        }
    }
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }
    fn point_at_t(&self, t: f32) -> Vec3 {
        self.origin + (t * self.direction)
    }
}

struct Hit<'a> {
    t: f32,
    point: Vec3,
    normal: Vec3,
    direction: Vec3,
    material: &'a dyn Material,
}

trait Primitive {
    fn intersect(&self, _ray: &Ray, _tmin: f32, _tmax: f32) -> Option<Hit> {
        None
    }
}

struct Sphere {
    center: Vec3,
    radius: f32,
    material: Box<dyn Material + Send + Sync>,
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

trait Material {
    fn scatter(&self, _hit: &Hit, _rng: &mut Xoshiro512StarStar) -> Option<(Vec3, Vec3, Ray)> {
        None
    }
}

struct Lambertian {
    albedo: Vec3,
    glow: f32,
    reflect: f32,
    roughness: f32,
    transparency: f32,
    ref_index: f32,
}

impl Lambertian {
    fn new(albedo: Vec3) -> Lambertian {
        Lambertian {
            albedo,
            glow: 0.0,
            reflect: 0.0,
            roughness: 0.0,
            transparency: 0.0,
            ref_index: 1.0,
        }
    }
    fn new_specular(albedo: Vec3, reflect: f32, roughness: f32) -> Lambertian {
        Lambertian {
            albedo,
            reflect,
            glow: 0.0,
            roughness,
            transparency: 0.0,
            ref_index: 1.0,
        }
    }
    fn new_glow(albedo: Vec3, glow: f32) -> Lambertian {
        Lambertian {
            albedo,
            reflect: 0.0,
            glow,
            roughness: 0.0,
            transparency: 0.0,
            ref_index: 1.0,
        }
    }
    fn new_transparent(
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

impl Material for Lambertian {
    fn scatter(&self, hit: &Hit, rng: &mut Xoshiro512StarStar) -> Option<(Vec3, Vec3, Ray)> {
        let direction = hit.direction.normalize();
        let normal = hit.normal.normalize();

        let (albedo, target) = if self.transparency > 0.0 {
            let (reflectance, refl_dir, refr_dir) = fresnel(direction, hit.normal, self.ref_index);

            match rnd(rng) {
                n if n < reflectance => (
                    Vec3::one() * self.reflect,
                    refl_dir
                        + if self.roughness > 0.0 {
                            random_in_unit_sphere(rng) * self.roughness
                        } else {
                            Vec3::zero()
                        },
                ),
                _ => (Vec3::one() * self.transparency, refr_dir),
            }
        } else {
            match rnd(rng) {
                n if n < self.reflect => (
                    Vec3::one() * self.reflect,
                    ((direction - 2.0 * direction.dot(normal) * normal).normalize()
                        + random_in_unit_sphere(rng) * self.roughness)
                        .normalize(),
                ),
                _ => (self.albedo, (hit.normal + random_in_unit_sphere(rng)) / 2.0),
            }
        };

        Some((
            albedo,
            self.albedo * self.glow,
            Ray::new(hit.point, target.normalize()),
        ))
    }
}

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
}

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
        let aperture = 0.1;
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

    fn get_ray(&self, s: f32, t: f32, rng: &mut Xoshiro512StarStar) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk(rng);
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

fn random_in_unit_sphere(rng: &mut Xoshiro512StarStar) -> Vec3 {
    Vec3::new(
        rnd(rng) * 2.0 - 1.0,
        rnd(rng) * 2.0 - 1.0,
        rnd(rng) * 2.0 - 1.0,
    )
    .normalize()
}

fn random_in_unit_disk(rng: &mut Xoshiro512StarStar) -> Vec3 {
    let p = 2.0 * Vec3::new(rnd(rng), rnd(rng), 0.0) - Vec3::new(1.0, 1.0, 0.0);
    if p.dot(p) < 1.0 {
        p
    } else {
        1.0 / p
    }
}

fn rnd(rng: &mut Xoshiro512StarStar) -> f32 {
    rng.next_u32() as f32 / u32::MAX as f32
}
