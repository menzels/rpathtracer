use crate::base::{Hit, Ray};
use crate::camera::Camera;
use crate::helpers::rnd;
use crate::scene::Scene;
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
    pub fn render(&self, buf: Arc<Mutex<Vec<f32>>>, rng: &mut Xoshiro512StarStar, frame: usize) {
        const CHUNK_SIZE: usize = 5000;
        let chunks = self.width * self.height / CHUNK_SIZE;
        let n_threads = 4;
        let (tx, rx) = mpsc::channel();

        let mut pool = Pool::new(n_threads);

        pool.scoped(|scope| {
            for i in 0..chunks {
                let tx = tx.clone();
                let mut rng = Xoshiro512StarStar::seed_from_u64(rng.next_u64());
                scope.execute(move || {
                    let buf = self.render_chunk(
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
                .for_each(|(b, d)| *b = mean(*b, *d, frame as f32))
        });
    }
    fn render_chunk(&self, chunk: Chunk, rng: &mut Xoshiro512StarStar) -> Vec<f32> {
        let start = chunk.i;
        let end = chunk.i + chunk.len;
        let mut buf = Vec::with_capacity(chunk.len * 4);
        for (y, x) in (start..end).map(|i| (i / self.width, i % self.width)) {
            const N: usize = 1000;
            let col = (0..N)
                .map(|_| {
                    let u: f32 = (x as f32 + rnd(rng)) / self.width as f32;
                    let v: f32 = ((self.height - y - 1) as f32 + rnd(rng)) / self.height as f32;
                    let ray = self.camera.get_ray(u, v, rng);
                    self.ray_trace(&ray, 0, rng)
                })
                .fold(Vec3::zero(), |acc, v| acc + v)
                / N as f32;
            buf.push(col.x());
            buf.push(col.y());
            buf.push(col.z());
            buf.push(0.0);
        }
        buf
    }
    fn ray_hit(&self, ray: &Ray, tmin: f32, tmax: f32) -> Option<Hit> {
        self.scene
            .primitives()
            .iter()
            .filter_map(|p| p.intersect(ray, tmin, tmax))
            // .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal))
    }
    fn ray_trace(&self, ray_in: &Ray, depth: u32, rng: &mut Xoshiro512StarStar) -> Vec3 {
        const MAX_DEPTH: u32 = 50;
        const MAX_T: f32 = f32::MAX;
        const MIN_T: f32 = 0.001;
        const NS: usize = 1;
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
            let unit_direction = ray_in.direction;
            let t = 0.5 * (unit_direction.y() + 1.0);
            (1.0 - t) * Vec3::new(0.1, 0.3, 0.2) + t * Vec3::new(0.2, 0.4, 1.0)
            // Vec3::zero()
        }
    }
}
