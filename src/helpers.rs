use glam::Vec3;
use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::Xoshiro512StarStar;

pub fn random_in_unit_sphere(rng: &mut Xoshiro512StarStar) -> Vec3 {
    Vec3::new(
        rnd(rng) * 2.0 - 1.0,
        rnd(rng) * 2.0 - 1.0,
        rnd(rng) * 2.0 - 1.0,
    )
    .normalize()
}

pub fn random_in_unit_disk(rng: &mut Xoshiro512StarStar) -> Vec3 {
    let p = 2.0 * Vec3::new(rnd(rng), rnd(rng), 0.0) - Vec3::new(1.0, 1.0, 0.0);
    if p.dot(p) < 1.0 {
        p
    } else {
        1.0 / p
    }
}

pub fn rnd(rng: &mut Xoshiro512StarStar) -> f32 {
    rng.next_u32() as f32 / u32::MAX as f32
}
