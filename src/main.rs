#![deny(clippy::all)]
#![forbid(unsafe_code)]

pub mod base;
pub mod camera;
pub mod helpers;
pub mod material;
pub mod renderer;
pub mod scene;
pub mod sphere;
pub mod triangle;

use camera::Camera;
use renderer::Renderer;
use scene::Scene;

use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro512StarStar;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalSize};
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit_input_helper::WinitInputHelper;

const SCREEN_WIDTH: u32 = 1000;
const SCREEN_HEIGHT: u32 = 1000;

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let (window, surface, p_width, p_height, mut _hidpi_factor) =
        create_window("Monte Carlo Path Tracer", &event_loop);

    let surface_texture = SurfaceTexture::new(p_width, p_height, surface);
    let mut pixels = Pixels::new(SCREEN_WIDTH, SCREEN_HEIGHT, surface_texture)?;
    let data: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![
        0f32;
        (SCREEN_HEIGHT * SCREEN_WIDTH * 4)
            as usize
    ]));

    let (tx, rx) = mpsc::channel();
    let dc = Arc::clone(&data);

    thread::spawn(move || {
        let scene = Scene::new_testscene();
        let camera = Camera::new_testcamera(SCREEN_WIDTH, SCREEN_HEIGHT);
        let r = Renderer::new(SCREEN_WIDTH as usize, SCREEN_HEIGHT as usize, scene, camera);

        let mut rng = Xoshiro512StarStar::seed_from_u64(0);
        let mut frame = 1;

        loop {
            let dp = Arc::clone(&data);
            let now = Instant::now();
            r.render(dp, &mut rng, frame);
            if tx.send(frame).is_err() {
                return;
            }
            frame += 1;
            println!("frame time: {}", now.elapsed().as_millis());
        }
    });

    event_loop.run(move |event, _, control_flow| {
        if rx.try_recv().is_ok() {
            {
                let m = dc.try_lock();
                if let Ok(data) = m {
                    pixels
                        .get_frame()
                        .iter_mut()
                        .zip(data.iter())
                        .for_each(|(f, d)| *f = (d.sqrt() * 255.99) as u8);
                }
            }
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }
        // The one and only event that winit_input_helper doesn't have for us...
        if let Event::RedrawRequested(_) = event {
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // For everything else, for let winit_input_helper collect events to build its state.
        // It returns `true` when it is time to update our game state and request a redraw.
        if input.update(event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }
            // Adjust high DPI factor
            if let Some(factor) = input.scale_factor_changed() {
                _hidpi_factor = factor;
            }
            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize(size.width, size.height);
            }
            window.request_redraw();
        }
    })
}

// COPYPASTE: ideally this could be shared.

/// Create a window for the game.
///
/// Automatically scales the window to cover about 2/3 of the monitor height.
///
/// # Returns
///
/// Tuple of `(window, surface, width, height, hidpi_factor)`
/// `width` and `height` are in `PhysicalSize` units.
fn create_window(
    title: &str,
    event_loop: &EventLoop<()>,
) -> (winit::window::Window, pixels::wgpu::Surface, u32, u32, f64) {
    // Create a hidden window so we can estimate a good default window size
    let window = winit::window::WindowBuilder::new()
        .with_visible(false)
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    let hidpi_factor = window.scale_factor();

    // Get dimensions
    let width = SCREEN_WIDTH as f64;
    let height = SCREEN_HEIGHT as f64;
    let (monitor_width, monitor_height) = {
        let size = window.current_monitor().size();
        (
            size.width as f64 / hidpi_factor,
            size.height as f64 / hidpi_factor,
        )
    };
    let scale = (monitor_height / height * 2.0 / 3.0).round();

    // Resize, center, and display the window
    let min_size: winit::dpi::LogicalSize<f64> =
        PhysicalSize::new(width, height).to_logical(hidpi_factor);
    let default_size = LogicalSize::new(width * scale, height * scale);
    let center = LogicalPosition::new(
        (monitor_width - width * scale) / 2.0,
        (monitor_height - height * scale) / 2.0,
    );
    window.set_inner_size(default_size);
    window.set_min_inner_size(Some(min_size));
    window.set_outer_position(center);
    window.set_visible(true);

    let surface = pixels::wgpu::Surface::create(&window);
    let size = default_size.to_physical::<f64>(hidpi_factor);

    (
        window,
        surface,
        size.width.round() as u32,
        size.height.round() as u32,
        hidpi_factor,
    )
}
