mod complex;
mod mandelbrot;

use mandelbrot::{initialize_opencl, generate_mandelbrot_opencl};

extern crate piston_window;
use piston_window::*;


fn main() {
    let mut window: PistonWindow = WindowSettings::new("Look into it.", [800, 800])
        .exit_on_esc(true)
        .build()
        .unwrap();

    // Initialize OpenCL
    let (context, queue, kernel) = initialize_opencl();

    let mut center_x = -0.75;
    let mut center_y = 0.0;
    let mut zoom = 1.0;
    let mut target_center_x = center_x;
    let mut target_center_y = center_y;
    let mut target_zoom = 1.0;

    let mut cursor_position: [f64; 2] = [0.0, 0.0];

    let mut texture_context = window.create_texture_context();

    // Precompute image dimensions and initial scale factors
    let img_width = 800;
    let img_height = 800;
    let mut scale_x = 3.5 / img_width as f64 / zoom;
    let mut scale_y = 2.0 / img_height as f64 / zoom;
    
    while let Some(event) = window.next() {
        if let Some(Button::Mouse(MouseButton::Left)) = event.press_args() {
            let (x, y) = (cursor_position[0], cursor_position[1]);
            //Used to determine the new center of the zoom in the complex plane when the user clicks to zoom in or out.
            let cx = center_x + (x / img_width as f64 - 0.5) * scale_x * img_width as f64;
            let cy = center_y + (y / img_height as f64 - 0.5) * scale_y * img_height as f64;
            target_center_x = cx;
            target_center_y = cy;
            target_zoom *= 2.0;
            println!("Zooming in: Center ({}, {}), Zoom {}", target_center_x, target_center_y, target_zoom);
        }
        if let Some(Button::Mouse(MouseButton::Right)) = event.press_args() {
            let (x, y) = (cursor_position[0], cursor_position[1]);
            let cx = center_x + (x / img_width as f64 - 0.5) * scale_x * img_width as f64;
            let cy = center_y + (y / img_height as f64 - 0.5) * scale_y * img_height as f64;
            target_center_x = cx;
            target_center_y = cy;
            target_zoom /= 2.0;
            println!("Zooming out: Center ({}, {}), Zoom {}", target_center_x, target_center_y, target_zoom);
        }
        if let Some([x, y]) = event.mouse_cursor_args() {
            cursor_position = [x, y];
        }

        let lerp_factor = 0.1;
        center_x += (target_center_x - center_x) * lerp_factor;
        center_y += (target_center_y - center_y) * lerp_factor;
        zoom += (target_zoom - zoom) * lerp_factor;

        // Recalculate scale factors only if zoom has changed
        scale_x = 3.5 / img_width as f64 / zoom;
        scale_y = 2.0 / img_height as f64 / zoom;

        window.draw_2d(&event, |c, g, _device| {
            clear([1.0; 4], g);
            let img = generate_mandelbrot_opencl(img_width, img_height, 1000, center_x, center_y, scale_x, scale_y, &context, &queue, &kernel);
            let texture: G2dTexture = Texture::from_image(&mut texture_context, &img, &TextureSettings::new()).unwrap();
            image(&texture, c.transform, g);
        });
    }
}

