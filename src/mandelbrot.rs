use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::context::Context;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING};
use opencl3::platform::get_platforms;
use opencl3::memory::{Buffer, CL_MEM_WRITE_ONLY};
use std::ptr;
use image::{ImageBuffer, RgbaImage, Rgba};


const MANDELBROT_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
    __global uchar4* img,
    const double center_x,
    const double center_y,
    const double scale_x,
    const double scale_y,
    const unsigned int max_iter,
    const unsigned int imgx,
    const unsigned int imgy)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    double cx = center_x + (x - imgx / 2.0) * scale_x;
    double cy = center_y + (y - imgy / 2.0) * scale_y;

    double zx = 0.0;
    double zy = 0.0;
    unsigned int iter = 0;

    while (zx * zx + zy * zy < 4.0 && iter < max_iter) {
        double tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        iter++;
    }

    uchar4 color = (uchar4)(0, 0, 0, 255);
    if (iter < max_iter) {
        double t = (double)iter / (double)max_iter;
        unsigned char r = (unsigned char)(9.0 * (1.0 - t) * t * t * t * 255.0 + 10);
        unsigned char g = (unsigned char)(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0 + 10);
        unsigned char b = (unsigned char)(9.0 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0 + 10);
        color = (uchar4)(r, g, b, 255);
    }

    img[y * imgx + x] = color;
}
"#;

pub fn initialize_opencl() -> (Context, CommandQueue, Kernel) {
    // Get all available OpenCL platforms
    let platform = get_platforms().unwrap()[0];

    // Get a GPU device from the platform
    let device = Device::new(platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()[0]);

    // Create a context for the device
    let context = Context::from_device(&device).unwrap();

    // Create a command queue for the device
    let queue = CommandQueue::create(&context, device.id(), CL_QUEUE_PROFILING_ENABLE).unwrap();

    // Create and build the program from the source kernel code
    let program = Program::create_and_build_from_source(&context, MANDELBROT_KERNEL, "").unwrap();

    // Create the kernel from the program
    let kernel = Kernel::create(&program, "mandelbrot").unwrap();

    (context, queue, kernel)
}

pub fn generate_mandelbrot_opencl(
    imgx: u32, imgy: u32, max_iter: usize,
    center_x: f64, center_y: f64, scale_x: f64, scale_y: f64,
    context: &Context, queue: &CommandQueue, kernel: &Kernel
) -> RgbaImage {
    let mut imgbuf = ImageBuffer::new(imgx, imgy);

    let buffer_size = (imgx * imgy * 4) as usize;
    let mut buffer = vec![0u8; buffer_size];

    let img_buf = Buffer::<u8>::create(context, CL_MEM_WRITE_ONLY, buffer_size, ptr::null_mut()).unwrap();

    kernel.set_arg(0, &img_buf).unwrap();
    kernel.set_arg(1, &center_x).unwrap(); 
    kernel.set_arg(2, &center_y).unwrap(); 
    kernel.set_arg(3, &scale_x).unwrap();  
    kernel.set_arg(4, &scale_y).unwrap();  
    kernel.set_arg(5, &(max_iter as u32)).unwrap();
    kernel.set_arg(6, &imgx).unwrap();
    kernel.set_arg(7, &imgy).unwrap();

    let global_work_size = [imgx as usize, imgy as usize];
    unsafe {
        queue.enqueue_nd_range_kernel(
            kernel.get(),                   // Pass kernel directly
            2,                        // Number of dimensions
            ptr::null(),              // Global work offset
            global_work_size.as_ptr(),// Global work size
            ptr::null(),              // Local work size
            &mut []                   // Event wait list
        ).unwrap();
    }

    queue.enqueue_read_buffer(&img_buf, CL_BLOCKING, 0, &mut buffer, &[]).unwrap();

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let idx = ((y * imgx + x) * 4) as usize;
        *pixel = Rgba([buffer[idx], buffer[idx + 1], buffer[idx + 2], buffer[idx + 3]]);
    }

    imgbuf
}
