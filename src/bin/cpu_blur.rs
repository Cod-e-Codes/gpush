use image::{DynamicImage, ImageBuffer, Rgba};
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

fn gaussian_kernel(radius: u32) -> Vec<f32> {
    let size = (radius * 2 + 1) as usize;
    let mut kernel = vec![0.0; size];
    let sigma = f32::max(radius as f32, 1.0) / 3.0; // Standard deviation, avoid div by zero

    let mut sum = 0.0;
    for (i, item) in kernel.iter_mut().enumerate().take(size) {
        let x = i as f32 - radius as f32;
        let value = (-(x * x) / (2.0 * sigma * sigma)).exp();
        *item = value;
        sum += value;
    }

    // Normalize kernel
    for item in kernel.iter_mut().take(size) {
        *item /= sum;
    }

    kernel
}

fn apply_gaussian_blur_cpu(
    input: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    radius: u32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (width, height) = input.dimensions();
    let kernel = gaussian_kernel(radius);
    let kernel_size = kernel.len() as i32;
    let half_kernel = (kernel_size / 2) as i32;

    // First pass: horizontal blur
    let mut horizontal = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for k in 0..kernel_size {
                let sample_x = (x as i32 + k - half_kernel).clamp(0, width as i32 - 1) as u32;
                let pixel = input.get_pixel(sample_x, y);
                let weight = kernel[k as usize];

                r += pixel[0] as f32 * weight;
                g += pixel[1] as f32 * weight;
                b += pixel[2] as f32 * weight;
                a += pixel[3] as f32 * weight;
            }

            horizontal.put_pixel(
                x,
                y,
                Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    a.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    // Second pass: vertical blur
    let mut output = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for k in 0..kernel_size {
                let sample_y = (y as i32 + k - half_kernel).clamp(0, height as i32 - 1) as u32;
                let pixel = horizontal.get_pixel(x, sample_y);
                let weight = kernel[k as usize];

                r += pixel[0] as f32 * weight;
                g += pixel[1] as f32 * weight;
                b += pixel[2] as f32 * weight;
                a += pixel[3] as f32 * weight;
            }

            output.put_pixel(
                x,
                y,
                Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    a.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    output
}

fn usage_and_exit() -> ! {
    eprintln!("Usage: cpu_blur <input> <output> --radius=<1-20>");
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| usage_and_exit());
    let output = args.next().unwrap_or_else(|| usage_and_exit());

    // Parse radius option
    let mut radius = 3u32;
    for arg in args {
        if let Some(radius_str) = arg.strip_prefix("--radius=") {
            radius = radius_str.parse::<u32>().unwrap_or_else(|_| {
                eprintln!("Invalid radius: {}. Use 1-20", radius_str);
                std::process::exit(1);
            });
            if !(1..=20).contains(&radius) {
                eprintln!("Radius must be between 1 and 20, got {}", radius);
                std::process::exit(1);
            }
        }
    }

    let img = image::open(&input)?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    println!(
        "Loaded {}x{} image, applying Gaussian blur with radius {}",
        width, height, radius
    );

    let start_time = Instant::now();

    // Apply Gaussian blur
    let blur_start = Instant::now();
    let blurred = apply_gaussian_blur_cpu(&rgba, radius);
    let blur_time = blur_start.elapsed();

    // Write output
    let write_start = Instant::now();
    let fout = File::create(&output)?;
    let mut writer = BufWriter::new(fout);
    DynamicImage::ImageRgba8(blurred).write_to(&mut writer, image::ImageFormat::Png)?;
    let write_time = write_start.elapsed();

    let total_time = start_time.elapsed();
    println!(
        "✓ CPU Gaussian blur completed in {:.2}ms",
        total_time.as_millis()
    );
    println!("  Blur computation: {:.2}ms", blur_time.as_millis());
    println!("  File write: {:.2}ms", write_time.as_millis());
    println!(
        "✓ Processed {} pixels with radius {} blur",
        width * height,
        radius
    );
    println!("✓ Wrote {}", output);
    Ok(())
}
