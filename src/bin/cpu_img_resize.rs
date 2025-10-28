use image::{DynamicImage, ImageBuffer};
use std::fs::File;
use std::io::BufWriter;

fn parse_size(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let w = parts[0].parse::<u32>().ok()?;
    let h = parts[1].parse::<u32>().ok()?;
    Some((w, h))
}

fn usage_and_exit() -> ! {
    eprintln!("Usage: cpu_img_resize <input> <output> <WIDTHxHEIGHT>");
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| usage_and_exit());
    let output = args.next().unwrap_or_else(|| usage_and_exit());
    let size_str = args.next().unwrap_or_else(|| usage_and_exit());
    let (dst_w, dst_h) = parse_size(&size_str).unwrap_or_else(|| usage_and_exit());

    let img = image::open(&input)?;
    let rgba = img.to_rgba8();
    let (src_w, src_h) = rgba.dimensions();
    println!(
        "Loaded {}x{} -> resizing to {}x{}",
        src_w, src_h, dst_w, dst_h
    );

    let start_time = std::time::Instant::now();

    // CPU resize using nearest neighbor
    let mut resized = ImageBuffer::new(dst_w, dst_h);

    for y in 0..dst_h {
        for x in 0..dst_w {
            // Calculate source coordinates (nearest neighbor)
            let src_x = (x as f32 * src_w as f32 / dst_w as f32) as u32;
            let src_y = (y as f32 * src_h as f32 / dst_h as f32) as u32;

            // Clamp to source bounds
            let src_x = src_x.min(src_w - 1);
            let src_y = src_y.min(src_h - 1);

            let pixel = rgba.get_pixel(src_x, src_y);
            resized.put_pixel(x, y, *pixel);
        }
    }

    let resize_time = start_time.elapsed();
    println!("✓ CPU resize completed in {:.2}ms", resize_time.as_millis());
    println!("✓ Processed {} pixels sequentially", dst_w * dst_h);

    // Write output
    let fout = File::create(&output)?;
    let mut writer = BufWriter::new(fout);
    DynamicImage::ImageRgba8(resized).write_to(&mut writer, image::ImageFormat::Png)?;

    let total_time = start_time.elapsed();
    println!("✓ Total CPU time: {:.2}ms", total_time.as_millis());
    println!("✓ Wrote {}", output);
    Ok(())
}
