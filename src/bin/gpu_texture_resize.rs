use bytemuck::cast_slice;
use image::{DynamicImage, ImageBuffer, Rgba};
use pollster::block_on;
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy)]
struct Dims {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

async fn setup_gpu() -> Option<(wgpu::Device, wgpu::Queue, wgpu::ComputePipeline)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("gpush-texture-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("texture_resize_shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../src/compute_texture_resize.wgsl").into(),
        ),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture_bg_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("texture_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("texture_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Some((device, queue, pipeline))
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

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
    eprintln!(
        "Usage: gpu_texture_resize <input> <output> <WIDTHxHEIGHT> [--sampler=nearest|linear]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| usage_and_exit());
    let output = args.next().unwrap_or_else(|| usage_and_exit());
    let size_str = args.next().unwrap_or_else(|| usage_and_exit());
    let (dst_w, dst_h) = parse_size(&size_str).unwrap_or_else(|| usage_and_exit());

    // Parse sampler option
    let mut sampler_mode = wgpu::FilterMode::Nearest;
    for arg in args {
        if let Some(mode) = arg.strip_prefix("--sampler=") {
            match mode {
                "nearest" => sampler_mode = wgpu::FilterMode::Nearest,
                "linear" => sampler_mode = wgpu::FilterMode::Linear,
                _ => {
                    eprintln!("Invalid sampler mode: {}. Use 'nearest' or 'linear'", mode);
                    std::process::exit(1);
                }
            }
        }
    }

    let img = image::open(&input)?;
    let rgba = img.to_rgba8();
    let (src_w, src_h) = rgba.dimensions();
    println!(
        "Loaded {}x{} -> resizing to {}x{} (sampler: {:?})",
        src_w, src_h, dst_w, dst_h, sampler_mode
    );

    let start_time = Instant::now();

    let (device, queue, pipeline) = match block_on(setup_gpu()) {
        Some(x) => {
            println!("✓ GPU initialized successfully");
            x
        }
        None => {
            eprintln!("No GPU available or device request failed.");
            std::process::exit(1);
        }
    };

    // Create source texture
    let upload_start = Instant::now();
    let src_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("src_texture"),
        size: wgpu::Extent3d {
            width: src_w,
            height: src_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload image data to texture
    let bytes_per_row = (src_w * 4).next_multiple_of(256);
    let actual_bytes_per_row = src_w * 4; // Actual data width
    println!(
        "Texture upload: {}x{}, bytes_per_row: {}, actual: {}",
        src_w, src_h, bytes_per_row, actual_bytes_per_row
    );

    // Pad the image data to match aligned bytes_per_row
    let mut padded_data = Vec::with_capacity((bytes_per_row * src_h) as usize);
    for y in 0..src_h {
        for x in 0..src_w {
            let pixel = rgba.get_pixel(x, y);
            padded_data.push(pixel[0]); // R
            padded_data.push(pixel[1]); // G
            padded_data.push(pixel[2]); // B
            padded_data.push(pixel[3]); // A
        }

        // Add padding if needed
        if bytes_per_row > actual_bytes_per_row {
            let padding_size = (bytes_per_row - actual_bytes_per_row) as usize;
            padded_data.resize(padded_data.len() + padding_size, 0);
        }
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfoBase {
            texture: &src_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(src_h),
        },
        wgpu::Extent3d {
            width: src_w,
            height: src_h,
            depth_or_array_layers: 1,
        },
    );
    let upload_time = upload_start.elapsed();

    // Create destination storage texture
    let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dst_texture"),
        size: wgpu::Extent3d {
            width: dst_w,
            height: dst_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Create sampler
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("texture_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: sampler_mode,
        min_filter: sampler_mode,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Create uniform buffer
    let params = Dims {
        src_w,
        src_h,
        dst_w,
        dst_h,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params_buf"),
        contents: cast_slice(&[params.src_w, params.src_h, params.dst_w, params.dst_h]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &src_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &dst_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // Execute compute
    let dispatch_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture_encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("texture_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_x = 16u32;
        let wg_y = 16u32;
        let gx = ceil_div(dst_w, wg_x);
        let gy = ceil_div(dst_h, wg_y);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    queue.submit(Some(encoder.finish()));
    let dispatch_time = dispatch_start.elapsed();

    // Copy texture to buffer for readback
    let readback_start = Instant::now();
    let dst_bytes_per_row = (dst_w * 4).next_multiple_of(256);
    let staging_size = dst_bytes_per_row * dst_h;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy_encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfoBase {
            texture: &dst_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfoBase {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dst_bytes_per_row),
                rows_per_image: Some(dst_h),
            },
        },
        wgpu::Extent3d {
            width: dst_w,
            height: dst_h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map and readback
    let slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        tx.send(v).ok();
    });

    block_on(async {
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = rx.receive().await;
    });

    let data = slice.get_mapped_range();
    let pixels_u8: &[u8] = &data;
    let readback_time = readback_start.elapsed();

    // Convert to image format
    let mut out_pixels = Vec::with_capacity((dst_w * dst_h * 4) as usize);
    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_idx = (y * dst_bytes_per_row + x * 4) as usize;
            out_pixels.push(pixels_u8[src_idx]); // R
            out_pixels.push(pixels_u8[src_idx + 1]); // G
            out_pixels.push(pixels_u8[src_idx + 2]); // B
            out_pixels.push(pixels_u8[src_idx + 3]); // A
        }
    }
    drop(data);
    staging_buffer.unmap();

    let imgbuf: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(dst_w, dst_h, out_pixels)
        .ok_or("Failed to construct output image buffer")?;
    let fout = File::create(&output)?;
    let mut writer = BufWriter::new(fout);
    DynamicImage::ImageRgba8(imgbuf).write_to(&mut writer, image::ImageFormat::Png)?;

    let total_time = start_time.elapsed();
    println!(
        "✓ GPU texture resize completed in {:.2}ms",
        total_time.as_millis()
    );
    println!("  Upload: {:.2}ms", upload_time.as_millis());
    println!("  Dispatch: {:.2}ms", dispatch_time.as_millis());
    println!("  Readback: {:.2}ms", readback_time.as_millis());
    println!("✓ Processed {} pixels in parallel", dst_w * dst_h);
    println!("✓ Wrote {}", output);
    Ok(())
}
