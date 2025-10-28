use bytemuck::cast_slice;
use image::{DynamicImage, ImageBuffer, Rgba};
use pollster::block_on;
use std::fs::File;
use std::io::BufWriter;
use std::mem::size_of;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy)]
struct Dims {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

fn pack_rgba_u32(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

fn unpack_rgba_bytes(p: u32) -> [u8; 4] {
    let r = (p & 0xff) as u8;
    let g = ((p >> 8) & 0xff) as u8;
    let b = ((p >> 16) & 0xff) as u8;
    let a = ((p >> 24) & 0xff) as u8;
    [r, g, b, a]
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
            label: Some("gpush-img-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("img_resize_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../src/compute_img_resize.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("img_bg_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("img_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("img_compute_pipeline"),
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

fn calculate_max_chunk_size(device: &wgpu::Device) -> u32 {
    let limits = device.limits();
    // Use 90% of max buffer size to be safe, divide by 4 bytes per pixel
    let max_bytes = limits.max_storage_buffer_binding_size * 9 / 10;
    let max_pixels = max_bytes / 4;
    // Take square root to get max dimension for square chunks
    (max_pixels as f64).sqrt() as u32
}

fn calculate_chunk_grid(src_w: u32, src_h: u32, max_chunk_size: u32) -> (u32, u32) {
    let chunks_x = ceil_div(src_w, max_chunk_size);
    let chunks_y = ceil_div(src_h, max_chunk_size);
    (chunks_x, chunks_y)
}

struct ChunkParams {
    src_pixels: Vec<u32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    chunk_x: u32,
    chunk_y: u32,
    chunk_size: u32,
}

fn process_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    params: ChunkParams,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Calculate chunk boundaries
    let src_start_x = params.chunk_x * params.chunk_size;
    let src_start_y = params.chunk_y * params.chunk_size;
    let src_end_x = (src_start_x + params.chunk_size).min(params.src_w);
    let src_end_y = (src_start_y + params.chunk_size).min(params.src_h);
    let chunk_src_w = src_end_x - src_start_x;
    let chunk_src_h = src_end_y - src_start_y;

    // Calculate corresponding destination chunk size
    let chunk_dst_w = ceil_div(chunk_src_w * params.dst_w, params.src_w);
    let chunk_dst_h = ceil_div(chunk_src_h * params.dst_h, params.src_h);

    // Extract source chunk
    let mut chunk_src_pixels = Vec::new();
    for y in src_start_y..src_end_y {
        for x in src_start_x..src_end_x {
            let idx = (y * params.src_w + x) as usize;
            chunk_src_pixels.push(params.src_pixels[idx]);
        }
    }

    // Create buffers for this chunk
    let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_src_buf"),
        contents: cast_slice(&chunk_src_pixels),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let dst_len = (chunk_dst_w * chunk_dst_h) as usize;
    let dst_bytes = (dst_len * size_of::<u32>()) as u64;
    let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chunk_dst_buf"),
        size: dst_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chunk_staging"),
        size: dst_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Parameters for this chunk
    let params = Dims {
        src_w: chunk_src_w,
        src_h: chunk_src_h,
        dst_w: chunk_dst_w,
        dst_h: chunk_dst_h,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_params_buf"),
        contents: cast_slice(&[params.src_w, params.src_h, params.dst_w, params.dst_h]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chunk_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // Execute compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chunk_encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chunk_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_x = 16u32;
        let wg_y = 16u32;
        let gx = ceil_div(chunk_dst_w, wg_x);
        let gy = ceil_div(chunk_dst_h, wg_y);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    encoder.copy_buffer_to_buffer(&dst_buf, 0, &staging, 0, dst_bytes);
    queue.submit(Some(encoder.finish()));

    // Read back results
    let slice = staging.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        tx.send(v).ok();
    });

    pollster::block_on(async {
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = rx.receive().await;
    });

    let data = slice.get_mapped_range();
    let pixels_u32: &[u32] = cast_slice(&data);
    let result = pixels_u32[..dst_len].to_vec();

    drop(data);
    staging.unmap();

    Ok(result)
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
    eprintln!("Usage: gpu_img_resize <input> <output> <WIDTHxHEIGHT>");
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

    let mut src_pixels_u32: Vec<u32> = Vec::with_capacity((src_w * src_h) as usize);
    for px in rgba.pixels() {
        src_pixels_u32.push(pack_rgba_u32(px[0], px[1], px[2], px[3]));
    }

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

    // Check if we need chunking
    let max_chunk_size = calculate_max_chunk_size(&device);
    let total_pixels = src_w * src_h;
    let max_pixels = max_chunk_size * max_chunk_size;

    let mut out_pixels: Vec<u32> = vec![0; (dst_w * dst_h) as usize];

    if total_pixels <= max_pixels {
        // Single chunk - use original method
        println!("✓ Processing as single chunk");

        let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("src_buf"),
            contents: cast_slice(&src_pixels_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let dst_len = (dst_w as usize) * (dst_h as usize);
        let dst_bytes = (dst_len * size_of::<u32>()) as u64;
        let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dst_buf"),
            size: dst_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: dst_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Dims {
            src_w,
            src_h,
            dst_w,
            dst_h,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_buf"),
            contents: cast_slice(&[params.src_w, params.src_h, params.dst_w, params.dst_h]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("img_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("img_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("img_pass"),
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
        encoder.copy_buffer_to_buffer(&dst_buf, 0, &staging, 0, dst_bytes);
        queue.submit(Some(encoder.finish()));
        println!("✓ GPU compute dispatched");

        let slice = staging.slice(..);
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
        let pixels_u32: &[u32] = cast_slice(&data);
        out_pixels = pixels_u32[..dst_len].to_vec();
        drop(data);
        staging.unmap();
    } else {
        // Multiple chunks needed
        let (chunks_x, chunks_y) = calculate_chunk_grid(src_w, src_h, max_chunk_size);
        println!(
            "✓ Processing {}x{} chunks (max chunk size: {})",
            chunks_x, chunks_y, max_chunk_size
        );

        for chunk_y in 0..chunks_y {
            for chunk_x in 0..chunks_x {
                println!("  Processing chunk ({}, {})", chunk_x, chunk_y);
                let chunk_result = process_chunk(
                    &device,
                    &queue,
                    &pipeline,
                    ChunkParams {
                        src_pixels: src_pixels_u32.clone(),
                        src_w,
                        src_h,
                        dst_w,
                        dst_h,
                        chunk_x,
                        chunk_y,
                        chunk_size: max_chunk_size,
                    },
                )?;

                // Calculate where this chunk goes in the final image
                let src_start_x = chunk_x * max_chunk_size;
                let src_start_y = chunk_y * max_chunk_size;
                let src_end_x = (src_start_x + max_chunk_size).min(src_w);
                let src_end_y = (src_start_y + max_chunk_size).min(src_h);
                let chunk_src_w = src_end_x - src_start_x;
                let chunk_src_h = src_end_y - src_start_y;

                let chunk_dst_w = ceil_div(chunk_src_w * dst_w, src_w);
                let chunk_dst_h = ceil_div(chunk_src_h * dst_h, src_h);

                let dst_start_x = (src_start_x * dst_w) / src_w;
                let dst_start_y = (src_start_y * dst_h) / src_h;

                // Copy chunk result to final image
                for y in 0..chunk_dst_h {
                    for x in 0..chunk_dst_w {
                        let src_idx = (y * chunk_dst_w + x) as usize;
                        let dst_x = dst_start_x + x;
                        let dst_y = dst_start_y + y;
                        if dst_x < dst_w && dst_y < dst_h {
                            let dst_idx = (dst_y * dst_w + dst_x) as usize;
                            out_pixels[dst_idx] = chunk_result[src_idx];
                        }
                    }
                }
            }
        }
    }

    // Convert u32 pixels to u8 bytes
    let mut out: Vec<u8> = Vec::with_capacity((dst_w * dst_h * 4) as usize);
    for &p in &out_pixels {
        let b = unpack_rgba_bytes(p);
        out.extend_from_slice(&b);
    }

    let imgbuf: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(dst_w, dst_h, out)
        .ok_or("Failed to construct output image buffer")?;
    let fout = File::create(&output)?;
    let mut writer = BufWriter::new(fout);
    DynamicImage::ImageRgba8(imgbuf).write_to(&mut writer, image::ImageFormat::Png)?;

    let total_time = start_time.elapsed();
    println!("✓ GPU resize completed in {:.2}ms", total_time.as_millis());
    println!("✓ Processed {} pixels in parallel", dst_w * dst_h);
    println!("✓ Wrote {}", output);
    Ok(())
}
