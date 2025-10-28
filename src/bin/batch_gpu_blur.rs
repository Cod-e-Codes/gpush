use bytemuck::cast_slice;
use image::{DynamicImage, ImageBuffer, Rgba};
use pollster::block_on;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy)]
struct BlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_pass: u32, // 0 = horizontal, 1 = vertical
    _padding: u32,
}

struct GpuBlurContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
}

async fn setup_gpu() -> Option<GpuBlurContext> {
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
            label: Some("gpush-batch-blur-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gaussian_blur_shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../src/compute_gaussian_blur.wgsl").into(),
        ),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blur_bg_layout"),
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
        label: Some("blur_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("blur_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("blur_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Some(GpuBlurContext {
        device,
        queue,
        pipeline,
        sampler,
        bind_group_layout,
    })
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn get_image_files(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension()
            && let Some(ext_str) = ext.to_str()
        {
            match ext_str.to_lowercase().as_str() {
                "png" | "jpg" | "jpeg" => files.push(path),
                _ => {}
            }
        }
    }
    files.sort();
    Ok(files)
}

fn usage_and_exit() -> ! {
    eprintln!("Usage: batch_gpu_blur <input_dir> <output_dir> --radius=<1-20> [--warmup=N]");
    std::process::exit(2);
}

fn process_single_image(
    ctx: &GpuBlurContext,
    input_path: &Path,
    output_path: &Path,
    radius: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    // Load image
    let img = image::open(input_path)?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    // Create textures
    let src_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("src_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let intermediate_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("intermediate_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let dst_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dst_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Upload image data
    let upload_start = Instant::now();
    let bytes_per_row = (width * 4).next_multiple_of(256);
    let actual_bytes_per_row = width * 4;

    let mut padded_data = Vec::with_capacity((bytes_per_row * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            padded_data.push(pixel[0]);
            padded_data.push(pixel[1]);
            padded_data.push(pixel[2]);
            padded_data.push(pixel[3]);
        }

        if bytes_per_row > actual_bytes_per_row {
            let padding_size = (bytes_per_row - actual_bytes_per_row) as usize;
            padded_data.resize(padded_data.len() + padding_size, 0);
        }
    }

    ctx.queue.write_texture(
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
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let upload_time = upload_start.elapsed();

    // Horizontal blur pass
    let pass1_start = Instant::now();
    let horizontal_params = BlurParams {
        width,
        height,
        radius,
        blur_pass: 0,
        _padding: 0,
    };
    let horizontal_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("horizontal_params_buf"),
            contents: cast_slice(&[
                horizontal_params.width,
                horizontal_params.height,
                horizontal_params.radius,
                horizontal_params.blur_pass,
                horizontal_params._padding,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let horizontal_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("horizontal_bind_group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &src_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&ctx.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &intermediate_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: horizontal_params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("horizontal_blur_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("horizontal_blur_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline);
        cpass.set_bind_group(0, &horizontal_bind_group, &[]);
        let wg_x = 16u32;
        let wg_y = 16u32;
        let gx = ceil_div(width, wg_x);
        let gy = ceil_div(height, wg_y);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    let pass1_time = pass1_start.elapsed();

    // Vertical blur pass
    let pass2_start = Instant::now();
    let vertical_params = BlurParams {
        width,
        height,
        radius,
        blur_pass: 1,
        _padding: 0,
    };
    let vertical_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertical_params_buf"),
            contents: cast_slice(&[
                vertical_params.width,
                vertical_params.height,
                vertical_params.radius,
                vertical_params.blur_pass,
                vertical_params._padding,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let vertical_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vertical_bind_group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &intermediate_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&ctx.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &dst_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: vertical_params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vertical_blur_encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vertical_blur_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline);
        cpass.set_bind_group(0, &vertical_bind_group, &[]);
        let wg_x = 16u32;
        let wg_y = 16u32;
        let gx = ceil_div(width, wg_x);
        let gy = ceil_div(height, wg_y);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    let pass2_time = pass2_start.elapsed();

    // Readback
    let readback_start = Instant::now();
    let dst_bytes_per_row = (width * 4).next_multiple_of(256);
    let staging_size = dst_bytes_per_row * height;
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: staging_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        tx.send(v).ok();
    });

    block_on(async {
        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = rx.receive().await;
    });

    let data = slice.get_mapped_range();
    let pixels_u8: &[u8] = &data;

    let mut out_pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let src_idx = (y * dst_bytes_per_row + x * 4) as usize;
            out_pixels.push(pixels_u8[src_idx]);
            out_pixels.push(pixels_u8[src_idx + 1]);
            out_pixels.push(pixels_u8[src_idx + 2]);
            out_pixels.push(pixels_u8[src_idx + 3]);
        }
    }
    drop(data);
    staging_buffer.unmap();
    let readback_time = readback_start.elapsed();

    // Write output
    let imgbuf: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, out_pixels)
        .ok_or("Failed to construct output image buffer")?;
    let fout = File::create(output_path)?;
    let mut writer = BufWriter::new(fout);
    DynamicImage::ImageRgba8(imgbuf).write_to(&mut writer, image::ImageFormat::Png)?;

    let total_time = start_time.elapsed();
    let total_ms = total_time.as_millis() as f64;

    println!(
        "  {}: {:.1}ms (upload: {:.1}ms, pass1: {:.1}ms, pass2: {:.1}ms, readback: {:.1}ms)",
        input_path.file_name().unwrap().to_string_lossy(),
        total_ms,
        upload_time.as_millis() as f64,
        pass1_time.as_millis() as f64,
        pass2_time.as_millis() as f64,
        readback_time.as_millis() as f64,
    );

    Ok(total_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input_dir = args.next().unwrap_or_else(|| usage_and_exit());
    let output_dir = args.next().unwrap_or_else(|| usage_and_exit());

    // Parse options
    let mut radius = 3u32;
    let mut warmup_count = 0u32;
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
        } else if let Some(warmup_str) = arg.strip_prefix("--warmup=") {
            warmup_count = warmup_str.parse::<u32>().unwrap_or_else(|_| {
                eprintln!("Invalid warmup count: {}", warmup_str);
                std::process::exit(1);
            });
        }
    }

    let input_path = Path::new(&input_dir);
    let output_path = Path::new(&output_dir);

    if !input_path.exists() || !input_path.is_dir() {
        eprintln!("Input directory does not exist: {}", input_dir);
        std::process::exit(1);
    }

    fs::create_dir_all(output_path)?;

    let image_files = get_image_files(input_path)?;
    if image_files.is_empty() {
        eprintln!("No image files found in {}", input_dir);
        std::process::exit(1);
    }

    println!(
        "Found {} images, applying Gaussian blur with radius {}",
        image_files.len(),
        radius
    );

    let total_start = Instant::now();

    // Setup GPU
    let ctx = match block_on(setup_gpu()) {
        Some(x) => {
            println!("✓ GPU initialized successfully");
            x
        }
        None => {
            eprintln!("No GPU available or device request failed.");
            std::process::exit(1);
        }
    };

    // Warmup if requested
    if warmup_count > 0 {
        println!("Warming up GPU with {} dummy operations...", warmup_count);
        let warmup_start = Instant::now();
        for _i in 0..warmup_count {
            // Create dummy textures for warmup
            let dummy_src_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("warmup_src_texture"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let dummy_dst_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("warmup_dst_texture"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });

            let dummy_params = BlurParams {
                width: 64,
                height: 64,
                radius: 1,
                blur_pass: 0,
                _padding: 0,
            };
            let dummy_params_buf =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("warmup_params_buf"),
                        contents: cast_slice(&[
                            dummy_params.width,
                            dummy_params.height,
                            dummy_params.radius,
                            dummy_params.blur_pass,
                            dummy_params._padding,
                        ]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            let dummy_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("warmup_bind_group"),
                layout: &ctx.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &dummy_src_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&ctx.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &dummy_dst_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dummy_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("warmup_encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("warmup_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&ctx.pipeline);
                cpass.set_bind_group(0, &dummy_bind_group, &[]);
                cpass.dispatch_workgroups(4, 4, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }
        let warmup_time = warmup_start.elapsed();
        println!("✓ Warmup completed in {:.1}ms", warmup_time.as_millis());
    }

    // Process all images
    println!("Processing images:");
    let mut total_time_ms = 0.0;
    let mut processed_count = 0;

    for input_file in &image_files {
        let filename = input_file.file_name().unwrap();
        let output_file = output_path.join(filename);

        match process_single_image(&ctx, input_file, &output_file, radius) {
            Ok(time_ms) => {
                total_time_ms += time_ms;
                processed_count += 1;
            }
            Err(e) => {
                eprintln!("  Error processing {}: {}", filename.to_string_lossy(), e);
            }
        }
    }

    let total_elapsed = total_start.elapsed();
    let avg_time = if processed_count > 0 {
        total_time_ms / processed_count as f64
    } else {
        0.0
    };

    println!();
    println!("✓ Batch processing completed!");
    println!("  Processed: {} images", processed_count);
    println!("  Total time: {:.1}ms", total_elapsed.as_millis());
    println!("  Average per image: {:.1}ms", avg_time);
    println!(
        "  Throughput: {:.1} images/second",
        processed_count as f64 * 1000.0 / total_elapsed.as_millis() as f64
    );

    Ok(())
}
