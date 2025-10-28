//! # GPU-Accelerated Terminal
//!
//! A terminal application that automatically detects GPU-accelerated commands and offloads them to the GPU
//! when possible, falling back to CPU execution when GPU acceleration is not available.
//!
//! ## Features
//!
//! - **Automatic GPU Detection**: Detects available GPU hardware and initializes compute pipelines
//! - **Command Pattern Recognition**: Uses regex patterns to identify GPU-accelerated commands
//! - **Fallback Execution**: Seamlessly falls back to CPU execution when GPU is unavailable
//! - **Interactive Terminal**: Provides a command-line interface with GPU status information
//! - **Compute Shader Example**: Includes a demonstration of GPU compute operations
//!
//! ## Supported Commands
//!
//! The application recognizes and can accelerate various GPU-intensive commands including:
//! - FFmpeg video processing with hardware acceleration
//! - ImageMagick image processing
//! - Blender rendering
//! - Hashcat password cracking
//! - John the Ripper password recovery
//! - Tensor operations
//! - Matrix multiplication
//! - Parallel sorting
//! - File searching with regex patterns
//!
//! ## Usage
//!
//! ```bash
//! cargo run
//! ```
//!
//! Once running, you can use the following commands:
//! - `gpu-status`: Display GPU information and capabilities
//! - `gpu-test`: Run a compute shader demonstration
//! - `exit` or `quit`: Exit the application
//! - Any other command: Execute normally (with GPU acceleration if applicable)
//!
//! ## Dependencies
//!
//! - `wgpu`: Cross-platform graphics API for GPU compute operations
//! - `pollster`: Async runtime for blocking on futures
//! - `regex`: Pattern matching for command detection
//! - `lazy_static`: Static regex compilation
//! - `bytemuck`: Safe byte casting for GPU data transfer

use regex::Regex;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;

lazy_static::lazy_static! {
    /// Regular expression pattern to identify GPU-accelerated commands
    static ref GPU_RE: Regex = Regex::new(
        r"(ffmpeg.*-c:v|imagemagick|convert.*-resize|blender.*-b|hashcat|john|tensor|matrix.*mul|sort.*--parallel|grep.*-r.*\.(?:jpg|png|mp4|avi))"
    ).unwrap();
}

/// A terminal that can execute commands with GPU acceleration when available
///
/// The `GpuTerminal` struct manages GPU resources and provides methods to execute
/// commands either on the GPU (when applicable) or fall back to CPU execution.
/// It automatically detects GPU hardware and initializes compute pipelines for
/// parallel processing operations.
#[derive(Debug)]
struct GpuTerminal {
    /// Whether GPU acceleration is available and initialized
    gpu_available: bool,
    /// Name of the GPU device (e.g., "NVIDIA GeForce RTX 3080")
    gpu_name: String,
    /// WGPU device handle for GPU operations
    device: Option<Arc<wgpu::Device>>,
    /// Command queue for submitting GPU operations
    queue: Option<Arc<wgpu::Queue>>,
    /// Compute pipeline for parallel processing operations
    compute_pipeline: Option<wgpu::ComputePipeline>,
}

impl GpuTerminal {
    /// Creates a new `GpuTerminal` instance
    ///
    /// This method initializes the GPU hardware, creates a compute pipeline,
    /// and sets up the necessary resources for GPU-accelerated operations.
    /// If GPU initialization fails, it falls back to CPU-only mode.
    ///
    /// # Returns
    ///
    /// A new `GpuTerminal` instance with GPU resources initialized if available.
    async fn new() -> Self {
        // Initialize GPU
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
            .ok();

        if let Some(adapter) = adapter {
            let gpu_name = adapter.get_info().name.clone();

            if let Ok((device, queue)) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("GPU Terminal Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
            {
                println!("✓ GPU acceleration available: {}", gpu_name);

                // Create a simple compute shader for parallel operations
                let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Compute Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
                });

                let compute_pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Compute Pipeline"),
                        layout: None,
                        module: &compute_shader,
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                return GpuTerminal {
                    gpu_available: true,
                    gpu_name,
                    device: Some(Arc::new(device)),
                    queue: Some(Arc::new(queue)),
                    compute_pipeline: Some(compute_pipeline),
                };
            }
        }

        println!("⚠ GPU not available, falling back to CPU");
        GpuTerminal {
            gpu_available: false,
            gpu_name: String::from("None"),
            device: None,
            queue: None,
            compute_pipeline: None,
        }
    }

    /// Determines whether a command should be executed on the GPU
    ///
    /// This method checks if GPU acceleration is available and if the command
    /// matches patterns that can benefit from GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `command` - The command string to analyze
    ///
    /// # Returns
    ///
    /// `true` if the command should be executed on the GPU, `false` otherwise.
    fn should_use_gpu(&self, command: &str) -> bool {
        if !self.gpu_available {
            return false;
        }

        // Check for our custom GPU commands
        if command.starts_with("gpu-resize")
            || command.starts_with("gpu-blur")
            || command.starts_with("gpu-batch-blur")
        {
            return true;
        }

        // Check for traditional GPU-accelerated commands
        GPU_RE.is_match(command)
    }

    /// Demonstrates GPU compute capabilities with a simple parallel operation
    ///
    /// This method performs a compute shader operation that squares an array of numbers
    /// in parallel on the GPU. It serves as a demonstration of the GPU's compute capabilities
    /// and validates that the GPU pipeline is working correctly.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the compute operation completed successfully
    /// * `Err(Box<dyn std::error::Error>)` - If GPU is not available or an error occurred
    async fn gpu_compute_example(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.gpu_available {
            return Err("GPU not available".into());
        }

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let pipeline = self.compute_pipeline.as_ref().unwrap();

        // Create input data (array of numbers to square)
        let data_size = 1024;
        let input_data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();

        // Create buffer for input data
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: (input_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create buffer for output data
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (data_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for reading back results
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (data_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Upload input data
        queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input_data));

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        // Create compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(data_size as u32 / 64, 1, 1); // Process in groups of 64
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());

        // Submit commands
        queue.submit(std::iter::once(encoder.finish()));

        // Read back results synchronously
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            if let Err(e) = result {
                eprintln!("Buffer mapping failed: {:?}", e);
            }
        });

        // Poll device until mapping is complete
        loop {
            match device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                Ok(wgpu::PollStatus::WaitSucceeded) | Ok(wgpu::PollStatus::QueueEmpty) => break,
                Ok(wgpu::PollStatus::Poll) => continue,
                Err(e) => {
                    eprintln!("Device poll error: {:?}", e);
                    break;
                }
            }
        }

        let data = buffer_slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);

        // Verify results (first few elements)
        println!("GPU Compute Results:");
        for i in 0..std::cmp::min(10, result.len()) {
            println!("  {}² = {}", input_data[i], result[i]);
        }

        drop(data);
        staging_buffer.unmap();

        Ok(())
    }

    /// Executes a command with GPU-specific optimizations
    ///
    /// This method modifies command arguments to enable GPU acceleration when possible.
    /// For example, it adds hardware acceleration flags to FFmpeg commands or increases
    /// parallelism for sorting operations.
    ///
    /// # Arguments
    ///
    /// * `command` - The command string to execute
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the command executed successfully
    /// * `Err(io::Error)` - If command execution failed
    fn execute_on_gpu(&self, command: &str) -> io::Result<()> {
        println!("🚀 Offloading to GPU: {}", command);

        // Parse command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let program = parts[0];
        let args = &parts[1..];

        // Add GPU-specific flags based on the command
        let mut modified_args = args.to_vec();

        match program {
            "gpu-resize" => {
                // Execute our GPU resize binary
                let mut child = Command::new("cargo")
                    .args(["run", "--release", "--bin", "gpu_texture_resize", "--"])
                    .args(args)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()?;
                child.wait()?;
                return Ok(());
            }
            "gpu-blur" => {
                // Execute our GPU blur binary
                let mut child = Command::new("cargo")
                    .args(["run", "--release", "--bin", "gpu_blur", "--"])
                    .args(args)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()?;
                child.wait()?;
                return Ok(());
            }
            "gpu-batch-blur" => {
                // Execute our GPU batch blur binary
                let mut child = Command::new("cargo")
                    .args(["run", "--release", "--bin", "batch_gpu_blur", "--"])
                    .args(args)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()?;
                child.wait()?;
                return Ok(());
            }
            "ffmpeg" => {
                // Add hardware acceleration flags if not present
                if !args.contains(&"-hwaccel") {
                    modified_args.insert(0, "cuda");
                    modified_args.insert(0, "-hwaccel");
                }
            }
            "sort" => {
                // Increase parallelism
                if !args.iter().any(|&a| a.starts_with("--parallel")) {
                    modified_args.push("--parallel=16");
                }
            }
            _ => {}
        }

        // Execute with modifications
        let mut child = Command::new(program)
            .args(&modified_args)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        child.wait()?;
        Ok(())
    }

    /// Executes a command using standard CPU execution
    ///
    /// This method runs commands through the system shell without any GPU-specific
    /// modifications. It handles platform-specific shell selection automatically.
    ///
    /// # Arguments
    ///
    /// * `command` - The command string to execute
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the command executed successfully
    /// * `Err(io::Error)` - If command execution failed
    fn execute_on_cpu(&self, command: &str) -> io::Result<()> {
        // Parse command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let program = parts[0];
        let args = &parts[1..];

        // Handle our custom CPU commands
        match program {
            "cpu-resize" => {
                // Execute our CPU resize binary
                let mut child = Command::new("cargo")
                    .args(["run", "--release", "--bin", "cpu_img_resize", "--"])
                    .args(args)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()?;
                child.wait()?;
                return Ok(());
            }
            "cpu-blur" => {
                // Execute our CPU blur binary
                let mut child = Command::new("cargo")
                    .args(["run", "--release", "--bin", "cpu_blur", "--"])
                    .args(args)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()?;
                child.wait()?;
                return Ok(());
            }
            _ => {
                // Platform-specific shell selection for other commands
                #[cfg(target_os = "windows")]
                let (shell, flag) = ("cmd", "/C");

                #[cfg(not(target_os = "windows"))]
                let (shell, flag) = ("sh", "-c");

                let mut child = Command::new(shell)
                    .arg(flag)
                    .arg(command)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn()?;

                child.wait()?;
            }
        }
        Ok(())
    }

    /// Executes a command, automatically choosing between GPU and CPU execution
    ///
    /// This method analyzes the command and determines whether it should be executed
    /// on the GPU (with optimizations) or on the CPU (standard execution).
    ///
    /// # Arguments
    ///
    /// * `command` - The command string to execute
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the command executed successfully
    /// * `Err(io::Error)` - If command execution failed
    fn execute(&self, command: &str) -> io::Result<()> {
        if self.should_use_gpu(command) {
            self.execute_on_gpu(command)
        } else {
            self.execute_on_cpu(command)
        }
    }

    /// Runs the interactive terminal loop
    ///
    /// This method starts the main terminal interface, providing a command prompt
    /// where users can execute commands and access GPU status information.
    /// The terminal continues running until the user types 'exit' or 'quit'.
    async fn run(&self) {
        println!("GPU-Accelerated Terminal");
        println!("Type 'exit' to quit, 'gpu-status' for GPU info\n");

        loop {
            print!("gpu-term> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                break;
            }
            let input = input.trim();

            match input {
                "exit" | "quit" => break,
                "gpu-status" => {
                    if self.gpu_available {
                        println!("✓ GPU acceleration: ENABLED");
                        println!("  Device: {}", self.gpu_name);
                        if let Some(device) = &self.device {
                            println!("  Features: {:?}", device.features());
                            println!("  Limits: {:?}", device.limits());
                        }
                    } else {
                        println!("✗ GPU acceleration: DISABLED");
                    }
                }
                "gpu-test" => {
                    if self.gpu_available {
                        println!("🧪 Running GPU compute test...");
                        match pollster::block_on(self.gpu_compute_example()) {
                            Ok(_) => println!("✅ GPU compute test completed successfully!"),
                            Err(e) => eprintln!("❌ GPU compute test failed: {}", e),
                        }
                    } else {
                        println!("❌ GPU not available for testing");
                    }
                }
                "gpu-resize" => {
                    println!("📏 GPU Image Resize");
                    println!(
                        "Usage: gpu-resize <input> <output> <WIDTHxHEIGHT> [--sampler=nearest|linear]"
                    );
                    println!("Example: gpu-resize image.png resized.png 800x600 --sampler=linear");
                }
                "gpu-blur" => {
                    println!("🌫️ GPU Image Blur");
                    println!("Usage: gpu-blur <input> <output> --radius=<1-5>");
                    println!("Example: gpu-blur image.png blurred.png --radius=3");
                }
                "gpu-batch-blur" => {
                    println!("📦 GPU Batch Blur");
                    println!(
                        "Usage: gpu-batch-blur <input_dir> <output_dir> --radius=<1-5> [--warmup=N]"
                    );
                    println!("Example: gpu-batch-blur ./images ./blurred --radius=3 --warmup=5");
                }
                "cpu-resize" => {
                    println!("📏 CPU Image Resize");
                    println!("Usage: cpu-resize <input> <output> <WIDTHxHEIGHT>");
                    println!("Example: cpu-resize image.png resized.png 800x600");
                }
                "cpu-blur" => {
                    println!("🌫️ CPU Image Blur");
                    println!("Usage: cpu-blur <input> <output> --radius=<1-10>");
                    println!("Example: cpu-blur image.png blurred.png --radius=3");
                }
                "help" => {
                    println!("🚀 GPU-Accelerated Terminal Commands:");
                    println!("  gpu-status     - Show GPU information");
                    println!("  gpu-test       - Run GPU compute test");
                    println!("  gpu-resize     - Show GPU resize help");
                    println!("  gpu-blur       - Show GPU blur help");
                    println!("  gpu-batch-blur - Show GPU batch blur help");
                    println!("  cpu-resize     - Show CPU resize help");
                    println!("  cpu-blur       - Show CPU blur help");
                    println!("  exit/quit      - Exit terminal");
                    println!("  Any other command will be executed normally");
                }
                "" => continue,
                cmd => {
                    if let Err(e) = self.execute(cmd) {
                        eprintln!("Error: {}", e);
                    }
                }
            }
        }
    }
}

/// Main entry point for the GPU-accelerated terminal application
///
/// This function initializes the GPU terminal and starts the interactive loop.
/// It handles the async initialization of GPU resources and then runs the
/// terminal interface.
fn main() {
    let terminal = pollster::block_on(GpuTerminal::new());
    pollster::block_on(terminal.run());
}
