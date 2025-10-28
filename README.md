# gpush

GPU-accelerated image processing toolkit written in Rust.

## Features

- GPU-accelerated image resize and Gaussian blur operations
- CPU fallback implementations for all operations
- Interactive terminal interface with automatic GPU detection
- Batch processing support for multiple images
- Built with wgpu compute shaders using WebGPU Shading Language (WGSL)
- Cross-platform GPU compute support
- Separable Gaussian blur implementation (two-pass)
- Chunking support for processing large images
- Both texture-based and buffer-based GPU implementations

## Installation

### Prerequisites

- Rust toolchain (tested with rustc 1.90.0)

### Build

```bash
git clone https://github.com/Cod-e-Codes/gpush.git
cd gpush
cargo build --release
```

## Usage

### Interactive Terminal

Run the main terminal interface:

```bash
cargo run
```

Available terminal commands:
- `gpu-status` - Display GPU information and capabilities
- `gpu-test` - Run GPU compute test
- `help` - Show available commands
- `exit` or `quit` - Exit the terminal

### Individual Binaries

#### Image Blur

**GPU Blur:**
```bash
cargo run --release --bin gpu_blur -- input.png output.png --radius=3
```

**CPU Blur:**
```bash
cargo run --release --bin cpu_blur -- input.png output.png --radius=3
```

#### Image Resize

**GPU Image Resize:**
```bash
cargo run --release --bin gpu_img_resize -- input.png output.png 800x600
```

**GPU Texture Resize:**
```bash
cargo run --release --bin gpu_texture_resize -- input.png output.png 800x600 --sampler=linear
```

**CPU Image Resize:**
```bash
cargo run --release --bin cpu_img_resize -- input.png output.png 800x600
```

#### Batch Processing

**Batch GPU Blur:**
```bash
cargo run --release --bin batch_gpu_blur -- ./input_dir ./output_dir --radius=3 --warmup=5
```

## Available Commands

| Binary | Description | Usage |
|--------|-------------|-------|
| `gpu_blur` | GPU-accelerated Gaussian blur | `gpu_blur <input> <output> --radius=<1-20>` |
| `cpu_blur` | CPU Gaussian blur implementation | `cpu_blur <input> <output> --radius=<1-20>` |
| `gpu_img_resize` | GPU image resize (buffer-based) | `gpu_img_resize <input> <output> <WIDTHxHEIGHT>` |
| `gpu_texture_resize` | GPU image resize (texture-based) | `gpu_texture_resize <input> <output> <WIDTHxHEIGHT> [--sampler=nearest\|linear]` |
| `cpu_img_resize` | CPU image resize implementation | `cpu_img_resize <input> <output> <WIDTHxHEIGHT>` |
| `batch_gpu_blur` | Batch GPU blur processing | `batch_gpu_blur <input_dir> <output_dir> --radius=<1-5> [--warmup=N]` |

## Performance

Benchmark results on AMD Radeon RX 9070 XT with 1717x2101 images:

**Image Resize (400x300):**
- GPU: 418ms (12ms upload, 1ms compute, 3ms readback)
- CPU: 3ms total

**Image Resize (800x600):**
- GPU: 413ms (14ms upload, 1ms compute, 2ms readback)  
- CPU: 5ms total

**Gaussian Blur (radius 3):**
- GPU: 347ms (11ms upload, 2ms compute, 10ms readback)
- CPU: 69ms (56ms computation, 13ms write)

**Gaussian Blur (radius 5):**
- GPU: 352ms (10ms upload, 2ms compute, 9ms readback)
- CPU: 96ms (81ms computation, 14ms write)

**Gaussian Blur (radius 10):**
- GPU: 476ms (15ms upload, 3ms compute, 11ms readback)
- CPU: 146ms (131ms computation, 15ms write)

**Batch Processing:**
- 10 images with radius 5 blur: 902ms total (45.9ms average per image)
- Throughput: 11.1 images/second

GPU implementations excel at compute-intensive operations like blur, while CPU implementations are faster for simple operations like resize due to lower overhead. The toolkit automatically falls back to CPU implementations when GPU acceleration is unavailable.

## License

MIT License
