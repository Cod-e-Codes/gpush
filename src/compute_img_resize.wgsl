// Image resize (nearest-neighbor) via compute shader
// Storage buffers for source/dest pixels and params to avoid uniform alignment pitfalls.

struct Dims {
  src_w: u32,
  src_h: u32,
  dst_w: u32,
  dst_h: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;       // packed RGBA8 per u32
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;  // packed RGBA8 per u32
@group(0) @binding(2) var<storage, read> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= dims.dst_w || y >= dims.dst_h) {
        return;
    }

    // Nearest-neighbor mapping from dst -> src
    let sxf = f32(x) * f32(dims.src_w) / f32(dims.dst_w);
    let syf = f32(y) * f32(dims.src_h) / f32(dims.dst_h);

    let sx = u32(sxf);
    let sy = u32(syf);

    let sidx = sy * dims.src_w + sx;
    let didx = y * dims.dst_w + x;

    dst[didx] = src[sidx];
}


