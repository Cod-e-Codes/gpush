// Texture-based image resize compute shader
// Uses proper texture sampling instead of storage buffers for better GPU performance

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;

struct Dims {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= dims.dst_w || y >= dims.dst_h) {
        return;
    }

    // Calculate normalized texture coordinates
    let u = (f32(x) + 0.5) / f32(dims.dst_w);
    let v = (f32(y) + 0.5) / f32(dims.dst_h);

    // Sample the source texture using the sampler
    let color = textureSampleLevel(src_tex, src_sampler, vec2<f32>(u, v), 0.0);
    
    // Store the result to the output storage texture
    textureStore(dst_tex, vec2<i32>(i32(x), i32(y)), color);
}
