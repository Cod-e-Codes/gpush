// ✅ FIXED: Proper Gaussian Blur (Separable, Normalized)
struct BlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_pass: u32, // 0=horiz, 1=vert
    _padding: u32,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: BlurParams;

fn gaussian_weight(radius: u32, offset: i32) -> f32 {
    let abs_offset = abs(offset);
    if (abs_offset > i32(radius)) {
        return 0.0;
    }
    // ✅ FIXED: sigma = max(1, r)/3 → no div0
    let sigma = f32(max(params.radius, 1u)) / 3.0;
    let x = f32(abs_offset);
    return exp( -(x * x) / (2.0 * sigma * sigma) );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    if (params.radius == 0u) {
        // ✅ IDENTITY for r=0
        let tex_coord = vec2<f32>(
            (f32(x) + 0.5) / f32(params.width),
            (f32(y) + 0.5) / f32(params.height)
        );
        textureStore(dst_tex, vec2<i32>(i32(x), i32(y)),
                     textureSampleLevel(src_tex, src_sampler, tex_coord, 0.0));
        return;
    }

    var color = vec4<f32>(0.0);
    var total_weight = 0.0;  // ✅ ACCUMULATE!
    let radius_i = i32(params.radius);

    if (params.blur_pass == 0u) {
        // Horizontal
        for (var i = -radius_i; i <= radius_i; i++) {
            let sample_x = f32(x) + f32(i);
            let clamped_x = clamp(sample_x, 0.0, f32(params.width - 1u));
            let tex_coord = vec2<f32>(
                (clamped_x + 0.5) / f32(params.width),
                (f32(y) + 0.5) / f32(params.height)
            );
            let sample_color = textureSampleLevel(src_tex, src_sampler, tex_coord, 0.0);
            let weight = gaussian_weight(params.radius, i);
            color += sample_color * weight;
            total_weight += weight;
        }
    } else {
        // Vertical (symmetric)
        for (var i = -radius_i; i <= radius_i; i++) {
            let sample_y = f32(y) + f32(i);
            let clamped_y = clamp(sample_y, 0.0, f32(params.height - 1u));
            let tex_coord = vec2<f32>(
                (f32(x) + 0.5) / f32(params.width),
                (clamped_y + 0.5) / f32(params.height)
            );
            let sample_color = textureSampleLevel(src_tex, src_sampler, tex_coord, 0.0);
            let weight = gaussian_weight(params.radius, i);
            color += sample_color * weight;
            total_weight += weight;
        }
    }

    // ✅ NORMALIZE!
    color /= select(total_weight, 1.0, total_weight <= 0.0);
    textureStore(dst_tex, vec2<i32>(i32(x), i32(y)), color);
}
