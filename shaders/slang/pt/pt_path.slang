// The path tracer's shared types and the interfaces its backends implement. Compiled to
// SPIR-V for the GPU tracers and to C++ for the CPU tracer, from this one source.
module pt_path;

__exported import basalt_types;
__exported import basalt_random;
__exported import basalt_shading;

// Everything a path needs besides geometry. Four-lane fields keep the C++ and GPU layouts
// identical. Values whose individual bits matter stay integer all the way to the shader:
// float cannot distinguish adjacent uints above 2^24.
public __extern_cpp struct PathUniforms {
  __extern_cpp float4 cameraPosition; // xyz the eye
  __extern_cpp float4 cameraForward;  // xyz unit forward
  __extern_cpp float4 cameraRight;    // xyz right, scaled to the half-width of the image at unit distance
  __extern_cpp float4 cameraUp;       // xyz up, scaled to the half-height of the image at unit distance
  __extern_cpp float4 image;          // x width, y height, z 0 raw / 1 guides / 2 node cost / 3 triangle cost, w pass samples
  __extern_cpp float4 path;           // x maximum bounces, y Russian roulette from this bounce, z firefly clamp (0 off),
                                      // w strategy: 0 MIS, 1 BSDF sampling only, 2 light sampling only
  __extern_cpp float4 environment;    // x intensity, y equirect width, z equirect height, w draw it behind the scene
  __extern_cpp float4 distribution;   // x columns, y rows, z integral, w present
  __extern_cpp float4 sunDirection;   // xyz unit vector towards the sun, w cosine of its angular radius (0: no sun)
  __extern_cpp float4 sunRadiance;    // rgb radiance of the disc, w its solid angle
  __extern_cpp uint4 counts;          // x punctual lights, y ray mask, z seed, w first sample index
  __extern_cpp uint4 emissive;        // x emissive triangle count, yzw reserved
  __extern_cpp float4 lens;           // x aperture radius (0 pinhole), y focus distance along the view axis,
                                      // z texture filter (0 level-zero bilinear, 1 ray cones), w pixel spread angle
  __extern_cpp uint4 estimator;       // x direct light (0 NEE, 1 ReSTIR DI), y ReSTIR candidates, z reuse bits, w frame
};

// Shirley and Chiu's concentric map of the unit square onto the unit disk.
public float2 ptConcentricDisk(float2 u) {
  const float2 offset = u * 2.0f - float2(1.0f);
  if (offset.x == 0.0f && offset.y == 0.0f) return float2(0.0f);
  float radius = offset.y;
  float angle = kPi * 0.5f - (kPi / 4.0f) * (offset.x / offset.y);
  if (abs(offset.x) > abs(offset.y)) {
    radius = offset.x;
    angle = (kPi / 4.0f) * (offset.y / offset.x);
  }
  return float2(cos(angle), sin(angle)) * radius;
}

// Thin-lens depth of field. The pinhole ray through the pixel sample fixes the focus point at
// the focus distance along the view axis; the ray then starts at a lens point from the
// concentric map of `xi` and passes through that point. Aperture 0 leaves the ray untouched.
public void ptApplyThinLens(float4 lens, float3 forward, float3 right, float3 up, float2 xi, inout float3 origin,
                            inout float3 direction) {
  if (lens.x > 0.0f) {
    const float3 focus = origin + direction * (lens.y / dot(direction, forward));
    const float2 disk = ptConcentricDisk(xi) * lens.x;
    origin = origin + normalize(right) * disk.x + normalize(up) * disk.y;
    direction = normalize(focus - origin);
  }
}

// The camera ray of a pixel's sample, lens included: random group 0 of bounce 0 jitters the
// pixel (xy) and picks the lens point (zw).
public float3 ptCameraRay(PathUniforms uniforms, uint pixelX, uint pixelY, uint pathSeedValue, out float3 origin) {
  const float4 jitter = pathRandom4(pathSeedValue, 0u, 0u);
  const float ndcX = (float(pixelX) + jitter.x) / uniforms.image.x * 2.0f - 1.0f;
  const float ndcY = 1.0f - (float(pixelY) + jitter.y) / uniforms.image.y * 2.0f;
  float3 direction =
      normalize(uniforms.cameraForward.xyz + uniforms.cameraRight.xyz * ndcX + uniforms.cameraUp.xyz * ndcY);
  origin = uniforms.cameraPosition.xyz;
  ptApplyThinLens(uniforms.lens, uniforms.cameraForward.xyz, uniforms.cameraRight.xyz, uniforms.cameraUp.xyz, jitter.zw,
                  origin, direction);
  return direction;
}

// Compact world-space area-light metadata. The emissive texture is read at the sampled
// barycentrics; the CDF only chooses a triangle.
public __extern_cpp struct PtEmissiveTriangle {
  __extern_cpp float4 v0Area;           // xyz first vertex, w area
  __extern_cpp float4 edge1Probability; // xyz edge 1, w discrete selection probability
  __extern_cpp float4 edge2Cdf;         // xyz edge 2, w inclusive normalized CDF
  __extern_cpp float4 normal;           // xyz geometric world normal
  __extern_cpp float4 uv01;             // sampled emissive UV at vertices 0 and 1
  __extern_cpp float4 uv2;              // xy sampled emissive UV at vertex 2
  __extern_cpp uint4 identity;          // x instance, y primitive, zw reserved
};

// A ray's closest hit, in the terms the hardware reports it: the instance, the triangle
// within the instance's primitive, and the barycentrics of the second and third vertices.
public __extern_cpp struct PtHit {
  __extern_cpp uint ambiguous; // a stochastic blended candidate was encountered, even if rejected
  __extern_cpp float t;
  __extern_cpp float2 barycentric;
  __extern_cpp uint instance;
  __extern_cpp uint primitive;
  __extern_cpp uint found;
  __extern_cpp uint nodeVisits;    // software-BVH diagnostic; zero for other intersectors
  __extern_cpp uint triangleTests; // software-BVH diagnostic; zero for other intersectors

  // No hit, at distance t.
  __init(float distance) {
    ambiguous = 0u;
    t = distance;
    barycentric = float2(0.0f);
    instance = 0u;
    primitive = 0u;
    found = 0u;
    nodeVisits = 0u;
    triangleTests = 0u;
  }
};

public static const float kPtInfinity = 3.0e38f;

public float ptMaxComponent(float3 v) { return max(max(v.x, v.y), v.z); }
public float ptLuminance(float3 v) { return dot(v, float3(0.2126f, 0.7152f, 0.0722f)); }

// Power heuristic in ratio form, so a near-delta pdf cannot overflow its square.
public float ptPowerHeuristic(float chosen, float other) {
  if (chosen <= 0.0f) return 0.0f;
  const float ratio = other / chosen;
  return 1.0f / (1.0f + ratio * ratio);
}

// Scales a contribution down to the clamp; a limit of zero leaves it alone.
public float3 ptClampContribution(float3 contribution, float limit) {
  float3 result = contribution;
  const float largest = ptMaxComponent(contribution);
  if (limit > 0.0f && largest > limit) result = contribution * (limit / largest);
  return result;
}

// Waechter and Binder's offset along the geometric normal, in units of the position's own
// floating-point spacing, so it holds at any scene scale; near the origin, where that
// spacing vanishes, a small fixed step instead.
public float3 ptOffsetRay(float3 position, float3 normal) {
  const int3 steps = int3(normal * 256.0f);
  float3 result = asfloat(asint(position) + steps * int3(sign(position)));
  const float3 nudged = position + normal * (1.0f / 65536.0f);
  if (abs(position.x) < 1.0f / 32.0f) result.x = nudged.x;
  if (abs(position.y) < 1.0f / 32.0f) result.y = nudged.y;
  if (abs(position.z) < 1.0f / 32.0f) result.z = nudged.z;
  return result;
}

// A 32-bit seed for a trace's stochastic decisions (blended surfaces), distinct per bounce.
public uint ptTraceSeed(uint pathSeedValue, uint bounce, uint purpose) {
  return pcgHash(pathSeedValue ^ pcgHash(bounce * 16u + purpose + 0x51ED270Bu));
}

// The hit texture table and the environment, as each backend reads them: the GPU through
// its bound textures and samplers, the CPU through the host's own copies.
public interface IPtMaps {
  // A table slot's texel at uv, with the glTF sampler code's wrap and filter; lodBase from
  // ptConeLodBase, kPtLevelZero sampling level zero exactly.
  float4 sampleTexture(uint slot, float2 uv, uint samplerCode, float lodBase);
  // The footprint's width in texels at a slot's texture (ptTextureFootprint).
  float textureFootprint(uint slot, float lodBase);
  // The environment's equirect at uv: bilinear, u repeating, v clamped.
  float3 sampleEnvironment(float2 uv);
};

// The scene's geometry: instances in the acceleration structure's order, their materials and
// the shared index and vertex buffers (kVertexFloats per vertex).
public struct PtGeometry {
  StructuredBuffer<TraceInstance> traceInstances;
  StructuredBuffer<Material> materials;
  StructuredBuffer<uint> indices;
  StructuredBuffer<float> vertices;

  // The first float of each of a hit triangle's vertices.
  uint3 triangleVertices(uint instance, uint primitive) {
    const TraceInstance info = traceInstances[instance];
    const uint base = info.firstIndex + primitive * 3u;
    return (uint3(indices[base], indices[base + 1u], indices[base + 2u]) + info.vertexOffset) * kVertexFloats;
  }

  // A vertex attribute of three, two or one floats at `offset` within the vertex, interpolated.
  float3 interpolate3(uint3 v, uint offset, float3 weights) {
    return float3(dot(weights, float3(vertices[v.x + offset], vertices[v.y + offset], vertices[v.z + offset])),
                  dot(weights, float3(vertices[v.x + offset + 1u], vertices[v.y + offset + 1u], vertices[v.z + offset + 1u])),
                  dot(weights, float3(vertices[v.x + offset + 2u], vertices[v.y + offset + 2u], vertices[v.z + offset + 2u])));
  }
  float2 interpolate2(uint3 v, uint offset, float3 weights) {
    return float2(dot(weights, float3(vertices[v.x + offset], vertices[v.y + offset], vertices[v.z + offset])),
                  dot(weights, float3(vertices[v.x + offset + 1u], vertices[v.y + offset + 1u], vertices[v.z + offset + 1u])));
  }
  float interpolate1(uint3 v, uint offset, float3 weights) {
    return dot(weights, float3(vertices[v.x + offset], vertices[v.y + offset], vertices[v.z + offset]));
  }
  float3 position(uint vertex) { return float3(vertices[vertex], vertices[vertex + 1u], vertices[vertex + 2u]); }
  float2 uv(uint vertex, uint offset) { return float2(vertices[vertex + offset], vertices[vertex + offset + 1u]); }
};

// Float4/uint4 lanes only: the shared CPU/GPU storage-buffer ABI, 128 bytes per sample.
public __extern_cpp struct PtReconstructionSample {
  __extern_cpp float4 positionDepth;   // world position, forward linear depth
  __extern_cpp float4 geometricNormal; // world normal, transmission/clearcoat layer flags (ptGuideLayers)
  __extern_cpp float4 normalRoughness; // world shading normal, perceptual roughness
  __extern_cpp float4 albedo;          // linear base color, reserved
  __extern_cpp float4 viewDistance;    // primary ray direction, secondary hit distance (-1 miss)
  __extern_cpp float4 diffuse;         // fresh diffuse/residual radiance, reserved
  __extern_cpp float4 specular;        // fresh specular radiance, reserved
  __extern_cpp uint4 identity;         // instance, material, coverage (0 miss, 1 opaque, 2 blend, 3 mask), primitive

  static PtReconstructionSample empty() {
    PtReconstructionSample s;
    s.positionDepth = float4(0.0f);
    s.geometricNormal = float4(0.0f);
    s.normalRoughness = float4(0.0f);
    s.albedo = float4(0.0f);
    s.viewDistance = float4(0.0f, 0.0f, 0.0f, -1.0f);
    s.diffuse = float4(0.0f);
    s.specular = float4(0.0f);
    s.identity = uint4(0u);
    return s;
  }
};

// Layers whose specular correspondence primary-surface reprojection cannot follow: 1 a
// transmissive surface (what shows through moves with the refraction, not with the
// surface), 2 a mirror-like clearcoat (coat roughness below the 0.15 mirror threshold).
public float ptGuideLayers(float4 transmission, float4 clearcoat) {
  float layers = 0.0f;
  if (transmission.x > 0.0f) layers = layers + 1.0f;
  if (clearcoat.x > 0.0f && clearcoat.y < 0.15f) layers = layers + 2.0f;
  return layers;
}

public __extern_cpp struct PtTemporalHistory {
  __extern_cpp float4 diffuse;  // RGB estimate, effective history length
  __extern_cpp float4 specular; // RGB estimate, effective history length
  __extern_cpp float4 moments;  // diffuse first/second moment, specular first/second moment
};

public __extern_cpp struct PtTemporalUniforms {
  __extern_cpp PathUniforms previousCamera;
  __extern_cpp uint4 image;        // width, height, reset, a-trous stride
  __extern_cpp float4 control;     // stable raw SPP, reserved
  __extern_cpp uint4 outputExtent; // display width/height, reserved (can exceed guide extent for CPU preview)
};
