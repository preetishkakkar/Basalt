// Buffer structures shared by the rasteriser, the GPU path tracer and the CPU path tracer.
// Only float4 and uint members, so std140, std430 and C++ lay them out identically.
module basalt_types;

public __extern_cpp struct Material {
  __extern_cpp float4 baseColorFactor;
  __extern_cpp float4 emissive; // rgb factor, w strength
  __extern_cpp float4 factors;  // x metallic, y roughness, z normal scale, w occlusion strength
  __extern_cpp float4 alpha;    // x cutoff, y mode (0 opaque, 1 mask, 2 blend), z double sided, w unused
  __extern_cpp uint4 texture;   // x UV-set bytes, y sampler-mode bytes: base, metallic/roughness, emissive, normal
  // Transmission and clearcoat; the rasteriser ignores these.
  __extern_cpp float4 transmission; // x KHR_materials_transmission factor, y IOR (1.5 default), z volume thickness
                                    // factor (0: thin-walled), w 1 when KHR_materials_ior was given
  __extern_cpp float4 clearcoat;    // x clearcoat factor, y clearcoat roughness, z clearcoat normal scale, w reserved
  __extern_cpp uint4 extensionTextures; // transmission, clearcoat, clearcoat roughness, clearcoat normal:
                                        // slot | UV set << 8 | sampler code << 16 (slot 0 white, 1 flat normal)
};

public __extern_cpp struct Light {
  __extern_cpp float4 position;  // xyz world position, w range
  __extern_cpp float4 color;     // rgb colour, w intensity in candela
  __extern_cpp float4 direction; // xyz spot direction, w inner cone cosine
  __extern_cpp float4 cone;      // x outer cone cosine, y type (0 point, 1 spot), zw unused
};

// Vertex layout as floats: position 3, normal 3, tangent 4, uv0 2, uv1 2, color 4.
public static const uint kVertexFloats = 18u;

// Size of the texture table the tracers and traced rasteriser passes index.
public static const uint kHitTextureSlots = 120u;

// One per top-level instance, in the acceleration structure's instance order, read by
// every ray: the rasteriser's traced shadows and reflections and all three path tracer
// backends. The software BVH's offsets are zero where it has not been built.
public __extern_cpp struct TraceInstance {
  __extern_cpp float4 objectToWorld0, objectToWorld1, objectToWorld2; // rows
  __extern_cpp float4 worldToObject0, worldToObject1, worldToObject2; // rows
  __extern_cpp float4 normalToWorld0, normalToWorld1, normalToWorld2; // rows of the inverse transpose
  __extern_cpp uint firstIndex;     // into the shared index buffer
  __extern_cpp uint vertexOffset;   // added to every index
  __extern_cpp uint material;       // into the material buffer
  __extern_cpp uint slots;          // texture table slots, 8 bits each: base colour, metallic-roughness, emissive, normal
  __extern_cpp uint blasRoot;       // software BVH: first node of this instance's bottom level
  __extern_cpp uint triangleOffset; // software BVH: first triangle of this instance's bottom level
  __extern_cpp uint mask;           // kRayMask* bit this instance answers to
  __extern_cpp uint flags;          // kInstance* bits
};

public static const uint kInstanceMasked = 1u;      // alpha-tested against the base colour
public static const uint kInstanceBlended = 2u;     // passed through with probability 1 - alpha
public static const uint kInstanceDoubleSided = 4u;
public static const uint kInstanceMirrored = 8u;    // object-to-world upper 3x3 has negative determinant

// Ray masks: the rasteriser's rays use scene and ground, never blended.
public static const uint kRayMaskScene = 1u;
public static const uint kRayMaskGround = 2u;
public static const uint kRayMaskBlended = 4u;
