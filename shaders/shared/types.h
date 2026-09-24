// Buffer structures shared by the rasteriser, the GPU path tracer and the CPU path tracer.
// Only float4 and uint members, so MSL and C++ lay them out identically.
#pragma once
#include "prelude.h"

struct Material {
  float4 baseColorFactor;
  float4 emissive;   // rgb factor, w strength
  float4 factors;    // x metallic, y roughness, z normal scale, w occlusion strength
  float4 alpha;      // x cutoff, y mode (0 opaque, 1 mask, 2 blend), z double sided, w unused
  uint4 texture;     // x UV-set bytes, y sampler-mode bytes: base, metallic/roughness, emissive, normal
  // Transmission and clearcoat; the rasteriser ignores these.
  float4 transmission;  // x KHR_materials_transmission factor, y IOR (1.5 default), z volume thickness
                        // factor (0: thin-walled), w 1 when KHR_materials_ior was given
  float4 clearcoat;     // x clearcoat factor, y clearcoat roughness, z clearcoat normal scale, w reserved
  uint4 extensionTextures;  // transmission, clearcoat, clearcoat roughness, clearcoat normal:
                            // slot | UV set << 8 | sampler code << 16 (slot 0 white, 1 flat normal)
};

struct Light {
  float4 position;   // xyz world position, w range
  float4 color;      // rgb colour, w intensity in candela
  float4 direction;  // xyz spot direction, w inner cone cosine
  float4 cone;       // x outer cone cosine, y type (0 point, 1 spot), zw unused
};

// Vertex layout as floats: position 3, normal 3, tangent 4, uv0 2, uv1 2, color 4.
PT_CONSTANT uint kVertexFloats = 18u;

// Texture table slots bound at texture(7); indices must stay within 0..127.
PT_CONSTANT uint kHitTextureSlots = 120u;

// One per top-level instance, in the acceleration structure's instance order, read by
// every ray: the rasteriser's traced shadows and reflections and all three path tracer
// backends. The software BVH's offsets are zero where it has not been built.
struct TraceInstance {
  float4 objectToWorld0, objectToWorld1, objectToWorld2;  // rows
  float4 worldToObject0, worldToObject1, worldToObject2;  // rows
  float4 normalToWorld0, normalToWorld1, normalToWorld2;  // rows of the inverse transpose
  uint firstIndex;      // into the shared index buffer
  uint vertexOffset;    // added to every index
  uint material;        // into the material buffer
  uint slots;           // texture table slots, 8 bits each: base colour, metallic-roughness, emissive, normal
  uint blasRoot;        // software BVH: first node of this instance's bottom level
  uint triangleOffset;  // software BVH: first triangle of this instance's bottom level
  uint mask;            // kRayMask* bit this instance answers to
  uint flags;           // kInstance* bits
};

PT_CONSTANT uint kInstanceMasked = 1u;       // alpha-tested against the base colour
PT_CONSTANT uint kInstanceBlended = 2u;      // passed through with probability 1 - alpha
PT_CONSTANT uint kInstanceDoubleSided = 4u;
PT_CONSTANT uint kInstanceMirrored = 8u;     // object-to-world upper 3x3 has negative determinant

// Ray masks: the rasteriser's rays use scene and ground, never blended.
PT_CONSTANT uint kRayMaskScene = 1u;
PT_CONSTANT uint kRayMaskGround = 2u;
PT_CONSTANT uint kRayMaskBlended = 4u;
