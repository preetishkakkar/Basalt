// The host's view of the shared Slang code: the structs, constants and functions of
// shaders/slang (basalt/*, pt/*) as the pt_cpu module's generated C++ declares them, under the
// names the Slang source gives them. One implementation for the GPU and the CPU.
#pragma once
#include "pt/Vector.h"
#include "pt_constants.gen.h"

#include <cstddef>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
// The exports are extern "C" for their names; only C++ calls them, so returning vectors is fine.
#pragma warning(disable : 4190)
#endif
// The generated header opens the prelude's namespace, which has its own float3 and friends; it
// stays inside pt::gen, and its structs and exports are brought into pt by name.
namespace pt::gen {
#include "pt_cpu.gen.hpp"
} // namespace pt::gen
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace pt {
using gen::PtReconstructionSample;
using gen::PtPath;
using gen::TraceInstance;
using gen::Material;
using gen::Light;
using gen::PtEmissiveTriangle;
using gen::PathUniforms;
using gen::PtCpuScene;
using gen::PtHit;
using gen::PtRestirCamera;
using gen::PtRestirSurface;
using gen::PtReservoir;
using gen::PtRayPrep;
using gen::BvhBuildDescriptor;
using gen::BvhBuildControl;
using gen::BvhBuildRecord;
using gen::BvhBuildStatus;
using gen::BvhBuildStatus2;
using gen::BvhSegment;
using gen::BvhLbvhControl;
using gen::BvhSortControl;
using gen::BvhCollapseControl;
using gen::BvhCollapseStatus;
using gen::AnimateControl;
using gen::PtTemporalHistory;
using gen::PtTemporalUniforms;
using gen::PtSurface;
using gen::PtBsdf;
using gen::PtWideNode;
} // namespace pt

namespace pt {

template <class T> using StructuredBuffer = SLANG_PRELUDE_NAMESPACE::StructuredBuffer<T>;

// Host data as the generated code's StructuredBuffer, of its own element type or, for the host
// vectors, of the prelude vector they derive from (the same layout).
template <class T> struct BufferView {
  const T *data;
  std::size_t count;
  template <class U> operator StructuredBuffer<U>() const {
    static_assert((std::is_same_v<U, T> || std::is_base_of_v<U, T>) && sizeof(U) == sizeof(T), "a buffer of another element type");
    StructuredBuffer<U> view;
    view.data = const_cast<U *>(static_cast<const U *>(data));
    view.count = count;
    return view;
  }
};
template <class T> BufferView<T> buffer(const T *data, std::size_t count) { return {data, count}; }
template <class T> BufferView<T> buffer(const std::vector<T> &values) { return {values.data(), values.size()}; }

// Russian roulette starts once a path has scattered this many times (PathUniforms.path.y),
// in every backend: the headless CLI, the window's CPU tracer and the GPU tracers.
inline constexpr float kRouletteStartBounce = 3.0f;

// The generated structs use natural C++ layout; these are the SPIR-V sizes, so a failure is a
// field the two targets pad differently.
static_assert(sizeof(Material) == 128 && sizeof(Light) == 64 && sizeof(TraceInstance) == 176);
static_assert(sizeof(PathUniforms) == 224 && sizeof(PtEmissiveTriangle) == 112 && sizeof(PtReconstructionSample) == 128);
static_assert(sizeof(PtReservoir) == 32 && sizeof(PtRestirSurface) == 64);
static_assert(sizeof(BvhBuildDescriptor) == 32 && sizeof(BvhBuildControl) == 32);

// slangc passes a struct in-parameter by pointer and never writes through it; the wrappers take
// values or const references and cast.

// basalt_random
inline float hashFloat(uint value) { return gen::ptCpuHashFloat(value); }
inline float4 pathRandom4(uint seed, uint bounce, uint group) { return gen::ptCpuPathRandom4(seed, bounce, group); }
inline uint pathSeed(uint x, uint y, uint sampleIndex, uint seed) { return gen::ptCpuPathSeed(x, y, sampleIndex, seed); }
inline uint pcgHash(uint value) { return gen::ptCpuPcgHash(value); }
inline float radicalInverse(uint bits) { return gen::ptCpuRadicalInverse(bits); }

// basalt_shading
inline float distributionGGX(float normalDotHalf, float alpha) { return gen::ptCpuDistributionGGXScalar(normalDotHalf, alpha); }
inline float visibilitySmith(float normalDotView, float normalDotLight, float alpha) {
  return gen::ptCpuVisibilitySmith(normalDotView, normalDotLight, alpha);
}
inline float2 equirectangularUV(float3 direction) { return gen::ptCpuEquirectangularUV(direction); }
inline float3 applyRows(float4 row0, float4 row1, float4 row2, float3 point) { return gen::ptCpuApplyRows(row0, row1, row2, point); }
inline float3 cosineSampleHemisphere(float3 normal, float2 xi) { return gen::ptCpuCosineSampleHemisphere(normal, xi); }
inline float3 fresnelSchlick(float3 f0, float viewDotHalf) { return gen::ptCpuFresnelSchlick(f0, viewDotHalf); }
inline void orthonormalBasis(float3 normal, float3 &tangent, float3 &bitangent) {
  gen::ptCpuOrthonormalBasis(normal, &tangent, &bitangent);
}

// pt_bsdf
inline float ptDistributionGGX(float3 normal, float3 halfVector, float alpha) { return gen::ptCpuDistributionGGX(normal, halfVector, alpha); }
inline float ptSmithG1(float cosine, float alpha) { return gen::ptCpuSmithG1(cosine, alpha); }
inline float ptSpecularAlbedo(const std::vector<float> &table, float normalDotView, float roughness) {
  return gen::ptCpuSpecularAlbedo(buffer(table), normalDotView, roughness);
}
inline float3 ptFresnelInterface(float3 f0, float cosine, float etap) { return gen::ptCpuFresnelInterface(f0, cosine, etap); }
inline float3 ptSampleVisibleNormal(float3 viewLocal, float alpha, float2 xi) { return gen::ptCpuSampleVisibleNormal(viewLocal, alpha, xi); }
inline PtBsdf ptMakeBsdf(PtSurface surface, float3 view, const std::vector<float> &albedoTable) {
  return gen::ptCpuMakeBsdf(&surface, view, buffer(albedoTable));
}
inline float3 ptBsdfEvaluate(PtBsdf bsdf, float3 geometricNormal, float3 light, float &pdf) {
  return gen::ptCpuBsdfEvaluate(&bsdf, geometricNormal, light, &pdf);
}
inline float3 ptBsdfSampleDirection(PtBsdf bsdf, float3 xi) { return gen::ptCpuBsdfSampleDirection(&bsdf, xi); }
inline float3 ptTransmittedDirection(PtBsdf bsdf, float3 micro) { return gen::ptCpuTransmittedDirection(&bsdf, micro); }
inline void ptClearV7Layers(PtSurface &surface, float ior, float entering) { gen::ptCpuClearLayers(&surface, ior, entering); }

// pt_bvh_build
inline float ptOrderedFloat(uint u) { return gen::ptCpuOrderedFloat(u); }
inline uint ptOrderedBits(float f) { return gen::ptCpuOrderedBits(f); }

// pt_lights
inline float ptEnvironmentPdf(const std::vector<float> &distribution, float4 info, float3 direction) {
  return gen::ptCpuEnvironmentPdf(buffer(distribution), info, direction);
}
inline float3 ptEquirectangularDirection(float2 uv) { return gen::ptCpuEquirectangularDirection(uv); }
inline float3 ptPunctualLight(Light light, float3 position, float3 &direction, float &distance) {
  return gen::ptCpuPunctualLight(&light, position, &direction, &distance);
}
inline float3 ptSampleCone(float3 axis, float cosineMax, float2 xi) { return gen::ptCpuSampleCone(axis, cosineMax, xi); }
inline float3 ptSampleEnvironmentDirection(const std::vector<float> &distribution, float4 info, float2 xi, float &pdf) {
  return gen::ptCpuSampleEnvironmentDirection(buffer(distribution), info, xi, &pdf);
}
inline uint ptFindInterval(const std::vector<float> &data, uint offset, uint size, float value) {
  return gen::ptCpuFindInterval(buffer(data), offset, size, value);
}
inline uint ptEmissiveSelect(const PtEmissiveTriangle *triangles, uint count, float xi) {
  return gen::ptCpuEmissiveSelect(buffer(triangles, count), count, xi);
}
inline uint ptEmissiveFind(const PtEmissiveTriangle *triangles, uint count, uint instance, uint primitive) {
  return gen::ptCpuEmissiveFind(buffer(triangles, count), count, instance, primitive);
}

// pt_path
inline float ptGuideLayers(float4 transmission, float4 clearcoat) { return gen::ptCpuGuideLayers(transmission, clearcoat); }
inline float ptLuminance(float3 v) { return gen::ptCpuLuminance(v); }
inline float ptMaxComponent(float3 v) { return gen::ptCpuMaxComponent(v); }
inline uint ptTraceSeed(uint seed, uint bounce, uint purpose) { return gen::ptCpuTraceSeed(seed, bounce, purpose); }
inline void ptApplyThinLens(float4 lens, float3 forward, float3 right, float3 up, float2 xi, float3 &origin, float3 &direction) {
  gen::ptCpuApplyThinLens(lens, forward, right, up, xi, &origin, &direction);
}
inline PtReconstructionSample ptEmptyReconstructionSample() { return gen::ptCpuEmptyReconstructionSample(); }

// pt_raycone
inline float ptConeLodBase(float uvCross, float3 worldCross, float3 direction, float coneWidth) {
  return gen::ptCpuConeLodBase(uvCross, worldCross, direction, coneWidth);
}
inline float ptConeWidthOrLevelZero(float2 cone, float t) { return gen::ptCpuConeWidthOrLevelZero(cone, t); }
inline float ptLog2(float x) { return gen::ptCpuLog2(x); }
inline float ptUvCross(float2 t0, float2 t1, float2 t2) { return gen::ptCpuUvCross(t0, t1, t2); }
inline float2 ptCameraCone(float4 lens) { return gen::ptCpuCameraCone(lens); }
inline float2 ptConeScattered(float2 cone, float width, float lobeAlpha) { return gen::ptCpuConeScattered(cone, width, lobeAlpha); }
inline float2 ptConeShadow(float2 cone, float width) { return gen::ptCpuConeShadow(cone, width); }
inline float3 ptWorldCross(float4 row0, float4 row1, float4 row2, float3 p0, float3 p1, float3 p2) {
  return gen::ptCpuWorldCross(row0, row1, row2, p0, p1, p2);
}
inline float ptTextureFootprint(float lodBase, float width, float height) { return gen::ptCpuTextureFootprint(lodBase, width, height); }
inline float ptTextureLevel(float lodBase, float width, float height, float levels) {
  return gen::ptCpuTextureLevel(lodBase, width, height, levels);
}

// pt_restir
inline PtReservoir ptEmptyReservoir() { return gen::ptCpuEmptyReservoir(); }
inline float ptRestirStreamRandom(uint stream, uint k) { return gen::ptCpuRestirStreamRandom(stream, k); }
inline uint ptRestirStream(uint seed, uint candidates) { return gen::ptCpuRestirStream(seed, candidates); }
inline void ptReservoirStream(PtReservoir &r, uint light, float2 params, float weight, float target, float random) {
  gen::ptCpuReservoirStream(&r, light, params, weight, target, random);
}
inline bool ptRestirSimilar(PtRestirSurface owner, PtRestirSurface neighbour, float depthAtNeighbour) {
  return gen::ptCpuRestirSimilar(&owner, &neighbour, depthAtNeighbour);
}

// pt_exact_float
inline bool ptExactIsNaN(uint a) { return gen::ptCpuExactIsNaN(a); }
inline uint ptExactAdd(uint a, uint b) { return gen::ptCpuExactAdd(a, b); }
inline uint ptExactSubtract(uint a, uint b) { return gen::ptCpuExactSubtract(a, b); }
inline uint ptExactMultiply(uint a, uint b) { return gen::ptCpuExactMultiply(a, b); }
inline uint ptExactDivide(uint a, uint b) { return gen::ptCpuExactDivide(a, b); }
inline bool ptExactLess(uint a, uint b) { return gen::ptCpuExactLess(a, b); }
inline uint ptExactMin(uint a, uint b) { return gen::ptCpuExactMin(a, b); }
inline uint ptExactMax(uint a, uint b) { return gen::ptCpuExactMax(a, b); }
inline uint ptExactNextAfter(uint a, bool up) { return gen::ptCpuExactNextAfter(a, up); }
inline uint ptExactQuantise(uint q, bool up) { return gen::ptCpuExactQuantise(q, up); }
inline uint ptExactQuantiseQuotient(uint n, uint d, bool up) { return gen::ptCpuExactQuantiseQuotient(n, d, up); }
inline uint ptExactQuotientBitwise(uint sigA, uint sigB, uint bits) { return gen::ptCpuExactQuotientBitwise(sigA, sigB, bits); }
inline uint ptExactQuotientFrom(uint sigA, uint sigB, uint bits, uint quotient) { return gen::ptCpuExactQuotientFrom(sigA, sigB, bits, quotient); }

// pt_temporal: the reconstruction's CPU reference over whole images of u.image.x * u.image.y.
inline PtTemporalHistory ptTemporalPixel(PtTemporalUniforms u, uint x, uint y, const PtReconstructionSample *samples,
                                         const PtReconstructionSample *previousSamples, const PtTemporalHistory *previous) {
  const std::size_t pixels = static_cast<std::size_t>(u.image.x) * u.image.y;
  return gen::ptCpuTemporalPixel(&u, x, y, buffer(samples, pixels), buffer(previousSamples, pixels), buffer(previous, pixels));
}
inline PtTemporalHistory ptAtrousPixel(PtTemporalUniforms u, uint x, uint y, const PtReconstructionSample *samples,
                                       const PtTemporalHistory *input) {
  const std::size_t pixels = static_cast<std::size_t>(u.image.x) * u.image.y;
  return gen::ptCpuAtrousPixel(&u, x, y, buffer(samples, pixels), buffer(input, pixels));
}
inline bool ptFiniteColor(float3 v) { return gen::ptCpuFiniteColor(v); }
inline float ptTemporalLuminance(float3 v) { return gen::ptCpuTemporalLuminance(v); }

} // namespace pt
