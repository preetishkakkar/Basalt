// The shared functions host code calls directly (texture tables, the BVH builders' reference
// code, the tests), exported to C++ under a ptCpu name; src/pt/Shared.h gives them their Slang
// names back on the host. Part of the pt_cpu module.
implementing pt_cpu;

// The records the host writes for, or reads back from, the GPU BVH builders and the vertex
// animation. Slang writes a type into the C++ header only when an export reaches it; this
// export is how these reach it.
public export __extern_cpp void ptCpuDeclareRecords(out BvhBuildDescriptor descriptor, out BvhBuildControl control,
                                                    out BvhBuildRecord record, out BvhBuildStatus status,
                                                    out BvhBuildStatus2 status2, out BvhSegment segment,
                                                    out BvhLbvhControl lbvh, out BvhSortControl sort,
                                                    out BvhCollapseControl collapse, out BvhCollapseStatus collapseStatus,
                                                    out AnimateControl animate) {
  descriptor = {};
  control = {};
  record = {};
  status = {};
  status2 = {};
  segment = {};
  lbvh = {};
  sort = {};
  collapse = {};
  collapseStatus = {};
  animate = {};
}

// basalt_random
public export __extern_cpp float ptCpuHashFloat(uint value) { return hashFloat(value); }
public export __extern_cpp float4 ptCpuPathRandom4(uint seed, uint bounce, uint group) { return pathRandom4(seed, bounce, group); }
public export __extern_cpp uint ptCpuPathSeed(uint x, uint y, uint sampleIndex, uint seed) { return pathSeed(x, y, sampleIndex, seed); }
public export __extern_cpp uint ptCpuPcgHash(uint value) { return pcgHash(value); }
public export __extern_cpp float ptCpuRadicalInverse(uint bits) { return radicalInverse(bits); }

// basalt_shading
public export __extern_cpp float ptCpuDistributionGGXScalar(float normalDotHalf, float alpha) { return distributionGGX(normalDotHalf, alpha); }
public export __extern_cpp float ptCpuVisibilitySmith(float normalDotView, float normalDotLight, float alpha) {
  return visibilitySmith(normalDotView, normalDotLight, alpha);
}
public export __extern_cpp float2 ptCpuEquirectangularUV(float3 direction) { return equirectangularUV(direction); }
public export __extern_cpp float3 ptCpuApplyRows(float4 row0, float4 row1, float4 row2, float3 point) {
  return applyRows(row0, row1, row2, point);
}
public export __extern_cpp float3 ptCpuCosineSampleHemisphere(float3 normal, float2 xi) { return cosineSampleHemisphere(normal, xi); }
public export __extern_cpp float3 ptCpuFresnelSchlick(float3 f0, float viewDotHalf) { return fresnelSchlick(f0, viewDotHalf); }
public export __extern_cpp void ptCpuOrthonormalBasis(float3 normal, out float3 tangent, out float3 bitangent) {
  orthonormalBasis(normal, tangent, bitangent);
}

// pt_bsdf
public export __extern_cpp float ptCpuDistributionGGX(float3 normal, float3 halfVector, float alpha) {
  return ptDistributionGGX(normal, halfVector, alpha);
}
public export __extern_cpp float ptCpuSmithG1(float cosine, float alpha) { return ptSmithG1(cosine, alpha); }
public export __extern_cpp float ptCpuSpecularAlbedo(StructuredBuffer<float> table, float normalDotView, float roughness) {
  return ptSpecularAlbedo(table, normalDotView, roughness);
}
public export __extern_cpp float3 ptCpuFresnelInterface(float3 f0, float cosine, float etap) { return ptFresnelInterface(f0, cosine, etap); }
public export __extern_cpp float3 ptCpuSampleVisibleNormal(float3 viewLocal, float alpha, float2 xi) {
  return ptSampleVisibleNormal(viewLocal, alpha, xi);
}

// pt_bvh_build
public export __extern_cpp float ptCpuOrderedFloat(uint u) { return ptOrderedFloat(u); }
public export __extern_cpp uint ptCpuOrderedBits(float f) { return ptOrderedBits(f); }

// pt_lights
public export __extern_cpp float ptCpuEnvironmentPdf(StructuredBuffer<float> distribution, float4 info, float3 direction) {
  return ptEnvironmentPdf(distribution, info, direction);
}
public export __extern_cpp float3 ptCpuEquirectangularDirection(float2 uv) { return ptEquirectangularDirection(uv); }
public export __extern_cpp float3 ptCpuPunctualLight(Light light, float3 position, out float3 direction, out float distance) {
  return ptPunctualLight(light, position, direction, distance);
}
public export __extern_cpp float3 ptCpuSampleCone(float3 axis, float cosineMax, float2 xi) { return ptSampleCone(axis, cosineMax, xi); }
public export __extern_cpp float3 ptCpuSampleEnvironmentDirection(StructuredBuffer<float> distribution, float4 info, float2 xi,
                                                                  out float pdf) {
  return ptSampleEnvironmentDirection(distribution, info, xi, pdf);
}

// pt_path
public export __extern_cpp float ptCpuGuideLayers(float4 transmission, float4 clearcoat) { return ptGuideLayers(transmission, clearcoat); }
public export __extern_cpp float ptCpuLuminance(float3 v) { return ptLuminance(v); }
public export __extern_cpp float ptCpuMaxComponent(float3 v) { return ptMaxComponent(v); }
public export __extern_cpp uint ptCpuTraceSeed(uint seed, uint bounce, uint purpose) { return ptTraceSeed(seed, bounce, purpose); }
public export __extern_cpp void ptCpuApplyThinLens(float4 lens, float3 forward, float3 right, float3 up, float2 xi, inout float3 origin,
                                                   inout float3 direction) {
  ptApplyThinLens(lens, forward, right, up, xi, origin, direction);
}

// pt_raycone
public export __extern_cpp float ptCpuConeLodBase(float uvCross, float3 worldCross, float3 direction, float coneWidth) {
  return ptConeLodBase(uvCross, worldCross, direction, coneWidth);
}
public export __extern_cpp float ptCpuConeWidthOrLevelZero(float2 cone, float t) { return ptConeWidthOrLevelZero(cone, t); }
public export __extern_cpp float ptCpuLog2(float x) { return ptLog2(x); }
public export __extern_cpp float ptCpuUvCross(float2 t0, float2 t1, float2 t2) { return ptUvCross(t0, t1, t2); }
public export __extern_cpp float2 ptCpuCameraCone(float4 lens) { return ptCameraCone(lens); }
public export __extern_cpp float2 ptCpuConeScattered(float2 cone, float width, float lobeAlpha) { return ptConeScattered(cone, width, lobeAlpha); }
public export __extern_cpp float2 ptCpuConeShadow(float2 cone, float width) { return ptConeShadow(cone, width); }
public export __extern_cpp float3 ptCpuWorldCross(float4 row0, float4 row1, float4 row2, float3 p0, float3 p1, float3 p2) {
  return ptWorldCross(row0, row1, row2, p0, p1, p2);
}

// pt_restir
public export __extern_cpp float ptCpuRestirStreamRandom(uint stream, uint k) { return ptRestirStreamRandom(stream, k); }
public export __extern_cpp uint ptCpuRestirStream(uint seed, uint candidates) { return ptRestirStream(seed, candidates); }

// pt_temporal: the reconstruction's CPU reference.
public export __extern_cpp PtTemporalHistory ptCpuTemporalPixel(PtTemporalUniforms u, uint x, uint y,
                                                                StructuredBuffer<PtReconstructionSample> samples,
                                                                StructuredBuffer<PtReconstructionSample> previousSamples,
                                                                StructuredBuffer<PtTemporalHistory> previous) {
  return ptTemporalPixel(u, x, y, samples, previousSamples, previous);
}
public export __extern_cpp PtTemporalHistory ptCpuAtrousPixel(PtTemporalUniforms u, uint x, uint y,
                                                              StructuredBuffer<PtReconstructionSample> samples,
                                                              StructuredBuffer<PtTemporalHistory> input) {
  return ptAtrousPixel(u, x, y, samples, input);
}
public export __extern_cpp bool ptCpuFiniteColor(float3 v) { return ptFiniteColor(v); }
public export __extern_cpp float ptCpuTemporalLuminance(float3 v) { return ptTemporalLuminance(v); }

// Records and the materials' response.
public export __extern_cpp PtReconstructionSample ptCpuEmptyReconstructionSample() { return PtReconstructionSample.empty(); }
public export __extern_cpp PtReservoir ptCpuEmptyReservoir() { return PtReservoir.empty(); }
public export __extern_cpp void ptCpuClearLayers(inout PtSurface surface, float ior, float entering) { surface.clearLayers(ior, entering); }
public export __extern_cpp PtSurface ptCpuSurfaceAt(PtCpuScene frame, PtHit hit, float3 rayDirection, float coneWidth) {
  const PtScene<HostMaps> scene = frame.scene();
  return ptSurfaceAt(scene.geometry, scene.maps, hit, rayDirection, coneWidth);
}
public export __extern_cpp PtBsdf ptCpuMakeBsdf(PtSurface surface, float3 view, StructuredBuffer<float> albedoTable) {
  return PtBsdf(surface, view, albedoTable);
}
public export __extern_cpp float3 ptCpuBsdfEvaluate(PtBsdf bsdf, float3 geometricNormal, float3 light, out float pdf) {
  return bsdf.evaluate(geometricNormal, light, pdf);
}
public export __extern_cpp float3 ptCpuBsdfSampleDirection(PtBsdf bsdf, float3 xi) { return bsdf.sampleDirection(xi); }
public export __extern_cpp float3 ptCpuTransmittedDirection(PtBsdf bsdf, float3 micro) { return bsdf.transmittedDirection(micro); }

// The environment distribution's CDF search.
public export __extern_cpp uint ptCpuFindInterval(StructuredBuffer<float> data, uint offset, uint size, float value) {
  return ptFindInterval(data, offset, size, value);
}

// The emitter list's searches.
public export __extern_cpp uint ptCpuEmissiveSelect(StructuredBuffer<PtEmissiveTriangle> triangles, uint count, float xi) {
  PtEmitters emitters;
  emitters.triangles = triangles;
  emitters.count = count;
  return emitters.select(xi);
}
public export __extern_cpp uint ptCpuEmissiveFind(StructuredBuffer<PtEmissiveTriangle> triangles, uint count, uint instance,
                                                  uint primitive) {
  PtEmitters emitters;
  emitters.triangles = triangles;
  emitters.count = count;
  return emitters.find(instance, primitive);
}

// The binary software BVH, whatever the frame's intersector: the reference the others match.
public export __extern_cpp PtHit ptCpuTraceBvh(PtCpuScene frame, float3 origin, float3 direction, float tMax, uint mask, uint seed,
                                               float2 cone, bool anyHit) {
  return frame.binary().trace(origin, direction, tMax, mask, seed, cone, anyHit);
}

// The quantized wide software BVH, children sorted far to near, over frame's instances and
// triangles (the wide tree's own): the host's scalar wide traversal.
public export __extern_cpp PtHit ptCpuTraceWideBvh(PtCpuScene frame, StructuredBuffer<PtWideNode> nodes, float3 origin,
                                                   float3 direction, float tMax, uint mask, uint seed, float2 cone, bool anyHit) {
  WideBvhTracer<HostMaps, kWideStackDeep, true> tracer;
  tracer.nodes = nodes;
  tracer.triangles = frame.bvhTriangles;
  tracer.geometry = frame.scene().geometry;
  tracer.maps.host = frame.host;
  return tracer.trace(origin, direction, tMax, mask, seed, cone, anyHit);
}

// pt_exact_float: the software IEEE operations, checked against the host's own.
public export __extern_cpp bool ptCpuExactIsNaN(uint a) { return ptExactIsNaN(a); }
public export __extern_cpp uint ptCpuExactAdd(uint a, uint b) { return ptExactAdd(a, b); }
public export __extern_cpp uint ptCpuExactSubtract(uint a, uint b) { return ptExactSubtract(a, b); }
public export __extern_cpp uint ptCpuExactMultiply(uint a, uint b) { return ptExactMultiply(a, b); }
public export __extern_cpp uint ptCpuExactDivide(uint a, uint b) { return ptExactDivide(a, b); }
public export __extern_cpp bool ptCpuExactLess(uint a, uint b) { return ptExactLess(a, b); }
public export __extern_cpp uint ptCpuExactMin(uint a, uint b) { return ptExactMin(a, b); }
public export __extern_cpp uint ptCpuExactMax(uint a, uint b) { return ptExactMax(a, b); }
public export __extern_cpp uint ptCpuExactNextAfter(uint a, bool up) { return ptExactNextAfter(a, up); }
public export __extern_cpp uint ptCpuExactQuantise(uint q, bool up) { return ptExactQuantise(q, up); }
public export __extern_cpp uint ptCpuExactQuantiseQuotient(uint n, uint d, bool up) { return ptExactQuantiseQuotient(n, d, up); }
public export __extern_cpp uint ptCpuExactQuotientBitwise(uint sigA, uint sigB, uint bits) { return ptExactQuotientBitwise(sigA, sigB, bits); }
public export __extern_cpp uint ptCpuExactQuotientFrom(uint sigA, uint sigB, uint bits, uint quotient) { return ptExactQuotientFrom(sigA, sigB, bits, quotient); }

// pt_restir: the reservoirs and the resampling, for the tests.
public export __extern_cpp void ptCpuReservoirStream(inout PtReservoir r, uint light, float2 params, float weight, float target,
                                                     float random) {
  r.stream(light, params, weight, target, random);
}
public export __extern_cpp bool ptCpuRestirSimilar(PtRestirSurface owner, PtRestirSurface neighbour, float depthAtNeighbour) {
  return owner.similar(neighbour, depthAtNeighbour);
}
public export __extern_cpp PtReservoir ptCpuRestirInitial(PtCpuScene frame, PtBsdf bsdf, float3 position, float3 geometricNormal,
                                                          float2 cone, uint seed, uint candidates) {
  return frame.restir().initial(bsdf, position, geometricNormal, cone, seed, candidates);
}
public export __extern_cpp float3 ptCpuRestirIntegrand(PtCpuScene frame, PtBsdf bsdf, float3 position, float3 geometricNormal, uint light,
                                                       float2 params, float2 cone, out float3 direction, out float reach,
                                                       out float sourcePdf, out float solidAnglePdf, out float bsdfPdf) {
  return frame.restir().integrand(bsdf, position, geometricNormal, light, params, cone, direction, reach, sourcePdf, solidAnglePdf,
                                  bsdfPdf);
}
