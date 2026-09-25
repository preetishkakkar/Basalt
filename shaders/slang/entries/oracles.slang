// Batched traversal oracles for the tests: the host supplies an independently generated ray
// corpus and compares these records with double-precision world-space triangle intersections.
// The same candidate policy runs through the binary and wide software BVHs, ray queries and the
// full ray pipeline, so the host compares traversal APIs without changing its tolerances.
import pt_integrator;
import pt_wide_bvh;
import pt_gpu;

struct OracleRay {
  float4 originAndMin;
  float4 directionAndMax;
  uint4 control; // x mask, y alpha seed, z nonzero for occlusion, w unused
};

struct OracleHit {
  float4 distanceAndBarycentric; // x distance, yz barycentrics
  uint4 identity;                // x found, y instance, z primitive, w unused
};

StructuredBuffer<OracleRay> rays;
RWStructuredBuffer<OracleHit> hits;
ConstantBuffer<uint4> control; // x ray count, y the wavefront queue's capacity
PtGeometry geometry;
GpuMaps maps;
StructuredBuffer<float4> bvhNodes;      // binary
StructuredBuffer<PtWideNode> wideNodes; // BVH4 / BVH8
StructuredBuffer<float4> bvhTriangles;
RaytracingAccelerationStructure accelerationStructure;

OracleHit missRecord(OracleRay input) {
  OracleHit output;
  output.distanceAndBarycentric = float4(input.directionAndMax.w, 0.0f, 0.0f, 0.0f);
  output.identity = uint4(0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u);
  return output;
}

// The software tracers take their rays from zero. Production paths offset their origins and use
// tMin zero; shifting here exercises an arbitrary finite lower extent without changing the
// shared traversal signature.
void softwareOracle<T : IPtTracer>(T tracer, uint id) {
  if (id >= control.x) return;
  const OracleRay input = rays[id];
  OracleHit output = missRecord(input);
  const float minimum = input.originAndMin.w;
  const float maximum = input.directionAndMax.w;
  const float3 origin = input.originAndMin.xyz + input.directionAndMax.xyz * minimum;
  const PtHit hit = tracer.trace(origin, input.directionAndMax.xyz, maximum - minimum, input.control.x, input.control.y,
                                 float2(-1.0f, 0.0f), input.control.z != 0u);
  if (hit.found != 0u) {
    output.distanceAndBarycentric = float4(hit.t + minimum, hit.barycentric, 0.0f);
    output.identity = uint4(1u, hit.instance, hit.primitive, 0u);
  }
  hits[id] = output;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_oracle(uint3 id: SV_DispatchThreadID) {
  BinaryBvhTracer<GpuMaps> tracer;
  tracer.nodes = bvhNodes;
  tracer.triangles = bvhTriangles;
  tracer.geometry = geometry;
  tracer.maps = maps;
  softwareOracle(tracer, id.x);
}

// The deep traversal stack: tests may trace any tree the conversion accepts.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_wide_oracle(uint3 id: SV_DispatchThreadID) {
  WideBvhTracer<GpuMaps, kWideStackDeep, false> tracer;
  tracer.nodes = wideNodes;
  tracer.triangles = bvhTriangles;
  tracer.geometry = geometry;
  tracer.maps = maps;
  softwareOracle(tracer, id.x);
}

// Ray queries take the corpus's own tMin; masked candidates are tested at level zero.
[shader("compute")]
[numthreads(64, 1, 1)]
void ray_query_oracle(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.x) return;
  const OracleRay input = rays[id];
  OracleHit output = missRecord(input);
  RayDesc ray;
  ray.Origin = input.originAndMin.xyz;
  ray.Direction = input.directionAndMax.xyz;
  ray.TMin = input.originAndMin.w;
  ray.TMax = input.directionAndMax.w;
  RayQuery<RAY_FLAG_NONE> query;
  query.TraceRayInline(accelerationStructure,
                       kPtRayFlags | (input.control.z != 0u ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE),
                       input.control.x, ray);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
        ptCandidateSolid(geometry, maps, query.CandidateInstanceIndex(), query.CandidatePrimitiveIndex(),
                         query.CandidateTriangleBarycentrics(), query.CandidateTriangleFrontFace() ? 1u : 0u, input.control.y,
                         float3(0.0f), -1.0f))
      query.CommitNonOpaqueTriangleHit();
  }
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    output.distanceAndBarycentric = float4(query.CommittedRayT(), query.CommittedTriangleBarycentrics(), 0.0f);
    output.identity = uint4(1u, query.CommittedInstanceIndex(), query.CommittedPrimitiveIndex(), 0u);
  }
  hits[id] = output;
}

// The full ray pipeline's form.
struct PipelineOraclePayload {
  float t;
  float2 barycentric;
  uint found;
  uint instance;
  uint primitive;
  uint alphaSeed;
  uint occlusion;
};

[shader("raygeneration")]
void pipeline_oracle_generate() {
  const uint id = DispatchRaysIndex().x;
  if (id >= control.x) return;
  const OracleRay input = rays[id];
  PipelineOraclePayload payload = {};
  payload.t = input.directionAndMax.w;
  payload.instance = 0xFFFFFFFFu;
  payload.primitive = 0xFFFFFFFFu;
  payload.alphaSeed = input.control.y;
  payload.occlusion = input.control.z;
  RayDesc ray;
  ray.Origin = input.originAndMin.xyz;
  ray.Direction = input.directionAndMax.xyz;
  ray.TMin = input.originAndMin.w;
  ray.TMax = input.directionAndMax.w;
  TraceRay(accelerationStructure, kPtRayFlags, input.control.x, 0u, 1u, 0u, ray, payload);
  OracleHit output;
  output.distanceAndBarycentric = float4(payload.t, payload.barycentric, 0.0f);
  output.identity = uint4(payload.found, payload.instance, payload.primitive, 0u);
  hits[id] = output;
}

[shader("miss")]
void pipeline_oracle_miss(inout PipelineOraclePayload payload) { payload.found = 0u; }

[shader("closesthit")]
void pipeline_oracle_closest(inout PipelineOraclePayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  payload.t = RayTCurrent();
  payload.barycentric = attributes.barycentrics;
  payload.instance = InstanceIndex();
  payload.primitive = PrimitiveIndex();
  payload.found = 1u;
}

[shader("anyhit")]
void pipeline_oracle_alpha(inout PipelineOraclePayload payload, BuiltInTriangleIntersectionAttributes attributes) {
  const uint instance = InstanceIndex();
  const uint front = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1u : 0u;
  if (!ptCandidateSolid(geometry, maps, instance, PrimitiveIndex(), attributes.barycentrics, front, payload.alphaSeed, float3(0.0f),
                        -1.0f)) // level zero
    IgnoreHit();
  if (payload.occlusion != 0u) {
    payload.t = RayTCurrent();
    payload.barycentric = attributes.barycentrics;
    payload.instance = instance;
    payload.primitive = PrimitiveIndex();
    payload.found = 1u;
    AcceptHitAndEndSearch();
  }
}

// The wavefront queue's form: an enqueue pass reserves a slot per live ray (counting what a full
// queue refuses), and a trace pass traces the queue through the binary software BVH.
struct WavefrontRay {
  OracleRay ray;
  uint4 identity; // x the ray's corpus index
};

RWStructuredBuffer<WavefrontRay> queue;
RWStructuredBuffer<uint> counters; // [0] reserved slots, [1] refused rays

[shader("compute")]
[numthreads(64, 1, 1)]
void wavefront_enqueue(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= control.x) return;
  const OracleRay ray = rays[id];
  if (!(ray.directionAndMax.w > ray.originAndMin.w)) return;
  uint slot;
  InterlockedAdd(counters[0], 1u, slot);
  if (slot < control.y) {
    queue[slot].ray = ray;
    queue[slot].identity = uint4(id, 0u, 0u, 0u);
  } else {
    InterlockedAdd(counters[1], 1u);
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wavefront_trace(uint3 thread: SV_DispatchThreadID) {
  const uint id = thread.x;
  if (id >= min(counters[0], control.y)) return;
  const WavefrontRay queued = queue[id];
  const OracleRay input = queued.ray;
  OracleHit output = missRecord(input);
  const float minimum = input.originAndMin.w;
  const float3 origin = input.originAndMin.xyz + input.directionAndMax.xyz * minimum;
  BinaryBvhTracer<GpuMaps> tracer;
  tracer.nodes = bvhNodes;
  tracer.triangles = bvhTriangles;
  tracer.geometry = geometry;
  tracer.maps = maps;
  const PtHit hit = tracer.trace(origin, input.directionAndMax.xyz, input.directionAndMax.w - minimum, input.control.x,
                                 input.control.y, float2(-1.0f, 0.0f), input.control.z != 0u);
  if (hit.found != 0u) {
    output.distanceAndBarycentric = float4(hit.t + minimum, hit.barycentric, 0.0f);
    output.identity = uint4(1u, hit.instance, hit.primitive, 0u);
  }
  hits[queued.identity.x] = output;
}
