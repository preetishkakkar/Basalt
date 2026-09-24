// The nearest hit of one wavefront ray, for the compute intersectors (ray queries, binary
// and wide own-BVH traversal). Included as statements by the intersect stage and the fused
// shade stage so both trace identically. In scope: origin, direction, seed, cone, a zeroed PtHit
// hit, uniforms, PT_SCENE_ARGS, PT_TEXTURE_ARGS and the backend's scene or bvhNodes /
// bvhTriangles. The ray-pipeline intersect stage traces through trace_ray instead.
#if defined(BASALT_RAY_TRACING)
{
  ray ptWaveRay(origin, direction, 0.0f, kPtInfinity);
  intersection_params ptWaveParams;
  intersection_query<triangle_data, instancing> q(ptWaveRay, scene, uniforms.counts.y, ptWaveParams);
  while (q.next()) {
    if (q.get_candidate_intersection_type() == intersection_type::triangle &&
        (traceInstances[q.get_candidate_instance_id()].flags & kInstanceBlended) != 0u)
      hit.ambiguous = 1u;
    if (q.get_candidate_intersection_type() == intersection_type::triangle &&
        ptCandidateSolid(PT_SCENE_ARGS, PT_TEXTURE_ARGS, q.get_candidate_instance_id(), q.get_candidate_primitive_id(),
                         q.get_candidate_triangle_barycentric_coord(),
                         (q.is_candidate_triangle_front_facing() ? 1u : 0u),
                         seed, direction, ptConeWidthOrLevelZero(cone, q.get_candidate_triangle_distance())))
      q.commit_triangle_intersection();
  }
  if (q.get_committed_intersection_type() == intersection_type::triangle) {
    hit.found = 1u;
    hit.t = q.get_committed_distance();
    hit.barycentric = q.get_committed_triangle_barycentric_coord();
    hit.instance = q.get_committed_instance_id();
    hit.primitive = q.get_committed_primitive_id();
  }
}
#elif defined(BASALT_WIDE_BVH)
hit = ptTraceWideBvh(bvhNodes, bvhTriangles, PT_SCENE_ARGS, PT_TEXTURE_ARGS, origin, direction, kPtInfinity,
                     uniforms.counts.y, seed, cone, 0u);
#else
hit = ptTraceBvh(bvhNodes, bvhTriangles, PT_SCENE_ARGS, PT_TEXTURE_ARGS, origin, direction, kPtInfinity,
                 uniforms.counts.y, seed, cone, 0u);
#endif
