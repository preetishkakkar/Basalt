// Early split clipping on the GPU: pt::clipReferences's references (src/pt/Bvh.cpp), the same
// bytes, operation for operation: floats compared by their ordered bits, the clipping in double
// precision with explicit comparisons, the rounding to float outwards by stepping the bits, and
// each bottom level's box area sum in the host's block order. No float values at all: a device
// may flush subnormals in float arithmetic, conversions and even moves (a Hairball-like mesh of
// subnormal triangles lost half its area), so positions are read as bits, boxes and split
// positions are float bits, compared by their ordered bits, widened to double and rounded back to
// float in integers (widen, narrow) as the host converts. Sequence: bvh_clip_area_blocks
// (a block's area sum), bvh_clip_sums (a bottom level's), bvh_clip_count (each triangle's
// references and their prefix in its block of kClipScan), bvh_clip_scan (the blocks'),
// bvh_clip_instances (each bottom level's range, which the host reads for its layout), then
// bvh_clip_emit writes the references.
import pt_bvh_build;

static const uint kClipDepth = 6u;   // pt::kMaxClipReferences = 1 << kClipDepth
static const uint kClipPoints = 3u + kClipDepth;
static const uint kClipScan = 256u;  // triangles per count workgroup

ConstantBuffer<BvhClipControl> control;
StructuredBuffer<TraceInstance> instances;
StructuredBuffer<uint> indices;
StructuredBuffer<uint> vertexBits;   // the positions, as bits
StructuredBuffer<uint4> levels;         // per instance: first triangle, triangles, first area block, area blocks
StructuredBuffer<uint2> areaBlocks;     // per area block: its instance and first triangle there
RWStructuredBuffer<double> blockSums;   // per area block
RWStructuredBuffer<double> factorSums;  // per instance: the factor times its box area sum
RWStructuredBuffer<uint> inBlock;       // per triangle: the references before it in its scan block
RWStructuredBuffer<uint> scanTotals;    // per scan block: its references
RWStructuredBuffer<uint> scanPrefix;    // per scan block: the references before it; [blocks]: all
RWStructuredBuffer<uint2> ranges;       // per instance: its first reference and its references
RWStructuredBuffer<BvhReference> references;

groupshared uint scanWords[kClipScan];

// Float bits in the order of their values (ptOrderedBits without the float).
uint orderedBits(uint u) { return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u); }
uint lesser(uint a, uint b) { return orderedBits(a) <= orderedBits(b) ? a : b; }
uint greater(uint a, uint b) { return orderedBits(a) >= orderedBits(b) ? a : b; }
double lesserDouble(double a, double b) { return a < b ? a : b; }
double greaterDouble(double a, double b) { return a > b ? a : b; }

// A float's bits as a double, exactly (subnormals too).
double widen(uint bits) {
  const uint sign = bits & 0x80000000u;
  uint exponent = (bits >> 23u) & 0xFFu, mantissa = bits & 0x7FFFFFu;
  uint high = 0u, low = 0u;
  if (exponent == 0xFFu) {
    high = sign | 0x7FF00000u | (mantissa >> 3u);
    low = mantissa << 29u;
  } else if (exponent != 0u || mantissa != 0u) {
    int e = int(exponent) - 127;
    if (exponent == 0u) {  // subnormal: normalise
      const uint shift = 23u - firstbithigh(mantissa);
      mantissa = (mantissa << shift) & 0x7FFFFFu;
      e = -126 - int(shift);
    }
    high = sign | (uint(e + 1023) << 20u) | (mantissa >> 3u);
    low = mantissa << 29u;
  } else {
    high = sign;
  }
  return asdouble(low, high);
}

// A double rounded to the nearest float's bits, ties to even, subnormals kept (as the host
// converts).
uint narrow(double d) {
  uint low, high;
  asuint(d, low, high);
  const uint sign = high & 0x80000000u;
  const uint exponent = (high >> 20u) & 0x7FFu;
  const uint64_t mantissa = (uint64_t(high & 0xFFFFFu) << 32u) | uint64_t(low);
  if (exponent == 0x7FFu) return sign | 0x7F800000u | (mantissa != 0u ? 0x400000u : 0u);
  if (exponent == 0u) return sign;  // a double subnormal: far below the floats
  int e = int(exponent) - 1023;
  if (e > 127) return sign | 0x7F800000u;
  const uint64_t significand = mantissa | (uint64_t(1u) << 52u);
  // Normal floats keep 24 of the 53 bits; subnormal ones fewer, at 2^-149.
  const uint shift = e >= -126 ? 29u : 29u + uint(-126 - e);
  if (shift >= 64u) return sign;
  uint64_t kept = significand >> shift;
  const uint64_t rest = significand & ((uint64_t(1u) << shift) - 1u), half = uint64_t(1u) << (shift - 1u);
  if (rest > half || (rest == half && (kept & 1u) != 0u)) kept += 1u;
  if (e < -126) return sign | uint(kept);  // a carry into 2^23 is the smallest normal
  if (kept == (uint64_t(1u) << 24u)) {
    kept >>= 1u;
    e += 1;
    if (e > 127) return sign | 0x7F800000u;
  }
  return sign | (uint(e + 127) << 23u) | (uint(kept) & 0x7FFFFFu);
}

uint nextDown(uint bits) {
  if ((bits & 0x7FFFFFFFu) == 0u) return 0x80000001u;
  return (bits & 0x80000000u) != 0u ? bits + 1u : bits - 1u;
}
uint nextUp(uint bits) {
  if ((bits & 0x7FFFFFFFu) == 0u) return 0x00000001u;
  return (bits & 0x80000000u) != 0u ? bits - 1u : bits + 1u;
}
uint roundDown(double d) {
  const uint f = narrow(d);
  return widen(f) > d ? nextDown(f) : f;
}
uint roundUp(double d) {
  const uint f = narrow(d);
  return widen(f) < d ? nextUp(f) : f;
}

struct ClipBox {  // float bits
  uint low[3];
  uint high[3];
};

double clipArea(ClipBox box) {
  const double dx = widen(box.high[0]) - widen(box.low[0]), dy = widen(box.high[1]) - widen(box.low[1]),
               dz = widen(box.high[2]) - widen(box.low[2]);
  return 2.0 * (dx * dy + dy * dz + dz * dx);
}

// A triangle's corners, as the host reads them.
void corners(uint instance, uint t, out uint corner[3][3]) {
  const TraceInstance info = instances[instance];
  for (uint k = 0u; k < 3u; ++k) {
    const uint v = (indices[info.firstIndex + t * 3u + k] + info.vertexOffset) * kVertexFloats;
    corner[k][0] = vertexBits[v];
    corner[k][1] = vertexBits[v + 1u];
    corner[k][2] = vertexBits[v + 2u];
  }
}

ClipBox triangleBox(uint corner[3][3]) {
  ClipBox box;
  for (uint a = 0u; a < 3u; ++a) {
    box.low[a] = corner[0][a];
    box.high[a] = corner[0][a];
  }
  for (uint k = 1u; k < 3u; ++k)
    for (uint a = 0u; a < 3u; ++a) {
      box.low[a] = lesser(box.low[a], corner[k][a]);
      box.high[a] = greater(box.high[a], corner[k][a]);
    }
  return box;
}

struct ClipPlane {
  uint axis;
  uint c;  // float bits
  bool low;
};

// pt::clipTriangle: the references of triangle t, depth first; written from `first` when EMIT
// (else only counted: the count kernel does not bind them). Returns their count.
uint clipTriangle<let EMIT : bool>(uint corner[3][3], uint t, double count, double factorSum, uint first) {
  const ClipBox triangle = triangleBox(corner);
  double scale = 0.0;
  for (uint a = 0u; a < 3u; ++a)
    scale = greaterDouble(scale, greaterDouble(abs(widen(triangle.low[a])), abs(widen(triangle.high[a]))));
  const double pad = scale * asdouble(0u, 0x3D700000u);  // 2^-40
  ClipPlane path[kClipDepth];
  bool pendingHigh[kClipDepth];
  for (uint j = 0u; j < kClipDepth; ++j) {
    path[j].axis = 0u;
    path[j].c = 0u;
    path[j].low = true;
    pendingHigh[j] = false;
  }
  double points[2][kClipPoints][3];
  uint depth = 0u, made = 0u;
  for (;;) {
    ClipBox box = triangle;
    bool empty = false;
    if (depth > 0u) {
      uint n = 3u;
      for (uint k = 0u; k < 3u; ++k)
        for (uint a = 0u; a < 3u; ++a) points[0][k][a] = widen(corner[k][a]);
      for (uint j = 0u; j < depth && n > 0u; ++j) {
        const uint from = j & 1u, to = from ^ 1u;
        const ClipPlane plane = path[j];
        const double c = widen(plane.c);
        uint out = 0u;
        for (uint i = 0u; i < n; ++i) {
          const uint b = i + 1u == n ? 0u : i + 1u;
          const double da = plane.low ? points[from][i][plane.axis] - c : c - points[from][i][plane.axis];
          const double db = plane.low ? points[from][b][plane.axis] - c : c - points[from][b][plane.axis];
          if (da <= 0.0) {
            for (uint a = 0u; a < 3u; ++a) points[to][out][a] = points[from][i][a];
            out += 1u;
          }
          if ((da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0)) {
            const double s = da / (da - db);
            for (uint a = 0u; a < 3u; ++a) points[to][out][a] = points[from][i][a] + s * (points[from][b][a] - points[from][i][a]);
            points[to][out][plane.axis] = c;
            out += 1u;
          }
        }
        n = out;
      }
      empty = n == 0u;
      if (!empty) {
        const uint polygon = depth & 1u;
        for (uint a = 0u; a < 3u; ++a) {
          double low = points[polygon][0][a], high = points[polygon][0][a];
          for (uint i = 1u; i < n; ++i) {
            low = lesserDouble(low, points[polygon][i][a]);
            high = greaterDouble(high, points[polygon][i][a]);
          }
          box.low[a] = greater(roundDown(low - pad), triangle.low[a]);
          box.high[a] = lesser(roundUp(high + pad), triangle.high[a]);
        }
        const ClipPlane last = path[depth - 1u];
        if (last.low) box.high[last.axis] = lesser(box.high[last.axis], last.c);
        else box.low[last.axis] = greater(box.low[last.axis], last.c);
      }
    }
    if (!empty) {
      const double d[3] = {widen(box.high[0]) - widen(box.low[0]), widen(box.high[1]) - widen(box.low[1]),
                           widen(box.high[2]) - widen(box.low[2])};
      if (depth < kClipDepth && clipArea(box) * count > factorSum && greaterDouble(d[0], greaterDouble(d[1], d[2])) > 0.0) {
        const uint axis = d[0] >= d[1] && d[0] >= d[2] ? 0u : (d[1] >= d[2] ? 1u : 2u);
        path[depth].axis = axis;
        path[depth].c = narrow((widen(box.low[axis]) + widen(box.high[axis])) * 0.5);
        path[depth].low = true;
        pendingHigh[depth] = true;
        depth += 1u;
        continue;
      }
      if (EMIT) {
        BvhReference reference;
        reference.low = float4(asfloat(box.low[0]), asfloat(box.low[1]), asfloat(box.low[2]), asfloat(t));
        reference.high = float4(asfloat(box.high[0]), asfloat(box.high[1]), asfloat(box.high[2]), 0.0f);
        references[first + made] = reference;
      }
      made += 1u;
    }
    while (depth > 0u && !pendingHigh[depth - 1u]) depth -= 1u;
    if (depth == 0u) break;
    pendingHigh[depth - 1u] = false;
    path[depth - 1u].low = false;
  }
  return made;
}

// The instance whose triangles hold global triangle `triangle`: the last whose first is <= it.
uint instanceOf(uint triangle) {
  uint low = 0u, high = control.sizes.x;
  while (high - low > 1u) {
    const uint middle = (low + high) >> 1u;
    if (levels[middle].x <= triangle) low = middle;
    else high = middle;
  }
  return low;
}

// Per area block (up to 256 of a bottom level's triangles): their box areas summed in order.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_clip_area_blocks(uint3 thread: SV_DispatchThreadID) {
  if (thread.x >= control.sizes.z) return;
  const uint2 block = areaBlocks[thread.x];
  const uint end = min(levels[block.x].y, block.y + 256u);
  double sum = 0.0;
  for (uint t = block.y; t < end; ++t) {
    uint corner[3][3];
    corners(block.x, t, corner);
    sum += clipArea(triangleBox(corner));
  }
  blockSums[thread.x] = sum;
}

// Per bottom level: its blocks' sums in order, times the factor.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_clip_sums(uint3 thread: SV_DispatchThreadID) {
  if (thread.x >= control.sizes.x) return;
  const uint4 level = levels[thread.x];
  double sum = 0.0;
  for (uint b = 0u; b < level.w; ++b) sum += blockSums[level.z + b];
  factorSums[thread.x] = widen(asuint(control.factor.x)) * sum;
}

// The references a triangle makes, or writes (from its offset).
uint trianglePass<let EMIT : bool>(uint triangle, uint first) {
  const uint instance = instanceOf(triangle);
  const uint4 level = levels[instance];
  const uint t = triangle - level.x;
  uint corner[3][3];
  corners(instance, t, corner);
  return clipTriangle<EMIT>(corner, t, double(level.y), factorSums[instance], first);  // an integer: exact
}

// Per triangle: its references counted, and the count before it in its scan block.
[shader("compute")]
[numthreads(256, 1, 1)]
void bvh_clip_count(uint3 thread: SV_DispatchThreadID, uint lane: SV_GroupIndex, uint3 group: SV_GroupID) {
  const uint triangle = thread.x;
  const uint made = triangle < control.sizes.y ? trianglePass<false>(triangle, 0u) : 0u;
  scanWords[lane] = made;
  GroupMemoryBarrierWithGroupSync();
  for (uint offset = 1u; offset < kClipScan; offset <<= 1u) {
    const uint add = lane >= offset ? scanWords[lane - offset] : 0u;
    GroupMemoryBarrierWithGroupSync();
    scanWords[lane] += add;
    GroupMemoryBarrierWithGroupSync();
  }
  if (triangle < control.sizes.y) inBlock[triangle] = scanWords[lane] - made;
  if (lane == kClipScan - 1u) scanTotals[group.x] = scanWords[lane];
}

// One workgroup: the scan blocks' exclusive prefix, each thread a run of them.
[shader("compute")]
[numthreads(256, 1, 1)]
void bvh_clip_scan(uint lane: SV_GroupIndex) {
  const uint blocks = control.sizes.w;
  const uint run = (blocks + kClipScan - 1u) / kClipScan;
  const uint first = lane * run, end = min(blocks, first + run);
  uint sum = 0u;
  for (uint b = first; b < end; ++b) sum += scanTotals[b];
  scanWords[lane] = sum;
  GroupMemoryBarrierWithGroupSync();
  for (uint offset = 1u; offset < kClipScan; offset <<= 1u) {
    const uint add = lane >= offset ? scanWords[lane - offset] : 0u;
    GroupMemoryBarrierWithGroupSync();
    scanWords[lane] += add;
    GroupMemoryBarrierWithGroupSync();
  }
  uint running = scanWords[lane] - sum;
  for (uint b = first; b < end; ++b) {
    scanPrefix[b] = running;
    running += scanTotals[b];
  }
  if (lane == kClipScan - 1u) scanPrefix[blocks] = scanWords[lane];
}

// A global triangle's first reference (the total, past the last triangle).
uint referenceOffset(uint triangle) {
  if (triangle >= control.sizes.y) return scanPrefix[control.sizes.w];
  return scanPrefix[triangle / kClipScan] + inBlock[triangle];
}

// Per bottom level: its references' range.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_clip_instances(uint3 thread: SV_DispatchThreadID) {
  if (thread.x >= control.sizes.x) return;
  const uint4 level = levels[thread.x];
  const uint first = referenceOffset(level.x);
  ranges[thread.x] = uint2(first, referenceOffset(level.x + level.y) - first);
}

// Per triangle: its references, written at its offset.
[shader("compute")]
[numthreads(64, 1, 1)]
void bvh_clip_emit(uint3 thread: SV_DispatchThreadID) {
  if (thread.x >= control.sizes.y) return;
  trianglePass<true>(thread.x, referenceOffset(thread.x));
}
