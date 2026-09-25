// Hashes and low-discrepancy sequences, shared by the rasteriser and the path tracer.
module basalt_random;

// Wellons's lowbias32 integer hash, scaled to [0, 1] (float rounding can reach 1).
public float hashFloat(uint value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return float(value) * 2.3283064365386963e-10f;
}

// R2 low-discrepancy rotation for a frame, in exact 32-bit fixed point so it neither
// repeats nor loses precision over thousands of frames.
public float3 frameRotation(float frame) {
  const uint n = uint(frame);
  return float3(float(n * 3242174889u), float(n * 2447445413u), float(n * 2654435769u)) * 2.3283064365386963e-10f;
}

// Van der Corput in base 2: the bits reversed, scaled to [0, 1] as hashFloat.
public float radicalInverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10f;
}

public float2 hammersley(uint index, uint count) { return float2(float(index) / float(count), radicalInverse(index)); }

// PCG output permutation (Jarzynski and Olano 2020): a good 32-bit hash. Generic over the lane
// count, so four independent hashes are one vector operation. The hash and the stream below are
// force-inlined: small and called on every bounce, and the generated C++ would otherwise call them.
[ForceInline]
public vector<uint, N> pcgHash<let N : int>(vector<uint, N> value) {
  const vector<uint, N> state = value * 747796405u + 2891336453u;
  const vector<uint, N> word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

[ForceInline]
public uint pcgHash(uint value) {
  const uint state = value * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// The path tracer's sample stream: a pure function of pixel, sample, bounce and dimension,
// so nothing is carried between bounces or threads and a run is reproducible from its seed.
public uint pathSeed(uint x, uint y, uint sampleIndex, uint seed) {
  return pcgHash(x + pcgHash(y + pcgHash(sampleIndex + pcgHash(seed))));
}

public float pathRandom(uint pathSeedValue, uint bounce, uint dimension) {
  const uint bits = pcgHash(pathSeedValue ^ pcgHash(bounce * 64u + dimension));
  return float(bits >> 8u) * (1.0f / 16777216.0f);
}

// Four numbers of the path's sample stream, dimensions 4 * group to 4 * group + 3 of a bounce.
[ForceInline]
public float4 pathRandom4(uint pathSeedValue, uint bounce, uint group) {
  const uint4 lanes = uint4(bounce * 64u + group * 4u) + uint4(0u, 1u, 2u, 3u);
  const uint4 bits = pcgHash(pcgHash(lanes) ^ pathSeedValue);
  return float4(bits >> 8u) * (1.0f / 16777216.0f);
}
