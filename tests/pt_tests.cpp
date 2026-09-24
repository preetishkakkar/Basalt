// Path tracer tests that need no GPU: the shared headers compiled as C++ and the CPU
// backend, checked against answers derived independently of the tracer's own sampling.
// Each test prints PASS or FAIL; the process fails if any test does.
#include "pt/CpuTracer.h"
#include "pt/BvhVariants.h"
#include "pt/Tables.h"
#include "pt/WavefrontPlan.h"
#include "temporal_fixture.h"
#include "pt_fixture.h"
#include "pt/ImageFile.h"
#include "pt/EnvironmentSun.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace pt;

namespace {

struct Test {
  const char *name;
  std::function<std::string()> run;  // An empty string is a pass; anything else says why not.
};

std::vector<Test> &registry() {
  static std::vector<Test> tests;
  return tests;
}

struct Register {
  Register(const char *name, std::function<std::string()> run) { registry().push_back({name, std::move(run)}); }
};

#define PT_TEST(name)                                                                             \
  static std::string name();                                                                      \
  static Register register_##name(#name, name);                                                   \
  static std::string name()

std::string format(const char *pattern, double a, double b = 0.0, double c = 0.0) {
  char text[256];
  std::snprintf(text, sizeof(text), pattern, a, b, c);
  return text;
}

using pt_fixture::Builder;
using pt_fixture::Affine;
using pt_fixture::affine;
using pt_fixture::sharedPool;

double meanLuminance(const std::vector<float> &image, uint width, uint x0, uint y0, uint x1, uint y1) {
  double sum = 0.0;
  for (uint y = y0; y < y1; ++y)
    for (uint x = x0; x < x1; ++x) {
      const float *p = image.data() + (static_cast<std::size_t>(y) * width + x) * 3;
      sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    }
  return sum / static_cast<double>((x1 - x0) * (y1 - y0));
}

PtSurface flatSurface(float3 normal, float3 base, float metallic, float roughness) {
  PtSurface surface;
  surface.position = float3(0.0f);
  surface.geometricNormal = normal;
  surface.normal = normal;
  surface.baseColor = base;
  surface.emissive = float3(0.0f);
  surface.metallic = metallic;
  surface.roughness = roughness;
  ptClearV7Layers(surface, 1.5f, 1.0f);
  return surface;
}

// The directional albedo of a material seen along `view` against a surface facing +z:
// f cos integrated over the hemisphere by stratified uniform sampling, independent of the
// tracer's importance sampling.
double integratedAlbedo(float3 base, float metallic, float roughness, float3 view) {
  static const std::vector<float> table = buildSpecularAlbedoTable();
  const PtSurface surface = flatSurface(float3(0.0f, 0.0f, 1.0f), base, metallic, roughness);
  const PtBsdf bsdf = ptMakeBsdf(surface, view, table.data());
  const int n = 1 << 21;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(n), v = radicalInverse(static_cast<uint>(i));
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - u * u)), phi = 2.0f * kPi * v;
    const float3 light(sinTheta * std::cos(phi), sinTheta * std::sin(phi), u);
    float pdf = 0.0f;
    sum += ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, light, pdf));
  }
  return sum / n * 2.0 * kPi;
}

// The integral of f over the hemisphere on `side` (+1 above the surface, -1 below) by the
// midpoint rule on a grid whose pole is `axis`, with rows packed towards it (theta = pi t^2):
// lobes down to alpha 0.0025 centred there get ~100 rows. Independent of the production
// sampler; the axis only decides where the grid is dense.
double hemisphereIntegral(float3 axis, float side, uint n, const std::function<double(float3)> &f) {
  float3 tangent, bitangent;
  orthonormalBasis(axis, tangent, bitangent);
  std::vector<double> rows(n, 0.0);
  std::vector<std::thread> workers;
  const uint threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  for (uint t = 0; t < threads; ++t)
    workers.emplace_back([&, t] {
      for (uint i = t; i < n; i += threads) {
        const double u = (i + 0.5) / n, theta = 3.14159265358979323846 * u * u;
        const double rowWidth = 2.0 * 3.14159265358979323846 * u / n;  // d theta
        const double cosine = std::cos(theta), sine = std::sin(theta);
        double sum = 0.0;
        for (uint j = 0; j < 2u * n; ++j) {
          const double phi = 2.0 * 3.14159265358979323846 * (j + 0.5) / (2.0 * n);
          const float3 d = normalize(tangent * static_cast<float>(sine * std::cos(phi)) +
                                     bitangent * static_cast<float>(sine * std::sin(phi)) + axis * static_cast<float>(cosine));
          if (d.z * side > 0.0f) sum += f(d);
        }
        rows[i] = sum * sine * rowWidth * (3.14159265358979323846 / n);  // * d phi
      }
    });
  for (std::thread &worker : workers) worker.join();
  double total = 0.0;
  for (const double row : rows) total += row;
  return total;
}

PT_TEST(random_stream_is_uniform) {
  double sum = 0.0, sumSquares = 0.0;
  float lowest = 1.0f, highest = 0.0f;
  const int count = 1 << 20;
  for (int i = 0; i < count; i += 4) {
    const uint seed = pathSeed(static_cast<uint>(i % 1024), static_cast<uint>(i / 1024), 7u, 1u);
    const float4 values = pathRandom4(seed, static_cast<uint>(i % 5), static_cast<uint>(i % 3));
    for (const float value : {values.x, values.y, values.z, values.w}) {
      sum += value;
      sumSquares += static_cast<double>(value) * value;
      lowest = std::min(lowest, value);
      highest = std::max(highest, value);
    }
  }
  const double mean = sum / count, variance = sumSquares / count - mean * mean;
  if (lowest < 0.0f || highest >= 1.0f) return format("out of [0, 1): %g .. %g", lowest, highest);
  if (std::abs(mean - 0.5) > 2e-3) return format("mean %g, expected 0.5", mean);
  if (std::abs(variance - 1.0 / 12.0) > 1e-3) return format("variance %g, expected %g", variance, 1.0 / 12.0);
  return {};
}

PT_TEST(sample_indices_remain_exact_above_the_float_integer_limit) {
  PathUniforms first{};
  PathUniforms second{};
  first.counts.w = 16'777'216u;
  second.counts.w = 16'777'217u;
  if (first.counts.w == second.counts.w)
    return "adjacent sample indices aliased across the shared CPU/MSL uniform ABI";
  if (first.counts.w != 16'777'216u || second.counts.w != 16'777'217u)
    return "sample indices did not round-trip through the integer uniform lane";
  return {};
}

PT_TEST(equirectangular_mapping_round_trips) {
  const uint width = 1024, height = 512;
  for (uint y = 1; y < height; y += 37)
    for (uint x = 0; x < width; x += 53) {
      const float2 uv((static_cast<float>(x) + 0.5f) / width, (static_cast<float>(y) + 0.5f) / height);
      const float2 back = equirectangularUV(ptEquirectangularDirection(uv));
      const float du = std::abs(back.x - uv.x) * width, dv = std::abs(back.y - uv.y) * height;
      if (std::min(du, width - du) > 1e-2f || dv > 1e-2f) return format("texel off by %g, %g", du, dv);
    }
  return {};
}

PT_TEST(environment_distribution_matches_its_pdf) {
  Builder b;
  // A bright patch on a dim gradient: a distribution far from uniform.
  b.environment([](float3 d) {
    const float patch = dot(d, normalize(float3(0.3f, 0.6f, 0.5f))) > 0.97f ? 200.0f : 0.0f;
    return float3(0.1f + 0.4f * std::max(d.y, 0.0f) + patch);
  });
  // Each sample's pdf must be what the pdf function says for its direction.
  std::mt19937 random(7);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  for (int i = 0; i < 20000; ++i) {
    float pdf = 0.0f;
    const float3 d = ptSampleEnvironmentDirection(b.scene.distribution.data(), b.scene.distributionInfo,
                                                  float2(unit(random), unit(random)), pdf);
    const float expected = ptEnvironmentPdf(b.scene.distribution.data(), b.scene.distributionInfo, d);
    if (pdf > 0.0f && std::abs(pdf - expected) > 1e-3f * std::max(pdf, expected))
      return format("sample pdf %g, pdf function %g", pdf, expected);
  }
  // And it must integrate to one over the sphere.
  double integral = 0.0;
  const int n = 1 << 20;
  for (int i = 0; i < n; ++i) {
    const float z = 1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / n, r = std::sqrt(1.0f - z * z);
    const float phi = 2.0f * kPi * radicalInverse(static_cast<uint>(i));
    integral += ptEnvironmentPdf(b.scene.distribution.data(), b.scene.distributionInfo,
                                 float3(r * std::cos(phi), z, r * std::sin(phi)));
  }
  integral *= 4.0 * kPi / n;
  if (std::abs(integral - 1.0) > 5e-3) return format("pdf integrates to %g", integral);
  return {};
}

PT_TEST(specular_albedo_table_matches_brute_force) {
  // The table integrates by visible-normal sampling; here D * Vis * cos is integrated by
  // uniform sampling instead, where that is accurate: lobes wide enough for it.
  const std::vector<float> table = buildSpecularAlbedoTable();
  for (const float roughness : {0.4f, 0.7f, 1.0f})
    for (const float cosine : {0.1f, 0.3f, 0.6f, 1.0f}) {
      const float alpha = roughness * roughness;
      const float3 view(std::sqrt(1.0f - cosine * cosine), 0.0f, cosine);
      const int n = 1 << 20;
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(n), v = radicalInverse(static_cast<uint>(i));
        const float s = std::sqrt(1.0f - u * u), phi = 2.0f * kPi * v;
        const float3 light(s * std::cos(phi), s * std::sin(phi), u);
        sum += distributionGGX(normalize(view + light).z, alpha) * visibilitySmith(cosine, u, alpha) * u;
      }
      const double brute = sum / n * 2.0 * kPi;
      const double tabled = ptSpecularAlbedo(table.data(), cosine, roughness);
      if (std::abs(tabled - brute) > 0.005)
        return format("roughness %g, cos %g: table %g", roughness, cosine, tabled) + format(", brute force %g", brute);
    }
  for (float value : table)
    if (!(value > 0.0f) || value > 1.0005f) return format("albedo %g out of (0, 1]", value);
  if (table[kAlbedoTableSize - 1] < 0.999f) return format("a smooth surface head-on reflects %g", table[kAlbedoTableSize - 1]);
  return {};
}

// The BVH against brute force.

// An independent double-precision Moller-Trumbore oracle. It deliberately shares no
// ray preparation, edge functions or object-space transform code with ptIntersectTriangle.
bool oracleTriangle(float3 origin, float3 direction, const float3 *v, double limit, double &distance,
                    double *outU = nullptr, double *outV = nullptr) {
  auto sub = [](float3 a, float3 b) {
    return std::array<double, 3>{static_cast<double>(a.x) - b.x, static_cast<double>(a.y) - b.y,
                                 static_cast<double>(a.z) - b.z};
  };
  auto crossD = [](const std::array<double, 3> &a, const std::array<double, 3> &b) {
    return std::array<double, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                 a[0] * b[1] - a[1] * b[0]};
  };
  auto dotD = [](const std::array<double, 3> &a, const std::array<double, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const auto e1 = sub(v[1], v[0]), e2 = sub(v[2], v[0]);
  const std::array<double, 3> d{direction.x, direction.y, direction.z};
  const auto p = crossD(d, e2);
  const double determinant = dotD(e1, p);
  if (std::abs(determinant) < 1e-14) return false;
  const double inverse = 1.0 / determinant;
  const auto fromVertex = sub(origin, v[0]);
  const double u = dotD(fromVertex, p) * inverse;
  if (u < 0.0 || u > 1.0) return false;
  const auto q = crossD(fromVertex, e1);
  const double vWeight = dotD(d, q) * inverse;
  if (vWeight < 0.0 || u + vWeight > 1.0) return false;
  const double t = dotD(e2, q) * inverse;
  if (!(t > 0.0) || !(t < limit)) return false;
  distance = t;
  if (outU) *outU = u;
  if (outV) *outV = vWeight;
  return true;
}

PT_TEST(bvh_matches_brute_force) {
  Builder b;
  std::mt19937 random(11);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  // Three instances of random triangle soup under different transforms.
  const uint m = b.material(float3(0.5f), 0.0f, 0.5f);
  for (int instance = 0; instance < 3; ++instance) {
    std::vector<float3> positions, normals;
    std::vector<uint> triangles;
    for (uint t = 0; t < 300; ++t) {
      const float3 centre(unit(random), unit(random), unit(random));
      for (int k = 0; k < 3; ++k) {
        positions.push_back(centre + float3(unit(random), unit(random), unit(random)) * 0.15f);
        normals.push_back(float3(0.0f, 1.0f, 0.0f));
        triangles.push_back(t * 3 + static_cast<uint>(k));
      }
    }
    const float3 scale = instance == 0 ? float3(0.5f, 0.8f, 1.1f)
                         : instance == 1 ? float3(1.3f, 0.6f, 0.9f)
                                         : float3(-0.7f, 1.2f, 0.55f);
    b.mesh(positions, normals, triangles, m,
           affine(0.7f * static_cast<float>(instance), scale,
                  float3(static_cast<float>(instance) * 1.2f - 1.2f, 0.1f * static_cast<float>(instance), 0.0f)));
  }
  b.environment([](float3) { return float3(1.0f); }, 8, 4);
  b.finish();

  // World-space triangles for brute force.
  struct WorldTriangle {
    float3 v[3];
    uint instance, primitive;
  };
  std::vector<WorldTriangle> world;
  for (uint i = 0; i < b.scene.instances.size(); ++i) {
    const TraceInstance &inst = b.scene.instances[i];
    for (uint t = 0; t < b.scene.triangleCounts[i]; ++t) {
      WorldTriangle w;
      for (uint k = 0; k < 3; ++k) {
        const float *p = b.scene.vertices.data() +
                         static_cast<std::size_t>(b.scene.indices[inst.firstIndex + t * 3 + k] + inst.vertexOffset) *
                             kVertexFloats;
        w.v[k] = applyRows(inst.objectToWorld0, inst.objectToWorld1, inst.objectToWorld2, float3(p[0], p[1], p[2]));
      }
      w.instance = i;
      w.primitive = t;
      world.push_back(w);
    }
  }

  int mismatches = 0, explainedTies = 0, hits = 0, anyMismatch = 0;
  const int rays = 100000;
  for (int r = 0; r < rays; ++r) {
    const float3 origin = float3(unit(random), unit(random), unit(random)) * 3.0f;
    const float3 target = float3(unit(random) * 1.5f, unit(random) * 0.6f, unit(random) * 0.6f);
    const float3 direction = normalize(target - origin);
    const PtHit hit = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
                                 b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
                                 b.scene.textures, origin, direction, kPtInfinity, 7u, 0u, float2(-1.0f, 0.0f), 0u);
    const PtHit any = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
                                 b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
                                 b.scene.textures, origin, direction, kPtInfinity, 7u, 0u, float2(-1.0f, 0.0f), 1u);
    double best = static_cast<double>(kPtInfinity);
    uint bestInstance = 0, bestPrimitive = 0;
    for (const WorldTriangle &w : world) {
      double t = 0.0;
      if (oracleTriangle(origin, direction, w.v, best, t)) {
        best = t;
        bestInstance = w.instance;
        bestPrimitive = w.primitive;
      }
    }
    const bool found = best < static_cast<double>(kPtInfinity);
    if (found) ++hits;
    if (any.found != (found ? 1u : 0u)) ++anyMismatch;
    if ((hit.found != 0u) != found) {
      ++mismatches;
      continue;
    }
    if (!found) continue;
    const bool sameTriangle = hit.instance == bestInstance && hit.primitive == bestPrimitive;
    const double distanceError = std::abs(static_cast<double>(hit.t) - best);
    // Object-space float traversal and world-space double intersection differ by a few
    // float ulps close to the ray origin. This absolute floor is fixed at roughly two
    // ulps near unit coordinates; farther hits use the relative bound.
    const double distanceTolerance = std::max(2e-7, 1e-4 * best);
    if (distanceError > distanceTolerance) {
      if (mismatches < 5)
        std::printf("  mismatch ray %d: oracle %u/%u t %.9g, BVH %u/%u t %.9g, error %.3g\n", r,
                    bestInstance, bestPrimitive, best, hit.instance, hit.primitive, hit.t, distanceError);
      ++mismatches;
    }
    else if (!sameTriangle) ++explainedTies;
  }
  if (hits < rays / 4) return format("only %g of the rays hit anything", hits);
  std::printf("  independent BVH oracle: %d rays, %d geometric ties\n", rays, explainedTies);
  if (mismatches != 0) return format("%g of %g rays disagree with the independent oracle", mismatches, rays);
  if (anyMismatch != 0) return format("any-hit disagrees on %g rays", anyMismatch);
  return {};
}

// Light transport against analytic answers.

PT_TEST(furnace_rough_metal_reflects_everything) {
  // A white metal sphere in a uniform white environment: with the energy compensation, it
  // must vanish into the background at every roughness, down to near-mirror lobes.
  for (const float roughness : {0.05f, 0.2f, 0.6f, 1.0f}) {
    Builder b;
    b.sphere(float3(0.0f), 1.0f, b.material(float3(1.0f), 1.0f, roughness));
    b.environment([](float3) { return float3(1.0f); }, 16, 8);
    b.camera(float3(0.0f, 0.0f, 4.0f), float3(0.0f), 0.6f, 48, 48);
    b.finish();
    const std::vector<float> image = b.render(256);
    const double centre = meanLuminance(image, 48, 16, 16, 32, 32);
    if (std::abs(centre - 1.0) > 0.01) return format("roughness %g: sphere at %g, expected 1", roughness, centre);
  }
  return {};
}

PT_TEST(furnace_glass_vanishes) {
  // White transmissive spheres in a uniform white environment neither absorb nor emit: with
  // the V7 energy compensation they vanish into the background, closed or thin-walled, smooth
  // or rough (the image-level counterpart of the white furnace integrals).
  struct Case { const char *name; float roughness; bool thin; };
  for (const Case c : {Case{"smooth glass", 0.05f, false}, Case{"rough glass", 0.5f, false},
                       Case{"thin-walled sheet", 0.3f, true}}) {
    Builder b;
    const uint glass = b.material(float3(1.0f), 0.0f, c.roughness);
    b.frame.materials[glass].transmission = float4(1.0f, 1.5f, c.thin ? 0.0f : 1.0f, 1.0f);
    b.sphere(float3(0.0f), 1.0f, glass);
    b.environment([](float3) { return float3(1.0f); }, 16, 8);
    b.camera(float3(0.0f, 0.0f, 4.0f), float3(0.0f), 0.6f, 48, 48);
    b.finish(32);
    const std::vector<float> image = b.render(1024);
    const double centre = meanLuminance(image, 48, 16, 16, 32, 32);
    if (std::abs(centre - 1.0) > 0.02) return std::string(c.name) + format(": sphere at %g, expected 1", centre);
  }
  return {};
}

PT_TEST(smooth_metal_albedo_integrates_to_one) {
  // Near-mirror lobes integrated on a grid centred on the mirror direction, independent of
  // the sampler: a white metal's compensated albedo is one down to the roughness floor. (The
  // image furnace above cannot see an error in D there: BSDF sampling dominates, and D
  // cancels in its value over pdf.)
  static const std::vector<float> table = buildSpecularAlbedoTable();
  for (const float roughness : {kMinRoughness, 0.05f, 0.08f})
    for (const float degrees : {0.0f, 45.0f}) {
      const float angle = degrees * kPi / 180.0f;
      const float3 view(std::sin(angle), 0.0f, std::cos(angle));
      const PtSurface surface = flatSurface(float3(0.0f, 0.0f, 1.0f), float3(1.0f), 1.0f, roughness);
      const PtBsdf bsdf = ptMakeBsdf(surface, view, table.data());
      const double albedo = hemisphereIntegral(float3(-view.x, -view.y, view.z), 1.0f, 2048u, [&](float3 light) {
        float pdf = 0.0f;
        return static_cast<double>(ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, light, pdf)));
      });
      if (std::abs(albedo - 1.0) > 0.005)
        return format("roughness %g at %g degrees: white metal albedo %g", roughness, degrees, albedo);
    }
  return {};
}

PT_TEST(lambertian_white_furnace_matches_the_analytic_integral) {
  // Independently integrate f*cos(theta) over the hemisphere with a uniform-solid-angle
  // sampler. For a white Lambertian f=1/pi, the answer is exactly one. This deliberately
  // does not call the production glTF BSDF when establishing the analytic oracle.
  constexpr uint samples = 1000000u;
  double estimate = 0.0;
  for (uint i = 0; i < samples; ++i) {
    const double cosine = (static_cast<double>(i) + 0.5) / samples;
    constexpr double brdf = 1.0 / 3.14159265358979323846;
    constexpr double pdf = 1.0 / (2.0 * 3.14159265358979323846);
    estimate += brdf * cosine / pdf;
  }
  estimate /= samples;
  if (std::abs(estimate - 1.0) > 0.005) return format("white furnace at %g, expected 1", estimate);
  return {};
}

PT_TEST(dielectric_albedo_matches_integration) {
  // A flat surface in a uniform environment of radiance one shows its directional albedo.
  Builder b;
  const float3 base(0.8f, 0.8f, 0.8f);
  b.wall(1.0f, 0.0f, b.material(base, 0.0f, 0.5f));
  b.environment([](float3) { return float3(1.0f); }, 16, 8);
  b.camera(float3(0.0f, 0.0f, 20.0f), float3(0.0f), 0.02f, 16, 16);  // nearly head-on everywhere
  b.finish();
  const std::vector<float> image = b.render(4096);
  const double rendered = meanLuminance(image, 16, 4, 4, 12, 12);
  const double expected = integratedAlbedo(base, 0.0f, 0.5f, float3(0.0f, 0.0f, 1.0f));
  if (std::abs(rendered - expected) > 0.01 * expected) return format("rendered %g, integrated %g", rendered, expected);
  return {};
}

PT_TEST(bsdf_sampling_mass_matches_the_reported_pdf) {
  const std::vector<float> table = buildSpecularAlbedoTable();
  constexpr uint samples = 500000u;
  for (const float roughness : {0.15f, 0.5f, 1.0f}) {
    const PtSurface surface = flatSurface(float3(0.0f, 0.0f, 1.0f), float3(0.7f, 0.4f, 0.2f),
                                          0.35f, roughness);
    const float3 view = normalize(float3(0.7f, 0.0f, 0.7141428f));
    const PtBsdf bsdf = ptMakeBsdf(surface, view, table.data());
    double integratedPdf = 0.0;
    uint validSamples = 0;
    for (uint i = 0; i < samples; ++i) {
      const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(samples);
      const float y = std::fmod((static_cast<float>(i) + 0.5f) * 0.7548776662466927f, 1.0f);
      const float z = std::fmod((static_cast<float>(i) + 0.5f) * 0.5698402909980532f, 1.0f);
      const float3 sampled = ptBsdfSampleDirection(bsdf, float3(x, y, z));
      float sampledPdf = 0.0f;
      ptBsdfEvaluate(bsdf, surface.geometricNormal, sampled, sampledPdf);
      validSamples += sampledPdf > 0.0f ? 1u : 0u;

      const float3 uniform = ptSampleCone(surface.normal, 0.0f, float2(x, y));
      float pdf = 0.0f;
      ptBsdfEvaluate(bsdf, surface.geometricNormal, uniform, pdf);
      integratedPdf += pdf * (2.0 * kPi);
    }
    const double sampledMass = static_cast<double>(validSamples) / samples;
    const double pdfMass = integratedPdf / samples;
    // The sharpest lobe has the largest quasi-Monte-Carlo integration error; 1.2%
    // bounds both integration and sampled mass without assuming the PDF integrates to one.
    if (std::abs(sampledMass - pdfMass) > 0.012)
      return format("roughness %g: sampler mass %g, integrated PDF %g", roughness, sampledMass, pdfMass);
  }
  return {};
}

// A camera straight above the origin looking down, x up the image.
void cameraAbove(Builder &b, float height, float halfAngle, uint size) {
  b.camera(float3(0.0f, height, 0.0f), float3(0.0f), 2.0f * halfAngle, size, size);
  b.frame.uniforms.cameraForward = float4(0.0f, -1.0f, 0.0f, 0.0f);
  b.frame.uniforms.cameraRight = float4(0.0f, 0.0f, std::tan(halfAngle), 0.0f);
  b.frame.uniforms.cameraUp = float4(std::tan(halfAngle), 0.0f, 0.0f, 0.0f);
}

PT_TEST(sun_on_a_plane_matches_the_brdf) {
  for (const float elevation : {0.3f, 0.8f, 1.4f}) {
    Builder b;
    const float3 base(0.5f);
    b.floor(10.0f, 0.0f, b.material(base, 0.0f, 1.0f));
    b.environment([](float3) { return float3(0.0f); }, 8, 4);
    const float3 sunDirection(std::cos(elevation), std::sin(elevation), 0.0f);
    cameraAbove(b, 5.0f, 0.025f, 8);
    b.sun(sunDirection, 3.0f, 0.005f);
    b.finish();
    const std::vector<float> image = b.render(1024);
    const double rendered = meanLuminance(image, 8, 0, 0, 8, 8);
    const PtSurface surface = flatSurface(float3(0.0f, 1.0f, 0.0f), base, 0.0f, 1.0f);
    const PtBsdf bsdf = ptMakeBsdf(surface, float3(0.0f, 1.0f, 0.0f), b.scene.specularAlbedo.data());
    float pdf = 0.0f;
    const double expected = ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, sunDirection, pdf)) * 3.0;
    if (std::abs(rendered - expected) > 0.01 * expected)
      return format("elevation %g: rendered %g, expected %g", elevation, rendered, expected);
  }
  return {};
}

PT_TEST(sun_cone_sampling_matches_independent_irradiance) {
  // For a cone wholly above the horizon, the vector solid-angle integral is
  // axis * pi*sin(radius)^2. This expected value is independent of the renderer,
  // its BSDF and ptSampleCone.
  constexpr uint samples = 1000000u;
  constexpr double radius = 0.02;
  constexpr double cosineMax = 0.9998000066665778; // cos(0.02), fixed independently
  constexpr double radiance = 3.0 / (2.0 * 3.14159265358979323846 * (1.0 - cosineMax));
  for (const double elevation : {0.3, 0.8, 1.4}) {
    const float3 axis(static_cast<float>(std::cos(elevation)), static_cast<float>(std::sin(elevation)), 0.0f);
    double estimate = 0.0;
    for (uint i = 0; i < samples; ++i) {
      // A two-dimensional irrational lattice avoids both RNG correlation and a
      // dependency on the production path hash.
      const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(samples);
      const float y = std::fmod((static_cast<float>(i) + 0.5f) * 0.6180339887498949f, 1.0f);
      estimate += std::max(0.0f, ptSampleCone(axis, static_cast<float>(cosineMax), float2(x, y)).y);
    }
    estimate = estimate / samples * radiance * (2.0 * 3.14159265358979323846 * (1.0 - cosineMax));
    const double expected = radiance * 3.14159265358979323846 * std::sin(radius) * std::sin(radius) *
                            std::sin(elevation);
    if (std::abs(estimate - expected) > 0.005 * expected)
      return format("elevation %g: sampled irradiance %g, analytic %g", elevation, estimate, expected);
  }
  return {};
}

PT_TEST(point_light_on_a_plane_matches_the_brdf) {
  Builder b;
  const float3 base(0.6f);
  b.floor(10.0f, 0.0f, b.material(base, 0.0f, 0.7f));
  b.environment([](float3) { return float3(0.0f); }, 8, 4);
  Light light{};
  light.position = float4(0.0f, 2.0f, 0.0f, 0.0f);  // range zero: pure inverse square
  light.color = float4(1.0f, 1.0f, 1.0f, 8.0f);
  light.direction = float4(0.0f, -1.0f, 0.0f, 1.0f);
  light.cone = float4(1.0f, 0.0f, 0.0f, 0.0f);
  b.frame.lights.push_back(light);
  cameraAbove(b, 5.0f, 0.002f, 4);
  b.finish();
  const std::vector<float> image = b.render(256);
  const double rendered = meanLuminance(image, 4, 0, 0, 4, 4);
  const PtSurface surface = flatSurface(float3(0.0f, 1.0f, 0.0f), base, 0.0f, 0.7f);
  const PtBsdf bsdf = ptMakeBsdf(surface, float3(0.0f, 1.0f, 0.0f), b.scene.specularAlbedo.data());
  float pdf = 0.0f;
  const double expected =
      ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, float3(0.0f, 1.0f, 0.0f), pdf)) * 8.0 / 4.0;
  if (std::abs(rendered - expected) > 0.01 * expected) return format("rendered %g, expected %g", rendered, expected);
  return {};
}

PT_TEST(emissive_triangle_metadata_tracks_world_area_and_power) {
  Builder b;
  const uint dim = b.emissiveMaterial(float3(1.0f));
  const uint bright = b.emissiveMaterial(float3(3.0f));
  const std::vector<float3> p{{-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, -1.0f},
                              {1.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}};
  const std::vector<float3> n(4, float3(0.0f, -1.0f, 0.0f));
  b.mesh(p, n, {0, 1, 2}, dim, affine(0.0f, float3(2.0f, 1.0f, 3.0f), float3(0.0f, 2.0f, 0.0f)));
  b.mesh(p, n, {0, 1, 2}, bright, affine(0.0f, float3(1.0f), float3(0.0f, 4.0f, 0.0f)));
  b.finish(1);
  if (b.scene.emissiveTriangles.size() != 2u) return "emissive triangles were omitted";
  const auto &large = b.scene.emissiveTriangles[0];
  const auto &brightTriangle = b.scene.emissiveTriangles[1];
  if (std::abs(large.v0Area.w - 12.0f) > 1e-5f || std::abs(brightTriangle.v0Area.w - 2.0f) > 1e-5f)
    return format("world areas are %g and %g, expected 12 and 2", large.v0Area.w, brightTriangle.v0Area.w);
  // Both are double-sided. Their selection weights are area*luminance, hence 12:6.
  if (std::abs(large.edge1Probability.w - 2.0f / 3.0f) > 1e-5f ||
      std::abs(brightTriangle.edge1Probability.w - 1.0f / 3.0f) > 1e-5f ||
      std::abs(large.edge2Cdf.w - 2.0f / 3.0f) > 1e-5f || brightTriangle.edge2Cdf.w != 1.0f)
    return "emissive power probabilities or CDF are incorrect";
  if (length(xyz(large.normal) - float3(0.0f, -1.0f, 0.0f)) > 1e-5f)
    return "emissive world normal is incorrect";
  return {};
}

PT_TEST(emissive_triangle_nee_handles_visibility_sidedness_and_mis) {
  auto estimate = [](uint strategy, bool blocked, bool backFacing, bool doubleSided, float scale,
                     uint seed, uint samples) {
    Builder b;
    b.floor(8.0f, 0.0f, b.material(float3(0.8f), 0.0f, 1.0f));
    const uint emitter = b.emissiveMaterial(float3(8.0f));
    if (backFacing) b.floor(2.0f * scale, 3.0f, emitter);
    else b.ceiling(2.0f * scale, 3.0f, emitter);
    if (!doubleSided) b.scene.instances.back().flags &= ~kInstanceDoubleSided;
    if (blocked) b.ceiling(1.5f, 1.0f, b.material(float3(0.0f), 0.0f, 1.0f));
    b.environment([](float3) { return float3(0.0f); }, 8, 4);
    b.camera(float3(0.0f, 1.0f, 5.0f), float3(0.0f), 0.01f, 1, 1);
    b.finish(2, strategy, seed);
    const auto image = b.render(samples);
    return static_cast<double>(ptLuminance(float3(image[0], image[1], image[2])));
  };

  const double light = estimate(2, false, false, false, 1.0f, 31u, 32768);
  const double mis = estimate(0, false, false, false, 1.0f, 31u, 32768);
  const double bsdf = estimate(1, false, false, false, 1.0f, 31u, 131072);
  if (!(light > 0.1)) return "next-event estimation did not illuminate the receiver";
  if (std::abs(mis - light) / light > 0.025 || std::abs(bsdf - light) / light > 0.04)
    return format("emitter strategies disagree: light %g, MIS %g, BSDF %g", light, mis, bsdf);
  const double occluded = estimate(2, true, false, false, 1.0f, 32u, 8192);
  if (occluded > light * 0.01) return format("occluded emitter retained %g of unoccluded light", occluded / light);
  const double back = estimate(2, false, true, false, 1.0f, 33u, 8192);
  const double twoSided = estimate(2, false, true, true, 1.0f, 33u, 8192);
  if (back > light * 0.01 || twoSided < light * 0.9)
    return format("emitter sidedness is incorrect: back %g, two-sided %g, front %g", back, twoSided, light);
  const double small = estimate(2, false, false, false, 0.25f, 34u, 32768);
  if (!(small > 0.0 && small < light * 0.25))
    return format("emitter scale did not reduce irradiance: small %g, full %g", small, light);
  return {};
}

PT_TEST(strategies_converge_to_the_same_image) {
  // Light sampling alone, BSDF sampling alone and their combination estimate one integral.
  auto renderWith = [](uint strategy) {
    Builder b;
    b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.9f, 0.6f, 0.3f), 0.0f, 0.4f));
    b.floor(6.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.8f));
    b.environment([](float3 d) {
      const float bright = dot(d, normalize(float3(0.4f, 0.8f, 0.3f))) > 0.95f ? 40.0f : 0.0f;
      return float3(0.2f + 0.5f * std::max(d.y, 0.0f) + bright);
    });
    b.camera(float3(0.0f, 2.5f, 6.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 32, 32);
    b.finish(8, strategy);
    return b.render(4096);
  };
  const std::vector<float> mis = renderWith(0), bsdf = renderWith(1), light = renderWith(2);
  double worst = 0.0;
  std::string where;
  for (uint by = 0; by < 4; ++by)
    for (uint bx = 0; bx < 4; ++bx) {
      const double a = meanLuminance(mis, 32, bx * 8, by * 8, bx * 8 + 8, by * 8 + 8);
      const double c = meanLuminance(bsdf, 32, bx * 8, by * 8, bx * 8 + 8, by * 8 + 8);
      const double d = meanLuminance(light, 32, bx * 8, by * 8, bx * 8 + 8, by * 8 + 8);
      const double difference = std::max(std::abs(a - c), std::abs(a - d)) / std::max(a, 1e-3);
      if (difference > worst) {
        worst = difference;
        where = format("block (%g, %g): ", bx, by) + format("MIS %g, BSDF %g, ", a, c) + format("light %g", d);
      }
    }
  if (worst > 0.03) return format("an 8 x 8 block differs by %g between strategies, ", worst) + where;
  return {};
}

PT_TEST(mis_reduces_error_on_the_bright_environment_fixture) {
  auto renderWith = [](uint strategy, uint seed, uint samples) {
    Builder b;
    b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.9f, 0.6f, 0.3f), 0.0f, 0.35f));
    b.floor(6.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.8f));
    b.environment([](float3 d) {
      const float bright = dot(d, normalize(float3(0.4f, 0.8f, 0.3f))) > 0.985f ? 120.0f : 0.0f;
      return float3(0.1f + 0.25f * std::max(d.y, 0.0f) + bright);
    }, 128, 64);
    b.camera(float3(0.0f, 2.5f, 6.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 24, 24);
    b.finish(8, strategy, seed);
    return b.render(samples);
  };
  const std::vector<float> reference = renderWith(0, 91u, 8192);
  double mse[3]{};
  for (uint strategy = 0; strategy < 3; ++strategy)
    for (uint seed : {101u, 202u, 303u, 404u}) {
      const std::vector<float> image = renderWith(strategy, seed, 128);
      for (std::size_t i = 0; i < image.size(); ++i) {
        const double difference = static_cast<double>(image[i]) - reference[i];
        mse[strategy] += difference * difference;
      }
    }
  for (double &value : mse) value /= 4.0 * static_cast<double>(reference.size());
  std::printf("  bright-environment MSE: MIS %.6g, BSDF %.6g, light %.6g\n", mse[0], mse[1], mse[2]);
  if (!(mse[0] < mse[1] && mse[0] < mse[2]))
    return format("MIS MSE %g was not below BSDF %g and light %g", mse[0], mse[1], mse[2]);
  return {};
}

PT_TEST(russian_roulette_preserves_expected_energy) {
  auto estimate = [](uint rouletteStart, uint seed) {
    Builder b;
    b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.55f), 0.0f, 0.65f));
    b.floor(5.0f, 0.0f, b.material(float3(0.6f), 0.0f, 0.8f));
    b.environment([](float3 d) { return float3(0.15f + 0.85f * std::max(d.y, 0.0f)); }, 64, 32);
    b.camera(float3(0.0f, 2.2f, 5.0f), float3(0.0f, 0.7f, 0.0f), 0.8f, 24, 24);
    b.finish(10, 0, seed);
    b.frame.uniforms.path.y = static_cast<float>(rouletteStart);
    return meanLuminance(b.render(512), 24, 0, 0, 24, 24);
  };
  double roulette = 0.0, disabled = 0.0;
  for (uint seed : {17u, 29u, 41u, 53u}) {
    roulette += estimate(2, seed);
    disabled += estimate(100, seed);
  }
  roulette *= 0.25;
  disabled *= 0.25;
  const double error = std::abs(roulette - disabled) / std::max(disabled, 1e-6);
  if (error > 0.015) return format("roulette energy %g differs from disabled %g", roulette, disabled);
  return {};
}

PT_TEST(blended_surfaces_pass_light_by_their_alpha) {
  // A black blended wall in front of a white environment: 1 - alpha of the light passes;
  // alpha of it sees the wall's own reflection.
  for (const float alpha : {0.25f, 0.75f}) {
    Builder b;
    b.wall(1.0f, 0.0f, b.material(float3(0.0f), 0.0f, 0.5f, alpha, 2), kInstanceBlended);
    b.scene.instances.back().mask = kRayMaskBlended;
    b.environment([](float3) { return float3(1.0f); }, 16, 8);
    b.camera(float3(0.0f, 0.0f, 20.0f), float3(0.0f), 0.02f, 8, 8);
    b.finish();
    const std::vector<float> image = b.render(4096);
    const double rendered = meanLuminance(image, 8, 0, 0, 8, 8);
    const double reflected = integratedAlbedo(float3(0.0f), 0.0f, 0.5f, float3(0.0f, 0.0f, 1.0f));
    const double expected = (1.0 - alpha) + alpha * reflected;
    if (std::abs(rendered - expected) > 0.015)
      return format("alpha %g: rendered %g, expected %g", alpha, rendered, expected);
  }
  return {};
}

PT_TEST(masked_surfaces_cut_by_their_cutoff) {
  for (const float alpha : {0.3f, 0.7f}) {
    Builder b;
    b.wall(1.0f, 0.0f, b.material(float3(0.0f), 0.0f, 0.5f, alpha, 1), kInstanceMasked);
    b.environment([](float3) { return float3(1.0f); }, 16, 8);
    b.camera(float3(0.0f, 0.0f, 20.0f), float3(0.0f), 0.02f, 8, 8);
    b.finish();
    const std::vector<float> image = b.render(512);
    const double rendered = meanLuminance(image, 8, 0, 0, 8, 8);
    const double reflected = integratedAlbedo(float3(0.0f), 0.0f, 0.5f, float3(0.0f, 0.0f, 1.0f));
    const double expected = alpha < 0.5f ? 1.0 : reflected;  // the cutoff is 0.5
    if (std::abs(rendered - expected) > 0.01)
      return format("alpha %g: rendered %g, expected %g", alpha, rendered, expected);
  }
  return {};
}

PT_TEST(a_seed_reproduces_its_image) {
  auto renderSeed = [](uint seed) {
    Builder b;
    b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.7f), 0.3f, 0.3f));
    b.floor(4.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.9f));
    b.environment([](float3 d) { return float3(0.3f + std::max(d.y, 0.0f)); });
    b.camera(float3(0.0f, 2.0f, 5.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 24, 24);
    b.finish(8, 0, seed);
    return b.render(4);
  };
  // These adjacent seeds collapse to one value if they pass through a float.
  constexpr uint seed = 16777216u;
  const std::vector<float> a = renderSeed(seed), again = renderSeed(seed), other = renderSeed(seed + 1u);
  if (a != again) return "the same seed gave two images";
  if (a == other) return "two seeds gave the same image";
  return {};
}

PT_TEST(thread_count_reproduces_the_same_image) {
  Builder b;
  b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.6f), 0.2f, 0.4f));
  b.floor(3.0f, 0.0f, b.material(float3(0.4f), 0.0f, 0.8f));
  b.environment([](float3 d) { return float3(0.2f + 0.5f * std::max(d.y, 0.0f)); }, 32, 16);
  b.camera(float3(0.0f, 1.5f, 4.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 31, 23);
  b.finish(6, 0, 0xf1234567u);
  WorkerPool one(1), four(4);
  const auto a = CpuTracer::render(b.scene, b.frame, 8, one);
  const auto c = CpuTracer::render(b.scene, b.frame, 8, four);
  if (a.color != c.color || a.albedo != c.albedo || a.normal != c.normal)
    return "changing CPU worker count changed a seeded render";
  return {};
}

PT_TEST(traced_guides_preserve_raw_paths_and_split_radiance) {
  Builder b;
  b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.6f), 0.2f, 0.4f));
  b.floor(3.0f, 0.0f, b.material(float3(0.4f), 0.0f, 0.8f));
  b.environment([](float3 d) { return float3(0.2f + 0.5f * std::max(d.y, 0.0f)); }, 32, 16);
  b.camera(float3(0.0f, 1.5f, 4.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 24, 18);
  b.finish(6, 0, 11);
  uint surfaces = 0, misses = 0;
  for (uint s = 0; s < 4; ++s) for (uint y = 0; y < 18; ++y) for (uint x = 0; x < 24; ++x) {
    b.frame.uniforms.image.z = 0.0f;
    const auto raw = CpuTracer::tracePixel(b.scene, b.frame, x, y, s);
    b.frame.uniforms.image.z = 1.0f;
    const auto guided = CpuTracer::tracePixel(b.scene, b.frame, x, y, s);
    if (std::memcmp(&raw.radiance, &guided.radiance, sizeof(float3)) != 0)
      return "collecting guides changed raw radiance";
    if (length(xyz(guided.guide.diffuse) + xyz(guided.guide.specular) - raw.radiance) >
        1e-5f * std::max(1.0f, length(raw.radiance))) return "component radiance does not sum to raw";
    if (guided.guide.identity.z == 0) { ++misses; continue; }
    ++surfaces;
    const auto &g = guided.guide;
    if (!(g.positionDepth.w > 0.0f) || std::abs(length(xyz(g.geometricNormal)) - 1.0f) > 1e-4f ||
        std::abs(length(xyz(g.normalRoughness)) - 1.0f) > 1e-4f)
      return "surface guide has invalid depth or normals";
    const float3 relative = xyz(g.positionDepth) - xyz(b.frame.uniforms.cameraPosition);
    if (length(normalize(relative) - xyz(g.viewDistance)) > 1e-4f)
      return "guide does not lie on the jittered primary ray";
    if (g.identity.x >= b.scene.instances.size() ||
        g.identity.y != b.scene.instances[g.identity.x].material) return "guide identity is incorrect";
  }
  if (surfaces == 0 || misses == 0) return "fixture did not cover both hits and misses";
  return {};
}

// Transmissive and mirror-coated primaries keep diffuse history but reject
// specular history; with depth of field, history is rejected where the lens moves the primary
// hit by more than a pixel, and kept where it does not (the focal plane, a small aperture).
PT_TEST(temporal_history_rejects_v7_layers_and_defocus) {
  constexpr uint w = 17, h = 11;
  PtTemporalUniforms u{};
  u.previousCamera = temporal_fixture::camera(w, h);
  u.image = uint4(w, h, 1, 1);
  const auto samples = temporal_fixture::plane(w, h, 0), previous = samples;
  std::vector<PtTemporalHistory> history(w * h);
  history = temporal_fixture::temporal(u, samples, previous, history);
  u.image.z = 0;
  const uint at = 5 * w + 8;
  auto reuse = [&](const PtReconstructionSample &replacement, const PtTemporalUniforms &uniforms) {
    auto changed = samples; changed[at] = replacement;
    return temporal_fixture::temporal(uniforms, changed, previous, history)[at];
  };
  if (ptGuideLayers(float4(0.0f, 1.5f, 1.0f, 1.0f), float4(0.0f, kMinRoughness, 0.0f, 0.0f)) != 0.0f)
    return "a surface without V7 layers was flagged";
  if (ptGuideLayers(float4(0.5f, 1.5f, 0.0f, 1.0f), float4(0.0f)) != 1.0f) return "transmission was not flagged";
  if (ptGuideLayers(float4(0.0f, 1.5f, 1.0f, 1.0f), float4(1.0f, 0.05f, 0.0f, 0.0f)) != 2.0f)
    return "a mirror-like clearcoat was not flagged";
  if (ptGuideLayers(float4(0.0f, 1.5f, 1.0f, 1.0f), float4(1.0f, 0.5f, 0.0f, 0.0f)) != 0.0f)
    return "a rough clearcoat was flagged";
  for (const float layers : {1.0f, 2.0f, 3.0f}) {
    auto s = samples[at];
    s.geometricNormal.w = layers;
    const auto h1 = reuse(s, u);
    if (h1.specular.w != 1) return format("layer flags %g retained specular history", layers);
    if (h1.diffuse.w != 2) return format("layer flags %g lost diffuse history", layers);
  }
  const float depth = samples[at].positionDepth.w;
  PtTemporalUniforms defocused = u;
  defocused.previousCamera.lens = float4(0.2f, depth * 0.5f, 0.0f, 0.0f);  // far behind focus
  if (reuse(samples[at], defocused).diffuse.w != 1) return "a defocused primary retained history";
  PtTemporalUniforms focused = u;
  focused.previousCamera.lens = float4(0.2f, depth, 0.0f, 0.0f);  // on the focal plane
  if (reuse(samples[at], focused).diffuse.w != 2) return "a primary on the focal plane lost history";
  PtTemporalUniforms tiny = u;
  tiny.previousCamera.lens = float4(1e-5f, depth * 0.5f, 0.0f, 0.0f);  // blur far below a pixel
  if (reuse(samples[at], tiny).diffuse.w != 2) return "a sub-pixel circle of confusion lost history";
  return {};
}

PT_TEST(temporal_history_rejects_disocclusion_and_ambiguous_correspondence) {
  constexpr uint w = 17, h = 11;
  PtTemporalUniforms u{};
  u.previousCamera = temporal_fixture::camera(w, h);
  u.image = uint4(w, h, 1, 1);
  auto samples = temporal_fixture::plane(w, h, 0), previous = samples;
  std::vector<PtTemporalHistory> history(w * h);
  history = temporal_fixture::temporal(u, samples, previous, history);
  u.image.z = 0;
  auto reused = temporal_fixture::temporal(u, samples, previous, history);
  for (const auto &p : reused) if (p.diffuse.w != 2 || p.specular.w != 2)
    return "a stationary compatible plane did not retain history";
  const uint at = 5 * w + 8;
  auto rejected = [&](const PtReconstructionSample &replacement) {
    auto changed = samples; changed[at] = replacement;
    return temporal_fixture::temporal(u, changed, previous, history)[at];
  };
  auto s = samples[at]; s.identity.x = 9;
  if (rejected(s).diffuse.w != 1) return "disoccluded identity retained history";
  s = samples[at]; s.identity.z = 2;
  if (rejected(s).diffuse.w != 1) return "blended surface retained history";
  s = samples[at]; s.identity.z = 0;
  if (rejected(s).diffuse.w != 1) return "miss retained surface history";
  s = samples[at]; s.positionDepth.z = 1;
  if (rejected(s).diffuse.w != 1) return "incompatible depth retained history";
  s = samples[at]; s.normalRoughness = float4(0, 1, 0, 0.6f);
  if (rejected(s).diffuse.w != 1) return "incompatible normal retained history";
  s = samples[at]; s.normalRoughness.w = 0.01f;
  if (rejected(s).specular.w != 1) return "mirror correspondence reused primary-surface history";
  s = samples[at]; s.viewDistance.w = 4;
  if (rejected(s).specular.w != 1) return "secondary hit/miss change retained history";
  for (auto &p : samples) { p.diffuse = float4(0); p.specular = float4(0); }
  u.image.z = 1;
  const auto black = temporal_fixture::filter(u, samples, temporal_fixture::temporal(u, samples, previous, history));
  for (const auto &p : black) if (length(xyz(p.diffuse)) + length(xyz(p.specular)) > 1e-6f || p.diffuse.w != 1)
    return "reset retained old bright history";
  // A newly visible black surface changes identity without a global reset. Even its
  // neighboring old bright pixels must not bleed across the new surface boundary.
  u.image.z = 0;
  for (auto &p : samples) p.identity.x = 91;
  auto disoccluded = temporal_fixture::temporal(u, samples, previous, history);
  for (uint frame = 0; frame < 3; ++frame) {
    const auto filtered = temporal_fixture::filter(u, samples, disoccluded);
    for (const auto &p : filtered) if (length(xyz(p.diffuse)) + length(xyz(p.specular)) > 1e-6f)
      return "old bright history survived disocclusion";
    disoccluded = temporal_fixture::temporal(u, samples, samples, disoccluded);
  }
  samples[at].diffuse = float4(std::numeric_limits<float>::infinity());
  samples[at + 1].specular = float4(1e30f);
  const auto sanitized = temporal_fixture::filter(u, samples, temporal_fixture::temporal(u, samples, samples, disoccluded));
  for (const auto &p : sanitized)
    if (!ptFiniteColor(xyz(p.diffuse)) || !ptFiniteColor(xyz(p.specular)) ||
        !std::isfinite(p.moments.y) || !std::isfinite(p.moments.w)) return "nonfinite reconstruction signal/moments";
  return {};
}

PT_TEST(material_sampler_modes_match_gltf_level_zero_rules) {
  HostTextures maps;
  HostTexture texture;
  texture.width = 2;
  texture.height = 2;
  texture.srgb = false;
  texture.texels = {0xFF0000FFu, 0xFF00FF00u, 0xFFFF0000u, 0xFFFFFFFFu};
  maps.textures.push_back(texture);
  maps.slots[2] = 2u;
  const float4 repeat = maps.sample(2u, float2(1.25f, 0.25f), 16u);
  const float4 clamp = maps.sample(2u, float2(1.25f, 0.25f), 16u | 1u | (1u << 2u));
  const float4 mirror = maps.sample(2u, float2(1.25f, 0.25f), 16u | 2u | (2u << 2u));
  if (repeat.x < 0.99f || repeat.y > 0.01f) return "repeat did not wrap to the first texel";
  if (clamp.y < 0.99f || clamp.x > 0.01f) return "clamp did not select the edge texel";
  if (mirror.y < 0.99f || mirror.x > 0.01f) return "mirrored repeat did not reflect the coordinate";
  const float4 linear = maps.sample(2u, float2(0.5f, 0.25f), 0u);
  if (std::abs(linear.x - 0.5f) > 1e-6f || std::abs(linear.y - 0.5f) > 1e-6f)
    return "linear filtering did not interpolate adjacent texels";
  return {};
}

PT_TEST(material_uv_vertex_colour_and_sidedness_are_shared) {
  Builder b;
  const uint material = b.material(float3(1.0f), 0.0f, 0.5f);
  b.mesh({float3(-1.0f, -1.0f, 0.0f), float3(1.0f, -1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f)},
         {float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 1.0f)},
         {0u, 1u, 2u}, material, affine(0.0f, 1.0f, float3(0.0f)));
  HostTexture texture;
  texture.width = 2;
  texture.height = 1;
  texture.srgb = false;
  texture.texels = {0xFFFFFFFFu, 0xFF0000FFu};
  b.scene.textures.textures.push_back(texture);
  b.scene.textures.slots[2] = 2u;
  b.scene.instances[0].slots = 2u | (1u << 24u);
  b.frame.materials[material].texture.x = 1u;
  b.frame.materials[material].texture.y = 16u;
  for (uint vertex = 0u; vertex < 3u; ++vertex) {
    float *v = b.scene.vertices.data() + vertex * kVertexFloats;
    v[10] = 0.25f; v[11] = 0.5f;
    v[12] = 0.75f; v[13] = 0.5f;
    v[14] = 0.5f; v[15] = 0.25f; v[16] = 1.0f; v[17] = 1.0f;
  }
  b.finish(1u);
  const PtHit front = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(),
      b.scene.instances.data(), b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
      b.scene.textures, float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, -1.0f), 10.0f,
      kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u);
  if (front.found == 0u) return "the front face was not accepted";
  const PtSurface surface = ptSurfaceAt(b.scene.instances.data(), b.frame.materials.data(), b.scene.indices.data(),
      b.scene.vertices.data(), b.scene.textures, front, float3(0.0f, 0.0f, -1.0f), -1.0f);
  if (std::abs(surface.baseColor.x - 0.5f) > 1e-5f || surface.baseColor.y > 1e-5f ||
      surface.baseColor.z > 1e-5f)
    return "UV1 selection or vertex-colour modulation did not reach the shared surface";
  b.scene.instances[0].flags &= ~kInstanceDoubleSided;
  const PtHit back = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(),
      b.scene.instances.data(), b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
      b.scene.textures, float3(0.0f, 0.0f, -1.0f), float3(0.0f, 0.0f, 1.0f), 10.0f,
      kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u);
  if (back.found != 0u) return "a one-sided material accepted a back-face ray";
  return {};
}

// A rejected masked candidate tested after the accepted hit must not leave its facing
// bit in PtHit::ambiguous: only bit 0 (a passed-through blend) is part of the contract,
// and temporal reconstruction reads any nonzero value as ambiguous coverage.
PT_TEST(rejected_mask_candidates_leave_no_coverage_bits) {
  for (const bool reversed : {false, true}) {
    Builder b;
    // Tilted so its bounds contain the ray origin and are visited first, but hit at t = 3.
    const std::vector<float3> far{{-10, -10, -6}, {10, -10, 4}, {10, 10, 4}, {-10, 10, -6}};
    b.mesh(far, std::vector<float3>(4, float3(0, 0, 1)), {0, 1, 2, 0, 2, 3},
           b.material(float3(0.5f), 0, 0.8f), affine(0.0f, 1.0f, float3(0.0f)));
    // A fully transparent masked square in front, at t = 1, tested second.
    const std::vector<float3> near{{-0.5f, -0.5f, 1}, {0.5f, -0.5f, 1}, {0.5f, 0.5f, 1}, {-0.5f, 0.5f, 1}};
    const std::vector<uint> winding = reversed ? std::vector<uint>{0, 2, 1, 0, 3, 2} : std::vector<uint>{0, 1, 2, 0, 2, 3};
    b.mesh(near, std::vector<float3>(4, float3(0, 0, 1)), winding, b.material(float3(1), 0, 0.8f, 0.0f, 1),
           affine(0.0f, 1.0f, float3(0.0f)), kRayMaskScene, kInstanceMasked);
    b.finish(1u);
    const float3 origin(0.0f, 0.0f, 2.0f), direction(0.0f, 0.0f, -1.0f);
    std::vector<std::pair<const char *, PtHit>> hits;
    hits.emplace_back("binary", ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(),
        b.scene.instances.data(), b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
        b.scene.textures, origin, direction, kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u));
    const WideBvh wide4 = buildWideBvh(b.scene.bvh, b.scene.instances, 4u);
    const WideBvh wide8 = buildWideBvh(b.scene.bvh, b.scene.instances, 8u);
    hits.emplace_back("bvh4", traceWideBvh(wide4, b.frame.materials.data(), b.scene.indices.data(),
        b.scene.vertices.data(), b.scene.textures, origin, direction, kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u));
    hits.emplace_back("bvh8", traceWideBvh(wide8, b.frame.materials.data(), b.scene.indices.data(),
        b.scene.vertices.data(), b.scene.textures, origin, direction, kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u));
    if (cpuAvx2Available())
      hits.emplace_back("avx2", traceWideBvhAvx2(wide8, b.frame.materials.data(), b.scene.indices.data(),
          b.scene.vertices.data(), b.scene.textures, origin, direction, kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u));
    for (const auto &[name, hit] : hits) {
      if (hit.found == 0u || hit.instance != 0u || std::abs(hit.t - 3.0f) > 1e-4f)
        return std::string(name) + " did not return the opaque surface behind the rejected mask";
      if (hit.ambiguous != 0u)
        return std::string(name) + " leaked candidate bits " + std::to_string(hit.ambiguous) +
               " from a rejected opaque-coverage mask";
    }
  }
  return {};
}

// One-sided mirrored instances: every CPU intersector must agree with the binary BVH on
// which side is the front after a negative-determinant transform. Every backend, the raster
// and both ray oracles follow glTF 2.0: the object-space front, here +z.
PT_TEST(mirrored_one_sided_instances_agree_across_cpu_intersectors) {
  if (!EmbreeScene::available()) return "SKIP: Embree is not in this build";
  Builder b;
  const std::vector<float3> quad{{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  const Affine mirrored = affine(0.0f, float3(-1.0f, 1.0f, 1.0f), float3(0.0f));
  b.mesh(quad, std::vector<float3>(4, float3(0, 0, 1)), {0, 1, 2, 0, 2, 3}, b.material(float3(0.5f), 0, 0.8f),
         mirrored);
  b.scene.instances[0].flags &= ~kInstanceDoubleSided;
  if ((b.scene.instances[0].flags & kInstanceMirrored) == 0u) return "the fixture transform is not mirrored";
  b.finish(1u);
  const EmbreeScene embree(b.scene);
  uint accepted = 0;
  for (const float side : {1.0f, -1.0f}) {
    const float3 origin(0.1f, 0.2f, 2.0f * side), direction(0.0f, 0.0f, -side);
    const PtHit own = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures, origin,
        direction, kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u);
    const PtHit other = embree.trace(b.scene, b.frame, origin, direction, kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u);
    if (own.found != other.found)
      return std::string("Embree ") + (other.found ? "accepted" : "rejected") + " the " +
             (side > 0 ? "+z" : "-z") + " side of a one-sided mirrored instance; the own BVH did not";
    accepted += own.found;
  }
  if (accepted != 1u) return "exactly one side of a one-sided quad must be visible";
  // glTF 2.0: the mirror (x -> -x) keeps the object-space front, +z, although its world winding
  // is now clockwise seen from there.
  const PtHit fromAbove = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
      b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
      float3(0.1f, 0.2f, 2.0f), float3(0.0f, 0.0f, -1.0f), kPtInfinity, kRayMaskScene, 0u, float2(-1.0f, 0.0f), 0u);
  if (fromAbove.found == 0u) return "the object-space front (+z) of a mirrored instance is not its front";
  return {};
}

PT_TEST(rejected_blend_candidates_invalidate_reconstruction_coverage) {
  Builder b;
  b.wall(10, 1, b.material(float3(1), 0, 0.8f, 0, 2), kInstanceBlended);
  b.wall(10, 0, b.material(float3(0.5f), 0, 0.8f));
  b.camera(float3(0, 0, 2), float3(0), 0.8f, 8, 8);
  b.finish(2, 0, 11);
  b.frame.uniforms.image.z = 1;
  const auto own = CpuTracer::tracePixel(b.scene, b.frame, 4, 4, 0);
  if (own.guide.identity.x != 1 || own.guide.identity.z != 2)
    return "a rejected transparent foreground allowed opaque-background history reuse";
  if (EmbreeScene::available()) {
    b.scene.embree = std::make_shared<EmbreeScene>(b.scene);
    b.frame.intersector = 1;
    const auto embree = CpuTracer::tracePixel(b.scene, b.frame, 4, 4, 0);
    if (embree.guide.identity.x != 1 || embree.guide.identity.z != 2)
      return "Embree lost the rejected blended candidate's coverage ambiguity";
  }
  return {};
}

PT_TEST(restart_discards_stale_cpu_results) {
  Builder b;
  b.wall(1.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.8f));
  b.environment([](float3) { return float3(0.5f); }, 8, 4);
  b.camera(float3(0.0f, 0.0f, 2.0f), float3(0.0f), 0.8f, 256, 256);
  b.finish(4, 0, 3);
  auto scene = std::make_shared<CpuScene>(std::move(b.scene));
  CpuTracer tracer(4);
  tracer.setScene(scene);
  CpuFrame old = b.frame;
  old.targetSamples = 32;
  tracer.start(old);
  CpuFrame current = b.frame;
  current.width = current.height = 8;
  current.uniforms.image.x = current.uniforms.image.y = 8.0f;
  current.targetSamples = 1;
  tracer.start(current);

  CpuTracer::Image image;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    tracer.latest(image, image.version);
    if (image.width == 8 && image.height == 8 && image.samples == 1) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (image.width != 8 || image.height != 8 || image.samples != 1)
    return "the replacement frame did not complete before the lifecycle timeout";
  const std::uint64_t version = image.version;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CpuTracer::Image later;
  if (tracer.latest(later, version) && (later.width != 8 || later.height != 8))
    return "a stale frame published after its replacement";
  return {};
}

PT_TEST(publication_carries_its_camera_and_rejects_replaced_scenes) {
  Builder b;
  b.wall(1.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.8f));
  b.camera(float3(0.0f, 0.0f, 2.0f), float3(0.0f), 0.8f, 8, 8);
  b.finish(2, 0, 11);
  b.frame.targetSamples = 1;
  auto scene = std::make_shared<CpuScene>(std::move(b.scene));
  CpuTracer tracer(2);
  tracer.setScene(scene);
  tracer.start(b.frame);
  CpuTracer::Image image;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline && image.samples != 1) {
    tracer.latest(image, image.version);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (image.samples != 1 || image.generation == 0 ||
      image.uniforms.cameraPosition.z != b.frame.uniforms.cameraPosition.z)
    return "publication did not retain the camera/generation that produced it";
  CpuTracer::Image duplicate;
  if (tracer.latest(duplicate, image.version)) return "a repeated display advanced publication";
  tracer.setScene(scene);
  if (tracer.latest(duplicate, 0)) return "scene replacement exposed the previous publication";
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (tracer.latest(duplicate, 0)) return "new scene traced with the old frame/materials";
  return {};
}

PT_TEST(cpu_preview_guides_match_odd_sized_color_and_source_camera) {
  Builder b;
  b.wall(10, 0, b.material(float3(0.6f), 0, 0.7f));
  b.environment([](float3 d) { return float3(0.2f + max(d.y, 0.0f)); }, 16, 8);
  b.camera(float3(0, 0, 2), float3(0), 0.8f, 513, 385);
  b.finish(4, 0, 11);
  b.frame.uniforms.image.z = 1;
  b.frame.targetSamples = 1;
  CpuTracer tracer(1);
  tracer.setScene(std::make_shared<CpuScene>(std::move(b.scene)));
  tracer.start(b.frame);
  CpuTracer::Image image;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (tracer.latest(image, 0)) break;
    std::this_thread::yield();
  }
  if (image.version == 0 || image.samples != 0 || image.previewScale != 4)
    return "coarse preview did not publish before the full-resolution single-worker frame";
  const uint gw = uint(image.uniforms.image.x), gh = uint(image.uniforms.image.y);
  if (gw != 129 || gh != 97 || image.guides.size() != std::size_t(gw) * gh)
    return "preview publication has inconsistent guide extent";
  for (uint y = 0; y < image.height; ++y) for (uint x = 0; x < image.width; ++x) {
    const auto &guide = image.guides[(y * gh / image.height) * gw + x * gw / image.width];
    const float *rgb = &image.mean[(std::size_t(y) * image.width + x) * 4];
    if (length(xyz(guide.diffuse) + xyz(guide.specular) - float3(rgb[0], rgb[1], rgb[2])) > 1e-5f)
      return "preview radiance and guides disagree at odd-sized block boundaries";
  }
  auto next = b.frame;
  next.width = next.height = 8;
  next.uniforms.image.x = next.uniforms.image.y = 8;
  next.uniforms.cameraPosition.x = 2;
  tracer.start(next);
  CpuTracer::Image replacement;
  if (tracer.latest(replacement, 0) && replacement.uniforms.cameraPosition.x != 2)
    return "a cancelled preview crossed a camera generation";
  return {};
}

PT_TEST(exact_target_and_stale_denoiser_lifecycle) {
  if (!Denoiser::available()) return "SKIP: this build has no Open Image Denoise";
  Builder b;
  b.wall(1.0f, 0.0f, b.material(float3(0.6f), 0.0f, 0.7f));
  b.environment([](float3 d) { return float3(0.3f + 0.4f * std::max(d.y, 0.0f)); }, 16, 8);
  b.camera(float3(0.0f, 0.0f, 2.0f), float3(0.0f), 0.8f, 8, 8);
  b.finish(4, 0, 9);
  auto scene = std::make_shared<CpuScene>(std::move(b.scene));
  CpuTracer tracer(2);
  tracer.setScene(scene);
  CpuFrame denoised = b.frame;
  denoised.targetSamples = 2;
  denoised.denoise = true;
  tracer.start(denoised);

  CpuTracer::Image image;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < deadline) {
    tracer.latest(image, image.version);
    if (image.samples == 2 && image.denoisedSamples == 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (image.samples != 2 || image.denoisedSamples != 2 || image.denoised.empty())
    return "OIDN result did not publish at the exact two-sample target";

  CpuFrame raw = b.frame;
  raw.width = 7;
  raw.height = 5;
  raw.uniforms.image.x = 7.0f;
  raw.uniforms.image.y = 5.0f;
  raw.targetSamples = 3;
  raw.denoise = false;
  tracer.start(raw);
  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    tracer.latest(image, image.version);
    if (image.width == 7 && image.height == 5 && image.samples == 3) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (image.samples != 3 || image.width != 7 || image.height != 5)
    return "the renderer did not stop at the exact three-sample replacement target";
  if (!image.denoised.empty() || image.denoisedSamples != 0)
    return "a stale denoised image crossed a resolution/denoiser change";
  const std::uint64_t version = image.version;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CpuTracer::Image later;
  if (tracer.latest(later, version) && later.samples > 3)
    return "the CPU tracer rendered beyond its exact target";
  return {};
}

// Embree and the denoiser.

// Random triangle soup under nonuniform and negative transforms, including candidates
// filtered by both alpha modes.
Builder soup(uint trianglesPerInstance) {
  Builder b;
  std::mt19937 random(23);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const uint opaque = b.material(float3(0.5f), 0.0f, 0.5f);
  const uint masked = b.material(float3(0.5f), 0.0f, 0.5f, 0.3f, 1);
  const uint blended = b.material(float3(0.5f), 0.0f, 0.5f, 0.5f, 2);
  for (int instance = 0; instance < 4; ++instance) {
    std::vector<float3> positions, normals;
    std::vector<uint> triangles;
    for (uint t = 0; t < trianglesPerInstance; ++t) {
      const float3 centre(unit(random), unit(random), unit(random));
      for (int k = 0; k < 3; ++k) {
        positions.push_back(centre + float3(unit(random), unit(random), unit(random)) * 0.15f);
        normals.push_back(float3(0.0f, 1.0f, 0.0f));
        triangles.push_back(t * 3 + static_cast<uint>(k));
      }
    }
    const uint material = instance == 2 ? masked : (instance == 3 ? blended : opaque);
    const uint flags = instance == 2 ? kInstanceMasked : (instance == 3 ? kInstanceBlended : 0u);
    const float sign = instance == 3 ? -1.0f : 1.0f;
    b.mesh(positions, normals, triangles, material,
           affine(0.5f * static_cast<float>(instance),
                  float3(sign * (0.5f + 0.2f * instance), 0.65f + 0.1f * instance, 1.1f - 0.1f * instance),
                  float3(static_cast<float>(instance) * 0.9f - 1.35f, 0.1f * static_cast<float>(instance), 0.0f)),
           kRayMaskScene, flags);
  }
  b.environment([](float3) { return float3(1.0f); }, 8, 4);
  b.finish();
  return b;
}

PT_TEST(bvh_refit_matches_rebuild) {
  Builder b = soup(2000);
  // Deform object-space geometry and move one instance while preserving topology.
  for (std::size_t vertex = 0; vertex < b.scene.vertices.size() / kVertexFloats; ++vertex) {
    float *p = b.scene.vertices.data() + vertex * kVertexFloats;
    p[1] += 0.025f * std::sin(static_cast<float>(vertex) * 0.17f);
  }
  const Affine moved = affine(0.83f, float3(0.9f, 0.72f, 1.25f), float3(-0.25f, 0.35f, 0.2f));
  TraceInstance &changed = b.scene.instances[1];
  changed.objectToWorld0 = moved.rows[0]; changed.objectToWorld1 = moved.rows[1]; changed.objectToWorld2 = moved.rows[2];
  changed.worldToObject0 = moved.inverse[0]; changed.worldToObject1 = moved.inverse[1]; changed.worldToObject2 = moved.inverse[2];
  changed.normalToWorld0 = moved.normal[0]; changed.normalToWorld1 = moved.normal[1]; changed.normalToWorld2 = moved.normal[2];

  Bvh refitted = b.scene.bvh;
  const std::size_t oldNodes = refitted.nodes.size(), oldTriangles = refitted.triangles.size();
  const double refitMs = refitBvh(b.scene.vertices, b.scene.indices, b.scene.instances, refitted);
  std::vector<TraceInstance> rebuiltInstances = b.scene.instances;
  Bvh rebuilt;
  const BvhStatistics rebuiltStats = buildBvh(b.scene.vertices, b.scene.indices, rebuiltInstances,
      b.scene.triangleCounts, rebuilt, sharedPool().size());
  if (refitted.nodes.size() != oldNodes || refitted.triangles.size() != oldTriangles)
    return "refit changed BVH topology or leaf storage";

  std::mt19937 random(913);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  constexpr int rays = 100000;
  for (int r = 0; r < rays; ++r) {
    const float3 origin = float3(unit(random), unit(random), unit(random)) * 3.5f;
    const float3 direction = normalize(float3(unit(random) * 1.6f, unit(random), unit(random)) - origin);
    const uint seed = pcgHash(static_cast<uint>(r));
    const PtHit a = ptTraceBvh(refitted.nodes.data(), refitted.triangles.data(), b.scene.instances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
        origin, direction, kPtInfinity, 7u, seed, float2(-1.0f, 0.0f), 0u);
    const PtHit c = ptTraceBvh(rebuilt.nodes.data(), rebuilt.triangles.data(), rebuiltInstances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
        origin, direction, kPtInfinity, 7u, seed, float2(-1.0f, 0.0f), 0u);
    const PtHit ao = ptTraceBvh(refitted.nodes.data(), refitted.triangles.data(), b.scene.instances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
        origin, direction, 4.0f, 7u, seed, float2(-1.0f, 0.0f), 1u);
    const PtHit co = ptTraceBvh(rebuilt.nodes.data(), rebuilt.triangles.data(), rebuiltInstances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
        origin, direction, 4.0f, 7u, seed, float2(-1.0f, 0.0f), 1u);
    if (ao.found != co.found || a.found != c.found) return "refit changed nearest/occlusion found state";
    if (a.found != 0u) {
      const float tolerance = std::max(2e-7f, 1e-4f * a.t);
      if (a.instance != c.instance || a.primitive != c.primitive || std::abs(a.t - c.t) > tolerance ||
          std::abs(a.barycentric.x - c.barycentric.x) > 3e-4f ||
          std::abs(a.barycentric.y - c.barycentric.y) > 3e-4f)
        return "refit changed nearest-hit identity, distance, or barycentrics";
    }
  }
  std::printf("  V6 refit %.3f ms, rebuild %.3f ms, %.2fx speed ratio\n",
              refitMs, rebuiltStats.milliseconds, rebuiltStats.milliseconds / std::max(refitMs, 1e-9));
  return {};
}

PT_TEST(quantized_bvh4_bvh8_match_binary) {
  Builder b = soup(2000);
  const WideBvh wide4 = buildWideBvh(b.scene.bvh, b.scene.instances, 4u);
  const WideBvh wide8 = buildWideBvh(b.scene.bvh, b.scene.instances, 8u);
  if (wide4.nodes.empty() || wide8.nodes.empty()) return "wide conversion emitted no nodes";
  if (wide4.maximumStack == 0u || wide4.maximumStack > 64u ||
      wide8.maximumStack == 0u || wide8.maximumStack > 64u)
    return "wide conversion did not prove its GPU traversal stack bound";
  std::mt19937 random(1447);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  constexpr int rays = 100000;
  std::vector<float3> origins(rays), directions(rays);
  for (int r = 0; r < rays; ++r) {
    origins[r] = float3(unit(random), unit(random), unit(random)) * 3.5f;
    directions[r] = normalize(float3(unit(random) * 1.7f, unit(random), unit(random)) - origins[r]);
  }
  auto timed = [&](auto trace) {
    const auto started = std::chrono::steady_clock::now();
    for (int r = 0; r < rays; ++r) trace(r);
    return rays / std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  };
  std::vector<PtHit> baseline(rays), four(rays), eight(rays), avx(rays);
  const double binaryRate = timed([&](int r) {
    baseline[r] = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
        origins[r], directions[r], kPtInfinity, 7u, pcgHash(static_cast<uint>(r)), float2(-1.0f, 0.0f), 0u);
  });
  const double fourRate = timed([&](int r) {
    four[r] = traceWideBvh(wide4, b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
                           b.scene.textures, origins[r], directions[r], kPtInfinity, 7u,
                           pcgHash(static_cast<uint>(r)), float2(-1.0f, 0.0f), 0u);
  });
  const double eightRate = timed([&](int r) {
    eight[r] = traceWideBvh(wide8, b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
                            b.scene.textures, origins[r], directions[r], kPtInfinity, 7u,
                            pcgHash(static_cast<uint>(r)), float2(-1.0f, 0.0f), 0u);
  });
  double avxRate = 0.0;
  if (cpuAvx2Available())
    avxRate = timed([&](int r) {
      avx[r] = traceWideBvhAvx2(wide8, b.frame.materials.data(), b.scene.indices.data(),
          b.scene.vertices.data(), b.scene.textures, origins[r], directions[r], kPtInfinity,
          7u, pcgHash(static_cast<uint>(r)), float2(-1.0f, 0.0f), 0u);
    });
  std::vector<int> order(rays);
  for (int r = 0; r < rays; ++r) order[r] = r;
  auto directionKey = [&](int r) {
    const float3 d = directions[r];
    const uint octant = (d.x < 0.0f ? 1u : 0u) | (d.y < 0.0f ? 2u : 0u) | (d.z < 0.0f ? 4u : 0u);
    const float3 magnitude = abs(d);
    const uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0u :
                      magnitude.y >= magnitude.z ? 1u : 2u;
    const uint qx = static_cast<uint>(std::clamp((origins[r].x + 3.5f) * (31.0f / 7.0f), 0.0f, 31.0f));
    const uint qy = static_cast<uint>(std::clamp((origins[r].y + 3.5f) * (31.0f / 7.0f), 0.0f, 31.0f));
    const uint qz = static_cast<uint>(std::clamp((origins[r].z + 3.5f) * (31.0f / 7.0f), 0.0f, 31.0f));
    return (octant << 17u) | (axis << 15u) | (qx << 10u) | (qy << 5u) | qz;
  };
  const auto sortStarted = std::chrono::steady_clock::now();
  std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return directionKey(a) < directionKey(c); });
  const double sortSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - sortStarted).count();
  std::vector<PtHit> sorted(rays);
  const auto sortedStarted = std::chrono::steady_clock::now();
  for (int index : order)
    sorted[index] = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
        origins[index], directions[index], kPtInfinity, 7u, pcgHash(static_cast<uint>(index)), float2(-1.0f, 0.0f), 0u);
  const double sortedRate = rays / std::chrono::duration<double>(
      std::chrono::steady_clock::now() - sortedStarted).count();
  const double sortedTotalRate = 1.0 / (1.0 / sortedRate + sortSeconds / rays);
  for (int r = 0; r < rays; ++r) {
    if (baseline[r].found != sorted[r].found || baseline[r].ambiguous != sorted[r].ambiguous ||
        baseline[r].t != sorted[r].t || baseline[r].barycentric.x != sorted[r].barycentric.x ||
        baseline[r].barycentric.y != sorted[r].barycentric.y ||
        baseline[r].instance != sorted[r].instance || baseline[r].primitive != sorted[r].primitive)
      return "ray sorting changed a deterministic traversal result";
    for (const PtHit *candidate : {&four[r], &eight[r]}) {
      const PtHit &expected = baseline[r];
      if (candidate->found != expected.found) return "quantized wide BVH changed found state";
      if (expected.found != 0u) {
        const float tolerance = std::max(2e-7f, 1e-4f * expected.t);
        if (candidate->instance != expected.instance || candidate->primitive != expected.primitive ||
            std::abs(candidate->t - expected.t) > tolerance ||
            std::abs(candidate->barycentric.x - expected.barycentric.x) > 3e-4f ||
            std::abs(candidate->barycentric.y - expected.barycentric.y) > 3e-4f)
          return "quantized wide BVH changed nearest identity, distance, or barycentrics";
      }
    }
    if (avxRate > 0.0) {
      const PtHit &candidate = avx[r], &expected = baseline[r];
      const float tolerance = expected.found != 0u ? std::max(2e-7f, 1e-4f * expected.t) : 0.0f;
      if (candidate.found != expected.found || (expected.found != 0u &&
          (candidate.instance != expected.instance || candidate.primitive != expected.primitive ||
           std::abs(candidate.t - expected.t) > tolerance ||
           std::abs(candidate.barycentric.x - expected.barycentric.x) > 3e-4f ||
           std::abs(candidate.barycentric.y - expected.barycentric.y) > 3e-4f)))
        return "AVX2 BVH8 changed nearest traversal results";
    }
    const uint seed = pcgHash(static_cast<uint>(r));
    const PtHit binaryOcclusion = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(),
        b.scene.instances.data(), b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
        b.scene.textures, origins[r], directions[r], 4.0f, 7u, seed, float2(-1.0f, 0.0f), 1u);
    const PtHit fourOcclusion = traceWideBvh(wide4, b.frame.materials.data(), b.scene.indices.data(),
        b.scene.vertices.data(), b.scene.textures, origins[r], directions[r], 4.0f, 7u, seed, float2(-1.0f, 0.0f), 1u);
    const PtHit eightOcclusion = traceWideBvh(wide8, b.frame.materials.data(), b.scene.indices.data(),
        b.scene.vertices.data(), b.scene.textures, origins[r], directions[r], 4.0f, 7u, seed, float2(-1.0f, 0.0f), 1u);
    if (binaryOcclusion.found != fourOcclusion.found || binaryOcclusion.found != eightOcclusion.found)
      return "quantized wide BVH changed finite occlusion";
    if (avxRate > 0.0) {
      const PtHit avxOcclusion = traceWideBvhAvx2(wide8, b.frame.materials.data(), b.scene.indices.data(),
          b.scene.vertices.data(), b.scene.textures, origins[r], directions[r], 4.0f, 7u, seed, float2(-1.0f, 0.0f), 1u);
      if (binaryOcclusion.found != avxOcclusion.found)
        return "AVX2 BVH8 changed finite occlusion";
    }
  }
  std::printf("  V6 traversal: binary %.2f M/s, sorted %.2f M/s (%.2f incl sort), BVH4 %.2f M/s, BVH8 %.2f M/s, AVX2-BVH8 %.2f M/s; nodes %zu/%zu/%zu\n",
      binaryRate * 1e-6, sortedRate * 1e-6, sortedTotalRate * 1e-6, fourRate * 1e-6, eightRate * 1e-6,
      avxRate * 1e-6,
      b.scene.bvh.nodes.size() / 4u, wide4.nodes.size(), wide8.nodes.size());
  return {};
}

PT_TEST(quantized_wide_empty_scene_is_finite) {
  Bvh binary;
  binary.nodes.resize(4u);
  binary.nodes[0].w = as_type<float>(kBvhEmpty);
  binary.nodes[2].w = as_type<float>(kBvhEmpty);
  for (uint width : {4u, 8u}) {
    const WideBvh wide = buildWideBvh(binary, {}, width);
    if (wide.nodes.size() != 1u || wide.maximumStack != 1u) return "empty wide BVH has invalid shape";
    const QuantizedWideNode &root = wide.nodes.front();
    if (!std::isfinite(root.origin.x) || !std::isfinite(root.origin.y) ||
        !std::isfinite(root.origin.z) || !std::isfinite(root.scale.x) ||
        !std::isfinite(root.scale.y) || !std::isfinite(root.scale.z) ||
        as_type<uint>(root.origin.w) != 0u)
      return "empty wide BVH contains non-finite bounds";
  }
  return {};
}

PT_TEST(embree_matches_the_own_bvh) {
  if (!EmbreeScene::available()) return "SKIP: this build has no Embree";
  Builder b = soup(5000);
  const EmbreeScene embree(b.scene);
  std::mt19937 random(5);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const int rays = 1000000;
  std::vector<float3> origins(rays), directions(rays);
  for (int r = 0; r < rays; ++r) {
    origins[r] = float3(unit(random), unit(random), unit(random)) * 3.0f;
    directions[r] = normalize(float3(unit(random) * 1.5f, unit(random) * 0.6f, unit(random) * 0.6f) - origins[r]);
  }
  std::vector<PtHit> own(rays), theirs(rays), ownBlocked(rays), embreeBlocked(rays);
  auto timed = [&](auto trace) {
    const auto started = std::chrono::steady_clock::now();
    for (int r = 0; r < rays; ++r) trace(r);
    return rays / std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  };
  const double ownRate = timed([&](int r) {
    own[r] = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
                        b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(), b.scene.textures,
                        origins[r], directions[r], kPtInfinity, 7u, 0u, float2(-1.0f, 0.0f), 0u);
  });
  const double embreeRate = timed([&](int r) {
    theirs[r] = embree.trace(b.scene, b.frame, origins[r], directions[r], kPtInfinity, 7u, 0u, float2(-1.0f, 0.0f), 0u);
  });
  timed([&](int r) {
    ownBlocked[r] = ptTraceBvh(b.scene.bvh.nodes.data(), b.scene.bvh.triangles.data(), b.scene.instances.data(),
                               b.frame.materials.data(), b.scene.indices.data(), b.scene.vertices.data(),
                               b.scene.textures, origins[r], directions[r], kPtInfinity, 7u, 0u, float2(-1.0f, 0.0f), 1u);
  });
  timed([&](int r) {
    embreeBlocked[r] = embree.trace(b.scene, b.frame, origins[r], directions[r], kPtInfinity, 7u, 0u, float2(-1.0f, 0.0f), 1u);
  });
  int differ = 0, explainedEdges = 0, explainedRounding = 0, hits = 0, occlusionDiffer = 0;
  for (int r = 0; r < rays; ++r) {
    hits += own[r].found != 0u ? 1 : 0;
    const bool sameFound = own[r].found == theirs[r].found;
    const bool sameIdentity = sameFound && (own[r].found == 0u ||
        (own[r].instance == theirs[r].instance && own[r].primitive == theirs[r].primitive));
    const float difference = own[r].found == 0u ? 0.0f : std::abs(own[r].t - theirs[r].t);
    auto nearEdge = [](const PtHit &hit) {
      return std::min({hit.barycentric.x, hit.barycentric.y,
                       1.0f - hit.barycentric.x - hit.barycentric.y}) < 1e-5f;
    };
    // Embree transforms geometry internally; our code transforms the ray. The same
    // primitive can differ by float ulps, and distinct answers at a triangle edge have
    // no unique geometric winner. Classify those two cases separately.
    const float roundingTolerance = std::max(3e-6f, 1e-4f * own[r].t);
    const bool rounding = sameIdentity && own[r].found != 0u && difference <= roundingTolerance;
    const bool edge = sameFound && own[r].found != 0u && !sameIdentity &&
                      (nearEdge(own[r]) || nearEdge(theirs[r]));
    const bool explained = sameFound && (own[r].found == 0u || rounding || edge);
    if (!explained) {
      if (differ < 10) {
        double oracle = static_cast<double>(kPtInfinity), oracleU = 0.0, oracleV = 0.0;
        uint oracleInstance = 0, oraclePrimitive = 0;
        for (uint i = 0; i < b.scene.instances.size(); ++i) {
          const TraceInstance &instance = b.scene.instances[i];
          if ((instance.flags & kInstanceMasked) != 0u) continue;  // this fixture's alpha is below its cutoff
          for (uint primitive = 0; primitive < b.scene.triangleCounts[i]; ++primitive) {
            if ((instance.flags & kInstanceBlended) != 0u &&
                hashFloat(pcgHash(i * 0x9E3779B9u + primitive)) >= 0.5f)
              continue;
            float3 triangle[3];
            for (uint k = 0; k < 3; ++k) {
              const uint vertex = b.scene.indices[instance.firstIndex + primitive * 3u + k] + instance.vertexOffset;
              const float *p = b.scene.vertices.data() + static_cast<std::size_t>(vertex) * kVertexFloats;
              triangle[k] = applyRows(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2,
                                      float3(p[0], p[1], p[2]));
            }
            double candidate = 0.0;
            double candidateU = 0.0, candidateV = 0.0;
            if (oracleTriangle(origins[r], directions[r], triangle, oracle, candidate, &candidateU, &candidateV)) {
              oracle = candidate;
              oracleU = candidateU;
              oracleV = candidateV;
              oracleInstance = i;
              oraclePrimitive = primitive;
            }
          }
        }
        std::printf("  Embree mismatch ray %d: own found %u %u/%u t %.9g, Embree found %u %u/%u t %.9g\n",
                    r, own[r].found, own[r].instance, own[r].primitive, own[r].t, theirs[r].found,
                    theirs[r].instance, theirs[r].primitive, theirs[r].t);
        std::printf("    own bary (%.9g, %.9g), Embree bary (%.9g, %.9g)\n",
                    own[r].barycentric.x, own[r].barycentric.y,
                    theirs[r].barycentric.x, theirs[r].barycentric.y);
        std::printf("    independent filtered oracle %u/%u t %.9g bary (%.9g, %.9g, %.9g)\n",
                    oracleInstance, oraclePrimitive, oracle, 1.0 - oracleU - oracleV, oracleU, oracleV);
      }
      ++differ;
    }
    else if (edge) ++explainedEdges;
    else if (difference > std::max(2e-7f, 1e-4f * own[r].t)) ++explainedRounding;
    if (ownBlocked[r].found != embreeBlocked[r].found) ++occlusionDiffer;
  }
  std::printf("  one million rays: own BVH %.2f M/s, Embree %.2f M/s, "
              "%d edge cases, %d same-triangle rounding cases\n",
              ownRate * 1e-6, embreeRate * 1e-6, explainedEdges, explainedRounding);
  if (hits < rays / 4) return format("only %g rays hit", hits);
  if (differ != 0) return format("%g of %g nearest rays disagree with Embree", differ, rays);
  if (occlusionDiffer != 0) return format("Embree's occlusion disagrees on %g rays", occlusionDiffer);
  return {};
}

PT_TEST(embree_renders_match_the_own_bvh) {
  if (!EmbreeScene::available()) return "SKIP: this build has no Embree";
  // A scene with an opaque sphere, a floor and a blended wall: the same paths, intersected two ways.
  auto build = [] {
    Builder b;
    b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.9f, 0.6f, 0.3f), 0.2f, 0.4f));
    b.floor(6.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.8f));
    b.wall(0.8f, 1.5f, b.material(float3(0.2f, 0.4f, 0.9f), 0.0f, 0.3f, 0.5f, 2), kInstanceBlended);
    b.scene.instances.back().mask = kRayMaskBlended;
    b.environment([](float3 d) { return float3(0.2f + 0.8f * std::max(d.y, 0.0f)); });
    b.camera(float3(0.0f, 2.0f, 6.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 32, 32);
    b.finish(8);
    return b;
  };
  Builder own = build(), embreeBuilder = build();
  embreeBuilder.scene.embree = std::make_shared<EmbreeScene>(embreeBuilder.scene);
  embreeBuilder.frame.intersector = 1;
  const std::vector<float> a = own.render(512), b = embreeBuilder.render(512);
  double worst = 0.0;
  for (uint by = 0; by < 4; ++by)
    for (uint bx = 0; bx < 4; ++bx) {
      const double x = meanLuminance(a, 32, bx * 8, by * 8, bx * 8 + 8, by * 8 + 8);
      const double y = meanLuminance(b, 32, bx * 8, by * 8, bx * 8 + 8, by * 8 + 8);
      worst = std::max(worst, std::abs(x - y) / std::max(x, 1e-3));
    }
  if (worst > 0.01) return format("an 8 x 8 block differs by %g between the intersectors", worst);
  return {};
}

PT_TEST(avx2_bvh8_renders_match_binary) {
  if (!cpuAvx2Available()) return "SKIP: this CPU has no AVX2";
  auto build = [] {
    Builder b;
    b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f,
             b.material(float3(0.9f, 0.6f, 0.3f), 0.2f, 0.4f), 40);
    b.floor(6.0f, 0.0f, b.material(float3(0.5f), 0.0f, 0.8f));
    b.wall(0.8f, 1.5f, b.material(float3(0.2f, 0.4f, 0.9f), 0.0f, 0.3f, 0.5f, 2),
           kInstanceBlended);
    b.scene.instances.back().mask = kRayMaskBlended;
    b.environment([](float3 d) { return float3(0.2f + 0.8f * std::max(d.y, 0.0f)); });
    b.camera(float3(0.0f, 2.0f, 6.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 32, 32);
    b.finish(8);
    return b;
  };
  Builder binary = build(), avx = build();
  avx.scene.wideAvx2 = std::make_shared<WideBvh>(buildWideBvh(avx.scene.bvh, avx.scene.instances, 8u));
  avx.frame.intersector = 2;
  const CpuTracer::Result a = binary.renderAll(64), b = avx.renderAll(64);
  for (std::size_t i = 0; i < a.color.size(); ++i)
    if (a.color[i] != b.color[i] || a.albedo[i] != b.albedo[i] || a.normal[i] != b.normal[i])
      return "AVX2 BVH8 changed a seeded color or guide channel";
  return {};
}

PT_TEST(denoising_brings_a_noisy_image_closer) {
  if (!Denoiser::available()) return "SKIP: this build has no Open Image Denoise";
  Builder b;
  b.sphere(float3(0.0f, 1.0f, 0.0f), 1.0f, b.material(float3(0.8f, 0.5f, 0.3f), 0.0f, 0.5f));
  b.floor(6.0f, 0.0f, b.material(float3(0.6f), 0.0f, 0.9f));
  b.environment([](float3 d) {
    const float bright = dot(d, normalize(float3(0.3f, 0.8f, 0.4f))) > 0.97f ? 60.0f : 0.0f;
    return float3(0.1f + 0.4f * std::max(d.y, 0.0f) + bright);
  });
  b.camera(float3(0.0f, 2.0f, 6.0f), float3(0.0f, 0.8f, 0.0f), 0.8f, 128, 128);
  b.finish(8);
  const CpuTracer::Result noisy = b.renderAll(4);
  const std::vector<float> reference = b.render(1024);
  Denoiser denoiser(Denoiser::Device::Cpu);
  if (!denoiser.error().empty()) return "the denoiser did not start: " + denoiser.error();
  if (denoiser.device() != "CPU") return "explicit CPU denoising selected " + denoiser.device();
  const std::vector<float> denoised = denoiser.denoise(noisy.color, noisy.albedo, noisy.normal, 128, 128);
  if (denoised.empty()) return "denoising failed: " + denoiser.error();
  auto error = [&](const std::vector<float> &image) {
    double sum = 0.0, total = 0.0;
    for (std::size_t i = 0; i < image.size(); ++i) {
      sum += std::abs(image[i] - reference[i]);
      total += reference[i];
    }
    return sum / total;
  };
  const double before = error(noisy.color), after = error(denoised);
  std::printf("  relative error at 4 samples against 1024: %.3f raw, %.3f denoised on %s\n", before, after,
              denoiser.device().c_str());
  if (after > before * 0.5) return format("denoising left the error at %g of %g", after, before);
  return {};
}

// Thin lens: aperture 0 is bitwise the pinhole; every lens sample
// crosses the focal plane where the pinhole ray does; at depth z a lens-edge ray lies
// A |1 - z / F| off the pinhole ray, the circle of confusion in object space.
PT_TEST(thin_lens_identity_focus_and_circle_of_confusion) {
  const float3 eye(0.3f, -0.2f, 5.0f);
  const float3 forward = normalize(float3(0.1f, 0.05f, -1.0f));
  const float3 right = normalize(cross(forward, float3(0.0f, 1.0f, 0.0f))) * 1.7f;  // scaled, as PathUniforms
  const float3 up = normalize(cross(right, forward)) * 0.9f;
  for (uint i = 0; i < 1000; ++i) {
    const float4 xi = pathRandom4(pcgHash(i), 0u, 0u);
    const float3 pinhole = normalize(forward + right * (xi.x * 2.0f - 1.0f) + up * (xi.y * 2.0f - 1.0f));
    float3 origin = eye, direction = pinhole;
    ptApplyThinLens(float4(0.0f, 3.0f, 0.0f, 0.0f), forward, right, up, float2(xi.z, xi.w), origin, direction);
    if (std::memcmp(&origin, &eye, sizeof(float3)) != 0 || std::memcmp(&direction, &pinhole, sizeof(float3)) != 0)
      return "aperture 0 changed the pinhole ray";
    const float aperture = 0.05f, focus = 3.0f;
    ptApplyThinLens(float4(aperture, focus, 0.0f, 0.0f), forward, right, up, float2(xi.z, xi.w), origin, direction);
    const float3 lensOffset = origin - eye;
    if (length(lensOffset) > aperture * 1.0001f || std::abs(dot(lensOffset, forward)) > 1e-6f)
      return "the lens point left the aperture disk in the lens plane";
    // Both rays meet the focal plane (depth `focus` along the axis) at one point.
    const float3 expected = eye + pinhole * (focus / dot(pinhole, forward));
    const float3 actual = origin + direction * (dot(expected - origin, forward) / dot(direction, forward));
    if (length(actual - expected) > 1e-4f * focus) return format("a lens ray missed the focus point by %g", length(actual - expected));
  }
  // Circle of confusion: the axial pixel, lens samples on the disk edge.
  for (const float depth : {1.0f, 2.0f, 6.0f, 12.0f}) {
    for (const float2 edge : {float2(1.0f, 0.5f), float2(0.5f, 1.0f), float2(0.0f, 0.5f), float2(0.5f, 0.0f)}) {
      float3 origin = eye, direction = forward;
      ptApplyThinLens(float4(0.05f, 3.0f, 0.0f, 0.0f), forward, right, up, edge, origin, direction);
      const float3 atDepth = origin + direction * ((depth - dot(origin - eye, forward)) / dot(direction, forward));
      const float3 lateral = atDepth - (eye + forward * depth);
      const float expectedRadius = 0.05f * std::abs(1.0f - depth / 3.0f);
      if (std::abs(length(lateral) - expectedRadius) > 1e-4f * std::max(expectedRadius, 1.0f))
        return format("circle of confusion %g at depth %g, expected %g", length(lateral), depth, expectedRadius);
    }
  }
  return {};
}

// V7 transmission and clearcoat (frozen tests).

// A flat +z surface with the V7 layers set: transmission factor, IOR, thin-walled or closed,
// arrival from the front (entering) or the back, and a clearcoat.
PtSurface layeredSurface(float3 base, float metallic, float roughness, float transmission, float ior, bool thin,
                         bool entering, float coat = 0.0f, float coatRoughness = kMinRoughness) {
  PtSurface surface = flatSurface(float3(0.0f, 0.0f, 1.0f), base, metallic, roughness);
  surface.transmission = float4(transmission, ior, thin ? 1.0f : 0.0f, entering ? 1.0f : 0.0f);
  surface.clearcoat = float4(coat, coatRoughness, 0.0f, 0.0f);
  return surface;
}

float3 viewAt(float degrees) {
  const float angle = degrees * kPi / 180.0f;
  return float3(std::sin(angle), 0.0f, std::cos(angle));
}

// The integral of f over the sphere around a +z surface seen from `view`: the upper half on a
// grid centred on the mirror direction, the lower half on one centred on the straight-through
// (or, with relative index etap, the refracted) direction.
double sphereIntegral(float3 view, float etap, uint n, const std::function<double(float3)> &f) {
  const float3 mirror(-view.x, -view.y, view.z);
  float3 through = -view;
  const float sine2 = (1.0f - view.z * view.z) / (etap * etap);
  if (etap != 1.0f && sine2 < 1.0f) {
    const float scale = 1.0f / etap;
    through = normalize(float3(-view.x * scale, -view.y * scale, -std::sqrt(1.0f - sine2)));
  }
  return hemisphereIntegral(mirror, 1.0f, n, f) + hemisphereIntegral(through, -1.0f, n, f);
}

// Directional albedo in importance (flux) mode: the radiance-mode refraction BTDF carries
// 1 / etap^2, so transmitted directions are weighted back by etap^2.
double layeredAlbedo(const PtSurface &surface, float3 view, uint n) {
  static const std::vector<float> table = buildSpecularAlbedoTable();
  const PtBsdf bsdf = ptMakeBsdf(surface, view, table.data());
  const double transmittedWeight = static_cast<double>(bsdf.transmission.y) * bsdf.transmission.y;
  return sphereIntegral(view, bsdf.transmission.y, n, [&](float3 light) {
    float pdf = 0.0f;
    const double value = ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, light, pdf));
    return light.z < 0.0f ? value * transmittedWeight : value;
  });
}

PT_TEST(v7_white_furnace_transmission_and_clearcoat) {
  const float3 white(1.0f);
  for (const float roughness : {0.05f, 0.5f, 1.0f})
    for (const float degrees : {0.0f, 45.0f, 80.0f}) {
      const uint n = 2048u;
      const double thin = layeredAlbedo(layeredSurface(white, 0.0f, roughness, 1.0f, 1.5f, true, true), viewAt(degrees), n);
      if (thin < 0.95 || thin > 1.005)
        return format("thin-walled white transmission, roughness %g at %g degrees: albedo %g", roughness, degrees, thin);
      const double closed = layeredAlbedo(layeredSurface(white, 0.0f, roughness, 1.0f, 1.5f, false, true), viewAt(degrees), n);
      if (closed < 0.90 || closed > 1.005)
        return format("closed white dielectric, roughness %g at %g degrees: albedo %g", roughness, degrees, closed);
    }
  for (const float coatRoughness : {0.05f, 0.5f})
    for (const float degrees : {0.0f, 45.0f, 80.0f}) {
      const uint n = 2048u;
      const double coated = layeredAlbedo(layeredSurface(white, 0.0f, 1.0f, 0.0f, 1.5f, true, true, 1.0f, coatRoughness),
                                          viewAt(degrees), n);
      if (coated > 1.005)
        return format("white Lambert under clearcoat roughness %g at %g degrees: albedo %g", coatRoughness, degrees, coated);
    }
  return {};
}

PT_TEST(v7_fresnel_ior_and_total_internal_reflection) {
  static const std::vector<float> table = buildSpecularAlbedoTable();
  for (const float ior : {1.33f, 1.5f, 2.0f}) {
    const PtBsdf bsdf = ptMakeBsdf(layeredSurface(float3(1.0f), 0.0f, 0.5f, 1.0f, ior, false, true), viewAt(0.0f),
                                   table.data());
    const double expected = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
    const double normal = ptFresnelInterface(bsdf.f0, 1.0f, bsdf.transmission.y).x;
    if (std::abs(normal - expected) > 0.005 * expected)
      return format("ior %g: normal-incidence reflectance %g, expected %g", ior, normal, expected);
  }
  // From inside glass at 60 degrees (critical angle 41.8): no transmission, all reflected.
  const PtSurface inside = layeredSurface(float3(1.0f), 0.0f, kMinRoughness, 1.0f, 1.5f, false, false);
  const PtBsdf bsdf = ptMakeBsdf(inside, viewAt(60.0f), table.data());
  auto value = [&](float3 light) {
    float pdf = 0.0f;
    return static_cast<double>(ptLuminance(ptBsdfEvaluate(bsdf, inside.geometricNormal, light, pdf)));
  };
  const float3 view = viewAt(60.0f);
  const double reflected = hemisphereIntegral(float3(-view.x, -view.y, view.z), 1.0f, 2048u, value);
  const double transmitted = hemisphereIntegral(-view, -1.0f, 2048u, value);
  if (transmitted > 1e-4) return format("total internal reflection transmits %g", transmitted);
  if (reflected < 0.95 || reflected > 1.005) return format("total internal reflection reflects %g", reflected);
  // A smooth micro-normal refracts nothing beyond the critical angle.
  const float3 none = ptTransmittedDirection(bsdf, float3(0.0f, 0.0f, 1.0f));
  if (dot(none, none) != 0.0f) return "refraction beyond the critical angle returned a direction";
  return {};
}

PT_TEST(v7_refraction_eta_transitions) {
  static const std::vector<float> table = buildSpecularAlbedoTable();
  const float3 normal(0.0f, 0.0f, 1.0f);
  for (const float degrees : {0.0f, 20.0f, 35.0f, 40.0f}) {
    // Enter a closed slab through its top face, leave through the parallel bottom face.
    const float3 incoming = -viewAt(degrees);
    const PtBsdf top = ptMakeBsdf(layeredSurface(float3(1.0f), 0.0f, 0.5f, 1.0f, 1.5f, false, true), -incoming,
                                  table.data());
    const float3 inside = ptTransmittedDirection(top, normal);
    if (inside.z >= 0.0f) return format("entering at %g degrees did not refract downwards", degrees);
    // The bottom face, seen from inside: its shading normal faces the ray (up), leaving glass.
    const PtBsdf bottom = ptMakeBsdf(layeredSurface(float3(1.0f), 0.0f, 0.5f, 1.0f, 1.5f, false, false), -inside,
                                     table.data());
    const float3 outgoing = ptTransmittedDirection(bottom, normal);
    const double error = std::acos(std::min(1.0f, dot(outgoing, incoming)));
    if (!(error < 1e-4)) return format("slab at %g degrees: exit direction off by %g rad", degrees, error);
    // A thin-walled sheet transmits a smooth micro-normal's ray straight on.
    const PtBsdf sheet = ptMakeBsdf(layeredSurface(float3(1.0f), 0.0f, 0.5f, 1.0f, 1.5f, true, true), -incoming,
                                    table.data());
    const float3 straight = ptTransmittedDirection(sheet, normal);
    const double bend = std::acos(std::min(1.0f, dot(straight, incoming)));
    if (!(bend < 1e-4)) return format("thin-walled sheet at %g degrees bent the ray by %g rad", degrees, bend);
  }
  return {};
}

// Wilson-Hilferty upper tail of the chi-square distribution.
double chiSquareTail(double statistic, double degrees) {
  const double k = 2.0 / (9.0 * degrees);
  const double z = (std::cbrt(statistic / degrees) - (1.0 - k)) / std::sqrt(k);
  return 0.5 * std::erfc(z / std::sqrt(2.0));
}

PT_TEST(v7_bsdf_sampling_matches_pdf) {
  static const std::vector<float> table = buildSpecularAlbedoTable();
  struct Case { const char *name; PtSurface surface; float degrees; };
  const float3 tint(0.8f, 0.7f, 0.6f);
  const Case cases[] = {
      {"thin-walled transmission", layeredSurface(tint, 0.0f, 0.5f, 1.0f, 1.5f, true, true), 30.0f},
      {"closed dielectric entering", layeredSurface(tint, 0.0f, 0.3f, 0.7f, 1.5f, false, true), 50.0f},
      {"closed dielectric leaving", layeredSurface(tint, 0.0f, 0.5f, 1.0f, 1.5f, false, false), 30.0f},
      {"inside beyond the critical angle", layeredSurface(tint, 0.0f, 0.3f, 1.0f, 1.5f, false, false), 60.0f},
      {"clearcoat over rough dielectric", layeredSurface(tint, 0.0f, 0.8f, 0.0f, 1.5f, true, true, 1.0f, 0.2f), 40.0f},
      {"clearcoat over transmission", layeredSurface(tint, 0.2f, 0.4f, 0.5f, 1.5f, false, true, 0.5f, 0.3f), 60.0f},
  };
  constexpr uint rows = 24u, columns = 48u, samples = 1u << 20;
  for (const Case &c : cases) {
    const PtBsdf bsdf = ptMakeBsdf(c.surface, viewAt(c.degrees), table.data());
    auto binOf = [&](float3 d) {
      const uint row = std::min(rows - 1u, static_cast<uint>((d.z + 1.0f) * 0.5f * rows));
      float phi = std::atan2(d.y, d.x);
      if (phi < 0.0f) phi += 2.0f * kPi;
      const uint column = std::min(columns - 1u, static_cast<uint>(phi / (2.0f * kPi) * columns));
      return row * columns + column;
    };
    // The reported pdf must be the sampler's density wherever the BSDF is nonzero (where it
    // is zero a sample contributes nothing, whatever its pdf). Expected mass per bin: the pdf
    // integrated over those directions on a grid fine enough to resolve each bin.
    std::vector<double> expected(rows * columns, 0.0);
    const uint n = 1536u;
    for (uint i = 0; i < n; ++i) {
      const double cosine = -1.0 + 2.0 * (i + 0.5) / n, sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
      for (uint j = 0; j < 2u * n; ++j) {
        const double phi = 2.0 * 3.14159265358979323846 * (j + 0.5) / (2.0 * n);
        const float3 d(static_cast<float>(sine * std::cos(phi)), static_cast<float>(sine * std::sin(phi)),
                       static_cast<float>(cosine));
        float pdf = 0.0f;
        if (ptMaxComponent(ptBsdfEvaluate(bsdf, c.surface.geometricNormal, d, pdf)) > 0.0f)
          expected[binOf(d)] += pdf * 4.0 * 3.14159265358979323846 / (2.0 * n * n);
      }
    }
    std::vector<double> observed(rows * columns, 0.0);
    std::mt19937 random(12345u);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    uint valid = 0;
    for (uint s = 0; s < samples; ++s) {
      const float3 d = ptBsdfSampleDirection(bsdf, float3(uniform(random), uniform(random), uniform(random)));
      float pdf = 0.0f;
      if (!(ptMaxComponent(ptBsdfEvaluate(bsdf, c.surface.geometricNormal, d, pdf)) > 0.0f)) continue;
      ++valid;
      observed[binOf(normalize(d))] += 1.0;
    }
    double mass = 0.0;
    for (const double e : expected) mass += e;
    const double validFraction = static_cast<double>(valid) / samples;
    if (std::abs(mass - validFraction) > 0.01)
      return std::string(c.name) + format(": pdf integrates to %g where the BSDF is nonzero, sampler's share there %g",
                                          mass, validFraction);
    double statistic = 0.0, pooledExpected = 0.0, pooledObserved = 0.0;
    uint bins = 0;
    for (uint b = 0; b < rows * columns; ++b) {
      const double e = expected[b] * samples;
      if (e < 20.0) {
        pooledExpected += e;
        pooledObserved += observed[b];
        continue;
      }
      statistic += (observed[b] - e) * (observed[b] - e) / e;
      ++bins;
    }
    if (pooledExpected >= 20.0) {
      statistic += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
      ++bins;
    }
    const double p = chiSquareTail(statistic, std::max(1.0, bins - 1.0));
    if (!(p > 0.001))
      return std::string(c.name) + format(": chi-square %g over %g bins, p %g", statistic, bins, p);
  }
  return {};
}

// The V6 evaluator and sampler as they stood before V7 (shaders/pt/bsdf.h with the unclamped
// ptDistributionGGX), the frozen reference for "clearcoat factor 0 is bitwise-identical to V6".
float3 v6BsdfEvaluate(const PtBsdf &bsdf, float3 geometricNormal, float3 light, float &pdf) {
  pdf = 0.0f;
  const float normalDotLight = dot(bsdf.normal, light);
  if (normalDotLight <= 0.0f || dot(geometricNormal, light) <= 0.0f) return float3(0.0f);
  const float3 halfVector = normalize(bsdf.view + light);
  const float viewDotHalf = saturate(dot(bsdf.view, halfVector));
  const float3 fresnel = fresnelSchlick(bsdf.f0, viewDotHalf);
  const float distribution = ptDistributionGGX(bsdf.normal, halfVector, bsdf.alpha);
  const float visibility = visibilitySmith(bsdf.normalDotView, normalDotLight, bsdf.alpha);
  const float3 specular = fresnel * (distribution * visibility) * bsdf.compensation;
  const float3 diffuse = (float3(1.0f) - fresnel) * bsdf.diffuseColor * (1.0f / kPi);
  const float specularPdf = distribution * ptSmithG1(bsdf.normalDotView, bsdf.alpha) / (4.0f * bsdf.normalDotView);
  const float diffusePdf = normalDotLight * (1.0f / kPi);
  pdf = bsdf.specularProbability * specularPdf + (1.0f - bsdf.specularProbability) * diffusePdf;
  return (specular + diffuse) * normalDotLight;
}

float3 v6BsdfSampleDirection(const PtBsdf &bsdf, float3 xi) {
  float3 direction = cosineSampleHemisphere(bsdf.normal, float2(xi.x, xi.y));
  if (xi.z < bsdf.specularProbability) {
    const float3 viewLocal = float3(dot(bsdf.view, bsdf.tangent), dot(bsdf.view, bsdf.bitangent),
                                    dot(bsdf.view, bsdf.normal));
    const float3 halfLocal = ptSampleVisibleNormal(viewLocal, bsdf.alpha, float2(xi.x, xi.y));
    direction = reflect(-bsdf.view,
                        bsdf.tangent * halfLocal.x + bsdf.bitangent * halfLocal.y + bsdf.normal * halfLocal.z);
  }
  return direction;
}

PT_TEST(v7_without_layers_is_bitwise_v6_and_coat_matches_integration) {
  static const std::vector<float> table = buildSpecularAlbedoTable();
  std::mt19937 random(777u);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (uint i = 0; i < 20000u; ++i) {
    const float3 base(uniform(random), uniform(random), uniform(random));
    const PtSurface surface = layeredSurface(base, uniform(random), std::max(kMinRoughness, uniform(random)), 0.0f,
                                             1.5f, true, true);
    const float3 view = normalize(float3(uniform(random) - 0.5f, uniform(random) - 0.5f, uniform(random) + 0.05f));
    const PtBsdf bsdf = ptMakeBsdf(surface, view, table.data());
    if (bsdf.f0.x != mix(float3(0.04f), base, surface.metallic).x) return "F0 at IOR 1.5 differs from V6";
    const float3 xi(uniform(random), uniform(random), uniform(random));
    const float3 sampled = ptBsdfSampleDirection(bsdf, xi), reference = v6BsdfSampleDirection(bsdf, xi);
    if (std::memcmp(&sampled, &reference, sizeof(float3)) != 0) return "a sampled direction differs from V6";
    const float3 light = normalize(float3(uniform(random) - 0.5f, uniform(random) - 0.5f, uniform(random) - 0.3f));
    float pdf = 0.0f, referencePdf = 0.0f;
    const float3 value = ptBsdfEvaluate(bsdf, surface.geometricNormal, light, pdf);
    const float3 referenceValue = v6BsdfEvaluate(bsdf, surface.geometricNormal, light, referencePdf);
    if (std::memcmp(&value, &referenceValue, sizeof(float3)) != 0 || std::memcmp(&pdf, &referencePdf, sizeof(float)) != 0)
      return "an evaluation differs from V6";
  }
  // A clearcoat over a black metal base (F0 = 0, so the base keeps only Schlick's grazing
  // (1 - v.h)^5 and no compensation): coat lobe plus attenuated base, integrated from
  // independent formulas.
  for (const float coatRoughness : {0.2f, 0.6f})
    for (const float degrees : {0.0f, 45.0f, 75.0f}) {
      const float3 view = viewAt(degrees);
      const double production =
          layeredAlbedo(layeredSurface(float3(0.0f), 1.0f, 0.5f, 0.0f, 1.5f, true, true, 1.0f, coatRoughness), view, 1024u);
      const double a = static_cast<double>(coatRoughness) * coatRoughness, a2 = a * a, nv = view.z;
      const double b2 = 0.25 * 0.25;  // base alpha^2, roughness 0.5
      auto ggx = [](double nh, double alpha2) {
        return alpha2 / (3.14159265358979323846 * std::pow(nh * nh * (alpha2 - 1.0) + 1.0, 2.0));
      };
      auto smith = [](double v, double l, double alpha2) {
        return 0.5 / (l * std::sqrt(v * v * (1.0 - alpha2) + alpha2) + v * std::sqrt(l * l * (1.0 - alpha2) + alpha2));
      };
      auto coatFresnel = [](double cosine) { return 0.04 + 0.96 * std::pow(1.0 - std::min(1.0, cosine), 5.0); };
      const double independent = sphereIntegral(view, 1.0f, 1024u, [&](float3 light) {
        const double nl = light.z;
        if (nl <= 0.0) return 0.0;
        const float3 h = normalize(view + light);
        const double nh = h.z, vh = std::max(0.0f, dot(view, h));
        const double coat = coatFresnel(vh) * ggx(nh, a2) * smith(nv, nl, a2);
        const double base = std::pow(1.0 - vh, 5.0) * ggx(nh, b2) * smith(nv, nl, b2);
        return (coat + (1.0 - coatFresnel(nv)) * (1.0 - coatFresnel(nl)) * base) * nl;
      });
      if (std::abs(production - independent) > 0.01 * independent)
        return format("clearcoat roughness %g at %g degrees: albedo %g", coatRoughness, degrees, production) +
               format(", independent %g", independent);
    }
  return {};
}

// V7 ray-cone texture filtering (frozen tests).

// A textured plane under a scaling instance transform, seen along rays at 0, 60 and 80 degrees
// from its normal at distances 1, 4 and 16: the production level of detail against the
// closed form 0.5 log2(uvArea / worldArea) + log2(w / cos) + 0.5 log2(W H), within 0.01.
PT_TEST(raycone_lod_matches_the_analytic_plane) {
  // Object-space unit square [0,1]^2 with UVs [0,1]^2, scaled by 2 in x and 3 in y into world.
  const float4 row0(2.0f, 0.0f, 0.0f, 5.0f), row1(0.0f, 3.0f, 0.0f, -1.0f), row2(0.0f, 0.0f, 1.0f, 2.0f);
  const float3 worldCross = ptWorldCross(row0, row1, row2, float3(0.0f), float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float uvCross = ptUvCross(float2(0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f));
  const float spread = 0.02f, width = 256.0f, height = 128.0f, levels = 9.0f;
  for (const float distance : {1.0f, 4.0f, 16.0f})
    for (const float degrees : {0.0f, 60.0f, 80.0f}) {
      const double angle = degrees * 3.14159265358979323846 / 180.0;
      const float3 direction(static_cast<float>(std::sin(angle)), 0.0f, static_cast<float>(-std::cos(angle)));
      const float coneWidth = ptConeWidthOrLevelZero(float2(0.0f, spread), distance);
      const float lod = ptTextureLevel(ptConeLodBase(uvCross, worldCross, direction, coneWidth), width, height, levels);
      const double expected = 0.5 * std::log2(1.0 / 6.0) + std::log2(double(spread) * distance / std::cos(angle)) +
                              0.5 * std::log2(double(width) * height);
      if (!(expected > 0.0 && expected < levels - 1.0)) return format("fixture level %g is clamped", expected);
      if (std::abs(lod - expected) > 0.01)
        return format("distance %g, %g degrees: level %g", distance, degrees, lod) + format(", expected %g", expected);
    }
  // Level zero (filter mode 0) is the sentinel, which texture reads map to level 0 exactly.
  if (ptConeLodBase(uvCross, worldCross, float3(0.0f, 0.0f, -1.0f), ptConeWidthOrLevelZero(ptCameraCone(float4(0.0f)), 3.0f)) !=
      kPtLevelZero)
    return "level-zero mode produced a level of detail";
  for (int e = -40; e <= 40; ++e)
    for (const float m : {1.0f, 1.25f, 1.5f, 1.9999f}) {
      const float x = std::ldexp(m, e);
      if (std::abs(ptLog2(x) - std::log2(double(x))) > 4e-6) return format("ptLog2(%g) = %g", x, ptLog2(x));
    }
  return {};
}

// Masked foliage at 1-64 m (alpha footprint): a 60%-opaque one-texel pattern at a
// fixed density (64 texels per 4 m) filling the view from 1 to 64 m, so the level of detail
// grows from 0 to about 5. The mean coverage over 256 SPP per distance may vary by at most
// 0.05 under ray cones. Prints the coverage table for both modes.
PT_TEST(raycone_alpha_footprint_coverage) {
  HostTexture leaves;
  leaves.width = leaves.height = 64;
  leaves.texels.resize(64 * 64);
  for (uint y = 0; y < 64; ++y)
    for (uint x = 0; x < 64; ++x)  // 60% opaque, in one-texel stripes and specks
      leaves.texels[y * 64 + x] = 0x00FFFFFFu | ((((x * 7u + y * 13u) % 10u) < 6u ? 255u : 0u) << 24u);
  buildMipChain(leaves);
  std::array<std::array<double, 6>, 2> coverage{};
  const float distances[6] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 64.0f};
  for (uint mode = 0; mode < 2; ++mode)
    for (uint d = 0; d < 6; ++d) {
      Builder b;
      const uint leaf = b.material(float3(0.3f, 0.6f, 0.2f), 0.0f, 0.8f, 1.0f, 1u);  // MASK, cutoff 0.5
      const float half = 40.0f;  // fills the view at every distance
      b.mesh({{-half, -half, 0}, {half, -half, 0}, {half, half, 0}, {-half, half, 0}}, std::vector<float3>(4, float3(0, 0, 1)),
             {0, 1, 2, 0, 2, 3}, leaf, affine(0.0f, float3(1.0f), float3(0.0f)), kRayMaskScene, kInstanceMasked);
      b.scene.textures.textures.push_back(leaves);
      b.scene.textures.slots[2] = static_cast<uint>(b.scene.textures.textures.size() - 1);
      b.scene.instances.back().slots = 2u | (1u << 24);
      // One texture repeat per 4 m: the pixel footprint in texels grows with the distance.
      for (uint v = 0; v < 4; ++v) {
        float *vertex = b.scene.vertices.data() + (b.scene.instances.back().vertexOffset + v) * kVertexFloats;
        vertex[10] = vertex[0] / 4.0f;
        vertex[11] = vertex[1] / 4.0f;
      }
      b.environment([](float3) { return float3(1.0f); }, 8, 4);
      b.camera(float3(0.0f, 0.0f, distances[d]), float3(0.0f), 0.9f, 32, 32);
      b.finish(1);
      b.frame.uniforms.lens = float4(0.0f, 1.0f, float(mode), std::atan(2.0f * std::tan(0.45f) / 32.0f));
      double covered = 0.0;
      for (uint s = 0; s < 256; ++s)
        for (uint p = 0; p < 32 * 32; ++p)
          covered += CpuTracer::tracePixel(b.scene, b.frame, p % 32, p / 32, s).albedo.x < 0.5f ? 1.0 : 0.0;
      coverage[mode][d] = covered / (256.0 * 32 * 32);
    }
  for (uint mode = 0; mode < 2; ++mode) {
    std::printf("  alpha footprint %s coverage:", mode == 0 ? "level zero" : "ray cones ");
    for (uint d = 0; d < 6; ++d) std::printf(" %gm %.3f", distances[d], coverage[mode][d]);
    std::printf("\n");
  }
  const auto [low, high] = std::minmax_element(coverage[1].begin(), coverage[1].end());
  if (*high - *low > 0.05) return format("ray-cone coverage varies by %g across distances", *high - *low);
  // The pattern is 60% opaque: a cutoff on averaged alpha would give 1.0 everywhere, and at
  // 1 m (level zero) the two modes must agree.
  double mean = 0.0;
  for (double c : coverage[1]) mean += c / 6.0;
  if (std::abs(mean - 0.6) > 0.05) return format("ray-cone coverage %g, the pattern is 60%% opaque", mean);
  if (std::abs(coverage[1][0] - coverage[0][0]) > 0.05)
    return format("at 1 m ray cones give %g, level zero %g", coverage[1][0], coverage[0][0]);
  return {};
}

// V7 ReSTIR DI (frozen reservoir invariants).

// Streaming RIS keeps candidate i with probability w_i / w_sum: chi-square over 1M trials
// of eight unequal weights (7 degrees of freedom; 24.3 is the 0.1% critical value).
PT_TEST(restir_streaming_selection_matches_the_weights) {
  const float weights[8] = {0.5f, 3.0f, 0.0f, 1.25f, 7.0f, 0.1f, 2.0f, 4.0f};
  double total = 0.0;
  for (float w : weights) total += w;
  std::array<double, 8> counts{};
  const uint trials = 1000000;
  for (uint t = 0; t < trials; ++t) {
    PtReservoir r = ptEmptyReservoir();
    const uint stream = pcgHash(t * 0x9E3779B9u + 17u);
    for (uint i = 0; i < 8; ++i) ptReservoirStream(r, i, float2(0.0f), weights[i], 1.0f, ptRestirStreamRandom(stream, i));
    if (r.light == kPtRestirNoLight) return "no candidate selected";
    counts[r.light] += 1.0;
  }
  if (counts[2] != 0.0) return "a zero-weight candidate was selected";
  double chiSquare = 0.0;
  for (uint i = 0; i < 8; ++i) {
    const double expected = trials * weights[i] / total;
    if (expected > 0.0) chiSquare += (counts[i] - expected) * (counts[i] - expected) / expected;
  }
  if (chiSquare > 24.3) return format("chi-square %g over 7 degrees of freedom", chiSquare);
  return {};
}

namespace {
// A receiver at the origin facing +z, seen head-on, for reservoir estimates without geometry.
struct RestirReceiver {
  Builder b;
  PtSurface surface;
  PtBsdf bsdf;
  PtRestirLights lightInfo;
  explicit RestirReceiver(uint pointLights, bool environment) {
    b.material(float3(0.6f, 0.5f, 0.4f), 0.0f, 0.8f);
    if (environment) b.environment([](float3 d) { return float3(1.0f + 0.5f * d.y, 1.0f, 1.0f - 0.25f * d.x); }, 64, 32);
    for (uint i = 0; i < pointLights; ++i) {
      Light light{};
      const float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(std::max(pointLights, 1u));
      light.position = float4(1.5f * std::cos(a), 1.5f * std::sin(a), 1.0f + 0.5f * static_cast<float>(i), 0.0f);
      light.color = float4(1.0f, 0.8f + 0.1f * static_cast<float>(i), 0.6f, 4.0f + static_cast<float>(i));
      light.direction = float4(0.0f, 0.0f, -1.0f, 1.0f);
      light.cone = float4(1.0f, 0.0f, 0.0f, 0.0f);
      b.frame.lights.push_back(light);
    }
    b.finish(1);
    if (!environment) {
      b.frame.uniforms.distribution = float4(0.0f);
    }
    surface = flatSurface(float3(0.0f, 0.0f, 1.0f), float3(0.6f, 0.5f, 0.4f), 0.0f, 0.8f);
    bsdf = ptMakeBsdf(surface, float3(0.0f, 0.0f, 1.0f), b.scene.specularAlbedo.data());
    const PathUniforms &u = b.frame.uniforms;
    lightInfo = ptRestirLights(u.environment, u.distribution, u.sunDirection, u.sunRadiance, u.path, u.counts, u.emissive);
  }
  // f L G W for the initial reservoir of seed `seed` with M candidates, before visibility.
  float3 estimate(uint seed, uint candidates) const {
    const Light *lights = b.frame.lights.data();
    const float *environmentDistribution = b.scene.distribution.data();
    const PtEmissiveTriangle *emissiveTriangles = b.scene.emissiveTriangles.data();
    const TraceInstance *traceInstances = b.scene.instances.data();
    const Material *materials = b.frame.materials.data();
    const HostTextures &maps = b.scene.textures;
    const HostEnvironment &environmentMap = b.scene.environment;
    const PtReservoir r = ptRestirInitial(PT_RESTIR_LIGHT_ARGS, lightInfo, bsdf, surface.position, surface.geometricNormal,
                                          float2(-1.0f, 0.0f), seed, candidates);
    if (r.light == kPtRestirNoLight) return float3(0.0f);
    float3 direction(0.0f);
    float reach = 0.0f, sourcePdf = 0.0f, solidAnglePdf = 0.0f, bsdfPdf = 0.0f;
    return ptRestirIntegrand(PT_RESTIR_LIGHT_ARGS, lightInfo, bsdf, surface.position, surface.geometricNormal, r.light,
                             float2(r.paramX, r.paramY), float2(-1.0f, 0.0f), direction, reach, sourcePdf,
                             solidAnglePdf, bsdfPdf) *
           r.weight;
  }
  // The same integral by quadrature: the environment over the hemisphere (stratified), and
  // every point light exactly.
  double expected() const {
    double sum = 0.0;
    if (b.frame.uniforms.distribution.w > 0.5f) {
      const int n = 1 << 20;
      double env = 0.0;
      for (int i = 0; i < n; ++i) {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(n), v = radicalInverse(static_cast<uint>(i));
        const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - u * u)), phi = 2.0f * kPi * v;
        const float3 light(sinTheta * std::cos(phi), sinTheta * std::sin(phi), u);
        float pdf = 0.0f;
        env += ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, light, pdf) *
                           b.scene.environment.sample(equirectangularUV(light)));
      }
      sum += env / n * 2.0 * kPi;
    }
    for (const Light &light : b.frame.lights) {
      float3 direction(0.0f);
      float distance = 0.0f;
      const float3 arriving = ptPunctualLight(light, surface.position, direction, distance);
      float pdf = 0.0f;
      sum += ptLuminance(ptBsdfEvaluate(bsdf, surface.geometricNormal, direction, pdf) * arriving);
    }
    return sum;
  }
};
}  // namespace

// W makes f L G / p unbiased for a fixed receiver (within 1% of quadrature), for zero, one and
// many lights and M = 1, 4 and 8; with no light the reservoir is empty.
PT_TEST(restir_contribution_weight_is_unbiased) {
  for (const auto [points, environment] : {std::pair<uint, bool>{0u, false}, {1u, false}, {3u, true}, {0u, true}}) {
    const RestirReceiver receiver(points, environment);
    const double expected = receiver.expected();
    for (const uint candidates : {1u, 4u, 8u}) {
      double sum = 0.0;
      const uint trials = 200000;
      for (uint t = 0; t < trials; ++t) sum += ptLuminance(receiver.estimate(pcgHash(t + 0x51u), candidates));
      const double mean = sum / trials;
      if (expected == 0.0) {
        if (mean != 0.0) return format("no lights: estimate %g", mean);
        continue;
      }
      if (std::abs(mean - expected) > 0.01 * expected)
        return format("%g point lights, environment %g", points, environment ? 1.0 : 0.0) +
               format(", M %g: estimate %g", candidates, mean) + format(", quadrature %g", expected);
    }
  }
  return {};
}

namespace {
// A floor under a point light, seen from above, optionally with a blocker between them.
Builder restirFloor(bool blocked, uint reuse, uint candidates = 8) {
  Builder b;
  b.floor(4.0f, 0.0f, b.material(float3(0.7f), 0.0f, 0.6f));
  if (blocked) b.ceiling(8.0f, 1.0f, b.material(float3(0.2f), 0.0f, 0.9f));
  Light light{};
  light.position = float4(0.3f, 2.0f, -0.2f, 0.0f);
  light.color = float4(1.0f, 1.0f, 1.0f, 10.0f);
  light.direction = float4(0.0f, -1.0f, 0.0f, 1.0f);
  light.cone = float4(1.0f, 0.0f, 0.0f, 0.0f);
  b.frame.lights.push_back(light);
  b.environment([](float3) { return float3(0.0f); }, 8, 4);
  b.camera(float3(0.0f, 0.8f, 1.5f), float3(0.0f), 0.9f, 24, 16);
  b.finish(2);
  b.frame.uniforms.distribution = float4(0.0f);
  b.frame.uniforms.estimator = uint4(1u, candidates, reuse, 0u);
  return b;
}
}  // namespace

// Reuse validity: a neighbour lends its reservoir only with the same material,
// shading normals within 0.9 and a relative depth within 0.1 of the owner's.
PT_TEST(restir_similarity_rejects_depth_normal_and_material) {
  auto record = [](float depth, float3 normal, uint material) {
    PtRestirSurface s{};
    s.positionDepth = float4(0.0f, 0.0f, -depth, depth);
    s.normalMaterial = float4(normal, as_type<float>(material));
    return s;
  };
  const PtRestirSurface owner = record(10.0f, float3(0.0f, 1.0f, 0.0f), 3u);
  if (!ptRestirSimilar(owner, record(10.5f, float3(0.0f, 1.0f, 0.0f), 3u), owner.positionDepth.w))
    return "a similar neighbour was rejected";
  if (ptRestirSimilar(owner, record(12.0f, float3(0.0f, 1.0f, 0.0f), 3u), owner.positionDepth.w))
    return "a neighbour 20% deeper was accepted";
  if (ptRestirSimilar(owner, record(10.0f, normalize(float3(0.5f, 1.0f, 0.0f)), 3u), owner.positionDepth.w))
    return "a neighbour with a normal 27 degrees away was accepted";
  if (ptRestirSimilar(owner, record(10.0f, float3(0.0f, 1.0f, 0.0f), 4u), owner.positionDepth.w))
    return "a neighbour of another material was accepted";
  if (ptRestirSimilar(owner, record(0.0f, float3(0.0f, 1.0f, 0.0f), 3u), owner.positionDepth.w))
    return "a neighbour without a surface was accepted";
  // Temporal reuse compares the reprojected depth: the same point seen at another depth.
  if (!ptRestirSimilar(owner, record(20.0f, float3(0.0f, 1.0f, 0.0f), 3u), 20.5f))
    return "a reprojected neighbour within 10% was rejected";
  return {};
}

// An occluded reservoir sample adds nothing: under a blocker every pixel's direct light is
// zero although its reservoirs hold samples with positive weight.
PT_TEST(restir_occluded_samples_contribute_zero) {
  // The camera sits below the blocker (y 0.8 < 1), the light above it.
  Builder b = restirFloor(true, 3u);
  RestirState state;
  CpuTracer::restirFrame(b.scene, b.frame, 0, state, sharedPool());
  CpuTracer::restirFrame(b.scene, b.frame, 1, state, sharedPool());
  uint held = 0;
  for (std::size_t p = 0; p < state.direct.size(); ++p) {
    if (ptMaxComponent(state.direct[p]) != 0.0f) return format("pixel %g received light through the blocker", double(p));
    if (state.previousReservoirs[p].weight > 0.0f) ++held;
  }
  if (held == 0) return "no reservoir held a sample";
  return {};
}

// Identical seeds give identical reservoirs, and invalidated history resets deterministically:
// a frame after an invalidation equals the same frame without history.
PT_TEST(restir_reservoirs_are_deterministic_and_reset) {
  Builder b = restirFloor(false, 3u);
  auto frames = [&](uint first, uint last, bool invalidateBeforeLast) {
    RestirState state;
    for (uint s = first; s <= last; ++s) {
      if (invalidateBeforeLast && s == last) state.previous.image.z = 0u;
      CpuTracer::restirFrame(b.scene, b.frame, s, state, sharedPool());
    }
    return state;
  };
  auto same = [](const RestirState &a, const RestirState &c) {
    return std::memcmp(a.previousReservoirs.data(), c.previousReservoirs.data(),
                       a.previousReservoirs.size() * sizeof(PtReservoir)) == 0 &&
           std::memcmp(a.direct.data(), c.direct.data(), a.direct.size() * sizeof(float3)) == 0;
  };
  if (!same(frames(0, 3, false), frames(0, 3, false))) return "identical seeds gave different reservoirs";
  if (!same(frames(0, 3, true), frames(3, 3, false))) return "invalidated history was not reset";
  if (same(frames(0, 3, false), frames(3, 3, false))) return "temporal reuse had no effect";
  return {};
}

// ReSTIR converges to next-event estimation: a floor lit by a point light, a sun and an
// emissive quad, with no occluder (so reuse is unbiased), the mean radiance over independent
// short chains (32 seeds x 32 frames) within 1% of NEE's for RIS only and within 2% with
// temporal, spatial and both. Independent short chains: one long static chain correlates its
// frames (a 512-frame chain varied by +-3%), and even 16-seed means of the reuse modes spread
// by +-1.5% around 1.0 (four trials: 1.012, 1.008, 0.988, 0.985).
PT_TEST(restir_converges_to_next_event_estimation) {
  for (uint reuse = 0; reuse < 4; ++reuse) {
    double restir = 0.0, nee = 0.0;
    for (uint seed = 0; seed < 32; ++seed) {
      Builder b = restirFloor(false, reuse);
      b.ceiling(0.5f, 1.5f, b.emissiveMaterial(float3(4.0f, 3.0f, 2.0f)));
      b.scene.instances.back().flags &= ~kInstanceDoubleSided;
      b.sun(float3(0.3f, 1.0f, 0.2f), 2.0f, 0.05f);
      b.finish(2, 0, seed);
      b.frame.uniforms.distribution = float4(0.0f);
      b.frame.uniforms.estimator = uint4(1u, 8u, reuse, 0u);
      for (float v : b.render(32)) restir += v;
      b.frame.uniforms.estimator = uint4(0u, 8u, reuse, 0u);
      for (float v : b.render(32)) nee += v;
    }
    const double tolerance = reuse == 0 ? 0.01 : 0.02;
    if (std::abs(restir - nee) > tolerance * nee) return format("reuse %g: ReSTIR/NEE %g", reuse, restir / nee);
  }
  return {};
}

// V8 image files: OpenEXR round trip and PFM identity.

// Every bit pattern survives an OpenEXR round trip: finite values of every magnitude, both
// zeros, denormals, infinities and NaN payloads, on an odd-sized image; the header names the
// channels, the Rec. 709 chromaticities and the content.
PT_TEST(exr_round_trip_is_bitwise) {
  const uint width = 7, height = 5;
  std::vector<float> rgb(width * height * 3);
  const std::uint32_t specials[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x7F7FFFFFu, 0xFF7FFFFFu,
                                    0x7F800000u, 0xFF800000u, 0x7FC00001u, 0xFFBFFFFFu, 0x3F800000u};
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    std::uint32_t bits = pcgHash(static_cast<uint>(i) * 2654435761u + 7u);
    if (i < std::size(specials)) bits = specials[i];
    std::memcpy(&rgb[i], &bits, sizeof(bits));
  }
  const std::string path = "pt-tests-roundtrip.exr";
  if (!writeLinearImage(path, rgb, width, height, "raw-linear-rgb")) return "could not write the OpenEXR file";
  const LinearImage image = readExr(path);
  std::remove(path.c_str());
  if (image.width != width || image.height != height) return format("read %g x %g", image.width, image.height);
  if (image.channels != std::vector<std::string>{"B", "G", "R"}) return "channels are not B, G, R";
  if (std::memcmp(image.rgb.data(), rgb.data(), rgb.size() * sizeof(float)) != 0) return "pixels changed in the round trip";
  bool labelled = false;
  for (const auto &[name, value] : image.strings) labelled = labelled || (name == "basaltContent" && value == "raw-linear-rgb");
  if (!labelled) return "the content attribute is missing";
  return {};
}

// PFM through writeLinearImage is byte for byte the historical capture format, and a path's
// extension alone chooses the format.
PT_TEST(pfm_output_is_unchanged) {
  const uint width = 3, height = 2;
  std::vector<float> rgb(width * height * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = 0.25f * static_cast<float>(i) - 1.0f;
  const std::string path = "pt-tests-identity.pfm";
  if (!writeLinearImage(path, rgb, width, height)) return "could not write the PFM file";
  std::ifstream file(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(file)), {});
  file.close();
  std::remove(path.c_str());
  std::string expected = "PF\n3 2\n-1.0\n";
  expected.append(reinterpret_cast<const char *>(rgb.data()), rgb.size() * sizeof(float));
  if (bytes != expected) return "the PFM bytes differ from the historical format";
  if (!isExrPath("a.EXR") || isExrPath("a.pfm") || isExrPath("exr")) return "the extension test is wrong";
  return {};
}

// Emitter selection and pdf lookup are binary searches; they must return exactly what the
// linear walks they replaced returned, including ties, the clamped tail and missing keys.
PT_TEST(emissive_binary_searches_match_linear_scans) {
  std::vector<PtEmissiveTriangle> list;
  uint state = 12345u;
  float cdf = 0.0f;
  for (uint instance = 0; instance < 40; ++instance)
    for (uint primitive = 0; primitive < 50; primitive += 1u + (pcgHash(instance * 97u + primitive) % 3u)) {
      PtEmissiveTriangle t{};
      state = pcgHash(state);
      // Some zero-width steps: equal consecutive CDF values must pick the first.
      cdf += (state % 7u == 0u) ? 0.0f : static_cast<float>(state % 1000u + 1u);
      t.edge2Cdf.w = cdf;
      t.identity = uint4(instance, primitive, 0u, 0u);
      list.push_back(t);
    }
  for (PtEmissiveTriangle &t : list) t.edge2Cdf.w /= cdf;
  list.back().edge2Cdf.w = 0.999f;  // a tail below one: numbers above it clamp to the last entry
  const uint count = static_cast<uint>(list.size());
  for (uint i = 0; i < 200000; ++i) {
    const float xi = hashFloat(pcgHash(i * 2654435761u));
    uint linear = 0u;
    while (linear + 1u < count && xi > list[linear].edge2Cdf.w) ++linear;
    if (ptEmissiveSelect(list.data(), count, xi) != linear)
      return "selection differs at xi " + std::to_string(xi) + ": binary " +
             std::to_string(ptEmissiveSelect(list.data(), count, xi)) + ", linear " + std::to_string(linear);
  }
  for (const float xi : {0.0f, 1.0f, 0.9995f, list[0].edge2Cdf.w, list[count / 2].edge2Cdf.w}) {
    uint linear = 0u;
    while (linear + 1u < count && xi > list[linear].edge2Cdf.w) ++linear;
    if (ptEmissiveSelect(list.data(), count, xi) != linear) return format("boundary selection differs at %.9g", xi);
  }
  for (uint instance = 0; instance < 42; ++instance)
    for (uint primitive = 0; primitive < 52; ++primitive) {
      uint linear = count;
      for (uint i = 0; i < count; ++i)
        if (list[i].identity.x == instance && list[i].identity.y == primitive) { linear = i; break; }
      if (ptEmissiveFind(list.data(), count, instance, primitive) != linear)
        return "lookup of (" + std::to_string(instance) + ", " + std::to_string(primitive) + ") differs";
    }
  if (ptEmissiveSelect(list.data(), 1u, 0.7f) != 0u || ptEmissiveFind(list.data(), 0u, 0u, 0u) != 0u)
    return "one-entry and empty lists are not handled";
  // The host builds the list in the order the lookup requires.
  Builder b;
  const uint light = b.emissiveMaterial(float3(4.0f));
  b.sphere(float3(0.0f), 1.0f, light, 16);
  b.sphere(float3(3.0f, 0.0f, 0.0f), 1.0f, b.material(float3(0.5f), 0, 0.5f), 16);
  b.sphere(float3(6.0f, 0.0f, 0.0f), 1.0f, light, 16);
  b.finish(1u);
  const auto &built = b.scene.emissiveTriangles;
  if (built.empty()) return "the fixture built no emitters";
  for (std::size_t i = 1; i < built.size(); ++i)
    if (!(built[i - 1].identity.x < built[i].identity.x ||
          (built[i - 1].identity.x == built[i].identity.x && built[i - 1].identity.y < built[i].identity.y)))
      return "the emitter list is not in strictly increasing (instance, primitive) order";
  return {};
}

// Wavefront capacity planning: whole pixels per batch, every device limit applied, checked
// 64-bit sizes, explicit errors instead of truncation, and exact coverage of every path.
PT_TEST(wavefront_capacity_plan_bounds) {
  const WavefrontLimits rtx{1ull << 32, 2147483647u, 1u << 30, kWavefrontBudgetBytes};
  // 1600x900 at 16 samples: 23.04 M paths in whole-pixel batches of the default capacity.
  WavefrontPlan plan = planWavefront(1600ull * 900, 16, 0, 0, rtx);
  if (!plan.error.empty() || plan.capacity % 16 != 0 || plan.capacity > kWavefrontDefaultPaths)
    return "the default plan is not a whole-pixel batch within the default capacity";
  if (static_cast<std::uint64_t>(plan.batches) * plan.capacity < 1600ull * 900 * 16 ||
      static_cast<std::uint64_t>(plan.batches - 1) * plan.capacity >= 1600ull * 900 * 16)
    return "the batches do not cover the frame's paths exactly";
  if (plan.bytes != static_cast<std::uint64_t>(plan.capacity) * kWavefrontPathBytes + kWavefrontFixedBytes)
    return "the byte count is not the per-path size times the capacity";
  // A small frame fits one batch of exactly its paths: zero waste.
  plan = planWavefront(17ull * 5, 3, 0, 85, rtx);
  if (plan.capacity != 255 || plan.batches != 1) return "a small frame is not one exact batch";
  // One pixel, one sample: the smallest queue.
  plan = planWavefront(1, 1, 0, 0, rtx);
  if (plan.capacity != 1 || plan.batches != 1) return "a one-path frame is not one path";
  // A requested capacity that is not a multiple of S rounds down to whole pixels.
  plan = planWavefront(1000, 7, 100, 0, rtx);
  if (plan.capacity != 98 || plan.batches != 72 || plan.limitedBy != "requested capacity")
    return "a requested capacity was not rounded to whole pixels";
  // Batch sizes for a smaller final-frame S never exceed the allocation.
  if (wavefrontBatchPaths(98, 3) != 96 || wavefrontBatchPaths(98, 7) != 98 || wavefrontBatchPaths(98, 0) != 0)
    return "tail batches are not whole pixels within the allocation";
  // Vulkan's guaranteed minima bound the queue: 2^27 / 48 paths per array.
  plan = planWavefront(3840ull * 2160, 64, 1ull << 30, 0, WavefrontLimits{});
  if (plan.limitPaths != (1ull << 27) / kWavefrontLargestElement || plan.limitedBy != "maxStorageBufferRange" ||
      static_cast<std::uint64_t>(plan.capacity) * kWavefrontLargestElement > (1ull << 27))
    return "maxStorageBufferRange did not bound the largest queue array";
  WavefrontLimits groups = rtx;
  groups.maxWorkGroupCountX = 100;
  plan = planWavefront(1u << 20, 1, 1ull << 30, 0, groups);
  if (plan.capacity != 6400 || plan.limitedBy != "maxComputeWorkGroupCount[0]")
    return "the dispatch group limit did not bound the queue";
  WavefrontLimits rays = rtx;
  rays.maxRayDispatchInvocations = 1000;
  plan = planWavefront(1u << 20, 4, 1ull << 30, 0, rays);
  if (plan.capacity != 1000 || plan.limitedBy != "maxRayDispatchInvocationCount")
    return "the ray dispatch limit did not bound the queue";
  // The allocation budget, including per-pixel guides, is never exceeded.
  WavefrontLimits budget = rtx;
  budget.budgetBytes = 1ull << 20;
  plan = planWavefront(1u << 20, 1, 0, 4096, budget);
  if (plan.bytes > budget.budgetBytes || plan.limitedBy != "allocation budget") return "the budget was exceeded";
  // Explicit errors: no pixels, no samples, more samples than any batch, 32-bit overflow.
  if (planWavefront(0, 4, 0, 0, rtx).error.empty() || planWavefront(4, 0, 0, 0, rtx).error.empty())
    return "an empty frame was planned";
  if (planWavefront(4, 64, 32, 0, rtx).error.empty()) return "a pixel's samples exceeding capacity was planned";
  if (planWavefront(1ull << 33, 1, 0, 0, rtx).error.empty()) return "a 33-bit pixel count was planned";
  budget.budgetBytes = 100;
  if (planWavefront(4, 1, 0, 1000, budget).error.empty()) return "guides larger than the budget were planned";
  return {};
}

// Environment sun extraction (pt/EnvironmentSun.h). The images are built here from a direction
// and an angular radius; the checks are the energy bookkeeping and the recovered geometry.
struct SunImage {
  uint width, height;
  std::vector<float> rgba;
};

double sunTexelSolidAngle(uint y, uint width, uint height) {
  const double theta = (y + 0.5) / height * 3.14159265358979323846;
  return (2.0 * 3.14159265358979323846 / width) * (3.14159265358979323846 / height) * std::sin(theta);
}

std::array<double, 3> sunTexelDirection(uint x, uint y, uint width, uint height) {
  const double pi = 3.14159265358979323846;
  const double theta = (y + 0.5) / height * pi, phi = ((x + 0.5) / width - 0.5) * 2.0 * pi;
  return {std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
}

// A sky that brightens towards the zenith, plus discs of the given radiance and radius.
struct SunDisc {
  std::array<double, 3> direction;
  double radiusDegrees;
  std::array<float, 3> radiance;
};
SunImage sunImage(uint width, uint height, const std::vector<SunDisc> &discs) {
  SunImage image{width, height, std::vector<float>(static_cast<std::size_t>(width) * height * 4)};
  for (uint y = 0; y < height; ++y)
    for (uint x = 0; x < width; ++x) {
      const auto d = sunTexelDirection(x, y, width, height);
      float *texel = &image.rgba[(static_cast<std::size_t>(y) * width + x) * 4];
      const float sky = 0.1f + 0.2f * static_cast<float>(std::max(0.0, d[1]));
      texel[0] = sky * 0.8f; texel[1] = sky * 0.9f; texel[2] = sky; texel[3] = 1.0f;
      for (const SunDisc &disc : discs) {
        const double c = d[0] * disc.direction[0] + d[1] * disc.direction[1] + d[2] * disc.direction[2];
        if (c >= std::cos(disc.radiusDegrees * 3.14159265358979323846 / 180.0))
          for (int k = 0; k < 3; ++k) texel[k] = disc.radiance[k];
      }
    }
  return image;
}

std::array<double, 3> sunImagePower(const SunImage &image) {
  std::array<double, 3> power{};
  for (uint y = 0; y < image.height; ++y)
    for (uint x = 0; x < image.width; ++x)
      for (int k = 0; k < 3; ++k)
        power[k] += image.rgba[(static_cast<std::size_t>(y) * image.width + x) * 4 + k] *
                    sunTexelSolidAngle(y, image.width, image.height);
  return power;
}

std::array<double, 3> sunUnit(double x, double y, double z) {
  const double n = std::sqrt(x * x + y * y + z * z);
  return {x / n, y / n, z / n};
}

double sunAngleDegrees(float3 a, std::array<double, 3> b) {
  const double c = a.x * b[0] + a.y * b[1] + a.z * b[2];
  return std::acos(std::clamp(c, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
}

// What leaves the image is what the disc carries, channel by channel; the direction and the
// radius are the disc's.
PT_TEST(environment_sun_moves_the_disc_energy) {
  const std::array<double, 3> towards = sunUnit(0.4, 0.7, -0.3);
  SunImage image = sunImage(1024, 512, {{towards, 2.0, {2000.0f, 1900.0f, 1700.0f}}});
  const std::array<double, 3> before = sunImagePower(image);
  const EnvironmentSun sun = extractEnvironmentSun(image.rgba, image.width, image.height);
  if (!sun.found) return "the disc was not found";
  const std::array<double, 3> after = sunImagePower(image);
  const double irradiance[3] = {sun.irradiance.x, sun.irradiance.y, sun.irradiance.z};
  for (int k = 0; k < 3; ++k)
    if (std::abs((before[k] - after[k]) - irradiance[k]) > 1e-5 * irradiance[k])
      return format("channel %.0f: the image lost %.9g but the sun carries %.9g", k, before[k] - after[k], irradiance[k]);
  // A 2 degree disc of radiance 1900 (green) holds 1900 * 2 pi (1 - cos 2 deg); the sky behind
  // it stays in the image.
  const double expected = (1900.0 - 0.9 * (0.1 + 0.2 * towards[1])) * 2.0 * 3.14159265358979323846 *
                          (1.0 - std::cos(2.0 * 3.14159265358979323846 / 180.0));
  if (std::abs(irradiance[1] / expected - 1.0) > 0.03)
    return format("green irradiance %.6g, expected about %.6g", irradiance[1], expected);
  if (sunAngleDegrees(sun.direction, towards) > 0.1) return format("direction off by %.3f degrees", sunAngleDegrees(sun.direction, towards));
  const double radius = sun.angularRadius * 180.0 / 3.14159265358979323846;
  if (radius < 1.8 || radius > 2.2) return format("angular radius %.3f degrees, expected 2", radius);
  for (std::size_t i = 0; i < image.rgba.size(); i += 4)
    if (image.rgba[i + 1] > 1.0f) return "a disc texel is left in the image";
  return {};
}

// No region stands out: the image is untouched, bit for bit, and there is no sun.
PT_TEST(environment_sun_leaves_a_plain_sky_alone) {
  SunImage image = sunImage(512, 256, {{sunUnit(0.0, 1.0, 0.0), 30.0, {0.35f, 0.4f, 0.45f}}});
  const std::vector<float> original = image.rgba;
  const EnvironmentSun sun = extractEnvironmentSun(image.rgba, image.width, image.height);
  if (sun.found) return "a sun was found in a plain sky";
  if (std::memcmp(original.data(), image.rgba.data(), original.size() * sizeof(float)) != 0)
    return "the image changed";
  if (sun.irradiance.x != 0.0f || sun.irradiance.y != 0.0f || sun.irradiance.z != 0.0f) return "a sunless result carries light";
  return {};
}

// A sun on the image's left and right edge is one region: longitude wraps.
PT_TEST(environment_sun_wraps_across_the_seam) {
  const std::array<double, 3> towards = sunUnit(-1.0, 0.3, 0.0);  // phi = pi: u = 0 and 1
  SunImage image = sunImage(1024, 512, {{towards, 1.5, {5000.0f, 5000.0f, 5000.0f}}});
  const std::array<double, 3> before = sunImagePower(image);
  const EnvironmentSun sun = extractEnvironmentSun(image.rgba, image.width, image.height);
  if (!sun.found) return "the disc was not found";
  if (sunAngleDegrees(sun.direction, towards) > 0.1) return format("direction off by %.3f degrees", sunAngleDegrees(sun.direction, towards));
  const std::array<double, 3> after = sunImagePower(image);
  if (std::abs((before[0] - after[0]) - sun.irradiance.x) > 1e-5 * sun.irradiance.x) return "energy was not conserved";
  for (uint y = 0; y < image.height; ++y)
    for (uint x : {0u, image.width - 1u})
      if (image.rgba[(static_cast<std::size_t>(y) * image.width + x) * 4] > 1.0f)
        return format("the disc is left at the seam, x %.0f", x);
  return {};
}

// A lone hot texel far brighter than any one sun texel does not win over a sun a few texels
// wide; the brightest neighbourhood does.
PT_TEST(environment_sun_prefers_a_disc_to_a_hot_pixel) {
  const std::array<double, 3> towards = sunUnit(0.2, 0.5, 0.8);
  SunImage image = sunImage(2048, 1024, {{towards, 0.4, {3000.0f, 3000.0f, 3000.0f}}});
  float *hot = &image.rgba[(static_cast<std::size_t>(300) * image.width + 100) * 4];
  hot[0] = hot[1] = hot[2] = 20000.0f;
  const EnvironmentSun sun = extractEnvironmentSun(image.rgba, image.width, image.height);
  if (!sun.found) return "no sun was found";
  if (sunAngleDegrees(sun.direction, towards) > 0.2) return format("the sun was placed %.2f degrees from the disc", sunAngleDegrees(sun.direction, towards));
  if (hot[1] != 20000.0f) return "the hot texel was taken as the sun";
  return {};
}

// The path tracer's disc: the extracted irradiance spread over the cone it samples.
PT_TEST(environment_sun_uniforms_carry_the_irradiance) {
  float4 direction, radiance;
  environmentSunUniforms(float3(0.0f, 2.0f, 0.0f), float3(3.0f, 2.0f, 1.0f), 0.01f, direction, radiance);
  if (std::abs(direction.y - 1.0f) > 1e-6f || std::abs(direction.w - std::cos(0.01f)) > 1e-7f) return "wrong cone";
  const double solidAngle = 2.0 * 3.14159265358979323846 * (1.0 - std::cos(0.01));
  if (std::abs(radiance.w / solidAngle - 1.0) > 1e-5) return "wrong solid angle";
  if (std::abs(radiance.x * radiance.w / 3.0 - 1.0) > 1e-5 || std::abs(radiance.z * radiance.w - 1.0) > 1e-5)
    return "radiance times solid angle is not the irradiance";
  environmentSunUniforms(float3(0.0f, 1.0f, 0.0f), float3(0.0f), 0.01f, direction, radiance);
  if (direction.w != 0.0f || radiance.w != 0.0f) return "a zero irradiance still made a sun";
  return {};
}

} // namespace

int main(int argc, char **argv) {
  int failures = 0;
  int skipped = 0;
  int selected = 0;
  for (const Test &test : registry()) {
    if (argc > 1 && std::string(test.name).find(argv[1]) == std::string::npos) continue;
    ++selected;
    const std::string why = test.run();
    const bool skip = why.starts_with("SKIP:");
    std::printf("%s %s%s%s\n", skip ? "SKIP" : why.empty() ? "PASS" : "FAIL", test.name,
                why.empty() ? "" : ": ", skip ? why.c_str() + 5 : why.c_str());
    std::fflush(stdout);
    skipped += skip ? 1 : 0;
    failures += why.empty() || skip ? 0 : 1;
  }
  std::printf("%d passed, %d skipped, %d failed\n", selected - failures - skipped, skipped, failures);
  return failures == 0 ? 0 : 1;
}
