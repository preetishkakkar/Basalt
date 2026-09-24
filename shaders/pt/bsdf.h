// The glTF metallic-roughness BSDF, with the rasteriser's terms (shared/shading.h): a GGX
// lobe with height-correlated Smith visibility and Schlick Fresnel, and a Lambert lobe
// weighted by one minus Fresnel. Sampled by the visible normals of the GGX distribution
// (Heitz 2018) and by the cosine, one lobe chosen per sample, weighted by both pdfs.
//
// Multiple scattering: single-scattering GGX loses energy at high roughness. The specular
// lobe is scaled by 1 + F0 (1 / E - 1), E its directional albedo with Fresnel at one
// (Kulla and Conty 2017, as Filament applies it), read from a table integrated on the host,
// so a white rough metal reflects all it receives. A transmissive base restores the
// energy of its reflection and transmission lobes together, from the thin-walled lobe's E or
// a closed dielectric interface's albedo A, tabulated the same way.
#pragma once
#include "path.h"
#include "surface.h"

// Directional albedo E(cos view, roughness) of the GGX lobe with Fresnel at one.
PT_CONSTANT uint kAlbedoTableSize = 32u;
// After E, the albedo A(cos view, roughness) of a closed dielectric interface over a white
// base: its reflection lobe with Schlick F plus its refraction lobe with 1 - F, in importance
// (flux) mode. One kAlbedoTableSize-squared slice per IOR, 1 to kDielectricIorMax in
// kDielectricIorSteps; entering (etap = IOR) slices first, then leaving (etap = 1 / IOR).
PT_CONSTANT uint kDielectricIorSteps = 16u;
PT_CONSTANT float kDielectricIorMax = 3.0f;

inline float ptAlbedoLookup(device const float *table, uint offset, float normalDotView, float roughness) {
  const float scale = float(kAlbedoTableSize - 1u);
  const float x = saturate(normalDotView) * scale;
  const float y = saturate(roughness) * scale;
  const uint x0 = min(uint(x), kAlbedoTableSize - 2u);
  const uint y0 = min(uint(y), kAlbedoTableSize - 2u);
  const float fx = x - float(x0);
  const float fy = y - float(y0);
  const float a = table[offset + y0 * kAlbedoTableSize + x0];
  const float b = table[offset + y0 * kAlbedoTableSize + x0 + 1u];
  const float c = table[offset + (y0 + 1u) * kAlbedoTableSize + x0];
  const float d = table[offset + (y0 + 1u) * kAlbedoTableSize + x0 + 1u];
  return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

inline float ptSpecularAlbedo(device const float *table, float normalDotView, float roughness) {
  return ptAlbedoLookup(table, 0u, normalDotView, roughness);
}

// A(cos view, roughness, IOR) of a closed dielectric, entering or leaving it.
inline float ptDielectricAlbedo(device const float *table, float normalDotView, float roughness, float ior,
                                bool leaving) {
  const uint slice = kAlbedoTableSize * kAlbedoTableSize;
  const uint first = slice * (1u + (leaving ? kDielectricIorSteps : 0u));
  const float z = saturate((ior - 1.0f) / (kDielectricIorMax - 1.0f)) * float(kDielectricIorSteps - 1u);
  const uint z0 = min(uint(z), kDielectricIorSteps - 2u);
  return mix(ptAlbedoLookup(table, first + z0 * slice, normalDotView, roughness),
             ptAlbedoLookup(table, first + (z0 + 1u) * slice, normalDotView, roughness), z - float(z0));
}

struct PtBsdf {
  float3 normal;
  float3 tangent;
  float3 bitangent;
  float3 view;          // towards the viewer
  float3 diffuseColor;
  float3 f0;
  float3 compensation;  // multiple-scattering scale on the specular lobe
  float alpha;          // roughness squared
  float normalDotView;
  float specularProbability;
  // transmission: x factor T (the base's non-specular
  // branch splits T to transmission, 1 - T to diffuse), y etap = index on the far side over
  // the view side (1: no interface), z 1 thin-walled / 0 closed, w the transmission lobe's
  // multiple-scattering compensation (1 / E thin-walled, 1 / A closed). clearcoat: x factor,
  // y coat alpha, z probability of sampling the coat, w coat normal . view.
  float4 transmission;
  float4 clearcoat;
  float3 clearcoatNormal;
};

inline float ptSchlick(float f0, float cosine) {
  const float m = saturate(1.0f - cosine);
  return f0 + (1.0f - f0) * m * m * m * m * m;
}

// Schlick's Fresnel at an interface with relative index etap (far side over near side):
// from the dense side (etap < 1) it uses the transmitted angle, and is 1 beyond the critical
// angle (total internal reflection). etap = 1 is plain Schlick.
inline float3 ptFresnelInterface(float3 f0, float cosine, float etap) {
  if (etap < 1.0f) {
    const float sinTransmittedSquared = (1.0f - cosine * cosine) / (etap * etap);
    if (sinTransmittedSquared >= 1.0f) return float3(1.0f);
    return fresnelSchlick(f0, sqrt(1.0f - sinTransmittedSquared));
  }
  return fresnelSchlick(f0, cosine);
}

inline PtBsdf ptMakeBsdf(thread const PtSurface &surface, float3 view, device const float *albedoTable) {
  PtBsdf bsdf;
  bsdf.normal = surface.normal;
  float3 tangent = float3(0.0f);
  float3 bitangent = float3(0.0f);
  orthonormalBasis(surface.normal, tangent, bitangent);
  bsdf.tangent = tangent;
  bsdf.bitangent = bitangent;
  bsdf.view = view;
  bsdf.diffuseColor = surface.baseColor * (1.0f - surface.metallic);
  // KHR_materials_ior sets the dielectric F0; the default 1.5 keeps the V6 constant exactly.
  const float ior = surface.transmission.y;
  const float dielectricF0 = ior == 1.5f ? 0.04f : ((ior - 1.0f) / (ior + 1.0f)) * ((ior - 1.0f) / (ior + 1.0f));
  bsdf.f0 = mix(float3(dielectricF0), surface.baseColor, surface.metallic);
  // A closed dielectric is an interface: the far side is glass when entering, air leaving.
  bsdf.transmission = float4(surface.transmission.x, 1.0f, surface.transmission.z, 0.0f);
  if (surface.transmission.x > 0.0f && surface.transmission.z < 0.5f)
    bsdf.transmission.y = surface.transmission.w > 0.5f ? ior : 1.0f / ior;
  bsdf.clearcoat = float4(0.0f);
  bsdf.clearcoatNormal = surface.clearcoatNormal;
  if (surface.clearcoat.x > 0.0f) {
    const float coatView = max(dot(surface.clearcoatNormal, view), 1e-4f);
    bsdf.clearcoat = float4(surface.clearcoat.x, surface.clearcoat.y * surface.clearcoat.y,
                            clamp(surface.clearcoat.x * ptSchlick(0.04f, coatView), 0.05f, 0.95f), coatView);
  }
  bsdf.alpha = surface.roughness * surface.roughness;
  bsdf.normalDotView = max(dot(surface.normal, view), 1e-4f);
  const float albedo = max(ptSpecularAlbedo(albedoTable, bsdf.normalDotView, surface.roughness), 1e-3f);
  bsdf.compensation = float3(1.0f) + bsdf.f0 * (1.0f / albedo - 1.0f);
  // Transmission carries 1 - F on a microfacet lobe, where the opaque base hands 1 - F to its
  // (lossless) diffuse lobe, so the energy lost between microfacets is restored to both lobes
  // together: scaled by 1 / (their joint single-scattering albedo), a white base at T = 1
  // passes on exactly what it receives. Thin-walled transmission is the reflection lobe
  // mirrored, so that albedo is E; a closed dielectric's is the interface's A. The reflection
  // lobe blends from the V6 compensation (T = 0) to the joint one (T = 1).
  bsdf.transmission.w = 1.0f;
  if (bsdf.transmission.x > 0.0f) {
    float joint = albedo;
    if (bsdf.transmission.z < 0.5f)
      joint = max(ptDielectricAlbedo(albedoTable, bsdf.normalDotView, surface.roughness, ior,
                                     surface.transmission.w < 0.5f), 1e-3f);
    bsdf.transmission.w = 1.0f / joint;
    bsdf.compensation = bsdf.compensation + (float3(bsdf.transmission.w) - bsdf.compensation) * bsdf.transmission.x;
  }
  // One lobe per sample, chosen by an estimate of how much each reflects from this view.
  float3 fresnel = fresnelSchlick(bsdf.f0, bsdf.normalDotView);
  // Through an interface the estimate uses its Fresnel, which is one beyond the critical
  // angle; clamped, so microfacets that still refract keep a transmission pdf.
  if (bsdf.transmission.x > 0.0f) fresnel = ptFresnelInterface(bsdf.f0, bsdf.normalDotView, bsdf.transmission.y);
  const float specular = ptLuminance(fresnel);
  const float diffuse = ptLuminance(bsdf.diffuseColor * (float3(1.0f) - fresnel));
  float probability = 1.0f;
  if (diffuse > 0.0f) probability = clamp(specular / (specular + diffuse), 0.05f, 0.95f);
  else if (bsdf.transmission.x > 0.0f) probability = 0.95f;
  bsdf.specularProbability = probability;
  return bsdf;
}

// The GGX distribution D of the angle between `normal` and `halfVector`, exact at the peak.
// shared/shading.h's distributionGGX (the rasteriser's) takes n.h, so its d = 1 - (n.h)^2
// (1 - alpha^2) cancels: near the peak 1 - (n.h)^2 is a few float steps from alpha^2 at the
// path tracer's smoothest roughness, a 1.7% energy gain, and it floors pi d^2 at 1e-7,
// cutting the peak below roughness ~0.115. Here sin^2 = |n x h|^2 keeps its precision, and
// d >= alpha^2 cos^2 needs no floor (h is never perpendicular to n where D is used).
inline float ptDistributionGGX(float3 normal, float3 halfVector, float alpha) {
  const float a2 = alpha * alpha;
  const float3 perpendicular = cross(normal, halfVector);
  const float cosine = dot(normal, halfVector);
  const float d = dot(perpendicular, perpendicular) + cosine * cosine * a2;
  return a2 / (kPi * d * d);
}

// Smith G1 for GGX, matching the height-correlated visibility's view term.
inline float ptSmithG1(float cosine, float alpha) {
  const float a2 = alpha * alpha;
  return 2.0f * cosine / (cosine + sqrt(a2 + (1.0f - a2) * cosine * cosine));
}

// The density, per solid angle of `direction`, of reflecting `view` about a visible GGX
// normal (Heitz 2018): D_v(h) / (4 v.h) = D(h) G1(v) / (4 n.v), wherever `direction` lies;
// zero when its half vector is not a visible normal.
inline float ptReflectionPdf(float3 normal, float3 view, float normalDotView, float alpha, float3 direction) {
  const float3 halfVector = normalize(view + direction);
  const float normalDotHalf = dot(normal, halfVector);
  if (normalDotHalf <= 0.0f || dot(view, halfVector) <= 0.0f) return 0.0f;
  return ptDistributionGGX(normal, halfVector, alpha) * ptSmithG1(normalDotView, alpha) / (4.0f * normalDotView);
}

// f(v, l) cos(l), and the pdf the sampler below would have chosen l with: the density of
// every lobe at l, including a lobe's samples that cross the surface (a reflection below the
// horizon, a mirrored thin-walled transmission above it), which transmission makes lit.
inline float3 ptBsdfEvaluate(thread const PtBsdf &bsdf, float3 geometricNormal, float3 light, thread float &pdf) {
  pdf = 0.0f;
  const float normalDotLight = dot(bsdf.normal, light);
  float3 value = float3(0.0f);
  if (bsdf.transmission.x > 0.0f && normalDotLight < 0.0f && dot(geometricNormal, light) < 0.0f) {
    // Transmission to the far side: thin-walled mirrors the reflection lobe through the
    // tangent plane; a closed dielectric refracts (Walter et al. 2007, as in pbrt-v4).
    const float branch = (1.0f - bsdf.specularProbability) * bsdf.transmission.x;
    const float etap = bsdf.transmission.y;  // 1 when thin-walled
    // One micro-normal for both policies: thin-walled reflects the mirrored light direction,
    // a closed dielectric refracts (generalized half vector). They differ only in two
    // Jacobian factors, so D, G2, F and G1 are evaluated once.
    float3 farSide = light * etap;
    if (bsdf.transmission.z > 0.5f) farSide = light - bsdf.normal * (2.0f * normalDotLight);
    float3 micro = normalize(bsdf.view + farSide);
    if (dot(micro, bsdf.normal) < 0.0f) micro = -micro;
    const float viewDotMicro = dot(bsdf.view, micro);
    const float lightDotMicro = dot(light, micro);
    float valueScale = 1.0f / (4.0f * bsdf.normalDotView);
    float jacobian = 1.0f / (4.0f * max(viewDotMicro, 1e-8f));
    bool valid = viewDotMicro > 0.0f;
    if (bsdf.transmission.z < 0.5f) {
      const float denominator = (lightDotMicro + viewDotMicro / etap) * (lightDotMicro + viewDotMicro / etap);
      valueScale = abs(lightDotMicro * viewDotMicro) / (bsdf.normalDotView * denominator * etap * etap);
      jacobian = abs(lightDotMicro) / denominator;
      if (lightDotMicro >= 0.0f) valid = false;
    }
    // Reflection samples that fall below the horizon arrive here too.
    pdf = bsdf.specularProbability * ptReflectionPdf(bsdf.normal, bsdf.view, bsdf.normalDotView, bsdf.alpha, light);
    if (valid) {
      const float distribution = ptDistributionGGX(bsdf.normal, micro, bsdf.alpha);
      // Height-correlated G2 from the visibility term V = G2 / (4 n.v n.l).
      const float g2 = visibilitySmith(bsdf.normalDotView, -normalDotLight, bsdf.alpha) * 4.0f *
                       bsdf.normalDotView * -normalDotLight;
      value = (float3(1.0f) - ptFresnelInterface(bsdf.f0, viewDotMicro, etap)) * bsdf.diffuseColor *
              (bsdf.transmission.x * bsdf.transmission.w * distribution * g2 * valueScale);
      pdf = pdf + branch * distribution * ptSmithG1(bsdf.normalDotView, bsdf.alpha) * viewDotMicro /
                      bsdf.normalDotView * jacobian;
    }
  } else
  // Both sides of the geometric surface must agree, or a normal map would leak light through it.
  if (normalDotLight > 0.0f && dot(geometricNormal, light) > 0.0f) {
    const float3 halfVector = normalize(bsdf.view + light);
    const float viewDotHalf = saturate(dot(bsdf.view, halfVector));
    const float3 fresnel = ptFresnelInterface(bsdf.f0, viewDotHalf, bsdf.transmission.y);
    const float diffuseShare = 1.0f - bsdf.transmission.x;
    const float distribution = ptDistributionGGX(bsdf.normal, halfVector, bsdf.alpha);
    const float visibility = visibilitySmith(bsdf.normalDotView, normalDotLight, bsdf.alpha);
    const float3 specular = fresnel * (distribution * visibility) * bsdf.compensation;
    const float3 diffuse = (float3(1.0f) - fresnel) * bsdf.diffuseColor * (1.0f / kPi);
    const float specularPdf = distribution * ptSmithG1(bsdf.normalDotView, bsdf.alpha) / (4.0f * bsdf.normalDotView);
    const float diffusePdf = normalDotLight * (1.0f / kPi);
    pdf = bsdf.specularProbability * specularPdf + (1.0f - bsdf.specularProbability) * diffusePdf * diffuseShare;
    value = (specular + diffuse * diffuseShare) * normalDotLight;
    // A thin-walled transmission mirrors its reflection sample, which can land above the horizon.
    if (bsdf.transmission.x > 0.0f && bsdf.transmission.z > 0.5f)
      pdf = pdf + (1.0f - bsdf.specularProbability) * bsdf.transmission.x *
                      ptReflectionPdf(bsdf.normal, bsdf.view, bsdf.normalDotView, bsdf.alpha,
                                      light - bsdf.normal * (2.0f * normalDotLight));
  }
  if (bsdf.clearcoat.x > 0.0f) {
    // The coat reflects c F_c D V; both passages through it attenuate the base by 1 - c F_c.
    const float coatLight = dot(bsdf.clearcoatNormal, light);
    value = value * ((1.0f - bsdf.clearcoat.x * ptSchlick(0.04f, bsdf.clearcoat.w)) *
                     (1.0f - bsdf.clearcoat.x * ptSchlick(0.04f, abs(coatLight))));
    // The coat's samples are lit on either side once the base transmits.
    const float coatPdf = ptReflectionPdf(bsdf.clearcoatNormal, bsdf.view, bsdf.clearcoat.w, bsdf.clearcoat.y, light);
    if (coatLight > 0.0f && dot(geometricNormal, light) > 0.0f) {
      const float3 halfVector = normalize(bsdf.view + light);
      const float distribution = ptDistributionGGX(bsdf.clearcoatNormal, halfVector, bsdf.clearcoat.y);
      value = value + float3(bsdf.clearcoat.x * ptSchlick(0.04f, saturate(dot(bsdf.view, halfVector))) * distribution *
                             visibilitySmith(bsdf.clearcoat.w, coatLight, bsdf.clearcoat.y) * coatLight);
    }
    pdf = bsdf.clearcoat.z * coatPdf + (1.0f - bsdf.clearcoat.z) * pdf;
  }
  return value;
}

// The diffuse share of `total`, the BSDF value in `direction`, for the reconstruction guides.
// Reconstruction only: deliberately separate from the raw evaluator/RNG.
inline float3 ptDiffuseFraction(thread const PtBsdf &bsdf, float3 direction, float3 total) {
  const float3 h = normalize(bsdf.view + direction);
  const float3 fresnel = fresnelSchlick(bsdf.f0, saturate(dot(bsdf.view, h)));
  float3 diffuse = (float3(1.0f) - fresnel) * bsdf.diffuseColor *
                  (max(dot(bsdf.normal, direction), 0.0f) / kPi);
  diffuse = diffuse * (1.0f - bsdf.transmission.x);  // transmission counts as specular
  return saturate(diffuse / max(total, float3(1e-20f)));
}

// A visible normal of the GGX distribution, in the local frame where the normal is +z.
inline float3 ptSampleVisibleNormal(float3 viewLocal, float alpha, float2 xi) {
  const float3 stretched = normalize(float3(alpha * viewLocal.x, alpha * viewLocal.y, viewLocal.z));
  const float lengthSquared = stretched.x * stretched.x + stretched.y * stretched.y;
  float3 t1 = float3(1.0f, 0.0f, 0.0f);
  if (lengthSquared > 0.0f) t1 = float3(-stretched.y, stretched.x, 0.0f) * rsqrt(lengthSquared);
  const float3 t2 = cross(stretched, t1);
  const float radius = sqrt(xi.x);
  const float angle = 2.0f * kPi * xi.y;
  const float p1 = radius * cos(angle);
  const float s = 0.5f * (1.0f + stretched.z);
  const float p2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - p1 * p1)) + s * radius * sin(angle);
  const float3 visible = t1 * p1 + t2 * p2 + stretched * sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2));
  return normalize(float3(alpha * visible.x, alpha * visible.y, max(0.0f, visible.z)));
}

// The transmitted direction through micro-normal `micro`: thin-walled mirrors the reflection
// through the tangent plane (no bending); a closed dielectric refracts by 1 / etap, and gives
// zero beyond the critical angle (total internal reflection).
inline float3 ptTransmittedDirection(thread const PtBsdf &bsdf, float3 micro) {
  if (bsdf.transmission.z > 0.5f) {
    const float3 reflected = reflect(-bsdf.view, micro);
    return reflected - bsdf.normal * (2.0f * dot(bsdf.normal, reflected));
  }
  return refract(-bsdf.view, micro, 1.0f / bsdf.transmission.y);
}

// The lobe the sampler below chooses for xi.z: 0 diffuse, 1 specular reflection, 2 transmission,
// 3 clearcoat. xi.z first chooses the clearcoat (probability clearcoat.z), then the base's
// specular lobe, then within the rest transmission (probability T) or diffuse, each choice
// rescaling xi.z to [0, 1) for the next.
inline uint ptBsdfSampledLobe(thread const PtBsdf &bsdf, float z) {
  float choice = z;
  if (bsdf.clearcoat.x > 0.0f) {
    if (choice < bsdf.clearcoat.z) return 3u;
    choice = (choice - bsdf.clearcoat.z) / (1.0f - bsdf.clearcoat.z);
  }
  if (choice < bsdf.specularProbability) return 1u;
  if (bsdf.transmission.x > 0.0f &&
      (choice - bsdf.specularProbability) / (1.0f - bsdf.specularProbability) < bsdf.transmission.x)
    return 2u;
  return 0u;
}

// The alpha (perceptual roughness squared) of that lobe, for the ray cone's spread; 1 for the
// diffuse lobe.
inline float ptBsdfSampledAlpha(thread const PtBsdf &bsdf, float z) {
  const uint lobe = ptBsdfSampledLobe(bsdf, z);
  if (lobe == 3u) return bsdf.clearcoat.y;
  if (lobe == 0u) return 1.0f;
  return bsdf.alpha;
}

// Samples a direction: a visible GGX normal's reflection or a cosine-weighted one, one lobe
// chosen by xi.z (ptBsdfSampledLobe). Its value and pdf come from ptBsdfEvaluate, which accounts
// for every lobe. A refraction beyond the critical angle returns zero.
inline float3 ptBsdfSampleDirection(thread const PtBsdf &bsdf, float3 xi) {
  const uint lobe = ptBsdfSampledLobe(bsdf, xi.z);
  float3 direction = cosineSampleHemisphere(bsdf.normal, float2(xi.x, xi.y));
  if (lobe != 0u) {
    // One visible-normal sample for every microfacet lobe, in that lobe's frame.
    float3 frameNormal = bsdf.normal, frameTangent = bsdf.tangent, frameBitangent = bsdf.bitangent;
    float frameAlpha = bsdf.alpha;
    if (lobe == 3u) {
      frameNormal = bsdf.clearcoatNormal;
      orthonormalBasis(frameNormal, frameTangent, frameBitangent);
      frameAlpha = bsdf.clearcoat.y;
    }
    const float3 halfLocal = ptSampleVisibleNormal(
        float3(dot(bsdf.view, frameTangent), dot(bsdf.view, frameBitangent), dot(bsdf.view, frameNormal)), frameAlpha,
        float2(xi.x, xi.y));
    const float3 micro = frameTangent * halfLocal.x + frameBitangent * halfLocal.y + frameNormal * halfLocal.z;
    direction = reflect(-bsdf.view, micro);
    if (lobe == 2u) direction = ptTransmittedDirection(bsdf, micro);
  }
  return direction;
}
