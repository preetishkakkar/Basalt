# Frozen V7 motion corpus

Scene/camera generator: `tests/temporal_motion_v7_fixture.h`, frozen before these references
were generated. It is the V3.1 corpus (`../temporal/README.md`: scene, trajectories, size,
seeds) with V7's features:

- `glass-translation`: the brushed-metal sphere is smooth closed glass (transmission 1, IOR
  1.5, roughness 0.05) and the orange sphere has a mirror-like clearcoat (factor 1, roughness
  0.05); V3.1 translation trajectory;
- `dof-translation`: V3.1 materials, thin lens with aperture radius 0.04 focused 4.3 units
  ahead (the look target); translation trajectory;
- `dof-stop-start`: the same lens, stop-start trajectory.

References: little-endian contiguous RGB float32, top-down rows, twelve frames per file,
seed 1701, 4096 real paths per pixel, four bounces, MIS, no clamp or reconstruction, CPU
own BVH, double-precision accumulation. `references.json` records the generator revision and
SHA-256 of each file. Generate explicitly with
`build/temporal-motion-tests.exe --v7 --generate-references`; ordinary CTest only reads them.
Do not regenerate as a way to pass a failed filter gate.

Gates (the V3.1 gates): finite output; mean spatial MAE and reference-subtracted
temporal flicker each at most 0.90 of raw, averaged across the four seeds, per sequence
(`temporal-v7-motion-tests`, own BVH and Embree). Reference convergence, fixed before its first
run: an independent 8192-SPP render within 1% normalized MAE per sequence
(`build/temporal-motion-tests.exe --v7 --check-reference-convergence`; 0.0065-0.0067 when
generated).
