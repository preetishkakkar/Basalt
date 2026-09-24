# Frozen V3.1 motion corpus

Scene/camera generator: `tests/temporal_motion_fixture.h`, frozen before filter tuning.
Three 12-frame sequences: 0.6-unit translation, 0.7-unit look-target rotation,
translation with four stationary frames then restart. Each frame is 32x24 linear RGB.
Includes diffuse and glossy spheres, a mirror sphere, thin strips, masked and stochastic
blended geometry, ground and a bright environment patch. Seeds: 11, 22, 33, 44.

References: little-endian contiguous RGB float32, top-down rows, twelve frames per file.
Reference seed 1701, 4096 real paths/pixel, four bounces, MIS, no clamp or reconstruction,
CPU own-BVH, double precision accumulation. No filter output enters a reference.
Generate explicitly with `build/temporal-motion-tests.exe --generate-references`;
ordinary CTest only reads these files. Reference source revision and SHA256 are recorded
beside them after generation. Do not regenerate as a way to pass a failed filter gate.

Gate: mean spatial MAE and reference-subtracted temporal flicker each <=0.90 of raw,
averaged across all four seeds, separately for each sequence. Full-frame metric, no
edge/mirror/alpha exclusion. The lifecycle and ghosting gates are separate tests.

Reference convergence check (fixed before its first run): independently render 8192 SPP
and require RGB MAE divided by mean RGB energy <=0.01, separately for each sequence.
Run `build/temporal-motion-tests.exe --check-reference-convergence`; never overwrite
the checked-in references during validation. Stationary frames continue actual sample
indices and raw accumulation, matching the interactive renderer's stop/start behavior.
