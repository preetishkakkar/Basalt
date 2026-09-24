#!/usr/bin/env python3
"""V7 image gates: the transmission/clearcoat
fixture, the same with depth of field, and the ray-cone fixture (minified textures and masked
foliage) with ray-cone filtering, rendered by the CPU tracer and by every GPU backend the
device offers, four seeds each, against the frozen gates (aggregate 2%, energy 1%, p99 5% over
8-pixel blocks), plus a doubled-SPP case, and raw invariance:
with temporal reconstruction on, every backend's raw capture is byte-identical to the one
without it.

A backend the device cannot run is reported as a capability skip; the test skips only if no
GPU backend runs at all.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness

SEEDS = (3, 17, 31, 97)
BACKENDS = {
    "query-megakernel": ["--renderer", "gpu", "--path-execution", "megakernel"],
    "query-wavefront": ["--renderer", "gpu", "--path-execution", "wavefront"],
    "bvh-megakernel": ["--renderer", "gpu-bvh", "--path-execution", "megakernel"],
    "bvh8-wavefront": ["--renderer", "gpu-bvh", "--bvh-width", "bvh8", "--path-execution", "wavefront"],
    "bvh-split-wavefront": ["--renderer", "gpu-bvh", "--path-execution", "wavefront", "--wavefront-fusion", "off"],
    "pipeline-iterative": ["--renderer", "gpu-pipeline", "--path-execution", "megakernel"],
    "pipeline-wavefront": ["--renderer", "gpu-pipeline", "--path-execution", "wavefront"],
}


def capture(exe, scene, output, arguments, seed, spp, case):
    command = [str(exe), str(scene), *arguments, "--size", "320x180", "--spp", str(spp), "--spf", "8",
               "--bounces", "8", "--seed", str(seed), "--view", *case["view"], "--pfm", str(output),
               "--no-ui", "--no-vsync", *case["options"]]
    capture_harness.run(command, output, scene.parents[2])


def gate(reference, candidate, label):
    result = compare_images.compare(compare_images.read_pfm(str(reference)),
                                    compare_images.read_pfm(str(candidate)), 8, 1e-3)
    passed = result["aggregate"] <= 0.02 and result["energy_error"] <= 0.01 and result["percentile99"] <= 0.05
    print(f"{'PASS' if passed else 'FAIL'}: {label}: aggregate {result['aggregate']:.4f}, "
          f"energy {result['energy_error']:.4f}, p99 {result['percentile99']:.4f}", flush=True)
    if not passed:
        capture_harness.preserve_failure([reference, candidate], label.replace(" ", "-"))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--raycone-scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spp", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures, ran = 0, set()
    # The V7 fixture framed on the spheres and the sheet (yaw 0, pitch 12, 0.35 of the fitted
    # distance); depth of field focuses on the orbit target (the fixture's centre): sharp spheres,
    # blurred wall. The ray-cone fixture is seen from behind two foliage cards, over the ground
    # to the horizon.
    cases = {
        "materials": {"scene": args.scene, "view": ["0", "12", "0.35"], "options": []},
        "depth-of-field": {"scene": args.scene, "view": ["0", "12", "0.35"], "options": ["--aperture", "0.05"]},
        "ray-cones": {"scene": args.raycone_scene, "view": ["180", "6", "0.1"],
                      "options": ["--texture-filter", "raycone"]},
    }
    for case, setup in cases.items():
        runs = [(seed, args.spp) for seed in SEEDS] + [(SEEDS[0], 2 * args.spp)]
        for seed, spp in runs:
            reference = args.output_dir / f"{case}-cpu-{seed}-{spp}.pfm"
            capture(args.exe, setup["scene"], reference, ["--renderer", "cpu"], seed, spp, setup)
            for name, arguments in BACKENDS.items():
                candidate = args.output_dir / f"{case}-{name}-{seed}-{spp}.pfm"
                try:
                    capture(args.exe, setup["scene"], candidate, arguments, seed, spp, setup)
                except SystemExit as skip:
                    if skip.code != capture_harness.SKIP:
                        raise
                    print(f"SKIP: {name} (capability)")
                    continue
                ran.add(name)
                failures += not gate(reference, candidate, f"{case} {name} seed {seed} {spp} spp")
                if case == "ray-cones" and seed == SEEDS[0] and spp == args.spp:
                    # Filtering must be active: the capture differs from the same backend at
                    # level zero (a lost cone or a one-level texture view would pass the gate).
                    level0 = args.output_dir / f"{case}-{name}-{seed}-{spp}-level0.pfm"
                    capture(args.exe, setup["scene"], level0, arguments, seed, spp,
                            {**setup, "options": ["--texture-filter", "level0"]})
                    result = compare_images.compare(compare_images.read_pfm(str(level0)),
                                                    compare_images.read_pfm(str(candidate)), 8, 1e-3)
                    active = result["aggregate"] >= 0.01
                    print(f"{'PASS' if active else 'FAIL'}: {case} {name} filtering active: "
                          f"{result['aggregate']:.4f} aggregate against level zero (at least 0.01)", flush=True)
                    if not active:
                        failures += 1
                        capture_harness.preserve_failure([level0, candidate], f"{case}-{name}-filtering-active")
                if seed == SEEDS[0] and spp == args.spp:
                    temporal = args.output_dir / f"{case}-{name}-{seed}-{spp}-temporal.pfm"
                    capture(args.exe, setup["scene"], temporal, [*arguments, "--path-temporal"], seed, spp, setup)
                    same = temporal.read_bytes() == candidate.read_bytes()
                    print(f"{'PASS' if same else 'FAIL'}: {case} {name} raw invariance with temporal reconstruction",
                          flush=True)
                    if not same:
                        failures += 1
                        capture_harness.preserve_failure([candidate, temporal], f"{case}-{name}-raw-invariance")
    if not ran:
        print("SKIP: no GPU backend available")
        return capture_harness.SKIP
    if failures:
        print(f"FAIL: {failures} comparisons exceed the V7 gates")
        return 1
    print(f"PASS: V7 image gates for {', '.join(sorted(ran))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
