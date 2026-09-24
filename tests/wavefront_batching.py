#!/usr/bin/env python3
"""V6.1 wavefront edge cases against the megakernel of the same backend.

Odd extents, 1/4/16 samples per frame, a non-divisible 17-SPP target, whole-pixel batches
down to one pixel, partial final batches, both queue allocation variants, fused and split
ray-query stages and every intersector. Each wavefront capture must meet the fixed same-estimator gates against its
megakernel (aggregate and energy <= 0.001, p99 <= 0.01, worst block <= 0.03) and match its
first-hit guides; queue overflow is read back every frame and fails a capture. A deliberately
undersized queue must fail with the reported overflow rather than lose paths silently.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import capture_harness
import compare_images

SKIP = capture_harness.SKIP


def gates(result):
    return (result["aggregate"] <= 0.001 and result["energy_error"] <= 0.001 and
            result["percentile99"] <= 0.01 and result["worst"] <= 0.03)


def capture(exe, scene, output, renderer, execution, size, spp, spf, extra=()):
    command = [exe, scene, "--renderer", renderer, "--path-execution", execution, "--size", size,
               "--spp", str(spp), "--spf", str(spf), "--bounces", "4", "--seed", "23", "--no-ui",
               "--no-vsync", "--pfm", output, "--albedo-pfm", output.with_suffix(".albedo.pfm"),
               "--normal-pfm", output.with_suffix(".normal.pfm"), *extra]
    if renderer == "gpu-bvh" and "--bvh-builder" not in extra:
        command += ["--bvh-builder", "cpu", "--bvh-width", "binary"]
    capture_harness.run(command, output, scene.parents[2])


def compare(reference, candidate, label):
    result = compare_images.compare(compare_images.read_pfm(str(reference)),
                                    compare_images.read_pfm(str(candidate)), 8, 1e-3)
    guide = max(float(abs(compare_images.read_pfm(str(reference.with_suffix(f".{g}.pfm"))) -
                          compare_images.read_pfm(str(candidate.with_suffix(f".{g}.pfm")))).max())
                for g in ("albedo", "normal"))
    ok = gates(result) and guide <= 1e-4
    print(f"{'PASS' if ok else 'FAIL'} {label}: aggregate {result['aggregate']:.3g} energy "
          f"{result['energy_error']:.3g} p99 {result['percentile99']:.3g} worst {result['worst']:.3g} "
          f"guide {guide:.3g}", flush=True)
    if not ok:
        capture_harness.preserve_failure([reference, candidate], label.replace(" ", "-"))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    backends = [("gpu-bvh", ()), ("gpu-bvh", ("--bvh-builder", "cpu", "--bvh-width", "bvh8")),
                ("gpu-bvh", ("--bvh-builder", "gpu", "--bvh-width", "binary")), ("gpu", ()),
                ("gpu-pipeline", ())]
    # (label, size, spp, spf, wavefront-only options)
    cases = [("odd-spf1", "641x359", 3, 1, ()),
             ("odd-spf4-target17", "641x359", 17, 4, ()),
             ("spf16", "320x181", 32, 16, ()),
             ("partial-batches", "321x179", 17, 4, ("--wavefront-capacity", "5000")),
             ("one-pixel-batches", "37x23", 6, 3, ("--wavefront-capacity", "3")),
             ("atomic-allocation", "641x359", 8, 4, ("--wavefront-allocation", "atomic")),
             ("subgroup-allocation", "641x359", 8, 4, ("--wavefront-allocation", "subgroup")),
             # Ray queries fuse intersect into shade by default; keep the split stages covered.
             ("split-stages", "641x359", 8, 4, ("--wavefront-fusion", "off"))]
    failures, ran, skipped = 0, 0, 0
    for renderer, backend_options in backends:
        name = renderer + "".join("-" + o for o in backend_options if not o.startswith("--"))
        for label, size, spp, spf, options in cases:
            reference = out / f"{name}-{label}-megakernel.pfm"
            candidate = out / f"{name}-{label}-wavefront.pfm"
            try:
                capture(args.exe, args.scene, reference, renderer, "megakernel", size, spp, spf, backend_options)
                capture(args.exe, args.scene, candidate, renderer, "wavefront", size, spp, spf,
                        (*backend_options, *options))
            except SystemExit as stop:
                if stop.code != SKIP:
                    raise
                skipped += 1
                print(f"SKIP {name} {label}: capability unavailable")
                continue
            ran += 1
            failures += 0 if compare(reference, candidate, f"{name} {label}") else 1
    # A queue deliberately smaller than one bounce's survivors must be reported.
    overflow = out / "deliberate-overflow.pfm"
    try:
        capture(args.exe, args.scene, overflow, "gpu-bvh", "wavefront", "64x64", 2, 2,
                ("--wavefront-queue-limit", "64"))
        print("FAIL deliberate overflow: the capture succeeded")
        failures += 1
    except capture_harness.CaptureError as error:
        if "refused" in str(error) and "reservations" in str(error):
            print("PASS deliberate overflow was reported and failed the capture")
        else:
            print(f"FAIL deliberate overflow failed for another reason:\n{error}")
            failures += 1
    print(f"{ran} comparisons, {skipped} skipped by capability, {failures} failed")
    if failures:
        raise SystemExit(1)
    if ran == 0:
        raise SystemExit(SKIP)


if __name__ == "__main__":
    main()
