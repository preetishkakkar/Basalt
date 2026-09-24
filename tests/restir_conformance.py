#!/usr/bin/env python3
"""ReSTIR DI gates: ReSTIR against baseline next-event
estimation at convergence, on the materials fixture and the many-light fixture (64 punctual
lights, emissive panels, a dim environment). The reference is the CPU tracer's NEE at 4096
SPP on its own seed; every backend renders ReSTIR at 4096 SPP (one path per pixel per frame)
with four seeds, RIS only (unbiased) and with temporal + spatial reuse (unbiased up to
visibility), against the frozen gates over 8-pixel blocks:

    RIS only             aggregate 2%, energy 1%, p99 5%
    temporal + spatial   aggregate 4%, energy 3%, p99 10%

A backend the device cannot run is reported as a capability skip; the test skips only if no
GPU backend runs at all.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness

SEEDS = (3, 17, 31, 97)
REFERENCE_SEED = 1000
BACKENDS = {
    "cpu": ["--renderer", "cpu"],
    "query-megakernel": ["--renderer", "gpu", "--path-execution", "megakernel"],
    "query-wavefront": ["--renderer", "gpu", "--path-execution", "wavefront"],
    "bvh-megakernel": ["--renderer", "gpu-bvh", "--path-execution", "megakernel"],
    "bvh8-wavefront": ["--renderer", "gpu-bvh", "--bvh-width", "bvh8", "--path-execution", "wavefront"],
    "pipeline-iterative": ["--renderer", "gpu-pipeline", "--path-execution", "megakernel"],
    "pipeline-wavefront": ["--renderer", "gpu-pipeline", "--path-execution", "wavefront"],
}
MODES = {
    "ris": (["--di-estimator", "restir", "--restir-reuse", "none"], (0.02, 0.01, 0.05), "unbiased"),
    "reuse": (["--di-estimator", "restir", "--restir-reuse", "both"], (0.04, 0.03, 0.10), "visibility-only"),
}


def capture(exe, fixture, output, arguments, seed, spp, size):
    command = [str(exe), *fixture["files"], *arguments, "--size", size, "--spp", str(spp), "--spf", "1",
               "--bounces", "4", "--seed", str(seed), "--view", *fixture["view"], "--pfm", str(output),
               "--no-ui", "--no-vsync"]
    capture_harness.run(command, output, Path(fixture["files"][0]).parents[2])


def gate(reference, candidate, limits, label):
    result = compare_images.compare(compare_images.read_pfm(str(reference)),
                                    compare_images.read_pfm(str(candidate)), 8, 1e-3)
    passed = (result["aggregate"] <= limits[0] and result["energy_error"] <= limits[1] and
              result["percentile99"] <= limits[2])
    print(f"{'PASS' if passed else 'FAIL'}: {label}: aggregate {result['aggregate']:.4f}, "
          f"energy {result['energy_error']:.4f}, p99 {result['percentile99']:.4f}", flush=True)
    if not passed:
        capture_harness.preserve_failure([reference, candidate], label.replace(" ", "-"))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="tests/data")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spp", type=int, default=4096)
    parser.add_argument("--size", default="160x90")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "materials": {"files": [str(args.data / "materials.gltf")], "view": ["0", "12", "0.6"]},
        "manylights": {"files": [str(args.data / "manylights.gltf"), str(args.data / "manylights_dim.hdr")],
                       "view": ["20", "25", "0.35"]},
    }
    failures, ran = 0, set()
    for name, fixture in fixtures.items():
        reference = args.output_dir / f"{name}-nee-{REFERENCE_SEED}.pfm"
        capture(args.exe, fixture, reference, BACKENDS["cpu"], REFERENCE_SEED, args.spp, args.size)
        for backend in args.backends.split(","):
            for mode, (arguments, limits, bias) in MODES.items():
                for seed in SEEDS:
                    candidate = args.output_dir / f"{name}-{backend}-{mode}-{seed}.pfm"
                    try:
                        capture(args.exe, fixture, candidate, [*BACKENDS[backend], *arguments], seed, args.spp,
                                args.size)
                    except SystemExit as skip:
                        if skip.code != capture_harness.SKIP:
                            raise
                        print(f"SKIP: {backend} (capability)")
                        break
                    ran.add(backend)
                    metadata = json.loads(Path(str(candidate) + ".json").read_text(encoding="utf-8"))
                    integrator = metadata["integrator"]
                    if integrator.get("di_estimator") != "restir" or integrator.get("restir_bias") != bias:
                        print(f"FAIL: {name} {backend} {mode} seed {seed}: metadata records "
                              f"{integrator.get('di_estimator')} / {integrator.get('restir_bias')}")
                        failures += 1
                    failures += not gate(reference, candidate, limits, f"{name} {backend} {mode} seed {seed}")
    if not ran - {"cpu"}:
        print("SKIP: no GPU backend available")
        return capture_harness.SKIP
    if failures:
        print(f"FAIL: {failures} comparisons exceed the ReSTIR gates")
        return 1
    print(f"PASS: ReSTIR DI gates for {', '.join(sorted(ran))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
