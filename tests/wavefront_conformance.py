#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness


def guide_paths(output):
    return output.with_suffix(".albedo.pfm"), output.with_suffix(".normal.pfm")


def check_guides(baseline, candidate, label):
    """First-hit guides must survive an execution switch: the first sample after a
    restart replaces the sums for hit and miss pixels alike. Both executions trace the
    same first hits, so only float ordering may differ."""
    for reference, result, name in zip(guide_paths(baseline), guide_paths(candidate), ("albedo", "normal")):
        a = compare_images.read_pfm(str(reference))
        b = compare_images.read_pfm(str(result))
        worst = float(abs(a - b).max())
        if worst > 1e-4:
            capture_harness.preserve_failure([baseline, candidate], f"{label}-{name}")
            raise RuntimeError(f"{label} {name} guide differs from the megakernel by {worst}")
        print(f"{label} {name} guide max difference {worst:.3g}")


def capture(exe, scene, output, execution, width="binary", temporal=False,
            renderer="gpu-bvh", builder="cpu", transition=False, fusion=None):
    initial_execution = "megakernel" if transition else execution
    command = [str(exe), str(scene), "--renderer", renderer,
               "--path-execution", initial_execution, "--spp", "16",
               "--spf", "4", "--bounces", "3", "--seed", "31", "--pfm", str(output),
               "--no-ui", "--no-vsync"]
    if transition:
        command += ["--switch-path-execution-frame", "1"]
    if renderer == "gpu-bvh":
        command += ["--bvh-builder", builder, "--bvh-width", width]
    if temporal:
        command.append("--path-temporal")
    if fusion:
        command += ["--wavefront-fusion", fusion]
    albedo, normal = guide_paths(output)
    command += ["--albedo-pfm", str(albedo), "--normal-pfm", str(normal)]
    capture_harness.run(command, output, scene.parents[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = args.output_dir / ("wavefront-rt-megakernel.pfm" if args.hardware else "wavefront-megakernel.pfm")
    renderer = "gpu" if args.hardware else "gpu-bvh"
    capture(args.exe, args.scene, baseline, "megakernel", renderer=renderer)
    if args.hardware:
        candidate = args.output_dir / "wavefront-rt.pfm"
        capture(args.exe, args.scene, candidate, "wavefront", renderer="gpu", temporal=True)
        result = compare_images.compare(compare_images.read_pfm(str(baseline)),
                                        compare_images.read_pfm(str(candidate)), 8, 1e-3)
        if (result["aggregate"] > 0.001 or result["energy_error"] > 0.001 or
                result["percentile99"] > 0.01 or result["worst"] > 0.03):
            capture_harness.preserve_failure([baseline, candidate], "hardware")
            raise RuntimeError("hardware wavefront exceeds fixed agreement gates")
        check_guides(baseline, candidate, "hardware")
        print("PASS: hardware-ray-query wavefront agrees with the megakernel")
        return
    # Automatic fuses traversal into the shade stage; the split stages must agree the same way,
    # guides included.
    for name, width, temporal, fusion in (("binary", "binary", False, None), ("bvh4", "bvh4", False, None),
                                          ("bvh8", "bvh8", False, None), ("temporal", "binary", True, None),
                                          ("split-binary", "binary", False, "off"), ("split-bvh8", "bvh8", False, "off"),
                                          ("split-temporal", "binary", True, "off")):
        candidate = args.output_dir / f"wavefront-{name}.pfm"
        capture(args.exe, args.scene, candidate, "wavefront", width, temporal,
                transition=name == "binary", fusion=fusion)
        result = compare_images.compare(compare_images.read_pfm(str(baseline)),
                                        compare_images.read_pfm(str(candidate)), 8, 1e-3)
        print(name, json.dumps({k: v for k, v in result.items() if k != "block_relative"}, default=str))
        check_guides(baseline, candidate, name)
        if (result["aggregate"] > 0.001 or result["energy_error"] > 0.001 or
                result["percentile99"] > 0.01 or result["worst"] > 0.03):
            capture_harness.preserve_failure([baseline, candidate], name)
            raise RuntimeError(f"{name} exceeds fixed wavefront agreement gates")
    gpu_builder = args.output_dir / "wavefront-gpu-builder.pfm"
    capture(args.exe, args.scene, gpu_builder, "wavefront", builder="gpu")
    result = compare_images.compare(compare_images.read_pfm(str(baseline)),
                                    compare_images.read_pfm(str(gpu_builder)), 8, 1e-3)
    if result["aggregate"] > 0.001 or result["worst"] > 0.03:
        capture_harness.preserve_failure([baseline, gpu_builder], "gpu-builder")
        raise RuntimeError("GPU-built wavefront exceeds fixed agreement gates")
    print("PASS: production wavefront backends agree with the megakernel")


if __name__ == "__main__":
    main()
