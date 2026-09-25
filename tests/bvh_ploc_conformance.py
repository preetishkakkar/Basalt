#!/usr/bin/env python3
"""The parallel GPU PLOC builder builds a different tree than the serial LBVH, so its captures
are held to the conformance gates rather than byte identity: a different tree finds the same
closest hits except at exact ties in distance, where traversal order picks the triangle. On the
fixtures in tests/data, with validation on: binary through the megakernel, BVH8 through the
wavefront, and animated (--animate both, per-frame refits and LBVH TLAS rebuilds of the PLOC
tree) against a PLOC build every frame. Each comparison also reports whether it was
byte-identical."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness


def capture(exe, scene, environment, output, builder, execution, width, extra=None, spp=("8", "4")):
    command = [str(exe), str(scene)] + ([str(environment)] if environment else []) + [
        "--renderer", "gpu-bvh", "--bvh-builder", builder, "--bvh-width", width,
        "--path-execution", execution, "--size", "320x180", "--spp", spp[0], "--spf", spp[1],
        "--bounces", "4", "--seed", "7", "--no-ui", "--no-vsync", "--pfm", str(output)] + (extra or [])
    capture_harness.run(command, output, scene.parents[2])


def check(reference, candidate, label):
    result = compare_images.compare(compare_images.read_pfm(str(reference)),
                                    compare_images.read_pfm(str(candidate)), 8, 1e-3)
    identical = reference.read_bytes() == candidate.read_bytes()
    if (result["aggregate"] > 0.001 or result["energy_error"] > 0.001 or
            result["percentile99"] > 0.01 or result["worst"] > 0.03):
        capture_harness.preserve_failure([reference, candidate], f"ploc-{label}")
        raise RuntimeError(f"{label}: the PLOC capture exceeds the conformance gates "
                           f"(aggregate {result['aggregate']:.3g}, worst block {result['worst']:.3g})")
    print(f"PASS: {label} PLOC capture within the conformance gates "
          f"({'byte-identical' if identical else 'aggregate %.3g, worst block %.3g' % (result['aggregate'], result['worst'])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [("materials", args.data / "materials.gltf", None),
              ("raycone", args.data / "raycone.gltf", None),
              ("manylights", args.data / "manylights.gltf", args.data / "manylights_dim.hdr")]
    for name, scene, environment in scenes:
        for execution, width in (("megakernel", "binary"), ("wavefront", "bvh8")):
            serial = args.output_dir / f"{name}-{execution}-{width}-serial.pfm"
            ploc = args.output_dir / f"{name}-{execution}-{width}-ploc.pfm"
            capture(args.exe, scene, environment, serial, "gpu-serial", execution, width)
            capture(args.exe, scene, environment, ploc, "gpu-ploc", execution, width)
            check(serial, ploc, f"{name} {execution} {width}")
        updated = args.output_dir / f"{name}-bvh8-animated-updated.pfm"
        rebuilt = args.output_dir / f"{name}-bvh8-animated-rebuilt.pfm"
        animated = ["--animate", "both"]
        capture(args.exe, scene, environment, updated, "gpu-ploc", "megakernel", "bvh8", animated, ("6", "1"))
        capture(args.exe, scene, environment, rebuilt, "gpu-ploc", "megakernel", "bvh8",
                animated + ["--bvh-update", "rebuild"], ("6", "1"))
        check(rebuilt, updated, f"{name} bvh8 animated (updates against rebuilds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
