#!/usr/bin/env python3
"""The parallel GPU LBVH builds the serial builder's tree, so captures traced through either
must be byte-identical: same topology, bounds and leaf order give the same traversal and the
same hits. Megakernel and wavefront, on the fixtures in tests/data, with validation on.

Animated (--animate both), the per-frame updates (a refit, a TLAS rebuild, the wide re-emit or
collapse) must trace like a full build every frame: the frames' closest hits are the same, so
the captures are byte-identical too, binary and BVH8. The deep-stack wide kernels trace like the
default ones."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import capture_harness


def capture(exe, scene, environment, output, builder, execution, extra=None, width="binary", spp=("8", "4")):
    command = [str(exe), str(scene)] + ([str(environment)] if environment else []) + [
        "--renderer", "gpu-bvh", "--bvh-builder", builder, "--bvh-width", width,
        "--path-execution", execution, "--size", "320x180", "--spp", spp[0], "--spf", spp[1],
        "--bounces", "4", "--seed", "7", "--no-ui", "--no-vsync", "--pfm", str(output)] + (extra or [])
    capture_harness.run(command, output, scene.parents[2])


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
        for execution in ("megakernel", "wavefront"):
            serial = args.output_dir / f"{name}-{execution}-serial.pfm"
            parallel = args.output_dir / f"{name}-{execution}-lbvh.pfm"
            capture(args.exe, scene, environment, serial, "gpu-serial", execution)
            capture(args.exe, scene, environment, parallel, "gpu-lbvh", execution)
            if serial.read_bytes() != parallel.read_bytes():
                capture_harness.preserve_failure([serial, parallel], f"lbvh-{name}-{execution}")
                raise RuntimeError(f"{name} {execution}: the parallel LBVH capture differs from the serial builder's")
            print(f"PASS: {name} {execution} parallel LBVH capture is byte-identical to the serial builder's")
    for name, scene, environment in scenes:
        for width in ("binary", "bvh8"):
            updated = args.output_dir / f"{name}-{width}-animated-updated.pfm"
            rebuilt = args.output_dir / f"{name}-{width}-animated-rebuilt.pfm"
            animated = ["--animate", "both"]
            capture(args.exe, scene, environment, updated, "gpu-lbvh", "megakernel", animated, width, ("6", "1"))
            capture(args.exe, scene, environment, rebuilt, "gpu-lbvh", "megakernel",
                    animated + ["--bvh-update", "rebuild"], width, ("6", "1"))
            if updated.read_bytes() != rebuilt.read_bytes():
                capture_harness.preserve_failure([updated, rebuilt], f"animated-{name}-{width}")
                raise RuntimeError(f"{name} {width}: animated frames from updates differ from full rebuilds")
            print(f"PASS: {name} {width} animated frame 6 from refits and TLAS rebuilds is byte-identical to full builds")
            refitted = args.output_dir / f"{name}-{width}-animated-refitted.pfm"
            capture(args.exe, scene, environment, refitted, "gpu-lbvh", "megakernel",
                    animated + ["--bvh-update", "refit"], width, ("6", "1"))
            if refitted.read_bytes() != rebuilt.read_bytes():
                capture_harness.preserve_failure([refitted, rebuilt], f"animated-refit-{name}-{width}")
                raise RuntimeError(f"{name} {width}: animated frames from refits alone differ from full rebuilds")
            print(f"PASS: {name} {width} animated frame 6 from refits alone is byte-identical to full builds")
    # The deep-stack wide kernels (--wide-stack deep, for trees whose bound exceeds 64) trace
    # exactly as the default ones.
    for name, scene, environment in scenes:
        for execution in ("megakernel", "wavefront"):
            default = args.output_dir / f"{name}-{execution}-bvh8-stack-auto.pfm"
            deep = args.output_dir / f"{name}-{execution}-bvh8-stack-deep.pfm"
            capture(args.exe, scene, environment, default, "gpu-lbvh", execution, None, "bvh8")
            capture(args.exe, scene, environment, deep, "gpu-lbvh", execution, ["--wide-stack", "deep"], "bvh8")
            if default.read_bytes() != deep.read_bytes():
                capture_harness.preserve_failure([default, deep], f"wide-stack-{name}-{execution}")
                raise RuntimeError(f"{name} {execution}: the deep-stack wide kernels differ from the default ones")
            print(f"PASS: {name} {execution} BVH8 deep-stack kernels are byte-identical to the default ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
