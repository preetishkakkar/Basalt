#!/usr/bin/env python3
"""glTF 2.0 sidedness of mirrored one-sided surfaces on every renderer.

Renders tests/data/mirrored.gltf (see make_mirrored_gltf.py) with the rasteriser, the hybrid
renderer, the CPU tracer and every GPU path tracer the device offers, and checks each third
of the image: the mirrored quad whose object-space front faces the camera (left) and the
plain control (right) must be visible, the mirrored quad whose object-space front faces away
(middle) must be culled. A renderer the device cannot run is a capability skip.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness

RENDERERS = {
    "raster": ["--renderer", "raster"],
    "hybrid": ["--renderer", "hybrid"],
    "cpu": ["--renderer", "cpu"],
    "query-megakernel": ["--renderer", "gpu"],
    "query-wavefront": ["--renderer", "gpu", "--path-execution", "wavefront"],
    "bvh-megakernel": ["--renderer", "gpu-bvh"],
    "bvh8-wavefront": ["--renderer", "gpu-bvh", "--bvh-width", "bvh8", "--path-execution", "wavefront"],
    "pipeline-iterative": ["--renderer", "gpu-pipeline"],
    "pipeline-wavefront": ["--renderer", "gpu-pipeline", "--path-execution", "wavefront"],
}


def red_fraction(image, x0, x1):
    region = image[:, x0:x1, :]
    red = (region[:, :, 0] > 0.5) & (region[:, :, 0] > 4.0 * (region[:, :, 1] + region[:, :, 2]) + 0.05)
    return float(red.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures, ran = 0, []
    for name, arguments in RENDERERS.items():
        output = args.output_dir / f"sidedness-{name}.pfm"
        command = [str(args.exe), str(args.scene), *arguments, "--size", "240x80", "--spp", "4",
                   "--bounces", "2", "--seed", "3", "--view", "0", "0", "0.35",  # one quad per third
                   "--pfm", str(output), "--no-ui", "--no-vsync"]
        try:
            capture_harness.run(command, output, args.scene.parents[2])
        except SystemExit as skip:
            if skip.code != capture_harness.SKIP:
                raise
            print(f"SKIP: {name} (capability)")
            continue
        ran.append(name)
        image = compare_images.read_pfm(str(output))
        width = image.shape[1]
        left, middle, right = (red_fraction(image, 0, width // 3), red_fraction(image, width // 3, 2 * width // 3),
                               red_fraction(image, 2 * width // 3, width))
        # A visible quad covers about 0.22 of its third; a culled one leaves only the emitters'
        # glow on the ground (about 0.002).
        passed = left > 0.1 and right > 0.1 and middle < 0.02
        print(f"{'PASS' if passed else 'FAIL'}: {name}: red coverage left {left:.3f} (mirrored front, visible), "
              f"middle {middle:.4f} (mirrored back, culled), right {right:.3f} (plain, visible)", flush=True)
        if not passed:
            failures += 1
            capture_harness.preserve_failure([output], f"sidedness-{name}")
    if not ran:
        return capture_harness.SKIP
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
