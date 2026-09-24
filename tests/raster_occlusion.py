#!/usr/bin/env python3
"""The rasteriser's traced occlusion modes, read through the occlusion debug view (--debug 5).

Renders the built-in cube on its ground plane with each --ao mode and checks the occlusion
term itself, so the result does not depend on lighting:
- 0 (material map): the untextured cube and ground are wholly open.
- 1 (traced contact): the ground away from the cube is open.
- 2 (traced sky visibility): the ground darkens towards the cube's base, strictly more at the
  base than beside it and beside it than far out; the cube's top sees the whole sky; the
  ground plane never occludes, so a vertical face, half of whose hemisphere looks at the
  ground, stays open.
A device without ray queries skips.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness

# Regions of the default view at 960x540: x0, y0, x1, y1.
REGIONS = {
    "contact": (380, 455, 470, 480),   # ground at the cube's front base
    "beside": (200, 400, 300, 470),    # ground a little to its left
    "far": (750, 300, 900, 400),       # ground well away
    "top": (400, 145, 560, 175),       # the cube's top face
    "face": (340, 220, 450, 380),      # its left vertical face
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = {}
    for mode in (0, 1, 2):
        output = args.output_dir / f"occlusion-{mode}.pfm"
        command = [str(args.exe), "--renderer", "raster", "--ao", str(mode), "--debug", "5", "--size", "960x540",
                   "--frame", "64", "--no-ui", "--no-vsync", "--pfm", str(output)]
        try:
            log = capture_harness.run(command, output, args.output_dir)
        except capture_harness.CaptureError as error:
            if "require Vulkan ray queries" in str(error):
                print("SKIP: traced occlusion needs ray queries on this device")
                return capture_harness.SKIP
            raise
        image = compare_images.read_pfm(str(output))
        values[mode] = {name: float(image[y0:y1, x0:x1, :].mean()) for name, (x0, y0, x1, y1) in REGIONS.items()}
        print(f"--ao {mode}: " + ", ".join(f"{name} {v:.3f}" for name, v in values[mode].items()), flush=True)

    checks = [
        ("the material map leaves everything open", all(v > 0.98 for v in values[0].values())),
        ("contact occlusion leaves the far ground open", values[1]["far"] > 0.98),
        ("sky visibility darkens the cube's base", values[2]["contact"] < 0.85),
        ("sky visibility grows away from the cube",
         values[2]["contact"] < values[2]["beside"] - 0.05 and values[2]["beside"] < values[2]["far"] - 0.03),
        ("sky visibility keeps the far ground mostly open", values[2]["far"] > 0.9),
        ("the cube's top sees the whole sky", values[2]["top"] > 0.98),
        ("the ground does not occlude a vertical face", values[2]["face"] > 0.95),
    ]
    failures = 0
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        failures += not passed
    if failures:
        capture_harness.preserve_failure([args.output_dir / f"occlusion-{m}.pfm" for m in (0, 1, 2)], "occlusion")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
