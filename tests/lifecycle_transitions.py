#!/usr/bin/env python3
"""Scripted lifecycle transitions with frames in flight, validation-scanned.

Each scenario starts one basalt.exe capture and applies --at-frame transitions: renderer
and execution changes, builder/layout changes, temporal display, resize and scene
replacement. A scenario passes when the process exits 0, writes its capture and logs no
Vulkan validation warning or error. A scenario whose renderer the device lacks is a
SKIP; the script exits 77 only if every scenario skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import capture_harness


def scenarios(scene):
    # (name, initial arguments, [(frame, "key=value"), ...])
    return [
        ("own-bvh-to-raster-and-back", ["--renderer", "gpu-bvh"],
         [(2, "renderer=raster"), (4, "renderer=gpu-bvh"), (6, "renderer=raster")]),
        ("renderer-cycle", ["--renderer", "gpu"],
         [(1, "renderer=gpu-bvh"), (2, "renderer=gpu-pipeline"), (3, "renderer=cpu"),
          (4, "renderer=hybrid"), (5, "renderer=raster"), (6, "renderer=gpu"),
          (7, "renderer=gpu-bvh")]),
        ("query-execution-toggles", ["--renderer", "gpu", "--path-temporal"],
         [(f, f"execution={'wavefront' if f % 2 else 'megakernel'}") for f in range(1, 8)]),
        ("own-bvh-execution-toggles", ["--renderer", "gpu-bvh"],
         [(f, f"execution={'wavefront' if f % 2 else 'megakernel'}") for f in range(1, 8)]),
        ("pipeline-execution-toggles", ["--renderer", "gpu-pipeline", "--path-temporal"],
         [(f, f"execution={'wavefront' if f % 2 else 'iterative'}") for f in range(1, 8)]),
        ("own-bvh-builder-and-layout", ["--renderer", "gpu-bvh", "--path-execution", "wavefront"],
         [(1, "builder=gpu"), (2, "builder=cpu"), (3, "layout=bvh8"), (4, "layout=bvh4"),
          (5, "layout=binary"), (6, "builder=gpu")]),
        ("wavefront-resize", ["--renderer", "gpu-bvh", "--path-execution", "wavefront", "--path-temporal"],
         [(2, "resize=641x359"), (4, "resize=320x181"), (6, "resize=1600x900")]),
        ("pipeline-wavefront-resize", ["--renderer", "gpu-pipeline", "--path-execution", "wavefront"],
         [(2, "resize=641x359"), (4, "resize=1600x900")]),
        ("wavefront-scene-replacement", ["--renderer", "gpu", "--path-execution", "wavefront", "--path-temporal"],
         [(2, f"scene={scene}"), (3, "yaw=40"), (5, f"scene={scene}")]),
        ("spf-changes", ["--renderer", "gpu-bvh", "--path-execution", "wavefront"],
         [(1, "spf=4"), (2, "spf=1"), (3, "spf=16"), (5, "spf=3")]),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--only", help="run the scenarios whose name contains this text")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed, skipped, failures = [], [], []
    for name, initial, actions in scenarios(args.scene.resolve()):
        if args.only and args.only not in name:
            continue
        output = args.output_dir / f"{name}.pfm"
        command = [args.exe, args.scene, *initial, "--frame", "10", "--pfm", output,
                   "--bounces", "3", "--no-ui", "--no-vsync"]
        for frame, assignment in actions:
            command += ["--at-frame", str(frame), assignment]
        try:
            capture_harness.run(command, output, args.scene.parents[2])
            passed.append(name)
            print(f"PASS {name}")
        except SystemExit as stop:
            if stop.code != capture_harness.SKIP:
                raise
            skipped.append(name)
            print(f"SKIP {name}")
        except capture_harness.CaptureError as error:
            failures.append(name)
            print(f"FAIL {name}: {error}")
    print(f"{len(passed)} passed, {len(skipped)} skipped, {len(failures)} failed")
    if failures:
        raise SystemExit(1)
    if not passed:
        raise SystemExit(capture_harness.SKIP)


if __name__ == "__main__":
    main()
