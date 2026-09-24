#!/usr/bin/env python3
"""OpenEXR captures (V8): the same deterministic render written as PFM and as OpenEXR reads
back bit for bit identical, for the headless CPU tracer (raw) and, when --exe is given and the
device offers a GPU tracer, for the application's raw, albedo and normal outputs; metadata names
each output's format.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness


def identical(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def check(label, pfm, exr):
    same = identical(compare_images.read_pfm(str(pfm)), compare_images.read_exr(str(exr)))
    print(f"{'PASS' if same else 'FAIL'}: {label}: OpenEXR {'equals' if same else 'differs from'} PFM bit for bit",
          flush=True)
    return same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-cli", required=True, type=Path)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    outputs = {}
    for extension in ("pfm", "exr"):
        path = args.output_dir / f"headless.{extension}"
        subprocess.run([str(args.pt_cli), str(args.scene), "--output", str(path), "--width", "24", "--height", "16",
                        "--spp", "4", "--bounces", "3", "--threads", "2"], check=True, capture_output=True)
        outputs[extension] = path
        kind = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))["outputs"][0]["kind"]
        if kind != f"raw-linear-rgb-{extension}":
            print(f"FAIL: headless {extension} metadata names {kind}")
            failures += 1
    failures += not check("headless CPU raw", outputs["pfm"], outputs["exr"])

    if args.exe:
        files = {}
        for extension in ("pfm", "exr"):
            raw, albedo, normal = (args.output_dir / f"gpu-{plane}.{extension}" for plane in ("raw", "albedo", "normal"))
            command = [str(args.exe), str(args.scene), "--renderer", "gpu-bvh", "--size", "64x36", "--spp", "4",
                       "--bounces", "3", "--pfm", str(raw), "--albedo-pfm", str(albedo), "--normal-pfm", str(normal),
                       "--no-ui", "--no-vsync"]
            try:
                capture_harness.run(command, raw, args.scene.parents[2])
            except SystemExit as skip:
                if skip.code != capture_harness.SKIP:
                    raise
                print("SKIP: GPU OpenEXR captures (capability)")
                break
            files[extension] = (raw, albedo, normal)
            kinds = [o["kind"] for o in json.loads(Path(str(raw) + ".json").read_text(encoding="utf-8"))["outputs"]]
            if not all(k.endswith(f"-{extension}") for k in kinds if k != "display-png"):
                print(f"FAIL: GPU {extension} metadata names {kinds}")
                failures += 1
        if len(files) == 2:
            for label, pfm, exr in zip(("raw", "albedo", "normal"), files["pfm"], files["exr"]):
                failures += not check(f"GPU {label}", pfm, exr)
    if failures:
        print(f"FAIL: {failures} OpenEXR checks")
        return 1
    print("PASS: OpenEXR captures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
