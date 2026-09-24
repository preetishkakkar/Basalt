#!/usr/bin/env python3
"""Image-level conformance for the production full ray-pipeline renderer."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import compare_images
import capture_harness


def capture(exe, scene, output, renderer, seed, spp, temporal=False, execution=None,
            switch=False):
    command = [str(exe), str(scene), "--renderer", renderer, "--spp", str(spp),
               "--spf", "4", "--bounces", "4", "--seed", str(seed),
               "--pfm", str(output), "--no-ui", "--no-vsync"]
    if temporal:
        command.append("--path-temporal")
    if execution:
        command += ["--path-execution", "megakernel" if switch else execution]
    if switch:
        command += ["--switch-path-execution-frame", "1"]
    capture_harness.run(command, output, scene.parents[2])


def compare(a, b, label):
    result = compare_images.compare(compare_images.read_pfm(str(a)),
                                    compare_images.read_pfm(str(b)), 8, 1e-3)
    if (result["aggregate"] > 0.001 or result["energy_error"] > 0.001 or
            result["percentile99"] > 0.01 or result["worst"] > 0.03):
        capture_harness.preserve_failure([a, b], label.replace(" ", "-"))
        raise RuntimeError(f"{label} exceeds fixed agreement gates: {result}")
    print(f"PASS: {label}; aggregate={result['aggregate']:.6g}, worst={result['worst']:.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in (3, 17, 31, 97):
        query = args.output_dir / f"query-{seed}.pfm"
        pipeline = args.output_dir / f"pipeline-{seed}.pfm"
        capture(args.exe, args.scene, query, "gpu", seed, 16)
        capture(args.exe, args.scene, pipeline, "gpu-pipeline", seed, 16)
        compare(query, pipeline, f"ray pipeline seed {seed}")
    query = args.output_dir / "query-32spp.pfm"
    pipeline = args.output_dir / "pipeline-32spp.pfm"
    capture(args.exe, args.scene, query, "gpu", 31, 32)
    capture(args.exe, args.scene, pipeline, "gpu-pipeline", 31, 32)
    compare(query, pipeline, "ray pipeline doubled-SPP")
    query = args.output_dir / "query-17spp.pfm"
    pipeline = args.output_dir / "pipeline-17spp.pfm"
    capture(args.exe, args.scene, query, "gpu", 17, 17)
    capture(args.exe, args.scene, pipeline, "gpu-pipeline", 17, 17)
    compare(query, pipeline, "ray pipeline non-divisible target SPP")
    temporal = args.output_dir / "pipeline-temporal.pfm"
    capture(args.exe, args.scene, temporal, "gpu-pipeline", 31, 16, temporal=True)
    print("PASS: ray-pipeline temporal display capture completed validation-clean")
    wavefront = args.output_dir / "pipeline-wavefront.pfm"
    capture(args.exe, args.scene, wavefront, "gpu-pipeline", 31, 16,
            execution="wavefront")
    compare(args.output_dir / "pipeline-31.pfm", wavefront,
            "pipeline iterative versus wavefront")
    transitioned = args.output_dir / "pipeline-transition.pfm"
    capture(args.exe, args.scene, transitioned, "gpu-pipeline", 31, 16,
            execution="wavefront", switch=True)
    compare(args.output_dir / "pipeline-31.pfm", transitioned,
            "pipeline execution switch")


if __name__ == "__main__":
    main()
