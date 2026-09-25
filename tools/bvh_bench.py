#!/usr/bin/env python3
"""Compares the software BVH builders on the corpus: build time, SAH cost and trace time.

For every scene x builder x width it renders one untimed warm-up, then --repeats unprofiled
captures interleaved across the combinations in alternating order (the build and trace
numbers; trace time is the median of the last quarter of the frames), and --profile-runs
captures with --gpu-profile (the wavefront stage times, and each GPU builder's stage
timestamps). It reads each capture's metadata and reports medians as a Markdown table and
report.json; "GPU build ms" is the sum of the builder's stage timestamps.
Missing scene assets and combinations the program rejects are skipped and listed.

With --animate the scene moves every frame (--spp counts frames) and the table adds the
per-frame BVH update's wall and GPU time (median of the last quarter of the frames); a GPU
parallel LBVH refits and rebuilds its TLAS, the others (and --bvh-update rebuild) rebuild.

Examples:
  python tools/bvh_bench.py --builders cpu,gpu-serial --widths binary,bvh8 --repeats 5
  python tools/bvh_bench.py --builders gpu-lbvh --widths binary,bvh8 --animate both --spp 128
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENES = {
    "materials": ["tests/data/materials.gltf"],
    "raycone": ["tests/data/raycone.gltf"],
    "manylights": ["tests/data/manylights.gltf", "tests/data/manylights_dim.hdr"],
    "sponza": ["assets/Sponza/Sponza.gltf", "assets/court.hdr"],
    "helmet": ["assets/DamagedHelmet.glb", "assets/court.hdr"],
}
BUILDERS = ["cpu", "gpu-serial", "gpu-lbvh", "gpu-ploc"]
WIDTHS = ["binary", "bvh4", "bvh8"]
BUILD_KEYS = ["bvh_build_ms", "bvh_collapse_ms", "bvh_sah_cost", "bvh_sah_cost_tlas", "bvh_layout_sah_cost",
              "bvh_layout_sah_cost_tlas", "bvh_nodes", "bvh_layout_nodes", "bvh_depth_tlas", "bvh_depth_blas"]
STAGE_KEYS = ["stage_ms_intersect", "stage_ms_shade", "stage_ms_shadow"]


class Rejected(Exception):
    pass


def run(exe, scene, builder, width, args, output, profile):
    command = [str(exe), *[str(ROOT / part) for part in SCENES[scene]], "--renderer", "gpu-bvh",
               "--bvh-builder", builder, "--bvh-width", width, "--path-execution", args.execution,
               "--bvh-update", args.bvh_update, "--animate", args.animate,
               "--size", args.size, "--spp", str(args.spp), "--seed", "11", "--no-ui", "--no-vsync",
               "--no-validation", "--pfm", str(output), "--log", str(output.with_suffix(".log"))]
    if profile:
        command.append("--gpu-profile")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    log = output.with_suffix(".log")
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    metadata = Path(str(output) + ".json")
    if result.returncode or not metadata.exists():
        rejection = [line for line in (text + result.stderr).splitlines() if "require" in line or "unknown BVH builder" in line]
        if rejection:
            raise Rejected(rejection[-1].strip())
        raise RuntimeError(f"run failed ({result.returncode}): {' '.join(command)}\n{text[-4000:]}")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    measured = data["measurements"]
    values = {key: measured[key] for key in BUILD_KEYS + STAGE_KEYS if key in measured}
    values.update({key: value for key, value in measured.items() if key.startswith("bvh_stage_ms_")})
    # Trace time over the last quarter of the frames: the GPU's clocks follow the work before
    # them, and a 1 s serial build warms the GPU up that a 1 ms parallel one does not, so the
    # first frames of identical trees differ by up to 9% (Sponza).
    series = data["series"].get("fresh_trace_gpu_ms", [])
    if series:
        values["trace_ms"] = statistics.median(series[len(series) * 3 // 4:])
    # Animated: the per-frame BVH updates, over the same last quarter.
    for key in ("bvh_update_ms", "bvh_update_gpu_ms"):
        updates = data["series"].get(key, [])
        if updates:
            values[key] = statistics.median(updates[len(updates) * 3 // 4:])
    return values, data["device"]["name"], data["source"]["revision"]


def medians(runs):
    keys = sorted({key for values in runs for key in values})
    return {key: statistics.median([values[key] for values in runs if key in values]) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=ROOT / "build" / "basalt.exe")
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--builders", default=",".join(BUILDERS))
    parser.add_argument("--widths", default=",".join(WIDTHS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-runs", type=int, default=1, help="extra --gpu-profile runs for stage times")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--spp", type=int, default=256, help="samples; with --animate, animated frames")
    parser.add_argument("--execution", default="wavefront", choices=["wavefront", "megakernel"])
    parser.add_argument("--animate", default="off", choices=["off", "instances", "vertices", "both"])
    parser.add_argument("--bvh-update", default="on-change", choices=["on-change", "rebuild", "refit"])
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "bench" / "bvh")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "exe": str(args.exe.resolve()), "size": args.size,
              "spp": args.spp, "repeats": args.repeats, "profile_runs": args.profile_runs,
              "execution": args.execution, "animate": args.animate, "bvh_update": args.bvh_update,
              "results": [], "skipped": []}
    for scene in args.scenes.split(","):
        missing = [part for part in SCENES[scene] if not (ROOT / part).exists()]
        if missing:
            report["skipped"].append({"scene": scene, "reason": f"missing {', '.join(missing)}"})
            print(f"SKIP {scene}: missing {', '.join(missing)}", flush=True)
            continue
        # One untimed run per combination (and the rejected ones found), then the timed repeats
        # interleaved in alternating order, so no combination always follows a long GPU build
        # (a 1 s serial build leaves the GPU clocked up for the trace after it).
        combos = []
        for builder in args.builders.split(","):
            for width in args.widths.split(","):
                label = f"{scene}-{builder}-{width}"
                try:
                    run(args.exe, scene, builder, width, args, args.output / f"{label}-warmup.pfm", False)
                    combos.append((builder, width, label))
                except Rejected as rejection:
                    report["skipped"].append({"scene": scene, "builder": builder, "width": width, "reason": str(rejection)})
                    print(f"SKIP {label}: {rejection}", flush=True)
        plain = {label: [] for _, _, label in combos}
        profiled = {label: [] for _, _, label in combos}
        device = revision = ""
        for repeat in range(args.repeats):
            for builder, width, label in (combos if repeat % 2 == 0 else combos[::-1]):
                values, device, revision = run(args.exe, scene, builder, width, args,
                                               args.output / f"{label}-{repeat}.pfm", False)
                plain[label].append(values)
        for repeat in range(args.profile_runs):
            for builder, width, label in combos:
                values, device, revision = run(args.exe, scene, builder, width, args,
                                               args.output / f"{label}-profile{repeat}.pfm", True)
                profiled[label].append({key: values[key] for key in STAGE_KEYS if key in values})
        for builder, width, label in combos:
            row = {"scene": scene, "builder": builder, "width": width, "device": device, "revision": revision,
                   "median": medians(plain[label]), "stages": medians(profiled[label]) if profiled[label] else {},
                   "runs": plain[label]}
            report["results"].append(row)
            m = row["median"]
            print(f"{label}: build {m.get('bvh_build_ms', 0):.2f} ms, SAH {m.get('bvh_layout_sah_cost', 0):.1f}, "
                  f"trace {m.get('trace_ms', 0):.3f} ms", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    stage_names = sorted({key for row in report["results"] for key in row["median"] if key.startswith("bvh_stage_ms_")})
    header = ["scene", "builder", "width", "build ms", "GPU build ms", "collapse ms", "SAH BLAS", "SAH TLAS", "layout SAH",
              "nodes", "depth", "trace ms", "update ms", "update GPU ms", "intersect ms", "shade ms",
              "shadow ms"] + [name[len("bvh_stage_ms_"):] + " ms" for name in stage_names]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
    for row in report["results"]:
        m, s = row["median"], row["stages"]
        gpu_build = sum(m.get(name, 0) for name in stage_names)
        cells = [row["scene"], row["builder"], row["width"], f"{m.get('bvh_build_ms', 0):.2f}",
                 f"{gpu_build:.3f}" if gpu_build else "-",
                 f"{m.get('bvh_collapse_ms', 0):.2f}", f"{m.get('bvh_sah_cost', 0):.2f}", f"{m.get('bvh_sah_cost_tlas', 0):.2f}",
                 f"{m.get('bvh_layout_sah_cost', 0):.2f}", f"{m.get('bvh_layout_nodes', 0):.0f}",
                 f"{m.get('bvh_depth_tlas', 0):.0f}+{m.get('bvh_depth_blas', 0):.0f}", f"{m.get('trace_ms', 0):.3f}",
                 f"{m['bvh_update_ms']:.3f}" if 'bvh_update_ms' in m else "-",
                 f"{m['bvh_update_gpu_ms']:.3f}" if 'bvh_update_gpu_ms' in m else "-",
                 f"{s.get('stage_ms_intersect', 0):.3f}", f"{s.get('stage_ms_shade', 0):.3f}", f"{s.get('stage_ms_shadow', 0):.3f}"]
        cells += [f"{m.get(name, 0):.3f}" for name in stage_names]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (args.output / "report.md").write_text(table + "\n", encoding="utf-8")
    devices = sorted({row["device"] for row in report["results"]})
    print(f"\nDevice: {', '.join(devices)}; {args.repeats} repeats (median), {args.size}, {args.spp} spp, "
          f"{args.execution}, animate {args.animate}, update {args.bvh_update}\n")
    print(table)
    for skip in report["skipped"]:
        print(f"skipped: {skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
