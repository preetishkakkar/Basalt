#!/usr/bin/env python3
"""Runs the V6.1 path-tracer benchmark manifest and reports paired timings.

A variant is LABEL=EXE:BACKEND[|ARG,ARG...], BACKEND naming an entry of the manifest's "backends". The
first variant is the reference for paired ratios. For every case each variant runs once
untimed (shaders, allocations and caches warm), then --repeats timed runs in alternating
order (ABAB..., then BABA...). A run's value is the median GPU trace time of its fresh
frames after the case's warm-up frames, from the capture's metadata; validation is off.

Example:
  python tools/benchmark_path_tracer.py --set anchor \
      --variant base=../Basalt-v61-baseline/build/basalt.exe:query-wavefront \
      --variant new=build/basalt.exe:query-wavefront --output build/bench/anchor
"""
import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_status(manifest, name):
    scene = manifest["scenes"][name]
    for relative, expected in scene["sha256"].items():
        path = ROOT / relative
        if not path.exists():
            return f"missing {relative}"
        if sha256(path) != expected:
            return f"changed {relative}"
    return None


def cases(manifest, which):
    if which in ("anchor", "all"):
        a = manifest["anchor"]
        yield {"set": "anchor", "scene": a["scene"], "size": a["size"], "spf": a["spf"], "bounces": a["bounces"],
               "spp": a["spp"], "seed": a["seed"], "warmup": 0, "motion": False}
    if which in ("grid", "all"):
        g = manifest["grid"]
        for scene, size, spf, bounces in itertools.product(g["scenes"], g["sizes"], g["spf"], g["bounces"]):
            yield {"set": "grid", "scene": scene, "size": size, "spf": spf, "bounces": bounces,
                   "spp": spf * g["frames"], "seed": g["seed"], "warmup": g["warmup_frames"], "motion": False}
    if which in ("motion", "all"):
        m = manifest["motion"]
        for scene in m["scenes"]:
            yield {"set": "motion", "scene": scene, "size": m["size"], "spf": m["spf"], "bounces": m["bounces"],
                   "frames": m["frames"], "spin": m["spin_degrees"], "seed": m["seed"],
                   "warmup": m["warmup_frames"], "motion": True}


def case_name(case):
    return f"{case['set']}-{case['scene']}-{case['size']}-spf{case['spf']}-b{case['bounces']}"


def run_once(manifest, exe, backend, case, output, profile=False, validation=False, extras=()):
    scene = manifest["scenes"][case["scene"]]
    b = manifest["backends"][backend]
    command = [str(exe), str(ROOT / scene["scene"])]
    if scene["environment"]:
        command.append(str(ROOT / scene["environment"]))
    command += ["--renderer", b["renderer"], "--path-execution", b["execution"], "--size", case["size"],
                "--spf", str(case["spf"]), "--bounces", str(case["bounces"]), "--seed", str(case["seed"]),
                "--no-ui", "--no-vsync", "--pfm", str(output), "--log", str(output.with_suffix(".log"))]
    if not validation:
        command.append("--no-validation")
    if profile:
        command.append("--gpu-profile")
    command += list(extras)
    if b["renderer"] == "gpu-bvh":
        command += ["--bvh-builder", b["builder"], "--bvh-width", b["layout"]]
    if case["motion"]:
        command += ["--spin", str(case["spin"]), "--frame", str(case["frames"])]
    else:
        command += ["--spp", str(case["spp"])]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
    metadata = Path(str(output) + ".json")
    if result.returncode or not metadata.exists():
        log = output.with_suffix(".log")
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        raise RuntimeError(f"run failed ({result.returncode}): {' '.join(command)}\n{text[-4000:]}")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    series = data["series"].get("fresh_trace_gpu_ms", [])
    timed = series[case["warmup"]:] if len(series) > case["warmup"] else series
    if not timed:
        raise RuntimeError(f"no fresh trace timings in {metadata}")
    return {"trace_ms": statistics.median(timed), "frames": len(timed),
            "last_fresh_frame_gpu_ms": data["measurements"].get("fresh_sample_frame_gpu_ms"),
            "queue_bytes": data["measurements"].get("wavefront_queue_bytes", 0),
            "measurements": data["measurements"] if profile else None,
            "series": {k: v for k, v in data["series"].items() if profile or k == "fresh_trace_gpu_ms"},
            "device": data["device"]["name"], "revision": data["source"]["revision"],
            "dirty": data["source"]["dirty_at_configure"]}


def summarize(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/benchmarks/v6_1_manifest.json")
    parser.add_argument("--set", choices=["anchor", "grid", "motion", "all"], default="anchor")
    parser.add_argument("--variant", action="append", required=True,
                        help="LABEL=EXE:BACKEND[|ARG,ARG...] (extra command-line arguments)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scenes", help="comma-separated scene filter")
    parser.add_argument("--case-filter", help="substring filter on case names")
    parser.add_argument("--profile", action="store_true", help="add --gpu-profile (instrumented runs)")
    parser.add_argument("--validation", action="store_true", help="keep Vulkan validation on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    variants = []
    for text in args.variant:
        label, rest = text.split("=", 1)
        rest, _, extra = rest.partition("|")
        exe, backend = rest.rsplit(":", 1)
        if backend not in manifest["backends"]:
            parser.error(f"unknown backend {backend}")
        variants.append((label, Path(exe).resolve(), backend, tuple(a for a in extra.split(",") if a)))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"manifest": str(args.manifest), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "variants": [{"label": l, "exe": str(e), "backend": b, "extra": list(x)} for l, e, b, x in variants],
              "repeats": args.repeats, "validation": args.validation, "profile": args.profile, "cases": []}
    scenes = set(args.scenes.split(",")) if args.scenes else None
    for case in cases(manifest, args.set):
        name = case_name(case)
        if (scenes and case["scene"] not in scenes) or (args.case_filter and args.case_filter not in name):
            continue
        entry = {"case": name, **case}
        status = scene_status(manifest, case["scene"])
        if status:
            entry["skip"] = status
            report["cases"].append(entry)
            print(f"SKIP {name}: {status}")
            continue
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        try:
            for label, exe, backend, extras in variants:  # warm-up, untimed
                run_once(manifest, exe, backend, case, folder / f"{label}-warmup.pfm", args.profile, args.validation,
                         extras)
            runs = {label: [] for label, _, _, _ in variants}
            details = {label: [] for label, _, _, _ in variants}
            for repeat in range(args.repeats):
                order = variants if repeat % 2 == 0 else list(reversed(variants))
                for label, exe, backend, extras in order:
                    result = run_once(manifest, exe, backend, case, folder / f"{label}-{repeat}.pfm",
                                      args.profile, args.validation, extras)
                    runs[label].append(result["trace_ms"])
                    details[label].append(result)
        except RuntimeError as error:
            entry["error"] = str(error)
            report["cases"].append(entry)
            print(f"FAIL {name}: {str(error)[:400]}")
            continue
        reference = variants[0][0]
        entry["results"] = {label: summarize(values) for label, values in runs.items()}
        entry["paired_ratio_to_" + reference] = {
            label: summarize([v / r for v, r in zip(values, runs[reference])])
            for label, values in runs.items() if label != reference}
        entry["details"] = details
        first = details[reference][0]
        entry["device"], entry["revision"] = first["device"], first["revision"]
        report["cases"].append(entry)
        line = "  ".join(f"{label} {s['median']:.3f} [{s['min']:.3f},{s['max']:.3f}]"
                         for label, s in entry["results"].items())
        ratios = "  ".join(f"{label}/{reference} {s['median']:.3f}"
                           for label, s in entry["paired_ratio_to_" + reference].items())
        print(f"{name}: {line}  {ratios}", flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    failed = [c for c in report["cases"] if "error" in c]
    print(f"report: {args.output / 'report.json'}; {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
