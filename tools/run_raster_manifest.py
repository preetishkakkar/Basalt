"""Run the named raster regression manifest against two clean source builds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


def revision(source):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"source tree is dirty: {source}\n{dirty}")
    return head


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_capture(executable, arguments, output, asset_root, log_output):
    command = [str(executable), *arguments, "--screenshot", str(output)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(command, cwd=asset_root, creationflags=flags)
    if completed.returncode:
        raise RuntimeError(f"capture exited {completed.returncode}: {command}")
    if not output.is_file():
        raise RuntimeError(f"capture did not create {output}")
    source_log = asset_root / "basalt.log"
    if not source_log.is_file():
        raise RuntimeError(f"capture did not create {source_log}")
    shutil.copy2(source_log, log_output)
    text = source_log.read_text(encoding="utf-8", errors="replace")
    if "[basalt] error:" in text:
        raise RuntimeError(f"capture logged an error; see {log_output}")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/raster_manifest.json")
    parser.add_argument("--baseline-exe", required=True, type=Path)
    parser.add_argument("--candidate-exe", required=True, type=Path)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--candidate-source", required=True, type=Path)
    parser.add_argument("--asset-root", default=".", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    baseline_revision = revision(args.baseline_source.resolve())
    candidate_revision = revision(args.candidate_source.resolve())
    expected_baseline = manifest["baseline_revision"]
    expected_candidate = manifest["candidate_revision"]
    if baseline_revision != expected_baseline:
        raise RuntimeError(f"baseline is {baseline_revision}, expected {expected_baseline}")
    if not candidate_revision.startswith(expected_candidate):
        raise RuntimeError(f"candidate is {candidate_revision}, expected prefix {expected_candidate}")

    asset_root = args.asset_root.resolve()
    output_root = args.output.resolve()
    records = []
    for case in manifest["cases"]:
        case_arguments = []
        for value in case["arguments"]:
            candidate = asset_root / value
            case_arguments.append(str(candidate) if candidate.exists() else value)
        case_arguments.extend(manifest["common_arguments"])
        images = {}
        commands = {}
        for label, executable in (("baseline", args.baseline_exe), ("candidate", args.candidate_exe)):
            directory = output_root / label
            directory.mkdir(parents=True, exist_ok=True)
            image_path = directory / f"{case['name']}.png"
            commands[label] = run_capture(executable.resolve(), case_arguments, image_path, asset_root,
                                          directory / f"{case['name']}.log")
            images[label] = image_path
        a = np.asarray(Image.open(images["baseline"]).convert("RGB"))
        b = np.asarray(Image.open(images["candidate"]).convert("RGB"))
        if a.shape != b.shape or not np.array_equal(a, b):
            changed = int(np.any(a != b, axis=2).sum()) if a.shape == b.shape else -1
            raise RuntimeError(f"{case['name']} differs ({changed} changed pixels)")
        records.append({"case": case["name"], "dimensions": [int(a.shape[1]), int(a.shape[0])],
                        "pixel_equal": True, "baseline_sha256": digest(images["baseline"]),
                        "candidate_sha256": digest(images["candidate"]), "commands": commands})
        print(f"PASS {case['name']}: {a.shape[1]}x{a.shape[0]}, pixel-identical")

    report = {"manifest": manifest["name"], "baseline_revision": baseline_revision,
              "candidate_revision": candidate_revision, "cases": records}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(records)} passed, report: {output_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
