#!/usr/bin/env python3
"""Negative CLI test: basalt.exe must exit nonzero and log a specific error.

A bare WILL_FAIL passes for any failure, including a crash or an unrelated error;
this checks the process reported the expected reason and wrote no capture.

usage: expect_capture_error.py --exe basalt.exe --expect TEXT --output-dir DIR -- ARGS...
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    split = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--expect", required=True, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(sys.argv[1:split])
    forwarded = sys.argv[split + 1:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.name}.log"
    capture = args.output_dir / f"{args.name}.pfm"
    for stale in (log_path, capture):
        if stale.exists():
            stale.unlink()
    command = [str(args.exe), *forwarded, "--pfm", str(capture), "--log", str(log_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if result.returncode == 0:
        raise SystemExit(f"FAIL: expected a nonzero exit, got 0\n{log}")
    if capture.exists():
        raise SystemExit(f"FAIL: a capture was written despite the error: {capture}")
    missing = [text for text in args.expect if text not in log]
    if missing:
        raise SystemExit(f"FAIL: exit {result.returncode} but log lacks {missing}:\n{log}")
    print(f"PASS: exit {result.returncode} with expected error")


if __name__ == "__main__":
    main()
