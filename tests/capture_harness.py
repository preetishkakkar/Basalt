"""Runs basalt.exe captures for the image conformance tests.

basalt.exe is a GUI-subsystem program: it has no stdout, and writes its log to the file
named by --log. Every capture therefore gets its own log, which is scanned in full for
Vulkan validation messages and for the explicit capability errors that mean SKIP.
"""
import shutil
import subprocess
import time
from pathlib import Path

SKIP = 77

# Errors main.cpp raises when the selected device lacks a renderer's capability.
CAPABILITY_ERRORS = (
    "requires Vulkan ray queries on the selected device",
    "requires VK_KHR_ray_tracing_pipeline on the selected device",
    "subgroup queue allocation needs subgroup arithmetic and ballot in compute",
)


class CaptureError(RuntimeError):
    pass


def validation_messages(log_text):
    return [line for line in log_text.splitlines()
            if "validation:" in line and ("] error:" in line or "] warning:" in line)]


def run(command, output, cwd):
    """Runs one capture. Raises SystemExit(77) for a missing capability, CaptureError on
    failure or on any validation message, and returns the log text otherwise."""
    output = Path(output)
    log_path = output.with_suffix(".log")
    if output.exists():
        output.unlink()
    if log_path.exists():
        log_path.unlink()
    command = [str(part) for part in command] + ["--log", str(log_path)]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    messages = validation_messages(log)
    if result.returncode or not output.exists():
        if any(reason in log for reason in CAPABILITY_ERRORS) and not messages:
            print(f"SKIP: {next(r for r in CAPABILITY_ERRORS if r in log)}")
            raise SystemExit(SKIP)
        raise CaptureError(f"capture failed ({result.returncode}), log {log_path}:\n{log}")
    if messages:
        raise CaptureError(f"Vulkan validation message in {log_path}:\n" + "\n".join(messages))
    return log


def preserve_failure(captures, label):
    """Copies each capture with its log, guides and metadata aside, so an intermittent
    failure can be diagnosed after later runs overwrite the originals."""
    captures = [Path(c) for c in captures]
    if not captures:
        return None
    target = captures[0].parent / f"failed-{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    for capture in captures:
        for related in capture.parent.glob(capture.stem + "*"):
            if related.is_file():
                shutil.copy2(related, target / related.name)
    print(f"preserved failing captures in {target}")
    return target
