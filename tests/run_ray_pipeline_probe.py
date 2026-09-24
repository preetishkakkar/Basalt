"""Compile and execute Basalt's V6.2 R1 four-stage ray-pipeline gate.

The compiler repository supplies its generic Vulkan test runner. This script owns the
Basalt shader, scene, expected results, and pass/fail policy. A missing pipeline device
is a skip (77); any failure on a capable device, including validation output, is fatal.
"""
import argparse
import importlib.util
import struct
import sys
from pathlib import Path


def load_harness(path):
    spec = importlib.util.spec_from_file_location("m2v_ray_pipeline_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler")
    parser.add_argument("runner")
    parser.add_argument("compiler_tests", help="Metal2VulkanSpirv tests/ray_pipeline_tests.py")
    parser.add_argument("artifacts")
    parser.add_argument("--spirv-val", default="spirv-val")
    args = parser.parse_args()

    h = load_harness(Path(args.compiler_tests))
    artifacts = Path(args.artifacts)
    source = Path(__file__).with_name("ray_pipeline_probe.metal").read_text(encoding="utf-8")
    suite = h.Suite(args.compiler, args.runner, artifacts, 4)
    stages = suite.compile("basalt-r1", source.removeprefix(h.HEADER),
                           ["probeGenerate", "probeMiss", "probeClosest", "probeAlpha"])

    import subprocess
    for stage in stages:
        checked = subprocess.run([args.spirv_val, "--target-env", "vulkan1.3", stage["spirv"]],
                                 capture_output=True, text=True)
        if checked.returncode:
            raise RuntimeError(f"spirv-val failed for {stage['entry']}:\n{checked.stdout}{checked.stderr}")

    scene = h.Scene()
    geometry = scene.triangles([
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))
    ], opaque=False)
    scene.instance(scene.bottom([geometry]), custom=37)
    # Center has interpolated alpha 0.75. Rays 0/1 prove accept/reject with identical
    # geometry; ray 2 is an ordinary miss; ray 3 hits near the fully opaque first vertex.
    rays = [
        ((0.0, 0.0, -1.0, 5.0), (0.0, 0.0, 1.0, 0.50), (0x10203040, 0, 0, 0)),
        ((0.0, 0.0, -1.0, 5.0), (0.0, 0.0, 1.0, 0.80), (0x50607080, 0, 0, 0)),
        ((2.0, 2.0, -1.0, 5.0), (0.0, 0.0, 1.0, 0.00), (0x90A0B0C0, 0, 0, 0)),
        ((-0.75, -0.75, -1.0, 5.0), (0.0, 0.0, 1.0, 0.90), (0xD0E0F001, 0, 0, 0)),
    ]
    ray_bytes = b"".join(struct.pack("<8f4I", *(a + b + c)) for a, b, c in rays)
    expected_found = [1, 0, 0, 1]
    expected_tokens = [r[2][0] for r in rays]

    def make_job(_device):
        case = artifacts / "basalt-r1"
        resources = [
            {"kind": "acceleration_structure", "binding": 0},
            h.buffer_resource(case, 1, "rays", ray_bytes),
            h.buffer_resource(case, 2, "results", size=4 * 32, output=True),
            h.buffer_resource(case, 3, "vertex-alpha", struct.pack("<4f", 1.0, 0.0, 1.0, 0.0)),
        ]
        job = {
            "stages": stages,
            "groups": [
                {"type": "general", "general": 0},
                {"type": "general", "general": 1},
                {"type": "triangles", "closest_hit": 2, "any_hit": 3},
            ],
            "sbt": {"raygen": {"group": 0}, "miss": [{"group": 1}], "hit": [{"group": 2}]},
            "max_recursion": 1,
            "launch": [4, 1, 1],
            "scene": scene.write(case),
            "resources": resources,
        }
        return job, {"results": case / "results.out"}

    def verify(data):
        raw = data["results"]
        problems = []
        for index in range(4):
            t, u, v, found = struct.unpack_from("<4f", raw, index * 32)
            instance, primitive, ambiguous, token = struct.unpack_from("<4I", raw, index * 32 + 16)
            if int(found) != expected_found[index]:
                problems.append((index, "found", found, expected_found[index]))
            if token != expected_tokens[index]:
                problems.append((index, "payload token", hex(token), hex(expected_tokens[index])))
            if expected_found[index]:
                if abs(t - 1.0) > 1e-5 or instance != 37 or primitive != 0:
                    problems.append((index, "hit payload", (t, instance, primitive)))
                if min(u, v, 1.0 - u - v) < -1e-5:
                    problems.append((index, "barycentrics", (u, v)))
            elif t != -1.0:
                problems.append((index, "miss distance", t))
            if ambiguous != 0:
                problems.append((index, "unexpected ambiguity", ambiguous))
        if not problems:
            print("  basalt-r1: nearest hit, alpha rejection, miss, IDs, barycentrics and payload tokens passed")
        return problems

    suite.each_device("basalt-r1", make_job, verify)
    if not suite.ran:
        return 77
    print("PASS Basalt V6.2 R1 ray-pipeline feasibility gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
