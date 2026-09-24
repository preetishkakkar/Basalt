#!/usr/bin/env python3
"""Environment sun extraction must move light, never add or lose it.

Writes an .hdr with a small bright sun over a sky, then renders the same scene with the
headless CPU path tracer twice: with the sun left in the image (--environment-sun keep, found
only by environment importance sampling) and with it extracted into the analytic disc
(extract). Both are unbiased estimates of one image, so they must agree within their noise,
and the extracted run must be recorded as having found the sun. A third render of the same sky
without its sun isolates what the sun contributes, which the two must also agree on: that is
the quantity a wrong extraction would change.

usage: environment_sun.py basalt-pt-cli.exe scene.gltf output-directory
"""
import json
import math
import struct
import subprocess
import sys
from pathlib import Path


def write_hdr(path, width, height, sun_direction, sun_radius_degrees, sun_radiance, with_sun=True):
    """A flat (uncompressed) Radiance RGBE image, laid out as the renderers read it."""
    cos_radius = math.cos(math.radians(sun_radius_degrees))
    rows = []
    for y in range(height):
        theta = (y + 0.5) / height * math.pi
        row = bytearray()
        for x in range(width):
            phi = ((x + 0.5) / width - 0.5) * 2.0 * math.pi
            d = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            sky = 0.02 + 0.06 * max(0.0, d[1]) if d[1] >= 0.0 else 0.01
            rgb = (sky * 0.75, sky * 0.9, sky * 1.1)
            if with_sun and sum(a * b for a, b in zip(d, sun_direction)) >= cos_radius:
                rgb = sun_radiance
            peak = max(rgb)
            if peak < 1e-32:
                row += bytes(4)
                continue
            mantissa, exponent = math.frexp(peak)
            scale = mantissa * 256.0 / peak
            row += bytes([min(255, int(c * scale)) for c in rgb] + [exponent + 128])
        rows.append(bytes(row))
    with open(path, 'wb') as f:
        f.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n')
        f.write(f'-Y {height} +X {width}\n'.encode())
        for row in rows:
            f.write(row)


def read_pfm(path):
    data = Path(path).read_bytes()
    lines = data.split(b'\n', 3)
    width, height = map(int, lines[1].split())
    scale = float(lines[2])
    fmt = '<' if scale < 0 else '>'
    values = struct.unpack(f'{fmt}{width * height * 3}f', lines[3][:width * height * 12])
    return width, height, values


def luminance(values, width, x0, y0, x1, y1):
    total = 0.0
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = (y * width + x) * 3
            total += 0.2126 * values[i] + 0.7152 * values[i + 1] + 0.0722 * values[i + 2]
    return total / ((x1 - x0) * (y1 - y0))


def main():
    cli, scene, output = str(Path(sys.argv[1]).resolve()), sys.argv[2], Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    elevation, azimuth = math.radians(40.0), math.radians(30.0)
    direction = (math.cos(elevation) * math.cos(azimuth), math.sin(elevation), math.cos(elevation) * math.sin(azimuth))
    hdr, sunless = output / 'environment-sun.hdr', output / 'environment-sunless.hdr'
    write_hdr(hdr, 512, 256, direction, 1.5, (1500.0, 1400.0, 1250.0))
    write_hdr(sunless, 512, 256, direction, 1.5, (1500.0, 1400.0, 1250.0), with_sun=False)

    # The sun left in the image is found only through environment importance sampling, the
    # noisy estimator extraction replaces: it gets four times the samples, for a mean within
    # about 0.6% (one standard deviation), and the tolerance below allows four of those.
    width, height = 96, 54
    samples = {'keep': 32768, 'extract': 8192, 'none': 8192}
    images = {}
    for mode in ('keep', 'extract', 'none'):
        target = output / f'environment-sun-{mode}.pfm'
        result = subprocess.run([cli, scene, '--environment', str(sunless if mode == 'none' else hdr),
                                 '--environment-sun', 'keep' if mode == 'none' else mode,
                                 '--width', str(width), '--height', str(height), '--spp', str(samples[mode]),
                                 '--seed', '7', '--output', str(target)], capture_output=True, text=True)
        if result.returncode:
            print(result.stdout, result.stderr)
            print(f'FAIL: the {mode} render failed')
            return 1
        images[mode] = read_pfm(target)
        metadata = json.loads(Path(str(target) + '.json').read_text())
        notes = dict(metadata['notes']) if isinstance(metadata['notes'], list) else metadata['notes']
        expected = 'extracted' if mode == 'extract' else 'kept in the image'
        if notes.get('environment_sun') != expected:
            print(f'FAIL: {mode} recorded environment_sun {notes.get("environment_sun")!r}, expected {expected!r}')
            return 1

    keep, extract, none = images['keep'][2], images['extract'][2], images['none'][2]
    whole_keep = luminance(keep, width, 0, 0, width, height)
    whole_extract = luminance(extract, width, 0, 0, width, height)
    whole_none = luminance(none, width, 0, 0, width, height)
    sun_keep, sun_extract = whole_keep - whole_none, whole_extract - whole_none
    whole = abs(sun_extract / sun_keep - 1.0)
    print(f'mean luminance: keep {whole_keep:.5f}, extract {whole_extract:.5f}, without the sun {whole_none:.5f}')
    print(f"the sun's contribution: keep {sun_keep:.5f}, extract {sun_extract:.5f} ({whole * 100:.2f}% apart)")
    failures = 0
    if not sun_keep > 0.2 * whole_keep:
        print("FAIL: the sun lights too little of the image for the test to mean anything")
        failures += 1
    if whole > 0.025:
        print("FAIL: the sun's contributions differ by more than 2.5%")
        failures += 1
    # Blocks of 12 x 9 pixels: shadowed and lit regions must each keep their light.
    worst = 0.0
    for by in range(0, height, 9):
        for bx in range(0, width, 12):
            a = luminance(keep, width, bx, by, bx + 12, by + 9)
            b = luminance(extract, width, bx, by, bx + 12, by + 9)
            worst = max(worst, abs(a - b) / max(a, b, 0.05 * whole_keep))
    print(f'worst block difference: {worst * 100:.2f}%')
    if worst > 0.04:
        print('FAIL: a block differs by more than 4%')
        failures += 1
    print('PASS' if failures == 0 else f'{failures} checks failed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
