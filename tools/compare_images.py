"""Compare two linear renders as converged RGB estimates.

Images are averaged over square blocks to reduce sampling noise. Partial blocks at the
right and bottom edges are included. PFM and OpenEXR input is scene-linear; formats read through
Pillow are normalized RGB and should only be used for display-image comparisons.
"""
import argparse
import math
import sys

import numpy as np


def read_pfm(path):
    with open(path, "rb") as source:
        kind = source.readline().strip()
        if kind not in (b"PF", b"Pf"):
            raise ValueError("PFM magic must be PF or Pf")
        dimensions = source.readline().split()
        if len(dimensions) != 2:
            raise ValueError("PFM dimensions must contain width and height")
        width, height = map(int, dimensions)
        if width <= 0 or height <= 0:
            raise ValueError("PFM dimensions must be positive")
        scale = float(source.readline().strip())
        if not math.isfinite(scale) or scale == 0.0:
            raise ValueError("PFM scale must be finite and nonzero")
        channels = 3 if kind == b"PF" else 1
        expected = width * height * channels
        data = np.fromfile(source, dtype="<f4" if scale < 0 else ">f4")
    if data.size != expected:
        raise ValueError(f"PFM contains {data.size} values; expected {expected}")
    image = data.reshape(height, width, channels)[::-1]
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def read_exr(path):
    """An OpenEXR capture as Basalt writes it (src/pt/ImageFile.h): single-part scanline,
    uncompressed, FLOAT channels B, G, R, rows from the top. Returns rows from the top, like
    read_pfm, and refuses anything else."""
    import struct
    with open(path, "rb") as source:
        data = source.read()
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != 20000630 or version != 2:
        raise ValueError("not a single-part scanline OpenEXR file")
    at, attributes = 8, {}

    def text():
        nonlocal at
        end = data.index(b"\0", at)
        value = data[at:end].decode("ascii")
        at = end + 1
        return value

    while True:
        name = text()
        if not name:
            break
        kind = text()
        (size,) = struct.unpack_from("<i", data, at)
        attributes[name] = (kind, data[at + 4:at + 4 + size])
        at += 4 + size
    channels, value = [], attributes["channels"][1]
    offset = 0
    while value[offset] != 0:
        end = value.index(b"\0", offset)
        name = value[offset:end].decode("ascii")
        pixel_type, _, xs, ys = struct.unpack_from("<iIii", value, end + 1)
        if pixel_type != 2 or xs != 1 or ys != 1:
            raise ValueError("only unsampled FLOAT channels")
        channels.append(name)
        offset = end + 1 + 16
    if channels != ["B", "G", "R"] or attributes["compression"][1] != b"\0" or attributes["lineOrder"][1] != b"\0":
        raise ValueError("only uncompressed increasing-y B, G, R files")
    x0, y0, x1, y1 = struct.unpack("<iiii", attributes["dataWindow"][1])
    width, height = x1 - x0 + 1, y1 - y0 + 1
    offsets = struct.unpack_from(f"<{height}Q", data, at)
    image = np.empty((height, width, 3), dtype=np.float32)
    for chunk in range(height):
        y, size = struct.unpack_from("<ii", data, offsets[chunk])
        planes = np.frombuffer(data, dtype="<f4", count=width * 3, offset=offsets[chunk] + 8).reshape(3, width)
        image[y] = planes[::-1].T  # B, G, R -> R, G, B
    return image


def read(path):
    if path.lower().endswith(".pfm"):
        return read_pfm(path)
    if path.lower().endswith(".exr"):
        return read_exr(path)
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def blocks(image, size):
    if size <= 0:
        raise ValueError("block size must be positive")
    height, width, channels = image.shape
    rows = (height + size - 1) // size
    columns = (width + size - 1) // size
    result = np.empty((rows, columns, channels), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            region = image[row * size:min((row + 1) * size, height),
                           column * size:min((column + 1) * size, width)]
            result[row, column] = region.mean(axis=(0, 1), dtype=np.float64)
    return result


def compare(a, b, block_size, dark_floor=1e-3):
    if a.shape != b.shape:
        raise ValueError(f"size mismatch: {a.shape} against {b.shape}")
    if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"images must be nonempty RGB arrays, got {a.shape}")
    for name, image in (("A", a), ("B", b)):
        bad = ~np.isfinite(image)
        if bad.any():
            raise ValueError(f"{name} has {int(bad.any(axis=2).sum())} non-finite pixels")

    ba, bb = blocks(a.astype(np.float64), block_size), blocks(b.astype(np.float64), block_size)
    difference = np.abs(ba - bb)
    scale = np.maximum((np.abs(ba) + np.abs(bb)) * 0.5, dark_floor)
    relative_rgb = difference / scale
    block_relative = relative_rgb.mean(axis=2)
    aggregate = float(difference.sum() / scale.sum())
    percentile99 = float(np.percentile(block_relative, 99))
    worst = np.unravel_index(np.argmax(block_relative), block_relative.shape)
    weights = np.array([0.2126, 0.7152, 0.0722])
    luminance_a, luminance_b = ba @ weights, bb @ weights
    energy_a = float(np.mean(a, dtype=np.float64))
    energy_b = float(np.mean(b, dtype=np.float64))
    energy_error = abs(energy_a - energy_b) / max((abs(energy_a) + abs(energy_b)) * 0.5, dark_floor)
    return {"aggregate": aggregate, "percentile99": percentile99,
            "worst": float(block_relative[worst]), "worst_index": worst,
            "energy_error": energy_error, "mean_rgb_energy_a": energy_a, "mean_rgb_energy_b": energy_b,
            "mean_absolute_rgb": float(difference.mean()),
            "mean_luminance_a": float(luminance_a.mean()),
            "mean_luminance_b": float(luminance_b.mean()),
            "block_relative": block_relative}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None,
                        help="maximum aggregate symmetric relative RGB error")
    parser.add_argument("--percentile-threshold", type=float, default=None,
                        help="maximum 99th-percentile block RGB error")
    parser.add_argument("--energy-threshold", type=float, default=None,
                        help="maximum symmetric relative mean RGB energy error")
    parser.add_argument("--worst-threshold", type=float, default=None,
                        help="maximum worst block RGB error")
    parser.add_argument("--dark-floor", type=float, default=1e-3)
    parser.add_argument("--heat", default=None)
    args = parser.parse_args()
    if args.block <= 0 or args.dark_floor <= 0 or not math.isfinite(args.dark_floor):
        parser.error("--block and --dark-floor must be positive")
    try:
        a, b = read(args.a), read(args.b)
        result = compare(a, b, args.block, args.dark_floor)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2

    row, column = result["worst_index"]
    print(f"mean relative RGB error {result['aggregate']:.4f}  "
          f"99th percentile {result['percentile99']:.3f}  "
          f"worst block {result['worst']:.3f} at ({column * args.block}, {row * args.block})  "
          f"energy error {result['energy_error']:.4f}  "
          f"mean absolute RGB {result['mean_absolute_rgb']:.5f}  "
          f"mean luminance {result['mean_luminance_a']:.4f} / {result['mean_luminance_b']:.4f}")
    if args.heat:
        from PIL import Image
        heat = np.clip(result["block_relative"] / 0.1, 0.0, 1.0)
        rgb = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=2)
        expanded = Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (rgb.shape[1] * args.block, rgb.shape[0] * args.block), Image.Resampling.NEAREST)
        expanded.crop((0, 0, a.shape[1], a.shape[0])).save(args.heat)

    failures = []
    for name, value, limit in (("aggregate", result["aggregate"], args.threshold),
                               ("mean RGB energy", result["energy_error"], args.energy_threshold),
                               ("99th percentile", result["percentile99"], args.percentile_threshold),
                               ("worst block", result["worst"], args.worst_threshold)):
        if limit is not None and value > limit:
            failures.append(f"{name} {value:.6g} exceeds {limit}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
