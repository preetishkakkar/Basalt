"""Writes materials.gltf: a small scene with every alpha mode and an embedded texture.

- an opaque panel with a checker base colour,
- a double-sided alpha-masked panel whose checker has holes,
- a blended panel at 40% opacity,
all with the texture embedded as a base64 PNG, so the file is self-contained.

usage: python make_materials_gltf.py [output.gltf]
"""
import base64
import io
import json
import struct
import sys
import zlib


def png(width, height, pixel):
    rows = b''.join(b'\x00' + b''.join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


def checker(x, y):
    on = ((x // 4) + (y // 4)) % 2 == 0
    return (230, 190, 60, 255) if on else (40, 90, 200, 0)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'materials.gltf'
    # Three unit quads side by side, facing +Z, standing on y = 0.
    positions, normals, uvs, indices = [], [], [], []
    for panel in range(3):
        x0 = panel * 1.2 - 1.8
        base = len(positions)
        for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            positions.append((x0 + u, v, 0.0))
            normals.append((0.0, 0.0, 1.0))
            uvs.append((u * 2.0, 1.0 - v * 2.0))
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    blob = io.BytesIO()
    views = []
    def view(data, target=None):
        while blob.tell() % 4:
            blob.write(b'\x00')
        offset = blob.tell()
        blob.write(data)
        v = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(data)}
        if target:
            v['target'] = target
        views.append(v)
        return len(views) - 1

    accessors = []
    def accessor(data, component, count, kind, target, minimum=None, maximum=None):
        a = {'bufferView': view(data, target), 'componentType': component, 'count': count, 'type': kind}
        if minimum is not None:
            a['min'], a['max'] = minimum, maximum
        accessors.append(a)
        return len(accessors) - 1

    flat = lambda rows: b''.join(struct.pack('<' + 'f' * len(r), *r) for r in rows)
    lo = [min(p[i] for p in positions) for i in range(3)]
    hi = [max(p[i] for p in positions) for i in range(3)]
    pos = accessor(flat(positions), 5126, len(positions), 'VEC3', 34962, lo, hi)
    nor = accessor(flat(normals), 5126, len(normals), 'VEC3', 34962)
    tex = accessor(flat(uvs), 5126, len(uvs), 'VEC2', 34962)

    primitives = []
    for panel in range(3):
        idx = accessor(struct.pack('<6I', *indices[panel * 6:panel * 6 + 6]), 5125, 6, 'SCALAR', 34963)
        primitives.append({'attributes': {'POSITION': pos, 'NORMAL': nor, 'TEXCOORD_0': tex},
                           'indices': idx, 'material': panel})

    image = base64.b64encode(png(16, 16, checker)).decode()
    gltf = {
        'asset': {'version': '2.0', 'generator': 'basalt tests/data/make_materials_gltf.py'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}],
        'meshes': [{'primitives': primitives}],
        'materials': [
            {'name': 'opaque', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                         'metallicFactor': 0.0, 'roughnessFactor': 0.6}},
            {'name': 'masked', 'alphaMode': 'MASK', 'alphaCutoff': 0.5, 'doubleSided': True,
             'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0,
                                      'roughnessFactor': 0.5}},
            {'name': 'blended', 'alphaMode': 'BLEND',
             'pbrMetallicRoughness': {'baseColorFactor': [0.2, 0.6, 1.0, 0.4], 'metallicFactor': 0.0,
                                      'roughnessFactor': 0.3}},
        ],
        'textures': [{'source': 0, 'sampler': 0}],
        'samplers': [{'magFilter': 9728, 'minFilter': 9728}],
        'images': [{'uri': 'data:image/png;base64,' + image}],
        'accessors': accessors,
        'bufferViews': views,
    }
    data = blob.getvalue()
    gltf['buffers'] = [{'byteLength': len(data),
                        'uri': 'data:application/octet-stream;base64,' + base64.b64encode(data).decode()}]
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(gltf, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
