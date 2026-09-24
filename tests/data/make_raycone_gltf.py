"""Writes raycone.gltf, the ray-cone filtering fixture: textures that the camera
sees minified, so level zero and ray-cone filtering differ.

- a 40 m ground whose 64x64 texture of per-texel colours repeats once per metre, seen at a
  grazing angle out to its far edge,
- four double-sided alpha-masked foliage cards (2 m, cutoff 0.5) at 2 to 20 m, whose 64x64
  one-texel pattern is 60% opaque and repeats four times across a card,
all self-contained (buffers and textures embedded as base64).

usage: python make_raycone_gltf.py [output.gltf]
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


def speckle(x, y):
    h = (x * 73856093 ^ y * 19349663) & 0xFFFFFF
    return (40 + (h & 0xFF) * 3 // 4, 40 + ((h >> 8) & 0xFF) * 3 // 4, 40 + ((h >> 16) & 0xFF) * 3 // 4, 255)


def leaves(x, y):
    return (70, 150, 50, 255 if (x * 7 + y * 13) % 10 < 6 else 0)


class Mesh:
    def __init__(self):
        self.positions, self.normals, self.uvs, self.indices = [], [], [], []

    def quad(self, corner, u, v, normal, uv_scale):
        base = len(self.positions)
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            self.positions.append(tuple(corner[i] + a * u[i] + b * v[i] for i in range(3)))
            self.normals.append(normal)
            self.uvs.append((a * uv_scale, (1 - b) * uv_scale))
        self.indices += [base, base + 1, base + 2, base, base + 2, base + 3]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'raycone.gltf'
    ground, cards = Mesh(), Mesh()
    ground.quad((-20.0, 0.0, 20.0), (40.0, 0.0, 0.0), (0.0, 0.0, -40.0), (0.0, 1.0, 0.0), 40.0)
    for x, z in ((-1.5, -2.0), (1.0, -6.0), (-3.0, -12.0), (4.0, -20.0)):
        cards.quad((x - 1.0, 0.0, z), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0), 4.0)

    blob = io.BytesIO()
    views, accessors = [], []
    def view(data, target):
        while blob.tell() % 4:
            blob.write(b'\x00')
        views.append({'buffer': 0, 'byteOffset': blob.tell(), 'byteLength': len(data), 'target': target})
        blob.write(data)
        return len(views) - 1
    def accessor(data, component, count, kind, target, minimum=None, maximum=None):
        a = {'bufferView': view(data, target), 'componentType': component, 'count': count, 'type': kind}
        if minimum is not None:
            a['min'], a['max'] = minimum, maximum
        accessors.append(a)
        return len(accessors) - 1
    flat = lambda rows: b''.join(struct.pack('<' + 'f' * len(r), *r) for r in rows)

    def primitive(mesh, material):
        lo = [min(p[i] for p in mesh.positions) for i in range(3)]
        hi = [max(p[i] for p in mesh.positions) for i in range(3)]
        return {'attributes': {'POSITION': accessor(flat(mesh.positions), 5126, len(mesh.positions), 'VEC3', 34962, lo, hi),
                               'NORMAL': accessor(flat(mesh.normals), 5126, len(mesh.normals), 'VEC3', 34962),
                               'TEXCOORD_0': accessor(flat(mesh.uvs), 5126, len(mesh.uvs), 'VEC2', 34962)},
                'indices': accessor(struct.pack('<%dI' % len(mesh.indices), *mesh.indices), 5125, len(mesh.indices),
                                    'SCALAR', 34963),
                'material': material}

    primitives = [primitive(ground, 0), primitive(cards, 1)]
    images = [base64.b64encode(png(64, 64, pattern)).decode() for pattern in (speckle, leaves)]
    gltf = {
        'asset': {'version': '2.0', 'generator': 'basalt tests/data/make_raycone_gltf.py'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}],
        'meshes': [{'primitives': primitives}],
        'materials': [
            {'name': 'speckled ground', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                                  'metallicFactor': 0.0, 'roughnessFactor': 0.7}},
            {'name': 'foliage', 'alphaMode': 'MASK', 'alphaCutoff': 0.5, 'doubleSided': True,
             'pbrMetallicRoughness': {'baseColorTexture': {'index': 1}, 'metallicFactor': 0.0,
                                      'roughnessFactor': 0.8}},
        ],
        'textures': [{'source': 0, 'sampler': 0}, {'source': 1, 'sampler': 0}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
        'images': [{'uri': 'data:image/png;base64,' + image} for image in images],
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
