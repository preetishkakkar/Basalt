"""Writes v7_materials.gltf: the V7 transmission and clearcoat fixture .

- a checker back wall and a grey floor, so refraction and reflection have something to show,
- a closed glass sphere (KHR_materials_transmission 1, KHR_materials_ior 1.5,
  KHR_materials_volume thickness 1),
- a rough closed dielectric sphere (IOR 1.33, roughness 0.4),
- a tinted thin-walled panel (transmission 0.8, no volume), double-sided,
- a red sphere under a smooth clearcoat,
all self-contained (buffers and the checker texture embedded as base64).

usage: python make_v7_gltf.py [output.gltf]
"""
import base64
import io
import json
import math
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
    return (220, 220, 210, 255) if ((x // 4) + (y // 4)) % 2 == 0 else (40, 60, 110, 255)


class Mesh:
    def __init__(self):
        self.positions, self.normals, self.uvs, self.indices = [], [], [], []

    def quad(self, corner, u, v, normal, uv_scale=1.0):
        """Counter-clockwise seen from `normal`: corner, corner + u, corner + u + v, corner + v."""
        base = len(self.positions)
        for (a, b) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            self.positions.append(tuple(corner[i] + u[i] * a + v[i] * b for i in range(3)))
            self.normals.append(normal)
            self.uvs.append((a * uv_scale, (1 - b) * uv_scale))
        self.indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    def sphere(self, centre, radius, rings=24, segments=48):
        """Outward normals, counter-clockwise from outside."""
        base = len(self.positions)
        for r in range(rings + 1):
            theta = math.pi * r / rings
            for s in range(segments + 1):
                phi = 2.0 * math.pi * s / segments
                n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
                self.positions.append(tuple(centre[i] + radius * n[i] for i in range(3)))
                self.normals.append(n)
                self.uvs.append((s / segments, r / rings))
        for r in range(rings):
            for s in range(segments):
                a = base + r * (segments + 1) + s
                b = a + segments + 1
                if r != 0:
                    self.indices += [a, a + 1, b]
                if r != rings - 1:
                    self.indices += [a + 1, b + 1, b]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'v7_materials.gltf'
    parts = []  # (mesh, material)

    wall, floor = Mesh(), Mesh()
    wall.quad((-4.0, 0.0, -2.0), (8.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 1.0), uv_scale=4.0)
    floor.quad((-4.0, 0.0, 3.0), (8.0, 0.0, 0.0), (0.0, 0.0, -5.0), (0.0, 1.0, 0.0))
    parts += [(wall, 0), (floor, 1)]
    glass, rough_glass, coated = Mesh(), Mesh(), Mesh()
    glass.sphere((-1.9, 0.8, 0.0), 0.8)
    rough_glass.sphere((0.0, 0.7, 0.4), 0.7)
    coated.sphere((1.9, 0.8, 0.0), 0.8)
    parts += [(glass, 2), (rough_glass, 3), (coated, 5)]
    panel = Mesh()
    panel.quad((-0.9, 0.0, 1.6), (1.2, 0.0, 0.0), (0.0, 1.4, 0.0), (0.0, 0.0, 1.0))
    parts.append((panel, 4))

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
    primitives = []
    for mesh, material in parts:
        lo = [min(p[i] for p in mesh.positions) for i in range(3)]
        hi = [max(p[i] for p in mesh.positions) for i in range(3)]
        attributes = {
            'POSITION': accessor(flat(mesh.positions), 5126, len(mesh.positions), 'VEC3', 34962, lo, hi),
            'NORMAL': accessor(flat(mesh.normals), 5126, len(mesh.normals), 'VEC3', 34962),
            'TEXCOORD_0': accessor(flat(mesh.uvs), 5126, len(mesh.uvs), 'VEC2', 34962),
        }
        index = accessor(struct.pack('<%dI' % len(mesh.indices), *mesh.indices), 5125, len(mesh.indices),
                         'SCALAR', 34963)
        primitives.append({'attributes': attributes, 'indices': index, 'material': material})

    def pbr(base, metallic, roughness, **extra):
        p = {'baseColorFactor': base + [1.0], 'metallicFactor': metallic, 'roughnessFactor': roughness}
        p.update(extra)
        return p

    materials = [
        {'name': 'checker wall', 'pbrMetallicRoughness': pbr([1.0, 1.0, 1.0], 0.0, 0.8,
                                                              baseColorTexture={'index': 0})},
        {'name': 'floor', 'pbrMetallicRoughness': pbr([0.6, 0.6, 0.6], 0.0, 0.7)},
        {'name': 'glass', 'pbrMetallicRoughness': pbr([1.0, 1.0, 1.0], 0.0, 0.05),
         'extensions': {'KHR_materials_transmission': {'transmissionFactor': 1.0},
                        'KHR_materials_ior': {'ior': 1.5},
                        'KHR_materials_volume': {'thicknessFactor': 1.0}}},
        {'name': 'rough water', 'pbrMetallicRoughness': pbr([0.8, 0.95, 1.0], 0.0, 0.4),
         'extensions': {'KHR_materials_transmission': {'transmissionFactor': 1.0},
                        'KHR_materials_ior': {'ior': 1.33},
                        'KHR_materials_volume': {'thicknessFactor': 1.0}}},
        {'name': 'thin tinted sheet', 'doubleSided': True, 'pbrMetallicRoughness': pbr([1.0, 0.7, 0.4], 0.0, 0.3),
         'extensions': {'KHR_materials_transmission': {'transmissionFactor': 0.8}}},
        {'name': 'clearcoated red', 'pbrMetallicRoughness': pbr([0.8, 0.08, 0.05], 0.0, 0.6),
         'extensions': {'KHR_materials_clearcoat': {'clearcoatFactor': 1.0, 'clearcoatRoughnessFactor': 0.05}}},
    ]
    image = base64.b64encode(png(16, 16, checker)).decode()
    gltf = {
        'asset': {'version': '2.0', 'generator': 'basalt tests/data/make_v7_gltf.py'},
        'extensionsUsed': ['KHR_materials_transmission', 'KHR_materials_ior', 'KHR_materials_volume',
                           'KHR_materials_clearcoat'],
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}],
        'meshes': [{'primitives': primitives}],
        'materials': materials,
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
