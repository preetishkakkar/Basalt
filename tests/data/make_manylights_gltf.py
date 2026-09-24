"""Writes manylights.gltf, the ReSTIR DI many-light fixture: 64 punctual lights
(KHR_lights_punctual; every fourth a spot) of varied colour and intensity over a floor with
eight pillars that shadow some of them from some receivers, and two one-sided emissive
panels, so the reservoirs choose among many lights of every kind the tracer samples. It also
writes manylights_dim.hdr, a dim uniform environment to render it with (a 4x2 Radiance file
of radiance 0.02), so the punctual and emissive lights dominate while the environment stays
one of the light kinds.

usage: python make_manylights_gltf.py [output.gltf]
"""
import base64
import io
import json
import math
import struct
import sys


def hashed(i, salt):
    h = (i * 2654435761 + salt * 40503) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return (h & 0xFFFF) / 65535.0


class Mesh:
    def __init__(self):
        self.positions, self.normals, self.indices = [], [], []

    def quad(self, corner, u, v):
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        length = math.sqrt(sum(c * c for c in n))
        n = [c / length for c in n]
        base = len(self.positions)
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            self.positions.append(tuple(corner[i] + a * u[i] + b * v[i] for i in range(3)))
            self.normals.append(tuple(n))
        self.indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    def box(self, centre, half):
        x, y, z = centre
        hx, hy, hz = half
        self.quad((x - hx, y - hy, z + hz), (2 * hx, 0, 0), (0, 2 * hy, 0))   # +z
        self.quad((x + hx, y - hy, z - hz), (-2 * hx, 0, 0), (0, 2 * hy, 0))  # -z
        self.quad((x + hx, y - hy, z + hz), (0, 0, -2 * hz), (0, 2 * hy, 0))  # +x
        self.quad((x - hx, y - hy, z - hz), (0, 0, 2 * hz), (0, 2 * hy, 0))   # -x
        self.quad((x - hx, y + hy, z + hz), (2 * hx, 0, 0), (0, 0, -2 * hz))  # +y


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'manylights.gltf'
    floor, pillars, panels = Mesh(), Mesh(), Mesh()
    floor.quad((-6.0, 0.0, 6.0), (12.0, 0.0, 0.0), (0.0, 0.0, -12.0))
    for i in range(8):
        a = 2.0 * math.pi * i / 8.0
        pillars.box((3.0 * math.cos(a), 1.0, 3.0 * math.sin(a)), (0.25, 1.0, 0.25))
    # Two panels facing the centre (one-sided emitters).
    panels.quad((-4.5, 0.2, -1.0), (0.0, 0.0, 2.0), (0.0, 1.2, 0.0))
    panels.quad((4.5, 0.2, 1.0), (0.0, 0.0, -2.0), (0.0, 1.2, 0.0))

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
                               'NORMAL': accessor(flat(mesh.normals), 5126, len(mesh.normals), 'VEC3', 34962)},
                'indices': accessor(struct.pack('<%dI' % len(mesh.indices), *mesh.indices), 5125, len(mesh.indices),
                                    'SCALAR', 34963),
                'material': material}

    lights, nodes = [], [{'mesh': 0}]
    for i in range(64):
        colour = [0.4 + 0.6 * hashed(i, 1), 0.4 + 0.6 * hashed(i, 2), 0.4 + 0.6 * hashed(i, 3)]
        light = {'type': 'point', 'color': colour, 'intensity': 5.0 + 60.0 * hashed(i, 4) ** 2}
        position = [-5.0 + 10.0 * hashed(i, 5), 0.3 + 2.2 * hashed(i, 6), -5.0 + 10.0 * hashed(i, 7)]
        node = {'light_index': i, 'translation': position}
        if i % 4 == 3:
            light['type'] = 'spot'
            light['spot'] = {'innerConeAngle': 0.3, 'outerConeAngle': 0.6}
            node['rotation'] = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]  # aim down (-y)
        lights.append(light)
        nodes.append({'translation': node['translation'], **({'rotation': node['rotation']} if 'rotation' in node else {}),
                      'extensions': {'KHR_lights_punctual': {'light': i}}})

    gltf = {
        'asset': {'version': '2.0', 'generator': 'basalt tests/data/make_manylights_gltf.py'},
        'extensionsUsed': ['KHR_lights_punctual', 'KHR_materials_emissive_strength'],
        'extensions': {'KHR_lights_punctual': {'lights': lights}},
        'scene': 0,
        'scenes': [{'nodes': list(range(len(nodes)))}],
        'nodes': nodes,
        'meshes': [{'primitives': [primitive(floor, 0), primitive(pillars, 1), primitive(panels, 2)]}],
        'materials': [
            {'name': 'floor', 'pbrMetallicRoughness': {'baseColorFactor': [0.7, 0.7, 0.7, 1.0], 'metallicFactor': 0.0,
                                                       'roughnessFactor': 0.6}},
            {'name': 'pillars', 'pbrMetallicRoughness': {'baseColorFactor': [0.8, 0.5, 0.3, 1.0], 'metallicFactor': 0.0,
                                                         'roughnessFactor': 0.4}},
            {'name': 'panels', 'emissiveFactor': [1.0, 0.9, 0.7],
             'extensions': {'KHR_materials_emissive_strength': {'emissiveStrength': 3.0}},
             'pbrMetallicRoughness': {'baseColorFactor': [0.0, 0.0, 0.0, 1.0], 'metallicFactor': 0.0}},
        ],
        'accessors': accessors,
        'bufferViews': views,
    }
    data = blob.getvalue()
    gltf['buffers'] = [{'byteLength': len(data),
                        'uri': 'data:application/octet-stream;base64,' + base64.b64encode(data).decode()}]
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(gltf, f, indent=1)
    print('wrote', out)
    hdr = out.replace('.gltf', '_dim.hdr')
    mantissa, exponent = math.frexp(0.02)
    rgbe = bytes([int(mantissa * 256.0)] * 3 + [exponent + 128])
    with open(hdr, 'wb') as f:
        f.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n' + rgbe * 8)
    print('wrote', hdr)


if __name__ == '__main__':
    main()
