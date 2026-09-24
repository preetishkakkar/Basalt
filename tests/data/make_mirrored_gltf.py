"""Writes mirrored.gltf: glTF 2.0 sidedness under negative-determinant node transforms.

Three one-sided, red-emissive unit quads in a row, all seen from +z by the default camera:
- A (left): node scale (-1, 1, 1); its object-space front faces +z, so glTF shows it;
- B (middle): node scale (-1, 1, 1); its object-space front faces -z, so glTF culls it,
  although in world space it winds counter-clockwise towards the camera;
- C (right): no mirror, front faces +z: visible (control).
glTF 2.0 keeps the object-space front when the determinant is negative (its world winding is
clockwise); a renderer using world-space winding would show B and hide A.

usage: python make_mirrored_gltf.py [output.gltf]
"""
import base64
import json
import struct
import sys


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'mirrored.gltf'
    positions = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 4
    front = [0, 1, 2, 0, 2, 3]   # counter-clockwise seen from +z
    back = [0, 2, 1, 0, 3, 2]    # counter-clockwise seen from -z
    blob = b''.join(struct.pack('<3f', *p) for p in positions)
    blob += b''.join(struct.pack('<3f', *n) for n in normals)
    blob += struct.pack('<6I', *front) + struct.pack('<6I', *back)
    views = [
        {'buffer': 0, 'byteOffset': 0, 'byteLength': 48, 'target': 34962},
        {'buffer': 0, 'byteOffset': 48, 'byteLength': 48, 'target': 34962},
        {'buffer': 0, 'byteOffset': 96, 'byteLength': 24, 'target': 34963},
        {'buffer': 0, 'byteOffset': 120, 'byteLength': 24, 'target': 34963},
    ]
    accessors = [
        {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
         'min': [-0.5, -0.5, 0.0], 'max': [0.5, 0.5, 0.0]},
        {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC3'},
        {'bufferView': 2, 'componentType': 5125, 'count': 6, 'type': 'SCALAR'},
        {'bufferView': 3, 'componentType': 5125, 'count': 6, 'type': 'SCALAR'},
    ]
    attributes = {'POSITION': 0, 'NORMAL': 1}
    gltf = {
        'asset': {'version': '2.0', 'generator': 'basalt tests/data/make_mirrored_gltf.py'},
        'scene': 0,
        'scenes': [{'nodes': [0, 1, 2]}],
        'nodes': [
            {'name': 'A mirrored, front towards +z', 'mesh': 0, 'translation': [-1.6, 0.0, 0.0], 'scale': [-1.0, 1.0, 1.0]},
            {'name': 'B mirrored, front towards -z', 'mesh': 1, 'translation': [0.0, 0.0, 0.0], 'scale': [-1.0, 1.0, 1.0]},
            {'name': 'C plain, front towards +z', 'mesh': 0, 'translation': [1.6, 0.0, 0.0]},
        ],
        'meshes': [{'primitives': [{'attributes': attributes, 'indices': 2, 'material': 0}]},
                   {'primitives': [{'attributes': attributes, 'indices': 3, 'material': 0}]}],
        'materials': [{'name': 'one-sided red emitter', 'doubleSided': False, 'emissiveFactor': [1.0, 0.0, 0.0],
                       'extensions': {'KHR_materials_emissive_strength': {'emissiveStrength': 4.0}},
                       'pbrMetallicRoughness': {'baseColorFactor': [0.0, 0.0, 0.0, 1.0], 'metallicFactor': 0.0,
                                                'roughnessFactor': 1.0}}],
        'extensionsUsed': ['KHR_materials_emissive_strength'],
        'accessors': accessors,
        'bufferViews': views,
        'buffers': [{'byteLength': len(blob), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(blob).decode()}],
    }
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(gltf, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
