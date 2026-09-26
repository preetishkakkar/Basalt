#!/usr/bin/env python3
"""Fetches the larger benchmark scenes from Morgan McGuire's Computer Graphics Archive
(https://casual-effects.com/data) and converts their OBJ files to binary glTF in assets/:
conference.glb (125k triangles), hairball.glb (2.9M, one dense mesh) and bistro.glb (Bistro
exterior, 2.8M in 1306 objects, without its textures). Each OBJ object becomes a node with one
primitive per material; materials keep their diffuse colour and emission. Needs numpy.

Usage: python tools/fetch_scenes.py [--scenes conference,hairball,bistro] [--keep-downloads]
"""
import argparse
import json
import shutil
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "https://casual-effects.com/g3d/data10/research/model"
SCENES = {
    "conference": (f"{ARCHIVE}/conference/conference.zip", "conference.obj"),
    "hairball": (f"{ARCHIVE}/hairball/hairball.zip", "hairball.obj"),
    "bistro": (f"{ARCHIVE}/bistro/Exterior.zip", "exterior.obj"),
}


def read_materials(path):
    """By material name: the diffuse colour (Kd) and the emission (Ke)."""
    materials, name = {}, None
    if not path.exists():
        return materials
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "newmtl":
            name = " ".join(parts[1:])
            materials[name] = {"Kd": [0.8, 0.8, 0.8], "Ke": [0.0, 0.0, 0.0]}
        elif parts[0] in ("Kd", "Ke") and name is not None and len(parts) >= 4:
            values = [max(float(v), 0.0) for v in parts[1:4]]
            materials[name][parts[0]] = [min(v, 1.0) for v in values] if parts[0] == "Kd" else values
    return materials


def gltf_material(name, source):
    """The glTF material: the diffuse colour as base colour; an emission above 1 is a factor
    within 1 times KHR_materials_emissive_strength."""
    material = {"name": name or "default",
                "pbrMetallicRoughness": {"baseColorFactor": source["Kd"] + [1.0], "metallicFactor": 0.0,
                                         "roughnessFactor": 0.8}}
    strength = max(source["Ke"])
    if strength > 0.0:
        material["emissiveFactor"] = [v / max(strength, 1.0) for v in source["Ke"]]
        if strength > 1.0:
            material["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": strength}}
    return material


def read_obj(path):
    """Positions, texture coordinates, normals, and per (object, material) the triangles'
    corners as (position, texcoord, normal) indices, -1 for a missing one."""
    positions, texcoords, normals = [], [], []
    groups = {}
    obj, material = "default", None
    corners = groups.setdefault((obj, material), [])
    library = None

    def index(token, count):
        value = int(token)
        return value - 1 if value > 0 else count + value

    with open(path, "r", errors="replace") as source:
        for line in source:
            if not line or line[0] == "#":
                continue
            kind, _, rest = line.partition(" ")
            if kind == "v":
                positions.append(rest.split()[:3])
            elif kind == "vt":
                texcoords.append(rest.split()[:2])
            elif kind == "vn":
                normals.append(rest.split()[:3])
            elif kind == "f":
                face = []
                for token in rest.split():
                    fields = token.split("/")
                    p = index(fields[0], len(positions))
                    t = index(fields[1], len(texcoords)) if len(fields) > 1 and fields[1] else -1
                    n = index(fields[2], len(normals)) if len(fields) > 2 and fields[2] else -1
                    face.append((p, t, n))
                for k in range(1, len(face) - 1):  # a fan
                    corners.extend((face[0], face[k], face[k + 1]))
            elif kind in ("o", "g"):
                obj = rest.strip() or obj
                corners = groups.setdefault((obj, material), [])
            elif kind == "usemtl":
                material = rest.strip()
                corners = groups.setdefault((obj, material), [])
            elif kind == "mtllib":
                library = rest.strip()
    as_array = lambda values, width: np.array(values, dtype=np.float32).reshape(-1, width)  # noqa: E731
    return (as_array(positions, 3), as_array(texcoords, 2), as_array(normals, 3),
            {key: value for key, value in groups.items() if value}, library)


def write_glb(path, positions, texcoords, normals, groups, colours):
    blob = bytearray()
    views, accessors = [], []

    def add(array, target, component, kind, bounds=False):
        data = array.tobytes()
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        accessor = {"bufferView": len(views) - 1, "componentType": component, "count": int(array.shape[0]), "type": kind}
        if bounds:
            accessor["min"] = [float(v) for v in array.min(axis=0)]
            accessor["max"] = [float(v) for v in array.max(axis=0)]
        accessors.append(accessor)
        return len(accessors) - 1

    material_index = {}
    materials = []
    meshes, nodes = {}, []
    for (obj, material), corners in groups.items():
        if material not in material_index:
            material_index[material] = len(materials)
            materials.append(gltf_material(material, colours.get(material, {"Kd": [0.8, 0.8, 0.8], "Ke": [0.0, 0.0, 0.0]})))
        table = np.array(corners, dtype=np.int64)
        unique, inverse = np.unique(table, axis=0, return_inverse=True)
        attributes = {"POSITION": add(positions[unique[:, 0]], 34962, 5126, "VEC3", True)}
        if (unique[:, 2] >= 0).all() and len(normals):
            attributes["NORMAL"] = add(normals[unique[:, 2]], 34962, 5126, "VEC3")
        if (unique[:, 1] >= 0).all() and len(texcoords):
            uv = texcoords[unique[:, 1]].copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # OBJ's v runs up, glTF's down
            attributes["TEXCOORD_0"] = add(uv, 34962, 5126, "VEC2")
        indices = add(inverse.reshape(-1).astype(np.uint32), 34963, 5125, "SCALAR")
        meshes.setdefault(obj, []).append({"attributes": attributes, "indices": indices,
                                           "material": material_index[material]})
    mesh_list = []
    for obj, primitives in meshes.items():
        mesh_list.append({"name": obj, "primitives": primitives})
        nodes.append({"name": obj, "mesh": len(mesh_list) - 1})
    document = {"asset": {"version": "2.0", "generator": "Basalt tools/fetch_scenes.py"}, "scene": 0,
                "extensionsUsed": ["KHR_materials_emissive_strength"],
                "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": mesh_list,
                "materials": materials, "accessors": accessors, "bufferViews": views,
                "buffers": [{"byteLength": len(blob)}]}
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    with open(path, "wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob)))
        out.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        out.write(struct.pack("<II", len(blob), 0x004E4942) + blob)
    return sum(len(c) for c in groups.values()) // 3, len(nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--keep-downloads", action="store_true")
    args = parser.parse_args()
    downloads = ROOT / "assets" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name in args.scenes.split(","):
        url, obj_name = SCENES[name]
        target = ROOT / "assets" / f"{name}.glb"
        folder = downloads / name
        archive = downloads / f"{name}.zip"
        if not (folder / obj_name).exists():
            if not archive.exists():
                print(f"{name}: downloading {url}", flush=True)
                urllib.request.urlretrieve(url, archive)
            zipfile.ZipFile(archive).extractall(folder)
        print(f"{name}: converting {obj_name}", flush=True)
        positions, texcoords, normals, groups, library = read_obj(folder / obj_name)
        colours = read_materials(folder / library) if library else {}
        triangles, objects = write_glb(target, positions, texcoords, normals, groups, colours)
        print(f"{name}: {target} ({triangles} triangles, {objects} objects)", flush=True)
        if not args.keep_downloads:
            shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
