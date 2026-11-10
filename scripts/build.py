#!/usr/bin/env python3
"""Build the g20_pbr_description package from the pinned LinkerHand L25/G20 URDFs.

Input: a checkout of linker-bot/linkerhand-urdf at the commit in config/upstream.json.
Every source file is verified against its recorded SHA256 before use.

Output (``--output``, default ``build/g20_pbr_description``):

- ``urdf/g20_<side>.urdf``        visuals as GLB with embedded PBR materials
- ``urdf/g20_<side>_basic.urdf``  visuals as OBJ with URDF RGBA materials, for loaders without glTF
- ``urdf/ros/*.urdf``             the same two files with ``package://`` mesh paths
- ``mjcf/g20_<side>.xml``         MuJoCo model compiled from the basic URDF
- ``meshes/visual/<side>/``       per-material GLB and OBJ surfaces with smoothed normals
- ``meshes/collision/<side>/``    the upstream STL files, unmodified
- ``materials.json``, ``build_manifest.json``, LICENSE, NOTICE, README.md, package.xml, CMakeLists.txt

Kinematics, inertials, joint limits, mimic tags and collision geometry are copied
from upstream unchanged; only <visual> elements are replaced.
"""
import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'g20_pbr_description'
SIDES = ('left', 'right')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def srgb_to_linear(hex_color):
    values = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def load_config():
    materials = json.loads((ROOT / 'config/materials.json').read_text())['materials']
    for entry in materials.values():
        entry['linear'] = srgb_to_linear(entry['hex']).round(6).tolist()
    parts = json.loads((ROOT / 'config/parts.json').read_text())
    upstream = json.loads((ROOT / 'config/upstream.json').read_text())
    return materials, parts, upstream


def verify_upstream(upstream_dir, upstream):
    checks = [(upstream['license_file']['path'], upstream['license_file']['sha256'])]
    for hand in upstream['hands'].values():
        checks.append((hand['urdf'], hand['urdf_sha256']))
        checks += [(f"{hand['meshes']}/{name}", digest) for name, digest in hand['mesh_sha256'].items()]
    bad = [path for path, digest in checks if sha256(upstream_dir / path) != digest]
    if bad:
        raise SystemExit(f'upstream files differ from config/upstream.json: {bad}')
    return len(checks)


def classify(stem, area, extents, parts):
    if area < parts['sliver_area_mm2']:
        return parts['sliver_material']
    found = {rule['material'] for rule in parts['rules'][stem]
             if abs(area - rule['area_mm2']) <= parts['area_tolerance'] * rule['area_mm2']
             and np.allclose(sorted(extents), rule['extents_mm'], atol=parts['extent_tolerance_mm'])}
    if len(found) != 1:
        raise ValueError(f'{stem}: body {area:.1f} mm^2, extents {np.round(sorted(extents), 1).tolist()} '
                         f'matches {sorted(found) or "no rule"}')
    return found.pop()


def split(path, parts):
    """Partition the source surface by solid body; geometry is not modified."""
    mesh = trimesh.load(path, process=True)
    bodies = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    grouped = {}
    for body in bodies:
        sub = trimesh.Trimesh(mesh.vertices, mesh.faces[body], process=False)
        grouped.setdefault(classify(path.stem, sub.area * 1e6, sub.extents * 1e3, parts), []).append(body)
    result = {name: trimesh.Trimesh(mesh.vertices, mesh.faces[np.concatenate(group)], process=True)
              for name, group in sorted(grouped.items())}
    error = abs(sum(p.area for p in result.values()) - mesh.area) / mesh.area
    if error > 1e-9:
        raise ValueError(f'{path.name}: partition does not reproduce the source surface ({error:.2e})')
    return result, mesh.area


def smooth(mesh, crease_degrees):
    """Vertex normals averaged within crease_degrees; sharper edges keep split normals."""
    shaded = trimesh.graph.smooth_shade(mesh, angle=np.radians(crease_degrees), facet_minarea=None)
    if abs(shaded.area - mesh.area) > 1e-12 * max(mesh.area, 1):
        raise ValueError('smoothing changed the surface')
    return shaded


def write_glb(mesh, name, material, path):
    out = mesh.copy()
    out.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=material['linear'] + [1.0],
        metallicFactor=material['metallic'], roughnessFactor=material['roughness']))
    path.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(out), include_normals=True))


def write_obj(mesh, name, material, path):
    """Plain OBJ with per-vertex normals and an MTL using Kd plus the PBR extension keys."""
    normals = mesh.vertex_normals
    lines = [f'# {PACKAGE}: {path.stem}', f'mtllib {path.stem}.mtl', f'o {path.stem}']
    lines += [f'v {x:.9g} {y:.9g} {z:.9g}' for x, y, z in mesh.vertices]
    lines += [f'vn {x:.6g} {y:.6g} {z:.6g}' for x, y, z in normals]
    lines.append(f'usemtl {name}')
    lines += [f'f {a}//{a} {b}//{b} {c}//{c}' for a, b, c in mesh.faces + 1]
    path.write_text('\n'.join(lines) + '\n')
    r, g, b = srgb_to_linear(material['hex'])
    path.with_suffix('.mtl').write_text(
        f'newmtl {name}\nKa 0 0 0\nKd {r:.6f} {g:.6f} {b:.6f}\nKs 0.04 0.04 0.04\nd 1\nillum 2\n'
        f"Pm {material['metallic']}\nPr {material['roughness']}\n")


def rgba(material):
    """URDF/MJCF RGBA is display sRGB in common viewers."""
    return [round(int(material['hex'][i:i + 2], 16) / 255, 4) for i in (1, 3, 5)] + [1.0]


def build_urdf(source, side, visuals, materials, mesh_root, basic):
    tree = copy.deepcopy(source)
    robot = tree.getroot()
    robot.set('name', f'g20_pbr_{side}')
    if basic:
        for name in sorted(materials):
            mat = ET.SubElement(robot, 'material', name=f'g20_{name}')
            ET.SubElement(mat, 'color', rgba=' '.join(map(str, rgba(materials[name]))))
        robot[:] = [e for e in robot if e.tag == 'material'] + [e for e in robot if e.tag != 'material']
    for link in robot.findall('link'):
        originals = link.findall('visual')
        for visual in originals:
            link.remove(visual)
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is not None:
                mesh.set('filename', f'{mesh_root}/collision/{side}/{Path(mesh.get("filename")).name}')
        if not originals:
            continue
        if len(originals) != 1:
            raise ValueError(f'{link.get("name")}: expected one upstream visual')
        origin = originals[0].find('origin')
        insert_at = list(link).index(link.find('collision')) if link.find('collision') is not None else len(link)
        for material_name in visuals[link.get('name')]:
            visual = ET.Element('visual', name=f'{link.get("name")}_{material_name}')
            if origin is not None:
                visual.append(copy.deepcopy(origin))
            geometry = ET.SubElement(visual, 'geometry')
            suffix = 'obj' if basic else 'glb'
            ET.SubElement(geometry, 'mesh', filename=f'{mesh_root}/visual/{side}/{link.get("name")}_{material_name}.{suffix}')
            if basic:
                ET.SubElement(visual, 'material', name=f'g20_{material_name}')
            link.insert(insert_at, visual)
            insert_at += 1
    ET.indent(tree, space='  ')
    return tree


def write_xml(tree, path, comment):
    body = ET.tostring(tree.getroot(), encoding='unicode')
    path.write_text(f'<?xml version="1.0" encoding="utf-8"?>\n<!-- {comment} -->\n{body}\n')


def build_mjcf(basic_urdf, side, out_dir):
    """Compile the basic URDF with MuJoCo, then keep visual and collision geoms apart."""
    import mujoco
    tree = ET.parse(basic_urdf)
    robot = tree.getroot()
    mujoco_tag = ET.SubElement(robot, 'mujoco')
    # Some upstream inertia tensors violate A + B >= C, which MuJoCo rejects.
    # balanceinertia adjusts only those; the URDFs keep the upstream values.
    ET.SubElement(mujoco_tag, 'compiler', meshdir='../meshes', balanceinertia='true',
                  discardvisual='false', strippath='false', fusestatic='false')
    for mesh in robot.iter('mesh'):
        mesh.set('filename', mesh.get('filename').removeprefix('../meshes/'))
    scratch = out_dir / f'.{side}_mujoco_input.urdf'
    tree.write(scratch)
    model = mujoco.MjModel.from_xml_path(str(scratch))
    target = out_dir / f'g20_{side}.xml'
    mujoco.mj_saveLastXML(str(target), model)
    scratch.unlink()
    mjcf = ET.parse(target)
    root = mjcf.getroot()
    root.set('model', f'g20_pbr_{side}')
    # Collision meshes keep the URDF behaviour; visual meshes never collide.
    for geom in root.iter('geom'):
        mesh = geom.get('mesh', '')
        if mesh and not mesh.endswith(('_shell_silver', '_rubber_black', '_steel_link', '_steel_dark', '_wrist_aluminium')):
            # MuJoCo's URDF import leaves collision geoms unnamed and gives them a visual colour.
            geom.set('name', f'{mesh}_collision')
            geom.set('group', '3')
            geom.attrib.pop('rgba', None)
        elif mesh:
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('group', '1')
            geom.set('density', '0')
    # URDF <mimic> is not imported by MuJoCo; express it as joint equality constraints.
    source = ET.parse(basic_urdf).getroot()
    mimics = [(j.get('name'), j.find('mimic')) for j in source.findall('joint') if j.find('mimic') is not None]
    if mimics:
        equality = root.find('equality') or ET.SubElement(root, 'equality')
        for name, mimic in mimics:
            ET.SubElement(equality, 'joint', joint1=name, joint2=mimic.get('joint'),
                          polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0")
    ET.indent(mjcf, space='  ')
    write_xml(mjcf, target, f'{PACKAGE} {side} hand, compiled from g20_{side}_basic.urdf by MuJoCo; see README')
    mujoco.MjModel.from_xml_path(str(target))
    return target, invalid_inertia(source)


def invalid_inertia(urdf_root):
    """Links whose upstream principal moments fail the triangle inequality."""
    bad = []
    for link in urdf_root.findall('link'):
        inertia = link.find('inertial/inertia')
        if inertia is None:
            continue
        a = {k: float(inertia.get(k)) for k in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')}
        tensor = np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]])
        moments = np.sort(np.linalg.eigvalsh(tensor))
        if moments[0] + moments[1] < moments[2] * (1 - 1e-9):
            bad.append(link.get('name'))
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--upstream', type=Path, required=True, help='linker-bot/linkerhand-urdf checkout')
    parser.add_argument('--output', type=Path, default=ROOT / 'build' / PACKAGE)
    parser.add_argument('--crease-degrees', type=float, default=30.)
    args = parser.parse_args()
    materials, parts, upstream = load_config()
    verified = verify_upstream(args.upstream, upstream)
    if args.output.exists():
        shutil.rmtree(args.output)
    out = args.output
    for sub in ('urdf/ros', 'mjcf', *(f'meshes/{k}/{s}' for k in ('visual', 'collision') for s in SIDES)):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = dict(package=PACKAGE, upstream=dict(repository=upstream['repository'], commit=upstream['commit']),
                    crease_degrees=args.crease_degrees, upstream_files_verified=verified, hands={})
    for side in SIDES:
        hand = upstream['hands'][side]
        source = ET.parse(args.upstream / hand['urdf'])
        visuals, records = {}, []
        for link in source.getroot().findall('link'):
            visual = link.find('visual/geometry/mesh')
            if visual is None:
                continue
            stl = args.upstream / hand['meshes'] / Path(visual.get('filename')).name
            pieces, area = split(stl, parts)
            visuals[link.get('name')] = list(pieces)
            shutil.copyfile(stl, out / f'meshes/collision/{side}/{stl.name}')
            for name, piece in pieces.items():
                shaded = smooth(piece, args.crease_degrees)
                stem = out / f'meshes/visual/{side}/{link.get("name")}_{name}'
                write_glb(shaded, f'g20_{name}', materials[name], stem.with_suffix('.glb'))
                write_obj(shaded, f'g20_{name}', materials[name], stem.with_suffix('.obj'))
                records.append(dict(link=link.get('name'), material=name, area_mm2=round(piece.area * 1e6, 3),
                                    faces=int(len(piece.faces)), source=stl.name, source_sha256=sha256(stl)))
        notice = (f'{PACKAGE}: LinkerHand G20 ({side}) with per-material visuals. Kinematics, inertials and '
                  f'collision from {upstream["repository"]} @ {upstream["commit"][:7]} ({hand["urdf"]}), '
                  f'{upstream["copyright"]}, Apache-2.0. Modified: visual meshes split by material, normals '
                  f'smoothed, PBR materials added.')
        for basic, suffix in ((False, ''), (True, '_basic')):
            for root, folder in (('../meshes', 'urdf'), (f'package://{PACKAGE}/meshes', 'urdf/ros')):
                tree = build_urdf(source, side, visuals, materials, root, basic)
                write_xml(tree, out / folder / f'g20_{side}{suffix}.urdf', notice)
        _, balanced = build_mjcf(out / f'urdf/g20_{side}_basic.urdf', side, out / 'mjcf')
        manifest['hands'][side] = dict(source_urdf=hand['urdf'], source_urdf_sha256=hand['urdf_sha256'],
                                       mjcf_balanced_inertia=balanced, parts=records)
    (out / 'materials.json').write_text(json.dumps(
        {name: dict(hex=m['hex'], linear_rgb=m['linear'], metallic=m['metallic'], roughness=m['roughness'],
                    parts=m['parts']) for name, m in materials.items()}, indent=1) + '\n')
    for name in ('LICENSE', 'NOTICE', 'README.md'):
        shutil.copyfile(ROOT / name, out / name)
    for name in ('package.xml', 'CMakeLists.txt'):
        shutil.copyfile(ROOT / 'ros' / name, out / name)
    files = sorted(p for p in out.rglob('*') if p.is_file())
    manifest['files'] = {str(p.relative_to(out)): sha256(p) for p in files}
    (out / 'build_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(dict(output=str(out), files=len(files) + 1,
                          visual_parts=sum(len(h['parts']) for h in manifest['hands'].values()))))


if __name__ == '__main__':
    main()
