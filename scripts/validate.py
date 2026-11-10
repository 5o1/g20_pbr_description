#!/usr/bin/env python3
"""Validate a built package against upstream and render it in SAPIEN and MuJoCo.

Checks, per hand:
1. Joints, inertials and collision geometry equal upstream (paths aside) in every URDF.
2. Visual parts reproduce each upstream visual surface (area), materials known.
3. yourdfpy forward kinematics of every link equals upstream at random joint values.
4. pybullet loads the basic URDF with the upstream joint count.
5. MuJoCo body positions equal yourdfpy at the same joint values (mimic joints
   set from their equality constraints), and the visual geom count matches.
6. SAPIEN loads the GLB URDF, and the rendered materials carry the palette values.
Renders go to --media; the summary JSON is printed and written to --media.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIDES = ('left', 'right')
VIEWS = {'palm': (0.42, 0.0), 'back': (-0.42, 0.0), 'radial': (0.0, 0.42), 'ulnar': (0.0, -0.42)}


def norm(elem):
    return None if elem is None else ET.tostring(elem, encoding='unicode').split()


def structure(urdf):
    root = ET.parse(urdf).getroot()
    joints = {j.get('name'): norm(j) for j in root.findall('joint')}
    inertials = {l.get('name'): norm(l.find('inertial')) for l in root.findall('link')}
    collisions = {}
    for link in root.findall('link'):
        items = []
        for c in link.findall('collision'):
            c = ET.fromstring(ET.tostring(c))
            mesh = c.find('geometry/mesh')
            if mesh is not None:
                mesh.set('filename', Path(mesh.get('filename')).name)
            items.append(norm(c))
        collisions[link.get('name')] = items
    return joints, inertials, collisions


def random_q(robot, rng):
    q = {}
    for name, joint in robot.joint_map.items():
        if joint.type in ('revolute', 'prismatic') and joint.mimic is None:
            lo, hi = joint.limit.lower, joint.limit.upper
            q[name] = rng.uniform(lo, hi)
    return q


def fk(robot, q):
    robot.update_cfg(q)
    return {link: robot.get_transform(link) for link in robot.link_map}


def look_at(position, target, up):
    import sapien
    from transforms3d.quaternions import mat2quat
    forward = np.asarray(target) - position
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward); left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    return sapien.Pose(position, mat2quat(np.column_stack([forward, left, up])))


def sapien_render(urdf, media, side, size, palette):
    import sapien
    from PIL import Image
    scene = sapien.Scene()
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0.4, 0.5, -1.0], [1.6, 1.6, 1.6])
    scene.add_directional_light([-0.6, -0.4, -0.3], [0.7, 0.7, 0.75])
    scene.add_point_light([0.5, 0.5, 0.8], [1.5, 1.5, 1.5])
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(urdf))
    # set_qpos updates link poses directly; a physics step would let overlapping
    # collision bodies of neighbouring links push each other apart.
    robot.set_qpos(np.zeros(robot.dof))
    seen = []
    for link in robot.get_links():
        body = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        for shape in (body.render_shapes if body else []):
            for part in shape.parts:
                m = part.material
                seen.append((np.asarray(m.base_color[:3]), m.metallic, m.roughness))
    # glTF export stores 8-bit colour factors, so compare within one 8-bit step.
    def found(entry):
        return any(np.allclose(c, entry['linear_rgb'], atol=1 / 255) and abs(mt - entry['metallic']) < 1e-3
                   and abs(r - entry['roughness']) < 1e-3 for c, mt, r in seen)
    camera = scene.add_camera('cam', size, size, np.radians(36), 0.01, 10.)
    images = []
    for name, (dx, dy) in VIEWS.items():
        camera.entity.set_pose(look_at(np.array([dx, dy, 0.12]), np.array([0., 0., 0.12]), np.array([0., 0., 1.])))
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture('Color')
        rgb = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        alpha = rgba[..., 3:4] if rgba.shape[-1] == 4 else np.ones_like(rgba[..., :1])
        background = np.array([58, 64, 72], np.uint8)
        composed = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        images.append(Image.fromarray(composed))
    sheet = Image.new('RGB', (size * len(images), size))
    for i, image in enumerate(images):
        sheet.paste(image, (i * size, 0))
    sheet.save(media / f'sapien_{side}.png')
    return dict(palette_in_render=all(found(m) for m in palette.values()),
                distinct_materials=len({(tuple(np.round(c, 3)), mt, r) for c, mt, r in seen}))


def mujoco_check(mjcf, robot, q_samples, media, side, size, n_visual):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    visual = sum(1 for g in range(model.ngeom) if model.geom_group[g] == 1)
    errors = []
    for q in q_samples:
        full = dict(q)
        for name, joint in robot.joint_map.items():
            if joint.mimic is not None:
                full[name] = joint.mimic.multiplier * full[joint.mimic.joint] + joint.mimic.offset
        for name, value in full.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        poses = fk(robot, full)
        for b in range(1, model.nbody):
            name = model.body(b).name
            if name in poses:
                errors.append(np.linalg.norm(data.xpos[b] - poses[name][:3, 3]))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    renderer = mujoco.Renderer(model, size, size)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 0, 0, 0, 0]
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    from PIL import Image
    images = []
    for azimuth in (0, 180):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.12]
        camera.distance, camera.azimuth, camera.elevation = 0.42, azimuth, -5
        renderer.update_scene(data, camera, scene_option=option)
        images.append(Image.fromarray(renderer.render()))
    sheet = Image.new('RGB', (size * 2, size))
    for i, image in enumerate(images):
        sheet.paste(image, (i * size, 0))
    sheet.save(media / f'mujoco_{side}.png')
    return dict(nbody=model.nbody, njnt=model.njnt, neq=model.neq, visual_geoms=visual,
                visual_geoms_expected=n_visual, max_body_position_error_m=float(max(errors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    import pybullet
    import trimesh
    import yourdfpy
    args.media.mkdir(parents=True, exist_ok=True)
    upstream = json.loads((Path(__file__).resolve().parents[1] / 'config/upstream.json').read_text())
    manifest = json.loads((args.package / 'build_manifest.json').read_text())
    palette = json.loads((args.package / 'materials.json').read_text())
    rng = np.random.default_rng(0)
    summary = {}
    for side in SIDES:
        source = args.upstream / upstream['hands'][side]['urdf']
        result = {}
        reference = structure(source)
        variants = {v: args.package / 'urdf' / f'g20_{side}{v}.urdf' for v in ('', '_basic')}
        variants.update({'ros' + v: args.package / 'urdf/ros' / f'g20_{side}{v}.urdf' for v in ('', '_basic')})
        result['structure_equal'] = {k: structure(p) == reference for k, p in variants.items()}
        parts = manifest['hands'][side]['parts']
        areas = {}
        for part in parts:
            areas[part['source']] = areas.get(part['source'], 0) + part['area_mm2']
        meshes = args.upstream / upstream['hands'][side]['meshes']
        result['max_visual_area_error'] = max(abs(a - trimesh.load(meshes / s).area * 1e6) / a for s, a in areas.items())
        result['materials_known'] = all(p['material'] in palette for p in parts)
        upstream_robot = yourdfpy.URDF.load(str(source), load_meshes=False)
        samples = [random_q(upstream_robot, rng) for _ in range(args.samples)]
        fk_error = 0.
        for name, path in variants.items():
            if name.startswith('ros'):
                continue
            built = yourdfpy.URDF.load(str(path), load_meshes=False)
            for q in samples:
                a, b = fk(upstream_robot, q), fk(built, q)
                fk_error = max(fk_error, max(np.abs(a[k] - b[k]).max() for k in a))
        result['max_fk_error'] = float(fk_error)
        client = pybullet.connect(pybullet.DIRECT)
        body = pybullet.loadURDF(str(variants['_basic']), useFixedBase=True, physicsClientId=client)
        result['pybullet_joints'] = pybullet.getNumJoints(body, physicsClientId=client)
        result['upstream_joints'] = len(upstream_robot.joint_map)
        pybullet.disconnect(client)
        result['mujoco'] = mujoco_check(args.package / 'mjcf' / f'g20_{side}.xml', upstream_robot, samples,
                                        args.media, side, args.size, len(parts))
        result['sapien'] = sapien_render(variants[''], args.media, side, args.size, palette)
        summary[side] = result
    text = json.dumps(summary, indent=1, default=str)
    (args.media / 'validation.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
