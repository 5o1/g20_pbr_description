"""Configuration consistency; runs without trimesh, MuJoCo or SAPIEN."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'config' / name).read_text())


def test_palette_values_are_valid():
    materials = load('materials.json')['materials']
    assert set(materials) == {'shell_silver', 'rubber_black', 'steel_link', 'steel_dark', 'wrist_aluminium'}
    for entry in materials.values():
        assert re.fullmatch(r'#[0-9a-f]{6}', entry['hex'])
        assert 0 <= entry['metallic'] <= 1 and 0 <= entry['roughness'] <= 1
        assert entry['parts']


def test_every_rule_uses_a_palette_material_and_covers_every_link():
    materials = set(load('materials.json')['materials'])
    parts = load('parts.json')
    upstream = load('upstream.json')
    assert parts['sliver_material'] in materials
    for rules in parts['rules'].values():
        assert rules and all(r['material'] in materials and len(r['extents_mm']) == 3 for r in rules)
    for hand in upstream['hands'].values():
        assert {Path(name).stem for name in hand['mesh_sha256']} == set(parts['rules'])


def test_upstream_pin_is_complete():
    upstream = load('upstream.json')
    assert re.fullmatch(r'[0-9a-f]{40}', upstream['commit'])
    digests = [upstream['license_file']['sha256']]
    for hand in upstream['hands'].values():
        digests += [hand['urdf_sha256'], *hand['mesh_sha256'].values()]
    assert len(digests) == 47 and all(re.fullmatch(r'[0-9a-f]{64}', d) for d in digests)


def test_license_and_notice_name_upstream():
    assert 'Apache License' in (ROOT / 'LICENSE').read_text()
    notice = (ROOT / 'NOTICE').read_text()
    assert 'Copyright 2025 LinkerHand,Ltd.' in notice and load('upstream.json')['commit'] in notice
