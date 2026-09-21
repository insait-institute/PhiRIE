"""Automatic scene reads stay authenticated when static writes use scratch."""
import json

import pytest

from agents.core import common as C
from robo.eval import e3_factory_materializer as materializer
from robo.sim import export_mjcf as export
from tests.test_e4_automatic_materializer import automatic_factory_input
from tests.test_e4_room_diagnostic_core import _spec


@pytest.fixture
def paired_sources(automatic_factory_input, monkeypatch):
    root, out, descriptor, *_ = automatic_factory_input
    factories = []
    for policy in ('A0', 'A4'):
        path = root / policy
        materializer.materialize_factory_variant(e3_root=out, scene_id='09c1414f1b',
            policy_id=policy, out=path, automatic_scene_contract=descriptor)
        factories.append(path)
    monkeypatch.setattr(C, 'SCENE_ID', '09c1414f1b')
    monkeypatch.setattr(C, 'OUT', factories[0])
    monkeypatch.setattr(C, 'PIPELINE_MESH_PLY', factories[0] / 'derived_mesh.ply')
    monkeypatch.setattr(C, 'env', lambda name, *a: '1' if name == 'AUTO' else None)
    return root, factories


def spec():
    return _spec(variant_id='legacy-equivalent', hull_bottom='raw', intrusive_primitive='fail')


def test_real_automatic_package_uses_source_instances_and_private_writes(paired_sources):
    root, factories = paired_sources
    before = {str(p): export._sha256_file(p) for f in factories for p in f.rglob('*') if p.is_file()}
    package = root / 'shared-static'
    built = export.build_common_room_static_package(factories, package, spec(), 'a' * 64)
    assert C.OUT == factories[0]
    after = {str(p): export._sha256_file(p) for f in factories for p in f.rglob('*') if p.is_file()}
    assert before == after
    carve = json.loads((package / 'background_carve.json').read_text())
    assert carve['discovered_slots'] == ['obj_1000'] and carve['carved_slots'] == []
    assert carve['source_face_count'] == carve['kept_face_count'] == 4
    assert carve['source_mesh_sha256'] == export._sha256_file(factories[0] / 'derived_mesh.ply')
    loaded = export.load_common_room_static_package(package, spec(), 'a' * 64)
    assert loaded['static_package_identity'] == built['static_package_identity']
    assert set(loaded['source_manifest_sha256']) == {'A0', 'A4'}
    with pytest.raises(FileExistsError):
        export.build_common_room_static_package(factories, package, spec(), 'a' * 64)


@pytest.mark.parametrize('damage', ['foreign_source', 'gt_mesh', 'auto_unset', 'scratch_in_factory', 'scratch_ancestor', 'scratch_symlink'])
def test_explicit_automatic_source_cannot_bypass_guards(paired_sources, monkeypatch, damage):
    root, factories = paired_sources
    source = factories[0]
    scratch = root / 'scratch'; scratch.mkdir()
    if damage == 'foreign_source':
        source = root / 'foreign'; source.mkdir()
    elif damage == 'gt_mesh':
        monkeypatch.setattr(C, 'PIPELINE_MESH_PLY', root / 'gt.ply')
    elif damage == 'auto_unset':
        monkeypatch.setattr(C, 'env', lambda *a: None)
    elif damage == 'scratch_in_factory':
        scratch = factories[0] / 'scratch'
    elif damage == 'scratch_ancestor':
        scratch = root
    elif damage == 'scratch_symlink':
        alias = root / 'alias'; alias.symlink_to(scratch, target_is_directory=True)
        scratch = alias
    monkeypatch.setattr(C, 'OUT', scratch)
    original_loader = C.load_auto_instances
    def validated_upstream_only(**kw):
        # Materializer replay legitimately rereads its sealed discovery inputs;
        # the active factory/scratch/GT route must not reach the final loader.
        assert kw['instances_path'] == root / 'discovery/construction/auto_instances.npz'
        assert kw['mesh_path'] == root / 'discovery/construction/derived_mesh.ply'
        return original_loader(**kw)
    monkeypatch.setattr(C, 'load_auto_instances', validated_upstream_only)
    with pytest.raises(ValueError):
        export._room_static_report(factories, spec(), root / 'collision', source_factory=source)
    assert not (root / 'collision').exists()


def test_package_rejects_symlink_parent_before_writing(paired_sources):
    root, factories = paired_sources
    outside = root / 'outside'; outside.mkdir()
    link = root / 'link'; link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        export.build_common_room_static_package(factories, link / 'package', spec(), 'a' * 64)
    assert list(outside.iterdir()) == []


def test_explicit_scene_load_is_independent_of_global_default(paired_sources, monkeypatch):
    root, factories = paired_sources
    package = root / 'explicit-scene'
    declared = spec()
    built = export.build_common_room_static_package(factories, package, declared, 'a' * 64)
    monkeypatch.setattr(C, 'SCENE_ID', 'c50d2d1d42')
    loaded = export.load_common_room_static_package(package, declared, 'a' * 64,
        expected_scene_id='09c1414f1b')
    assert loaded['static_package_identity'] == built['static_package_identity']
    assert C.SCENE_ID == 'c50d2d1d42'
    # Legacy callers retain their global-scene constraint.
    with pytest.raises(ValueError, match='scene binding'):
        export.load_common_room_static_package(package, declared, 'a' * 64)
    with pytest.raises(ValueError, match='scene binding'):
        export.load_common_room_static_package(package, declared, 'a' * 64,
            expected_scene_id='27dd4da69e')


def test_automatic_static_caller_passes_explicit_scene(paired_sources, monkeypatch):
    from robo.eval import e4_candidate_screen as screen
    root, factories = paired_sources
    declared = screen._automatic_static_spec('09c1414f1b')
    path = root / 'shared_room_static_spec.json'
    path.write_bytes(screen._json_bytes(declared))
    expected = {arm: export._sha256_file(factory/'materialization_manifest.json')
                for arm, factory in zip(('A0','A4'), factories)}
    def load(package, diagnostic, digest, *, expected_scene_id):
        assert expected_scene_id == '09c1414f1b' and diagnostic == declared
        assert digest == export._sha256_file(path)
        return {'source_manifest_sha256': expected}
    monkeypatch.setattr(export, 'load_common_room_static_package', load)
    monkeypatch.setattr(C, 'SCENE_ID', 'c50d2d1d42')
    result = screen._load_automatic_static_package(factories[0],dict(zip(('A0','A4'),factories)),
        scene_id='09c1414f1b',root=root)
    assert result['source_manifest_sha256'] == expected
