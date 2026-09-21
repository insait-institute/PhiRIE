"""The candidate validator checks the actual shared static XML and mesh bytes."""
import json
import sys
from pathlib import Path
import pytest

from agents.core import common as C
from robo.sim import export_mjcf as exporter
from robo.eval import e4_candidate_screen as screen
from tests.test_e4_automatic_static_package import paired_sources
from tests.test_e4_automatic_materializer import automatic_factory_input


def test_both_actual_exports_consume_one_verified_static_package(paired_sources,monkeypatch):
    root,factory_list=paired_sources
    factories=dict(zip(screen.POLICIES,factory_list,strict=True))
    spec_path=root/'shared_room_static_spec.json';spec_path.write_bytes(screen._json_bytes(screen._automatic_static_spec('09c1414f1b')))
    package=root/'shared_room_static'
    built=exporter.build_common_room_static_package(factory_list,package,screen._automatic_static_spec('09c1414f1b'),screen._sha256(spec_path))
    results={}
    for policy,factory in factories.items():
        monkeypatch.setattr(C,'OUT',factory)
        monkeypatch.setattr(C,'PIPELINE_MESH_PLY',factory/'derived_mesh.ply')
        monkeypatch.setattr(sys,'argv',['export_mjcf','--test','--collision-mode','room',
            '--background-carve-factory',str(factories['A0']),'--background-carve-factory',str(factories['A4']),
            '--room-diagnostic-spec',str(spec_path),'--room-static-package',str(package)])
        exporter.main()
        results[policy]=screen.validate_full_room_export(factory,scene_id='09c1414f1b',policy=policy,
            root=root,expected_object_slots=[],expected_discovered_slots=['obj_1000'],automatic_factories=factories)
    assert results['A0']['static_collision']==results['A4']['static_collision']
    assert results['A0']['shared_static_package']==results['A4']['shared_static_package']
    assert results['A0']['shared_static_package']['manifest_sha256']==built['static_package_identity']['manifest_sha256']
    # Alter actual emitted static geometry while retaining the copied package receipt.
    xml=factories['A4']/'sim_export/scene.xml'
    xml.write_text(xml.read_text().replace('size="20 20 1"','size="10 10 1"'))
    with pytest.raises(screen.CandidateScreenError,match='floor differs'):
        screen.validate_full_room_export(factories['A4'],scene_id='09c1414f1b',policy='A4',root=root,
            expected_object_slots=[],expected_discovered_slots=['obj_1000'],automatic_factories=factories)
