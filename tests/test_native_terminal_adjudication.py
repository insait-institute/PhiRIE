import copy
import json
from pathlib import Path

import pytest
import trimesh
import yaml

from robo.roundtrip import matrix as m


def unit():
    config=yaml.safe_load(Path('configs/experiments/sim_recon_sim/scale_up/dev.yaml').read_text())
    _,rows=m.enumerate_plan(config)
    return {**rows[0],'canonical_instance_id':'native-'+'a'*64,'unit_id':'unit-a','terminal_status':'NOT_SCHEDULED'}


def fixture(tmp_path, *, valid=False):
    import inspect
    import subprocess
    import sys
    import numpy as np
    from robo.roundtrip.importers import robocasa
    u=unit();u['controller_method']='B0_FIXED_NATIVE'
    directory=tmp_path/'object';part=directory/'collision/part_04.obj'
    part.parent.mkdir(parents=True)
    mesh=trimesh.creation.box()
    if not valid:mesh.update_faces([0,1,2])
    mesh.export(part)
    def bind(p):return {'path':str(p),'sha256':m.sha(p)}
    config=tmp_path/'unit_config.json';m.save_new(config,{'frozen':'episode'})
    u.update(config_path=str(config),config_sha256=m.canonical_hash(json.loads(config.read_text())))
    terminal=tmp_path/'workers'/u['unit_id']/'terminal.json'
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original={**u,'planned_unit_sha256':m.canonical_hash(u),'terminal_status':'CODE_FAILED',
              'executed':False,'success':None,'returncode':1,'result':None,
              'source_code':{'commit':source,'dirty':False},
              'argv':['existing_runner','--object-dir',str(directory),'--config',str(config)]}
    m.save_new(terminal,original);planned=terminal.parent/'planned_unit.json';m.save_new(planned,u)
    log=terminal.parent/'worker.log'
    error='ValueError: collision part is not a closed convex volume: part_04.obj'
    log.write_text(error+'\n')
    bc=tmp_path/'build_config.json';m.save_new(bc,{'frozen':'construction'})
    build=tmp_path/'build_manifest.json';m.save_new(build,dict(canonical_instance_id=u['canonical_instance_id'],
        config_sha256=m.canonical_hash(json.loads(bc.read_text())),
        source_hashes={'build_config.json':m.sha(bc),'object/collision/part_04.obj':m.sha(part)}))
    audit=tmp_path/'audit.json'
    m.save_new(audit,dict(schema_version=1,kind='independent_original_collision_import_audit',
        classification='METHOD_CONSTRUCTION_IMPORT_VALIDITY_FAILURE',adjudication_supported=True,
        source_dirty=False,geometry_repaired=False,threshold_changed=False,new_policy_outcomes=0,
        proposed_status='BUILD_FAILED',proposed_executed=False,proposed_success=None,
        source_commit=source,trimesh_version=trimesh.__version__,numpy_version=np.__version__,python=sys.executable,
        trimesh_convex_source=bind(inspect.getsourcefile(trimesh.convex)),importer_source=bind(robocasa.__file__),
        canonical_instance_id=u['canonical_instance_id'],build_manifest=bind(build),constructor_config=bind(bc),
        collision_part=bind(part),construction_source_hashes_verified={str(part):m.sha(part)},object_dir=str(directory),
        exact_importer_exception=error,units=[dict(unit_id=u['unit_id'],original_status='CODE_FAILED',
            proposed_status='BUILD_FAILED',executed=False,success=None,reset_id=u['reset_id'],
            terminal=bind(terminal),worker_log=bind(log),planned_unit=bind(planned),config=bind(config),
            config_canonical_sha256=u['config_sha256'])]))
    manifest={'schema_version':1,'kind':'native_collision_failure_adjudication',
              'trimesh_version':trimesh.__version__,'independent_audit':bind(audit),
              'entries':[{'unit_id':u['unit_id'],'original_terminal':bind(terminal),
                          'worker_log':bind(log),'collision_part':bind(part)}]}
    path=tmp_path/'adjudications.json';m.save_new(path,manifest)
    return u,terminal,path,manifest


def test_adjudication_preserves_original_and_unknown_native_outcome(tmp_path):
    u,terminal,path,_=fixture(tmp_path);before=terminal.read_bytes()
    rows=m.collect([u],tmp_path/'workers',tmp_path/'merged',adjudications=path)
    assert terminal.read_bytes()==before
    assert rows[0]['terminal_status']=='BUILD_FAILED'
    assert rows[0]['original_terminal_status']=='CODE_FAILED'
    assert rows[0]['executed'] is False and rows[0]['success'] is None
    s=json.loads((tmp_path/'merged/status.json').read_text())['groups'][0]
    assert s['unmeasured']==0 and s['executed']==0 and s['success_per_planned']==0
    assert rows[0]['classification_adjudication']['original_terminal']['sha256']==m.sha(terminal)
    with pytest.raises(FileExistsError):m.collect([u],tmp_path/'workers',tmp_path/'merged',adjudications=path)


def test_valid_original_collision_cannot_be_classified_as_build_failure(tmp_path):
    u,_,path,_=fixture(tmp_path,valid=True)
    with pytest.raises(ValueError,match='passes the unchanged'):
        m.collect([u],tmp_path/'workers',tmp_path/'merged',adjudications=path)
    assert not (tmp_path/'merged').exists()


@pytest.mark.parametrize('change',['hash','executed','external','duplicate','wrong_asset','wrong_log','runtime','result'])
def test_adjudication_rejects_unproven_failure(tmp_path,change):
    u,terminal,path,manifest=fixture(tmp_path);entry=manifest['entries'][0]
    if change=='hash':entry['collision_part']['sha256']='0'*64
    elif change in ('executed','external'):
        t=json.loads(terminal.read_text())
        if change=='executed':t.update(terminal_status='RECORDED',executed=True,success=True,result={'executed':True,'success':True})
        else:t['terminal_status']='RESOURCE_FAILED'
        terminal.write_text(json.dumps(t));entry['original_terminal']['sha256']=m.sha(terminal)
    elif change=='duplicate':manifest['entries'].append(copy.deepcopy(entry))
    elif change=='wrong_asset':
        other=tmp_path/'part_04.obj';other.write_bytes(Path(entry['collision_part']['path']).read_bytes())
        entry['collision_part']={'path':str(other),'sha256':m.sha(other)}
    elif change=='wrong_log':
        p=Path(entry['worker_log']['path']);p.write_text('CUDA unavailable\n');entry['worker_log']['sha256']=m.sha(p)
    elif change=='runtime':manifest['trimesh_version']='unfrozen-version'
    elif change=='result':
        p=terminal.parent/'runner/episode/result.json';p.parent.mkdir(parents=True);p.write_text('{}')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):m.collect([u],tmp_path/'workers',tmp_path/'merged',adjudications=path)
    assert not (tmp_path/'merged').exists()


@pytest.mark.parametrize('defect',[
    'arbitrary_audit','unsupported_conclusion','changed_source','changed_importer',
    'changed_convex_implementation','audit_runtime','unbound_part','unbound_build_part',
    'wrong_unit','wrong_audit_log','changed_episode_config','changed_construction_config',
    'changed_part_after_build','missing_original_source'])
def test_adjudication_requires_independent_source_and_construction_proof(tmp_path,defect):
    u,terminal,path,manifest=fixture(tmp_path)
    ap=Path(manifest['independent_audit']['path']);audit=json.loads(ap.read_text())
    if defect=='arbitrary_audit':audit={'independent_validation':'asserted without proof'}
    elif defect=='unsupported_conclusion':audit['adjudication_supported']=False
    elif defect=='changed_source':audit['source_commit']='a'*40
    elif defect=='changed_importer':audit['importer_source']['sha256']='0'*64
    elif defect=='changed_convex_implementation':audit['trimesh_convex_source']['sha256']='0'*64
    elif defect=='audit_runtime':audit['trimesh_version']='unfrozen'
    elif defect=='unbound_part':audit['collision_part']['sha256']='0'*64
    elif defect=='unbound_build_part':
        bp=Path(audit['build_manifest']['path']);build=json.loads(bp.read_text());build['source_hashes'].pop('object/collision/part_04.obj')
        bp.write_text(json.dumps(build));audit['build_manifest']['sha256']=m.sha(bp)
    elif defect=='wrong_unit':audit['units'][0]['unit_id']='some-other-unit'
    elif defect=='wrong_audit_log':audit['units'][0]['worker_log']['sha256']='0'*64
    elif defect=='changed_episode_config':Path(audit['units'][0]['config']['path']).write_text('{}')
    elif defect=='changed_construction_config':Path(audit['constructor_config']['path']).write_text('{}')
    elif defect=='changed_part_after_build':
        pp=Path(audit['collision_part']['path']);pp.write_text(pp.read_text()+'\n# post-build changed bytes\n')
        digest=m.sha(pp);audit['collision_part']['sha256']=digest
        audit['construction_source_hashes_verified'][str(pp)]=digest
        manifest['entries'][0]['collision_part']['sha256']=digest
        # The independently sealed original build still binds the original bytes.
    elif defect=='missing_original_source':
        record=json.loads(terminal.read_text());record.pop('source_code');terminal.write_text(json.dumps(record))
        digest=m.sha(terminal);manifest['entries'][0]['original_terminal']['sha256']=digest
        audit['units'][0]['terminal']['sha256']=digest
    ap.write_text(json.dumps(audit));manifest['independent_audit']['sha256']=m.sha(ap);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):m.collect([u],tmp_path/'workers',tmp_path/'merged',adjudications=path)
    assert not (tmp_path/'merged').exists()
