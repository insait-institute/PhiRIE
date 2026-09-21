"""Reuse contract tests; historical validator is the only substituted producer."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from robo.manifest.hash import canonical_hash
from run.icra2027 import e4_automatic_export_reuse as reuse


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    root=tmp_path/'evidence'; root.mkdir()
    code=tmp_path/'old-code'; code.mkdir()
    subprocess.run(['git','init','-q',str(code)],check=True)
    (code/'producer.txt').write_text('frozen original producer\n')
    write(code/'qualification.json', {'scope':'unit fixture'})
    subprocess.run(['git','-C',str(code),'add','.'],check=True)
    subprocess.run(['git','-C',str(code),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','original source'],check=True)
    commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    stage=root/'outputs/icra2027'/f'20260905-{commit[:7]}-v1'
    scene='27dd4da69e'; parent=stage/'automatic_candidates'/scene/'materialized'
    for policy in reuse.screen.POLICIES:
        factory=parent/policy
        mesh=factory/'objects/obj_1000/mesh_sim.obj'
        mesh.parent.mkdir(parents=True); mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
        manifest={'schema_version':2,'scene_id':scene,'policy_id':policy,'destination':str(factory.relative_to(root)),
                  'created_utc':'historical','provenance':{'code_root':str(code),'materializer_commit':commit,'validator_commit':commit},
                  'roster':{'accepted_slots':['obj_1000'],'object_slots':['obj_1000']},
                  'output_members':{'objects/obj_1000/mesh_sim.obj':{'sha256':reuse.api.identity(mesh)['sha256']}}}
        write(factory/'materialization_manifest.json',manifest)
        write(factory/'seal.json',{'manifest_sha256':reuse.api.identity(factory/'materialization_manifest.json')['sha256']})
    package=parent/'shared_room_static'
    write(package/'manifest.json',{'source_manifest_sha256':{p:reuse.api.identity(parent/p/'materialization_manifest.json')['sha256'] for p in reuse.screen.POLICIES}})
    write(package/'seal.json',{'manifest_sha256':reuse.api.identity(package/'manifest.json')['sha256']})
    (package/'room.obj').write_text('static mesh bytes\n')
    (parent/'shared_room_static_spec.json').write_bytes(reuse.screen._json_bytes(reuse.screen._automatic_static_spec(scene)))
    write(parent/'A0/sim/body.json',{'object':'obj_1000'})
    export=parent/'A0/sim_export'; export.mkdir()
    (export/'scene.xml').write_text(f'<mujoco><compiler meshdir="{parent / "A0"}"/></mujoco>\n')
    write(export/'mujoco_settle.json',{'n':1,'stable_3cm':0,'drift_m':{'obj_1000':.5}})
    write(export/'room_collision_report.json',{'mode':'room'})
    write(export/'isaac_manifest.json',{'objects':[{'name':'obj_1000'}]})
    runtime=root/'runtime/python'; runtime.parent.mkdir(); runtime.write_text('fixture runtime\n')
    resources=[]
    for name,path in [('e4_compact_qualification_config',code/'qualification.json'),('qualification_python',runtime)]:
        identity=reuse.api.identity(path)
        resources.append({'id':name,'resolved_path':str(path),'sha256':identity['sha256'],'size_bytes':identity['size_bytes']})
    contract={'code':{'repository':str(code),'commit':commit,'dirty':False},'freeze_id':stage.name,'resource_inventory':resources}
    contract['contract_sha256']=canonical_hash(contract)
    write(stage/'contract/freeze_manifest.json',contract)
    monkeypatch.setattr(reuse.screen,'evidence_root',lambda:root)
    actual_run=subprocess.run
    calls=[]
    def historical(args,**kwargs):
        if args[:2]==[str(runtime),'-c']:
            calls.append((args,kwargs))
            assert kwargs['cwd']==str(code)
            assert kwargs['env']['SIMANY_SCENE']==scene
            assert 'PYTHONPATH' not in kwargs['env']
            supplied=json.loads(kwargs['input'])
            report={'materializations':{},'export':{'artifacts':{}}}
            for p in reuse.screen.POLICIES:
                report['materializations'][p]={'manifest_sha256':supplied['identities'][p+'_manifest']['sha256'],
                    'materializer_commit':commit,'validator_commit':commit,'code_root':str(code),
                    'scene_id':scene,'policy_id':p}
            for role,name in {'scene_xml':'scene.xml','mujoco_settle':'mujoco_settle.json',
                              'room_collision_report':'room_collision_report.json','isaac_manifest':'isaac_manifest.json'}.items():
                report['export']['artifacts'][role]=reuse.screen._identity(export/name,root=root)
            return SimpleNamespace(stdout=json.dumps(report),returncode=0)
        return actual_run(args,**kwargs)
    monkeypatch.setattr(reuse.subprocess,'run',historical)
    source=reuse.inspect_source(stage,scene)
    assert len(calls)==1
    return SimpleNamespace(root=root,source=source,parent=parent,code=code,scene=scene)


def test_inspection_and_copy_preserve_original_xml_and_negative_settle(source_fixture):
    f=source_fixture
    assert reuse.validate_source(f.source)==f.source
    target=f.root/'new-stage/materialized'; (target/'A0').mkdir(parents=True)
    reuse.copy_a0(target,f.source)
    for dirname in ('sim','sim_export'):
        for relative,identity in f.source['trees']['A0_'+dirname]['members'].items():
            assert (target/'A0'/dirname/relative).read_bytes()==Path(identity['path']).read_bytes()
    assert str(f.parent/'A0') in (target/'A0/sim_export/scene.xml').read_text()
    assert json.loads((target/'A0/sim_export/mujoco_settle.json').read_text())['stable_3cm']==0
    reuse.copy_a0(target,f.source)  # Valid identical resume does not regenerate.


@pytest.mark.parametrize('field',['manifest_path','static_path','sim_path','factory_path','identity_roster'])
def test_hashed_but_unrelated_source_paths_are_rejected(source_fixture,field):
    f=source_fixture; source=copy.deepcopy(f.source)
    if field=='manifest_path':
        wrong=f.root/'unrelated-manifest.json'; shutil.copyfile(source['identities']['A0_manifest']['path'],wrong)
        source['identities']['A0_manifest']=reuse.api.identity(wrong)
    elif field=='static_path': source['trees']['static']['path']=str(f.root/'other-static')
    elif field=='sim_path': source['trees']['A0_sim']['path']=source['trees']['A0_sim_export']['path']
    elif field=='factory_path': source['factories']['A0']=source['factories']['A4']
    else: source['identities'].pop('A4:objects/obj_1000/mesh_sim.obj')
    with pytest.raises(ValueError): reuse.validate_source(source)


@pytest.mark.parametrize('member',['A0:objects/obj_1000/mesh_sim.obj','static_spec','A4_manifest'])
def test_changed_object_paired_manifest_or_static_spec_rejected(source_fixture,member):
    f=source_fixture; path=Path(f.source['identities'][member]['path'])
    if member=='A4_manifest':
        value=json.loads(path.read_text()); value['scene_id']='40aec5fffa'; write(path,value)
    else:
        path.write_bytes(path.read_bytes()+b' changed')
    with pytest.raises((ValueError,KeyError)): reuse.validate_source(f.source)


def test_original_settle_tampering_and_destination_overwrite_are_rejected(source_fixture):
    f=source_fixture; target=f.root/'new-stage/materialized'; (target/'A0').mkdir(parents=True)
    reuse.copy_a0(target,f.source)
    destination=target/'A0/sim_export/mujoco_settle.json'
    destination.write_text('do not overwrite me')
    with pytest.raises(ValueError,match='copied A0 export differs'):
        reuse.copy_a0(target,f.source)
    assert destination.read_text()=='do not overwrite me'
    original=f.parent/'A0/sim_export/mujoco_settle.json'
    original.write_text('tampered original')
    with pytest.raises(ValueError): reuse.validate_source(f.source)


def test_dirty_or_changed_original_code_is_rejected(source_fixture):
    f=source_fixture
    (f.code/'producer.txt').write_text('changed code')
    with pytest.raises(ValueError,match='exact and clean'): reuse.validate_source(f.source)


def test_inventory_rejects_symlink_root_and_members(tmp_path):
    directory=tmp_path/'real'; directory.mkdir(); (directory/'mesh.obj').write_text('mesh')
    link=tmp_path/'alias'; link.symlink_to(directory,target_is_directory=True)
    with pytest.raises(ValueError,match='root'): reuse.inventory(link)
    (directory/'linked.obj').symlink_to(directory/'mesh.obj')
    with pytest.raises(ValueError,match='symlink'): reuse.inventory(directory)


def destination_fixture(f, monkeypatch):
    parent=f.root/'new-stage/materialized'
    for policy in reuse.screen.POLICIES:
        target=parent/policy
        target.mkdir(parents=True)
        value=json.loads((f.parent/policy/'materialization_manifest.json').read_text())
        value['destination']=str(target.relative_to(f.root))
        value['created_utc']='new-stage'
        value['provenance']={'code_root':'current source','materializer_commit':'b'*40,'validator_commit':'b'*40}
        write(target/'materialization_manifest.json',value)
    code={'code_root':'current source','commit':'b'*40,'dirty':False}
    monkeypatch.setattr(reuse.screen,'_code_snapshot',lambda expected:code if expected==code['commit'] else None)
    reuse.publish(parent,f.source,code=code)
    return parent,code


@pytest.mark.parametrize('field',['scene_id','policy_id','roster','output_members'])
def test_destination_core_drift_rejected(source_fixture,monkeypatch,field):
    f=source_fixture; parent,_=destination_fixture(f,monkeypatch)
    assert reuse.resolve(parent/'A0',scene_id=f.scene,root=f.root)==f.source
    path=parent/'A4/materialization_manifest.json'; value=json.loads(path.read_text())
    if field=='scene_id': value[field]='40aec5fffa'
    elif field=='policy_id': value[field]='A0'
    elif field=='roster': value[field]['accepted_slots']=[]
    else: value[field]['objects/obj_1000/mesh_sim.obj']['sha256']='0'*64
    write(path,value)
    with pytest.raises(ValueError,match='construction/config/object population'):
        reuse.resolve(parent/'A0',scene_id=f.scene,root=f.root)


def test_destination_receipt_tampering_and_paired_path_drift_rejected(source_fixture,monkeypatch):
    f=source_fixture; parent,code=destination_fixture(f,monkeypatch)
    with pytest.raises(ValueError,match='paired destination'):
        reuse.resolve(parent/'A0',{'A0':parent/'A0','A4':f.parent/'A4'},scene_id=f.scene,root=f.root)
    path=parent/'export_reuse/source.json'; value=json.loads(path.read_text())
    value['destination_parent']=str(f.root/'elsewhere'); write(path,value)
    with pytest.raises(reuse.screen.CandidateScreenError,match='member changed'):
        reuse.resolve(parent/'A0',scene_id=f.scene,root=f.root)
    with pytest.raises(reuse.screen.CandidateScreenError,match='member changed'):
        reuse.publish(parent,f.source,code=code)


def test_original_validator_report_cannot_claim_different_settle_bytes(source_fixture):
    source=copy.deepcopy(source_fixture.source)
    source['original_validation']['export']['artifacts']['mujoco_settle']['sha256']='0'*64
    with pytest.raises(ValueError,match='export identity'): reuse.validate_source(source)


def test_new_a4_export_authenticates_current_context_then_uses_original_static(source_fixture,monkeypatch):
    f=source_fixture; parent,_=destination_fixture(f,monkeypatch)
    current={p:parent/p for p in reuse.screen.POLICIES}
    events=[]
    def current_context(factories,*,scene_id,root):
        assert factories==current and scene_id==f.scene and root==f.root
        events.append('current_context_validated')
        return {}
    monkeypatch.setattr(reuse.screen,'_automatic_export_context',current_context)
    original_run=reuse.subprocess.run
    exported=[]
    def export(args,**kwargs):
        if '-m' in args and 'robo.sim.export_mjcf' in args:
            assert events==['current_context_validated']
            exported.append((args,kwargs))
            return SimpleNamespace(returncode=0)
        return original_run(args,**kwargs)
    monkeypatch.setattr(reuse.subprocess,'run',export)
    monkeypatch.delenv('TMPDIR',raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME',raising=False)
    reuse.screen._run_full_room_export(current['A4'],scene_id=f.scene,root=f.root,
        common_carve_factories=list(current.values()),automatic=True)
    assert len(exported)==1  # No shared static generation was requested.
    args,kwargs=exported[0]
    carve=[args[i+1] for i,v in enumerate(args) if v=='--background-carve-factory']
    assert carve==[str(f.parent/p) for p in reuse.screen.POLICIES]
    assert args[args.index('--room-static-package')+1]==str(f.parent/'shared_room_static')
    assert args[args.index('--room-diagnostic-spec')+1]==str(f.parent/'shared_room_static_spec.json')
    assert kwargs['env']['SIMANY_OUT']==str(current['A4'])
    assert kwargs['env']['SIMANY_SCENE']==f.scene
    assert kwargs['env']['SIMANY_NO_GT']=='1'


def test_verified_a0_copy_rejects_any_added_or_modified_member(source_fixture):
    f=source_fixture; parent=f.root/'new-stage/materialized'; (parent/'A0').mkdir(parents=True)
    reuse.copy_a0(parent,f.source)
    reuse.verify_a0(parent,f.source)
    (parent/'A0/sim_export/unrecorded.json').write_text('{}')
    with pytest.raises(ValueError,match='measured bytes changed'):
        reuse.verify_a0(parent,f.source)
    (parent/'A0/sim_export/unrecorded.json').unlink()
    (parent/'A0/sim_export/scene.xml').write_text('changed meshdir')
    with pytest.raises(ValueError,match='measured bytes changed'):
        reuse.verify_a0(parent,f.source)


def test_static_package_must_bind_both_exact_original_factory_manifests(source_fixture):
    f=source_fixture; source=copy.deepcopy(f.source)
    path=Path(source['identities']['static_manifest']['path'])
    package=json.loads(path.read_text())
    package['source_manifest_sha256']['A4']='0'*64
    write(path,package)
    # Rehashing an unrelated package must not turn it into the paired package.
    source['identities']['static_manifest']=reuse.api.identity(path)
    source['trees']['static']['members']=reuse.inventory(path.parent)
    with pytest.raises(ValueError,match='paired'):
        reuse.validate_source(source)
