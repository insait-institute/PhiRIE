"""The CPU wrapper only delegates to the canonical sealed materializer."""
import json
import sys

import pytest
import yaml

from run.icra2027 import e4_automatic_materialization as launcher
from tests.test_e4_automatic_materializer import automatic_factory_input
from tests.test_e3_factory_materializer import _write_json
from robo.manifest.hash import canonical_hash


@pytest.fixture
def launch_input(automatic_factory_input):
    root,out,descriptor,jobs,audit=automatic_factory_input
    freeze=root/'e4-freeze';(freeze/'contract').mkdir(parents=True)
    config=dict(schema_version=1,scope='automatic_materialization_engineering',paper_ready=False,
        freeze_id=freeze.name,evidence_root=str(root),e3_root=str(out),e3_freeze_id=jobs['freeze_id'],
        e3_source_commit=jobs['source_contract']['code_commit'],scene_id='09c1414f1b',
        planned_jobs_per_policy=1,policies=['A0','A4'],python=sys.executable,
        automatic_scene_descriptor=json.loads(descriptor.read_text()),source_files={
            'inventory_seal':launcher.materializer._identity(out/'input_inventory/seal.json')})
    path=root/'launcher.yaml';path.write_text(yaml.safe_dump(config))
    contract=dict(freeze_id=freeze.name,code={'commit':'d'*40,'dirty':False},resource_inventory=[{
        'id':'e4_automatic_materialization_config','sha256':launcher.e3.sha256_file(path)},
        {'id':'materialization_python','sha256':launcher.e3.sha256_file(sys.executable)}])
    contract['contract_sha256']=canonical_hash(contract)
    _write_json(freeze/'contract/freeze_manifest.json',contract)
    return config,path,freeze,contract


def test_cpu_launcher_preserves_both_arms_and_complete_population(launch_input,monkeypatch):
    config,path,freeze,_=launch_input
    monkeypatch.setattr(launcher.socket,'gethostname',lambda:'sof1-test')
    result=launcher.run(path,freeze,'preflight')
    assert result['planned_jobs_per_policy']==1 and result['planned_policy_object_rows']==2
    result=launcher.run(path,freeze,'materialize')
    assert result['policies']['A0']['roster']['rejected_count']==1
    assert result['policies']['A4']['roster']['abstained_count']==1
    assert result['paper_ready'] is False and result['gpu_requested'] is False


@pytest.mark.parametrize('mutation',['source_commit','population','config_hash','contract_digest','source_file'])
def test_launcher_rejects_drift_before_materialization(launch_input,mutation):
    config,path,freeze,contract=launch_input
    if mutation=='source_commit':contract['code']['commit']='b'*40
    elif mutation=='population':config['planned_jobs_per_policy']=0
    elif mutation=='config_hash':contract['resource_inventory'][0]['sha256']='0'*64
    elif mutation=='contract_digest':contract['contract_sha256']='0'*64
    else:config['source_files']['inventory_seal']['sha256']='0'*64
    with pytest.raises(ValueError):
        launcher.validate_inputs(config,freeze,contract,{'commit':'d'*40},path)
    assert not (freeze/'harness/materialized').exists()


def test_launcher_disallows_nonapproved_execution_host(launch_input,monkeypatch):
    _,path,freeze,_=launch_input
    monkeypatch.setattr(launcher.socket,'gethostname',lambda:'msp3')
    with pytest.raises(ValueError,match='allowed hosts'):
        launcher.run(path,freeze,'materialize')
    assert not (freeze/'harness/materialized').exists()


@pytest.mark.parametrize('module',['materializer','e3'])
def test_launcher_rejects_foreign_imported_source(launch_input,monkeypatch,module):
    config,path,freeze,contract=launch_input
    target=getattr(launcher,module)
    monkeypatch.setattr(target,'__file__','/foreign-checkout/robo/eval/module.py')
    with pytest.raises(ValueError,match='implementation differs from launcher source'):
        launcher.validate_inputs(config,freeze,contract,{'commit':'d'*40},path)
    assert not (freeze/'harness/materialized').exists()


def test_launcher_rejects_harness_symlink_before_descriptor_write(launch_input):
    _,path,freeze,_=launch_input
    outside=freeze.parent/'outside';outside.mkdir()
    (freeze/'harness').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        launcher.run(path,freeze,'preflight')
    assert list(outside.iterdir())==[]


def test_launcher_does_not_adopt_valid_foreign_source_materialization(launch_input,monkeypatch):
    import shutil
    config,path,freeze,_=launch_input
    monkeypatch.setattr(launcher.socket,'gethostname',lambda:'sof1-test')
    launcher.run(path,freeze,'preflight')
    other=freeze.parent/'other-agentic'
    shutil.copytree(config['e3_root'],other)
    destination=freeze/'harness/materialized/A0'
    launcher.materializer.materialize_factory_variant(e3_root=other,
        scene_id=config['scene_id'],policy_id='A0',out=destination,
        automatic_scene_contract=freeze/'harness/automatic_scene_descriptor.json')
    original=(destination/'materialization_manifest.json').read_bytes()
    launcher.materializer.validate_materialized_factory(destination)
    with pytest.raises(ValueError,match='another frozen source'):
        launcher.run(path,freeze,'materialize')
    assert (destination/'materialization_manifest.json').read_bytes()==original
    assert not (freeze/'harness/materialized/A4').exists()


def test_slurm_spool_copy_uses_explicit_frozen_source(tmp_path):
    """Slurm executes a copied script, not the source checkout's script path."""
    import os
    import shutil
    import subprocess
    spool = tmp_path/'slurmd/job000001'; spool.mkdir(parents=True)
    script = spool/'slurm_script'
    shutil.copyfile(launcher.CODE/'run/icra2027/run_e4_automatic_materialization.sh', script)
    env = dict(os.environ, E4_CODE=str(launcher.CODE))
    # Invalid phase reaches the real module's argparse without touching a freeze.
    result = subprocess.run(['bash', str(script), 'not-a-phase'], cwd=spool,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "invalid choice: 'not-a-phase'" in result.stderr
    assert 'No module named' not in result.stderr
    assert not (spool/'outputs').exists()


@pytest.mark.parametrize('value', [None, 'relative-source', '/does-not-exist'])
def test_slurm_spool_copy_rejects_missing_or_invalid_source(tmp_path, value):
    import os
    import shutil
    import subprocess
    script = tmp_path/'slurm_script'
    shutil.copyfile(launcher.CODE/'run/icra2027/run_e4_automatic_materialization.sh', script)
    env = dict(os.environ)
    env.pop('E4_CODE', None)
    if value is not None:
        env['E4_CODE'] = value
    result = subprocess.run(['bash', str(script), 'preflight'], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'No module named' not in result.stderr
    assert not (tmp_path/'outputs').exists()
