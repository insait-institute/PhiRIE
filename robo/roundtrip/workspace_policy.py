"""Seal a DEV-only partial workspace contrast for the canonical native harness."""
import argparse
import copy
import json
from pathlib import Path
from robo.roundtrip.scope_bundle import artifact_hashes
from robo.roundtrip.identity import file_hash
from robo.roundtrip.spec import validate_spec
from robo.manifest.hash import canonical_hash


def _checked(path,digest):
    if file_hash(path)!=digest:raise ValueError('workspace source changed')
    return json.loads(Path(path).read_text())


def validate_workspace_bundle(bundle,config,baseline_episode,baseline_config):
    validate_spec(config);baseline=json.loads(Path(baseline_config).read_text());validate_spec(baseline)
    if (config['scope']!='DEV_partial_observed_workspace' or baseline['scope']!='L0_target_only' or
            config['controller_method']!=baseline['controller_method'] or config['instance']['split']!='development'):
        raise ValueError('explicit DEV partial workspace to L0 same-method contrast required')
    if canonical_hash({k:v for k,v in config.items() if k not in ('scope','replacement_scope')})!=canonical_hash({k:v for k,v in baseline.items() if k not in ('scope','replacement_scope')}):
        raise ValueError('workspace contrast changes another frozen field')
    source=_checked(bundle['engineering_config']['path'],bundle['engineering_config']['sha256'])
    if source['tier']!='DEV' or source['canonical_instance_id']!=config['canonical_instance_id']:
        raise ValueError('workspace constructor canonical differs')
    for p,h in source['input_files'].items():
        if file_hash(p)!=h:raise ValueError('frozen workspace construction closure differs')
    coverage=_checked(bundle['workspace_coverage']['path'],bundle['workspace_coverage']['sha256'])
    if bundle['workspace_coverage']['path']!=source['coverage'] or coverage['native_asset_access'] is not False:
        raise ValueError('workspace coverage source differs')
    expected={e['object_id']:e for e in source['entities']}
    if [e['object_id'] for e in bundle['entities']]!=[e['object_id'] for e in source['entities']]:raise ValueError('fixed four-role inventory differs')
    for entity in bundle['entities']:
        prior=expected[entity['object_id']]
        native_role={'target':'obj','destination':'container'}.get(entity['object_id'],prior['body_name'].removesuffix('_main'))
        if entity.get('native_role')!=native_role:raise ValueError('workspace evaluator role differs')
        if entity['object_dir']!=prior['object_dir'] or entity['role']!=prior['role'] or entity.get('component_kind')!=prior.get('component_kind') or entity['body_name']!=prior['body_name']:
            raise ValueError('workspace native role/artifact binding differs')
        if artifact_hashes(entity['object_dir'])!=prior['artifact_hashes'] or entity['artifact_hashes']!=prior['artifact_hashes']:
            raise ValueError('workspace geometry/pose/physics changed')
    result_path=Path(baseline_episode)/'result.json';result=json.loads(result_path.read_text())
    if (result.get('executed') is not True or result.get('error') is not None or
            result.get('config_sha256')!=canonical_hash(baseline) or
            file_hash(result_path)!=bundle['baseline_result_sha256']):
        raise ValueError('matching executed baseline required; success not an admission gate')
    if bundle.get('L2_READY') is not False or bundle.get('scope')!='DEV_partial_observed_workspace':
        raise ValueError('partial diagnostic cannot become complete L2')
    return bundle


def seal(engineering_config,baseline_config,baseline_episode,out):
    engineering_config=Path(engineering_config);source=json.loads(engineering_config.read_text())
    baseline=json.loads(Path(baseline_config).read_text());config=copy.deepcopy(baseline)
    config.update(scope='DEV_partial_observed_workspace',replacement_scope='partial_observed_workspace')
    entities=[]
    for prior in source['entities']:
        e=copy.deepcopy(prior);e['native_role']=e['body_name'].removesuffix('_main')
        if e['object_id']=='target':e['native_role']='obj'
        if e['object_id']=='destination':e['native_role']='container'
        e['constructor_method']=('B3_AGENT_NATIVE' if e['object_id'] in ('target','destination') else
            'OBSERVED_WORKSPACE_COMPONENT_TSDF_EXPLICIT_COLLISION_REPAIR' if e['object_id']=='source_support' else 'OBSERVED_WORKSPACE_COMPONENT_TSDF')
        entities.append(e)
    result=dict(schema_version=1,scope=config['scope'],L2_READY=False,system_variant='DEV_B3_target_plate_observed_supports',
        entities=entities,canonical_instance_id=source['canonical_instance_id'],controller_decision=None,
        engineering_config=dict(path=str(engineering_config.resolve()),sha256=file_hash(engineering_config)),
        workspace_coverage=dict(path=source['coverage'],sha256=file_hash(source['coverage'])),
        baseline_result_sha256=file_hash(Path(baseline_episode)/'result.json'))
    validate_workspace_bundle(result,config,baseline_episode,baseline_config)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('config.json',config),('scope_bundle.json',result)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('engineering-config','baseline-config','baseline-episode','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();seal(a.engineering_config,a.baseline_config,a.baseline_episode,a.out)
if __name__=='__main__':main()
