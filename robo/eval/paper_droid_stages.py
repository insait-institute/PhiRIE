"""Authenticate already measured DROID construction stages for paper formatting."""
import json
from pathlib import Path
from robo.manifest.hash import canonical_hash,git_snapshot
from agents.recon.droid_extract import verify_input
from run.icra2027.e7_droid_gaussian import tree,verify_tree,validate_training_report


def checked(ref):
    p=Path(ref['path'])
    if not p.is_absolute() or p.resolve(strict=True)!=p:raise ValueError('DROID stage identity uses an alias')
    return verify_input(ref)


def stage_contract(path,audit,resource):
    root=path.parent;e=json.loads((root/'contract/freeze_manifest.json').read_text())
    payload={k:v for k,v in e.items() if k not in {'created_utc','environment','contract_sha256'}}
    config_path=checked(audit['config']);config=json.loads(config_path.read_text())
    code=git_snapshot(Path(e['code']['repository']))
    if (canonical_hash(payload)!=e['contract_sha256'] or e['contract_sha256']!=audit['contract_sha256']
        or e['freeze_id']!=root.name or audit['freeze_id']!=root.name or e['code']['dirty']
        or code['dirty'] or code['commit']!=audit['source_commit'] or e['code']['commit']!=audit['source_commit']):
        raise ValueError('DROID stage source/E0 changed')
    refs=[r for r in e['resource_inventory'] if r['id']==resource]
    if len(refs)!=1 or refs[0]['sha256']!=audit['config']['sha256']:raise ValueError('DROID stage config binding changed')
    scope={'e7_tail_config':'complete_original_droid_public_tail_integrity',
           'e7_gaussian_config':'complete_original_droid_gaussian_integrity'}[resource]
    if (audit['scope'] != scope
        or audit['status']!='PASS' or audit['planned_workspaces']!=10 or audit['schema_version']!=1
        or any(audit.get(k) is not False for k in ['paper_ready','full_build','full_e7_complete','independent_image_fidelity'])):
        raise ValueError('DROID stage scope/claim differs')
    if len(config['workspace_ids'])!=10 or [r['workspace_id'] for r in audit['rows']]!=config['workspace_ids'] or len(set(config['workspace_ids']))!=10:
        raise ValueError('DROID stage full roster changed')
    return config


def receipt(ref,path,audit,wid):
    if checked(ref)!=path:raise ValueError('DROID receipt workspace/path changed')
    r=json.loads(path.read_text())
    if r['code']['commit']!=audit['source_commit'] or r['code'].get('dirty') is not False or r['config']!=audit['config'] or r['workspace_id']!=wid:
        raise ValueError('DROID receipt source/config changed')
    return r


def validate_stages(tail_path,tail,gs_path,gs,expected_workspace_ids,expected_cpu_source=None):
    """No fitting, evaluation, or policy execution; all failed/null slots survive."""
    tail_path=Path(tail_path);gs_path=Path(gs_path)
    if tail_path.name!='full_public_tail_completion_audit.json' or gs_path.name!='full_gaussian_completion_audit.json':
        raise ValueError('unexpected DROID completion source')
    tc=stage_contract(tail_path,tail,'e7_tail_config');gc=stage_contract(gs_path,gs,'e7_gaussian_config')
    if tc['workspace_ids']!=gc['workspace_ids'] or tc['workspace_ids']!=expected_workspace_ids:
        raise ValueError('DROID CPU/GS/tail workspaces must be identical and ordered')
    if expected_cpu_source is not None:
        cpu=gc['cpu_source']
        if (cpu['stage_root']!=expected_cpu_source['stage_root']
                or cpu['commit']!=expected_cpu_source['commit']
                or checked(cpu['contract'])!=Path(cpu['stage_root'])/'contract/freeze_manifest.json'):
            raise ValueError('Gaussian continuation uses another CPU source')
    if tail['qualification_or_policy_success'] is not None:
        raise ValueError('unmeasured policy success must remain null')
    src=tc['gaussian_source']
    if src['config']!=gs['config'] or src['commit']!=gs['source_commit'] or Path(src['stage_root'])!=gs_path.parent:
        raise ValueError('tail comes from another Gaussian stage')
    if checked(src['contract'])!=gs_path.parent/'contract/freeze_manifest.json':raise ValueError('tail Gaussian E0 differs')
    result=[]
    for tr,gr in zip(tail['rows'],gs['rows']):
        wid=tr['workspace_id'];gd=gs_path.parent/'gaussian'/wid;td=tail_path.parent/'public_tail'/wid
        prep=receipt(gr['preparation'],gd/'prepare_receipt.json',gs,wid)
        if gr['full_build'] is not False or gr['independent_image_fidelity'] is not False:
            raise ValueError('Gaussian row claims a later stage')
        train=None
        if gr['gaussian_status'] in {'PASS','FAIL'}:
            train=receipt(gr['training_receipt'],gd/'train_receipt.json',gs,wid)
            if train['status']!=gr['gaussian_status'] or not gr['train_fit_available']:
                raise ValueError('Gaussian status changed')
            if train['status']=='PASS':
                verify_tree(gd/'train',train['artifacts'])
                for name in ['scene.ply','train_report.json']:
                    if not (gd/'train'/name).is_file() or (gd/'train'/name).stat().st_size<=0:
                        raise ValueError('Gaussian canonical product missing')
                report=json.loads((gd/'train/train_report.json').read_text())
                validate_training_report(report,prep,30000)
                if report!=train['training_report'] or gr['n_gaussians']!=report['n_gaussians']:
                    raise ValueError('Gaussian training measurement differs')
        elif gr['gaussian_status']!='NOT_RUN' or gr.get('training_receipt') is not None or gr['train_fit_available'] is not False or prep['status']!='NOT_RUN':
            raise ValueError('Gaussian unavailable slot changed')
        for name,ref in tr['receipts'].items():
            if checked(ref)!=td/name:raise ValueError('tail phase receipt path differs')
        r=receipt(tr['receipts']['tail_receipt.json'],td/'tail_receipt.json',tail,wid)
        actual=tree(td);actual.pop('tail_receipt.json')
        if actual!=r['artifacts_without_receipt']:raise ValueError('tail artifacts changed')
        for k in ['status','discovered_instances','prepared_instances','jobs','full_build','paper_ready']:
            if tr[k]!=r[k]:raise ValueError('tail stage count or outcome changed')
        if r['status'] not in {'PASS','FAIL','NOT_RUN'} or r['full_build'] is not False or r['paper_ready'] is not False:
            raise ValueError('tail scope changed')
        if tr['reason']!=r.get('reason',r.get('error')):raise ValueError('tail failure reason changed')
        original=r['original_gaussian']
        if original['source_commit']!=gs['source_commit'] or original['contract_sha256']!=gs['contract_sha256'] or original['available']!=(gr['gaussian_status']=='PASS'):
            raise ValueError('tail original Gaussian availability changed')
        if original['available'] and original['training']!=gr['training_receipt']:
            raise ValueError('tail original Gaussian training changed')
        if r['status']=='PASS':
            if not original['available'] or tr['phase_executions']!=r['stages'] or [s['stage'] for s in r['stages']]!=['render','fuse','discover','prepare'] or any(s['returncode'] for s in r['stages']):
                raise ValueError('tail completed stage population changed')
            if type(r['discovered_instances']) is not int or type(r['prepared_instances']) is not int or not 0<=r['prepared_instances']<=r['discovered_instances']:
                raise ValueError('tail instance count invalid')
            if len(r['jobs'])!=r['discovered_instances'] or sum(j['prepared'] is True for j in r['jobs'])!=r['prepared_instances']:
                raise ValueError('tail job denominator differs')
        elif r['discovered_instances'] is not None or r['prepared_instances'] is not None or r['jobs'] is not None:
            raise ValueError('failed/unavailable instance counts must be null')
        result.append({**tr,'gaussian_status':gr['gaussian_status']})
    for key,status in [('processing_complete','PASS'),('processing_failed','FAIL'),('not_run','NOT_RUN')]:
        if tail[key]!=sum(r['status']==status for r in result):raise ValueError('tail terminal denominator differs')
    for key in ['discovered_instances','prepared_instances']:
        if tail[key]!=sum(r[key] or 0 for r in result):raise ValueError('tail aggregate count differs')
    if tail['null_instance_count_workspaces']!=sum(r['prepared_instances'] is None for r in result):raise ValueError('tail null count differs')
    for key,status in [('gaussian_completed','PASS'),('gaussian_failed','FAIL'),('gaussian_not_run','NOT_RUN')]:
        if gs[key]!=sum(r['gaussian_status']==status for r in result):raise ValueError('Gaussian terminal denominator differs')
    if gs['valid_train_fits']!=sum(r['train_fit_available'] for r in gs['rows']):raise ValueError('Gaussian TRAIN denominator differs')
    return {'scope':'droid_public_construction_engineering','rows':result,'paper_ready':False,'full_build':False}
