import csv
import json
from pathlib import Path
import pytest
import yaml
from robo.eval import paper_task_support_closure as d


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True)+'\n')
    return identity(path)


def identity(path):
    return {'path': str(path), 'sha256': d.canonical._sha256(path), 'size_bytes': path.stat().st_size}


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@pytest.fixture
def closure(tmp_path, monkeypatch):
    stage=tmp_path/'closure';full=tmp_path/'features';fd=full/'audit/features';out=stage/'audit/validity_closure'
    code=Path(d.__file__).resolve().parents[2]
    source={'commit': d.PRODUCER, 'dirty': False, 'branch': 'fixture'}
    monkeypatch.setattr(d, 'git_snapshot', lambda _: source)
    monkeypatch.setattr(d, '_public_sources', lambda cfg, rows: d.require(len(rows)==72, 'public roster'))
    protocol=yaml.safe_load((code/'configs/experiments/icra2027/audit_validity_protocol.yaml').read_text())
    protocol_ref=write(tmp_path/'protocol.json',protocol)
    rows=[];records=[];labels=[]
    for s in range(6):
        for c in ['clean','mild','severe']:
            for q in range(4):
                row=dict(freeze_id=full.name,scene_id=f'scene{s}',task_id=f'q{q}',condition_id=c,
                         task_family='place',scene_build_failed=str(int(len(rows)<16)),
                         robot_frame_missing='1',geometry_policy_cameras_missing='1')
                record={k:row[k] for k in d.canonical.JOIN_KEY}
                record.update(dict.fromkeys(list(protocol['thresholds'])+protocol['required_booleans']+list(d.BASELINES)))
                record.update(measurement_status='NOT_RUN',measurement_scope='all_required_task_invariants')
                reasons=d.audit_labels.label_record(record,d.audit_labels.ValidityThresholds(**protocol['thresholds']))['invalid_reasons']
                if int(row['scene_build_failed']):reasons.append('construction_failed')
                reasons+=['construction_robot_frame_missing','construction_policy_cameras_missing','robot_alignment_failed_or_missing']
                labels.append({**{k:row[k] for k in d.canonical.JOIN_KEY},'invalid_label':'1','invalid_reasons':';'.join(reasons)})
                rows.append(row);records.append(record)
    table(fd/'features_unlabeled.csv',rows)
    resolved=write(fd/'config_resolved.json',{'tier':'full'})
    seal=write(fd/'feature_seal.json',dict(source_kind='real',query_rows=72,feature_generation_read_labels=False,
       paper_ready=False,driver_sha256=d.canonical._sha256(Path(d.canonical.__file__)),evaluation_baseline_columns=list(d.BASELINES),
       members={'features_unlabeled.csv':identity(fd/'features_unlabeled.csv')['sha256'],'config_resolved.json':resolved['sha256']},
       config_sha256=resolved['sha256'],freeze_id=full.name,label_protocol_sha256=protocol_ref['sha256'],source_refs=[]))
    full_cfg=write(full/'execution.json',{'source_commit':d.PRODUCER,'tier':'full'})
    full_qa=write(full/'full_independent_qa.json',{'source':source,'status':'PASS','feature_seal':seal,'config':full_cfg})
    measurement=write(stage/'measurements_not_run.json',dict(rows=records,feature_seal_sha256=seal['sha256'],freeze_id=full.name,
        measurement_status='NOT_RUN',reference_data_opened=False,rollouts_invoked=False,baseline_measurements_invoked=False))
    table(out/'labels.csv',labels)
    middle=[{**r,'invalid_label':l['invalid_label'],'invalid_reasons':l['invalid_reasons']} for r,l in zip(rows,labels)]
    table(out/'features_with_validity.csv',middle)
    table(out/'task_local_features_and_labels.csv',[{**r,**dict.fromkeys(d.BASELINES,'')} for r in middle])
    members={n:identity(out/n)['sha256'] for n in d.MEMBERS}
    folds=[{'heldout_scene':f'scene{s}','training_has_both_labels':False,'heldout_has_both_labels':False} for s in range(6)]
    manifest=write(out/'label_join_manifest.json',dict(members=members,protocol_sha256=protocol_ref['sha256'],
       feature_seal_sha256=seal['sha256'],measurement_sha256=measurement['sha256'],freeze_id=full.name,
       fold_health=folds,loso_admissible=False,paper_ready=False,query_rows=72))
    cfg=write(stage/'execution.json',dict(source_commit=d.PRODUCER,freeze_id=stage.name,predictions_allowed=False,
       paper_predictive_claim=False,feature_seal=seal,measurements=measurement,feature_data_freeze_id=full.name,
       protocol=protocol_ref,expected_returncode=2,command=['canonical','--stage','join']))
    def e0(root,config):
        m=dict(code={'repository':str(code),**source},freeze_id=root.name,configs=[],
               resource_inventory=[dict(resolved_path=config['path'],sha256=config['sha256'],hash_method='content_sha256')])
        m['contract_sha256']=d.canonical_hash(m)
        return write(root/'contract/freeze_manifest.json',m),m['contract_sha256']
    e0(full,full_cfg);contract,digest=e0(stage,cfg)
    (stage/'stderr.log').write_text('LOSO blocked: every training/test fold requires both labels')
    (stage/'stdout.log').write_text('')
    execution=write(stage/'join_execution_receipt.json',dict(source=source,cwd=str(code),contract_sha256=digest,
        config=cfg,returncode=2,command=['canonical','--stage','join'],stderr=identity(stage/'stderr.log'),stdout=identity(stage/'stdout.log')))
    qa=dict(schema_version=1,status='PASS',scope='artifact_integrity_and_missing_evidence_validity_closure',source=source,
      paper_ready=False,reference_data_opened=False,rollouts_invoked=False,predictions_generated=False,
      predictive_claim_gate='FAIL',loso_status='NOT_RUN',reference_measurement_status='NOT_RUN',baseline_measurement_status='NOT_RUN',
      metrics=dict.fromkeys(d.METRICS),execution_config=cfg,contract=contract,execution_receipt=execution,
      feature_seal=seal,full_feature_qa=full_qa,measurements=measurement,label_join_manifest=manifest,
      qa_script=identity(Path(__file__).resolve()),freeze_id=stage.name,feature_data_freeze_id=full.name,members=members,
      fold_health=folds,planned_conditions=18,planned_queries=72,invalid_queries=72,valid_queries=0,
      failed_constructor_query_rows=16,missing_robot_frames=72,missing_policy_cameras=72)
    path=stage/'validity_independent_qa.json';write(path,qa)
    return path,qa,fd,out


def validate(path):
    return d.validate_closure(path,expected_sha256=identity(path)['sha256'])


def test_complete_negative_scientific_result(closure):
    p,_,_,_=closure;result=validate(p)
    assert result['rows']==[dict(queries=72,constructor_failed_queries=16,invalid_queries=72,estimable_folds=0,planned_folds=6)]
    assert result['claim_gate']=='FAIL' and result['loso_status']=='NOT_RUN'
    assert all(v is None for v in result['metrics'].values())


@pytest.mark.parametrize('flag,status', [('0', 'complete'), ('1', 'failed')])
def test_failure_count_binds_to_authenticated_build(flag, status):
    d.validate_build_flag({'scene_build_failed': flag}, {'build_status': status})


@pytest.mark.parametrize('flag,status', [('0', 'failed'), ('1', 'complete'),
    ('0.5', 'complete'), ('1.5', 'failed'), ('-1', 'failed'), ('nan', 'complete')])
def test_forged_or_nonbinary_failure_flag_is_rejected(flag, status):
    with pytest.raises(ValueError, match='constructor failure flag'):
        d.validate_build_flag({'scene_build_failed': flag}, {'build_status': status})


@pytest.mark.parametrize('field,value', [('planned_queries',71),('failed_constructor_query_rows',0),
    ('invalid_queries',0),('paper_ready',True),('predictive_claim_gate','PASS'),('predictions_generated',True),
    ('metrics',{'AUROC':1}),('reference_measurement_status','PASS')])
def test_changed_count_or_claim(closure,field,value):
    p,qa,_,_=closure;qa[field]=value;write(p,qa)
    with pytest.raises(ValueError):validate(p)


@pytest.mark.parametrize('which',['label','seal','source','fold','measurements','feature'])
def test_changed_evidence_even_with_new_entrypoint_hash(closure,which):
    p,qa,fd,out=closure
    if which=='label':target=out/'labels.csv'
    elif which=='seal':target=fd/'feature_seal.json'
    elif which=='feature':target=fd/'features_unlabeled.csv'
    elif which=='measurements':target=Path(qa['measurements']['path'])
    elif which=='fold':
        qa['fold_health'][0]['heldout_has_both_labels']=True;write(p,qa)
        with pytest.raises(ValueError):validate(p)
        return
    else:
        target=Path(qa['contract']['path']);m=d.read(target);m['code']['commit']='0'*40
        m['contract_sha256']=d.canonical_hash({k:v for k,v in m.items() if k!='contract_sha256'})
        qa['contract']=write(target,m);write(p,qa)
        with pytest.raises(ValueError,match='source/E0'):validate(p)
        return
    target.write_text(target.read_text()+' ')
    with pytest.raises(ValueError):validate(p)


@pytest.mark.parametrize('location',['closure','predictions','table'])
def test_added_prediction_or_metric_artifact(closure,location):
    p,_,_,out=closure
    target=out/'heldout_predictions.csv' if location=='closure' else p.parent/'audit'/location/'artifact.json'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{}')
    with pytest.raises(ValueError,match='prediction/metric'):validate(p)


def test_external_hash_is_required(closure):
    with pytest.raises(ValueError,match='bytes changed'):
        d.validate_closure(closure[0],expected_sha256='0'*64)


def test_symlink_alias_rejected(closure):
    p=closure[0];alias=p.parent/'alias.json';alias.symlink_to(p)
    with pytest.raises(ValueError,match='alias'):
        d.validate_closure(alias,expected_sha256=identity(p)['sha256'])


def test_rehashed_label_cannot_promote_valid_query(closure):
    p,qa,_,out=closure
    labels=d.canonical._read_csv(out/'labels.csv');labels[0]['invalid_label']='0';table(out/'labels.csv',labels)
    m=d.read(out/'label_join_manifest.json');m['members']['labels.csv']=identity(out/'labels.csv')['sha256']
    qa['label_join_manifest']=write(out/'label_join_manifest.json',m);qa['members']=m['members'];write(p,qa)
    with pytest.raises(ValueError,match='canonical missing-evidence label'):validate(p)


def test_rehashed_fold_health_cannot_claim_estimable(closure):
    p,qa,_,out=closure;m=d.read(out/'label_join_manifest.json')
    m['fold_health'][0]['training_has_both_labels']=True;qa['fold_health']=m['fold_health']
    qa['label_join_manifest']=write(out/'label_join_manifest.json',m);write(p,qa)
    with pytest.raises(ValueError,match='fold health'):validate(p)


def test_unsealed_feature_prediction_rejected(closure):
    p,_,fd,_=closure;(fd/'heldout_predictions.csv').write_text('fake')
    with pytest.raises(ValueError,match='unsealed feature artifact'):validate(p)


@pytest.mark.parametrize('mutation',['omit','reorder','mixed'])
def test_public_source_roster_cannot_change(tmp_path,monkeypatch,mutation):
    refs=[];rows=[]
    for scene in d.bridge.shared.SCENES:
        for condition in d.bridge.shared.CONDITIONS:
            refs.append({'producer':{'path':'original','commit':'original'},'config':{'path':'one'},
                'gate':{'scene_id':scene,'condition_id':condition,'original_gate':{'queries':[]}}})
    if mutation=='omit':refs.pop()
    if mutation=='reorder':refs.reverse()
    if mutation=='mixed':refs[-1]['config']={'path':'other'}
    monkeypatch.setattr(d.bridge,'_reference_context',lambda ref:(None,None,None,ref['gate'],repr(ref)))
    with pytest.raises(ValueError,match='roster|mixed'):
        d._public_sources({'source_commit':d.PRODUCER,'tier':'full','references':refs},rows)
