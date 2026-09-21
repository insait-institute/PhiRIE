"""Read-only original-source worker used by e4_planning_terminal's full audit.

This file is passed to the authenticated historical interpreter as source text;
its imports resolve in that interpreter's exact original repository checkout.
It never calls a constructor, simulator step, reset, or rollout producer.
"""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sys

COLLISION = 'room collision re-enters the protected object carve'
NO_TABLE = 'no tabletop cluster with graspable objects'


def audit(request, *, scene_ids=None):
    from robo.eval import e4_candidate_screen as screen
    from run.icra2027 import e4_full_qualification as full
    from run.icra2027 import e4_compact_materialization as api
    source=request['source']; stage=Path(source['e0']['path']).parent.parent
    config,protocol=full.validate_stage(source['config']['path'],stage,source['code_commit'])
    full.cpu_guard()
    if set(request['jobs']) != set(config['source']['scenes']):
        raise ValueError('terminal job roster differs from full source')
    selected=sorted(request['jobs']) if scene_ids is None else list(scene_ids)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(request['jobs']):
        raise ValueError('read-only diagnostic scene subset differs from original roster')
    root=screen.evidence_root(); scene_results={}; all_cells=[]
    def identity(path):return api.identity(path)
    for sid in selected:
        tasks=sorted(q['task_id'] for q in protocol['qualification_tasks'] if q['scene_id']==sid)
        common=dict(screen_id=stage.name,scene_id=sid,expected_commit=source['code_commit'])
        handoff=stage/'qualification_handoff'/sid
        evidence={}; measured=None; reason=None; status=None
        if handoff.exists():
            if request['jobs'][sid]['state'] != 'COMPLETED' or request['jobs'][sid]['exit_code'] != '0:0':
                raise ValueError('sealed handoff job was not completed successfully')
            bundle=screen._validate_bundle(handoff,root=root,expected_kind=full.SCOPE)
            expected_code={'code_root':source['code_root'],'commit':source['code_commit'],'dirty':False}
            if any(bundle['manifest'].get(k)!=v for k,v in {'code':expected_code,'freeze_id':stage.name,'scene_id':sid}.items()):
                raise ValueError('original handoff manifest source differs')
            prior=screen._read_json_member(handoff,'result.json')
            if (prior['scene_id']!=sid or prior['freeze_id']!=stage.name
                    or prior['source_commit']!=source['code_commit']
                    or prior['config']!=source['config'] or prior['selected_task_ids']!=tasks
                    or prior['planned_qualification_cells']!=len(tasks)*10
                    or prior.get('scope')!=full.SCOPE or prior.get('schema_version')!=1
                    or prior.get('policy_executed')!=0 or prior.get('paper_ready') is not False
                    or any(prior.get(k) is not None for k in ('policy_success','camera','rollout_ledger'))):
                raise ValueError('original terminal handoff source/task identities differ')
            evidence.update({str(handoff/n):identity(handoff/n) for n in ('result.json','manifest.json','seal.json')})
            status=prior['status']; reason=prior.get('reason')
            if status=='QUALIFICATION_COMPLETE':
                measured=screen._validate_qualifier_output(root=root,**common)
                full.validate_selected_cells(measured,tasks)
                qdir=stage/'scene_qualifiers'/sid
                if prior['qualifier']!=identity(qdir/'manifest.json'):
                    raise ValueError('original qualifier identity differs')
                evidence.update({str(qdir/n):identity(qdir/n) for n in ('gate.json','metrics.jsonl','manifest.json','seal.json')})
            elif status=='NOT_RUN':
                if tasks or reason!='no_queries_in_original_protocol':raise ValueError('false no-query terminal')
                status='SOURCE_NO_QUERIES'
            elif status=='PLANNING_UNQUALIFIED':
                if reason!=full.PLANNING_UNQUALIFIED:raise ValueError('unsupported original planning terminal')
                try:screen._automatic_prepare_inputs(root=root,**common)
                except screen.CandidateScreenError as exc:
                    if str(exc)!=reason:raise
                else:raise ValueError('claimed planning rejection does not replay')
                status='PLANNING_UNAVAILABLE'
            else:raise ValueError('unsupported original terminal status')
        else:
            job=request['jobs'][sid]
            if job['state']!='FAILED' or job['exit_code']!='1:0':
                raise ValueError('missing handoff is not a supported observed failure')
            if not tasks:raise ValueError('no-query source cannot be a construction rejection')
            for name in ('scene_qualifiers','scene_prepares','task_freezes'):
                if (stage/name/sid).exists():raise ValueError('rejected source already has downstream task/measurement artifacts')
            err=stage/'logs'/f"{sid}-{job['job_id']}.err"
            out=stage/'logs'/f"{sid}-{job['job_id']}.out"
            for path in (err,out):evidence[str(path)]=identity(path)
            last=err.read_text().splitlines()[-1]
            prefix='robo.eval.e4_candidate_screen.CandidateScreenError: '
            if not last.startswith(prefix):raise ValueError('unsupported original failure traceback')
            reason=last.removeprefix(prefix)
            d=stage/'automatic_candidates'/sid
            factories={p:d/'materialized'/p for p in screen.POLICIES}
            if reason==COLLISION:
                context=screen._automatic_export_context(factories,scene_id=sid,root=root)
                # Failed A0 is the first constructor arm. Do not invent an A4 export.
                factory=factories['A0'];report_path=factory/'sim_export/room_collision_report.json'
                report=api.read(report_path)
                package=screen._load_automatic_static_package(factory,factories,scene_id=sid,root=root)
                exclusion=report.get('collision_exclusion')
                if exclusion!=package['room_report'].get('collision_exclusion'):
                    raise ValueError('export rejection report differs from sealed static package')
                if (not isinstance(exclusion,dict) or not exclusion.get('primitive_intrusions')
                        or exclusion.get('unresolved_intrusion_count',0)<=0):
                    raise ValueError('collision rejection has no sealed primitive intrusion evidence')
                try:
                    screen.validate_full_room_export(factory,scene_id=sid,policy='A0',root=root,
                        expected_object_slots=context['rosters']['A0']['accepted_slots'],
                        expected_discovered_slots=context['object_slots'],automatic_factories=factories)
                except screen.CandidateScreenError as exc:
                    if str(exc)!=COLLISION:raise
                else:raise ValueError('claimed collision rejection does not replay')
                for path in [factory/'sim_export'/n for n in ('scene.xml','room_collision_report.json','mujoco_settle.json','isaac_manifest.json')]+[d/'materialized/shared_room_static'/n for n in ('manifest.json','seal.json','payload.json')]:
                    evidence[str(path)]=identity(path)
                status='EXPORT_REJECTED'
            elif reason==NO_TABLE:
                try:screen._automatic_prepare_inputs(root=root,**common)
                except screen.CandidateScreenError as exc:
                    if str(exc)!=NO_TABLE:raise
                else:raise ValueError('claimed no-tabletop rejection does not replay')
                status='PLANNING_UNAVAILABLE'
                for n in ('gate.json','manifest.json','seal.json'):
                    path=d/'population'/n;evidence[str(path)]=identity(path)
            else:raise ValueError('unsupported failure; no silent classification')
            for arm in screen.POLICIES:
                path=factories[arm]/'materialization_manifest.json';evidence[str(path)]=identity(path)
        if measured is not None:
            rows=measured['metric_rows']
            for row in rows:
                all_cells.append({'scene_id':sid,'task_id':row['task_id'],'policy_id':row['policy_id'],
                    'episode':row['episode'],'cell_id':row['cell_id'],
                    'qualification_state':'PASS' if row['passed'] else 'FAIL',
                    'source_kind':'canonical_qualifier','qualification_evidence':row,
                    'policy_executed':False,'policy_success':None})
            simulator_cells=measured['gate']['exact_900_step_cells']
            qualified_cells=sum(r['passed'] is True for r in rows)
            strict_tasks=measured['gate']['strict_pass_task_ids']
        else:
            for task in tasks:
                for arm in screen.POLICIES:
                    for ep in range(screen.EPISODES):
                        all_cells.append({'scene_id':sid,'task_id':task,'policy_id':arm,'episode':ep,
                            'cell_id':f'{arm.lower()}__{task}__seed{screen.BASE_SEED}__ep{ep}',
                            'qualification_state':'NOT_RUN','source_kind':status.lower(),
                            'qualification_evidence':None,'reset_definition':None,'camera':None,
                            'physics':None,'policy_executed':False,'policy_success':None})
            simulator_cells=0;qualified_cells=0;strict_tasks=[]
        for path,original in evidence.items():
            if identity(path)!=original:raise ValueError('terminal evidence changed during replay')
        scene_results[sid]={'status':status,'reason':reason,'selected_task_ids':tasks,
            'planned_cells':len(tasks)*10,'actual_900_step_cells':simulator_cells,
            'qualified_cells':qualified_cells,'strict_pass_task_ids':strict_tasks,
            'job':request['jobs'][sid],'evidence':evidence,'policy_executed':0,'policy_success':None}
    return {'source_config':config,'protocol':protocol,'scenes':scene_results,
            'cells':sorted(all_cells,key=lambda x:x['cell_id'])}


if __name__=='__main__':
    request=json.load(sys.stdin);scene_ids=request.pop('_scene_ids',None)
    with contextlib.redirect_stdout(sys.stderr):result=audit(request,scene_ids=scene_ids)
    print(json.dumps(result,sort_keys=True))
