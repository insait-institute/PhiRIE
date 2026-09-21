"""Bounded scheduler controller for already registered canonical engines only."""
import argparse
import json
import copy
import glob
import hashlib
from pathlib import Path
import subprocess
import time
from robo.roundtrip.matrix import save_new

FAILED={'FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL','CANCELLED','PREEMPTED'}
ACTIVE={'RUNNING','COMPLETING','CONFIGURING','SUSPENDED'}


def choose_releases(records,states,capacity):
    """Scheduling uses readiness/Slurm state only, never native outcomes."""
    if capacity<1:raise ValueError('positive eligible-engine capacity required')
    unknown=set(states)-set(records)
    if unknown:raise ValueError('scheduler state contains unregistered jobs')
    if set(records)-set(states):return [],'scheduler_state_unknown'
    if any(s['state'].split()[0].split('+')[0] in FAILED for s in states.values()):return [],'scheduler_failure'
    occupied=sum(s['state'] in ACTIVE or s['state']=='PENDING' and s.get('reason')!='JobHeldUser' for s in states.values())
    held=[j for j,s in states.items() if s['state']=='PENDING' and s.get('reason')=='JobHeldUser']
    held.sort(key=lambda j:(records[j].get('scheduler_priority',0),records[j]['executable_units']==10,int(j)))
    return held[:max(0,capacity-occupied)],None


def choose_group_releases(records, states, routing):
    """Separate hardware capacity, preserving the global failure boundary."""
    groups=routing['job_groups'];capacities=dict(routing['capacities'])
    if set(groups)!=set(records):raise ValueError('routing must bind every registered job exactly')
    if not capacities or any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in capacities.values()):
        raise ValueError('positive integer group capacities required')
    if set(groups.values())-set(capacities):raise ValueError('unknown routing group')
    _,blocked=choose_releases(records,states,1)
    if blocked:return [],blocked
    loan=routing.get('work_conserving_policy_pool');loan_pools=[]
    if loan:
        core,scope,prep=(loan[k] for k in ('core_group','scope_group','prep_group'))
        maximum=loan['total_capacity'];reserved=loan['scope_reservation']
        if (len({core,scope,prep})!=3 or not {core,scope,prep}<=set(capacities)
                or not isinstance(maximum,int) or maximum<2 or reserved!=1):
            raise ValueError('invalid work-conserving policy pool')
        demand=lambda group:sum(groups[j]==group and (s['state'] in ACTIVE or s['state']=='PENDING') for j,s in states.items())
        # A pending preparation job keeps its reserved slot. When no such job
        # exists, lend that idle GPU without interrupting work on its return.
        policy_capacity=max(1,maximum-min(capacities[prep],demand(prep)))
        capacities[core]=policy_capacity
        capacities[scope]=max(reserved,policy_capacity-min(demand(core),policy_capacity-reserved))
        loan_pools=[{'groups':[core,scope],'capacity':policy_capacity}]
    result=[]
    order=routing.get('group_order',sorted(capacities))
    if len(order)!=len(set(order)) or set(order)!=set(capacities):raise ValueError('resource group order must be complete and unique')
    for group in order:
        subset={j:r for j,r in records.items() if groups[j]==group}
        selected,_=choose_releases(subset,{j:states[j] for j in subset},capacities[group])
        result.extend(selected)
    for pool in loan_pools+routing.get('shared_capacity_pools',[]):
        members=set(pool['groups']);limit=pool['capacity']
        if not members<=set(capacities) or not isinstance(limit,int) or limit<1:raise ValueError('invalid shared capacity pool')
        occupied=sum((v['state'] in ACTIVE or v['state']=='PENDING' and v.get('reason')!='JobHeldUser')
                     for j,v in states.items() if groups[j] in members)
        retained=[];free=max(0,limit-occupied)
        for j in result:
            if groups[j] not in members:retained.append(j)
            elif free:retained.append(j);free-=1
        result=retained
    return result,None


def choose_hardware_refill(records,states,primary_ids,routing):
    """One whole held primary instance, only after the admitted group drains."""
    spec=routing.get('hardware_refill')
    if not spec:return None
    _,blocked=choose_releases(records,states,1)
    if blocked:return None
    groups=routing['job_groups'];target=spec['target_group'];source=spec['source_group']
    if target==source or target not in routing['capacities'] or source not in routing['capacities']:
        raise ValueError('invalid hardware refill groups')
    if any(groups[j]==target and (s['state'] in ACTIVE or s['state']=='PENDING') for j,s in states.items()):return None
    candidates=[j for j in primary_ids if groups[j]==source and states[j]=={'state':'PENDING','reason':'JobHeldUser'}]
    return min(candidates,key=int) if candidates else None


def refill_hardware(job,records,routing,current):
    """Preserve Slurm IDs and worker/config bytes; never migrate an engine."""
    from robo.roundtrip.local_policy_instance import preflight_engine
    spec=routing['hardware_refill'];identity=routing['original_submissions'][job]
    path=Path(identity['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=identity['sha256'] or json.loads(raw)!=records[job]:
        raise ValueError('refill original submission changed')
    gates={}
    for key in ('hardware_admission','same_device_identity'):
        gate=Path(routing[key]);digest=hashlib.sha256(gate.read_bytes()).hexdigest()
        if digest!=routing[key+'_sha256']:raise ValueError('admitted hardware gate changed')
        gates[key]={'path':str(gate),'sha256':digest}
    directory=path.parent
    for name,digest in spec['input_hashes'][job].items():
        if name not in {'planned_units.jsonl','commands.json','engine.sbatch'} or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen refill worker/config inputs changed')
    if set(spec['input_hashes'][job])!={'planned_units.jsonl','commands.json','engine.sbatch'}:
        raise ValueError('refill input identity incomplete')
    if (directory/'endpoint').exists():raise ValueError('started endpoint cannot migrate')
    units=[json.loads(line) for line in (directory/'planned_units.jsonl').read_text().splitlines()]
    commands=json.loads((directory/'commands.json').read_text())
    preflight_engine(units,commands,spec['worker_root'],['REF_NATIVE','B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE'],10)
    check=subprocess.run(['squeue','-j',job,'-h','-o','%T|%r'],capture_output=True,text=True,check=True)
    if check.stdout.strip()!='PENDING|JobHeldUser':return False
    before=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,check=True)
    if 'StartTime=' not in before.stdout or 'JobState=PENDING' not in before.stdout:
        raise ValueError('held job state changed before hardware amendment')
    cmd=['scontrol','update','JobId='+job,'Partition='+spec['partition'],'QOS='+spec['qos'],'Gres='+spec['gres'],'NodeList='+spec['node']]
    save_new(current/'hardware_refill_preflight.json',{'job_id':job,'source_group':spec['source_group'],'target_group':spec['target_group'],
        'submission':identity,'gates':gates,'planned_sha256':hashlib.sha256((directory/'planned_units.jsonl').read_bytes()).hexdigest(),
        'commands_sha256':hashlib.sha256((directory/'commands.json').read_bytes()).hexdigest(),'all50_units_unstarted':True,
        'before':before.stdout,'argv':cmd,'selection':'lowest original held primary job ID; no outcomes consulted'})
    result=subprocess.run(cmd,capture_output=True,text=True)
    after=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,check=True)
    save_new(current/'hardware_refill_result.json',{'job_id':job,'argv':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'after':after.stdout})
    if result.returncode:raise RuntimeError('pending hardware amendment rejected')
    if any(token not in after.stdout for token in ['JobState=PENDING','Partition='+spec['partition'],'QOS='+spec['qos'],'ReqNodeList='+spec['node'],spec['gres']]):
        raise RuntimeError('pending hardware amendment verification failed')
    routing['job_groups'][job]=spec['target_group']
    save_new(current/'routing_after_refill.json',routing)
    return True


def external_records(routing):
    records={};groups={};lineage={}
    def add(job,group,path,digest):
        job=str(job)
        if not job.isdigit():raise ValueError('ordinary numeric external job required')
        if job in groups and groups[job]!=group:raise ValueError('external job has conflicting resource groups')
        records[job]={'job_id':job,'executable_units':50,'scheduler_priority':routing.get('job_priority_overrides',{}).get(job,0)};groups[job]=group
        lineage.setdefault(job,[]).append({'path':path,'sha256':digest})
    for row in routing.get('external_jobs',[]):
        path=Path(row['receipt']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['receipt_sha256']:raise ValueError('external job authority receipt changed')
        add(row['job_id'],row['group'],str(path),row['receipt_sha256'])
    for source in routing.get('external_submission_globs',[]):
        for name in sorted(glob.glob(source['pattern'])):
            raw=Path(name).read_bytes()
            try:row=json.loads(raw)
            except json.JSONDecodeError:continue
            if not row.get('job_id'):continue
            command=row.get('command',row.get('argv',[]))
            if source.get('require_a6000_gres') and 'gpu:a6000:1' not in ' '.join(command):continue
            add(row['job_id'],source['group'],name,hashlib.sha256(raw).hexdigest())
    return records,groups,lineage


def resolved_scheduler_states(states,routing):
    """Manual receipt clears only scheduling blockage, never a failed result."""
    effective=copy.deepcopy(states);lineage={}
    for name in sorted(glob.glob(routing.get('manual_resolution_glob','/nonexistent/manual-resolution'))):
        raw=Path(name).read_bytes()
        try:receipt=json.loads(raw)
        except json.JSONDecodeError:continue
        job=str(receipt['job_id'])
        if job not in states:raise ValueError('manual resolution references unregistered job')
        evidence=Path(receipt['failure_evidence_path'])
        if (receipt.get('allow_other_jobs') is not True or not receipt.get('resolution')
                or receipt.get('classification') not in {'code_bug','environment_issue','resource_failure','missing_data','missing_checkpoint'}
                or hashlib.sha256(evidence.read_bytes()).hexdigest()!=receipt['failure_evidence_sha256']):
            raise ValueError('manual scheduler resolution lacks valid audited evidence')
        actual=states[job]['state'].split()[0].split('+')[0]
        if actual not in FAILED or actual!=receipt['scheduler_state']:raise ValueError('manual resolution does not match actual terminal failure')
        effective[job]={**states[job],'state':'RESOLVED_SCHEDULER_FAILURE'}
        lineage[job]={'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'original_state':states[job]}
    return effective,lineage


def dynamic_producers_done(routing):
    sources=routing.get('external_submission_globs',[]) if routing else []
    if not sources:return True
    receipt=routing.get('producer_completion_receipt')
    if not receipt or not Path(receipt).is_file():return False
    value=json.loads(Path(receipt).read_text())
    if value.get('completed') is not True or value.get('submission_patterns')!=[s['pattern'] for s in sources]:
        raise ValueError('producer completion must bind every dynamic submission source')
    return True


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    save_new(out/'configuration.json',vars(args))
    routing=json.loads(Path(args.routing).read_text()) if args.routing else None
    if routing:save_new(out/'routing.json',routing)
    for cycle in range(args.max_cycles):
        if (out/'STOP').exists():return 'stopped_by_owner'
        capacity=args.capacity
        if (out/'capacity.json').exists():capacity=int(json.loads((out/'capacity.json').read_text())['capacity'])
        records={}
        for root in args.instances:
            for p in Path(root).glob('*/submission.json'):
                r=json.loads(p.read_text());job=r.get('job_id')
                if job is None:continue
                if not str(job).isdigit():raise ValueError('ordinary numeric Slurm job ID required')
                if job in records and records[job]!=r:raise ValueError('conflicting scheduler receipt')
                records[job]=r
        primary_ids=set(records)
        effective_routing=copy.deepcopy(routing)
        external_lineage={}
        if routing:
            extras,extra_groups,external_lineage=external_records(routing)
            if set(extras)&primary_ids:raise ValueError('external registry overlaps primary experiment')
            records.update(extras);effective_routing['job_groups'].update(extra_groups)
        current=out/f'cycle_{cycle:05d}'
        if not records:time.sleep(args.interval);continue
        ids=','.join(records)
        q=subprocess.run(['squeue','-j',ids,'-h','-o','%i|%T|%r'],capture_output=True,text=True,check=True)
        states={}
        for line in q.stdout.splitlines():
            job,state,reason=line.split('|',2);states[job]={'state':state,'reason':reason}
        a=subprocess.run(['sacct','-X','-j',ids,'-P','-n','--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True)
        for line in a.stdout.splitlines():
            job,state,exitcode=line.split('|')[:3]
            if job in records and job not in states:states[job]={'state':state,'exit_code':exitcode}
        effective_states,resolutions=resolved_scheduler_states(states,routing) if routing else (states,{})
        if routing:
            refill=choose_hardware_refill(records,effective_states,primary_ids,effective_routing)
            if refill and refill_hardware(refill,records,routing,current):
                effective_routing['job_groups'][refill]=routing['job_groups'][refill]
        release,blocked=(choose_group_releases(records,effective_states,effective_routing) if routing else choose_releases(records,states,capacity))
        save_new(current/'observed.json',{'capacity':routing['capacities'] if routing else capacity,'registered_instances':len(primary_ids),'registered_external_jobs':len(records)-len(primary_ids),'external_lineage':external_lineage,'manual_resolutions':resolutions,'states':states,'proposed_release':release,'blocked':blocked})
        if blocked:return blocked
        for job in release:
            # Recheck immediately; never act on a running or unrelated job.
            check=subprocess.run(['squeue','-j',job,'-h','-o','%T|%r'],capture_output=True,text=True,check=True)
            if check.stdout.strip()!='PENDING|JobHeldUser':continue
            cmd=['scontrol','release',job];r=subprocess.run(cmd,capture_output=True,text=True)
            save_new(current/(job+'_release.json'),{'argv':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            if r.returncode:return 'scheduler_release_failed'
        if len(primary_ids)==args.expected_instances and all(j in effective_states and effective_states[j]['state'] in {'COMPLETED','RESOLVED_SCHEDULER_FAILURE'} for j in records):
            if not dynamic_producers_done(routing):
                save_new(current/'quiescent.json',{'current_registered_jobs_terminal':True,'dynamic_producers_complete':False,'action':'keep polling until bounded cycle limit or explicit complete producer receipt'})
                time.sleep(args.interval)
                continue
            save_new(out/'scheduler_complete.json',{'registered_instances':len(primary_ids),'registered_external_jobs':len(records)-len(primary_ids),'all_scheduler_jobs_terminal':True,'all_scheduler_jobs_completed':not bool(resolutions),'audited_scheduler_failures':resolutions,'scientific_claim_status':'requires canonical ledger collection; not inferred from Slurm'})
            return 'all_registered_jobs_terminal_with_audited_failures' if resolutions else 'all_registered_jobs_completed'
        time.sleep(args.interval)
    return 'bounded_cycle_limit'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--instances',action='append',required=True);p.add_argument('--out',required=True)
    p.add_argument('--routing',help='Immutable complete job-to-capacity-group manifest');p.add_argument('--capacity',type=int,default=3);p.add_argument('--expected-instances',type=int,default=48)
    p.add_argument('--max-cycles',type=int,default=1440);p.add_argument('--interval',type=float,default=60)
    args=p.parse_args(argv)
    if args.interval<10 or args.max_cycles<1:raise ValueError('bounded low-frequency queue controller required')
    result=run(args);save_new(Path(args.out)/'controller_terminal.json',{'status':result});print(result)

if __name__=='__main__':main()
