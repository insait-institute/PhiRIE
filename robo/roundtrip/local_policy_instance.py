"""One allocated GPU: unchanged warm policy server plus canonical native workers.

Legacy placement requires cross-service identity. The explicitly versioned
per-canonical engine protocol instead binds all paired arms to one process,
with same-process exact gates and a genuine two-arm DEV admission pilot.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
from robo.roundtrip.matrix import read_rows,run_instance,save_new,sha,canonical_hash,validate_terminal
from robo.manifest.hash import git_snapshot


def local_commands(commands,host,port):
    """Change transport address only, preserving every unit/config hash."""
    result=copy.deepcopy(commands)
    for command in result.values():
        argv=command['argv']
        for flag,value in [('--host',host),('--port',str(port))]:
            if argv.count(flag)!=1:raise ValueError('exactly one endpoint flag required')
            index=argv.index(flag)
            if index+1>=len(argv):raise ValueError('missing endpoint value')
            argv[index+1]=value
    return result


def validate_admission(path,metadata):
    receipt=json.loads(Path(path).read_text())
    if receipt.get('kind')!='local_policy_admission' or receipt.get('passed') is not True:
        raise ValueError('co-location admission missing or failed')
    for label in ('cross_service_gate','native_episode'):
        if sha(receipt[label])!=receipt[label+'_sha256']:raise ValueError('admission evidence changed')
    gate=json.loads(Path(receipt['cross_service_gate']).read_text())
    episode=json.loads(Path(receipt['native_episode']).read_text())
    if gate.get('passed') is not True:raise ValueError('cross-service chunk gate failed')
    if canonical_hash(metadata)!=receipt['policy_metadata_sha256']:raise ValueError('local policy identity changed')
    if episode.get('executed') is not True or not isinstance(episode.get('success'),bool) or episode.get('error') is not None:
        raise ValueError('actual native co-location pilot was not validly executed')
    return receipt


ENGINE_PROTOCOL='per_canonical_engine_v1'


def policy_definition(metadata):
    return canonical_hash({k:v for k,v in metadata.items() if k!='policy_engine'})


def validate_engine(metadata,instance_id,engine_id=None):
    engine=metadata.get('policy_engine',{})
    if engine.get('engine_protocol')!=ENGINE_PROTOCOL or engine.get('canonical_instance_id')!=instance_id:
        raise ValueError('policy engine protocol/canonical binding mismatch')
    if not engine.get('process_uuid') or not engine.get('runtime_fingerprint') or not engine.get('engine_id'):
        raise ValueError('policy engine process/runtime identity missing')
    if engine_id is not None and engine['engine_id']!=engine_id:raise ValueError('policy engine name mismatch')
    return engine


def preflight_engine(selected,commands,worker_root,methods,expected_resets,*,pilot=False):
    """A process cannot close before any later arm is ready, or reuse old REF."""
    expected_methods=['REF_NATIVE','B0_FIXED_NATIVE'] if pilot else ['REF_NATIVE','B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE']
    if methods == ['REF_NATIVE', 'B0_FIXED_NATIVE', 'B1_FIXED_PRIORITY', 'B2_EVIDENCE', 'B3_AGENT_NATIVE']:
        from robo.roundtrip.mechanism_followup import expected_engine_methods
        expected_methods = expected_engine_methods(selected, pilot=pilot)
    if methods!=expected_methods or set(methods)!={u['controller_method'] for u in selected}:
        raise ValueError('one engine requires the complete ordered canonical method roster')
    if pilot and (expected_resets!=1 or any(u['split']!='development' for u in selected)):
        raise ValueError('engine admission pilot must be fresh DEV two-arm one-reset')
    refs={str(Path(u['result_path_planned']).resolve()) for u in selected if u['controller_method']=='REF_NATIVE'}
    reset_sets=[]
    for method in methods:
        units=[u for u in selected if u['controller_method']==method]
        resets={u['reset_id'] for u in units};reset_sets.append(resets)
        if len(units)!=expected_resets or len(resets)!=expected_resets:raise ValueError('incomplete reset roster')
        for u in units:
            shard=Path(worker_root).resolve()/u['unit_id']
            if Path(u['result_path_planned']).resolve()!=shard/'runner/episode/result.json':raise ValueError('frozen worker path mismatch')
            config=json.loads(Path(u['config_path']).read_text())
            if canonical_hash(config)!=u['config_sha256']:raise ValueError('frozen unit config changed')
            if config.get('policy_engine_protocol')!=ENGINE_PROTOCOL:raise ValueError('unit does not declare engine protocol')
            terminal=shard/'terminal.json'
            if terminal.exists():
                t=validate_terminal(u,json.loads(terminal.read_text()))
                if t['terminal_status'] not in ('BUILD_FAILED','ABSTAINED'):
                    raise ValueError('cannot resume executed or failed runtime units in a new policy process')
                continue
            if shard.exists():raise ValueError('partial runtime shard cannot resume across policy processes')
            command=commands.get(u['unit_id'])
            if command is None or command.get('planned_unit_sha256')!=canonical_hash(u):raise ValueError('all canonical builds must be resolved before engine starts')
            if any(not Path(f).is_file() and str(Path(f).resolve()) not in refs for f in command.get('requires_files',[])):
                raise ValueError('constructor artifact missing before engine starts')
    if any(r!=reset_sets[0] for r in reset_sets):raise ValueError('paired reset rosters differ')


SCOPE_BLOCKS={
    'l1_b3_v1':[('REF_NATIVE','L0_target_only'),('B3_AGENT_NATIVE','L0_target_only'),('B3_AGENT_NATIVE','L1_target_destination')],
    'l1_b3_b4_v1':[('REF_NATIVE','L0_target_only'),('B3_AGENT_NATIVE','L0_target_only'),('B4_ROOM_REPAIR_NATIVE','L0_target_only'),('B3_AGENT_NATIVE','L1_target_destination'),('B4_ROOM_REPAIR_NATIVE','L1_target_destination')]}


def preflight_scope_engine(selected,commands,worker_root,protocol,expected_resets):
    if protocol not in SCOPE_BLOCKS:raise ValueError('unknown separately declared scope engine protocol')
    blocks=SCOPE_BLOCKS[protocol]
    if {(u['controller_method'],u['scope']) for u in selected}!=set(blocks):raise ValueError('incomplete scope/control block roster')
    results={str(Path(u['result_path_planned']).resolve()) for u in selected}
    resets=[];run_ids=set()
    for method,scope in blocks:
        units=[u for u in selected if (u['controller_method'],u['scope'])==(method,scope)]
        ids={u['reset_id'] for u in units};resets.append(ids)
        if len(units)!=expected_resets or len(ids)!=expected_resets:raise ValueError('scope reset denominator differs')
        for unit in units:
            shard=Path(worker_root).resolve()/unit['unit_id'];config=json.loads(Path(unit['config_path']).read_text())
            if Path(unit['result_path_planned']).resolve()!=shard/'runner/episode/result.json':raise ValueError('scope result path differs')
            if canonical_hash(config)!=unit['config_sha256'] or config.get('policy_engine_protocol')!=ENGINE_PROTOCOL:
                raise ValueError('scope config or engine identity differs')
            if config.get('scope_engine_protocol')!=protocol or not config.get('scope_run_id'):raise ValueError('scope experiment identity missing')
            run_ids.add(config['scope_run_id'])
            terminal=shard/'terminal.json'
            if terminal.exists():
                record=validate_terminal(unit,json.loads(terminal.read_text()))
                if record['terminal_status'] not in ('BUILD_FAILED','ABSTAINED'):raise ValueError('cannot reuse executed scope controls across processes')
                continue
            if shard.exists():raise ValueError('partial scope runtime cannot resume across processes')
            for path,digest in unit.get('scope_input_hashes',{}).items():
                if sha(path)!=digest:raise ValueError('scope constructor input changed before process startup')
            command=commands.get(unit['unit_id'])
            if command is None or command.get('planned_unit_sha256')!=canonical_hash(unit):raise ValueError('scope build or command unresolved')
            for path,digest in command.get('scope_input_hashes',{}).items():
                if sha(path)!=digest:raise ValueError('scope command artifact changed before process startup')
            if any(not Path(p).is_file() and str(Path(p).resolve()) not in results for p in command.get('requires_files',[])):
                raise ValueError('scope dependency unavailable before engine startup')
    if len(run_ids)!=1 or any(r!=resets[0] for r in resets):raise ValueError('scope blocks differ in experiment/reset identity')
    return blocks


def engine_pilot_receipt(metadata,endpoint,gates,selected,worker_root):
    records={}
    for u in selected:
        terminal=validate_terminal(u,json.loads((Path(worker_root)/u['unit_id']/'terminal.json').read_text()))
        result=terminal.get('result',{})
        if terminal['terminal_status']!='RECORDED' or result.get('error') is not None:
            raise ValueError('engine pilot requires two actually executed native episodes')
        if result.get('policy_engine')!=metadata['policy_engine']:raise ValueError('pilot episode used a different policy process')
        records[u['controller_method']]=terminal['result_path']
    if set(records)!={'REF_NATIVE','B0_FIXED_NATIVE'}:raise ValueError('engine pilot requires REF and B0')
    evidence={'endpoint':str(endpoint),**gates,'reference_episode':records['REF_NATIVE'],'comparison_episode':records['B0_FIXED_NATIVE']}
    return {'kind':'per_canonical_engine_admission','schema_version':1,'passed':True,'engine_protocol':ENGINE_PROTOCOL,
        'policy_definition_sha256':policy_definition(metadata),'pilot_engine':metadata['policy_engine'],
        'evidence':{k:{'path':str(p),'sha256':sha(p)} for k,p in evidence.items()},
        'cross_process_exact_gate':'FAIL; no cross-engine reference reuse is permitted'}


def validate_engine_admission(path,metadata):
    receipt=json.loads(Path(path).read_text())
    if receipt.get('kind')!='per_canonical_engine_admission' or receipt.get('passed') is not True or receipt.get('engine_protocol')!=ENGINE_PROTOCOL:
        raise ValueError('per-canonical engine admission missing or failed')
    if receipt.get('policy_definition_sha256')!=policy_definition(metadata):raise ValueError('frozen policy definition changed')
    evidence=receipt['evidence'];loaded={}
    for key in ('endpoint','interleaving','full_chunks','reference_episode','comparison_episode'):
        record=evidence[key]
        if sha(record['path'])!=record['sha256']:raise ValueError('engine admission evidence changed')
        loaded[key]=json.loads(Path(record['path']).read_text())
    engine=receipt['pilot_engine']
    if loaded['endpoint']['policy_metadata']['policy_engine']!=engine:raise ValueError('pilot endpoint engine mismatch')
    pilot_runtime=engine.get('runtime',{});current_runtime=metadata['policy_engine'].get('runtime',{})
    for key in ('source_hashes','python','packages'):
        if not pilot_runtime.get(key) or pilot_runtime[key]!=current_runtime.get(key):raise ValueError('engine policy source or package runtime changed: '+key)
    def stable_env(runtime):return {k:v for k,v in runtime.get('environment',{}).items() if k!='CUDA_VISIBLE_DEVICES'}
    if stable_env(pilot_runtime)!=stable_env(current_runtime):raise ValueError('engine inference environment changed')
    for key in ('interleaving','full_chunks'):
        gate=loaded[key]
        if gate.get('passed') is not True or gate.get('metadata',{}).get('policy_engine')!=engine:raise ValueError('same-process exact gate failed')
    full=loaded['full_chunks']
    if full.get('repeat_reference') is not True or full.get('model_chunk_requests')!=18 or full.get('full_actions_per_chunk')!=50 or full.get('endpoints',[None,None])[0]!=full.get('endpoints',[None,None])[1]:
        raise ValueError('full-chunk gate must repeat within the same endpoint')
    for key in ('reference_episode','comparison_episode'):
        episode=loaded[key]
        if episode.get('executed') is not True or not isinstance(episode.get('success'),bool) or episode.get('error') is not None or episode.get('policy_engine')!=engine:
            raise ValueError('pilot native pair was not executed within one process')
    return receipt


def owns_listening_port(pid,port):
    """Ensure readiness belongs to our server process, not another job's port."""
    try:
        inodes={line.split()[9] for line in Path('/proc/net/tcp').read_text().splitlines()[1:]
                if int(line.split()[1].split(':')[1],16)==port and line.split()[3]=='0A'}
        return any(os.readlink(fd)==f'socket:[{inode}]' for fd in Path(f'/proc/{pid}/fd').iterdir()
                   for inode in inodes)
    except (OSError,ValueError):return False


def execute(args):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('an explicit Slurm GPU allocation is required')
    engine_mode=args.engine_protocol==ENGINE_PROTOCOL
    scope_protocol=getattr(args,'scope_protocol',None)
    if scope_protocol and (not engine_mode or args.engine_pilot or not args.admission):raise ValueError('scope requires independently admitted same-process engine')
    if engine_mode:
        if args.reference_host or bool(args.engine_pilot)==bool(args.admission):raise ValueError('choose fresh DEV engine pilot or engine admission')
    elif bool(args.reference_host)==bool(args.admission):raise ValueError('choose comparison pilot or validated admission')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    rows=read_rows(args.planned);selected=[u for u in rows if u['canonical_instance_id']==args.instance_id]
    if not selected:raise ValueError('instance is outside frozen roster')
    methods=args.method or list(dict.fromkeys(u['controller_method'] for u in selected))
    if len(methods)!=len(set(methods)):raise ValueError('duplicate methods')
    if 'REF_NATIVE' in methods and methods[0]!='REF_NATIVE':raise ValueError('reference must execute before dependent methods')
    if any(m not in {u['controller_method'] for u in selected} for m in methods):raise ValueError('method outside frozen roster')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    commands=local_commands(json.loads(Path(args.commands).read_text()),'127.0.0.1',port)
    blocks=None
    if scope_protocol:
        blocks=preflight_scope_engine(selected,commands,args.worker_root,scope_protocol,args.expected_resets)
        if methods!=list(dict.fromkeys(m for m,_ in blocks)):raise ValueError('scope engine method order differs from admitted blocks')
    elif engine_mode:preflight_engine(selected,commands,args.worker_root,methods,args.expected_resets,pilot=args.engine_pilot)
    save_new(out/'runtime_commands.json',commands)
    receipt={'schema_version':1,'kind':'local_policy_instance_endpoint','source_code':git_snapshot(),
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'host':socket.gethostname(),'endpoint_host':'127.0.0.1','endpoint_port':port,
        'canonical_instance_id':args.instance_id,'controller_methods':methods,
        'planned_units_sha256':sha(args.planned),'original_commands_sha256':sha(args.commands),
        'runtime_commands_sha256':sha(out/'runtime_commands.json'),'treatment_change':'none; transport address only'}
    server_argv=['bash','run/sim_recon_sim/serve_policy.sh','--host','127.0.0.1','--port',str(port),
                 '--metadata-out',str(out/'server_metadata.json')]
    engine_id='canonical-engine-'+uuid.uuid4().hex if engine_mode else None
    if engine_mode:
        server_argv+=['--engine-id',engine_id,'--engine-protocol',ENGINE_PROTOCOL,'--canonical-instance-id',args.instance_id]
        receipt.update(engine_protocol=ENGINE_PROTOCOL,engine_id=engine_id,treatment_change='versioned per-canonical runtime process binding; same process for all paired arms')
    receipt['server_argv']=server_argv;start=time.monotonic();server=None
    try:
        with (out/'server.log').open('x') as log:
            server=subprocess.Popen(server_argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while not owns_listening_port(server.pid,port):
                if server.poll() is not None:raise RuntimeError('local policy server exited before readiness')
                if time.monotonic()-start>args.startup_timeout:raise TimeoutError('local server startup timeout')
                time.sleep(.5)
            from robo.roundtrip.native_policy import NativePolicy
            policy=NativePolicy('127.0.0.1',port)
            try:metadata=policy.metadata
            finally:policy.close()
            receipt['policy_metadata']=metadata;receipt['policy_metadata_sha256']=canonical_hash(metadata)
            gates={}
            if engine_mode:
                validate_engine(metadata,args.instance_id,engine_id)
                if args.admission:validate_engine_admission(args.admission,metadata)
                for label in ('interleaving','full_chunks'):
                    gate=out/(label+'.json')
                    cmd=[sys.executable,'-m','robo.roundtrip.policy_interleaving','--host','127.0.0.1','--port',str(port),'--out',str(gate)]
                    if label=='full_chunks':cmd+=['--compare-host','127.0.0.1','--compare-port',str(port),'--repeat-reference','--artifact-dir',str(out/'full_chunk_arrays')]
                    subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
                    data=json.loads(gate.read_text())
                    if data.get('passed') is not True or data.get('metadata',{}).get('policy_engine')!=metadata['policy_engine']:raise ValueError('same-process identity gate failed')
                    gates[label]=str(gate)
                receipt['same_process_gates']={k:{'path':p,'sha256':sha(p)} for k,p in gates.items()}
                if args.admission:receipt.update(admission_path=str(Path(args.admission).resolve()),admission_sha256=sha(args.admission))
            elif args.reference_host:
                gate=out/'cross_service_gate.json'
                cmd=[sys.executable,'-m','robo.roundtrip.policy_interleaving','--host',args.reference_host,
                     '--port',str(args.reference_port),'--compare-host','127.0.0.1','--compare-port',str(port),'--out',str(gate)]
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
                if json.loads(gate.read_text()).get('passed') is not True:raise ValueError('cross-service gate failed')
                receipt['cross_service_gate']=str(gate);receipt['cross_service_gate_sha256']=sha(gate)
            else:
                validate_admission(args.admission,metadata);receipt['admission_path']=str(Path(args.admission).resolve());receipt['admission_sha256']=sha(args.admission)
            save_new(out/'endpoint_receipt.json',receipt)
            reports=[]
            for method,scope in (blocks or [(m,None) for m in methods]):
                if server.poll() is not None:raise RuntimeError('local policy server died during native execution')
                kwargs={'scope':scope} if scope is not None else {}
                reports.append(run_instance(rows,commands,instance_id=args.instance_id,method=method,
                    out=args.worker_root,expected_resets=args.expected_resets,**kwargs))
            if engine_mode and any(r['pending_unit_ids'] for r in reports):raise ValueError('canonical engine ended with pending paired units')
            if engine_mode and args.engine_pilot:
                admission=engine_pilot_receipt(metadata,out/'endpoint_receipt.json',gates,selected,args.worker_root)
                save_new(out/'engine_admission.json',admission)
                validate_engine_admission(out/'engine_admission.json',metadata)
            save_new(out/'instance_reports.json',{'reports':reports,'wall_s':time.monotonic()-start})
            return reports
    except Exception as exc:
        save_new(out/'failure.json',{'type':type(exc).__name__,'reason':str(exc),'wall_s':time.monotonic()-start})
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('planned','commands','instance-id','worker-root','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--method',action='append');p.add_argument('--expected-resets',type=int,default=10)
    p.add_argument('--reference-host');p.add_argument('--reference-port',type=int,default=8017)
    p.add_argument('--engine-protocol',choices=['cross_service_exact_v1',ENGINE_PROTOCOL],default='cross_service_exact_v1')
    p.add_argument('--engine-pilot',action='store_true')
    p.add_argument('--scope-protocol',choices=list(SCOPE_BLOCKS))
    p.add_argument('--admission');p.add_argument('--startup-timeout',type=float,default=300)
    execute(p.parse_args(argv));return 0

if __name__=='__main__':raise SystemExit(main())
