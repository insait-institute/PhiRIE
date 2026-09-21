"""Bounded same-engine REF/L0-B3/L1-B3 DEV launcher, canonical ledgers only."""
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
from robo.roundtrip.local_policy_instance import owns_listening_port,validate_engine,validate_engine_admission
from robo.roundtrip.scope_bundle import selected_destination,seal_bundle
from robo.roundtrip.identity import file_hash
from robo.roundtrip.spec import validate_spec


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('scope pilot requires an allocated GPU')
    config=json.loads(Path(args.baseline_config).read_text());validate_spec(config)
    if config.get('scope')!='L0_target_only' or config.get('controller_method')!='B3_AGENT_NATIVE' or config['instance']['task_id'] not in ('PickPlaceSinkToCounter','PickPlaceCounterToSink','PickPlaceCounterToCabinet'):
        raise ValueError('bounded scope pilot requires fixed supported native L0 B3 config')
    workspace=getattr(args,'workspace_engineering_config',None)
    build_path=None if workspace else Path(args.destination_build_manifest)
    build={} if workspace else json.loads(build_path.read_text())
    if workspace:
        engineering=json.loads(Path(workspace).read_text())
        if config['instance']['split']!='development' or engineering['tier']!='DEV' or engineering['canonical_instance_id']!=config['canonical_instance_id']:
            raise ValueError('workspace pilot must match the frozen DEV engineering canonical')
        for path,digest in engineering['input_files'].items():
            if file_hash(path)!=digest:raise ValueError('workspace pilot input closure differs')
        if Path(args.target_dir).resolve()!=Path(engineering['entities'][0]['object_dir']).resolve():raise ValueError('workspace L0 target differs')
        if args.destination_candidate_pool is not None or args.destination_rvg_receipt is not None:raise ValueError('workspace uses already sealed role artifacts')
    elif build.get('canonical_instance_id')!=config['canonical_instance_id'] or build.get('object_role') not in ('receptacle','support'):
        raise ValueError('scope pilot destination differs from canonical role')
    if args.destination_candidate_pool is not None:
        selected_destination(args.destination_candidate_pool,build_path,build,args.destination_rvg_receipt)
    elif args.destination_rvg_receipt is not None:
        raise ValueError('RVG receipt requires destination pool')
    l1_label='L1_B3' if args.destination_candidate_pool is not None else 'L1_HYBRID'
    if workspace:l1_label='DEV_PARTIAL_WORKSPACE'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    engine_id='scope-engine-'+uuid.uuid4().hex
    config.update(policy_engine_protocol='per_canonical_engine_v1',policy_engine_id=engine_id)
    configs={}
    for label,method in [('REF','REF_NATIVE'),('L0_B3','B3_AGENT_NATIVE')]:
        c=copy.deepcopy(config);c['controller_method']=method;p=out/(label+'.json');write_new(p,c);configs[label]=p
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    argv=['bash','run/sim_recon_sim/serve_policy.sh','--host','127.0.0.1','--port',str(port),
        '--metadata-out',str(out/'server_metadata.json'),'--engine-id',engine_id,
        '--engine-protocol','per_canonical_engine_v1','--canonical-instance-id',config['canonical_instance_id']]
    server=None;start=time.monotonic();rows=[]
    try:
        with (out/'server.log').open('x') as log:
            server=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while not owns_listening_port(server.pid,port):
                if server.poll() is not None:raise RuntimeError('scope policy server exited before readiness')
                if time.monotonic()-start>300:raise TimeoutError('scope policy server startup timeout')
                time.sleep(.5)
            from robo.roundtrip.native_policy import NativePolicy
            policy=NativePolicy('127.0.0.1',port)
            try:metadata=policy.metadata
            finally:policy.close()
            validate_engine(metadata,config['canonical_instance_id'],engine_id)
            validate_engine_admission(args.admission,metadata)
            for label in ['interleaving','full_chunks']:
                command=[sys.executable,'-m','robo.roundtrip.policy_interleaving','--host','127.0.0.1','--port',str(port),'--out',str(out/(label+'.json'))]
                if label=='full_chunks':command+=['--compare-host','127.0.0.1','--compare-port',str(port),'--repeat-reference','--artifact-dir',str(out/'chunk_arrays')]
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                gate=json.loads((out/(label+'.json')).read_text())
                if gate['passed'] is not True or gate['metadata']['policy_engine']!=metadata['policy_engine']:raise ValueError('scope same-process gate failed')
            write_new(out/'endpoint.json',dict(metadata=metadata,admission_path=args.admission,admission_sha256=file_hash(args.admission),server_pid=server.pid,port=port))
            for label in ['REF','L0_B3',l1_label]:
                if label==l1_label:
                    if workspace:
                        from robo.roundtrip.workspace_policy import seal
                        bundle=seal(workspace,configs['L0_B3'],out/'L0_B3/episode',out/'workspace_binding')
                        configs[label]=out/'workspace_binding/config.json'
                    else:
                        bundle=seal_bundle(out/'L0_B3/episode',configs['L0_B3'],args.target_dir,build_path,args.destination_candidate_pool,args.destination_rvg_receipt)
                        c=copy.deepcopy(config);c.update(scope='L1_target_destination',replacement_scope='target_destination')
                        configs[label]=out/(l1_label+'.json');write_new(configs[label],c)
                    write_new(out/'scope_bundle.json',bundle)
                command=[sys.executable,'-m','robo.roundtrip.paired','--config',str(configs[label]),'--canonical-reference',args.canonical_reference,
                    '--canonical-manifest',args.canonical_manifest,'--reset-bank',args.reset_bank,'--out',str(out/label),'--host','127.0.0.1','--port',str(port)]
                if label=='L0_B3':command+=['--object-dir',args.target_dir,'--reference-episode',str(out/'REF/episode')]
                elif label==l1_label:command+=['--scope-bundle',str(out/'scope_bundle.json'),'--baseline-config',str(configs['L0_B3']),'--reference-episode',str(out/'L0_B3/episode')]
                write_new(out/(label+'_command.json'),command)
                with (out/(label+'.log')).open('x') as episode_log:subprocess.run(command,check=True,stdout=episode_log,stderr=subprocess.STDOUT)
                p=out/label/'episode/result.json';result=json.loads(p.read_text())
                if not result.get('executed') or result.get('error') is not None or result.get('policy_engine')!=metadata['policy_engine']:
                    raise ValueError('scope canonical episode incomplete or engine differs')
                rows.append(dict(label=label,result_path=str(p),result_sha256=file_hash(p),ticks=result['ticks'],success=result['success']))
        write_new(out/'scope_pilot_summary.json',dict(planned_episodes=3,executed_episodes=3,rows=rows,engine=metadata['policy_engine'],
            variant=bundle['system_variant'],wall_s=time.monotonic()-start,claim_scope=('DEV partial observed workspace only; incomplete obstacle inventory, no L2 benchmark claim' if workspace else 'single predeclared DEV canonical; binding-specific native scope comparison')))
        return rows
    except Exception as exc:
        write_new(out/'failure.json',dict(type=type(exc).__name__,reason=str(exc),planned_episodes=3,completed_rows=rows,wall_s=time.monotonic()-start))
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline-config','canonical-reference','canonical-manifest','reset-bank','target-dir','admission','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--destination-candidate-pool');p.add_argument('--destination-rvg-receipt')
    p.add_argument('--destination-build-manifest');p.add_argument('--workspace-engineering-config')
    args=p.parse_args(argv)
    if bool(args.destination_build_manifest)==bool(args.workspace_engineering_config):p.error('exactly one destination build or DEV workspace engineering config required')
    run(args);return 0


if __name__=='__main__':raise SystemExit(main())
