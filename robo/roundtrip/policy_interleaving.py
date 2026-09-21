"""Bounded real-service request-order/reconnect smoke; never a policy episode."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from robo.roundtrip.native_policy import NativePolicy, STATE_KEYS, CAMERA_KEYS


def run_check(host,port):
    obs={key:np.zeros(n) for key,n in zip(STATE_KEYS,[3,4,3,4,2])}
    obs.update({key:np.full((256,256,3),i*37,dtype=np.uint8) for i,key in enumerate(CAMERA_KEYS.values())})
    obs['annotation.human.task_description']='Pick the object from the counter and place it in the sink.'
    metadata=[]
    def create(index):
        p=NativePolicy(host,port);p.bind_stream(canonical_instance_id=f'interleaving-smoke-instance-{index}',
                                             reset_id='reset-0',policy_rng_seed=42)
        p.reset(0);metadata.append(p.metadata);return p
    def sequence(index):
        p=create(index)
        try:return np.asarray([p.infer(obs) for _ in range(6)])
        finally:p.close()
    start=time.monotonic()
    serial=[sequence(i) for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel=list(pool.map(sequence,[1,0]))[::-1]
    resumed=[]
    for index in range(2):
        p=create(index)
        prefix=[p.infer(obs) for _ in range(3)];state=p.state_dict();p.close()
        # Reconnection restores pending actions and next chunk key, not server RNG.
        p=create(index);p.load_state_dict(state)
        try:resumed.append(np.asarray(prefix+[p.infer(obs) for _ in range(3)]))
        finally:p.close()
    matched=[np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(serial,parallel,resumed)]
    def hashes(rows):return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]
    return {'schema_version':2,'kind':'real_model_synthetic_observation_request_smoke',
        'native_episodes':0,'streams':2,'actions_per_stream':6,'model_chunk_requests':12,
        'passed':all(matched) and all(m==metadata[0] for m in metadata),
        'serial_action_sha256':hashes(serial),'reordered_concurrent_action_sha256':hashes(parallel),
        'resumed_action_sha256':hashes(resumed),'metadata':metadata[0],'wall_s':time.monotonic()-start,
        'interleave_rule':'independent clients reversed submit order; explicit per-chunk keys',
        'resume_rule':'new websocket client; exact saved queue/next chunk index'}


def compare_services(host,port,compare_host,compare_port,*,policy_factory=NativePolicy,artifact_dir=None,repeat_reference=False):
    """Compare complete official chunks, including actions the 5-step client discards."""
    from robo.roundtrip.native_policy import pack_observation,chunk_seed
    start=time.monotonic();policies=[policy_factory(host,port),policy_factory(compare_host,compare_port)]
    rows=[]
    artifact_dir=None if artifact_dir is None else Path(artifact_dir)
    if artifact_dir is not None:artifact_dir.mkdir(parents=True,exist_ok=False)
    from robo.manifest.hash import canonical_hash
    def input_hash(element):
        return canonical_hash({k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) if isinstance(v,np.ndarray) else v for k,v in element.items()})
    try:
        if policies[0].metadata!=policies[1].metadata:raise ValueError('cross-service policy/checkpoint/runtime metadata differs')
        for stream in range(2):
            for chunk in range(3):
                obs={key:np.zeros(n,dtype=np.float64) for key,n in zip(STATE_KEYS,[3,4,3,4,2])}
                obs[STATE_KEYS[1]][0]=1;obs[STATE_KEYS[3]][0]=1
                obs.update({key:np.full((256,256,3),i*37+stream*11+chunk,dtype=np.uint8) for i,key in enumerate(CAMERA_KEYS.values())})
                obs['annotation.human.task_description']='Pick the object from the counter and place it in the sink.'
                seed=chunk_seed(42,chunk,instance_id=f'cross-service-control-{stream}',reset_id='reset0')
                arrays=[];latencies=[];input_hashes=[]
                requests=policies+([policies[0]] if repeat_reference else [])
                for endpoint_index,policy in enumerate(requests):
                    element=pack_observation(obs);element['_simany_rng_seed']=seed
                    input_hashes.append(input_hash(element))
                    t=time.monotonic();response=policy._client.infer(element);latencies.append(time.monotonic()-t)
                    if response.get('simany_rng_seed')!=seed:raise ValueError('cross-service RNG receipt mismatch')
                    actions=np.asarray(response['actions'])
                    if actions.shape!=(50,12) or not np.isfinite(actions).all():raise ValueError('official full native chunk must be finite50x12')
                    arrays.append(actions)
                    if artifact_dir is not None:np.save(artifact_dir/f'stream{stream}_chunk{chunk}_endpoint{endpoint_index}.npy',actions,allow_pickle=False)
                identical=arrays[0].dtype==arrays[1].dtype and arrays[0].tobytes()==arrays[1].tobytes()
                if repeat_reference:identical=identical and arrays[0].dtype==arrays[2].dtype and arrays[0].tobytes()==arrays[2].tobytes()
                rows.append(dict(stream=stream,chunk=chunk,rng_seed=seed,shape=[50,12],input_sha256=input_hashes,
                    reference_repeat_byte_exact=None if not repeat_reference else arrays[0].tobytes()==arrays[2].tobytes(),
                    per_action_dimension_max_abs_difference=np.max(abs(arrays[0].astype(float)-arrays[1].astype(float)),axis=0).tolist(),
                    dtypes=[str(a.dtype) for a in arrays],action_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
                    byte_exact=bool(identical),max_abs_difference=float(np.max(abs(arrays[0].astype(float)-arrays[1].astype(float)))),latency_s=latencies))
        return dict(schema_version=2,kind='cross_service_full_chunk_identity_gate',native_episodes=0,
            endpoints=[dict(host=host,port=port),dict(host=compare_host,port=compare_port)],
            metadata=policies[0].metadata,model_chunk_requests=18 if repeat_reference else 12,full_actions_per_chunk=50,
            artifact_dir=None if artifact_dir is None else str(artifact_dir),repeat_reference=repeat_reference,
            passed=all(r['byte_exact'] for r in rows),rows=rows,wall_s=time.monotonic()-start,
            implementation_changed=False,required_before_independent_native_worker=True)
    finally:
        for policy in policies:policy.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True)
    p.add_argument('--port',type=int,default=8017);p.add_argument('--out',required=True)
    p.add_argument('--compare-host');p.add_argument('--compare-port',type=int,default=8017)
    p.add_argument('--artifact-dir');p.add_argument('--repeat-reference',action='store_true')
    args=p.parse_args(argv)
    result=compare_services(args.host,args.port,args.compare_host,args.compare_port,artifact_dir=args.artifact_dir,repeat_reference=args.repeat_reference) if args.compare_host else run_check(args.host,args.port)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
