from pathlib import Path
import json,hashlib,subprocess,datetime,sys,inspect,collections
import numpy as np,trimesh
from robo.manifest.hash import canonical_hash
from robo.roundtrip.importers import robocasa as importer
out=Path(__file__).resolve().parent
runtime=Path('/group/worldcept/code/SimAny-wt/n1-test-execution-73ff4ab')
execution=Path('/group/worldcept/code/SimAny/outputs/icra2027/20260907-73ff4ab-v1/sim_recon_sim/scale_up/test_execution')
iid='native-c1d6ffb137352ef5f4f2625c933fa3a98d5801448076dc097c84c3682347eafe'
bindings=Path('/group/worldcept/code/SimAny/outputs/icra2027/20260907-4f5c745-v2/sim_recon_sim/scale_up/test_dispatch/build_readiness_snapshot_004/build_bindings.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
binding=next(r for r in map(json.loads,bindings.read_text().splitlines()) if r['canonical_instance_id']==iid and r['controller_method']=='B0_FIXED_NATIVE')
assert sha(binding['binding_source'])==binding['binding_source_sha256']
manifest_path=Path(binding['build_manifest']);assert sha(manifest_path)==binding['build_manifest_sha256'];manifest=json.loads(manifest_path.read_text());obj=Path(binding['object_dir'])
assert manifest['canonical_instance_id']==iid and manifest['method']=='B0_fixed_trellis'
verified={}
for rel,digest in manifest['source_hashes'].items():
 p=manifest_path.parent/rel;assert sha(p)==digest,str(p);verified[str(p)]=digest
part=obj/'collision/part_04.obj';m=trimesh.load(part,process=False,force='mesh')
nonzero=m.area_faces>trimesh.constants.tol.zero;adj_ok=nonzero[m.face_adjacency].all(axis=1)
projections=m.face_adjacency_projections[adj_ok];threshold=trimesh.constants.tol.planar*m.scale
_,edges=np.unique(np.sort(m.edges,axis=1),axis=0,return_counts=True)
properties=dict(vertices=len(m.vertices),faces=len(m.faces),is_watertight=bool(m.is_watertight),is_convex=bool(m.is_convex),is_winding_consistent=bool(m.is_winding_consistent),volume_local=float(m.volume),body_count=int(m.body_count),vertices_finite=bool(np.isfinite(m.vertices).all()),edge_incidence_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(edges,return_counts=True))},duplicate_vertex_coordinates=len(m.vertices)-len(np.unique(m.vertices,axis=0)),mesh_scale=float(m.scale),planar_tolerance=float(trimesh.constants.tol.planar),convexity_threshold=float(threshold),maximum_adjacent_projection=float(projections.max()),violating_adjacencies=int(sum(projections>=threshold)),geometry_units='unscaled reconstructed local mesh; not a native fidelity error')
assert not m.is_convex and m.is_watertight and m.volume>0
parts=[]
for p in sorted((obj/'collision').glob('part_*.obj')):
 a=trimesh.load(p,process=False,force='mesh');parts.append(dict(file=bind(p),watertight=bool(a.is_watertight),convex=bool(a.is_convex),positive_volume=bool(a.volume>0)))
xml='<mujoco><worldbody><body name="obj_main"><freejoint name="obj_joint0"/><geom name="old" type="box" size=".01 .01 .01" mass=".3"/></body></worldbody></mujoco>'
expected='ValueError: collision part is not a closed convex volume: part_04.obj'
try:importer.import_reconstructed_object(xml,body_name='obj_main',object_dir=obj,object_id='obj',role='target')
except ValueError as exc:actual=type(exc).__name__+': '+str(exc)
else:raise AssertionError('frozen importer unexpectedly accepted original artifact')
assert actual==expected
source=subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip();assert source=='73ff4ab0a6e53c847960b03385f0c4f040d5e4a0'
assert not subprocess.check_output(['git','-C',str(runtime),'status','--porcelain'],text=True).strip()
importer_path=Path(importer.__file__);assert importer_path.resolve()==runtime/'robo/roundtrip/importers/robocasa.py'
committed=subprocess.check_output(['git','-C',str(runtime),'show',source+':robo/roundtrip/importers/robocasa.py']);assert hashlib.sha256(committed).hexdigest()==sha(importer_path)
ledger=execution/'release_final_001/merged/episode_ledger.jsonl';rows=[r for r in map(json.loads,ledger.read_text().splitlines()) if r['canonical_instance_id']==iid and r['controller_method']=='B0_FIXED_NATIVE'];assert len(rows)==10
units=[]
for row in rows:
 w=execution/'workers'/row['unit_id'];tp=w/'terminal.json';terminal=json.loads(tp.read_text());planned=json.loads((w/'planned_unit.json').read_text());config=Path(planned['config_path']);pair=w/'runner/pair_receipt.json';pr=json.loads(pair.read_text());log=w/'worker.log'
 assert terminal['terminal_status']=='CODE_FAILED' and terminal['executed'] is False and terminal['success'] is None and terminal['result'] is None
 assert terminal['source_code']['commit']==source and terminal['slurm_job_id']=='840315'
 assert terminal['argv'][terminal['argv'].index('--object-dir')+1]==str(obj)
 assert canonical_hash(json.loads(config.read_text()))==planned['config_sha256']==pr['config_sha256']
 assert canonical_hash(planned)==terminal['planned_unit_sha256']
 assert expected in log.read_text() and pr['failure']==expected and pr['executed_comparison_episodes']==0
 assert not (w/'runner/episode/result.json').exists()
 units.append(dict(unit_id=row['unit_id'],reset_id=row['reset_id'],original_status=row['terminal_status'],terminal=bind(tp),worker_log=bind(log),pair_receipt=bind(pair),planned_unit=bind(w/'planned_unit.json'),config=bind(config),config_canonical_sha256=planned['config_sha256'],executed=False,success=None,proposed_status='BUILD_FAILED'))
assert {r['reset_id'] for r in units}=={'r'+str(i) for i in range(10)}
# Recheck immutable construction bytes after the read-only CPU validation.
assert all(sha(p)==h for p,h in verified.items())
plan_paths=[runtime/'plan/icra2027/11_sim_recon_sim/PROTOCOL.md',runtime/'plan/icra2027/11_sim_recon_sim/09_scale_up/EXPERIMENT_MATRIX.md']
result=dict(schema_version=1,kind='independent_original_collision_import_audit',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source,source_dirty=False,python=sys.executable,numpy_version=np.__version__,trimesh_version=trimesh.__version__,trimesh_convex_source=bind(inspect.getsourcefile(trimesh.convex)),importer_source=bind(importer_path),script=bind(__file__),canonical_instance_id=iid,original_job_id='840315',scheduler_output=subprocess.check_output(['sacct','-j','840315','--format=JobID,State,Elapsed,ExitCode','-P','-n'],text=True),original_ledger=bind(ledger),original_build_binding=bind(bindings),build_binding=binding,build_manifest=bind(manifest_path),constructor_config=bind(manifest_path.parent/'build_config.json'),constructor_config_canonical_sha256=manifest['config_sha256'],object_dir=str(obj),collision_part=bind(part),part_properties=properties,all_parts=parts,construction_source_hashes_verified=verified,exact_importer_exception=actual,reproduction_scope='same frozen importer and native Python; neutral synthetic XML body; no native scene, GT geometry, policy or physics execution',units=units,plan_sources=[bind(p) for p in plan_paths],classification='METHOD_CONSTRUCTION_IMPORT_VALIDITY_FAILURE',adjudication_supported=True,proposed_status='BUILD_FAILED',proposed_executed=False,proposed_success=None,preserve_original_CODE_FAILED=True,adjudication_must_be_separate_and_source_bound=True,geometry_repaired=False,threshold_changed=False,new_policy_outcomes=0,limitations=['Near-threshold local nonconvexity: do not describe this part as open, negative-volume, or grossly malformed.','Original builder BUILT marks artifact production; manifest explicitly records native_import NOT_RUN and its isolated probe used a single hull, not this16part collision.','The generic runner terminal classification was broad CODE_FAILED; diagnostic reproduction supports a typed construction/import failure, not a completed task failure.','No claim that MuJoCo itself cannot auto-convexify the asset: the frozen explicit importer rejects it before compilation, and bypassing that gate would change the declared treatment.'])
with (out/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ('classification','adjudication_supported','exact_importer_exception')}));print(properties)
