"""E0-bound publication of independently replayed per-scene terminal caches."""
import argparse,contextlib,json,os,socket,subprocess,sys,time
from pathlib import Path
with contextlib.redirect_stdout(sys.stderr):
 from run.icra2027 import e4_planning_terminal as terminal

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',nargs='?',choices=('produce-full','validate-full'),default='produce-full');p.add_argument('--config',required=True);p.add_argument('--freeze-root',required=True);p.add_argument('--out')
 p.add_argument('--expected-code-commit',required=True);p.add_argument('--resume-validation',action='store_true');a=p.parse_args()
 stage=Path(a.freeze_root);out=stage/'full_qualification';audit=stage/'completion_audit.json'
 e0_path=stage/'contract/freeze_manifest.json';e0=terminal.api.read(e0_path);terminal.api.contract_digest(e0)
 if (Path.cwd()!=Path(e0['code']['repository'])
     or terminal.api.resource(e0,'terminal_python')!=terminal.api.identity(Path(sys.executable).resolve())
     or terminal.api.resource(e0,'terminal_driver')!=terminal.api.identity(__file__)
     or e0['code']['commit']!=a.expected_code_commit or e0['code']['dirty'] is not False):raise ValueError('terminal launcher E0/source binding differs')
 if a.out is not None and Path(a.out)!=out:raise ValueError('validation output differs from stage')
 if a.phase=='validate-full':
  with contextlib.redirect_stdout(sys.stderr):r=terminal.validate_full_coverage(out,expected_producer_commit=a.expected_code_commit)
  print(json.dumps(r,sort_keys=True));return
 if audit.exists():raise FileExistsError('completion audit already exists')
 if a.resume_validation:
  if not out.is_dir():raise ValueError('no sealed output to resume validating')
 else:terminal.produce_full(config_path=a.config,stage_root=stage,expected_code_commit=a.expected_code_commit,out=out)
 terminal.screen._validate_bundle(out,root=terminal.screen.evidence_root(),expected_kind=terminal.FULL_SCOPE)
 gate=terminal.api.read(out/'gate.json');source=gate['source']
 def identities():
  return dict(config=terminal.api.identity(a.config),E0=terminal.api.identity(e0_path),
   manifest=terminal.api.identity(out/'manifest.json'),seal=terminal.api.identity(out/'seal.json'),
   members={n:terminal.api.identity(out/n) for n in ('gate.json','scene_receipts.json','planned_qualification_cells.jsonl')},
   source_evidence={name:terminal.api.identity(source[key]['path']) for name,key in
    [('qualification_config','config'),('qualification_E0','e0'),('qualification_python','runtime'),('qualification_environment','environment')]},
   compatibility_envelope=terminal.api.identity(terminal.api.read(a.config)['compatibility']['path']))
 before=identities();command=[sys.executable,str(Path(__file__).resolve()),'validate-full','--out',str(out),'--config',a.config,'--freeze-root',str(stage),'--expected-code-commit',a.expected_code_commit]
 started=time.monotonic();process=subprocess.run(command,capture_output=True,text=True);elapsed=time.monotonic()-started
 process_path=stage/('validation_process_'+os.environ['SLURM_JOB_ID']+'.json')
 with process_path.open('x') as f:json.dump(dict(command=command,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,runtime_seconds=elapsed),f,indent=2)
 process.check_returncode()
 validated=json.loads(process.stdout)
 if identities()!=before:raise ValueError('sealed metadata changed during independent validation')
 # Compatibility is grouped with the other source identities for the agreed E9 schema.
 before['source_evidence']['compatibility_envelope']=before.pop('compatibility_envelope')
 receipt=dict(schema_version=1,status='PASS',scope=terminal.FULL_SCOPE,
  source={'commit':a.expected_code_commit,'repository':str(Path.cwd()),'dirty':False},
  full_output=str(out),**before,validated_gate=validated,command=command,
  job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),runtime_seconds=elapsed,
  driver=terminal.api.identity(__file__),validation_process=terminal.api.identity(process_path),paper_ready=False,headline_eligible=False,claim_gate='NOT_RUN',
  checks={'validate_full_coverage':'PASS','sealed_members_unchanged_during_validation':'PASS','source_config_E0_unchanged':'PASS'},
  interpretation='Canonical historical validators independently replayed per scene before publication; final metadata and composition verified after publication. No constructor, simulator measurement, or policy invocation; not a second geometric metric implementation.')
 with audit.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps({'status':'PASS','audit':terminal.api.identity(audit),'gate':validated},sort_keys=True))
if __name__=='__main__':main()
