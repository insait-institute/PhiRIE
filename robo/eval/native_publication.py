"""Guarded continuation for existing native tables, TeX build and publication.

No measurements are computed here. A failed gate leaves the generated diff and
an explicit BLOCKED receipt; no editorial or scientific repair is attempted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):return json.loads(Path(path).read_text())


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def check_code(plan):
    repo=Path(__file__).resolve().parents[2]
    if git(repo,'rev-parse','HEAD')!=plan['code_commit'] or git(repo,'status','--porcelain'):
        raise ValueError('publication producer source is changed or dirty')
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',plan['code_commit'],'refs/heads/main'],check=True)
    return repo


def check_paper_base(plan,*,remote=True):
    paper=Path(plan['paper_repo'])
    if git(paper,'rev-parse','HEAD')!=plan['paper_base_commit'] or git(paper,'status','--porcelain'):
        raise ValueError('pinned paper base changed or is dirty')
    if remote:
        found=git(paper,'ls-remote',plan['paper_remote'],'refs/heads/main').split()
        if len(found)!=2 or found[0]!=plan['paper_base_commit']:
            raise ValueError('remote paper main changed from pinned base')
    return paper


def closed_matrix_gate(primary,warm,status):
    """Return waiting for incomplete work; reject changed declared populations."""
    if primary['planned_units']!=2400:raise ValueError('primary population changed')
    if primary['state']!='COMPLETE_BLOCKS' or primary['measured_units']!=2400:return False
    if status['planned']!=1200:raise ValueError('L1 phase population changed')
    if status['terminal']!=1200 or any(g['unmeasured'] for g in status['groups']):return False
    if warm['planned_render_units']!=86:raise ValueError('warm render population changed')
    if warm['state']!='COMPLETE_AVAILABLE_RENDER_ATTEMPTS' or warm['unattempted']:return False
    ready,metrics,failed=set(warm['ready']),set(warm['metrics']),set(warm['failed'])
    if ready & failed or ready!=metrics or warm['metric_failures']:
        raise ValueError('warm admitted metrics are incomplete or overlap failures')
    if len(ready)+len(failed)!=86:raise ValueError('warm terminal denominator differs')
    return True


def readiness(plan):
    from robo.eval.native_final_release import bind,read_bound,prepare_inputs
    path=Path(plan['primary_release_receipt'])
    if not path.exists():return None
    receipt=read(path)
    if receipt['state']!='GENERATED':raise ValueError('primary-only final publisher failed: '+receipt['state'])
    prior=read_bound(receipt['release_manifest'])
    for name,digest in prior['artifacts'].items():read_path=Path(receipt['release_manifest']['path']).parent/name;read_bound_artifact(read_path,digest)
    experiment=read_bound(plan['experiment_plan'])
    warm_paths=sorted(Path(experiment['warm_collections']).glob('*/collection.json'))
    scope_paths=sorted(Path(experiment['scope_collections']).glob('*/collection.json'))
    if not warm_paths or not scope_paths:return None
    warm=read(warm_paths[-1]);scope_path=scope_paths[-1];scope_status=read(scope_path.with_name('status.json'))
    if not closed_matrix_gate(prior,warm,scope_status):return None
    prepared=prepare_inputs(experiment)
    if prepared is None:return None
    config,boundary=prepared
    if boundary['primary']['planned']!=2400 or boundary['primary']['measured']!=2400 or boundary['scope_planned']!=1200 or boundary['scope_unmeasured']!=0 or boundary['scope_measured']!=1200:
        raise ValueError('validated complete matrices differ')
    if boundary['warm_unattempted'] or boundary['warm_metric_failures'] or boundary['warm_render_ready']+len(boundary['warm_render_failures'])!=86:
        raise ValueError('validated warm terminal population differs')
    manifest=read_bound(experiment['warm_manifest'])
    frozen_methods={method for row in manifest['rows'] for method in row['methods']}
    if set(config['native_scale_up']['appearance_methods'])!=frozen_methods:
        raise ValueError('appearance methods differ from frozen render manifest')
    boundary['publication_primary_release']=bind(path)
    boundary['publication_complete_scope_required']=True
    config['native_scale_up']['native_paper_section']=True
    return experiment,(config,boundary)


def read_bound_artifact(path,digest):
    from robo.eval.native_final_release import bind
    if bind(path)['sha256']!=digest:raise ValueError('publication artifact changed: '+str(path))


def preserve_snapshot(paper):
    from robo.eval.native_final_release import bind
    tracked=git(paper,'ls-files','-z').split('\0')
    return {name:bind(Path(paper)/name)['sha256'] for name in tracked if name}


def verify_preserved(paper,preserved):
    for name,digest in preserved.items():read_bound_artifact(Path(paper)/name,digest)


def verify_generated_claims(tables,paper,preserved):
    from robo.eval.native_final_release import bind
    from robo.eval.native_scale_paper import DISABLED_CLAIMS
    verify_preserved(paper,preserved)
    gates=read(Path(tables)/'native_paper_claim_gates.json')
    expected={k:sentence for k,sentence,_ in DISABLED_CLAIMS}
    if len(gates)!=len(expected) or {r['claim_id']:r['disabled_sentence'] for r in gates}!=expected or any(r['enabled_sentence'] or r['scientific_gate']!='NOT_RUN' for r in gates):
        raise ValueError('unsupported native scientific claim was enabled')
    section=Path(paper)/'paper_sections/06_native_test.tex'
    text=section.read_text()
    if 'Incomplete TEST snapshot' in text or any(sentence in text for sentence in expected.values()):
        raise ValueError('partial or unsupported TEST prose')
    transfer=read(Path(paper)/'audit/native_test_transfer.json')
    if transfer.get('primary_complete') is not True:raise ValueError('transfer is not complete primary')
    for name,item in transfer['files'].items():
        read_bound_artifact(Path(tables)/name,item['sha256'])
        read_bound_artifact(Path(paper)/item['destination'],item['sha256'])
    allowed={r['destination'] for r in transfer['files'].values()}|{'audit/native_test_transfer.json'}
    actual=set(git(paper,'ls-files','--others','--exclude-standard').splitlines())
    if actual!=allowed:raise ValueError('unplanned new paper artifact: '+str(sorted(actual^allowed)))
    return dict(gates=bind(Path(tables)/'native_paper_claim_gates.json'),transfer=bind(Path(paper)/'audit/native_test_transfer.json'),preserved_files=len(preserved),all_original_tracked_bytes_preserved=True)


def pdf_qa(paper,out):
    """Use installed PyMuPDF, as in existing paper audits; no metric producer."""
    import fitz
    paper=Path(paper);out=Path(out);out.mkdir(parents=True,exist_ok=False);documents={}
    for name in ('conference','root'):
        pdf=paper/(name+'.pdf');log=(paper/(name+'.log')).read_text(errors='replace')
        problems=[s for s in log.splitlines() if any(w in s for w in ('LaTeX Error','Overfull','undefined'))]
        if problems:raise ValueError(name+' TeX audit: '+' | '.join(problems))
        doc=fitz.open(pdf);fonts={};page_paths=[]
        if name=='conference' and len(doc)>8:raise ValueError(f'conference has {len(doc)} pages, limit8')
        for i,page in enumerate(doc):
            if re.search(r'\b(?:TBD|TODO)\b|\?\?',page.get_text()):raise ValueError('unresolved visible placeholder on '+name+' page'+str(i+1))
            p=out/f'{name}-{i+1:02}.png';page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(p);page_paths.append(str(p))
            for font in page.get_fonts(full=True):
                data=doc.extract_font(font[0]);fonts[font[0]]=dict(name=font[3],embedded=bool(data[3]))
        if not fonts or not all(f['embedded'] for f in fonts.values()):raise ValueError(name+' has unembedded fonts')
        import hashlib
        documents[name]=dict(pages=len(doc),page_limit_pass=name!='conference' or len(doc)<=8,
            all_pages_rendered=True,rendered_pages=page_paths,fonts=list(fonts.values()),
            sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),log_sha256=hashlib.sha256((paper/(name+'.log')).read_bytes()).hexdigest())
    report=dict(status='PASS',documents=documents,qa_scope='TeX errors/undefined/overfull; conference<=8 including references; full-page rasterization; embedded fonts; visible placeholders')
    write(out/'qa.json',report);return report



def render_native_demo(plan,tables,control):
    from robo.eval.native_final_release import bind
    repo=Path(__file__).resolve().parents[2];tables=Path(tables)
    out=tables.parent/'native_demo'
    argv=[plan['demo_python'],'-m','interface.demo_agentic','native-hero','--release',str(tables),'--out',str(out)]
    with (control/'native_demo.log').open('x') as log:
        subprocess.run(argv,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
    manifest=read(out/'demo_source_manifest.json');qa=read(out/'qa.json')
    if (manifest['state']!='COMPLETE_TEST_PRESENTATION' or manifest['release']!=str(tables.resolve())
        or manifest['release_sha256']!=bind(tables/'release_manifest.json')['sha256']
        or manifest['native_track_demo_ready'] is not True or manifest['full_demo_ready'] is not False):
        raise ValueError('native demo does not bind the same complete TEST release')
    if (qa.get('status')!='PASS' or qa.get('decoded_all_hero_frames') is not True
        or qa.get('frames')!=2700 or qa.get('duration_s')!=90 or qa.get('resolution')!=[1920,1080]
        or qa.get('loop_seam_exact') is not True or qa.get('native_track_demo_ready') is not True):
        raise ValueError('native demo QA failed')
    for path,digest in manifest['source_files'].items():read_bound_artifact(path,digest)
    outputs={name:bind(out/name) for name in ('native_test_90s_1080p.mp4','teaser_30s_1080p.mp4',
        'static_evidence_loop_8s.mp4','poster.png','native_test.srt','B4_repair_continuous.mp4')}
    receipt=dict(state='COMPLETE_NATIVE_TRACK',manifest=bind(out/'demo_source_manifest.json'),qa=bind(out/'qa.json'),outputs=outputs,command=argv,full_room_demo_claim=False)
    write(control/'native_demo_receipt.json',receipt)
    return receipt


def publish(plan,prepared,control):
    from robo.eval.native_final_release import bind,run_release
    repo=check_code(plan);paper=check_paper_base(plan);work=Path(plan['paper_worktree'])
    if work.exists():raise ValueError('publication paper worktree already exists')
    preserved=preserve_snapshot(paper)
    write(control/'preserved_paper.json',dict(paper_base=plan['paper_base_commit'],files=preserved))
    subprocess.run(['git','-C',str(paper),'worktree','add','-b',plan['paper_branch'],str(work),plan['paper_base_commit']],check=True)
    experiment,inputs=prepared
    result=run_release(experiment,inputs,plan['code_commit'],control,paper_root=work)
    write(control/'generation.json',result)
    if result['state']!='GENERATED':raise ValueError('existing paper pipeline failed')
    tables=Path(result['release_manifest']['path']).parent
    claims=verify_generated_claims(tables,work,preserved);write(control/'claim_audit.json',claims)
    with (control/'build.log').open('x') as log:subprocess.run(['bash',str(work/'build.sh')],stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[plan['qa_python'],'-m','robo.eval.native_publication','qa','--paper',str(work),'--out',str(control/'pdf_qa')]
    with (control/'qa.log').open('x') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    qa=read(control/'pdf_qa/qa.json')
    if qa['status']!='PASS':raise ValueError('PDF QA did not pass')
    demo=render_native_demo(plan,tables,control)
    verify_preserved(work,preserved);check_code(plan);check_paper_base(plan)
    write(work/'audit/native_publication_qa.json',dict(qa=bind(control/'pdf_qa/qa.json'),claim_audit=bind(control/'claim_audit.json'),release=result['release_manifest'],source_commit=plan['code_commit'],paper_base=plan['paper_base_commit'],all_original_tracked_bytes_preserved=True,preserved_source=bind(control/'preserved_paper.json'),native_demo=bind(control/'native_demo_receipt.json')))
    allowed=[*read(work/'audit/native_test_transfer.json')['files'].values()]
    paths=[r['destination'] for r in allowed]+['audit/native_test_transfer.json','audit/native_publication_qa.json']
    subprocess.run(['git','-C',str(work),'add','--',*paths],check=True)
    subprocess.run(['git','-C',str(work),'commit','-m','Publish complete native TEST results with coverage and preserved limitations'],check=True)
    commit=git(work,'rev-parse','HEAD')
    if git(work,'status','--porcelain'):raise ValueError('publication worktree is dirty after commit')
    write(control/'push_admission.json',dict(commit=commit,remote=plan['paper_remote'],target='refs/heads/main',paper_base=plan['paper_base_commit'],qa=bind(control/'pdf_qa/qa.json'),all_preserved_bytes_checked=True))
    check_paper_base(plan)
    subprocess.run(['git','-C',str(work),'push',plan['paper_remote'],commit+':refs/heads/main'],check=True)
    remote=git(work,'ls-remote',plan['paper_remote'],'refs/heads/main').split()[0]
    write(control/'push_result.json',dict(paper_commit=commit,remote_commit=remote,push_command_succeeded=True))
    if remote!=commit:raise ValueError('remote publication commit could not be verified')
    # Updating the shared checkout is reversible, and guarded against local edits.
    check_paper_base(plan,remote=False)
    subprocess.run(['git','-C',str(paper),'merge','--ff-only',commit],check=True)
    return dict(state='PUBLISHED',paper_commit=commit,remote_commit=remote,release=result['release_manifest'],qa=bind(control/'pdf_qa/qa.json'),paper_worktree=str(work),native_demo=demo,scientific_claim_promotion=False)


def preserve_failure(plan,control,exc):
    """Keep tracked changes, newly generated sources and any already-pushed state."""
    from robo.eval.native_final_release import bind
    import shutil
    work=Path(plan['paper_worktree']);archive=control/'preserved_generated_files'
    changes={};diff='';audit_errors=[]
    if work.exists():
        try:
            diff=git(work,'diff','--binary',plan['paper_base_commit'])
            names=set(git(work,'ls-files','--others','--exclude-standard').splitlines())
            names.update(git(work,'diff','--name-only',plan['paper_base_commit']).splitlines())
            for name in sorted(names):
                source=work/name
                if source.is_file():
                    target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,target);changes[name]=bind(target)
        except Exception as audit_exc:audit_errors.append(str(audit_exc))
    (control/'preserved_paper.diff').write_text(diff)
    push=read(control/'push_result.json') if (control/'push_result.json').exists() else None
    return dict(state='BLOCKED',error_type=type(exc).__name__,error=str(exc),paper_worktree=str(work),
        paper_diff=str(control/'preserved_paper.diff'),preserved_generated_files=changes,
        push_attempted=(control/'push_admission.json').exists(),push_result=push,
        remote_published=bool(push and push['paper_commit']==push['remote_commit']),audit_errors=audit_errors)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action in ('check','watch'):
        p=sub.add_parser(action);p.add_argument('--plan',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--max-hours',type=float,default=24)
    p=sub.add_parser('qa');p.add_argument('--paper',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action=='qa':print(json.dumps(pdf_qa(args.paper,args.out),indent=2));return 0
    from robo.eval.native_final_release import bind
    plan=read(args.plan);digest=bind(args.plan);control=args.control.resolve();control.mkdir(parents=True,exist_ok=False)
    write(control/'launch.json',dict(pid=os.getpid(),plan=digest,action=args.action,max_hours=args.max_hours,source_commit=plan['code_commit']))
    started=time.monotonic();result=dict(state='WAIT_LIMIT_REACHED',paper_mutated=False)
    try:
        while time.monotonic()-started<args.max_hours*3600:
            if bind(args.plan)!=digest:raise ValueError('publication plan changed')
            check_code(plan);ready=readiness(plan)
            if args.action=='check':
                check_paper_base(plan);result=dict(state='READY' if ready else 'WAITING',paper_mutated=False);break
            if ready is not None:result=publish(plan,ready,control);break
            time.sleep(60)
    except Exception as exc:
        result=preserve_failure(plan,control,exc)
    write(control/'completion_receipt.json',result);return 0 if result['state'] in ('PUBLISHED','READY','WAITING') else 1


if __name__=='__main__':raise SystemExit(main())
