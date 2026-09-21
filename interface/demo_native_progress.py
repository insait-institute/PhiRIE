"""Continuous native presentation from an existing paper-pipeline release."""
import json
from pathlib import Path
import shutil
import subprocess
from agents.orchestrator.artifact import sha256_file


def native_tier(first):
    tiers = {'development': 'DEV', 'dev': 'DEV', 'test': 'TEST'}
    if first.get('split') not in tiers:
        raise ValueError('native presentation requires a declared DEV or TEST split')
    return tiers[first['split']]


def released_video(sources, result_source_key, video_path):
    """Require the selected result's companion release-time media seal."""
    suffix='_result_path'
    if not result_source_key.endswith(suffix):
        raise ValueError('video requires a released result identity')
    key=result_source_key[:-len(suffix)]+'_video_path'
    record=sources.get(key)
    video=Path(video_path).resolve()
    if not record or Path(record['path']).resolve()!=video:
        raise ValueError('selected episode video has no matching release media seal')
    if sha256_file(video)!=record['sha256']:
        raise ValueError('selected episode video changed after its release media seal')
    return record['sha256']


def resolve_pair(release_dir):
    root=Path(release_dir);release=json.loads((root/'release_manifest.json').read_text())
    for name,digest in release['artifacts'].items():
        if sha256_file(root/name)!=digest:raise ValueError('release artifact changed')
    sources=release['source_lineage']
    planned_path=Path(sources['planned_units']['path'])
    if sha256_file(planned_path)!=sources['planned_units']['sha256']:raise ValueError('planned roster changed')
    first=json.loads(planned_path.read_text().splitlines()[0])
    native_tier(first)
    identity=(first['canonical_instance_id'],first['reset_id'],first['policy_rng_seed'])
    pair={}
    for name,item in sources.items():
        if not name.endswith('_result_path'):continue
        path=Path(item['path'])
        if sha256_file(path)!=item['sha256']:raise ValueError('released episode changed')
        row=json.loads(path.read_text())
        if tuple(row.get(k) for k in ('canonical_instance_id','reset_id','policy_rng_seed'))!=identity:continue
        method=row.get('controller_method')
        if method not in ('REF_NATIVE','B0_FIXED_NATIVE'):continue
        if method in pair:raise ValueError('duplicate episode for presentation pair')
        if not row['executed'] or row['error'] is not None or row.get('video_error') is not None:raise ValueError('episode/video incomplete')
        video=Path(row['video_path']);video_sha=released_video(sources,name,video)
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)],text=True))
        stream=next(s for s in meta['streams'] if s['codec_type']=='video')
        if int(stream['nb_frames'])!=row['ticks']+1 or stream['r_frame_rate']!='20/1':raise ValueError('continuous source frame coverage differs')
        pair[method]={'result':row,'result_path':str(path),'result_sha256':item['sha256'],'video_path':str(video),'video_sha256':video_sha}
    if set(pair)!= {'REF_NATIVE','B0_FIXED_NATIVE'}:raise ValueError('first declared native pair is not fully released')
    a,b=[pair[m]['result'] for m in ('REF_NATIVE','B0_FIXED_NATIVE')]
    for key in ('canonical_instance_id','reset_id','policy_rng_seed','policy_identity_sha256','horizon','execution_protocol','scope','renderer'):
        if a[key]!=b[key]:raise ValueError('presentation pair differs on '+key)
    if (a['execution_protocol']!='primary_native' or a['scope']!='L0_target_only' or
            a['horizon'] != 600 or not all(0 < r['ticks'] <= 600 for r in (a,b))):
        raise ValueError('this 30-second native presentation requires a600tick primary L0 horizon')
    return release,first,pair


def render(release_dir,out):
    from PIL import Image,ImageDraw
    from interface.demo_movie import _font
    release,first,pair=resolve_pair(release_dir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    tier=native_tier(first)
    bg=Image.new('RGB',(1920,1080),(13,20,31));d=ImageDraw.Draw(bg)
    def text(x,y,s,size=32,color=(220,229,240)):d.text((x,y),s,font=_font(size),fill=color)
    text(48,44,'SimAnyRoom',56,(92,216,231));text(540,61,'NATIVE RECONSTRUCTION / '+tier,34)
    text(48,133,f"First declared instance | layout {first['layout_id']} | {first['task_id']} | {first['reset_id']}",30)
    text(48,205,'REF_NATIVE / native target',32);text(960,205,'B0_FIXED_NATIVE / reconstructed target',32)
    text(48,824,'L0 target only; original room retained as oracle context.',32)
    text(48,872,'Ideal posed RGB-D; uniform native rendering; frozen visual policy.',30)
    text(48,920,'Independent continuous episodes at real time. Ended footage holds its last frame.',28)
    text(48,966,'Official native success is body-frame binding-specific. No preservation claim.',28,(247,187,112))
    counts=json.loads((Path(release_dir)/'T2_manipulation.json').read_text())
    countline=' | '.join(f"{r['method'].replace('_NATIVE','')}: executed {r['executed']}/{r['planned']}" for r in counts if r['method'] in pair)
    text(48,1018,'Release snapshot: '+countline,28)
    bg.save(out/'background.png')
    filters=['[1:v]trim=start_frame=1,setpts=PTS-STARTPTS,scale=912:456,tpad=stop_mode=clone:stop_duration=30[r]',
             '[2:v]trim=start_frame=1,setpts=PTS-STARTPTS,scale=912:456,tpad=stop_mode=clone:stop_duration=30[b]',
             '[0:v][r]overlay=48:260[x]','[x][b]overlay=960:260[y]']
    last='y'
    for i,(method,entry) in enumerate(pair.items()):
        # Draw from released outcomes only; no manually supplied score values.
        row=entry['result'];x=48 if method=='REF_NATIVE' else 960
        end=row['ticks']/20;caption=f"Final result: {'SUCCESS' if row['success'] else 'TASK FAILURE'} / {row['ticks']} actions"
        label=out/(method+'.txt');label.write_text(caption)
        new='z'+str(i)
        filters.append(f"[{last}]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile={label}:fontsize=28:fontcolor=white:x={x}:y=748[{new}]");last=new
        held=out/(method+'_held.txt');held.write_text('Episode ended; final frame held')
        new='held'+str(i)
        filters.append(f"[{last}]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile={held}:fontsize=28:fontcolor=0xF7BB70:x={x}:y=786:enable='gte(t,{end})'[{new}]");last=new
    script=out/'filters.txt';script.write_text(';\n'.join(filters))
    video=out/'progress_30s_1080p.mp4'
    args=['ffmpeg','-v','error','-nostdin','-loop','1','-i',str(out/'background.png')]
    for method in ('REF_NATIVE','B0_FIXED_NATIVE'):args+=['-i',pair[method]['video_path']]
    args+=['-filter_complex_threads','2','-filter_complex_script',str(script),'-map','['+last+']','-t','30','-r','30','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video)]
    (out/'render_command.json').write_text(json.dumps(args,indent=2));subprocess.run(args,check=True)
    subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(video),'-f','null','-'],check=True)
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    v=next(s for s in metadata['streams'] if s['codec_type']=='video')
    if (v['width'],v['height'],int(v['nb_frames']))!=(1920,1080,900) or float(metadata['format']['duration'])!=30:raise ValueError('rendered profile differs')
    subprocess.run(['ffmpeg','-v','error','-ss','18','-i',str(video),'-frames:v','1','-n',str(out/'poster.png')],check=True)
    for method,entry in pair.items():shutil.copyfile(entry['video_path'],out/(method+'_continuous.mp4'))
    (out/'progress.srt').write_text(f'1\n00:00:00,000 --> 00:00:30,000\nContinuous paired {tier} episodes. Target-only reconstruction; native room retained. Official native outcomes are binding-specific.\n')
    manifest={'state':tier+'_PROGRESS_ONLY','release':str(Path(release_dir).resolve()),'release_sha256':sha256_file(Path(release_dir)/'release_manifest.json'),
        'selection_rule':f'first canonical instance/reset in predeclared {tier} roster, independent of outcomes','pair':pair,
        'editing':'drop only initial static frame; all action frames continuous; hold ended reference frame explicitly; no episode splice; 20 to30fps resampling',
        'source_resolution':[512,256],'master_resolution':[1920,1080],'duration_s':30,'source_lineage':release['source_lineage']}
    (out/'demo_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'qa.json').write_text(json.dumps({'status':'PASS','decoded_all_frames':True,'frames':900,'size':[1920,1080],'duration_s':30,'full_demo_ready':False},indent=2))
    return str(video)


_SCOPE_LABELS = ('REF', 'L0_B3', 'L1_B3')


def _validate_scope_rows(diagnosis, entries):
    """Presentation admission only; all outcomes, including failures, survive."""
    if diagnosis.get('kind') != 'first_predeclared_scope_r0_diagnostic':
        raise ValueError('requires first declared scope diagnostic')
    if len(entries) != 3 or {e['label'] for e in entries} != set(_SCOPE_LABELS):
        raise ValueError('missing or duplicate scope arm')
    pair = {e['label']: e for e in entries}
    ref = pair['REF']['result']
    keys = ('canonical_instance_id', 'canonical_manifest_sha256', 'reset_id',
            'reset_contract_sha256', 'policy_rng_seed', 'policy_identity_sha256',
            'policy_engine', 'horizon', 'execution_protocol', 'renderer', 'sensor_regime', 'task_id')
    for label in _SCOPE_LABELS:
        row = pair[label]['result']; source = pair[label]['diagnostic']
        for key in keys:
            if key not in row or row[key] != ref[key]:
                raise ValueError('scope presentation changes ' + key)
        if (row['canonical_instance_id'] != diagnosis['canonical_instance_id'] or
            row['reset_id'] != 'r0' or not row.get('policy_engine') or
            row['execution_protocol'] != 'primary_native' or row['horizon'] != 600 or
            row.get('execution_kind') != 'closed_loop_visual_policy' or
            row.get('executed') is not True or row.get('error') is not None or row.get('video_error') is not None):
            raise ValueError('incomplete or incompatible first reset episode')
        for key in ('ticks', 'success', 'first_success_step', 'initial_state_sha256'):
            if source[key] != row[key]:raise ValueError('diagnostic differs from result: ' + key)
        if source['engine'] != row['policy_engine'] or source['reset_state_id'] != row['reset_id']:
            raise ValueError('diagnostic engine/reset differs')
        expected_method = 'REF_NATIVE' if label == 'REF' else 'B3_AGENT_NATIVE'
        expected_scope = 'L1_target_destination' if label == 'L1_B3' else 'L0_target_only'
        if row['controller_method'] != expected_method or row['scope'] != expected_scope:
            raise ValueError('scope label differs from actual treatment')
    a,b = pair['L0_B3']['result'],pair['L1_B3']['result']
    if a['initial_state_sha256'] != b['initial_state_sha256']:
        raise ValueError('L0/L1 reset state differs')
    if max(e['result']['ticks'] for e in entries) != 600:
        raise ValueError('30s diagnostic requires a continuous full-horizon arm')
    contact = pair['L1_B3']['diagnostic'].get('max_penetration_contact')
    if (not contact or contact['distance_m'] >= 0 or 'destination' not in contact['object_ids'] or
        not any(name.startswith('robot0_') for name in contact['geom_names'])):
        raise ValueError('robot-destination interference diagnosis is unavailable')
    return pair


def resolve_scope_diagnostic(diagnosis_path):
    from robo.manifest.hash import canonical_hash
    from robo.roundtrip.scope_bundle import validate_scope_pair
    path = Path(diagnosis_path).resolve(); diagnosis = json.loads(path.read_text()); entries = []
    for source in diagnosis['rows']:
        result_path = Path(source['result_path'])
        if sha256_file(result_path) != source['result_sha256']:raise ValueError('diagnostic result changed')
        row = json.loads(result_path.read_text()); trace = Path(row['timeseries_path']); video = Path(row['video_path'])
        if sha256_file(trace) != source['trace_sha256']:raise ValueError('diagnostic trace changed')
        if str(video) != source['video_path']:raise ValueError('diagnostic video differs')
        if sha256_file(row['actions_path']) != row['actions_sha256']:raise ValueError('episode actions changed')
        planned_path = result_path.parents[2]/'planned_unit.json'
        planned = json.loads(planned_path.read_text()); config_path = Path(planned['config_path'])
        config = json.loads(config_path.read_text())
        if canonical_hash(config) != planned['config_sha256']:raise ValueError('planned config changed')
        if canonical_hash(config) != row['config_sha256']:raise ValueError('episode config changed')
        meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)],text=True))
        stream = next(s for s in meta['streams'] if s['codec_type']=='video')
        if (int(stream['nb_frames']) != row['ticks']+1 or stream['r_frame_rate'] != '20/1' or
            (stream['width'],stream['height']) != (512,256)):
            raise ValueError('continuous source frame coverage differs')
        entries.append(dict(label=source['label'],result=row,diagnostic=source,result_path=str(result_path),
            result_sha256=source['result_sha256'],video_path=str(video),video_sha256=sha256_file(video),
            trace_path=str(trace),trace_sha256=source['trace_sha256'],config=config,config_path=str(config_path),
            config_file_sha256=sha256_file(config_path),config_canonical_sha256=planned['config_sha256'],planned_path=str(planned_path),planned_sha256=sha256_file(planned_path)))
    pair = _validate_scope_rows(diagnosis,entries)
    validate_scope_pair(pair['L1_B3']['config'],pair['L0_B3']['config'],pair['L0_B3']['result'])
    return diagnosis,pair


def render_scope_diagnostic(diagnosis_path,out):
    """Existing continuous FFmpeg compositor, with three declared scope arms."""
    from PIL import Image,ImageDraw
    from interface.demo_movie import _font
    diagnosis,pair = resolve_scope_diagnostic(diagnosis_path)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    bg=Image.new('RGB',(1920,1080),(13,20,31));d=ImageDraw.Draw(bg)
    def text(x,y,s,size=30,color=(220,229,240)):d.text((x,y),s,font=_font(size),fill=color)
    text(48,42,'SimAnyRoom',54,(92,216,231));text(560,59,'NATIVE SCOPE / FIRST TEST RESET',32)
    text(48,128,'n = 1 instance diagnostic | same policy engine and reset r0 | no aggregate claim',30)
    text(48,179,'What changes when we also reconstruct the destination?',38)
    labels={'REF':('REF / native scene','Native target and destination'),
            'L0_B3':('L0 / reconstructed target','Native destination and room'),
            'L1_B3':('L1 / target + destination','Reconstructed destination; retained room')}
    for i,label in enumerate(_SCOPE_LABELS):
        x=48+i*616;text(x,270,labels[label][0],27,(92,216,231));text(x,314,labels[label][1],23)
    text(48,780,'Measured L1 interference: robot link contacts the reconstructed destination.',33,(247,187,112))
    contact=pair['L1_B3']['diagnostic']['max_penetration_contact']
    text(48,834,f"Logged contact distance: {contact['distance_m']*1000:.1f} mm at tick {contact['tick']} (penetration).",30)
    text(48,884,'Failure diagnosis, not an attribution to the policy. Grasp is not inferred.',29)
    text(48,947,'Continuous episodes at real time; ended panels explicitly hold their final frame.',28)
    text(48,994,'Ideal posed RGB-D; native rendering. Body-origin success; mesh containment not certified.',26)
    bg.save(out/'background.png')
    filters=[];last='0:v'
    for i,label in enumerate(_SCOPE_LABELS):
        x=48+i*616;row=pair[label]['result'];stream='panel'+str(i);new='over'+str(i)
        filters.append(f'[{i+1}:v]trim=start_frame=1,setpts=PTS-STARTPTS,scale=592:296,tpad=stop_mode=clone:stop_duration=30[{stream}]')
        filters.append(f'[{last}][{stream}]overlay={x}:366[{new}]');last=new
        caption=out/(label+'.txt');caption.write_text(f"{'SUCCESS' if row['success'] else 'TASK FAILURE'} / {row['ticks']} actions")
        new='status'+str(i);filters.append(f'[{last}]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile={caption}:fontsize=28:fontcolor=white:x={x}:y=690[{new}]');last=new
        held=out/(label+'_held.txt');held.write_text('Episode ended; final frame held')
        new='held'+str(i);filters.append(f"[{last}]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile={held}:fontsize=24:fontcolor=0xF7BB70:x={x}:y=733:enable='gte(t,{row['ticks']/20})'[{new}]");last=new
    script=out/'filters.txt';script.write_text(';\n'.join(filters));video=out/'scope_diagnostic_30s_1080p.mp4'
    args=['ffmpeg','-v','error','-nostdin','-loop','1','-i',str(out/'background.png')]
    for label in _SCOPE_LABELS:args+=['-i',pair[label]['video_path']]
    args+=['-filter_complex_threads','2','-filter_complex_script',str(script),'-map','['+last+']','-t','30','-r','30','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video)]
    (out/'render_command.json').write_text(json.dumps(args,indent=2));subprocess.run(args,check=True)
    subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(video),'-f','null','-'],check=True)
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    v=next(s for s in metadata['streams'] if s['codec_type']=='video')
    if (v['width'],v['height'],int(v['nb_frames']))!=(1920,1080,900) or float(metadata['format']['duration'])!=30:raise ValueError('rendered profile differs')
    subprocess.run(['ffmpeg','-v','error','-ss','24','-i',str(video),'-frames:v','1','-n',str(out/'poster.png')],check=True)
    for label,entry in pair.items():
        copy=out/(label+'_continuous.mp4');shutil.copyfile(entry['video_path'],copy)
        if sha256_file(copy)!=entry['video_sha256']:raise ValueError('continuous source copy differs')
    (out/'scope_diagnostic.srt').write_text('1\n00:00:00,000 --> 00:00:30,000\nFirst declared TEST reset; one-instance scope diagnostic. Same policy engine. Continuous episodes; ended panels hold. Measured robot-destination interference in L1. No aggregate or preservation claim.\n')
    manifest=dict(state='TEST_SINGLE_INSTANCE_DIAGNOSTIC',diagnosis_path=str(Path(diagnosis_path).resolve()),diagnosis_sha256=sha256_file(diagnosis_path),
        selection_rule=diagnosis['planned_selection'],pair=pair,producer_sha256=sha256_file(__file__),
        editing='Only initial static frame removed; every action frame remains continuous. 20-to-30fps resampling; explicitly hold ended panels; no splice.',
        action_time_s={k:e['result']['ticks']/20 for k,e in pair.items()},source_resolution=[512,256],master_resolution=[1920,1080],duration_s=30,
        preservation_claim=False,aggregate_claim=False,artifacts={p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
    (out/'demo_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'qa.json').write_text(json.dumps(dict(status='PASS',decoded_all_frames=True,frames=900,size=[1920,1080],duration_s=30,source_copies_byte_exact=True,full_demo_ready=False),indent=2))
    return str(video)
