"""A native evidence film assembled from a complete, generated release.

The two policy segments remain continuous. Evidence selection uses construction
records and declared roster order, never a search for successful policy clips.
"""
import json
from pathlib import Path
import shutil
import subprocess
from agents.orchestrator.artifact import sha256_file
from interface.demo_native_progress import native_tier, released_video, resolve_pair, render as render_pair


def repair_playback(first, episode):
    # Fixed by split, never selected to favor a successful episode or omit actions.
    speed = 2 if native_tier(first) == 'TEST' else 1
    seconds = episode['ticks'] / 20 / speed
    if not 0 < seconds <= 24:
        raise ValueError('whole repair episode does not fit; never truncate or select another outcome')
    return speed, seconds


def result_note(counts, tier):
    by_method = {r['method']: r for r in counts}
    a, b = by_method['B4_ROOM_REPAIR_NATIVE'], by_method['B3_AGENT_NATIVE']
    if (a['executed'] * b['planned'] < b['executed'] * a['planned'] and
            a['successes_observed'] * b['planned'] <= b['successes_observed'] * a['planned']):
        return f'B4 loses coverage and does not improve overall success over B3 in this {tier} cohort.'
    return 'Coverage and native success are separate; this comparison does not establish preservation.'


def retry_evidence(pool, pool_path=None, *, source_files=None, checked=None):
    """Bind the declared parent, executed registration and selected B3 bytes."""
    import numpy as np
    initial={r['proposal_id']:r for r in pool['initial_candidates']}
    retry=pool['retry_candidate'];rows={r['native_method']:r for r in pool['outcomes']}
    if (not retry or retry['proposal_id'] in initial or
            len(retry.get('parent_proposal_ids',[]))!=1 or
            retry['parent_proposal_ids'][0] not in initial):
        raise ValueError('new retry proposal with exactly one known initial parent required')
    parent=initial[retry['parent_proposal_ids'][0]]
    b3=rows['B3_AGENT_NATIVE']
    if (retry.get('tool')!='registration_retry' or not b3['retry_invoked'] or
            b3['selected_proposal_id']!=retry['proposal_id']):
        raise ValueError('first declared object did not select its genuine registration retry')
    if pool_path is None:
        raise ValueError('actual pool path is required for retry artifact validation')
    root=Path(pool_path).resolve().parent
    if json.loads(Path(pool_path).read_text())!=pool:
        raise ValueError('retry pool differs from its source file')
    used={}
    def read_bound(path,digest=None):
        path=Path(path)
        if checked is not None:checked(path,digest)
        actual=sha256_file(path)
        if digest is not None and actual!=digest:raise ValueError('retry artifact changed: '+str(path))
        used[str(path.resolve())]=actual
        return json.loads(path.read_text())
    def selected(row):
        recorded=Path(row['object_dir']);directory=root/recorded.name
        # Empty-root constructors record /output paths. Resolve only the exact
        # selected directory next to this sealed pool, never an arbitrary tree.
        if (not recorded.name.startswith('selected_') or
                not (recorded.parent.resolve()==root or str(recorded.parent)=='/output/selection')):
            raise ValueError('selected retry directory is outside its pool')
        hashes=row['artifact_hashes']
        if not {'aligned.json','mesh_sim.ply','mesh_sim.obj','physics.json'}<=set(hashes):
            raise ValueError('selected retry artifact closure is incomplete')
        for name,digest in hashes.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe retry artifact path')
            file=directory/name
            if file.is_symlink() or sha256_file(file)!=digest:raise ValueError('selected retry artifact changed')
            if checked is not None:checked(file,digest)
            used[str(file.resolve())]=digest
        return directory,read_bound(directory/'aligned.json',hashes['aligned.json'])
    parents=[r for r in pool['outcomes'] if r.get('selected_proposal_id')==parent['proposal_id']]
    if not parents:raise ValueError('declared retry parent lacks materialized artifact proof')
    parent_dir,parent_alignment=selected(parents[0])
    for row in parents[1:]:
        directory,alignment=selected(row)
        if directory!=parent_dir or row['artifact_hashes']!=parents[0]['artifact_hashes']:
            raise ValueError('ambiguous declared retry parent artifacts')
    retry_dir,selected_alignment=selected(b3)
    if retry_dir==parent_dir:raise ValueError('unchanged parent directory is not a retry artifact')
    registration=read_bound(root/'registration_retry/registration.json')
    evidence=read_bound(root/'registration_retry/evidence.json')
    if (evidence.get('producer')!='agents.orchestrator.runtime.align_and_probe' or
            evidence.get('producer_commit')!=pool['source_commit'] or
            evidence.get('raw_values')!=retry['evidence'] or
            evidence.get('input_hashes',{}).get('mesh')!=parents[0]['artifact_hashes']['mesh_sim.ply']):
        raise ValueError('retry registration action/evidence does not bind its declared parent')
    if (registration.get('source_up_hypothesis') not in ('+x','-x','+y','-y','+z','-z') or
            registration.get('symmetric_clipped_registration_residual_m')!=retry['evidence'].get('symmetric_clipped_registration_residual_m')):
        raise ValueError('retry registration action differs from selected evidence')
    transforms=[np.asarray(r['T'],dtype=float) for r in (parent_alignment,registration,selected_alignment)]
    if any(t.shape!=(4,4) or not np.isfinite(t).all() for t in transforms):
        raise ValueError('invalid parent/retry registration transform')
    if np.array_equal(transforms[0],transforms[1]):
        raise ValueError('unchanged parent registration is not a retry')
    if (not np.array_equal(transforms[1],transforms[2]) or
            registration['scale']!=selected_alignment['scale'] or
            any(b3['artifact_hashes'][name]!=parents[0]['artifact_hashes'][name]
                for name in ('mesh_sim.ply','mesh_sim.obj','physics.json'))):
        raise ValueError('selected B3 artifact does not implement its parent registration retry')
    if source_files is not None:source_files.update(used)
    return parent,retry


def choose_repair(planned, diagnostics):
    order=list(dict.fromkeys(r['canonical_instance_id'] for r in planned))
    candidates=[r for r in diagnostics if r['method']=='B4_ROOM_REPAIR_NATIVE'
                and r['accepted'] is True and r['actual_calls']>0]
    if not candidates:raise ValueError('no recorded accepted repair')
    return min(candidates,key=lambda r:order.index(r['canonical_instance_id']))


def repair_slot(planned, repair):
    slots={r['instance_slot_id'] for r in planned
           if r['canonical_instance_id']==repair['canonical_instance_id']}
    if len(slots)!=1:
        raise ValueError('repair canonical identity has no unique declared instance slot')
    slot=next(iter(slots))
    if repair.get('instance_slot_id') not in (None,slot):
        raise ValueError('repair diagnostic changes canonical instance slot')
    return slot


def resolve(release_dir, *, require_complete=True):
    release,first,pair=resolve_pair(release_dir);root=Path(release_dir)
    if require_complete and release['state']!='COMPLETE_BLOCKS':raise ValueError('hero requires complete native release')
    tier=native_tier(first)
    sources={str(Path(v['path']).resolve()):v['sha256'] for v in release['source_lineage'].values()}
    used={}
    def checked(path,digest=None):
        path=Path(path).resolve();expected=digest or sources.get(str(path))
        if expected is None or sha256_file(path)!=expected:raise ValueError('unbound or changed hero source: '+str(path))
        used[str(path)]=expected;return path
    for entry in pair.values():
        checked(entry['video_path'],entry['video_sha256'])
    planned=[json.loads(x) for x in checked(release['source_lineage']['planned_units']['path']).read_text().splitlines()]
    pool_path=next(Path(p) for p in sources if p.endswith('/selection/candidate_pool.json') and first['instance_slot_id'] in p)
    pool=json.loads(checked(pool_path).read_text());parent,retry=retry_evidence(pool,pool_path,source_files=used,checked=checked)
    # Public TRAIN scan only; its identity is already fixed by the released pool.
    from robo.roundtrip.build import read_train
    instance=Path(first['native_config_source']).parent
    public=next(p.parent for p in (instance/'public').glob('*/capture_manifest.json') if sha256_file(p)==pool['capture_manifest_sha256'])
    checked(public/'capture_manifest.json',pool['capture_manifest_sha256'])
    _,train=read_train(public);capture=checked(public/train[0]['rgb'],train[0]['rgb_sha256'])
    previews=[]
    for method in ('B0_FIXED_NATIVE','B1_FIXED_PRIORITY','B3_AGENT_NATIVE'):
        choices=sorted(p for p in sources if first['instance_slot_id'] in p and '/'+method+'/' in p and p.endswith('.png'))
        if not choices:raise ValueError('missing released common-view preview')
        previews.append(checked(choices[0]))
    diagnostic=json.loads((root/'context_diagnostics.json').read_text());repair=choose_repair(planned,diagnostic)
    slot=repair_slot(planned,repair)
    manifests=[p for p in sources if slot in p and p.endswith('/B4/build_manifest.json')]
    if len(manifests)!=1:raise ValueError('ambiguous repair manifest')
    repair_manifest=checked(manifests[0]);m=json.loads(repair_manifest.read_text())
    pictures=[checked(repair_manifest.parent/name,m['synchronized_diagnostic_frames']['files'][name]) for name in ('before_collision.png','after_collision.png')]
    episodes=[]
    for name,v in release['source_lineage'].items():
        if not name.endswith('_result_path'):continue
        row=json.loads(checked(v['path']).read_text())
        if (row.get('canonical_instance_id')==repair['canonical_instance_id'] and
                row.get('controller_method')=='B4_ROOM_REPAIR_NATIVE' and row.get('reset_id')=='r0'):
            episodes.append((name,row))
    if len(episodes)!=1:raise ValueError('missing or ambiguous first-reset repair episode')
    episode_source,episode=episodes[0]
    if not episode['executed'] or episode['error'] is not None or episode.get('video_error') is not None:
        raise ValueError('selected repair episode/video incomplete; do not select another outcome')
    video=Path(episode['video_path'])
    video_sha=released_video(release['source_lineage'],episode_source,video)
    checked(video,video_sha)
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)],text=True))
    stream=next(s for s in meta['streams'] if s['codec_type']=='video')
    if int(stream['nb_frames'])!=episode['ticks']+1 or stream['r_frame_rate']!='20/1':raise ValueError('repair video trace coverage differs')
    playback_speed,playback_seconds=repair_playback(first,episode)
    return dict(release=release,first=first,pair=pair,planned=planned,capture=capture,previews=previews,
        parent=parent,retry=retry,repair=repair,repair_manifest=m,repair_pictures=pictures,repair_episode=episode,sources=used,
        tier=tier,playback_speed=playback_speed,playback_seconds=playback_seconds,
        ready=release['state']=='COMPLETE_BLOCKS')


def render(release_dir,out):
    from PIL import Image,ImageDraw,ImageOps
    from interface.demo_movie import _font
    data=resolve(release_dir);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    root=Path(release_dir);commands=[];tier=data['tier'];slug=tier.lower()
    def run(args):commands.append(args);subprocess.run(args,check=True)
    def canvas(title,subtitle):
        im=Image.new('RGB',(1920,1080),(13,20,31));d=ImageDraw.Draw(im)
        d.text((64,42),'SimAnyRoom / NATIVE '+tier,font=_font(28),fill=(92,216,231))
        d.text((64,95),title,font=_font(46),fill='white');d.text((64,158),subtitle,font=_font(26),fill=(210,220,232))
        d.text((64,1024),f'Target-only reconstruction; native room retained. Binding-specific native success. {tier} evidence.',font=_font(25),fill=(247,187,112))
        return im,d
    def picture(im,path,box):
        x,y,w,h=box;src=ImageOps.contain(Image.open(path).convert('RGB'),(w,h));im.paste(src,(x+(w-src.width)//2,y+(h-src.height)//2))
    def slide(im,name,duration):
        png=out/(name+'.png');im.save(png);video=out/(name+'.mp4')
        run(['ffmpeg','-v','error','-nostdin','-loop','1','-i',str(png),'-t',str(duration),'-r','30','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-n',str(video)])
        return video
    im,d=canvas('Capture the interaction context','Fixed posed RGB-D views; automatic TRAIN masks; held-out images excluded from construction.')
    picture(im,data['capture'],(100,225,1720,740));clips=[slide(im,'01_capture',6)]
    im,d=canvas('Build one shared proposal pool','Same observed object; specialized generators; one frozen object identity.')
    for i,(p,label) in enumerate(zip(data['previews'][:2],['Fixed TRELLIS proposal','Fixed-priority multi-proposal asset'])):
        picture(im,p,(64+i*912,260,880,590));d.text((64+i*912,885),label,font=_font(29),fill='white')
    d.text((64,955),'Released held-out native renders shown for illustration; uniform generated materials.',font=_font(25),fill=(210,220,232))
    clips.append(slide(im,'02_proposals',8))
    im,d=canvas('A genuine registration retry','A new transform and proposal ID; construction evidence drives the decision.')
    picture(im,data['previews'][2],(70,245,1100,680))
    y=280
    for label,row in [('Initial registration',data['parent']),('Bounded retry',data['retry'])]:
        value=row['evidence']['symmetric_clipped_registration_residual_m']*1000
        d.text((1220,y),label,font=_font(28),fill='white');d.text((1220,y+48),f'{value:.2f} mm residual',font=_font(31),fill=(92,216,231));y+=160
    d.text((1220,650),'Construction residual,',font=_font(25),fill='white');d.text((1220,690),'not held-out fidelity.',font=_font(25),fill='white')
    clips.append(slide(im,'03_retry',8))
    r=data['repair'];im,d=canvas('Verify and repair actual native context','First accepted B4 repair in the declared roster; selected without policy outcomes.')
    for i,p in enumerate(data['repair_pictures']):
        picture(im,p,(64+912*i,240,880,575));d.text((64+912*i,205),('Before','After')[i],font=_font(27),fill='white')
    d.text((64,850),'Matched orthographic collision diagnostics; these are not photoreal policy views.',font=_font(27),fill='white')
    d.text((64,910),f"Penetration {r['before_penetration_m']*1000:.3f} → {r['after_penetration_m']*1000:.3f} mm | measured extra actions: {r['actual_calls']}",font=_font(30),fill=(92,216,231))
    clips.append(slide(im,'04_repair',8))
    pair_video=render_pair(release_dir,out/'paired_30s');clips.append(Path(pair_video))
    episode=data['repair_episode'];seconds=data['playback_seconds'];speed=data['playback_speed']
    playback_label='real time' if speed==1 else f'{speed}x playback; entire episode'
    im,d=canvas('Repaired asset in a continuous native rollout',f"B4 | {episode['ticks']} actions | final native result: {'SUCCESS' if episode['success'] else 'TASK FAILURE'} | {playback_label}")
    bg=out/'repair_rollout.png';im.save(bg);video=out/'06_repair_episode.mp4'
    run(['ffmpeg','-v','error','-nostdin','-loop','1','-i',str(bg),'-i',episode['video_path'],
         '-filter_complex_threads','2','-filter_complex',f'[1:v]trim=start_frame=1,setpts=(PTS-STARTPTS)/{speed},scale=1600:800[r];[0:v][r]overlay=160:200[v]',
         '-map','[v]','-t',str(seconds),'-r','30','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-n',str(video)])
    clips.append(video)
    im,d=canvas('Coverage before conditional quality',f'Complete generated {tier} release. Scientific negatives remain visible.')
    counts=json.loads((root/'T2_manipulation.json').read_text());y=270
    for row in counts:
        d.text((80,y),row['method'].replace('_NATIVE',''),font=_font(30),fill='white')
        d.text((800,y),f"success {row['successes_observed']}/{row['planned']} | executed {row['executed']}/{row['planned']}",font=_font(30),fill=(92,216,231));y+=105
    layouts=len({p['layout_id'] for p in data['planned']});instances=len({p['canonical_instance_id'] for p in data['planned']})
    d.text((80,860),f'{instances} independent instances / {layouts} layouts. No population preservation claim.',font=_font(31),fill='white')
    d.text((80,922),result_note(counts,tier),font=_font(29),fill=(247,187,112))
    clips.append(slide(im,'07_results',30-seconds));im.save(out/'poster.png')
    listing=out/'clips.txt';listing.write_text(''.join("file '"+str(p.resolve()).replace("'","'\\''")+"'\n" for p in clips))
    hero=out/f'native_{slug}_90s_1080p.mp4';run(['ffmpeg','-v','error','-nostdin','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart','-n',str(hero)])
    run(['ffmpeg','-v','error','-nostdin','-i',str(hero),'-f','null','-'])
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(hero)],text=True));v=next(s for s in meta['streams'] if s['codec_type']=='video')
    if (v['width'],v['height'],int(v['nb_frames']))!=(1920,1080,2700) or abs(float(meta['format']['duration'])-90)>.1:raise ValueError('hero profile differs')
    loop=out/'static_evidence_loop_8s.mp4';run(['ffmpeg','-v','error','-nostdin','-loop','1','-i',str(out/'poster.png'),'-t','8','-r','30','-an','-c:v','libx264','-threads','2','-crf','0','-pix_fmt','yuv420p','-n',str(loop)])
    for i,frame in enumerate([0,239]):run(['ffmpeg','-v','error','-i',str(loop),'-vf',f'select=eq(n\\,{frame})','-frames:v','1','-n',str(out/f'loop_check_{i}.png')])
    import numpy as np
    if not np.array_equal(np.asarray(Image.open(out/'loop_check_0.png')),np.asarray(Image.open(out/'loop_check_1.png'))):raise ValueError('static loop seam differs')
    shutil.copyfile(pair_video,out/'teaser_30s_1080p.mp4');shutil.copyfile(episode['video_path'],out/'B4_repair_continuous.mp4')
    labels=[(0,6,'Fixed public TRAIN capture'),(6,14,'Shared proposals'),(14,22,'Genuine bounded registration retry'),(22,30,'Actual context repair'),(30,60,'Continuous independent REF and B0 episodes'),(60,60+seconds,'Continuous B4 repaired episode; '+playback_label),(60+seconds,90,f'Complete {tier} results; coverage losses retained')]
    def stamp(t):return f'{int(t)//3600:02}:{int(t)//60%60:02}:{int(t)%60:02},{round(t%1*1000):03}'
    (out/f'native_{slug}.srt').write_text('\n'.join(f'{i+1}\n{stamp(a)} --> {stamp(b)}\n{label}\n' for i,(a,b,label) in enumerate(labels)))
    manifest=dict(state=f'COMPLETE_{tier}_PRESENTATION',full_demo_ready=False,native_track_demo_ready=tier=='TEST',release=str(root.resolve()),release_sha256=sha256_file(root/'release_manifest.json'),
        source_files=data['sources'],selection_rule=f'first declared {tier} pair; first accepted construction repair in roster order, first reset; no policy-outcome search',
        repair_playback_speed=speed,
        segments=labels,commands=commands,loop='static evidence card, byte-exact seam; not a cyclic robot rollout',
        limitations=f'{tier} target-only; oracle native room, uniform generated materials, binding-specific native success; no GS/full-room/preservation claim')
    (out/'demo_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'qa.json').write_text(json.dumps(dict(status='PASS',decoded_all_hero_frames=True,frames=2700,duration_s=90,resolution=[1920,1080],loop_seam_exact=True,full_demo_ready=False,native_track_demo_ready=tier=='TEST'),indent=2))
    return str(hero)
