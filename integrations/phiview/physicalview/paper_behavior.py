"""Keep one consistent BEHAVIOR object state and preserve native RGB resolution.

Camera registration and state filtering use dataset-provided mesh trajectories;
this is GT-assisted pose evidence, even though object discovery uses AUTO masks.
"""
import argparse,json
from pathlib import Path
import numpy as np


def select_consistent_clips(registered, pose_to_matrix, translation_tol=.02, rotation_tol_degrees=5.):
    episodes={}
    for clip in registered:episodes.setdefault(clip['ep'],[]).append(clip)
    trials=[]
    for episode,clips in sorted(episodes.items()):
        def clip_time(clip):
            start=clip['clip_id'].rsplit('-',1)[-1].split(':',1)[0]
            return (int(start) if start.isdigit() else float('inf'),clip['clip_id'])
        clips=sorted(clips,key=clip_time)
        poses_by_clip={c['clip_id']:{k:c['D']@pose_to_matrix(np.asarray(t)[0]) for k,t in c['traj'].items()} for c in clips}
        references=[]
        for reference in clips:
            poses=poses_by_clip[reference['clip_id']];keep=[];audit=[]
            for clip in clips:
                current_poses=poses_by_clip[clip['clip_id']];shared=sorted(set(poses)&set(current_poses))
                if shared:
                    current=np.stack([current_poses[k] for k in shared]);base=np.stack([poses[k] for k in shared])
                    delta=float(np.linalg.norm(current[:,:3,3]-base[:,:3,3],axis=1).max())
                    rotations=current[:,:3,:3]@base[:,:3,:3].transpose(0,2,1)
                    angle=float(np.rad2deg(np.arccos(np.clip((np.trace(rotations,axis1=1,axis2=2)-1)/2,-1,1))).max())
                else:delta=angle=float('inf')
                accepted=len(shared)>=3 and delta<=translation_tol and angle<=rotation_tol_degrees
                if accepted:keep.append(clip)
                audit.append({'clip':clip['clip_id'],'shared_meshes':len(shared),'max_translation_m':delta if shared else None,'max_rotation_degrees':angle if shared else None,'kept':accepted})
            references.append({'episode':episode,'reference':reference['clip_id'],'kept':keep,'audit':audit})
        selected=max(references,key=lambda t:sum(len(c['cam_frames']) for c in t['kept']))
        selected['reference_search']=[{'reference':t['reference'],'compatible_views':sum(len(c['cam_frames']) for c in t['kept'])} for t in references]
        trials.append(selected)
    if not trials:raise ValueError('No registered BEHAVIOR clips')
    best=max(trials,key=lambda t:sum(len(c['cam_frames']) for c in t['kept']))
    report={'selected_episode':best['episode'],'reference_clip':best['reference'],
        'registered_clips':len(registered),'selected_clips':len(best['kept']),
        'selected_camera_frames':sum(len(c['cam_frames']) for c in best['kept']),
        'translation_tolerance_m':translation_tol,'rotation_tolerance_degrees':rotation_tol_degrees,
        'selection':'consistent initial sampled state of one episode, maximizing observed views before reconstruction',
        'pose_source':'dataset mesh trajectories, GT-assisted',
        'episodes':[{k:v for k,v in t.items() if k!='kept'} for t in trials]}
    return best['kept'],report


def main():
    from oracle import capture_generator as source
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--scene-name',required=True);ap.add_argument('--root',required=True);ap.add_argument('--episodes',type=int,default=40);ap.add_argument('--max-frames',type=int,default=120);ap.add_argument('--tasks-config',required=True);ap.add_argument('--static-clips-only',action='store_true');ap.add_argument('--shard-count',type=int,default=4);a=ap.parse_args()
    scene=Path(a.root)/'data'/a.scene_name;scene.mkdir(parents=True,exist_ok=True)
    original_register=source._register_clips_to_reference
    original_clip_ids=source._clip_ids_for_task
    primary=source.find_task_shards(a.task,source.DEFAULT_WDS_ROOT,a.tasks_config)
    candidates=list(primary)
    for path in primary:
        candidates.extend(sorted(path.parent.glob('*-'+path.name.rsplit('-',1)[-1])))
    shards=[];clip_ids={}
    for path in dict.fromkeys(candidates):
        ids=original_clip_ids(path,a.task)
        if ids:shards.append(path);clip_ids[path]=ids
        if len(shards)>=max(1,a.shard_count):break
    searched_episodes=sorted({cid.split('_episode_')[1].split('-')[0] for ids in clip_ids.values() for cid in ids})[:a.episodes]
    selected_episodes=set(searched_episodes)
    source.find_task_shards=lambda *args,**kwargs:shards
    source._clip_ids_for_task=lambda path,task:[cid for cid in clip_ids[path] if cid.split('_episode_')[1].split('-')[0] in selected_episodes]
    def register(clips):
        unique={}
        for clip in clips:
            old=unique.get(clip['clip_id'])
            if old is not None and [c['jpg'] for c in old['cam_frames']]!=[c['jpg'] for c in clip['cam_frames']]:
                raise ValueError('Conflicting RGB bytes for a duplicated WDS clip ID')
            unique[clip['clip_id']]=clip
        registered=original_register(list(unique.values()));kept,report=select_consistent_clips(registered,source.pose7_to_mat)
        report.update(source_shards=list(map(str,shards)),searched_unique_episodes=searched_episodes,
                      duplicate_clip_ids_removed=len(clips)-len(unique))
        (scene/'pose-selection.json').write_text(json.dumps(report,indent=2))
        if report['selected_camera_frames']<8:
            raise ValueError('Fewer than eight posed frames of one consistent scene state; do not fuse different object states to manufacture coverage')
        return kept
    source._register_clips_to_reference=register
    # WDS stores an initial RGB/depth image, not RGB video. Later motion within
    # the clip does not invalidate that first image: compare its actual object
    # poses against the same-state filter instead of discarding the entire clip.
    # The upstream loop counts episode/shard pairs, so let every chosen episode
    # contribute clips from every shard; selection above bounds UNIQUE episodes.
    source.extract_task(a.task,a.scene_name,a.root,episodes=max(1,len(searched_episodes)*len(shards)),static_only=a.static_clips_only,max_frames=a.max_frames,tasks_config=a.tasks_config,upscale=1)
    calibration=json.loads((scene/'dslr/nerfstudio/transforms_undistorted.json').read_text())
    (scene/'paper-source-resolution.json').write_text(json.dumps({'width':calibration['w'],'height':calibration['h'],'upscale':1,'original_encoded_rgb_bytes_preserved':True,'static_clips_only':a.static_clips_only,'episode_search_budget':a.episodes,'source_shards':list(map(str,shards)),'initial_rgb_only':True,'source':'PointWorld-BEHAVIOR WDS initial RGB frames'},indent=2))

if __name__=='__main__':main()
