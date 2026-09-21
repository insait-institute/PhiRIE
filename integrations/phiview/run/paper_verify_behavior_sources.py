"""Verify each prepared RGB against its WDS entry and audit camera coverage."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from agents.core.common import load_colmap_w2c
from physicalview.phiview import save_json

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve();results=[]
for row in json.loads((root/'behavior-roster.json').read_text()):
    scene=root/'behavior-recon/data'/row['scene'];pose=json.loads((scene/'pose-selection.json').read_text())
    images=sorted((scene/'dslr/resized_undistorted_images').glob('*.jpg'));wanted={}
    for path in images:
        episode,start,end,cam=path.stem.split('_')
        assert episode==pose['selected_episode']
        name=f"{row['task']}_episode_{episode}-{start}:{end}.camera_{cam}_initial_rgb.jpg"
        assert Image.open(path).size==(320,180)
        wanted[name]=path
    matches={}
    for shard in pose['source_shards']:
        with tarfile.open(shard) as stream:
            for member in stream:
                if member.name not in wanted:continue
                original=stream.extractfile(member).read();path=wanted[member.name]
                assert original==path.read_bytes(),f'Changed source RGB: {path}'
                matches[member.name]={'file':path.name,'source_shard':shard,'source_entry':member.name,'sha256':hashlib.sha256(original).hexdigest()}
    assert len(matches)==len(images),f'Unmatched source images: {row["scene"]}'
    cameras=load_colmap_w2c(scene/'dslr/colmap/images.txt');assert set(cameras)=={p.name for p in images}
    matrices=np.stack(list(cameras.values()));assert np.isfinite(matrices).all()
    assert np.allclose(np.linalg.det(matrices[:,:3,:3]),1,atol=1e-5)
    origins=np.linalg.inv(matrices)[:,:3,3]
    result={'scene':row['scene'],'source_images_verified':len(images),'native_resolution':[320,180],
            'selected_episode':pose['selected_episode'],'unique_episode_search_count':len(pose['searched_unique_episodes']),
            'camera_origin_span_m':np.ptp(origins,axis=0),'frames':list(matches.values()),
            'boundary':'Input integrity and calibrated-view coverage; no PhiView feature execution or paper-quality approval'}
    results.append(result);save_json(root/'behavior-source-verification.json',results)
    print(row['scene'],len(images),'source RGB entries byte-identical',flush=True)
