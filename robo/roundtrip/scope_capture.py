"""Fixed DEV cabinet-support acquisition through the canonical capture producer."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robo.roundtrip.capture_native import NATIVE_CAMERAS
from robo.roundtrip.capture import CaptureError, validate_camera


def cabinet_view_plan(camera_poses):
    """Camera-only positions/pitches; no fixture/target pose or visibility oracle."""
    if set(camera_poses)!=set(NATIVE_CAMERAS):
        raise CaptureError('fixed native camera configuration required')
    frames=[]
    for name in NATIVE_CAMERAS:
        validate_camera(np.eye(3),camera_poses[name],width=1,height=1)
        for pitch in (0.,25.):
            for x in (-.12,0.,.12):
                T=np.asarray(camera_poses[name],float).copy()
                T[:3,3]+=T[:3,:3] @ np.asarray([x,0.,0.])
                T[:3,:3]=T[:3,:3] @ Rotation.from_euler('x',pitch,degrees=True).as_matrix()
                frames.append({'frame_id':f'f{len(frames):06d}','split':'train',
                               'camera':{'native_name':name,'T_world_from_camera':T.tolist()}})
    for name in NATIVE_CAMERAS:
        T=np.asarray(camera_poses[name],float).copy()
        T[:3,3]+=T[:3,:3] @ np.asarray([.06,-.18,0.])
        T[:3,:3]=T[:3,:3] @ Rotation.from_euler('x',12.5,degrees=True).as_matrix()
        frames.append({'frame_id':f'f{len(frames):06d}','split':'test',
                       'camera':{'native_name':name,'T_world_from_camera':T.tolist()}})
    return frames


def main(argv=None):
    from robo.roundtrip.matrix import acquire_instance,enumerate_plan
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args(argv);config=json.loads(Path(args.config).read_text())
    slots,_=enumerate_plan(config)
    if len(slots)!=1 or slots[0]['task_id']!='PickPlaceCounterToCabinet' or config['split']!='development':
        raise ValueError('bounded single predeclared DEV Cabinet instance required')
    result=acquire_instance(config,slots[0]['instance_slot_id'],args.out,capture_view_plan=cabinet_view_plan)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
