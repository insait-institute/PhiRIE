import numpy as np
from types import SimpleNamespace as NS
import pytest
from robo.roundtrip.scorer import predicate_components


def test_cabinet_dispatch_keeps_upstream_full_bbox_defaults(monkeypatch):
    OU=pytest.importorskip('robocasa.utils.object_utils')
    calls=[]
    def inside(*args,**kwargs):
        calls.append((args[1:],kwargs));return True
    monkeypatch.setattr(OU,'obj_inside_of',inside)
    monkeypatch.setattr(OU,'gripper_obj_far',lambda env:False)
    cab=object();env=NS(sim=NS(data=NS(body_xpos=np.zeros((1,3)),site_xpos=np.array([[.2,0,0]]))),obj_body_id={'obj':0},robots=[NS(eef_site_id={'right':0})],cab=cab)
    r=predicate_components(env,'PickPlaceCounterToCabinet')
    assert calls==[(('obj',cab),{})] # no partial_check=True or altered threshold
    assert r['object_bbox_inside_cabinet'] and not r['native_task_success']
    monkeypatch.setattr(OU,'gripper_obj_far',lambda env:True)
    assert predicate_components(env,'PickPlaceCounterToCabinet')['native_task_success']
