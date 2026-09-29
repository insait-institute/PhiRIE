"""Simulation moves observed geometry without pretending missing background is filled."""
import numpy as np
import pytest
import torch

from physicalview.phiview_scene import GaussianScene

pytestmark = pytest.mark.backend


def scene():
    s = GaussianScene.__new__(GaussianScene)
    s.raw = {'means':torch.tensor([[0.,0,0],[1.,0,0],[10.,0,0]]),
             'scales':torch.ones(3,3), 'quats':torch.tensor([[1.,0,0,0]]*3),
             'opacities':torch.ones(3), 'sh':torch.zeros(3,1,3), 'sh_degree':0}
    s.labels=torch.tensor([0,1,0]);s.removed=s.labels>0;s.clean=None
    s.names=['obj_00'];s.ids={'obj_00':1};s.indices={'obj_00':torch.tensor([1])}
    s.variants={'obj_00':'original'};s.prompt_backgrounds={}
    return s


def test_stationary_simulation_preserves_all_observed_gaussians():
    s=scene();gs,labels=s.compose('simulation',None,{'obj_00'})
    assert len(gs['means'])==3
    assert sorted(gs['means'][:,0].tolist())==[0,1,10]
    assert sorted(labels.tolist())==[0,0,1]


def test_moving_object_does_not_leave_duplicate_at_original_location():
    s=scene();before=s.raw['means'].clone();T=np.eye(4);T[0,3]=2
    gs,labels=s.compose('simulation','obj_00',{'obj_00'},{'obj_00':T})
    assert gs['means'][labels==1].tolist()==[[3.,0,0]]
    assert sorted(gs['means'][labels==0][:,0].tolist())==[0,10]
    assert torch.equal(s.raw['means'],before)


@pytest.mark.parametrize('mode',['clean_selected','clean_all'])
def test_clean_views_still_require_real_background_completion(mode):
    with pytest.raises(ValueError,match='inpainted background'):
        scene().compose(mode,'obj_00',{'obj_00'})
