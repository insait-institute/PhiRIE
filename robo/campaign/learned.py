"""Small trainable state-conditioned appearance residual, optional second stage.

Training pairs must describe the SAME physical geometry/state/camera. Do not
supervise a divergent reconstruction trajectory with reference frames by index.
This model is distinct from the frozen NVIDIA Harmonizer backbone.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .core import digest, load, receipt, rows, save


class StateResidual(nn.Module):
    """raw3 + teacher3 + depth1 + normals3 + confidence1 + warp3 + valid1 + protect1."""
    def __init__(self, width=32, bound=32/255):
        super().__init__(); self.bound = float(bound)
        self.net = nn.Sequential(nn.Conv2d(16,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1,dilation=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=2,dilation=2),nn.SiLU(),nn.Conv2d(width,3,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 16: raise ValueError('expected Bx16xHxW input')
        delta = self.bound*torch.tanh(self.net(x))
        protection = x[:,15:16].clamp(0,1)
        return (x[:,:3]+delta*(1-protection)).clamp(0,1)


def validate_pairs(train, dev):
    seen = set(); groups = []
    for roster in (train, dev):
        group = set()
        for r in roster:
            if r['sample_id'] in seen: raise ValueError('duplicate train/dev sample')
            seen.add(r['sample_id']); group.add(r['scene_group'])
            if r['input_state_sha256'] != r['target_state_sha256'] or r.get('physical_geometry_equal') is not True:
                raise ValueError('harmonization training pair changes physical state/geometry')
            if r['split'] not in ('train','dev'): raise ValueError('TEST/DEMO forbidden for training')
        groups.append(group)
    if groups[0] & groups[1]: raise ValueError('train/dev scene-family overlap')
    if not train or not dev: raise ValueError('nonempty train/dev pairs required')
    if any(r['split']!='train' for r in train) or any(r['split']!='dev' for r in dev):
        raise ValueError('wrong split in pair file')


def sample(r, device):
    from .core import checked_path
    a = np.load(checked_path(r['arrays']),allow_pickle=False)
    x = np.asarray(a['inputs'],np.float32); target = np.asarray(a['target'],np.float32)
    if x.ndim!=3 or x.shape[0]!=16 or target.shape!=(3,*x.shape[1:]) or not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError('invalid paired arrays')
    if np.any((target<0)|(target>1)) or np.any((x[:6]<0)|(x[:6]>1)):
        raise ValueError('RGB must be in [0,1]')
    return torch.from_numpy(x).unsqueeze(0).to(device),torch.from_numpy(target).unsqueeze(0).to(device)


def fit(config_path, out):
    c = load(config_path); train,dev = rows(c['train_pairs']),rows(c['dev_pairs']); validate_pairs(train,dev)
    torch.manual_seed(int(c.get('seed',2027))); device = c.get('device','cuda')
    model = StateResidual(int(c.get('width',32)),float(c.get('bound',32/255))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(c.get('lr',1e-4)))
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    save(out/'training_manifest.json',{'config':receipt(config_path),'train':receipt(c['train_pairs']),
         'dev':receipt(c['dev_pairs']),'test_access':False,'architecture':'StateResidual16',
         'dataset_hash':digest(train+dev)})
    history = []; best = float('inf')
    for epoch in range(int(c.get('epochs',10))):
        model.train(); losses = []
        for r in train:
            x,target = sample(r,device); prediction = model(x); editable = 1-x[:,15:16]
            denom = (editable.sum()*3).clamp_min(1)
            loss = ((prediction-target).abs()*editable).sum()/denom
            # Gradient residual on the same fixed state. It is not a geometry guarantee.
            grad = (prediction[...,1:]-prediction[...,:-1])-(target[...,1:]-target[...,:-1])
            loss = loss+float(c.get('gradient_weight',.1))*grad.abs().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); values=[]
        with torch.no_grad():
            for r in dev:
                x,target=sample(r,device); values.append(float((model(x)-target).abs().mean()))
        val=float(np.mean(values)); history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_l1':val})
        if val<best:
            best=val
            torch.save({'state_dict':model.state_dict(),'width':c.get('width',32),'bound':c.get('bound',32/255),
                        'epoch':epoch,'dev_l1':val,'train_manifest':receipt(out/'training_manifest.json')},out/'best.pt')
    save(out/'history.json',history)
    return {'checkpoint':str(out/'best.pt'),'epochs':len(history),'best_dev_l1':best}


def predict(config, raw, teacher, fields, protect, warped=None, valid=None):
    """Inference for a frozen StateResidual, keeping the same 16-channel contract."""
    from .core import local_model
    path=local_model(config)
    device=config.get('device','cuda')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=StateResidual(checkpoint['width'],checkpoint['bound']).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    h,w=raw.shape[:2]
    depth=np.asarray(fields['depth'],np.float32)
    normals=np.asarray(fields['normals'],np.float32)
    confidence=np.asarray(fields['confidence'],np.float32)
    if normals.shape!=(h,w,3) or depth.shape!=(h,w) or confidence.shape!=(h,w):
        raise ValueError('depth, normals and confidence must use raw camera grid')
    # This normalization is fixed for training and inference, not fit per image.
    depth=np.where(np.isfinite(depth),np.clip(depth/10.,0,1),0)
    warp=np.zeros_like(raw) if warped is None else warped
    valid=np.zeros((h,w),bool) if valid is None else valid
    channels=np.concatenate([raw/255.,teacher/255.,depth[...,None],normals,
        confidence[...,None],warp/255.,valid[...,None],protect[...,None]],axis=-1)
    if channels.shape!=(h,w,16) or not np.isfinite(channels).all():raise ValueError('invalid network input')
    tensor=torch.from_numpy(channels.transpose(2,0,1).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():result=model(tensor)[0].permute(1,2,0).cpu().numpy()
    result=np.rint(result*255).clip(0,255).astype(np.uint8);result[protect]=raw[protect]
    return result
