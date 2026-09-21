"""Distill fixed-state RGB corrections into bounded Gaussian DC appearance.

Input weights are actual alpha-compositing responsibilities from the frozen
renderer, not nearest-splat assignments. Geometry, opacity, scale and rotation
are never optimization variables. View-dependent shadows stay image-space.
"""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,eye,vstack
from scipy.sparse.linalg import lsqr
from .core import load,receipt,save


def solve_residual(pixel, primitive, weights, target, confidence, n_gaussians,
                   *, ridge=.01, max_delta=.125, iterations=500):
    target=np.asarray(target,float);confidence=np.asarray(confidence,float)
    p=np.asarray(pixel);g=np.asarray(primitive);w=np.asarray(weights,float)
    if target.ndim!=2 or target.shape[1]!=3 or confidence.shape!=(len(target),):raise ValueError('Nx3 residual and N confidence required')
    if not np.issubdtype(p.dtype,np.integer) or not np.issubdtype(g.dtype,np.integer):raise ValueError('integer sparse indices required')
    if p.shape!=g.shape or w.shape!=p.shape or p.ndim!=1 or not len(p):raise ValueError('nonempty contribution tuples required')
    if n_gaussians<1 or np.any(p<0) or np.any(p>=len(target)) or np.any(g<0) or np.any(g>=n_gaussians):raise ValueError('contribution index outside fixed source')
    if not all(np.isfinite(a).all() for a in (w,target,confidence)) or np.any(w<0) or np.any((confidence<0)|(confidence>1)):
        raise ValueError('finite nonnegative compositing weights and [0,1] confidence required')
    if ridge<=0 or max_delta<=0 or iterations<1:raise ValueError('positive regularization and solver budget')
    A=coo_matrix((w,(p,g)),shape=(len(target),n_gaussians)).tocsr()
    if np.any(np.asarray(A.sum(1)).reshape(-1)>1.0001):raise ValueError('responsibilities must be compositing weights, not arbitrary splat counts')
    weighted=A.multiply(np.sqrt(confidence)[:,None]).tocsr()
    system=vstack([weighted,np.sqrt(ridge)*eye(n_gaussians,format='csr')],format='csr')
    residual=[];diagnostics=[]
    for c in range(3):
        rhs=np.r_[target[:,c]*np.sqrt(confidence),np.zeros(n_gaussians)]
        result=lsqr(system,rhs,iter_lim=iterations,atol=1e-7,btol=1e-7)
        if result[1] not in (0,1,2):raise RuntimeError('color-cache solver did not converge within declared budget')
        residual.append(result[0]);diagnostics.append({'stop':int(result[1]),'iterations':int(result[2]),'norm':float(result[3])})
    return np.clip(np.stack(residual,1),-max_delta,max_delta),diagnostics


def run(config,inputs,out):
    from plyfile import PlyData
    meta=load(inputs['correspondence_manifest'])
    if meta.get('color_space')!='linear_rgb' or meta.get('same_physical_state') is not True or meta.get('responsibility_source')!='actual_alpha_compositing':
        raise ValueError('same-state linear-RGB true renderer responsibilities required')
    if meta.get('gaussians_sha256')!=receipt(inputs['gaussians'])['sha256']:
        raise ValueError('responsibilities refer to different Gaussian geometry/order')
    if meta.get('fit_split') not in ('train','dev') or meta.get('target_uses_evaluation_reference') is not False:
        raise ValueError('cache fit cannot use TEST reference appearance')
    ply=PlyData.read(inputs['gaussians']);v=ply['vertex'].data
    if not all(k in v.dtype.names for k in ('f_dc_0','f_dc_1','f_dc_2')):raise ValueError('standard SH-DC Gaussian PLY required')
    a=np.load(inputs['contributions'],allow_pickle=False)
    delta,diag=solve_residual(a['pixel_index'],a['gaussian_index'],a['weight'],a['residual_rgb'],a['confidence'],len(v),**config.get('solver',{}))
    before={k:v[k].copy() for k in v.dtype.names if not k.startswith('f_dc_')}
    for c in range(3):v[f'f_dc_{c}']+=delta[:,c]/0.28209479177387814
    for k,values in before.items():
        if not np.array_equal(values,v[k],equal_nan=True):raise AssertionError('non-DC Gaussian attribute changed')
    out=Path(out);ply.write(out/'appearance_cached.ply');np.save(out/'delta_rgb.npy',delta)
    save(out/'cache.json',{'inputs':{k:receipt(v) for k,v in inputs.items()},'solver':diag,
         'changed_attributes':['f_dc_0','f_dc_1','f_dc_2'],'geometry_opacity_exact':True,
         'max_delta_rgb':float(np.abs(delta).max()),'multiview_or_policy_improvement':'NOT_ESTABLISHED'})
    return {'gaussians':out/'appearance_cached.ply','residual':out/'delta_rgb.npy','receipt':out/'cache.json'}
