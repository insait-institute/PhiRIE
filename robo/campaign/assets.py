"""Method-neutral canonical asset -> existing PhiRoom native-import contract.

Uses baseline-estimated placement, never reference placement. The existing
RoboCasa importer uses a common box inertia approximation; this is therefore an
ASSET-CONTROLLED comparison, not a bit-exact native physics reproduction.
"""
from pathlib import Path
import shutil
import numpy as np
from .core import load, receipt, save


def factor_affine(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        raise ValueError('finite affine world-from-asset transform required')
    if np.linalg.det(T[:3,:3])<=0:raise ValueError('right-handed positive asset transform required')
    U,_,V=np.linalg.svd(T[:3,:3]);R=U@V;S=R.T@T[:3,:3]
    rigid=np.eye(4);rigid[:3,:3]=R;rigid[:3,3]=T[:3,3]
    return rigid,S


def bridge(config,inputs,out):
    import trimesh
    out=Path(out);meta=load(inputs['placement'])
    if meta.get('source')!='method_estimated' or meta.get('uses_reference_pose') is not False:
        raise ValueError('cannot use evaluator reference to place reconstructed assets')
    T,S=factor_affine(meta['T_world_from_asset'])
    mesh=trimesh.load(inputs['mesh'],process=False,force='mesh')
    mesh.vertices=np.asarray(mesh.vertices)@S.T
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)==0:raise ValueError('invalid visual asset')
    mesh.export(out/'mesh_sim.obj')
    collision=out/'collision';collision.mkdir()
    if config.get('collision_mode')=='preserve_supplied':
        parts=[]
        for key in config['collision_inputs']:
            part=trimesh.load(inputs[key],process=False,force='mesh');part.vertices=np.asarray(part.vertices)@S.T
            parts.append(part)
    elif config.get('collision_mode')=='shared_coacd':
        import coacd
        parts=[trimesh.Trimesh(v,f,process=False) for v,f in coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),**config.get('coacd',{}))]
    else:raise ValueError('declare preserved baseline collision or shared-decomposition treatment')
    if not parts:raise ValueError('empty collision decomposition')
    for i,part in enumerate(parts):
        if not(part.is_watertight and part.is_convex and part.volume>0):raise ValueError('supplied collision fails unchanged native import predicate')
        part.export(collision/f'part_{i:03d}.obj')
    physics=load(inputs['physics'])
    if not np.isfinite([float(physics['mass_kg']),float(physics['friction'])]).all() or float(physics['mass_kg'])<=0 or float(physics['friction'])<0:raise ValueError('invalid physical parameters')
    save(out/'physics.json',physics)
    save(out/'aligned.json',{'T':T.tolist(),'scale':1.0,'rejected':False,
        'world_dims':np.ptp(mesh.vertices,axis=0).tolist(),'source_method':config['method_label']})
    save(out/'asset.json',{'method_label':config['method_label'],'inputs':{k:receipt(v) for k,v in inputs.items()},
        'collision_mode':config['collision_mode'],'linear_scale_baked_once':True,
        'T_body_from_asset_linear':S.tolist(),'body_pose':T.tolist(),
        'scope':'asset-controlled common native importer; not native-end-to-end physics',
        'native_controls':'NOT_RUN','object_dir':str(out),
        'output_assets':{str(p.relative_to(out)):receipt(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    return {'asset':out/'asset.json','mesh':out/'mesh_sim.obj','placement':out/'aligned.json','physics':out/'physics.json'}
