"""Seed inpainted Gaussians on observed background geometry and local planes.

Source geometry provenance follows the scene contract (GT scan in current
ScanNet++ demos, derived geometry in AUTO scenes). This is not a GT-free claim.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from physicalview.phiview import save_json


def make_seeds(names):
    import open3d as o3d
    from scipy.spatial.transform import Rotation
    from PIL import Image
    from agents.core import common as C
    o3d.utility.random.seed(0)
    mesh=o3d.io.read_triangle_mesh(str(C.PIPELINE_MESH_PLY));vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
    instances={g['object_id']:g for g in C.load_instances()}
    objects={f"obj_{m['index']:02d}":m for m in json.loads((C.OUT/'objects/objects.json').read_text())}
    removed=np.zeros(len(vertices),bool)
    for name in names:removed[instances[objects[name]['gt_object_id']]['vert_idx']]=True
    faces=faces[~removed[faces].any(axis=1)]
    background=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces));background.compute_triangle_normals()
    raycaster=o3d.t.geometry.RaycastingScene();raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(background))
    K,W,H,_=C.load_intrinsics();arrays={k:[] for k in ['means','quats','scales','rgb']};slices={};cursor=0;reports=[]
    for name in names:
        folder=C.OUT/'inpaint'/name;meta=objects[name];center=np.asarray(meta['centroid']);radius=max(.5,float(max(meta['extent']))*2)
        if not (folder/'views.json').exists():raise ValueError(f'No views for {name}')
        pl=json.loads((folder/'plane.json').read_text());planes=[(np.asarray(pl['normal']),-float(np.dot(pl['normal'],pl['origin'])))]
        local=vertices[(np.linalg.norm(vertices-center,axis=1)<radius)&~removed]
        # Nearby observed wall/support planes supply a bounded extrapolation when
        # the scan itself has a hole behind the removed foreground object.
        for _ in range(3):
            if len(local)<100:break
            pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local));eq,inliers=pc.segment_plane(.012,3,500)
            if len(inliers)<100:break
            n=np.asarray(eq[:3]);n/=np.linalg.norm(n);b=float(eq[3])/np.linalg.norm(eq[:3]);planes.append((n,b))
            local=np.delete(local,inliers,axis=0)
        pts=[];normals=[];colors=[];surface_count=0;extrapolated_count=0
        for i,view in enumerate(json.loads((folder/'views.json').read_text())):
            mask=np.asarray(Image.open(folder/f'mask_{i}.png'))>127;rgb=np.asarray(Image.open(folder/f'inpainted_{i}.png').convert('RGB'))/255.
            yy,xx=np.where(mask);take=(xx%3==0)&(yy%3==0);yy,xx=yy[take],xx[take]
            if not len(xx):continue
            w2c=np.asarray(view['w2c']);c2w=np.linalg.inv(w2c);origin=c2w[:3,3]
            ray=np.stack([(xx+.5-K[0,2])/K[0,0],(yy+.5-K[1,2])/K[1,1],np.ones(len(xx))],1)@c2w[:3,:3].T;ray/=np.linalg.norm(ray,axis=1)[:,None]
            rays=np.concatenate([np.tile(origin,(len(ray),1)),ray],1).astype(np.float32)
            result=raycaster.cast_rays(o3d.core.Tensor(rays));dist=result['t_hit'].numpy();ns=result['primitive_normals'].numpy()
            hit=origin+ray*np.where(np.isfinite(dist),dist,0)[:,None]
            good=np.isfinite(dist)&(dist>.05)&(np.linalg.norm(hit-center,axis=1)<radius)
            surface_count+=int(good.sum())
            for normal,b in planes:
                denom=ray@normal;ds=-(origin@normal+b)/np.where(np.abs(denom)>1e-6,denom,np.nan)
                p=origin+ray*ds[:,None]
                # Extrapolated background must be beyond the target's center
                # along the ray, allowing a small depth tolerance for support.
                minimum=np.linalg.norm(center-origin)-max(meta['extent'])*.7
                accept=(~good)&np.isfinite(ds)&(ds>max(.05,minimum))&(np.linalg.norm(p-center,axis=1)<radius)
                hit[accept]=p[accept];ns[accept]=normal;good[accept]=True;extrapolated_count+=int(accept.sum())
            nn=np.linalg.norm(ns,axis=1);good&=nn>.5
            pts.append(hit[good]);normals.append(ns[good]/nn[good,None]);colors.append(rgb[yy[good],xx[good]])
        if not pts or not sum(len(p) for p in pts):raise ValueError(f'No supported background seeds for {name}')
        points=np.concatenate(pts);normal=np.concatenate(normals);color=np.concatenate(colors)
        _,keep=np.unique(np.floor(points/.003).astype(np.int64),axis=0,return_index=True);points=points[keep];normal=normal[keep];color=color[keep]
        axis=np.tile([0.,0.,1.],(len(normal),1));axis[np.abs(normal[:,2])>.9]=[1.,0.,0.]
        u=np.cross(axis,normal);u/=np.linalg.norm(u,axis=1)[:,None];v=np.cross(normal,u)
        q=Rotation.from_matrix(np.stack([u,v,normal],axis=-1)).as_quat()[:,[3,0,1,2]]
        arrays['means'].append(points);arrays['quats'].append(q);arrays['scales'].append(np.tile([.0035,.0035,.0008],(len(points),1)));arrays['rgb'].append(color)
        slices[name]=(cursor,cursor+len(points));cursor+=len(points)
        reports.append({'object':name,'seeds':len(points),'rays_on_observed_mesh':surface_count,'rays_on_extrapolated_local_planes':extrapolated_count,'planes':len(planes),'radius_m':radius})
    output=C.OUT/'inpaint/surface-seeds.npz';np.savez_compressed(output,**{k:np.concatenate(v) for k,v in arrays.items()},slices=json.dumps(slices))
    save_json(C.OUT/'inpaint/surface-seeds.json',{'source_mesh':C.PIPELINE_MESH_PLY,'mesh_sha256':hashlib.sha256(C.PIPELINE_MESH_PLY.read_bytes()).hexdigest(),'objects':reports,'geometry':'observed carved mesh with bounded local-plane extrapolation','input_geometry_provenance':'AUTO derived mesh' if C.env('AUTO')=='1' else 'GT-assisted source mesh','seed_count':cursor})
    print(output,cursor,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--objects',required=True);a=ap.parse_args();make_seeds([f'obj_{int(n):02d}' for n in a.objects.split(',')])
