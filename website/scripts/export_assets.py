"""Export existing PhiRIE evidence for the website; never run model inference.

Run with the control Python environment. Originals remain untouched. All media
derivatives and 3D approximations are recorded in public/provenance.json.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import mujoco

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
DEMO = WORKSPACE / 'results/demo'
PUBLIC = ROOT / 'public'
RECORDS = []
GALLERY = []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(dst, srcs, operation, **kwargs):
    RECORDS.append(dict(file=str(dst.relative_to(PUBLIC)), sha256=sha(dst),
        bytes=dst.stat().st_size, operation=operation,
        sources=[dict(path=str(p.relative_to(WORKSPACE)), sha256=sha(p)) for p in srcs], **kwargs))


def picture(src, name, width=1600):
    dst = PUBLIC / 'media' / (name + '.webp')
    im = Image.open(src).convert('RGB')
    im.thumbnail((width, width * 2), Image.Resampling.LANCZOS)
    im.save(dst, quality=87, method=6)
    record(dst, [src], 'Aspect-preserving downsample and WebP encode; no retouching')
    return 'media/' + dst.name


def item(src, name, title, category, scene, caption):
    url = picture(src, name)
    GALLERY.append(dict(id=name, src=url, title=title, category=category, scene=scene, caption=caption))
    return url


def media():
    q = DEMO / 'qualitative_harmonizer'
    picture(q/'qualitative_results_figure_preview.png', 'paper-qualitative', 2200)
    paper = WORKSPACE/'code/SimAnyRoom'
    picture(paper/'figures/agentic_paired_uncertainty.png', 'paper-uncertainty', 2000)
    picture(q/'with_without_harmonizer/harmonizer_comparison.png', 'paper-teaser', 2200)
    for name, title, scene, caption in [
        ('Q01_27dd4da69e', 'Kitchen & corridor', '27dd4da69e', 'Original Gaussian scene from captured ScanNet++ observations.'),
        ('Q02_7831862f02', 'Shared lounge', '7831862f02', 'Saved 3D Gaussian background completion; training-view diagnostic.'),
        ('Q03_578511c8a9', 'Laboratory', '578511c8a9', 'Saved 3D Gaussian background completion; training-view diagnostic.'),
        ('Q04_45b0dac5e3', 'Bathroom & laundry', '45b0dac5e3', 'Saved 3D Gaussian background completion; training-view diagnostic.'),
        ('Q05_libero_living_room_scene5', 'Robot workspace', 'libero_living_room_scene5', 'Gaussian environment and SAM3D cup with a mesh robot at a recorded scripted state.'),
        ('Q06_libero_study_scene1', 'Study workspace', 'libero_study_scene1', 'Gaussian fit of native synthetic LIBERO observations.')]:
        item(q/'panels'/f'{name}.png', name, title, 'Scenes', scene, caption)
    bottle = DEMO/'FIGURE2_C50_BOTTLE'
    labels = ['Captured scene', 'Selected target', 'Observed surface', 'TRELLIS proposal',
        'ReconViaGen proposal', 'Metric registration', 'Target removed', 'Background completed',
        'Scene reassembled', 'Surface-only removal', 'Geometry + mask removal', 'Object underside',
        'Completed background detail', 'Gaussian appearance · initial', 'Gaussian appearance · moving',
        'Collision geometry · initial', 'Collision geometry · moving']
    for i, title in enumerate(labels, 1):
        item(bottle/'assets'/f'P{i:02}.png', f'bottle-{i:02}', title, 'Construction' if i<14 else 'Motion',
             'c50d2d1d42 / obj_13', 'Paper Figure 2 asset collection. GT-assisted bottle registration; recorded component stages and MuJoCo motion. Residual artifacts retained.')
    base = DEMO/'qualitative/HEADPHONE_CUP_KEYBOARD_SAM3D_TWOHIT_20260915'
    for obj, scene in [('HEADPHONE','c50d2d1d42'),('CUP','libero_living_room_scene5'),('KEYBOARD','3e8bba0176')]:
        for name, title in [('04_object_registered','Registered reconstruction'),('06_shooting_1','First impact'),
                            ('08_shooting_3','After interaction'),('10_robot_2','Robot interaction')]:
            p=base/'images'/obj/(name+'.png')
            item(p, obj.lower()+'-'+name, obj.title()+' · '+title, 'Objects' if name.startswith('04') else 'Motion', scene,
                 'SAM3D appearance in a recorded scripted sequence. Robot views for headphone and keyboard reuse original collision dynamics; task success is not established.')
    pairs = q/'with_without_harmonizer/pairs'
    for obj in ['CUP','HEADPHONE','KEYBOARD','BOWL_overview','BOWL_DSC01217']:
        for side in ['without','with']:
            item(pairs/obj/(side+'.png'), f'harmony-{obj.lower()}-{side}',
                f'{obj.replace("_", " ").title()} · {side} harmonizer', 'Appearance', obj,
                'Matched saved view. Harmonization is image postprocessing, separate from geometry and physics.')
    videos = DEMO/'qualitative/VIDEOS24FPS_STRONG_IMPACTS_20260915/delivery'
    for obj in ['HEADPHONE','CUP','KEYBOARD']:
        for family in ['shooting','robot']:
            src=videos/f'{obj}_{family}.mp4'; dst=PUBLIC/'media'/src.name.lower()
            shutil.copyfile(src,dst);record(dst,[src],'Unmodified recorded 24 fps video')
    (PUBLIC/'gallery.json').write_text(json.dumps(GALLERY,indent=2)+'\n')


def read_gs(p):
    v = PlyData.read(p)['vertex'].data
    col = lambda ns: np.column_stack([v[n] for n in ns]).astype(np.float32)
    return dict(means=col(['x','y','z']), scales=np.exp(col(['scale_0','scale_1','scale_2'])),
                quats=col(['rot_0','rot_1','rot_2','rot_3']),
                opacities=1/(1+np.exp(-np.clip(v['opacity'], -30,30))),
                sh0=col(['f_dc_0','f_dc_1','f_dc_2']))


def packed_gs(g, label, indices=None, transform=None):
    a={k:v if indices is None else v[indices] for k,v in g.items()}
    q=a['quats']; rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    rs=rot*a['scales'][:,None,:];cov=rs @ rs.transpose(0,2,1)
    means=a['means']
    if transform is not None:
        m=transform[:3,:3]; means=means@m.T+transform[:3,3];cov=m@cov@m.T
    out=np.zeros((len(means),16),np.float32)
    out[:,:3]=means;out[:,3]=label
    out[:,4:7]=np.clip(.5+.28209479177387814*a['sh0'],0,1);out[:,7]=a['opacities']
    out[:,8:11]=cov[:,0,:];out[:,11]=cov[:,1,1];out[:,12]=cov[:,1,2];out[:,13]=cov[:,2,2]
    return out


def gaussians():
    b=DEMO/'FIGURE2_C50_BOTTLE/data'
    g=read_gs(b/'original_scene.ply');n=len(g['means'])
    target=np.load(b/'obj_13_geometry_mask_idx.npy')
    # Keep a bounded desk-region preview, preserving every target Gaussian.
    xyz=g['means']; keep=np.where((np.linalg.norm(xyz-np.array([4.35,2.25,1.0]),axis=1)<2.6)&(g['opacities']>.06))[0]
    keep=np.setdiff1d(keep,target)
    rng=np.random.default_rng(201)
    if len(keep)>350000:keep=np.sort(rng.choice(keep,350000,replace=False))
    fixed=packed_gs(g,0,keep); original=packed_gs(g,1,target)
    z=np.load(b/'fill_gaussians.npz'); fill=packed_gs({k:z[k] for k in ['means','scales','quats','opacities','sh0']},2,np.arange(30595,31071))
    p=b/'object/trellis/trellis_gs.ply';t=read_gs(p)
    T=np.array(json.loads((b/'object/trellis/aligned.json').read_text())['T'])
    mesh=packed_gs(t,3,np.arange(0,len(t['means']),2),T)
    packed=np.concatenate([fixed,original,fill,mesh]);dst=PUBLIC/'models/desk.splat'
    packed.astype('<f4').tofile(dst)
    cam=json.loads((b/'cameras.json').read_text())['source']['DSC01617.JPG']
    meta=dict(count=len(packed),stride=16,source_count=n,preview='SH0 only, spatial crop and deterministic subsampling',
        groups=dict(background=len(fixed),original_target=len(original),fill=len(fill),replacement=len(mesh)),
        center=T[:3,3].tolist(),camera=cam)
    (PUBLIC/'models/desk.json').write_text(json.dumps(meta,indent=2)+'\n')
    record(dst,[b/'original_scene.ply',b/'obj_13_geometry_mask_idx.npy',b/'fill_gaussians.npz',p,b/'object/trellis/aligned.json'],
        'Bounded SH0 anisotropic Gaussian browser preview; deterministic spatial crop/subsample, original covariance and opacity',**meta)


def object_meshes():
    b=DEMO/'qualitative/HEADPHONE_CUP_KEYBOARD_SAM3D_TWOHIT_20260915/data'
    for obj in ['CUP','HEADPHONE','KEYBOARD']:
        src=b/obj/'registered_mesh.ply';gsp=b/obj/'registered_gaussian.ply'
        mesh=trimesh.load(src,force='mesh'); original_faces=len(mesh.faces)
        mesh=mesh.simplify_quadric_decimation(face_count=18000)
        g=read_gs(gsp);nearest=cKDTree(g['means']).query(mesh.vertices)[1]
        colors=np.clip((.5+.28209479177387814*g['sh0'][nearest])*255,0,255).astype(np.uint8)
        mesh.visual.vertex_colors=np.column_stack([colors,np.full(len(colors),255,np.uint8)])
        center=mesh.bounds.mean(axis=0);mesh.vertices-=center
        # Preserve metric scale, map source Z-up to the browser Y-up convention.
        mesh.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]))
        dst=PUBLIC/'models'/f'{obj.lower()}.glb';mesh.export(dst)
        record(dst,[src,gsp], 'Mesh decimation to 18000 faces; nearest Gaussian SH0 vertex color; centered, metric units, Y-up',original_faces=original_faces,faces=len(mesh.faces),source_center=center.tolist())


def robot():
    b=DEMO/'qualitative/VIDEOS24FPS_STRONG_IMPACTS_20260915/episodes/CUP/robot_new'
    xml=b/'robot.xml';trajectory=b/'robot/trajectory.npz'
    # Resolve the workspace relocation without touching the historical XML.
    source=xml.read_text().replace('/group/worldcept/code/','/group/worldcept/PhiRIE/code/')
    m=mujoco.MjModel.from_xml_string(source);d=mujoco.MjData(m);z=np.load(trajectory)
    ids=[];sc=trimesh.Scene();offsets=[]
    for i in range(m.ngeom):
        body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[i]) or ''
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or f'geom_{i}'
        is_robot=body.startswith('robot/')
        # Include robot visual meshes, exact cup convex pieces and tabletop.
        include=(is_robot and m.geom_group[i]!=3) or name.startswith('sam3d_cup_') or ('table' in name and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH)
        if not include:continue
        typ=m.geom_type[i]
        if typ==mujoco.mjtGeom.mjGEOM_MESH:
            j=m.geom_dataid[i];va=m.mesh_vertadr[j];fa=m.mesh_faceadr[j]
            mesh=trimesh.Trimesh(m.mesh_vert[va:va+m.mesh_vertnum[j]].copy(),m.mesh_face[fa:fa+m.mesh_facenum[j]].copy(),process=False)
            # Preserve robot visual part boundaries: simplifying the separate
            # overlapping shells can introduce visible depth fighting.
            if not is_robot and len(mesh.faces)>5000:mesh=mesh.simplify_quadric_decimation(face_count=5000)
        elif typ==mujoco.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(extents=m.geom_size[i]*2)
        else:continue
        color=m.geom_rgba[i].copy()
        mid=m.geom_matid[i]
        if mid>=0:color=m.mat_rgba[mid].copy()
        if 'table' in name:color=np.array([.44,.50,.28,1])
        if name.startswith('sam3d_cup_'):color=np.array([.16,.68,.64,1])
        color[3]=1;mesh.visual.face_colors=(color*255).astype(np.uint8)
        sc.add_geometry(mesh,node_name=f'geom_{i}',geom_name=f'geom_{i}')
        ids.append(i)
    frames=[];path=[]
    for index in range(len(z['time'])):
        mujoco.mj_setState(m,d,z['integration_state'][index],int(z['state_spec']));mujoco.mj_forward(m,d)
        positions=d.geom_xpos[ids]
        quats=Rotation.from_matrix(d.geom_xmat[ids].reshape(-1,3,3)).as_quat()
        frames.append(np.column_stack([positions,quats]).round(6).tolist())
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/2f85/base')
        path.append(d.xpos[bid].round(6).tolist())
    dst=PUBLIC/'models/robot.glb';sc.export(dst)
    record(dst,[xml], 'Original full MuJoCo robot visual geometry and SAM3D cup convex collision pieces; only non-robot meshes may be simplified')
    dst=PUBLIC/'models/robot-motion.json'
    dst.write_text(json.dumps(dict(ids=ids,times=z['time'].round(6).tolist(),frames=frames,toolpath=path,
        duration=float(z['time'][-1]),contact_time=2.8383333333332734,
        provenance='Recorded MuJoCo states, scripted cup push and retract; task success unmeasured. No new simulation or learned policy.',
        source='qualitative/VIDEOS24FPS_STRONG_IMPACTS_20260915/episodes/CUP/robot_new'),separators=(',',':'))+'\n')
    record(dst,[xml,trajectory,b/'robot/result.json'],'Forward kinematics at all 213 original integration states; no interpolation or physics stepping')
    print('robot geoms',len(ids),'frames',len(frames))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['media','gaussians','objects','robot']);args=parser.parse_args()
    for d in ['media','models']:(PUBLIC/d).mkdir(parents=True,exist_ok=True)
    manifest=PUBLIC/'provenance.json'
    if args.only and manifest.exists():RECORDS=json.loads(manifest.read_text())['assets']
    for key,fn in [('media',media),('gaussians',gaussians),('objects',object_meshes),('robot',robot)]:
        if not args.only or args.only==key:fn()
    # Last record wins when rebuilding one export group.
    rows={r['file']:r for r in RECORDS}
    manifest.write_text(json.dumps(dict(version=1,asset_count=len(rows),assets=list(rows.values())),indent=2)+'\n')
    print('Exported',len(rows),'assets')
