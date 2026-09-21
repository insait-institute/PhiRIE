"""Construction-only association of public query regions and discovered geometry.

Inputs are explicit predicted geometry/cameras and public query annotations.
No dataset, oracle, label, policy outcome or reference-geometry loader is used.
"""
from __future__ import annotations
import numpy as np
from robo.certification.grounding import label_match_score

PROTOCOL = {
    'version': 'public_rgb_grounding_v1', 'render_stride': 1, 'discovery_stride': 1,
    'association_min_iou': 0.15, 'association_margin': 0.05,
    'visible_depth_tolerance_m': 0.05, 'minimum_visible_vertices': 10,
    'region_min_points': 30, 'region_pixel_stride': 4, 'alpha_min': 0.6,
    'alignment_min_cameras': 3, 'alignment_min_rank': 2,
    'alignment_rank_tolerance_m': 1e-4, 'alignment_max_rms_m': 0.05,
    'alignment_max_orientation_rms_deg': 10.0,
    'alignment_transform': 'SE3_only_no_scale_fit',
    'virtual_robot_reference': 'clean_public_geometry_only',
    'missing_anchor_action': 'unresolved_no_cross_condition_view_injection',
}


def _matrix(value, shape):
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError('malformed/nonfinite predicted geometry or camera')
    return arr


def _pose(value):
    p = _matrix(value, (4,4))
    if (not np.allclose(p[3],[0,0,0,1]) or not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-5)
        or not np.isclose(np.linalg.det(p[:3,:3]),1.,atol=1e-5)):
        raise ValueError('predicted camera must be a rigid OpenCV world-to-camera pose')
    return p


def _intrinsics(value):
    K=_matrix(value,(3,3))
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('invalid predicted camera intrinsics')
    return K


def projected_instance(vertices, indices, K, w2c, depth, alpha):
    """Visible projected extent, checked against predicted rendered depth only."""
    vertices = np.asarray(vertices,dtype=float); indices = np.asarray(indices)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('invalid discovered mesh vertices')
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(vertices)):
        raise ValueError('instance vertex identity out of bounds')
    depth=np.asarray(depth); alpha=np.asarray(alpha)
    if depth.ndim!=2 or alpha.shape!=depth.shape:raise ValueError('predicted render shapes differ')
    K=_intrinsics(K); pose=_pose(w2c); h,w=depth.shape
    cam=vertices[indices]@pose[:3,:3].T+pose[:3,3]
    good=cam[:,2]>0.05; cam=cam[good]
    if not len(cam):return None
    pix=cam@K.T; pix=pix[:,:2]/pix[:,2,None]
    good=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    pix,cam=pix[good],cam[good]
    if not len(cam):return None
    uv=np.floor(pix).astype(int); d=depth[uv[:,1],uv[:,0]]; a=alpha[uv[:,1],uv[:,0]]
    good=np.isfinite(d)&np.isfinite(a)&(a>=PROTOCOL['alpha_min'])&(abs(d-cam[:,2])<=PROTOCOL['visible_depth_tolerance_m'])
    pix=pix[good]
    if len(pix)<PROTOCOL['minimum_visible_vertices']:return None
    return dict(bbox_normalized=[float(pix[:,0].min()/w),float(pix[:,1].min()/h),
                                float(pix[:,0].max()/w),float(pix[:,1].max()/h)],visible_vertices=len(pix))


def _iou(a,b):
    a=_matrix(a,(4,));b=_matrix(b,(4,))
    if np.any(a<0) or np.any(a>1) or np.any(b<0) or np.any(b>1):raise ValueError('image boxes must be normalized')
    if np.any(a[2:]<=a[:2]) or np.any(b[2:]<=b[:2]):return 0.
    intersection=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    return intersection/(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersection)


def associate_role(annotation, projected):
    """Score every observed candidate; no arbitrary tie/winner substitution."""
    candidates=[]
    for obj in projected:
        if obj.get('projection') is None:continue
        lexical=label_match_score(obj['label'],annotation['description'])
        overlap=_iou(annotation['bbox_xyxy_normalized'],obj['projection']['bbox_normalized'])
        if lexical>0 and overlap>=PROTOCOL['association_min_iou']:
            candidates.append(dict(object_id=obj['id'],overlap_iou=overlap,label_support=lexical,
                                   evidence=['public_rgb_region_overlap','predicted_depth_visibility','automatic_discovery']))
    candidates.sort(key=lambda r:(-r['overlap_iou'],r['object_id']))
    ambiguous=len(candidates)>1 and candidates[0]['overlap_iou']-candidates[1]['overlap_iou']<PROTOCOL['association_margin']
    return dict(status='ambiguous' if ambiguous else ('resolved' if candidates else 'unresolved'),
                candidates=candidates,selected_object_id=candidates[0]['object_id'] if candidates and not ambiguous else None,
                reason='overlap_tie' if ambiguous else (None if candidates else 'no_supported_discovered_instance'))


def region_surface(annotation, depth, alpha, K, w2c):
    """Metric observed patch from predicted depth; never a fabricated solid."""
    from matplotlib.path import Path
    if 'polygon_xy_normalized' not in annotation:return None
    depth=np.asarray(depth);alpha=np.asarray(alpha);h,w=depth.shape
    polygon=np.asarray(annotation['polygon_xy_normalized'],dtype=float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
        raise ValueError('malformed public region polygon')
    yy,xx=np.mgrid[0:h:PROTOCOL['region_pixel_stride'],0:w:PROTOCOL['region_pixel_stride']]
    x,y=xx.ravel(),yy.ravel();inside=Path(polygon).contains_points(np.stack([(x+.5)/w,(y+.5)/h],axis=1))
    d=depth[y,x];a=alpha[y,x]
    keep=inside&np.isfinite(d)&np.isfinite(a)&(d>.05)&(a>=PROTOCOL['alpha_min'])
    if int(keep.sum())<PROTOCOL['region_min_points']:return None
    rays=np.stack([x[keep]+.5,y[keep]+.5,np.ones(keep.sum())],axis=1)@np.linalg.inv(_intrinsics(K)).T
    pts=rays*d[keep,None];pose=np.linalg.inv(_pose(w2c));pts=pts@pose[:3,:3].T+pose[:3,3]
    return dict(aabb=[pts.min(0).tolist(),pts.max(0).tolist()],observed_points=len(pts),
                evidence=['public_query_polygon','predicted_surface_depth'],physical_volume_known=False)


def ground_query(query, *, source_present, projected_instances, surface=None, unavailable_reason=None):
    """Retain exact query identity and all declared roles, including missing ones."""
    roles={}
    for name,annotation in query['roles'].items():
        if name=='robot_visual_reference':
            roles[name]=dict(status='unresolved',reason='image_robot_reference_is_not_metric_calibration',selected_object_id=None)
        elif not source_present:
            roles[name]=dict(status='unresolved',reason=unavailable_reason or 'query_source_view_missing',selected_object_id=None)
        elif 'polygon_xy_normalized' in annotation:
            patch=surface(annotation) if surface is not None else None
            roles[name]=dict(status='observed_region' if patch else 'unresolved',reason=None if patch else 'metric_region_surface_missing',
                region_id=query['task_id']+':'+name,selected_object_id=None,metric_surface=patch)
        else:roles[name]=associate_role(annotation,projected_instances)
    return dict(scene_id=query['scene_id'],task_id=query['task_id'],task_family=query['task_family'],
        query_sha256=query['query_sha256'],source_state_anchor=query['source_state_anchor'],
        roles=roles,robot_frame=None,physics_verified=False,paper_ready=False)


def rigid_camera_alignment(reference, moving):
    """Map moving predicted frame into clean predicted frame without fitting scale."""
    names=sorted(set(reference)&set(moving))
    result=dict(status='unresolved',reason=None,common_frames=names,transform=None,scale_applied=1.,
                translation_rms_m=None,orientation_rms_deg=None,rank=None,baseline_scale_ratio=None,
                reference_kind='clean_public_predicted_frame',real_calibration_claimed=False)
    if len(names)<PROTOCOL['alignment_min_cameras']:
        result['reason']='insufficient_common_public_cameras';return result
    a=np.stack([np.linalg.inv(_pose(reference[n])) for n in names]);b=np.stack([np.linalg.inv(_pose(moving[n])) for n in names])
    ac=a[:,:3,3];bc=b[:,:3,3];aa=ac-ac.mean(0);bb=bc-bc.mean(0)
    rank=min(np.linalg.matrix_rank(aa,tol=PROTOCOL['alignment_rank_tolerance_m']),np.linalg.matrix_rank(bb,tol=PROTOCOL['alignment_rank_tolerance_m']))
    result['rank']=int(rank)
    if rank<PROTOCOL['alignment_min_rank']:
        result['reason']='degenerate_public_camera_baseline';return result
    u,_,vt=np.linalg.svd(bb.T@aa);D=np.eye(3);D[2,2]=np.linalg.det(vt.T@u.T);R=vt.T@D@u.T;t=ac.mean(0)-R@bc.mean(0)
    rms=float(np.sqrt(np.mean(np.sum((bc@R.T+t-ac)**2,axis=1))))
    relative=np.einsum('nij,njk->nik',np.transpose(a[:,:3,:3],(0,2,1)),np.einsum('ij,njk->nik',R,b[:,:3,:3]))
    angles=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    angle=float(np.sqrt(np.mean(angles**2)))
    result.update(translation_rms_m=rms,orientation_rms_deg=angle,baseline_scale_ratio=float(np.linalg.norm(aa)/np.linalg.norm(bb)))
    if rms>PROTOCOL['alignment_max_rms_m'] or angle>PROTOCOL['alignment_max_orientation_rms_deg']:
        result['reason']='public_camera_rigid_alignment_residual';return result
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    result.update(status='aligned_public_frame',transform=T.tolist())
    return result


def declare_virtual_robot(objects):
    """One prospective clean-frame placement; never real robot calibration.

    Existing planner geometry is reused with an explicitly geometry-only predicate.
    Unknown mass, physics, drift and policy competence are not imputed as passing.
    """
    from robo.tasks import pi05_tasks as planner
    result=dict(status='unresolved',reason=None,base_pos=None,base_yaw=None,
                reference_kind='clean_public_geometry_virtual_benchmark',real_calibration_claimed=False,
                physical_stability_known=False,source_object_ids=[o['id'] for o in objects])
    rows=[]
    for obj in objects:
        box=_matrix(obj['aabb'],(2,3))
        if np.any(box[1]<box[0]):raise ValueError('invalid observed object bounds')
        rows.append(dict(id=obj['id'],label=obj['label'],aabb=box,center=box.mean(0),dims=box[1]-box[0],bottom_z=float(box[0,2])))
    def predicate(o):
        return o['label'] in planner.GRASP_LABELS and o['dims'].max()<=.28 and o['dims'].min()>=.005
    if not rows:
        result['reason']='no_public_discovered_geometry';return result
    try:
        _,members=planner._pick_table(rows,planning_predicate=predicate)
    except SystemExit:
        result['reason']='no_observed_tabletop_geometry_cluster';return result
    placement=planner._place_robot(members,planning_predicate=predicate)
    point=np.asarray(placement['base_pos'][:2])
    for o in members:
        lo,hi=o['aabb'][0,:2]-.03,o['aabb'][1,:2]+.03
        distance=float(np.linalg.norm(np.clip(point,lo,hi)-point))
        if distance<planner.BASE_CLEAR:
            result['reason']='existing_planner_returned_unclear_fallback';return result
    result.update(status='declared_virtual_frame',base_pos=placement['base_pos'],base_yaw=placement['base_yaw'])
    return result


def transport_virtual_frame(declaration,alignment):
    """Carry the same clean declaration into another predicted frame via SE3."""
    result=dict(status='unresolved',reason=None,world_from_robot=None,scale_applied=1.,
                real_calibration_claimed=False,physical_stability_known=False)
    if declaration is None or declaration.get('status')!='declared_virtual_frame':
        result['reason']='clean_virtual_declaration_unavailable';return result
    if alignment.get('status')!='aligned_public_frame' or alignment.get('scale_applied')!=1.:
        result['reason']='public_frame_alignment_unavailable';return result
    T=_pose(alignment['transform']);base=_matrix(declaration['base_pos'],(3,));yaw=float(declaration['base_yaw'])
    if not np.isfinite(yaw):raise ValueError('nonfinite virtual base heading')
    robot=np.eye(4);robot[:3,:3]=[[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]];robot[:3,3]=base
    result.update(status='transported_virtual_declaration',world_from_robot=(np.linalg.inv(T)@robot).tolist())
    return result
