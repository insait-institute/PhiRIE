"""Export-bound dimensions and proven legacy orphan-vertex metadata correction.

No registration, mesh surface, collision, physical prior, or selection is changed.
The native importer's strict frame/scale guard remains authoritative.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import numpy as np


def _mesh(path):
    import trimesh
    mesh=trimesh.load(path,process=False,force='mesh')
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('finite referenced mesh surface required')
    return mesh


def exported_dimensions(object_dir,scale):
    """Metric local-axis bounds of face-referenced exported simulator geometry."""
    if not np.isfinite(scale) or scale<=0:raise ValueError('positive finite scale required')
    mesh=_mesh(Path(object_dir)/'mesh_sim.obj')
    return (np.ptp(mesh.vertices[np.unique(mesh.faces)],axis=0)*scale).tolist()


def correct_legacy_orphan_dimensions(source,out):
    """Copy a legacy artifact only after proving orphan vertices explain its bounds."""
    started=time.monotonic();source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    aligned=json.loads((source/'aligned.json').read_text());scale=float(aligned['scale'])
    ply,obj=_mesh(source/'mesh_sim.ply'),_mesh(source/'mesh_sim.obj')
    referenced=np.unique(ply.faces);orphans=len(ply.vertices)-len(referenced)
    if orphans<=0:raise ValueError('no orphan vertices explain legacy dimensions')
    # OBJ exports use eight fractional digits. This comparison is a file-format
    # round-trip proof, not a relaxed geometry/registration acceptance threshold.
    if ply.triangles.shape!=obj.triangles.shape or not np.allclose(ply.triangles,obj.triangles,rtol=0,atol=1e-8):
        raise ValueError('PLY and OBJ referenced triangles differ')
    if not np.allclose(aligned['world_dims'],np.ptp(ply.vertices,axis=0)*scale,rtol=1e-5,atol=1e-7):
        raise ValueError('stored dimensions are not the proven legacy PLY bounds')
    dimensions=exported_dimensions(source,scale)
    if np.allclose(aligned['world_dims'],dimensions,rtol=1e-5,atol=1e-7):
        raise ValueError('legacy dimensions already satisfy strict native contract')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    parents={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source,out);corrected=dict(aligned,world_dims=dimensions)
    (out/'aligned.json').write_text(json.dumps(corrected,indent=2,allow_nan=False)+'\n')
    children={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    changed=[p for p in parents if parents[p]!=children[p]]
    if set(parents)!=set(children) or changed!=['aligned.json']:raise ValueError('metadata correction changed unrelated artifact bytes')
    return dict(schema_version=1,kind='implementation_metadata_correction',source=str(source),corrected_asset=str(out),
        parent_hashes=parents,child_hashes=children,changed_files=changed,changed_fields=['world_dims'],
        orphan_vertices=orphans,referenced_triangles=len(ply.faces),triangle_serialization_atol=1e-8,
        referenced_surface_equal=True,old_world_dims=aligned['world_dims'],new_world_dims=dimensions,
        geometry_transform_collision_physics_unchanged=True,controller_action=False,
        outcome_input_used=False,wall_s=time.monotonic()-started)


def validate_metadata_correction(receipt_path):
    """Recheck an immutable correction proof before evaluator-side import reuse."""
    receipt_path=Path(receipt_path);proof=json.loads(receipt_path.read_text())
    if (proof.get('kind')!='implementation_metadata_correction' or
        proof.get('changed_files')!=['aligned.json'] or proof.get('changed_fields')!=['world_dims'] or
        proof.get('controller_action') is not False or proof.get('outcome_input_used') is not False):
        raise ValueError('metadata correction declaration differs')
    source,child=Path(proof['source']),Path(proof['corrected_asset'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for directory,key in ((source,'parent_hashes'),(child,'child_hashes')):
        actual={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
        if actual!=proof[key]:raise ValueError('metadata correction artifact changed')
    parents,children=proof['parent_hashes'],proof['child_hashes']
    if set(parents)!=set(children) or [n for n in parents if parents[n]!=children[n]]!=['aligned.json']:
        raise ValueError('metadata correction changed unrelated bytes')
    old=json.loads((source/'aligned.json').read_text());new=json.loads((child/'aligned.json').read_text())
    if {k:v for k,v in old.items() if k!='world_dims'}!={k:v for k,v in new.items() if k!='world_dims'}:
        raise ValueError('metadata correction changed transform or other fields')
    expected=exported_dimensions(child,float(new['scale']))
    if not np.allclose(new['world_dims'],expected,rtol=1e-5,atol=1e-7):
        raise ValueError('metadata correction does not satisfy strict exported bounds')
    return proof
