import json
import hashlib
import pytest
from robo.eval.native_scale_tables import geometry_table


def test_geometry_denominators_and_changed_artifacts(tmp_path):
    mesh=tmp_path/'mesh_sim.obj';mesh.write_text('frozen')
    record={'geometry_producer':'robo.eval.fidelity_metrics.geometry_metrics','evaluator_alignment':'NONE',
      'cohort_id':'dev','scope':'L0_target_only','canonical_instance_id':'a','sensor_regime':'ideal_rgbd','tier':'DEV',
      'frozen_estimated_artifacts':{'B0_FIXED_NATIVE':{'mesh_sim.obj':hashlib.sha256(mesh.read_bytes()).hexdigest()}},
      'rows':[{'method':'B0_FIXED_NATIVE','object_dir':str(tmp_path),'cd_cm':1.,'f1_20':.8,'collapse':False}]}
    path=tmp_path/'metrics.json';path.write_text(json.dumps(record))
    units=[dict(cohort_id='dev',scope='L0_target_only',sensor_regime='ideal_rgbd_posed',split='development',canonical_instance_id=i) for i in ['a','a','b']]
    rows,_=geometry_table(units,[path]);b0=next(r for r in rows if r['method']=='B0_FIXED_NATIVE')
    assert b0['planned_objects']==2 and b0['matched_objects']==1 and b0['cd_cm']==1.
    assert b0['geometry_coverage']==.5 and b0['accepted_objects'] is None
    with pytest.raises(ValueError,match='duplicate'):geometry_table(units,[path,path])
    with pytest.raises(ValueError,match='unplanned'):geometry_table(units[2:],[path])
    mesh.write_text('modified')
    with pytest.raises(ValueError,match='changed'):geometry_table(units,[path])
