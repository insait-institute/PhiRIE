import pytest
from robo.roundtrip.robot_fill import prepare


def test_robot_fill_rejects_changed_provenance_before_mesh_use(tmp_path):
    pytest.importorskip('plyfile')
    e=tmp_path/'erase';e.mkdir();(e/'robot_erasure_carve.json').write_text('{"erased_views":6,"planned_train_views":6}')
    with pytest.raises(ValueError,match='sealed robot erasure'):
        prepare({'robot_erasure':str(e),'robot_erasure_sha256':'wrong'},tmp_path/'out')


def test_fill_requires_admissible_plane_and_exact_source(tmp_path):
    import json,hashlib
    from robo.roundtrip.robot_fill import fill
    prep=tmp_path/'prep';prep.mkdir();p=prep/'robot_fill_preparation.json';p.write_text(json.dumps({'status':'NO_PLANE'}))
    c={'preparation':str(prep),'preparation_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='admissible observed plane'):fill(c,tmp_path/'out')
    p.write_text('{}')
    with pytest.raises(ValueError,match='preparation changed'):fill(c,tmp_path/'out')
