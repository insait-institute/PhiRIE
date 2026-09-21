import pytest
from robo.roundtrip.scope_matrix import scope_units,public_destination


def inputs():
 s={'kind':'native_scope_subset','layout_ids':[1],'controller_methods':['B3','B4'],'replacement_scopes':['L1_target_destination','L2_task_workspace'],'canonical_instances_planned':1,'resets_per_instance':2,'extra_units_planned':8}
 rows=[dict(cohort_id='test',canonical_instance_id='a',instance_slot_id='slot',layout_id=1,task_id='PickPlaceCounterToSink',generation_seed=0,reset_id=r,policy_rng_seed=r,controller_method=m,policy_id='p',sensor_regime='ideal',renderer='native',unit_id=f'{r}{m}') for r in range(2) for m in ['B3','B4']]
 return rows,s


def test_exact_scope_and_explicit_extra_control_denominator():
 rows,s=inputs();units,controls=scope_units(rows,s)
 assert len(units)==8 and len(controls)==6
 assert all(x['executed'] is None for x in units)
 assert len({x['scope_unit_id'] for x in units})==8


def test_missing_or_duplicate_not_dropped():
 rows,s=inputs()
 with pytest.raises(ValueError):scope_units(rows[:-1],s)
 rows[-1]=rows[0]
 with pytest.raises(ValueError,match='duplicate'):scope_units(rows,s)


def test_unknown_task_cannot_fallback():
 with pytest.raises(ValueError):public_destination('new_task')
 assert public_destination('PickPlaceCounterToCabinet')['component_kind']=='cabinet_bottom_shelf'
