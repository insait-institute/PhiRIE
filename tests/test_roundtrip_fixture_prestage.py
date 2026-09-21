from types import SimpleNamespace as NS
import pytest
from robo.roundtrip.fixture_scope import staged_fixture_components,bind_fixture_component


class Fixture:
    root_body='sink_main';naming_prefix='sink_'
    def __init__(self):
        self._contact_geoms=['old','spout'];self._visual_geoms=['old','spout'];self._regions={'native_goal':'unchanged'}
    @property
    def contact_geoms(self):return [self.naming_prefix+n for n in self._contact_geoms]
    @property
    def visual_geoms(self):return [self.naming_prefix+n for n in self._visual_geoms]


XML='<mujoco><worldbody><body name="sink_main"><geom name="sink_new"/><body name="sink_child"><geom name="sink_spout"/></body></body></worldbody></mujoco>'
RECEIPT=dict(import_route='static_fixture_component_v1',body_name='sink_main',contact_geoms=['sink_new'],visual_geoms=['sink_new'],removed_original_geoms=['sink_old'])


def env():
    fixture=Fixture()
    return NS(fixtures={'sink':fixture},model=NS(mujoco_objects=[fixture]),get_fixture=lambda _:fixture,
              sim=NS(model=NS(geom_name2id=lambda n:['sink_new','sink_spout'].index(n))))


def test_native_id_mapping_sees_new_names_before_compile_and_postbind_is_idempotent():
    native=env();fixture=native.fixtures['sink'];regions=fixture._regions
    with staged_fixture_components(native,XML,[RECEIPT]):
        # This is the upstream generate_id_mappings lookup order that failed.
        for model in native.model.mujoco_objects:
            for name in model.visual_geoms+model.contact_geoms:native.sim.model.geom_name2id(name)
    bind_fixture_component(native,fixture_name='sink',receipt=RECEIPT)
    assert fixture.contact_geoms==['sink_spout','sink_new']
    assert fixture._regions is regions


def test_failed_native_reset_rolls_back_handles():
    native=env();fixture=native.fixtures['sink'];old=fixture._contact_geoms
    with pytest.raises(RuntimeError):
        with staged_fixture_components(native,XML,[RECEIPT]):
            raise RuntimeError('native XML reset failure')
    assert fixture._contact_geoms is old and fixture.contact_geoms==['sink_old','sink_spout']


def test_missing_new_geom_refuses_before_any_handle_mutation():
    native=env();old=native.fixtures['sink']._visual_geoms
    with pytest.raises(ValueError,match='final XML'):
        with staged_fixture_components(native,XML.replace('sink_new','wrong'),[RECEIPT]):pass
    assert native.fixtures['sink']._visual_geoms is old
