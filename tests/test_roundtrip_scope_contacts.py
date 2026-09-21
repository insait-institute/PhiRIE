import numpy as np
import pytest
import mujoco
from robo.roundtrip.scope_contacts import bind_contact_inventory,sample_contacts


def fixture():
 m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/><body pos="0 0 .09"><freejoint/><geom name="generated" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
 d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 return m,d


def test_actual_contacts_classified_without_state_mutation():
 m,d=fixture();before=d.qpos.copy();inventory=bind_contact_inventory(m,[{'object_id':'a','contact_geoms':['generated']}],[[1,1,1],[2,2,2]])
 r=sample_contacts(m,d,inventory)
 assert r['retained_context_contact_count']==1 and r['outside_workspace_contact_count']==1
 assert r['contacts'][0]['touches_retained_reference']
 np.testing.assert_array_equal(d.qpos,before)
 assert sample_contacts(m,d,bind_contact_inventory(m,[{'object_id':'a','contact_geoms':['generated']}]))['outside_workspace_contact_count'] is None


def test_unknown_or_duplicate_geom_fails():
 m,d=fixture()
 with pytest.raises(ValueError):bind_contact_inventory(m,[{'object_id':'a','contact_geoms':['missing']}])
 with pytest.raises(ValueError):bind_contact_inventory(m,[{'object_id':'a','contact_geoms':['generated','generated']}])


def test_substep_transient_contact_and_exact_native_physics_identity():
 pytest.importorskip("robosuite", reason="requires the separate native robot runtime")
 from robosuite.utils.binding_utils import MjSim
 from robo.roundtrip.scope_contacts import record_physics_contacts,substep_report
 xml='<mujoco><option timestep=".001" gravity="0 0 0"/><default><geom solref=".002 .05" friction="0 0 0"/></default><worldbody><geom name="floor" type="plane" size="2 2 .1"/><body pos="0 0 .1"><freejoint/><geom name="generated" type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>'
 a=MjSim.from_xml_string(xml);b=MjSim.from_xml_string(xml)
 a.data.qvel[2]=-1;b.data.qvel[2]=-1
 inv=bind_contact_inventory(b.model._model,[{'object_id':'a','contact_geoms':['generated']}],[[-.01,-.01,.1],[.01,.01,.2]])
 before_step=b.step
 with record_physics_contacts(b,inv) as rows:
  for i in range(300):
   a.step();b.step()
   mask=mujoco.mjtState.mjSTATE_INTEGRATION
   sa=np.empty(mujoco.mj_stateSize(a.model._model,mask));sb=np.empty_like(sa)
   mujoco.mj_getState(a.model._model,a.data._data,sa,mask);mujoco.mj_getState(b.model._model,b.data._data,sb,mask)
   np.testing.assert_array_equal(sa,sb)
 assert b.step==before_step and 'step' not in vars(b)
 report=substep_report(rows,inv)
 assert report['physics_steps']==300 and report['any_outside_workspace_contact']
 assert any(r['contacts'] for r in rows) and not rows[-1]['contacts']
 assert sample_contacts(b.model._model,b.data._data,inv)['reconstructed_contact_count']==0


def test_step2_hook_and_exception_restore():
 pytest.importorskip("robosuite", reason="requires the separate native robot runtime")
 from robosuite.utils.binding_utils import MjSim
 from robo.roundtrip.scope_contacts import record_physics_contacts
 sim=MjSim.from_xml_string('<mujoco><worldbody/></mujoco>');inv=bind_contact_inventory(sim.model._model,[])
 with pytest.raises(RuntimeError):
  with record_physics_contacts(sim,inv) as rows:
   sim.step1();sim.step2();assert len(rows)==1
   raise RuntimeError('policy boundary failure')
 assert 'step' not in vars(sim) and 'step2' not in vars(sim)
