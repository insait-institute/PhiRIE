"""Absent public state stays absent; fixture geometry is not experiment evidence."""
import copy
import hashlib
from pathlib import Path
import pytest
from robo.certification import task_graph as TG
from robo.certification import grounding as G
from robo.certification.features import robot_control, geometry, support_contact

ROOT=Path(__file__).resolve().parents[1]

def scene():return TG.load_scene(ROOT/'tests/data/scenes/task_graph_fixture')
def task(name='food_bussing'):return TG.load_task(ROOT/'tests/data/tasks'/f'{name}.yaml')

@pytest.mark.parametrize('name,digest',[
 ('food_bussing','a33391b074c31042e0cfe420834b4419bb37cfd4adaaf6ee29c632d2511f027f'),
 ('spatial_hint_disambiguation','c00d2a67ca854b948b7d43aff246a95ded4c2fc79c33954b2433ca1183eab241'),
 ('missing_evidence','3e077c3b940cb73793426bded1ea39d454d9422e3001eb416c577d4aad6bb781'),
 ('ambiguous_receptacle','a826c9a977561c5ecb8a9663be0ae5fd4334a7e9ce48f77509f556ea1953f0d2'),
 ('benchmark_mapping','374394e606e1db329735d07e56607d530b1576bc5ec6011594783de5fd70c3cc')])
def test_existing_complete_frame_graphs_byte_unchanged(name,digest):
 assert hashlib.sha256(TG.to_json(TG.build_graph(scene(),task(name))).encode()).hexdigest()==digest

@pytest.mark.parametrize('form',['absent','null_robot','null_position','null_yaw'])
def test_missing_robot_frame_has_no_zero_pose_reach_or_sweep(form):
 s=scene()
 if form=='absent':s.pop('robot')
 elif form=='null_robot':s['robot']=None
 elif form=='null_position':s['robot']['base_pos']=None
 else:s['robot']['base_yaw']=None
 before=copy.deepcopy(s);g=TG.build_graph(s,task())
 assert s==before
 assert 'robot_frame' in g['unresolved_references']
 node=next(n for n in g['nodes'] if n['kind']=='robot')
 assert node['transform'] is None and node['construction_status']=='unresolved'
 f=robot_control.extract_robot_control_features(task_graph=g)
 for key in ('robot_reach_margin_m','robot_obstacle_count_in_sweep'):
  assert f[key] is None and f[key+'_missing']==1.
 assert not g['role_resolutions']['obstacle']['hypotheses']


def test_missing_robot_cannot_resolve_spatial_hint_using_origin():
 s=scene();s.pop('robot');g=TG.build_graph(s,task('spatial_hint_disambiguation'))
 r=g['role_resolutions']['manipulated_object']
 assert r['status']=='ambiguous'
 assert all(not any(tag.startswith(G.EV_SPATIAL_QUALIFIER) for tag in h['evidence']) for h in r['hypotheses'])

@pytest.mark.parametrize('form',['absent','null','empty','incomplete_one'])
def test_missing_camera_is_unknown_not_negative_visibility(form):
 s=scene()
 if form=='absent':s.pop('cameras')
 elif form=='null':s['cameras']=None
 elif form=='empty':s['cameras']=[]
 else:s['cameras'][0]['pos']=None
 g=TG.build_graph(s,task())
 assert 'policy_cameras' in g['unresolved_references']
 assert geometry.extract_geometry_features({},task_graph=g)['geom_visible_fraction'] is None
 assert robot_control.extract_robot_control_features(task_graph=g)['robot_target_visible'] is None
 if form=='incomplete_one':
  unknown=next(n for n in g['nodes'] if n['node_id']=='camera:'+s['cameras'][0]['id'])
  assert unknown['transform'] is None


def test_declared_missing_target_does_not_choose_category_prior_substitute():
 t=task();t['unresolved_roles']={'receptacle':'reference_view_absent'}
 g=TG.build_graph(scene(),t)
 assert g['role_resolutions']['receptacle']['hypotheses']==[]
 assert g['role_resolutions']['receptacle']['missing_evidence']==['reference_view_absent']
 assert 'role:receptacle' in g['unresolved_references']


def test_missing_support_cannot_use_existing_support_edge_as_pass():
 t=task();t['roles'].append('support');t['unresolved_roles']={'support':'support_surface_unobserved'}
 g=TG.build_graph(scene(),t)
 assert g['role_resolutions']['support']['hypotheses']==[]
 assert not [edge for edge in g['edges'] if edge['kind']=='support']
 assert not [role for node in g['nodes'] for role in node.get('roles',[]) if role['role']=='support']
 assert support_contact.extract_support_contact_features({},task_graph=g)['support_com_margin_risk'] is None


def test_absent_scene_geometry_retains_required_roles():
 s=scene();s['objects']=None;g=TG.build_graph(s,task())
 assert 'scene_geometry' in g['missing_evidence']
 assert {'manipulated_object','receptacle'}<=set(g['unresolved_references'])
 assert not [n for n in g['nodes'] if n['kind']=='object']
 assert robot_control.extract_robot_control_features(task_graph=g)['robot_obstacle_count_in_sweep'] is None

@pytest.mark.parametrize('kind',['robot_nan','robot_shape','camera_nan','camera_shape','camera_direction','projection_nan','unknown_role','empty_reason'])
def test_malformed_values_fail_closed_instead_of_becoming_missing(kind):
 s=scene();t=task()
 if kind=='robot_nan':s['robot']['base_pos'][0]=float('nan')
 elif kind=='robot_shape':s['robot']['base_pos']=[0,0]
 elif kind=='camera_nan':s['cameras'][0]['pos'][0]=float('nan')
 elif kind=='camera_shape':s['cameras'][0]['pos']=[0,0]
 elif kind=='camera_direction':s['cameras'][0]['look_at']=s['cameras'][0]['pos']
 elif kind=='projection_nan':s['cameras'][0]['fovy_deg']=float('nan')
 elif kind=='unknown_role':t['unresolved_roles']={'something':'absent'}
 else:t['unresolved_roles']={'receptacle':''}
 with pytest.raises(ValueError):TG.build_graph(s,t)


def test_explicit_missing_support_does_not_still_exclude_it_from_obstacle_sweep(monkeypatch):
 t=task();t['roles'].extend(['support','obstacle'])
 t['unresolved_roles']={'support':'surface_unobserved','obstacle':'sweep_unobserved'}
 original=G.resolve_obstacle_role;observed=[]
 def capture(objects,base,target,exclude,**kw):
  observed.append(set(exclude));return original(objects,base,target,exclude,**kw)
 monkeypatch.setattr(G,'resolve_obstacle_role',capture)
 g=TG.build_graph(scene(),t)
 assert 'obj_05' not in observed[0]
 assert not [edge for edge in g['edges'] if edge['kind'] in {'support','obstacle'}]
 assert not g['role_resolutions']['obstacle']['hypotheses']
 assert robot_control.extract_robot_control_features(task_graph=g)['robot_obstacle_count_in_sweep'] is None
 assert not [role for node in g['nodes'] for role in node.get('roles',[]) if role['role'] in {'support','obstacle'}]
