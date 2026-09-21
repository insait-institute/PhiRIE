"""Contract fixtures for the real-input E6 driver, never paper observations."""
import json
from pathlib import Path

import pytest
import yaml

from robo.eval import build_task_support_dataset as builder


@pytest.fixture
def population(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    source = Path(builder.__file__).resolve().parents[2]
    public = tmp_path / "public"
    public.mkdir()
    protocol = source / "configs/experiments/icra2027/audit_validity_protocol.yaml"
    scene = yaml.safe_load((source / "tests/data/scenes/task_graph_fixture/scene.yaml").read_text())
    task = yaml.safe_load((source / "tests/data/tasks/food_bussing.yaml").read_text())
    rows = []
    for scene_id in ("s0", "s1"):
        for condition in ("clean", "severe"):
            ident = dict(freeze_id="fixture-only", scene_id=scene_id, task_id="q", condition_id=condition)
            bundle = {**ident, "evidence_source": "construction_observation", "source_kind": "synthetic_fixture",
                      "build_status": "complete" if condition == "clean" else "failed",
                      "scene": {**scene, "scene_id": scene_id}, "task": {**task, "task_id": "q"},
                      "audit": {"objects": [{"object_id": obj["id"], "checks": {
                          "removal_alpha_coverage": {"status": "pass", "alpha_cov": index / 10.0},
                          "registration_residual": {"status": "pass", "chamfer_med_mm": index * 2.0}}}
                          for index, obj in enumerate(scene["objects"])]}}
            path = public / f"{scene_id}-{condition}.json"
            path.write_text(json.dumps(bundle))
            rows.append({**ident, "task_family": "object_to_receptacle",
                         "build_manifest": {"path": str(path), "sha256": builder._sha256(path)}})
    manifest = public / "manifest.json"
    manifest.write_text(json.dumps({"freeze_id": "fixture-only", "rows": rows}))
    config = {"freeze_id": "fixture-only", "label_definition": "construction_validity", "tier": "fixture",
              "source_kind": "synthetic_fixture", "scene_ids": ["s0", "s1"], "conditions": ["clean", "severe"],
              "task_ids_by_scene": {"s0": ["q"], "s1": ["q"]}, "public_roots": [str(public)],
              "public_manifest": {"path": str(manifest), "sha256": builder._sha256(manifest)},
              "label_protocol_sha256": builder._sha256(protocol)}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return tmp_path, config_path, config, protocol, rows


def measurements(population, feature_dir):
    root, _, _, _, rows = population
    values = []
    for row in rows:
        values.append({**{key: row[key] for key in builder.JOIN_KEY},
                       "measurement_scope": "all_required_task_invariants", "object_recovered": True,
                       "translation_cm": 1.0, "rotation_deg": 1.0, "scale_error_pct": 1.0,
                       "penetration_cm": 0.0, "support_correct": True, "collision_valid": True,
                       "required_visible": True, "robot_alignment_correct": True,
                       "global_psnr_db": 30.0, "global_f1_score": 0.8})
    path = root / "evaluation.json"
    path.write_text(json.dumps({"freeze_id": "fixture-only",
                              "feature_seal_sha256": builder._sha256(feature_dir / "feature_seal.json"),
                              "rows": values}))
    return path


def test_features_then_sealed_validity_join_preserves_failed_jobs(population):
    root, config_path, _, protocol, _ = population
    feature_dir = root / "features"
    result = builder.generate_features(config_path, feature_dir)
    assert result["source_kind"] == "synthetic_fixture"
    assert result["query_rows"] == 4
    features = builder._read_csv(feature_dir / "features_unlabeled.csv")
    assert all("invalid_label" not in row and "global_f1_score" not in row for row in features)
    graph = json.loads((feature_dir / features[0]["graph_path"]).read_text())
    assert set(graph["feature_scopes"]["manipulated_object"]) < set(graph["feature_scopes"]["task_graph"])
    assert features[0]["object_visual_removal_alpha_coverage"] != features[0]["visual_removal_alpha_coverage"]
    assert features[0]["support_drop_unstable_any"] == ""
    assert features[0]["support_drop_unstable_any_missing"] == "1.0"
    evaluation = measurements(population, feature_dir)
    builder.join_evaluation(feature_dir, protocol, evaluation, builder._sha256(evaluation), root / "joined")
    joined = builder._read_csv(root / "joined/task_local_features_and_labels.csv")
    assert len(joined) == 4
    assert [int(row["invalid_label"]) for row in joined] == [0, 1, 0, 1]
    assert all("construction_failed" in row["invalid_reasons"] for row in joined if row["condition_id"] == "severe")
    report = json.loads((root / "joined/label_join_manifest.json").read_text())
    assert report["loso_admissible"]
    assert not report["paper_ready"]
    groups = report["signal_groups"]
    assert all(col.startswith("object_") for col in groups["Manipulated-object evidence"])
    assert not set(groups["Manipulated-object evidence"]) & set(groups["Task graph, visual + geometry"])
    with pytest.raises(FileExistsError):
        builder.generate_features(config_path, feature_dir)


def test_feature_stage_does_not_open_protocol_or_evaluation(population, monkeypatch):
    root, config_path, _, protocol, _ = population
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != protocol and "vault" not in str(path) and path.name != "evaluation.json"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    builder.generate_features(config_path, root / "features")


@pytest.mark.parametrize("field", ["gt_object_id", "invalid_label", "held_out", "role_refs", "global_f1_score"])
def test_public_payload_rejects_evaluation_channels(field):
    with pytest.raises(ValueError, match="forbidden evaluation"):
        builder._public_payload({"objects": [{field: 1}]})


@pytest.mark.parametrize("directory", ["oracle_gt_vault", "gt"])
def test_public_reader_rejects_vault_symlink_before_open(population, directory):
    root, _, _, _, _ = population
    vault = root / directory
    vault.mkdir()
    target = vault / "protected.json"
    target.write_text("{}")
    alias = root / "public/alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="escapes approved roots or enters vault"):
        builder._public_read({"path": str(alias), "sha256": "unread"}, [root / "public"])


@pytest.mark.parametrize("fault", ["hash", "mixed_freeze", "missing_row", "duplicate_row", "missing_roster"])
def test_public_manifest_fails_closed(population, fault):
    root, config_path, config, _, _ = population
    manifest_path = Path(config["public_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    if fault == "hash":
        config["public_manifest"]["sha256"] = "0" * 64
    elif fault == "mixed_freeze":
        manifest["freeze_id"] = "other"
    elif fault == "missing_row":
        manifest["rows"].pop()
    elif fault == "duplicate_row":
        manifest["rows"].append(manifest["rows"][0])
    else:
        config["task_ids_by_scene"]["s0"] = []
    manifest_path.write_text(json.dumps(manifest))
    if fault != "hash":
        config["public_manifest"]["sha256"] = builder._sha256(manifest_path)
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        builder.generate_features(config_path, root / "features")
    assert not (root / "features").exists()


@pytest.mark.parametrize("fault", ["tamper_features", "tamper_graph", "protocol", "mixed_freeze", "missing_label", "nan", "success"])
def test_join_rejects_leakage_tampering_and_missing_pairs(population, fault):
    root, config_path, _, protocol, _ = population
    feature_dir = root / "features"
    builder.generate_features(config_path, feature_dir)
    evaluation = measurements(population, feature_dir)
    data = json.loads(evaluation.read_text())
    if fault.startswith("tamper"):
        path = feature_dir / ("features_unlabeled.csv" if fault == "tamper_features" else "graphs/0000.json")
        path.write_text(path.read_text() + " ")
    elif fault == "protocol":
        other = root / "other-protocol.yaml"
        other.write_text(protocol.read_text() + "\n")
        protocol = other
    elif fault == "mixed_freeze":
        data["freeze_id"] = "other"
    elif fault == "missing_label":
        data["rows"].pop()
    elif fault == "nan":
        data["rows"][0]["translation_cm"] = float("nan")
    else:
        data["rows"][0]["success"] = True
    evaluation.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        builder.join_evaluation(feature_dir, protocol, evaluation, builder._sha256(evaluation), root / "joined")
    assert not (root / "joined").exists()


def test_degenerate_folds_stop_but_preserve_every_row(population):
    root, config_path, _, protocol, _ = population
    feature_dir = root / "features"
    builder.generate_features(config_path, feature_dir)
    evaluation = measurements(population, feature_dir)
    data = json.loads(evaluation.read_text())
    for row in data["rows"]:
        row["object_recovered"] = False
    evaluation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="every training/test fold requires both labels"):
        builder.join_evaluation(feature_dir, protocol, evaluation, builder._sha256(evaluation), root / "joined")
    assert len(builder._read_csv(root / "joined/task_local_features_and_labels.csv")) == 4
    assert not json.loads((root / "joined/label_join_manifest.json").read_text())["loso_admissible"]


def test_full_population_requires_exact_72_before_reading_any_artifact(population):
    root, config_path, config, _, _ = population
    config.update(tier="full", source_kind="real")
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="72 rows"):
        builder.generate_features(config_path, root / "features")

@pytest.mark.parametrize('missing',['robot','camera','target','support'])
def test_missing_public_state_retains_every_query_and_missing_evidence(population,missing):
    root,cp,config,protocol,rows=population
    for item in rows:
        path=Path(item['build_manifest']['path']);b=json.loads(path.read_text())
        if missing=='robot':b['scene']['robot']['base_pos']=None
        elif missing=='camera':b['scene']['cameras']=[]
        elif missing=='target':b['task']['unresolved_roles']={'receptacle':'source_view_unavailable'}
        else:
            b['task']['roles'].append('support')
            b['task']['unresolved_roles']={'support':'surface_unobserved'}
        path.write_text(json.dumps(b));item['build_manifest']['sha256']=builder._sha256(path)
    mp=Path(config['public_manifest']['path']);mp.write_text(json.dumps({'freeze_id':'fixture-only','rows':rows}))
    config['public_manifest']['sha256']=builder._sha256(mp);cp.write_text(yaml.safe_dump(config))
    result=builder.generate_features(cp,root/'features');assert result['query_rows']==4
    values=builder._read_csv(root/'features/features_unlabeled.csv');assert len(values)==4
    valid_build=next(r for r in values if r['condition_id']=='clean')
    if missing=='robot':
        assert valid_build['robot_frame_missing']=='1.0' and valid_build['robot_reach_margin_m']==''
    elif missing=='camera':
        assert valid_build['geometry_policy_cameras_missing']=='1.0' and valid_build['geometry_visible_fraction']==''
    elif missing=='target':assert valid_build['geometry_required_role_missing']=='1.0'
    else:assert valid_build['support_com_margin_risk']==''
    graph=json.loads(Path(valid_build['graph_path']).read_text())
    assert graph['unresolved_references']
    assert all('invalid_label' not in value for value in values)
    evaluation=measurements(population,root/'features')
    if missing in {'robot','camera'}:
        with pytest.raises(ValueError,match='LOSO blocked'):
            builder.join_evaluation(root/'features',protocol,evaluation,builder._sha256(evaluation),root/'joined')
    else:
        builder.join_evaluation(root/'features',protocol,evaluation,builder._sha256(evaluation),root/'joined')
    joined=builder._read_csv(root/'joined/task_local_features_and_labels.csv')
    assert len(joined)==4
    if missing in {'robot','camera'}:
        assert all(r['invalid_label']=='1' for r in joined)
        reason='construction_robot_frame_missing' if missing=='robot' else 'construction_policy_cameras_missing'
        assert all(reason in r['invalid_reasons'] for r in joined)
    else:
        # Graph grounding uncertainty is not an oracle construction-validity label.
        assert [r['invalid_label'] for r in joined if r['condition_id']=='clean']==['0','0']


def test_distinct_region_queries_survive_identical_unresolved_feature_values(population):
    import copy
    root,cp,config,_,old_rows=population
    rows=[]
    for old in old_rows:
        for suffix,region in [('left',[10,20,30,40]),('right',[60,20,80,40])]:
            item=copy.deepcopy(old);item['task_id']='q_'+suffix
            b=json.loads(Path(old['build_manifest']['path']).read_text())
            b['task_id']=item['task_id'];b['task']['task_id']=item['task_id']
            b['task']['public_target_region_pixels']=region
            b['task']['unresolved_roles']={'receptacle':'metric_region_unavailable'}
            path=Path(old['build_manifest']['path']).with_name(old['scene_id']+'-'+old['condition_id']+'-'+suffix+'.json')
            path.write_text(json.dumps(b));item['build_manifest']={'path':str(path),'sha256':builder._sha256(path)}
            rows.append(item)
    mp=Path(config['public_manifest']['path']);mp.write_text(json.dumps({'freeze_id':'fixture-only','rows':rows}))
    config['public_manifest']['sha256']=builder._sha256(mp)
    config['task_ids_by_scene']={s:['q_left','q_right'] for s in config['scene_ids']}
    cp.write_text(yaml.safe_dump(config));builder.generate_features(cp,root/'features')
    values=builder._read_csv(root/'features/features_unlabeled.csv')
    assert len(values)==8 and {r['task_id'] for r in values}=={'q_left','q_right'}
    assert len({r['graph_path'] for r in values})==8
    for row in values:
        graph=json.loads(Path(row['graph_path']).read_text())
        assert graph['task_id']==row['task_id']
        source=json.loads(Path(row['build_manifest_path']).read_text())
        expected=[10,20,30,40] if row['task_id']=='q_left' else [60,20,80,40]
        assert source['task']['public_target_region_pixels']==expected
