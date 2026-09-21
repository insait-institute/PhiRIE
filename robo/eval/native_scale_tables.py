"""Native scale tables from the N1 canonical roster and merged ledger.

This module computes outcome statistics only. Geometry, appearance and physical
metrics remain owned by their existing producers; their unprovided cells are null.
Invoke through robo.eval.paper_pipeline with a native_scale_up configuration.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from robo.eval.metric_utils import paired_hierarchical_bootstrap, write_csv, write_json

IDENTITY = ("cohort_id", "canonical_instance_id", "reset_id", "policy_rng_seed",
            "policy_id", "controller_method", "scope", "sensor_regime", "renderer",
            "execution_protocol", "layout_id", "style_id", "task_id", "split", "native_horizon")
BLOCK = ("cohort_id", "policy_id", "scope", "sensor_regime", "renderer", "execution_protocol", "split")
CORE = ("REF_NATIVE", "B0_FIXED_NATIVE", "B3_AGENT_NATIVE", "B4_ROOM_REPAIR_NATIVE", "BM_BUDGET_MATCHED_NATIVE")
# N1's merger is authoritative; these are interpretation groups, not new statuses.
UNMEASURED = {"NOT_SCHEDULED", "ENVIRONMENT_FAILED", "CODE_FAILED", "RESOURCE_FAILED"}
UNEXECUTED_FAILURE = {"BUILD_FAILED", "ABSTAINED"}
EXECUTED = {"RECORDED"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row["unit_id"]
        if uid in result:
            raise ValueError(f"duplicate unit_id: {uid}")
        for key in IDENTITY:
            if key not in row or row[key] is None:
                raise ValueError(f"unit {uid} missing identity field {key}")
        result[uid] = row
    return result


def load_units(planned_path, ledger_path):
    planned, ledger = _rows(planned_path), _rows(ledger_path)
    if not planned:
        raise ValueError("planned roster is empty")
    if set(ledger) - set(planned):
        raise ValueError("ledger contains unplanned unit IDs")
    identities = set()
    units = []
    for uid, plan in planned.items():
        identity = tuple(plan[k] for k in IDENTITY)
        if identity in identities:
            raise ValueError("duplicate planned identity under different unit IDs")
        identities.add(identity)
        row = ledger.get(uid)
        if row is None:
            units.append(dict(plan, terminal_status="MISSING_LEDGER_ROW", measured=False,
                              executed=None, success=None, service_success=None))
            continue
        if any(plan[k] != row[k] for k in IDENTITY):
            raise ValueError(f"ledger identity differs from plan: {uid}")
        status, executed, success = row["terminal_status"], row.get("executed"), row.get("success")
        if status in UNMEASURED:
            if (executed is not None if status == "NOT_SCHEDULED" else executed is not False) or success is not None:
                raise ValueError(f"unmeasured unit has an outcome: {uid}")
            measured = False
        elif status in UNEXECUTED_FAILURE:
            if executed is not False or success is not None:
                raise ValueError(f"build failure/abstention must have unexecuted, unknown native success: {uid}")
            measured = True
        elif status in EXECUTED:
            if executed is not True or type(success) is not bool:
                raise ValueError(f"executed terminal unit must have a boolean outcome: {uid}")
            measured = True
        else:
            raise ValueError(f"unknown N1 terminal status {status}")
        result = row.get("result")
        if result and (result.get("success") != success or
                       ("executed" in result and result["executed"] != executed)):
            raise ValueError(f"canonical result disagrees with merged outcome: {uid}")
        if row.get("result_path") and _sha(row["result_path"]) != row.get("result_sha256"):
            raise ValueError(f"canonical result bytes changed: {uid}")
        units.append(dict(row, measured=measured,
                          service_success=False if status in UNEXECUTED_FAILURE else success))
    return units


def load_planning_slots(path):
    """Read N1 unresolved plan IDs without inventing canonical instance IDs."""
    units, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row["plan_unit_id"]
        if uid in seen:
            raise ValueError("duplicate planning slot")
        seen.add(uid)
        if row.get("canonical_instance_id") is not None or row.get("terminal_status") != "UNRESOLVED_INSTANCE":
            raise ValueError("planning mode accepts only unresolved N1 slots")
        if row.get("success") is not None or row.get("executed") is not None:
            raise ValueError("unresolved planning slot cannot contain an outcome")
        for field in IDENTITY:
            if field != "canonical_instance_id" and row.get(field) is None:
                raise ValueError(f"planning slot missing {field}")
        units.append(dict(row, unit_id=uid, measured=False, service_success=None))
    if not units:
        raise ValueError("empty planning roster")
    return units


def _aggregate(rows):
    measured = [r for r in rows if r["measured"]]
    executed = [r for r in measured if r["executed"]]
    successes = sum(r["service_success"] for r in measured)
    complete = len(measured) == len(rows)
    by_instance = defaultdict(list)
    for r in measured:
        by_instance[r["canonical_instance_id"]].append(int(r["service_success"]))
    return {"planned": len(rows), "measured": len(measured), "executed": len(executed),
            "successes_observed": successes, "complete": complete,
            "layouts": len({r["layout_id"] for r in rows}),
            "instances": len({r["canonical_instance_id"] for r in rows if r["canonical_instance_id"] is not None}),
            "planned_instance_slots": len({r.get("instance_slot_id", r["canonical_instance_id"]) for r in rows}),
            "build_failed": sum(r["terminal_status"] in {"BUILD_FAILED"} for r in rows),
            "abstained": sum(r["terminal_status"] in UNEXECUTED_FAILURE - {"BUILD_FAILED"} for r in rows),
            "unmeasured": len(rows) - len(measured),
            "rollout_coverage_observed": len(executed) / len(rows),
            "success_per_planned": successes / len(rows) if complete else None,
            "instance_macro_success": sum(sum(v) / len(v) for v in by_instance.values()) / len(by_instance) if complete else None,
            "success_per_executed": sum(r["success"] for r in executed) / len(executed) if executed else None}


def tables(units):
    groups = defaultdict(list)
    for row in units:
        groups[tuple(row[k] for k in BLOCK)].append(row)
    table, comparisons, per_task = [], [], []
    for block_key, rows in sorted(groups.items()):
        block = dict(zip(BLOCK, block_key))
        methods = defaultdict(list)
        for row in rows:
            methods[row["controller_method"]].append(row)
        def pair_key(r):
            return (r["canonical_instance_id"], r["reset_id"], r["policy_rng_seed"])
        reference = {pair_key(r): r for r in methods.get("REF_NATIVE", [])}
        for method, arm in sorted(methods.items()):
            aggregate = _aggregate(arm)
            aggregate.update(delta_vs_ref_pp=None, delta_ci95_low_pp=None, delta_ci95_high_pp=None)
            table.append(dict(block, method=method, **aggregate))
            for task in sorted({r["task_id"] for r in arm}):
                per_task.append(dict(block, method=method, task_id=task,
                                     **_aggregate([r for r in arm if r["task_id"] == task])))
            if method == "REF_NATIVE" or any(r["canonical_instance_id"] is None for r in arm):
                continue
            if len({pair_key(r) for r in arm}) != len(arm):
                raise ValueError("multiple units for one paired reset")
            reference_methods = ["REF_NATIVE"]
            reference_methods += {"B1_FIXED_PRIORITY": ["B0_FIXED_NATIVE"],
                                  "B2_EVIDENCE": ["B1_FIXED_PRIORITY"],
                                  "B3_AGENT_NATIVE": ["B0_FIXED_NATIVE", "B2_EVIDENCE"],
                                  "B4_ROOM_REPAIR_NATIVE": ["B3_AGENT_NATIVE", "BM_BUDGET_MATCHED_NATIVE", "V1_VERIFY_ABSTAIN_NATIVE"]}.get(method, [])
            for reference_method in reference_methods:
                reference = {pair_key(r): r for r in methods.get(reference_method, [])}
                pairs = []
                for row in arm:
                    ref = reference.get(pair_key(row))
                    if ref and any(ref[k] != row[k] for k in ("layout_id", "style_id", "task_id", "native_horizon")):
                        raise ValueError("paired native task/config differs")
                    if ref and ref["measured"] and row["measured"]:
                        pairs.append({"layout_id": row["layout_id"],
                                      "canonical_instance_id": row["canonical_instance_id"],
                                      "reset_id": row["reset_id"], "a": int(ref["service_success"]),
                                      "b": int(row["service_success"])})
                stats = paired_hierarchical_bootstrap(pairs)
                paired_complete = bool(reference) and {pair_key(r) for r in arm} == set(reference) and len(pairs) == len(arm)
                if paired_complete and reference_method == "REF_NATIVE":
                    table[-1].update(delta_vs_ref_pp=100 * stats["delta"],
                                     delta_ci95_low_pp=100 * stats["ci95"][0],
                                     delta_ci95_high_pp=100 * stats["ci95"][1])
                comparisons.append(dict(block, method=method, reference=reference_method,
                                        paired_population_complete=paired_complete,
                                        planned_method_resets=len(arm), planned_reference_resets=len(reference),
                                        delta_pp=100 * stats["delta"] if paired_complete else None,
                                        ci95_pp=[100 * v for v in stats["ci95"]] if paired_complete else [None, None],
                                        observed_pairs_descriptive=stats,
                                        reference_has_success=any(p["a"] for p in pairs),
                                        preservation_claim="NOT_RUN"))
    return table, comparisons, per_task


def geometry_table(units, records, construction_bindings=(), common_methods=None):
    """Ingest canonical evaluator receipts on a deduplicated build population.

    Missing geometry stays null, even for executed native episodes. Method
    build failure is retained in planned construction, not matched quality.
    """
    populations = defaultdict(dict)
    for row in units:
        key = (row["cohort_id"], row["scope"], row["sensor_regime"], row["split"])
        instance = row["canonical_instance_id"]
        if instance is not None:
            populations[key][instance] = row
    from robo.eval.native_scale_construction import construction_decisions
    decisions,lineage=construction_decisions(units,construction_bindings)
    common_methods=tuple(common_methods or ('B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE'))
    if len(set(common_methods))!=len(common_methods) or len(common_methods)<2:raise ValueError('declare distinct geometry common-support methods')
    found = {}; reference_ids={}
    for path in records:
        path = Path(path).resolve(); record = json.loads(path.read_text())
        if record.get("geometry_producer") != "robo.eval.fidelity_metrics.geometry_metrics" or record.get("evaluator_alignment") != "NONE":
            raise ValueError("geometry must use canonical metrics and frozen estimated alignment")
        lineage[str(path)] = {"path": str(path), "sha256": _sha(path)}
        # First evaluator revision omitted sensor spelling; authenticate its
        # existing immutable B0 source rather than recomputing correct metrics.
        if 'sensor_regime' not in record:
            b0 = next((r for r in record['rows'] if r['method'] == 'B0_FIXED_NATIVE'), None)
            if b0 is None:raise ValueError('legacy geometry needs frozen B0 sensor binding')
            build_path = Path(b0['object_dir']).parents[2] / 'build_manifest.json'
            if _sha(build_path) != record['b0_manifest_sha256']:
                raise ValueError('legacy geometry sensor source changed')
            record['sensor_regime'] = json.loads(build_path.read_text())['sensor_regime']
            lineage[str(build_path)] = {'path':str(build_path),'sha256':_sha(build_path)}
        if record.get('reference_parent'):
            parent=record['reference_parent'];parent_path=Path(parent['path'])
            if _sha(parent_path)!=parent['sha256']:raise ValueError('geometry reference parent changed')
            lineage[str(parent_path)]={'path':str(parent_path),'sha256':parent['sha256']}
        for method,receipt in record.get('context_sources',{}).items():
            if _sha(receipt['path'])!=receipt['sha256']:raise ValueError('geometry context source changed')
            context=json.loads(Path(receipt['path']).read_text())
            if context.get('accepted') is not True or receipt['accepted'] is not True:raise ValueError('unaccepted context geometry')
            lineage[receipt['path']]={'path':receipt['path'],'sha256':receipt['sha256']}
        for row in record["rows"]:
            key = (record["cohort_id"], record["scope"], {"ideal_rgbd": "ideal_rgbd_posed"}.get(record["sensor_regime"], record["sensor_regime"]), {"DEV": "development", "TEST": "test"}.get(record["tier"], record["tier"]), record["canonical_instance_id"], row["method"])
            if key in found:
                raise ValueError("duplicate canonical geometry measurement")
            for name, digest in record["frozen_estimated_artifacts"][row["method"]].items():
                if _sha(Path(row["object_dir"]) / name) != digest:
                    raise ValueError("frozen geometry source changed")
            if not all(not isinstance(row[k], bool) and isinstance(row[k], (float, int)) and __import__('math').isfinite(row[k]) for k in ("cd_cm", "f1_20")):
                raise ValueError("nonfinite geometry metric")
            if type(row.get('collapse')) is not bool:raise ValueError('invalid catastrophic collapse flag')
            if row['cd_cm'] < 0 or not 0 <= row['f1_20'] <= 1:
                raise ValueError("invalid geometry metric range")
            decision=decisions.get((record['canonical_instance_id'],row['method']))
            if decision and decision['accepted'] is False:raise ValueError('geometry supplied for unaccepted construction')
            reference_ids[key]=(record.get('reference_points_sha256'),record.get('canonical_manifest_sha256'))
            found[key] = row
    expected = {(c, scope, sensor, split, iid) for (c, scope, sensor, split), pop in populations.items() for iid in pop}
    if any(key[:5] not in expected for key in found):
        raise ValueError("geometry contains unplanned canonical instance")
    methods = ("OBSERVED_SURFACE_TSDF", "B0_FIXED_NATIVE", "B1_FIXED_PRIORITY", "B2_EVIDENCE", "B3_AGENT_NATIVE", "B4_ROOM_REPAIR_NATIVE", "BM_BUDGET_MATCHED_NATIVE")
    if any(m not in methods for m in common_methods):raise ValueError("unknown common geometry method")
    table = []
    for (cohort, scope, sensor, split), pop in sorted(populations.items()):
        prefix=(cohort,scope,sensor,split)
        common_ids=[]
        for iid in pop:
            keys=[(*prefix,iid,m) for m in common_methods]
            if all(k in found for k in keys):
                refs=[reference_ids[k] for k in keys]
                if any(x[0] is None or x[1] is None for x in refs):continue
                if any(x!=refs[0] for x in refs[1:]):raise ValueError('common geometry reference surfaces differ')
                common_ids.append(iid)
        for method in methods:
            measured = [found[(cohort, scope, sensor, split, iid, method)] for iid in pop if (cohort, scope, sensor, split, iid, method) in found]
            known=[decisions[(iid,method)] for iid in pop if (iid,method) in decisions and decisions[(iid,method)]['accepted'] is not None]
            common=[found[(*prefix,iid,method)] for iid in common_ids] if method in common_methods else []
            table.append(dict(cohort_id=cohort, scope=scope, sensor_regime=sensor, split=split,
                method=method, planned_objects=len(pop), matched_objects=len(measured),
                accepted_objects=sum(r['accepted'] for r in known) if len(known)==len(pop) else None,
                accepted_objects_observed=sum(r['accepted'] for r in known),build_decisions_available=len(known),
                build_decisions_unmeasured=len(pop)-len(known),abstained_objects=sum(r['terminal_status']=='ABSTAINED' for r in known),
                failed_build_objects=sum(r['terminal_status']=='BUILD_FAILED' for r in known),
                common_methods=list(common_methods),common_matched_objects=len(common),
                common_matched_instance_ids=sorted(common_ids) if method in common_methods else [],
                common_cd_cm=sum(r['cd_cm'] for r in common)/len(common) if common else None,
                common_f1_at_20mm=sum(r['f1_20'] for r in common)/len(common) if common else None,
                common_catastrophic_collapse_count=sum(r['collapse'] for r in common) if common else None,
                common_geometry_coverage=len(common)/len(pop),geometry_coverage=len(measured)/len(pop),
                cd_cm=sum(r['cd_cm'] for r in measured)/len(measured) if measured else None,
                f1_at_20mm=sum(r['f1_20'] for r in measured)/len(measured) if measured else None,
                catastrophic_collapse_count=sum(r['collapse'] for r in measured) if measured else None,
                matched_instance_ids=sorted(iid for iid in pop if (cohort, scope, sensor, split, iid, method) in found),
                aggregation='available quality is method-conditional; common quality uses identical canonical instances and reference samples; no unconditional selection gain',
                actual_stage_seconds_per_build=None))
    return table, lineage


def _latex(path, rows):
    """Mechanical rendering of CSV cells, including blank unmeasured cells."""
    if path.stem == "T2_manipulation":
        # JSON/CSV retain all accounting fields; paper view remains compact.
        rows = [{"Method": r["method"], "Block": "/".join(str(r[k]) for k in BLOCK),
                 "Instances/layouts": f"{r['instances']}/{r['layouts']}",
                 "Executed/planned": f"{r['executed']}/{r['planned']}",
                 "Success/planned": r["success_per_planned"],
                 "Success/executed": r["success_per_executed"],
                 "Delta pp": r["delta_vs_ref_pp"],
                 "95 pct CI": None if r["delta_vs_ref_pp"] is None else
                     f"[{r['delta_ci95_low_pp']:.2f}, {r['delta_ci95_high_pp']:.2f}]"}
                for r in rows]
    compact = {
        "T1a_appearance": ("method", "scenes_available", "scenes_planned", "common_scenes", "common_views", "psnr", "ssim", "lpips"),
        "T3_verification": ("method", "instances_probed", "instances_planned", "tasks_accepted", "tasks_planned", "success_per_planned", "independent_room_stability", "extra_tool_calls", "measured_context_seconds"),
        "construction_stage_costs": ("stage", "instances_timed", "instances_planned", "conditional_seconds_per_timed_instance", "whole_method_seconds"),
        "full_horizon_diagnostic": ("method", "executed", "planned", "completed_full_horizon", "success_ever", "success_at_horizon", "absolute_position_error_cm"),
        "full_horizon_episodes": ("seed", "method", "recorded_steps", "planned_steps", "success_at_horizon", "relative_displacement_rmse_cm", "position_rmse_cm"),
        "context_diagnostics": ("instance_slot_id", "method", "accepted", "actual_calls", "after_drift_m", "after_penetration_m", "after_public_residual_m", "independent_room_stability"),
        "T4_scope": ("scope_run_id", "method", "scope", "instances", "layouts", "executed", "planned", "success_per_planned", "success_per_executed", "rollout_coverage_observed"),
        "T4_native_controls": ("scope_run_id", "method", "scope", "instances", "layouts", "executed", "planned", "success_per_planned", "success_per_executed", "rollout_coverage_observed"),
        "T4_scope_comparisons": ("scope_run_id", "method", "baseline_scope", "scope", "planned_pairs", "measured_pairs", "delta_pp", "ci95_pp"),
        "T4_scope_failures": ("scope_run_id", "canonical_instance_id", "method", "scope", "reset_id", "terminal_status", "measured", "executed", "success"),
    }
    if path.stem in compact:
        rows = [{key: (float("inf") if key == "psnr" and row.get("psnr_status") == "POSITIVE_INFINITY" else row.get(key)) for key in compact[path.stem]} for row in rows]
    fields = list(rows[0]) if rows else ["method"]
    def escape(value):
        if isinstance(value, float) and value == float("inf"):
            return r"$\infty$"
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    content = [r"\begin{tabular}{" + "l" * len(fields) + "}", " & ".join(map(escape, fields)) + r" \\"]
    content += [" & ".join(escape(row.get(k)) for k in fields) + r" \\" for row in rows]
    content += [r"\end{tabular}"]
    path.write_text("\n".join(content) + "\n")



def episode_media_sources(units):
    """Bind existing worker media at release time, without claiming an earlier seal.

    Missing media never changes an experiment outcome. Final demo consumers must
    require a matching sealed source instead of hashing an unbound current file.
    """
    sources, records = {}, []
    for index, unit in enumerate(units):
        if not unit.get('executed'):
            continue
        record = dict(unit_id=unit['unit_id'], canonical_instance_id=unit['canonical_instance_id'],
                      seal_boundary='table_release_creation', original_acquisition_seal_established=False,
                      media={})
        if not unit.get('result_path'):
            record['state']='NO_BOUND_WORKER_RESULT'; records.append(record); continue
        path=Path(unit['result_path']).resolve()
        if _sha(path)!=unit['result_sha256']:
            raise ValueError('worker result changed before media sealing')
        result=json.loads(path.read_text())
        if result.get('executed') is not True or result.get('success')!=unit['success']:
            raise ValueError('worker execution outcome differs during media sealing')
        record.update(state='RESULT_BOUND',result=dict(path=str(path),sha256=unit['result_sha256']),
                      video_error=result.get('video_error'))
        for key, digest_key in (('video_path','video_sha256'), ('timeseries_path','timeseries_sha256'), ('actions_path','actions_sha256')):
            value=result.get(key)
            if not value:
                record['media'][key]=dict(state='NOT_PROVIDED'); continue
            artifact=Path(value)
            if not artifact.is_absolute():artifact=path.parent/artifact
            artifact=artifact.resolve()
            if not artifact.is_file():
                record['media'][key]=dict(state='MISSING',path=str(artifact)); continue
            source_key=f'unit_{index}_{key}'
            sources[source_key]=artifact
            record['media'][key]=dict(state='AVAILABLE_TO_SEAL',source_key=source_key,
                expected_result_sha256=result.get(digest_key),prior_result_digest_present=bool(result.get(digest_key)))
        records.append(record)
    return sources,records


def finish_media_seal(records,lineage):
    from datetime import datetime,timezone
    for record in records:
        for media in record['media'].values():
            if media['state']!='AVAILABLE_TO_SEAL':continue
            binding=lineage[media['source_key']]
            expected=media.pop('expected_result_sha256')
            if expected is not None and expected!=binding['sha256']:
                raise ValueError('worker media bytes differ from existing result digest')
            media.update(state='SEALED_AT_RELEASE',**binding)
    return dict(schema_version=1,seal_boundary='table_release_creation',
        sealed_at_utc=datetime.now(timezone.utc).isoformat(),original_acquisition_seal_established=False,
        interpretation='Existing trusted worker artifacts sealed at this release boundary. No claim of cryptographic video sealing at original acquisition; unavailable media does not alter measured outcomes.',
        executed_units=len(records),records=records)



def seal_retry_action_sources(lineage):
    """Expose existing genuine registration-action records to strict demo consumers."""
    records=[]
    for source in list(lineage.values()):
        path=Path(source['path'])
        if path.name!='candidate_pool.json':continue
        if _sha(path)!=source['sha256']:raise ValueError('candidate pool changed before retry sealing')
        pool=json.loads(path.read_text())
        if pool.get('retry_candidate') is None:continue
        record=dict(candidate_pool=source,seal_boundary='table_release_creation',sidecars={})
        for filename in ('registration.json','evidence.json'):
            sidecar=(path.parent/'registration_retry'/filename).resolve()
            if sidecar.is_file():
                binding=dict(path=str(sidecar),sha256=_sha(sidecar))
                lineage[str(sidecar)]=binding
                record['sidecars'][filename]=dict(state='SEALED_AT_RELEASE',**binding)
            else:record['sidecars'][filename]=dict(state='MISSING',path=str(sidecar))
        records.append(record)
    return dict(seal_boundary='table_release_creation',original_acquisition_seal_established=False,records=records)



def adjudication_tables(units, geometry):
    """Pass audited classification facts through to generated paper disclosure."""
    records=[];audits={}
    for row in units:
        proof=row.get('classification_adjudication')
        if not proof:continue
        if (row.get('original_terminal_status')!='CODE_FAILED' or row['terminal_status']!='BUILD_FAILED'
            or row['executed'] is not False or row['success'] is not None
            or row.get('failure_classification')!='invalid_supplied_collision_volume'):
            raise ValueError('paper classification disclosure requires prepolicy invalid-collision build failure')
        source=proof['independent_audit'];key=(source['path'],source['sha256'])
        if key not in audits:
            if _sha(source['path'])!=source['sha256']:raise ValueError('classification audit bytes changed')
            audits[key]=json.loads(Path(source['path']).read_text())
        audit=audits[key];properties=audit['part_properties']
        if audit['geometry_repaired'] is not False or audit['threshold_changed'] is not False or audit['new_policy_outcomes']!=0:
            raise ValueError('classification audit changed geometry, thresholds or outcomes')
        records.append(dict(unit_id=row['unit_id'],canonical_instance_id=row['canonical_instance_id'],
            method=row['controller_method'],cohort_id=row['cohort_id'],scope=row['scope'],split=row['split'],
            original_terminal_status=row['original_terminal_status'],terminal_status=row['terminal_status'],
            executed=row['executed'],native_success=row['success'],failure_classification=row['failure_classification'],
            classification_adjudication=proof,collision_watertight=properties['is_watertight'],collision_convex=properties['is_convex'],
            near_threshold_nonconvexity=any('Near-threshold local nonconvexity' in x for x in audit['limitations']),
            geometry_repaired=False,threshold_changed=False,new_policy_outcomes=0))
    summary=[]
    for method in sorted({r['method'] for r in records}):
        selected=[r for r in records if r['method']==method]
        arm=[r for r in units if r['controller_method']==method]
        blocks={(r['cohort_id'],r['scope'],r['split'].lower()) for r in arm}
        if len(blocks)!=1:raise ValueError('adjudication summary mixes native cohorts')
        block=next(iter(blocks))
        geom=[r for r in geometry if r['method']==method and (r.get('cohort_id'),r.get('scope'),r.get('split','').lower())==block]
        if len(geom)>1:raise ValueError('adjudication producer-acceptance source is ambiguous')
        summary.append(dict(method=method,cohort_id=block[0],scope=block[1],split=block[2],
            classified_units=len(selected),affected_instances=len({r['canonical_instance_id'] for r in selected}),
            producer_accepted_objects=geom[0].get('accepted_objects') if geom else None,
            native_executed_instances=len({r['canonical_instance_id'] for r in arm if r['executed']}),
            cohort_complete=all(r['measured'] for r in arm),
            all_watertight_near_threshold_nonconvex=all(r['collision_watertight'] and not r['collision_convex'] and r['near_threshold_nonconvexity'] for r in selected),
            geometry_repaired=False,threshold_changed=False,native_success_values_changed=False))
    return records,summary


def generate(config_path, out_dir, paper_root=None):
    config_path, out_dir = Path(config_path).resolve(), Path(out_dir)
    config = yaml.safe_load(config_path.read_text())["native_scale_up"]
    if paper_root is not None and config.get("native_paper_section") is not True:
        raise ValueError("native paper transfer requires an explicit native_paper_section configuration")
    def source(key):
        p = Path(config[key])
        return p.resolve() if p.is_absolute() else (config_path.parent / p).resolve()
    sources = {"config": config_path, "planned_units": source("planned_units"),
               "table_implementation": Path(__file__).resolve(),
               "statistics_implementation": Path(__file__).with_name("metric_utils.py").resolve(),
               "pipeline_implementation": Path(__file__).with_name("paper_pipeline.py").resolve()}
    if config.get("input_boundary"):
        boundary = config["input_boundary"]
        path = Path(boundary["path"]).resolve()
        if _sha(path) != boundary["sha256"]:
            raise ValueError("release input boundary bytes changed")
        sources["input_boundary"] = path
    if config.get("mode") == "planning":
        if "episode_ledger" in config:
            raise ValueError("unresolved planning mode cannot ingest measured ledgers")
        units = load_planning_slots(sources["planned_units"])
    else:
        sources["episode_ledger"] = source("episode_ledger")
        units = load_units(sources["planned_units"], sources["episode_ledger"])
    main, comparisons, per_task = tables(units)
    if config.get("native_paper_section"):
        sources["native_paper_implementation"] = Path(__file__).with_name("native_scale_paper.py").resolve()
    for index, row in enumerate(units):
        for key, binding in row.get('classification_adjudication', {}).items():
            path=Path(binding['path']).resolve()
            if _sha(path)!=binding['sha256']:
                raise ValueError('terminal classification evidence changed')
            sources[f'unit_{index}_classification_{key}']=path
        for key, hash_key in (("result_path", "result_sha256"), ("canonical_manifest", "canonical_manifest_file_sha256"),
                              ("reset_bank", "reset_bank_sha256")):
            if row.get(key):
                path = Path(row[key]).resolve()
                if _sha(path) != row.get(hash_key):
                    raise ValueError(f"source binding differs: {key} for {row['unit_id']}")
                sources[f"unit_{index}_{key}"] = path
    media_sources, media_records = episode_media_sources(units)
    sources.update(media_sources)
    lineage = {k: {"path": str(p), "sha256": _sha(p)} for k, p in sources.items()}
    media_seal = finish_media_seal(media_records, lineage)
    from robo.eval.native_scale_evidence import evidence_tables
    extra_tables, extra_lineage = evidence_tables(units, main, comparisons, config)
    lineage.update(extra_lineage)
    retry_action_seal = seal_retry_action_sources(lineage)
    scope_tables_out = {}
    if config.get("scope_sources"):
        from robo.eval.native_scale_scope import scope_tables
        scope_tables_out, scope_lineage = scope_tables(config["scope_sources"])
        scope_implementation = Path(__file__).with_name("native_scale_scope.py").resolve()
        lineage["scope_implementation"] = {"path": str(scope_implementation), "sha256": _sha(scope_implementation)}
        lineage.update(scope_lineage)
    if out_dir.exists():
        raise FileExistsError(f"immutable release output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    write_json(out_dir / "episode_media_seal.json", media_seal)
    write_json(out_dir / "retry_action_seal.json", retry_action_seal)
    skeletons = {
        "T1a_appearance": [dict(method=m, scenes_available=None, scenes_planned=None,
                                  views_available=None, views_planned=None, psnr=None, ssim=None, lpips=None)
                           for m in ("SOURCE_GS", "B0_COMPOSITE", "B3_COMPOSITE", "B4_COMPOSITE")],
        "T1b_geometry": [dict(method=m, accepted_objects=None, planned_objects=None,
                                matched_objects=None, cd_cm=None, f1_at_20mm=None, actual_stage_seconds_per_build=None)
                         for m in ("OBSERVED_SURFACE_TSDF", "B0_FIXED", "B1_PRIORITY", "B2_SELECTION", "B3_AGENT", "B4_ROOM_REPAIR", "BM_BUDGET_MATCHED")],
        "T2_manipulation": main,
        "T3_verification": [dict(method=m, tasks_accepted=None, tasks_planned=None,
                                    success_per_planned=None, independent_room_stability=None,
                                    extra_tool_calls=None, actual_seconds=None)
                             for m in ("B3_AGENT_NATIVE", "V1_VERIFY_ABSTAIN_NATIVE", "B4_ROOM_REPAIR_NATIVE", "BM_BUDGET_MATCHED_NATIVE")],
        "T4_scope": [dict(method=m, scope=s, reconstructed_context=None, retained_context=None,
                             executed=None, planned=None, success_per_planned=None,
                             absolute_replay_position_error_cm=None, rollout_coverage=None)
                      for m in ("B3_AGENT_NATIVE", "B4_ROOM_REPAIR_NATIVE") for s in ("L0_target_only", "L1_target_destination", "L2_workspace")],
        "per_task_manipulation": per_task,
    }
    if config.get("geometry_records"):
        geometry, geometry_lineage = geometry_table(units, config["geometry_records"], config.get("construction_bindings",()), config.get("geometry_common_methods"))
        skeletons["T1b_geometry"] = geometry
        skeletons['T1b_geometry_common']=[{k:r[k] for k in ('cohort_id','scope','sensor_regime','split','method','planned_objects','accepted_objects','accepted_objects_observed','build_decisions_available','matched_objects','common_matched_objects','common_matched_instance_ids','common_methods','common_cd_cm','common_f1_at_20mm','common_catastrophic_collapse_count','aggregation')} for r in geometry if r['method'] in r['common_methods']]
        lineage.update({"geometry_" + str(i): r for i, r in enumerate(geometry_lineage.values())})
    skeletons.update(extra_tables)
    skeletons.update(scope_tables_out)
    adjudications, adjudication_summary = adjudication_tables(units, skeletons['T1b_geometry'])
    skeletons['native_import_adjudications'] = adjudications
    skeletons['native_import_adjudication_summary'] = adjudication_summary
    for name, rows in skeletons.items():
        write_json(out_dir / f"{name}.json", rows)
        # Different scope rows carry distinct applicability fields. Preserve the
        # union in CSV; an absent field stays blank rather than being dropped.
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(out_dir / f"{name}.csv", rows, fields or None)
        _latex(out_dir / f"{name}.tex", rows)
    paper_outputs = {}
    if config.get("native_paper_section"):
        from robo.eval.native_scale_paper import render_section
        paper_outputs = render_section(out_dir, skeletons, require_complete=paper_root is not None, comparisons=comparisons)
    write_json(out_dir / "comparison_statistics.json", comparisons)
    failures = [dict(unit_id=r["unit_id"], terminal_status=r["terminal_status"],
                     measured=r["measured"], executed=r["executed"], success=r["success"])
                for r in units if r["success"] is not True]
    write_csv(out_dir / "failure_inventory.csv", failures,
              ["unit_id", "terminal_status", "measured", "executed", "success"])
    scopes = [dict(zip(BLOCK, k), planned=v) for k, v in
              sorted(Counter(tuple(r[f] for f in BLOCK) for r in units).items())]
    write_csv(out_dir / "scope_inventory.csv", scopes)
    claims = []
    names = ("feasibility", "multi_instance_usability", "agentic_benefit", "beyond_compute",
             "repair_beyond_rejection", "closed_loop_compensation", "scope_expansion", "appearance_preservation")
    for name in names:
        claims.append(dict(claim_id=name, implementation_gate="PASS", scientific_gate="NOT_RUN",
                           observed_result="See measured counts; no automatic scientific promotion",
                           enabled_sentence="", required_gate="Independent measured evidence and declared claim-specific gate"))
    write_csv(out_dir / "claim_ledger.csv", claims)
    provenance = {"schema_version": 1, "sources": lineage,
                  "producer": "robo.eval.paper_pipeline:native_scale_up",
                  "units": {"success": "binary native evaluator", "delta_pp": "percentage points",
                            "coverage": "executed/planned", "absolute_replay_position_error_cm": "unmeasured"},
                  "denominators": "planned roster; external/pending missing outcomes are null; method build failures and abstentions are service failures",
                  "limitations": ["Unprovided appearance, scope and independent physics remain unmeasured; context self-checks are not independent validation", "Scientific claim gates are not automatically promoted", "Small layout count gives descriptive uncertainty"]}
    write_json(out_dir / "table_provenance.json", provenance)
    write_json(out_dir / "paper_table_provenance.json", provenance)
    write_json(out_dir / "demo_source_manifest.json", {"state": "NOT_RUN", "sources": [],
               "reason": "Requires real multi-instance continuous footage bound to this release"})
    release = {"schema_version": 1, "state": "COMPLETE_BLOCKS" if all(r["complete"] for r in main) else "INCOMPLETE",
               "source_lineage": lineage, "planned_units": len(units),
               "measured_units": sum(r["measured"] for r in units),
               "artifacts": {p.name: _sha(p) for p in sorted(out_dir.iterdir()) if p.is_file()}}
    write_json(out_dir / "release_manifest.json", release)
    if paper_root is not None:
        from robo.eval.paper_pipeline import _copy
        for name, destination in paper_outputs.items():
            _copy(out_dir / name, Path(paper_root), destination)
        paper_tier = "test" if "native_test_section.tex" in paper_outputs else "dev"
        write_json(Path(paper_root) / f"audit/native_{paper_tier}_transfer.json", {"release": str(out_dir.resolve()),
            "release_sha256": _sha(out_dir / "release_manifest.json"), "producer": "robo.eval.paper_pipeline",
            "files": {name: {"destination": destination, "sha256": _sha(out_dir / name)} for name, destination in paper_outputs.items()},
            "scope": f"complete primary native {paper_tier.upper()}; no preservation claim",
            "primary_complete": all(r["complete"] for r in main),
            "scope_phase_is_separate": True})
    return release
