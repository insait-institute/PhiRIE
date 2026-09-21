"""Typed specification for the SimAny paired robot-evaluation harness.

The paper compares scene construction, collision, and observation treatments.
This module makes those axes explicit and rejects configurations that change
more than one axis inside a declared comparison block.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

TreatmentAxis = Literal["scene", "collision", "observation"]
# ``simany``/``box_proxy`` are the original harness treatments.  The two
# construction-policy names are the E3 -> E4 hand-off: A0 is the fixed
# single-path build and A4 is the agentic selected build.
SceneVariant = Literal[
    "simany", "box_proxy", "fixed_single_path", "agentic",
]
CollisionVariant = Literal["full_room", "private_shims"]
ObservationVariant = Literal["raster", "composite_raw", "harmonizer_c"]

_AXIS_OPTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "scene": ("factory_dir", "tasks_json", "scene_xml"),
    "collision": ("collision_variants", "tasks_json", "scene_xml"),
    "observation": (
        "source_factory", "camera_names", "robot_roots", "enhancer",
        "restoration", "image_keys",
    ),
}


def _flatten_options(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_options(child, path))
        return result
    return {prefix: value}


@dataclass(frozen=True)
class TreatmentSpec:
    """One executable harness condition."""

    id: str
    scene: SceneVariant = "simany"
    collision: CollisionVariant = "full_room"
    observation: ObservationVariant = "raster"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or any(c.isspace() for c in self.id):
            raise ValueError("treatment id must be a non-empty token without spaces")
        if self.scene not in {
            "simany", "box_proxy", "fixed_single_path", "agentic",
        }:
            raise ValueError(f"unsupported scene variant: {self.scene}")
        if self.collision not in {"full_room", "private_shims"}:
            raise ValueError(f"unsupported collision variant: {self.collision}")
        if self.observation not in {"raster", "composite_raw", "harmonizer_c"}:
            raise ValueError(f"unsupported observation variant: {self.observation}")

    def semantic_dict(self) -> dict[str, str]:
        return {"scene": self.scene, "collision": self.collision,
                "observation": self.observation}

    def changed_axes(self, other: "TreatmentSpec") -> set[str]:
        return {k for k, value in self.semantic_dict().items()
                if value != other.semantic_dict()[k]}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonSpec:
    id: str
    axis: TreatmentAxis
    treatments: tuple[str, ...]
    baseline: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("comparison id must not be empty")
        if self.axis not in {"scene", "collision", "observation"}:
            raise ValueError(f"invalid treatment axis: {self.axis}")
        if len(self.treatments) < 2:
            raise ValueError("a comparison needs at least two treatments")
        if len(set(self.treatments)) != len(self.treatments):
            raise ValueError(f"duplicate treatment in comparison {self.id}")
        if self.baseline is not None and self.baseline not in self.treatments:
            raise ValueError("comparison baseline must be one of its treatments")


@dataclass(frozen=True)
class HarnessSpec:
    treatments: dict[str, TreatmentSpec]
    comparisons: tuple[ComparisonSpec, ...]
    raw: dict[str, Any]

    def validate(self) -> None:
        for treatment in self.treatments.values():
            overrides = treatment.options.get("contract_overrides", {})
            if overrides:
                if not isinstance(overrides, dict):
                    raise ValueError(
                        f"treatment {treatment.id!r} contract_overrides must be a mapping")
                fields = sorted(str(key) for key in overrides)
                raise ValueError(
                    f"treatment {treatment.id!r} has undeclared treatment drift in "
                    f"frozen contract fields: {fields}")
        for comparison in self.comparisons:
            missing = [t for t in comparison.treatments if t not in self.treatments]
            if missing:
                raise ValueError(
                    f"comparison {comparison.id} references unknown treatments {missing}")
            baseline = self.treatments[comparison.baseline or comparison.treatments[0]]
            for treatment_id in comparison.treatments:
                current = self.treatments[treatment_id]
                changed = baseline.changed_axes(current)
                if changed - {comparison.axis}:
                    raise ValueError(
                        f"comparison {comparison.id!r} declares axis={comparison.axis!r} "
                        f"but {baseline.id!r}->{current.id!r} changes {sorted(changed)}")
                if treatment_id != baseline.id and not changed:
                    raise ValueError(
                        f"comparison {comparison.id!r} contains duplicate semantics: "
                        f"{baseline.id!r} and {current.id!r}")
                baseline_options = _flatten_options(baseline.options)
                current_options = _flatten_options(current.options)
                allowed_prefixes = _AXIS_OPTION_PREFIXES[comparison.axis]
                for path in sorted(set(baseline_options) | set(current_options)):
                    if path == "label" or any(
                        path == prefix or path.startswith(prefix + ".")
                        for prefix in allowed_prefixes
                    ):
                        continue
                    if baseline_options.get(path) != current_options.get(path):
                        raise ValueError(
                            f"comparison {comparison.id!r} has undeclared treatment "
                            f"drift at options.{path}")

    def treatment_ids(self) -> tuple[str, ...]:
        return tuple(self.treatments)


def _parse_treatment(value: Any) -> TreatmentSpec:
    if isinstance(value, str):
        if value == "simany":
            return TreatmentSpec(id=value, scene="simany")
        if value in {"reference", "box_proxy"}:
            return TreatmentSpec(id=value, scene="box_proxy")
        raise ValueError(f"unknown legacy condition {value!r}")
    if not isinstance(value, dict):
        raise TypeError(f"treatment must be a string or mapping, got {type(value)}")
    return TreatmentSpec(
        id=str(value["id"]), scene=value.get("scene", "simany"),
        collision=value.get("collision", "full_room"),
        observation=value.get("observation", "raster"),
        options=dict(value.get("options", {})))


def load_harness_spec(config: dict[str, Any] | str | Path) -> HarnessSpec:
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text())
    if not isinstance(config, dict):
        raise TypeError("harness config must be a mapping")
    values = config.get("treatments", config.get("conditions", ()))
    treatments_list = [_parse_treatment(v) for v in values]
    treatments = {t.id: t for t in treatments_list}
    if len(treatments) != len(treatments_list):
        raise ValueError("treatment ids must be unique")
    comparison_values = config.get("comparisons")
    if comparison_values is None:
        comparison_values = [{
            "id": "legacy_scene_comparison", "axis": "scene",
            "treatments": [t.id for t in treatments_list],
            "baseline": treatments_list[0].id if treatments_list else None}]
    comparisons = tuple(ComparisonSpec(
        id=str(v["id"]), axis=v["axis"], treatments=tuple(v["treatments"]),
        baseline=v.get("baseline")) for v in comparison_values)
    spec = HarnessSpec(treatments=treatments, comparisons=comparisons, raw=config)
    spec.validate()
    # Legacy smoke configs predate the global frozen contract.  Once a config
    # opts into the contract (and always in paper mode), validate it at load
    # time so a run cannot start with a placeholder policy/Harmonizer or an
    # incomplete paired-field declaration.
    requires_e4_contract = any(
        treatment.scene in {"fixed_single_path", "agentic"}
        for treatment in treatments.values()
    )
    if requires_e4_contract and "contract" not in config:
        raise ValueError(
            "E4 construction variants require an explicit global frozen contract"
        )
    if "contract" in config or bool(config.get("paper_mode", False)):
        from robo.eval.harness_validation import validate_harness_contract

        report = validate_harness_contract(spec)
        if not report["ok"]:
            raise ValueError("invalid harness contract: " + " | ".join(
                report["violations"]))
    return spec
