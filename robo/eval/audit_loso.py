"""Leave-one-scene-out low-capacity task-local audit predictions.

The input is a CSV containing ``scene_id``, ``task_id``, ``invalid_label``, and
numeric evidence columns. Evidence groups are identified by stable prefixes:
``global_psnr``, ``global_f1``, ``scene_``, ``visual_``, ``geometry_``,
``support_``, and ``robot_``. Standardization, median imputation, and missingness
indicators are fit inside each training fold. The model is L2-regularized
logistic regression implemented with NumPy to keep the evaluation auditable.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robo.eval.metric_utils import write_csv, write_json


SIGNAL_GROUP_NAMES = (
    "Global PSNR only",
    "Global geometry F1 only",
    "Scene-global evidence",
    "Manipulated-object evidence",
    "Task graph, visual + geometry",
    "Full task-interaction graph",
)


@dataclass
class FoldTransform:
    columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows, columns):
        matrix = _matrix(rows, columns)
        median = np.nanmedian(matrix, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        return cls(list(columns), median, mean, std)

    def transform(self, rows):
        matrix = _matrix(rows, self.columns)
        missing = (~np.isfinite(matrix)).astype(np.float64)
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        normalized = (filled - self.mean) / self.std
        return np.concatenate([normalized, missing], axis=1)


def _value(row, column):
    value = row.get(column)
    if value in {None, "", "--", "nan", "NaN"}:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _matrix(rows, columns):
    return np.asarray([[_value(row, column) for column in columns]
                       for row in rows], dtype=np.float64)


def sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(features, labels, *, l2=1.0, iterations=100, tolerance=1e-8):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    if labels.min() == labels.max():
        p = np.clip(labels.mean(), 1e-4, 1.0 - 1e-4)
        weights = np.zeros(design.shape[1], dtype=np.float64)
        weights[0] = np.log(p / (1.0 - p))
        return weights
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1]) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        probability = sigmoid(design @ weights)
        curvature = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (probability - labels) + penalty @ weights
        hessian = design.T @ (design * curvature[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights -= step
        if float(np.linalg.norm(step)) < tolerance:
            break
    return weights


def predict_logistic(features, weights):
    features = np.asarray(features, dtype=np.float64)
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    return sigmoid(design @ weights)


def numeric_columns(rows):
    excluded = {"scene_id", "task_id", "invalid_label", "invalid_reasons"}
    columns = []
    for key in rows[0]:
        if key in excluded:
            continue
        values = [_value(row, key) for row in rows]
        if any(np.isfinite(values)):
            columns.append(key)
    return columns


def signal_groups(columns):
    def prefixed(*prefixes):
        return [column for column in columns
                if any(column.startswith(prefix) for prefix in prefixes)]
    stale = prefixed("global_lpips")
    if stale:
        raise ValueError(
            "stale Global LPIPS audit columns are not part of the six-row "
            f"E6 contract: {stale}")
    groups = {
        "Global PSNR only": prefixed("global_psnr_"),
        "Global geometry F1 only": prefixed("global_f1_"),
        "Scene-global evidence": prefixed("scene_"),
        # Real E6 rows separate manipulated-object scope from graph scope.
        # Preserve the original synthetic smoke schema for old fixtures.
        "Manipulated-object evidence": prefixed("object_") or prefixed("visual_"),
        "Task graph, visual + geometry": prefixed("visual_", "geometry_"),
        "Full task-interaction graph": prefixed(
            "visual_", "geometry_", "support_", "robot_"),
    }
    missing = [name for name, values in groups.items() if not values]
    if missing:
        raise ValueError(f"missing required E6 signal group(s): {missing}")
    return groups


def loso_predictions(rows, *, l2=1.0):
    columns = numeric_columns(rows)
    groups = signal_groups(columns)
    if not groups:
        raise ValueError("no recognized audit evidence columns")
    scenes = sorted({str(row["scene_id"]) for row in rows})
    output = []
    for heldout_scene in scenes:
        train = [row for row in rows if str(row["scene_id"]) != heldout_scene]
        test = [row for row in rows if str(row["scene_id"]) == heldout_scene]
        if not train or not test:
            continue
        training_scenes = sorted({str(row["scene_id"]) for row in train})
        train_labels = np.asarray([int(row["invalid_label"]) for row in train])
        for signal, signal_columns in groups.items():
            transform = FoldTransform.fit(train, signal_columns)
            x_train = transform.transform(train)
            x_test = transform.transform(test)
            weights = fit_logistic(x_train, train_labels, l2=l2)
            probabilities = predict_logistic(x_test, weights)
            for row, probability in zip(test, probabilities):
                output.append({
                    "scene_id": row["scene_id"], "task_id": row["task_id"],
                    "signal": signal,
                    "invalid_label": int(row["invalid_label"]),
                    "invalid_probability": float(probability),
                    "heldout_scene": heldout_scene,
                    "training_scenes": ";".join(training_scenes),
                    "feature_columns": ";".join(signal_columns),
                })
    return output, groups


def _read(path):
    return list(csv.DictReader(Path(path).open()))


def generate(input_csv, out_dir, *, l2=1.0):
    rows = _read(input_csv)
    if not rows:
        raise ValueError("audit feature CSV is empty")
    predictions, groups = loso_predictions(rows, l2=l2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "heldout_predictions.csv", predictions)
    write_json(out_dir / "heldout_predictions.json", {
        "l2": l2, "groups": groups, "rows": predictions})
    return {"n_queries": len(rows), "n_predictions": len(predictions),
            "signals": groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.input, args.out, l2=args.l2), indent=2))


if __name__ == "__main__":
    main()
