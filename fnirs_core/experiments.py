"""Experiment runner with subject-wise validation for fNIRS workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .data import load_fNIRS_data, make_demo_nirs_data
from .models import ModelConfig, create_model
from .preprocessing import PreprocessConfig, PreprocessingPipeline


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot run."""


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "Quick fNIRS Experiment"
    dataset_id: str | None = None
    dataset_path: str | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    validation_strategy: str = "loso"
    num_folds: int = 5
    seed: int = 42
    output_dir: str = "artifacts/experiments"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FoldResult:
    fold_name: str
    train_size: int
    val_size: int
    accuracy: float
    labels: list[int]
    predictions: list[int]
    subject_ids: list[str]


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    name: str
    status: str
    metrics: dict[str, Any]
    folds: list[FoldResult]
    output_dir: str
    checkpoint_path: str
    config: dict[str, Any]
    preprocessing_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["folds"] = [asdict(fold) for fold in self.folds]
        return payload


def run_experiment(
    experiment_id: str,
    config: ExperimentConfig | dict[str, Any],
    *,
    progress: Callable[[float, str], None] | None = None,
) -> ExperimentResult:
    cfg = config if isinstance(config, ExperimentConfig) else ExperimentConfig(**config)
    rng = np.random.default_rng(cfg.seed)
    progress = progress or (lambda _progress, _message: None)
    progress(0.05, "正在加载 fNIRS 数据")

    if cfg.dataset_path:
        data = load_fNIRS_data(cfg.dataset_path)
    else:
        data = make_demo_nirs_data(seed=cfg.seed)

    progress(0.15, "正在执行预处理和 epoch 提取")
    preprocess_cfg = PreprocessConfig(**cfg.preprocessing)
    preprocessing_result = PreprocessingPipeline(preprocess_cfg).run(data)
    x = preprocessing_result.epochs
    y = preprocessing_result.labels.astype(int)
    groups = np.asarray(preprocessing_result.groups).astype(str)
    if len(np.unique(y)) < 2:
        raise ExperimentError("Experiment requires at least two labels.")

    folds = build_subject_folds(groups, strategy=cfg.validation_strategy, num_folds=cfg.num_folds, seed=cfg.seed)
    if not folds:
        raise ExperimentError("No validation folds could be created.")

    model_cfg = ModelConfig(**cfg.model, seed=cfg.seed)
    fold_results: list[FoldResult] = []
    all_labels: list[int] = []
    all_predictions: list[int] = []
    progress_base = 0.25

    for fold_index, (fold_name, train_indices, val_indices) in enumerate(folds):
        progress(
            progress_base + 0.6 * fold_index / max(len(folds), 1),
            f"正在训练 {fold_name}",
        )
        model = create_model(model_cfg)
        model.fit(x[train_indices], y[train_indices])
        predictions = model.predict(x[val_indices]).astype(int)
        labels = y[val_indices].astype(int)
        accuracy = float(np.mean(predictions == labels)) if labels.size else 0.0
        fold_results.append(
            FoldResult(
                fold_name=fold_name,
                train_size=int(len(train_indices)),
                val_size=int(len(val_indices)),
                accuracy=round(accuracy, 4),
                labels=labels.tolist(),
                predictions=predictions.tolist(),
                subject_ids=groups[val_indices].tolist(),
            )
        )
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

    progress(0.9, "正在保存实验结果")
    output_dir = Path(cfg.output_dir) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "prototype_model.json"
    checkpoint_payload = {
        "model_family": model_cfg.normalized_family(),
        "seed": int(cfg.seed),
        "note": "Prototype classifier checkpoint metadata. Re-run experiment to rebuild centroids.",
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = build_metrics(np.asarray(all_labels, dtype=int), np.asarray(all_predictions, dtype=int))
    result = ExperimentResult(
        experiment_id=experiment_id,
        name=cfg.name,
        status="succeeded",
        metrics=metrics,
        folds=fold_results,
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        config=cfg.to_dict(),
        preprocessing_summary=preprocessing_result.summary,
    )
    (output_dir / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    progress(1.0, "实验完成")
    return result


def build_subject_folds(
    groups: np.ndarray,
    *,
    strategy: str = "loso",
    num_folds: int = 5,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        indices = np.arange(len(groups))
        if len(indices) < 2:
            return []
        split = min(max(1, int(len(indices) * 0.8)), len(indices) - 1)
        return [("holdout", indices[:split], indices[split:])]

    if strategy.lower() == "loso":
        folds = []
        for subject in unique_groups:
            val_indices = np.where(groups == subject)[0]
            train_indices = np.where(groups != subject)[0]
            if len(train_indices) and len(val_indices):
                folds.append((f"LOSO {subject}", train_indices, val_indices))
        return folds

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_groups)
    rng.shuffle(shuffled)
    num_folds = max(2, min(int(num_folds), len(shuffled)))
    group_splits = np.array_split(shuffled, num_folds)
    folds = []
    for index, split_groups in enumerate(group_splits, start=1):
        val_mask = np.isin(groups, split_groups)
        val_indices = np.where(val_mask)[0]
        train_indices = np.where(~val_mask)[0]
        if len(train_indices) and len(val_indices):
            folds.append((f"GroupKFold {index}", train_indices, val_indices))
    return folds


def build_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    classes = sorted(set(labels.tolist()) | set(predictions.tolist()))
    accuracy = float(np.mean(labels == predictions)) if labels.size else 0.0
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    class_index = {label: index for index, label in enumerate(classes)}
    for label, prediction in zip(labels, predictions):
        confusion[class_index[int(label)], class_index[int(prediction)]] += 1
    per_class = {}
    for label in classes:
        idx = class_index[label]
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-8), 4),
        }
    return {
        "accuracy": round(accuracy, 4),
        "classes": [str(label) for label in classes],
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "n_samples": int(labels.size),
    }
