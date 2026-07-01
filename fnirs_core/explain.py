"""Explainability helpers for fNIRS experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_fNIRS_data, make_demo_nirs_data
from .models import ModelConfig, create_model
from .preprocessing import PreprocessConfig, PreprocessingPipeline


@dataclass(slots=True)
class ExplanationResult:
    experiment_id: str
    method: str
    channel_importance: list[float]
    time_importance: list[float]
    band_importance: list[float]
    top_channels: list[dict[str, Any]]
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "channel_importance": self.channel_importance,
            "time_importance": self.time_importance,
            "band_importance": self.band_importance,
            "top_channels": self.top_channels,
            "output_path": self.output_path,
        }


def explain_experiment(experiment_id: str, experiment_config: dict[str, Any], output_dir: str | Path) -> ExplanationResult:
    dataset_path = experiment_config.get("dataset_path")
    data = load_fNIRS_data(dataset_path) if dataset_path else make_demo_nirs_data(seed=int(experiment_config.get("seed", 42)))
    preprocessing = PreprocessingPipeline(PreprocessConfig(**experiment_config.get("preprocessing", {}))).run(data)
    x = preprocessing.epochs
    y = preprocessing.labels.astype(int)
    model = create_model(ModelConfig(**experiment_config.get("model", {}), seed=int(experiment_config.get("seed", 42))))
    model.fit(x, y)
    raw = model.explain_features(x, y)
    channel_importance = np.asarray(raw["channel_importance"], dtype=float)
    if channel_importance.size:
        denom = max(float(channel_importance.max()), 1e-8)
        channel_importance = channel_importance / denom
    top_indices = np.argsort(channel_importance)[::-1][:8] if channel_importance.size else np.asarray([], dtype=int)
    top_channels = [
        {
            "channel": preprocessing.channel_names[int(index)] if int(index) < len(preprocessing.channel_names) else f"Ch{int(index) + 1}",
            "importance": round(float(channel_importance[int(index)]), 4),
        }
        for index in top_indices
    ]
    output_path = Path(output_dir) / "explanation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = ExplanationResult(
        experiment_id=experiment_id,
        method=raw["method"],
        channel_importance=[round(float(value), 6) for value in channel_importance.tolist()],
        time_importance=[round(float(value), 6) for value in raw["time_importance"]],
        band_importance=[round(float(value), 6) for value in raw["band_importance"]],
        top_channels=top_channels,
        output_path=str(output_path),
    )
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result

