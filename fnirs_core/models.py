"""Model registry and lightweight classifiers for fNIRS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


MODEL_FAMILIES = {
    "fnirs-eegnet": "fNIRS-EEGNet",
    "cnn-lstm": "CNN-LSTM",
    "tcn": "Temporal Convolution Network",
    "graph-tcn": "Graph-TCN",
    "hybrid-3d-cnn": "Hybrid 3D CNN",
}


class ModelError(RuntimeError):
    """Raised when a model cannot be constructed or trained."""


@dataclass(slots=True)
class ModelConfig:
    model_family: str = "cnn-lstm"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 16
    max_epochs: int = 20
    seed: int = 42
    extra_params: dict[str, Any] = field(default_factory=dict)

    def normalized_family(self) -> str:
        value = self.model_family.lower().strip().replace("_", "-")
        if value not in MODEL_FAMILIES:
            raise ModelError(f"Unsupported model family: {self.model_family}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.normalized_family(),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "seed": int(self.seed),
            "extra_params": dict(self.extra_params),
        }


class PrototypeClassifier:
    """A tiny deterministic classifier used for quick local experiments.

    It is intentionally lightweight so the platform can prove the whole
    subject-wise workflow even before GPU/PyTorch dependencies are installed.
    """

    def __init__(self, *, model_family: str = "cnn-lstm", seed: int = 42) -> None:
        self.model_family = model_family
        self.seed = seed
        self.class_centroids: dict[int, np.ndarray] = {}
        self.classes_: np.ndarray = np.asarray([], dtype=int)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PrototypeClassifier":
        features = self._features(x)
        self.feature_mean = features.mean(axis=0, keepdims=True)
        self.feature_std = features.std(axis=0, keepdims=True)
        self.feature_std[self.feature_std == 0] = 1.0
        features = (features - self.feature_mean) / self.feature_std
        self.classes_ = np.unique(y.astype(int))
        if self.classes_.size < 1:
            raise ModelError("Training labels are empty.")
        self.class_centroids = {
            int(label): features[y == label].mean(axis=0)
            for label in self.classes_
        }
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None or not self.class_centroids:
            raise ModelError("Classifier has not been fitted.")
        features = self._features(x)
        features = (features - self.feature_mean) / self.feature_std
        logits = []
        for label in self.classes_:
            centroid = self.class_centroids[int(label)]
            distance = np.linalg.norm(features - centroid, axis=1)
            logits.append(-distance)
        logits_arr = np.stack(logits, axis=1)
        logits_arr -= logits_arr.max(axis=1, keepdims=True)
        exp = np.exp(logits_arr)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def explain_features(self, x: np.ndarray, y: np.ndarray | None = None) -> dict[str, Any]:
        features = np.asarray(x, dtype=np.float32)
        channel_importance = np.abs(features).mean(axis=(0, 1, 3))
        time_importance = np.abs(features).mean(axis=(0, 1, 2))
        band_importance = np.abs(features).mean(axis=(0, 2, 3))
        return {
            "channel_importance": channel_importance.tolist(),
            "time_importance": time_importance.tolist(),
            "band_importance": band_importance.tolist(),
            "method": "prototype-activation",
        }

    def _features(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 4:
            raise ModelError("Expected experiment data shaped as (epochs, bands, channels, times).")
        mean = array.mean(axis=-1)
        std = array.std(axis=-1)
        peak = np.max(array, axis=-1)
        trough = np.min(array, axis=-1)
        temporal_slope = array[..., -1] - array[..., 0]
        features = np.concatenate(
            [
                mean.reshape(array.shape[0], -1),
                std.reshape(array.shape[0], -1),
                peak.reshape(array.shape[0], -1),
                trough.reshape(array.shape[0], -1),
                temporal_slope.reshape(array.shape[0], -1),
            ],
            axis=1,
        )
        if self.model_family in {"tcn", "graph-tcn"}:
            diffs = np.diff(array, axis=-1)
            features = np.concatenate([features, diffs.mean(axis=-1).reshape(array.shape[0], -1)], axis=1)
        if self.model_family == "hybrid-3d-cnn":
            energy = (array**2).mean(axis=-1).reshape(array.shape[0], -1)
            features = np.concatenate([features, energy], axis=1)
        return features.astype(np.float32)


def create_model(config: ModelConfig | dict[str, Any] | None = None) -> PrototypeClassifier:
    model_config = config if isinstance(config, ModelConfig) else ModelConfig(**(config or {}))
    return PrototypeClassifier(model_family=model_config.normalized_family(), seed=model_config.seed)


def model_registry() -> list[dict[str, str]]:
    return [{"id": key, "name": value} for key, value in MODEL_FAMILIES.items()]

