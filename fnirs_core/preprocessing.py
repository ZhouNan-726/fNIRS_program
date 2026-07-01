"""NumPy-based fNIRS preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .data import NIRSData, NIRSDataError


@dataclass(slots=True)
class PreprocessConfig:
    apply_optical_density: bool = True
    apply_beer_lambert: bool = True
    apply_tddr: bool = True
    bandpass_low: float | None = 0.01
    bandpass_high: float | None = 0.2
    baseline_start: float | None = -2.0
    baseline_end: float | None = 0.0
    epoch_start: float = -2.0
    epoch_end: float = 10.0
    include_hbo_hbr: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreprocessingResult:
    epochs: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    band_names: list[str]
    summary: dict[str, Any]
    config: dict[str, Any]


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot be completed."""


def recommend_preprocessing(data_stats: dict[str, Any]) -> dict[str, Any]:
    sampling_rate = float(data_stats.get("sampling_rate") or 10.0)
    high = min(0.2, sampling_rate * 0.45)
    return PreprocessConfig(
        apply_optical_density=not data_stats.get("has_hbo"),
        apply_beer_lambert=not data_stats.get("has_hbo"),
        apply_tddr=True,
        bandpass_low=0.01 if sampling_rate >= 1 else None,
        bandpass_high=high if high > 0.02 else None,
        include_hbo_hbr=bool(data_stats.get("has_hbo") or data_stats.get("has_hbr")),
    ).to_dict()


class PreprocessingPipeline:
    def __init__(self, config: PreprocessConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, PreprocessConfig):
            self.config = config
        else:
            self.config = PreprocessConfig(**(config or {}))

    def run(self, data: NIRSData) -> PreprocessingResult:
        signal = data.raw_data.astype(np.float32)
        if self.config.apply_optical_density:
            signal = optical_density(signal)
        if self.config.apply_beer_lambert:
            signal = beer_lambert(signal)
        if self.config.apply_tddr:
            signal = tddr(signal)
        if self.config.bandpass_low is not None or self.config.bandpass_high is not None:
            signal = bandpass_filter(signal, data.sampling_rate, self.config.bandpass_low, self.config.bandpass_high)

        working = NIRSData(
            raw_data=signal,
            sampling_rate=data.sampling_rate,
            channel_names=data.channel_names,
            events=data.events,
            hbo=data.hbo,
            hbr=data.hbr,
            metadata=data.metadata,
            event_id_map=data.event_id_map,
            subject_ids=data.subject_ids,
        )
        if working.events.size == 0:
            return _fallback_whole_recording(working, self.config)

        baseline = None
        if self.config.baseline_start is not None and self.config.baseline_end is not None:
            baseline = (float(self.config.baseline_start), float(self.config.baseline_end))
        epochs = working.get_epochs(
            tmin=self.config.epoch_start,
            tmax=self.config.epoch_end,
            baseline=baseline,
            include_hbo_hbr=self.config.include_hbo_hbr,
        )
        stacked, band_names = _stack_bands(epochs)
        return PreprocessingResult(
            epochs=stacked.astype(np.float32),
            labels=np.asarray(epochs["labels"], dtype=int),
            groups=np.asarray(epochs["groups"]),
            times=np.asarray(epochs["times"], dtype=np.float32),
            channel_names=list(epochs["channel_names"]),
            band_names=band_names,
            summary={
                "n_epochs": int(stacked.shape[0]),
                "n_bands": int(stacked.shape[1]),
                "n_channels": int(stacked.shape[2]),
                "n_times": int(stacked.shape[3]),
                "label_distribution": _label_distribution(epochs["labels"]),
                "subject_count": int(len(set(map(str, epochs["groups"])))),
            },
            config=self.config.to_dict(),
        )


def build_preprocessing_pipeline(config: dict[str, Any] | None = None) -> PreprocessingPipeline:
    return PreprocessingPipeline(config)


def optical_density(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    baseline = np.maximum(data.mean(axis=1, keepdims=True), 1e-6)
    clipped = np.maximum(data, 1e-6)
    return -np.log(clipped / baseline)


def beer_lambert(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    scale = np.std(data, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return data / scale


def tddr(data: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32).copy()
    derivative = np.diff(data, axis=1, prepend=data[:, :1])
    mad = np.median(np.abs(derivative - np.median(derivative, axis=1, keepdims=True)), axis=1, keepdims=True)
    mad[mad == 0] = 1.0
    z = np.abs(derivative) / mad
    spikes = z > threshold
    if spikes.any():
        derivative[spikes] = np.median(derivative, axis=1, keepdims=True).repeat(derivative.shape[1], axis=1)[spikes]
        data = np.cumsum(derivative, axis=1)
    return data.astype(np.float32)


def bandpass_filter(data: np.ndarray, sampling_rate: float, low: float | None, high: float | None) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    try:
        from scipy.signal import butter, filtfilt

        nyquist = sampling_rate / 2.0
        if low is not None and high is not None:
            btype = "bandpass"
            cutoff = [max(low / nyquist, 1e-5), min(high / nyquist, 0.999)]
        elif low is not None:
            btype = "highpass"
            cutoff = max(low / nyquist, 1e-5)
        elif high is not None:
            btype = "lowpass"
            cutoff = min(high / nyquist, 0.999)
        else:
            return data
        b, a = butter(3, cutoff, btype=btype)
        return filtfilt(b, a, data, axis=1).astype(np.float32)
    except Exception:
        return _fft_bandpass(data, sampling_rate, low, high)


def _fft_bandpass(data: np.ndarray, sampling_rate: float, low: float | None, high: float | None) -> np.ndarray:
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sampling_rate)
    spectrum = np.fft.rfft(data, axis=1)
    mask = np.ones(freqs.shape, dtype=bool)
    if low is not None:
        mask &= freqs >= low
    if high is not None:
        mask &= freqs <= high
    spectrum *= mask[None, :]
    return np.fft.irfft(spectrum, n=data.shape[1], axis=1).astype(np.float32)


def _stack_bands(epoch_payload: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    bands: list[np.ndarray] = []
    names: list[str] = []
    if "hbo" in epoch_payload:
        bands.append(np.asarray(epoch_payload["hbo"], dtype=np.float32))
        names.append("HbO")
    if "hbr" in epoch_payload:
        bands.append(np.asarray(epoch_payload["hbr"], dtype=np.float32))
        names.append("HbR")
    if not bands:
        bands.append(np.asarray(epoch_payload["data"], dtype=np.float32))
        names.append("raw")
    stacked = np.stack(bands, axis=1)
    return stacked, names


def _fallback_whole_recording(data: NIRSData, config: PreprocessConfig) -> PreprocessingResult:
    window = int(min(max(data.sampling_rate * 10, 20), data.n_samples))
    if window <= 0:
        raise PreprocessingError("Recording has no samples.")
    step = max(window // 2, 1)
    epochs = []
    labels = []
    groups = []
    for start in range(0, max(data.n_samples - window + 1, 1), step):
        end = min(start + window, data.n_samples)
        segment = data.raw_data[:, start:end]
        if segment.shape[1] < window:
            pad = np.zeros((data.n_channels, window - segment.shape[1]), dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=1)
        epochs.append(segment)
        labels.append(len(labels) % 2)
        groups.append(str(data.metadata.get("subject_id", "subject_0")))
    if len(epochs) == 1:
        epochs.append(epochs[0].copy())
        labels.append(1)
        groups.append(groups[0])
    stacked = np.asarray(epochs, dtype=np.float32)[:, None, :, :]
    times = np.arange(window, dtype=np.float32) / data.sampling_rate
    return PreprocessingResult(
        epochs=stacked,
        labels=np.asarray(labels, dtype=int),
        groups=np.asarray(groups),
        times=times,
        channel_names=data.channel_names,
        band_names=["raw"],
        summary={
            "n_epochs": int(stacked.shape[0]),
            "n_bands": int(stacked.shape[1]),
            "n_channels": int(stacked.shape[2]),
            "n_times": int(stacked.shape[3]),
            "label_distribution": _label_distribution(labels),
            "subject_count": int(len(set(groups))),
            "warning": (
                "No events found; generated sliding windows with alternating placeholder labels. "
                "Use only to verify the local pipeline, not for scientific conclusions."
            ),
        },
        config=config.to_dict(),
    )


def _label_distribution(labels: Any) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}
