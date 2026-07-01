"""Data loading and summarization helpers for fNIRS datasets."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_DATA_SUFFIXES = {".snirf", ".nirs", ".csv", ".mat", ".zip", ".json"}
EVENT_COLUMN_CANDIDATES = ("event", "label", "stim", "trigger", "condition", "class")
SUBJECT_COLUMN_CANDIDATES = ("subject", "subject_id", "participant", "participant_id", "sub")
TIME_COLUMN_CANDIDATES = ("time", "timestamp", "t", "seconds")
MAT_SIGNAL_CANDIDATES = (
    "raw_data",
    "dataTimeSeries",
    "data_time_series",
    "d",
    "Y",
    "signal",
    "signals",
    "x",
    "dc",
    "dod",
    "hbo",
    "hbr",
    "data",
)
MAT_SAMPLING_RATE_CANDIDATES = ("sampling_rate", "sample_rate", "fs", "srate", "sfreq")
MAT_EVENT_CANDIDATES = ("events", "event", "stim", "trigger", "triggers", "s", "labels", "label", "condition", "class")
MAT_NON_SIGNAL_NAMES = {
    "__header__",
    "__version__",
    "__globals__",
    "time",
    "timestamp",
    "t",
    "fs",
    "srate",
    "sfreq",
    "sampling_rate",
    "sample_rate",
    "events",
    "event",
    "stim",
    "trigger",
    "triggers",
    "s",
    "label",
    "labels",
    "condition",
    "class",
    "subject",
    "subject_id",
    "participant",
    "participant_id",
    "measlist",
    "ml",
    "srcpos",
    "detpos",
    "lambda",
    "wavelength",
    "probe",
    "sd",
    "aux",
}


class NIRSDataError(RuntimeError):
    """Raised when an fNIRS file cannot be parsed."""


@dataclass(slots=True)
class DatasetSample:
    signal: np.ndarray
    label: Any | None = None
    subject_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NIRSData:
    raw_data: np.ndarray
    sampling_rate: float
    channel_names: list[str]
    events: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    hbo: np.ndarray | None = None
    hbr: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id_map: dict[int, str] = field(default_factory=dict)
    subject_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.raw_data = np.asarray(self.raw_data, dtype=np.float32)
        if self.raw_data.ndim != 2:
            raise NIRSDataError("raw_data must be shaped as (n_channels, n_samples).")
        if self.sampling_rate <= 0:
            raise NIRSDataError("sampling_rate must be positive.")
        if len(self.channel_names) != self.raw_data.shape[0]:
            raise NIRSDataError("channel_names length must match n_channels.")
        self.events = _normalize_events(self.events)
        self.hbo = _optional_signal(self.hbo, self.raw_data.shape, "hbo")
        self.hbr = _optional_signal(self.hbr, self.raw_data.shape, "hbr")
        if self.subject_ids is not None:
            self.subject_ids = np.asarray(self.subject_ids)

    @property
    def n_channels(self) -> int:
        return int(self.raw_data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.raw_data.shape[1])

    @property
    def duration(self) -> float:
        return float(self.n_samples / self.sampling_rate)

    def summary(self) -> dict[str, Any]:
        label_counts = Counter(self.event_id_map.get(int(code), str(int(code))) for code in self.events[:, 2]) if self.events.size else Counter()
        return {
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "duration_seconds": round(self.duration, 4),
            "sampling_rate": float(self.sampling_rate),
            "channel_names": self.channel_names,
            "has_hbo": self.hbo is not None,
            "has_hbr": self.hbr is not None,
            "n_events": int(self.events.shape[0]),
            "event_label_distribution": dict(label_counts),
            "subject_count": int(len(set(map(str, self.subject_ids)))) if self.subject_ids is not None else 1,
            "source_format": self.metadata.get("source_format", "unknown"),
        }

    def get_epochs(
        self,
        *,
        tmin: float = 0.0,
        tmax: float = 10.0,
        baseline: tuple[float, float] | None = None,
        include_hbo_hbr: bool = True,
    ) -> dict[str, Any]:
        if tmax <= tmin:
            raise NIRSDataError("tmax must be greater than tmin.")
        if self.events.size == 0:
            raise NIRSDataError("No events are available for epoch extraction.")

        start_offset = int(round(tmin * self.sampling_rate))
        end_offset = int(round(tmax * self.sampling_rate))
        expected_length = end_offset - start_offset
        if expected_length <= 0:
            raise NIRSDataError("Epoch window must include at least one sample.")

        epochs: list[np.ndarray] = []
        hbo_epochs: list[np.ndarray] = []
        hbr_epochs: list[np.ndarray] = []
        labels: list[int] = []
        groups: list[str] = []
        kept_events: list[np.ndarray] = []

        for event_index, event in enumerate(self.events):
            onset = int(event[0])
            start = onset + start_offset
            end = onset + end_offset
            if start < 0 or end > self.n_samples:
                continue
            epoch = self.raw_data[:, start:end].copy()
            if baseline is not None:
                epoch -= _baseline_value(self.raw_data, onset, baseline, self.sampling_rate)
            epochs.append(epoch)
            labels.append(int(event[2]))
            kept_events.append(event)
            if self.subject_ids is not None and event_index < len(self.subject_ids):
                groups.append(str(self.subject_ids[event_index]))
            else:
                groups.append(str(self.metadata.get("subject_id", "subject_0")))
            if include_hbo_hbr and self.hbo is not None:
                hbo_epochs.append(self.hbo[:, start:end])
            if include_hbo_hbr and self.hbr is not None:
                hbr_epochs.append(self.hbr[:, start:end])

        if not epochs:
            raise NIRSDataError("No valid epochs could be extracted for the requested window.")

        output: dict[str, Any] = {
            "data": np.stack(epochs, axis=0),
            "labels": np.asarray(labels, dtype=int),
            "events": np.asarray(kept_events, dtype=int),
            "groups": np.asarray(groups),
            "times": np.arange(expected_length, dtype=np.float32) / self.sampling_rate + tmin,
            "sampling_rate": float(self.sampling_rate),
            "channel_names": list(self.channel_names),
        }
        if hbo_epochs:
            output["hbo"] = np.stack(hbo_epochs, axis=0)
        if hbr_epochs:
            output["hbr"] = np.stack(hbr_epochs, axis=0)
        return output


def load_fNIRS_data(file_path: str | Path) -> NIRSData:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DATA_SUFFIXES:
        raise NIRSDataError(f"Unsupported file extension: {suffix}.")
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".zip":
        return _load_zip(path)
    if suffix in {".mat", ".nirs"}:
        return _load_mat(path)
    if suffix == ".snirf":
        return _load_snirf(path)
    raise NIRSDataError(f"Unsupported file extension: {suffix}.")


def summarize_file(path: str | Path) -> dict[str, Any]:
    data = load_fNIRS_data(path)
    return data.summary()


def make_demo_nirs_data(
    *,
    n_subjects: int = 6,
    trials_per_subject: int = 6,
    n_channels: int = 16,
    n_times: int = 160,
    sampling_rate: float = 10.0,
    seed: int = 42,
) -> NIRSData:
    rng = np.random.default_rng(seed)
    total_trials = n_subjects * trials_per_subject
    rest = np.zeros((n_channels, 30), dtype=np.float32)
    trial_segments: list[np.ndarray] = []
    events: list[list[int]] = []
    labels: list[int] = []
    groups: list[str] = []
    cursor = 0
    channel_pattern = np.linspace(0.2, 1.0, n_channels, dtype=np.float32)[:, None]
    time = np.linspace(0, math.pi, n_times, dtype=np.float32)[None, :]

    for trial in range(total_trials):
        subject = trial // trials_per_subject
        label = trial % 2
        noise = rng.normal(0, 0.08, size=(n_channels, n_times)).astype(np.float32)
        response = np.sin(time) * channel_pattern * (0.35 + 0.4 * label)
        segment = response.astype(np.float32) + noise + subject * 0.01
        trial_segments.extend([rest, segment])
        cursor += rest.shape[1]
        events.append([cursor, 0, label])
        labels.append(label)
        groups.append(f"subject_{subject + 1}")
        cursor += segment.shape[1]

    raw_data = np.concatenate(trial_segments, axis=1)
    hbo = raw_data + rng.normal(0, 0.02, raw_data.shape).astype(np.float32)
    hbr = -0.35 * raw_data + rng.normal(0, 0.02, raw_data.shape).astype(np.float32)
    return NIRSData(
        raw_data=raw_data,
        sampling_rate=sampling_rate,
        channel_names=[f"Ch{i + 1:02d}" for i in range(n_channels)],
        events=np.asarray(events, dtype=int),
        hbo=hbo,
        hbr=hbr,
        event_id_map={0: "control", 1: "task"},
        subject_ids=np.asarray(groups),
        metadata={"source_format": "demo", "labels": labels},
    )


def _load_csv(path: Path) -> NIRSData:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        dialect = _sniff_csv_dialect(sample)
        reader = csv.reader(handle, dialect)
        rows = list(reader)
    if _needs_comma_fallback(rows, sample):
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            rows = list(csv.reader(handle, csv.excel))
    if not rows:
        raise NIRSDataError("CSV file is empty.")

    has_header = _rows_have_header(rows, sample)
    headers = [item.strip() for item in rows[0]] if has_header else [f"col_{i}" for i in range(len(rows[0]))]
    body = rows[1:] if has_header else rows
    columns = {name.lower(): index for index, name in enumerate(headers)}
    label_column = _first_match(columns, EVENT_COLUMN_CANDIDATES)
    subject_column = _first_match(columns, SUBJECT_COLUMN_CANDIDATES)
    time_column = _first_match(columns, TIME_COLUMN_CANDIDATES)

    valid_body = [row for row in body if any(cell.strip() for cell in row)]
    if not valid_body:
        raise NIRSDataError("CSV file does not contain data rows.")

    channel_indices = [
        index
        for index, name in enumerate(headers)
        if index not in {label_column, subject_column, time_column} and _is_numeric_column(valid_body, index)
    ]
    if not channel_indices:
        raise NIRSDataError("CSV file does not contain numeric channel columns.")

    matrix = np.asarray(
        [[_to_float(row[index]) if index < len(row) else 0.0 for row in valid_body] for index in channel_indices],
        dtype=np.float32,
    )
    channel_names = [headers[index] for index in channel_indices]
    sampling_rate = _infer_sampling_rate(valid_body, time_column)
    events, event_id_map, subject_ids = _events_from_rows(valid_body, label_column, subject_column)
    return NIRSData(
        raw_data=matrix,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        events=events,
        event_id_map=event_id_map,
        subject_ids=subject_ids,
        metadata={"source_format": "csv", "source_path": str(path), "columns": headers},
    )


def _load_json(path: Path) -> NIRSData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_data = np.asarray(payload["raw_data"], dtype=np.float32)
    if raw_data.ndim == 2 and raw_data.shape[0] > raw_data.shape[1]:
        raw_data = raw_data.T
    return NIRSData(
        raw_data=raw_data,
        sampling_rate=float(payload.get("sampling_rate", 1.0)),
        channel_names=list(payload.get("channel_names") or [f"Ch{i + 1}" for i in range(raw_data.shape[0])]),
        events=np.asarray(payload.get("events", []), dtype=int),
        hbo=np.asarray(payload["hbo"], dtype=np.float32) if payload.get("hbo") is not None else None,
        hbr=np.asarray(payload["hbr"], dtype=np.float32) if payload.get("hbr") is not None else None,
        event_id_map={int(key): str(value) for key, value in payload.get("event_id_map", {}).items()},
        subject_ids=np.asarray(payload.get("subject_ids")) if payload.get("subject_ids") is not None else None,
        metadata={"source_format": "json", "source_path": str(path)},
    )


def _load_zip(path: Path) -> NIRSData:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path) as archive:
            _safe_extract_zip(archive, Path(tmpdir))
        candidates = [
            candidate
            for candidate in Path(tmpdir).rglob("*")
            if (
                candidate.is_file()
                and candidate.suffix.lower() in SUPPORTED_DATA_SUFFIXES - {".zip"}
                and not _is_hidden_archive_artifact(candidate)
            )
        ]
        if not candidates:
            raise NIRSDataError("Zip archive does not contain a supported fNIRS data file.")
        errors: list[str] = []
        for candidate in sorted(candidates, key=lambda item: (item.suffix.lower() not in {".snirf", ".nirs", ".mat"}, str(item))):
            try:
                return load_fNIRS_data(candidate)
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
        detail = "; ".join(errors[:5])
        raise NIRSDataError(f"Zip archive does not contain a parseable fNIRS data file. {detail}")


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise NIRSDataError("Zip archive contains an unsafe path.") from exc
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            output.write(source.read())


def _is_hidden_archive_artifact(path: Path) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in path.parts)


def _load_mat(path: Path) -> NIRSData:
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover - optional dependency
        raise NIRSDataError("MAT/NIRS parsing requires scipy.") from exc
    try:
        payload = loadmat(path, simplify_cells=True)
    except TypeError:  # pragma: no cover - older scipy fallback
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)

    signal_source = _find_mat_signal(payload)
    if signal_source is None:
        raise NIRSDataError("MAT file does not contain a numeric signal array.")
    signal = _normalize_signal_matrix(signal_source)
    sampling_rate = _mat_sampling_rate(payload)
    events = _mat_events(payload)
    channel_names = [f"Ch{i + 1}" for i in range(signal.shape[0])]
    return NIRSData(
        raw_data=signal,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        events=events,
        metadata={"source_format": path.suffix.lower().lstrip("."), "source_path": str(path)},
    )


def _find_mat_signal(payload: dict[str, Any]) -> np.ndarray | None:
    for name in MAT_SIGNAL_CANDIDATES:
        if name in payload:
            candidate = _best_numeric_array(payload[name], preferred_name=name)
            if candidate is not None:
                return candidate
    return _best_numeric_array(payload)


def _best_numeric_array(value: Any, *, preferred_name: str | None = None) -> np.ndarray | None:
    candidates = _numeric_arrays(value, preferred_name=preferred_name)
    if not candidates:
        return None
    candidates.sort(key=_signal_score, reverse=True)
    return candidates[0][2]


def _numeric_arrays(value: Any, *, preferred_name: str | None = None) -> list[tuple[str, int, np.ndarray]]:
    arrays: list[tuple[str, int, np.ndarray]] = []

    def visit(item: Any, name: str, depth: int) -> None:
        if depth > 6 or item is None:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_name = str(key)
                if child_name in {"__header__", "__version__", "__globals__"}:
                    continue
                visit(child, child_name, depth + 1)
            return
        if hasattr(item, "__dict__") and item.__class__.__module__.startswith("scipy"):
            for key, child in vars(item).items():
                if key.startswith("_"):
                    continue
                visit(child, key, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            converted = _coerce_numeric_array(item)
            if converted is not None:
                arrays.append((name, depth, converted))
                return
            for index, child in enumerate(item[:20]):
                visit(child, f"{name}_{index}", depth + 1)
            return
        if not isinstance(item, np.ndarray):
            return
        converted = _coerce_numeric_array(item)
        if converted is not None:
            arrays.append((name, depth, converted))
            return
        if item.dtype == object:
            for child in item.ravel()[:50]:
                visit(child, name, depth + 1)

    visit(value, preferred_name or "", 0)
    return arrays


def _coerce_numeric_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.size < 2:
        return None
    if array.dtype == object:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).any():
        return None
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _signal_score(candidate: tuple[str, int, np.ndarray]) -> tuple[int, int, int, int]:
    name, depth, array = candidate
    normalized_name = name.lower()
    positive_name = int(normalized_name in MAT_SIGNAL_CANDIDATES)
    negative_name = int(normalized_name in MAT_NON_SIGNAL_NAMES)
    shape = array.shape
    signal_like = int(array.ndim >= 2 and min(shape[:2]) >= 2 and max(shape[:2]) >= 10)
    size_score = min(int(array.size), 1_000_000)
    return (signal_like, positive_name - negative_name, size_score, -depth)


def _normalize_signal_matrix(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.squeeze(array)
    if array.ndim == 0:
        raise NIRSDataError("MAT signal array is scalar.")
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] > array.shape[1]:
        array = array.T
    if array.shape[0] < 1 or array.shape[1] < 2:
        raise NIRSDataError("MAT signal array must contain at least one channel and two samples.")
    return array.astype(np.float32, copy=False)


def _mat_sampling_rate(payload: dict[str, Any]) -> float:
    value = _first_scalar(payload, MAT_SAMPLING_RATE_CANDIDATES)
    if value is None:
        time_array = _first_numeric_array(payload, ("t", "time", "timestamp"))
        if time_array is not None:
            diffs = np.diff(np.asarray(time_array, dtype=np.float32).ravel())
            diffs = diffs[diffs > 0]
            if diffs.size:
                return float(1.0 / np.median(diffs))
        return 1.0
    try:
        sampling_rate = float(np.asarray(value).ravel()[0])
    except Exception:
        return 1.0
    return sampling_rate if sampling_rate > 0 else 1.0


def _first_scalar(payload: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in payload:
            candidate = _scalar_or_array(payload[name])
            if candidate is not None:
                return candidate
    for value in payload.values():
        if isinstance(value, dict):
            candidate = _first_scalar(value, names)
            if candidate is not None:
                return candidate
    return None


def _first_numeric_array(payload: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in payload:
            candidate = _best_numeric_array(payload[name], preferred_name=name)
            if candidate is not None:
                return candidate
    for value in payload.values():
        if isinstance(value, dict):
            candidate = _first_numeric_array(value, names)
            if candidate is not None:
                return candidate
    return None


def _scalar_or_array(value: Any) -> Any | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if np.issubdtype(arr.dtype, np.number) and arr.size >= 1:
        return arr
    return None


def _load_snirf(path: Path) -> NIRSData:
    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency
        raise NIRSDataError("SNIRF parsing requires mne.") from exc
    raw = mne.io.read_raw_snirf(str(path), preload=True, verbose=False)
    data = raw.get_data()
    events = np.empty((0, 3), dtype=int)
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        event_id_map = {int(code): label for label, code in event_id.items()}
    except Exception:
        event_id_map = {}
    return NIRSData(
        raw_data=np.asarray(data, dtype=np.float32),
        sampling_rate=float(raw.info["sfreq"]),
        channel_names=list(raw.ch_names),
        events=events,
        event_id_map=event_id_map,
        metadata={"source_format": "snirf", "source_path": str(path)},
    )


def _normalize_events(events: Any) -> np.ndarray:
    array = np.asarray(events)
    if array.size == 0:
        return np.empty((0, 3), dtype=int)
    array = np.asarray(events, dtype=int)
    if array.ndim != 2 or array.shape[1] != 3:
        raise NIRSDataError("events must be shaped as (n_events, 3).")
    return array


def _optional_signal(signal: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    if signal is None:
        return None
    array = np.asarray(signal, dtype=np.float32)
    if array.shape != shape:
        raise NIRSDataError(f"{name} must match raw_data shape.")
    return array


def _baseline_value(data: np.ndarray, onset: int, baseline: tuple[float, float], sampling_rate: float) -> np.ndarray:
    start = onset + int(round(baseline[0] * sampling_rate))
    end = onset + int(round(baseline[1] * sampling_rate))
    if start < 0 or end <= start or end > data.shape[1]:
        return np.zeros((data.shape[0], 1), dtype=np.float32)
    return data[:, start:end].mean(axis=1, keepdims=True)


def _first_match(columns: dict[str, int], candidates: tuple[str, ...]) -> int | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _sniff_csv_dialect(sample: str) -> csv.Dialect:
    if not sample.strip():
        return csv.excel
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return csv.excel


def _sniffer_has_header(sample: str) -> bool:
    if not sample.strip():
        return False
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return False


def _rows_have_header(rows: list[list[str]], sample: str) -> bool:
    if not rows:
        return False
    first_row = [cell.strip().lower() for cell in rows[0]]
    known_columns = set(EVENT_COLUMN_CANDIDATES + SUBJECT_COLUMN_CANDIDATES + TIME_COLUMN_CANDIDATES)
    if any(cell in known_columns or cell.startswith(("ch", "channel")) for cell in first_row):
        return True
    if _sniffer_has_header(sample):
        return True
    if len(rows) < 2:
        return False
    first_numeric = sum(_is_float(cell) for cell in rows[0] if cell.strip())
    second_numeric = sum(_is_float(cell) for cell in rows[1] if cell.strip())
    return second_numeric > first_numeric


def _needs_comma_fallback(rows: list[list[str]], sample: str) -> bool:
    return bool(sample.strip() and "," in sample and rows and max(len(row) for row in rows) <= 1)


def _is_numeric_column(rows: list[list[str]], index: int) -> bool:
    checked = 0
    numeric = 0
    for row in rows[:50]:
        if index >= len(row):
            continue
        checked += 1
        try:
            float(row[index])
            numeric += 1
        except Exception:
            pass
    return checked > 0 and numeric >= max(1, int(checked * 0.8))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _infer_sampling_rate(rows: list[list[str]], time_column: int | None) -> float:
    if time_column is None:
        return 1.0
    times = [_to_float(row[time_column]) for row in rows if time_column < len(row)]
    diffs = np.diff(np.asarray(times, dtype=np.float32))
    diffs = diffs[diffs > 0]
    if not diffs.size:
        return 1.0
    return float(1.0 / np.median(diffs))


def _events_from_rows(
    rows: list[list[str]],
    label_column: int | None,
    subject_column: int | None,
) -> tuple[np.ndarray, dict[int, str], np.ndarray | None]:
    if label_column is None:
        return np.empty((0, 3), dtype=int), {}, None

    labels: list[str] = []
    subject_ids: list[str] = []
    for row in rows:
        labels.append(row[label_column].strip() if label_column < len(row) and row[label_column].strip() else "0")
        if subject_column is not None and subject_column < len(row):
            subject_ids.append(row[subject_column].strip() or "subject_0")

    label_to_code = {label: index for index, label in enumerate(sorted(set(labels)))}
    events = np.asarray([[index, 0, label_to_code[label]] for index, label in enumerate(labels)], dtype=int)
    event_id_map = {code: label for label, code in label_to_code.items()}
    subject_array = np.asarray(subject_ids) if subject_ids else None
    return events, event_id_map, subject_array


def _first_array(payload: dict[str, Any], names: tuple[str, ...], default: Any | None = None) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    if default is not None:
        return default
    for value in payload.values():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.size > 0:
            return value
    raise NIRSDataError("MAT file does not contain a signal array.")


def _mat_events(payload: dict[str, Any]) -> np.ndarray:
    for key in MAT_EVENT_CANDIDATES:
        if key not in payload:
            continue
        events = _events_from_mat_value(payload[key])
        if events.size:
            return events
    for value in payload.values():
        if isinstance(value, dict):
            events = _mat_events(value)
            if events.size:
                return events
    return np.empty((0, 3), dtype=int)


def _events_from_mat_value(value: Any) -> np.ndarray:
    arrays = _numeric_arrays(value)
    arrays.sort(key=_signal_score, reverse=True)
    for _, _, array in arrays:
        events = _events_from_numeric_array(array)
        if events.size:
            return events
    return np.empty((0, 3), dtype=int)


def _events_from_numeric_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return np.empty((0, 3), dtype=int)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(int)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T.astype(int)
    if arr.ndim == 2 and min(arr.shape) >= 2:
        sample_axis = 0 if arr.shape[0] >= arr.shape[1] else 1
        matrix = arr if sample_axis == 0 else arr.T
        rows: list[list[int]] = []
        for index, row in enumerate(matrix):
            active = np.flatnonzero(row)
            if active.size:
                rows.append([index, 0, int(active[0] + 1)])
        return np.asarray(rows, dtype=int) if rows else np.empty((0, 3), dtype=int)
    flat = arr.ravel()
    nonzero = np.flatnonzero(flat)
    if nonzero.size:
        return np.asarray([[int(index), 0, int(flat[index])] for index in nonzero], dtype=int)
    return np.empty((0, 3), dtype=int)
