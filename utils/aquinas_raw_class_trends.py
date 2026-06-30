from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from .aquinas_common import (
    BANDS,
    BAND_COLS,
    EPS,
    FS,
    TABLE_DIR,
    band_energy_and_ratio_from_psd,
    parse_alias,
    read_event_hdf5,
    valid_time_bounds,
    welch_psd,
)

RAW_CLASS_FEATURE_PATH = TABLE_DIR / "event_sensor_raw_class_energy_v1.csv"
RAW_CLASS_SIGNAL_FEATURE_PATH = TABLE_DIR / "event_sensor_dominant_class_signal_processing_metrics_v1.csv"


def _signal_metrics(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "signal_n_samples": 0,
            "response_rms": 0.0,
            "response_mean_abs": 0.0,
            "response_median_abs": 0.0,
            "response_p95_abs": 0.0,
            "response_iqr": 0.0,
            "response_robust_range": 0.0,
            "response_peak_to_peak": 0.0,
        }

    x = x - np.nanmedian(x)
    q05, q25, q75, q95 = np.nanpercentile(x, [5, 25, 75, 95])
    return {
        "signal_n_samples": int(x.size),
        "response_rms": float(np.sqrt(np.nanmean(x**2))),
        "response_mean_abs": float(np.nanmean(np.abs(x))),
        "response_median_abs": float(np.nanmedian(np.abs(x))),
        "response_p95_abs": float(np.nanpercentile(np.abs(x), 95)),
        "response_iqr": float(q75 - q25),
        "response_robust_range": float(q95 - q05),
        "response_peak_to_peak": float(np.nanmax(x) - np.nanmin(x)),
    }


def _zero_dominant_band_metrics() -> dict[str, float]:
    return {
        "dominant_band_signal_rms": 0.0,
        "dominant_band_hilbert_mean": 0.0,
        "dominant_band_hilbert_rms": 0.0,
        "dominant_band_hilbert_max": 0.0,
        "dominant_band_teager_mean_abs": 0.0,
    }


def _fft_band_limited_signal(x: np.ndarray, low: float, high: float, fs: float = FS) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.zeros_like(x)
    x = x - np.nanmedian(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spectrum = np.fft.rfft(x)
    keep = (freqs >= low) & (freqs < high)
    if not keep.any():
        return np.zeros_like(x)
    filtered = np.zeros_like(spectrum)
    filtered[keep] = spectrum[keep]
    return np.fft.irfft(filtered, n=x.size)


def _dominant_band_signal_metrics(x_centered: np.ndarray, dominant_class: int) -> dict[str, float]:
    if dominant_class < 1 or dominant_class > len(BANDS):
        return _zero_dominant_band_metrics()

    _, low, high, _ = BANDS[dominant_class - 1]
    y = _fft_band_limited_signal(x_centered, low, high)
    if y.size < 8 or not np.isfinite(y).any():
        return _zero_dominant_band_metrics()

    envelope = np.abs(hilbert(y))
    if y.size >= 3:
        teager = y[1:-1] ** 2 - y[:-2] * y[2:]
        teager_mean_abs = float(np.nanmean(np.abs(teager)))
    else:
        teager_mean_abs = 0.0

    return {
        "dominant_band_signal_rms": float(np.sqrt(np.nanmean(y**2))),
        "dominant_band_hilbert_mean": float(np.nanmean(envelope)),
        "dominant_band_hilbert_rms": float(np.sqrt(np.nanmean(envelope**2))),
        "dominant_band_hilbert_max": float(np.nanmax(envelope)),
        "dominant_band_teager_mean_abs": teager_mean_abs,
    }


def _event_raw_class_rows(event_row: dict) -> list[dict]:
    path = Path(str(event_row["path"]))
    deck = str(event_row["deck"]).upper()
    values, mask, aliases, time_grid = read_event_hdf5(path)

    start_s = float(event_row.get("active_start_s_v2", np.nan))
    end_s = float(event_row.get("active_end_s_v2", np.nan))
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        start_s, end_s = valid_time_bounds(mask, time_grid)

    active = (time_grid >= start_s) & (time_grid <= end_s)
    rows: list[dict] = []

    for sensor_idx, alias in enumerate(aliases):
        meta = parse_alias(alias)
        valid = mask[sensor_idx].astype(bool) & active
        present = bool(valid.sum() >= 32)
        energies = np.zeros(len(BAND_COLS), dtype=float)
        dominant_class = 0
        dominant_energy = 0.0
        row_energy_sum = 0.0
        metrics = _signal_metrics(np.array([], dtype=float))
        dominant_band_metrics = _zero_dominant_band_metrics()

        if present:
            x = values[sensor_idx, valid].astype(float)
            x = x[np.isfinite(x)]
            if x.size >= 8:
                metrics = _signal_metrics(x)
                x = x - np.nanmedian(x)
                freq, psd = welch_psd(x)
                if len(freq):
                    energies, _ = band_energy_and_ratio_from_psd(freq, psd)
                    row_energy_sum = float(np.nansum(energies))
                    if row_energy_sum > EPS:
                        dominant_idx = int(np.nanargmax(energies))
                        dominant_class = dominant_idx + 1
                        dominant_energy = float(energies[dominant_idx])
                        dominant_band_metrics = _dominant_band_signal_metrics(x, dominant_class)

        out = {
            "path": str(path),
            "set_name": event_row["set_name"],
            "deck": deck,
            "event_id": int(event_row["event_id"]),
            "campaign_month": event_row["campaign_month"],
            "part_index": int(event_row["part_index"]),
            "sensor_id": f"{deck}_{alias}",
            "sensor_alias": alias,
            "present": present,
            "quantity": meta["quantity"],
            "span": meta["span"],
            "side": meta["side"],
            "location": meta["location"],
            "axis": meta["axis"],
            "active_start_s": start_s,
            "active_end_s": end_s,
            "active_duration_s": float(end_s - start_s),
            "row_energy_sum": row_energy_sum,
            "dominant_class": dominant_class,
            "dominant_energy": dominant_energy,
            **metrics,
            **dominant_band_metrics,
        }
        for idx, value in enumerate(energies, start=1):
            out[f"class_{idx}_energy"] = float(value)
            out[f"dominant_class_{idx}_energy"] = float(value) if dominant_class == idx else 0.0
        rows.append(out)
    return rows


def _load_event_rows() -> pd.DataFrame:
    timing_path = TABLE_DIR / "event_hilbert_timing_diagnostics_v2.csv"
    if not timing_path.exists():
        raise FileNotFoundError(f"Timing table was not found: {timing_path}")

    events = pd.read_csv(timing_path, low_memory=False)
    events = events[events["path"].map(lambda p: Path(str(p)).exists())].copy()
    events["deck"] = events["deck"].astype(str).str.upper()
    events["part_index"] = events["part_index"].astype(int)
    events["event_id"] = events["event_id"].astype(int)
    return events.sort_values(["campaign_month", "deck", "set_name", "part_index", "event_id"]).reset_index(drop=True)


def build_raw_class_feature_table(
    output_path: str | Path = RAW_CLASS_FEATURE_PATH,
    *,
    force: bool = False,
    max_workers: int | None = None,
    max_events: int | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    if output_path.exists() and not force and max_events is None:
        return pd.read_csv(output_path, low_memory=False)

    events = _load_event_rows()
    if max_events is not None:
        events = events.head(int(max_events)).copy()

    records = events.to_dict("records")
    if max_workers is None:
        max_workers = max(1, min(8, os.cpu_count() or 1))

    rows: list[dict] = []
    if max_workers <= 1:
        for record in records:
            rows.extend(_event_raw_class_rows(record))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_event_raw_class_rows, record) for record in records]
            for future in as_completed(futures):
                rows.extend(future.result())

    out = pd.DataFrame(rows)
    sort_cols = ["campaign_month", "deck", "set_name", "part_index", "event_id", "sensor_alias"]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    if max_events is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return out


def load_or_build_raw_class_feature_table(
    output_path: str | Path = RAW_CLASS_FEATURE_PATH,
    *,
    max_workers: int | None = None,
) -> pd.DataFrame:
    return build_raw_class_feature_table(output_path, force=False, max_workers=max_workers)


def build_dominant_class_signal_metric_table(
    output_path: str | Path = RAW_CLASS_SIGNAL_FEATURE_PATH,
    *,
    force: bool = False,
    max_workers: int | None = None,
    max_events: int | None = None,
) -> pd.DataFrame:
    return build_raw_class_feature_table(
        output_path=output_path,
        force=force,
        max_workers=max_workers,
        max_events=max_events,
    )


def load_or_build_dominant_class_signal_metric_table(
    output_path: str | Path = RAW_CLASS_SIGNAL_FEATURE_PATH,
    *,
    max_workers: int | None = None,
) -> pd.DataFrame:
    return build_dominant_class_signal_metric_table(output_path, force=False, max_workers=max_workers)
