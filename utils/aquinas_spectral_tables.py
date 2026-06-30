from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

from .aquinas_common import (
    BAND_COLS,
    EPS,
    HDF5_ROOT,
    ROOT,
    TABLE_DIR,
    band_energy_and_ratio_from_psd,
    parse_alias,
    read_event_hdf5,
    robust_sensor_band_scale,
    valid_time_bounds,
    welch_psd,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):
        return iterable if iterable is not None else []


HILBERT_FEATURE_CACHE = ROOT / "KYM" / "outputs" / "hilbert_aligned_reference_free_health" / "event_features_hilbert_aligned.csv"

SCORE_PATH = TABLE_DIR / "event_spectral_health_scores_v2.csv"
TIMING_PATH = TABLE_DIR / "event_hilbert_timing_diagnostics_v2.csv"
FEATURE_PATH = TABLE_DIR / "event_sensor_spectral_features_v2.csv"
REFERENCE_PATH = TABLE_DIR / "sensor_spectral_reference_fingerprint_v2.csv"
SCALE_PATH = TABLE_DIR / "sensor_band_robust_scale_image_score_v1.csv"


def _campaign_month(set_name: str) -> str:
    match = re.search(r"(\d{4})_(\d{2})", str(set_name))
    if not match:
        return ""
    return f"{match.group(1)}-{match.group(2)}"


def _load_hilbert_events() -> pd.DataFrame:
    if not HILBERT_FEATURE_CACHE.exists():
        raise FileNotFoundError(
            f"Required Hilbert event cache was not found: {HILBERT_FEATURE_CACHE}"
        )

    usecols = [
        "path",
        "set_name",
        "deck",
        "part_index",
        "event_id",
        "active_start_s",
        "active_end_s",
        "active_duration_s",
        "timing_ok",
        "health_event",
        "confidence_event",
        "error",
    ]
    events = pd.read_csv(HILBERT_FEATURE_CACHE, usecols=usecols, low_memory=False)
    events = events[events["error"].fillna("").eq("")].copy()
    events = events[events["path"].map(lambda p: Path(str(p)).exists())].copy()
    events["campaign_month"] = events["set_name"].map(_campaign_month)
    events["deck"] = events["deck"].astype(str).str.upper()
    events["part_index"] = events["part_index"].astype(int)
    events["event_id"] = events["event_id"].astype(int)
    return events.reset_index(drop=True)


def _write_score_and_timing_tables(events: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    score = events[
        [
            "path",
            "set_name",
            "deck",
            "event_id",
            "campaign_month",
            "part_index",
            "health_event",
            "confidence_event",
        ]
    ].copy()
    score["event_health_score_v2"] = score["health_event"].astype(float)
    score["event_anomaly_score_v2"] = 100.0 - score["event_health_score_v2"]
    score["event_confidence_score_v2"] = 100.0 * score["confidence_event"].astype(float)
    score = score.drop(columns=["health_event", "confidence_event"])
    score.to_csv(SCORE_PATH, index=False)

    timing = events[
        [
            "path",
            "set_name",
            "deck",
            "event_id",
            "campaign_month",
            "part_index",
            "active_start_s",
            "active_end_s",
            "active_duration_s",
            "timing_ok",
        ]
    ].copy()
    timing = timing.rename(
        columns={
            "active_start_s": "active_start_s_v2",
            "active_end_s": "active_end_s_v2",
            "active_duration_s": "active_duration_s_v2",
        }
    )
    timing.to_csv(TIMING_PATH, index=False)


def _event_sensor_feature_rows(event_row: dict) -> list[dict]:
    path = Path(str(event_row["path"]))
    deck = str(event_row["deck"]).upper()
    values, mask, aliases, time_grid = read_event_hdf5(path)

    start_s = float(event_row.get("active_start_s", np.nan))
    end_s = float(event_row.get("active_end_s", np.nan))
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        start_s, end_s = valid_time_bounds(mask, time_grid)
    active = (time_grid >= start_s) & (time_grid <= end_s)

    rows = []
    for sensor_idx, alias in enumerate(aliases):
        meta = parse_alias(alias)
        valid = mask[sensor_idx].astype(bool) & active
        present = bool(valid.sum() >= 32)
        ratio = np.zeros(len(BAND_COLS), dtype=float)
        if present:
            x = values[sensor_idx, valid].astype(float)
            x = x[np.isfinite(x)]
            if x.size >= 8:
                x = x - np.nanmedian(x)
                freq, psd = welch_psd(x)
                if len(freq):
                    _, ratio = band_energy_and_ratio_from_psd(freq, psd)

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
        }
        for col, value in zip(BAND_COLS, ratio):
            out[col] = float(value)
        rows.append(out)
    return rows


def _build_event_sensor_features(events: pd.DataFrame, num_workers: int, max_events: int | None = None) -> pd.DataFrame:
    rows = events.sort_values(["campaign_month", "deck", "part_index", "event_id"]).copy()
    if max_events is not None:
        rows = rows.head(int(max_events)).copy()

    records = rows.to_dict("records")
    all_rows: list[dict] = []
    if num_workers <= 1:
        iterator = tqdm(records, total=len(records), desc="Building Welch PSD feature table", unit="event")
        for row in iterator:
            all_rows.extend(_event_sensor_feature_rows(row))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_event_sensor_feature_rows, row) for row in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building Welch PSD feature table", unit="event"):
                all_rows.extend(future.result())

    features = pd.DataFrame(all_rows)
    features = features.sort_values(["campaign_month", "deck", "part_index", "event_id", "sensor_id"]).reset_index(drop=True)
    return features


def _build_reference_table(features: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["sensor_id", "deck", "sensor_alias", "quantity", "span", "side", "location", "axis"]
    meta = features[meta_cols].drop_duplicates("sensor_id").copy()
    ref_values = (
        features[features["present"].astype(bool)]
        .groupby("sensor_id", as_index=False)[BAND_COLS]
        .median()
    )
    reference = meta.merge(ref_values, on="sensor_id", how="left")
    for col in BAND_COLS:
        reference[col] = pd.to_numeric(reference[col], errors="coerce").fillna(0.0)
    return reference.sort_values(["deck", "sensor_alias"]).reset_index(drop=True)


def ensure_spectral_tables(require_psd: bool = True, force: bool = False, num_workers: int | None = None, max_events: int | None = None) -> dict:
    """Create the notebook CSV dependencies when they are missing.

    The score and timing tables are restored from the cached Hilbert event table.
    The PSD feature/reference tables are rebuilt from the HDF5 event files only
    when they are requested and missing.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    events = _load_hilbert_events()

    if force or not SCORE_PATH.exists() or not TIMING_PATH.exists():
        _write_score_and_timing_tables(events)

    if require_psd:
        if num_workers is None:
            num_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
        psd_missing = force or not FEATURE_PATH.exists() or not REFERENCE_PATH.exists() or not SCALE_PATH.exists()
        if psd_missing:
            if FEATURE_PATH.exists() and not force and max_events is None:
                features = pd.read_csv(FEATURE_PATH, low_memory=False)
            else:
                features = _build_event_sensor_features(events, num_workers=num_workers, max_events=max_events)
                features.to_csv(FEATURE_PATH, index=False)

            reference = _build_reference_table(features)
            reference.to_csv(REFERENCE_PATH, index=False)

            scale_table, _ = robust_sensor_band_scale(features, reference)
            scale_table.to_csv(SCALE_PATH, index=False)

    return {
        "score_path": SCORE_PATH,
        "timing_path": TIMING_PATH,
        "feature_path": FEATURE_PATH,
        "reference_path": REFERENCE_PATH,
        "scale_path": SCALE_PATH,
        "event_count": int(len(events)),
        "hdf5_root": HDF5_ROOT,
    }
