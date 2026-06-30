import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET_PATH, DECKS, EVENT_MATCH_TOLERANCE_SEC, OUTPUT_ROOT, SET_ORDER

TABLE_STATS = [
    "File",
    "Start_Row",
    "End_Row",
    "Duration",
    "Start_Value",
    "End_Value",
    "Diff_Value",
    "Min_Value",
    "Max_Value",
    "Mean_Value",
    "Range",
    "Temperature",
]

SENSOR_PATTERN = re.compile(
    r"^(?P<deck>OLD|NEW)_(?P<span>S1|S2)_(?P<side>UP|DO)_(?P<location>[A-Z]+)_(?P<quantity>ACC|STR)(?:_(?P<axis>[YZ]))?$"
)


def parse_mixed_datetime(values) -> pd.Series:
    return pd.to_datetime(values, format="mixed")


def to_column_suffix(name: str) -> str:
    return name.lower()


def parse_sensor_name(sensor_name: str) -> Dict[str, str]:
    match = SENSOR_PATTERN.match(sensor_name)
    if not match:
        raise ValueError(f"Unexpected sensor name: {sensor_name}")
    meta = match.groupdict()
    meta["axis"] = meta["axis"] or ""
    return meta


def dataset_cache_path() -> Path:
    cache_dir = OUTPUT_ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "aquinas_event_tables.joblib"


@lru_cache(maxsize=256)
def load_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_table(table_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(load_json(str(table_path)))
    frame["Start_Time"] = parse_mixed_datetime(frame["Start_Time"])
    frame["End_Time"] = parse_mixed_datetime(frame["End_Time"])
    frame["source_table"] = table_path.name
    return frame


@lru_cache(maxsize=128)
def load_chunk(chunk_path: str) -> pd.DataFrame:
    raw = load_json(chunk_path)
    keys = [key for key in raw.keys() if key != "timestamp"]
    if len(keys) != 1:
        raise ValueError(f"Expected one sensor payload in {chunk_path}, found {keys}")
    sensor_key = keys[0]
    return pd.DataFrame(
        {
            "timestamp": parse_mixed_datetime(raw["timestamp"]),
            "value": np.asarray(raw[sensor_key], dtype=float),
        }
    )


def get_set_directories(dataset_path: Path = DEFAULT_DATASET_PATH) -> List[Path]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}. "
            "Set AQUINAS_DATASET_PATH or update aquinas.config.DEFAULT_DATASET_PATH."
        )
    dirs = [dataset_path / set_name for set_name in SET_ORDER if (dataset_path / set_name).exists()]
    if not dirs:
        raise FileNotFoundError(f"No AQUINAS set directories found under {dataset_path}.")
    return dirs


def get_table_paths(set_dir: Path, deck: str = "") -> List[Path]:
    pattern = f"TABLE_{deck}_*_SET*.json" if deck else "TABLE_*_SET*.json"
    return sorted(set_dir.glob(pattern))


def sensor_name_from_table(table_path: Path) -> str:
    return table_path.name.split("TABLE_", 1)[1].rsplit("_SET", 1)[0]


def get_deck_sensor_tables(set_dir: Path, deck: str) -> Dict[str, pd.DataFrame]:
    tables = {}
    for table_path in get_table_paths(set_dir, deck=deck):
        sensor = sensor_name_from_table(table_path)
        frame = load_table(table_path).copy()
        frame["sensor_name"] = sensor
        frame["deck"] = deck
        tables[sensor] = frame
    return tables


def cluster_event_times(sensor_tables: Dict[str, pd.DataFrame], tolerance_sec: float = EVENT_MATCH_TOLERANCE_SEC) -> pd.DataFrame:
    unique_times = []
    for frame in sensor_tables.values():
        unique_times.extend(frame["Start_Time"].drop_duplicates().tolist())
    ordered = pd.Series(sorted(parse_mixed_datetime(pd.Series(unique_times)).drop_duplicates().tolist()))
    cluster_id = 0
    cluster_ids = []
    for index, timestamp in enumerate(ordered):
        if index:
            delta = (timestamp - ordered.iloc[index - 1]).total_seconds()
            if delta > tolerance_sec:
                cluster_id += 1
        cluster_ids.append(cluster_id)
    clustered = pd.DataFrame({"event_time": ordered, "event_id": cluster_ids})
    summary = (
        clustered.groupby("event_id")["event_time"]
        .agg(["min", "max", "median", "size"])
        .rename(columns={"min": "event_start_min", "max": "event_start_max", "median": "event_start"})
        .reset_index()
    )
    return summary


def nearest_event_ids(timestamps: pd.Series, event_times: pd.Series, tolerance_sec: float = EVENT_MATCH_TOLERANCE_SEC) -> np.ndarray:
    event_int = event_times.astype("int64").to_numpy()
    ts_int = timestamps.astype("int64").to_numpy()
    right = np.searchsorted(event_int, ts_int, side="left")
    indices = np.clip(right, 0, len(event_int) - 1)
    left = np.clip(right - 1, 0, len(event_int) - 1)
    left_dist = np.abs(ts_int - event_int[left])
    right_dist = np.abs(ts_int - event_int[indices])
    choose_left = left_dist <= right_dist
    best = np.where(choose_left, left, indices)
    tolerance_ns = int(tolerance_sec * 1e9)
    invalid = np.abs(ts_int - event_int[best]) > tolerance_ns
    best[invalid] = -1
    return best


def build_deck_event_table(set_dir: Path, deck: str, tolerance_sec: float = EVENT_MATCH_TOLERANCE_SEC) -> pd.DataFrame:
    sensor_tables = get_deck_sensor_tables(set_dir, deck)
    event_summary = cluster_event_times(sensor_tables, tolerance_sec=tolerance_sec)
    event_summary["set_name"] = set_dir.name
    event_summary["deck"] = deck
    event_summary["event_key"] = (
        event_summary["set_name"]
        + "::"
        + event_summary["deck"]
        + "::"
        + event_summary["event_id"].astype(str)
    )

    merged = event_summary.copy()
    matched_columns = []
    for sensor, frame in sensor_tables.items():
        assigned = frame.copy()
        nearest_idx = nearest_event_ids(assigned["Start_Time"], event_summary["event_start"], tolerance_sec=tolerance_sec)
        assigned = assigned.loc[nearest_idx >= 0].copy()
        assigned["event_id"] = event_summary.iloc[nearest_idx[nearest_idx >= 0]]["event_id"].to_numpy()
        sensor_columns = ["event_id"]
        rename_map = {}
        for stat in TABLE_STATS:
            if stat not in assigned.columns:
                continue
            source_col = stat
            sensor_columns.append(source_col)
            rename_map[source_col] = f"{sensor}__{to_column_suffix(stat)}"
        sensor_frame = assigned[sensor_columns].drop_duplicates(subset=["event_id"]).rename(columns=rename_map)
        matched_col = f"{sensor}__matched"
        sensor_frame[matched_col] = 1.0
        matched_columns.append(matched_col)
        merged = merged.merge(sensor_frame, on="event_id", how="left")

    merged["sensor_coverage"] = merged[matched_columns].sum(axis=1, min_count=1).fillna(0.0)
    merged["event_duration_sec"] = np.nanmedian(
        np.column_stack(
            [
                merged[col].to_numpy(dtype=float)
                for col in merged.columns
                if col.endswith("__duration")
            ]
        ),
        axis=1,
    )
    merged["temperature_c"] = np.nanmedian(
        np.column_stack(
            [
                merged[col].to_numpy(dtype=float)
                for col in merged.columns
                if col.endswith("__temperature")
            ]
        ),
        axis=1,
    )
    merged["event_date"] = parse_mixed_datetime(merged["event_start"]).dt.floor("D")
    return merged.sort_values("event_start").reset_index(drop=True)


def build_all_event_tables(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    cache_path: Path = None,
    force: bool = False,
) -> pd.DataFrame:
    cache_path = cache_path or dataset_cache_path()
    if cache_path.exists() and not force:
        return joblib.load(cache_path)
    frames = []
    for set_dir in get_set_directories(dataset_path):
        for deck in DECKS:
            frames.append(build_deck_event_table(set_dir, deck))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["set_index"] = combined["set_name"].map({name: idx for idx, name in enumerate(SET_ORDER)})
    joblib.dump(combined, cache_path)
    return combined


def waveform_from_event_row(
    event_row: pd.Series,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    sensor_name: str = "",
) -> pd.DataFrame:
    sensor_name = sensor_name or f"{event_row['deck']}_S1_DO_MID_ACC_Z"
    file_col = f"{sensor_name}__file"
    start_row_col = f"{sensor_name}__start_row"
    end_row_col = f"{sensor_name}__end_row"
    if pd.isna(event_row.get(file_col)):
        raise ValueError(f"No waveform metadata for {sensor_name}")
    set_dir = Path(dataset_path) / event_row["set_name"] / sensor_name
    chunk_path = set_dir / event_row[file_col]
    frame = load_chunk(str(chunk_path))
    start_idx = int(event_row[start_row_col]) - 1
    end_idx = int(event_row[end_row_col])
    return frame.iloc[start_idx:end_idx].reset_index(drop=True)


def summarize_dataset(events_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        events_df.groupby(["set_name", "deck"])
        .agg(
            events=("event_key", "count"),
            start=("event_start", "min"),
            end=("event_start", "max"),
            temp_min=("temperature_c", "min"),
            temp_max=("temperature_c", "max"),
            coverage_median=("sensor_coverage", "median"),
        )
        .reset_index()
        .sort_values(["set_name", "deck"])
    )
    return summary


def available_sensor_names(events_df: pd.DataFrame) -> List[str]:
    sensors = sorted({column.split("__", 1)[0] for column in events_df.columns if "__" in column})
    return sensors


def sensor_names_for_deck(events_df: pd.DataFrame, deck: str) -> List[str]:
    return [sensor for sensor in available_sensor_names(events_df) if sensor.startswith(f"{deck}_")]


def iter_sensor_pairs(prefixes: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
    for left, right in prefixes:
        yield left, right
