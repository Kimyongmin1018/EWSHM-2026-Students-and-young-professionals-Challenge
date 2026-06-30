from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
KYM_SRC = PROJECT_ROOT / "KYM" / "src"
if str(KYM_SRC) not in sys.path:
    sys.path.insert(0, str(KYM_SRC))

from aquinas.config import DEFAULT_DATASET_PATH, DECKS, SET_ORDER  # noqa: E402
from aquinas.dataset import build_all_event_tables, parse_mixed_datetime, sensor_names_for_deck  # noqa: E402

TARGET_LENGTH = 4096
SAMPLE_RATE_HZ = 100.0
STEP_NS = int(1e9 / SAMPLE_RATE_HZ)
TOLERANCE_NS = int(5e6)
BASELINE_POINTS = 50
OUTPUT_DIR = PROJECT_ROOT / "EWSHM_dataset_preprocessed"


def parse_set_parts(set_name: str) -> Tuple[str, str, str]:
    match = re.match(r"^AQUINAS_(SET\d+)_(\d{4})_(\d{2})$", set_name)
    if not match:
        raise ValueError(f"Unexpected set name format: {set_name}")
    return match.group(1), match.group(2), match.group(3)


def output_filename(set_name: str, deck: str, chunk_number: int | None = None) -> str:
    set_id, year, month = parse_set_parts(set_name)
    if chunk_number is None:
        return f"AQUINAS_{set_id}_{deck}_{year}_{month}.hdf5"
    return f"AQUINAS_{set_id}_{deck}_{year}_{month}_PART{chunk_number:03d}.hdf5"


def source_chunk_label(set_name: str, chunk_number: int) -> str:
    set_id, _, _ = parse_set_parts(set_name)
    return f"{set_id}_{chunk_number}"


def source_chunk_from_file(file_name: object) -> int | None:
    if not isinstance(file_name, str):
        return None
    match = re.search(r"_SET\d+_(\d+)\.json$", file_name)
    return int(match.group(1)) if match else None


def event_source_chunk(event_row: pd.Series, sensors: List[str]) -> int:
    chunks = []
    for sensor in sensors:
        chunk_number = source_chunk_from_file(event_row.get(f"{sensor}__file"))
        if chunk_number is not None:
            chunks.append(chunk_number)
    if not chunks:
        return -1
    unique_chunks = sorted(set(chunks))
    if len(unique_chunks) != 1:
        raise ValueError(f"Event spans multiple raw chunks: {unique_chunks}")
    return unique_chunks[0]


def add_source_chunk_columns(part: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    part = part.copy()
    source_chunks = [event_source_chunk(event_row, sensors) for _, event_row in part.iterrows()]
    if any(chunk < 0 for chunk in source_chunks):
        missing = sum(chunk < 0 for chunk in source_chunks)
        raise ValueError(f"Cannot infer source chunk for {missing} events.")
    part["source_chunk"] = np.asarray(source_chunks, dtype=np.int16)
    part["source_chunk_label"] = [source_chunk_label(set_name, int(chunk)) for set_name, chunk in zip(part["set_name"], part["source_chunk"])]
    return part


def sensor_alias(sensor_name: str, deck: str) -> str:
    prefix = f"{deck}_"
    return sensor_name[len(prefix):] if sensor_name.startswith(prefix) else sensor_name


@lru_cache(maxsize=128)
def load_chunk_arrays(chunk_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(chunk_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    value_keys = [key for key in raw.keys() if key != "timestamp"]
    if len(value_keys) != 1:
        raise ValueError(f"Expected exactly one sensor payload in {chunk_path}, found {value_keys}")
    timestamps = parse_mixed_datetime(pd.Series(raw["timestamp"])).astype("int64").to_numpy()
    values = np.asarray(raw[value_keys[0]], dtype=np.float32)
    return timestamps, values


def get_event_sensor_slice(event_row: pd.Series, dataset_path: Path, sensor: str) -> Tuple[np.ndarray, np.ndarray] | None:
    file_col = f"{sensor}__file"
    start_col = f"{sensor}__start_row"
    end_col = f"{sensor}__end_row"
    if file_col not in event_row or pd.isna(event_row[file_col]):
        return None
    chunk_path = dataset_path / event_row["set_name"] / sensor / str(event_row[file_col])
    timestamps, values = load_chunk_arrays(str(chunk_path))
    start_idx = max(int(event_row[start_col]) - 1, 0)
    end_idx = min(int(event_row[end_col]), len(values))
    if end_idx <= start_idx:
        return None
    return timestamps[start_idx:end_idx], values[start_idx:end_idx]


def empty_processed(target_length: int) -> Dict[str, object]:
    return {
        "values": np.zeros(target_length, dtype=np.float32),
        "mask": np.zeros(target_length, dtype=np.uint8),
        "baseline": np.float32(0.0),
        "raw_sample_count": 0,
        "mapped_sample_count": 0,
        "valid_length": 0,
        "first_valid_index": -1,
        "last_valid_index": -1,
        "truncated": False,
    }


def zero_align_pad(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    grid_ns: np.ndarray,
    baseline_points: int = BASELINE_POINTS,
    tolerance_ns: int = TOLERANCE_NS,
) -> Dict[str, object]:
    target_length = len(grid_ns)
    if len(timestamps_ns) == 0 or len(values) == 0:
        return empty_processed(target_length)

    baseline_count = min(baseline_points, len(values))
    baseline = np.nanmean(values[:baseline_count], dtype=np.float64)
    if not np.isfinite(baseline):
        baseline = 0.0
    zeroed_values = values.astype(np.float32, copy=True) - np.float32(baseline)

    right = np.searchsorted(timestamps_ns, grid_ns, side="left")
    right_clipped = np.clip(right, 0, len(timestamps_ns) - 1)
    left_clipped = np.clip(right - 1, 0, len(timestamps_ns) - 1)
    left_dist = np.abs(grid_ns - timestamps_ns[left_clipped])
    right_dist = np.abs(grid_ns - timestamps_ns[right_clipped])
    use_left = left_dist <= right_dist
    nearest_idx = np.where(use_left, left_clipped, right_clipped)
    nearest_dist = np.where(use_left, left_dist, right_dist)

    valid = nearest_dist <= tolerance_ns
    aligned = np.full(target_length, np.nan, dtype=np.float32)
    aligned[valid] = zeroed_values[nearest_idx[valid]]
    valid_indices = np.flatnonzero(np.isfinite(aligned))

    out = np.zeros(target_length, dtype=np.float32)
    mask = np.zeros(target_length, dtype=np.uint8)
    truncated = bool(timestamps_ns[-1] > grid_ns[-1] + tolerance_ns)
    if len(valid_indices) == 0:
        processed = empty_processed(target_length)
        processed.update({"baseline": np.float32(baseline), "raw_sample_count": int(len(values)), "truncated": truncated})
        return processed

    first_idx = int(valid_indices[0])
    last_idx = int(valid_indices[-1])
    mask[first_idx:last_idx + 1] = 1
    if len(valid_indices) == 1:
        out[first_idx:] = aligned[valid_indices[0]]
        out[:first_idx] = 0.0
    else:
        x = np.arange(target_length, dtype=np.float64)
        interpolated = np.interp(x, valid_indices.astype(np.float64), aligned[valid_indices].astype(np.float64))
        out[:] = interpolated.astype(np.float32)
        out[:first_idx] = 0.0
        out[last_idx + 1:] = out[last_idx]

    return {
        "values": out,
        "mask": mask,
        "baseline": np.float32(baseline),
        "raw_sample_count": int(len(values)),
        "mapped_sample_count": int(len(valid_indices)),
        "valid_length": int(mask.sum()),
        "first_valid_index": first_idx,
        "last_valid_index": last_idx,
        "truncated": truncated,
    }


def create_sensor_group(
    h5: h5py.File,
    sensor: str,
    alias: str,
    n_events: int,
    target_length: int,
    compression: str | None,
    compression_opts: int | None,
) -> Dict[str, h5py.Dataset]:
    group = h5.create_group(f"sensors/{alias}")
    group.attrs["full_sensor_name"] = sensor
    group.attrs["sensor_alias"] = alias
    chunk_events = max(1, min(64, n_events))
    common_kwargs = {
        "shape": (n_events, target_length),
        "chunks": (chunk_events, target_length),
        "compression": compression,
        "shuffle": True,
    }
    if compression == "gzip" and compression_opts is not None:
        common_kwargs["compression_opts"] = compression_opts
    return {
        "values": group.create_dataset("values", dtype=np.float32, **common_kwargs),
        "mask": group.create_dataset("mask", dtype=np.uint8, **common_kwargs),
        "valid_length": group.create_dataset("valid_length", shape=(n_events,), dtype=np.int32, compression=compression),
        "raw_sample_count": group.create_dataset("raw_sample_count", shape=(n_events,), dtype=np.int32, compression=compression),
        "mapped_sample_count": group.create_dataset("mapped_sample_count", shape=(n_events,), dtype=np.int32, compression=compression),
        "first_valid_index": group.create_dataset("first_valid_index", shape=(n_events,), dtype=np.int16, compression=compression),
        "last_valid_index": group.create_dataset("last_valid_index", shape=(n_events,), dtype=np.int16, compression=compression),
        "baseline_value": group.create_dataset("baseline_value", shape=(n_events,), dtype=np.float32, compression=compression),
        "truncated": group.create_dataset("truncated", shape=(n_events,), dtype=np.uint8, compression=compression),
    }


def write_event_metadata(h5: h5py.File, part: pd.DataFrame, sensors: List[str], aliases: List[str], target_length: int) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    events_group = h5.create_group("events")
    event_start = parse_mixed_datetime(part["event_start"])
    events_group.create_dataset("event_key", data=part["event_key"].astype(str).to_numpy(), dtype=str_dtype)
    events_group.create_dataset("event_id", data=part["event_id"].to_numpy(dtype=np.int32))
    events_group.create_dataset("event_start", data=event_start.dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(), dtype=str_dtype)
    events_group.create_dataset("event_start_unix_ns", data=event_start.astype("int64").to_numpy(dtype=np.int64))
    events_group.create_dataset("temperature_c", data=part["temperature_c"].to_numpy(dtype=np.float32))
    events_group.create_dataset("event_duration_sec", data=part["event_duration_sec"].to_numpy(dtype=np.float32))
    events_group.create_dataset("sensor_coverage", data=part["sensor_coverage"].to_numpy(dtype=np.float32))
    events_group.create_dataset("set_index", data=part["set_index"].to_numpy(dtype=np.int16))
    if "source_chunk" in part.columns:
        events_group.create_dataset("source_chunk", data=part["source_chunk"].to_numpy(dtype=np.int16))
    if "source_chunk_label" in part.columns:
        events_group.create_dataset("source_chunk_label", data=part["source_chunk_label"].astype(str).to_numpy(), dtype=str_dtype)
    h5.create_dataset("sensor_names", data=np.asarray(sensors, dtype=object), dtype=str_dtype)
    h5.create_dataset("sensor_aliases", data=np.asarray(aliases, dtype=object), dtype=str_dtype)
    h5.create_dataset("time_grid_seconds", data=np.arange(target_length, dtype=np.float32) / np.float32(SAMPLE_RATE_HZ), dtype=np.float32)


def write_set_deck_hdf5(
    events_df: pd.DataFrame,
    dataset_path: Path,
    output_dir: Path,
    set_name: str,
    deck: str,
    target_length: int,
    baseline_points: int,
    tolerance_ns: int,
    compression: str | None,
    compression_opts: int | None,
    overwrite: bool,
    chunk_number: int | None = None,
    limit_events: int | None = None,
) -> Path:
    part = events_df.loc[events_df["set_name"].eq(set_name) & events_df["deck"].eq(deck)].sort_values("event_start").reset_index(drop=True)
    if part.empty:
        raise ValueError(f"No events found for {set_name} {deck}")

    sensors = sensor_names_for_deck(part, deck)
    part = add_source_chunk_columns(part, sensors)
    if chunk_number is not None:
        part = part.loc[part["source_chunk"].eq(chunk_number)].reset_index(drop=True)
    if limit_events is not None:
        part = part.head(limit_events).copy()
    if part.empty:
        raise ValueError(f"No events found for {set_name} {deck} chunk {chunk_number}")

    aliases = [sensor_alias(sensor, deck) for sensor in sensors]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(set_name, deck, chunk_number=chunk_number)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() and not overwrite:
        print(f"Skip existing file: {output_path}")
        return output_path
    if tmp_path.exists():
        tmp_path.unlink()

    n_events = len(part)
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["dataset_path"] = str(dataset_path)
        h5.attrs["set_name"] = set_name
        h5.attrs["deck"] = deck
        h5.attrs["grouping"] = "source_chunk" if chunk_number is not None else "set_deck"
        h5.attrs["source_chunk"] = -1 if chunk_number is None else chunk_number
        h5.attrs["source_chunk_label"] = "combined" if chunk_number is None else source_chunk_label(set_name, chunk_number)
        h5.attrs["event_count"] = n_events
        h5.attrs["target_length"] = target_length
        h5.attrs["sample_rate_hz"] = SAMPLE_RATE_HZ
        h5.attrs["sample_interval_ms"] = 10.0
        h5.attrs["alignment_tolerance_ms"] = tolerance_ns / 1e6
        h5.attrs["zeroing_baseline_points"] = baseline_points
        h5.attrs["zeroing_baseline_seconds"] = baseline_points / SAMPLE_RATE_HZ
        h5.attrs["padding_policy"] = "tail forward-fill with mask=0; head fill zero"
        h5.attrs["mask_policy"] = "1 from first to last aligned real sample; 0 for leading gap and padded tail"
        h5.attrs["internal_layout"] = "/sensors/<deckless_sensor_alias>/{values,mask,...} and /events/* metadata"
        write_event_metadata(h5, part, sensors, aliases, target_length)
        sensor_datasets = {sensor: create_sensor_group(h5, sensor, alias, n_events, target_length, compression, compression_opts) for sensor, alias in zip(sensors, aliases)}
        truncated_sensor_count = np.zeros(n_events, dtype=np.int16)
        available_sensor_count = np.zeros(n_events, dtype=np.int16)
        grid_start_unix_ns = np.zeros(n_events, dtype=np.int64)
        offsets = np.arange(target_length, dtype=np.int64) * STEP_NS

        desc = f"{set_name} {deck}" if chunk_number is None else f"{set_name} {deck} PART{chunk_number:03d}"
        iterator = tqdm(part.iterrows(), total=n_events, desc=desc, unit="event")
        for event_idx, (_, event_row) in enumerate(iterator):
            slices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for sensor in sensors:
                sensor_slice = get_event_sensor_slice(event_row, dataset_path, sensor)
                if sensor_slice is not None:
                    slices[sensor] = sensor_slice
            if slices:
                min_time_ns = min(int(timestamps[0]) for timestamps, _ in slices.values() if len(timestamps) > 0)
            else:
                min_time_ns = int(pd.Timestamp(event_row["event_start"]).value)
            grid_start_unix_ns[event_idx] = min_time_ns
            grid_ns = min_time_ns + offsets

            for sensor in sensors:
                datasets = sensor_datasets[sensor]
                processed = zero_align_pad(*slices[sensor], grid_ns, baseline_points=baseline_points, tolerance_ns=tolerance_ns) if sensor in slices else empty_processed(target_length)
                datasets["values"][event_idx, :] = processed["values"]
                datasets["mask"][event_idx, :] = processed["mask"]
                datasets["valid_length"][event_idx] = processed["valid_length"]
                datasets["raw_sample_count"][event_idx] = processed["raw_sample_count"]
                datasets["mapped_sample_count"][event_idx] = processed["mapped_sample_count"]
                datasets["first_valid_index"][event_idx] = processed["first_valid_index"]
                datasets["last_valid_index"][event_idx] = processed["last_valid_index"]
                datasets["baseline_value"][event_idx] = processed["baseline"]
                datasets["truncated"][event_idx] = np.uint8(processed["truncated"])
                available_sensor_count[event_idx] += np.int16(processed["raw_sample_count"] > 0)
                truncated_sensor_count[event_idx] += np.int16(processed["truncated"])

        h5["events"].create_dataset("grid_start_unix_ns", data=grid_start_unix_ns)
        h5["events"].create_dataset("available_sensor_count", data=available_sensor_count)
        h5["events"].create_dataset("truncated_sensor_count", data=truncated_sensor_count)
    tmp_path.replace(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-length masked AQUINAS HDF5 files.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sets", nargs="*", default=list(SET_ORDER))
    parser.add_argument("--decks", nargs="*", default=list(DECKS), choices=list(DECKS))
    parser.add_argument("--target-length", type=int, default=TARGET_LENGTH)
    parser.add_argument("--baseline-points", type=int, default=BASELINE_POINTS)
    parser.add_argument("--tolerance-ms", type=float, default=5.0)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-event-cache", action="store_true")
    parser.add_argument("--grouping", choices=["chunk", "set-deck"], default="chunk", help="Default chunk writes one file per set/deck/raw JSON chunk.")
    parser.add_argument("--chunks", nargs="*", type=int, default=None, help="Only write selected raw chunk numbers when --grouping chunk.")
    parser.add_argument("--limit-events", type=int, default=None, help="Debug option: write only the first N events per file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if compression == "gzip" else None
    tolerance_ns = int(args.tolerance_ms * 1e6)
    events = build_all_event_tables(dataset_path=args.dataset_path, force=args.force_event_cache)
    written = []
    for set_name in args.sets:
        for deck in args.decks:
            if args.grouping == "set-deck":
                chunk_numbers = [None]
            else:
                part = events.loc[events["set_name"].eq(set_name) & events["deck"].eq(deck)].sort_values("event_start").reset_index(drop=True)
                if part.empty:
                    raise ValueError(f"No events found for {set_name} {deck}")
                sensors = sensor_names_for_deck(part, deck)
                part = add_source_chunk_columns(part, sensors)
                chunk_numbers = sorted(int(chunk) for chunk in part["source_chunk"].drop_duplicates())
                if args.chunks is not None:
                    requested = set(args.chunks)
                    chunk_numbers = [chunk for chunk in chunk_numbers if chunk in requested]
            for chunk_number in chunk_numbers:
                path = write_set_deck_hdf5(
                    events_df=events,
                    dataset_path=args.dataset_path,
                    output_dir=args.output_dir,
                    set_name=set_name,
                    deck=deck,
                    target_length=args.target_length,
                    baseline_points=args.baseline_points,
                    tolerance_ns=tolerance_ns,
                    compression=compression,
                    compression_opts=compression_opts,
                    overwrite=args.overwrite,
                    chunk_number=chunk_number,
                    limit_events=args.limit_events,
                )
                written.append(path)
                print(f"Wrote {path}")
    print("Completed HDF5 preprocessing.")
    for path in written:
        if path.exists():
            print(f"{path} ({path.stat().st_size / (1024 ** 2):.1f} MiB)")


if __name__ == "__main__":
    main()
