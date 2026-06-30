from __future__ import annotations

import argparse
import sys
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
from aquinas.dataset import build_all_event_tables, sensor_names_for_deck  # noqa: E402
from preprocess_aquinas_hdf5 import (  # noqa: E402
    BASELINE_POINTS,
    SAMPLE_RATE_HZ,
    STEP_NS,
    TARGET_LENGTH,
    add_source_chunk_columns,
    empty_processed,
    get_event_sensor_slice,
    parse_set_parts,
    sensor_alias,
    source_chunk_label,
    write_event_metadata,
    zero_align_pad,
)

EVENT_OUTPUT_DIR = PROJECT_ROOT / "EWSHM_dataset_preprocessed_event_level"


def event_output_filename(set_name: str, deck: str, chunk_number: int, event_id: int, event_index_in_chunk: int) -> str:
    set_id, year, month = parse_set_parts(set_name)
    return f"AQUINAS_{set_id}_{deck}_{year}_{month}_PART{chunk_number:03d}_EVENT{event_index_in_chunk:03d}_ID{event_id:06d}.hdf5"


def event_output_path(output_dir: Path, event_row: pd.Series, chunk_number: int, event_index_in_chunk: int) -> Path:
    set_name = str(event_row["set_name"])
    deck = str(event_row["deck"])
    event_id = int(event_row["event_id"])
    return (
        output_dir
        / set_name
        / deck
        / f"PART{chunk_number:03d}"
        / event_output_filename(set_name, deck, chunk_number, event_id, event_index_in_chunk)
    )


def compression_kwargs(compression: str | None, compression_opts: int | None) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if compression == "gzip" and compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    return kwargs


def write_single_event_hdf5(
    event_row: pd.Series,
    sensors: List[str],
    aliases: List[str],
    dataset_path: Path,
    output_path: Path,
    target_length: int,
    baseline_points: int,
    tolerance_ns: int,
    compression: str | None,
    compression_opts: int | None,
    event_index_in_chunk: int,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    part = pd.DataFrame([event_row])
    chunk_number = int(event_row["source_chunk"])
    set_name = str(event_row["set_name"])
    deck = str(event_row["deck"])
    n_sensors = len(sensors)

    values = np.zeros((n_sensors, target_length), dtype=np.float32)
    masks = np.zeros((n_sensors, target_length), dtype=np.uint8)
    valid_length = np.zeros(n_sensors, dtype=np.int32)
    raw_sample_count = np.zeros(n_sensors, dtype=np.int32)
    mapped_sample_count = np.zeros(n_sensors, dtype=np.int32)
    first_valid_index = np.full(n_sensors, -1, dtype=np.int16)
    last_valid_index = np.full(n_sensors, -1, dtype=np.int16)
    baseline_value = np.zeros(n_sensors, dtype=np.float32)
    truncated = np.zeros(n_sensors, dtype=np.uint8)

    slices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sensor in sensors:
        sensor_slice = get_event_sensor_slice(event_row, dataset_path, sensor)
        if sensor_slice is not None:
            slices[sensor] = sensor_slice

    if slices:
        min_time_ns = min(int(timestamps[0]) for timestamps, _ in slices.values() if len(timestamps) > 0)
    else:
        min_time_ns = int(pd.Timestamp(event_row["event_start"]).value)
    grid_ns = min_time_ns + np.arange(target_length, dtype=np.int64) * STEP_NS

    for sensor_idx, sensor in enumerate(sensors):
        processed = (
            zero_align_pad(*slices[sensor], grid_ns, baseline_points=baseline_points, tolerance_ns=tolerance_ns)
            if sensor in slices
            else empty_processed(target_length)
        )
        values[sensor_idx, :] = processed["values"]
        masks[sensor_idx, :] = processed["mask"]
        valid_length[sensor_idx] = processed["valid_length"]
        raw_sample_count[sensor_idx] = processed["raw_sample_count"]
        mapped_sample_count[sensor_idx] = processed["mapped_sample_count"]
        first_valid_index[sensor_idx] = processed["first_valid_index"]
        last_valid_index[sensor_idx] = processed["last_valid_index"]
        baseline_value[sensor_idx] = processed["baseline"]
        truncated[sensor_idx] = np.uint8(processed["truncated"])

    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["dataset_path"] = str(dataset_path)
        h5.attrs["set_name"] = set_name
        h5.attrs["deck"] = deck
        h5.attrs["grouping"] = "event"
        h5.attrs["source_chunk"] = chunk_number
        h5.attrs["source_chunk_label"] = source_chunk_label(set_name, chunk_number)
        h5.attrs["event_count"] = 1
        h5.attrs["event_id"] = int(event_row["event_id"])
        h5.attrs["event_key"] = str(event_row["event_key"])
        h5.attrs["event_index_in_chunk"] = int(event_index_in_chunk)
        h5.attrs["target_length"] = target_length
        h5.attrs["sample_rate_hz"] = SAMPLE_RATE_HZ
        h5.attrs["sample_interval_ms"] = 10.0
        h5.attrs["alignment_tolerance_ms"] = tolerance_ns / 1e6
        h5.attrs["zeroing_baseline_points"] = baseline_points
        h5.attrs["zeroing_baseline_seconds"] = baseline_points / SAMPLE_RATE_HZ
        h5.attrs["padding_policy"] = "tail forward-fill with mask=0; head fill zero"
        h5.attrs["mask_policy"] = "1 from first to last aligned real sample; 0 for leading gap and padded tail"
        h5.attrs["internal_layout"] = "/values[sensor, time], /mask[sensor, time], sensor order in /sensor_aliases"

        write_event_metadata(h5, part, sensors, aliases, target_length)
        h5["events"].create_dataset("grid_start_unix_ns", data=np.asarray([min_time_ns], dtype=np.int64))
        h5["events"].create_dataset("available_sensor_count", data=np.asarray([int((raw_sample_count > 0).sum())], dtype=np.int16))
        h5["events"].create_dataset("truncated_sensor_count", data=np.asarray([int(truncated.sum())], dtype=np.int16))
        h5["events"].create_dataset("event_index_in_chunk", data=np.asarray([event_index_in_chunk], dtype=np.int16))

        data_kwargs = compression_kwargs(compression, compression_opts)
        h5.create_dataset("values", data=values, chunks=(1, target_length), shuffle=True, **data_kwargs)
        h5.create_dataset("mask", data=masks, chunks=(1, target_length), shuffle=True, **data_kwargs)
        h5.create_dataset("valid_length", data=valid_length, **data_kwargs)
        h5.create_dataset("raw_sample_count", data=raw_sample_count, **data_kwargs)
        h5.create_dataset("mapped_sample_count", data=mapped_sample_count, **data_kwargs)
        h5.create_dataset("first_valid_index", data=first_valid_index, **data_kwargs)
        h5.create_dataset("last_valid_index", data=last_valid_index, **data_kwargs)
        h5.create_dataset("baseline_value", data=baseline_value, **data_kwargs)
        h5.create_dataset("truncated", data=truncated, **data_kwargs)

    tmp_path.replace(output_path)
    return output_path


def iter_event_rows(events: pd.DataFrame, set_name: str, deck: str, chunks: set[int] | None):
    part = events.loc[events["set_name"].eq(set_name) & events["deck"].eq(deck)].sort_values("event_start").reset_index(drop=True)
    if part.empty:
        return [], [], []
    sensors = sensor_names_for_deck(part, deck)
    aliases = [sensor_alias(sensor, deck) for sensor in sensors]
    part = add_source_chunk_columns(part, sensors)
    if chunks is not None:
        part = part.loc[part["source_chunk"].isin(chunks)].reset_index(drop=True)
    if part.empty:
        return [], sensors, aliases

    rows = []
    for _, chunk_frame in part.groupby("source_chunk", sort=True):
        for event_index_in_chunk, (_, event_row) in enumerate(chunk_frame.sort_values("event_start").iterrows(), start=1):
            rows.append((event_row, event_index_in_chunk))
    return rows, sensors, aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one fixed-length AQUINAS HDF5 file per event.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=EVENT_OUTPUT_DIR)
    parser.add_argument("--sets", nargs="*", default=list(SET_ORDER))
    parser.add_argument("--decks", nargs="*", default=list(DECKS), choices=list(DECKS))
    parser.add_argument("--chunks", nargs="*", type=int, default=None, help="Only write selected raw chunk numbers.")
    parser.add_argument("--target-length", type=int, default=TARGET_LENGTH)
    parser.add_argument("--baseline-points", type=int, default=BASELINE_POINTS)
    parser.add_argument("--tolerance-ms", type=float, default=5.0)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-event-cache", action="store_true")
    parser.add_argument("--limit-events", type=int, default=None, help="Debug option: write only the first N events after filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if compression == "gzip" else None
    tolerance_ns = int(args.tolerance_ms * 1e6)
    chunk_filter = set(args.chunks) if args.chunks is not None else None

    events = build_all_event_tables(dataset_path=args.dataset_path, force=args.force_event_cache)

    jobs = []
    sensor_cache = {}
    for set_name in args.sets:
        for deck in args.decks:
            rows, sensors, aliases = iter_event_rows(events, set_name, deck, chunk_filter)
            sensor_cache[(set_name, deck)] = (sensors, aliases)
            for event_row, event_index_in_chunk in rows:
                output_path = event_output_path(args.output_dir, event_row, int(event_row["source_chunk"]), event_index_in_chunk)
                jobs.append((event_row, event_index_in_chunk, output_path))

    if args.limit_events is not None:
        jobs = jobs[: args.limit_events]

    print(f"event_files_to_write={len(jobs)}")
    written = []
    for event_row, event_index_in_chunk, output_path in tqdm(jobs, desc="event-level HDF5", unit="event"):
        sensors, aliases = sensor_cache[(str(event_row["set_name"]), str(event_row["deck"]))]
        path = write_single_event_hdf5(
            event_row=event_row,
            sensors=sensors,
            aliases=aliases,
            dataset_path=args.dataset_path,
            output_path=output_path,
            target_length=args.target_length,
            baseline_points=args.baseline_points,
            tolerance_ns=tolerance_ns,
            compression=compression,
            compression_opts=compression_opts,
            event_index_in_chunk=event_index_in_chunk,
            overwrite=args.overwrite,
        )
        written.append(path)

    print("Completed event-level HDF5 preprocessing.")
    print(f"written_or_existing={len(written)}")
    if written:
        total_size = sum(path.stat().st_size for path in written if path.exists()) / (1024 ** 2)
        print(f"total_size_mib={total_size:.1f}")
        print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
