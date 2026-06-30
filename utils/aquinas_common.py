from __future__ import annotations

import os
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, detrend, hilbert, sosfiltfilt, welch

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(os.environ.get("AQUINAS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
HDF5_ROOT = Path(os.environ.get("AQUINAS_HDF5_ROOT", ROOT / "EWSHM_dataset_preprocessed_event_level")).expanduser()
OUT = Path(os.environ.get("AQUINAS_OUTPUT_DIR", ROOT / "AutoResearch_generated_method")).expanduser()
TABLE_DIR = OUT / "tables_v2"
FIG_DIR = OUT / "figures_v2"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

FS = 100.0
EPS = 1e-12
PLOT_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = 11
COLORBAR_TICK_FONTSIZE = 10
SENSOR_INDEX_LEGEND_FONTSIZE = 12.0
SENSOR_INDEX_LEGEND_TITLE_FONTSIZE = 16.0

BANDS = [
    ("quasi_static", 0.05, 0.80, "vehicle passage / bending"),
    ("low_dynamic", 0.80, 2.00, "global low dynamic"),
    ("mid_dynamic", 2.00, 5.00, "modal / VBI candidate"),
    ("high_dynamic", 5.00, 12.00, "local / roughness"),
    ("noise_sensitive", 12.00, 25.00, "noise / impact-like"),
]
BAND_COLS = [f"band_{name}" for name, *_ in BANDS]
BAND_LABELS_SHORT = ["Quasi-static", "Low dyn.", "Mid dyn.", "High dyn.", "Noise-sens."]
BAND_LABELS = [
    "Quasi-static\n0.05-0.80 Hz",
    "Low dyn.\n0.80-2.00 Hz",
    "Mid dyn.\n2.00-5.00 Hz",
    "High dyn.\n5.00-12.00 Hz",
    "Noise-sens.\n12.00-25.00 Hz",
]

DECK_COLORS = {"OLD": "#c43c39", "NEW": "#2f6fb0"}


def configure_notebook_style(dpi: int = 130, grid: bool = True) -> None:
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 220)
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.unicode_minus"] = False


def decode_aliases(raw):
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw]


def read_event_with_attrs(path):
    path = Path(path)
    with h5py.File(path, "r") as h5:
        values = h5["values"][()]
        mask = h5["mask"][()].astype(bool)
        aliases = decode_aliases(h5["sensor_aliases"][()])
        time_grid = h5["time_grid_seconds"][()]
        attrs = dict(h5.attrs)
    return values, mask, aliases, time_grid, attrs


def read_event_hdf5(path):
    values, mask, aliases, time_grid, _ = read_event_with_attrs(path)
    return values, mask, aliases, time_grid


def read_event_arrays(path):
    values, mask, aliases, time_grid, attrs = read_event_with_attrs(path)
    return values, mask, time_grid, aliases, attrs


def choose_events(score_path: Path | None = None):
    score_path = TABLE_DIR / "event_spectral_health_scores_v2.csv" if score_path is None else Path(score_path)
    usecols = [
        "path",
        "campaign_month",
        "deck",
        "event_id",
        "event_health_score_v2",
        "event_anomaly_score_v2",
        "event_confidence_score_v2",
    ]
    scores = pd.read_csv(score_path, usecols=usecols, low_memory=False)
    normal = scores.iloc[
        (scores["event_health_score_v2"] - scores["event_health_score_v2"].median()).abs().argsort()[:1]
    ].iloc[0]
    anomaly = scores[scores["event_confidence_score_v2"] >= 70].sort_values("event_health_score_v2").iloc[0]
    return scores, normal, anomaly


def choose_reference_event(score_path: Path | None = None):
    score_path = TABLE_DIR / "event_spectral_health_scores_v2.csv" if score_path is None else Path(score_path)
    usecols = [
        "path",
        "campaign_month",
        "deck",
        "event_id",
        "part_index",
        "event_health_score_v2",
    ]
    scores = pd.read_csv(score_path, usecols=usecols, low_memory=False)
    ordered = scores.sort_values(
        [
            "event_health_score_v2",
            "campaign_month",
            "deck",
            "part_index",
            "event_id",
        ],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    return ordered.iloc[0]


def choose_event_row(prefer_low_health: bool = False, deck: str | None = None):
    scores, normal, anomaly = choose_events()
    if deck is not None:
        scores = scores[scores["deck"].eq(deck)].copy()
    if prefer_low_health:
        row = scores[scores["event_confidence_score_v2"] >= 70].sort_values("event_health_score_v2").iloc[0]
    else:
        row = scores.iloc[
            (scores["event_health_score_v2"] - scores["event_health_score_v2"].median()).abs().argsort()[:1]
        ].iloc[0]
    return scores, row


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def choose_event_path(set_name="AQUINAS_SET1_2022_07", deck="OLD", part=4, event=52):
    part_dir = HDF5_ROOT / set_name / deck.upper() / f"PART{int(part):03d}"
    matches = sorted(part_dir.glob(f"*EVENT{int(event):03d}_*.hdf5"))
    if not matches:
        raise FileNotFoundError(f"No matching event under {part_dir} for EVENT{int(event):03d}")
    return matches[0]


def parse_alias(alias):
    parts = str(alias).split("_")
    return {
        "sensor_alias": alias,
        "span": parts[0] if len(parts) > 0 else "",
        "side": parts[1] if len(parts) > 1 else "",
        "location": parts[2] if len(parts) > 2 else "",
        "quantity": parts[3] if len(parts) > 3 else "",
        "axis": parts[4] if len(parts) > 4 else "",
    }


def compact_display_path(path):
    path = str(path)
    marker = "EWSHM_dataset_preprocessed_event_level/"
    if marker in path:
        return path.split(marker, 1)[1]
    return path


def pretty_alias(alias):
    return str(alias)


def clean_signal(x, valid, fill_nan: bool = False):
    y = np.asarray(x, dtype=float).copy()
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() == 0:
        return np.array([])
    med = np.nanmedian(y[valid])
    y = y - med
    if fill_nan:
        y[~valid] = 0.0
    else:
        y[~valid] = np.nan
    return y


def fill_signal(x, valid):
    y = np.asarray(x, dtype=float).copy()
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() == 0:
        return np.zeros_like(y, dtype=float)
    med = np.nanmedian(y[valid])
    y = y - med
    y[~valid] = 0.0
    return y


def contiguous_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def valid_time_bounds(mask, time_grid):
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim == 2:
        valid = valid.any(axis=0)
    if not valid.any():
        return float(time_grid[0]), float(time_grid[-1])
    return float(time_grid[valid][0]), float(time_grid[valid][-1])


def timing_bounds_for_event(path):
    timing = pd.read_csv(TABLE_DIR / "event_hilbert_timing_diagnostics_v2.csv", low_memory=False)
    row = timing[timing["path"].eq(str(path))]
    if row.empty:
        return None
    row = row.iloc[0]
    return float(row["active_start_s_v2"]), float(row["active_end_s_v2"])


def active_window_mask(time_grid, valid_mask, start_s, end_s):
    return (np.asarray(time_grid) >= start_s) & (np.asarray(time_grid) <= end_s) & np.asarray(valid_mask, dtype=bool)


def active_waveform(values, mask, time_grid, sensor_idx, start_s, end_s, detrend_signal: bool = True):
    valid = active_window_mask(time_grid, mask[sensor_idx], start_s, end_s)
    t = np.asarray(time_grid)[valid]
    x = np.asarray(values[sensor_idx], dtype=float)[valid]
    if x.size:
        x = x - np.nanmedian(x)
        if detrend_signal and x.size >= 3:
            x = detrend(x, type="linear")
    return t, x


def bandpass_strain(x, fs: float = FS, band=(0.5, 5.0)):
    sos = butter(3, list(band), btype="bandpass", fs=fs, output="sos")
    if len(x) < 64:
        return x
    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        return x


def sensor_hilbert_window(envelope, valid, time_grid, threshold_fraction: float = 0.25, smooth_seconds: float = 0.18):
    """Return a sensor-specific active window around the largest Hilbert-response peak."""
    valid = np.asarray(valid, dtype=bool) & np.isfinite(envelope)
    if valid.sum() < 16:
        return np.nan, np.nan, np.asarray(envelope, dtype=float)

    env = np.asarray(envelope, dtype=float)
    dt = float(np.nanmedian(np.diff(time_grid))) if len(time_grid) > 1 else 0.01
    smooth_n = max(3, int(round(smooth_seconds / max(dt, EPS))))
    kernel = np.ones(smooth_n, dtype=float)
    filled = np.where(valid, env, 0.0)
    smooth = np.convolve(filled, kernel, mode="same") / np.maximum(
        np.convolve(valid.astype(float), kernel, mode="same"),
        1.0,
    )

    baseline = np.nanpercentile(smooth[valid], 20)
    peak = np.nanmax(smooth[valid])
    if not np.isfinite(peak) or peak <= baseline + EPS:
        valid_times = time_grid[valid]
        return float(valid_times[0]), float(valid_times[-1]), smooth

    threshold = baseline + threshold_fraction * (peak - baseline)
    above = valid & (smooth >= threshold)
    peak_idx = int(np.nanargmax(np.where(valid, smooth, -np.inf)))
    if not above[peak_idx]:
        valid_times = time_grid[valid]
        return float(valid_times[0]), float(valid_times[-1]), smooth

    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1

    return float(time_grid[left]), float(time_grid[right]), smooth


def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.zeros(x.shape, dtype=float)
    median = np.nanmedian(x[finite])
    mad = np.nanmedian(np.abs(x[finite] - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        q75, q25 = np.nanpercentile(x[finite], [75, 25])
        scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < EPS:
        scale = np.nanstd(x[finite])
    if not np.isfinite(scale) or scale < EPS:
        return np.zeros(x.shape, dtype=float)
    filled = x.copy()
    filled[~finite] = median
    return (filled - median) / scale


def normalized_abs_group_response(values, mask, aliases, quantity="STR", fs: float = FS):
    curves = []
    selected = []
    for idx, alias in enumerate(aliases):
        meta = parse_alias(alias)
        if meta["quantity"] != quantity:
            continue
        y = fill_signal(values[idx], mask[idx])
        if quantity == "STR":
            y = bandpass_strain(y, fs=fs)
        env = np.abs(hilbert(y))
        scale = np.nanpercentile(env[mask[idx]], 95) if mask[idx].any() else np.nan
        if np.isfinite(scale) and scale > EPS:
            curves.append(env / scale)
            selected.append(alias)
    if not curves:
        return selected, np.array([])
    return selected, np.nanmedian(np.vstack(curves), axis=0)


def normalized_group_response(values, mask, indices, active, use_bandpass=False, fs: float = FS):
    curves = []
    active = np.asarray(active, dtype=bool)
    for idx in indices:
        valid = mask[idx].astype(bool)
        y = fill_signal(values[idx], valid)
        if use_bandpass:
            y = bandpass_strain(y, fs=fs)
        response = np.abs(hilbert(y))
        scale_region = valid & active
        scale = np.nanpercentile(response[scale_region], 95) if scale_region.any() else np.nan
        if np.isfinite(scale) and scale > EPS:
            curves.append(response / scale)
    if not curves:
        return np.zeros(values.shape[1], dtype=float)
    out = np.nanmedian(np.vstack(curves), axis=0)
    peak = np.nanmax(out[active]) if np.isfinite(out[active]).any() else np.nanmax(out)
    return out / (peak + EPS)


def welch_psd(x, fs: float = FS, nperseg: int | None = None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.array([]), np.array([])
    if nperseg is None:
        n = len(x)
        nperseg = 512 if n >= 1024 else 256 if n >= 512 else 128 if n >= 256 else max(32, n // 2)
        nperseg = min(nperseg, n)
    return welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant", scaling="density")


def segment_fft_power(x, fs: float = FS, nperseg: int = 256, noverlap: int = 128):
    x = np.asarray(x, dtype=float)
    step = max(1, int(nperseg - noverlap))
    starts = np.arange(0, max(1, len(x) - nperseg + 1), step)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    powers = []
    for start in starts:
        segment = x[start : start + nperseg]
        if len(segment) < nperseg:
            continue
        segment = segment - np.nanmean(segment)
        window = np.hanning(nperseg)
        spectrum = np.fft.rfft(segment * window)
        powers.append((np.abs(spectrum) ** 2) / max(EPS, np.sum(window**2)))
    return freqs, np.asarray(powers), starts


def segment_fft_power_detailed(x, fs: float = FS):
    x = np.asarray(x, dtype=float)
    n = len(x)
    nperseg = 512 if n >= 1024 else 256 if n >= 512 else 128 if n >= 256 else max(32, n // 2)
    nperseg = min(nperseg, n)
    noverlap = nperseg // 2
    freqs, powers, starts = segment_fft_power(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if powers.size == 0:
        powers = np.zeros((1, len(freqs)), dtype=float)
    return freqs, powers, starts, nperseg


def band_energy_ratio(freqs, power, bands=BANDS):
    freqs = np.asarray(freqs)
    power = np.asarray(power, dtype=float)
    if power.ndim == 2:
        power = np.nanmean(power, axis=0)
    energies = []
    for _, low, high, _ in bands:
        band_mask = (freqs >= low) & (freqs < high)
        energies.append(float(np.trapz(power[band_mask], freqs[band_mask])) if band_mask.any() else 0.0)
    energies = np.asarray(energies, dtype=float)
    total = float(np.nansum(energies))
    return energies / total if total > EPS else np.zeros(len(bands), dtype=float)


def band_ratio_from_psd(freqs, psd, bands=BANDS):
    return band_energy_ratio(freqs, psd, bands=bands)


def band_energy_and_ratio_from_psd(freqs, psd, bands=BANDS):
    freqs = np.asarray(freqs)
    psd = np.asarray(psd, dtype=float)
    energies = []
    for _, low, high, _ in bands:
        band_mask = (freqs >= low) & (freqs < high)
        if band_mask.sum() >= 2:
            energies.append(float(np.trapz(psd[band_mask], freqs[band_mask])))
        elif band_mask.sum() == 1:
            energies.append(float(psd[band_mask][0]))
        else:
            energies.append(0.0)
    energies = np.asarray(energies, dtype=float)
    return energies, energies / (energies.sum() + EPS)


def choose_sensor_index(aliases, mask, time_grid, start_s, end_s, keyword, min_samples=128):
    active = (time_grid >= start_s) & (time_grid <= end_s)
    candidates = [
        idx
        for idx, alias in enumerate(aliases)
        if keyword in alias and np.sum(active & mask[idx].astype(bool)) >= min_samples
    ]
    if not candidates:
        candidates = [
            idx
            for idx, _ in enumerate(aliases)
            if np.sum(active & mask[idx].astype(bool)) >= min_samples
        ]
    if not candidates:
        raise ValueError(f"No sensor matched keyword={keyword!r} with at least {min_samples} active samples.")
    return candidates[0]


def active_waveform_from_bounds(values, mask, time_grid, sensor_idx, start_s, end_s):
    t, x = active_waveform(values, mask, time_grid, sensor_idx, start_s, end_s, detrend_signal=False)
    if t.size:
        t = t - t[0]
    return t, x


def welch_psd_for_sensor(values, mask, time_grid, sensor_idx, start_s, end_s, fs: float = FS):
    _, x = active_waveform_from_bounds(values, mask, time_grid, sensor_idx, start_s, end_s)
    return welch_psd(x, fs=fs)


def sorted_feature_rows(features_df):
    d = features_df.copy()
    d["axis"] = d["axis"].fillna("")
    d["_quantity"] = d["quantity"].map({"STR": 0, "ACC": 1}).fillna(9)
    d["_span"] = d["span"].map({"S1": 0, "S2": 1}).fillna(9)
    d["_side"] = d["side"].map({"DO": 0, "UP": 1}).fillna(9)
    d["_location"] = d["location"].map({"INF": 0, "SHE": 1, "SUP": 2, "INT": 3, "MID": 4}).fillna(9)
    d["_axis"] = d["axis"].map({"Y": 0, "Z": 1, "": 2}).fillna(9)
    return d.sort_values(["_quantity", "_side", "_span", "_location", "_axis"]).reset_index(drop=True)


def segment_starts(n_samples, nperseg=256, noverlap=128):
    step = max(1, nperseg - noverlap)
    return np.arange(0, max(1, n_samples - nperseg + 1), step)


METHOD_SENSOR_INDEX = {
    "S1_DO_INF_STR": 1,
    "S1_DO_SHE_STR": 2,
    "S1_DO_SUP_STR": 3,
    "S2_DO_INF_STR": 4,
    "S2_DO_SHE_STR": 5,
    "S2_DO_SUP_STR": 6,
    "S1_UP_INF_STR": 7,
    "S1_UP_SHE_STR": 8,
    "S1_UP_SUP_STR": 9,
    "S2_UP_INF_STR": 10,
    "S2_UP_SHE_STR": 11,
    "S2_UP_SUP_STR": 12,
    "S1_DO_INT_ACC_Y": 13,
    "S1_DO_INT_ACC_Z": 14,
    "S1_DO_MID_ACC_Y": 15,
    "S1_DO_MID_ACC_Z": 16,
    "S2_DO_INT_ACC_Y": 17,
    "S2_DO_INT_ACC_Z": 18,
    "S2_DO_MID_ACC_Y": 19,
    "S2_DO_MID_ACC_Z": 20,
    "S1_UP_INT_ACC_Z": 21,
    "S1_UP_MID_ACC_Z": 22,
    "S2_UP_INT_ACC_Z": 23,
    "S2_UP_MID_ACC_Z": 24,
}


def method_sensor_index(sensor_alias):
    return METHOD_SENSOR_INDEX.get(str(sensor_alias), np.nan)


def sensor_order_table(reference_df):
    span_order = {"S1": 0, "S2": 1}
    side_order = {"DO": 0, "UP": 1}
    quantity_order = {"STR": 0, "ACC": 1}
    str_location_order = {"INF": 0, "SHE": 1, "SUP": 2, "INT": 3, "MID": 4}
    acc_location_order = {"INT": 0, "MID": 1, "INF": 2, "SHE": 3, "SUP": 4}
    axis_order = {"": 0, "Y": 1, "Z": 2}

    order = reference_df.copy()
    order["axis"] = order["axis"].fillna("")
    order["span_rank"] = order["span"].map(span_order).fillna(99)
    order["side_rank"] = order["side"].map(side_order).fillna(99)
    order["quantity_rank"] = order["quantity"].map(quantity_order).fillna(99)
    order["location_rank"] = np.where(
        order["quantity"].eq("ACC"),
        order["location"].map(acc_location_order).fillna(99),
        order["location"].map(str_location_order).fillna(99),
    )
    order["axis_rank"] = order["axis"].map(axis_order).fillna(99)
    order["support_group"] = order[["quantity", "span", "side"]].fillna("").agg("|".join, axis=1)
    return order.sort_values(
        ["deck", "quantity_rank", "side_rank", "span_rank", "location_rank", "axis_rank", "sensor_alias"]
    ).reset_index(drop=True)


def robust_sensor_band_scale(features_df, reference_df, band_cols=BAND_COLS):
    merged = features_df.merge(reference_df[["sensor_id"] + band_cols], on="sensor_id", suffixes=("_event", "_ref"), how="left")
    rows = []
    for sensor_id, g in merged.groupby("sensor_id", sort=False):
        row = {"sensor_id": sensor_id}
        for col in band_cols:
            delta = g[f"{col}_event"].to_numpy(float) - g[f"{col}_ref"].to_numpy(float)
            med = np.nanmedian(delta)
            row[col] = 1.4826 * np.nanmedian(np.abs(delta - med))
        rows.append(row)
    scale = pd.DataFrame(rows).set_index("sensor_id")
    scale_floor = max(0.005, float(np.nanmedian(scale[band_cols].to_numpy())) * 0.15)
    scale[band_cols] = scale[band_cols].clip(lower=scale_floor)
    return scale.reset_index(), scale_floor


def set_sensor_band_axis(ax, sensor_labels=None, show_y=True, xlabel="Frequency band"):
    ax.set_xticks(np.arange(len(BAND_COLS)))
    ax.set_xticklabels(BAND_LABELS, rotation=0, ha="center", fontsize=TICK_LABEL_FONTSIZE)
    if sensor_labels is not None:
        ax.set_yticks(np.arange(len(sensor_labels)))
        ax.set_yticklabels(sensor_labels if show_y else [], fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=9)
    ax.set_xticks(np.arange(-0.5, len(BAND_COLS), 1), minor=True)
    if sensor_labels is not None:
        ax.set_yticks(np.arange(-0.5, len(sensor_labels), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.35, alpha=0.58)
    ax.tick_params(which="minor", bottom=False, left=False)


def sensor_row_groups_from_aliases(sensor_labels):
    labels = [str(label) for label in sensor_labels]
    if not labels:
        return []

    groups = []
    start = 0
    current = None
    for idx, alias in enumerate(labels):
        meta = parse_alias(alias)
        key = (meta["span"], meta["quantity"])
        if current is None:
            current = key
            continue
        if key != current:
            groups.append(
                {
                    "label": f"{current[0]} {current[1]}".strip(),
                    "start": start,
                    "end": idx,
                }
            )
            start = idx
            current = key

    groups.append(
        {
            "label": f"{current[0]} {current[1]}".strip(),
            "start": start,
            "end": len(labels),
        }
    )
    return groups


def set_sensor_index_axis(
    ax,
    sensor_labels,
    show_y=True,
    xlabel="Frequency band",
    band_ticklabels=None,
    show_group_labels=False,
    row_line_kwargs=None,
    group_label_kwargs=None,
    xlabel_fontsize=AXIS_LABEL_FONTSIZE,
    xtick_fontsize=TICK_LABEL_FONTSIZE,
    ytick_fontsize=TICK_LABEL_FONTSIZE,
):
    labels = [str(label) for label in sensor_labels]
    n_rows = len(labels)
    ticklabels = BAND_LABELS if band_ticklabels is None else list(band_ticklabels)
    ax.set_xticks(np.arange(len(BAND_COLS)))
    ax.set_xticklabels(ticklabels, rotation=0, ha="center", fontsize=xtick_fontsize)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(i + 1) for i in range(n_rows)] if show_y else [], fontsize=ytick_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=9)
    ax.set_xticks(np.arange(-0.5, len(BAND_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="0.70", linewidth=0.45, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    groups = sensor_row_groups_from_aliases(labels)
    if row_line_kwargs is None:
        row_line_kwargs = dict(color="black", linewidth=1.0, alpha=0.95)
    if group_label_kwargs is None:
        group_label_kwargs = dict(fontsize=11, fontweight="bold", color="black")

    for group in groups[1:]:
        ax.axhline(group["start"] - 0.5, **row_line_kwargs)

    # Group labels intentionally suppressed in the current notebook set.
    if show_group_labels:
        for group in groups:
            center = (group["start"] + group["end"] - 1) / 2.0
            ax.text(
                -0.06,
                center,
                group["label"],
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                clip_on=False,
                **group_label_kwargs,
            )


def format_sensor_index_legend(sensor_labels, ncols=8):
    labels = [str(label) for label in sensor_labels]
    if not labels:
        return ""

    entries = [f"{idx + 1:02d} {label}" for idx, label in enumerate(labels)]
    width = max(len(entry) for entry in entries) + 2
    lines = []
    for start in range(0, len(entries), ncols):
        row = entries[start : start + ncols]
        lines.append("".join(entry.ljust(width) for entry in row).rstrip())
    return "\n".join(lines)


def add_sensor_index_legend(
    fig,
    sensor_labels,
    ncols=8,
    y=0.02,
    fontsize=SENSOR_INDEX_LEGEND_FONTSIZE,
    title_fontsize=SENSOR_INDEX_LEGEND_TITLE_FONTSIZE,
):
    legend_text = format_sensor_index_legend(sensor_labels, ncols=ncols)
    if not legend_text:
        return None
    body = fig.text(
        0.5,
        y,
        legend_text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        family="monospace",
        color="0.15",
        linespacing=1.28,
    )
    title = fig.text(
        0.5,
        y + 0.080,
        "Sensor index legend",
        ha="center",
        va="bottom",
        fontsize=title_fontsize,
        family="monospace",
        color="0.15",
    )
    return title, body
