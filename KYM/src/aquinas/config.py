import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KYM_ROOT = PROJECT_ROOT / "KYM"
NOTEBOOK_ROOT = KYM_ROOT / "notebooks"


def _resolve_output_root() -> Path:
    env_path = os.environ.get("AQUINAS_OUTPUT_ROOT")
    if env_path:
        return Path(env_path).expanduser()
    return KYM_ROOT / "outputs"


def _resolve_dataset_path() -> Path:
    env_path = os.environ.get("AQUINAS_DATASET_PATH")
    if env_path:
        return Path(env_path).expanduser()

    candidates = (
        PROJECT_ROOT / "EWSHM-contest-data",
        PROJECT_ROOT / "data" / "EWSHM-contest-data",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


OUTPUT_ROOT = _resolve_output_root()
DEFAULT_DATASET_PATH = _resolve_dataset_path()

BASELINE_SET = "AQUINAS_SET1_2022_07"
BASELINE_QUANTILE = 0.995
EVENT_MATCH_TOLERANCE_SEC = 1.5

DECKS = ("OLD", "NEW")
SET_ORDER = (
    "AQUINAS_SET1_2022_07",
    "AQUINAS_SET2_2023_04",
    "AQUINAS_SET3_2023_08",
    "AQUINAS_SET4_2024_01",
    "AQUINAS_SET5_2024_06",
)

METHOD_COLORS = {
    "physics_xgb_mahalanobis": "#0b6e4f",
    "pca_t2_spe": "#4c6ef5",
    "localized_pcd": "#1f7a8c",
    "gaussian_process_warning": "#c77d00",
    "one_class_svm": "#6a4c93",
    "isolation_forest": "#006d77",
    "autoencoder": "#ae2012",
    "pca_autoencoder_ensemble": "#9d4edd",
    "graph_consistency": "#3a5a40",
    "ewma_cusum_trend": "#9c6644",
}
