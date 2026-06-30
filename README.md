# EWSHM 2026 Bridge Health Monitoring Challenge

Core preprocessing scripts and analysis notebooks for event-level bridge response processing, Welch PSD spectral response vectors, and ACC-classwise input-normalized STR health scoring.

## Repository structure

```text
preprocess_aquinas_hdf5.py              # Raw AQUINAS JSON to set/deck HDF5 preprocessing
preprocess_aquinas_event_hdf5.py        # Event-level HDF5 preprocessing
notebooks/1_Broad_band_spectral_partitioning.ipynb
notebooks/2_Welch_PSD_Calculation.ipynb
notebooks/3_ACC_Classwise_Input_Normalized_STR_Health_Scoring.ipynb
utils/                                  # Shared signal-processing helpers
KYM/src/aquinas/                         # Dataset table loading helpers used by preprocessing
docs/                                   # Sensor labels and spectral class reference notes
```

## Data

Raw contest data and generated HDF5/table outputs are intentionally not included. Place the AQUINAS dataset at `EWSHM-contest-data/` or set `AQUINAS_DATASET_PATH`. Generated outputs are ignored by git.

## Main workflow

1. Build aligned/event-level HDF5 files using the preprocessing scripts.
2. Run Notebook 1 for active-window extraction and broad-band response visualization.
3. Run Notebook 2 for Welch PSD class energy and STR/ACC spectral response vectors.
4. Run Notebook 3 for ACC-classwise input-normalized STR inter-sensor similarity and health scoring.
