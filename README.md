# EWSHM 2026 Subject 1

## Event-Level Bridge Health Scoring Using Spectral Response Similarity

[![Challenge](https://img.shields.io/badge/EWSHM_2026-Young_Researcher_Challenge-1f6feb)](https://ewshm2026.com/ewshm-challenge)
[![Subject](https://img.shields.io/badge/Subject_1-Data--driven_trends_and_anomalies-0f766e)](https://ewshm2026.com/ewshm-challenge)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](https://www.python.org/)

This repository contains the core preprocessing code and analysis notebooks for **Subject 1: Data-driven detection of trends and anomalies in civil engineering applications** in the **12th European Workshop on Structural Health Monitoring (EWSHM 2026) Young Researcher Challenge**.

The goal is to estimate event-level bridge health trends from traffic-induced sensor responses without relying on predefined damage labels, reference states, or long-term baseline records. The proposed workflow converts raw bridge responses into spectral response vectors, compares structurally comparable sensors through inter-sensor similarity, and summarizes the result as STR sensor health scores.

## Challenge Context

The EWSHM Challenge encourages young researchers to develop and present practical SHM methods to an international community from academia, industry, authorities, and research organizations.

- **Conference:** 12th EWSHM, July 7-10, 2026
- **Challenge:** EWSHM 2026 Young Researcher Challenge
- **Selected subject:** Subject 1, data-driven trend and anomaly detection in civil engineering
- **Official page:** <https://ewshm2026.com/ewshm-challenge>

The submitted work focuses on traffic-induced bridge monitoring data and aims to provide a transparent signal-processing pipeline rather than a black-box supervised classifier.

## Key Contributions

- Event-level preprocessing pipeline for aligned STR and ACC bridge response records.
- Broad-band spectral response characterization using Welch PSD energy classes.
- Sensor-network health scoring using ACC-normalized STR response similarity.
- Consistent method-level sensor indexing for reproducible figures and tables.

## Challenge Alignment

| Challenge criterion | Repository focus |
|---|---|
| Scientific approach and presentation clarity | Transparent signal-processing workflow from raw events to spectral response vectors and health scores. |
| Result quality and code quality | Reproducible notebooks, preprocessing scripts, shared utilities, and documented sensor-index conventions. |
| Innovation and expected impact | Reference-light event scoring based on inter-sensor spectral response consistency under unknown traffic excitation. |

## Core Idea

Measured bridge events are represented by frequency-class response vectors. STR responses are corrected using local ACC input information, and the corrected STR vectors are compared pairwise to build an inter-sensor spectral response similarity map.

The health score is derived from two complementary consistency measures:

1. **Same-group consistency:** response similarity among STR sensors with comparable structural roles.
2. **Opposite-span consistency:** response similarity across opposite spans, used as a structural redundancy check.

## Processing Pipeline

```mermaid
flowchart LR
    A[Raw contest data] --> B[HDF5 preprocessing]
    B --> C[Event-level aligned STR and ACC responses]
    C --> D[Active-window extraction]
    D --> E[Welch PSD calculation]
    E --> F[STR and ACC spectral response vectors]
    F --> G[Class-weighted response vectors]
    G --> H[ACC-normalized STR response vectors]
    H --> I[Inter-sensor spectral response similarity map]
    I --> J[STR sensor health score]
```

## Repository Layout

```text
.
|-- preprocess_aquinas_hdf5.py
|   Raw contest data to set/deck-level HDF5 preprocessing.
|
|-- preprocess_aquinas_event_hdf5.py
|   Event-level HDF5 preprocessing and aligned sensor response export.
|
|-- notebooks/
|   |-- 1_Broad_band_spectral_partitioning.ipynb
|   |   Defines the broad spectral response classes used in the study.
|   |
|   |-- 2_Welch_PSD_Calculation.ipynb
|   |   Computes Welch PSD energy and STR/ACC spectral response vectors.
|   |
|   `-- 3_ACC_Classwise_Input_Normalized_STR_Health_Scoring.ipynb
|       Builds ACC-normalized STR similarity maps and health scores.
|
|-- utils/
|   Shared plotting, metadata, sensor-index, PSD, and trend-table helpers.
|
|-- KYM/src/aquinas/
|   Lightweight dataset table loaders used by the preprocessing scripts.
|
`-- docs/
    Sensor-index mapping and spectral response class reference notes.
```

## Notebook Sequence

| Step | Notebook | Purpose |
|---:|---|---|
| 1 | `1_Broad_band_spectral_partitioning.ipynb` | Defines the PSD frequency classes and visualizes STR/ACC event responses. |
| 2 | `2_Welch_PSD_Calculation.ipynb` | Converts active-window signals into class-wise spectral response vectors. |
| 3 | `3_ACC_Classwise_Input_Normalized_STR_Health_Scoring.ipynb` | Calculates ACC-normalized STR similarity maps and sensor health trends. |

## Sensor Index Convention

The analysis uses a method-level sensor index so that figures and tables are consistent across notebooks.

- `1-12`: STR sensors
- `13-24`: ACC sensors
- STR groups: `STR DO` and `STR UP`
- ACC groups: local input correction groups based on span and side

The full mapping is documented in:

```text
docs/sensor_index_label_mapping_kr.yaml
```

## Spectral Response Classes

The PSD response vectors are summarized into five broad frequency classes:

| Class | Frequency range | Interpretation |
|---|---:|---|
| Quasi-static | 0.05-0.80 Hz | Vehicle-induced slow deflection response |
| Low dynamic | 0.80-2.00 Hz | Slow bending with low-frequency vibration |
| Mid dynamic | 2.00-5.00 Hz | Dominant vehicle-induced bridge vibration |
| High dynamic | 5.00-12.00 Hz | Higher-frequency vibration with axle/road effects |
| Noise-sensitive | 12.00-25.00 Hz | Impact, road-surface vibration, and sensor-noise-sensitive range |

Reference notes for these classes are kept in:

```text
docs/spectral_response_class_reference_papers.yaml
```

## Quick Start

Create an environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Point the code to the raw contest dataset:

```bash
export AQUINAS_DATASET_PATH=/path/to/EWSHM-contest-data
```

Run preprocessing:

```bash
python preprocess_aquinas_hdf5.py
python preprocess_aquinas_event_hdf5.py
```

Then open the notebooks in order:

```bash
jupyter lab notebooks/
```

## Data Policy

Raw contest data, generated HDF5 files, figures, and intermediate analysis tables are intentionally not included in this repository.

Ignored local artifacts include:

- `EWSHM-contest-data/`
- `EWSHM_dataset_preprocessed/`
- `EWSHM_dataset_preprocessed_event_level/`
- `AutoResearch_generated_method/`
- `KYM/outputs/`

This keeps the repository focused on reproducible code and documentation while avoiding accidental upload of large contest data or generated outputs.

## Method Summary

1. Extract event-level active response windows from aligned STR and ACC signals.
2. Compute Welch PSD energy and construct class-wise STR/ACC spectral response vectors.
3. Normalize STR response vectors using local ACC input information.
4. Build inter-sensor spectral response similarity maps.
5. Estimate STR health scores from same-group and opposite-span response consistency.

## Notes

This repository is organized for the EWSHM 2026 Challenge submission. It is not a general-purpose bridge SHM package, and the frequency classes and sensor grouping rules are tailored to the provided contest dataset and sensor layout.
