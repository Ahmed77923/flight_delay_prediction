# Data

## Purpose

This page documents the data contract used by training and inference without inventing a dataset source that is not stored in the repository.

## Architecture / Concept

```mermaid
flowchart LR
    CSV[Monthly CSV files] --> Load[load_data]
    Load --> Clean[clean_data]
    Clean --> Split[Chronological split]
    Split --> Features[Feature engineering]
```

## Implementation

`load_data(data_dir)` reads sorted files matching `data_dir/*.csv`, parses each with pandas, and concatenates them. The current checkout has no `data/` directory, so no training dataset is available locally. The EDA notebook expects files such as `../data/month_1.csv`; that is evidence of the expected layout, not a committed dataset.

Training cleaning requires `CANCELLED`, `DIVERTED`, `CRS_ELAPSED_TIME`, and `ARR_DELAY`. It keeps rows with zero cancellation/diversion, positive scheduled duration, and non-null target, then removes target values outside the 1.5 IQR bounds. Feature construction additionally requires `FL_DATE`, `CRS_DEP_TIME`, `CRS_ARR_TIME`, `DISTANCE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, and `DEST`.

The target is numeric `ARR_DELAY` in minutes. Inference requests do not include the target.

## Data types and validation

- `FL_DATE`: parseable date/datetime.
- `CRS_DEP_TIME` and `CRS_ARR_TIME`: integer `HHMM` values.
- `CRS_ELAPSED_TIME` and `DISTANCE`: numeric values.
- Carrier, origin, and destination: strings.
- Training-only flags and target: numeric values used by cleaning.

The trainer calls a chronological `split_data` function, but `src/data/split_data.py` is missing from the current checkout. The intended split behavior is described in the existing README as earliest 80% versus latest 20%, but that cannot be verified from the absent module and should be treated as unresolved until restored.

## Configuration

`Config.DATA.DATA_PATH` points to `<project>/data`. `TRAINING_YEAR` is defined but is not consumed by the active trainer flow.

## Limitations

Dataset provenance, schema versioning, data validation reports, and a data download script are not implemented. Do not treat notebook sample output as a production dataset contract.
