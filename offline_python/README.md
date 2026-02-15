# offline_python

Pure Python core for adaptive event encoding experiments.

## Includes

- `adaptive_core/transforms.py`: event-driven DCT/DTFT/DWT encoders.
- `adaptive_core/selection.py`: coefficient selection methods.
- `adaptive_core/density.py`: density score and transform choice.
- `adaptive_core/calibration.py`: threshold calibration from NPZ datasets.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Calibrate thresholds

```bash
python scripts/calibrate_thresholds.py --input ../../gt
```

To write thresholds to a portable JSON (copy to other environments):

```bash
python scripts/calibrate_thresholds.py \
  --input ../../gt \
  --output ./adaptive_core/config/adaptive_thresholds.json \
  --max-files 30 \
  --samples-per-file 3 \
  --sparse-percentile 25 \
  --dense-percentile 75
```

## Select transform for a file

```bash
python scripts/choose_transform_from_npz.py --file ../../data_example/00000.npz
```

## Encode + inverse decode for a file

```bash
python scripts/choose_transform_from_npz.py \
  --file ../../data_example/00000.npz \
  --decode \
  --t-resolution 0.02 \
  --threshold-method adaptive \
  --polarity-method sign
```

## Real GT Dataset: Full Offline Pipeline

Use this when you have a real dataset folder with many `.npz` files (for example `../../gt`).

```bash
# 1) Calibrate thresholds from real dataset
python scripts/calibrate_thresholds.py \
  --input ../../gt \
  --output ./adaptive_core/config/adaptive_thresholds.json

# 2) Quick check on one file (selection + encode + decode)
python scripts/choose_transform_from_npz.py \
  --file ../../data_example/00000.npz \
  --thresholds ./adaptive_core/config/adaptive_thresholds.json \
  --decode \
  --M 16 \
  --keep-ratio 0.3 \
  --t-resolution 0.02
```

## Fine-Tune for a New Dataset (Generates TXT + JSON)

This script calibrates thresholds and sweeps core parameters to recommend a portable setup.

```bash
python scripts/tune_dataset_params.py \
  --input ../../gt \
  --thresholds-output ./adaptive_core/config/adaptive_thresholds.json \
  --report-output ./tuning_report.txt \
  --params-output ./recommended_params.json \
  --cal-max-files 30 \
  --cal-samples-per-file 3 \
  --eval-max-files 8 \
  --M-list 8,16 \
  --keep-ratio-list 0.2,0.3,0.5 \
  --t-resolution-list 0.01,0.02,0.05 \
  --threshold-mode-list adaptive,percentile \
  --percentile-value-list 95
```

Outputs:

- `adaptive_core/config/adaptive_thresholds.json`: calibrated thresholds (portable).
- `tuning_report.txt`: human-readable summary and recommended parameters.
- `recommended_params.json`: machine-readable recommended parameters.

You can copy `adaptive_thresholds.json` and `recommended_params.json` to another environment and reuse them directly.
