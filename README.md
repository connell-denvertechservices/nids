# NIDS — ML-based Network Intrusion Detection

Trains classifiers on the NSL-KDD dataset to distinguish normal network
connections from attacks, with security-relevant evaluation (precision,
recall, false-positive rate).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

NSL-KDD (`KDDTrain+.txt`, `KDDTest+.txt`) is downloaded into `data/`
by `src/data_loader.py`. Files are gitignored.

## Usage

```bash
python -m src.data_loader      # fetch + verify dataset
python -m src.train            # fit models, save to models/
python -m src.evaluate         # metrics on the test set
python -m src.predict          # score a sample record
```

## Layout

- `src/config.py` — paths, column names, constants
- `src/data_loader.py` — download + load into DataFrame
- `src/features.py` — labels + feature grouping
- `src/preprocess.py` — sklearn ColumnTransformer
- `src/train.py` — fit + persist pipelines
- `src/evaluate.py` — metrics + confusion matrix
- `src/predict.py` — load model + score new records

## Models

Logistic Regression (interpretable baseline) and Random Forest
(stronger, exposes feature importances).