from __future__ import annotations

import sys 
import urllib.request
from pathlib import Path

import pandas as pd
from src import config

def _download(url: str, dest: Path) -> None:
    """Fetch a single file to 'dest' if it is not already present."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [get ] {dest.name} <- {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc: # network, DNS, 404, etc.
        raise RuntimeError(
            f"Failed to download {url}\n"
            f"  reason: {exc}\n"
            f"  If WSL egress is restricted, download manually and place "
            f"the file at {dest}"
        ) from exc
    print(f"  [ok  ] saved {dest.name} ({dest.stat().st_size:,} bytes)")

def download_dataset() -> None:
    """Ensure both NSL-KDD files exist locally."""
    print("Downloading NSL-KDD dataset...")
    _download(config.TRAIN_URL, config.TRAIN_FILE)
    _download(config.TEST_URL, config.TEST_FILE)

def load_frame(path: Path) -> pd.DataFrame:
    """Load one NSL-KDD file into a DataFrame with names columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.data_loader' first."
        )
    df = pd.read_csv(path, header=None, names=config.COLUMN_NAMES)
    return df

def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both splits, downloading first if needed."""
    download_dataset()
    train = load_frame(config.TRAIN_FILE)
    test = load_frame(config.TEST_FILE)
    return train, test

def _verify(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}:")
    print(f"  shape         : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  null values   : {int(df.isnull().sum().sum())}")
    print(f"  duplicate rows: {int(df.duplicated().sum())}")

    # Sanity: the three categorical columns should be strings (object dtype)
    for col in config.CATEGORICAL_COLUMNS:
        kind = df[col].dtype
        print(f"  {col:<13}: {df[col].nunique()} unique values ({kind})")

    n_normal = (df["label"] == "normal").sum()
    n_attack = (df["label"] != "normal").sum()

    pct_attack = 100 *n_attack / len(df)
    print(f"  normal        : {n_normal:,}")
    print(f"  attack        : {n_attack:,} ({pct_attack:.1f}% of rows)")

def main() -> int:
    train, test = load_train_test()
    _verify(train, "KDDTrain+")
    _verify(test, "KDDTest+")

    #Show a couple of rows so the schema is visibly correct
    print("\nFirst training row (transposed):")
    print(train.iloc[0])
    return 0

if __name__ == "__main__":
    sys.exit(main()) 