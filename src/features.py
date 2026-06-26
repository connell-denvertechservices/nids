"""
Feature engineering and label construction for NSL-KDD.

This module does NOT scale or encode — that is preprocess.py's job.
Here we:
  1. Build the prediction targets from the raw `label` column.
  2. Group feature columns into numeric vs. categorical for downstream
     transformers.
  3. Split a DataFrame into X (features) and y (target).

Run directly to inspect the label distributions:
    python -m src.features

"""

from __future__ import annotations

import pandas as pd
from src import config

def make_binary_label(df: pd.DataFrame) -> pd.Series:
    """
    Map the raw label to 0 (normal) / 1 (attack).

    Every row whose label is the string 'normal' is benign (0); everything 
    else is an attack (1). This is the primary detection target.
    """
    return (df["label"] != "normal").astype(int)

def make_category_label(df: pd.DataFrame) -> pd.Series:
    """
    Map the raw label to one of five classes.

    'normal' stays normal; named attacks map to their family via
    config.ATTACK_CATEGORY (dos / probe / r2l / u2r). Any attack name not
    present in the map becomes 'unknown' — this is what catches the novel
    attack types that appear in the test set but not in train.
    """
    def _map(label: str) -> str:
        if label == "normal":
            return "normal"
        return config.ATTACK_CATEGORY.get(label, "unknown")

    return df["label"].apply(_map)

def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numeric_columns, categorical_columns) for the model.

    Categoricals come from config. Numerics are everything else, minus the
    label/difficulty columns that are never model inputs. Derived from the
    actual DataFrame so a schema drift surfaces here instead of mid-training.
    """
    exclude = set(config.CATEGORICAL_COLUMNS) | {"label"} | set(config.DROP_COLUMNS)
    numeric = [c for c in df.columns if c not in exclude]
    categorical = list(config.CATEGORICAL_COLUMNS)
    return numeric, categorical

def split_xy(
    df: pd.DataFrame, target: str = "binary"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features X and target y.

    target='binary'   -> 0/1 normal-vs-attack (default)
    target='category' -> 5-class normal/dos/probe/r2l/u2r/unknown

    X contains only model-input columns (categoricals + numerics); the raw
    label and difficulty are dropped so they can't leak into the model.
    """
    if target == "binary":
        y = make_binary_label(df)
    elif target == "category":
        y = make_category_label(df)
    else:
        raise ValueError(f"target must be 'binary' or category, got {target!r}")

    numeric, categorical = get_feature_columns(df)
    X = df[categorical + numeric].copy()
    return X, y

def main() -> int:
    from src.data_loader import load_train_test

    train, test = load_train_test()

    print("Binary label distribution (train):")
    print(make_binary_label(train).value_counts().sort_index())

    print("\nCategory label distribution (train):")
    print(make_category_label(train).value_counts())

    print("\nCategory label distribution (test):")
    print(make_category_label(test).value_counts())

    numeric, categorical = get_feature_columns(train)
    print(f"\nFeature grouping:")
    print(f"  categorical ({len(categorical)}): {categorical}")
    print(f"  numeric     ({len(numeric)}): {len(numeric)} columns")

    X, y = split_xy(train, target="binary")
    print(f"\nsplit_xy -> X{X.shape}, y{y.shape}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())