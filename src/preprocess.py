"""
Build the preprocessing transformer for NSL - KDD features.

This module produces a single sklearn ColumnTransformer that:
- One-hot encodes the 3 categorical columns, tolerating categories that
appear in test but not train (and vice versa)

- scales the 38 numeric columns to zero / unit variance

It does NOT fit anything here. The transformer is returned unfitted 
so it can be fit *inside* a Pipeline on training data only - this is what 
prevents test-set statistics from leaking into the model. 

Run directly to sanity-check the output shape:

    python -m src.process
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features import get_feature_columns

def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the given column groups.

    Numeric - StandardScaler (mean 0, variance 1).
    Categorical -> OneHotEncoder with handle_unknown='ignore'.

    handle_unknown='ignore' is the critical setting: at transform time, a 
    category never seen during fit becomes an all-zero vector instead of 
    raising. This is exactly what we need because the test set contains
    services absent from train, and vice versa.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False, # dense output keeps downstream code simple
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop", #anything not listed is dropped, not passed through
    )
    return preprocessor

def main() -> int:
    from src.data_loader import load_train_test
    from src.features import split_xy

    train, test = load_train_test()
    X_train, y_train = split_xy(train, target="binary")
    X_test, y_test = split_xy(test, target="binary")

    numeric, categorical = get_feature_columns(train)
    preprocessor = build_preprocessor(numeric, categorical)

    # Fit on TRAIN ONLY, then transform both. This mirrors how the real 
    #Pipeline will behave and proves the test set's unseen categories
    #don't break transform. 
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print(f"Raw    : X_train{X_train.shape}, X_test{X_test.shape}")
    print(f"Encoded: X_train{X_train_t.shape}, X_test{X_test_t.shape}")

    # The encoded column counts MUST match between train and test, even 
    # though test has fewer unique services. handle_unknown="ignore" is
    # what guarantees this.
    assert X_train_t.shape[1] == X_test_t.shape[1], (
        "Encoded feature counts differ - categorical handling is broken"
    )

    # Report how the width breaks down.
    n_numeric = len(numeric)
    n_onehot = X_train_t.shape[1] - n_numeric
    print(f"  numeric columns : {n_numeric}")
    print(f"  one-hot columns : {n_onehot}")
    print(f"  total features  : {X_train_t.shape[1]}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())