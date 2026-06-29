"""
Train and persist intrusion-detection models.

Each model is a Pipeline = [preprocessor -> classifier], so the fitted
artifact carries its own preprocessing. This means predict.py can load
one file and feed it raw NSL - KDD rows - no seperate scaler to manage, and no
chance of a train/serve preprocessing mismatch. 

Run directly to fit both models and save them 

    python -m src.train

"""

from __future__ import annotations

import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_train_test
from src.features import get_feature_columns, split_xy
from src.preprocess import build_preprocessor

def build_models() -> dict [str, Pipeline]:
    """Construct one Pipeiline per classifier, sharing a fresh preprocessor.

    Each Pipeline gets its OWN preprocessor instance. They must not share a 
    fitted object - that's why build-preprocessor is called inside the loop
    conceptually; here we build two seperate ones explicitly.
    """

    #We need the column groups to construct the preprocessor. Any loaded
    #frame would do; the grouping is schema-based, not data-based.

    train, _ = load_train_test()
    numeric, categorical = get_feature_columns(train)
    log_reg = Pipeline(steps=[
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )),
    ])

    random_forest = Pipeline(steps=[
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return {"logistic_regression": log_reg, "random_forest": random_forest}

def train_all(target: str = "binary") -> dict[str, Pipeline]:
    """Fit every model on the training split and persist to models/.

    Returns the fitted pipelines so a caller (or evaluate.py) can use them
    in-memory without reloading.
    """
    train, _ = load_train_test()
    X_train, y_train = split_xy(train, target=target)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fitted: dict[str, Pipeline] = {}
    for name, pipe in build_models().items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        # Training-set accuracy - a sanity figure, NOT a performance claim.
        # The real number comes from evaluate.py on the held-out test set.

        train_acc = pipe.score(X_train, y_train)
        print(f"  fit in {elapsed:.1f}s | train accuracy {train_acc:.4f}")

        out_path = config.MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  saved -> {out_path.name} ({out_path.stat().st_size:,} bytes)")

        fitted[name] = pipe
    return fitted

def main() -> int:
    fitted = train_all(target="binary")
    print(f"\nDone. {len(fitted)} models trained and saved to {config.MODELS_DIR}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
