"""
Evaluate trained intrusion detection models on the held-out test set.

Loads each persisted Pipeline and reports the metrics that matter for a 
detector - not just accuracy, but precision, recall, and the false-positive
rate, plus a confusion matrix. The test set carries a deliberate distribution
shift (esp. R2L), so expect test numbers well below train. 

Run directly:
    python -m src.evaluate
"""
from __future__ import annotations

import joblib
from sklearn.metrics import ( 
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src import config
from src.data_loader import load_train_test
from src.features import split_xy

def load_models() -> dict[str, object]:
    """
    Load every persisted pipeline from models/.

    Fails loudly with an actionable message if a model is missing, rather
    than silently evaluating fewer models than expected.
    """
    names = ["logistic_regression", "random_forest"]
    models: dict[str, object] = {}
    for name in names:
        path = config.MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python -m src.train' first."
            )
        models[name] = joblib.load(path)
    return models

def evaluate_one(name: str, model, X_test, y_test) -> None:
    """Print a full metric report for a single model on the test set."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # pos_label1 -> 'attack' is the positive class. For a detector, recall
    # on attacks = the fraction of real attacks we caught (miss rate = 1-recall).
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Confusion matrix layout for binary [0,1]:
    #            predicted
    #            normal  attack
    # actual normal  TN     FP
    # actual attack  FN     TP
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    # False - positive rate = FP / (FP + TN). For an IDS this is the alarm-
    # fatigue number: how often benign traffic gets flagged. A high FPR
    # makes a detector unusable in practice even at high accuracy

    fpr = fp / (fp + tn) if (fp+tn) else 0.0

    print(f"\n{'='*56}")
    print(f"  {name}")
    print(f"{'='*56}")
    print(f"  accuracy            : {acc:.4f}")
    print(f"  precision (attack)  : {prec:.4f}")
    print(f"  recall    (attack)  : {rec:.4f}   <- fraction of attacks caught")
    print(f"  f1        (attack)  : {f1:.4f}")
    print(f"  false-positive rate : {fpr:.4f}   <- benign flagged as attack")
    print(f"\n  confusion matrix:")
    print(f"                  pred_normal  pred_attack")
    print(f"    act_normal    {tn:>11,}  {fp:>11,}")
    print(f"    act_attack    {fn:>11,}  {tp:>11,}")
    print(f"\n  per-class detail:")
    print(classification_report(
        y_test, y_pred, target_names=["normal", "attack"], digits=4
    ))

def main() -> int:
    _, test = load_train_test()
    X_test, y_test = split_xy(test, target='binary')

    models = load_models()
    print(f"Evaluating {len(models)} models on {len(y_test):,} test rows "
        f"({(y_test == 1).sum():,} attacks, {(y_test == 0).sum():,} normal)")
    
    for name, model in models.items():
        evaluate_one(name, model, X_test, y_test)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())