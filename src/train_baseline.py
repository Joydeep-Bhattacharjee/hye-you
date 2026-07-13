"""Train RandomForest and SVM on MFCC features; keep the better one."""
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)

from config import PROCESSED, MODELS, RESULTS, LABELS


def main() -> None:
    d = np.load(PROCESSED / "features.npz", allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"].astype(str)
    tr, te = split == "train", split == "test"
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"train {Xtr.shape[0]} | test {Xte.shape[0]}")

    candidates = {
        "randomforest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=42)),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=10, gamma="scale", class_weight="balanced",
                probability=True, random_state=42)),
    }

    best_name, best_f1, best_model = None, -1.0, None
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        f1 = f1_score(yte, pred, average="macro")
        acc = accuracy_score(yte, pred)
        print(f"\n=== {name} ===  accuracy={acc:.3f}  macro-F1={f1:.3f}")
        print(classification_report(yte, pred, target_names=LABELS, digits=3))
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, model

    pred = best_model.predict(Xte)
    cm = confusion_matrix(yte, pred)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues", values_format="d")
    plt.title(f"Baseline ({best_name}) - test set")
    plt.tight_layout()
    plt.savefig(RESULTS / "baseline_confusion.png", dpi=150)

    joblib.dump({"model": best_model, "labels": LABELS, "name": best_name},
                MODELS / "baseline.joblib")
    print(f"\nBEST = {best_name} (macro-F1 {best_f1:.3f})")
    print(f"Saved -> {MODELS / 'baseline.joblib'}")


if __name__ == "__main__":
    main()
