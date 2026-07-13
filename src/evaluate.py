"""Patient-grouped CV, overfit check, normal-vs-abnormal (referral) eval."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (f1_score, classification_report,
                             recall_score, accuracy_score)

from config import PROCESSED, LABELS


def build_model():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=42))


def main() -> None:
    d = np.load(PROCESSED / "features.npz", allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"].astype(str)
    patient = pd.read_csv(PROCESSED / "cycles.csv")["patient"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_model(), X, y, groups=patient,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[CV] 4-class macro-F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
    print("     per fold:", np.round(scores, 3))

    tr, te = split == "train", split == "test"
    m = build_model().fit(X[tr], y[tr])
    f1_tr = f1_score(y[tr], m.predict(X[tr]), average="macro")
    f1_te = f1_score(y[te], m.predict(X[te]), average="macro")
    print(f"\n[Overfit] train={f1_tr:.3f} test={f1_te:.3f} gap={f1_tr-f1_te:.3f}")

    yb = (y > 0).astype(int)
    mb = build_model().fit(X[tr], yb[tr])
    pb = mb.predict(X[te])
    print(f"\n[Binary normal-vs-abnormal]  accuracy={accuracy_score(yb[te],pb):.3f}"
          f"  F1={f1_score(yb[te],pb):.3f}")
    print(f"  sensitivity = {recall_score(yb[te],pb,pos_label=1):.3f}")
    print(f"  specificity = {recall_score(yb[te],pb,pos_label=0):.3f}")
    print("\n[4-class report]")
    print(classification_report(y[te], m.predict(X[te]), target_names=LABELS, digits=3))


if __name__ == "__main__":
    main()
