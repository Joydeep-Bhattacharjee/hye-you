"""Optional RandomForest grid search (patient-grouped)."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from config import PROCESSED

d = np.load(PROCESSED / "features.npz", allow_pickle=True)
X, y = d["X"], d["y"]
groups = pd.read_csv(PROCESSED / "cycles.csv")["patient"].to_numpy()
pipe = make_pipeline(StandardScaler(),
                     RandomForestClassifier(class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=42))
grid = {"randomforestclassifier__n_estimators": [300, 600],
        "randomforestclassifier__max_depth": [None, 20, 40],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4]}
gs = GridSearchCV(pipe, grid, scoring="f1_macro",
                  cv=StratifiedGroupKFold(5), n_jobs=-1, verbose=1)
gs.fit(X, y, groups=groups)
print("best F1:", round(gs.best_score_, 3))
print("best params:", gs.best_params_)
