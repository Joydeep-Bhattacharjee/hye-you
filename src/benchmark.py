"""Measure per-window inference latency for the baseline model."""
import time
import numpy as np
import joblib
from config import MODELS, SAMPLE_RATE
from features import preprocess, extract_features

bundle = joblib.load(MODELS / "baseline.joblib")
model = bundle["model"]
y = np.random.randn(SAMPLE_RATE).astype("float32")
extract_features(preprocess(y))
N = 100
t0 = time.perf_counter()
for _ in range(N):
    f = extract_features(preprocess(y)).reshape(1, -1)
    model.predict(f)
ms = (time.perf_counter() - t0) / N * 1000
print(f"avg feature+predict = {ms:.1f} ms/window")
print("real-time budget = 1000 ms -> " + ("OK" if ms < 1000 else "TOO SLOW"))
