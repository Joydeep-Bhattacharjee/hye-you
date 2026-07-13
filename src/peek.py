import pandas as pd, librosa, numpy as np
from config import PROCESSED, RAW_ICBHI, SAMPLE_RATE

df = pd.read_csv(PROCESSED / "cycles.csv")
row = df.iloc[0]
print("First cycle:", dict(row))
y, sr = librosa.load(RAW_ICBHI / row["wav"], sr=SAMPLE_RATE,
                     offset=row["start"], duration=row["duration"])
print(f"Loaded slice: {len(y)} samples @ {sr} Hz = {len(y)/sr:.2f} s")
print("Signal min/max:", float(np.min(y)), float(np.max(y)))
