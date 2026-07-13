"""Read cycles.csv, extract features for every cycle, save features.npz."""
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from config import PROCESSED, RAW_ICBHI, SAMPLE_RATE, LABEL_TO_ID
from features import preprocess, extract_features, FEATURE_DIM


def main() -> None:
    df = pd.read_csv(PROCESSED / "cycles.csv")
    X = np.zeros((len(df), FEATURE_DIM), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    split = df["split"].to_numpy()

    cache = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="features"):
        wav = row["wav"]
        if wav not in cache:
            audio, _ = librosa.load(RAW_ICBHI / wav, sr=SAMPLE_RATE)
            cache[wav] = preprocess(audio)
            if len(cache) > 3:
                oldest = next(iter(cache))
                if oldest != wav:
                    cache.pop(oldest)
        audio = cache[wav]
        a = int(row["start"] * SAMPLE_RATE)
        b = int(row["end"] * SAMPLE_RATE)
        seg = audio[a:b]
        if len(seg) < 10:
            seg = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        X[i] = extract_features(seg)
        y[i] = LABEL_TO_ID[row["label"]]

    out = PROCESSED / "features.npz"
    np.savez_compressed(out, X=X, y=y, split=split)
    print(f"Saved {X.shape} features -> {out}")
    print("Any NaN in features:", bool(np.isnan(X).any()))


if __name__ == "__main__":
    main()
