"""Build fixed-size log-mel images -> specs.npz (upload to Colab, Phase 7)."""
import numpy as np, pandas as pd, librosa
from tqdm import tqdm
from config import PROCESSED, RAW_ICBHI, SAMPLE_RATE, LABEL_TO_ID
from features import preprocess, logmel_image, N_MELS_IMG, IMG_FRAMES


def main() -> None:
    df = pd.read_csv(PROCESSED / "cycles.csv")
    X = np.zeros((len(df), N_MELS_IMG, IMG_FRAMES), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    split = df["split"].to_numpy()
    cache = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="specs"):
        wav = row["wav"]
        if wav not in cache:
            a, _ = librosa.load(RAW_ICBHI / wav, sr=SAMPLE_RATE)
            cache = {wav: preprocess(a)}
        audio = cache[wav]
        s = int(row["start"] * SAMPLE_RATE); e = int(row["end"] * SAMPLE_RATE)
        seg = audio[s:e]
        if len(seg) < 10:
            seg = np.zeros(SAMPLE_RATE, dtype=np.float32)
        X[i] = logmel_image(seg)
        y[i] = LABEL_TO_ID[row["label"]]
    out = PROCESSED / "specs.npz"
    np.savez_compressed(out, X=X, y=y, split=split)
    print(f"Saved {X.shape} -> {out}  (~{out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
