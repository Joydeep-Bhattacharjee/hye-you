"""Parse ICBHI .txt annotations into one labeled table of breath cycles."""
from pathlib import Path
import pandas as pd

from config import RAW_ICBHI, PROCESSED, cycle_label


def patient_id(stem: str) -> str:
    return stem.split("_")[0]


def load_official_split(folder: Path) -> dict:
    f = folder / "ICBHI_challenge_train_test.txt"
    mapping = {}
    if f.exists():
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1].strip().lower()
    return mapping


def main() -> None:
    wavs = sorted(RAW_ICBHI.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files in {RAW_ICBHI}. Re-check Step 1.1.")

    official = load_official_split(RAW_ICBHI)
    if official:
        print(f"Found official train/test split for {len(official)} recordings.")
    else:
        print("No official split file found; will split by patient automatically.")

    rows = []
    skipped = 0
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            skipped += 1
            continue
        pid = patient_id(wav.stem)
        split = official.get(wav.stem, "")
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            start, end, crackle, wheeze = parts[:4]
            start, end = float(start), float(end)
            crackle, wheeze = int(float(crackle)), int(float(wheeze))
            rows.append({
                "wav": wav.name, "patient": pid,
                "start": start, "end": end,
                "duration": round(end - start, 3),
                "crackle": crackle, "wheeze": wheeze,
                "label": cycle_label(crackle, wheeze), "split": split,
            })

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} cycles from {df['wav'].nunique()} recordings "
          f"({df['patient'].nunique()} patients). Skipped {skipped}.")

    if (df["split"] == "").any():
        import numpy as np
        rng = np.random.default_rng(42)
        patients = df.loc[df["split"] == "", "patient"].unique()
        rng.shuffle(patients)
        n_test = max(1, int(len(patients) * 0.2))
        test_patients = set(patients[:n_test])
        mask = df["split"] == ""
        df.loc[mask, "split"] = df.loc[mask, "patient"].apply(
            lambda p: "test" if p in test_patients else "train")

    bad = (df.groupby("patient")["split"].nunique() > 1)
    if bad.any():
        for p in bad[bad].index:
            df.loc[df["patient"] == p, "split"] = "train"
        print(f"Fixed {bad.sum()} patients that spanned both splits -> train.")

    out = PROCESSED / "cycles.csv"
    df.to_csv(out, index=False)
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nBy split:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
