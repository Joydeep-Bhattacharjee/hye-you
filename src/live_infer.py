"""Live 1-second predictions with smoothing. Usage: python src\live_infer.py COM5"""
import sys
import time
from collections import deque, Counter
import numpy as np
import joblib

from config import MODELS, SAMPLE_RATE
from features import preprocess, extract_features
from serial_stream import SerialAudio, list_ports

WIN = SAMPLE_RATE


def main() -> None:
    if len(sys.argv) < 2:
        for p in list_ports.comports():
            print("  ", p.device, "-", p.description)
        print("Run: python src\\live_infer.py COM5")
        return

    bundle = joblib.load(MODELS / "baseline.joblib")
    model, labels = bundle["model"], bundle["labels"]
    audio = SerialAudio(sys.argv[1], maxlen=SAMPLE_RATE * 3)
    print("warming up..."); time.sleep(1.5)
    history = deque(maxlen=3)
    try:
        while True:
            w = preprocess(audio.get_window(WIN))
            f = extract_features(w).reshape(1, -1)
            proba = model.predict_proba(f)[0]
            history.append(int(np.argmax(proba)))
            smooth = Counter(history).most_common(1)[0][0]
            label = labels[smooth]
            flag = "normal" if label == "normal" else ">>> NEEDS REFERRAL <<<"
            probs = "  ".join(f"{labels[i]}:{proba[i]:.2f}" for i in range(len(labels)))
            print(f"{label:8s}  {flag:24s}  [{probs}]")
            time.sleep(1.0)
    except KeyboardInterrupt:
        audio.close()
        print("\nstopped.")


if __name__ == "__main__":
    main()
