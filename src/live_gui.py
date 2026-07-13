"""Live GUI: waveform + spectrogram + prediction. Usage: python src\live_gui.py COM5"""
import sys
from collections import deque, Counter
import numpy as np
import joblib
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import MODELS, SAMPLE_RATE
from features import preprocess, extract_features
from serial_stream import SerialAudio, list_ports

WIN = SAMPLE_RATE


def main() -> None:
    if len(sys.argv) < 2:
        for p in list_ports.comports():
            print("  ", p.device, "-", p.description)
        print("Run: python src\\live_gui.py COM5")
        return

    bundle = joblib.load(MODELS / "baseline.joblib")
    model, labels = bundle["model"], bundle["labels"]
    audio = SerialAudio(sys.argv[1], maxlen=SAMPLE_RATE * 3)
    history = deque(maxlen=3)

    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(9, 6))
    fig.suptitle("Digital Stethoscope - live", fontsize=14)
    t = np.arange(WIN) / SAMPLE_RATE
    (line,) = ax_w.plot(t, np.zeros(WIN))
    ax_w.set_ylim(-3000, 3000); ax_w.set_xlim(0, 1)
    spec_img = ax_s.imshow(np.zeros((40, 63)), origin="lower",
                           aspect="auto", cmap="magma", vmin=-80, vmax=0)
    title = ax_w.text(0.5, 1.05, "", transform=ax_w.transAxes,
                      ha="center", fontsize=13, fontweight="bold")

    def update(_):
        w = audio.get_window(WIN)
        yp = preprocess(w)
        f = extract_features(yp).reshape(1, -1)
        proba = model.predict_proba(f)[0]
        pred = int(np.argmax(proba))
        history.append(pred)
        smooth = Counter(history).most_common(1)[0][0]
        label = labels[smooth]
        flag = "" if label == "normal" else "  - NEEDS REFERRAL"
        line.set_ydata(w)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=yp, sr=SAMPLE_RATE,
                                           n_fft=1024, hop_length=256, n_mels=40),
            ref=np.max)
        spec_img.set_data(mel)
        title.set_text(f"{label.upper()}  ({proba[pred]:.0%}){flag}")
        title.set_color("green" if label == "normal" else "red")
        return line, spec_img, title

    FuncAnimation(fig, update, interval=700, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        audio.close()


if __name__ == "__main__":
    main()
