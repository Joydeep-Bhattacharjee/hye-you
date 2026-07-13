"""Live GUI using the CNN. Usage: python src\live_gui_cnn.py COM5"""
import sys
from collections import deque, Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

from config import MODELS, SAMPLE_RATE, LABELS
from features import preprocess, logmel_image
from serial_stream import SerialAudio, list_ports

WIN = SAMPLE_RATE * 2


def main() -> None:
    if len(sys.argv) < 2:
        for p in list_ports.comports():
            print("  ", p.device, "-", p.description)
        return
    model = tf.keras.models.load_model(MODELS / "stetho_cnn.keras")
    audio = SerialAudio(sys.argv[1], maxlen=SAMPLE_RATE * 3)
    history = deque(maxlen=3)
    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(9, 6))
    t = np.arange(WIN) / SAMPLE_RATE
    (line,) = ax_w.plot(t, np.zeros(WIN)); ax_w.set_ylim(-3000, 3000); ax_w.set_xlim(0, 2)
    img = ax_s.imshow(np.zeros((64, 128)), origin="lower", aspect="auto",
                      cmap="magma", vmin=0, vmax=1)
    title = ax_w.text(0.5, 1.05, "", transform=ax_w.transAxes, ha="center",
                      fontsize=13, fontweight="bold")

    def update(_):
        w = audio.get_window(WIN)
        im = logmel_image(preprocess(w))
        proba = model.predict(im[None, ..., None], verbose=0)[0]
        history.append(int(np.argmax(proba)))
        smooth = Counter(history).most_common(1)[0][0]
        label = LABELS[smooth]
        flag = "" if label == "normal" else "  - NEEDS REFERRAL"
        line.set_ydata(w); img.set_data(im)
        title.set_text(f"{label.upper()} ({proba[smooth]:.0%}){flag}")
        title.set_color("green" if label == "normal" else "red")
        return line, img, title

    FuncAnimation(fig, update, interval=800, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        audio.close()


if __name__ == "__main__":
    main()
