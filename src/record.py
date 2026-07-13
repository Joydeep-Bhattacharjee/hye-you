"""Record N seconds to data/field as 16 kHz WAV.
Usage: python src\record.py COM5 8 normal p01"""
import sys
import time
import soundfile as sf
from config import FIELD, SAMPLE_RATE
from serial_stream import SerialAudio


def main() -> None:
    port, dur, label, sid = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    need = int(dur * SAMPLE_RATE)
    audio = SerialAudio(port, maxlen=need + SAMPLE_RATE)
    time.sleep(0.6)
    print(f"Recording {dur:.0f}s... breathe now.")
    time.sleep(dur)
    w = audio.get_window(need)
    audio.close()
    out = FIELD / f"{label}_{sid}_{int(time.time())}.wav"
    sf.write(out, (w / 32768.0), SAMPLE_RATE)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
