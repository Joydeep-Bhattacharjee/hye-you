"""Background thread reading framed ESP32 audio into a rolling buffer."""
import threading
import numpy as np
import serial
from serial.tools import list_ports

BAUD = 921600
SYNC = b"\xA5\x5A\xA5\x5A"


class SerialAudio:
    def __init__(self, port: str, maxlen: int = 48000):
        self.ser = serial.Serial(port, BAUD, timeout=1)
        self.buf = bytearray()
        self.maxlen = maxlen
        self.samples = np.zeros(maxlen, dtype=np.float32)
        self.lock = threading.Lock()
        self.running = True
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _read_packet(self) -> np.ndarray:
        b = self.buf
        while True:
            while len(b) < 6:
                b += self.ser.read(4096)
            idx = b.find(SYNC)
            if idx < 0:
                b[:] = b[-3:]
                b += self.ser.read(4096)
                continue
            b[:] = b[idx + 4:]
            while len(b) < 2:
                b += self.ser.read(64)
            cnt = int.from_bytes(b[:2], "little")
            b[:] = b[2:]
            if cnt == 0 or cnt > 4096:
                continue
            need = cnt * 2
            while len(b) < need:
                b += self.ser.read(need - len(b))
            out = np.frombuffer(bytes(b[:need]), dtype="<i2").copy()
            b[:] = b[need:]
            return out.astype(np.float32)

    def _run(self) -> None:
        while self.running:
            try:
                x = self._read_packet()
            except Exception:
                continue
            n = len(x)
            with self.lock:
                if n >= self.maxlen:
                    self.samples = x[-self.maxlen:].copy()
                else:
                    self.samples = np.roll(self.samples, -n)
                    self.samples[-n:] = x

    def get_window(self, n: int) -> np.ndarray:
        with self.lock:
            return self.samples[-n:].copy()

    def close(self) -> None:
        self.running = False
        self.t.join(timeout=1.0)
        self.ser.close()
