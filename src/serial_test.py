"""Read framed audio from ESP32, print live RMS.
Usage: python src\serial_test.py          (list ports)
       python src\serial_test.py COM5"""
import sys
import numpy as np
import serial
from serial.tools import list_ports

BAUD = 921600
SYNC = b"\xA5\x5A\xA5\x5A"


def read_packet(ser, buf: bytearray) -> np.ndarray:
    while True:
        while len(buf) < 6:
            buf += ser.read(4096)
        idx = buf.find(SYNC)
        if idx < 0:
            buf[:] = buf[-3:]
            buf += ser.read(4096)
            continue
        buf[:] = buf[idx + 4:]
        while len(buf) < 2:
            buf += ser.read(64)
        cnt = int.from_bytes(buf[:2], "little")
        buf[:] = buf[2:]
        if cnt == 0 or cnt > 4096:
            continue
        need = cnt * 2
        while len(buf) < need:
            buf += ser.read(need - len(buf))
        out = np.frombuffer(bytes(buf[:need]), dtype="<i2").copy()
        buf[:] = buf[need:]
        return out


def main() -> None:
    if len(sys.argv) < 2:
        for p in list_ports.comports():
            print("  ", p.device, "-", p.description)
        print("Run: python src\\serial_test.py COM5")
        return
    ser = serial.Serial(sys.argv[1], BAUD, timeout=1)
    buf = bytearray()
    print("Streaming. TAP the mic - RMS should jump. Ctrl+C to stop.")
    try:
        while True:
            x = read_packet(ser, buf).astype(np.float32)
            rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            print(f"n={len(x):4d}  rms={rms:8.1f}  " + "#" * min(50, int(rms / 30)))
    except KeyboardInterrupt:
        ser.close()
        print("\nstopped.")


if __name__ == "__main__":
    main()
