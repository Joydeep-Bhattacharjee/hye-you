import sys
import importlib

print("Python:", sys.version.split()[0])

packages = [
    "numpy", "scipy", "pandas", "sklearn", "imblearn",
    "librosa", "soundfile", "pywt", "matplotlib",
    "seaborn", "sounddevice", "serial", "joblib", "tqdm",
]

ok = True
for name in packages:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK]   {name:14s} {ver}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name:14s} -> {e}")

try:
    import numpy as np, librosa
    y = np.random.randn(16000).astype("float32")
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
    print(f"[OK]   librosa MFCC shape {m.shape}")
except Exception as e:
    ok = False
    print(f"[FAIL] librosa MFCC -> {e}")

print("\nRESULT:", "ALL GOOD" if ok else "SOMETHING FAILED - read the [FAIL] lines")
