"""Signal preprocessing + fixed-length feature extraction.
Used by BOTH training (ICBHI) and live hardware inference."""
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

from config import SAMPLE_RATE

BAND_LOW = 100
BAND_HIGH = 2000
N_MFCC = 13
N_FFT = 1024
HOP = 256
MIN_SAMPLES = SAMPLE_RATE // 2

_SOS = butter(4, [BAND_LOW / (SAMPLE_RATE / 2), BAND_HIGH / (SAMPLE_RATE / 2)],
              btype="band", output="sos")


def preprocess(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 30:
        return y
    return sosfiltfilt(_SOS, y).astype(np.float32)


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    delta = librosa.feature.delta(mfcc)
    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
    ])
    return feats.astype(np.float32)


FEATURE_DIM = 52

# --- Optional CNN spectrogram image (Phase 7) ---
N_MELS_IMG = 64
IMG_FRAMES = 128
IMG_SAMPLES = SAMPLE_RATE * 2


def logmel_image(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) < IMG_SAMPLES:
        y = np.pad(y, (0, IMG_SAMPLES - len(y)))
    else:
        y = y[:IMG_SAMPLES]
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=1024,
                                         hop_length=256, n_mels=N_MELS_IMG)
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] < IMG_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, IMG_FRAMES - mel.shape[1])), mode="edge")
    else:
        mel = mel[:, :IMG_FRAMES]
    return ((mel + 80.0) / 80.0).astype(np.float32)
