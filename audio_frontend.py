"""Audio pre-processing: 16 kHz mono PCM -> input feature chunks for the model."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
CHUNK_FRAMES = 3000          # processing window
SPF = 1600                   # audio samples per 10 fps command frame

_FILT = None


def _filters(device):
    global _FILT
    if _FILT is None:
        with np.load(Path(__file__).with_name("mel_filters.npz")) as f:
            _FILT = torch.from_numpy(f["mel_80"])
    return _FILT.to(device)


def to_mono_16k(a: np.ndarray, sr: int) -> np.ndarray:
    """Any waveform -> float32 mono @16 kHz."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a.astype(np.float32)


def load_wav(path) -> np.ndarray:
    """Read any wav file -> float32 mono @16 kHz."""
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32")
    return to_mono_16k(a, sr)


def read_audio_bytes(data: bytes) -> np.ndarray:
    """Decode an in-memory audio file (wav/flac/ogg…) -> float32 mono @16 kHz."""
    import io
    import soundfile as sf
    a, sr = sf.read(io.BytesIO(data), dtype="float32")
    return to_mono_16k(a, sr)


def _features(audio, device="cuda") -> torch.Tensor:
    """(N,) waveform -> (80, M) input features."""
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    audio = audio.to(device)
    window = torch.hann_window(N_FFT, device=device)
    spec = torch.stft(audio, N_FFT, HOP, window=window, return_complex=True)
    mag = spec[:, :-1].abs() ** 2
    m = _filters(device) @ mag
    m = torch.clamp(m, min=1e-10).log10()
    m = torch.maximum(m, m.max() - 8.0)
    return (m + 4.0) / 4.0                                   # (80, M)


def _pad_or_trim(x, length=CHUNK_FRAMES):
    n = x.shape[1]
    if n > length:
        return x[:, :length]
    if n < length:
        return F.pad(x, (0, length - n))
    return x


def wav_to_chunks(audio, device="cuda"):
    """Waveform -> (list of (1,80,3000) input chunks, valid-frames per chunk, T command frames)."""
    n = len(audio)
    T = int(round(n / SPF))
    feat = _features(audio, device)                          # (80, M)
    M = feat.shape[1]
    chunks, nvalid, seek = [], [], 0
    while seek < M:
        seg = _pad_or_trim(feat[:, seek:seek + CHUNK_FRAMES])
        chunks.append(seg.unsqueeze(0))
        end = min(seek + CHUNK_FRAMES, M)
        nvalid.append((end - seek) // 2)
        seek += CHUNK_FRAMES
    return chunks, nvalid, T
