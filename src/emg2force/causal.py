"""Causal EMG preprocessing: past-only IIR filter, 10 ms log-RMS frames, running-mean input normalisation.

Frame j holds exactly the EMG samples in (t_{j-1}, t_j], t_j = j / 100 s. Offline and streaming paths share
coefficients and framing, and filter state restarts at every recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from emg2force.hyser import FS_EMG, FS_FORCE, GRID_SIDE, GRIDS, emg_count_at

# Spatial montage of the raw EMG, applied before filtering (the log-RMS is per channel, so a model cannot undo it).
# Hyser is monopolar against one distant reference: every channel shares reference noise and sees a wide area.
MONTAGES = ("monopolar", "car")  # as recorded, or minus each grid's mean


def apply_montage(emg: np.ndarray, name: str, exclude: np.ndarray | None = None) -> np.ndarray:
    """(samples, 256) EMG in canonical order -> the montage's 256 channels.

    `exclude` (256 bool, streaming only): electrodes the input monitor flagged. They leave each grid's mean, and the
    montage channels built from them read 0 (the decoder then fills them in). The montage is linear,
    so applying it before or after the (linear, per-channel) filter gives the same frames.
    """
    if name == "monopolar":
        return emg
    g = emg.reshape(len(emg), len(GRIDS), GRID_SIDE, GRID_SIDE)
    keep = None if exclude is None or not exclude.any() else ~exclude.reshape(len(GRIDS), GRID_SIDE, GRID_SIDE)
    if name == "car":
        if keep is None:
            return (g - g.mean(axis=(2, 3), keepdims=True)).reshape(len(emg), -1)
        mean = (g * keep).sum(axis=(2, 3), keepdims=True) / np.maximum(keep.sum(axis=(1, 2)), 1)[None, :, None, None]
        return ((g - mean) * keep).reshape(len(emg), -1)
    raise ValueError(f"unknown montage {name!r}, use one of {MONTAGES}")


WARMUP_FRAMES = 70  # unscored frames at the start of every recording (0.7 s). The decoder is silent until then
LOG_EPS = 1e-12  # floor inside the log, in squared EMG header units (1e-6 mV = 1 nV RMS). Only hit by exact zeros


@dataclass(frozen=True)
class FilterConfig:
    """Front end: a spatial montage, then a Butterworth band-pass plus notches at 50 Hz and its harmonics, applied
    causally (sosfilt)."""

    band: tuple[float, float] = (10.0, 500.0)
    order: int = 4  # Butterworth order per band edge
    notch_base: float = 50.0
    notch_max: float = 500.0  # notch 50 Hz and harmonics up to this frequency
    notch_q: float = 30.0
    fs: float = FS_EMG  # recorded in checkpoints. Framing assumes FS_EMG, so nothing else is accepted
    montage: str = "monopolar"  # spatial montage applied before the filter (MONTAGES)

    def __post_init__(self) -> None:
        if self.montage not in MONTAGES:
            raise ValueError(f"unknown montage {self.montage!r}, use one of {MONTAGES}")
        if self.fs != FS_EMG:
            raise ValueError(f"fs {self.fs} != {FS_EMG}: the 10 ms framing (emg_count_at) is defined for {FS_EMG} Hz only")


def design_sos(cfg: FilterConfig = FilterConfig()) -> np.ndarray:
    """Second-order sections of the band-pass followed by the notch comb, for `scipy.signal.sosfilt`."""
    sos = [signal.butter(cfg.order, cfg.band, btype="bandpass", fs=cfg.fs, output="sos")]
    if cfg.notch_base > 0:
        for f0 in np.arange(cfg.notch_base, cfg.notch_max + 1e-9, cfg.notch_base):
            b, a = signal.iirnotch(f0, cfg.notch_q, fs=cfg.fs)
            sos.append(signal.tf2sos(b, a))
    return np.vstack(sos)


def frame_ends(n_frames: int) -> np.ndarray:
    """Exclusive EMG end index of each frame: frame j covers [ends[j-1], ends[j])."""
    return emg_count_at(np.arange(n_frames), FS_EMG, FS_FORCE)


def log_rms(sumsq: np.ndarray, count: np.ndarray | int) -> np.ndarray:
    return 0.5 * np.log(sumsq / count + LOG_EPS)


def offline_frames(emg: np.ndarray, n_frames: int, sos: np.ndarray) -> np.ndarray:
    """(n_frames, C) causal log-RMS frames for one recording, computed in one batch."""
    ends = frame_ends(n_frames)
    if len(emg) < ends[-1]:
        raise ValueError(f"EMG has {len(emg)} samples, frames need {ends[-1]}")
    zi = np.zeros((sos.shape[0], 2, emg.shape[1]))
    y, _ = signal.sosfilt(sos, emg[: ends[-1]].astype(np.float64), axis=0, zi=zi)
    starts = np.concatenate([[0], ends[:-1]])
    sumsq = np.add.reduceat(y * y, starts, axis=0)
    return log_rms(sumsq, (ends - starts)[:, None]).astype(np.float32)


class StreamingFrontEnd:
    """Push raw EMG chunks of any size. Receive completed log-RMS frames as they close.

    With a montage, the raw channels are filtered and the montage is applied to the filtered samples: `push_pair`
    returns each frame both raw (for the input monitor) and in the montage (for the model).
    """

    def __init__(self, n_channels: int, sos: np.ndarray, montage: str = "monopolar"):
        self.sos = sos
        self.n_channels = n_channels
        self.montage = montage
        self.n_out = apply_montage(np.zeros((1, n_channels)), montage).shape[1]
        self.reset()

    def reset(self) -> None:
        """Call at every recording boundary: filter state and framing restart from zero."""
        self.zi = np.zeros((self.sos.shape[0], 2, self.n_channels))
        self.n_seen = 0  # EMG samples consumed so far in this recording
        self.frame = 0  # index of the frame currently being filled
        self.frame_end = int(emg_count_at(0))
        self.sumsq = np.zeros(self.n_channels)
        self.sumsq_out = np.zeros(self.n_out)
        self.count = 0

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """(frame index, raw log-RMS frame) for every frame the chunk completes."""
        return [(j, raw) for j, raw, _ in self.push_pair(chunk)]

    def push_pair(self, chunk: np.ndarray, exclude: np.ndarray | None = None) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """(frame index, raw frame, montage frame) for every frame the chunk completes. `exclude` as in apply_montage."""
        y, self.zi = signal.sosfilt(self.sos, chunk.astype(np.float64), axis=0, zi=self.zi)
        mono = self.montage == "monopolar"
        out = []
        pos = 0
        while pos < len(y):
            take = min(self.frame_end - self.n_seen, len(y) - pos)
            seg = y[pos : pos + take]
            self.sumsq += np.einsum("ij,ij->j", seg, seg)
            if not mono:
                m = apply_montage(seg, self.montage, exclude)
                self.sumsq_out += np.einsum("ij,ij->j", m, m)
            self.count += take
            self.n_seen += take
            pos += take
            if self.n_seen == self.frame_end:
                raw = log_rms(self.sumsq, self.count).astype(np.float32)
                out.append((self.frame, raw, raw if mono else log_rms(self.sumsq_out, self.count).astype(np.float32)))
                self.frame += 1
                self.frame_end = int(emg_count_at(self.frame))
                self.sumsq = np.zeros(self.n_channels)
                self.sumsq_out = np.zeros(self.n_out)
                self.count = 0
        return out


# Per-recording input normalisation from unlabelled EMG only (causal). "running_mean" subtracts each channel's mean
# of every frame so far, so a held press slowly becomes the baseline. The final recipes use "none".
INPUT_NORMS = ("none", "running_mean")


def normalize_input(x: np.ndarray, mode: str) -> np.ndarray:
    """Frames after the per-recording input normalisation `mode`."""
    if mode == "none":
        return x
    if mode == "running_mean":
        cs = np.cumsum(x, axis=0, dtype=np.float64)
        return (x - cs / np.arange(1, len(x) + 1)[:, None]).astype(np.float32)
    raise ValueError(f"unknown input normalisation {mode!r}, use one of {INPUT_NORMS}")
