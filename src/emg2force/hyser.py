"""Loader for Hyser HD-sEMG (PhysioNet hd-semg v2.0.0): random-task recordings, MVC values and %MVC labels.

EMG headers say "V" but the values are mV in practice (tests/test_data.py checks the amplitude). Training is
unit-insensitive (log-RMS, then standardisation), but a trained decoder expects mV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy import signal

FS_EMG = 2048
FS_FORCE = 100
# The EMG clock runs slow against the force clock: mains hum sits near 50.14 Hz on the nominal 2048 Hz time base
# (median, 50.10 to 50.17 over the 177 of 200 recordings with a clear hum: the spectral peak near each 50 Hz harmonic
# of the channel mean, divided by its order), and the EMG-to-force lag grows through each trial at about that rate. The loader uses
# 50.13 Hz, 4 ms from the measured median by the end of a 25 s recording.
# Force is resampled onto the EMG clock, so a label never lags the EMG it is paired with (up to 65 ms by 25 s).
EMG_CLOCK_SCALE = 50.13 / 50
FINGERS = ("thumb", "index", "middle", "ring", "little")
GRIDS = ("ED", "EP", "FD", "FP")  # extensor-distal, extensor-proximal, flexor-distal, flexor-proximal
GRID_SIDE = 8  # electrodes per row and per column of one grid
GRID_CHANNELS = GRID_SIDE * GRID_SIDE
# Canonical EMG order: grid, then row i (1..8), then column j (1..8). Channel c sits at
# grid c // 64, row (c % 64) // 8, col c % 8 (0-based).
EMG_CHANNELS = tuple(f"{g}-{i}-{j}" for g in GRIDS for i in range(1, 9) for j in range(1, 9))
N_SUBJECTS = 20
N_SESSIONS = 2
N_RANDOM_TRIALS = 5
MVC_TOP_N = 200  # MVC = mean of the N largest |force| samples (reference get_mvc.m)
MVC_CONVENTIONS = ("session", "day1")
LABEL_FILTERS = ("none", "zero_phase_lowpass_10hz")
DEFAULT_LABEL_FILTER = "zero_phase_lowpass_10hz"  # reference: 8th-order 10 Hz zero-phase low-pass


# WFDB format-16 reading (header + interleaved little-endian int16)

_SIG_RE = re.compile(
    r"^(?P<file>\S+)\s+(?P<fmt>\d+)\s+(?P<gain>[-+0-9.eE]+)(?:\((?P<baseline>[-+0-9]+)\))?/?(?P<units>\S*)"
    r"\s+(?P<adcres>-?\d+)\s+(?P<adczero>-?\d+)\s+(?P<init>-?\d+)\s+(?P<checksum>-?\d+)\s+(?P<block>-?\d+)\s+(?P<name>\S+)$"
)


@dataclass(frozen=True)
class SignalSpec:
    file: str
    fmt: int
    gain: float
    baseline: int
    units: str
    name: str


@dataclass(frozen=True)
class Header:
    record: str
    n_sig: int
    fs: float
    n_samples: int
    signals: tuple[SignalSpec, ...]


def read_header(path: Path) -> Header:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    rec, n_sig, fs, n_samples = lines[0].split()[:4]
    sigs = []
    for ln in lines[1 : 1 + int(n_sig)]:
        m = _SIG_RE.match(ln)
        if m is None:
            raise ValueError(f"{path}: cannot parse signal line {ln!r}")
        sigs.append(
            SignalSpec(
                file=m["file"],
                fmt=int(m["fmt"]),
                gain=float(m["gain"]),
                baseline=int(m["baseline"]) if m["baseline"] is not None else int(m["adczero"]),
                units=m["units"],
                name=m["name"],
            )
        )
    if len(sigs) != int(n_sig):
        raise ValueError(f"{path}: expected {n_sig} signals, found {len(sigs)}")
    return Header(rec, int(n_sig), float(fs), int(n_samples), tuple(sigs))


def read_record(stem: Path) -> tuple[np.ndarray, Header]:
    """Read `<stem>.hea` / `<stem>.dat` and return physical values (n_samples, n_sig) in header units."""
    stem = Path(stem)
    hdr = read_header(stem.with_suffix(".hea"))
    if any(s.fmt != 16 for s in hdr.signals):
        raise ValueError(f"{stem}: only WFDB format 16 is supported")
    raw: np.ndarray = np.fromfile(stem.with_suffix(".dat"), dtype="<i2")
    if raw.size != hdr.n_samples * hdr.n_sig:
        raise ValueError(f"{stem}: .dat has {raw.size} values, header implies {hdr.n_samples * hdr.n_sig}")
    raw = raw.reshape(hdr.n_samples, hdr.n_sig)
    gain = np.array([s.gain for s in hdr.signals])
    base = np.array([s.baseline for s in hdr.signals])
    return (raw - base) / gain, hdr


# Timeline alignment


def emg_count_at(force_idx: np.ndarray | int, fs_emg: int = FS_EMG, fs_force: int = FS_FORCE) -> np.ndarray:
    """Number of EMG samples with timestamp <= j / fs_force (integer arithmetic, 2048 / 100 = 20.48)."""
    j = np.asarray(force_idx, dtype=np.int64)
    return (j * fs_emg) // fs_force + 1


def on_emg_clock(force: np.ndarray, scale: float = EMG_CLOCK_SCALE) -> np.ndarray:
    """Force at the times of the EMG clock: sample j becomes force at j * scale on the force clock (linear
    interpolation), ending where the force recording ends."""
    n = len(force)
    t = np.arange(int(np.floor((n - 1) / scale)) + 1) * scale
    return np.stack([np.interp(t, np.arange(n), force[:, f]) for f in range(force.shape[1])], axis=1)


# Dataset access


@dataclass
class Recording:
    """One RANDOM trial. EMG in header units (canonical channel order), force in volts (thumb..little)."""

    subject: int
    session: int
    trial: int
    emg: np.ndarray  # (n_emg, 256) float32, header units ("V", consistent with mV)
    force_v: np.ndarray  # (n_force, 5) float64, volts. Extension +, flexion -
    fs_force: int = FS_FORCE


def recording_key(subject: int, session: int, trial: int) -> str:
    return f"s{subject:02d}_d{session}_t{trial}"


def parse_key(key: str) -> tuple[int, int, int]:
    """'s03_d2_t4' -> (subject 3, session 2, trial 4)."""
    m = re.fullmatch(r"s(\d{2})_d(\d)_t(\d)", key)
    if m is None:
        raise ValueError(f"bad recording key {key!r}")
    return int(m[1]), int(m[2]), int(m[3])


@dataclass
class Target:
    """Signed %MVC force on the native 100 Hz force timeline, with its provenance."""

    values: np.ndarray  # (n_force, 5) %MVC. Extension +, flexion -
    mvc_convention: str
    mvc_source: str  # e.g. "subject03_session1"
    label_filter: str
    mvc: np.ndarray | None = field(default=None, repr=False)  # (5, 2) volts: [:,0] flexion, [:,1] extension


class Hyser:
    """The Hyser release on disk. EMG comes in canonical channel order, force at 100 Hz resampled onto the EMG clock."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # Paths
    def _random_stem(self, subject: int, session: int, trial: int, kind: str) -> Path:
        return self.root / "random_dataset" / f"subject{subject:02d}_session{session}" / f"random_{kind}_sample{trial}"

    def _mvc_stem(self, subject: int, session: int, finger: int, direction: str) -> Path:
        return self.root / "mvc_dataset" / f"subject{subject:02d}_session{session}" / f"mvc_force_finger{finger}_{direction}"

    def available(self) -> list[tuple[int, int, int]]:
        """(subject, session, trial) of every RANDOM trial whose raw EMG and force files both exist."""
        ids = product(range(1, N_SUBJECTS + 1), range(1, N_SESSIONS + 1), range(1, N_RANDOM_TRIALS + 1))
        return [i for i in ids if self._trial_on_disk(*i)]

    def _trial_on_disk(self, subject: int, session: int, trial: int) -> bool:
        stems = (self._random_stem(subject, session, trial, kind) for kind in ("raw", "force"))
        return all(stem.with_suffix(ext).exists() for stem in stems for ext in (".hea", ".dat"))

    # RANDOM
    def _force(self, subject: int, session: int, trial: int) -> np.ndarray:
        force, fh = read_record(self._random_stem(subject, session, trial, "force"))
        if fh.fs != FS_FORCE:
            raise ValueError(f"unexpected force sampling rate {fh.fs}")
        if {s.units for s in fh.signals} != {"V"}:
            raise ValueError("force header units are not volts")
        fnames = [s.name for s in fh.signals]
        return on_emg_clock(force[:, [fnames.index(f) for f in FINGERS]])

    def load(self, subject: int, session: int, trial: int) -> Recording:
        emg, eh = read_record(self._random_stem(subject, session, trial, "raw"))
        if eh.fs != FS_EMG:
            raise ValueError(f"unexpected EMG sampling rate {eh.fs}")
        names = [s.name for s in eh.signals]
        emg = emg[:, [names.index(c) for c in EMG_CHANNELS]]
        return Recording(subject, session, trial, emg.astype(np.float32), self._force(subject, session, trial))

    def load_force(self, subject: int, session: int, trial: int) -> Recording:
        """Force only (EMG left empty). Cheap when EMG features come from the frame cache."""
        return Recording(subject, session, trial, np.empty((0, 0), np.float32), self._force(subject, session, trial))

    # MVC
    def mvc_trace(self, subject: int, session: int, finger: int, direction: str) -> np.ndarray:
        """Force (V) of `finger` (0..4) in its MVC recording for `direction` ("flexion" or "extension")."""
        x, hdr = read_record(self._mvc_stem(subject, session, finger + 1, direction))
        if hdr.fs != FS_FORCE or {s.units for s in hdr.signals} != {"V"}:
            raise ValueError(f"unexpected MVC recording format (fs {hdr.fs}, units {sorted({s.units for s in hdr.signals})})")
        return x[:, [s.name for s in hdr.signals].index(FINGERS[finger])]

    @lru_cache(maxsize=64)
    def mvc(self, subject: int, session: int) -> np.ndarray:
        """(5, 2) MVC in volts, [:, 0] flexion, [:, 1] extension: mean of the top MVC_TOP_N |force| samples (`get_mvc.m`)."""
        out = np.zeros((5, 2))
        for f in range(5):
            for di, direction in enumerate(("flexion", "extension")):
                col = np.abs(self.mvc_trace(subject, session, f, direction))
                out[f, di] = np.sort(col)[::-1][:MVC_TOP_N].mean()
        out.setflags(write=False)  # cached and shared by every caller
        return out

    def mvc_for(self, subject: int, session: int, convention: str) -> tuple[np.ndarray, str]:
        """MVC values and source: the recording's own session ("session") or day 1 ("day1")."""
        if convention == "session":
            src = session
        elif convention == "day1":
            src = 1
        else:
            raise ValueError(f"unknown MVC convention {convention!r}, use one of {MVC_CONVENTIONS}")
        return self.mvc(subject, src), f"subject{subject:02d}_session{src}"

    def target(self, rec: Recording, convention: str, label_filter: str = DEFAULT_LABEL_FILTER) -> Target:
        mvc, src = self.mvc_for(rec.subject, rec.session, convention)
        values = make_target(rec.force_v, mvc, label_filter, rec.fs_force)
        return Target(values, convention, src, label_filter, mvc=mvc)


def normalize_force(force_v: np.ndarray, mvc: np.ndarray) -> np.ndarray:
    """Signed %MVC: flexion samples divided by the flexion MVC, others by the extension MVC (`normalize_force.m`)."""
    flex, ext = mvc[:, 0], mvc[:, 1]
    return 100.0 * np.where(force_v < 0, force_v / flex, force_v / ext)


def label_lowpass(x: np.ndarray, fs: float = FS_FORCE) -> np.ndarray:
    """8th-order 10 Hz zero-phase Butterworth. NON-CAUSAL: labels only.

    Padding (odd, 24 samples) matches MATLAB `filtfilt`. scipy's default of 27 differs at trial edges.
    """
    sos = signal.butter(8, 10, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0, padtype="odd", padlen=24)


def make_target(force_v: np.ndarray, mvc: np.ndarray, label_filter: str, fs: float = FS_FORCE) -> np.ndarray:
    """(n, 5) volts -> signed %MVC (extension +, flexion -), then the optional zero-phase label filter."""
    y = normalize_force(force_v, mvc)  # reference order: normalise, then filter
    if label_filter == "zero_phase_lowpass_10hz":
        y = label_lowpass(y, fs)
    elif label_filter != "none":
        raise ValueError(f"unknown label filter {label_filter!r}, use one of {LABEL_FILTERS}")
    return y
