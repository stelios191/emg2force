"""Synthetic Hyser-shaped WFDB files so the tests run without the real dataset."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from emg2force.hyser import EMG_CHANNELS, FINGERS, FS_EMG, FS_FORCE


def write_record(stem: Path, digital: np.ndarray, gains, baselines, names, fs: float) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    n, m = digital.shape
    lines = [f"{stem.name} {m} {fs:g} {n}"]
    for g, b, name in zip(gains, baselines, names, strict=True):
        lines.append(f"{stem.name}.dat 16 {g}({b})/V 0 0 0 0 0 {name}")
    stem.with_suffix(".hea").write_text("\n".join(lines) + "\n")
    digital.astype("<i2").tofile(stem.with_suffix(".dat"))


FAKE_EMG_GAIN = 1000.0
FAKE_FORCE_GAIN = 10000.0
MVC_PLATEAU = slice(100, 400)  # 300 plateau samples, more than MVC_TOP_N


def _write_random_trial(folder: Path, trial: int, seconds: float, rng, emg_names: list[str]) -> None:
    emg = rng.integers(-3000, 3000, size=(int(seconds * FS_EMG), len(EMG_CHANNELS)))
    write_record(folder / f"random_raw_sample{trial}", emg, [FAKE_EMG_GAIN] * len(EMG_CHANNELS), [0] * len(EMG_CHANNELS), emg_names, FS_EMG)
    force = rng.integers(-2000, 2000, size=(int(seconds * FS_FORCE), len(FINGERS)))
    write_record(
        folder / f"random_force_sample{trial}", force, [FAKE_FORCE_GAIN] * len(FINGERS), [100] * len(FINGERS), list(FINGERS), FS_FORCE
    )


def _write_mvc_session(folder: Path, rng) -> np.ndarray:
    """One plateau recording per finger and direction. Returns the true MVC (5, 2) in volts."""
    mvc = np.zeros((len(FINGERS), 2))
    for f, (di, direction) in product(range(len(FINGERS)), enumerate(("flexion", "extension"))):
        peak = int(rng.integers(5000, 20000))
        x = rng.integers(-200, 200, size=(1000, len(FINGERS)))
        x[MVC_PLATEAU, f] = (-1 if direction == "flexion" else 1) * peak
        write_record(
            folder / f"mvc_force_finger{f + 1}_{direction}",
            x,
            [FAKE_FORCE_GAIN] * len(FINGERS),
            [0] * len(FINGERS),
            list(FINGERS),
            FS_FORCE,
        )
        mvc[f, di] = peak / FAKE_FORCE_GAIN
    return mvc


def make_fake_hyser(root: Path, subjects=(1, 2), sessions=(1, 2), trials=(1,), seconds=2.0, seed=0) -> dict:
    """Write RANDOM raw/force and MVC records. Return true MVC (volts) per (subject, session)."""
    rng = np.random.default_rng(seed)
    emg_names = list(EMG_CHANNELS)[::-1]  # stored order differs from canonical, as in real headers
    truth = {}
    for s, d in product(subjects, sessions):
        for t in trials:
            _write_random_trial(root / "random_dataset" / f"subject{s:02d}_session{d}", t, seconds, rng, emg_names)
        truth[(s, d)] = _write_mvc_session(root / "mvc_dataset" / f"subject{s:02d}_session{d}", rng)
    return truth


# experiments.run config of the shared smoke run (tests/conftest.py): constant, ridge and a TCN, seconds to train
SMOKE_CFG = {
    "name": "smoke",
    "subjects": [2],
    "protocol": "cross_day",
    "mvc_convention": "session",
    "label_filter": "zero_phase_lowpass_10hz",
    "history": 32,
    "warmup_s": 0.7,
    "input_norm": "running_mean",
    "seeds": [0],
    "save_predictions": True,
    "save_models": True,
    "models": {
        "constant": {},
        "ridge": {"alpha": 10.0},
        "tcn": {"hidden": 8, "kernel": 2, "dilations": [1, 2, 4, 8, 16], "steps": 3, "batch": 2, "crop": 64, "device": "cpu"},
    },
}
