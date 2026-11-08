"""Held-press test: does the decoded force stay up while a press is held for 12 s?

    python -m experiments.hold_test runs/final_cross_day_pop_256 [runs/...] [--root ../hyser_raw]

Hyser presses are short, so the evaluation never holds one. Per person (that person's own fold model) and day-2 trial,
real EMG is replayed as 2 s of the trial's opening rest followed by the trial's strongest 1 s window repeated 12 times.
For each finger pressing at least ACTIVE_PCT in that window, ratio = decoded force 10-11 s into the hold divided by
the decoded force 1-2 s in. A decoder that holds gives about 1. Writes <run>/hold_test.csv and prints the medians.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml

from emg2force.hyser import FS_EMG, FS_FORCE, Hyser
from emg2force.models import Predictor
from emg2force.pipeline import load_decoder, offline_predict
from emg2force.provenance import REPO
from emg2force.splits import DAY2, TRIALS
from experiments.run import fold_checkpoints

REST_S, WINDOW_S, REPEATS = 2.0, 1.0, 12
REST_PCT = 5.0  # %MVC: every finger below this in the opening rest, or the trial is skipped
ACTIVE_PCT = 10.0  # %MVC: a finger counts as pressing if its mean force in the window is at least this
EARLY_S, LATE_S = (1.0, 2.0), (10.0, 11.0)  # seconds into the hold
MIN_EARLY = 2.0  # %MVC: an early output smaller than this (or of the wrong sign) gives no meaningful ratio


def strongest_window(y: np.ndarray, start: int) -> int:
    """First frame of the WINDOW_S window (from `start`) with the largest summed |force| over fingers."""
    n = int(WINDOW_S * FS_FORCE)
    s = np.convolve(np.abs(y).sum(1), np.ones(n), "valid")
    return start + int(np.argmax(s[start:]))


def hold_rows(ds: Hyser, model_path: Path, subject: int, mvc_convention: str, label_filter: str) -> list[dict]:
    decoder = load_decoder(model_path)
    rest_frames, win_frames = int(REST_S * FS_FORCE), int(WINDOW_S * FS_FORCE)
    n_frames = rest_frames + REPEATS * win_frames
    rows = []
    for t in TRIALS:
        rec = ds.load(subject, DAY2, t)
        y = ds.target(rec, mvc_convention, label_filter).values
        if np.abs(y[:rest_frames]).max() >= REST_PCT:
            continue
        w = strongest_window(y, rest_frames)
        target = y[w : w + win_frames].mean(0)
        rest = rec.emg[: int(REST_S * FS_EMG)]
        window = rec.emg[int(w * FS_EMG / FS_FORCE) : int((w + win_frames) * FS_EMG / FS_FORCE)]
        emg = np.concatenate([rest, *[window] * REPEATS])
        pred = offline_predict(
            emg, n_frames, decoder.front.sos, cast(Predictor, decoder.model), decoder.input_norm, decoder.n_base, decoder.montage
        )
        early = pred[rest_frames + int(EARLY_S[0] * FS_FORCE) : rest_frames + int(EARLY_S[1] * FS_FORCE)].mean(0)
        late = pred[rest_frames + int(LATE_S[0] * FS_FORCE) : rest_frames + int(LATE_S[1] * FS_FORCE)].mean(0)
        for f in np.flatnonzero(np.abs(target) >= ACTIVE_PCT):
            usable = abs(early[f]) >= MIN_EARLY and np.sign(early[f]) == np.sign(target[f])
            rows.append(
                {
                    "subject": subject,
                    "trial": t,
                    "finger": int(f),
                    "target": float(target[f]),
                    "early": float(early[f]),
                    "late": float(late[f]),
                    "ratio": float(late[f] / early[f]) if usable else np.nan,
                }
            )
    return rows


def run(run_dir: Path, root: Path) -> pd.DataFrame:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ds = Hyser(root)
    rows = []
    for subject, path in fold_checkpoints(run_dir, "tcn"):
        rows += hold_rows(ds, path, subject, cfg["mvc_convention"], cfg["label_filter"])
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "hold_test.csv", index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    args = ap.parse_args(argv)
    for run_dir in args.run_dirs:
        df = run(run_dir, args.root)
        print(
            f"{run_dir.name}: median ratio {df.ratio.median():.2f} over {df.ratio.notna().sum()} presses "
            f"({df.subject.nunique()} people, {df.ratio.isna().sum()} with no usable early output)"
        )


if __name__ == "__main__":
    main()
