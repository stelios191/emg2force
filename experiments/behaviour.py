"""Timing and height of the decoded force, from the saved new-day predictions (seed 0, no training).

    python -m experiments.behaviour [--root ../hyser_raw]

Lag: per day-2 recording and finger whose label moves (SD >= MIN_SD %MVC), the shift of the prediction, from
-MAX_LEAD to +MAX_LAG frames, with the highest correlation to the label. Positive = late. Peaks: a press is at least
MIN_FRAMES frames of one finger at STRONG %MVC or more in one direction. Per press, the decoded peak in that direction
over the label's. MVC: the error with each day's own maximal-force recordings (as everywhere else) and with day 1's,
which is all a user would have. Writes runs/behaviour/lag.csv, peaks.csv and mvc.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import label as runs_of

from emg2force.dataset import warmup_frames
from emg2force.hyser import FINGERS, Hyser, parse_key
from emg2force.provenance import REPO

RUNS = {"tcn": "final_cross_day_pop_256", "ridge": "final_cross_day_pop_256_ridge_weighted"}
MAX_LEAD, MAX_LAG = 50, 50  # frames: 500 ms early to 500 ms late
MIN_SD = 2.0  # %MVC
STRONG = 20.0  # %MVC
MIN_FRAMES = 10  # 100 ms


def best_lag(y: np.ndarray, yhat: np.ndarray) -> int | None:
    """Shift L (frames) maximising corr(y[t], yhat[t + L]) over the overlap. Positive = yhat late."""
    n = len(y)
    corr = {}
    for lag in range(-MAX_LEAD, MAX_LAG + 1):
        a, b = (y[: n - lag], yhat[lag:]) if lag >= 0 else (y[-lag:], yhat[: n + lag])
        corr[lag] = np.corrcoef(a, b)[0, 1] if np.std(b) > 0 else np.nan
    if all(np.isnan(v) for v in corr.values()):
        return None  # a constant prediction has no timing
    return max(corr, key=lambda k: -np.inf if np.isnan(corr[k]) else corr[k])


def press_peaks(y: np.ndarray, yhat: np.ndarray) -> list[tuple[float, float]]:
    """(label peak, decoded peak) per press of one finger, both as positive magnitudes in the press direction."""
    out = []
    for sign in (1.0, -1.0):
        labels, n = runs_of(sign * y >= STRONG)
        for i in range(1, n + 1):
            m = labels == i
            if m.sum() >= MIN_FRAMES:
                out.append((float((sign * y[m]).max()), float((sign * yhat[m]).max())))
    return out


def run_rows(model: str, run: str, ds: Hyser) -> tuple[list[dict], list[dict], list[dict]]:
    run_dir = REPO / "runs" / run
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    start = warmup_frames(cfg)
    with np.load(run_dir / "predictions.npz") as z:
        preds = dict(z)
    lags, peaks, mvc = [], [], []
    for name in (k for k in preds if k.startswith(f"{model}|0|")):
        key = name.split("|")[2]
        s, d, t = parse_key(key)
        rec = ds.load_force(s, d, t)
        y = ds.target(rec, cfg["mvc_convention"], cfg["label_filter"]).values[start:]
        y1 = ds.target(rec, "day1", cfg["label_filter"]).values[start:]
        yhat = preds[name][start : start + len(y)]
        row = {"model": model, "subject": s, "recording": key}
        for m, out in ((model, yhat), ("zero", np.zeros_like(yhat))) if model == "tcn" else ((model, yhat),):
            mvc += [{**row, "model": m, "session": d, "finger": finger, "mae_session": float(np.mean(np.abs(out[:, f] - y[:, f]))),
                     "mae_day1": float(np.mean(np.abs(out[:, f] - y1[:, f])))} for f, finger in enumerate(FINGERS)]  # fmt: skip
        for f, finger in enumerate(FINGERS):
            if np.std(y[:, f]) >= MIN_SD and (lag := best_lag(y[:, f], yhat[:, f])) is not None:
                lags.append({**row, "finger": finger, "lag_ms": lag * 10})
            peaks += [{**row, "finger": finger, "peak": p, "peak_hat": ph} for p, ph in press_peaks(y[:, f], yhat[:, f])]
    return lags, peaks, mvc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    args = ap.parse_args(argv)
    ds = Hyser(args.root)
    results = [run_rows(model, run, ds) for model, run in RUNS.items()]
    out = REPO / "runs" / "behaviour"
    out.mkdir(parents=True, exist_ok=True)
    lag, peaks, mvc = (pd.DataFrame([r for res in results for r in res[i]]) for i in range(3))
    lag.to_csv(out / "lag.csv", index=False)
    peaks.to_csv(out / "peaks.csv", index=False)
    mvc.to_csv(out / "mvc.csv", index=False)
    print(lag.groupby(["model", "subject"]).lag_ms.median().groupby("model").median().to_string())
    print((peaks.peak_hat / peaks.peak).groupby([peaks.model, peaks.subject]).mean().groupby("model").mean().round(2).to_string())


if __name__ == "__main__":
    main()
