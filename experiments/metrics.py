"""Per-finger, per-recording metrics in %MVC and their hierarchical mean (fingers -> recordings -> sessions -> people).

Rest per finger: label |y| < REST_THRESH.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from emg2force.hyser import FINGERS

REST_THRESH = 5.0  # %MVC


def recording_metrics(y: np.ndarray, yhat: np.ndarray, scored: np.ndarray) -> list[dict]:
    """One row per finger over the scored frames. A non-finite prediction on a scored frame raises."""
    if not np.isfinite(yhat[scored]).all():
        raise ValueError(f"{int((~np.isfinite(yhat[scored]).all(1)).sum())} scored frames have non-finite predictions")
    y, yhat = y[scored], yhat[scored]
    rest = np.abs(y) < REST_THRESH
    rows = []
    for f, name in enumerate(FINGERS):
        e = yhat[:, f] - y[:, f]
        ss_tot = np.sum((y[:, f] - y[:, f].mean()) ** 2)
        row = {
            "finger": name,
            "n": int(len(e)),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "bias": float(np.mean(e)),
            "r2": float(1 - np.sum(e**2) / ss_tot) if ss_tot > 0 else np.nan,
            "r": float(np.corrcoef(yhat[:, f], y[:, f])[0, 1]) if np.ptp(yhat[:, f]) > 0 and ss_tot > 0 else np.nan,
            "mae_rest": float(np.mean(np.abs(e[rest[:, f]]))) if rest[:, f].any() else np.nan,
        }
        rows.append(row)
    return rows


def hierarchical_mean(df: pd.DataFrame, metric: str, by: list[str] | None = None) -> pd.DataFrame:
    """Mean over fingers, then recordings, then sessions: one row per person (plus the `by` columns)."""
    by = list(by or [])
    rec = df.groupby([*by, "subject", "session", "recording"], dropna=False)[metric].mean()
    ses = rec.groupby(level=[*by, "subject", "session"], dropna=False).mean()
    return ses.groupby(level=[*by, "subject"], dropna=False).mean().reset_index()


def summary(df: pd.DataFrame, metric: str, by: list[str]) -> pd.DataFrame:
    """Mean, sd, median, min and max over people of the per-person hierarchical metric."""
    subj = hierarchical_mean(df, metric, by)
    g = subj.groupby(by)[metric]
    return g.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()


def check_complete(df: pd.DataFrame, recordings: Iterable[str], by: list[str]) -> None:
    """Raise unless every `by` group scored every expected recording on all fingers, with equal frame counts."""
    want = set(recordings)
    for name, g in df.groupby(by):
        got = set(g.recording)
        if got != want:
            raise ValueError(f"{name}: missing {sorted(want - got)[:5]}, unexpected {sorted(got - want)[:5]}")
        if (g.groupby("recording").finger.nunique() != len(FINGERS)).any():
            raise ValueError(f"{name}: a recording is missing fingers")
    if (df.groupby(["recording", "finger"]).n.nunique() > 1).any():
        raise ValueError("scored frame counts differ between groups for the same recording")


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean of per-person values (people are the resampled unit)."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
