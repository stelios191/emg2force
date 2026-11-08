"""Demo video and README GIF, with the recording chosen by a fixed rule and every card number read from the runs.

    python -m experiments.demo_video [--root ../hyser_raw]

Rule (final new-day TCN, seed 0): the person with the median error of the 16. Among their day-2 recordings with an
error within 10 % of theirs, the one with the most grip activity. For the GIF, the 10 s of it with the most grip
activity whose error is also within 10 %. If none is within 10 %, the closest recording or window. The video shows
that recording decoded first by the TCN trained on the person's day 1 only, then by the final one. Writes results/demo.mp4, results/demo.gif and results/new_day_trial.png
(the whole recording, every finger).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from emg2force.dataset import warmup_frames
from emg2force.gripper import GripMapping
from emg2force.hyser import FINGERS, FS_FORCE, Hyser, parse_key
from emg2force.provenance import REPO
from experiments import render
from experiments.metrics import hierarchical_mean
from experiments.report import AFTER, FIRST_TCN_RUNS, RUNS, TCN_RUNS, frames, load, median_person, per_person

FINAL, FIRST, BASELINES = TCN_RUNS[0], FIRST_TCN_RUNS[0], "final_cross_day_pop_256_ridge"
WEIGHTED_RIDGE = "final_cross_day_pop_256_ridge_weighted"


WITHIN = 0.10  # a recording or window counts as typical if its error is within 10 % of the person's


def _grip(ds: Hyser, key: str, cfg: dict) -> np.ndarray:
    """Grip target (N) from the measured force, from the end of the warm-up on."""
    s, d, t = parse_key(key)
    y = ds.target(ds.load_force(s, d, t), cfg["mvc_convention"], cfg["label_filter"]).values
    return GripMapping()(y)[warmup_frames(cfg) :]


def demo_recording(root: Path, run: str = FINAL) -> tuple[str, float]:
    """(recording, start of the GIF window in seconds) by the rule in the module docstring."""
    cfg = yaml.safe_load((RUNS / run / "config.yaml").read_text())
    df = load(run)
    df = df[(df.model == "tcn") & (df.seed == 0)]
    s, person = median_person(df)
    errors = df[df.subject == s].groupby("recording").mae.mean()
    typical = errors[(errors - person[s]).abs() <= WITHIN * person[s]]
    if typical.empty:  # no recording within 10 %: the closest one
        typical = errors[[(errors - person[s]).abs().idxmin()]]
    ds = Hyser(root)
    n = int(render.PAGE_S * FS_FORCE)
    key = max(typical.index, key=lambda k: np.convolve(_grip(ds, k, cfg) - 1.0, np.ones(n), "valid").max())
    with np.load(RUNS / run / "predictions.npz") as z:
        yhat = z[f"tcn|0|{key}"]
    s_, d, t = parse_key(key)
    start = warmup_frames(cfg)
    y = ds.target(ds.load_force(s_, d, t), cfg["mvc_convention"], cfg["label_filter"]).values[start:]
    err = np.abs(yhat[start : start + len(y)] - y).mean(axis=1)
    grip = _grip(ds, key, cfg)
    starts = range(0, len(grip) - n + 1, 10)
    windows = [a for a in starts if abs(err[a : a + n].mean() - person[s]) <= WITHIN * person[s]]
    if not windows:  # no window within 10 %: the closest one
        windows = [min(starts, key=lambda a: abs(err[a : a + n].mean() - person[s]))]
    a = max(windows, key=lambda a: (grip[a : a + n] - 1.0).sum())
    return str(key), (a + start) / FS_FORCE


def trial_plot(root: Path, key: str, out: Path, run: str = FINAL) -> None:
    """The whole recording: each finger's measured force and the final TCN's output (seed 0), with each finger's error."""
    cfg = yaml.safe_load((RUNS / run / "config.yaml").read_text())
    s, d, t = parse_key(key)
    ds = Hyser(root)
    y = -ds.target(ds.load_force(s, d, t), cfg["mvc_convention"], cfg["label_filter"]).values  # flexion up, as in the demo
    with np.load(RUNS / run / "predictions.npz") as z:
        yhat = -z[f"tcn|0|{key}"][: len(y)]
    start = warmup_frames(cfg)
    time = np.arange(len(y)) / FS_FORCE
    mean = per_person(frames((run,), "tcn")).mean()  # the same seed as the recording
    fig, axes = plt.subplots(5, 1, figsize=(8.8, 6.6), dpi=200, sharex=True)
    for f, ax in enumerate(axes):
        ax.axvspan(0, start / FS_FORCE, color="#F2F2F2", lw=0)
        ax.axhline(0, color="#CCCCCC", lw=0.6)
        ax.plot(time, y[:, f], color="black", lw=1.2, label="measured")
        ax.plot(time[start:], yhat[start:, f], color=AFTER, lw=1.0, label="decoded")
        ax.set_ylabel(FINGERS[f], fontsize=10, rotation=0, ha="right", va="center")
        ax.text(1.005, 0.5, f"MAE {np.abs(yhat[start:, f] - y[start:, f]).mean():.1f}", transform=ax.transAxes, fontsize=8.5,
                color="#555555", va="center")  # fmt: skip
        ax.tick_params(labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.35))
    axes[-1].set_xlabel("time in the recording (s), first 0.7 s warm-up (grey)", fontsize=10)
    mae = recording_mae(run, "tcn", key)
    fig.text(0.015, 0.975, "One new-day recording, every finger", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.015, 0.917, f"Person {s} (the median person), day {d}, trial {t}, the demo's recording, decoded by the final TCN.\n"
             f"Force in %MVC, flexion up. This recording's MAE {mae:.2f}, the mean over 16 people {mean:.2f} (same seed).",
             fontsize=9.5, color="#555555", ha="left", linespacing=1.4)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 0.96, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def recording_mae(run: str, model: str, key: str) -> float:
    df = load(run)
    return float(df[(df.model == model) & (df.seed == 0) & (df.recording == key)].mae.mean())


def calibration_75() -> tuple[float, float] | None:
    """Fine-tuned TCN and refitted ridge with 75 s of labelled new-day data (day-2 trials 1-3, scored on 4-5)."""
    f = RUNS / "calibration_curve" / "whole_trials.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df.assign(model=df.model.where(~df.model.str.startswith("tcn_ft"), "tcn_ft"))  # the fine-tuned TCN
    per = hierarchical_mean(df, "mae", ["model", "budget_s"]).groupby(["model", "budget_s"]).mae.mean()
    return float(per["tcn_ft", 75.0]), float(per["ridge", 75.0])


def cards(key: str) -> dict[str, str]:
    """Card text, lines separated by |."""
    first, final = per_person(frames(FIRST_TCN_RUNS, "tcn")), per_person(frames(TCN_RUNS, "tcn"))
    zero = per_person(frames((BASELINES,), "zero")).mean()
    ridge = per_person(frames((WEIGHTED_RIDGE,), "ridge")).mean()
    better = int(((final - first) < 0).sum())
    s, d, t = parse_key(key)
    result = [
        "Result",
        f"16 evaluation people: {first.mean():.2f} → {final.mean():.2f} %MVC, better for {better} of {len(final)}, no new-day labels",
        f"Ridge on the same recordings and person weighting: {ridge:.2f}. Predicting zero force: {zero:.2f}",
        f"This recording: {recording_mae(FIRST, 'tcn', key):.2f} → {recording_mae(FINAL, 'tcn', key):.2f}",
    ]
    if (cal := calibration_75()) is not None:
        result.append(f"75 s of new-day labels (trials 1-3, scored on 4-5): fine-tuned TCN {cal[0]:.2f}, refitted ridge {cal[1]:.2f}")
    return {
        "intro": "Streaming HD-sEMG → five finger forces → simulated gripper|"
        "256-channel forearm EMG (public Hyser data), causal decoder updating every 10 ms|"
        f"A typical held-out recording (the median person): person {s}, day {d}, trial {t}. Recorded EMG replayed",
        "problem": "The problem|Day 1: the model learns from this person. Day 2: the electrodes have been re-applied.|"
        f"Trained on day 1 only: {first.mean():.2f} %MVC error on the new day. Predicting zero force: {zero:.2f}",
        "fix": "The fix|Also train on 19 other people's recordings, with simulated electrode shifts|"
        "Chosen on 4 development people, then scored on the 16",
        "result": "|".join(result),
        "limits": "Limits|Lab recordings of isometric presses. Force relative to each person's maximum (%MVC)|"
        "Recorded EMG replayed into a simulator: not live control or hardware",
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "demo.mp4")
    ap.add_argument("--gif", type=Path, default=REPO / "results" / "demo.gif")
    ap.add_argument("--trial", type=Path, default=REPO / "results" / "new_day_trial.png")
    args = ap.parse_args(argv)
    key, gif_from = demo_recording(args.root)
    trial_plot(args.root, key, args.trial)
    c = cards(key)
    render.write_video(
        args.out,
        args.root,
        [
            ("card", c["intro"].split("|"), 4.0),
            ("card", c["problem"].split("|"), render.CARD_SECONDS),
            ("clip", RUNS / FIRST, "tcn", key, "Trained on this person's day 1 only"),
            ("card", c["fix"].split("|"), render.CARD_SECONDS),
            ("clip", RUNS / FINAL, "tcn", key, "Trained on day 1 plus 19 other people, with simulated electrode shifts"),
            ("card", c["result"].split("|"), render.CARD_SECONDS),
            ("card", c["limits"].split("|"), render.CARD_SECONDS),
        ],
    )
    print(f"wrote {args.out}")
    render.render_gif(RUNS / FINAL, key, args.root, args.gif, gif_from)
    print(f"wrote {args.gif} ({key}, from {gif_from:.1f} s)")


if __name__ == "__main__":
    main()
