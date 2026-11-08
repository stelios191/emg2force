"""The streaming input monitor: false flags on clean recordings, and what it saves when electrodes fail.

    python -m experiments.monitor_check runs/final_cross_day_pop_256 [--root ../hyser_raw]

The run needs every fold model (save_models: all). Each person's day-2 recordings go through their own fold model.

1. Clean recordings: the share of channel-frames the monitor replaces, and the MAE with the monitor off (what the
   offline evaluation scores) and on (what the streaming decoder outputs by default).
2. Faults: from 5 s, (a) a row of 8 electrodes reads zero, (b) one channel is stuck at a constant, (c) one channel
   picks up noise at 100x its own level, (d) 100 ms of NaN on every channel. MAE against the labels with the monitor
   off and on, and frames from the fault to the first flag.
Writes <run>/monitor_check.csv (faults) and <run>/monitor_false_flags.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from emg2force.hyser import FS_EMG, FS_FORCE, Hyser, emg_count_at
from emg2force.pipeline import StreamingDecoder, load_decoder
from emg2force.provenance import REPO
from emg2force.splits import DAY2, TRIALS
from experiments.run import fold_checkpoints

FAULT_S = 5.0
FAULTS = ("dead_row", "stuck_channel", "noisy_channel", "nan_burst")
FAULT_CHANNELS = {"dead_row": slice(64, 72), "stuck_channel": slice(100, 101), "noisy_channel": slice(100, 101)}  # nan_burst: non-finite
NOISE_GAIN = 100.0  # the noisy channel's added noise, in multiples of its own RMS before the fault


def inject(emg: np.ndarray, fault: str) -> np.ndarray:
    out = emg.copy()
    k = int(FAULT_S * FS_EMG)
    if fault == "dead_row":
        out[k:, 64:72] = 0.0  # one row of the extensor-proximal grid
    elif fault == "stuck_channel":
        out[k:, 100] = out[k, 100]
    elif fault == "noisy_channel":  # a loose contact: broadband noise far above the channel's own level
        rms = float(np.sqrt(np.mean(out[:k, 100].astype(np.float64) ** 2)))
        out[k:, 100] += (NOISE_GAIN * rms * np.random.default_rng(0).standard_normal(len(out) - k)).astype(out.dtype)
    elif fault == "nan_burst":
        out[k : k + int(0.1 * FS_EMG)] = np.nan
    else:
        raise ValueError(fault)
    return out


def stream(decoder: StreamingDecoder, emg: np.ndarray, monitor: bool) -> tuple[np.ndarray, np.ndarray]:
    """Predictions and the monitor's channel mask per frame. EMG is pushed one frame at a time so the mask is exact."""
    decoder.monitor = monitor
    decoder.reset()
    n_frames = int(np.searchsorted(emg_count_at(np.arange(len(emg))), len(emg), side="right"))  # frames whose EMG fits
    ends = emg_count_at(np.arange(n_frames))
    out, flags, start = [], [], 0
    for end in ends:
        for _, p in decoder.push(emg[start:end]):
            out.append(p)
            flags.append(decoder.replaced.copy())
        start = end
    return np.array(out), np.array(flags)


def first_flag(flags: np.ndarray, channels: slice) -> int | None:
    """Frames from the fault to the first frame flagging any of `channels` (None if never)."""
    k = int(FAULT_S * FS_FORCE)
    hits = np.flatnonzero(flags[k:, channels].any(1))
    return int(hits[0]) if len(hits) else None


def clean_rows(run_dir: Path, ds: Hyser, cfg: dict) -> pd.DataFrame:
    """Clean day-2 recordings, each through the person's own fold model: the share of
    channel-frames the monitor replaces, and the MAE with the monitor off (= the offline evaluation) and on."""
    rows = []
    for subject, path in fold_checkpoints(run_dir, "tcn"):
        decoder = load_decoder(path)
        for t in TRIALS:
            rec = ds.load(subject, DAY2, t)
            y = ds.target(rec, cfg["mvc_convention"], cfg["label_filter"]).values
            row: dict[str, float] = {"subject": subject, "trial": t}
            for monitor in (False, True):
                pred, flags = stream(decoder, rec.emg, monitor)
                n = min(len(pred), len(y))
                row[f"mae_{'on' if monitor else 'off'}"] = float(np.nanmean(np.abs(pred[:n] - y[:n])))
                if monitor:
                    row["flagged_share"] = float(flags[decoder.n_base :].mean())  # channel-frames replaced after the warm-up
            rows.append(row)
    return pd.DataFrame(rows)


def faults(run_dir: Path, ds: Hyser, cfg: dict) -> pd.DataFrame:
    rows = []
    for subject, path in fold_checkpoints(run_dir, "tcn"):
        decoder = load_decoder(path)
        for t in TRIALS:
            rec = ds.load(subject, DAY2, t)
            y = ds.target(rec, cfg["mvc_convention"], cfg["label_filter"]).values
            clean, _ = stream(decoder, rec.emg, monitor=True)
            n = min(len(clean), len(y))
            after = slice(int(FAULT_S * FS_FORCE), n)
            row: dict[str, float | None] = {"subject": subject, "trial": t, "clean": float(np.nanmean(np.abs(clean[after] - y[after])))}
            for fault in FAULTS:
                bad = inject(rec.emg, fault)
                for monitor in (False, True):
                    pred, flags = stream(decoder, bad, monitor)
                    row[f"{fault}_{'on' if monitor else 'off'}"] = float(np.nanmean(np.abs(pred[after] - y[after])))
                    if monitor and fault in FAULT_CHANNELS:
                        row[f"{fault}_flag_frames"] = first_flag(flags, FAULT_CHANNELS[fault])
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="run with every fold model")
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    args = ap.parse_args(argv)
    ds = Hyser(args.root)
    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    ff = clean_rows(args.run_dir, ds, cfg)
    ff.to_csv(args.run_dir / "monitor_false_flags.csv", index=False)
    fd = faults(args.run_dir, ds, cfg)
    fd.to_csv(args.run_dir / "monitor_check.csv", index=False)
    print(
        f"clean recordings: {ff.flagged_share.mean():.2e} of channel-frames flagged ({(ff.flagged_share > 0).sum()} of {len(ff)} "
        f"recordings), MAE monitor off {ff.mae_off.mean():.3f}, on {ff.mae_on.mean():.3f}"
    )
    print(fd.drop(columns=["subject", "trial"]).mean().round(2).to_string())


if __name__ == "__main__":
    main()
