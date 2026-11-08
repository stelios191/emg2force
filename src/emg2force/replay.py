"""Replay saved predictions and reference labels through the simulated gripper. Report force errors in N.

  python -m emg2force.replay --run runs/final_cross_day_pop_256 --models tcn --out results/mujoco_cross_day_pop

Errors: decoder (setpoint vs reference setpoint), tracking (contact vs setpoint), whole chain (contact vs reference).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from emg2force.dataset import warmup_frames
from emg2force.gripper import GripMapping, GripperSim, response_times
from emg2force.hyser import Hyser, parse_key, recording_key
from emg2force.provenance import REPO
from emg2force.splits import EVAL_SUBJECTS

UNDER_OVER_TOL = 1.0  # N, tolerance for the under-/over-grip time shares
SUMMARY_COLUMNS = [
    "decoder_mae_N",
    "tracking_mae_N",
    "whole_chain_mae_N",
    "whole_chain_bias_N",
    "time_under_grip",
    "time_over_grip",
]
RECORDINGS = [recording_key(s, 2, 1) for s in EVAL_SUBJECTS]  # fixed before any replay
STEPS = ((1.0, 8.0), (8.0, 1.0), (1.0, 15.0), (15.0, 1.0))  # N: setpoint steps for the gripper's response time


def error_row(contact: np.ndarray, setpoint: np.ndarray, reference: np.ndarray) -> dict:
    """Decoder, controller and whole-chain errors of one replay against the reference setpoint."""
    over = contact - reference
    return {
        "decoder_mae_N": float(np.abs(setpoint - reference).mean()),
        "tracking_mae_N": float(np.abs(contact - setpoint).mean()),
        "whole_chain_mae_N": float(np.abs(over).mean()),
        "whole_chain_bias_N": float(over.mean()),
        "time_under_grip": float(np.mean(over < -UNDER_OVER_TOL)),
        "time_over_grip": float(np.mean(over > UNDER_OVER_TOL)),
    }


def setpoints(ds: Hyser, cfg: dict, preds: Mapping[str, np.ndarray], key: str, models: Iterable[str], start: int) -> dict[str, np.ndarray]:
    """Grip setpoints (N) from the reference label and each model's saved predictions (seed 0)."""
    mapping = GripMapping()
    s, d, t = parse_key(key)
    y = ds.target(ds.load_force(s, d, t), cfg["mvc_convention"], cfg["label_filter"]).values
    out = {"reference": mapping(y)[start:]}
    for name in models:
        out[name] = mapping(preds[f"{name}|0|{key}"].astype(np.float64))[start:]
    return out


def write_summary(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    df.to_csv(out / "mujoco_per_recording.csv", index=False)
    summ = df.groupby("source")[SUMMARY_COLUMNS].mean().reset_index()
    summ.to_csv(out / "mujoco_summary.csv", index=False)
    return summ


def run(run_dir: Path, root: Path, out: Path, models: Iterable[str] = ("ridge", "tcn")) -> pd.DataFrame:
    """Replay every fixed recording for the reference and `models`."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    with np.load(run_dir / "predictions.npz") as z:
        preds = dict(z)
    ds = Hyser(root)
    start = warmup_frames(cfg)
    rows = []
    sim = GripperSim()
    sim.calibrate()
    for key in RECORDINGS:
        sources = setpoints(ds, cfg, preds, key, models, start)
        for src, sp in sources.items():
            r = sim.replay(sp)
            rows.append(
                {"recording": key, "subject": parse_key(key)[0], "source": src, **error_row(r["contact"], sp, sources["reference"])}
            )
    out.mkdir(parents=True, exist_ok=True)
    summ = write_summary(pd.DataFrame(rows), out)
    return summ


def step_responses(out: Path) -> pd.DataFrame:
    """The closed loop's response to setpoint steps (no EMG): how fast the gripper follows a new force."""
    sim = GripperSim()
    sim.calibrate()
    df = pd.DataFrame([{"from_N": a, "to_N": b, **response_times(sim.step_response(a, b), a, b)} for a, b in STEPS])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "step_response.csv", index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "mujoco")
    ap.add_argument("--models", nargs="+", default=["ridge", "tcn"])
    args = ap.parse_args(argv)
    print(run(args.run, args.root, args.out, tuple(args.models)).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(step_responses(args.out).to_string(index=False))


if __name__ == "__main__":
    main()
