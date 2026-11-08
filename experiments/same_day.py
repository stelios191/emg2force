"""One new-day TCN per evaluation person, scored on the same day and on the new day.

    python -m experiments.same_day [--root ../hyser_raw]

Each fold trains the final new-day recipe (person's day 1 + 19 other people, as in calibration_curve.yaml) but without
the person's day-1 trials 4-5, then scores those (same day) and day-2 trials 4-5 (new day): the same model, people and
trial numbers on both days. Writes runs/same_day/metrics.csv, splits.json and env.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from emg2force.hyser import Hyser, recording_key
from emg2force.provenance import REPO
from emg2force.splits import DAY1, DAY2, EVAL_SUBJECTS, Split, make_splits, session_of, subject_of, trial_of
from experiments.metrics import recording_metrics
from experiments.run import config_sequences, env_info, fresh_run_dir, make_tcn, splits_record

CONFIG = REPO / "experiments" / "configs" / "calibration_curve.yaml"  # its recipe and splits
HELD_OUT_TRIALS = (4, 5)
RUN = "same_day"


def same_day_split(sp: Split) -> Split:
    """The fold without the person's day-1 trials 4-5, tested on those and on day-2 trials 4-5."""
    s = sp.held_out_subject
    held = {recording_key(s, DAY1, t) for t in HELD_OUT_TRIALS}
    train = tuple(k for k in sp.train if k not in held)
    test = tuple(recording_key(s, d, t) for d in (DAY1, DAY2) for t in HELD_OUT_TRIALS)
    if set(train) & set(test) or any(subject_of(k) == s and session_of(k) == DAY2 for k in train):
        raise ValueError(f"{sp.name}: same-day split leaks test recordings into training")
    return Split("same_day", sp.name, train, test, s)


def fold_rows(sp: Split, cfg: dict, ds: Hyser) -> list[dict]:
    s = sp.held_out_subject
    seqs = config_sequences(cfg, ds, sorted(set(sp.train) | set(sp.test)))
    model = make_tcn(cfg["tcn"], int(cfg["history"]), int(cfg.get("seed", 0)))
    t0 = time.time()
    model.fit([seqs[k] for k in sp.train], focus_subject=s)
    print(f"  {sp.name}: trained {time.time() - t0:.0f}s", flush=True)
    own = [seqs[k] for k in sp.train if subject_of(k) == s]
    const = np.median(np.concatenate([q.y[q.scored] for q in own]), 0)  # constant: median of the person's training labels
    preds = {
        k: {"tcn": model.predict(seqs[k].x), "constant": np.broadcast_to(const, seqs[k].y.shape), "zero": np.zeros(seqs[k].y.shape)}
        for k in sp.test
    }
    return [
        {"subject": s, "session": session_of(k), "trial": trial_of(k), "recording": k, "model": name, **m}
        for k in sp.test
        for name, pred in preds[k].items()
        for m in recording_metrics(seqs[k].y, pred, seqs[k].scored)
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    args = ap.parse_args(argv)
    env = env_info()  # at the start: the code that actually ran
    cfg = yaml.safe_load(CONFIG.read_text())
    ds = Hyser(args.root)
    splits = [same_day_split(sp) for sp in make_splits(cfg["protocol"], EVAL_SUBJECTS)]
    rows = [r for sp in splits for r in fold_rows(sp, cfg, ds)]
    with fresh_run_dir(REPO / "runs" / RUN) as out:
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        (out / "splits.json").write_text(json.dumps(splits_record(splits), indent=1))
        (out / "env.json").write_text(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
