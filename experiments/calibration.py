"""Labelled new-day calibration: the frozen new-day TCN fine-tuned, against ridges refitted on day 1 plus the same data.

    python -m experiments.calibration experiments/configs/calibration_curve.yaml [--root ../hyser_raw]

Per fold, one base TCN is trained and saved. It is fine-tuned on the first N s of day-2 trial 1 (scored on trials 2-5,
metrics.csv) and on whole day-2 trials 1, 1-2 and 1-3 (scored on trials 4-5, whole_trials.csv).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import yaml

from emg2force.dataset import DEFAULT_WARMUP_S, prefix_sequence
from emg2force.hyser import Hyser, recording_key
from emg2force.models import Predictor, RidgeModel, Sequence
from emg2force.provenance import REPO, git_state
from emg2force.splits import DAY1, DAY2, Split, calibration_key, check_calibration, make_splits, own_day1, subject_of
from emg2force.tcn import TCNModel
from experiments.metrics import check_complete, hierarchical_mean, recording_metrics
from experiments.run import (
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    check_config,
    config_sequences,
    env_info,
    filter_config,
    fresh_run_dir,
    front_end_of,
    make_tcn,
    recordings_on_disk,
    resolve_subjects,
    run_name,
    splits_record,
)

CALIBRATION_KEYS = frozenset({"input_norm", "budgets_s", "tcn", "ridge", "fine_tune"})  # on top of the run keys
FROZEN, FINE_TUNED = "tcn_frozen", "tcn_ft"
TEST_TRIALS = (4, 5)  # whole-trial calibration: scored on these day-2 trials
CAL_TRIALS = {25.0: (1,), 50.0: (1, 2), 75.0: (1, 2, 3)}  # labelled seconds -> day-2 trials used


def score(model: Predictor, method: str, budget: float, split: Split, seqs: dict[str, Sequence]) -> list[dict]:
    """Metrics of `model` on every test recording, labelled with method and budget."""
    label = {"split": split.name, "model": method, "budget_s": budget}
    return [
        {"subject": seqs[key].subject, "session": seqs[key].session, "recording": key, **m, **label}
        for key in split.test
        for m in recording_metrics(seqs[key].y, model.predict(seqs[key].x), seqs[key].scored)
    ]


def _prefix(cfg: dict, ds: Hyser, s: int, budget: float, input_norm: str) -> Sequence:
    return prefix_sequence(
        ds,
        calibration_key(s),
        budget,
        cfg["mvc_convention"],
        cfg["label_filter"],
        filter_config(cfg),
        cfg.get("warmup_s", DEFAULT_WARMUP_S),
        input_norm,
    )


def fine_tune(base: TCNModel, cal: list[Sequence], cfg: dict) -> TCNModel:
    return base.fine_tuned(cal, int(cfg["fine_tune"]["steps"]), float(cfg["fine_tune"]["lr"]))


def ridges(cfg: dict, fit_on: dict[str, list[Sequence]]) -> dict[str, RidgeModel]:
    """Each ridge baseline of the config, fitted on the sequences of its own input normalisation."""
    return {r["name"]: RidgeModel(int(cfg["history"]), alpha=float(r["alpha"])).fit(fit_on[r["input_norm"]]) for r in cfg["ridge"]}


def ridge_rows(cfg: dict, models: dict[str, RidgeModel], budget: float, split: Split, seqs: dict[str, dict[str, Sequence]]) -> list[dict]:
    return [row for r in cfg["ridge"] for row in score(models[r["name"]], r["name"], budget, split, seqs[r["input_norm"]])]


def short_fold(split: Split, seqs: dict, ds: Hyser, cfg: dict, save_dir: Path, log: Callable[..., None]) -> list[dict]:
    """Train one fold's base TCN, save it, and score it frozen and fine-tuned on the first N s of day-2 trial 1, with
    the ridges refitted on day 1 plus the same seconds. `seqs` is keyed by input normalisation."""
    s = split.held_out_subject
    check_calibration(split, [calibration_key(s)])
    base = make_tcn(cfg["tcn"], int(cfg["history"]), int(cfg.get("seed", 0)))
    t0 = time.time()
    base.fit([seqs[cfg["input_norm"]][k] for k in split.train], focus_subject=s)
    log(f"  {split.name}: base TCN {time.time() - t0:.0f}s")
    save_dir.mkdir(parents=True, exist_ok=True)
    base.save_state(save_dir / f"base_{split.name}.state.pt", {"training": f"runs/{run_name(save_dir.parent)}, fold {split.name}", **git_state()},
                    front_end_of(cfg))  # fmt: skip
    rows: list[dict] = []
    for budget in cfg["budgets_s"]:
        cal = {n: _prefix(cfg, ds, s, budget, n) for n in seqs} if budget > 0 else {}
        fit_on = {n: [seqs[n][k] for k in own_day1(split)] + ([cal[n]] if cal else []) for n in seqs}
        if cal:
            rows += score(fine_tune(base, [cal[cfg["input_norm"]]], cfg), FINE_TUNED, budget, split, seqs[cfg["input_norm"]])
        else:
            rows += score(base, FROZEN, 0.0, split, seqs[cfg["input_norm"]])
        refit = ridges(cfg, fit_on)
        rows += ridge_rows(cfg, refit, budget, split, seqs)
    return rows


def whole_trial_fold(split: Split, base: TCNModel, seqs: dict, cfg: dict) -> list[dict]:
    """The saved base TCN, frozen and fine-tuned on whole day-2 trials, and the ridges refitted on day 1 plus them."""
    s = split.held_out_subject
    day1 = [recording_key(s, DAY1, t) for t in range(1, 6)]
    rows = score(base, FROZEN, 0.0, split, seqs[cfg["input_norm"]]) + ridge_rows(
        cfg, ridges(cfg, {n: [seqs[n][k] for k in day1] for n in seqs}), 0.0, split, seqs
    )
    for budget, trials in CAL_TRIALS.items():
        cal_keys = [recording_key(s, DAY2, t) for t in trials]
        check_calibration(split, cal_keys)
        rows += score(
            fine_tune(base, [seqs[cfg["input_norm"]][k] for k in cal_keys], cfg), FINE_TUNED, budget, split, seqs[cfg["input_norm"]]
        )
        rows += ridge_rows(cfg, ridges(cfg, {n: [seqs[n][k] for k in day1 + cal_keys] for n in seqs}), budget, split, seqs)
    return rows


def run(cfg: dict, root: Path, out_dir: Path | None = None, quiet: bool = False) -> pd.DataFrame:
    """Both calibration tables of one config: metrics.csv (first seconds of trial 1) and whole_trials.csv."""
    env = env_info()  # at the start: the code that actually ran
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    check_config(cfg, REQUIRED_KEYS | CALIBRATION_KEYS, REQUIRED_KEYS | OPTIONAL_KEYS | CALIBRATION_KEYS | {"seed"})
    out_dir = out_dir or REPO / "runs" / cfg["name"]
    ds = Hyser(root)
    splits = make_splits(cfg["protocol"], resolve_subjects(cfg["subjects"]))
    keys = recordings_on_disk(splits, ds)
    norms = sorted({cfg["input_norm"]} | {r["input_norm"] for r in cfg["ridge"]})
    tcn_seqs = config_sequences(cfg, ds, keys)
    ridge_keys = sorted(
        {k for sp in splits for k in sp.test} | {k for sp in splits for k in sp.train if subject_of(k) == sp.held_out_subject}
    )
    seqs = {n: tcn_seqs if n == cfg["input_norm"] else config_sequences({**cfg, "input_norm": n}, ds, ridge_keys) for n in norms}
    own = sorted(recording_key(sp.held_out_subject, d, t) for sp in splits for d in (DAY1, DAY2) for t in range(1, 6))
    whole_seqs = {n: config_sequences({**cfg, "input_norm": n}, ds, own) for n in norms}  # both days of each held-out person
    short, whole = [], []
    with fresh_run_dir(out_dir) as work:
        for sp in splits:
            short += short_fold(sp, seqs, ds, cfg, work / "models", log)
            s = sp.held_out_subject
            scored = Split(sp.protocol, sp.name, sp.train, tuple(recording_key(s, DAY2, t) for t in TEST_TRIALS), s)
            whole += whole_trial_fold(scored, TCNModel.load_state(work / "models" / f"base_{sp.name}.state.pt"), whole_seqs, cfg)
        df, wf = pd.DataFrame(short), pd.DataFrame(whole)
        check_complete(df, {k for sp in splits for k in sp.test}, ["model", "budget_s"])
        cal = {
            sp.name: {f"{b:g}s": f"first {b:g} s of {calibration_key(sp.held_out_subject)}" for b in cfg["budgets_s"] if b > 0}
            for sp in splits
        }
        (work / "splits.json").write_text(json.dumps(splits_record(splits, cal), indent=1))
        df.to_csv(work / "metrics.csv", index=False)
        wf.to_csv(work / "whole_trials.csv", index=False)
        (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (work / "env.json").write_text(json.dumps(env, indent=2))
    for name, d in (("first seconds of trial 1", df), ("whole trials", wf)):
        table = hierarchical_mean(d, "mae", ["model", "budget_s"]).groupby(["model", "budget_s"]).mae.mean().unstack("budget_s")
        log(f"\nMAE (%MVC), {name}, by labelled seconds\n" + table.to_string(float_format=lambda v: f"{v:.2f}"))
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(int(cfg.get("threads", 4)))
    run(cfg, args.root)


if __name__ == "__main__":
    main()
