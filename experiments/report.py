"""docs/experiments.md and assets/new_day_per_person.png from the runs: the tables behind the README.

    python -m experiments.report [--allow-partial]

Every number is computed from runs/ and results/, except the two tables at the end, which summarise development runs
and dropped ideas kept outside the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emg2force.pipeline import DROP_LOG, RISE_LOG
from emg2force.provenance import REPO
from experiments.hold_test import MIN_EARLY
from experiments.metrics import boot_ci, hierarchical_mean
from experiments.monitor_check import FAULTS

RUNS, RES, DOCS, ASSETS = REPO / "runs", REPO / "results", REPO / "docs", REPO / "assets"
TCN_RUNS = ("final_cross_day_pop_256", "final_cross_day_pop_256_seeds")  # final new-day recipe, seeds 0-4
SEEDS = 5  # seeds of each new-day network, checked before the page is written
FIRST_TCN_RUNS = ("final_cross_day_256", "final_cross_day_256_seeds")  # first recipe (person's day 1 only), seeds 0-4

# label -> (runs, model column). Seeds of one model are averaged per person before any comparison.
Entry = tuple[tuple[str, ...], str]
NEW_DAY: dict[str, Entry] = {
    "zero force": (("final_cross_day_pop_256_ridge",), "zero"),
    "ridge": (("final_cross_day_pop_256_ridge",), "ridge"),
    "ridge, person-weighted": (("final_cross_day_pop_256_ridge_weighted",), "ridge"),
    "LSTM (5 seeds)": (("final_cross_day_pop_256_lstm",), "lstm"),
    "TCN (5 seeds)": (TCN_RUNS, "tcn"),
}
NEW_PERSON: dict[str, Entry] = {
    "zero force": (("final_loso_256",), "zero"),
    "ridge": (("final_loso_256",), "ridge"),
    "TCN": (("final_loso_shift_256",), "tcn"),
}
Step = tuple[str, tuple[str, ...], str]  # (recipe, runs, model column), seed 0
NEW_DAY_STEPS: list[Step] = [
    ("TCN on the person's day 1 (running-mean inputs)", ("final_cross_day_256",), "tcn"),
    ("+ 19 other people, electrode shifts and CAR", ("final_cross_day_pop_256_running_mean",), "tcn"),
    ("− running mean (final recipe)", ("final_cross_day_pop_256",), "tcn"),
]
NEW_PERSON_STEPS: list[Step] = [
    ("TCN on 19 other people (running-mean inputs)", ("final_loso_256",), "tcn"),
    ("+ electrode shifts and CAR", ("final_loso_shift_256_running_mean",), "tcn"),
    ("− running mean (final recipe)", ("final_loso_shift_256",), "tcn"),
]
HOLD_RUNS = {"no running mean (final)": "final_cross_day_pop_256", "running mean (previous)": "final_cross_day_pop_256_running_mean"}
BEHAVIOUR_MODELS = {"tcn": "TCN", "ridge": "ridge, person-weighted"}
CALIBRATION_RUN = "calibration_curve"
CALIBRATION_LABELS = {"tcn_frozen": "TCN, frozen", "tcn_ft": "TCN fine-tuned", "ridge": "ridge refit, no normalisation",
                      "ridge_rm": "ridge refit, running mean"}  # fmt: skip
FAULT_LABELS = {
    "dead_row": "a row of 8 electrodes reads zero",
    "stuck_channel": "one channel stuck at a constant",
    "noisy_channel": "one channel picks up noise at 100x its level",
    "nan_burst": "100 ms of NaN on every channel",
}
LATENCY_FILES = [f"latency_256ch_recorded_rep{i}.json" for i in (1, 2, 3)]  # emg2force.benchmark output in results/latency
LATENCY_MODES = {
    "model_only": "network only (ONNX)",
    "pipeline_sustained": "whole update, back to back (ONNX)",
    "pipeline_paced_sleep": "whole update, paced every 10 ms (ONNX)",
    "pipeline_paced_sleep_pytorch": "whole update, paced every 10 ms (PyTorch)",
}
SIM_RUNS = {  # label -> (emg2force.replay output folder, decoder column)
    "ridge (person's day 1 only)": ("mujoco_cross_day", "ridge"),
    "TCN, person's day 1 only": ("mujoco_cross_day", "tcn"),
    "TCN, final recipe": ("mujoco_cross_day_pop", "tcn"),
}
STEP_FILE = "mujoco_cross_day_pop/step_response.csv"

SELECTION = [  # (setting, tried, rule, kept)
    (
        "TCN regularisation",
        "dropout 0.1, weight decay 1e-4, no normalisation: 14.08",
        "0.30 rule",
        "dropout 0.3, weight decay 1e-2, running mean: 9.15",
    ),
    (
        "other people's data",
        "person's day 1 only: 9.15",
        "lowest mean, set after the first recipe was scored on the 16",
        "+ 19 other people, half of each batch from the person: 8.46",
    ),
    ("electrode-shift augmentation", "none: 8.46", "the same", "shifts up to 1 electrode: 8.24"),
    ("electrodes", "64: 8.30", "the same", "256: 8.24"),
    (
        "input normalisation",
        "running mean: 8.24, held press 0.23",
        "holds a press (ratio ≥ 0.7), at most 0.30 worse",
        "none: 7.66, held press 1.00",
    ),
    (
        "training length and dropout",
        "1000 to 4000 steps, dropout 0.1 or 0.3: 7.67 to 8.10",
        "change only if 0.30 lower",
        "2000 steps, dropout 0.3: 7.66",
    ),
    ("network", "causal LSTM: 8.20", "lower mean, holds a press, streams in under 10 ms", "TCN: 7.80 (3 seeds each, lower for 4 of 4)"),
    (
        "spatial montage",
        "as recorded: 7.80, row differences: 8.33",
        "0.30 rule",
        "CAR: 7.46. Rerun without the evaluation people's day 2: 7.72 against 7.78, kept, the 16 had already been scored with it",
    ),
    (
        "calibration",
        "head-only, longer or faster fine-tuning: 7.21 to 8.17",
        "lowest mean over 5, 10 and 25 s",
        "all layers, learning rate 1e-4, 100 steps: 7.00",
    ),
]
DROPPED = [  # (idea, tested on, result, compared with)
    ("640 ms window instead of 320 ms", "the 16, seed 0", "+0.01 (−0.07 to +0.11)", "TCN"),
    ("per-finger output gain, 25 s labelled", "the 16", "−0.14 (−0.30 to −0.01)", "frozen TCN"),
    ("warm-up mean, or running mean and SD, as input normalisation", "development, ridge", "+0.05 to +0.65", "running mean"),
    ("slow exponential mean (30 s) or rest tracker as normalisation", "development", "10.87 and 11.16", "none 7.71"),
    ("channel dropout on top of electrode shifts", "development, new person", "9.99", "shifts only 9.73"),
    ("random channel masks in ridge training", "development", "9.56", "ridge 9.81"),
    ("day-2 input alignment: per-channel z-score or CORAL", "development", "9.59", "ridge 9.81"),
    ("electrode-shift estimation from activation maps", "development", "9.78", "ridge 9.81"),
    ("non-negative factorisation, components refitted on day 2", "development", "10.21", "ridge 9.81"),
    ("motor-unit decoding with day-1 separation filters", "development", "worse than ridge for 2 of 3", "ridge"),
    ("input standardisation re-estimated on unlabelled day-2 EMG", "development", "8.97", "frozen TCN 8.48"),
    ("alignment of the frozen TCN's internal features", "development", "8.88", "frozen TCN 8.48"),
    ("per-channel input scale and bias, 25 s labelled", "development", "7.95", "fine-tune 7.80"),
    ("learned 256 × 256 input map, 25 s labelled", "development", "7.60", "fine-tune 7.80"),
    ("5 × 5 output re-mix, 75 s labelled", "the 16", "+0.11, better for 7 of 16", "frozen TCN"),
    ("average of the frozen TCN and a CAR ridge", "the 16", "+0.38, better for 4 of 16", "frozen TCN"),
]


def markdown_table(df: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    """GitHub-flavoured table. Floats formatted with `float_fmt`, everything else with str()."""

    def cell(v: object) -> str:
        return float_fmt.format(v) if isinstance(v, float | np.floating) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / name / "metrics.csv")


def has_run(run: str) -> bool:
    return (RUNS / run / "metrics.csv").exists()


def frames(runs: tuple[str, ...], model: str) -> pd.DataFrame | None:
    """The metric rows of one model over runs holding different seeds of it (None if a run or the model is missing)."""
    if not all(has_run(r) for r in runs):
        return None
    df = pd.concat([load(r) for r in runs])
    df = df[df.model == model]
    return df if len(df) else None


def per_person(df: pd.DataFrame, metric: str = "mae") -> pd.Series:
    """Fingers, then recordings (and seeds), then days, then one value per person."""
    return hierarchical_mean(df, metric).set_index("subject")[metric]


def median_person(df: pd.DataFrame) -> tuple[int, pd.Series]:
    """The person with the median error (the lower middle of an even count) and every person's mean error."""
    person = per_person(df).sort_values()
    return int(person.index[(len(person) - 1) // 2]), person


def paired(a: pd.Series, b: pd.Series) -> str:
    """Mean of a − b over people, 95 % person-bootstrap CI, and people for whom a is lower."""
    d = (a - b).dropna()
    if len(d) != len(a):
        raise ValueError("paired comparison over different people")
    lo, hi = boot_ci(d.values)
    mean, lo, hi = (round(v, 2) + 0.0 for v in (d.mean(), lo, hi))  # + 0.0: no "-0.00"
    return f"{mean:+.2f} ({lo:+.2f} to {hi:+.2f}), {int((d < 0).sum())}/{len(d)}"


def results_table(entries: dict[str, Entry], reference: str) -> list[str]:
    """One row per model: MAE ± SD over people, other metrics, and the paired difference from `reference`."""
    data = {label: frames(runs, m) for label, (runs, m) in entries.items()}
    if any(df is None for df in data.values()):  # a missing run leaves the section empty, so the report stops
        return []
    ref = per_person(data[reference])
    rows = []
    for label, df in data.items():
        mae, r = per_person(df), per_person(df, "r")
        rows.append(
            {
                "model": label,
                "MAE ± SD": f"{mae.mean():.2f} ± {mae.std():.2f}",
                "R²": f"{per_person(df, 'r2').mean():.2f}",
                "r": f"{r.mean():.2f}" if r.notna().all() else "–",
                "MAE at rest": f"{per_person(df, 'mae_rest').mean():.2f}",
                f"− {reference}": "–" if label == reference else paired(mae, ref),
            }
        )
    return [markdown_table(pd.DataFrame(rows)), ""]


def steps_table(steps: list[Step]) -> list[str]:
    """Each change against the step before it, same people and seed."""
    data = [(label, frames(runs, m)) for label, runs, m in steps]
    if any(df is None for _, df in data):
        return []
    rows, prev = [], None
    for label, df in data:
        mae = per_person(df)
        rows.append({"recipe": label, "MAE": mae.mean(), "− previous": "–" if prev is None else paired(mae, prev)})
        prev = mae
    return [markdown_table(pd.DataFrame(rows)), ""]


def new_day_section() -> list[str]:
    table = results_table(NEW_DAY, "ridge, person-weighted")
    tcn, first, lstm = frames(TCN_RUNS, "tcn"), frames(FIRST_TCN_RUNS, "tcn"), frames(NEW_DAY["LSTM (5 seeds)"][0], "lstm")
    mvc_file = RUNS / "behaviour" / "mvc.csv"
    unseen, own = frames(("final_loso_shift_256",), "tcn"), frames((TCN_RUNS[0],), "tcn")
    if unseen is None or own is None:
        return []
    unseen = unseen[unseen.session == 2]  # the same day-2 recordings
    if not table or not mvc_file.exists() or any(df is None or df.seed.nunique() != SEEDS for df in (tcn, first, lstm)):
        return []
    mvc = hierarchical_mean(pd.read_csv(mvc_file), "mae_day1", ["model"]).groupby("model").mae_day1.mean()
    return [
        "## New day, no labels from it",
        "",
        "Trained on the person's day 1 and both days of 19 other people, scored on the five day-2 trials. Both ridges",
        "use running-mean inputs. The person-weighted one gives the person the TCN's share of each batch. MAE at rest:",
        "frames with the label below 5 %MVC.",
        "",
        *table,
        f"Trained on the person's day 1 only (5 seeds): {per_person(first).mean():.2f}. Final − that: "
        f"{paired(per_person(tcn), per_person(first))}.",
        "",
        f"Never trained on the person (the new-person TCN, seed 0): {per_person(unseen).mean():.2f} on the same recordings.",
        f"Adding their day 1 (seed 0): {paired(per_person(own), per_person(unseen))}.",
        "",
        "Labels use each day's own maximal-force (MVC) recordings. A user would have only day 1's: rescored with them",
        f"(seed 0), the TCN scores {mvc['tcn']:.2f}, the person-weighted ridge {mvc['ridge']:.2f} and zero force {mvc['zero']:.2f}.",
        "",
    ]


def new_person_section() -> list[str]:
    table = results_table(NEW_PERSON, "ridge")
    if not table:
        return []
    return ["## New person, no data from them", "", "Trained on both days of 19 other people. TCN: seed 0.", "", *table]


def steps_section() -> list[str]:
    day, person = steps_table(NEW_DAY_STEPS), steps_table(NEW_PERSON_STEPS)
    if not (day and person):
        return []
    return ["## What each change contributed", "", "Same people, seed 0.", "", "New day:", "", *day, "New person:", "", *person]


def same_day_section() -> list[str]:
    """One TCN per person on held-out day-1 trials and on day-2 trials with the same numbers."""
    if not has_run("same_day"):
        return []
    df = load("same_day")
    per = {(m, d): per_person(df[(df.model == m) & (df.session == d)]) for m in ("tcn", "constant") for d in (1, 2)}
    rows = [
        {
            "scored on": label,
            "TCN MAE": per["tcn", d].mean(),
            "constant MAE": per["constant", d].mean(),
            "TCN / constant": (per["tcn", d] / per["constant", d]).mean(),
        }
        for d, label in ((1, "same day (day-1 trials 4-5)"), (2, "new day (day-2 trials 4-5)"))
    ]
    d = (per["tcn", 2] - per["tcn", 1]).dropna()
    lo, hi = boot_ci(d.values)
    return [
        "## Same day against new day",
        "",
        "Each person's TCN trains without their day-1 trials 4-5 and is scored on those and on day-2 trials 4-5.",
        "Constant = the median of the person's own training labels. Day-2 trials contain more force, hence the ratio.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        f"New day − same day: {d.mean():+.2f} ({lo:+.2f} to {hi:+.2f}), higher for {int((d > 0).sum())}/{len(d)}.",
        "",
    ]


def hold_section() -> list[str]:
    """Held-press test (experiments.hold_test): does the decoded force stay up through a 12 s press?"""
    files = {label: RUNS / run / "hold_test.csv" for label, run in HOLD_RUNS.items()}
    if not all(f.exists() for f in files.values()):
        return []
    rows = []
    for label, f in files.items():
        every = pd.read_csv(f)
        df = every.dropna(subset=["ratio"])  # presses with no usable early output give no ratio
        person = df.groupby("subject").ratio.median()
        rows.append(
            {
                "TCN inputs": label,
                "median ratio": df.ratio.median(),
                "presses ≥ 0.7": f"{(df.ratio >= 0.7).mean():.0%}",
                "people with median ≥ 0.7": f"{int((person >= 0.7).sum())}/{len(person)}",
                f"presses used (early output ≥ {MIN_EARLY:g} %MVC)": f"{len(df)}/{len(every)}",
            }
        )
    return [
        "## A held press",
        "",
        "Hyser presses last a few seconds. Each person's model decodes 2 s of day-2 rest, then the trial's strongest 1 s",
        "repeated 12 times. Ratio = output 10-11 s into the hold / output 1-2 s into it (1 = holds), per finger pressing",
        "at least 10 %MVC.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
    ]


def behaviour_section() -> list[str]:
    """Timing and height of the decoded force (experiments.behaviour)."""
    lag_f, peak_f = RUNS / "behaviour" / "lag.csv", RUNS / "behaviour" / "peaks.csv"
    if not (lag_f.exists() and peak_f.exists()):
        return []
    lag = pd.read_csv(lag_f).groupby(["model", "subject"]).lag_ms.median()
    peaks = pd.read_csv(peak_f).assign(ratio=lambda d: d.peak_hat / d.peak).groupby(["model", "subject"]).ratio.mean()
    rows = []
    for m, label in BEHAVIOUR_MODELS.items():
        lo, hi = boot_ci(peaks[m].values)
        rows.append({
            "model": label,
            "shift (ms), median over people": f"{lag[m].median():.0f}",
            "decoded / measured peak": f"{peaks[m].mean():.2f} ({lo:.2f} to {hi:.2f})",
            "people with low peaks": f"{int((peaks[m] < 1).sum())}/{len(peaks[m])}",
        })  # fmt: skip
    return [
        "## Timing and strong presses",
        "",
        "Shift = the delay of the decoded force (±500 ms, 10 ms steps) that best correlates with the label, per day-2",
        "recording and moving finger. Negative = ahead. Peak = the highest decoded force in a press of at least 100 ms",
        "at 20 %MVC or more, over the measured one.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
    ]


def calibration_section() -> list[str]:
    """Labelled new-day data adapting the frozen TCN (experiments.calibration)."""
    short, longer = RUNS / CALIBRATION_RUN / "metrics.csv", RUNS / CALIBRATION_RUN / "whole_trials.csv"
    if not (short.exists() and longer.exists()):
        return []

    def table(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        df = df.assign(model=df.model.where(~df.model.str.startswith("tcn_ft"), "tcn_ft"))
        df = df[df.model.isin(list(CALIBRATION_LABELS))]
        per = hierarchical_mean(df, "mae", ["model", "budget_s"]).pivot_table(index="subject", columns=["model", "budget_s"], values="mae")
        budgets = sorted(df.budget_s.unique())
        rows = [{"method": label, **{f"{b:g} s": per[m, b].mean() if (m, b) in per else "–" for b in budgets}}
                for m, label in CALIBRATION_LABELS.items()]  # fmt: skip
        return markdown_table(pd.DataFrame(rows)), per

    short_table, _ = table(pd.read_csv(short))
    long_table, per = table(pd.read_csv(longer))
    best = max(b for _, b in per.columns)
    return [
        "## Labelled calibration on the new day",
        "",
        "The frozen new-day TCN fine-tuned (all layers, learning rate 1e-4, 100 steps), against ridges refitted on day 1",
        "plus the same labelled data. Seed 0. First seconds of day-2 trial 1, scored on trials 2-5:",
        "",
        short_table,
        "",
        "Whole day-2 trials 1 to 3, scored on trials 4-5:",
        "",
        long_table,
        "",
        f"Fine-tuned at {best:g} s − frozen: {paired(per['tcn_ft', best], per['tcn_frozen', 0.0])}.",
        "",
    ]


def _flag_delay(fd: pd.DataFrame, fault: str) -> str:
    col = f"{fault}_flag_frames"
    if col not in fd:
        return "as non-finite input"
    if fd[col].isna().any():
        return f"never in {int(fd[col].isna().sum())} of {len(fd)} recordings"
    return f"{fd[col].median() * 10:.0f} ms (median)"


def monitor_section() -> list[str]:
    """Simulated electrode faults, and false flags on clean recordings (experiments.monitor_check)."""
    faults, clean_file = RUNS / TCN_RUNS[0] / "monitor_check.csv", RUNS / TCN_RUNS[0] / "monitor_false_flags.csv"
    if not (faults.exists() and clean_file.exists()):
        return []
    fd, ff = pd.read_csv(faults), pd.read_csv(clean_file)
    per = fd.groupby("subject").mean(numeric_only=True)  # recordings, then people
    clean = ff.groupby("subject")[["mae_off", "mae_on"]].mean()
    rows = [
        {
            "fault from 5 s": FAULT_LABELS[f],
            "MAE, clean": per.clean.mean(),
            "MAE, monitor off": per[f"{f}_off"].mean(),
            "MAE, monitor on": per[f"{f}_on"].mean(),
            "flagged within": _flag_delay(fd, f),
        }
        for f in FAULTS
    ]
    return [
        "## Input monitor",
        "",
        f"A channel whose log-RMS falls more than {DROP_LOG:g} units below or rises more than {RISE_LOG:g} above its usual",
        "offset from its grid's median is replaced by median + offset. Non-finite samples become 0. A whole grid going",
        "silent is not detected. Faults start 5 s into each day-2 recording, MAE after the fault.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        f"Clean recordings: {ff.flagged_share.mean():.3%} of channel-frames replaced, in {int((ff.flagged_share > 0).sum())} of {len(ff)} recordings.",
        f"MAE {clean.mae_off.mean():.2f} with the monitor off (as scored everywhere else), {clean.mae_on.mean():.2f} on.",
        "",
    ]


def latency_section() -> list[str]:
    """Time per 10 ms update of the released decoder (emg2force.benchmark) over three runs."""
    files = [RES / "latency" / name for name in LATENCY_FILES]
    if not all(f.exists() for f in files):
        return []
    runs = [json.loads(f.read_text()) for f in files]
    rows = [{"timed": what, "p50 ms": r[mode]["p50_ms"], "p99 ms": r[mode]["p99_ms"], "max ms": r[mode]["max_ms"],
             "late": r[mode].get("late_updates", np.nan), "updates": r[mode]["n"]}
            for r in runs for mode, what in LATENCY_MODES.items()]  # fmt: skip
    t = pd.DataFrame(rows).groupby("timed", sort=False)
    table = t[["p50 ms", "p99 ms"]].median().join(t["max ms"].max()).reset_index()
    late, n = t["late"].sum(min_count=1), t["updates"].sum()
    table["late, of all updates"] = [("–" if pd.isna(late[w]) else f"{late[w]:.0f} of {n[w]}") for w in table.timed]
    setup = runs[0]["setup"]
    return [
        "## Latency",
        "",
        f"The TCN in `assets/` on {setup['cpu']}, one thread, batch 1, on recorded EMG. An update = filter, framing, input",
        "monitor and network. Paced runs sleep between updates like a timer-driven loop. p50 and p99: median of the three",
        "runs. Max: over all three. Late: done more than 10 ms after its deadline.",
        "",
        markdown_table(table, "{:.3f}"),
        "",
    ]


def simulation_section() -> list[str]:
    """Simulated gripper on replayed day-2 recordings, in newtons (emg2force.replay)."""
    per = {}
    for label, (folder, source) in SIM_RUNS.items():
        f = RES / folder / "mujoco_per_recording.csv"
        if f.exists():
            df = pd.read_csv(f)
            per[label] = df[df.source == source].groupby("subject").mean(numeric_only=True)
    if len(per) < len(SIM_RUNS):
        return []
    rows = [
        {
            "decoder": label,
            "decoder MAE (N)": d.decoder_mae_N.mean(),
            "controller tracking MAE (N)": d.tracking_mae_N.mean(),
            "whole chain MAE (N)": d.whole_chain_mae_N.mean(),
            "time under-grip": f"{d.time_under_grip.mean():.0%}",
            "time over-grip": f"{d.time_over_grip.mean():.0%}",
        }
        for label, d in per.items()
    ]
    d = (per["TCN, final recipe"].whole_chain_mae_N - per["TCN, person's day 1 only"].whole_chain_mae_N).dropna()
    lo, hi = boot_ci(d.values)
    return [
        "## Simulated gripper",
        "",
        "Decoded thumb and index flexion (day-2 trial 1) set a 1-15 N grip, tracked by a proportional-integral (PI) force",
        "controller on a fixed object in MuJoCo. Errors against the grip set from the force label. Under- and over-grip =",
        "more than 1 N below or above it.",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        f"Whole chain, final − day-1-only TCN: {d.mean():+.2f} N ({lo:+.2f} to {hi:+.2f}), {int((d < 0).sum())}/{len(d)}.",
        "",
    ]


def budget_rows() -> list[tuple[str, str]]:
    """The README's Real time table: delay from an EMG sample to the gripper's force, stage by stage."""
    files, steps = [RES / "latency" / name for name in LATENCY_FILES], RES / STEP_FILE
    if not (all(f.exists() for f in files) and steps.exists()):
        return []
    runs = [json.loads(f.read_text()) for f in files]
    filt = runs[0]["setup"]["filter_timing"]["group_delay_ms_median"]
    frame = float(str(runs[0]["setup"]["frame_lag_ms"]).split()[0])
    compute = float(np.median([r["pipeline_paced_sleep"]["p50_ms"] for r in runs]))
    st = pd.read_csv(steps)
    rows = [
        ("band-pass filter (median group delay)", f"{filt:.1f}"),
        ("framing (mean age of a 10 ms frame's samples)", f"{frame:.1f}"),
        (f"compute per update, paced every 10 ms (one CPU thread, {runs[0]['setup']['cpu']})", f"{compute:.1f}"),
        ("simulated gripper to half of a new force (assumed 30 ms motor lag)", f"{st.t50_ms.median():.0f}"),
        ("**estimated, sum of the stages above**", f"**{filt + frame + compute + st.t50_ms.median():.0f}**"),
    ]
    return rows


def selection_section() -> list[str]:
    return [
        "## How each setting was chosen",
        "",
        "On the 4 development people (1, 8, 10, 12), new day, development MAE. 0.30 rule = at least 0.30 %MVC lower on",
        "the mean and lower for at least 3 of the 4 people. Those runs are kept outside the repo.",
        "",
        markdown_table(pd.DataFrame(SELECTION, columns=["setting", "tried", "rule", "kept"])),
        "",
    ]


def dropped_section() -> list[str]:
    return [
        "## What didn't help",
        "",
        "MAE or change in MAE (%MVC), on the recipe of the time. Each was worse, or better by less than 0.30 %MVC. Code",
        "kept outside the repo.",
        "",
        markdown_table(pd.DataFrame(DROPPED, columns=["idea", "tested on", "result", "compared with"])),
        "",
    ]


SECTIONS: list[Callable[[], list[str]]] = [
    new_day_section,
    new_person_section,
    steps_section,
    same_day_section,
    hold_section,
    behaviour_section,
    calibration_section,
    monitor_section,
    latency_section,
    simulation_section,
    selection_section,
    dropped_section,
]
HEADER = [
    "# Experiments",
    "",
    "Mean absolute error (MAE) in %MVC (percent of maximal force) on the 16 evaluation people, mean over fingers,",
    "recordings and days, then people, unless stated. A − B: mean difference, 95 % bootstrap confidence interval over",
    "people, and how many people are lower with A. Written by `python -m experiments.report`.",
    "",
]


BEFORE, AFTER = "#9A9A9A", "#D55E00"  # day-1-only model (grey), final recipe (Okabe-Ito vermillion)


def headline_plot() -> None:
    """assets/new_day_per_person.png: each evaluation person's new-day error with the day-1-only TCN and the final TCN
    (5 seeds each), sorted by the first, the means with the paired change, and zero force and ridge as references."""
    before, after = per_person(frames(FIRST_TCN_RUNS, "tcn")), per_person(frames(TCN_RUNS, "tcn"))
    zero = per_person(frames(("final_cross_day_pop_256_ridge",), "zero")).mean()
    ridge = per_person(frames(("final_cross_day_pop_256_ridge_weighted",), "ridge")).mean()
    final = load(TCN_RUNS[0])
    demo_person, _ = median_person(final[(final.model == "tcn") & (final.seed == 0)])  # the demo's rule
    order = before.sort_values().index  # highest baseline at the top
    better = int((after < before).sum())

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=200)
    y = np.arange(len(order)) + 2.0  # rows 0-1: the mean and a gap

    def arrow(x0: float, x1: float, yi: float, dashed: bool = False) -> None:
        ax.annotate("", (x1, yi), (x0, yi), zorder=1, arrowprops={
            "arrowstyle": "-|>", "color": "#AAAAAA", "lw": 1.5, "ls": "--" if dashed else "-",
            "mutation_scale": 11, "shrinkA": 5, "shrinkB": 5})  # fmt: skip

    for yi, s in zip(y, order, strict=True):
        arrow(before[s], after[s], yi, dashed=after[s] > before[s])
    ax.scatter(before[order], y, s=46, facecolor="white", edgecolor=BEFORE, lw=1.6, zorder=2, label="trained on the person's day 1 only")
    ax.scatter(
        after[order],
        y,
        s=46,
        color=AFTER,
        zorder=3,
        label="final recipe: + 19 other people, electrode shifts,\nCAR, no running mean",
    )
    arrow(before.mean(), after.mean(), 0)
    for value, color in ((before, BEFORE), (after, AFTER)):
        m = value.mean()
        ax.scatter([m], [0], s=110, facecolor="white" if color == BEFORE else color, edgecolor=color, lw=1.8, zorder=3)
        ax.annotate(f"{m:.2f}", (m, 0), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=10.5,
                    color=AFTER if color == AFTER else "#555555", fontweight="bold")  # fmt: skip
    label_y = len(order) + 2.4
    for x, label, ha, nudge in (
        (ridge, f"ridge, same data\n{ridge:.2f}", "right", -0.08),
        (zero, f"predict zero force\n{zero:.2f}", "left", 0.08),
    ):
        ax.axvline(x, color="#777777", ls=":", lw=1.1, zorder=0)
        ax.text(x + nudge, label_y, label, ha=ha, va="bottom", fontsize=9.5, color="#555555", linespacing=1.1)
    names = [f"person {s}" + (" (demo)" if s == demo_person else "") for s in order]
    ax.set_yticks([0, *y], ["mean (16)", *names], fontsize=10)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(-0.9, len(order) + 4.2)
    ax.set_xlabel("new-day error, MAE (%MVC)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.grid(axis="x", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), frameon=False, fontsize=10, handletextpad=0.3)
    fig.text(0.015, 0.965, f"New-day error falls for {better} of {len(order)} evaluation people", fontsize=15, fontweight="bold", ha="left")
    change = after - before
    lo, hi = boot_ci(change.values)
    fig.text(0.015, 0.928, f"Mean {before.mean():.2f} → {after.mean():.2f} %MVC, change {change.mean():.2f} (95 % CI {lo:.2f} to "
             f"{hi:.2f}). No labels from the new day. 5 training seeds each.".replace("-", "−"),
             fontsize=10.5, color="#555555", ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(ASSETS / "new_day_per_person.png", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true", help="write the page even if some sections have no inputs yet")
    args = ap.parse_args(argv)
    if missing := [r for r in (*TCN_RUNS, *FIRST_TCN_RUNS) if not has_run(r)]:
        raise SystemExit(f"missing runs {missing} in {RUNS}: run ./reproduce.sh first (the committed page is unchanged)")
    built = [(s.__name__, s()) for s in SECTIONS]
    if empty := [name for name, lines in built if not lines]:
        if not args.allow_partial:
            raise SystemExit(f"no inputs for {empty}. Run ./reproduce.sh or pass --allow-partial (the committed page is unchanged)")
        print(f"warning: no inputs for {empty}, written without them", file=sys.stderr)
    headline_plot()
    if rows := budget_rows():
        print("README Real time table:\n" + markdown_table(pd.DataFrame(rows, columns=["stage", "ms"])))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "experiments.md").write_text("\n".join([*HEADER, *(line for _, lines in built for line in lines)]) + "\n")


if __name__ == "__main__":
    main()
