"""Calibration protocol: leakage rules, the cut-off, fine-tuning, and a synthetic run over every fold."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import experiments.calibration as calib
from emg2force.causal import FilterConfig, design_sos, offline_frames
from emg2force.dataset import prefix_sequence
from emg2force.hyser import EMG_CHANNELS, EMG_CLOCK_SCALE, FINGERS, FS_FORCE, Hyser, emg_count_at, recording_key
from emg2force.models import Sequence
from emg2force.splits import (
    CAL_TRIAL,
    DAY2,
    EVAL_SUBJECTS,
    SESSIONS,
    SUBJECTS,
    TRIALS,
    calibration_key,
    check_split,
    make_splits,
    session_of,
    trial_of,
)
from emg2force.tcn import TCNConfig, TCNModel
from tests.helpers import make_fake_hyser

HISTORY = 32
TINY_TCN = {"hidden": 4, "dilations": [1, 2, 4, 8, 16], "steps": 2, "batch": 2, "crop": 64, "device": "cpu", "personal_share": 0.5}
CUT_S = 1.5


# Leakage rules
def test_calibration_trial_in_neither_train_nor_test():
    for sp in make_splits("cross_day_pop_cal", EVAL_SUBJECTS):
        cal = calibration_key(sp.held_out_subject)
        assert cal not in sp.train
        assert cal not in sp.test
        assert {session_of(k) for k in sp.test} == {DAY2}
        assert sorted(trial_of(k) for k in sp.test) == [t for t in TRIALS if t != CAL_TRIAL]
        assert not any(k.startswith(f"s{sp.held_out_subject:02d}_d{DAY2}") for k in sp.train)


def test_split_containing_calibration_trial_is_rejected():
    sp = make_splits("cross_day_pop_cal", [2])[0]
    with pytest.raises(ValueError, match="session 1|calibration data"):  # two rules guard training
        check_split(replace(sp, train=(*sp.train, calibration_key(2))))
    with pytest.raises(ValueError, match="calibration data"):
        check_split(replace(sp, test=(*sp.test, calibration_key(2))))


# The cutoff: nothing after it reaches calibration data or fitted models
def _perturb_after_cutoff(root, n_force: int) -> None:
    """Overwrite EMG and force after the cutoff of subject 1, session 2, trial CAL_TRIAL, in place."""
    folder = root / "random_dataset" / "subject01_session2"
    for name, width, start in (
        (f"random_raw_sample{CAL_TRIAL}", len(EMG_CHANNELS), int(emg_count_at(n_force - 1))),
        # the last label before the cut reads raw force at (n - 1) * EMG_CLOCK_SCALE on the force clock
        (f"random_force_sample{CAL_TRIAL}", len(FINGERS), int(np.ceil((n_force - 1) * EMG_CLOCK_SCALE)) + 1),
    ):
        path = folder / f"{name}.dat"
        d = np.fromfile(path, dtype="<i2").reshape(-1, width)
        d[start:] = np.random.default_rng(1).integers(-3000, 3000, size=d[start:].shape)
        d.tofile(path)


def _prefix(root) -> Sequence:
    return prefix_sequence(Hyser(root), calibration_key(1), CUT_S, "session", "zero_phase_lowpass_10hz", input_norm="running_mean")


def test_prefix_frames_equal_the_full_recordings_first_frames(tmp_path):
    make_fake_hyser(tmp_path, subjects=(1,), sessions=SESSIONS, trials=(CAL_TRIAL,), seconds=3.0)
    rec = Hyser(tmp_path).load(1, DAY2, CAL_TRIAL)
    full = offline_frames(rec.emg, len(rec.force_v), design_sos(FilterConfig()))
    raw = prefix_sequence(Hyser(tmp_path), calibration_key(1), CUT_S, "session", "none")
    assert raw.x.shape[0] == int(CUT_S * FS_FORCE)
    assert np.array_equal(raw.x, full[: raw.x.shape[0]])


def test_changing_data_after_the_cutoff_changes_nothing(tmp_path):
    make_fake_hyser(tmp_path, subjects=(1,), sessions=SESSIONS, trials=(CAL_TRIAL,), seconds=3.0)
    before, full_before = _prefix(tmp_path), Hyser(tmp_path).load(1, DAY2, CAL_TRIAL)
    _perturb_after_cutoff(tmp_path, int(CUT_S * FS_FORCE))
    after, full_after = _prefix(tmp_path), Hyser(tmp_path).load(1, DAY2, CAL_TRIAL)
    n = int(CUT_S * FS_FORCE)
    assert not np.array_equal(full_before.force_v[n:], full_after.force_v[n:])  # the perturbation is real
    assert not np.array_equal(full_before.emg[int(emg_count_at(n - 1)) :], full_after.emg[int(emg_count_at(n - 1)) :])
    assert np.array_equal(before.x, after.x)
    assert np.array_equal(before.y, after.y)
    assert np.array_equal(before.scored, after.scored)
    base = TCNModel(HISTORY, TCNConfig(hidden=4, steps=3, batch=2, crop=64, device="cpu")).fit([before])
    a, b = base.fine_tuned([before], 3, 1e-3), base.fine_tuned([after], 3, 1e-3)
    assert np.array_equal(a.predict(before.x), b.predict(before.x), equal_nan=True)


# Fine-tuning
def _synthetic(n=300, n_ch=6, seed=0, subject=1, key="r") -> Sequence:
    rng = np.random.default_rng(seed)
    x = (np.cumsum(rng.standard_normal((n, n_ch)), 0) * 0.1).astype(np.float32)
    y = (10 * x @ rng.standard_normal((n_ch, 5))).astype(np.float32)
    return Sequence(key, subject, 1, x, y, np.arange(n) >= HISTORY)


def test_fine_tune_copies_and_leaves_the_frozen_model_unchanged():
    train, cal = [_synthetic(seed=0)], _synthetic(seed=1)
    base = TCNModel(HISTORY, TCNConfig(hidden=8, steps=20, batch=4, crop=64, device="cpu")).fit(train)
    before = base.predict(cal.x)
    assert np.array_equal(base.fine_tuned([cal], 0, 1e-3).predict(cal.x), before, equal_nan=True)
    tuned = base.fine_tuned([cal], 30, 1e-2)
    assert np.array_equal(base.predict(cal.x), before, equal_nan=True)
    m = cal.scored & np.isfinite(before).all(1)
    assert np.abs(tuned.predict(cal.x)[m] - cal.y[m]).mean() < np.abs(before[m] - cal.y[m]).mean()


# Synthetic runs through every fold
CFG = {
    "history": HISTORY,
    "input_norm": "running_mean",
    "budgets_s": [0, 2],
    "tcn": TINY_TCN,
    "fine_tune": {"lr": 1e-3, "steps": 2},
    "ridge": [{"name": "ridge", "alpha": 0.1, "input_norm": "none"}],
}


def _all_sequences() -> dict[str, Sequence]:
    keys = [recording_key(s, d, t) for s in SUBJECTS for d in SESSIONS for t in TRIALS]
    return {k: replace(_synthetic(n=260, seed=i, subject=int(k[1:3]), key=k), session=session_of(k)) for i, k in enumerate(keys)}


def _fake_prefix(seqs_by_norm):
    def prefix(cfg, ds, s, budget, input_norm):
        full = seqs_by_norm[input_norm][calibration_key(s)]
        n = int(budget * FS_FORCE)
        return replace(full, x=full.x[:n], y=full.y[:n], scored=full.scored[:n])

    return prefix


def test_every_fold_runs_and_scores_only_its_test_trials(monkeypatch, tmp_path):
    seqs = _all_sequences()
    by_norm = {"running_mean": seqs, "none": seqs}
    monkeypatch.setattr(calib, "_prefix", _fake_prefix(by_norm))
    rows = []
    for sp in make_splits("cross_day_pop_cal", EVAL_SUBJECTS):
        r = calib.short_fold(sp, by_norm, None, CFG, tmp_path / "models", lambda *a: None)
        assert {x["recording"] for x in r} == set(sp.test)
        assert (tmp_path / "models" / f"base_{sp.name}.state.pt").is_file()
        rows += r
    df = pd.DataFrame(rows)
    assert set(df[df.budget_s == 0].model) == {calib.FROZEN, "ridge"}
    assert set(df[df.budget_s == 2].model) == {calib.FINE_TUNED, "ridge"}
    assert df.subject.nunique() == len(EVAL_SUBJECTS)


def test_whole_trial_calibration_scores_trials_4_and_5_at_every_budget():
    seqs = _all_sequences()
    sp0 = make_splits("cross_day_pop_cal", EVAL_SUBJECTS[:1])[0]
    s = sp0.held_out_subject
    sp = replace(sp0, test=tuple(recording_key(s, DAY2, t) for t in calib.TEST_TRIALS))
    base = TCNModel(HISTORY, TCNConfig(hidden=4, steps=2, batch=2, crop=64, device="cpu")).fit([seqs[k] for k in sp.train[:3]])
    df = pd.DataFrame(calib.whole_trial_fold(sp, base, {"running_mean": seqs, "none": seqs}, CFG))
    assert set(df.recording) == set(sp.test)
    assert set(df.budget_s) == {0.0, *calib.CAL_TRIALS}
    assert set(df[df.budget_s > 0].model) == {calib.FINE_TUNED, "ridge"}


def test_each_ridge_baseline_uses_its_own_normalisation(monkeypatch, tmp_path):
    """The ridge given uninformative inputs scores worse, so each uses its own sequences. Both run at every budget."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 5)).astype(np.float32)  # one map for all recordings, so ridge can transfer
    seqs = {k: replace(v, y=10 * v.x @ w) for k, v in _all_sequences().items()}
    noise = {k: replace(v, x=rng.standard_normal(v.x.shape).astype(np.float32)) for k, v in seqs.items()}
    by_norm = {"running_mean": seqs, "none": noise}
    monkeypatch.setattr(calib, "_prefix", _fake_prefix(by_norm))
    cfg = {
        **CFG,
        "ridge": [{"name": "ridge", "alpha": 0.1, "input_norm": "none"}, {"name": "ridge_rm", "alpha": 0.1, "input_norm": "running_mean"}],
    }
    sp = make_splits("cross_day_pop_cal", EVAL_SUBJECTS)[0]
    mae = (
        pd.DataFrame(calib.short_fold(sp, by_norm, None, cfg, tmp_path / "models", lambda *a: None))
        .groupby(["model", "budget_s"])
        .mae.mean()
    )
    for budget in cfg["budgets_s"]:
        assert mae["ridge_rm", budget] < mae["ridge", budget]
