"""End to end on synthetic Hyser folders: a run, its record and report, the analysis scripts and the demo."""

import json
import shutil
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
import pytest
import yaml

from emg2force import replay
from emg2force.hyser import Hyser
from emg2force.pipeline import decoder_for, load_decoder
from emg2force.splits import EVAL_SUBJECTS, Split, make_splits, session_of, subject_of
from emg2force.tcn import TCNModel
from experiments import render, report
from experiments.hold_test import strongest_window
from experiments.metrics import boot_ci, hierarchical_mean
from experiments.run import _save_outputs, check_config, config_sequences, front_end_of, make_model, run_summary
from experiments.same_day import same_day_split
from tests.helpers import SMOKE_CFG, make_fake_hyser

TOL = 1e-4  # %MVC, tolerance between implementations (as in tests/test_released_decoder.py)


def test_run_trains_scores_and_records_every_model(smoke_run):
    root, out, df = smoke_run
    test_keys = {f"s02_d2_t{t}" for t in range(1, 6)}
    assert set(df.model) == {"constant", "ridge", "tcn"}
    assert set(df.recording) == test_keys
    assert np.isfinite(df.mae).all()
    for name in ("metrics.csv", "config.yaml", "env.json", "splits.json", "summary.json", "predictions.npz", "train_loss.csv"):
        assert (out / name).is_file()
    loss = pd.read_csv(out / "train_loss.csv")  # one row per TCN training step
    assert set(loss.model) == {"tcn"}
    assert loss.step.tolist() == [1, 2, 3]
    assert np.isfinite(loss.loss).all()
    assert set(json.loads((out / "splits.json").read_text())["folds"]["s02"]["test"]) == test_keys

    # the saved checkpoint reproduces the run's own predictions
    preds = np.load(out / "predictions.npz")
    model = TCNModel.load_state(out / "models" / "tcn_seed0_s02.state.pt")
    x = config_sequences(SMOKE_CFG, Hyser(root), ["s02_d2_t3"])["s02_d2_t3"].x
    np.testing.assert_allclose(model.predict(x), preds["tcn|0|s02_d2_t3"], rtol=0, atol=TOL, equal_nan=True)

    # the run record says what trained each model, and the checkpoint says what features it expects
    summary = json.loads((out / "summary.json").read_text())
    assert {m["model"]: m["train_device"] for m in summary["models"]} == {"constant": "cpu", "ridge": "cpu", "tcn": "cpu"}
    assert model.provenance["train_device"] == "cpu"
    assert model.front_end == front_end_of(SMOKE_CFG)


def test_decoder_built_from_the_checkpoint_alone_reproduces_the_run(smoke_run):
    """load_decoder reads filter, normalisation and warm-up from the file. Streamed raw EMG gives the run's predictions."""
    root, out, _ = smoke_run
    decoder = load_decoder(out / "models" / "tcn_seed0_s02.state.pt")
    rec = Hyser(root).load(2, 2, 3)
    want = np.load(out / "predictions.npz")["tcn|0|s02_d2_t3"]
    got = np.full_like(want, np.nan)
    for j, p in decoder.push(rec.emg):
        if j < len(got):
            got[j] = p
    valid = ~np.isnan(want).any(1) & (np.arange(len(want)) >= decoder.n_base)  # the decoder is silent in the warm-up
    assert valid.sum() > 0
    np.testing.assert_allclose(got[valid], want[valid], rtol=0, atol=TOL)
    assert np.isnan(got[~valid]).all()


def test_decoder_for_refuses_checkpoints_it_cannot_rebuild(smoke_run):
    _, out, _ = smoke_run
    model = TCNModel.load_state(out / "models" / "tcn_seed0_s02.state.pt")
    model.front_end = None  # a file written before the front end was recorded
    with pytest.raises(ValueError, match="does not record its front end"):
        decoder_for(model)


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({**SMOKE_CFG, "input_nrom": "running_mean"}, "unknown config keys"),
        ({**SMOKE_CFG, "warmup": 0.7}, "unknown config keys"),
        ({k: v for k, v in SMOKE_CFG.items() if k != "models"}, "missing keys"),
    ],
)
def test_top_level_config_typos_and_omissions_are_errors(cfg, message):
    with pytest.raises(ValueError, match=message):
        check_config(cfg)


@pytest.mark.parametrize(
    ("kind", "params", "message"),
    [
        ("tcn", {"aug_shfit": 1.0}, "unknown config keys"),
        ("ridge", {"alpah": 1.0}, "unknown config keys"),
        ("constant", {"alpha": 1.0}, "unknown config keys"),
        ("lstm", {"kernel_size": 2}, "unknown config keys"),
        ("lsmt", {}, "unknown model"),
        ("tcn", {"lr": "1e-3"}, "must be a number"),
        ("tcn", {"dropout": 1.5}, "dropout must be in"),
    ],
)
def test_config_typos_are_errors(kind, params, message):
    with pytest.raises(ValueError, match=message):
        make_model(kind, params, 32, 0, {})


def test_replay_drives_the_simulated_gripper_from_saved_predictions(smoke_run, tmp_path, monkeypatch):
    root, out, _ = smoke_run
    monkeypatch.setattr(replay, "RECORDINGS", ["s02_d2_t1"])
    summary = replay.run(out, root, tmp_path)
    assert set(summary.source) == {"reference", "ridge", "tcn"}
    assert summary[replay.SUMMARY_COLUMNS].notna().all().all()
    assert (tmp_path / "mujoco_summary.csv").is_file()


def test_a_rerun_replaces_the_run_folder_and_an_interrupted_one_keeps_it(tmp_path):
    from experiments.run import fresh_run_dir

    out = tmp_path / "runs" / "x"
    with fresh_run_dir(out) as work:
        (work / "models").mkdir()
        (work / "models" / "old_fold.pt").write_text("old")
        (work / "metrics.csv").write_text("first")
    with fresh_run_dir(out) as work:
        (work / "metrics.csv").write_text("second")
    assert sorted(p.name for p in out.iterdir()) == ["metrics.csv"]  # nothing left from the first run
    assert (out / "metrics.csv").read_text() == "second"

    def interrupted_run() -> None:
        with fresh_run_dir(out) as work:
            (work / "metrics.csv").write_text("third")
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        interrupted_run()
    assert (out / "metrics.csv").read_text() == "second"


def _metrics(splits):
    rows = []
    for sp in splits:
        for rec in sp.test:
            for f, err in (("thumb", 2.0), ("index", 4.0)):
                rows.append({"subject": sp.held_out_subject, "session": 2, "recording": rec, "finger": f, "model": "ridge", "seed": 0,
                             "mae": err, "rmse": err, "r2": 0.1, "bias": -err})  # fmt: skip
    return pd.DataFrame(rows)


def test_run_writes_config_env_splits_and_summary(tmp_path):
    splits = make_splits("cross_day", [1, 8])
    df = _metrics(splits)
    cfg = {"name": "t", "protocol": "cross_day"}
    _save_outputs(tmp_path, cfg, df, {}, splits, run_summary(df, splits, 10, time.time() - 5, {("ridge", 0): "cpu"}))

    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == cfg
    assert {"git_commit", "git_dirty", "torch"} <= set(json.loads((tmp_path / "env.json").read_text()))
    folds = json.loads((tmp_path / "splits.json").read_text())["folds"]
    assert {f: (v["train"], v["test"]) for f, v in folds.items()} == {sp.name: (list(sp.train), list(sp.test)) for sp in splits}
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["splits"] == 2
    assert summ["subjects_scored"] == 2
    assert summ["seconds"] >= 5
    assert summ["models"] == [{"model": "ridge", "seed": 0, "train_device": "cpu", "mae": 3.0, "rmse": 3.0, "r2": 0.1, "bias": -3.0}]
    assert pd.read_csv(tmp_path / "metrics.csv").shape == df.shape


def _point_report_at(tmp_path, monkeypatch):
    runs, res, docs, assets = (tmp_path / d for d in ("runs", "results", "docs", "assets"))
    for d in (runs, res, assets):
        d.mkdir(exist_ok=True)
    for name, value in (("RUNS", runs), ("RES", res), ("DOCS", docs), ("ASSETS", assets)):
        monkeypatch.setattr(report, name, value)
    return runs, docs


RUN_NAMES = (*report.TCN_RUNS, *report.FIRST_TCN_RUNS, "final_cross_day_pop_256_ridge", "final_cross_day_pop_256_ridge_weighted",
             "final_loso_256", "final_loso_shift_256")  # fmt: skip


def test_report_tables_match_the_run_metrics(smoke_run, tmp_path, monkeypatch):
    runs, _ = _point_report_at(tmp_path, monkeypatch)
    _, out, _ = smoke_run
    shutil.copytree(out, runs / "r")
    text = "\n".join(report.results_table({"ridge": (("r",), "ridge"), "TCN": (("r",), "tcn")}, "ridge"))
    df = pd.read_csv(runs / "r" / "metrics.csv")
    per = hierarchical_mean(df, "mae", ["model"]).pivot(index="subject", columns="model", values="mae")
    for label, model in (("ridge", "ridge"), ("TCN", "tcn")):
        assert f"| {label} | {per[model].mean():.2f} ± " in text
    d = per["tcn"] - per["ridge"]
    lo, hi = boot_ci(d.values)
    assert f"{d.mean():+.2f} ({lo:+.2f} to {hi:+.2f}), {int((d < 0).sum())}/{len(d)}" in text
    assert report.results_table({"ridge": (("r",), "ridge"), "LSTM": (("r",), "lstm")}, "ridge") == []  # a missing model


def test_report_names_missing_inputs_and_leaves_the_page_alone(smoke_run, tmp_path, monkeypatch):
    runs, docs = _point_report_at(tmp_path, monkeypatch)
    docs.mkdir()
    (docs / "experiments.md").write_text("committed\n")
    with pytest.raises(SystemExit, match="missing runs"):
        report.main([])
    _, out, _ = smoke_run
    for name in RUN_NAMES:
        shutil.copytree(out, runs / name)
    with pytest.raises(SystemExit, match="no inputs for .*same_day_section.*hold_section.*monitor_section"):
        report.main([])
    assert (docs / "experiments.md").read_text() == "committed\n"


@pytest.fixture(scope="module")
def six_second_hyser(tmp_path_factory):
    """Subject 2, both days, trial 1, 6 s long: long enough for a fault injected at 5 s."""
    root = tmp_path_factory.mktemp("hyser6")
    make_fake_hyser(root, subjects=(2,), trials=(1,), seconds=6.0)
    return Hyser(root)


def test_monitor_check_catches_a_dead_row_with_the_saved_fold_model(smoke_run, six_second_hyser, monkeypatch):
    import experiments.monitor_check as mc

    _, out, _ = smoke_run
    monkeypatch.setattr(mc, "TRIALS", (1,))
    faults = mc.faults(out, six_second_hyser, SMOKE_CFG)
    assert len(faults) == 1
    delay = faults.dead_row_flag_frames.iloc[0]
    assert delay is not None
    assert 0 <= delay < 50
    assert np.isfinite(faults[[f"{f}_{s}" for f in mc.FAULTS for s in ("on", "off")]].to_numpy(float)).all()
    clean = mc.clean_rows(out, six_second_hyser, SMOKE_CFG)
    assert clean.flagged_share.iloc[0] < 0.01  # steady synthetic EMG: almost nothing replaced


def test_held_press_runner_refuses_a_run_without_fold_models(smoke_run, tmp_path):
    import experiments.hold_test as ht

    _, out, _ = smoke_run
    (tmp_path / "models").mkdir()
    (tmp_path / "config.yaml").write_text((out / "config.yaml").read_text())
    with pytest.raises(FileNotFoundError, match="save_models: all"):
        ht.run(tmp_path, out)


def test_same_day_fold_scores_both_days_with_every_model(smoke_run):
    from emg2force.splits import Split
    from experiments.same_day import fold_rows

    root, _, _ = smoke_run
    train = tuple(f"s02_d1_t{t}" for t in (1, 2, 3))
    test = tuple(f"s02_d{d}_t{t}" for d in (1, 2) for t in (4, 5))
    cfg = {
        **{k: SMOKE_CFG[k] for k in ("mvc_convention", "label_filter", "history", "warmup_s")},
        "input_norm": "none",
        "tcn": {"hidden": 8, "steps": 2, "batch": 2, "crop": 64, "device": "cpu"},
    }
    rows = pd.DataFrame(fold_rows(Split("same_day", "s02", train, test, 2), cfg, Hyser(root)))
    assert set(rows.model) == {"tcn", "constant", "zero"}
    assert set(rows.recording) == set(test)
    assert set(rows.session) == {1, 2}
    assert np.isfinite(rows.mae).all()


def test_strongest_window_finds_the_press_after_the_rest():
    y = np.zeros((1000, 5))
    y[600:700, 2] = -30.0  # flexion press
    y[50:150, 0] = 50.0  # stronger, but inside the opening rest that is skipped
    assert strongest_window(y, start=200) == 600


def test_same_day_split_holds_out_both_test_days():
    for sp in make_splits("cross_day_pop_cal", EVAL_SUBJECTS):
        out = same_day_split(sp)
        s = sp.held_out_subject
        assert sorted(out.test) == sorted(f"s{s:02d}_d{d}_t{t}" for d in (1, 2) for t in (4, 5))
        assert not set(out.train) & set(out.test)
        assert not any(subject_of(k) == s and session_of(k) == 2 for k in out.train)
        assert set(sp.train) - set(out.train) <= set(out.test)


def test_same_day_split_rejects_new_day_training():
    sp = make_splits("cross_day_pop_cal", EVAL_SUBJECTS)[0]
    leaky = Split(sp.protocol, sp.name, (*sp.train, f"s{sp.held_out_subject:02d}_d2_t3"), sp.test, sp.held_out_subject)
    with pytest.raises(ValueError, match="leaks"):
        same_day_split(leaky)


def _can_render() -> bool:
    """Offscreen rendering needs an OpenGL context, which headless CI runners do not have."""
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"), 16, 16).close()
    except Exception:  # mujoco raises FatalError, CGLError or GLFWError depending on the platform
        return False
    return True


@pytest.mark.skipif(not _can_render(), reason="no OpenGL context for offscreen MuJoCo rendering")
def test_demo_renders_cards_and_a_clip_and_the_gif(smoke_run, tmp_path):
    from PIL import Image

    root, out, _ = smoke_run
    video, gif = tmp_path / "demo.mp4", tmp_path / "demo.gif"
    starts = render.write_video(
        video,
        root,
        [
            ("card", ["line one", "line two"], 4.0),
            ("clip", out, "tcn", "s02_d2_t1", "Smoke model"),
            ("card", ["a caption", "another line"], render.CARD_SECONDS),
        ],
    )
    frames = imageio.get_reader(video).count_frames()
    assert starts == [4.0 + 2.0]  # seconds: the clip starts after the intro card and its own title card
    assert frames > (4.0 + 2.0 + render.CARD_SECONDS) * render.FPS  # cards plus the clip itself
    render.render_gif(out, "s02_d2_t1", root, gif, start_s=0.7)
    with Image.open(gif) as im:
        assert im.size == (960, 480)
        assert im.n_frames > 1
