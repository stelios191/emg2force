"""Models and metrics: filter and ridge numerics, TCN training and shift augmentation, ONNX export, metrics, timing and peaks."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import signal

from emg2force.causal import design_sos, offline_frames
from emg2force.hyser import FINGERS, label_lowpass
from emg2force.models import RidgeModel, Sequence, StatsCache, lag_blocks, lag_features
from emg2force.pipeline import OnnxModel, export_onnx
from emg2force.splits import make_splits
from emg2force.tcn import TCNConfig, TCNModel, per_frame_l1
from experiments.behaviour import MIN_FRAMES, STRONG, best_lag, press_peaks
from experiments.metrics import boot_ci, check_complete, hierarchical_mean, recording_metrics, summary
from experiments.run import run_summary


def _gain_db(sos, f):
    _, h = signal.sosfreqz(sos, worN=np.atleast_1d(f), fs=2048)
    return 20 * np.log10(np.abs(h))


def test_emg_filter_passband_stopband_and_notches():
    sos = design_sos()
    for sec in sos:  # every section stable: poles strictly inside the unit circle
        assert np.all(np.abs(np.roots(sec[3:])) < 1)
    assert np.all(np.abs(_gain_db(sos, [30, 75, 125, 175, 225, 275, 325])) < 1.0)  # between notches
    assert np.all(np.abs(_gain_db(sos, [375, 425])) < 2.0)  # notch skirts near the upper edge
    assert np.all(np.abs(_gain_db(sos, [10, 480]) + 3.0) < 0.5)  # -3 dB band edges
    assert np.all(_gain_db(sos, [1, 2]) < -40)  # DC drift / motion artefact removed
    assert np.all(_gain_db(sos, [900, 1000]) < -20)  # above band
    assert np.all(_gain_db(sos, np.arange(50, 501, 50)) < -30)  # mains and harmonics


def test_frames_are_log_rms_of_filtered_samples():
    rng = np.random.default_rng(0)
    emg = rng.standard_normal((4096, 3))
    sos = design_sos()
    y = signal.sosfilt(sos, emg, axis=0)
    f = offline_frames(emg, 150, sos)
    # frame 100 covers EMG samples [floor(99*20.48)+1, floor(100*20.48)+1) = [2028, 2049)
    seg = y[2028:2049]
    np.testing.assert_allclose(f[100], 0.5 * np.log((seg**2).mean(0) + 1e-12), rtol=1e-5)


# Label filter vs MATLAB reference
def matlab_filtfilt(x):
    """MATLAB `filtfilt(b, a, x)` with `[b, a] = butter(8, 10 / 50)`: odd reflection, 3*(nfilt-1) pad."""
    b, a = signal.butter(8, 10, fs=100)
    return signal.filtfilt(b, a, x, axis=0, padtype="odd", padlen=3 * (len(a) - 1))


def test_label_filter_matches_matlab_filtfilt():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal((2500, 5)), axis=0)
    ours, ref = label_lowpass(x), matlab_filtfilt(x)
    scale = np.abs(ref).max()
    assert np.abs(ours - ref).max() < 1e-9 * scale  # identical, including trial edges


# Lag features and ridge
def test_lag_features_match_brute_force():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((80, 3))
    H = 12
    f = lag_features(x, H)
    blocks = lag_blocks(H)
    assert sum(b - a for a, b in blocks) == H
    for j in range(H - 1, 80):
        expect = np.concatenate([x[j - b + 1 : j - a + 1].mean(0) for a, b in blocks])
        np.testing.assert_allclose(f[j], expect, rtol=1e-10)
    assert np.isnan(f[: H - 1]).any(1).all()  # a lag block lacks history -> NaN
    assert not np.isnan(f[H - 1 :]).any()


def test_ridge_matches_direct_solution():
    rng = np.random.default_rng(3)
    H, T, C = 4, 400, 3
    seqs = []
    for k in range(3):
        x = rng.standard_normal((T, C)).astype(np.float32)
        y = rng.standard_normal((T, 5)).astype(np.float32)
        seqs.append(Sequence(f"r{k}", 1, 1, x, y, np.arange(T) >= 10))
    alpha = 0.7
    m = RidgeModel(H, alpha=alpha).fit(seqs)
    # direct: standardise training rows, solve (Z'Z/n + alpha I) w = Z'y/n
    X = np.concatenate([lag_features(s.x, H)[s.scored] for s in seqs])
    Y = np.concatenate([s.y[s.scored] for s in seqs]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z / len(Z) + alpha * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)) / len(Z))
    xt = rng.standard_normal((50, C)).astype(np.float32)
    ref = (lag_features(xt, H) - mu) / sd @ w + Y.mean(0)
    np.testing.assert_allclose(m.predict(xt)[H - 1 :], ref[H - 1 :], rtol=1e-8, atol=1e-8)


def test_ridge_without_scored_frames_is_an_error():
    s = Sequence("r", 1, 1, np.zeros((50, 4), np.float32), np.zeros((50, 5), np.float32), np.zeros(50, bool))
    for train in ([], [s]):
        with pytest.raises(ValueError, match="at least one scored frame"):
            RidgeModel(1).fit(train)


# Metrics
def test_recording_metrics_by_hand():
    y = np.zeros((6, 5))
    y[:, 0] = [0, 10, 20, 30, 40, 50]
    yhat = y.copy()
    yhat[:, 0] += [1, -1, 2, -2, 3, -3]
    scored = np.array([False, True, True, True, True, True])  # first frame not scored
    r = recording_metrics(y, yhat, scored)[0]
    e = np.array([-1, 2, -2, 3, -3.0])
    assert r["n"] == 5
    assert np.isclose(r["mae"], np.abs(e).mean())
    assert np.isclose(r["rmse"], np.sqrt((e**2).mean()))
    assert np.isclose(r["bias"], e.mean())
    yt = y[1:, 0]
    assert np.isclose(r["r2"], 1 - (e**2).sum() / ((yt - yt.mean()) ** 2).sum())
    assert np.isclose(r["r"], np.corrcoef(yhat[1:, 0], yt)[0, 1])
    constant = recording_metrics(y, np.zeros_like(y), scored)[0]
    assert np.isnan(constant["r"])  # correlation undefined for a constant prediction


def test_r2_can_be_negative_and_unscored_nans_are_allowed():
    y = np.tile(np.linspace(-10, 10, 50)[:, None], (1, 5))
    yhat = -y
    yhat[:3] = np.nan  # undefined inside the unscored warm-up: fine
    r = recording_metrics(y, yhat, np.arange(50) >= 3)
    assert all(row["r2"] < -1 for row in r)
    assert r[0]["n"] == 47


def test_a_non_finite_prediction_on_a_scored_frame_is_an_error():
    y = np.zeros((50, 5))
    yhat = np.zeros((50, 5))
    yhat[20, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        recording_metrics(y, yhat, np.ones(50, bool))


def test_check_complete_catches_missing_people_fingers_and_frame_counts():
    rows = [{"model": m, "recording": r, "finger": f, "n": 100} for m in ("a", "b") for r in ("r1", "r2") for f in FINGERS]
    df = pd.DataFrame(rows)
    check_complete(df, ["r1", "r2"], ["model"])
    with pytest.raises(ValueError, match="missing"):
        check_complete(df[~((df.model == "b") & (df.recording == "r2"))], ["r1", "r2"], ["model"])
    with pytest.raises(ValueError, match="fingers"):
        check_complete(df[~((df.model == "b") & (df.finger == "little"))], ["r1", "r2"], ["model"])
    with pytest.raises(ValueError, match="frame counts"):
        check_complete(df.assign(n=np.where(df.model == "b", 99, 100)), ["r1", "r2"], ["model"])


HISTORY = 32
SMALL = {"hidden": 8, "steps": 150, "batch": 8, "crop": 64, "device": "cpu"}


def _synthetic(n_rec=4, n_frames=400, n_ch=6, seed=0) -> list[Sequence]:
    """Force is a fixed linear map of the (smoothed) channels, so a working trainer beats a constant."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_ch, 5))
    out = []
    for i in range(n_rec):
        x = np.cumsum(rng.standard_normal((n_frames, n_ch)), 0) * 0.1
        x = (x - x.mean(0)).astype(np.float32)
        y = (10 * x @ w).astype(np.float32)
        scored = np.arange(n_frames) >= HISTORY
        out.append(Sequence(f"r{i}", 1, 1, x, y, scored))
    return out


def test_loss_is_l1_over_fingers():
    err = torch.tensor([[[1.0, -2.0, 10.0]]]).expand(1, 5, 3)
    assert torch.allclose(per_frame_l1(err), torch.tensor([[1.0, 2.0, 10.0]]))


@pytest.mark.parametrize(("arch", "steps"), [("tcn", 150), ("lstm", 600)])  # the LSTM learns the static map slower
def test_trainer_fits_synthetic_map(arch, steps):
    *train, test = _synthetic(n_rec=5, seed=0)  # the same map, the fifth recording held out
    pred = TCNModel(HISTORY, TCNConfig(**{**SMALL, "steps": steps}, arch=arch)).fit(train).predict(test.x)
    m = test.scored
    assert np.isfinite(pred[m]).all()
    mae = np.abs(pred[m] - test.y[m]).mean()
    constant = np.abs(test.y[m] - np.median(test.y[m], 0)).mean()
    assert mae < 0.5 * constant, (mae, constant)  # measured on the held-out recording: TCN 0.31, LSTM 0.41


@pytest.mark.parametrize("arch", ["tcn", "lstm"])
def test_state_file_round_trip_predicts_and_fine_tunes_identically(tmp_path, arch):
    train, test = _synthetic(), _synthetic(n_rec=1, seed=1)[0]
    model = TCNModel(HISTORY, TCNConfig(**{**SMALL, "steps": 20}, arch=arch)).fit(train)
    path = tmp_path / "m.state.pt"
    model.save_state(path, {"training": "unit test"})
    loaded = TCNModel.load_state(path)  # weights_only=True: no pickled classes
    np.testing.assert_array_equal(loaded.predict(test.x), model.predict(test.x))
    assert loaded.provenance == {"train_device": "cpu", "training": "unit test"}
    tuned = [m.fine_tuned(train[:1], steps=5, lr=1e-3).predict(test.x) for m in (model, loaded)]
    np.testing.assert_array_equal(*tuned)


def test_a_file_without_the_state_format_is_refused(tmp_path):
    torch.save({"format": "something else"}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="not a"):
        TCNModel.load_state(tmp_path / "x.pt")


def test_personal_share_needs_recordings_of_the_focus_subject():
    model = TCNModel(HISTORY, TCNConfig(**{**SMALL, "personal_share": 0.5}))
    for focus in (None, 99):
        with pytest.raises(ValueError, match="personal_share"):
            model.fit(_synthetic(), focus_subject=focus)


def test_shift_augmentation_needs_all_256_electrodes():
    with pytest.raises(ValueError, match="all 256 channels"):
        TCNModel(HISTORY, TCNConfig(**SMALL, aug_shift=1.0)).fit(_synthetic())


def _model(shift):
    return TCNModel(32, TCNConfig(aug_shift=shift))


def test_shift_zero_is_identity_and_full_step_moves_one_electrode():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 10, 256)).astype(np.float32)
    assert np.allclose(_model(0.0)._shift(x, np.random.default_rng(1)), x)
    m = _model(1.0)

    class FixedRng:  # rows: +1 step (take the electrode below), cols: 0
        calls = 0

        def uniform(self, lo, hi, size):
            FixedRng.calls += 1
            return np.full(size, 1.0 if FixedRng.calls == 1 else 0.0)

    out = m._shift(x, FixedRng())
    nb = m.neighbours
    assert np.allclose(out, x[:, :, nb["down"]])


def test_fractional_shift_interpolates_toward_the_neighbour_by_hand():
    """A 0.5-electrode shift is halfway to the neighbour. An electrode at the edge stays as it is."""
    m = _model(1.0)
    nb = m.neighbours
    x = np.arange(256, dtype=np.float32).reshape(1, 1, 256)  # one frame, distinct values per channel

    class Rows:  # +0.5 electrode along rows on every grid, no column shift
        calls = 0

        def uniform(self, lo, hi, size):
            Rows.calls += 1
            return np.full(size, 0.5 if Rows.calls == 1 else 0.0)

    out = m._shift(x, Rows())
    i = 1 * 8 + 2  # grid 0, row 1, column 2 (interior)
    expected = 0.5 * x[0, 0, i] + 0.5 * x[0, 0, nb["down"][i]]
    assert out[0, 0, i] == pytest.approx(expected)
    edge = 7 * 8 + 2  # bottom row: its "down" neighbour is itself, so nothing changes
    assert out[0, 0, edge] == pytest.approx(x[0, 0, edge])


def test_every_neighbour_is_in_the_same_grid_and_one_step_away():
    """No shift crosses a grid border or wraps a row: each neighbour is the electrode itself (at an edge) or one step."""
    nb = _model(1.0).neighbours
    ch = np.arange(256)
    grid, row, col = ch // 64, ch % 64 // 8, ch % 8
    for d, (dr, dc) in {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }.items():
        j = nb[d]
        assert (grid[j] == grid).all()
        moved = j != np.arange(len(ch))
        assert ((row[j] - row)[moved] == dr).all()
        assert ((col[j] - col)[moved] == dc).all()


def test_augmentation_is_training_only():
    """predict never shifts: the same input gives the same output whatever aug_shift was set to."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((80, 256)).astype(np.float32)
    seq = [Sequence("a", 1, 1, x, rng.standard_normal((80, 5)).astype(np.float32), np.ones(80, bool))]
    a, b = (TCNModel(32, TCNConfig(aug_shift=s, steps=1, batch=2, crop=40, device="cpu")).fit(seq) for s in (0.0, 1.0))
    b.net.load_state_dict(a.net.state_dict())  # same weights. Only the training option differs
    np.testing.assert_allclose(a.predict(x), b.predict(x), equal_nan=True)


H = 8


def _seq(subject, seed, n=300):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6)).astype(np.float32)
    w = rng.standard_normal((6, 5))
    return Sequence(f"s{subject}_{seed}", subject, 1, x, (x @ w).astype(np.float32), np.ones(n, bool))


def test_person_weighting_of_the_ridge():
    mine, other = _seq(1, 0), _seq(2, 1)
    x = _seq(3, 2).x
    plain = RidgeModel(H, 1.0).fit([mine, other]).predict(x)
    # two equal-length recordings, one each: half the weight on the person is the unweighted fit
    np.testing.assert_allclose(RidgeModel(H, 1.0, personal_share=0.5).fit([mine, other], focus_subject=1).predict(x), plain, atol=1e-8)
    # almost all the weight on the person approaches a fit on the person alone
    alone = RidgeModel(H, 1.0).fit([mine]).predict(x)
    nearly = RidgeModel(H, 1.0, personal_share=0.999).fit([mine, other], focus_subject=1).predict(x)
    assert np.nanmax(np.abs(nearly - alone)) < 0.05 * np.nanmax(np.abs(plain - alone))


def test_ridge_is_the_same_whether_its_statistics_are_cached_or_not():
    rng = np.random.default_rng(4)
    seqs = [
        Sequence(f"r{i}", 1 + i % 2, 1, rng.standard_normal((120, 6)).astype(np.float32), rng.standard_normal((120, 5)), np.ones(120, bool))
        for i in range(4)
    ]
    full, tiny = StatsCache(), StatsCache(max_gb=1e-6)  # tiny: keeps none, recomputes every time
    fits = [RidgeModel(8, 1.0, cache).fit(seqs) for cache in (None, full, full, tiny)]
    for m in fits[1:]:
        np.testing.assert_array_equal(m.coef, fits[0].coef)
    assert len(full) == 4
    assert len(tiny) == 0


TOL = 1e-4  # %MVC


def test_onnx_matches_torch_window_by_window(tmp_path):
    rng = np.random.default_rng(0)
    seqs = [
        Sequence(
            f"r{k}",
            1,
            1,
            rng.standard_normal((300, 16)).astype(np.float32),
            rng.standard_normal((300, 5)).astype(np.float32),
            np.arange(300) >= 40,
        )
        for k in range(2)
    ]
    m = TCNModel(32, TCNConfig(hidden=16, steps=5, batch=4, crop=64, device="cpu")).fit(seqs)
    onnx = OnnxModel(export_onnx(m, tmp_path / "m.onnx"), m.history)
    for _ in range(50):
        x = rng.standard_normal((32, 16)).astype(np.float32)
        a, b = m.predict(x), onnx.predict(x)
        np.testing.assert_allclose(a, b, rtol=0, atol=TOL, equal_nan=True)  # every row, not only the last
    assert np.isnan(onnx.predict(x)[: m.history - 1]).all()
    assert not np.isnan(onnx.predict(x)[m.history - 1 :]).any()


def test_onnx_decoder_refuses_a_window_of_the_wrong_length(tmp_path):
    rng = np.random.default_rng(1)
    seq = Sequence(
        "r", 1, 1, rng.standard_normal((200, 8)).astype(np.float32), rng.standard_normal((200, 5)).astype(np.float32), np.ones(200, bool)
    )
    m = TCNModel(32, TCNConfig(hidden=8, steps=2, batch=2, crop=64, device="cpu")).fit([seq])
    onnx = OnnxModel(export_onnx(m, tmp_path / "m.onnx"), m.history)
    for t in (31, 33, 64):
        with pytest.raises(ValueError, match="exactly 32 frames"):
            onnx.predict(rng.standard_normal((t, 8)).astype(np.float32))


def test_boot_ci_brackets_the_mean_and_collapses_on_constant_data():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    lo, hi = boot_ci(x)
    assert lo <= x.mean() <= hi
    assert x.min() <= lo < hi <= x.max()
    assert boot_ci(np.full(16, 3.5)) == (3.5, 3.5)
    assert boot_ci(x) == boot_ci(x)  # seeded: reports are reproducible
    assert boot_ci(x, seed=1) != boot_ci(x, seed=2)


def test_boot_ci_is_not_narrowed_by_sign_flips():
    """A paired difference and its negation give mirrored intervals."""
    d = np.array([-1.0, -0.5, 0.2, -2.0, -0.1, -0.7])
    lo, hi = boot_ci(d)
    nlo, nhi = boot_ci(-d)
    assert (lo, hi) == pytest.approx((-nhi, -nlo), abs=1e-9)


def _rows():
    # subject 1: one session, two recordings. Subject 2: two sessions with unequal recording counts
    rows = []
    for subject, session, rec, mae in [
        (1, 1, "a", 1.0),
        (1, 1, "b", 3.0),
        (2, 1, "c", 10.0),
        (2, 2, "d", 0.0),
        (2, 2, "e", 2.0),
        (2, 2, "f", 4.0),
    ]:
        for finger, extra in (("thumb", 0.0), ("index", 1.0)):
            rows.append(
                {"subject": subject, "session": session, "recording": rec, "finger": finger, "model": "m", "seed": 0, "mae": mae + extra}
            )
    return pd.DataFrame(rows)


def test_hierarchical_mean_weights_sessions_not_recordings():
    per = hierarchical_mean(_rows(), "mae", ["model"]).set_index("subject").mae
    assert per[1] == pytest.approx(2.5)  # fingers 1.5, 3.5 -> recordings -> one session
    assert per[2] == pytest.approx((10.5 + 2.5) / 2)  # session 1: 10.5. Session 2: mean(0.5, 2.5, 4.5) = 2.5. Then sessions
    assert summary(_rows(), "mae", ["model"]).loc[0, "mean"] == pytest.approx((2.5 + 6.5) / 2)


def test_run_summary_keeps_metrics_aligned_per_model_and_seed():
    """Each model's row carries its own metrics even when metric tables could be ordered differently."""
    df = _rows()
    other = df.assign(model="a_first", mae=df.mae * 10, rmse=df.mae * 100, r2=0.5, bias=-1.0)
    df = pd.concat([df.assign(rmse=df.mae * 2, r2=0.1, bias=1.0), other], ignore_index=True)
    out = run_summary(df, make_splits("cross_day", [1, 2]), 6, 0.0, {("m", 0): "cpu", ("a_first", 0): "mps"})
    by = {r["model"]: r for r in out["models"]}
    assert by["m"]["rmse"] == pytest.approx(2 * by["m"]["mae"])
    assert by["a_first"]["rmse"] == pytest.approx(10 * by["a_first"]["mae"])
    assert by["a_first"]["train_device"] == "mps"
    assert by["m"]["train_device"] == "cpu"


def test_mae_at_rest_uses_only_frames_with_the_label_below_5_percent():
    from experiments.metrics import recording_metrics

    y = np.zeros((100, 5))
    y[50:] = 40.0  # a press in the second half
    yhat = y + np.where(np.arange(100) < 50, 1.0, 10.0)[:, None]  # error 1 at rest, 10 while pressing
    row = recording_metrics(y, yhat, np.ones(100, bool))[0]
    assert row["mae_rest"] == pytest.approx(1.0)
    assert row["mae"] == pytest.approx(5.5)


@pytest.mark.parametrize("delay", [-12, 0, 7, 30])
def test_best_lag_recovers_a_known_delay(delay):
    rng = np.random.default_rng(0)
    y = np.convolve(rng.standard_normal(2000), np.ones(25) / 25, "same")  # smooth, force-like
    yhat = np.roll(y, delay)  # yhat[t] = y[t - delay]: late by `delay` frames
    assert best_lag(y[100:-100], yhat[100:-100]) == delay


def test_a_constant_prediction_has_no_timing():
    y = np.sin(np.linspace(0, 20, 1000))
    assert best_lag(y, np.zeros_like(y)) is None


def test_presses_in_both_directions_with_a_half_height_decoder():
    y = np.zeros(300)
    y[50:80] = 40.0  # extension press
    y[150:200] = -30.0  # flexion press
    y[250 : 250 + MIN_FRAMES - 1] = 50.0  # too short to count
    assert press_peaks(y, y / 2) == [(40.0, 20.0), (30.0, 15.0)]
    assert press_peaks(np.full(200, STRONG - 1), np.zeros(200)) == []
