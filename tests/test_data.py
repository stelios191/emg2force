"""Loader, splits, feature cache, leakage and download: the data side, on real excerpts and synthetic WFDB files."""

import dataclasses
import hashlib
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import emg2force.dataset as dataset
from emg2force.causal import FilterConfig
from emg2force.download import fetch, main, select
from emg2force.hyser import (
    EMG_CHANNELS,
    FS_EMG,
    Hyser,
    emg_count_at,
    make_target,
    normalize_force,
    on_emg_clock,
    read_header,
    read_record,
)
from emg2force.models import ChannelScaler, Model, Sequence
from emg2force.splits import (
    DEV_SEED,
    DEV_SUBJECTS,
    EVAL_SUBJECTS,
    Split,
    check_calibration,
    check_split,
    make_splits,
    own_day1,
    session_of,
    subject_of,
)
from emg2force.tcn import TCNConfig, TCNModel
from experiments.run import evaluate_split
from tests.helpers import make_fake_hyser, write_record

FIXTURE = Path(__file__).resolve().parent / "data" / "hyser_s04_d1_t1"


def test_read_record_applies_gain_and_baseline(tmp_path):
    d = np.array([[100, -50], [2674, 0]])
    write_record(tmp_path / "r", d, [74674.9621, 2.0], [2674, -10], ["a", "b"], 100)
    x, h = read_record(tmp_path / "r")
    np.testing.assert_allclose(x, [[(100 - 2674) / 74674.9621, (-50 + 10) / 2.0], [0.0, 5.0]])
    assert h.signals[0].units == "V"
    assert h.fs == 100


def test_real_header_line_parses(tmp_path):
    p = tmp_path / "random_force_sample1.hea"
    p.write_text("random_force_sample1 1 100 2500\n" "random_force_sample1.dat 16 74674.9621(2674)/V 0 0 2934 -3053 0 thumb\n")
    s = read_header(p).signals[0]
    assert (s.gain, s.baseline, s.units, s.name) == (74674.9621, 2674, "V", "thumb")


def test_emg_reordered_to_canonical_grid_order(tmp_path):
    make_fake_hyser(tmp_path, subjects=(1,), sessions=(1,))
    stem = tmp_path / "random_dataset/subject01_session1/random_raw_sample1"
    stored, h = read_record(stem)
    rec = Hyser(tmp_path).load(1, 1, 1)
    names = [s.name for s in h.signals]
    for c in (0, 7, 63, 64, 200, 255):
        np.testing.assert_allclose(rec.emg[:, c], stored[:, names.index(EMG_CHANNELS[c])].astype(np.float32))
    assert EMG_CHANNELS[0] == "ED-1-1"
    assert EMG_CHANNELS[255] == "FP-8-8"


def test_emg_count_at_matches_exact_rational_timestamps():
    for j in range(2501):
        t = Fraction(j, 100)
        expected = int(t * 2048) + 1  # k / 2048 <= t for k = 0..floor(2048 t), exact rational
        assert emg_count_at(j) == expected
    assert emg_count_at(1) == 21  # 20.48 -> samples 0..20
    assert emg_count_at(25) == 513  # exactly 512 / 2048 = 0.25 s, inclusive


def test_normalize_force_keeps_sign_and_uses_direction_mvc():
    mvc = np.tile([[2.0, 4.0]], (5, 1))  # flexion 2 V, extension 4 V
    f = np.array([[-1.0, 1.0, 0.0, -2.0, 4.0]])
    np.testing.assert_allclose(normalize_force(f, mvc), [[-50.0, 25.0, 0.0, -100.0, 100.0]])


def test_mvc_matches_reference_definition(tmp_path):
    truth = make_fake_hyser(tmp_path)
    h = Hyser(tmp_path)
    for (s, d), mvc in truth.items():
        np.testing.assert_allclose(h.mvc(s, d), mvc)


def test_mvc_conventions_never_mix(tmp_path):
    truth = make_fake_hyser(tmp_path)
    h = Hyser(tmp_path)
    rec = h.load(2, 2, 1)
    t_sess = h.target(rec, "session", "none")
    t_day1 = h.target(rec, "day1", "none")
    assert t_sess.mvc_source == "subject02_session2"
    assert t_day1.mvc_source == "subject02_session1"
    np.testing.assert_allclose(t_sess.mvc, truth[(2, 2)])
    np.testing.assert_allclose(t_day1.mvc, truth[(2, 1)])
    np.testing.assert_allclose(t_sess.values, normalize_force(rec.force_v, truth[(2, 2)]))
    # session 1 under both conventions is the same normalisation
    r1 = h.load(2, 1, 1)
    np.testing.assert_allclose(h.target(r1, "session").values, h.target(r1, "day1").values)
    with pytest.raises(ValueError, match="unknown MVC convention"):
        h.mvc_for(2, 2, "pooled")


def test_label_filter_is_explicit():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(500, 5))
    mvc = np.ones((5, 2))
    raw = make_target(f, mvc, "none")
    filt = make_target(f, mvc, "zero_phase_lowpass_10hz")
    assert raw.shape == filt.shape == (500, 5)
    assert np.std(np.diff(filt, axis=0)) < np.std(np.diff(raw, axis=0))
    with pytest.raises(ValueError, match="unknown label filter"):
        make_target(f, mvc, "smooth")


def test_reader_matches_wfdb_on_a_real_recording_excerpt():
    wfdb = pytest.importorskip("wfdb")
    for kind, shape in (("raw", (1024, 256)), ("force", (50, 5))):
        ours, h = read_record(FIXTURE / f"random_{kind}_sample1")
        np.testing.assert_allclose(ours, wfdb.rdrecord(str(FIXTURE / f"random_{kind}_sample1")).p_signal, rtol=1e-6, atol=1e-9)
        assert ours.shape == shape
        assert h.signals[0].units == "V"


def test_emg_labelled_volts_has_millivolt_amplitudes():
    """20-500 Hz RMS of surface EMG is tens of uV: read as mV that fits. Read as V it would be tens of mV."""
    emg, _ = read_record(FIXTURE / "random_raw_sample1")
    sos = signal.butter(4, (20, 500), btype="bandpass", fs=FS_EMG, output="sos")
    rms_mv = np.median(np.sqrt((signal.sosfiltfilt(sos, emg, axis=0) ** 2).mean(0)))
    assert 0.005 < rms_mv < 1.0, rms_mv  # 5 uV to 1 mV


def test_cached_mvc_values_cannot_be_changed_in_place(tmp_path):
    make_fake_hyser(tmp_path, subjects=(1,), sessions=(1,))
    mvc = Hyser(tmp_path).mvc(1, 1)
    with pytest.raises(ValueError, match="read-only"):
        mvc[0, 0] = 0.0


def test_force_on_the_emg_clock_samples_the_force_at_the_scaled_times():
    t = np.arange(1000.0)
    force = np.stack([np.sin(t / 37), np.cos(t / 53)], axis=1)
    out = on_emg_clock(force, 1.0026)
    j = np.arange(len(out))
    assert len(out) == 997  # the last sample, 996 x 1.0026 = 998.6, is the last time inside the force recording
    np.testing.assert_allclose(out[:, 0], np.sin(j * 1.0026 / 37), atol=1e-3)
    np.testing.assert_array_equal(on_emg_clock(force, 1.0), force)


def test_dev_and_eval_subjects_partition_all_twenty():
    assert len(DEV_SUBJECTS) == 4
    assert len(EVAL_SUBJECTS) == 16
    assert set(DEV_SUBJECTS) | set(EVAL_SUBJECTS) == set(range(1, 21))
    assert not set(DEV_SUBJECTS) & set(EVAL_SUBJECTS)


def test_dev_subjects_are_the_seeded_draw():
    assert tuple(sorted(int(s) for s in np.random.default_rng(DEV_SEED).choice(range(1, 21), 4, replace=False))) == DEV_SUBJECTS


@pytest.mark.parametrize("protocol", ["cross_day", "cross_day_pop", "cross_day_pop_cal", "loso"])
def test_no_recording_in_train_and_test(protocol):
    for sp in make_splits(protocol, range(1, 21)):
        assert not set(sp.train) & set(sp.test)


def test_loso_held_out_subject_absent_from_training():
    splits = make_splits("loso", EVAL_SUBJECTS)
    assert len(splits) == 16
    for sp in splits:
        assert sp.held_out_subject not in {subject_of(k) for k in sp.train}
        assert {subject_of(k) for k in sp.test} == {sp.held_out_subject}
        assert len({subject_of(k) for k in sp.train}) == 19  # dev subjects may train final folds
        assert {session_of(k) for k in sp.test} == {1, 2}


def test_cross_day_is_same_person_day1_to_day2():
    for sp in make_splits("cross_day", EVAL_SUBJECTS):
        assert {subject_of(k) for k in sp.train + sp.test} == {sp.held_out_subject}
        assert {session_of(k) for k in sp.train} == {1}
        assert {session_of(k) for k in sp.test} == {2}


def test_check_split_rejects_leaks():
    good = make_splits("loso", [5])[0]
    with pytest.raises(ValueError, match="in both train and test"):
        check_split(dataclasses.replace(good, train=(*good.train, good.test[0])))
    with pytest.raises(ValueError, match="held-out subject in training"):
        check_split(Split("loso", "x", ("s05_d1_t1",), ("s05_d2_t1",), 5))
    with pytest.raises(ValueError, match="must train on session 1"):
        check_split(Split("cross_day", "x", ("s05_d2_t1",), ("s05_d1_t1",), 5))
    with pytest.raises(ValueError, match="mixes subjects"):
        check_split(Split("cross_day", "x", ("s05_d1_t1", "s06_d1_t1"), ("s05_d2_t1",), 5))
    with pytest.raises(ValueError, match="someone other than the held-out subject"):
        check_split(Split("loso", "x", ("s06_d1_t1",), ("s05_d2_t1", "s07_d2_t1"), 5))


def test_cross_day_pop_never_sees_the_subjects_second_day():
    for sp in make_splits("cross_day_pop", EVAL_SUBJECTS + DEV_SUBJECTS):
        s = sp.held_out_subject
        mine = [k for k in sp.train if subject_of(k) == s]
        assert mine
        assert all(session_of(k) == 1 for k in mine)
        assert {subject_of(k) for k in sp.train} == set(range(1, 21))
        assert all(subject_of(k) == s and session_of(k) == 2 for k in sp.test)
        leaky = dataclasses.replace(sp, train=(*sp.train, f"s{s:02d}_d2_t1"), test=sp.test[1:])
        with pytest.raises(ValueError, match="only the subject's session 1"):
            check_split(leaky)


def test_own_day1_is_the_scored_persons_session1_training_recordings():
    for sp in make_splits("cross_day_pop", DEV_SUBJECTS):
        keys = own_day1(sp)
        assert keys
        assert all(subject_of(k) == sp.held_out_subject and session_of(k) == 1 for k in keys)
        assert keys == [k for k in sp.train if k in keys]  # training order kept


def test_calibration_designs_must_use_the_persons_own_unused_test_session_recordings():
    sp = make_splits("cross_day_pop_cal", [2])[0]
    check_calibration(sp, ["s02_d2_t1"])
    with pytest.raises(ValueError, match="also training or test"):
        check_calibration(sp, [sp.test[0]])  # a scored recording used as calibration
    with pytest.raises(ValueError, match="also training or test"):
        check_calibration(sp, ["s02_d1_t1"])  # a training recording
    for wrong in ("s02_d1_t1", "s03_d2_t1"):  # wrong session. Another person's (neither used for training here)
        unused = dataclasses.replace(sp, train=tuple(k for k in sp.train if k != wrong))
        with pytest.raises(ValueError, match="scored person's own, from the test session"):
            check_calibration(unused, [wrong])


def test_cache_reuses_then_recomputes_when_source_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "CACHE", tmp_path / "cache")
    make_fake_hyser(tmp_path / "data", subjects=(1,), sessions=(1,))
    ds, key, cfg = Hyser(tmp_path / "data"), "s01_d1_t1", FilterConfig()
    first = dataset.frames_for(ds, key, cfg)
    cached = next((tmp_path / "cache").rglob("*.npy"))
    np.save(cached, np.zeros_like(first))  # poison the cache: a hit must return this
    assert np.all(dataset.frames_for(ds, key, cfg) == 0)
    # source changes (new mtime) -> poisoned frames discarded and recomputed
    dat = tmp_path / "data" / "random_dataset" / "subject01_session1" / "random_raw_sample1.dat"
    st = dat.stat()
    os.utime(dat, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    np.testing.assert_array_equal(dataset.frames_for(ds, key, cfg), first)


def test_cache_folder_changes_with_the_filter_and_the_frame_version(monkeypatch):
    assert dataset.filter_tag(FilterConfig()) != dataset.filter_tag(FilterConfig(notch_q=10.0))
    before = dataset.code_tag()
    monkeypatch.setattr(dataset, "FRAME_VERSION", dataset.FRAME_VERSION + 1)
    assert dataset.code_tag() != before


class SpyModel(Model):
    history = 1

    def fit(self, train, focus_subject=None):
        self.fit_keys = {s.key for s in train}
        self.fit_x = np.concatenate([s.x for s in train])
        return self

    def predict(self, x):
        self.predicted = getattr(self, "predicted", 0) + 1
        return np.zeros((len(x), 5))


def _seqs(keys):
    out = {}
    for i, k in enumerate(keys):
        x = np.full((60, 4), float(i), np.float32)
        out[k] = Sequence(k, int(k[1:3]), int(k[5]), x, np.zeros((60, 5), np.float32), np.ones(60, bool))
    return out


def test_runner_fits_on_train_recordings_only():
    for protocol in ("cross_day", "cross_day_pop", "loso"):
        for sp in make_splits(protocol, [5]):
            seqs = _seqs(sp.train + sp.test)
            m = SpyModel()
            rows, preds = evaluate_split(sp, seqs, m)
            assert m.fit_keys == set(sp.train)
            test_values = {float(seqs[k].x[0, 0]) for k in sp.test}
            assert not test_values & set(np.unique(m.fit_x).tolist())
            assert set(preds) == set(sp.test)
            assert m.predicted == len(sp.test)


def test_network_standardisation_is_fitted_on_training_recordings_only():
    """The TCN's input mean/std buffers equal a scaler fitted on split.train. The test recordings have a different level."""
    sp = make_splits("cross_day", [5])[0]
    rng = np.random.default_rng(0)
    seqs = {}
    for k in sp.train + sp.test:
        level = 0.0 if k in sp.train else 50.0  # test recordings sit far from the training distribution
        x = (level + rng.standard_normal((80, 4))).astype(np.float32)
        seqs[k] = Sequence(k, int(k[1:3]), int(k[5]), x, rng.standard_normal((80, 5)).astype(np.float32), np.arange(80) >= 31)
    model = TCNModel(32, TCNConfig(hidden=4, steps=1, batch=2, crop=40, device="cpu"))
    evaluate_split(sp, seqs, model)
    scaler = ChannelScaler().fit([seqs[k] for k in sp.train])
    np.testing.assert_allclose(model.net.mean.numpy().ravel(), scaler.mean, rtol=1e-6)
    np.testing.assert_allclose(model.net.std.numpy().ravel(), scaler.std, rtol=1e-6)
    assert abs(model.net.mean.mean().item()) < 1.0  # not pulled toward the test level of 50


MANIFEST = {
    "random_dataset/subject01_session1/random_raw_sample1.dat": "a",
    "random_dataset/subject02_session1/random_force_sample1.hea": "b",
    "mvc_dataset/subject01_session2/mvc_force_finger1_flexion.dat": "c",
    "pr_dataset/subject01_session1/maintenance_raw_sample1.dat": "d",
}


def test_select_filters_by_subset_and_subject():
    assert select(MANIFEST, ["random"], None) == sorted(k for k in MANIFEST if k.startswith("random"))
    assert select(MANIFEST, ["random", "mvc"], [1]) == [
        "mvc_dataset/subject01_session2/mvc_force_finger1_flexion.dat",
        "random_dataset/subject01_session1/random_raw_sample1.dat",
    ]


def test_fetch_verifies_the_checksum_and_reuses_a_good_copy(tmp_path):
    src, root = tmp_path / "mirror", tmp_path / "root"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.dat").write_bytes(b"emg")
    base = src.as_uri() + "/"
    good = hashlib.sha256(b"emg").hexdigest()
    assert fetch(root, "d/f.dat", good, base) == ("d/f.dat", "ok")
    assert fetch(root, "d/f.dat", good, base) == ("d/f.dat", "cached")
    rel, status = fetch(root, "d/g.dat", good, base, retries=1)  # missing file
    assert status.startswith("FAILED")
    (src / "d" / "h.dat").write_bytes(b"corrupt")
    assert fetch(root, "d/h.dat", good, base, retries=2) == ("d/h.dat", "FAILED: checksum mismatch")
    assert not (root / "d" / "h.dat").exists()
    assert not (root / "d" / "h.dat.part").exists()


def test_download_fetches_the_selected_subset_and_reports_failures(tmp_path, capsys):
    mirror, root = tmp_path / "mirror", tmp_path / "root"
    files = {
        "random_dataset/subject01_session1/random_raw_sample1.dat": b"emg",
        "random_dataset/subject01_session1/random_raw_sample1.hea": b"hdr",
        "random_dataset/subject02_session1/random_raw_sample1.dat": b"other person",
        "mvc_dataset/subject01_session1/mvc_force_finger1_flexion.dat": b"mvc",
    }
    for rel, data in files.items():
        (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
        (mirror / rel).write_bytes(data)
    (root / "meta").mkdir(parents=True)
    sums = "".join(f"{hashlib.sha256(d).hexdigest()} {rel}\n" for rel, d in files.items())
    (root / "meta" / "SHA256SUMS.txt").write_text(sums)  # present, so nothing is fetched from PhysioNet
    args = ["--root", str(root), "--subset", "random", "--subjects", "1", "--base-url", mirror.as_uri() + "/"]

    main(args)  # returns normally when every file arrives
    got = sorted(str(p.relative_to(root)) for p in root.rglob("*.*") if p.parent.name != "meta")
    assert got == sorted(k for k in files if k.startswith("random_dataset/subject01"))
    (mirror / "random_dataset/subject01_session1/random_raw_sample1.hea").write_bytes(b"changed upstream")
    (root / "random_dataset/subject01_session1/random_raw_sample1.hea").unlink()
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 1
    assert "FAILED random_dataset/subject01_session1/random_raw_sample1.hea" in capsys.readouterr().out
