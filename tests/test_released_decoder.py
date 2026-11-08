"""Released decoder through the streaming implementation on synthetic EMG: PyTorch and ONNX agree within 1e-4 %MVC,
and chunking, future EMG and resets change nothing beyond that."""

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from emg2force import benchmark
from emg2force.causal import FilterConfig, design_sos
from emg2force.hyser import emg_count_at
from emg2force.pipeline import OnnxModel, StreamingDecoder, export_onnx, load_decoder, offline_predict
from emg2force.tcn import TCNModel

STATE = Path(__file__).resolve().parents[1] / "assets" / "decoder.state.pt"
TOL = 1e-4  # %MVC
N_FRAMES = 300
SOS = design_sos(FilterConfig())
pytestmark = pytest.mark.skipif(not STATE.exists(), reason="assets/decoder.state.pt missing")


@cache
def _front_end() -> tuple[str, str]:
    """The released decoder's own input normalisation and montage, so every test runs the shipped settings."""
    fe = TCNModel.load_state(STATE).front_end
    return fe["input_norm"], fe["filter"].get("montage", "monopolar")


@pytest.fixture(scope="module")
def released():
    return TCNModel.load_state(STATE)


@pytest.fixture(scope="module")
def onnx_model(released, tmp_path_factory):
    return OnnxModel(export_onnx(released, tmp_path_factory.mktemp("onnx") / "tcn.onnx"), released.history)


def _noise(seed: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """White EMG-like noise, ~50 uV (header units are mV)."""
    return (np.random.default_rng(seed).standard_normal((int(emg_count_at(n_frames)), 256)) * 0.05).astype(np.float32)


def _emg(seed: int, n_frames: int = N_FRAMES) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(emg_count_at(n_frames - 1))
    envelope = 1 + np.abs(np.sin(np.linspace(0, 6, n)))[:, None] * rng.uniform(0.5, 2, 256)
    return (5e-2 * envelope * rng.standard_normal((n, 256))).astype(np.float32)  # header units (mV), ~50 uV


def _stream(model, emg: np.ndarray, chunks, decoder=None) -> np.ndarray:
    norm, montage = _front_end()
    dec = decoder or StreamingDecoder(256, SOS, model, norm, montage=montage)
    out, pos, i = np.full((N_FRAMES, 5), np.nan), 0, 0
    while pos < len(emg):
        n = chunks[i % len(chunks)]
        for j, p in dec.push(emg[pos : pos + n]):
            if j < N_FRAMES:
                out[j] = p
        pos, i = pos + n, i + 1
    return out


def _close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=TOL, equal_nan=True)


def test_offline_and_streaming_pytorch_agree(released):
    emg = _emg(0)
    _close(_stream(released, emg, [21]), offline_predict(emg, N_FRAMES, SOS, released, _front_end()[0], montage=_front_end()[1]))


def test_streaming_onnx_matches_streaming_pytorch_frame_for_frame(released, onnx_model):
    emg = _emg(1)
    a, b = _stream(released, emg, [20, 21]), _stream(onnx_model, emg, [20, 21])
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.isnan(a[: released.history - 1]).all()
    _close(a, b)


@pytest.mark.parametrize("chunks", [[1], [7, 300], [64], [2048]])
def test_chunk_boundaries_do_not_change_the_output(released, chunks):
    emg = _emg(2)
    _close(_stream(released, emg, chunks), _stream(released, emg, [21]))


def test_future_samples_never_change_earlier_predictions(onnx_model):
    emg = _emg(3)
    cut = 180
    mutated = emg.copy()
    mutated[int(emg_count_at(cut)) :] = np.random.default_rng(9).normal(0, 1, mutated[int(emg_count_at(cut)) :].shape)
    a, b = _stream(onnx_model, emg, [33]), _stream(onnx_model, mutated, [33])
    assert np.array_equal(a[: cut + 1], b[: cut + 1], equal_nan=True)
    assert not np.allclose(a[cut + 1 :], b[cut + 1 :])


def test_reset_equals_a_fresh_decoder(onnx_model):
    first, second = _emg(4), _emg(5)
    norm, montage = _front_end()
    dec = StreamingDecoder(256, SOS, onnx_model, norm, montage=montage)
    _stream(onnx_model, first, [50], dec)
    dec.reset()
    assert np.array_equal(_stream(onnx_model, second, [50], dec), _stream(onnx_model, second, [50]), equal_nan=True)


def test_load_decoder_rebuilds_the_released_front_end_from_the_file(released):
    """The state file records its input normalisation, montage, the default filter and the 0.7 s warm-up. Outputs match the manual build."""
    assert released.front_end["input_norm"] in ("none", "running_mean")
    montage = released.front_end["filter"]["montage"]
    manual = StreamingDecoder(256, SOS, released, released.front_end["input_norm"], montage=montage)
    from_file = load_decoder(STATE)
    emg = _emg(3)
    a = np.array([p for _, p in manual.push(emg)])
    b = np.array([p for _, p in from_file.push(emg)])
    np.testing.assert_array_equal(a, b)


def test_warm_up_bad_samples_and_empty_chunks():
    """Silent until the 0.7 s warm-up ends. One NaN sample is replaced and counted, not propagated. Empty chunks are a no-op."""
    decoder = load_decoder(STATE)
    emg = _noise(5)
    emg[int(emg_count_at(100)), 7] = np.nan
    assert decoder.push(emg[:0]) == []
    with pytest.raises(ValueError, match="expected EMG of shape"):
        decoder.push(emg[:10, :128])  # wrong channel count
    with pytest.raises(ValueError, match="expected EMG of shape"):
        decoder.push(emg[:10, 0])  # 1-D
    out = np.array([p for _, p in decoder.push(emg)])
    assert np.isnan(out[: decoder.n_base]).all()
    assert np.isfinite(out[decoder.n_base :]).all()
    assert decoder.nonfinite == 1
    decoder.reset()
    assert decoder.nonfinite == 0


def test_onnx_decoder_runs_without_torch(released, tmp_path):
    """export_onnx writes the front end beside the graph. load_onnx_decoder rebuilds the same decoder in a torch-free process."""
    import subprocess
    import sys

    path = export_onnx(released, tmp_path / "decoder.onnx")
    emg = _noise(6)
    np.save(tmp_path / "emg.npy", emg)
    want = np.array([p for _, p in load_decoder(STATE).push(emg)])
    code = (
        "import sys, numpy as np\n"
        "from emg2force.pipeline import load_onnx_decoder\n"
        f"d = load_onnx_decoder({str(path)!r})\n"
        f"np.save({str(tmp_path / 'got.npy')!r}, np.array([p for _, p in d.push(np.load({str(tmp_path / 'emg.npy')!r}))]))\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    np.testing.assert_allclose(np.load(tmp_path / "got.npy"), want, rtol=0, atol=TOL, equal_nan=True)


DATA_ROOT = Path(__file__).resolve().parents[1].parent / "hyser_raw"


@pytest.mark.data
@pytest.mark.skipif(not (DATA_ROOT / "random_dataset/subject02_session2/random_raw_sample3.dat").exists(), reason="no data")
def test_streaming_equals_offline_on_a_real_recording():
    """The released decoder on a full 25 s Hyser recording: random chunk sizes reproduce the offline frames and prediction.

    The monitor is off, as in the offline evaluation. experiments.monitor_check measures what it changes on real data."""
    from emg2force.causal import StreamingFrontEnd, offline_frames
    from emg2force.hyser import Hyser

    emg = Hyser(DATA_ROOT).load(2, 2, 3).emg
    decoder = load_decoder(STATE)
    decoder.monitor = False
    n = 2500
    front, frames, pos = StreamingFrontEnd(256, decoder.front.sos), [], 0
    for step in (1, 997, 4096, 13) * 5000:
        frames += [f for _, f in front.push(emg[pos : pos + step])]
        pos += step
        if pos >= len(emg):
            break
    np.testing.assert_array_equal(np.stack(frames)[:n], offline_frames(emg, n, decoder.front.sos))
    offline = offline_predict(emg, n, decoder.front.sos, decoder.model, decoder.input_norm, decoder.n_base, montage=decoder.montage)
    stream = np.full((n, 5), np.nan)
    rng, pos = np.random.default_rng(7), 0
    while pos < len(emg):
        step = int(rng.choice([1, 13, 20, 21, 997, 4096]))
        for j, p in decoder.push(emg[pos : pos + step]):
            if j < n:
                stream[j] = p
        pos += step
    np.testing.assert_allclose(stream, offline, rtol=0, atol=TOL, equal_nan=True)
    assert np.isfinite(stream[decoder.n_base :]).all()


def test_monitor_replaces_a_dead_electrode_row_and_leaves_steady_noise_alone():
    """A row of 8 electrodes goes dead after the warm-up: flagged within a frame, and the output stays near the clean one."""
    emg = _noise(8)
    dead = emg.copy()
    dead[int(emg_count_at(150)) :, 64:72] = 0.0
    run = {}
    for name, x, monitor in (("clean", emg, True), ("on", dead, True), ("off", dead, False)):
        d = load_decoder(STATE)
        d.monitor = monitor
        run[name] = (np.array([p for _, p in d.push(x)]), d)
    assert run["clean"][1].replaced_frames == 0
    assert run["on"][1].replaced[64:72].all()
    err = {k: np.abs(run[k][0][160:] - run["clean"][0][160:]).mean() for k in ("on", "off")}
    assert err["on"] < 0.5 * err["off"]


def test_monitor_ignores_rest_after_a_long_press():
    """20 s of strong activity on the flexor grids, then rest: healthy channels must not be flagged afterwards."""
    rng = np.random.default_rng(11)
    n = int(emg_count_at(3000))
    amp = np.full((n, 256), 0.01, np.float32)
    amp[int(emg_count_at(200)) : int(emg_count_at(2200)), 128:] = 0.5  # 50x RMS: well past DROP_LOG
    d = load_decoder(STATE)
    d.push((amp * rng.standard_normal((n, 256))).astype(np.float32))
    assert d.replaced_frames == 0


def test_monitor_check_switches_the_monitor_on_for_a_decoder_without_input_normalisation(released):
    from experiments.monitor_check import stream

    emg = _noise(10)
    emg[int(emg_count_at(150)) :, 64:72] = 0.0
    d = StreamingDecoder(256, SOS, released, "none")
    _, on = stream(d, emg, monitor=True)
    _, off = stream(d, emg, monitor=False)
    assert on[:, 64:72].any()
    assert not on[:, :64].any()
    assert not off.any()


def test_monitor_flags_a_noisy_channel_after_the_fault_only():
    """A loose contact: one channel picks up noise far above its level. It is replaced, its neighbours are not."""
    from experiments.monitor_check import FAULT_CHANNELS, FAULT_S, first_flag, inject, stream

    emg = _noise(13, 800)
    _, flags = stream(load_decoder(STATE), inject(emg, "noisy_channel"), monitor=True)
    k = int(FAULT_S * 100)
    assert not flags[:k].any()
    delay = first_flag(flags, FAULT_CHANNELS["noisy_channel"])
    assert delay is not None
    assert 0 <= delay < 5
    others = np.ones(flags.shape[1], bool)
    others[FAULT_CHANNELS["noisy_channel"]] = False
    assert not flags[k:, others].any()


def test_benchmark_writes_parity_and_every_timing_mode(tmp_path):
    benchmark.main(["--seconds", "1.5", "--out", str(tmp_path)])
    res = json.loads((tmp_path / "latency_256ch_synthetic.json").read_text())
    assert res["setup"]["channels"] == 256
    assert res["setup"]["weights"].endswith("decoder.state.pt")
    parity = res["pytorch_vs_onnx_decoder"]
    assert parity["same_undefined_frames"]
    assert parity["max_abs_diff_pct_mvc"] < 1e-3
    for mode in ("model_only", "pipeline_sustained", "pipeline_paced_sleep", "pipeline_paced_sleep_pytorch"):
        assert res[mode]["n"] > 0
        assert res[mode]["p99_ms"] > 0
