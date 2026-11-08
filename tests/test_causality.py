"""Causality and streaming: no future sample changes a prediction, streaming equals offline, also with CAR and the input monitor."""

import copy
import itertools

import numpy as np
import pytest
from scipy import signal

from emg2force import dataset
from emg2force.causal import (
    INPUT_NORMS,
    WARMUP_FRAMES,
    FilterConfig,
    apply_montage,
    design_sos,
    frame_ends,
    log_rms,
    offline_frames,
)
from emg2force.hyser import emg_count_at
from emg2force.models import ConstantModel, RidgeModel, Sequence, lag_blocks
from emg2force.pipeline import StreamingDecoder, export_onnx, load_decoder, load_onnx_decoder, offline_predict
from emg2force.tcn import TCNConfig, TCNModel
from experiments.monitor_check import FAULT_CHANNELS, FAULT_S, first_flag, inject, stream
from experiments.run import run
from tests.helpers import SMOKE_CFG, make_fake_hyser

C, N_FRAMES, H = 12, 300, 32
SOS = design_sos()
# Tolerance 1e-4 %MVC absolute: offline (whole sequence) and streaming (H-frame window) sum float32 in
# different orders. Differences are ~1e-5 %MVC, 4 orders below any real error.
TOL = {"rtol": 1e-5, "atol": 1e-4}


def _emg(seed, n_frames=N_FRAMES):
    rng = np.random.default_rng(seed)
    n = int(emg_count_at(n_frames - 1))
    amp = 1e-4 * (1 + np.abs(np.sin(np.linspace(0, 6, n)))[:, None])
    return (amp * rng.standard_normal((n, C))).astype(np.float32)


def _train_seqs():
    seqs = []
    for k in range(3):
        x = offline_frames(_emg(k), N_FRAMES, SOS)
        y = np.repeat(x[:, :5].mean(1, keepdims=True), 5, 1) * 10.0
        scored = np.arange(N_FRAMES) >= H - 1
        seqs.append(Sequence(f"r{k}", 1, 1, x, y, scored))
    return seqs


def _fitted(name: str):
    seqs = _train_seqs()
    if name == "constant":
        return ConstantModel(H).fit(seqs)
    if name == "ridge":
        return RidgeModel(H, alpha=1.0).fit(seqs)
    cfg = TCNConfig(hidden=8, steps=5, batch=4, crop=64, dropout=0.0, weight_decay=1e-4, arch=name)  # pinned: no defaults
    return TCNModel(H, cfg).fit(seqs)


@pytest.fixture(scope="module", params=["constant", "ridge", "tcn", "lstm"])
def model(request):
    return _fitted(request.param)


@pytest.fixture(scope="module", params=["constant", "ridge", "tcn"])
def streamed_model(request):
    """The models the streaming decoder runs (the LSTM baseline is scored offline only)."""
    return _fitted(request.param)


@pytest.mark.parametrize("input_norm", INPUT_NORMS)
@pytest.mark.parametrize("cut", [H - 1, 100, 223, N_FRAMES - 2])
def test_future_mutation_full_pipeline(model, cut, input_norm):
    """Nothing after t_cut changes any prediction up to t_cut, for every model and input normalisation."""
    emg = _emg(10)
    base = offline_predict(emg, N_FRAMES, SOS, model, input_norm)
    mutated = emg.copy()
    k = int(emg_count_at(cut))  # first EMG sample that does not exist yet at t_cut
    mutated[k:] = 1e-2 * np.random.default_rng(99).standard_normal(mutated[k:].shape)
    pred = offline_predict(mutated, N_FRAMES, SOS, model, input_norm)
    np.testing.assert_allclose(pred[: cut + 1], base[: cut + 1], **TOL, equal_nan=True)
    if not isinstance(model, ConstantModel):
        assert not np.allclose(pred[cut + 1 :], base[cut + 1 :], equal_nan=True), "mutation had no effect"


def test_frame_boundary_is_tight():
    """The last EMG sample existing at t_j changes frame j. The next sample does not."""
    emg = _emg(3)
    base = offline_frames(emg, N_FRAMES, SOS)
    j = 150
    last = int(emg_count_at(j)) - 1
    e1 = emg.copy()
    e1[last] += 1e-3
    assert not np.allclose(offline_frames(e1, N_FRAMES, SOS)[j], base[j])
    e2 = emg.copy()
    e2[last + 1] += 1e-3
    np.testing.assert_array_equal(offline_frames(e2, N_FRAMES, SOS)[: j + 1], base[: j + 1])


def test_mutation_test_catches_noncausal_filter():
    def noncausal_frames(emg):
        y = signal.sosfiltfilt(SOS, emg.astype(np.float64), axis=0)
        ends = frame_ends(N_FRAMES)
        starts = np.concatenate([[0], ends[:-1]])
        return log_rms(np.add.reduceat(y[: ends[-1]] ** 2, starts, axis=0), (ends - starts)[:, None])

    emg = _emg(4)
    cut = 150
    mutated = emg.copy()
    mutated[int(emg_count_at(cut)) :] = 1e-2
    assert not np.allclose(noncausal_frames(mutated)[: cut + 1], noncausal_frames(emg)[: cut + 1], **TOL)


@pytest.mark.parametrize("input_norm", INPUT_NORMS)
@pytest.mark.parametrize("seed", [0, 1])
def test_streaming_equals_offline(streamed_model, seed, input_norm):
    """Chunked streaming reproduces the offline prediction frame for frame."""
    model = streamed_model
    rng = np.random.default_rng(seed)
    dec = StreamingDecoder(C, SOS, model, input_norm, monitor=False)  # fewer channels than the monitor's 4 grids
    out = {}
    for rec_seed in (20, 21):  # two recordings: state must reset at the boundary
        emg = _emg(rec_seed)
        dec.reset()
        stream = np.full((N_FRAMES, 5), np.nan)
        pos = 0
        while pos < len(emg):
            n = int(rng.choice([1, 7, 20, 21, 64, 300]))
            for j, p in dec.push(emg[pos : pos + n]):
                stream[j] = p
            pos += n
        out[rec_seed] = (stream, offline_predict(emg, N_FRAMES, SOS, model, input_norm))
    first = max(H - 1, WARMUP_FRAMES)  # first frame with a full window after the warm-up (scoring starts there too)
    for stream, offline in out.values():
        np.testing.assert_allclose(stream, offline, **TOL, equal_nan=True)
        assert np.isnan(stream[:first]).all()
        assert not np.isnan(stream[first:]).any()


def test_ridge_and_tcn_see_the_same_history():
    blocks = lag_blocks(H)
    assert blocks[0] == (0, 1)
    assert blocks[-1][1] == H
    assert all(b[1] == nb[0] for b, nb in itertools.pairwise(blocks))
    assert TCNConfig().receptive_field == H
    with pytest.raises(ValueError, match="receptive field"):
        TCNModel(H + 1)


def test_filter_refuses_a_sampling_rate_the_framing_does_not_use():
    with pytest.raises(ValueError, match="framing"):
        FilterConfig(fs=1000.0)


def test_held_press_fades_with_the_running_mean_but_not_without_normalisation():
    """2 s at rest, then a press held for 12 s: the normalised input 10 s into the hold, relative to 1 s in."""
    from emg2force.causal import normalize_input

    x = np.zeros((1400, 3), np.float32)
    x[200:] = 2.0  # log-RMS step at the press
    ratio = {m: normalize_input(x, m)[1200] / normalize_input(x, m)[300] for m in INPUT_NORMS}
    assert ratio["running_mean"][0] < 0.3  # 0.25 exactly: the press has become most of the baseline
    assert ratio["none"][0] == 1.0


def test_montages_keep_the_grid_structure():
    from emg2force.causal import apply_montage

    rng = np.random.default_rng(0)
    emg = rng.standard_normal((50, 256))
    car = apply_montage(emg, "car").reshape(50, 4, 64)
    np.testing.assert_allclose(car.mean(axis=2), 0, atol=1e-12)  # each grid's mean removed
    assert apply_montage(emg, "monopolar") is emg


N = 800  # frames of synthetic EMG


@pytest.fixture(scope="module")
def montage_decoder(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("car")
    make_fake_hyser(tmp / "h", subjects=(2,), trials=(1, 2, 3, 4, 5))
    cfg = copy.deepcopy(SMOKE_CFG)
    cfg["filter"] = {"montage": "car"}
    cfg["input_norm"] = "none"
    cfg["models"] = {"tcn": {**SMOKE_CFG["models"]["tcn"], "aug_shift": 1.0}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset, "CACHE", tmp / "cache")
        run(cfg, tmp / "h", tmp / "run", quiet=True)
    return "car", next((tmp / "run" / "models").glob("tcn_*.state.pt")), tmp


def _grid_emg(seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal((int(emg_count_at(N)), 256)) * 0.05).astype(np.float32)


def test_streaming_equals_offline_with_the_montage(montage_decoder):
    montage, path, _ = montage_decoder
    decoder = load_decoder(path)
    assert decoder.montage == montage
    emg = _grid_emg(1)
    pred, _ = stream(decoder, emg, monitor=False)
    offline = offline_predict(emg, N, decoder.front.sos, decoder.model, decoder.input_norm, decoder.n_base, montage)
    n = min(len(pred), len(offline))  # streaming also closes the frame that ends at the last sample
    np.testing.assert_allclose(pred[decoder.n_base : n], offline[decoder.n_base : n], rtol=1e-5, atol=1e-4)  # outputs are large


def test_onnx_equals_pytorch_with_the_montage(montage_decoder):
    _, path, tmp = montage_decoder
    torch_dec = load_decoder(path)
    onnx_dec = load_onnx_decoder(export_onnx(torch_dec.model, tmp / "m.onnx"))
    emg = _grid_emg(2)
    a = np.array([p for _, p in torch_dec.push(emg)])
    b = np.array([p for _, p in onnx_dec.push(emg)])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("fault", ["dead_row", "noisy_channel"])
def test_monitor_flags_raw_electrodes_and_fills_their_montage_channels(montage_decoder, fault):
    montage, path, _ = montage_decoder
    decoder = load_decoder(path)
    clean, _ = stream(decoder, _grid_emg(3), monitor=True)
    bad, flags = stream(decoder, inject(_grid_emg(3), fault), monitor=True)
    k = int(FAULT_S * 100)
    assert not flags[:k].any()
    delay = first_flag(flags, FAULT_CHANNELS[fault])
    assert delay is not None
    assert delay < 50
    after = slice(k + 60, None)  # once the fault is flagged, the output stays near the clean output
    assert np.nanmean(np.abs(bad[after] - clean[after])) < 0.5 * np.nanmean(np.abs(clean[after])) + 0.5


def test_excluded_electrodes_leave_the_montage():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((20, 256))
    exclude = np.zeros(256, bool)
    exclude[64 + 9] = True  # extensor-proximal grid, row 1, column 1
    car = apply_montage(x, "car", exclude)
    assert (car[:, 64 + 9] == 0).all()
    keep = np.arange(64, 128) != 64 + 9
    np.testing.assert_allclose(car[:, 64:128][:, keep].sum(axis=1), 0, atol=1e-12)  # the mean leaves out the flagged one


def test_monitor_output_does_not_depend_on_the_chunk_size(montage_decoder):
    """A noisy electrode flagged mid-recording: the whole recording in one push gives the frame-by-frame output."""
    _, path, _ = montage_decoder
    emg = inject(_grid_emg(3), "noisy_channel")
    framewise, _ = stream(load_decoder(path), emg, monitor=True)
    decoder = load_decoder(path)
    whole = np.array([p for _, p in decoder.push(emg)])
    assert decoder.replaced.any()
    n = min(len(framewise), len(whole))
    np.testing.assert_allclose(whole[:n], framewise[:n], rtol=1e-5, atol=1e-4, equal_nan=True)
