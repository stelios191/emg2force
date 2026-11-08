"""Offline and streaming decoders built from the same front end and model.

Running an exported decoder (`load_onnx_decoder`) needs numpy, scipy and onnxruntime only. torch is imported on use.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort

from emg2force.causal import (
    INPUT_NORMS,
    WARMUP_FRAMES,
    FilterConfig,
    StreamingFrontEnd,
    apply_montage,
    design_sos,
    normalize_input,
    offline_frames,
)
from emg2force.hyser import FS_FORCE, GRID_CHANNELS, GRIDS
from emg2force.models import Predictor

if TYPE_CHECKING:
    from emg2force.tcn import TCNModel


class OnnxModel:
    """ONNX Runtime backend with the Predictor interface. The graph takes one (1, C, history) window."""

    def __init__(self, path: Path, history: int, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.history = history

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.history:  # the exported graph has a fixed window. A wrong length must not pass silently
            raise ValueError(f"ONNX decoder takes exactly {self.history} frames, got {x.shape[0]}")
        out = self.session.run(None, {"frames": np.ascontiguousarray(x.T[None], dtype=np.float32)})[0][0].T.astype(np.float64)
        out[: self.history - 1] = np.nan
        return out


def export_onnx(model: TCNModel, path: Path) -> Path:
    """Export a TCN's network (standardisation buffers included) for a (1, C, history) window, opset 17.

    The front end the weights were trained with goes to `<path>.json`, so `load_onnx_decoder` needs no torch.
    """
    import torch

    if model.cfg.arch != "tcn":
        raise ValueError("only the TCN is exported. The LSTM baseline is scored offline")
    net = model.net.eval()
    n_channels = net.inp.in_channels
    dummy = torch.randn(1, n_channels, model.history)
    with warnings.catch_warnings():  # the legacy (TorchScript) exporter, used on purpose, warns about itself
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="Constant folding")
        torch.onnx.export(net, (dummy,), str(path), input_names=["frames"], output_names=["force"], opset_version=17, dynamo=False)
    meta = {"history": model.history, "n_channels": n_channels, "front_end": model.front_end}  # front_end may be null
    Path(f"{path}.json").write_text(json.dumps(meta, indent=1))
    return path


def offline_predict(
    emg: np.ndarray,
    n_frames: int,
    sos: np.ndarray,
    model: Predictor,
    input_norm: str = "none",
    n_base: int = WARMUP_FRAMES,
    montage: str = "monopolar",
) -> np.ndarray:
    emg, _ = finite_samples(emg)
    pred = model.predict(normalize_input(offline_frames(apply_montage(emg, montage), n_frames, sos), input_norm))
    pred[:n_base] = np.nan  # silent in the warm-up: frames before n_base are never trained on or scored
    return pred


def finite_samples(emg: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN/inf samples with 0 (one bad sample would otherwise poison every later output). Return the count."""
    bad = ~np.isfinite(emg)
    n = int(bad.sum())
    return (np.where(bad, 0, emg) if n else emg), n


DROP_LOG = 3.0  # log-RMS units below the channel's usual level within its grid (RMS 20x lower): the channel has dropped out
# log-RMS units above it (RMS 20x higher): a loose contact or an artefact. Chosen on the development people's own models
# as the lowest of 3, 4 and 5 that kept the clean error within 0.02 %MVC and caught a noisy channel in every recording.
RISE_LOG = 3.0


def grid_reference(frame: np.ndarray) -> np.ndarray:
    """Per channel, the median log-RMS of its electrode grid in this frame. `frame` holds the 4 grids in order."""
    per_grid = len(frame) // len(GRIDS)
    return np.repeat(np.median(frame.reshape(len(GRIDS), per_grid), axis=1), per_grid)


class StreamingDecoder:
    """The streaming decoder: raw EMG chunks in, one prediction per 100 Hz force timestamp out.

    Input: (samples, 256) at 2048 Hz in mV, channels in `hyser.EMG_CHANNELS` order (grids ED, EP, FD, FP, then row,
    then column), as `Hyser.load` returns them. Output: thumb..little in %MVC, flexion negative, extension positive.

    Input monitor: non-finite samples are replaced with 0 and counted (`nonfinite`). Each channel's log-RMS is compared
    with the median of its electrode grid in the same frame. After the warm-up, a channel more than DROP_LOG below its
    usual offset from that median (the running mean of the offset) is treated as dropped out, and one more than
    RISE_LOG above it as noisy. Either is fed median + usual offset instead, before any input normalisation
    (`replaced`: the current channel mask, `replaced_frames`: channel-frames replaced since the last reset). Comparing
    within the grid and frame keeps a long press, which raises a whole region together, from moving the reference.
    A whole grid going silent at once moves its median too and is not detected.
    """

    def __init__(
        self,
        n_channels: int,
        sos: np.ndarray,
        model: Predictor,
        input_norm: str = "none",
        n_base: int = WARMUP_FRAMES,
        monitor: bool = True,
        montage: str = "monopolar",
    ):
        """`model` is re-run on the last `history` frames for every frame (TCN or ONNX TCN). `n_channels` is the raw EMG
        width (256), `montage` the spatial montage the model was trained on."""
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"unknown input normalisation {input_norm!r}, use one of {INPUT_NORMS}")
        self.front = StreamingFrontEnd(n_channels, sos, montage)
        self.montage = montage
        if getattr(getattr(model, "cfg", None), "arch", "tcn") != "tcn":  # by config, so ONNX decoding needs no torch
            raise ValueError("the streaming decoder runs the TCN. The LSTM baseline is scored offline")
        self.model = model
        self.history = model.history
        if monitor and n_channels != len(GRIDS) * GRID_CHANNELS:
            raise ValueError(
                f"the input monitor compares channels within the {len(GRIDS)} grids of {GRID_CHANNELS}, got {n_channels} channels"
            )
        self.input_norm, self.n_base = input_norm, n_base
        self.monitor = monitor
        self.reset()

    def reset(self) -> None:
        """Recording boundary: drop filter state, frame history and normalisation statistics."""
        self.front.reset()
        self.buf: deque[np.ndarray] = deque(maxlen=self.history)
        self.sum: np.ndarray | None = None
        self.count = 0
        self.nonfinite = 0  # NaN/inf input samples replaced with 0 since the last reset
        self.replaced = np.zeros(self.front.n_channels, bool)
        self.replaced_frames = 0
        self.offset_sum = np.zeros(self.front.n_channels)  # running sum of each channel's offset from its grid median
        self.offset_n = 0
        self.out_offset_sum = np.zeros(self.front.n_out)  # the same for the montage channels the model reads

    def _ingest(self, frame: np.ndarray, out: np.ndarray) -> None:
        """`frame`: the raw electrodes, which the monitor checks. `out`: the montage channels the model reads (the same
        array without a montage). A montage channel built from a flagged electrode is filled in like a raw one."""
        mono = self.montage == "monopolar"
        if self.monitor:
            ref = grid_reference(frame)
            offset = frame - ref
            warm = self.offset_n >= self.n_base
            if warm:
                usual = self.offset_sum / self.offset_n  # running mean of the (accepted) offsets so far
                self.replaced = (offset < usual - DROP_LOG) | (offset > usual + RISE_LOG)
                if self.replaced.any():
                    offset = np.where(self.replaced, usual, offset)
                    frame = (ref + offset).astype(np.float32)
                    self.replaced_frames += int(self.replaced.sum())
            self.offset_sum += offset
            if not mono:
                ref_o = grid_reference(out)
                offset_o = out - ref_o
                if warm and self.replaced.any():  # CAR: a flagged electrode's own channel (the others drop it from the mean)
                    offset_o = np.where(self.replaced, self.out_offset_sum / self.offset_n, offset_o)
                    out = (ref_o + offset_o).astype(np.float32)
                self.out_offset_sum += offset_o
            self.offset_n += 1
        frame = frame if mono else out
        total = frame.astype(np.float64) if self.sum is None else self.sum + frame
        self.sum = total
        self.count += 1
        z = (frame - total / self.count).astype(np.float32) if self.input_norm == "running_mean" else frame
        self.buf.append(z)

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Feed raw EMG (samples, channels). Return (frame index, five forces) for every frame the chunk completes."""
        out: list[tuple[int, np.ndarray]] = []
        if chunk.ndim != 2 or chunk.shape[1] != self.front.n_channels:
            raise ValueError(f"expected EMG of shape (samples, {self.front.n_channels}), got {chunk.shape}")
        if len(chunk) == 0:
            return out
        chunk, n_bad = finite_samples(chunk)
        self.nonfinite += n_bad
        # With the monitor feeding the montage, each frame's samples use the flags of the frame before, so the chunk is
        # cut at frame boundaries and the output does not depend on the chunk size.
        per_frame = self.monitor and self.montage != "monopolar"
        pos = 0
        while pos < len(chunk):
            take = min(len(chunk) - pos, self.front.frame_end - self.front.n_seen) if per_frame else len(chunk)
            for j, raw, frame in self.front.push_pair(chunk[pos : pos + take], self.replaced if per_frame else None):
                self._ingest(raw, frame)
                if len(self.buf) < self.history or j < self.n_base:  # window not full, or still in the warm-up
                    out.append((j, np.full(5, np.nan)))
                    continue
                x = np.stack(self.buf)  # the whole window is re-run each frame on purpose: simple, about 0.35 ms of a 10 ms frame
                out.append((j, self.model.predict(x)[-1]))
            pos += take
        return out


def decoder_for(model: TCNModel, backend: Predictor | None = None) -> StreamingDecoder:
    """A StreamingDecoder with the front end recorded in `model`'s state file (`TCNModel.save_state(front_end=...)`).

    `backend` runs the network instead of `model` (an OnnxModel of the same weights). The front end still
    comes from `model`.
    """
    if model.front_end is None:
        raise ValueError("this checkpoint does not record its front end. Build StreamingDecoder(...) with the training settings")
    return _decoder(model.front_end, model.net.inp.in_channels, backend or model)


def _decoder(fe: dict[str, Any], n_channels: int, backend: Predictor) -> StreamingDecoder:
    kw: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in fe["filter"].items()}
    filt = FilterConfig(**kw)
    n_base = int(round(float(fe["warmup_s"]) * FS_FORCE))
    decoder = StreamingDecoder(len(GRIDS) * GRID_CHANNELS, design_sos(filt), backend, fe["input_norm"], n_base, montage=filt.montage)
    if decoder.front.n_out != n_channels:
        raise ValueError(f"the {filt.montage!r} montage gives {decoder.front.n_out} channels, the network takes {n_channels}")
    return decoder


def load_decoder(path: str | os.PathLike[str]) -> StreamingDecoder:
    """A decoder from its state file: raw EMG chunks in, five finger forces out every 10 ms."""
    from emg2force.tcn import TCNModel

    return decoder_for(TCNModel.load_state(Path(path)))


def load_onnx_decoder(path: str | os.PathLike[str], threads: int = 1) -> StreamingDecoder:
    """The same decoder from an exported `.onnx` and its `.onnx.json` front end, without torch."""
    meta = json.loads(Path(f"{path}.json").read_text())
    if meta["front_end"] is None:
        raise ValueError(f"{path}: exported from a checkpoint without a recorded front end, so no decoder can be built")
    return _decoder(meta["front_end"], meta["n_channels"], OnnxModel(Path(path), meta["history"], threads))
