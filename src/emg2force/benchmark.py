"""ONNX export, PyTorch/ONNX agreement and batch-one CPU latency of the streaming decoder.

  python -m emg2force.benchmark --model assets/decoder.state.pt --out results/latency \
      [--emg recorded --root ../hyser_raw] [--threads 1] [--seconds 60]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy
import torch
from scipy import signal

from emg2force.hyser import EMG_CHANNELS, FS_EMG, Hyser, emg_count_at
from emg2force.pipeline import OnnxModel, StreamingDecoder, decoder_for, export_onnx
from emg2force.provenance import REPO, git_state
from emg2force.tcn import TCNModel

DEADLINE_MS = 10.0
UPDATE_S = 0.01  # one decoder update per 100 Hz force sample
MODEL_WARMUP_RUNS = 200
SYNTH_EMG_SCALE = 5e-2  # synthetic EMG amplitude in header units (mV in practice, so ~50 uV)
PARITY_FRAMES = 2000  # input frames on which PyTorch and ONNX decoders are compared


def filter_timing(sos: np.ndarray, fs: float = FS_EMG) -> dict:
    """Median filter group delay over 20-450 Hz, away from the notches."""
    f = np.arange(20.0, 450.0, 0.5)
    f = f[np.abs((f + 25) % 50 - 25) > 2]
    gd = sum(signal.group_delay((sec[:3], sec[3:]), w=f, fs=fs)[1] for sec in sos) / fs * 1e3
    return {"group_delay_ms_median": round(float(np.median(gd)), 2)}


def cpu_name() -> str:
    """CPU model: sysctl on macOS, /proc/cpuinfo on Linux, else whatever platform reports."""
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def percentiles(ms: np.ndarray) -> dict:
    return {
        "n": int(len(ms)),
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def recorded_emg(root: Path, subject: int = 2, session: int = 2) -> np.ndarray:
    """One subject-session's five random-task trials, end to end, as a continuous stream."""
    ds = Hyser(root)
    return np.concatenate([ds.load(subject, session, t).emg for t in range(1, 6)]).astype(np.float32)


def synthetic_emg(n_channels: int, n_updates: int) -> np.ndarray:
    n = int(emg_count_at(n_updates)) + 1
    return (SYNTH_EMG_SCALE * np.random.default_rng(1).standard_normal((n, n_channels))).astype(np.float32)


def _wait(due: float, pacing: str) -> None:
    if pacing == "sleep":
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


def _time_decoder(decoder: StreamingDecoder, emg: np.ndarray, n_updates: int, pacing: str) -> tuple[np.ndarray, int]:
    """Per-update decoder.push time (ms) and late-update count, excluding the warm-up and buffer fill (no prediction yet)."""
    decoder.reset()
    ends = emg_count_at(np.arange(n_updates))
    starts = np.concatenate([[0], ends[:-1]])
    ms, late, history = [], 0, max(decoder.history, decoder.n_base)
    t_start = time.perf_counter()
    for j in range(n_updates):
        due = t_start + (j + 1) * UPDATE_S  # when frame j's last EMG sample would arrive
        _wait(due, pacing)
        t0 = time.perf_counter()
        decoder.push(emg[starts[j] : ends[j]])
        t1 = time.perf_counter()
        if j >= history:
            ms.append((t1 - t0) * 1e3)
            late += int(pacing != "none" and t1 > due + DEADLINE_MS / 1e3)
    return np.array(ms), late


def _time_model_only(onnx: OnnxModel, n_channels: int, n_updates: int) -> np.ndarray:
    """Network time per update: one window through the TCN."""
    x = np.random.default_rng(2).standard_normal((onnx.history, n_channels)).astype(np.float32)
    for _ in range(MODEL_WARMUP_RUNS):
        onnx.predict(x)
    ms = []
    for _ in range(n_updates):
        t0 = time.perf_counter()
        onnx.predict(x)
        ms.append((time.perf_counter() - t0) * 1e3)
    return np.array(ms)


def decoder_parity(model: TCNModel, onnx: OnnxModel, emg: np.ndarray) -> dict:
    """PyTorch vs ONNX through the same StreamingDecoder (the checkpoint's front end) on the first PARITY_FRAMES input frames."""
    outs = []
    for backend in (model, onnx):
        dec = decoder_for(model, backend)
        outs.append(np.array([p for _, p in dec.push(emg[: int(emg_count_at(PARITY_FRAMES - 1))])]))
    same_nan = bool(np.array_equal(np.isnan(outs[0]), np.isnan(outs[1])))
    return {"frames": len(outs[0]), "same_undefined_frames": same_nan, "max_abs_diff_pct_mvc": float(np.nanmax(np.abs(outs[0] - outs[1])))}


def bench(model: TCNModel, onnx: OnnxModel, emg: np.ndarray, n_channels: int, n_updates: int) -> dict:
    """Model-only, sustained and paced timings of a decoder file (its own front end) with the ONNX
    backend, and the paced timing with the PyTorch backend."""
    res = {"model_only": percentiles(_time_model_only(onnx, n_channels, n_updates))}
    torch.set_num_threads(onnx.session.get_session_options().intra_op_num_threads)
    for name, backend, pacing in (
        ("pipeline_sustained", onnx, "none"),
        ("pipeline_paced_sleep", onnx, "sleep"),
        ("pipeline_paced_sleep_pytorch", model, "sleep"),
    ):
        decoder = decoder_for(model, backend)
        _time_decoder(decoder, emg, min(n_updates, 500), "none")  # warm-up pass, discarded
        ms, late = _time_decoder(decoder, emg, n_updates, pacing)
        res[name] = {**percentiles(ms), **({"late_updates": int(late)} if pacing != "none" else {})}
    return res


def describe(model: TCNModel, threads: int, n_channels: int, emg_source: str) -> dict:
    decoder = decoder_for(model)
    return {
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        **git_state(),
        "intra_op_threads": threads,
        "inter_op_threads": 1,
        "batch": 1,
        "channels": n_channels,
        "update_interval_ms": 10.0,
        "input": emg_source,
        "filter_timing": filter_timing(decoder.front.sos),
        "frame_lag_ms": 5.0,  # mean age of the samples in a 10 ms frame
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=REPO / "assets" / "decoder.state.pt", help="a .state.pt checkpoint (TCNModel.save_state)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emg", choices=("synthetic", "recorded"), default="synthetic")
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--tag", default="", help="suffix of the output file, e.g. _rep1")
    args = ap.parse_args(argv)
    if not args.model.is_file():
        ap.error(f"no checkpoint at {args.model}, pass --model <file>.state.pt")
    model = TCNModel.load_state(args.model)
    n_ch = model.net.inp.in_channels  # what the network reads
    n_raw = len(EMG_CHANNELS)  # what the decoder takes in, and names the output files
    n_updates = round(args.seconds / UPDATE_S)
    emg = recorded_emg(args.root) if args.emg == "recorded" else synthetic_emg(n_raw, n_updates)
    n_updates = min(n_updates, int(np.searchsorted(emg_count_at(np.arange(n_updates)), len(emg))) - 1)
    args.out.mkdir(parents=True, exist_ok=True)
    onnx = OnnxModel(export_onnx(model, args.out / f"tcn_{n_raw}ch.onnx"), model.history, args.threads)
    source = "recorded EMG, subject 2 day 2 trials 1-5 back to back" if args.emg == "recorded" else "synthetic Gaussian EMG"
    res = {
        "setup": {
            **describe(model, args.threads, n_ch, source),
            "weights": str(args.model),
        },
        "pytorch_vs_onnx_decoder": decoder_parity(model, onnx, emg),
        **bench(model, onnx, emg, n_ch, n_updates),
    }
    out = args.out / f"latency_{n_raw}ch_{args.emg}{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
