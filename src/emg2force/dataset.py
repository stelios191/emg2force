"""Per-recording Sequences (causal frames + %MVC target) with an on-disk frame cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import scipy

from emg2force.causal import WARMUP_FRAMES, FilterConfig, apply_montage, design_sos, normalize_input, offline_frames
from emg2force.hyser import (
    FS_FORCE,
    Hyser,
    Recording,
    emg_count_at,
    make_target,
    parse_key,
)
from emg2force.models import Sequence
from emg2force.provenance import REPO

CACHE = REPO / "cache"  # feature cache
DEFAULT_WARMUP_S = WARMUP_FRAMES / FS_FORCE  # 0.7 s


def warmup_frames(cfg: dict) -> int:
    """Frames at the start of each recording a run config leaves unscored."""
    return int(round(cfg.get("warmup_s", DEFAULT_WARMUP_S) * FS_FORCE))


FRAME_VERSION = 1  # part of the cache folder name: bump it when the loader, montage, filter or framing code changes


def code_tag() -> str:
    """The frame version and the numerical library versions, as a short hash."""
    return hashlib.sha1(repr((FRAME_VERSION, np.__version__, scipy.__version__)).encode()).hexdigest()[:10]


def filter_tag(cfg: FilterConfig) -> str:
    return hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:10]


def data_fingerprint(ds: Hyser, key: str) -> dict:
    """Size and mtime of each source file of the frames. A change forces a recompute."""
    s, d, t = parse_key(key)
    out = {}
    for kind, exts in (("raw", (".hea", ".dat")), ("force", (".hea",))):
        stem = ds._random_stem(s, d, t, kind)
        for ext in exts:
            st = stem.with_suffix(ext).stat()
            out[stem.with_suffix(ext).name] = [st.st_size, st.st_mtime_ns]
    return out


def frames_for(ds: Hyser, key: str, cfg: FilterConfig) -> np.ndarray:
    """Causal log-RMS frames, cached by filter config and frame version. Recomputed when the source files change."""
    path = CACHE / f"frames_{filter_tag(cfg)}_{code_tag()}" / f"{key}.npy"
    meta = path.with_suffix(".json")
    fp = data_fingerprint(ds, key)
    if path.exists() and meta.exists() and json.loads(meta.read_text()) == fp:
        return np.load(path)
    rec = ds.load(*parse_key(key))
    x = offline_frames(apply_montage(rec.emg, cfg.montage), len(rec.force_v), design_sos(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")  # write then rename: a concurrent run never reads a half-written file
    np.save(tmp, x)
    os.replace(tmp, path)
    tmp_meta = meta.with_suffix(f".{os.getpid()}.tmp.json")
    tmp_meta.write_text(json.dumps(fp))
    os.replace(tmp_meta, meta)
    return x


def build_sequences(
    ds: Hyser,
    keys: Iterable[str],
    mvc_convention: str,
    label_filter: str,
    filter_cfg: FilterConfig = FilterConfig(),
    warmup_s: float = DEFAULT_WARMUP_S,
    input_norm: str = "none",
) -> dict[str, Sequence]:
    """One Sequence per key: input-normalised frames, %MVC target, and a mask scoring frames after `warmup_s`."""
    out = {}
    for key in keys:
        s, d, t = parse_key(key)
        x = frames_for(ds, key, filter_cfg)
        n_warm = int(round(warmup_s * FS_FORCE))
        x = normalize_input(x, input_norm)
        rec = ds.load_force(s, d, t)
        y = ds.target(rec, mvc_convention, label_filter).values
        scored = np.arange(len(y)) >= n_warm
        out[key] = Sequence(key, s, d, np.ascontiguousarray(x), y.astype(np.float32), scored)
    return out


def cut_sequence(
    rec: Recording,
    key: str,
    seconds: float,
    mvc: np.ndarray,
    label_filter: str,
    filter_cfg: FilterConfig = FilterConfig(),
    warmup_s: float = DEFAULT_WARMUP_S,
    input_norm: str = "none",
) -> Sequence:
    """The first `seconds` of a recording as a Sequence, built from the truncated EMG and force only.

    Nothing after the cut reaches frames, normalisation or labels. The zero-phase label filter therefore runs on the
    truncated force, so the last ~0.1 s of labels before the cut differ slightly from the full recording's.
    """
    n = min(int(round(seconds * FS_FORCE)), len(rec.force_v))
    emg = rec.emg[: int(emg_count_at(n - 1))]
    x = offline_frames(apply_montage(emg, filter_cfg.montage), n, design_sos(filter_cfg))
    n_warm = int(round(warmup_s * FS_FORCE))
    x = normalize_input(x, input_norm)
    y = make_target(rec.force_v[:n], mvc, label_filter, rec.fs_force)
    return Sequence(key, rec.subject, rec.session, np.ascontiguousarray(x), y.astype(np.float32), np.arange(n) >= n_warm)


def prefix_sequence(
    ds: Hyser,
    key: str,
    seconds: float,
    mvc_convention: str,
    label_filter: str,
    filter_cfg: FilterConfig = FilterConfig(),
    warmup_s: float = DEFAULT_WARMUP_S,
    input_norm: str = "none",
) -> Sequence:
    """The first `seconds` of a random-task recording (see `cut_sequence`)."""
    s, d, t = parse_key(key)
    mvc, _ = ds.mvc_for(s, d, mvc_convention)
    return cut_sequence(ds.load(s, d, t), key, seconds, mvc, label_filter, filter_cfg, warmup_s, input_norm)
