"""Model interface, the zero and constant baselines, and the ridge on lag features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Sequence:
    """One recording's model inputs and labels. Never concatenated with another recording."""

    key: str
    subject: int
    session: int
    x: np.ndarray  # (T, C) log-RMS frames
    y: np.ndarray  # (T, 5) %MVC target
    scored: np.ndarray  # (T,) bool: frames used for training loss / evaluation


def recording_probabilities(train: list[Sequence], focus_subject: int | None, share: float) -> np.ndarray | None:
    """Per-recording sampling probability: `share` in total for the focus subject. None means uniform."""
    if not share:
        return None
    mine = np.array([s.subject == focus_subject for s in train])
    if not mine.any():
        raise ValueError(f"personal_share={share} needs training recordings of focus_subject (got {focus_subject})")
    if mine.all():  # only the focus subject: uniform is the same thing
        return None
    return np.where(mine, share / mine.sum(), (1 - share) / (~mine).sum())


class ChannelScaler:
    """Per-channel standardisation fitted on training sequences only."""

    def fit(self, seqs: list[Sequence]) -> ChannelScaler:
        x = np.concatenate([s.x for s in seqs])
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-6
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class Predictor(Protocol):
    """Anything that maps (T, C) frames to (T, 5) %MVC with a fixed history: a fitted Model or an ONNX backend."""

    history: int

    def predict(self, x: np.ndarray) -> np.ndarray: ...


class Model(ABC):
    """Interface: fit on training Sequences. Predict (T, C) frames -> (T, 5) %MVC, NaN before a full history."""

    history: int

    @abstractmethod
    def fit(self, train: list[Sequence], focus_subject: int | None = None) -> Model:
        """`focus_subject`: the person being evaluated. Models that weight their training data use it, others ignore it."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class ConstantModel(Model):
    """Training-set median force per finger. The floor every model must beat."""

    def __init__(self, history: int):
        self.history = history

    def fit(self, train: list[Sequence], focus_subject: int | None = None) -> ConstantModel:
        self.median = np.median(np.concatenate([s.y[s.scored] for s in train]), axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.tile(self.median, (len(x), 1))
        out[: self.history - 1] = np.nan
        return out


class ZeroModel(Model):
    """Predicts zero force (rest). On the new day it scores lower than the training median, so it is the stricter floor."""

    def __init__(self, history: int):
        self.history = history

    def fit(self, train: list[Sequence], focus_subject: int | None = None) -> ZeroModel:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), 5))
        out[: self.history - 1] = np.nan
        return out


def lag_blocks(history: int) -> list[tuple[int, int]]:
    """(start_lag, end_lag) exclusive, lag 0 = current frame. Covers exactly `history` frames."""
    sizes, total, nxt = [1], 1, 1
    while total < history:
        sizes.append(min(nxt, history - total))
        total += sizes[-1]
        nxt *= 2
    out, lag = [], 0
    for size in sizes:
        out.append((lag, lag + size))
        lag += size
    return out


def lag_features(x: np.ndarray, history: int) -> np.ndarray:
    """(T, C * n_blocks) block means of past frames. Rows j < history-1 are NaN."""
    T, C = x.shape
    cs = np.concatenate([np.zeros((1, C)), np.cumsum(x, axis=0, dtype=np.float64)])
    feats = []
    j = np.arange(T)
    for a, b in lag_blocks(history):
        hi = j - a + 1  # frames j-b+1 .. j-a inclusive -> cumsum indices [j-b+1, j-a+1)
        lo = j - b + 1
        ok = lo >= 0
        f = np.full((T, C), np.nan)
        f[ok] = (cs[hi[ok]] - cs[lo[ok]]) / (b - a)
        feats.append(f)
    return np.concatenate(feats, axis=1)


class StatsCache(dict):
    """Per-recording ridge statistics (about 18 MiB each at 256 channels and 32 frames), kept until `max_gb` is used.
    Later recordings are computed again when needed, so the results are the same at any budget."""

    def __init__(self, max_gb: float = 2.0):
        super().__init__()
        self.max_bytes = max_gb * 2**30
        self.used = 0

    def __setitem__(self, key: object, value: tuple) -> None:
        size = sum(getattr(v, "nbytes", 0) for v in value)
        if self.used + size <= self.max_bytes:
            self.used += size
            super().__setitem__(key, value)


class RidgeModel(Model):
    """Closed-form ridge on standardised lag features.

    `stats_cache` keeps per-recording sufficient statistics so LOSO folds only re-sum them (one cache per set of input
    features). `personal_share` weights the focus subject's recordings like the TCN's batch sampling.
    """

    def __init__(self, history: int, alpha: float = 1.0, stats_cache: dict | None = None, personal_share: float = 0.0):
        self.history = history
        self.alpha = alpha
        self.stats_cache = stats_cache
        self.personal_share = personal_share

    def _stats(self, s: Sequence) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ck = (s.key, self.history)
        if self.stats_cache is not None and ck in self.stats_cache:
            return self.stats_cache[ck]
        f = lag_features(s.x, self.history)
        m = s.scored & ~np.isnan(f).any(1)
        X, Y = f[m], s.y[m].astype(np.float64)
        st = (len(X), X.sum(0), Y.sum(0), X.T @ X, X.T @ Y)
        if self.stats_cache is not None:
            self.stats_cache[ck] = st
        return st

    def fit(self, train: list[Sequence], focus_subject: int | None = None) -> RidgeModel:
        n = 0.0
        sx: np.ndarray | float = 0.0  # running (weighted) sums of the per-recording statistics
        sy: np.ndarray | float = 0.0
        xtx: np.ndarray | float = 0.0
        xty: np.ndarray | float = 0.0
        p_rec = recording_probabilities(train, focus_subject, self.personal_share)
        for i, s in enumerate(train):
            a, b, c, d, e = self._stats(s)
            w = 1.0 if p_rec is None or not a else p_rec[i] / a  # a recording's frames share its sampling probability
            n, sx, sy, xtx, xty = n + w * a, sx + w * b, sy + w * c, xtx + w * d, xty + w * e
        if not n:
            raise ValueError("ridge fit needs at least one scored frame with a full history")
        self.x_mean, self.y_mean = sx / n, sy / n
        cxx = xtx / n - np.outer(self.x_mean, self.x_mean)
        cxy = xty / n - np.outer(self.x_mean, self.y_mean)
        self.x_std = np.sqrt(np.clip(np.diag(cxx), 1e-12, None))
        cxx = cxx / np.outer(self.x_std, self.x_std)
        cxy = cxy / self.x_std[:, None]
        # alpha acts on the correlation matrix, so it is independent of data scale/size
        w = np.linalg.solve(cxx + self.alpha * np.eye(len(cxx)), cxy)
        self.coef = w / self.x_std[:, None]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (lag_features(x, self.history) - self.x_mean) @ self.coef + self.y_mean
