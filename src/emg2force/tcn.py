"""Causal TCN over log-RMS frames, and the causal LSTM baseline trained by the same code (`arch="lstm"`). The TCN's
receptive field equals the history H that the ridge also sees."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from emg2force.hyser import GRID_SIDE, GRIDS
from emg2force.models import ChannelScaler, Model, Sequence, recording_probabilities

STATE_FORMAT = "emg2force/tcn-state/1"  # explicit config + state_dict, loadable with weights_only=True


def train_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def grid_neighbours() -> dict[str, npt.NDArray[np.intp]]:
    """Per channel of the 4 x 8 x 8 grids, for shift augmentation: the nearest electrode in each direction (itself at
    the edge) and the channel's grid."""
    last = GRID_SIDE - 1
    out: dict[str, list[int]] = {k: [] for k in ("up", "down", "left", "right", "grid")}
    for g in range(len(GRIDS)):
        for r in range(GRID_SIDE):
            for c in range(GRID_SIDE):
                out["up"].append((g * GRID_SIDE + max(r - 1, 0)) * GRID_SIDE + c)
                out["down"].append((g * GRID_SIDE + min(r + 1, last)) * GRID_SIDE + c)
                out["left"].append((g * GRID_SIDE + r) * GRID_SIDE + max(c - 1, 0))
                out["right"].append((g * GRID_SIDE + r) * GRID_SIDE + min(c + 1, last))
                out["grid"].append(g)
    return {k: np.array(v, dtype=np.intp) for k, v in out.items()}


@dataclass
class TCNConfig:
    """Architecture, training schedule and training-only augmentation options. The defaults are the final recipe.

    `kernel` and `dilations` shape the TCN only, `layers` the LSTM only.
    """

    hidden: int = 64
    kernel: int = 2
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.3
    lr: float = 2e-3
    weight_decay: float = 0.01
    steps: int = 2000
    batch: int = 32
    crop: int = 256  # frames per training crop (scored positions only enter the loss)
    target_scale: float = 100.0  # %MVC -> fraction of MVC inside the network
    seed: int = 0
    device: str = "auto"  # training device. "auto" = CUDA, else MPS, else CPU. Inference is always CPU.
    personal_share: float = 0.0  # share of crops from the focus subject (0 = uniform)
    aug_shift: float = 0.0  # max grid shift in electrodes, per crop and grid (at most 1)
    arch: str = "tcn"  # "tcn", or "lstm": the recurrent baseline (same width, same training)
    layers: int = 2  # LSTM layers (arch "lstm" only)

    def __post_init__(self) -> None:
        checks = {
            "hidden, steps, batch, crop and layers must be positive": min(self.hidden, self.steps, self.batch, self.crop, self.layers) > 0,
            "dropout must be in [0, 1)": 0 <= self.dropout < 1,
            "lr must be positive and weight_decay non-negative": self.lr > 0 and self.weight_decay >= 0,
            "personal_share must be in [0, 1)": 0 <= self.personal_share < 1,
            "aug_shift must be in [0, 1]": 0 <= self.aug_shift <= 1,
            "arch must be tcn or lstm": self.arch in ("tcn", "lstm"),
        }
        if failed := [rule for rule, ok in checks.items() if not ok]:
            raise ValueError(f"invalid TCN settings: {', '.join(failed)}")

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.kernel - 1) * d for d in self.dilations)


class CausalConv(nn.Module):
    """Dilated Conv1d padded on the left only, so output t depends on inputs <= t."""

    def __init__(self, cin: int, cout: int, k: int, d: int):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class Block(nn.Module):
    """Residual block: causal dilated conv -> GELU -> dropout -> 1x1 mix, added to the input."""

    def __init__(self, h: int, k: int, d: int, p: float):
        super().__init__()
        self.conv = CausalConv(h, h, k, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.mix = nn.Conv1d(h, h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(self.drop(self.act(self.conv(x))))


class TCNNet(nn.Module):
    """Standardise (stored buffers) -> 1x1 projection -> residual blocks -> 1x1 head to 5 fingers."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, n_channels: int, cfg: TCNConfig, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1))
        self.inp = nn.Conv1d(n_channels, cfg.hidden, 1)
        self.blocks = nn.Sequential(*[Block(cfg.hidden, cfg.kernel, d, cfg.dropout) for d in cfg.dilations])
        self.head = nn.Conv1d(cfg.hidden, 5, 1)
        self.scale = cfg.target_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T) raw log-RMS -> (B, 5, T) %MVC
        # NNPACK's 1x1 Conv1d is ~5x slower on Apple M1 CPUs. Disabling it leaves outputs unchanged.
        with torch.backends.nnpack.flags(enabled=False):
            x = (x - self.mean) / self.std
            return self.head(self.blocks(self.inp(x))) * self.scale


class LSTMNet(nn.Module):
    """Baseline with the TCN's interface: standardise -> 1x1 projection -> causal (unidirectional) LSTM -> head."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, n_channels: int, cfg: TCNConfig, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1))
        self.inp = nn.Conv1d(n_channels, cfg.hidden, 1)
        self.lstm = nn.LSTM(cfg.hidden, cfg.hidden, cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.head = nn.Linear(cfg.hidden, 5)
        self.scale = cfg.target_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T) raw log-RMS -> (B, 5, T) %MVC. State starts at zero
        h, _ = self.lstm(self.inp((x - self.mean) / self.std).transpose(1, 2))
        return self.head(h).transpose(1, 2) * self.scale


NETS: dict[str, type[TCNNet] | type[LSTMNet]] = {"tcn": TCNNet, "lstm": LSTMNet}


def per_frame_l1(err: torch.Tensor) -> torch.Tensor:
    """(B, 5, L) error -> (B, L) L1, mean over fingers."""
    return err.abs().mean(1)


class TCNModel(Model):
    """Trains a TCNNet, or the LSTMNet baseline when `cfg.arch == "lstm"`, on random crops. Predicts whole recordings
    causally on the CPU."""

    def __init__(self, history: int, cfg: TCNConfig | None = None):
        cfg = cfg if cfg is not None else TCNConfig()
        if cfg.arch not in NETS:
            raise ValueError(f"unknown arch {cfg.arch!r}, use one of {sorted(NETS)}")
        if cfg.arch == "tcn" and cfg.receptive_field != history:
            raise ValueError(f"TCN receptive field {cfg.receptive_field} != history {history}")
        self.history = history
        self.cfg = cfg
        self.neighbours = grid_neighbours()  # electrode layout for shift augmentation
        self.provenance: dict = {}  # where saved weights came from (load_state)
        self.train_device: str | None = None  # set by fit. Recorded in the run summary and the state file
        self.front_end: dict | None = None  # input_norm, warmup_s, filter the weights were trained with (save_state)
        self.train_loss: list[float] = []  # training loss per step (set by fit)

    def fit(self, train: list[Sequence], focus_subject: int | None = None) -> TCNModel:
        """Train on random crops. The loss covers scored frames with a full history.

        With `personal_share`, that share of crops comes from `focus_subject`.
        """
        cfg = self.cfg
        if short := [s.key for s in train if len(s.x) < cfg.crop]:
            raise ValueError(f"recordings shorter than the {cfg.crop}-frame training crop: {short}")
        if cfg.aug_shift and train[0].x.shape[1] != len(self.neighbours["grid"]):  # fail before training
            raise ValueError("aug_shift shifts the 4 x 8 x 8 electrode grids, so it needs all 256 channels")
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        dev = train_device(cfg.device)
        self.train_device = str(dev)
        scaler = ChannelScaler().fit(train)
        self.net = NETS[cfg.arch](train[0].x.shape[1], cfg, scaler.mean, scaler.std).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
        p_rec = recording_probabilities(train, focus_subject, cfg.personal_share)
        self.net.train()
        losses = []
        for _ in range(cfg.steps):
            xt, yt, wt = (torch.from_numpy(np.ascontiguousarray(v)).to(dev) for v in self._batch(train, rng, p_rec))
            err = per_frame_l1(self.net(xt) - yt)
            loss = (err * wt).sum() / wt.sum().clamp_min(1.0) / cfg.target_scale
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.detach())  # kept on the device: no per-step synchronisation
        self.train_loss = torch.stack(losses).cpu().tolist()
        self.net = self.net.to("cpu").eval()
        return self

    def fine_tuned(self, cal: list[Sequence], steps: int, lr: float) -> TCNModel:
        """A copy with every layer fine-tuned (full-batch Adam on whole calibration sequences, CPU, deterministic).
        self is unchanged."""
        out = copy.deepcopy(self)
        net = out.net.to("cpu").eval()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        batch = [(torch.from_numpy(s.x.T.copy())[None], torch.from_numpy(s.y.T.copy())[None], self._loss_weights(s)) for s in cal]
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.stack([(per_frame_l1(net(x) - y) * w).sum() / w.sum().clamp_min(1.0) for x, y, w in batch]).mean()
            loss.backward()
            opt.step()
        return out

    def _loss_weights(self, s: Sequence) -> torch.Tensor:
        """(1, T) weights: scored frames with a full history window."""
        w = s.scored & (np.arange(len(s.x)) >= self.history - 1)
        return torch.from_numpy(w.astype(np.float32))[None]

    def _batch(
        self, train: list[Sequence], rng: np.random.Generator, p_rec: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One batch of random crops: inputs (B, C, L), targets (B, 5, L), loss weights (B, L)."""
        cfg, L = self.cfg, self.cfg.crop
        idx = rng.choice(len(train), size=cfg.batch, p=p_rec)
        starts = [int(rng.integers(0, len(train[i].x) - L + 1)) for i in idx]
        crops = list(zip(idx, starts, strict=True))
        xb = np.stack([train[i].x[a : a + L] for i, a in crops])  # (B, L, C)
        if cfg.aug_shift:
            xb = self._shift(xb, rng)
        yb = np.stack([train[i].y[a : a + L].astype(np.float32) for i, a in crops])
        full_history = (np.arange(L) >= self.history - 1).astype(np.float32)
        wb = np.stack([train[i].scored[a : a + L].astype(np.float32) for i, a in crops]) * full_history
        return xb.transpose(0, 2, 1), yb.transpose(0, 2, 1), wb

    def _shift(self, xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Simulate re-donning: shift each grid by a random offset of up to `aug_shift` electrodes, interpolating neighbours."""
        nb, B = self.neighbours, xb.shape[0]
        n_grids = int(nb["grid"].max()) + 1
        for a, b in (("up", "down"), ("left", "right")):
            d = rng.uniform(-self.cfg.aug_shift, self.cfg.aug_shift, (B, n_grids))[:, nb["grid"]]
            d = np.clip(d, -1, 1)[:, None, :].astype(np.float32)  # (B, 1, C) in electrodes
            toward = np.where(d >= 0, xb[:, :, nb[b]], xb[:, :, nb[a]])
            xb = (1 - np.abs(d)) * xb + np.abs(d) * toward
        return xb.astype(np.float32)

    def save_state(self, path: Path, provenance: dict, front_end: dict | None = None) -> None:
        """Save config + state_dict (no pickled classes) with the weights' `provenance` and training device.

        `front_end` (input_norm, warmup_s, filter) records the features the weights expect, so a decoder
        can be rebuilt from the file alone (`pipeline.decoder_for`).
        """
        cfg = {f.name: getattr(self.cfg, f.name) for f in fields(TCNConfig)}
        cfg = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
        torch.save(
            {"format": STATE_FORMAT, "history": self.history, "n_channels": self.net.inp.in_channels, "config": cfg,
             "state_dict": self.net.state_dict(), "provenance": {"train_device": self.train_device, **provenance},
             "front_end": front_end if front_end is not None else self.front_end},
            path,
        )  # fmt: skip

    @classmethod
    def load_state(cls, path: Path) -> TCNModel:
        """Rebuild a model saved by `save_state` for prediction or fine-tuning (loads with weights_only=True)."""
        d = torch.load(path, weights_only=True, map_location="cpu")
        if d.get("format") != STATE_FORMAT:
            raise ValueError(f"{path}: not a {STATE_FORMAT} file")
        conf: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in d["config"].items()}
        cfg = TCNConfig(**conf)
        model = cls(int(d["history"]), cfg)
        n = int(d["n_channels"])
        model.net = NETS[cfg.arch](n, cfg, np.zeros(n), np.ones(n))
        model.net.load_state_dict(d["state_dict"])
        model.net.eval()
        model.provenance = d["provenance"]
        model.front_end = d.get("front_end")
        return model

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self.net(torch.from_numpy(np.ascontiguousarray(x.T, dtype=np.float32))[None])[0].T.numpy().astype(np.float64)
        out[: self.history - 1] = np.nan
        return out
