"""Config-driven experiment runner.

    python -m experiments.run experiments/configs/final_cross_day_pop_256.yaml [--root ../hyser_raw]

Writes runs/<name>/: metrics.csv (finger x recording x model x seed), predictions.npz, train_loss.csv, config,
environment, splits, summary and models/ (when saved).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from emg2force.causal import FilterConfig
from emg2force.dataset import DEFAULT_WARMUP_S, build_sequences
from emg2force.hyser import Hyser, recording_key
from emg2force.models import ConstantModel, Model, RidgeModel, Sequence, StatsCache, ZeroModel
from emg2force.provenance import REPO, git_state
from emg2force.splits import DEV_SUBJECTS, EVAL_SUBJECTS, Split, check_split, make_splits
from emg2force.tcn import TCNConfig, TCNModel
from experiments.metrics import check_complete, recording_metrics, summary

SUMMARY_METRICS = ("mae", "rmse", "r2", "bias")
TCN_KEYS = frozenset(f.name for f in fields(TCNConfig)) - {"seed", "arch"}  # seed from `seeds`. Arch from the model kind
REQUIRED_KEYS = frozenset({"name", "subjects", "protocol", "mvc_convention", "label_filter", "history"})
OPTIONAL_KEYS = frozenset({"warmup_s", "input_norm", "filter", "threads", "save_models"})
RUN_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS | {"models", "seeds", "save_predictions"}  # experiments.run configs


def check_config(cfg: dict, required: frozenset[str] = REQUIRED_KEYS | {"models"}, allowed: frozenset[str] = RUN_KEYS) -> None:
    """A misspelt or missing top-level key is an error, never a silent default (model keys: `make_model`)."""
    if missing := sorted(required - set(cfg)):
        raise ValueError(f"config is missing keys {missing}")
    if unknown := sorted(set(cfg) - allowed):
        raise ValueError(f"unknown config keys {unknown}")


def front_end_of(cfg: dict) -> dict:
    """The feature settings a checkpoint trained under `cfg` expects (saved with it, see `pipeline.decoder_for`)."""
    filt = filter_config(cfg)
    return {
        "input_norm": cfg.get("input_norm", "none"),
        "warmup_s": float(cfg.get("warmup_s", DEFAULT_WARMUP_S)),
        "filter": {f.name: list(v) if isinstance(v := getattr(filt, f.name), tuple) else v for f in fields(FilterConfig)},
    }


def resolve_subjects(spec: str | list[int]) -> list[int]:
    """'dev', 'eval' or an explicit list of person numbers."""
    named = {"dev": DEV_SUBJECTS, "eval": EVAL_SUBJECTS}
    return list(named[spec]) if isinstance(spec, str) else [int(s) for s in spec]


def make_model(kind: str, params: dict, history: int, seed: int, ridge_cache: dict) -> Model:
    """Untrained model for one config entry (`constant`, `zero`, `ridge`, `tcn` or `lstm`). An unknown key is an error, never ignored."""
    accepted = {"constant": set(), "zero": set(), "ridge": {"alpha", "personal_share"}, "tcn": TCN_KEYS, "lstm": TCN_KEYS}
    if kind not in accepted:
        raise ValueError(f"unknown model {kind!r}")
    if unknown := sorted(set(params) - accepted[kind]):
        raise ValueError(f"{kind}: unknown config keys {unknown}")
    if kind == "constant":
        return ConstantModel(history)
    if kind == "zero":
        return ZeroModel(history)
    if kind == "ridge":
        return RidgeModel(history, float(params.get("alpha", 1.0)), ridge_cache, personal_share=float(params.get("personal_share", 0.0)))
    return make_tcn(params, history, seed, arch=kind)


def make_tcn(params: dict, history: int, seed: int, arch: str = "tcn") -> TCNModel:
    """Untrained network from a config's `tcn` (or `lstm`) mapping (YAML lists become tuples). An unknown key is an error."""
    if unknown := sorted(set(params) - TCN_KEYS):
        raise ValueError(f"{arch}: unknown config keys {unknown}")
    kw: dict[str, Any] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
    defaults = TCNConfig()
    for k, v in kw.items():
        d = getattr(defaults, k)
        if isinstance(d, int | float) and not isinstance(d, bool) and not isinstance(v, int | float):
            raise ValueError(f"{arch}.{k} must be a number, got {v!r} (YAML reads 1e-3 as text, write 1.0e-3)")
    return TCNModel(history, TCNConfig(**kw, seed=seed, arch=arch))


def evaluate_split(split: Split, seqs: dict[str, Sequence], model: Model) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit on split.train only and score every split.test recording."""
    check_split(split)
    model.fit([seqs[k] for k in split.train], focus_subject=split.focus_subject)
    rows, preds = [], {}
    for key in split.test:
        s = seqs[key]
        yhat = model.predict(s.x)
        preds[key] = yhat.astype(np.float32)
        rows += [
            {"split": split.name, "subject": s.subject, "session": s.session, "recording": key, **r}
            for r in recording_metrics(s.y, yhat, s.scored)
        ]
    return rows, preds


def env_info() -> dict:
    """Git commit, platform and package versions, saved with every run."""
    import onnxruntime
    import scipy

    return {
        **git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def model_variants(models_cfg: dict, seeds: list[int]) -> Iterator[tuple[str, str, dict, int]]:
    """(kind, label, params, seed) per model, grid point and (TCN and LSTM only) seed."""
    for kind, params in models_cfg.items():
        params = dict(params or {})
        grid = params.pop("grid", None) or [{}]
        for point in grid:
            label = kind + "".join(f"_{k}{v}" for k, v in point.items())
            for seed in seeds if kind in ("tcn", "lstm") else [0]:
                yield kind, label, {**params, **point}, seed


def recordings_on_disk(splits: list[Split], ds: Hyser) -> list[str]:
    """Recordings the splits need. A missing file is an error, never a smaller evaluation set."""
    needed = sorted({k for sp in splits for k in sp.train + sp.test})
    missing = sorted(set(needed) - {recording_key(*k) for k in ds.available()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} expected recordings missing under {ds.root}: {missing[:5]}...")
    return needed


def filter_config(cfg: dict) -> FilterConfig:
    """Optional `filter:` mapping as a FilterConfig (YAML lists become tuples)."""
    kw: dict[str, Any] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.get("filter", {}).items()}
    return FilterConfig(**kw)


def _features(cfg: dict) -> tuple:
    """(mvc_convention, label_filter, filter, warmup_s, input_norm) of a config."""
    filt = filter_config(cfg)
    return (
        cfg["mvc_convention"],
        cfg["label_filter"],
        filt,
        cfg.get("warmup_s", DEFAULT_WARMUP_S),
        cfg.get("input_norm", "none"),
    )


def config_sequences(cfg: dict, ds: Hyser, keys: list[str]) -> dict[str, Sequence]:
    return build_sequences(ds, keys, *_features(cfg))


def _evaluate_variant(
    splits: list[Split],
    seqs: dict[str, Sequence],
    new_model: Callable[[], Model],
    tag: str,
    save_dir: Path | None,
    front_end: dict,
    all_folds: bool = False,
) -> tuple[list[dict], dict[str, np.ndarray], str, list[dict]]:
    """Score one variant on every split: (rows, predictions keyed '<label>|<seed>|<recording>', training device, loss curve rows).

    Saves the first fold's network (every fold's with `all_folds`) to `save_dir` with `front_end`, so it can be
    streamed from the file alone.
    """
    rows, preds, device, losses = [], {}, "cpu", []  # ridge and constant are NumPy
    for i, sp in enumerate(splits):
        model = new_model()
        split_rows, split_preds = evaluate_split(sp, seqs, model)
        rows += split_rows
        preds.update({f"{tag}|{k}": v for k, v in split_preds.items()})
        if isinstance(model, TCNModel):
            device = model.train_device or device
            losses += [{"split": sp.name, "step": step, "loss": v} for step, v in enumerate(model.train_loss, 1)]
            if save_dir is not None and (i == 0 or all_folds):
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_state(
                    save_dir / f"{tag.replace('|', '_seed')}_{sp.name}.state.pt",
                    {"training": f"runs/{run_name(save_dir.parent)}, fold {sp.name}, {tag}", **git_state()},
                    front_end,
                )
    return rows, preds, device, losses


def run_summary(df: pd.DataFrame, splits: list[Split], n_recordings: int, started: float, devices: dict[tuple[str, int], str]) -> dict:
    """Timing, sizes, training device and mean over people of each summary metric per model and seed."""
    means = {m: summary(df, m, ["model", "seed"]).set_index(["model", "seed"]) for m in SUMMARY_METRICS}
    per_model = [
        {
            "model": model,
            "seed": int(seed),
            "train_device": devices[(model, int(seed))],
            **{m: round(float(t.loc[key, "mean"]), 4) for m, t in means.items()},
        }
        for key in means["mae"].index
        for model, seed in [key]
    ]
    return {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seconds": round(time.time() - started, 1),
        "splits": len(splits),
        "recordings": n_recordings,
        "subjects_scored": int(df.subject.nunique()),
        "models": per_model,
    }


def splits_record(splits: list[Split], calibration: dict | None = None) -> dict:
    """Recordings each fold trained, tested (and, if given, calibrated) on, for the run record."""
    folds = {sp.name: {"held_out_subject": sp.held_out_subject, "train": list(sp.train), "test": list(sp.test)} for sp in splits}
    for name, cal in (calibration or {}).items():
        folds[name]["calibration"] = cal
    return {"protocol": splits[0].protocol, "folds": folds}


def fold_checkpoints(run_dir: Path, model: str = "tcn", seed: int = 0) -> list[tuple[int, Path]]:
    """(person, checkpoint) for every fold model a run saved (`save_models: all`). An error if it saved none."""
    paths = sorted((run_dir / "models").glob(f"{model}_seed{seed}_s*.state.pt"))
    if not paths:
        raise FileNotFoundError(f"no {model} fold checkpoints in {run_dir / 'models'} (the run needs save_models: all)")
    return [(int(p.name.split("_s")[-1].split(".")[0]), p) for p in paths]


PARTIAL = ".partial"  # suffix of a run folder while the run is being written


def run_name(folder: Path) -> str:
    """The run a folder belongs to, also while it is still being written."""
    return folder.name.removesuffix(PARTIAL)


@contextmanager
def fresh_run_dir(out_dir: Path) -> Iterator[Path]:
    """Write a run into `<out_dir>.partial/` and replace `out_dir` with it at the end: a rerun leaves no file of an
    earlier one behind, and an interrupted run leaves the previous results untouched."""
    work = out_dir.with_name(out_dir.name + PARTIAL)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    yield work
    shutil.rmtree(out_dir, ignore_errors=True)
    work.rename(out_dir)


def _save_outputs(
    out_dir: Path, cfg: dict, df: pd.DataFrame, preds: dict, splits: list[Split], summary_info: dict, env: dict | None = None
) -> None:
    """`env` is the environment captured when the run started (a commit made during a long run must not be recorded)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "env.json").write_text(json.dumps(env or env_info(), indent=2))
    (out_dir / "splits.json").write_text(json.dumps(splits_record(splits), indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary_info, indent=2))
    if preds:
        np.savez_compressed(out_dir / "predictions.npz", **preds)


def run(cfg: dict, root: Path, out_dir: Path | None = None, quiet: bool = False) -> pd.DataFrame:
    """Run every model variant on every split. Return per-finger metrics."""
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    check_config(cfg)
    env = env_info()  # at the start: the code that actually ran
    out_dir = out_dir or REPO / "runs" / cfg["name"]
    ds = Hyser(root)
    splits = make_splits(cfg["protocol"], resolve_subjects(cfg["subjects"]))
    front_end = front_end_of(cfg)
    started = t0 = time.time()
    keys = recordings_on_disk(splits, ds)
    seqs = config_sequences(cfg, ds, keys)
    log(f"{len(splits)} splits, {len(keys)} recordings, features in {time.time() - t0:.0f}s")

    ridge_cache = StatsCache()  # shared by the ridge variants and folds of this run. Bounded in memory
    rows, preds, devices, losses = [], {}, {}, []
    with fresh_run_dir(out_dir) as work:
        for kind, label, params, seed in model_variants(cfg["models"], cfg.get("seeds", [0])):
            t0 = time.time()
            new_model = partial(make_model, kind, params, int(cfg["history"]), seed, ridge_cache)
            save_dir = work / "models" if cfg.get("save_models") and kind in ("tcn", "lstm") else None
            variant_rows, variant_preds, devices[(label, seed)], variant_losses = _evaluate_variant(
                splits, seqs, new_model, f"{label}|{seed}", save_dir, front_end, cfg.get("save_models") == "all"
            )
            rows += [{**r, "model": label, "seed": seed} for r in variant_rows]
            losses += [{"model": label, "seed": seed, **r} for r in variant_losses]
            if cfg.get("save_predictions"):
                preds.update(variant_preds)
            log(f"  {label} seed {seed}: {time.time() - t0:.0f}s")

        df = pd.DataFrame(rows).assign(
            protocol=cfg["protocol"], mvc_convention=cfg["mvc_convention"], input_norm=cfg.get("input_norm", "none")
        )
        check_complete(df, {k for sp in splits for k in sp.test}, ["model", "seed"])
        _save_outputs(work, cfg, df, preds, splits, run_summary(df, splits, len(keys), started, devices), env)
        if losses:
            pd.DataFrame(losses).to_csv(work / "train_loss.csv", index=False)  # one row per network, fold and step
    for metric in SUMMARY_METRICS:
        log(f"\n{metric} (%MVC, hierarchical mean over fingers -> recordings -> sessions -> people)")
        log(summary(df, metric, ["model", "seed"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(int(cfg.get("threads", 4)))
    run(cfg, args.root)


if __name__ == "__main__":
    main()
