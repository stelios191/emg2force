"""Shared fixtures: one small end-to-end run on a synthetic Hyser download, reused by the pipeline tests."""

import pytest

from emg2force import dataset
from experiments.run import run
from tests.helpers import SMOKE_CFG, make_fake_hyser


@pytest.fixture(autouse=True)
def _private_frame_cache(request, tmp_path, monkeypatch):
    """Synthetic recordings share keys with real ones (s02_d1_t1, ...), so only tests on the real download use the real cache."""
    if request.node.get_closest_marker("data") is None:
        monkeypatch.setattr(dataset, "CACHE", tmp_path / "cache")


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory):
    """(hyser root, run dir, metrics) of experiments.run on subject 2: constant, ridge and a 3-step TCN, cross-day."""
    tmp = tmp_path_factory.mktemp("smoke")
    root, out = tmp / "hyser", tmp / "run"
    make_fake_hyser(root, subjects=(2,), trials=(1, 2, 3, 4, 5))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset, "CACHE", tmp / "cache")
        yield root, out, run(SMOKE_CFG, root, out, quiet=True)
