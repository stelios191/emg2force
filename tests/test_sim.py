"""Simulator: force mapping, contact-force aggregation, deterministic reset and the reported error definitions."""

import mujoco
import numpy as np
import pytest

from emg2force.gripper import DT, GripMapping, GripperSim, response_times
from emg2force.replay import error_row


def test_mapping_uses_thumb_and_index_flexion_only_and_stays_in_range():
    m = GripMapping()
    f = np.zeros((6, 5))
    f[1, :2] = -15.0  # thumb + index flexion at half of full scale (30 %MVC)
    f[2, :2] = -300.0  # far beyond full scale
    f[3, :2] = +50.0  # extension does not grip
    f[4, 2:] = -100.0  # middle, ring, little are ignored
    f[5, 0] = -30.0  # thumb alone at full scale = half of the mean
    g = m(f)
    np.testing.assert_allclose(g, [1.0, 8.0, 15.0, 1.0, 1.0, 8.0])
    assert g.min() >= m.f_hold
    assert g.max() <= m.f_hold + m.f_range


def test_error_definitions():
    ref = np.array([2.0, 2.0, 2.0, 2.0])
    setp = np.array([1.0, 2.0, 3.0, 2.0])
    contact = np.array([1.5, 2.0, 13.0, 0.0])
    r = error_row(contact, setp, ref)
    assert r["decoder_mae_N"] == pytest.approx(np.mean([1, 0, 1, 0]))  # |setpoint - reference|
    assert r["tracking_mae_N"] == pytest.approx(np.mean([0.5, 0, 10, 2]))  # |contact - setpoint|
    assert r["whole_chain_mae_N"] == pytest.approx(np.mean([0.5, 0, 11, 2]))  # |contact - reference|
    assert r["whole_chain_bias_N"] == pytest.approx(np.mean([-0.5, 0, 11, -2]))
    assert r["time_under_grip"] == 0.25
    assert r["time_over_grip"] == 0.25


@pytest.fixture(scope="module")
def sim():
    s = GripperSim()
    s.calibrate()
    return s


def test_open_gripper_measures_no_contact_and_closing_does(sim):
    mujoco.mj_resetData(sim.m, sim.d)
    assert sim.force() == 0.0
    assert sim.settle(2.0, 0.5) > 0.0


def test_replay_is_deterministic_after_reset_and_tracks_a_constant_setpoint(sim):
    setpoint = np.full(150, 6.0)
    a, b = sim.replay(setpoint), sim.replay(setpoint)
    assert np.array_equal(a["contact"], b["contact"])
    assert abs(a["contact"][-50:].mean() - 6.0) < 0.5


def test_response_times_of_a_known_first_order_step():
    tau = 0.03
    t = (np.arange(500) + 1) * DT  # sample i is read after physics step i + 1
    r = response_times(1 + 7 * (1 - np.exp(-t / tau)), 1.0, 8.0)
    assert 0 <= r["t50_ms"] - 1e3 * tau * np.log(2) < 1e3 * DT  # the first sample past the crossing
    assert 0 <= r["t90_ms"] - 1e3 * tau * np.log(10) < 1e3 * DT
    assert r["overshoot"] == 0.0
    assert np.isnan(response_times(np.ones(50), 1.0, 8.0)["t10_ms"])  # a force that never moves has no crossing


def test_gripper_follows_a_setpoint_step_up_and_down(sim):
    for a, b in ((1.0, 8.0), (8.0, 1.0)):
        r = response_times(sim.step_response(a, b), a, b)
        assert 0 < r["t50_ms"] < r["t90_ms"] < 200  # reaches the new force within a fraction of a second
        assert r["overshoot"] < 0.2
