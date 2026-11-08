"""MuJoCo Robotiq 2F-85 (Menagerie) under PI force control, driven by a grip setpoint from decoded finger force.

The position servo is replaced by a tendon motor with a 30 ms lag. The object is a fixed box. Physics and
controller run at 500 Hz, the setpoint at 100 Hz. Grip force is the normal pad-object contact force.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSET = Path(__file__).resolve().parent / "robotiq_2f85" / "2f85.xml"
DT = 0.002
SUBSTEPS = 5  # 5 x 2 ms = one 10 ms setpoint period
OBJECT_HALF = (0.015, 0.02, 0.02)
OBJECT_POS = (0.0, 0.0, 0.152)
ACTUATOR_TAU = 0.03  # s, first-order motor/transmission lag (design choice, instant force is unrealistic)
OBJECT_SOLREF = 0.02  # s, contact time constant of the grasped object (larger = softer)


# Mapping from decoded finger forces to a grasp setpoint
@dataclass(frozen=True)
class GripMapping:
    """F_set = F_hold + F_range * clip(mean(thumb, index flexion) / full_scale, 0, 1). Flexion = negative %MVC."""

    full_scale_pct: float = 30.0  # %MVC flexion that maps to the maximum grip
    f_hold: float = 1.0  # N, light hold so the object stays grasped
    f_range: float = 14.0  # N, so setpoints span 1-15 N

    def __call__(self, force_pct: np.ndarray) -> np.ndarray:
        flex = np.clip(-force_pct[:, :2], 0, None).mean(1)  # thumb, index
        g = np.clip(flex / self.full_scale_pct, 0, 1)
        return self.f_hold + self.f_range * g


# Model
def build() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.option.gravity = [0, 0, 0]
    spec.option.timestep = DT
    spec.delete(spec.actuator("fingers_actuator"))
    act = spec.add_actuator(name="grip_motor", target="split", trntype=mujoco.mjtTrn.mjTRN_TENDON)
    act.gainprm[0] = 1.0
    act.dyntype = mujoco.mjtDyn.mjDYN_FILTER  # first-order actuator lag: force follows ctrl with tau
    act.dynprm[0] = ACTUATOR_TAU
    act.ctrllimited = 1
    act.ctrlrange = [-3.0, 5.0]
    body = spec.worldbody.add_body(name="object", pos=list(OBJECT_POS))
    body.add_geom(
        name="object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(OBJECT_HALF),
        rgba=[0.5, 0.7, 0.5, 1],
        solref=[OBJECT_SOLREF, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        friction=[0.8, 0.005, 0.0001],
        priority=2,  # object's contact parameters win over the pads'
    )
    return spec.compile()


def grip_force(m: mujoco.MjModel, d: mujoco.MjData, obj_id: int, left: set, right: set) -> float:
    f6 = np.zeros(6)
    per = {"l": 0.0, "r": 0.0}
    for i in range(d.ncon):
        c = d.contact[i]
        g = {c.geom1, c.geom2}
        if obj_id not in g:
            continue
        side = "l" if g & left else "r" if g & right else None
        if side is None:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        per[side] += f6[0]
    return 0.5 * (per["l"] + per["r"])


@dataclass
class PIController:
    kp: float = 0.05  # tendon N per N of grip-force error (frozen: tuned on synthetic steps only)
    ki: float = 0.5  # per second
    ff_gain: float = 1.0  # tendon N per N of grip force (set by `calibrate`)
    u_min: float = -3.0
    u_max: float = 5.0
    integral: float = 0.0  # integrator state. reset() clears it between recordings

    def reset(self) -> None:
        self.integral = 0.0

    def __call__(self, f_set: float, f_meas: float) -> float:
        e = f_set - f_meas
        u_unsat = self.ff_gain * f_set + self.kp * e + self.integral
        u = float(np.clip(u_unsat, self.u_min, self.u_max))
        if u == u_unsat or np.sign(e) != np.sign(u_unsat - u):  # conditional integration (anti-windup)
            self.integral += self.ki * e * DT
        return u


class GripperSim:
    def __init__(self, ctrl: PIController | None = None):
        self.m = build()
        self.d = mujoco.MjData(self.m)
        self.obj_id = self.m.geom("object").id
        self.left = {self.m.geom("left_pad1").id, self.m.geom("left_pad2").id}
        self.right = {self.m.geom("right_pad1").id, self.m.geom("right_pad2").id}
        self.ctrl = ctrl or PIController()

    def force(self) -> float:
        return grip_force(self.m, self.d, self.obj_id, self.left, self.right)

    def settle(self, u: float, seconds: float) -> float:
        for _ in range(int(seconds / DT)):
            self.d.ctrl[0] = u
            mujoco.mj_step(self.m, self.d)
        return self.force()

    def calibrate(self) -> float:
        """Static tendon-force -> grip-force gain from open-loop holds. Sets the feed-forward."""
        mujoco.mj_resetData(self.m, self.d)
        us = [0.5, 1.0, 2.0, 3.0]
        fs = [self.settle(u, 1.0) for u in us]
        k = np.polyfit(fs, us, 1)[0]  # tendon N per grip N
        self.ctrl.ff_gain = float(k)
        return float(k)

    def step_response(self, f_from: float, f_to: float, before_s: float = 1.0, after_s: float = 1.0) -> np.ndarray:
        """Contact force every DT after the setpoint steps from f_from to f_to (N), in closed loop."""
        mujoco.mj_resetData(self.m, self.d)
        self.ctrl.reset()
        self.settle(self.ctrl.ff_gain * f_from, 1.0)  # close onto the object first, as `replay` does
        out = []
        for k in range(int(round((before_s + after_s) / DT))):
            self.d.ctrl[0] = self.ctrl(f_from if k * DT < before_s else f_to, self.force())
            mujoco.mj_step(self.m, self.d)
            out.append(self.force())
        return np.array(out[int(round(before_s / DT)) :])

    def replay(self, setpoint: np.ndarray, pregrasp_s: float = 1.0, record_qpos: bool = False) -> dict:
        """Run the closed loop on a 100 Hz setpoint. Return 100 Hz logs."""
        mujoco.mj_resetData(self.m, self.d)
        self.ctrl.reset()
        self.settle(self.ctrl.ff_gain * setpoint[0], pregrasp_s)  # close onto the object first
        meas, cmd = np.zeros(len(setpoint)), np.zeros(len(setpoint))
        qpos = np.zeros((len(setpoint), self.m.nq)) if record_qpos else None
        for j, fs in enumerate(setpoint):
            for _ in range(SUBSTEPS):
                u = self.ctrl(fs, self.force())
                self.d.ctrl[0] = u
                mujoco.mj_step(self.m, self.d)
            meas[j] = self.force()
            cmd[j] = u
            if qpos is not None:
                qpos[j] = self.d.qpos
        return {"setpoint": setpoint, "contact": meas, "u": cmd, "qpos": qpos}


def response_times(force: np.ndarray, f_from: float, f_to: float, dt: float = DT) -> dict:
    """Times (ms) to 10, 50 and 90 % of a step, to stay within 5 % of the target, and the overshoot.

    `force[i]` is read after physics step i + 1, so it sits at (i + 1) * dt. NaN = never reached.
    """
    frac = (force - f_from) / (f_to - f_from)
    t = (np.arange(len(frac)) + 1) * dt * 1e3
    outside = np.flatnonzero(np.abs(frac - 1) > 0.05)
    return {
        **{f"t{q}_ms": float(t[np.argmax(frac >= q / 100)]) if (frac >= q / 100).any() else float("nan") for q in (10, 50, 90)},
        "settle_5pct_ms": (float(t[outside[-1] + 1]) if outside[-1] + 1 < len(t) else float("nan")) if len(outside) else 0.0,
        "overshoot": float(max(frac.max() - 1, 0.0)),
    }
