"""Rendering of the demo video and README GIF (experiments.demo_video chooses what to show): EMG input, decoded
against measured force, and the simulated gripper it drives.

Every frame has three zones, input to output: the four 8 x 8 electrode grids (EMG level over the last 100 ms), the
decoded and measured force on a fixed 10 s axis with a sweeping cursor (one grip finger in %MVC, and the grip force in
newtons), and the MuJoCo gripper with its object coloured by force and a force gauge. The video pages through whole
recordings 10 s at a time. `render_gif` writes one 10 s window. Everything shown is replayed from saved predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from PIL import Image

from emg2force.causal import FilterConfig
from emg2force.dataset import frames_for, warmup_frames
from emg2force.gripper import OBJECT_POS, GripMapping, GripperSim
from emg2force.hyser import FINGERS, FS_FORCE, GRID_CHANNELS, GRIDS, Hyser, parse_key

Writer = Any  # imageio's video writer (no type information)
FPS = 25  # video
GIF_FPS = 10  # GIF delays are whole centiseconds: 100 ms frames play the window at exactly 1x
PAGE_S = 10.0  # seconds on one time axis
CARD_SECONDS = 5.0  # caption cards: long enough to read on mute
FIG_IN = (9.6, 4.8)  # frame size in inches: 960 x 480 at 100 dpi (GIF), 1280 x 640 at 133 dpi (video)
VIDEO_DPI = 400 / 3
MEASURED, DECODED, DEAD, BACKDROP = "#222222", "#D55E00", "#8C8C8C", (238, 238, 238)  # Okabe-Ito vermillion
EMG_MAP = LinearSegmentedColormap.from_list("emg", ["#D6EAF6", "#3A8FC7", "#08304A"])
FORCE_MAP = LinearSegmentedColormap.from_list("force", ["#F6E3D6", DECODED])
GRIP_FINGERS = (0, 1)  # thumb and index: the fingers whose flexion sets the grip (GripMapping)
GAUGE_N = 12.0  # gauge and grip axis: most grips stay well below the 15 N maximum
COLOUR_FULL_N = 8.0  # the object is fully coloured at this grip force (most grips stay below it)
QUIET_LOG = 1.5  # display only: an electrode this far (log-RMS) below its grid median over the recording has little or no signal
GRID_NAMES = ("extensors\ndistal", "extensors\nproximal", "flexors\ndistal", "flexors\nproximal")


def recording_name(key: str) -> str:
    """'s11_d2_t4' -> 'held-out recording: person 11, day 2, trial 4'."""
    s, d, t = parse_key(key)
    return f"held-out recording: person {s}, day {d}, trial {t}"


@dataclass
class Clip:
    """One recording replayed: labels, predictions, EMG frames and the gripper driven by the predictions."""

    y: np.ndarray  # (T, 5) measured force, %MVC, from the end of the warm-up on
    yhat: np.ndarray  # (T, 5) decoded force
    emg: np.ndarray  # (T, 256) log-RMS frames
    target_n: np.ndarray  # (T,) grip target from the measured force, N
    gripper_n: np.ndarray  # (T,) force the simulated gripper reached, N
    qpos: np.ndarray  # (T, nq) gripper state per frame
    t: np.ndarray  # (T,) seconds since the recording began
    dead: np.ndarray  # (256,) electrodes with little or no signal (shown grey)
    level: tuple[float, float]  # EMG colour range

    @classmethod
    def load(cls, run_dir: Path, key: str, root: Path, model: str = "tcn") -> Clip:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        with np.load(run_dir / "predictions.npz") as z:
            yhat = z[f"{model}|0|{key}"].astype(np.float64)
        ds = Hyser(root)
        s, d, t = parse_key(key)
        y = ds.target(ds.load_force(s, d, t), cfg["mvc_convention"], cfg["label_filter"]).values
        kw: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("filter", {}).items()}
        emg = frames_for(ds, key, FilterConfig(**kw))  # the EMG level, whatever the model's inputs
        start = warmup_frames(cfg)
        mapping = GripMapping()
        target, decoded = mapping(y)[start:], mapping(yhat)[start:]
        sim = GripperSim()
        sim.calibrate()
        log = sim.replay(decoded, record_qpos=True)
        level = np.median(emg[start:], axis=0)
        dead = level < np.repeat(np.median(level.reshape(len(GRIDS), GRID_CHANNELS), axis=1), GRID_CHANNELS) - QUIET_LOG
        lo, hi = np.percentile(emg[start:][:, ~dead], [5, 99.5])
        n = len(target)
        return cls(y[start:n + start], yhat[start:n + start], emg[start:n + start], target, log["contact"][:n],
                   log["qpos"][:n], (np.arange(n) + start) / FS_FORCE, dead, (float(lo), float(hi)))  # fmt: skip


class GripperView:
    """Renders the gripper on a light backdrop, close to the pads, with the object coloured by the force it holds."""

    def __init__(self, size: int):
        self.sim = GripperSim()
        self.renderer = mujoco.Renderer(self.sim.m, size, size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2] - 0.01)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.22, 90.0, -12.0
        self.sim.m.vis.headlight.ambient[:] = 0.55
        self.sim.m.vis.headlight.diffuse[:] = 0.7

    def render(self, qpos: np.ndarray, force_n: float) -> np.ndarray:
        m, d = self.sim.m, self.sim.d
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        m.geom_rgba[self.sim.obj_id] = (*FORCE_MAP(min(force_n / COLOUR_FULL_N, 1.0))[:3], 1.0)  # display only
        self.renderer.update_scene(d, self.cam)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        background = self.renderer.render()[..., 0] == -1
        self.renderer.disable_segmentation_rendering()
        rgb[background] = BACKDROP
        return rgb

    def close(self) -> None:
        self.renderer.close()


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5")
    ax.tick_params(labelsize=10)


def _end_labels(ax: Any, t: float, t0: float, values: tuple[float, float], names: tuple[str, str], span: float) -> None:
    """Label the two lines at the cursor, kept apart when the lines meet, and inside the axis near its end."""
    black, orange = values
    gap = 0.1 * span
    if abs(black - orange) < gap:
        mid = (black + orange) / 2
        black, orange = (mid + gap / 2, mid - gap / 2) if black >= orange else (mid - gap / 2, mid + gap / 2)
    near_end = t - t0 > 0.82 * PAGE_S
    x, ha = (t - 0.012 * PAGE_S, "right") if near_end else (t + 0.012 * PAGE_S, "left")
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5}
    ax.text(x, black, names[0], fontsize=10.5, color=MEASURED, va="center", ha=ha, bbox=box, zorder=5)
    ax.text(x, orange, names[1], fontsize=10.5, color=DECODED, va="center", ha=ha, bbox=box, zorder=5)


def render_frame(clip: Clip, view: GripperView, j: int, page: slice, title: str, tag: str, dpi: float) -> np.ndarray:
    """(H, W, 3) frame at index j of `clip`, with the time axis of `page` revealed up to j."""
    fig = plt.figure(figsize=FIG_IN, dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.text(0.012, 0.955, title, fontsize=12.5, ha="left", va="center")
    fig.text(0.988, 0.955, tag, fontsize=10, color="#777777", ha="right", va="center")

    level = clip.emg[max(0, j - 9) : j + 1].mean(0)  # last 100 ms
    lo, hi = clip.level
    for g in range(len(GRIDS)):
        ax = fig.add_axes((0.012 + (g % 2) * 0.085, 0.50 - (g // 2) * 0.30, 0.078, 0.156))
        cells = slice(g * GRID_CHANNELS, (g + 1) * GRID_CHANNELS)
        img = EMG_MAP(np.clip((level[cells] - lo) / (hi - lo), 0, 1)).reshape(8, 8, 4)
        img[clip.dead[cells].reshape(8, 8)] = to_rgba(DEAD)
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
        ax.set_title(GRID_NAMES[g], fontsize=9.5, color="#555555", pad=2)
    fig.text(0.098, 0.135, "256-ch forearm EMG", fontsize=10.5, ha="center")
    if clip.dead.any():
        fig.text(0.098, 0.085, "grey: little or no signal", fontsize=9.5, ha="center", color="#777777")
    fig.text(0.198, 0.50, "→", fontsize=22, ha="center", va="center", color="#555555")

    finger = GRIP_FINGERS[int(np.argmax(clip.y[page, list(GRIP_FINGERS)].std(axis=0)))]  # the grip finger moving more
    seen = slice(page.start, j + 1)
    t0 = clip.t[page.start]
    ax_f = fig.add_axes((0.255, 0.53, 0.38, 0.33))
    ax_g = fig.add_axes((0.255, 0.12, 0.38, 0.33))
    flex, flex_hat = -clip.y[:, finger], -clip.yhat[:, finger]  # flexion up: it is what closes the gripper
    for ax, measured, decoded in ((ax_f, flex, flex_hat), (ax_g, clip.target_n, clip.gripper_n)):
        ax.plot(clip.t[seen], measured[seen], color=MEASURED, lw=1.8)
        ax.plot(clip.t[seen], decoded[seen], color=DECODED, lw=1.6)
        ax.set_xlim(t0, t0 + PAGE_S)
        ax.axvline(clip.t[j], color="#999999", lw=0.8)
        _style(ax)
    window = np.r_[flex[page], flex_hat[page]]
    ax_f.set_ylim(window.min() - 5, window.max() + 5)
    grip_top = max(GAUGE_N, float(clip.target_n[page].max()), float(clip.gripper_n[page].max())) + 1
    ax_g.set_ylim(0, grip_top)
    ax_g.set_yticks(np.arange(0, grip_top, 5))
    ax_f.set_title(f"{FINGERS[finger]} flexion (%MVC): measured vs decoded", fontsize=11, loc="left", pad=3)
    ax_g.set_title("grip force (N): target vs gripper", fontsize=11, loc="left", pad=3)
    ax_f.set_xticklabels([])
    ax_g.set_xlabel("time in the recording (s)", fontsize=10)
    _end_labels(ax_f, clip.t[j], t0, (flex[j], flex_hat[j]), ("measured", "decoded"), float(np.ptp(window)) + 10)
    _end_labels(ax_g, clip.t[j], t0, (clip.target_n[j], clip.gripper_n[j]), ("target", "gripper"), grip_top)

    force, target = clip.gripper_n[j], clip.target_n[j]
    ax_r = fig.add_axes((0.655, 0.06, 0.285, 0.84))
    ax_r.imshow(view.render(clip.qpos[j], force))
    ax_r.axis("off")
    ax_b = fig.add_axes((0.955, 0.16, 0.018, 0.62))
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, GAUGE_N)
    ax_b.bar(0.5, min(force, GAUGE_N), width=1, color=DECODED)
    ax_b.axhline(target, color="black", lw=2.2)
    ax_b.set_xticks([])
    ax_b.set_yticks([0, GAUGE_N])
    ax_b.yaxis.tick_right()
    ax_b.tick_params(labelsize=9.5, length=2)
    for spine in ax_b.spines.values():
        spine.set_color("#999999")
    fig.text(0.964, 0.84, f"{force:.1f} N", fontsize=13, ha="center", color=DECODED, fontweight="bold")
    fig.text(0.964, 0.10, "gripper\n(— target)", fontsize=9.5, ha="center", color="#555555")
    out = rgb(fig).copy()
    plt.close(fig)
    return out


def _pages(n: int) -> list[slice]:
    per = int(PAGE_S * FS_FORCE)
    return [slice(a, min(a + per, n)) for a in range(0, n, per)]


def render_clip(run_dir: Path, key: str, title: str, root: Path, writer: Writer, model: str = "tcn") -> int:
    """The whole recording, 10 s per page at the video frame rate. Returns the number of video frames written."""
    clip = Clip.load(run_dir, key, root, model)
    view = GripperView(480)
    frames = 0
    try:
        tag = f"{recording_name(key).removeprefix('held-out recording: ')} · replayed 1×"
        for page in _pages(len(clip.t)):
            for j in range(page.start, page.stop, int(FS_FORCE // FPS)):
                writer.append_data(render_frame(clip, view, j, page, title, tag, VIDEO_DPI))
                frames += 1
    finally:
        view.close()
    return frames


def render_gif(run_dir: Path, key: str, root: Path, out: Path, start_s: float, model: str = "tcn") -> None:
    """The README GIF: the PAGE_S window from `start_s` (seconds in the recording), 960 x 480, GIF_FPS, looping."""
    clip = Clip.load(run_dir, key, root, model)
    a = int(np.searchsorted(clip.t, start_s))
    page = slice(a, min(a + int(PAGE_S * FS_FORCE), len(clip.t)))
    view = GripperView(360)
    frames = []
    try:
        for j in range(page.start, page.stop, int(FS_FORCE // GIF_FPS)):  # one image per 100 ms of data
            img = render_frame(
                clip, view, int(j), page, "Held-out day-2 recording → decoded force → simulated gripper", "replayed · 1× · simulation", 100
            )
            frames.append(Image.fromarray(img).convert("P", palette=Image.Palette.ADAPTIVE, colors=96, dither=Image.Dither.NONE))
    finally:
        view.close()
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000 / GIF_FPS)), loop=0, optimize=True)


def rgb(fig: Figure) -> np.ndarray:
    """(H, W, 3) view of a drawn Agg figure."""
    fig.canvas.draw()
    return np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())[..., :3]


def title_card(writer: Writer, lines: list[str], seconds: float = 4.0) -> int:
    fig = plt.figure(figsize=FIG_IN, dpi=VIDEO_DPI)
    fig.text(0.5, 0.62, lines[0], ha="center", va="center", fontsize=19, fontweight="bold")  # the first line is the heading
    fig.text(0.5, 0.47, "\n\n".join(lines[1:]), ha="center", va="top", fontsize=14, wrap=True, linespacing=1.3)
    frame = rgb(fig).copy()
    plt.close(fig)
    for _ in range(int(seconds * FPS)):
        writer.append_data(frame)
    return int(seconds * FPS)  # video frames written


def write_video(out: Path, root: Path, steps: list[tuple[Any, ...]]) -> list[float]:
    """Cards and clips in order. A step is ("card", lines, seconds) or ("clip", run_dir, model, recording_key, title).
    Return the video time, in seconds, at which each clip's recording starts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, clip_starts = 0, []
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=1) as w:
        for step in steps:
            if step[0] == "card":
                frames += title_card(w, step[1], step[2])
                continue
            _, run_dir, model, key, title = step
            frames += title_card(w, [title, recording_name(key)], 2.0)
            clip_starts.append(frames / FPS)
            frames += render_clip(Path(run_dir), key, title, root, w, model=model)
    return clip_starts
