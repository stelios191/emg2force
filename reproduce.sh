#!/usr/bin/env bash
# Every computed result in README.md and docs/experiments.md (the two development tables at the end are kept as text):
# ./reproduce.sh [HYSER_ROOT]
# Needs uv. ~5 GB download, about 7 h on an Apple M1. Trains on the Apple GPU (MPS) if present, else CPU.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="${1:-../hyser_raw}"
run() {
  echo
  echo ">>> $*"
  "$@"
}

run uv sync --frozen
run uv run emg2force-download --root "$ROOT" --subset random mvc  # checksum-verified
run uv run pytest -q

# The 16 evaluation people: final recipes, baselines and the first recipe (experiments/configs/*.yaml)
for c in experiments/configs/final_*.yaml
do
  run uv run python -m experiments.run "$c" --root "$ROOT"
done
run uv run python -m experiments.calibration experiments/configs/calibration_curve.yaml --root "$ROOT"
run uv run python -m experiments.same_day --root "$ROOT"
run uv run python -m experiments.hold_test runs/final_cross_day_pop_256 runs/final_cross_day_pop_256_running_mean --root "$ROOT"
run uv run python -m experiments.behaviour --root "$ROOT"
run uv run python -m experiments.monitor_check runs/final_cross_day_pop_256 --root "$ROOT"

# Latency of the released decoder (3 repeats on an idle machine), and the simulated gripper
RELEASED=runs/final_cross_day_pop_256/models/tcn_seed0_s02.state.pt  # = assets/decoder.state.pt
for rep in 1 2 3
do
  run uv run emg2force-benchmark --model "$RELEASED" --emg recorded --root "$ROOT" --out results/latency --threads 1 --tag "_rep$rep"
done
run uv run emg2force-replay --run runs/final_cross_day_256 --root "$ROOT" --out results/mujoco_cross_day
run uv run emg2force-replay --run runs/final_cross_day_pop_256 --root "$ROOT" --out results/mujoco_cross_day_pop --models tcn
run uv run python -m experiments.report

# Demo video and GIF (MuJoCo's offscreen rendering needs an OpenGL context), then the committed README assets
run uv run python -m experiments.demo_video --root "$ROOT"
cp "$RELEASED" assets/decoder.state.pt
cp results/demo.gif assets/demo.gif
cp results/new_day_trial.png assets/new_day_trial.png
