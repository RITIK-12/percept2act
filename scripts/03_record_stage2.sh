#!/usr/bin/env bash
# STAGE 2 dataset -- SmolVLA: take the brick from the inspection station and
# place it in the plate named by the instruction.
#
# Recorded as ONE dataset containing BOTH instruction strings, which is the
# multi-task form SmolVLA fine-tuning expects. The instruction -- not the visual
# scene -- is what selects the destination, so at inference time Anomalib's
# verdict picks the string and the same policy routes the brick.
#
# Usage:
#   bash scripts/03_record_stage2.sh good      [n]   # -> blue plate
#   bash scripts/03_record_stage2.sh defective [n]   # -> coral plate
#
# Record BOTH classes. Roughly equal counts. Order does not matter; the second
# invocation appends via --resume.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
ARGS="$REPO/src/percept2act/lerobot_args.py"
export PATH="$HOME/miniforge3/envs/hack_lerobot/bin:$PATH"

CLASS="${1:?usage: 03_record_stage2.sh <good|defective> [num_episodes]}"
case "$CLASS" in good|defective) ;; *) echo "class must be good or defective"; exit 2;; esac

N="${2:-$("$PY" -c "
import sys;sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('policies.stage2_sort.episodes_per_instruction'))")}"

CAMS=$("$PY" "$ARGS" cameras)
read -r TASK PLATE < <("$PY" -c "
import sys;sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
s=Scenario.load()
print(s.instruction_for_class('$CLASS'), s.plate_for_class('$CLASS'))")
REPO_ID=$("$PY" -c "
import sys;sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('policies.stage2_sort.dataset_repo_id'))")

ROOT="$REPO/datasets/stage2_sort"
RESUME=false
[[ -d "$ROOT" ]] && RESUME=true

echo "STAGE 2 · class=$CLASS · $N episodes · resume=$RESUME"
echo "  instruction : \"$TASK\""
echo "  destination : $PLATE plate"
echo "  dataset     : $REPO_ID -> $ROOT"
echo
echo "Per episode: ONE brick on the mat. Pick it up, place it in the $PLATE"
echo "plate, return toward home. Vary the brick colour, size and position."
echo "The policy learns the destination from the INSTRUCTION, so keep the visual"
echo "scene as similar as you can between the two classes."
echo
read -rp "Press Enter when a brick is staged on the mat... " _

lerobot-record \
  --robot.type="$("$PY" "$ARGS" follower-type)" \
  --robot.port="$("$PY" "$ARGS" follower-port)" \
  --robot.id="$("$PY" "$ARGS" follower-id)" \
  --robot.cameras="$CAMS" \
  --teleop.type="$("$PY" "$ARGS" leader-type)" \
  --teleop.port="$("$PY" "$ARGS" leader-port)" \
  --teleop.id="$("$PY" "$ARGS" leader-id)" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$N" \
  --dataset.fps=30 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --resume="$RESUME" \
  --display_data=true \
  "${@:3}"

echo
echo "Recorded $N '$CLASS' episodes."
echo "When BOTH classes are done, verify both task strings are present:"
echo "  bash scripts/03b_verify_stage2.sh"
