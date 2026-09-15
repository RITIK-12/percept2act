#!/usr/bin/env bash
# OPTIONAL UPGRADE -- do this only after the end-to-end loop works.
#
# STAGE 1 dataset -- ACT: pick a brick off the pile and present it at a fixed
# inspection station.
#
# Why bother: presenting every brick at the same pose, scale and lighting
# collapses the nuisance variance PatchCore has to model, which is what makes
# a memory-bank detector accurate. Worth it if time remains; not required for
# a closed loop.
#
# This motion is identical for every brick regardless of its condition, so this
# dataset is defect-independent: it stays valid no matter what the defect turns
# out to be.
#
# Drive the follower with the LEADER arm. One episode = one brick moved from the
# pile to the station, then hands off.
#
# Usage:  bash scripts/02_record_stage1.sh [num_episodes]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
ARGS="$REPO/src/percept2act/lerobot_args.py"
export PATH="$HOME/miniforge3/envs/hack_lerobot/bin:$PATH"

N="${1:-$("$PY" -c "
import sys;sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('policies.stage1_present.episodes'))")}"

CAMS=$("$PY" "$ARGS" cameras)
TASK=$("$PY" "$ARGS" instruction --key present)
REPO_ID=$("$PY" -c "
import sys;sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('policies.stage1_present.dataset_repo_id'))")

echo "STAGE 1 · $N episodes"
echo "  task     : $TASK"
echo "  dataset  : $REPO_ID -> $REPO/datasets"
echo "  cameras  : $CAMS"
echo
echo "Per episode: pick ONE brick from the pile, place it at the inspection"
echo "station, return toward home. Vary brick colour, size and start position."
echo "Right arrow = end episode early · Left arrow = redo · Esc = stop"
echo
read -rp "Press Enter when the scene is staged... " _

lerobot-record \
  --robot.type="$("$PY" "$ARGS" follower-type)" \
  --robot.port="$("$PY" "$ARGS" follower-port)" \
  --robot.id="$("$PY" "$ARGS" follower-id)" \
  --robot.cameras="$CAMS" \
  --teleop.type="$("$PY" "$ARGS" leader-type)" \
  --teleop.port="$("$PY" "$ARGS" leader-port)" \
  --teleop.id="$("$PY" "$ARGS" leader-id)" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$REPO/datasets/stage1_present" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$N" \
  --dataset.fps=30 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  "${@:2}"

echo
echo "Done. Inspect with:  lerobot-dataset-viz --repo-id $REPO_ID --root $REPO/datasets/stage1_present"
