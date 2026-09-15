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

CLASS="${1:?usage: 03_record_stage2.sh <coral|blue> [num_episodes]}"
# Plate names are accepted as aliases, and are the clearer way to think about
# it: this argument selects the DESTINATION, never the brick's condition.
case "$CLASS" in
  coral)     CLASS=defective ;;
  blue)      CLASS=good ;;
  good|defective) ;;
  *) echo "argument must be coral, blue, good or defective"; exit 2 ;;
esac

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
echo "############################################################"
echo "# USE ANY BRICKS. Undamaged ones are fine for BOTH classes. #"
echo "############################################################"
echo
echo "'$CLASS' here names the DESTINATION PLATE, not the brick's condition."
echo "You are recording the motion '-> $PLATE plate'. Nothing in this recording"
echo "looks at whether a brick is damaged -- Anomalib decides that at runtime and"
echo "picks which instruction to send."
echo
echo "Recording damaged bricks only for one class would teach the policy to route"
echo "by how the brick LOOKS instead of by the instruction, which is the one thing"
echo "that must not happen. Keep the visual scene as similar as possible between"
echo "the two classes; only the destination should differ."
echo
echo "Per episode: ONE brick at the taped spot. Pick it up, place it in the"
echo "$PLATE plate, return toward home."
echo
echo "KEYS (do NOT use Ctrl-C -- it kills the process before the dataset is saved)"
echo "  Right arrow : end this episode, go to the next"
echo "  Left arrow  : discard this episode and redo it"
echo "  Escape      : stop recording early and SAVE cleanly"
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
