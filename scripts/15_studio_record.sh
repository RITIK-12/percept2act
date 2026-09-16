#!/usr/bin/env bash
# Record episodes through the Physical AI Studio UI instead of the lerobot CLI.
#
# This does not drive the recording -- Studio records over a websocket from its
# own web UI, so you teleoperate there. What this script does is the part that
# is easy to get wrong: check Studio is up, hand you the exact task strings, and
# verify the dataset afterwards.
#
# Usage:
#   bash scripts/15_studio_record.sh          # pre-flight + the strings to paste
#   bash scripts/15_studio_record.sh verify   # check what you just recorded
#
# Studio must already be running:
#   bash scripts/13_start_studio.sh backend   # terminal 1
#   bash scripts/13_start_studio.sh ui        # terminal 2
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
DATASET="$REPO/$("$PY" -c "
import sys; sys.path.insert(0, '$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('replay.dataset_root'))")"

read_instruction() {
  "$PY" -c "
import sys; sys.path.insert(0, '$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('task.instructions.$1'))"
}

if [[ "${1:-record}" == "verify" ]]; then
  exec bash "$REPO/scripts/07_verify_dataset.sh"
fi

# A training run holds this dataset open, and LeRobot v3 rewrites meta/ on
# append. Recording into it mid-run can corrupt the run or the dataset.
if pgrep -fa "lerobot-train|physicalai fit" >/dev/null 2>&1; then
  echo "!! A training run is using $DATASET right now:"
  pgrep -fa "lerobot-train|physicalai fit" | sed 's/^/     /'
  echo
  echo "   Recording into it while it is being read can corrupt both. Wait for"
  echo "   training to finish, or point Studio at a different dataset name."
  echo
fi

echo -n "Studio backend : "
curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 http://127.0.0.1:8000/docs \
  2>/dev/null || echo "DOWN -- run: bash scripts/13_start_studio.sh backend"
echo -n "Studio UI      : "
curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 http://127.0.0.1:5173 \
  2>/dev/null || echo "DOWN -- run: bash scripts/13_start_studio.sh ui"
echo

cat <<EOF
Open http://localhost:5173

Point the project at this dataset so new episodes append to the existing 30:

  repo id : $("$PY" -c "
import sys; sys.path.insert(0, '$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('policies.stage2_sort.dataset_repo_id'))")
  root    : $DATASET

COPY THESE TASK STRINGS. Do not retype them.

  defective brick -> coral plate:
    $(read_instruction defective)

  good brick -> blue plate:
    $(read_instruction good)

The policy is conditioned on this sentence and nothing else. A typo, a capital
letter or a different wording creates a separate task class, and the instruction
conditioning silently stops working -- the arm goes to one plate regardless of
the verdict. 06_record.sh guards against this; the Studio UI will not.

When you are done:
  bash scripts/15_studio_record.sh verify
EOF
