#!/usr/bin/env bash
# Fine-tune the sorting policy on the recorded dataset.
#
# SmolVLA is FINE-TUNED from lerobot/smolvla_base, never trained from scratch --
# the base checkpoint is what already understands language, and our ~60 episodes
# only teach it this specific pick-and-place.
#
# Usage:
#   bash scripts/14_train_vla.sh                    # fine-tune with defaults
#   bash scripts/14_train_vla.sh --steps=8000       # shorter run
#   bash scripts/14_train_vla.sh --dry-run          # print the command, run nothing
#
# Any other flag passes straight through to lerobot-train, e.g. --batch_size=32.
#
# Takes 45-120 min on the Arc iGPU. Run it in its own terminal and get on with
# the demo while it trains -- the detector is on the NPU and replay is CPU-only,
# so they do not contend.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
export PATH="$HOME/miniforge3/envs/hack_lerobot/bin:$PATH"

# There is only one policy, so every argument is a pass-through flag. Taking a
# positional here meant `--steps=8000` was read as the policy name.
DRY=0
ARGS=()
for a in "$@"; do
  [[ "$a" == "--dry-run" ]] && DRY=1 || ARGS+=("$a")
done

WHICH=smolvla
ROOT="datasets/stage2_sort"
OUT="experiments/smolvla_sort"
STEPS=20000
REPO_ID="$("$PY" -c "
import sys; sys.path.insert(0, 'src')
from percept2act.config import Scenario
print(Scenario.load().require('policies.stage2_sort.dataset_repo_id'))")"

DATASET_ROOT="$REPO/$ROOT"
OUTPUT_DIR="$REPO/$OUT"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "No dataset at $DATASET_ROOT"
  echo "Record it first:  bash scripts/06_record.sh coral 5 && bash scripts/06_record.sh blue 5"
  exit 1
fi

# Fail early on a one-sided stage-2 dataset. A SmolVLA fine-tune on a single
# instruction trains happily and then ignores the instruction at inference --
# which looks exactly like "the VLA does not work" during the demo.
if [[ "$WHICH" == "smolvla" ]]; then
  bash "$REPO/scripts/07_verify_dataset.sh" || true
  echo
fi

# Drop our default --steps if the caller supplied their own, or lerobot-train
# sees the flag twice.
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  [[ "$a" == --steps=* ]] && STEPS=""
done

CMD=(lerobot-train
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --policy.type="$WHICH"
  --output_dir="$OUTPUT_DIR"
  ${STEPS:+--steps=$STEPS}
  --save_freq=2000
  --log_freq=100
  --policy.device=xpu          # Arc iGPU; use cpu if xpu errors out
  --wandb.enable=false
  # Defaults to true, which makes lerobot-train demand a Hub repo id before it
  # will start. We only ever want a local checkpoint.
  --policy.push_to_hub=false
)
# SmolVLA must start from the pretrained base, already in the HF cache.
[[ "$WHICH" == "smolvla" ]] && CMD+=(--policy.pretrained_path=lerobot/smolvla_base)
CMD+=("${ARGS[@]+"${ARGS[@]}"}")

echo "=== training $WHICH ==="
echo "  dataset : $REPO_ID  ($DATASET_ROOT)"
echo "  output  : $OUTPUT_DIR"
printf '  command : '; printf '%q ' "${CMD[@]}"; echo
echo

if [[ -d "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR"
  echo "  resume it:  bash scripts/14_train_vla.sh --resume=true"
  echo "  or start over:  rm -rf $OUTPUT_DIR"
  exit 1
fi

if [[ "$DRY" == 1 ]]; then
  echo "(dry run -- nothing executed)"
  echo "If a flag is rejected, check the exact names with:  lerobot-train --help"
  exit 0
fi

# Deliberately do NOT create OUTPUT_DIR: lerobot-train refuses to start if it
# already exists and --resume is false, so creating it here guaranteed a
# FileExistsError on the very first run.
"${CMD[@]}"

echo
echo "Done. Checkpoints in $OUTPUT_DIR"
echo "Next: export to OpenVINO ->  bash scripts/08_export_policy.sh $WHICH"
