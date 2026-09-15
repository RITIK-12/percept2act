#!/usr/bin/env bash
# Fine-tune the sorting policy on the recorded dataset.
#
# SmolVLA is FINE-TUNED from lerobot/smolvla_base, never trained from scratch --
# the base checkpoint is what already understands language, and our ~60 episodes
# only teach it this specific pick-and-place.
#
# Usage:
#   bash scripts/14_train_vla.sh                      # fine-tune the sorting policy
#   bash scripts/14_train_vla.sh smolvla --dry-run    # print the command, run nothing
#
# Extra flags pass straight through, e.g.:
#   bash scripts/14_train_vla.sh smolvla --steps=30000 --batch_size=32
#
# Takes 45-120 min on the Arc iGPU. Run it in its own terminal and get on with
# the detector while it trains -- the two are independent.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
export PATH="$HOME/miniforge3/envs/hack_lerobot/bin:$PATH"

WHICH="${1:-smolvla}"
shift || true

DRY=0
ARGS=()
for a in "$@"; do
  [[ "$a" == "--dry-run" ]] && DRY=1 || ARGS+=("$a")
done

read -r REPO_ID ROOT OUT STEPS <<<"$("$PY" - "$WHICH" <<'PY'
import sys
sys.path.insert(0, "src")
from percept2act.config import Scenario

which = sys.argv[1]
s = Scenario.load()
if which == "smolvla":
    key, steps = "policies.stage2_sort", 20000
    root, out = "datasets/stage2_sort", "experiments/smolvla_sort"
else:
    sys.exit(f"unknown policy {which!r}")
print(s.require(f"{key}.dataset_repo_id"), root, out, steps)
PY
)"

DATASET_ROOT="$REPO/$ROOT"
OUTPUT_DIR="$REPO/$OUT"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "No dataset at $DATASET_ROOT"
  echo "Record it first:  bash scripts/06_record.sh good && bash scripts/06_record.sh defective"
  exit 1
fi

# Fail early on a one-sided stage-2 dataset. A SmolVLA fine-tune on a single
# instruction trains happily and then ignores the instruction at inference --
# which looks exactly like "the VLA does not work" during the demo.
if [[ "$WHICH" == "smolvla" ]]; then
  bash "$REPO/scripts/07_verify_dataset.sh" || true
  echo
fi

CMD=(lerobot-train
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --policy.type="$WHICH"
  --output_dir="$OUTPUT_DIR"
  --steps="$STEPS"
  --save_freq=2000
  --log_freq=100
  --policy.device=xpu          # Arc iGPU; use cpu if xpu errors out
  --wandb.enable=false
)
# SmolVLA must start from the pretrained base, already in the HF cache.
[[ "$WHICH" == "smolvla" ]] && CMD+=(--policy.pretrained_path=lerobot/smolvla_base)
CMD+=("${ARGS[@]+"${ARGS[@]}"}")

echo "=== training $WHICH ==="
echo "  dataset : $REPO_ID  ($DATASET_ROOT)"
echo "  output  : $OUTPUT_DIR"
printf '  command : '; printf '%q ' "${CMD[@]}"; echo
echo

if [[ "$DRY" == 1 ]]; then
  echo "(dry run -- nothing executed)"
  echo "If a flag is rejected, check the exact names with:  lerobot-train --help"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${CMD[@]}"

echo
echo "Done. Checkpoints in $OUTPUT_DIR"
echo "Next: export to OpenVINO ->  bash scripts/08_export_policy.sh $WHICH"
