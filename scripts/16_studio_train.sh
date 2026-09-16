#!/usr/bin/env bash
# Fine-tune SmolVLA through Intel Physical AI Studio (Lightning).
#
# This is the alternative to scripts/14_train_vla.sh, and the difference that
# matters is the checkpoint format:
#
#   14 (lerobot-train)  -> pretrained_model/model.safetensors   -- runs in torch
#   16 (physicalai fit) -> a Lightning .ckpt                    -- ALSO exports
#                                                                  to OpenVINO
#
# `physicalai export` loads through Lightning's load_from_checkpoint, so only
# the second one can be handed to scripts/17_export_policy.sh. If you want the
# policy running on OpenVINO rather than torch-xpu, train here.
#
# Usage:
#   bash scripts/16_studio_train.sh                          # defaults from the config
#   bash scripts/16_studio_train.sh --trainer.max_steps=2000 # shorter run
#   bash scripts/16_studio_train.sh --dry-run                # resolve config, run nothing
#
# Any other flag passes through to `physicalai fit`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
STUDIO_BIN="$HOME/physical-ai-studio/application/backend/.venv/bin/physicalai"

CONFIG="$REPO/config/physicalai_smolvla.yaml"
DATASET_ROOT="$REPO/datasets/stage2_sort"
OUTPUT_DIR="$REPO/experiments/smolvla_studio"

DRY=0
ARGS=()
for a in "$@"; do
  [[ "$a" == "--dry-run" ]] && DRY=1 || ARGS+=("$a")
done

[[ -x "$STUDIO_BIN" ]] || {
  echo "physicalai CLI not found at $STUDIO_BIN"
  echo "  reinstall the Studio backend:  ~/hackathon_install/2_install_software.sh 5"
  exit 1
}

[[ -d "$DATASET_ROOT" ]] || {
  echo "No dataset at $DATASET_ROOT"
  echo "  record it first:  bash scripts/15_studio_record.sh"
  exit 1
}

# Two trainers reading the same LeRobot dataset is fine; two writing the same
# output tree is not, and a second run also contends for the one iGPU.
if [[ "$DRY" == 0 ]] && pgrep -f "lerobot-train|physicalai fit" >/dev/null 2>&1; then
  echo "!! Another training run is already going:"
  pgrep -fa "lerobot-train|physicalai fit" | sed 's/^/     /' | cut -c1-160
  echo
  echo "   There is one iGPU. Let it finish first."
  exit 1
fi

# Same guard as 14: a one-sided dataset trains happily and then ignores the
# instruction at inference, which looks exactly like "the VLA does not work".
bash "$REPO/scripts/07_verify_dataset.sh" || true
echo

CMD=("$STUDIO_BIN" fit
  --config "$CONFIG"
  # Pinned absolutely so the run does not depend on how jsonargparse resolves
  # a relative path -- against cwd or against the config file.
  --data.init_args.root "$DATASET_ROOT"
  --trainer.default_root_dir "$OUTPUT_DIR"
)
CMD+=("${ARGS[@]+"${ARGS[@]}"}")

echo "=== training smolvla via Physical AI Studio ==="
echo "  config  : $CONFIG"
echo "  dataset : $DATASET_ROOT"
echo "  output  : $OUTPUT_DIR"
printf '  command : '; printf '%q ' "${CMD[@]}"; echo
echo

if [[ "$DRY" == 1 ]]; then
  echo "--- resolved config ---"
  "${CMD[@]}" --print_config
  exit 0
fi

"${CMD[@]}"

echo
echo "Done. Lightning checkpoints under $OUTPUT_DIR"
echo "Next: export to OpenVINO ->  bash scripts/17_export_policy.sh"
