#!/usr/bin/env bash
# Export the trained SmolVLA policy to OpenVINO IR.
#
# This is the step that moves the policy off torch-xpu and onto OpenVINO, so
# both models in the loop -- PatchCore and SmolVLA -- run on OpenVINO runtime.
#
# SmolVLA's OpenVINO export includes the tokenizer (`export_tokenizer=True`,
# an `ov_tokenizer` component), so the instruction string is tokenized by
# OpenVINO too. The Anomalib -> instruction -> policy seam runs end to end on
# OpenVINO, which is exactly what the optimization criterion asks for.
#
# Requires a LIGHTNING checkpoint from scripts/16_studio_train.sh. A
# lerobot-train checkpoint (scripts/14) is a different format and will not load
# here -- see the note in 16_studio_train.sh.
#
# Usage:
#   bash scripts/17_export_policy.sh                     # newest checkpoint
#   bash scripts/17_export_policy.sh path/to/model.ckpt  # a specific one
#   bash scripts/17_export_policy.sh --backend onnx      # onnx | openvino | torch | executorch
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_BIN="$HOME/physical-ai-studio/application/backend/.venv/bin/physicalai"

TRAIN_DIR="$REPO/experiments/smolvla_studio"
BACKEND=openvino
CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --backend=*) BACKEND="${1#*=}"; shift ;;
    *) CKPT="$1"; shift ;;
  esac
done

[[ -x "$STUDIO_BIN" ]] || {
  echo "physicalai CLI not found at $STUDIO_BIN"
  exit 1
}

# Newest checkpoint by mtime, so this keeps working across resumed runs.
if [[ -z "$CKPT" ]]; then
  CKPT="$(find "$TRAIN_DIR" -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d' ' -f2- || true)"
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "No Lightning checkpoint found under $TRAIN_DIR"
  echo
  echo "  train one:  bash scripts/16_studio_train.sh"
  echo
  echo "  If you trained with scripts/14_train_vla.sh instead, that writes a"
  echo "  LeRobot-format checkpoint which this exporter cannot load. Either"
  echo "  retrain via 16, or keep running the policy in torch with:"
  echo "      python scripts/12_demo.py --executor policy"
  exit 1
fi

OUT="$REPO/experiments/smolvla_sort_ov"

echo "=== exporting smolvla -> $BACKEND ==="
echo "  checkpoint : $CKPT"
echo "  output     : $OUT"
echo

"$STUDIO_BIN" export \
  --policy physicalai.policies.SmolVLA \
  --ckpt_path "$CKPT" \
  --backend "$BACKEND" \
  --output_dir "$OUT"

echo
echo "Artifacts:"
ls -la "$OUT" 2>/dev/null || true
echo
echo "Point the policy executor at it by setting in config/scenario.yaml:"
echo "  policies.stage2_sort.export_dir: experiments/smolvla_sort_ov"
