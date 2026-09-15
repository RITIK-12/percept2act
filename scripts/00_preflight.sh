#!/usr/bin/env bash
# Challenge Day preflight. Run this in every new shell before touching the rig.
#
# Verifies the things that silently break a demo: NPU visibility, serial
# permissions, camera presence, and the wrist camera's exposure (its v4l2
# controls reset on every replug).
set -uo pipefail

# Refuse to run under sudo. sudo resets $HOME to /root, so every path below
# ($HOME/miniforge3, the lerobot calibration cache, the Studio venv) silently
# resolves to somewhere that does not exist -- and the report comes back as a
# wall of FAILs that look like broken hardware. Nothing here needs root.
if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this under sudo."
  echo
  echo "  sudo resets \$HOME to /root, so the conda envs, the arm calibration"
  echo "  cache and the Studio venv all resolve to paths that do not exist."
  echo "  Every check then fails for the wrong reason."
  echo
  echo "Run it as yourself:"
  echo "  bash scripts/00_preflight.sh"
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_PY="$HOME/miniforge3/envs/hack_lerobot/bin/python"
STUDIO_PY="$HOME/physical-ai-studio/application/backend/.venv/bin/python"

pass=0; fail=0; warn=0
ok()   { echo "  [PASS] $*"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail+1)); }
note() { echo "  [WARN] $*"; warn=$((warn+1)); }

echo "=== 1. compute ==="
[[ -e /dev/accel/accel0 ]]     && ok "NPU node /dev/accel/accel0"   || bad "NPU node missing"
[[ -e /dev/dri/renderD128 ]]   && ok "iGPU node /dev/dri/renderD128" || bad "iGPU node missing"
devs=$("$RUNTIME_PY" -c "import openvino as ov;print(','.join(ov.Core().available_devices))" 2>/dev/null)
if [[ "$devs" == *NPU* && "$devs" == *GPU* ]]; then ok "OpenVINO sees [$devs]"
else bad "OpenVINO devices = [${devs:-none}] — expected CPU,GPU,NPU"; fi

echo "=== 2. environments ==="
"$RUNTIME_PY" -c "import lerobot,anomalib,cv2,pyrealsense2" 2>/dev/null \
  && ok "runtime env (hack_lerobot): lerobot+anomalib+cv2+pyrealsense2" \
  || bad "runtime env broken"
"$STUDIO_PY" -c "import physicalai" 2>/dev/null \
  && ok "studio env: physicalai imports" \
  || bad "studio env: physicalai import FAILS — run 2_install_software.sh 5"

echo "=== 3. arms ==="
for arm in follower leader; do
  port=$("$RUNTIME_PY" "$REPO/src/percept2act/lerobot_args.py" "$arm-port" 2>/dev/null)
  if [[ -z "$port" ]]; then
    bad "$arm port: could not read arms.$arm.port from config (is the runtime env OK?)"
  elif [[ -e "$port" ]]; then
    if [[ -r "$port" && -w "$port" ]]; then ok "$arm port readable/writable ($(basename "$port"))"
    else bad "$arm port exists but no access — sudo usermod -aG dialout $USER, then re-login"; fi
  else bad "$arm port missing: $port"; fi
done
id -nG | grep -qw dialout && ok "user in dialout group" \
  || note "user NOT in dialout — ports work only while chmod'ed; udev resets them on replug"

echo "=== 4. calibration ==="
CAL="$HOME/.cache/huggingface/lerobot/calibration"
[[ -f "$CAL/robots/so_follower/hack_follower.json" ]] && ok "follower calibration present" || bad "follower calibration MISSING"
[[ -f "$CAL/teleoperators/so_leader/hack_leader.json" ]] && ok "leader calibration present" || bad "leader calibration MISSING"

echo "=== 5. cameras ==="
n_rs=$("$RUNTIME_PY" -c "import pyrealsense2 as rs;print(len(rs.context().query_devices()))" 2>/dev/null || echo 0)
[[ "$n_rs" -ge 2 ]] && ok "$n_rs RealSense cameras" || bad "expected 2 RealSense, found $n_rs"
if "$RUNTIME_PY" -c "
import sys; sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
s=Scenario.load()
for r in ('overhead','inspect'): s.require(f'cameras.{r}.serial')
" 2>/dev/null; then ok "camera roles assigned in scenario.yaml"
else bad "camera roles unassigned — run scripts/01_assign_cameras.py --preview"; fi

# The Sonix wrist module loses its exposure settings on replug and defaults to
# blowing out. Re-apply and check we are not saturating.
WRIST=$(readlink -f "$("$RUNTIME_PY" -c "
import sys; sys.path.insert(0,'$REPO/src')
from percept2act.config import Scenario
print(Scenario.load().require('cameras.wrist.index_or_path'))" 2>/dev/null)")
if [[ -z "$WRIST" ]]; then
  note "wrist camera: could not read cameras.wrist.index_or_path from config"
elif [[ -e "$WRIST" ]]; then
  v4l2-ctl -d "$WRIST" --set-ctrl=auto_exposure=1            2>/dev/null
  v4l2-ctl -d "$WRIST" --set-ctrl=exposure_time_absolute=60  2>/dev/null
  read -r mean blown < <("$RUNTIME_PY" -c "
import cv2
cap=cv2.VideoCapture('$WRIST'); best=None
for _ in range(25):
    ok,f=cap.read()
    if ok and f is not None: best=f
cap.release()
print(f'{best.mean():.0f} {(best>250).mean()*100:.1f}' if best is not None else '0 0')" 2>/dev/null)
  if (( $(echo "$mean > 20" | bc -l) )); then
    ok "wrist cam exposing (mean=$mean, blown=${blown}%)"
    (( $(echo "$blown > 10" | bc -l) )) && note "wrist cam ${blown}% saturated — lower exposure_time_absolute"
  else
    bad "wrist cam is black (mean=$mean) — LENS CAP ON, or wrong node"
  fi
else note "wrist camera path not found: $WRIST"; fi

echo
echo "=== PASS $pass · FAIL $fail · WARN $warn ==="
[[ "$fail" -eq 0 ]] && echo "Stack ready." || echo "Fix the FAILs before recording."
exit $(( fail > 0 ))
