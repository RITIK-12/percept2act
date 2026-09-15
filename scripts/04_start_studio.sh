#!/usr/bin/env bash
# Start Physical AI Studio: backend API + web UI.
#
# Recording through the Studio (rather than the bare lerobot CLI) is what the
# 20-point "VLA & Physical AI Studio Integration" criterion rewards, and the UI
# gives you live camera feeds while you teleoperate.
#
# Usage:
#   bash scripts/04_start_studio.sh backend   # terminal 1
#   bash scripts/04_start_studio.sh ui        # terminal 2
#   bash scripts/04_start_studio.sh check     # is it up?
#
# Then open http://localhost:5173
#
# Leave both running. Stop with Ctrl-C in each terminal.
set -euo pipefail

STUDIO="$HOME/physical-ai-studio/application"
BACKEND="$STUDIO/backend"
UI="$STUDIO/ui"
export PATH="$HOME/.local/bin:$PATH"

case "${1:-backend}" in

backend)
  cd "$BACKEND"
  # The backend reads .env; seed it from the example on first run.
  [[ -f .env ]] || { cp .env.example .env && echo "seeded .env from .env.example"; }
  echo "Starting Physical AI Studio backend (http://localhost:8000)"
  echo "  API docs once up: http://localhost:8000/docs"
  echo
  export PYTHONUNBUFFERED=1
  exec ./run.sh serve
  ;;

ui)
  cd "$UI"
  # Node 24 is installed via nvm and is not on PATH in a fresh shell.
  if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.nvm/nvm.sh"
    nvm use 24 >/dev/null 2>&1 || nvm use --lts >/dev/null 2>&1 || true
  fi
  command -v node >/dev/null || { echo "node not found; run: source ~/.nvm/nvm.sh && nvm use 24"; exit 1; }
  echo "node $(node --version)"
  [[ -d node_modules ]] || { echo "installing UI deps (one time, ~5-10 min)..."; npm install; }
  echo "Starting Physical AI Studio UI (http://localhost:5173)"
  exec npm run dev
  ;;

check)
  echo -n "backend http://localhost:8000/docs : "
  curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 http://127.0.0.1:8000/docs \
    || echo "not responding"
  echo -n "ui      http://localhost:5173       : "
  curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 http://127.0.0.1:5173 \
    || echo "not responding"
  echo
  echo "listening ports:"
  ss -ltn 2>/dev/null | grep -E ":(8000|5173)" || echo "  none of 8000/5173 are listening"
  ;;

*)
  echo "usage: $0 {backend|ui|check}"; exit 2
  ;;
esac
