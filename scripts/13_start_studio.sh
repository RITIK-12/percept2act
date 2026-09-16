#!/usr/bin/env bash
# Start Physical AI Studio: backend API + web UI.
#
# Recording through the Studio (rather than the bare lerobot CLI) is what the
# 20-point "VLA & Physical AI Studio Integration" criterion rewards, and the UI
# gives you live camera feeds while you teleoperate.
#
# Usage:
#   bash scripts/13_start_studio.sh backend   # terminal 1
#   bash scripts/13_start_studio.sh ui        # terminal 2
#   bash scripts/13_start_studio.sh check     # is it up?
#
# Then open http://localhost:3000
#
# Ports come from the Studio install, not from us: the backend's .env.example
# sets PORT=7860, and the UI is an rsbuild dev server on its default 3000 which
# proxies /api to 7860.
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
  echo "Starting Physical AI Studio backend (http://localhost:7860)"
  echo "  API docs once up: http://localhost:7860/docs"
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
  echo "Starting Physical AI Studio UI (http://localhost:3000)"
  # The script is `start` (rsbuild dev); there is no `dev` script in this UI.
  exec npm run start
  ;;

check)
  echo -n "backend http://localhost:7860/docs : "
  curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 http://127.0.0.1:7860/docs \
    2>/dev/null || echo "not responding"
  echo -n "ui      http://localhost:3000       : "
  curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 http://127.0.0.1:3000 \
    2>/dev/null || echo "not responding"
  echo
  echo "listening ports:"
  ss -ltn 2>/dev/null | grep -E ":(7860|3000)" || echo "  neither 7860 nor 3000 is listening"
  ;;

*)
  echo "usage: $0 {backend|ui|check}"; exit 2
  ;;
esac
