#!/usr/bin/env bash
# Rebuild the dashboard image from this checkout and restart the local container on
# http://127.0.0.1:8501 (loopback only). The plan store and the login stay where they are:
#   ~/.local/share/riskrank/riskrank.db        plan store (mounted at /data)
#   ~/.config/riskrank/dashboard-auth.json     login made by `python -m riskrank.auth`
# Run from anywhere:  bash deploy/restart-dashboard.sh
set -euo pipefail
cd "$(dirname "$0")/.."
auth="$HOME/.config/riskrank/dashboard-auth.json"
[ -f "$auth" ] || { echo "No login at $auth: run python -m riskrank.auth $auth first." >&2; exit 1; }
mkdir -p "$HOME/.local/share/riskrank"
docker build -q -f deploy/Dockerfile.dashboard -t riskrank-dashboard . >/dev/null
docker rm -f rr-dashboard >/dev/null 2>&1 || true
docker run -d --name rr-dashboard -p 127.0.0.1:8501:8501 \
  -v "$HOME/.local/share/riskrank:/data" \
  -v "$auth:/run/secrets/dashboard_auth:ro" \
  riskrank-dashboard >/dev/null
for _ in $(seq 1 30); do
  if curl -fs http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "Dashboard running from commit $(git rev-parse --short HEAD): http://127.0.0.1:8501"
    exit 0
  fi
  sleep 1
done
echo "The container started but the dashboard did not answer; see: docker logs rr-dashboard" >&2
exit 1
