#!/usr/bin/env bash
# Push laptop -> robot. Run after any code change.
#
# Sends the pella_app/ tree to the robot dock and restarts the
# pella-camera systemd service. Preserves:
#   .venv/                      host-specific Python env (Jetson CUDA build)
#   .git/                       git metadata
#   .env                        secrets
#   __pycache__/                bytecode cache
#   data/face_ids/              live enrollment data (canonical on robot)
#   data/models/w600k_r50.onnx  170 MB ArcFace model (only on robot)
#
# Usage:
#   PELLA_ROBOT_HOST=unitree@<dock-ip> scripts/push.sh
#
# Optional env vars:
#   PELLA_ROBOT_PATH     /home/unitree/Development/pella/pella_app
#   PELLA_SKIP_RESTART   if set non-empty, skips systemctl restart

set -euo pipefail

ROBOT_HOST="${PELLA_ROBOT_HOST:?Set PELLA_ROBOT_HOST=user@dock-ip}"
ROBOT_PATH="${PELLA_ROBOT_PATH:-/home/unitree/Development/pella/pella_app}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing"
echo "  from: ${REPO_DIR}/"
echo "  to:   ${ROBOT_HOST}:${ROBOT_PATH}/"
echo ""

rsync -av --checksum --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/face_ids/' \
    --exclude='data/models/w600k_r50.onnx' \
    "${REPO_DIR}/" "${ROBOT_HOST}:${ROBOT_PATH}/"

if [ -n "${PELLA_SKIP_RESTART:-}" ]; then
    echo ""
    echo "PELLA_SKIP_RESTART set — leaving service running unchanged."
    exit 0
fi

echo ""
echo "Restarting pella-camera on robot..."
ssh "${ROBOT_HOST}" \
    "sudo systemctl daemon-reload && sudo systemctl restart pella-camera"

echo ""
echo "Done. Watch the log:"
echo "  ssh ${ROBOT_HOST} 'journalctl -u pella-camera -f'"
