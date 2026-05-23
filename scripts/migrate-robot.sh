#!/usr/bin/env bash
# One-time migration: prepare the robot dock for the new src/ tests/ data/
# layout. After this, run scripts/push.sh whenever you want to deploy code.
#
# Idempotent — safe to re-run.
#
# What this does on the robot:
#   1. Moves /home/unitree/Development/pella/data/face_ids -> pella_app/data/face_ids
#      (if not already done) so the live enrollment data follows the
#      new in-repo location.
#   2. Deletes the OLD flat-layout .py files at pella_app/ root that have
#      now moved into src/ or tests/ — so the next rsync from the laptop
#      doesn't leave both old and new copies side by side.
#   3. Removes old binary dirs (audio/, models/) that have moved under
#      data/ or tests/.
#   4. Symlinks pella-camera.service into /etc/systemd/system/ so future
#      pushes automatically pick up service-file changes without a manual
#      sudo cp.
#
# Usage:
#   PELLA_ROBOT_HOST=unitree@<dock-ip> scripts/migrate-robot.sh
#
# Optional env vars (defaults match the dev/robot convention):
#   PELLA_ROBOT_PATH        /home/unitree/Development/pella/pella_app
#   PELLA_ROBOT_OLD_DATA    /home/unitree/Development/pella/data

set -euo pipefail

ROBOT_HOST="${PELLA_ROBOT_HOST:?Set PELLA_ROBOT_HOST=user@dock-ip}"
ROBOT_PATH="${PELLA_ROBOT_PATH:-/home/unitree/Development/pella/pella_app}"
OLD_DATA_DIR="${PELLA_ROBOT_OLD_DATA:-/home/unitree/Development/pella/data}"

echo "Migrating ${ROBOT_HOST}:${ROBOT_PATH}"
echo "  old data sibling: ${OLD_DATA_DIR}"
echo ""

ssh "${ROBOT_HOST}" \
    "ROBOT_PATH='${ROBOT_PATH}' OLD_DATA_DIR='${OLD_DATA_DIR}' bash -s" \
    <<'REMOTE'
set -euo pipefail
cd "$ROBOT_PATH"

mkdir -p data tests

# 1. Move sibling data/face_ids -> pella_app/data/face_ids
if [ -d "$OLD_DATA_DIR/face_ids" ] && [ ! -d "data/face_ids" ]; then
    mv "$OLD_DATA_DIR/face_ids" data/face_ids
    echo "  Moved $OLD_DATA_DIR/face_ids -> data/face_ids"
elif [ -d "data/face_ids" ]; then
    echo "  data/face_ids already in place"
else
    echo "  WARNING: no face_ids found at $OLD_DATA_DIR or data/"
fi

# 2. Old flat-layout production .py files — same files now live under src/
for f in pella_main.py stt.py tts.py vision.py recog_greeting.py \
         actions.py chat.py face_recognizer.py front_camera.py \
         task_manager.py front_camera_display.py \
         front_camera_display.py.bak; do
    [ -f "$f" ] && rm "$f" && echo "  Removed old $f"
done

# 3. Loose test / exploratory scripts (now under tests/)
for f in test_actions.py test_mic.py test_chat_go.py \
         test_assistant_recorder.py test_enable_vui.py test_gpt_feedback.py \
         test_start_chatgo.py test_vui_start_voice.py \
         say.py say_webrtc.py listen_webrtc.py generate_greetings.py; do
    [ -f "$f" ] && rm "$f" && echo "  Removed loose $f"
done

# 4. Old binary dirs at root
[ -d "audio" ] && rm -rf audio && echo "  Removed audio/"
[ -d "models" ] && rm -rf models && echo "  Removed models/"

# 5. Symlink the systemd unit. After this, editing pella-camera.service
#    in the repo + a push.sh + daemon-reload is enough — no sudo cp.
SVC_TARGET="/etc/systemd/system/pella-camera.service"
SVC_SRC="$ROBOT_PATH/pella-camera.service"
if [ -L "$SVC_TARGET" ] && [ "$(readlink "$SVC_TARGET")" = "$SVC_SRC" ]; then
    echo "  pella-camera.service already symlinked"
elif [ -f "$SVC_SRC" ]; then
    sudo rm -f "$SVC_TARGET"
    sudo ln -s "$SVC_SRC" "$SVC_TARGET"
    sudo systemctl daemon-reload
    echo "  Symlinked $SVC_SRC -> $SVC_TARGET"
else
    echo "  NOTE: $SVC_SRC not present yet — first push will install it."
    echo "        Re-run this script after the first push to symlink."
fi

echo ""
echo "Robot pella_app/ now contains:"
ls -F
REMOTE

echo ""
echo "Migration complete. Next: scripts/push.sh"
