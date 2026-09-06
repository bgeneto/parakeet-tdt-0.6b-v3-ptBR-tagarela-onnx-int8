#!/bin/sh
# Ensure the bind-mounted MODEL_DIR is writable, download weights if missing, then drop to stt.
set -eu

MODEL_DIR="${MODEL_DIR:-/opt/models/parakeet}"
HF_HOME="${HF_HOME:-$MODEL_DIR/.hf}"
export HF_HOME
mkdir -p "$MODEL_DIR" "$HF_HOME"

drop_privs() {
    if [ "$(id -u)" = "0" ]; then
        exec setpriv --reuid=stt --regid=stt --init-groups -- "$@"
    fi
    exec "$@"
}

if [ "$(id -u)" = "0" ]; then
    # Compose creates a missing host bind-mount as root; do not chown -R (multi-GB).
    chown stt:stt "$MODEL_DIR" "$HF_HOME"
    setpriv --reuid=stt --regid=stt --init-groups -- \
        python /app/download_model.py --dest "$MODEL_DIR"
else
    python /app/download_model.py --dest "$MODEL_DIR"
fi

drop_privs "$@"
