#!/bin/bash
#
# Service management functions for Machine Vision Flow.
#

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Python backend ------------------------------------------------------------

ensure_python_backend_env() {
    local force="${1:-false}"

    mkdir -p "$BACKEND_DIR"

    require_command python3 "Install Python 3 to run the backend."

    if [ ! -d "$BACKEND_VENV_DIR" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$BACKEND_VENV_DIR"
        force=true
    fi

    if [ "$force" = true ] || [ ! -f "$BACKEND_SENTINEL" ]; then
        log_info "Installing Python backend dependencies..."
        "$BACKEND_VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
        touch "$BACKEND_SENTINEL"
    fi
}

start_python_backend() {
    ensure_python_backend_env "${1:-false}"

    # Ensure runtime directories exist
    mkdir -p "$LOG_DIR" "$RUN_DIR"

    pushd "$BACKEND_DIR" >/dev/null || return 1
    nohup "$BACKEND_VENV_DIR/bin/python" main.py > "$BACKEND_LOG_FILE" 2>&1 &
    local pid=$!
    popd >/dev/null || true

    echo "$pid" > "$BACKEND_PID_FILE"
}

stop_python_backend() {
    local pid="${1:-}"

    if [ -z "$pid" ] && [ -f "$BACKEND_PID_FILE" ]; then
        pid="$(cat "$BACKEND_PID_FILE")"
    fi

    if [ -n "$pid" ]; then
        terminate_pid "$pid" "Python backend" 5
    else
        local matches
        matches=$(pgrep -f "python[0-9.]* .*main.py" || true)
        if [ -n "$matches" ]; then
            for match in $matches; do
                terminate_pid "$match" "Python backend" 5
            done
        fi
    fi

    rm -f "$BACKEND_PID_FILE"
}
