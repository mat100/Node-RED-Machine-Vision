#!/bin/bash
#
# Machine Vision Flow - Start Script
# Starts the Python backend service.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/services.sh"

BACKEND_STARTED=false
BACKEND_PID=""
FOLLOW_LOGS=false
FORCE_DEPS=false

usage() {
    cat <<'EOF'
Usage: start.sh [options]

Options:
  --follow, -f     Follow service logs after startup
  --force-deps     Reinstall runtime dependencies before starting
  --help           Show this help and exit
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --follow|-f)
            FOLLOW_LOGS=true
            ;;
        --force-deps)
            FORCE_DEPS=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_warn "Ignoring unknown option: $1"
            ;;
    esac
    shift
done

cleanup() {
    set +e
    echo
    log_warn "Caught interrupt signal, shutting down service..."

    if [ "$FOLLOW_LOGS" = true ] && [ -n "${TAIL_PID:-}" ]; then
        kill "$TAIL_PID" 2>/dev/null || true
    fi

    if [ "$BACKEND_STARTED" = true ]; then
        log_info "Stopping Python backend..."
        stop_python_backend "$BACKEND_PID"
        log_success "Python backend stopped"
    fi

    exit 0
}
trap cleanup INT TERM

print_banner "Machine Vision Flow Backend - Startup" "$GREEN"

require_command python3 "Install Python 3 to run the backend."

echo

if check_port 8000; then
    log_warn "⚠ Python backend already running on port 8000"
else
    log_info "Starting Python backend..."
    start_python_backend "$FORCE_DEPS"
    BACKEND_PID="$(cat "$BACKEND_PID_FILE")"
    BACKEND_STARTED=true
    log_success "✓ Python backend PID: $BACKEND_PID"
    wait_for_port 8000 "Python backend"
fi

echo
print_banner "Backend is ready!" "$GREEN"
echo -e "Python Backend: ${GREEN}http://localhost:8000${NC}"
echo -e "API Docs:       ${GREEN}http://localhost:8000/docs${NC}"
echo
echo -e "${YELLOW}To stop the service, run:${NC} ./stop.sh"

if [ "$FOLLOW_LOGS" = true ]; then
    echo
    log_info "Following logs (Ctrl+C to exit)..."
    touch "$BACKEND_LOG_FILE"
    tail -f "$BACKEND_LOG_FILE" &
    TAIL_PID=$!
    wait "$TAIL_PID"
fi
