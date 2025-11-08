#!/bin/bash
#
# Machine Vision Flow - Stop Script
# Stops Python backend service.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/services.sh"

print_banner "Machine Vision Flow Backend - Shutdown" "$RED"

if [ -f "$BACKEND_PID_FILE" ] || check_port 8000; then
    log_info "Stopping Python backend..."
    stop_python_backend
    log_success "✓ Python backend stopped"
else
    log_warn "Python backend not running"
fi

echo
print_banner "Backend stopped" "$GREEN"
