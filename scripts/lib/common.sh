#!/bin/bash
#
# Shared helpers for Machine Vision Flow management scripts.
# Provides reusable paths, colors, and utility functions.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#

# Resolve key directories relative to this library.
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$LIB_DIR")"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"

# Project paths
BACKEND_DIR="$PROJECT_ROOT/python-backend"
BACKEND_VENV_DIR="$BACKEND_DIR/venv"
BACKEND_SENTINEL="$BACKEND_DIR/.deps_installed"

# Runtime directories (fallback to project var/ for development)
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/var/log}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/var/run}"

# Log and PID files
BACKEND_LOG_FILE="$LOG_DIR/backend.log"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"

# Port configuration
PORT_BACKEND="${PORT_BACKEND:-8000}"

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Logging helpers -----------------------------------------------------------

log_info() {
    echo -e "${BLUE}$*${NC}"
}

log_success() {
    echo -e "${GREEN}$*${NC}"
}

log_warn() {
    echo -e "${YELLOW}$*${NC}"
}

log_error() {
    echo -e "${RED}$*${NC}" >&2
}

print_banner() {
    local message="$1"
    local color="${2:-$BLUE}"
    local padding="${3:-2}"
    local len=${#message}
    local width=$((len + padding * 2))
    local border

    border=$(printf '═%.0s' $(seq 1 "$width"))
    printf "%b╔%s╗%b\n" "$color" "$border" "$NC"
    printf "%b║%*s%s%*s║%b\n" "$color" "$padding" "" "$message" "$padding" "" "$NC"
    printf "%b╚%s╝%b\n\n" "$color" "$border" "$NC"
}

# Generic helpers -----------------------------------------------------------

require_command() {
    local cmd="$1"
    local hint="$2"

    if ! command -v "$cmd" >/dev/null 2>&1; then
        if [ -n "$hint" ]; then
            log_error "✗ Required command '$cmd' not found. $hint"
        else
            log_error "✗ Required command '$cmd' not found."
        fi
        return 1
    fi
}

check_port() {
    local port="$1"
    if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

wait_for_port() {
    local port="$1"
    local service="$2"
    local max_wait="${3:-30}"
    local waited=0

    printf "Waiting for %s to start on port %s" "$service" "$port"
    while ! check_port "$port" && [ "$waited" -lt "$max_wait" ]; do
        printf "."
        sleep 1
        waited=$((waited + 1))
    done

    if [ "$waited" -ge "$max_wait" ]; then
        printf " %bTIMEOUT%b\n" "$RED" "$NC"
        return 1
    fi

    printf " %bOK%b\n" "$GREEN" "$NC"
    return 0
}

terminate_pid() {
    local pid="$1"
    local label="${2:-process}"
    local timeout="${3:-5}"

    if [ -z "$pid" ]; then
        return 0
    fi

    if ! ps -p "$pid" >/dev/null 2>&1; then
        return 0
    fi

    kill "$pid" 2>/dev/null || true

    local waited=0
    while ps -p "$pid" >/dev/null 2>&1 && [ "$waited" -lt "$timeout" ]; do
        sleep 1
        waited=$((waited + 1))
    done

    if ps -p "$pid" >/dev/null 2>&1; then
        log_warn "Force stopping $label (PID: $pid)"
        kill -9 "$pid" 2>/dev/null || true
    fi

    return 0
}
