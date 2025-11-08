#!/bin/bash
#
# File watching utilities for Machine Vision Flow development mode
#

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Check if inotify-tools is installed
check_inotify() {
    if ! command -v inotifywait >/dev/null 2>&1; then
        log_warn "inotify-tools not installed. Install with: sudo apt-get install inotify-tools"
        return 1
    fi
    return 0
}

# Check if entr is installed (alternative watcher)
check_entr() {
    if ! command -v entr >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Generic file watcher with debouncing
# Usage: watch_files <paths> <extensions> <callback> [debounce_seconds]
watch_files() {
    local paths="$1"
    local extensions="$2"
    local callback="$3"
    local debounce="${4:-1}"

    if ! check_inotify; then
        log_error "File watching requires inotify-tools"
        return 1
    fi

    local last_run=0
    local now

    log_info "Watching for changes in: $paths"
    log_info "File extensions: $extensions"

    # Build the include pattern for inotify
    local include_pattern=""
    IFS=',' read -ra EXTS <<< "$extensions"
    for ext in "${EXTS[@]}"; do
        include_pattern="$include_pattern --include '.*\.${ext}$'"
    done

    # Watch for file changes
    while true; do
        # Use eval to properly handle the include pattern
        eval "inotifywait -r -e modify,create,delete,move $include_pattern $paths" 2>/dev/null

        # Debouncing logic
        now=$(date +%s)
        if [ $((now - last_run)) -ge "$debounce" ]; then
            last_run=$now
            log_info "Changes detected, executing callback..."
            $callback
        else
            log_debug "Debouncing... skipping rapid change"
        fi
    done
}

# Watch Python files and restart backend
watch_python_backend() {
    local backend_dir="${1:-$BACKEND_DIR}"
    local restart_cmd="${2:-reload_python_backend}"

    log_info "Starting Python backend file watcher..."

    watch_files "$backend_dir" "py,yaml,yml,json" "$restart_cmd" 2
}

# Reload Python backend (graceful)
reload_python_backend() {
    log_info "Reloading Python backend..."

    # If using uvicorn with --reload, it handles this automatically
    # Otherwise, we need to restart the service
    if pgrep -f "uvicorn.*--reload" >/dev/null 2>&1; then
        log_debug "Uvicorn auto-reload is handling the restart"
    else
        # Manual restart
        if [ -f "$BACKEND_PID_FILE" ]; then
            local pid=$(cat "$BACKEND_PID_FILE")
            log_info "Sending HUP signal to Python backend (PID: $pid)"
            kill -HUP "$pid" 2>/dev/null || {
                log_warn "Failed to send HUP, performing full restart..."
                stop_python_backend
                start_python_backend
            }
        else
            log_warn "Backend PID file not found, starting backend..."
            start_python_backend
        fi
    fi

    wait_for_port "$PORT_BACKEND" "Python backend" 10
    log_success "Python backend reloaded"
}

# Watch configuration files
watch_config_files() {
    local config_files="${1:-$BACKEND_DIR/config.yaml,$BACKEND_DIR/config.dev.yaml}"
    local callback="${2:-reload_on_config_change}"

    log_info "Watching configuration files..."

    if check_inotify; then
        while IFS= read -r file; do
            inotifywait -e modify "$file" 2>/dev/null
            log_info "Config file changed: $file"
            $callback "$file"
        done < <(echo "$config_files" | tr ',' '\n')
    else
        log_warn "inotify not available, config watching disabled"
    fi
}

# Callback for config changes
reload_on_config_change() {
    local changed_file="$1"

    log_info "Configuration changed: $changed_file"

    if [[ "$changed_file" == *"python"* ]] || [[ "$changed_file" == *"backend"* ]]; then
        reload_python_backend
    fi
}

# Multi-watcher manager (runs multiple watchers in background)
start_watchers() {
    local watch_python="${1:-true}"
    local watch_config="${2:-true}"

    local pids=()

    if [ "$watch_python" = true ]; then
        watch_python_backend &
        pids+=($!)
        log_info "Started Python watcher (PID: ${pids[-1]})"
    fi

    if [ "$watch_config" = true ]; then
        watch_config_files &
        pids+=($!)
        log_info "Started config watcher (PID: ${pids[-1]})"
    fi

    # Return PIDs for cleanup
    echo "${pids[@]}"
}

# Stop all watchers
stop_watchers() {
    local pids="$1"

    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping watcher (PID: $pid)"
            kill "$pid" 2>/dev/null
        fi
    done
}

# Alternative watcher using entr (if available)
watch_with_entr() {
    local paths="$1"
    local extensions="$2"
    local callback="$3"

    if ! check_entr; then
        log_debug "entr not available, falling back to inotify"
        watch_files "$paths" "$extensions" "$callback"
        return
    fi

    log_info "Using entr for file watching..."

    # Find files and watch with entr
    find "$paths" -name "*.$extensions" | entr -r "$callback"
}

# Export functions for use in other scripts
export -f watch_files
export -f watch_python_backend
export -f reload_python_backend
export -f start_watchers
export -f stop_watchers
