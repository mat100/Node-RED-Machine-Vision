#!/bin/bash
#
# Machine Vision Flow - Status Script
# Shows status of backend service.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

print_banner "Machine Vision Flow Backend - Status" "$BLUE"

echo -e "${BLUE}Python Backend:${NC}"
if check_port 8000; then
    echo -e "  Status:  ${GREEN}● Running${NC}"
    echo -e "  URL:     http://localhost:8000"
    echo -e "  API:     http://localhost:8000/docs"
    if [ -f "$BACKEND_PID_FILE" ]; then
        echo -e "  PID:     $(cat "$BACKEND_PID_FILE")"
    fi
else
    echo -e "  Status:  ${RED}○ Stopped${NC}"
fi

echo
echo -e "${BLUE}Log Files:${NC}"
if [ -f "$BACKEND_LOG_FILE" ]; then
    size=$(du -h "$BACKEND_LOG_FILE" | cut -f1)
    echo -e "  Backend: $BACKEND_LOG_FILE ($size)"
fi

echo
echo -e "${BLUE}System Info:${NC}"
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version 2>/dev/null | awk '{print $2}')
else
    python_version="missing"
fi

echo -e "  Python:  $python_version"

echo
echo -e "${BLUE}Cameras:${NC}"
if check_port 8000; then
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -X POST http://localhost:8000/api/camera/list 2>/dev/null || echo "[]")
        if [ "$response" != "[]" ] && [ -n "$response" ]; then
            if command -v python3 >/dev/null 2>&1; then
                echo "$response" | python3 - <<'PY' 2>/dev/null || echo -e "  ${YELLOW}Unable to parse camera list${NC}"
import json, sys
cameras = json.load(sys.stdin)
for cam in cameras:
    status = '✓' if cam.get('connected') else ' '
    print(f"  [{status}] {cam.get('id')}: {cam.get('name')} ({cam.get('type')})")
PY
            else
                echo -e "  ${YELLOW}Python not available to parse camera list${NC}"
            fi
        else
            echo -e "  ${YELLOW}No cameras detected${NC}"
        fi
    else
        echo -e "  ${YELLOW}curl command not available${NC}"
    fi
else
    echo -e "  ${YELLOW}Backend not running${NC}"
fi

echo
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "  ${GREEN}make start${NC}  - Start backend service"
echo -e "  ${RED}make stop${NC}   - Stop backend service"
echo -e "  ${YELLOW}make reload${NC} - Restart backend service"
echo -e "  ${BLUE}make logs${NC}   - View backend logs"
echo -e "${BLUE}════════════════════════════════════════${NC}"
