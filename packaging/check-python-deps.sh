#!/bin/bash
#
# Check if all Python dependencies from requirements.txt are available
#

set -e

REQUIREMENTS_FILE="$1"

if [ -z "$REQUIREMENTS_FILE" ] || [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: Requirements file not found: $REQUIREMENTS_FILE" >&2
    exit 1
fi

echo "Checking Python dependencies..."

MISSING_DEPS=()
CHECKED_MODULES=()

# Map package names to module names (package name != import name in some cases)
declare -A MODULE_MAP=(
    ["opencv-python-headless"]="cv2"
    ["python-multipart"]="multipart"
    ["python-dotenv"]="dotenv"
    ["pyyaml"]="yaml"
)

# Parse requirements.txt and check each package
while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    # Extract package name (before ==, >=, <, etc.)
    PACKAGE=$(echo "$line" | sed -E 's/^([a-zA-Z0-9_-]+).*/\1/')

    # Skip if already checked
    [[ " ${CHECKED_MODULES[@]} " =~ " ${PACKAGE} " ]] && continue

    # Get module name (use map if exists, otherwise use package name)
    MODULE="${MODULE_MAP[$PACKAGE]:-$PACKAGE}"

    # Try to import the module
    if ! python3 -c "import $MODULE" 2>/dev/null; then
        MISSING_DEPS+=("$PACKAGE (import $MODULE)")
    fi

    CHECKED_MODULES+=("$PACKAGE")

done < "$REQUIREMENTS_FILE"

# Report results
if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "✓ All Python dependencies are available"
    exit 0
else
    echo ""
    echo "ERROR: Missing Python dependencies:" >&2
    echo "" >&2
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  ✗ $dep" >&2
    done
    echo "" >&2
    echo "Please install missing dependencies before installing this package:" >&2
    echo "" >&2
    echo "Option 1 - Install via pip (recommended):" >&2
    echo "  sudo pip3 install -r /usr/lib/machinevision/python-backend/requirements.txt" >&2
    echo "" >&2
    echo "Option 2 - Install via apt (if available):" >&2
    echo "  sudo apt install python3-fastapi python3-numpy python3-opencv" >&2
    echo "" >&2
    exit 1
fi
