#!/bin/bash
#
# Build script for creating Machine Vision DEB package
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
VERSION_FILE="$SCRIPT_DIR/VERSION"
BUILD_DIR="$BACKEND_DIR/build"

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    local missing_deps=()

    if ! command -v dpkg-deb &> /dev/null; then
        missing_deps+=("dpkg-dev")
    fi

    if ! command -v debuild &> /dev/null; then
        missing_deps+=("devscripts")
    fi

    if ! command -v dh &> /dev/null; then
        missing_deps+=("debhelper")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo "Install with: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi

    print_success "All dependencies satisfied"
}

# Get version from VERSION file
get_version() {
    if [ ! -f "$VERSION_FILE" ]; then
        print_error "VERSION file not found at $VERSION_FILE"
        exit 1
    fi

    local version=$(cat "$VERSION_FILE" | tr -d '[:space:]')

    if [ -z "$version" ]; then
        print_error "VERSION file is empty"
        exit 1
    fi

    echo "$version"
}

# Clean previous build artifacts
clean_build() {
    print_info "Cleaning previous build artifacts..."

    cd "$BACKEND_DIR"

    # Remove debian build artifacts
    rm -rf debian/.debhelper debian/machinevision debian/files debian/*.substvars
    rm -f ../*.deb ../*.changes ../*.buildinfo ../*.tar.* ../*.dsc

    # Clean build directory
    rm -rf "$BUILD_DIR"

    print_success "Clean complete"
}

# Prepare build directory
prepare_build() {
    local version=$1

    print_info "Preparing build directory for version $version..."

    # Create build directory
    mkdir -p "$BUILD_DIR"

    # Update VERSION file in package
    echo "$version" > "$SCRIPT_DIR/VERSION"

    # Remove __pycache__ and .pyc files from source
    find "$BACKEND_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$BACKEND_DIR/src" -type f -name "*.pyc" -delete 2>/dev/null || true

    print_success "Build directory prepared"
}

# Build the package
build_package() {
    local version=$1

    print_info "Building DEB package version $version..."

    cd "$BACKEND_DIR"

    # Build unsigned package (for local use)
    dpkg-buildpackage -us -uc -b

    # Move .deb to build directory
    if [ -f "../machinevision_${version}_all.deb" ]; then
        mv "../machinevision_${version}_all.deb" "$BUILD_DIR/"
        print_success "Package built successfully: $BUILD_DIR/machinevision_${version}_all.deb"
    else
        # Try without version in filename
        local deb_file=$(ls -1 ../machinevision*.deb 2>/dev/null | head -n 1)
        if [ -n "$deb_file" ]; then
            mv "$deb_file" "$BUILD_DIR/machinevision_${version}_all.deb"
            print_success "Package built successfully: $BUILD_DIR/machinevision_${version}_all.deb"
        else
            print_error "DEB package not found after build"
            exit 1
        fi
    fi

    # Clean up build artifacts
    rm -f ../*.changes ../*.buildinfo ../*.tar.* ../*.dsc
}

# Show package info
show_package_info() {
    local version=$1
    local deb_file="$BUILD_DIR/machinevision_${version}_all.deb"

    print_info "\n=== Package Information ==="
    dpkg-deb --info "$deb_file"

    print_info "\n=== Package Contents ==="
    dpkg-deb --contents "$deb_file" | head -n 20
    echo "..."

    print_info "\n=== Installation Instructions ==="
    echo "To install this package:"
    echo "  sudo apt install $deb_file"
    echo ""
    echo "Or for upgrade:"
    echo "  sudo apt install --reinstall $deb_file"
    echo ""
    echo "To verify installation:"
    echo "  machinevision version"
    echo "  machinevision status"
}

# Main build process
main() {
    print_info "=== Machine Vision DEB Package Builder ==="
    echo ""

    # Check dependencies
    check_dependencies

    # Get version
    VERSION=$(get_version)
    print_info "Building version: $VERSION"
    echo ""

    # Clean previous builds
    clean_build

    # Prepare build
    prepare_build "$VERSION"

    # Build package
    build_package "$VERSION"

    # Show info
    show_package_info "$VERSION"

    echo ""
    print_success "=== Build Complete ==="
}

# Run main
main "$@"
