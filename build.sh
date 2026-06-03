#!/bin/bash

#  Copyright (c) 2026 LG Electronics, Inc.
#
#  Licensed under the MIT License (the "License"); you may not use this file
#  except in compliance with the License.
#
#  You may obtain a copy of the License in the LICENSE file at the project
#  root or at
#
#  https://mit-license.org/
#
#  SPDX-License-Identifier: MIT

# HyDia Build Script - Choose CPU or GPU mode
# This script makes it easy to switch between OpenFHE versions

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
DEFAULT_FIDESLIB_DIR="$SCRIPT_DIR/FIDESlib"
SIBLING_FIDESLIB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/fideslib"
BOOTSTRAP_FIDESLIB_DIR="${BOOTSTRAP_FIDESLIB_DIR:-$SCRIPT_DIR/.deps/FIDESlib}"
FIDESLIB_DIR="${FIDESLIB_DIR:-$DEFAULT_FIDESLIB_DIR}"
DEFAULT_FIDESLIB_GIT_URL="$(dirname $(git remote get-url origin --no-push))/fideslib.git"
FIDESLIB_GIT_URL="${FIDESLIB_GIT_URL:-$DEFAULT_FIDESLIB_GIT_URL}"
FIDESLIB_ARCHIVE_URL="${FIDESLIB_ARCHIVE_URL:-}"
JOBS="${JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo "HyDia Build Script"
    echo ""
    echo "Usage: $0 [cpu|gpu|clean|clean-all|help]"
    echo ""
    echo "Options:"
    echo "  cpu       Build for CPU only (uses system OpenFHE 1.3.0)"
    echo "  gpu       Build with GPU support (uses FIDESlib's patched OpenFHE 1.4.2)"
    echo "            - Automatically builds FIDESlib with optimized OpenFHE if needed"
    echo "            - OpenFHE built with -O3 -march=native for best performance"
    echo "  clean     Clean improved-hydia build directory only"
    echo "  clean-all Clean all builds including FIDESlib (forces full rebuild)"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 cpu                                    # CPU build with system OpenFHE"
    echo "  $0 gpu                                    # GPU build (auto-builds FIDESlib)"
    echo "  $0 clean-all && $0 gpu                    # Full clean rebuild for GPU"
    echo "  FIDESLIB_DIR=/path/to/fideslib $0 gpu     # GPU build with external FIDESlib clone"
    echo "  FIDESLIB_GIT_URL=$DEFAULT_FIDESLIB_GIT_URL $0 gpu"
    echo "  FIDESLIB_ARCHIVE_URL=https://host/fideslib.tar.gz $0 gpu"
    echo ""
    echo "Environment overrides:"
    echo "  BUILD_DIR               Custom HyDia build directory"
    echo "  FIDESLIB_DIR            Use an existing FIDESlib checkout"
    echo "  FIDESLIB_GIT_URL        Clone FIDESlib automatically when missing"
    echo "  FIDESLIB_ARCHIVE_URL    Download and extract a FIDESlib source archive"
    echo "  BOOTSTRAP_FIDESLIB_DIR  Where auto-fetched FIDESlib sources are cached"
    echo "  CMAKE_CUDA_ARCHITECTURES Override detected CUDA architecture(s)"
    echo "  JOBS                    Parallel build jobs"
    echo ""
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    if ! command_exists "$1"; then
        echo -e "${RED}Required command not found: $1${NC}"
        exit 1
    fi
}

detect_cuda_architectures() {
    if [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
        echo "$CMAKE_CUDA_ARCHITECTURES"
        return 0
    fi

    if command_exists nvidia-smi; then
        local detected
        detected="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk '{gsub(/\./, ""); print}' | sort -u | paste -sd ';' -)"
        if [ -n "$detected" ]; then
            echo "$detected"
            return 0
        fi
    fi

    echo "native"
}

clone_fideslib_checkout() {
    local clone_url="$1"

    require_command git

    mkdir -p "$(dirname "$BOOTSTRAP_FIDESLIB_DIR")"

    if [ -d "$BOOTSTRAP_FIDESLIB_DIR/.git" ]; then
        echo -e "${YELLOW}Refreshing cached FIDESlib checkout at $BOOTSTRAP_FIDESLIB_DIR${NC}"
        git -C "$BOOTSTRAP_FIDESLIB_DIR" fetch --tags --all --prune
    else
        rm -rf "$BOOTSTRAP_FIDESLIB_DIR"
        echo -e "${YELLOW}Cloning FIDESlib from $clone_url${NC}"
        git clone "$clone_url" "$BOOTSTRAP_FIDESLIB_DIR"
    fi

    if [ ! -f "$BOOTSTRAP_FIDESLIB_DIR/CMakeLists.txt" ]; then
        echo -e "${RED}Cloned FIDESlib checkout is missing CMakeLists.txt${NC}"
        exit 1
    fi

    FIDESLIB_DIR="$BOOTSTRAP_FIDESLIB_DIR"
}

download_fideslib_archive() {
    local archive_url="$1"
    local archive_file="$SCRIPT_DIR/.deps/fideslib-archive.tar.gz"

    mkdir -p "$SCRIPT_DIR/.deps"
    rm -rf "$BOOTSTRAP_FIDESLIB_DIR"
    mkdir -p "$BOOTSTRAP_FIDESLIB_DIR"

    if command_exists curl; then
        echo -e "${YELLOW}Downloading FIDESlib archive with curl...${NC}"
        curl -L "$archive_url" -o "$archive_file"
    elif command_exists wget; then
        echo -e "${YELLOW}Downloading FIDESlib archive with wget...${NC}"
        wget -O "$archive_file" "$archive_url"
    else
        echo -e "${RED}Need curl or wget to download FIDESlib archive${NC}"
        exit 1
    fi

    tar -xzf "$archive_file" -C "$BOOTSTRAP_FIDESLIB_DIR" --strip-components=1

    if [ ! -f "$BOOTSTRAP_FIDESLIB_DIR/CMakeLists.txt" ]; then
        echo -e "${RED}Downloaded archive did not contain a valid FIDESlib source tree${NC}"
        exit 1
    fi

    FIDESLIB_DIR="$BOOTSTRAP_FIDESLIB_DIR"
}

resolve_fideslib_dir() {
    if [ -f "$FIDESLIB_DIR/CMakeLists.txt" ]; then
        return 0
    fi

    if [ "$FIDESLIB_DIR" = "$DEFAULT_FIDESLIB_DIR" ] && [ -d "$SCRIPT_DIR/.git" ] && [ ! -d "$FIDESLIB_DIR/build" -o -z "$(find "$FIDESLIB_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
        echo -e "${YELLOW}FIDESlib submodule source not checked out. Attempting to initialize submodule...${NC}"
        if git -C "$SCRIPT_DIR" submodule update --init --recursive FIDESlib 2>/dev/null; then
            if [ -f "$FIDESLIB_DIR/CMakeLists.txt" ]; then
                echo -e "${GREEN}✓ Initialized FIDESlib submodule${NC}"
                return 0
            fi
        fi
    fi

    if [ "$FIDESLIB_DIR" = "$DEFAULT_FIDESLIB_DIR" ] && [ -d "$FIDESLIB_DIR" ] && [ ! -f "$FIDESLIB_DIR/CMakeLists.txt" ]; then
        echo -e "${YELLOW}Found a partial FIDESlib directory at $FIDESLIB_DIR; using fallback clone path instead.${NC}"
    fi

    if [ "$FIDESLIB_DIR" = "$DEFAULT_FIDESLIB_DIR" ] && [ -f "$SIBLING_FIDESLIB_DIR/CMakeLists.txt" ]; then
        FIDESLIB_DIR="$SIBLING_FIDESLIB_DIR"
        echo -e "${YELLOW}Using sibling FIDESlib checkout at $FIDESLIB_DIR${NC}"
        return 0
    fi

    if [ -n "$FIDESLIB_GIT_URL" ]; then
        clone_fideslib_checkout "$FIDESLIB_GIT_URL"
        echo -e "${GREEN}✓ Using bootstrapped FIDESlib checkout at $FIDESLIB_DIR${NC}"
        return 0
    fi

    if [ -n "$FIDESLIB_ARCHIVE_URL" ]; then
        download_fideslib_archive "$FIDESLIB_ARCHIVE_URL"
        echo -e "${GREEN}✓ Using bootstrapped FIDESlib archive at $FIDESLIB_DIR${NC}"
        return 0
    fi

    echo -e "${RED}FIDESlib source directory not found: $FIDESLIB_DIR${NC}"
    echo ""
    echo "Expected to find: $FIDESLIB_DIR/CMakeLists.txt"
    echo ""
    echo "Fix one of these and retry:"
    echo "  1. Initialize the submodule: git submodule update --init --recursive FIDESlib"
    echo "  2. Point to an existing clone: FIDESLIB_DIR=/path/to/fideslib ./build.sh gpu"
    echo "  3. Place a sibling clone at: $SIBLING_FIDESLIB_DIR"
    echo "  4. Provide a clone URL: FIDESLIB_GIT_URL=$DEFAULT_FIDESLIB_GIT_URL ./build.sh gpu"
    echo "  5. Provide an archive URL: FIDESLIB_ARCHIVE_URL=https://host/fideslib.tar.gz ./build.sh gpu"
    exit 1
}

preflight_gpu_build() {
    require_command cmake

    if ! command_exists nvcc && [ ! -d "/usr/local/cuda" ]; then
        echo -e "${RED}CUDA toolkit not detected. Install CUDA or set up CUDAToolkit for CMake.${NC}"
        exit 1
    fi

    if ! command_exists nvidia-smi; then
        echo -e "${YELLOW}nvidia-smi not found. GPU architecture autodetection may be limited.${NC}"
    fi
}

clean_build() {
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"/*
    echo -e "${GREEN}✓ Build directory cleaned${NC}"
}

clean_all() {
    echo -e "${YELLOW}Cleaning all build directories...${NC}"
    rm -rf "$BUILD_DIR"/*
    resolve_fideslib_dir
    rm -rf "$FIDESLIB_DIR/build"
    rm -rf "$FIDESLIB_DIR/openfhe-install"
    rm -rf "$FIDESLIB_DIR/cmake/build"
    echo -e "${GREEN}✓ All build directories cleaned${NC}"
}

build_fideslib() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Building FIDESlib with optimized OpenFHE${NC}"
    echo -e "${BLUE}========================================${NC}"

    resolve_fideslib_dir

    local cuda_arches
    cuda_arches="$(detect_cuda_architectures)"

    echo -e "${YELLOW}Configuring FIDESlib (this will build OpenFHE 1.4.2 with -O3 -march=native)...${NC}"
    echo -e "${YELLOW}This may take several minutes on first build...${NC}"
    mkdir -p "$FIDESLIB_DIR/openfhe-install" || true

    cmake -S "$FIDESLIB_DIR" -B "$FIDESLIB_DIR/build" \
          -DFIDESLIB_INSTALL_OPENFHE=ON \
          -DCMAKE_BUILD_TYPE=Debug \
          -DFIDESLIB_ARCH="$cuda_arches" \
          -DOPENFHE_INSTALL_PREFIX="${FIDESLIB_DIR}/openfhe-install"

    echo -e "${YELLOW}Building FIDESlib...${NC}"
    cmake --build "$FIDESLIB_DIR/build" -j"$JOBS"

    echo -e "${GREEN}✓ FIDESlib build complete!${NC}"
}

build_cpu() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Building HyDia for CPU (OpenFHE 1.3.0)${NC}"
    echo -e "${BLUE}========================================${NC}"

    require_command cmake

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Clean CMake cache to ensure fresh configuration
    # This prevents GPU settings from affecting CPU build
    if [ -f "CMakeCache.txt" ]; then
        echo -e "${YELLOW}Cleaning CMake cache for fresh CPU configuration...${NC}"
        rm -f CMakeCache.txt
        rm -rf CMakeFiles/
    fi

    echo -e "${YELLOW}Running CMake...${NC}"
    cmake -DUSE_GPU=OFF \
          -DCOMP_DEPTH_VAL="${COMP_DEPTH_VAL:-8}" \
          "$SCRIPT_DIR" || exit 1

    echo -e "${YELLOW}Building...${NC}"
        cmake --build "$BUILD_DIR" -j"$JOBS" || exit 1

    echo -e "${GREEN}✓ CPU build complete!${NC}"
    echo -e "${GREEN}OpenFHE Version: 1.3.0 (system)${NC}"
    echo -e "${GREEN}Supported approaches: 1-8, 11${NC}"
    echo ""
    echo "Run with: ./run_hydia.sh test/test_1024_k10.dat 6 11 45"
}

build_gpu() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Building HyDia for GPU (OpenFHE 1.4.2)${NC}"
    echo -e "${BLUE}========================================${NC}"

    preflight_gpu_build
    resolve_fideslib_dir
    local cuda_arches
    cuda_arches="$(detect_cuda_architectures)"

    # Check if FIDESlib needs to be built
    if [ ! -f "$FIDESLIB_DIR/build/fideslib.a" ]; then
        echo -e "${YELLOW}FIDESlib not found. Building automatically...${NC}"
        build_fideslib
    else
        echo -e "${GREEN}✓ FIDESlib found at $FIDESLIB_DIR/build/${NC}"
    fi

    # Verify FIDESlib's OpenFHE installation
    FIDESLIB_OPENFHE="$FIDESLIB_DIR/openfhe-install"
    if [ ! -d "$FIDESLIB_OPENFHE/lib/OpenFHE" ]; then
        echo -e "${YELLOW}FIDESlib's OpenFHE not found. Rebuilding FIDESlib...${NC}"
        build_fideslib
    fi

    echo -e "${GREEN}Using FIDESlib's patched OpenFHE: $FIDESLIB_OPENFHE${NC}"

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Clean CMake cache to ensure fresh configuration
    # This prevents CPU settings from affecting GPU build
    if [ -f "CMakeCache.txt" ]; then
        echo -e "${YELLOW}Cleaning CMake cache for fresh GPU configuration...${NC}"
        rm -f CMakeCache.txt
        rm -rf CMakeFiles/
    fi

    echo -e "${YELLOW}Using CUDA architectures: $cuda_arches${NC}"
    echo -e "${YELLOW}Running CMake with GPU support (Release mode with -O3)...${NC}"
    cmake -DUSE_GPU=ON \
          -DFIDESLIB_ROOT="$FIDESLIB_DIR" \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
          -DCMAKE_CUDA_FLAGS_RELEASE="-O3" \
          -DCMAKE_CUDA_ARCHITECTURES="$cuda_arches" \
          -DCOMP_DEPTH_VAL="${COMP_DEPTH_VAL:-8}" \
          "$SCRIPT_DIR" || exit 1

    echo -e "${YELLOW}Building with CUDA...${NC}"
    cmake --build "$BUILD_DIR" -j"$JOBS" || exit 1

    echo -e "${GREEN}✓ GPU build complete!${NC}"
    echo -e "${GREEN}OpenFHE Version: 1.4.2 (FIDESlib-patched, -O3 -march=native)${NC}"
    echo -e "${GREEN}Supported approaches: 1-11, 51, 81 (51 HyDia-GPU, 81 BSGS-GPU)${NC}"
    echo ""
    echo "Run with: ./run_hydia.sh test/test_1024_k10.dat 9 11 45"
}

# Main script logic
case "${1:-help}" in
    cpu)
        build_cpu
        ;;
    gpu)
        build_gpu
        ;;
    clean)
        clean_build
        ;;
    clean-all)
        clean_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
