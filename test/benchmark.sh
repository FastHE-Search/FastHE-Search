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

#!/bin/bash

# Benchmark script for HyDia - runs benchmarks for CPU and/or GPU builds
#
# Usage:
#   ./benchmark.sh both logn=[10,12,14] kmatch=[16,32,32]
#   ./benchmark.sh gpu logn=[10,12] kmatch=[16,128] approaches_gpu=[6,51]
#   ./benchmark.sh cpu logn=10 kmatch=32 approaches_cpu=[5,8]
#   ./benchmark.sh both logn=[10,12,14] kmatch=[16,32,32] approaches_gpu=[6,51] approaches_cpu=[5,8]
#   ./benchmark.sh gpu logn=[10,12] kmatch=[16,32] t=3   # Run each case 3 times
#   ./benchmark.sh gpu logn=10 t=3 fresh_dataset=true   # New dataset for each trial
#   ./benchmark.sh gpu logn=10 t=3 fresh_dataset=false  # Same dataset for all trials (default)
#
# Defaults:
#   logn=[10] (2^10 = 1024 vectors)
#   kmatch=[16]
#   t=1 (number of trials per configuration)
#   fresh_dataset=false (reuse same dataset across trials for statistical consistency)
#   approaches_cpu=[5,8]
#   approaches_gpu=[51,81]

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_all.py"
OUTPUT_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build directory (can be overridden with BUILD_DIR env var)
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"

# Test data directory (can be overridden with TEST_DATA_DIR env var)
TEST_DATA_DIR="${TEST_DATA_DIR:-$PROJECT_ROOT/test}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values (arrays)
LOGN_ARRAY=(10)
KMATCH_ARRAY=(16)
NUM_TRIALS=1
FRESH_DATASET=false
BUILD_MODE="current"
APPROACHES_GPU=""
APPROACHES_CPU=""
FIXED_KEYS=false

# Parse bracket array notation [1,2,3] into bash array
parse_array() {
    local input="$1"
    # Remove brackets and split by comma
    echo "$input" | tr -d '[]' | tr ',' ' '
}

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        logn=*)
            LOGN_VAL="${arg#logn=}"
            if [[ "$LOGN_VAL" == \[*\] ]]; then
                # Array notation
                LOGN_ARRAY=($(parse_array "$LOGN_VAL"))
            else
                # Single value
                LOGN_ARRAY=($LOGN_VAL)
            fi
            ;;
        kmatch=*)
            KMATCH_VAL="${arg#kmatch=}"
            if [[ "$KMATCH_VAL" == \[*\] ]]; then
                # Array notation
                KMATCH_ARRAY=($(parse_array "$KMATCH_VAL"))
            else
                # Single value
                KMATCH_ARRAY=($KMATCH_VAL)
            fi
            ;;
        t=*)
            NUM_TRIALS="${arg#t=}"
            if ! [[ "$NUM_TRIALS" =~ ^[0-9]+$ ]] || [ "$NUM_TRIALS" -lt 1 ]; then
                echo -e "${RED}Error: t must be a positive integer${NC}"
                exit 1
            fi
            ;;
        fresh_dataset=*)
            FRESH_VAL="${arg#fresh_dataset=}"
            if [[ "$FRESH_VAL" == "true" || "$FRESH_VAL" == "1" || "$FRESH_VAL" == "yes" ]]; then
                FRESH_DATASET=true
            elif [[ "$FRESH_VAL" == "false" || "$FRESH_VAL" == "0" || "$FRESH_VAL" == "no" ]]; then
                FRESH_DATASET=false
            else
                echo -e "${RED}Error: fresh_dataset must be true/false${NC}"
                exit 1
            fi
            ;;
        fixed_keys=*)
            FIXED_KEYS_VAL="${arg#fixed_keys=}"
            if [[ "$FIXED_KEYS_VAL" == "true" || "$FIXED_KEYS_VAL" == "1" || "$FIXED_KEYS_VAL" == "yes" ]]; then
                FIXED_KEYS=true
            elif [[ "$FIXED_KEYS_VAL" == "false" || "$FIXED_KEYS_VAL" == "0" || "$FIXED_KEYS_VAL" == "no" ]]; then
                FIXED_KEYS=false
            else
                echo -e "${RED}Error: fixed_keys must be true/false${NC}"
                exit 1
            fi
            ;;
        comp_depth=*)
            export COMP_DEPTH_VAL="${arg#comp_depth=}"
            echo -e "${BLUE}Using COMP_DEPTH_VAL=${COMP_DEPTH_VAL}${NC}"
            ;;
        approaches_gpu=*)
            APPROACHES_GPU="${arg#approaches_gpu=}"
            ;;
        approaches_cpu=*)
            APPROACHES_CPU="${arg#approaches_cpu=}"
            ;;
        cpu|gpu|both|current)
            BUILD_MODE="$arg"
            ;;
        -h|--help)
            echo "Usage: $0 [cpu|gpu|both|current] [logn=N|logn=[N1,N2,...]] [kmatch=K|kmatch=[K1,K2,...]] [t=N] [comp_depth=N] [fresh_dataset=true|false] [fixed_keys=true|false] [approaches_gpu=[...]] [approaches_cpu=[...]]"
            echo ""
            echo "Build modes:"
            echo "  cpu      Build with OpenFHE v1.3.0 and run benchmark"
            echo "  gpu      Build with OpenFHE v1.2.3 (optimized) and run benchmark"
            echo "  both     Build and benchmark both OpenFHE versions"
            echo "  current  Run benchmark with current build (default)"
            echo ""
            echo "Dataset options (can be single value or array):"
            echo "  logn=N           Single dataset: 2^N vectors"
            echo "  logn=[N1,N2,N3]  Multiple datasets: 2^N1, 2^N2, 2^N3 vectors"
            echo "  kmatch=K         Single k value for all datasets"
            echo "  kmatch=[K1,K2,K3] Corresponding k values (must match logn array length)"
            echo ""
            echo "Trial options:"
            echo "  t=N                  Number of trials per configuration (default: 1)"
            echo "  fresh_dataset=false  Reuse same dataset for all trials (default, for statistical consistency)"
            echo "  fresh_dataset=true   Generate new dataset for each trial (for diversity testing)"
            echo "  fixed_keys=true      Reuse the same HE key pair across trials while regenerating the encrypted DB"
            echo ""
            echo "Build options:"
            echo "  comp_depth=N         Comparison depth (default: 8). Passed to cmake as COMP_DEPTH_VAL."
            echo ""
            echo "Approach selection (optional - defaults used if not specified):"
            echo "  approaches_gpu=[51,81]     Approaches for GPU build (default)"
            echo "  approaches_cpu=[5,8]       Approaches for CPU build (default)"
            echo ""
            echo "Available approaches:"
            echo "  5   - HyDia_CPU (baseline)"
            echo "  6   - BSGS-Orig (CPU)"
            echo "  7   - BSGS-Precomp (CPU)"
            echo "  8   - BSGS-Precomp-Opt (CPU)"
            echo "  9   - BSGS-OnlineAgg (CPU)"
            echo "  51  - HyDia-GPU (GPU build only)"
            echo "  81  - BSGS-GPU (GPU build only)"
            echo ""
            echo "Examples:"
            echo "  $0 both                                          # Use all defaults"
            echo "  $0 gpu logn=12                                   # GPU, 2^12 vectors, k=16 (default)"
            echo "  $0 both logn=[10,12,14] kmatch=[16,32,32]        # Multiple datasets"
            echo "  $0 gpu logn=[10,12] kmatch=[16,128] approaches_gpu=[6,51]"
            echo "  $0 gpu logn=10 t=3                               # 3 trials, same dataset (default)"
            echo "  $0 gpu logn=10 t=3 fresh_dataset=true            # 3 trials, new dataset each"
            echo "  BUILD_DIR=/path/to/build TEST_DATA_DIR=/path/to/test $0 gpu logn=18"
            echo ""
            echo "Environment variables:"
            echo "  BUILD_DIR       Custom build directory (default: ./build)"
            echo "  TEST_DATA_DIR   Custom test data directory (default: ./test)"
            exit 0
            ;;
    esac
done

# Validate array lengths match
if [ ${#LOGN_ARRAY[@]} -ne ${#KMATCH_ARRAY[@]} ]; then
    echo -e "${RED}Error: logn and kmatch arrays must have the same length${NC}"
    echo -e "${RED}  logn has ${#LOGN_ARRAY[@]} elements: ${LOGN_ARRAY[*]}${NC}"
    echo -e "${RED}  kmatch has ${#KMATCH_ARRAY[@]} elements: ${KMATCH_ARRAY[*]}${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEST_DATA_DIR"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}HyDia Benchmark Suite${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Build directory: ${BUILD_DIR}${NC}"
echo -e "${BLUE}Test data directory: ${TEST_DATA_DIR}${NC}"
echo -e "${BLUE}Trials per configuration: ${NUM_TRIALS}${NC}"
echo -e "${BLUE}Fresh dataset per trial: ${FRESH_DATASET}${NC}"
echo -e "${BLUE}Fixed key pair across trials: ${FIXED_KEYS}${NC}"
echo -e "${BLUE}Datasets: ${#LOGN_ARRAY[@]} configuration(s)${NC}"
for i in "${!LOGN_ARRAY[@]}"; do
    echo -e "${BLUE}  [$((i+1))] 2^${LOGN_ARRAY[$i]} = $((2**LOGN_ARRAY[$i])) vectors, k=${KMATCH_ARRAY[$i]}${NC}"
done
if [ -n "$APPROACHES_GPU" ]; then
    echo -e "${BLUE}GPU Approaches: ${APPROACHES_GPU}${NC}"
else
    echo -e "${BLUE}GPU Approaches: [51,81] (default)${NC}"
fi
if [ -n "$APPROACHES_CPU" ]; then
    echo -e "${BLUE}CPU Approaches: ${APPROACHES_CPU}${NC}"
else
    echo -e "${BLUE}CPU Approaches: [5,8] (default)${NC}"
fi
echo -e "${BLUE}======================================${NC}"

# Build and benchmark function
run_benchmark() {
    local build_type=$1
    local logn=$2
    local kmatch=$3
    local extra_args=$4
    local output_csv=$5
    local trial=$6
    local clean_serial=${7:-true}
    local keep_serial=${8:-false}
    local reuse_keys_only=${9:-false}
    
    echo ""
    echo -e "${GREEN}--------------------------------------${NC}"
    echo -e "${GREEN}Running benchmark: 2^${logn} vectors, k=${kmatch}, trial ${trial}/${NUM_TRIALS}${NC}"
    echo -e "${GREEN}--------------------------------------${NC}"
    
    # Run benchmark (logn and kmatch are positional args, others are optional)
    # Pass trial number to Python script
    # Export BUILD_DIR and TEST_DATA_DIR for the Python script
    cd "$SCRIPT_DIR"
    CLEAN_SERIAL="$clean_serial" KEEP_SERIAL="$keep_serial" REUSE_KEYS_ONLY="$reuse_keys_only" BUILD_DIR="$BUILD_DIR" TEST_DATA_DIR="$TEST_DATA_DIR" python3 benchmark_all.py "$logn" "$kmatch" --output_file "$output_csv" --trial "$trial" $extra_args
}

set_trial_serial_policy() {
    local trial=$1

    CLEAN_SERIAL=true
    KEEP_SERIAL=false
    REUSE_KEYS_ONLY=false

    if [ "$FIXED_KEYS" = "true" ]; then
        if [ "$FRESH_DATASET" = "false" ]; then
            if [ "$trial" -eq 1 ]; then
                CLEAN_SERIAL=true
                if [ "$NUM_TRIALS" -gt 1 ]; then
                    KEEP_SERIAL=true
                fi
            else
                CLEAN_SERIAL=false
                if [ "$trial" -lt "$NUM_TRIALS" ]; then
                    KEEP_SERIAL=true
                fi
            fi
        else
            if [ "$trial" -eq 1 ]; then
                CLEAN_SERIAL=true
                if [ "$NUM_TRIALS" -gt 1 ]; then
                    KEEP_SERIAL=true
                fi
            else
                CLEAN_SERIAL=true
                REUSE_KEYS_ONLY=true
                if [ "$trial" -lt "$NUM_TRIALS" ]; then
                    KEEP_SERIAL=true
                fi
            fi
        fi
        return
    fi

    if [ "$FRESH_DATASET" = "false" ]; then
        if [ "$trial" -eq 1 ]; then
            CLEAN_SERIAL=true
            if [ "$NUM_TRIALS" -gt 1 ]; then
                KEEP_SERIAL=true
            fi
        else
            CLEAN_SERIAL=false
            if [ "$trial" -lt "$NUM_TRIALS" ]; then
                KEEP_SERIAL=true
            fi
        fi
    fi
}

# Build function (separate from benchmark)
build_for_mode() {
    local build_type=$1
    
    echo ""
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Building for ${build_type}...${NC}"
    echo -e "${YELLOW}Build directory: ${BUILD_DIR}${NC}"
    echo -e "${YELLOW}======================================${NC}"
    
    cd "$PROJECT_ROOT"
    if [ "$build_type" == "cpu" ]; then
        echo "Building with OpenFHE v1.3.0..."
        ./build.sh cpu 2>&1 | tail -20
    else
        echo "Building with OpenFHE v1.2.3 (optimized)..."
        ./build.sh gpu 2>&1 | tail -20
    fi
}

# Run benchmarks based on mode
case "$BUILD_MODE" in
    cpu)
        build_for_mode "cpu"
        BASE_EXTRA_ARGS=""
        if [ -n "$APPROACHES_CPU" ]; then
            BASE_EXTRA_ARGS="--approaches $APPROACHES_CPU"
        fi
        
        # Single output CSV for all runs - include t value in filename
        OUTPUT_CSV="$OUTPUT_DIR/benchmark_cpu_t${NUM_TRIALS}_${TIMESTAMP}.csv"
        mkdir -p "$OUTPUT_DIR"
        
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}Running CPU benchmarks (${NUM_TRIALS} trial(s) each)...${NC}"
        echo -e "${GREEN}Output: ${OUTPUT_CSV}${NC}"
        echo -e "${GREEN}======================================${NC}"
        
        # Store indices per configuration for reuse across trials (when fresh_dataset=false)
        declare -A CONFIG_INDICES
        
        for i in "${!LOGN_ARRAY[@]}"; do
            LOGN="${LOGN_ARRAY[$i]}"
            KMATCH="${KMATCH_ARRAY[$i]}"
            
            for trial in $(seq 1 $NUM_TRIALS); do
                set_trial_serial_policy "$trial"

                if [ "$FRESH_DATASET" = "true" ]; then
                    # fresh_dataset=true: Generate new dataset for every trial
                    run_benchmark "cpu" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                elif [ "$trial" -eq 1 ]; then
                    # fresh_dataset=false, first trial: generate dataset, save indices
                    run_benchmark "cpu" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    INDICES_FILE="$PROJECT_ROOT/data/last_indices.txt"
                    if [ -f "$INDICES_FILE" ]; then
                        CONFIG_INDICES["${LOGN}_${KMATCH}"]=$(cat "$INDICES_FILE")
                    fi
                else
                    # fresh_dataset=false, subsequent trials: reuse same dataset via --indices
                    INDICES="${CONFIG_INDICES["${LOGN}_${KMATCH}"]}"
                    if [ -n "$INDICES" ]; then
                        EXTRA_ARGS="$BASE_EXTRA_ARGS --indices $INDICES"
                        run_benchmark "cpu" "$LOGN" "$KMATCH" "$EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    else
                        echo -e "${RED}Error: No cached indices for logn=$LOGN kmatch=$KMATCH${NC}"
                    fi
                fi
            done
        done
        ;;
    gpu)
        build_for_mode "gpu"
        BASE_EXTRA_ARGS=""
        if [ -n "$APPROACHES_GPU" ]; then
            BASE_EXTRA_ARGS="--approaches $APPROACHES_GPU"
        fi
        
        # Single output CSV for all runs - include t value in filename
        OUTPUT_CSV="$OUTPUT_DIR/benchmark_gpu_t${NUM_TRIALS}_${TIMESTAMP}.csv"
        mkdir -p "$OUTPUT_DIR"
        
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}Running GPU benchmarks (${NUM_TRIALS} trial(s) each)...${NC}"
        echo -e "${GREEN}Output: ${OUTPUT_CSV}${NC}"
        echo -e "${GREEN}======================================${NC}"
        
        # Store indices per configuration for reuse across trials (when fresh_dataset=false)
        declare -A CONFIG_INDICES
        
        for i in "${!LOGN_ARRAY[@]}"; do
            LOGN="${LOGN_ARRAY[$i]}"
            KMATCH="${KMATCH_ARRAY[$i]}"
            
            for trial in $(seq 1 $NUM_TRIALS); do
                set_trial_serial_policy "$trial"

                if [ "$FRESH_DATASET" = "true" ]; then
                    # fresh_dataset=true: Generate new dataset for every trial
                    run_benchmark "gpu" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                elif [ "$trial" -eq 1 ]; then
                    # fresh_dataset=false, first trial: generate dataset, save indices
                    run_benchmark "gpu" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    INDICES_FILE="$PROJECT_ROOT/data/last_indices.txt"
                    if [ -f "$INDICES_FILE" ]; then
                        CONFIG_INDICES["${LOGN}_${KMATCH}"]=$(cat "$INDICES_FILE")
                    fi
                else
                    # fresh_dataset=false, subsequent trials: reuse same dataset via --indices
                    INDICES="${CONFIG_INDICES["${LOGN}_${KMATCH}"]}"
                    if [ -n "$INDICES" ]; then
                        EXTRA_ARGS="$BASE_EXTRA_ARGS --indices $INDICES"
                        run_benchmark "gpu" "$LOGN" "$KMATCH" "$EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    else
                        echo -e "${RED}Error: No cached indices for logn=$LOGN kmatch=$KMATCH${NC}"
                    fi
                fi
            done
        done
        ;;
    both)
        echo -e "${YELLOW}Running benchmarks for both CPU and GPU builds${NC}"
        
        # Generate a random seed for reproducibility
        SEED=$RANDOM
        echo -e "${BLUE}Using seed: ${SEED} for reproducible dataset generation${NC}"
        
        # Single output CSV for ALL runs (both CPU and GPU) - include t value in filename
        OUTPUT_CSV="$OUTPUT_DIR/benchmark_both_t${NUM_TRIALS}_${TIMESTAMP}.csv"
        mkdir -p "$OUTPUT_DIR"
        
        # Build and run CPU benchmarks
        build_for_mode "cpu"
        BASE_EXTRA_ARGS_CPU="--seed $SEED"
        if [ -n "$APPROACHES_CPU" ]; then
            BASE_EXTRA_ARGS_CPU="$BASE_EXTRA_ARGS_CPU --approaches $APPROACHES_CPU"
        fi
        
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}Running CPU benchmarks (${NUM_TRIALS} trial(s) each)...${NC}"
        echo -e "${GREEN}Output: ${OUTPUT_CSV}${NC}"
        echo -e "${GREEN}======================================${NC}"
        
        # Store indices per dataset for reuse across trials and GPU runs (when fresh_dataset=false)
        declare -A INDICES_MAP
        
        for i in "${!LOGN_ARRAY[@]}"; do
            LOGN="${LOGN_ARRAY[$i]}"
            KMATCH="${KMATCH_ARRAY[$i]}"
            
            for trial in $(seq 1 $NUM_TRIALS); do
                set_trial_serial_policy "$trial"

                if [ "$FRESH_DATASET" = "true" ]; then
                    # fresh_dataset=true: Generate new dataset for every trial
                    run_benchmark "cpu" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS_CPU" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    # Save indices for GPU run on same trial
                    INDICES_FILE="$PROJECT_ROOT/data/last_indices.txt"
                    if [ -f "$INDICES_FILE" ]; then
                        INDICES_MAP["${LOGN}_${KMATCH}_${trial}"]=$(cat "$INDICES_FILE")
                    fi
                elif [ "$trial" -eq 1 ]; then
                    # fresh_dataset=false, first trial: generate dataset, save indices
                    run_benchmark "cpu" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS_CPU" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    INDICES_FILE="$PROJECT_ROOT/data/last_indices.txt"
                    if [ -f "$INDICES_FILE" ]; then
                        INDICES_MAP["${LOGN}_${KMATCH}"]=$(cat "$INDICES_FILE")
                    fi
                else
                    # fresh_dataset=false, subsequent trials: reuse same dataset via --indices
                    INDICES="${INDICES_MAP["${LOGN}_${KMATCH}"]}"
                    if [ -n "$INDICES" ]; then
                        EXTRA_ARGS_CPU="$BASE_EXTRA_ARGS_CPU --indices $INDICES"
                        run_benchmark "cpu" "$LOGN" "$KMATCH" "$EXTRA_ARGS_CPU" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    else
                        echo -e "${RED}Error: No cached indices for logn=$LOGN kmatch=$KMATCH${NC}"
                    fi
                fi
            done
        done
        
        # Build and run GPU benchmarks
        build_for_mode "gpu"
        
        BASE_EXTRA_ARGS_GPU=""
        if [ -n "$APPROACHES_GPU" ]; then
            BASE_EXTRA_ARGS_GPU="--approaches $APPROACHES_GPU"
        fi
        
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}Running GPU benchmarks (${NUM_TRIALS} trial(s) each)...${NC}"
        echo -e "${GREEN}Output: ${OUTPUT_CSV}${NC}"
        echo -e "${GREEN}======================================${NC}"
        
        for i in "${!LOGN_ARRAY[@]}"; do
            LOGN="${LOGN_ARRAY[$i]}"
            KMATCH="${KMATCH_ARRAY[$i]}"
            
            for trial in $(seq 1 $NUM_TRIALS); do
                set_trial_serial_policy "$trial"

                if [ "$FRESH_DATASET" = "true" ]; then
                    # fresh_dataset=true: Use indices from same trial's CPU run
                    INDICES="${INDICES_MAP["${LOGN}_${KMATCH}_${trial}"]}"
                else
                    # fresh_dataset=false: Use shared indices from trial 1
                    INDICES="${INDICES_MAP["${LOGN}_${KMATCH}"]}"
                fi
                
                if [ -z "$INDICES" ]; then
                    echo -e "${RED}Error: No indices found for dataset logn=$LOGN kmatch=$KMATCH trial=$trial${NC}"
                    continue
                fi
                
                EXTRA_ARGS_GPU="$BASE_EXTRA_ARGS_GPU --indices $INDICES"
                run_benchmark "gpu" "$LOGN" "$KMATCH" "$EXTRA_ARGS_GPU" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
            done
        done
        
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}Both benchmarks complete!${NC}"
        echo -e "${GREEN}======================================${NC}"
        ;;
    current)
        BASE_EXTRA_ARGS=""
        # Try to detect current build type and use appropriate approaches
        if [ -f "$BUILD_DIR/ImageMatching" ]; then
            if ldd "$BUILD_DIR/ImageMatching" 2>/dev/null | grep -q "libcuda"; then
                BUILD_LABEL="gpu"
                if [ -n "$APPROACHES_GPU" ]; then
                    BASE_EXTRA_ARGS="$BASE_EXTRA_ARGS --approaches $APPROACHES_GPU"
                fi
            else
                BUILD_LABEL="cpu"
                if [ -n "$APPROACHES_CPU" ]; then
                    BASE_EXTRA_ARGS="$BASE_EXTRA_ARGS --approaches $APPROACHES_CPU"
                fi
            fi
        else
            BUILD_LABEL="unknown"
        fi
        
        # Single output CSV for all runs - include t value in filename
        OUTPUT_CSV="$OUTPUT_DIR/benchmark_${BUILD_LABEL}_t${NUM_TRIALS}_${TIMESTAMP}.csv"
        mkdir -p "$OUTPUT_DIR"
        
        echo ""
        echo -e "${GREEN}Running benchmarks with current build (${NUM_TRIALS} trial(s) each)...${NC}"
        echo -e "${GREEN}Build directory: ${BUILD_DIR}${NC}"
        echo -e "${GREEN}Output: ${OUTPUT_CSV}${NC}"
        
        # Store indices per dataset for reuse across trials (when fresh_dataset=false)
        declare -A INDICES_MAP
        
        for i in "${!LOGN_ARRAY[@]}"; do
            LOGN="${LOGN_ARRAY[$i]}"
            KMATCH="${KMATCH_ARRAY[$i]}"
            
            for trial in $(seq 1 $NUM_TRIALS); do
                set_trial_serial_policy "$trial"

                if [ "$FRESH_DATASET" = "true" ]; then
                    # fresh_dataset=true: Generate new dataset for every trial
                    run_benchmark "current" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                elif [ "$trial" -eq 1 ]; then
                    # fresh_dataset=false, first trial: generate dataset, save indices
                    run_benchmark "current" "$LOGN" "$KMATCH" "$BASE_EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    INDICES_FILE="$PROJECT_ROOT/data/last_indices.txt"
                    if [ -f "$INDICES_FILE" ]; then
                        INDICES_MAP["${LOGN}_${KMATCH}"]=$(cat "$INDICES_FILE")
                    fi
                else
                    # fresh_dataset=false, subsequent trials: reuse same dataset via --indices
                    INDICES="${INDICES_MAP["${LOGN}_${KMATCH}"]}"
                    if [ -n "$INDICES" ]; then
                        EXTRA_ARGS="$BASE_EXTRA_ARGS --indices $INDICES"
                        run_benchmark "current" "$LOGN" "$KMATCH" "$EXTRA_ARGS" "$OUTPUT_CSV" "$trial" "$CLEAN_SERIAL" "$KEEP_SERIAL" "$REUSE_KEYS_ONLY"
                    else
                        echo -e "${RED}Error: No cached indices for logn=$LOGN kmatch=$KMATCH${NC}"
                    fi
                fi
            done
        done
        ;;
esac

# Cleanup: Remove serial directory to free disk space
SERIAL_DIR="$BUILD_DIR/serial"
if [ -d "$SERIAL_DIR" ]; then
    echo ""
    echo -e "${YELLOW}Cleaning up serialized keys directory...${NC}"
    SERIAL_SIZE=$(du -sh "$SERIAL_DIR" 2>/dev/null | cut -f1)
    rm -rf "$SERIAL_DIR"
    echo -e "${GREEN}Removed $SERIAL_DIR (freed ~$SERIAL_SIZE)${NC}"
fi

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "${GREEN}Results saved in: ${OUTPUT_DIR}${NC}"
