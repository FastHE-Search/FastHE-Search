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

# Benchmark runner script for light-hydia
# Runs benchmark.sh with fixed campaign settings and writes filtered logs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
export TEST_DATA_DIR="${TEST_DATA_DIR:-$SCRIPT_DIR/test}"

set -o pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERBOSE_LOG="$SCRIPT_DIR/benchmark_results/benchmark-verbose-${TIMESTAMP}.log"
CPP_STDOUT_LOG="$SCRIPT_DIR/benchmark_results/benchmark-cpp-stdout-${TIMESTAMP}.log"
mkdir -p "$SCRIPT_DIR/benchmark_results"
: > "$VERBOSE_LOG"
: > "$CPP_STDOUT_LOG"

log() {
  echo "$*" | tee -a "$VERBOSE_LOG"
}

run_and_filter() {
  stdbuf -oL -eL "$@" 2>&1 | awk -v cpp_log="$CPP_STDOUT_LOG" -v verbose_log="$VERBOSE_LOG" '
    {
      raw = $0
      clean = raw
      gsub(/\033\[[0-9;]*[[:alpha:]]/, "", clean)

      print clean >> verbose_log
      fflush(verbose_log)

      is_cpp = 0
      if (index(clean, "[HyDia]") > 0) is_cpp = 1
      else if (index(clean, "[GPU") > 0) is_cpp = 1
      else if (index(clean, "[Sender]") > 0) is_cpp = 1
      else if (index(clean, "[Receiver]") > 0) is_cpp = 1
      else if (index(clean, "[Diagonal") > 0) is_cpp = 1
      else if (index(clean, "[Approach") > 0) is_cpp = 1
      else if (index(clean, "[BSGS") > 0) is_cpp = 1
      else if (index(clean, "[CUDA Streams]") > 0) is_cpp = 1
      else if (index(clean, "[GPUHydiaHelper]") > 0) is_cpp = 1
      else if (clean ~ /\[[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/) is_cpp = 1
      else if (index(clean, "MembComp=") > 0) is_cpp = 1
      else if (index(clean, "IdxComp=") > 0) is_cpp = 1
      else if (clean ~ /Membership scenario:/) is_cpp = 1
      else if (clean ~ /Index scenario:/) is_cpp = 1
      else if (clean ~ /Program successfully terminated/) is_cpp = 1
      else if (clean ~ /^[[:space:]]*\|/) is_cpp = 1
      else if (clean ~ /--- stdout ---/) is_cpp = 1
      else if (clean ~ /--- end stdout ---/) is_cpp = 1
      else if (clean ~ /C\+\+ stdout/) is_cpp = 1

      if (is_cpp) {
        print clean >> cpp_log
        fflush(cpp_log)
      } else {
        print clean
        fflush("/dev/stdout")
      }
    }
  '
  return "${PIPESTATUS[0]}"
}

run_campaign() {
  local label=$1
  shift

  log ""
  log "------------------------------------------"
  log "$label"
  log "------------------------------------------"

  run_and_filter env OUTER_THREADS=40 ./test/benchmark.sh "$@"
  BENCH_RC=$?
  if [ "$BENCH_RC" -ne 0 ]; then
    log "ERROR: $label failed with exit code $BENCH_RC"
    exit "$BENCH_RC"
  fi
}

log "Starting benchmark suite at $(date)"
log "Project root: $SCRIPT_DIR"
log "Build directory: $BUILD_DIR"
log "Test data directory: $TEST_DATA_DIR"
log "Verbose log: $VERBOSE_LOG"
log "C++ stdout log: $CPP_STDOUT_LOG"

log ""
log "=========================================="
log "Part 1: Fixed keys+DB per approach and dataset"
log "Started at $(date)"
log "=========================================="

run_campaign "GPU build (depth 8), approach 51, sizes 2^10..2^15" gpu \
  logn=[10,11,12,13,14,15] \
  kmatch=[16,16,32,32,64,64] \
  approaches_gpu=[51] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "GPU current build (depth 8), approach 81, sizes 2^10..2^16" current \
  logn=[10,11,12,13,14,15,16] \
  kmatch=[16,16,32,32,64,64,128] \
  approaches_gpu=[81] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "GPU current build (depth 8), approach 812, sizes 2^10..2^16" current \
  logn=[10,11,12,13,14,15,16] \
  kmatch=[16,16,32,32,64,64,128] \
  approaches_gpu=[812] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "GPU current build (depth 8), approach 5, sizes 2^10..2^20" current \
  logn=[10,11,12,13,14,15,16,17,18,19,20] \
  kmatch=[16,16,32,32,64,64,128,128,256,256,512] \
  approaches_gpu=[5] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "GPU current build (depth 8), approach 8, sizes 2^10..2^20" gpu \
  logn=[10,11,12,13,14,15,16,17,18,19,20] \
  kmatch=[16,16,32,32,64,64,128,128,256,256,512] \
  approaches_gpu=[8] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8


run_campaign "CPU build (depth 10), approach 5, sizes 2^10..2^20" cpu \
  logn=[10,11,12,13,14,15,16,17,18,19,20] \
  kmatch=[16,16,32,32,64,64,128,128,256,256,512] \
  approaches_cpu=[5] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=10

run_campaign "CPU current build (depth 10), approach 8, sizes 2^10..2^20" current \
  logn=[10,11,12,13,14,15,16,17,18,19,20] \
  kmatch=[16,16,32,32,64,64,128,128,256,256,512] \
  approaches_cpu=[8] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=10

log ""
log "=========================================="
log "Part 2: Literature baseline approaches (GROTE, Blind, HERS)"
log "Server-side membership/index overhead, sizes 2^10..2^14"
log "Started at $(date)"
log "=========================================="

Literature baselines (GROTE=2, Blind=3, HERS=4) are CPU algorithms measured in
BOTH builds to compare OpenFHE versions, exactly like HyDia approach 5.
Note: only the server-side computation overhead is benchmarked; client-side
decryption of these baselines may exceed CKKS approximation limits (an inherent
limitation of the baseline approaches), which is expected and does not affect
the membership/index computation timing reported here.

run_campaign "GPU build (depth 8), approach 2 (GROTE), sizes 2^10..2^14" gpu \
  logn=[10,11,12,13,14] \
  kmatch=[16,16,32,32,64] \
  approaches_gpu=[2] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "GPU current build (depth 8), approach 3 (Blind), sizes 2^10..2^14" current \
  logn=[10,11,12,13,14] \
  kmatch=[16,16,32,32,64] \
  approaches_gpu=[3] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "GPU current build (depth 8), approach 4 (HERS), sizes 2^10..2^14" current \
  logn=[10,11,12,13,14] \
  kmatch=[16,16,32,32,64] \
  approaches_gpu=[4] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=8

run_campaign "CPU build (depth 10), approach 2 (GROTE), sizes 2^10..2^14" cpu \
  logn=[10,11,12,13,14] \
  kmatch=[16,16,32,32,64] \
  approaches_cpu=[2] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=10

run_campaign "CPU current build (depth 10), approach 3 (Blind), sizes 2^10..2^14" current \
  logn=[10,11,12,13,14] \
  kmatch=[16,16,32,32,64] \
  approaches_cpu=[3] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=10

run_campaign "CPU current build (depth 10), approach 4 (HERS), sizes 2^10..2^14" current \
  logn=[10,11,12,13,14] \
  kmatch=[16,16,32,32,64] \
  approaches_cpu=[4] \
  t=2 fixed_keys=true fresh_dataset=false comp_depth=10

log ""
log "=========================================="
log "All benchmarks complete at $(date)"
log "=========================================="
