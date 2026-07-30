#!/bin/bash

#  Quick test version of run_benchmarks.sh
#  - Reduces trials from 11 to 1
#  - Reduces dataset sizes (logn values)
#  - Runs only subset of approaches for quick validation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
export TEST_DATA_DIR="${TEST_DATA_DIR:-$SCRIPT_DIR/test}"

set -o pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERBOSE_LOG="$SCRIPT_DIR/benchmark_results/benchmark-lite-verbose-${TIMESTAMP}.log"
CPP_STDOUT_LOG="$SCRIPT_DIR/benchmark_results/benchmark-lite-cpp-stdout-${TIMESTAMP}.log"
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

log "Starting LITE benchmark suite at $(date)"
log "Project root: $SCRIPT_DIR"
log "Build directory: $BUILD_DIR"
log "Test data directory: $TEST_DATA_DIR"
log "Verbose log: $VERBOSE_LOG"
log "C++ stdout log: $CPP_STDOUT_LOG"
log ""
log "⚡ LITE MODE: t=1 (single trial), reduced dataset sizes"
log ""

log "=========================================="
log "Part 1: Quick validation per approach"
log "Started at $(date)"
log "=========================================="

# Test 1: GPU approach 51 - small dataset
run_campaign "GPU approach 51 (quick test, 2^10..2^12)" gpu \
  logn=[10,11,12] \
  kmatch=[16,16,32] \
  approaches_gpu=[51] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

# Test 2: GPU approach 81 - small dataset
run_campaign "GPU approach 81 (quick test, 2^10..2^12)" current \
  logn=[10,11,12] \
  kmatch=[16,16,32] \
  approaches_gpu=[81] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

# Test 3: GPU approach 812 - small dataset
run_campaign "GPU approach 812 (quick test, 2^10..2^13)" current \
  logn=[10,11,12,13] \
  kmatch=[16,16,32,32] \
  approaches_gpu=[812] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

# Test 4: GPU approach 5 - small dataset
run_campaign "GPU approach 5 (quick test, 2^10..2^13)" current \
  logn=[10,11,12,13] \
  kmatch=[16,16,32,32] \
  approaches_gpu=[5] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

# Test 5: CPU approach 5 - small dataset
run_campaign "CPU approach 5 (quick test, 2^10..2^13)" cpu \
  logn=[10,11,12,13] \
  kmatch=[16,16,32,32] \
  approaches_cpu=[5] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=10

log ""
log "=========================================="
log "Part 2: Literature baseline approaches (GROTE, Blind, HERS) - quick check"
log "Started at $(date)"
log "=========================================="

# Test 6: GROTE (approach 2) - small dataset
# Note: GROTE's index decryption may exceed CKKS approximation limits (an
# inherent limitation of the alpha-norm group-testing step); the server-side
# membership/index computation overhead is still measured and recorded.
run_campaign "GROTE approach 2 (quick test, 2^10..2^12)" current \
  logn=[10,11,12] \
  kmatch=[16,16,32] \
  approaches_gpu=[2] \
  approaches_cpu=[2] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

# Test 7: Blind-Match (approach 3) - small dataset
run_campaign "Blind-Match approach 3 (quick test, 2^10..2^12)" current \
  logn=[10,11,12] \
  kmatch=[16,16,32] \
  approaches_gpu=[3] \
  approaches_cpu=[3] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

# Test 8: HERS (approach 4) - small dataset
run_campaign "HERS approach 4 (quick test, 2^10..2^12)" current \
  logn=[10,11,12] \
  kmatch=[16,16,32] \
  approaches_gpu=[4] \
  approaches_cpu=[4] \
  t=1 fixed_keys=true fresh_dataset=false comp_depth=8

log ""
log "=========================================="
log "Lite benchmark complete at $(date)"
log "=========================================="
log ""
log "✅ All systems working! Ready for full benchmarks."
log "Full run: ./run_benchmarks.sh"
