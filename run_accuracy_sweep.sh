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
# Accuracy sweep: approaches {6, 8} × comp_depth {8, 10}, queries 0–49
# The serial/ folder is cleaned up automatically at the end of each run,
# so each combination starts fresh with the correct parameters.
#
# Outputs (in accuracy_test/):
#   accuracy_sweep_verbose.log   – full stdout from every run
#   accuracy_sweep_summary.txt   – aggregate results only
#   accuracy_approach*_*.csv     – per-run CSV files

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/ImageMatchingAccuracy"
FRGC2_DIR="${SCRIPT_DIR}/../improved-hydia/test/"
OUTPUT_DIR="${SCRIPT_DIR}/accuracy_test"

START=0
END=49

# Create output directory
mkdir -p "${OUTPUT_DIR}"

VERBOSE_LOG="${OUTPUT_DIR}/accuracy_sweep_verbose.log"
SUMMARY_LOG="${OUTPUT_DIR}/accuracy_sweep_summary.txt"

# Clear previous logs
> "${VERBOSE_LOG}"
> "${SUMMARY_LOG}"

echo "Accuracy Sweep Summary" >> "${SUMMARY_LOG}"
echo "======================" >> "${SUMMARY_LOG}"
printf "%-22s %-6s %-6s | %-7s %-7s %-7s %-7s | %-10s %-10s %-10s %-10s | %-10s %-10s\n" \
  "Approach" "ID" "Depth" "TP" "FP" "TN" "FN" "Precision" "Recall" "F1" "Accuracy" "Time(s)" "Amort(s)" >> "${SUMMARY_LOG}"
printf -- "--------------------------------------------------------------------------------------------------------------------------------------\n" >> "${SUMMARY_LOG}"

RUN=0
for APPROACH in 5 8; do
  for DEPTH in 10 8; do
    RUN=$((RUN + 1))
    HEADER="[Run ${RUN}/4] Approach=${APPROACH}  comp_depth=${DEPTH}  queries=[${START},${END}]"

    echo ""
    echo "================================================================"
    echo "  ${HEADER}"
    echo "================================================================"
    echo ""

    # Write header to verbose log
    {
      echo "================================================================"
      echo "  ${HEADER}"
      echo "================================================================"
    } >> "${VERBOSE_LOG}"

    # Run from build dir so the binary can find/create serial/ there
    OUTPUT=$(cd "${BUILD_DIR}" && "${BINARY}" ${START} ${END} ${APPROACH} ${FRGC2_DIR} ${DEPTH} 2>&1)

    # Write full output to verbose log
    echo "${OUTPUT}" >> "${VERBOSE_LOG}"
    echo "" >> "${VERBOSE_LOG}"

    # Also print to terminal
    echo "${OUTPUT}"

    # Move per-run CSV from build dir to output dir
    CSV_FILE="${BUILD_DIR}/accuracy_approach${APPROACH}_depth${DEPTH}_q${START}-${END}.csv"
    if [[ -f "${CSV_FILE}" ]]; then
      mv "${CSV_FILE}" "${OUTPUT_DIR}/"
    fi

    # Extract summary lines for the condensed log
    APPROACH_LINE=$(echo "${OUTPUT}" | grep "^Approach:" || true)
    ENC_TOTALS=$(echo "${OUTPUT}" | grep "^Encrypted  Total" || true)
    ENC_METRICS=$(echo "${OUTPUT}" | grep "^Encrypted  Precision=" || true)
    TIME_LINE=$(echo "${OUTPUT}" | grep "^Total index scenario" || true)

    # Parse TP/FP/TN/FN from aggregate totals
    ENC_TP=$(echo "${ENC_TOTALS}" | grep -oP 'TP=\K[0-9]+' || echo "N/A")
    ENC_FN=$(echo "${ENC_TOTALS}" | grep -oP 'FN=\K[0-9]+' || echo "N/A")
    ENC_TN=$(echo "${ENC_TOTALS}" | grep -oP 'TN=\K[0-9]+' || echo "N/A")
    ENC_FP=$(echo "${ENC_TOTALS}" | grep -oP 'FP=\K[0-9]+' || echo "N/A")

    # Parse encrypted metrics for the table row
    ENC_PREC=$(echo "${ENC_METRICS}" | grep -oP 'Precision=\K[0-9.]+' || echo "N/A")
    ENC_REC=$(echo "${ENC_METRICS}" | grep -oP 'Recall=\K[0-9.]+' || echo "N/A")
    ENC_F1=$(echo "${ENC_METRICS}" | grep -oP 'F1=\K[0-9.]+' || echo "N/A")
    ENC_ACC=$(echo "${ENC_METRICS}" | grep -oP 'Accuracy=\K[0-9.]+' || echo "N/A")
    TOTAL_TIME=$(echo "${TIME_LINE}" | grep -oP 'time=\K[0-9.]+' || echo "N/A")
    AMORT_TIME=$(echo "${TIME_LINE}" | grep -oP 'Amortized=\K[0-9.]+' || echo "N/A")

    # Get approach display name
    APPROACH_NAME=$(echo "${APPROACH_LINE}" | grep -oP 'Approach: \K[^(]+' | xargs || echo "approach${APPROACH}")

    printf "%-22s %-6s %-6s | %-7s %-7s %-7s %-7s | %-10s %-10s %-10s %-10s | %-10s %-10s\n" \
      "${APPROACH_NAME}" "${APPROACH}" "${DEPTH}" \
      "${ENC_TP}" "${ENC_FP}" "${ENC_TN}" "${ENC_FN}" \
      "${ENC_PREC}" "${ENC_REC}" "${ENC_F1}" "${ENC_ACC}" \
      "${TOTAL_TIME}" "${AMORT_TIME}" >> "${SUMMARY_LOG}"

    echo ""
    echo "---- Finished approach=${APPROACH} depth=${DEPTH} ----"
    echo ""
  done
done

echo "" >> "${SUMMARY_LOG}"
echo "Queries: ${START}–${END} (50 total)  |  FRGC2 dir: ${FRGC2_DIR}" >> "${SUMMARY_LOG}"
echo "Generated: $(date)" >> "${SUMMARY_LOG}"

echo ""
echo "All runs complete."
echo "  Output dir:   ${OUTPUT_DIR}/"
echo "  Verbose log:  ${OUTPUT_DIR}/accuracy_sweep_verbose.log"
echo "  Summary:      ${OUTPUT_DIR}/accuracy_sweep_summary.txt"
echo ""
cat "${SUMMARY_LOG}"
