#!/usr/bin/env python3

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


"""
Aggregate benchmark results: compute mean ± std and median for numeric metrics.
For failed experiments, output N/A.

Usage:
    python aggregate_benchmarks.py <input_csv> [output_csv]

Example:
    python aggregate_benchmarks.py benchmark_gpu_t10_20260110_142159.csv aggregated_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
import re
from pathlib import Path


DECIMAL_PLACES = 4

EXACT_FIRST_TRIAL_COLS = {
    "setup_keygen_s",
    "setup_rotkeygen_s",
    "setup_db_encrypt_s",
    "setup_diag_prerot_s",
    "setup_offline_total_s",
    "peak_disk_gb",
}


def ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    """Return group ordered by trial when available, else preserve row order."""
    if "trial" in group.columns:
        return group.sort_values("trial", kind="stable")
    return group.sort_index(kind="stable")


def stats_window(group: pd.DataFrame) -> pd.DataFrame:
    """Use trials 2..n for stats when n>1; otherwise use trial 1."""
    ordered = ordered_group(group)
    if len(ordered) > 1:
        return ordered.iloc[1:]
    return ordered


def extract_csv_from_log(log_file: str) -> str:
    """
    Extract the CSV filename from a log file.

    Args:
        log_file: Path to the log file

    Returns:
        Path to the CSV file mentioned in the log

    Raises:
        ValueError: If no CSV file is found in the log
    """
    with open(log_file, "r") as f:
        content = f.read()

    # Look for lines like "Output: /path/to/benchmark_cpu_t2_20260203_173120.csv"
    # or "Results appended to: /path/to/benchmark_cpu_t2_20260203_173120.csv"
    pattern = r"(?:Output:|Results appended to:)\s+([^\s]+\.csv)"
    matches = re.findall(pattern, content)

    if not matches:
        raise ValueError(f"No CSV file found in log file: {log_file}")

    # Use the first CSV file found (should be consistent throughout)
    csv_file = matches[0]

    # Check if the path is absolute or relative
    if not os.path.isabs(csv_file):
        # If relative, make it relative to the log file's directory
        log_dir = os.path.dirname(os.path.abspath(log_file))
        csv_file = os.path.join(log_dir, os.path.basename(csv_file))

    if not os.path.exists(csv_file):
        raise ValueError(f"CSV file not found: {csv_file}")

    return csv_file


def aggregate_benchmarks(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Aggregate benchmark results by (approach, dataset_size, k_value).

    For each group:
    - If all trials PASS: compute mean ± std and median
    - If any trial FAIL: output N/A for that metric

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)

    Returns:
        DataFrame with aggregated results
    """
    # Read the CSV
    df = pd.read_csv(input_file)

    # Identify grouping columns and numeric columns
    group_cols = ["build_type", "approach", "dataset_size", "k_value"]

    # Define all possible numeric columns, filter to only those present in the CSV
    all_numeric_cols = [
        "setup_keygen_s",
        "setup_rotkeygen_s",
        "setup_db_encrypt_s",
        "setup_diag_prerot_s",
        "setup_gpu_key_upload_s",
        "setup_gpu_db_cache_s",
        "setup_offline_total_s",
        "setup_total_s",
        "wall_time_s",
        "enroll_time_s",
        "membership_time_s",
        "index_time_s",
        "peak_ram_gb",
        "peak_disk_gb",
    ]
    numeric_cols = [col for col in all_numeric_cols if col in df.columns]

    # Result rows
    results = []

    # Group by approach, dataset_size, k_value
    for name, group in df.groupby(group_cols):
        build_type, approach, dataset_size, k_value = name
        ordered = ordered_group(group)
        stats_group = stats_window(group)

        row = {
            "build_type": build_type,
            "approach": approach,
            "dataset_size": dataset_size,
            "k_value": k_value,
            "num_trials": len(group),
        }

        # Count passing and failing trials
        passing_trials = group[group["success"] == "PASS"]
        num_pass = len(passing_trials)
        stats_passing_trials = stats_group[stats_group["success"] == "PASS"]

        row["num_pass"] = num_pass
        row["success"] = (
            "PASS"
            if num_pass == len(group)
            else ("PARTIAL" if num_pass > 0 else "FAIL")
        )
        row["membership_pass"] = (
            "PASS"
            if (passing_trials["membership_pass"] == "PASS").all()
            else "FAIL" if num_pass > 0 else "N/A"
        )
        row["index_pass"] = (
            "PASS"
            if (passing_trials["index_pass"] == "PASS").all()
            else "FAIL" if num_pass > 0 else "N/A"
        )

        # Some setup metrics are emitted only on trial 1; preserve that exact
        # value instead of aggregating trials 2..n.
        for exact_col in sorted(EXACT_FIRST_TRIAL_COLS.intersection(numeric_cols)):
            exact_trial1 = ordered[exact_col].dropna()
            if len(exact_trial1) > 0:
                v = float(exact_trial1.iloc[0])
                if exact_col == "peak_disk_gb":
                    row[f"{exact_col}_mean"] = f"{v:.2f}"
                    row[f"{exact_col}_std"] = "0.00"
                    row[f"{exact_col}_mean_std"] = f"{v:.2f}"
                    row[f"{exact_col}_median"] = f"{v:.2f}"
                else:
                    row[f"{exact_col}_mean"] = f"{v:.{DECIMAL_PLACES}f}"
                    row[f"{exact_col}_std"] = f"0.{('0' * (DECIMAL_PLACES - 1))}0" if DECIMAL_PLACES > 0 else "0"
                    row[f"{exact_col}_mean_std"] = f"{v:.{DECIMAL_PLACES}f}"
                    row[f"{exact_col}_median"] = f"{v:.{DECIMAL_PLACES}f}"
            else:
                row[f"{exact_col}_mean"] = "N/A"
                row[f"{exact_col}_std"] = "N/A"
                row[f"{exact_col}_mean_std"] = "N/A"
                row[f"{exact_col}_median"] = "N/A"

        # For other numeric columns, use only passing trials from stats window.
        for col in numeric_cols:
            if col in EXACT_FIRST_TRIAL_COLS:
                continue

            if len(stats_passing_trials) > 0:
                values = stats_passing_trials[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std(
                        ddof=0
                    )  # ddof=0 so single trial gives 0 instead of NaN
                    median_val = values.median()

                    # Format: mean ± std
                    row[f"{col}_mean"] = f"{mean_val:.{DECIMAL_PLACES}f}"
                    row[f"{col}_std"] = f"{std_val:.{DECIMAL_PLACES}f}"
                    row[f"{col}_mean_std"] = f"{mean_val:.{DECIMAL_PLACES}f} ± {std_val:.{DECIMAL_PLACES}f}"
                    row[f"{col}_median"] = f"{median_val:.{DECIMAL_PLACES}f}"
                else:
                    row[f"{col}_mean"] = "N/A"
                    row[f"{col}_std"] = "N/A"
                    row[f"{col}_mean_std"] = "N/A"
                    row[f"{col}_median"] = "N/A"
            else:
                row[f"{col}_mean"] = "N/A"
                row[f"{col}_std"] = "N/A"
                row[f"{col}_mean_std"] = "N/A"
                row[f"{col}_median"] = "N/A"

        results.append(row)

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Sort by dataset_size, then approach
    result_df = result_df.sort_values(["dataset_size", "approach"])

    # Save to file if specified
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

    return result_df


def create_summary_table(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Create a cleaner summary table with just mean±std for key metrics.
    For partial success (some trials pass, some fail), compute stats from passing trials only.
    """
    df = pd.read_csv(input_file)

    group_cols = ["build_type", "approach", "dataset_size", "k_value"]

    def format_mean_std(series: pd.Series, decimals: int = 2) -> str:
        return f"{series.mean():.{decimals}f} ± {series.std(ddof=0):.{decimals}f}"

    summary_metric_specs = [
        ("setup_keygen_s", "setup_keygen", DECIMAL_PLACES),
        ("setup_rotkeygen_s", "setup_rotkeygen", DECIMAL_PLACES),
        ("setup_db_encrypt_s", "setup_db_encrypt", DECIMAL_PLACES),
        ("setup_diag_prerot_s", "setup_diag_prerot", DECIMAL_PLACES),
        ("setup_gpu_key_upload_s", "setup_gpu_key_upload", DECIMAL_PLACES),
        ("setup_gpu_db_cache_s", "setup_gpu_db_cache", DECIMAL_PLACES),
        ("setup_offline_total_s", "setup_offline_total", DECIMAL_PLACES),
        ("setup_total_s", "setup_total", DECIMAL_PLACES),
        ("wall_time_s", "wall_time", DECIMAL_PLACES),
        ("enroll_time_s", "enroll_time", DECIMAL_PLACES),
        ("membership_time_s", "membership_time", DECIMAL_PLACES),
        ("index_time_s", "index_time", DECIMAL_PLACES),
        ("peak_ram_gb", "peak_ram_gb", DECIMAL_PLACES),
        ("peak_disk_gb", "peak_disk_gb", DECIMAL_PLACES),
        ("peak_gpu_gb", "peak_gpu_gb", DECIMAL_PLACES),
    ]

    results = []

    for name, group in df.groupby(group_cols):
        build_type, approach, dataset_size, k_value = name
        ordered = ordered_group(group)
        stats_group = stats_window(group)

        num_pass = (group["success"] == "PASS").sum()
        num_total = len(group)

        row = {
            "build_type": build_type,
            "approach": approach,
            "dataset_size": dataset_size,
            "k_value": k_value,
            "trials": f"{num_pass}/{num_total}",
        }

        # Use only passing trials for computing statistics
        passing_trials = group[group["success"] == "PASS"]
        stats_passing_trials = stats_group[stats_group["success"] == "PASS"]

        # Keep output backward-compatible: include only metrics that exist in input CSV
        active_metric_specs = [spec for spec in summary_metric_specs if spec[0] in df.columns]

        if num_pass > 0:
            for in_col, out_col, decimals in active_metric_specs:
                if in_col in EXACT_FIRST_TRIAL_COLS:
                    # Keep the exact trial-1 value for setup metrics and disk use.
                    exact_v = ordered[in_col].dropna()
                    row[out_col] = f"{float(exact_v.iloc[0]):.{decimals}f}" if len(exact_v) > 0 else "N/A"
                    continue

                values = stats_passing_trials[in_col].dropna()
                row[out_col] = format_mean_std(values, decimals) if len(values) > 0 else "N/A"

            # Correctness (from passing trials only)
            row["membership_correct"] = (
                "PASS"
                if (passing_trials["membership_pass"] == "PASS").all()
                else "FAIL"
            )
            row["index_correct"] = (
                "PASS" if (passing_trials["index_pass"] == "PASS").all() else "FAIL"
            )
        else:
            for _, out_col, _ in active_metric_specs:
                row[out_col] = "N/A"
            row["membership_correct"] = "N/A"
            row["index_correct"] = "N/A"

        results.append(row)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(["dataset_size", "approach", "build_type"])

    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")

    return result_df


def resolve_input(path: str) -> str:
    """Resolve a single input path, extracting the CSV from a .log file if needed."""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)

    if path.endswith(".log"):
        print(f"Detected log file: {path}")
        try:
            csv_file = extract_csv_from_log(path)
            print(f"Found CSV file: {csv_file}")
            return csv_file
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate one or more benchmark CSV/log files.",
        epilog=(
            "Examples:\n"
            "  python aggregate_benchmarks.py benchmark_gpu_t10_20260110_142159.csv\n"
            "  python aggregate_benchmarks.py benchmark_Feb03.log\n"
            "  python aggregate_benchmarks.py benchmark_gpu_t10_20260110_142159.csv -o aggregated.csv\n"
            "  python aggregate_benchmarks.py benchmark_cpu_t2_*.csv benchmark_gpu_t2_*.csv -o combined_aggregated.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", help="One or more benchmark CSV or .log files (shell glob supported)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path for the full aggregation (default: <name>_aggregated.csv)",
    )
    args = parser.parse_args()

    resolved_inputs = [resolve_input(p) for p in args.inputs]

    if len(resolved_inputs) == 1:
        actual_input = resolved_inputs[0]
        base = Path(args.inputs[0]).stem
    else:
        # Multiple input files (e.g. cpu + gpu csvs, or many trial csvs from a glob):
        # merge them into a single dataframe/csv instead of silently dropping all
        # but the first, or treating the 2nd filename as an output path.
        print(f"Merging {len(resolved_inputs)} input files:")
        for f in resolved_inputs:
            print(f"  - {f}")
        merged_df = pd.concat([pd.read_csv(f) for f in resolved_inputs], ignore_index=True)
        merge_dir = os.path.dirname(os.path.abspath(resolved_inputs[0])) or "."
        actual_input = os.path.join(merge_dir, "_merged_benchmarks.csv")
        merged_df.to_csv(actual_input, index=False)
        base = "combined"

    output_file = args.output or f"{base}_aggregated.csv"

    # Also create summary file
    summary_file = output_file.replace(".csv", "_summary.csv")

    print(f"Output (full): {output_file}")
    print(f"Output (summary): {summary_file}")
    print()

    # Run aggregation (both full and summary group by build_type, so cpu/gpu
    # results for the same approach name never get averaged together)
    full_df = aggregate_benchmarks(actual_input, output_file)
    summary_df = create_summary_table(actual_input, summary_file)

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
