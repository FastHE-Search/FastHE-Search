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
Benchmark script for FastHE-Search - measures timing and resource usage for all approaches.

Usage:
    python3 benchmark_all.py [logn] [kmatch]

    logn   - Power of 2 for dataset size (default: 10, means 2^10 = 1024)
    kmatch - Number of matching vectors (default: 10)

Outputs a clean CSV file with:
- Build type (OpenFHE version)
- Approach name
- Dataset size
- Wall-clock time
- Membership time
- Index time
- Peak RAM (GB)
- Peak Disk (GB)
- Peak GPU VRAM (GB)
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import re
import threading
import csv
import argparse
import queue
from datetime import datetime

try:
    import psutil
except ImportError:
    print("❌ Error: psutil is not installed")
    print("Please install it with: pip3 install psutil")
    sys.exit(1)

# GPU memory monitoring (optional - gracefully degrades)
_nvml_available = False
try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _nvml_available = True
except Exception:
    _nvml_handle = None

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = Path(os.environ.get("BUILD_DIR", PROJECT_ROOT / "build"))
TEST_DATA_DIR = Path(os.environ.get("TEST_DATA_DIR", PROJECT_ROOT / "test"))
DATASET_GEN_SCRIPT = PROJECT_ROOT / "tools" / "gen_dataset.py"
IMAGE_MATCHING_BIN = BUILD_DIR / "ImageMatching"

# Benchmark configuration
CPU_APPROACHES = [5, 8]  # Approaches available in CPU build (OpenFHE v1.3.0)
GPU_APPROACHES = [
    51,
    81,
    812,
]  # Approaches available in GPU build (OpenFHE v1.4.2 optimized)

# Output order for results
OUTPUT_ORDER = [5, 8, 51, 81, 812]

# BSGS parameters (for approaches 6, 7, 8, 9)
BSGS_MULT_DEPTH = 11
BSGS_SCALE_FACTOR = 45

# Build type display names
BUILD_TYPE_NAMES = {"CPU": "OpenFHE_v1.3.0", "GPU": "OpenFHE_v1.4.2_optimized"}

# Approach full names
APPROACH_NAMES = {
    5: "FastHE-Search_CPU",
    6: "BSGS-Orig (CPU)",
    7: "BSGS-Precomp (CPU)",
    8: "BSGS-Precomp-Opt (CPU)",
    9: "BSGS-OnlineAgg (CPU)",
    51: "FastHE-Search-GPU",
    81: "BSGS-GPU",
    812: "BSGS-GPU-PreRot",
}

# Dataset configuration (defaults, can be overridden by command line args)
DATASET_SIZE = 1024  # 2^10
DATASET_K = 10  # Number of matches


def parse_approaches(approaches_str):
    """Parse approaches string like '[5,6,11]' or '5,6,11' into a list of ints."""
    if not approaches_str:
        return None
    # Remove brackets and whitespace
    cleaned = approaches_str.strip().strip("[]").strip()
    if not cleaned:
        return None
    # Split by comma and convert to ints
    try:
        return [int(x.strip()) for x in cleaned.split(",")]
    except ValueError:
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FastHE-Search Benchmark Suite - measures timing and resource usage for all approaches"
    )
    parser.add_argument(
        "logn",
        type=int,
        nargs="?",
        default=10,
        help="Power of 2 for dataset size (default: 10, means 2^10 = 1024)",
    )
    parser.add_argument(
        "kmatch",
        type=int,
        nargs="?",
        default=10,
        help="Number of matching vectors (default: 10)",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default=None,
        help="Comma-separated list of approaches to run, e.g., '[5,6,8]' or '5,6,8'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for dataset generation (for reproducibility)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of expected matching indices (skip dataset generation)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file path (will append if exists)",
    )
    parser.add_argument(
        "--trial", type=int, default=1, help="Trial number for this run (default: 1)"
    )
    return parser.parse_args()


class ResourceMonitor:
    """Monitor RAM, disk, and optionally GPU VRAM usage during test execution."""

    def __init__(self, pid, interval=0.5, monitor_gpu=False):
        self.pid = pid
        self.interval = interval
        self.peak_ram_gb = 0
        self.peak_disk_gb = 0
        self.peak_gpu_gb = 0
        self.monitor_gpu = monitor_gpu and _nvml_available
        self.monitoring = False
        self.thread = None
        self.initial_disk_gb = 0
        self.initial_gpu_gb = 0

    def _get_disk_usage(self):
        """Get current disk usage of serial directory in GB."""
        serial_dir = BUILD_DIR / "serial"
        if not serial_dir.exists():
            return 0.0
        total_size = 0
        try:
            for dirpath, _, filenames in os.walk(serial_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            pass
        return total_size / (1024**3)

    def _monitor(self):
        """Monitor resources in background thread."""
        try:
            process = psutil.Process(self.pid)
            self.initial_disk_gb = self._get_disk_usage()
            if self.monitor_gpu:
                info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                self.initial_gpu_gb = info.used / (1024**3)

            while self.monitoring:
                try:
                    mem_info = process.memory_info()
                    ram_gb = mem_info.rss / (1024**3)
                    self.peak_ram_gb = max(self.peak_ram_gb, ram_gb)

                    current_disk = self._get_disk_usage()
                    disk_used_gb = current_disk - self.initial_disk_gb
                    self.peak_disk_gb = max(self.peak_disk_gb, disk_used_gb)

                    if self.monitor_gpu:
                        info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                        gpu_used_gb = info.used / (1024**3) - self.initial_gpu_gb
                        self.peak_gpu_gb = max(self.peak_gpu_gb, gpu_used_gb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(self.interval)
        except Exception as e:
            print(f"[WARNING] Monitoring error: {e}")

    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        gpu = self.peak_gpu_gb if self.monitor_gpu else None
        return self.peak_ram_gb, self.peak_disk_gb, gpu


def detect_build_type():
    """Detect if current build is CPU or GPU."""
    if not IMAGE_MATCHING_BIN.exists():
        return None

    # Check if binary has GPU support by running with --help or checking linked libs
    try:
        result = subprocess.run(
            ["ldd", str(IMAGE_MATCHING_BIN)], capture_output=True, text=True
        )
        if "libcudart" in result.stdout or "cuda" in result.stdout.lower():
            return "GPU"
        return "CPU"
    except Exception:
        return "Unknown"


def generate_dataset(size, k, seed=None, force_regenerate=False):
    """Generate a test dataset and return expected indices.

    If seed is provided, use it for reproducible random generation.
    If force_regenerate is True, always regenerate even if file exists.
    This ensures we have the correct expected_indices.
    """
    filename = f"test_{size}_k{k}.dat"
    filepath = TEST_DATA_DIR / filename

    if filepath.exists() and not force_regenerate:
        # File exists, but we don't know the expected indices
        # We must regenerate to get them
        print(
            f"  📁 Dataset file exists: {filename}, regenerating to get expected indices..."
        )

    print(f"  📝 Generating dataset: {filename}")

    # Pass full path to gen_dataset.py
    cmd = [sys.executable, str(DATASET_GEN_SCRIPT), str(filepath), str(size), str(k)]
    if seed is not None:
        cmd.append(str(seed))

    # Run from tools directory so gen_dataset.py can find its relative path if needed
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=DATASET_GEN_SCRIPT.parent
    )

    if result.returncode != 0:
        raise RuntimeError(f"Dataset generation failed: {result.stderr}")

    # Extract matching positions from generator output
    expected_indices = []
    for line in result.stdout.split("\n"):
        if "Matching vectors will be at positions:" in line:
            match = re.search(r"\[([\d,\s]+)\]", line)
            if match:
                expected_indices = [int(x.strip()) for x in match.group(1).split(",")]

    return filepath, expected_indices


def parse_output(output):
    """Parse timing and correctness information from ImageMatching output."""
    result = {
        "enroll": 0.0,
        "membership": 0.0,
        "index": 0.0,
        "setup_keygen": 0.0,
        "setup_rotkeygen": 0.0,
        "setup_db_encrypt": 0.0,
        "setup_diag_prerot": 0.0,
        "setup_gpu_key_upload": 0.0,
        "setup_gpu_db_cache": 0.0,
        "setup_offline_total": 0.0,
        "setup_total": 0.0,
        "setup_costs_found": False,
        "fatal_error": False,
        "membership_result": None,  # True/False/None
        "found_indices": [],
    }

    result["fatal_error"] = bool(re.search(r"FATAL:\s", output))

    # Enrollment time: "encrypt DB vectors done (32.6528s)"
    enroll_match = re.search(r"encrypt DB vectors done\s*\(([\d.]+)s\)", output)
    if enroll_match:
        result["enroll"] = float(enroll_match.group(1))

    # Setup/offline costs summary (structured line from ImageMatching).
    setup_costs = re.search(
        r"\[SETUP_COSTS\].*?"
        r"KeyGen=([\d.]+)s.*?"
        r"RotKeyGen=([\d.]+)s.*?"
        r"DBEncrypt=([\d.]+)s.*?"
        r"DiagPreRot=([\d.]+)s.*?"
        r"GPUKeyUpload=([\d.]+)s.*?"
        r"GPUDBCache=([\d.]+)s.*?"
        r"OfflineTotal=([\d.]+)s.*?"
        r"SetupTotal=([\d.]+)s",
        output,
        re.DOTALL,
    )
    if setup_costs:
        result["setup_keygen"] = float(setup_costs.group(1))
        result["setup_rotkeygen"] = float(setup_costs.group(2))
        result["setup_db_encrypt"] = float(setup_costs.group(3))
        result["setup_diag_prerot"] = float(setup_costs.group(4))
        result["setup_gpu_key_upload"] = float(setup_costs.group(5))
        result["setup_gpu_db_cache"] = float(setup_costs.group(6))
        result["setup_offline_total"] = float(setup_costs.group(7))
        result["setup_total"] = float(setup_costs.group(8))
        result["setup_costs_found"] = True

    # Fallback for runs that abort before printing [SETUP_COSTS] but still
    # complete expensive DB encryption/setup work.
    if not result["setup_costs_found"] and result["enroll"] > 0.0:
        result["setup_db_encrypt"] = result["enroll"]
        result["setup_offline_total"] = result["enroll"]
        result["setup_total"] = result["enroll"]

    # PRIMARY: Parse from final timing summary string (most reliable)
    # Format: "[TIMESTAMP] DB=2^14 Approach=81 N=512 EncQuery=0.0896s MembComp=0.7915s MembDec=0.0259s IdxComp=0.4729s IdxDec=0.0251s MembResult=SUCCESS"
    timing_summary = re.search(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\].*?"
        r"MembComp=([\d.]+)s.*?"
        r"IdxComp=([\d.]+)s.*?"
        r"MembResult=(\w+)",
        output,
    )
    if timing_summary:
        result["membership"] = float(timing_summary.group(1))
        result["index"] = float(timing_summary.group(2))
        result["membership_result"] = timing_summary.group(3).upper() == "SUCCESS"
    else:
        # FALLBACK: Use legacy parsing if timing summary not found
        # Membership time - multiple formats
        membership_total = re.search(r"Membership scenario total:\s*([\d.]+)s", output)
        if membership_total:
            result["membership"] = float(membership_total.group(1))
        else:
            # Match "done (wall: X.XXs)" format
            membership_done = re.search(
                r"Computing membership scenario.*?done\s*\((?:wall:\s*)?([\d.]+)s\)",
                output,
                re.DOTALL,
            )
            if membership_done:
                result["membership"] = float(membership_done.group(1))

        # Index time - multiple formats
        index_total = re.search(r"Index scenario total:\s*([\d.]+)s", output)
        if index_total:
            result["index"] = float(index_total.group(1))
        else:
            # Match "done (wall: X.XXs)" format
            index_done = re.search(
                r"Computing index scenario.*?done\s*\((?:wall:\s*)?([\d.]+)s\)",
                output,
                re.DOTALL,
            )
            if index_done:
                result["index"] = float(index_done.group(1))

        # Parse membership result - look for "Membership scenario: true" or "false"
        membership_match = re.search(
            r"Membership scenario:\s*(true|false)", output, re.IGNORECASE
        )
        if membership_match:
            result["membership_result"] = membership_match.group(1).lower() == "true"

    # Parse index results - look for "Index scenario: [ ... ]"
    index_match = re.search(r"Index scenario:\s*\[([\d\s]*)\]", output)
    if index_match:
        numbers_str = index_match.group(1).strip()
        if numbers_str:
            result["found_indices"] = [int(n) for n in numbers_str.split()]

    return result


def clean_serial_dir():
    """Clean the serial directory for a fresh run."""
    import shutil

    serial_dir = BUILD_DIR / "serial"
    if serial_dir.exists():
        # Use ignore_errors to handle race conditions where files are deleted
        # by the ImageMatching process while we're cleaning up
        shutil.rmtree(serial_dir, ignore_errors=True)


def clean_serial_db_only():
    """Remove only serialized DB payloads while keeping key material."""
    import shutil

    serial_dir = BUILD_DIR / "serial"
    if not serial_dir.exists():
        return

    for path in serial_dir.iterdir():
        if path.name.startswith("db"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def serial_signature_path():
    return BUILD_DIR / "serial" / ".benchmark_signature"


def get_serial_signature(approach):
    """Return a conservative compatibility signature for serialized artifacts."""
    return f"approach={approach}"


def read_serial_signature():
    signature_file = serial_signature_path()
    if not signature_file.exists():
        return None
    try:
        return signature_file.read_text().strip() or None
    except OSError:
        return None


def write_serial_signature(signature):
    serial_dir = BUILD_DIR / "serial"
    if not serial_dir.exists():
        return
    try:
        serial_signature_path().write_text(f"{signature}\n")
    except OSError:
        pass


def serial_keep_marker_path():
    return BUILD_DIR / ".keep_serial"


def set_serial_keep_marker(enabled):
    marker = serial_keep_marker_path()
    if enabled:
        marker.write_text("keep\n")
    elif marker.exists():
        marker.unlink()


def env_flag_enabled(name, default=False):
    """Interpret common truthy/falsey environment variable values."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_benchmark(approach, expected_indices, mult_depth=None, scale_factor=None):
    """Run a single benchmark and return results."""
    # Use the existing dataset file - do NOT regenerate!
    filename = f"test_{DATASET_SIZE}_k{DATASET_K}.dat"
    dataset_path = TEST_DATA_DIR / filename

    # Always use absolute path since we run from BUILD_DIR, not PROJECT_ROOT
    data_path_arg = str(dataset_path.resolve())

    cmd = [str(IMAGE_MATCHING_BIN), data_path_arg, str(approach)]
    if mult_depth is not None and scale_factor is not None:
        cmd.extend([str(mult_depth), str(scale_factor)])

    print(f"    Running: {' '.join(str(c) for c in cmd[-3:])}")

    clean_serial_before_run = env_flag_enabled("CLEAN_SERIAL", default=True)
    keep_serial_after_run = env_flag_enabled("KEEP_SERIAL", default=False)
    reuse_keys_only = env_flag_enabled("REUSE_KEYS_ONLY", default=False)
    serial_signature = get_serial_signature(approach)

    if clean_serial_before_run:
        if reuse_keys_only:
            clean_serial_db_only()
        else:
            clean_serial_dir()
    else:
        previous_signature = read_serial_signature()
        serial_dir = BUILD_DIR / "serial"
        if serial_dir.exists() and previous_signature != serial_signature:
            previous_label = previous_signature or "unmarked-serial"
            print(
                "    Incompatible serialized data detected "
                f"({previous_label} -> {serial_signature}); cleaning serial/"
            )
            clean_serial_dir()
        else:
            print("    Reusing serialized data from previous trial")

    set_serial_keep_marker(keep_serial_after_run)
    if keep_serial_after_run:
        cmd.append("--keep-serial")
    if reuse_keys_only:
        cmd.append("--reuse-keys")

    start_time = time.time()
    child_env = os.environ.copy()
    if keep_serial_after_run:
        child_env["KEEP_SERIAL"] = "1"
    else:
        child_env.pop("KEEP_SERIAL", None)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=BUILD_DIR,  # Run from BUILD_DIR so serial/ is created there
        env=child_env,
    )

    is_gpu_approach = approach >= 50
    monitor = ResourceMonitor(process.pid, monitor_gpu=is_gpu_approach)
    monitor.start()

    output_queue = queue.Queue()
    combined_lines = []

    def _pump_output(pipe, target_queue):
        try:
            for line in iter(pipe.readline, ""):
                target_queue.put(line)
        finally:
            pipe.close()
            target_queue.put(None)

    reader = threading.Thread(
        target=_pump_output, args=(process.stdout, output_queue), daemon=True
    )
    reader.start()

    saw_output = False
    reader_done = False
    while not reader_done:
        try:
            line = output_queue.get(timeout=0.2)
        except queue.Empty:
            # Keep waiting while the reader thread may still have buffered lines
            # to enqueue. This avoids losing trailing output right after process exit.
            if process.poll() is not None and not reader.is_alive():
                break
            continue

        if line is None:
            reader_done = True
            continue

        saw_output = True
        combined_lines.append(line)
        print(f"    | {line.rstrip()}", flush=True)

    process.wait()
    reader.join(timeout=1)
    wall_time = time.time() - start_time
    stdout = "".join(combined_lines)

    peak_ram, peak_disk, peak_gpu = monitor.stop()

    if not saw_output:
        print(
            f"    | [no subprocess output captured for approach {approach}]", flush=True
        )

    parsed = parse_output(stdout)

    # Treat explicit fatal runtime messages as benchmark failure even if the
    # process exits with code 0.
    success = process.returncode == 0 and not parsed["fatal_error"]

    if not success:
        fail_reason = "fatal runtime error" if parsed["fatal_error"] else "non-zero exit"
        print(f"    ❌ Failed ({fail_reason}): {stdout[:200]}")
        return {
            "wall_time": wall_time,
            "enroll_time": parsed["enroll"],
            "membership_time": parsed["membership"],
            "index_time": parsed["index"],
            "setup_keygen_s": parsed["setup_keygen"],
            "setup_rotkeygen_s": parsed["setup_rotkeygen"],
            "setup_db_encrypt_s": parsed["setup_db_encrypt"],
            "setup_diag_prerot_s": parsed["setup_diag_prerot"],
            "setup_gpu_key_upload_s": parsed["setup_gpu_key_upload"],
            "setup_gpu_db_cache_s": parsed["setup_gpu_db_cache"],
            "setup_offline_total_s": parsed["setup_offline_total"],
            "setup_total_s": parsed["setup_total"],
            "peak_ram_gb": peak_ram,
            "peak_disk_gb": peak_disk,
            "peak_gpu_gb": peak_gpu,
            "success": False,
            "membership_pass": False,
            "index_pass": False,
            "missed_indices": list(expected_indices),
            "found_indices": parsed["found_indices"],
        }

    write_serial_signature(serial_signature)

    # Remove the marker after a non-reuse run so later unrelated invocations
    # start from a clean default state.
    if not keep_serial_after_run:
        set_serial_keep_marker(False)

    # Check membership correctness (should be true if k > 0)
    membership_pass = parsed["membership_result"] == (DATASET_K > 0)

    # Check index correctness (all expected indices should be found)
    found_set = set(parsed["found_indices"])
    expected_set = set(expected_indices)
    missed_indices = sorted(expected_set - found_set)
    index_pass = len(missed_indices) == 0 and len(found_set) == len(expected_set)

    return {
        "wall_time": wall_time,
        "enroll_time": parsed["enroll"],
        "membership_time": parsed["membership"],
        "index_time": parsed["index"],
        "setup_keygen_s": parsed["setup_keygen"],
        "setup_rotkeygen_s": parsed["setup_rotkeygen"],
        "setup_db_encrypt_s": parsed["setup_db_encrypt"],
        "setup_diag_prerot_s": parsed["setup_diag_prerot"],
        "setup_gpu_key_upload_s": parsed["setup_gpu_key_upload"],
        "setup_gpu_db_cache_s": parsed["setup_gpu_db_cache"],
        "setup_offline_total_s": parsed["setup_offline_total"],
        "setup_total_s": parsed["setup_total"],
        "peak_ram_gb": peak_ram,
        "peak_disk_gb": peak_disk,
        "peak_gpu_gb": peak_gpu,
        "success": True,
        "membership_pass": membership_pass,
        "index_pass": index_pass,
        "missed_indices": missed_indices,
        "found_indices": parsed["found_indices"],
    }


def run_all_benchmarks(build_type, approaches, expected_indices, trial=1):
    """Run benchmarks for all approaches."""
    results = []

    for approach in approaches:
        approach_name = APPROACH_NAMES.get(approach, f"Approach_{approach}")
        print(f"\n  🔧 {approach_name} (Approach {approach}):")

        # Determine if approach needs BSGS params
        if approach in [6, 7, 8, 9]:
            result = run_benchmark(
                approach, expected_indices, BSGS_MULT_DEPTH, BSGS_SCALE_FACTOR
            )
        else:
            result = run_benchmark(approach, expected_indices)

        if result:
            result["trial"] = trial
            result["build_type"] = BUILD_TYPE_NAMES.get(build_type, build_type)
            result["approach"] = approach_name
            result["dataset_size"] = DATASET_SIZE
            result["k_value"] = DATASET_K
            results.append(result)

            status = "✅" if result["success"] else "❌"
            mem_status = "✓" if result["membership_pass"] else "✗"
            idx_status = "✓" if result["index_pass"] else "✗"

            print(
                f"    {status} Wall: {result['wall_time']:.4f}s | "
                f"Enroll: {result['enroll_time']:.4f}s | "
                f"Membership: {result['membership_time']:.4f}s | "
                f"Index: {result['index_time']:.4f}s"
            )
            gpu_str = (
                f" | GPU: {result['peak_gpu_gb']:.4f} GB"
                if result["peak_gpu_gb"] is not None
                else ""
            )
            print(
                f"       RAM: {result['peak_ram_gb']:.4f} GB | "
                f"Disk: {result['peak_disk_gb']:.4f} GB{gpu_str} | "
                f"Membership: {mem_status} | Index: {idx_status}"
            )

            # Print missed indices if any
            if result["missed_indices"]:
                print(
                    f"       ⚠️  Missed {len(result['missed_indices'])} indices: {result['missed_indices'][:10]}{'...' if len(result['missed_indices']) > 10 else ''}"
                )

    return results


def write_results(results, output_file, append=False):
    """Write results to CSV file. If append=True, adds to existing file."""
    fieldnames = [
        "trial",
        "build_type",
        "approach",
        "dataset_size",
        "k_value",
        "success",
        "membership_pass",
        "index_pass",
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
        "peak_gpu_gb",
    ]

    # Check if file exists and has content (for append mode)
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    mode = "a" if append and file_exists else "w"
    write_header = not (append and file_exists)

    with open(output_file, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "trial": r["trial"],
                    "build_type": r["build_type"],
                    "approach": r["approach"],
                    "dataset_size": r["dataset_size"],
                    "k_value": r["k_value"],
                    "success": "PASS" if r["success"] else "FAIL",
                    "membership_pass": "PASS" if r["membership_pass"] else "FAIL",
                    "index_pass": "PASS" if r["index_pass"] else "FAIL",
                    "setup_keygen_s": f"{r['setup_keygen_s']:.4f}",
                    "setup_rotkeygen_s": f"{r['setup_rotkeygen_s']:.4f}",
                    "setup_db_encrypt_s": f"{r['setup_db_encrypt_s']:.4f}",
                    "setup_diag_prerot_s": f"{r['setup_diag_prerot_s']:.4f}",
                    "setup_gpu_key_upload_s": f"{r['setup_gpu_key_upload_s']:.4f}",
                    "setup_gpu_db_cache_s": f"{r['setup_gpu_db_cache_s']:.4f}",
                    "setup_offline_total_s": f"{r['setup_offline_total_s']:.4f}",
                    "setup_total_s": f"{r['setup_total_s']:.4f}",
                    "wall_time_s": f"{r['wall_time']:.4f}",
                    "enroll_time_s": f"{r['enroll_time']:.4f}",
                    "membership_time_s": f"{r['membership_time']:.4f}",
                    "index_time_s": f"{r['index_time']:.4f}",
                    "peak_ram_gb": f"{r['peak_ram_gb']:.4f}",
                    "peak_disk_gb": f"{r['peak_disk_gb']:.4f}",
                    "peak_gpu_gb": (
                        f"{r['peak_gpu_gb']:.4f}"
                        if r["peak_gpu_gb"] is not None
                        else ""
                    ),
                }
            )


def print_summary(results):
    """Print a nice summary table."""
    print("\n" + "=" * 140)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 140)
    print(
        f"{'Build Type':<26} {'Approach':<18} {'Success':<8} {'Mem':<5} {'Idx':<5} {'Wall(s)':<9} {'Enroll(s)':<10} {'Member(s)':<10} {'Index(s)':<10} {'RAM(GB)':<8} {'Disk(GB)':<9} {'GPU(GB)':<8}"
    )
    print("-" * 140)

    for r in results:
        success = "✓" if r["success"] else "✗"
        mem = "✓" if r["membership_pass"] else "✗"
        idx = "✓" if r["index_pass"] else "✗"
        gpu_val = f"{r['peak_gpu_gb']:.4f}" if r["peak_gpu_gb"] is not None else "-"
        print(
            f"{r['build_type']:<26} {r['approach']:<18} "
            f"{success:<8} {mem:<5} {idx:<5} "
            f"{r['wall_time']:<9.4f} {r['enroll_time']:<10.4f} "
            f"{r['membership_time']:<10.4f} {r['index_time']:<10.4f} "
            f"{r['peak_ram_gb']:<8.4f} {r['peak_disk_gb']:<9.4f} {gpu_val:<8}"
        )

    print("=" * 140)


def print_setup_costs_summary(results):
    """Print separate setup/offline costs table."""
    print("\n" + "=" * 156)
    print("🧱 SETUP/OFFLINE COSTS SUMMARY")
    print("=" * 156)
    print(
        f"{'Build Type':<26} {'Approach':<18} {'KeyGen(s)':<10} {'RotKey(s)':<10} {'DBEnc(s)':<10} {'PreRot(s)':<10} {'GPUKeyUp(s)':<12} {'GPUCache(s)':<11} {'Offline(s)':<11} {'SetupTotal(s)':<13}"
    )
    print("-" * 156)

    for r in results:
        print(
            f"{r['build_type']:<26} {r['approach']:<18} "
            f"{r['setup_keygen_s']:<10.4f} {r['setup_rotkeygen_s']:<10.4f} "
            f"{r['setup_db_encrypt_s']:<10.4f} {r['setup_diag_prerot_s']:<10.4f} "
            f"{r['setup_gpu_key_upload_s']:<12.4f} {r['setup_gpu_db_cache_s']:<11.4f} "
            f"{r['setup_offline_total_s']:<11.4f} {r['setup_total_s']:<13.4f}"
        )

    print("=" * 156)


def main():
    global DATASET_SIZE, DATASET_K

    # Parse command line arguments
    args = parse_args()
    DATASET_SIZE = 2**args.logn
    DATASET_K = args.kmatch

    # Parse custom approaches if provided
    custom_approaches = parse_approaches(args.approaches)

    # Validate K
    if DATASET_K < 1 or DATASET_K >= DATASET_SIZE:
        print(f"❌ Error: kmatch must be between 1 and {DATASET_SIZE - 1}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🚀 FastHE-Search Benchmark Suite")
    print("=" * 70)

    # Detect build type
    build_type = detect_build_type()
    if build_type is None:
        print("❌ ImageMatching binary not found. Please build first:")
        print("   ./build.sh cpu   OR   ./build.sh gpu")
        sys.exit(1)

    print(
        f"\n📦 Detected build type: {build_type} ({BUILD_TYPE_NAMES.get(build_type, build_type)})"
    )
    print(f"📊 Dataset: 2^{args.logn} = {DATASET_SIZE} vectors, k={DATASET_K}")

    # Handle indices: either use provided indices or generate dataset
    if args.indices:
        # Use pre-computed indices (for "both" mode coordination)
        expected_indices = [int(x.strip()) for x in args.indices.split(",")]
        # Dataset should already exist in test/ directory
        filename = f"test_{DATASET_SIZE}_k{DATASET_K}.dat"
        dataset_path = TEST_DATA_DIR / filename
        if not dataset_path.exists():
            print(f"❌ Error: Dataset file not found: {dataset_path}")
            print("   When using --indices, the dataset must already exist.")
            sys.exit(1)
        print(f"\n📝 Using existing dataset...")
        print(f"   Dataset: {dataset_path.name}")
        print(f"   Pre-computed matching indices: {sorted(expected_indices)}")
    else:
        # Generate dataset (optionally with seed for reproducibility)
        print(f"\n📝 Preparing dataset...")
        dataset_path, expected_indices = generate_dataset(
            DATASET_SIZE, DATASET_K, seed=args.seed
        )
        print(f"   Dataset ready: {dataset_path.name}")
        print(f"   Expected matching indices: {sorted(expected_indices)}")

        # Always write indices to a file for coordination between trials
        indices_file = PROJECT_ROOT / "data" / "last_indices.txt"
        indices_file.parent.mkdir(parents=True, exist_ok=True)
        indices_file.write_text(",".join(str(i) for i in expected_indices))

        # If seed was provided, also report it
        if args.seed is not None:
            print(f"   Seed used: {args.seed}")
        print(f"   Indices saved to: {indices_file}")

    # Select approaches based on build type or custom list
    if custom_approaches:
        approaches = custom_approaches
        print(f"🔧 Using custom approaches: {approaches}")
    elif build_type == "GPU":
        approaches = GPU_APPROACHES
    else:
        approaches = CPU_APPROACHES

    print(f"🔧 Testing approaches: {approaches}")
    print(f"🔧 Trial: {args.trial}")

    # Run benchmarks
    results = run_all_benchmarks(
        build_type, approaches, expected_indices, trial=args.trial
    )

    if not results:
        print("\n❌ No successful benchmarks")
        sys.exit(1)

    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        append_mode = True  # Append to existing file if provided
    else:
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{build_type.lower()}_n{DATASET_SIZE}_k{DATASET_K}_{timestamp}.csv"
        output_dir = PROJECT_ROOT / "benchmark_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename
        append_mode = False

    # Write results (append if file was provided, else create new)
    write_results(results, output_file, append=append_mode)
    print(
        f"\n📄 Results {'appended to' if append_mode else 'written to'}: {output_file}"
    )

    # Print summary
    print_summary(results)
    print_setup_costs_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
