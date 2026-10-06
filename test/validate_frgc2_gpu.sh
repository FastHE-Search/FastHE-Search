#!/usr/bin/env bash

# FRGC2 false-positive/false-negative validation for GPU approaches 81
# (BSGS-GPU) and 812 (BSGS-GPU-PreRot). Only the index scenario is evaluated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || true)"
        if [[ -z "${PYTHON_BIN}" ]]; then
            echo "Error: no usable Python interpreter found" >&2
            exit 1
        fi
    fi
fi

exec "${PYTHON_BIN}" - "${PROJECT_ROOT}" "$@" <<'PY'
import argparse
import csv
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


DEFAULT_APPROACHES = (81, 812)
SUPPORTED_APPROACHES = frozenset(DEFAULT_APPROACHES)
VECTOR_DIM = 512
DEFAULT_THRESHOLD = 0.44


def parse_args(project_root: Path):
    parser = argparse.ArgumentParser(
        prog="./test/validate_frgc2_gpu.sh",
        description=(
            "Validate the index results of GPU approaches 81 and 812 "
            "against labeled FRGC2 data."
        ),
    )
    parser.add_argument(
        "--approaches",
        default=",".join(map(str, DEFAULT_APPROACHES)),
        help="Comma-separated approaches chosen from 81,812 (default: both)",
    )
    parser.add_argument(
        "--frgc2-dir",
        type=Path,
        default=project_root / "test" / "frgc2",
        help="Directory containing the four FRGC2 files (default: ./test/frgc2)",
    )
    parser.add_argument(
        "--query-indices",
        help="Comma-separated query indices (default: all labeled queries)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Plaintext cosine threshold (default: 0.44)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=project_root / "build",
        help="GPU build directory (default: ./build)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (default: benchmark_results/frgc2_gpu_<timestamp>)",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPU_DEVICES", "0"),
        help="Comma-separated GPU IDs (default: GPU_DEVICES or 0)",
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout budget per query process (default: 180)",
    )
    parser.add_argument(
        "--query-retries",
        type=int,
        default=2,
        help="Fresh-process retries after a timeout (default: 2)",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep generated ImageMatching input files",
    )
    return parser.parse_args()


def parse_approaches(raw: str):
    try:
        approaches = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --approaches value: {raw}") from exc
    if not approaches:
        raise SystemExit("--approaches must not be empty")
    unsupported = sorted(set(approaches) - SUPPORTED_APPROACHES)
    if unsupported:
        raise SystemExit(
            f"Unsupported approaches {unsupported}; choose from "
            f"{sorted(SUPPORTED_APPROACHES)}"
        )
    return approaches


def parse_query_indices(raw, query_count: int):
    if raw is None:
        return list(range(query_count))
    try:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --query-indices value: {raw}") from exc
    if not indices or any(index < 0 or index >= query_count for index in indices):
        raise SystemExit(
            f"--query-indices must contain values between 0 and {query_count - 1}"
        )
    return indices


def validate_gpu_list(raw: str):
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        raise SystemExit("--gpus must be a comma-separated list of non-negative IDs")
    return ",".join(parts)


def require_file(path: Path, description: str):
    if not path.is_file():
        raise SystemExit(f"Missing {description}: {path}")
    return path.resolve()


def load_dataset(frgc2_dir: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "FRGC2 validation requires NumPy in the selected Python environment."
        ) from exc

    database_path = require_file(frgc2_dir / "frgc2-db.dat", "FRGC2 database")
    query_path = require_file(frgc2_dir / "frgc2-query.dat", "FRGC2 queries")
    database_ids_path = require_file(
        frgc2_dir / "frgc2-dbid.txt", "FRGC2 database IDs"
    )
    query_ids_path = require_file(frgc2_dir / "frgc2-qid.txt", "FRGC2 query IDs")

    database_values = np.fromfile(database_path, dtype=np.float64, sep=" ")
    if database_values.size < 1:
        raise SystemExit(f"Empty FRGC2 database: {database_path}")
    database_count = int(database_values[0])
    expected_values = 1 + database_count * VECTOR_DIM
    if database_count <= 0 or database_values.size != expected_values:
        raise SystemExit(
            f"Invalid FRGC2 database: found {database_values.size} values, "
            f"expected {expected_values} for {database_count} vectors"
        )
    database = database_values[1:].reshape(database_count, VECTOR_DIM)

    query_values = np.fromfile(query_path, dtype=np.float64, sep=" ")
    if query_values.size == 0 or query_values.size % VECTOR_DIM:
        raise SystemExit(
            f"Invalid FRGC2 query file: {query_values.size} values is not a "
            f"positive multiple of {VECTOR_DIM}"
        )
    queries = query_values.reshape(-1, VECTOR_DIM)

    database_ids = np.fromfile(database_ids_path, dtype=np.int64, sep=" ")
    query_ids = np.fromfile(query_ids_path, dtype=np.int64, sep=" ")
    if database_ids.size != database_count:
        raise SystemExit(
            f"{database_ids_path} has {database_ids.size} IDs; expected "
            f"{database_count}"
        )
    if query_ids.size != queries.shape[0]:
        raise SystemExit(
            f"{query_ids_path} has {query_ids.size} IDs; expected "
            f"{queries.shape[0]}"
        )

    return database_path, database, queries, database_ids, query_ids


def plaintext_matches(database, query, threshold: float):
    import numpy as np

    products = database @ query
    denominators = np.linalg.norm(database, axis=1) * np.linalg.norm(query)
    similarities = np.divide(
        products,
        denominators,
        out=np.full(products.shape, np.nan, dtype=np.float64),
        where=denominators != 0.0,
    )
    return set(np.flatnonzero(similarities >= threshold).tolist())


def create_image_matching_input(
    database_path: Path, database_count: int, query, destination: Path
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as output, database_path.open("r") as source:
        source.readline()
        output.write(f"{database_count}\n")
        output.write(" ".join(format(value, ".17g") for value in query))
        output.write("\n")
        for row_index in range(database_count):
            row = source.readline()
            if not row:
                raise RuntimeError(
                    f"{database_path} ended at row {row_index}; "
                    f"expected {database_count} rows"
                )
            output.write(row)
    temporary.replace(destination)


def parse_found_indices(output: str):
    match = re.search(r"Index scenario:\s*\[([\d\s]*)\]", output)
    if match is None:
        raise RuntimeError("Could not find 'Index scenario: [...]' in output")
    return {int(value) for value in match.group(1).split()}


def parse_index_time(output: str):
    summaries = re.findall(
        r"^\s*\[\d{4}-\d{2}-\d{2} .*?\].*$", output, re.MULTILINE
    )
    if not summaries:
        return None
    match = re.search(r"IdxComp=([\d.]+)s", summaries[-1])
    return float(match.group(1)) if match else None


def confusion(predicted, truth, count: int):
    true_positives = len(predicted & truth)
    false_positives = len(predicted - truth)
    false_negatives = len(truth - predicted)
    true_negatives = count - true_positives - false_positives - false_negatives
    return true_positives, false_negatives, true_negatives, false_positives


def metrics(true_positives, false_negatives, true_negatives, false_positives):
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    total = true_positives + false_negatives + true_negatives + false_positives
    accuracy = (true_positives + true_negatives) / total if total else 0.0
    return precision, recall, f1, accuracy


def stream_process(command, cwd: Path, log_path: Path, timeout_seconds: float):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        captured = []
        assert process.stdout is not None

        def consume_output():
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                captured.append(line)

        reader = threading.Thread(target=consume_output)
        reader.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"ERROR: process exceeded {timeout_seconds:.0f}s; terminating it",
                file=sys.stderr,
                flush=True,
            )
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return_code = 124
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            reader.join()
            raise
        reader.join()
    return return_code, "".join(captured)


def empty_aggregate(approach: int, database_count: int, threshold: float):
    return {
        "approach": approach,
        "num_vectors": database_count,
        "threshold": threshold,
        "successful_queries": 0,
        "failed_queries": 0,
        "encrypted_TP": 0,
        "encrypted_FN": 0,
        "encrypted_TN": 0,
        "encrypted_FP": 0,
        "plaintext_TP": 0,
        "plaintext_FN": 0,
        "plaintext_TN": 0,
        "plaintext_FP": 0,
        "total_index_time_s": 0.0,
    }


def print_summary(rows):
    print("\nFRGC2 GPU validation summary:")
    print(
        f"{'Approach':>8} {'Success':>7} {'Failed':>6} "
        f"{'TP':>7} {'FN':>7} {'TN':>9} {'FP':>7} {'F1':>11}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['approach']:>8} {row['successful_queries']:>7} "
            f"{row['failed_queries']:>6} {row['encrypted_TP']:>7} "
            f"{row['encrypted_FN']:>7} {row['encrypted_TN']:>9} "
            f"{row['encrypted_FP']:>7} {row['f1']:>11}"
        )


def main():
    project_root = Path(sys.argv[1]).resolve()
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    args = parse_args(project_root)
    approaches = parse_approaches(args.approaches)
    gpu_list = validate_gpu_list(args.gpus)
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.query_timeout_seconds <= 0:
        raise SystemExit("--query-timeout-seconds must be positive")
    if args.query_retries < 0:
        raise SystemExit("--query-retries must be non-negative")

    build_dir = args.build_dir.resolve()
    binary = build_dir / "ImageMatching"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(
            f"GPU executable not found or not executable: {binary}\n"
            "Build GitHub main first with: ./build.sh gpu"
        )

    (
        database_path,
        database,
        queries,
        database_ids,
        query_ids,
    ) = load_dataset(args.frgc2_dir.resolve())
    database_count = database.shape[0]
    query_indices = parse_query_indices(args.query_indices, queries.shape[0])

    prepared_queries = []
    print(
        f"Computing plaintext baselines for {len(query_indices)} queries "
        f"against {database_count} vectors...",
        flush=True,
    )
    for query_index in query_indices:
        query = queries[query_index]
        truth = {
            int(index)
            for index in (database_ids == query_ids[query_index]).nonzero()[0]
        }
        plain = plaintext_matches(database, query, args.threshold)
        prepared_queries.append((query_index, query, truth, plain))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "benchmark_results" / f"frgc2_gpu_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "frgc2_gpu_summary.csv"
    aggregate_path = output_dir / "frgc2_gpu_aggregate.csv"

    summary_fields = [
        "query_index",
        "query_subject_id",
        "approach",
        "num_vectors",
        "threshold",
        "encrypted_TP",
        "encrypted_FN",
        "encrypted_TN",
        "encrypted_FP",
        "plaintext_TP",
        "plaintext_FN",
        "plaintext_TN",
        "plaintext_FP",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "index_time_s",
        "false_negative_indices",
        "false_positive_indices",
    ]
    failures = 0
    aggregates = {}
    serial_dir = build_dir / "serial"

    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()

        for approach in approaches:
            shutil.rmtree(serial_dir, ignore_errors=True)
            aggregate = empty_aggregate(approach, database_count, args.threshold)
            aggregates[approach] = aggregate
            print(f"\n========== GPU approach {approach} ==========")

            for query_index, query, truth, plain in prepared_queries:
                input_path = work_dir / f"frgc2_q{query_index}.dat"
                log_path = output_dir / f"frgc2_q{query_index}_a{approach}.log"
                create_image_matching_input(
                    database_path, database_count, query, input_path
                )
                command = [
                    str(binary),
                    str(input_path),
                    str(approach),
                    "--keep-serial",
                    "--gpus",
                    gpu_list,
                ]
                print(
                    f"\n--- query={query_index} approach={approach} "
                    f"expected_subject_matches={len(truth)} ---"
                )

                return_code = 124
                output = ""
                started = time.monotonic()
                for attempt in range(args.query_retries + 1):
                    if attempt:
                        print(
                            f"Retrying query {query_index} in a fresh process "
                            f"({attempt + 1}/{args.query_retries + 1})...",
                            flush=True,
                        )
                    return_code, output = stream_process(
                        command,
                        build_dir,
                        log_path,
                        args.query_timeout_seconds,
                    )
                    if return_code != 124:
                        break
                elapsed = time.monotonic() - started
                if not args.keep_work:
                    input_path.unlink(missing_ok=True)

                if return_code != 0:
                    failures += 1
                    aggregate["failed_queries"] += 1
                    print(
                        f"ERROR: approach {approach}, query {query_index} exited "
                        f"with status {return_code}; see {log_path}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    found = parse_found_indices(output)
                except RuntimeError as exc:
                    failures += 1
                    aggregate["failed_queries"] += 1
                    print(f"ERROR: {exc}; see {log_path}", file=sys.stderr)
                    continue

                encrypted_counts = confusion(found, truth, database_count)
                plaintext_counts = confusion(plain, truth, database_count)
                true_positives, false_negatives, true_negatives, false_positives = (
                    encrypted_counts
                )
                precision, recall, f1, accuracy = metrics(*encrypted_counts)
                index_time = parse_index_time(output)
                index_time = index_time if index_time is not None else elapsed
                false_negative_indices = sorted(truth - found)
                false_positive_indices = sorted(found - truth)

                aggregate["successful_queries"] += 1
                aggregate["encrypted_TP"] += true_positives
                aggregate["encrypted_FN"] += false_negatives
                aggregate["encrypted_TN"] += true_negatives
                aggregate["encrypted_FP"] += false_positives
                aggregate["plaintext_TP"] += plaintext_counts[0]
                aggregate["plaintext_FN"] += plaintext_counts[1]
                aggregate["plaintext_TN"] += plaintext_counts[2]
                aggregate["plaintext_FP"] += plaintext_counts[3]
                aggregate["total_index_time_s"] += index_time

                writer.writerow(
                    {
                        "query_index": query_index,
                        "query_subject_id": int(query_ids[query_index]),
                        "approach": approach,
                        "num_vectors": database_count,
                        "threshold": args.threshold,
                        "encrypted_TP": true_positives,
                        "encrypted_FN": false_negatives,
                        "encrypted_TN": true_negatives,
                        "encrypted_FP": false_positives,
                        "plaintext_TP": plaintext_counts[0],
                        "plaintext_FN": plaintext_counts[1],
                        "plaintext_TN": plaintext_counts[2],
                        "plaintext_FP": plaintext_counts[3],
                        "precision": f"{precision:.9f}",
                        "recall": f"{recall:.9f}",
                        "f1": f"{f1:.9f}",
                        "accuracy": f"{accuracy:.9f}",
                        "index_time_s": f"{index_time:.4f}",
                        "false_negative_indices": " ".join(
                            map(str, false_negative_indices)
                        ),
                        "false_positive_indices": " ".join(
                            map(str, false_positive_indices)
                        ),
                    }
                )
                summary_file.flush()
                print(
                    f"RESULT approach={approach} query={query_index}: "
                    f"TP={true_positives} FN={false_negatives} "
                    f"TN={true_negatives} FP={false_positives} "
                    f"precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}"
                )

            shutil.rmtree(serial_dir, ignore_errors=True)

    aggregate_fields = [
        "approach",
        "num_vectors",
        "threshold",
        "successful_queries",
        "failed_queries",
        "encrypted_TP",
        "encrypted_FN",
        "encrypted_TN",
        "encrypted_FP",
        "plaintext_TP",
        "plaintext_FN",
        "plaintext_TN",
        "plaintext_FP",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "total_index_time_s",
        "mean_index_time_s",
    ]
    aggregate_rows = []
    with aggregate_path.open("w", newline="") as aggregate_file:
        writer = csv.DictWriter(aggregate_file, fieldnames=aggregate_fields)
        writer.writeheader()
        for aggregate in aggregates.values():
            precision, recall, f1, accuracy = metrics(
                aggregate["encrypted_TP"],
                aggregate["encrypted_FN"],
                aggregate["encrypted_TN"],
                aggregate["encrypted_FP"],
            )
            successful_queries = aggregate["successful_queries"]
            total_index_time = aggregate["total_index_time_s"]
            row = {
                **aggregate,
                "precision": f"{precision:.9f}",
                "recall": f"{recall:.9f}",
                "f1": f"{f1:.9f}",
                "accuracy": f"{accuracy:.9f}",
                "total_index_time_s": f"{total_index_time:.4f}",
                "mean_index_time_s": (
                    f"{total_index_time / successful_queries:.4f}"
                    if successful_queries
                    else ""
                ),
            }
            writer.writerow(row)
            aggregate_rows.append(row)

    print_summary(aggregate_rows)
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nPer-query results: {summary_path}")
    print(f"Aggregate results: {aggregate_path}")
    print(f"Detailed logs: {output_dir}")
    if failures:
        raise SystemExit(f"Validation completed with {failures} failed run(s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
PY