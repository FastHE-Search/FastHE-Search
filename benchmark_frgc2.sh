#!/usr/bin/env bash

# FRGC2 false-positive/false-negative benchmark for GPU approaches 81 and 812,
# with optional approach 8011 regression coverage via --approaches.
#
# Full labeled FRGC2 mode requires:
#   frgc2-db.dat, frgc2-query.dat, frgc2-dbid.txt, frgc2-qid.txt
#
# The checked-in subset files contain no identities or separate queries. For
# those files, this script uses selected database rows as queries and measures
# encrypted-classifier disagreements against the plaintext cosine comparator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${SCRIPT_DIR}/venv/bin/python" ]]; then
        PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || true)"
        if [[ -z "${PYTHON_BIN}" ]]; then
            echo "Error: no usable Python interpreter found" >&2
            exit 1
        fi
        if [[ -e "${SCRIPT_DIR}/venv/bin/python" || -L "${SCRIPT_DIR}/venv/bin/python" ]]; then
            echo "Warning: ${SCRIPT_DIR}/venv/bin/python is not executable; using ${PYTHON_BIN}" >&2
        fi
    fi
fi

exec "${PYTHON_BIN}" - "${SCRIPT_DIR}" "$@" <<'PY'
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_APPROACHES = (81, 812)
SUPPORTED_APPROACHES = {81, 812, 8011}
VECTOR_DIM = 512
DEFAULT_THRESHOLD = 0.44


def parse_args(project_root: Path):
    parser = argparse.ArgumentParser(
        prog="./benchmark_frgc2.sh",
        description=(
            "Run FRGC2 false-positive/false-negative tests for GPU approaches "
            "81 and 812 (default), with optional approach 8011 coverage."
        )
    )
    parser.add_argument(
        "--approaches",
        default=",".join(map(str, DEFAULT_APPROACHES)),
        help="Comma-separated GPU approaches to test: 81, 812, 8011 (default: 81,812)",
    )
    parser.add_argument(
        "--frgc2-dir",
        type=Path,
        default=project_root.parent / "improved-hydia" / "test",
        help="Directory containing FRGC2 data (default: ../improved-hydia/test)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        action="append",
        help=(
            "Database file to test; may be repeated. If omitted, use the full "
            "FRGC2 database when available, otherwise the 1024/16384 subsets."
        ),
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Raw FRGC2 query vectors (required for labeled mode)",
    )
    parser.add_argument("--query-ids", type=Path, help="FRGC2 query identity file")
    parser.add_argument("--db-ids", type=Path, help="FRGC2 database identity file")
    parser.add_argument(
        "--query-indices",
        default="0",
        help=(
            "Comma-separated query indices (default: 0). In subset mode these "
            "are database rows; in labeled mode they index the query file."
        ),
    )
    parser.add_argument(
        "--probe-similarities",
        help=(
            "Comma-separated target cosine similarities. In unlabeled subset "
            "mode, replace each selected self-query with deterministic queries "
            "having these similarities to the selected database row. Use values "
            "near 0.44 to test comparator decision-boundary correctness."
        ),
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
        help="Result directory (default: benchmark_results/frgc2_<timestamp>)",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPU_DEVICES", "0"),
        help="Comma-separated GPU IDs (default: GPU_DEVICES or 0)",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep generated ImageMatching input files",
    )
    return parser.parse_args()


def parse_indices(raw: str):
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --query-indices value: {raw}") from exc
    if not values or any(value < 0 for value in values):
        raise SystemExit("--query-indices must contain non-negative integers")
    return values


def parse_approaches(raw: str):
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --approaches value: {raw}") from exc
    if not values:
        raise SystemExit("--approaches must not be empty")
    unsupported = sorted(set(values) - SUPPORTED_APPROACHES)
    if unsupported:
        raise SystemExit(
            f"Unsupported approaches {unsupported}; choose from {sorted(SUPPORTED_APPROACHES)}"
        )
    return values


def parse_similarities(raw):
    if raw is None:
        return []
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --probe-similarities value: {raw}") from exc
    if not values or any(not -1.0 <= value <= 1.0 for value in values):
        raise SystemExit("--probe-similarities must contain values in [-1, 1]")
    return values


def select_databases(args, project_root: Path):
    if args.db:
        return [path.resolve() for path in args.db]

    full_db = args.frgc2_dir / "frgc2-db.dat"
    if full_db.is_file():
        return [full_db.resolve()]

    subsets = [
        args.frgc2_dir / "frgc2-db-subset-1024.dat",
        args.frgc2_dir / "frgc2-db-subset-16384.dat",
    ]
    existing = [path.resolve() for path in subsets if path.is_file()]
    if existing:
        return existing

    raise SystemExit(
        f"No FRGC2 database found in {args.frgc2_dir}. Use --db to specify one."
    )


def read_database_header(path: Path):
    with path.open("r") as handle:
        first = handle.readline().strip()
    try:
        count = int(first)
    except ValueError as exc:
        raise SystemExit(f"Invalid vector-count header in {path}: {first!r}") from exc
    if count <= 0:
        raise SystemExit(f"Invalid vector count in {path}: {count}")
    return count


def iter_database_rows(path: Path, expected_count: int):
    with path.open("r") as handle:
        handle.readline()
        for index in range(expected_count):
            line = handle.readline()
            if not line:
                raise RuntimeError(
                    f"{path} ended at database row {index}; expected {expected_count} rows"
                )
            values = [float(value) for value in line.split()]
            if len(values) != VECTOR_DIM:
                raise RuntimeError(
                    f"{path} row {index} has {len(values)} values; expected {VECTOR_DIM}"
                )
            yield index, line, values
        if any(line.strip() for line in handle):
            raise RuntimeError(f"{path} contains rows beyond its declared count")


def read_self_query(path: Path, count: int, query_index: int):
    if query_index >= count:
        raise RuntimeError(
            f"Query index {query_index} is outside database {path} (N={count})"
        )
    for index, line, values in iter_database_rows(path, count):
        if index == query_index:
            return line, values
    raise RuntimeError(f"Could not read query row {query_index} from {path}")


def read_query_vectors(path: Path):
    tokens = path.read_text().split()
    if len(tokens) % VECTOR_DIM:
        raise RuntimeError(
            f"{path} contains {len(tokens)} values, not a multiple of {VECTOR_DIM}"
        )
    return [
        [float(value) for value in tokens[offset : offset + VECTOR_DIM]]
        for offset in range(0, len(tokens), VECTOR_DIM)
    ]


def query_at_cosine(anchor, target):
    anchor_norm = math.sqrt(sum(value * value for value in anchor))
    if anchor_norm == 0.0:
        raise RuntimeError("Cannot construct a cosine probe from a zero vector")
    unit_anchor = [value / anchor_norm for value in anchor]

    # Choose the coordinate least aligned with the anchor, project it onto the
    # anchor's orthogonal complement, and normalize it. The resulting query has
    # dot(query, unit_anchor) == target and norm(query) == 1.
    axis = min(range(len(unit_anchor)), key=lambda index: abs(unit_anchor[index]))
    projection = unit_anchor[axis]
    orthogonal = [-projection * value for value in unit_anchor]
    orthogonal[axis] += 1.0
    orthogonal_norm = math.sqrt(sum(value * value for value in orthogonal))
    unit_orthogonal = [value / orthogonal_norm for value in orthogonal]
    orthogonal_weight = math.sqrt(max(0.0, 1.0 - target * target))
    return [
        target * left + orthogonal_weight * right
        for left, right in zip(unit_anchor, unit_orthogonal)
    ]


def read_ids(path: Path):
    try:
        return [int(value) for value in path.read_text().split()]
    except ValueError as exc:
        raise RuntimeError(f"Identity file contains a non-integer value: {path}") from exc


def cosine_similarity(query, query_norm, candidate):
    candidate_norm_sq = sum(value * value for value in candidate)
    if query_norm == 0.0 or candidate_norm_sq == 0.0:
        return float("nan")
    dot = sum(left * right for left, right in zip(query, candidate))
    return dot / (query_norm * math.sqrt(candidate_norm_sq))


def plaintext_matches(db_path: Path, count: int, query, threshold: float):
    query_norm = math.sqrt(sum(value * value for value in query))
    matches = set()
    for index, _, candidate in iter_database_rows(db_path, count):
        similarity = cosine_similarity(query, query_norm, candidate)
        if similarity >= threshold:
            matches.add(index)
    return matches


def create_image_matching_input(db_path: Path, count: int, query, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as output, db_path.open("r") as source:
        source.readline()
        output.write(f"{count}\n")
        output.write(" ".join(format(value, ".17g") for value in query))
        output.write("\n")
        shutil.copyfileobj(source, output, length=1024 * 1024)
    temporary.replace(destination)


def parse_found_indices(output: str):
    match = re.search(r"Index scenario:\s*\[([\d\s]*)\]", output)
    if not match:
        raise RuntimeError("Could not find 'Index scenario: [...]' in program output")
    return {int(value) for value in match.group(1).split()}


def parse_index_time(output: str):
    summaries = re.findall(
        r"^\s*\[\d{4}-\d{2}-\d{2} .*?\].*$", output, re.MULTILINE
    )
    if not summaries:
        return ""
    match = re.search(r"IdxComp=([\d.]+)s", summaries[-1])
    return match.group(1) if match else ""


def confusion(predicted, truth, count):
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    tn = count - tp - fp - fn
    return tp, fn, tn, fp


def metrics(tp, fn, tn, fp):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return precision, recall, f1, accuracy


def stream_process(command, cwd: Path, log_path: Path):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            captured.append(line)
        return_code = process.wait()
    return return_code, "".join(captured)


def safe_name(path: Path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def main():
    project_root = Path(sys.argv[1]).resolve()
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    args = parse_args(project_root)
    approaches = parse_approaches(args.approaches)
    query_indices = parse_indices(args.query_indices)
    probe_similarities = parse_similarities(args.probe_similarities)
    databases = select_databases(args, project_root)
    build_dir = args.build_dir.resolve()
    binary = build_dir / "ImageMatching"

    if not binary.is_file():
        raise SystemExit(
            f"GPU executable not found: {binary}\n"
            "Build it first with ./build.sh gpu."
        )

    labeled_arguments = (args.query_file, args.query_ids, args.db_ids)
    if any(labeled_arguments) and not all(labeled_arguments):
        raise SystemExit(
            "Labeled mode requires --query-file, --query-ids, and --db-ids together."
        )

    if not any(labeled_arguments):
        auto_query = args.frgc2_dir / "frgc2-query.dat"
        auto_query_ids = args.frgc2_dir / "frgc2-qid.txt"
        auto_db_ids = args.frgc2_dir / "frgc2-dbid.txt"
        if len(databases) == 1 and all(
            path.is_file() for path in (auto_query, auto_query_ids, auto_db_ids)
        ):
            args.query_file = auto_query
            args.query_ids = auto_query_ids
            args.db_ids = auto_db_ids

    labeled_mode = all((args.query_file, args.query_ids, args.db_ids))
    if labeled_mode and probe_similarities:
        raise SystemExit("--probe-similarities is only available in unlabeled subset mode")
    query_vectors = read_query_vectors(args.query_file) if labeled_mode else None
    query_ids = read_ids(args.query_ids) if labeled_mode else None
    database_ids = read_ids(args.db_ids) if labeled_mode else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "benchmark_results" / f"frgc2_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "frgc2_fp_fn_summary.csv"
    aggregate_path = output_dir / "frgc2_fp_fn_aggregate.csv"

    fieldnames = [
        "database",
        "query_index",
        "probe_similarity",
        "query_subject_id",
        "truth_source",
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
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()

        for db_path in databases:
            if not db_path.is_file():
                raise SystemExit(f"Database not found: {db_path}")
            count = read_database_header(db_path)
            if labeled_mode and len(database_ids) < count:
                raise SystemExit(
                    f"{args.db_ids} has {len(database_ids)} IDs, but {db_path} declares {count} vectors"
                )

            prepared_queries = []
            for query_index in query_indices:
                if labeled_mode:
                    if query_index >= len(query_vectors) or query_index >= len(query_ids):
                        raise SystemExit(
                            f"Query index {query_index} exceeds the labeled query data"
                        )
                    query = query_vectors[query_index]
                    query_subject_id = query_ids[query_index]
                    truth = {
                        index
                        for index, subject_id in enumerate(database_ids[:count])
                        if subject_id == query_subject_id
                    }
                    query_cases = [(None, query, truth, "subject_id")]
                else:
                    _, anchor = read_self_query(db_path, count, query_index)
                    query_subject_id = ""
                    if probe_similarities:
                        query_cases = [
                            (
                                target,
                                query_at_cosine(anchor, target),
                                None,
                                "plaintext_threshold_probe",
                            )
                            for target in probe_similarities
                        ]
                    else:
                        query_cases = [(None, anchor, None, "plaintext_threshold")]

                for probe_similarity, query, truth, truth_source in query_cases:
                    plain = plaintext_matches(db_path, count, query, args.threshold)
                    if truth is None:
                        truth = plain
                    plain_confusion = confusion(plain, truth, count)
                    prepared_queries.append(
                        (
                            query_index,
                            probe_similarity,
                            query_subject_id,
                            query,
                            truth,
                            plain,
                            plain_confusion,
                            truth_source,
                        )
                    )

            dataset_name = safe_name(db_path)
            print(f"\nDatabase: {db_path} (N={count})")
            print(
                "Ground truth: subject IDs"
                if labeled_mode
                else "Ground truth: plaintext cosine comparator (unlabeled subset mode)"
            )

            for approach in approaches:
                serial_dir = build_dir / "serial"
                shutil.rmtree(serial_dir, ignore_errors=True)
                print(f"\n========== Approach {approach} / {dataset_name} ==========")
                aggregate = aggregates.setdefault(
                    (str(db_path), approach),
                    {
                        "database": str(db_path),
                        "truth_source": prepared_queries[0][-1],
                        "approach": approach,
                        "num_vectors": count,
                        "threshold": args.threshold,
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
                    },
                )

                for (
                    query_index,
                    probe_similarity,
                    query_subject_id,
                    query,
                    truth,
                    plain,
                    plain_confusion,
                    truth_source,
                ) in prepared_queries:
                    query_tag = f"q{query_index}"
                    if probe_similarity is not None:
                        similarity_tag = format(probe_similarity, ".6g").replace("-", "m").replace(".", "p")
                        query_tag += f"_cos{similarity_tag}"
                    input_path = work_dir / f"{dataset_name}_{query_tag}.dat"
                    create_image_matching_input(db_path, count, query, input_path)
                    log_path = output_dir / f"{dataset_name}_{query_tag}_a{approach}.log"
                    command = [
                        str(binary),
                        str(input_path),
                        str(approach),
                        "--keep-serial",
                        "--gpus",
                        args.gpus,
                    ]

                    print(
                        f"\n--- query={query_index} approach={approach} "
                        f"probe_similarity={probe_similarity if probe_similarity is not None else 'self'} "
                        f"expected_positives={len(truth)} ---"
                    )
                    started = time.monotonic()
                    return_code, output = stream_process(command, build_dir, log_path)
                    elapsed = time.monotonic() - started
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

                    encrypted_confusion = confusion(found, truth, count)
                    tp, fn, tn, fp = encrypted_confusion
                    precision, recall, f1, accuracy = metrics(tp, fn, tn, fp)
                    false_negatives = sorted(truth - found)
                    false_positives = sorted(found - truth)
                    plain_tp, plain_fn, plain_tn, plain_fp = plain_confusion
                    index_time = parse_index_time(output)
                    index_time_value = float(index_time) if index_time else elapsed

                    aggregate["successful_queries"] += 1
                    aggregate["encrypted_TP"] += tp
                    aggregate["encrypted_FN"] += fn
                    aggregate["encrypted_TN"] += tn
                    aggregate["encrypted_FP"] += fp
                    aggregate["plaintext_TP"] += plain_tp
                    aggregate["plaintext_FN"] += plain_fn
                    aggregate["plaintext_TN"] += plain_tn
                    aggregate["plaintext_FP"] += plain_fp
                    aggregate["total_index_time_s"] += index_time_value

                    writer.writerow(
                        {
                            "database": str(db_path),
                            "query_index": query_index,
                            "probe_similarity": (
                                f"{probe_similarity:.9f}" if probe_similarity is not None else ""
                            ),
                            "query_subject_id": query_subject_id,
                            "truth_source": truth_source,
                            "approach": approach,
                            "num_vectors": count,
                            "threshold": args.threshold,
                            "encrypted_TP": tp,
                            "encrypted_FN": fn,
                            "encrypted_TN": tn,
                            "encrypted_FP": fp,
                            "plaintext_TP": plain_tp,
                            "plaintext_FN": plain_fn,
                            "plaintext_TN": plain_tn,
                            "plaintext_FP": plain_fp,
                            "precision": f"{precision:.9f}",
                            "recall": f"{recall:.9f}",
                            "f1": f"{f1:.9f}",
                            "accuracy": f"{accuracy:.9f}",
                            "index_time_s": f"{index_time_value:.4f}",
                            "false_negative_indices": " ".join(map(str, false_negatives)),
                            "false_positive_indices": " ".join(map(str, false_positives)),
                        }
                    )
                    summary_file.flush()
                    print(
                        f"RESULT approach={approach} query={query_index}: "
                        f"TP={tp} FN={fn} TN={tn} FP={fp} "
                        f"precision={precision:.6f} recall={recall:.6f} f1={f1:.6f}"
                    )
                    if false_negatives:
                        print(f"  False negatives: {false_negatives}")
                    if false_positives:
                        print(f"  False positives: {false_positives}")

                shutil.rmtree(serial_dir, ignore_errors=True)

    aggregate_fieldnames = [
        "database",
        "truth_source",
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
    with aggregate_path.open("w", newline="") as aggregate_file:
        writer = csv.DictWriter(aggregate_file, fieldnames=aggregate_fieldnames)
        writer.writeheader()
        for aggregate in aggregates.values():
            precision, recall, f1, accuracy = metrics(
                aggregate["encrypted_TP"],
                aggregate["encrypted_FN"],
                aggregate["encrypted_TN"],
                aggregate["encrypted_FP"],
            )
            successful = aggregate["successful_queries"]
            total_index_time = aggregate["total_index_time_s"]
            writer.writerow(
                {
                    **aggregate,
                    "precision": f"{precision:.9f}",
                    "recall": f"{recall:.9f}",
                    "f1": f"{f1:.9f}",
                    "accuracy": f"{accuracy:.9f}",
                    "total_index_time_s": f"{total_index_time:.4f}",
                    "mean_index_time_s": (
                        f"{total_index_time / successful:.4f}" if successful else ""
                    ),
                }
            )

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nPer-query results: {summary_path}")
    print(f"Aggregate results: {aggregate_path}")
    print(f"Detailed logs: {output_dir}")
    if failures:
        raise SystemExit(f"Benchmark completed with {failures} failed run(s)")



if __name__ == "__main__":
    main()
PY