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

import sys
import random
from pathlib import Path

# ---------- constants ----------
DIM = 512


# ---------- usage ----------
def usage():
    print("Usage: gen_dataset.py [FILEPATH] [SIZE] [K] [SEED (optional)]")
    print("\nParameters:")
    print("\tFILEPATH\tPath to create dataset (absolute or relative)")
    print("\tSIZE    \tInteger number of database vectors")
    print("\tK       \tNumber of matching vectors (1 <= K < SIZE)")
    print("\tSEED    \tOptional random seed for reproducibility")
    sys.exit(1)


# ---------- main ----------
if len(sys.argv) < 4 or len(sys.argv) > 5:
    usage()

filepath_arg = sys.argv[1]
size_str = sys.argv[2]
k_str = sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) == 5 else None

if not size_str.isdigit() or not k_str.isdigit():
    usage()

SIZE = int(size_str)
K = int(k_str)

# Set random seed if provided
if seed is not None:
    random.seed(seed)
    print(f"Using random seed: {seed}")

# Validate K
if K < 1 or K >= SIZE:
    print(f"Error: K must be between 1 and {SIZE-1} (SIZE-1)")
    usage()

# Support both absolute paths and simple filenames (backward compatible)
FILEPATH = Path(filepath_arg)
if not FILEPATH.is_absolute() and not FILEPATH.parent.exists():
    # Legacy behavior: relative filename -> put in ../test
    FILEPATH = Path(__file__).parent / "../test" / filepath_arg
FILEPATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Generating dataset: {FILEPATH}  (size={SIZE}, dim={DIM}, matches={K})")

# Generate random query vector
query_vec = [random.randint(-99, 99) for _ in range(DIM)]

# Choose K random positions for matching vectors
match_positions = random.sample(range(SIZE), K)
match_positions_set = set(match_positions)

print(f"Matching vectors will be at positions: {sorted(match_positions)}")

with open(FILEPATH, "w") as f:
    # 1) Write number of vectors
    f.write(f"{SIZE}\n")

    # 2) Write query vector
    f.write(" ".join(str(x) for x in query_vec) + "\n")

    # 3) Write database vectors
    for i in range(SIZE):
        if i in match_positions_set:
            # Write matching vector (query + small random noise)
            matching_vec = [x + random.randint(-2, 2) for x in query_vec]
            f.write(" ".join(str(x) for x in matching_vec) + "\n")
        else:
            # Write random vector
            random_vec = (str(random.randint(-99, 99)) for _ in range(DIM))
            f.write(" ".join(random_vec) + "\n")

print("✅ Done.")
