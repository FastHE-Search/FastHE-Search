//  Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
//  Portions copyright (c) 2026 LG Electronics, Inc.
//
//  Licensed under the MIT License (the "License"); you may not use this file
//  except in compliance with the License.
//
//  You may obtain a copy of the License in the LICENSE file at the project
//  root or at
//
//  https://mit-license.org/
//
//  SPDX-License-Identifier: MIT

// ** Holds configuration parameters like file paths, default values, and any
// other constant values

#pragma once

#include <cstdlib>
#include <string>

// similarity threshold value used to determine a match between vectors
const double MATCH_THRESHOLD = 0.44;

// Depth to be consumed by the comparison approximation function
// Relationship with multiplicative depth described at the below link
// https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md
// COMP_DEPTH_VAL can be set via cmake -DCOMP_DEPTH_VAL=N (default: 8)
#ifndef COMP_DEPTH_VAL
#define COMP_DEPTH_VAL 8
#endif
const size_t COMP_DEPTH = COMP_DEPTH_VAL;

// GPU comparison depth (used by GPU-accelerated approaches)
const size_t COMP_DEPTH_GPU = COMP_DEPTH_VAL;

// BSGS comparison depth (used by BSGS approaches)
const size_t COMP_DEPTH_BSGS = COMP_DEPTH_VAL;

// Baby-step size for BSGS: ceil(sqrt(VECTOR_DIM)) = ceil(sqrt(512)) = 23
const size_t BABY_STEP = 23;

// Number of squares to be taken during alpha-norm approximation of max values
// Invokes a mult depth of alpha in the group-testing approach
const size_t ALPHA_DEPTH = 2;

// Number of threads used in multithreaded sections (can be overridden by OUTER_THREADS env var)
// Default value if environment variable is not set
const size_t DEFAULT_MAX_NUM_CORES = 48;

// Global variable for the actual number of threads to use
// Will be set during initialization based on environment variable or default
extern size_t MAX_NUM_CORES;

// Function to initialize threading configuration from environment variables
inline size_t getMaxNumCores() {
	const char* outer_threads = std::getenv("OUTER_THREADS");
	if (outer_threads != nullptr) {
		int threads = std::atoi(outer_threads);
		if (threads > 0) {
			return static_cast<size_t>(threads);
		}
	}
	return DEFAULT_MAX_NUM_CORES;
}

// ---------- Variables below should not be changed ----------

// TODO: make serialization into separate executable or remove entirely
const bool READ_FROM_SERIAL = false;

// Dimension (length) of inputted query / database vectors
const size_t VECTOR_DIM = 512;

// Dimension (length) of subvector partitions used in Blind-Match approach
// Must equal a power of 2
const size_t CHUNK_LEN = 128;

const std::string EXP_FILEPATH = "latency.csv";