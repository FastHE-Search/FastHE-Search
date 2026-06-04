//  Copyright (c) 2026 LG Electronics, Inc.
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
// ** sender_diag_bsgs_simple.cpp: Simple BSGS implementation (BSGS-Plain)
// BSGS with fully hoisted rotations (both baby and giant steps).
// Reduces rotation keys from O(n) to O(sqrt(n)) while keeping all rotations hoisted.
//
// Algorithm:
// 1. Compute babyStepSize = ceil(sqrt(VECTOR_DIM))
// 2. Precompute baby-step rotations using hoisting from query
// 3. Precompute hoisting data for each baby step
// 4. Compose giant-step rotations using hoisted rotation from baby steps
// 5. Per-thread incremental accumulation for diagonal processing
//    (matches light-hydia Approach 8: T accumulators instead of 512 products)

#include "../../include/sender_diag_bsgs_simple.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSSimpleSender::DiagonalBSGSSimpleSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam)
: DiagonalSender(ccParam, pkParam, vectorParam) {
	// Compute BSGS parameters: split VECTOR_DIM into baby steps and giant steps
	// For VECTOR_DIM = 512, babyStepSize = 23, giantStepSize = 23 (23 * 23 = 529 >= 512)
	babyStepSize  = static_cast<size_t>(ceil(sqrt(double(VECTOR_DIM))));
	giantStepSize = static_cast<size_t>(ceil(double(VECTOR_DIM) / double(babyStepSize)));

	cout << "[BSGS-Plain] Simple BSGS Sender initialized (hoisted giant steps):" << endl;
	cout << "  VECTOR_DIM = " << VECTOR_DIM << endl;
	cout << "  babyStepSize = " << babyStepSize << endl;
	cout << "  giantStepSize = " << giantStepSize << endl;
	cout << "  Rotation keys: ~" << (babyStepSize - 1 + giantStepSize - 1) << " (vs " << (VECTOR_DIM - 1) << " in Approach 5)" << endl;
}

// -------------------- PUBLIC FUNCTIONS --------------------

vector<Ciphertext<DCRTPoly>> DiagonalBSGSSimpleSender::computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) {
	cout << "[BSGS-Plain] Computing similarity with hoisted BSGS..." << endl;
	auto totalStart = chrono::high_resolution_clock::now();

	size_t batchSize	   = cc->GetEncodingParams()->GetBatchSize();
	size_t cyclotomicOrder = 2 * cc->GetRingDimension();
	size_t numMatrices	   = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));
	vector<Ciphertext<DCRTPoly>> similarityCipher(numMatrices);

	// Step 1: Generate all rotations using BSGS
	vector<Ciphertext<DCRTPoly>> rotatedQueryCipher(VECTOR_DIM);
	rotatedQueryCipher[0] = queryCipher[0];

	// BSGS: precompute baby steps (rotations 0, 1, ..., babyStepSize-1) using hoisting
	cout << "[BSGS-Plain] Computing " << babyStepSize << " baby-step rotations (hoisted)..." << flush;
	auto babyStart = chrono::high_resolution_clock::now();

	vector<Ciphertext<DCRTPoly>> babySteps(babyStepSize);
	babySteps[0]							  = queryCipher[0];
	shared_ptr<vector<DCRTPoly>> queryPrecomp = cc->EvalFastRotationPrecompute(queryCipher[0]);

#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 1; i < babyStepSize; i++) {
		babySteps[i] = cc->EvalFastRotation(queryCipher[0], static_cast<int>(i), cyclotomicOrder, queryPrecomp);
	}

	auto babyEnd	= chrono::high_resolution_clock::now();
	double babyTime = chrono::duration<double>(babyEnd - babyStart).count();
	cout << " done (" << fixed << setprecision(3) << babyTime << "s)" << endl;

	// Copy baby steps to rotatedQueryCipher
	for (size_t b = 0; b < babyStepSize && b < VECTOR_DIM; b++) {
		rotatedQueryCipher[b] = babySteps[b];
	}

	// Step 2: Precompute hoisting data for each baby step
	cout << "[BSGS-Plain] Precomputing hoisting for " << babyStepSize << " baby steps..." << flush;
	auto precompStart = chrono::high_resolution_clock::now();

	vector<shared_ptr<vector<DCRTPoly>>> babyPrecomp(babyStepSize);
#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 0; i < babyStepSize; i++) {
		babyPrecomp[i] = cc->EvalFastRotationPrecompute(babySteps[i]);
	}

	auto precompEnd	   = chrono::high_resolution_clock::now();
	double precompTime = chrono::duration<double>(precompEnd - precompStart).count();
	cout << " done (" << fixed << setprecision(3) << precompTime << "s)" << endl;

	// Step 3: Compute remaining rotations using hoisted giant steps
	cout << "[BSGS-Plain] Computing " << (VECTOR_DIM - babyStepSize) << " giant-step rotations (hoisted)..." << flush;
	auto giantStart = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = babyStepSize; i < VECTOR_DIM; i++) {
		size_t giantStep	  = i / babyStepSize;
		size_t babyStep		  = i % babyStepSize;
		int giantRotation	  = static_cast<int>(giantStep * babyStepSize);
		rotatedQueryCipher[i] = cc->EvalFastRotation(babySteps[babyStep], giantRotation, cyclotomicOrder, babyPrecomp[babyStep]);
	}

	auto giantEnd	 = chrono::high_resolution_clock::now();
	double giantTime = chrono::duration<double>(giantEnd - giantStart).count();
	cout << " done (" << fixed << setprecision(3) << giantTime << "s)" << endl;

	// Step 4: Process each matrix using per-thread incremental accumulation
	// (matches light-hydia Approach 8 — T accumulators instead of 512 products)
	cout << "[BSGS-Plain] Computing similarity for " << numMatrices << " matrices (incremental accumulation)..." << flush;
	auto matrixStart = chrono::high_resolution_clock::now();

	for (size_t m = 0; m < numMatrices; m++) {
		similarityCipher[m] = computeSimilarityMatrixPrecompOpt(rotatedQueryCipher, m);
	}

	auto matrixEnd	  = chrono::high_resolution_clock::now();
	double matrixTime = chrono::duration<double>(matrixEnd - matrixStart).count();
	cout << " done (" << fixed << setprecision(3) << matrixTime << "s)" << endl;

	auto totalEnd	 = chrono::high_resolution_clock::now();
	double totalTime = chrono::duration<double>(totalEnd - totalStart).count();
	cout << "[BSGS-Plain] Total similarity computation: " << fixed << setprecision(3) << totalTime << "s" << endl;
	cout << "  Baby-step rotations: " << fixed << setprecision(3) << babyTime << "s" << endl;
	cout << "  Baby-step precompute: " << fixed << setprecision(3) << precompTime << "s" << endl;
	cout << "  Giant-step rotations: " << fixed << setprecision(3) << giantTime << "s" << endl;
	cout << "  Matrix processing: " << fixed << setprecision(3) << matrixTime << "s" << endl;

	return similarityCipher;
}

// -------------------- PROTECTED FUNCTIONS --------------------

Ciphertext<DCRTPoly> DiagonalBSGSSimpleSender::computeSimilarityMatrixPrecompOpt(vector<Ciphertext<DCRTPoly>>& rotatedQueryCipher, size_t matrix) {

	// ================================================================
	// Parallel reduction with per-thread accumulators
	// (matches light-hydia Approach 8)
	//
	// Instead of allocating 512 ciphertext products simultaneously (~8 GiB),
	// each OpenMP thread keeps its own running sum.  With T threads we hold
	// only T accumulators in memory (~T x 16 MiB).
	//
	// Each thread:
	//   1. Deserializes its share of diagonal ciphertexts from disk
	//   2. Multiplies with the corresponding pre-rotated query (ct-ct)
	//   3. Adds the product into its local accumulator (product is freed)
	//
	// After the parallel region, the T partial sums are tree-reduced.
	// ================================================================

	size_t numThreads = MAX_NUM_CORES;
	vector<Ciphertext<DCRTPoly>> threadAccumulators(numThreads);
	vector<bool> threadHasData(numThreads, false);

#pragma omp parallel for num_threads(numThreads) schedule(static)
	for (size_t d = 0; d < VECTOR_DIM; d++) {
		size_t tid = omp_get_thread_num();

		// Load diagonal ciphertext from disk
		string filepath = "serial/db_diagonal/index" + to_string(matrix * VECTOR_DIM + d) + ".bin";
		Ciphertext<DCRTPoly> diagCipher;
		if (!Serial::DeserializeFromFile(filepath, diagCipher, SerType::BINARY)) {
			cerr << "Error: cannot deserialize from \"" << filepath << "\"" << endl;
		}

		// ct-ct multiply (no relinearization yet)
		Ciphertext<DCRTPoly> product = cc->EvalMultNoRelin(rotatedQueryCipher[d], diagCipher);

		// Accumulate into this thread's running sum; product is freed after move/add
		if (!threadHasData[tid]) {
			threadAccumulators[tid] = move(product);
			threadHasData[tid]		= true;
		} else {
			cc->EvalAddInPlace(threadAccumulators[tid], product);
		}
	}

	// Tree-reduce the per-thread partial sums
	Ciphertext<DCRTPoly> accumulator;
	bool first = true;
	for (size_t t = 0; t < numThreads; t++) {
		if (threadHasData[t]) {
			if (first) {
				accumulator = move(threadAccumulators[t]);
				first		= false;
			} else {
				cc->EvalAddInPlace(accumulator, threadAccumulators[t]);
			}
		}
	}

	// Single relinearization + rescale on the accumulated sum
	cc->RelinearizeInPlace(accumulator);
	cc->RescaleInPlace(accumulator);

	return accumulator;
}
