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
#include "../../include/sender_diag_orig_bsgs.h"

// implementation of functions declared in sender_diag_orig_bsgs.h

// -------------------- CONSTRUCTOR --------------------

DiagonalOrigBSGSSender::DiagonalOrigBSGSSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam)
: DiagonalSender(ccParam, pkParam, vectorParam) {
	// Compute BSGS parameters: split VECTOR_DIM into baby steps and giant steps
	// For VECTOR_DIM = 512, babyStepSize = 23, giantStepSize = 23 (23 * 23 = 529 >= 512)
	babyStepSize  = ceil(sqrt(double(VECTOR_DIM)));
	giantStepSize = ceil(double(VECTOR_DIM) / double(babyStepSize));
}

// -------------------- PUBLIC FUNCTIONS --------------------
vector<Ciphertext<DCRTPoly>> DiagonalOrigBSGSSender::computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) {

	size_t batchSize	   = cc->GetEncodingParams()->GetBatchSize();
	size_t cyclotomicOrder = 2 * cc->GetRingDimension();
	size_t numMatrices	   = ceil(double(numVectors) / double(batchSize));
	vector<Ciphertext<DCRTPoly>> similarityCipher(numMatrices);

	// HALEVI-SHOUP BSGS WITH HOISTING:
	// 1. Precompute baby steps (rotations 0, 1, ..., babyStepSize-1)
	// 2. Precompute digit decomposition for each baby step (hoisting)
	// 3. Apply giant step rotations on-the-fly using EvalFastRotation

	// Step 1: Compute baby steps
	vector<Ciphertext<DCRTPoly>> babySteps(babyStepSize);
	babySteps[0]							  = queryCipher[0];
	shared_ptr<vector<DCRTPoly>> queryPrecomp = cc->EvalFastRotationPrecompute(queryCipher[0]);

#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 1; i < babyStepSize; i++) {
		babySteps[i] = cc->EvalFastRotation(queryCipher[0], i, cyclotomicOrder, queryPrecomp);
	}

	// Step 2: Hoisting - precompute digit decomposition for each baby step
	// This allows fast giant-step rotations via EvalFastRotation
	vector<shared_ptr<vector<DCRTPoly>>> babyStepsPrecomp(babyStepSize);

#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 0; i < babyStepSize; i++) {
		babyStepsPrecomp[i] = cc->EvalFastRotationPrecompute(babySteps[i]);
	}

	// Step 3: Process each matrix using BSGS with hoisted precomputation
	for (size_t m = 0; m < numMatrices; m++) {
		similarityCipher[m] = computeSimilarityMatrixBSGS(babySteps, babyStepsPrecomp, m);
	}

	return similarityCipher;
}

// -------------------- PROTECTED FUNCTIONS --------------------

Ciphertext<DCRTPoly>
DiagonalOrigBSGSSender::computeSimilarityMatrixBSGS(vector<Ciphertext<DCRTPoly>>& babySteps, vector<shared_ptr<vector<DCRTPoly>>>& babyStepsPrecomp, size_t matrix) {

	vector<Ciphertext<DCRTPoly>> scoreCipher(VECTOR_DIM);

// For each diagonal d, we need rotation by d
// Using BSGS: rotation(d) = rotation(babyStep) composed with rotation(giantStep)
// where d = giantIdx * babyStepSize + babyIdx
#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t d = 0; d < VECTOR_DIM; d++) {
		scoreCipher[d] = computeSimilarityThreadBSGS(babySteps, babyStepsPrecomp, matrix, d);
	}

	// Sum all diagonal contributions
	for (size_t i = 1; i < VECTOR_DIM; i++) {
		cc->EvalAddInPlace(scoreCipher[0], scoreCipher[i]);
	}

	cc->RelinearizeInPlace(scoreCipher[0]);
	cc->RescaleInPlace(scoreCipher[0]);

	return scoreCipher[0];
}

Ciphertext<DCRTPoly> DiagonalOrigBSGSSender::computeSimilarityThreadBSGS(vector<Ciphertext<DCRTPoly>>& babySteps,
  vector<shared_ptr<vector<DCRTPoly>>>& babyStepsPrecomp,
  size_t matrix,
  size_t diagonalIndex) {

	size_t cyclotomicOrder = 2 * cc->GetRingDimension();

	// Decompose diagonal index into baby and giant step indices
	size_t giantIdx = diagonalIndex / babyStepSize;
	size_t babyIdx	= diagonalIndex % babyStepSize;

	// Load the precomputed diagonal ciphertext from disk
	string filepath = "serial/db_diagonal/index" + to_string(matrix * VECTOR_DIM + diagonalIndex) + ".bin";
	Ciphertext<DCRTPoly> databaseCipher;
	if (Serial::DeserializeFromFile(filepath, databaseCipher, SerType::BINARY) == false) {
		cerr << "Error: cannot deserialize from \"" << filepath << "\"" << endl;
	}

	// BSGS with hoisting: rotation(d) = EvalFastRotation(babySteps[babyIdx], giantIdx * babyStepSize)
	// Using precomputed digit decomposition for fast rotation
	Ciphertext<DCRTPoly> rotatedQuery;
	if (giantIdx == 0) {
		// No giant step needed, use baby step directly
		rotatedQuery = babySteps[babyIdx];
	} else {
		// Apply giant step rotation using hoisted precomputation (EvalFastRotation)
		rotatedQuery = cc->EvalFastRotation(babySteps[babyIdx], giantIdx * babyStepSize, cyclotomicOrder, babyStepsPrecomp[babyIdx]);
	}

	return cc->EvalMultNoRelin(rotatedQuery, databaseCipher);
}
