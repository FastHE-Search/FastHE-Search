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
#include "../../include/sender_diag_bsgs_precomp.h"

// implementation of functions declared in sender_diag_bsgs_precomp.h

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSPrecompSender::DiagonalBSGSPrecompSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam)
: DiagonalSender(ccParam, pkParam, vectorParam) {
	// Compute BSGS parameters: split VECTOR_DIM into baby steps and giant steps
	babyStepSize  = ceil(sqrt(double(VECTOR_DIM)));
	giantStepSize = ceil(double(VECTOR_DIM) / double(babyStepSize));
}

// -------------------- PUBLIC FUNCTIONS --------------------
vector<Ciphertext<DCRTPoly>> DiagonalBSGSPrecompSender::computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) {

	size_t batchSize	   = cc->GetEncodingParams()->GetBatchSize();
	size_t cyclotomicOrder = 2 * cc->GetRingDimension();
	size_t numMatrices	   = ceil(double(numVectors) / double(batchSize));
	vector<Ciphertext<DCRTPoly>> similarityCipher(numMatrices);

	// Generate all rotations of batched query vector using BSGS with hoisting
	vector<Ciphertext<DCRTPoly>> rotatedQueryCipher(VECTOR_DIM);
	rotatedQueryCipher[0] = queryCipher[0];

	// Step 1: Precompute baby steps (rotations 0, 1, ..., babyStepSize-1)
	vector<Ciphertext<DCRTPoly>> babySteps(babyStepSize);
	babySteps[0]							  = queryCipher[0];
	shared_ptr<vector<DCRTPoly>> queryPrecomp = cc->EvalFastRotationPrecompute(queryCipher[0]);

#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 1; i < babyStepSize; i++) {
		babySteps[i] = cc->EvalFastRotation(queryCipher[0], i, cyclotomicOrder, queryPrecomp);
	}

	// Step 2: Hoisting - precompute digit decomposition for each baby step
	vector<shared_ptr<vector<DCRTPoly>>> babyStepsPrecomp(babyStepSize);

#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 0; i < babyStepSize; i++) {
		babyStepsPrecomp[i] = cc->EvalFastRotationPrecompute(babySteps[i]);
	}

	// Copy baby steps to rotatedQueryCipher
	for (size_t b = 0; b < babyStepSize && b < VECTOR_DIM; b++) {
		rotatedQueryCipher[b] = babySteps[b];
	}

// Step 3: Compute remaining rotations using hoisted baby steps + giant step rotations
#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = babyStepSize; i < VECTOR_DIM; i++) {
		size_t giantStep = i / babyStepSize;
		size_t babyStep	 = i % babyStepSize;
		// Use EvalFastRotation with precomputed digit decomposition (hoisting)
		rotatedQueryCipher[i] = cc->EvalFastRotation(babySteps[babyStep], giantStep * babyStepSize, cyclotomicOrder, babyStepsPrecomp[babyStep]);
	}

	// Process each matrix using the same approach as approach 5
	for (size_t m = 0; m < numMatrices; m++) {
		similarityCipher[m] = computeSimilarityMatrix(rotatedQueryCipher, m);
	}

	return similarityCipher;
}
