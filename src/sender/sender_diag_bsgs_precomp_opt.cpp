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
#include "../../include/sender_diag_bsgs_precomp_opt.h"
#include <omp.h>

// implementation of functions declared in sender_diag_bsgs_precomp_opt.h
//
// Key optimizations over approach 6 (DiagonalBSGSPrecompSender):
//   1. Per-thread incremental accumulation: instead of allocating 512
//      ciphertext products in memory, each thread keeps a single running
//      sum. With T threads this uses T accumulators instead of 512.
//   2. Double hoisting for baby/giant step rotations.
//   3. All rotations computed upfront in computeSimilarity, avoiding redundant rotations across matrix groups and enabling better pipelining of I/O and computation.

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSPrecompOptSender::DiagonalBSGSPrecompOptSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam,
               size_t vectorParam)
    : DiagonalSender(ccParam, pkParam, vectorParam) {
  babyStepSize = ceil(sqrt(double(VECTOR_DIM)));
  giantStepSize = ceil(double(VECTOR_DIM) / double(babyStepSize));
}

// -------------------- PUBLIC FUNCTIONS --------------------
vector<Ciphertext<DCRTPoly>> DiagonalBSGSPrecompOptSender::computeSimilarity(vector<Ciphertext<DCRTPoly>> &queryCipher) {

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t cyclotomicOrder = 2 * cc->GetRingDimension();
  size_t numMatrices = ceil(double(numVectors) / double(batchSize));
  vector<Ciphertext<DCRTPoly>> similarityCipher(numMatrices);

  // ============================================================
  // Step 1: BSGS baby-step rotations with hoisting
  // ============================================================
  vector<Ciphertext<DCRTPoly>> rotatedQueryCipher(VECTOR_DIM);
  rotatedQueryCipher[0] = queryCipher[0];

  vector<Ciphertext<DCRTPoly>> babySteps(babyStepSize);
  babySteps[0] = queryCipher[0];
  shared_ptr<vector<DCRTPoly>> queryPrecomp = cc->EvalFastRotationPrecompute(queryCipher[0]);

  #pragma omp parallel for num_threads(MAX_NUM_CORES)
  for(size_t i = 1; i < babyStepSize; i++) {
    babySteps[i] = cc->EvalFastRotation(queryCipher[0], i, cyclotomicOrder, queryPrecomp);
  }

  // ============================================================
  // Step 2: Double hoisting — precompute digit decomposition for each baby step
  // ============================================================
  vector<shared_ptr<vector<DCRTPoly>>> babyStepsPrecomp(babyStepSize);

  #pragma omp parallel for num_threads(MAX_NUM_CORES)
  for(size_t i = 0; i < babyStepSize; i++) {
    babyStepsPrecomp[i] = cc->EvalFastRotationPrecompute(babySteps[i]);
  }

  // Copy baby steps into the rotated-query array
  for(size_t b = 0; b < babyStepSize && b < VECTOR_DIM; b++) {
    rotatedQueryCipher[b] = babySteps[b];
  }

  // ============================================================
  // Step 3: Giant-step rotations using double hoisting
  // ============================================================
  #pragma omp parallel for num_threads(MAX_NUM_CORES)
  for(size_t i = babyStepSize; i < VECTOR_DIM; i++) {
    size_t giantStep = i / babyStepSize;
    size_t babyStep = i % babyStepSize;
    rotatedQueryCipher[i] = cc->EvalFastRotation(babySteps[babyStep], giantStep * babyStepSize,
                                                  cyclotomicOrder, babyStepsPrecomp[babyStep]);
  }

  // ============================================================
  // Step 4: Process each matrix group.
  // Each group's inner diagonal loop is parallelized via per-thread
  // accumulators (see computeSimilarityMatrixPrecompOpt).
  // ============================================================
  for(size_t m = 0; m < numMatrices; m++) {
    similarityCipher[m] = computeSimilarityMatrixPrecompOpt(rotatedQueryCipher, m);
  }

  return similarityCipher;
}

// -------------------- PROTECTED FUNCTIONS --------------------

Ciphertext<DCRTPoly> DiagonalBSGSPrecompOptSender::computeSimilarityMatrixPrecompOpt(
    vector<Ciphertext<DCRTPoly>> &rotatedQueryCipher, size_t matrix) {

  // ================================================================
  // Parallel reduction with per-thread accumulators
  //
  // Instead of allocating 512 ciphertext products simultaneously (~8 GiB),
  // each OpenMP thread keeps its own running sum.  With T threads we hold
  // only T accumulators in memory (~T × 16 MiB).
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
  for(size_t d = 0; d < VECTOR_DIM; d++) {
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
      threadHasData[tid] = true;
    } else {
      cc->EvalAddInPlace(threadAccumulators[tid], product);
    }
  }

  // Tree-reduce the per-thread partial sums
  Ciphertext<DCRTPoly> accumulator;
  bool first = true;
  for(size_t t = 0; t < numThreads; t++) {
    if (threadHasData[t]) {
      if (first) {
        accumulator = move(threadAccumulators[t]);
        first = false;
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
