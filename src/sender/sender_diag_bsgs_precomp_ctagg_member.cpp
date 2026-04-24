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
#include "../../include/sender_diag_bsgs_precomp_ctagg_member.h"
#include <omp.h>

// implementation of functions declared in sender_diag_bsgs_precomp_ctagg_member.h
//
// Key optimization for the membership scenario:
//   Instead of computing numMatrices separate similarity vectors and
//   applying Chebyshev to each independently, we fuse the matrix
//   summation into the diagonal multiplication step:
//
//   Original (approach 8):
//     for each matrix m:
//       score_m = sum_d rot(q,d) * diag_{m,d}     [512 ct-ct mults]
//       compared_m = chebyshev(score_m)             [1 Chebyshev eval]
//     result = EvalSum(sum_m compared_m)
//     Total: numMatrices * 512 mults + numMatrices Chebyshev evals
//
//   Optimized (approach 9):
//     for each diagonal d:
//       mergedDiag_d = sum_m diag_{m,d}            [(numMatrices-1) ct-ct adds]
//     score = sum_d rot(q,d) * mergedDiag_d         [512 ct-ct mults]
//     compared = chebyshev(score, range=[-numMatrices, numMatrices])
//     result = EvalSum(compared)
//     Total: 512 mults + (numMatrices-1)*512 adds + 1 Chebyshev eval
//
//   Savings: (numMatrices-1)*512 mults replaced by (numMatrices-1)*512 adds
//            + (numMatrices-1) Chebyshev evaluations saved
//
//   The merged similarity values are in [-numMatrices, numMatrices].
//   We accommodate this by widening the Chebyshev approximation range.
//   The comparison threshold remains at MATCH_THRESHOLD because:
//   - A matching slot: one match (~0.44+) + (numMatrices-1) near-zero ≈ 0.44
//   - A non-matching slot: numMatrices near-zero values ≈ 0

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSPrecompCtAggMemberSender::DiagonalBSGSPrecompCtAggMemberSender(
    CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam)
    : DiagonalBSGSPrecompOptSender(ccParam, pkParam, vectorParam) {}

// -------------------- PUBLIC FUNCTIONS --------------------

Ciphertext<DCRTPoly> DiagonalBSGSPrecompCtAggMemberSender::membershipScenario(
    vector<Ciphertext<DCRTPoly>> &queryCipher) {

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t cyclotomicOrder = 2 * cc->GetRingDimension();
  size_t numMatrices = ceil(double(numVectors) / double(batchSize));

  // If only one matrix, fall back to the standard approach (no savings possible)
  if (numMatrices <= 1) {
    return DiagonalBSGSPrecompOptSender::membershipScenario(queryCipher);
  }

  // ============================================================
  // Step 1: BSGS baby-step rotations with hoisting
  //         (identical to approach 8's computeSimilarity)
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
  // Step 4: Fused membership computation — sum diagonals across
  //         all matrices, then single multiply per rotation
  // ============================================================
  Ciphertext<DCRTPoly> scoreCipher = computeMembershipFused(rotatedQueryCipher, numMatrices);

  // ============================================================
  // Step 5: Standard Chebyshev comparison after membership-only
  //         query normalization.
  //
  //         The receiver scales the normalized query by 1/maxValue, where
  //           maxValue = 1 + (numMatrices - 1) * 2 / sqrt(VECTOR_DIM)
  //         so the aggregated similarity score remains within [-1,1].
  //         This lets us reuse the standard [-1,1] comparator at
  //         COMP_DEPTH = 8 instead of the widened-range overload.
  // ============================================================
  double maxValue = 1.0 + (static_cast<double>(numMatrices) - 1.0) * 2.0 / sqrt(double(VECTOR_DIM));
  double adjustedThreshold = MATCH_THRESHOLD / maxValue;
  scoreCipher = OpenFHEWrapper::chebyshevCompare(cc, scoreCipher, adjustedThreshold, COMP_DEPTH);

  return scoreCipher;
}

// -------------------- PRIVATE FUNCTIONS --------------------

Ciphertext<DCRTPoly> DiagonalBSGSPrecompCtAggMemberSender::computeMembershipFused(
    vector<Ciphertext<DCRTPoly>> &rotatedQueryCipher, size_t numMatrices) {

  // ================================================================
  // For each diagonal index d in [0, VECTOR_DIM):
  //   1. Load diag_{0,d}, diag_{1,d}, ..., diag_{numMatrices-1,d}
  //   2. Sum them: mergedDiag_d = sum_m diag_{m,d}       [ct-ct adds, cheap]
  //   3. Multiply: product_d = rot(q,d) * mergedDiag_d   [1 ct-ct mult]
  //   4. Accumulate product_d into running sum
  //
  // This replaces numMatrices multiplications per diagonal with
  // 1 multiplication + (numMatrices-1) additions per diagonal.
  //
  // Uses the same per-thread accumulator pattern as approach 8
  // to keep memory bounded.
  // ================================================================

  size_t numThreads = MAX_NUM_CORES;
  vector<Ciphertext<DCRTPoly>> threadAccumulators(numThreads);
  vector<bool> threadHasData(numThreads, false);

  #pragma omp parallel for num_threads(numThreads) schedule(static)
  for(size_t d = 0; d < VECTOR_DIM; d++) {
    size_t tid = omp_get_thread_num();

    // --- Phase A: Load and sum diagonals across all matrices ---
    string filepath = "serial/db_diagonal/index" + to_string(0 * VECTOR_DIM + d) + ".bin";
    Ciphertext<DCRTPoly> mergedDiag;
    if (!Serial::DeserializeFromFile(filepath, mergedDiag, SerType::BINARY)) {
      cerr << "Error: cannot deserialize from \"" << filepath << "\"" << endl;
    }

    // Add remaining matrices' diagonals (cheap ct-ct additions)
    for(size_t m = 1; m < numMatrices; m++) {
      filepath = "serial/db_diagonal/index" + to_string(m * VECTOR_DIM + d) + ".bin";
      Ciphertext<DCRTPoly> diagCipher;
      if (!Serial::DeserializeFromFile(filepath, diagCipher, SerType::BINARY)) {
        cerr << "Error: cannot deserialize from \"" << filepath << "\"" << endl;
      }
      cc->EvalAddInPlace(mergedDiag, diagCipher);
    }

    // --- Phase B: Single ct-ct multiply with rotated query ---
    Ciphertext<DCRTPoly> product = cc->EvalMultNoRelin(rotatedQueryCipher[d], mergedDiag);

    // --- Phase C: Accumulate into this thread's running sum ---
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
