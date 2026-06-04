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
// ** sender_diag_bsgs_precomp_opt: sender class with diagonal approach using BSGS + double hoisting
// Optimizations over the base DiagonalBSGSPrecompSender (approach 6):
//   1. Outer-loop parallelism: matrix groups processed in parallel (not just diagonals)
//   2. Incremental accumulation: running sum instead of O(n) ciphertext array per group,
//      reducing per-group memory from ~8 GiB to ~16 MiB and enabling outer parallelism
//   3. Pipelined I/O: async prefetching of diagonal ciphertexts overlaps with computation

#pragma once

#include "sender_diag.h"

class DiagonalBSGSPrecompOptSender : public DiagonalSender {
  public:
	// constructor
	DiagonalBSGSPrecompOptSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	vector<Ciphertext<DCRTPoly>> computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

  protected:
	// Override computeSimilarityMatrix to use incremental accumulation
	Ciphertext<DCRTPoly> computeSimilarityMatrixPrecompOpt(vector<Ciphertext<DCRTPoly>>& rotatedQueryCipher, size_t matrix);

	// BSGS parameters
	size_t babyStepSize;
	size_t giantStepSize;
};
