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
// ** sender_diag_orig_bsgs: sender class with original Halevi-Shoup BSGS algorithm
// Original BSGS with hoisting optimization:
// - Stores only O(sqrt(n)) baby steps for O(sqrt(n)) memory usage
// - Precomputes digit decomposition (hoisting) for each baby step
// - Applies giant step rotations on-the-fly using EvalFastRotation

#pragma once

#include "sender_diag.h"

class DiagonalOrigBSGSSender : public DiagonalSender {
  public:
	// constructor
	DiagonalOrigBSGSSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	vector<Ciphertext<DCRTPoly>> computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

  protected:
	// Override computeSimilarityMatrix to use BSGS on-the-fly combination with hoisting
	Ciphertext<DCRTPoly> computeSimilarityMatrixBSGS(vector<Ciphertext<DCRTPoly>>& babySteps, vector<shared_ptr<vector<DCRTPoly>>>& babyStepsPrecomp, size_t matrix);

	// Compute similarity for a single diagonal using BSGS with hoisting
	Ciphertext<DCRTPoly>
	computeSimilarityThreadBSGS(vector<Ciphertext<DCRTPoly>>& babySteps, vector<shared_ptr<vector<DCRTPoly>>>& babyStepsPrecomp, size_t matrix, size_t diagonalIndex);

  private:
	// BSGS parameters
	size_t babyStepSize;
	size_t giantStepSize;
};
