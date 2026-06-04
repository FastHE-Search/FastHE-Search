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
// ** sender_diag_bsgs_precomp: sender class with diagonal approach optimized using BSGS algorithm
// BSGS (Baby-Step Giant-Step) reduces rotations from O(n) to O(sqrt(n))

#pragma once

#include "sender_diag.h"

class DiagonalBSGSPrecompSender : public DiagonalSender {
  public:
	// constructor
	DiagonalBSGSPrecompSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	vector<Ciphertext<DCRTPoly>> computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

  private:
	// BSGS parameters
	size_t babyStepSize;
	size_t giantStepSize;
};
