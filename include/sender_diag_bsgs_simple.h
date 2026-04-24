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
// ** sender_diag_bsgs_simple.h: Simple BSGS sender class (BSGS-Plain)
// BSGS (Baby-Step Giant-Step) reduces rotations from O(n) to O(sqrt(n))
// This is a clean, minimal implementation similar to light-hydia's BSGS approach

#pragma once

#include "sender_diag.h"

class DiagonalBSGSSimpleSender : public DiagonalSender {
public:
  // constructor (no sk param - light-hydia's DiagonalSender doesn't take it)
  DiagonalBSGSSimpleSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, 
                           size_t vectorParam);

  // public methods - override computeSimilarity with BSGS optimization
  vector<Ciphertext<DCRTPoly>>
  computeSimilarity(vector<Ciphertext<DCRTPoly>> &queryCipher) override;

protected:
  // Per-matrix incremental accumulation (matches light-hydia Approach 8)
  // Uses per-thread running sums instead of allocating all 512 products
  Ciphertext<DCRTPoly>
  computeSimilarityMatrixPrecompOpt(vector<Ciphertext<DCRTPoly>> &rotatedQueryCipher, size_t matrix);

  // BSGS parameters (protected so GPU subclass can access)
  size_t babyStepSize;
  size_t giantStepSize;
};
