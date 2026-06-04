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
// ** sender_diag_bsgs_precomp_ctagg_member: membership-optimized diagonal BSGS sender
//
// Key optimization over approach 8 (DiagonalBSGSPrecompOptSender):
//   For the membership scenario we exploit the linearity of the diagonal
//   matrix-vector product to fuse the matrix dimension:
//
//     sum_m  sum_d  rot(q,d) * diag_{m,d}
//         = sum_d  rot(q,d) * (sum_m diag_{m,d})
//
//   By summing the diagonal ciphertexts across all numMatrices groups
//   *before* the ct-ct multiplication with the rotated query, we replace
//   (numMatrices - 1) * VECTOR_DIM expensive ct-ct multiplications
//   with the same number of cheap ct-ct additions.
//
//   The merged similarity values are in [-numMatrices, numMatrices] rather
//   than [-1, 1].  We accommodate this by widening the Chebyshev
//   approximation range to [-numMatrices, numMatrices] while keeping the
//   comparison threshold at MATCH_THRESHOLD (0.44).  This works because:
//   - A matching slot contributes one high similarity (~0.44+) plus
//     (numMatrices-1) near-zero random cosine similarities.
//   - A non-matching slot sums numMatrices near-zero random similarities.
//   The step function at 0.44 cleanly separates these two populations.
//
//   Additionally saves (numMatrices - 1) Chebyshev evaluations since only
//   one merged score ciphertext needs comparison (vs numMatrices individual).
//
//   The indexScenario and computeSimilarity remain unchanged (inherited
//   from approach 8) since those require per-matrix results.

#pragma once

#include "sender_diag_bsgs_precomp_opt.h"

class DiagonalBSGSPrecompCtAggMemberSender : public DiagonalBSGSPrecompOptSender {
  public:
	// constructor
	DiagonalBSGSPrecompCtAggMemberSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// Override membershipScenario with the fused diagonal-sum optimization
	Ciphertext<DCRTPoly> membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

  private:
	// Compute the membership similarity by summing diagonals across matrices
	// before multiplying by rotated query vectors
	Ciphertext<DCRTPoly> computeMembershipFused(vector<Ciphertext<DCRTPoly>>& rotatedQueryCipher, size_t numMatrices);
};
