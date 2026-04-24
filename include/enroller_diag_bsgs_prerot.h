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
// ** enroller_diag_bsgs_prerot: Enroller that pre-rotates diagonals in PLAINTEXT
// before encryption, so the server never needs negative rotation keys.
//
// For BSGS with baby-step size n1, diagonal k (within a VECTOR_DIM block)
// belongs to giant-step group j = floor(k / n1).  Standard Approach 81 encrypts
// diag[k] as-is and has the server homomorphically rotate by -j*n1 on the GPU.
//
// This enroller instead applies a cyclic left-shift by j*n1 positions on the
// plaintext diagonal vector BEFORE encryption.  The result is mathematically
// identical, but eliminates:
//   - All homomorphic pre-rotations at the server (G*(ell-n1) rotations)
//   - All negative rotation keys (n2-1 keys)
//
// The encrypted output is in the same serialized format (serial/db_diagonal/indexN.bin).

#pragma once

#include "enroller_diag.h"
#include <cmath>

class DiagonalBSGSPreRotEnroller : public DiagonalEnroller {
public:
  DiagonalBSGSPreRotEnroller(CryptoContext<DCRTPoly> ccParam,
                              PublicKey<DCRTPoly> pkParam,
                              size_t vectorParam);

  /// Serialize database with plaintext pre-rotation of diagonals.
  void serializeDB(vector<vector<double>> &database);

protected:
  size_t babyStepSize_;
  size_t giantStepSize_;

  /// Apply CKKS-slot-level cyclic left-shift to a plaintext row.
  /// Within each VECTOR_DIM block, shifts by `amount` positions cyclically.
  /// Multiple blocks (numSlots / VECTOR_DIM) are each shifted independently.
  static void applyPlaintextRotation(vector<double>& row, int amount, size_t vectorDim);
};
