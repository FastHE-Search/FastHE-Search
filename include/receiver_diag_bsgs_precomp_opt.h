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
// ** receiver_diag_bsgs_precomp_opt: Defines the receiver class for diagonal approach with BSGS PRECOMP OPT

#pragma once

#include "receiver_diag.h"

class DiagonalBSGSPrecompOptReceiver : public DiagonalReceiver {
public:
  // constructor
  DiagonalBSGSPrecompOptReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam,
                      PrivateKey<DCRTPoly> skParam, size_t vectorParam);

  bool decryptMembership(Ciphertext<DCRTPoly> &membershipCipher) override;

  // Inherits all methods from DiagonalReceiver — same query encryption
};
