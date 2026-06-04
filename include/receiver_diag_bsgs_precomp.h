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
// ** receiver_diag_bsgs_precomp: Defines the receiver class according to diagonal approach with BSGS PRECOMP

#pragma once

#include "receiver_diag.h"

class DiagonalBSGSPrecompReceiver : public DiagonalReceiver {
  public:
	// constructor
	DiagonalBSGSPrecompReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, PrivateKey<DCRTPoly> skParam, size_t vectorParam);

	// The BSGS receiver uses the same query encryption as the standard diagonal receiver
};
