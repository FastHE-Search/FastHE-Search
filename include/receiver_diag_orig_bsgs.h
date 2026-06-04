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
// ** receiver_diag_orig_bsgs: Defines the receiver class according to diagonal approach with full BSGS

#pragma once

#include "receiver_diag.h"

class DiagonalOrigBSGSReceiver : public DiagonalReceiver {
  public:
	// constructor
	DiagonalOrigBSGSReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, PrivateKey<DCRTPoly> skParam, size_t vectorParam);

	// The Original BSGS receiver uses the same query encryption as the standard diagonal receiver
};
