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
// ** enroller_diag_bsgs_precomp_opt: class for encrypting and/or serializing all database vectors
// done according to our novel diagonal approach with BSGS PRECOMP OPT optimization
// Database preprocessing is identical to the standard diagonal enroller;
// only the sender-side computation differs.

#pragma once

#include "enroller_diag.h"

class DiagonalBSGSPrecompOptEnroller : public DiagonalEnroller {
  public:
	// constructor
	DiagonalBSGSPrecompOptEnroller(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// Inherits all methods from DiagonalEnroller — same database preprocessing
};
