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
// ** enroller_diag_bsgs_precomp: class for encrypting and/or serializing all database vectors
// done according to our novel diagonal approach with BSGS PRECOMP optimization

#pragma once

#include "enroller_diag.h"

class DiagonalBSGSPrecompEnroller : public DiagonalEnroller {
  public:
	// constructor
	DiagonalBSGSPrecompEnroller(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// The BSGS enroller uses the same database preprocessing as the standard diagonal enroller
	// The only difference is in the rotation key generation and sender-side computation
};
