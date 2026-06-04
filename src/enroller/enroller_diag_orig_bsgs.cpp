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
#include "../../include/enroller_diag_orig_bsgs.h"

// implementation of functions declared in enroller_diag_orig_bsgs.h

// -------------------- CONSTRUCTOR --------------------

DiagonalOrigBSGSEnroller::DiagonalOrigBSGSEnroller(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam)
: DiagonalEnroller(ccParam, pkParam, vectorParam) {
}

// The Original BSGS enroller inherits all functionality from DiagonalEnroller
// Database preprocessing is identical - only the sender-side rotation strategy differs
