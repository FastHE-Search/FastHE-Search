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
#include "../../include/receiver_diag_orig_bsgs.h"

// implementation of functions declared in receiver_diag_orig_bsgs.h

// -------------------- CONSTRUCTOR --------------------

DiagonalOrigBSGSReceiver::DiagonalOrigBSGSReceiver(CryptoContext<DCRTPoly> ccParam,
                         PublicKey<DCRTPoly> pkParam, PrivateKey<DCRTPoly> skParam, size_t vectorParam)
    : DiagonalReceiver(ccParam, pkParam, skParam, vectorParam) {}

// The Original BSGS receiver inherits all functionality from DiagonalReceiver
// Query encryption is identical - only the sender-side rotation strategy differs
