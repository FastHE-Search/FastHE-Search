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
#include "../../include/receiver_diag_bsgs_precomp_opt.h"

// implementation of functions declared in receiver_diag_bsgs_precomp_opt.h

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSPrecompOptReceiver::DiagonalBSGSPrecompOptReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam,
                                       PrivateKey<DCRTPoly> skParam, size_t vectorParam)
    : DiagonalReceiver(ccParam, pkParam, skParam, vectorParam) {}

bool DiagonalBSGSPrecompOptReceiver::decryptMembership(Ciphertext<DCRTPoly> &membershipCipher) {

    vector<double> membershipValues = OpenFHEWrapper::decryptToVector(cc, sk, membershipCipher);

    for (double value : membershipValues) {
        if (value >= 1.0) {
            return true;
        }
    }

    return false;
}
