//  Copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
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

#include "../../include/receiver.h"

// implementation of functions declared in receiver.h

// -------------------- CONSTRUCTOR --------------------

Receiver::Receiver(CryptoContext<DCRTPoly> ccParam,
                   PublicKey<DCRTPoly> pkParam, PrivateKey<DCRTPoly> skParam, size_t vectorParam)
    : cc(ccParam), pk(pkParam), sk(skParam), numVectors(vectorParam) {}