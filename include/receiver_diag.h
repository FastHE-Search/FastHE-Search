//  Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
//  Portions copyright (c) 2026 LG Electronics, Inc.
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

// ** Holds configuration parameters like file paths, default values, and any
// other constant values

// ** receiver: Defines the receiver class according to our novel diagonalization approach

#pragma once

#include "receiver_hers.h"

class DiagonalReceiver : public HersReceiver
{
public:
  // constructor
  DiagonalReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam,
                   PrivateKey<DCRTPoly> skParam, size_t vectorParam);

  // public methods
  vector<Ciphertext<DCRTPoly>> encryptQuery(vector<double> query) override;
  vector<Ciphertext<DCRTPoly>> encryptScaledQuery(vector<double> normalizedScaledQuery);
};