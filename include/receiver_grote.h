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

// ** receiver: Defines the receiver (querier) base class
// encrypts / decrypts queries according to literature baseline approach

#pragma once

#include "receiver_base.h"

class GroteReceiver : public BaseReceiver {
  public:
	// constructor
	GroteReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, PrivateKey<DCRTPoly> skParam, size_t vectorParam);

	// public methods
	vector<size_t> decryptIndex(vector<Ciphertext<DCRTPoly>>& indexCipher);
};