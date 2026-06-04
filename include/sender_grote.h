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

#pragma once

#include "sender_base.h"

class GroteSender : public BaseSender {
  public:
	// constructor
	GroteSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	Ciphertext<DCRTPoly> membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	vector<Ciphertext<DCRTPoly>> indexScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;
};