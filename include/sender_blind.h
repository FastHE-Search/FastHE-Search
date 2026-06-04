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

#include "sender_hers.h"

class BlindSender : public HersSender {
  public:
	// constructor
	BlindSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	vector<Ciphertext<DCRTPoly>> computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	Ciphertext<DCRTPoly> membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	vector<Ciphertext<DCRTPoly>> indexScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

  protected:
	// protected methods
	Ciphertext<DCRTPoly> computeSimilarityMatrix(vector<Ciphertext<DCRTPoly>>& queryCipher, size_t chunkLength, size_t matrix);

	Ciphertext<DCRTPoly> computeSimilaritySerial(Ciphertext<DCRTPoly>& queryCipher, size_t matrix, size_t index);
};