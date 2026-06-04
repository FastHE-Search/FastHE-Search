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

#pragma once

#include "sender_hers.h"

class DiagonalSender : public HersSender {
  public:
	// constructor
	DiagonalSender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	vector<Ciphertext<DCRTPoly>> computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	Ciphertext<DCRTPoly> membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	vector<Ciphertext<DCRTPoly>> indexScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

  protected:
	// protected methods - can be used by derived classes
	Ciphertext<DCRTPoly> computeSimilarityMatrix(vector<Ciphertext<DCRTPoly>>& queryCipher, size_t matrix);

	Ciphertext<DCRTPoly> computeSimilarityThread(Ciphertext<DCRTPoly>& queryCipher, size_t matrix, size_t index);
};