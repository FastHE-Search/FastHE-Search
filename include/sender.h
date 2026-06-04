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

// ** sender: defines the abstract sender class to be overwritten with specific approaches
// Stores encrypted database vectors and homomorphically computes membership/index queries

#pragma once

#include "../include/config.h"
#include "../include/openFHE_wrapper.h"
#include "../include/vector_utils.h"
#include "openfhe.h"
#include <ctime>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <vector>

using namespace lbcrypto;
using namespace std;

class Sender {
  public:
	// constructor
	Sender(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// destructor
	virtual ~Sender() = default;

	// virtual methods -- must be overridden in derived sender classes
	virtual vector<Ciphertext<DCRTPoly>> computeSimilarity(vector<Ciphertext<DCRTPoly>>& queryCipher) = 0;

	virtual Ciphertext<DCRTPoly> membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) = 0;

	virtual vector<Ciphertext<DCRTPoly>> indexScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) = 0;

  protected:
	// protected members (accessible by derived classes)
	CryptoContext<DCRTPoly> cc;
	PublicKey<DCRTPoly> pk;
	size_t numVectors;
	vector<vector<Ciphertext<DCRTPoly>>> databaseCipher;
};