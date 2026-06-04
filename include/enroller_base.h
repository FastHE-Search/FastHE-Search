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

// ** base_enroller: class for encrypting and/or serializing all database vectors
// done according to the lit baseline approach

#pragma once

#include "enroller_hers.h"

class BaseEnroller : public HersEnroller {
  public:
	// constructor
	BaseEnroller(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam);

	// public methods
	void serializeDB(vector<vector<double>>& database);
};