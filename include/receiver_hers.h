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
// Performs the vector normalization step in the plaintext domain

#pragma once

#include "receiver.h"

class HersReceiver : public Receiver
{
public:
  // constructor
  HersReceiver(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam,
               PrivateKey<DCRTPoly> skParam, size_t vectorParam);

  // public methods
  vector<Ciphertext<DCRTPoly>> encryptQuery(vector<double> query) override;

  bool decryptMembership(Ciphertext<DCRTPoly> &membershipCipher) override;

  vector<size_t> decryptIndex(vector<Ciphertext<DCRTPoly>> &indexCipher) override;

protected:
  // protected functions
  Ciphertext<DCRTPoly> encryptQueryAlt(vector<double> query);

private:
  // private functions
  Ciphertext<DCRTPoly> encryptQueryThread(double indexValue);
};