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

#include "../../include/receiver_diag.h"

// implementation of functions declared in receiver_base.h

// -------------------- CONSTRUCTOR --------------------

DiagonalReceiver::DiagonalReceiver(CryptoContext<DCRTPoly> ccParam,
                                   PublicKey<DCRTPoly> pkParam, PrivateKey<DCRTPoly> skParam, size_t vectorParam)
    : HersReceiver(ccParam, pkParam, skParam, vectorParam) {}

// -------------------- PUBLIC FUNCTIONS --------------------

vector<Ciphertext<DCRTPoly>> DiagonalReceiver::encryptQuery(vector<double> query)
{

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();

  query = VectorUtils::plaintextNormalize(query, VECTOR_DIM);
  vector<double> queryBatch(batchSize);
  for (size_t i = 0; i < batchSize; i += VECTOR_DIM)
  {
    copy(query.begin(), query.end(), queryBatch.begin() + i);
  }

  vector<Ciphertext<DCRTPoly>> queryCipher({OpenFHEWrapper::encryptFromVector(cc, pk, queryBatch)});

  return queryCipher;
}

vector<Ciphertext<DCRTPoly>> DiagonalReceiver::encryptScaledQuery(vector<double> normalizedScaledQuery)
{

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();

  vector<double> queryBatch(batchSize);
  for (size_t i = 0; i < batchSize; i += VECTOR_DIM)
  {
    copy(normalizedScaledQuery.begin(), normalizedScaledQuery.end(), queryBatch.begin() + i);
  }

  vector<Ciphertext<DCRTPoly>> queryCipher({OpenFHEWrapper::encryptFromVector(cc, pk, queryBatch)});

  return queryCipher;
}
