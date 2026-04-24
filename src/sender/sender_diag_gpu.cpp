//  Copyright (c) 2026 LG Electronics, Inc.
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
#include "../../include/sender_diag_gpu.h"
#include "../../include/gpu_hydia_helper.cuh"
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;

DiagonalSenderGPU::DiagonalSenderGPU(CryptoContext<DCRTPoly> ccParam,
                                     PublicKey<DCRTPoly> pkParam,
                                     size_t vectorParam,
                                     GPUHydiaHelper* gpuHelper)
    : DiagonalSender(ccParam, pkParam, vectorParam),
      gpuHelper_(gpuHelper),
      gpuCacheInitialized_(false) {
  if (gpuHelper_ && gpuHelper_->isReady()) {
    cout << "[Approach 51] HyDia-GPU sender initialized" << endl;
  } else {
    cerr << "[Approach 51] WARNING: GPU helper not ready, will fall back to CPU" << endl;
  }
}

DiagonalSenderGPU::~DiagonalSenderGPU() {
  // gpuHelper_ is owned externally
}

bool DiagonalSenderGPU::isGPUReady() const {
  return gpuHelper_ != nullptr && gpuHelper_->isReady();
}

void DiagonalSenderGPU::initializeGPUCache() {
  if (!isGPUReady()) {
    cerr << "[Approach 51] Cannot initialize GPU cache - GPU not ready" << endl;
    return;
  }

  if (gpuCacheInitialized_) {
    cout << "[Approach 51] GPU cache already initialized" << endl;
    return;
  }

  cacheDiagonalsOnGPU();
  gpuCacheInitialized_ = true;
}

void DiagonalSenderGPU::initializeGPUCache(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals) {
  if (!isGPUReady()) {
    cerr << "[Approach 51] Cannot initialize GPU cache - GPU not ready" << endl;
    return;
  }

  if (gpuCacheInitialized_) {
    cout << "[Approach 51] GPU cache already initialized" << endl;
    return;
  }

  cout << "[Approach 51] Using preloaded diagonals (skipping file I/O)" << endl;
  cacheDiagonalsOnGPU(preloadedDiagonals);
  gpuCacheInitialized_ = true;
}

void DiagonalSenderGPU::cacheDiagonalsOnGPU() {
  if (!isGPUReady()) return;

  cout << "[Approach 51] Loading and caching diagonals on GPU..." << endl;
  auto startTime = chrono::steady_clock::now();

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

  vector<Ciphertext<DCRTPoly>> allDiagonals;
  allDiagonals.reserve(numMatrices * VECTOR_DIM);

  for (size_t m = 0; m < numMatrices; ++m) {
    for (size_t i = 0; i < VECTOR_DIM; ++i) {
      string filepath = "serial/db_diagonal/index" + to_string(m * VECTOR_DIM + i) + ".bin";
      Ciphertext<DCRTPoly> diagonalCipher;
      if (!Serial::DeserializeFromFile(filepath, diagonalCipher, SerType::BINARY)) {
        cerr << "[Approach 51] Error: cannot deserialize from \"" << filepath << "\"" << endl;
        continue;
      }
      allDiagonals.push_back(diagonalCipher);
    }
  }

  // n1 = VECTOR_DIM => no effective giant-step pre-rotation; matches pure diagonal behavior.
  gpuHelper_->CacheDiagonalsWithPreRotationOnGPU(allDiagonals, numMatrices, VECTOR_DIM, static_cast<int>(VECTOR_DIM));

  auto endTime = chrono::steady_clock::now();
  double elapsed = chrono::duration<double>(endTime - startTime).count();
  cout << "[Approach 51] Cached " << allDiagonals.size()
       << " diagonals on GPU in " << fixed << setprecision(2) << elapsed << "s" << endl;
}

void DiagonalSenderGPU::cacheDiagonalsOnGPU(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals) {
  if (!isGPUReady()) return;

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

  cout << "[Approach 51] Caching " << preloadedDiagonals.size() << " preloaded diagonals on GPU..." << endl;
  auto startTime = chrono::steady_clock::now();

  // n1 = VECTOR_DIM => pure diagonal behavior, no giant-step decomposition.
  gpuHelper_->CacheDiagonalsWithPreRotationOnGPU(preloadedDiagonals, numMatrices, VECTOR_DIM, static_cast<int>(VECTOR_DIM));

  auto endTime = chrono::steady_clock::now();
  double elapsed = chrono::duration<double>(endTime - startTime).count();
  cout << "[Approach 51] Cached diagonals on GPU in " << fixed << setprecision(2) << elapsed << "s" << endl;
}

void DiagonalSenderGPU::streamDiagonalsFromDisk(const string& serialDir) {
  if (!isGPUReady()) {
    cerr << "[Approach 51] Cannot stream diagonals - GPU not ready" << endl;
    return;
  }

  if (gpuCacheInitialized_) {
    cout << "[Approach 51] GPU cache already initialized" << endl;
    return;
  }

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

  cout << "[Approach 51] Streaming diagonals disk→GPU (memory-efficient mode)..." << endl;
  auto startTime = chrono::steady_clock::now();

  // n1 = VECTOR_DIM for pure diagonal (no BSGS decomposition)
  bool success = gpuHelper_->StreamDiagonalsToGPU(serialDir, VECTOR_DIM, numMatrices, static_cast<int>(VECTOR_DIM));

  auto endTime = chrono::steady_clock::now();
  double elapsed = chrono::duration<double>(endTime - startTime).count();

  if (success) {
    gpuCacheInitialized_ = true;
    cout << "[Approach 51] Streamed " << (numMatrices * VECTOR_DIM)
         << " diagonals disk→GPU in " << fixed << setprecision(2) << elapsed << "s" << endl;
  } else {
    cerr << "[Approach 51] Failed to stream diagonals from disk" << endl;
  }
}

Ciphertext<DCRTPoly> DiagonalSenderGPU::membershipScenario(vector<Ciphertext<DCRTPoly>> &queryCipher) {
  cout << "[Approach 51] Running HyDia-GPU membership scenario..." << endl;

  if (!isGPUReady() || !gpuHelper_->areDiagonalsCached()) {
    cout << "[Approach 51] GPU not ready, falling back to CPU" << endl;
    return DiagonalSender::membershipScenario(queryCipher);
  }

  auto totalStart = chrono::steady_clock::now();
  queryCipherForClone_ = queryCipher[0];

  if (!gpuHelper_->areBabyStepsCached()) {
    // For pure diagonal mode, cache all VECTOR_DIM rotations on GPU.
    gpuHelper_->ComputeAndCacheBabyStepsOnGPU(queryCipher[0], static_cast<int>(VECTOR_DIM));
  }

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

  Ciphertext<DCRTPoly> membershipCipher = gpuHelper_->EvalSimilarityChebyshevAndAggregateOnGPU(
      numMatrices,
      VECTOR_DIM,
      queryCipherForClone_,
      MATCH_THRESHOLD,
      COMP_DEPTH_GPU,
      batchSize);

  auto totalEnd = chrono::steady_clock::now();
  double totalTime = chrono::duration<double>(totalEnd - totalStart).count();
  cout << "[Approach 51] Membership total: " << fixed << setprecision(2) << totalTime << "s" << endl;

  return membershipCipher;
}

vector<Ciphertext<DCRTPoly>> DiagonalSenderGPU::indexScenario(vector<Ciphertext<DCRTPoly>> &queryCipher) {
  cout << "[Approach 51] Running HyDia-GPU index scenario..." << endl;

  if (!isGPUReady() || !gpuHelper_->areDiagonalsCached()) {
    cout << "[Approach 51] GPU not ready, falling back to CPU" << endl;
    return DiagonalSender::indexScenario(queryCipher);
  }

  auto totalStart = chrono::steady_clock::now();
  queryCipherForClone_ = queryCipher[0];

  if (!gpuHelper_->areBabyStepsCached()) {
    gpuHelper_->ComputeAndCacheBabyStepsOnGPU(queryCipher[0], static_cast<int>(VECTOR_DIM));
  }

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

  vector<Ciphertext<DCRTPoly>> scoreCipher = gpuHelper_->EvalSimilarityAndChebyshevBatchedOnGPU(
      numMatrices,
      VECTOR_DIM,
      queryCipherForClone_,
      MATCH_THRESHOLD,
      COMP_DEPTH_GPU);

  auto totalEnd = chrono::steady_clock::now();
  double totalTime = chrono::duration<double>(totalEnd - totalStart).count();
  cout << "[Approach 51] Index total: " << fixed << setprecision(2) << totalTime << "s" << endl;

  return scoreCipher;
}
