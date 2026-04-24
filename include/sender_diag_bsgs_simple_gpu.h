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
// ** sender_diag_bsgs_simple_gpu.h: GPU-accelerated Simple BSGS sender (Approach 81)
// GPU version using lazy-relin BSGS on GPU with pre-rotated diagonals.
// 
// Pipeline:
// 1. Pre-rotate and cache diagonals on GPU (one-time setup)
// 2. Upload query baby-step rotations to GPU
// 3. Lazy-relin BSGS: multNoRelin + accumulate, then single relinearize per giant step
// 4. Chebyshev comparison (Paterson-Stockmeyer, fully on GPU)
// 5. EvalAddMany + EvalSum (membership) or bulk download (index)

#pragma once

#include "sender_diag_bsgs_simple.h"

// Forward declaration
class GPUHydiaHelper;

class DiagonalBSGSSimpleSenderGPU : public DiagonalBSGSSimpleSender {
public:
  // Constructor (no sk param - light-hydia's DiagonalSender doesn't take it)
  DiagonalBSGSSimpleSenderGPU(CryptoContext<DCRTPoly> ccParam, 
                               PublicKey<DCRTPoly> pkParam, 
                               size_t vectorParam,
                               GPUHydiaHelper* gpuHelper);
  
  // Destructor
  ~DiagonalBSGSSimpleSenderGPU();

  // Override membershipScenario to use GPU
  Ciphertext<DCRTPoly> 
  membershipScenario(vector<Ciphertext<DCRTPoly>> &queryCipher) override;

  // Override indexScenario to use GPU
  vector<Ciphertext<DCRTPoly>> 
  indexScenario(vector<Ciphertext<DCRTPoly>> &queryCipher) override;

  // Initialize GPU caching (call once before queries)
  void initializeGPUCache();
  
  // Initialize GPU caching with preloaded diagonals (avoids file I/O)
  void initializeGPUCache(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals);
  
  // Stream diagonals from disk directly to GPU (one-by-one, saves ~15 GB CPU RAM)
  void streamDiagonalsFromDisk(const std::string& serialDir);

  // Check if GPU is ready
  bool isGPUReady() const;

protected:
  GPUHydiaHelper* gpuHelper_;
  bool gpuCacheInitialized_;
  bool rotationsCached_;
  Ciphertext<DCRTPoly> queryCipherForClone_;

private:
  // Load and cache diagonals on GPU (same format as Approach 5/9)
  void cacheDiagonalsOnGPU();
  void cacheDiagonalsOnGPU(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals);
};
