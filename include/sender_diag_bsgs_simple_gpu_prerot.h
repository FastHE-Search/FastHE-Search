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
// ** sender_diag_bsgs_simple_gpu_prerot.h: GPU BSGS sender with enroller-side pre-rotation (Approach 812)
//
// Variant of Approach 81 where diagonals were pre-rotated in plaintext by the
// enroller, so the server only needs to bulk-upload them to GPU without any
// homomorphic rotation.  The online query pipeline (baby steps, giant-step
// accumulation, Chebyshev, EvalSum) is identical to Approach 81.
//
// Key differences from Approach 81:
//   - No negative rotation keys generated or stored
//   - No homomorphic pre-rotation of diagonals on GPU
//   - Enrollment data produced by DiagonalBSGSPreRotEnroller

#pragma once

#include "sender_diag_bsgs_simple_gpu.h"

class DiagonalBSGSSimpleSenderGPU812 : public DiagonalBSGSSimpleSenderGPU {
  public:
	DiagonalBSGSSimpleSenderGPU812(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam, GPUHydiaHelper* gpuHelper);

	/// Initialize GPU cache using pre-rotated diagonals loaded from files (bulk mode)
	void initializeGPUCache();
	void initializeGPUCache(const std::vector<Ciphertext<DCRTPoly>>& preloadedDiagonals);

	/// Stream pre-rotated diagonals from disk → GPU (no rotation on GPU)
	void streamDiagonalsFromDisk(const std::string& serialDir);

  private:
	void cacheDiagonalsOnGPU();
	void cacheDiagonalsOnGPU(const std::vector<Ciphertext<DCRTPoly>>& preloadedDiagonals);
};
