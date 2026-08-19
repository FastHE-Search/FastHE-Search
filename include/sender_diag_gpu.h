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
#pragma once

#include "sender_diag.h"

// Forward declaration
class GPUFastHESearchHelper;

/**
 * FastHE-Search-GPU sender (Approach 51).
 *
 * Ports the legacy GPU diagonal pipeline from improved-hydia with the
 * minimum required integration for light-hydia. Uses GPU helper cache and
 * lazy GPU pipeline while preserving all existing approaches unchanged.
 */
class DiagonalSenderGPU : public DiagonalSender {
  public:
	DiagonalSenderGPU(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam, GPUFastHESearchHelper* gpuHelper);

	~DiagonalSenderGPU();

	Ciphertext<DCRTPoly> membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	vector<Ciphertext<DCRTPoly>> indexScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) override;

	void initializeGPUCache();
	void initializeGPUCache(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals);

	// Stream diagonals from disk directly to GPU (one-by-one, saves ~15 GB CPU RAM)
	void streamDiagonalsFromDisk(const std::string& serialDir);

	bool isGPUReady() const;

  private:
	GPUFastHESearchHelper* gpuHelper_;
	bool gpuCacheInitialized_;
	Ciphertext<DCRTPoly> queryCipherForClone_;

	void cacheDiagonalsOnGPU();
	void cacheDiagonalsOnGPU(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals);
};
