//  clang-format : off
//
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
//
// clang-format : on

#include "../include/gpu_debug_profiler.h"
#include "../include/gpu_hydia_helper.cuh"
#include "../include/openFHE_wrapper.h"
#include <openfhe.h>

#include <algorithm> // std::min, std::max for stream pool sizing
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <omp.h> // OMP parallel deserialization in streaming mode
#include <stdexcept>

// Set to 1 to enable GPU Chebyshev, 0 to fallback to CPU
#define USE_GPU_CHEBYSHEV 1

// Set to 1 to enable debug output, 0 to disable
// NOTE: For comprehensive profiling, use GPU_DEBUG_PROFILER_ENABLED in gpu_debug_profiler.h
#define GPU_HYDIA_DEBUG 0

#ifdef USE_GPU_ACCELERATION
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"

// Helper function to estimate GPU memory requirements
static size_t EstimateCiphertextGPUMemory(size_t N, size_t L) {
	// Each ciphertext: 2 polynomials × L limbs × N coefficients × 8 bytes
	// FIDESlib needs additional buffers for NTT, key switching, etc.
	// Empirical factor: ~2.5x for working memory overhead during upload
	return static_cast<size_t>(2 * L * N * 8 * 2.5);
}

[[maybe_unused]] static size_t GetAvailableGPUMemory() {
	size_t free_bytes  = 0;
	size_t total_bytes = 0;
	cudaError_t err	   = cudaMemGetInfo(&free_bytes, &total_bytes);
	if (err != cudaSuccess) {
		std::cerr << "[GPUHydiaHelper] Warning: Could not query GPU memory" << std::endl;
		return 0;
	}
	return free_bytes;
}

// Compute Chebyshev coefficients for a function using DCT-like approach
static std::vector<double> ComputeChebyshevCoefficients(std::function<double(double)> func, double a, double b, uint32_t degree) {

	std::vector<double> coeffs(degree + 1, 0.0);
	uint32_t n = degree + 1;
	std::vector<double> fvals(n);

	for (uint32_t k = 0; k < n; ++k) {
		double x = std::cos(M_PI * (2.0 * k + 1.0) / (2.0 * n));
		double t = 0.5 * ((b - a) * x + (b + a));
		fvals[k] = func(t);
	}

	for (uint32_t j = 0; j <= degree; ++j) {
		double sum = 0.0;
		for (uint32_t k = 0; k < n; ++k) {
			sum += fvals[k] * std::cos(M_PI * j * (2.0 * k + 1.0) / (2.0 * n));
		}
		coeffs[j] = (2.0 / n) * sum;
	}
	return coeffs;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

GPUHydiaHelper::GPUHydiaHelper(CryptoContext<DCRTPoly> cc, const KeyPair<DCRTPoly>& keys, const std::vector<int>& gpuDevices, int batchSize, double gpuSafeMemoryLimitGB)
: cc_(cc), keys_(keys), gpuContextReady_(false), batchSize_(batchSize), diagonalsCached_(false), babyStepsCached_(false), numCachedDiagonals_(0),
  numCachedGroups_(0), perGroupCaching_(false), diagonalsPreRotated_(false), preRotationN1_(0) {

	// Initialize unified GPU safe memory limit (single source of truth)
	if (gpuSafeMemoryLimitGB > 0) {
		gpuSafeMemoryLimitGB_ = gpuSafeMemoryLimitGB;
	} else {
		const char* env = std::getenv("GPU_SAFE_MEMORY_GB");
		if (env) {
			gpuSafeMemoryLimitGB_ = std::atof(env);
		} else {
			size_t freeMem = 0, totalMem = 0;
			cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
			if (err == cudaSuccess && totalMem > 0) {
				gpuSafeMemoryLimitGB_ = static_cast<double>(totalMem) / (1024.0 * 1024.0 * 1024.0) * GPU_MEMORY_SAFE_FRACTION;
			} else {
				gpuSafeMemoryLimitGB_ = 48.0 * GPU_MEMORY_SAFE_FRACTION; // Fallback: RTX 8000
			}
		}
	}
	std::cout << "[GPUHydiaHelper] GPU safe memory limit: " << std::fixed << std::setprecision(2) << gpuSafeMemoryLimitGB_ << " GB" << std::endl;

	try {
		std::cout << "[GPUHydiaHelper] Initializing GPU context with FIDESlib v2..." << std::endl;

		::FIDESlib::CKKS::Parameters fidesParams;
		fidesParams.batch = batchSize_;

		::FIDESlib::CKKS::RawParams rawParams = ::FIDESlib::CKKS::GetRawParams(cc_);
		auto adaptedParams					  = fidesParams.adaptTo(rawParams);

		std::cout << "[GPUHydiaHelper] Using GPU device 0 of " << gpuDevices.size() << ": " << gpuDevices[0] << std::endl;

		gpuContext_									 = FIDESlib::CKKS::GenCryptoContextGPU(adaptedParams, gpuDevices);
		::FIDESlib::CKKS::RawKeySwitchKey rawKskEval = ::FIDESlib::CKKS::GetEvalKeySwitchKey(keys_);
		kskEval_									 = std::make_unique<FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
		kskEval_->Initialize(rawKskEval);
		gpuContext_->AddEvalKey(std::move(*kskEval_));

		gpuContextReady_ = true;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] ✗ GPU initialization failed: " << e.what() << std::endl;
		std::cerr << "[GPUHydiaHelper] Falling back to CPU operations" << std::endl;
		gpuContextReady_ = false;
	}
}

GPUHydiaHelper::~GPUHydiaHelper() {
	if (gpuContextReady_) {
		std::cout << "[GPUHydiaHelper] Cleaning up GPU resources..." << std::endl;

		// Synchronize GPU before cleanup
		cudaDeviceSynchronize();

		// Destroy CUDA stream pool
		DestroyMatrixStreams();

		// NOTE: We intentionally don't delete the FIDESlib ciphertexts here.
		// FIDESlib ciphertexts hold references to the Context, and there appears
		// to be an issue with the order of destruction that causes double-free errors.
		// The OS will reclaim all memory at process exit anyway.

		cachedGPUDiagonals_.clear();
		diagonalsCached_ = false;

		cachedGPUBabySteps_.clear();
		babyStepsCached_ = false;

		std::cout << "[GPUHydiaHelper] GPU cleanup complete." << std::endl;
	}
	gpuContextReady_ = false;
}

// =============================================================================
// Diagonal caching
// =============================================================================

bool GPUHydiaHelper::areDiagonalsCached() const {
	return diagonalsCached_;
}

// APPROACH 81: Cache diagonals with pre-rotation for on-demand BSGS
void GPUHydiaHelper::CacheDiagonalsWithPreRotationOnGPU(const std::vector<Ciphertext<DCRTPoly>>& diagonals, size_t numMatrices, size_t vectorDim, int n1) {

	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot cache diagonals" << std::endl;
		return;
	}

	size_t totalDiags = numMatrices * vectorDim;
	if (diagonals.size() != totalDiags) {
		std::cerr << "[GPUHydiaHelper] Diagonal count mismatch: got " << diagonals.size() << " expected " << totalDiags << std::endl;
		return;
	}

	// Memory check removed — preflight in main.cpp handles estimation.
	// CUDA runtime OOM will throw if we exceed GPU memory during upload.
	size_t N = cc_->GetRingDimension();
	size_t L = cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();

	try {
		std::cout << "[GPUHydiaHelper] Caching " << totalDiags << " diagonals with pre-rotation (n1=" << n1 << ") on GPU..." << std::flush;
		auto startTime = std::chrono::steady_clock::now();

		// Clear any existing cache
		for (void* ptr : cachedGPUDiagonals_) {
			delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();

		// Initialize rotation keys needed for pre-rotation
		// We need keys for -(j*n1) for j = 1, 2, ..., ceil(vectorDim/n1)-1
		int numGiantSteps = (static_cast<int>(vectorDim) + n1 - 1) / n1;
		for (int j = 1; j < numGiantSteps; ++j) {
			int giantStep = j * n1;
			if (giantStep >= static_cast<int>(vectorDim))
				continue;
			int negRot = -giantStep;
			if (kskRotations_.find(negRot) == kskRotations_.end()) {
				::FIDESlib::CKKS::RawKeySwitchKey rawKskRot = ::FIDESlib::CKKS::GetRotationKeySwitchKey(keys_.publicKey, negRot);
				kskRotations_[negRot]						= std::make_unique<::FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[negRot]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(negRot, std::move(*kskRotations_[negRot]));
			}
		}

		[[maybe_unused]] size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);

		// Upload and pre-rotate diagonals
		for (size_t m = 0; m < numMatrices; ++m) {
			for (size_t k = 0; k < vectorDim; ++k) {
				size_t flatIdx = m * vectorDim + k;

				FIDESlib::CKKS::RawCipherText raw	= FIDESlib::CKKS::GetRawCipherText(cc_, diagonals[flatIdx]);
				FIDESlib::CKKS::Ciphertext* gpuDiag = new FIDESlib::CKKS::Ciphertext(gpuContext_, raw);
				GPU_TRACK_CPU_TO_GPU(ctSizeBytes, "CKKS::Ciphertext", "prerot diagonal[" + std::to_string(flatIdx) + "]");

				// Pre-rotate: rotate by -(k - k%n1) = -(giantStep for this k)
				int giantStep = (static_cast<int>(k) / n1) * n1;
				if (giantStep > 0) {
					int negRot = -giantStep;
					gpuDiag->rotate(negRot);
					// Free special limbs — disabled: causes BSGS-GPU perf regression
					// gpuDiag->c0.freeSpecialLimbs();
					// gpuDiag->c1.freeSpecialLimbs();
				}

				cachedGPUDiagonals_.push_back(static_cast<void*>(gpuDiag));
			}
		}

		cudaDeviceSynchronize();

		auto endTime   = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		diagonalsCached_	 = true;
		diagonalsPreRotated_ = true;
		preRotationN1_		 = n1;
		perGroupCaching_	 = false;
		numCachedDiagonals_	 = cachedGPUDiagonals_.size();
		numCachedGroups_	 = 1;

		std::cout << " done (" << std::fixed << std::setprecision(2) << elapsed << "s)" << std::endl;
		std::cout << "[GPUHydiaHelper] ✓ " << cachedGPUDiagonals_.size() << " diagonals pre-rotated and cached on GPU" << std::endl;

		// Free pre-rotation keys (negative indices) — no longer needed after diags are pre-rotated
		int preRotKeysFreed = 0;
		for (auto it = kskRotations_.begin(); it != kskRotations_.end();) {
			if (it->first < 0) {
				it = kskRotations_.erase(it);
				preRotKeysFreed++;
			} else {
				++it;
			}
		}
		if (preRotKeysFreed > 0) {
			std::cout << "[GPUHydiaHelper] Freed " << preRotKeysFreed << " pre-rotation keys (no longer needed)" << std::endl;
		}
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to cache diagonals: " << e.what() << std::endl;
		// Free partially uploaded GPU diagonals
		for (void* ptr : cachedGPUDiagonals_) {
			if (ptr)
				delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();
		cudaDeviceSynchronize();
		diagonalsCached_ = false;
		std::cerr << "[GPUHydiaHelper] Cleaned up partial GPU diagonal cache" << std::endl;
	}
}

// =============================================================================
// APPROACH 812: Upload pre-rotated diagonals (NO homomorphic rotation on GPU)
// The enroller already applied plaintext cyclic shifts before encryption.
// =============================================================================

void GPUHydiaHelper::CachePreRotatedDiagonalsOnGPU(const std::vector<Ciphertext<DCRTPoly>>& diagonals, size_t numMatrices, size_t vectorDim, int n1) {

	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot cache diagonals" << std::endl;
		return;
	}

	size_t totalDiags = numMatrices * vectorDim;
	if (diagonals.size() != totalDiags) {
		std::cerr << "[GPUHydiaHelper] Diagonal count mismatch: got " << diagonals.size() << " expected " << totalDiags << std::endl;
		return;
	}

	size_t N = cc_->GetRingDimension();
	size_t L = cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();

	try {
		std::cout << "[GPUHydiaHelper] Uploading " << totalDiags << " pre-rotated diagonals to GPU (no server-side rotation, n1=" << n1 << ")..." << std::flush;
		auto startTime = std::chrono::steady_clock::now();

		// Clear any existing cache
		for (void* ptr : cachedGPUDiagonals_) {
			delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();

		[[maybe_unused]] size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);

		// Upload diagonals — NO rotation needed (already pre-rotated in plaintext)
		for (size_t i = 0; i < totalDiags; ++i) {
			FIDESlib::CKKS::RawCipherText raw	= FIDESlib::CKKS::GetRawCipherText(cc_, diagonals[i]);
			FIDESlib::CKKS::Ciphertext* gpuDiag = new FIDESlib::CKKS::Ciphertext(gpuContext_, raw);
			GPU_TRACK_CPU_TO_GPU(ctSizeBytes, "CKKS::Ciphertext", "prerot diagonal[" + std::to_string(i) + "]");
			cachedGPUDiagonals_.push_back(static_cast<void*>(gpuDiag));
		}

		cudaDeviceSynchronize();

		auto endTime   = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		diagonalsCached_	 = true;
		diagonalsPreRotated_ = true;
		preRotationN1_		 = n1;
		perGroupCaching_	 = false;
		numCachedDiagonals_	 = cachedGPUDiagonals_.size();
		numCachedGroups_	 = 1;

		std::cout << " done (" << std::fixed << std::setprecision(2) << elapsed << "s)" << std::endl;
		std::cout << "[GPUHydiaHelper] ✓ " << cachedGPUDiagonals_.size() << " pre-rotated diagonals uploaded to GPU (zero homomorphic rotations)" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to cache pre-rotated diagonals: " << e.what() << std::endl;
		for (void* ptr : cachedGPUDiagonals_) {
			if (ptr)
				delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();
		cudaDeviceSynchronize();
		diagonalsCached_ = false;
		std::cerr << "[GPUHydiaHelper] Cleaned up partial GPU diagonal cache" << std::endl;
	}
}

bool GPUHydiaHelper::StreamPreRotatedDiagonalsToGPU(const std::string& serialDir, size_t numDiagonals, size_t numMatrices, int n1) {
	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot stream diagonals" << std::endl;
		return false;
	}

	size_t totalDiagonals = numDiagonals * numMatrices;
	size_t N			  = cc_->GetRingDimension();
	size_t L			  = cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();

	std::cout << "[GPUHydiaHelper] Streaming " << totalDiagonals << " pre-rotated diagonals to GPU (no server-side rotation)..." << std::endl;

	try {
		auto startTime = std::chrono::steady_clock::now();

		// Clear any existing cache
		for (void* ptr : cachedGPUDiagonals_) {
			delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();
		cachedGPUDiagonals_.reserve(totalDiagonals);

		// NO pre-rotation keys needed — diagonals already pre-rotated by enroller

		[[maybe_unused]] size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);
		size_t loaded						= 0;
		size_t errors						= 0;

		// OMP-parallel streaming (same batching pattern as StreamDiagonalsToGPU)
		size_t ompThreads = static_cast<size_t>(omp_get_max_threads());
		size_t batchSize  = std::min(ompThreads, totalDiagonals);

		const char* batchEnv = std::getenv("GPU_STREAM_BATCH");
		if (batchEnv) {
			int val = std::atoi(batchEnv);
			if (val > 0)
				batchSize = static_cast<size_t>(val);
		}
		batchSize = std::max(batchSize, static_cast<size_t>(1));
		batchSize = std::min(batchSize, totalDiagonals);

		std::cout << "[GPUHydiaHelper] OMP-parallel streaming: batch=" << batchSize << ", threads=" << ompThreads << " (no rotation)" << std::endl;

		for (size_t batchStart = 0; batchStart < totalDiagonals; batchStart += batchSize) {
			size_t batchEnd		= std::min(batchStart + batchSize, totalDiagonals);
			size_t currentBatch = batchEnd - batchStart;

			// Phase 1: OMP parallel deserialization from disk
			std::vector<Ciphertext<DCRTPoly>> batchCiphers(currentBatch);
			std::vector<int> loadOk(currentBatch, 0);

#pragma omp parallel for num_threads(ompThreads) schedule(dynamic)
			for (size_t i = 0; i < currentBatch; ++i) {
				size_t flatIdx		 = batchStart + i;
				std::string filepath = serialDir + "/db_diagonal/index" + std::to_string(flatIdx) + ".bin";
				if (Serial::DeserializeFromFile(filepath, batchCiphers[i], SerType::BINARY)) {
					loadOk[i] = 1;
				}
			}

			// Phase 2: Sequential GPU upload — NO rotation
			for (size_t i = 0; i < currentBatch; ++i) {
				size_t flatIdx = batchStart + i;

				if (!loadOk[i]) {
					if (errors < 5) {
						std::cerr << "[GPUHydiaHelper] Error loading diagonal flatIdx=" << flatIdx << std::endl;
					}
					errors++;
					cachedGPUDiagonals_.push_back(nullptr);
					continue;
				}

				// Upload to GPU — no rotation needed
				FIDESlib::CKKS::RawCipherText raw	= FIDESlib::CKKS::GetRawCipherText(cc_, batchCiphers[i]);
				FIDESlib::CKKS::Ciphertext* gpuDiag = new FIDESlib::CKKS::Ciphertext(gpuContext_, raw);
				GPU_TRACK_CPU_TO_GPU(ctSizeBytes, "CKKS::Ciphertext", "prerot diagonal[" + std::to_string(flatIdx) + "]");

				cachedGPUDiagonals_.push_back(static_cast<void*>(gpuDiag));
				batchCiphers[i] = Ciphertext<DCRTPoly>();

				loaded++;
			}

			std::cout << "\r[GPUHydiaHelper] Streamed " << loaded << "/" << totalDiagonals << " diagonals (" << std::fixed << std::setprecision(1)
					  << (100.0 * loaded / totalDiagonals) << "%)" << std::flush;
		}
		std::cout << std::endl;

		if (errors > 0) {
			std::cerr << "[GPUHydiaHelper] Total load errors: " << errors << "/" << totalDiagonals << std::endl;
		}

		cudaDeviceSynchronize();

		auto endTime   = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		diagonalsCached_	 = true;
		diagonalsPreRotated_ = true;
		preRotationN1_		 = n1;
		perGroupCaching_	 = false;
		numCachedDiagonals_	 = cachedGPUDiagonals_.size();
		numCachedGroups_	 = 1;

		std::cout << "[GPUHydiaHelper] ✓ Streamed " << loaded << " pre-rotated diagonals to GPU in " << std::fixed << std::setprecision(2) << elapsed << "s"
				  << " (zero homomorphic rotations, zero negative rotation keys)" << std::endl;

		return errors == 0;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to stream pre-rotated diagonals: " << e.what() << std::endl;
		for (void* ptr : cachedGPUDiagonals_) {
			if (ptr)
				delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();
		cudaDeviceSynchronize();
		diagonalsCached_ = false;
		return false;
	}
}

// =============================================================================
// Baby step computation and caching
// =============================================================================

void GPUHydiaHelper::ComputeAndCacheBabyStepsOnGPU(const Ciphertext<DCRTPoly>& queryVector, int n1) {

	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot compute baby steps" << std::endl;
		throw std::runtime_error("GPU not initialized");
	}

	GPU_SCOPED_TIMER("ComputeAndCacheBabyStepsOnGPU_TOTAL");

	try {
		std::cout << "[GPUHydiaHelper] Computing " << n1 << " baby steps directly on GPU..." << std::flush;

		auto startTime = std::chrono::high_resolution_clock::now();

		// =====================================================================
		// PHASE 1: Upload query to GPU
		// =====================================================================
		GPU_PROFILE_START("BabyStep_Phase1_Upload");
		size_t N							= cc_->GetRingDimension();
		size_t L							= cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();
		[[maybe_unused]] size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);

		FIDESlib::CKKS::RawCipherText rawQuery = FIDESlib::CKKS::GetRawCipherText(cc_, queryVector);
		FIDESlib::CKKS::Ciphertext* gpuQuery   = new FIDESlib::CKKS::Ciphertext(gpuContext_, rawQuery);
		GPU_TRACK_CPU_TO_GPU(ctSizeBytes, "CKKS::Ciphertext", "query vector for baby steps");
		GPU_PROFILE_END("BabyStep_Phase1_Upload");

		// =====================================================================
		// PHASE 2: Initialize rotation keys (Consolidated V2 Constructor Layout)
		// =====================================================================
		GPU_PROFILE_START("BabyStep_Phase2_KeyInit");
		int keysInitialized = 0;
		int keysCached		= 0;
		for (int i = 1; i < n1; ++i) {
			if (kskRotations_.find(i) == kskRotations_.end()) {
				// Extract using the valid 3-argument signature
				FIDESlib::CKKS::RawKeySwitchKey rawKskRot = FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, i, cc_);

				// Single-step instantiation passing the raw payload and context pointer wrapper directly
				kskRotations_[i] = std::make_unique<FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[i]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(i, std::move(*kskRotations_[i]));

				keysInitialized++;
				GPU_TRACK_CPU_TO_GPU(ctSizeBytes * 3, "CKKS::KeySwitchingKey", "rotation key [" + std::to_string(i) + "]");
			} else {
				keysCached++;
			}
		}
		GPU_PROFILE_END("BabyStep_Phase2_KeyInit");

		// =====================================================================
		// PHASE 3: Compute rotations on GPU
		// =====================================================================
		GPU_PROFILE_START("BabyStep_Phase3_Compute");

		// Clear existing baby step cache
		for (void* ptr : cachedGPUBabySteps_) {
			delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUBabySteps_.clear();
		cachedGPUBabySteps_.resize(n1);

		// Baby step 0 = original query (no rotation)
		cachedGPUBabySteps_[0] = static_cast<void*>(gpuQuery); // Transfer ownership

		// Baby steps 1 to n1-1 = rotations of query
		for (int i = 1; i < n1; ++i) {
			FIDESlib::CKKS::Ciphertext* rotated = new FIDESlib::CKKS::Ciphertext(gpuContext_);
			rotated->copy(*gpuQuery);
			rotated->rotate(i);
			// Free transient special limbs after rotation — disabled: causes BSGS-GPU perf regression
			// rotated->c0.freeSpecialLimbs();
			// rotated->c1.freeSpecialLimbs();
			GPU_COUNT_ROTATE();
			cachedGPUBabySteps_[i] = static_cast<void*>(rotated);
		}

		cudaDeviceSynchronize();
		GPU_PROFILE_END("BabyStep_Phase3_Compute");

		auto endTime   = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		babyStepsCached_ = true;

		std::cout << " done (" << std::fixed << std::setprecision(2) << elapsed << "s)" << std::endl;
		std::cout << "[GPUHydiaHelper] ✓ " << n1 << " baby steps computed and cached on GPU" << std::endl;
		std::cout << "  Rotation keys: " << keysInitialized << " initialized, " << keysCached << " already cached" << std::endl;

		// Free baby-step rotation keys that are NOT powers of 2 (EvalSum candidates).
		// Baby-step ciphertexts are already computed and cached — the rotation keys
		// themselves are never used again in any pipeline operation.
		int babyKeysFreed = 0;
		for (int i = 1; i < n1; ++i) {
			if ((i & (i - 1)) != 0) { // Not a power of 2 → not an EvalSum key
				auto it = kskRotations_.find(i);
				if (it != kskRotations_.end()) {
					kskRotations_.erase(it);
					babyKeysFreed++;
				}
			}
		}
		if (babyKeysFreed > 0) {
			std::cout << "[GPUHydiaHelper] Freed " << babyKeysFreed << " baby-step rotation keys (no longer needed, kept "
					  << (keysInitialized + keysCached - babyKeysFreed) << " EvalSum-overlap keys)" << std::endl;
		}
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to compute baby steps: " << e.what() << std::endl;
		// Free partially uploaded GPU baby steps
		for (void* ptr : cachedGPUBabySteps_) {
			if (ptr)
				delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUBabySteps_.clear();
		cudaDeviceSynchronize();
		babyStepsCached_ = false;
		std::cerr << "[GPUHydiaHelper] Cleaned up partial GPU baby step cache" << std::endl;
		throw;
	}
}

void GPUHydiaHelper::CacheBabyStepsOnGPU(const std::vector<Ciphertext<DCRTPoly>>& babySteps) {
	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot cache baby steps" << std::endl;
		return;
	}

	try {
		std::cout << "[GPUHydiaHelper] Caching " << babySteps.size() << " baby steps on GPU..." << std::flush;
		auto startTime = std::chrono::steady_clock::now();

		for (void* ptr : cachedGPUBabySteps_) {
			delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUBabySteps_.clear();

		for (const auto& baby : babySteps) {
			FIDESlib::CKKS::RawCipherText raw	= FIDESlib::CKKS::GetRawCipherText(cc_, baby);
			FIDESlib::CKKS::Ciphertext* gpuBaby = new FIDESlib::CKKS::Ciphertext(gpuContext_, raw);
			cachedGPUBabySteps_.push_back(static_cast<void*>(gpuBaby));
		}

		cudaDeviceSynchronize();

		auto endTime   = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		babyStepsCached_ = true;
		std::cout << " done (" << std::fixed << std::setprecision(2) << elapsed << "s)" << std::endl;
		std::cout << "[GPUHydiaHelper] ✓ " << cachedGPUBabySteps_.size() << " baby steps cached on GPU memory" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to cache baby steps: " << e.what() << std::endl;
		// Free partially uploaded GPU baby steps
		for (void* ptr : cachedGPUBabySteps_) {
			if (ptr)
				delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUBabySteps_.clear();
		cudaDeviceSynchronize();
		babyStepsCached_ = false;
		std::cerr << "[GPUHydiaHelper] Cleaned up partial GPU baby step cache" << std::endl;
	}
}

bool GPUHydiaHelper::areBabyStepsCached() const {
	return babyStepsCached_;
}

// =============================================================================
// Rotation key initialization
// =============================================================================

// APPROACH 81: Pre-initialize Simple BSGS rotation keys on GPU
void GPUHydiaHelper::InitializeSimpleBSGSRotationKeysOnGPU(int n1, int vectorDim, size_t numSlots) {

	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot initialize BSGS rotation keys" << std::endl;
		throw std::runtime_error("GPU not initialized");
	}

	try {
		std::cout << "[GPUHydiaHelper] Pre-initializing Simple BSGS rotation keys on GPU..." << std::endl;
		std::cout << "  Parameters: n1=" << n1 << " vectorDim=" << vectorDim << " numSlots=" << numSlots << std::endl;

		auto startTime	  = std::chrono::high_resolution_clock::now();
		int giantKeysInit = 0, evalSumKeysInit = 0;

		// NOTE: Baby-step keys (1..n1-1) are NOT uploaded to GPU.
		// A81 computes baby steps on CPU via cc->EvalRotate, then uploads
		// the resulting ciphertexts via CacheBabyStepsOnGPU.  The GPU rotation
		// keys for baby-step indices are never consumed by any .rotate() call.

		// 1. Giant step keys: n1, 2*n1, 3*n1, ...
		int numGiantSteps = (vectorDim + n1 - 1) / n1;
		for (int j = 1; j < numGiantSteps; ++j) {
			int giantStep = j * n1;
			if (giantStep < vectorDim && kskRotations_.find(giantStep) == kskRotations_.end()) {
				FIDESlib::CKKS::RawKeySwitchKey rawKskRot = FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, giantStep, cc_);

				// Single-step instantiation passing the raw payload and context pointer wrapper directly
				kskRotations_[giantStep] = std::make_unique<FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[giantStep]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(giantStep, std::move(*kskRotations_[giantStep]));
				giantKeysInit++;
			}
		}

		// 3. EvalSum keys: powers of 2
		for (size_t shift = 1; shift < numSlots; shift *= 2) {
			int idx = static_cast<int>(shift);
			if (kskRotations_.find(idx) == kskRotations_.end()) {
				FIDESlib::CKKS::RawKeySwitchKey rawKskRot = FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, idx, cc_);

				kskRotations_[idx] = std::make_unique<FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[idx]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(idx, std::move(*kskRotations_[idx]));

				evalSumKeysInit++;
			}
		}

		auto endTime   = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		int totalKeys = giantKeysInit + evalSumKeysInit;
		std::cout << "[GPUHydiaHelper] ✓ Simple BSGS rotation keys initialized: " << totalKeys << " total" << std::endl;
		std::cout << "  Giant: " << giantKeysInit << ", EvalSum: " << evalSumKeysInit << " (baby-step keys skipped — computed on CPU)" << std::endl;
		std::cout << "  Total time: " << std::fixed << std::setprecision(2) << elapsed << "s" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to initialize Simple BSGS rotation keys: " << e.what() << std::endl;
		throw;
	}
}

// =============================================================================
// CUDA Stream Parallelism — lazy allocation (only when needed, exactly sized)
// =============================================================================

void GPUHydiaHelper::EnsureMatrixStreams(size_t numNeeded) {
	// No pool needed for single-matrix workloads
	if (numNeeded <= 1)
		return;

	// Already have enough streams for this workload?
	if (streamsInitialized_ && numMatrixStreams_ >= numNeeded)
		return;

	try {
		int deviceId = 0;
		cudaGetDevice(&deviceId);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceId);

		// Compute caps:
		// - SM-based: SMs / 4, typically 18 for RTX 8000
		// - Memory: 20% of free GPU reserved for concurrent stream intermediates
		//   (each in-flight matrix needs ~40× ciphertext size)
		// - Demand: never create more streams than matrices
		size_t freeMem = 0, totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);

		size_t N			  = cc_->GetRingDimension();
		size_t L			  = cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();
		size_t ctBytes		  = EstimateCiphertextGPUMemory(N, L);
		size_t perMatrixBytes = 40 * ctBytes;

		size_t availableForStreams = static_cast<size_t>(freeMem * 0.2);
		size_t memLimitedStreams   = (availableForStreams > 0 && perMatrixBytes > 0) ? availableForStreams / perMatrixBytes : 2;

		size_t smBasedStreams = static_cast<size_t>(std::max(2, prop.multiProcessorCount / 4));

		// Target = min(demand, SM cap, memory cap, hard cap)
		size_t target = std::min({ numNeeded, smBasedStreams, memLimitedStreams, static_cast<size_t>(32) });
		target		  = std::max(target, static_cast<size_t>(2)); // at least 2 if we got here

		// Allow override via environment variable
		const char* envStreams = std::getenv("GPU_NUM_STREAMS");
		if (envStreams) {
			int userStreams = std::atoi(envStreams);
			if (userStreams >= 1 && userStreams <= 64) {
				target = static_cast<size_t>(userStreams);
				std::cout << "[GPUHydiaHelper] Using GPU_NUM_STREAMS=" << target << " from environment" << std::endl;
			}
		}

		// Already sufficient after computing target?
		if (streamsInitialized_ && numMatrixStreams_ >= target)
			return;

		// Destroy any existing streams before reallocating
		if (streamsInitialized_) {
			DestroyMatrixStreams();
		}

		// Create exactly `target` non-blocking streams.
		// NOTE: CUDA streams are lightweight driver queue handles (~few KB each,
		// in host memory, NOT GPU VRAM). They do NOT consume ciphertext/key memory.
		matrixStreams_.resize(target);
		for (size_t i = 0; i < target; ++i) {
			cudaError_t err = cudaStreamCreateWithFlags(&matrixStreams_[i], cudaStreamNonBlocking);
			if (err != cudaSuccess) {
				std::cerr << "[GPUHydiaHelper] Failed to create CUDA stream " << i << ": " << cudaGetErrorString(err) << std::endl;
				for (size_t j = 0; j < i; ++j) {
					cudaStreamDestroy(matrixStreams_[j]);
				}
				matrixStreams_.clear();
				streamsInitialized_ = false;
				numMatrixStreams_	= 0;
				return;
			}
		}

		numMatrixStreams_	= target;
		streamsInitialized_ = true;

		std::cout << "[GPUHydiaHelper] ✓ CUDA streams: " << target << " created on-demand" << " (needed=" << numNeeded << ", GPU: " << prop.name
				  << ", SMs: " << prop.multiProcessorCount << ", free: " << (freeMem / (1024 * 1024)) << " MB" << ", ~" << (perMatrixBytes / (1024 * 1024))
				  << " MB/concurrent matrix)" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Warning: Failed to create CUDA streams: " << e.what() << std::endl;
		streamsInitialized_ = false;
		numMatrixStreams_	= 0;
	}
}

void GPUHydiaHelper::DestroyMatrixStreams() {
	if (!streamsInitialized_)
		return;

	for (size_t i = 0; i < numMatrixStreams_; ++i) {
		cudaStreamDestroy(matrixStreams_[i]);
	}
	matrixStreams_.clear();
	numMatrixStreams_	= 0;
	streamsInitialized_ = false;
}

// =============================================================================
// StreamDiagonalsToGPU — memory-efficient disk→GPU loader
// =============================================================================

bool GPUHydiaHelper::StreamDiagonalsToGPU(const std::string& serialDir, size_t numDiagonals, size_t numMatrices, int n1) {
	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready, cannot stream diagonals" << std::endl;
		return false;
	}

	size_t totalDiagonals = numDiagonals * numMatrices;

	// Memory check removed — preflight in main.cpp handles estimation.
	// CUDA runtime OOM will throw if we exceed GPU memory during streaming.
	size_t N = cc_->GetRingDimension();
	size_t L = cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();

	std::cout << "[GPUHydiaHelper] Streaming " << totalDiagonals << " diagonals to GPU..." << std::endl;

	try {
		auto startTime = std::chrono::steady_clock::now();

		// Clear any existing cache
		for (void* ptr : cachedGPUDiagonals_) {
			delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();
		cachedGPUDiagonals_.reserve(totalDiagonals);

		// Initialize pre-rotation keys if n1 < numDiagonals (BSGS mode)
		int numGiantSteps = (static_cast<int>(numDiagonals) + n1 - 1) / n1;
		for (int j = 1; j < numGiantSteps; ++j) {
			int giantStep = j * n1;
			if (giantStep >= static_cast<int>(numDiagonals))
				continue;
			int negRot = -giantStep;
			if (kskRotations_.find(negRot) == kskRotations_.end()) {
				::FIDESlib::CKKS::RawKeySwitchKey rawKskRot = ::FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, negRot, cc_);
				kskRotations_[negRot]						= std::make_unique<::FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[negRot]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(negRot, std::move(*kskRotations_[negRot]));
			}
		}

		size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);
		size_t loaded	   = 0;
		size_t errors	   = 0;

		// --- Batched OMP-parallel streaming ---
		// Phase 1 (per batch): OMP threads deserialize B diagonals from disk in parallel
		// Phase 2 (per batch): Main thread uploads each to GPU + pre-rotates sequentially
		// Peak CPU RAM = batchSize × ctSize (much less than bulk-load's totalDiags × ctSize)
		// Storage layout: serialDir/db_diagonal/index{flatIdx}.bin
		size_t ompThreads = static_cast<size_t>(omp_get_max_threads());
		size_t batchSize  = std::min(ompThreads, totalDiagonals);

		// Allow env var override: GPU_STREAM_BATCH=<N>
		const char* batchEnv = std::getenv("GPU_STREAM_BATCH");
		if (batchEnv) {
			int val = std::atoi(batchEnv);
			if (val > 0)
				batchSize = static_cast<size_t>(val);
		}
		batchSize = std::max(batchSize, static_cast<size_t>(1));
		batchSize = std::min(batchSize, totalDiagonals);

		std::cout << "[GPUHydiaHelper] OMP-parallel streaming: batch=" << batchSize << ", threads=" << ompThreads << std::endl;

		for (size_t batchStart = 0; batchStart < totalDiagonals; batchStart += batchSize) {
			size_t batchEnd		= std::min(batchStart + batchSize, totalDiagonals);
			size_t currentBatch = batchEnd - batchStart;

			// Phase 1: OMP parallel deserialization from disk
			std::vector<Ciphertext<DCRTPoly>> batchCiphers(currentBatch);
			std::vector<int> loadOk(currentBatch, 0); // int, not bool (bool is bitfield → not thread-safe)

#pragma omp parallel for num_threads(ompThreads) schedule(dynamic)
			for (size_t i = 0; i < currentBatch; ++i) {
				size_t flatIdx		 = batchStart + i;
				std::string filepath = serialDir + "/db_diagonal/index" + std::to_string(flatIdx) + ".bin";
				if (Serial::DeserializeFromFile(filepath, batchCiphers[i], SerType::BINARY)) {
					loadOk[i] = 1;
				}
			}

			// Phase 2: Sequential GPU upload + pre-rotation
			for (size_t i = 0; i < currentBatch; ++i) {
				size_t flatIdx = batchStart + i;
				size_t k	   = flatIdx % numDiagonals;

				if (!loadOk[i]) {
					if (errors < 5) {
						std::cerr << "[GPUHydiaHelper] Error loading diagonal flatIdx=" << flatIdx << " path=" << serialDir << "/db_diagonal/index" << flatIdx
								  << ".bin" << std::endl;
					}
					errors++;
					cachedGPUDiagonals_.push_back(nullptr);
					continue;
				}

				// Upload to GPU
				FIDESlib::CKKS::RawCipherText raw	= FIDESlib::CKKS::GetRawCipherText(cc_, batchCiphers[i]);
				FIDESlib::CKKS::Ciphertext* gpuDiag = new FIDESlib::CKKS::Ciphertext(gpuContext_, raw);
				GPU_TRACK_CPU_TO_GPU(ctSizeBytes, "CKKS::Ciphertext", "streamed diagonal[" + std::to_string(flatIdx) + "]");

				// Pre-rotate: rotate by -(k - k%n1)
				int giantStep = (static_cast<int>(k) / n1) * n1;
				if (giantStep > 0) {
					int negRot = -giantStep;
					gpuDiag->rotate(negRot);
					// Free special limbs — disabled: causes BSGS-GPU perf regression
					// (was reclaiming ~24 GB of special-limb buffers for H200 OOM fix,
					//  but adds overhead that slows BSGS-GPU by ~35-75%)
					// gpuDiag->c0.freeSpecialLimbs();
					// gpuDiag->c1.freeSpecialLimbs();
				}

				cachedGPUDiagonals_.push_back(static_cast<void*>(gpuDiag));
				batchCiphers[i] = Ciphertext<DCRTPoly>(); // Release CPU copy immediately

				loaded++;
			}
			// Remaining CPU copies released when batchCiphers goes out of scope

			// Progress
			std::cout << "\r[GPUHydiaHelper] Streamed " << loaded << "/" << totalDiagonals << " diagonals (" << std::fixed << std::setprecision(1)
					  << (100.0 * loaded / totalDiagonals) << "%)" << std::flush;
		}
		std::cout << std::endl;

		if (errors > 0) {
			std::cerr << "[GPUHydiaHelper] Total load errors: " << errors << "/" << totalDiagonals << std::endl;
		}

		cudaDeviceSynchronize();

		auto endTime   = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();

		diagonalsCached_	 = true;
		diagonalsPreRotated_ = true;
		preRotationN1_		 = n1;
		perGroupCaching_	 = false;
		numCachedDiagonals_	 = cachedGPUDiagonals_.size();
		numCachedGroups_	 = 1;

		std::cout << "[GPUHydiaHelper] ✓ Streamed " << loaded << " diagonals to GPU in " << std::fixed << std::setprecision(2) << elapsed << "s" << std::endl;
		std::cout << "[GPUHydiaHelper] Peak CPU RAM: ~" << std::fixed << std::setprecision(1) << (batchSize * ctSizeBytes / (1024.0 * 1024.0 * 1024.0))
				  << " GB (" << batchSize << " diags × " << (ctSizeBytes / (1024 * 1024)) << " MB)  vs bulk-load ~"
				  << (totalDiagonals * ctSizeBytes / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

		// === Dead key cleanup: free pre-rotation keys no longer needed ===
		{
			size_t preRotKeysFreed = 0;
			for (auto it = kskRotations_.begin(); it != kskRotations_.end();) {
				if (it->first < 0) {
					it = kskRotations_.erase(it);
					preRotKeysFreed++;
				} else {
					++it;
				}
			}
			if (preRotKeysFreed > 0) {
				std::cout << "[GPUHydiaHelper] Freed " << preRotKeysFreed << " pre-rotation keys (negative indices) — no longer needed" << std::endl;
			}
		}

		return errors == 0;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] Failed to stream diagonals: " << e.what() << std::endl;
		// Free partially uploaded GPU diagonals
		for (void* ptr : cachedGPUDiagonals_) {
			if (ptr)
				delete static_cast<FIDESlib::CKKS::Ciphertext*>(ptr);
		}
		cachedGPUDiagonals_.clear();
		cudaDeviceSynchronize();
		diagonalsCached_ = false;
		std::cerr << "[GPUHydiaHelper] Cleaned up partial GPU diagonal cache" << std::endl;
		return false;
	}
}

// =============================================================================
// Per-matrix workers for multi-stream dispatch
// =============================================================================

// BSGS per-matrix worker: on-demand BSGS + lazy-relin + Chebyshev for ONE matrix
void* GPUHydiaHelper::ProcessMatrixOnStream(size_t matrixIdx, size_t vectorDim, int n1, int numGiantSteps, double delta, size_t signDepth, cudaStream_t stream) {
	// Complete BSGS + Chebyshev pipeline for ONE matrix on the given CUDA stream.
	// The stream parameter enables concurrent execution across matrices when
	// the GPU scheduler can overlap independent kernel launches.

	FIDESlib::CKKS::Ciphertext* matrixAccum = nullptr;

	for (int j = 0; j < numGiantSteps; ++j) {
		int giantStep = j * n1;
		int iEnd	  = std::min(n1, static_cast<int>(vectorDim) - giantStep);
		if (iEnd <= 0)
			continue;

		std::vector<FIDESlib::CKKS::Ciphertext*> babies;
		std::vector<FIDESlib::CKKS::Ciphertext*> diagonals;
		babies.reserve(static_cast<size_t>(iEnd));
		diagonals.reserve(static_cast<size_t>(iEnd));

		for (int i = 0; i < iEnd; ++i) {
			int diagIndex = static_cast<int>(matrixIdx * vectorDim) + giantStep + i;
			if (diagIndex >= static_cast<int>(cachedGPUDiagonals_.size()))
				continue;

			babies.push_back(static_cast<FIDESlib::CKKS::Ciphertext*>(cachedGPUBabySteps_[i]));
			diagonals.push_back(static_cast<FIDESlib::CKKS::Ciphertext*>(cachedGPUDiagonals_[diagIndex]));
		}

		if (babies.empty()) {
			continue;
		}

		FIDESlib::CKKS::Ciphertext* giantStepAccum = new FIDESlib::CKKS::Ciphertext(gpuContext_);
		giantStepAccum->dotProduct(babies, diagonals, false);
		GPU_COUNT_MULT();

		giantStepAccum->rescale();
		GPU_COUNT_RESCALE();

		if (j > 0 && giantStep < static_cast<int>(vectorDim)) {
			giantStepAccum->rotate(giantStep);
			GPU_COUNT_ROTATE();
		}

		if (matrixAccum == nullptr) {
			matrixAccum = giantStepAccum;
		} else {
			matrixAccum->add(*matrixAccum, *giantStepAccum);
			GPU_COUNT_ADD();
			delete giantStepAccum;
		}
	}

	if (matrixAccum == nullptr) {
		return static_cast<void*>(new FIDESlib::CKKS::Ciphertext(gpuContext_));
	}

	// Apply Chebyshev comparison — stays on GPU
	void* chebResult = EvalChebyshevCompareOnGPU(static_cast<void*>(matrixAccum), delta, signDepth);
	delete matrixAccum;
	return chebResult;
}

// Pure-diagonal per-matrix worker: similarity + lazy-relin + Chebyshev for ONE matrix
void* GPUHydiaHelper::ProcessSimilarityMatrixOnStream(size_t matrixIdx, size_t vectorDim, double delta, size_t signDepth, cudaStream_t stream) {
	// Complete pure-diagonal pipeline for ONE matrix.
	// No giant-step decomposition — all vectorDim baby steps are used directly.
	// This is more efficient than BSGS when n1 = vectorDim (no giant-step rotation overhead).

	std::vector<FIDESlib::CKKS::Ciphertext*> babies;
	std::vector<FIDESlib::CKKS::Ciphertext*> diagonals;
	babies.reserve(vectorDim);
	diagonals.reserve(vectorDim);

	for (size_t i = 0; i < vectorDim; ++i) {
		int babyIdx = static_cast<int>(i);
		int diagIdx = static_cast<int>(matrixIdx * vectorDim + i);
		if (diagIdx >= static_cast<int>(cachedGPUDiagonals_.size()))
			continue;

		babies.push_back(static_cast<FIDESlib::CKKS::Ciphertext*>(cachedGPUBabySteps_[babyIdx]));
		diagonals.push_back(static_cast<FIDESlib::CKKS::Ciphertext*>(cachedGPUDiagonals_[diagIdx]));
	}

	if (babies.empty()) {
		return static_cast<void*>(new FIDESlib::CKKS::Ciphertext(gpuContext_));
	}

	FIDESlib::CKKS::Ciphertext* gpuAccumulator = new FIDESlib::CKKS::Ciphertext(gpuContext_);
	gpuAccumulator->dotProduct(babies, diagonals, false);
	GPU_COUNT_MULT();

	gpuAccumulator->rescale();
	GPU_COUNT_RESCALE();

	// Apply Chebyshev comparison — stays on GPU
	void* chebResult = EvalChebyshevCompareOnGPU(static_cast<void*>(gpuAccumulator), delta, signDepth);
	delete gpuAccumulator;
	return chebResult;
}

// =============================================================================
// Core pipeline: Lazy-relin BSGS + Chebyshev (Approach 81) — MULTI-STREAM
// =============================================================================

Ciphertext<DCRTPoly>
GPUHydiaHelper::EvalOnDemandBSGSLazyChebyshevAndAggregateOnGPU(size_t numMatrices, size_t vectorDim, int n1, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth, size_t numSlots) {

	if (!gpuContextReady_ || !diagonalsCached_ || !babyStepsCached_) {
		throw std::runtime_error("[GPUHydiaHelper] GPU/diagonals/baby steps not ready");
	}

	if (!diagonalsPreRotated_) {
		throw std::runtime_error("[GPUHydiaHelper] Diagonals not pre-rotated. Use CacheDiagonalsWithPreRotationOnGPU.");
	}

	if (preRotationN1_ != n1) {
		throw std::runtime_error(
		  "[GPUHydiaHelper] n1 mismatch: diagonals pre-rotated with n1=" + std::to_string(preRotationN1_) + " but query uses n1=" + std::to_string(n1));
	}

	GPU_SCOPED_TIMER("EvalOnDemandBSGSLazyChebyshevAndAggregateOnGPU_TOTAL");

	size_t N							= cc_->GetRingDimension();
	size_t L							= cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();
	[[maybe_unused]] size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);

	try {
		int numGiantSteps = (static_cast<int>(vectorDim) + n1 - 1) / n1;

		// =========================================================================
		// PHASE 0: Ensure giant step and EvalSum rotation keys are cached
		// =========================================================================
		GPU_PROFILE_START("A81_Phase0_KeyInit");
		int keysInit = 0;

		for (int j = 1; j < numGiantSteps; ++j) {
			int giantStep = j * n1;
			if (giantStep < static_cast<int>(vectorDim) && kskRotations_.find(giantStep) == kskRotations_.end()) {
				::FIDESlib::CKKS::RawKeySwitchKey rawKskRot = ::FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, giantStep, cc_);
				kskRotations_[giantStep]					= std::make_unique<::FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[giantStep]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(giantStep, std::move(*kskRotations_[giantStep]));
				keysInit++;
			}
		}

		for (size_t shift = 1; shift < numSlots; shift *= 2) {
			int idx = static_cast<int>(shift);
			if (kskRotations_.find(idx) == kskRotations_.end()) {
				::FIDESlib::CKKS::RawKeySwitchKey rawKskRot = ::FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, idx, cc_);
				kskRotations_[idx]							= std::make_unique<::FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[idx]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(idx, std::move(*kskRotations_[idx]));
				keysInit++;
			}
		}
		GPU_PROFILE_END("A81_Phase0_KeyInit");

		// =========================================================================
		// PHASE 1: BSGS + Chebyshev — MULTI-STREAM WAVE DISPATCH
		// =========================================================================
		GPU_PROFILE_START("A81_Phase1_BSGSChebyshev");
		auto phase1Start = std::chrono::steady_clock::now();

		std::vector<FIDESlib::CKKS::Ciphertext*> gpuChebResults(numMatrices, nullptr);

		// Lazily create exactly the CUDA streams needed for this workload
		EnsureMatrixStreams(numMatrices);

		size_t effectiveStreams = streamsInitialized_ ? std::min(numMatrixStreams_, numMatrices) : 1;

		if (effectiveStreams > 1) {
			size_t totalWaves = (numMatrices + effectiveStreams - 1) / effectiveStreams;
			std::cout << "[GPUHydiaHelper] 🚀 A81 Multi-stream: " << effectiveStreams << " concurrent streams for " << numMatrices << " matrices" << " ("
					  << totalWaves << " wave" << (totalWaves > 1 ? "s" : "") << ")" << std::endl;
			std::cout << "[CUDA Streams] batchSize=" << cc_->GetEncodingParams()->GetBatchSize() << ", numCKKSBatches=" << numMatrices
					  << ", streamsAvailable=" << numMatrixStreams_ << ", streamsUsed=" << effectiveStreams << " ("
					  << (effectiveStreams * 100 / numMatrixStreams_) << "% utilization)" << std::endl;

			size_t waveNum = 0;
			for (size_t waveStart = 0; waveStart < numMatrices; waveStart += effectiveStreams) {
				size_t waveEnd	= std::min(waveStart + effectiveStreams, numMatrices);
				size_t waveSize = waveEnd - waveStart;
				waveNum++;

				std::cout << "[CUDA Streams] A81-Membership Wave " << waveNum << "/" << totalWaves << ": matrices [" << waveStart << "-" << (waveEnd - 1) << "]" << " on "
						  << waveSize << "/" << numMatrixStreams_ << " streams" << (waveSize < numMatrixStreams_ ? " (UNDERUTILIZED)" : " (FULL)") << std::endl;

				// Launch all matrices in this wave concurrently
				for (size_t w = 0; w < waveSize; ++w) {
					size_t m			= waveStart + w;
					cudaStream_t stream = matrixStreams_[w % numMatrixStreams_];
					gpuChebResults[m] = static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessMatrixOnStream(m, vectorDim, n1, numGiantSteps, delta, signDepth, stream));
				}

				// Synchronize all streams before next wave
				for (size_t w = 0; w < waveSize; ++w) {
					cudaStreamSynchronize(matrixStreams_[w % numMatrixStreams_]);
				}
			}
		} else {
			// Fallback: single-stream sequential
			std::cout << "[GPUHydiaHelper] ⚠ A81 Single-stream mode (numMatrices=" << numMatrices << ") — no CUDA stream parallelism" << std::endl;
			for (size_t m = 0; m < numMatrices; ++m) {
				gpuChebResults[m] = static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessMatrixOnStream(m, vectorDim, n1, numGiantSteps, delta, signDepth, nullptr));
			}
			cudaDeviceSynchronize();
		}

		auto phase1End	  = std::chrono::steady_clock::now();
		double phase1Time = std::chrono::duration<double>(phase1End - phase1Start).count();

		GPU_PROFILE_END("A81_Phase1_BSGSChebyshev");

		std::cout << "[GPUHydiaHelper] A81 BSGS+Chebyshev: " << numMatrices << " matrices in " << std::fixed << std::setprecision(3) << phase1Time << "s"
				  << " (" << effectiveStreams << " streams, ~" << std::setprecision(1) << (phase1Time / numMatrices * 1000.0) << " ms/matrix)" << std::endl;

		// =========================================================================
		// PHASE 2: EvalAddMany on GPU
		// =========================================================================
		GPU_PROFILE_START("A81_Phase2_EvalAddMany");
		FIDESlib::CKKS::Ciphertext* gpuSum = new FIDESlib::CKKS::Ciphertext(gpuContext_);
		bool sumInitialized				   = false;
		for (size_t m = 0; m < numMatrices; ++m) {
			if (gpuChebResults[m]) {
				if (!sumInitialized) {
					gpuSum->copy(*gpuChebResults[m]);
					sumInitialized = true;
				} else {
					gpuSum->add(*gpuSum, *gpuChebResults[m]);
					GPU_COUNT_ADD();
				}
				delete gpuChebResults[m];
			}
		}
		GPU_PROFILE_END("A81_Phase2_EvalAddMany");

		GPU_PROFILE_START("A81_Phase3_EvalSum");
		FIDESlib::CKKS::Ciphertext* tempSum = new FIDESlib::CKKS::Ciphertext(gpuContext_);
		for (size_t shift = 1; shift < numSlots; shift *= 2) {
			int idx = static_cast<int>(shift);
			tempSum->copy(*gpuSum);
			tempSum->rotate(idx);
			GPU_COUNT_ROTATE();
			gpuSum->add(*gpuSum, *tempSum);
			GPU_COUNT_ADD();
		}
		delete tempSum;
		GPU_PROFILE_END("A81_Phase3_EvalSum");

		GPU_PROFILE_START("A81_Phase4_Download");
		cudaDeviceSynchronize();
		FIDESlib::CKKS::RawCipherText rawResult;
		gpuSum->store(rawResult);
		Ciphertext<DCRTPoly> result = templateCt->Clone();
		FIDESlib::CKKS::GetOpenFHECipherText(result, rawResult);
		delete gpuSum;
		GPU_TRACK_GPU_TO_CPU(ctSizeBytes, "CKKS::Ciphertext", "final A81 Lazy BSGS aggregated result");
		GPU_PROFILE_END("A81_Phase4_Download");

		return result;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalOnDemandBSGSLazyChebyshevAndAggregateOnGPU failed: " << e.what() << std::endl;
		throw;
	}
}

std::vector<Ciphertext<DCRTPoly>>
GPUHydiaHelper::EvalOnDemandBSGSLazyChebyshevBatchedOnGPU(size_t numMatrices, size_t vectorDim, int n1, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth) {

	if (!gpuContextReady_ || !diagonalsCached_ || !babyStepsCached_) {
		throw std::runtime_error("[GPUHydiaHelper] GPU/diagonals/baby steps not ready");
	}

	if (!diagonalsPreRotated_) {
		throw std::runtime_error("[GPUHydiaHelper] Diagonals not pre-rotated.");
	}

	GPU_SCOPED_TIMER("EvalOnDemandBSGSLazyChebyshevBatchedOnGPU_TOTAL");

	try {
		int numGiantSteps = (static_cast<int>(vectorDim) + n1 - 1) / n1;

		// =========================================================================
		// PHASE 0: Ensure giant step rotation keys are cached
		// =========================================================================
		GPU_PROFILE_START("A81_Batched_Phase0_KeyInit");
		for (int j = 1; j < numGiantSteps; ++j) {
			int giantStep = j * n1;
			if (giantStep < static_cast<int>(vectorDim) && kskRotations_.find(giantStep) == kskRotations_.end()) {
				::FIDESlib::CKKS::RawKeySwitchKey rawKskRot = ::FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, giantStep, cc_);
				kskRotations_[giantStep]					= std::make_unique<::FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				kskRotations_[giantStep]->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(giantStep, std::move(*kskRotations_[giantStep]));
			}
		}
		GPU_PROFILE_END("A81_Batched_Phase0_KeyInit");

		// =========================================================================
		// PHASE 1: BSGS + Chebyshev for ALL matrices — MULTI-STREAM WAVE DISPATCH
		// =========================================================================
		GPU_PROFILE_START("A81_Batched_Phase1_BSGSCheb");
		auto phase1Start = std::chrono::steady_clock::now();

		std::vector<FIDESlib::CKKS::Ciphertext*> gpuChebResults(numMatrices, nullptr);
		std::vector<Ciphertext<DCRTPoly>> results(numMatrices);
		size_t effectiveStreams = 1;

		// FIDESlib v2 + OpenFHE 1.4.2 is stable for A81 membership, but the
		// batched index path can still produce undecodable host ciphertexts
		// when multiple per-matrix post-Chebyshev results remain live on the
		// GPU. Use a conservative matrix-by-matrix execution/download path for
		// correctness.
		std::cout << "[GPUHydiaHelper] A81 v2 index path: serial matrix-by-matrix download for correctness" << std::endl;
		for (size_t m = 0; m < numMatrices; ++m) {
			gpuChebResults[m] = static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessMatrixOnStream(m, vectorDim, n1, numGiantSteps, delta, signDepth, nullptr));
			cudaDeviceSynchronize();

			if (gpuChebResults[m]) {
				// DebugDumpGpuCt(cc_, keys_, gpuContext_, *gpuChebResults[m],
				//                templateCt, "81-idx-m" + std::to_string(m) + "-POST-cheb");
				FIDESlib::CKKS::RawCipherText rawResult;
				gpuChebResults[m]->store(rawResult);
				results[m] = templateCt->Clone();
				FIDESlib::CKKS::GetOpenFHECipherText(results[m], rawResult);
				delete gpuChebResults[m];
				gpuChebResults[m] = nullptr;
			}
		}

		auto phase1End	  = std::chrono::steady_clock::now();
		double phase1Time = std::chrono::duration<double>(phase1End - phase1Start).count();

		GPU_PROFILE_END("A81_Batched_Phase1_BSGSCheb");

		std::cout << "[GPUHydiaHelper] A81 Index BSGS+Cheb (" << numMatrices << " matrices): " << std::fixed << std::setprecision(3) << phase1Time << "s"
				  << " (" << effectiveStreams << " streams, ~" << std::setprecision(1) << (phase1Time / numMatrices * 1000.0) << " ms/matrix)" << std::endl;

		// =========================================================================
		// PHASE 2: Bulk download all results
		// =========================================================================
		GPU_PROFILE_START("A81_Batched_Phase2_BulkDownload");

		cudaDeviceSynchronize();
		GPU_TRACK_GPU_TO_CPU(ctSizeBytes * numMatrices, "CKKS::Ciphertext[]", "A81 bulk download all matrices");

		GPU_PROFILE_END("A81_Batched_Phase2_BulkDownload");

		return results;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalOnDemandBSGSLazyChebyshevBatchedOnGPU failed: " << e.what() << std::endl;
		throw;
	}
}

// =============================================================================
// GPU Chebyshev evaluation (Paterson-Stockmeyer)
// =============================================================================

std::pair<uint32_t, uint32_t> GPUHydiaHelper::ComputeDegreesPS(uint32_t n) {
	if (n <= 1)
		return { 1, 1 };
	uint32_t bestK = n, bestM = 1, bestCost = n + 1;
	for (uint32_t m = 1; m <= 10; ++m) {
		uint32_t power = 1 << (m - 1);
		uint32_t k	   = (n + power - 1) / power;
		if (k >= 1 && k + m < bestCost) {
			bestK	 = k;
			bestM	 = m;
			bestCost = k + m;
		}
	}
	return { bestK, bestM };
}

std::vector<void*> GPUHydiaHelper::BuildChebyshevPowersBinaryTree(void* gpuY_void, uint32_t k) {
	GPU_SCOPED_TIMER("BuildChebyshevPowersBinaryTree");

	FIDESlib::CKKS::Ciphertext* gpuY = static_cast<FIDESlib::CKKS::Ciphertext*>(gpuY_void);
	std::vector<FIDESlib::CKKS::Ciphertext*> T(k, nullptr);
	if (k == 0)
		return std::vector<void*>();

	T[0] = new FIDESlib::CKKS::Ciphertext(gpuContext_);
	T[0]->copy(*gpuY);
	// Ensure NoiseLevel == 1 for FIXEDMANUAL scaling (required for square operation)
	if (T[0]->NoiseLevel == 2) {
		T[0]->rescale();
		GPU_COUNT_RESCALE();
	}
	if (k == 1) {
		std::vector<void*> r(1);
		r[0] = T[0];
		return r;
	}

	size_t treeSquares = 0, treeMults = 0, treeRescales = 0;
	for (uint32_t i = 2; i <= k; ++i) {
		T[i - 1] = new FIDESlib::CKKS::Ciphertext(gpuContext_);
		if ((i & (i - 1)) == 0) { // power of 2
			uint32_t half = i / 2;
			T[i - 1]->copy(*T[half - 1]);
			// Ensure NoiseLevel == 1 before square (FIXEDMANUAL requirement)
			if (T[i - 1]->NoiseLevel == 2) {
				T[i - 1]->rescale();
				GPU_COUNT_RESCALE();
				treeRescales++;
			}
			T[i - 1]->square(false);
			GPU_COUNT_SQUARE();
			treeSquares++;
			FIDESlib::CKKS::Ciphertext doubled(gpuContext_);
			doubled.add(*T[i - 1], *T[i - 1]);
			GPU_COUNT_ADD();
			T[i - 1]->copy(doubled);
			T[i - 1]->rescale();
			GPU_COUNT_RESCALE();
			treeRescales++;
			T[i - 1]->addScalar(-1.0);
			int targetLevel = T[i - 1]->getLevel();
			if (T[0]->getLevel() > targetLevel)
				T[0]->dropToLevel(targetLevel);
		} else if (i % 2 == 1) {
			uint32_t low = i / 2, high = (i + 1) / 2;
			int lowLevel = T[low - 1]->getLevel(), highLevel = T[high - 1]->getLevel();
			if (lowLevel > highLevel)
				T[low - 1]->dropToLevel(highLevel);
			else if (highLevel > lowLevel)
				T[high - 1]->dropToLevel(lowLevel);
			T[i - 1]->mult(*T[low - 1], *T[high - 1], false);
			GPU_COUNT_MULT();
			treeMults++;
			FIDESlib::CKKS::Ciphertext doubled(gpuContext_);
			doubled.add(*T[i - 1], *T[i - 1]);
			GPU_COUNT_ADD();
			T[i - 1]->copy(doubled);
			T[i - 1]->rescale();
			GPU_COUNT_RESCALE();
			treeRescales++;
			int resultLevel = T[i - 1]->getLevel();
			if (T[0]->getLevel() > resultLevel)
				T[0]->dropToLevel(resultLevel);
			T[i - 1]->sub(*T[0]);
			GPU_COUNT_ADD(); // sub is essentially an add
		}
	}
	GPU_DEBUG_PRINT("[ChebyTree] k=" << k << ", squares=" << treeSquares << ", mults=" << treeMults << ", rescales=" << treeRescales);

	std::vector<void*> result(k);
	for (uint32_t i = 0; i < k; ++i)
		result[i] = T[i];
	return result;
}

void* GPUHydiaHelper::EvalChebyshevSeriesPSOnGPU(void* gpuCtVoid, const std::vector<double>& coefficients) {
	GPU_SCOPED_TIMER("EvalChebyshevSeriesPSOnGPU");

	uint32_t n	= static_cast<uint32_t>(coefficients.size()) - 1;
	auto [k, m] = ComputeDegreesPS(n);

	GPU_DEBUG_PRINT("[Chebyshev PS] degree=" << n << ", k=" << k << ", m=" << m);

	GPU_PROFILE_START("Cheby_BuildPowerTree");
	std::vector<void*> T_void = BuildChebyshevPowersBinaryTree(gpuCtVoid, k);
	std::vector<FIDESlib::CKKS::Ciphertext*> T(k);
	for (uint32_t i = 0; i < k; ++i)
		T[i] = static_cast<FIDESlib::CKKS::Ciphertext*>(T_void[i]);
	GPU_PROFILE_END("Cheby_BuildPowerTree");

	GPU_PROFILE_START("Cheby_BuildT2Powers");
	size_t chebySquares = 0;
	std::vector<FIDESlib::CKKS::Ciphertext*> T2(m, nullptr);
	T2[0] = new FIDESlib::CKKS::Ciphertext(gpuContext_);
	T2[0]->copy(*T[k - 1]);
	for (uint32_t i = 1; i < m; ++i) {
		T2[i] = new FIDESlib::CKKS::Ciphertext(gpuContext_);
		T2[i]->copy(*T2[i - 1]);
		// Ensure NoiseLevel == 1 before square (FIXEDMANUAL requirement)
		if (T2[i]->NoiseLevel == 2) {
			T2[i]->rescale();
			GPU_COUNT_RESCALE();
		}
		T2[i]->square(true);
		GPU_COUNT_SQUARE();
		chebySquares++;
		// Use add instead of multScalar(2.0) to avoid NoiseLevel increase
		FIDESlib::CKKS::Ciphertext doubled(gpuContext_);
		doubled.add(*T2[i], *T2[i]); // doubled = T2[i] + T2[i] = 2*T2[i]
		GPU_COUNT_ADD();
		T2[i]->copy(doubled);
		T2[i]->addScalar(-1.0);
	}
	GPU_PROFILE_END("Cheby_BuildT2Powers");
	GPU_DEBUG_PRINT("[Chebyshev PS] T2 squares: " << chebySquares);

	GPU_PROFILE_START("Cheby_ComputeChunks");
	uint32_t numChunks = (n + k) / k;
	std::vector<FIDESlib::CKKS::Ciphertext*> Q(numChunks, nullptr);
	for (uint32_t j = 0; j < numChunks; ++j) {
		Q[j]			 = new FIDESlib::CKKS::Ciphertext(gpuContext_);
		uint32_t baseIdx = j * k;
		bool initialized = false;
		for (uint32_t i = 1; i < k && baseIdx + i <= n; ++i) {
			double coef = (baseIdx + i < coefficients.size()) ? coefficients[baseIdx + i] : 0.0;
			if (std::abs(coef) > 1e-15) {
				if (!initialized) {
					Q[j]->copy(*T[i - 1]);
					Q[j]->multScalar(coef, true);
					initialized = true;
				} else {
					FIDESlib::CKKS::Ciphertext term(gpuContext_);
					term.copy(*T[i - 1]);
					term.multScalar(coef, true);
					Q[j]->add(term);
					GPU_COUNT_ADD();
				}
			}
		}
		double c0 = (baseIdx < coefficients.size()) ? coefficients[baseIdx] : 0.0;
		if (!initialized) {
			Q[j]->copy(*T[0]);
			Q[j]->multScalar(0.0, true);
		}
		if (std::abs(c0) > 1e-15)
			Q[j]->addScalar(c0);
	}
	GPU_PROFILE_END("Cheby_ComputeChunks");
	GPU_DEBUG_PRINT("[Chebyshev PS] chunks computed: " << numChunks);

	GPU_PROFILE_START("Cheby_Combine");
	size_t combineMults				   = 0;
	FIDESlib::CKKS::Ciphertext* result = new FIDESlib::CKKS::Ciphertext(gpuContext_);
	result->copy(*Q[numChunks - 1]);
	for (int j = static_cast<int>(numChunks) - 2; j >= 0; --j) {
		int resLevel = result->getLevel(), tkLevel = T2[0]->getLevel();
		if (resLevel > tkLevel)
			result->dropToLevel(tkLevel);
		else if (tkLevel > resLevel)
			T2[0]->dropToLevel(resLevel);

		if (result->NoiseLevel == 2) {
			result->rescale();
			GPU_COUNT_RESCALE();
		}
		if (T2[0]->NoiseLevel == 2) {
			T2[0]->rescale();
			GPU_COUNT_RESCALE();
		}

		result->mult(*T2[0], true);
		GPU_COUNT_MULT();
		combineMults++;

		if (result->NoiseLevel == 2) {
			result->rescale();
			GPU_COUNT_RESCALE();
		}

		int newResLevel = result->getLevel(), qjLevel = Q[j]->getLevel();
		if (Q[j]->NoiseLevel == 2) {
			Q[j]->rescale();
			GPU_COUNT_RESCALE();
			qjLevel = Q[j]->getLevel();
		}
		if (qjLevel > newResLevel)
			Q[j]->dropToLevel(newResLevel);
		result->add(*Q[j]);
		GPU_COUNT_ADD();
	}
	GPU_PROFILE_END("Cheby_Combine");
	GPU_DEBUG_PRINT("[Chebyshev PS] combine mults: " << combineMults);

	for (auto* t : T)
		if (t)
			delete t;
	for (auto* t2 : T2)
		if (t2)
			delete t2;
	for (auto* q : Q)
		if (q)
			delete q;
	return static_cast<void*>(result);
}

// FULLY GPU Chebyshev comparison - no CPU roundtrip!
void* GPUHydiaHelper::EvalChebyshevCompareOnGPU(void* gpuCtVoid, double delta, size_t signDepth) {
	if (!gpuContextReady_) {
		std::cerr << "[GPUHydiaHelper] GPU not ready for EvalChebyshevCompareOnGPU" << std::endl;
		return nullptr;
	}

	if (signDepth < 7 || signDepth > 15) {
		std::cerr << "[GPUHydiaHelper] Error: signDepth must be between 7 and 15" << std::endl;
		return nullptr;
	}

	// Depth-to-degree mapping
	const std::vector<int> DEPTH_TO_DEGREE({ -1, -1, -1, 5, 13, 27, 59, 119, 247, 495, 1007, 2031 });

	size_t polyDegree = DEPTH_TO_DEGREE[signDepth - 4];

	// Compute Chebyshev coefficients for sign(x - delta) on [-1, 1]
	std::vector<double> chebyCoeffs = ComputeChebyshevCoefficients([&delta](double x) -> double { return (x >= delta) ? 1.0 : -1.0; }, -1.0, 1.0, polyDegree);
	chebyCoeffs[0] /= 2.0; // Standard Chebyshev convention

	// Step 1: Apply Chebyshev series on GPU
	void* signResult = EvalChebyshevSeriesPSOnGPU(gpuCtVoid, chebyCoeffs);

	// Step 2: Shift range from [-1,1] to [0,2] - add 1.0 on GPU
	FIDESlib::CKKS::Ciphertext* finalResult = static_cast<FIDESlib::CKKS::Ciphertext*>(signResult);
	finalResult->addScalar(1.0);

	return static_cast<void*>(finalResult);
}

// =============================================================================
// Approach 51: Pure-diagonal pipelines (dedicated, no BSGS decomposition)
// =============================================================================

// A51 Membership: Similarity + Chebyshev + EvalAddMany + EvalSum
Ciphertext<DCRTPoly>
GPUHydiaHelper::EvalSimilarityChebyshevAndAggregateOnGPU(size_t numMatrices, size_t vectorDim, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth, size_t numSlots) {

	if (!gpuContextReady_ || !diagonalsCached_ || !babyStepsCached_) {
		throw std::runtime_error("[GPUHydiaHelper] GPU/diagonals/baby steps not ready");
	}

	GPU_SCOPED_TIMER("EvalSimilarityChebyshevAndAggregateOnGPU_TOTAL");

	std::cout << "[GPUHydiaHelper] A51 Membership: Full GPU pipeline" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	size_t N						   = cc_->GetRingDimension();
	size_t L						   = cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();
	size_t ctSizeBytes				   = GPU_ESTIMATE_CT_SIZE(N, L);
	[[maybe_unused]] size_t rotKeySize = ctSizeBytes * 3;

	try {
		// =========================================================================
		// PHASE 0: Pre-initialize EvalSum rotation keys (powers of 2)
		// =========================================================================
		GPU_PROFILE_START("A51_Phase0_EvalSumKeyInit");
		int newKeysInit = 0;
		for (size_t shift = 1; shift < numSlots; shift *= 2) {
			int idx = static_cast<int>(shift);
			if (kskRotations_.find(idx) == kskRotations_.end()) {
				auto kskRot									= std::make_unique<FIDESlib::CKKS::KeySwitchingKey>(gpuContext_);
				::FIDESlib::CKKS::RawKeySwitchKey rawKskRot = ::FIDESlib::CKKS::GetRotationKeySwitchKey(keys_, idx, cc_);
				kskRot->Initialize(rawKskRot);
				gpuContext_->AddRotationKey(idx, std::move(*kskRot));
				kskRotations_[idx] = std::move(kskRot);

				GPU_TRACK_CPU_TO_GPU(rotKeySize, "CKKS::KeySwitchingKey", "EvalSum rotation key [" + std::to_string(idx) + "]");
				newKeysInit++;
			}
		}
		GPU_PROFILE_END("A51_Phase0_EvalSumKeyInit");
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalSimilarityChebyshevAndAggregateOnGPU PHASE 0failed: " << e.what() << std::endl;
		throw;
	}

	double simTime = 0.0;
	std::vector<FIDESlib::CKKS::Ciphertext*> gpuChebResults(numMatrices, nullptr);

	try {
		// =========================================================================
		// PHASE 1: Similarity + Chebyshev — MULTI-STREAM WAVE DISPATCH
		// =========================================================================
		GPU_PROFILE_START("A51_Phase1_SimChebyshev");
		auto simStart = std::chrono::steady_clock::now();

		// Lazily create exactly the CUDA streams needed for this workload
		EnsureMatrixStreams(numMatrices);

		size_t effectiveStreams = streamsInitialized_ ? std::min(numMatrixStreams_, numMatrices) : 1;

		if (effectiveStreams > 1) {
			size_t totalWaves = (numMatrices + effectiveStreams - 1) / effectiveStreams;
			std::cout << "[GPUHydiaHelper] 🚀 A51 Multi-stream: " << effectiveStreams << " concurrent streams for " << numMatrices << " matrices" << " ("
					  << totalWaves << " wave" << (totalWaves > 1 ? "s" : "") << ")" << std::endl;
			std::cout << "[CUDA Streams] batchSize=" << cc_->GetEncodingParams()->GetBatchSize() << ", numCKKSBatches=" << numMatrices
					  << ", streamsAvailable=" << numMatrixStreams_ << ", streamsUsed=" << effectiveStreams << " ("
					  << (effectiveStreams * 100 / numMatrixStreams_) << "% utilization)" << std::endl;

			size_t waveNum = 0;
			for (size_t waveStart = 0; waveStart < numMatrices; waveStart += effectiveStreams) {
				size_t waveEnd	= std::min(waveStart + effectiveStreams, numMatrices);
				size_t waveSize = waveEnd - waveStart;
				waveNum++;

				std::cout << "[CUDA Streams] A51-Membership Wave " << waveNum << "/" << totalWaves << ": matrices [" << waveStart << "-" << (waveEnd - 1) << "]" << " on "
						  << waveSize << "/" << numMatrixStreams_ << " streams" << (waveSize < numMatrixStreams_ ? " (UNDERUTILIZED)" : " (FULL)") << std::endl;

				for (size_t w = 0; w < waveSize; ++w) {
					size_t m			= waveStart + w;
					cudaStream_t stream = matrixStreams_[w % numMatrixStreams_];
					gpuChebResults[m]	= static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessSimilarityMatrixOnStream(m, vectorDim, delta, signDepth, stream));
				}

				for (size_t w = 0; w < waveSize; ++w) {
					cudaStreamSynchronize(matrixStreams_[w % numMatrixStreams_]);
				}
			}
		} else {
			std::cout << "[GPUHydiaHelper] ⚠ A51 Single-stream mode (numMatrices=" << numMatrices << ") — no CUDA stream parallelism" << std::endl;
			for (size_t m = 0; m < numMatrices; ++m) {
				gpuChebResults[m] = static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessSimilarityMatrixOnStream(m, vectorDim, delta, signDepth, nullptr));
			}
			cudaDeviceSynchronize();
		}

		auto simEnd = std::chrono::steady_clock::now();
		simTime		= std::chrono::duration<double>(simEnd - simStart).count();
		GPU_PROFILE_END("A51_Phase1_SimChebyshev");
		std::cout << "[GPUHydiaHelper] A51 Similarity+Chebyshev: " << numMatrices << " matrices in " << std::fixed << std::setprecision(3) << simTime << "s"
				  << " (" << effectiveStreams << " streams, ~" << std::setprecision(1) << (simTime / numMatrices * 1000.0) << " ms/matrix)" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalSimilarityChebyshevAndAggregateOnGPU PHASE 1 failed: " << e.what() << std::endl;
		throw;
	}

	FIDESlib::CKKS::Ciphertext* gpuSum = nullptr;
	double addTime					   = 0.0;

	try {

		// =========================================================================
		// PHASE 2: EvalAddMany on GPU
		// =========================================================================
		GPU_PROFILE_START("A51_Phase2_EvalAddMany");
		auto addStart		= std::chrono::steady_clock::now();
		gpuSum				= new FIDESlib::CKKS::Ciphertext(gpuContext_);
		bool sumInitialized = false;

		for (size_t m = 0; m < numMatrices; ++m) {
			if (gpuChebResults[m]) {
				if (!sumInitialized) {
					gpuSum->copy(*gpuChebResults[m]);
					sumInitialized = true;
				} else {
					gpuSum->add(*gpuSum, *gpuChebResults[m]);
					GPU_COUNT_ADD();
				}
				delete gpuChebResults[m];
			}
		}

		auto addEnd = std::chrono::steady_clock::now();
		addTime		= std::chrono::duration<double>(addEnd - addStart).count();
		GPU_PROFILE_END("A51_Phase2_EvalAddMany");
		std::cout << "[GPUHydiaHelper] A51 EvalAddMany: " << std::fixed << std::setprecision(3) << addTime << "s" << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalSimilarityChebyshevAndAggregateOnGPU PHASE 2 failed: " << e.what() << std::endl;
		throw;
	}

	try {

		// =========================================================================
		// PHASE 3: EvalSum on GPU
		// =========================================================================
		GPU_PROFILE_START("A51_Phase3_EvalSum");
		auto sumStart						= std::chrono::steady_clock::now();
		FIDESlib::CKKS::Ciphertext* tempSum = new FIDESlib::CKKS::Ciphertext(gpuContext_);

		for (size_t shift = 1; shift < numSlots; shift *= 2) {
			int idx = static_cast<int>(shift);
			tempSum->copy(*gpuSum);
			tempSum->rotate(idx);
			GPU_COUNT_ROTATE();
			gpuSum->add(*gpuSum, *tempSum);
			GPU_COUNT_ADD();
		}
		delete tempSum;

		auto sumEnd	   = std::chrono::steady_clock::now();
		double sumTime = std::chrono::duration<double>(sumEnd - sumStart).count();
		GPU_PROFILE_END("A51_Phase3_EvalSum");
		std::cout << "[GPUHydiaHelper] A51 EvalSum (" << static_cast<int>(std::log2(numSlots)) << " rotations): " << std::fixed << std::setprecision(3)
				  << sumTime << "s" << std::endl;

		// =========================================================================
		// PHASE 4: Download single result
		// =========================================================================
		GPU_PROFILE_START("A51_Phase4_Download");
		cudaDeviceSynchronize();

		FIDESlib::CKKS::RawCipherText rawResult;
		gpuSum->store(rawResult);
		Ciphertext<DCRTPoly> result = templateCt->Clone();
		FIDESlib::CKKS::GetOpenFHECipherText(result, rawResult);
		delete gpuSum;

		GPU_TRACK_GPU_TO_CPU(ctSizeBytes, "CKKS::Ciphertext", "final A51 aggregated result");
		GPU_PROFILE_END("A51_Phase4_Download");

		auto endTime	 = std::chrono::steady_clock::now();
		double totalTime = std::chrono::duration<double>(endTime - startTime).count();
		std::cout << "[GPUHydiaHelper] A51 Membership total: " << std::fixed << std::setprecision(2) << totalTime << "s (Sim+Cheb=" << simTime
				  << "s, Add=" << addTime << "s, Sum=" << sumTime << "s)" << std::endl;

		GPU_PROFILER_PRINT_SUMMARY();

		return result;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalSimilarityChebyshevAndAggregateOnGPU PHASE 4 failed: " << e.what() << std::endl;
		throw;
	}
}

// A51 Index: Similarity + Chebyshev + bulk download
std::vector<Ciphertext<DCRTPoly>>
GPUHydiaHelper::EvalSimilarityAndChebyshevBatchedOnGPU(size_t numMatrices, size_t vectorDim, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth) {

	if (!gpuContextReady_ || !diagonalsCached_ || !babyStepsCached_) {
		throw std::runtime_error("[GPUHydiaHelper] GPU/diagonals/baby steps not ready");
	}

	GPU_SCOPED_TIMER("EvalSimilarityAndChebyshevBatchedOnGPU_TOTAL");

	std::cout << "[GPUHydiaHelper] A51 Batched Index: Multi-stream similarity+Chebyshev" << std::endl;
	std::cout << "  vectorDim=" << vectorDim << " numMatrices=" << numMatrices << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	size_t N							= cc_->GetRingDimension();
	size_t L							= cc_->GetCryptoParameters()->GetElementParams()->GetParams().size();
	[[maybe_unused]] size_t ctSizeBytes = GPU_ESTIMATE_CT_SIZE(N, L);

	try {
		// =========================================================================
		// PHASE 1: Similarity + Chebyshev — MULTI-STREAM WAVE DISPATCH
		// =========================================================================
		GPU_PROFILE_START("A51_Batched_Phase1_SimCheb");
		auto phase1Start = std::chrono::steady_clock::now();

		std::vector<FIDESlib::CKKS::Ciphertext*> gpuChebResults(numMatrices, nullptr);

		// Lazily create exactly the CUDA streams needed for this workload
		EnsureMatrixStreams(numMatrices);

		size_t effectiveStreams = streamsInitialized_ ? std::min(numMatrixStreams_, numMatrices) : 1;

		if (effectiveStreams > 1) {
			size_t totalWaves = (numMatrices + effectiveStreams - 1) / effectiveStreams;
			std::cout << "[GPUHydiaHelper] 🚀 A51 Index Multi-stream: " << effectiveStreams << " concurrent streams for " << numMatrices << " matrices" << " ("
					  << totalWaves << " wave" << (totalWaves > 1 ? "s" : "") << ")" << std::endl;
			std::cout << "[CUDA Streams] batchSize=" << cc_->GetEncodingParams()->GetBatchSize() << ", numCKKSBatches=" << numMatrices
					  << ", streamsAvailable=" << numMatrixStreams_ << ", streamsUsed=" << effectiveStreams << " ("
					  << (effectiveStreams * 100 / numMatrixStreams_) << "% utilization)" << std::endl;

			size_t waveNum = 0;
			for (size_t waveStart = 0; waveStart < numMatrices; waveStart += effectiveStreams) {
				size_t waveEnd	= std::min(waveStart + effectiveStreams, numMatrices);
				size_t waveSize = waveEnd - waveStart;
				waveNum++;

				std::cout << "[CUDA Streams] A51-Index Wave " << waveNum << "/" << totalWaves << ": matrices [" << waveStart << "-" << (waveEnd - 1) << "]" << " on "
						  << waveSize << "/" << numMatrixStreams_ << " streams" << (waveSize < numMatrixStreams_ ? " (UNDERUTILIZED)" : " (FULL)") << std::endl;

				for (size_t w = 0; w < waveSize; ++w) {
					size_t m			= waveStart + w;
					cudaStream_t stream = matrixStreams_[w % numMatrixStreams_];
					gpuChebResults[m]	= static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessSimilarityMatrixOnStream(m, vectorDim, delta, signDepth, stream));
				}

				for (size_t w = 0; w < waveSize; ++w) {
					cudaStreamSynchronize(matrixStreams_[w % numMatrixStreams_]);
				}
			}
		} else {
			std::cout << "[GPUHydiaHelper] ⚠ A51 Index Single-stream mode (numMatrices=" << numMatrices << ") — no CUDA stream parallelism" << std::endl;
			for (size_t m = 0; m < numMatrices; ++m) {
				gpuChebResults[m] = static_cast<FIDESlib::CKKS::Ciphertext*>(ProcessSimilarityMatrixOnStream(m, vectorDim, delta, signDepth, nullptr));
			}
			cudaDeviceSynchronize();
		}

		auto phase1End	  = std::chrono::steady_clock::now();
		double phase1Time = std::chrono::duration<double>(phase1End - phase1Start).count();
		GPU_PROFILE_END("A51_Batched_Phase1_SimCheb");

		std::cout << "[GPUHydiaHelper] A51 Batched SimCheb: " << numMatrices << " matrices in " << std::fixed << std::setprecision(3) << phase1Time << "s"
				  << " (" << effectiveStreams << " streams, ~" << std::setprecision(1) << (phase1Time / numMatrices * 1000.0) << " ms/matrix)" << std::endl;

		// =========================================================================
		// PHASE 2: Bulk download all results
		// =========================================================================
		GPU_PROFILE_START("A51_Batched_Phase2_BulkDownload");
		auto phase2Start = std::chrono::steady_clock::now();

		cudaDeviceSynchronize();

		std::vector<Ciphertext<DCRTPoly>> results(numMatrices);
		for (size_t m = 0; m < numMatrices; ++m) {
			if (gpuChebResults[m]) {
				FIDESlib::CKKS::RawCipherText rawResult;
				gpuChebResults[m]->store(rawResult);
				results[m] = templateCt->Clone();
				FIDESlib::CKKS::GetOpenFHECipherText(results[m], rawResult);
				delete gpuChebResults[m];
			}
		}

		GPU_TRACK_GPU_TO_CPU(ctSizeBytes * numMatrices, "CKKS::Ciphertext[]", "A51 bulk download all matrices");

		auto phase2End	  = std::chrono::steady_clock::now();
		double phase2Time = std::chrono::duration<double>(phase2End - phase2Start).count();
		GPU_PROFILE_END("A51_Batched_Phase2_BulkDownload");

		std::cout << "[GPUHydiaHelper] A51 Batched Download (" << numMatrices << " cts): " << std::fixed << std::setprecision(3) << phase2Time << "s" << std::endl;

		auto endTime	 = std::chrono::steady_clock::now();
		double totalTime = std::chrono::duration<double>(endTime - startTime).count();
		std::cout << "[GPUHydiaHelper] A51 Batched Index total: " << std::fixed << std::setprecision(3) << totalTime << "s" << std::endl;

		GPU_PROFILER_PRINT_SUMMARY();

		return results;
	} catch (const std::exception& e) {
		std::cerr << "[GPUHydiaHelper] EvalSimilarityAndChebyshevBatchedOnGPU failed: " << e.what() << std::endl;
		throw;
	}
}

#else
// =============================================================================
// Stub implementation when GPU is disabled
// =============================================================================

GPUHydiaHelper::GPUHydiaHelper(CryptoContext<DCRTPoly> cc, const KeyPair<DCRTPoly>& keys, const std::vector<int>& gpuDevices, int batchSize, double gpuSafeMemoryLimitGB)
: cc_(cc), keys_(keys), gpuContextReady_(false), batchSize_(batchSize) {

	std::cerr << "[GPUHydiaHelper] WARNING: GPU support not compiled in this build." << std::endl;
	std::cerr << "[GPUHydiaHelper] Rebuild with -DUSE_GPU=ON to enable GPU acceleration." << std::endl;
	gpuContextReady_ = false;
}

GPUHydiaHelper::~GPUHydiaHelper() {
	// Nothing to cleanup in stub
}

bool GPUHydiaHelper::areDiagonalsCached() const {
	return false;
}

void GPUHydiaHelper::ComputeAndCacheBabyStepsOnGPU(const Ciphertext<DCRTPoly>& queryVector, int n1) {
	std::cerr << "[GPUHydiaHelper] GPU baby step computation not available in CPU-only build" << std::endl;
	throw std::runtime_error("GPU not available");
}

bool GPUHydiaHelper::areBabyStepsCached() const {
	return false;
}

void GPUHydiaHelper::CacheBabyStepsOnGPU(const std::vector<Ciphertext<DCRTPoly>>& babySteps) {
	std::cerr << "[GPUHydiaHelper] Baby step caching not available in CPU-only build" << std::endl;
}

void GPUHydiaHelper::CacheDiagonalsWithPreRotationOnGPU(const std::vector<Ciphertext<DCRTPoly>>& diagonals, size_t numMatrices, size_t vectorDim, int n1) {
	std::cerr << "[GPUHydiaHelper] Diagonal caching not available in CPU-only build" << std::endl;
}

bool GPUHydiaHelper::StreamDiagonalsToGPU(const std::string& serialDir, size_t numDiagonals, size_t numMatrices, int n1) {
	std::cerr << "[GPUHydiaHelper] StreamDiagonalsToGPU not available in CPU-only build" << std::endl;
	return false;
}

void GPUHydiaHelper::InitializeSimpleBSGSRotationKeysOnGPU(int n1, int vectorDim, size_t numSlots) {
	std::cerr << "[GPUHydiaHelper] Rotation key init not available in CPU-only build" << std::endl;
	throw std::runtime_error("GPU not available");
}

Ciphertext<DCRTPoly>
GPUHydiaHelper::EvalOnDemandBSGSLazyChebyshevAndAggregateOnGPU(size_t numMatrices, size_t vectorDim, int n1, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth, size_t numSlots) {
	throw std::runtime_error("[GPUHydiaHelper] GPU not available in CPU-only build");
}

std::vector<Ciphertext<DCRTPoly>>
GPUHydiaHelper::EvalOnDemandBSGSLazyChebyshevBatchedOnGPU(size_t numMatrices, size_t vectorDim, int n1, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth) {
	throw std::runtime_error("[GPUHydiaHelper] GPU not available in CPU-only build");
}

Ciphertext<DCRTPoly>
GPUHydiaHelper::EvalSimilarityChebyshevAndAggregateOnGPU(size_t numMatrices, size_t vectorDim, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth, size_t numSlots) {
	throw std::runtime_error("[GPUHydiaHelper] GPU not available in CPU-only build");
}

std::vector<Ciphertext<DCRTPoly>>
GPUHydiaHelper::EvalSimilarityAndChebyshevBatchedOnGPU(size_t numMatrices, size_t vectorDim, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth) {
	throw std::runtime_error("[GPUHydiaHelper] GPU not available in CPU-only build");
}

void* GPUHydiaHelper::EvalChebyshevCompareOnGPU(void* gpuCtVoid, double delta, size_t signDepth) {
	std::cerr << "[GPUHydiaHelper] EvalChebyshevCompareOnGPU not available in CPU-only build" << std::endl;
	return nullptr;
}

void* GPUHydiaHelper::EvalChebyshevSeriesPSOnGPU(void* gpuCt, const std::vector<double>& coefficients) {
	std::cerr << "[GPUHydiaHelper] EvalChebyshevSeriesPSOnGPU not available in CPU-only build" << std::endl;
	return nullptr;
}

std::pair<uint32_t, uint32_t> GPUHydiaHelper::ComputeDegreesPS(uint32_t n) {
	if (n <= 1)
		return { 1, 1 };
	uint32_t bestK = n, bestM = 1, bestCost = n + 1;
	for (uint32_t m = 1; m <= 10; ++m) {
		uint32_t power = 1 << (m - 1);
		uint32_t k	   = (n + power - 1) / power;
		if (k >= 1 && k + m < bestCost) {
			bestK	 = k;
			bestM	 = m;
			bestCost = k + m;
		}
	}
	return { bestK, bestM };
}

#endif // USE_GPU_ACCELERATION
