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

#ifndef GPU_HYDIA_HELPER_CUH
#define GPU_HYDIA_HELPER_CUH

#include <map>
#include <memory>
#include <openfhe.h>
#include <string>
#include <vector>
#ifdef USE_GPU_ACCELERATION
#include <CKKS/forwardDefs.cuh>
#include <cuda_runtime.h> // cudaStream_t for multi-stream parallelism
#else
typedef void* cudaStream_t; // Stub for non-GPU builds
#endif

using namespace lbcrypto;

// Unified GPU memory safety fraction (single place to change)
constexpr double GPU_MEMORY_SAFE_FRACTION = 0.95;

/**
 * GPU Hydia Helper — Approaches 51 (Pure Diagonal) and 81 (BSGS)
 *
 * Provides GPU-accelerated CKKS operations with:
 *   - Wave-based multi-stream parallelism across CKKS batches
 *   - Lazy relinearization for minimal multiplicative depth
 *   - Dedicated pure-diagonal pipeline (Approach 51)
 *   - On-demand BSGS pipeline with pre-rotated diagonals (Approach 81)
 *   - Memory-efficient streaming diagonal loader
 *   - Unified GPU safe memory limit
 *
 * Pipeline (Approach 81):
 *   1. Pre-rotate and cache encrypted diagonals on GPU (one-time setup)
 *   2. Pre-initialize BSGS rotation keys on GPU (one-time setup)
 *   3. Upload query and compute baby step rotations on GPU
 *   4. Lazy-relin BSGS: multNoRelin + accumulate per giant step, then single relinearize
 *   5. Chebyshev comparison (Paterson-Stockmeyer, fully on GPU)
 *   6. EvalAddMany + EvalSum (membership) or bulk download (index)
 *
 * Pipeline (Approach 51):
 *   Same as 81 but with vectorDim baby steps, no giant-step decomposition,
 *   and a dedicated pure-diagonal code path for efficiency.
 */
class GPUHydiaHelper {
  public:
	GPUHydiaHelper(CryptoContext<DCRTPoly> cc, const KeyPair<DCRTPoly>& keys, const std::vector<int>& gpuDevices = { 0 }, int batchSize = 8, double gpuSafeMemoryLimitGB = -1.0);

	~GPUHydiaHelper();

	// =========================================================================
	// Diagonal caching
	// =========================================================================

	/**
	 * Cache diagonals with pre-rotation for on-demand BSGS.
	 * For each diagonal k = j*n1 + i, pre-rotates by -j*n1.
	 * When n1 = vectorDim (Approach 51), pre-rotation is a no-op.
	 *
	 * @param diagonals Flat vector (numMatrices * vectorDim)
	 * @param numMatrices Number of database matrices
	 * @param vectorDim Dimension per matrix (e.g. 512)
	 * @param n1 Baby step size for BSGS (e.g. 23 for A81, 512 for A51)
	 */
	void CacheDiagonalsWithPreRotationOnGPU(const std::vector<Ciphertext<DCRTPoly>>& diagonals, size_t numMatrices, size_t vectorDim, int n1);

	/**
	 * Stream diagonals from disk directly to GPU one-by-one.
	 * Loads ONE diagonal at a time → uploads to GPU → releases CPU copy.
	 * Reduces peak CPU RAM by ~15 GB for large databases.
	 *
	 * @param serialDir Directory containing serialized diagonal files
	 * @param numDiagonals Number of diagonals per matrix (e.g. 512)
	 * @param numMatrices Number of database matrices
	 * @param n1 Baby step size for BSGS pre-rotation
	 * @return true if all diagonals loaded successfully
	 */
	bool StreamDiagonalsToGPU(const std::string& serialDir, size_t numDiagonals, size_t numMatrices, int n1, size_t fileIndexOffset = 0);

	/**
	 * Upload pre-rotated diagonals to GPU WITHOUT any homomorphic rotation.
	 * Use when the enroller has already applied plaintext pre-rotation (Approach 812).
	 * Sets diagonalsPreRotated_ = true and preRotationN1_ = n1 so the online
	 * BSGS pipeline treats them identically to server-pre-rotated diagonals.
	 *
	 * @param diagonals Flat vector (numMatrices * vectorDim) of already-pre-rotated ciphertexts
	 * @param numMatrices Number of database matrices
	 * @param vectorDim Dimension per matrix (e.g. 512)
	 * @param n1 Baby step size that was used for plaintext pre-rotation
	 */
	void CachePreRotatedDiagonalsOnGPU(const std::vector<Ciphertext<DCRTPoly>>& diagonals, size_t numMatrices, size_t vectorDim, int n1);

	/**
	 * Stream pre-rotated diagonals from disk directly to GPU (no homomorphic rotation).
	 * Like StreamDiagonalsToGPU but skips the rotate() call — the enroller already
	 * applied the plaintext rotation before encryption (Approach 812).
	 */
	bool StreamPreRotatedDiagonalsToGPU(const std::string& serialDir, size_t numDiagonals, size_t numMatrices, int n1, size_t fileIndexOffset = 0);

	bool areDiagonalsCached() const;

	bool areDiagonalsPreRotated() const {
		return diagonalsPreRotated_;
	}

	int getPreRotationN1() const {
		return preRotationN1_;
	}

	// =========================================================================
	// Baby step computation and caching
	// =========================================================================

	/**
	 * Upload query to GPU and compute n1 baby step rotations directly on GPU.
	 * Results are stored in cachedGPUBabySteps_.
	 */
	void ComputeAndCacheBabyStepsOnGPU(const Ciphertext<DCRTPoly>& queryVector, int n1);

	/**
	 * Upload pre-computed baby steps from CPU to GPU cache.
	 */
	void CacheBabyStepsOnGPU(const std::vector<Ciphertext<DCRTPoly>>& babySteps);

	bool areBabyStepsCached() const;

	// =========================================================================
	// Rotation key pre-initialization
	// =========================================================================

	/**
	 * Pre-initialize Simple BSGS rotation keys on GPU (baby, giant, EvalSum).
	 * Call once during setup to avoid key upload during queries.
	 */
	void InitializeSimpleBSGSRotationKeysOnGPU(int n1, int vectorDim, size_t numSlots);

	// =========================================================================
	// Approach 81: On-demand BSGS pipelines (with multi-stream parallelism)
	// =========================================================================

	/**
	 * Membership scenario: on-demand BSGS + lazy-relin + Chebyshev +
	 * EvalAddMany + EvalSum. Multi-stream wave dispatch across matrices.
	 * Downloads only the single aggregated result.
	 */
	Ciphertext<DCRTPoly>
	EvalOnDemandBSGSLazyChebyshevAndAggregateOnGPU(size_t numMatrices, size_t vectorDim, int n1, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth, size_t numSlots);

	/**
	 * Index scenario: on-demand BSGS + lazy-relin + Chebyshev for ALL matrices
	 * with multi-stream wave dispatch, then bulk-download all results.
	 */
	std::vector<Ciphertext<DCRTPoly>>
	EvalOnDemandBSGSLazyChebyshevBatchedOnGPU(size_t numMatrices, size_t vectorDim, int n1, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth);

	// =========================================================================
	// Approach 51: Pure-diagonal pipelines (with multi-stream parallelism)
	// =========================================================================

	/**
	 * Membership scenario: pure diagonal similarity + lazy-relin + Chebyshev +
	 * EvalAddMany + EvalSum. Uses all vectorDim baby steps directly (no BSGS).
	 * Multi-stream wave dispatch across matrices.
	 */
	Ciphertext<DCRTPoly>
	EvalSimilarityChebyshevAndAggregateOnGPU(size_t numMatrices, size_t vectorDim, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth, size_t numSlots);

	/**
	 * Index scenario: pure diagonal similarity + lazy-relin + Chebyshev for
	 * ALL matrices with multi-stream wave dispatch, then bulk-download.
	 */
	std::vector<Ciphertext<DCRTPoly>>
	EvalSimilarityAndChebyshevBatchedOnGPU(size_t numMatrices, size_t vectorDim, const Ciphertext<DCRTPoly>& templateCt, double delta, size_t signDepth);

	// =========================================================================
	// State queries
	// =========================================================================

	bool isReady() const {
		return gpuContextReady_;
	}

	bool isGPUMemoryInsufficient() const {
		return gpuMemoryInsufficient_;
	}

	double getGPUSafeMemoryLimitGB() const {
		return gpuSafeMemoryLimitGB_;
	}

	// =========================================================================
	// GPU Chebyshev internals (public for flexibility, use void* for C++ compat)
	// =========================================================================

	std::pair<uint32_t, uint32_t> ComputeDegreesPS(uint32_t n);

	void* EvalChebyshevSeriesPSOnGPU(void* gpuCt, const std::vector<double>& coefficients);

	void* EvalChebyshevCompareOnGPU(void* gpuCt, double delta, size_t signDepth);

  private:
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
	bool gpuContextReady_;
	int batchSize_;
	bool gpuMemoryInsufficient_ = false;

	// Unified GPU safe memory limit (GB) — single source of truth
	double gpuSafeMemoryLimitGB_ = -1.0;

	// =========================================================================
	// MULTI-GPU DATABASE SHARDING
	// =========================================================================
	// When constructed with more than one GPU device, this helper becomes a
	// thin COORDINATOR that owns one independent single-GPU child helper per
	// device. The database matrices are split into contiguous shards (one per
	// child); setup fans out to the children and queries run them in parallel
	// (one host thread per shard) and merge the results on the CPU.
	//
	// Concurrency is safe because FIDESlib's currentContext is thread_local
	// (each worker thread keeps its own current context) and each shard context
	// maps to a distinct GPU.
	bool isCoordinator_ = false;
	std::vector<int> shardDevices_;
	std::vector<std::unique_ptr<GPUHydiaHelper>> shardHelpers_;

	// Balanced contiguous matrix ranges [start, end) for numShards shards.
	std::vector<std::pair<size_t, size_t>> ComputeShardRanges(size_t numMatrices, size_t numShards) const;

#ifdef USE_GPU_ACCELERATION
	// GPU-specific members
	::FIDESlib::CKKS::Context gpuContext_;
	std::unique_ptr<::FIDESlib::CKKS::KeySwitchingKey> kskEval_;
	std::map<int, std::unique_ptr<::FIDESlib::CKKS::KeySwitchingKey>> kskRotations_;

	// Cached diagonals on GPU (void* = FIDESlib::CKKS::Ciphertext*)
	std::vector<void*> cachedGPUDiagonals_;
	bool diagonalsCached_;

	// Per-group diagonal organization
	size_t numCachedDiagonals_;
	size_t numCachedGroups_;
	bool perGroupCaching_;

	// Pre-rotated diagonals for on-demand BSGS
	bool diagonalsPreRotated_;
	int preRotationN1_;

	// Cached baby steps on GPU
	std::vector<void*> cachedGPUBabySteps_;
	bool babyStepsCached_;

	// =========================================================================
	// CUDA STREAM PARALLELISM: Process multiple matrices concurrently
	// =========================================================================
	// Dynamic stream pool — sized at runtime based on GPU capabilities.
	// RTX-class GPUs typically sustain 4-8 concurrent streams effectively;
	// H100/H200 with higher SM counts can leverage 16-32+.
	std::vector<cudaStream_t> matrixStreams_; // Pool of CUDA streams
	size_t numMatrixStreams_ = 0;			  // Active stream count
	bool streamsInitialized_ = false;

	void EnsureMatrixStreams(size_t numNeeded);
	void DestroyMatrixStreams();

	// BSGS per-matrix worker: on-demand BSGS + Chebyshev for ONE matrix on a stream
	void* ProcessMatrixOnStream(size_t matrixIdx, size_t vectorDim, int n1, int numGiantSteps, double delta, size_t signDepth, cudaStream_t stream);

	// A51 per-matrix worker: pure diagonal similarity + Chebyshev for ONE matrix on a stream
	void* ProcessSimilarityMatrixOnStream(size_t matrixIdx, size_t vectorDim, double delta, size_t signDepth, cudaStream_t stream);
	// =========================================================================

	// Depth-optimal Chebyshev helper (internal, uses void* for C++ compatibility)
	std::vector<void*> BuildChebyshevPowersBinaryTree(void* gpuY, uint32_t k);
#endif
};

#endif // GPU_HYDIA_HELPER_CUH
