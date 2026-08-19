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
// ** sender_diag_bsgs_simple_gpu.cpp: GPU-accelerated Simple BSGS (Approach 81)
// Lazy-relin BSGS on GPU with pre-rotated diagonals.
//
// Pipeline:
// 1. Pre-rotate and cache diagonals on GPU (one-time setup)
// 2. Upload query baby-step rotations to GPU
// 3. Lazy-relin BSGS + Chebyshev comparison (fully on GPU)
// 4. Download aggregated membership result or per-matrix index results

#include "../../include/sender_diag_bsgs_simple_gpu.h"
#include "../../include/gpu_fasthe_search_helper.cuh"
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace std;

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSSimpleSenderGPU::DiagonalBSGSSimpleSenderGPU(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam, GPUFastHESearchHelper* gpuHelper)
: DiagonalBSGSSimpleSender(ccParam, pkParam, vectorParam), gpuHelper_(gpuHelper), gpuCacheInitialized_(false), rotationsCached_(false) {
	if (gpuHelper_ && gpuHelper_->isReady()) {
		cout << "[Approach 81] GPU Simple BSGS Sender initialized:" << endl;
		cout << "  VECTOR_DIM = " << VECTOR_DIM << endl;
		cout << "  babyStepSize = " << babyStepSize << endl;
		cout << "  giantStepSize = " << giantStepSize << endl;
		cout << "  GPU helper ready" << endl;
	} else {
		cerr << "[Approach 81] WARNING: GPU helper not ready, will fall back to CPU" << endl;
	}
}

DiagonalBSGSSimpleSenderGPU::~DiagonalBSGSSimpleSenderGPU() {
	// gpuHelper_ is owned externally, don't delete it
}

// -------------------- GPU INITIALIZATION --------------------

bool DiagonalBSGSSimpleSenderGPU::isGPUReady() const {
	return gpuHelper_ != nullptr && gpuHelper_->isReady();
}

void DiagonalBSGSSimpleSenderGPU::initializeGPUCache() {
	if (!isGPUReady()) {
		cerr << "[Approach 81] Cannot initialize GPU cache - GPU not ready" << endl;
		return;
	}

	if (gpuCacheInitialized_) {
		cout << "[Approach 81] GPU cache already initialized" << endl;
		return;
	}

	cacheDiagonalsOnGPU();
	gpuCacheInitialized_ = true;
}

void DiagonalBSGSSimpleSenderGPU::initializeGPUCache(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals) {
	if (!isGPUReady()) {
		cerr << "[Approach 81] Cannot initialize GPU cache - GPU not ready" << endl;
		return;
	}

	if (gpuCacheInitialized_) {
		cout << "[Approach 81] GPU cache already initialized" << endl;
		return;
	}

	cout << "[Approach 81] Using preloaded diagonals (skipping file I/O)" << endl;
	cacheDiagonalsOnGPU(preloadedDiagonals);
	gpuCacheInitialized_ = true;
}

void DiagonalBSGSSimpleSenderGPU::cacheDiagonalsOnGPU() {
	if (!isGPUReady())
		return;

	cout << "[Approach 81] Loading and caching diagonals with pre-rotation on GPU..." << endl;
	auto startTime = chrono::steady_clock::now();

	size_t batchSize   = cc->GetEncodingParams()->GetBatchSize();
	size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

	// Load all diagonals from serialized files (same format as Approach 5/9)
	vector<Ciphertext<DCRTPoly>> allDiagonals;
	allDiagonals.reserve(numMatrices * VECTOR_DIM);

	for (size_t m = 0; m < numMatrices; ++m) {
		for (size_t i = 0; i < VECTOR_DIM; ++i) {
			string filepath = "serial/db_diagonal/index" + to_string(m * VECTOR_DIM + i) + ".bin";
			Ciphertext<DCRTPoly> diagonalCipher;
			if (Serial::DeserializeFromFile(filepath, diagonalCipher, SerType::BINARY) == false) {
				cerr << "[Approach 81] Error: cannot deserialize from \"" << filepath << "\"" << endl;
				continue;
			}
			allDiagonals.push_back(diagonalCipher);
		}
	}

	// Cache with pre-rotation on GPU (enables on-demand BSGS during query)
	gpuHelper_->CacheDiagonalsWithPreRotationOnGPU(allDiagonals, numMatrices, VECTOR_DIM, babyStepSize);

	auto endTime   = chrono::steady_clock::now();
	double elapsed = chrono::duration<double>(endTime - startTime).count();
	cout << "[Approach 81] Cached " << allDiagonals.size() << " diagonals with pre-rotation (n1=" << babyStepSize << ") on GPU in " << fixed << setprecision(2)
		 << elapsed << "s" << endl;
}

void DiagonalBSGSSimpleSenderGPU::cacheDiagonalsOnGPU(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals) {
	if (!isGPUReady())
		return;

	size_t batchSize   = cc->GetEncodingParams()->GetBatchSize();
	size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

	cout << "[Approach 81] Caching " << preloadedDiagonals.size() << " preloaded diagonals with pre-rotation on GPU..." << endl;
	auto startTime = chrono::steady_clock::now();

	// Cache with pre-rotation on GPU (enables on-demand BSGS during query)
	gpuHelper_->CacheDiagonalsWithPreRotationOnGPU(preloadedDiagonals, numMatrices, VECTOR_DIM, babyStepSize);

	auto endTime   = chrono::steady_clock::now();
	double elapsed = chrono::duration<double>(endTime - startTime).count();
	cout << "[Approach 81] Cached diagonals with pre-rotation (n1=" << babyStepSize << ") on GPU in " << fixed << setprecision(2) << elapsed << "s" << endl;
}

void DiagonalBSGSSimpleSenderGPU::streamDiagonalsFromDisk(const string& serialDir) {
	if (!isGPUReady()) {
		cerr << "[Approach 81] Cannot stream diagonals - GPU not ready" << endl;
		return;
	}

	if (gpuCacheInitialized_) {
		cout << "[Approach 81] GPU cache already initialized" << endl;
		return;
	}

	size_t batchSize   = cc->GetEncodingParams()->GetBatchSize();
	size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

	cout << "[Approach 81] Streaming diagonals disk→GPU (memory-efficient mode)..." << endl;
	auto startTime = chrono::steady_clock::now();

	// Use babyStepSize (n1=23) for BSGS pre-rotation during streaming
	bool success = gpuHelper_->StreamDiagonalsToGPU(serialDir, VECTOR_DIM, numMatrices, babyStepSize);

	auto endTime   = chrono::steady_clock::now();
	double elapsed = chrono::duration<double>(endTime - startTime).count();

	if (success) {
		gpuCacheInitialized_ = true;
		cout << "[Approach 81] Streamed " << (numMatrices * VECTOR_DIM) << " diagonals disk→GPU with pre-rotation (n1=" << babyStepSize << ") in " << fixed
			 << setprecision(2) << elapsed << "s" << endl;
	} else {
		cerr << "[Approach 81] Failed to stream diagonals from disk" << endl;
	}
}

// -------------------- MEMBERSHIP SCENARIO --------------------

Ciphertext<DCRTPoly> DiagonalBSGSSimpleSenderGPU::membershipScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) {
	cout << "[Approach 81] Running GPU membership scenario with on-demand BSGS..." << endl;

	if (!isGPUReady() || !gpuHelper_->areDiagonalsCached()) {
		cout << "[Approach 81] GPU not ready, falling back to CPU" << endl;
		return DiagonalBSGSSimpleSender::membershipScenario(queryCipher);
	}

	auto totalStart = chrono::steady_clock::now();

	queryCipherForClone_ = queryCipher[0];

	// Compute baby step rotations only (no giant steps needed - done on-demand)
	// Check GPU helper cache first - main.cpp may have pre-cached baby steps
	if (!gpuHelper_->areBabyStepsCached()) {
		vector<Ciphertext<DCRTPoly>> babySteps(babyStepSize);
		babySteps[0] = queryCipher[0];
		for (size_t i = 1; i < static_cast<size_t>(babyStepSize); ++i) {
			babySteps[i] = cc->EvalRotate(queryCipher[0], static_cast<int>(i));
		}
		gpuHelper_->CacheBabyStepsOnGPU(babySteps);
		rotationsCached_ = true;
	}

	size_t batchSize   = cc->GetEncodingParams()->GetBatchSize();
	size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

	// On-demand BSGS pipeline with lazy relinearization
	Ciphertext<DCRTPoly> membershipCipher;
	try {
		membershipCipher = gpuHelper_->EvalOnDemandBSGSLazyChebyshevAndAggregateOnGPU(
		  numMatrices, VECTOR_DIM, babyStepSize, queryCipherForClone_, MATCH_THRESHOLD, COMP_DEPTH_GPU, batchSize);
	} catch (const std::exception& e) {
		cerr << "[Approach 81] On-demand BSGS pipeline failed: " << e.what() << endl;
		throw;
	}

	auto totalEnd	 = chrono::steady_clock::now();
	double totalTime = chrono::duration<double>(totalEnd - totalStart).count();

	cout << "[Approach 81] Membership total: " << fixed << setprecision(2) << totalTime << "s" << endl;

	return membershipCipher;
}

// -------------------- INDEX SCENARIO --------------------

vector<Ciphertext<DCRTPoly>> DiagonalBSGSSimpleSenderGPU::indexScenario(vector<Ciphertext<DCRTPoly>>& queryCipher) {
	cout << "[Approach 81] Running GPU index scenario..." << endl;

	if (!isGPUReady() || !gpuHelper_->areDiagonalsCached()) {
		cout << "[Approach 81] GPU not ready, falling back to CPU" << endl;
		return DiagonalBSGSSimpleSender::indexScenario(queryCipher);
	}

	cout << "[Approach 81] Running GPU index scenario with batched on-demand BSGS..." << endl;

	auto totalStart = chrono::steady_clock::now();

	queryCipherForClone_ = queryCipher[0];

	// Compute baby step rotations if needed (only 23 rotations)
	if (!gpuHelper_->areBabyStepsCached()) {
		vector<Ciphertext<DCRTPoly>> babySteps(babyStepSize);
		babySteps[0] = queryCipher[0];
		for (size_t i = 1; i < static_cast<size_t>(babyStepSize); ++i) {
			babySteps[i] = cc->EvalRotate(queryCipher[0], static_cast<int>(i));
		}
		gpuHelper_->CacheBabyStepsOnGPU(babySteps);
		rotationsCached_ = true;
	}

	size_t batchSize   = cc->GetEncodingParams()->GetBatchSize();
	size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

	// On-demand BSGS + Chebyshev (lazy relin)
	vector<Ciphertext<DCRTPoly>> scoreCipher =
	  gpuHelper_->EvalOnDemandBSGSLazyChebyshevBatchedOnGPU(numMatrices, VECTOR_DIM, babyStepSize, queryCipherForClone_, MATCH_THRESHOLD, COMP_DEPTH_GPU);

	auto totalEnd	 = chrono::steady_clock::now();
	double totalTime = chrono::duration<double>(totalEnd - totalStart).count();

	cout << "[Approach 81] Index total: " << fixed << setprecision(2) << totalTime << "s" << endl;

	return scoreCipher;
}
