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
// ** sender_diag_bsgs_simple_gpu_prerot.cpp: Approach 812 GPU sender
//
// Uses CachePreRotatedDiagonalsOnGPU / StreamPreRotatedDiagonalsToGPU
// (no homomorphic rotation) instead of the standard pre-rotation path.
// The online pipeline (membershipScenario / indexScenario) is inherited
// from DiagonalBSGSSimpleSenderGPU (Approach 81) and is unchanged.

#include "../../include/sender_diag_bsgs_simple_gpu_prerot.h"
#include "../../include/gpu_hydia_helper.cuh"
#include <chrono>
#include <iomanip>
#include <cmath>
#include <iostream>

using namespace std;

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSSimpleSenderGPU812::DiagonalBSGSSimpleSenderGPU812(
    CryptoContext<DCRTPoly> ccParam,
    PublicKey<DCRTPoly> pkParam,
    size_t vectorParam,
    GPUHydiaHelper* gpuHelper)
    : DiagonalBSGSSimpleSenderGPU(ccParam, pkParam, vectorParam, gpuHelper)
{
    if (gpuHelper_ && gpuHelper_->isReady()) {
        cout << "[Approach 812] GPU BSGS Sender (enroller pre-rotation) initialized:" << endl;
        cout << "  VECTOR_DIM = " << VECTOR_DIM << endl;
        cout << "  babyStepSize = " << babyStepSize << endl;
        cout << "  giantStepSize = " << giantStepSize << endl;
        cout << "  Diagonals pre-rotated by enroller — NO negative rotation keys needed" << endl;
    }
}

// -------------------- GPU CACHE (NO ROTATION) --------------------

void DiagonalBSGSSimpleSenderGPU812::initializeGPUCache() {
    if (!isGPUReady()) {
        cerr << "[Approach 812] Cannot initialize GPU cache - GPU not ready" << endl;
        return;
    }
    if (gpuCacheInitialized_) {
        cout << "[Approach 812] GPU cache already initialized" << endl;
        return;
    }
    cacheDiagonalsOnGPU();
    gpuCacheInitialized_ = true;
}

void DiagonalBSGSSimpleSenderGPU812::initializeGPUCache(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals) {
    if (!isGPUReady()) {
        cerr << "[Approach 812] Cannot initialize GPU cache - GPU not ready" << endl;
        return;
    }
    if (gpuCacheInitialized_) {
        cout << "[Approach 812] GPU cache already initialized" << endl;
        return;
    }
    cout << "[Approach 812] Using preloaded pre-rotated diagonals (skipping file I/O)" << endl;
    cacheDiagonalsOnGPU(preloadedDiagonals);
    gpuCacheInitialized_ = true;
}

void DiagonalBSGSSimpleSenderGPU812::cacheDiagonalsOnGPU() {
    if (!isGPUReady()) return;

    cout << "[Approach 812] Loading pre-rotated diagonals to GPU (no server-side rotation)..." << endl;
    auto startTime = chrono::steady_clock::now();

    size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
    size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

    // Load all diagonals from serialized files
    vector<Ciphertext<DCRTPoly>> allDiagonals;
    allDiagonals.reserve(numMatrices * VECTOR_DIM);

    for (size_t m = 0; m < numMatrices; ++m) {
        for (size_t i = 0; i < VECTOR_DIM; ++i) {
            string filepath = "serial/db_diagonal/index" + to_string(m * VECTOR_DIM + i) + ".bin";
            Ciphertext<DCRTPoly> diagonalCipher;
            if (!Serial::DeserializeFromFile(filepath, diagonalCipher, SerType::BINARY)) {
                cerr << "[Approach 812] Error: cannot deserialize from \"" << filepath << "\"" << endl;
                continue;
            }
            allDiagonals.push_back(diagonalCipher);
        }
    }

    // Upload to GPU WITHOUT rotation — diagonals already pre-rotated by enroller
    gpuHelper_->CachePreRotatedDiagonalsOnGPU(allDiagonals, numMatrices, VECTOR_DIM, babyStepSize);

    auto endTime = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(endTime - startTime).count();
    cout << "[Approach 812] Cached " << allDiagonals.size()
         << " pre-rotated diagonals on GPU (n1=" << babyStepSize << ") in "
         << fixed << setprecision(2) << elapsed << "s — zero homomorphic rotations" << endl;
}

void DiagonalBSGSSimpleSenderGPU812::cacheDiagonalsOnGPU(const vector<Ciphertext<DCRTPoly>>& preloadedDiagonals) {
    if (!isGPUReady()) return;

    size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
    size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

    cout << "[Approach 812] Caching " << preloadedDiagonals.size()
         << " preloaded pre-rotated diagonals on GPU..." << endl;
    auto startTime = chrono::steady_clock::now();

    gpuHelper_->CachePreRotatedDiagonalsOnGPU(preloadedDiagonals, numMatrices, VECTOR_DIM, babyStepSize);

    auto endTime = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(endTime - startTime).count();
    cout << "[Approach 812] Cached pre-rotated diagonals (n1=" << babyStepSize
         << ") on GPU in " << fixed << setprecision(2) << elapsed << "s" << endl;
}

void DiagonalBSGSSimpleSenderGPU812::streamDiagonalsFromDisk(const string& serialDir) {
    if (!isGPUReady()) {
        cerr << "[Approach 812] Cannot stream diagonals - GPU not ready" << endl;
        return;
    }
    if (gpuCacheInitialized_) {
        cout << "[Approach 812] GPU cache already initialized" << endl;
        return;
    }

    size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
    size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));

    cout << "[Approach 812] Streaming pre-rotated diagonals disk→GPU (no rotation)..." << endl;
    auto startTime = chrono::steady_clock::now();

    // Use StreamPreRotatedDiagonalsToGPU — NO negative rotation keys, NO rotation on GPU
    bool success = gpuHelper_->StreamPreRotatedDiagonalsToGPU(serialDir, VECTOR_DIM, numMatrices, babyStepSize);

    auto endTime = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(endTime - startTime).count();

    if (success) {
        gpuCacheInitialized_ = true;
        cout << "[Approach 812] Streamed " << (numMatrices * VECTOR_DIM)
             << " pre-rotated diagonals disk→GPU (n1=" << babyStepSize
             << ") in " << fixed << setprecision(2) << elapsed << "s" << endl;
    } else {
        cerr << "[Approach 812] Failed to stream diagonals from disk" << endl;
    }
}
