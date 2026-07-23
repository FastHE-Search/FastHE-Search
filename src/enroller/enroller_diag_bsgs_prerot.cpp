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
// ** enroller_diag_bsgs_prerot.cpp: Plaintext pre-rotation enroller for Approach 812
//
// Identical to DiagonalEnroller::serializeDB() EXCEPT:
//   After concatenateRows() produces the final plaintext rows, each row
//   corresponding to diagonal k (within its VECTOR_DIM block) is cyclically
//   shifted by -(floor(k / n1) * n1) positions in plaintext, BEFORE encryption.
//
// This makes the encrypted diagonals "pre-rotated" from the server's perspective,
// so the server can skip the homomorphic pre-rotation step entirely.

#include "../../include/enroller_diag_bsgs_prerot.h"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>

using namespace std;

// -------------------- CONSTRUCTOR --------------------

DiagonalBSGSPreRotEnroller::DiagonalBSGSPreRotEnroller(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam, size_t vectorParam)
: DiagonalEnroller(ccParam, pkParam, vectorParam), babyStepSize_(static_cast<size_t>(ceil(sqrt(double(VECTOR_DIM))))),
  giantStepSize_(static_cast<size_t>(ceil(double(VECTOR_DIM) / double(babyStepSize_)))) {
}

// -------------------- PLAINTEXT ROTATION --------------------

void DiagonalBSGSPreRotEnroller::applyPlaintextRotation(vector<double>& row, int amount, size_t vectorDim) {
	// Must match CKKS Rot(ct, amount) semantics EXACTLY:
	//   output[i] = input[(i + amount) mod numSlots]
	//
	// A CKKS rotation operates on ALL numSlots slots as one circular buffer,
	// NOT per-block.  Data can shift across VECTOR_DIM block boundaries.
	// The giant-step Rot(partial, +j*n1) in the online phase will undo this
	// rotation perfectly, just as it undoes the homomorphic pre-rotation in A81.
	//
	// Uses std::rotate for O(1) extra space (no temp buffer allocation).
	// std::rotate(first, middle, last) moves [middle, last) to the front.
	// For output[i] = input[(i+shift) % n], we left-rotate by `shift`.

	if (amount == 0)
		return;

	size_t numSlots = row.size();
	int shift		= ((amount % (int)numSlots) + (int)numSlots) % (int)numSlots;
	if (shift == 0)
		return;

	std::rotate(row.begin(), row.begin() + shift, row.end());
}

// -------------------- SERIALIZE DB --------------------

void DiagonalBSGSPreRotEnroller::serializeDB(vector<vector<double>>& database) {
	// Create directories
	string dirName = "serial/";
	if (!filesystem::exists(dirName)) {
		if (!filesystem::create_directory(dirName)) {
			cerr << "Error: Failed to create directory \"" + dirName + "\"" << endl;
			return;
		}
	}
	dirName = "serial/db_diagonal";
	if (!filesystem::exists(dirName)) {
		if (!filesystem::create_directory(dirName)) {
			cerr << "Error: Failed to create directory \"" + dirName + "\"" << endl;
		}
	}

// Normalize all database vectors
#pragma omp parallel for num_threads(MAX_NUM_CORES)
	for (size_t i = 0; i < numVectors; i++) {
		database[i] = VectorUtils::plaintextNormalize(database[i], VECTOR_DIM);
	}

	// Standard diagonal packing (same as DiagonalEnroller)
	vector<vector<vector<double>>> squareMatrices = splitIntoSquareMatrices(database, VECTOR_DIM);

	vector<vector<vector<double>>> allDiagonalMatrices;
	for (auto& squareMatrix : squareMatrices) {
		vector<vector<double>> diagonals = preprocessToDiagonalForm(squareMatrix);
		allDiagonalMatrices.push_back(diagonals);
	}

	vector<vector<double>> concatenatedRows = concatenateRows(allDiagonalMatrices);

	// =====================================================================
	// KEY DIFFERENCE: Apply plaintext pre-rotation BEFORE encryption
	// For diagonal index k (within its VECTOR_DIM block), rotate by
	// -(floor(k / n1) * n1) positions.  This is the same rotation that
	// Approach 81 does homomorphically on the GPU.
	// =====================================================================
	int n1			  = static_cast<int>(babyStepSize_);
	size_t preRotated = 0;

	cout << "[Approach 812] Applying plaintext pre-rotations (n1=" << n1 << ")..." << flush;
	auto preRotStart = chrono::steady_clock::now();

#pragma omp parallel for num_threads(MAX_NUM_CORES) reduction(+ : preRotated)
	for (size_t i = 0; i < concatenatedRows.size(); ++i) {
		int k		  = static_cast<int>(i % VECTOR_DIM); // diagonal index within the block
		int giantStep = (k / n1) * n1;
		if (giantStep > 0) {
			// Rotate by -giantStep (same as Rot_{-j*n1} in BSGS)
			applyPlaintextRotation(concatenatedRows[i], -giantStep, VECTOR_DIM);
			preRotated++;
		}
	}
	auto preRotEnd		= chrono::steady_clock::now();
	lastPreRotationTimeSec_ = chrono::duration<double>(preRotEnd - preRotStart).count();

	cout << " done (" << preRotated << "/" << concatenatedRows.size() << " rows shifted, " << fixed << setprecision(4) << lastPreRotationTimeSec_ << "s)"
		 << endl;

	// =====================================================================
	// Encrypt and serialize (identical to DiagonalEnroller from here on)
	// =====================================================================
	struct WorkItem {
		Ciphertext<DCRTPoly> cipher;
		size_t index;
	};

	queue<WorkItem> workQueue;
	mutex queueMutex;
	condition_variable queueCV;
	atomic<bool> encryptionDone(false);
	const size_t maxQueueSize = 32;
	const size_t numIOThreads = 6;

	// Consumer threads
	vector<thread> ioThreads;
	for (size_t t = 0; t < numIOThreads; t++) {
		ioThreads.emplace_back([&]() {
			while (true) {
				WorkItem item;
				{
					unique_lock<mutex> lock(queueMutex);
					queueCV.wait(lock, [&]() { return !workQueue.empty() || encryptionDone.load(); });
					if (workQueue.empty() && encryptionDone.load())
						break;
					if (!workQueue.empty()) {
						item = move(workQueue.front());
						workQueue.pop();
						queueCV.notify_all();
					} else {
						continue;
					}
				}
				string filepath = "serial/db_diagonal/index" + to_string(item.index) + ".bin";
				if (!Serial::SerializeToFile(filepath, item.cipher, SerType::BINARY)) {
					cerr << "Error: serialization failed (cannot write to " + filepath + ")" << endl;
				}
			}
		});
	}

// Producer: encrypt in parallel
#pragma omp parallel for num_threads(MAX_NUM_CORES) schedule(dynamic)
	for (size_t i = 0; i < concatenatedRows.size(); i++) {
		Ciphertext<DCRTPoly> encrypted = OpenFHEWrapper::encryptFromVector(cc, pk, concatenatedRows[i]);
		{
			unique_lock<mutex> lock(queueMutex);
			queueCV.wait(lock, [&]() { return workQueue.size() < maxQueueSize; });
			workQueue.push({ move(encrypted), i });
			queueCV.notify_one();
		}
	}

	encryptionDone.store(true);
	queueCV.notify_all();
	for (auto& t : ioThreads) {
		t.join();
	}
}
