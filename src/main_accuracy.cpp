//  Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens,
//  Micah Brody, Taeho Jung Portions copyright (c) 2026 LG Electronics, Inc.
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

// General functionality header files
#include "../include/config.h"
#include "../include/exp_approaches.h"
#include "../include/openFHE_wrapper.h"
#include "../include/vector_utils.h"
#include "openfhe.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Receiver class header files
#include "../include/receiver_base.h"
#include "../include/receiver_blind.h"
#include "../include/receiver_diag.h"
#include "../include/receiver_diag_bsgs_precomp.h"
#include "../include/receiver_diag_bsgs_precomp_opt.h"
#include "../include/receiver_diag_orig_bsgs.h"
#include "../include/receiver_grote.h"
#include "../include/receiver_hers.h"

// Enroller class header files
#include "../include/enroller_base.h"
#include "../include/enroller_blind.h"
#include "../include/enroller_diag.h"
#include "../include/enroller_diag_bsgs_precomp.h"
#include "../include/enroller_diag_bsgs_precomp_opt.h"
#include "../include/enroller_diag_orig_bsgs.h"
#include "../include/enroller_hers.h"

// Sender class header files
#include "../include/sender_base.h"
#include "../include/sender_blind.h"
#include "../include/sender_diag.h"
#include "../include/sender_diag_bsgs_precomp.h"
#include "../include/sender_diag_bsgs_precomp_opt.h"
#include "../include/sender_diag_orig_bsgs.h"
#include "../include/sender_grote.h"
#include "../include/sender_hers.h"

// Header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using namespace std;
namespace fs = std::filesystem;

// Define the global variable for MAX_NUM_CORES
size_t MAX_NUM_CORES;

#include <sys/stat.h>

// Helper function to check if a file exists
static bool fileExists(const std::string& path) {
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
}

// Check if all required serialized files exist for a given approach
static bool serializedDataExists(ExperimentalApproach approach, size_t numVectors, size_t batchSizeHint = 16384) {
	if (!fileExists("serial/cryptocontext.bin") || !fileExists("serial/publickey.bin") || !fileExists("serial/privatekey.bin") ||
	  !fileExists("serial/multkey.bin") || !fileExists("serial/sumkey.bin") || !fileExists("serial/rotkey.bin")) {
		return false;
	}

	size_t numMatrices = (numVectors + batchSizeHint - 1) / batchSizeHint;

	switch (approach) {
	case ExperimentalApproach::FastHESearchCPU:
	case ExperimentalApproach::BSGSOrigCPU:
	case ExperimentalApproach::BSGSPrecompCPU:
	case ExperimentalApproach::BSGSPrecompOptCPU:
		if (!fileExists("serial/db_diagonal/index0.bin"))
			return false;
		{
			size_t lastIdx = numMatrices * VECTOR_DIM - 1;
			if (!fileExists("serial/db_diagonal/index" + std::to_string(lastIdx) + ".bin"))
				return false;
		}
		break;

	case ExperimentalApproach::LiteratureBaseline:
	case ExperimentalApproach::Grote:
		if (!fileExists("serial/db/database0.bin"))
			return false;
		break;

	case ExperimentalApproach::BlindMatch:
		if (!fileExists("serial/db_blind/database0.bin"))
			return false;
		break;

	case ExperimentalApproach::Hers:
		if (!fileExists("serial/db/database0.bin"))
			return false;
		break;

	default: return false;
	}

	return true;
}

// Check if approach is a GPU approach
static bool isGPUApproach(ExperimentalApproach approach) {
	return approach == ExperimentalApproach::FastHESearchGPU || approach == ExperimentalApproach::BSGSGPU || approach == ExperimentalApproach::BSGSGPUPreRot;
}

// Check if approach is unsupported for accuracy testing
static bool isUnsupportedForAccuracy(ExperimentalApproach approach) {
	return approach == ExperimentalApproach::BSGSOnlineAggCPU;
}

// Compute depth for a given approach with a custom comparison depth override.
// This mirrors OpenFHEWrapper::computeRequiredDepth() but uses the
// caller-supplied compDepth instead of the compile-time COMP_DEPTH constant.
static size_t computeDepthWithOverride(ExperimentalApproach approach, size_t compDepth) {
	size_t depth = 0;
	switch (approach) {
	case ExperimentalApproach::LiteratureBaseline: depth = 1 + 2 + compDepth; break;
	case ExperimentalApproach::Grote: depth = 1 + 2 + ALPHA_DEPTH + 3 + compDepth; break;
	case ExperimentalApproach::BlindMatch: depth = 1 + 1 + compDepth; break;
	case ExperimentalApproach::Hers:
	case ExperimentalApproach::FastHESearchCPU:
	case ExperimentalApproach::BSGSOrigCPU:
	case ExperimentalApproach::BSGSPrecompCPU:
	case ExperimentalApproach::BSGSPrecompOptCPU: depth = 1 + compDepth; break;
	default: depth = 1 + compDepth; break;
	}
	return depth;
}

// Entry point
int main(int argc, char* argv[]) {

	// Initialize threading configuration from environment variables
	MAX_NUM_CORES = 1;
	omp_set_num_threads(MAX_NUM_CORES);

	const char* inner_threads_env = std::getenv("INNER_THREADS");
	if (inner_threads_env != nullptr) {
		int inner_threads = std::atoi(inner_threads_env);
		if (inner_threads > 0) {
			omp_set_max_active_levels(2);
			std::cout << "\tConfigured INNER_THREADS: " << inner_threads << std::endl;
		}
	}

	std::cout << "\tConfigured OUTER_THREADS: " << MAX_NUM_CORES << std::endl;
	cout << "\tRunning Setup Operations:" << endl;

	// ==================== Argument Parsing ====================
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " <startIndex> <endIndex> <approach> [frgc2_dir] [comp_depth]" << endl;
		cerr << endl;
		cerr << "  startIndex : first query index (0-based, max 49)" << endl;
		cerr << "  endIndex   : last query index (0-based, inclusive, max 49)" << endl;
		cerr << "  approach   : algorithm ID (CPU approaches only)" << endl;
		cerr << "  frgc2_dir  : path to directory containing FRGC2 files (default: "
				"../test)"
			 << endl;
		cerr << "  comp_depth : comparison depth override (default: " << COMP_DEPTH << ")" << endl;
		cerr << endl;
		cerr << "  Supported CPU approaches:" << endl;
		cerr << "    1 = Literature baseline" << endl;
		cerr << "    2 = GROTE" << endl;
		cerr << "    3 = Blind-Match" << endl;
		cerr << "    4 = HERS" << endl;
		cerr << "    5 = FastHE-Search diagonal (CPU)" << endl;
		cerr << "    6 = BSGS-Orig (CPU)" << endl;
		cerr << "    7 = BSGS-Precomp (CPU)" << endl;
		cerr << "    8 = BSGS-Precomp-Opt (CPU)" << endl;
		cerr << endl;
		cerr << "  Unsupported approaches:" << endl;
		cerr << "    9        = membership-only, no index scenario" << endl;
		cerr << "    51/81/812 = GPU (FRGC2 dataset does not fit in GPU memory)" << endl;
		cerr << endl;
		cerr << "  Example: " << argv[0] << " 0 49 5 ../test 8" << endl;
		return 1;
	}

	size_t queryStart = static_cast<size_t>(atoi(argv[1]));
	size_t queryEnd	  = static_cast<size_t>(atoi(argv[2]));
	int approachInt	  = atoi(argv[3]);

	// Optional: FRGC2 data directory (default: ../test)
	std::string frgc2Dir = "../test";
	if (argc > 4) {
		frgc2Dir = argv[4];
	}
	// Strip trailing slash for consistency
	if (!frgc2Dir.empty() && frgc2Dir.back() == '/') {
		frgc2Dir.pop_back();
	}

	// Optional: comparison depth override
	size_t compDepth = COMP_DEPTH; // default from config.h
	if (argc > 5) {
		compDepth = static_cast<size_t>(atoi(argv[5]));
		if (compDepth == 0) {
			cerr << "Error: comp_depth must be a positive integer, got: " << argv[5] << endl;
			return 1;
		}
		cout << "Comparison depth override: " << compDepth << " (compile-time default was " << COMP_DEPTH << ")" << endl;
	}

	if (queryStart > queryEnd || queryEnd >= 50) {
		cerr << "Error: invalid query range [" << queryStart << ", " << queryEnd << "]. Must be within [0, 49]." << endl;
		return 1;
	}

	// Resolve the approach enum
	const ExperimentalApproachMeta* approachMeta = get_approach_meta(approachInt);
	if (approachMeta == nullptr) {
		cerr << "Error: unknown approach ID " << approachInt << endl;
		return 1;
	}
	ExperimentalApproach expApproach = approachMeta->id;

	// Reject GPU approaches
	if (isGPUApproach(expApproach)) {
		cerr << "Error: GPU approach " << approachInt << " (" << approachMeta->display << ") is not supported for FRGC2 accuracy testing." << endl;
		cerr << "The FRGC2 dataset is too large to fit in GPU memory for this "
				"workload."
			 << endl;
		cerr << "Use CPU approaches (1-8) instead." << endl;
		return 1;
	}

	// Reject membership-only approaches
	if (isUnsupportedForAccuracy(expApproach)) {
		cerr << "Error: Approach " << approachInt << " (" << approachMeta->display << ") is membership-only and does not support the index scenario" << endl;
		cerr << "required for per-vector accuracy evaluation. Use approaches 1-8." << endl;
		return 1;
	}

	// Compute multiplicative depth (using CLI comp_depth if provided)
	size_t multDepth = computeDepthWithOverride(expApproach, compDepth);
	int scalFactor	 = 45;

	cout << "Query range: [" << queryStart << ", " << queryEnd << "] (" << (queryEnd - queryStart + 1) << " queries)" << endl;
	cout << "Experimental approach: " << approachMeta->display << " (ID=" << approachInt << ")" << endl;
	cout << "FRGC2 data directory: " << frgc2Dir << endl;
	cout << "Comparison depth: " << compDepth << "  Total multiplicative depth: " << multDepth << endl;

	// ==================== Read FRGC2 Dataset ====================
	std::string dbPath	  = frgc2Dir + "/frgc2-db.dat";
	std::string queryPath = frgc2Dir + "/frgc2-query.dat";
	std::string qidPath	  = frgc2Dir + "/frgc2-qid.txt";
	std::string dbidPath  = frgc2Dir + "/frgc2-dbid.txt";

	ifstream fileStream;
	fileStream.open(dbPath, ios::in);
	if (!fileStream.is_open()) {
		cerr << "Error: unable to open database file " << dbPath << endl;
		cerr << "Please ensure the FRGC2 accuracy test files exist in: " << frgc2Dir << endl;
		return 1;
	}

	size_t numVectors;
	fileStream >> numVectors;

	ifstream queryStream;
	queryStream.open(queryPath, ios::in);
	if (!queryStream.is_open()) {
		cerr << "Error: unable to open query file " << queryPath << endl;
		return 1;
	}
	vector<vector<double>> queryVector(50, vector<double>(VECTOR_DIM));
	for (size_t i = 0; i < 50; i++) {
		for (size_t j = 0; j < VECTOR_DIM; j++) {
			queryStream >> queryVector[i][j];
		}
	}
	queryStream.close();

	ifstream idStream;
	idStream.open(qidPath, ios::in);
	if (!idStream.is_open()) {
		cerr << "Error: unable to open query ID file " << qidPath << endl;
		return 1;
	}
	vector<size_t> queryID(50);
	for (size_t i = 0; i < 50; i++) {
		idStream >> queryID[i];
	}
	idStream.close();

	idStream.open(dbidPath, ios::in);
	if (!idStream.is_open()) {
		cerr << "Error: unable to open database ID file " << dbidPath << endl;
		return 1;
	}
	vector<size_t> databaseID(44228);
	for (size_t i = 0; i < 44228; i++) {
		idStream >> databaseID[i];
	}
	idStream.close();

	// ==================== CKKS Setup ====================
	CryptoContext<DCRTPoly> cc;
	cc->ClearEvalMultKeys();
	cc->ClearEvalAutomorphismKeys();
	CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
	PublicKey<DCRTPoly> pk;
	PrivateKey<DCRTPoly> sk;
	size_t batchSize;

	// Auto-detect whether to load from serialized data
	bool loadFromSerial = serializedDataExists(expApproach, numVectors);
	if (loadFromSerial) {
		cout << "Found existing serialized data, loading from disk..." << endl;
	} else {
		cout << "No serialized data found, generating fresh keys and encrypting "
				"database..."
			 << endl;
	}

	if (loadFromSerial) {

		if (!Serial::DeserializeFromFile("serial/cryptocontext.bin", cc, SerType::BINARY)) {
			cerr << "Error deserializing CryptoContext" << endl;
		}
		batchSize = cc->GetEncodingParams()->GetBatchSize();

		if (!Serial::DeserializeFromFile("serial/publickey.bin", pk, SerType::BINARY)) {
			cerr << "Error deserializing public key" << endl;
		}

		if (!Serial::DeserializeFromFile("serial/privatekey.bin", sk, SerType::BINARY)) {
			cerr << "Error deserializing private key" << endl;
		}

		ifstream multKeyDeserialFile("serial/multkey.bin", ios::in | ios::binary);
		if (multKeyDeserialFile.is_open()) {
			if (!cc->DeserializeEvalMultKey(multKeyDeserialFile, SerType::BINARY)) {
				cerr << "Error deserializing mult keys" << endl;
			}
			multKeyDeserialFile.close();
		} else {
			cerr << "Error deserializing mult keys" << endl;
		}

		ifstream sumKeyDeserialFile("serial/sumkey.bin", ios::in | ios::binary);
		if (sumKeyDeserialFile.is_open()) {
			if (!cc->DeserializeEvalSumKey(sumKeyDeserialFile, SerType::BINARY)) {
				cerr << "Error deserializing sum keys" << endl;
			}
			sumKeyDeserialFile.close();
		} else {
			cerr << "Error deserializing sum keys" << endl;
		}

		ifstream rotKeyDeserialFile("serial/rotkey.bin", ios::in | ios::binary);
		if (rotKeyDeserialFile.is_open()) {
			if (!cc->DeserializeEvalAutomorphismKey(rotKeyDeserialFile, SerType::BINARY)) {
				cerr << "Error deserializing rotation keys" << endl;
			}
			rotKeyDeserialFile.close();
		} else {
			cerr << "Error deserializing rotation keys" << endl;
		}
	} else {

		CCParams<CryptoContextCKKSRNS> parameters;
		parameters.SetSecurityLevel(HEStd_128_classic);
		parameters.SetMultiplicativeDepth(multDepth);
		parameters.SetScalingModSize(scalFactor);
		parameters.SetScalingTechnique(FIXEDMANUAL);

		cc = GenCryptoContext(parameters);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);

		batchSize = cc->GetEncodingParams()->GetBatchSize();

		cout << "Generating key pair... " << endl;
		auto keyPair = cc->KeyGen();
		pk			 = keyPair.publicKey;
		sk			 = keyPair.secretKey;

		cout << "Generating mult keys... " << endl;
		cc->EvalMultKeyGen(sk);

		cout << "Generating sum keys... " << endl;
		cc->EvalSumKeyGen(sk);

		cout << "Generating rotation keys... " << endl;
		vector<int> rotationFactors;

		if (expApproach == ExperimentalApproach::BSGSOrigCPU || expApproach == ExperimentalApproach::BSGSPrecompCPU || expApproach == ExperimentalApproach::BSGSPrecompOptCPU) {
			// BSGS optimization: only generate sqrt(VECTOR_DIM) rotation keys
			size_t babyStepSize = static_cast<size_t>(ceil(sqrt(double(VECTOR_DIM))));

			for (size_t i = 1; i < babyStepSize; i++) {
				rotationFactors.push_back(static_cast<int>(i));
			}

			for (size_t i = 1; i * babyStepSize < VECTOR_DIM; i++) {
				rotationFactors.push_back(static_cast<int>(i * babyStepSize));
			}

			for (int i = VECTOR_DIM; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(i);
			}
			for (int i = 1; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(-i);
			}

			cout << "BSGS mode: generating " << rotationFactors.size() << " rotation keys (vs " << (VECTOR_DIM - 1) << " in standard mode)" << endl;
		} else {
			// Standard mode: generate all rotation keys from 1 to VECTOR_DIM
			rotationFactors.resize(VECTOR_DIM - 1);
			iota(rotationFactors.begin(), rotationFactors.end(), 1);

			for (int i = VECTOR_DIM; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(i);
			}
			for (int i = 1; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(-i);
			}
		}

		cc->EvalRotateKeyGen(sk, rotationFactors);
	}

	cout << "CKKS scheme set up (depth = " << multDepth << ", batch size = " << batchSize << ", scaling factor = " << scalFactor
		 << ", comp_depth = " << compDepth << ")" << endl;

	// ==================== Read & Encrypt Database ====================
	vector<vector<double>> plaintextVectors(numVectors, vector<double>(VECTOR_DIM));
	cout << "Reading database vectors from file... " << endl;
	for (size_t i = 0; i < numVectors; i++) {
		for (size_t j = 0; j < VECTOR_DIM; j++) {
			fileStream >> plaintextVectors[i][j];
		}
		if (plaintextVectors[i][0] == 0.0) {
			cout << "Error at " << i << endl;
		}
	}

	if (!loadFromSerial) {

		cout << "Encrypting database vectors... " << endl;
		HersEnroller* enroller = nullptr;

		if (expApproach == ExperimentalApproach::LiteratureBaseline || expApproach == ExperimentalApproach::Grote) {
			enroller = new BaseEnroller(cc, pk, numVectors);
			static_cast<BaseEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::BlindMatch) {
			enroller = new BlindEnroller(cc, pk, numVectors);
			static_cast<BlindEnroller*>(enroller)->serializeDB(plaintextVectors, CHUNK_LEN);
		} else if (expApproach == ExperimentalApproach::Hers) {
			enroller = new HersEnroller(cc, pk, numVectors);
			static_cast<HersEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::FastHESearchCPU) {
			enroller = new DiagonalEnroller(cc, pk, numVectors);
			static_cast<DiagonalEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::BSGSOrigCPU) {
			enroller = new DiagonalOrigBSGSEnroller(cc, pk, numVectors);
			static_cast<DiagonalOrigBSGSEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::BSGSPrecompCPU) {
			enroller = new DiagonalBSGSPrecompEnroller(cc, pk, numVectors);
			static_cast<DiagonalBSGSPrecompEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::BSGSPrecompOptCPU) {
			enroller = new DiagonalBSGSPrecompOptEnroller(cc, pk, numVectors);
			static_cast<DiagonalBSGSPrecompOptEnroller*>(enroller)->serializeDB(plaintextVectors);
		}

		if (enroller == nullptr) {
			cerr << "Error: unsupported approach for enrollment in accuracy binary" << endl;
			return 1;
		}
		delete enroller;

		Serial::SerializeToFile("serial/cryptocontext.bin", cc, SerType::BINARY);
		Serial::SerializeToFile("serial/publickey.bin", pk, SerType::BINARY);
		Serial::SerializeToFile("serial/privatekey.bin", sk, SerType::BINARY);

		ofstream multKeyFile("serial/multkey.bin", ios::out | ios::binary);
		if (multKeyFile.is_open()) {
			if (!cc->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
				cerr << "Error serializing mult keys" << endl;
			}
			multKeyFile.close();
		} else {
			cerr << "Error serializing mult keys" << endl;
		}

		ofstream rotKeyFile("serial/rotkey.bin", ios::out | ios::binary);
		if (rotKeyFile.is_open()) {
			if (!cc->SerializeEvalAutomorphismKey(rotKeyFile, SerType::BINARY)) {
				cerr << "Error serializing rotation keys" << endl;
			}
			rotKeyFile.close();
		} else {
			cerr << "Error serializing rotation keys" << endl;
		}

		ofstream sumKeyFile("serial/sumkey.bin", ios::out | ios::binary);
		if (sumKeyFile.is_open()) {
			if (!cc->SerializeEvalSumKey(sumKeyFile, SerType::BINARY)) {
				cerr << "Error serializing sum keys" << endl;
			}
			sumKeyFile.close();
		} else {
			cerr << "Error serializing sum keys" << endl;
		}
	}

	fileStream.close();

	// ==================== Allocate Receiver & Sender ====================
	cout << endl << "\tRunning Experiments:" << endl;
	chrono::steady_clock::time_point start, end;
	chrono::duration<double> duration;

	Receiver* receiver = nullptr;
	Sender* sender	   = nullptr;
	vector<size_t> indexResults;

	switch (expApproach) {

	case ExperimentalApproach::LiteratureBaseline:
		receiver = new BaseReceiver(cc, pk, sk, numVectors);
		sender	 = new BaseSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::Grote:
		receiver = new GroteReceiver(cc, pk, sk, numVectors);
		sender	 = new GroteSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::BlindMatch:
		receiver = new BlindReceiver(cc, pk, sk, numVectors);
		sender	 = new BlindSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::Hers:
		receiver = new HersReceiver(cc, pk, sk, numVectors);
		sender	 = new HersSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::FastHESearchCPU:
		receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::BSGSOrigCPU:
		receiver = new DiagonalOrigBSGSReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalOrigBSGSSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::BSGSPrecompCPU:
		receiver = new DiagonalBSGSPrecompReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalBSGSPrecompSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::BSGSPrecompOptCPU:
		receiver = new DiagonalBSGSPrecompOptReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalBSGSPrecompOptSender(cc, pk, numVectors);
		break;

	default: cerr << "Error: unsupported approach for accuracy binary" << endl; return 1;
	}

	// ==================== Open Log File ====================
	std::string logFileName = "accuracy_approach" + std::to_string(approachInt) + "_depth" + std::to_string(compDepth) + "_q" + std::to_string(queryStart) +
	  "-" + std::to_string(queryEnd) + ".csv";
	ofstream logFile(logFileName, ios::out);
	if (!logFile.is_open()) {
		cerr << "Error: could not open log file: " << logFileName << endl;
		return 1;
	}
	logFile << "queryIndex,querySubjectID,approach,depth,comp_depth,scale,"
			   "numDBVectors,"
			<< "enc_TP,enc_FN,enc_TN,enc_FP," << "plain_TP,plain_FN,plain_TN,plain_FP," << "encQueryTime_s,idxCompTime_s,idxDecTime_s" << endl;
	cout << "Logging results to: " << logFileName << endl;

	// Aggregate counters across all queries
	size_t totalTP = 0, totalTN = 0, totalFP = 0, totalFN = 0;
	size_t totalTPPlain = 0, totalTNPlain = 0, totalFPPlain = 0, totalFNPlain = 0;
	double totalIdxCompTime = 0.0;

	// ==================== Query Loop ====================
	for (size_t queryIndex = queryStart; queryIndex <= queryEnd; ++queryIndex) {
		cout << endl << "========== Query " << queryIndex + 1 << "/" << queryEnd + 1 << " (Subject ID: " << queryID[queryIndex] << ") ==========" << endl;

		// Encrypt the query vector
		cout << "[Receiver]\tEncrypting query vector... " << flush;
		start									 = chrono::steady_clock::now();
		vector<Ciphertext<DCRTPoly>> queryCipher = receiver->encryptQuery(queryVector[queryIndex]);
		end										 = chrono::steady_clock::now();
		duration								 = end - start;
		double encQueryTime						 = duration.count();
		cout << "done (" << encQueryTime << "s)" << endl;

		// Perform index scenario
		cout << "[Sender]\tComputing index scenario... " << flush;
		start			   = chrono::steady_clock::now();
		auto indexCipher   = sender->indexScenario(queryCipher);
		end				   = chrono::steady_clock::now();
		duration		   = end - start;
		double idxCompTime = duration.count();
		cout << "done (" << idxCompTime << "s)" << endl;

		// Decrypt index results
		cout << "[Receiver]\tDecrypting index results... " << flush;
		start			  = chrono::steady_clock::now();
		indexResults	  = receiver->decryptIndex(indexCipher);
		end				  = chrono::steady_clock::now();
		duration		  = end - start;
		double idxDecTime = duration.count();
		cout << "done (" << idxDecTime << "s)" << endl;

		// Accuracy evaluation: decrypt raw slot values
		vector<double> boolVec = OpenFHEWrapper::decryptVectorToVector(cc, sk, indexCipher);
		bool isPositive, guessPositive, plaintextPositive;
		size_t tp = 0, tn = 0, fp = 0, fn = 0;
		size_t tpPlain = 0, tnPlain = 0, fpPlain = 0, fnPlain = 0;

		for (size_t i = 0; i < numVectors; i++) {
			isPositive		  = (queryID[queryIndex] == databaseID[i]);
			guessPositive	  = (boolVec[i] >= 1.0);
			plaintextPositive = (VectorUtils::plaintextCosineSim(queryVector[queryIndex], plaintextVectors[i]) >= MATCH_THRESHOLD);

			if (isPositive) {
				if (guessPositive)
					tp += 1;
				else
					fn += 1;
				if (plaintextPositive)
					tpPlain += 1;
				else
					fnPlain += 1;
			} else {
				if (guessPositive)
					fp += 1;
				else
					tn += 1;
				if (plaintextPositive)
					fpPlain += 1;
				else
					tnPlain += 1;
			}
		}

		// Print per-query results
		cout << "Query Subject ID:\t" << queryID[queryIndex] << endl;
		cout << "Total comparisons:\t" << numVectors << endl;
		cout << "Encrypted  TP=" << tp << " FN=" << fn << " TN=" << tn << " FP=" << fp << endl;
		cout << "Plaintext  TP=" << tpPlain << " FN=" << fnPlain << " TN=" << tnPlain << " FP=" << fpPlain << endl;

		// Write per-query CSV row
		logFile << queryIndex << "," << queryID[queryIndex] << "," << approachInt << "," << multDepth << "," << compDepth << "," << scalFactor << ","
				<< numVectors << "," << tp << "," << fn << "," << tn << "," << fp << "," << tpPlain << "," << fnPlain << "," << tnPlain << "," << fpPlain << ","
				<< fixed << setprecision(4) << encQueryTime << "," << idxCompTime << "," << idxDecTime << endl;

		// Accumulate totals
		totalTP += tp;
		totalTN += tn;
		totalFP += fp;
		totalFN += fn;
		totalTPPlain += tpPlain;
		totalTNPlain += tnPlain;
		totalFPPlain += fpPlain;
		totalFNPlain += fnPlain;
		totalIdxCompTime += idxCompTime;
	}
	// ==================== End Query Loop ====================

	logFile.close();

	// ==================== Aggregate Summary ====================
	size_t numQueries = queryEnd - queryStart + 1;
	cout << endl << "========== Aggregate Results (" << numQueries << " queries) ==========" << endl;
	cout << "Approach: " << approachMeta->display << " (ID=" << approachInt << ")  Depth: " << multDepth << "  Comp_depth: " << compDepth
		 << "  Scale: " << scalFactor << endl;
	cout << "Encrypted  Total TP=" << totalTP << " FN=" << totalFN << " TN=" << totalTN << " FP=" << totalFP << endl;
	cout << "Plaintext  Total TP=" << totalTPPlain << " FN=" << totalFNPlain << " TN=" << totalTNPlain << " FP=" << totalFPPlain << endl;

	double precision = (totalTP + totalFP > 0) ? double(totalTP) / double(totalTP + totalFP) : 0.0;
	double recall	 = (totalTP + totalFN > 0) ? double(totalTP) / double(totalTP + totalFN) : 0.0;
	double f1		 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
	double accuracy	 = (totalTP + totalTN + totalFP + totalFN > 0) ? double(totalTP + totalTN) / double(totalTP + totalTN + totalFP + totalFN) : 0.0;

	double precisionPlain = (totalTPPlain + totalFPPlain > 0) ? double(totalTPPlain) / double(totalTPPlain + totalFPPlain) : 0.0;
	double recallPlain	  = (totalTPPlain + totalFNPlain > 0) ? double(totalTPPlain) / double(totalTPPlain + totalFNPlain) : 0.0;
	double f1Plain		  = (precisionPlain + recallPlain > 0) ? 2.0 * precisionPlain * recallPlain / (precisionPlain + recallPlain) : 0.0;
	double accuracyPlain  = (totalTPPlain + totalTNPlain + totalFPPlain + totalFNPlain > 0) ?
	   double(totalTPPlain + totalTNPlain) / double(totalTPPlain + totalTNPlain + totalFPPlain + totalFNPlain) :
	   0.0;

	cout << "Encrypted  Precision=" << fixed << setprecision(6) << precision << "  Recall=" << recall << "  F1=" << f1 << "  Accuracy=" << accuracy << endl;
	cout << "Plaintext  Precision=" << fixed << setprecision(6) << precisionPlain << "  Recall=" << recallPlain << "  F1=" << f1Plain
		 << "  Accuracy=" << accuracyPlain << endl;

	double amortizedIdxTime = totalIdxCompTime / numQueries;
	cout << "Total index scenario time=" << fixed << setprecision(4) << totalIdxCompTime << "s  Amortized=" << amortizedIdxTime << "s/query" << endl;

	cout << "Results written to: " << logFileName << endl;

	delete receiver;
	delete sender;

	// Clean up serialized data after successful completion
	try {
		std::uintmax_t numFilesRemoved = fs::remove_all("serial");
		cout << "Cleaned up serialized data (" << numFilesRemoved << " items removed from serial/)" << endl;
	} catch (const fs::filesystem_error& e) {
		std::cerr << "Warning: Could not remove serial/ folder: " << e.what() << endl;
	}

	cout << endl << "\tProgram successfully terminated" << endl;
	return 0;
}
