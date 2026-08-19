// clang-format off
//
//  Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
//  Portions copyright (c) 2026 LG Electronics, Inc.
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
// clang-format on

// ** Holds configuration parameters like file paths, default values, and any
// other constant values

// General functionality header files
#include "../include/config.h"
#include "../include/exp_approaches.h"
#include "../include/openFHE_wrapper.h"
#include "../include/vector_utils.h"
#include "openfhe.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <omp.h>
#include <sys/resource.h>
#include <sys/stat.h>

// Define the global variable for MAX_NUM_CORES
size_t MAX_NUM_CORES;

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
#include "../include/enroller_diag_bsgs_prerot.h"
#include "../include/sender_base.h"
#include "../include/sender_blind.h"
#include "../include/sender_diag.h"
#include "../include/sender_diag_bsgs_precomp.h"
#include "../include/sender_diag_bsgs_precomp_member_aggr_ct.h"
#include "../include/sender_diag_bsgs_precomp_opt.h"
#include "../include/sender_diag_bsgs_simple.h"
#include "../include/sender_diag_bsgs_simple_gpu.h"
#include "../include/sender_diag_bsgs_simple_gpu_prerot.h"
#include "../include/sender_diag_gpu.h"
#include "../include/sender_diag_orig_bsgs.h"
#include "../include/sender_grote.h"
#include "../include/sender_hers.h"

// GPU acceleration headers
#ifdef USE_GPU_ACCELERATION
#include "../include/gpu_fasthe_search_helper.cuh"
#endif

// Header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using namespace std;

// Helper function to get peak memory usage (RSS)
inline size_t getPeakMemoryKB() {
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss; // Returns KB on Linux, bytes on macOS
}

// Helper function to print memory usage delta
inline void print_mem_usage(const char* msg, size_t prevPeakKB = 0) {
	size_t memKB = getPeakMemoryKB();
	cout << msg << " Memory: " << (memKB / 1024.0) << " MB";
	if (prevPeakKB > 0) {
		long long deltaKB = (long long)memKB - (long long)prevPeakKB;
		cout << " [delta: " << (deltaKB > 0 ? "+" : "") << (deltaKB / 1024.0) << " MB]";
	}
	cout << endl;
}

// Helper function to get CPU time in seconds (user + system time)
inline double getCPUTime() {
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 + // user time
	  usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;		  // system time
}

// Parse whitespace-separated doubles from an in-memory buffer into fixed-length
// rows, using all available cores. Formatted extraction (`stream >> double`) is
// single-threaded and locale-aware, which dominates setup for large datasets
// (2^18 x 512 = 134M values). Chunks are cut on whitespace so no token spans two
// chunks; a counting pass then fixes each chunk's output offset.
static void parseDoubleRowsParallel(const char* data, size_t size, double* const* rows, size_t rowLength, size_t totalValues) {
	auto isSpace = [](char ch) { return std::isspace(static_cast<unsigned char>(ch)) != 0; };

	const int threadCount = std::max<int>(1, static_cast<int>(std::min<size_t>(MAX_NUM_CORES, size / (1 << 20) + 1)));
	std::vector<size_t> chunkBegin(static_cast<size_t>(threadCount) + 1, size);
	chunkBegin[0] = 0;
	for (int t = 1; t < threadCount; ++t) {
		size_t pos = std::min(size, static_cast<size_t>(t) * (size / static_cast<size_t>(threadCount)));
		while (pos < size && !isSpace(data[pos]))
			++pos; // push a straddling token entirely into the previous chunk
		chunkBegin[static_cast<size_t>(t)] = pos;
	}
	chunkBegin[static_cast<size_t>(threadCount)] = size;

	// Pass 1: count tokens per chunk.
	std::vector<size_t> tokenCount(static_cast<size_t>(threadCount), 0);
#pragma omp parallel for num_threads(threadCount) schedule(static)
	for (int t = 0; t < threadCount; ++t) {
		size_t count = 0;
		bool inToken  = false;
		for (size_t i = chunkBegin[static_cast<size_t>(t)]; i < chunkBegin[static_cast<size_t>(t) + 1]; ++i) {
			if (isSpace(data[i])) {
				inToken = false;
			} else if (!inToken) {
				inToken = true;
				++count;
			}
		}
		tokenCount[static_cast<size_t>(t)] = count;
	}

	// Exclusive prefix sum gives each chunk its first output index.
	std::vector<size_t> chunkOffset(static_cast<size_t>(threadCount) + 1, 0);
	for (int t = 0; t < threadCount; ++t)
		chunkOffset[static_cast<size_t>(t) + 1] = chunkOffset[static_cast<size_t>(t)] + tokenCount[static_cast<size_t>(t)];

	// Pass 2: parse each chunk directly into its destination rows.
#pragma omp parallel for num_threads(threadCount) schedule(static)
	for (int t = 0; t < threadCount; ++t) {
		size_t index      = chunkOffset[static_cast<size_t>(t)];
		const char* cursor = data + chunkBegin[static_cast<size_t>(t)];
		const char* end    = data + chunkBegin[static_cast<size_t>(t) + 1];
		size_t row = index / rowLength;
		size_t col = index % rowLength;
		while (cursor < end && index < totalValues) {
			while (cursor < end && isSpace(*cursor))
				++cursor;
			if (cursor >= end)
				break;
			double value    = 0.0;
			auto parsed = std::from_chars(cursor, end, value);
			if (parsed.ec != std::errc())
				break;
			cursor       = parsed.ptr;
			rows[row][col] = value;
			++index;
			if (++col == rowLength) {
				col = 0;
				++row;
			}
		}
	}

}

// GPU memory detection functions
#ifdef USE_GPU_ACCELERATION
static double getGPUTotalMemoryGB() {
	static double cached = -1.0;
	if (cached < 0) {
		const char* env = getenv("GPU_TOTAL_MEMORY_GB");
		if (env) {
			cached = atof(env);
		} else {
			size_t freeMem = 0, totalMem = 0;
			cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
			if (err == cudaSuccess && totalMem > 0) {
				cached = static_cast<double>(totalMem) / (1024.0 * 1024.0 * 1024.0);
			} else {
				cached = 48.0; // Fallback: RTX 8000
				cerr << "[GPU] Warning: Could not detect GPU memory, defaulting to " << cached << " GB" << endl;
			}
		}
	}
	return cached;
}

static double getGPUSafeMemoryLimitGB() {
	static double cached = -1.0;
	if (cached < 0) {
		const char* env = getenv("GPU_SAFE_MEMORY_GB");
		if (env) {
			cached = atof(env);
		} else {
			cached = getGPUTotalMemoryGB() * 0.95; // 95% of total
		}
	}
	return cached;
}

#define GPU_TOTAL_MEMORY_GB getGPUTotalMemoryGB()
#define GPU_MEMORY_SAFE_LIMIT_GB getGPUSafeMemoryLimitGB()
#endif

// Helper function to check if a file exists
bool fileExists(const string& path) {
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
}

bool envFlagEnabled(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) {
		return false;
	}

	std::string normalized(value);
	std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool keepSerialRequested() {
	return envFlagEnabled("KEEP_SERIAL") || fileExists(".keep_serial");
}

bool serializedKeyMaterialExists() {
	return fileExists("serial/cryptocontext.bin") && fileExists("serial/publickey.bin") && fileExists("serial/privatekey.bin") &&
	  fileExists("serial/multkey.bin") && fileExists("serial/sumkey.bin") && fileExists("serial/rotkey.bin");
}

// Check if all required serialized files exist for a given approach
bool serializedDataExists(ExperimentalApproach approach, size_t numVectors, size_t batchSizeHint = 16384) {
	if (!serializedKeyMaterialExists()) {
		return false;
	}

	// Check database files based on approach
	size_t numMatrices = (numVectors + batchSizeHint - 1) / batchSizeHint;

	if (approach == ExperimentalApproach::FastHESearchCPU || approach == ExperimentalApproach::BSGSOrigCPU || approach == ExperimentalApproach::BSGSPrecompCPU ||
	  approach == ExperimentalApproach::BSGSPrecompOptCPU || approach == ExperimentalApproach::BSGSOnlineAggCPU || approach == ExperimentalApproach::FastHESearchGPU ||
	  approach == ExperimentalApproach::BSGSGPU || approach == ExperimentalApproach::BSGSGPUPreRot) {
		// Diagonal approaches: check db_diagonal folder
		// Check first and last file as a quick validation
		if (!fileExists("serial/db_diagonal/index0.bin")) {
			return false;
		}
		size_t lastIdx = numMatrices * VECTOR_DIM - 1;
		if (!fileExists("serial/db_diagonal/index" + to_string(lastIdx) + ".bin")) {
			return false;
		}
	} else if (approach == ExperimentalApproach::LiteratureBaseline || approach == ExperimentalApproach::Grote) {
		// Base approach: check db folder
		if (!fileExists("serial/db/database0.bin")) {
			return false;
		}
	} else if (approach == ExperimentalApproach::BlindMatch) {
		// Blind approach: check db_blind folder
		if (!fileExists("serial/db_blind/database0.bin")) {
			return false;
		}
	} else if (approach == ExperimentalApproach::Hers) {
		// HERS approach: check db folder
		if (!fileExists("serial/db/database0.bin")) {
			return false;
		}
	}

	return true;
}

// Entry point of the application that orchestrates the flow

int main(int argc, char* argv[]) {

	// Initialize threading configuration from environment variables
	MAX_NUM_CORES = getMaxNumCores();

	// Set OpenMP threads for outer parallel regions (OUTER_THREADS)
	omp_set_num_threads(MAX_NUM_CORES);

	// Set OpenFHE internal thread limit (INNER_THREADS)
	// OpenFHE uses OpenMP internally, so we set the nested thread limit
	const char* inner_threads_env = std::getenv("INNER_THREADS");
	int inner_threads			  = 0;
	if (inner_threads_env != nullptr) {
		inner_threads = std::atoi(inner_threads_env);
		if (inner_threads > 0) {
			// Set the maximum number of threads for nested parallel regions
			omp_set_max_active_levels(2); // Allow nested parallelism
			std::cout << "\tConfigured INNER_THREADS: " << inner_threads << std::endl;
			// Note: The actual limiting happens in OpenFHE calls via OMP_THREAD_LIMIT
			// or by setting the nested thread count dynamically
		}
	}

	std::cout << "\tConfigured OUTER_THREADS: " << MAX_NUM_CORES << std::endl;
	cout << "\tRunning Setup Operations:" << endl;

	// Parse command line arg for experimental vector dataset
	ifstream fileStream;
	if (argc > 1) {
		fileStream.open(argv[1], ios::in);
	} else {
		cerr << "Error: input file not included" << endl;
		return 1;
	}
	if (!fileStream.is_open()) {
		cerr << "Error: unable to open input file" << endl;
		return 1;
	}
	size_t numVectors;
	fileStream >> numVectors;

	// Parse command line arg for experimental approach
	ExperimentalApproach expApproach;
	const ExperimentalApproachMeta* approachMeta = nullptr;
	if (argc > 2) {
		approachMeta = get_approach_meta(atoi(argv[2]));
		if (approachMeta != nullptr) {
			expApproach = approachMeta->id;
		}
	} else {
		cerr << "Error: approach argument not included" << endl;
		return 1;
	}
	if (approachMeta == nullptr) {
		cerr << "Error: approach must be from 1 to 9, or 51 (FastHE-Search-GPU), 81 "
				"(BSGS-GPU), or 812 (BSGS-GPU-PreRot)"
			 << endl;
		return 1;
	}

	bool keepSerialAfterRun = keepSerialRequested();
	bool reuseKeysOnly		= false;

	// Multi-GPU device selection. Default: single GPU 0.
	// Accepts "--gpus 0,1,2,3" or "--gpus=0,1" on the command line, or the
	// GPU_DEVICES environment variable (e.g. GPU_DEVICES=0,1). The database is
	// sharded across the listed GPUs and kept hot on each from the setup phase.
	auto parseGpuList = [](const std::string& s) {
		std::vector<int> v;
		std::string tok;
		for (char ch : s) {
			if (ch == ',' || ch == ' ') {
				if (!tok.empty()) {
					v.push_back(std::atoi(tok.c_str()));
					tok.clear();
				}
			} else {
				tok.push_back(ch);
			}
		}
		if (!tok.empty())
			v.push_back(std::atoi(tok.c_str()));
		return v;
	};
	std::vector<int> gpuDeviceList = { 0 };
	bool gpusFromCli			   = false;
	for (int argIndex = 3; argIndex < argc; ++argIndex) {
		std::string argValue(argv[argIndex]);
		if (argValue == "--keep-serial") {
			keepSerialAfterRun = true;
		} else if (argValue == "--reuse-keys") {
			reuseKeysOnly = true;
		} else if (argValue.rfind("--gpus=", 0) == 0) {
			auto v = parseGpuList(argValue.substr(7));
			if (!v.empty()) {
				gpuDeviceList = v;
				gpusFromCli	  = true;
			}
		} else if (argValue == "--gpus" && argIndex + 1 < argc) {
			auto v = parseGpuList(argv[++argIndex]);
			if (!v.empty()) {
				gpuDeviceList = v;
				gpusFromCli	  = true;
			}
		}
	}
	if (!gpusFromCli) {
		const char* gpuEnv = std::getenv("GPU_DEVICES");
		if (gpuEnv) {
			auto v = parseGpuList(gpuEnv);
			if (!v.empty())
				gpuDeviceList = v;
		}
	}
	if (gpuDeviceList.size() > 1) {
		cout << "[main] Multi-GPU enabled: devices = [";
		for (size_t i = 0; i < gpuDeviceList.size(); ++i)
			cout << gpuDeviceList[i] << (i + 1 < gpuDeviceList.size() ? ", " : "");
		cout << "]" << endl;
	}

	// Open global experiment-tracking file
	ofstream expStream;
	expStream.open(EXP_FILEPATH, ios::app);
	if (!expStream.is_open()) {
		cerr << "Error: experiment file not found" << endl;
		return 1;
	}

	// Setup/offline instrumentation metrics (seconds)
	double keyGenTime	  = 0.0;
	double rotKeyGenTime  = 0.0;
	double dbEncryptTime  = 0.0;
	double diagPreRotTime = 0.0;
	double gpuKeyUploadTime = 0.0;
	double gpuDBCacheTime = 0.0;

	// Compute required multiplicative depth based on approach used
	// Write approach used to stdout and experiment .csv file
	size_t multDepth = OpenFHEWrapper::computeRequiredDepth(static_cast<size_t>(expApproach));
	cout << "Experimental approach: " << approachMeta->display << endl;
	expStream << approachMeta->csvLabel << "," << flush;

	// Declare CKKS scheme elements
	CryptoContext<DCRTPoly> cc;
	cc->ClearEvalMultKeys();
	cc->ClearEvalAutomorphismKeys();
	CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
	PublicKey<DCRTPoly> pk;
	PrivateKey<DCRTPoly> sk;
	size_t batchSize;

// GPU helper pointer (used for GPU approaches)
#ifdef USE_GPU_ACCELERATION
	GPUFastHESearchHelper* gpuHelper __attribute__((unused)) = nullptr;
#endif

	// Auto-detect whether to load serialized keys and/or serialized database
	// payloads
	bool keyMaterialExists		= serializedKeyMaterialExists();
	bool databaseExists			= serializedDataExists(expApproach, numVectors);
	bool loadKeysFromSerial		= keyMaterialExists && (reuseKeysOnly || databaseExists);
	bool loadDatabaseFromSerial = databaseExists && !reuseKeysOnly;
	if (loadDatabaseFromSerial) {
		cout << "Found existing serialized data, loading from disk..." << endl;
	} else if (loadKeysFromSerial) {
		cout << "Found existing serialized keys, re-encrypting database for "
				"current dataset..."
			 << endl;
	} else {
		cout << "No serialized data found, generating fresh keys and encrypting "
				"database..."
			 << endl;
	}

	// Deserialize scheme context and keys if already serialized
	if (loadKeysFromSerial) {

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
	} else { // Else generate and serialize scheme context and keys

		CCParams<CryptoContextCKKSRNS> parameters;
		parameters.SetSecurityLevel(HEStd_128_classic);
		parameters.SetMultiplicativeDepth(multDepth);
		parameters.SetScalingModSize(45);
		parameters.SetScalingTechnique(FIXEDMANUAL);

		cc = GenCryptoContext(parameters);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);

		batchSize = cc->GetEncodingParams()->GetBatchSize();

		cout << "Generating key pair... " << endl;
		auto keyGenStart = chrono::steady_clock::now();
		auto keyPair = cc->KeyGen();
		pk			 = keyPair.publicKey;
		sk			 = keyPair.secretKey;

		cout << "Generating mult keys... " << endl;
		cc->EvalMultKeyGen(sk);

		cout << "Generating sum keys... " << endl;
		cc->EvalSumKeyGen(sk);
		auto keyGenEnd = chrono::steady_clock::now();
		keyGenTime	 = chrono::duration<double>(keyGenEnd - keyGenStart).count();

		cout << "Generating rotation keys... " << endl;
		vector<int> rotationFactors;

		if (expApproach == ExperimentalApproach::BSGSGPU) {
			// Approach 81 (GPU Simple BSGS): needs baby steps, positive giant steps,
			// NEGATIVE giant steps (for V2 pre-rotation), and binary EvalSum keys.
			size_t babyStepSize	 = ceil(sqrt(double(VECTOR_DIM)));
			size_t giantStepSize = ceil(double(VECTOR_DIM) / double(babyStepSize));

			// Baby step rotations: 1 to babyStepSize-1
			for (size_t i = 1; i < babyStepSize; i++) {
				rotationFactors.push_back(i);
			}

			// Positive giant step rotations: babyStepSize, 2*babyStepSize, ...
			for (size_t g = 1; g < giantStepSize; g++) {
				rotationFactors.push_back(static_cast<int>(g * babyStepSize));
			}

			// Negative giant step rotations for V2 pre-rotation during setup:
			// diag'[k] = rot(diag[k], -(k - k%n1))
			for (size_t g = 1; g < giantStepSize; g++) {
				rotationFactors.push_back(-static_cast<int>(g * babyStepSize));
			}

			// Positive binary rotation keys for GPU EvalSum (powers of 2)
			// Note: Negative binary keys are NOT needed — the GPU EvalSum loop
			// only uses positive shifts, and chebyshevCompare does no rotations.
			for (int i = 1; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(i);
			}

			// Remove duplicates (baby steps overlap with small binary keys)
			sort(rotationFactors.begin(), rotationFactors.end());
			rotationFactors.erase(unique(rotationFactors.begin(), rotationFactors.end()), rotationFactors.end());

			cout << "Approach 81 BSGS-GPU mode: generating " << rotationFactors.size() << " rotation keys (baby=" << (babyStepSize - 1) << ", giant=±"
				 << (giantStepSize - 1) << ", evalSum=" << int(log2(batchSize)) << ")" << endl;
		} else if (expApproach == ExperimentalApproach::BSGSGPUPreRot) {
			// Approach 812 (GPU BSGS with enroller pre-rotation):
			// Same as 81 but NO negative giant step keys — diagonals pre-rotated by
			// enroller.
			size_t babyStepSize	 = ceil(sqrt(double(VECTOR_DIM)));
			size_t giantStepSize = ceil(double(VECTOR_DIM) / double(babyStepSize));

			// Baby step rotations: 1 to babyStepSize-1
			for (size_t i = 1; i < babyStepSize; i++) {
				rotationFactors.push_back(i);
			}

			// Positive giant step rotations ONLY (no negative keys needed)
			for (size_t g = 1; g < giantStepSize; g++) {
				rotationFactors.push_back(static_cast<int>(g * babyStepSize));
			}

			// Positive binary rotation keys for GPU EvalSum (powers of 2)
			for (int i = 1; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(i);
			}

			// Remove duplicates
			sort(rotationFactors.begin(), rotationFactors.end());
			rotationFactors.erase(unique(rotationFactors.begin(), rotationFactors.end()), rotationFactors.end());

			cout << "Approach 812 BSGS-GPU-PreRot mode: generating " << rotationFactors.size() << " rotation keys (baby=" << (babyStepSize - 1) << ", giant=+"
				 << (giantStepSize - 1) << ", evalSum=" << int(log2(batchSize)) << ", NO negative keys)" << endl;
		} else if (expApproach == ExperimentalApproach::BSGSOrigCPU || expApproach == ExperimentalApproach::BSGSPrecompCPU ||
		  expApproach == ExperimentalApproach::BSGSPrecompOptCPU || expApproach == ExperimentalApproach::BSGSOnlineAggCPU) {
			// BSGS optimization: only generate sqrt(VECTOR_DIM) rotation keys
			size_t babyStepSize = ceil(sqrt(double(VECTOR_DIM)));

			// Baby step rotations: 1, 2, 3, ..., babyStepSize-1
			for (size_t i = 1; i < babyStepSize; i++) {
				rotationFactors.push_back(i);
			}

			// Giant step rotations: babyStepSize, 2*babyStepSize, 3*babyStepSize, ...
			for (size_t i = 1; i * babyStepSize < VECTOR_DIM; i++) {
				rotationFactors.push_back(i * babyStepSize);
			}

			// Note: No binary rotation keys needed for BSGS approaches.
			// EvalSum uses separate sum keys (from EvalSumKeyGen), and
			// chebyshevCompare performs no rotations.

			cout << "BSGS mode: generating " << rotationFactors.size() << " rotation keys (vs " << (VECTOR_DIM - 1) << " in standard mode)" << endl;
		} else if (expApproach == ExperimentalApproach::FastHESearchGPU) {
			// Approach 51 (GPU Pure Diagonal): same as standard but NO negative
			// binary keys. GPU pipeline never uses negative rotations (no giant-step
			// pre-rotation when n1=VECTOR_DIM).
			rotationFactors.resize(VECTOR_DIM - 1);
			iota(rotationFactors.begin(), rotationFactors.end(), 1); // {1..511}

			// positive binary rotation keys ≥ VECTOR_DIM for GPU EvalSum
			for (int i = VECTOR_DIM; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(i); // {512, 1024, 2048, 4096, 8192}
			}

			// NOTE: negative binary keys omitted — GPU EvalSum uses positive shifts
			// only, and there's no BSGS pre-rotation for pure diagonal mode.
			cout << "Approach 51 FastHE-Search-GPU mode: generating " << rotationFactors.size() << " rotation keys (diag 1-" << (VECTOR_DIM - 1) << " + binary "
				 << VECTOR_DIM << "-" << (batchSize / 2) << ", no negative keys)" << endl;
		} else {
			// Standard mode (CPU approaches): all rotation keys from 1 to VECTOR_DIM
			// + binary ± keys
			rotationFactors.resize(VECTOR_DIM - 1);
			iota(rotationFactors.begin(), rotationFactors.end(), 1);

			// generate positive binary rotation keys greater than VECTOR_DIM
			for (int i = VECTOR_DIM; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(i);
			}
			// generate negative binary rotation keys
			for (int i = 1; i < int(batchSize); i *= 2) {
				rotationFactors.push_back(-i);
			}
		}

		auto rotKeyStart = chrono::steady_clock::now();
		cc->EvalRotateKeyGen(sk, rotationFactors);
		auto rotKeyEnd = chrono::steady_clock::now();
		rotKeyGenTime = chrono::duration<double>(rotKeyEnd - rotKeyStart).count();
	}

	// OpenFHEWrapper::printSchemeDetails(parameters, cc);
	cout << "CKKS scheme set up (depth = " << multDepth << ", batch size = " << batchSize << ")" << endl;

	// Log number of vectors to experiment file
	expStream << numVectors << "," << flush;

	// Read in query vector from file
	vector<double> queryVector(VECTOR_DIM);
	for (size_t i = 0; i < VECTOR_DIM; i++) {
		fileStream >> queryVector[i];
	}

	// Serialize the context, keys and database vectors if not already
	vector<vector<double>> plaintextVectors(numVectors, vector<double>(VECTOR_DIM));
	double encDBVectorsTime = 0;
	diagPreRotTime			  = 0.0;
	if (!loadDatabaseFromSerial) {

		cout << "Reading database vectors from file... " << flush;
		auto readStart = chrono::steady_clock::now();
		{
			// Bulk-read the remaining payload once, then parse it with all cores.
			const streampos payloadStart = fileStream.tellg();
			fileStream.seekg(0, ios::end);
			const size_t payloadSize = static_cast<size_t>(fileStream.tellg() - payloadStart);
			fileStream.seekg(payloadStart);

			vector<char> payload(payloadSize);
			fileStream.read(payload.data(), static_cast<streamsize>(payloadSize));


			vector<double*> rows(numVectors);
			for (size_t i = 0; i < numVectors; i++)
				rows[i] = plaintextVectors[i].data();

			parseDoubleRowsParallel(payload.data(), static_cast<size_t>(fileStream.gcount()), rows.data(), VECTOR_DIM, numVectors * VECTOR_DIM);
		}
		cout << "done (" << fixed << setprecision(4) << chrono::duration<double>(chrono::steady_clock::now() - readStart).count() << "s)" << endl;

		cout << "Encrypting database vectors... " << flush;
		auto enrollStart = chrono::steady_clock::now();
		// Classes stored on heap to allow for cleaner polymorphism
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
		} else if (expApproach == ExperimentalApproach::BSGSOnlineAggCPU) {
			enroller = new DiagonalBSGSPrecompOptEnroller(cc, pk, numVectors);
			static_cast<DiagonalBSGSPrecompOptEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::FastHESearchGPU) {
			// Approach 51 uses the same serialized diagonal format as approach 5
			enroller = new DiagonalEnroller(cc, pk, numVectors);
			static_cast<DiagonalEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::BSGSGPU) {
			// Approach 81 uses same diagonal format as approach 5/9
			enroller = new DiagonalBSGSPrecompOptEnroller(cc, pk, numVectors);
			static_cast<DiagonalBSGSPrecompOptEnroller*>(enroller)->serializeDB(plaintextVectors);
		} else if (expApproach == ExperimentalApproach::BSGSGPUPreRot) {
			// Approach 812: pre-rotate diagonals in plaintext before encryption
			auto* preRotEnroller = new DiagonalBSGSPreRotEnroller(cc, pk, numVectors);
			enroller				 = preRotEnroller;
			preRotEnroller->serializeDB(plaintextVectors);
			diagPreRotTime = preRotEnroller->getLastPreRotationTimeSec();
		}
		delete enroller;
		{
			auto enrollEnd	 = chrono::steady_clock::now();
			encDBVectorsTime = chrono::duration<double>(enrollEnd - enrollStart).count();
			dbEncryptTime	 = std::max(0.0, encDBVectorsTime - diagPreRotTime);
			cout << "encrypt DB vectors done (" << fixed << setprecision(4) << encDBVectorsTime << "s)" << endl;
		}

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
	if (loadDatabaseFromSerial) {
		dbEncryptTime = 0.0;
	}
	fileStream.close();

	// Individual-query experiments begin here
	cout << endl << "\tRunning Experiments:" << endl;
	chrono::steady_clock::time_point start, end;
	chrono::duration<double> duration;
	double encQueryTime = 0, membCompTime = 0, membDecTime = 0, idxCompTime = 0, idxDecTime = 0;
	std::string resultStatusMembership = "UNKNOWN";

	Receiver* receiver = nullptr;
	Sender* sender	   = nullptr;
	bool membershipResult;
	vector<size_t> indexResults;

	// Allocate receiver and sender objects
	// receiver = new BaseReceiver(cc, pk, sk, numVectors);
	// sender = new BaseSender(cc, pk, numVectors);
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

	case ExperimentalApproach::BSGSOnlineAggCPU:
		receiver = new DiagonalBSGSPrecompOptReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalBSGSPrecompMemberAggrCtSender(cc, pk, numVectors);
		break;

	case ExperimentalApproach::FastHESearchGPU: {
#ifdef USE_GPU_ACCELERATION
		cout << "[FastHE-Search-GPU] Initializing GPU diagonal sender..." << endl;

		std::vector<double> dummy_vec(1, 1.0);
		auto dummy_pt = cc->MakeCKKSPackedPlaintext(dummy_vec);
		auto dummy_ct = cc->Encrypt(pk, dummy_pt); // Forces RNS scaling matrix completion

		KeyPair<DCRTPoly> keys;
		keys.publicKey		   = pk;
		keys.secretKey		   = sk;
		vector<int> gpuDevices = gpuDeviceList;
		auto gpuInitStart	   = chrono::steady_clock::now();
		gpuHelper			   = new GPUFastHESearchHelper(cc, keys, gpuDevices, batchSize);
		auto gpuInitEnd		   = chrono::steady_clock::now();
		gpuKeyUploadTime += chrono::duration<double>(gpuInitEnd - gpuInitStart).count();

		if (!gpuHelper->isReady()) {
			cerr << "[FastHE-Search-GPU] GPU initialization failed, falling back to CPU "
					"Diagonal approach"
				 << endl;
			delete gpuHelper;
			gpuHelper = nullptr;
			receiver  = new DiagonalReceiver(cc, pk, sk, numVectors);
			sender	  = new DiagonalSender(cc, pk, numVectors);
		} else {
			size_t numMatrices	= static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));
			size_t numDiagonals = numMatrices * VECTOR_DIM;

			size_t evalSumKeyCount = 0;
			for (size_t shift = 1; shift < batchSize; shift *= 2)
				evalSumKeyCount++;
			size_t numRotationKeys = (VECTOR_DIM - 1) + evalSumKeyCount;

			size_t ringDim	= cc->GetRingDimension();
			size_t numLimbs = cc->GetCryptoParameters()->GetElementParams()->GetParams().size();

			size_t bytesPerCt  = static_cast<size_t>(2 * numLimbs * ringDim * 8 * 2.5);
			size_t bytesPerKey = bytesPerCt * 3;

			double diagMemGB = (numDiagonals * bytesPerCt) / (1024.0 * 1024.0 * 1024.0);
			double keyMemGB	 = (numRotationKeys * bytesPerKey) / (1024.0 * 1024.0 * 1024.0);
			// Per-GPU footprint: diagonals are sharded across GPUs, rotation keys
			// are replicated on every GPU.
			size_t numGpusShard	   = gpuDeviceList.size();
			double diagMemPerGpuGB = diagMemGB / static_cast<double>(numGpusShard);
			double totalMemGB	   = diagMemPerGpuGB + keyMemGB;

			cout << "\n[FastHE-Search-GPU] Memory estimation:" << endl;
			cout << "  Diagonals:      " << numDiagonals << " x ~15MB = " << fixed << setprecision(2) << diagMemGB << " GB" << endl;
			cout << "  Rotation keys:  " << numRotationKeys << " x ~45MB = " << fixed << setprecision(2) << keyMemGB << " GB"
				 << " (DiagRot=" << (VECTOR_DIM - 1) << ", EvalSum=" << evalSumKeyCount << ")" << endl;
			cout << "  Total needed:   " << fixed << setprecision(2) << totalMemGB << " GB" << endl;
			cout << "  GPU safe limit: " << fixed << setprecision(1) << GPU_MEMORY_SAFE_LIMIT_GB << " GB (" << fixed << setprecision(0)
				 << (GPU_MEMORY_SAFE_LIMIT_GB / GPU_TOTAL_MEMORY_GB * 100) << "% of " << GPU_TOTAL_MEMORY_GB << "GB)" << endl;

			if (totalMemGB > GPU_MEMORY_SAFE_LIMIT_GB) {
				cerr << "\n[FastHE-Search-GPU] FATAL: Insufficient GPU memory for this dataset." << endl;
				cerr << "[FastHE-Search-GPU] Need " << fixed << setprecision(1) << totalMemGB << " GB but only " << GPU_MEMORY_SAFE_LIMIT_GB << " GB safely available." << endl;
				cerr << "[FastHE-Search-GPU] Options:" << endl;
				cerr << "  1. Use a smaller dataset" << endl;
				cerr << "  2. Use CPU-only approach 5" << endl;
				delete gpuHelper;
				gpuHelper = nullptr;
				return 1;
			}

			receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
			sender	 = new DiagonalSenderGPU(cc, pk, numVectors, gpuHelper);

			auto gpuSender = static_cast<DiagonalSenderGPU*>(sender);

			// Default: stream diagonals directly from disk→GPU (saves ~15 GB CPU RAM)
			// GPU_STREAM_DIAGS=0: opt-in to bulk-load all diagonals into CPU RAM
			// first
			const char* streamDiagsEnv = std::getenv("GPU_STREAM_DIAGS");
			bool useBulkMode		   = (streamDiagsEnv && std::string(streamDiagsEnv) == "0");

			if (!useBulkMode) {
				cout << "[FastHE-Search-GPU] Streaming diagonals disk→GPU (default mode)" << endl;
				auto cacheStart = chrono::steady_clock::now();
				gpuSender->streamDiagonalsFromDisk("serial");
				auto cacheEnd = chrono::steady_clock::now();
				gpuDBCacheTime += chrono::duration<double>(cacheEnd - cacheStart).count();
			} else {
				cout << "[FastHE-Search-GPU] Loading diagonals to GPU (bulk mode, "
						"GPU_STREAM_DIAGS=0)..."
					 << endl;
				auto cacheStart = chrono::steady_clock::now();
				auto loadStart = chrono::steady_clock::now();

				vector<Ciphertext<DCRTPoly>> preloadedDiagonals;
				preloadedDiagonals.resize(numDiagonals);

#pragma omp parallel for num_threads(MAX_NUM_CORES) schedule(dynamic)
				for (size_t i = 0; i < numDiagonals; ++i) {
					string filepath = "serial/db_diagonal/index" + to_string(i) + ".bin";
					Ciphertext<DCRTPoly> diagonal;
					if (Serial::DeserializeFromFile(filepath, diagonal, SerType::BINARY)) {
						preloadedDiagonals[i] = diagonal;
					} else {
						cerr << "[FastHE-Search-GPU] Error loading diagonal " << i << endl;
					}
				}

				auto loadEnd	= chrono::steady_clock::now();
				double loadTime = chrono::duration<double>(loadEnd - loadStart).count();
				cout << "[FastHE-Search-GPU] Loaded " << numDiagonals << " diagonals in " << loadTime << "s" << endl;

				gpuSender->initializeGPUCache(preloadedDiagonals);
					auto cacheEnd = chrono::steady_clock::now();
					gpuDBCacheTime += chrono::duration<double>(cacheEnd - cacheStart).count();
			}
		}
#else
		cout << "[FastHE-Search-GPU] GPU support not compiled. Falling back to CPU "
				"Diagonal approach."
			 << endl;
		receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalSender(cc, pk, numVectors);
#endif
		break;
	}

	case ExperimentalApproach::BSGSGPU: {
#ifdef USE_GPU_ACCELERATION
		// BSGS-Simple-GPU: GPU version of Simple BSGS
		cout << "[BSGS-Simple-GPU] Initializing GPU Simple BSGS sender..." << endl;

		// Initialize GPU helper
		KeyPair<DCRTPoly> keys;
		keys.publicKey		   = pk;
		keys.secretKey		   = sk;
		vector<int> gpuDevices = gpuDeviceList;
		auto gpuInitStart	   = chrono::steady_clock::now();
		gpuHelper			   = new GPUFastHESearchHelper(cc, keys, gpuDevices, batchSize);
		auto gpuInitEnd		   = chrono::steady_clock::now();
		gpuKeyUploadTime += chrono::duration<double>(gpuInitEnd - gpuInitStart).count();

		if (!gpuHelper->isReady()) {
			cerr << "[BSGS-Simple-GPU] GPU initialization failed, falling back to "
					"BSGS-Simple (CPU)"
				 << endl;
			delete gpuHelper;
			gpuHelper = nullptr;
			receiver  = new DiagonalReceiver(cc, pk, sk, numVectors);
			sender	  = new DiagonalBSGSSimpleSender(cc, pk, numVectors);
		} else {
			cout << "[BSGS-Simple-GPU] GPU ready!" << endl;

			// Memory estimation for Simple BSGS GPU
			size_t numMatrices	= static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));
			size_t numDiagonals = numMatrices * VECTOR_DIM;

			int n1_est			   = static_cast<int>(ceil(sqrt(double(VECTOR_DIM))));
			int giantStepsEst	   = (VECTOR_DIM + n1_est - 1) / n1_est;
			size_t babyKeyCount	   = static_cast<size_t>((n1_est > 1) ? (n1_est - 1) : 0);
			size_t giantKeyCount   = static_cast<size_t>((giantStepsEst > 1) ? (giantStepsEst - 1) : 0);
			size_t evalSumKeyCount = 0;
			for (size_t shift = 1; shift < batchSize; shift *= 2)
				evalSumKeyCount++;
			size_t numRotationKeys = babyKeyCount + giantKeyCount + evalSumKeyCount;

			size_t ringDim	   = cc->GetRingDimension();
			size_t numLimbs	   = cc->GetCryptoParameters()->GetElementParams()->GetParams().size();
			size_t bytesPerCt  = 2 * numLimbs * ringDim * 8 * 2.5;
			size_t bytesPerKey = bytesPerCt * 3;

			double diagMemGB = (numDiagonals * bytesPerCt) / (1024.0 * 1024.0 * 1024.0);
			double keyMemGB	= (numRotationKeys * bytesPerKey) / (1024.0 * 1024.0 * 1024.0);
			// Per-GPU footprint: diagonals are sharded across GPUs, rotation keys
			// are replicated on every GPU.
			size_t numGpusShard	   = gpuDeviceList.size();
			double diagMemPerGpuGB = diagMemGB / static_cast<double>(numGpusShard);
			double totalMemGB	   = diagMemPerGpuGB + keyMemGB;

			cout << "\n[BSGS-Simple-GPU] Memory estimation:" << endl;
			cout << "  Diagonals:      " << numDiagonals << " x ~15MB = " << fixed << setprecision(2) << diagMemGB << " GB"
				 << (numGpusShard > 1 ? ("  (" + to_string(numGpusShard) + " GPUs -> " + to_string(diagMemPerGpuGB) + " GB/GPU)") : "") << endl;
			cout << "  Rotation keys:  " << numRotationKeys << " x ~45MB = " << fixed << setprecision(2) << keyMemGB << " GB" << " (Baby=" << babyKeyCount
				 << ", Giant=" << giantKeyCount << ", EvalSum=" << evalSumKeyCount << ")" << endl;
			cout << "  Total/GPU:      " << fixed << setprecision(2) << totalMemGB << " GB" << endl;
			cout << "  GPU safe limit: " << fixed << setprecision(1) << GPU_MEMORY_SAFE_LIMIT_GB << " GB (" << fixed << setprecision(0)
				 << (GPU_MEMORY_SAFE_LIMIT_GB / GPU_TOTAL_MEMORY_GB * 100) << "% of " << GPU_TOTAL_MEMORY_GB << "GB)" << endl;

			if (totalMemGB > GPU_MEMORY_SAFE_LIMIT_GB) {
				cerr << "\n[BSGS-Simple-GPU] FATAL: Insufficient GPU memory for this "
						"dataset."
					 << endl;
				cerr << "[BSGS-Simple-GPU] Need " << fixed << setprecision(1) << totalMemGB << " GB but only " << GPU_MEMORY_SAFE_LIMIT_GB
					 << " GB safely available." << endl;
				cerr << "[BSGS-Simple-GPU] Options:" << endl;
				cerr << "  1. Use a smaller dataset" << endl;
				cerr << "  2. Use CPU-only approaches (5 or 8)" << endl;
				delete gpuHelper;
				gpuHelper = nullptr;
				return 1;
			} else {
				// Initialize GPU sender
				receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
				sender	 = new DiagonalBSGSSimpleSenderGPU(cc, pk, numVectors, gpuHelper);

				// Load diagonals to GPU
				auto gpuSender = static_cast<DiagonalBSGSSimpleSenderGPU*>(sender);

				// Default: stream diagonals directly from disk→GPU (saves ~15 GB CPU
				// RAM) GPU_STREAM_DIAGS=0: opt-in to bulk-load all diagonals into CPU
				// RAM first
				const char* streamDiagsEnv81 = std::getenv("GPU_STREAM_DIAGS");
				bool useBulkMode81			 = (streamDiagsEnv81 && std::string(streamDiagsEnv81) == "0");

				if (!useBulkMode81) {
					cout << "[BSGS-Simple-GPU] Streaming diagonals disk→GPU (default mode)" << endl;
					auto cacheStart = chrono::steady_clock::now();
					gpuSender->streamDiagonalsFromDisk("serial");
					auto cacheEnd = chrono::steady_clock::now();
					gpuDBCacheTime += chrono::duration<double>(cacheEnd - cacheStart).count();
				} else {
					cout << "[BSGS-Simple-GPU] Loading diagonals to GPU (bulk mode, "
							"GPU_STREAM_DIAGS=0)..."
						 << endl;
					auto cacheStart = chrono::steady_clock::now();
					auto loadStart = chrono::steady_clock::now();

					vector<Ciphertext<DCRTPoly>> preloadedDiagonals;
					preloadedDiagonals.resize(numDiagonals);

#pragma omp parallel for num_threads(MAX_NUM_CORES) schedule(dynamic)
					for (size_t i = 0; i < numDiagonals; ++i) {
						string filepath = "serial/db_diagonal/index" + to_string(i) + ".bin";
						Ciphertext<DCRTPoly> diagonal;
						if (Serial::DeserializeFromFile(filepath, diagonal, SerType::BINARY)) {
							preloadedDiagonals[i] = diagonal;
						} else {
							cerr << "[BSGS-Simple-GPU] Error loading diagonal " << i << endl;
						}
					}

					auto loadEnd	= chrono::steady_clock::now();
					double loadTime = chrono::duration<double>(loadEnd - loadStart).count();
					cout << "[BSGS-Simple-GPU] Loaded " << numDiagonals << " diagonals in " << loadTime << "s" << endl;

					gpuSender->initializeGPUCache(preloadedDiagonals);
						auto cacheEnd = chrono::steady_clock::now();
						gpuDBCacheTime += chrono::duration<double>(cacheEnd - cacheStart).count();
				}

				// Pre-initialize Simple BSGS rotation keys on GPU
				int n1 = static_cast<int>(ceil(sqrt(double(VECTOR_DIM))));
				cout << "[BSGS-Simple-GPU] Pre-initializing Simple BSGS rotation keys "
						"on GPU..."
					 << endl;
				cout << "[BSGS-Simple-GPU] Parameters: n1=" << n1 << " (babyStepSize)" << endl;
				auto keyInitStart = chrono::steady_clock::now();
				gpuHelper->InitializeSimpleBSGSRotationKeysOnGPU(n1, VECTOR_DIM, batchSize);
				auto keyInitEnd	   = chrono::steady_clock::now();
				double keyInitTime = chrono::duration<double>(keyInitEnd - keyInitStart).count();
				gpuKeyUploadTime += keyInitTime;
				cout << "[BSGS-Simple-GPU] Rotation keys pre-initialized in " << fixed << setprecision(2) << keyInitTime << "s" << endl;
			}
		}
#else
		cout << "[BSGS-Simple-GPU] GPU support not compiled. Falling back to "
				"BSGS-Simple (CPU)."
			 << endl;
		receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalBSGSSimpleSender(cc, pk, numVectors);
#endif
		break;
	}

	case ExperimentalApproach::BSGSGPUPreRot: {
#ifdef USE_GPU_ACCELERATION
		// BSGS-GPU-PreRot: Approach 812 — diagonals pre-rotated by enroller
		cout << "[BSGS-GPU-PreRot] Initializing GPU BSGS sender (enroller "
				"pre-rotation)..."
			 << endl;

		KeyPair<DCRTPoly> keys;
		keys.publicKey		   = pk;
		keys.secretKey		   = sk;
		vector<int> gpuDevices = gpuDeviceList;
		auto gpuInitStart	   = chrono::steady_clock::now();
		gpuHelper			   = new GPUFastHESearchHelper(cc, keys, gpuDevices, batchSize);
		auto gpuInitEnd		   = chrono::steady_clock::now();
		gpuKeyUploadTime += chrono::duration<double>(gpuInitEnd - gpuInitStart).count();

		if (!gpuHelper->isReady()) {
			cerr << "[BSGS-GPU-PreRot] GPU initialization failed, falling back to "
					"BSGS-Simple (CPU)"
				 << endl;
			delete gpuHelper;
			gpuHelper = nullptr;
			receiver  = new DiagonalReceiver(cc, pk, sk, numVectors);
			sender	  = new DiagonalBSGSSimpleSender(cc, pk, numVectors);
		} else {
			cout << "[BSGS-GPU-PreRot] GPU ready!" << endl;

			// Memory estimation (same as 81 but NO negative rotation keys)
			size_t numMatrices	= static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));
			size_t numDiagonals = numMatrices * VECTOR_DIM;

			int n1_est			   = static_cast<int>(ceil(sqrt(double(VECTOR_DIM))));
			int giantStepsEst	   = (VECTOR_DIM + n1_est - 1) / n1_est;
			size_t babyKeyCount	   = static_cast<size_t>((n1_est > 1) ? (n1_est - 1) : 0);
			size_t giantKeyCount   = static_cast<size_t>((giantStepsEst > 1) ? (giantStepsEst - 1) : 0);
			size_t evalSumKeyCount = 0;
			for (size_t shift = 1; shift < batchSize; shift *= 2)
				evalSumKeyCount++;
			// NO negative keys — enroller pre-rotated diagonals in plaintext
			size_t numRotationKeys = babyKeyCount + giantKeyCount + evalSumKeyCount;

			size_t ringDim	   = cc->GetRingDimension();
			size_t numLimbs	   = cc->GetCryptoParameters()->GetElementParams()->GetParams().size();
			size_t bytesPerCt  = 2 * numLimbs * ringDim * 8 * 2.5;
			size_t bytesPerKey = bytesPerCt * 3;

			double diagMemGB = (numDiagonals * bytesPerCt) / (1024.0 * 1024.0 * 1024.0);
			double keyMemGB	 = (numRotationKeys * bytesPerKey) / (1024.0 * 1024.0 * 1024.0);
			// Per-GPU footprint: diagonals are sharded across GPUs, rotation keys
			// are replicated on every GPU.
			size_t numGpusShard	   = gpuDeviceList.size();
			double diagMemPerGpuGB = diagMemGB / static_cast<double>(numGpusShard);
			double totalMemGB	   = diagMemPerGpuGB + keyMemGB;

			cout << "\n[BSGS-GPU-PreRot] Memory estimation:" << endl;
			cout << "  Diagonals:      " << numDiagonals << " x ~15MB = " << fixed << setprecision(2) << diagMemGB << " GB"
				 << (numGpusShard > 1 ? ("  (" + to_string(numGpusShard) + " GPUs -> " + to_string(diagMemPerGpuGB) + " GB/GPU)") : "") << endl;
			cout << "  Rotation keys:  " << numRotationKeys << " x ~45MB = " << fixed << setprecision(2) << keyMemGB << " GB" << " (Baby=" << babyKeyCount
				 << ", Giant=+" << giantKeyCount << ", EvalSum=" << evalSumKeyCount << ", NO negative keys)" << endl;
			cout << "  Total/GPU:      " << fixed << setprecision(2) << totalMemGB << " GB" << endl;
			cout << "  GPU safe limit: " << fixed << setprecision(1) << GPU_MEMORY_SAFE_LIMIT_GB << " GB (" << fixed << setprecision(0)
				 << (GPU_MEMORY_SAFE_LIMIT_GB / GPU_TOTAL_MEMORY_GB * 100) << "% of " << GPU_TOTAL_MEMORY_GB << "GB)" << endl;

			if (totalMemGB > GPU_MEMORY_SAFE_LIMIT_GB) {
				cerr << "\n[BSGS-GPU-PreRot] FATAL: Insufficient GPU memory for this "
						"dataset."
					 << endl;
				cerr << "[BSGS-GPU-PreRot] Need " << fixed << setprecision(1) << totalMemGB << " GB but only " << GPU_MEMORY_SAFE_LIMIT_GB
					 << " GB safely available." << endl;
				delete gpuHelper;
				gpuHelper = nullptr;
				return 1;
			} else {
				// Initialize GPU sender (uses CachePreRotatedDiagonalsOnGPU — no
				// rotation)
				receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
				sender	 = new DiagonalBSGSSimpleSenderGPU812(cc, pk, numVectors, gpuHelper);

				auto gpuSender = static_cast<DiagonalBSGSSimpleSenderGPU812*>(sender);

				const char* streamDiagsEnv812 = std::getenv("GPU_STREAM_DIAGS");
				bool useBulkMode812			  = (streamDiagsEnv812 && std::string(streamDiagsEnv812) == "0");

				if (!useBulkMode812) {
					cout << "[BSGS-GPU-PreRot] Streaming pre-rotated diagonals disk→GPU "
							"(default mode)"
						 << endl;
					auto cacheStart = chrono::steady_clock::now();
					gpuSender->streamDiagonalsFromDisk("serial");
					auto cacheEnd = chrono::steady_clock::now();
					gpuDBCacheTime += chrono::duration<double>(cacheEnd - cacheStart).count();
				} else {
					cout << "[BSGS-GPU-PreRot] Loading pre-rotated diagonals to GPU "
							"(bulk mode)..."
						 << endl;
					auto cacheStart = chrono::steady_clock::now();
					auto loadStart = chrono::steady_clock::now();

					vector<Ciphertext<DCRTPoly>> preloadedDiagonals;
					preloadedDiagonals.resize(numDiagonals);

#pragma omp parallel for num_threads(MAX_NUM_CORES) schedule(dynamic)
					for (size_t i = 0; i < numDiagonals; ++i) {
						string filepath = "serial/db_diagonal/index" + to_string(i) + ".bin";
						Ciphertext<DCRTPoly> diagonal;
						if (Serial::DeserializeFromFile(filepath, diagonal, SerType::BINARY)) {
							preloadedDiagonals[i] = diagonal;
						} else {
							cerr << "[BSGS-GPU-PreRot] Error loading diagonal " << i << endl;
						}
					}

					auto loadEnd	= chrono::steady_clock::now();
					double loadTime = chrono::duration<double>(loadEnd - loadStart).count();
					cout << "[BSGS-GPU-PreRot] Loaded " << numDiagonals << " diagonals in " << loadTime << "s" << endl;

					gpuSender->initializeGPUCache(preloadedDiagonals);
						auto cacheEnd = chrono::steady_clock::now();
						gpuDBCacheTime += chrono::duration<double>(cacheEnd - cacheStart).count();
				}

				// Pre-initialize BSGS rotation keys on GPU (same as 81, but no negative
				// keys)
				int n1 = static_cast<int>(ceil(sqrt(double(VECTOR_DIM))));
				cout << "[BSGS-GPU-PreRot] Pre-initializing BSGS rotation keys on GPU..." << endl;
				cout << "[BSGS-GPU-PreRot] Parameters: n1=" << n1 << " (babyStepSize, NO negative keys)" << endl;
				auto keyInitStart = chrono::steady_clock::now();
				gpuHelper->InitializeSimpleBSGSRotationKeysOnGPU(n1, VECTOR_DIM, batchSize);
				auto keyInitEnd	   = chrono::steady_clock::now();
				double keyInitTime = chrono::duration<double>(keyInitEnd - keyInitStart).count();
				gpuKeyUploadTime += keyInitTime;
				cout << "[BSGS-GPU-PreRot] Rotation keys pre-initialized in " << fixed << setprecision(2) << keyInitTime << "s" << endl;
			}
		}
#else
		cout << "[BSGS-GPU-PreRot] GPU support not compiled. Falling back to "
				"BSGS-Simple (CPU)."
			 << endl;
		receiver = new DiagonalReceiver(cc, pk, sk, numVectors);
		sender	 = new DiagonalBSGSSimpleSender(cc, pk, numVectors);
#endif
		break;
	}
	}

	// Structured setup-cost summary for benchmark parsing/reporting.
	if (encDBVectorsTime > 0.0 && dbEncryptTime <= 0.0) {
		dbEncryptTime = encDBVectorsTime;
	}
	double offlineTotalTime = keyGenTime + rotKeyGenTime + dbEncryptTime + diagPreRotTime;
	double setupTotalTime   = offlineTotalTime + gpuKeyUploadTime + gpuDBCacheTime;
	cout << "[SETUP_COSTS] KeyGen=" << fixed << setprecision(4) << keyGenTime << "s"
		 << " RotKeyGen=" << fixed << setprecision(4) << rotKeyGenTime << "s"
		 << " DBEncrypt=" << fixed << setprecision(4) << dbEncryptTime << "s"
		 << " DiagPreRot=" << fixed << setprecision(4) << diagPreRotTime << "s"
		 << " GPUKeyUpload=" << fixed << setprecision(4) << gpuKeyUploadTime << "s"
		 << " GPUDBCache=" << fixed << setprecision(4) << gpuDBCacheTime << "s"
		 << " OfflineTotal=" << fixed << setprecision(4) << offlineTotalTime << "s"
		 << " SetupTotal=" << fixed << setprecision(4) << setupTotalTime << "s" << endl;

	// Track baseline memory after setup
	size_t baselineMemKB = 0;
	size_t prevPeakMemKB = 0;
	baselineMemKB		 = getPeakMemoryKB();
	prevPeakMemKB		 = baselineMemKB;
	cout << "[Baseline] Memory after setup: " << (baselineMemKB / 1024.0) << " MB" << endl;

	// Normalize, batch, and encrypt the query vector
	cout << "[Receiver]\tEncrypting query vector... " << flush;
	start											   = chrono::steady_clock::now();
	vector<Ciphertext<DCRTPoly>> queryCipher		   = receiver->encryptQuery(queryVector);
	vector<Ciphertext<DCRTPoly>> membershipQueryCipher = queryCipher;

	if (expApproach == ExperimentalApproach::BSGSOnlineAggCPU) {
		auto diagonalReceiver = dynamic_cast<DiagonalReceiver*>(receiver);
		if (diagonalReceiver != nullptr) {
			size_t batchSize   = cc->GetEncodingParams()->GetBatchSize();
			size_t numMatrices = static_cast<size_t>(ceil(double(numVectors) / double(batchSize)));
			double maxValue	   = 1.0 + (static_cast<double>(numMatrices) - 1.0) * 2.0 / sqrt(double(VECTOR_DIM));

			vector<double> scaledQuery = VectorUtils::plaintextNormalize(queryVector, VECTOR_DIM);
			for (size_t i = 0; i < VECTOR_DIM; i++) {
				scaledQuery[i] /= maxValue;
			}

			membershipQueryCipher = diagonalReceiver->encryptScaledQuery(scaledQuery);
			cout << "[Approach 9] Membership query scaled by 1/" << maxValue << " for standard depth-" << COMP_DEPTH << " comparison" << endl;
		}
	}
	end			 = chrono::steady_clock::now();
	duration	 = end - start;
	encQueryTime = duration.count();
	cout << "done (" << fixed << setprecision(4) << encQueryTime << "s)" << endl;
	expStream << encQueryTime << "," << flush;		 // report query encryption time
	expStream << queryCipher.size() << "," << flush; // report query communication overhead

#ifdef USE_GPU_ACCELERATION
	// Pre-cache baby-step rotations OUTSIDE membership timing window for fair
	// comparison. Approach 81 (BSGS-Simple-GPU): cache only n1 baby steps.
	if (gpuHelper != nullptr && (expApproach == ExperimentalApproach::BSGSGPU || expApproach == ExperimentalApproach::BSGSGPUPreRot)) {
		int nBaby = static_cast<int>(ceil(sqrt(double(VECTOR_DIM))));
		cout << "[GPUFastHESearchHelper] Caching " << nBaby << " baby steps on GPU... " << flush;
		auto babyStart = chrono::steady_clock::now();
		gpuHelper->ComputeAndCacheBabyStepsOnGPU(queryCipher[0], nBaby);
		auto babyEnd	= chrono::steady_clock::now();
		double babyTime = chrono::duration<double>(babyEnd - babyStart).count();
		cout << "done (" << fixed << setprecision(4) << babyTime << "s)" << endl;
		cout << "[GPUFastHESearchHelper] " << nBaby << " baby steps cached on GPU memory" << endl;
	} else if (gpuHelper != nullptr && expApproach == ExperimentalApproach::FastHESearchGPU) {
		cout << "[GPUFastHESearchHelper] Caching " << VECTOR_DIM << " diagonal rotations on GPU... " << flush;
		auto babyStart = chrono::steady_clock::now();
		gpuHelper->ComputeAndCacheBabyStepsOnGPU(queryCipher[0], static_cast<int>(VECTOR_DIM));
		auto babyEnd	= chrono::steady_clock::now();
		double babyTime = chrono::duration<double>(babyEnd - babyStart).count();
		cout << "done (" << fixed << setprecision(4) << babyTime << "s)" << endl;
		cout << "[GPUFastHESearchHelper] " << VECTOR_DIM << " rotations cached on GPU memory" << endl;
	}
#endif

	// Perform membership scenario
	cout << "[Sender]\tComputing membership scenario... " << flush;
	start								  = chrono::steady_clock::now();
	Ciphertext<DCRTPoly> membershipCipher = sender->membershipScenario(membershipQueryCipher);
	end									  = chrono::steady_clock::now();
	duration							  = end - start;
	membCompTime						  = duration.count();
	cout << "done (wall: " << fixed << setprecision(4) << membCompTime << "s)" << endl;
	expStream << membCompTime << "," << flush; // report membership computation time
	expStream << 1 << "," << flush;			   // report membership communication overhead

	print_mem_usage("[Peak RAM after membership]", prevPeakMemKB);
	prevPeakMemKB = getPeakMemoryKB();

	cout << "[Receiver]\tDecrypting membership results... " << flush;
	start			 = chrono::steady_clock::now();
	membershipResult = receiver->decryptMembership(membershipCipher);
	end				 = chrono::steady_clock::now();
	duration		 = end - start;
	membDecTime		 = duration.count();
	cout << "done (" << fixed << setprecision(4) << membDecTime << "s)" << endl;
	expStream << membDecTime << "," << flush;

	// Perform index scenario (skip for Approach 9 — membership-only approach)
	if (expApproach == ExperimentalApproach::BSGSOnlineAggCPU) {
		cout << "[Sender]\tIndex scenario skipped (Approach 9 is membership-only)" << endl;
		expStream << "0," << flush; // idxCompTime
		expStream << "0," << flush; // communication overhead
		expStream << "0," << flush; // idxDecTime
	} else {
		cout << "[Sender]\tComputing index scenario... " << flush;
		start			 = chrono::steady_clock::now();
		auto indexCipher = sender->indexScenario(queryCipher);
		end				 = chrono::steady_clock::now();
		duration		 = end - start;
		idxCompTime		 = duration.count();
		cout << "done (wall: " << fixed << setprecision(4) << idxCompTime << "s)" << endl;
		expStream << idxCompTime << "," << flush;		 // report index computation time
		expStream << indexCipher.size() << "," << flush; // report index communication overhead

		print_mem_usage("[Peak RAM after index]", prevPeakMemKB);

		cout << "[Receiver]\tDecrypting index results... " << flush;
		start		 = chrono::steady_clock::now();
		indexResults = receiver->decryptIndex(indexCipher);
		end			 = chrono::steady_clock::now();
		duration	 = end - start;
		idxDecTime	 = duration.count();
		cout << "done (" << fixed << setprecision(4) << idxDecTime << "s)" << endl;
		expStream << idxDecTime << "," << flush;
	}

	// Displaying query results
	// The dataset-generation script creates datasets of size N with matches at
	// indices 2 and N-1
	cout << endl << "\tDisplaying Query Results:" << endl;
	cout << "Membership scenario: " << flush;
	if (membershipResult) {
		cout << "true" << endl;
		expStream << "true" << "," << flush;
		resultStatusMembership = "SUCCESS";
	} else {
		cout << "false" << endl;
		expStream << "false" << "," << flush;
		resultStatusMembership = "FAIL";
	}
	cout << "Index scenario: " << flush;
	cout << indexResults << endl;
	expStream << indexResults << "," << flush;

	// Build timing summary string (structured, parseable by benchmark_all.py)
	{
		std::time_t t = std::time(nullptr);
		char timebuf[64];
		std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));

		std::ostringstream timingSummary;
		timingSummary << "[" << timebuf << "] " << "DB=2^" << static_cast<int>(std::log2(numVectors)) << " Approach=" << static_cast<int>(expApproach)
					  << " N=" << VECTOR_DIM << " EncQuery=" << std::fixed << std::setprecision(4) << encQueryTime << "s" << " MembComp=" << std::fixed
					  << std::setprecision(4) << membCompTime << "s" << " MembDec=" << std::fixed << std::setprecision(4) << membDecTime << "s"
					  << " IdxComp=" << std::fixed << std::setprecision(4) << idxCompTime << "s" << " IdxDec=" << std::fixed << std::setprecision(4)
					  << idxDecTime << "s" << " MembResult=" << resultStatusMembership;

		cout << endl << "\t" << timingSummary.str() << endl;
	}

	// Program cleanup
	expStream << endl;
	expStream.close();

#ifdef USE_GPU_ACCELERATION
	if (gpuHelper != nullptr) {
		delete gpuHelper;
		gpuHelper = nullptr;
	}
#endif

	delete receiver;
	delete sender;

	// Clean up serialized data after successful completion (match old repo
	// behavior)
	if (keepSerialAfterRun) {
		cout << "Preserving serialized data in serial/ for reuse" << endl;
	} else {
		try {
			std::uintmax_t numFilesRemoved = fs::remove_all("serial");
			cout << "Cleaned up serialized data (" << numFilesRemoved << " items removed from serial/)" << endl;
		} catch (const fs::filesystem_error& e) {
			cerr << "Warning: Could not remove serial/ folder: " << e.what() << endl;
		}
	}

	cout << endl << "\tProgram successfully terminated" << endl;
	return 0;
}