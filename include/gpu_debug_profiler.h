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
/**
 * GPU Debug Profiler for FastHE-Search Approaches 9 and 10
 *
 * This header provides comprehensive debugging and profiling utilities for
 * GPU-accelerated operations. It tracks:
 * - Detailed timing breakdowns for sub-major steps
 * - Exact data transfer sizes (CPU→GPU and GPU→CPU)
 * - Memory allocation patterns
 * - Operation counts and types
 *
 * Usage:
 *   Set GPU_DEBUG_PROFILER_ENABLED to 1 to enable debugging
 *   Set GPU_DEBUG_PROFILER_ENABLED to 0 to disable (zero overhead)
 *
 * Author: GPU FastHE-Search Team
 * Date: January 2026
 */

#ifndef GPU_DEBUG_PROFILER_H
#define GPU_DEBUG_PROFILER_H

//=============================================================================
// MASTER DEBUG SWITCH - Set to 1 to enable, 0 to disable
//=============================================================================
#define GPU_DEBUG_PROFILER_ENABLED 0

//=============================================================================
// Debug sub-categories (only effective when GPU_DEBUG_PROFILER_ENABLED=1)
//=============================================================================
#define GPU_DEBUG_TIMING 1			// Detailed timing for each phase
#define GPU_DEBUG_MEMORY_ALLOC 1	// Memory allocation tracking
#define GPU_DEBUG_TRANSFER 0		// CPU↔GPU data transfer sizes (disabled - too verbose)
#define GPU_DEBUG_OPERATIONS 1		// Operation counts (mult, add, rotate)
#define GPU_DEBUG_CIPHERTEXT_INFO 1 // Ciphertext level, noise, size info
#define GPU_DEBUG_VERBOSE 0			// Extra verbose output (per-iteration)

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if GPU_DEBUG_PROFILER_ENABLED

//=============================================================================
// Data Transfer Tracker
//=============================================================================
struct GPUTransferStats {
	std::atomic<size_t> totalCPUtoGPU_bytes{ 0 };
	std::atomic<size_t> totalGPUtoCPU_bytes{ 0 };
	std::atomic<size_t> numCPUtoGPU_transfers{ 0 };
	std::atomic<size_t> numGPUtoCPU_transfers{ 0 };

	// Per-type tracking
	std::map<std::string, size_t> cpuToGpuByType;
	std::map<std::string, size_t> gpuToCpuByType;
	std::mutex typeMutex;

	void reset() {
		totalCPUtoGPU_bytes	  = 0;
		totalGPUtoCPU_bytes	  = 0;
		numCPUtoGPU_transfers = 0;
		numGPUtoCPU_transfers = 0;
		std::lock_guard<std::mutex> lock(typeMutex);
		cpuToGpuByType.clear();
		gpuToCpuByType.clear();
	}

	void recordCPUtoGPU(size_t bytes, const std::string& dataType) {
		totalCPUtoGPU_bytes += bytes;
		numCPUtoGPU_transfers++;
		std::lock_guard<std::mutex> lock(typeMutex);
		cpuToGpuByType[dataType] += bytes;
	}

	void recordGPUtoCPU(size_t bytes, const std::string& dataType) {
		totalGPUtoCPU_bytes += bytes;
		numGPUtoCPU_transfers++;
		std::lock_guard<std::mutex> lock(typeMutex);
		gpuToCpuByType[dataType] += bytes;
	}

	void printSummary() const {
		std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
		std::cout << "║              GPU DATA TRANSFER SUMMARY                           ║" << std::endl;
		std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
		std::cout << "║ Direction      │ Transfers │ Total Size     │ Avg Size          ║" << std::endl;
		std::cout << "╠════════════════╪═══════════╪════════════════╪═══════════════════╣" << std::endl;

		double cpuToGpuMB  = totalCPUtoGPU_bytes.load() / (1024.0 * 1024.0);
		double gpuToCpuMB  = totalGPUtoCPU_bytes.load() / (1024.0 * 1024.0);
		double avgCpuToGpu = numCPUtoGPU_transfers > 0 ? (totalCPUtoGPU_bytes.load() / (double)numCPUtoGPU_transfers.load()) / 1024.0 : 0;
		double avgGpuToCpu = numGPUtoCPU_transfers > 0 ? (totalGPUtoCPU_bytes.load() / (double)numGPUtoCPU_transfers.load()) / 1024.0 : 0;

		std::cout << std::fixed << std::setprecision(2);
		std::cout << "║ CPU → GPU      │ " << std::setw(9) << numCPUtoGPU_transfers.load() << " │ " << std::setw(10) << cpuToGpuMB << " MB │ " << std::setw(10)
				  << avgCpuToGpu << " KB     ║" << std::endl;
		std::cout << "║ GPU → CPU      │ " << std::setw(9) << numGPUtoCPU_transfers.load() << " │ " << std::setw(10) << gpuToCpuMB << " MB │ " << std::setw(10)
				  << avgGpuToCpu << " KB     ║" << std::endl;
		std::cout << "╠════════════════╧═══════════╧════════════════╧═══════════════════╣" << std::endl;
		std::cout << "║ Transfer Breakdown by Data Type:                                 ║" << std::endl;
		std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;

		// Note: We can't lock in const method, so this is approximate
		std::cout << "║ (See detailed breakdown in verbose logs)                         ║" << std::endl;
		std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;
	}
};

//=============================================================================
// Operation Counter
//=============================================================================
struct GPUOperationStats {
	std::atomic<size_t> numMultiplications{ 0 };
	std::atomic<size_t> numAdditions{ 0 };
	std::atomic<size_t> numRotations{ 0 };
	std::atomic<size_t> numRescales{ 0 };
	std::atomic<size_t> numSquares{ 0 };
	std::atomic<size_t> numKeySwitches{ 0 };

	void reset() {
		numMultiplications = 0;
		numAdditions	   = 0;
		numRotations	   = 0;
		numRescales		   = 0;
		numSquares		   = 0;
		numKeySwitches	   = 0;
	}

	void printSummary() const {
		std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
		std::cout << "║              GPU OPERATION COUNTS                                ║" << std::endl;
		std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
		std::cout << "║ Operation       │ Count                                          ║" << std::endl;
		std::cout << "╠═════════════════╪════════════════════════════════════════════════╣" << std::endl;
		std::cout << "║ Multiplications │ " << std::setw(44) << numMultiplications.load() << " ║" << std::endl;
		std::cout << "║ Additions       │ " << std::setw(44) << numAdditions.load() << " ║" << std::endl;
		std::cout << "║ Rotations       │ " << std::setw(44) << numRotations.load() << " ║" << std::endl;
		std::cout << "║ Rescales        │ " << std::setw(44) << numRescales.load() << " ║" << std::endl;
		std::cout << "║ Squares         │ " << std::setw(44) << numSquares.load() << " ║" << std::endl;
		std::cout << "║ Key Switches    │ " << std::setw(44) << numKeySwitches.load() << " ║" << std::endl;
		std::cout << "╚═════════════════╧════════════════════════════════════════════════╝" << std::endl;
	}
};

//=============================================================================
// Timing Profiler
//=============================================================================
class GPUTimingProfiler {
  public:
	struct TimingEntry {
		std::string name;
		double duration_ms;
		size_t count;
	};

  private:
	std::vector<TimingEntry> entries_;
	std::chrono::high_resolution_clock::time_point sectionStart_;
	std::string currentSection_;
	std::mutex mutex_;

  public:
	void reset() {
		std::lock_guard<std::mutex> lock(mutex_);
		entries_.clear();
	}

	void startSection(const std::string& name) {
		std::lock_guard<std::mutex> lock(mutex_);
		currentSection_ = name;
		sectionStart_	= std::chrono::high_resolution_clock::now();
	}

	void endSection() {
		auto end		= std::chrono::high_resolution_clock::now();
		double duration = std::chrono::duration<double, std::milli>(end - sectionStart_).count();

		std::lock_guard<std::mutex> lock(mutex_);
		// Find or create entry
		bool found = false;
		for (auto& entry : entries_) {
			if (entry.name == currentSection_) {
				entry.duration_ms += duration;
				entry.count++;
				found = true;
				break;
			}
		}
		if (!found) {
			entries_.push_back({ currentSection_, duration, 1 });
		}
	}

	void recordDuration(const std::string& name, double duration_ms) {
		std::lock_guard<std::mutex> lock(mutex_);
		bool found = false;
		for (auto& entry : entries_) {
			if (entry.name == name) {
				entry.duration_ms += duration_ms;
				entry.count++;
				found = true;
				break;
			}
		}
		if (!found) {
			entries_.push_back({ name, duration_ms, 1 });
		}
	}

	void printSummary() const {
		std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
		std::cout << "║              GPU TIMING BREAKDOWN                                ║" << std::endl;
		std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
		std::cout << "║ Section                        │ Time (ms)   │ Calls │ Avg (ms) ║" << std::endl;
		std::cout << "╠════════════════════════════════╪═════════════╪═══════╪══════════╣" << std::endl;

		double total = 0;
		for (const auto& entry : entries_) {
			double avg = entry.count > 0 ? entry.duration_ms / entry.count : 0;
			std::cout << std::fixed << std::setprecision(2);
			std::cout << "║ " << std::left << std::setw(30) << entry.name.substr(0, 30) << " │ " << std::right << std::setw(11) << entry.duration_ms << " │ "
					  << std::setw(5) << entry.count << " │ " << std::setw(8) << avg << " ║" << std::endl;
			total += entry.duration_ms;
		}

		std::cout << "╠════════════════════════════════╪═════════════╪═══════╪══════════╣" << std::endl;
		std::cout << "║ TOTAL                          │ " << std::setw(11) << total << " │       │          ║" << std::endl;
		std::cout << "╚════════════════════════════════╧═════════════╧═══════╧══════════╝" << std::endl;
	}
};

//=============================================================================
// Global Profiler Instance
//=============================================================================
inline GPUTransferStats& getTransferStats() {
	static GPUTransferStats instance;
	return instance;
}

inline GPUOperationStats& getOperationStats() {
	static GPUOperationStats instance;
	return instance;
}

inline GPUTimingProfiler& getTimingProfiler() {
	static GPUTimingProfiler instance;
	return instance;
}

//=============================================================================
// Helper Macros for Easy Use
//=============================================================================

// Timing macros
#define GPU_PROFILE_START(name) getTimingProfiler().startSection(name)
#define GPU_PROFILE_END(name) getTimingProfiler().endSection()
#define GPU_PROFILE_RECORD(name, duration_ms) getTimingProfiler().recordDuration(name, duration_ms)

// Transfer tracking macros (3-argument version with description)
#define GPU_TRACK_CPU_TO_GPU(bytes, type, desc)                                                                                                          \
	do {                                                                                                                                                 \
		getTransferStats().recordCPUtoGPU(bytes, type);                                                                                                  \
		if (GPU_DEBUG_TRANSFER) {                                                                                                                        \
			std::cout << "[GPU_TRANSFER] CPU→GPU: " << (bytes) << " bytes (" << ((bytes) / 1024.0) << " KB) [" << (type) << "] " << (desc) << std::endl; \
		}                                                                                                                                                \
	} while (0)

#define GPU_TRACK_GPU_TO_CPU(bytes, type, desc)                                                                                                          \
	do {                                                                                                                                                 \
		getTransferStats().recordGPUtoCPU(bytes, type);                                                                                                  \
		if (GPU_DEBUG_TRANSFER) {                                                                                                                        \
			std::cout << "[GPU_TRANSFER] GPU→CPU: " << (bytes) << " bytes (" << ((bytes) / 1024.0) << " KB) [" << (type) << "] " << (desc) << std::endl; \
		}                                                                                                                                                \
	} while (0)

// Operation counting macros
#define GPU_COUNT_MULT() getOperationStats().numMultiplications++
#define GPU_COUNT_ADD() getOperationStats().numAdditions++
#define GPU_COUNT_ROTATE() getOperationStats().numRotations++
#define GPU_COUNT_RESCALE() getOperationStats().numRescales++
#define GPU_COUNT_SQUARE() getOperationStats().numSquares++
#define GPU_COUNT_KEYSWITCH() getOperationStats().numKeySwitches++

// Debug print macros
#define GPU_DEBUG_PRINT(msg)                             \
	do {                                                 \
		std::cout << "[GPU_DEBUG] " << msg << std::endl; \
	} while (0)

#define GPU_DEBUG_PRINT_VERBOSE(msg)                           \
	do {                                                       \
		if (GPU_DEBUG_VERBOSE) {                               \
			std::cout << "[GPU_DEBUG_V] " << msg << std::endl; \
		}                                                      \
	} while (0)

// Ciphertext info macro
#define GPU_DEBUG_CT_INFO(name, N, L, level)                                                                                                                         \
	do {                                                                                                                                                             \
		if (GPU_DEBUG_CIPHERTEXT_INFO) {                                                                                                                             \
			size_t ctSize = 2 * (L) * (N) * 8;                                                                                                                       \
			std::cout << "[GPU_CT_INFO] " << (name) << ": N=" << (N) << " L=" << (L) << " level=" << (level) << " size=" << (ctSize / 1024.0) << " KB" << std::endl; \
		}                                                                                                                                                            \
	} while (0)

// Memory tracking macro
#define GPU_DEBUG_MEMORY(msg, bytes)                                                                                                      \
	do {                                                                                                                                  \
		if (GPU_DEBUG_MEMORY_ALLOC) {                                                                                                     \
			std::cout << "[GPU_MEMORY] " << msg << ": " << (bytes) << " bytes (" << ((bytes) / (1024.0 * 1024.0)) << " MB)" << std::endl; \
		}                                                                                                                                 \
	} while (0)

// Reset all stats
#define GPU_PROFILER_RESET()         \
	do {                             \
		getTransferStats().reset();  \
		getOperationStats().reset(); \
		getTimingProfiler().reset(); \
	} while (0)

// Print all summaries
#define GPU_PROFILER_PRINT_SUMMARY()        \
	do {                                    \
		getTimingProfiler().printSummary(); \
		getOperationStats().printSummary(); \
		getTransferStats().printSummary();  \
	} while (0)

// Estimate ciphertext size (for transfer tracking)
#define GPU_ESTIMATE_CT_SIZE(N, L) (2 * (L) * (N) * 8)

//=============================================================================
// Scoped Timer Helper
//=============================================================================
class GPUScopedTimer {
	std::string name_;
	std::chrono::high_resolution_clock::time_point start_;

  public:
	GPUScopedTimer(const std::string& name) : name_(name) {
		start_ = std::chrono::high_resolution_clock::now();
		GPU_DEBUG_PRINT(">>> Starting: " + name);
	}

	~GPUScopedTimer() {
		auto end		= std::chrono::high_resolution_clock::now();
		double duration = std::chrono::duration<double, std::milli>(end - start_).count();
		getTimingProfiler().recordDuration(name_, duration);
		std::cout << "[GPU_TIMER] <<< " << name_ << ": " << std::fixed << std::setprecision(2) << duration << " ms" << std::endl;
	}
};

#define GPU_SCOPED_TIMER(name) GPUScopedTimer _gpu_timer_##__LINE__(name)

#else // GPU_DEBUG_PROFILER_ENABLED == 0

//=============================================================================
// Disabled versions (zero overhead)
// Macros must accept same arguments as enabled versions but do nothing
//=============================================================================

#define GPU_PROFILE_START(name) ((void)0)
#define GPU_PROFILE_END(name) ((void)0)
#define GPU_PROFILE_RECORD(name, duration_ms) ((void)0)

#define GPU_TRACK_CPU_TO_GPU(bytes, type, desc) ((void)0)
#define GPU_TRACK_GPU_TO_CPU(bytes, type, desc) ((void)0)

#define GPU_COUNT_MULT() ((void)0)
#define GPU_COUNT_ADD() ((void)0)
#define GPU_COUNT_ROTATE() ((void)0)
#define GPU_COUNT_RESCALE() ((void)0)
#define GPU_COUNT_SQUARE() ((void)0)
#define GPU_COUNT_KEYSWITCH() ((void)0)

#define GPU_DEBUG_PRINT(msg) ((void)0)
#define GPU_DEBUG_PRINT_VERBOSE(msg) ((void)0)
#define GPU_DEBUG_CT_INFO(name, N, L, level) ((void)0)
#define GPU_DEBUG_MEMORY(msg, bytes) ((void)0)

#define GPU_PROFILER_RESET() ((void)0)
#define GPU_PROFILER_PRINT_SUMMARY() ((void)0)

#define GPU_ESTIMATE_CT_SIZE(N, L) (2 * (L) * (N) * 8)

#define GPU_SCOPED_TIMER(name) ((void)0)

#endif // GPU_DEBUG_PROFILER_ENABLED

#endif // GPU_DEBUG_PROFILER_H
