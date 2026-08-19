#!/usr/bin/env python3

#  Copyright (c) 2026 LG Electronics, Inc.
#
#  Licensed under the MIT License (the "License"); you may not use this file
#  except in compliance with the License.
#
#  You may obtain a copy of the License in the LICENSE file at the project
#  root or at
#
#  https://mit-license.org/
#
#  SPDX-License-Identifier: MIT

"""
Generate comparison plots: BSGS_CPU vs FastHE-Search_CPU
Based on aggregated benchmark data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark_cpu_t10_20260127_164220_aggregated_summary.csv
# Extracting mean values (ignoring ± std for plotting)

data = {
    "BSGS_CPU": {
        "dataset_size": [
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
        ],
        "membership_time": [
            8.14,
            8.34,
            8.32,
            8.23,
            8.28,
            10.86,
            15.65,
            25.61,
            43.83,
            82.02,
            163.24,
        ],
        "index_time": [
            8.14,
            8.23,
            8.18,
            8.34,
            8.04,
            10.83,
            15.89,
            25.61,
            42.95,
            79.97,
            153.22,
        ],
        "peak_ram_gb": [
            11.4,
            11.3,
            11.3,
            11.4,
            11.3,
            11.5,
            11.5,
            12.1,
            13.1,
            16.5,
            21.2,
        ],
        "peak_disk_gb": [6.1, 6.1, 6.1, 6.1, 6.1, 9.1, 15.1, 27.1, 51.1, 99.2, 195.2],
    },
    "FastHE-Search_CPU": {
        "dataset_size": [
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
        ],
        "membership_time": [
            7.30,
            7.22,
            7.24,
            7.29,
            7.16,
            9.68,
            13.88,
            22.93,
            41.31,
            90.01,
            178.17,
        ],
        "index_time": [
            6.90,
            6.78,
            6.79,
            6.85,
            6.92,
            9.02,
            13.63,
            23.10,
            40.46,
            80.89,
            162.43,
        ],
        "peak_ram_gb": [
            22.5,
            22.5,
            22.4,
            22.6,
            22.4,
            22.7,
            22.8,
            23.3,
            24.2,
            26.6,
            31.9,
        ],
        "peak_disk_gb": [
            27.8,
            27.8,
            27.8,
            27.8,
            27.8,
            30.8,
            36.8,
            48.9,
            72.9,
            120.9,
            216.9,
        ],
    },
}

# Calculate log2 of dataset sizes for x-axis labels
dataset_sizes = data["BSGS_CPU"]["dataset_size"]
log2_sizes = [int(np.log2(s)) for s in dataset_sizes]

# Create figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BSGS_CPU vs FastHE-Search_CPU Comparison", fontsize=14, fontweight="bold")

# Colors for the approaches
colors = {"BSGS_CPU": "#2196F3", "FastHE-Search_CPU": "#FF5722"}  # Blue  # Orange
markers = {"BSGS_CPU": "o", "FastHE-Search_CPU": "s"}

# Plot 1: Membership Time vs Dataset Size
ax1 = axes[0, 0]
for approach in ["BSGS_CPU", "FastHE-Search_CPU"]:
    y_data = data[approach]["membership_time"]
    ax1.plot(
        log2_sizes,
        y_data,
        marker=markers[approach],
        color=colors[approach],
        linewidth=2,
        markersize=8,
        label=approach,
    )
    # Add data labels
    offset = 5 if approach == "FastHE-Search_CPU" else -12
    for x, y in zip(log2_sizes, y_data):
        ax1.annotate(
            f"{int(round(y))}",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=7,
            color=colors[approach],
        )
# Add arrow at max difference
y1 = np.array(data["BSGS_CPU"]["membership_time"])
y2 = np.array(data["FastHE-Search_CPU"]["membership_time"])
diffs = np.abs(y1 - y2)
max_idx = np.argmax(diffs)
x_max = log2_sizes[max_idx]
y_low, y_high = min(y1[max_idx], y2[max_idx]), max(y1[max_idx], y2[max_idx])
ax1.annotate(
    "",
    xy=(x_max, y_high),
    xytext=(x_max, y_low),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax1.text(
    x_max + 0.3,
    (y_low + y_high) / 2,
    f"Δ={int(round(diffs[max_idx]))}",
    fontsize=9,
    fontweight="bold",
    va="center",
)
ax1.set_xlabel("Dataset Size (log₂)", fontsize=11)
ax1.set_ylabel("Membership Time (s)", fontsize=11)
ax1.set_title("Membership Time vs Dataset Size", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xticks(log2_sizes)
ax1.set_xticklabels([f"2^{{{x}}}" for x in log2_sizes], fontsize=9)

# Plot 2: Index Time vs Dataset Size
ax2 = axes[0, 1]
for approach in ["BSGS_CPU", "FastHE-Search_CPU"]:
    y_data = data[approach]["index_time"]
    ax2.plot(
        log2_sizes,
        y_data,
        marker=markers[approach],
        color=colors[approach],
        linewidth=2,
        markersize=8,
        label=approach,
    )
    # Add data labels
    offset = 5 if approach == "FastHE-Search_CPU" else -12
    for x, y in zip(log2_sizes, y_data):
        ax2.annotate(
            f"{int(round(y))}",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=7,
            color=colors[approach],
        )
# Add arrow at max difference
y1 = np.array(data["BSGS_CPU"]["index_time"])
y2 = np.array(data["FastHE-Search_CPU"]["index_time"])
diffs = np.abs(y1 - y2)
max_idx = np.argmax(diffs)
x_max = log2_sizes[max_idx]
y_low, y_high = min(y1[max_idx], y2[max_idx]), max(y1[max_idx], y2[max_idx])
ax2.annotate(
    "",
    xy=(x_max, y_high),
    xytext=(x_max, y_low),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax2.text(
    x_max + 0.3,
    (y_low + y_high) / 2,
    f"Δ={int(round(diffs[max_idx]))}",
    fontsize=9,
    fontweight="bold",
    va="center",
)
ax2.set_xlabel("Dataset Size (log₂)", fontsize=11)
ax2.set_ylabel("Index Time (s)", fontsize=11)
ax2.set_title("Index Time vs Dataset Size", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(log2_sizes)
ax2.set_xticklabels([f"2^{{{x}}}" for x in log2_sizes], fontsize=9)

# Plot 3: Peak RAM vs Dataset Size
ax3 = axes[1, 0]
for approach in ["BSGS_CPU", "FastHE-Search_CPU"]:
    y_data = data[approach]["peak_ram_gb"]
    ax3.plot(
        log2_sizes,
        y_data,
        marker=markers[approach],
        color=colors[approach],
        linewidth=2,
        markersize=8,
        label=approach,
    )
    # Add data labels
    offset = 5 if approach == "FastHE-Search_CPU" else -12
    for x, y in zip(log2_sizes, y_data):
        ax3.annotate(
            f"{int(round(y))}",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=7,
            color=colors[approach],
        )
# Add arrow at max difference
y1 = np.array(data["BSGS_CPU"]["peak_ram_gb"])
y2 = np.array(data["FastHE-Search_CPU"]["peak_ram_gb"])
diffs = np.abs(y1 - y2)
max_idx = np.argmax(diffs)
x_max = log2_sizes[max_idx]
y_low, y_high = min(y1[max_idx], y2[max_idx]), max(y1[max_idx], y2[max_idx])
ax3.annotate(
    "",
    xy=(x_max, y_high),
    xytext=(x_max, y_low),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax3.text(
    x_max + 0.3,
    (y_low + y_high) / 2,
    f"Δ={int(round(diffs[max_idx]))}",
    fontsize=9,
    fontweight="bold",
    va="center",
)
ax3.set_xlabel("Dataset Size (log₂)", fontsize=11)
ax3.set_ylabel("Peak RAM (GiB)", fontsize=11)
ax3.set_title("Peak RAM vs Dataset Size", fontsize=12)
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)
ax3.set_xticks(log2_sizes)
ax3.set_xticklabels([f"2^{{{x}}}" for x in log2_sizes], fontsize=9)

# Plot 4: Peak Disk vs Dataset Size
ax4 = axes[1, 1]
for approach in ["BSGS_CPU", "FastHE-Search_CPU"]:
    y_data = data[approach]["peak_disk_gb"]
    ax4.plot(
        log2_sizes,
        y_data,
        marker=markers[approach],
        color=colors[approach],
        linewidth=2,
        markersize=8,
        label=approach,
    )
    # Add data labels
    offset = 5 if approach == "FastHE-Search_CPU" else -12
    for x, y in zip(log2_sizes, y_data):
        ax4.annotate(
            f"{int(round(y))}",
            (x, y),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=7,
            color=colors[approach],
        )
# Add arrow at max difference
y1 = np.array(data["BSGS_CPU"]["peak_disk_gb"])
y2 = np.array(data["FastHE-Search_CPU"]["peak_disk_gb"])
diffs = np.abs(y1 - y2)
max_idx = np.argmax(diffs)
x_max = log2_sizes[max_idx]
y_low, y_high = min(y1[max_idx], y2[max_idx]), max(y1[max_idx], y2[max_idx])
ax4.annotate(
    "",
    xy=(x_max, y_high),
    xytext=(x_max, y_low),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax4.text(
    x_max + 0.3,
    (y_low + y_high) / 2,
    f"Δ={int(round(diffs[max_idx]))}",
    fontsize=9,
    fontweight="bold",
    va="center",
)
ax4.set_xlabel("Dataset Size (log₂)", fontsize=11)
ax4.set_ylabel("Peak Disk (GiB)", fontsize=11)
ax4.set_title("Peak Disk vs Dataset Size", fontsize=12)
ax4.legend(loc="upper left")
ax4.grid(True, alpha=0.3)
ax4.set_xticks(log2_sizes)
ax4.set_xticklabels([f"2^{{{x}}}" for x in log2_sizes], fontsize=9)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = (
    "/space/gpereira/image_matching/tools/figures/bsgs_vs_fasthe_search_comparison.png"
)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"✅ Figure saved to: {output_path}")

# Also save as PDF for publication quality
pdf_path = "/space/gpereira/image_matching/tools/figures/bsgs_vs_fasthe_search_comparison.pdf"
plt.savefig(pdf_path, bbox_inches="tight")
print(f"✅ PDF saved to: {pdf_path}")

plt.show()
