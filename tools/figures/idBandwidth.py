#!/usr/bin/env python3

#  Copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data from CSV file
dfMem = pd.read_csv("./tools/figures/15MembershipTotals.csv")
dfId = pd.read_csv("./tools/figures/15IndexTotals.csv")

# Extract columns
network = dfId["Network"].values
bandwidths = ["64Kbps", "2Mbps", "1Gbps", "20Gbps"]

# Define x-ticks with powers of 2 up to 2^20
# x_ticks = [2**i for i in range(10, 21)]  # 2^10 to 2^20

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plotting
ax.plot(network, dfId["Baseline"].values, marker="o", linestyle="-", label="Baseline")
ax.plot(network, dfId["GROTE"].values, marker="s", linestyle="--", label="GROTE")
ax.plot(
    network, dfId["Blind-Match"].values, marker="*", linestyle="-", label="Blind-Match"
)
ax.plot(network, dfId["HERS"].values, marker="^", linestyle="--", label="HERS")
ax.plot(network, dfId["Ours"].values, marker="v", linestyle="-", label="Ours")

# Formatting
# ax.set_xscale('log', base=2)
ax.set_yscale("log")
ax.set_xticklabels(bandwidths)  # Proper exponent notation
ax.set_xlabel("Network Bandwidth", fontsize=18)
ax.set_ylabel("End-To-End Query Time (seconds)", fontsize=18)
ax.set_title(
    "Identification Scenario End-to-End Overhead\nover $2^{15}$ Database Subjects",
    fontsize=18,
)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(fontsize=16)

plt.tick_params(axis="both", labelsize=16)

# Save as PDF
pdf_filename = "/tmp/manuscript_figures/identificationBandwidthLarge.pdf"
fig.savefig(pdf_filename, format="pdf", dpi=300, bbox_inches="tight")

# Show plot
# plt.show()
# pdf_filename
