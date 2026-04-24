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
#pragma once

#include <array>
#include <cstddef>

enum class ExperimentalApproach {
    LiteratureBaseline = 1,
    Grote = 2,
    BlindMatch = 3,
    Hers = 4,
    HydiaCPU = 5,
    BSGSOrigCPU = 6,
    BSGSPrecompCPU = 7,
    BSGSPrecompOptCPU = 8,
    BSGSOnlineAggCPU = 9,
    HyDiaGPU = 51,
    BSGSGPU = 81,
    BSGSGPUPreRot = 812,
};

struct ExperimentalApproachMeta {
    ExperimentalApproach id;
    const char* key;
    const char* display;
    const char* csvLabel;
};

inline constexpr std::array<ExperimentalApproachMeta, 12> ExperimentalApproachTable = {{
    {ExperimentalApproach::LiteratureBaseline, "LiteratureBaseline", "Literature baseline", "Baseline"},
    {ExperimentalApproach::Grote, "Grote", "GROTE Paper", "GROTE"},
    {ExperimentalApproach::BlindMatch, "BlindMatch", "Blind-Match paper", "Blind"},
    {ExperimentalApproach::Hers, "Hers", "HERS paper", "HERS"},
    {ExperimentalApproach::HydiaCPU, "HyDia_CPU", "Novel diagonal transform", "Diagonal"},
    {ExperimentalApproach::BSGSOrigCPU, "BSGS-Orig (CPU)", "BSGS-Orig (CPU)", "BSGS-Orig (CPU)"},
    {ExperimentalApproach::BSGSPrecompCPU, "BSGS-Precomp (CPU)", "BSGS-Precomp (CPU)", "BSGS-Precomp (CPU)"},
    {ExperimentalApproach::BSGSPrecompOptCPU, "BSGS-Precomp-Opt (CPU)", "BSGS-Precomp-Opt (CPU)", "BSGS-Precomp-Opt (CPU)"},
    {ExperimentalApproach::BSGSOnlineAggCPU, "BSGS-OnlineAgg (CPU)", "BSGS-OnlineAgg (CPU)", "BSGS-OnlineAgg (CPU)"},
    {ExperimentalApproach::HyDiaGPU, "HyDia-GPU", "HyDia-GPU", "HyDia-GPU"},
    {ExperimentalApproach::BSGSGPU, "BSGS-GPU", "BSGS-GPU", "BSGS-GPU"},
    {ExperimentalApproach::BSGSGPUPreRot, "BSGS-GPU-PreRot", "BSGS-GPU-PreRot", "BSGS-GPU-PreRot"},
}};

constexpr const ExperimentalApproachMeta* get_approach_meta(int value) noexcept {
    for (const auto& meta : ExperimentalApproachTable) {
        if (static_cast<int>(meta.id) == value) {
            return &meta;
        }
    }
    return nullptr;
}

constexpr const ExperimentalApproachMeta* get_approach_meta(ExperimentalApproach approach) noexcept {
    for (const auto& meta : ExperimentalApproachTable) {
        if (meta.id == approach) {
            return &meta;
        }
    }
    return nullptr;
}

constexpr const char* to_string(ExperimentalApproach approach) noexcept {
    const auto* meta = get_approach_meta(approach);
    return (meta != nullptr) ? meta->key : "Unknown";
}

constexpr const char* display_name(ExperimentalApproach approach) noexcept {
    const auto* meta = get_approach_meta(approach);
    return (meta != nullptr) ? meta->display : "Unknown";
}

constexpr const char* csv_label(ExperimentalApproach approach) noexcept {
    const auto* meta = get_approach_meta(approach);
    return (meta != nullptr) ? meta->csvLabel : "Unknown";
}
