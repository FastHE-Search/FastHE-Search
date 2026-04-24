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

// ** Holds configuration parameters like file paths, default values, and any
// other constant values

// ** Wrapper functions for the OpenFHE library

#pragma once

#include "config.h"
#include "openfhe.h"

using namespace std;
using namespace lbcrypto;

namespace OpenFHEWrapper
{

    size_t
    computeRequiredDepth(size_t approach);

    void
    printSchemeDetails(CCParams<CryptoContextCKKSRNS> parameters, CryptoContext<DCRTPoly> cc);

    void
    printCipherDetails(Ciphertext<DCRTPoly> ctxt);

    Ciphertext<DCRTPoly>
    encryptFromVector(CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> pk, vector<double> vec);

    vector<double>
    decryptToVector(CryptoContext<DCRTPoly> cc, PrivateKey<DCRTPoly> sk, Ciphertext<DCRTPoly> ctxt);

    vector<double>
    decryptVectorToVector(CryptoContext<DCRTPoly> cc, PrivateKey<DCRTPoly> sk, vector<Ciphertext<DCRTPoly>> ctxt);

    Ciphertext<DCRTPoly>
    binaryRotate(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ctxt, int factor);

    Ciphertext<DCRTPoly>
    sign(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ctxt, size_t maxDepth);

    Ciphertext<DCRTPoly>
    sumAllSlots(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ctxt);

    Ciphertext<DCRTPoly>
    chebyshevCompare(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ctxt, double delta, size_t signDepth);

    // Overload with custom Chebyshev approximation range [lowerBound, upperBound].
    // Skips the f4 sharpening polynomial, instead allocating ALL signDepth levels
    // to a single high-degree Chebyshev approximation (degree up to 495 at signDepth=10).
    // This is correct because the higher polynomial degree provides sufficient step-function
    // accuracy without needing the additional f4 sharpening pass.
    // Use when input values may exceed [-1, 1], e.g. after summing similarities across matrices.
    Ciphertext<DCRTPoly>
    chebyshevCompare(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ctxt, double delta, size_t signDepth,
                     double lowerBound, double upperBound);

    vector<Ciphertext<DCRTPoly>>
    mergeCiphers(CryptoContext<DCRTPoly> cc, vector<Ciphertext<DCRTPoly>> &ctxts, size_t dimension);

    Ciphertext<DCRTPoly>
    mergeSingleCipher(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> &ctxt, size_t dimension);

    Plaintext
    generateMergeMask(CryptoContext<DCRTPoly> cc, size_t dimension, size_t segmentLength);

    vector<Ciphertext<DCRTPoly>>
    compressCiphers(CryptoContext<DCRTPoly> cc, vector<Ciphertext<DCRTPoly>> &ctxts, size_t dimension);
}