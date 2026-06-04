//  Copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
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

// ** Contains the functionalities for loading and processing of plaintext data vectors.

#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace VectorUtils {

void concatenateVectors(vector<double>& dest, vector<double> source, int n);

double plaintextCosineSim(vector<double> x, vector<double> y);

double plaintextMagnitude(vector<double> x, int vectorDim);

vector<double> plaintextNormalize(vector<double> x, int vectorDim);

double plaintextInnerProduct(vector<double> x, vector<double> y, int vectorDim);
} // namespace VectorUtils