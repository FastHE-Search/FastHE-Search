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

#include "../../include/enroller_diag.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

// implementation of functions declared in enroller_diag.h

// -------------------- CONSTRUCTOR --------------------

DiagonalEnroller::DiagonalEnroller(CryptoContext<DCRTPoly> ccParam, PublicKey<DCRTPoly> pkParam,
                                   size_t vectorParam)
    : HersEnroller(ccParam, pkParam, vectorParam) {}

// -------------------- PUBLIC FUNCTIONS --------------------
void DiagonalEnroller::serializeDB(vector<vector<double>> &database)
{

  // create necessary directory if does not exist
  string dirName = "serial/";
  if (!filesystem::exists(dirName))
  {
    if (!filesystem::create_directory(dirName))
    {
      cerr << "Error: Failed to create directory \"" + dirName + "\"" << endl;
      return;
    }
  }

  // create matrix-specific directories if they don't exist
  dirName = "serial/db_diagonal";
  if (!filesystem::exists(dirName))
  {
    if (!filesystem::create_directory(dirName))
    {
      cerr << "Error: Failed to create directory \"" + dirName + "\"" << endl;
    }
  }

// normalize all database vectors
#pragma omp parallel for num_threads(MAX_NUM_CORES)
  for (size_t i = 0; i < numVectors; i++)
  {
    database[i] = VectorUtils::plaintextNormalize(database[i], VECTOR_DIM);
  }

  vector<vector<vector<double>>> squareMatrices = splitIntoSquareMatrices(database, VECTOR_DIM);

  vector<vector<vector<double>>> allDiagonalMatrices;
  for (auto &squareMatrix : squareMatrices)
  {
    vector<vector<double>> diagonals = preprocessToDiagonalForm(squareMatrix);
    allDiagonalMatrices.push_back(diagonals);
  }

  vector<vector<double>> concatenatedRows = concatenateRows(allDiagonalMatrices);

  // Producer-consumer pattern: overlap encryption (compute) with serialization (I/O)
  struct WorkItem
  {
    Ciphertext<DCRTPoly> cipher;
    size_t index;
  };

  queue<WorkItem> workQueue;
  mutex queueMutex;
  condition_variable queueCV;
  atomic<bool> encryptionDone(false);
  const size_t maxQueueSize = 32; // Limit memory usage
  const size_t numIOThreads = 6;  // Optimal for disk I/O

  // Consumer threads: serialize encrypted ciphertexts to disk
  vector<thread> ioThreads;
  for (size_t t = 0; t < numIOThreads; t++)
  {
    ioThreads.emplace_back([&]()
                           {
      while (true) {
        WorkItem item;
        {
          unique_lock<mutex> lock(queueMutex);
          queueCV.wait(lock, [&]() { return !workQueue.empty() || encryptionDone.load(); });

          if (workQueue.empty() && encryptionDone.load()) {
            break;
          }

          if (!workQueue.empty()) {
            item = move(workQueue.front());
            workQueue.pop();
            queueCV.notify_all(); // Notify producers that queue has space
          } else {
            continue;
          }
        }

        // Write to disk (outside lock)
        string filepath = "serial/db_diagonal/index" + to_string(item.index) + ".bin";
        if (!Serial::SerializeToFile(filepath, item.cipher, SerType::BINARY)) {
          cerr << "Error: serialization failed (cannot write to " + filepath + ")" << endl;
        }
      } });
  }

// Producer threads: encrypt rows in parallel
#pragma omp parallel for num_threads(MAX_NUM_CORES) schedule(dynamic)
  for (size_t i = 0; i < concatenatedRows.size(); i++)
  {
    // Encrypt (compute-intensive, outside lock)
    Ciphertext<DCRTPoly> encrypted = OpenFHEWrapper::encryptFromVector(cc, pk, concatenatedRows[i]);

    // Add to queue (brief lock)
    {
      unique_lock<mutex> lock(queueMutex);
      queueCV.wait(lock, [&]()
                   { return workQueue.size() < maxQueueSize; });
      workQueue.push({move(encrypted), i});
      queueCV.notify_one(); // Notify consumers
    }
  }

  // Signal completion and wait for all I/O to finish
  encryptionDone.store(true);
  queueCV.notify_all();

  for (auto &t : ioThreads)
  {
    t.join();
  }
}

// -------------------- PROTECTED FUNCTIONS --------------------

vector<vector<vector<double>>> DiagonalEnroller::splitIntoSquareMatrices(vector<vector<double>> &matrix, size_t k)
{
  size_t rows = ceil(double(matrix.size()) / double(k)) * k;
  size_t cols = matrix[0].size();

  vector<vector<vector<double>>> result;

  // Split the matrix into k x k square matrices
  for (size_t i = 0; i < rows; i += k)
  {
    vector<vector<double>> squareMatrix;

    for (size_t j = 0; j < k; ++j)
    {

      if (i + j < matrix.size())
      {
        vector<double> row;
        for (size_t col = 0; col < cols; ++col)
        {
          row.push_back(matrix[i + j][col]);
        }
        squareMatrix.push_back(row);
      }
      else
      {
        vector<double> row(k, 0.0);
        squareMatrix.push_back(row);
      }
    }

    result.push_back(squareMatrix);
  }

  return result;
}

// function to print the matrix
void DiagonalEnroller::printMatrix(vector<vector<double>> matrix)
{
  for (const auto &row : matrix)
  {
    for (double value : row)
    {
      cout << value << " ";
    }
    cout << endl;
  }
}

// pre-process the matrix into diagonal form
vector<vector<double>> DiagonalEnroller::preprocessToDiagonalForm(vector<vector<double>> &matrix)
{
  int K = matrix.size();    // number of rows
  int N = matrix[0].size(); // number of columns

  vector<vector<double>> diagonals(N, vector<double>(K));

  // fill each diagonal vector
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < K; ++j)
    {
      // calculate the column index for diagonal i in row j
      int colIndex = (j + i) % N;
      diagonals[i][j] = matrix[j][colIndex];
    }
  }

  return diagonals;
}

// Function to concatenate rows from multiple matrices
vector<vector<double>> DiagonalEnroller::concatenateRows(const vector<vector<vector<double>>> &matrices)
{

  size_t batchSize = cc->GetEncodingParams()->GetBatchSize();
  size_t matricesPerBatch = batchSize / VECTOR_DIM; // should be 64 with our current params
  size_t numOutputBatches = ceil(double(matrices.size()) / double(matricesPerBatch)) * VECTOR_DIM;

  vector<vector<double>> outputBatches(numOutputBatches);
  vector<double> copyRow;

  size_t matrixNum, rowNum, rowIndex;

  // iterate over all output batches to be filled
  for (size_t i = 0; i < numOutputBatches; i++)
  {
    vector<double> currentBatch(batchSize, 0.0);

    // iterate over the matrices to be concatenated into current batch
    for (size_t j = 0; j < matricesPerBatch; j++)
    {

      matrixNum = ((i / VECTOR_DIM) * matricesPerBatch) + j;
      rowNum = i % VECTOR_DIM;
      rowIndex = j * VECTOR_DIM;

      // check if we still have matrices remaining to concatenate into batch
      // if so, copy i-th row from each matrix into current batch
      if (matrixNum < matrices.size())
      {
        copy(
            matrices[matrixNum][rowNum].begin(),
            matrices[matrixNum][rowNum].end(),
            currentBatch.begin() + rowIndex);
      }
    }

    outputBatches[i] = currentBatch;
  }

  return outputBatches;
}

void DiagonalEnroller::serializeDBThread(vector<double> &currentRow, size_t index)
{
  // This method is kept for compatibility but is no longer used in the optimized path
  Ciphertext<DCRTPoly> currentCtxt = OpenFHEWrapper::encryptFromVector(cc, pk, currentRow);
  string filepath = "serial/db_diagonal/index" + to_string(index) + ".bin";
  if (!Serial::SerializeToFile(filepath, currentCtxt, SerType::BINARY))
  {
    cerr << "Error: serialization failed (cannot write to " + filepath + ")" << endl;
  }
}