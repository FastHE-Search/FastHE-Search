<!--
Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
Portions copyright (c) 2026 LG Electronics, Inc.

Licensed under the MIT License (the "License"); you may not use this file
except in compliance with the License.

You may obtain a copy of the License in the LICENSE file at the project
root or at

https://mit-license.org/

SPDX-License-Identifier: MIT
-->

# FastHE-Search: Lightweight, Practical Encrypted Face Recognition with GPU Support

### Artifact – README

FastHE-Search extends the [FastHE-Search](https://github.com/n7koirala/image_matching) framework with a GPU backend featuring optimized CUDA kernels, bringing query latency from many seconds (CPU-only FastHE-Search) to sub-second for certain database sizes. It also introduces Baby-Step Giant-Step (BSGS) matrix-vector products on top of FastHE-Search, reducing VRAM requirements so that the encrypted database can remain resident in GPU memory.

This repository contains the full artifact (code + environment) for our paper:

> G. De Micheli, S. M. Hafiz, G. Pereira, E. L. Cominetti, T. B. Paiva, J. Choi, M. A. Simplicio Jr., and B. Yildiz, "Lightweight, Practical Encrypted Face Recognition with GPU Support," arXiv:2604.00546, 2026.

Follow these steps below to (1) build the Docker image, (2) run the face‑matching approaches, (3) capture their latency/parameter output and (4) generate the figures present in the manuscript.

---

## 1 · Clone the repository

```
git clone --recurse-submodules git@<server url>/FastHE-Search.git
cd FastHE-Search
```

If you already cloned without submodules, initialize `FIDESlib` with:

```bash
git submodule update --init --recursive
```

## 2 · Build the Docker image

If Docker is not installed, please refer to [Docker Installation](https://docs.docker.com/engine/install/) to install Docker for Ubuntu 22.04. Then, proceed with building the docker file:

```
docker build --tag popets2025-hydia .
```

- Installs all system prerequisites on top of Ubuntu 22.04
(build‑essential, cmake, libomp, etc.).

- Fetches and compiles OpenFHE v1.2.3.

- Fetches and installs all the dependendies for generating the figures.

- Compiles the project into ```/opt/image_matching/build```.

## 3 · Run the test & capture output

First create a directory where you want to store the generated figures from the manuscript.

```
mkdir -p ~/artifacts_output
```

Run the command below to execute all five approaches described in the paper and store the result (latency statistics for the **Membership** and **Index** scenarios, FHE parameters used, and basic correctness checks etc.) under ```output.log```. It then generates all the figures present in manuscript under ```~/artifact_output``` .

```
docker run --rm -v ~/artifact_output:/tmp popets2025-hydia | tee output.log
```

All the figures present in the main manuscript can be obtained under ```~/artifact_output/manuscript_figures```.

## Limitations

The code above only reproduce the graphs based on the data over a subset of the the FRGC 2.0 RGB dataset and is located inside ```image_matching/tools/figures```.  The full set of data can be found in ```image_matching/HyDia_full_data.zip```. The data present in ```image_matching/tools/figures``` are obtained using the full set of data.
We have not included the full FRGC 2.0 RGB dataset (including the images), as it was provided by the CVRL lab at the University of Notre Dame and may contain private or proprietary content that cannot be publicly shared.
Therefore, we record our result based on the obtained data (embeddings) and provide the code to generate graphs based on them, as they are the exact ones we present in the paper.

## 4 · Using a larger database (optional) and other features

```run_artifact.sh``` inside the container will use a pre-generated small encrypted database of 2<sup>10</sup> facial‐feature vectors. Optionally, the database size can be easily changed to a higher number (up to 2<sup>20</sup>) after that specific amount of vectors are generated using the ```generate_data.sh``` script located under ```/tool```. For instance to generate 2<sup>15</sup>, run ```generate_data.sh "2_15.dat" $((2**15))```.
Then, edit ```run_artifact.sh``` to update your changes.

## Usage

### Generating Experimental Datasets

To generate an experimental dataset, run the following script from the `build` folder:

```bash
../tools/generate_data.sh [FILENAME] [SIZE]
```

Note that the dataset will be automatically placed in the `test` folder. For example, to generate a dataset with 1024 database vectors located at `/test/2_10.dat`, try:

```bash
../tools/generate_data.sh "2_10.dat" $((2**10))
```

### Building for CPU or GPU

From the repository root, use `./build.sh` to select the build mode:

```bash
./build.sh [cpu|gpu|clean|clean-all|help]
```

- `./build.sh cpu` builds the project in CPU-only mode using the system OpenFHE installation.
- `./build.sh gpu` builds the project with CUDA/FIDESlib support. If FIDESlib or its patched OpenFHE build is missing, the script builds them automatically before compiling FastHE-Search.
- `./build.sh clean` removes the local `build/` contents.
- `./build.sh clean-all` also removes the FIDESlib build artifacts so the next GPU build is a full rebuild.

For a GPU build, make sure the machine has a CUDA-capable NVIDIA GPU, working NVIDIA drivers, and the CUDA toolkit available to CMake.

The preferred GPU setup is:

- clone `FastHE-Search` with `--recurse-submodules`, or
- run `git submodule update --init --recursive` after cloning.

If `FIDESlib` is still missing, `./build.sh gpu` will try these fallback sources automatically:

- existing checkout via `FIDESLIB_DIR=/path/to/fideslib ./build.sh gpu`
- initialized submodule under `FIDESlib/`
- sibling checkout at `../fideslib`
- download from a source archive with `FIDESLIB_ARCHIVE_URL=https://host/fideslib.tar.gz ./build.sh gpu`

The script also tries to detect CUDA architecture automatically with `nvidia-smi` and falls back to `native` if needed. You can always override this manually:

```bash
CMAKE_CUDA_ARCHITECTURES=86 ./build.sh gpu
```

If a user has SSH access configured for the git server, plain `./build.sh gpu` should usually be enough even after a non-recursive clone.

Typical GPU build commands:

```bash
./build.sh clean-all
./build.sh gpu
```

You can optionally override the comparison approximation depth passed to CMake:

```bash
COMP_DEPTH_VAL=8 ./build.sh gpu
```

### Latency Experiments

To run the latency experiments upon the image matching application, navigate to the `build` folder and use the following command in your terminal:

```bash
./ImageMatching ../test/[FILENAME] [APPROACH]
```

The `[FILENAME]` parameter must correspond to an existing file generated by the above scripts.

The `[APPROACH]` parameter determines which algorithm is used to perform the encrypted facial matching upon the provided dataset. The currently available approaches for `ImageMatching` are:

| Parameter | Build Mode | Experimental Approach |
|-----------|------------|-----------------------|
| 1         | CPU / GPU  | Literature baseline |
| 2         | CPU / GPU  | GROTE Paper |
| 3         | CPU / GPU  | Blind-Match paper |
| 4         | CPU / GPU  | HERS paper |
| 5         | CPU / GPU  | HyDia_CPU (novel diagonal transform) |
| 6         | CPU / GPU  | BSGS-Orig (CPU) |
| 7         | CPU / GPU  | BSGS-Precomp (CPU) |
| 8         | CPU / GPU  | BSGS-Precomp-Opt (CPU) |
| 9         | CPU / GPU  | BSGS-OnlineAgg (CPU) |
| 51        | GPU build  | HyDia-GPU |
| 81        | GPU build  | BSGS-GPU |
| 812       | GPU build  | BSGS-GPU-PreRot |

GPU-specific approaches `51`, `81`, and `812` require a successful `./build.sh gpu` build.

For instance, try:

```bash
./ImageMatching ../test/2_10.dat 8
```

This will execute the main application, showcasing both image matching algorithms, more specifically their encryption, matching, and decryption steps.

If you want to run a GPU approach after a GPU build, for example:

```bash
./ImageMatching ../test/2_10.dat 51
```

#### Scaling to larger databases with multiple GPUs

A single GPU can keep the encrypted database "hot" up to roughly `2^16`
vectors (for example, a 46 GiB Quadro RTX 8000). Approaches `81` (BSGS-GPU)
and `812` (BSGS-GPU-PreRot) can scale beyond that limit by **sharding the
database across several GPUs**. Each GPU holds a contiguous shard of the
database, kept resident from the setup phase, and queries run in parallel
(one CPU thread per GPU) with the partial results merged on the host. This
keeps end-to-end latency sub-second while the database grows with the number
of GPUs.

Select the GPUs with the `--gpus` flag. The list is a comma- or space-separated
set of CUDA device IDs. Multi-GPU sharding applies **only** to approaches `81`
and `812`; all other approaches always use device `0`.

Set `FIDESLIB_USE_GRAPH_CAPTURE=0` for multi-GPU runs. FIDESlib's CUDA graph
capture cache is not stable across sharded GPU contexts yet, and disabling it
avoids graph-update failures while keeping the GPU shards hot and parallel.

```bash
# 2^17 database split across 2 GPUs (2 × 2^16 shards)
FIDESLIB_USE_GRAPH_CAPTURE=0 ./ImageMatching ../test/test_131072_k128.dat 81 --gpus 0,1

# 2^18 database split across 4 GPUs (4 × 2^16 shards)
FIDESLIB_USE_GRAPH_CAPTURE=0 ./ImageMatching ../test/test_262144_k128.dat 81 --gpus 0,1,2,3

```

Notes:

- Use **distinct** device IDs (for example `--gpus 0,1`). Repeating an ID
  such as `--gpus 0,0` does not create independent shards.
- The database is split into balanced contiguous shards. The number of
  shards equals the number of GPUs listed, so pick a device count that lets
  each shard fit within a single GPU's memory (about `2^16` vectors per GPU).
- Diagonals are sharded across GPUs while rotation/evaluation keys are
  replicated on each GPU; the startup memory report prints the per-GPU split.
- Query baby-step rotations use a `+1` rotation ladder by default to reduce
  online query-to-GPU cache latency. Set `GPU_BABY_STEP_LADDER=0` to use the
  direct per-offset rotation path for debugging or comparison.
- Set `GPU_SHARD_SEQUENTIAL=1` to process shards one at a time instead of in
  parallel (useful for debugging or memory-constrained runs).

### Automated Benchmark Script

From the repository root, you can build and benchmark the project with:

```bash
./test/benchmark.sh [cpu|gpu|both|current] [options]
```

The benchmark script can:

- rebuild the project in CPU mode (`cpu`) or GPU mode (`gpu`),
- run both builds back-to-back on the same datasets (`both`), or
- reuse the current contents of `build/` without rebuilding (`current`).

With no build mode specified, it builds and benchmarks the GPU configuration.

Defaults used by `./test/benchmark.sh`:

- `gpu` build mode using FIDESlib's patched OpenFHE 1.4.2
- `logn=[10]` → dataset size `2^10 = 1024`
- `kmatch=[16]`
- `t=1`
- `fresh_dataset=false`
- `fixed_keys=false`
- `approaches_cpu=[5,8]`
- `approaches_gpu=[51,81,812]`

Useful examples:

```bash
./test/benchmark.sh
./test/benchmark.sh gpu
./test/benchmark.sh gpu logn=12 kmatch=32 approaches_gpu=[51,81] t=3
./test/benchmark.sh cpu logn=[10,12] kmatch=[16,32] approaches_cpu=[5,8,9]
./test/benchmark.sh both logn=[10,12,14] kmatch=[16,32,32] t=3
./test/benchmark.sh current logn=10 kmatch=16 approaches=[5,8]
./test/benchmark.sh gpu logn=19 kmatch=256 approaches=[81,812] gpus=0,1
```

Additional useful options:

- `comp_depth=N` passes `COMP_DEPTH_VAL=N` to the build.
- `fresh_dataset=true` generates a new dataset for each trial.
- `fixed_keys=true` reuses the same HE key pair across trials.
- `approaches=[...]` applies the same approach list to the active build mode.
- `gpus=0,1,...` selects GPU devices for multi-GPU sharding.
- `BUILD_DIR=/path/to/build` overrides the build directory.
- `TEST_DATA_DIR=/path/to/test` overrides where generated datasets are stored.

Benchmark CSV files are written under `benchmark_results/`.

#### Benchmarking across multiple GPUs

To benchmark larger databases that do not fit on a single GPU, set the
`gpus` parameter when running `./test/benchmark.sh`. The
benchmark harness forwards the parameter info to `ImageMatching`, which shards the
database across the listed GPUs for approaches `81` and `812` (see
[Scaling to larger databases with multiple GPUs](#scaling-to-larger-databases-with-multiple-gpus)).
Use a comma-separated list of CUDA device IDs, set `FIDESLIB_USE_GRAPH_CAPTURE=0`,
and restrict `approaches` to the multi-GPU-capable approaches:

```bash
# 2^17 database sharded across 2 GPUs (2 x 2^16 shards)
FIDESLIB_USE_GRAPH_CAPTURE=0 ./test/benchmark.sh gpu logn=17 kmatch=128 approaches=[81,812] gpus=0,1

# 2^18 database sharded across 4 GPUs (4 x 2^16 shards)
FIDESLIB_USE_GRAPH_CAPTURE=0 ./test/benchmark.sh gpu logn=18 kmatch=128 approaches=[81,812] gpus=0,1,2,3
```

Notes:

- Only approaches `81` and `812` shard across GPUs. Any other approach listed in
  `approaches` still runs on device `0` only,
  so keep the list restricted to `[81,812]` (or `[81]`) for a clean multi-GPU run.
- Choose the GPU count so each shard fits on one GPU (about `2^16` vectors per
  GPU): 2 GPUs for `logn=17`, 4 GPUs for `logn=18`, and so on.
- Use distinct device IDs (`0,1`), not repeats such as `0,0`.
- Keep `FIDESLIB_USE_GRAPH_CAPTURE=0` set for multi-GPU runs until FIDESlib's
  CUDA graph cache is device/context-aware.
- `Baby(s)` is the online encrypted-query upload and baby-step cache time.
  `Member(s)` and `Index(s)` include `Baby(s)` so each scenario is reported as
  a standalone online latency. Set `GPU_BABY_STEP_LADDER=0` to benchmark the
  older direct baby-rotation path.
- Set `GPU_SHARD_SEQUENTIAL=1` to process shards one at a time instead of in
  parallel (useful for debugging or memory-constrained runs), for example
  `FIDESLIB_USE_GRAPH_CAPTURE=0 GPU_SHARD_SEQUENTIAL=1 ./test/benchmark.sh gpu logn=17 kmatch=128 approaches=[81] gpus=0,1`.

### Accuracy Experiments

To run the accuracy experiments upon the image matching application, navigate to the `build` folder and use the following command in your terminal:

```bash
./ImageMatchingAccuracy [SUBJECT_INDEX] [APPROACH]
```

The `[SUBJECT_INDEX]` parameter determines which facial template vector is used as the query vector. The query dataset includes 50 randomly sampled facial template vectors which can be used, therefore this parameter must be an integer in the range 0-49.

The `[APPROACH]` parameter determines which algorithm is used to perform the encrypted facial matching upon the provided dataset. The `ImageMatchingAccuracy` binary currently supports the following CPU-oriented approaches:

| Parameter | Experimental Approach |
|-----------|-----------------------|
| 1         | Literature baseline |
| 2         | GROTE Paper |
| 3         | Blind-Match paper |
| 4         | HERS paper |
| 5         | HyDia_CPU (novel diagonal transform) |
| 6         | BSGS-Orig (CPU) |
| 7         | BSGS-Precomp (CPU) |
| 8         | BSGS-Precomp-Opt (CPU) |

Approaches `9`, `51`, and `81` are not accepted by `ImageMatchingAccuracy`.

For instance, try:

```bash
./ImageMatchingAccuracy 0 5
```

This experiment performs the designated approach upon the FRGC 2.0 dataset, reporting the number of true/false positives and negatives produced by the approach. The experiment also reports the number of true/false positives and negatives produced by the facial feature extractor without any encryption, for purposes of comparison.

## Configuration

### Parameters

The application can be configured using various parameters defined in the source code. Key parameters include:

- **Similarity Match Threshold**: Set the cosine similarity value above which vectors are considered to be matching.
- **Comparison Depth**: Set the multiplicative depth to be used by the comparison-approximating function.
- **Alpha-Norm Depth**: Set the multiplicative depth to be used by the alpha-norm maximum approximation in the group-testing approach.
- **CPU Cores**: Set the maximum number of CPU cores to be allotted to the enroller, receiver, and sender in multi-threaded operations.
- **Security Level**: Configure the security level of the CKKS scheme.
- **Scaling Mod Size**: Configure the size for the scaling modulus of the CKKS scheme.

### Example Configuration

```cpp
// include/config.h
const double MATCH_THRESHOLD = 0.85;
const size_t COMP_DEPTH = 10;
const size_t ALPHA_DEPTH = 2;
const size_t MAX_NUM_CORES = 32;
```

```cpp
// src/main.cpp
CCParams<CryptoContextCKKSRNS> parameters;
parameters.SetSecurityLevel(HEStd_128_classic);
parameters.SetScalingModSize(45);
parameters.SetScalingTechnique(FIXEDMANUAL);
```

## Contributing

We welcome contributions from the community to enhance the functionality and performance of the image matching project. Here’s how you can contribute:

1. **Fork the Repository**: Click on the fork button at the top right of the repository page.
2. **Create a Branch**: Create a new branch for your feature or bugfix.

    ```bash
    git checkout -b feature-name
    ```

3. **Make Changes**: Implement your changes in the new branch.
4. **Submit a Pull Request**: Push your changes to your forked repository and submit a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

:warning:

## Important Warning

This code is designed strictly for academic and research purposes. It has NOT undergone scrutiny by security professionals. No part of this code should be used in any real-world or production setting.
