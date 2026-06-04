# ------------------------------------------------------------------
# PETS 2025 Artifact – Image‑Matching (code + figures)
# ------------------------------------------------------------------

#  Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
#  Portions copyright (c) 2026 LG Electronics, Inc
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

# Note that the apt cache is cleaned after every update/install command to ensure
# each layer is as small as possible

FROM ubuntu:24.04

# This simply ensures the variable is defined, and suppresses a warning later
ARG LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive \
    OFHE_VERSION=v1.4.2 \
    OFHE_DIR=/opt/openfhe \
    IMATCH_DIR=/opt/image_matching

# -------------------------------------------------
# 1. Perform a system update and install ca-certificates
# -------------------------------------------------

# At the time of writing, the main Canonical archives were under a major DDoS attack, making their use impossible.
#
# There are two approaches below available. If the image is built with `--build-arg DDOS_STRATEGY=true`, this script
# will work around the attack by
#
#  * Updating the apt sources to use a German mirror which was never under attack
#  * Bootstrapping the initial "apt update" without checknig SSL certificates
#  * Using the same technique to install ca-certificates (not included in the official image)
#
# Otherwise it just uses the normal approach

RUN <<EOF
  set -e
  if [ "$DDOS_STRATEGY" = "true" ]; then
    echo "DDoS Strategy: Switching to FAU mirror and disabling SSL verification for bootstrap..."

    # Update the main archive and security URIs
    sed -i 's|http://archive.ubuntu.com/ubuntu|http://ftp.fau.de/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
    sed -i 's|http://security.ubuntu.com/ubuntu|http://ftp.fau.de/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources

    # Bootstrap ca-certificates (ignoring SSL errors)
    apt-get update -o "Acquire::https::Verify-Peer=false"
    apt-get install -y -o "Acquire::https::Verify-Peer=false" ca-certificates
  else
    echo "Normal Strategy: Using default Canonical archives..."

    # Conventional approach
    apt-get update
    apt-get install -y ca-certificates
  fi

  apt-get clean
  rm -rf /var/lib/apt/lists/*
EOF

# -------------------------------------------------
# 2. Install required packages (python3 + venv + build deps)
# -------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        build-essential git cmake g++ \
        libjsonrpccpp-dev libjsonrpccpp-tools \
        libomp-dev openssl libssl-dev parallel \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# -------------------------------------------------
# 3. NVIDIA Support (cuda toolkit)
# -------------------------------------------------
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install --no-install-recommends -y cuda-toolkit-12-6 libtbb-dev libnccl2 libnccl-dev && \
    rm -rf /usr/local/cuda/doc \
           /usr/local/cuda/samples \
           /usr/local/cuda/bin/nvvp \
           /usr/local/cuda/libnvvp && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV PATH="${CUDA_HOME}/bin:${PATH}"


# -------------------------------------------------
# 4. OpenFHE (pinned to OFHE_VERSION)
# -------------------------------------------------
RUN git clone --depth 1 --branch ${OFHE_VERSION} \
    https://github.com/openfheorg/openfhe-development.git ${OFHE_DIR} && \
    mkdir ${OFHE_DIR}/build && \
    cd ${OFHE_DIR}/build && \
    cmake .. && make -j$(nproc) && make install && \
    ldconfig && \
    cd / && rm -rf ${OFHE_DIR}

# -------------------------------------------------
# 5. Project source (C++ build)
# -------------------------------------------------
WORKDIR /opt
COPY  . ${IMATCH_DIR}

RUN mkdir -p ${IMATCH_DIR}/build && \
    cd ${IMATCH_DIR}/build && \
    cmake .. && \
    make -j$(nproc)

# -------------------------------------------------
# 6. Python virtual environment for figure scripts
# -------------------------------------------------
WORKDIR ${IMATCH_DIR}
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir matplotlib pandas psutil

# Expose venv binaries to every subsequent shell,
#
# Defining this pair of variables  is basically the equivalent of sourcing
# the venv/bin/activate script when running the container.
ENV PATH="${IMATCH_DIR}/venv/bin:${PATH}"
ENV VIRTUAL_ENV="${IMATCH_DIR}/venv"
