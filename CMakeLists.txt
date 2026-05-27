# Portions copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah
# Brody, Taeho Jung Portions copyright (c) 2026 LG Electronics, Inc.
#
# Licensed under the MIT License (the "License"); you may not use this file
# except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the project root
# or at
#
# https://mit-license.org/
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

# To use gcc/g++ on a Macintosh, you must set the Compilers here, not inside the
# project if(APPLE) set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
# set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7") endif() TODO: for now, we use
# CLang for Mac
#
# In order to create OpenFHE's static libraries you should enable the
# BUILD_STATIC option. For that, you run "cmake .. -DBUILD_STATIC=ON". After
# having your link completed you will find static libs with the suffix "_static"
# in ./build/libs/. Examples: OPENFHEpke_static.a, OPENFHEcore_static.a, etc.
# After you run "make install" in your build directory, you can build your
# custom application. If you need your application to be linked statically, then
# run "cmake .. -DBUILD_STATIC=ON"

if(NOT USE_GPU)
  project(FastHE-Search CXX)
else()
  project(FastHE-Search CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)
option(USE_GPU "Set to ON to enable GPU acceleration with FIDESlib" OFF)

# ############# CUDA Configuration ###################
if(USE_GPU)
  # Set CUDA_PATH if not defined previously.
  if(NOT DEFINED CUDA_PATH)
    set(CUDA_PATH "/usr/local/cuda")
  endif()

  set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc")

  # Check if CUDA is installed.
  if(NOT EXISTS "${CUDA_PATH}")
    message(FATAL_ERROR "CUDA not found at ${CUDA_PATH}. Please install CUDA.")
  endif()

  # Global language config. set(CMAKE_CUDA_ARCHITECTURES "${FIDESLIB_ARCH}")
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS ON)
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

set(FIDESLIB_ROOT
    "${CMAKE_SOURCE_DIR}/FIDESlib"
    CACHE PATH "Path to the FIDESlib source tree")

if(USE_GPU)
  # GPU enabled: Use FIDESlib's patched OpenFHE 1.2.3 with -O3 -march=native
  message(
    STATUS
      "GPU enabled: Using FIDESlib's patched OpenFHE 1.2.3 with -O3 optimization"
  )

  # FIDESlib's internal OpenFHE installation
  set(OPENFHE_INSTALL_PREFIX "${FIDESLIB_ROOT}/openfhe-install")

  if(EXISTS "${OPENFHE_INSTALL_PREFIX}/lib/OpenFHE")
    # Use FIDESlib's internal OpenFHE (preferred for new builds)
    set(OpenFHE_DIR "${OPENFHE_INSTALL_PREFIX}/lib/OpenFHE")
    set(LOCAL_OPENFHE_DIR "${OPENFHE_INSTALL_PREFIX}/lib")
    set(USE_LOCAL_OPENFHE TRUE)
    message(STATUS "Using FIDESlib's internal OpenFHE at: ${OpenFHE_DIR}")
  else()
    message(
      FATAL_ERROR
        "FIDESlib's OpenFHE installation not found at: ${OPENFHE_INSTALL_PREFIX}\n"
        "Please build FIDESlib with: cd FIDESlib && mkdir build && cd build && cmake -DFIDESLIB_INSTALL_OPENFHE=ON .. && make"
    )
  endif()
else()
  set(USE_LOCAL_OPENFHE FALSE)
endif()

find_package(OpenFHE CONFIG REQUIRED)
if(OpenFHE_FOUND)
  message(STATUS "FOUND PACKAGE OpenFHE")
  message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
  message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
  message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
  message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
  message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
  message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

# Comparison depth (configurable via cmake -DCOMP_DEPTH_VAL=N or env var)
set(COMP_DEPTH_VAL
    "8"
    CACHE STRING "Comparison approximation depth (default 8)")
add_definitions(-DCOMP_DEPTH_VAL=${COMP_DEPTH_VAL})
message(STATUS "COMP_DEPTH_VAL = ${COMP_DEPTH_VAL}")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
# add directories for other OpenFHE modules as needed for your project

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
if(USE_GPU)
  # GPU mode: Don't use global link_libraries - we'll link explicitly per target
  message(STATUS "GPU mode: Using FIDESlib static OpenFHE libraries")
elseif(BUILD_STATIC)
  set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
  link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
  set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
  link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

# GPU acceleration with FIDESlib (optional)
if(USE_GPU)
  message(STATUS "GPU acceleration enabled")

  # Find CUDA
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # FIDESlib paths
  set(FIDESLIB_DIR "${FIDESLIB_ROOT}/build")

  # FIDESlib-OpenFHE static libraries path
  if(DEFINED FIDESLIB_OPENFHE_DIR)
    # Use CMake option (already set)
  elseif(DEFINED ENV{FIDESLIB_OPENFHE_DIR})
    set(FIDESLIB_OPENFHE_DIR "$ENV{FIDESLIB_OPENFHE_DIR}")
  elseif(EXISTS "${FIDESLIB_ROOT}/openfhe-install/lib/libOPENFHEcore_static.a")
    set(FIDESLIB_OPENFHE_DIR "${FIDESLIB_ROOT}/openfhe-install/lib")
  else()
    set(FIDESLIB_OPENFHE_DIR "/usr/local/fideslib-openfhe")
  endif()

  # Validate paths exist
  if(NOT EXISTS "${FIDESLIB_DIR}/fideslib.a")
    message(
      FATAL_ERROR
        "FIDESlib not found at: ${FIDESLIB_DIR}/fideslib.a\n"
        "Please build FIDESlib first:\n"
        "  cd FIDESlib && mkdir build && cd build\n"
        "  cmake -DFIDESLIB_INSTALL_OPENFHE=ON ..\n"
        "  make -j\$(nproc)")
  endif()

  if(NOT EXISTS "${FIDESLIB_OPENFHE_DIR}/libOPENFHEcore_static.a")
    message(
      FATAL_ERROR
        "FIDESlib-OpenFHE static libs not found at: ${FIDESLIB_OPENFHE_DIR}\n"
        "Please rebuild FIDESlib with OpenFHE:\n"
        "  cd FIDESlib && rm -rf build && mkdir build && cd build\n"
        "  cmake -DFIDESLIB_INSTALL_OPENFHE=ON ..\n"
        "  make -j\$(nproc)")
  endif()

  # Include FIDESlib headers
  include_directories(${FIDESLIB_ROOT}/api)
  include_directories(${FIDESLIB_ROOT}/src)
  # #include "openfhe/pke/openfhe.h" in .cuh files)
  include_directories(${OPENFHE_INSTALL_PREFIX}/include)
  include_directories(${CMAKE_SOURCE_DIR}/include)

  # Link directories
  link_directories(${FIDESLIB_DIR})
  link_directories(${FIDESLIB_OPENFHE_DIR})

  # Set GPU source files
  set(GPU_SOURCES
      src/gpu_hydia_helper.cu src/sender/sender_diag_gpu.cpp
      src/sender/sender_diag_bsgs_simple_gpu.cpp
      src/sender/sender_diag_bsgs_simple_gpu_prerot.cpp)

  # Add compile definition
  add_definitions(-DUSE_GPU_ACCELERATION)

  message(STATUS "FIDESlib directory: ${FIDESLIB_DIR}")
  message(STATUS "FIDESlib OpenFHE directory: ${FIDESLIB_OPENFHE_DIR}")
else()
  message(STATUS "GPU acceleration disabled (use -DUSE_GPU=ON to enable)")
  set(GPU_SOURCES "")
endif()

# ADD YOUR EXECUTABLE(s) HERE add_executable( EXECUTABLE-NAME SOURCES )
#
# EXAMPLE: add_executable( test demo-simple-example.cpp )
add_executable(
  ImageMatching
  src/enroller/enroller_base.cpp
  src/enroller/enroller_blind.cpp
  src/enroller/enroller_diag.cpp
  src/enroller/enroller_diag_bsgs_precomp.cpp
  src/enroller/enroller_diag_orig_bsgs.cpp
  src/enroller/enroller_diag_bsgs_precomp_opt.cpp
  src/enroller/enroller_diag_bsgs_prerot.cpp
  src/enroller/enroller_hers.cpp
  src/receiver/receiver.cpp
  src/receiver/receiver_base.cpp
  src/receiver/receiver_blind.cpp
  src/receiver/receiver_diag.cpp
  src/receiver/receiver_diag_bsgs_precomp.cpp
  src/receiver/receiver_diag_orig_bsgs.cpp
  src/receiver/receiver_diag_bsgs_precomp_opt.cpp
  src/receiver/receiver_grote.cpp
  src/receiver/receiver_hers.cpp
  src/sender/sender.cpp
  src/sender/sender_base.cpp
  src/sender/sender_blind.cpp
  src/sender/sender_diag.cpp
  src/sender/sender_diag_bsgs_precomp.cpp
  src/sender/sender_diag_orig_bsgs.cpp
  src/sender/sender_diag_bsgs_precomp_opt.cpp
  src/sender/sender_diag_bsgs_precomp_member_aggr_ct.cpp
  src/sender/sender_diag_bsgs_simple.cpp
  src/sender/sender_grote.cpp
  src/sender/sender_hers.cpp
  src/main.cpp
  src/openFHE_wrapper.cpp
  src/vector_utils.cpp
  ${GPU_SOURCES})

# Link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(ImageMatching PRIVATE OpenMP::OpenMP_CXX)

# Set RPATH based on build mode
if(USE_LOCAL_OPENFHE)
  set_target_properties(
    ImageMatching PROPERTIES BUILD_RPATH "${LOCAL_OPENFHE_DIR}"
                             INSTALL_RPATH "${LOCAL_OPENFHE_DIR}")
endif()

# Link GPU libraries if enabled
if(USE_GPU)
  # Pass OpenMP flag to host compiler for CUDA files
  target_compile_options(
    ImageMatching PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
                          $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=611>)
  target_link_libraries(
    ImageMatching
    PRIVATE ${FIDESLIB_DIR}/fideslib.a
            CUDA::cudart
            cuda
            ${FIDESLIB_OPENFHE_DIR}/libOPENFHEpke_static.a
            ${FIDESLIB_OPENFHE_DIR}/libOPENFHEbinfhe_static.a
            ${FIDESLIB_OPENFHE_DIR}/libOPENFHEcore_static.a
            nccl
            tbb
            pthread
            dl)
  set_target_properties(ImageMatching PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                 CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(
  ImageMatchingAccuracy
  src/enroller/enroller_base.cpp
  src/enroller/enroller_blind.cpp
  src/enroller/enroller_diag.cpp
  src/enroller/enroller_diag_bsgs_precomp.cpp
  src/enroller/enroller_diag_orig_bsgs.cpp
  src/enroller/enroller_diag_bsgs_precomp_opt.cpp
  src/enroller/enroller_diag_bsgs_prerot.cpp
  src/enroller/enroller_hers.cpp
  src/receiver/receiver.cpp
  src/receiver/receiver_base.cpp
  src/receiver/receiver_blind.cpp
  src/receiver/receiver_diag.cpp
  src/receiver/receiver_diag_bsgs_precomp.cpp
  src/receiver/receiver_diag_orig_bsgs.cpp
  src/receiver/receiver_diag_bsgs_precomp_opt.cpp
  src/receiver/receiver_grote.cpp
  src/receiver/receiver_hers.cpp
  src/sender/sender.cpp
  src/sender/sender_base.cpp
  src/sender/sender_blind.cpp
  src/sender/sender_diag.cpp
  src/sender/sender_diag_bsgs_precomp.cpp
  src/sender/sender_diag_orig_bsgs.cpp
  src/sender/sender_diag_bsgs_precomp_opt.cpp
  src/sender/sender_diag_bsgs_precomp_member_aggr_ct.cpp
  src/sender/sender_diag_bsgs_simple.cpp
  src/sender/sender_grote.cpp
  src/sender/sender_hers.cpp
  src/main_accuracy.cpp
  src/openFHE_wrapper.cpp
  src/vector_utils.cpp
  ${GPU_SOURCES})

# Link OpenMP to ImageMatchingAccuracy
target_link_libraries(ImageMatchingAccuracy PRIVATE OpenMP::OpenMP_CXX)

# Set RPATH for ImageMatchingAccuracy
if(USE_LOCAL_OPENFHE)
  set_target_properties(
    ImageMatchingAccuracy PROPERTIES BUILD_RPATH "${LOCAL_OPENFHE_DIR}"
                                     INSTALL_RPATH "${LOCAL_OPENFHE_DIR}")
endif()

# Link GPU libraries for ImageMatchingAccuracy if enabled
if(USE_GPU)
  target_compile_options(
    ImageMatchingAccuracy
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=611>)
  target_link_libraries(
    ImageMatchingAccuracy
    PRIVATE ${FIDESLIB_DIR}/fideslib.a
            CUDA::cudart
            cuda
            ${FIDESLIB_OPENFHE_DIR}/libOPENFHEpke_static.a
            ${FIDESLIB_OPENFHE_DIR}/libOPENFHEbinfhe_static.a
            ${FIDESLIB_OPENFHE_DIR}/libOPENFHEcore_static.a
            nccl
            tbb
            pthread
            dl)
  set_target_properties(
    ImageMatchingAccuracy PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                     CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
