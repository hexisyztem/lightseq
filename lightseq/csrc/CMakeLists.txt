cmake_minimum_required(VERSION 3.10)
project(LightseqProtoType LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 87)
find_package(CUDA 11.6 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

option(FP16_MODE "inference with fp16" OFF)
option(DYNAMIC_API "build dynamic lightseq api library" OFF)

# setting compiler flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -Xcompiler -Wall")


if(FP16_MODE)
  add_definitions(-DFP16_MODE)
  message(STATUS "Build using fp16 precision")
else()
  message(STATUS "Build using fp32 precision")
endif()

if(DYNAMIC_API)
  # dynamic link to cuda libraries and protobuf
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
  set(HDF5_USE_STATIC_LIBRARIES OFF)
else()
  # static link to cuda libraries and protobuf
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
  set(HDF5_USE_STATIC_LIBRARIES ON)
endif()

if(DEBUG_MODE)
  if(FP16_MODE)
    add_definitions(-DDEBUG_TYPE=FP16)
  else()
    add_definitions(-DDEBUG_TYPE=FP32)
  endif()
endif()

set(Protobuf_USE_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CUDA_PATH}/include
    kernels/includes
    layers_new/includes
    lsflow/includes
    models/includes
    ops_new/includes
    proto/includes
    tools/includes)

set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

include_directories(${COMMON_HEADER_DIRS})
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/3rdparty/cub)

link_directories(${COMMON_LIB_DIRS})

add_subdirectory(tools)
add_subdirectory(kernels)
add_subdirectory(layers_new)
add_subdirectory(lsflow)
add_subdirectory(models)
add_subdirectory(ops_new)
add_subdirectory(proto)
add_subdirectory(example)
