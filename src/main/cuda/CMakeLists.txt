#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# need to set this before project() in Linux
if (UNIX)
    set(CMAKE_CUDA_HOST_COMPILER g++-8 CACHE INTERNAL "")
    set(CMAKE_CUDA_COMPILER nvcc CACHE INTERNAL "")
    set(CMAKE_CXX_COMPILER g++ CACHE INTERNAL "")
endif()

project(SystemDS LANGUAGES CXX CUDA)

set(HEADERS agg_ops.cuh  cum_max.cuh  cum_min.cuh  cum_prod.cuh  cum_scan.cuh  cum_sum.cuh  cum_sum_prod.cuh  utils.cuh)
set(SOURCES SystemDS.cu)
add_library(SystemDS OBJECT ${HEADERS} ${SOURCES})

find_package(CUDAToolkit REQUIRED)
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES  "30")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
set_property(TARGET SystemDS PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_property(TARGET SystemDS PROPERTY CUDA_PTX_COMPILATION ON)

# sets the installation path to src/main/cpp/lib
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  #message(STATUS "setting install prefix")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "sets the installation path to src/main/cpp/lib" FORCE)
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH True CACHE INTERNAL "")
install(FILES $<TARGET_OBJECTS:SystemDS> DESTINATION ptx)
