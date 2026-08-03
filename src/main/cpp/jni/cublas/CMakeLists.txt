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


cmake_minimum_required(VERSION 3.18)

project(CuJavaCublas LANGUAGES CXX)

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)  # provides CUDA::cublas (and CUDA::cudart)

add_library(CuJavaCublas SHARED
    cujava_cublas.cpp
)

set_target_properties(CuJavaCublas PROPERTIES
    CXX_STANDARD 11
    OUTPUT_NAME cujava_cublas                      # -> libcujava_cublas.so
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
)

target_include_directories(CuJavaCublas
    PRIVATE
        ${JNI_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}            # headers in cublas/
        ${CMAKE_CURRENT_SOURCE_DIR}/../common  # if including common headers
)

# Link cuBLAS v2. Add cudart only if you call CUDA runtime APIs in this package.
target_link_libraries(CuJavaCublas
    PRIVATE
        CuJavaCommonJNI
        CUDA::cublas
        CUDA::cudart         # needed for cudaDeviceSynchronize()
        ${JNI_LIBRARIES}
)



