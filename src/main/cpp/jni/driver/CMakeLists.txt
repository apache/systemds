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

project(CuJavaDriver LANGUAGES CXX)

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)  # for CUDA::cuda_driver

add_library(CuJavaDriver SHARED
    cujava_driver.cpp
)

set_target_properties(CuJavaDriver PROPERTIES
    CXX_STANDARD 11
    OUTPUT_NAME cujava_driver                         # -> libcujava_driver.so
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
)

target_include_directories(CuJavaDriver
    PRIVATE
        ${JNI_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}            # headers in driver/
        ${CMAKE_CURRENT_SOURCE_DIR}/../common  # if including common headers
)

target_link_libraries(CuJavaRuntime
    PRIVATE
        CuJavaCommonJNI
        CUDA::cuda_driver            # driver API (libcuda)
        ${JNI_LIBRARIES}
)
