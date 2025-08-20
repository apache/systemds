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

project(CuJavaCommonJNI LANGUAGES CXX)

find_package(JNI REQUIRED)

add_library(CuJavaCommonJNI STATIC
    cujava_logger.cpp
    cujava_jni_utils.cpp
    cujava_pointer_utils.cpp
)

# PIC because this static lib is linked into shared libs
set_target_properties(CuJavaCommonJNI PROPERTIES
    CXX_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
)

target_include_directories(CuJavaCommonJNI
    PUBLIC
        ${JNI_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}          # headers in common/
)

# Propagate JNI to consumers (runtime, etc.)
target_link_libraries(CuJavaCommonJNI
    PUBLIC
        ${JNI_LIBRARIES}
)

