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


if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX lib)
    set(CMAKE_IMPORT_LIBRARY_PREFIX lib)
else()
    set(CMAKE_CUDA_HOST_COMPILER g++-8)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
endif ()

set(CMAKE_BUILD_TYPE Release)

project (systemds_spoof_native LANGUAGES CUDA CXX)

set(HEADERS spoof_native_jni.h spoof_native_cuda.h)
set(SOURCES spoof_native_jni.cpp spoof_native_cuda.cpp)

# Build a shared libraray
add_library(spoof_native_cuda SHARED ${SOURCES} ${HEADERS} )

find_package(CUDAToolkit REQUIRED)
target_link_libraries(spoof_native_cuda CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)
target_compile_features(spoof_native_cuda PUBLIC cxx_std_11)
set_target_properties(spoof_native_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(spoof_native_cuda PROPERTIES OUTPUT_NAME "systemds_spoof_native_cuda-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# sets the installation path to src/main/cpp/lib
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ".." CACHE PATH "sets the installation path to src/main/cpp/lib" FORCE)
endif()

install(TARGETS spoof_native_cuda LIBRARY DESTINATION lib)
install(TARGETS spoof_native_cuda RUNTIME DESTINATION lib)

include_directories($ENV{JAVA_HOME}/include/)
include_directories($ENV{JAVA_HOME}/include/darwin)
include_directories($ENV{JAVA_HOME}/include/linux)
include_directories($ENV{JAVA_HOME}/include/win32)
