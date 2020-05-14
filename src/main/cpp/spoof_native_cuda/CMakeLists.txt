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
endif()

project (systemds_spoof_native LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_BUILD_WITH_INSTALL_RPATH True CACHE INTERNAL "")

set(HEADERS spoof_native_jni.h spoof_native_cuda.h)
set(SOURCES spoof_native_jni.cpp spoof_native_cuda.cpp)

# Build a shared libraray
add_library(spoof_native_cuda SHARED ${SOURCES} ${HEADERS} )

if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX lib CACHE INTERNAL "")
    set(CMAKE_IMPORT_LIBRARY_PREFIX lib CACHE INTERNAL "")
    target_link_libraries(spoof_native_cuda DbgHelp.lib)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(spoof_native_cuda CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)
target_compile_features(spoof_native_cuda PUBLIC cxx_std_11)
set_target_properties(spoof_native_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(spoof_native_cuda PROPERTIES OUTPUT_NAME "systemds_spoof_native_cuda-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

#set(prop OUTPUT_NAME)
#set(tgt spoof_native_cuda)

#message ("Checking ${prop}")
#get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
#if (propval)
#    get_target_property(propval ${tgt} ${prop})
#    message ("${tgt} ${prop} = ${propval}")
#endif()

#message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# sets the installation path to src/main/cpp/lib
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  #message(STATUS "setting install prefix")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/.." CACHE PATH "sets the installation path to src/main/cpp/lib" FORCE)
endif()

install(TARGETS spoof_native_cuda LIBRARY DESTINATION lib)
install(TARGETS spoof_native_cuda RUNTIME DESTINATION lib)

include_directories($ENV{JAVA_HOME}/include/)
include_directories($ENV{JAVA_HOME}/include/darwin)
include_directories($ENV{JAVA_HOME}/include/linux)
include_directories($ENV{JAVA_HOME}/include/win32)
