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

cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW) # make use of <package>_ROOT variable
project (systemds LANGUAGES CXX)

# All custom find modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Options to Use OpenBLAS or Intel MKL
option(USE_OPEN_BLAS "Whether to use OpenBLAS (Defaults to compiling with Intel MKL, if both set, MKL has priority)" OFF)
option(USE_INTEL_MKL "Whether to use Intel MKL (Defaults to compiling with Intel MKL)" OFF)

# Build a shared libraray
set(HEADER_FILES libmatrixdnn.h libmatrixmult.h systemds.h)
set(SOURCE_FILES libmatrixdnn.cpp libmatrixmult.cpp systemds.cpp)

# Build a shared libraray
add_library(systemds SHARED ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(systemds PROPERTIES MACOSX_RPATH 1)
set(MATH_LIBRARIES "")

# sets the installation path to src/main/cpp/lib
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "sets the installation path to src/main/cpp/lib" FORCE)
endif()

# sets the installation path to src/main/cpp/lib
# install(TARGETS systemds LIBRARY DESTINATION lib)
install(TARGETS systemds RUNTIME DESTINATION lib)

# unify library filenames to libsystemds_<...>
if (WIN32)
    set(CMAKE_IMPORT_LIBRARY_PREFIX lib CACHE INTERNAL "")
    set(CMAKE_SHARED_LIBRARY_PREFIX lib CACHE INTERNAL "")
endif()

set(CMAKE_BUILD_TYPE Release)

if (USE_OPEN_BLAS)
  find_package(OpenBLAS REQUIRED)
  # sets the name of the output to include the os and the architecture
  set_target_properties(systemds PROPERTIES OUTPUT_NAME "systemds_openblas-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  include_directories(${OpenBLAS_INCLUDE_DIR})
  set(MATH_LIBRARIES "${OpenBLAS_LIB}")
  add_definitions(-DUSE_OPEN_BLAS)
elseif(USE_INTEL_MKL)
  find_package(MKL REQUIRED)
  # sets the name of the output to include the os and the architecture
  set_target_properties(systemds PROPERTIES OUTPUT_NAME "systemds_mkl-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  include_directories(${MKL_INCLUDE_DIR})
  set(MATH_LIBRARIES "${MKL_LIBRARIES}")
  add_definitions(-DUSE_INTEL_MKL)
endif()

# Include directories. (added for Linux & Darwin, fix later for windows)
# include paths can be spurious
include_directories($ENV{JAVA_HOME}/include/)
include_directories($ENV{JAVA_HOME}/include/darwin)
include_directories($ENV{JAVA_HOME}/include/linux)
include_directories($ENV{JAVA_HOME}/include/win32)

# Setting CXX compiler flags
if (USE_OPEN_BLAS)
  # OpenMP is required
  find_package(OpenMP REQUIRED)
  set_target_properties(systemds PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS} ${MATH_LIBRARIES}")
elseif(USE_INTEL_MKL)
  set_target_properties(systemds PROPERTIES LINK_FLAGS "${MATH_LIBRARIES}")
endif()

