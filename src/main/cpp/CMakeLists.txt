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


cmake_minimum_required(VERSION 2.8)

project (systemml)

# All custom find modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Options to Use OpenBLAS or Intel MKL
option(USE_OPEN_BLAS "Whether to use OpenBLAS (Defaults to compiling with Intel MKL, if both set, MKL has priority)" OFF)
option(USE_INTEL_MKL "Whether to use Intel MKL (Defaults to compiling with Intel MKL)" ON)

# Build a shared libraray
add_library(systemml SHARED libmatrixdnn.cpp  libmatrixmult.cpp  systemml.cpp)

set(MATH_LIBRARIES "")

# sets the installation path to src/main/cpp/lib
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS systemml LIBRARY DESTINATION lib)

set(CMAKE_BUILD_TYPE Release)

if (USE_OPEN_BLAS)
  find_package(OpenBLAS REQUIRED)
  # sets the name of the output to include the os and the architecture
  set_target_properties(systemml PROPERTIES OUTPUT_NAME "systemml_openblas-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  include_directories(${OpenBLAS_INCLUDE_DIR})
  set(MATH_LIBRARIES "${OpenBLAS_LIB}")
elseif(USE_INTEL_MKL)
  find_package(MKL REQUIRED)
  # sets the name of the output to include the os and the architecture
  set_target_properties(systemml PROPERTIES OUTPUT_NAME "systemml_mkl-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  include_directories(${MKL_INCLUDE_DIR})
  set(MATH_LIBRARIES "${MKL_LIBRARIES}")
endif()

# Option written to a config.h file
configure_file(${CMAKE_SOURCE_DIR}/config.h.cmake ${CMAKE_BINARY_DIR}/config.h)

# Include directories. (added for Linux & Darwin, fix later for windows)
# include paths can be spurious
include_directories($ENV{JAVA_HOME}/include/)
include_directories($ENV{JAVA_HOME}/include/darwin)
include_directories($ENV{JAVA_HOME}/include/linux)
include_directories($ENV{JAVA_HOME}/include/win32)

# Include the binary dir which contains "config.h"
include_directories(${CMAKE_BINARY_DIR})


# Setting CXX compiler flags
if (USE_OPEN_BLAS)
  # OpenMP is required
  find_package(OpenMP REQUIRED)
  set_target_properties(systemml PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS} ${MATH_LIBRARIES}")
elseif(USE_INTEL_MKL)
  set_target_properties(systemml PROPERTIES LINK_FLAGS ${MATH_LIBRARIES}")
endif()
