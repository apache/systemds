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
project (he LANGUAGES CXX)

# All custom find modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Build a shared libraray
set(HEADER_FILES libhe.h he.h)
set(SOURCE_FILES libhe.cpp he.cpp)

# Build a shared libraray
add_library(he SHARED ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(he PROPERTIES MACOSX_RPATH 1)

# sets the installation path to src/main/cpp/lib
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/.." CACHE PATH "sets the installation path to src/main/cpp/lib" FORCE)
endif()

# sets the installation path to src/main/cpp/lib
# install(TARGETS he LIBRARY DESTINATION lib)
install(TARGETS he RUNTIME DESTINATION lib)

# unify library filenames to libhe_<...>
if (WIN32)
    set(CMAKE_IMPORT_LIBRARY_PREFIX lib CACHE INTERNAL "")
    set(CMAKE_SHARED_LIBRARY_PREFIX lib CACHE INTERNAL "")
endif()

set(CMAKE_BUILD_TYPE Release)
set_target_properties(he PROPERTIES OUTPUT_NAME "he-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

find_package(SEAL 3.7 REQUIRED)
target_link_libraries(he SEAL::seal_shared)

# Include directories. (added for Linux & Darwin, fix later for windows)
# include paths can be spurious
include_directories($ENV{JAVA_HOME}/include/)
include_directories($ENV{JAVA_HOME}/include/darwin)
include_directories($ENV{JAVA_HOME}/include/linux)
include_directories($ENV{JAVA_HOME}/include/win32)
