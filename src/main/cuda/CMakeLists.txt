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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# default to gcc 8.x while we're still supporting CUDA 10.x only
if (UNIX)
    set(CMAKE_CUDA_HOST_COMPILER g++-8 CACHE INTERNAL "")
    set(CMAKE_CUDA_COMPILER nvcc CACHE INTERNAL "")
    set(CMAKE_CXX_COMPILER g++ CACHE INTERNAL "")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_WITH_INSTALL_RPATH True CACHE INTERNAL "")

project(SystemDS LANGUAGES CXX CUDA)

set(SYSDS_HEADERS 
	headers/agg_ops.cuh
	headers/cum_max.cuh  
	headers/cum_min.cuh  
	headers/cum_prod.cuh  
	headers/cum_scan.cuh  
	headers/cum_sum.cuh  
	headers/cum_sum_prod.cuh  
	headers/utils.cuh)
set(SYSDS_SOURCES kernels/SystemDS.cu)

add_library(SystemDS OBJECT ${SYSDS_HEADERS} ${SYSDS_SOURCES})
target_include_directories(SystemDS PUBLIC "${CMAKE_SOURCE_DIR}/headers")

find_package(CUDAToolkit REQUIRED)
cmake_policy(SET CMP0104 NEW)

set(CMAKE_CUDA_ARCHITECTURES  OFF)
#ToDo: more compiler flag settings for Debug/Release compilation
#set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures" FORCE)
#set(CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
#message("CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

set_property(TARGET SystemDS PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_property(TARGET SystemDS PROPERTY CUDA_PTX_COMPILATION ON)

# sets the installation path to src/main/cuda
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "sets the installation path to src/main/cpp/lib" FORCE)
endif()

install(FILES $<TARGET_OBJECTS:SystemDS> DESTINATION kernels)

#-------------------------------------------------------------
#project (spoof_cuda LANGUAGES CXX CUDA)

add_library(reduction OBJECT kernels/reduction.cu headers/reduction.cuh)
target_include_directories(reduction PUBLIC "${CMAKE_SOURCE_DIR}/headers")
set_property(TARGET reduction PROPERTY CUDA_PTX_COMPILATION ON)
install(FILES $<TARGET_OBJECTS:reduction> DESTINATION kernels)

include_directories($ENV{JAVA_HOME}/include/)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    include_directories($ENV{JAVA_HOME}/include/darwin)
endif()
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    include_directories($ENV{JAVA_HOME}/include/linux)
endif()
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    include_directories($ENV{JAVA_HOME}/include/win32)
endif()

set(SPOOF_HEADERS 
	spoof-launcher/jni_bridge.h
	spoof-launcher/SpoofCUDAContext.h
	headers/Matrix.h
	spoof-launcher/SpoofOperator.h
	spoof-launcher/SpoofRowwise.h
	spoof-launcher/SpoofCellwise.h)

set(SPOOF_SOURCES 
	spoof-launcher/jni_bridge.cpp
	spoof-launcher/SpoofCUDAContext.cpp
		)

set(SPOOF_CUDA_HEADERS
	headers/agg_ops.cuh
	headers/reduction.cuh
	headers/spoof_utils.cuh
	headers/TempStorage.cuh
	headers/utils.cuh
	headers/operators.cuh
	headers/Matrix.h
	headers/vector_write.cuh
	headers/vector_add.cuh)

set(SPOOF_TEMPLATES
	spoof/cellwise.cu    
	spoof/rowwise.cu)
source_group("SPOOF Templates" FILES ${SPOOF_TEMPLATES})
set_source_files_properties( ${SPOOF_TEMPLATES} PROPERTIES HEADER_FILE_ONLY ON)
add_library(spoof_cuda SHARED ${SPOOF_HEADERS} ${SPOOF_CUDA_HEADERS} ${SPOOF_SOURCES} ${SPOOF_TEMPLATES})

set_property(TARGET reduction PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_property(TARGET spoof_cuda PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

target_include_directories(spoof_cuda PRIVATE "${CMAKE_SOURCE_DIR}/ext/jitify" headers)
target_link_libraries(spoof_cuda CUDA::nvrtc CUDA::cuda_driver CUDA::cudart CUDA::cublas)
target_compile_features(spoof_cuda PUBLIC cxx_std_11)
set_target_properties(spoof_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(spoof_cuda PROPERTIES OUTPUT_NAME "systemds_spoof_cuda-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# unify naming convention to libsystemds_...
if (WIN32)    
    set(CMAKE_IMPORT_LIBRARY_PREFIX lib CACHE INTERNAL "")
    set(CMAKE_SHARED_LIBRARY_PREFIX lib CACHE INTERNAL "")
    target_link_libraries(spoof_cuda DbgHelp.lib)
    install(TARGETS spoof_cuda RUNTIME DESTINATION ../cpp/lib)
endif()

if(UNIX)
    install(TARGETS spoof_cuda LIBRARY DESTINATION ../cpp/lib)
endif()
