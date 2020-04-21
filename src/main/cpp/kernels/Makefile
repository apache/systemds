#-------------------------------------------------------------
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
#-------------------------------------------------------------

NVCC=nvcc
GCC=gcc-8

# set the CUDA-supported version of gcc through -ccbin
CUDAFLAGS= -ptx -c -arch=sm_30 --std c++11 -ccbin $(GCC)

# Flags for compilation on recent Ubuntu + P100
#CUDAFLAGS= -ptx -c --std c++11 -ccbin gcc-8 -m64 -gencode arch=compute_60,code=sm_60

# Use these flags for precise math
#CUDAFLAGS= -ptx -c -arch=sm_30 -ftz=false -prec-div=true -prec-sqrt=true


SystemDS.o: SystemDS.cu
	$(NVCC) $(CUDAFLAGS)  SystemDS.cu

all: SystemDS.o


clean:
	rm -rf SystemDS.ptx
