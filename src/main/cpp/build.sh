#!/bin/bash
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

# This shell script compiles the required shared libraries for 64-bit Linux on x86 machine

# yum whatprovides libgcc_s.so.1
# GNU Standard C++ Library: libstdc++.so.6
# GCC version 4.8 shared support library: libgcc_s.so.1
# The GNU libc libraries: libm.so.6, libdl.so.2, libc.so.6, libpthread.so.0
# GCC OpenMP v3.0 shared support library: libgomp.so.1
gcc_toolkit="libgcc_s.so\|libm.so\|libstdc++\|libc.so\|libdl.so\|libgomp.so\|libpthread.so"
linux_loader="linux-vdso.so\|ld-linux-x86-64.so"
intel_mkl="libmkl_rt.so"

# Fortran runtime: libgfortran.so.3
# GCC __float128 shared support library: libquadmath.so.0
openblas="libopenblas.so\|libgfortran.so\|libquadmath.so"

export DEBIAN_FRONTEND=noninteractive

if ! [ -x "$(command -v cmake)" ]; then
  echo 'Error: cmake is not installed.' >&2
  exit 1
fi

if ! [ -x "$(command -v patchelf)" ]; then
  echo 'Error: patchelf is not installed.' >&2
  exit 1
fi

#!/bin/bash

LIBRARY_DIR1="/opt/intel/oneapi/mkl/latest/lib/intel64"
LIBRARY_DIR2="/usr/lib/x86_64-linux-gnu"

CONF_FILE1="/etc/ld.so.conf.d/my_library1.conf"
CONF_FILE2="/etc/ld.so.conf.d/my_library2.conf"

# Check if the directories exist
if [ -d "$LIBRARY_DIR1" ]; then
    # Create a new configuration file for directory 1
    echo "$LIBRARY_DIR1" > "$CONF_FILE1"
    echo "Directory added to ldconfig: $LIBRARY_DIR1"
else
    echo "Error: Directory does not exist: $LIBRARY_DIR1"
fi

if [ -d "$LIBRARY_DIR2" ]; then
    # Create a new configuration file for directory 2
    echo "$LIBRARY_DIR2" > "$CONF_FILE2"
    echo "Directory added to ldconfig: $LIBRARY_DIR2"
else
    echo "Error: Directory does not exist: $LIBRARY_DIR2"
fi

# Update ldconfig cache
ldconfig


# Check if Intel MKL is installed
if ! ldconfig -p | grep -q libmkl_rt; then
  echo "Intel MKL not found. Installing Intel MKL..."

  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
  sudo apt install intel-oneapi-mkl

  source /opt/intel/oneapi/setvars.sh

fi

# Check if OpenBLAS is installed
if ! ldconfig -p | grep -q libopenblas; then
  echo "OpenBLAS not found. Installing OpenBLAS..."

  apt-get update
  apt-get install libopenblas-dev -y
fi

# configure and compile INTEL MKL
cmake . -B INTEL -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-DUSE_GNU_THREADING -m64"
cmake --build INTEL --target install --config Release
patchelf --add-needed libmkl_rt.so lib/libsystemds_mkl-Linux-x86_64.so
rm -R INTEL

# configure and compile OPENBLAS
cmake . -B OPENBLAS -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-m64"
cmake --build OPENBLAS --target install --config Release
patchelf --add-needed libopenblas.so.0 lib/libsystemds_openblas-Linux-x86_64.so
rm -R OPENBLAS

# check dependencies linux x86_64
echo "-----------------------------------------------------------------------"
echo "Check for unexpected dependencies added after code change or new setup:"
echo "Non-standard dependencies for libsystemds_mkl-linux-x86_64.so"
ldd lib/libsystemds_mkl-Linux-x86_64.so | grep -v $gcc_toolkit"\|$linux_loader\|"$intel_mkl
echo "Non-standard dependencies for libsystemds_openblas-linux-x86_64.so"
ldd lib/libsystemds_openblas-Linux-x86_64.so | grep -v $gcc_toolkit"\|$linux_loader\|"$openblas
echo "-----------------------------------------------------------------------"

# compile HE
cmake he/ -B HE -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build HE --target install --config Release
rm -R HE


