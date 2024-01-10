#!/usr/bin/env bash
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

gcc_toolkit="libgcc_s.so\|libm.so\|libstdc++\|libc.so\|libdl.so\|libgomp.so\|libpthread.so"
linux_loader="linux-vdso.so\|ld-linux-x86-64.so"
intel_mkl="libmkl_rt.so"

echo "Build MKL:"

# configure and compile INTEL MKL
cmake . -B INTEL -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-DUSE_GNU_THREADING -m64"
cmake --build INTEL --target install --config Release
patchelf --add-needed libmkl_rt.so lib/libsystemds_mkl-Linux-x86_64.so
rm -R INTEL

echo ""

echo "Non-standard dependencies for libsystemds_mkl-linux-x86_64.so"
ldd lib/libsystemds_mkl-Linux-x86_64.so | grep -v $gcc_toolkit"\|$linux_loader\|"$intel_mkl

echo ""

echo "Sucessfull install of MKL"
