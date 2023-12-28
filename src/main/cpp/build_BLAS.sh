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
openblas="libopenblas.so\|libgfortran.so\|libquadmath.so"

echo "Build OpenBLAS:"

# configure and compile OPENBLAS
cmake . -B OPENBLAS -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="-m64"
cmake --build OPENBLAS --target install --config Release
patchelf --add-needed libopenblas.so.0 lib/libsystemds_openblas-Linux-x86_64.so
rm -R OPENBLAS

echo ""

echo "Non-standard dependencies for libsystemds_openblas-linux-x86_64.so"
ldd lib/libsystemds_openblas-Linux-x86_64.so | grep -v $gcc_toolkit"\|$linux_loader\|"$openblas

echo "" 
echo "Sucessfull install of OpenBLAS"
