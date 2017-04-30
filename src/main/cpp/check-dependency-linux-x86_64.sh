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

echo "-----------------------------------------------------------------------"
echo "Check for unexpected dependencies added after code change or new setup:"
echo "Non-standard dependencies for libpreload_systemml-linux-x86_64.so"
ldd lib/libpreload_systemml-Linux-x86_64.so | grep -v $gcc_toolkit"\|"$linux_loader
echo "Non-standard dependencies for libsystemml_mkl-linux-x86_64.so"
ldd lib/libsystemml_mkl-Linux-x86_64.so | grep -v $gcc_toolkit"\|"$linux_loader"\|"$intel_mkl
echo "Non-standard dependencies for libsystemml_openblas-linux-x86_64.so"
ldd lib/libsystemml_openblas-Linux-x86_64.so | grep -v $gcc_toolkit"\|"$linux_loader"\|"$openblas
echo "-----------------------------------------------------------------------"
