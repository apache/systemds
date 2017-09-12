#!/usr/bin/bash
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

# Steps to install caffe

# Install OpenBLAS if not already installed

if [ ! -d "/opt/OpenBLAS/" ]; then
	git clone https://github.com/xianyi/OpenBLAS.git
	cd OpenBLAS/
	make clean
	make USE_OPENMP=1
	sudo make install
cd ..
fi

git clone https://github.com/BVLC/caffe.git
cd caffe 
sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel
sudo yum install gflags-devel glog-devel lmdb-devel
cp Makefile.config.example Makefile.config
# Set following in Makefile.config
# BLAS := open
# BLAS_INCLUDE := /opt/OpenBLAS/include/
# BLAS_LIB := /opt/OpenBLAS/lib/
# You may also have to change PYTHON_INCLUDE in Makefile.config if numpy is installed in custom path
echo 'Please set following in Makefile.config:'
echo 'BLAS := open'
echo 'BLAS_INCLUDE := /opt/OpenBLAS/include/'
echo 'BLAS_LIB := /opt/OpenBLAS/lib/'
echo 'Finally, do make all and make pycaffe'
# make all
# make pycaffe