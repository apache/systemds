#!/bin/bash

#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


# Find benchmark kit base directory
#export TPCx_AI_HOME_DIR=$(cd `dirname -- $0` && pwd);
cd "$(dirname ${BASH_SOURCE[0]})"
export TPCx_AI_HOME_DIR="$PWD"
cd "$OLDPWD"
#


### Common variables for Python and Spark ###

 # Verbosity: Set to True for verbose execution
export TPCx_AI_VERBOSE=False

# The scale factor is the dataset size in GB that will be generated and used to run the benchmark.
export TPCxAI_SCALE_FACTOR=1

# Number of current streams to use in the SERVING_THROUGHPUT test
export TPCxAI_SERVING_THROUGHPUT_STREAMS=2

# The absolute path to the configuration file used to run the validation test
export TPCxAI_VALIDATION_CONFIG_FILE_PATH=${TPCx_AI_HOME_DIR}/driver/config/default.yaml

# The absolute path to the configuration file used for the benchmark run
export TPCxAI_BENCHMARKRUN_CONFIG_FILE_PATH=${TPCx_AI_HOME_DIR}/driver/config/default.yaml

# Location of the subdirectory containing scripts to collect system configuration information
export TPCxAI_ENV_TOOLS_DIR=${TPCx_AI_HOME_DIR}/tools/python

# Binary for Parallel SSH used for parallel data gen and getEnvInfo
export TPCxAI_PSSH=pssh
export TPCxAI_PSCP=pscp.pssh

# Java options for PDGF
export TPCxAI_PDGF_JAVA_OPTS=""

# Set path to Java 8 for data generation
export JAVA8_HOME= # TODO set the path to Java 8 home directory

# Set path to Java 11 for benchmark run
export JAVA11_HOME= # TODO set the path to Java 11 home directory

### Configuration variables for Spark only ###

# export YARN_CONF_DIR=/etc/hadoop/conf.cloudera.yarn

# Location of the Python binary of the virtual environment used to run the DL use cases
# export PYSPARK_PYTHON=/usr/envs/adabench_dl/bin/python
# export PYSPARK_DRIVER_PYTHON=/usr/envs/adabench_dl/bin/python
