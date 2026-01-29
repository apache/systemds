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
# Copyright 2019 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them
# is governed by the express license under which they were provided to you ("License"). Unless the
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties,
# other than those that are expressly stated in the License.
#
#

# Stop if any command fails
set -e

. setenv_sds.sh

LOG_DEST="tpcxai_benchmark_run"
TPCxAI_CONFIG_FILE_PATH=${TPCxAI_BENCHMARKRUN_CONFIG_FILE_PATH}
if [[ ${IS_VALIDATION_RUN} -eq "1" ]]; then
   echo "Benchmark validation run. Setting scale factor value to 1..."
   export TPCxAI_SCALE_FACTOR=1
   TPCxAI_CONFIG_FILE_PATH=${TPCxAI_VALIDATION_CONFIG_FILE_PATH}
   LOG_DEST="tpcxai_benchmark_validation"
fi

if [[ ${TPCx_AI_VERBOSE} == "True" ]]; then
   VFLAG="-v"
fi

echo "TPCx-AI_HOME directory: ${TPCx_AI_HOME_DIR}";
echo "Using configuration file: ${TPCxAI_CONFIG_FILE_PATH} and Scale factor ${TPCxAI_SCALE_FACTOR}..."

PATH=$JAVA11_HOME/bin:$PATH
export JAVA11_HOME
export PATH
echo "Using Java at $JAVA11_HOME"

echo "Starting Benchmark run..."
sleep 1;

bash ${TPCxAI_ENV_TOOLS_DIR}/clock_check.sh start

./bin/tpcxai.sh --phase {CLEAN,DATA_GENERATION,LOADING,TRAINING,SERVING,SERVING,SERVING_THROUGHPUT,SCORING,CHECK_INTEGRITY} -sf ${TPCxAI_SCALE_FACTOR}  --streams ${TPCxAI_SERVING_THROUGHPUT_STREAMS}  -c ${TPCxAI_CONFIG_FILE_PATH} ${VFLAG}

BENCHMARK_RUN_EXIT_CODE=$?

if [ ${BENCHMARK_RUN_EXIT_CODE} -eq "0" ]; then
   echo "Generating report..."
   lib/python-venv/bin/python tools/tpcxai_fdr.py -d logs/tpcxai.db -f logs/report.txt
   lib/python-venv/bin/python tools/tpcxai_fdr.py -d logs/tpcxai.db -t html -f logs/report.html
   echo "Finished generating report";

   echo ""

   echo "Saving output data. This may take a few minutes ..."
   bash ${TPCxAI_ENV_TOOLS_DIR}/saveOutputData.sh
   echo "Finished saving output data"
   echo ""

   echo "Saving execution enviroment details..."
   bash ${TPCxAI_ENV_TOOLS_DIR}/getEnvInfo.sh
   echo "Finished saving environment details"
   echo ""
   echo "Saving data redundancy information..."
   bash ${TPCxAI_ENV_TOOLS_DIR}/dataRedundancyInformation.sh
   echo "Finished saving data redundancy"
fi

bash ${TPCxAI_ENV_TOOLS_DIR}/clock_check.sh end
LOG_DEST="${LOG_DEST}_$(date +"%m%d%Y_%H%m%S")"
mkdir -p "logs/history/${LOG_DEST}"
find logs/ -mindepth 1 -maxdepth 1 -path logs/history -prune -o -print | xargs -I file mv file "logs/history/${LOG_DEST}"
zip -vr "logs/history/${LOG_DEST}.zip" "logs/history/${LOG_DEST}" -x logs/history/*
