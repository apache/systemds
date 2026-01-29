#!/bin/bash


# Stop if any command fails
set -e
. setenv_sds.sh

LOG_DEST="tpcxai_benchmark_run"
TPCxAI_CONFIG_FILE_PATH=${TPCxAI_BENCHMARKRUN_CONFIG_FILE_PATH}


if [[ ${TPCx_AI_VERBOSE} == "True" ]]; then
   VFLAG="-v"
fi

echo "TPCx-AI_HOME directory: ${TPCx_AI_HOME_DIR}";
echo "Using configuration file: ${TPCxAI_CONFIG_FILE_PATH} and scale factor ${TPCxAI_SCALE_FACTOR}..."
echo "Starting data generation..."
sleep 1;

PATH=$JAVA8_HOME/bin:$PATH
export JAVA8_HOME
export PATH
echo "Using Java at $JAVA8_HOME"
DATA_GEN_FLAG="--data-gen"
./bin/tpcxai.sh --phase {CLEAN,DATA_GENERATION,SCORING_DATAGEN,SCORING_LOADING} -sf ${TPCxAI_SCALE_FACTOR}  --streams ${TPCxAI_SERVING_THROUGHPUT_STREAMS}  -c ${TPCxAI_CONFIG_FILE_PATH} ${VFLAG} ${DATA_GEN_FLAG}

echo "Successfully generated data with scale factor ${TPCxAI_SCALE_FACTOR}."