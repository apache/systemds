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

# Ensure script is run from the 'pipeline' directory
if [ "$(basename $PWD)" != "pipeline" ]; then
  echo "Please execute scripts from directory 'pipeline'"
  exit 1
fi


# Set up the output directory
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# Define row numbers for slabMultiplicationChain.dml
ROW_NUMBERS=("1000" "10000" "100000" "1000000")
MULTIPLICATION_CHAIN_FILE="slabMultiplicationChain.dml"
MULTIPLICATION_CHAIN_OUTPUT="${OUTPUT_DIR}/slabMultiplicationChain_stats.txt"

# Clear the output file before writing
> $MULTIPLICATION_CHAIN_OUTPUT

# Iterate over each row number and execute slabMultiplicationChain.dml
for ROW in ${ROW_NUMBERS[@]}; do
  TEMP_FILE=$(mktemp)
  if systemds $MULTIPLICATION_CHAIN_FILE -exec spark -args $ROW -stats > $TEMP_FILE 2>&1; then
    echo "Number of rows: $ROW" >> $MULTIPLICATION_CHAIN_OUTPUT
    awk '/SystemDS Statistics:/{flag=1}flag' $TEMP_FILE >> $MULTIPLICATION_CHAIN_OUTPUT
  else
    echo "An error occurred while executing ${MULTIPLICATION_CHAIN_FILE} with rows ${ROW}. Check ${TEMP_FILE} for details." >> $MULTIPLICATION_CHAIN_OUTPUT
  fi
  echo -e "\n\n\n\n" >> $MULTIPLICATION_CHAIN_OUTPUT  # Add empty lines for separation
  rm $TEMP_FILE
  echo "Execution of ${MULTIPLICATION_CHAIN_FILE} with ${ROW} rows completed. Statistics appended to ${MULTIPLICATION_CHAIN_OUTPUT}"
done

# Define datasets for slabSVD.dml
DATASET_PATH="../../../../src/test/resources/datasets/slab/dense"
DATASETS=("M_dense_tall.csv" "M_dense_wide.csv")
SVD_FILE="slabSVD.dml"
SVD_OUTPUT="${OUTPUT_DIR}/slabSVD_stats.txt"

# Clear the output file before writing
> $SVD_OUTPUT

# Iterate over each dataset and execute slabSVD.dml
for DATASET in ${DATASETS[@]}; do
  SHAPE=$(echo $DATASET | grep -oP '(tall|wide)')
  TEMP_FILE=$(mktemp)
  if systemds $SVD_FILE -exec spark -args ${DATASET_PATH}/${DATASET} -stats > $TEMP_FILE 2>&1; then
    echo "Shape: $SHAPE" >> $SVD_OUTPUT
    awk '/SystemDS Statistics:/{flag=1}flag' $TEMP_FILE >> $SVD_OUTPUT
  else
    echo "An error occurred while executing ${SVD_FILE} with dataset ${DATASET}. Check ${TEMP_FILE} for details." >> $SVD_OUTPUT
  fi
  echo -e "\n\n\n\n" >> $SVD_OUTPUT  # Add empty lines for separation
  rm $TEMP_FILE
  echo "Execution of ${SVD_FILE} with dataset ${DATASET} completed. Statistics appended to ${SVD_OUTPUT}"
done
