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
if [ "$(basename $PWD)" != "distributed" ]; then
  echo "Please execute scripts from directory 'distributed'"
  exit 1
fi


# Set up the output directory
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# Define datasets
DATASET_PATH_DENSE="../../../../../src/test/resources/datasets/slab/dense"
DATASET_PATH_SPARSE="../../../../../src/test/resources/datasets/slab/sparse"
DENSE_DATASETS=("M_dense_tall.csv" "M_dense_wide.csv")
SPARSE_DATASETS=("M_sparsity_0_0001_tall.csv" "M_sparsity_0_0001_wide.csv" "M_sparsity_0_001_tall.csv" "M_sparsity_0_001_wide.csv" "M_sparsity_0_01_tall.csv" "M_sparsity_0_01_wide.csv" "M_sparsity_0_1_tall.csv" "M_sparsity_0_1_wide.csv")

# Define DML files and corresponding output files
DML_FILES=("slabHeteroscedasticityRobustStandardErrorsDistr.dml" "slabLogisticRegressionDistr.dml" "slabNonNegativeMatrixFactorizationDistr.dml" "slabOrdinaryLeastSquaresRegressionDistr.dml" "slabPCADistr.dml")
OUTPUT_FILES=("slabHeteroscedasticityRobustStandardErrorsDistr_stats.txt" "slabLogisticRegressionDistr_stats.txt" "slabNonNegativeMatrixFactorizationDistr_stats.txt" "slabOrdinaryLeastSquaresRegressionDistr_stats.txt" "slabPCADistr_stats.txt")

# Function to run DML script and handle errors
run_dml() {
  local DML_FILE=$1
  local ARGS=$2
  local SPARSITY=$3
  local SHAPE=$4
  local OUTPUT_FILE=$5

  # Run the DML script with -exec spark and -stats flag, and capture the output
  TEMP_FILE=$(mktemp)
  if systemds $DML_FILE -exec spark -args $ARGS -stats > $TEMP_FILE 2>&1; then
    # Write the sparsity, shape, and SystemDS Statistics section to the output file
    echo "Sparsity: $SPARSITY, Shape: $SHAPE" >> $OUTPUT_FILE
    awk '/SystemDS Statistics:/{flag=1}flag' $TEMP_FILE >> $OUTPUT_FILE
  else
    echo "An error occurred while executing ${DML_FILE} with arguments ${ARGS}. Check ${TEMP_FILE} for details." >> $OUTPUT_FILE
  fi
  echo -e "\n\n\n\n" >> $OUTPUT_FILE  # Add empty lines for separation
  rm $TEMP_FILE
}

# Iterate over each DML file
for index in ${!DML_FILES[@]}; do
  DML_FILE=${DML_FILES[$index]}
  OUTPUT_FILE=${OUTPUT_DIR}/${OUTPUT_FILES[$index]}

  # Clear the output file before writing
  > $OUTPUT_FILE

  # Run with dense datasets
  for DATASET in ${DENSE_DATASETS[@]}; do
    SHAPE=$(echo $DATASET | grep -oP '(tall|wide)')
    SPARSITY="dense"
    run_dml $DML_FILE "${DATASET_PATH_DENSE}/${DATASET}" $SPARSITY $SHAPE $OUTPUT_FILE
    echo "Execution of ${DML_FILE} with dataset ${DATASET} completed. Statistics appended to ${OUTPUT_FILE}"
  done

  # Run with sparse datasets
  for DATASET in ${SPARSE_DATASETS[@]}; do
    SHAPE=$(echo $DATASET | grep -oP '(tall|wide)')
    SPARSITY=$(echo $DATASET | grep -oP '0_\d+')
    SPARSITY=${SPARSITY//_/\.}  # Replace underscore with dot
    run_dml $DML_FILE "${DATASET_PATH_SPARSE}/${DATASET}" $SPARSITY $SHAPE $OUTPUT_FILE
    echo "Execution of ${DML_FILE} with dataset ${DATASET} completed. Statistics appended to ${OUTPUT_FILE}"
  done
done
