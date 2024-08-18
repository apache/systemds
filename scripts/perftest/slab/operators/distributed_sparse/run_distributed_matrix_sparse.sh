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

# Ensure script is run from the 'single_node_dense' directory
#if [ "$(basename $PWD)" != "single_node_dense" ]; then
#  echo "Please execute scripts from directory 'single_node_dense'"
#  exit 1
#fi


# Set up the output directory
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# Define datasets
SPARSITY=("0_0001" "0_001" "0_01" "0_1")
SHAPES=("tall" "wide")

# Define DML files and corresponding output files
DML_FILES=("slabFrobeniusNormSparse.dml" "slabGramMatrixSparse.dml" "slabMatrixAdditionSparse.dml" "slabMatrixMultSparse.dml" "slabMatrixVectorMultSparse.dml" "slabTransposeSparse.dml")
OUTPUT_FILES=("slabFrobeniusNormSparse_stats.txt" "slabGramMatrixSparse_stats.txt" "slabMatrixAdditionSparse_stats.txt" "slabMatrixMultSparse_stats.txt" "slabMatrixVectorMultSparse_stats.txt" "slabTransposeSparse_stats.txt")

# Base path to datasets
DATASET_PATH="../../../../../src/test/resources/datasets/slab/sparse"

# Iterate over each DML file
for index in ${!DML_FILES[@]}; do
  DML_FILE=${DML_FILES[$index]}
  OUTPUT_FILE=${OUTPUT_DIR}/${OUTPUT_FILES[$index]}

  # Clear the output file before writing
  > $OUTPUT_FILE

  # Special handling for slabMatrixMultSparse.dml
  if [ "$DML_FILE" == "slabMatrixMultSparse.dml" ]; then
    for SPARSE in ${SPARSITY[@]}; do
      for SHAPE in ${SHAPES[@]}; do
        if [ "$SHAPE" == "tall" ]; then
          CSV_FILE1="${DATASET_PATH}/M_sparsity_${SPARSE}_tall.csv"
          CSV_FILE2="${DATASET_PATH}/M_sparsity_${SPARSE}_wide.csv"
        else
          CSV_FILE1="${DATASET_PATH}/M_sparsity_${SPARSE}_wide.csv"
          CSV_FILE2="${DATASET_PATH}/M_sparsity_${SPARSE}_tall.csv"
        fi

        # Run the DML script with -stats flag and capture the output
        TEMP_FILE=$(mktemp)
        systemds $DML_FILE -exec spark -args $CSV_FILE1 $CSV_FILE2 -stats > $TEMP_FILE 2>&1

        # Write the sparsity and shape and SystemDS Statistics section to the output file
        echo "Sparsity: ${SPARSE//_/\.}, Shape: $SHAPE" >> $OUTPUT_FILE
        awk '/SystemDS Statistics:/{flag=1}flag' $TEMP_FILE >> $OUTPUT_FILE
        echo -e "\n\n\n\n" >> $OUTPUT_FILE  # Add empty lines for separation

        # Clean up temporary file
        rm $TEMP_FILE

        echo "Execution of ${DML_FILE} with ${CSV_FILE1} and ${CSV_FILE2} completed. Statistics appended to ${OUTPUT_FILE}"
      done
    done
  else
    # Handling for other DML files
    for SPARSE in ${SPARSITY[@]}; do
      for SHAPE in ${SHAPES[@]}; do
        CSV_FILE="${DATASET_PATH}/M_sparsity_${SPARSE}_${SHAPE}.csv"

        # Run the DML script with -stats flag and capture the output
        TEMP_FILE=$(mktemp)
        systemds $DML_FILE -exec spark -args $CSV_FILE -stats > $TEMP_FILE 2>&1

        # Write the sparsity and shape and SystemDS Statistics section to the output file
        echo "Sparsity: ${SPARSE//_/\.}, Shape: $SHAPE" >> $OUTPUT_FILE
        awk '/SystemDS Statistics:/{flag=1}flag' $TEMP_FILE >> $OUTPUT_FILE
        echo -e "\n\n\n\n" >> $OUTPUT_FILE  # Add empty lines for separation

        # Clean up temporary file
        rm $TEMP_FILE

        echo "Execution of ${DML_FILE} with ${CSV_FILE} completed. Statistics appended to ${OUTPUT_FILE}"
      done
    done
  fi
done
