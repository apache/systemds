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
if [ "$(basename $PWD)" != "single_node_dense" ]; then
  echo "Please execute scripts from directory 'single_node_dense'"
  exit 1
fi


# Set up the output directory
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# List of row numbers for testing
ROW_NUMBERS=("2500000" "5000000" "10000000" "20000000")

# List of DML files and corresponding output files
DML_FILES=("slabFrobeniusNorm.dml" "slabGramMatrix.dml" "slabMatrixAddition.dml" "slabMatrixMult.dml" "slabMatrixVectorMult.dml" "slabTranspose.dml")
OUTPUT_FILES=("slabFrobeniusNorm_stats.txt" "slabGramMatrix_stats.txt" "slabMatrixAddition_stats.txt" "slabMatrixMult_stats.txt" "slabMatrixVectorMult_stats.txt" "slabTranspose_stats.txt")

# Iterate over each DML file and execute it with different row numbers
for index in ${!DML_FILES[@]}; do
  DML_FILE=${DML_FILES[$index]}
  OUTPUT_FILE=${OUTPUT_DIR}/${OUTPUT_FILES[$index]}

  # Clear the output file before writing
  > $OUTPUT_FILE

  for ROW in ${ROW_NUMBERS[@]}; do
    # Run the DML script with -stats flag and capture the output
    TEMP_FILE=$(mktemp)
    systemds $DML_FILE -args $ROW -stats > $TEMP_FILE 2>&1

    # Write the number of rows and SystemDS Statistics section to the output file
    echo "Number of rows: $ROW" >> $OUTPUT_FILE
    awk '/SystemDS Statistics:/{flag=1}flag' $TEMP_FILE >> $OUTPUT_FILE
    echo -e "\n\n\n\n" >> $OUTPUT_FILE  # Add empty lines for separation

    # Clean up temporary file
    rm $TEMP_FILE

    echo "Execution of ${DML_FILE} with ${ROW} rows completed. Statistics appended to ${OUTPUT_FILE}"
  done
done
