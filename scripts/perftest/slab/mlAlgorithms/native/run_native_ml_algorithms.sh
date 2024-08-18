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
if [ "$(basename $PWD)" != "native" ]; then
  echo "Please execute scripts from directory 'native'"
  exit 1
fi



# Set up the output directory
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# Define row numbers for the DML scripts
ROW_NUMBERS=("1000" "10000" "100000" "1000000")

# Define DML files and corresponding output files
DML_FILES=("slabLinearRegCG.dml" "slabMultiLogitReg.dml" "slabNativePCA.dml")
OUTPUT_FILES=("slabLinearRegCG_stats.txt" "slabMultiLogitReg_stats.txt" "slabNativePCA_stats.txt")

# Function to run DML script and handle errors
run_dml() {
  local DML_FILE=$1
  local ARGS=$2
  local OUTPUT_FILE=$3

  # Run the DML script with -stats flag and capture the output
  TEMP_FILE=$(mktemp)
  if systemds $DML_FILE -args $ARGS -stats > $TEMP_FILE 2>&1; then
    # Write the number of rows and SystemDS Statistics section to the output file
    echo "Number of rows: $ARGS" >> $OUTPUT_FILE
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

  # Iterate over each row number and execute the DML file
  for ROW in ${ROW_NUMBERS[@]}; do
    run_dml $DML_FILE $ROW $OUTPUT_FILE
    echo "Execution of ${DML_FILE} with ${ROW} rows completed. Statistics appended to ${OUTPUT_FILE}"
  done
done
