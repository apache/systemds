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
#/bin/bash

# from https://github.com/damslab/reproducibility/blob/e90f169ffa4bca37ec4cc1f231eea0cb41e910cb/sigmod2023-AWARE-p5/experiments/data/get_census.sh
echo "Beginning download of Census"

# Change directory to data.
if [[ pwd != *"data"* ]]; then
    cd "../data"
fi

# Download file if not already downloaded.
if [[ ! -f "census/census.csv" ]]; then
    mkdir -p census/
    #the download is very slow
    wget -nv -O census/census.csv https://kdd.ics.uci.edu/databases/census1990/USCensus1990.data.txt
    if [[ ! -f "census/census.csv" ]]; then
      echo "Successfully downloaded census dataset."
    else
      echo "Could not download dataset."
      exit
    fi
else
    echo "Census is already downloaded"
fi

if [[ ! -f "census/census.csv.mtd" ]]; then
    echo '{"format":csv,"header":true,"rows":2458285,"cols":69,"value_type":"int"}' > census/census.csv.mtd
else
    echo "Already constructed metadata for census.csv"
fi

# CD out of the data directory.
cd ../examples

if [[ ! -f "../data/census/census_bias.csv" ]]; then
    systemds Census1990_l2svm_prep.dml &
else
    echo "Already trained census svm model."
fi

wait

echo "Census Download / Training Done"

echo ""
echo ""