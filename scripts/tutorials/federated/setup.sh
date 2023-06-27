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
source parameters.sh

export LOG4JPROP='conf/log4j-off.properties'
# export LOG4JPROP='conf/log4j-debug.properties'
export SYSDS_QUIET=1

if [[ ! -d "python_venv" ]]; then
    # Install systemds python environment if not already installed.
    # this also install the system on the remotes defined in parameters.sh
    ./install.sh
fi

# Activate python environment
source "python_venv/bin/activate"

# Make the data directiory if it does not exist
mkdir -p data

# Syncronize the basic scripts with the remotes.
./sync.sh &

python code/dataGen/generate_mnist.py &

# Generate the fedreated json files.
python code/dataGen/federatedMetaDataGenerator.py \
    -p ${ports[@]} -a ${address[@]} -n $numWorkers -d "mnist_features_" \
    -f 784 -e 60000 &
python code/dataGen/federatedMetaDataGenerator.py \
    -p ${ports[@]} -a ${address[@]} -n $numWorkers -d "mnist_labels_" \
    -f 1 -e 60000 &
python code/dataGen/federatedMetaDataGenerator.py \
    -p ${ports[@]} -a ${address[@]} -n $numWorkers -d "mnist_labels_hot_" \
    -f 10 -e 60000 &

python code/dataGen/federatedMetaDataGenerator.py \
    -p ${ports[@]} -a ${address[@]} -n $numWorkers -d "mnist_test_features_" \
    -f 784 -e 10000 &
python code/dataGen/federatedMetaDataGenerator.py \
    -p ${ports[@]} -a ${address[@]} -n $numWorkers -d "mnist_test_labels_" \
    -f 1 -e 10000 &
python code/dataGen/federatedMetaDataGenerator.py \
    -p ${ports[@]} -a ${address[@]} -n $numWorkers -d "mnist_test_labels_hot_" \
    -f 10 -e 10000 &

wait

# Make Slices of dataset
datasets="mnist_features mnist_labels mnist_labels_hot mnist_test_features mnist_test_labels mnist_test_labels_hot"
for name in $datasets; do
    if [[ ! -f "data/${name}_${numWorkers}_1.data.mtd" ]]; then
        echo "Generating data/${name}_${numWorkers}_1.data"
        systemds code/dataGen/slice.dml \
            -config conf/def.xml \
            -args $name $numWorkers &
    fi
done

wait

# Distribute the slices to individual workers.
for index in ${!address[@]}; do
    if [ "${address[$index]}" != "localhost" ]; then
        sleep 0.2
        echo "Syncronize and distribute data partitions."
        ## File ID is the federated Indentification number
        fileId=$((index + 1))
        # ssh -q ${address[$index]} [[ -f "${remoteDir}/data/mnist_features_${numWorkers}_${fileId}.data" ]] &&
        #     echo "Skipping transfer since ${address[$index]} already have the file" ||
        rsync -ah -e ssh --include="**_${numWorkers}_${fileId}.da**" --exclude='*' data/ ${address[$index]}:$remoteDir/data/ &
    fi
done

wait
