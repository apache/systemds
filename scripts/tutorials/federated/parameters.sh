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

# Memory allowed to be used by each worker and coordinator
# Path to the systemds clone of the repository.
export SYSTEMDS_ROOT="$HOME/github/systemds"
# Add SystemDS bin to path
export PATH="$SYSTEMDS_ROOT/bin:$PATH"

## Logging variables:
# Set logging properties for the system
# Off disable the logging
export LOG4JPROP='conf/log4j-off.properties'
# export LOG4JPROP='conf/log4j-debug.properties'
# export LOG4JPROP='conf/log4j-info.properties'

# Set the system to start up on quiet mode, to not print excessively on every execution.
export SYSDS_QUIET=1

# export COMMAND='java -Xmx8g -Xms8g -cp "./lib/*;./SystemDS_old.jar" org.apache.sysds.api.DMLScript -f'

# Set the addresses of your federated workers.
# address=("so007" "so004" "so005" "so006")
address=("localhost" "localhost" "localhost" "localhost")

# We assume for the scripts to work that each worker have a unique port
ports=("8001" "8002" "8003" "8004")
numWorkers=${#address[@]}

# Set memory usage:
addressesString=${address// /|}
## if distributed set memory higher!
if [[ "$addressesString" == *"so0"* ]]; then 
    export SYSTEMDS_STANDALONE_OPTS="-Xmx16g -Xms16g -Xmn1600m"
else 
    export SYSTEMDS_STANDALONE_OPTS="-Xmx4g -Xms4g -Xmn400m"
fi

if [[ $HOSTNAME == *"so0"* ]]; then 
    ## Set scale out nodes memory higher!
    export SYSTEMDS_STANDALONE_OPTS="-Xmx230g -Xms230g -Xmn23000m"
fi 

# If remote workers are used make and use this directory on the sites.
# Note this is a directory relative to the $home on the sites.
remoteDir="github/federatedTutorial-v3/"

# configuration:
# This define the configuration file to be used for the execution.
# Change this to enable different settings of SystemDS
conf="def"
# conf="ssl"

# Federated dataset variables:
x="data/fed_mnist_features_${numWorkers}.json"
y="data/fed_mnist_labels_${numWorkers}.json"
y_hot="data/fed_mnist_labels_hot_${numWorkers}.json"

xt="data/fed_mnist_test_features_${numWorkers}.json"
yt="data/fed_mnist_test_labels_${numWorkers}.json"
yt_hot="data/fed_mnist_test_labels_hot_${numWorkers}.json"

# Local dataset variables:
x_loc="data/mnist_features.data"
y_loc="data/mnist_labels.data"
xt_loc="data/mnist_test_features.data"
yt_loc="data/mnist_test_labels.data"
