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
export SYSTEMDS_STANDALONE_OPTS="-Xmx4g -Xms4g -Xmn400m"
# Path to the systemds clone of the repository.
export SYSTEMDS_ROOT="$HOME/github/systemds"
# Add SystemDS bin to path
export PATH="$SYSTEMDS_ROOT/bin:$PATH"

## Logging variables:
# Set logging properties for the system
export LOG4JPROP='conf/log4j-off.properties'
# export LOG4JPROP='conf/log4j-debug.properties'
# export LOG4JPROP='conf/log4j-info.properties'
export SYSDS_QUIET=1

# address=("tango" "delta" "india" "echo")
# address=("tango" "delta")

address=("localhost" "localhost" "localhost" "localhost")
ports=("8001" "8002" "8003" "8004")
numWorkers=${#address[@]}

# If remote workers are used make and use this directory on the sites.
# Note this is a directory relative to the $home on the sites.
remoteDir="github/federatedTutorial-v3/"

# configuration:
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
