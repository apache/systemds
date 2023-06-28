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

# Execute a sum of the dataset
# systemds code/exp/sum.dml -args $x
# Get statistics output
# systemds code/exp/sum.dml -stats -args $x
# Get execution explaination
# systemds code/exp/sum.dml -explain -args $x

# Execute a Linear model algorithm
# systemds code/exp/lm.dml \
#     -config conf/$conf.xml \
#     -stats 100 \
#     -debug \
#     -args $x $y_hot TRUE "results/fed_mnist_${numWorkers}.res" \
#     -fedMonitoringAddress "http://localhost:8080"

# Execute a Multi Log Regression model, do prediction and print confusion matrix
# systemds code/exp/mLogReg.dml \
#     -config conf/$conf.xml \
#     -stats 30 \
#     -args $x $y $xt $yt TRUE \
#     -fedMonitoringAddress "http://localhost:8080"

# Execute locally to compare
# systemds code/exp/mLogReg.dml \
#     -config conf/$conf.xml \
#     -stats 100 \
#     -args $x_loc $y_loc $xt_loc $yt_loc TRUE

# systemds code/exp/CNN.dml \
#     -stats \
#     -args $x $y_hot $xt $yt_hot \
#     -fedMonitoringAddress "http://localhost:8080"

# systemds code/exp/sumRepeat.dml \
#     -config conf/$conf.xml \
#     -stats 30 \
#     -args $x 100 \
#     -fedMonitoringAddress "http://localhost:8080"

# systemds code/exp/adult.dml \
#     -config conf/$conf.xml \
#     -stats 30 \
#     -fedMonitoringAddress "http://localhost:8080"

systemds code/exp/criteo.dml \
    -config conf/$conf.xml \
    -stats 30 \
    -fedMonitoringAddress "http://localhost:8080"


