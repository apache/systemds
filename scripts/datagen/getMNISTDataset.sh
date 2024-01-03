#!/bin/bash
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

BASE=$1
if [ "$BASE" = "" ]; then BASE=$PWD; fi
RET=$PWD
if [ ! -d "$BASE" ]; then mkdir "$BASE"; fi
cd "$BASE" || exit

echo "Downloading"
if [ ! -f "mnist_train.csv.zip" ]; then wget --no-check-certificate https://github.com/phoebetronic/mnist/raw/main/mnist_train.csv.zip; fi
if [ ! -f "mnist_test.csv.zip" ]; then wget --no-check-certificate https://github.com/phoebetronic/mnist/raw/main/mnist_test.csv.zip; fi

echo "Unzipping"
unzip -u mnist_train.csv.zip
unzip -u mnist_test.csv.zip

# have to create metadata for these external csv files

echo '{
          "data_type": "matrix",
          "value_type": "double",
          "rows": 60000,
          "cols": 785,
          "nnz": 0,
          "format": "csv",
          "author": "anon",
          "header": false,
          "sep": ",",
          "created": "2023-06-26 18:35:22 CEST"
      }' > mnist_train.csv.mtd

echo '{
          "data_type": "matrix",
          "value_type": "double",
          "rows": 10000,
          "cols": 785,
          "nnz": 0,
          "format": "csv",
          "author": "nobody",
          "header": false,
          "sep": ",",
          "created": "2023-06-26 18:35:22 CEST"
      }' > mnist_test.csv.mtd

cd "$RET" || exit
echo "Done"
